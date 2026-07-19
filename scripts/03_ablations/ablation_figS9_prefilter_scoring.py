"""Prefilter SCORING ablation, K=80 deployed VQ-VAE, PDB-disjoint 5-fold CV.
All variants use the contiguous 4-mer seed and Structure-BLOSUM SW reranking;
they differ only in how 1-substitution similar k-mers are selected:
  E4          : no expansion (exact match only)
  cos>=0      : codebook cosine, thresh 3.0  (currently deployed)
  blosum>=0   : Structure-BLOSUM, thresh 3.0
  blosum>=.25 : Structure-BLOSUM, thresh 3.25
ORIGINAL HEADER:
Variants (all use k=4 informative positions, cosine-substitution reranking,
top-100 candidates, same SW gap penalties):
  - E4  : exact contiguous 4-mer, pattern 1111 (baseline)
  - G4  : gapped 4-mer, pattern 11_11 (2+gap+2 over 5 positions)
  - S4  : spaced 4-mer, pattern 1_1_1_1 (over 7 positions)
  - SimE4: exact 4-mer + 1-substitution similar-k-mer expansion (deployed)
"""
import sys, os, json, time, math, copy
from collections import defaultdict
import numpy as np, torch, torch.nn as nn
from numba import njit

sys.path.insert(0, "models")
sys.path.insert(0, "evaluation/search")
from vqvae import DescriptorVQVAE, descriptor_loss
from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels
from smith_waterman import smith_waterman_numba

QMAX = 500
EPOCHS = 50
BATCH = 512
LR = 3e-4
WD = 1e-2
WARMUP = 200
CLIP = 1.0
K = 80
TOPN = 100
N_ENC, N_DEC, DH, DL = 2, 4, 128, 128
D_DESC = 49
ACTIVE = {"g_ctx", "g_nn1"}
SW_GAP_OPEN = -2.0
SW_GAP_EXTEND = -0.5
PATTERNS = {
    "E4": [1, 1, 1, 1],  # contiguous
    "G4": [1, 1, 0, 1, 1],  # 11_11
    "S4": [1, 0, 1, 0, 1, 0, 1],  # 1_1_1_1
    "SimE4": [1, 1, 1, 1],  # contiguous + 1-sub expansion
}
SIMILAR_EXPAND = {"SimE4": True}
EXPAND_THRESH_FRAC = 0.75
STRUCT_TM_THRESH = 0.5
USALIGN_CACHE = "files/production/retrieval/usalign_full.cache.jsonl"
BLOSUM_SCALE = 2.0
BLOSUM_GAP_OPEN = -11.0
BLOSUM_GAP_EXTEND = -1.0
db = torch.load("files/production/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
N = len(ents)
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
fams = attach_labels(ents, load_labels("Splits/labels.tsv"))
clans = attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc"))


def idl(p):
    return [str(json.loads(l)["id"]) for l in open(p) if l.strip()]


pool = [
    id2i[c] for c in idl("Splits/pdb_out/train.json") + idl("Splits/pdb_out/val.json") if c in id2i
]
DESC = "files/ablation/kmeans_baseline/current_desc/pdb_out"
KEEP = set(list(range(42, 49)) + list(range(0, 10)))
MASK = np.array([1.0 if i in KEEP else 0.0 for i in range(49)], dtype=np.float32)
desc_raw = {}
for sp in ["train", "val"]:
    D = torch.load(f"{DESC}/{sp}_desc.pt", map_location="cpu", weights_only=False).numpy()
    off = 0
    for c in idl(f"Splits/pdb_out/{sp}.json"):
        di = id2i[c]
        L = len(ents[di]["tokens"])
        desc_raw[di] = D[off : off + L].astype(np.float32)
        off += L


def build_desc_all(mn, sd):
    out = {}
    for di in pool:
        r = desc_raw[di]
        if len(r) == 0:
            out[di] = np.zeros((0, 49), np.float32)
            continue
        out[di] = ((r - mn) / sd).astype(np.float32) * MASK
    return out


print(
    f"CV pool: {len(pool)} chains  prefilter ablation at K={K}, deployed VQ-VAE, patterns={list(PATTERNS.keys())}",
    flush=True,
)
import re as _re


def _pdb_of(di):
    m = _re.search(r"PDB_0000([A-Z0-9]{4})_", str(ents[di]["id"]))
    return m.group(1) if m else str(ents[di]["id"])


_pdb_groups = defaultdict(list)
for _di in pool:
    _pdb_groups[_pdb_of(_di)].append(_di)
_pdbs = sorted(_pdb_groups)
np.random.default_rng(0).shuffle(_pdbs)
folds = [[] for _ in range(5)]
for _pi, _p in enumerate(_pdbs):
    folds[_pi % 5].extend(_pdb_groups[_p])
print(f"PDB-disjoint folds: sizes={[len(f) for f in folds]}  n_pdbs={len(_pdbs)}", flush=True)


def cosine_warmup(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def train_one(seed, Xtr, Xval):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = DescriptorVQVAE(
        d_descriptor=D_DESC,
        d_hidden=DH,
        d_latent=DL,
        n_codes=K,
        n_enc_layers=N_ENC,
        n_dec_layers=N_DEC,
        ema_decay=0.99,
        l2_normalize_encoder=True,
        l2_normalize_codebook=True,
    )
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    Xt = torch.from_numpy(Xtr).float()
    Xv = torch.from_numpy(Xval).float()
    steps_per = max(1, (len(Xt) + BATCH - 1) // BATCH)
    total = steps_per * EPOCHS
    step = 0
    best_val = float("inf")
    best_state = None
    for ep in range(EPOCHS):
        m.train()
        pi = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            xb = Xt[pi[i : i + BATCH]]
            out = m(xb)
            ld = descriptor_loss(
                out,
                xb,
                beta_commit=1.0,
                lambda_geom=1.0,
                lambda_pair=1.0,
                lambda_stack=1.0,
                lambda_mask=1.0,
                active_blocks=ACTIVE,
            )
            loss = ld["loss"]
            for g in opt.param_groups:
                g["lr"] = LR * cosine_warmup(step, WARMUP, total)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            opt.step()
            step += 1
        m.eval()
        with torch.no_grad():
            vout = m(Xv)
            vld = descriptor_loss(
                vout,
                Xv,
                beta_commit=1.0,
                lambda_geom=1.0,
                lambda_pair=1.0,
                lambda_stack=1.0,
                lambda_mask=1.0,
                active_blocks=ACTIVE,
            )
            vl = float(vld["loss"])
        if vl < best_val:
            best_val = vl
            best_state = copy.deepcopy(m.state_dict())
    m.load_state_dict(best_state)
    m.eval()
    return m


def get_codes(m, X):
    with torch.no_grad():
        z = m.encoder(torch.from_numpy(X).float())
        if m.encoder.l2_normalize:
            z = torch.nn.functional.normalize(z, dim=-1)
        cb = m.codebook.embeddings
        if m.codebook.l2_normalize:
            cb = torch.nn.functional.normalize(cb, dim=-1)
        d = (z**2).sum(1, keepdim=True) - 2 * z @ cb.t() + (cb**2).sum(1)
        return d.argmin(1).numpy().astype(np.int64), cb.numpy()


# Load US-align pair cache (>= TM threshold)
STRUCT_PAIRS = []
_seen = set()
with open(USALIGN_CACHE) as _f:
    for _l in _f:
        _d = json.loads(_l)
        _tm = max(_d["tm_query"], _d["tm_target"])
        if _tm < STRUCT_TM_THRESH:
            continue
        _a, _b = _d["i"], _d["j"]
        if _a == _b:
            continue
        _k = (_a, _b) if _a < _b else (_b, _a)
        if _k in _seen:
            continue
        _seen.add(_k)
        STRUCT_PAIRS.append(_k)
print(f"Structure-BLOSUM pool: {len(STRUCT_PAIRS)} pairs with TM>={STRUCT_TM_THRESH}", flush=True)


@njit(cache=True)
def sw_traceback(q, t, sub, go, ge):
    m = len(q)
    n = len(t)
    if m == 0 or n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    H = np.zeros((m + 1, n + 1), dtype=np.float32)
    E = np.full((m + 1, n + 1), -1e9, dtype=np.float32)
    F = np.full((m + 1, n + 1), -1e9, dtype=np.float32)
    bt = np.zeros((m + 1, n + 1), dtype=np.int8)
    bi = 0
    bj = 0
    bv = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            e = H[i, j - 1] + go
            e2 = E[i, j - 1] + ge
            E[i, j] = e if e > e2 else e2
            f = H[i - 1, j] + go
            f2 = F[i - 1, j] + ge
            F[i, j] = f if f > f2 else f2
            diag = H[i - 1, j - 1] + sub[q[i - 1], t[j - 1]]
            v = 0.0
            tag = np.int8(0)
            if diag > v:
                v = diag
                tag = 1
            if F[i, j] > v:
                v = F[i, j]
                tag = 2
            if E[i, j] > v:
                v = E[i, j]
                tag = 3
            H[i, j] = v
            bt[i, j] = tag
            if v > bv:
                bv = v
                bi = i
                bj = j
    pi = np.zeros(m + n, np.int64)
    pj = np.zeros(m + n, np.int64)
    np_ = 0
    i = bi
    j = bj
    while i > 0 and j > 0 and bt[i, j] != 0:
        b = bt[i, j]
        if b == 1:
            pi[np_] = q[i - 1]
            pj[np_] = t[j - 1]
            np_ += 1
            i -= 1
            j -= 1
        elif b == 2:
            i -= 1
        else:
            j -= 1
    return pi[:np_][::-1].copy(), pj[:np_][::-1].copy()


def make_struct_blosum(tok_by_di, tr_dis_set, K_):
    sub_id = np.zeros((K_, K_), dtype=np.float32)
    np.fill_diagonal(sub_id, 1.0)
    joint = np.zeros((K_, K_), dtype=np.float64)
    n_al = 0
    for a, b in STRUCT_PAIRS:
        if a not in tr_dis_set or b not in tr_dis_set:
            continue
        q = tok_by_di.get(a)
        t = tok_by_di.get(b)
        if q is None or t is None or len(q) == 0 or len(t) == 0:
            continue
        pi, pj = sw_traceback(q, t, sub_id, BLOSUM_GAP_OPEN, BLOSUM_GAP_EXTEND)
        for x, y in zip(pi, pj):
            joint[x, y] += 1
            joint[y, x] += 1
        n_al += 1
    total = max(joint.sum(), 1.0)
    p_ij = joint / total
    p_i = np.maximum(p_ij.sum(1), 1e-12)
    S = np.log2(np.maximum(p_ij, 1e-12) / np.maximum(np.outer(p_i, p_i), 1e-12)) * BLOSUM_SCALE
    S = S.astype(np.float32)
    diag = np.diag(S)
    s = 1.0 / max(diag.mean(), 1e-6)
    S = S * s
    return S, n_al


def extract_seed(tokens, pattern, start):
    """Return tuple of tokens at positions where pattern==1, offset from start."""
    return tuple(int(tokens[start + i]) for i, p in enumerate(pattern) if p == 1)


def encode_seed(seed, K_):
    x = 0
    for t in seed:
        x = x * K_ + int(t)
    return x


def expand_seed(seed, cos, K_, thresh):
    """1-substitution neighbours of a seed tuple (length k=4) with kmer-kmer cosine>=thresh."""
    k = len(seed)
    powers = [K_ ** (k - 1 - i) for i in range(k)]
    out = {sum(seed[i] * powers[i] for i in range(k))}
    for pos in range(k):
        for tok in range(K_):
            if tok == seed[pos]:
                continue
            score = (k - 1) * 1.0 + cos[seed[pos], tok]
            if score >= thresh:
                trial = list(seed)
                trial[pos] = tok
                out.add(sum(trial[i] * powers[i] for i in range(k)))
    return list(out)


def retrieval_eval(
    name,
    pattern,
    tok_list,
    pos_map,
    fam_pool,
    clan_pool,
    famcnt,
    heldout_dis,
    exp_mat,
    rerank_sub,
    exp_thresh,
    do_exp,
):
    order_len = len(tok_list)
    w = len(pattern)
    k = sum(pattern)
    idx = defaultdict(list)
    for ti, tt in enumerate(tok_list):
        if len(tt) < w:
            continue
        for p in range(len(tt) - w + 1):
            seed = extract_seed(tt, pattern, p)
            kid = encode_seed(seed, K)
            idx[kid].append((ti, p))
    smith_waterman_numba(
        np.zeros(4, np.int64), np.zeros(4, np.int64), rerank_sub, SW_GAP_OPEN, SW_GAP_EXTEND
    )
    do_expand = do_exp
    thresh = exp_thresh
    qs = [
        pos_map[d]
        for d in heldout_dis
        if fam_pool[pos_map[d]]
        and famcnt[fam_pool[pos_map[d]]] >= 2
        and w <= len(tok_list[pos_map[d]]) <= QMAX
    ]
    famS = []
    clanS = []
    for q in qs:
        tt = tok_list[q]
        td = defaultdict(dict)
        for qp in range(len(tt) - w + 1):
            seed = extract_seed(tt, pattern, qp)
            if do_expand:
                kids = expand_seed(seed, exp_mat, K, thresh)
            else:
                kids = [encode_seed(seed, K)]
            for kid in kids:
                for ti, tp in idx.get(kid, ()):
                    if ti != q:
                        d = qp - tp
                        td[ti][d] = td[ti].get(d, 0) + 1
        sc = [(ti, max((c for c in dm.values() if c >= 2), default=-1)) for ti, dm in td.items()]
        sc = [x for x in sc if x[1] > 0]
        sc.sort(key=lambda x: -x[1])
        cand = np.array([x[0] for x in sc[:TOPN]], np.int64)
        if len(cand) == 0:
            famS.append(0.0)
            continue
        r = cand[
            np.argsort(
                -np.array(
                    [
                        smith_waterman_numba(
                            tt, tok_list[t], rerank_sub, SW_GAP_OPEN, SW_GAP_EXTEND
                        )
                        for t in cand
                    ]
                ),
                kind="stable",
            )
        ]
        va = np.arange(order_len) != q
        rel_f = (fam_pool == fam_pool[q]) & va
        neu_f = (
            ((clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va) if clan_pool[q] else None
        )
        famS.append(first_fp_sensitivity(r, rel_f, neu_f))
        if clan_pool[q]:
            rel_c = (clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va
            if rel_c.any():
                neu_c = (fam_pool == fam_pool[q]) & va
                clanS.append(first_fp_sensitivity(r, rel_c, neu_c))
    fam_m = float(np.mean(famS)) if famS else 0.0
    clan_m = float(np.mean(clanS)) if clanS else 0.0
    return fam_m, clan_m, len(famS), len(clanS)


VARIANTS = {  # name -> (expansion matrix key, threshold, expand?)
    "E4": (None, 0.0, False),
    "cos>=0": ("cos", 3.0, True),
    "blosum>=0": ("blosum", 3.0, True),
    "blosum>=.25": ("blosum", 3.25, True),
}
CONTIG = (1, 1, 1, 1)


def eval_fold(fold_idx):
    heldout = set(folds[fold_idx])
    tr_dis = [di for di in pool if di not in heldout]
    rng_f = np.random.default_rng(1000 + fold_idx)
    rng_f.shuffle(tr_dis)
    cut = int(len(tr_dis) * 0.8)
    val_dis = set(tr_dis[cut:])
    tr_dis_set = set(tr_dis[:cut])
    _raw_tr = np.vstack([desc_raw[di] for di in tr_dis_set if len(desc_raw[di]) > 0])
    mn = _raw_tr.mean(0)
    sd = _raw_tr.std(0) + 1e-9
    desc_all = build_desc_all(mn, sd)
    Xtr = np.vstack([desc_all[di] for di in tr_dis_set if len(desc_all[di]) > 0])
    Xval = np.vstack([desc_all[di] for di in val_dis if len(desc_all[di]) > 0])
    m = train_one(42, Xtr, Xval)
    toks = {}
    cb = None
    for di in pool:
        X = desc_all[di]
        if len(X) == 0:
            toks[di] = np.zeros(0, np.int64)
            continue
        c, cb_arr = get_codes(m, X)
        toks[di] = c
        cb = cb_arr
    cbn = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-9)
    cos = (cbn @ cbn.T).astype(np.float32)
    order = pool
    pos_map = {d: i for i, d in enumerate(order)}
    tok_list = [toks[d] for d in order]
    fam_pool = np.array([fams[d] for d in order], dtype=object)
    clan_pool = np.array([clans[d] for d in order], dtype=object)
    famcnt = defaultdict(int)
    for d in order:
        if fams[d]:
            famcnt[fams[d]] += 1
    tok_by_di = {d: t for d, t in zip(order, tok_list)}
    struct_sub, n_al = make_struct_blosum(tok_by_di, tr_dis_set, K)
    print(f"    (struct BLOSUM built from {n_al} intra-training pairs)", flush=True)
    mats = {"cos": cos, "blosum": struct_sub}
    res = {}
    for name, (mk, th, dx) in VARIANTS.items():
        fs, cs, nf, nc = retrieval_eval(
            name,
            CONTIG,
            tok_list,
            pos_map,
            fam_pool,
            clan_pool,
            famcnt,
            heldout,
            mats.get(mk),
            struct_sub,
            th,
            dx,
        )
        res[name] = (fs, cs, nf, nc)
    return res


all_out = {n: {"fam_folds": [], "clan_folds": [], "nqf": [], "nqc": []} for n in VARIANTS}
t_all = time.time()
for f in range(5):
    t0 = time.time()
    res = eval_fold(f)
    for name, (fs, cs, nf, nc) in res.items():
        all_out[name]["fam_folds"].append(fs)
        all_out[name]["clan_folds"].append(cs)
        all_out[name]["nqf"].append(nf)
        all_out[name]["nqc"].append(nc)
        print(
            f"    fold={f} {name:12s}: famSens={fs:.3f} clanSens={cs:.3f} nqf={nf} nqc={nc}",
            flush=True,
        )
    print(f"  fold={f} done ({time.time()-t0:.0f}s)", flush=True)
    for name in all_out:
        ff = all_out[name]["fam_folds"]
        cf = all_out[name]["clan_folds"]
        if ff:
            all_out[name]["fam_sens_mean"] = float(np.mean(ff))
            all_out[name]["fam_sens_std"] = float(np.std(ff))
            all_out[name]["clan_sens_mean"] = float(np.mean(cf))
            all_out[name]["clan_sens_std"] = float(np.std(cf))
    json.dump(all_out, open("files/ablation/cv_prefilter_score_5fold.json", "w"), indent=2)
print("\n== Summary ==", flush=True)
for name, v in all_out.items():
    print(
        f"  {name:8s} [{v['pattern']}]: famSens={v['fam_sens_mean']:.3f}±{v['fam_sens_std']:.3f}  clanSens={v['clan_sens_mean']:.3f}±{v['clan_sens_std']:.3f}"
    )
print(f"\ntotal: {time.time()-t_all:.0f}s\nDONE", flush=True)
