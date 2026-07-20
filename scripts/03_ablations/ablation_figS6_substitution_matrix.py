"""S6 substitution-matrix ablation, K=80, deployed VQ-VAE, PDB-disjoint 5-fold CV.
Reranking matrices compared (the prefilter always expands by codebook cosine):
  - Identity: +2/-1
  - Codebook cosine (unsupervised baseline)
  - Codebook BLOSUM (unsupervised): softmax→log-odds from codebook similarity
  - Rfam-BLOSUM diagmean1 (family-supervised)
  - Structure-BLOSUM diagmean1 (deployed; Foldseek-style, from structurally-
    similar pairs with symmetric US-align TM >= 0.5 in the fold's training
    subset, no family labels)
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
from pathlib import Path
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
KMER_K = 4
THRESH_FRAC = 0.75
N_SEEDS = 1
K = 80
N_ENC, N_DEC, DH, DL = 2, 4, 128, 128
D_DESC = 49
ACTIVE = {"g_ctx", "g_nn1"}
MAX_PAIRS_PER_FAM = 100
BLOSUM_SCALE = 2.0
BLOSUM_GAP_OPEN = -11.0
BLOSUM_GAP_EXTEND = -1.0
STRUCT_TM_THRESH = 0.5
USALIGN_CACHE = "files/production/retrieval/usalign_full.cache.jsonl"
SW_GAP_OPEN = -2.0
SW_GAP_EXTEND = -0.5
SOFTMAX_T = 0.5
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
    f"CV pool: {len(pool)} chains  substitution-matrix ablation at K={K}, deployed VQ-VAE pipeline",
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
    return m, best_val


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


def make_identity(K_):
    m = np.full((K_, K_), -1.0, dtype=np.float32)
    np.fill_diagonal(m, 2.0)
    return m


def make_cosine(cb):
    cbn = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-9)
    return (cbn @ cbn.T).astype(np.float32)


def make_codebook_blosum(cb, T=SOFTMAX_T):
    cos = make_cosine(cb)
    logits = cos / T
    logits = logits - logits.max()
    q = np.exp(logits)
    q = q / q.sum()
    p = q.sum(1)
    p = np.maximum(p, 1e-12)
    expected = np.outer(p, p)
    S = 2.0 * np.log2(np.maximum(q, 1e-12) / np.maximum(expected, 1e-12))
    # normalize diagonal mean to 1.0
    diag = np.diag(S)
    s = 1.0 / max(diag.mean(), 1e-6)
    S = S * s
    return S.astype(np.float32)


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


def make_rfam_blosum(tok_list, fam_list, K_, diagmean1=True):
    fam_ids = defaultdict(list)
    for ti, f in enumerate(fam_list):
        if f:
            fam_ids[f].append(ti)
    fam_ids = {f: v for f, v in fam_ids.items() if len(v) >= 2}
    sub_id = make_identity(K_)
    sub_id = (sub_id == 2.0).astype(np.float32)  # match=1 mismatch=0
    joint = np.zeros((K_, K_), dtype=np.float64)
    rng_b = np.random.default_rng(0)
    for fam, ids in fam_ids.items():
        pairs = [(a, b) for a in ids for b in ids if a < b]
        if len(pairs) > MAX_PAIRS_PER_FAM:
            pairs = [
                pairs[k] for k in rng_b.choice(len(pairs), size=MAX_PAIRS_PER_FAM, replace=False)
            ]
        for a, b in pairs:
            q = tok_list[a]
            t = tok_list[b]
            if len(q) == 0 or len(t) == 0:
                continue
            pi, pj = sw_traceback(q, t, sub_id, BLOSUM_GAP_OPEN, BLOSUM_GAP_EXTEND)
            for x, y in zip(pi, pj):
                joint[x, y] += 1
                joint[y, x] += 1
    total = max(joint.sum(), 1.0)
    p_ij = joint / total
    p_i = np.maximum(p_ij.sum(1), 1e-12)
    S = np.log2(np.maximum(p_ij, 1e-12) / np.maximum(np.outer(p_i, p_i), 1e-12)) * BLOSUM_SCALE
    S = S.astype(np.float32)
    if diagmean1:
        diag = np.diag(S)
        s = 1.0 / max(diag.mean(), 1e-6)
        S = S * s
    return S


# Load US-align pair cache (i,j → tm_short = max symmetric TM)
STRUCT_PAIRS = []  # list of (i,j,tm_short) with tm_short>=thresh (undirected canonicalised)
_seen_pair = set()
with open(USALIGN_CACHE) as _f:
    for _l in _f:
        _d = json.loads(_l)
        _tm = 0.5 * (_d["tm_query"] + _d["tm_target"])
        if _tm < STRUCT_TM_THRESH:
            continue
        _a, _b = _d["i"], _d["j"]
        if _a == _b:
            continue
        _k = (_a, _b) if _a < _b else (_b, _a)
        if _k in _seen_pair:
            continue
        _seen_pair.add(_k)
        STRUCT_PAIRS.append((_k[0], _k[1], _tm))
print(f"Structure-BLOSUM: loaded {len(STRUCT_PAIRS)} pairs with TM>={STRUCT_TM_THRESH}", flush=True)


def make_struct_blosum(tok_list_by_di, tr_dis_set, K_, diagmean1=True):
    """BLOSUM from Foldseek-style structural pairs (US-align TM>=0.5) restricted
    to fold's training subset. tok_list_by_di is a dict di->tokens."""
    joint = np.zeros((K_, K_), dtype=np.float64)
    n = 0
    sub_id = np.zeros((K_, K_), dtype=np.float32)
    np.fill_diagonal(sub_id, 1.0)
    for a, b, _tm in STRUCT_PAIRS:
        if a not in tr_dis_set or b not in tr_dis_set:
            continue
        q = tok_list_by_di.get(a)
        t = tok_list_by_di.get(b)
        if q is None or t is None or len(q) == 0 or len(t) == 0:
            continue
        pi, pj = sw_traceback(q, t, sub_id, BLOSUM_GAP_OPEN, BLOSUM_GAP_EXTEND)
        for x, y in zip(pi, pj):
            joint[x, y] += 1
            joint[y, x] += 1
        n += 1
    total = max(joint.sum(), 1.0)
    p_ij = joint / total
    p_i = np.maximum(p_ij.sum(1), 1e-12)
    S = np.log2(np.maximum(p_ij, 1e-12) / np.maximum(np.outer(p_i, p_i), 1e-12)) * BLOSUM_SCALE
    S = S.astype(np.float32)
    if diagmean1:
        diag = np.diag(S)
        s = 1.0 / max(diag.mean(), 1e-6)
        S = S * s
    return S, n


def expand_kmer(qk, cos, K_, thresh):
    k = len(qk)
    powers = np.array([K_ ** (k - 1 - i) for i in range(k)], dtype=np.int64)
    out = {int(np.dot(qk, powers))}
    for pos in range(k):
        for tok in range(K_):
            if tok == qk[pos]:
                continue
            score = (k - 1) * 1.0 + cos[qk[pos], tok]
            if score >= thresh:
                trial = qk.copy()
                trial[pos] = tok
                out.add(int(np.dot(trial, powers)))
    return list(out)


def retrieval_eval(
    sub, K_, tok_list, pos_map, fam_pool, clan_pool, famcnt, heldout_dis, cos_for_prefilter
):
    order_len = len(tok_list)
    k = KMER_K
    powers = np.array([K_ ** (k - 1 - i) for i in range(k)], dtype=np.int64)
    idx = defaultdict(list)
    for ti, tt in enumerate(tok_list):
        for p in range(len(tt) - k + 1):
            kid = int(np.dot(tt[p : p + k], powers))
            idx[kid].append((ti, p))
    smith_waterman_numba(
        np.zeros(4, np.int64), np.zeros(4, np.int64), sub, SW_GAP_OPEN, SW_GAP_EXTEND
    )
    thresh = THRESH_FRAC * k
    qs = [
        pos_map[d]
        for d in heldout_dis
        if fam_pool[pos_map[d]]
        and famcnt[fam_pool[pos_map[d]]] >= 2
        and k <= len(tok_list[pos_map[d]]) <= QMAX
    ]
    famS = []
    clanS = []
    for q in qs:
        tt = tok_list[q]
        td = defaultdict(dict)
        for qp in range(len(tt) - k + 1):
            qk = tt[qp : qp + k]
            for kid in expand_kmer(qk, cos_for_prefilter, K_, thresh):
                for ti, tp in idx.get(kid, ()):
                    if ti != q:
                        d = qp - tp
                        td[ti][d] = td[ti].get(d, 0) + 1
        sc = [(ti, max((c for c in dm.values() if c >= 2), default=-1)) for ti, dm in td.items()]
        sc = [x for x in sc if x[1] > 0]
        sc.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break: score, then target index
        cand = np.array([x[0] for x in sc[:100]], np.int64)
        if len(cand) == 0:
            famS.append(0.0)
            continue
        r = cand[
            np.argsort(
                -np.array(
                    [
                        smith_waterman_numba(tt, tok_list[t], sub, SW_GAP_OPEN, SW_GAP_EXTEND)
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
    m, _ = train_one(42, Xtr, Xval)
    # tokenize
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
    order = pool
    pos_map = {d: i for i, d in enumerate(order)}
    tok_list = [toks[d] for d in order]
    fam_pool = np.array([fams[d] for d in order], dtype=object)
    clan_pool = np.array([clans[d] for d in order], dtype=object)
    famcnt = defaultdict(int)
    for d in order:
        if fams[d]:
            famcnt[fams[d]] += 1
    # build 5 matrices
    cos = make_cosine(cb)
    fam_list_tr = [fams[d] if d in tr_dis_set else None for d in order]
    tok_by_di = {d: t for d, t in zip(order, tok_list)}
    struct_S, struct_n = make_struct_blosum(tok_by_di, tr_dis_set, K, diagmean1=True)
    mats = {
        "identity": make_identity(K),
        "cosine": cos,
        "codebook_blosum": make_codebook_blosum(cb),
        "rfam_blosum_diagmean1": make_rfam_blosum(tok_list, fam_list_tr, K, diagmean1=True),
        "structure_blosum_diagmean1": struct_S,
    }
    print(f"    (struct BLOSUM used {struct_n} pairs)", flush=True)
    out = {}
    for name, sub in mats.items():
        fs, cs, nf, nc = retrieval_eval(
            sub, K, tok_list, pos_map, fam_pool, clan_pool, famcnt, heldout, cos
        )
        out[name] = (fs, cs, nf, nc)
    return out


MATNAMES = [
    "identity",
    "cosine",
    "codebook_blosum",
    "rfam_blosum_diagmean1",
    "structure_blosum_diagmean1",
]
all_out = {n: {"fam_folds": [], "clan_folds": [], "nqf": [], "nqc": []} for n in MATNAMES}
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
            f"    fold={f} {name:26s}: famSens={fs:.3f} clanSens={cs:.3f} nqf={nf} nqc={nc}",
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
    json.dump(all_out, open("files/ablation/cv_s6_clan_5fold.json", "w"), indent=2)
print("\n== Summary ==", flush=True)
for name, v in all_out.items():
    print(
        f"  {name:26s}: famSens={v['fam_sens_mean']:.3f}±{v['fam_sens_std']:.3f}  clanSens={v['clan_sens_mean']:.3f}±{v['clan_sens_std']:.3f}"
    )
print(f"\ntotal: {time.time()-t_all:.0f}s\nDONE", flush=True)
