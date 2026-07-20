"""Final deployed VQ-VAE tokenizer training (Foldseek-style):
 - 30 independently initialised seeds trained on TRAIN partition (n=610)
 - deployed A4 arch (2-enc, 4-dec, 128/128, L2 normalize, EMA 0.99, β=1.0)
 - descriptor_loss active_blocks={g_ctx, g_nn1}, cosine LR + 200 warmup, clip 1
 - val-loss best checkpoint per seed
 - final selection by val-retrieval (fam + clan Sens@1FP, expansion prefilter,
   Structure-BLOSUM diagmean-1 reranker built from train US-align TM>=0.5 pairs)
 - retokenise all 885 db chains with best seed
 - build final Structure-BLOSUM from TRAIN US-align pairs only
 - save files/production_new/{db.pt, sub_matrix.pt, checkpoint.pt}
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

EPOCHS = 50
BATCH = 512
LR = 3e-4
WD = 1e-2
WARMUP = 200
CLIP = 1.0
K = 80
TOPN = 100
KMER_K = 4
THRESH_FRAC = 0.75
N_SEEDS = 30
N_ENC, N_DEC, DH, DL = 2, 4, 128, 128
D_DESC = 49
ACTIVE = {"g_ctx", "g_nn1"}
GO = -2.0
GE = -0.5
QMAX = 500
STRUCT_TM_THRESH = 0.5
USALIGN_CACHE = "files/production/retrieval/usalign_full.cache.jsonl"
BLOSUM_SCALE = 2.0
BLOSUM_GAP_OPEN = -11.0
BLOSUM_GAP_EXTEND = -1.0
OUT_DIR = "files/production_new"
os.makedirs(OUT_DIR, exist_ok=True)
db = torch.load("files/production/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
N = len(ents)
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
fams = attach_labels(ents, load_labels("Splits/labels.tsv"))
clans = attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc"))


def idl(p):
    return [str(json.loads(l)["id"]) for l in open(p) if l.strip()]


train_dis = [id2i[c] for c in idl("Splits/pdb_out/train.json") if c in id2i]
val_dis = [id2i[c] for c in idl("Splits/pdb_out/val.json") if c in id2i]
all_dis = list(range(N))
print(f"train n={len(train_dis)}, val n={len(val_dis)}, db n={N}", flush=True)
DESC = "files/ablation/kmeans_baseline/current_desc/pdb_out"
KEEP = set(list(range(42, 49)) + list(range(0, 10)))
MASK = np.array([1.0 if i in KEEP else 0.0 for i in range(49)], dtype=np.float32)
# raw descriptors for train+val
desc_raw = {}
for sp in ["train", "val"]:
    D = torch.load(f"{DESC}/{sp}_desc.pt", map_location="cpu", weights_only=False).numpy()
    off = 0
    for c in idl(f"Splits/pdb_out/{sp}.json"):
        di = id2i[c]
        L = len(ents[di]["tokens"])
        desc_raw[di] = D[off : off + L].astype(np.float32)
        off += L
# also test descriptors (for final tokenisation only)
Dtest = torch.load(f"{DESC}/test_desc.pt", map_location="cpu", weights_only=False).numpy()
off = 0
for c in idl("Splits/pdb_out/test.json"):
    di = id2i[c]
    L = len(ents[di]["tokens"])
    desc_raw[di] = Dtest[off : off + L].astype(np.float32)
    off += L
# normalisation stats: TRAIN only (deployed convention)
_raw_tr = np.vstack([desc_raw[di] for di in train_dis if len(desc_raw[di]) > 0])
MN = _raw_tr.mean(0)
SD = _raw_tr.std(0) + 1e-9


def normed(di):
    r = desc_raw[di]
    if len(r) == 0:
        return np.zeros((0, 49), np.float32)
    return ((r - MN) / SD).astype(np.float32) * MASK


Xtr = np.vstack([normed(di) for di in train_dis if len(desc_raw[di]) > 0])
Xval = np.vstack([normed(di) for di in val_dis if len(desc_raw[di]) > 0])
print(f"Xtr shape={Xtr.shape}  Xval shape={Xval.shape}", flush=True)
# US-align pairs restricted to train
STRUCT_PAIRS_TRAIN = []
_seen = set()
train_set = set(train_dis)
with open(USALIGN_CACHE) as _f:
    for _l in _f:
        _d = json.loads(_l)
        _tm = max(_d["tm_query"], _d["tm_target"])
        if _tm < STRUCT_TM_THRESH:
            continue
        _a, _b = _d["i"], _d["j"]
        if _a == _b:
            continue
        if _a not in train_set or _b not in train_set:
            continue
        _k = (_a, _b) if _a < _b else (_b, _a)
        if _k in _seen:
            continue
        _seen.add(_k)
        STRUCT_PAIRS_TRAIN.append(_k)
print(
    f"Structure-BLOSUM training pool: {len(STRUCT_PAIRS_TRAIN)} intra-train pairs with TM>={STRUCT_TM_THRESH}",
    flush=True,
)


def cosine_warmup(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def train_seed(seed):
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
    best_epoch = -1
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
            best_epoch = ep
    m.load_state_dict(best_state)
    m.eval()
    return m, best_val, best_epoch


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


def build_struct_blosum(tok_by_di, pair_pool, K_):
    sub_id = np.zeros((K_, K_), dtype=np.float32)
    np.fill_diagonal(sub_id, 1.0)
    joint = np.zeros((K_, K_), dtype=np.float64)
    n_al = 0
    for a, b in pair_pool:
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


def val_retrieval_sens(m):
    """Tokenise train+val, build train-only Structure-BLOSUM, retrieve val queries
    over the full train+val pool. Return fam-Sens + clan-Sens averaged."""
    pool_di = train_dis + val_dis
    toks = {}
    cb = None
    for di in pool_di:
        X = normed(di)
        if len(X) == 0:
            toks[di] = np.zeros(0, np.int64)
            continue
        c, cb_arr = get_codes(m, X)
        toks[di] = c
        cb = cb_arr
    cbn = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-9)
    cos = (cbn @ cbn.T).astype(np.float32)
    struct_sub, _ = build_struct_blosum(toks, STRUCT_PAIRS_TRAIN, K)
    order = pool_di
    pos_map = {d: i for i, d in enumerate(order)}
    tok_list = [toks[d] for d in order]
    fam_pool = np.array([fams[d] for d in order], dtype=object)
    clan_pool = np.array([clans[d] for d in order], dtype=object)
    famcnt = defaultdict(int)
    for d in order:
        if fams[d]:
            famcnt[fams[d]] += 1
    k = KMER_K
    powers = np.array([K ** (k - 1 - i) for i in range(k)], dtype=np.int64)
    idx = defaultdict(list)
    for ti, tt in enumerate(tok_list):
        for p in range(len(tt) - k + 1):
            kid = int(np.dot(tt[p : p + k], powers))
            idx[kid].append((ti, p))
    smith_waterman_numba(np.zeros(4, np.int64), np.zeros(4, np.int64), struct_sub, GO, GE)
    thresh = THRESH_FRAC * k
    val_set = set(val_dis)
    qs = [
        pos_map[d]
        for d in val_dis
        if fams[d] and famcnt[fams[d]] >= 2 and k <= len(tok_list[pos_map[d]]) <= QMAX
    ]
    r1s = []
    famS = []
    clanS = []
    n_pool = len(order)
    for q in qs:
        tt = tok_list[q]
        td = defaultdict(dict)
        for qp in range(len(tt) - k + 1):
            qk = tt[qp : qp + k]
            for kid in expand_kmer(qk, cos, K, thresh):
                for ti, tp in idx.get(kid, ()):
                    if ti != q:
                        d = qp - tp
                        td[ti][d] = td[ti].get(d, 0) + 1
        sc = [(ti, max((c for c in dm.values() if c >= 2), default=-1)) for ti, dm in td.items()]
        sc = [x for x in sc if x[1] > 0]
        sc.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break: score, then target index
        cand = np.array([x[0] for x in sc[:TOPN]], np.int64)
        if len(cand) == 0:
            r1s.append(0)
            famS.append(0.0)
            clanS.append(0.0)
            continue
        r = cand[
            np.argsort(
                -np.array(
                    [smith_waterman_numba(tt, tok_list[t], struct_sub, GO, GE) for t in cand]
                ),
                kind="stable",
            )
        ]
        va = np.arange(n_pool) != q
        # Family Sens@1FP (Foldseek convention): TP=same family,
        # neutral=same clan different family, FP=different/absent clan.
        rel_f = (fam_pool == fam_pool[q]) & va
        neu_f = (
            ((clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va) if clan_pool[q] else None
        )
        famS.append(first_fp_sensitivity(r, rel_f, neu_f))
        # Clan Sens@1FP (Foldseek convention): TP=same clan *different* family,
        # neutral=same family, FP=different clan. Only over queries that have
        # >=1 same-clan-different-family target.
        if clan_pool[q]:
            rel_c = (clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va
            if rel_c.any():
                neu_c = (fam_pool == fam_pool[q]) & va
                clanS.append(first_fp_sensitivity(r, rel_c, neu_c))
        r1s.append(int(fam_pool[r[0]] == fam_pool[q]))
    fam_m = float(np.mean(famS)) if famS else 0.0
    clan_m = float(np.mean(clanS)) if clanS else 0.0
    r1_m = float(np.mean(r1s)) if r1s else 0.0
    return fam_m + clan_m, fam_m, clan_m, r1_m, len(qs)


t_all = time.time()
results = []
best = {"score": -1, "seed": None, "model": None, "epoch": None}
for s in range(N_SEEDS):
    seed = 100 + s
    t0 = time.time()
    m, vl, ep = train_seed(seed)
    t1 = time.time()
    score, fs, cs, r1, nq = val_retrieval_sens(m)
    t2 = time.time()
    results.append(
        {
            "seed": seed,
            "val_loss": vl,
            "best_epoch": ep,
            "val_fam_sens": fs,
            "val_clan_sens": cs,
            "val_r1": r1,
            "selection_score": score,
            "nq": nq,
            "train_s": t1 - t0,
            "eval_s": t2 - t1,
        }
    )
    print(
        f"  seed {seed}: valLoss={vl:.4f}  bestEp={ep}  famSens={fs:.3f}  clanSens={cs:.3f}  R@1={r1:.3f}  score={score:.3f}  ({t1-t0:.0f}s train, {t2-t1:.0f}s eval)",
        flush=True,
    )
    if score > best["score"]:
        best = {
            "score": score,
            "seed": seed,
            "model": m,
            "epoch": ep,
            "val_loss": vl,
            "val_fam_sens": fs,
            "val_clan_sens": cs,
            "val_r1": r1,
        }
    json.dump(results, open(f"{OUT_DIR}/seed_selection.json", "w"), indent=2)
print(
    f"\n== Best seed: {best['seed']}  score={best['score']:.3f}  famSens={best['val_fam_sens']:.3f}  clanSens={best['val_clan_sens']:.3f}  R@1={best['val_r1']:.3f} ==",
    flush=True,
)
# Finalise: retokenise all 885 db chains with best seed
m = best["model"]
new_entries = []
cb = None
for di in all_dis:
    e = copy.deepcopy(ents[di])
    X = normed(di)
    if len(X) == 0:
        e["tokens"] = np.zeros(0, np.int64)
    else:
        c, cb_arr = get_codes(m, X)
        e["tokens"] = c
        cb = cb_arr
    new_entries.append(e)
db_new = {
    "n_codes": K,
    "entries": new_entries,
    "checkpoint": f"{OUT_DIR}/desc_vqvae_best.pt",
    "data": db.get("data"),
}
torch.save(db_new, f"{OUT_DIR}/db.pt")
# save Structure-BLOSUM (from train pairs only, using best seed's tokens)
tok_by_di = {di: new_entries[di]["tokens"] for di in all_dis}
struct_sub, n_al = build_struct_blosum(tok_by_di, STRUCT_PAIRS_TRAIN, K)
torch.save(
    {
        "matrix": torch.from_numpy(struct_sub),
        "n_codes": K,
        "method": "structure-blosum-diagmean1",
        "scale": BLOSUM_SCALE,
        # Gaps used to build the matrix (traceback that collects aligned token
        # pairs from TM>=0.5 chain pairs):
        "construction_gap_open": BLOSUM_GAP_OPEN,
        "construction_gap_extend": BLOSUM_GAP_EXTEND,
        # Gaps to use with this matrix during token Smith-Waterman reranking:
        "rerank_gap_open": GO,
        "rerank_gap_extend": GE,
        "n_alignments": n_al,
        "tm_threshold": STRUCT_TM_THRESH,
        "pool": "train-only",
    },
    f"{OUT_DIR}/sub_matrix.pt",
)
# save checkpoint metadata
torch.save(
    {
        "model": m.state_dict(),
        "n_codes": K,
        "d_descriptor": D_DESC,
        "d_hidden": DH,
        "d_latent": DL,
        "n_enc_layers": N_ENC,
        "n_dec_layers": N_DEC,
        "beta_commit": 1.0,
        "ema_decay": 0.99,
        "lambda_geom": 1.0,
        "lambda_pair": 1.0,
        "lambda_stack": 1.0,
        "lambda_mask": 1.0,
        "pair_stack_reduce": "sum",
        "lambda_latent_norm": 0.0,
        "active_blocks": sorted(ACTIVE),
        "seed": best["seed"],
        "epoch": best["epoch"],
        "loss": best["val_loss"],
        "val_fam_sens": best["val_fam_sens"],
        "val_clan_sens": best["val_clan_sens"],
        "l2_normalize_encoder": True,
        "l2_normalize_codebook": True,
        "split_name": "pdb_out",
        "norm_mean": torch.from_numpy(MN),
        "norm_std": torch.from_numpy(SD),
        "selection_score": best["score"],
        "selection_metric": "val_fam_sens+val_clan_sens",
    },
    f"{OUT_DIR}/desc_vqvae_best.pt",
)
print(f"\nSaved: {OUT_DIR}/db.pt, sub_matrix.pt, desc_vqvae_best.pt", flush=True)
print(f"Structure-BLOSUM built from {n_al} intra-train alignments", flush=True)
print(f"total: {time.time()-t_all:.0f}s\nDONE", flush=True)
