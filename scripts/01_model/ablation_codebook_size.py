"""K-sweep using the deployed VQ-VAE training and configurable unlabelled truth."""
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

sys.path.insert(0, "models")
sys.path.insert(0, "evaluation/search")
from vqvae import DescriptorVQVAE, descriptor_loss
from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels
from smith_waterman import smith_waterman_numba
from token_alignment import smith_waterman_traceback

QMAX = 500
EPOCHS = 50
BATCH = 512
LR = 3e-4
WD = 1e-2
WARMUP = 200
CLIP = 1.0
KMER_K = int(os.environ.get("RFOLDSEARCH_KMER_K", "4"))
THRESH_FRAC = 0.75
N_SEEDS = 1
D_DESC = 49
D_HIDDEN = 128
D_LATENT = 128
N_ENC = 2
N_DEC = 4
BLOSUM_SCALE = 2.0
BLOSUM_GAP_OPEN = -11.0
BLOSUM_GAP_EXTEND = -1.0
ACTIVE = {"g_ctx", "g_nn1"}
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
# 17D restriction: zero out inactive dims (blocks not in {g_ctx,g_nn1}) so descriptor_loss stays consistent
KEEP = set(list(range(42, 49)) + list(range(0, 10)))  # g_ctx (42-48) + g_nn1 (0-9)
MASK = np.array([1.0 if i in KEEP else 0.0 for i in range(49)], dtype=np.float32)
# Store RAW descriptors; z-score is computed PER-FOLD from training-subset only
desc_raw = {}
for sp in ["train", "val"]:
    D = torch.load(f"{DESC}/{sp}_desc.pt", map_location="cpu", weights_only=False).numpy()
    off = 0
    for c in idl(f"Splits/pdb_out/{sp}.json"):
        di = id2i[c]
        L = len(ents[di]["tokens"])
        desc_raw[di] = D[off : off + L].astype(np.float32)
        off += L


def build_desc_all(mn, sd, dis=None):
    """Return dict di->normalised+masked descriptor, using given mn/sd."""
    keys = dis if dis is not None else pool
    out = {}
    for di in keys:
        r = desc_raw[di]
        if len(r) == 0:
            out[di] = np.zeros((0, 49), np.float32)
            continue
        out[di] = ((r - mn) / sd).astype(np.float32) * MASK
    return out


print(
    f"CV pool: {len(pool)} chains  DEPLOYED training, K sweep + expansion(thresh={THRESH_FRAC}*k), seeds={N_SEEDS}",
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
_ks_env = os.environ.get("RFOLDSEARCH_CODEBOOK_K", "")
KS = ([int(_ks_env)] if _ks_env else [20, 40, 60, 80, 100])
_fold_env = os.environ.get("RFOLDSEARCH_CODEBOOK_FOLD", "")
FOLD_LIST = ([int(_fold_env)] if _fold_env else list(range(5)))

# Structural training pairs are loaded once.  A fold only uses pairs whose
# two members are in that fold's training subset, preventing test leakage.
STRUCT_PAIRS = []
_usalign_cache = os.path.join("files", "production", "retrieval",
                              "usalign_full.cache.jsonl")
if os.path.exists(_usalign_cache):
    with open(_usalign_cache) as _handle:
        for _line in _handle:
            _row = json.loads(_line)
            if 0.5 * (float(_row["tm_query"]) + float(_row["tm_target"])) >= 0.5:
                _a, _b = int(_row["i"]), int(_row["j"])
                if _a != _b:
                    STRUCT_PAIRS.append((_a, _b) if _a < _b else (_b, _a))
STRUCT_PAIRS = sorted(set(STRUCT_PAIRS))
print(f"Structural matrix pool: {len(STRUCT_PAIRS)} pairs with symmetric TM>=0.5",
      flush=True)
OUT_PATH = os.environ.get(
    "RFOLDSEARCH_CODEBOOK_OUT", "files/ablation/cv_s4_codebook_size_5fold.json")
UNLABELLED_POLICY = os.environ.get("RFOLDSEARCH_UNLABELLED_POLICY", "neutral").lower()
if UNLABELLED_POLICY not in {"neutral", "false_positive"}:
    raise ValueError("RFOLDSEARCH_UNLABELLED_POLICY must be neutral or false_positive")
print(f"Unlabelled-target policy: {UNLABELLED_POLICY}", flush=True)
MATRIX_MODE = os.environ.get("RFOLDSEARCH_MATRIX", "cosine").lower()
if MATRIX_MODE not in {"structure_blosum", "cosine"}:
    raise ValueError("RFOLDSEARCH_MATRIX must be structure_blosum or cosine")
print(f"Substitution matrix: {MATRIX_MODE}", flush=True)
print(f"Prefilter k-mer length: {KMER_K}", flush=True)


def cosine_warmup(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def train_one(K, seed, Xtr, Xval):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = DescriptorVQVAE(
        d_descriptor=D_DESC,
        d_hidden=D_HIDDEN,
        d_latent=D_LATENT,
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


def expand_kmer(qk, cos, K, thresh):
    k = len(qk)
    powers = np.array([K ** (k - 1 - i) for i in range(k)], dtype=np.int64)
    out = {int(np.dot(qk, powers))}
    for pos in range(k):
        for tok in range(K):
            if tok == qk[pos]:
                continue
            score = (k - 1) * 1.0 + cos[qk[pos], tok]
            if score >= thresh:
                trial = qk.copy()
                trial[pos] = tok
                out.add(int(np.dot(trial, powers)))
    return list(out)


def build_struct_blosum(tok_by_di, pair_pool, K):
    """Build the deployed Structure-BLOSUM analogue for one fold/codebook."""
    identity = np.eye(K, dtype=np.float32)
    joint = np.zeros((K, K), dtype=np.float64)
    n_aligned = 0
    for a, b in pair_pool:
        q, t = tok_by_di.get(a), tok_by_di.get(b)
        if q is None or t is None or len(q) == 0 or len(t) == 0:
            continue
        _, pairs = smith_waterman_traceback(
            q, t, identity, BLOSUM_GAP_OPEN, BLOSUM_GAP_EXTEND)
        for qi, ti in pairs:
            x, y = int(q[qi]), int(t[ti])
            joint[x, y] += 1.0
            joint[y, x] += 1.0
        n_aligned += 1
    total = max(float(joint.sum()), 1.0)
    p_ij = joint / total
    p_i = np.maximum(p_ij.sum(axis=1), 1e-12)
    S = BLOSUM_SCALE * np.log2(
        np.maximum(p_ij, 1e-12) / np.maximum(np.outer(p_i, p_i), 1e-12))
    S = S.astype(np.float32)
    S *= 1.0 / max(float(np.diag(S).mean()), 1e-6)
    if n_aligned == 0:
        raise RuntimeError("no structural training pairs available for matrix")
    return S


def retrieval_eval(m, K, heldout_dis, desc_all, struct_pairs, collect_postings=False):
    """Run retrieval over ALL pool chains; return (r1_mean, sens_mean, nq)."""
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
    if MATRIX_MODE == "cosine":
        cbn = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-9)
        sub = (cbn @ cbn.T).astype(np.float32)
    else:
        sub = build_struct_blosum(toks, struct_pairs, K)
    order = pool
    pos = {d: i for i, d in enumerate(order)}
    tok_list = [toks[d] for d in order]
    k = KMER_K
    powers = np.array([K ** (k - 1 - i) for i in range(k)], dtype=np.int64)
    idx = defaultdict(list)
    for ti, tt in enumerate(tok_list):
        for p in range(len(tt) - k + 1):
            kid = int(np.dot(tt[p : p + k], powers))
            idx[kid].append((ti, p))
    smith_waterman_numba(np.zeros(4, np.int64), np.zeros(4, np.int64), sub, -2.0, -0.5)
    thresh = THRESH_FRAC * k
    n_pool = len(order)
    fam_pool = np.array([fams[d] for d in order], dtype=object)
    clan_pool = np.array([clans[d] for d in order], dtype=object)
    famcnt = defaultdict(int)
    for d in order:
        if fams[d]:
            famcnt[fams[d]] += 1
    qs = [
        pos[d]
        for d in heldout_dis
        if fams[d] and famcnt[fams[d]] >= 2 and k <= len(tok_list[pos[d]]) <= QMAX
    ]
    famS = []
    clanS = []
    postings = []
    for q in qs:
        tt = tok_list[q]
        td = defaultdict(dict)
        n_post = 0
        for qp in range(len(tt) - k + 1):
            qk = tt[qp : qp + k]
            for kid in expand_kmer(qk, sub, K, thresh):
                for ti, tp in idx.get(kid, ()):
                    if ti != q:
                        n_post += 1
                        d = qp - tp
                        td[ti][d] = td[ti].get(d, 0) + 1
        if collect_postings:
            postings.append(n_post)
        sc = [(ti, max((c for c in dm.values() if c >= 2), default=-1)) for ti, dm in td.items()]
        sc = [x for x in sc if x[1] > 0]
        sc.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break: score, then target index
        cand = np.array([x[0] for x in sc[:100]], np.int64)
        if len(cand) == 0:
            famS.append(0.0)
            continue
        r = cand[
            np.argsort(
                -np.array([smith_waterman_numba(tt, tok_list[t], sub, -2.0, -0.5) for t in cand]),
                kind="stable",
            )
        ]
        va = np.arange(n_pool) != q
        rel_f = (fam_pool == fam_pool[q]) & va
        # Unlabelled targets can either be neutral (the current default) or
        # count as false positives for the historical ablation convention.
        neu_f = ((fam_pool == "") if UNLABELLED_POLICY == "neutral" else False) | (
            ((clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va) if clan_pool[q] else False
        )
        famS.append(first_fp_sensitivity(r, rel_f, neu_f))
        if clan_pool[q]:
            rel_c = (clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va
            if rel_c.any():
                # Apply the same unlabelled-target policy at clan level.
                neu_c = ((fam_pool == "") if UNLABELLED_POLICY == "neutral" else False) | ((fam_pool == fam_pool[q]) & va)
                clanS.append(first_fp_sensitivity(r, rel_c, neu_c))
    fam_m = float(np.mean(famS)) if famS else 0.0
    clan_m = float(np.mean(clanS)) if clanS else 0.0
    if collect_postings:
        return fam_m, clan_m, len(famS), len(clanS), float(np.mean(postings)) if postings else 0.0
    return fam_m, clan_m, len(famS), len(clanS)


def eval_fold(K, fold_idx):
    heldout = set(folds[fold_idx])
    # Split training-fold chains into train (80%) and val (20%) for checkpoint selection
    tr_dis = [di for di in pool if di not in heldout]
    rng_f = np.random.default_rng(1000 + fold_idx)
    rng_f.shuffle(tr_dis)
    cut = int(len(tr_dis) * 0.8)
    val_dis = set(tr_dis[cut:])
    tr_dis_set = set(tr_dis[:cut])
    struct_pairs = [p for p in STRUCT_PAIRS
                    if p[0] in tr_dis_set and p[1] in tr_dis_set]
    # FOLD-SPECIFIC z-score: compute mn/sd from THIS fold's training subset only
    _raw_tr = np.vstack([desc_raw[di] for di in tr_dis_set if len(desc_raw[di]) > 0])
    mn = _raw_tr.mean(0)
    sd = _raw_tr.std(0) + 1e-9
    desc_all = build_desc_all(mn, sd)  # normalise all pool chains with this fold's stats
    Xtr = np.vstack([desc_all[di] for di in tr_dis_set if len(desc_all[di]) > 0])
    Xval = np.vstack([desc_all[di] for di in val_dis if len(desc_all[di]) > 0])
    best_score = -1
    best_m = None
    for seed in range(N_SEEDS):
        m, vl = train_one(K, 42 + seed, Xtr, Xval)
        # tokenizer selection by val retrieval (fam+clan Sens, queries = val_dis)
        fs_v, cs_v, _, _ = retrieval_eval(m, K, val_dis, desc_all, struct_pairs)
        if fs_v + cs_v > best_score:
            best_score = fs_v + cs_v
            best_m = m
    # Evaluate on the actual CV heldout fold
    if os.environ.get("RFOLDSEARCH_POSTINGS_ONLY", "0") == "1":
        return retrieval_eval(best_m, K, heldout, desc_all, struct_pairs, collect_postings=True)
    fs, cs, nf, nc = retrieval_eval(best_m, K, heldout, desc_all, struct_pairs)
    return fs, cs, nf, nc


out = {}
t_all = time.time()
for K in KS:
    key = f"K{K}"
    fam_folds = []
    clan_folds = []
    nqf = []
    nqc = []
    post_folds = []
    t0 = time.time()
    for f in FOLD_LIST:
        result = eval_fold(K, f)
        if os.environ.get("RFOLDSEARCH_POSTINGS_ONLY", "0") == "1":
            fs, cs, nf, nc, pq = result
        else:
            fs, cs, nf, nc = result
            pq = None
        fam_folds.append(fs)
        clan_folds.append(cs)
        nqf.append(nf)
        nqc.append(nc)
        if pq is not None:
            post_folds.append(pq)
            print(f"    K={K} fold={f}: postings/query={pq:.1f}", flush=True)
        print(
            f"    K={K} fold={f}: famSens={fs:.3f} clanSens={cs:.3f} nqf={nf} nqc={nc}  ({time.time()-t0:.0f}s cumulative)",
            flush=True,
        )
    fam_mean, fam_std = float(np.mean(fam_folds)), float(np.std(fam_folds))
    clan_mean, clan_std = float(np.mean(clan_folds)), float(np.std(clan_folds))
    out[key] = {
        "K": K,
        "fam_sens_mean": fam_mean,
        "fam_sens_std": fam_std,
        "clan_sens_mean": clan_mean,
        "clan_sens_std": clan_std,
        "fam_sens_folds": fam_folds,
        "clan_sens_folds": clan_folds,
        "nq_fam": nqf,
        "nq_clan": nqc,
    }
    if os.environ.get("RFOLDSEARCH_POSTINGS_ONLY", "0") == "1":
        out[key]["postings_per_query_folds"] = post_folds
        out[key]["postings_per_query_mean"] = float(np.mean(post_folds)) if post_folds else 0.0
        out[key]["postings_per_query_std"] = float(np.std(post_folds)) if post_folds else 0.0
    print(
        f"  K={K}: famSens={fam_mean:.3f}±{fam_std:.3f}  clanSens={clan_mean:.3f}±{clan_std:.3f}  ({time.time()-t0:.0f}s)",
        flush=True,
    )
    out["_provenance"] = {
        "truth_policy": f"unlabelled targets are {UNLABELLED_POLICY}",
        "cv": "PDB-disjoint 5-fold",
        "descriptor": "17-D g_ctx + g_nn1",
        "matrix": MATRIX_MODE,
        "kmer_k": KMER_K,
        "candidate_depth": 100,
    }
    json.dump(out, open(OUT_PATH, "w"), indent=2)
print(f"\ntotal: {time.time()-t_all:.0f}s\nDONE", flush=True)
