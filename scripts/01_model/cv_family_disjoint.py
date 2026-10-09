"""Family-disjoint vs PDB-disjoint 5-fold CV

  --split pdb     PDB-ID-disjoint folds, as in ablation_substitution_matrix.py.
  --split family  Rfam-family-disjoint folds. Labelled families are assigned
                  greedily to 5 folds, balanced by query count. For each fold,
                  unlabelled chains similar to any test chain are removed from
                  that fold's training set (VQ-VAE, z-score, Structure-BLOSUM);
                  unlabelled chains similar to no test chain stay in every
                  training set. Evaluation labels and the search database
                  (whole pool) are unchanged.

An unlabelled chain is removed from a fold's training set if any criterion holds:
  tm      symmetric TM-score, mean(TM_query, TM_target) >= 0.45, against any test chain of the fold
  blastn  BLASTN E < 0.01 against any test chain (either direction)
  nhmmer  nhmmer E < 0.01 against any test chain (either direction)
  cmscan  relaxed cmscan Rfam hit (E < 0.1) to one of the fold's test families
Sequence hits come from cv_unlabelled_homology.py (--seq-hits).
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import CV_DESC, DATA_ROOT, PRODUCTION_DB, RFOLDSEARCH_REPO, SRC_MODELS, SRC_SEARCH, resolve

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_sys.path.insert(0, RFOLDSEARCH_REPO)
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
USALIGN_CACHE = resolve("files/production/retrieval/usalign_full.cache.jsonl")
_os.chdir(DATA_ROOT)
import argparse, json, time, math, copy, re
from collections import defaultdict
import numpy as np, torch
from numba import njit

from vqvae import DescriptorVQVAE, descriptor_loss
from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels
from smith_waterman import smith_waterman_numba
from rfoldseek.structure_score import c4_coords, score_pair

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["pdb", "family", "homology", "pdb-random"], required=True)
ap.add_argument("--folds", default="0,1,2,3,4")
ap.add_argument("--unlab-tm", type=float, default=0.45)
ap.add_argument("--criteria", default="tm,blastn,nhmmer,cmscan")
ap.add_argument("--seq-hits", default="files/ablation/cv_family_disjoint_unlabelled_seqhits.tsv")
ap.add_argument("--dry-run", action="store_true", help="print fold assignment only")
ap.add_argument("--seed", type=int, default=42, help="VQ-VAE training seed; folds and val split fixed")
ap.add_argument("--out", default=None)
args = ap.parse_args()

QMAX = 500
EPOCHS = 50
BATCH = 512
LR = 3e-4
WD = 1e-2
WARMUP = 200
CLIP = 1.0
KMER_K = 4
THRESH_FRAC = 0.75
TOPN = 200
TIE_SEEDS = (0, 1, 2)   # sha256 tie-breaking seeds (rfoldsearch.tiebreak); per-query values are their mean
K = 80
N_ENC, N_DEC, DH, DL = 2, 4, 128, 128
D_DESC = 49
ACTIVE = {"g_ctx", "g_nn1"}
BLOSUM_SCALE = 2.0
BLOSUM_GAP_OPEN = -11.0
BLOSUM_GAP_EXTEND = -1.0
STRUCT_TM_THRESH = 0.5
SW_GAP_OPEN = -2.0
SW_GAP_EXTEND = -0.5
db = torch.load(PRODUCTION_DB, map_location="cpu", weights_only=False)
ents = db["entries"]
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
fams = attach_labels(ents, load_labels("Splits/labels.tsv"))
clans = attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc"))


def idl(p):
    return [str(json.loads(l)["id"]) for l in open(p) if l.strip()]


pool = [
    id2i[c] for c in idl("Splits/pdb_out/train.json") + idl("Splits/pdb_out/val.json") if c in id2i
]
DESC = CV_DESC
# Per-fold tokens and Structure-BLOSUM, so the search can be re-scored without retraining.
CACHE = _os.path.join(_REPO, "files", "ablation", "cv_cache")
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

famcnt = defaultdict(int)
for d in pool:
    if fams[d]:
        famcnt[fams[d]] += 1


def is_query(d):
    return bool(fams[d]) and famcnt[fams[d]] >= 2 and KMER_K <= len(ents[d]["tokens"]) <= QMAX


# US-align cache: symmetric TM >= 0.5 pairs for Structure-BLOSUM, and
# symmetric TM per pair for the homology filters.
STRUCT_PAIRS = []
TM_SYM = {}
LAB_SEQ_HITS = "files/ablation/cv_labelled_seqhits.tsv"
with open(USALIGN_CACHE) as _f:
    for _l in _f:
        _d = json.loads(_l)
        _a, _b = _d["i"], _d["j"]
        if _a == _b:
            continue
        _k = (_a, _b) if _a < _b else (_b, _a)
        _tm = 0.5 * (_d["tm_query"] + _d["tm_target"])
        TM_SYM[_k] = max(TM_SYM.get(_k, 0.0), _tm)
        if _tm >= STRUCT_TM_THRESH:
            STRUCT_PAIRS.append((_k[0], _k[1], _tm))
STRUCT_PAIRS = sorted({(a, b): t for a, b, t in STRUCT_PAIRS}.items())
STRUCT_PAIRS = [(a, b, t) for (a, b), t in STRUCT_PAIRS]

# ---------------------------------------------------------------- folds
def pdb_folds():
    def _pdb_of(di):
        m = re.search(r"PDB_0000([A-Z0-9]{4})_", str(ents[di]["id"]))
        return m.group(1) if m else str(ents[di]["id"])

    groups = defaultdict(list)
    for di in pool:
        groups[_pdb_of(di)].append(di)
    keys = sorted(groups)
    np.random.default_rng(0).shuffle(keys)
    out = [[] for _ in range(5)]
    for pi, p in enumerate(keys):
        out[pi % 5].extend(groups[p])
    return out, [sorted({fams[d] for d in out[f] if fams[d]}) for f in range(5)]


def family_folds():
    groups = defaultdict(list)
    for di in pool:
        if fams[di]:
            groups[fams[di]].append(di)
    nq = {f: sum(is_query(d) for d in ds) for f, ds in groups.items()}
    # largest families first; ties broken by accession after a seeded shuffle
    keys = sorted(groups)
    np.random.default_rng(0).shuffle(keys)
    keys.sort(key=lambda f: (-nq[f], -len(groups[f])))
    ffams = [[] for _ in range(5)]
    q, nn = [0] * 5, [0] * 5
    for f in keys:
        tgt = min(range(5), key=lambda i: (q[i], nn[i], i))
        ffams[tgt].append(f)
        q[tgt] += nq[f]
        nn[tgt] += len(groups[f])
    return [sorted(d for f in ffams[i] for d in groups[f]) for i in range(5)], ffams


_tag2i = {_os.path.basename(str(ents[d]["id"])).removesuffix(".cif"): d for d in pool}
# unlabelled -> labelled chains (blastn/nhmmer) or families (cmscan)
hits = defaultdict(lambda: defaultdict(set))
for _l in list(open(_os.path.join(_REPO, args.seq_hits)))[1:]:
    _u, _t, _fam, _m, _e = _l.rstrip("\n").split("\t")
    hits[_m][_tag2i[_u]].add(_tag2i[_t] if _t else _fam)
unlab = [d for d in pool if not fams[d]]


def unlabelled_removal(folds_, ffams):
    """--split family: unlabelled chains related to a held-out chain (symmetric TM, sequence, cmscan)."""
    criteria = args.criteria.split(",")
    rem, rem_by = [], []
    for i in range(5):
        test_i, fam_i = set(folds_[i]), set(ffams[i])
        by = {}
        for c in criteria:
            if c == "tm":
                by[c] = {u for u in unlab
                         if any(TM_SYM.get((min(u, t), max(u, t)), 0.0) >= args.unlab_tm for t in folds_[i])}
            elif c == "cmscan":
                by[c] = {u for u in unlab if hits[c][u] & fam_i}
            else:
                by[c] = {u for u in unlab if hits[c][u] & test_i}
        rem.append(sorted(set().union(*by.values())))
        rem_by.append({c: len(v) for c, v in by.items()})
    return rem, rem_by


def homology_removal(folds_, ffams):
    """--split homology: every training chain, labelled or not, related to a held-out chain.

    tm      symmetric TM-score (mean of the two normalisations) >= 0.45
    blastn  BLASTN E < 0.01    (labelled: cv_labelled_seqhits.tsv; unlabelled: --seq-hits)
    nhmmer  nhmmer E < 0.01    (same sources)
    clan    labelled chain of another family in a held-out family's Rfam clan
    cmscan  unlabelled chain with a relaxed cmscan hit (E < 0.1) to a held-out family
    """
    lab_hits = defaultdict(lambda: defaultdict(set))
    for _l in list(open(_os.path.join(_REPO, LAB_SEQ_HITS)))[1:]:
        a, b, m, _e = _l.rstrip("\n").split("\t")
        if a in _tag2i and b in _tag2i:
            lab_hits[m][_tag2i[a]].add(_tag2i[b])
            lab_hits[m][_tag2i[b]].add(_tag2i[a])
    rem, rem_by = [], []
    for i in range(5):
        test_i, fam_i = set(folds_[i]), set(ffams[i])
        clan_i = {clans[d] for d in folds_[i] if clans[d]}
        cand = [d for d in pool if d not in test_i]
        by = {
            "tm": {d for d in cand if any(TM_SYM.get((min(d, t), max(d, t)), 0.0) >= args.unlab_tm for t in folds_[i])},
            "blastn": {d for d in cand if (lab_hits if fams[d] else hits)["blastn"][d] & test_i},
            "nhmmer": {d for d in cand if (lab_hits if fams[d] else hits)["nhmmer"][d] & test_i},
            "clan": {d for d in cand if fams[d] and clans[d] in clan_i},
            "cmscan": {d for d in cand if not fams[d] and hits["cmscan"][d] & fam_i},
        }
        rem.append(sorted(set().union(*by.values())))
        rem_by.append({c: len(v) for c, v in by.items()})
    return rem, rem_by


if args.split == "pdb":
    folds, fold_fams = pdb_folds()
    removed, removed_by = [[] for _ in range(5)], [None] * 5
elif args.split == "family":
    folds, fold_fams = family_folds()
    removed, removed_by = unlabelled_removal(folds, fold_fams)
elif args.split == "homology":
    folds, fold_fams = family_folds()
    removed, removed_by = homology_removal(folds, fold_fams)
else:  # pdb-random: PDB folds, random training chains removed in the proportions of --split homology
    _hf, _hff = family_folds()
    _hrem, _ = homology_removal(_hf, _hff)
    folds, fold_fams = pdb_folds()
    removed, removed_by = [], []
    for i in range(5):
        _rng = np.random.default_rng(7 + i)
        _hcand = [d for d in pool if d not in set(_hf[i])]
        _cand = [d for d in pool if d not in set(folds[i])]
        _out = []
        for is_lab in (True, False):
            _frac = (sum(bool(fams[d]) == is_lab for d in _hrem[i])
                     / max(1, sum(bool(fams[d]) == is_lab for d in _hcand)))
            _c = sorted(d for d in _cand if bool(fams[d]) == is_lab)
            _out += list(_rng.choice(_c, size=round(_frac * len(_c)), replace=False))
        removed.append(sorted(int(d) for d in _out))
        removed_by.append({"random_labelled": sum(bool(fams[d]) for d in _out),
                           "random_unlabelled": sum(not fams[d] for d in _out)})
test_sets = [set(f) for f in folds]
train_pools = [[d for d in pool if d not in test_sets[i] and d not in set(removed[i])] for i in range(5)]

fold_info = []
for i in range(5):
    fold_info.append(
        {
            "fold": i,
            "n_test_chains": len(folds[i]),
            "n_test_families": len(fold_fams[i]),
            "n_queries": sum(is_query(d) for d in folds[i]),
            "n_clan_queries": sum(
                is_query(d)
                and bool(clans[d])
                and any(clans[x] == clans[d] and fams[x] != fams[d] for x in pool)
                for d in folds[i]
            ),
            "n_unlabelled_removed": sum(not fams[d] for d in removed[i]),
            "n_labelled_removed": sum(bool(fams[d]) for d in removed[i]),
            "n_unlabelled_removed_by": removed_by[i],
            "n_labelled_candidates": sum(bool(fams[d]) for d in pool if d not in test_sets[i]),
            "n_unlabelled_candidates": sum(not fams[d] for d in pool if d not in test_sets[i]),
            "unlabelled_removed_ids": [str(ents[d]["id"]) for d in removed[i] if not fams[d]],
            "labelled_removed_ids": [str(ents[d]["id"]) for d in removed[i] if fams[d]],
            "n_train_chains": len(train_pools[i]),
            "test_families": fold_fams[i] if args.split in ("family", "homology") else None,
        }
    )
    print(
        f"fold {i + 1}: test={len(folds[i])} fams={len(fold_fams[i])} "
        f"queries={fold_info[i]['n_queries']} clanQ={fold_info[i]['n_clan_queries']} "
        f"removed unlab={fold_info[i]['n_unlabelled_removed']} lab={fold_info[i]['n_labelled_removed']}"
        f"/{fold_info[i]['n_labelled_candidates']} {fold_info[i]['n_unlabelled_removed_by'] or ''} "
        f"train={len(train_pools[i])}",
        flush=True,
    )
if args.dry_run:
    _u = set().union(*map(set, removed))
    print(f"removed: per-fold sum={sum(map(len, removed))} unique={len(_u)} "
          f"(labelled {sum(bool(fams[d]) for d in _u)}, unlabelled {sum(not fams[d] for d in _u)})")
    _sys.exit(0)


# ---------------------------------------------------------------- model
def build_desc_all(mn, sd):
    out = {}
    for di in pool:
        r = desc_raw[di]
        if len(r) == 0:
            out[di] = np.zeros((0, 49), np.float32)
            continue
        out[di] = ((r - mn) / sd).astype(np.float32) * MASK
    return out


def cosine_warmup(step, warmup, total):
    if step < warmup:
        return step / max(warmup, 1)
    p = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1 + math.cos(math.pi * p))


def _loss(out, x):
    return descriptor_loss(
        out,
        x,
        beta_commit=1.0,
        lambda_geom=1.0,
        lambda_pair=1.0,
        lambda_stack=1.0,
        lambda_mask=1.0,
        active_blocks=ACTIVE,
    )["loss"]


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
    total = max(1, (len(Xt) + BATCH - 1) // BATCH) * EPOCHS
    step = 0
    best_val = float("inf")
    best_state = None
    for ep in range(EPOCHS):
        m.train()
        pi = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BATCH):
            xb = Xt[pi[i : i + BATCH]]
            loss = _loss(m(xb), xb)
            for g in opt.param_groups:
                g["lr"] = LR * cosine_warmup(step, WARMUP, total)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
            opt.step()
            step += 1
        m.eval()
        with torch.no_grad():
            vl = float(_loss(m(Xv), Xv))
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


def make_cosine(cb):
    cbn = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-9)
    return (cbn @ cbn.T).astype(np.float32)


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


def make_struct_blosum(tok_by_di, tr_set, K_):
    """Structure-BLOSUM from US-align TM>=0.5 pairs inside the fold's training subset."""
    joint = np.zeros((K_, K_), dtype=np.float64)
    n = 0
    sub_id = np.zeros((K_, K_), dtype=np.float32)
    np.fill_diagonal(sub_id, 1.0)
    for a, b, _tm in STRUCT_PAIRS:
        if a not in tr_set or b not in tr_set:
            continue
        q = tok_by_di.get(a)
        t = tok_by_di.get(b)
        if q is None or t is None or len(q) == 0 or len(t) == 0:
            continue
        pi, pj = sw_traceback(q, t, sub_id, BLOSUM_GAP_OPEN, BLOSUM_GAP_EXTEND)
        for x, y in zip(pi, pj):
            joint[x, y] += 1
            joint[y, x] += 1
        n += 1
    p_ij = joint / max(joint.sum(), 1.0)
    p_i = np.maximum(p_ij.sum(1), 1e-12)
    S = np.log2(np.maximum(p_ij, 1e-12) / np.maximum(np.outer(p_i, p_i), 1e-12)) * BLOSUM_SCALE
    S = S.astype(np.float32)
    S = S / max(np.diag(S).mean(), 1e-6)
    return S, n


def expand_kmer(qk, sub, K_, thresh):
    """Similar k-mers as in the final model (Foldseek/MMseqs2 convention): a
    one-substitution variant is admitted iff its full k-mer score under the fold's
    Structure-BLOSUM, sum_i sub[q_i, v_i], is >= thresh."""
    k = len(qk)
    powers = np.array([K_ ** (k - 1 - i) for i in range(k)], dtype=np.int64)
    base = int(np.dot(qk, powers))
    out = [base]
    self_score = float(sum(sub[int(qk[i]), int(qk[i])] for i in range(k)))
    for pos in range(k):
        cur = int(qk[pos])
        rest = self_score - sub[cur, cur]
        for tok in range(K_):
            if tok != cur and rest + sub[cur, tok] >= thresh:
                out.append(base + int(powers[pos]) * (tok - cur))
    return out


def rank_queries(sub, tok_list, q_positions, xyz_list):
    """Deployed search for every query: the similar-k-mer prefilter (top 200), then ranking
    by the structure score S * aTM (Smith-Waterman score times alignment TM-score); one
    ranking per tie-breaking seed (ties at the top-N cut and in the ranking)."""
    k = KMER_K
    powers = np.array([K ** (k - 1 - i) for i in range(k)], dtype=np.int64)
    idx = defaultdict(list)
    for ti, tt in enumerate(tok_list):
        for p in range(len(tt) - k + 1):
            idx[int(np.dot(tt[p : p + k], powers))].append((ti, p))
    thresh = THRESH_FRAC * k
    go, ge = np.float32(SW_GAP_OPEN), np.float32(SW_GAP_EXTEND)
    out = {}
    for q in q_positions:
        tt = tok_list[q]
        td = defaultdict(dict)
        for qp in range(len(tt) - k + 1):
            for kid in expand_kmer(tt[qp : qp + k], sub, K, thresh):
                for ti, tp in idx.get(kid, ()):
                    if ti != q:
                        d = qp - tp
                        td[ti][d] = td[ti].get(d, 0) + 1
        sc = [(ti, max((c for c in dm.values() if c >= 2), default=-1)) for ti, dm in td.items()]
        sc = [x for x in sc if x[1] > 0]
        c0 = np.array([x[0] for x in sc], np.int64)
        s0 = np.array([x[1] for x in sc])
        score_cache = {}
        out[q] = []
        for keys in tie_keys_by_seed:
            cand = c0[np.lexsort((keys[c0], -s0))][:TOPN] if len(c0) else c0
            for t in cand:
                if int(t) not in score_cache:
                    s_, atm, _ = score_pair(tt, xyz_list[q], tok_list[t], xyz_list[t], sub, go, ge)
                    score_cache[int(t)] = s_ * atm
            s = np.array([score_cache[int(t)] for t in cand])
            out[q].append(cand[np.lexsort((keys[cand], -s))] if len(cand) else cand)
    return out


def score_queries(rankings, fam_pool, clan_pool):
    """Per-query family/clan Sens@1FP (Section 2.2 definitions, unlabelled targets neutral),
    mean over tie-breaking seeds."""
    n = len(fam_pool)
    unlabelled = fam_pool == ""
    recs = []
    for q, runs in rankings.items():
        va = np.arange(n) != q
        has_clan = bool(clan_pool[q])
        rel_c = (clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va if has_clan else None
        clan_q = has_clan and bool(rel_c.any())
        rel_f = (fam_pool == fam_pool[q]) & va
        neu_f = unlabelled | (((clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va) if has_clan else False)
        neu_c = unlabelled | ((fam_pool == fam_pool[q]) & va)
        sens = lambda r, rel, neu: first_fp_sensitivity(r, rel, neu) if len(r) else 0.0  # noqa: E731
        recs.append({"pos": int(q), "fam": float(np.mean([sens(r, rel_f, neu_f) for r in runs])),
                     "clan": float(np.mean([sens(r, rel_c, neu_c) for r in runs])) if clan_q else None})
    return recs


def load_rankings(path):
    z = np.load(path)
    out = defaultdict(list)
    for q, a, b in zip(z["query"], z["offsets"][:-1], z["offsets"][1:]):
        out[int(q)].append(z["flat"][a:b].astype(np.int64))
    return dict(out)


def save_rankings(path, rankings):
    rows = [(q, r) for q, runs in rankings.items() for r in runs]
    np.savez_compressed(path, query=np.array([q for q, _ in rows], np.int32),
                        offsets=np.concatenate([[0], np.cumsum([len(r) for _, r in rows])]),
                        flat=np.concatenate([r for _, r in rows] or [np.zeros(0)]).astype(np.int32))


from rfoldseek.tiebreak import tie_keys  # noqa: E402

tie_keys_by_seed = [np.asarray(tie_keys([_os.path.splitext(_os.path.basename(str(ents[d]["id"])))[0]
                                         for d in pool], sd)) for sd in TIE_SEEDS]


def eval_fold(fi):
    heldout = test_sets[fi]
    tr_dis = list(train_pools[fi])
    rng_f = np.random.default_rng(1000 + fi)
    rng_f.shuffle(tr_dis)
    cut = int(len(tr_dis) * 0.8)
    val_dis = set(tr_dis[cut:])
    tr_set = set(tr_dis[:cut])
    cache = _os.path.join(CACHE, f"{args.split}_seed{args.seed}_fold{fi}.npz")
    if _os.path.exists(cache):
        z = np.load(cache)
        assert len(z["offsets"]) == len(pool) + 1, f"{cache} was written for another pool"
        tok_list = [z["tokens"][z["offsets"][i]:z["offsets"][i + 1]] for i in range(len(pool))]
        toks = dict(zip(pool, tok_list))
        S, n_pairs = z["matrix"], int(z["n_pairs"])
        print(f"  fold {fi + 1}: tokens and matrix from {cache}", flush=True)
    else:
        raw_tr = np.vstack([desc_raw[d] for d in tr_set if len(desc_raw[d]) > 0])
        mn = raw_tr.mean(0)
        sd = raw_tr.std(0) + 1e-9
        desc_all = build_desc_all(mn, sd)
        Xtr = np.vstack([desc_all[d] for d in tr_set if len(desc_all[d]) > 0])
        Xval = np.vstack([desc_all[d] for d in val_dis if len(desc_all[d]) > 0])
        m, _ = train_one(args.seed, Xtr, Xval)
        toks = {}
        cb = None
        for d in pool:
            if len(desc_all[d]) == 0:
                toks[d] = np.zeros(0, np.int64)
                continue
            toks[d], cb = get_codes(m, desc_all[d])
        tok_list = [toks[d] for d in pool]
        S, n_pairs = make_struct_blosum(toks, tr_set, K)
        _os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(cache, tokens=np.concatenate(tok_list).astype(np.int16),
                            offsets=np.concatenate([[0], np.cumsum([len(t) for t in tok_list])]),
                            matrix=np.asarray(S, np.float32), n_pairs=n_pairs)
    tok_list = [np.asarray(t, np.int64) for t in tok_list]
    S = np.asarray(S, np.float32)
    xyz_list = [c4_coords(ents[d], len(t)) if len(t) else np.zeros((0, 3)) for d, t in zip(pool, tok_list)]
    assert all(x is not None for x in xyz_list), "fold tokens do not match the chains' resolved nucleotides"
    fam_pool = np.array([fams[d] for d in pool], dtype=object)
    clan_pool = np.array([clans[d] for d in pool], dtype=object)
    smith_waterman_numba(np.zeros(4, np.int64), np.zeros(4, np.int64), S, SW_GAP_OPEN, SW_GAP_EXTEND)
    q_pos = [i for i, d in enumerate(pool) if d in heldout and is_query(d)]
    rank_cache = cache.replace(".npz", "_rankings.npz")
    if _os.path.exists(rank_cache):
        rankings = load_rankings(rank_cache)
        assert sorted(rankings) == sorted(q_pos), f"{rank_cache} was written for other queries"
    else:
        rankings = rank_queries(S, tok_list, q_pos, xyz_list)
        save_rankings(rank_cache, rankings)
    recs = score_queries(rankings, fam_pool, clan_pool)
    for r in recs:
        d = pool[r["pos"]]
        r.update({"id": str(ents[d]["id"]), "family": fams[d], "clan": r["clan"], "clan_acc": clans[d] or None, "fold": fi})
        del r["pos"]
    return recs, n_pairs, len(tr_set), len(val_dis)


_suffix = "" if args.seed == 42 else f"_seed{args.seed}"
out_path = args.out or _os.path.join(_REPO, f"files/ablation/cv_{args.split}_disjoint_5fold{_suffix}.json")
result = {
    "split": args.split,
    "seed": args.seed,
    "unlabelled_filter": (
        {"criteria": args.criteria.split(","), "tm": "mean(TM_q, TM_t)", "tm_thr": args.unlab_tm,
         "blastn_nhmmer_E": 0.01, "cmscan_E": 0.1, "seq_hits": args.seq_hits}
        if args.split == "family" else
        {"criteria": ["tm", "blastn", "nhmmer", "clan", "cmscan"], "tm": "mean(TM_q, TM_t)",
         "tm_thr": args.unlab_tm, "blastn_nhmmer_E": 0.01, "cmscan_E": 0.1, "applies_to": "labelled and unlabelled",
         "seq_hits": [args.seq_hits, LAB_SEQ_HITS]}
        if args.split == "homology" else
        {"random_removal": "labelled and unlabelled fractions matched to --split homology per fold"}
        if args.split == "pdb-random" else None
    ),
    "folds": fold_info,
    "per_query": [],
}
t_all = time.time()
for fi in [int(x) for x in args.folds.split(",")]:
    t0 = time.time()
    recs, n_pairs, n_tr, n_val = eval_fold(fi)
    fam_s = [r["fam"] for r in recs]
    clan_s = [r["clan"] for r in recs if r["clan"] is not None]
    fold_info[fi].update(
        {
            "struct_blosum_pairs": n_pairs,
            "vqvae_train_chains": n_tr,
            "vqvae_val_chains": n_val,
            "fam_sens": float(np.mean(fam_s)) if fam_s else None,
            "clan_sens": float(np.mean(clan_s)) if clan_s else None,
            "nqf": len(fam_s),
            "nqc": len(clan_s),
        }
    )
    result["per_query"].extend(recs)
    print(
        f"fold={fi + 1} famSens={fold_info[fi]['fam_sens']:.3f} clanSens={fold_info[fi]['clan_sens'] or 0:.3f} "
        f"nqf={len(fam_s)} nqc={len(clan_s)} pairs={n_pairs} ({time.time()-t0:.0f}s)",
        flush=True,
    )
    json.dump(result, open(out_path, "w"), indent=1)
done = [f for f in fold_info if "fam_sens" in f]
if len(done) == 5:
    pq = result["per_query"]
    result["summary"] = {
        "fam_fold_mean": float(np.mean([f["fam_sens"] for f in done])),
        "fam_fold_std": float(np.std([f["fam_sens"] for f in done])),
        # folds without cross-family clan queries (e.g. a clan-less fold) are skipped
        "clan_fold_mean": float(np.mean([f["clan_sens"] for f in done if f["nqc"]])),
        "clan_fold_std": float(np.std([f["clan_sens"] for f in done if f["nqc"]])),
        "fam_pooled": float(np.mean([r["fam"] for r in pq])),
        "clan_pooled": float(np.mean([r["clan"] for r in pq if r["clan"] is not None])),
        "nqf": len(pq),
        "nqc": sum(r["clan"] is not None for r in pq),
    }
    json.dump(result, open(out_path, "w"), indent=1)
    print(json.dumps(result["summary"], indent=1))
print(f"total {time.time()-t_all:.0f}s -> {out_path}", flush=True)
