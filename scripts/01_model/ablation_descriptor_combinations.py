"""S3 tier-2 cumulative-combo ablation via 5-fold CV on train+val."""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
from pathlib import Path
import sys, os, json, time
from collections import defaultdict
import numpy as np, torch
from sklearn.cluster import MiniBatchKMeans

from prefilter_rfam_benchmark import (
    attach_labels,
    evaluable_queries,
    first_fp_sensitivity,
    load_labels,
)
from smith_waterman import smith_waterman_numba

QMAX = 500
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
_tr = torch.load(f"{DESC}/train_desc.pt", map_location="cpu", weights_only=False).numpy()
mn = _tr.mean(0)
sd = _tr.std(0) + 1e-9
desc_all = {}
for sp in ["train", "val"]:
    D = torch.load(f"{DESC}/{sp}_desc.pt", map_location="cpu", weights_only=False).numpy()
    off = 0
    for c in idl(f"Splits/pdb_out/{sp}.json"):
        di = id2i[c]
        L = len(ents[di]["tokens"])
        desc_all[di] = ((D[off : off + L] - mn) / sd).astype(np.float32)
        off += L
print(f"CV pool: {len(pool)} chains", flush=True)
import re as _re


def _pdb_of(di):
    m = _re.search(r"PDB_0000([A-Z0-9]{4})_", str(ents[di]["id"]))
    return m.group(1) if m else str(ents[di]["id"])


# PDB-disjoint 5-fold: each PDB's chains stay in one fold (matches deployed protocol)
_pdb_groups = defaultdict(list)
for _di in pool:
    _pdb_groups[_pdb_of(_di)].append(_di)
_pdbs = sorted(_pdb_groups)
np.random.default_rng(0).shuffle(_pdbs)
folds = [[] for _ in range(5)]
for _pi, _p in enumerate(_pdbs):
    folds[_pi % 5].extend(_pdb_groups[_p])
print(f"PDB-disjoint folds: sizes={[len(f) for f in folds]}  n_pdbs={len(_pdbs)}", flush=True)
# tier-2: cumulative feature combinations
COMBOS = {
    "ctx+nn1 (17D)": list(range(42, 49)) + list(range(0, 10)),
    "ctx+nn1+nn2 (27D)": list(range(42, 49)) + list(range(0, 10)) + list(range(10, 20)),
    "ctx+nn1+bp (28D)": list(range(42, 49)) + list(range(0, 10)) + list(range(20, 31)),
    "ctx+nn1+nn2+bp (38D)": list(range(42, 49))
    + list(range(0, 10))
    + list(range(10, 20))
    + list(range(20, 31)),
    "ctx+nn1+bp+st (39D)": list(range(42, 49))
    + list(range(0, 10))
    + list(range(20, 31))
    + list(range(31, 42)),
}
KS = [20, 40, 60, 80]
famv = np.array(fams, dtype=object)
clanv = np.array(clans, dtype=object)


def eval_config(name, K, fold_idx):
    dims = COMBOS[name]
    heldout = set(folds[fold_idx])
    # fit K-means on all training-fold residues (combos already include mask indicators)
    Xtr = np.vstack(
        [desc_all[di][:, dims] for di in pool if di not in heldout and len(desc_all[di]) > 0]
    )
    km = MiniBatchKMeans(
        n_clusters=K, batch_size=10000, max_iter=200, random_state=42, n_init=3
    ).fit(Xtr)
    toks = {
        di: (
            km.predict(desc_all[di][:, dims]).astype(np.int64)
            if len(desc_all[di]) > 0
            else np.zeros(0, np.int64)
        )
        for di in pool
    }
    # centroid cosine substitution matrix
    C = km.cluster_centers_
    Cn = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    cos = (Cn @ Cn.T).astype(np.float32)
    Ktot = K
    order = pool
    pos = {d: i for i, d in enumerate(order)}
    tok_list = [toks[d] for d in order]
    # k-mer prefilter (exact contiguous 4-mer, top-100)
    P = [Ktot**3, Ktot**2, Ktot, 1]
    idx = defaultdict(list)
    for ti, tt in enumerate(tok_list):
        for p in range(len(tt) - 3):
            idx[tt[p] * P[0] + tt[p + 1] * P[1] + tt[p + 2] * P[2] + tt[p + 3]].append((ti, p))
    smith_waterman_numba(np.zeros(4, np.int64), np.zeros(4, np.int64), cos, -2.0, -0.5)
    n_pool = len(order)
    fam_pool = np.array([fams[d] for d in order], dtype=object)
    clan_pool = np.array([clans[d] for d in order], dtype=object)
    famcnt = defaultdict(int)
    for d in order:
        if fams[d]:
            famcnt[fams[d]] += 1
    qs = [
        pos[d]
        for d in heldout
        if fams[d] and famcnt[fams[d]] >= 2 and 4 <= len(tok_list[pos[d]]) <= QMAX
    ]
    famS = []
    clanS = []
    for q in qs:
        tt = tok_list[q]
        td = defaultdict(dict)
        for p in range(len(tt) - 3):
            kid = tt[p] * P[0] + tt[p + 1] * P[1] + tt[p + 2] * P[2] + tt[p + 3]
            for ti, tp in idx.get(kid, ()):
                if ti != q:
                    d = p - tp
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
                -np.array([smith_waterman_numba(tt, tok_list[t], cos, -2.0, -0.5) for t in cand]),
                kind="stable",
            )
        ]
        va = np.arange(n_pool) != q
        rel_f = (fam_pool == fam_pool[q]) & va
        neu_f = (fam_pool == "") | (
            ((clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va) if clan_pool[q] else False
        )
        famS.append(first_fp_sensitivity(r, rel_f, neu_f))
        if clan_pool[q]:
            rel_c = (clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va
            if rel_c.any():
                neu_c = (fam_pool == "") | ((fam_pool == fam_pool[q]) & va)
                clanS.append(first_fp_sensitivity(r, rel_c, neu_c))
    fam_m = float(np.mean(famS)) if famS else 0.0
    clan_m = float(np.mean(clanS)) if clanS else 0.0
    return fam_m, clan_m, len(famS), len(clanS)


out = {}
t_all = time.time()
for K in KS:
    for name in COMBOS:
        key = f"{name}|K{K}"
        fam_folds = []
        clan_folds = []
        nqf = []
        nqc = []
        t0 = time.time()
        for f in range(5):
            fs, cs, nf, nc = eval_config(name, K, f)
            fam_folds.append(fs)
            clan_folds.append(cs)
            nqf.append(nf)
            nqc.append(nc)
        fam_mean, fam_std = float(np.mean(fam_folds)), float(np.std(fam_folds))
        clan_mean, clan_std = float(np.mean(clan_folds)), float(np.std(clan_folds))
        out[key] = {
            "fam_sens_mean": fam_mean,
            "fam_sens_std": fam_std,
            "clan_sens_mean": clan_mean,
            "clan_sens_std": clan_std,
            "fam_sens_folds": fam_folds,
            "clan_sens_folds": clan_folds,
            "nq_fam": nqf,
            "nq_clan": nqc,
        }
        print(
            f"  {name} K={K}: famSens={fam_mean:.3f}±{fam_std:.3f}  clanSens={clan_mean:.3f}±{clan_std:.3f}  nqf={nqf} nqc={nqc} ({time.time()-t0:.0f}s)",
            flush=True,
        )
        json.dump(out, open("files/ablation/cv_s3_clan_5fold.json", "w"), indent=2)
print(f"\ntotal: {time.time()-t_all:.0f}s", flush=True)
print("DONE", flush=True)
