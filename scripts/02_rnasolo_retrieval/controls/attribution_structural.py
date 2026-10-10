"""Attribution of the RiboSeek -> RFoldSearch gap under the STRUCTURE-DEFINED
ground truth, mirroring attribution_paths.py.
"""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
import numpy as np, torch
import importlib.resources as R

PC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(PC, "scripts", "02_rnasolo_retrieval", "controls"))
os.environ.setdefault("RFS_ALPHABET_CONTROL_MATRIX", "native")
import riboseek_alphabet_control as F
from riboseek import Searcher
from riboseek.align import NWAligner
from riboseek.prefilter import KmerPrefilter

sys.path.insert(0, os.path.join(PC, "src", "search"))
from prefilter_rfam_benchmark import first_fp_sensitivity

B, SEED = 2000, 123
LEVEL_SEED = {"family": SEED + 1, "clan": SEED + 2}
PREF_K, PREF_N = 6, 500
THRESHOLDS = [0.40, 0.45, 0.50]

db = torch.load(F.resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
ents = db["entries"]
n = len(ents)
lab = F.resolve("Splits/labels.tsv")
fam = F.attach_labels(ents, F.load_labels(lab))
clan = F.attach_labels(ents, F.load_labels(lab, label_column="clan_acc"))
fam_arr = np.array(fam, dtype=object)

# ---- structure-defined ground truth: the same TM matrix Fig 2C,D uses --------
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
test_ids = [str(json.loads(l)["id"]) for l in open(F.resolve("Splits/pdb_out/test.json")) if l.strip()]
test_dis = set(id2i[c] for c in test_ids if c in id2i)
tm = np.full((n, n), np.nan, dtype=np.float32)
cov = defaultdict(int)
for line in open(F.resolve("files/production/retrieval/usalign_full.cache.jsonl")):
    if not line.strip():
        continue
    r = json.loads(line)
    i, j = r["i"], r["j"]
    a = 0.5 * (r["tm_query"] + r["tm_target"])
    tm[i, j] = a
    tm[j, i] = a
    cov[i] += 1
    cov[j] += 1
queries = sorted(q for q in range(n) if cov[q] >= n - 5 and q in test_dis)
print(f"test covered queries: {len(queries)}", flush=True)

# ---- the five configurations, identical to attribution_paths.py -------------
enc = json.load(open(F.resolve("files/ablation/riboseek_real_encoded.json")))
sa20 = [np.asarray(enc[str(i)]["labels"], dtype=np.int64) for i in range(n)]
tdr = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
with R.as_file(R.files("riboseek.data").joinpath("demo_db.json")) as p:
    pre = Searcher.from_pretrained("sa20", db=str(p))
M_sa20 = np.asarray(pre.alphabet.score_matrix, dtype=np.float64)
M_3dr = torch.load(F.resolve("files/models/sub_matrix.pt"), map_location="cpu",
                   weights_only=False)["matrix"].numpy().astype(np.float64)
pairs, _ = F.load_train_pairs(ents)
M_sa20_sb = F.build_self_blosum(sa20, pairs, 20)[0].astype(np.float64)


def ribo_ranks(toks, mat, name):
    """RiboSeek as published: IDF 6-mer set prefilter + linear-gap NW+SW."""
    t0 = time.time()
    aln = NWAligner(mat, gap_penalty=-2.0)
    chains = {str(i): {"labels": toks[i].tolist()} for i in range(n)}
    pf = KmerPrefilter(chains, k=PREF_K)
    out = {}
    for q in queries:
        cand = [int(k) for k, _ in pf.candidates(toks[q].tolist(), top_n=PREF_N) if int(k) != q]
        if not cand:
            out[q] = np.zeros(0, dtype=np.int64)
            continue
        nw = np.array([aln.align(toks[q], toks[t], local=False) for t in cand])
        sw = np.array([aln.align(toks[q], toks[t], local=True) for t in cand])
        # RiboSeek ranks by the mean of the two z-scores over the candidate set
        sc = ((nw - nw.mean()) / (nw.std() + 1e-10)
              + (sw - sw.mean()) / (sw.std() + 1e-10)) / 2.0
        out[q] = np.asarray(cand, dtype=np.int64)[np.argsort(-sc, kind="stable")]
    print(f"  {name}: {time.time()-t0:.0f}s", flush=True)
    return out


def rf_ranks(toks, mat, name, xyz=None):
    """RFoldSearch pipeline: expansion 4-mer prefilter + affine-gap local SW,
    ranked by the structure score S * aTM when C4' coordinates are given."""
    t0 = time.time()
    ranker = F.SamePipelineRanker(toks, mat.astype(np.float32), xyz=xyz)
    out = {q: np.asarray(ranker.rank(q), dtype=np.int64) for q in queries}
    print(f"  {name}: {time.time()-t0:.0f}s", flush=True)
    return out


t0 = time.time()
RANKS = {}
RANKS["A_ribo_published"] = ribo_ranks(sa20, M_sa20, "A sa20 | RiboSeek pipeline | own matrix")
RANKS["B_fwd_pipeline"] = rf_ranks(sa20, M_sa20, "B sa20 | RF pipeline | own matrix")
RANKS["C_fwd_matrix"] = rf_ranks(sa20, M_sa20_sb, "C sa20 | RF pipeline | learned matrix")
RANKS["D_3dr_sw"] = rf_ranks(tdr, M_3dr, "D' 3Dr | RF pipeline, SW ranking | learned matrix")
RANKS["D_rfoldsearch"] = rf_ranks(tdr, M_3dr, "D 3Dr | RF pipeline, structure score | learned matrix",
                                xyz=[F.c4_coords(e, len(t)) for e, t in zip(ents, tdr)])
M_3dr_id = F.normalize_matrix(F.identity_matrix(80), 1.0).astype(np.float64)   # +2/-1, unit diagonal mean
RANKS["E0_rev_alphabet"] = ribo_ranks(tdr, M_3dr_id, "E0 3Dr | RiboSeek pipeline | identity matrix")
RANKS["E_rev_alphabet"] = ribo_ranks(tdr, M_3dr, "E 3Dr | RiboSeek pipeline | learned matrix")
print(f"[rankings {time.time()-t0:.0f}s]", flush=True)


def score(cfg, th, level):
    """Per-query Sens@1FP under the structure-defined ground truth at `th`."""
    per = {}
    va_all = np.arange(n)
    for q in queries:
        va = va_all != q
        pos = (~np.isnan(tm[q])) & (tm[q] >= th) & va
        r = RANKS[cfg][q]
        if level == "family":
            if not pos.any():
                continue
            per[q] = 0.0 if len(r) == 0 else first_fp_sensitivity(r, pos, None)
        else:
            if not fam_arr[q]:
                continue
            rel_c = pos & (fam_arr != fam_arr[q])
            if not rel_c.any():
                continue
            neu_c = (fam_arr == fam_arr[q]) & va
            per[q] = first_fp_sensitivity(r if len(r) else np.zeros(0, np.int64), rel_c, neu_c)
    return per


def ci(a, b, level, seed):
    """Paired block bootstrap over Rfam families (family) or clans (clan)."""
    bd = defaultdict(list)
    labs = fam if level == "family" else clan
    for q in a:
        blk = labs[int(q)]
        if blk:
            bd[blk].append(a[q] - b.get(q, 0.0))
    names = list(bd)
    if not names:
        return None
    rng = np.random.default_rng(seed)
    reps = [rng.choice(names, size=len(names), replace=True) for _ in range(B)]
    dm = np.array([np.mean([v for x in s for v in bd[x]]) for s in reps])
    lo, hi = np.percentile(dm, [2.5, 97.5])
    return {"diff": float(np.mean([v for x in names for v in bd[x]])),
            "lo": float(lo), "hi": float(hi), "n_blocks": len(names),
            "sig": bool(lo > 0 or hi < 0)}


FWD = [("+ RFoldSearch pipeline", "B_fwd_pipeline", "A_ribo_published"),
       ("+ learned matrix", "C_fwd_matrix", "B_fwd_pipeline"),
       ("+ 3Dr alphabet", "D_3dr_sw", "C_fwd_matrix"),
       ("+ structure score", "D_rfoldsearch", "D_3dr_sw")]
# Reverse order: the 3Dr alphabet first (scored with an unlearned identity matrix),
# then its learned Structure-BLOSUM, then the RFoldSearch pipeline.
REV = [("+ 3Dr alphabet", "E0_rev_alphabet", "A_ribo_published"),
       ("+ learned matrix", "E_rev_alphabet", "E0_rev_alphabet"),
       ("+ RFoldSearch pipeline", "D_rfoldsearch", "E_rev_alphabet")]

out = {"_config": {"B": B, "seed": SEED, "thresholds": THRESHOLDS,
                   "n_queries": len(queries),
                   "_note": "structure-defined ground truth; blocks are Rfam families/clans"},
       "thresholds": {}}
for th in THRESHOLDS:
    lv_out = {}
    for level in ("family", "clan"):
        per = {c: score(c, th, level) for c in RANKS}
        means = {c: (float(np.mean(list(v.values()))) if v else float("nan")) for c, v in per.items()}
        fwd = {lbl: ci(per[a], per[b], level, LEVEL_SEED[level]) for lbl, a, b in FWD}
        rev = {lbl: ci(per[a], per[b], level, LEVEL_SEED[level]) for lbl, a, b in REV}
        lv_out[level] = {"means": means, "forward": fwd, "reverse": rev,
                         "n_scored": {c: len(v) for c, v in per.items()}}
        print(f"th={th} {level}: " + "  ".join(f"{c.split('_')[0]}={means[c]:.3f}" for c in RANKS),
              flush=True)
        for lbl, e in fwd.items():
            if e:
                print(f"    {lbl:32s} {e['diff']:+.3f} [{e['lo']:+.3f},{e['hi']:+.3f}] "
                      f"blocks={e['n_blocks']} sig={e['sig']}", flush=True)
    out["thresholds"][f"{th}"] = lv_out

dst = F.resolve("files/ablation/attribution_paths_structural.json")
json.dump(out, open(dst, "w"), indent=1)
print(f"\nwrote {dst}  [{time.time()-t0:.0f}s]")
