"""Attribution of the RiboSeek -> RFoldSearch gap, along two orderings.

Forward : pipeline -> matrix -> alphabet
Reverse : alphabet -> pipeline -> matrix
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

B, SEED = 2000, 123
LEVEL_SEED = {"family": SEED + 1, "clan": SEED + 2}
PREF_K, PREF_N = 6, 500

db = torch.load(F.resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
ents = db["entries"]
lab = F.resolve("Splits/labels.tsv")
fam = F.attach_labels(ents, F.load_labels(lab))
clan = F.attach_labels(ents, F.load_labels(lab, label_column="clan_acc"))
fam_q, clan_q = F.load_query_sets(ents, fam, clan)
n = len(ents)

enc = json.load(open(F.resolve("files/ablation/riboseek_real_encoded.json")))
sa20 = [np.asarray(enc[str(i)]["labels"], dtype=np.int64) for i in range(n)]
tdr = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
with R.as_file(R.files("riboseek.data").joinpath("demo_db.json")) as p:
    pre = Searcher.from_pretrained("sa20", db=str(p))
M_sa20 = np.asarray(pre.alphabet.score_matrix, dtype=np.float64)
M_3dr = torch.load(F.resolve("files/models/sub_matrix.pt"), map_location="cpu",
                   weights_only=False)["matrix"].numpy().astype(np.float64)


def ribo_pipeline(toks, mat, name):
    """RiboSeek as published: IDF 6-mer set prefilter + linear-gap NW+SW."""
    aln = NWAligner(mat, gap_penalty=-2.0)
    chains = {str(i): {"labels": toks[i].tolist()} for i in range(n)}
    pf = KmerPrefilter(chains, k=PREF_K)
    per = {"family": {}, "clan": {}}
    for level, qs in (("family", fam_q), ("clan", clan_q)):
        for q in qs:
            cand = [int(kk) for kk, _ in pf.candidates(toks[q].tolist(), top_n=PREF_N) if int(kk) != q]
            if not cand:
                per[level][q] = 0.0; continue
            nw = np.array([aln.align(toks[q], toks[t], local=False) for t in cand])
            sw = np.array([aln.align(toks[q], toks[t], local=True) for t in cand])
            # RiboSeek ranks by the mean of the two z-scores over the candidate set
            sc = ((nw - nw.mean()) / (nw.std() + 1e-10)
                  + (sw - sw.mean()) / (sw.std() + 1e-10)) / 2.0
            order = np.argsort(-sc, kind="stable")
            c = np.asarray(cand)
            fq, cq = fam[q], clan[q]
            if level == "family":
                rel = np.array([fam[j] == fq for j in c]); neu = np.array([not fam[j] or (fam[j] != fq and cq and clan[j] == cq) for j in c])
                n_rel = sum(1 for j in range(n) if j != q and fam[j] == fq)
            else:
                rel = np.array([cq and clan[j] == cq and fam[j] != fq for j in c]); neu = np.array([not fam[j] or fam[j] == fq for j in c])
                n_rel = sum(1 for j in range(n) if j != q and cq and clan[j] == cq and fam[j] != fq)
            tp = 0
            for r, nu in zip(rel[order], neu[order]):
                if r: tp += 1
                elif nu: continue
                else: break
            per[level][q] = tp / n_rel if n_rel else 0.0
    print(f"  {name}: family {np.mean(list(per['family'].values())):.3f} "
          f"clan {np.mean(list(per['clan'].values())):.3f}", flush=True)
    return per


def rf_pipeline(toks, mat, name, xyz=None):
    """RFoldSearch pipeline: expansion 4-mer prefilter + affine-gap local SW,
    ranked by the structure score S * aTM when C4' coordinates are given."""
    _, pq = F.evaluate_ranker(name, F.SamePipelineRanker(toks, mat.astype(np.float32), xyz=xyz),
                              fam_q, clan_q, fam, clan)
    return {lv: {x["query"]: x["sens"] for x in pq[lv]} for lv in ("family", "clan")}


pairs, _ = F.load_train_pairs(ents)
M_sa20_sb = F.build_self_blosum(sa20, pairs, 20)[0].astype(np.float64)

t0 = time.time()
cfg = {}
cfg["A_ribo_published"] = ribo_pipeline(sa20, M_sa20, "A sa20 | RiboSeek pipeline | own matrix")
cfg["B_fwd_pipeline"] = rf_pipeline(sa20, M_sa20, "B sa20 | RF pipeline | own matrix")
cfg["C_fwd_matrix"] = rf_pipeline(sa20, M_sa20_sb, "C sa20 | RF pipeline | learned matrix")
cfg["D_3dr_sw"] = rf_pipeline(tdr, M_3dr, "D' 3Dr | RF pipeline, SW ranking | learned matrix")
cfg["D_rfoldsearch"] = rf_pipeline(tdr, M_3dr, "D 3Dr | RF pipeline, structure score | learned matrix",
                                 xyz=[F.c4_coords(e, len(t)) for e, t in zip(ents, tdr)])
M_3dr_id = F.normalize_matrix(F.identity_matrix(80), 1.0).astype(np.float64)   # +2/-1, unit diagonal mean
cfg["E0_rev_alphabet"] = ribo_pipeline(tdr, M_3dr_id, "E0 3Dr | RiboSeek pipeline | identity matrix")
cfg["E_rev_alphabet"] = ribo_pipeline(tdr, M_3dr, "E 3Dr | RiboSeek pipeline | learned matrix")
print(f"[{time.time()-t0:.0f}s]", flush=True)

def ci(a, b, level, seed):
    bd = defaultdict(list)
    labs = fam if level == "family" else clan
    for q in a:
        blk = labs[int(q)]
        if blk: bd[blk].append(a[q] - b.get(q, 0.0))
    names = list(bd); rng = np.random.default_rng(seed)
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
out = {"levels": {}, "_config": {"B": B, "seed": SEED}}
for level in ("family", "clan"):
    means = {k: float(np.mean(list(v[level].values()))) for k, v in cfg.items()}
    out["levels"][level] = {"means": means, "forward": {}, "reverse": {}}
    print(f"\n--- {level} ---")
    for lbl, hi_k, lo_k in FWD:
        e = ci(cfg[hi_k][level], cfg[lo_k][level], level, LEVEL_SEED[level])
        out["levels"][level]["forward"][lbl] = e
        print("  FWD %-32s %+.3f CI[%+.3f,%+.3f] sig=%s" % (lbl, e["diff"], e["lo"], e["hi"], e["sig"]))
    for lbl, hi_k, lo_k in REV:
        e = ci(cfg[hi_k][level], cfg[lo_k][level], level, LEVEL_SEED[level])
        out["levels"][level]["reverse"][lbl] = e
        print("  REV %-32s %+.3f CI[%+.3f,%+.3f] sig=%s" % (lbl, e["diff"], e["lo"], e["hi"], e["sig"]))
json.dump(out, open(f"{PC}/files/ablation/attribution_paths.json", "w"), indent=2)
print(f"\nWrote {PC}/files/ablation/attribution_paths.json")
