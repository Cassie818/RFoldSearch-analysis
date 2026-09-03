"""Is local alignment enough for the 80-state 3Dr alphabet?

RiboSeek ranks candidates by nw_score + sw_score. Using its own alphabet,
matrix and aligner over the same 885-chain library, this compares ranking the
identical scores three ways: combined (its default), local only, global only.
Exhaustive over all targets, so the prefilter plays no part.
"""
import json, os, sys, time
import numpy as np
import importlib.resources as R
PC = os.path.expanduser("~/Desktop/RFoldseek-paper-code")
sys.path.insert(0, os.path.join(PC, "scripts", "03_ablations"))
os.environ.setdefault("RFS_ALPHABET_CONTROL_MATRIX", "native")
import fig2ef as F
from riboseek import Searcher
from riboseek.align import NWAligner

db = __import__("torch").load(F.resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
ents = db["entries"]
lab = F.resolve("Splits/labels.tsv")
fam = F.attach_labels(ents, F.load_labels(lab))
clan = F.attach_labels(ents, F.load_labels(lab, label_column="clan_acc"))
fam_q, clan_q = F.load_query_sets(ents, fam, clan)

import torch as _t
toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
mat = _t.load(F.resolve("files/models/sub_matrix.pt"), map_location="cpu",
              weights_only=False)["matrix"].numpy().astype(np.float64)
aln = NWAligner(mat, gap_penalty=-2.0)
print(f"alphabet={mat.shape[0]} chains={len(toks)}", flush=True)

n = len(toks)
modes = {"combined (NW+SW)": lambda nw, sw: nw + sw,
         "local only (SW)": lambda nw, sw: sw,
         "global only (NW)": lambda nw, sw: nw}
res = {m: {"family": [], "clan": []} for m in modes}
t0 = time.time()
for level, qs in (("family", fam_q), ("clan", clan_q)):
    for c, q in enumerate(qs, 1):
        others = np.array([j for j in range(n) if j != q])
        nw = np.array([aln.align(toks[q], toks[j], local=False) for j in others])
        sw = np.array([aln.align(toks[q], toks[j], local=True) for j in others])
        fq, cq = fam[q], clan[q]
        if level == "family":
            rel = np.array([fam[j] == fq for j in others])
            neu = np.array([fam[j] != fq and cq and clan[j] == cq for j in others])
        else:
            rel = np.array([cq and clan[j] == cq and fam[j] != fq for j in others])
            neu = np.array([fam[j] == fq for j in others])
        for m, f in modes.items():
            idx = np.argsort(-f(nw, sw), kind="stable")
            res[m][level].append(F.first_fp_sensitivity(idx, rel, neu))
        if c % 40 == 0:
            print(f"  {level} {c}/{len(qs)} ({time.time()-t0:.0f}s)", flush=True)

print("\n%-28s %9s %9s" % ("ranking", "famSens", "clanSens"))
out = {}
for m in modes:
    f = float(np.mean(res[m]["family"])); c = float(np.mean(res[m]["clan"]))
    out[m] = {"family_sens_1fp": f, "clan_sens_1fp": c}
    print("%-28s %9.3f %9.3f" % (m, f, c))
json.dump(out, open(f"{PC}/files/ablation/rfoldseek_nw_vs_sw.json", "w"), indent=2)
print(f"\nWrote {PC}/files/ablation/rfoldseek_nw_vs_sw.json")
