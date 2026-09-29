"""Paired block bootstrap anchored on RFoldseek-TM, reusing the Fig 2A/B protocol."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.expanduser("~/Desktop/RFoldseek-paper-code"))
from paths import SRC_SEARCH, resolve
_sys.path.insert(0, SRC_SEARCH)
import json
from collections import defaultdict
import numpy as np, torch
from prefilter_rfam_benchmark import attach_labels, load_labels

B, SEED = 2000, 123
LEVEL_SEED = {"family": SEED + 1, "clan": SEED + 2}
ANCHOR = "RFoldseek-TM"
COMPARE = ["US-align", "GTalign", "RTM-align", "RFoldseek"]

def _block_map(qs, sens, labels):
    bd = defaultdict(list)
    for q, v in zip(qs, sens):
        blk = labels[int(q)]
        if blk:
            bd[blk].append(v)
    return bd

data = json.load(open(resolve("files/production/retrieval/per_query_sens_test.json")))
db = torch.load(resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
ents = db["entries"]
lab = {"family": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"))),
       "clan": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"), label_column="clan_acc"))}

out = {"_anchor": ANCHOR, "_n_replicates": B, "_seed": SEED}
for level in ("family", "clan"):
    out[level] = {}
    a_bd = _block_map(data["_query_ids"][level], data[ANCHOR][level]["sens"], lab[level])
    names = list(a_bd)
    rng = np.random.default_rng(LEVEL_SEED[level])
    reps = [rng.choice(names, size=len(names), replace=True) for _ in range(B)]
    a_means = np.array([np.mean([v for b in s for v in a_bd[b]]) for s in reps])
    a_obs = float(np.mean([v for b in names for v in a_bd[b]]))
    print(f"--- {level}: {ANCHOR} = {a_obs:.3f} ---")
    for m in COMPARE:
        c_bd = _block_map(data["_query_ids"][level], data[m][level]["sens"], lab[level])
        c_means = np.array([np.mean([v for b in s for v in c_bd[b]]) for s in reps])
        d = a_means - c_means
        lo, hi = np.percentile(d, [2.5, 97.5])
        sig = bool(lo > 0 or hi < 0)
        c_obs = float(np.mean([v for b in names for v in c_bd[b]]))
        out[level][m] = {"anchor_mean": a_obs, "method_mean": c_obs,
                         "diff_mean": float(d.mean()), "diff_ci_lo": float(lo),
                         "diff_ci_hi": float(hi), "sig": sig}
        print(f"  vs {m:12s} {c_obs:.3f}   diff={d.mean():+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  sig={sig}")
p = resolve("files/production/retrieval/bootstrap_fig2ab_paired_tm.json")
json.dump(out, open(p, "w"), indent=2)
print(f"\nwrote {p}")
