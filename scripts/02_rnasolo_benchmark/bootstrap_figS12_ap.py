"""Paired block bootstrap on per-query average precision, for Fig. S12B/C.

Same protocol as bootstrap_fig2ab.py: Rfam families (family level) or clans
(clan level) are resampled with replacement, RFoldseek and each competitor are
scored on the SAME resampled blocks, and the 95% percentile CI of the paired
difference decides significance. Writes
files/production/retrieval/bootstrap_figS12_ap.json.
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import SRC_SEARCH, resolve
_sys.path.insert(0, SRC_SEARCH)
import json
from collections import defaultdict
import numpy as np, torch
from prefilter_rfam_benchmark import attach_labels, load_labels

B, SEED = 2000, 123
LEVEL_SEED = {"family": SEED + 1, "clan": SEED + 2}
ANCHOR = "RFoldseek"


def _blocks(qs, vals, labels):
    bd = defaultdict(list)
    for q, v in zip(qs, vals):
        b = labels[int(q)]
        if b:
            bd[b].append(v)
    return bd


def main():
    RET = resolve("files/production/retrieval")
    wp = json.load(open(f"{RET}/weighted_pr_rfam_test.json"))
    pq = json.load(open(f"{RET}/per_query_sens_test.json"))
    ents = torch.load(resolve("files/models/db.pt"), map_location="cpu", weights_only=False)["entries"]
    lab = {"family": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"))),
           "clan": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"), label_column="clan_acc"))}
    methods = [m for m in wp if not m.startswith("_") and m != ANCHOR]

    out = {"_anchor": ANCHOR, "_n_replicates": B, "_seed": SEED,
           "_metric": "per-query average precision"}
    for level in ("family", "clan"):
        out[level] = {}
        qs = pq["_query_ids"][level]
        a = _blocks(qs, wp[ANCHOR][level]["per_query_ap"], lab[level])
        names = list(a)
        rng = np.random.default_rng(LEVEL_SEED[level])
        reps = [rng.choice(names, size=len(names), replace=True) for _ in range(B)]
        a_means = np.array([np.mean([v for b in s for v in a[b]]) for s in reps])
        a_obs = float(np.mean([v for b in names for v in a[b]]))
        for m in methods:
            c = _blocks(qs, wp[m][level]["per_query_ap"], lab[level])
            c_means = np.array([np.mean([v for b in s for v in c[b]]) for s in reps])
            d = a_means - c_means
            lo, hi = np.percentile(d, [2.5, 97.5])
            out[level][m] = {"anchor_mean": a_obs,
                             "method_mean": float(np.mean([v for b in names for v in c[b]])),
                             "diff_mean": float(d.mean()), "diff_ci_lo": float(lo),
                             "diff_ci_hi": float(hi), "sig_vs_rf": bool(lo > 0 or hi < 0)}
    dst = f"{RET}/bootstrap_figS12_ap.json"
    json.dump(out, open(dst, "w"), indent=2)
    print(f"wrote {dst}")
    for level in ("family", "clan"):
        for m, e in out[level].items():
            print(f"  {level:6s} {m:13s} {e['method_mean']:.3f}  diff={e['diff_mean']:+.4f} "
                  f"CI=[{e['diff_ci_lo']:+.4f},{e['diff_ci_hi']:+.4f}] sig={e['sig_vs_rf']}")


if __name__ == "__main__":
    main()
