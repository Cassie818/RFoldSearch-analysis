#!/usr/bin/env python3
"""Block-bootstrap 95% CI band + paired-difference significance for Fig 2A/B.

Same-family (and same-clan) queries are not independent, so the bootstrap
resamples Rfam *family* (panel A) or *clan* (panel B) blocks with replacement
rather than individual queries.

Input : files/production/retrieval/per_query_sens_test.json, files/models/db.pt,
        Splits/labels.tsv
Output: files/production/retrieval/bootstrap_fig2ab_band.json (CI band)
        files/production/retrieval/bootstrap_fig2ab_paired.json (paired sig)
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import SRC_SEARCH, resolve  # noqa: E402

_sys.path.insert(0, SRC_SEARCH)
import json  # noqa: E402
from collections import defaultdict  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from prefilter_rfam_benchmark import attach_labels, load_labels  # noqa: E402

B = 2000
SEED = 123
# Deterministic per-level seed offsets. Python's built-in hash("family") is
# randomised across processes unless PYTHONHASHSEED is fixed, so we use an
# explicit mapping to keep bootstrap CI bit-for-bit reproducible.
LEVEL_SEED = {"family": SEED + 1, "clan": SEED + 2}
GRID = np.linspace(0, 1, 60)
METHODS = [
    "RFoldseek",
    "RFoldseek-TM",
    "GTalign",
    "RTM-align",
    "US-align",
    "RMalign",
    "RiboSeek",
    "BLASTN",
]


def curve_on_grid(vals):
    """Sorted-descending Sens@1FP curve interpolated onto GRID."""
    y = np.sort(np.asarray(vals, dtype=float))[::-1]
    x = np.arange(1, len(y) + 1) / len(y)
    return np.interp(GRID, np.concatenate([[0.0], x]),
                     np.concatenate([[y[0] if len(y) else 0.0], y]))


def _block_map(qs, sens, labels):
    """Group per-query Sens@1FP by Rfam block. Queries with no block are dropped."""
    bd = defaultdict(list)
    for q, v in zip(qs, sens):
        blk = labels[int(q)]
        if blk:
            bd[blk].append(v)
    return bd


def main():
    data = json.load(open(resolve("files/production/retrieval/per_query_sens_test.json")))
    db = torch.load(resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
    ents = db["entries"]
    lab = {
        "family": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"))),
        "clan": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"),
                                                label_column="clan_acc")),
    }
    rng = np.random.default_rng(SEED)

    # ---------------- 1. per-method CI band on the sorted curve ----------------
    band = {}
    for m in METHODS:
        if m not in data:
            continue
        band[m] = {}
        for level in ("family", "clan"):
            bd = _block_map(data["_query_ids"][level], data[m][level]["sens"], lab[level])
            names = list(bd)
            curves = []
            for _ in range(B):
                samp = rng.choice(names, size=len(names), replace=True)
                curves.append(curve_on_grid([v for b in samp for v in bd[b]]))
            curves = np.asarray(curves)
            band[m][level] = {
                "grid": GRID.tolist(),
                "lo": np.quantile(curves, 0.025, axis=0).tolist(),
                "hi": np.quantile(curves, 0.975, axis=0).tolist(),
            }
    dst_band = resolve("files/production/retrieval/bootstrap_fig2ab_band.json")
    json.dump(band, open(dst_band, "w"))
    print(f"wrote {dst_band}  ({len(band)} methods)")

    # ---------------- 2. paired-difference significance vs RFoldseek ----------
    # Same-block resampling as above, but for each replicate we compute both
    # RFoldseek's and each competitor's mean Sens@1FP on the SAME resampled
    # blocks, and record the paired difference (RFoldseek - competitor). The
    # 95% percentile CI of that per-block difference is the "did the paired
    # advantage survive block-level resampling" test Methods 4.6 describes.
    ANCHOR = "RFoldseek"
    if ANCHOR not in data:
        print(f"{ANCHOR} missing from per_query_sens_test.json; skipping paired-diff", flush=True)
        return
    paired = {"_anchor": ANCHOR, "_n_replicates": B, "_seed": SEED}
    for level in ("family", "clan"):
        paired[level] = {}
        # anchor's per-block per-query Sens
        anchor_bd = _block_map(data["_query_ids"][level], data[ANCHOR][level]["sens"], lab[level])
        names = list(anchor_bd)
        # RNG re-seeded per level so competitors within a level share replicates
        rng2 = np.random.default_rng(LEVEL_SEED[level])
        replicates = [rng2.choice(names, size=len(names), replace=True) for _ in range(B)]
        anchor_means = np.array([
            np.mean([v for b in samp for v in anchor_bd[b]]) for samp in replicates
        ])
        anchor_obs = float(np.mean([v for b in names for v in anchor_bd[b]]))
        for m in METHODS:
            if m == ANCHOR or m not in data:
                continue
            comp_bd = _block_map(data["_query_ids"][level], data[m][level]["sens"], lab[level])
            comp_means = np.array([
                np.mean([v for b in samp for v in comp_bd[b]]) for samp in replicates
            ])
            comp_obs = float(np.mean([v for b in names for v in comp_bd[b]]))
            diffs = anchor_means - comp_means
            lo, hi = float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))
            paired[level][m] = {
                "anchor_mean": anchor_obs,
                "method_mean": comp_obs,
                "diff_mean": anchor_obs - comp_obs,
                "diff_ci_lo": lo,
                "diff_ci_hi": hi,
                "sig_vs_rf": bool(lo > 0.0 or hi < 0.0),
            }
    dst_paired = resolve("files/production/retrieval/bootstrap_fig2ab_paired.json")
    json.dump(paired, open(dst_paired, "w"), indent=2)
    print(f"wrote {dst_paired}")
    for level in ("family", "clan"):
        for m, e in paired[level].items():
            print(f"  {level:6s} {m:12s}  {ANCHOR} {e['anchor_mean']:.3f}  vs  {m} {e['method_mean']:.3f}  "
                  f"diff={e['diff_mean']:+.3f}  95%CI=[{e['diff_ci_lo']:+.3f},{e['diff_ci_hi']:+.3f}]  "
                  f"sig={e['sig_vs_rf']}")


if __name__ == "__main__":
    main()
