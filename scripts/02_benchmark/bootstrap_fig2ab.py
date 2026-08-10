#!/usr/bin/env python3
"""Block-bootstrap 95% CI band for the Fig 2A/B AUROC1 (sorted Sens@1FP) curves.

Same-family (and same-clan) queries are not independent, so the bootstrap
resamples Rfam *family* (panel A) or *clan* (panel B) blocks with replacement
rather than individual queries. On each resample the sorted Sens@1FP curve is
recomputed and interpolated onto a common fraction grid; the per-fraction 2.5/
97.5 percentiles give the CI band. plot_fig2.py shades the RFoldseek band.

Input : files/production/retrieval/per_query_sens_test.json, files/models/db.pt,
        Splits/labels.tsv
Output: files/production/retrieval/bootstrap_fig2ab_band.json
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
GRID = np.linspace(0, 1, 60)
METHODS = ["RFoldseek", "GTalign", "RTM-align", "US-align", "RMalign", "RiboSeek", "BLASTN"]


def curve_on_grid(vals):
    """Sorted-descending Sens@1FP curve interpolated onto GRID."""
    y = np.sort(np.asarray(vals, dtype=float))[::-1]
    x = np.arange(1, len(y) + 1) / len(y)
    return np.interp(GRID, np.concatenate([[0.0], x]),
                     np.concatenate([[y[0] if len(y) else 0.0], y]))


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

    out = {}
    for m in METHODS:
        if m not in data:
            continue
        out[m] = {}
        for level in ("family", "clan"):
            bd = defaultdict(list)
            for q, v in zip(data["_query_ids"][level], data[m][level]["sens"]):
                blk = lab[level][int(q)]
                if blk:
                    bd[blk].append(v)
            names = list(bd)
            curves = []
            for _ in range(B):
                samp = rng.choice(names, size=len(names), replace=True)
                curves.append(curve_on_grid([v for b in samp for v in bd[b]]))
            curves = np.asarray(curves)
            out[m][level] = {
                "grid": GRID.tolist(),
                "lo": np.quantile(curves, 0.025, axis=0).tolist(),
                "hi": np.quantile(curves, 0.975, axis=0).tolist(),
            }

    dst = resolve("files/production/retrieval/bootstrap_fig2ab_band.json")
    json.dump(out, open(dst, "w"))
    print(f"wrote {dst}  ({len(out)} methods)")


if __name__ == "__main__":
    main()
