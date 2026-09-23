#!/usr/bin/env python3
"""Family/clan block-bootstrap 95% CI and paired significance for Fig 2C/D.

Reads the per-query structure-defined Sens@1FP written by
eval_structural_sens_fig2cd.py (structural_tmscore_per_query_test.json), then
for each TM-score threshold resamples Rfam family/clan blocks with replacement.
For each method it reports the 95% CI of its Sens@1FP, and a paired-difference
significance flag against RFoldseek (``sig_vs_rf``: the RFoldseek-minus-method
difference is recomputed on the *same* resampled blocks, and flagged when its
95% CI excludes zero). plot_fig2.py draws the CI as error bars and marks
significant methods with an asterisk.

Input : files/production/retrieval/structural_tmscore_per_query_test.json,
        files/models/db.pt, Splits/labels.tsv
Output: files/production/retrieval/bootstrap_fig2cd_results.json
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import SRC_SEARCH, resolve  # noqa: E402

_sys.path.insert(0, SRC_SEARCH)
import json  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from prefilter_rfam_benchmark import attach_labels, load_labels  # noqa: E402

B = 2000
SEED = 123


def main():
    pq = json.load(open(resolve(
        "files/production/retrieval/structural_tmscore_per_query_test.json")))
    db = torch.load(resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
    ents = db["entries"]
    lab = {
        "family": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"))),
        "clan": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"),
                                                label_column="clan_acc")),
    }
    rng = np.random.default_rng(SEED)

    out = {}
    for th in sorted(pq):
        out[th] = {"family": {}, "clan": {}}
        for level in ("family", "clan"):
            methods = [m for m in pq[th] if pq[th][m][level]]
            if not methods:
                continue
            qset = list(pq[th][methods[0]][level])
            blocks = {}
            for q in qset:
                blk = lab[level][int(q)]
                if blk:
                    blocks.setdefault(blk, []).append(q)
            names = list(blocks)
            boot = {m: [] for m in methods}
            for _ in range(B):
                sq = [q for b in rng.choice(names, size=len(names), replace=True)
                      for q in blocks[b]]
                for m in methods:
                    boot[m].append(np.mean([pq[th][m][level][q] for q in sq]))
            boot = {m: np.asarray(v) for m, v in boot.items()}
            for m in methods:
                lo, hi = np.quantile(boot[m], [0.025, 0.975])
                sig = None
                if m != "RFoldseek" and "RFoldseek" in boot:
                    d = boot["RFoldseek"] - boot[m]
                    dlo, dhi = np.quantile(d, [0.025, 0.975])
                    sig = bool(dlo > 0 or dhi < 0)
                out[th][level][m] = {
                    "estimate": float(np.mean([pq[th][m][level][q] for q in qset])),
                    "ci_lower": float(lo),
                    "ci_upper": float(hi),
                    "sig_vs_rf": sig,
                    "n_blocks": len(names),
                }

    dst = resolve("files/production/retrieval/bootstrap_fig2cd_results.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
