"""Supplementary figure: calibration of the deployed RFoldseek E-values.

Observed number of false-positive target clusters per query against the reported
E-value, for the two deployed calibrations. In both, targets are grouped into
80%-identity sequence clusters, a hit is a false positive when its symmetric
US-align TM-score to the query is below 0.45, and the calibration map is
evaluated leave-one-Rfam-family-out, so each query is scored with a map fitted
without its family. Initial E0 is the fitted extreme-value E-value before the
empirical map; E_cal is the deployed, calibrated E-value. Shaded bands are 95%
family-block bootstrap intervals of E_cal.

  A  experimental database (RNAsolo, 885 chains): build_experimental_empirical_evalue.py,
     summarize_experimental_evalue_lofo.py --dense
  B  predicted database (NuFold): query-conditional extreme-value parameters from
     length and token entropy (fit_entropy_conditional_evalue.py) followed by the
     empirical map (build_entropy_deployment_candidates.py,
     adjudicate_entropy_conditional_candidates.py, summarize_entropy_conditional_evalue.py,
     fit_entropy_deployment_map.py)

Writes figures/supp_evalue_calibration.png.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import FIG_DIR, resolve  # noqa: E402

EXPERIMENTAL = "files/production/evalue_experimental/lofo_validation/dense_all_query_background_summary.json"
PREDICTED = "files/production/evalue_predicted/entropy_conditional_empirical_lofo_summary_E100.json"
INITIAL, CALIBRATED = "#E69F00", "#1F9BD8"


def experimental():
    d = json.load(open(resolve(EXPERIMENTAL)))
    m = d["models"]
    return dict(levels=d["levels"], n=(d["n_queries"], d["n_families"]),
                raw=m["raw_clustermax"]["observed_clusters_per_query"],
                cal=m["empirical_lofo"]["observed_clusters_per_query"],
                ci=m["empirical_lofo"]["family_block_95ci"])


def predicted():
    d = json.load(open(resolve(PREDICTED)))
    return dict(levels=d["levels"], n=(d["n_queries"], d["n_families"]),
                raw=d["raw"]["observed"], cal=d["calibrated"]["observed"],
                ci=d["calibrated"]["family_block_95ci"])


def panel(ax, data, title):
    lv = np.asarray(data["levels"])
    lo, hi = (np.asarray(x) for x in zip(*data["ci"]))
    floor = 4e-3
    ax.fill_between(lv, np.maximum(lo, floor), np.maximum(hi, floor), color=CALIBRATED, alpha=0.15, lw=0,
                    zorder=1)
    ax.plot([1e-2, 1e1], [1e-2, 1e1], ls="--", color="#777777", lw=1.2, zorder=2)
    for key, color, label in (("raw", INITIAL, r"Initial $E_0$"), ("cal", CALIBRATED, r"Calibrated $E_{\mathrm{cal}}$")):
        v = np.asarray(data[key], dtype=float)
        keep = v > 0
        ax.plot(lv[keep], v[keep], "o", ms=5.5, color=color, mec="white", mew=0.6, label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(7e-3, 1.4e1)
    ax.set_ylim(floor, 1e2)
    ax.set_xlabel("Reported E-value", fontsize=12)
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    panel(axes[0], experimental(), "Experimental database")
    panel(axes[1], predicted(), "Predicted database")
    # observed E-value: false-positive target clusters per query at that reported E-value
    axes[0].set_ylabel("Observed E-value", fontsize=12)
    for ax, letter in zip(axes, "AB"):
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.insert(0, plt.Line2D([], [], ls="--", color="#777777", lw=1.2))
    labels.insert(0, "Perfect calibration")
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = os.path.join(FIG_DIR, "supp_evalue_calibration.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
