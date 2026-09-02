"""Fig S17: calibration of RFoldseek E-values. Nominal E-value against the
observed number of false positives per query, for four null models fitted and
assessed on identical, disjoint sets of database chains."""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, resolve

_os.chdir(DATA_ROOT)
from pathlib import Path
import json, numpy as np, matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt

SERIES = [
    ("iid-background-200", "Background composition",    "#B3413A", "v"),
    ("shuffle-200",       "Block shuffle",              "#E07A5F", "o"),
    ("random-200",        "Random sample",              "#8C8C8C", "^"),
    ("random-200-trim10", "Random sample, censored",    "#1F9BD8", "s"),
    ("unrelated-200",     "Known-unrelated (idealised)", "#5B7C99", "D"),
]


def main():
    d = json.load(open(resolve("files/ablation/evalue_calibration.json")))
    obs = d["observed_fp_per_query_full_db_scale"]
    grid = sorted((float(k) for k in obs[SERIES[0][0]]), key=float)

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    lo, hi = min(grid) / 3, max(grid) * 3
    ax.plot([lo, hi], [lo, hi], color="#333333", linestyle=(0, (4, 3)), linewidth=1.4,
            zorder=1, label="Perfect calibration")
    for key, lab, c, mk in SERIES:
        y = np.array([obs[key][str(e)] for e in grid], float)
        m = y > 0
        ax.plot(np.array(grid)[m], y[m], marker=mk, color=c, linewidth=2, markersize=7,
                markerfacecolor=c, markeredgecolor="white", markeredgewidth=1.1,
                zorder=3, label=lab)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("Nominal E-value", fontsize=12)
    ax.set_ylabel("Observed false positives per query", fontsize=12)
    ax.annotate("anti-conservative\n(too significant)", xy=(0.03, 0.93), xycoords="axes fraction",
                fontsize=9, color="#8a4b3a", ha="left", va="top")
    ax.annotate("conservative", xy=(0.97, 0.07), xycoords="axes fraction",
                fontsize=9, color="#33607a", ha="right", va="bottom")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left", bbox_to_anchor=(0.0, 0.86))
    fig.tight_layout()
    out = Path(FIG_DIR) / "S17.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
