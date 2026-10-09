"""Family-disjoint vs PDB-disjoint 5-fold CV (RNA analogue of Foldseek Supp. Fig. 14).

A  family / clan Sens@1FP, macro-averaged over Rfam families (family bars) or
   clans (clan bars); bars are the mean over VQ-VAE training seeds, error bars
   ±1 SD across seeds, dots individual seeds. Δ: per-seed paired difference
   (mean ± SD) and a family/clan-block bootstrap CI on seed-averaged queries.
B  per-family family Sens@1FP (seed-averaged), PDB- vs family-disjoint.
C  per-fold unlabelled chains removed from training, per criterion and in total.

Folds are numbered 1-5 here; the "fold" field in the JSON is 0-based.
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import FIG_DIR, resolve

import json
from collections import defaultdict
import numpy as np, matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SPLITS = [("pdb", "PDB-disjoint"), ("family", "Family-disjoint")]
# both bars are RFoldSearch: Fig. 2 RFoldSearch blue, lighter tint for the family-disjoint split
COL = {"pdb": "#1F9BD8", "family": "#9FD3EE"}
# metric -> block unit used for macro-averaging and bootstrap
BLOCK = {"fam": "family", "clan": "clan_acc"}
NAMES = {"RF00005": "tRNA", "RF00009": "RNaseP_nuc", "RF00504": "Glycine"}
B, SEED = 2000, 123

SUM = json.load(open(resolve("files/ablation/cv_family_disjoint_seed_summary.json")))
SEEDS = SUM["seeds"]


def _load(s, sd):
    sfx = "" if sd == 42 else f"_seed{sd}"
    return json.load(open(resolve(f"files/ablation/cv_{s}_disjoint_5fold{sfx}.json")))


R = {s: _load(s, 42) for s, _ in SPLITS}
# per-query sensitivities averaged over seeds
P = {}
for s, _ in SPLITS:
    runs = [{r["id"]: r for r in _load(s, sd)["per_query"]} for sd in SEEDS]
    P[s] = {}
    for i, r in runs[0].items():
        r = dict(r)
        for key in ("fam", "clan"):
            if r[key] is not None:
                r[key] = float(np.mean([run[i][key] for run in runs]))
        P[s][i] = r
ids = sorted(P["pdb"])
assert set(ids) == set(P["family"])


def grouped(s, key):
    out = defaultdict(list)
    for i in ids:
        v = P[s][i][key]
        if v is not None:
            out[P[s][i][BLOCK[key]]].append(v)
    return out


def boot(vals):
    rng = np.random.default_rng(SEED)
    bs = vals[rng.integers(0, len(vals), (B, len(vals)))].mean(1)
    return vals.mean(), *np.percentile(bs, [2.5, 97.5])


def macro_ci(s, key):
    g = grouped(s, key)
    return boot(np.array([np.mean(g[k]) for k in sorted(g)]))


def paired_diff(key):
    a, b = grouped("family", key), grouped("pdb", key)
    return boot(np.array([np.mean(a[k]) - np.mean(b[k]) for k in sorted(a)]))


fig, axes = plt.subplots(1, 3, figsize=(15, 4.9), gridspec_kw={"width_ratios": [1.1, 1.25, 1.1]})

# A: all four schemes; values from the seed summary
# solid fills only: the control in grey, the family-grouped schemes light and dark blue
BARS = [("pdb", "PDB-disjoint", "#5DB3E2", None), ("pdb-random", "PDB-disjoint (random removal)", "#C3CBD1", None),
        ("family", "Family-disjoint", "#B9DFF3", None), ("homology", "Homology-disjoint", "#3F86B8", None)]
SENS = SUM["sensitivity"]
ax = axes[0]
xs = np.arange(2)
w = 0.2
for k, (s_, lab, col, hatch) in enumerate(BARS):
    m = np.array([SENS[key][s_]["mean"] for key in ("fam", "clan")])
    sd = np.array([SENS[key][s_]["sd"] for key in ("fam", "clan")])
    x = xs + (k - 1.5) * w
    ax.bar(x, m, w, color=col, hatch=hatch, edgecolor="white" if hatch else col, lw=0, label=lab, zorder=2)
    ax.errorbar(x, m, yerr=sd, fmt="none", ecolor="#333333", elinewidth=1, capsize=2.5, zorder=4)
    for xi, key in zip(x, ("fam", "clan")):
        pts = SENS[key][s_]["per_seed"]
        ax.scatter(xi + np.linspace(-0.05, 0.05, len(pts)), pts, s=10, color="white", edgecolor="#333333",
                   lw=0.7, zorder=5)
    for xi, mi, e, key in zip(x, m, sd, ("fam", "clan")):
        top = max(mi + e, max(SENS[key][s_]["per_seed"]))
        ax.text(xi, top + 0.015, f"{mi:.2f}", ha="center", fontsize=7.5, color="#333333")
ax.set_xticks(xs)
ax.set_xticklabels(["Family", "Clan"], fontsize=11)
ax.set_ylabel("Sens@1FP (macro-averaged)", fontsize=11)
ax.set_ylim(0, 0.7)
ax.legend(frameon=False, fontsize=8.5, loc="upper right")
ax.spines[["top", "right"]].set_visible(False)

# B
ax = axes[1]
a, b = grouped("family", "fam"), grouped("pdb", "fam")
F = sorted(a)
xa = np.array([np.mean(b[f]) for f in F])
ya = np.array([np.mean(a[f]) for f in F])
n = np.array([len(a[f]) for f in F])
size = lambda q: 14 + 3.5 * q
jit = np.random.default_rng(0).uniform(-0.015, 0.015, (2, len(F)))
ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", zorder=1)
ax.scatter(xa + jit[0], ya + jit[1], s=size(n), color=COL["pdb"], alpha=0.5,
           edgecolor="#10618C", lw=0.6, zorder=2)
for f, x, y in zip(F, xa, ya):
    if f in NAMES:
        off = {"RF00005": (30, -22), "RF00009": (-25, -38), "RF00504": (-30, 26)}[f]
        ax.annotate(NAMES[f], (x, y), xytext=off, textcoords="offset points", fontsize=8.5,
                    ha="right" if off[0] < 0 else "left", va="center",
                    arrowprops=dict(arrowstyle="-", color="#555555", lw=0.7, shrinkA=1, shrinkB=4))
handles = [Line2D([], [], ls="", marker="o", ms=np.sqrt(size(q)), mfc=COL["pdb"], alpha=0.5,
                  mec="#10618C", label=f"{q}") for q in (2, 10, 30, 104)]
ax.legend(handles=handles, title="Queries", frameon=False, fontsize=8.5, title_fontsize=8.5,
          loc="lower right", labelspacing=1.3, borderpad=0.8, handletextpad=1.2)
ax.set_xlim(-0.04, 1.04)
ax.set_ylim(-0.04, 1.04)
ax.set_aspect("equal")
ax.set_xlabel("PDB-disjoint family Sens@1FP", fontsize=11)
ax.set_ylabel("Family-disjoint family Sens@1FP", fontsize=11)
ax.spines[["top", "right"]].set_visible(False)

# C: training chains removed per fold, family-disjoint (unlabelled only) vs homology-disjoint
ax = axes[2]
H = _load("homology", 42)["folds"]
fo = R["family"]["folds"]
x = np.arange(len(fo))
bw = 0.34
ax.bar(x - bw / 2, [f["n_unlabelled_removed"] for f in fo], bw, color="#B9DFF3", label="Family-disjoint (unlabelled)",
       zorder=2)
hu = np.array([f["n_unlabelled_removed"] for f in H])
hl = np.array([f["n_labelled_removed"] for f in H])
ax.bar(x + bw / 2, hu, bw, color="#3F86B8", label="Homology-disjoint (unlabelled)", zorder=2)
ax.bar(x + bw / 2, hl, bw, bottom=hu, color="#8DB9D8", label="Homology-disjoint (labelled)", zorder=2)
ax.legend(frameon=False, fontsize=8.5, loc="upper left")
ax.set_xticks(x)
ax.set_xticklabels([f"Fold {f['fold'] + 1}" for f in fo], fontsize=10)
ax.set_ylabel("Chains removed from training", fontsize=11)
ax.set_ylim(0, (hu + hl).max() * 1.3)
ax.spines[["top", "right"]].set_visible(False)

for ax, L in zip(axes, "ABC"):
    ax.text(-0.14, 1.04, L, transform=ax.transAxes, fontsize=15, fontweight="bold")
fig.tight_layout()
out = _os.path.join(FIG_DIR, "supp_family_disjoint_cv.png")
fig.savefig(out, dpi=300, bbox_inches="tight")
print(out)
for key in ("fam", "clan"):
    print(key, "pdb", macro_ci("pdb", key), "family", macro_ci("family", key), "diff", paired_diff(key))
