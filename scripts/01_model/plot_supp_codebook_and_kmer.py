"""Combined S4: codebook-size and k-mer-length sweeps."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, resolve
_os.chdir(DATA_ROOT)
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt

C1, C2, C3 = "#5B7C99", "#B08D5C", "#4C8C69"


def draw_sensitivity(ax, xs, fam, fam_sd, clan, clan_sd, xlabel, panel):
    ax.plot(xs, fam, "o-", color=C1, linewidth=2, markersize=7,
            markerfacecolor=C1, markeredgecolor="white", markeredgewidth=1,
            label="Family Sens@1FP", zorder=3)
    ax.plot(xs, clan, "s-", color=C2, linewidth=2, markersize=7,
            markerfacecolor=C2, markeredgecolor="white", markeredgewidth=1,
            label="Clan Sens@1FP", zorder=3)
    cap = 0.08 if len(xs) == 4 else 2.0
    for values, errors, color in ((fam, fam_sd, C1), (clan, clan_sd, C2)):
        for x, value, error in zip(xs, values, errors):
            ax.vlines(x, value, value + error, colors="#333333", linewidth=1.0, alpha=0.9)
            ax.hlines(value + error, x - cap, x + cap, colors="#333333", linewidth=1.0, alpha=0.9)
            ax.text(x, value + error + 0.012, f"{value:.3f}", ha="center",
                    fontsize=8, color=color)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Sens@1FP", fontsize=11)
    ax.set_xticks(xs)
    ax.set_ylim(0, 0.60)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.09, 1.04, panel, transform=ax.transAxes, fontsize=16,
            fontweight="bold", ha="right")


def draw_postings(ax, xs, post, post_sd, xlabel, panel, ylim):
    ax.plot(xs, post, "o-", color=C3, linewidth=2, markersize=7,
            markerfacecolor=C3, markeredgecolor="white", markeredgewidth=1,
            label="Postings/query", zorder=3)
    cap = 0.08 if len(xs) == 4 else 2.0
    for x, value, error in zip(xs, post, post_sd):
        ax.vlines(x, value, value + error, colors="#333333", linewidth=1.0, alpha=0.9)
        ax.hlines(value + error, x - cap, x + cap, colors="#333333", linewidth=1.0, alpha=0.9)
        ax.annotate(f"{value / 1000:.1f}k", (x, value + error), xytext=(0, 7),
                    textcoords="offset points", ha="center", fontsize=8, color=C3)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Postings per query", fontsize=11)
    ax.set_xticks(xs)
    ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.09, 1.04, panel, transform=ax.transAxes, fontsize=16,
            fontweight="bold", ha="right")


# Codebook-size sweep, cosine matrix and k=4.
codebook_K = np.array([20, 40, 60, 80, 100])
codebook_results = json.load(open(resolve("files/ablation/cv_s4_codebook_size_5fold.json")))
code_rows = [codebook_results[f"K{K}"] for K in codebook_K]
code_fam = np.array([r["fam_sens_mean"] for r in code_rows])
code_fam_sd = np.array([r["fam_sens_std"] for r in code_rows])
code_clan = np.array([r["clan_sens_mean"] for r in code_rows])
code_clan_sd = np.array([r["clan_sens_std"] for r in code_rows])
code_post = np.array([r["postings_per_query_mean"] for r in code_rows])
code_post_sd = np.array([r["postings_per_query_std"] for r in code_rows])

# k-mer-length sweep at fixed K=80.
kmer_k = np.array([3, 4, 5, 6])
kmer_results = json.load(open(resolve("files/ablation/cv_s4_kmer_length_5fold.json")))
kmer_rows = [kmer_results[f"k{k}"] for k in kmer_k]
kmer_fam = np.array([r["fam_sens_mean"] for r in kmer_rows])
kmer_fam_sd = np.array([r["fam_sens_std"] for r in kmer_rows])
kmer_clan = np.array([r["clan_sens_mean"] for r in kmer_rows])
kmer_clan_sd = np.array([r["clan_sens_std"] for r in kmer_rows])
kmer_post = np.array([r["postings_per_query_mean"] for r in kmer_rows])
kmer_post_sd = np.array([r["postings_per_query_std"] for r in kmer_rows])

fig, axes = plt.subplots(2, 2, figsize=(13, 9.0))
draw_sensitivity(axes[0, 0], codebook_K, code_fam, code_fam_sd,
                 code_clan, code_clan_sd, "Codebook size $K$", "A")
draw_postings(axes[0, 1], codebook_K, code_post, code_post_sd,
              "Codebook size $K$", "B", (9e3, 5e5))
draw_sensitivity(axes[1, 0], kmer_k, kmer_fam, kmer_fam_sd,
                 kmer_clan, kmer_clan_sd, "k-mer length $k$", "C")
draw_postings(axes[1, 1], kmer_k, kmer_post, kmer_post_sd,
              "k-mer length $k$", "D", (8e2, 1.5e5))

h1, l1 = axes[0, 0].get_legend_handles_labels()
h2, l2 = axes[0, 1].get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, frameon=False,
           fontsize=12, bbox_to_anchor=(0.5, -0.005), handlelength=1.8,
           columnspacing=2.2)
fig.tight_layout(rect=(0, 0.055, 1, 1), h_pad=2.8, w_pad=2.8)
out = Path(FIG_DIR) / "supp_codebook_size.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
