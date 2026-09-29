"""Supplementary quantiser-control figure: quantiser control. The identical 17-D production descriptor is
tokenised either by k-means (K in {10,20,40,80}) or by the deployed VQ-VAE
(K=80); all alphabets are searched with the same deployed pipeline, each with
its own Structure-BLOSUM. Family- and clan-level Sens@1FP under the Rfam
annotation (A,B) and the structure-defined ground truth at TM >= 0.45 (C,D),
and accuracy against prefilter selectivity (E)."""
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

KS = [10, 20, 40, 80]
TAU = "0.45"
C_KM = "#C56B57"   # k-means (muted terracotta)
C_VQ = "#1F9BD8"   # deployed VQ-VAE (RFoldseek blue, as in Fig. 2)


def load():
    a = json.load(open(resolve("files/ablation/kmeans_quantiser_control.json")))
    b = json.load(open(resolve("files/ablation/kmeans_quantiser_control_k10k40.json")))
    return {k: v for k, v in {**a, **b}.items() if k != "_provenance"}


def load_selectivity():
    a = json.load(open(resolve("files/ablation/prefilter_selectivity.json")))
    b = json.load(open(resolve("files/ablation/prefilter_selectivity_k10k40.json")))
    return {**a, **b}


CTRL_BOOT = json.load(open(resolve("files/production/retrieval/bootstrap_controls.json")))
STRUCT = json.load(open(resolve("files/ablation/quantiser_structural.json")))


def ann_rfam(K, level):
    e = CTRL_BOOT["quantiser_control"][f"KMeans_K{K}"][level]
    return f"{e['diff_mean']:+.3f}{'*' if e['sig'] else ''}"


def ann_struct(K, level):
    e = STRUCT["thresholds"][TAU][level]["vs_vqvae"][f"KMeans_K{K}"]
    return f"{e['diff_mean']:+.3f}{'*' if e['sig'] else ''}"


def draw_kv(ax, km, vq, ann_fn, level, title):
    """k-means Sens@1FP against codebook size, with the VQ-VAE as a reference
    line and the paired difference (VQ - k-means, * when significant) per point."""
    ax.axhline(vq, color=C_VQ, linestyle="-", linewidth=1.4, zorder=2, label="VQ-VAE")
    ax.annotate(f"{vq:.3f}", (0.995, vq), xycoords=("axes fraction", "data"),
                textcoords="offset points", xytext=(0, 5), ha="right",
                fontsize=9, color=C_VQ, fontweight="bold")
    ax.plot(KS, km, "o-", color=C_KM, linewidth=1.4, markersize=6, zorder=3,
            markerfacecolor=C_KM, markeredgecolor="white", markeredgewidth=1.0,
            label="K-means")
    for K, v in zip(KS, km):
        ax.annotate(ann_fn(K, level), (K, v), textcoords="offset points", xytext=(0, -20),
                    ha="center", fontsize=8.0, color="#333333")
        ax.annotate(f"{v:.3f}", (K, v), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=9, color="#555555")
    ax.set_xscale("log")
    ax.set_xticks(KS)
    ax.set_xticklabels([str(K) for K in KS])
    ax.minorticks_off()
    ax.set_xlabel("Codebook size $K$", fontsize=10.5)
    ax.set_ylabel("Sens@1FP", fontsize=10.5)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    lo, hi = min(km + [vq]), max(km + [vq])
    pad = (hi - lo) * 0.42 + 0.008
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(6.0, 145)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def draw_sel(ax, km_x, km_y, vq_x, vq_y, ylab, title):
    """Accuracy against prefilter selectivity (postings per query, log axis);
    selectivity is ground-truth independent, so only the y-axis changes."""
    ax.plot(km_x, km_y, "o-", color=C_KM, linewidth=1.4, markersize=6, zorder=3,
            markerfacecolor=C_KM, markeredgecolor="white", markeredgewidth=1.0)
    # K=80 sits at similar x as VQ-VAE (both K=80 codebooks): push its label
    # DOWN so it does not collide with the VQ-VAE marker or its text label
    # (which is offset to the right of the VQ marker).
    _off = {10: (0, -15), 20: (0, 11), 40: (0, 11), 80: (0, -15)}
    for K, x, y in zip(KS, km_x, km_y):
        dx, dy = _off[K]
        ax.annotate(f"$K$={K}", (x, y), textcoords="offset points",
                    xytext=(dx, dy), ha="center", fontsize=8.5, color="#555555")
    ax.plot([vq_x], [vq_y], "o", color=C_VQ, markersize=9, zorder=4,
            markerfacecolor=C_VQ, markeredgecolor="white", markeredgewidth=1.2)
    ax.annotate("VQ-VAE", (vq_x, vq_y), textcoords="offset points",
                xytext=(10, 0), ha="left", va="center", fontsize=8.5,
                color=C_VQ, fontweight="bold")
    ax.set_xscale("log")
    ax.set_xlabel("Postings per query", fontsize=10.5)
    ax.set_ylabel(ylab, fontsize=10.5)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    ax.set_xlim(6.5e3, 2.4e6)
    ys = list(km_y) + [vq_y]
    ax.set_ylim(min(ys) - 0.018, max(ys) + 0.028)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def main():
    summ = load()
    sel = load_selectivity()
    st = STRUCT["thresholds"][TAU]

    fig = plt.figure(figsize=(9.2, 9.4))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.52, wspace=0.30)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    axE, axF = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])

    # A,B: Rfam annotation
    draw_kv(axA, [summ[f"KMeans_K{K}"]["family_sens_1fp"] for K in KS],
            summ["VQ-VAE_K80_deployed"]["family_sens_1fp"], ann_rfam, "family", "Family (Rfam)")
    draw_kv(axB, [summ[f"KMeans_K{K}"]["clan_sens_1fp"] for K in KS],
            summ["VQ-VAE_K80_deployed"]["clan_sens_1fp"], ann_rfam, "clan", "Clan (Rfam)")
    # C,D: structure-defined ground truth at TM >= 0.45
    draw_kv(axC, [st["family"]["means"][f"KMeans_K{K}"] for K in KS],
            st["family"]["means"]["VQ-VAE_K80_deployed"], ann_struct, "family",
            f"Family (TM $\\geq$ {TAU})")
    draw_kv(axD, [st["clan"]["means"][f"KMeans_K{K}"] for K in KS],
            st["clan"]["means"]["VQ-VAE_K80_deployed"], ann_struct, "clan",
            f"Clan (TM $\\geq$ {TAU})")

    # E,F: accuracy against prefilter selectivity, Rfam (E) and structure-defined (F)
    km_x = [sel[f"KMeans_K{K}"]["postings_per_query"] for K in KS]
    vq_x = sel["VQ-VAE_K80"]["postings_per_query"]
    draw_sel(axE, km_x, [summ[f"KMeans_K{K}"]["family_sens_1fp"] for K in KS],
             vq_x, summ["VQ-VAE_K80_deployed"]["family_sens_1fp"],
             "Family Sens@1FP", "Selectivity (Rfam)")
    draw_sel(axF, km_x, [st["family"]["means"][f"KMeans_K{K}"] for K in KS],
             vq_x, st["family"]["means"]["VQ-VAE_K80_deployed"],
             "Family Sens@1FP", f"Selectivity (TM $\\geq$ {TAU})")

    for ax, lab in zip([axA, axB, axC, axD, axE, axF], "ABCDEF"):
        ax.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(-30, 6),
                    textcoords="offset points", fontsize=15, fontweight="bold",
                    ha="left", va="bottom")
    h, l = axA.get_legend_handles_labels()
    order = [1, 0] if "means" in l[1] else [0, 1]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
               ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.004),
               handlelength=2.6, columnspacing=2.2)

    out = Path(FIG_DIR) / "supp_quantiser_control.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
