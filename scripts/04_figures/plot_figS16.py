"""Fig S16: quantiser control. The identical 17-D production descriptor is
tokenised either by k-means (K in {10,20,40,80}) or by the deployed VQ-VAE
(K=80); all alphabets are searched with the same deployed pipeline, each with
its own Structure-BLOSUM. Family- and clan-level Sens@1FP on the test split."""
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
C_KM = "#E07A5F"   # k-means (terracotta)
C_VQ = "#1F9BD8"   # deployed VQ-VAE (RFoldseek blue, as in Fig. 2)
N_PERM = 500_000


def load():
    a = json.load(open(resolve("files/ablation/kmeans_quantiser_control.json")))
    b = json.load(open(resolve("files/ablation/kmeans_quantiser_control_k10k40.json")))
    s = {k: v for k, v in {**a, **b}.items() if k != "_provenance"}
    pa = json.load(open(resolve("files/ablation/kmeans_quantiser_control_per_query.json")))
    pb = json.load(open(resolve("files/ablation/kmeans_quantiser_control_k10k40_per_query.json")))
    p = {k: v for k, v in {**pa, **pb}.items() if k != "_provenance"}
    return s, p


CTRL_BOOT = json.load(open(resolve("files/production/retrieval/bootstrap_controls.json")))


def annotation(K, level):
    """Paired block-bootstrap difference from the VQ-VAE: the effect size with
    its interval, marked when the interval excludes zero. Clan-level intervals
    rest on eight Rfam blocks, so a missing mark is not evidence of no
    difference."""
    e = CTRL_BOOT["quantiser_control"][f"KMeans_K{K}"][level]
    star = "*" if e["sig"] else ""
    return (f"{e['diff_mean']:+.3f}{star}\n[{e['diff_ci_lo']:+.3f}, {e['diff_ci_hi']:+.3f}]",
            e["sig"])


def main():
    summ, _ = load()
    fig, axes = plt.subplots(2, 1, figsize=(4.6, 7.6))
    for ax, level, key, title in [
        (axes[0], "family", "family_sens_1fp", "Family"),
        (axes[1], "clan", "clan_sens_1fp", "Clan"),
    ]:
        km = [summ[f"KMeans_K{K}"][key] for K in KS]
        vq = summ["VQ-VAE_K80_deployed"][key]
        ax.axhline(vq, color=C_VQ, linestyle="-", linewidth=2, zorder=2,
                   label="VQ-VAE")
        ax.annotate(f"{vq:.3f}", (0.995, vq), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 5), ha="right",
                    fontsize=9, color=C_VQ, fontweight="bold")
        ax.plot(KS, km, "o-", color=C_KM, linewidth=2, markersize=8, zorder=3,
                markerfacecolor=C_KM, markeredgecolor="white", markeredgewidth=1.2,
                label="K-means")
        for K, v in zip(KS, km):
            txt, is_sig = annotation(K, level)
            ax.annotate(txt, (K, v), textcoords="offset points", xytext=(0, -32),
                        ha="center", fontsize=7.0, color="#333333", linespacing=1.15)
            ax.annotate(f"{v:.3f}", (K, v), textcoords="offset points", xytext=(0, 10),
                        ha="center", fontsize=9, color="#555555")
        ax.set_xscale("log")
        ax.set_xticks(KS)
        ax.set_xticklabels([str(K) for K in KS])
        ax.minorticks_off()
        ax.set_xlabel("Codebook size $K$", fontsize=12)
        ax.set_ylabel("Sens@1FP", fontsize=12)
        ax.set_title(title, fontsize=13)
        lo, hi = min(km + [vq]), max(km + [vq])
        pad = (hi - lo) * 0.75 + 0.012
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(7.5, 115)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax, lab in zip(axes, "AB"):
        ax.text(-0.10, 1.06, lab, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    h, l = axes[0].get_legend_handles_labels()
    order = [1, 0] if "means" in l[1] else [0, 1]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="lower center",
               ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, -0.005),
               handlelength=2.6, columnspacing=2.2)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out = Path(FIG_DIR) / "S16.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
