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
C_KM = "#B08D5C"   # k-means
C_VQ = "#1F9BD8"   # deployed VQ-VAE
N_PERM = 500_000


def load():
    a = json.load(open(resolve("files/ablation/kmeans_quantiser_control.json")))
    b = json.load(open(resolve("files/ablation/kmeans_quantiser_control_k10k40.json")))
    s = {k: v for k, v in {**a, **b}.items() if k != "_provenance"}
    pa = json.load(open(resolve("files/ablation/kmeans_quantiser_control_per_query.json")))
    pb = json.load(open(resolve("files/ablation/kmeans_quantiser_control_k10k40_per_query.json")))
    p = {k: v for k, v in {**pa, **pb}.items() if k != "_provenance"}
    return s, p


def stars(pv):
    return "***" if pv < 1e-3 else "**" if pv < 1e-2 else "*" if pv < 5e-2 else "n.s."


def signflip(pq, other, level, rng):
    a = {x["query"]: x["sens"] for x in pq["VQ-VAE_K80_deployed"][level]}
    b = {x["query"]: x["sens"] for x in pq[other][level]}
    d = np.array([a[q] - b[q] for q in a], float)
    perm = (rng.choice([-1, 1], size=(N_PERM, len(d))) * np.abs(d)).mean(1)
    return (np.sum(perm >= d.mean()) + 1) / (N_PERM + 1)


def main():
    summ, pq = load()
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, level, key, title in [
        (axes[0], "family", "family_sens_1fp", "Family"),
        (axes[1], "clan", "clan_sens_1fp", "Clan"),
    ]:
        km = [summ[f"KMeans_K{K}"][key] for K in KS]
        vq = summ["VQ-VAE_K80_deployed"][key]
        ax.axhline(vq, color=C_VQ, linestyle="--", linewidth=2, zorder=2,
                   label=f"VQ-VAE, $K$=80 (deployed) = {vq:.3f}")
        ax.plot(KS, km, "o-", color=C_KM, linewidth=2, markersize=8, zorder=3,
                markerfacecolor=C_KM, markeredgecolor="white", markeredgewidth=1.2,
                label="$k$-means on the same descriptor")
        for K, v in zip(KS, km):
            s = stars(signflip(pq, f"KMeans_K{K}", level, rng))
            ax.annotate(s, (K, v), textcoords="offset points", xytext=(0, -17),
                        ha="center", fontsize=10, color="#333333")
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
        pad = (hi - lo) * 0.55 + 0.01
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlim(8, 100)
        ax.grid(axis="y", alpha=0.25, linewidth=0.7)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.legend(frameon=False, fontsize=9.5, loc="lower left")
    for ax, lab in zip(axes, "AB"):
        ax.text(-0.10, 1.06, lab, transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    fig.tight_layout()
    out = Path(FIG_DIR) / "S16.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
