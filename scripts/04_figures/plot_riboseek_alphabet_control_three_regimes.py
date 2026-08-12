"""RiboSeek alphabet-vs-RFoldseek matched-pipeline control, three scoring regimes.

One figure that tells the ablation's story in two panels (Family, Clan). Within
each panel, three scoring regimes are shown side by side:

    Native      - each alphabet keeps its own substitution matrix
    Identity    - both alphabets scored by a shared identity matrix
    Self-BLOSUM - each alphabet learns its own Structure-BLOSUM from the SAME
                  training-partition US-align (TM>=0.5) pairs

The pipeline (similar-k-mer prefilter, top-200 pool, affine-gap Smith-Waterman,
gap penalties, query set) is held fixed; only the alphabet and its scoring
matrix change. The gap between RFoldseek and RiboSeek is large under native
scoring, collapses under identity scoring, and re-opens under matched
self-BLOSUM - i.e. the sensitivity gain is driven by learned substitution
geometry, not by the pipeline or by exact symbol matching alone.

Reads files/ablation/riboseek_alphabet_same_pipeline_control{,_identity,_self_blosum}
(+ _per_query for sign-flip significance). Writes
figures/riboseek_alphabet_control_three_regimes.{png,pdf}.
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import FIG_DIR, resolve  # noqa: E402

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "Arial"
matplotlib.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RF = "RFoldseek_tokens_same_pipeline"
RB = "RiboSeek_sa20_same_pipeline"
COLOR = {"RF": "#1F9BD8", "RB": "#E377C2"}
LABEL = {"RF": "RFoldseek (80-token)", "RB": "RiboSeek (sa20)"}
REGIMES = [("", "Native"), ("_identity", "Identity"), ("_self_blosum", "Self-BLOSUM")]


def load(suffix):
    return json.load(open(resolve(
        f"files/ablation/riboseek_alphabet_same_pipeline_control{suffix}.json")))


def load_pq(suffix):
    p = Path(resolve(
        f"files/ablation/riboseek_alphabet_same_pipeline_control{suffix}_per_query.json"))
    return json.load(open(p)) if p.exists() else None


def signflip_p(pq, level, n=2_000_000, seed=0):
    """One-sided (RFoldseek > RiboSeek) paired sign-flip permutation p-value."""
    if pq is None:
        return None
    a = {r["query"]: r["sens"] for r in pq[RF][level]}
    b = {r["query"]: r["sens"] for r in pq[RB][level]}
    q = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in q])
    if not len(d):
        return None
    obs = d.mean()
    rng = np.random.default_rng(seed)
    perm = (rng.choice([1.0, -1.0], size=(n, len(d))) * d).mean(1)
    return (np.sum(perm >= obs) + 1) / (n + 1)


def stars(p):
    if p is None:
        return ""
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "n.s."


def panel(ax, level, key, title, label):
    data = {s: load(s) for s, _ in REGIMES}
    pqs = {s: signflip_p(load_pq(s), level) for s, _ in REGIMES}
    x = np.arange(len(REGIMES))
    width = 0.36
    vals = {"RF": [], "RB": []}
    for suffix, _ in REGIMES:
        vals["RF"].append(data[suffix][RF][key])
        vals["RB"].append(data[suffix][RB][key])
    for who, off in (("RF", -width / 2), ("RB", width / 2)):
        bars = ax.bar(x + off, vals[who], width, color=COLOR[who], edgecolor="none",
                      label=LABEL[who], zorder=3)
        for b, v in zip(bars, vals[who]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.2, color="#333333")
    # significance bracket over each regime (RFoldseek vs RiboSeek)
    for i, (suffix, _) in enumerate(REGIMES):
        s = stars(pqs[suffix])
        top = max(vals["RF"][i], vals["RB"][i])
        y = top + 0.055
        ax.plot([x[i] - width / 2, x[i] - width / 2, x[i] + width / 2, x[i] + width / 2],
                [y, y + 0.014, y + 0.014, y], lw=0.9, color="#777777", zorder=4)
        ax.text(x[i], y + 0.012, s, ha="center", va="bottom",
                fontsize=9 if s != "n.s." else 7.5,
                color="#333333", fontweight="bold" if s not in ("", "n.s.") else "normal")
    ax.set_xticks(x, [lab for _, lab in REGIMES])
    ax.set_ylabel("Sens@1FP", fontsize=12)
    ax.set_ylim(0, 0.78)
    ax.set_title(title, fontsize=13, pad=8)
    ax.tick_params(axis="both", labelsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(-0.14, 1.02, label, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="bottom", ha="right")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    panel(axes[0], "family", "family_sens_1fp", "Family", "A")
    panel(axes[1], "clan", "clan_sens_1fp", "Clan", "B")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_png = Path(FIG_DIR) / "riboseek_alphabet_control_three_regimes.png"
    out_pdf = Path(FIG_DIR) / "riboseek_alphabet_control_three_regimes.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}\nWrote {out_pdf}")


if __name__ == "__main__":
    main()
