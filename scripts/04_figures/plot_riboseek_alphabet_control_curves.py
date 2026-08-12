"""RiboSeek alphabet-vs-RFoldseek matched-pipeline control, sorted-sensitivity curves.

Foldseek-style AUROC1 curves (per-query Sens@1FP sorted descending vs the
fraction of queries), laid out as a 2x3 grid:

    rows    = ground truth level   (Family, Clan)
    columns = scoring regime       (Native, Identity, Self-BLOSUM)

Each cell overlays RFoldseek (80-token) vs RiboSeek (sa20) under the SAME search
pipeline. Reading across a row: the RFoldseek-over-RiboSeek gap is wide under
native scoring, collapses under identity scoring, and re-opens under matched
self-BLOSUM - the sensitivity gain comes from learned substitution geometry, not
the pipeline or exact symbol matching. Area between the two curves is shaded to
make the gap legible; the mean Sens@1FP and paired one-sided sign-flip p-value
are annotated per cell.

Reads files/ablation/riboseek_alphabet_same_pipeline_control{,_identity,
_self_blosum}_per_query.json. Writes
figures/riboseek_alphabet_control_curves.{png,pdf}.
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
LEVELS = [("family", "Family"), ("clan", "Clan")]


def load_pq(suffix):
    return json.load(open(resolve(
        f"files/ablation/riboseek_alphabet_same_pipeline_control{suffix}_per_query.json")))


def sorted_curve(rows):
    y = np.sort(np.array([r["sens"] for r in rows]))[::-1]
    x = np.arange(1, len(y) + 1) / len(y)
    return x, y


def signflip_p(pq, level, n=2_000_000, seed=0):
    a = {r["query"]: r["sens"] for r in pq[RF][level]}
    b = {r["query"]: r["sens"] for r in pq[RB][level]}
    q = sorted(set(a) & set(b))
    d = np.array([a[k] - b[k] for k in q])
    obs = d.mean()
    rng = np.random.default_rng(seed)
    perm = (rng.choice([1.0, -1.0], size=(n, len(d))) * d).mean(1)
    return obs, (np.sum(perm >= obs) + 1) / (n + 1)


def p_text(p):
    if p < 1e-4:
        return "p < 10$^{-4}$"
    return f"p = {p:.1e}".replace("e-0", "e$-$").replace("e-", "e$-$") + ""


def cell(ax, pq, level, key_curve, show_ylabel, col_title, row_title):
    means = {}
    for who, name in (("RF", RF), ("RB", RB)):
        x, y = sorted_curve(pq[name][level])
        means[who] = y.mean()
    # shade the gap between the two sorted curves
    xr, yr = sorted_curve(pq[RF][level])
    xb, yb = sorted_curve(pq[RB][level])
    grid = np.linspace(0, 1, 200)
    fr = np.interp(grid, xr, yr)
    fb = np.interp(grid, xb, yb)
    ax.fill_between(grid, fb, fr, where=fr >= fb, color=COLOR["RF"], alpha=0.10,
                    linewidth=0, zorder=1)
    for who, name in (("RB", RB), ("RF", RF)):
        x, y = sorted_curve(pq[name][level])
        step = max(1, len(x) // 14)
        ax.plot(x, y, "-", color=COLOR[who], linewidth=2,
                marker="o" if who == "RF" else "v", markevery=step, markersize=5,
                markeredgecolor="white", markeredgewidth=0.8,
                label=LABEL[who], zorder=3)
    obs, p = signflip_p(pq, level)
    ax.text(0.97, 0.94, f"$\\Delta$ = {means['RF'] - means['RB']:+.2f}\n{p_text(p)}",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.2, color="#333333")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    if show_ylabel:
        ax.set_ylabel(f"{row_title}\nSens up to 1st FP", fontsize=11)
    ax.set_xlabel("Fraction of queries", fontsize=10)
    if col_title:
        ax.set_title(col_title, fontsize=12, pad=6)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    pqs = {s: load_pq(s) for s, _ in REGIMES}
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6), sharex=True, sharey=True)
    for r, (level, row_title) in enumerate(LEVELS):
        for c, (suffix, col_title) in enumerate(REGIMES):
            cell(axes[r, c], pqs[suffix], level, None,
                 show_ylabel=(c == 0),
                 col_title=col_title if r == 0 else "",
                 row_title=row_title)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False,
               fontsize=11, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out_png = Path(FIG_DIR) / "riboseek_alphabet_control_curves.png"
    out_pdf = Path(FIG_DIR) / "riboseek_alphabet_control_curves.pdf"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Wrote {out_png}\nWrote {out_pdf}")


if __name__ == "__main__":
    main()
