"""Supplementary figure for Fig. 3: the identity background and the negative controls."""

import os

from plot_fig3 import COLOR, DATA, FIG_DIR, LABEL, MARK, S, boot_methods, json, marker_size, np, plt
from sequence_divergent_rescore import background_identity, load

CONTROLS = ["RFoldSearch", "Length", "Composition", "Shuffled"]
GREY = "#666666"


def background_panel(ax):
    _, bg = background_identity()
    p50, p99 = np.quantile(bg, [0.5, 0.99])
    R, ident = load()
    fam = dict(zip(R["tags"], R["fam"]))
    pos = [x for q in R["queries"] for t, x in ident[q].items() if fam[t] == fam[q]]
    bins = np.arange(0, 1.0001, 0.02)
    ax.hist(bg, bins=bins, density=True, color="#BBBBBB", label="Unrelated pairs (background)", zorder=2)
    ax.hist(pos, bins=bins, density=True, histtype="step", lw=1.8, color=COLOR["RFoldSearch"],
            label="Same-family query–target pairs", zorder=3)
    top = ax.get_ylim()[1]
    for x, lab in ((p50, "50th pct"), (p99, "99th pct"), (0.80, "80% cap")):
        ax.axvline(x, color=GREY, lw=1.0, ls="--", zorder=4)
        ax.text(x + 0.006, top * 0.98, f"{lab} ({100 * x:.0f}%)", fontsize=8.5, color=GREY, va="top",
                rotation=90, zorder=5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.0))
    ax.set_xlim(0.4, 1.0)
    S.style_axes(ax, xlabel="Pairwise sequence identity", ylabel="Density")
    ax.legend(frameon=False, fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), labelcolor=GREY)


def control_panel(ax, V, blocks):
    """Absolute Sens@1FP of RFoldSearch and the negative controls, all families and tRNA excluded."""
    sets = (("all families", lambda q: True, -0.14), ("tRNA excluded", lambda q: blocks[q] != "RF00005", 0.14))
    for lab, keep, dy in sets:
        sub = {m: {q: v for q, v in V[m].items() if keep(q)} for m in CONTROLS}
        est, _ = boot_methods(sub, CONTROLS, blocks)
        filled = lab == "all families"
        for i, m in enumerate(CONTROLS):
            mean, lo, hi = est[m]
            ax.hlines(i + dy, lo, hi, color=COLOR[m], lw=2.0, zorder=2)
            ax.plot(mean, i + dy, MARK[m], ms=marker_size(m) + 1, color=COLOR[m],
                    mfc=COLOR[m] if filled else "white", mec="white" if filled else COLOR[m],
                    mew=0.8 if filled else 1.6, zorder=3, clip_on=False)
            ax.text(hi + 0.015, i + dy, f"{mean:.2f}", va="center", fontsize=9,
                    color=S.TEXT["RFoldSearch n500"] if m == "RFoldSearch" else GREY)
    ax.axhline(0.5, color="#DDDDDD", lw=1.2, zorder=0)
    ax.set_yticks(range(len(CONTROLS)), [LABEL[m] for m in CONTROLS], fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(len(CONTROLS) - 0.45, -0.55)
    ax.set_xlim(0, 0.8)
    ax.set_xticks(np.arange(0, 0.81, 0.2))
    ax.grid(axis="x", color="#E6E6E6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    S.style_axes(ax, xlabel="Family-level Sens@1FP")
    ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", color="#555555", ms=8, label="All families"),
                       plt.Line2D([], [], marker="o", ls="", color="#555555", mfc="white", mew=1.6, ms=8,
                                  label="tRNA excluded")],
              frameon=False, fontsize=9.5, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              labelcolor=GREY)


def main():
    fam = json.load(open(os.path.join(DATA, "family_perquery.json")))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={"wspace": 0.42})
    background_panel(ax_a)
    control_panel(ax_b, fam["values"]["B/N E<=10"], fam["blocks"])
    for ax, x, lab in ((ax_a, -0.12, "A"), (ax_b, -0.36, "B")):
        ax.text(x, 1.08, lab, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom")
    out = os.path.join(FIG_DIR, "supp_sequence_divergent.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
