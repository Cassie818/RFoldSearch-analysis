"""Supplementary figure for Fig. 2E/F: the identity background, the negative controls and the
clan-level effect of neutralising sequence-detectable positives, overall and per clan."""

import os

from sequence_divergent_panels import (COLOR, DATA, FIG_DIR, S, clan_panel, filter_handles, interval_panel,
                                       json, np, plt)
from sequence_divergent_rescore import background_identity, load

CONTROLS = ["RFoldSearch", "RFoldSearch-TM", "Length", "Composition", "Shuffled"]
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


def main():
    fam = json.load(open(os.path.join(DATA, "family_perquery.json")))
    cl = json.load(open(os.path.join(DATA, "clan_perquery.json")))
    fig = plt.figure(figsize=(12.6, 9.6))
    grid = fig.add_gridspec(2, 1, height_ratios=(1, 1), hspace=0.34, left=0.13, right=0.99, top=0.95, bottom=0.11)
    top = grid[0].subgridspec(1, 2, wspace=0.34)
    bottom = grid[1].subgridspec(1, 2, width_ratios=(1, 2.1), wspace=0.26)
    ax_a, ax_b = fig.add_subplot(top[0]), fig.add_subplot(top[1])
    ax_c, ax_d = fig.add_subplot(bottom[0]), fig.add_subplot(bottom[1])
    background_panel(ax_a)
    # B, C and D share one fill code: open, all positives; filled, sequence-detectable
    # positives (BLASTN or nhmmer, E <= 10) neutralised
    f = "B/N E<=10"
    # B: RFoldSearch against rankings that ignore structure (negative controls)
    interval_panel(ax_b, fam["values"]["none"], fam["values"][f], fam["blocks"], CONTROLS)
    # C: clan-level counterpart of Fig. 2F over all clans; D: the same split by clan, with the
    # methods' hues and shapes keyed by C
    interval_panel(ax_c, cl["values"]["none"], cl["values"][f], cl["clan"],
                   ["RFoldSearch", "RFoldSearch-TM", "US-align", "Infernal", "nhmmer", "BLASTN",
                    "RS-20", "RS-80"], level="Clan")
    clan_panel(ax_d, cl, method_legend=False, filter_legend=False)
    fig.legend(handles=filter_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.01), ncol=2,
               frameon=False, fontsize=11, handletextpad=0.4, columnspacing=2.4)
    for ax, x, y, lab in ((ax_a, -0.12, 1.08, "A"), (ax_b, -0.48, 1.08, "B"), (ax_c, -0.52, 1.08, "C"),
                          (ax_d, -0.09, 1.08, "D")):
        ax.text(x, y, lab, transform=ax.transAxes, fontsize=18, fontweight="bold", va="bottom")
    out = os.path.join(FIG_DIR, "supp_sequence_divergent.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(out)


if __name__ == "__main__":
    main()
