"""Fig S18: two controls that Fig. 2E/F leaves out.

(A,B) Scoring-regime control -- RFoldseek and RiboSeek alphabets under one
      matched pipeline, scored with their native matrices, a shared identity
      matrix, and a self-BLOSUM learned for each alphabet. The gap closes under
      identity scoring, so exact token agreement alone does not explain it.
(C,D) The Fig. 2E/F attribution recomputed in the reverse order, changing the
      alphabet before the pipeline and matrix.
"""
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

RF, RB = "RFoldseek_tokens_same_pipeline", "RiboSeek_sa20_same_pipeline"
REGIMES = [("", "Native", "native"), ("_identity", "Identity", "identity"),
           ("_self_blosum", "Self-BLOSUM", "self_blosum")]
C_RF, C_RB = "#4FB3E3", "#EC9BD3"          # lightened method colours
C_STEP = {"+ 3Dr alphabet": "#B5651D", "+ RFoldseek pipeline & matrix": "#C9A227"}
BOOT = json.load(open(resolve("files/production/retrieval/bootstrap_controls.json")))
ATTR = json.load(open(resolve("files/ablation/attribution_paths.json")))


def regime_panel(ax, key, title):
    x = np.arange(len(REGIMES)); w = 0.36
    vals = {}
    for who, tag in (("RFoldseek", RF), ("RiboSeek", RB)):
        vals[who] = [json.load(open(resolve(
            f"files/ablation/riboseek_alphabet_same_pipeline_control{sfx}.json")))[tag][key]
            for sfx, _, _ in REGIMES]
    for who, off, col in (("RFoldseek", -w / 2, C_RF), ("RiboSeek", w / 2, C_RB)):
        ax.bar(x + off, vals[who], w, color=col, edgecolor="#666666", linewidth=0.6,
               label=who, zorder=3)
        for xi, v in zip(x + off, vals[who]):
            ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=7.5,
                    color="#1F9BD8" if who == "RFoldseek" else "#E377C2")
    lvl = "family" if key.startswith("family") else "clan"
    for i, (_, _, rk) in enumerate(REGIMES):
        e = BOOT["alphabet_control"][rk][lvl]
        top = max(vals["RFoldseek"][i], vals["RiboSeek"][i]) + 0.045
        ax.plot([x[i] - w / 2, x[i] - w / 2, x[i] + w / 2, x[i] + w / 2],
                [top, top + 0.012, top + 0.012, top], lw=0.9, color="#777777", zorder=4)
        star = "*" if e["sig"] else ""
        ax.text(x[i], top + 0.016, f"{e['diff_mean']:+.3f}{star}", ha="center",
                va="bottom", fontsize=8, color="#333333")
    ax.set_xticks(x, [lbl for _, lbl, _ in REGIMES], fontsize=9)
    ax.set_xlabel("Scoring regime", fontsize=11)
    ax.set_ylabel("Sens@1FP", fontsize=11)
    ax.set_ylim(0, max(max(v) for v in vals.values()) * 1.42)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def reverse_panel(ax, level, title):
    lv = ATTR["levels"][level]; m, rev = lv["means"], lv["reverse"]
    keys = ["A_ribo_published", "E_rev_alphabet", "D_rfoldseek"]
    steps = ["+ 3Dr alphabet", "+ RFoldseek pipeline & matrix"]
    vals = [m[k] for k in keys]
    ax.bar(0, vals[0], 0.6, color=C_RB, edgecolor="#b05a94", linewidth=0.6, zorder=3)
    ax.text(0, vals[0] + 0.006, f"{vals[0]:.3f}", ha="center", va="bottom", fontsize=8,
            color="#E377C2")
    for n, lbl in enumerate(steps, start=1):
        lo, hi = min(vals[n - 1], vals[n]), max(vals[n - 1], vals[n])
        e = rev[lbl]
        ax.bar(n, hi - lo, 0.6, bottom=lo, color=C_STEP[lbl], edgecolor="#555555",
               linewidth=0.6, zorder=3)
        ax.errorbar(n, hi, yerr=[[max(e["diff"] - e["lo"], 0)], [max(e["hi"] - e["diff"], 0)]],
                    fmt="none", ecolor="#555555", elinewidth=0.9, capsize=2.5, zorder=4)
        ax.text(n, hi + max(e["hi"] - e["diff"], 0) + max(vals) * 0.03,
                f"{e['diff']:+.3f}{'*' if e['sig'] else ''}", ha="center", va="bottom",
                fontsize=8, color=C_STEP[lbl])
        ax.plot([n - 1 + 0.3, n - 0.3], [vals[n - 1], vals[n - 1]], color="#999999",
                lw=0.8, ls=(0, (3, 2)), zorder=2)
    ax.bar(3, vals[-1], 0.6, color=C_RF, edgecolor="#15709c", linewidth=0.6, zorder=3)
    ax.text(3, vals[-1] + 0.006, f"{vals[-1]:.3f}", ha="center", va="bottom", fontsize=8,
            color="#1F9BD8")
    ax.plot([2 + 0.3, 3 - 0.3], [vals[-1], vals[-1]], color="#999999", lw=0.8,
            ls=(0, (3, 2)), zorder=2)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["RiboSeek", "+ 3Dr\nalphabet", "+ RFoldseek\npipeline & matrix",
                        "RFoldseek"], fontsize=7.6)
    ax.set_ylabel("Sens@1FP", fontsize=11)
    top = max(max(vals), max(m[k] + max(rev[l]["hi"] - rev[l]["diff"], 0)
                             for k, l in zip(keys[1:], steps)))
    ax.set_ylim(0, top * 1.30)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=9)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig, ax = plt.subplots(2, 2, figsize=(10.6, 8.4))
    regime_panel(ax[0, 0], "family_sens_1fp", "Family")
    regime_panel(ax[0, 1], "clan_sens_1fp", "Clan")
    reverse_panel(ax[1, 0], "family", "Family")
    reverse_panel(ax[1, 1], "clan", "Clan")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.004))
    for a, lab in zip(ax.flat, "ABCD"):
        a.text(-0.12, 1.05, lab, transform=a.transAxes, fontsize=15, fontweight="bold",
               va="top")
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = Path(FIG_DIR) / "S18.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
