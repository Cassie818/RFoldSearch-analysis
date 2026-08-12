"""Plot the RiboSeek alphabet same-pipeline control."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]

COLORS = {
    "RFoldseek_tokens_same_pipeline": "#1F9BD8",
    "RiboSeek_sa20_same_pipeline": "#E377C2",
}
LABELS = {
    "RFoldseek_tokens_same_pipeline": "RFoldseek tokens",
    "RiboSeek_sa20_same_pipeline": "RiboSeek sa20",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["native", "identity", "self_blosum"],
        default="native",
        help="Which matrix-control result JSON to plot.",
    )
    args = parser.parse_args()
    suffix = "" if args.mode == "native" else f"_{args.mode}"
    in_json = REPO / "files" / "ablation" / f"riboseek_alphabet_same_pipeline_control{suffix}.json"
    out_png = REPO / "figures" / f"riboseek_alphabet_same_pipeline_control{suffix}.png"
    out_pdf = REPO / "figures" / f"riboseek_alphabet_same_pipeline_control{suffix}.pdf"

    raw = json.load(open(in_json))
    data = {key: value for key, value in raw.items() if not key.startswith("_")}
    methods = ["RFoldseek_tokens_same_pipeline", "RiboSeek_sa20_same_pipeline"]
    metrics = [
        ("family_sens_1fp", "Family\nSens@1FP"),
        ("clan_sens_1fp", "Clan\nSens@1FP"),
        ("family_recall1", "Family\nRecall@1"),
        ("clan_recall1", "Clan\nRecall@1"),
    ]

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.2, 2.65), constrained_layout=True)
    x = np.arange(len(metrics), dtype=float)
    width = 0.34
    for offset, method in [(-width / 2, methods[0]), (width / 2, methods[1])]:
        values = [data[method][metric] for metric, _label in metrics]
        bars = ax.bar(x + offset, values, width=width, color=COLORS[method], label=LABELS[method])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f"{value:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _metric, label in metrics])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("score")
    mode_label = {
        "native": "native substitution matrices",
        "identity": "identity scoring",
        "self_blosum": "alphabet-specific self-BLOSUM",
    }[args.mode]
    ax.set_title("RiboSeek alphabet under the RFoldseek search pipeline", loc="left", fontweight="bold", fontsize=10)
    ax.text(
        0.02,
        0.95,
        f"{mode_label}; k-mer prefilter, SW rerank,\ngaps, and query set are held fixed.",
        transform=ax.transAxes,
        va="top",
        fontsize=6.6,
        color="#55514B",
    )
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2, fontsize=7)
    ax.grid(axis="y", color="#E8E4DE", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out_png, dpi=350, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(out_png)
    print(out_pdf)


if __name__ == "__main__":
    main()
