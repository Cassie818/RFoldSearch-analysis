"""Search-time breakdown: where a default RFoldSearch query spends
its time on NuFold, by database size (A) and by query length at full size (B).
Default mode = prefilter + score_candidates (Smith-Waterman + C4 superposition), one core."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(HERE, "breakdown_nu.json")))
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Nimbus Sans", "DejaVu Sans"],
                     "font.size": 10, "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
PARTS = [("expand", "k-mer expansion", "#BBBBBB"), ("kernel", "prefilter: posting scan", "#0072B2"),
         ("select", "prefilter: candidate selection", "#56B4E9"),
         ("tm_a", "alignment + structure score (500 candidates)", "#E69F00")]
fig, (a, b) = plt.subplots(1, 2, figsize=(180 / 25.4, 85 / 25.4), constrained_layout=True)

def bars(ax, labels, groups):
    x = np.arange(len(labels)); bottom = np.zeros(len(labels))
    for k, name, col in PARTS:
        v = np.array([np.mean([r[k] for r in g]) for g in groups])
        ax.bar(x, v, 0.65, bottom=bottom, color=col, label=name); bottom += v
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Mean time per query (s)")
    for xi, tot in zip(x, bottom):
        ax.text(xi, tot, f"{tot:.2f}", ha="center", va="bottom", fontsize=10)

sizes = sorted(d["sizes"], key=int)
bars(a, [f"{int(n) // 1000}k" for n in sizes], [d["sizes"][n]["per_query"] for n in sizes])
a.set_xlabel("Database size (structures)")
rows = d["sizes"][max(sizes, key=int)]["per_query"]
L = np.array([r["tokens"] for r in rows])
bins = [(20, 50), (50, 100), (100, 200), (200, 400)]
bars(b, [f"{lo}-{hi - 1}\n(n={((L >= lo) & (L < hi)).sum()})" for lo, hi in bins],
     [[r for r, l in zip(rows, L) if lo <= l < hi] for lo, hi in bins])
b.set_xlabel("Query length (nt), 531,888 structures")
for ax, t in ((a, "A"), (b, "B")):
    ax.text(-0.17, 1.02, t, transform=ax.transAxes, fontsize=12, fontweight="bold")
fig.legend(*a.get_legend_handles_labels(), loc="outside lower center", ncol=2, frameon=False)
fig.set_size_inches(180 / 25.4, 100 / 25.4)
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"breakdown.{ext}"), dpi=300)
