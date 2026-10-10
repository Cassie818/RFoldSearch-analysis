"""Block bootstrap for the RNAsolo benchmark (Fig. 2 and the retrieval-metrics figure)."""

import json
import os
from collections import defaultdict

import numpy as np

import benchmark as BM

B, SEED = 2000, 123
LEVEL_SEED = {"family": SEED + 1, "clan": SEED + 2}
GRID = np.linspace(0, 1, 60)
BAND_METHODS = ["RFoldSearch", "RFoldSearch-TM", "GTalign", "RTM-align", "US-align", "RMalign",
                "RiboSeek", "BLASTN", "Infernal"]


def read(name):
    return json.load(open(os.path.join(BM.RET, name)))


def write(name, data, indent=2):
    json.dump(data, open(os.path.join(BM.RET, name), "w"), indent=indent)
    print(f"wrote {name}", flush=True)


def blocks(qs, vals, labels):
    out = defaultdict(list)
    for q, v in zip(qs, vals):
        if labels[int(q)]:
            out[labels[int(q)]].append(v)
    return out


def curve_on_grid(vals):
    """Sorted-descending Sens@1FP curve interpolated onto GRID."""
    y = np.sort(np.asarray(vals, dtype=float))[::-1]
    x = np.arange(1, len(y) + 1) / len(y)
    return np.interp(GRID, np.concatenate([[0.0], x]), np.concatenate([[y[0] if len(y) else 0.0], y]))


def band(pq, lab):
    rng = np.random.default_rng(SEED)
    out = {}
    for m in BAND_METHODS:
        out[m] = {}
        for level in ("family", "clan"):
            bd = blocks(pq["_query_ids"][level], pq[m][level]["sens"], lab[level])
            names = list(bd)
            curves = np.asarray([curve_on_grid([v for b in rng.choice(names, size=len(names), replace=True)
                                                for v in bd[b]]) for _ in range(B)])
            out[m][level] = {"grid": GRID.tolist(), "lo": np.quantile(curves, 0.025, axis=0).tolist(),
                             "hi": np.quantile(curves, 0.975, axis=0).tolist()}
    return out


def paired(values, qids, lab, anchor, methods, metric):
    """values: {method: {level: per-query list aligned with qids[level]}}."""
    out = {"_anchor": anchor, "_n_replicates": B, "_seed": SEED, "_metric": metric}
    for level in ("family", "clan"):
        out[level] = {}
        a = blocks(qids[level], values[anchor][level], lab[level])
        names = list(a)
        rng = np.random.default_rng(LEVEL_SEED[level])
        reps = [rng.choice(names, size=len(names), replace=True) for _ in range(B)]
        a_means = np.array([np.mean([v for b in s for v in a[b]]) for s in reps])
        a_obs = float(np.mean([v for b in names for v in a[b]]))
        for m in methods:
            c = blocks(qids[level], values[m][level], lab[level])
            d = a_means - np.array([np.mean([v for b in s for v in c[b]]) for s in reps])
            lo, hi = np.quantile(d, [0.025, 0.975])
            c_obs = float(np.mean([v for b in names for v in c[b]]))
            out[level][m] = {"anchor_mean": a_obs, "method_mean": c_obs, "diff_mean": a_obs - c_obs,
                             "diff_ci_lo": float(lo), "diff_ci_hi": float(hi), "sig": bool(lo > 0 or hi < 0)}
    return out


def structure(pq, lab):
    rng = np.random.default_rng(SEED)
    out = {}
    for th in sorted(pq):
        out[th] = {}
        for level in ("family", "clan"):
            methods = list(pq[th])
            qs = list(pq[th]["RFoldSearch"][level])
            bl = defaultdict(list)
            for q in qs:
                if lab[level][int(q)]:
                    bl[lab[level][int(q)]].append(q)
            names = list(bl)
            boot = {m: [] for m in methods}
            for _ in range(B):
                sq = [q for b in rng.choice(names, size=len(names), replace=True) for q in bl[b]]
                for m in methods:
                    boot[m].append(np.mean([pq[th][m][level][q] for q in sq]))
            boot = {m: np.asarray(v) for m, v in boot.items()}
            out[th][level] = {}
            for m in methods:
                lo, hi = np.quantile(boot[m], [0.025, 0.975])
                sig = None
                if m not in ("RFoldSearch", "RFoldSearch-TM"):
                    dlo, dhi = np.quantile(boot["RFoldSearch"] - boot[m], [0.025, 0.975])
                    sig = bool(dlo > 0 or dhi < 0)
                out[th][level][m] = {"estimate": float(np.mean([pq[th][m][level][q] for q in qs])),
                                     "ci_lower": float(lo), "ci_upper": float(hi),
                                     "sig_vs_rf": sig, "n_blocks": len(names)}
    return out


def main():
    b = BM.load()
    lab = {"family": b.fam, "clan": b.clan}
    others = [m for m in BM.METHODS if m != "RFoldSearch"]
    pq = read("per_query_sens_test.json")
    qids = pq["_query_ids"]
    sens = {m: {lv: pq[m][lv]["sens"] for lv in ("family", "clan")} for m in BM.METHODS}
    write("bootstrap_fig2ab_band.json", band(pq, lab), indent=None)
    write("bootstrap_fig2ab_paired.json", paired(sens, qids, lab, "RFoldSearch", others, "Sens@1FP"))
    write("bootstrap_fig2ab_paired_tm.json",
          paired(sens, qids, lab, "RFoldSearch-TM", ["US-align", "GTalign", "RTM-align", "RFoldSearch"], "Sens@1FP"))
    write("bootstrap_fig2cd_results.json", structure(read("structural_tmscore_per_query_test.json"), lab))
    pr = read("weighted_pr_rfam_test.json")
    ap = {m: {lv: pr[m][lv]["per_query_ap"] for lv in ("family", "clan")} for m in BM.METHODS}
    write("bootstrap_figS12_ap.json", paired(ap, qids, lab, "RFoldSearch", others, "per-query average precision"))


if __name__ == "__main__":
    main()
