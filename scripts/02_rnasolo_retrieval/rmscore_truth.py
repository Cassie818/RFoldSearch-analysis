"""Fig. 2E/F: retrieval under the RMscore truth (symmetric RMscore >= a threshold quantile-matched to
the TM-score thresholds of Fig. 2C/D)."""
import json
import os
from collections import defaultdict

import numpy as np

import benchmark as BM

RET = BM.RET
THRESHOLDS = BM.THRESHOLDS
B, SEED = 2000, 123
METHODS = ["RFoldseek", "RFoldseek-TM", "GTalign", "RTM-align", "US-align", "RMalign",
           "RiboSeek", "RiboSeek RS-80", "BLASTN", "Infernal"]                   # Fig. 2 order


def rmscore_matrix(n):
    rm = np.full((n, n), np.nan, dtype=np.float32)
    for line in open(os.path.join(RET, "rmalign_bidirectional.cache.jsonl")):
        r = json.loads(line)
        if "rm_i" in r:
            rm[r["i"], r["j"]] = rm[r["j"], r["i"]] = 0.5 * (r["rm_i"] + r["rm_j"])
    return rm


def main():
    b = BM.load()
    n, tm, fam, clan = b.n, b.tm, b.fam, b.clan
    rm = rmscore_matrix(n)
    cov_rm = (~np.isnan(rm)).sum(1)
    queries = [q for q in b.struct_q if cov_rm[q] >= n - 6]
    print(f"queries with complete US-align and RMalign rows: {len(queries)} of {len(b.struct_q)}", flush=True)

    # quantile-matched RMscore thresholds over the evaluated rows
    mask = np.zeros((n, n), bool)
    mask[queries] = True
    np.fill_diagonal(mask, False)
    both = mask & ~np.isnan(rm) & ~np.isnan(tm)
    rthr = {}
    for t in THRESHOLDS:
        frac = float(np.mean(tm[both] >= t))
        rthr[t] = float(np.quantile(rm[both], 1 - frac))
        print(f"  TM >= {t}: {frac:.4f} of pairs -> RMscore >= {rthr[t]:.3f}", flush=True)

    ranks = {m: BM.load_rankings(m, b) for m in METHODS}
    out = {"queries": len(queries), "rmscore_thresholds": {str(t): rthr[t] for t in THRESHOLDS}, "thresholds": {}}
    for t in THRESHOLDS:
        per = {m: {"family": {}, "clan": {}} for m in METHODS}
        for q in queries:
            other = np.arange(n) != q
            pos = (~np.isnan(rm[q])) & (rm[q] >= rthr[t]) & other
            neu = np.isnan(rm[q]) & other
            levels = {"family": (pos, neu)}
            if fam[q]:
                levels["clan"] = (pos & (fam != fam[q]), neu | ((fam == fam[q]) & other))
            for level, (P, N) in levels.items():
                if P.any():
                    for m in METHODS:
                        per[m][level][q] = float(np.mean([BM.sens1fp(r, P, N) for r in ranks[m][q]]))
        out["thresholds"][str(t)] = bootstrap(per, fam, clan)
        print(f"RMscore >= {rthr[t]:.3f}: " + "  ".join(
            f"{m} {out['thresholds'][str(t)]['family'][m]['estimate']:.3f}" for m in METHODS), flush=True)
    json.dump(out, open(os.path.join(RET, "rmscore_truth.json"), "w"), indent=1)


def bootstrap(per, fam, clan):
    res = {}
    rng = np.random.default_rng(SEED)
    for level, labels in (("family", fam), ("clan", clan)):
        qs = sorted(per["RFoldseek"][level])
        blocks = defaultdict(list)
        for q in qs:
            if labels[q]:
                blocks[labels[q]].append(q)
        names = list(blocks)
        reps = [rng.choice(names, size=len(names), replace=True) for _ in range(B)]
        res[level] = {}
        rf = per["RFoldseek"][level]
        for m in METHODS:
            v = per[m][level]
            est = float(np.mean([v[q] for q in qs]))
            boots = np.array([np.mean([v[q] for b in s for q in blocks[b]]) for s in reps])
            diffs = np.array([np.mean([rf[q] - v[q] for b in s for q in blocks[b]]) for s in reps])
            lo, hi = np.percentile(boots, [2.5, 97.5])
            dlo, dhi = np.percentile(diffs, [2.5, 97.5])
            res[level][m] = {"estimate": est, "ci_lower": float(lo), "ci_upper": float(hi),
                             "sig_vs_rf": None if m == "RFoldseek" else bool(dlo > 0 or dhi < 0),
                             "n_queries": len(qs), "n_blocks": len(names)}
    return res


if __name__ == "__main__":
    main()
