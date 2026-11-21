"""AUROC (area under the full ROC curve) on the RNAsolo benchmark for every Fig. 2 method and truth (Supplementary)."""
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import benchmark as BM  # noqa: E402
from rmscore_truth import rmscore_matrix  # noqa: E402

B, SEED = 2000, 123
LEVEL_SEED = {"family": SEED + 1, "clan": SEED + 2}
METHODS = list(BM.METHODS)
TABLE_TAU = "0.45"


def roc_auc(ranking, pos, neg):
    """AUROC of one ranking; unreturned targets tie below every returned one. None if undefined."""
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return None
    r = np.asarray(ranking, np.int64)
    if len(r):
        _, first = np.unique(r, return_index=True)
        r = r[np.sort(first)]                      # keep first occurrence, in rank order
    seen = np.zeros(len(pos), bool)
    seen[r] = True
    neg_unret = int((neg & ~seen).sum())
    pos_unret = int((pos & ~seen).sum())
    is_pos, is_neg = pos[r], neg[r]
    neg_after = np.cumsum(is_neg[::-1])[::-1] - is_neg     # negatives strictly below each rank
    wins = float((neg_after[is_pos] + neg_unret).sum()) + 0.5 * pos_unret * neg_unret
    return wins / (n_pos * n_neg)


def per_query(runs, q, pos, neu, n):
    other = np.arange(n) != q
    neg = other & ~pos if neu is None else other & ~pos & ~neu
    vals = [roc_auc(r, pos & other, neg) for r in runs.get(q, [np.zeros(0, np.int64)])]
    return None if vals[0] is None else float(np.mean(vals))


def truths(b):
    """{panel: (level, [(query, pos, neu)])} for the Fig. 2 truths."""
    out = {}
    for level, qs in (("family", b.fam_q), ("clan", b.clan_q)):
        out[f"rfam|{level}"] = (level, [(q, *BM.rfam_masks(b, q, level)) for q in qs])
    for tau in BM.THRESHOLDS:
        for level in ("family", "clan"):
            rows = []
            for q in b.struct_q:
                pos, neu = BM.structure_masks(b, q, tau, level)
                if pos.any():
                    rows.append((q, pos, neu))
            out[f"tm{tau}|{level}"] = (level, rows)
    rm = rmscore_matrix(b.n)
    rmthr = json.load(open(os.path.join(BM.RET, "rmscore_truth.json")))["rmscore_thresholds"]
    cov_rm = (~np.isnan(rm)).sum(1)
    rq = [q for q in b.struct_q if cov_rm[q] >= b.n - 6]
    for tau in BM.THRESHOLDS:
        thr = rmthr[str(tau)]
        fam_rows, clan_rows = [], []
        for q in rq:
            other = np.arange(b.n) != q
            pos = (~np.isnan(rm[q])) & (rm[q] >= thr) & other
            neu = np.isnan(rm[q]) & other
            if pos.any():
                fam_rows.append((q, pos, neu))
            if b.fam[q]:
                pc = pos & (b.fam != b.fam[q])
                if pc.any():
                    clan_rows.append((q, pc, neu | ((b.fam == b.fam[q]) & other)))
        out[f"rm{tau}|family"] = ("family", fam_rows)
        out[f"rm{tau}|clan"] = ("clan", clan_rows)
    return out


def bootstrap(per, level, labels):
    """Block bootstrap of the mean (per method) and of RFoldSearch - method, shared replicates."""
    qs = sorted(per["RFoldSearch"])
    blocks = defaultdict(list)
    for q in qs:
        if labels[q]:
            blocks[labels[q]].append(q)
    names = list(blocks)
    rng = np.random.default_rng(LEVEL_SEED[level])
    reps = [rng.choice(names, size=len(names), replace=True) for _ in range(B)]
    rf = per["RFoldSearch"]
    res = {}
    for m in METHODS:
        v = per[m]
        boots = np.array([np.mean([v[q] for k in s for q in blocks[k]]) for s in reps])
        diffs = np.array([np.mean([rf[q] - v[q] for k in s for q in blocks[k]]) for s in reps])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        dlo, dhi = np.percentile(diffs, [2.5, 97.5])
        res[m] = {"estimate": float(np.mean([v[q] for q in qs])), "ci_lower": float(lo), "ci_upper": float(hi),
                  "diff_vs_rf": float(np.mean([rf[q] - v[q] for q in qs])),
                  "diff_ci_lower": float(dlo), "diff_ci_upper": float(dhi),
                  "sig_vs_rf": None if m == "RFoldSearch" else bool(dlo > 0 or dhi < 0),
                  "n_queries": len(qs), "n_blocks": len(names)}
    return res


def main():
    b = BM.load()
    ranks = {m: BM.load_rankings(m, b) for m in METHODS}
    labels = {"family": b.fam, "clan": b.clan}
    out = {"_note": __doc__.split("\n\n")[1], "_bootstrap": {"B": B, "level_seed": LEVEL_SEED,
                                                           "blocks": "Rfam family (family level) / clan (clan level)"},
           "panels": {}}
    for panel, (level, rows) in truths(b).items():
        per = {m: {} for m in METHODS}
        for q, pos, neu in rows:
            vals = {m: per_query(ranks[m], q, pos, neu, b.n) for m in METHODS}
            if all(v is not None for v in vals.values()):
                for m, v in vals.items():
                    per[m][q] = v
        res = bootstrap(per, level, labels[level])
        out["panels"][panel] = {"summary": res, "per_query": {m: {b.tags[q]: v for q, v in d.items()}
                                                             for m, d in per.items()}}
        print(f"{panel:14s} n={res['RFoldSearch']['n_queries']:3d}  " + "  ".join(
            f"{m} {res[m]['estimate']:.3f}{'*' if res[m]['sig_vs_rf'] else ''}" for m in METHODS), flush=True)
    json.dump(out, open(os.path.join(BM.RET, "roc_auc_test.json"), "w"), indent=1)

    cols = [("rfam|family", "Rfam family"), ("rfam|clan", "Rfam clan"),
            (f"tm{TABLE_TAU}|family", f"TM-score >= {TABLE_TAU}, all"),
            (f"tm{TABLE_TAU}|clan", f"TM-score >= {TABLE_TAU}, beyond-family"),
            (f"rm{TABLE_TAU}|family", "RMscore, all"), (f"rm{TABLE_TAU}|clan", "RMscore, beyond-family")]
    with open(os.path.join(BM.RET, "roc_auc_table.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["method"] + [f"{lab} ({k})" for k, lab in cols])
        w.writerow(["n queries / blocks"] + [f"{out['panels'][k]['summary']['RFoldSearch']['n_queries']} / "
                                             f"{out['panels'][k]['summary']['RFoldSearch']['n_blocks']}" for k, _ in cols])
        for m in METHODS:
            row = [m]
            for k, _ in cols:
                s = out["panels"][k]["summary"][m]
                row.append(f"{s['estimate']:.3f} [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]"
                           + ("" if s["sig_vs_rf"] is None else (" *" if s["sig_vs_rf"] else "")))
            w.writerow(row)
    print("wrote roc_auc_test.json and roc_auc_table.tsv")


if __name__ == "__main__":
    main()
