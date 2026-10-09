"""Score every method's rankings on the RNAsolo benchmark (Fig. 2 and Supplementary).

Reads the ranking files written by rankings/run_*.py and writes all per-query and summary
values in one pass; no other script edits these files.

  per_query_sens_test.json               Rfam truth, Sens@1FP per query       (Fig. 2A/B)
  structural_tmscore_per_query_test.json structure truth, Sens@1FP per query  (Fig. 2C/D)
  structural_tmscore_metrics_test.json   structure truth, Recall@1 and mean Sens@1FP
  recall1_rfam_test.json                 Rfam truth, Recall@1                  (Supplementary)
  weighted_pr_rfam_test.json             Rfam truth, per-query AP and PR curve (Supplementary)

Under the Rfam truth, unlabelled targets are neutral (skipped).

"""
import json
import os

import numpy as np

import benchmark as BM
from prefilter_rfam_benchmark import per_query_average_precision

TIE_BREAK = {
    "methods": ["RFoldSearch", "RFoldSearch-TM"],
    "scheme": 'sha256("<seed>:<chain_id>"), applied at the prefilter top_n cut and at the reranking',
    "seeds": list(range(BM.NSEEDS)),
    "per_query_values": "mean over seeds",
    "note": "Other methods rank the whole database with no top_n cut and keep the index tie-break.",
}


def write(name, data):
    json.dump(data, open(os.path.join(BM.RET, name), "w"), indent=2)
    print(f"wrote {name}", flush=True)


def per_seed(runs, q, fn):
    """Mean of fn(ranking) over the seeds of query q (one empty ranking if q has none)."""
    return float(np.mean([fn(r) for r in runs.get(q, [np.zeros(0, np.int64)])]))


def rfam_truth(b, ranks):
    out = {"_query_ids": {"family": b.fam_q, "clan": b.clan_q}}
    for m, runs in ranks.items():
        out[m] = {}
        for level, qs in (("family", b.fam_q), ("clan", b.clan_q)):
            sens = [per_seed(runs, q, lambda r: BM.sens1fp(r, *BM.rfam_masks(b, q, level))) for q in qs]
            out[m][level] = {"sens": sens, "n": len(sens), "mean": float(np.mean(sens))}
            nseed = len(next(iter(runs.values())))
            if nseed > 1:
                out[m][level]["seed_means"] = [
                    float(np.mean([BM.sens1fp(runs[q][s], *BM.rfam_masks(b, q, level)) for q in qs]))
                    for s in range(nseed)]
    out["_tie_break"] = TIE_BREAK
    return out


def structure_truth(b, ranks):
    per_query, metrics = {}, {}
    for tau in BM.THRESHOLDS:
        per_query[f"{tau}"], metrics[f"{tau}"] = {}, {}
        for m, runs in ranks.items():
            name = "US-align*" if m == "US-align" else m
            fam, clan, r1 = {}, {}, []
            for q in b.struct_q:
                pos, _ = BM.structure_masks(b, q, tau, "family")
                if pos.any():
                    fam[str(q)] = per_seed(runs, q, lambda r: BM.sens1fp(r, pos))
                    r1.append(per_seed(runs, q, lambda r: float(len(r) > 0 and pos[r[0]])))
                pos_c, neu_c = BM.structure_masks(b, q, tau, "clan")
                if pos_c.any():
                    clan[str(q)] = per_seed(runs, q, lambda r: BM.sens1fp(r, pos_c, neu_c))
            per_query[f"{tau}"][name] = {"family": fam, "clan": clan}
            metrics[f"{tau}"][name] = {
                "Recall@1": float(np.mean(r1)), "Sens@1FP": float(np.mean(list(fam.values()))),
                "Sens@1FP_clan": float(np.mean(list(clan.values()))),
                "n_queries": len(fam), "n_clan": len(clan)}
    return per_query, {"_note": "test-only covered queries; US-align* is the TM ceiling.",
                       "thresholds": metrics, "_tie_break": TIE_BREAK}


def recall1(b, ranks):
    def top_hit(r, q, same):
        r = r[b.fam[r] != ""]                        # unlabelled targets skipped
        return float(len(r) > 0 and same(r[0]))
    out = {"_note": ("Rfam family/clan Recall@1, test queries: the top-ranked labelled non-self target is a "
                     "TP; family TP = same family, clan TP = same clan including the same family.")}
    cq = [q for q in b.clan_q if b.clan[q]]
    for m, runs in ranks.items():
        out[m] = {
            "family_recall1": float(np.mean([per_seed(runs, q, lambda r: top_hit(r, q, lambda t: b.fam[t] == b.fam[q]))
                                             for q in b.fam_q])),
            "clan_recall1": float(np.mean([per_seed(runs, q, lambda r: top_hit(r, q, lambda t: b.clan[t] == b.clan[q]))
                                           for q in cq])),
            "n_family": len(b.fam_q), "n_clan": len(cq)}
    return out


def precision_recall(b, ranks):
    out = {"_note": ("Per-query-averaged precision-recall for the test queries: each query contributes one "
                     "average precision (AP) and one interpolated PR curve, weighted equally. 'aupr' is the "
                     "mean AP; 'precision' is the mean interpolated curve on the 'recall' grid. Family: TP = "
                     "same Rfam family, neutral = same clan, other family. Clan: TP = same clan, other family, "
                     "neutral = same family. Unlabelled targets are neutral. Unretrieved relevant targets "
                     "remain false negatives. Seeded "
                     "methods: per-query AP and curve averaged over seeds."),
           "_query_ids": {"family": b.fam_q, "clan": b.clan_q}}
    for m, runs in ranks.items():
        out[m] = {}
        for level, qs in (("family", b.fam_q), ("clan", b.clan_q)):
            nseed = len(next(iter(runs.values())))
            curves = [pr_curve(b, qs, {q: runs[q][s] for q in qs if q in runs}, level) for s in range(nseed)]
            c = dict(curves[0])
            if nseed > 1:
                ap = np.mean([x["per_query_ap"] for x in curves], axis=0)
                c.update(precision=np.mean([x["precision"] for x in curves], axis=0).tolist(),
                         per_query_ap=ap.tolist(), aupr=float(ap.mean()), aupr_std=float(ap.std()))
            out[m][level] = c
    return out


def pr_curve(b, qs, ranked, level, grid=np.linspace(0.0, 1.0, 101)):
    """Per-query-averaged precision-recall curve and mean AP (unlabelled targets neutral)."""
    aps, curves = [], []
    for q in qs:
        pos, neu = BM.rfam_masks(b, q, level)
        neu = np.zeros(b.n, bool) if neu is None else neu.copy()
        neu[q] = True
        res = per_query_average_precision(ranked.get(q, np.zeros(0, np.int64)), pos, neu, grid)
        if res is not None:
            aps.append(res[0])
            curves.append(res[1])
    return {"recall": grid.tolist(), "precision": np.mean(curves, axis=0).tolist(),
            "aupr": float(np.mean(aps)), "aupr_std": float(np.std(aps)),
            "per_query_ap": [float(a) for a in aps], "n_queries": len(aps), "level": level,
            "weighting": "per-query mean average precision (each query weighted equally)"}


def main():
    b = BM.load()
    ranks = {m: BM.load_rankings(m, b) for m in BM.METHODS}
    missing = {m: [b.tags[q] for q in b.queries if q not in r] for m, r in ranks.items()}
    for m, qs in missing.items():
        if qs and m != "Infernal":
            raise SystemExit(f"{m}: no ranking for {len(qs)} queries")
    write("per_query_sens_test.json", rfam_truth(b, ranks))
    per_query, metrics = structure_truth(b, ranks)
    write("structural_tmscore_per_query_test.json", per_query)
    write("structural_tmscore_metrics_test.json", metrics)
    write("recall1_rfam_test.json", recall1(b, ranks))
    write("weighted_pr_rfam_test.json", precision_recall(b, ranks))


if __name__ == "__main__":
    main()
