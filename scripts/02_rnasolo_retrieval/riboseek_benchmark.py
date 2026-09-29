#!/usr/bin/env python3
"""RiboSeek 0.2.0 (RS-80 and RS-20) on the Fig. 2 benchmark.

``run`` encodes the 885 benchmark chains with RiboSeek's own feature extractor
under the defaults of ``riboseek build-db`` (modified nucleotides dropped; RS-20
labels plus sequence, from which RS-80 letters are formed) and writes, for
each alphabet, the Top500 ranking of every Fig. 2 query:

    Searcher(Alphabet.from_pretrained(a), db, gap_penalty=-2.0, prefilter_k=6)
        .search(query, top_n=500, prefilter=True, prefilter_candidates=500)

``score`` adds one ranking file to the Fig. 2 summaries under a method name,
using the relevance, neutral-hit, query-selection and TM-score definitions of
``_infernal_retrieval_scorer.py`` unchanged, and rebuilds the C/D bootstrap.
``score-s12`` adds the same ranking to the the retrieval-metrics supplementary figure Recall@1 and per-query PR
summaries with the definitions of ``_infernal_metrics_scorer.py``; rerun
``bootstrap_retrieval_metrics.py`` afterwards.

    python riboseek_benchmark.py run
    python riboseek_benchmark.py score --rankings .../riboseek_rs80_rankings_test.json \
        --method "RiboSeek RS-80"
    python riboseek_benchmark.py score-s12 --rankings .../riboseek_rs80_rankings_test.json \
        --method "RiboSeek RS-80"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
from _infernal_retrieval_scorer import (RET, first_fp, load_rankings, load_tm,  # noqa: E402
                                   read_database_order, rfam_masks, update_bootstrap)
from paths import NR80_DIR  # noqa: E402

ENCODED = HERE.parents[1] / "files/ablation/riboseek020_encoded.json"
ALPHABETS = {"rs80": "riboseek_rs80_rankings_test.json",
             "sa20": "riboseek_rs20_rankings_test.json"}


def fig2_queries() -> list[int]:
    sens = json.load(open(RET / "per_query_sens_test.json"))
    structural = json.load(open(RET / "structural_tmscore_per_query_test.json"))
    queries = {int(q) for level in ("family", "clan") for q in sens["_query_ids"][level]}
    for per_threshold in structural.values():
        reference = next(iter(per_threshold.values()))
        queries |= {int(q) for level in ("family", "clan") for q in reference[level]}
    return sorted(queries)


def _encode(tag: str):
    from riboseek.alphabet import Alphabet
    from riboseek.features import pdb_to_features
    f = pdb_to_features(str(Path(NR80_DIR) / f"{tag}.cif"), keep_modified=False)
    labels = Alphabet.from_pretrained("sa20").encode(f["features"])
    return {"labels": labels.tolist(), "length": int(len(labels)), "seq": f["sequence"]}


def run() -> None:
    import riboseek
    from concurrent.futures import ProcessPoolExecutor
    from riboseek import Searcher
    from riboseek.alphabet import Alphabet
    if riboseek.__version__ != "0.2.0":
        raise SystemExit(f"expected riboseek 0.2.0, found {riboseek.__version__}")
    tags, _, _ = read_database_order()
    if ENCODED.exists():
        encoded = json.load(open(ENCODED))
    else:
        with ProcessPoolExecutor() as pool:
            encoded = {str(i): e for i, e in enumerate(pool.map(_encode, tags, chunksize=8))}
        json.dump(encoded, open(ENCODED, "w"))
    queries = fig2_queries()
    for alphabet, name in ALPHABETS.items():
        searcher = Searcher(Alphabet.from_pretrained(alphabet), encoded,
                            gap_penalty=-2.0, prefilter_k=6)
        rankings = {}
        for q in queries:
            hits = searcher.search(str(q), top_n=500, prefilter=True, prefilter_candidates=500)
            rankings[tags[q]] = {"ranking": [tags[int(h["chain"])] for h in hits]}
        json.dump(rankings, open(RET / name, "w"))
        size = np.mean([len(r["ranking"]) for r in rankings.values()])
        print(f"{alphabet}: {len(queries)} queries, {size:.0f} ranked candidates per query -> {name}")


def score(rankings_path: Path, method: str) -> None:
    tags, fam, clan = read_database_order()
    ranks = load_rankings(rankings_path, tags)
    empty = np.zeros(0, np.int64)

    path = RET / "per_query_sens_test.json"
    data = json.load(open(path))
    data[method] = {}
    for level in ("family", "clan"):
        values = [first_fp(ranks.get(q, empty), *rfam_masks(q, fam, clan, level))
                  for q in data["_query_ids"][level]]
        data[method][level] = {"sens": values, "n": len(values), "mean": float(np.mean(values))}
    json.dump(data, open(path, "w"), indent=2)

    pq_path = RET / "structural_tmscore_per_query_test.json"
    metric_path = RET / "structural_tmscore_metrics_test.json"
    per_query = json.load(open(pq_path))
    metrics = json.load(open(metric_path))
    tm = load_tm(len(fam))
    for key in sorted(per_query, key=float):
        threshold = float(key)
        reference = next(iter(per_query[key].values()))
        family_scores, clan_scores, recall1 = {}, {}, []
        for q in map(int, reference["family"]):
            valid = np.ones(len(fam), bool); valid[q] = False
            positive = (~np.isnan(tm[q])) & (tm[q] >= threshold) & valid
            ranking = ranks.get(q, empty)
            family_scores[str(q)] = first_fp(ranking, positive)
            recall1.append(int(len(ranking) > 0 and positive[ranking[0]]))
        for q in map(int, reference["clan"]):
            valid = np.ones(len(fam), bool); valid[q] = False
            positive = (~np.isnan(tm[q])) & (tm[q] >= threshold) & valid
            clan_scores[str(q)] = first_fp(ranks.get(q, empty), positive & (fam != fam[q]),
                                           (fam == fam[q]) & valid)
        per_query[key][method] = {"family": family_scores, "clan": clan_scores}
        metrics["thresholds"][key][method] = {
            "Recall@1": float(np.mean(recall1)),
            "Sens@1FP": float(np.mean(list(family_scores.values()))),
            "Sens@1FP_clan": float(np.mean(list(clan_scores.values()))),
            "n_queries": len(family_scores), "n_clan": len(clan_scores)}
    json.dump(per_query, open(pq_path, "w"), indent=2)
    json.dump(metrics, open(metric_path, "w"), indent=2)
    update_bootstrap(per_query, fam, clan)
    print(f"Added {method} from {rankings_path.name} ({len(ranks)} queries)")


def score_s12(rankings_path: Path, method: str) -> None:
    """the retrieval-metrics supplementary figure Recall@1 and per-query PR, as in ``_infernal_metrics_scorer.py``."""
    from _infernal_metrics_scorer import (attach_labels, load_labels, r1_clan, r1_family,
                                         rfam_per_query_pr_curve, torch)
    root = HERE.parents[1]
    ents = torch.load(root / "files/models/db.pt", map_location="cpu", weights_only=False)["entries"]
    tags = [Path(str(e["id"])).stem for e in ents]
    fam = np.array(attach_labels(ents, load_labels(root / "Splits/labels.tsv")), dtype=object)
    clan = np.array(attach_labels(ents, load_labels(root / "Splits/labels.tsv",
                                                    label_column="clan_acc")), dtype=object)
    ranks = load_rankings(rankings_path, tags)
    sens = json.load(open(RET / "per_query_sens_test.json"))
    fam_q = [int(q) for q in sens["_query_ids"]["family"]]
    clan_q = [int(q) for q in sens["_query_ids"]["clan"]]
    clan_eval = [q for q in clan_q if clan[q]]
    ranked = {q: ranks.get(q, np.zeros(0, np.int64)) for q in set(fam_q) | set(clan_q)}

    r1_path = RET / "recall1_rfam_test.json"
    r1 = json.load(open(r1_path))
    r1[method] = {"family_recall1": float(np.mean([r1_family(ranked[q], q, fam) for q in fam_q])),
                  "clan_recall1": float(np.mean([r1_clan(ranked[q], q, clan) for q in clan_eval])),
                  "n_family": len(fam_q), "n_clan": len(clan_eval)}
    json.dump(r1, open(r1_path, "w"), indent=2)
    pr_path = RET / "weighted_pr_rfam_test.json"
    pr = json.load(open(pr_path))
    pr[method] = {"family": rfam_per_query_pr_curve(fam_q, ranked, fam, clan, "family"),
                  "clan": rfam_per_query_pr_curve(clan_q, ranked, fam, clan, "clan")}
    json.dump(pr, open(pr_path, "w"), indent=2)
    print(f"Added {method} to the retrieval-metrics supplementary figure: R@1 family {r1[method]['family_recall1']:.3f}, "
          f"clan {r1[method]['clan_recall1']:.3f}; AUPR family {pr[method]['family']['aupr']:.3f}, "
          f"clan {pr[method]['clan']['aupr']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    for name in ("score", "score-s12"):
        s = sub.add_parser(name)
        s.add_argument("--rankings", type=Path, required=True)
        s.add_argument("--method", required=True)
    args = parser.parse_args()
    if args.cmd == "run":
        run()
    elif args.cmd == "score":
        score(args.rankings, args.method)
    else:
        score_s12(args.rankings, args.method)


if __name__ == "__main__":
    main()
