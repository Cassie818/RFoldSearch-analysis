#!/usr/bin/env python3
"""Add Infernal to the the retrieval-metrics supplementary figure supplementary retrieval metrics."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from paths import SRC_SEARCH  # noqa: E402

sys.path.insert(0, SRC_SEARCH)
from prefilter_rfam_benchmark import (  # noqa: E402
    attach_labels, load_labels, rfam_per_query_pr_curve,
)

import torch  # noqa: E402

RET = ROOT / "files/production/retrieval"


def r1_family(r, q, fam):
    """Top-ranked non-self target is a TP iff it shares the query Rfam family."""
    for t in r:
        if t == q:
            continue
        return 1 if fam[t] == fam[q] else 0
    return 0


def r1_clan(r, q, clan):
    """Top-ranked non-self target is a TP iff same clan (incl. same family)."""
    for t in r:
        if t == q:
            continue
        return 1 if (clan[q] and clan[t] == clan[q]) else 0
    return 0


def main() -> None:
    infernal_json = RET / "infernal_rankings_test.json"

    ents = torch.load(ROOT / "files/models/db.pt", map_location="cpu",
                      weights_only=False)["entries"]
    tags = [os.path.splitext(os.path.basename(str(e["id"])))[0] for e in ents]
    id2i = {t: i for i, t in enumerate(tags)}
    fam = np.array(attach_labels(ents, load_labels(ROOT / "Splits/labels.tsv")),
                   dtype=object)
    clan = np.array(attach_labels(ents, load_labels(ROOT / "Splits/labels.tsv",
                    label_column="clan_acc")), dtype=object)

    # raw Infernal rankings, chain_id -> db.pt index, self dropped
    raw = json.load(open(infernal_json))
    unknown = sorted(set(raw) - set(id2i))
    if unknown:
        raise RuntimeError(f"Unknown Infernal query IDs: {unknown[:3]}")
    ranks: dict[int, np.ndarray] = {}
    for qid, result in raw.items():
        q = id2i[qid]
        seen = {q}
        ordered = []
        for tid in result.get("ranking", []):
            t = id2i.get(tid)
            if t is not None and t not in seen:
                ordered.append(t)
                seen.add(t)
        ranks[q] = np.asarray(ordered, dtype=np.int64)

    # same query id lists and evaluability rule as eval_retrieval_metrics.py
    pq = json.load(open(RET / "per_query_sens_test.json"))
    fam_q = [int(q) for q in pq["_query_ids"]["family"]]
    clan_q = [int(q) for q in pq["_query_ids"]["clan"]]
    cq = [q for q in clan_q if clan[q]]

    fq = {q: ranks.get(q, np.zeros(0, np.int64)) for q in set(fam_q) | set(clan_q)}
    fr = float(np.mean([r1_family(fq[q], q, fam) for q in fam_q]))
    cr = float(np.mean([r1_clan(fq[q], q, clan) for q in cq]))

    r1_path = RET / "recall1_rfam_test.json"
    r1 = json.load(open(r1_path))
    r1["Infernal"] = {
        "family_recall1": fr, "clan_recall1": cr,
        "n_family": len(fam_q), "n_clan": len(cq),
    }
    json.dump(r1, open(r1_path, "w"), indent=2)

    pr_path = RET / "weighted_pr_rfam_test.json"
    pr = json.load(open(pr_path))
    pr["Infernal"] = {
        "family": rfam_per_query_pr_curve(fam_q, fq, fam, clan, "family"),
        "clan": rfam_per_query_pr_curve(clan_q, fq, fam, clan, "clan"),
    }
    json.dump(pr, open(pr_path, "w"), indent=2)

    print(f"Infernal  famR@1={fr:.3f}  clanR@1={cr:.3f}  "
          f"famAUPR={pr['Infernal']['family']['aupr']:.3f}  "
          f"clanAUPR={pr['Infernal']['clan']['aupr']:.3f}  "
          f"(n_family={len(fam_q)}, n_clan={len(cq)})")


if __name__ == "__main__":
    main()
