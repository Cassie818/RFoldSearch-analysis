"""Shared definitions for the RNAsolo benchmark (Fig. 2 and its supplementary figures).

Database   the 885 RNAsolo chains of files/models/db.pt, in database order.
Queries    held-out test chains (Splits/pdb_out/test.json):
             family  queries with another chain of the same Rfam family
             clan    queries with a chain of the same clan but another family
             struct  queries with a complete US-align row (structure truth)
Truths     Rfam:      family positives share the family, same-clan other-family
                      targets are neutral; clan positives share the clan but not the
                      family, same-family targets are neutral. Unlabelled chains are
                      neutral.
           structure: positives have symmetric US-align TM-score >= tau (family
                      level); at clan level they must also come from another Rfam
                      family, and same-family targets are neutral.
Rankings   one file per method in files/production/retrieval/rankings/<slug>.json,
           {query chain: {"ranking": [chain, ...]}} for a deterministic method, or
           {"rankings": [[chain, ...], ...]} with one ranking per tie-breaking seed.
"""
import json
import os
import sys
from collections import defaultdict
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from paths import DATA_ROOT, RFOLDSEARCH_REPO, SRC_SEARCH  # noqa: E402

sys.path.insert(0, SRC_SEARCH)
sys.path.insert(0, RFOLDSEARCH_REPO)
from prefilter_rfam_benchmark import (  # noqa: E402
    attach_labels,
    first_fp_sensitivity,
    load_labels,
    rfam_relevance_masks,
)

RET = os.path.join(DATA_ROOT, "files", "production", "retrieval")
RANKINGS = os.path.join(RET, "rankings")
THRESHOLDS = (0.40, 0.45, 0.50)
NSEEDS = 10

# method name -> rankings file slug
METHODS = {
    "RFoldSearch": "rfoldsearch", "RFoldSearch-TM": "rfoldsearch_tm",
    "US-align": "usalign", "RMalign": "rmalign", "GTalign": "gtalign", "RTM-align": "rtmalign",
    "RiboSeek": "riboseek_rs20", "RiboSeek RS-80": "riboseek_rs80",
    "BLASTN": "blastn", "Infernal": "infernal",
}


@lru_cache(maxsize=None)
def load():
    """Database, labels, query sets and the symmetric US-align TM-score matrix."""
    import torch
    ents = torch.load(os.path.join(DATA_ROOT, "files", "models", "db.pt"), map_location="cpu",
                      weights_only=False)["entries"]
    n = len(ents)
    tags = [os.path.splitext(os.path.basename(str(e["id"])))[0] for e in ents]
    labels = os.path.join(DATA_ROOT, "Splits", "labels.tsv")
    fam = np.array(attach_labels(ents, load_labels(labels)), dtype=object)
    clan = np.array(attach_labels(ents, load_labels(labels, label_column="clan_acc")), dtype=object)
    id2i = {str(e["id"]): i for i, e in enumerate(ents)}
    test = set()
    for line in open(os.path.join(DATA_ROOT, "Splits", "pdb_out", "test.json")):
        if line.strip() and json.loads(line)["id"] in id2i:
            test.add(id2i[json.loads(line)["id"]])

    tm = np.full((n, n), np.nan, dtype=np.float32)
    cov = defaultdict(int)
    for line in open(os.path.join(RET, "usalign_full.cache.jsonl")):
        if line.strip():
            r = json.loads(line)
            tm[r["i"], r["j"]] = tm[r["j"], r["i"]] = 0.5 * (r["tm_query"] + r["tm_target"])
            cov[r["i"]] += 1
            cov[r["j"]] += 1

    va = np.arange(n)
    fam_q = [q for q in sorted(test) if fam[q] and np.any((fam == fam[q]) & (va != q))]
    clan_q = [q for q in sorted(test)
              if clan[q] and np.any((clan == clan[q]) & (fam != fam[q]) & (va != q))]
    struct_q = sorted(q for q in test if cov[q] >= n - 5)
    return SimpleNamespace(ents=ents, n=n, tags=tags, tag2i={t: i for i, t in enumerate(tags)},
                           fam=fam, clan=clan, test=test, tm=tm, fam_q=fam_q, clan_q=clan_q,
                           struct_q=struct_q, queries=sorted(set(fam_q) | set(clan_q) | set(struct_q)))


def rfam_masks(b, q, level):
    """(positives, neutrals) of query q under the Rfam truth; unlabelled targets are neutral."""
    return rfam_relevance_masks(q, b.fam, b.clan, level)


def structure_masks(b, q, tau, level, score=None):
    """(positives, neutrals) of query q under a structure truth (default: US-align TM)."""
    score = b.tm if score is None else score
    other = np.arange(b.n) != q
    pos = (~np.isnan(score[q])) & (score[q] >= tau) & other
    if level == "family":
        return pos, None
    if not b.fam[q]:
        return np.zeros(b.n, bool), None
    return pos & (b.fam != b.fam[q]), (b.fam == b.fam[q]) & other


def sens1fp(ranking, pos, neu=None):
    return first_fp_sensitivity(ranking, pos, neu) if len(ranking) else 0.0


def rankings_path(method):
    return os.path.join(RANKINGS, METHODS[method] + ".json")


def save_rankings(method, runs, b=None):
    """runs: {query index: [ranking, ...]} with one ranking per seed (or a single one)."""
    b = b or load()
    out = {}
    for q, rs in sorted(runs.items()):
        rs = [[b.tags[int(t)] for t in r if int(t) != q] for r in rs]
        out[b.tags[q]] = {"ranking": rs[0]} if len(rs) == 1 else {"rankings": rs}
    os.makedirs(RANKINGS, exist_ok=True)
    json.dump(out, open(rankings_path(method), "w"))
    print(f"wrote {rankings_path(method)} ({len(out)} queries)", flush=True)


def load_rankings(method, b=None):
    """{query index: [ranking array, ...]}, the query itself removed."""
    b = b or load()
    d = json.load(open(rankings_path(method)))
    out = {}
    for tag, v in d.items():
        q = b.tag2i[tag]
        rs = v["rankings"] if "rankings" in v else [v["ranking"]]
        out[q] = [np.asarray([b.tag2i[t] for t in r if t in b.tag2i and b.tag2i[t] != q], np.int64)
                  for r in rs]
    return out


def rank_by_score(row, q):
    """Targets with a score, highest first (ties by database index), query excluded."""
    idx = np.flatnonzero(~np.isnan(row))
    idx = idx[idx != q]
    return idx[np.lexsort((idx, -row[idx]))]
