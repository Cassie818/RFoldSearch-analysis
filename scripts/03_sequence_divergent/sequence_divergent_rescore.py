"""Re-score saved rankings of the sequence-divergent analysis under any identity stratification.

Reads files/production/retrieval/sequence_divergent/family_rankings.json (rankings of every
method for the 106 family-evaluable RNAsolo test queries; RFoldseek and RFoldseek-TM one
ranking per tie-breaking seed) and family_pair_identity.json (query--target identity for
every same-family pair). Section 2.2 family-level definitions: within a stratum only
same-family targets in the stratum are positives, same-family targets outside it and
same-clan targets of other families are neutral, all other targets (including unlabelled
chains) are false positives.

    from sequence_divergent_rescore import rescore
    per = rescore([("bg", 0.0, 0.67), ("mid", 0.67, 0.80)])   # {stratum: {method: {query: Sens@1FP}}}
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "src", "search"))
from prefilter_rfam_benchmark import first_fp_sensitivity         # noqa: E402

DATA = os.path.join(REPO, "files", "production", "retrieval", "sequence_divergent")
SEEDED = ("RFoldseek", "RFoldseek-TM", "Shuffled")                # several rankings, averaged


def load():
    R = json.load(open(os.path.join(DATA, "family_rankings.json")))
    pid = json.load(open(os.path.join(DATA, "family_pair_identity.json")))
    ident = defaultdict(dict)
    for k, v in pid.items():
        a, b = k.split("|")
        ident[a][b] = v
        ident[b][a] = v
    return R, ident


def rescore(strata, methods=None, ident=None):
    """ident: optional {query: {target: value}} replacing pairwise identity as the stratifying value."""
    R, ident0 = load()
    ident = ident or ident0
    tags, fam, clan = R["tags"], np.array(R["fam"], dtype=object), np.array(R["clan"], dtype=object)
    n = len(tags)
    pos = {t: i for i, t in enumerate(tags)}
    per = {name: defaultdict(dict) for name, _, _ in strata}
    for qt in R["queries"]:
        q = pos[qt]
        va = np.arange(n) != q
        same = (fam == fam[q]) & va
        sameclan = ((clan == clan[q]) & (fam != fam[q]) & va) if clan[q] else np.zeros(n, bool)
        idv = np.full(n, np.nan)
        for t, v in ident.get(qt, {}).items():
            idv[pos[t]] = v
        for name, lo, hi in strata:
            P = same & (idv >= lo) & (idv < hi)
            if not P.any():
                continue
            N = sameclan | (same & ~P)
            for m, r in R["ranks"][qt].items():
                if m in ("seq_detect_E", "infernal_any") or (methods and m not in methods):
                    continue
                runs = r if m in SEEDED else [r]
                per[name][m][qt] = float(np.mean([first_fp_sensitivity(np.asarray(x, np.int64), P, N) if len(x) else 0.0
                                                  for x in runs]))
    return per, {qt: fam[pos[qt]] for qt in R["queries"]}


# ---- pairwise sequence identity and its background ---------------------------------------
# Global alignment with the settings of the sequence-alignment baseline of the alignment
# benchmark (Biopython PairwiseAligner, match +2 / mismatch -1, gap open -3 / extend -0.5);
# identity = identical aligned positions / length of the shorter sequence.
def _aligner():
    from Bio.Align import PairwiseAligner
    al = PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score = 2.0, -1.0
    al.open_gap_score, al.extend_gap_score = -3.0, -0.5
    return al


def _clean(x):
    return "".join(c if c in "ACGU" else "N" for c in x.upper().replace("T", "U"))


def seq_identity(a, b, al=None):
    al = al or _aligner()
    a, b = _clean(a), _clean(b)
    aln = al.align(a, b)[0]
    same = sum(1 for (qs, qe), (ts, te) in zip(*aln.aligned)
               for x, y in zip(a[qs:qe], b[ts:te]) if x == y and x != "N")
    return same / min(len(a), len(b))


def unrelated(a, b, fam, clan):
    """Labelled chains from different Rfam families and different clans."""
    return fam[a] != fam[b] and not (clan[a] and clan[a] == clan[b])


def background_identity(q=0.95, n_pairs=5000, seed=0, length=(40, 200)):
    """Identity of unrelated chains: random pairs of labelled chains from different Rfam
    families and clans, lengths in `length`, length ratio 0.8-1.25. Returns (quantile q, all values)."""
    import csv
    import random
    R = json.load(open(os.path.join(DATA, "family_rankings.json")))
    fam, clan = dict(zip(R["tags"], R["fam"])), dict(zip(R["tags"], R["clan"]))
    seqs = {r["chain_id"]: r["sequence"] for r in
            csv.DictReader(open(os.path.join(REPO, "files", "production", "chain_sequences.tsv")), delimiter="\t")}
    pool = [t for t in R["tags"] if fam[t] and length[0] <= len(seqs[t]) <= length[1]]
    rng = random.Random(seed)
    al = _aligner()
    vals = []
    while len(vals) < n_pairs:
        a, b = rng.sample(pool, 2)
        if unrelated(a, b, fam, clan) and 0.8 <= len(seqs[a]) / len(seqs[b]) <= 1.25:
            vals.append(seq_identity(seqs[a], seqs[b], al))
    return float(np.quantile(vals, q)), vals
