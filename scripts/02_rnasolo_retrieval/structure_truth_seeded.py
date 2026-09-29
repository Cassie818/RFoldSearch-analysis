"""Fig 2C/D: RFoldseek and its prefilter recall under the structure-defined truth,
averaged over the ten sha256 tie-breaking seeds (Section 2.2 convention).

Same queries and truth as eval_structure_truth.py: test queries with
complete US-align coverage; positives are targets with symmetric TM-score >= tau
(family level), or TM-similar targets from another Rfam family with same-family
targets neutral (clan level).

RFoldseek-TM ranks the RFoldseek prefilter candidates (top 200) by the same
TM-score that defines the truth, so all candidate positives precede all candidate
negatives and nothing outside the candidate set is returned. Its Sens@1FP is
therefore exactly the fraction of positives in the candidate set, i.e. the
prefilter recall, and it is reported as the ceiling of the candidate set rather
than as a competing method. The script checks this identity for every query and
seed.

Output: files/production/retrieval/structural_seeded_test.json
        {tau: {"RFoldseek" | "RFoldseek-TM": {"family" | "clan": {query: value}}}}
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, RFOLDSEEK_REPO, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, RFOLDSEEK_REPO)
_os.chdir(DATA_ROOT)
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels
from rfoldseek.tiebreak import tie_keys
from smith_waterman import smith_waterman_numba

RET = Path("files/production/retrieval")
K, TOPN, KMER_K, THRESH_FRAC, GO, GE = 80, 200, 4, 0.75, -2.0, -0.5
THRESHOLDS = [0.40, 0.45, 0.50]
NSEEDS = 10

ents = torch.load("files/models/db.pt", map_location="cpu", weights_only=False)["entries"]
n = len(ents)
tags = [os.path.splitext(os.path.basename(str(e["id"])))[0] for e in ents]
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
fam = np.array(attach_labels(ents, load_labels("Splits/labels.tsv")), dtype=object)
test = {id2i[json.loads(l)["id"]] for l in open("Splits/pdb_out/test.json")
        if l.strip() and json.loads(l)["id"] in id2i}

tm = np.full((n, n), np.nan, dtype=np.float32)
cov = defaultdict(int)
for line in open(RET / "usalign_full.cache.jsonl"):
    if line.strip():
        r = json.loads(line)
        tm[r["i"], r["j"]] = tm[r["j"], r["i"]] = 0.5 * (r["tm_query"] + r["tm_target"])
        cov[r["i"]] += 1
        cov[r["j"]] += 1
queries = sorted(q for q in range(n) if cov[q] >= n - 5 and q in test)

# ---- deployed token pipeline (as in eval_structure_truth.py) ----------
toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
sub = np.asarray(torch.load("files/models/sub_matrix.pt", map_location="cpu",
                            weights_only=False)["matrix"].numpy(), dtype=np.float32)
powers = np.array([K ** (KMER_K - 1 - i) for i in range(KMER_K)], dtype=np.int64)
idx = defaultdict(list)
for ti, tt in enumerate(toks):
    for p in range(len(tt) - KMER_K + 1):
        idx[int(np.dot(tt[p:p + KMER_K], powers))].append((ti, p))
smith_waterman_numba(np.zeros(4, np.int64), np.zeros(4, np.int64), sub, GO, GE)
thr = THRESH_FRAC * KMER_K
DIAG = np.array([sub[t, t] for t in range(K)], dtype=np.float64)


def expand(qk):
    base = int(np.dot(qk, powers))
    out = [base]
    self_score = float(sum(DIAG[int(qk[i])] for i in range(KMER_K)))
    for pos in range(KMER_K):
        cur = int(qk[pos])
        base_wo = self_score - DIAG[cur]
        for tok in range(K):
            if tok != cur and base_wo + sub[cur][tok] >= thr:
                out.append(base + int(powers[pos]) * (tok - cur))
    return out


def prefilter_scores(q):
    """All prefilter hits of q with their best diagonal count (before the top-N cut)."""
    tt = toks[q]
    td = defaultdict(dict)
    for qp in range(len(tt) - KMER_K + 1):
        for kid in expand(tt[qp:qp + KMER_K]):
            for ti, tp in idx.get(kid, ()):
                if ti != q:
                    td[ti][qp - tp] = td[ti].get(qp - tp, 0) + 1
    sc = [(ti, max((c for c in dm.values() if c >= 2), default=-1)) for ti, dm in td.items()]
    sc = [x for x in sc if x[1] > 0]
    return np.array([x[0] for x in sc], np.int64), np.array([x[1] for x in sc])


KEYS = [tie_keys(tags, seed) for seed in range(NSEEDS)]


def main():
    out = {f"{t}": {m: {"family": {}, "clan": {}} for m in ("RFoldseek", "RFoldseek-TM")} for t in THRESHOLDS}
    va_all = np.arange(n)
    for qi, q in enumerate(queries):
        c0, s0 = prefilter_scores(q)
        sw_cache = {}
        runs = []                                   # (candidates, RFoldseek ranking, RFoldseek-TM ranking)
        for keys in KEYS:
            if len(c0) == 0:
                runs.append((c0, c0, c0))
                continue
            cand = c0[np.lexsort((keys[c0], -s0))][:TOPN]
            sw = np.array([sw_cache.setdefault(int(t), smith_waterman_numba(toks[q], toks[t], sub, GO, GE))
                           for t in cand])
            covered = cand[~np.isnan(tm[q, cand])]
            runs.append((cand, cand[np.lexsort((keys[cand], -sw))],
                         covered[np.lexsort((keys[covered], -tm[q, covered]))]))
        va = va_all != q
        for t in THRESHOLDS:
            pos = (~np.isnan(tm[q])) & (tm[q] >= t) & va
            levels = {"family": (pos, None)}
            if fam[q]:
                levels["clan"] = (pos & (fam != fam[q]), (fam == fam[q]) & va)
            for level, (P, N) in levels.items():
                if not P.any():
                    continue
                rf, ceil = [], []
                for cand, rank_rf, rank_tm in runs:
                    rf.append(first_fp_sensitivity(rank_rf, P, N) if len(rank_rf) else 0.0)
                    s = first_fp_sensitivity(rank_tm, P, N) if len(rank_tm) else 0.0
                    recall = float(P[cand].sum() / P.sum()) if len(cand) else 0.0
                    assert abs(s - recall) < 1e-9, (q, t, level, s, recall)
                    ceil.append(s)
                out[f"{t}"]["RFoldseek"][level][str(q)] = float(np.mean(rf))
                out[f"{t}"]["RFoldseek-TM"][level][str(q)] = float(np.mean(ceil))
        if (qi + 1) % 20 == 0:
            print(f"  {qi + 1}/{len(queries)} queries", flush=True)
    for t, d in out.items():
        print(f"TM>={t}: " + "  ".join(f"{m} {lv} {np.mean(list(d[m][lv].values())):.3f} (n={len(d[m][lv])})"
                                       for m in d for lv in ("family", "clan")))
    dst = RET / "structural_seeded_test.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
