"""RFoldSearch and RFoldSearch-TM rankings on the RNAsolo benchmark, ten tie-breaking seeds.

RFoldSearch is the deployed pipeline: the similar-k-mer prefilter (exact 4-mer index with
one-substitution Structure-BLOSUM expansion, best diagonal with >= 2 hits), the top 200
candidates, then ranking by the structure score S * aTM.
RFoldSearch-TM ranks the same candidates by the symmetric US-align TM-score; candidates
without a US-align score are dropped. Ties are broken by the seeded chain-id hash
both at the prefilter cut and at the ranking."""
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import benchmark as BM  # noqa: E402

K, TOPN, KMER_K, THRESH_FRAC, GO, GE = 80, 200, 4, 0.75, -2.0, -0.5


def main():
    import torch
    from rfoldseek.structure_score import c4_coords, score_pair
    from rfoldseek.tiebreak import tie_keys
    b = BM.load()
    toks = [np.asarray(e["tokens"], dtype=np.int64) for e in b.ents]
    xyz = [c4_coords(e, len(t)) for e, t in zip(b.ents, toks)]
    assert all(x is not None for x in xyz), "every chain needs C4' coordinates"
    sub = np.asarray(torch.load(os.path.join(BM.DATA_ROOT, "files", "models", "sub_matrix.pt"),
                                map_location="cpu", weights_only=False)["matrix"].numpy(), dtype=np.float32)
    powers = np.array([K ** (KMER_K - 1 - i) for i in range(KMER_K)], dtype=np.int64)
    index = defaultdict(list)
    for ti, tt in enumerate(toks):
        for p in range(len(tt) - KMER_K + 1):
            index[int(np.dot(tt[p:p + KMER_K], powers))].append((ti, p))
    diag = np.array([sub[t, t] for t in range(K)], dtype=np.float64)
    thr = THRESH_FRAC * KMER_K

    def expand(kmer):
        base = int(np.dot(kmer, powers))
        out = [base]
        self_score = float(sum(diag[int(kmer[i])] for i in range(KMER_K)))
        for pos in range(KMER_K):
            cur = int(kmer[pos])
            for tok in range(K):
                if tok != cur and self_score - diag[cur] + sub[cur][tok] >= thr:
                    out.append(base + int(powers[pos]) * (tok - cur))
        return out

    def prefilter(q):
        """All prefilter hits of q with their best diagonal count, before the top-N cut."""
        tt = toks[q]
        diagonals = defaultdict(dict)
        for qp in range(len(tt) - KMER_K + 1):
            for kid in expand(tt[qp:qp + KMER_K]):
                for ti, tp in index.get(kid, ()):
                    if ti != q:
                        diagonals[ti][qp - tp] = diagonals[ti].get(qp - tp, 0) + 1
        hits = [(ti, max((c for c in dm.values() if c >= 2), default=-1)) for ti, dm in diagonals.items()]
        hits = [h for h in hits if h[1] > 0]
        return np.array([h[0] for h in hits], np.int64), np.array([h[1] for h in hits])

    keys = [tie_keys(b.tags, seed) for seed in range(BM.NSEEDS)]
    go, ge = np.float32(GO), np.float32(GE)
    rf, rtm = {}, {}
    for k, q in enumerate(b.queries):
        c0, s0 = prefilter(q)
        cache = {}
        rf[q], rtm[q] = [], []
        for key in keys:
            if len(c0) == 0:
                rf[q].append(c0)
                rtm[q].append(c0)
                continue
            cand = c0[np.lexsort((key[c0], -s0))][:TOPN]
            for t in cand:
                if int(t) not in cache:
                    s, atm, _ = score_pair(toks[q], xyz[q], toks[t], xyz[t], sub, go, ge)
                    cache[int(t)] = s * atm
            score = np.array([cache[int(t)] for t in cand])
            rf[q].append(cand[np.lexsort((key[cand], -score))])
            covered = cand[~np.isnan(b.tm[q, cand])]
            rtm[q].append(covered[np.lexsort((key[covered], -b.tm[q, covered]))])
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(b.queries)} queries", flush=True)
    BM.save_rankings("RFoldSearch", rf, b)
    BM.save_rankings("RFoldSearch-TM", rtm, b)


if __name__ == "__main__":
    main()
