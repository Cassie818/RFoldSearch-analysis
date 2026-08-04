#!/usr/bin/env python3
"""Calibrate Karlin-Altschul lambda/K for RFoldseek E-values and cache them.

Fits the Gumbel parameters of the token Smith-Waterman null score distribution
(random token-sequence pairs from the target database's background frequencies,
same Structure-BLOSUM matrix and gap penalties as the search) and writes them to
files/models/evalue_params.json. The RFoldseek CLI reads that file to report a
database E-value for every hit. Re-run only when the database or substitution
matrix changes.

    python src/search/build_evalue_params.py                 # bundled db/matrix
    python src/search/build_evalue_params.py --db my.pt --matrix m.pt --out o.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", ".."))
from paths import resolve  # noqa: E402

sys.path.insert(0, _HERE)
import evalue as ev  # noqa: E402

GAP_OPEN = -2.0
GAP_EXTEND = -0.5


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=resolve("files/models/db.pt"),
                    help="target token database (.pt)")
    ap.add_argument("--matrix", default=resolve("files/models/sub_matrix.pt"),
                    help="Structure-BLOSUM substitution matrix (.pt)")
    ap.add_argument("--out", default=resolve("files/models/evalue_params.json"),
                    help="output JSON for the calibrated lambda/K")
    ap.add_argument("--lengths", type=int, nargs="+",
                    default=[20, 30, 40, 60, 80, 120, 160, 220, 300, 500, 1000, 2000],
                    help="length grid to calibrate lambda/K on")
    ap.add_argument("--ref_length", type=int, default=300,
                    help="grid length whose lambda/K become the scalar fallback")
    ap.add_argument("--n_pairs", type=int, default=4000,
                    help="random pairs per length (<= long_thresh)")
    ap.add_argument("--long_n_pairs", type=int, default=800,
                    help="random pairs per length (> long_thresh)")
    ap.add_argument("--long_thresh", type=int, default=300)
    ap.add_argument("--no_monotone", action="store_true",
                    help="skip the non-increasing (isotonic) smoothing of lambda/K")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    db = torch.load(args.db, map_location="cpu", weights_only=False)
    ents = db["entries"]
    n_codes = int(db.get("n_codes", 0)) or int(max(int(np.max(e["tokens"])) for e in ents) + 1)
    toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
    n_db_tokens = int(sum(len(t) for t in toks))
    sub = torch.load(args.matrix, map_location="cpu", weights_only=False)
    sub = np.asarray(sub["matrix"].numpy() if isinstance(sub, dict) else sub, dtype=np.float32)

    print(f"database: {len(toks)} chains, {n_db_tokens} tokens, K={n_codes} codes", flush=True)
    print(f"length-dependent calibration on lengths {args.lengths} ...", flush=True)
    table = ev.calibrate_length_dependent(
        sub, ev.background_frequencies(toks, n_codes),
        GAP_OPEN, GAP_EXTEND, tuple(args.lengths),
        args.n_pairs, args.long_n_pairs, args.long_thresh, args.seed,
        monotone=not args.no_monotone,
    )
    # scalar lambda/K (backward-compatible fallback) = the reference-length row
    ref = min(table, key=lambda r: abs(r["length"] - args.ref_length))
    par = {
        "lambda": ref["lambda"],
        "K": ref["K"],
        "reference_length": ref["length"],
        "gap_open": GAP_OPEN,
        "gap_extend": GAP_EXTEND,
        "seed": args.seed,
        "length_table": table,
        "note": (
            "Length-dependent Karlin-Altschul Gumbel parameters (method-of-moments) "
            "for the token Smith-Waterman score under Structure-BLOSUM. The CLI "
            "interpolates (lambda, K) to the query length via evalue.params_for_length "
            "(lambda linearly, K in log space) to correct the finite-length edge "
            "effect that inflates significance for short RNAs. Database E-value for a "
            "hit scoring S is E = K * m * N * exp(-lambda * S), with m the query length "
            "and N the total token count of the target database. The scalar lambda/K "
            "are the reference_length row, kept as a fallback for older readers."
        ),
        "n_db_tokens": n_db_tokens,
        "n_chains": len(toks),
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(par, open(args.out, "w"), indent=2)
    print("length :  lambda      K        null_mean  null_std", flush=True)
    for r in table:
        print(f"{r['length']:>6} : {r['lambda']:.4f}  {r['K']:.4g}  "
              f"{r['null_score_mean']:>8.2f}  {r['null_score_std']:>7.3f}", flush=True)
    print(f"scalar fallback = length {ref['length']}: "
          f"lambda={ref['lambda']:.4f} K={ref['K']:.4g}", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
