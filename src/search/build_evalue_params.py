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
    ap.add_argument("--length", type=int, default=300, help="random-sequence length")
    ap.add_argument("--n_pairs", type=int, default=4000, help="number of random pairs")
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
    print(f"calibrating on {args.n_pairs} random {args.length}-token pairs...", flush=True)
    par = ev.calibrate(sub, ev.background_frequencies(toks, n_codes),
                       GAP_OPEN, GAP_EXTEND, args.length, args.n_pairs, args.seed)
    # store the database size so the CLI need not re-scan the db to size E-values
    par["n_db_tokens"] = n_db_tokens
    par["n_chains"] = len(toks)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(par, open(args.out, "w"), indent=2)
    print(f"lambda={par['lambda']:.4f}  K={par['K']:.4g}  "
          f"(null mean={par['null_score_mean']:.2f} std={par['null_score_std']:.2f} "
          f"min={par['null_score_min']:.2f})", flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
