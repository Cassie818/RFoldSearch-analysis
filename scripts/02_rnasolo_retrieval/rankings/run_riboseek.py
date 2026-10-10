"""RiboSeek 0.2.0 rankings (RS-20 and RS-80) on the RNAsolo benchmark."""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import benchmark as BM  # noqa: E402
from paths import NR80_DIR  # noqa: E402

ENCODED = os.path.join(BM.DATA_ROOT, "files", "ablation", "riboseek020_encoded.json")
ALPHABETS = {"sa20": "RiboSeek", "rs80": "RiboSeek RS-80"}


def encode(tag):
    from riboseek.alphabet import Alphabet
    from riboseek.features import pdb_to_features
    f = pdb_to_features(os.path.join(NR80_DIR, f"{tag}.cif"), keep_modified=False)
    labels = Alphabet.from_pretrained("sa20").encode(f["features"])
    return {"labels": labels.tolist(), "length": int(len(labels)), "seq": f["sequence"]}


def main():
    import riboseek
    from riboseek import Searcher
    from riboseek.alphabet import Alphabet
    if riboseek.__version__ != "0.2.0":
        raise SystemExit(f"expected riboseek 0.2.0, found {riboseek.__version__}")
    b = BM.load()
    if os.path.exists(ENCODED):
        encoded = json.load(open(ENCODED))
    else:
        with ProcessPoolExecutor() as pool:
            encoded = {str(i): e for i, e in enumerate(pool.map(encode, b.tags, chunksize=8))}
        json.dump(encoded, open(ENCODED, "w"))
    for alphabet, method in ALPHABETS.items():
        searcher = Searcher(Alphabet.from_pretrained(alphabet), encoded, gap_penalty=-2.0, prefilter_k=6)
        runs = {}
        for q in b.queries:
            hits = searcher.search(str(q), top_n=500, prefilter=True, prefilter_candidates=500)
            runs[q] = [np.asarray([int(h["chain"]) for h in hits], np.int64)]
        BM.save_rankings(method, runs, b)


if __name__ == "__main__":
    main()
