"""RiboSeek 0.1.0 sa20 encodings of the 885 RNAsolo chains, for the alphabet controls.

The controls (riboseek_alphabet_control.py, attribution_*.py) search the RiboSeek sa20
alphabet with the RFoldseek pipeline. They read the encodings from
files/ablation/riboseek_real_encoded.json, produced here with the package's own
encoder (riboseek==0.1.0, Searcher.from_pretrained("sa20").encode). Chains that fail
to encode get an empty label list.

    python riboseek_encode_v010.py
"""
import importlib.resources as R
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from paths import PRODUCTION_DB, resolve  # noqa: E402

CACHE = resolve("files/ablation/riboseek_real_encoded.json")


def main():
    import riboseek
    import torch
    from riboseek import Searcher
    if riboseek.__version__ != "0.1.0":
        raise SystemExit(f"expected riboseek 0.1.0, found {riboseek.__version__}")
    ents = torch.load(PRODUCTION_DB, map_location="cpu", weights_only=False)["entries"]
    with R.as_file(R.files("riboseek.data").joinpath("demo_db.json")) as demo:
        searcher = Searcher.from_pretrained("sa20", db=str(demo))
    enc = {}
    for i, e in enumerate(ents):
        try:
            lab = np.asarray(searcher.encode(str(e["id"])), dtype=np.int32)
        except Exception:
            lab = np.zeros(0, np.int32)
        enc[str(i)] = {"labels": lab.tolist(), "length": int(len(lab))}
    json.dump(enc, open(CACHE, "w"))
    print(f"wrote {CACHE} ({len(enc)} chains)")


if __name__ == "__main__":
    main()
