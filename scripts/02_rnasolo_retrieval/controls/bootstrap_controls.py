"""Block-bootstrap significance for the control experiments (RiboSeek alphabet and quantiser controls)."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))
from paths import SRC_SEARCH, resolve
_sys.path.insert(0, SRC_SEARCH)
import json
from collections import defaultdict
import numpy as np, torch
from prefilter_rfam_benchmark import attach_labels, load_labels

B, SEED = 2000, 123
LEVEL_SEED = {"family": SEED + 1, "clan": SEED + 2}

ALPHABET = {  # attribution controls: RFoldSearch tokens vs RiboSeek sa20, same pipeline
    "native": "riboseek_alphabet_same_pipeline_control_per_query.json",
    "identity": "riboseek_alphabet_same_pipeline_control_identity_per_query.json",
    "self_blosum": "riboseek_alphabet_same_pipeline_control_self_blosum_per_query.json",
}
QUANTISER = [  # the quantiser-control supplementary figure: deployed VQ-VAE vs k-means on the same descriptor
    "kmeans_quantiser_control_per_query.json",
    "kmeans_quantiser_control_k10k40_per_query.json",
]


def paired_ci(diff_by_block, seed):
    names = list(diff_by_block)
    rng = np.random.default_rng(seed)
    reps = [rng.choice(names, size=len(names), replace=True) for _ in range(B)]
    dm = np.array([np.mean([v for b in s for v in diff_by_block[b]]) for s in reps])
    lo, hi = np.percentile(dm, [2.5, 97.5])
    obs = float(np.mean([v for b in names for v in diff_by_block[b]]))
    return {"diff_mean": obs, "diff_ci_lo": float(lo), "diff_ci_hi": float(hi),
            "n_blocks": len(names), "sig": bool(lo > 0 or hi < 0)}


def blocks(a, b, labels):
    bd = defaultdict(list)
    for q in a:
        blk = labels[int(q)]
        if blk:
            bd[blk].append(a[q] - b[q])
    return bd


def main():
    ents = torch.load(resolve("files/models/db.pt"), map_location="cpu", weights_only=False)["entries"]
    lab = {"family": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"))),
           "clan": attach_labels(ents, load_labels(resolve("Splits/labels.tsv"), label_column="clan_acc"))}
    out = {"_n_replicates": B, "_seed": SEED,
           "_test": "paired block bootstrap over Rfam families (family) or clans (clan)"}

    out["alphabet_control"] = {}
    for regime, f in ALPHABET.items():
        d = json.load(open(resolve("files/ablation/" + f)))
        R, S = d["RFoldSearch_tokens_same_pipeline"], d["RiboSeek_sa20_same_pipeline"]
        out["alphabet_control"][regime] = {}
        for level in ("family", "clan"):
            a = {x["query"]: x["sens"] for x in R[level]}
            b = {x["query"]: x["sens"] for x in S[level]}
            out["alphabet_control"][regime][level] = paired_ci(blocks(a, b, lab[level]), LEVEL_SEED[level])

    pq = {}
    for f in QUANTISER:
        pq.update({k: v for k, v in json.load(open(resolve("files/ablation/" + f))).items()
                   if k != "_provenance"})
    out["quantiser_control"] = {}
    V = "VQ-VAE_K80_deployed"
    for K in (10, 20, 40, 80):
        m = f"KMeans_K{K}"
        out["quantiser_control"][m] = {}
        for level in ("family", "clan"):
            a = {x["query"]: x["sens"] for x in pq[V][level]}
            b = {x["query"]: x["sens"] for x in pq[m][level]}
            out["quantiser_control"][m][level] = paired_ci(blocks(a, b, lab[level]), LEVEL_SEED[level])

    dst = resolve("files/production/retrieval/bootstrap_controls.json")
    json.dump(out, open(dst, "w"), indent=2)
    print(f"wrote {dst}\n")
    for regime, lv in out["alphabet_control"].items():
        for level, e in lv.items():
            print("alphabet %-12s %-7s blocks=%-3d diff=%+.4f CI=[%+.4f,%+.4f] sig=%s"
                  % (regime, level, e["n_blocks"], e["diff_mean"], e["diff_ci_lo"], e["diff_ci_hi"], e["sig"]))
    for m, lv in out["quantiser_control"].items():
        for level, e in lv.items():
            print("quantiser %-12s %-7s blocks=%-3d diff=%+.4f CI=[%+.4f,%+.4f] sig=%s"
                  % (m, level, e["n_blocks"], e["diff_mean"], e["diff_ci_lo"], e["diff_ci_hi"], e["sig"]))


if __name__ == "__main__":
    main()
