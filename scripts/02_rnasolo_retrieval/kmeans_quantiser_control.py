"""K-means vs VQ-VAE quantiser control, identical descriptor + deployed pipeline.

Isolates the contribution of the LEARNED quantiser from the descriptor design:
the same 17-D production descriptor is tokenised either by the deployed VQ-VAE
(K=80) or by K-means (K=80, K=20), and all three alphabets are then searched with
the identical deployed pipeline (expansion prefilter, top-200, affine SW) using
their own Structure-BLOSUM matrix built from the same training pairs.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import numpy as np, torch
from sklearn.cluster import MiniBatchKMeans

PC = os.path.expanduser("~/Desktop/RFoldseek-paper-code")
sys.path.insert(0, os.path.join(PC, "scripts", "02_rnasolo_retrieval"))
os.environ["RFS_ALPHABET_CONTROL_MATRIX"] = "self_blosum"
import riboseek_alphabet_control as F  # noqa: E402

SPLITS = os.path.expanduser("~/Desktop/RFoldSeek/Splits/pdb_out")
KEEP = sorted(list(range(0, 10)) + list(range(42, 49)))  # g_nn1 (0-9) + g_ctx (42-48)
SEED = 42
KS = [int(x) for x in os.environ.get("KM_KS", "80,20").split(",")]
INC_VQ = os.environ.get("KM_SKIP_VQ", "") != "1"
SUF = os.environ.get("KM_SUFFIX", "")

def main() -> None:
    db = torch.load(F.resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
    ents = db["entries"]
    lab = F.resolve("Splits/labels.tsv")
    families = F.attach_labels(ents, F.load_labels(lab))
    clans = F.attach_labels(ents, F.load_labels(lab, label_column="clan_acc"))
    fam_q, clan_q = F.load_query_sets(ents, families, clans)
    print(f"queries: family={len(fam_q)} clan={len(clan_q)}", flush=True)

    id2i = {os.path.basename(str(e["id"])): i for i, e in enumerate(ents)}
    raw, split_of = {}, {}
    for sp in ["train", "val", "test"]:
        D = torch.load(f"{SPLITS}/{sp}_desc.pt", map_location="cpu", weights_only=False).numpy()
        off = 0
        for line in open(f"{SPLITS}/{sp}.json"):
            line = line.strip()
            if not line:
                continue
            di = id2i[os.path.basename(json.loads(line)["id"])]
            L = len(ents[di]["tokens"])
            raw[di] = D[off:off + L].astype(np.float32)[:, KEEP]
            split_of[di] = sp
            off += L
        assert off == len(D), f"{sp}: {off} != {len(D)}"
    print(f"descriptors loaded for {len(raw)} chains, {len(KEEP)}-D active", flush=True)

    # z-score from TRAIN residues only (no leakage into val/test)
    tr = np.concatenate([raw[i] for i in raw if split_of[i] == "train"])
    mn, sd = tr.mean(0), tr.std(0)
    sd[sd < 1e-6] = 1.0
    norm = {i: (raw[i] - mn) / sd for i in raw}
    tr_norm = np.concatenate([norm[i] for i in norm if split_of[i] == "train"])
    print(f"k-means fit pool: {tr_norm.shape[0]} training residues", flush=True)

    configs = {}
    vq = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
    if INC_VQ:
        configs["VQ-VAE_K80_deployed"] = (vq, 80)
    for K in KS:
        km = MiniBatchKMeans(n_clusters=K, batch_size=10000, max_iter=200,
                             random_state=SEED, n_init=3).fit(tr_norm)
        toks = [np.asarray(km.predict(norm[i]), dtype=np.int64) if len(norm[i]) else np.zeros(0, np.int64)
                for i in range(len(ents))]
        configs[f"KMeans_K{K}"] = (toks, K)
        print(f"k-means K={K} fitted; inertia={km.inertia_:.1f}", flush=True)

    pairs, pair_meta = F.load_train_pairs(ents)
    print(f"self-BLOSUM pool: {len(pairs)} train pairs (TM>={F.BLOSUM_TM_THRESH})", flush=True)

    summaries, per_query = {}, {}
    for name, (toks, K) in configs.items():
        mat, meta = F.build_self_blosum(toks, pairs, K)
        print(f"\n{name}: alphabet={K}", flush=True)
        s, pq = F.evaluate_ranker(name, F.SamePipelineRanker(toks, mat), fam_q, clan_q, families, clans)
        s.update({"alphabet_size": K, "quantiser": "vq-vae" if "VQ" in name else "k-means",
                  "matrix": "self-BLOSUM (train pairs, TM>=0.5)", "self_blosum": meta})
        summaries[name], per_query[name] = s, pq
        print(json.dumps({k: v for k, v in s.items() if k != "self_blosum"}, indent=2), flush=True)

    out = Path(PC) / "files/ablation"
    prov = {"pair_pool": pair_meta, "kmeans_seed": SEED, "active_dims": KEEP,
            "zscore": "train-split residues only"}
    json.dump({"_provenance": prov, **summaries}, open(out / f"kmeans_quantiser_control{SUF}.json", "w"), indent=2)
    json.dump({"_provenance": prov, **per_query}, open(out / f"kmeans_quantiser_control{SUF}_per_query.json", "w"), indent=2)
    print(f"\nWrote {out}/kmeans_quantiser_control{SUF}.json", flush=True)

if __name__ == "__main__":
    main()
