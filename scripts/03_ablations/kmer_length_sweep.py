"""Seed length k against alphabet size K.

Foldseek uses 20 states with long seeds; RFoldseek uses 80 with k=4. What
matters for prefilter sparsity is K^k, so the two are exchangeable in principle.
This sweeps k for the deployed 80-state alphabet and for a 20-state k-means
alphabet on the identical descriptors, recording retrieval sensitivity, the
postings traversed per query, and wall time.
"""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
import numpy as np, torch
from sklearn.cluster import MiniBatchKMeans

PC = os.path.expanduser("~/Desktop/RFoldseek-paper-code")
sys.path.insert(0, os.path.join(PC, "scripts", "03_ablations"))
os.environ.setdefault("RFS_ALPHABET_CONTROL_MATRIX", "native")
import fig2ef as F  # noqa: E402

SPLITS = os.path.expanduser("~/Desktop/RFoldSeek/Splits/pdb_out")
KEEP = sorted(list(range(0, 10)) + list(range(42, 49)))
CONFIGS = [("VQ-VAE_K80", k) for k in (3, 4, 5, 6)] + [("KMeans_K20", k) for k in (4, 5, 6)]


def postings_per_query(toks, K, k, queries):
    powers = np.array([K ** (k - 1 - i) for i in range(k)], dtype=np.int64)
    index = defaultdict(list)
    for ti, seq in enumerate(toks):
        for pos in range(len(seq) - k + 1):
            index[int(np.dot(seq[pos:pos + k], powers))].append((ti, pos))
    tot, surv = [], []
    for q in queries:
        seq = toks[q]
        dg = defaultdict(dict); n = 0
        for qp in range(len(seq) - k + 1):
            for ti, tp in index.get(int(np.dot(seq[qp:qp + k], powers)), ()):
                if ti == q: continue
                n += 1; d = qp - tp; dg[ti][d] = dg[ti].get(d, 0) + 1
        tot.append(n); surv.append(sum(1 for c in dg.values() if max(c.values()) >= 2))
    return float(np.mean(tot)), float(np.mean(surv))


def main():
    db = torch.load(F.resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
    ents = db["entries"]
    lab = F.resolve("Splits/labels.tsv")
    fam = F.attach_labels(ents, F.load_labels(lab))
    clan = F.attach_labels(ents, F.load_labels(lab, label_column="clan_acc"))
    fam_q, clan_q = F.load_query_sets(ents, fam, clan)

    id2i = {os.path.basename(str(e["id"])): i for i, e in enumerate(ents)}
    raw, split_of = {}, {}
    for sp in ("train", "val", "test"):
        D = torch.load(f"{SPLITS}/{sp}_desc.pt", map_location="cpu", weights_only=False).numpy()
        off = 0
        for line in open(f"{SPLITS}/{sp}.json"):
            line = line.strip()
            if not line: continue
            di = id2i[os.path.basename(json.loads(line)["id"])]
            L = len(ents[di]["tokens"]); raw[di] = D[off:off + L].astype(np.float32)
            split_of[di] = sp; off += L
    tr = np.concatenate([raw[i][:, KEEP] for i in raw if split_of[i] == "train"])
    mn, sd = tr.mean(0), tr.std(0); sd[sd < 1e-6] = 1.0
    norm = {i: ((raw[i][:, KEEP] - mn) / sd).astype(np.float64) for i in raw}
    km = MiniBatchKMeans(n_clusters=20, batch_size=10000, max_iter=200,
                         random_state=42, n_init=3).fit(
        ((tr - mn) / sd).astype(np.float64))

    alpha = {"VQ-VAE_K80": ([np.asarray(e["tokens"], dtype=np.int64) for e in ents], 80),
             "KMeans_K20": ([km.predict(norm[i]).astype(np.int64) if len(norm[i]) else np.zeros(0, np.int64)
                             for i in range(len(ents))], 20)}
    pairs, _ = F.load_train_pairs(ents)
    mats = {n: F.build_self_blosum(t, pairs, K)[0] for n, (t, K) in alpha.items()}
    print("alphabets and matrices ready", flush=True)

    out = {}
    print("\n%-12s %3s %10s %9s %9s %14s %11s %9s" %
          ("alphabet", "k", "K^k", "famSens", "clanSens", "postings/query", "survivors", "sec"))
    for name, k in CONFIGS:
        toks, K = alpha[name]
        t0 = time.time()
        s, _ = F.evaluate_ranker(f"{name}|k{k}", F.SamePipelineRanker(toks, mats[name], kmer_k=k),
                                 fam_q, clan_q, fam, clan)
        el = time.time() - t0
        post, surv = postings_per_query(toks, K, k, fam_q)
        out[f"{name}|k{k}"] = {"alphabet": name, "K": K, "k": k, "space": K ** k,
                               "family_sens_1fp": s["family_sens_1fp"], "clan_sens_1fp": s["clan_sens_1fp"],
                               "postings_per_query": post, "survivors": surv, "sec": el}
        print("%-12s %3d %10.3g %9.3f %9.3f %14.0f %11.1f %9.1f" %
              (name, k, K ** k, s["family_sens_1fp"], s["clan_sens_1fp"], post, surv, el), flush=True)
    json.dump(out, open(f"{PC}/files/ablation/kmer_length_sweep.json", "w"), indent=2)
    print(f"\nWrote {PC}/files/ablation/kmer_length_sweep.json")


if __name__ == "__main__":
    main()
