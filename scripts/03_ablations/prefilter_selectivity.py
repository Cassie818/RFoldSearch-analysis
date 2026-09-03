"""Does a larger alphabet actually make the prefilter more selective?

Counts, per query, the postings traversed and the candidates surviving the
double-diagonal rule, for the deployed 80-state VQ-VAE alphabet and for k-means
alphabets of several sizes on the identical descriptors.
"""
import json, os, sys
from collections import defaultdict
import numpy as np, torch
PC = os.path.expanduser("~/Desktop/RFoldseek-paper-code")
sys.path.insert(0, os.path.join(PC, "scripts", "03_ablations"))
os.environ.setdefault("RFS_ALPHABET_CONTROL_MATRIX", "native")
import fig2ef as F
from sklearn.cluster import MiniBatchKMeans

SPLITS = os.path.expanduser("~/Desktop/RFoldSeek/Splits/pdb_out")
KEEP = sorted(list(range(0, 10)) + list(range(42, 49)))
K_MER = 4
SUFFIX = os.environ.get("SEL_SUFFIX", "")

db = torch.load(F.resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
ents = db["entries"]
lab = F.resolve("Splits/labels.tsv")
fam = F.attach_labels(ents, F.load_labels(lab))
clan = F.attach_labels(ents, F.load_labels(lab, label_column="clan_acc"))
famq, _ = F.load_query_sets(ents, fam, clan)

id2i = {os.path.basename(str(e["id"])): i for i, e in enumerate(ents)}
raw, split_of = {}, {}
for sp in ("train", "val", "test"):
    D = torch.load(f"{SPLITS}/{sp}_desc.pt", map_location="cpu", weights_only=False).numpy()
    off = 0
    for line in open(f"{SPLITS}/{sp}.json"):
        line = line.strip()
        if not line: continue
        di = id2i[os.path.basename(json.loads(line)["id"])]
        L = len(ents[di]["tokens"]); raw[di] = D[off:off+L].astype(np.float32)
        split_of[di] = sp; off += L
tr = np.concatenate([raw[i][:, KEEP] for i in raw if split_of[i] == "train"])
mn, sd = tr.mean(0), tr.std(0); sd[sd < 1e-6] = 1.0
norm = {i: ((raw[i][:, KEEP] - mn) / sd).astype(np.float64) for i in raw}
Xtr = ((tr - mn) / sd).astype(np.float64)

alphabets = {"VQ-VAE_K80": ([np.asarray(e["tokens"], dtype=np.int64) for e in ents], 80)}
SEL_KS = [int(x) for x in os.environ.get("SEL_KS", "20,80").split(",")]
for K in SEL_KS:
    km = MiniBatchKMeans(n_clusters=K, batch_size=10000, max_iter=200, random_state=42, n_init=3).fit(Xtr)
    alphabets[f"KMeans_K{K}"] = ([km.predict(norm[i]).astype(np.int64) if len(norm[i]) else np.zeros(0, np.int64)
                                  for i in range(len(ents))], K)

print("%-14s %6s %14s %16s %14s" % ("alphabet", "K", "postings/query", "distinct targets", "survive >=2"))
out = {}
for name, (toks, K) in alphabets.items():
    powers = np.array([K ** (K_MER - 1 - i) for i in range(K_MER)], dtype=np.int64)
    index = defaultdict(list)
    for ti, seq in enumerate(toks):
        for pos in range(len(seq) - K_MER + 1):
            index[int(np.dot(seq[pos:pos+K_MER], powers))].append((ti, pos))
    post, tgts, surv = [], [], []
    for q in famq:                                     # exact-kmer lookup only
        seq = toks[q]; dg = defaultdict(dict); n = 0
        for qp in range(len(seq) - K_MER + 1):
            for ti, tp in index.get(int(np.dot(seq[qp:qp+K_MER], powers)), ()):
                if ti == q: continue
                n += 1; d = qp - tp; dg[ti][d] = dg[ti].get(d, 0) + 1
        post.append(n); tgts.append(len(dg))
        surv.append(sum(1 for c in dg.values() if max(c.values()) >= 2))
    out[name] = {"K": K, "postings_per_query": float(np.mean(post)),
                 "distinct_targets": float(np.mean(tgts)), "survivors": float(np.mean(surv))}
    print("%-14s %6d %14.0f %16.1f %14.1f" % (name, K, np.mean(post), np.mean(tgts), np.mean(surv)), flush=True)
json.dump(out, open(f"{PC}/files/ablation/prefilter_selectivity{SUFFIX}.json", "w"), indent=2)
print("\nwrote files/ablation/prefilter_selectivity.json")
