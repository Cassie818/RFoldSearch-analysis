"""K-means vs VQ-VAE quantiser control under the STRUCTURE-DEFINED ground truth."""
from __future__ import annotations
import json, os, sys, time
from collections import defaultdict
import numpy as np, torch
from sklearn.cluster import MiniBatchKMeans

PC = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, os.path.join(PC, "scripts", "02_rnasolo_retrieval", "controls"))
os.environ["RFS_ALPHABET_CONTROL_MATRIX"] = "self_blosum"
import riboseek_alphabet_control as F
sys.path.insert(0, os.path.join(PC, "src", "search"))
from prefilter_rfam_benchmark import first_fp_sensitivity

SPLITS = F.SPLITS_DESC
KEEP = sorted(list(range(0, 10)) + list(range(42, 49)))
SEED = 42
KS = [int(x) for x in os.environ.get("KM_KS", "10,20,40,80").split(",")]
B, BOOT_SEED = 2000, 123
LEVEL_SEED = {"family": BOOT_SEED + 1, "clan": BOOT_SEED + 2}
THRESHOLDS = [0.40, 0.45, 0.50]

db = torch.load(F.resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
ents = db["entries"]
n = len(ents)
lab = F.resolve("Splits/labels.tsv")
fam = F.attach_labels(ents, F.load_labels(lab))
clan = F.attach_labels(ents, F.load_labels(lab, label_column="clan_acc"))
fam_arr = np.array(fam, dtype=object)

# ---- structure-defined ground truth (same TM cache as Fig 2C,D) -------------
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
test_ids = [str(json.loads(l)["id"]) for l in open(F.resolve("Splits/pdb_out/test.json")) if l.strip()]
test_dis = set(id2i[c] for c in test_ids if c in id2i)
tm = np.full((n, n), np.nan, dtype=np.float32)
cov = defaultdict(int)
for line in open(F.resolve("files/production/retrieval/usalign_full.cache.jsonl")):
    if not line.strip():
        continue
    r = json.loads(line)
    i, j = r["i"], r["j"]
    a = 0.5 * (r["tm_query"] + r["tm_target"])
    tm[i, j] = a; tm[j, i] = a; cov[i] += 1; cov[j] += 1
queries = sorted(q for q in range(n) if cov[q] >= n - 5 and q in test_dis)
print(f"test covered queries: {len(queries)}", flush=True)

# ---- descriptors, z-scored on train residues (identical to Rfam control) ----
id2base = {os.path.basename(str(e["id"])): i for i, e in enumerate(ents)}
raw, split_of = {}, {}
for sp in ("train", "val", "test"):
    D = torch.load(f"{SPLITS}/{sp}_desc.pt", map_location="cpu", weights_only=False).numpy()
    off = 0
    for line in open(f"{SPLITS}/{sp}.json"):
        line = line.strip()
        if not line:
            continue
        di = id2base[os.path.basename(json.loads(line)["id"])]
        L = len(ents[di]["tokens"])
        raw[di] = D[off:off + L].astype(np.float32)[:, KEEP]
        split_of[di] = sp; off += L
tr = np.concatenate([raw[i] for i in raw if split_of[i] == "train"])
mn, sd = tr.mean(0), tr.std(0); sd[sd < 1e-6] = 1.0
norm = {i: (raw[i] - mn) / sd for i in raw}
tr_norm = np.concatenate([norm[i] for i in norm if split_of[i] == "train"])

configs = {"VQ-VAE_K80_deployed": ([np.asarray(e["tokens"], dtype=np.int64) for e in ents], 80)}
for K in KS:
    km = MiniBatchKMeans(n_clusters=K, batch_size=10000, max_iter=200,
                         random_state=SEED, n_init=3).fit(tr_norm)
    configs[f"KMeans_K{K}"] = (
        [np.asarray(km.predict(norm[i]), dtype=np.int64) if len(norm[i]) else np.zeros(0, np.int64)
         for i in range(n)], K)
    print(f"k-means K={K} fitted", flush=True)

pairs, _ = F.load_train_pairs(ents)

# ---- per-query ranking for each alphabet, then structure-defined scoring ----
def score_ranks(ranks, th, level):
    per = {}
    va_all = np.arange(n)
    for q in queries:
        va = va_all != q
        pos = (~np.isnan(tm[q])) & (tm[q] >= th) & va
        r = ranks[q]
        if level == "family":
            if not pos.any():
                continue
            per[q] = 0.0 if len(r) == 0 else first_fp_sensitivity(r, pos, None)
        else:
            if not fam_arr[q]:
                continue
            rel_c = pos & (fam_arr != fam_arr[q])
            if not rel_c.any():
                continue
            neu_c = (fam_arr == fam_arr[q]) & va
            per[q] = first_fp_sensitivity(r if len(r) else np.zeros(0, np.int64), rel_c, neu_c)
    return per


t0 = time.time()
PER = {}   # config -> threshold -> level -> {q: sens}
for name, (toks, K) in configs.items():
    mat, _ = F.build_self_blosum(toks, pairs, K)
    ranker = F.SamePipelineRanker(toks, mat, xyz=[F.c4_coords(e, len(e["tokens"])) for e in ents])
    ranks = {q: np.asarray(ranker.rank(q), dtype=np.int64) for q in queries}
    PER[name] = {f"{th}": {lv: score_ranks(ranks, th, lv) for lv in ("family", "clan")}
                 for th in THRESHOLDS}
    m45 = PER[name]["0.45"]
    print(f"{name}: fam@0.45={np.mean(list(m45['family'].values())):.3f} "
          f"clan@0.45={np.mean(list(m45['clan'].values())):.3f}  [{time.time()-t0:.0f}s]", flush=True)


def ci(a, b, level, seed):
    bd = defaultdict(list)
    labs = fam if level == "family" else clan
    for q in a:
        blk = labs[int(q)]
        if blk:
            bd[blk].append(a[q] - b.get(q, 0.0))
    names = list(bd)
    if not names:
        return None
    rng = np.random.default_rng(seed)
    reps = [rng.choice(names, size=len(names), replace=True) for _ in range(B)]
    dm = np.array([np.mean([v for x in s for v in bd[x]]) for s in reps])
    lo, hi = np.percentile(dm, [2.5, 97.5])
    return {"diff_mean": float(np.mean([v for x in names for v in bd[x]])),
            "diff_ci_lo": float(lo), "diff_ci_hi": float(hi),
            "n_blocks": len(names), "sig": bool(lo > 0 or hi < 0)}


out = {"_config": {"B": B, "seed": BOOT_SEED, "thresholds": THRESHOLDS,
                   "n_queries": len(queries), "anchor": "VQ-VAE_K80_deployed",
                   "_note": "diff = VQ - k-means; structure-defined ground truth; Rfam blocks"},
       "thresholds": {}}
anchor = "VQ-VAE_K80_deployed"
for th in THRESHOLDS:
    tk = f"{th}"; lv_out = {}
    for level in ("family", "clan"):
        means = {c: float(np.mean(list(PER[c][tk][level].values()))) for c in PER}
        diffs = {c: ci(PER[anchor][tk][level], PER[c][tk][level], level, LEVEL_SEED[level])
                 for c in PER if c != anchor}
        lv_out[level] = {"means": means, "vs_vqvae": diffs}
        print(f"th={th} {level}: " + "  ".join(f"{c.split('_')[0]}={means[c]:.3f}" for c in PER))
        for c, e in diffs.items():
            print(f"    {c:12s} diff={e['diff_mean']:+.3f} "
                  f"[{e['diff_ci_lo']:+.3f},{e['diff_ci_hi']:+.3f}] sig={e['sig']}")
    out["thresholds"][tk] = lv_out

dst = F.resolve("files/ablation/quantiser_structural.json")
json.dump(out, open(dst, "w"), indent=1)
print(f"\nwrote {dst}  [{time.time()-t0:.0f}s]")
