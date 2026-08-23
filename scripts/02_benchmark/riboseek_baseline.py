"""RiboSeek (real PyPI package riboseek==0.1.0, sa20 alphabet) baseline on
TEST-only queries (n=106 fam / 81 clan), full 885 db. Uses the package's own
pdb_to_features + sa20 alphabet + Searcher (contiguous 6-mer prefilter,
prefilter_candidates=500, NW+SW z-score combined_score). Foldseek hierarchical
family/clan definitions, matching our other test-only evals."""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
from pathlib import Path
import os, sys, json, time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np, torch, importlib.resources as R
from collections import defaultdict
from riboseek import Searcher
from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels

db = torch.load("files/production/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
N = len(ents)
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
fams = attach_labels(ents, load_labels("Splits/labels.tsv"))
clans = attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc"))


def idl(p):
    return [str(json.loads(l)["id"]) for l in open(p) if l.strip()]


test_dis = set(id2i[c] for c in idl("Splits/pdb_out/test.json") if c in id2i)
with R.as_file(R.files("riboseek.data").joinpath("demo_db.json")) as p:
    demo = str(p)
sr0 = Searcher.from_pretrained("sa20", db=demo)
alpha = sr0.alphabet
CACHE = "files/ablation/riboseek_real_encoded.json"
if os.path.exists(CACHE):
    enc = json.load(open(CACHE))
    print(f"loaded cached encodings ({len(enc)} chains)", flush=True)
else:
    print(
        f"sa20 alphabet loaded ({alpha.score_matrix.shape[0]} states); encoding {N} chains...",
        flush=True,
    )
    t0 = time.time()
    enc = {}
    for i, e in enumerate(ents):
        try:
            lab = np.asarray(sr0.encode(str(e["id"])), dtype=np.int32)
            enc[str(i)] = {"labels": lab.tolist(), "length": int(len(lab))}
        except Exception as ex:
            enc[str(i)] = {"labels": [], "length": 0}
        if (i + 1) % 150 == 0:
            print(f"  {i+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    json.dump(enc, open(CACHE, "w"))
    print(f"encoded ({time.time()-t0:.0f}s, cached); building Searcher over our db...", flush=True)
sr = Searcher(alpha, enc, gap_penalty=-2.0, prefilter_k=6)
fam_pool = np.array(fams, dtype=object)
clan_pool = np.array(clans, dtype=object)
famcnt = defaultdict(int)
for l in fams:
    if l:
        famcnt[l] += 1
fam_q = [
    q
    for q in sorted(test_dis)
    if fams[q] and famcnt[fams[q]] >= 2 and len(enc[str(q)]["labels"]) >= 6
]
print(f"test family-evaluable queries: {len(fam_q)}", flush=True)
va_all = np.arange(N)
r1f = []
famS = []
r1c = []
clanS = []
t1 = time.time()
for qi, q in enumerate(fam_q):
    res = sr.search(str(q), top_n=500, prefilter=True, prefilter_candidates=500)
    ranked = [int(h["chain"]) for h in res if int(h["chain"]) != q]
    r = np.asarray(ranked, dtype=np.int64)
    va = va_all != q
    if len(r) == 0:
        r1f.append(0)
        famS.append(0.0)
    else:
        rel_f = (fam_pool == fam_pool[q]) & va
        neu_f = (
            ((clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va) if clan_pool[q] else None
        )
        r1f.append(int(fam_pool[r[0]] == fam_pool[q]))
        famS.append(first_fp_sensitivity(r, rel_f, neu_f))
    if clan_pool[q]:
        if len(r) > 0:
            r1c.append(int(clan_pool[r[0]] == clan_pool[q]))
        rel_c = (clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va
        if rel_c.any():
            clanS.append(
                first_fp_sensitivity(
                    r if len(r) > 0 else np.zeros(0, np.int64),
                    rel_c,
                    (fam_pool == fam_pool[q]) & va,
                )
            )
    if (qi + 1) % 30 == 0:
        print(f"  searched {qi+1}/{len(fam_q)} ({time.time()-t1:.0f}s)", flush=True)


def m(x):
    return float(np.mean(x)) if x else 0.0


print(
    "\n=== RiboSeek (REAL package, sa20) on TEST (n_fam=%d, n_clan=%d) ==="
    % (len(famS), len(clanS)),
    flush=True,
)
print(f"  Family Recall@1 = {m(r1f):.3f}", flush=True)
print(f"  Family Sens@1FP = {m(famS):.3f}", flush=True)
print(f"  Clan   Recall@1 = {m(r1c):.3f}", flush=True)
print(f"  Clan   Sens@1FP = {m(clanS):.3f}", flush=True)
json.dump(
    {
        "fam_r1": m(r1f),
        "fam_sens": m(famS),
        "clan_r1": m(r1c),
        "clan_sens": m(clanS),
        "n_fam": len(famS),
        "n_clan": len(clanS),
        "source": "real riboseek 0.1.0 sa20",
    },
    open("files/ablation/riboseek_test_real.json", "w"),
    indent=2,
)
print("DONE", flush=True)
