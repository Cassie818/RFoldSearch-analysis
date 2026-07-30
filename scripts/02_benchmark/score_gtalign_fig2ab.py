"""Score GTalign for Fig 2A/B and merge it into per_query_sens_test.json.

GTalign is a GPU-optimised coordinate-level aligner; like RTM-align it is shown
only in Fig 2A/B (near-oracle under the structure-defined ground truth).

Prerequisite: run GTalign as an all-versus-database search first, e.g.
    gtalign --qrs=<queries> --rfs=<targets> -s 0 --pre-score=0 --speed=8 -o <out_dir>
one .out ranked table per query. Point RFS_GT_OUT at the directory of *.out
files (default: /tmp/gt_final). Targets are ranked by the symmetric TM-score
(mean of the reference- and query-length normalisations).
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_os.chdir(DATA_ROOT)
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import json
import re
import numpy as np
import torch

from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels

RET = "files/production/retrieval"
GT_OUT = _os.environ.get("RFS_GT_OUT", "/tmp/gt_final")
db = torch.load("files/models/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
n = len(ents)
base2i = {_os.path.basename(str(e["id"])).replace(".cif", ""): i for i, e in enumerate(ents)}
fam = np.array(attach_labels(ents, load_labels("Splits/labels.tsv")), dtype=object)
clan = np.array(attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc")), dtype=object)

pq = json.load(open(f"{RET}/per_query_sens_test.json"))
fam_q = pq["_query_ids"]["family"]
clan_q = pq["_query_ids"]["clan"]
va_all = np.arange(n)

row_re = re.compile(r"PDB_[0-9A-Za-z_]+\.cif")


def parse_out(path):
    """Ranked db indices by symmetric TM = mean(refTM, queryTM)."""
    rows = []
    for line in open(path):
        m = row_re.search(line)
        if not m:
            continue
        nums = re.findall(r"\d+\.\d+", line)
        if len(nums) < 2:
            continue
        ti = base2i.get(m.group(0).replace(".cif", ""))
        if ti is None:
            continue
        reftm, qtm = float(nums[0]), float(nums[1])
        rows.append((ti, 0.5 * (reftm + qtm)))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return [ti for ti, _ in rows]


ranks = {}
for f in glob.glob(f"{GT_OUT}/*.out"):
    qi = base2i.get(_os.path.basename(f).split("__")[0])
    if qi is None:
        continue
    ranks[qi] = np.asarray([t for t in parse_out(f) if t != qi], dtype=np.int64)
print(f"parsed {len(ranks)} GTalign query outputs from {GT_OUT}", flush=True)

fam_sens, clan_sens = [], []
for q in fam_q:
    r = ranks.get(q, np.zeros(0, np.int64))
    va = va_all != q
    rel = (fam == fam[q]) & va
    neu = ((clan == clan[q]) & (fam != fam[q]) & va) if clan[q] else None
    fam_sens.append(first_fp_sensitivity(r, rel, neu) if len(r) else 0.0)
for q in clan_q:
    r = ranks.get(q, np.zeros(0, np.int64))
    va = va_all != q
    rel = (clan == clan[q]) & (fam != fam[q]) & va
    neu = (fam == fam[q]) & va
    clan_sens.append(first_fp_sensitivity(r, rel, neu) if len(r) else 0.0)

entry = {
    "family": {"sens": fam_sens, "n": len(fam_sens), "mean": float(np.mean(fam_sens))},
    "clan": {"sens": clan_sens, "n": len(clan_sens), "mean": float(np.mean(clan_sens))},
}
pq["GTalign"] = entry
json.dump(pq, open(f"{RET}/per_query_sens_test.json", "w"), indent=2)
print(f"GTalign  family={entry['family']['mean']:.4f}  clan={entry['clan']['mean']:.4f}")
print("merged into per_query_sens_test.json\nDONE", flush=True)
