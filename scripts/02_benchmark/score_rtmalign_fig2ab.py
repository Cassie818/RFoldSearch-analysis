"""Score RTM-align for Fig 2A/B and merge it into per_query_sens_test.json.

RTM-align is a coordinate-level RNA aligner (an RNA-tuned TM-align variant); it
is shown only in Fig 2A/B, since under the structure-defined ground truth it
ranks by its own TM-like score and would be a near-oracle.

Prerequisite: run RTM-align as an all-versus-database search first, e.g.
    RTMalign <query>.cif -dir2 <targets_dir>/ <target_list> -mol RNA -outfmt 2 -o out.tab
one tabular output per query. Point RFS_RTM_OUT at the directory of *.tab files
(default: /tmp/rtm_out). Targets are ranked by the symmetric RTM-score
(mean of the two length-normalisations, columns TM1/TM2), matching the
convention used for every other method.
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_os.chdir(DATA_ROOT)
_os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import glob
import json
import numpy as np
import torch

from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels

RET = "files/production/retrieval"
RTM_OUT = _os.environ.get("RFS_RTM_OUT", "/tmp/rtm_out")
db = torch.load("files/production_new/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
n = len(ents)
base2i = {_os.path.basename(str(e["id"])).replace(".cif", ""): i for i, e in enumerate(ents)}
fam = np.array(attach_labels(ents, load_labels("Splits/labels.tsv")), dtype=object)
clan = np.array(attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc")), dtype=object)

pq = json.load(open(f"{RET}/per_query_sens_test.json"))
fam_q = pq["_query_ids"]["family"]
clan_q = pq["_query_ids"]["clan"]
va_all = np.arange(n)


def parse_tab(path):
    """Ranked db indices by symmetric RTM-score = mean(TM1, TM2)."""
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) < 4:
            continue
        ti = base2i.get(_os.path.basename(p[1]).replace(".cif", ""))
        if ti is None:
            continue
        try:
            tm1, tm2 = float(p[2]), float(p[3])
        except ValueError:
            continue
        rows.append((ti, 0.5 * (tm1 + tm2)))
    rows.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break
    return [ti for ti, _ in rows]


ranks = {}
for f in glob.glob(f"{RTM_OUT}/*.tab"):
    qi = base2i.get(_os.path.basename(f).replace(".tab", ""))
    if qi is None:
        continue
    ranks[qi] = np.asarray([t for t in parse_tab(f) if t != qi], dtype=np.int64)
print(f"parsed {len(ranks)} RTM-align query outputs from {RTM_OUT}", flush=True)

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
# archive + merge into the shared per-query file (preserving all other methods)
json.dump({"family": entry["family"], "clan": entry["clan"],
           "note": "RTM-align, -mol RNA, symmetric RTM-score"},
          open(f"{RET}/rtmalign_test.json", "w"), indent=2)
pq["RTM-align"] = entry
json.dump(pq, open(f"{RET}/per_query_sens_test.json", "w"), indent=2)
print(f"RTM-align  family={entry['family']['mean']:.4f}  clan={entry['clan']['mean']:.4f}")
print("merged into per_query_sens_test.json\nDONE", flush=True)
