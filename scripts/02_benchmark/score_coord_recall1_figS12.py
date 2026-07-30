"""Add GTalign and RTM-align Recall@1 to recall1_rfam_test.json (Fig S12).

Same definition as eval_recall1_figS12.py: the top-ranked non-self target is a
true positive; family TP = same Rfam family, clan TP = same Rfam clan (including
same family). Both coordinate aligners are ranked by symmetric TM (mean of the
two length-normalisations), consistent with Fig 2A/B.

Prerequisite: the GTalign / RTM-align all-versus-database outputs (see
score_gtalign_fig2ab.py / score_rtmalign_fig2ab.py). Point RFS_GT_OUT / RFS_RTM_OUT
at the directories of *.out / *.tab files (defaults /tmp/gt_final, /tmp/rtm_out).
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

from prefilter_rfam_benchmark import attach_labels, load_labels

RET = "files/production/retrieval"
GT_OUT = _os.environ.get("RFS_GT_OUT", "/tmp/gt_final")
RTM_OUT = _os.environ.get("RFS_RTM_OUT", "/tmp/rtm_out")
db = torch.load("files/models/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
base2i = {_os.path.basename(str(e["id"])).replace(".cif", ""): i for i, e in enumerate(ents)}
fam = np.array(attach_labels(ents, load_labels("Splits/labels.tsv")), dtype=object)
clan = np.array(attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc")), dtype=object)
pq = json.load(open(f"{RET}/per_query_sens_test.json"))
fam_q = pq["_query_ids"]["family"]
clan_q = pq["_query_ids"]["clan"]

row_re = re.compile(r"PDB_[0-9A-Za-z_]+\.cif")


def rank_gtalign(path):
    rows = []
    for line in open(path):
        m = row_re.search(line)
        if not m:
            continue
        nums = re.findall(r"\d+\.\d+", line)
        if len(nums) < 2:
            continue
        ti = base2i.get(m.group(0).replace(".cif", ""))
        if ti is not None:
            rows.append((ti, 0.5 * (float(nums[0]) + float(nums[1]))))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return [ti for ti, _ in rows]


def rank_rtm(path):
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
            rows.append((ti, 0.5 * (float(p[2]) + float(p[3]))))
        except ValueError:
            continue
    rows.sort(key=lambda x: (-x[1], x[0]))
    return [ti for ti, _ in rows]


def collect(pattern, parse, keyfn):
    ranks = {}
    for f in glob.glob(pattern):
        qi = keyfn(f)
        if qi is None:
            continue
        ranks[qi] = np.asarray([t for t in parse(f) if t != qi], dtype=np.int64)
    return ranks


RANKS = {
    "GTalign": collect(f"{GT_OUT}/*.out", rank_gtalign,
                       lambda f: base2i.get(_os.path.basename(f).split("__")[0])),
    "RTM-align": collect(f"{RTM_OUT}/*.tab", rank_rtm,
                         lambda f: base2i.get(_os.path.basename(f).replace(".tab", ""))),
}


def r1_family(r, q):
    for t in r:
        if t == q:
            continue
        return 1 if fam[t] == fam[q] else 0
    return 0


def r1_clan(r, q):
    for t in r:
        if t == q:
            continue
        return 1 if (clan[q] and clan[t] == clan[q]) else 0
    return 0


s12 = json.load(open(f"{RET}/recall1_rfam_test.json"))
cq = [q for q in clan_q if clan[q]]
for m, ranks in RANKS.items():
    fr = float(np.mean([r1_family(ranks.get(q, np.zeros(0, np.int64)), q) for q in fam_q]))
    cr = float(np.mean([r1_clan(ranks.get(q, np.zeros(0, np.int64)), q) for q in cq]))
    s12[m] = {"family_recall1": fr, "clan_recall1": cr, "n_family": len(fam_q), "n_clan": len(cq)}
    print(f"  {m:10s} famR@1={fr:.3f}  clanR@1={cr:.3f}  (parsed {len(ranks)} queries)")
json.dump(s12, open(f"{RET}/recall1_rfam_test.json", "w"), indent=2)
print("merged GTalign + RTM-align into recall1_rfam_test.json\nDONE")
