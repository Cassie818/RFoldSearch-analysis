"""Rfam family- and clan-level Recall@1 and per-query-averaged AUPR for the
benchmark methods on the TEST partition (both supplementary retrieval metrics).

Recall@1: the top-ranked non-self target is a true positive.
  family: TP=same Rfam family
  clan  : TP=same Rfam clan, including same-family targets

Per-query AUPR: mean average precision over queries (each query weighted
equally, using each method's own hit order); see rfam_per_query_pr_curve.

Both reuse the same per-method rankings. Writes:
  files/production/retrieval/recall1_rfam_test.json
  files/production/retrieval/weighted_pr_rfam_test.json
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
import os, sys, json, re, glob, csv, subprocess, tempfile

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

from prefilter_rfam_benchmark import attach_labels, load_labels, rfam_per_query_pr_curve
from smith_waterman import smith_waterman_numba

K = 80
TOPN = 200  # candidate-pool depth selected on the validation split (see Table 9)
KMER_K = 4
THRESH_FRAC = 0.75
GO = -2.0
GE = -0.5
RET = Path("files/production/retrieval")
ND = "files/models"
db = torch.load(f"{ND}/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
n = len(ents)
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
tags = [os.path.splitext(os.path.basename(str(e["id"])))[0] for e in ents]
fams = attach_labels(ents, load_labels("Splits/labels.tsv"))
clans = attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc"))
fam = np.array(fams, dtype=object)
clan = np.array(clans, dtype=object)

pq = json.load(open(RET / "per_query_sens_test.json"))
fam_q = [i for i in pq["_query_ids"]["family"]]
clan_q = [i for i in pq["_query_ids"]["clan"]]
print(f"fam queries={len(fam_q)}  clan queries={len(clan_q)}", flush=True)
# ===== RFoldseek: token pipeline (k-mer prefilter + SW rerank) =====
toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
sm = torch.load(f"{ND}/sub_matrix.pt", map_location="cpu", weights_only=False)
sub = np.asarray(sm["matrix"].numpy(), dtype=np.float32)  # Structure-BLOSUM
powers = np.array([K ** (KMER_K - 1 - i) for i in range(KMER_K)], dtype=np.int64)
idx = defaultdict(list)
for ti, tt in enumerate(toks):
    for p in range(len(tt) - KMER_K + 1):
        idx[int(np.dot(tt[p : p + KMER_K], powers))].append((ti, p))
smith_waterman_numba(np.zeros(4, np.int64), np.zeros(4, np.int64), sub, GO, GE)
thr = THRESH_FRAC * KMER_K
# Similar k-mers follow the Foldseek/MMseqs2: a one-substitution
# variant is admissible iff the FULL k-mer score under Structure-BLOSUM clears
# `thr`. 
DIAG = np.array([sub[t, t] for t in range(K)], dtype=np.float64)


def expand(qk):
    base = int(np.dot(qk, powers))
    out = [base]
    self_score = float(sum(DIAG[int(qk[i])] for i in range(KMER_K)))
    for pos in range(KMER_K):
        delta = int(powers[pos])
        cur = int(qk[pos])
        base_wo = self_score - DIAG[cur]
        row = sub[cur]
        for tok in range(K):
            if tok == cur:
                continue
            if base_wo + row[tok] >= thr:
                out.append(base + delta * (int(tok) - cur))
    return out


def prefilter_candidates(q):
    """RFoldseek prefilter for query q: similar-k-mer-expansion over the inverted
    index, double-match on a common diagonal, top-N candidates by hit count."""
    tt = toks[q]
    td = defaultdict(dict)
    for qp in range(len(tt) - KMER_K + 1):
        for kid in expand(tt[qp : qp + KMER_K]):
            for ti, tp in idx.get(kid, ()):
                if ti != q:
                    d = qp - tp
                    td[ti][d] = td[ti].get(d, 0) + 1
    sc = [(ti, max((c for c in dm.values() if c >= 2), default=-1)) for ti, dm in td.items()]
    sc = [x for x in sc if x[1] > 0]
    sc.sort(key=lambda x: (-x[1], x[0]))  # deterministic tie-break: score, then target index
    return np.array([x[0] for x in sc[:TOPN]], np.int64)


def rank_new(q):
    """RFoldseek: prefilter candidates reranked by token Smith-Waterman."""
    tt = toks[q]
    cand = prefilter_candidates(q)
    if len(cand) == 0:
        return np.zeros(0, np.int64)
    return cand[
        np.argsort(
            -np.array([smith_waterman_numba(tt, toks[t], sub, GO, GE) for t in cand]), kind="stable"
        )
    ]


def rank_rfoldseek_tm(q):
    """RFoldseek-TM: the same prefilter candidates reranked by US-align TM."""
    d = us.get(q, {})
    rows = [(t, d[t], r) for r, t in enumerate(prefilter_candidates(q)) if t in d and t != q]
    rows.sort(key=lambda x: (-x[1], x[2], x[0]))
    return np.asarray([t for t, _, _ in rows], dtype=np.int64)


from riboseek import Searcher
import importlib.resources as R

enc = json.load(open("files/ablation/riboseek_real_encoded.json"))
with R.as_file(R.files("riboseek.data").joinpath("demo_db.json")) as p:
    demo = str(p)
sr0 = Searcher.from_pretrained("sa20", db=demo)
srr = Searcher(sr0.alphabet, enc, gap_penalty=-2.0, prefilter_k=6)


def rank_ribo(q):
    res = srr.search(str(q), top_n=500, prefilter=True, prefilter_candidates=500)
    return np.asarray([int(h["chain"]) for h in res if int(h["chain"]) != q], dtype=np.int64)


def load_pair(fn, key):
    d = defaultdict(dict)
    for line in open(fn):
        if not line.strip():
            continue
        r = json.loads(line)
        if key not in r:
            continue
        i, j = r["i"], r["j"]
        d[i][j] = r[key]
        d[j][i] = r[key]
    return d


us = {}
for line in open(RET / "usalign_full.cache.jsonl"):
    if not line.strip():
        continue
    r = json.loads(line)
    i, j = r["i"], r["j"]
    a = 0.5 * (r["tm_query"] + r["tm_target"])
    us.setdefault(i, {})[j] = a
    us.setdefault(j, {})[i] = a
rm = load_pair(RET / "rmalign_full.cache.jsonl", "rmscore")


def rank_from(d):
    def f(q):
        dd = d.get(q, {})
        return np.asarray(
            [t for t, _ in sorted(dd.items(), key=lambda kv: -kv[1]) if t != q], dtype=np.int64
        )

    return f


rank_us = rank_from(us)
rank_rm = rank_from(rm)


def rank_blastn_all():
    BIN = Path(os.environ.get("RFS_BLAST_BIN", ""))  
    seqs = {}
    for r in csv.DictReader(open("files/production/chain_sequences.tsv"), delimiter="\t"):
        seqs[r["chain_id"]] = r["sequence"]
    tag2i = {t: k for k, t in enumerate(tags)}
    work = Path(tempfile.mkdtemp())
    fasta = work / "db.fasta"
    with open(fasta, "w") as f:
        for t in tags:
            s = seqs.get(t, "")
            if s:
                f.write(f">{t}\n{s}\n")
    subprocess.run(
        [
            str(BIN / "makeblastdb") if str(BIN) else "makeblastdb",
            "-in",
            str(fasta),
            "-dbtype",
            "nucl",
            "-out",
            str(work / "db"),
        ],
        check=True,
        capture_output=True,
    )
    out = work / "hits.tsv"
    subprocess.run(
        [
            str(BIN / "blastn") if str(BIN) else "blastn",
            "-query",
            str(fasta),
            "-db",
            str(work / "db"),
            "-evalue",
            "10",
            "-out",
            str(out),
            "-outfmt",
            "6 qseqid sseqid bitscore evalue",
            "-num_threads",
            "8",
            "-max_target_seqs",
            str(n),
        ],
        check=True,
        capture_output=True,
    )
    best = defaultdict(dict)
    for line in open(out):
        q, t, bs, _ = line.rstrip("\n").split("\t")
        if q == t or q not in tag2i or t not in tag2i:
            continue
        bs = float(bs)
        qi = tag2i[q]
        ti = tag2i[t]
        if ti not in best[qi] or bs > best[qi][ti]:
            best[qi][ti] = bs
    return best


print("running BLASTN...", flush=True)
import shutil as _shutil
_bb = Path(os.environ.get('RFS_BLAST_BIN', ''))
_have_blast = bool(_shutil.which(str(_bb / 'makeblastdb') if str(_bb) else 'makeblastdb'))
if _have_blast:
    blast = rank_blastn_all()
    rank_bl = rank_from(blast)
else:
    print('makeblastdb not found; keeping the previous BLASTN results', flush=True)


base2i = {os.path.basename(str(e["id"])).replace(".cif", ""): i for i, e in enumerate(ents)}
GT_OUT = os.environ.get("RFS_GT_OUT", "/tmp/gt_final")
RTM_OUT = os.environ.get("RFS_RTM_OUT", "/tmp/rtm_out")
_row_re = re.compile(r"PDB_[0-9A-Za-z_]+\.cif")


def _parse_gtalign(path):
    rows = []
    for line in open(path):
        m = _row_re.search(line)
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


def _parse_rtm(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if len(p) < 4:
            continue
        ti = base2i.get(os.path.basename(p[1]).replace(".cif", ""))
        if ti is None:
            continue
        try:
            rows.append((ti, 0.5 * (float(p[2]) + float(p[3]))))
        except ValueError:
            continue
    rows.sort(key=lambda x: (-x[1], x[0]))
    return [ti for ti, _ in rows]


def _collect(pattern, parse, keyfn):
    r = {}
    for f in glob.glob(pattern):
        qi = keyfn(f)
        if qi is None:
            continue
        r[qi] = np.asarray([t for t in parse(f) if t != qi], dtype=np.int64)
    return r


gt_ranks = _collect(f"{GT_OUT}/*.out", _parse_gtalign,
                    lambda f: base2i.get(os.path.basename(f).split("__")[0]))
rtm_ranks = _collect(f"{RTM_OUT}/*.tab", _parse_rtm,
                     lambda f: base2i.get(os.path.basename(f).replace(".tab", "")))

def rank_gtalign(q):
    return gt_ranks.get(q, np.zeros(0, dtype=np.int64))


def rank_rtm(q):
    return rtm_ranks.get(q, np.zeros(0, dtype=np.int64))


RANKERS = {
    "RFoldseek": rank_new,
    "RFoldseek-TM": rank_rfoldseek_tm,
    "US-align": rank_us,
    "RMalign": rank_rm,
    "RiboSeek": rank_ribo,
}
# GTalign / RTM-align: added when their all-vs-database outputs are present
if gt_ranks:
    RANKERS["GTalign"] = rank_gtalign
if rtm_ranks:
    RANKERS["RTM-align"] = rank_rtm
if _have_blast:
    RANKERS["BLASTN"] = rank_bl


def r1_family(r, q):
    """Paper definition (Methods, 'Evaluation on the RNAsolo benchmark'):
    top-ranked NON-SELF target is a true positive; no neutral set.
    family TP = same Rfam family."""
    for t in r:
        if t == q:
            continue
        return 1 if fam[t] == fam[q] else 0
    return 0


def r1_clan(r, q):
    """clan TP = any target from the same Rfam clan, INCLUDING the same family."""
    for t in r:
        if t == q:
            continue
        return 1 if (clan[q] and clan[t] == clan[q]) else 0
    return 0


out = {
    "_note": "Rfam family/clan Recall@1, test queries. Paper definition (Methods 'Evaluation on the RNAsolo benchmark'): top-ranked non-self target is a TP, no neutral set; family TP=same family, clan TP=same clan including same family."
}
pr_out = {
    "_note": (
        "Per-query-averaged precision-recall for the test queries. Each query "
        "contributes one average precision (AP) and one interpolated PR curve, "
        "weighted equally regardless of family/clan size or query length, using "
        "each method's own hit order (no cross-query score comparison needed). "
        "This avoids the length bias of a pooled curve, important because RNA "
        "query lengths span two orders of magnitude. 'aupr' is the mean AP over "
        "queries; 'precision' is the mean interpolated PR curve on the 'recall' "
        "grid. Family: TP=same Rfam family, neutral=same clan/different family. "
        "Clan: TP=same clan/different family, neutral=same family. Unretrieved "
        "relevant targets remain false negatives."
    ),
    "_query_ids": {"family": fam_q, "clan": clan_q},
}
# carry over any method not recomputed here (its tool or outputs being absent)
_prev = json.load(open(RET / "recall1_rfam_test.json")) if os.path.exists(RET / "recall1_rfam_test.json") else {}
_pr_prev = json.load(open(RET / "weighted_pr_rfam_test.json")) if os.path.exists(RET / "weighted_pr_rfam_test.json") else {}
for _m in set(_prev) | set(_pr_prev):
    if not _m.startswith("_") and _m not in RANKERS:
        if _m in _prev:
            out[_m] = _prev[_m]
        if _m in _pr_prev:
            pr_out[_m] = _pr_prev[_m]
for m in RANKERS:
    fq = {q: RANKERS[m](q) for q in set(fam_q) | set(clan_q)}
    fr = np.mean([r1_family(fq[q], q) for q in fam_q])
    cq = [q for q in clan_q if clan[q]]  # clan-evaluable test queries
    cr = np.mean([r1_clan(fq[q], q) for q in cq])
    out[m] = {
        "family_recall1": float(fr),
        "clan_recall1": float(cr),
        "n_family": len(fam_q),
        "n_clan": len(cq),
    }
    pr_out[m] = {
        "family": rfam_per_query_pr_curve(fam_q, fq, fam, clan, "family"),
        "clan": rfam_per_query_pr_curve(clan_q, fq, fam, clan, "clan"),
    }
    print(f"  {m:12s} famR@1={fr:.3f} clanR@1={cr:.3f}  "
          f"famAUPR={pr_out[m]['family']['aupr']:.3f} clanAUPR={pr_out[m]['clan']['aupr']:.3f}", flush=True)
json.dump(out, open(RET / "recall1_rfam_test.json", "w"), indent=2)
json.dump(pr_out, open(RET / "weighted_pr_rfam_test.json", "w"), indent=2)
print("wrote recall1_rfam_test.json and weighted_pr_rfam_test.json\nDONE", flush=True)
