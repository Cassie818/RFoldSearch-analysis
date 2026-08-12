"""Fig 2 panels C/D: structure-defined (US-align TM-score) ground-truth retrieval
on the TEST-only covered queries.
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
import os, sys, json, csv, subprocess, tempfile, re, glob

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels
from smith_waterman import smith_waterman_numba

RET = Path("files/production/retrieval")
ND = "files/models"
K = 80
TOPN = 200  # candidate-pool depth selected on the validation split (see Table 9)
KMER_K = 4
THRESH_FRAC = 0.75
GO = -2.0
GE = -0.5
THRESHOLDS = [0.40, 0.45, 0.50]
db = torch.load(f"{ND}/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
n = len(ents)
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
tags = [os.path.splitext(os.path.basename(str(e["id"])))[0] for e in ents]


def idl(p):
    return [str(json.loads(l)["id"]) for l in open(p) if l.strip()]


test_dis = set(id2i[c] for c in idl("Splits/pdb_out/test.json") if c in id2i)
# TM matrix + covered queries
tm = np.full((n, n), np.nan, dtype=np.float32)
cov = defaultdict(int)
for line in open(RET / "usalign_full.cache.jsonl"):
    if not line.strip():
        continue
    r = json.loads(line)
    i, j = r["i"], r["j"]
    a = 0.5 * (r["tm_query"] + r["tm_target"])
    tm[i, j] = a
    tm[j, i] = a
    cov[i] += 1
    cov[j] += 1
queries = sorted(q for q in range(n) if cov[q] >= n - 5 and q in test_dis)
print(f"test covered queries: {len(queries)}", flush=True)
# ===== RFoldseek: deployed token pipeline (k-mer prefilter + SW rerank) =====
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
# Similar k-mers follow the Foldseek/MMseqs2 convention: a one-substitution
# variant is admissible iff the FULL k-mer score under Structure-BLOSUM clears
# `thr`. The k-mer self-score depends on all positions (the matrix is
# diag-mean-1, not diag == 1 per token), so the threshold is evaluated per
# k-mer rather than tabulated per token.
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


def rank_new(q):
    """RFoldseek ranking for query q: similar-k-mer-expansion prefilter over the
    inverted index (double-match on a common diagonal), top-N candidates, then
    token Smith-Waterman reranking; returns db indices by descending score."""
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
    sc.sort(key=lambda x: (-x[1], x[0]))  
    cand = np.array([x[0] for x in sc[:TOPN]], np.int64)
    if len(cand) == 0:
        return np.zeros(0, np.int64)
    return cand[
        np.argsort(
            -np.array([smith_waterman_numba(tt, toks[t], sub, GO, GE) for t in cand]), kind="stable"
        )
    ]


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


# RMalign cache
rm = defaultdict(dict)
for line in open(RET / "rmalign_full.cache.jsonl"):
    if not line.strip():
        continue
    r = json.loads(line)
    if "rmscore" not in r:
        continue
    i, j = r["i"], r["j"]
    rm[i][j] = r["rmscore"]
    rm[j][i] = r["rmscore"]


def rank_rm(q):
    d = rm.get(q, {})
    return np.asarray(
        [t for t, _ in sorted(d.items(), key=lambda kv: -kv[1]) if t != q], dtype=np.int64
    )


def rank_us(q):
    row = tm[q]
    cand = [(t, row[t]) for t in range(n) if t != q and not np.isnan(row[t])]
    return np.asarray([t for t, _ in sorted(cand, key=lambda kv: -kv[1])], dtype=np.int64)


def rank_blastn_all():
    BIN = Path(os.environ.get("RFS_BLAST_BIN", ""))  # empty -> resolve makeblastdb/blastn from PATH
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
try:
    blast = rank_blastn_all()
except FileNotFoundError:
    blast = {}
    print("BLASTN skipped (no blastn binary on PATH)", flush=True)


def rank_bl(q):
    d = blast.get(q, {})
    return np.asarray(
        [t for t, _ in sorted(d.items(), key=lambda kv: -kv[1]) if t != q], dtype=np.int64
    )


# ===== GTalign / RTM-align: rank by their own symmetric TM-score =====
# These aligners rank candidates by a TM-score, the same quantity that defines
# the structural ground truth here, so in C/D they are near-oracles (they hug
# the US-align ceiling). Parsed from raw tool output (RFS_GT_OUT/*.out,
# RFS_RTM_OUT/*.tab); skipped gracefully if the output dirs are absent.
base2i = {t: i for i, t in enumerate(tags)}
GT_OUT = os.environ.get("RFS_GT_OUT", "/tmp/gt_final")
RTM_OUT = os.environ.get("RFS_RTM_OUT", "/tmp/rtm_out")
_row_re = re.compile(r"PDB_[0-9A-Za-z_]+\.cif")


def _parse_gtalign(path):  # ranked db indices by symmetric TM = mean(refTM, queryTM)
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
    return rows


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
    return rows


def _collect(pattern, parse, keyfn):
    ranks = {}
    for f in glob.glob(pattern):
        qi = keyfn(f)
        if qi is None:
            continue
        ranks[qi] = np.asarray([t for t, _ in parse(f) if t != qi], dtype=np.int64)
    return ranks


gt_ranks = _collect(f"{GT_OUT}/*.out", _parse_gtalign,
                    lambda f: base2i.get(os.path.basename(f).split("__")[0]))
rtm_ranks = _collect(f"{RTM_OUT}/*.tab", _parse_rtm,
                     lambda f: base2i.get(os.path.basename(f).replace(".tab", "")))
print(f"GTalign queries: {len(gt_ranks)}   RTM-align queries: {len(rtm_ranks)}", flush=True)


def rank_gtalign(q):
    return gt_ranks.get(q, np.zeros(0, dtype=np.int64))


def rank_rtm(q):
    return rtm_ranks.get(q, np.zeros(0, dtype=np.int64))


RANKERS = {
    "RFoldseek": rank_new,
    "US-align*": rank_us,
    "RMalign": rank_rm,
    "RiboSeek": rank_ribo,
}
if gt_ranks:
    RANKERS["GTalign"] = rank_gtalign
if rtm_ranks:
    RANKERS["RTM-align"] = rank_rtm
if blast:
    RANKERS["BLASTN"] = rank_bl
# precompute rankings for test queries
print("computing rankings...", flush=True)
RANKS = {m: {q: RANKERS[m](q) for q in queries} for m in RANKERS}
out = {"_note": "test-only covered queries; US-align* is the TM ceiling.", "thresholds": {}}
va_all = np.arange(n)
# Rfam family labels: at the clan level a structural positive is TM-similar but
# from a *different* Rfam family (same-family hits are treated as neutral).
fam_arr = np.array(attach_labels(ents, load_labels("Splits/labels.tsv")), dtype=object)
per_query_out = {}
for th in THRESHOLDS:
    res = {}
    res_pq = {}
    for m in RANKERS:
        r1 = []
        sn = []; sn_q = []
        sn_clan = []; sn_clan_q = []
        for q in queries:
            va = va_all != q
            pos = (~np.isnan(tm[q])) & (tm[q] >= th) & va
            r = RANKS[m][q]
            if pos.any():
                if len(r) == 0:
                    r1.append(0)
                    sn.append(0.0); sn_q.append(int(q))
                else:
                    r1.append(int(pos[r[0]]))
                    sn.append(first_fp_sensitivity(r, pos, None)); sn_q.append(int(q))
            # clan-level structural: TM-similar AND different family; same-family = neutral
            if fam_arr[q]:
                rel_c = pos & (fam_arr != fam_arr[q])
                if rel_c.any():
                    neu_c = (fam_arr == fam_arr[q]) & va
                    sn_clan.append(
                        first_fp_sensitivity(
                            r if len(r) > 0 else np.zeros(0, np.int64), rel_c, neu_c
                        )
                    ); sn_clan_q.append(int(q))
        res[m] = {
            "Recall@1": float(np.mean(r1)) if r1 else 0.0,
            "Sens@1FP": float(np.mean(sn)) if sn else 0.0,
            "Sens@1FP_clan": float(np.mean(sn_clan)) if sn_clan else 0.0,
            "n_queries": len(r1),
            "n_clan": len(sn_clan),
        }
        # per-query values keyed by query index (for family/clan block bootstrap)
        res_pq[m] = {"family": dict(zip(sn_q, sn)),
                     "clan": dict(zip(sn_clan_q, sn_clan))}
    out["thresholds"][f"{th}"] = res
    per_query_out[f"{th}"] = res_pq
    print(
        f"  TM>={th}: fam["
        + " ".join(f"{m}={res[m]['Sens@1FP']:.3f}" for m in RANKERS)
        + f"]  clan["
        + " ".join(f"{m}={res[m]['Sens@1FP_clan']:.3f}" for m in RANKERS)
        + "]",
        flush=True,
    )
json.dump(out, open(RET / "structural_tmscore_metrics_test.json", "w"), indent=2)
json.dump(per_query_out, open(RET / "structural_tmscore_per_query_test.json", "w"), indent=2)
print("wrote structural_tmscore_metrics_test.json\nDONE", flush=True)
