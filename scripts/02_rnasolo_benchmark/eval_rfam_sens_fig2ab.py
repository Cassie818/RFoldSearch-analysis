"""Assemble the TEST-ONLY per_query_sens_test.json for Fig 2A/B: per-query
family and clan Sens@1FP for all benchmark methods.

The per-query-averaged AUPR (a supplementary retrieval metric) is computed
alongside Recall@1 in eval_recall1_figS12.py, which reuses the same rankings.
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
from pathlib import Path
import os, sys, json, re, glob
import csv, subprocess, tempfile

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import defaultdict
import numpy as np, torch

from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels
from smith_waterman import smith_waterman_numba

K = 80
TOPN = 200  # candidate-pool depth selected on the validation split (see Table 9)
KMER_K = 4
THRESH_FRAC = 0.75
GO = -2.0
GE = -0.5
ND = "files/models"
RET = Path("files/production/retrieval")
db = torch.load(f"{ND}/db.pt", map_location="cpu", weights_only=False)
ents = db["entries"]
N = len(ents)
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
tags = [os.path.splitext(os.path.basename(str(e["id"])))[0] for e in ents]
fams = attach_labels(ents, load_labels("Splits/labels.tsv"))
clans = attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc"))
fam_pool = np.array(fams, dtype=object)
clan_pool = np.array(clans, dtype=object)

def idl(p):
    return [str(json.loads(l)["id"]) for l in open(p) if l.strip()]


test_dis = set(id2i[c] for c in idl("Splits/pdb_out/test.json") if c in id2i)
va_all = np.arange(N)
fam_test_q = []
clan_test_q = []
for q in sorted(test_dis):
    va = va_all != q
    if fam_pool[q] and np.any((fam_pool == fam_pool[q]) & va):
        fam_test_q.append(q)
    if clan_pool[q] and np.any((clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va):
        clan_test_q.append(q)
print(f"test fam queries={len(fam_test_q)}  clan queries={len(clan_test_q)}", flush=True)
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


def prefilter_candidates(q):
    """RFoldseek similar-k-mer prefilter candidates before final reranking."""
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
    """RFoldseek ranking for query q: token prefilter, then token SW reranking."""
    tt = toks[q]
    cand = prefilter_candidates(q)
    if len(cand) == 0:
        return np.zeros(0, np.int64)
    sw = np.array([smith_waterman_numba(tt, toks[t], sub, GO, GE) for t in cand], dtype=np.float64)
    return cand[np.argsort(-sw, kind="stable")]


# ---- RiboSeek rankings from cached encodings ----
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


# US-align ranked by the symmetric TM-score (mean of the two length-
# normalisations), consistent with GTalign/RTM-align and the structural
# ground truth. (The all-query per_query_sens.json historically ranked
# US-align by the query-length normalisation only; we recompute it here so
# every method uses one convention.)
_us = defaultdict(dict)
_us_cache = "files/production/retrieval/usalign_full.cache.jsonl"
if os.path.exists(_us_cache):
    for _line in open(_us_cache):
        if not _line.strip():
            continue
        _r = json.loads(_line)
        _i, _j = _r["i"], _r["j"]
        _a = 0.5 * (_r["tm_query"] + _r["tm_target"])
        _us[_i][_j] = _a
        _us[_j][_i] = _a
else:
    print(f"US-align: cache not found at {_us_cache}; skipping", flush=True)


def rank_us(q):
    d = _us.get(q, {})
    order = sorted(((t, s) for t, s in d.items() if t != q), key=lambda x: (-x[1], x[0]))
    return np.asarray([t for t, _ in order], dtype=np.int64)


def rank_rfoldseek_tm(q):
    """RFoldseek-TM: RFoldseek prefilter candidates reranked by US-align TM."""
    d = _us.get(q, {})
    rows = [(t, d[t], rank) for rank, t in enumerate(prefilter_candidates(q)) if t in d and t != q]
    rows.sort(key=lambda x: (-x[1], x[2], x[0]))
    return np.asarray([t for t, _, _ in rows], dtype=np.int64)


def _load_pair_scores(path, key):
    d = defaultdict(dict)
    if not os.path.exists(path):
        return d
    for line in open(path):
        if not line.strip():
            continue
        r = json.loads(line)
        if key not in r:
            continue
        i, j = r["i"], r["j"]
        d[i][j] = float(r[key])
        d[j][i] = float(r[key])
    return d


_rm = _load_pair_scores(RET / "rmalign_full.cache.jsonl", "rmscore")


def _rank_from_pair_dict(d, q):
    order = sorted(((t, s) for t, s in d.get(q, {}).items() if t != q), key=lambda x: (-x[1], x[0]))
    return np.asarray([t for t, _ in order], dtype=np.int64)


def rank_rm(q):
    return _rank_from_pair_dict(_rm, q)


def _run_blastn_scores():
    BIN = Path(os.environ.get("RFS_BLAST_BIN", ""))
    seq_path = Path("files/production/chain_sequences.tsv")
    if not seq_path.exists():
        return defaultdict(dict)
    seqs = {}
    for r in csv.DictReader(open(seq_path), delimiter="\t"):
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
            str(N),
        ],
        check=True,
        capture_output=True,
    )
    best = defaultdict(dict)
    for line in open(out):
        q, t, bs, _ = line.rstrip("\n").split("\t")
        if q == t or q not in tag2i or t not in tag2i:
            continue
        qi = tag2i[q]
        ti = tag2i[t]
        bs = float(bs)
        if ti not in best[qi] or bs > best[qi][ti]:
            best[qi][ti] = bs
    return best


_blast = defaultdict(dict)
if os.environ.get("RFS_RUN_BLASTN_PR", "0") == "1":
    print("running BLASTN for weighted PR...", flush=True)
    _blast = _run_blastn_scores()


def rank_blastn(q):
    return _rank_from_pair_dict(_blast, q)


base2i = {os.path.basename(str(e["id"])).replace(".cif", ""): i for i, e in enumerate(ents)}
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


def rank_gtalign(q):
    return gt_ranks.get(q, np.zeros(0, dtype=np.int64))


def rank_rtm(q):
    return rtm_ranks.get(q, np.zeros(0, dtype=np.int64))


ALL_TEST_Q = sorted(set(fam_test_q) | set(clan_test_q))


def per_query_sens(rank_fn):
    fam_sens = {}
    clan_sens = {}
    va_all = np.arange(N)
    ranks = {q: rank_fn(q) for q in ALL_TEST_Q}
    for q in fam_test_q:
        r = ranks[q]
        va = va_all != q
        rel = (fam_pool == fam_pool[q]) & va
        neu = (
            ((clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va) if clan_pool[q] else None
        )
        fam_sens[q] = first_fp_sensitivity(r, rel, neu) if len(r) else 0.0
    for q in clan_test_q:
        r = ranks[q]
        va = va_all != q
        rel = (clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & va
        neu = (fam_pool == fam_pool[q]) & va
        clan_sens[q] = first_fp_sensitivity(r, rel, neu) if len(r) else 0.0
    return fam_sens, clan_sens


print("computing NEW RFoldseek per-query...", flush=True)
nf, nc = per_query_sens(rank_new)
print("computing RFoldseek-TM per-query...", flush=True)
tmf, tmc = per_query_sens(rank_rfoldseek_tm)
print("computing RiboSeek per-query...", flush=True)
rf, rc = per_query_sens(rank_ribo)
# assemble output in the same array order as the test subset positions
out = {"_query_ids": {"family": fam_test_q, "clan": clan_test_q}}


def pack(fam_map, clan_map):
    fs = [fam_map[q] for q in fam_test_q]
    cs = [clan_map[q] for q in clan_test_q]
    return {
        "family": {"sens": fs, "n": len(fs), "mean": float(np.mean(fs))},
        "clan": {"sens": cs, "n": len(cs), "mean": float(np.mean(cs))},
    }


out["RFoldseek"] = pack(nf, nc)
out["RFoldseek-TM"] = pack(tmf, tmc)
out["RiboSeek"] = pack(rf, rc)
# US-align is recomputed here with the symmetric-TM convention (see rank_us)
if _us:
    print("computing US-align per-query (symmetric TM)...", flush=True)
    out["US-align"] = pack(*per_query_sens(rank_us))
# GTalign / RTM-align: coordinate aligners scored from their external outputs.
# Computed here when the outputs are present; otherwise carried over from any
# existing per_query_sens_test.json so a partial re-run does not drop them.
_dst = "files/production/retrieval/per_query_sens_test.json"
_prev = json.load(open(_dst)) if os.path.exists(_dst) else {}


def _same_query_counts(entry):
    try:
        return entry["family"]["n"] == len(fam_test_q) and entry["clan"]["n"] == len(clan_test_q)
    except Exception:
        return False


for _m, _rankfn, _ranks in (("GTalign", rank_gtalign, gt_ranks),
                            ("RTM-align", rank_rtm, rtm_ranks)):
    if _ranks:
        print(f"computing {_m} per-query (symmetric TM, {len(_ranks)} query outputs)...", flush=True)
        out[_m] = pack(*per_query_sens(_rankfn))
    elif _m in _prev and _same_query_counts(_prev[_m]):
        print(f"{_m}: no outputs under RFS_*_OUT, carrying over the existing entry", flush=True)
        out[_m] = _prev[_m]
    elif _m in _prev:
        print(f"{_m}: existing entry has old query counts; skipping", flush=True)
    else:
        print(f"{_m}: no outputs and no existing entry, skipping", flush=True)
if _rm:
    print("computing RMalign per-query...", flush=True)
    out["RMalign"] = pack(*per_query_sens(rank_rm))
elif "RMalign" in _prev and _same_query_counts(_prev["RMalign"]):
    print("RMalign: no cache, carrying over the existing Sens@1FP entry", flush=True)
    out["RMalign"] = _prev["RMalign"]
elif "RMalign" in _prev:
    print("RMalign: existing entry has old query counts; skipping", flush=True)
if _blast:
    print("computing BLASTN per-query...", flush=True)
    out["BLASTN"] = pack(*per_query_sens(rank_blastn))
elif "BLASTN" in _prev and _same_query_counts(_prev["BLASTN"]):
    print(
        "BLASTN: set RFS_RUN_BLASTN_PR=1 to recompute score-aware rankings; "
        "carrying over the existing Sens@1FP entry",
        flush=True,
    )
    out["BLASTN"] = _prev["BLASTN"]
elif "BLASTN" in _prev:
    print(
        "BLASTN: set RFS_RUN_BLASTN_PR=1 to recompute score-aware rankings; "
        "existing entry has old query counts, skipping",
        flush=True,
    )
json.dump(out, open(_dst, "w"), indent=2)
print("\n=== TEST-ONLY per-query means ===", flush=True)
for t in ["RFoldseek", "RFoldseek-TM", "GTalign", "RTM-align", "US-align", "RMalign", "RiboSeek", "BLASTN"]:
    if t in out:
        print(
            f"  {t:12s} famSens={out[t]['family']['mean']:.3f}  clanSens={out[t]['clan']['mean']:.3f}",
            flush=True,
        )
print("\nwrote files/production/retrieval/per_query_sens_test.json\nDONE", flush=True)
