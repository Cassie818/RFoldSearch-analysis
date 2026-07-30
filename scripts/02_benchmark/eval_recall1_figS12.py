"""Rfam family- and clan-level Recall@1 for the 5 benchmark methods on the
TEST partition, matching the Fig 2A/B query sets and label conventions.
Recall@1: first non-neutral retrieved hit (excluding self) is a true positive.
  family: TP=same family;  neutral=same clan, diff family
  clan  : TP=same clan & diff family;  neutral=same family (restricted to
          queries that have >=1 such target)
Writes files/production/retrieval/recall1_rfam_test.json.
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, FIG_DIR, SRC_MODELS, SRC_SEARCH

_sys.path.insert(0, SRC_SEARCH)
_sys.path.insert(0, SRC_MODELS)
_os.chdir(DATA_ROOT)
import os, sys, json

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

from prefilter_rfam_benchmark import attach_labels, load_labels
from smith_waterman import smith_waterman_numba

K = 80
TOPN = 100
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
# canonical test query sets (same as Fig 2)
pq = json.load(open(RET / "per_query_sens_test.json"))
fam_q = [i for i in pq["_query_ids"]["family"]]
clan_q = [i for i in pq["_query_ids"]["clan"]]
print(f"fam queries={len(fam_q)}  clan queries={len(clan_q)}", flush=True)
# ---- rankers (from cd_tmscore_new.py) ----
toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
sm = torch.load(f"{ND}/sub_matrix.pt", map_location="cpu", weights_only=False)
sub = np.asarray(sm["matrix"].numpy(), dtype=np.float32)
ck = torch.load(f"{ND}/desc_vqvae_best.pt", map_location="cpu", weights_only=False)
cb = ck["model"]["codebook.embeddings"].numpy()
cb = cb / (np.linalg.norm(cb, axis=1, keepdims=True) + 1e-9)
cos = (cb @ cb.T).astype(np.float32)
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
# BLASTN
import csv, subprocess, tempfile


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
blast = rank_blastn_all()
rank_bl = rank_from(blast)
RANKERS = {
    "RFoldseek": rank_new,
    "US-align": rank_us,
    "RMalign": rank_rm,
    "RiboSeek": rank_ribo,
    "BLASTN": rank_bl,
}


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
    print(f"  {m:12s} famR@1={fr:.3f}  clanR@1={cr:.3f}  (n_clan={len(cq)})", flush=True)
json.dump(out, open(RET / "recall1_rfam_test.json", "w"), indent=2)
print("wrote recall1_rfam_test.json\nDONE", flush=True)
