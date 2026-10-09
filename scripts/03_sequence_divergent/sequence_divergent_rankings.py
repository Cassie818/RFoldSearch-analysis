"""Rankings and per-query Sens@1FP behind Fig. 3 (sequence-divergent homologues)."""
import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
from paths import BLAST_BIN, RFOLDSEARCH_REPO, SRC_SEARCH

sys.path.insert(0, SRC_SEARCH)
sys.path.insert(0, RFOLDSEARCH_REPO)
from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels
from rfoldseek.structure_score import c4_coords, score_pair
from rfoldseek.tiebreak import tie_keys
from smith_waterman import smith_waterman_numba

from sequence_divergent_rescore import seq_identity

HMMER_BIN = os.environ.get("RFOLDSEARCH_HMMER_BIN", os.environ.get("RFOLDSEEK_HMMER_BIN", ""))
RET = os.path.join(REPO, "files", "production", "retrieval")
OUT = os.path.join(RET, "sequence_divergent")
K, TOPN, KMER_K, THRESH_FRAC, GO, GE = 80, 200, 4, 0.75, -2.0, -0.5
NSHUF = 3
FILTERS = ("none", "B/N E<0.01", "B/N E<=10", "B/N E<=10 + Infernal any")


def _bin(d, name):
    return os.path.join(d, name) if d else name


# ---- database, labels, deployed search ------------------------------------------------
os.chdir(REPO)
ents = torch.load("files/models/db.pt", map_location="cpu", weights_only=False)["entries"]
N = len(ents)
tags = [os.path.splitext(os.path.basename(str(e["id"])))[0] for e in ents]
t2i = {t: i for i, t in enumerate(tags)}
fam = np.array(attach_labels(ents, load_labels("Splits/labels.tsv")), dtype=object)
clan = np.array(attach_labels(ents, load_labels("Splits/labels.tsv", label_column="clan_acc")), dtype=object)
id2i = {str(e["id"]): i for i, e in enumerate(ents)}
test = sorted(id2i[json.loads(l)["id"]] for l in open("Splits/pdb_out/test.json")
              if l.strip() and json.loads(l)["id"] in id2i)
ALL = np.arange(N)
FAM_Q = [q for q in test if fam[q] and np.any((fam == fam[q]) & (ALL != q))]
CLAN_Q = [q for q in test if clan[q] and np.any((clan == clan[q]) & (fam != fam[q]) & (ALL != q))]

toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
sub = np.asarray(torch.load("files/models/sub_matrix.pt", map_location="cpu", weights_only=False)["matrix"].numpy(),
                 dtype=np.float32)
powers = np.array([K ** (KMER_K - 1 - i) for i in range(KMER_K)], dtype=np.int64)
idx = defaultdict(list)
for ti, tt in enumerate(toks):
    for p in range(len(tt) - KMER_K + 1):
        idx[int(np.dot(tt[p:p + KMER_K], powers))].append((ti, p))
smith_waterman_numba(np.zeros(4, np.int64), np.zeros(4, np.int64), sub, GO, GE)
XYZ = [c4_coords(e, len(t)) for e, t in zip(ents, toks)]
assert all(x is not None for x in XYZ), "every chain needs C4' coordinates"
GO32, GE32 = np.float32(GO), np.float32(GE)


def structure_score(qt, qxyz, t):
    """RFoldSearch's ranking score S * aTM of query tokens qt (C4' qxyz) against chain t."""
    s, atm, _ = score_pair(qt, qxyz, toks[t], XYZ[t], sub, GO32, GE32)
    return s * atm


THR = THRESH_FRAC * KMER_K
DIAG = np.array([sub[t, t] for t in range(K)], dtype=np.float64)


def expand(qk):
    """Similar k-mers: one substitution whose full Structure-BLOSUM k-mer score is >= THR."""
    base = int(np.dot(qk, powers))
    out = [base]
    self_score = float(sum(DIAG[int(qk[i])] for i in range(KMER_K)))
    for pos in range(KMER_K):
        cur = int(qk[pos])
        rest = self_score - DIAG[cur]
        for tok in range(K):
            if tok != cur and rest + sub[cur][tok] >= THR:
                out.append(base + int(powers[pos]) * (tok - cur))
    return out


def prefilter(tt, q):
    """All prefilter hits with their best diagonal count (at least two k-mer hits)."""
    td = defaultdict(dict)
    for qp in range(len(tt) - KMER_K + 1):
        for kid in expand(tt[qp:qp + KMER_K]):
            for ti, tp in idx.get(kid, ()):
                if ti != q:
                    td[ti][qp - tp] = td[ti].get(qp - tp, 0) + 1
    sc = [(ti, max((c for c in dm.values() if c >= 2), default=-1)) for ti, dm in td.items()]
    return [x for x in sc if x[1] > 0]


def rank_tokens(tt, qxyz, q):
    """Deployed search with index tie-breaking (used for the shuffled-token control)."""
    sc = sorted(prefilter(tt, q), key=lambda x: (-x[1], x[0]))
    cand = np.array([x[0] for x in sc][:TOPN], np.int64)
    if len(cand) == 0:
        return np.zeros(0, np.int64)
    key = np.array([structure_score(tt, qxyz, t) for t in cand])
    return cand[np.argsort(-key, kind="stable")]


US = defaultdict(dict)
for line in open(os.path.join(RET, "usalign_full.cache.jsonl")):
    r = json.loads(line)
    US[r["i"]][r["j"]] = US[r["j"]][r["i"]] = 0.5 * (r["tm_query"] + r["tm_target"])


def rank_seeded(q, keys_by_seed):
    """RFoldSearch and RFoldSearch-TM rankings, one per tie-breaking seed."""
    sc = prefilter(toks[q], q)
    c0 = np.array([x[0] for x in sc], np.int64)
    s0 = np.array([x[1] for x in sc])
    cache, rf, tm = {}, [], []
    for keys in keys_by_seed:
        if len(c0) == 0:
            rf.append(c0)
            tm.append(c0)
            continue
        c = c0[np.lexsort((keys[c0], -s0))][:TOPN]
        key = np.array([cache.setdefault(int(t), structure_score(toks[q], XYZ[q], t)) for t in c])
        rf.append(c[np.lexsort((keys[c], -key))])
        tmv = np.array([US[q].get(int(t), -1.0) for t in c])
        tm.append(c[np.lexsort((keys[c], -tmv))])
    return rf, tm


# ---- sequence searches -------------------------------------------------------------------
SEQS = {r["chain_id"]: r["sequence"] for r in csv.DictReader(open("files/production/chain_sequences.tsv"),
                                                             delimiter="\t")}
WORK = tempfile.mkdtemp(prefix="seqdiv_")
DB_RNA, DB_DNA = f"{WORK}/db.fa", f"{WORK}/dbT.fa"
with open(DB_RNA, "w") as f:
    for i, t in enumerate(tags):
        f.write(f">t{i}\n{SEQS[t]}\n")
open(DB_DNA, "w").write(open(DB_RNA).read().replace("U", "T"))
subprocess.run([_bin(BLAST_BIN, "makeblastdb"), "-in", DB_DNA, "-dbtype", "nucl"], check=True, capture_output=True)


def sequence_hits(q):
    """Best E-value per target from BLASTN and from nhmmer (E <= 10), query excluded."""
    qs = SEQS[tags[q]]
    open(f"{WORK}/qT.fa", "w").write(f">q\n{qs.replace('U', 'T')}\n")
    out = subprocess.run([_bin(BLAST_BIN, "blastn"), "-task", "blastn", "-query", f"{WORK}/qT.fa", "-db", DB_DNA,
                          "-evalue", "10", "-strand", "both", "-dust", "no", "-max_target_seqs", "5000",
                          "-outfmt", "6 sseqid evalue bitscore"], capture_output=True, text=True).stdout
    bl = {}
    for line in out.splitlines():
        s_, e, _ = line.split("\t")
        bl[int(s_[1:])] = min(bl.get(int(s_[1:]), 99.0), float(e))
    open(f"{WORK}/q.fa", "w").write(f">q\n{qs}\n")
    subprocess.run([_bin(HMMER_BIN, "nhmmer"), "--rna", "-E", "10", "--noali", "--cpu", "4", "--tblout",
                    f"{WORK}/o.tbl", "-o", os.devnull, f"{WORK}/q.fa", DB_RNA], capture_output=True)
    nh = {}
    for line in open(f"{WORK}/o.tbl"):
        if not line.startswith("#"):
            f = line.split()
            nh[int(f[0][1:])] = min(nh.get(int(f[0][1:]), 99.0), float(f[12]))
    bl.pop(q, None)
    nh.pop(q, None)
    return bl, nh


INF = json.load(open(os.path.join(RET, "rankings", "infernal.json")))
RS = {m: json.load(open(os.path.join(RET, "rankings", f"riboseek_{k}.json")))
      for m, k in (("RS-20", "rs20"), ("RS-80", "rs80"))}
TLEN = np.array([len(SEQS[t]) for t in tags])
COMP = np.stack([np.bincount(t, minlength=K) for t in toks]).astype(float)
COMP /= np.linalg.norm(COMP, axis=1, keepdims=True) + 1e-12


def rankings(q, keys_by_seed, cache):
    """All rankings of query q except the shuffled control (shared by both analyses)."""
    if q in cache:
        return cache[q]
    rf, tm = rank_seeded(q, keys_by_seed)
    bl, nh = sequence_hits(q)
    rank = {
        "US-align": np.asarray([t for t, _ in sorted(((t, s) for t, s in US[q].items() if t != q),
                                                     key=lambda x: (-x[1], x[0]))], np.int64),
        "BLASTN": np.asarray(sorted(bl, key=lambda i: (bl[i], i)), np.int64),
        "nhmmer": np.asarray(sorted(nh, key=lambda i: (nh[i], i)), np.int64),
        "Infernal": np.asarray([t2i[x] for x in INF.get(tags[q], {}).get("ranking", [])
                                if x in t2i and t2i[x] != q], np.int64),
        "Length": np.asarray([i for i in np.lexsort((ALL, np.abs(TLEN - TLEN[q]))) if i != q], np.int64),
        "Composition": np.asarray([i for i in np.lexsort((ALL, -(COMP @ COMP[q]))) if i != q], np.int64),
    }
    for m, d in RS.items():
        rank[m] = np.asarray([t2i[x] for x in d.get(tags[q], {}).get("ranking", [])
                              if x in t2i and t2i[x] != q], np.int64)
    cache[q] = (rf, tm, rank, bl, nh)
    return cache[q]


def score(queries, truth, keys_by_seed, cache, family=False):
    """Per-query Sens@1FP under each sequence filter; truth(q) -> (positives, neutrals)."""
    rng = np.random.default_rng(0)
    per = defaultdict(lambda: defaultdict(dict))
    saved, pair_id = {}, {}
    for n, q in enumerate(queries):
        rf, tm, rank, bl, nh = rankings(q, keys_by_seed, cache)
        perms = [rng.permutation(len(toks[q])) for _ in range(NSHUF)]
        shuf = [rank_tokens(toks[q][pm], XYZ[q][pm], q) for pm in perms]
        P, Nn = truth(q)
        s01, s10, ia = np.zeros(N, bool), np.zeros(N, bool), np.zeros(N, bool)
        for d in (bl, nh):
            for i, e in d.items():
                s10[i] = True
                s01[i] |= e < 0.01
        ia[rank["Infernal"]] = True
        if family:
            for t in np.flatnonzero(P):
                key = (min(q, t), max(q, t))
                if key not in pair_id:
                    pair_id[key] = seq_identity(SEQS[tags[q]], SEQS[tags[t]])
            saved[tags[q]] = {"RFoldSearch": [r.tolist() for r in rf], "RFoldSearch-TM": [r.tolist() for r in tm],
                              "Shuffled": [r.tolist() for r in shuf], **{m: r.tolist() for m, r in rank.items()},
                              "seq_detect_E": {str(i): float(min(bl.get(i, 99.0), nh.get(i, 99.0)))
                                               for i in set(bl) | set(nh)},
                              "infernal_any": [int(i) for i in np.flatnonzero(ia)]}
        sens = lambda r, pp, nn: first_fp_sensitivity(r, pp, nn) if len(r) else 0.0
        for fl, sn in zip(FILTERS, (None, s01, s10, s10 | ia)):
            pp, nn = (P, Nn) if sn is None else (P & ~sn, Nn | (P & sn))
            if not pp.any():
                continue
            for m, r in rank.items():
                per[fl][m][tags[q]] = sens(r, pp, nn)
            per[fl]["RFoldSearch"][tags[q]] = float(np.mean([sens(r, pp, nn) for r in rf]))
            per[fl]["RFoldSearch-TM"][tags[q]] = float(np.mean([sens(r, pp, nn) for r in tm]))
            per[fl]["Shuffled"][tags[q]] = float(np.mean([sens(r, pp, nn) for r in shuf]))
            per[fl]["_npos"][tags[q]] = int(pp.sum())
        if (n + 1) % 20 == 0:
            print(f"  {n + 1}/{len(queries)}", flush=True)
    return {fl: {m: dict(v) for m, v in d.items()} for fl, d in per.items()}, saved, pair_id


def compare(new, path, label):
    old = json.load(open(path))["values"]
    worst, n = 0.0, 0
    for fl, d in new.items():
        for m, v in d.items():
            for q, x in v.items():
                y = old.get(fl, {}).get(m, {}).get(q)
                if y is None:
                    print(f"  {label}: missing {fl}/{m}/{q}")
                    continue
                worst, n = max(worst, abs(x - y)), n + 1
    print(f"{label}: {n} values compared with {os.path.basename(path)}; largest difference {worst:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    keys_by_seed = [tie_keys(tags, sd) for sd in range(args.seeds)]
    print(f"family queries {len(FAM_Q)} ({len(set(fam[FAM_Q]))} families), clan queries {len(CLAN_Q)}", flush=True)
    cache = {}

    def family_truth(q):
        va = ALL != q
        neutral = ((clan == clan[q]) & (fam != fam[q]) & va) if clan[q] else np.zeros(N, bool)
        return (fam == fam[q]) & va, neutral | (fam == "")

    def clan_truth(q):
        va = ALL != q
        return (clan == clan[q]) & (fam != fam[q]) & va, ((fam == fam[q]) & va) | (fam == "")

    fam_per, saved, pair_id = score(FAM_Q, family_truth, keys_by_seed, cache, family=True)
    clan_per, _, _ = score(CLAN_Q, clan_truth, keys_by_seed, cache)
    dst = OUT if not args.check else tempfile.mkdtemp(prefix="seqdiv_check_")
    os.makedirs(dst, exist_ok=True)
    json.dump({"tags": tags, "fam": list(fam), "clan": list(clan), "queries": [tags[q] for q in FAM_Q],
               "ranks": saved}, open(os.path.join(dst, "family_rankings.json"), "w"))
    json.dump({f"{tags[a]}|{tags[b]}": v for (a, b), v in pair_id.items()},
              open(os.path.join(dst, "family_pair_identity.json"), "w"))
    json.dump({"blocks": {tags[q]: fam[q] for q in FAM_Q}, "clan": {tags[q]: clan[q] for q in FAM_Q},
               "values": fam_per}, open(os.path.join(dst, "family_perquery.json"), "w"))
    json.dump({"clan": {tags[q]: clan[q] for q in CLAN_Q}, "values": clan_per,
               "note": f"RFoldSearch and RFoldSearch-TM: mean over {args.seeds} sha256 tie-breaking seeds"},
              open(os.path.join(dst, "clan_perquery.json"), "w"))
    if args.check:
        compare(fam_per, os.path.join(OUT, "family_perquery.json"), "family")
        compare(clan_per, os.path.join(OUT, "clan_perquery.json"), "clan")
        old = json.load(open(os.path.join(OUT, "family_pair_identity.json")))
        new = {f"{tags[a]}|{tags[b]}": v for (a, b), v in pair_id.items()}
        diff = [k for k in set(old) | set(new) if abs(old.get(k, -1) - new.get(k, -2)) > 1e-12]
        print(f"pair identity: {len(new)} pairs, {len(diff)} differ from the stored file")
    print("wrote", dst)


if __name__ == "__main__":
    main()
