"""RiboSeek alphabet control under the RFoldseek search pipeline.

The standard RiboSeek baseline uses the RiboSeek package's own Searcher. That
comparison mixes representation and search-procedure differences. This script
holds the RFoldseek search procedure fixed:

  similar-k-mer expansion prefilter -> top-N candidates -> token SW rerank

and swaps only the discrete alphabet:

  1. RFoldseek 80-state learned tokens + Structure-BLOSUM
  2. RiboSeek sa20 labels + RiboSeek score matrix

By default, each alphabet uses its native substitution matrix scaled to the same
diagonal mean. For the stricter alphabet-only control, set
``RFS_ALPHABET_CONTROL_MATRIX=identity``; both alphabets then use the same
symbolic scoring rule (match=+1, mismatch=-1), with only matrix dimensionality
following alphabet size. For the fairest learned-matrix control, set
``RFS_ALPHABET_CONTROL_MATRIX=self_blosum``; each alphabet learns its own
Structure-BLOSUM from the same training-partition US-align pairs.
"""
from __future__ import annotations

import json
import os
import sys
import time
import hashlib
from collections import defaultdict
from pathlib import Path

import importlib.resources as R
import numpy as np
import torch
from numba import njit
from riboseek import Searcher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from paths import DATA_ROOT, SRC_SEARCH, resolve  # noqa: E402

sys.path.insert(0, SRC_SEARCH)
from prefilter_rfam_benchmark import attach_labels, first_fp_sensitivity, load_labels  # noqa: E402
from smith_waterman import smith_waterman_numba  # noqa: E402


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

OUT_DIR = Path(resolve("files/ablation"))
MATRIX_MODE = os.environ.get("RFS_ALPHABET_CONTROL_MATRIX", "native").strip().lower()
if MATRIX_MODE not in {"native", "identity", "self_blosum"}:
    raise ValueError("RFS_ALPHABET_CONTROL_MATRIX must be 'native', 'identity', or 'self_blosum'")

OUT_SUFFIX = "" if MATRIX_MODE == "native" else f"_{MATRIX_MODE}"
OUT_SUMMARY = OUT_DIR / f"riboseek_alphabet_same_pipeline_control{OUT_SUFFIX}.json"
OUT_PER_QUERY = OUT_DIR / f"riboseek_alphabet_same_pipeline_control{OUT_SUFFIX}_per_query.json"
RIBOSEEK_CACHE = Path(resolve("files/ablation/riboseek_real_encoded.json"))

KMER_K = 4
TOPN = 200
THRESH_FRAC = 0.75
GAP_OPEN = -2.0
GAP_EXTEND = -0.5
BLOSUM_TM_THRESH = 0.5
BLOSUM_SCALE = 2.0
BLOSUM_GAP_OPEN = -11.0
BLOSUM_GAP_EXTEND = -1.0


def resolve_env_or_repo(env_name: str, relpath: str) -> Path:
    override = os.environ.get(env_name)
    if override:
        return Path(override)
    return Path(resolve(relpath))


def id_list(path: str) -> list[str]:
    return [str(json.loads(line)["id"]) for line in open(resolve(path)) if line.strip()]


def file_md5(path: str | Path) -> str:
    path = Path(path)
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance() -> dict:
    relpaths = [
        "files/models/db.pt",
        "files/models/sub_matrix.pt",
        "files/ablation/riboseek_real_encoded.json",
        "Splits/pdb_out/test.json",
        "Splits/labels.tsv",
    ]
    paths = {rel: resolve(rel) for rel in relpaths}
    extra_paths = {}
    if MATRIX_MODE == "self_blosum":
        extra_paths["RFS_TRAIN_SPLIT"] = str(resolve_env_or_repo("RFS_TRAIN_SPLIT", "Splits/pdb_out/train.json"))
        extra_paths["RFS_USALIGN_CACHE"] = str(resolve_env_or_repo("RFS_USALIGN_CACHE", "files/production/retrieval/usalign_full.cache.jsonl"))
    input_hashes = {rel: file_md5(path) for rel, path in paths.items()}
    input_hashes.update({key: file_md5(path) for key, path in extra_paths.items()})
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "data_root": DATA_ROOT,
        "prefer_external": os.environ.get("RFOLDSEEK_PREFER_EXTERNAL", "") == "1",
        "matrix_mode": MATRIX_MODE,
        "resolved_paths": {**paths, **extra_paths},
        "input_hashes_md5": input_hashes,
    }


def load_query_sets(entries: list[dict], families: list[str], clans: list[str]) -> tuple[list[int], list[int]]:
    id_to_i = {str(entry["id"]): i for i, entry in enumerate(entries)}
    test_dis = {id_to_i[entry_id] for entry_id in id_list("Splits/pdb_out/test.json") if entry_id in id_to_i}
    fam_counts = defaultdict(int)
    clan_counts = defaultdict(int)
    for fam in families:
        if fam:
            fam_counts[fam] += 1
    for clan in clans:
        if clan:
            clan_counts[clan] += 1
    fam_q = [q for q in sorted(test_dis) if families[q] and fam_counts[families[q]] >= 2]
    # The clan-evaluable set is taken from the main benchmark's stored clan query
    # list (per_query_sens_test.json, used by Fig 2A/B) intersected with the test
    # partition, so panels A/B and E/F share an identical 79-query clan set. The
    # purely local rule below (clan-count >= 2 and a same-clan/different-family
    # positive exists) would additionally admit db indices 772 and 868, which
    # shift clan Sens@1FP by < 0.006; it is kept here only as a fallback.
    ab_path = resolve("files/production/retrieval/per_query_sens_test.json")
    if os.path.exists(ab_path):
        ab_clan = set(int(x) for x in json.load(open(ab_path))["_query_ids"]["clan"])
        clan_q = [q for q in sorted(test_dis) if q in ab_clan]
    else:
        clan_q = [
            q
            for q in sorted(test_dis)
            if clans[q]
            and clan_counts[clans[q]] >= 2
            and np.any((np.asarray(clans, dtype=object) == clans[q]) & (np.asarray(families, dtype=object) != families[q]))
        ]
    return fam_q, clan_q


def normalize_matrix(matrix: np.ndarray, target_diag_mean: float) -> np.ndarray:
    mat = np.asarray(matrix, dtype=np.float32)
    diag_mean = float(np.mean(np.diag(mat)))
    if diag_mean == 0:
        raise ValueError("score matrix diagonal mean is zero")
    return mat * (target_diag_mean / diag_mean)


def identity_matrix(size: int) -> np.ndarray:
    matrix = np.full((size, size), -1.0, dtype=np.float32)
    np.fill_diagonal(matrix, 1.0)
    return matrix


@njit(cache=True)
def traceback_token_pairs(q: np.ndarray, t: np.ndarray, sub: np.ndarray, go: float, ge: float) -> tuple[np.ndarray, np.ndarray]:
    m, n = len(q), len(t)
    if m == 0 or n == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    H = np.zeros((m + 1, n + 1), dtype=np.float32)
    E = np.full((m + 1, n + 1), -1e9, dtype=np.float32)
    F = np.full((m + 1, n + 1), -1e9, dtype=np.float32)
    bt = np.zeros((m + 1, n + 1), dtype=np.int8)
    best_i = best_j = 0
    best_v = 0.0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            e = H[i, j - 1] + go
            e2 = E[i, j - 1] + ge
            E[i, j] = e if e > e2 else e2
            f = H[i - 1, j] + go
            f2 = F[i - 1, j] + ge
            F[i, j] = f if f > f2 else f2
            diag = H[i - 1, j - 1] + sub[q[i - 1], t[j - 1]]
            value = 0.0
            tag = np.int8(0)
            if diag > value:
                value = diag
                tag = np.int8(1)
            if F[i, j] > value:
                value = F[i, j]
                tag = np.int8(2)
            if E[i, j] > value:
                value = E[i, j]
                tag = np.int8(3)
            H[i, j] = value
            bt[i, j] = tag
            if value > best_v:
                best_v = value
                best_i, best_j = i, j

    left = np.zeros(m + n, dtype=np.int64)
    right = np.zeros(m + n, dtype=np.int64)
    n_pair = 0
    i, j = best_i, best_j
    while i > 0 and j > 0 and bt[i, j] != 0:
        tag = bt[i, j]
        if tag == 1:
            left[n_pair] = q[i - 1]
            right[n_pair] = t[j - 1]
            n_pair += 1
            i -= 1
            j -= 1
        elif tag == 2:
            i -= 1
        else:
            j -= 1
    return left[:n_pair][::-1].copy(), right[:n_pair][::-1].copy()


def load_train_pairs(entries: list[dict]) -> tuple[list[tuple[int, int]], dict]:
    train_path = resolve_env_or_repo("RFS_TRAIN_SPLIT", "Splits/pdb_out/train.json")
    cache_path = resolve_env_or_repo("RFS_USALIGN_CACHE", "files/production/retrieval/usalign_full.cache.jsonl")
    if not train_path.exists():
        raise FileNotFoundError(
            f"Missing training split {train_path}. Set RFS_TRAIN_SPLIT to the train.json used for Structure-BLOSUM."
        )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"Missing US-align cache {cache_path}. Set RFS_USALIGN_CACHE to usalign_full.cache.jsonl."
        )

    id_to_i = {str(entry["id"]): i for i, entry in enumerate(entries)}
    train_ids = id_list(str(train_path))
    train_set = {id_to_i[entry_id] for entry_id in train_ids if entry_id in id_to_i}
    pair_set: set[tuple[int, int]] = set()
    n_cache = 0
    with cache_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            n_cache += 1
            record = json.loads(line)
            i = int(record["i"])
            j = int(record["j"])
            if i not in train_set or j not in train_set or i == j:
                continue
            tm = 0.5 * (float(record["tm_query"]) + float(record["tm_target"]))
            if tm < BLOSUM_TM_THRESH:
                continue
            a, b = (i, j) if i < j else (j, i)
            pair_set.add((a, b))
    pairs = sorted(pair_set)
    meta = {
        "train_split": str(train_path),
        "usalign_cache": str(cache_path),
        "train_ids_in_db": len(train_set),
        "cache_records": n_cache,
        "tm_threshold": BLOSUM_TM_THRESH,
        "n_train_pairs": len(pairs),
        "construction_gap_open": BLOSUM_GAP_OPEN,
        "construction_gap_extend": BLOSUM_GAP_EXTEND,
    }
    return pairs, meta


def build_self_blosum(tokens: list[np.ndarray], pair_pool: list[tuple[int, int]], alphabet_size: int) -> tuple[np.ndarray, dict]:
    sub_id = np.zeros((alphabet_size, alphabet_size), dtype=np.float32)
    np.fill_diagonal(sub_id, 1.0)
    joint = np.zeros((alphabet_size, alphabet_size), dtype=np.float64)
    n_alignments = 0
    n_columns = 0
    for n, (a, b) in enumerate(pair_pool, start=1):
        q = tokens[a]
        t = tokens[b]
        if len(q) == 0 or len(t) == 0:
            continue
        pi, pj = traceback_token_pairs(q, t, sub_id, BLOSUM_GAP_OPEN, BLOSUM_GAP_EXTEND)
        if len(pi) == 0:
            continue
        for x, y in zip(pi, pj):
            joint[int(x), int(y)] += 1
            joint[int(y), int(x)] += 1
        n_alignments += 1
        n_columns += int(len(pi))
        if n % 500 == 0:
            print(f"  self-BLOSUM alignments {n}/{len(pair_pool)}", flush=True)

    total = max(float(joint.sum()), 1.0)
    p_ij = joint / total
    p_i = np.maximum(p_ij.sum(1), 1e-12)
    scores = np.log2(np.maximum(p_ij, 1e-12) / np.maximum(np.outer(p_i, p_i), 1e-12)) * BLOSUM_SCALE
    scores = scores.astype(np.float32)
    diag_mean_before = float(np.mean(np.diag(scores)))
    scores *= 1.0 / max(diag_mean_before, 1e-6)
    meta = {
        "alphabet_size": alphabet_size,
        "n_input_train_pairs": len(pair_pool),
        "n_aligned_train_pairs": n_alignments,
        "n_aligned_columns": n_columns,
        "scale": BLOSUM_SCALE,
        "diag_mean_before_rescale": diag_mean_before,
        "diag_mean_after_rescale": float(np.mean(np.diag(scores))),
    }
    return scores, meta


class SamePipelineRanker:
    def __init__(
        self,
        tokens: list[np.ndarray],
        score_matrix: np.ndarray,
        kmer_k: int = KMER_K,
        top_n: int = TOPN,
        thresh_frac: float = THRESH_FRAC,
    ) -> None:
        self.tokens = [np.asarray(seq, dtype=np.int64) for seq in tokens]
        self.score_matrix = np.asarray(score_matrix, dtype=np.float32)
        self.kmer_k = kmer_k
        self.top_n = top_n
        self.K = int(self.score_matrix.shape[0])
        self.powers = np.array([self.K ** (kmer_k - 1 - i) for i in range(kmer_k)], dtype=np.int64)
        self.threshold = thresh_frac * kmer_k
        self.diag = np.array([self.score_matrix[i, i] for i in range(self.K)], dtype=np.float64)
        self.index = self._build_index()
        # JIT warmup.
        smith_waterman_numba(np.zeros(4, np.int64), np.zeros(4, np.int64), self.score_matrix, GAP_OPEN, GAP_EXTEND)

    def _build_index(self) -> dict[int, list[tuple[int, int]]]:
        index: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for ti, seq in enumerate(self.tokens):
            if len(seq) < self.kmer_k:
                continue
            for pos in range(len(seq) - self.kmer_k + 1):
                kid = int(np.dot(seq[pos : pos + self.kmer_k], self.powers))
                index[kid].append((ti, pos))
        return index

    def expand(self, qk: np.ndarray) -> list[int]:
        base = int(np.dot(qk, self.powers))
        out = [base]
        self_score = float(sum(self.diag[int(qk[i])] for i in range(self.kmer_k)))
        for pos in range(self.kmer_k):
            delta = int(self.powers[pos])
            cur = int(qk[pos])
            base_without = self_score - self.diag[cur]
            row = self.score_matrix[cur]
            for tok in range(self.K):
                if tok == cur:
                    continue
                if base_without + row[tok] >= self.threshold:
                    out.append(base + delta * (int(tok) - cur))
        return out

    def rank(self, query_index: int) -> np.ndarray:
        q = self.tokens[query_index]
        if len(q) < self.kmer_k:
            return np.zeros(0, dtype=np.int64)
        diag_counts: dict[int, dict[int, int]] = defaultdict(dict)
        for qpos in range(len(q) - self.kmer_k + 1):
            for kid in self.expand(q[qpos : qpos + self.kmer_k]):
                for ti, tpos in self.index.get(kid, ()):
                    if ti == query_index:
                        continue
                    diag = qpos - tpos
                    diag_counts[ti][diag] = diag_counts[ti].get(diag, 0) + 1
        scored = [(ti, max((count for count in counts.values() if count >= 2), default=-1)) for ti, counts in diag_counts.items()]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (-item[1], item[0]))
        candidates = np.asarray([ti for ti, _score in scored[: self.top_n]], dtype=np.int64)
        if len(candidates) == 0:
            return np.zeros(0, dtype=np.int64)
        sw = np.asarray(
            [
                smith_waterman_numba(q, self.tokens[int(ti)], self.score_matrix, GAP_OPEN, GAP_EXTEND)
                for ti in candidates
            ],
            dtype=np.float64,
        )
        return candidates[np.argsort(-sw, kind="stable")]


def evaluate_ranker(
    name: str,
    ranker: SamePipelineRanker,
    fam_q: list[int],
    clan_q: list[int],
    families: list[str],
    clans: list[str],
) -> tuple[dict, dict]:
    fam_pool = np.asarray(families, dtype=object)
    clan_pool = np.asarray(clans, dtype=object)
    all_idx = np.arange(len(families))
    per_query = {"family": [], "clan": []}

    t0 = time.time()
    fam_r1 = []
    fam_sens = []
    rank_cache = {}
    for n, q in enumerate(fam_q, start=1):
        ranked = ranker.rank(q)
        rank_cache[q] = ranked
        valid = all_idx != q
        rel = (fam_pool == fam_pool[q]) & valid
        neutral = ((clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & valid) if clan_pool[q] else None
        recall1 = int(len(ranked) > 0 and fam_pool[int(ranked[0])] == fam_pool[q])
        sens = first_fp_sensitivity(ranked, rel, neutral)
        fam_r1.append(recall1)
        fam_sens.append(sens)
        per_query["family"].append({"query": int(q), "recall1": recall1, "sens": float(sens)})
        if n % 30 == 0:
            print(f"{name}: family {n}/{len(fam_q)} ({time.time() - t0:.0f}s)", flush=True)

    clan_r1 = []
    clan_sens = []
    for q in clan_q:
        ranked = rank_cache.get(q)
        if ranked is None:
            ranked = ranker.rank(q)
            rank_cache[q] = ranked
        valid = all_idx != q
        rel = (clan_pool == clan_pool[q]) & (fam_pool != fam_pool[q]) & valid
        if not rel.any():
            continue
        neutral = (fam_pool == fam_pool[q]) & valid
        recall1 = int(len(ranked) > 0 and clan_pool[int(ranked[0])] == clan_pool[q])
        sens = first_fp_sensitivity(ranked, rel, neutral)
        clan_r1.append(recall1)
        clan_sens.append(sens)
        per_query["clan"].append({"query": int(q), "recall1": recall1, "sens": float(sens)})

    summary = {
        "family_recall1": float(np.mean(fam_r1)) if fam_r1 else 0.0,
        "family_sens_1fp": float(np.mean(fam_sens)) if fam_sens else 0.0,
        "clan_recall1": float(np.mean(clan_r1)) if clan_r1 else 0.0,
        "clan_sens_1fp": float(np.mean(clan_sens)) if clan_sens else 0.0,
        "n_family_queries": len(fam_sens),
        "n_clan_queries": len(clan_sens),
        "time_sec": float(time.time() - t0),
    }
    return summary, per_query


def load_riboseek_tokens(n_entries: int) -> tuple[list[np.ndarray], np.ndarray]:
    if not RIBOSEEK_CACHE.exists():
        raise FileNotFoundError(f"Missing {RIBOSEEK_CACHE}; run scripts/02_benchmark/riboseek_baseline.py first")
    encoded = json.load(open(RIBOSEEK_CACHE))
    tokens = [np.asarray(encoded.get(str(i), {}).get("labels", []), dtype=np.int64) for i in range(n_entries)]
    with R.as_file(R.files("riboseek.data").joinpath("demo_db.json")) as path:
        pretrained = Searcher.from_pretrained("sa20", db=str(path))
    return tokens, np.asarray(pretrained.alphabet.score_matrix, dtype=np.float32)


def main() -> None:
    db = torch.load(resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
    entries = db["entries"]
    families = attach_labels(entries, load_labels(resolve("Splits/labels.tsv")))
    clans = attach_labels(entries, load_labels(resolve("Splits/labels.tsv"), label_column="clan_acc"))
    fam_q, clan_q = load_query_sets(entries, families, clans)
    print(f"queries: family={len(fam_q)} clan={len(clan_q)}", flush=True)

    rf_tokens = [np.asarray(entry["tokens"], dtype=np.int64) for entry in entries]
    ribo_tokens, ribo_matrix_raw = load_riboseek_tokens(len(entries))
    rf_matrix_raw = torch.load(resolve("files/models/sub_matrix.pt"), map_location="cpu", weights_only=False)["matrix"].numpy()
    self_blosum_meta = {}
    if MATRIX_MODE == "identity":
        rf_matrix = identity_matrix(80)
        ribo_matrix = identity_matrix(int(ribo_matrix_raw.shape[0]))
    elif MATRIX_MODE == "self_blosum":
        train_pairs, pair_meta = load_train_pairs(entries)
        print(
            f"self-BLOSUM training pool: {len(train_pairs)} intra-train pairs "
            f"with symmetric TM>={BLOSUM_TM_THRESH}",
            flush=True,
        )
        print("Building RFoldseek self-BLOSUM", flush=True)
        rf_matrix, rf_meta = build_self_blosum(rf_tokens, train_pairs, 80)
        print("Building RiboSeek self-BLOSUM", flush=True)
        ribo_matrix, ribo_meta = build_self_blosum(ribo_tokens, train_pairs, int(ribo_matrix_raw.shape[0]))
        self_blosum_meta = {
            "train_pair_pool": pair_meta,
            "RFoldseek_tokens_same_pipeline": rf_meta,
            "RiboSeek_sa20_same_pipeline": ribo_meta,
        }
    else:
        target_diag = float(np.mean(np.diag(rf_matrix_raw)))
        rf_matrix = normalize_matrix(rf_matrix_raw, target_diag)
        ribo_matrix = normalize_matrix(ribo_matrix_raw, target_diag)

    configs = {
        "RFoldseek_tokens_same_pipeline": (rf_tokens, rf_matrix),
        "RiboSeek_sa20_same_pipeline": (ribo_tokens, ribo_matrix),
    }
    summaries = {}
    per_query = {}
    run_provenance = provenance()
    for name, (tokens, matrix) in configs.items():
        empty = int(sum(len(seq) < KMER_K for seq in tokens))
        print(f"\n{name}: alphabet={matrix.shape[0]} empty_or_short={empty}", flush=True)
        ranker = SamePipelineRanker(tokens, matrix)
        summary, pq = evaluate_ranker(name, ranker, fam_q, clan_q, families, clans)
        summary.update(
            {
                "alphabet_size": int(matrix.shape[0]),
                "kmer_k": KMER_K,
                "top_n": TOPN,
                "threshold_fraction": THRESH_FRAC,
                "gap_open": GAP_OPEN,
                "gap_extend": GAP_EXTEND,
                "matrix_mode": MATRIX_MODE,
                "matrix_rule": (
                    "match=+1,mismatch=-1"
                    if MATRIX_MODE == "identity"
                    else (
                        "alphabet-specific Structure-BLOSUM learned from the same train US-align pairs"
                        if MATRIX_MODE == "self_blosum"
                        else "native matrix scaled to RFoldseek diagonal mean"
                    )
                ),
                "matrix_diag_mean_after_scaling": float(np.mean(np.diag(matrix))),
                "n_empty_or_short_sequences": empty,
            }
        )
        if MATRIX_MODE == "self_blosum":
            summary["self_blosum"] = self_blosum_meta[name]
        summaries[name] = summary
        per_query[name] = pq
        print(json.dumps(summary, indent=2), flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if MATRIX_MODE == "self_blosum":
        run_provenance["self_blosum"] = self_blosum_meta
    json.dump({"_provenance": run_provenance, **summaries}, open(OUT_SUMMARY, "w"), indent=2)
    json.dump({"_provenance": run_provenance, **per_query}, open(OUT_PER_QUERY, "w"), indent=2)
    print(f"\nWrote {OUT_SUMMARY}")
    print(f"Wrote {OUT_PER_QUERY}")


if __name__ == "__main__":
    main()
