"""Learn a BLOSUM-style substitution matrix from token alignments
within the same Rfam family.

Method
------
For every pair of chains that share a primary Rfam family, align their
token sequences with Smith--Waterman using an *identity* matrix as
initial scoring (match = +1, mismatch = 0, affine gaps). Collect the
matched (token_i, token_j) pairs along the optimal local trace and
estimate

    f_ij  = observed joint frequency of (i, j) in same-family alignments
    f_i   = marginal token frequency in the same alignment column set
    s_ij  = log2 ( f_ij / (f_i * f_j) ) · SCALE

This is the same construction as BLOSUM (Henikoff & Henikoff 1992) but
on the structural-alphabet alignments inside one Rfam family.

Outputs
-------
files/production/sub_matrix_blosum.pt
    {"matrix": (K,K) float32, "n_codes": K, "method": "blosum-rfam"}
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from smith_waterman import smith_waterman_numba

REPO = Path(__file__).resolve().parents[2]
DB_PATH = REPO / "files" / "production" / "db.pt"
LABELS_TSV = REPO / "Splits" / "labels.tsv"
OUT_PT = REPO / "files" / "production" / "sub_matrix_blosum.pt"
TRAIN_JSON = REPO / "Splits" / "pdb_out" / "train.json"

MAX_PAIRS_PER_FAMILY = 100  # cap so giant families don't dominate
SCALE = 2.0  # multiply the log-odds (similar to BLOSUM62)
GAP_OPEN, GAP_EXTEND = -1.0, -0.2

# ─── Numba traceback to extract aligned column pairs ──────────────────
from numba import njit


@njit(cache=True)
def sw_traceback(q, t, sub, gap_open, gap_extend):
    Lq, Lt = q.shape[0], t.shape[0]
    H = np.zeros((Lq + 1, Lt + 1), dtype=np.float32)
    E = np.full((Lq + 1, Lt + 1), -1e9, dtype=np.float32)
    F = np.full((Lq + 1, Lt + 1), -1e9, dtype=np.float32)
    TB = np.zeros((Lq + 1, Lt + 1), dtype=np.int8)
    best = 0.0
    bi = 0
    bj = 0
    for i in range(1, Lq + 1):
        qi = q[i - 1]
        for j in range(1, Lt + 1):
            tj = t[j - 1]
            E[i, j] = max(E[i, j - 1] + gap_extend, H[i, j - 1] + gap_open)
            F[i, j] = max(F[i - 1, j] + gap_extend, H[i - 1, j] + gap_open)
            m = H[i - 1, j - 1] + sub[qi, tj]
            v = 0.0
            tb = 0
            if m > v:
                v = m
                tb = 1  # diag
            if E[i, j] > v:
                v = E[i, j]
                tb = 2  # gap in q (left)
            if F[i, j] > v:
                v = F[i, j]
                tb = 3  # gap in t (up)
            H[i, j] = v
            TB[i, j] = tb
            if v > best:
                best = v
                bi = i
                bj = j
    # traceback
    pairs_i = np.empty(Lq + Lt, dtype=np.int64)
    pairs_j = np.empty(Lq + Lt, dtype=np.int64)
    n = 0
    i, j = bi, bj
    while i > 0 and j > 0 and TB[i, j] != 0:
        if TB[i, j] == 1:
            pairs_i[n] = q[i - 1]
            pairs_j[n] = t[j - 1]
            n += 1
            i -= 1
            j -= 1
        elif TB[i, j] == 2:
            j -= 1
        else:
            i -= 1
    return pairs_i[:n], pairs_j[:n], best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DB_PATH))
    ap.add_argument("--labels", default=str(LABELS_TSV))
    ap.add_argument("--out", default=str(OUT_PT))
    ap.add_argument(
        "--fit_entries",
        default=str(TRAIN_JSON),
        help="JSON/JSONL entries used to estimate the matrix. "
        "Defaults to the PDB-out training split to avoid "
        "using evaluation labels.",
    )
    ap.add_argument(
        "--allow_full_db_fit",
        action="store_true",
        help="Estimate from all labelled DB entries. Use only for "
        "diagnostic ablations because this uses evaluation "
        "labels when the DB contains test chains.",
    )
    args = ap.parse_args()

    db = torch.load(args.db, map_location="cpu", weights_only=False)
    entries = db["entries"]
    K = int(db["n_codes"])
    print(f"Loaded {len(entries)} chains, K={K}")

    labels: dict[str, str] = {}
    with open(args.labels) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            labels[row["entry_id"]] = row.get("rfam_acc", "") or ""

    fit_ids: set[str] | None = None
    fit_path = Path(args.fit_entries) if args.fit_entries else None
    if fit_path and fit_path.exists() and not args.allow_full_db_fit:
        fit_ids = set()
        text = fit_path.read_text().strip()
        try:
            blob = json.loads(text)
            rows = blob if isinstance(blob, list) else blob.get("entries", blob.get("data", []))
        except json.JSONDecodeError:
            rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        for row in rows:
            eid = row.get("id") or row.get("entry_id") or row.get("path")
            if eid:
                fit_ids.add(str(eid))
        print(f"Restricting BLOSUM estimation to {len(fit_ids)} fit entries: {fit_path}")
    elif not args.allow_full_db_fit:
        print(f"WARNING: fit_entries not found ({fit_path}); using all labelled DB entries.")
    else:
        print("WARNING: estimating BLOSUM matrix from all labelled DB entries.")

    fam_to_idx: dict[str, list[int]] = defaultdict(list)
    for i, e in enumerate(entries):
        eid = str(e["id"])
        if fit_ids is not None and eid not in fit_ids:
            continue
        fam = labels.get(eid, "")
        if fam:
            fam_to_idx[fam].append(i)

    # only families with >=2 members
    fam_to_idx = {f: ids for f, ids in fam_to_idx.items() if len(ids) >= 2}
    print(
        f"Same-family pools: {len(fam_to_idx)} families, "
        f"sizes {sorted([len(v) for v in fam_to_idx.values()], reverse=True)[:10]} ..."
    )

    # identity scoring for the alignment that *generates* the BLOSUM signal.
    # Match = +1, mismatch = 0; affine gaps.
    sub_id = np.zeros((K, K), dtype=np.float32)
    np.fill_diagonal(sub_id, 1.0)

    # Accumulate joint counts
    joint = np.zeros((K, K), dtype=np.float64)
    n_alignments = 0
    n_aligned_pairs = 0

    rng = np.random.default_rng(0)
    for fam, ids in fam_to_idx.items():
        pairs = [(a, b) for a in ids for b in ids if a < b]
        if len(pairs) > MAX_PAIRS_PER_FAMILY:
            pairs = [
                pairs[k] for k in rng.choice(len(pairs), size=MAX_PAIRS_PER_FAMILY, replace=False)
            ]
        for a, b in pairs:
            q = entries[a]["tokens"].astype(np.int64)
            t = entries[b]["tokens"].astype(np.int64)
            pi, pj, _ = sw_traceback(q, t, sub_id, GAP_OPEN, GAP_EXTEND)
            if pi.size == 0:
                continue
            # accumulate symmetrically
            for x, y in zip(pi, pj):
                joint[x, y] += 1
                joint[y, x] += 1
            n_alignments += 1
            n_aligned_pairs += int(pi.size)
    print(f"  {n_alignments} same-family alignments, " f"{n_aligned_pairs} aligned (i,j) pairs")

    # Marginals (symmetric)
    total = joint.sum()
    p_ij = joint / max(total, 1.0)  # joint probability
    p_i = p_ij.sum(axis=1)  # marginal token frequency
    p_i = np.maximum(p_i, 1e-8)

    # BLOSUM log-odds: s_ij = log2(p_ij / (p_i p_j))
    expected = np.outer(p_i, p_i)
    log_odds = np.log2(np.maximum(p_ij, 1e-12) / np.maximum(expected, 1e-12))
    log_odds *= SCALE
    log_odds = log_odds.astype(np.float32)

    # Zero out tokens with vanishing marginal (numerical hygiene)
    dead = p_i < 1e-6
    if dead.any():
        log_odds[dead, :] = 0.0
        log_odds[:, dead] = 0.0
        print(f"  {int(dead.sum())} tokens had vanishing marginal " f"(zeroed)")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "matrix": torch.from_numpy(log_odds),
            "n_codes": K,
            "method": "blosum-rfam",
            "scale": SCALE,
            "gap_open": GAP_OPEN,
            "gap_extend": GAP_EXTEND,
            "fit_entries": str(fit_path) if fit_path else "",
            "allow_full_db_fit": bool(args.allow_full_db_fit),
            "n_fit_entries": len(fit_ids) if fit_ids is not None else len(entries),
            "n_alignments": n_alignments,
            "n_aligned_pairs": n_aligned_pairs,
            "marginal_freq": torch.from_numpy(p_i.astype(np.float32)),
        },
        args.out,
    )
    print(f"Saved {args.out}")

    print(f"\nDiagonal (self-substitution) summary:")
    diag = np.diag(log_odds)
    print(
        f"  min/mean/median/max = "
        f"{diag.min():.2f} / {diag.mean():.2f} / "
        f"{np.median(diag):.2f} / {diag.max():.2f}"
    )
    print(f"Off-diagonal:")
    off = log_odds[~np.eye(K, dtype=bool)]
    print(
        f"  min/mean/median/max = "
        f"{off.min():.2f} / {off.mean():.2f} / "
        f"{np.median(off):.2f} / {off.max():.2f}"
    )


if __name__ == "__main__":
    main()
