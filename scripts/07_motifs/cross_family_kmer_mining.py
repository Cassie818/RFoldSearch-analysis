"""Cross-family k-mer motif mining."""
from __future__ import annotations
import argparse, csv, json, os, random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
DB = REPO / "files" / "models" / "db.pt"
LABELS = REPO / "Splits" / "labels.tsv"
OUT = REPO / "files" / "motifs"
OUT.mkdir(parents=True, exist_ok=True)

# coords axis layout: [C3', C4', O4', C1', base_anchor]
C4_IDX = 1
ALL_ATOM_IDX = slice(None)


def _slice(use_all):
    return ALL_ATOM_IDX if use_all else C4_IDX


def _frag(coords_slice):
    """Flatten (k, atoms, 3) → (k*atoms, 3) if needed."""
    if coords_slice.ndim == 3:
        return coords_slice.reshape(-1, 3)
    return coords_slice


def _valid_coords(entry):
    """Coordinates aligned to the compact token sequence."""
    coords = np.asarray(entry["coords"], dtype=np.float32)
    mask = np.asarray(entry["mask"], dtype=bool)
    coords = coords[mask]
    if len(coords) != len(entry["tokens"]):
        raise ValueError(
            f"Token/valid-coordinate length mismatch for {entry['id']}: "
            f"{len(entry['tokens'])} != {len(coords)}"
        )
    return coords


def load_labels():
    fam, clan = {}, {}
    with open(LABELS) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            fam[r["entry_id"]] = (r.get("rfam_acc") or "").strip() or None
            clan[r["entry_id"]] = (r.get("clan_acc") or "").strip() or None
    return fam, clan


def kabsch_rmsd(P, Q):
    """RMSD of P aligned to Q (both (N,3))."""
    Pc = P - P.mean(0, keepdims=True)
    Qc = Q - Q.mean(0, keepdims=True)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    P_al = Pc @ R.T
    return float(np.sqrt(np.mean(np.sum((P_al - Qc) ** 2, axis=1))))


def pairwise_rmsd(frags):
    """All-pairs Kabsch RMSD on a list of (k,3) coord fragments."""
    n = len(frags)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append(kabsch_rmsd(frags[i], frags[j]))
    return np.asarray(out, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min_fam", type=int, default=3,
                    help="minimum number of distinct Rfam families")
    ap.add_argument("--min_clan", type=int, default=2,
                    help="minimum number of distinct Rfam clans (set 2+ to "
                         "require true cross-clan conservation)")
    ap.add_argument("--min_chains", type=int, default=5)
    ap.add_argument("--min_occ", type=int, default=8)
    ap.add_argument("--max_pairs", type=int, default=400,
                    help="cap pairwise comparisons per candidate (random subsample)")
    ap.add_argument("--top_n_case", type=int, default=12)
    ap.add_argument("--rmsd_cutoff_case", type=float, default=1.0,
                    help="sub-Å cross-family median for case-study export")
    ap.add_argument("--exclude_clans_diverse", default="CL00111,CL00112",
                    help="comma-separated clans to exclude in the 'diverse' "
                         "case-study tier only (default: rRNA clans)")
    ap.add_argument("--exclude_clans_index", default="",
                    help="comma-separated Rfam clans to exclude before k-mer "
                         "enumeration; use this for discovery screens that "
                         "must remove rRNA/tRNA occurrences before counting")
    atom_group = ap.add_mutually_exclusive_group()
    atom_group.add_argument("--use_all_atoms", dest="use_all_atoms", action="store_true",
                            help="use all 5 RFoldseek anchor atoms for Kabsch "
                                 "(default; reproduces Section 2.4)")
    atom_group.add_argument("--c4_only", dest="use_all_atoms", action="store_false",
                            help="use only C4' for Kabsch as a sensitivity analysis")
    ap.set_defaults(use_all_atoms=True)
    ap.add_argument("--out_suffix", default="",
                    help="suffix appended to all output filenames")
    ap.add_argument("--seed", type=int, default=26)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    fam_lut, clan_lut = load_labels()
    db = torch.load(DB, map_location="cpu", weights_only=False)
    entries = db["entries"]
    print(f"loaded {len(entries)} chains")

    # ── Step 1: build k-mer occurrence index ────────────────────────────
    # kmer (tuple) -> list of (entry_idx, position, family, clan, chain_stem)
    index = defaultdict(list)
    exclude_index = {c.strip() for c in args.exclude_clans_index.split(",") if c.strip()}
    skipped_by_index_clan = 0
    for e_idx, e in enumerate(entries):
        toks = np.asarray(e["tokens"], dtype=np.int32)
        _valid_coords(e)  # validate compact token-to-coordinate alignment
        L = len(toks)
        fam = fam_lut.get(e["id"])
        clan = clan_lut.get(e["id"])
        if exclude_index and clan in exclude_index:
            skipped_by_index_clan += 1
            continue
        chain_stem = Path(e["id"]).stem
        if L < args.k or fam is None:
            continue
        for p in range(L - args.k + 1):
            kmer = tuple(int(t) for t in toks[p:p + args.k])
            index[kmer].append((e_idx, p, fam, clan, chain_stem))

    print(f"distinct k-mers (k={args.k}): {len(index)}")
    if exclude_index:
        print(f"excluded chains before indexing: {skipped_by_index_clan} "
              f"(clans={','.join(sorted(exclude_index))})")

    # ── Step 2: filter ──────────────────────────────────────────────────
    candidates = []
    for kmer, occs in index.items():
        fams = {o[2] for o in occs}
        clans = {o[3] for o in occs if o[3]}
        chains = {o[4] for o in occs}
        if (len(fams) >= args.min_fam
                and len(chains) >= args.min_chains
                and len(occs) >= args.min_occ
                and len(clans) >= args.min_clan):
            candidates.append((kmer, occs, fams, clans, chains))
    print(f"candidates after filter (>={args.min_fam} fam, "
          f">={args.min_clan} clan, "
          f">={args.min_chains} chain, >={args.min_occ} occ): {len(candidates)}")

    # ── Step 3: pairwise RMSD per candidate ─────────────────────────────
    rows = []
    case_rows = []
    atom_slc = _slice(args.use_all_atoms)
    for kmer, occs, fams, clans, chains in candidates:
        # Pull coordinate traces
        frags = []
        for e_idx, p, fam, clan, stem in occs:
            c = _valid_coords(entries[e_idx])[p:p + args.k, atom_slc, :]
            frags.append((_frag(c), fam, stem))

        # Cross-family pair sampling
        cross_pairs, within_pairs = [], []
        for i in range(len(frags)):
            for j in range(i + 1, len(frags)):
                same = (frags[i][1] == frags[j][1])
                (within_pairs if same else cross_pairs).append((i, j))
        rng.shuffle(cross_pairs)
        rng.shuffle(within_pairs)
        cross_pairs = cross_pairs[:args.max_pairs]
        within_pairs = within_pairs[:args.max_pairs]

        cross_rmsd = np.asarray(
            [kabsch_rmsd(frags[i][0], frags[j][0]) for i, j in cross_pairs],
            dtype=np.float32) if cross_pairs else np.array([], dtype=np.float32)
        within_rmsd = np.asarray(
            [kabsch_rmsd(frags[i][0], frags[j][0]) for i, j in within_pairs],
            dtype=np.float32) if within_pairs else np.array([], dtype=np.float32)

        row = {
            "kmer": "-".join(str(t) for t in kmer),
            "n_occ": len(occs),
            "n_fam": len(fams),
            "n_clan": len(clans),
            "n_chain": len(chains),
            "cross_med_rmsd":  float(np.median(cross_rmsd)) if len(cross_rmsd) else float("nan"),
            "cross_p90_rmsd":  float(np.percentile(cross_rmsd, 90)) if len(cross_rmsd) else float("nan"),
            "cross_n_pairs":   len(cross_rmsd),
            "within_med_rmsd": float(np.median(within_rmsd)) if len(within_rmsd) else float("nan"),
            "within_n_pairs":  len(within_rmsd),
            "families": ";".join(sorted(fams)),
            "clans":    ";".join(sorted(clans)),
        }
        rows.append(row)

        # Case-study eligibility: sub-Å cross-family median
        if (not np.isnan(row["cross_med_rmsd"])
                and row["cross_med_rmsd"] < args.rmsd_cutoff_case
                and len(cross_rmsd) >= 5):
            # pick one representative per family for the 3D overlay
            reps = {}
            for e_idx, p, fam, clan, stem in occs:
                if fam not in reps:
                    reps[fam] = (stem, p)
            for fam, (stem, p) in reps.items():
                case_rows.append({
                    "kmer": row["kmer"],
                    "rfam": fam,
                    "chain": stem,
                    "pos": p,
                    "n_fam": len(fams),
                    "cross_med_rmsd": row["cross_med_rmsd"],
                })

    rows.sort(key=lambda r: (r["cross_med_rmsd"] if not np.isnan(r["cross_med_rmsd"]) else 1e9))

    # ── Step 4: random length-matched control ───────────────────────────
    control_rmsd = []
    n_ctrl_kmers = min(2000, len(candidates) * 2)
    entries_with_fam = [
        (i, e) for i, e in enumerate(entries)
        if fam_lut.get(e["id"]) and len(e["tokens"]) >= args.k
        and (not exclude_index or clan_lut.get(e["id"]) not in exclude_index)
    ]
    for _ in range(n_ctrl_kmers):
        # sample ~20 fragments from random chain positions
        frags = []
        attempts = 0
        while len(frags) < 20 and attempts < 80:
            attempts += 1
            e_idx, e = random.choice(entries_with_fam)
            L = len(e["tokens"])
            p = rng.integers(0, L - args.k + 1)
            c = _valid_coords(e)[p:p + args.k, atom_slc, :]
            frags.append(_frag(c))
        if len(frags) < 4:
            continue
        pairs = [(i, j) for i in range(len(frags))
                 for j in range(i + 1, len(frags))]
        rng.shuffle(pairs)
        pairs = pairs[:50]
        for i, j in pairs:
            control_rmsd.append(kabsch_rmsd(frags[i], frags[j]))

    # ── Step 5: write outputs ───────────────────────────────────────────
    sfx = args.out_suffix
    cand_p = OUT / f"kmer_candidates{sfx}.tsv"
    val_p = OUT / f"motif_rmsd_validation{sfx}.tsv"
    case_p = OUT / f"case_study_top{sfx}.tsv"
    ctrl_p = OUT / f"random_control_rmsd{sfx}.tsv"
    div_p_target = OUT / f"case_study_diverse{sfx}.tsv"

    keys = ["kmer", "n_occ", "n_fam", "n_clan", "n_chain",
            "cross_med_rmsd", "cross_p90_rmsd", "cross_n_pairs",
            "within_med_rmsd", "within_n_pairs", "families", "clans"]
    with open(cand_p, "w") as f:
        w = csv.DictWriter(f, keys, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {cand_p}  ({len(rows)} rows)")

    # validation table = same as candidates, but trimmed to those with
    # both cross and within pairs (the comparable set)
    with open(val_p, "w") as f:
        w = csv.DictWriter(f, keys, delimiter="\t")
        w.writeheader()
        for r in rows:
            if r["cross_n_pairs"] > 0 and r["within_n_pairs"] > 0:
                w.writerow(r)
    print(f"wrote {val_p}")

    # case-study tier 1: top N kmer by cross_med_rmsd (any clan)
    seen = [r["kmer"] for r in rows[:args.top_n_case]]
    case_top = [c for c in case_rows if c["kmer"] in seen]
    case_keys = ["kmer", "rfam", "chain", "pos", "n_fam", "cross_med_rmsd"]
    with open(case_p, "w") as f:
        w = csv.DictWriter(f, case_keys, delimiter="\t")
        w.writeheader()
        for c in case_top:
            w.writerow(c)
    print(f"wrote {case_p}  (top {args.top_n_case} kmers, "
          f"{len(case_top)} chain-occurrences)")

    # case-study tier 2: top kmer that span clans EXCLUDING configured ones
    exclude = set(c.strip() for c in args.exclude_clans_diverse.split(",")
                  if c.strip())
    diverse = []
    for r in rows:
        if np.isnan(r["cross_med_rmsd"]) or r["cross_n_pairs"] < 5:
            continue
        cls = set(r["clans"].split(";")) if r["clans"] else set()
        if cls & exclude:
            continue
        if r["n_clan"] < 2:
            continue
        if r["cross_med_rmsd"] >= args.rmsd_cutoff_case:
            continue
        diverse.append(r)
    diverse_top = [d["kmer"] for d in diverse[:args.top_n_case]]
    case_diverse = [c for c in case_rows if c["kmer"] in diverse_top]
    div_p = div_p_target
    with open(div_p, "w") as f:
        w = csv.DictWriter(f, case_keys, delimiter="\t")
        w.writeheader()
        for c in case_diverse:
            w.writerow(c)
    print(f"wrote {div_p}  ({len(diverse_top)} cross-clan-no-rRNA kmers, "
          f"{len(case_diverse)} chain-occurrences)")

    with open(ctrl_p, "w") as f:
        f.write("rmsd\n")
        for v in control_rmsd:
            f.write(f"{v:.4f}\n")
    print(f"wrote {ctrl_p}  ({len(control_rmsd)} random pairs)")

    # ── Quick summary stats ─────────────────────────────────────────────
    cross_meds = np.array([r["cross_med_rmsd"] for r in rows
                           if not np.isnan(r["cross_med_rmsd"])])
    if len(cross_meds):
        print("\n=== Summary ===")
        print(f"candidates with cross-family pairs : {len(cross_meds)}")
        print(f"  sub-1.0 Å median cross-fam RMSD  : {(cross_meds < 1.0).sum()}")
        print(f"  sub-0.5 Å median cross-fam RMSD  : {(cross_meds < 0.5).sum()}")
        print(f"  median over candidates           : {np.median(cross_meds):.3f} Å")
    multi_clan = [r for r in rows if r["n_clan"] >= 2]
    if multi_clan:
        mc = np.array([r["cross_med_rmsd"] for r in multi_clan
                       if not np.isnan(r["cross_med_rmsd"])])
        print(f"multi-clan candidates             : {len(multi_clan)}")
        if len(mc):
            print(f"  sub-1.0 Å (multi-clan)            : {(mc < 1.0).sum()}")
            print(f"  sub-0.5 Å (multi-clan)            : {(mc < 0.5).sum()}")
            print(f"  median (multi-clan)               : {np.median(mc):.3f} Å")
    if control_rmsd:
        ctrl = np.array(control_rmsd)
        print(f"random control median RMSD        : {np.median(ctrl):.3f} Å")
        print(f"random control p10                : {np.percentile(ctrl, 10):.3f} Å")


if __name__ == "__main__":
    main()
