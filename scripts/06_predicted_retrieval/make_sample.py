"""Sampling design for the NuFold 9-14-2026 benchmark.

Universe: every entry of the release with sequence_length >= MIN_LEN.

Accuracy database (~50k)
  labelled    every Rfam family contributes up to FAMILY_CAP members, drawn by
              round-robin over the family's taxa (taxa in random order,
              members in random order within a taxon), so that a family's
              sampled members are spread across as many species as possible.
  unlabelled  matched to the labelled part: for every stratum defined by
              organism x length bin x mean-pLDDT bin, as many unlabelled
              entries are drawn at random as there are labelled entries in
              that stratum. Where an organism has too few unlabelled entries
              in a stratum, the remainder is drawn from the same length x
              pLDDT stratum in any organism; any remaining shortfall is
              reported. Labelled targets therefore cannot be told apart from
              distractors by length or prediction confidence alone.
  No pLDDT filter: prediction confidence is analysed, not selected on.
  Only entries with an extracted structure file are eligible.

Independent random streams (SEED, SEED+1, ...) are used for the labelled
part, the unlabelled part, the queries and the speed ladder, so that changing
one component does not alter the others.

Queries
  predicted   families with >= P2P_MIN_MEMBERS sampled members spanning >= 2
              taxa; N_P2P_FAMILIES of them at random, one random member each.
              Evaluation is cross-species: same-family targets from the
              query's own taxon are neutral.
  experimental  held-out RNAsolo test chains of MIN_LEN..MAX_QUERY_LEN tokens
              whose Rfam family is present in the accuracy database.

Speed ladder
  nested random subsets of the universe (one permutation), for runtime only.

All randomness comes from SEED.
"""
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, REPO)
import paths                                                   # noqa: E402
HERE = paths.NUFOLD_ROOT                                       # NuFold release: metadata/, structures/
META = os.path.join(HERE, "metadata", "data.csv")
OUT = os.path.join(REPO, "files", "predicted_retrieval", "sample")
SEED = 20260917
MIN_LEN = 30
FAMILY_CAP = 50
P2P_MIN_MEMBERS = 10
N_P2P_FAMILIES = 150
MAX_QUERY_LEN = 400
LADDER = [1_000, 10_000, 50_000, 100_000]
LENGTH_BINS = [30, 50, 75, 100, 150, 200, 300, 401]
PLDDT_BINS = [0, 60, 70, 80, 90, 101]
STRUCTURES = os.path.join(HERE, "structures", "rank1")

PAPER_DB = os.path.join(REPO, "files", "models", "db.pt")
LABELS = os.path.join(REPO, "Splits", "labels.tsv")
TEST = os.path.join(REPO, "Splits", "pdb_out", "test.json")


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def quartiles(v):
    v = np.asarray([x for x in v if np.isfinite(x)])
    return "median %.1f [%.1f, %.1f]" % (np.median(v), np.percentile(v, 25), np.percentile(v, 75))


def describe(name, rows, fam_of):
    tax = Counter(r["taxonomy_id"] for r in rows)
    lab = [r for r in rows if fam_of(r)]
    fams = Counter(fam_of(r) for r in lab)
    top = fams.most_common(1)[0] if fams else ("-", 0)
    print(f"\n[{name}] {len(rows):,} entries | taxa {len(tax)} | labelled {len(lab):,} "
          f"({len(lab)/max(1,len(rows)):.1%}) in {len(fams):,} families (largest {top[0]} = {top[1]:,})")
    print(f"   length {quartiles([fnum(r['sequence_length']) for r in rows])}   "
          f"pLDDT {quartiles([fnum(r['avg_plddt']) for r in rows])}")
    shares = ", ".join(f"{t} {c/len(rows):.1%}" for t, c in tax.most_common(6))
    print(f"   top taxa: {shares}")


def stratum(r):
    L, p = fnum(r["sequence_length"]), fnum(r["avg_plddt"])
    lb = int(np.searchsorted(LENGTH_BINS, L, side="right")) - 1
    pb = -1 if not np.isfinite(p) else int(np.searchsorted(PLDDT_BINS, p, side="right")) - 1
    return lb, pb


def main():
    rng = np.random.default_rng(SEED)            # labelled part
    rng_unl = np.random.default_rng(SEED + 1)    # unlabelled part
    rng_q = np.random.default_rng(SEED + 2)      # queries
    rng_ladder = np.random.default_rng(SEED + 3) # speed ladder
    os.makedirs(OUT, exist_ok=True)
    have = {f[: -len("_rank_1.pdb")] for f in os.listdir(STRUCTURES) if f.endswith("_rank_1.pdb")}
    all_rows = list(csv.DictReader(open(META)))
    no_file = sum(1 for r in all_rows if r["urs_id"] not in have)
    rows = [r for r in all_rows if fnum(r["sequence_length"]) >= MIN_LEN and r["urs_id"] in have]
    print(f"release entries {len(all_rows):,}; without a structure file {no_file}")
    fam_of = lambda r: r["rfam_family"] if r["rfam_family"] not in ("", "N/A") else ""
    describe("universe (>= %d nt)" % MIN_LEN, rows, fam_of)

    # ---------------- labelled: family cap with taxon round-robin
    byfam = defaultdict(lambda: defaultdict(list))
    for r in rows:
        f = fam_of(r)
        if f:
            byfam[f][r["taxonomy_id"]].append(r)
    labelled = []
    for f in sorted(byfam):
        taxa = sorted(byfam[f])
        rng.shuffle(taxa)
        pools = {t: list(rng.permutation(len(byfam[f][t]))) for t in taxa}
        picked = []
        while len(picked) < FAMILY_CAP and any(pools[t] for t in taxa):
            for t in taxa:
                if pools[t] and len(picked) < FAMILY_CAP:
                    picked.append(byfam[f][t][pools[t].pop()])
        labelled.extend(picked)

    # ---------------- unlabelled: matched on organism x length x pLDDT
    need = Counter((r["taxonomy_id"],) + stratum(r) for r in labelled)
    pool_org = defaultdict(list)          # (taxon, lbin, pbin) -> unlabelled entries
    for r in rows:
        if not fam_of(r):
            pool_org[(r["taxonomy_id"],) + stratum(r)].append(r)
    for k in pool_org:
        pool_org[k] = [pool_org[k][i] for i in rng_unl.permutation(len(pool_org[k]))]
    taken = set()
    unlabelled, shortfall = [], Counter()
    for key in sorted(need):
        pool = pool_org.get(key, [])
        k = min(need[key], len(pool))
        for r in pool[:k]:
            unlabelled.append(r)
            taken.add(r["urs_id"])
        if need[key] > k:
            shortfall[key[1:]] += need[key] - k
    exact = len(unlabelled)
    # fall back to the same length x pLDDT stratum in any organism
    pool_any = defaultdict(list)
    for r in rows:
        if not fam_of(r) and r["urs_id"] not in taken:
            pool_any[stratum(r)].append(r)
    unmet = 0
    for key in sorted(shortfall):
        pool = pool_any.get(key, [])
        idx = rng_unl.permutation(len(pool))[: shortfall[key]]
        unlabelled.extend(pool[i] for i in idx)
        unmet += shortfall[key] - len(idx)
    print(f"\n[unlabelled matching] {len(labelled):,} requested; {exact:,} matched on organism x length x pLDDT; "
          f"{len(unlabelled) - exact:,} matched on length x pLDDT only; {unmet} unmatched")

    accuracy_db = labelled + unlabelled
    describe("accuracy database", accuracy_db, fam_of)
    describe("  labelled part", labelled, fam_of)
    describe("  unlabelled part", unlabelled, fam_of)

    # ---------------- predicted queries
    members = defaultdict(list)
    for r in labelled:
        members[fam_of(r)].append(r)
    eligible = sorted(f for f, m in members.items()
                      if len(m) >= P2P_MIN_MEMBERS and len({x["taxonomy_id"] for x in m}) >= 2)
    chosen = sorted(rng_q.choice(eligible, size=min(N_P2P_FAMILIES, len(eligible)), replace=False))
    p2p = []
    for f in chosen:
        q = members[f][int(rng_q.integers(len(members[f])))]
        other = sum(1 for m in members[f] if m["taxonomy_id"] != q["taxonomy_id"])
        p2p.append(dict(urs_id=q["urs_id"], family=f, taxonomy_id=q["taxonomy_id"],
                        length=int(fnum(q["sequence_length"])), plddt=fnum(q["avg_plddt"]),
                        family_members=len(members[f]), cross_species_positives=other))
    print(f"\n[predicted queries] {len(eligible)} eligible families (>= {P2P_MIN_MEMBERS} members, >= 2 taxa); "
          f"chose {len(p2p)}; cross-species positives per query {quartiles([q['cross_species_positives'] for q in p2p])}; "
          f"query pLDDT {quartiles([q['plddt'] for q in p2p])}")

    # ---------------- experimental queries
    exp = []
    try:
        import torch
        import sys
        sys.path.insert(0, os.path.join(REPO, "src", "search"))
        from prefilter_rfam_benchmark import attach_labels, load_labels
        ents = torch.load(PAPER_DB, map_location="cpu", weights_only=False)["entries"]
        fam_exp = attach_labels(ents, load_labels(LABELS))
        id2i = {str(e["id"]): i for i, e in enumerate(ents)}
        present = set(members)
        for l in open(TEST):
            if not l.strip():
                continue
            cid = json.loads(l)["id"]
            i = id2i.get(cid)
            if i is None or not fam_exp[i] or fam_exp[i] not in present:
                continue
            L = len(ents[i]["tokens"])
            if MIN_LEN <= L <= MAX_QUERY_LEN:
                exp.append(dict(index=i, id=os.path.splitext(os.path.basename(cid))[0],
                                family=fam_exp[i], length=L, family_members=len(members[fam_exp[i]])))
        fc = Counter(q["family"] for q in exp)
        print(f"[experimental queries] {len(exp)} queries from {len(fc)} families; largest: {fc.most_common(4)}")
    except Exception as exc:                                  # reported, not fatal
        print(f"[experimental queries] not built: {type(exc).__name__}: {exc}")

    # ---------------- speed ladder
    perm = rng_ladder.permutation(len(rows))
    ladder = {str(n): [rows[i]["urs_id"] for i in perm[:n]] for n in LADDER if n < len(rows)}
    ladder[str(len(rows))] = [rows[i]["urs_id"] for i in perm]
    print(f"[speed ladder] sizes {[int(k) for k in ladder]}")

    # ---------------- write
    cols = ["urs_id", "taxonomy_id", "rfam_family", "sequence_length", "avg_plddt", "description"]
    with open(os.path.join(OUT, "accuracy_db.tsv"), "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(cols + ["part"])
        for part, rs in (("labelled", labelled), ("unlabelled", unlabelled)):
            for r in rs:
                w.writerow([r[c] for c in cols] + [part])
    json.dump(dict(seed=SEED, min_len=MIN_LEN, family_cap=FAMILY_CAP, queries=p2p),
              open(os.path.join(OUT, "p2p_queries.json"), "w"), indent=1)
    json.dump(dict(seed=SEED, queries=exp), open(os.path.join(OUT, "exp_queries.json"), "w"), indent=1)
    json.dump(dict(seed=SEED, min_len=MIN_LEN, ladder=ladder), open(os.path.join(OUT, "speed_ladder.json"), "w"))
    json.dump(dict(seed=SEED, min_len=MIN_LEN, family_cap=FAMILY_CAP, p2p_min_members=P2P_MIN_MEMBERS,
                   unlabelled_sampling="matched: organism x length x pLDDT", length_bins=LENGTH_BINS, plddt_bins=PLDDT_BINS,
                   n_p2p_families=N_P2P_FAMILIES, universe=len(rows), accuracy_db=len(accuracy_db),
                   labelled=len(labelled), unlabelled=len(unlabelled), p2p_queries=len(p2p),
                   exp_queries=len(exp)),
              open(os.path.join(OUT, "sample_summary.json"), "w"), indent=1)
    print(f"\nwrote manifests to {OUT}")


if __name__ == "__main__":
    main()
