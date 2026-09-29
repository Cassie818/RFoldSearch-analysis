"""Checks on the identity zones of Fig. 3a (Section 2.2 family-level Sens@1FP, 10 tie-break seeds).

Zones after Rost (1999), from the identity of unrelated chains (different Rfam family and
clan): above background (> 99th percentile, pairs above 80% excluded), twilight (50th-99th),
midnight (< 50th).

  1  paired differences RFoldseek - comparator in each zone, all families and with tRNA
     (RF00005) excluded; 95% hierarchical block-bootstrap CIs over families
  2  sensitivity of the midnight result to the cut-off (30th-70th percentile of the background)
  3  length-stratified zones: the background is taken separately for the length of the
     shorter chain of a pair (<100, 100-200, 200-1000, >=1000 nt; unrelated pairs with length
     ratio 0.8-1.25 whose shorter chain falls in the bin), and each query-target pair is placed
     by its percentile in the background of its own length bin

Writes files/production/retrieval/sequence_divergent/zone_checks.json.
"""
import csv
import itertools
import json
import os
import random

import numpy as np

from plot_fig3 import boot
from sequence_divergent_rescore import (DATA, REPO, _aligner, background_identity, load, rescore,
                                        seq_identity, unrelated)

CMP = ["Infernal", "nhmmer", "BLASTN", "RS-20", "RS-80", "Composition", "Length", "Shuffled", "US-align"]
MEANS = ["RFoldseek", "RFoldseek-TM", "US-align"] + CMP[:-1]
LBINS = [(0, 100), (100, 200), (200, 1000), (1000, 100000)]
MAXPAIRS = 3000
CAP = 0.80


def summarise(per, blocks, zones):
    out = {}
    for z in zones:
        out[z] = {}
        for lab, keep in (("all", lambda q: True), ("tRNA excluded", lambda q: blocks[q] != "RF00005")):
            qs = [q for q in per[z].get("RFoldseek", {}) if keep(q)]
            if not qs:
                continue
            r = dict(n=len(qs), families=len({blocks[q] for q in qs}),
                     family_counts={f: sum(blocks[q] == f for q in qs) for f in sorted({blocks[q] for q in qs})},
                     mean={m: float(np.mean([per[z][m][q] for q in qs])) for m in MEANS},
                     diff={m: boot({q: per[z]["RFoldseek"][q] - per[z][m][q] for q in qs}, blocks) for m in CMP})
            out[z][lab] = r
    return out


def show(title, res):
    print(f"\n#### {title}")
    for z, d in res.items():
        for lab, r in d.items():
            print(f"  {z:9s} {lab:14s} n={r['n']:3d} ({r['families']:2d} fam)  "
                  + "  ".join(f"{m}={r['mean'][m]:.3f}" for m in ("RFoldseek", "Infernal", "nhmmer", "RS-80")))
            print(" " * 28 + "  ".join(f"-{m} {d_:+.3f}[{lo:+.3f},{hi:+.3f}]" for m, (d_, lo, hi) in r["diff"].items()
                                        if m in ("Infernal", "nhmmer", "RS-80", "Composition", "US-align")))


def length_backgrounds(seed=0):
    R, _ = load()
    fam, clan = dict(zip(R["tags"], R["fam"])), dict(zip(R["tags"], R["clan"]))
    seqs = {r["chain_id"]: r["sequence"] for r in
            csv.DictReader(open(os.path.join(REPO, "files", "production", "chain_sequences.tsv")), delimiter="\t")}
    lab = [t for t in R["tags"] if fam[t]]
    al = _aligner()
    rng = random.Random(seed)
    bg = {}
    for lo, hi in LBINS:
        pairs = [(a, b) for a, b in itertools.combinations(lab, 2)
                 if unrelated(a, b, fam, clan) and 0.8 <= len(seqs[a]) / len(seqs[b]) <= 1.25
                 and lo <= min(len(seqs[a]), len(seqs[b])) < hi]
        fp = {tuple(sorted((fam[a], fam[b]))) for a, b in pairs}
        if len(pairs) > MAXPAIRS:
            pairs = rng.sample(pairs, MAXPAIRS)
        vals = np.sort([seq_identity(seqs[a], seqs[b], al) for a, b in pairs])
        bg[(lo, hi)] = dict(values=vals, family_pairs=len(fp))
        print(f"  background {lo}-{hi} nt: {len(vals)} pairs ({len(fp)} family pairs)  "
              f"p50={np.quantile(vals, 0.5):.3f} p99={np.quantile(vals, 0.99):.3f}")
    return bg, seqs


def main():
    report = {}
    _, v = background_identity()
    p50, p99 = np.quantile(v, [0.5, 0.99])
    report["global_background"] = dict(pairs=len(v), p50=float(p50), p99=float(p99))
    print(f"global background (40-200 nt, {len(v)} pairs): p50={p50:.4f} p99={p99:.4f}")

    zones = [("above", p99, CAP), ("twilight", p50, p99), ("midnight", 0.0, p50)]
    per, blocks = rescore(zones)
    report["zones"] = summarise(per, blocks, [z for z, _, _ in zones])
    show("1  global zones", report["zones"])

    cuts = {f"p{int(100 * c)}": float(np.quantile(v, c)) for c in (0.3, 0.4, 0.5, 0.6, 0.7)}
    per, _ = rescore([(k, 0.0, t) for k, t in cuts.items()])
    report["midnight_cutoff"] = {k: dict(cutoff=cuts[k], **summarise(per, blocks, [k])[k]) for k in cuts}
    print("\n#### 2  midnight cut-off sensitivity (RFoldseek - Infernal)")
    for k, d in report["midnight_cutoff"].items():
        print(f"  <{k} ({d['cutoff']:.3f}): " + "   ".join(
            f"{lab} n={d[lab]['n']} ({d[lab]['families']} fam) {d[lab]['diff']['Infernal'][0]:+.3f}"
            f"[{d[lab]['diff']['Infernal'][1]:+.3f},{d[lab]['diff']['Infernal'][2]:+.3f}]"
            for lab in ("all", "tRNA excluded") if lab in d))

    print("\n#### 3  length-stratified zones")
    bg, seqs = length_backgrounds()
    _, ident = load()
    pct = {}
    for q, d in ident.items():
        for t, x in d.items():
            L = min(len(seqs[q]), len(seqs[t]))
            vals = next(b["values"] for (lo, hi), b in bg.items() if lo <= L < hi)
            pct.setdefault(q, {})[t] = 2.0 if x >= CAP else float(np.searchsorted(vals, x, side="right") / len(vals))
    per, _ = rescore([("above", 0.99, 1.0001), ("twilight", 0.5, 0.99), ("midnight", 0.0, 0.5)], ident=pct)
    report["length_background"] = {f"{lo}-{hi}": dict(pairs=len(b["values"]), family_pairs=b["family_pairs"],
                                                      p50=float(np.quantile(b["values"], 0.5)),
                                                      p99=float(np.quantile(b["values"], 0.99)))
                                    for (lo, hi), b in bg.items()}
    report["length_zones"] = summarise(per, blocks, ["above", "twilight", "midnight"])
    show("3  length-stratified zones", report["length_zones"])

    out = os.path.join(DATA, "zone_checks.json")
    json.dump(report, open(out, "w"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
