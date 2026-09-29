"""Seed-replicated summary of family- vs PDB-disjoint CV.

For each VQ-VAE training seed: family / clan Sens@1FP macro-averaged over Rfam
families or clans, and the paired Δ (family-disjoint − PDB-disjoint). Reports
mean ± SD across seeds, the per-seed Δ, and a family/clan-block bootstrap of Δ
on seed-averaged per-query sensitivities.

Sensitivity analyses (when present): homology-disjoint folds (--split homology;
labelled and unlabelled training chains related to a held-out chain removed) and
the matched random-removal control on PDB-disjoint folds (--split pdb-random),
written under "sensitivity".
"""
import sys as _sys, os as _os

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", ".."))
from paths import resolve

import json
from collections import defaultdict
import numpy as np

SEEDS = [42, 43, 44, 45, 46]
BLOCK = {"fam": "family", "clan": "clan_acc"}
_REPO = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), "..", ".."))
OUT = _os.path.join(_REPO, "files/ablation/cv_family_disjoint_seed_summary.json")


def load(split, seed):
    sfx = "" if seed == 42 else f"_seed{seed}"
    d = json.load(open(resolve(f"files/ablation/cv_{split}_disjoint_5fold{sfx}.json")))
    return {r["id"]: r for r in d["per_query"]}


P = {(s, sd): load(s, sd) for s in ("pdb", "family") for sd in SEEDS}
ids = sorted(P[("pdb", 42)])
assert all(set(v) == set(ids) for v in P.values())


def grouped(recs, key):
    out = defaultdict(list)
    for i in ids:
        if recs[i][key] is not None:
            out[recs[i][BLOCK[key]]].append(recs[i][key])
    return {k: float(np.mean(v)) for k, v in out.items()}


def seed_mean(split, key):
    """per-query value averaged over seeds, then grouped"""
    avg = {i: dict(P[(split, 42)][i]) for i in ids}
    for i in ids:
        if avg[i][key] is not None:
            avg[i][key] = float(np.mean([P[(split, sd)][i][key] for sd in SEEDS]))
    return grouped(avg, key)


out = {"seeds": SEEDS, "n_queries": len(ids)}
rng = np.random.default_rng(123)
for key in ("fam", "clan"):
    per = {"pdb": [], "family": [], "delta": []}
    for sd in SEEDS:
        a, b = grouped(P[("family", sd)], key), grouped(P[("pdb", sd)], key)
        per["pdb"].append(float(np.mean(list(b.values()))))
        per["family"].append(float(np.mean(list(a.values()))))
        per["delta"].append(float(np.mean([a[k] - b[k] for k in a])))
    a, b = seed_mean("family", key), seed_mean("pdb", key)
    d = np.array([a[k] - b[k] for k in sorted(a)])
    bs = d[rng.integers(0, len(d), (2000, len(d)))].mean(1)
    out[key] = {
        "n_blocks": len(d),
        "per_seed": per,
        "pdb_mean": float(np.mean(per["pdb"])), "pdb_sd": float(np.std(per["pdb"], ddof=1)),
        "family_mean": float(np.mean(per["family"])), "family_sd": float(np.std(per["family"], ddof=1)),
        "delta_mean": float(np.mean(per["delta"])), "delta_sd": float(np.std(per["delta"], ddof=1)),
        "delta_seedavg_block_ci95": [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))],
    }
    print(f"{key}: PDB {out[key]['pdb_mean']:.3f}±{out[key]['pdb_sd']:.3f}  "
          f"family {out[key]['family_mean']:.3f}±{out[key]['family_sd']:.3f}  "
          f"Δ per seed {[round(x, 3) for x in per['delta']]}  "
          f"Δ {out[key]['delta_mean']:+.3f}±{out[key]['delta_sd']:.3f}  "
          f"seed-avg block CI {out[key]['delta_seedavg_block_ci95']}")
# within-split seed-to-seed variability, per fold
for s in ("pdb", "family"):
    fold = defaultdict(lambda: defaultdict(list))
    for sd in SEEDS:
        for r in P[(s, sd)].values():
            fold[r["fold"]][sd].append(r["fam"])
    out[f"{s}_fold_fam_by_seed"] = {
        str(f + 1): [float(np.mean(fold[f][sd])) for sd in SEEDS] for f in sorted(fold)
    }
    print(s, {k: [round(x, 3) for x in v] for k, v in out[f"{s}_fold_fam_by_seed"].items()})
EXTRA = [s for s in ("homology", "pdb-random")
         if all(_os.path.exists(resolve(f"files/ablation/cv_{s}_disjoint_5fold{'' if sd == 42 else f'_seed{sd}'}.json"))
                for sd in SEEDS)]
if EXTRA:
    for s in EXTRA:
        for sd in SEEDS:
            P[(s, sd)] = load(s, sd)
    out["sensitivity"] = {}
    for key in ("fam", "clan"):
        o = out["sensitivity"][key] = {}
        for s in ["pdb", "family"] + EXTRA:
            v = [float(np.mean(list(grouped(P[(s, sd)], key).values()))) for sd in SEEDS]
            o[s] = {"per_seed": v, "mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
        for a_, b_ in (("homology", "pdb"), ("pdb-random", "pdb"), ("homology", "pdb-random"),
                       ("homology", "family")):
            if a_ in EXTRA and (b_ in EXTRA or b_ in ("pdb", "family")):
                A, Bm = seed_mean(a_, key), seed_mean(b_, key)
                d = np.array([A[k] - Bm[k] for k in sorted(A)])
                bs = d[np.random.default_rng(123).integers(0, len(d), (2000, len(d)))].mean(1)
                o[f"{a_} - {b_}"] = [float(d.mean()), *map(float, np.percentile(bs, [2.5, 97.5]))]
                print(f"{key}: {a_} - {b_} {d.mean():+.3f} CI {np.percentile(bs, [2.5, 97.5]).round(3)}")
json.dump(out, open(OUT, "w"), indent=1)
print(OUT)
