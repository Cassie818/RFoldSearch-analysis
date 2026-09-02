"""Does a RANDOM sample of database chains work as the null, instead of a
known-unrelated sample? Random sampling is what a deployed search can actually
do for a novel query. Here all 884 scores per query are computed once, then
three nulls are compared on the SAME held-out unrelated targets:
  (a) unrelated-200   : the idealised null (needs relationships known in advance)
  (b) random-200      : the deployable null (may include related chains)
  (c) random-200-trim : same, with the top 10% of scores censored before fitting
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np, torch
from scipy.stats import gumbel_r

PC = os.path.expanduser("~/Desktop/RFoldseek-paper-code")
sys.path.insert(0, os.path.join(PC, "scripts", "03_ablations"))
os.environ.setdefault("RFS_ALPHABET_CONTROL_MATRIX", "native")
import fig2ef as F  # noqa: E402
from smith_waterman import smith_waterman_numba as SW  # noqa: E402

GO, GE = F.GAP_OPEN, F.GAP_EXTEND
B, TM_MAX, SEED = 200, 0.30, 0
GRID = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]


def bshuf(seq, L, rng):
    b = [seq[i:i + L] for i in range(0, len(seq), L)]
    rng.shuffle(b)
    return np.concatenate(b)


def fit(scores, trim=0.0):
    s = np.sort(scores)
    if trim > 0:
        s = s[: max(10, int(len(s) * (1 - trim)))]
    return gumbel_r.fit(s)


def main():
    db = torch.load(F.resolve("files/models/db.pt"), map_location="cpu", weights_only=False)
    ents = db["entries"]
    toks = [np.asarray(e["tokens"], dtype=np.int64) for e in ents]
    mat = torch.load(F.resolve("files/models/sub_matrix.pt"), map_location="cpu",
                     weights_only=False)["matrix"].numpy().astype(np.float32)
    lab = F.resolve("Splits/labels.tsv")
    fam = F.attach_labels(ents, F.load_labels(lab))
    clan = F.attach_labels(ents, F.load_labels(lab, label_column="clan_acc"))
    famq, _ = F.load_query_sets(ents, fam, clan)
    SW(toks[0], toks[1], mat, GO, GE)
    tmc = {}
    for line in open(F.resolve("files/production/retrieval/usalign_full.cache.jsonl")):
        r = json.loads(line)
        tmc[(int(r["i"]), int(r["j"]))] = 0.5 * (float(r["tm_query"]) + float(r["tm_target"]))

    def unrel(qi, j):
        if fam[j] == fam[qi]: return False
        if clan[qi] and clan[j] and clan[j] == clan[qi]: return False
        t = tmc.get((qi, j), tmc.get((j, qi)))
        return True if t is None else t < TM_MAX

    rng = np.random.default_rng(SEED)
    n = len(toks)
    variants = ["shuffle-200", "unrelated-200", "random-200", "random-200-trim10"]
    obs = {v: {str(e): [] for e in GRID} for v in variants}
    contam, used, t0 = [], 0, time.time()
    for k, qi in enumerate(famq, 1):
        q = toks[qi]
        others = np.array([j for j in range(n) if j != qi])
        sc = np.array([SW(q, toks[int(j)], mat, GO, GE) for j in others])   # one pass
        um = np.array([unrel(qi, int(j)) for j in others])
        U = np.where(um)[0]
        if len(U) < B + 50:
            continue
        used += 1
        pu = rng.permutation(len(U)); cal_u, held = U[pu[:B]], U[pu[B:]]
        ridx = rng.choice(len(others), size=B, replace=False)
        contam.append(float((~um[ridx]).mean()))
        shuf = np.array([SW(bshuf(q, 4, rng), toks[int(j)], mat, GO, GE)
                         for j in rng.choice(others, size=B, replace=False)])
        fits = {"shuffle-200": fit(shuf),
                "unrelated-200": fit(sc[cal_u]),
                "random-200": fit(sc[ridx]),
                "random-200-trim10": fit(sc[ridx], trim=0.10)}
        scale = (n - 1) / len(held)
        for v, (mu, beta) in fits.items():
            E = (n - 1) * gumbel_r.sf(sc[held], loc=mu, scale=beta)
            for e in GRID:
                obs[v][str(e)].append(float((E <= e).sum()) * scale)
        if k % 25 == 0:
            print(f"  {k}/{len(famq)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nqueries used: {used}   mean contamination of a random 200: {np.mean(contam)*100:.1f}% related")
    print("\n%-20s %s" % ("null", "  ".join(f"{e:>9g}" for e in GRID)))
    out = {"_config": {"B": B, "tm_max": TM_MAX, "queries_used": used,
                       "mean_contamination_random_sample": float(np.mean(contam)),
                       "scorer": "token-only Structure-BLOSUM (w_nt=0)"}, "observed_fp_per_query_full_db_scale": {}}
    for v in variants:
        row = [float(np.mean(obs[v][str(e)])) for e in GRID]
        out["observed_fp_per_query_full_db_scale"][v] = {str(e): r for e, r in zip(GRID, row)}
        print("%-20s %s" % (v, "  ".join(f"{r:>9.3f}" for r in row)))
    print("\n(nominal expectation on the held-out subset is lower than E itself;\n compare variants to each other and to 'unrelated-200')")
    p = Path(PC) / "files/ablation/evalue_calibration_all_nulls.json"
    json.dump(out, open(p, "w"), indent=2); print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
