"""Base-pair detection and a base-pair agreement score.

Motivation: the 3Dr alphabet is a per-nucleotide LOCAL descriptor and RNA
LDDT measures local backbone distance preservation, which A-form helices make
generic. Base pairing is the one signal that is neither local nor generic --
i and j can pair across fifty intervening nucleotides -- so it is the
candidate plausibly orthogonal to both.

Pairs are detected geometrically rather than with DSSR so that every chain in
the corpus can be annotated without an external dependency. The criterion is
a Watson-Crick/wobble approximation, not a full annotation: complementary
bases whose glycosidic nitrogens are hydrogen-bond close and whose C1' atoms
sit at helical separation.
"""
from __future__ import annotations

import numpy as np

PURINES = {"A", "G"}
PYRIMIDINES = {"C", "U"}
# glycosidic-edge nitrogen that carries the central Watson-Crick hydrogen bond
WC_NITROGEN = {"A": "N1", "G": "N1", "C": "N3", "U": "N3"}
COMPLEMENTARY = {frozenset("AU"), frozenset("GC"), frozenset("GU")}

N_DIST_MAX = 3.5        # central hydrogen bond, angstrom
C1_DIST_RANGE = (8.5, 11.5)
MIN_SEPARATION = 3      # nucleotides, excludes trivial near-neighbours


def detect_pairs(bases, wc_atoms, c1_atoms):
    """Base pairs as an (n, 2) array of ascending index pairs.

    ``bases`` is a sequence of one-letter codes, ``wc_atoms`` and ``c1_atoms``
    are (L, 3) arrays; rows that are NaN are treated as missing.
    """
    wc = np.asarray(wc_atoms, dtype=np.float64)
    c1 = np.asarray(c1_atoms, dtype=np.float64)
    n = len(bases)
    usable = np.isfinite(wc).all(1) & np.isfinite(c1).all(1)
    usable &= np.asarray([b in WC_NITROGEN for b in bases])
    idx = np.flatnonzero(usable)
    if len(idx) < 2:
        return np.zeros((0, 2), dtype=np.int64)

    dn = np.linalg.norm(wc[idx][:, None, :] - wc[idx][None, :, :], axis=-1)
    dc = np.linalg.norm(c1[idx][:, None, :] - c1[idx][None, :, :], axis=-1)
    sep = np.abs(idx[:, None] - idx[None, :])
    comp = np.zeros_like(dn, dtype=bool)
    for a in range(len(idx)):
        ba = bases[idx[a]]
        for b in range(a + 1, len(idx)):
            comp[a, b] = frozenset((ba, bases[idx[b]])) in COMPLEMENTARY
    ok = (comp & (dn < N_DIST_MAX) & (dc > C1_DIST_RANGE[0])
          & (dc < C1_DIST_RANGE[1]) & (sep >= MIN_SEPARATION))

    # one partner per nucleotide: resolve conflicts by shortest hydrogen bond
    candidates = sorted(((dn[a, b], idx[a], idx[b])
                         for a, b in zip(*np.nonzero(ok))), key=lambda x: x[0])
    taken = np.zeros(n, dtype=bool)
    pairs = []
    for _, i, j in candidates:
        if taken[i] or taken[j]:
            continue
        taken[i] = taken[j] = True
        pairs.append((int(i), int(j)))
    pairs.sort()
    return np.asarray(pairs, dtype=np.int64).reshape(-1, 2)


def bp_agreement(query_pairs, target_pairs, alignment, n_query_pairs=None):
    """Fraction of query base pairs recovered in the target under ``alignment``.

    ``alignment`` is the (n, 2) token/residue correspondence. A query pair
    counts as recovered when both of its nucleotides are aligned and their
    target partners are themselves a base pair. The denominator is every
    query base pair, so a pair whose nucleotides fall outside the aligned
    region counts against the score -- the same choice Foldseek makes for
    LDDT's denominator.

    Returns nan when the query has no base pairs, so an unstructured query
    does not silently score zero.
    """
    query_pairs = np.asarray(query_pairs, dtype=np.int64).reshape(-1, 2)
    alignment = np.asarray(alignment, dtype=np.int64).reshape(-1, 2)
    total = len(query_pairs) if n_query_pairs is None else int(n_query_pairs)
    if total == 0:
        return float("nan")
    if len(alignment) == 0:
        return 0.0

    target_set = {(int(a), int(b)) for a, b in
                  np.asarray(target_pairs, dtype=np.int64).reshape(-1, 2)}
    target_set |= {(b, a) for a, b in target_set}

    partner = {}
    for qi, ti in alignment:
        partner[int(qi)] = int(ti)

    recovered = 0
    for i, j in query_pairs:
        ti, tj = partner.get(int(i)), partner.get(int(j))
        if ti is None or tj is None:
            continue
        if (ti, tj) in target_set:
            recovered += 1
    return recovered / total
