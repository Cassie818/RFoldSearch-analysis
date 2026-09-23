"""Affine-gap token Smith-Waterman that also returns the correspondence.

src/search/smith_waterman.py returns only the score, but T_align, RNA LDDT
and base-pair agreement are all functions of the aligned pairs, so the
alignment path has to be recovered. The recurrence here is identical to
smith_waterman() -- same three tables, same gap convention where opening a
gap costs gap_open on its first cell -- so for the same inputs this must
return that function's score; tests assert exactly that, on random and on
autocorrelated token strings.

The kernel is numba-compiled when numba is importable, matching how
smith_waterman_numba is used in the deployed pipeline, and falls back to the
same code in pure Python otherwise. Both paths are tested.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except Exception:                                    # pragma: no cover
    njit = None

# how H[i, j] was reached
_STOP, _DIAG, _FROM_E, _FROM_F = 0, 1, 2, 3
# traceback states
_IN_H, _IN_E, _IN_F = 0, 1, 2


def _core(query, target, sub_matrix, gap_open, gap_extend, pairs_out):
    lq, lt = len(query), len(target)
    H = np.zeros((lq + 1, lt + 1), dtype=np.float64)
    E = np.full((lq + 1, lt + 1), -np.inf, dtype=np.float64)
    F = np.full((lq + 1, lt + 1), -np.inf, dtype=np.float64)
    from_h = np.zeros((lq + 1, lt + 1), dtype=np.int8)
    e_opened = np.zeros((lq + 1, lt + 1), dtype=np.bool_)
    f_opened = np.zeros((lq + 1, lt + 1), dtype=np.bool_)

    best = 0.0
    bi, bj = 0, 0
    for i in range(1, lq + 1):
        qi = query[i - 1]
        for j in range(1, lt + 1):
            open_e = H[i, j - 1] + gap_open
            ext_e = E[i, j - 1] + gap_extend
            if open_e >= ext_e:
                E[i, j] = open_e
                e_opened[i, j] = True
            else:
                E[i, j] = ext_e
            open_f = H[i - 1, j] + gap_open
            ext_f = F[i - 1, j] + gap_extend
            if open_f >= ext_f:
                F[i, j] = open_f
                f_opened[i, j] = True
            else:
                F[i, j] = ext_f

            diag = H[i - 1, j - 1] + sub_matrix[qi, target[j - 1]]
            value = 0.0
            source = _STOP
            if diag > value:
                value = diag
                source = _DIAG
            if E[i, j] > value:
                value = E[i, j]
                source = _FROM_E
            if F[i, j] > value:
                value = F[i, j]
                source = _FROM_F
            H[i, j] = value
            from_h[i, j] = source
            if value > best:
                best = value
                bi, bj = i, j

    if best <= 0.0:
        return 0.0, 0

    n = 0
    i, j = bi, bj
    state = _IN_H
    while i > 0 and j > 0:
        if state == _IN_H:
            source = from_h[i, j]
            if source == _STOP:
                break
            if source == _DIAG:
                pairs_out[n, 0] = i - 1
                pairs_out[n, 1] = j - 1
                n += 1
                i -= 1
                j -= 1
            elif source == _FROM_E:
                state = _IN_E
            else:
                state = _IN_F
        elif state == _IN_E:
            if e_opened[i, j]:
                state = _IN_H
                j -= 1
            else:
                j -= 1
        else:
            if f_opened[i, j]:
                state = _IN_H
                i -= 1
            else:
                i -= 1
    return best, n


_core_fast = njit(cache=True)(_core) if njit is not None else None


def smith_waterman_traceback(query, target, sub_matrix,
                             gap_open: float = -2.0,
                             gap_extend: float = -0.5,
                             use_numba: bool = True):
    """Return (score, pairs) for the optimal local alignment.

    ``pairs`` is an (n, 2) int array of aligned (query index, target index) in
    ascending order, containing only match/mismatch columns -- gap columns
    have no correspondence to report.
    """
    query = np.ascontiguousarray(query, dtype=np.int64)
    target = np.ascontiguousarray(target, dtype=np.int64)
    sub_matrix = np.ascontiguousarray(sub_matrix, dtype=np.float64)
    if len(query) == 0 or len(target) == 0:
        return 0.0, np.zeros((0, 2), dtype=np.int64)

    buffer = np.zeros((min(len(query), len(target)), 2), dtype=np.int64)
    kernel = _core_fast if (use_numba and _core_fast is not None) else _core
    score, n = kernel(query, target, sub_matrix, float(gap_open),
                      float(gap_extend), buffer)
    if n == 0:
        return float(score), np.zeros((0, 2), dtype=np.int64)
    return float(score), buffer[:n][::-1].copy()
