"""Karlin-Altschul E-values for RFoldseek token Smith-Waterman scores.

RFoldseek reranks candidates by a raw local Smith-Waterman score under the
Structure-BLOSUM matrix. A raw score is not, on its own, interpretable: the user
cannot tell whether SW=30 is significant. As in BLAST/Foldseek we convert it to
an E-value, the expected number of database hits scoring at least this high by
chance:

    E = K * m * N * exp(-lambda * S)

where ``m`` is the query length, ``N`` is the total number of tokens in the
target database, and ``lambda``/``K`` are the Gumbel parameters of the null
score distribution. Gapped local alignment has no closed form for
``lambda``/``K``; we calibrate them once by scoring random token-sequence pairs
drawn from the database's background token frequencies, with the *same* matrix
and gap penalties as the search, and fitting a Gumbel distribution to the
optimal local scores. Calibration is done by build_evalue_params.py and cached
to files/models/evalue_params.json; the RFoldseek CLI only reads that file.
"""

from __future__ import annotations

import numpy as np

from smith_waterman import smith_waterman_numba

EULER_GAMMA = 0.5772156649015329


def background_frequencies(token_lists, n_codes: int) -> np.ndarray:
    """Empirical token frequencies over the whole database (Laplace-smoothed)."""
    counts = np.ones(n_codes, dtype=np.float64)  # +1 smoothing avoids p == 0
    for t in token_lists:
        if len(t):
            counts += np.bincount(np.asarray(t, dtype=np.int64), minlength=n_codes)[:n_codes]
    return counts / counts.sum()


def calibrate(
    sub_matrix: np.ndarray,
    background: np.ndarray,
    gap_open: float = -2.0,
    gap_extend: float = -0.5,
    length: int = 300,
    n_pairs: int = 4000,
    seed: int = 0,
) -> dict:
    """Fit Karlin-Altschul lambda/K from random token-pair SW scores.

    Random sequences of ``length`` tokens are drawn i.i.d. from ``background``
    and aligned with the deployed matrix / gap penalties. The optimal local
    score of independent sequences is asymptotically Gumbel-distributed:

        E[S]  = (ln(K m n) + gamma) / lambda,
        Var[S] = pi^2 / (6 lambda^2).

    Method-of-moments estimator (Altschul & Gish, 1996):
        lambda = pi / (std(S) * sqrt(6)),
        K      = exp(lambda * mean(S) - gamma) / (m * n).
    """
    n_codes = sub_matrix.shape[0]
    rng = np.random.default_rng(seed)
    sub = np.asarray(sub_matrix, dtype=np.float32)
    p = np.asarray(background, dtype=np.float64)
    p = p / p.sum()

    scores = np.empty(n_pairs, dtype=np.float64)
    for i in range(n_pairs):
        a = rng.choice(n_codes, size=length, p=p).astype(np.int64)
        b = rng.choice(n_codes, size=length, p=p).astype(np.int64)
        scores[i] = smith_waterman_numba(a, b, sub, gap_open, gap_extend)

    mean = float(scores.mean())
    std = float(scores.std(ddof=1))
    if std <= 0:
        raise ValueError("degenerate null score distribution (std=0); check matrix/gaps")
    lam = float(np.pi / (std * np.sqrt(6.0)))
    mn = float(length) * float(length)
    k_param = float(np.exp(lam * mean - EULER_GAMMA) / mn)
    return {
        "lambda": lam,
        "K": k_param,
        "gap_open": float(gap_open),
        "gap_extend": float(gap_extend),
        "calib_length": int(length),
        "n_pairs": int(n_pairs),
        "seed": int(seed),
        "null_score_mean": mean,
        "null_score_std": std,
        "null_score_min": float(scores.min()),
        "note": (
            "Karlin-Altschul Gumbel parameters (method-of-moments) for the token "
            "Smith-Waterman score under Structure-BLOSUM. Database E-value for a "
            "hit scoring S is E = K * m * N * exp(-lambda * S), with m the query "
            "length and N the total token count of the target database."
        ),
    }


def evalue(score, m, n, lam: float, k_param: float):
    """Karlin-Altschul E-value E = K*m*n*exp(-lambda*S).

    For a database search, pass ``n`` = total tokens in the target database to
    obtain the database E-value (expected number of chance hits this good).
    ``score`` may be a scalar or an array.
    """
    s = np.asarray(score, dtype=np.float64)
    return k_param * float(m) * float(n) * np.exp(-lam * s)


def neg_log10_evalue(score, m, n, lam: float, k_param: float):
    """-log10(E); larger = more significant. Numerically safe for tiny E."""
    s = np.asarray(score, dtype=np.float64)
    return (lam * s) / np.log(10.0) - np.log10(max(k_param * float(m) * float(n), 1e-300))


if __name__ == "__main__":
    # Sanity: random pairs -> E ~ O(m*n) (not significant); identical -> E << 1.
    rng = np.random.default_rng(0)
    n_codes = 80
    sub = rng.standard_normal((n_codes, n_codes)).astype(np.float32) * 0.4
    sub = (sub + sub.T) / 2
    np.fill_diagonal(sub, 1.0)
    bg = np.ones(n_codes) / n_codes
    par = calibrate(sub, bg, length=200, n_pairs=1500, seed=1)
    lam, kp = par["lambda"], par["K"]
    print("lambda=%.4f K=%.4g null mean=%.2f std=%.2f" %
          (lam, kp, par["null_score_mean"], par["null_score_std"]))
    a = rng.choice(n_codes, size=200, p=bg).astype(np.int64)
    b = rng.choice(n_codes, size=200, p=bg).astype(np.int64)
    N = 200 * 500  # pretend a 500-chain database of length-200 chains
    print("random  : S=%.1f  E=%.3g" % (smith_waterman_numba(a, b, sub, -2, -0.5),
          evalue(smith_waterman_numba(a, b, sub, -2, -0.5), 200, N, lam, kp)))
    print("identical: S=%.1f  E=%.3g" % (smith_waterman_numba(a, a, sub, -2, -0.5),
          evalue(smith_waterman_numba(a, a, sub, -2, -0.5), 200, N, lam, kp)))
