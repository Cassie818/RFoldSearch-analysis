"""The traceback aligner must agree with the repository's scorer exactly."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "search"))
from token_alignment import smith_waterman_traceback

# the repository's own score-only implementation
_REPO_SW = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "search")
if os.path.isdir(_REPO_SW):
    sys.path.insert(0, _REPO_SW)
try:
    from smith_waterman import smith_waterman as repo_smith_waterman
except ImportError:                                   # pragma: no cover
    repo_smith_waterman = None

GO, GE = -2.0, -0.5
K = 80


def random_matrix(seed, k=K):
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((k, k)) * 1.2
    m = (m + m.T) / 2
    np.fill_diagonal(m, 2.0)
    return m


@pytest.mark.skipif(repo_smith_waterman is None,
                    reason="repository smith_waterman not importable")
@pytest.mark.parametrize("seed", range(12))
def test_score_matches_the_repository_scorer(seed):
    rng = np.random.default_rng(seed)
    sub = random_matrix(seed)
    q = rng.integers(0, K, rng.integers(5, 60))
    t = rng.integers(0, K, rng.integers(5, 60))
    score, _ = smith_waterman_traceback(q, t, sub, GO, GE)
    assert score == pytest.approx(repo_smith_waterman(q, t, sub, GO, GE), abs=1e-4)


@pytest.mark.skipif(repo_smith_waterman is None,
                    reason="repository smith_waterman not importable")
@pytest.mark.parametrize("seed", range(6))
def test_score_matches_on_autocorrelated_tokens(seed):
    """Real token strings repeat; random i.i.d. tokens are an easy case."""
    rng = np.random.default_rng(100 + seed)
    sub = random_matrix(seed)

    def runs(n):
        out = []
        while len(out) < n:
            out += [int(rng.integers(0, K))] * int(rng.integers(1, 6))
        return np.asarray(out[:n], dtype=np.int64)

    q, t = runs(int(rng.integers(20, 80))), runs(int(rng.integers(20, 80)))
    score, _ = smith_waterman_traceback(q, t, sub, GO, GE)
    assert score == pytest.approx(repo_smith_waterman(q, t, sub, GO, GE), abs=1e-4)


# ------------------------------------------------------- traceback integrity

def test_self_alignment_recovers_the_identity_correspondence():
    sub = random_matrix(1)
    q = np.array([3, 17, 42, 5, 5, 61, 12, 8], dtype=np.int64)
    score, pairs = smith_waterman_traceback(q, q, sub, GO, GE)
    assert pairs[:, 0].tolist() == pairs[:, 1].tolist() == list(range(len(q)))
    assert score == pytest.approx(sum(sub[x, x] for x in q))


def test_pairs_are_strictly_increasing_in_both_sequences():
    rng = np.random.default_rng(7)
    sub = random_matrix(2)
    for _ in range(20):
        q = rng.integers(0, K, rng.integers(10, 50))
        t = rng.integers(0, K, rng.integers(10, 50))
        _, pairs = smith_waterman_traceback(q, t, sub, GO, GE)
        if len(pairs) < 2:
            continue
        assert np.all(np.diff(pairs[:, 0]) > 0)
        assert np.all(np.diff(pairs[:, 1]) > 0)


def test_pair_scores_reconstruct_the_alignment_score():
    """Score = sum of substitution scores minus the gap cost implied by the
    gaps between consecutive aligned pairs. Checks the path is consistent
    with the score rather than merely plausible."""
    rng = np.random.default_rng(11)
    sub = random_matrix(3)
    for _ in range(20):
        q = rng.integers(0, K, rng.integers(15, 60))
        t = rng.integers(0, K, rng.integers(15, 60))
        score, pairs = smith_waterman_traceback(q, t, sub, GO, GE)
        if len(pairs) < 2:
            continue
        total = sum(sub[q[i], t[j]] for i, j in pairs)
        for (i0, j0), (i1, j1) in zip(pairs[:-1], pairs[1:]):
            gi, gj = i1 - i0 - 1, j1 - j0 - 1
            for g in (gi, gj):
                if g > 0:
                    total += GO + GE * (g - 1)
        assert total == pytest.approx(score, abs=1e-6)


def test_empty_and_no_alignment():
    sub = np.full((K, K), -5.0)
    np.fill_diagonal(sub, -5.0)                 # nothing scores positive
    q = np.array([1, 2, 3], dtype=np.int64)
    t = np.array([4, 5, 6], dtype=np.int64)
    score, pairs = smith_waterman_traceback(q, t, sub, GO, GE)
    assert score == 0.0 and len(pairs) == 0
    score, pairs = smith_waterman_traceback(np.zeros(0, np.int64), t, sub)
    assert score == 0.0 and len(pairs) == 0


# ------------------------------------------- compiled and interpreted agree

@pytest.mark.parametrize("seed", range(8))
def test_numba_and_python_kernels_agree(seed):
    import token_alignment as ta
    if ta._core_fast is None:
        pytest.skip("numba not installed")
    rng = np.random.default_rng(500 + seed)
    sub = random_matrix(seed)
    q = rng.integers(0, K, rng.integers(10, 70))
    t = rng.integers(0, K, rng.integers(10, 70))
    s1, p1 = smith_waterman_traceback(q, t, sub, GO, GE, use_numba=True)
    s2, p2 = smith_waterman_traceback(q, t, sub, GO, GE, use_numba=False)
    assert s1 == pytest.approx(s2, abs=1e-9)
    assert p1.tolist() == p2.tolist()
