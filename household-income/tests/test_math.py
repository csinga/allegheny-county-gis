"""Unit tests for the mathematical building blocks (classification, statistics, income model)."""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aci import classify, model  # noqa: E402
from aci.spatial_stats import gini, morans_i, row_standardise, weighted_quantile  # noqa: E402


# ----------------------------------------------------------------------------- classification
def brute_force_jenks(x: np.ndarray, k: int) -> float:
    """Minimum within-class SSD over every contiguous k-partition of sorted x."""
    x = np.sort(x)
    best = np.inf
    for cuts in itertools.combinations(range(1, x.size), k - 1):
        parts = np.split(x, cuts)
        best = min(best, sum(((p - p.mean()) ** 2).sum() for p in parts))
    return best


def test_fisher_jenks_obvious_clusters():
    x = np.array([1, 2, 3, 10, 11, 12, 20, 21, 22], float)
    assert classify.fisher_jenks(x, 3) == [3.0, 12.0]


@pytest.mark.parametrize("seed", range(5))
def test_fisher_jenks_matches_brute_force(seed):
    x = np.random.default_rng(seed).lognormal(11, 0.5, 11)
    br = classify.fisher_jenks(x, 4)
    c = classify.assign_upper_inclusive(x, br)
    ssd = sum(((x[c == j] - x[c == j].mean()) ** 2).sum() for j in np.unique(c))
    assert ssd == pytest.approx(brute_force_jenks(x, 4), rel=1e-9)


def test_median_relative_breaks_and_assignment():
    br = classify.median_relative_breaks(72_537, (0.5, 0.8, 1.2, 1.6), 1_000)
    assert br == [36_000, 58_000, 87_000, 116_000]
    assert list(classify.assign(np.array([10_000, 36_000, 57_999, 87_000, 500_000]), br)) == [1, 2, 2, 4, 5]


def test_gvf_bounds():
    x = np.array([1.0, 1.0, 5.0, 5.0])
    assert classify.gvf(x, np.array([1, 1, 2, 2])) == pytest.approx(1.0)
    assert classify.gvf(x, np.array([1, 1, 1, 1])) == pytest.approx(0.0)


# ----------------------------------------------------------------------------- statistics
def test_gini_extremes():
    assert gini(np.ones(10)) == pytest.approx(0.0, abs=1e-12)
    x = np.zeros(10)
    x[-1] = 1.0
    assert gini(x) == pytest.approx(0.9)


def test_weighted_quantile_unweighted_median():
    assert weighted_quantile(np.array([1.0, 2.0, 3.0]), np.ones(3), 0.5) == pytest.approx(2.0)


def rook_grid(n: int) -> np.ndarray:
    W = np.zeros((n * n, n * n))
    for r in range(n):
        for c in range(n):
            i = r * n + c
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    W[i, rr * n + cc] = 1
    return row_standardise(W)


def test_morans_i_signs():
    W = rook_grid(6)
    checker = np.array([(r + c) % 2 for r in range(6) for c in range(6)], float)
    gradient = np.array([r + c for r in range(6) for c in range(6)], float)
    assert morans_i(checker, W, 199).I == pytest.approx(-1.0)
    res = morans_i(gradient, W, 199)
    assert res.I > 0.6 and res.p_sim < 0.01


# ----------------------------------------------------------------------------- income model
def test_dagum_median_and_mean():
    a, p, m = np.array([3.5]), 0.3, np.array([70_000.0])
    assert model.dagum_cdf(70_000.0, m, a, p)[0, 0] == pytest.approx(0.5)
    b = model.dagum_scale(m, a, p)[0]
    pdf = lambda x: a[0] * p / x * (x / b) ** (a[0] * p) / ((x / b) ** a[0] + 1) ** (p + 1)  # noqa: E731
    # integrate on a log scale (x = e^u) for numerical stability of the heavy right tail
    mean_numeric = quad(lambda u: np.exp(2 * u) * pdf(np.exp(u)), -10, 25, limit=400)[0]
    assert quad(lambda u: np.exp(u) * pdf(np.exp(u)), -10, 25, limit=400)[0] == pytest.approx(1.0)
    assert model.dagum_mean(m, a, p)[0] == pytest.approx(mean_numeric, rel=1e-4)


def test_mixture_quantile_inverts_cdf():
    med = np.array([30_000.0, 90_000.0])
    a = np.array([3.0, 4.0])
    w = np.array([2.0, 1.0])
    q = model.mixture_quantile(0.5, med, a, 0.6, w)
    assert model.mixture_cdf(q, med, a, 0.6, w)[0] == pytest.approx(0.5, abs=1e-9)


def test_largest_remainder_preserves_total():
    out = model.largest_remainder(1001, np.array([0.2, 0.3, 0.5]))
    assert out.sum() == 1001 and (out >= 0).all()


def test_ridge_reduces_to_ols():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = 2 + X @ np.array([1.0, -0.5, 0.25]) + rng.normal(0, 0.01, 50)
    b0, b = model.ridge(X, y, 0.0)
    ref = np.linalg.lstsq(np.c_[np.ones(50), X], y, rcond=None)[0]
    assert np.allclose([b0, *b], ref)


def test_kriging_exact_without_nugget():
    xy = np.array([[0.0, 0.0], [5_000.0, 0.0], [0.0, 7_000.0]])
    r = np.array([0.3, -0.2, 0.1])
    vg = model.Variogram(nugget=0.0, psill=0.05, range_ft=6_000)
    assert np.allclose(model.krige_residuals(xy, xy, r, vg), r)
    far = model.krige_residuals(np.array([[1e7, 1e7]]), xy, r, vg)
    assert abs(far[0]) < 1e-9  # reverts to the trend far from anchors
