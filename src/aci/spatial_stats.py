"""Spatial weights, spatial autocorrelation (global / local Moran's I) and inequality measures.

Implemented from first principles with NumPy so the mathematics is visible and testable.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np


def queen_weights(gdf: gpd.GeoDataFrame, snap: float = 1.0) -> np.ndarray:
    """Binary queen-contiguity matrix (shared edge or vertex), dense n x n.

    A small buffer (``snap``, in CRS units) absorbs digitising slivers between neighbours.
    """
    geoms = gdf.geometry.buffer(snap).reset_index(drop=True)
    left = gpd.GeoDataFrame(geometry=geoms, crs=gdf.crs)
    pairs = gpd.sjoin(left, left, predicate="intersects")
    i = pairs.index.to_numpy()
    j = pairs["index_right"].to_numpy()
    n = len(gdf)
    W = np.zeros((n, n))
    W[i, j] = 1.0
    np.fill_diagonal(W, 0.0)
    return W


def row_standardise(W: np.ndarray) -> np.ndarray:
    rs = W.sum(axis=1, keepdims=True)
    return np.divide(W, rs, out=np.zeros_like(W), where=rs > 0)


@dataclass
class MoranResult:
    I: float
    expected: float
    z_norm: float
    p_sim: float


def morans_i(y: np.ndarray, W: np.ndarray, permutations: int = 999, seed: int = 0) -> MoranResult:
    """Global Moran's I = (n / S0) * (z' W z) / (z' z), with a conditional-free permutation test."""
    y = np.asarray(y, dtype=float)
    n = y.size
    z = y - y.mean()
    s0 = W.sum()
    denom = z @ z

    def stat(v: np.ndarray) -> float:
        return float(n / s0 * (v @ W @ v) / denom)

    I = stat(z)
    # Analytical moments under normality (Cliff & Ord 1981).
    s1 = 0.5 * ((W + W.T) ** 2).sum()
    s2 = ((W.sum(axis=0) + W.sum(axis=1)) ** 2).sum()
    EI = -1.0 / (n - 1)
    VI = (n**2 * s1 - n * s2 + 3 * s0**2) / ((n**2 - 1) * s0**2) - EI**2
    rng = np.random.default_rng(seed)
    sims = np.array([stat(rng.permutation(z)) for _ in range(permutations)])
    larger = (sims >= I).sum() if I >= EI else (sims <= I).sum()
    return MoranResult(I=I, expected=EI, z_norm=(I - EI) / np.sqrt(VI), p_sim=(larger + 1) / (permutations + 1))


def local_morans(
    y: np.ndarray, W: np.ndarray, permutations: int = 999, alpha: float = 0.05, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anselin (1995) LISA with conditional randomisation.

    Returns (I_i, pseudo p-values, cluster label) where labels are HH, LL, HL, LH or 'ns'.
    W must be row-standardised.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    z = (y - y.mean()) / y.std()
    lag = W @ z
    Ii = z * lag
    rng = np.random.default_rng(seed)
    p = np.ones(n)
    idx = np.arange(n)
    for i in range(n):
        nbrs = np.flatnonzero(W[i])
        k = nbrs.size
        if k == 0:
            continue
        others = np.delete(idx, i)
        # Draw k random "neighbours" from the other n-1 observations, holding z_i fixed.
        draws = others[np.argsort(rng.random((permutations, n - 1)), axis=1)[:, :k]]
        sim = z[i] * (z[draws] * W[i, nbrs]).sum(axis=1)
        larger = (sim >= Ii[i]).sum()
        larger = min(larger, permutations - larger)
        p[i] = (larger + 1) / (permutations + 1)
    quad = np.where(z > 0, np.where(lag > 0, "HH", "HL"), np.where(lag > 0, "LH", "LL"))
    label = np.where(p <= alpha, quad, "ns")
    return Ii, p, label


def gini(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Gini coefficient of a (weighted) sample via the sorted-cumulative-share formula."""
    x = np.asarray(values, dtype=float)
    w = np.ones_like(x) if weights is None else np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    cw = np.cumsum(w)
    cxw = np.cumsum(x * w)
    # Area under the Lorenz curve via trapezoids over population shares.
    pop = np.concatenate([[0.0], cw / cw[-1]])
    inc = np.concatenate([[0.0], cxw / cxw[-1]])
    area = np.sum((pop[1:] - pop[:-1]) * (inc[1:] + inc[:-1]) / 2.0)
    return float(1.0 - 2.0 * area)


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values)[order], np.asarray(weights, dtype=float)[order]
    cw = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(np.interp(q, cw, v))
