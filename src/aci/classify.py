"""Thematic classification: county-median-relative classes, Fisher-Jenks optimal breaks, quantiles,
and the goodness-of-variance-fit (GVF) statistic used to compare them.
"""
from __future__ import annotations

import numpy as np


def fisher_jenks(values: np.ndarray, k: int) -> list[float]:
    """Exact optimal 1-D k-class partition minimising within-class sum of squared deviations.

    Dynamic programme (Fisher 1958): cost[j][m] = min over i of cost[j-1][i-1] + SSD(i..m),
    with SSD computed in O(1) from prefix sums. O(k n^2) time; fine for a few hundred tracts.
    Returns the k-1 interior upper bounds (each break is the maximum of its class).
    """
    x = np.sort(np.asarray(values, dtype=float))
    n = x.size
    if k < 2 or k > n:
        raise ValueError("need 2 <= k <= n")
    s1 = np.concatenate([[0.0], np.cumsum(x)])
    s2 = np.concatenate([[0.0], np.cumsum(x * x)])

    def ssd(i: np.ndarray | int, m: int) -> np.ndarray:
        """SSD of x[i..m] inclusive (0-based), vectorised over start index i."""
        cnt = m - i + 1
        tot = s1[m + 1] - s1[i]
        return (s2[m + 1] - s2[i]) - tot * tot / cnt

    cost = np.full((k, n), np.inf)
    back = np.zeros((k, n), dtype=int)
    cost[0] = [ssd(0, m) for m in range(n)]
    for j in range(1, k):
        for m in range(j, n):
            i = np.arange(j, m + 1)  # first index of the last class
            c = cost[j - 1, i - 1] + ssd(i, m)
            best = int(np.argmin(c))
            cost[j, m] = c[best]
            back[j, m] = i[best]
    breaks: list[float] = []
    m = n - 1
    for j in range(k - 1, 0, -1):
        i = back[j, m]
        breaks.append(float(x[i - 1]))
        m = i - 1
    return sorted(breaks)


def quantile_breaks(values: np.ndarray, k: int) -> list[float]:
    return [float(b) for b in np.quantile(values, np.arange(1, k) / k)]


def median_relative_breaks(median: float, ratios: tuple[float, ...], round_to: int) -> list[float]:
    """Breaks at fixed fractions of the county median (HUD 'area median income' convention)."""
    return [float(round(median * r / round_to) * round_to) for r in ratios]


def assign(values: np.ndarray, breaks: list[float]) -> np.ndarray:
    """1-based class index; class c holds breaks[c-2] <= v < breaks[c-1]."""
    return np.searchsorted(np.asarray(breaks), np.asarray(values), side="right") + 1


def assign_upper_inclusive(values: np.ndarray, breaks: list[float]) -> np.ndarray:
    """1-based class index where each break is the inclusive maximum of its class (Jenks style)."""
    return np.searchsorted(np.asarray(breaks), np.asarray(values), side="left") + 1


def gvf(values: np.ndarray, classes: np.ndarray) -> float:
    """Goodness of variance fit: 1 - SDCM / SDAM (1 = classes perfectly homogeneous)."""
    x = np.asarray(values, dtype=float)
    sdam = ((x - x.mean()) ** 2).sum()
    sdcm = sum(((x[classes == c] - x[classes == c].mean()) ** 2).sum() for c in np.unique(classes))
    return float(1.0 - sdcm / sdam)


def tai(values: np.ndarray, classes: np.ndarray) -> float:
    """Tabular accuracy index (Jenks & Caspall 1971): 1 - SAD within classes / SAD about the mean."""
    x = np.asarray(values, dtype=float)
    sadam = np.abs(x - x.mean()).sum()
    sadcm = sum(np.abs(x[classes == c] - x[classes == c].mean()).sum() for c in np.unique(classes))
    return float(1.0 - sadcm / sadam)
