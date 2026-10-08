"""Synthetic, ACS-style household-income estimates for every census tract.

The supplied package has tract geometry but no income attributes, so this module *creates* them
with a transparent statistical generator (documented in docs/METHODOLOGY.md):

1. Trend: ridge regression of log median income on built-environment covariates, fitted at
   tracts that contain a calibration anchor (approximate published community medians).
2. Residual surface: simple kriging of the anchor residuals with an exponential covariance
   (regression kriging); the nugget equals the anchors' measurement-error variance.
3. Spatially autocorrelated noise from a simultaneous autoregressive process.
4. Calibration: one multiplicative constant so the household-weighted county median of the
   tract mixture equals the target county median.
5. Within-tract Dagum (Burr III) household income distributions, whose shape is fitted so the
   county mixture matches a reference bracket distribution -> ACS B19001-style bracket counts,
   means, and ACS-style 90% margins of error.

Every output is flagged synthetic. Results are deterministic for a given seed.
"""
from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.special import gamma

from . import settings as S
from .spatial_stats import row_standardise

# ACS table B19001 household-income brackets (USD, lower bound inclusive).
BRACKETS = [
    (0, 10_000), (10_000, 15_000), (15_000, 20_000), (20_000, 25_000), (25_000, 30_000),
    (30_000, 35_000), (35_000, 40_000), (40_000, 45_000), (45_000, 50_000), (50_000, 60_000),
    (60_000, 75_000), (75_000, 100_000), (100_000, 125_000), (125_000, 150_000),
    (150_000, 200_000), (200_000, np.inf),
]


def bracket_label(lo: float, hi: float) -> str:
    if lo == 0:
        return f"Less than ${hi/1000:.0f}k"
    if np.isinf(hi):
        return f"${lo/1000:.0f}k or more"
    return f"${lo/1000:.0f}k to ${hi/1000 - 0.001:.0f}k".replace(".0k", "k")


@dataclass
class FitReport:
    feature_names: list[str]
    coefficients: np.ndarray
    intercept: float
    n_anchor_tracts: int
    r2_trend: float
    loo_rmse_log: float
    loo_mape: float
    scale_factor: float
    county_median: float
    variogram: Variogram
    dagum_a: float
    dagum_p: float
    dagum_fit_chi2: float


def _standardise(X: np.ndarray, mask: np.ndarray) -> np.ndarray:
    mu = X[mask].mean(axis=0)
    sd = X[mask].std(axis=0)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def ridge(X: np.ndarray, y: np.ndarray, lam: float) -> tuple[float, np.ndarray]:
    """Closed-form ridge with an unpenalised intercept: beta = (Xc'Xc + lam I)^-1 Xc'yc."""
    xm, ym = X.mean(axis=0), y.mean()
    Xc, yc = X - xm, y - ym
    beta = np.linalg.solve(Xc.T @ Xc + lam * np.eye(X.shape[1]), Xc.T @ yc)
    return float(ym - xm @ beta), beta


def empirical_semivariogram(xy: np.ndarray, r: np.ndarray, n_bins: int = 12, max_lag: float | None = None):
    """Matheron estimator: gamma(h) = 1/(2 N(h)) * sum (r_i - r_j)^2 over pairs with |s_i - s_j| in bin h."""
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    g = 0.5 * (r[:, None] - r[None]) ** 2
    iu = np.triu_indices(len(r), 1)
    d, g = d[iu], g[iu]
    max_lag = max_lag or np.quantile(d, 0.5)
    edges = np.linspace(0, max_lag, n_bins + 1)
    k = np.digitize(d, edges) - 1
    ok = (k >= 0) & (k < n_bins)
    lag = np.array([d[ok & (k == b)].mean() if np.any(ok & (k == b)) else np.nan for b in range(n_bins)])
    gam = np.array([g[ok & (k == b)].mean() if np.any(ok & (k == b)) else np.nan for b in range(n_bins)])
    cnt = np.array([np.sum(ok & (k == b)) for b in range(n_bins)])
    good = np.isfinite(gam)
    return lag[good], gam[good], cnt[good]


@dataclass
class Variogram:
    nugget: float
    psill: float
    range_ft: float

    def cov(self, h: np.ndarray) -> np.ndarray:
        """Exponential covariance C(h) = psill * exp(-h / range) (+ nugget at h = 0)."""
        return self.psill * np.exp(-h / self.range_ft) + self.nugget * (h == 0)


def krige_residuals(targets: np.ndarray, anchors: np.ndarray, resid: np.ndarray, vg: Variogram) -> np.ndarray:
    """Simple kriging (known zero mean) of trend residuals: r*(s0) = c0' (C + nugget I)^-1 r.

    Far from anchors the prediction decays to zero, i.e. back to the covariate trend. With a
    nugget the predictor smooths rather than interpolates exactly, which absorbs anchor noise.
    """
    D = np.linalg.norm(anchors[:, None] - anchors[None], axis=2)
    C = vg.cov(D)
    d0 = np.linalg.norm(targets[:, None] - anchors[None], axis=2)
    c0 = vg.psill * np.exp(-d0 / vg.range_ft)
    return c0 @ np.linalg.solve(C, resid)


def sar_noise(W: np.ndarray, rho: float, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Draw eps = (I - rho W)^-1 u with u ~ N(0, sigma^2 I) and W row-standardised."""
    n = W.shape[0]
    u = rng.normal(0.0, sigma, n)
    return np.linalg.solve(np.eye(n) - rho * row_standardise(W), u)


def dagum_scale(med: np.ndarray, a: np.ndarray, p: float) -> np.ndarray:
    """Dagum scale b from the median m: F(m) = 1/2  =>  b = m (2^(1/p) - 1)^(1/a)."""
    return med * (2.0 ** (1.0 / p) - 1.0) ** (1.0 / a)


def dagum_cdf(x: np.ndarray, med: np.ndarray, a: np.ndarray, p: float) -> np.ndarray:
    """Dagum (Burr type III) CDF F(x) = (1 + (x / b)^-a)^-p, evaluated on an (x, tract) grid."""
    x = np.atleast_1d(np.asarray(x, dtype=float))[:, None]
    b = dagum_scale(med, a, p)[None, :]
    return (1.0 + (x / b) ** (-a[None, :])) ** (-p)


def dagum_mean(med: np.ndarray, a: np.ndarray, p: float) -> np.ndarray:
    """E[X] = b * Gamma(p + 1/a) * Gamma(1 - 1/a) / Gamma(p), finite for a > 1."""
    return dagum_scale(med, a, p) * gamma(p + 1.0 / a) * gamma(1.0 - 1.0 / a) / gamma(p)


def mixture_cdf(x: float | np.ndarray, med: np.ndarray, a: np.ndarray, p: float, w: np.ndarray) -> np.ndarray:
    """CDF of the household-weighted mixture of tract Dagum distributions."""
    return (dagum_cdf(x, med, a, p) * w[None, :]).sum(axis=1) / w.sum()


def mixture_quantile(q: float, med: np.ndarray, a: np.ndarray, p: float, w: np.ndarray) -> float:
    lo, hi = np.log(500.0), np.log(10_000_000.0)
    for _ in range(60):  # bisection on log-income; monotone CDF guarantees convergence
        mid = 0.5 * (lo + hi)
        if mixture_cdf(np.exp(mid), med, a, p, w)[0] < q:
            lo = mid
        else:
            hi = mid
    return float(np.exp(0.5 * (lo + hi)))


def mixture_gini(med: np.ndarray, a: np.ndarray, p: float, w: np.ndarray) -> float:
    """Gini of the county mixture: G = 1 - (1/mu) * int_0^inf (1 - F(x))^2 dx (trapezoid on a log grid)."""
    mu = float((w * dagum_mean(med, a, p)).sum() / w.sum())
    x = np.geomspace(50.0, 50_000_000.0, 8_000)
    sf = 1.0 - mixture_cdf(x, med, a, p, w)
    return float(1.0 - np.trapz(sf**2, x) / mu)


def bracket_shares(med: np.ndarray, a: np.ndarray, p: float, w: np.ndarray) -> np.ndarray:
    edges = np.array([lo for lo, _ in BRACKETS[1:]], dtype=float)
    cdf = mixture_cdf(edges, med, a, p, w)
    return np.diff(np.concatenate([[0.0], cdf, [1.0]]))


def fit_dagum_shape(
    med: np.ndarray, w: np.ndarray, target_shares: np.ndarray, target_median: float
) -> tuple[float, float, float]:
    """Grid search for the common Dagum shape (a, p) whose county mixture best reproduces the
    reference bracket shares (chi-square distance), re-centring the median for every candidate."""
    best = (np.inf, None, None)
    t = target_shares / target_shares.sum()
    for a in np.arange(1.8, 6.01, 0.1):
        for pp in np.arange(0.20, 1.61, 0.05):
            av = np.full(med.size, a)
            c = target_median / mixture_quantile(0.5, med, av, pp, w)
            s = bracket_shares(med * c, av, pp, w)
            loss = float(((s - t) ** 2 / t).sum())
            if loss < best[0]:
                best = (loss, float(a), float(pp))
    return best[1], best[2], best[0]


def largest_remainder(total: int, shares: np.ndarray) -> np.ndarray:
    raw = total * shares / shares.sum()
    base = np.floor(raw).astype(int)
    short = total - base.sum()
    if short > 0:
        base[np.argsort(raw - base)[::-1][:short]] += 1
    return base


def generate(
    tracts: gpd.GeoDataFrame,
    cov: pd.DataFrame,
    X: np.ndarray,
    feature_names: list[str],
    anchors: gpd.GeoDataFrame,
    W: np.ndarray,
    p: S.ModelParams = S.MODEL,
) -> tuple[pd.DataFrame, pd.DataFrame, FitReport, pd.DataFrame]:
    rng = np.random.default_rng(p.seed)
    n = len(tracts)
    eligible = cov["dwelling_units_est"].to_numpy() >= p.min_dwelling_units
    Z = _standardise(X, eligible)
    xy = cov[["cx", "cy"]].to_numpy()

    # Anchor tracts: mean log income of the anchors that fall inside each tract.
    hit = gpd.sjoin(anchors, tracts[["GEOID", "geometry"]], predicate="within")
    a = hit.groupby("GEOID")["median_income_usd"].apply(lambda v: np.log(v).mean())
    idx = cov.reset_index().set_index("GEOID").loc[a.index, "index"].to_numpy()
    keep = eligible[idx]
    idx, ya = idx[keep], a.to_numpy()[keep]

    b0, beta = ridge(Z[idx], ya, p.ridge_lambda)
    trend = b0 + Z @ beta
    resid = ya - trend[idx]
    ss_res = (resid**2).sum()
    r2 = 1.0 - ss_res / ((ya - ya.mean()) ** 2).sum()

    # Residual covariance. The nugget is the anchors' own measurement error (ACS-like CV); the
    # partial sill is the remaining residual variance. Anchors are ~1-2 miles apart, too sparse to
    # resolve neighbourhood-scale correlation empirically, so the range is a stated prior.
    vg = Variogram(
        nugget=p.anchor_measurement_var,
        psill=max(float(resid.var()) - p.anchor_measurement_var, 1e-4),
        range_ft=p.residual_range_ft,
    )
    variogram_table = pd.DataFrame(
        dict(zip(("lag_ft", "semivariance", "pairs"), empirical_semivariogram(xy[idx], resid, 10, 40_000.0)))
    )

    # Leave-one-anchor-out validation of the full deterministic predictor (trend + kriged residual).
    loo = np.empty(idx.size)
    for t in range(idx.size):
        m = np.ones(idx.size, bool)
        m[t] = False
        c0, cb = ridge(Z[idx[m]], ya[m], p.ridge_lambda)
        r = ya[m] - (c0 + Z[idx[m]] @ cb)
        loo[t] = c0 + Z[idx[t]] @ cb + krige_residuals(xy[idx[t]][None], xy[idx[m]], r, vg)[0]
    loo_err = loo - ya

    surface = trend + krige_residuals(xy, xy[idx], resid, vg)
    el = np.flatnonzero(eligible)
    eps = np.zeros(n)
    eps[el] = sar_noise(W[np.ix_(el, el)], p.sar_rho, p.sar_sigma, rng)
    log_med = surface + eps

    # Households: proportional to estimated dwelling units, scaled to the county total.
    units = cov["dwelling_units_est"].to_numpy(dtype=float) * eligible
    households = largest_remainder(p.county_households_target, units)
    ref = pd.read_csv(S.CONFIG_DIR / "county_income_distribution_reference.csv", comment="#")
    target = ref["pct_households"].to_numpy(float) / 100.0
    med0 = np.exp(log_med)
    dg_a, dg_p, dist_loss = fit_dagum_shape(med0[el], households[el], target, p.county_median_target)
    shape = dg_a * np.exp(rng.normal(0.0, p.dagum_shape_jitter, n))  # mild tract-to-tract variation

    # Calibrate the level: the mixture median scales linearly with a common factor c.
    raw_median = mixture_quantile(0.5, med0[el], shape[el], dg_p, households[el])
    c = p.county_median_target / raw_median
    med = np.where(eligible, np.round(med0 * c, -1), np.nan)

    # Bracket counts per tract.
    edges = np.array([lo for lo, _ in BRACKETS[1:]], dtype=float)
    rows = []
    bracket_hh = np.zeros((n, len(BRACKETS)), dtype=int)
    for i in el:
        cdf = dagum_cdf(edges, med[i : i + 1], shape[i : i + 1], dg_p)[:, 0]
        shares = np.diff(np.concatenate([[0.0], cdf, [1.0]]))
        bracket_hh[i] = largest_remainder(int(households[i]), shares)
        for b, (lo, hi) in enumerate(BRACKETS):
            rows.append((cov["GEOID"].iat[i], b + 1, bracket_label(lo, hi), lo, None if np.isinf(hi) else hi, int(bracket_hh[i, b])))
    brackets = pd.DataFrame(rows, columns=["GEOID", "bracket_id", "bracket", "lower_usd", "upper_usd", "households"])

    # ACS-style 90% margin of error: coefficient of variation shrinks with sample size.
    cv = 0.06 + 2.2 / np.sqrt(np.maximum(households, 1))
    moe = np.where(eligible, np.round(1.645 * cv * med, -1), np.nan)
    hh_safe = np.maximum(households, 1)
    out = pd.DataFrame(
        {
            "GEOID": cov["GEOID"].to_numpy(),
            "households": np.where(eligible, households, 0),
            "median_hh_income": med,
            "median_hh_income_moe": moe,
            "mean_hh_income": np.where(eligible, np.round(dagum_mean(np.nan_to_num(med, nan=1.0), shape, dg_p), -1), np.nan),
            "dagum_a": np.where(eligible, np.round(shape, 4), np.nan),
            "pct_hh_under_25k": np.where(eligible, 100 * bracket_hh[:, :4].sum(1) / hh_safe, np.nan),
            "pct_hh_150k_plus": np.where(eligible, 100 * bracket_hh[:, 14:].sum(1) / hh_safe, np.nan),
            "has_estimate": eligible.astype(int),
            "model_trend_log": trend,
            "model_noise_log": eps,
        }
    )
    report = FitReport(
        feature_names=feature_names,
        coefficients=beta,
        intercept=b0,
        n_anchor_tracts=int(idx.size),
        r2_trend=float(r2),
        loo_rmse_log=float(np.sqrt((loo_err**2).mean())),
        loo_mape=float(np.mean(np.abs(np.exp(loo_err) - 1.0))),
        scale_factor=float(c),
        variogram=vg,
        county_median=float(mixture_quantile(0.5, med[el], shape[el], dg_p, households[el])),
        dagum_a=dg_a,
        dagum_p=dg_p,
        dagum_fit_chi2=dist_loss,
    )
    return out, brackets, report, variogram_table


def county_distribution(brackets: pd.DataFrame) -> pd.DataFrame:
    g = brackets.groupby(["bracket_id", "bracket", "lower_usd", "upper_usd"], dropna=False)["households"].sum()
    df = g.reset_index().sort_values("bracket_id")
    df["pct_households"] = 100 * df["households"] / df["households"].sum()
    df["cum_pct"] = df["pct_households"].cumsum()
    return df.reset_index(drop=True)
