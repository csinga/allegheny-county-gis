#!/usr/bin/env python3
"""Stage 1: process raw inputs into the analysis GeoPackage and the layout chart.

Usage:  python scripts/01_build_geopackage.py
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyogrio

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from aci import chart, classify, features, model, reference, sources  # noqa: E402
from aci import settings as S  # noqa: E402
from aci.spatial_stats import gini, local_morans, morans_i, queen_weights, row_standardise, weighted_quantile  # noqa: E402

log = logging.getLogger("build")


def write_layer(df, name: str) -> None:
    pyogrio.write_dataframe(df, S.GPKG_PATH, layer=name, driver="GPKG", append=False)
    log.info("  wrote %-28s %6d rows", name, len(df))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    S.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if S.GPKG_PATH.exists():
        S.GPKG_PATH.unlink()

    log.info("Loading inputs")
    tracts = sources.load_tracts()
    water = sources.load_water()
    anchors = sources.load_anchors()
    boundary = reference.county_boundary(tracts)
    pa = sources.load_pa_counties()
    roads = sources.load_roads(boundary)
    fp = sources.load_footprints()
    log.info("  %d tracts, %d water features, %d footprints, %d anchors", len(tracts), len(water), len(fp), len(anchors))

    log.info("Deriving tract covariates")
    cov = features.building_covariates(tracts, fp).merge(features.location_covariates(tracts, water), on="GEOID")
    cov = tracts[["GEOID"]].merge(cov, on="GEOID", how="left")
    X, names = features.design_matrix(cov)

    log.info("Generating synthetic income estimates")
    W = queen_weights(tracts)
    income, brackets, fit, variogram = model.generate(tracts, cov, X, names, anchors, W)
    log.info(
        f"  trend R2={fit.r2_trend:.3f}  LOO RMSE(log)={fit.loo_rmse_log:.3f}  "
        f"LOO MAPE={100 * fit.loo_mape:.1f}%  county median=${fit.county_median:,.0f}"
    )

    est = income["has_estimate"].to_numpy() == 1
    med = income["median_hh_income"].to_numpy()
    county_median = fit.county_median
    hh = income["households"].to_numpy()

    log.info("Classifying")
    cs = S.CLASSES
    breaks = classify.median_relative_breaks(county_median, cs.ratios, cs.round_to)
    cls = np.zeros(len(income), dtype=int)
    cls[est] = classify.assign(med[est], breaks)
    edges = [None, *breaks, None]

    def label(c: int) -> str:
        lo, hi = edges[c - 1], edges[c]
        if lo is None:
            return f"Under ${hi:,.0f}"
        if hi is None:
            return f"${lo:,.0f} or more"
        return f"${lo:,.0f} to ${hi - 1:,.0f}"

    income["income_class"] = cls
    income["class_name"] = [cs.names[c - 1] if c else "No estimate" for c in cls]
    income["class_label"] = [label(c) if c else "Too few households for an estimate" for c in cls]
    income["ratio_to_county_median"] = np.round(med / county_median, 3)

    comparison = []
    v = med[est]
    for method, br, assign in (
        ("Median-relative (50/80/120/160%) - used on map", breaks, classify.assign),
        ("Fisher-Jenks natural breaks", classify.fisher_jenks(v, 5), classify.assign_upper_inclusive),
        ("Quantile (quintiles)", classify.quantile_breaks(v, 5), classify.assign),
    ):
        c = assign(v, br)
        counts = np.bincount(c, minlength=6)[1:]
        comparison.append(
            {
                "method": method,
                "breaks_usd": ", ".join(f"{b:,.0f}" for b in br),
                "gvf": round(classify.gvf(v, c), 4),
                "tai": round(classify.tai(v, c), 4),
                "tracts_per_class": ", ".join(map(str, counts)),
            }
        )
    comparison = pd.DataFrame(comparison)

    log.info("Spatial autocorrelation")
    el = np.flatnonzero(est)
    Wel = row_standardise(W[np.ix_(el, el)])
    logv = np.log(v)
    mi = morans_i(logv, Wel, S.MODEL.permutations, S.MODEL.seed)
    Ii, p_lisa, lab = local_morans(logv, Wel, S.MODEL.permutations, S.MODEL.lisa_alpha, S.MODEL.seed)
    income["lisa_I"] = np.nan
    income["lisa_p"] = np.nan
    income["lisa_cluster"] = "n/a"
    income.loc[el, "lisa_I"] = np.round(Ii, 4)
    income.loc[el, "lisa_p"] = p_lisa
    income.loc[el, "lisa_cluster"] = lab

    tr = tracts.merge(income, on="GEOID").merge(
        cov.drop(columns=["cx", "cy"]).round(4), on="GEOID"
    )
    tr["land_sqmi"] = tr["ALAND"] / 2_589_988.11
    tr["is_synthetic_income"] = 1
    tr = tr.set_geometry("geometry")

    # Class summary table
    rows = []
    for c in range(1, 6):
        m = cls == c
        rows.append(
            {
                "income_class": c,
                "class_name": cs.names[c - 1],
                "class_label": label(c),
                "color_hex": cs.colors[c - 1],
                "tracts": int(m.sum()),
                "households": int(hh[m].sum()),
                "pct_households": round(100 * hh[m].sum() / hh[est].sum(), 2),
                "land_sqmi": round(float(tr.loc[m, "land_sqmi"].sum()), 2),
                "min_tract_median": float(np.nanmin(med[m])) if m.any() else None,
                "max_tract_median": float(np.nanmax(med[m])) if m.any() else None,
            }
        )
    class_summary = pd.DataFrame(rows)

    county_dist = model.county_distribution(brackets)
    munis = reference.municipalities(tracts, anchors, income)
    labels = reference.map_labels(water, munis, anchors)

    q1 = weighted_quantile(v, hh[est], 0.25)
    q3 = weighted_quantile(v, hh[est], 0.75)
    lisa_counts = pd.Series(lab).value_counts().to_dict()
    shape_a = income["dagum_a"].to_numpy()
    stats = {
        "county_median_hh_income": round(county_median),
        "county_households": int(hh.sum()),
        "tracts_total": int(len(tr)),
        "tracts_with_estimate": int(est.sum()),
        "tract_median_min": float(v.min()),
        "tract_median_max": float(v.max()),
        "tract_median_of_medians": float(np.median(v)),
        "tract_median_cv": float(v.std() / v.mean()),
        "hh_weighted_tract_q1": round(q1),
        "hh_weighted_tract_q3": round(q3),
        "gini_household_mixture": round(model.mixture_gini(med[est], shape_a[est], fit.dagum_p, hh[est]), 4),
        "gini_between_tracts": round(gini(v, hh[est]), 4),
        "pct_hh_under_25k": round(float(county_dist.loc[county_dist.bracket_id <= 4, "pct_households"].sum()), 2),
        "pct_hh_150k_plus": round(float(county_dist.loc[county_dist.bracket_id >= 15, "pct_households"].sum()), 2),
        "morans_I": round(mi.I, 4),
        "morans_E_I": round(mi.expected, 4),
        "morans_z": round(mi.z_norm, 2),
        "morans_p_sim": mi.p_sim,
        "lisa_counts": lisa_counts,
        "class_breaks_usd": breaks,
        "class_ratios": list(cs.ratios),
        "model_trend_r2": round(fit.r2_trend, 4),
        "model_loo_rmse_log": round(fit.loo_rmse_log, 4),
        "model_loo_mape": round(fit.loo_mape, 4),
        "model_anchor_tracts": fit.n_anchor_tracts,
        "model_scale_factor": round(fit.scale_factor, 4),
        "dagum_a": round(fit.dagum_a, 3),
        "dagum_p": round(fit.dagum_p, 3),
        "dagum_fit_chi2": round(fit.dagum_fit_chi2, 5),
        "variogram_nugget": round(fit.variogram.nugget, 5),
        "variogram_partial_sill": round(fit.variogram.psill, 5),
        "variogram_range_ft": round(fit.variogram.range_ft),
        "model_seed": S.MODEL.seed,
        "built": date.today().isoformat(),
    }

    coef = pd.DataFrame(
        {"term": ["(intercept)", *fit.feature_names], "coefficient_log_income_per_sd": [fit.intercept, *fit.coefficients]}
    )
    stats_table = pd.DataFrame(
        [(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v)) for k, v in stats.items()],
        columns=["statistic", "value"],
    )
    data_sources = pd.DataFrame(
        [
            ("income_tracts (geometry)", "Allegheny County GIS - Census Tracts 2016 (TIGER-derived)", "provided", 0),
            ("income_tracts (income attributes)", "Modelled by this pipeline (see docs/METHODOLOGY.md)", "generated", 1),
            ("household_income_brackets", "Modelled log-normal tract distributions, ACS B19001 brackets", "generated", 1),
            ("waterways", "U.S. Census Bureau TIGER/Line 2025 Area Water, Allegheny County", "provided", 0),
            ("allegheny_boundary", "Dissolved from the provided census tracts", "derived", 0),
            ("roads", "Natural Earth 1:10m Roads v5 (public domain)", "downloaded", 0),
            ("pennsylvania_counties", "Natural Earth 1:10m Admin-2 Counties v5 (public domain)", "downloaded", 0),
            ("municipalities", "Approximate: tracts grouped by nearest community reference point", "generated", 1),
            ("building covariates", "Allegheny County GIS - Building Footprint Locations", "provided", 0),
        ],
        columns=["layer", "source", "provenance", "is_synthetic"],
    )

    log.info("Writing %s", S.GPKG_PATH)
    keep_tr = [
        "GEOID", "TRACTCE", "NAME", "land_sqmi", "households", "median_hh_income", "median_hh_income_moe",
        "mean_hh_income", "pct_hh_under_25k", "pct_hh_150k_plus", "ratio_to_county_median", "income_class",
        "class_name", "class_label", "has_estimate", "lisa_I", "lisa_p", "lisa_cluster", "bldg_count",
        "res_bldg_count", "dwelling_units_est", "bldg_per_km2", "sfh_share", "sfh_median_sqft",
        "industrial_share", "commercial_share", "dist_river_ft", "dist_core_ft", "dagum_a",
        "model_trend_log", "model_noise_log", "is_synthetic_income", "geometry",
    ]
    tr = tr[keep_tr]
    for c in ("pct_hh_under_25k", "pct_hh_150k_plus", "land_sqmi", "bldg_per_km2"):
        tr[c] = tr[c].round(2)
    write_layer(boundary, "allegheny_boundary")
    write_layer(tr, "income_tracts")
    write_layer(munis, "municipalities")
    write_layer(roads, "roads")
    write_layer(water[["FULLNAME", "MTFCC", "HYDROID", "AWATER", "geometry"]].rename(columns={"FULLNAME": "name"}), "waterways")
    write_layer(pa, "pennsylvania_counties")
    write_layer(labels, "map_labels")
    write_layer(anchors.drop(columns=["lat", "lon"]), "calibration_anchors")
    write_layer(brackets, "household_income_brackets")
    write_layer(county_dist, "county_income_distribution")
    write_layer(class_summary, "income_class_summary")
    write_layer(comparison, "classification_comparison")
    write_layer(coef, "model_coefficients")
    write_layer(variogram, "residual_semivariogram")
    write_layer(stats_table, "summary_statistics")
    write_layer(data_sources, "data_sources")

    (S.OUTPUT_DIR / "summary_statistics.json").write_text(json.dumps(stats, indent=2))
    chart.summary_chart(county_dist, med, breaks, county_median, S.CHART_PATH, *S.CHART_MM)
    log.info(f"Done. County median ${county_median:,.0f}; Moran's I {mi.I:.3f} (p={mi.p_sim:.3f})")


if __name__ == "__main__":
    main()
