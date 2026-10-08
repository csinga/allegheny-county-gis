"""Tract-level covariates derived from the real building-footprint inventory and hydrography.

These covariates describe the built environment of each tract (how dense, how suburban, how
industrial, how close to the river corridors and the urban core). They drive the trend term of the
synthetic income model in :mod:`aci.model`.
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

from . import settings as S

SQFT_PER_SQKM = 1e6 * S.FT_PER_M**2

# Allegheny County assessment land-use codes for small residential structures -> dwelling units.
UNITS_BY_LUC = {10: 1.0, 20: 2.0, 30: 3.0, 40: 4.0}
APARTMENT_LUC = range(401, 410)
SQFT_PER_APARTMENT_UNIT = 1_000.0
ASSUMED_APARTMENT_FLOORS = 3.0


def estimate_dwelling_units(fp: pd.DataFrame) -> np.ndarray:
    """Heuristic dwelling-unit count per structure.

    * single/two/three/four-family codes map to 1-4 units;
    * apartment codes (401-409) get floor_area / 1,000 sq ft with 3 assumed floors (min 5);
    * any other residential structure counts as max(1, footprint / 1,100 sq ft);
    * non-residential structures hold no households.
    """
    units = np.zeros(len(fp))
    luc = fp["LUC"].to_numpy()
    area = fp["area_sqft"].to_numpy(dtype=float)
    is_res = fp["CLASS"].to_numpy() == "R"
    small = np.isin(luc, list(UNITS_BY_LUC))
    units[small] = pd.Series(luc[small]).map(UNITS_BY_LUC).to_numpy()
    apt = np.isin(luc, list(APARTMENT_LUC))
    units[apt] = np.maximum(5.0, area[apt] * ASSUMED_APARTMENT_FLOORS / SQFT_PER_APARTMENT_UNIT)
    other_res = is_res & ~small & ~apt
    units[other_res] = np.maximum(1.0, np.round(area[other_res] / 1_100.0))
    return units


def building_covariates(tracts: gpd.GeoDataFrame, fp: gpd.GeoDataFrame) -> pd.DataFrame:
    fp = fp.copy()
    fp["units"] = estimate_dwelling_units(fp)
    joined = gpd.sjoin(fp, tracts[["GEOID", "geometry"]], how="inner", predicate="within")
    a = joined["area_sqft"].astype(float)
    joined["is_res"] = joined["CLASS"].eq("R")
    joined["is_sfh"] = joined["LUC"].eq(10)
    joined["ind_area"] = np.where(joined["CLASS"].eq("I"), a, 0.0)
    joined["com_area"] = np.where(joined["CLASS"].eq("C"), a, 0.0)
    joined["sfh_area"] = np.where(joined["is_sfh"], a, np.nan)

    g = joined.groupby("GEOID")
    out = pd.DataFrame(
        {
            "bldg_count": g.size(),
            "res_bldg_count": g["is_res"].sum(),
            "sfh_count": g["is_sfh"].sum(),
            "dwelling_units_est": g["units"].sum(),
            "footprint_sqft": g["area_sqft"].sum(),
            "ind_sqft": g["ind_area"].sum(),
            "com_sqft": g["com_area"].sum(),
            "sfh_median_sqft": g["sfh_area"].median(),
        }
    )
    out = tracts[["GEOID", "ALAND"]].set_index("GEOID").join(out).fillna(
        {c: 0 for c in out.columns if c != "sfh_median_sqft"}
    )
    land_km2 = (out["ALAND"] / 1e6).clip(lower=0.05)
    out["bldg_per_km2"] = out["bldg_count"] / land_km2
    out["sfh_share"] = (out["sfh_count"] / out["res_bldg_count"].where(out["res_bldg_count"] > 0)).fillna(0)
    total = out["footprint_sqft"].where(out["footprint_sqft"] > 0)
    out["industrial_share"] = (out["ind_sqft"] / total).fillna(0)
    out["commercial_share"] = (out["com_sqft"] / total).fillna(0)
    out["coverage"] = (out["footprint_sqft"] / (out["ALAND"] * S.FT_PER_M**2).clip(lower=1)).clip(upper=1)
    # A median of a handful of houses is unstable (e.g. one mansion downtown): require 15.
    out["sfh_median_sqft"] = out["sfh_median_sqft"].where(out["sfh_count"] >= 15)
    out["sfh_median_sqft"] = out["sfh_median_sqft"].fillna(out["sfh_median_sqft"].median())
    return out.drop(columns="ALAND").reset_index()


def location_covariates(tracts: gpd.GeoDataFrame, water: gpd.GeoDataFrame) -> pd.DataFrame:
    centroids = tracts.geometry.representative_point()
    rivers = water[water["FULLNAME"].isin(S.MAIN_RIVERS)].geometry.union_all()
    core = gpd.GeoSeries([Point(S.DOWNTOWN_LONLAT)], crs=4326).to_crs(S.CRS_ANALYSIS).iloc[0]
    return pd.DataFrame(
        {
            "GEOID": tracts["GEOID"].to_numpy(),
            "dist_river_ft": centroids.distance(rivers).to_numpy(),
            "dist_core_ft": centroids.distance(core).to_numpy(),
            "cx": centroids.x.to_numpy(),
            "cy": centroids.y.to_numpy(),
        }
    )


def design_matrix(cov: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Transformed covariates used by the trend model (columns are later standardised)."""
    cols = {
        "log_bldg_density": np.log1p(cov["bldg_per_km2"]),
        "sfh_share": cov["sfh_share"],
        "log_sfh_median_sqft": np.log(cov["sfh_median_sqft"].clip(lower=300)),
        "industrial_share": np.sqrt(cov["industrial_share"]),
        "commercial_share": np.sqrt(cov["commercial_share"]),
        "river_proximity": np.exp(-cov["dist_river_ft"] / 5_000.0),
        "log_dist_core": np.log1p(cov["dist_core_ft"] / 5_280.0),
    }
    X = pd.DataFrame(cols)
    return X.to_numpy(dtype=float), list(X.columns)
