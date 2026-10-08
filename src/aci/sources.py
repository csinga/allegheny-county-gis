"""Readers for every raw input. All functions return GeoDataFrames in CRS_ANALYSIS unless noted."""
from __future__ import annotations

import logging
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

from . import settings as S

log = logging.getLogger(__name__)


def _vsizip(zip_path: Path, member_suffix: str = ".shp") -> str:
    """GDAL virtual path to the first shapefile inside a zip archive (no extraction needed)."""
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(member_suffix)]
    if not members:
        raise FileNotFoundError(f"No {member_suffix} inside {zip_path}")
    return f"/vsizip/{zip_path}/{members[0]}"


def load_tracts() -> gpd.GeoDataFrame:
    tracts = gpd.read_file(_vsizip(S.TRACTS_ZIP))
    tracts = tracts[(tracts.STATEFP == S.STATE_FIPS) & (tracts.COUNTYFP == S.COUNTY_FIPS)]
    tracts = tracts[["GEOID", "TRACTCE", "NAME", "ALAND", "AWATER", "geometry"]].copy()
    tracts["ALAND"] = tracts["ALAND"].astype(float)
    tracts["AWATER"] = tracts["AWATER"].astype(float)
    tracts = tracts.to_crs(S.CRS_ANALYSIS)
    tracts["geometry"] = tracts.geometry.make_valid()
    return tracts.sort_values("GEOID").reset_index(drop=True)


def load_water() -> gpd.GeoDataFrame:
    water = gpd.read_file(_vsizip(S.WATER_ZIP)).to_crs(S.CRS_ANALYSIS)
    water["geometry"] = water.geometry.make_valid()
    return water


def load_footprints() -> gpd.GeoDataFrame:
    """Building footprints reduced to representative points plus the attributes the model uses.

    ShapeSTAre is the footprint area in square feet (PA South), CLASS is the assessment class
    (R residential, C commercial, I industrial, ...), LUC the county land-use code (10 = single family).
    """
    if not S.FOOTPRINTS_ZIP.exists():
        raise FileNotFoundError(
            f"Building footprints not found at {S.FOOTPRINTS_ZIP}; set ACI_FOOTPRINTS_ZIP."
        )
    cols = ["status", "CLASS", "LUC", "ShapeSTAre"]
    fp = gpd.read_file(_vsizip(S.FOOTPRINTS_ZIP), columns=cols, engine="pyogrio")
    fp = fp.to_crs(S.CRS_ANALYSIS)
    fp["geometry"] = fp.geometry.representative_point()
    fp["CLASS"] = fp["CLASS"].fillna("X")
    fp["LUC"] = pd.to_numeric(fp["LUC"], errors="coerce").fillna(-1).astype(int)
    fp = fp.rename(columns={"ShapeSTAre": "area_sqft"})
    return fp


def _natural_earth(name: str) -> Path:
    """Download (once) and return the local path to a Natural Earth shapefile archive."""
    S.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local = S.CACHE_DIR / (Path(name).name + ".zip")
    if not local.exists():
        url = f"{S.NATURAL_EARTH_BASE}/{name}.zip"
        log.info("Downloading %s", url)
        urllib.request.urlretrieve(url, local)  # noqa: S310 - fixed public https URL
    return local


def load_pa_counties() -> gpd.GeoDataFrame:
    counties = gpd.read_file(_vsizip(_natural_earth(S.NE_COUNTIES)))
    pa = counties[counties["REGION"] == "PA"].copy()
    pa["GEOID"] = pa["CODE_LOCAL"].astype(str).str.zfill(5)
    pa = pa[["GEOID", "NAME", "geometry"]]
    pa["is_study_area"] = (pa["GEOID"] == S.STATE_FIPS + S.COUNTY_FIPS).astype(int)
    pa = pa.to_crs(S.CRS_ANALYSIS)
    pa["geometry"] = pa.geometry.make_valid()
    return pa.sort_values("GEOID").reset_index(drop=True)


ROUTE_PREFIX = {
    **{n: "I" for n in ("70", "76", "79", "279", "376", "579")},
    **{n: "US" for n in ("19", "22", "30")},
    **{n: "PA" for n in ("8", "18", "28", "51", "65", "60")},
}


def load_roads(clip_to: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Major highways (Natural Earth 1:10m) clipped to the study area with route designations."""
    bbox = tuple(clip_to.to_crs(4326).total_bounds)
    roads = gpd.read_file(_vsizip(_natural_earth(S.NE_ROADS)), bbox=bbox).to_crs(S.CRS_ANALYSIS)
    roads = gpd.clip(roads, clip_to.geometry.union_all())
    roads = roads[~roads.geometry.is_empty].copy()
    roads["route_num"] = roads["name"].astype(str).str.strip()
    roads["route_sys"] = roads["route_num"].map(ROUTE_PREFIX).fillna("PA")
    roads["route"] = roads["route_sys"] + "-" + roads["route_num"]
    roads["road_class"] = roads["route_sys"].map(
        {"I": "Interstate", "US": "US highway", "PA": "State route"}
    )
    roads["toll"] = roads["toll"].fillna(0).astype(int)
    roads = roads.explode(index_parts=False)
    roads = roads[roads.geom_type == "LineString"]
    roads = roads.dissolve(by=["route", "route_sys", "road_class"], as_index=False, aggfunc={"toll": "max"})
    return roads[["route", "route_sys", "road_class", "toll", "geometry"]].reset_index(drop=True)


def load_anchors() -> gpd.GeoDataFrame:
    df = pd.read_csv(S.ANCHORS_CSV, comment="#")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    return gdf.to_crs(S.CRS_ANALYSIS)
