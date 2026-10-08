"""Reference layers built from the inputs: county boundary, approximate community areas and map labels."""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import nearest_points

from . import settings as S


def county_boundary(tracts: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    geom = tracts.geometry.buffer(1.0).union_all().buffer(-1.0)
    gdf = gpd.GeoDataFrame(
        {
            "GEOID": [S.STATE_FIPS + S.COUNTY_FIPS],
            "NAME": ["Allegheny County"],
            "STATE": ["Pennsylvania"],
            "area_sqmi": [geom.area / 5280.0**2],
            "source": ["Dissolved from Allegheny County GIS census tracts (2016)"],
        },
        geometry=[geom],
        crs=tracts.crs,
    )
    return gdf


def municipalities(
    tracts: gpd.GeoDataFrame, anchors: gpd.GeoDataFrame, income: pd.DataFrame
) -> gpd.GeoDataFrame:
    """Approximate community-reference polygons (SYNTHETIC).

    No municipal boundary file was supplied and none could be downloaded, so each tract is
    assigned to the municipality of its nearest community reference point, then tracts are
    dissolved by municipality. Boundaries therefore follow tract lines and are approximate;
    the layer is used for labelling and community summaries, not for drawing legal limits.
    """
    pts = tracts.geometry.representative_point()
    a_xy = np.c_[anchors.geometry.x, anchors.geometry.y]
    t_xy = np.c_[pts.x, pts.y]
    nearest = np.argmin(((t_xy[:, None, :] - a_xy[None, :, :]) ** 2).sum(axis=2), axis=1)
    t = tracts[["GEOID", "geometry"]].merge(income[["GEOID", "households", "median_hh_income"]], on="GEOID")
    t["municipality"] = anchors["municipality"].to_numpy()[nearest]
    t["w_inc"] = t["households"] * t["median_hh_income"].fillna(0)

    agg = t.dissolve(
        by="municipality",
        aggfunc={"GEOID": "count", "households": "sum", "w_inc": "sum"},
    ).rename(columns={"GEOID": "n_tracts"})
    agg["hh_weighted_tract_median"] = np.round(agg["w_inc"] / agg["households"].where(agg["households"] > 0), -2)
    agg = agg.drop(columns="w_inc")

    lab = anchors.groupby("municipality").agg(
        label_priority=("label_priority", "max"), lx=("geometry", lambda g: g.x.mean()), ly=("geometry", lambda g: g.y.mean())
    )
    agg = agg.join(lab)
    downtown = gpd.GeoSeries([Point(S.DOWNTOWN_LONLAT)], crs=4326).to_crs(tracts.crs).iloc[0]
    agg.loc["Pittsburgh", ["lx", "ly"]] = [downtown.x + 2500, downtown.y + 1500]
    agg["geometry"] = agg.geometry.buffer(1.0).buffer(-1.0)
    agg["is_synthetic"] = 1
    agg["source"] = "SYNTHETIC approximation: tracts assigned to nearest community reference point"
    return agg.reset_index().rename(columns={"municipality": "name"}).set_crs(tracts.crs, allow_override=True)


def _river_label(water: gpd.GeoDataFrame, name: str, lon: float, lat: float, text: str) -> dict:
    """Snap a label to the named river and orient it along the local channel direction."""
    p = gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(water.crs).iloc[0]
    river = water[water["FULLNAME"] == name].geometry.union_all()
    p = nearest_points(river, p)[0]
    local = river.intersection(p.buffer(4_000))
    coords = np.vstack([np.asarray(g.exterior.coords) for g in getattr(local, "geoms", [local])])
    centred = coords - coords.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    dx, dy = vt[0]
    angle = np.degrees(np.arctan2(dy, dx))
    angle = (angle + 90) % 180 - 90  # keep text upright
    return {"text": text, "kind": "river", "priority": 1, "rotation": -angle, "geometry": p}


def map_labels(water: gpd.GeoDataFrame, munis: gpd.GeoDataFrame, anchors: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    rows = [
        _river_label(water, "Ohio Riv", -80.105, 40.505, "Ohio River"),
        _river_label(water, "Allegheny Riv", -79.835, 40.530, "Allegheny River"),
        _river_label(water, "Monongahela Riv", -79.880, 40.330, "Monongahela River"),
        _river_label(water, "Youghiogheny Riv", -79.835, 40.300, "Youghiogheny R."),
    ]
    for _, m in munis[munis["label_priority"] >= 1].iterrows():
        kind = "city" if m["name"] == "Pittsburgh" else "community"
        rows.append(
            {
                "text": "PITTSBURGH" if kind == "city" else m["name"],
                "kind": kind,
                "priority": int(m["label_priority"]),
                "rotation": 0.0,
                "geometry": Point(m["lx"], m["ly"]),
            }
        )
    hoods = anchors[(anchors["municipality"] == "Pittsburgh") & (anchors["label_priority"] == 1)]
    for _, a in hoods.iterrows():
        rows.append({"text": a["name"], "kind": "neighborhood", "priority": 1, "rotation": 0.0, "geometry": a.geometry})
    return gpd.GeoDataFrame(rows, crs=water.crs)
