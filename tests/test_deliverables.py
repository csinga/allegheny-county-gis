"""Checks on the built deliverables. Run after `make all`; skipped if outputs are missing."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aci import settings as S  # noqa: E402

pytestmark = pytest.mark.skipif(not S.PNG_PATH.exists(), reason="run `make all` first")

REQUIRED_LAYERS = {
    "allegheny_boundary", "income_tracts", "municipalities", "roads", "waterways",
    "pennsylvania_counties", "household_income_brackets", "county_income_distribution",
}


def test_geopackage_layers_and_fields():
    import pyogrio

    names = {row[0] for row in pyogrio.list_layers(S.GPKG_PATH)}
    assert REQUIRED_LAYERS <= names
    t = pyogrio.read_dataframe(S.GPKG_PATH, layer="income_tracts", read_geometry=False)
    est = t[t.has_estimate == 1]
    assert est.median_hh_income.notna().all()
    assert set(est.income_class) == {1, 2, 3, 4, 5}
    assert (t.is_synthetic_income == 1).all()


def test_png_is_300_dpi_tabloid():
    from PIL import Image

    im = Image.open(S.PNG_PATH)
    assert im.size == (5100, 3300)  # 17 x 11 in at 300 DPI
    assert round(im.info["dpi"][0]) == 300


def test_qgz_reopens_with_valid_layers_and_layout():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    qgis_core = pytest.importorskip("qgis.core")
    app = qgis_core.QgsApplication([], False)
    app.initQgis()
    project = qgis_core.QgsProject.instance()
    assert project.read(str(S.QGZ_PATH))
    layers = project.mapLayers().values()
    assert layers and all(lyr.isValid() for lyr in layers)
    layout = project.layoutManager().layoutByName("Household Income Map 17x11")
    assert layout is not None
    ids = {item.id() for item in layout.items() if hasattr(item, "id")}
    assert {"main_map", "locator_inset"} <= ids
