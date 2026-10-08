"""Project-wide constants: paths, coordinate reference systems and model parameters.

Every tunable number used by the pipeline lives here so that a reviewer can audit the
analysis without reading the implementation modules.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# Debian/Ubuntu's pyproj (shared with PyQGIS) may not find proj.db on its own.
if Path("/usr/share/proj/proj.db").exists():
    os.environ.setdefault("PROJ_DATA", "/usr/share/proj")

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
OUTPUT_DIR = ROOT / "output"

GPKG_PATH = OUTPUT_DIR / "allegheny_household_income.gpkg"
QGZ_PATH = OUTPUT_DIR / "allegheny_household_income.qgz"
PNG_PATH = OUTPUT_DIR / "allegheny_household_income_map_300dpi.png"
PDF_PATH = OUTPUT_DIR / "allegheny_household_income_map.pdf"
CHART_PATH = OUTPUT_DIR / "figures" / "income_summary_chart.svg"

TRACTS_ZIP = RAW_DIR / "alcogis_allegheny-county-census-tracts-2016.zip"
WATER_ZIP = RAW_DIR / "tl_2025_42003_areawater_allegheny_county.zip"
ANCHORS_CSV = CONFIG_DIR / "calibration_anchors.csv"

# The building-footprint archive is ~90 MB, so it is not committed. Point ACI_FOOTPRINTS_ZIP
# at a local copy; the default is the shared project folder this work was produced in.
FOOTPRINTS_ZIP = Path(
    os.environ.get(
        "ACI_FOOTPRINTS_ZIP",
        "/mnt/project-files/alcogisallegheny-county-building-footprint-locations.zip",
    )
)

NATURAL_EARTH_BASE = "https://naturalearth.s3.amazonaws.com"
NE_COUNTIES = "10m_cultural/ne_10m_admin_2_counties"
NE_ROADS = "10m_cultural/ne_10m_roads"

# Layout geometry shared by the chart (stage 1) and the QGIS layout (stage 2), in millimetres.
PAGE_MM = (431.8, 279.4)  # ANSI B / tabloid landscape, 17 x 11 in
CHART_MM = (137.8, 64.0)

# Coordinate reference systems
CRS_ANALYSIS = "EPSG:2272"  # NAD83 / Pennsylvania South (ftUS): county map + all metrics
# Equal-area Albers centred on Pennsylvania for the statewide locator inset (keeps the state upright).
CRS_LOCATOR = "+proj=aea +lat_0=40.9 +lon_0=-77.6 +lat_1=40.0 +lat_2=41.8 +datum=NAD83 +units=m +no_defs"
FT_PER_M = 3.280833333  # US survey foot

STATE_FIPS = "42"
COUNTY_FIPS = "003"

# Pittsburgh's Golden Triangle (Point State Park), used for the distance-to-core covariate.
DOWNTOWN_LONLAT = (-80.0046, 40.4414)
MAIN_RIVERS = ("Allegheny Riv", "Monongahela Riv", "Ohio Riv", "Youghiogheny Riv")


@dataclass(frozen=True)
class ModelParams:
    """Parameters of the synthetic household-income generator (see docs/METHODOLOGY.md)."""

    seed: int = 20261008
    # Approximate ACS 2018-2022 5-year county median household income (USD). The
    # synthetic surface is rescaled so the household-weighted county median equals this.
    county_median_target: float = 72_537.0
    # Approximate number of households in the county, used to scale household counts.
    county_households_target: int = 553_000
    # Ridge penalty for the covariate trend (on standardised covariates).
    ridge_lambda: float = 2.0
    # Regression-kriging residual covariance: nugget = anchor measurement-error variance in log
    # income (CV ~ 11%, typical of ACS place-level medians); range = neighbourhood scale prior.
    anchor_measurement_var: float = 0.012
    residual_range_ft: float = 6_500.0
    # Simultaneous autoregressive (SAR) noise: eps = (I - rho W)^-1 u, u ~ N(0, sigma^2)
    sar_rho: float = 0.55
    sar_sigma: float = 0.07
    # Within-tract household incomes follow a Dagum distribution whose common shape (a, p) is
    # fitted to the reference county bracket shares; a varies by tract with this log-sd.
    dagum_shape_jitter: float = 0.04
    # Tracts with fewer estimated dwelling units than this get no estimate (ACS-style suppression).
    min_dwelling_units: int = 60
    # Monte-Carlo permutations for Moran's I / LISA significance.
    permutations: int = 999
    lisa_alpha: float = 0.05


@dataclass(frozen=True)
class ClassScheme:
    """Income classes defined relative to the county median (HUD area-median-income style)."""

    ratios: tuple[float, ...] = (0.50, 0.80, 1.20, 1.60)
    round_to: int = 1_000
    names: tuple[str, ...] = field(
        default=(
            "Low income",
            "Lower-middle income",
            "Middle income",
            "Upper-middle income",
            "High income",
        )
    )
    # ColorBrewer PuOr (5-class, colour-blind safe), orange = lower, purple = higher.
    colors: tuple[str, ...] = ("#d8641f", "#f6b86b", "#f3efe4", "#ada3cf", "#54308c")


MODEL = ModelParams()
CLASSES = ClassScheme()
