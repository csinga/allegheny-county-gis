# Methodology: the applied mathematics behind the map

This document explains every quantitative step, in the order the pipeline runs it. Numbers quoted
are from the committed build (seed `20261008`); `output/summary_statistics.json` holds the full set.

> **Important.** The supplied package contains tract *geometry* but no household-income
> attributes, and the network policy blocked the Census API. The income values are therefore
> **synthetic**: produced by the statistical generator below, calibrated to approximate published
> community medians. They are realistic in pattern and scale, flagged `is_synthetic_income = 1`,
> and must not be cited as official ACS estimates. Swapping in real ACS table B19013/B19001 values
> only requires replacing `model.generate()`; everything downstream (classification, statistics,
> layout) is data-agnostic.

---

## 1. Study area and reference frame

* **Projection.** All metrics use NAD83 / Pennsylvania South (US survey feet, EPSG:2272), the
  state-plane zone that contains Allegheny County. Distances and areas are therefore conformal and
  near-true at county scale. The locator inset uses an Albers equal-area projection centred on
  Pennsylvania (standard parallels 40.0°N and 41.8°N, central meridian 77.6°W) so the state is
  upright and the county's areal share is honest.
* **County boundary.** The boundary is the topological union of the 402 tract polygons, with a
  ±1 ft buffer/unbuffer to close digitising slivers: `B = buffer(buffer(∪ T_i, +1), −1)`.

## 2. Built-environment covariates (real data)

The 573,908 building footprints are reduced to representative points and joined to tracts
(point-in-polygon). For tract *i*:

| covariate | definition | rationale |
|---|---|---|
| `log_bldg_density` | log(1 + buildings / km² land) | urban intensity |
| `sfh_share` | single-family (LUC 10) / residential structures | owner-occupied suburban form |
| `log_sfh_median_sqft` | log median single-family footprint (sq ft; needs ≥ 15 houses) | house size is the strongest observable correlate of income |
| `industrial_share` | √(industrial footprint area / total footprint area) | mill-town / industrial corridors |
| `commercial_share` | √(commercial footprint area / total footprint area) | commercial corridors |
| `river_proximity` | exp(−d_river / 5,000 ft), d to the Ohio, Allegheny, Monongahela, Youghiogheny | river-valley industrial legacy |
| `log_dist_core` | log(1 + miles to the Point) | urban–suburban gradient |

Dwelling units per structure: land-use codes 10/20/30/40 → 1/2/3/4 units; apartment codes
401–409 → max(5, 3 floors × footprint / 1,000 sq ft); other residential → max(1, footprint / 1,100).
Household counts are allocated in proportion to dwelling units and scaled to 553,000 county
households with the **largest-remainder method**, which preserves the integer total exactly.
Tracts with fewer than 60 estimated dwelling units (parks, the airport, institutional land) get no
estimate, mimicking ACS suppression: 388 of 402 tracts are mapped.

## 3. Synthetic income surface (regression kriging)

Let *y* be log median household income and **z** the standardised covariate vector.

**3a. Trend (ridge regression).** At the 149 tracts that contain a calibration anchor
(`config/calibration_anchors.csv`, approximate community medians), fit

  β̂ = (ZᵀZ + λI)⁻¹ Zᵀ(y − ȳ), λ = 2, intercept unpenalised.

Ridge shrinkage stabilises correlated covariates (density, single-family share and house size
are collinear). Trend R² = 0.62. The largest effect is house size (+0.29 log-income per SD);
industrial share is negative (−0.08 per SD).

**3b. Residual surface (simple kriging).** Residuals r = y − ŷ_trend at anchors are interpolated with an
exponential covariance C(h) = σ²ₚ·exp(−h / ρ) plus a nugget τ² on the diagonal:

  r̂(s₀) = c₀ᵀ (C + τ²I)⁻¹ r.

τ² = 0.012 is the anchors' own measurement error (≈11 % CV, typical of ACS place medians),
σ²ₚ = var(r) − τ² = 0.102, and ρ = 6,500 ft (≈1.2 mi, a neighbourhood). The empirical
semivariogram (Matheron estimator, stored as layer `residual_semivariogram`) is essentially flat
beyond the shortest lag because anchors are 1–2 miles apart, so it cannot identify the range; ρ is
therefore a stated prior rather than a fitted value. Far from anchors r̂ → 0, so the surface reverts
to the covariate trend. **Leave-one-anchor-out validation** of trend + kriging gives
RMSE = 0.35 in log income (median absolute percentage error ≈ 25 %).

**3c. Spatially autocorrelated noise.** Unexplained neighbourhood texture is a simultaneous
autoregressive (SAR) draw on queen-contiguity weights W (row-standardised):

  ε = (I − ϱW)⁻¹ u, u ~ N(0, 0.07² I), ϱ = 0.55.

**3d. Level calibration.** Tract medians are mᵢ = c·exp(ŷ_trend,i + r̂ᵢ + εᵢ). The constant c is
solved so the **household-weighted county mixture median** (§4) equals $72,537, the approximate
ACS 2018–2022 county median. Because the mixture is a scale family in c, one bisection solve is
exact; c = 1.20 (the anchors are older and slightly lower than the target vintage).

## 4. Within-tract household distributions (Dagum model)

Each tract's households follow a **Dagum (Burr type III)** distribution, the standard parametric
model for income:

  F(x) = (1 + (x/b)^(−a))^(−p), median-parameterised by b = m·(2^(1/p) − 1)^(1/a),
  E[X] = b·Γ(p + 1/a)·Γ(1 − 1/a) / Γ(p) (finite for a > 1).

The common shape (a, p) is chosen by grid search so the county **mixture**
F_county(x) = Σ hᵢFᵢ(x) / Σ hᵢ reproduces a reference ACS-style bracket distribution
(`config/county_income_distribution_reference.csv`) under a χ² distance, re-solving c for each
candidate. Result: a = 3.6, p = 0.30, χ² = 0.0019. The heavy lower tail (p < 1) is what a
log-normal cannot produce; the log-normal version under-counted households below $10k by two-thirds.
Shape *a* varies by tract with a 4 % log-normal jitter. Bracket counts per tract (ACS B19001's 16
brackets) come from differences of Fᵢ at bracket edges, rounded by largest remainder. Margins of
error follow ACS practice: MOE₉₀ = 1.645·CV·m with CV = 0.06 + 2.2/√households.

## 5. Thematic classification

The map uses **county-median-relative classes** at 50 %, 80 %, 120 % and 160 % of the county
median, rounded to $1,000: breaks **$36,000 / $58,000 / $87,000 / $116,000**. These are the
"area median income" bands used by HUD and community-development programs, so every colour has a
policy meaning (below 50 % ≈ very low income, 80–120 % ≈ moderate/middle, etc.), and the diverging
ColorBrewer PuOr palette (colour-blind safe) puts the neutral colour exactly on the county median.

Alternatives were computed for comparison (layer `classification_comparison`):

| method | breaks | GVF | TAI | tracts per class |
|---|---|---|---|---|
| Median-relative (map) | 36k / 58k / 87k / 116k | 0.875 | 0.716 | 26, 113, 118, 79, 52 |
| Fisher–Jenks | 56k / 86k / 126k / 186k | 0.924 | 0.708 | 133, 124, 92, 37, 2 |
| Quintiles | 47k / 60k / 79k / 103k | 0.843 | 0.712 | 78, 77, 78, 76, 79 |

*Fisher–Jenks* is implemented exactly by dynamic programming,
cost[j][m] = minᵢ cost[j−1][i−1] + SSD(i..m), with SSD from prefix sums (O(k·n²)); a unit test checks
it against brute-force enumeration. GVF = 1 − SDCM/SDAM; TAI = 1 − SADCM/SADAM. Jenks maximises GVF
but isolates two tracts in its top class and merges every tract below $56k (a third of the county)
into one colour, hiding the low-income contrast the brief asks for. The median-relative scheme gives
up 0.05 GVF to deliver interpretable, policy-standard, well-populated classes; it has the best TAI.

## 6. Summary statistics

* **Gini coefficient** of the household mixture: G = 1 − (1/μ) ∫₀^∞ (1 − F(x))² dx, evaluated by the
  trapezoid rule on an 8,000-point log grid: **G = 0.48** (Allegheny's published figure is ≈ 0.49).
  The between-tract Gini of medians (household weighted, Lorenz trapezoids) is 0.24, so roughly half
  of county inequality is *within* neighbourhoods.
* **Global Moran's I** on log tract medians with row-standardised queen weights:
  I = (n/S₀)·(zᵀWz)/(zᵀz) = **0.67**, E[I] = −1/(n−1) = −0.003, z = 21.9, permutation p = 0.001
  (999 permutations). Income is strongly spatially clustered.
* **Local Moran's I (LISA)** with conditional randomisation (999 draws, α = 0.05): 83 high-high
  tracts (north and south-western suburbs, Fox Chapel, Squirrel Hill) and 80 low-low tracts (the Hill
  District, Homewood, the Mon Valley mill towns). Stored per tract as `lisa_I`, `lisa_p`,
  `lisa_cluster`.
* Countywide: 17.4 % of households earn under $25k and 20.3 % earn $150k or more.

## 7. Reproducibility

Everything is deterministic given the seed. `make all` rebuilds the GeoPackage, chart, QGIS
project and exports; `make test` runs 19 tests (the mathematics plus checks on the deliverables).
