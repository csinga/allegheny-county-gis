# Allegheny County household income analysis: reproducible build.
# Requires QGIS >= 3.30 with Python bindings plus the packages in requirements.txt
# (see README "Setup"). PY must be an interpreter that can `import qgis.core`.
PY ?= python3
export QT_QPA_PLATFORM ?= offscreen

.PHONY: all data qgis test clean

all: data qgis

data:   ## stage 1: inputs -> GeoPackage + chart
	$(PY) scripts/01_build_geopackage.py

qgis:   ## stage 2: GeoPackage -> .qgz + 300 DPI PNG + PDF
	$(PY) scripts/02_build_qgis_project.py

test:
	$(PY) -m pytest -q

clean:
	rm -rf output/*.gpkg output/*.qgz output/*.png output/*.pdf output/*.json output/figures
