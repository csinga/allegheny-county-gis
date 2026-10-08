#!/usr/bin/env python3
"""Stage 2: build the QGIS project (.qgz), its print layout, and export the 300 DPI PNG (+ PDF).

Runs headless with PyQGIS (QGIS >= 3.30). Reads only the GeoPackage and chart from stage 1.
Usage:  python scripts/02_build_qgis_project.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qgis.core import (  # noqa: E402
    Qgis,
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsFillSymbol,
    QgsLayerTreeLayer,
    QgsLayoutExporter,
    QgsLayoutItemLabel,
    QgsLayoutItemLegend,
    QgsLayoutItemMap,
    QgsLayoutItemPicture,
    QgsLayoutItemScaleBar,
    QgsLayoutItemShape,
    QgsLayoutMeasurement,
    QgsLayoutPoint,
    QgsLayoutSize,
    QgsLegendStyle,
    QgsLinePatternFillSymbolLayer,
    QgsLineSymbol,
    QgsNullSymbolRenderer,
    QgsPalLayerSettings,
    QgsPrintLayout,
    QgsProject,
    QgsProperty,
    QgsRectangle,
    QgsRendererCategory,
    QgsRuleBasedLabeling,
    QgsRuleBasedRenderer,
    QgsSimpleFillSymbolLayer,
    QgsSingleSymbolRenderer,
    QgsTextBackgroundSettings,
    QgsTextBufferSettings,
    QgsTextFormat,
    QgsUnitTypes,
    QgsVectorLayer,
    QgsVectorLayerSimpleLabeling,
)
from qgis.PyQt.QtCore import QSizeF, Qt  # noqa: E402
from qgis.PyQt.QtGui import QColor, QFont  # noqa: E402

from aci import settings as S  # noqa: E402

FONT = "Arial"
INK = "#222222"
MUTED = "#555555"
WATER = "#a9cbe3"
WATER_TEXT = "#2a6592"
MM = Qgis.RenderUnit.Millimeters
PT = Qgis.RenderUnit.Points
LMM = getattr(Qgis, "LayoutUnit", QgsUnitTypes).Millimeters if hasattr(Qgis, "LayoutUnit") else QgsUnitTypes.LayoutMillimeters


# --------------------------------------------------------------------------- helpers
def text_format(size: float, color: str = INK, bold: bool = False, italic: bool = False,
                buffer: float = 0.0, spacing: float = 0.0) -> QgsTextFormat:
    f = QFont(FONT)
    f.setBold(bold)
    f.setItalic(italic)
    if spacing:
        f.setLetterSpacing(QFont.AbsoluteSpacing, spacing)
    fmt = QgsTextFormat()
    fmt.setFont(f)
    fmt.setSize(size)
    fmt.setSizeUnit(PT)
    fmt.setColor(QColor(color))
    if buffer:
        b = QgsTextBufferSettings()
        b.setEnabled(True)
        b.setSize(buffer)
        b.setSizeUnit(MM)
        b.setColor(QColor(255, 255, 255, 235))
        fmt.setBuffer(b)
    return fmt


def layer(name: str, title: str) -> QgsVectorLayer:
    lyr = QgsVectorLayer(f"{S.GPKG_PATH}|layername={name}", title, "ogr")
    if not lyr.isValid():
        raise RuntimeError(f"Cannot open layer {name} from {S.GPKG_PATH}")
    return lyr


def set_dd(settings: QgsPalLayerSettings, prop: str, field: str) -> None:
    key = getattr(QgsPalLayerSettings.Property, prop, None) if hasattr(QgsPalLayerSettings, "Property") else None
    key = key if key is not None else getattr(QgsPalLayerSettings, prop)
    settings.dataDefinedProperties().setProperty(key, QgsProperty.fromField(field))


# --------------------------------------------------------------------------- symbology
def style_tracts(lyr: QgsVectorLayer, classes: list[dict]) -> None:
    outline = "92,92,92,150"
    cats = []
    for c in classes:
        sym = QgsFillSymbol.createSimple(
            {"color": c["color_hex"], "outline_color": outline, "outline_width": "0.09", "outline_width_unit": "MM"}
        )
        pct = {1: "below 50%", 2: "50-80%", 3: "80-120%", 4: "120-160%", 5: "above 160%"}[c["income_class"]]
        lab = f'{c["class_label"]}   {c["class_name"]} ({pct} of county median)'
        cats.append(QgsRendererCategory(c["income_class"], sym, lab))
    hatch = QgsFillSymbol.createSimple(
        {"color": "236,236,236", "outline_color": outline, "outline_width": "0.09", "outline_width_unit": "MM"}
    )
    lines = QgsLinePatternFillSymbolLayer()
    lines.setLineAngle(45)
    lines.setDistance(0.9)
    lines.setDistanceUnit(MM)
    lines.setLineWidth(0.12)
    lines.setColor(QColor("#9b9b9b"))
    hatch.appendSymbolLayer(lines)
    cats.append(QgsRendererCategory(0, hatch, "No estimate (parks, airport, institutional tracts)"))
    lyr.setRenderer(QgsCategorizedSymbolRenderer("income_class", cats))


def style_boundary(lyr: QgsVectorLayer) -> None:
    halo = QgsSimpleFillSymbolLayer.create(
        {"color": "0,0,0,0", "outline_color": "255,255,255,220", "outline_width": "1.4", "outline_width_unit": "MM"}
    )
    line = QgsSimpleFillSymbolLayer.create(
        {"color": "0,0,0,0", "outline_color": INK, "outline_width": "0.55", "outline_width_unit": "MM"}
    )
    sym = QgsFillSymbol([halo, line])
    lyr.setRenderer(QgsSingleSymbolRenderer(sym))


def style_water(lyr: QgsVectorLayer) -> None:
    lyr.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(
        {"color": WATER, "outline_color": "#7fb0d3", "outline_width": "0.08", "outline_width_unit": "MM"}
    )))


def style_roads(lyr: QgsVectorLayer) -> None:
    root = QgsRuleBasedRenderer.Rule(None)
    for flt, label, color, width in (
        ("\"route_sys\" <> 'I'", "US and state highway", "#7a7a7a", "0.28"),
        ("\"route_sys\" = 'I'", "Interstate highway", "#3d3d3d", "0.5"),
    ):
        casing = QgsLineSymbol.createSimple({"line_color": "255,255,255,170", "line_width": str(float(width) + 0.35), "capstyle": "round"})
        core = QgsLineSymbol.createSimple({"line_color": color, "line_width": width, "capstyle": "round"})
        casing.appendSymbolLayer(core.symbolLayer(0).clone())
        root.appendChild(QgsRuleBasedRenderer.Rule(casing, 0, 0, flt, label))
    lyr.setRenderer(QgsRuleBasedRenderer(root))

    s = QgsPalLayerSettings()
    s.isExpression = True
    s.fieldName = "replace(\"route\", 'I-', 'I-')"
    s.placement = Qgis.LabelPlacement.Horizontal
    s.repeatDistance = 85
    s.repeatDistanceUnit = MM
    s.priority = 3
    fmt = text_format(5.5, "#ffffff", bold=True)
    bg = QgsTextBackgroundSettings()
    bg.setEnabled(True)
    bg.setType(QgsTextBackgroundSettings.ShapeRectangle)
    bg.setSizeType(QgsTextBackgroundSettings.SizeBuffer)
    bg.setSize(QSizeF(0.7, 0.35))
    bg.setSizeUnit(MM)
    bg.setRadii(QSizeF(0.6, 0.6))
    bg.setFillColor(QColor("#3d5a80"))
    bg.setStrokeColor(QColor("#ffffff"))
    bg.setStrokeWidth(0.25)
    bg.setStrokeWidthUnit(MM)
    fmt.setBackground(bg)
    s.setFormat(fmt)
    rule = QgsRuleBasedLabeling.Rule(s, 0, 0, "\"route_sys\" IN ('I','US')")
    lroot = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
    lroot.appendChild(rule)
    lyr.setLabeling(QgsRuleBasedLabeling(lroot))
    lyr.setLabelsEnabled(True)


def style_labels(lyr: QgsVectorLayer) -> None:
    lyr.setRenderer(QgsNullSymbolRenderer())
    root = QgsRuleBasedLabeling.Rule(QgsPalLayerSettings())
    specs = (
        ("\"kind\" = 'river'", text_format(7.5, WATER_TEXT, italic=True, buffer=0.6, spacing=0.4), 9),
        ("\"kind\" = 'city'", text_format(10, INK, bold=True, buffer=0.9, spacing=1.2), 10),
        ("\"kind\" = 'community'", text_format(6.4, "#303030", buffer=0.65), 6),
        ("\"kind\" = 'neighborhood'", text_format(5.6, "#404040", italic=True, buffer=0.55), 4),
    )
    for flt, fmt, prio in specs:
        s = QgsPalLayerSettings()
        s.fieldName = "text"
        s.placement = Qgis.LabelPlacement.OverPoint
        s.priority = prio
        s.setFormat(fmt)
        set_dd(s, "LabelRotation", "rotation")
        root.appendChild(QgsRuleBasedLabeling.Rule(s, 0, 0, flt))
    lyr.setLabeling(QgsRuleBasedLabeling(root))
    lyr.setLabelsEnabled(True)


def style_pa(lyr: QgsVectorLayer) -> None:
    root = QgsRuleBasedRenderer.Rule(None)
    other = QgsFillSymbol.createSimple({"color": "#e4e4e4", "outline_color": "#ffffff", "outline_width": "0.12"})
    study = QgsFillSymbol.createSimple({"color": "#c0392b", "outline_color": "#5c1a13", "outline_width": "0.3"})
    root.appendChild(QgsRuleBasedRenderer.Rule(other, 0, 0, "\"is_study_area\" = 0", "Pennsylvania counties"))
    root.appendChild(QgsRuleBasedRenderer.Rule(study, 0, 0, "\"is_study_area\" = 1", "Allegheny County"))
    lyr.setRenderer(QgsRuleBasedRenderer(root))


def style_municipalities(lyr: QgsVectorLayer) -> None:
    lyr.setRenderer(QgsSingleSymbolRenderer(QgsFillSymbol.createSimple(
        {"color": "0,0,0,0", "outline_color": "#444444", "outline_width": "0.25", "outline_style": "dash"}
    )))
    s = QgsPalLayerSettings()
    s.fieldName = "name"
    s.setFormat(text_format(6, INK, buffer=0.5))
    lyr.setLabeling(QgsVectorLayerSimpleLabeling(s))
    lyr.setLabelsEnabled(True)


# --------------------------------------------------------------------------- layout helpers
def add_label(layout, text, x, y, w, h, fmt, halign=Qt.AlignLeft, valign=Qt.AlignTop) -> QgsLayoutItemLabel:
    item = QgsLayoutItemLabel(layout)
    item.setText(text)
    item.setTextFormat(fmt)
    item.setHAlign(halign)
    item.setVAlign(valign)
    item.setMarginX(0)
    item.setMarginY(0)
    layout.addLayoutItem(item)
    item.attemptMove(QgsLayoutPoint(x, y, LMM))
    item.attemptResize(QgsLayoutSize(w, h, LMM))
    return item


def add_rule(layout, x, y, w, color=INK, thickness=0.35) -> None:
    rect = QgsLayoutItemShape(layout)
    rect.setShapeType(QgsLayoutItemShape.Rectangle)
    rect.setSymbol(QgsFillSymbol.createSimple({"color": color, "outline_style": "no"}))
    layout.addLayoutItem(rect)
    rect.attemptMove(QgsLayoutPoint(x, y, LMM))
    rect.attemptResize(QgsLayoutSize(w, thickness, LMM))


def heading(layout, text, x, y, w) -> None:
    add_label(layout, text.upper(), x, y, w, 4.5, text_format(7.5, "#3d3d3d", bold=True, spacing=0.6))
    add_rule(layout, x, y + 5.0, w, "#9a9a9a", 0.2)


def fit_extent(extent: QgsRectangle, w_mm: float, h_mm: float, pad: float) -> QgsRectangle:
    """Grow an extent so it fills a frame of the given aspect ratio, with fractional padding."""
    cx, cy = extent.center().x(), extent.center().y()
    w, h = extent.width() * (1 + pad), extent.height() * (1 + pad)
    target = w_mm / h_mm
    if w / h < target:
        w = h * target
    else:
        h = w / target
    return QgsRectangle(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def style_legend(legend: QgsLayoutItemLegend) -> None:
    for style, size, bold, space in (
        (QgsLegendStyle.Title, 8.5, True, 0.0),
        (QgsLegendStyle.Group, 7.2, True, 2.6),
        (QgsLegendStyle.Subgroup, 7.0, True, 2.4),
        (QgsLegendStyle.SymbolLabel, 6.6, False, 0.0),
    ):
        st = legend.rstyle(style)
        st.setTextFormat(text_format(size, INK, bold=bold))
        if space:
            st.setMargin(QgsLegendStyle.Top, space)
    legend.rstyle(QgsLegendStyle.Symbol).setMargin(QgsLegendStyle.Top, 1.1)
    legend.setSymbolWidth(7.5)
    legend.setSymbolHeight(3.6)
    legend.setBoxSpace(0)
    legend.setFrameEnabled(False)
    legend.setBackgroundEnabled(False)


# --------------------------------------------------------------------------- main
def main() -> None:
    qgs = QgsApplication([], False)
    qgs.initQgis()
    stats = json.loads((S.OUTPUT_DIR / "summary_statistics.json").read_text())

    project = QgsProject.instance()
    project.clear()
    project.setFileName(str(S.QGZ_PATH))
    project.setTitle("Household Income Distribution - Allegheny County, PA")
    project.setCrs(QgsCoordinateReferenceSystem(S.CRS_ANALYSIS))
    project.setFilePathStorage(Qgis.FilePathType.Relative)

    pa = layer("pennsylvania_counties", "Pennsylvania counties")
    munis = layer("municipalities", "Community reference areas (approximate, synthetic)")
    tracts = layer("income_tracts", "Median household income by census tract")
    water = layer("waterways", "Rivers and water bodies")
    roads = layer("roads", "Major roads")
    boundary = layer("allegheny_boundary", "Allegheny County boundary")
    labels = layer("map_labels", "Map labels")

    classes_lyr = layer("income_class_summary", "income_class_summary")
    classes = sorted(
        ({k: f[k] for k in ("income_class", "class_name", "class_label", "color_hex")} for f in classes_lyr.getFeatures()),
        key=lambda d: d["income_class"],
    )
    style_tracts(tracts, classes)
    style_boundary(boundary)
    style_water(water)
    style_roads(roads)
    style_labels(labels)
    style_pa(pa)
    style_municipalities(munis)

    # Layer tree: draw order top -> bottom. Analysis tables are added (not drawn) for review.
    for lyr in (labels, boundary, roads, water, munis, tracts, pa):
        project.addMapLayer(lyr)
    project.layerTreeRoot().findLayer(munis.id()).setItemVisibilityChecked(False)
    project.layerTreeRoot().findLayer(pa.id()).setItemVisibilityChecked(False)
    group = project.layerTreeRoot().addGroup("Analysis tables")
    for t in ("county_income_distribution", "income_class_summary", "household_income_brackets",
              "classification_comparison", "model_coefficients", "residual_semivariogram",
              "summary_statistics", "data_sources", "calibration_anchors"):
        lyr = layer(t, t)
        project.addMapLayer(lyr, False)
        group.addLayer(lyr)
    group.setExpanded(False)
    root = project.layerTreeRoot()
    order = [labels, boundary, roads, water, munis, tracts, pa]
    for i, lyr in enumerate(order):
        node = root.findLayer(lyr.id())
        clone = node.clone()
        root.insertChildNode(i, clone)
        root.removeChildNode(node)

    # ----------------------------------------------------------------- layout
    layout = QgsPrintLayout(project)
    layout.initializeDefaults()
    layout.setName("Household Income Map 17x11")
    page = layout.pageCollection().page(0)
    pw, ph = S.PAGE_MM
    page.setPageSize(QgsLayoutSize(pw, ph, LMM))
    margin = 12.0
    col_x = 284.0
    col_w = pw - margin - col_x

    add_label(layout, "Household Income Across Allegheny County, Pennsylvania", margin, 9.5, 330, 12,
              text_format(22, INK, bold=True))
    add_label(
        layout,
        "Median household income by census tract, classed relative to the county median "
        f"(${stats['county_median_hh_income']:,.0f}). Modelled 5-year estimates for neighbourhood comparison.",
        margin, 22.5, 330, 6, text_format(10.5, MUTED),
    )
    add_label(layout, "Prepared for the Allegheny County\nDepartment of Economic Development",
              pw - margin - 85, 11, 85, 10, text_format(8, MUTED), halign=Qt.AlignRight)
    add_rule(layout, margin, 31.0, pw - 2 * margin, INK, 0.5)

    # Main map
    mx, my, mw, mh = margin, 35.0, 266.0, 221.0
    main_map = QgsLayoutItemMap(layout)
    main_map.setId("main_map")
    layout.addLayoutItem(main_map)
    main_map.attemptMove(QgsLayoutPoint(mx, my, LMM))
    main_map.attemptResize(QgsLayoutSize(mw, mh, LMM))
    main_map.setCrs(QgsCoordinateReferenceSystem(S.CRS_ANALYSIS))
    main_map.setLayers([labels, boundary, roads, water, tracts])
    main_map.setKeepLayerSet(True)
    main_map.setExtent(fit_extent(boundary.extent(), mw, mh, 0.04))
    main_map.setFrameEnabled(True)
    main_map.setFrameStrokeWidth(QgsLayoutMeasurement(0.3, LMM))
    main_map.setFrameStrokeColor(QColor("#7a7a7a"))
    main_map.setBackgroundColor(QColor("#ffffff"))

    # North arrow (top-left inside the frame, where the county leaves white space)
    arrow = QgsLayoutItemPicture(layout)
    arrow.setPicturePath(str(Path(QgsApplication.svgPaths()[0]) / "arrows" / "NorthArrow_02.svg")
                         if (Path(QgsApplication.svgPaths()[0]) / "arrows" / "NorthArrow_02.svg").exists()
                         else "/usr/share/qgis/svg/arrows/NorthArrow_02.svg")
    arrow.setSvgFillColor(QColor(INK))
    arrow.setSvgStrokeColor(QColor(INK))
    arrow.setLinkedMap(main_map)
    layout.addLayoutItem(arrow)
    arrow.attemptMove(QgsLayoutPoint(mx + 8, my + 7, LMM))
    arrow.attemptResize(QgsLayoutSize(11, 17, LMM))

    # Scale bar (bottom-left inside the frame)
    sb = QgsLayoutItemScaleBar(layout)
    sb.setStyle("Single Box")
    sb.setLinkedMap(main_map)
    sb.setUnits(Qgis.DistanceUnit.Miles)
    sb.setUnitLabel("Miles")
    sb.setUnitsPerSegment(2.5)
    sb.setNumberOfSegments(4)
    sb.setNumberOfSegmentsLeft(0)
    sb.setHeight(2.0)
    sb.setTextFormat(text_format(7, INK))
    sb.setLineSymbol(QgsLineSymbol.createSimple({"line_color": INK, "line_width": "0.25"}))
    sb.setFillSymbol(QgsFillSymbol.createSimple({"color": INK, "outline_style": "no"}))
    sb.setAlternateFillSymbol(QgsFillSymbol.createSimple({"color": "#ffffff", "outline_style": "no"}))
    sb.setBackgroundEnabled(False)
    layout.addLayoutItem(sb)
    sb.attemptMove(QgsLayoutPoint(mx + 8, my + mh - 20, LMM))
    add_label(layout, "Projection: NAD83 / Pennsylvania South (ftUS), EPSG:2272", mx + 8, my + mh - 9, 110, 4,
              text_format(6, MUTED))

    # Locator inset
    heading(layout, "Location", col_x, my, col_w)
    inset_y, inset_h = my + 8, 54.0
    inset = QgsLayoutItemMap(layout)
    inset.setId("locator_inset")
    layout.addLayoutItem(inset)
    inset.attemptMove(QgsLayoutPoint(col_x, inset_y, LMM))
    inset.attemptResize(QgsLayoutSize(col_w, inset_h, LMM))
    lcrs = QgsCoordinateReferenceSystem.fromProj(S.CRS_LOCATOR)
    inset.setCrs(lcrs)
    inset.setLayers([pa])
    inset.setKeepLayerSet(True)
    xform = QgsCoordinateTransform(pa.crs(), lcrs, project)
    inset.setExtent(fit_extent(xform.transformBoundingBox(pa.extent()), col_w, inset_h, 0.08))
    inset.setFrameEnabled(True)
    inset.setFrameStrokeWidth(QgsLayoutMeasurement(0.25, LMM))
    inset.setFrameStrokeColor(QColor("#9a9a9a"))
    inset.setBackgroundColor(QColor("#f7f9fb"))
    add_label(layout, "PENNSYLVANIA", col_x + 2, inset_y + 2, 40, 4, text_format(6.5, "#7a7a7a", bold=True, spacing=0.8))
    add_label(layout, "Allegheny County (red)\nin southwestern Pennsylvania", col_x + col_w - 52, inset_y + inset_h - 9,
              50, 8, text_format(6, MUTED), halign=Qt.AlignRight)

    # Legend
    leg_y = inset_y + inset_h + 5
    heading(layout, "Legend", col_x, leg_y, col_w)
    legend = QgsLayoutItemLegend(layout)
    legend.setTitle("")
    legend.setLinkedMap(main_map)
    legend.setAutoUpdateModel(False)
    lroot = legend.model().rootGroup()
    lroot.removeAllChildren()
    for lyr, title in (
        (tracts, "Median household income (tract)"),
        (boundary, "Allegheny County boundary"),
        (water, "Rivers and water bodies"),
        (roads, "Major roads"),
    ):
        node = lroot.addLayer(lyr)
        node.setUseLayerName(False)
        node.setName(title)
    layout.addLayoutItem(legend)
    style_legend(legend)
    legend.setColumnCount(1)
    legend.attemptMove(QgsLayoutPoint(col_x, leg_y + 7, LMM))
    legend.setResizeToContents(True)
    legend.adjustBoxSize()

    # Summary chart
    chart_y = 176.0
    heading(layout, "Countywide income structure", col_x, chart_y, col_w)
    pic = QgsLayoutItemPicture(layout)
    pic.setPicturePath(str(S.CHART_PATH.with_suffix(".png")))
    pic.setResizeMode(QgsLayoutItemPicture.Zoom)
    layout.addLayoutItem(pic)
    cw, ch = S.CHART_MM
    pic.attemptMove(QgsLayoutPoint(col_x, chart_y + 6.5, LMM))
    pic.attemptResize(QgsLayoutSize(cw, ch, LMM))

    # Key statistics strip under the chart
    lisa = stats["lisa_counts"]
    stat_text = (
        f"Tracts mapped: {stats['tracts_with_estimate']} of {stats['tracts_total']}   |   "
        f"Range of tract medians: ${stats['tract_median_min']:,.0f} to ${stats['tract_median_max']:,.0f}\n"
        f"Household Gini: {stats['gini_household_mixture']:.2f}   |   "
        f"Global Moran's I: {stats['morans_I']:.2f} (p = {stats['morans_p_sim']:.3f}): incomes cluster spatially\n"
        f"Significant clusters (LISA, p < 0.05): {lisa.get('HH', 0)} high-high and {lisa.get('LL', 0)} low-low tracts"
    )
    add_label(layout, stat_text, col_x, chart_y + 6.5 + ch + 1.5, col_w, 11, text_format(6.3, INK))

    # Footer: classification note, sources, credits
    add_rule(layout, margin, 259.5, pw - 2 * margin, "#9a9a9a", 0.2)
    b = stats["class_breaks_usd"]
    note = (
        "Classification: five classes at 50%, 80%, 120% and 160% of the county median household income "
        f"(breaks ${b[0]:,.0f} / ${b[1]:,.0f} / ${b[2]:,.0f} / ${b[3]:,.0f}), the area-median-income convention used in "
        "housing and community-development programs. Fisher-Jenks and quantile alternatives are compared in the GeoPackage."
    )
    sources_txt = (
        "Data sources: census tracts and building footprints, Allegheny County GIS (tracts 2016); rivers and water bodies, "
        "U.S. Census Bureau TIGER/Line 2025 Area Water; major roads and Pennsylvania counties, Natural Earth 1:10m (public domain). "
        "Household income values are SYNTHETIC modelled estimates (regression kriging of building-stock covariates, calibrated to approximate "
        "community medians and to the county median); they illustrate the method and are not official ACS figures. Labelled community "
        "locations are approximate."
    )
    add_label(layout, note + "\n" + sources_txt, margin, 262.0, 330, 12, text_format(6.2, MUTED))
    add_label(
        layout,
        f"Map produced {date.today():%B %-d, %Y}\nQGIS {Qgis.version().split('-')[0]}  |  Allegheny County Household Income Analysis",
        pw - margin - 80, 262.0, 80, 9, text_format(6.2, MUTED), halign=Qt.AlignRight,
    )

    project.layoutManager().addLayout(layout)
    if not project.write(str(S.QGZ_PATH)):
        raise RuntimeError("Failed to write project")
    relativise_legend_sources(S.QGZ_PATH)

    exporter = QgsLayoutExporter(layout)
    img = QgsLayoutExporter.ImageExportSettings()
    img.dpi = 300
    res = exporter.exportToImage(str(S.PNG_PATH), img)
    if res != QgsLayoutExporter.Success:
        raise RuntimeError(f"PNG export failed: {res}")
    pdf = QgsLayoutExporter.PdfExportSettings()
    pdf.dpi = 300
    pdf.rasterizeWholeImage = False
    exporter.exportToPdf(str(S.PDF_PATH), pdf)
    print(f"Wrote {S.QGZ_PATH}\nWrote {S.PNG_PATH}")
    qgs.exitQgis()


def relativise_legend_sources(qgz: Path) -> None:
    """Layout legends store layer sources as absolute paths; make them relative so the
    project folder stays portable like the rest of the .qgz."""
    import zipfile

    with zipfile.ZipFile(qgz) as zf:
        members = {n: zf.read(n) for n in zf.namelist()}
    prefix = (str(qgz.parent) + "/").encode()
    members = {n: (b.replace(prefix, b"./") if n.endswith(".qgs") else b) for n, b in members.items()}
    with zipfile.ZipFile(qgz, "w", zipfile.ZIP_DEFLATED) as zf:
        for n, b in members.items():
            zf.writestr(n, b)


if __name__ == "__main__":
    main()
