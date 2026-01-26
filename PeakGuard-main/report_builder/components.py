# report_builder/components.py
"""
PDF-Komponenten für PeakGuard.
Enthält Tabellen, Karten und UI-Elemente für ReportLab.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    Image,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from .config import DesignTokens, PeakGuardColors
from .models import Recommendation, TableData
from .utils import fmt_num


# ============================================================================
# STYLES
# ============================================================================
def get_custom_styles():
    """Erstellt das PeakGuard-Stylesheet für ReportLab"""
    styles = getSampleStyleSheet()

    # Executive Summary - sehr prominent
    styles.add(ParagraphStyle(
        name='ExecTitle',
        parent=styles['Heading1'],
        fontSize=DesignTokens.FONT_SIZE_HUGE,
        textColor=PeakGuardColors.PRIMARY,
        spaceAfter=DesignTokens.SPACE_L,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    ))

    # KPI-Kachel Wert (große Zahl)
    styles.add(ParagraphStyle(
        name='KPIValue',
        parent=styles['Normal'],
        fontSize=DesignTokens.FONT_SIZE_XXL,
        textColor=PeakGuardColors.PRIMARY,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT,
        leading=24
    ))

    # KPI-Kachel Label
    styles.add(ParagraphStyle(
        name='KPILabel',
        parent=styles['Normal'],
        fontSize=DesignTokens.FONT_SIZE_S,
        textColor=PeakGuardColors.GRAY_DARK,
        fontName='Helvetica',
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=DesignTokens.SPACE_XS
    ))

    # Standard Title
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=DesignTokens.FONT_SIZE_XXL,
        textColor=PeakGuardColors.PRIMARY,
        spaceAfter=DesignTokens.SPACE_L,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    ))

    # Section Heading
    styles.add(ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=DesignTokens.FONT_SIZE_L,
        textColor=PeakGuardColors.DARK,
        spaceBefore=DesignTokens.SPACE_XL,
        spaceAfter=DesignTokens.SPACE_M,
        fontName='Helvetica-Bold'
    ))

    # Subsection
    styles.add(ParagraphStyle(
        name='CustomHeading3',
        parent=styles['Heading3'],
        fontSize=DesignTokens.FONT_SIZE_M,
        textColor=PeakGuardColors.GRAY_DARK,
        spaceBefore=DesignTokens.SPACE_M,
        spaceAfter=DesignTokens.SPACE_S,
        fontName='Helvetica-Bold'
    ))

    # Body
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['BodyText'],
        fontSize=DesignTokens.FONT_SIZE_BASE,
        textColor=PeakGuardColors.DARK,
        leading=14
    ))

    # Small Body (für Cards)
    styles.add(ParagraphStyle(
        name='BodySmall',
        parent=styles['BodyText'],
        fontSize=DesignTokens.FONT_SIZE_S,
        textColor=PeakGuardColors.GRAY_DARK,
        leading=12
    ))

    # Caption
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontSize=DesignTokens.FONT_SIZE_XS,
        textColor=PeakGuardColors.GRAY,
        alignment=TA_LEFT,
        spaceBefore=DesignTokens.SPACE_XS
    ))

    return styles


# ============================================================================
# TABLE FACTORY (DRY)
# ============================================================================
class TableFactory:
    """
    Factory für konsistent gestylte Tabellen.
    Zentralisiert die Tabellen-Erstellung und reduziert Code-Duplikation.
    """

    @staticmethod
    def _base_style() -> list:
        """Basis-Style für alle Tabellen"""
        return [
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('BOX', (0, 0), (-1, -1), 0.5, PeakGuardColors.GRAY),
        ]

    @classmethod
    def info_table(cls, data: TableData) -> Table:
        """Info-Tabelle (Metadaten) mit modernem Design"""
        style = cls._base_style() + [
            ('BACKGROUND', (0, 0), (0, -1), PeakGuardColors.GRAY_LIGHTER),
            ('TEXTCOLOR', (0, 0), (0, -1), PeakGuardColors.GRAY_DARK),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('LINEBELOW', (0, 0), (-1, -2), 0.25, PeakGuardColors.GRAY_LIGHT),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ]
        return Table(data, colWidths=[45 * mm, 135 * mm], style=TableStyle(style))

    @classmethod
    def data_table(cls, data: TableData, highlight_last: bool = False) -> Table:
        """Daten-Tabelle mit optionaler Hervorhebung der letzten Zeile"""
        style = cls._base_style() + [
            ('BACKGROUND', (0, 0), (0, -1), PeakGuardColors.GRAY_LIGHTER),
            ('TEXTCOLOR', (0, 0), (0, -1), PeakGuardColors.GRAY_DARK),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('LINEBELOW', (0, 0), (-1, -2), 0.25, PeakGuardColors.GRAY_LIGHT),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ]

        if highlight_last:
            style.extend([
                ('BACKGROUND', (0, -1), (-1, -1), PeakGuardColors.SUCCESS),
                ('TEXTCOLOR', (0, -1), (-1, -1), colors.white),
                ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
                ('TOPPADDING', (0, -1), (-1, -1), 10),
                ('BOTTOMPADDING', (0, -1), (-1, -1), 10),
            ])

        return Table(data, colWidths=[75 * mm, 105 * mm], style=TableStyle(style))

    @classmethod
    def scenario_table(cls, data: TableData) -> Table:
        """Szenario-Vergleichstabelle"""
        style = cls._base_style() + [
            ('BACKGROUND', (0, 0), (-1, 0), PeakGuardColors.PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PeakGuardColors.GRAY_LIGHTER]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]
        return Table(
            data,
            colWidths=[38 * mm, 25 * mm, 32 * mm, 35 * mm, 25 * mm, 25 * mm],
            style=TableStyle(style)
        )

    @classmethod
    def recommendations_table(cls, data: TableData) -> Table:
        """Handlungsempfehlungen-Tabelle"""
        style = cls._base_style() + [
            ('BACKGROUND', (0, 0), (-1, 0), PeakGuardColors.DARK),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('LINEBELOW', (0, 0), (-1, -1), 0.25, PeakGuardColors.GRAY_LIGHT),
        ]
        return Table(data, colWidths=[45 * mm, 55 * mm, 80 * mm], style=TableStyle(style))

    @classmethod
    def peaks_table(cls, data: TableData) -> Table:
        """Top-Peaks-Tabelle"""
        style = cls._base_style() + [
            ('BACKGROUND', (0, 0), (-1, 0), PeakGuardColors.ACCENT),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('LINEBELOW', (0, 0), (-1, -1), 0.25, PeakGuardColors.GRAY_LIGHT),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PeakGuardColors.GRAY_LIGHTER]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (3, 1), (4, -1), 'RIGHT'),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]
        return Table(
            data,
            colWidths=[10 * mm, 32 * mm, 34 * mm, 34 * mm, 34 * mm],
            style=TableStyle(style)
        )


# Backwards compatibility
create_info_table = TableFactory.info_table
create_data_table = TableFactory.data_table
create_scenario_table = TableFactory.scenario_table
create_recommendations_table = TableFactory.recommendations_table
create_peaks_table = TableFactory.peaks_table


# ============================================================================
# CARD COMPONENTS
# ============================================================================
def create_kpi_card(label: str, value: str, subtext: str = "", styles=None) -> Table:
    """Erstellt eine moderne KPI-Kachel (Card-Design)"""
    if styles is None:
        styles = get_custom_styles()

    content = [
        [Paragraph(label, styles['KPILabel'])],
        [Paragraph(value, styles['KPIValue'])],
    ]
    if subtext:
        content.append([Paragraph(subtext, styles['BodySmall'])])

    card = Table(
        content,
        colWidths=[DesignTokens.COL_3 - 4 * mm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PeakGuardColors.GRAY_LIGHTER),
            ('BOX', (0, 0), (-1, -1), 1, PeakGuardColors.GRAY),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 2),
            ('TOPPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 2),
        ])
    )
    return card


def create_action_card(priority: str, title: str, description: str, styles=None) -> Table:
    """Erstellt eine Action-Card für Top-3-Hebel"""
    if styles is None:
        styles = get_custom_styles()

    # Farbe nach Priorität
    prio_color = PeakGuardColors.SUCCESS
    if "invest" in priority.lower():
        prio_color = PeakGuardColors.WARNING
    elif "quick" in priority.lower():
        prio_color = PeakGuardColors.SUCCESS

    content = [
        [Paragraph(f"<b>{priority}</b>", styles['BodySmall'])],
        [Paragraph(title, styles['CustomHeading3'])],
        [Paragraph(description, styles['BodySmall'])],
    ]

    card = Table(
        content,
        colWidths=[DesignTokens.COL_3 - 4 * mm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 1, prio_color),
            ('LINEABOVE', (0, 0), (-1, 0), 3, prio_color),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('TOPPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('BOTTOMPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
        ])
    )
    return card


def create_scenario_card(
    name: str,
    cap_kw: float,
    peak_after: float,
    savings: float,
    util_hours: float,
    tariff_label: str,
    styles=None
) -> Table:
    """Erstellt eine Szenario-Card (Bronze/Silber/Gold)"""
    if styles is None:
        styles = get_custom_styles()

    # Farbcodierung nach Paket
    color_map = {
        "Bronze": colors.HexColor("#CD7F32"),
        "Silber": colors.HexColor("#C0C0C0"),
        "Gold": colors.HexColor("#FFD700"),
    }
    border_color = color_map.get(name, PeakGuardColors.GRAY)

    content = [
        [Paragraph(f"<b>{name}</b>", styles['CustomHeading3'])],
        [Paragraph(f"<b>Cap:</b> {fmt_num(cap_kw, 1, 'kW')}", styles['BodySmall'])],
        [Paragraph(f"<b>Peak nachher:</b> {fmt_num(peak_after, 1, 'kW')}", styles['BodySmall'])],
        [Paragraph(f"<b>Benutzungsdauer:</b> {fmt_num(util_hours, 0, 'h/a')}", styles['BodySmall'])],
        [Paragraph(f"<b>Tarif:</b> {tariff_label}", styles['BodySmall'])],
        [Paragraph(f"<b>Einsparung:</b> {fmt_num(savings, 0, '€/a')}", styles['KPIValue'])],
    ]

    card = Table(
        content,
        colWidths=[DesignTokens.COL_3 - 4 * mm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 2, border_color),
            ('LINEABOVE', (0, 0), (-1, 0), 4, border_color),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('TOPPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('BOTTOMPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
        ])
    )
    return card


# ============================================================================
# CHART HELPERS
# ============================================================================
def add_chart_with_caption(
    img_path: Path,
    caption: str,
    styles,
    width: float = 170 * mm
) -> List[Flowable]:
    """Fügt Chart + Caption als Flowable-Liste hinzu"""
    elements: List[Flowable] = []
    elements.append(Image(str(img_path), width=width, height=width * 0.5))
    if caption:
        elements.append(Spacer(1, DesignTokens.SPACE_XS))
        elements.append(Paragraph(caption, styles['Caption']))
    return elements


# ============================================================================
# TABLE BUILDERS
# ============================================================================
def build_recommendations_table_rows(
    recs: List[Recommendation],
    styles
) -> TableData:
    """Erstellt Zeilen für die Empfehlungstabelle"""
    def p(text: str) -> Paragraph:
        return Paragraph((text or "—").replace("\n", "<br/>"), styles["CustomBody"])

    rows: TableData = [["Kategorie", "Identifizierter Trigger", "Empfohlene Maßnahme"]]
    for r in recs:
        cat = r.category
        trig = f"{r.code}: {r.trigger}" if r.code and r.code != "—" else r.trigger
        rows.append([p(cat), p(trig), p(r.action)])

    return rows


def build_top_peaks_rows(df_15, n: int = 20) -> TableData:
    """Erstellt Zeilen für die Top-Peaks-Tabelle"""
    import numpy as np
    import pandas as pd
    from typing import cast

    idx = cast(pd.DatetimeIndex, df_15.index)

    cosphi: Optional[pd.Series] = None
    if "c_total" in df_15.columns:
        cosphi = pd.to_numeric(cast(pd.Series, df_15["c_total"]), errors="coerce").clip(lower=0.0, upper=1.0)
    elif all(c in df_15.columns for c in ["c1", "c2", "c3"]):
        c1 = pd.to_numeric(cast(pd.Series, df_15["c1"]), errors="coerce").clip(lower=0.0, upper=1.0)
        c2 = pd.to_numeric(cast(pd.Series, df_15["c2"]), errors="coerce").clip(lower=0.0, upper=1.0)
        c3 = pd.to_numeric(cast(pd.Series, df_15["c3"]), errors="coerce").clip(lower=0.0, upper=1.0)
        cosphi = (c1 + c2 + c3) / 3.0

    tmp = pd.DataFrame({
        "ts": idx,
        "p_kw": pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce"),
    })
    if cosphi is not None:
        tmp["cosphi"] = pd.to_numeric(cosphi, errors="coerce").to_numpy(dtype=float)

    tmp = tmp.dropna(subset=["p_kw"]).sort_values("p_kw", ascending=False).head(int(n))

    rows: TableData = [["#", "Datum", "Uhrzeit (15-min Block)", "Leistung (kW)", "cosϕ"]]

    for i, r in enumerate(tmp.itertuples(index=False), start=1):
        ts = cast(pd.Timestamp, getattr(r, "ts"))
        p_kw = float(getattr(r, "p_kw"))
        cval = getattr(r, "cosphi", None)

        date_s = f"{ts:%d.%m.%Y}"
        time_s = f"{ts:%H:%M}"
        p_s = f"{p_kw:.1f}".replace(".", ",") + " kW"

        if cval is None or (isinstance(cval, float) and (np.isnan(cval) or not np.isfinite(cval))):
            c_s = "—"
        else:
            c_s = f"{float(cval):.3f}".replace(".", ",")

        rows.append([str(i), date_s, time_s, p_s, c_s])

    if len(rows) == 1:
        rows.append(["—", "—", "—", "—", "—"])

    return rows
