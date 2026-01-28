# report_builder/components.py
"""
PDF-Komponenten für PeakGuard.
Enthält Tabellen, Karten und UI-Elemente für ReportLab.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER
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

    # KPI-Kachel Wert (große Zahl) - zentriert wie auf Website
    styles.add(ParagraphStyle(
        name='KPIValue',
        parent=styles['Normal'],
        fontSize=DesignTokens.FONT_SIZE_XXL,
        textColor=PeakGuardColors.PRIMARY,
        fontName='Helvetica-Bold',
        alignment=TA_CENTER,
        leading=26,
        spaceBefore=4,
        spaceAfter=4
    ))

    # KPI-Kachel Label - zentriert, etwas größer
    styles.add(ParagraphStyle(
        name='KPILabel',
        parent=styles['Normal'],
        fontSize=DesignTokens.FONT_SIZE_BASE,
        textColor=PeakGuardColors.GRAY_DARK,
        fontName='Helvetica',
        alignment=TA_CENTER,
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

    # Small Body (für Cards) - zentriert für KPI-Subtexte
    styles.add(ParagraphStyle(
        name='BodySmall',
        parent=styles['BodyText'],
        fontSize=DesignTokens.FONT_SIZE_S,
        textColor=PeakGuardColors.GRAY,
        leading=12,
        alignment=TA_CENTER
    ))

    # Body Small Left (für Beschreibungen, die linksbündig sein sollen)
    styles.add(ParagraphStyle(
        name='BodySmallLeft',
        parent=styles['BodyText'],
        fontSize=DesignTokens.FONT_SIZE_S,
        textColor=PeakGuardColors.GRAY_DARK,
        leading=12,
        alignment=TA_LEFT
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
        """Basis-Style für alle Tabellen - modernes, cleanes Design"""
        return [
            # Schriftart für schärfere Darstellung
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),  # Etwas größer für bessere Lesbarkeit
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 10),
            ('RIGHTPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            # Dezenter Rahmen statt starker Box - schärfere Linien
            ('LINEABOVE', (0, 0), (-1, 0), 1.5, PeakGuardColors.PRIMARY),
            ('LINEBELOW', (0, -1), (-1, -1), 0.75, PeakGuardColors.GRAY_LIGHT),
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
        """Szenario-Vergleichstabelle - modernes Card-Design"""
        style = cls._base_style() + [
            # Header im Primary-Blau
            ('BACKGROUND', (0, 0), (-1, 0), PeakGuardColors.PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            # Dezente Zebra-Streifen
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PeakGuardColors.GRAY_LIGHTER]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            # Mehr Padding für bessere Lesbarkeit
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
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
        """Top-Peaks-Tabelle - modernes Design mit Akzentfarbe"""
        style = cls._base_style() + [
            # Header im Accent-Blau (heller, freundlicher)
            ('BACKGROUND', (0, 0), (-1, 0), PeakGuardColors.ACCENT),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            # Dezente Zebra-Streifen
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PeakGuardColors.GRAY_LIGHTER]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (3, 1), (4, -1), 'RIGHT'),
            # Guter Padding für Lesbarkeit
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
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
def create_kpi_card(label: str, value: str, subtext: str = "", styles=None,
                    accent_color=None) -> Table:
    """Erstellt eine moderne KPI-Kachel (Card-Design) - Website-Stil
    Mit farbigem Akzent-Streifen oben für professionelles Aussehen
    """
    if styles is None:
        styles = get_custom_styles()

    if accent_color is None:
        accent_color = PeakGuardColors.PRIMARY

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
            # Weißer Hintergrund wie Website
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            # Dezenter Rahmen
            ('BOX', (0, 0), (-1, -1), 0.5, PeakGuardColors.GRAY_LIGHT),
            # Farbiger Akzent-Streifen oben (wie Website-Gradient)
            ('LINEABOVE', (0, 0), (-1, 0), 3, accent_color),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 4),
            ('TOPPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 6),
        ])
    )
    return card


def create_action_card(priority: str, title: str, description: str, styles=None) -> Table:
    """Erstellt eine Action-Card für Top-3-Hebel - Website-Design"""
    if styles is None:
        styles = get_custom_styles()

    # Farbe nach Priorität - abgestimmt auf Website
    prio_color = PeakGuardColors.SUCCESS  # Grün für Quick-Wins
    if "invest" in priority.lower():
        prio_color = PeakGuardColors.WARNING  # Orange für Investitionen
    elif "quick" in priority.lower():
        prio_color = PeakGuardColors.SUCCESS  # Grün für Quick-Wins
    elif "prozess" in priority.lower() or "organisation" in priority.lower():
        prio_color = PeakGuardColors.ACCENT  # Blau für Prozesse

    # Linksbündige Styles für Beschreibungen
    body_style = styles.get('BodySmallLeft', styles['BodySmall'])

    content = [
        [Paragraph(f"<b>{priority}</b>", body_style)],
        [Paragraph(title, styles['CustomHeading3'])],
        [Paragraph(description, body_style)],
    ]

    card = Table(
        content,
        colWidths=[DesignTokens.COL_3 - 4 * mm],
        style=TableStyle([
            # Weißer Hintergrund
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            # Dezenter Rahmen
            ('BOX', (0, 0), (-1, -1), 0.5, PeakGuardColors.GRAY_LIGHT),
            # Farbiger Akzent-Streifen oben
            ('LINEABOVE', (0, 0), (-1, 0), 3, prio_color),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 4),
            ('TOPPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 4),
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

    # Farbcodierung nach Paket - abgestimmt auf Website MPL-Farben
    color_map = {
        "Bronze": colors.HexColor("#CD7F32"),  # Bronze
        "Silber": colors.HexColor("#95A5A6"),  # Silber (Website)
        "Gold": colors.HexColor("#F1C40F"),  # Gold (Website)
    }
    border_color = color_map.get(name, PeakGuardColors.GRAY)

    # Styles für linksbündige Texte in der Szenario-Card
    body_style = styles.get('BodySmallLeft', styles['BodySmall'])

    content = [
        [Paragraph(f"<b>{name}</b>", styles['CustomHeading3'])],
        [Paragraph(f"<b>Cap:</b> {fmt_num(cap_kw, 1, 'kW')}", body_style)],
        [Paragraph(f"<b>Peak nachher:</b> {fmt_num(peak_after, 1, 'kW')}", body_style)],
        [Paragraph(f"<b>Benutzungsdauer:</b> {fmt_num(util_hours, 0, 'h/a')}", body_style)],
        [Paragraph(f"<b>Tarif:</b> {tariff_label}", body_style)],
        [Paragraph(f"<b>Einsparung:</b> {fmt_num(savings, 0, '€/a')}", styles['KPIValue'])],
    ]

    card = Table(
        content,
        colWidths=[DesignTokens.COL_3 - 4 * mm],
        style=TableStyle([
            # Weißer Hintergrund
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            # Dezenter Rahmen
            ('BOX', (0, 0), (-1, -1), 0.5, PeakGuardColors.GRAY_LIGHT),
            # Farbiger Akzent-Streifen oben (wie KPI-Card)
            ('LINEABOVE', (0, 0), (-1, 0), 4, border_color),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            # Letzte Zeile (Einsparung) zentriert
            ('ALIGN', (0, -1), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 4),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 4),
            ('TOPPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING + 2),
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
