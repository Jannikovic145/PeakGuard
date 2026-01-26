# report_builder.py (v3.1: Optimiertes Design & moderne Visualisierungen)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, cast

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    Flowable,
)

# ============================================================================
# DESIGN SYSTEM - PeakGuard Corporate Identity v2
# ============================================================================
class DesignTokens:
    """Zentrale Design-Tokens für konsistente Gestaltung (v2)"""

    # === SPACING SYSTEM (mm) ===
    SPACE_XS = 2 * mm   # 2mm - Minimaler Abstand
    SPACE_S = 4 * mm    # 4mm - Kleiner Abstand
    SPACE_M = 6 * mm    # 6mm - Standard-Abstand
    SPACE_L = 8 * mm    # 8mm - Großer Abstand
    SPACE_XL = 12 * mm  # 12mm - Sehr großer Abstand
    SPACE_XXL = 16 * mm # 16mm - Section-Trenner

    # === TYPOGRAPHY ===
    FONT_SIZE_HUGE = 24      # Executive Summary Title
    FONT_SIZE_XXL = 20       # Hauptüberschrift
    FONT_SIZE_XL = 16        # Section Heading
    FONT_SIZE_L = 14         # Subsection
    FONT_SIZE_M = 12         # KPI-Kachel Wert
    FONT_SIZE_BASE = 10      # Body Text
    FONT_SIZE_S = 9          # Secondary Text
    FONT_SIZE_XS = 8         # Footer/Caption
    FONT_SIZE_XXS = 7        # Mini-Labels

    # === CARD DESIGN ===
    CARD_PADDING = 8         # Innerer Abstand in Cards (pt)
    CARD_RADIUS = 4          # Eckenradius (nur für Charts, PDF-tauglich)
    CARD_BORDER_WIDTH = 0.5  # Rahmenbreite

    # === GRID SYSTEM ===
    PAGE_WIDTH = 180 * mm    # Nutzbare Seitenbreite
    COL_2 = 88 * mm          # 2-Spalten Layout (mit Gutter)
    COL_3 = 58 * mm          # 3-Spalten Layout
    GUTTER = 4 * mm          # Abstand zwischen Spalten

class PeakGuardDesign:
    # KORRIGIERTE PeakGuard Farben (unverändert)
    PRIMARY = colors.HexColor("#0f1729")      # PeakGuard Dunkelblau
    ACCENT = colors.HexColor("#0da2e7")       # PeakGuard Hellblau
    SUCCESS = colors.HexColor("#28A745")      # Erfolg Grün
    WARNING = colors.HexColor("#FFC107")      # Warnung Gelb
    DANGER = colors.HexColor("#DC3545")       # Kritisch Rot

    DARK = colors.HexColor("#1A1A1A")
    GRAY_DARK = colors.HexColor("#4A4A4A")
    GRAY = colors.HexColor("#6C757D")
    GRAY_LIGHT = colors.HexColor("#E9ECEF")
    GRAY_LIGHTER = colors.HexColor("#F8F9FA")
    WHITE = colors.white

    # Matplotlib Theme (aktualisiert)
    MPL_COLORS = {
        'primary': '#0da2e7',      # Hellblau für Linien
        'primary_dark': '#0f1729',  # Dunkelblau für Text/Akzente
        'accent': '#FF6B35',
        'success': '#28A745',
        'warning': '#FFC107',
        'danger': '#DC3545',
        'grid': '#E9ECEF',
        'text': '#1A1A1A',
        'background': '#FFFFFF'
    }
    
    @staticmethod
    def setup_mpl_theme():
        """Modernes Chart-Theme mit mehr Weißraum und reduzierten Linien (v2)"""
        plt.rcParams.update({
            'figure.facecolor': '#FFFFFF',
            'axes.facecolor': '#FAFBFC',
            'axes.edgecolor': '#E9ECEF',
            'axes.labelcolor': '#1A1A1A',
            'axes.grid': True,
            'grid.color': '#E9ECEF',
            'grid.linewidth': 0.4,  # Dünner
            'grid.alpha': 0.5,       # Dezenter
            'xtick.color': '#1A1A1A',
            'ytick.color': '#1A1A1A',
            'text.color': '#1A1A1A',
            'font.size': DesignTokens.FONT_SIZE_BASE,
            'axes.titlesize': DesignTokens.FONT_SIZE_L,
            'axes.labelsize': DesignTokens.FONT_SIZE_BASE,
            'xtick.labelsize': DesignTokens.FONT_SIZE_S,
            'ytick.labelsize': DesignTokens.FONT_SIZE_S,
            'legend.fontsize': DesignTokens.FONT_SIZE_S,
            'figure.titlesize': DesignTokens.FONT_SIZE_XL,
            'lines.linewidth': 2.5,  # Etwas dicker für bessere Sichtbarkeit
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.8,   # Achsen etwas dicker
        })

PeakGuardDesign.setup_mpl_theme()

GOAL_TO_QUANTILE: Dict[str, float] = {"Bronze": 0.95, "Silber": 0.90, "Gold": 0.85}

NumberLike = Union[int, float, np.number]
TableData = List[List[object]]


@dataclass(frozen=True)
class ReportProfile:
    """Report-Variante: lite, standard, pro"""
    name: str  # "lite", "standard", "pro"
    include_exec_summary: bool = True
    include_scenarios: bool = True
    include_heatmap: bool = True
    include_peak_cluster: bool = True
    include_roadmap: bool = True
    include_top_peaks: bool = True
    include_phase_unbalance: bool = True
    include_blk: bool = True
    include_peak_context: bool = False  # Nur Pro: 12h/3d Fenster
    include_glossary: bool = False       # Standard/Pro
    max_pages_target: int = 10          # Richtgröße


# Vordefinierte Profile
PROFILE_LITE = ReportProfile(
    name="lite",
    include_scenarios=False,
    include_peak_cluster=False,
    include_phase_unbalance=False,
    include_blk=False,
    include_glossary=False,
    max_pages_target=4
)

PROFILE_STANDARD = ReportProfile(
    name="standard",
    include_peak_context=False,
    include_glossary=True,
    max_pages_target=10
)

PROFILE_PRO = ReportProfile(
    name="pro",
    include_peak_context=True,
    include_glossary=True,
    max_pages_target=16
)


def apply_intelligent_triggers(
    profile: ReportProfile,
    savings_eur: float,
    n_peak_events: int,
    blk_available: bool,
    unbalance_available: bool,
    config: Optional['ReportConfig'] = None,
) -> ReportProfile:
    """
    Wendet intelligente Trigger an, um Module automatisch zu aktivieren/deaktivieren

    Trigger-Regeln (PRD):
    - Peak-Kontext: Nur wenn Einsparung > config.peak_context_min_savings ODER Peaks > config.peak_context_min_events
    - BLK: Nur wenn Daten vorhanden UND relevant
    - Phase-Unbalance: Nur wenn Daten vorhanden
    """
    # Default-Config wenn nicht angegeben (wird später definiert)
    if config is None:
        config = ReportConfig()  # Inline-Instantiierung

    # Kopie erstellen, um Original nicht zu verändern
    from dataclasses import replace

    adjusted = replace(profile)

    # Trigger 1: Peak-Kontext nur bei hohem Potenzial (Pro) - konfigurierbar
    if profile.name == "pro" and profile.include_peak_context:
        if savings_eur < config.peak_context_min_savings and n_peak_events < config.peak_context_min_events:
            # Deaktiviere Peak-Kontext, wenn Potenzial gering
            adjusted = replace(adjusted, include_peak_context=False)

    # Trigger 2: BLK nur wenn verfügbar
    if profile.include_blk and not blk_available:
        adjusted = replace(adjusted, include_blk=False)

    # Trigger 3: Phase-Unbalance nur wenn verfügbar
    if profile.include_phase_unbalance and not unbalance_available:
        adjusted = replace(adjusted, include_phase_unbalance=False)

    return adjusted


@dataclass(frozen=True)
class Tariffs:
    switch_hours: float = 2500.0
    work_ct_low: float = 8.27
    demand_eur_kw_a_low: float = 19.93
    work_ct_high: float = 4.25
    demand_eur_kw_a_high: float = 120.43


@dataclass(frozen=True)
class ReportConfig:
    """
    Zentrale Konfiguration für Report-Parameter (v4)
    Kann später aus YAML/JSON geladen werden
    """
    # Schwellenwerte
    peak_context_min_savings: float = 5000.0  # €/a - Mindest-Einsparung für Peak-Kontext
    peak_context_min_events: int = 20          # Mindest-Anzahl Peak-Events
    unbalance_threshold_kw: float = 3.0        # kW - Unwucht-Schwelle
    blk_cosphi_threshold: float = 0.9          # cosϕ-Grenzwert

    # Chart-Einstellungen
    chart_dpi: int = 180
    chart_width_mm: float = 170.0
    chart_height_mm: float = 85.0

    # Top-Peaks
    top_peaks_lite: int = 10
    top_peaks_standard: int = 10
    top_peaks_pro: int = 20

    # Farben (optional überschreibbar)
    custom_primary: Optional[str] = None  # z.B. "#0f1729"
    custom_accent: Optional[str] = None   # z.B. "#0da2e7"


# Standard-Config
DEFAULT_CONFIG = ReportConfig()


@dataclass
class PeakEventsResult:
    n_events: int
    avg_duration_min: float
    max_duration_min: float
    max_shift_kw: float
    top_months: str
    interpretation: str
    events_df: pd.DataFrame
    peak_problem_type: str = "Kurzspitzen"  # Neu für Executive Summary


@dataclass
class PeakContextInfo:
    """Info zu einem einzelnen Peak für Kontext-Analyse (Pro)"""
    timestamp: pd.Timestamp
    power_kw: float
    duration_blocks: int
    diagnosis: str  # "Gleichzeitigkeit", "Dauerlast", "Anfahrvorgang"


@dataclass
class UnbalanceResult:
    available: bool
    share_over: float = 0.0
    max_unbalance_kw: float = 0.0
    dominant_phase: str = "—"
    dominant_phase_name: str = "—"
    dominant_phase_share: float = 0.0
    recommendation: str = ""


@dataclass(frozen=True)
class BlkResult:
    available: bool
    ratio: float = 0.0
    blocks_over: int = 0
    share_over: float = 0.0
    q95: float = 0.0
    assessment: str = ""


@dataclass(frozen=True)
class Recommendation:
    code: str
    category: str
    trigger: str
    action: str
    priority: str


# ============================================================================
# CUSTOM STYLES v2 (modernisiert mit Design-Tokens)
# ============================================================================
def get_custom_styles():
    styles = getSampleStyleSheet()

    # Executive Summary - sehr prominent
    styles.add(ParagraphStyle(
        name='ExecTitle',
        parent=styles['Heading1'],
        fontSize=DesignTokens.FONT_SIZE_HUGE,
        textColor=PeakGuardDesign.PRIMARY,
        spaceAfter=DesignTokens.SPACE_L,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    ))

    # KPI-Kachel Wert (große Zahl)
    styles.add(ParagraphStyle(
        name='KPIValue',
        parent=styles['Normal'],
        fontSize=DesignTokens.FONT_SIZE_XXL,
        textColor=PeakGuardDesign.PRIMARY,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT,
        leading=24
    ))

    # KPI-Kachel Label (klein, oben drüber)
    styles.add(ParagraphStyle(
        name='KPILabel',
        parent=styles['Normal'],
        fontSize=DesignTokens.FONT_SIZE_S,
        textColor=PeakGuardDesign.GRAY_DARK,
        fontName='Helvetica',
        alignment=TA_LEFT,
        spaceBefore=0,
        spaceAfter=DesignTokens.SPACE_XS
    ))

    # Standard Title (v1 Kompatibilität)
    styles.add(ParagraphStyle(
        name='CustomTitle',
        parent=styles['Heading1'],
        fontSize=DesignTokens.FONT_SIZE_XXL,
        textColor=PeakGuardDesign.PRIMARY,
        spaceAfter=DesignTokens.SPACE_L,
        fontName='Helvetica-Bold',
        alignment=TA_LEFT
    ))

    # Section Heading - mehr Weißraum
    styles.add(ParagraphStyle(
        name='CustomHeading2',
        parent=styles['Heading2'],
        fontSize=DesignTokens.FONT_SIZE_L,
        textColor=PeakGuardDesign.DARK,
        spaceBefore=DesignTokens.SPACE_XL,
        spaceAfter=DesignTokens.SPACE_M,
        fontName='Helvetica-Bold'
    ))

    # Subsection
    styles.add(ParagraphStyle(
        name='CustomHeading3',
        parent=styles['Heading3'],
        fontSize=DesignTokens.FONT_SIZE_M,
        textColor=PeakGuardDesign.GRAY_DARK,
        spaceBefore=DesignTokens.SPACE_M,
        spaceAfter=DesignTokens.SPACE_S,
        fontName='Helvetica-Bold'
    ))

    # Body - mehr Leading (Zeilenhöhe)
    styles.add(ParagraphStyle(
        name='CustomBody',
        parent=styles['BodyText'],
        fontSize=DesignTokens.FONT_SIZE_BASE,
        textColor=PeakGuardDesign.DARK,
        leading=14  # mehr Durchschuss
    ))

    # Small Body (für Cards)
    styles.add(ParagraphStyle(
        name='BodySmall',
        parent=styles['BodyText'],
        fontSize=DesignTokens.FONT_SIZE_S,
        textColor=PeakGuardDesign.GRAY_DARK,
        leading=12
    ))

    # Caption (unter Charts)
    styles.add(ParagraphStyle(
        name='Caption',
        parent=styles['Normal'],
        fontSize=DesignTokens.FONT_SIZE_XS,
        textColor=PeakGuardDesign.GRAY,
        alignment=TA_LEFT,
        spaceBefore=DesignTokens.SPACE_XS
    ))

    return styles

# ============================================================================
# CARD COMPONENTS v2 (moderneres Layout)
# ============================================================================
def create_kpi_card(label: str, value: str, subtext: str = "", styles=None) -> Table:
    """Erstellt eine moderne KPI-Kachel (Card-Design)"""
    if styles is None:
        styles = get_custom_styles()

    # Card-Inhalt: Label oben, großer Wert, optional Subtext
    content = [
        [Paragraph(label, styles['KPILabel'])],
        [Paragraph(value, styles['KPIValue'])],
    ]
    if subtext:
        content.append([Paragraph(subtext, styles['BodySmall'])])

    card = Table(
        content,
        colWidths=[DesignTokens.COL_3 - 4*mm],  # Padding berücksichtigen
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), PeakGuardDesign.GRAY_LIGHTER),
            ('BOX', (0, 0), (-1, -1), 1, PeakGuardDesign.GRAY),  # Stärkerer Border v4
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
    prio_color = PeakGuardDesign.SUCCESS  # Default: grün
    if "invest" in priority.lower():
        prio_color = PeakGuardDesign.WARNING
    elif "quick" in priority.lower():
        prio_color = PeakGuardDesign.SUCCESS

    content = [
        [Paragraph(f"<b>{priority}</b>", styles['BodySmall'])],
        [Paragraph(title, styles['CustomHeading3'])],
        [Paragraph(description, styles['BodySmall'])],
    ]

    card = Table(
        content,
        colWidths=[DesignTokens.COL_3 - 4*mm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 1, prio_color),
            ('LINEABOVE', (0, 0), (-1, 0), 3, prio_color),  # Farbiger Top-Border
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('TOPPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('BOTTOMPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
        ])
    )
    return card


def create_scenario_card(name: str, cap_kw: float, peak_after: float, savings: float, util_hours: float, tariff_label: str, styles=None) -> Table:
    """Erstellt eine Szenario-Card (Bronze/Silber/Gold) - v2 modernes Design"""
    if styles is None:
        styles = get_custom_styles()

    # Farbcodierung nach Paket
    color_map = {
        "Bronze": colors.HexColor("#CD7F32"),
        "Silber": colors.HexColor("#C0C0C0"),
        "Gold": colors.HexColor("#FFD700"),
    }
    border_color = color_map.get(name, PeakGuardDesign.GRAY)

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
        colWidths=[DesignTokens.COL_3 - 4*mm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.white),
            ('BOX', (0, 0), (-1, -1), 2, border_color),
            ('LINEABOVE', (0, 0), (-1, 0), 4, border_color),  # Farbiger Top-Border
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('TOPPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
            ('BOTTOMPADDING', (0, 0), (-1, -1), DesignTokens.CARD_PADDING),
        ])
    )
    return card


# ============================================================================
# TABELLEN-HELPER (mit modernem Design)
# ============================================================================
def create_info_table(data: TableData) -> Table:
    """Erstellt Info-Tabelle (Metadaten) mit modernem Design"""
    return Table(
        data,
        colWidths=[45 * mm, 135 * mm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), PeakGuardDesign.GRAY_LIGHTER),
            ('TEXTCOLOR', (0, 0), (0, -1), PeakGuardDesign.GRAY_DARK),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOX', (0, 0), (-1, -1), 0.5, PeakGuardDesign.GRAY),
            ('LINEBELOW', (0, 0), (-1, -2), 0.25, PeakGuardDesign.GRAY_LIGHT),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ])
    )


def create_data_table(data: TableData, highlight_last: bool = False) -> Table:
    """Erstellt Daten-Tabelle mit optionaler Hervorhebung der letzten Zeile"""
    styles_list = [
        ('BACKGROUND', (0, 0), (0, -1), PeakGuardDesign.GRAY_LIGHTER),
        ('TEXTCOLOR', (0, 0), (0, -1), PeakGuardDesign.GRAY_DARK),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('BOX', (0, 0), (-1, -1), 0.5, PeakGuardDesign.GRAY),
        ('LINEBELOW', (0, 0), (-1, -2), 0.25, PeakGuardDesign.GRAY_LIGHT),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),  # GEÄNDERT: TOP -> MIDDLE
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('TOPPADDING', (0, 0), (-1, -1), 8),     # ERHÖHT für besseren Abstand
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]
    
    if highlight_last:
        styles_list.extend([
            ('BACKGROUND', (0, -1), (-1, -1), PeakGuardDesign.SUCCESS),
            ('TEXTCOLOR', (0, -1), (-1, -1), colors.white),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
            ('TOPPADDING', (0, -1), (-1, -1), 10),    # EXTRA Padding für grüne Zeile
            ('BOTTOMPADDING', (0, -1), (-1, -1), 10),
        ])
    
    return Table(
        data,
        colWidths=[75 * mm, 105 * mm],  # MEHR Platz rechts
        style=TableStyle(styles_list)
    )


def create_scenario_table(data: TableData) -> Table:
    """Erstellt Szenario-Vergleichstabelle"""
    return Table(
        data,
        colWidths=[38 * mm, 25 * mm, 32 * mm, 35 * mm, 25 * mm, 25 * mm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PeakGuardDesign.PRIMARY),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 1), (-1, -1), 'RIGHT'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('BOX', (0, 0), (-1, -1), 0.5, PeakGuardDesign.GRAY),
            ('LINEBELOW', (0, 0), (-1, 0), 1, colors.white),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PeakGuardDesign.GRAY_LIGHTER]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ])
    )


def create_recommendations_table(data: TableData) -> Table:
    """Erstellt Handlungsempfehlungen-Tabelle"""
    return Table(
        data,
        colWidths=[45 * mm, 55 * mm, 80 * mm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PeakGuardDesign.DARK),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 0.5, PeakGuardDesign.GRAY),
            ('LINEBELOW', (0, 0), (-1, -1), 0.25, PeakGuardDesign.GRAY_LIGHT),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ])
    )


def create_peaks_table(data: TableData) -> Table:
    """Erstellt Top-Peaks-Tabelle"""
    return Table(
        data,
        colWidths=[10 * mm, 32 * mm, 34 * mm, 34 * mm, 34 * mm],
        style=TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), PeakGuardDesign.ACCENT),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('BOX', (0, 0), (-1, -1), 0.5, PeakGuardDesign.GRAY),
            ('LINEBELOW', (0, 0), (-1, -1), 0.25, PeakGuardDesign.GRAY_LIGHT),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PeakGuardDesign.GRAY_LIGHTER]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('ALIGN', (3, 1), (4, -1), 'RIGHT'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ])
    )


# ============================================================================
# CHART HELPER
# ============================================================================
def add_chart_with_caption(img_path: Path, caption: str, styles, width: float = 170*mm) -> List[Flowable]:
    """Fügt Chart + Caption als Flowable-Liste hinzu"""
    elements: List[Flowable] = []
    elements.append(Image(str(img_path), width=width, height=width*0.5))
    if caption:
        elements.append(Spacer(1, DesignTokens.SPACE_XS))
        elements.append(Paragraph(caption, styles['Caption']))
    return elements


# ============================================================================
# VISUALISIERUNGEN (Optimiert mit PeakGuard-Design v2)
# ============================================================================
def make_timeseries_plot(df_15: pd.DataFrame, cap_kw: float) -> Path:
    """Zeitreihen-Plot mit Cap-Linie"""
    tmp = Path(_tempfile_path("timeseries.png"))
    idx = cast(pd.DatetimeIndex, df_15.index)
    y = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')
    
    # Hauptlinie
    ax.plot(idx, y, color=PeakGuardDesign.MPL_COLORS['primary'], linewidth=2, label='Leistung (15-min)')
    
    # Cap-Linie
    ax.axhline(y=cap_kw, color=PeakGuardDesign.MPL_COLORS['danger'], 
               linestyle='--', linewidth=2, label=f'Cap ({cap_kw:.1f} kW)', alpha=0.8)
    
    # Überschreitungen farbig markieren
    over_mask = y > cap_kw
    if over_mask.any():
        ax.fill_between(idx, y, cap_kw, where=over_mask.values, #type: ignore
                    color=PeakGuardDesign.MPL_COLORS['danger'], alpha=0.15)
    
    ax.set_ylabel('Leistung (kW)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Zeitraum', fontweight='bold', fontsize=11)
    ax.set_title('Lastgang-Verlauf mit Peak-Shaving Cap', fontweight='bold', fontsize=12, pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_duration_curve(df_15: pd.DataFrame, cap_kw: float) -> Path:
    """Jahresdauerlinie mit Percentil-Markierungen"""
    tmp = Path(_tempfile_path("duration.png"))
    s = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce").dropna().sort_values(ascending=False).to_numpy(dtype=float)
    n = int(len(s))
    
    if n == 0:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.text(0.5, 0.5, "Keine Daten verfügbar", ha="center", va="center", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(tmp, dpi=180, bbox_inches='tight')
        plt.close()
        return tmp

    x = 100.0 * (np.arange(1, n + 1) / float(n))

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')
    
    # Hauptkurve
    ax.plot(x, s, color=PeakGuardDesign.MPL_COLORS['primary'], linewidth=2.5, label='Leistung')
    
    # Cap-Linie (horizontal)
    ax.axhline(y=cap_kw, color=PeakGuardDesign.MPL_COLORS['danger'], 
               linestyle='--', linewidth=2.5, label=f'Cap ({cap_kw:.1f} kW)', alpha=0.9)
    
    # Percentil-WERTE berechnen (NICHT x-Position!)
    p95_kw = float(np.percentile(s, 95))  # 95. Perzentil der LEISTUNG
    p90_kw = float(np.percentile(s, 90))
    p85_kw = float(np.percentile(s, 85))
    
    # Horizontale Linien für Bronze/Silber/Gold
    ax.axhline(p95_kw, color='#CD7F32', linestyle=':', linewidth=2, 
               alpha=0.8, label=f'P95 Bronze ({p95_kw:.1f} kW)')
    ax.axhline(p90_kw, color='#C0C0C0', linestyle=':', linewidth=2, 
               alpha=0.8, label=f'P90 Silber ({p90_kw:.1f} kW)')
    ax.axhline(p85_kw, color='#FFD700', linestyle=':', linewidth=2, 
               alpha=0.8, label=f'P85 Gold ({p85_kw:.1f} kW)')
    
    ax.set_ylabel('Leistung (kW)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Zeitanteil (%)', fontweight='bold', fontsize=11)
    ax.set_title('Jahresdauerlinie (sortiert nach Leistung)', fontweight='bold', fontsize=12, pad=15)
    ax.set_xlim(0, 100)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp

def make_heatmap(df_15: pd.DataFrame) -> Path:
    """Heatmap Wochentag x Stunde - AMPEL-FARBEN"""
    tmp = Path(_tempfile_path("heatmap.png"))
    idx = cast(pd.DatetimeIndex, df_15.index)
    ts_np = idx.to_numpy(dtype="datetime64[ns]")

    def _weekday(x: np.datetime64) -> int:
        return int(pd.Timestamp(x).weekday())

    def _hour(x: np.datetime64) -> int:
        return int(pd.Timestamp(x).hour)

    weekday = np.array([_weekday(x) for x in ts_np], dtype=int)
    hour = np.array([_hour(x) for x in ts_np], dtype=int)

    p = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce").to_numpy(dtype=float)
    dfh = pd.DataFrame({"weekday": weekday, "hour": hour, "p": p})
    pivot = dfh.pivot_table(index="weekday", columns="hour", values="p", aggfunc="mean")
    pivot = pivot.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=list(range(24)))

    vals = pivot.to_numpy(dtype=float)
    
    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')
    
    # AMPEL-FARBSKALA: Grün (niedrig) -> Gelb (mittel) -> Rot (hoch)
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Circle  # ← KORREKTER IMPORT!
    
    # Farbübergänge: Grün -> Gelbgrün -> Gelb -> Orange -> Rot
    colors_list = [
        '#2ECC71',  # Grün (niedrig)
        '#A8E063',  # Hellgrün
        '#F4D03F',  # Gelb (mittel)
        '#F39C12',  # Orange
        '#E74C3C'   # Rot (hoch)
    ]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('ampel', colors_list, N=n_bins)
    
    im = ax.imshow(vals, aspect='auto', cmap=cmap, interpolation='nearest')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ø Leistung (kW)', rotation=270, labelpad=20, fontweight='bold', fontsize=10)
    
    # Werte in Zellen - IMMER SCHWARZE SCHRIFT mit weißem Hintergrund für Lesbarkeit
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                # Weißer Hintergrund-Kreis für bessere Lesbarkeit
                circle = Circle((j, i), 0.35, color='white', alpha=0.8, zorder=10)
                ax.add_patch(circle)
                
                # Schwarzer Text
                ax.text(j, i, f"{vals[i, j]:.0f}", 
                       ha="center", va="center", 
                       color='black', fontsize=9, fontweight='bold', zorder=11)

    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"], fontsize=10)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=9)
    ax.set_xlabel("Stunde", fontweight='bold', fontsize=11)
    ax.set_ylabel("Wochentag", fontweight='bold', fontsize=11)
    ax.set_title("Lastprofil-Heatmap (Ø Leistung je Wochentag/Stunde)", 
                 fontweight='bold', fontsize=12, pad=15)
    
    # Grid für bessere Orientierung
    ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)
    
    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp

def make_monthly_peaks_bar(df_15: pd.DataFrame, cap_kw: float) -> Path:
    """Balkendiagramm: Peaks pro Monat (v2 NEU)"""
    tmp = Path(_tempfile_path("monthly_peaks.png"))
    idx = cast(pd.DatetimeIndex, df_15.index)
    p = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce")

    # Zähle Überschreitungen pro Monat
    over = (p > cap_kw)
    monthly_data = pd.DataFrame({
        'year_month': idx.to_period('M'),
        'over': over.astype(int),
        'peak_kw': p
    })

    # Aggregiere: Count Überschreitungen + Max Peak pro Monat
    monthly_agg = monthly_data.groupby('year_month').agg({
        'over': 'sum',
        'peak_kw': 'max'
    }).reset_index()

    monthly_agg['month_str'] = monthly_agg['year_month'].astype(str)

    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor='white')

    # Balken: Anzahl Überschreitungen
    bars = ax.bar(
        range(len(monthly_agg)),
        monthly_agg['over'],
        color=PeakGuardDesign.MPL_COLORS['danger'],
        alpha=0.7,
        label='Anzahl 15-min-Blöcke > Cap'
    )

    # Zweite Y-Achse: Max Peak pro Monat
    ax2 = ax.twinx()
    ax2.plot(
        range(len(monthly_agg)),
        monthly_agg['peak_kw'],
        color=PeakGuardDesign.MPL_COLORS['primary'],
        marker='o',
        linewidth=2.5,
        markersize=6,
        label='Max. Peak des Monats'
    )

    # Cap-Linie
    ax2.axhline(cap_kw, color=PeakGuardDesign.MPL_COLORS['warning'],
                linestyle='--', linewidth=2, alpha=0.8, label=f'Cap ({cap_kw:.1f} kW)')

    ax.set_xlabel('Monat', fontweight='bold', fontsize=11)
    ax.set_ylabel('Anzahl Überschreitungen', fontweight='bold', fontsize=10, color=PeakGuardDesign.MPL_COLORS['danger'])
    ax2.set_ylabel('Leistung (kW)', fontweight='bold', fontsize=10, color=PeakGuardDesign.MPL_COLORS['primary'])

    ax.set_xticks(range(len(monthly_agg)))
    ax.set_xticklabels(monthly_agg['month_str'], rotation=45, ha='right')

    ax.tick_params(axis='y', labelcolor=PeakGuardDesign.MPL_COLORS['danger'])
    ax2.tick_params(axis='y', labelcolor=PeakGuardDesign.MPL_COLORS['primary'])

    ax.set_title('Peaks pro Monat: Überschreitungen & Max-Werte', fontweight='bold', fontsize=12, pad=15)

    # Legende kombiniert
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_events_scatter(mod1: PeakEventsResult) -> Path:
    """Scatter-Plot: Peak-Ereignisse Dauer vs. Verschiebe-Leistung"""
    tmp = Path(_tempfile_path("events.png"))
    ev = mod1.events_df

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')
    
    if ev.empty:
        ax.text(0.5, 0.5, "Keine Peak-Ereignisse über Cap erkannt", 
               ha="center", va="center", fontsize=14, color=PeakGuardDesign.MPL_COLORS['text'])
        ax.axis("off")
    else:
        # Scatter mit Größe basierend auf max_shift_kw
        scatter = ax.scatter(
            ev["duration_min"], 
            ev["max_shift_kw"],
            s=ev["max_shift_kw"]*10,
            c=ev["duration_min"],
            cmap='YlOrRd',
            alpha=0.6,
            edgecolors=PeakGuardDesign.MPL_COLORS['text'],
            linewidth=0.5
        )
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Dauer (min)', rotation=270, labelpad=20, fontweight='bold')
        
        ax.set_xlabel('Dauer des Peak-Ereignisses (min)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Max. benötigte Verschiebe-Leistung (kW)', fontweight='bold', fontsize=11)
        ax.set_title('Peak-Ereignisse: Dauer vs. Verschiebe-Leistung', fontweight='bold', fontsize=12, pad=15)
        
        # Referenzlinien
        ax.axvline(15, color=PeakGuardDesign.MPL_COLORS['success'], 
                  linestyle='--', linewidth=1, alpha=0.5, label='Kurze Peaks (≤15 min)')
        ax.axvline(60, color=PeakGuardDesign.MPL_COLORS['warning'], 
                  linestyle='--', linewidth=1, alpha=0.5, label='Lange Peaks (≥60 min)')
        ax.legend(loc='upper right', framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_peak_context_plot(df_15: pd.DataFrame, peak_timestamp: pd.Timestamp, window_hours: int, cap_kw: float) -> Path:
    """
    Peak-Kontext-Plot: Zeigt Umfeld eines Peaks (12h oder 3d Fenster)
    Nur für Pro-Profile
    """
    tmp = Path(_tempfile_path(f"peak_context_{window_hours}h.png"))

    # Zeitfenster berechnen
    half_window = pd.Timedelta(hours=window_hours / 2)
    start = peak_timestamp - half_window
    end = peak_timestamp + half_window

    # Daten filtern
    mask = (df_15.index >= start) & (df_15.index <= end)
    df_window = df_15[mask].copy()

    if df_window.empty:
        # Fallback: leeres Chart
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
        ax.text(0.5, 0.5, "Nicht genügend Daten im Zeitfenster",
               ha="center", va="center", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(tmp, dpi=180, bbox_inches='tight')
        plt.close()
        return tmp

    idx = cast(pd.DatetimeIndex, df_window.index)
    y = pd.to_numeric(cast(pd.Series, df_window["p_kw"]), errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')

    # Hauptlinie
    ax.plot(idx, y, color=PeakGuardDesign.MPL_COLORS['primary'],
           linewidth=2.5, label='Leistung (15-min)', zorder=2)

    # Cap-Linie
    ax.axhline(y=cap_kw, color=PeakGuardDesign.MPL_COLORS['danger'],
               linestyle='--', linewidth=2, label=f'Cap ({cap_kw:.1f} kW)', alpha=0.8, zorder=1)

    # Peak-Marker (vertikale Linie)
    ax.axvline(x=peak_timestamp, color=PeakGuardDesign.MPL_COLORS['warning'],
              linestyle='-', linewidth=3, alpha=0.7, label='Peak-Zeitpunkt', zorder=3)

    # Überschreitungen füllen
    over_mask = y > cap_kw
    if over_mask.any():
        ax.fill_between(idx, y, cap_kw, where=over_mask.values, #type: ignore
                    color=PeakGuardDesign.MPL_COLORS['danger'], alpha=0.15, zorder=0)

    ax.set_ylabel('Leistung (kW)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Zeit', fontweight='bold', fontsize=11)

    window_label = f"{window_hours}h" if window_hours < 48 else f"{window_hours//24}d"
    ax.set_title(f'Peak-Kontext ({window_label}-Fenster um {peak_timestamp:%d.%m.%Y %H:%M})',
                 fontweight='bold', fontsize=12, pad=15)

    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_blk_plot(df_15: pd.DataFrame) -> Path:
    """Blindleistungs-Plot"""
    tmp = Path(_tempfile_path("blk.png"))

    q = pd.to_numeric(cast(pd.Series, df_15.get("q_kvar", pd.Series(index=df_15.index, dtype=float))), errors="coerce")
    qlim = pd.to_numeric(cast(pd.Series, df_15.get("q_limit", pd.Series(index=df_15.index, dtype=float))), errors="coerce")

    idx = cast(pd.DatetimeIndex, df_15.index)

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')
    
    ax.plot(idx, q, color=PeakGuardDesign.MPL_COLORS['primary'], 
           linewidth=2, label='Q (kvar)', alpha=0.8)
    ax.plot(idx, qlim, color=PeakGuardDesign.MPL_COLORS['danger'], 
           linestyle='--', linewidth=2, label='Q-Limit (cosϕ=0,9)', alpha=0.8)
    
    # Überschreitungen markieren
    over_mask = q > qlim
    if over_mask.any():
        ax.fill_between(idx, q, qlim, where=over_mask.values, #type: ignore
                    color=PeakGuardDesign.MPL_COLORS['danger'], alpha=0.15)
    
    ax.set_ylabel('Blindleistung (kvar)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Zeitraum', fontweight='bold', fontsize=11)
    ax.set_title('Blindleistungs-Verlauf mit cosϕ-Grenzwert', fontweight='bold', fontsize=12, pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp

# ============================================================================
# EXECUTIVE SUMMARY BUILDER (v2 - Seite 1)
# ============================================================================
def build_glossary(styles) -> List[Flowable]:
    """
    Erstellt 1/2-Seite Glossar 'So lesen Sie den Report'
    Für Standard/Pro-Profile
    """
    story: List[Flowable] = []

    story.append(Paragraph("So lesen Sie den Report", styles["CustomHeading2"]))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    glossary_items = [
        ("15-min-Mittelwert", "Leistung gemittelt über 15 Minuten. Basis für Netzentgelte und Peak-Shaving. Nicht zu verwechseln mit Spitzenlast (1-Sekunden-Peak)."),
        ("Cap / Peak-Shaving", "Zielvorgabe: Leistung soll diesen Wert nicht überschreiten. Durch Lastverschiebung oder -abwurf realisierbar."),
        ("Benutzungsstunden (h/a)", "Energie geteilt durch Peak. Hohe Werte (>2500h) → günstiger Tarif möglich. Niedrig (< 2500h) → teurer Leistungspreis."),
        ("Verschiebbare kWh", "Energie oberhalb des Caps, die durch Lastmanagement in andere Zeiten verschoben werden müsste."),
        ("P95 / P90 / P85", "95./90./85. Perzentil der Leistung. Bedeutet: 95%/90%/85% der Zeit liegt die Leistung darunter (Bronze/Silber/Gold)."),
        ("Peak-Problemtyp", "Kurzspitzen: viele kurze Überschreitungen (≤15 min) → Gleichzeitigkeit vermeiden. Langspitzen: lange Überschreitungen (≥60 min) → Grundlast/Prozess optimieren."),
    ]

    for term, explanation in glossary_items:
        story.append(Paragraph(f"<b>{term}:</b> {explanation}", styles['BodySmall']))
        story.append(Spacer(1, DesignTokens.SPACE_S))

    return story


def build_executive_summary(
    period_str: str,
    data_quality_str: str,
    peak_15_kw: float,
    peak_timestamp: pd.Timestamp,
    savings_eur: float,
    problem_type: str,
    top_3_actions: List[Recommendation],
    styles,
) -> List[Flowable]:
    """
    Erstellt die Executive Summary (Seite 1)
    - 3 KPI-Kacheln: Peak, Einsparung, Problem-Typ
    - Top-3-Hebel als Action-Cards
    - Data-Quality-Box
    """
    story: List[Flowable] = []

    # === HEADER ===
    story.append(Paragraph("Executive Summary", styles["ExecTitle"]))
    story.append(Spacer(1, DesignTokens.SPACE_S))

    # === ZEITRAUM & DATENQUALITÄT (kompakt) ===
    info_text = f"<b>Zeitraum:</b> {period_str} | <b>Datenqualität:</b> {data_quality_str}"
    story.append(Paragraph(info_text, styles['BodySmall']))
    story.append(Spacer(1, DesignTokens.SPACE_L))

    # === 3 KPI-KACHELN (nebeneinander) ===
    peak_time_str = peak_timestamp.strftime("%d.%m.%Y %H:%M") if peak_timestamp else "—"

    kpi_cards = Table(
        [[
            create_kpi_card(
                "Höchster 15-min Peak",
                fmt_num(peak_15_kw, 1, "kW"),
                f"am {peak_time_str}",
                styles
            ),
            create_kpi_card(
                "Einsparpotenzial",
                fmt_num(savings_eur, 0, "€/Jahr"),
                "bei optimalem Szenario",
                styles
            ),
            create_kpi_card(
                "Peak-Problemtyp",
                problem_type,
                "dominantes Muster",
                styles
            ),
        ]],
        colWidths=[DesignTokens.COL_3, DesignTokens.COL_3, DesignTokens.COL_3],
        style=TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.GUTTER),
        ])
    )
    story.append(kpi_cards)
    story.append(Spacer(1, DesignTokens.SPACE_XL))

    # === TOP-3-HEBEL ===
    story.append(Paragraph("Diese 3 Hebel zuerst", styles["CustomHeading2"]))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    # Action-Cards nebeneinander (falls 3 vorhanden)
    if len(top_3_actions) >= 3:
        action_row = [[
            create_action_card(
                top_3_actions[0].priority or "Maßnahme",
                top_3_actions[0].category,
                top_3_actions[0].action[:120] + "..." if len(top_3_actions[0].action) > 120 else top_3_actions[0].action,
                styles
            ),
            create_action_card(
                top_3_actions[1].priority or "Maßnahme",
                top_3_actions[1].category,
                top_3_actions[1].action[:120] + "..." if len(top_3_actions[1].action) > 120 else top_3_actions[1].action,
                styles
            ),
            create_action_card(
                top_3_actions[2].priority or "Maßnahme",
                top_3_actions[2].category,
                top_3_actions[2].action[:120] + "..." if len(top_3_actions[2].action) > 120 else top_3_actions[2].action,
                styles
            ),
        ]]

        actions_table = Table(
            action_row,
            colWidths=[DesignTokens.COL_3, DesignTokens.COL_3, DesignTokens.COL_3],
            style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.GUTTER),
            ])
        )
        story.append(actions_table)
    else:
        # Fallback: Liste statt Cards
        for i, action in enumerate(top_3_actions[:3]):
            story.append(Paragraph(f"<b>{i+1}. {action.category}</b>", styles['CustomHeading3']))
            story.append(Paragraph(action.action, styles['BodySmall']))
            story.append(Spacer(1, DesignTokens.SPACE_S))

    return story


# ============================================================================
# HAUPTFUNKTION - PDF Report Builder
# ============================================================================

# ============================================================================
# HEADER & FOOTER
# ============================================================================
def add_page_template(canvas, doc, site_name: str):
    """Fügt Header/Footer zu jeder Seite hinzu"""
    canvas.saveState()
    
    # Footer
    footer_y = 15 * mm
    page_num = canvas.getPageNumber()
    
    # Footer Links: Datum
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(PeakGuardDesign.GRAY)
    canvas.drawString(18 * mm, footer_y, f"{pd.Timestamp.now():%d.%m.%Y}")
    
    # Footer Mitte: PeakGuard + Kunde
    canvas.setFont('Helvetica-Bold', 8)
    canvas.setFillColor(PeakGuardDesign.PRIMARY)
    footer_text = f"PeakGuard Report"
    if site_name:
        footer_text += f" – {site_name}"
    canvas.drawCentredString(A4[0] / 2, footer_y, footer_text)
    
    # Footer Rechts: Seitenzahl
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(PeakGuardDesign.GRAY)
    canvas.drawRightString(A4[0] - 18 * mm, footer_y, f"Seite {page_num}")
    
    # Disclaimer (nur auf Seite 1)
    if page_num == 1:
        disclaimer_y = footer_y - 8
        canvas.setFont('Helvetica', 6)
        canvas.setFillColor(PeakGuardDesign.GRAY)
        disclaimer = "Alle Angaben ohne Gewähr. Berechnungen basieren auf historischen Daten und stellen keine Garantie für zukünftige Einsparungen dar."
        canvas.drawCentredString(A4[0] / 2, disclaimer_y, disclaimer)
    
    canvas.restoreState()

def build_pdf_report(
    df: pd.DataFrame,
    out_path: Path,
    timestamp_col: str,
    power_col: Optional[str],
    power_cols: Optional[List[str]] = None,
    power_unit: Optional[str] = "Auto",
    pf_cols: Optional[List[str]] = None,
    source_name: str = "",
    site_name: str = "",
    data_quality: str = "OK",
    meter_type: str = "RLM",
    reduction_goal: str = "Bronze",
    manual_value: str = "",
    manual_cap_kw: Optional[float] = None,
    tariffs: Optional[Tariffs] = None,
    include_reactive: bool = True,
    input_resolution_minutes: Optional[int] = None,
    demand_interval_minutes: int = 15,
    profile: Optional[ReportProfile] = None,  # NEU: Profil-Support
) -> None:
    tariffs = tariffs or Tariffs()
    profile = profile or PROFILE_STANDARD  # Default: Standard-Profil
    d0 = df.copy()

    # --- Parse timestamps ---
    d0[timestamp_col] = pd.to_datetime(d0[timestamp_col], errors="coerce")
    d0 = d0.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    if d0.empty:
        raise ValueError("Keine gültigen Zeitstempel nach Parsing vorhanden.")

    # --- Build raw power series in kW ---
    raw = _prepare_raw_power_and_optional_phases(
        d0=d0,
        timestamp_col=timestamp_col,
        power_col=power_col,
        power_cols=power_cols,
        power_unit=power_unit or "Auto",
        pf_cols=pf_cols,
    )
    df_raw = raw["df"]
    if df_raw.empty:
        raise ValueError("Keine gültigen Leistungswerte nach Parsing vorhanden.")

    idx_raw = cast(pd.DatetimeIndex, df_raw.index)

    inferred_res = _infer_resolution_minutes(idx_raw) if input_resolution_minutes is None else input_resolution_minutes
    missing_quote = _missing_quote(idx_raw, inferred_res)

    # --- Aggregate to demand interval (15min) ---
    df_15 = _aggregate_to_interval(df_raw, minutes=demand_interval_minutes)
    if df_15.empty:
        raise ValueError("Aggregation auf 15-Minuten ergab keine Daten.")

    idx_15 = cast(pd.DatetimeIndex, df_15.index)

    # --- Peak info ---
    peak_15_kw = float(pd.to_numeric(df_15["p_kw"], errors="coerce").max())
    peak_1m_kw: Optional[float] = None
    if inferred_res is not None and inferred_res < demand_interval_minutes:
        peak_1m_kw = float(pd.to_numeric(df_raw["p_kw"], errors="coerce").max())

    # --- Energy on 15-min mean power ---
    block_h = float(demand_interval_minutes) / 60.0
    df_15["kwh"] = pd.to_numeric(df_15["p_kw"], errors="coerce").clip(lower=0) * block_h
    energy_kwh = float(pd.to_numeric(df_15["kwh"], errors="coerce").sum())

    # --- Duration ---
    duration_hours = float((idx_15.max() - idx_15.min()).total_seconds() / 3600.0)
    duration_hours = max(duration_hours, block_h)

    # --- Annualization ---
    annual_energy_kwh = energy_kwh * (8760.0 / duration_hours)

    # --- Utilization hours ---
    util_hours_before = (annual_energy_kwh / peak_15_kw) if peak_15_kw > 0 else 0.0
    work_ct_before, demand_eur_before, tariff_label_before = _tariff_for_util_hours(tariffs, util_hours_before)
    cost_before = annual_energy_kwh * (work_ct_before / 100.0) + peak_15_kw * demand_eur_before

    # --- Selected cap scenario ---
    cap_kw_sel, cap_label_sel = compute_cap(df_15["p_kw"], reduction_goal, manual_cap_kw, manual_value)
    scenario_sel = _compute_scenario(
        name="Ausgewählt",
        cap_kw=cap_kw_sel,
        cap_label=cap_label_sel,
        annual_energy_kwh=annual_energy_kwh,
        peak_before_kw=peak_15_kw,
        tariffs=tariffs,
        df_15=df_15,
        block_h=block_h,
    )

    # --- Module 1: Peak events ---
    mod1 = compute_peak_events(df_15, cap_kw_sel, interval_minutes=demand_interval_minutes)

    # --- Module 2: Unbalance ---
    mod2 = compute_unbalance_module(df_15, threshold_kw=3.0)

    # --- Optional BLK ---
    blk = compute_blk_metrics_15min(df_15) if include_reactive else BlkResult(available=False)

    # --- Package scenarios ---
    pkg_scenarios: List[_Scenario] = []
    for g in ["Bronze", "Silber", "Gold"]:
        cap_g, cap_lbl = compute_cap(df_15["p_kw"], g, manual_cap_kw=None, manual_value="")
        pkg_scenarios.append(
            _compute_scenario(
                name=g,
                cap_kw=cap_g,
                cap_label=cap_lbl,
                annual_energy_kwh=annual_energy_kwh,
                peak_before_kw=peak_15_kw,
                tariffs=tariffs,
                df_15=df_15,
                block_h=block_h,
            )
        )

    # --- Recommendations ---
    recs = build_recommendations_v3(
        mod1=mod1,
        mod2=mod2,
        blk=blk,
        util_hours_before=util_hours_before,
        util_hours_after=scenario_sel.util_hours_after,
        p_verschiebbar_kw=_estimate_p_verschiebbar_kw(mod1=mod1, scenario=scenario_sel),
        p_gesamt_kw=_estimate_p_gesamt_kw(peak_15_kw),
        tariffs=tariffs,
    )

    # --- INTELLIGENTE TRIGGER: Profil anpassen basierend auf Daten (nach Berechnungen) ---
    profile = apply_intelligent_triggers(
        profile=profile,
        savings_eur=scenario_sel.savings_eur,
        n_peak_events=mod1.n_events,
        blk_available=blk.available,
        unbalance_available=mod2.available,
    )

    # --- Charts (profilabhängig) ---
    figs: List[Path] = []

    if profile.include_heatmap:
        figs.append(make_timeseries_plot(df_15, cap_kw_sel))
        figs.append(make_duration_curve(df_15, cap_kw_sel))
        figs.append(make_heatmap(df_15))
        figs.append(make_monthly_peaks_bar(df_15, cap_kw_sel))  # NEU: Monatsbalken

    if profile.include_peak_cluster:
        figs.append(make_events_scatter(mod1))

    if blk.available and profile.include_blk:
        figs.append(make_blk_plot(df_15))

    # --- PDF Generation ---
    # PDF Metadaten setzen
    pdf_title = f"PeakGuard Report - {site_name if site_name else 'Bericht'} - {pd.Timestamp.now():%d.%m.%Y}"
    pdf_author = "PeakGuard"
    pdf_subject = f"Lastgang-Analyse für {site_name if site_name else 'Kunde'}"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=pdf_title,
        author=pdf_author,
        subject=pdf_subject,
    )
    
    styles = get_custom_styles()
    story: List[Flowable] = []

    def p_wrap(text: str, style_name: str = 'CustomBody') -> Paragraph:
        safe = (text or "—").replace("\n", "<br/>")
        return Paragraph(safe, styles[style_name])

    # Zeitraum-String (wird mehrfach gebraucht)
    period_str = (
        f"{pd.Timestamp(idx_15.min()):%d.%m.%Y %H:%M} – "
        f"{pd.Timestamp(idx_15.max()):%d.%m.%Y %H:%M} ({duration_hours:.1f} h)"
    )

    # === EXECUTIVE SUMMARY (v2 - Seite 1) ===
    if profile.include_exec_summary:
        # Finde Peak-Zeitpunkt
        peak_idx = pd.to_numeric(df_15["p_kw"], errors="coerce").idxmax()
        peak_timestamp = pd.Timestamp(peak_idx) if pd.notna(peak_idx) else pd.Timestamp.now()

        # Data Quality String
        data_quality_str = f"{data_quality} | Abdeckung: {(1-missing_quote)*100:.1f}%"

        exec_summary_story = build_executive_summary(
            period_str=period_str,
            data_quality_str=data_quality_str,
            peak_15_kw=peak_15_kw,
            peak_timestamp=peak_timestamp,
            savings_eur=scenario_sel.savings_eur,
            problem_type=mod1.peak_problem_type,
            top_3_actions=recs[:3],  # Top-3 Empfehlungen
            styles=styles,
        )
        story.extend(exec_summary_story)
        story.append(PageBreak())

    # === HEADER / TITLE (nur wenn KEIN Exec Summary) ===
    if not profile.include_exec_summary:
        story.append(Paragraph("PeakGuard – Lastgang- & Lastspitzen-Report", styles["CustomTitle"]))
        story.append(Paragraph(f"<font color='#{PeakGuardDesign.GRAY.hexval()[2:]}'>Version 4.0 | Erstellt: {pd.Timestamp.now():%d.%m.%Y %H:%M}</font>", styles["CustomBody"]))
        story.append(Spacer(1, DesignTokens.SPACE_M))

    # === METADATEN TABLE ===
    story.append(create_info_table([
        ["Quelle", source_name],
        ["Standort", site_name or "—"],
        ["Zählertyp", meter_type],
        ["Datenqualität", data_quality],
        ["Zeitraum", period_str],
        [
            "Validierung",
            f"Input ∆t~{inferred_res if inferred_res is not None else '—'} min | "
            f"Missing-Quote: {missing_quote:.1%} | Demand-Basis: {demand_interval_minutes} min",
        ],
    ]))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    # === KERNKENNZAHLEN ===
    story.append(Paragraph("Kernkennzahlen (Lastgang-Logik: 15-min Mittelwerte)", styles["CustomHeading2"]))

    util_before_str = f"{fmt_num(util_hours_before, 0, 'h/a')} ({tariff_label_before})"
    rows_kpi: TableData = [
        ["Energie (Messzeitraum)", fmt_num(energy_kwh, 0, "kWh")],
        ["Energie (hochgerechnet)", fmt_num(annual_energy_kwh, 0, "kWh/a")],
        ["Max. Leistung (Peak, 15-min)", fmt_num(peak_15_kw, 1, "kW")],
    ]
    if peak_1m_kw is not None:
        rows_kpi.append(["Max. Leistung (Peak, 1-min Info)", fmt_num(peak_1m_kw, 1, "kW")])

    rows_kpi.extend([
        ["Benutzungsdauer (hochgerechnet)", util_before_str],
        ["Arbeitspreis (akt. Tarif)", f"{work_ct_before:.2f} ct/kWh".replace(".", ",")],
        ["Leistungspreis (akt. Tarif)", f"{demand_eur_before:.2f} €/kW/a".replace(".", ",")],
        ["Kosten (Ist, hochgerechnet)", fmt_num(cost_before, 0, "€/a")],
    ])

    story.append(create_data_table(rows_kpi))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    # === PEAK SHAVING ZIEL ===
    story.append(Paragraph("Peak-Shaving Ziel (Cap) & rechnerische Wirkung", styles["CustomHeading2"]))

    util_after_str = f"{fmt_num(scenario_sel.util_hours_after, 0, 'h/a')} ({scenario_sel.tariff_label_after})"
    tariff_switch_note = (
        "✓ Tarifwechsel möglich (nach Peak-Shaving > Schwelle)"
        if scenario_sel.tariff_switched
        else "○ Kein Tarifwechsel (weiterhin unter/über Schwelle)"
    )

    rows_sel: TableData = [
        ["Ziel", scenario_sel.cap_label],
        ["Cap", fmt_num(scenario_sel.cap_kw, 1, "kW")],
        ["Peak vorher (15-min)", fmt_num(peak_15_kw, 1, "kW")],
        ["Peak nachher (Cap)", fmt_num(scenario_sel.peak_after_kw, 1, "kW")],
        ["Neue Benutzungsdauer", util_after_str],
        ["Tarifwechsel-Check", tariff_switch_note],
        ["15-min Blöcke über Cap", f"{scenario_sel.blocks_over_cap} ({scenario_sel.share_over_cap:.1%})"],
        ["Energie über Cap (Indikator)", fmt_num(scenario_sel.kwh_to_shift, 1, "kWh")],
        ["Kosten nachher", fmt_num(scenario_sel.cost_after, 0, "€/a")],
        ["Einsparung (rechnerisch)", fmt_num(scenario_sel.savings_eur, 0, "€/a")],
    ]

    story.append(create_data_table(rows_sel, highlight_last=True))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    # === WEITERE SZENARIEN (v2: moderne Cards) ===
    if profile.include_scenarios and len(pkg_scenarios) >= 3:
        story.append(Paragraph("Weitere Szenarien (rechnerisch, inkl. Tarifwechsel)", styles["CustomHeading3"]))
        story.append(Spacer(1, DesignTokens.SPACE_M))

        # Szenarien als 3er-Card-Grid
        scenario_row = [[
            create_scenario_card(
                name=pkg_scenarios[0].name,
                cap_kw=pkg_scenarios[0].cap_kw,
                peak_after=pkg_scenarios[0].peak_after_kw,
                savings=pkg_scenarios[0].savings_eur,
                util_hours=pkg_scenarios[0].util_hours_after,
                tariff_label=pkg_scenarios[0].tariff_label_after,
                styles=styles
            ),
            create_scenario_card(
                name=pkg_scenarios[1].name,
                cap_kw=pkg_scenarios[1].cap_kw,
                peak_after=pkg_scenarios[1].peak_after_kw,
                savings=pkg_scenarios[1].savings_eur,
                util_hours=pkg_scenarios[1].util_hours_after,
                tariff_label=pkg_scenarios[1].tariff_label_after,
                styles=styles
            ),
            create_scenario_card(
                name=pkg_scenarios[2].name,
                cap_kw=pkg_scenarios[2].cap_kw,
                peak_after=pkg_scenarios[2].peak_after_kw,
                savings=pkg_scenarios[2].savings_eur,
                util_hours=pkg_scenarios[2].util_hours_after,
                tariff_label=pkg_scenarios[2].tariff_label_after,
                styles=styles
            ),
        ]]

        scenarios_table = Table(
            scenario_row,
            colWidths=[DesignTokens.COL_3, DesignTokens.COL_3, DesignTokens.COL_3],
            style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.GUTTER),
            ])
        )
        story.append(scenarios_table)
        story.append(Spacer(1, DesignTokens.SPACE_L))

    # === MODULE 1: Peak-Cluster (nur wenn Profil es vorsieht) ===
    if profile.include_peak_cluster:
        story.append(Paragraph("Peak-Cluster & Ursachenmuster", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_S))

        # Kompaktere Darstellung
        story.append(create_data_table([
            ["Peak-Ereignisse gesamt", str(mod1.n_events)],
            ["Ø Dauer pro Ereignis", f"{mod1.avg_duration_min:.1f} min".replace(".", ",")],
            ["Max. Verschiebe-Leistung", f"{mod1.max_shift_kw:.1f} kW".replace(".", ",")],
            ["Problemtyp", mod1.peak_problem_type],
        ]))
        story.append(Spacer(1, DesignTokens.SPACE_M))

        # "Was heißt das praktisch?" - Interpretation
        story.append(Paragraph("<b>Was heißt das praktisch?</b>", styles["CustomHeading3"]))
        story.append(p_wrap(mod1.interpretation))
        story.append(Spacer(1, DesignTokens.SPACE_L))

    # === MODULE 2: Phasen (nur wenn verfügbar & Profil) ===
    if profile.include_phase_unbalance:
        story.append(Paragraph("Phasen-Symmetrie & Unwucht-Check", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_S))

        if not mod2.available:
            story.append(p_wrap("Keine 3-Phasen-Daten vorhanden – Unwucht-Check nicht berechnet."))
        else:
            story.append(create_data_table([
                ["Unwucht-Schwelle", "> 3,0 kW (Pmax − Pmin)"],
                ["Anteil Blöcke > Schwelle", f"{mod2.share_over:.1%}"],
                ["Max. Unwucht", f"{mod2.max_unbalance_kw:.1f} kW".replace(".", ",")],
                ["Dominante Phase", mod2.dominant_phase],
            ]))
            story.append(Spacer(1, DesignTokens.SPACE_M))
            story.append(Paragraph("<b>Was heißt das praktisch?</b>", styles["CustomHeading3"]))
            story.append(p_wrap(mod2.recommendation))

        story.append(Spacer(1, DesignTokens.SPACE_L))

    # === MODULE 3: Blindleistung (nur wenn verfügbar & Profil) ===
    if profile.include_blk:
        story.append(Paragraph("Blindleistung / BLK-Analyse", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_S))

        if not blk.available:
            story.append(p_wrap("Keine cosϕ-/Phasen-Daten vorhanden – Blindleistungsanalyse nicht berechnet."))
        else:
            story.append(create_data_table([
                ["Ratio ΣQ/ΣP", f"{blk.ratio:.3f}".replace(".", ",")],
                ["15-min Blöcke > cosϕ=0,9", f"{blk.blocks_over} ({blk.share_over:.1%})"],
                ["Q95 (Empfehlung)", fmt_num(blk.q95, 1, "kvar")],
            ]))
            story.append(Spacer(1, DesignTokens.SPACE_M))
            story.append(Paragraph("<b>Was heißt das praktisch?</b>", styles["CustomHeading3"]))
            story.append(p_wrap(blk.assessment))

        story.append(Spacer(1, DesignTokens.SPACE_L))

    # === HANDLUNGSEMPFEHLUNGEN (NEUE SEITE) ===
    story.append(PageBreak())
    story.append(Paragraph("Individuelle Maßnahmen-Roadmap", styles["CustomHeading2"]))
    story.append(Spacer(1, DesignTokens.SPACE_XS))

    rec_rows = build_recommendations_table_rows(recs, styles)
    story.append(create_recommendations_table(rec_rows))
    story.append(Spacer(1, DesignTokens.SPACE_S))

    # === TOP LASTSPITZEN (profilabhängig) ===
    if profile.include_top_peaks:
        # Lite/Standard: Top-10, Pro: Top-20
        top_n = 10 if profile.name in ["lite", "standard"] else 20

        story.append(PageBreak())
        story.append(Paragraph(f"Top {top_n} Lastspitzen (15-min Mittelwerte)", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_M))

        top_rows = build_top_peaks_rows(df_15, n=top_n)
        story.append(create_peaks_table(top_rows))
        story.append(Spacer(1, DesignTokens.SPACE_M))

    # === PEAK-KONTEXT (nur Pro) ===
    if profile.include_peak_context:
        story.append(PageBreak())
        story.append(Paragraph("Peak-Kontext: Detailanalyse Top-3-Peaks", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_M))

        # Top-3-Peaks analysieren
        top_peaks = analyze_top_peaks(df_15, cap_kw_sel, n=3)

        for i, peak_info in enumerate(top_peaks, start=1):
            # Überschrift für Peak
            story.append(Paragraph(
                f"Peak #{i}: {peak_info.timestamp:%d.%m.%Y %H:%M} ({fmt_num(peak_info.power_kw, 1, 'kW')})",
                styles["CustomHeading3"]
            ))
            story.append(Spacer(1, DesignTokens.SPACE_S))

            # Diagnose-Text
            story.append(Paragraph(f"<b>Diagnose:</b> {peak_info.diagnosis}", styles["BodySmall"]))
            story.append(Spacer(1, DesignTokens.SPACE_M))

            # 12h-Fenster
            fig_12h = make_peak_context_plot(df_15, peak_info.timestamp, 12, cap_kw_sel)
            story.extend(add_chart_with_caption(
                fig_12h,
                "12-Stunden-Fenster: Zeigt unmittelbares Umfeld des Peaks.",
                styles,
                width=170*mm
            ))
            story.append(Spacer(1, DesignTokens.SPACE_M))

            # 3-Tage-Fenster
            fig_3d = make_peak_context_plot(df_15, peak_info.timestamp, 72, cap_kw_sel)
            story.extend(add_chart_with_caption(
                fig_3d,
                "3-Tage-Fenster: Zeigt größeren Kontext (Wochenmuster erkennbar).",
                styles,
                width=170*mm
            ))

            if i < len(top_peaks):
                story.append(Spacer(1, DesignTokens.SPACE_L))

    # === GLOSSAR (Standard/Pro) ===
    if profile.include_glossary:
        story.append(PageBreak())
        glossary_story = build_glossary(styles)
        story.extend(glossary_story)

    # === VISUALISIERUNGEN (mit Captions v2) ===
    from reportlab.platypus import KeepTogether

    # Captions für die Charts (v2: dynamisch je nach Profil)
    chart_captions = []
    if profile.include_heatmap:
        chart_captions.extend([
            "Zeitverlauf der Leistung mit Peak-Shaving Cap. Überschreitungen sind rot markiert.",
            "Sortierte Leistungswerte (Jahresdauerlinie). Zeigt, wie oft welche Leistung erreicht wird.",
            "Heatmap: Durchschnittliche Leistung je Wochentag und Uhrzeit. Zeigt typische Lastmuster.",
            "Peaks pro Monat: Anzahl Überschreitungen (Balken) und maximale Leistung (Linie).",
        ])
    if profile.include_peak_cluster:
        chart_captions.append("Peak-Ereignisse: Dauer vs. benötigte Verschiebeleistung. Größere Punkte = höhere Verschiebeleistung.")
    if blk.available and profile.include_blk:
        chart_captions.append("Blindleistungsverlauf mit cosϕ-Grenzwert (0,9). Überschreitungen zeigen Kompensationsbedarf.")

    # Erste Grafik mit Überschrift zusammenhalten
    viz_header = [
        Paragraph("Visualisierungen", styles["CustomHeading2"]),
        Spacer(1, DesignTokens.SPACE_M)
    ]

    if figs:  # Wenn Grafiken vorhanden
        # Erste Chart + Caption
        first_chart = add_chart_with_caption(figs[0], chart_captions[0] if len(chart_captions) > 0 else "", styles)
        viz_header.extend(first_chart)
        story.append(KeepTogether(viz_header))

        # Rest der Grafiken einzeln mit Captions
        for i, img_path in enumerate(figs[1:], start=1):
            story.append(Spacer(1, DesignTokens.SPACE_M))
            chart_elements = add_chart_with_caption(
                img_path,
                chart_captions[i] if i < len(chart_captions) else "",
                styles
            )
            story.extend(chart_elements)

    doc.build(story, onFirstPage=lambda c, d: add_page_template(c, d, site_name),
                  onLaterPages=lambda c, d: add_page_template(c, d, site_name))


# ============================================================================
# DATENVERARBEITUNG (Original-Logik beibehalten)
# ============================================================================
def _prepare_raw_power_and_optional_phases(
    d0: pd.DataFrame,
    timestamp_col: str,
    power_col: Optional[str],
    power_cols: Optional[List[str]],
    power_unit: str,
    pf_cols: Optional[List[str]],
) -> Dict[str, pd.DataFrame]:
    canonical_total = "power_total" if "power_total" in d0.columns else None
    canonical_p = (
        ["power_1", "power_2", "power_3"]
        if all(c in d0.columns for c in ["power_1", "power_2", "power_3"])
        else None
    )
    canonical_c_total = "cosphi_total" if "cosphi_total" in d0.columns else None
    canonical_c = (
        ["cosphi_1", "cosphi_2", "cosphi_3"]
        if all(c in d0.columns for c in ["cosphi_1", "cosphi_2", "cosphi_3"])
        else None
    )

    use_cols: List[str] = [timestamp_col]
    if canonical_p is not None:
        use_cols += canonical_p
    elif power_cols:
        use_cols += list(power_cols)
    if canonical_total is not None:
        use_cols.append(canonical_total)
    elif power_col:
        use_cols.append(power_col)
    if canonical_c is not None:
        use_cols += canonical_c
    elif canonical_c_total is not None:
        use_cols.append(canonical_c_total)
    elif pf_cols:
        use_cols += list(pf_cols)

    d = d0[use_cols].copy()
    d = d.dropna(subset=[timestamp_col])
    d = d.set_index(timestamp_col)
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()
    if d.empty:
        return {"df": pd.DataFrame()}

    unit = (power_unit or "Auto").lower().strip()

    def _to_kw(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        if unit == "w":
            return x / 1000.0
        if unit == "kw":
            return x
        med = float(x.dropna().median()) if x.notna().any() else 0.0
        return (x / 1000.0) if med > 200.0 else x

    out = pd.DataFrame(index=cast(pd.DatetimeIndex, d.index))

    if canonical_p is not None:
        p1 = _to_kw(d[canonical_p[0]])
        p2 = _to_kw(d[canonical_p[1]])
        p3 = _to_kw(d[canonical_p[2]])
        out["p1_kw"] = p1
        out["p2_kw"] = p2
        out["p3_kw"] = p3
        out["p_kw"] = p1.fillna(0.0) + p2.fillna(0.0) + p3.fillna(0.0)

        if canonical_c is not None:
            out["c1"] = pd.to_numeric(d[canonical_c[0]], errors="coerce").clip(lower=0.0, upper=1.0)
            out["c2"] = pd.to_numeric(d[canonical_c[1]], errors="coerce").clip(lower=0.0, upper=1.0)
            out["c3"] = pd.to_numeric(d[canonical_c[2]], errors="coerce").clip(lower=0.0, upper=1.0)

    elif power_cols is not None and len(power_cols) == 3:
        p1 = _to_kw(d[power_cols[0]])
        p2 = _to_kw(d[power_cols[1]])
        p3 = _to_kw(d[power_cols[2]])
        out["p1_kw"] = p1
        out["p2_kw"] = p2
        out["p3_kw"] = p3
        out["p_kw"] = p1.fillna(0.0) + p2.fillna(0.0) + p3.fillna(0.0)

        if pf_cols is not None and len(pf_cols) == 3:
            out["c1"] = pd.to_numeric(d[pf_cols[0]], errors="coerce").clip(lower=0.0, upper=1.0)
            out["c2"] = pd.to_numeric(d[pf_cols[1]], errors="coerce").clip(lower=0.0, upper=1.0)
            out["c3"] = pd.to_numeric(d[pf_cols[2]], errors="coerce").clip(lower=0.0, upper=1.0)

    else:
        if canonical_total is not None:
            out["p_kw"] = _to_kw(d[canonical_total])
        else:
            if power_col is None:
                raise ValueError("power_col ist None")
            out["p_kw"] = _to_kw(d[power_col])

        if canonical_c_total is not None:
            out["c_total"] = pd.to_numeric(d[canonical_c_total], errors="coerce").clip(lower=0.0, upper=1.0)
        elif pf_cols is not None and len(pf_cols) == 1:
            out["c_total"] = pd.to_numeric(d[pf_cols[0]], errors="coerce").clip(lower=0.0, upper=1.0)

    out = out.dropna(subset=["p_kw"])
    return {"df": out}


def _aggregate_to_interval(df_raw: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{int(minutes)}T"
    res = df_raw.resample(rule, label="left", closed="left").mean()

    agg = pd.DataFrame(index=cast(pd.DatetimeIndex, res.index))
    agg["p_kw"] = cast(pd.Series, res["p_kw"])

    for c in ["p1_kw", "p2_kw", "p3_kw", "c1", "c2", "c3", "c_total"]:
        if c in res.columns:
            agg[c] = cast(pd.Series, res[c])

    agg = agg.dropna(subset=["p_kw"])
    return agg


def _infer_resolution_minutes(idx: pd.DatetimeIndex) -> Optional[int]:
    if len(idx) < 2:
        return None
    diffs = idx.to_series().diff().dropna()
    if diffs.empty:
        return None
    med_sec = float(diffs.dt.total_seconds().median())
    if not np.isfinite(med_sec) or med_sec <= 0:
        return None
    med_min = int(round(med_sec / 60.0))
    return max(1, med_min)


def _missing_quote(idx: pd.DatetimeIndex, resolution_minutes: Optional[int]) -> float:
    if resolution_minutes is None or len(idx) < 2:
        return 0.0
    start, end = idx.min(), idx.max()
    expected = int(((end - start).total_seconds() / 60.0) / float(resolution_minutes)) + 1
    expected = max(expected, 1)
    actual = int(idx.nunique())
    missing = max(expected - actual, 0)
    return float(missing) / float(expected)

# ============================================================================
# BERECHNUNGSMODULE
# ============================================================================
def analyze_top_peaks(df_15: pd.DataFrame, cap_kw: float, n: int = 3) -> List[PeakContextInfo]:
    """
    Analysiert die Top-N-Peaks und erstellt einfache Diagnosen
    Für Pro-Profile: Peak-Kontext-Module
    """
    p = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce")
    top_indices = p.nlargest(n).index

    results: List[PeakContextInfo] = []

    for peak_idx in top_indices:
        peak_ts = pd.Timestamp(peak_idx)
        peak_kw = float(p.loc[peak_idx])

        # Diagnose: Schaue auf Umfeld (±1h)
        window = pd.Timedelta(hours=1)
        start = peak_ts - window
        end = peak_ts + window

        mask = (df_15.index >= start) & (df_15.index <= end)
        window_data = p[mask]

        # Einfache Heuristik
        if len(window_data) < 3:
            diagnosis = "Isolierter Peak (Datenlücke)"
        else:
            mean_window = float(window_data.mean())
            std_window = float(window_data.std())

            # Gleichzeitigkeit: Peak deutlich über Umfeld, aber kurz
            over_cap = (window_data > cap_kw).sum()
            duration_blocks = int(over_cap)

            if duration_blocks <= 2 and (peak_kw - mean_window) > cap_kw * 0.2:
                diagnosis = "Gleichzeitigkeit (kurzer Peak, deutlich über Umfeld)"
            elif duration_blocks >= 8:
                diagnosis = "Dauerlast (lange Überschreitung, ≥2h)"
            elif std_window > cap_kw * 0.15:
                diagnosis = "Anfahrvorgang (starke Schwankungen im Umfeld)"
            else:
                diagnosis = "Normaler Lastgang (moderate Schwankung)"

        results.append(PeakContextInfo(
            timestamp=peak_ts,
            power_kw=peak_kw,
            duration_blocks=duration_blocks if 'duration_blocks' in locals() else 0,
            diagnosis=diagnosis
        ))

    return results


def compute_peak_events(df_15: pd.DataFrame, cap_kw: float, interval_minutes: int = 15) -> PeakEventsResult:
    over = (pd.to_numeric(df_15["p_kw"], errors="coerce") > float(cap_kw)).fillna(False).to_numpy(dtype=bool)
    idx = cast(pd.DatetimeIndex, df_15.index)

    if len(over) == 0:
        return PeakEventsResult(
            n_events=0,
            avg_duration_min=0.0,
            max_duration_min=0.0,
            max_shift_kw=0.0,
            top_months="—",
            interpretation="Keine Überschreitungen erkannt.",
            events_df=pd.DataFrame(),
        )

    events: List[Dict[str, object]] = []
    i = 0
    while i < len(over):
        if not over[i]:
            i += 1
            continue
        start_i = i
        while i < len(over) and over[i]:
            i += 1
        end_i = i - 1

        start_ts = idx[start_i]
        end_ts = idx[end_i]
        blocks = int(end_i - start_i + 1)
        duration_min = float(blocks * interval_minutes)

        excess = (pd.to_numeric(df_15["p_kw"].iloc[start_i : end_i + 1], errors="coerce") - cap_kw).clip(lower=0)
        max_shift = float(pd.to_numeric(excess, errors="coerce").max()) if not excess.empty else 0.0

        events.append(
            {
                "start": start_ts,
                "end": end_ts,
                "blocks": blocks,
                "duration_min": duration_min,
                "max_shift_kw": max_shift,
            }
        )

    ev = pd.DataFrame(events)
    n_events = int(len(ev))
    avg_dur = float(pd.to_numeric(ev["duration_min"], errors="coerce").mean()) if n_events > 0 else 0.0
    max_dur = float(pd.to_numeric(ev["duration_min"], errors="coerce").max()) if n_events > 0 else 0.0
    max_shift_kw = float(pd.to_numeric(ev["max_shift_kw"], errors="coerce").max()) if n_events > 0 else 0.0

    top_months = "—"
    if n_events > 0:
        start_np = ev["start"].to_numpy(dtype="datetime64[ns]")

        def _ym(x: np.datetime64) -> str:
            t = pd.Timestamp(x)
            return f"{t.year:04d}-{t.month:02d}"

        months = pd.Series([_ym(x) for x in start_np], index=ev.index, dtype="string")
        top = months.value_counts().head(3)
        top_months = ", ".join([f"{m}: {int(c)}" for m, c in top.items()])

    share_short = float((pd.to_numeric(ev["duration_min"], errors="coerce") <= float(interval_minutes)).mean()) if n_events > 0 else 0.0
    share_long = float((pd.to_numeric(ev["duration_min"], errors="coerce") >= 60.0).mean()) if n_events > 0 else 0.0

    # Peak-Problemtyp bestimmen (für Executive Summary)
    if n_events == 0:
        interp = "Keine Peak-Ereignisse oberhalb des Caps erkannt."
        problem_type = "Keine Peaks"
    elif share_short >= 0.6:
        interp = (
            "Viele kurze Lastspitzen (≈ ein 15-min Block) → gut geeignet für Lastmanagement/Sequenzierung.\n"
            "Typische Maßnahme: Verbraucher zeitlich staffeln oder harte Gleichzeitigkeit vermeiden."
        )
        problem_type = "Kurzspitzen"
    elif share_long >= 0.4:
        interp = (
            "Signifikanter Anteil längerer Überschreitungen → Hinweis auf Dauerlasten.\n"
            "Typische Maßnahme: Prozess/Grundlast prüfen (z. B. Kühlung, Heizung, Druckluft, Ofenphasen)."
        )
        problem_type = "Langspitzen / Grundlast"
    else:
        interp = "Gemischtes Muster: sowohl kurze als auch längere Peak-Ereignisse."
        problem_type = "Gemischtes Muster"

    return PeakEventsResult(
        n_events=n_events,
        avg_duration_min=avg_dur,
        max_duration_min=max_dur,
        max_shift_kw=max_shift_kw,
        top_months=top_months,
        interpretation=interp,
        events_df=ev,
        peak_problem_type=problem_type,
    )


def compute_unbalance_module(df_15: pd.DataFrame, threshold_kw: float = 3.0) -> UnbalanceResult:
    need = {"p1_kw", "p2_kw", "p3_kw"}
    if not need.issubset(set(df_15.columns)):
        return UnbalanceResult(available=False)

    p = df_15[["p1_kw", "p2_kw", "p3_kw"]].copy()
    p = p.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    p = p.dropna(how="any")
    if p.empty:
        return UnbalanceResult(available=False)

    pmax = p.max(axis=1)
    pmin = p.min(axis=1)
    unb = (pmax - pmin).astype(float)

    over = unb > float(threshold_kw)
    share_over = float(over.mean())
    max_unb = float(unb.max())

    dom = p.idxmax(axis=1).value_counts(normalize=True)
    map_phase = {"p1_kw": "L1", "p2_kw": "L2", "p3_kw": "L3"}
    dom_named: Dict[str, float] = {map_phase.get(str(k), str(k)): float(v) for k, v in dom.items()}

    if dom_named:
        dominant_phase_name = max(dom_named, key=lambda k: dom_named[k])
        dominant_share = float(dom_named[dominant_phase_name])
        dominant_phase = f"{dominant_phase_name} ({dominant_share:.0%})"
    else:
        dominant_phase_name, dominant_share = "—", 0.0
        dominant_phase = "—"

    if dominant_phase_name == "L3" and dominant_share >= 0.55:
        rec = (
            "L3 dominiert konsistent.\n"
            "Empfehlung: einphasige Lasten von L3 auf L1/L2 umklemmen/verteilen (Netzschonung & Kapazitätsfreigabe)."
        )
    elif share_over >= 0.2:
        rec = (
            "Unwucht häufig über Schwelle.\n"
            "Empfehlung: Phasenbelegung prüfen (einphasige Verbraucher verteilen), ggf. Elektriker-Check."
        )
    else:
        rec = (
            "Unwucht selten/gering.\n"
            "Empfehlung: kein akuter Handlungsdruck – bei Erweiterungen trotzdem auf symmetrische Verteilung achten."
        )

    return UnbalanceResult(
        available=True,
        share_over=share_over,
        max_unbalance_kw=max_unb,
        dominant_phase=dominant_phase,
        dominant_phase_name=dominant_phase_name,
        dominant_phase_share=dominant_share,
        recommendation=rec,
    )


def compute_blk_metrics_15min(df_15: pd.DataFrame) -> BlkResult:
    def q_from_p_cosphi(p: pd.Series, c: pd.Series) -> pd.Series:
        c_np = np.clip(pd.to_numeric(c, errors="coerce").to_numpy(dtype=float), 0.0, 1.0)
        p_np = pd.to_numeric(p, errors="coerce").to_numpy(dtype=float)
        phi = np.arccos(c_np)
        tanphi = np.tan(phi)
        return pd.Series(p_np * tanphi, index=p.index)

    has_phase = all(k in df_15.columns for k in ["p1_kw", "p2_kw", "p3_kw", "c1", "c2", "c3"])
    has_total = all(k in df_15.columns for k in ["p_kw", "c_total"])

    if has_phase:
        p1 = cast(pd.Series, df_15["p1_kw"])
        p2 = cast(pd.Series, df_15["p2_kw"])
        p3 = cast(pd.Series, df_15["p3_kw"])
        c1 = cast(pd.Series, df_15["c1"])
        c2 = cast(pd.Series, df_15["c2"])
        c3 = cast(pd.Series, df_15["c3"])

        q1 = q_from_p_cosphi(p1, c1)
        q2 = q_from_p_cosphi(p2, c2)
        q3 = q_from_p_cosphi(p3, c3)

        pges = pd.to_numeric(p1, errors="coerce").fillna(0.0) + pd.to_numeric(p2, errors="coerce").fillna(0.0) + pd.to_numeric(p3, errors="coerce").fillna(0.0)
        qges = q1.fillna(0.0) + q2.fillna(0.0) + q3.fillna(0.0)

        df_15["q_kvar"] = qges

    elif has_total:
        pges = cast(pd.Series, df_15["p_kw"])
        c = cast(pd.Series, df_15["c_total"])
        qges = q_from_p_cosphi(pges, c)
        df_15["q_kvar"] = qges

    else:
        return BlkResult(available=False)

    p_abs = pd.to_numeric(pges, errors="coerce").abs().replace(0.0, np.nan)
    q_limit = p_abs * 0.4843
    over = (qges > q_limit) & p_abs.notna()

    denom = float(p_abs.fillna(0.0).sum())
    ratio = float(qges.fillna(0.0).sum()) / denom if denom > 0 else 0.0
    blocks_over = int(over.fillna(False).sum())
    share_over = float(over.fillna(False).mean())
    q95 = float(qges.dropna().quantile(0.95)) if qges.notna().any() else 0.0

    if ratio > 0.5:
        assessment = "Wirtschaftlich oft sinnvoll (Ratio > 0,5).\nKonkrete Kosten/Nutzen bitte mit Tarifdaten prüfen."
    elif 0.3 <= ratio <= 0.5:
        assessment = "Technisch empfohlen (Ratio 0,3–0,5).\nWirtschaftlichkeit hängt von Blindarbeitsregelung/Strafen ab."
    else:
        assessment = "Eher niedrige Blindleistungsrelevanz (Ratio < 0,3).\nTrotzdem Grenzwert-Zeitschritte prüfen (Ausreißer möglich)."

    df_15["q_limit"] = q_limit

    return BlkResult(
        available=True,
        ratio=ratio,
        blocks_over=blocks_over,
        share_over=share_over,
        q95=q95,
        assessment=assessment,
    )


# ============================================================================
# CAP / TARIFF / SCENARIOS
# ============================================================================
def compute_cap(power_kw: pd.Series, reduction_goal: str, manual_cap_kw: Optional[float], manual_value: str) -> Tuple[float, str]:
    goal = (reduction_goal or "").strip()
    if goal in GOAL_TO_QUANTILE:
        q = GOAL_TO_QUANTILE[goal]
        cap = float(pd.to_numeric(power_kw, errors="coerce").quantile(q))
        return cap, f"{goal} (P{int(q * 100)})"
    if goal.lower().startswith("manuell") or goal == "Manuell":
        if manual_cap_kw is None:
            raise ValueError("Manueller Cap wurde ausgewählt, aber manual_cap_kw ist None.")
        label = manual_value or f"{manual_cap_kw:.1f} kW"
        return float(manual_cap_kw), f"Manuell ({label})"
    return float(pd.to_numeric(power_kw, errors="coerce").max()), "Kein Ziel (Fallback)"


def _tariff_for_util_hours(t: Tariffs, util_hours: float) -> Tuple[float, float, str]:
    if util_hours < t.switch_hours:
        return t.work_ct_low, t.demand_eur_kw_a_low, f"< {t.switch_hours:.0f} h/a"
    return t.work_ct_high, t.demand_eur_kw_a_high, f"> {t.switch_hours:.0f} h/a"


@dataclass(frozen=True)
class _Scenario:
    name: str
    cap_kw: float
    cap_label: str
    peak_after_kw: float
    util_hours_after: float
    tariff_label_after: str
    tariff_switched: bool
    cost_after: float
    savings_eur: float
    blocks_over_cap: int
    share_over_cap: float
    kwh_to_shift: float


def _compute_scenario(
    name: str,
    cap_kw: float,
    cap_label: str,
    annual_energy_kwh: float,
    peak_before_kw: float,
    tariffs: Tariffs,
    df_15: pd.DataFrame,
    block_h: float,
) -> _Scenario:
    peak_after = float(min(float(peak_before_kw), float(cap_kw)))
    util_after = (annual_energy_kwh / peak_after) if peak_after > 0 else 0.0

    work_ct_before, demand_eur_before, label_before = _tariff_for_util_hours(
        tariffs, (annual_energy_kwh / peak_before_kw) if peak_before_kw > 0 else 0.0
    )
    work_ct_after, demand_eur_after, label_after = _tariff_for_util_hours(tariffs, util_after)

    tariff_switched = (label_before != label_after)

    cost_after = annual_energy_kwh * (work_ct_after / 100.0) + peak_after * demand_eur_after
    cost_before = annual_energy_kwh * (work_ct_before / 100.0) + float(peak_before_kw) * demand_eur_before
    savings = cost_before - cost_after

    excess_kw = (pd.to_numeric(df_15["p_kw"], errors="coerce") - float(cap_kw)).clip(lower=0)
    blocks_over = int((excess_kw > 0).sum())
    share_over = float((excess_kw > 0).mean())
    kwh_to_shift = float((excess_kw * block_h).sum())

    return _Scenario(
        name=name,
        cap_kw=float(cap_kw),
        cap_label=cap_label,
        peak_after_kw=peak_after,
        util_hours_after=float(util_after),
        tariff_label_after=label_after,
        tariff_switched=tariff_switched,
        cost_after=float(cost_after),
        savings_eur=float(savings),
        blocks_over_cap=blocks_over,
        share_over_cap=share_over,
        kwh_to_shift=kwh_to_shift,
    )


# ============================================================================
# RECOMMENDATIONS
# ============================================================================
def _estimate_p_gesamt_kw(peak_15_kw: float) -> float:
    return float(max(peak_15_kw, 0.0))


def _estimate_p_verschiebbar_kw(mod1: PeakEventsResult, scenario: _Scenario) -> float:
    return float(max(mod1.max_shift_kw, 0.0))


def build_recommendations_v3(
    mod1: PeakEventsResult,
    mod2: UnbalanceResult,
    blk: BlkResult,
    util_hours_before: float,
    util_hours_after: float,
    p_verschiebbar_kw: float,
    p_gesamt_kw: float,
    tariffs: Tariffs,
) -> List[Recommendation]:
    recs: List[Recommendation] = []

    ev = mod1.events_df
    if ev is None or ev.empty:
        n_short = 0
        n_long = 0
        max_dur = 0.0
        share_short = 0.0
    else:
        dur = pd.to_numeric(ev["duration_min"], errors="coerce").dropna()
        n_short = int((dur <= 15.0).sum())
        n_long = int((dur >= 60.0).sum())
        max_dur = float(dur.max()) if not dur.empty else 0.0
        share_short = float(n_short / max(int(len(dur)), 1))

    dominant_agg = (p_gesamt_kw > 0) and (p_verschiebbar_kw / p_gesamt_kw > 0.40)

    unb_kw = float(mod2.max_unbalance_kw) if mod2.available else 0.0
    dom_phase = mod2.dominant_phase_name if mod2.available else "—"
    dom_share = float(mod2.dominant_phase_share) if mod2.available else 0.0

    ratio_qp = float(blk.ratio) if blk.available else 0.0
    share_cosphi_under_0_9 = float(blk.share_over) if blk.available else 0.0

    tariff_before = util_hours_before
    tariff_after = util_hours_after
    crossed_to_high = (tariff_before < tariffs.switch_hours) and (tariff_after > tariffs.switch_hours)

    # Z1
    if (n_short > 10) and (share_short > 0.50):
        recs.append(
            Recommendation(
                code="Z1",
                category="Peak-Strategie – Quick Win",
                trigger=f"{n_short} kurze Peaks (≤15 min), Anteil kurz {share_short:.0%}",
                action="Quick Win: Startzeiten staffeln. Beispiel: Max. 1 Ofen pro 15-min-Fenster aufheizen; Spülmaschinen außerhalb der Backzeiten nutzen.",
                priority="Quick Win",
            )
        )

    # Z2
    if dominant_agg:
        recs.append(
            Recommendation(
                code="Z2",
                category="Peak-Strategie – Investition erforderlich",
                trigger=f"Verschiebbar ~{(p_verschiebbar_kw / p_gesamt_kw):.0%} der Gesamtleistung",
                action="Installation eines Lastabwurfrelais für nicht-kritische Lasten (z. B. Lüftung, Warmwasser) bei Annäherung an das Cap.",
                priority="Investition erforderlich",
            )
        )

    # Z4
    if (n_long >= 3) and (max_dur > 60.0):
        recs.append(
            Recommendation(
                code="Z4",
                category="Peak-Strategie",
                trigger=f"{n_long} lange Peak-Ereignisse (≥60 min), max. Dauer {max_dur:.0f} min",
                action="Prozess-/Produktionsplanung: Backfenster in Nebenzeiten (NT) verschieben, um Überschneidungen mit Heizungspeaks zu vermeiden.",
                priority="",
            )
        )

    # Z6
    if max_dur >= 120.0:
        recs.append(
            Recommendation(
                code="Z6",
                category="Peak-Strategie – Investition erforderlich",
                trigger=f"Max. Peak-Dauer {max_dur:.0f} min (Hinweis auf Dauerlast: Kälte/Heizung/Grundlast)",
                action="Technische Effizienzprüfung: veraltete Aggregate (>15 J.) gegen drehzahlgeregelte Modelle tauschen; Sollwert-Optimierung Kühlräume um +1–2 K.",
                priority="Investition erforderlich",
            )
        )

    ratio_shift = (p_verschiebbar_kw / p_gesamt_kw) if p_gesamt_kw > 0 else 0.0

    # Z7
    if ratio_shift < 0.10 and p_gesamt_kw > 0:
        recs.append(
            Recommendation(
                code="Z7",
                category="Wirtschaftlichkeit (LM) – Quick Win",
                trigger=f"P_verschiebbar ≈ {ratio_shift:.0%} von P_gesamt",
                action="Quick Win: Fokus auf Mitarbeiterschulung und manuelle Schaltpläne (Verhaltensänderung).",
                priority="Quick Win",
            )
        )
    # Z8
    elif 0.10 <= ratio_shift <= 0.30:
        recs.append(
            Recommendation(
                code="Z8",
                category="Wirtschaftlichkeit (LM)",
                trigger=f"P_verschiebbar ≈ {ratio_shift:.0%} von P_gesamt",
                action="Automatisches Lastmanagement mit Prioritäten (z. B. Kompressoren & Spülmaschinen) empfohlen.",
                priority="",
            )
        )
    # Z9
    elif ratio_shift > 0.30:
        recs.append(
            Recommendation(
                code="Z9",
                category="Wirtschaftlichkeit (LM) – Investition erforderlich",
                trigger=f"P_verschiebbar ≈ {ratio_shift:.0%} von P_gesamt",
                action="Investition erforderlich: Vollwertiges Lastmanagementsystem mit Messung je Hauptverbraucher und PV-Integration wirtschaftlich hoch attraktiv.",
                priority="Investition erforderlich",
            )
        )

    # Z10
    if (unb_kw > 3.0) and (dom_share > 0.80):
        recs.append(
            Recommendation(
                code="Z10",
                category="Technik (Phasen/Unwucht) – Quick Win",
                trigger=f"Unwucht max. {unb_kw:.1f} kW, dominante Phase {dom_phase} ({dom_share:.0%})",
                action="Quick Win: Elektriker-Check & Umklemmung. Einphasige Großverbraucher von der dominanten Phase auf schwächere Phasen verteilen.",
                priority="Quick Win",
            )
        )

    # Z13
    if (ratio_qp > 0.4) or (share_cosphi_under_0_9 > 0.40):
        recs.append(
            Recommendation(
                code="Z13",
                category="Technik (Blindleistung) – Investition erforderlich",
                trigger=f"Ratio ΣQ/ΣP={ratio_qp:.2f} oder Anteil Grenzwert-Verletzung ≈ {share_cosphi_under_0_9:.0%}",
                action="Investition erforderlich: Blindleistungskompensation (BLK) dringend. Ziel-cos φ: 0,95–0,98; Amortisation oft < 3 Jahre.",
                priority="Investition erforderlich",
            )
        )

    # Z14
    if crossed_to_high:
        recs.append(
            Recommendation(
                code="Z14",
                category="Tarif-Strategie",
                trigger=f"Benutzungsdauer vorher {tariff_before:.0f} h/a → nachher {tariff_after:.0f} h/a (Schwelle {tariffs.switch_hours:.0f})",
                action="Tarif neu verhandeln: Durch höhere Benutzungsdauer ist ein Wechsel in günstigere Netznutzungsgruppen möglich.",
                priority="",
            )
        )

    if not recs:
        recs.append(
            Recommendation(
                code="—",
                category="Allgemein",
                trigger="Keine klaren Trigger über den definierten Schwellenwerten erkannt.",
                action="Empfehlung: Cap/Schwellenwerte prüfen (manueller Cap), weitere Messdauer erhöhen, oder Hauptverbraucher separat messen (Submeter) für bessere Ursachenanalyse.",
                priority="",
            )
        )

    prio_rank = {"Quick Win": 0, "Investition erforderlich": 1, "": 2}
    recs_sorted = sorted(recs, key=lambda r: (prio_rank.get(r.priority, 9), r.code))

    return recs_sorted


def build_recommendations_table_rows(recs: List[Recommendation], styles) -> TableData:
    def p(text: str) -> Paragraph:
        return Paragraph((text or "—").replace("\n", "<br/>"), styles["CustomBody"])

    rows: TableData = [["Kategorie", "Identifizierter Trigger", "Empfohlene Maßnahme"]]
    for r in recs:
        cat = r.category
        trig = f"{r.code}: {r.trigger}" if r.code and r.code != "—" else r.trigger
        rows.append([p(cat), p(trig), p(r.action)])
    return rows


# ============================================================================
# TOP PEAKS TABLE
# ============================================================================
def build_top_peaks_rows(df_15: pd.DataFrame, n: int = 20) -> TableData:
    idx = cast(pd.DatetimeIndex, df_15.index)

    cosphi: Optional[pd.Series] = None
    if "c_total" in df_15.columns:
        cosphi = pd.to_numeric(cast(pd.Series, df_15["c_total"]), errors="coerce").clip(lower=0.0, upper=1.0)
    elif all(c in df_15.columns for c in ["c1", "c2", "c3"]):
        c1 = pd.to_numeric(cast(pd.Series, df_15["c1"]), errors="coerce").clip(lower=0.0, upper=1.0)
        c2 = pd.to_numeric(cast(pd.Series, df_15["c2"]), errors="coerce").clip(lower=0.0, upper=1.0)
        c3 = pd.to_numeric(cast(pd.Series, df_15["c3"]), errors="coerce").clip(lower=0.0, upper=1.0)
        cosphi = (c1 + c2 + c3) / 3.0

    tmp = pd.DataFrame(
        {
            "ts": idx,
            "p_kw": pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce"),
        }
    )
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


# ============================================================================
# FORMATTING / UTILITIES
# ============================================================================
def fmt_num(x: Optional[NumberLike], decimals: int, suffix: str) -> str:
    if x is None:
        return f"— {suffix}".strip()
    try:
        xf = float(x)
    except Exception:
        return f"— {suffix}".strip()
    s = f"{xf:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} {suffix}".strip()


def fmt_pct(x: float, decimals: int = 1) -> str:
    s = f"{x:.{decimals}f}".replace(".", ",")
    return f"{s} %"


def _tempfile_path(filename: str) -> str:
    import tempfile
    from uuid import uuid4
    p = Path(tempfile.gettempdir()) / f"peakguard_{uuid4().hex}_{filename}"
    return str(p)