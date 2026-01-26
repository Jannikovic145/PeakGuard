# report_builder/config.py
"""
Zentrale Konfiguration und Design-Tokens für PeakGuard.
Enthält alle Konstanten, Farben und Einstellungen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List

import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.units import mm


# ============================================================================
# DESIGN TOKENS - PeakGuard Corporate Identity v2
# ============================================================================
class DesignTokens:
    """Zentrale Design-Tokens für konsistente Gestaltung"""

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


class PeakGuardColors:
    """PeakGuard Farbpalette - ReportLab und Matplotlib"""

    # ReportLab Colors
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

    # Matplotlib Colors (hex strings)
    MPL = {
        'primary': '#0da2e7',
        'primary_dark': '#0f1729',
        'accent': '#FF6B35',
        'success': '#28A745',
        'warning': '#FFC107',
        'danger': '#DC3545',
        'grid': '#E9ECEF',
        'text': '#1A1A1A',
        'background': '#FFFFFF',
        'bronze': '#CD7F32',
        'silver': '#C0C0C0',
        'gold': '#FFD700',
    }

    @classmethod
    def setup_matplotlib_theme(cls) -> None:
        """Konfiguriert Matplotlib mit PeakGuard-Design"""
        plt.rcParams.update({
            'figure.facecolor': '#FFFFFF',
            'axes.facecolor': '#FAFBFC',
            'axes.edgecolor': '#E9ECEF',
            'axes.labelcolor': '#1A1A1A',
            'axes.grid': True,
            'grid.color': '#E9ECEF',
            'grid.linewidth': 0.4,
            'grid.alpha': 0.5,
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
            'lines.linewidth': 2.5,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.linewidth': 0.8,
        })


# Theme beim Import aktivieren
PeakGuardColors.setup_matplotlib_theme()


# ============================================================================
# KONSTANTEN
# ============================================================================
GOAL_TO_QUANTILE: Dict[str, float] = {
    "Bronze": 0.95,
    "Silber": 0.90,
    "Gold": 0.85,
}


# ============================================================================
# DATACLASSES FÜR KONFIGURATION
# ============================================================================
@dataclass(frozen=True)
class Tariffs:
    """Tarifkonfiguration für Kostenberechnungen"""
    switch_hours: float = 2500.0
    work_ct_low: float = 8.27
    demand_eur_kw_a_low: float = 19.93
    work_ct_high: float = 4.25
    demand_eur_kw_a_high: float = 120.43


@dataclass(frozen=True)
class ReportConfig:
    """
    Zentrale Konfiguration für Report-Parameter
    Kann später aus YAML/JSON geladen werden
    """
    # Schwellenwerte
    peak_context_min_savings: float = 5000.0
    peak_context_min_events: int = 20
    unbalance_threshold_kw: float = 3.0
    blk_cosphi_threshold: float = 0.9

    # Chart-Einstellungen
    chart_dpi: int = 180
    chart_width_mm: float = 170.0
    chart_height_mm: float = 85.0

    # Top-Peaks
    top_peaks_lite: int = 10
    top_peaks_standard: int = 10
    top_peaks_pro: int = 20

    # Validierung
    max_file_size_mb: float = 100.0
    max_rows: int = 1_000_000

    # Farben (optional überschreibbar)
    custom_primary: Optional[str] = None
    custom_accent: Optional[str] = None


@dataclass(frozen=True)
class ReportProfile:
    """Report-Variante: lite, standard, pro"""
    name: str
    include_exec_summary: bool = True
    include_scenarios: bool = True
    include_heatmap: bool = True
    include_peak_cluster: bool = True
    include_roadmap: bool = True
    include_top_peaks: bool = True
    include_phase_unbalance: bool = True
    include_blk: bool = True
    include_peak_context: bool = False
    include_glossary: bool = False
    max_pages_target: int = 10


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


# Standard-Config
DEFAULT_CONFIG = ReportConfig()


def apply_intelligent_triggers(
    profile: ReportProfile,
    savings_eur: float,
    n_peak_events: int,
    blk_available: bool,
    unbalance_available: bool,
    config: Optional[ReportConfig] = None,
) -> ReportProfile:
    """
    Wendet intelligente Trigger an, um Module automatisch zu aktivieren/deaktivieren
    """
    from dataclasses import replace

    config = config or DEFAULT_CONFIG

    adjusted = replace(profile)

    # Trigger 1: Peak-Kontext nur bei hohem Potenzial (Pro)
    if profile.name == "pro" and profile.include_peak_context:
        if savings_eur < config.peak_context_min_savings and n_peak_events < config.peak_context_min_events:
            adjusted = replace(adjusted, include_peak_context=False)

    # Trigger 2: BLK nur wenn verfügbar
    if profile.include_blk and not blk_available:
        adjusted = replace(adjusted, include_blk=False)

    # Trigger 3: Phase-Unbalance nur wenn verfügbar
    if profile.include_phase_unbalance and not unbalance_available:
        adjusted = replace(adjusted, include_phase_unbalance=False)

    return adjusted
