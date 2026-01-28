# report_builder/models.py
"""
Datenmodelle und Dataclasses für PeakGuard.
Enthält alle Ergebnistypen und Strukturen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd


# Type Aliases
NumberLike = Union[int, float, np.number]
TableData = List[List[Union[str, float, int, None, object]]]


@dataclass
class PeakEventsResult:
    """Ergebnis der Peak-Event-Analyse"""
    n_events: int
    avg_duration_min: float
    max_duration_min: float
    max_shift_kw: float
    top_months: str
    interpretation: str
    events_df: pd.DataFrame
    peak_problem_type: str = "Kurzspitzen"


@dataclass
class PeakContextInfo:
    """Info zu einem einzelnen Peak für Kontext-Analyse (Pro)"""
    timestamp: pd.Timestamp
    power_kw: float
    duration_blocks: int
    diagnosis: str


@dataclass
class UnbalanceResult:
    """Ergebnis der Phasen-Unwucht-Analyse"""
    available: bool
    share_over: float = 0.0
    max_unbalance_kw: float = 0.0
    dominant_phase: str = "—"
    dominant_phase_name: str = "—"
    dominant_phase_share: float = 0.0
    recommendation: str = ""


@dataclass(frozen=True)
class BlkResult:
    """Ergebnis der Blindleistungsanalyse"""
    available: bool
    ratio: float = 0.0
    blocks_over: int = 0
    share_over: float = 0.0
    q95: float = 0.0
    assessment: str = ""


@dataclass(frozen=True)
class Recommendation:
    """Einzelne Handlungsempfehlung"""
    code: str
    category: str
    trigger: str
    action: str
    priority: str


@dataclass(frozen=True)
class Scenario:
    """Berechnetes Peak-Shaving-Szenario"""
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


@dataclass
class ReportInput:
    """
    Konsolidierte Eingabedaten für Report-Generierung.
    Ersetzt die 15+ Parameter der build_pdf_report Funktion.
    """
    # DataFrame
    df: pd.DataFrame

    # Spalten-Konfiguration
    timestamp_col: str
    power_col: Optional[str] = None
    power_cols: Optional[List[str]] = None
    power_unit: str = "Auto"
    pf_cols: Optional[List[str]] = None

    # Metadaten
    source_name: str = ""
    site_name: str = ""
    data_quality: str = "OK"
    meter_type: str = "RLM"

    # Peak-Shaving Konfiguration
    reduction_goal: str = "Bronze"
    manual_value: str = ""
    manual_cap_kw: Optional[float] = None

    # Optionen
    include_reactive: bool = True
    input_resolution_minutes: Optional[int] = None
    demand_interval_minutes: int = 15

    def validate(self) -> List[str]:
        """Validiert die Eingabedaten und gibt Liste von Fehlern zurück"""
        errors = []

        if self.df is None or self.df.empty:
            errors.append("DataFrame ist leer oder None")
            return errors

        if self.timestamp_col not in self.df.columns:
            errors.append(f"Zeitstempel-Spalte '{self.timestamp_col}' nicht gefunden")

        if self.power_col is None and self.power_cols is None:
            errors.append("Entweder power_col oder power_cols muss angegeben werden")

        if self.power_col and self.power_col not in self.df.columns:
            errors.append(f"Leistungsspalte '{self.power_col}' nicht gefunden")

        if self.power_cols:
            for col in self.power_cols:
                if col not in self.df.columns:
                    errors.append(f"Phasen-Spalte '{col}' nicht gefunden")

        return errors


@dataclass
class AnalysisResult:
    """
    Konsolidierte Ergebnisse der Analyse.
    Wird von Analytics-Modul erstellt und von PDF-Builder verwendet.
    """
    # Aggregierte Daten
    df_15: pd.DataFrame
    df_raw: pd.DataFrame

    # Kennzahlen
    peak_15_kw: float
    peak_1m_kw: Optional[float]
    energy_kwh: float
    annual_energy_kwh: float
    duration_hours: float

    # Kosten vor Peak-Shaving
    util_hours_before: float
    cost_before: float
    work_ct_before: float
    demand_eur_before: float
    tariff_label_before: str

    # Zeitraum
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    inferred_resolution: Optional[int]
    missing_quote: float

    # Ausgewähltes Szenario
    scenario_selected: Scenario

    # Package-Szenarien
    pkg_scenarios: List[Scenario]

    # Modul-Ergebnisse
    peak_events: PeakEventsResult
    unbalance: UnbalanceResult
    blk: BlkResult

    # Empfehlungen
    recommendations: List[Recommendation]
