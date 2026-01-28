# report_builder/utils.py
"""
Hilfsfunktionen für PeakGuard.
Enthält Formatierung, Parsing und allgemeine Utilities.
"""
from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path
from typing import Optional, Union
from uuid import uuid4

import numpy as np
import pandas as pd

from .models import NumberLike


# ============================================================================
# LOGGING
# ============================================================================
logger = logging.getLogger("peakguard")


def setup_logging(level: int = logging.INFO) -> None:
    """Konfiguriert das Logging für PeakGuard"""
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)


# ============================================================================
# NUMERIC PARSING (DRY - zentralisiert)
# ============================================================================
class RobustNumericParser:
    """
    Robuste Zahlen-Parsing-Klasse für verschiedene Formate.
    Unterstützt:
    - Dezimalkomma (12,34)
    - Tausenderpunkte (1.234,56)
    - Einheiten/Strings ("12,3 kW", "123 W")
    - NBSP/Spaces
    """

    @staticmethod
    def parse_series(s: pd.Series) -> pd.Series:
        """
        Konvertiert eine pandas Series robust zu numerischen Werten.

        Args:
            s: Input Series (kann strings, numbers oder mixed enthalten)

        Returns:
            Series mit float-Werten (NaN für nicht-konvertierbare)
        """
        if pd.api.types.is_numeric_dtype(s):
            return pd.to_numeric(s, errors="coerce")

        x = s.astype("string")

        # NBSP & normale Spaces entfernen
        x = x.str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)

        # Alles außer Zahlen, . , - und + entfernen (Einheiten etc.)
        x = x.str.replace(r"[^0-9\.,\-\+]", "", regex=True)

        # Deutsches Format erkennen: 1.234,56 -> 1234.56
        looks_german = x.str.contains(r"\d{1,3}(\.\d{3})+,\d+", regex=True, na=False).mean() > 0.05

        if looks_german:
            x = x.str.replace(".", "", regex=False)
            x = x.str.replace(",", ".", regex=False)
        else:
            # Wenn nur Komma als Dezimaltrenner vorkommt -> Komma zu Punkt
            has_comma_decimal = x.str.contains(r"\d+,\d+", regex=True, na=False).mean() > 0.05
            has_dot_decimal = x.str.contains(r"\d+\.\d+", regex=True, na=False).mean() > 0.05

            if has_comma_decimal and not has_dot_decimal:
                x = x.str.replace(",", ".", regex=False)

        return pd.to_numeric(x, errors="coerce")

    @staticmethod
    def parse_value(value: Union[str, float, int, None]) -> Optional[float]:
        """Parst einen einzelnen Wert"""
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)

        try:
            s = pd.Series([value])
            result = RobustNumericParser.parse_series(s)
            return float(result.iloc[0]) if pd.notna(result.iloc[0]) else None
        except Exception:
            return None


# ============================================================================
# FORMATIERUNG
# ============================================================================
def fmt_num(x: Optional[NumberLike], decimals: int, suffix: str) -> str:
    """
    Formatiert eine Zahl im deutschen Format mit Suffix.

    Args:
        x: Zahl oder None
        decimals: Anzahl Dezimalstellen
        suffix: Einheit (z.B. "kW", "€/a")

    Returns:
        Formatierter String (z.B. "1.234,56 kW")
    """
    if x is None:
        return f"— {suffix}".strip()

    try:
        xf = float(x)
    except (ValueError, TypeError):
        logger.warning(f"Konnte Wert '{x}' nicht zu float konvertieren")
        return f"— {suffix}".strip()

    # Formatieren mit Tausendertrenner
    s = f"{xf:,.{decimals}f}"
    # Umwandeln in deutsches Format: 1,234.56 -> 1.234,56
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")

    return f"{s} {suffix}".strip()


def fmt_pct(x: float, decimals: int = 1) -> str:
    """Formatiert einen Prozentwert im deutschen Format"""
    s = f"{x:.{decimals}f}".replace(".", ",")
    return f"{s} %"


def fmt_date(ts: pd.Timestamp, include_time: bool = True) -> str:
    """Formatiert ein Datum im deutschen Format"""
    if include_time:
        return ts.strftime("%d.%m.%Y %H:%M")
    return ts.strftime("%d.%m.%Y")


# ============================================================================
# TEMPORÄRE DATEIEN
# ============================================================================
def get_temp_path(filename: str) -> Path:
    """
    Erstellt einen eindeutigen temporären Dateipfad.

    Args:
        filename: Basis-Dateiname (z.B. "timeseries.png")

    Returns:
        Path-Objekt zum temporären Dateipfad
    """
    return Path(tempfile.gettempdir()) / f"peakguard_{uuid4().hex}_{filename}"


# ============================================================================
# VALIDIERUNG
# ============================================================================
def validate_dataframe(
    df: pd.DataFrame,
    required_cols: list[str],
    max_rows: int = 1_000_000
) -> list[str]:
    """
    Validiert einen DataFrame auf Vollständigkeit und Größe.

    Args:
        df: Zu validierender DataFrame
        required_cols: Liste erforderlicher Spalten
        max_rows: Maximale Zeilenanzahl

    Returns:
        Liste von Fehlermeldungen (leer wenn valide)
    """
    errors = []

    if df is None:
        errors.append("DataFrame ist None")
        return errors

    if df.empty:
        errors.append("DataFrame ist leer")
        return errors

    if len(df) > max_rows:
        errors.append(f"DataFrame hat zu viele Zeilen: {len(df)} > {max_rows}")

    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Erforderliche Spalte fehlt: {col}")

    return errors


def sanitize_filename(filename: str) -> str:
    """
    Bereinigt einen Dateinamen von unsicheren Zeichen.

    Args:
        filename: Original-Dateiname

    Returns:
        Bereinigter Dateiname
    """
    # Pfad-Traversal verhindern
    filename = filename.replace("..", "").replace("/", "_").replace("\\", "_")

    # Nur erlaubte Zeichen behalten
    filename = re.sub(r'[^a-zA-Z0-9_\-.]', '_', filename)

    # Maximale Länge
    if len(filename) > 255:
        filename = filename[:255]

    return filename


# ============================================================================
# ZEITREIHEN-UTILITIES
# ============================================================================
def infer_resolution_minutes(idx: pd.DatetimeIndex) -> Optional[int]:
    """
    Ermittelt die Auflösung einer Zeitreihe in Minuten.

    Args:
        idx: DatetimeIndex der Zeitreihe

    Returns:
        Geschätzte Auflösung in Minuten oder None
    """
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


def calculate_missing_quote(
    idx: pd.DatetimeIndex,
    resolution_minutes: Optional[int]
) -> float:
    """
    Berechnet den Anteil fehlender Datenpunkte.

    Args:
        idx: DatetimeIndex der Zeitreihe
        resolution_minutes: Erwartete Auflösung in Minuten

    Returns:
        Anteil fehlender Punkte (0.0 - 1.0)
    """
    if resolution_minutes is None or len(idx) < 2:
        return 0.0

    start, end = idx.min(), idx.max()
    expected = int(((end - start).total_seconds() / 60.0) / float(resolution_minutes)) + 1
    expected = max(expected, 1)
    actual = int(idx.nunique())
    missing = max(expected - actual, 0)

    return float(missing) / float(expected)
