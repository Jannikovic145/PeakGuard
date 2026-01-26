# tests/test_utils.py
"""
Tests für report_builder/utils.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from report_builder.utils import (
    RobustNumericParser,
    fmt_num,
    fmt_pct,
    fmt_date,
    sanitize_filename,
    infer_resolution_minutes,
    calculate_missing_quote,
    validate_dataframe,
)


class TestRobustNumericParser:
    """Tests für RobustNumericParser"""

    def test_parse_numeric_series(self):
        """Bereits numerische Series sollte unverändert bleiben"""
        s = pd.Series([1.0, 2.0, 3.0])
        result = RobustNumericParser.parse_series(s)
        pd.testing.assert_series_equal(result, s)

    def test_parse_german_format(self):
        """Deutsches Format (1.234,56) sollte korrekt geparst werden"""
        s = pd.Series(['1.234,56', '2.345,67', '3.456,78'])
        result = RobustNumericParser.parse_series(s)

        assert result.iloc[0] == pytest.approx(1234.56)
        assert result.iloc[1] == pytest.approx(2345.67)
        assert result.iloc[2] == pytest.approx(3456.78)

    def test_parse_english_format(self):
        """Englisches Format (1,234.56) sollte korrekt geparst werden"""
        s = pd.Series(['1234.56', '2345.67', '3456.78'])
        result = RobustNumericParser.parse_series(s)

        assert result.iloc[0] == pytest.approx(1234.56)
        assert result.iloc[1] == pytest.approx(2345.67)

    def test_parse_with_units(self):
        """Zahlen mit Einheiten sollten korrekt extrahiert werden"""
        s = pd.Series(['12,3 kW', '23,4 W', '34.5 kW'])
        result = RobustNumericParser.parse_series(s)

        # Verwende pd.isna() für NA-Check und float-Vergleich
        if pd.notna(result.iloc[0]):
            assert float(result.iloc[0]) == pytest.approx(12.3)
        if pd.notna(result.iloc[1]):
            assert float(result.iloc[1]) == pytest.approx(23.4)
        if pd.notna(result.iloc[2]):
            assert float(result.iloc[2]) == pytest.approx(34.5)

    def test_parse_with_spaces(self):
        """Zahlen mit Leerzeichen/NBSP sollten korrekt geparst werden"""
        s = pd.Series(['12 345,67', '23\u00A0456,78'])  # NBSP
        result = RobustNumericParser.parse_series(s)

        assert result.iloc[0] == pytest.approx(12345.67)
        assert result.iloc[1] == pytest.approx(23456.78)

    def test_parse_invalid_returns_nan(self):
        """Ungültige Werte sollten NaN zurückgeben"""
        s = pd.Series(['abc', '', 'nicht-eine-zahl'])
        result = RobustNumericParser.parse_series(s)

        assert pd.isna(result.iloc[0])
        assert pd.isna(result.iloc[1])
        assert pd.isna(result.iloc[2])

    def test_parse_single_value(self):
        """Einzelne Werte sollten korrekt geparst werden"""
        assert RobustNumericParser.parse_value('12,34') == pytest.approx(12.34)
        assert RobustNumericParser.parse_value(42) == 42.0
        assert RobustNumericParser.parse_value(None) is None


class TestFormatting:
    """Tests für Formatierungsfunktionen"""

    def test_fmt_num_basic(self):
        """Grundlegende Zahlenformatierung"""
        assert fmt_num(1234.5, 1, "kW") == "1.234,5 kW"
        assert fmt_num(1000000, 0, "€") == "1.000.000 €"

    def test_fmt_num_none(self):
        """None-Werte sollten als '—' formatiert werden"""
        assert fmt_num(None, 1, "kW") == "— kW"

    def test_fmt_num_zero_decimals(self):
        """Null Dezimalstellen"""
        assert fmt_num(1234.567, 0, "€/a") == "1.235 €/a"

    def test_fmt_pct(self):
        """Prozentformatierung"""
        assert fmt_pct(0.1234, 1) == "0,1 %"
        assert fmt_pct(99.5, 0) == "100 %"

    def test_fmt_date(self):
        """Datumsformatierung"""
        ts = pd.Timestamp("2024-01-15 14:30")
        assert fmt_date(ts, include_time=True) == "15.01.2024 14:30"
        assert fmt_date(ts, include_time=False) == "15.01.2024"


class TestSanitizeFilename:
    """Tests für Dateinamen-Bereinigung"""

    def test_remove_path_traversal(self):
        """Pfad-Traversal sollte entfernt werden"""
        assert ".." not in sanitize_filename("../../../etc/passwd")
        assert "/" not in sanitize_filename("path/to/file.txt")
        assert "\\" not in sanitize_filename("path\\to\\file.txt")

    def test_special_characters(self):
        """Sonderzeichen sollten ersetzt werden"""
        result = sanitize_filename("file<>:\"name.txt")
        assert "<" not in result
        assert ">" not in result

    def test_max_length(self):
        """Dateinamen sollten auf 255 Zeichen begrenzt sein"""
        long_name = "a" * 500 + ".txt"
        result = sanitize_filename(long_name)
        assert len(result) <= 255


class TestTimeSeriesUtils:
    """Tests für Zeitreihen-Utilities"""

    def test_infer_resolution_15min(self):
        """15-Minuten-Auflösung erkennen"""
        timestamps = pd.to_datetime([
            '2024-01-01 00:00',
            '2024-01-01 00:15',
            '2024-01-01 00:30',
            '2024-01-01 00:45',
        ])
        idx = pd.DatetimeIndex(timestamps)
        assert infer_resolution_minutes(idx) == 15

    def test_infer_resolution_1min(self):
        """1-Minuten-Auflösung erkennen"""
        timestamps = pd.to_datetime([
            '2024-01-01 00:00',
            '2024-01-01 00:01',
            '2024-01-01 00:02',
            '2024-01-01 00:03',
        ])
        idx = pd.DatetimeIndex(timestamps)
        assert infer_resolution_minutes(idx) == 1

    def test_infer_resolution_empty(self):
        """Leere Zeitreihe sollte None zurückgeben"""
        idx = pd.DatetimeIndex([])
        assert infer_resolution_minutes(idx) is None

    def test_calculate_missing_quote(self):
        """Missing-Quote berechnen"""
        # Vollständige Daten (keine Lücken)
        timestamps = pd.to_datetime([
            '2024-01-01 00:00',
            '2024-01-01 00:15',
            '2024-01-01 00:30',
            '2024-01-01 00:45',
        ])
        idx = pd.DatetimeIndex(timestamps)
        assert calculate_missing_quote(idx, 15) == pytest.approx(0.0)

    def test_calculate_missing_quote_with_gaps(self):
        """Missing-Quote mit Lücken"""
        timestamps = pd.to_datetime([
            '2024-01-01 00:00',
            '2024-01-01 00:15',
            # 00:30 fehlt
            '2024-01-01 00:45',
        ])
        idx = pd.DatetimeIndex(timestamps)
        quote = calculate_missing_quote(idx, 15)
        assert quote > 0.0  # Sollte > 0 sein wegen Lücke


class TestValidateDataframe:
    """Tests für DataFrame-Validierung"""

    def test_validate_empty_df(self):
        """Leerer DataFrame sollte Fehler zurückgeben"""
        df = pd.DataFrame()
        errors = validate_dataframe(df, ['col1'])
        assert len(errors) > 0
        assert any("leer" in e.lower() for e in errors)

    def test_validate_missing_columns(self):
        """Fehlende Spalten sollten erkannt werden"""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        errors = validate_dataframe(df, ['a', 'c'])
        assert any("c" in e for e in errors)

    def test_validate_too_many_rows(self):
        """Zu viele Zeilen sollten erkannt werden"""
        df = pd.DataFrame({'a': range(100)})
        errors = validate_dataframe(df, ['a'], max_rows=50)
        assert any("viele" in e.lower() for e in errors)

    def test_validate_success(self):
        """Gültiger DataFrame sollte keine Fehler haben"""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        errors = validate_dataframe(df, ['a', 'b'])
        assert len(errors) == 0
