# tests/test_export.py
"""
Tests für Export-Funktionen (Excel, PowerPoint, CSV)
"""
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from report_builder.export import (
    export_to_excel,
    export_analysis_to_csv,
    ExportConfig,
)


# ============================================================================
# FIXTURES
# ============================================================================
@pytest.fixture
def sample_df():
    """Sample DataFrame für Export-Tests"""
    dates = pd.date_range(start="2024-01-01", periods=100, freq="15min")
    return pd.DataFrame({
        "timestamp": dates,
        "power_kw": np.random.uniform(50, 150, 100),
        "p1_kw": np.random.uniform(15, 50, 100),
        "p2_kw": np.random.uniform(15, 50, 100),
        "p3_kw": np.random.uniform(15, 50, 100),
    }, index=dates)


@pytest.fixture
def sample_analysis():
    """Sample Analyse-Ergebnisse"""
    return {
        "kpis": {
            "Max. Leistung": "145.3 kW",
            "Ø Leistung": "87.2 kW",
            "Auslastung": "60.0%"
        },
        "peaks": [
            {"timestamp": "2024-01-01 10:00", "power_kw": 145.3, "duration": "15 min", "type": "kurz"},
            {"timestamp": "2024-01-02 14:00", "power_kw": 142.1, "duration": "30 min", "type": "mittel"},
        ],
        "scenarios": [
            {"name": "Bronze", "target_cap": 120.0, "savings": 2500.0, "reduction_pct": 17.4},
            {"name": "Silber", "target_cap": 110.0, "savings": 3800.0, "reduction_pct": 24.3},
        ],
        "recommendations": [
            {"priority": "Hoch", "category": "Peak-Shaving", "text": "Lastspitzen vermeiden", "savings": "3.800 €/a"},
            {"priority": "Mittel", "category": "Effizienz", "text": "LED-Beleuchtung", "savings": "500 €/a"},
        ]
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Temporäres Verzeichnis für Output"""
    return tmp_path


# ============================================================================
# EXCEL EXPORT TESTS
# ============================================================================
class TestExportToExcel:

    def test_basic_excel_export(self, sample_df, sample_analysis, temp_dir):
        """Grundlegender Excel-Export"""
        pytest.importorskip("openpyxl")

        output_path = temp_dir / "test_report.xlsx"

        result = export_to_excel(
            df=sample_df,
            analysis_results=sample_analysis,
            output_path=output_path
        )

        assert result.exists()
        assert result.suffix == ".xlsx"
        assert result.stat().st_size > 0

    def test_excel_with_config(self, sample_df, sample_analysis, temp_dir):
        """Excel-Export mit Konfiguration"""
        pytest.importorskip("openpyxl")

        output_path = temp_dir / "test_config.xlsx"
        config = ExportConfig(
            company_name="Test GmbH",
            site_name="Werk 1",
            report_date="01.01.2024"
        )

        result = export_to_excel(
            df=sample_df,
            analysis_results=sample_analysis,
            output_path=output_path,
            config=config
        )

        assert result.exists()

    def test_excel_sheets_exist(self, sample_df, sample_analysis, temp_dir):
        """Excel hat alle erwarteten Sheets"""
        openpyxl = pytest.importorskip("openpyxl")

        output_path = temp_dir / "test_sheets.xlsx"

        export_to_excel(
            df=sample_df,
            analysis_results=sample_analysis,
            output_path=output_path
        )

        wb = openpyxl.load_workbook(output_path)
        sheet_names = wb.sheetnames

        assert "Übersicht" in sheet_names
        assert "Rohdaten" in sheet_names

        wb.close()

    def test_excel_without_raw_data(self, sample_df, sample_analysis, temp_dir):
        """Excel ohne Rohdaten"""
        openpyxl = pytest.importorskip("openpyxl")

        output_path = temp_dir / "test_no_raw.xlsx"
        config = ExportConfig(include_raw_data=False)

        export_to_excel(
            df=sample_df,
            analysis_results=sample_analysis,
            output_path=output_path,
            config=config
        )

        wb = openpyxl.load_workbook(output_path)
        # Rohdaten-Sheet sollte trotzdem existieren, aber keine Daten haben
        wb.close()

    def test_excel_progress_callback(self, sample_df, sample_analysis, temp_dir):
        """Progress-Callback wird aufgerufen"""
        pytest.importorskip("openpyxl")

        output_path = temp_dir / "test_progress.xlsx"
        progress_calls = []

        def track_progress(pct, msg):
            progress_calls.append((pct, msg))

        export_to_excel(
            df=sample_df,
            analysis_results=sample_analysis,
            output_path=output_path,
            progress_callback=track_progress
        )

        assert len(progress_calls) > 0
        assert progress_calls[-1][0] == 1.0  # Letzter Call sollte 100% sein


# ============================================================================
# CSV EXPORT TESTS
# ============================================================================
class TestExportToCSV:

    def test_basic_csv_export(self, sample_df, temp_dir):
        """Grundlegender CSV-Export"""
        output_path = temp_dir / "test_export.csv"

        result = export_analysis_to_csv(sample_df, output_path)

        assert result.exists()
        assert result.suffix == ".csv"

    def test_csv_german_format(self, sample_df, temp_dir):
        """CSV mit deutschem Format (Semikolon, Komma-Dezimal)"""
        output_path = temp_dir / "test_german.csv"

        export_analysis_to_csv(sample_df, output_path)

        content = output_path.read_text(encoding="utf-8-sig")
        # Semikolon als Separator
        assert ";" in content

    def test_csv_without_index(self, sample_df, temp_dir):
        """CSV ohne Index"""
        output_path = temp_dir / "test_no_index.csv"

        export_analysis_to_csv(sample_df, output_path, include_index=False)

        df_read = pd.read_csv(output_path, sep=";", decimal=",")
        # Sollte keine Index-Spalte haben
        assert "Unnamed: 0" not in df_read.columns


# ============================================================================
# EXPORT CONFIG TESTS
# ============================================================================
class TestExportConfig:

    def test_default_config(self):
        """Standard-Konfiguration"""
        config = ExportConfig()

        assert config.include_raw_data == True
        assert config.include_analysis == True
        assert config.include_charts == True

    def test_custom_config(self):
        """Benutzerdefinierte Konfiguration"""
        config = ExportConfig(
            include_raw_data=False,
            company_name="Test AG",
            site_name="Hauptwerk"
        )

        assert config.include_raw_data == False
        assert config.company_name == "Test AG"
        assert config.site_name == "Hauptwerk"
