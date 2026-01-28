# tests/test_pdf_builder.py
"""
Tests für report_builder/pdf_builder.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from report_builder import (
    build_pdf_report,
    PROFILE_LITE,
    PROFILE_STANDARD,
    PROFILE_PRO,
    Tariffs,
)


class TestBuildPdfReport:
    """Tests für PDF-Generierung"""

    def test_generate_simple_report(self, simple_df, temp_output_dir):
        """Einfacher Report mit einer Leistungsspalte"""
        out_path = temp_output_dir / "test_simple.pdf"

        build_pdf_report(
            df=simple_df,
            out_path=out_path,
            timestamp_col='timestamp',
            power_col='power_kw',
            power_unit='kW',
            source_name='Test',
            site_name='Test Site',
            reduction_goal='Silber',
            tariffs=Tariffs(),
            profile=PROFILE_LITE,
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_generate_three_phase_report(self, three_phase_df, temp_output_dir):
        """Report mit 3-Phasen-Daten"""
        out_path = temp_output_dir / "test_3phase.pdf"

        build_pdf_report(
            df=three_phase_df,
            out_path=out_path,
            timestamp_col='timestamp',
            power_col=None,
            power_cols=['power_1', 'power_2', 'power_3'],
            power_unit='kW',
            pf_cols=['cosphi_1', 'cosphi_2', 'cosphi_3'],
            source_name='Test 3-Phase',
            site_name='Test Site',
            reduction_goal='Bronze',
            tariffs=Tariffs(),
            include_reactive=True,
            profile=PROFILE_STANDARD,
        )

        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_all_profiles(self, simple_df, temp_output_dir):
        """Alle Profile generieren"""
        for profile, name in [
            (PROFILE_LITE, "lite"),
            (PROFILE_STANDARD, "standard"),
            (PROFILE_PRO, "pro"),
        ]:
            out_path = temp_output_dir / f"test_{name}.pdf"

            build_pdf_report(
                df=simple_df,
                out_path=out_path,
                timestamp_col='timestamp',
                power_col='power_kw',
                power_unit='kW',
                source_name=f'Test {name}',
                reduction_goal='Silber',
                tariffs=Tariffs(),
                profile=profile,
            )

            assert out_path.exists(), f"Profil {name} nicht erstellt"

    def test_invalid_timestamp_column(self, simple_df, temp_output_dir):
        """Ungültige Zeitstempel-Spalte sollte Fehler werfen"""
        out_path = temp_output_dir / "test_invalid.pdf"

        with pytest.raises(KeyError):
            build_pdf_report(
                df=simple_df,
                out_path=out_path,
                timestamp_col='nonexistent_column',
                power_col='power_kw',
                power_unit='kW',
                tariffs=Tariffs(),
            )

    def test_empty_dataframe(self, temp_output_dir):
        """Leerer DataFrame sollte Fehler werfen"""
        out_path = temp_output_dir / "test_empty.pdf"
        empty_df = pd.DataFrame({'timestamp': [], 'power_kw': []})

        with pytest.raises(ValueError):
            build_pdf_report(
                df=empty_df,
                out_path=out_path,
                timestamp_col='timestamp',
                power_col='power_kw',
                power_unit='kW',
                tariffs=Tariffs(),
            )

    def test_manual_cap(self, simple_df, temp_output_dir):
        """Manueller Cap-Wert"""
        out_path = temp_output_dir / "test_manual_cap.pdf"

        build_pdf_report(
            df=simple_df,
            out_path=out_path,
            timestamp_col='timestamp',
            power_col='power_kw',
            power_unit='kW',
            reduction_goal='Manuell',
            manual_cap_kw=30.0,
            manual_value='30 kW',
            tariffs=Tariffs(),
            profile=PROFILE_LITE,
        )

        assert out_path.exists()

    def test_progress_callback(self, simple_df, temp_output_dir):
        """Progress-Callback wird aufgerufen"""
        out_path = temp_output_dir / "test_progress.pdf"
        progress_steps = []

        def callback(step, percent):
            progress_steps.append((step, percent))

        build_pdf_report(
            df=simple_df,
            out_path=out_path,
            timestamp_col='timestamp',
            power_col='power_kw',
            power_unit='kW',
            tariffs=Tariffs(),
            profile=PROFILE_LITE,
            progress_callback=callback,
        )

        assert len(progress_steps) > 0
        # Sollte mit 100% enden
        assert progress_steps[-1][1] == 100


class TestReportContent:
    """Tests für Report-Inhalt"""

    def test_lite_profile_page_count(self, simple_df, temp_output_dir):
        """Lite-Profil sollte weniger Seiten haben"""
        lite_path = temp_output_dir / "lite.pdf"
        standard_path = temp_output_dir / "standard.pdf"

        build_pdf_report(
            df=simple_df,
            out_path=lite_path,
            timestamp_col='timestamp',
            power_col='power_kw',
            tariffs=Tariffs(),
            profile=PROFILE_LITE,
        )

        build_pdf_report(
            df=simple_df,
            out_path=standard_path,
            timestamp_col='timestamp',
            power_col='power_kw',
            tariffs=Tariffs(),
            profile=PROFILE_STANDARD,
        )

        # Lite sollte kleiner sein als Standard
        assert lite_path.stat().st_size < standard_path.stat().st_size
