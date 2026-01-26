# tests/test_analytics.py
"""
Tests für report_builder/analytics.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(__file__).rsplit('/tests', 1)[0])

from report_builder.analytics import (
    prepare_raw_power_data,
    aggregate_to_interval,
    compute_peak_events,
    compute_unbalance_module,
    compute_blk_metrics,
    compute_cap,
    tariff_for_util_hours,
    compute_scenario,
    build_recommendations,
)
from report_builder.config import Tariffs
from report_builder.models import BlkResult, PeakEventsResult, UnbalanceResult


class TestPrepareRawPowerData:
    """Tests für Rohdaten-Aufbereitung"""

    def test_single_column_kw(self, simple_df):
        """Einzelne Leistungsspalte in kW"""
        result = prepare_raw_power_data(
            d0=simple_df,
            timestamp_col='timestamp',
            power_col='power_kw',
            power_cols=None,
            power_unit='kW',
            pf_cols=None,
        )

        df = result['df']
        assert not df.empty
        assert 'p_kw' in df.columns
        assert df['p_kw'].notna().all()

    def test_single_column_watts(self, simple_df):
        """Einzelne Leistungsspalte in Watt"""
        # Konvertiere zu Watt
        df_w = simple_df.copy()
        df_w['power_w'] = df_w['power_kw'] * 1000

        result = prepare_raw_power_data(
            d0=df_w,
            timestamp_col='timestamp',
            power_col='power_w',
            power_cols=None,
            power_unit='W',
            pf_cols=None,
        )

        df = result['df']
        assert not df.empty
        # Nach Konvertierung sollte es in kW sein
        assert df['p_kw'].max() < 1000  # Nicht in Watt-Größenordnung

    def test_three_phase_data(self, three_phase_df):
        """3-Phasen-Daten"""
        result = prepare_raw_power_data(
            d0=three_phase_df,
            timestamp_col='timestamp',
            power_col=None,
            power_cols=['power_1', 'power_2', 'power_3'],
            power_unit='kW',
            pf_cols=['cosphi_1', 'cosphi_2', 'cosphi_3'],
        )

        df = result['df']
        assert not df.empty
        assert 'p1_kw' in df.columns
        assert 'p2_kw' in df.columns
        assert 'p3_kw' in df.columns
        assert 'p_kw' in df.columns
        assert 'c1' in df.columns

    def test_auto_unit_detection(self, simple_df):
        """Automatische Einheitenerkennung"""
        # Multipliziere mit 1000 für Watt-Bereich
        df_w = simple_df.copy()
        df_w['power_kw'] = df_w['power_kw'] * 1000

        result = prepare_raw_power_data(
            d0=df_w,
            timestamp_col='timestamp',
            power_col='power_kw',
            power_cols=None,
            power_unit='Auto',  # Auto-Erkennung
            pf_cols=None,
        )

        df = result['df']
        # Sollte automatisch nach kW konvertieren
        assert df['p_kw'].median() < 200  # Sollte kW sein, nicht W


class TestAggregateToInterval:
    """Tests für Aggregation"""

    def test_aggregate_1min_to_15min(self):
        """1-Minuten zu 15-Minuten Aggregation"""
        # Erstelle 1-Minuten-Daten
        timestamps = pd.date_range('2024-01-01', periods=60, freq='1min')
        df = pd.DataFrame({
            'p_kw': np.random.uniform(10, 20, 60),
        }, index=timestamps)

        result = aggregate_to_interval(df, minutes=15)

        # Sollte 4 Blöcke haben (60min / 15min)
        assert len(result) == 4
        assert 'p_kw' in result.columns

    def test_aggregate_preserves_phase_columns(self):
        """Phasen-Spalten sollten erhalten bleiben"""
        timestamps = pd.date_range('2024-01-01', periods=30, freq='1min')
        df = pd.DataFrame({
            'p_kw': np.random.uniform(10, 20, 30),
            'p1_kw': np.random.uniform(3, 7, 30),
            'p2_kw': np.random.uniform(3, 7, 30),
            'p3_kw': np.random.uniform(3, 7, 30),
        }, index=timestamps)

        result = aggregate_to_interval(df, minutes=15)

        assert 'p1_kw' in result.columns
        assert 'p2_kw' in result.columns
        assert 'p3_kw' in result.columns


class TestComputePeakEvents:
    """Tests für Peak-Event-Erkennung"""

    def test_no_peaks(self):
        """Keine Peaks über Cap"""
        timestamps = pd.date_range('2024-01-01', periods=10, freq='15min')
        df = pd.DataFrame({
            'p_kw': [10.0] * 10,  # Konstant unter Cap
        }, index=timestamps)

        result = compute_peak_events(df, cap_kw=50.0)

        assert result.n_events == 0
        assert result.events_df.empty

    def test_single_peak(self):
        """Einzelner Peak"""
        timestamps = pd.date_range('2024-01-01', periods=10, freq='15min')
        df = pd.DataFrame({
            'p_kw': [10, 10, 10, 60, 10, 10, 10, 10, 10, 10],  # Ein Peak bei Index 3
        }, index=timestamps)

        result = compute_peak_events(df, cap_kw=50.0)

        assert result.n_events == 1
        assert result.avg_duration_min == 15.0  # Ein 15-min Block
        assert result.max_shift_kw == pytest.approx(10.0)  # 60 - 50 = 10

    def test_multiple_peaks(self):
        """Mehrere Peaks"""
        timestamps = pd.date_range('2024-01-01', periods=20, freq='15min')
        power = [10] * 20
        power[3] = 60  # Peak 1
        power[10] = 70  # Peak 2
        power[11] = 65  # Peak 2 (fortgesetzt)

        df = pd.DataFrame({'p_kw': power}, index=timestamps)
        result = compute_peak_events(df, cap_kw=50.0)

        assert result.n_events == 2

    def test_peak_problem_type_short(self):
        """Kurzspitzen-Erkennung"""
        timestamps = pd.date_range('2024-01-01', periods=100, freq='15min')
        power = [20.0] * 100
        # Viele kurze einzelne Peaks
        for i in range(0, 100, 10):
            power[i] = 60.0

        df = pd.DataFrame({'p_kw': power}, index=timestamps)
        result = compute_peak_events(df, cap_kw=50.0)

        assert "Kurzspitzen" in result.peak_problem_type


class TestComputeUnbalanceModule:
    """Tests für Phasen-Unwucht-Analyse"""

    def test_no_phase_data(self, simple_df):
        """Keine Phasen-Daten verfügbar"""
        timestamps = pd.date_range('2024-01-01', periods=10, freq='15min')
        df = pd.DataFrame({
            'p_kw': [10.0] * 10,
        }, index=timestamps)

        result = compute_unbalance_module(df)

        assert result.available is False

    def test_balanced_phases(self):
        """Symmetrische Phasen"""
        timestamps = pd.date_range('2024-01-01', periods=10, freq='15min')
        df = pd.DataFrame({
            'p1_kw': [10.0] * 10,
            'p2_kw': [10.0] * 10,
            'p3_kw': [10.0] * 10,
        }, index=timestamps)

        result = compute_unbalance_module(df, threshold_kw=3.0)

        assert result.available is True
        assert result.share_over == 0.0  # Keine Unwucht
        assert result.max_unbalance_kw == 0.0

    def test_unbalanced_phases(self):
        """Asymmetrische Phasen"""
        timestamps = pd.date_range('2024-01-01', periods=10, freq='15min')
        df = pd.DataFrame({
            'p1_kw': [10.0] * 10,
            'p2_kw': [10.0] * 10,
            'p3_kw': [20.0] * 10,  # L3 dominiert
        }, index=timestamps)

        result = compute_unbalance_module(df, threshold_kw=3.0)

        assert result.available is True
        assert result.share_over > 0.0  # Unwucht erkannt
        assert result.dominant_phase_name == "L3"


class TestComputeCap:
    """Tests für Cap-Berechnung"""

    def test_bronze_cap(self):
        """Bronze (P95) Cap"""
        power = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        cap, label = compute_cap(power, "Bronze", None, "")

        assert cap == pytest.approx(95.5, rel=0.1)  # P95
        assert "Bronze" in label
        assert "P95" in label

    def test_silber_cap(self):
        """Silber (P90) Cap"""
        power = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        cap, label = compute_cap(power, "Silber", None, "")

        assert cap == pytest.approx(91.0, rel=0.1)  # P90
        assert "Silber" in label

    def test_gold_cap(self):
        """Gold (P85) Cap"""
        power = pd.Series([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        cap, label = compute_cap(power, "Gold", None, "")

        assert cap == pytest.approx(86.5, rel=0.1)  # P85
        assert "Gold" in label

    def test_manual_cap(self):
        """Manueller Cap"""
        power = pd.Series([10, 20, 30, 40, 50])
        cap, label = compute_cap(power, "Manuell", 25.0, "25 kW")

        assert cap == 25.0
        assert "Manuell" in label

    def test_manual_cap_missing_value(self):
        """Manueller Cap ohne Wert sollte Fehler werfen"""
        power = pd.Series([10, 20, 30])

        with pytest.raises(ValueError):
            compute_cap(power, "Manuell", None, "")


class TestTariffForUtilHours:
    """Tests für Tarif-Ermittlung"""

    def test_low_utilization(self):
        """Niedrige Benutzungsstunden -> niedriger Tarif"""
        tariffs = Tariffs(switch_hours=2500)
        work_ct, demand_eur, label = tariff_for_util_hours(tariffs, 2000.0)

        assert work_ct == tariffs.work_ct_low
        assert demand_eur == tariffs.demand_eur_kw_a_low
        assert "< 2500" in label

    def test_high_utilization(self):
        """Hohe Benutzungsstunden -> hoher Tarif"""
        tariffs = Tariffs(switch_hours=2500)
        work_ct, demand_eur, label = tariff_for_util_hours(tariffs, 3000.0)

        assert work_ct == tariffs.work_ct_high
        assert demand_eur == tariffs.demand_eur_kw_a_high
        assert "> 2500" in label


class TestComputeScenario:
    """Tests für Szenario-Berechnung"""

    def test_scenario_savings(self):
        """Einsparungen berechnen"""
        timestamps = pd.date_range('2024-01-01', periods=100, freq='15min')
        df = pd.DataFrame({
            'p_kw': np.random.uniform(20, 100, 100),
        }, index=timestamps)

        scenario = compute_scenario(
            name="Test",
            cap_kw=50.0,
            cap_label="Test Cap",
            annual_energy_kwh=100000.0,
            peak_before_kw=100.0,
            tariffs=Tariffs(),
            df_15=df,
            block_h=0.25,
        )

        assert scenario.peak_after_kw == 50.0  # Cap wird angewendet
        assert scenario.savings_eur >= 0  # Sollte Einsparungen geben


class TestBuildRecommendations:
    """Tests für Empfehlungs-Generierung"""

    def test_generates_recommendations(self):
        """Sollte mindestens eine Empfehlung generieren"""
        # Mock-Ergebnisse erstellen
        mod1 = PeakEventsResult(
            n_events=15,
            avg_duration_min=15.0,
            max_duration_min=30.0,
            max_shift_kw=20.0,
            top_months="2024-01: 5",
            interpretation="Test",
            events_df=pd.DataFrame({
                'duration_min': [15.0] * 15,
                'max_shift_kw': [20.0] * 15,
            }),
            peak_problem_type="Kurzspitzen",
        )

        mod2 = UnbalanceResult(available=False)
        blk = BlkResult(available=False)

        recs = build_recommendations(
            mod1=mod1,
            mod2=mod2,
            blk=blk,
            util_hours_before=2000.0,
            util_hours_after=2800.0,
            p_verschiebbar_kw=20.0,
            p_gesamt_kw=100.0,
            tariffs=Tariffs(),
        )

        assert len(recs) > 0

    def test_quick_win_priority(self):
        """Quick Wins sollten am Anfang stehen"""
        mod1 = PeakEventsResult(
            n_events=20,
            avg_duration_min=15.0,
            max_duration_min=15.0,
            max_shift_kw=10.0,
            top_months="2024-01: 10",
            interpretation="Viele kurze Peaks",
            events_df=pd.DataFrame({
                'duration_min': [15.0] * 20,
                'max_shift_kw': [10.0] * 20,
            }),
            peak_problem_type="Kurzspitzen",
        )

        mod2 = UnbalanceResult(available=False)
        blk = BlkResult(available=False)

        recs = build_recommendations(
            mod1=mod1,
            mod2=mod2,
            blk=blk,
            util_hours_before=2000.0,
            util_hours_after=2000.0,
            p_verschiebbar_kw=5.0,
            p_gesamt_kw=100.0,
            tariffs=Tariffs(),
        )

        # Quick Wins sollten zuerst kommen
        if any(r.priority == "Quick Win" for r in recs):
            quick_wins = [r for r in recs if r.priority == "Quick Win"]
            other = [r for r in recs if r.priority != "Quick Win"]
            # Check that quick wins are before others
            quick_win_indices = [recs.index(r) for r in quick_wins]
            other_indices = [recs.index(r) for r in other]
            if quick_win_indices and other_indices:
                assert max(quick_win_indices) < min(other_indices)
