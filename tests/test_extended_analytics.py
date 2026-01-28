# tests/test_extended_analytics.py
"""
Tests für erweiterte Analysen (CO2, ROI-Rechner, Lastprognose)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from report_builder.extended_analytics import (
    compute_co2_analysis,
    compute_battery_roi,
    compute_load_forecast,
    compute_compensation_roi,
    BatterySpec,
    CO2Result,
    ROIResult,
)


# ============================================================================
# FIXTURES
# ============================================================================
@pytest.fixture
def sample_power_df():
    """DataFrame mit Leistungsdaten für ein Jahr"""
    dates = pd.date_range(start="2024-01-01", periods=365*24*4, freq="15min")
    np.random.seed(42)

    # Simuliere typischen Tagesgang
    power = np.zeros(len(dates))
    for i, ts in enumerate(dates):
        hour = ts.hour
        base = 50.0
        if 6 <= hour < 22:
            day_factor = 1.5
        else:
            day_factor = 0.5
        power[i] = base * day_factor + np.random.normal(0, 10)

    df = pd.DataFrame({"p_kw": power}, index=dates)
    return df


@pytest.fixture
def short_power_df():
    """Kurzer DataFrame (1 Woche)"""
    dates = pd.date_range(start="2024-01-01", periods=7*24*4, freq="15min")
    np.random.seed(42)
    power = 50 + np.random.normal(0, 10, len(dates))
    return pd.DataFrame({"p_kw": power}, index=dates)


# ============================================================================
# CO2-ANALYSE TESTS
# ============================================================================
class TestComputeCO2Analysis:

    def test_basic_co2_calculation(self, sample_power_df):
        """Grundlegende CO2-Berechnung"""
        result = compute_co2_analysis(sample_power_df, power_col="p_kw")

        assert result.available
        assert result.total_consumption_kwh > 0
        assert result.total_co2_kg > 0
        assert 300 < result.co2_factor_g_kwh < 500  # Realistischer Bereich

    def test_manual_co2_factor(self, sample_power_df):
        """Manueller CO2-Faktor"""
        result = compute_co2_analysis(
            sample_power_df,
            power_col="p_kw",
            co2_factor_manual=30.0  # Ökostrom
        )

        assert result.available
        # Bei niedrigem Faktor sollte CO2 geringer sein
        assert result.co2_factor_g_kwh == pytest.approx(30.0, rel=0.1)

    def test_empty_dataframe(self):
        """Leerer DataFrame"""
        result = compute_co2_analysis(pd.DataFrame(), power_col="p_kw")
        assert not result.available

    def test_missing_column(self, sample_power_df):
        """Fehlende Leistungsspalte"""
        result = compute_co2_analysis(sample_power_df, power_col="nonexistent")
        assert not result.available

    def test_monthly_breakdown(self, sample_power_df):
        """Monatliche Aufschlüsselung vorhanden"""
        result = compute_co2_analysis(sample_power_df, power_col="p_kw")

        assert result.available
        assert len(result.monthly_co2) > 0

    def test_potential_savings(self, sample_power_df):
        """Einsparpotenzial wird berechnet"""
        result = compute_co2_analysis(sample_power_df, power_col="p_kw")

        assert result.available
        assert result.potential_savings_kg >= 0


# ============================================================================
# ROI-RECHNER TESTS
# ============================================================================
class TestComputeBatteryROI:

    def test_basic_roi_calculation(self):
        """Grundlegende ROI-Berechnung"""
        result = compute_battery_roi(
            annual_peak_cost=10000.0,
            current_peak_kw=100.0,
            target_peak_kw=80.0,
            peak_price_eur_kw_a=100.0
        )

        assert result.available
        assert result.investment_cost > 0
        assert result.annual_savings > 0
        assert result.payback_years > 0

    def test_no_reduction_possible(self):
        """Keine Peak-Reduktion möglich"""
        result = compute_battery_roi(
            annual_peak_cost=10000.0,
            current_peak_kw=80.0,
            target_peak_kw=100.0,  # Ziel > Aktuell
            peak_price_eur_kw_a=100.0
        )

        assert not result.available

    def test_custom_battery_spec(self):
        """Benutzerdefinierte Batterie-Spezifikation"""
        battery = BatterySpec(
            capacity_kwh=100.0,
            power_kw=50.0,
            efficiency=0.92,
            cost_per_kwh=400.0
        )

        result = compute_battery_roi(
            annual_peak_cost=10000.0,
            current_peak_kw=100.0,
            target_peak_kw=50.0,
            peak_price_eur_kw_a=100.0,
            battery_spec=battery
        )

        assert result.available
        assert result.battery_size_kwh == 100.0
        assert result.battery_power_kw == 50.0

    def test_npv_positive_for_good_investment(self):
        """NPV positiv bei guter Investition"""
        result = compute_battery_roi(
            annual_peak_cost=50000.0,  # Hohe Kosten
            current_peak_kw=200.0,
            target_peak_kw=100.0,  # 50% Reduktion
            peak_price_eur_kw_a=250.0
        )

        assert result.available
        # Bei hohen Einsparungen sollte NPV positiv sein
        if result.payback_years < 10:
            assert result.npv_10_years > 0

    def test_yearly_cashflows(self):
        """Jährliche Cashflows werden berechnet"""
        result = compute_battery_roi(
            annual_peak_cost=10000.0,
            current_peak_kw=100.0,
            target_peak_kw=80.0,
            peak_price_eur_kw_a=100.0
        )

        assert result.available
        assert len(result.yearly_cashflows) == 15  # Default Lebensdauer


# ============================================================================
# LASTPROGNOSE TESTS
# ============================================================================
class TestComputeLoadForecast:

    def test_basic_forecast(self, sample_power_df):
        """Grundlegende Lastprognose"""
        result = compute_load_forecast(sample_power_df, power_col="p_kw")

        assert result.available
        assert result.forecast_horizon_days == 7
        assert len(result.forecast_values) > 0
        assert result.predicted_peak_kw > 0

    def test_insufficient_data(self, short_power_df):
        """Zu wenig Daten für Prognose"""
        # 7 Tage sind Minimum, daher sollte es gerade noch funktionieren
        result = compute_load_forecast(short_power_df, power_col="p_kw")
        assert result.available

    def test_confidence_intervals(self, sample_power_df):
        """Konfidenzintervalle werden berechnet"""
        result = compute_load_forecast(sample_power_df, power_col="p_kw")

        assert result.available
        assert len(result.confidence_lower) == len(result.forecast_values)
        assert len(result.confidence_upper) == len(result.forecast_values)

        # Lower sollte < Forecast < Upper sein
        for low, val, up in zip(result.confidence_lower, result.forecast_values, result.confidence_upper):
            assert low <= val <= up

    def test_trend_detection(self, sample_power_df):
        """Trend wird erkannt"""
        result = compute_load_forecast(sample_power_df, power_col="p_kw")

        assert result.available
        assert result.trend in ["steigend", "fallend", "stabil"]


# ============================================================================
# BLINDLEISTUNGS-KOMPENSATION TESTS
# ============================================================================
class TestComputeCompensationROI:

    def test_basic_compensation(self):
        """Grundlegende Kompensations-Berechnung"""
        result = compute_compensation_roi(
            avg_active_power_kw=100.0,
            avg_reactive_power_kvar=50.0,
            target_cosphi=0.95
        )

        assert result.available
        assert result.required_compensation_kvar > 0

    def test_no_compensation_needed(self):
        """Keine Kompensation nötig wenn cos(phi) bereits gut"""
        result = compute_compensation_roi(
            avg_active_power_kw=100.0,
            avg_reactive_power_kvar=10.0,  # Sehr niedrig
            target_cosphi=0.90
        )

        assert result.available
        # cos(phi) bereits > 0.95, keine Kompensation nötig
        assert result.required_compensation_kvar == pytest.approx(0.0, abs=1.0)

    def test_zero_power(self):
        """Null-Leistung"""
        result = compute_compensation_roi(
            avg_active_power_kw=0.0,
            avg_reactive_power_kvar=50.0
        )

        assert not result.available
