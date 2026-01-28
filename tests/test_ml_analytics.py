# tests/test_ml_analytics.py
"""
Tests für ML-Analytics (Lastprognose, Anomalie-Erkennung, Peak-Wahrscheinlichkeiten)
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from report_builder.ml_analytics import (
    compute_load_forecast_ml,
    detect_anomalies,
    predict_peak_probability,
    create_time_features,
    ForecastResult,
    AnomalyResult,
    PeakPrediction,
)


# ============================================================================
# FIXTURES
# ============================================================================
@pytest.fixture
def sample_power_df():
    """DataFrame mit typischem Lastgang für 30 Tage"""
    dates = pd.date_range(start="2024-01-01", periods=30*24*4, freq="15min")
    np.random.seed(42)

    power = np.zeros(len(dates))
    for i, ts in enumerate(dates):
        hour = ts.hour
        weekday = ts.weekday()

        # Basis-Last
        base = 50.0

        # Tagesgang (höher tagsüber)
        if 6 <= hour < 22:
            day_factor = 1.5
        else:
            day_factor = 0.5

        # Wochenend-Reduktion
        if weekday >= 5:
            weekend_factor = 0.7
        else:
            weekend_factor = 1.0

        # Zufällige Variation
        noise = np.random.normal(0, 5)

        power[i] = base * day_factor * weekend_factor + noise

    df = pd.DataFrame({
        "timestamp": dates,
        "p_kw": power
    })
    return df


@pytest.fixture
def short_power_df():
    """Kurzer DataFrame (3 Tage)"""
    dates = pd.date_range(start="2024-01-01", periods=3*24*4, freq="15min")
    np.random.seed(42)
    power = 50 + np.random.normal(0, 10, len(dates))
    return pd.DataFrame({
        "timestamp": dates,
        "p_kw": power
    })


@pytest.fixture
def df_with_anomalies():
    """DataFrame mit eingebauten Anomalien"""
    dates = pd.date_range(start="2024-01-01", periods=7*24*4, freq="15min")
    np.random.seed(42)

    # Normaler Lastgang
    power = 50 + np.random.normal(0, 5, len(dates))

    # Anomalien einfügen
    # Spike am Tag 2
    power[24*4 + 10] = 200.0  # Extrem hoher Wert
    # Negativer Wert am Tag 3
    power[48*4 + 5] = -10.0
    # Plötzlicher Abfall am Tag 4
    power[72*4:72*4+10] = 5.0
    # Sehr hohe Varianz am Tag 5
    power[96*4:96*4+20] = power[96*4:96*4+20] + np.random.normal(0, 30, 20)

    return pd.DataFrame({
        "timestamp": dates,
        "p_kw": power
    })


# ============================================================================
# LASTPROGNOSE TESTS
# ============================================================================
class TestComputeLoadForecastML:

    def test_basic_forecast(self, sample_power_df):
        """Grundlegende Prognose funktioniert"""
        result = compute_load_forecast_ml(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            forecast_horizon_hours=24
        )

        assert result.available
        assert len(result.forecast_values) > 0
        assert len(result.forecast_timestamps) == len(result.forecast_values)
        assert result.predicted_peak_kw > 0

    def test_forecast_horizon(self, sample_power_df):
        """Prognosehorizont wird respektiert"""
        horizon_hours = 48
        result = compute_load_forecast_ml(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            forecast_horizon_hours=horizon_hours
        )

        assert result.available
        # Bei 15-min Intervallen: horizon_hours * 4 Punkte
        expected_points = horizon_hours * 4
        assert len(result.forecast_values) == expected_points

    def test_confidence_intervals(self, sample_power_df):
        """Konfidenzintervalle werden berechnet"""
        result = compute_load_forecast_ml(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        assert result.available
        assert len(result.confidence_lower) == len(result.forecast_values)
        assert len(result.confidence_upper) == len(result.forecast_values)

        # Untere Grenze <= Prognose <= Obere Grenze
        for i in range(len(result.forecast_values)):
            assert result.confidence_lower[i] <= result.forecast_values[i]
            assert result.forecast_values[i] <= result.confidence_upper[i]

    def test_trend_detection(self, sample_power_df):
        """Trend wird erkannt"""
        result = compute_load_forecast_ml(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        assert result.available
        assert result.trend in ["steigend", "fallend", "stabil"]

    def test_insufficient_data(self, short_power_df):
        """Zu wenig Daten wird erkannt"""
        result = compute_load_forecast_ml(
            short_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            forecast_horizon_hours=168  # 1 Woche - zu lang für 3 Tage Daten
        )

        # Sollte trotzdem funktionieren, aber mit Warnung
        # Die Prognose könnte weniger zuverlässig sein
        assert isinstance(result, ForecastResult)

    def test_empty_dataframe(self):
        """Leerer DataFrame gibt unavailable zurück"""
        result = compute_load_forecast_ml(
            pd.DataFrame(),
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        assert not result.available
        assert result.error_message != ""

    def test_missing_column(self, sample_power_df):
        """Fehlende Spalte wird erkannt"""
        result = compute_load_forecast_ml(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="nonexistent_column"
        )

        assert not result.available

    def test_model_metrics(self, sample_power_df):
        """Modell-Metriken werden berechnet"""
        result = compute_load_forecast_ml(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        assert result.available
        assert result.mae >= 0  # Mean Absolute Error
        assert result.mape >= 0  # Mean Absolute Percentage Error


# ============================================================================
# ANOMALIE-ERKENNUNG TESTS
# ============================================================================
class TestDetectAnomalies:

    def test_basic_detection(self, df_with_anomalies):
        """Grundlegende Anomalie-Erkennung"""
        result = detect_anomalies(
            df_with_anomalies,
            timestamp_col="timestamp",
            power_col="p_kw",
            sensitivity=2.5
        )

        assert result.available
        assert result.total_anomalies > 0
        assert len(result.anomaly_timestamps) == result.total_anomalies

    def test_sensitivity_effect(self, df_with_anomalies):
        """Sensitivität beeinflusst Anomalie-Anzahl"""
        result_low = detect_anomalies(
            df_with_anomalies,
            timestamp_col="timestamp",
            power_col="p_kw",
            sensitivity=1.5  # Niedriger = mehr Anomalien
        )

        result_high = detect_anomalies(
            df_with_anomalies,
            timestamp_col="timestamp",
            power_col="p_kw",
            sensitivity=3.5  # Höher = weniger Anomalien
        )

        # Niedrigere Sensitivität sollte mehr Anomalien finden
        assert result_low.total_anomalies >= result_high.total_anomalies

    def test_detection_methods(self, df_with_anomalies):
        """Verschiedene Erkennungsmethoden funktionieren"""
        methods = ["zscore", "iqr", "isolation_forest", "ensemble"]

        for method in methods:
            result = detect_anomalies(
                df_with_anomalies,
                timestamp_col="timestamp",
                power_col="p_kw",
                method=method
            )

            assert result.available, f"Methode {method} sollte funktionieren"

    def test_ensemble_method(self, df_with_anomalies):
        """Ensemble-Methode kombiniert mehrere Ansätze"""
        result = detect_anomalies(
            df_with_anomalies,
            timestamp_col="timestamp",
            power_col="p_kw",
            method="ensemble"
        )

        assert result.available
        # Ensemble sollte robuster sein und die offensichtlichen Anomalien finden
        assert result.total_anomalies > 0

    def test_anomaly_types(self, df_with_anomalies):
        """Anomalie-Typen werden klassifiziert"""
        result = detect_anomalies(
            df_with_anomalies,
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        assert result.available
        if result.total_anomalies > 0:
            assert len(result.anomaly_types) == result.total_anomalies
            # Alle Typen sollten nicht-leer sein
            for atype in result.anomaly_types:
                assert atype != ""

    def test_severity_distribution(self, df_with_anomalies):
        """Schweregrad-Verteilung wird berechnet"""
        result = detect_anomalies(
            df_with_anomalies,
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        assert result.available
        if result.total_anomalies > 0:
            assert len(result.severity_distribution) > 0
            # Summe der Verteilung sollte == total_anomalies sein
            total_in_dist = sum(result.severity_distribution.values())
            assert total_in_dist == result.total_anomalies

    def test_recommendations(self, df_with_anomalies):
        """Empfehlungen werden generiert"""
        result = detect_anomalies(
            df_with_anomalies,
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        assert result.available
        if result.total_anomalies > 0:
            assert len(result.recommendations) > 0

    def test_no_anomalies_in_clean_data(self, sample_power_df):
        """Saubere Daten haben wenige/keine Anomalien"""
        result = detect_anomalies(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            sensitivity=3.5  # Hohe Schwelle
        )

        assert result.available
        # Normale Daten sollten wenig Anomalien haben
        assert result.anomaly_percentage < 5.0

    def test_empty_dataframe(self):
        """Leerer DataFrame gibt unavailable zurück"""
        result = detect_anomalies(
            pd.DataFrame(),
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        assert not result.available


# ============================================================================
# PEAK-WAHRSCHEINLICHKEIT TESTS
# ============================================================================
class TestPredictPeakProbability:

    def test_basic_prediction(self, sample_power_df):
        """Grundlegende Peak-Wahrscheinlichkeit"""
        threshold = 70.0  # Oberhalb des Durchschnitts

        predictions = predict_peak_probability(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            threshold_kw=threshold,
            forecast_hours=24
        )

        assert len(predictions) > 0
        for pred in predictions:
            assert isinstance(pred, PeakPrediction)
            assert 0 <= pred.probability <= 1
            # predicted_power_kw kann 0 sein bei niedrigem Risiko
            assert pred.predicted_power_kw >= 0

    def test_probability_range(self, sample_power_df):
        """Wahrscheinlichkeiten sind im gültigen Bereich"""
        threshold = 60.0

        predictions = predict_peak_probability(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            threshold_kw=threshold,
            forecast_hours=48
        )

        for pred in predictions:
            assert 0.0 <= pred.probability <= 1.0

    def test_threshold_effect(self, sample_power_df):
        """Höhere Schwelle = niedrigere Wahrscheinlichkeit"""
        low_threshold = 50.0
        high_threshold = 100.0

        preds_low = predict_peak_probability(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            threshold_kw=low_threshold,
            forecast_hours=24
        )

        preds_high = predict_peak_probability(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            threshold_kw=high_threshold,
            forecast_hours=24
        )

        # Durchschnittliche Wahrscheinlichkeit bei niedriger Schwelle sollte höher sein
        avg_prob_low = np.mean([p.probability for p in preds_low]) if preds_low else 0
        avg_prob_high = np.mean([p.probability for p in preds_high]) if preds_high else 0

        assert avg_prob_low >= avg_prob_high

    def test_timestamp_assignment(self, sample_power_df):
        """Zeitstempel werden korrekt zugewiesen"""
        predictions = predict_peak_probability(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            threshold_kw=60.0,
            forecast_hours=24
        )

        for pred in predictions:
            # Zeitstempel sollte ein datetime sein
            assert isinstance(pred.timestamp, datetime)
            # Stunde sollte gueltig sein
            assert 0 <= pred.timestamp.hour <= 23


# ============================================================================
# FEATURE ENGINEERING TESTS
# ============================================================================
class TestCreateTimeFeatures:

    def test_basic_features(self, sample_power_df):
        """Grundlegende Zeitfeatures werden erstellt"""
        df = sample_power_df.copy()
        df_with_features = create_time_features(df, "timestamp")

        expected_features = ["hour", "dayofweek", "month", "is_weekend"]
        for feature in expected_features:
            assert feature in df_with_features.columns

    def test_cyclical_encoding(self, sample_power_df):
        """Zyklische Kodierung für Stunde und Tag"""
        df = sample_power_df.copy()
        df_with_features = create_time_features(df, "timestamp")

        # Zyklische Features sollten zwischen -1 und 1 liegen
        cyclical_features = ["hour_sin", "hour_cos", "dayofweek_sin", "dayofweek_cos"]
        for feature in cyclical_features:
            if feature in df_with_features.columns:
                assert df_with_features[feature].min() >= -1.0
                assert df_with_features[feature].max() <= 1.0

    def test_weekend_flag(self, sample_power_df):
        """Wochenend-Flag ist korrekt"""
        df = sample_power_df.copy()
        df_with_features = create_time_features(df, "timestamp")

        # Prüfe einige bekannte Daten
        # 2024-01-01 ist ein Montag (kein Wochenende)
        # 2024-01-06 ist ein Samstag (Wochenende)
        monday_rows = df_with_features[
            pd.to_datetime(df_with_features["timestamp"]).dt.date == pd.to_datetime("2024-01-01").date()
        ]
        if len(monday_rows) > 0:
            assert monday_rows["is_weekend"].iloc[0] == 0

        saturday_rows = df_with_features[
            pd.to_datetime(df_with_features["timestamp"]).dt.date == pd.to_datetime("2024-01-06").date()
        ]
        if len(saturday_rows) > 0:
            assert saturday_rows["is_weekend"].iloc[0] == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================
class TestMLIntegration:

    def test_forecast_then_anomaly(self, sample_power_df):
        """Prognose und Anomalie-Erkennung können nacheinander ausgeführt werden"""
        forecast = compute_load_forecast_ml(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        anomalies = detect_anomalies(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        assert forecast.available
        assert anomalies.available

    def test_full_analysis_pipeline(self, sample_power_df):
        """Komplette Analyse-Pipeline"""
        # 1. Prognose
        forecast = compute_load_forecast_ml(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw",
            forecast_horizon_hours=24
        )

        # 2. Anomalien
        anomalies = detect_anomalies(
            sample_power_df,
            timestamp_col="timestamp",
            power_col="p_kw"
        )

        # 3. Peak-Wahrscheinlichkeiten basierend auf Prognose
        if forecast.available:
            threshold = forecast.predicted_peak_kw * 0.9
            peaks = predict_peak_probability(
                sample_power_df,
                timestamp_col="timestamp",
                power_col="p_kw",
                threshold_kw=threshold,
                forecast_hours=24
            )

            assert len(peaks) > 0

        assert forecast.available
        assert anomalies.available
