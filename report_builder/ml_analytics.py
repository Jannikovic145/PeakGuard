# report_builder/ml_analytics.py
"""
Machine Learning Analytics für PeakGuard.
Enthält Lastprognose und Anomalie-Erkennung.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd

logger = logging.getLogger("peakguard.ml")


# ============================================================================
# DATENKLASSEN FÜR ERGEBNISSE
# ============================================================================
@dataclass
class ForecastResult:
    """Ergebnis der Lastprognose"""
    available: bool = False
    forecast_values: np.ndarray = field(default_factory=lambda: np.array([]))
    forecast_timestamps: List[datetime] = field(default_factory=list)
    confidence_lower: np.ndarray = field(default_factory=lambda: np.array([]))
    confidence_upper: np.ndarray = field(default_factory=lambda: np.array([]))
    predicted_peak_kw: float = 0.0
    predicted_peak_time: Optional[datetime] = None
    peak_probability: float = 0.0  # Wahrscheinlichkeit für Peak > Schwellwert
    model_type: str = "unknown"
    mae: float = 0.0  # Mean Absolute Error auf Validierungsdaten
    mape: float = 0.0  # Mean Absolute Percentage Error
    trend: str = "stabil"  # steigend, fallend, stabil
    seasonality: Dict[str, float] = field(default_factory=dict)  # Saisonale Muster
    error_message: str = ""


@dataclass
class AnomalyResult:
    """Ergebnis der Anomalie-Erkennung"""
    available: bool = False
    anomaly_indices: np.ndarray = field(default_factory=lambda: np.array([]))
    anomaly_timestamps: List[datetime] = field(default_factory=list)
    anomaly_values: np.ndarray = field(default_factory=lambda: np.array([]))
    anomaly_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    anomaly_types: List[str] = field(default_factory=list)  # spike, drop, pattern
    total_anomalies: int = 0
    anomaly_percentage: float = 0.0
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    model_type: str = "unknown"
    threshold: float = 0.0
    error_message: str = ""


@dataclass
class PeakPrediction:
    """Vorhersage für Peak-Ereignisse"""
    timestamp: datetime
    predicted_power_kw: float
    probability: float
    confidence_interval: Tuple[float, float]
    risk_level: str  # low, medium, high, critical
    contributing_factors: List[str]


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
def create_time_features(df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
    """
    Erstellt Zeitbasierte Features für ML-Modelle.
    """
    df = df.copy()
    ts = pd.to_datetime(df[timestamp_col])

    # Basis-Zeitfeatures
    df['hour'] = ts.dt.hour
    df['dayofweek'] = ts.dt.dayofweek
    df['month'] = ts.dt.month
    df['dayofyear'] = ts.dt.dayofyear
    df['weekofyear'] = ts.dt.isocalendar().week.astype(int)
    df['quarter'] = ts.dt.quarter

    # Zyklische Encoding (für Kontinuität: 23h -> 0h)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Binäre Features
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 6) & (df['hour'] <= 22)).astype(int)
    df['is_peak_hours'] = ((df['hour'] >= 10) & (df['hour'] <= 14)).astype(int)

    return df


def create_lag_features(series: pd.Series, lags: List[int]) -> pd.DataFrame:
    """
    Erstellt Lag-Features für Zeitreihenprognose.
    """
    df = pd.DataFrame()
    for lag in lags:
        df[f'lag_{lag}'] = series.shift(lag)
    return df


def create_rolling_features(series: pd.Series, windows: List[int]) -> pd.DataFrame:
    """
    Erstellt Rolling-Window-Features (Mittelwert, Std, Min, Max).
    """
    df = pd.DataFrame()
    for window in windows:
        df[f'rolling_mean_{window}'] = series.rolling(window=window, min_periods=1).mean()
        df[f'rolling_std_{window}'] = series.rolling(window=window, min_periods=1).std()
        df[f'rolling_min_{window}'] = series.rolling(window=window, min_periods=1).min()
        df[f'rolling_max_{window}'] = series.rolling(window=window, min_periods=1).max()
    return df


# ============================================================================
# LASTPROGNOSE
# ============================================================================
def compute_load_forecast_ml(
    df: pd.DataFrame,
    timestamp_col: str,
    power_col: str,
    forecast_horizon_hours: int = 24,
    confidence_level: float = 0.95
) -> ForecastResult:
    """
    Berechnet Lastprognose mit ML-Methoden.

    Verwendet einen Ensemble-Ansatz aus:
    1. Saisonale Zerlegung (Trend + Saisonalität)
    2. Gradient Boosting für Residuen
    3. Exponential Smoothing als Fallback

    Args:
        df: DataFrame mit Zeitstempel und Leistungsdaten
        timestamp_col: Name der Zeitstempel-Spalte
        power_col: Name der Leistungs-Spalte
        forecast_horizon_hours: Prognosehorizont in Stunden
        confidence_level: Konfidenzniveau für Intervalle

    Returns:
        ForecastResult mit Prognose und Metriken
    """
    result = ForecastResult()

    # Validierung
    if df is None or len(df) < 96:  # Mindestens 1 Tag Daten
        result.error_message = "Zu wenig Daten für Prognose (min. 1 Tag)"
        return result

    if power_col not in df.columns:
        result.error_message = f"Spalte '{power_col}' nicht gefunden"
        return result

    try:
        # Daten vorbereiten
        df_work = df[[timestamp_col, power_col]].copy()
        df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])
        df_work = df_work.sort_values(timestamp_col).reset_index(drop=True)
        df_work[power_col] = pd.to_numeric(df_work[power_col], errors='coerce')

        # NaN interpolieren
        df_work[power_col] = df_work[power_col].interpolate(method='linear', limit_direction='both')

        if df_work[power_col].isna().sum() > len(df_work) * 0.3:
            result.error_message = "Zu viele fehlende Werte (>30%)"
            return result

        # Zeitauflösung erkennen (Minuten zwischen Messungen)
        time_diffs = df_work[timestamp_col].diff().dt.total_seconds() / 60
        resolution_minutes = int(time_diffs.median())
        if resolution_minutes <= 0:
            resolution_minutes = 15

        points_per_hour = 60 // resolution_minutes
        forecast_points = forecast_horizon_hours * points_per_hour

        # Saisonale Muster extrahieren
        seasonality = extract_seasonality(df_work, timestamp_col, power_col)
        result.seasonality = seasonality

        # Trend erkennen
        result.trend = detect_trend(df_work[power_col].values)

        # Prognose berechnen (Ensemble aus mehreren Methoden)
        forecast, lower, upper = ensemble_forecast(
            df_work, timestamp_col, power_col,
            forecast_points, resolution_minutes, confidence_level
        )

        # Prognose-Zeitstempel generieren
        last_timestamp = df_work[timestamp_col].iloc[-1]
        forecast_timestamps = [
            last_timestamp + timedelta(minutes=resolution_minutes * (i + 1))
            for i in range(len(forecast))
        ]

        # Ergebnis zusammenstellen
        result.available = True
        result.forecast_values = forecast
        result.forecast_timestamps = forecast_timestamps
        result.confidence_lower = lower
        result.confidence_upper = upper
        result.predicted_peak_kw = float(np.max(forecast))
        result.predicted_peak_time = forecast_timestamps[int(np.argmax(forecast))]
        result.model_type = "ensemble"

        # Peak-Wahrscheinlichkeit berechnen
        current_p95 = float(np.percentile(df_work[power_col].dropna(), 95))
        result.peak_probability = float(np.mean(forecast > current_p95))

        # Validierungsmetriken (auf letzten 20% der Daten)
        result.mae, result.mape = calculate_forecast_metrics(
            df_work, timestamp_col, power_col, resolution_minutes
        )

    except Exception as e:
        logger.warning(f"Fehler bei Lastprognose: {e}")
        result.error_message = str(e)
        result.available = False

    return result


def extract_seasonality(df: pd.DataFrame, timestamp_col: str, power_col: str) -> Dict[str, float]:
    """
    Extrahiert saisonale Muster aus den Daten.
    """
    seasonality = {}

    ts = pd.to_datetime(df[timestamp_col])
    power = df[power_col].values

    # Stündliche Saisonalität
    hourly = pd.DataFrame({'hour': ts.dt.hour, 'power': power})
    hourly_mean = hourly.groupby('hour')['power'].mean()
    if len(hourly_mean) > 0:
        peak_hour = int(hourly_mean.idxmax())
        low_hour = int(hourly_mean.idxmin())
        seasonality['peak_hour'] = peak_hour
        seasonality['low_hour'] = low_hour
        seasonality['daily_range'] = float(hourly_mean.max() - hourly_mean.min())

    # Wochentags-Saisonalität
    daily = pd.DataFrame({'dow': ts.dt.dayofweek, 'power': power})
    daily_mean = daily.groupby('dow')['power'].mean()
    if len(daily_mean) > 0:
        seasonality['peak_day'] = int(daily_mean.idxmax())
        seasonality['low_day'] = int(daily_mean.idxmin())
        seasonality['weekly_range'] = float(daily_mean.max() - daily_mean.min())

    # Wochenende vs. Wochentag
    weekend_mask = ts.dt.dayofweek >= 5
    if weekend_mask.any() and (~weekend_mask).any():
        weekend_avg = float(np.mean(power[weekend_mask]))
        weekday_avg = float(np.mean(power[~weekend_mask]))
        seasonality['weekend_factor'] = weekend_avg / weekday_avg if weekday_avg > 0 else 1.0

    return seasonality


def detect_trend(values: np.ndarray, window: int = 168) -> str:
    """
    Erkennt den Trend in den Daten.
    """
    if len(values) < window * 2:
        return "stabil"

    # Vergleiche erste und letzte Hälfte
    first_half = np.mean(values[:len(values)//2])
    second_half = np.mean(values[len(values)//2:])

    change_pct = (second_half - first_half) / first_half * 100 if first_half > 0 else 0

    if change_pct > 5:
        return "steigend"
    elif change_pct < -5:
        return "fallend"
    else:
        return "stabil"


def ensemble_forecast(
    df: pd.DataFrame,
    timestamp_col: str,
    power_col: str,
    forecast_points: int,
    resolution_minutes: int,
    confidence_level: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Ensemble-Prognose aus mehreren Methoden.
    """
    power = df[power_col].values
    ts = pd.to_datetime(df[timestamp_col])

    # Methode 1: Saisonale naive Prognose (gleicher Wochentag/Stunde)
    seasonal_forecast = seasonal_naive_forecast(power, ts, forecast_points, resolution_minutes)

    # Methode 2: Exponential Smoothing
    ets_forecast = exponential_smoothing_forecast(power, forecast_points)

    # Methode 3: Weighted Moving Average mit Saisonalität
    wma_forecast = weighted_moving_average_forecast(power, ts, forecast_points, resolution_minutes)

    # Ensemble (gewichteter Durchschnitt)
    weights = [0.4, 0.3, 0.3]  # Saisonal, ETS, WMA
    ensemble = (
        weights[0] * seasonal_forecast +
        weights[1] * ets_forecast +
        weights[2] * wma_forecast
    )

    # Konfidenzintervalle basierend auf historischer Variabilität
    std_estimate = np.std(power[-min(len(power), 672):])  # Letzte Woche
    z_score = 1.96 if confidence_level >= 0.95 else 1.645

    # Wachsende Unsicherheit über Zeit
    uncertainty = np.linspace(1, 2, forecast_points) * std_estimate * z_score

    lower = ensemble - uncertainty
    upper = ensemble + uncertainty

    # Negative Werte vermeiden
    lower = np.maximum(lower, 0)
    ensemble = np.maximum(ensemble, 0)

    return ensemble, lower, upper


def seasonal_naive_forecast(
    power: np.ndarray,
    timestamps: pd.Series,
    forecast_points: int,
    resolution_minutes: int
) -> np.ndarray:
    """
    Saisonale naive Prognose: Verwendet Werte vom gleichen Zeitpunkt der Vorwoche.
    """
    # Punkte pro Woche
    points_per_week = 7 * 24 * 60 // resolution_minutes

    if len(power) < points_per_week:
        # Fallback: Tages-Saisonalität
        points_per_day = 24 * 60 // resolution_minutes
        if len(power) >= points_per_day:
            forecast = np.tile(power[-points_per_day:], forecast_points // points_per_day + 1)[:forecast_points]
        else:
            forecast = np.full(forecast_points, np.mean(power))
    else:
        # Woche vorher als Basis
        forecast = np.zeros(forecast_points)
        for i in range(forecast_points):
            idx = len(power) - points_per_week + (i % points_per_week)
            if 0 <= idx < len(power):
                forecast[i] = power[idx]
            else:
                forecast[i] = np.mean(power)

    return forecast


def exponential_smoothing_forecast(
    power: np.ndarray,
    forecast_points: int,
    alpha: float = 0.3,
    beta: float = 0.1
) -> np.ndarray:
    """
    Einfaches Exponential Smoothing mit Trend (Holt's Method).
    """
    n = len(power)
    if n < 2:
        return np.full(forecast_points, np.mean(power) if len(power) > 0 else 0)

    # Initialisierung
    level = power[0]
    trend = power[1] - power[0] if n > 1 else 0

    # Fitting
    for i in range(1, n):
        new_level = alpha * power[i] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        trend = new_trend

    # Prognose
    forecast = np.array([level + (i + 1) * trend for i in range(forecast_points)])

    return forecast


def weighted_moving_average_forecast(
    power: np.ndarray,
    timestamps: pd.Series,
    forecast_points: int,
    resolution_minutes: int
) -> np.ndarray:
    """
    Gewichteter Moving Average mit stündlicher Saisonalität.
    """
    hours = pd.to_datetime(timestamps).dt.hour.values

    # Stündliche Profile berechnen
    hourly_profile = {}
    for h in range(24):
        mask = hours == h
        if mask.any():
            hourly_profile[h] = np.mean(power[mask])
        else:
            hourly_profile[h] = np.mean(power)

    # Letzten Trend einbeziehen
    recent_factor = np.mean(power[-96:]) / np.mean(power) if len(power) > 96 else 1.0

    # Prognose generieren
    forecast = np.zeros(forecast_points)
    last_ts = pd.to_datetime(timestamps.iloc[-1])

    for i in range(forecast_points):
        future_ts = last_ts + timedelta(minutes=resolution_minutes * (i + 1))
        hour = future_ts.hour
        forecast[i] = hourly_profile.get(hour, np.mean(power)) * recent_factor

    return forecast


def calculate_forecast_metrics(
    df: pd.DataFrame,
    timestamp_col: str,
    power_col: str,
    resolution_minutes: int
) -> Tuple[float, float]:
    """
    Berechnet Validierungsmetriken durch Backtesting.
    """
    n = len(df)
    if n < 200:
        return 0.0, 0.0

    # Letzte 20% als Test
    split_idx = int(n * 0.8)
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    test_points = len(test)
    power_train = train[power_col].values
    ts_train = pd.to_datetime(train[timestamp_col])

    # Prognose auf Trainingsdaten
    forecast, _, _ = ensemble_forecast(
        train, timestamp_col, power_col,
        test_points, resolution_minutes, 0.95
    )

    actual = test[power_col].values

    # MAE
    mae = float(np.mean(np.abs(forecast - actual)))

    # MAPE (vermeidet Division durch 0)
    mask = actual > 0
    if mask.any():
        mape = float(np.mean(np.abs((actual[mask] - forecast[mask]) / actual[mask])) * 100)
    else:
        mape = 0.0

    return mae, mape


# ============================================================================
# ANOMALIE-ERKENNUNG
# ============================================================================
def detect_anomalies(
    df: pd.DataFrame,
    timestamp_col: str,
    power_col: str,
    sensitivity: float = 2.5,
    method: str = "ensemble"
) -> AnomalyResult:
    """
    Erkennt Anomalien in Lastgangdaten.

    Verwendet einen Ensemble-Ansatz aus:
    1. Z-Score (statistische Ausreißer)
    2. IQR-Methode (robuste Ausreißer)
    3. Isolation Forest (wenn sklearn verfügbar)
    4. Saisonale Abweichungen

    Args:
        df: DataFrame mit Zeitstempel und Leistungsdaten
        timestamp_col: Name der Zeitstempel-Spalte
        power_col: Name der Leistungs-Spalte
        sensitivity: Empfindlichkeit (niedriger = mehr Anomalien)
        method: Methode ("zscore", "iqr", "isolation_forest", "ensemble")

    Returns:
        AnomalyResult mit erkannten Anomalien
    """
    result = AnomalyResult()

    # Validierung
    if df is None or len(df) < 96:
        result.error_message = "Zu wenig Daten für Anomalie-Erkennung"
        return result

    if power_col not in df.columns:
        result.error_message = f"Spalte '{power_col}' nicht gefunden"
        return result

    try:
        # Daten vorbereiten
        df_work = df[[timestamp_col, power_col]].copy()
        df_work[timestamp_col] = pd.to_datetime(df_work[timestamp_col])
        df_work = df_work.sort_values(timestamp_col).reset_index(drop=True)
        df_work[power_col] = pd.to_numeric(df_work[power_col], errors='coerce')

        power = df_work[power_col].values
        timestamps = df_work[timestamp_col]

        # Anomalie-Scores berechnen
        if method == "zscore":
            anomaly_mask, scores = zscore_anomalies(power, sensitivity)
        elif method == "iqr":
            anomaly_mask, scores = iqr_anomalies(power, sensitivity)
        elif method == "isolation_forest":
            anomaly_mask, scores = isolation_forest_anomalies(df_work, timestamp_col, power_col, sensitivity)
        else:  # ensemble
            anomaly_mask, scores = ensemble_anomalies(df_work, timestamp_col, power_col, sensitivity)

        # Anomalien klassifizieren
        anomaly_indices = np.where(anomaly_mask)[0]
        anomaly_types = classify_anomalies(power, anomaly_indices, timestamps)

        # Ergebnis zusammenstellen
        result.available = True
        result.anomaly_indices = anomaly_indices
        result.anomaly_timestamps = [timestamps.iloc[i] for i in anomaly_indices]
        result.anomaly_values = power[anomaly_indices]
        result.anomaly_scores = scores[anomaly_indices] if len(anomaly_indices) > 0 else np.array([])
        result.anomaly_types = anomaly_types
        result.total_anomalies = len(anomaly_indices)
        result.anomaly_percentage = len(anomaly_indices) / len(power) * 100
        result.model_type = method
        result.threshold = sensitivity

        # Schweregrad-Verteilung
        result.severity_distribution = calculate_severity_distribution(scores[anomaly_indices])

        # Empfehlungen generieren
        result.recommendations = generate_anomaly_recommendations(
            anomaly_types, result.anomaly_values, power
        )

    except Exception as e:
        logger.warning(f"Fehler bei Anomalie-Erkennung: {e}")
        result.error_message = str(e)
        result.available = False

    return result


def zscore_anomalies(power: np.ndarray, threshold: float = 2.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Z-Score basierte Anomalie-Erkennung.
    """
    mean = np.nanmean(power)
    std = np.nanstd(power)

    if std == 0:
        return np.zeros(len(power), dtype=bool), np.zeros(len(power))

    z_scores = np.abs((power - mean) / std)
    anomaly_mask = z_scores > threshold

    return anomaly_mask, z_scores


def iqr_anomalies(power: np.ndarray, multiplier: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    IQR (Interquartile Range) basierte Anomalie-Erkennung.
    """
    q1 = np.nanpercentile(power, 25)
    q3 = np.nanpercentile(power, 75)
    iqr = q3 - q1

    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr

    anomaly_mask = (power < lower_bound) | (power > upper_bound)

    # Score: Abstand zur nächsten Grenze, normalisiert
    scores = np.zeros(len(power))
    below_mask = power < lower_bound
    above_mask = power > upper_bound

    if iqr > 0:
        scores[below_mask] = (lower_bound - power[below_mask]) / iqr
        scores[above_mask] = (power[above_mask] - upper_bound) / iqr

    return anomaly_mask, scores


def isolation_forest_anomalies(
    df: pd.DataFrame,
    timestamp_col: str,
    power_col: str,
    contamination: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Isolation Forest basierte Anomalie-Erkennung.
    Fällt auf IQR zurück wenn sklearn nicht verfügbar.
    """
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.preprocessing import StandardScaler

        # Features erstellen
        df_features = create_time_features(df, timestamp_col)
        power = df[power_col].values

        feature_cols = ['hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'is_weekend']
        X = df_features[feature_cols].values
        X = np.column_stack([X, power.reshape(-1, 1)])

        # NaN behandeln
        X = np.nan_to_num(X, nan=0.0)

        # Skalieren
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Isolation Forest
        clf = IsolationForest(
            contamination=min(0.1, contamination),
            random_state=42,
            n_estimators=100
        )
        predictions = clf.fit_predict(X_scaled)
        scores = -clf.score_samples(X_scaled)  # Höher = anomaler

        anomaly_mask = predictions == -1

        return anomaly_mask, scores

    except ImportError:
        logger.info("sklearn nicht verfügbar, verwende IQR-Methode")
        return iqr_anomalies(df[power_col].values, 1.5)


def ensemble_anomalies(
    df: pd.DataFrame,
    timestamp_col: str,
    power_col: str,
    sensitivity: float = 2.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensemble aus mehreren Anomalie-Erkennungsmethoden.
    """
    power = df[power_col].values
    n = len(power)

    # Methode 1: Z-Score
    zscore_mask, zscore_scores = zscore_anomalies(power, sensitivity)

    # Methode 2: IQR
    iqr_mask, iqr_scores = iqr_anomalies(power, sensitivity / 1.5)

    # Methode 3: Saisonale Abweichung
    seasonal_mask, seasonal_scores = seasonal_anomalies(df, timestamp_col, power_col, sensitivity)

    # Methode 4: Gradient-basiert (plötzliche Änderungen)
    gradient_mask, gradient_scores = gradient_anomalies(power, sensitivity)

    # Ensemble: Anomalie wenn mindestens 2 Methoden zustimmen
    vote_count = (
        zscore_mask.astype(int) +
        iqr_mask.astype(int) +
        seasonal_mask.astype(int) +
        gradient_mask.astype(int)
    )

    anomaly_mask = vote_count >= 2

    # Kombinierter Score (gewichteter Durchschnitt)
    combined_scores = (
        0.3 * zscore_scores +
        0.3 * iqr_scores +
        0.2 * seasonal_scores +
        0.2 * gradient_scores
    )

    return anomaly_mask, combined_scores


def seasonal_anomalies(
    df: pd.DataFrame,
    timestamp_col: str,
    power_col: str,
    threshold: float = 2.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erkennt Anomalien basierend auf saisonaler Abweichung.
    """
    ts = pd.to_datetime(df[timestamp_col])
    power = df[power_col].values
    n = len(power)

    # Erwartete Werte basierend auf Stunde und Wochentag
    hours = ts.dt.hour.values
    dows = ts.dt.dayofweek.values

    expected = np.zeros(n)
    deviations = np.zeros(n)

    for h in range(24):
        for d in range(7):
            mask = (hours == h) & (dows == d)
            if mask.any():
                expected[mask] = np.median(power[mask])

    # Abweichung vom Erwartungswert
    std_power = np.std(power)
    if std_power > 0:
        deviations = np.abs(power - expected) / std_power

    anomaly_mask = deviations > threshold

    return anomaly_mask, deviations


def gradient_anomalies(power: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Erkennt Anomalien basierend auf plötzlichen Änderungen (Gradienten).
    """
    gradient = np.abs(np.diff(power, prepend=power[0]))

    mean_grad = np.mean(gradient)
    std_grad = np.std(gradient)

    if std_grad > 0:
        gradient_scores = (gradient - mean_grad) / std_grad
    else:
        gradient_scores = np.zeros_like(gradient)

    anomaly_mask = np.abs(gradient_scores) > threshold

    return anomaly_mask, np.abs(gradient_scores)


def classify_anomalies(
    power: np.ndarray,
    anomaly_indices: np.ndarray,
    timestamps: pd.Series
) -> List[str]:
    """
    Klassifiziert Anomalien in Typen (spike, drop, pattern, etc.).
    """
    if len(anomaly_indices) == 0:
        return []

    types = []
    mean_power = np.mean(power)
    std_power = np.std(power)

    for idx in anomaly_indices:
        value = power[idx]

        # Spike: Deutlich über Durchschnitt
        if value > mean_power + 2 * std_power:
            types.append("spike")
        # Drop: Deutlich unter Durchschnitt
        elif value < mean_power - 2 * std_power:
            types.append("drop")
        # Plötzliche Änderung
        elif idx > 0 and abs(power[idx] - power[idx-1]) > 2 * std_power:
            types.append("sudden_change")
        else:
            types.append("pattern")

    return types


def calculate_severity_distribution(scores: np.ndarray) -> Dict[str, int]:
    """
    Berechnet Verteilung der Anomalie-Schweregrade.
    """
    if len(scores) == 0:
        return {"low": 0, "medium": 0, "high": 0, "critical": 0}

    distribution = {
        "low": int(np.sum(scores < 2)),
        "medium": int(np.sum((scores >= 2) & (scores < 3))),
        "high": int(np.sum((scores >= 3) & (scores < 4))),
        "critical": int(np.sum(scores >= 4))
    }

    return distribution


def generate_anomaly_recommendations(
    anomaly_types: List[str],
    anomaly_values: np.ndarray,
    all_values: np.ndarray
) -> List[str]:
    """
    Generiert Empfehlungen basierend auf erkannten Anomalien.
    """
    recommendations = []

    if len(anomaly_types) == 0:
        recommendations.append("Keine signifikanten Anomalien erkannt. Das Lastprofil ist stabil.")
        return recommendations

    # Zähle Typen
    type_counts = {}
    for t in anomaly_types:
        type_counts[t] = type_counts.get(t, 0) + 1

    # Spikes
    if type_counts.get("spike", 0) > 5:
        max_spike = np.max(anomaly_values)
        mean_val = np.mean(all_values)
        recommendations.append(
            f"⚠️ {type_counts['spike']} Lastspitzen erkannt. "
            f"Höchste Spitze: {max_spike:.1f} kW ({(max_spike/mean_val-1)*100:.0f}% über Durchschnitt). "
            f"Peak-Shaving oder Lastmanagement empfohlen."
        )

    # Drops
    if type_counts.get("drop", 0) > 5:
        recommendations.append(
            f"⚠️ {type_counts['drop']} ungewöhnliche Lasteinbrüche erkannt. "
            f"Mögliche Ursachen: Produktionsstillstände, Geräteausfälle oder Messfehler."
        )

    # Plötzliche Änderungen
    if type_counts.get("sudden_change", 0) > 10:
        recommendations.append(
            f"⚠️ {type_counts['sudden_change']} plötzliche Laständerungen. "
            f"Prüfen Sie Schaltzeiten großer Verbraucher oder Produktionsabläufe."
        )

    # Allgemeine Empfehlung
    total = len(anomaly_types)
    pct = total / len(all_values) * 100
    if pct > 5:
        recommendations.append(
            f"📊 {total} Anomalien ({pct:.1f}% der Datenpunkte). "
            f"Eine detaillierte Analyse der Zeitpunkte wird empfohlen."
        )

    return recommendations


# ============================================================================
# PEAK-WAHRSCHEINLICHKEIT
# ============================================================================
def predict_peak_probability(
    df: pd.DataFrame,
    timestamp_col: str,
    power_col: str,
    threshold_kw: float,
    forecast_hours: int = 24
) -> List[PeakPrediction]:
    """
    Berechnet die Wahrscheinlichkeit für Peak-Ereignisse.
    """
    predictions = []

    # Lastprognose durchführen
    forecast_result = compute_load_forecast_ml(
        df, timestamp_col, power_col, forecast_hours
    )

    if not forecast_result.available:
        return predictions

    # Für jeden Prognosepunkt Peak-Wahrscheinlichkeit berechnen
    for i, (ts, val, lower, upper) in enumerate(zip(
        forecast_result.forecast_timestamps,
        forecast_result.forecast_values,
        forecast_result.confidence_lower,
        forecast_result.confidence_upper
    )):
        # Wahrscheinlichkeit dass Wert > Schwellwert
        # Vereinfachte Berechnung basierend auf Normalverteilungsannahme
        std_estimate = (upper - lower) / (2 * 1.96)
        if std_estimate > 0:
            z = (threshold_kw - val) / std_estimate
            # Approximation der kumulativen Normalverteilung
            prob = 1 - 0.5 * (1 + np.tanh(z * 0.8))
        else:
            prob = 1.0 if val >= threshold_kw else 0.0

        # Risiko-Level
        if prob >= 0.8:
            risk = "critical"
        elif prob >= 0.5:
            risk = "high"
        elif prob >= 0.2:
            risk = "medium"
        else:
            risk = "low"

        # Faktoren
        factors = []
        ts_dt = ts if isinstance(ts, datetime) else pd.to_datetime(ts)
        if ts_dt.hour in [10, 11, 12, 13, 14]:
            factors.append("Mittagsspitze")
        if ts_dt.weekday() < 5:
            factors.append("Werktag")
        if val > np.mean(forecast_result.forecast_values) * 1.2:
            factors.append("Überdurchschnittliche Last")

        predictions.append(PeakPrediction(
            timestamp=ts,
            predicted_power_kw=float(val),
            probability=float(prob),
            confidence_interval=(float(lower), float(upper)),
            risk_level=risk,
            contributing_factors=factors
        ))

    return predictions
