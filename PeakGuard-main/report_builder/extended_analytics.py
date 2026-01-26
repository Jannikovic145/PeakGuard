# report_builder/extended_analytics.py
"""
Erweiterte Analysen für PeakGuard.
Enthält CO2-Analyse, ROI-Rechner und Lastprognosen.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("peakguard.extended_analytics")


# ============================================================================
# DATENMODELLE
# ============================================================================
@dataclass
class CO2Result:
    """Ergebnis der CO2-Analyse"""
    available: bool = False
    total_consumption_kwh: float = 0.0
    total_co2_kg: float = 0.0
    co2_factor_g_kwh: float = 400.0  # Deutscher Strommix 2024
    monthly_co2: Dict[str, float] = field(default_factory=dict)
    hourly_profile: Dict[int, float] = field(default_factory=dict)
    peak_co2_kg: float = 0.0
    base_co2_kg: float = 0.0
    potential_savings_kg: float = 0.0
    recommendation: str = ""


@dataclass
class ROIResult:
    """Ergebnis der ROI-Berechnung für Peak-Shaving-Maßnahmen"""
    available: bool = False
    investment_cost: float = 0.0
    annual_savings: float = 0.0
    payback_years: float = 0.0
    roi_percent: float = 0.0
    npv_10_years: float = 0.0  # Net Present Value
    irr_percent: float = 0.0   # Internal Rate of Return
    battery_size_kwh: float = 0.0
    battery_power_kw: float = 0.0
    recommendation: str = ""
    yearly_cashflows: List[float] = field(default_factory=list)


@dataclass
class BatterySpec:
    """Spezifikation für Batteriespeicher"""
    capacity_kwh: float
    power_kw: float
    efficiency: float = 0.90
    cycles_per_year: int = 365
    lifespan_years: int = 15
    cost_per_kwh: float = 500.0  # €/kWh
    installation_factor: float = 1.2  # 20% Zusatzkosten


@dataclass
class LoadForecastResult:
    """Ergebnis der Lastprognose"""
    available: bool = False
    forecast_horizon_days: int = 7
    predicted_peak_kw: float = 0.0
    predicted_avg_kw: float = 0.0
    confidence_lower: List[float] = field(default_factory=list)
    confidence_upper: List[float] = field(default_factory=list)
    forecast_values: List[float] = field(default_factory=list)
    forecast_timestamps: List[datetime] = field(default_factory=list)
    trend: str = ""  # "steigend", "fallend", "stabil"


# ============================================================================
# CO2-EMISSIONSFAKTOREN
# ============================================================================
# Durchschnittliche CO2-Emissionsfaktoren nach Tageszeit (g/kWh)
# Basierend auf typischem deutschen Strommix-Profil
CO2_HOURLY_FACTORS = {
    0: 380, 1: 370, 2: 365, 3: 360, 4: 365, 5: 380,
    6: 420, 7: 450, 8: 470, 9: 460, 10: 440, 11: 430,
    12: 420, 13: 410, 14: 420, 15: 440, 16: 460, 17: 480,
    18: 490, 19: 470, 20: 450, 21: 430, 22: 410, 23: 390
}

# Monatliche Variation (Sommer mehr Solar = weniger CO2)
CO2_MONTHLY_VARIATION = {
    1: 1.10, 2: 1.08, 3: 1.02, 4: 0.95, 5: 0.88, 6: 0.85,
    7: 0.84, 8: 0.86, 9: 0.92, 10: 1.00, 11: 1.05, 12: 1.08
}


# ============================================================================
# CO2-ANALYSE
# ============================================================================
def compute_co2_analysis(
    df: pd.DataFrame,
    power_col: str = "p_kw",
    resolution_minutes: int = 15,
    co2_factor_manual: Optional[float] = None
) -> CO2Result:
    """
    Berechnet CO2-Emissionen basierend auf Verbrauchsdaten.

    Args:
        df: DataFrame mit Zeitindex und Leistungsdaten
        power_col: Name der Leistungsspalte (kW)
        resolution_minutes: Zeitauflösung in Minuten
        co2_factor_manual: Manueller CO2-Faktor (g/kWh), sonst stündlich variabel

    Returns:
        CO2Result mit detaillierter Analyse
    """
    if df.empty or power_col not in df.columns:
        return CO2Result(available=False)

    try:
        # Leistungsdaten extrahieren
        power = pd.to_numeric(df[power_col], errors="coerce").fillna(0.0)

        # kWh pro Intervall
        hours_per_interval = resolution_minutes / 60.0
        energy_kwh = power * hours_per_interval

        # Gesamtverbrauch
        total_kwh = float(energy_kwh.sum())

        if total_kwh <= 0:
            return CO2Result(available=False)

        # CO2 berechnen
        if co2_factor_manual:
            # Konstanter Faktor
            total_co2_kg = total_kwh * co2_factor_manual / 1000.0
            hourly_profile = {h: co2_factor_manual for h in range(24)}
        else:
            # Zeitabhängiger Faktor
            hours = df.index.hour
            months = df.index.month

            # CO2-Faktoren pro Zeile berechnen
            co2_factors = pd.Series(
                [CO2_HOURLY_FACTORS.get(h, 400) * CO2_MONTHLY_VARIATION.get(m, 1.0)
                 for h, m in zip(hours, months)],
                index=df.index
            )

            # CO2 pro Intervall (kg)
            co2_per_interval = energy_kwh * co2_factors / 1000.0
            total_co2_kg = float(co2_per_interval.sum())

            # Stündliches Profil
            hourly_profile = {}
            for h in range(24):
                mask = hours == h
                if mask.sum() > 0:
                    hourly_profile[h] = float(co2_per_interval[mask].sum())

        # Monatliche Aggregation
        monthly_co2 = {}
        for month in range(1, 13):
            mask = df.index.month == month
            if mask.sum() > 0:
                month_energy = float(energy_kwh[mask].sum())
                month_factor = CO2_MONTHLY_VARIATION.get(month, 1.0) * 400 / 1000.0
                monthly_co2[f"{month:02d}"] = month_energy * month_factor

        # Peak vs. Base Analyse
        # Peak: oberstes Quartil der Last
        q75 = power.quantile(0.75)
        peak_mask = power >= q75
        base_mask = power < power.quantile(0.25)

        peak_energy = float(energy_kwh[peak_mask].sum())
        base_energy = float(energy_kwh[base_mask].sum())

        # Peak-Zeiten haben typisch höhere CO2-Faktoren (Spitzenlastkraftwerke)
        peak_co2_kg = peak_energy * 480 / 1000.0
        base_co2_kg = base_energy * 370 / 1000.0

        # Einsparpotenzial durch Lastverschiebung
        # Annahme: 20% der Peak-Last könnte in Nebenzeiten verschoben werden
        shiftable_energy = peak_energy * 0.20
        co2_diff = (480 - 370) / 1000.0  # g/kWh -> kg/kWh
        potential_savings = shiftable_energy * co2_diff

        # Empfehlung generieren
        avg_factor = total_co2_kg / total_kwh * 1000 if total_kwh > 0 else 400
        if avg_factor > 450:
            rec = (
                f"Hoher CO₂-Fußabdruck ({avg_factor:.0f} g/kWh).\n"
                "Empfehlung: Lastverschiebung in emissionsarme Zeiten (nachts, mittags bei Sonne) prüfen."
            )
        elif avg_factor > 400:
            rec = (
                f"Durchschnittlicher CO₂-Fußabdruck ({avg_factor:.0f} g/kWh).\n"
                "Empfehlung: Flexible Lasten in Mittagsstunden (PV-Einspeisung) verschieben."
            )
        else:
            rec = (
                f"Guter CO₂-Fußabdruck ({avg_factor:.0f} g/kWh).\n"
                "Das Lastprofil nutzt bereits emissionsarme Zeiten. Weiter optimieren durch PV-Eigenverbrauch."
            )

        return CO2Result(
            available=True,
            total_consumption_kwh=total_kwh,
            total_co2_kg=total_co2_kg,
            co2_factor_g_kwh=avg_factor,
            monthly_co2=monthly_co2,
            hourly_profile=hourly_profile,
            peak_co2_kg=peak_co2_kg,
            base_co2_kg=base_co2_kg,
            potential_savings_kg=potential_savings,
            recommendation=rec
        )

    except Exception as e:
        logger.warning(f"CO2-Analyse fehlgeschlagen: {e}")
        return CO2Result(available=False)


# ============================================================================
# ROI-RECHNER FÜR BATTERIESPEICHER
# ============================================================================
def compute_battery_roi(
    annual_peak_cost: float,
    current_peak_kw: float,
    target_peak_kw: float,
    peak_price_eur_kw_a: float = 100.0,
    battery_spec: Optional[BatterySpec] = None,
    discount_rate: float = 0.05,
    electricity_price_increase: float = 0.03
) -> ROIResult:
    """
    Berechnet ROI für Batteriespeicher zur Peak-Reduktion.

    Args:
        annual_peak_cost: Aktuelle jährliche Leistungskosten
        current_peak_kw: Aktuelle Spitzenlast (kW)
        target_peak_kw: Ziel-Spitzenlast nach Peak-Shaving (kW)
        peak_price_eur_kw_a: Leistungspreis (€/kW/Jahr)
        battery_spec: Batterie-Spezifikation (oder automatisch berechnet)
        discount_rate: Diskontierungssatz für NPV
        electricity_price_increase: Jährliche Strompreissteigerung

    Returns:
        ROIResult mit Wirtschaftlichkeitsanalyse
    """
    peak_reduction = current_peak_kw - target_peak_kw

    if peak_reduction <= 0:
        return ROIResult(
            available=False,
            recommendation="Keine Peak-Reduktion möglich."
        )

    # Batterie-Dimensionierung (falls nicht vorgegeben)
    if battery_spec is None:
        # Faustformel: 1.5h Entladezeit bei Peak-Reduktion
        battery_power = peak_reduction
        battery_capacity = battery_power * 1.5  # 1.5h
        battery_spec = BatterySpec(
            capacity_kwh=battery_capacity,
            power_kw=battery_power
        )

    # Investitionskosten
    battery_cost = battery_spec.capacity_kwh * battery_spec.cost_per_kwh
    total_investment = battery_cost * battery_spec.installation_factor

    # Jährliche Einsparung (erstes Jahr)
    annual_savings_year1 = peak_reduction * peak_price_eur_kw_a

    # Cashflows über Lebensdauer
    cashflows = [-total_investment]  # Jahr 0
    yearly_cashflows = []

    for year in range(1, battery_spec.lifespan_years + 1):
        # Strompreissteigerung erhöht Einsparungen
        year_savings = annual_savings_year1 * ((1 + electricity_price_increase) ** (year - 1))

        # Effizienz-Degradation (ca. 2% pro Jahr)
        degradation = 1.0 - (0.02 * (year - 1))
        effective_savings = year_savings * max(degradation, 0.7)

        cashflows.append(effective_savings)
        yearly_cashflows.append(effective_savings)

    # NPV berechnen
    npv = sum(cf / ((1 + discount_rate) ** i) for i, cf in enumerate(cashflows))

    # IRR approximieren (Newton-Raphson vereinfacht)
    irr = _calculate_irr(cashflows)

    # Payback Period
    cumulative = 0.0
    payback_years = float('inf')
    for i, cf in enumerate(cashflows[1:], 1):
        cumulative += cf
        if cumulative >= total_investment:
            # Lineare Interpolation für genauere Payback-Zeit
            prev_cumulative = cumulative - cf
            remaining = total_investment - prev_cumulative
            payback_years = i - 1 + (remaining / cf)
            break

    # ROI (Return on Investment)
    total_savings = sum(yearly_cashflows)
    roi_percent = ((total_savings - total_investment) / total_investment) * 100

    # Empfehlung generieren
    if payback_years <= 5:
        rec = (
            f"Sehr wirtschaftlich: Amortisation in {payback_years:.1f} Jahren.\n"
            f"NPV über {battery_spec.lifespan_years} Jahre: {npv:,.0f} €\n"
            "Empfehlung: Investition stark empfohlen."
        )
    elif payback_years <= 8:
        rec = (
            f"Wirtschaftlich: Amortisation in {payback_years:.1f} Jahren.\n"
            f"NPV über {battery_spec.lifespan_years} Jahre: {npv:,.0f} €\n"
            "Empfehlung: Investition sinnvoll, besonders bei steigenden Strompreisen."
        )
    elif payback_years <= 12:
        rec = (
            f"Grenzwertig wirtschaftlich: Amortisation in {payback_years:.1f} Jahren.\n"
            f"NPV über {battery_spec.lifespan_years} Jahre: {npv:,.0f} €\n"
            "Empfehlung: Nur bei zusätzlichen Förderungen oder strategischen Gründen."
        )
    else:
        rec = (
            f"Nicht wirtschaftlich: Amortisation > {battery_spec.lifespan_years} Jahre.\n"
            "Empfehlung: Andere Maßnahmen (Lastverschiebung, Prozessoptimierung) prüfen."
        )

    return ROIResult(
        available=True,
        investment_cost=total_investment,
        annual_savings=annual_savings_year1,
        payback_years=payback_years if payback_years != float('inf') else 99.0,
        roi_percent=roi_percent,
        npv_10_years=npv,
        irr_percent=irr * 100,
        battery_size_kwh=battery_spec.capacity_kwh,
        battery_power_kw=battery_spec.power_kw,
        recommendation=rec,
        yearly_cashflows=yearly_cashflows
    )


def _calculate_irr(cashflows: List[float], max_iter: int = 100, tol: float = 1e-6) -> float:
    """Berechnet IRR mit Newton-Raphson Methode."""
    rate = 0.10  # Startwert 10%

    for _ in range(max_iter):
        npv = sum(cf / ((1 + rate) ** i) for i, cf in enumerate(cashflows))
        dnpv = sum(-i * cf / ((1 + rate) ** (i + 1)) for i, cf in enumerate(cashflows))

        if abs(dnpv) < 1e-10:
            break

        new_rate = rate - npv / dnpv

        if abs(new_rate - rate) < tol:
            return new_rate

        rate = new_rate

        # Begrenzen
        rate = max(-0.99, min(10.0, rate))

    return rate


# ============================================================================
# EINFACHE LASTPROGNOSE
# ============================================================================
def compute_load_forecast(
    df: pd.DataFrame,
    power_col: str = "p_kw",
    forecast_days: int = 7
) -> LoadForecastResult:
    """
    Einfache Lastprognose basierend auf historischen Mustern.

    Verwendet einen gewichteten Durchschnitt der letzten Wochen
    für jede Stunde des Tages.

    Args:
        df: DataFrame mit Zeitindex und Leistungsdaten
        power_col: Name der Leistungsspalte
        forecast_days: Anzahl Tage für Prognose

    Returns:
        LoadForecastResult mit Prognose
    """
    if df.empty or power_col not in df.columns:
        return LoadForecastResult(available=False)

    try:
        power = pd.to_numeric(df[power_col], errors="coerce").dropna()

        if len(power) < 24 * 7:  # Mindestens eine Woche Daten
            return LoadForecastResult(
                available=False
            )

        # Stündliche Profile pro Wochentag
        hourly_profiles: Dict[Tuple[int, int], List[float]] = {}

        for ts, val in power.items():
            dow = ts.dayofweek  # 0=Montag
            hour = ts.hour
            key = (dow, hour)
            if key not in hourly_profiles:
                hourly_profiles[key] = []
            hourly_profiles[key].append(val)

        # Durchschnitt und Standardabweichung pro Slot
        avg_profiles: Dict[Tuple[int, int], Tuple[float, float]] = {}
        for key, values in hourly_profiles.items():
            avg_profiles[key] = (np.mean(values), np.std(values))

        # Prognose generieren
        last_ts = df.index.max()
        forecast_values = []
        forecast_timestamps = []
        confidence_lower = []
        confidence_upper = []

        for day_offset in range(1, forecast_days + 1):
            for hour in range(24):
                future_ts = last_ts + pd.Timedelta(days=day_offset, hours=hour - last_ts.hour)
                dow = future_ts.dayofweek
                key = (dow, hour)

                if key in avg_profiles:
                    avg, std = avg_profiles[key]
                else:
                    # Fallback auf Gesamtdurchschnitt
                    avg = float(power.mean())
                    std = float(power.std())

                forecast_values.append(avg)
                forecast_timestamps.append(future_ts)
                confidence_lower.append(avg - 1.96 * std)
                confidence_upper.append(avg + 1.96 * std)

        # Trend analysieren
        recent_avg = power[-24*7:].mean() if len(power) >= 24*7 else power.mean()
        older_avg = power[-24*14:-24*7].mean() if len(power) >= 24*14 else recent_avg

        if recent_avg > older_avg * 1.05:
            trend = "steigend"
        elif recent_avg < older_avg * 0.95:
            trend = "fallend"
        else:
            trend = "stabil"

        return LoadForecastResult(
            available=True,
            forecast_horizon_days=forecast_days,
            predicted_peak_kw=max(forecast_values),
            predicted_avg_kw=np.mean(forecast_values),
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            forecast_values=forecast_values,
            forecast_timestamps=forecast_timestamps,
            trend=trend
        )

    except Exception as e:
        logger.warning(f"Lastprognose fehlgeschlagen: {e}")
        return LoadForecastResult(available=False)


# ============================================================================
# BLINDLEISTUNGS-KOMPENSATION ROI
# ============================================================================
@dataclass
class CompensationROIResult:
    """Ergebnis der Blindleistungs-Kompensations-Wirtschaftlichkeit"""
    available: bool = False
    current_reactive_kvar: float = 0.0
    target_cosphi: float = 0.95
    required_compensation_kvar: float = 0.0
    investment_cost: float = 0.0
    annual_savings: float = 0.0
    payback_years: float = 0.0
    recommendation: str = ""


def compute_compensation_roi(
    avg_active_power_kw: float,
    avg_reactive_power_kvar: float,
    target_cosphi: float = 0.95,
    reactive_penalty_eur_kvar: float = 0.02,  # €/kvar/Monat
    compensation_cost_eur_kvar: float = 40.0   # €/kvar Anlage
) -> CompensationROIResult:
    """
    Berechnet Wirtschaftlichkeit einer Blindleistungskompensation.

    Args:
        avg_active_power_kw: Durchschnittliche Wirkleistung
        avg_reactive_power_kvar: Durchschnittliche Blindleistung
        target_cosphi: Ziel-cos(phi)
        reactive_penalty_eur_kvar: Blindarbeitspreis
        compensation_cost_eur_kvar: Kosten pro kvar Kompensationsanlage

    Returns:
        CompensationROIResult mit Analyse
    """
    if avg_active_power_kw <= 0:
        return CompensationROIResult(available=False)

    # Aktueller cos(phi)
    s_current = np.sqrt(avg_active_power_kw**2 + avg_reactive_power_kvar**2)
    current_cosphi = avg_active_power_kw / s_current if s_current > 0 else 1.0

    if current_cosphi >= target_cosphi:
        return CompensationROIResult(
            available=True,
            current_reactive_kvar=avg_reactive_power_kvar,
            target_cosphi=target_cosphi,
            required_compensation_kvar=0.0,
            investment_cost=0.0,
            annual_savings=0.0,
            payback_years=0.0,
            recommendation="cos(φ) bereits im Zielbereich. Keine Kompensation erforderlich."
        )

    # Ziel-Blindleistung berechnen
    target_tan_phi = np.tan(np.arccos(target_cosphi))
    target_reactive = avg_active_power_kw * target_tan_phi

    # Erforderliche Kompensation
    required_compensation = avg_reactive_power_kvar - target_reactive

    # Kosten
    investment = required_compensation * compensation_cost_eur_kvar

    # Jährliche Einsparung (Blindarbeitskosten)
    # Annahme: 10 Stunden Überschreitung pro Tag
    annual_penalty_before = avg_reactive_power_kvar * reactive_penalty_eur_kvar * 12
    annual_penalty_after = target_reactive * reactive_penalty_eur_kvar * 12
    annual_savings = annual_penalty_before - annual_penalty_after

    # Payback
    payback = investment / annual_savings if annual_savings > 0 else float('inf')

    # Empfehlung
    if payback <= 3:
        rec = f"Sehr wirtschaftlich: Amortisation in {payback:.1f} Jahren.\n" \
              f"Kompensationsanlage mit {required_compensation:.0f} kvar empfohlen."
    elif payback <= 5:
        rec = f"Wirtschaftlich: Amortisation in {payback:.1f} Jahren.\n" \
              f"Kompensationsanlage mit {required_compensation:.0f} kvar sinnvoll."
    else:
        rec = f"Lange Amortisation ({payback:.1f} Jahre).\n" \
              "Prüfen Sie zunächst, ob induktive Verbraucher reduziert werden können."

    return CompensationROIResult(
        available=True,
        current_reactive_kvar=avg_reactive_power_kvar,
        target_cosphi=target_cosphi,
        required_compensation_kvar=required_compensation,
        investment_cost=investment,
        annual_savings=annual_savings,
        payback_years=payback if payback != float('inf') else 99.0,
        recommendation=rec
    )
