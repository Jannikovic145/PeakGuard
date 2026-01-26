# report_builder/analytics.py
"""
Analyse-Funktionen für PeakGuard.
Enthält Peak-Erkennung, Aggregation und Berechnungen.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, cast

import numpy as np
import pandas as pd

from .config import GOAL_TO_QUANTILE, Tariffs
from .models import (
    BlkResult,
    PeakContextInfo,
    PeakEventsResult,
    Recommendation,
    Scenario,
    UnbalanceResult,
)
from .utils import RobustNumericParser

logger = logging.getLogger("peakguard.analytics")


# ============================================================================
# DATENVERARBEITUNG
# ============================================================================
def prepare_raw_power_data(
    d0: pd.DataFrame,
    timestamp_col: str,
    power_col: Optional[str],
    power_cols: Optional[List[str]],
    power_unit: str,
    pf_cols: Optional[List[str]],
) -> Dict[str, pd.DataFrame]:
    """
    Bereitet Rohdaten für die Analyse auf.

    Args:
        d0: Input DataFrame
        timestamp_col: Name der Zeitstempel-Spalte
        power_col: Name der Gesamt-Leistungsspalte (oder None)
        power_cols: Namen der Phasen-Spalten (oder None)
        power_unit: Einheit ("W", "kW", "Auto")
        pf_cols: Namen der cos-phi-Spalten (oder None)

    Returns:
        Dict mit "df" -> aufbereiteter DataFrame
    """
    # Canonical column detection
    canonical_total = "power_total" if "power_total" in d0.columns else None
    canonical_p = (
        ["power_1", "power_2", "power_3"]
        if all(c in d0.columns for c in ["power_1", "power_2", "power_3"])
        else None
    )
    canonical_c_total = "cosphi_total" if "cosphi_total" in d0.columns else None
    canonical_c = (
        ["cosphi_1", "cosphi_2", "cosphi_3"]
        if all(c in d0.columns for c in ["cosphi_1", "cosphi_2", "cosphi_3"])
        else None
    )

    # Build column list
    use_cols: List[str] = [timestamp_col]
    if canonical_p is not None:
        use_cols += canonical_p
    elif power_cols:
        use_cols += list(power_cols)
    if canonical_total is not None:
        use_cols.append(canonical_total)
    elif power_col:
        use_cols.append(power_col)
    if canonical_c is not None:
        use_cols += canonical_c
    elif canonical_c_total is not None:
        use_cols.append(canonical_c_total)
    elif pf_cols:
        use_cols += list(pf_cols)

    d = d0[use_cols].copy()
    d = d.dropna(subset=[timestamp_col])
    d = d.set_index(timestamp_col)
    d.index = pd.to_datetime(d.index, errors="coerce")
    d = d[~d.index.isna()].sort_index()

    if d.empty:
        return {"df": pd.DataFrame()}

    unit = (power_unit or "Auto").lower().strip()

    def _to_kw(s: pd.Series) -> pd.Series:
        x = pd.to_numeric(s, errors="coerce")
        if unit == "w":
            return x / 1000.0
        if unit == "kw":
            return x
        # Auto: Schätze anhand Median
        med = float(x.dropna().median()) if x.notna().any() else 0.0
        return (x / 1000.0) if med > 200.0 else x

    out = pd.DataFrame(index=cast(pd.DatetimeIndex, d.index))

    # Phase-based or total power
    if canonical_p is not None:
        p1 = _to_kw(d[canonical_p[0]])
        p2 = _to_kw(d[canonical_p[1]])
        p3 = _to_kw(d[canonical_p[2]])
        out["p1_kw"] = p1
        out["p2_kw"] = p2
        out["p3_kw"] = p3
        out["p_kw"] = p1.fillna(0.0) + p2.fillna(0.0) + p3.fillna(0.0)

        if canonical_c is not None:
            out["c1"] = pd.to_numeric(d[canonical_c[0]], errors="coerce").clip(0.0, 1.0)
            out["c2"] = pd.to_numeric(d[canonical_c[1]], errors="coerce").clip(0.0, 1.0)
            out["c3"] = pd.to_numeric(d[canonical_c[2]], errors="coerce").clip(0.0, 1.0)

    elif power_cols is not None and len(power_cols) == 3:
        p1 = _to_kw(d[power_cols[0]])
        p2 = _to_kw(d[power_cols[1]])
        p3 = _to_kw(d[power_cols[2]])
        out["p1_kw"] = p1
        out["p2_kw"] = p2
        out["p3_kw"] = p3
        out["p_kw"] = p1.fillna(0.0) + p2.fillna(0.0) + p3.fillna(0.0)

        if pf_cols is not None and len(pf_cols) == 3:
            out["c1"] = pd.to_numeric(d[pf_cols[0]], errors="coerce").clip(0.0, 1.0)
            out["c2"] = pd.to_numeric(d[pf_cols[1]], errors="coerce").clip(0.0, 1.0)
            out["c3"] = pd.to_numeric(d[pf_cols[2]], errors="coerce").clip(0.0, 1.0)

    else:
        if canonical_total is not None:
            out["p_kw"] = _to_kw(d[canonical_total])
        else:
            if power_col is None:
                raise ValueError("power_col ist None")
            out["p_kw"] = _to_kw(d[power_col])

        if canonical_c_total is not None:
            out["c_total"] = pd.to_numeric(d[canonical_c_total], errors="coerce").clip(0.0, 1.0)
        elif pf_cols is not None and len(pf_cols) == 1:
            out["c_total"] = pd.to_numeric(d[pf_cols[0]], errors="coerce").clip(0.0, 1.0)

    out = out.dropna(subset=["p_kw"])
    return {"df": out}


def aggregate_to_interval(df_raw: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """
    Aggregiert Rohdaten auf ein bestimmtes Intervall (z.B. 15 Minuten).

    Args:
        df_raw: Rohdaten-DataFrame
        minutes: Ziel-Intervall in Minuten

    Returns:
        Aggregierter DataFrame
    """
    rule = f"{int(minutes)}min"
    res = df_raw.resample(rule, label="left", closed="left").mean()

    agg = pd.DataFrame(index=cast(pd.DatetimeIndex, res.index))
    agg["p_kw"] = cast(pd.Series, res["p_kw"])

    for c in ["p1_kw", "p2_kw", "p3_kw", "c1", "c2", "c3", "c_total"]:
        if c in res.columns:
            agg[c] = cast(pd.Series, res[c])

    agg = agg.dropna(subset=["p_kw"])
    return agg


# ============================================================================
# PEAK-ANALYSE
# ============================================================================
def analyze_top_peaks(
    df_15: pd.DataFrame,
    cap_kw: float,
    n: int = 3
) -> List[PeakContextInfo]:
    """
    Analysiert die Top-N-Peaks und erstellt einfache Diagnosen.

    Args:
        df_15: 15-min aggregierter DataFrame
        cap_kw: Cap-Wert in kW
        n: Anzahl zu analysierender Peaks

    Returns:
        Liste von PeakContextInfo-Objekten
    """
    p = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce")
    top_indices = p.nlargest(n).index

    results: List[PeakContextInfo] = []

    for peak_idx in top_indices:
        peak_ts = pd.Timestamp(peak_idx)
        peak_kw = float(p.loc[peak_idx])

        # Diagnose: Schaue auf Umfeld (±1h)
        window = pd.Timedelta(hours=1)
        start = peak_ts - window
        end = peak_ts + window

        mask = (df_15.index >= start) & (df_15.index <= end)
        window_data = p[mask]

        duration_blocks = 0

        if len(window_data) < 3:
            diagnosis = "Isolierter Peak (Datenlücke)"
        else:
            mean_window = float(window_data.mean())
            std_window = float(window_data.std())

            over_cap = (window_data > cap_kw).sum()
            duration_blocks = int(over_cap)

            if duration_blocks <= 2 and (peak_kw - mean_window) > cap_kw * 0.2:
                diagnosis = "Gleichzeitigkeit (kurzer Peak, deutlich über Umfeld)"
            elif duration_blocks >= 8:
                diagnosis = "Dauerlast (lange Überschreitung, ≥2h)"
            elif std_window > cap_kw * 0.15:
                diagnosis = "Anfahrvorgang (starke Schwankungen im Umfeld)"
            else:
                diagnosis = "Normaler Lastgang (moderate Schwankung)"

        results.append(PeakContextInfo(
            timestamp=peak_ts,
            power_kw=peak_kw,
            duration_blocks=duration_blocks,
            diagnosis=diagnosis
        ))

    return results


def compute_peak_events(
    df_15: pd.DataFrame,
    cap_kw: float,
    interval_minutes: int = 15
) -> PeakEventsResult:
    """
    Identifiziert und analysiert Peak-Ereignisse über dem Cap.

    Args:
        df_15: 15-min aggregierter DataFrame
        cap_kw: Cap-Wert in kW
        interval_minutes: Intervall-Länge in Minuten

    Returns:
        PeakEventsResult mit Statistiken und Events-DataFrame
    """
    over = (pd.to_numeric(df_15["p_kw"], errors="coerce") > float(cap_kw)).fillna(False).to_numpy(dtype=bool)
    idx = cast(pd.DatetimeIndex, df_15.index)

    if len(over) == 0:
        return PeakEventsResult(
            n_events=0,
            avg_duration_min=0.0,
            max_duration_min=0.0,
            max_shift_kw=0.0,
            top_months="—",
            interpretation="Keine Überschreitungen erkannt.",
            events_df=pd.DataFrame(),
        )

    events: List[Dict[str, object]] = []
    i = 0

    while i < len(over):
        if not over[i]:
            i += 1
            continue
        start_i = i
        while i < len(over) and over[i]:
            i += 1
        end_i = i - 1

        start_ts = idx[start_i]
        end_ts = idx[end_i]
        blocks = int(end_i - start_i + 1)
        duration_min = float(blocks * interval_minutes)

        excess = (pd.to_numeric(df_15["p_kw"].iloc[start_i:end_i + 1], errors="coerce") - cap_kw).clip(lower=0)
        max_shift = float(pd.to_numeric(excess, errors="coerce").max()) if not excess.empty else 0.0

        events.append({
            "start": start_ts,
            "end": end_ts,
            "blocks": blocks,
            "duration_min": duration_min,
            "max_shift_kw": max_shift,
        })

    ev = pd.DataFrame(events)
    n_events = int(len(ev))
    avg_dur = float(pd.to_numeric(ev["duration_min"], errors="coerce").mean()) if n_events > 0 else 0.0
    max_dur = float(pd.to_numeric(ev["duration_min"], errors="coerce").max()) if n_events > 0 else 0.0
    max_shift_kw = float(pd.to_numeric(ev["max_shift_kw"], errors="coerce").max()) if n_events > 0 else 0.0

    # Top-Monate ermitteln
    top_months = "—"
    if n_events > 0:
        start_np = ev["start"].to_numpy(dtype="datetime64[ns]")

        def _ym(x: np.datetime64) -> str:
            t = pd.Timestamp(x)
            return f"{t.year:04d}-{t.month:02d}"

        months = pd.Series([_ym(x) for x in start_np], index=ev.index, dtype="string")
        top = months.value_counts().head(3)
        top_months = ", ".join([f"{m}: {int(c)}" for m, c in top.items()])

    # Anteile berechnen
    share_short = float((pd.to_numeric(ev["duration_min"], errors="coerce") <= float(interval_minutes)).mean()) if n_events > 0 else 0.0
    share_long = float((pd.to_numeric(ev["duration_min"], errors="coerce") >= 60.0).mean()) if n_events > 0 else 0.0

    # Peak-Problemtyp und Interpretation
    if n_events == 0:
        interp = "Keine Peak-Ereignisse oberhalb des Caps erkannt."
        problem_type = "Keine Peaks"
    elif share_short >= 0.6:
        interp = (
            "Viele kurze Lastspitzen (≈ ein 15-min Block) → gut geeignet für Lastmanagement/Sequenzierung.\n"
            "Typische Maßnahme: Verbraucher zeitlich staffeln oder harte Gleichzeitigkeit vermeiden."
        )
        problem_type = "Kurzspitzen"
    elif share_long >= 0.4:
        interp = (
            "Signifikanter Anteil längerer Überschreitungen → Hinweis auf Dauerlasten.\n"
            "Typische Maßnahme: Prozess/Grundlast prüfen (z. B. Kühlung, Heizung, Druckluft, Ofenphasen)."
        )
        problem_type = "Langspitzen / Grundlast"
    else:
        interp = "Gemischtes Muster: sowohl kurze als auch längere Peak-Ereignisse."
        problem_type = "Gemischtes Muster"

    return PeakEventsResult(
        n_events=n_events,
        avg_duration_min=avg_dur,
        max_duration_min=max_dur,
        max_shift_kw=max_shift_kw,
        top_months=top_months,
        interpretation=interp,
        events_df=ev,
        peak_problem_type=problem_type,
    )


# ============================================================================
# PHASEN-ANALYSE
# ============================================================================
def compute_unbalance_module(
    df_15: pd.DataFrame,
    threshold_kw: float = 3.0
) -> UnbalanceResult:
    """
    Analysiert Phasen-Unwucht bei 3-Phasen-Daten.

    Args:
        df_15: 15-min aggregierter DataFrame
        threshold_kw: Schwellenwert für Unwucht in kW

    Returns:
        UnbalanceResult mit Analyse-Ergebnissen
    """
    need = {"p1_kw", "p2_kw", "p3_kw"}
    if not need.issubset(set(df_15.columns)):
        return UnbalanceResult(available=False)

    p = df_15[["p1_kw", "p2_kw", "p3_kw"]].copy()
    p = p.apply(lambda s: pd.to_numeric(s, errors="coerce"))
    p = p.dropna(how="any")

    if p.empty:
        return UnbalanceResult(available=False)

    pmax = p.max(axis=1)
    pmin = p.min(axis=1)
    unb = (pmax - pmin).astype(float)

    over = unb > float(threshold_kw)
    share_over = float(over.mean())
    max_unb = float(unb.max())

    dom = p.idxmax(axis=1).value_counts(normalize=True)
    map_phase = {"p1_kw": "L1", "p2_kw": "L2", "p3_kw": "L3"}
    dom_named: Dict[str, float] = {map_phase.get(str(k), str(k)): float(v) for k, v in dom.items()}

    if dom_named:
        dominant_phase_name = max(dom_named, key=lambda k: dom_named[k])
        dominant_share = float(dom_named[dominant_phase_name])
        dominant_phase = f"{dominant_phase_name} ({dominant_share:.0%})"
    else:
        dominant_phase_name, dominant_share = "—", 0.0
        dominant_phase = "—"

    # Empfehlung generieren
    if dominant_phase_name == "L3" and dominant_share >= 0.55:
        rec = (
            "L3 dominiert konsistent.\n"
            "Empfehlung: einphasige Lasten von L3 auf L1/L2 umklemmen/verteilen (Netzschonung & Kapazitätsfreigabe)."
        )
    elif share_over >= 0.2:
        rec = (
            "Unwucht häufig über Schwelle.\n"
            "Empfehlung: Phasenbelegung prüfen (einphasige Verbraucher verteilen), ggf. Elektriker-Check."
        )
    else:
        rec = (
            "Unwucht selten/gering.\n"
            "Empfehlung: kein akuter Handlungsdruck – bei Erweiterungen trotzdem auf symmetrische Verteilung achten."
        )

    return UnbalanceResult(
        available=True,
        share_over=share_over,
        max_unbalance_kw=max_unb,
        dominant_phase=dominant_phase,
        dominant_phase_name=dominant_phase_name,
        dominant_phase_share=dominant_share,
        recommendation=rec,
    )


# ============================================================================
# BLINDLEISTUNGS-ANALYSE
# ============================================================================
def compute_blk_metrics(df_15: pd.DataFrame) -> BlkResult:
    """
    Berechnet Blindleistungs-Kennzahlen.

    Args:
        df_15: 15-min aggregierter DataFrame (mit cos-phi Spalten)

    Returns:
        BlkResult mit Analyse-Ergebnissen
    """
    def q_from_p_cosphi(p: pd.Series, c: pd.Series) -> pd.Series:
        c_np = np.clip(pd.to_numeric(c, errors="coerce").to_numpy(dtype=float), 0.0, 1.0)
        p_np = pd.to_numeric(p, errors="coerce").to_numpy(dtype=float)
        phi = np.arccos(c_np)
        tanphi = np.tan(phi)
        return pd.Series(p_np * tanphi, index=p.index)

    has_phase = all(k in df_15.columns for k in ["p1_kw", "p2_kw", "p3_kw", "c1", "c2", "c3"])
    has_total = all(k in df_15.columns for k in ["p_kw", "c_total"])

    if has_phase:
        p1 = cast(pd.Series, df_15["p1_kw"])
        p2 = cast(pd.Series, df_15["p2_kw"])
        p3 = cast(pd.Series, df_15["p3_kw"])
        c1 = cast(pd.Series, df_15["c1"])
        c2 = cast(pd.Series, df_15["c2"])
        c3 = cast(pd.Series, df_15["c3"])

        q1 = q_from_p_cosphi(p1, c1)
        q2 = q_from_p_cosphi(p2, c2)
        q3 = q_from_p_cosphi(p3, c3)

        pges = pd.to_numeric(p1, errors="coerce").fillna(0.0) + \
               pd.to_numeric(p2, errors="coerce").fillna(0.0) + \
               pd.to_numeric(p3, errors="coerce").fillna(0.0)
        qges = q1.fillna(0.0) + q2.fillna(0.0) + q3.fillna(0.0)

        df_15["q_kvar"] = qges

    elif has_total:
        pges = cast(pd.Series, df_15["p_kw"])
        c = cast(pd.Series, df_15["c_total"])
        qges = q_from_p_cosphi(pges, c)
        df_15["q_kvar"] = qges

    else:
        return BlkResult(available=False)

    p_abs = pd.to_numeric(pges, errors="coerce").abs().replace(0.0, np.nan)
    q_limit = p_abs * 0.4843  # tan(arccos(0.9))
    over = (qges > q_limit) & p_abs.notna()

    denom = float(p_abs.fillna(0.0).sum())
    ratio = float(qges.fillna(0.0).sum()) / denom if denom > 0 else 0.0
    blocks_over = int(over.fillna(False).sum())
    share_over = float(over.fillna(False).mean())
    q95 = float(qges.dropna().quantile(0.95)) if qges.notna().any() else 0.0

    # Bewertung
    if ratio > 0.5:
        assessment = "Wirtschaftlich oft sinnvoll (Ratio > 0,5).\nKonkrete Kosten/Nutzen bitte mit Tarifdaten prüfen."
    elif 0.3 <= ratio <= 0.5:
        assessment = "Technisch empfohlen (Ratio 0,3–0,5).\nWirtschaftlichkeit hängt von Blindarbeitsregelung/Strafen ab."
    else:
        assessment = "Eher niedrige Blindleistungsrelevanz (Ratio < 0,3).\nTrotzdem Grenzwert-Zeitschritte prüfen (Ausreißer möglich)."

    df_15["q_limit"] = q_limit

    return BlkResult(
        available=True,
        ratio=ratio,
        blocks_over=blocks_over,
        share_over=share_over,
        q95=q95,
        assessment=assessment,
    )


# ============================================================================
# CAP UND SZENARIEN
# ============================================================================
def compute_cap(
    power_kw: pd.Series,
    reduction_goal: str,
    manual_cap_kw: Optional[float],
    manual_value: str
) -> Tuple[float, str]:
    """
    Berechnet den Cap-Wert basierend auf dem Reduktionsziel.

    Args:
        power_kw: Leistungs-Series
        reduction_goal: Ziel ("Bronze", "Silber", "Gold", "Manuell")
        manual_cap_kw: Manueller Cap-Wert (wenn Ziel = "Manuell")
        manual_value: Label für manuellen Cap

    Returns:
        Tuple aus (cap_kw, label)
    """
    goal = (reduction_goal or "").strip()

    if goal in GOAL_TO_QUANTILE:
        q = GOAL_TO_QUANTILE[goal]
        cap = float(pd.to_numeric(power_kw, errors="coerce").quantile(q))
        return cap, f"{goal} (P{int(q * 100)})"

    if goal.lower().startswith("manuell") or goal == "Manuell":
        if manual_cap_kw is None:
            raise ValueError("Manueller Cap wurde ausgewählt, aber manual_cap_kw ist None.")
        label = manual_value or f"{manual_cap_kw:.1f} kW"
        return float(manual_cap_kw), f"Manuell ({label})"

    return float(pd.to_numeric(power_kw, errors="coerce").max()), "Kein Ziel (Fallback)"


def tariff_for_util_hours(t: Tariffs, util_hours: float) -> Tuple[float, float, str]:
    """
    Ermittelt den anzuwendenden Tarif basierend auf Benutzungsstunden.

    Args:
        t: Tarifkonfiguration
        util_hours: Benutzungsstunden (h/a)

    Returns:
        Tuple aus (work_ct, demand_eur, label)
    """
    if util_hours < t.switch_hours:
        return t.work_ct_low, t.demand_eur_kw_a_low, f"< {t.switch_hours:.0f} h/a"
    return t.work_ct_high, t.demand_eur_kw_a_high, f"> {t.switch_hours:.0f} h/a"


def compute_scenario(
    name: str,
    cap_kw: float,
    cap_label: str,
    annual_energy_kwh: float,
    peak_before_kw: float,
    tariffs: Tariffs,
    df_15: pd.DataFrame,
    block_h: float,
) -> Scenario:
    """
    Berechnet ein Peak-Shaving-Szenario.

    Args:
        name: Name des Szenarios
        cap_kw: Cap-Wert in kW
        cap_label: Label für den Cap
        annual_energy_kwh: Jährliche Energie in kWh
        peak_before_kw: Peak vor Shaving in kW
        tariffs: Tarifkonfiguration
        df_15: 15-min aggregierter DataFrame
        block_h: Block-Länge in Stunden

    Returns:
        Scenario-Objekt mit allen berechneten Werten
    """
    peak_after = float(min(float(peak_before_kw), float(cap_kw)))
    util_after = (annual_energy_kwh / peak_after) if peak_after > 0 else 0.0

    work_ct_before, demand_eur_before, label_before = tariff_for_util_hours(
        tariffs, (annual_energy_kwh / peak_before_kw) if peak_before_kw > 0 else 0.0
    )
    work_ct_after, demand_eur_after, label_after = tariff_for_util_hours(tariffs, util_after)

    tariff_switched = (label_before != label_after)

    cost_after = annual_energy_kwh * (work_ct_after / 100.0) + peak_after * demand_eur_after
    cost_before = annual_energy_kwh * (work_ct_before / 100.0) + float(peak_before_kw) * demand_eur_before
    savings = cost_before - cost_after

    excess_kw = (pd.to_numeric(df_15["p_kw"], errors="coerce") - float(cap_kw)).clip(lower=0)
    blocks_over = int((excess_kw > 0).sum())
    share_over = float((excess_kw > 0).mean())
    kwh_to_shift = float((excess_kw * block_h).sum())

    return Scenario(
        name=name,
        cap_kw=float(cap_kw),
        cap_label=cap_label,
        peak_after_kw=peak_after,
        util_hours_after=float(util_after),
        tariff_label_after=label_after,
        tariff_switched=tariff_switched,
        cost_after=float(cost_after),
        savings_eur=float(savings),
        blocks_over_cap=blocks_over,
        share_over_cap=share_over,
        kwh_to_shift=kwh_to_shift,
    )


# ============================================================================
# EMPFEHLUNGEN
# ============================================================================
def build_recommendations(
    mod1: PeakEventsResult,
    mod2: UnbalanceResult,
    blk: BlkResult,
    util_hours_before: float,
    util_hours_after: float,
    p_verschiebbar_kw: float,
    p_gesamt_kw: float,
    tariffs: Tariffs,
) -> List[Recommendation]:
    """
    Generiert Handlungsempfehlungen basierend auf der Analyse.

    Args:
        mod1: Peak-Events-Ergebnis
        mod2: Unwucht-Ergebnis
        blk: Blindleistungs-Ergebnis
        util_hours_before: Benutzungsstunden vorher
        util_hours_after: Benutzungsstunden nachher
        p_verschiebbar_kw: Verschiebbare Leistung in kW
        p_gesamt_kw: Gesamtleistung in kW
        tariffs: Tarifkonfiguration

    Returns:
        Sortierte Liste von Empfehlungen
    """
    recs: List[Recommendation] = []

    ev = mod1.events_df
    if ev is None or ev.empty:
        n_short = 0
        n_long = 0
        max_dur = 0.0
        share_short = 0.0
    else:
        dur = pd.to_numeric(ev["duration_min"], errors="coerce").dropna()
        n_short = int((dur <= 15.0).sum())
        n_long = int((dur >= 60.0).sum())
        max_dur = float(dur.max()) if not dur.empty else 0.0
        share_short = float(n_short / max(int(len(dur)), 1))

    dominant_agg = (p_gesamt_kw > 0) and (p_verschiebbar_kw / p_gesamt_kw > 0.40)

    unb_kw = float(mod2.max_unbalance_kw) if mod2.available else 0.0
    dom_phase = mod2.dominant_phase_name if mod2.available else "—"
    dom_share = float(mod2.dominant_phase_share) if mod2.available else 0.0

    ratio_qp = float(blk.ratio) if blk.available else 0.0
    share_cosphi_under_0_9 = float(blk.share_over) if blk.available else 0.0

    crossed_to_high = (util_hours_before < tariffs.switch_hours) and (util_hours_after > tariffs.switch_hours)

    # Z1: Viele kurze Peaks
    if (n_short > 10) and (share_short > 0.50):
        recs.append(Recommendation(
            code="Z1",
            category="Peak-Strategie – Quick Win",
            trigger=f"{n_short} kurze Peaks (≤15 min), Anteil kurz {share_short:.0%}",
            action="Quick Win: Startzeiten staffeln. Beispiel: Max. 1 Ofen pro 15-min-Fenster aufheizen; Spülmaschinen außerhalb der Backzeiten nutzen.",
            priority="Quick Win",
        ))

    # Z2: Dominante Aggregation
    if dominant_agg:
        recs.append(Recommendation(
            code="Z2",
            category="Peak-Strategie – Investition erforderlich",
            trigger=f"Verschiebbar ~{(p_verschiebbar_kw / p_gesamt_kw):.0%} der Gesamtleistung",
            action="Installation eines Lastabwurfrelais für nicht-kritische Lasten (z. B. Lüftung, Warmwasser) bei Annäherung an das Cap.",
            priority="Investition erforderlich",
        ))

    # Z4: Lange Peak-Ereignisse
    if (n_long >= 3) and (max_dur > 60.0):
        recs.append(Recommendation(
            code="Z4",
            category="Peak-Strategie",
            trigger=f"{n_long} lange Peak-Ereignisse (≥60 min), max. Dauer {max_dur:.0f} min",
            action="Prozess-/Produktionsplanung: Backfenster in Nebenzeiten (NT) verschieben, um Überschneidungen mit Heizungspeaks zu vermeiden.",
            priority="",
        ))

    # Z6: Sehr lange Dauerlast
    if max_dur >= 120.0:
        recs.append(Recommendation(
            code="Z6",
            category="Peak-Strategie – Investition erforderlich",
            trigger=f"Max. Peak-Dauer {max_dur:.0f} min (Hinweis auf Dauerlast: Kälte/Heizung/Grundlast)",
            action="Technische Effizienzprüfung: veraltete Aggregate (>15 J.) gegen drehzahlgeregelte Modelle tauschen; Sollwert-Optimierung Kühlräume um +1–2 K.",
            priority="Investition erforderlich",
        ))

    ratio_shift = (p_verschiebbar_kw / p_gesamt_kw) if p_gesamt_kw > 0 else 0.0

    # Z7/Z8/Z9: Lastmanagement-Empfehlungen
    if ratio_shift < 0.10 and p_gesamt_kw > 0:
        recs.append(Recommendation(
            code="Z7",
            category="Wirtschaftlichkeit (LM) – Quick Win",
            trigger=f"P_verschiebbar ≈ {ratio_shift:.0%} von P_gesamt",
            action="Quick Win: Fokus auf Mitarbeiterschulung und manuelle Schaltpläne (Verhaltensänderung).",
            priority="Quick Win",
        ))
    elif 0.10 <= ratio_shift <= 0.30:
        recs.append(Recommendation(
            code="Z8",
            category="Wirtschaftlichkeit (LM)",
            trigger=f"P_verschiebbar ≈ {ratio_shift:.0%} von P_gesamt",
            action="Automatisches Lastmanagement mit Prioritäten (z. B. Kompressoren & Spülmaschinen) empfohlen.",
            priority="",
        ))
    elif ratio_shift > 0.30:
        recs.append(Recommendation(
            code="Z9",
            category="Wirtschaftlichkeit (LM) – Investition erforderlich",
            trigger=f"P_verschiebbar ≈ {ratio_shift:.0%} von P_gesamt",
            action="Investition erforderlich: Vollwertiges Lastmanagementsystem mit Messung je Hauptverbraucher und PV-Integration wirtschaftlich hoch attraktiv.",
            priority="Investition erforderlich",
        ))

    # Z10: Phasen-Unwucht
    if (unb_kw > 3.0) and (dom_share > 0.80):
        recs.append(Recommendation(
            code="Z10",
            category="Technik (Phasen/Unwucht) – Quick Win",
            trigger=f"Unwucht max. {unb_kw:.1f} kW, dominante Phase {dom_phase} ({dom_share:.0%})",
            action="Quick Win: Elektriker-Check & Umklemmung. Einphasige Großverbraucher von der dominanten Phase auf schwächere Phasen verteilen.",
            priority="Quick Win",
        ))

    # Z13: Blindleistung
    if (ratio_qp > 0.4) or (share_cosphi_under_0_9 > 0.40):
        recs.append(Recommendation(
            code="Z13",
            category="Technik (Blindleistung) – Investition erforderlich",
            trigger=f"Ratio ΣQ/ΣP={ratio_qp:.2f} oder Anteil Grenzwert-Verletzung ≈ {share_cosphi_under_0_9:.0%}",
            action="Investition erforderlich: Blindleistungskompensation (BLK) dringend. Ziel-cos φ: 0,95–0,98; Amortisation oft < 3 Jahre.",
            priority="Investition erforderlich",
        ))

    # Z14: Tarifwechsel
    if crossed_to_high:
        recs.append(Recommendation(
            code="Z14",
            category="Tarif-Strategie",
            trigger=f"Benutzungsdauer vorher {util_hours_before:.0f} h/a → nachher {util_hours_after:.0f} h/a (Schwelle {tariffs.switch_hours:.0f})",
            action="Tarif neu verhandeln: Durch höhere Benutzungsdauer ist ein Wechsel in günstigere Netznutzungsgruppen möglich.",
            priority="",
        ))

    # Fallback
    if not recs:
        recs.append(Recommendation(
            code="—",
            category="Allgemein",
            trigger="Keine klaren Trigger über den definierten Schwellenwerten erkannt.",
            action="Empfehlung: Cap/Schwellenwerte prüfen (manueller Cap), weitere Messdauer erhöhen, oder Hauptverbraucher separat messen (Submeter) für bessere Ursachenanalyse.",
            priority="",
        ))

    # Sortieren nach Priorität
    prio_rank = {"Quick Win": 0, "Investition erforderlich": 1, "": 2}
    recs_sorted = sorted(recs, key=lambda r: (prio_rank.get(r.priority, 9), r.code))

    return recs_sorted
