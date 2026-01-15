# report_builder.py  (v3.0: Handlungsempfehlungen-Seite + 15-min Demand Logik, Pylance-sicher)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Union, cast

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    Flowable,
)

# Requested: Gold = P85
GOAL_TO_QUANTILE: Dict[str, float] = {"Bronze": 0.95, "Silber": 0.90, "Gold": 0.85}

NumberLike = Union[int, float, np.number]
TableData = List[List[object]]


@dataclass(frozen=True)
class Tariffs:
    switch_hours: float = 2500.0
    work_ct_low: float = 8.27
    demand_eur_kw_a_low: float = 19.93
    work_ct_high: float = 4.25
    demand_eur_kw_a_high: float = 120.43


@dataclass
class PeakEventsResult:
    n_events: int
    avg_duration_min: float
    max_duration_min: float
    max_shift_kw: float
    top_months: str
    interpretation: str
    events_df: pd.DataFrame


@dataclass
class UnbalanceResult:
    available: bool
    share_over: float = 0.0
    max_unbalance_kw: float = 0.0
    dominant_phase: str = "—"          # e.g. "L3 (55%)"
    dominant_phase_name: str = "—"     # e.g. "L3"
    dominant_phase_share: float = 0.0  # 0..1
    recommendation: str = ""


@dataclass(frozen=True)
class BlkResult:
    available: bool
    ratio: float = 0.0
    blocks_over: int = 0
    share_over: float = 0.0
    q95: float = 0.0
    assessment: str = ""


# -----------------------------
# v3.0 Recommendation model
# -----------------------------
@dataclass(frozen=True)
class Recommendation:
    code: str
    category: str
    trigger: str
    action: str
    priority: str  # "Quick Win" | "Investition erforderlich" | ""


def build_pdf_report(
    df: pd.DataFrame,
    out_path: Path,
    timestamp_col: str,
    power_col: Optional[str],
    power_cols: Optional[List[str]] = None,
    power_unit: Optional[str] = "Auto",
    pf_cols: Optional[List[str]] = None,
    source_name: str = "",
    site_name: str = "",
    data_quality: str = "OK",
    meter_type: str = "RLM",
    reduction_goal: str = "Bronze",
    manual_value: str = "",
    manual_cap_kw: Optional[float] = None,
    tariffs: Optional[Tariffs] = None,
    include_reactive: bool = True,
    input_resolution_minutes: Optional[int] = None,
    demand_interval_minutes: int = 15,
) -> None:
    tariffs = tariffs or Tariffs()
    d0 = df.copy()

    # --- Parse timestamps ---
    d0[timestamp_col] = pd.to_datetime(d0[timestamp_col], errors="coerce")
    d0 = d0.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    if d0.empty:
        raise ValueError("Keine gültigen Zeitstempel nach Parsing vorhanden.")

    # --- Build raw power series in kW (from canonical columns if present) ---
    raw = _prepare_raw_power_and_optional_phases(
        d0=d0,
        timestamp_col=timestamp_col,
        power_col=power_col,
        power_cols=power_cols,
        power_unit=power_unit or "Auto",
        pf_cols=pf_cols,
    )
    df_raw = raw["df"]
    if df_raw.empty:
        raise ValueError("Keine gültigen Leistungswerte nach Parsing vorhanden.")

    idx_raw = cast(pd.DatetimeIndex, df_raw.index)

    inferred_res = _infer_resolution_minutes(idx_raw) if input_resolution_minutes is None else input_resolution_minutes
    missing_quote = _missing_quote(idx_raw, inferred_res)

    # --- Aggregate to demand interval (15min) ---
    df_15 = _aggregate_to_interval(df_raw, minutes=demand_interval_minutes)
    if df_15.empty:
        raise ValueError("Aggregation auf 15-Minuten ergab keine Daten (prüfe Zeitstempel / Mapping).")

    idx_15 = cast(pd.DatetimeIndex, df_15.index)

    # --- Peak info ---
    peak_15_kw = float(pd.to_numeric(df_15["p_kw"], errors="coerce").max())
    peak_1m_kw: Optional[float] = None
    if inferred_res is not None and inferred_res < demand_interval_minutes:
        peak_1m_kw = float(pd.to_numeric(df_raw["p_kw"], errors="coerce").max())

    # --- Energy on 15-min mean power ---
    block_h = float(demand_interval_minutes) / 60.0
    df_15["kwh"] = pd.to_numeric(df_15["p_kw"], errors="coerce").clip(lower=0) * block_h
    energy_kwh = float(pd.to_numeric(df_15["kwh"], errors="coerce").sum())

    # --- Duration ---
    duration_hours = float((idx_15.max() - idx_15.min()).total_seconds() / 3600.0)
    duration_hours = max(duration_hours, block_h)

    # --- Annualization ---
    annual_energy_kwh = energy_kwh * (8760.0 / duration_hours)

    # --- Utilization hours (before) based on 15-min peak ---
    util_hours_before = (annual_energy_kwh / peak_15_kw) if peak_15_kw > 0 else 0.0
    work_ct_before, demand_eur_before, tariff_label_before = _tariff_for_util_hours(tariffs, util_hours_before)
    cost_before = annual_energy_kwh * (work_ct_before / 100.0) + peak_15_kw * demand_eur_before

    # --- Selected cap scenario ---
    cap_kw_sel, cap_label_sel = compute_cap(df_15["p_kw"], reduction_goal, manual_cap_kw, manual_value)
    scenario_sel = _compute_scenario(
        name="Ausgewählt",
        cap_kw=cap_kw_sel,
        cap_label=cap_label_sel,
        annual_energy_kwh=annual_energy_kwh,
        peak_before_kw=peak_15_kw,
        tariffs=tariffs,
        df_15=df_15,
        block_h=block_h,
    )

    # --- Module 1: Peak events (selected cap) ---
    mod1 = compute_peak_events(df_15, cap_kw_sel, interval_minutes=demand_interval_minutes)

    # --- Module 2: Unbalance ---
    mod2 = compute_unbalance_module(df_15, threshold_kw=3.0)

    # --- Optional BLK ---
    blk = compute_blk_metrics_15min(df_15) if include_reactive else BlkResult(available=False)

    # --- Other package scenarios (always shown) ---
    pkg_scenarios: List[_Scenario] = []
    for g in ["Bronze", "Silber", "Gold"]:
        cap_g, cap_lbl = compute_cap(df_15["p_kw"], g, manual_cap_kw=None, manual_value="")
        pkg_scenarios.append(
            _compute_scenario(
                name=g,
                cap_kw=cap_g,
                cap_label=cap_lbl,
                annual_energy_kwh=annual_energy_kwh,
                peak_before_kw=peak_15_kw,
                tariffs=tariffs,
                df_15=df_15,
                block_h=block_h,
            )
        )

    # --- v3.0: build recommendations from module results ---
    recs = build_recommendations_v3(
        mod1=mod1,
        mod2=mod2,
        blk=blk,
        util_hours_before=util_hours_before,
        util_hours_after=scenario_sel.util_hours_after,
        p_verschiebbar_kw=_estimate_p_verschiebbar_kw(mod1=mod1, scenario=scenario_sel),
        p_gesamt_kw=_estimate_p_gesamt_kw(peak_15_kw),
        tariffs=tariffs,
    )

    # --- Charts ---
    figs: List[Path] = [
        make_timeseries_plot(df_15),
        make_duration_curve(df_15),
        make_heatmap(df_15),
        make_events_scatter(mod1),
    ]
    if blk.available:
        figs.append(make_blk_plot(df_15))

    # --- PDF ---
    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
    )
    styles = getSampleStyleSheet()
    story: List[Flowable] = []

    def p_wrap(text: str) -> Paragraph:
        safe = (text or "—").replace("\n", "<br/>")
        return Paragraph(safe, styles["BodyText"])

    story.append(Paragraph("PeakGuard – Lastgang- & Lastspitzen-Report (v3.0)", styles["Title"]))
    story.append(Spacer(1, 6 * mm))

    period_str = (
        f"{pd.Timestamp(idx_15.min()):%d.%m.%Y %H:%M} – "
        f"{pd.Timestamp(idx_15.max()):%d.%m.%Y %H:%M} ({duration_hours:.1f} h)"
    )

    story.append(
        Table(
            [
                ["Quelle", source_name],
                ["Standort", site_name or "—"],
                ["Zählertyp", meter_type],
                ["Datenqualität", data_quality],
                ["Zeitraum", period_str],
                [
                    "Validierung",
                    f"Input ∆t~{inferred_res if inferred_res is not None else '—'} min | "
                    f"Missing-Quote: {missing_quote:.1%} | Demand-Basis: {demand_interval_minutes} min",
                ],
            ],
            colWidths=[40 * mm, 140 * mm],
            style=TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            ),
        )
    )
    story.append(Spacer(1, 6 * mm))

    # --- KPIs ---
    story.append(Paragraph("Kernkennzahlen (Lastgang-Logik: 15-min Mittelwerte)", styles["Heading2"]))

    util_before_str = f"{fmt_num(util_hours_before, 0, 'h/a')} ({tariff_label_before})"
    rows_kpi: TableData = [
        ["Energie (Messzeitraum)", fmt_num(energy_kwh, 0, "kWh")],
        ["Energie (hochgerechnet)", fmt_num(annual_energy_kwh, 0, "kWh/a")],
        ["Max. Leistung (Peak, 15-min)", fmt_num(peak_15_kw, 1, "kW")],
    ]
    if peak_1m_kw is not None:
        rows_kpi.append(["Max. Leistung (Peak, 1-min Info)", fmt_num(peak_1m_kw, 1, "kW")])

    rows_kpi.extend(
        [
            ["Benutzungsdauer (hochgerechnet)", util_before_str],
            ["Arbeitspreis (akt. Tarif)", f"{work_ct_before:.2f} ct/kWh".replace(".", ",")],
            ["Leistungspreis (akt. Tarif)", f"{demand_eur_before:.2f} €/kW/a".replace(".", ",")],
            ["Kosten (Ist, hochgerechnet)", fmt_num(cost_before, 0, "€/a")],
        ]
    )

    story.append(
        Table(
            rows_kpi,
            colWidths=[95 * mm, 85 * mm],
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            ),
        )
    )
    story.append(Spacer(1, 6 * mm))

    # --- Selected Peak shaving ---
    story.append(Paragraph("Peak-Shaving Ziel (Cap) & rechnerische Wirkung (15-min Basis)", styles["Heading2"]))

    util_after_str = f"{fmt_num(scenario_sel.util_hours_after, 0, 'h/a')} ({scenario_sel.tariff_label_after})"
    tariff_switch_note = (
        "Tarifwechsel möglich (nach Peak-Shaving > Schwelle)."
        if scenario_sel.tariff_switched
        else "Kein Tarifwechsel (weiterhin unter/über Schwelle)."
    )

    rows_sel: TableData = [
        ["Ziel", scenario_sel.cap_label],
        ["Cap", fmt_num(scenario_sel.cap_kw, 1, "kW")],
        ["Peak vorher (15-min)", fmt_num(peak_15_kw, 1, "kW")],
        ["Peak nachher (Cap)", fmt_num(scenario_sel.peak_after_kw, 1, "kW")],
        ["Neue Benutzungsdauer (hochgerechnet)", util_after_str],
        ["Tarifwechsel-Check", tariff_switch_note],
        ["15-min Blöcke über Cap", f"{scenario_sel.blocks_over_cap} ({scenario_sel.share_over_cap:.1%})"],
        ["Energie über Cap (Indikator)", fmt_num(scenario_sel.kwh_to_shift, 1, "kWh")],
        ["Kosten nachher (hochgerechnet)", fmt_num(scenario_sel.cost_after, 0, "€/a")],
        ["Einsparung (rein rechnerisch)", fmt_num(scenario_sel.savings_eur, 0, "€/a")],
    ]

    story.append(
        Table(
            rows_sel,
            colWidths=[95 * mm, 85 * mm],
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            ),
        )
    )
    story.append(Spacer(1, 6 * mm))

    # --- Other scenarios table ---
    story.append(Paragraph("Weitere Szenarien (rechnerisch, inkl. möglichem Tarifwechsel)", styles["Heading3"]))

    rows_other: TableData = [["Szenario", "Cap", "Peak nachher", "Benutzungsdauer", "Kosten nachher", "Einsparung"]]
    for s in pkg_scenarios:
    # Prozentuale Reduzierung relativ zum bisherigen 15-min Peak
        red_pct = 0.0
        if peak_15_kw > 0:
            red_pct = (1.0 - (float(s.peak_after_kw) / float(peak_15_kw))) * 100.0

        peak_cell = Paragraph(
            f"{fmt_num(s.peak_after_kw, 1, 'kW')}<br/><font size=8>Reduzierung: {fmt_pct(red_pct, 1)}</font>",
            styles["BodyText"],
        )

        rows_other.append(
            [
                s.cap_label,
                fmt_num(s.cap_kw, 1, "kW"),
                peak_cell,
                f"{fmt_num(s.util_hours_after, 0, 'h/a')} ({s.tariff_label_after})",
                fmt_num(s.cost_after, 0, "€/a"),
                fmt_num(s.savings_eur, 0, "€/a"),
            ]
        )
    story.append(
        Table(
            rows_other,
            colWidths=[44 * mm, 25 * mm, 28 * mm, 40 * mm, 22 * mm, 21 * mm],
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                    ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ]
            ),
        )
    )
    story.append(Spacer(1, 6 * mm))

    # --- Module 1 ---
    story.append(Paragraph("Modul 1: Peak-Cluster & Shift-Analyse (Ereignisse, 15-min)", styles["Heading2"]))
    story.append(
        Table(
            [
                ["Peak-Ereignisse gesamt", str(mod1.n_events)],
                ["Ø Dauer pro Ereignis", f"{mod1.avg_duration_min:.1f} min".replace(".", ",")],
                ["Max. Dauer (längster Peak)", f"{mod1.max_duration_min:.1f} min".replace(".", ",")],
                ["Max. benötigte Verschiebe-Leistung", f"{mod1.max_shift_kw:.1f} kW".replace(".", ",")],
                ["Top-3 Monate (Anzahl)", mod1.top_months],
                ["Interpretation", p_wrap(mod1.interpretation)],
            ],
            colWidths=[95 * mm, 85 * mm],
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            ),
        )
    )
    story.append(Spacer(1, 6 * mm))

    # --- Module 2 ---
    story.append(Paragraph("Modul 2: Phasen-Symmetrie & Unwucht-Check (15-min)", styles["Heading2"]))
    if not mod2.available:
        story.append(Paragraph("Keine 3-Phasen-Leistungsdaten vorhanden – Unwucht-Check nicht berechnet.", styles["BodyText"]))
    else:
        story.append(
            Table(
                [
                    ["Unwucht-Schwelle", "> 3,0 kW (Pmax_phase − Pmin_phase)"],
                    ["Anteil Blöcke > Schwelle", f"{mod2.share_over:.1%}"],
                    ["Max. Unwucht", f"{mod2.max_unbalance_kw:.1f} kW".replace(".", ",")],
                    ["Dominante Phase", mod2.dominant_phase],
                    ["Empfehlung", p_wrap(mod2.recommendation)],
                ],
                colWidths=[95 * mm, 85 * mm],
                style=TableStyle(
                    [
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                ),
            )
        )
    story.append(Spacer(1, 6 * mm))

    # --- Module 3 (BLK) ---
    story.append(Paragraph("Blindleistung / BLK-Analyse (optional, 15-min)", styles["Heading2"]))
    if not blk.available:
        story.append(Paragraph("Keine ausreichenden cosϕ-/Phasen-Daten vorhanden – Blindleistungsanalyse nicht berechnet.", styles["BodyText"]))
    else:
        story.append(
            Table(
                [
                    ["Ratio ΣQ/ΣP", f"{blk.ratio:.3f}".replace(".", ",")],
                    ["15-min Blöcke > cosϕ=0,9", f"{blk.blocks_over} ({blk.share_over:.1%})"],
                    ["Q95 (Empfehlung)", fmt_num(blk.q95, 1, "kvar")],
                    ["Einschätzung", p_wrap(blk.assessment)],
                ],
                colWidths=[95 * mm, 85 * mm],
                style=TableStyle(
                    [
                        ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                        ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                ),
            )
        )
    story.append(Spacer(1, 6 * mm))

    # =========================
    # v3.0 NEW PAGE:
    # Individuelle Handlungsempfehlungen
    # =========================
    story.append(PageBreak())
    story.append(Paragraph("Individuelle Maßnahmen-Roadmap", styles["Heading2"]))
    story.append(Spacer(1, 2 * mm))

    rec_rows = build_recommendations_table_rows(recs, styles)
    story.append(
        Table(
            rec_rows,
            colWidths=[45 * mm, 55 * mm, 80 * mm],  # Kategorie | Trigger | Maßnahme
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                ]
            ),
        )
    )
    story.append(Spacer(1, 4 * mm))

    # --- Top 20 Lastspitzen (15-min) ---
    story.append(PageBreak())
    story.append(Paragraph("Top 20 Lastspitzen (15-min Mittelwerte)", styles["Heading2"]))
    story.append(Spacer(1, 2 * mm))

    top_rows = build_top_peaks_rows(df_15, n=20)
    story.append(
        Table(
            top_rows,
            colWidths=[10 * mm, 32 * mm, 34 * mm, 34 * mm, 34 * mm],
            style=TableStyle(
                [
                    ("BOX", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("ALIGN", (3, 1), (3, -1), "RIGHT"),
                    ("ALIGN", (4, 1), (4, -1), "RIGHT"),
                ]
            ),
        )
    )
    story.append(Spacer(1, 4 * mm))

    # --- Visuals ---
    story.append(Paragraph("Visualisierungen", styles["Heading2"]))
    for img_path in figs:
        story.append(Spacer(1, 2 * mm))
        story.append(Image(str(img_path), width=170 * mm, height=80 * mm))
        story.append(Spacer(1, 4 * mm))

    doc.build(story)


# -----------------------------
# v3.0 Recommendation logic
# -----------------------------
def _estimate_p_gesamt_kw(peak_15_kw: float) -> float:
    # Proxy: ohne Submeter/Verbraucheraufschlüsselung ist Peak (15-min) die beste "Anschlussleistungs-Näherung".
    return float(max(peak_15_kw, 0.0))


def _estimate_p_verschiebbar_kw(mod1: PeakEventsResult, scenario: "_Scenario") -> float:
    # Proxy: größte benötigte Verschiebe-Leistung aus Events (gegenüber Cap) ist ein robuster Indikator.
    # (Alternativ wäre max(excess_kw) möglich, aber mod1.max_shift_kw basiert genau darauf.)
    return float(max(mod1.max_shift_kw, 0.0))


def build_recommendations_v3(
    mod1: PeakEventsResult,
    mod2: UnbalanceResult,
    blk: BlkResult,
    util_hours_before: float,
    util_hours_after: float,
    p_verschiebbar_kw: float,
    p_gesamt_kw: float,
    tariffs: Tariffs,
) -> List[Recommendation]:
    recs: List[Recommendation] = []

    # ---- Derived variables (PRD inputs) ----
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

    # "Dominantes Aggregat" Proxy: ein einzelnes Ereignis verlangt >40% der Gesamtleistung als Verschiebung
    dominant_agg = (p_gesamt_kw > 0) and (p_verschiebbar_kw / p_gesamt_kw > 0.40)

    # Unwucht
    unb_kw = float(mod2.max_unbalance_kw) if mod2.available else 0.0
    dom_phase = mod2.dominant_phase_name if mod2.available else "—"
    dom_share = float(mod2.dominant_phase_share) if mod2.available else 0.0

    # BLK / cosphi
    ratio_qp = float(blk.ratio) if blk.available else 0.0
    share_cosphi_under_0_9 = float(blk.share_over) if blk.available else 0.0  # proxy: Grenzwertverletzungen

    # Tarifwechsel
    tariff_before = util_hours_before
    tariff_after = util_hours_after
    crossed_to_high = (tariff_before < tariffs.switch_hours) and (tariff_after > tariffs.switch_hours)

    # ---- Modul A: Peak-Strategie (Cluster-Analyse) ----
    # Z1
    if (n_short > 10) and (share_short > 0.50):
        recs.append(
            Recommendation(
                code="Z1",
                category="Peak-Strategie",
                trigger=f"{n_short} kurze Peaks (≤15 min), Anteil kurz {share_short:.0%}",
                action="Quick Win: Startzeiten staffeln. Beispiel: Max. 1 Ofen pro 15-min-Fenster aufheizen; Spülmaschinen außerhalb der Backzeiten nutzen.",
                priority="Quick Win",
            )
        )

    # Z2
    if dominant_agg:
        recs.append(
            Recommendation(
                code="Z2",
                category="Peak-Strategie",
                trigger=f"Verschiebbar ~{(p_verschiebbar_kw / p_gesamt_kw):.0%} der Gesamtleistung (Proxy für dominantes Aggregat)",
                action="Installation eines Lastabwurfrelais für nicht-kritische Lasten (z. B. Lüftung, Warmwasser) bei Annäherung an das Cap.",
                priority="",
            )
        )

    # Z4
    if (n_long >= 3) and (max_dur > 60.0):
        recs.append(
            Recommendation(
                code="Z4",
                category="Peak-Strategie",
                trigger=f"{n_long} lange Peak-Ereignisse (≥60 min), max. Dauer {max_dur:.0f} min",
                action="Prozess-/Produktionsplanung: Backfenster in Nebenzeiten (NT) verschieben, um Überschneidungen mit Heizungspeaks zu vermeiden.",
                priority="",
            )
        )

    # Z6
    if max_dur >= 120.0:
        recs.append(
            Recommendation(
                code="Z6",
                category="Peak-Strategie",
                trigger=f"Max. Peak-Dauer {max_dur:.0f} min (Hinweis auf Dauerlast: Kälte/Heizung/Grundlast)",
                action="Technische Effizienzprüfung: veraltete Aggregate (>15 J.) gegen drehzahlgeregelte Modelle tauschen; Sollwert-Optimierung Kühlräume um +1–2 K.",
                priority="Investition erforderlich",
            )
        )

    # ---- Modul B: Wirtschaftlichkeit Hardware (kW-Verschiebung) ----
    ratio_shift = (p_verschiebbar_kw / p_gesamt_kw) if p_gesamt_kw > 0 else 0.0

    # Z7
    if ratio_shift < 0.10 and p_gesamt_kw > 0:
        recs.append(
            Recommendation(
                code="Z7",
                category="Wirtschaftlichkeit (LM)",
                trigger=f"P_verschiebbar ≈ {ratio_shift:.0%} von P_gesamt",
                action="Quick Win: Fokus auf Mitarbeiterschulung und manuelle Schaltpläne (Verhaltensänderung).",
                priority="Quick Win",
            )
        )
    # Z8
    elif 0.10 <= ratio_shift <= 0.30:
        recs.append(
            Recommendation(
                code="Z8",
                category="Wirtschaftlichkeit (LM)",
                trigger=f"P_verschiebbar ≈ {ratio_shift:.0%} von P_gesamt",
                action="Automatisches Lastmanagement mit Prioritäten (z. B. Kompressoren & Spülmaschinen) empfohlen.",
                priority="",
            )
        )
    # Z9
    elif ratio_shift > 0.30:
        recs.append(
            Recommendation(
                code="Z9",
                category="Wirtschaftlichkeit (LM)",
                trigger=f"P_verschiebbar ≈ {ratio_shift:.0%} von P_gesamt",
                action="Investition erforderlich: Vollwertiges Lastmanagementsystem mit Messung je Hauptverbraucher und PV-Integration wirtschaftlich hoch attraktiv.",
                priority="Investition erforderlich",
            )
        )

    # ---- Modul C: Technische Korrekturen (Unwucht & Blindleistung) ----
    # Z10
    if (unb_kw > 3.0) and (dom_share > 0.80):
        recs.append(
            Recommendation(
                code="Z10",
                category="Technik (Phasen/Unwucht)",
                trigger=f"Unwucht max. {unb_kw:.1f} kW, dominante Phase {dom_phase} ({dom_share:.0%})",
                action="Quick Win: Elektriker-Check & Umklemmung. Einphasige Großverbraucher von der dominanten Phase auf schwächere Phasen verteilen.",
                priority="Quick Win",
            )
        )

    # Z13
    if (ratio_qp > 0.4) or (share_cosphi_under_0_9 > 0.40):
        recs.append(
            Recommendation(
                code="Z13",
                category="Technik (Blindleistung)",
                trigger=f"Ratio ΣQ/ΣP={ratio_qp:.2f} oder Anteil Grenzwert-Verletzung ≈ {share_cosphi_under_0_9:.0%}",
                action="Investition erforderlich: Blindleistungskompensation (BLK) dringend. Ziel-cos φ: 0,95–0,98; Amortisation oft < 3 Jahre.",
                priority="Investition erforderlich",
            )
        )

    # ---- Modul D: Tarif-Strategie ----
    # Z14
    if crossed_to_high:
        recs.append(
            Recommendation(
                code="Z14",
                category="Tarif-Strategie",
                trigger=f"Benutzungsdauer vorher {tariff_before:.0f} h/a → nachher {tariff_after:.0f} h/a (Schwelle {tariffs.switch_hours:.0f})",
                action="Tarif neu verhandeln: Durch höhere Benutzungsdauer ist ein Wechsel in günstigere Netznutzungsgruppen möglich.",
                priority="",
            )
        )

    # Fallback, wenn nichts triggert
    if not recs:
        recs.append(
            Recommendation(
                code="—",
                category="Allgemein",
                trigger="Keine klaren Trigger über den definierten Schwellenwerten erkannt.",
                action="Empfehlung: Cap/Schwellenwerte prüfen (manueller Cap), weitere Messdauer erhöhen, oder Hauptverbraucher separat messen (Submeter) für bessere Ursachenanalyse.",
                priority="",
            )
        )

    # Priorisierung: Quick Wins zuerst, dann Investitionen, dann Rest
    prio_rank = {"Quick Win": 0, "Investition erforderlich": 1, "": 2}
    recs_sorted = sorted(recs, key=lambda r: (prio_rank.get(r.priority, 9), r.code))

    return recs_sorted


def build_recommendations_table_rows(recs: List[Recommendation], styles) -> TableData:
    def p(text: str) -> Paragraph:
        return Paragraph((text or "—").replace("\n", "<br/>"), styles["BodyText"])

    rows: TableData = [["Kategorie", "Identifizierter Trigger", "Empfohlene Maßnahme"]]
    for r in recs:
        cat = r.category
        if r.priority:
            cat = f"{r.category} – {r.priority}"
        # Optional: Code sichtbar machen ohne extra Spalte
        trig = f"{r.code}: {r.trigger}" if r.code and r.code != "—" else r.trigger
        rows.append([p(cat), p(trig), p(r.action)])
    return rows


# -----------------------------
# Data prep / aggregation
# -----------------------------
def _prepare_raw_power_and_optional_phases(
    d0: pd.DataFrame,
    timestamp_col: str,
    power_col: Optional[str],
    power_cols: Optional[List[str]],
    power_unit: str,
    pf_cols: Optional[List[str]],
) -> Dict[str, pd.DataFrame]:
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
        med = float(x.dropna().median()) if x.notna().any() else 0.0
        return (x / 1000.0) if med > 200.0 else x

    out = pd.DataFrame(index=cast(pd.DatetimeIndex, d.index))

    if canonical_p is not None:
        p1 = _to_kw(d[canonical_p[0]])
        p2 = _to_kw(d[canonical_p[1]])
        p3 = _to_kw(d[canonical_p[2]])
        out["p1_kw"] = p1
        out["p2_kw"] = p2
        out["p3_kw"] = p3
        out["p_kw"] = p1.fillna(0.0) + p2.fillna(0.0) + p3.fillna(0.0)

        if canonical_c is not None:
            out["c1"] = pd.to_numeric(d[canonical_c[0]], errors="coerce").clip(lower=0.0, upper=1.0)
            out["c2"] = pd.to_numeric(d[canonical_c[1]], errors="coerce").clip(lower=0.0, upper=1.0)
            out["c3"] = pd.to_numeric(d[canonical_c[2]], errors="coerce").clip(lower=0.0, upper=1.0)

    elif power_cols is not None and len(power_cols) == 3:
        p1 = _to_kw(d[power_cols[0]])
        p2 = _to_kw(d[power_cols[1]])
        p3 = _to_kw(d[power_cols[2]])
        out["p1_kw"] = p1
        out["p2_kw"] = p2
        out["p3_kw"] = p3
        out["p_kw"] = p1.fillna(0.0) + p2.fillna(0.0) + p3.fillna(0.0)

        if pf_cols is not None and len(pf_cols) == 3:
            out["c1"] = pd.to_numeric(d[pf_cols[0]], errors="coerce").clip(lower=0.0, upper=1.0)
            out["c2"] = pd.to_numeric(d[pf_cols[1]], errors="coerce").clip(lower=0.0, upper=1.0)
            out["c3"] = pd.to_numeric(d[pf_cols[2]], errors="coerce").clip(lower=0.0, upper=1.0)

    else:
        if canonical_total is not None:
            out["p_kw"] = _to_kw(d[canonical_total])
        else:
            if power_col is None:
                raise ValueError("power_col ist None, aber es wurden keine 3 Phasen-Spalten angegeben.")
            out["p_kw"] = _to_kw(d[power_col])

        if canonical_c_total is not None:
            out["c_total"] = pd.to_numeric(d[canonical_c_total], errors="coerce").clip(lower=0.0, upper=1.0)
        elif pf_cols is not None and len(pf_cols) == 1:
            out["c_total"] = pd.to_numeric(d[pf_cols[0]], errors="coerce").clip(lower=0.0, upper=1.0)

    out = out.dropna(subset=["p_kw"])
    return {"df": out}


def _aggregate_to_interval(df_raw: pd.DataFrame, minutes: int) -> pd.DataFrame:
    rule = f"{int(minutes)}T"
    res = df_raw.resample(rule, label="left", closed="left").mean()

    agg = pd.DataFrame(index=cast(pd.DatetimeIndex, res.index))
    agg["p_kw"] = cast(pd.Series, res["p_kw"])

    for c in ["p1_kw", "p2_kw", "p3_kw", "c1", "c2", "c3", "c_total"]:
        if c in res.columns:
            agg[c] = cast(pd.Series, res[c])

    agg = agg.dropna(subset=["p_kw"])
    return agg


def _infer_resolution_minutes(idx: pd.DatetimeIndex) -> Optional[int]:
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


def _missing_quote(idx: pd.DatetimeIndex, resolution_minutes: Optional[int]) -> float:
    if resolution_minutes is None or len(idx) < 2:
        return 0.0
    start, end = idx.min(), idx.max()
    expected = int(((end - start).total_seconds() / 60.0) / float(resolution_minutes)) + 1
    expected = max(expected, 1)
    actual = int(idx.nunique())
    missing = max(expected - actual, 0)
    return float(missing) / float(expected)


# -----------------------------
# Modules
# -----------------------------
def compute_peak_events(df_15: pd.DataFrame, cap_kw: float, interval_minutes: int = 15) -> PeakEventsResult:
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

        excess = (pd.to_numeric(df_15["p_kw"].iloc[start_i : end_i + 1], errors="coerce") - cap_kw).clip(lower=0)
        max_shift = float(pd.to_numeric(excess, errors="coerce").max()) if not excess.empty else 0.0

        events.append(
            {
                "start": start_ts,
                "end": end_ts,
                "blocks": blocks,
                "duration_min": duration_min,
                "max_shift_kw": max_shift,
            }
        )

    ev = pd.DataFrame(events)
    n_events = int(len(ev))
    avg_dur = float(pd.to_numeric(ev["duration_min"], errors="coerce").mean()) if n_events > 0 else 0.0
    max_dur = float(pd.to_numeric(ev["duration_min"], errors="coerce").max()) if n_events > 0 else 0.0
    max_shift_kw = float(pd.to_numeric(ev["max_shift_kw"], errors="coerce").max()) if n_events > 0 else 0.0

    top_months = "—"
    if n_events > 0:
        start_np = ev["start"].to_numpy(dtype="datetime64[ns]")

        def _ym(x: np.datetime64) -> str:
            t = pd.Timestamp(x)
            return f"{t.year:04d}-{t.month:02d}"

        months = pd.Series([_ym(x) for x in start_np], index=ev.index, dtype="string")
        top = months.value_counts().head(3)
        top_months = ", ".join([f"{m}: {int(c)}" for m, c in top.items()])

    share_short = float((pd.to_numeric(ev["duration_min"], errors="coerce") <= float(interval_minutes)).mean()) if n_events > 0 else 0.0
    if n_events == 0:
        interp = "Keine Peak-Ereignisse oberhalb des Caps erkannt."
    elif share_short >= 0.6:
        interp = (
            "Viele kurze Lastspitzen (≈ ein 15-min Block) → gut geeignet für Lastmanagement/Sequenzierung.\n"
            "Typische Maßnahme: Verbraucher zeitlich staffeln oder harte Gleichzeitigkeit vermeiden."
        )
    else:
        interp = (
            "Signifikanter Anteil längerer Überschreitungen → Hinweis auf Dauerlasten.\n"
            "Typische Maßnahme: Prozess/Grundlast prüfen (z. B. Kühlung, Heizung, Druckluft, Ofenphasen)."
        )

    return PeakEventsResult(
        n_events=n_events,
        avg_duration_min=avg_dur,
        max_duration_min=max_dur,
        max_shift_kw=max_shift_kw,
        top_months=top_months,
        interpretation=interp,
        events_df=ev,
    )


def compute_unbalance_module(df_15: pd.DataFrame, threshold_kw: float = 3.0) -> UnbalanceResult:
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


def compute_blk_metrics_15min(df_15: pd.DataFrame) -> BlkResult:
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

        pges = pd.to_numeric(p1, errors="coerce").fillna(0.0) + pd.to_numeric(p2, errors="coerce").fillna(0.0) + pd.to_numeric(p3, errors="coerce").fillna(0.0)
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
    q_limit = p_abs * 0.4843
    over = (qges > q_limit) & p_abs.notna()

    denom = float(p_abs.fillna(0.0).sum())
    ratio = float(qges.fillna(0.0).sum()) / denom if denom > 0 else 0.0
    blocks_over = int(over.fillna(False).sum())
    share_over = float(over.fillna(False).mean())
    q95 = float(qges.dropna().quantile(0.95)) if qges.notna().any() else 0.0

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


# -----------------------------
# Cap / Tariff / Scenarios
# -----------------------------
def compute_cap(power_kw: pd.Series, reduction_goal: str, manual_cap_kw: Optional[float], manual_value: str) -> Tuple[float, str]:
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


def _tariff_for_util_hours(t: Tariffs, util_hours: float) -> Tuple[float, float, str]:
    if util_hours < t.switch_hours:
        return t.work_ct_low, t.demand_eur_kw_a_low, f"< {t.switch_hours:.0f} h/a"
    return t.work_ct_high, t.demand_eur_kw_a_high, f"> {t.switch_hours:.0f} h/a"


@dataclass(frozen=True)
class _Scenario:
    name: str
    cap_kw: float
    cap_label: str
    peak_after_kw: float
    util_hours_after: float
    tariff_label_after: str
    tariff_switched: bool
    cost_after: float
    savings_eur: float
    blocks_over_cap: int
    share_over_cap: float
    kwh_to_shift: float


def _compute_scenario(
    name: str,
    cap_kw: float,
    cap_label: str,
    annual_energy_kwh: float,
    peak_before_kw: float,
    tariffs: Tariffs,
    df_15: pd.DataFrame,
    block_h: float,
) -> _Scenario:
    peak_after = float(min(float(peak_before_kw), float(cap_kw)))
    util_after = (annual_energy_kwh / peak_after) if peak_after > 0 else 0.0

    work_ct_before, demand_eur_before, label_before = _tariff_for_util_hours(
        tariffs, (annual_energy_kwh / peak_before_kw) if peak_before_kw > 0 else 0.0
    )
    work_ct_after, demand_eur_after, label_after = _tariff_for_util_hours(tariffs, util_after)

    tariff_switched = (label_before != label_after)

    cost_after = annual_energy_kwh * (work_ct_after / 100.0) + peak_after * demand_eur_after
    cost_before = annual_energy_kwh * (work_ct_before / 100.0) + float(peak_before_kw) * demand_eur_before
    savings = cost_before - cost_after

    excess_kw = (pd.to_numeric(df_15["p_kw"], errors="coerce") - float(cap_kw)).clip(lower=0)
    blocks_over = int((excess_kw > 0).sum())
    share_over = float((excess_kw > 0).mean())
    kwh_to_shift = float((excess_kw * block_h).sum())

    return _Scenario(
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


# -----------------------------
# Plots
# -----------------------------
def make_timeseries_plot(df_15: pd.DataFrame) -> Path:
    tmp = Path(_tempfile_path("timeseries.png"))
    idx = cast(pd.DatetimeIndex, df_15.index)
    y = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce")

    plt.figure(figsize=(10, 3.2))
    plt.plot(idx, y)
    plt.ylabel("Leistung (kW) – 15-min Mittelwert")
    plt.xlabel("Zeit")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(tmp, dpi=170)
    plt.close()
    return tmp


def make_duration_curve(df_15: pd.DataFrame) -> Path:
    tmp = Path(_tempfile_path("duration.png"))
    s = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce").dropna().sort_values().to_numpy(dtype=float)
    n = int(len(s))
    if n == 0:
        plt.figure(figsize=(10, 3.2))
        plt.text(0.5, 0.5, "Keine Daten", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(tmp, dpi=170)
        plt.close()
        return tmp

    x = 100.0 * (np.arange(1, n + 1) / float(n))

    plt.figure(figsize=(10, 3.2))
    plt.plot(x, s)
    for pct in range(10, 100, 10):
        plt.axvline(pct, linewidth=0.5)

    plt.ylabel("Leistung (kW) – 15-min Mittelwert")
    plt.xlabel("Anteil der 15-min Blöcke (%)")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(tmp, dpi=170)
    plt.close()
    return tmp


def make_heatmap(df_15: pd.DataFrame) -> Path:
    tmp = Path(_tempfile_path("heatmap.png"))
    idx = cast(pd.DatetimeIndex, df_15.index)
    ts_np = idx.to_numpy(dtype="datetime64[ns]")

    def _weekday(x: np.datetime64) -> int:
        return int(pd.Timestamp(x).weekday())

    def _hour(x: np.datetime64) -> int:
        return int(pd.Timestamp(x).hour)

    weekday = np.array([_weekday(x) for x in ts_np], dtype=int)
    hour = np.array([_hour(x) for x in ts_np], dtype=int)

    p = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce").to_numpy(dtype=float)
    dfh = pd.DataFrame({"weekday": weekday, "hour": hour, "p": p})
    pivot = dfh.pivot_table(index="weekday", columns="hour", values="p", aggfunc="mean")
    pivot = pivot.reindex(index=[0, 1, 2, 3, 4, 5, 6], columns=list(range(24)))

    vals = pivot.to_numpy(dtype=float)
    flat = pd.Series(vals.ravel()).dropna()

    if flat.empty:
        bounds = [0.0, 1.0, 2.0, 3.0]
        vals_plot = np.nan_to_num(vals, nan=0.0)
    else:
        q1 = float(flat.quantile(0.33))
        q2 = float(flat.quantile(0.66))
        vmin = float(flat.min())
        vmax = float(flat.max())
        bounds = [vmin - 1e-9, q1, q2, vmax + 1e-9]
        vals_plot = vals

    cmap = ListedColormap(["#dff3df", "#fff6cc", "#ffd6d6"])
    norm = BoundaryNorm(bounds, cmap.N)

    plt.figure(figsize=(10, 3.2))
    plt.imshow(vals_plot, aspect="auto", cmap=cmap, norm=norm)

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                plt.text(j, i, f"{vals[i, j]:.0f}", ha="center", va="center", color="black", fontsize=7)

    plt.yticks(range(7), ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"])
    plt.xticks(range(24), [str(h) for h in range(24)])
    plt.xlabel("Stunde")
    plt.title("Heatmap (Ø Leistung je Wochentag/Stunde) – 15-min Basis")
    plt.tight_layout()
    plt.savefig(tmp, dpi=170)
    plt.close()
    return tmp


def make_events_scatter(mod1: PeakEventsResult) -> Path:
    tmp = Path(_tempfile_path("events.png"))
    ev = mod1.events_df

    plt.figure(figsize=(10, 3.2))
    if ev.empty:
        plt.text(0.5, 0.5, "Keine Peak-Ereignisse", ha="center", va="center")
        plt.axis("off")
    else:
        plt.scatter(ev["duration_min"], ev["max_shift_kw"])
        plt.xlabel("Dauer Peak-Ereignis (min)")
        plt.ylabel("Max. benötigte Verschiebe-Leistung je Ereignis (kW)")
        plt.title("Peak-Ereignisse: Dauer vs. Verschiebe-Leistung")

    plt.tight_layout()
    plt.savefig(tmp, dpi=170)
    plt.close()
    return tmp


def make_blk_plot(df_15: pd.DataFrame) -> Path:
    tmp = Path(_tempfile_path("blk.png"))

    q = pd.to_numeric(cast(pd.Series, df_15.get("q_kvar", pd.Series(index=df_15.index, dtype=float))), errors="coerce")
    qlim = pd.to_numeric(cast(pd.Series, df_15.get("q_limit", pd.Series(index=df_15.index, dtype=float))), errors="coerce")

    idx = cast(pd.DatetimeIndex, df_15.index)

    plt.figure(figsize=(10, 3.2))
    plt.plot(idx, q, label="Q (kvar)")
    plt.plot(idx, qlim, label="Q-Limit (cosϕ=0,9)")
    plt.ylabel("Blindleistung (kvar) – 15-min Basis")
    plt.xlabel("Zeit")
    plt.xticks(rotation=90)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(tmp, dpi=170)
    plt.close()
    return tmp


# -----------------------------
# Top peaks table
# -----------------------------
def build_top_peaks_rows(df_15: pd.DataFrame, n: int = 20) -> TableData:
    idx = cast(pd.DatetimeIndex, df_15.index)

    cosphi: Optional[pd.Series] = None
    if "c_total" in df_15.columns:
        cosphi = pd.to_numeric(cast(pd.Series, df_15["c_total"]), errors="coerce").clip(lower=0.0, upper=1.0)
    elif all(c in df_15.columns for c in ["c1", "c2", "c3"]):
        c1 = pd.to_numeric(cast(pd.Series, df_15["c1"]), errors="coerce").clip(lower=0.0, upper=1.0)
        c2 = pd.to_numeric(cast(pd.Series, df_15["c2"]), errors="coerce").clip(lower=0.0, upper=1.0)
        c3 = pd.to_numeric(cast(pd.Series, df_15["c3"]), errors="coerce").clip(lower=0.0, upper=1.0)
        cosphi = (c1 + c2 + c3) / 3.0

    tmp = pd.DataFrame(
        {
            "ts": idx,
            "p_kw": pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce"),
        }
    )
    if cosphi is not None:
        tmp["cosphi"] = pd.to_numeric(cosphi, errors="coerce").to_numpy(dtype=float)

    tmp = tmp.dropna(subset=["p_kw"]).sort_values("p_kw", ascending=False).head(int(n))

    rows: TableData = [["#", "Datum", "Uhrzeit (15-min Block)", "Leistung (kW)", "cosϕ"]]

    for i, r in enumerate(tmp.itertuples(index=False), start=1):
        ts = cast(pd.Timestamp, getattr(r, "ts"))
        p_kw = float(getattr(r, "p_kw"))
        cval = getattr(r, "cosphi", None)

        date_s = f"{ts:%d.%m.%Y}"
        time_s = f"{ts:%H:%M}"
        p_s = f"{p_kw:.1f}".replace(".", ",") + " kW"

        if cval is None or (isinstance(cval, float) and (np.isnan(cval) or not np.isfinite(cval))):
            c_s = "—"
        else:
            c_s = f"{float(cval):.3f}".replace(".", ",")

        rows.append([str(i), date_s, time_s, p_s, c_s])

    if len(rows) == 1:
        rows.append(["—", "—", "—", "—", "—"])

    return rows


# -----------------------------
# Formatting / utils
# -----------------------------
def fmt_num(x: Optional[NumberLike], decimals: int, suffix: str) -> str:
    if x is None:
        return f"— {suffix}".strip()
    try:
        xf = float(x)
    except Exception:
        return f"— {suffix}".strip()
    s = f"{xf:,.{decimals}f}"
    s = s.replace(",", "X").replace(".", ",").replace("X", ".")
    return f"{s} {suffix}".strip()

def fmt_pct(x: float, decimals: int = 1) -> str:
    s = f"{x:.{decimals}f}".replace(".", ",")
    return f"{s} %"

def _tempfile_path(filename: str) -> str:
    import tempfile
    from uuid import uuid4

    p = Path(tempfile.gettempdir()) / f"peakguard_{uuid4().hex}_{filename}"
    return str(p)
