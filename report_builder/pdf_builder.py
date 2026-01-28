# report_builder/pdf_builder.py
"""
PDF-Generierung für PeakGuard.
Enthält die Hauptfunktion zum Erstellen von PDF-Reports.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, cast

import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .analytics import (
    aggregate_to_interval,
    analyze_top_peaks,
    build_recommendations,
    compute_blk_metrics,
    compute_cap,
    compute_peak_events,
    compute_scenario,
    compute_unbalance_module,
    prepare_raw_power_data,
    tariff_for_util_hours,
)
from .components import (
    add_chart_with_caption,
    build_recommendations_table_rows,
    build_top_peaks_rows,
    create_action_card,
    create_data_table,
    create_info_table,
    create_kpi_card,
    create_peaks_table,
    create_recommendations_table,
    create_scenario_card,
    get_custom_styles,
)
from .config import (
    DesignTokens,
    PeakGuardColors,
    PROFILE_STANDARD,
    ReportProfile,
    Tariffs,
    apply_intelligent_triggers,
)
from .models import BlkResult, Recommendation, ReportInput
from .utils import (
    calculate_missing_quote,
    fmt_num,
    infer_resolution_minutes,
)
from .visualization import (
    make_blk_plot,
    make_duration_curve,
    make_events_scatter,
    make_heatmap,
    make_monthly_peaks_bar,
    make_peak_context_plot,
    make_timeseries_plot,
)


logger = logging.getLogger("peakguard.pdf")


# ============================================================================
# HEADER / FOOTER
# ============================================================================
def add_page_template(canvas, doc, site_name: str) -> None:
    """Fügt Header/Footer zu jeder Seite hinzu"""
    canvas.saveState()

    # Footer
    footer_y = 15 * mm
    page_num = canvas.getPageNumber()

    # Footer Links: Datum
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(PeakGuardColors.GRAY)
    canvas.drawString(18 * mm, footer_y, f"{pd.Timestamp.now():%d.%m.%Y}")

    # Footer Mitte: PeakGuard + Kunde
    canvas.setFont('Helvetica-Bold', 8)
    canvas.setFillColor(PeakGuardColors.PRIMARY)
    footer_text = "PeakGuard Report"
    if site_name:
        footer_text += f" – {site_name}"
    canvas.drawCentredString(A4[0] / 2, footer_y, footer_text)

    # Footer Rechts: Seitenzahl
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(PeakGuardColors.GRAY)
    canvas.drawRightString(A4[0] - 18 * mm, footer_y, f"Seite {page_num}")

    # Disclaimer (nur auf Seite 1)
    if page_num == 1:
        disclaimer_y = footer_y - 8
        canvas.setFont('Helvetica', 6)
        canvas.setFillColor(PeakGuardColors.GRAY)
        disclaimer = "Alle Angaben ohne Gewähr. Berechnungen basieren auf historischen Daten und stellen keine Garantie für zukünftige Einsparungen dar."
        canvas.drawCentredString(A4[0] / 2, disclaimer_y, disclaimer)

    canvas.restoreState()


# ============================================================================
# EXECUTIVE SUMMARY
# ============================================================================
def build_executive_summary(
    period_str: str,
    data_quality_str: str,
    peak_15_kw: float,
    peak_timestamp: pd.Timestamp,
    savings_eur: float,
    problem_type: str,
    top_3_actions: List[Recommendation],
    styles,
) -> List[Flowable]:
    """Erstellt die Executive Summary (Seite 1)"""
    story: List[Flowable] = []

    # Header
    story.append(Paragraph("Executive Summary", styles["ExecTitle"]))
    story.append(Spacer(1, DesignTokens.SPACE_S))

    # Zeitraum & Datenqualität
    info_text = f"<b>Zeitraum:</b> {period_str} | <b>Datenqualität:</b> {data_quality_str}"
    story.append(Paragraph(info_text, styles['BodySmall']))
    story.append(Spacer(1, DesignTokens.SPACE_L))

    # 3 KPI-Kacheln
    peak_time_str = peak_timestamp.strftime("%d.%m.%Y %H:%M") if peak_timestamp else "—"

    kpi_cards = Table(
        [[
            create_kpi_card(
                "Höchster 15-min Peak",
                fmt_num(peak_15_kw, 1, "kW"),
                f"am {peak_time_str}",
                styles
            ),
            create_kpi_card(
                "Einsparpotenzial",
                fmt_num(savings_eur, 0, "€/Jahr"),
                "bei optimalem Szenario",
                styles
            ),
            create_kpi_card(
                "Peak-Problemtyp",
                problem_type,
                "dominantes Muster",
                styles
            ),
        ]],
        colWidths=[DesignTokens.COL_3, DesignTokens.COL_3, DesignTokens.COL_3],
        style=TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.GUTTER),
        ])
    )
    story.append(kpi_cards)
    story.append(Spacer(1, DesignTokens.SPACE_XL))

    # Top-3-Hebel
    story.append(Paragraph("Diese 3 Hebel zuerst", styles["CustomHeading2"]))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    if len(top_3_actions) >= 3:
        action_row = [[
            create_action_card(
                top_3_actions[0].priority or "Maßnahme",
                top_3_actions[0].category,
                top_3_actions[0].action[:120] + "..." if len(top_3_actions[0].action) > 120 else top_3_actions[0].action,
                styles
            ),
            create_action_card(
                top_3_actions[1].priority or "Maßnahme",
                top_3_actions[1].category,
                top_3_actions[1].action[:120] + "..." if len(top_3_actions[1].action) > 120 else top_3_actions[1].action,
                styles
            ),
            create_action_card(
                top_3_actions[2].priority or "Maßnahme",
                top_3_actions[2].category,
                top_3_actions[2].action[:120] + "..." if len(top_3_actions[2].action) > 120 else top_3_actions[2].action,
                styles
            ),
        ]]

        actions_table = Table(
            action_row,
            colWidths=[DesignTokens.COL_3, DesignTokens.COL_3, DesignTokens.COL_3],
            style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.GUTTER),
            ])
        )
        story.append(actions_table)
    else:
        for i, action in enumerate(top_3_actions[:3]):
            story.append(Paragraph(f"<b>{i + 1}. {action.category}</b>", styles['CustomHeading3']))
            story.append(Paragraph(action.action, styles['BodySmall']))
            story.append(Spacer(1, DesignTokens.SPACE_S))

    return story


# ============================================================================
# GLOSSAR
# ============================================================================
def build_glossary(styles) -> List[Flowable]:
    """Erstellt das Glossar 'So lesen Sie den Report'"""
    story: List[Flowable] = []

    story.append(Paragraph("So lesen Sie den Report", styles["CustomHeading2"]))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    glossary_items = [
        ("15-min-Mittelwert", "Leistung gemittelt über 15 Minuten. Basis für Netzentgelte und Peak-Shaving. Nicht zu verwechseln mit Spitzenlast (1-Sekunden-Peak)."),
        ("Cap / Peak-Shaving", "Zielvorgabe: Leistung soll diesen Wert nicht überschreiten. Durch Lastverschiebung oder -abwurf realisierbar."),
        ("Benutzungsstunden (h/a)", "Energie geteilt durch Peak. Hohe Werte (>2500h) → günstiger Tarif möglich. Niedrig (< 2500h) → teurer Leistungspreis."),
        ("Verschiebbare kWh", "Energie oberhalb des Caps, die durch Lastmanagement in andere Zeiten verschoben werden müsste."),
        ("P95 / P90 / P85", "95./90./85. Perzentil der Leistung. Bedeutet: 95%/90%/85% der Zeit liegt die Leistung darunter (Bronze/Silber/Gold)."),
        ("Peak-Problemtyp", "Kurzspitzen: viele kurze Überschreitungen (≤15 min) → Gleichzeitigkeit vermeiden. Langspitzen: lange Überschreitungen (≥60 min) → Grundlast/Prozess optimieren."),
    ]

    for term, explanation in glossary_items:
        story.append(Paragraph(f"<b>{term}:</b> {explanation}", styles['BodySmall']))
        story.append(Spacer(1, DesignTokens.SPACE_S))

    return story


# ============================================================================
# HAUPTFUNKTION
# ============================================================================
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
    profile: Optional[ReportProfile] = None,
    progress_callback: Optional[callable] = None,
) -> None:
    """
    Generiert einen PeakGuard PDF-Report.

    Args:
        df: Input DataFrame mit Lastgangdaten
        out_path: Ausgabepfad für das PDF
        timestamp_col: Name der Zeitstempel-Spalte
        power_col: Name der Leistungsspalte (für Einzelmessung)
        power_cols: Namen der Phasenspalten (für 3-Phasen)
        power_unit: Einheit ("W", "kW", "Auto")
        pf_cols: Namen der cos-phi-Spalten
        source_name: Quellenname für Report
        site_name: Standortname
        data_quality: Datenqualitäts-Bewertung
        meter_type: Zählertyp
        reduction_goal: Reduktionsziel ("Bronze", "Silber", "Gold", "Manuell")
        manual_value: Label für manuellen Cap
        manual_cap_kw: Manueller Cap-Wert in kW
        tariffs: Tarifkonfiguration
        include_reactive: Blindleistungsanalyse einbeziehen
        input_resolution_minutes: Auflösung der Eingangsdaten
        demand_interval_minutes: Aggregationsintervall (Standard: 15 min)
        profile: Report-Profil (Lite/Standard/Pro)
        progress_callback: Optional callback für Fortschrittsanzeige
    """
    tariffs = tariffs or Tariffs()
    profile = profile or PROFILE_STANDARD
    d0 = df.copy()

    def update_progress(step: str, percent: int):
        if progress_callback:
            progress_callback(step, percent)
        logger.info(f"[{percent}%] {step}")

    update_progress("Starte Report-Generierung...", 0)

    # --- Parse timestamps ---
    d0[timestamp_col] = pd.to_datetime(d0[timestamp_col], errors="coerce")
    d0 = d0.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    if d0.empty:
        raise ValueError("Keine gültigen Zeitstempel nach Parsing vorhanden.")

    update_progress("Zeitstempel geparst", 10)

    # --- Build raw power series in kW ---
    raw = prepare_raw_power_data(
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

    inferred_res = infer_resolution_minutes(idx_raw) if input_resolution_minutes is None else input_resolution_minutes
    missing_quote = calculate_missing_quote(idx_raw, inferred_res)

    update_progress("Daten vorbereitet", 20)

    # --- Aggregate to demand interval (15min) ---
    df_15 = aggregate_to_interval(df_raw, minutes=demand_interval_minutes)
    if df_15.empty:
        raise ValueError("Aggregation auf 15-Minuten ergab keine Daten.")

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

    # --- Utilization hours ---
    util_hours_before = (annual_energy_kwh / peak_15_kw) if peak_15_kw > 0 else 0.0
    work_ct_before, demand_eur_before, tariff_label_before = tariff_for_util_hours(tariffs, util_hours_before)
    cost_before = annual_energy_kwh * (work_ct_before / 100.0) + peak_15_kw * demand_eur_before

    update_progress("Kennzahlen berechnet", 30)

    # --- Selected cap scenario ---
    cap_kw_sel, cap_label_sel = compute_cap(df_15["p_kw"], reduction_goal, manual_cap_kw, manual_value)
    scenario_sel = compute_scenario(
        name="Ausgewählt",
        cap_kw=cap_kw_sel,
        cap_label=cap_label_sel,
        annual_energy_kwh=annual_energy_kwh,
        peak_before_kw=peak_15_kw,
        tariffs=tariffs,
        df_15=df_15,
        block_h=block_h,
    )

    # --- Module 1: Peak events ---
    mod1 = compute_peak_events(df_15, cap_kw_sel, interval_minutes=demand_interval_minutes)

    # --- Module 2: Unbalance ---
    mod2 = compute_unbalance_module(df_15, threshold_kw=3.0)

    # --- Optional BLK ---
    blk = compute_blk_metrics(df_15) if include_reactive else BlkResult(available=False)

    update_progress("Analysen abgeschlossen", 40)

    # --- Package scenarios ---
    pkg_scenarios = []
    for g in ["Bronze", "Silber", "Gold"]:
        cap_g, cap_lbl = compute_cap(df_15["p_kw"], g, manual_cap_kw=None, manual_value="")
        pkg_scenarios.append(
            compute_scenario(
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

    # --- Recommendations ---
    recs = build_recommendations(
        mod1=mod1,
        mod2=mod2,
        blk=blk,
        util_hours_before=util_hours_before,
        util_hours_after=scenario_sel.util_hours_after,
        p_verschiebbar_kw=mod1.max_shift_kw,
        p_gesamt_kw=peak_15_kw,
        tariffs=tariffs,
    )

    # --- Intelligente Trigger ---
    profile = apply_intelligent_triggers(
        profile=profile,
        savings_eur=scenario_sel.savings_eur,
        n_peak_events=mod1.n_events,
        blk_available=blk.available,
        unbalance_available=mod2.available,
    )

    update_progress("Szenarien berechnet", 50)

    # --- Charts generieren ---
    figs = []
    if profile.include_heatmap:
        update_progress("Erstelle Zeitreihen-Chart...", 55)
        figs.append(make_timeseries_plot(df_15, cap_kw_sel))
        figs.append(make_duration_curve(df_15, cap_kw_sel))
        figs.append(make_heatmap(df_15))
        figs.append(make_monthly_peaks_bar(df_15, cap_kw_sel))

    if profile.include_peak_cluster:
        update_progress("Erstelle Peak-Event-Chart...", 65)
        figs.append(make_events_scatter(mod1))

    if blk.available and profile.include_blk:
        update_progress("Erstelle Blindleistungs-Chart...", 70)
        figs.append(make_blk_plot(df_15))

    update_progress("Charts erstellt", 75)

    # --- PDF Generation ---
    pdf_title = f"PeakGuard Report - {site_name if site_name else 'Bericht'} - {pd.Timestamp.now():%d.%m.%Y}"
    pdf_author = "PeakGuard"
    pdf_subject = f"Lastgang-Analyse für {site_name if site_name else 'Kunde'}"

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=pdf_title,
        author=pdf_author,
        subject=pdf_subject,
    )

    styles = get_custom_styles()
    story: List[Flowable] = []

    def p_wrap(text: str, style_name: str = 'CustomBody') -> Paragraph:
        safe = (text or "—").replace("\n", "<br/>")
        return Paragraph(safe, styles[style_name])

    # Zeitraum-String
    period_str = (
        f"{pd.Timestamp(idx_15.min()):%d.%m.%Y %H:%M} – "
        f"{pd.Timestamp(idx_15.max()):%d.%m.%Y %H:%M} ({duration_hours:.1f} h)"
    )

    update_progress("Erstelle PDF-Inhalt...", 80)

    # === EXECUTIVE SUMMARY ===
    if profile.include_exec_summary:
        peak_idx = pd.to_numeric(df_15["p_kw"], errors="coerce").idxmax()
        peak_timestamp = pd.Timestamp(peak_idx) if pd.notna(peak_idx) else pd.Timestamp.now()
        data_quality_str = f"{data_quality} | Abdeckung: {(1 - missing_quote) * 100:.1f}%"

        exec_summary_story = build_executive_summary(
            period_str=period_str,
            data_quality_str=data_quality_str,
            peak_15_kw=peak_15_kw,
            peak_timestamp=peak_timestamp,
            savings_eur=scenario_sel.savings_eur,
            problem_type=mod1.peak_problem_type,
            top_3_actions=recs[:3],
            styles=styles,
        )
        story.extend(exec_summary_story)
        story.append(PageBreak())

    # === HEADER (nur wenn kein Exec Summary) ===
    if not profile.include_exec_summary:
        story.append(Paragraph("PeakGuard – Lastgang- & Lastspitzen-Report", styles["CustomTitle"]))
        story.append(Paragraph(
            f"<font color='#{PeakGuardColors.GRAY.hexval()[2:]}'>Version 4.0 | Erstellt: {pd.Timestamp.now():%d.%m.%Y %H:%M}</font>",
            styles["CustomBody"]
        ))
        story.append(Spacer(1, DesignTokens.SPACE_M))

    # === METADATEN TABLE ===
    story.append(create_info_table([
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
    ]))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    # === KERNKENNZAHLEN ===
    story.append(Paragraph("Kernkennzahlen (Lastgang-Logik: 15-min Mittelwerte)", styles["CustomHeading2"]))

    util_before_str = f"{fmt_num(util_hours_before, 0, 'h/a')} ({tariff_label_before})"
    rows_kpi = [
        ["Energie (Messzeitraum)", fmt_num(energy_kwh, 0, "kWh")],
        ["Energie (hochgerechnet)", fmt_num(annual_energy_kwh, 0, "kWh/a")],
        ["Max. Leistung (Peak, 15-min)", fmt_num(peak_15_kw, 1, "kW")],
    ]
    if peak_1m_kw is not None:
        rows_kpi.append(["Max. Leistung (Peak, 1-min Info)", fmt_num(peak_1m_kw, 1, "kW")])

    rows_kpi.extend([
        ["Benutzungsdauer (hochgerechnet)", util_before_str],
        ["Arbeitspreis (akt. Tarif)", f"{work_ct_before:.2f} ct/kWh".replace(".", ",")],
        ["Leistungspreis (akt. Tarif)", f"{demand_eur_before:.2f} €/kW/a".replace(".", ",")],
        ["Kosten (Ist, hochgerechnet)", fmt_num(cost_before, 0, "€/a")],
    ])

    story.append(create_data_table(rows_kpi))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    # === PEAK SHAVING ZIEL ===
    story.append(Paragraph("Peak-Shaving Ziel (Cap) & rechnerische Wirkung", styles["CustomHeading2"]))

    util_after_str = f"{fmt_num(scenario_sel.util_hours_after, 0, 'h/a')} ({scenario_sel.tariff_label_after})"
    tariff_switch_note = (
        "✓ Tarifwechsel möglich (nach Peak-Shaving > Schwelle)"
        if scenario_sel.tariff_switched
        else "○ Kein Tarifwechsel (weiterhin unter/über Schwelle)"
    )

    rows_sel = [
        ["Ziel", scenario_sel.cap_label],
        ["Cap", fmt_num(scenario_sel.cap_kw, 1, "kW")],
        ["Peak vorher (15-min)", fmt_num(peak_15_kw, 1, "kW")],
        ["Peak nachher (Cap)", fmt_num(scenario_sel.peak_after_kw, 1, "kW")],
        ["Neue Benutzungsdauer", util_after_str],
        ["Tarifwechsel-Check", tariff_switch_note],
        ["15-min Blöcke über Cap", f"{scenario_sel.blocks_over_cap} ({scenario_sel.share_over_cap:.1%})"],
        ["Energie über Cap (Indikator)", fmt_num(scenario_sel.kwh_to_shift, 1, "kWh")],
        ["Kosten nachher", fmt_num(scenario_sel.cost_after, 0, "€/a")],
        ["Einsparung (rechnerisch)", fmt_num(scenario_sel.savings_eur, 0, "€/a")],
    ]

    story.append(create_data_table(rows_sel, highlight_last=True))
    story.append(Spacer(1, DesignTokens.SPACE_M))

    # === WEITERE SZENARIEN ===
    if profile.include_scenarios and len(pkg_scenarios) >= 3:
        story.append(Paragraph("Weitere Szenarien (rechnerisch, inkl. Tarifwechsel)", styles["CustomHeading3"]))
        story.append(Spacer(1, DesignTokens.SPACE_M))

        scenario_row = [[
            create_scenario_card(
                name=pkg_scenarios[0].name,
                cap_kw=pkg_scenarios[0].cap_kw,
                peak_after=pkg_scenarios[0].peak_after_kw,
                savings=pkg_scenarios[0].savings_eur,
                util_hours=pkg_scenarios[0].util_hours_after,
                tariff_label=pkg_scenarios[0].tariff_label_after,
                styles=styles
            ),
            create_scenario_card(
                name=pkg_scenarios[1].name,
                cap_kw=pkg_scenarios[1].cap_kw,
                peak_after=pkg_scenarios[1].peak_after_kw,
                savings=pkg_scenarios[1].savings_eur,
                util_hours=pkg_scenarios[1].util_hours_after,
                tariff_label=pkg_scenarios[1].tariff_label_after,
                styles=styles
            ),
            create_scenario_card(
                name=pkg_scenarios[2].name,
                cap_kw=pkg_scenarios[2].cap_kw,
                peak_after=pkg_scenarios[2].peak_after_kw,
                savings=pkg_scenarios[2].savings_eur,
                util_hours=pkg_scenarios[2].util_hours_after,
                tariff_label=pkg_scenarios[2].tariff_label_after,
                styles=styles
            ),
        ]]

        scenarios_table = Table(
            scenario_row,
            colWidths=[DesignTokens.COL_3, DesignTokens.COL_3, DesignTokens.COL_3],
            style=TableStyle([
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 0),
                ('RIGHTPADDING', (0, 0), (-1, -1), DesignTokens.GUTTER),
            ])
        )
        story.append(scenarios_table)
        story.append(Spacer(1, DesignTokens.SPACE_L))

    # === PEAK-CLUSTER ===
    if profile.include_peak_cluster:
        story.append(Paragraph("Peak-Cluster & Ursachenmuster", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_S))

        story.append(create_data_table([
            ["Peak-Ereignisse gesamt", str(mod1.n_events)],
            ["Ø Dauer pro Ereignis", f"{mod1.avg_duration_min:.1f} min".replace(".", ",")],
            ["Max. Verschiebe-Leistung", f"{mod1.max_shift_kw:.1f} kW".replace(".", ",")],
            ["Problemtyp", mod1.peak_problem_type],
        ]))
        story.append(Spacer(1, DesignTokens.SPACE_M))

        story.append(Paragraph("<b>Was heißt das praktisch?</b>", styles["CustomHeading3"]))
        story.append(p_wrap(mod1.interpretation))
        story.append(Spacer(1, DesignTokens.SPACE_L))

    # === PHASEN-SYMMETRIE ===
    if profile.include_phase_unbalance:
        story.append(Paragraph("Phasen-Symmetrie & Unwucht-Check", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_S))

        if not mod2.available:
            story.append(p_wrap("Keine 3-Phasen-Daten vorhanden – Unwucht-Check nicht berechnet."))
        else:
            story.append(create_data_table([
                ["Unwucht-Schwelle", "> 3,0 kW (Pmax − Pmin)"],
                ["Anteil Blöcke > Schwelle", f"{mod2.share_over:.1%}"],
                ["Max. Unwucht", f"{mod2.max_unbalance_kw:.1f} kW".replace(".", ",")],
                ["Dominante Phase", mod2.dominant_phase],
            ]))
            story.append(Spacer(1, DesignTokens.SPACE_M))
            story.append(Paragraph("<b>Was heißt das praktisch?</b>", styles["CustomHeading3"]))
            story.append(p_wrap(mod2.recommendation))

        story.append(Spacer(1, DesignTokens.SPACE_L))

    # === BLINDLEISTUNG ===
    if profile.include_blk:
        story.append(Paragraph("Blindleistung / BLK-Analyse", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_S))

        if not blk.available:
            story.append(p_wrap("Keine cosϕ-/Phasen-Daten vorhanden – Blindleistungsanalyse nicht berechnet."))
        else:
            story.append(create_data_table([
                ["Ratio ΣQ/ΣP", f"{blk.ratio:.3f}".replace(".", ",")],
                ["15-min Blöcke > cosϕ=0,9", f"{blk.blocks_over} ({blk.share_over:.1%})"],
                ["Q95 (Empfehlung)", fmt_num(blk.q95, 1, "kvar")],
            ]))
            story.append(Spacer(1, DesignTokens.SPACE_M))
            story.append(Paragraph("<b>Was heißt das praktisch?</b>", styles["CustomHeading3"]))
            story.append(p_wrap(blk.assessment))

        story.append(Spacer(1, DesignTokens.SPACE_L))

    # === HANDLUNGSEMPFEHLUNGEN ===
    story.append(PageBreak())
    story.append(Paragraph("Individuelle Maßnahmen-Roadmap", styles["CustomHeading2"]))
    story.append(Spacer(1, DesignTokens.SPACE_XS))

    rec_rows = build_recommendations_table_rows(recs, styles)
    story.append(create_recommendations_table(rec_rows))
    story.append(Spacer(1, DesignTokens.SPACE_S))

    # === TOP LASTSPITZEN ===
    if profile.include_top_peaks:
        top_n = 10 if profile.name in ["lite", "standard"] else 20

        story.append(PageBreak())
        story.append(Paragraph(f"Top {top_n} Lastspitzen (15-min Mittelwerte)", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_M))

        top_rows = build_top_peaks_rows(df_15, n=top_n)
        story.append(create_peaks_table(top_rows))
        story.append(Spacer(1, DesignTokens.SPACE_M))

    # === PEAK-KONTEXT (nur Pro) ===
    if profile.include_peak_context:
        story.append(PageBreak())
        story.append(Paragraph("Peak-Kontext: Detailanalyse Top-3-Peaks", styles["CustomHeading2"]))
        story.append(Spacer(1, DesignTokens.SPACE_M))

        top_peaks = analyze_top_peaks(df_15, cap_kw_sel, n=3)

        for i, peak_info in enumerate(top_peaks, start=1):
            story.append(Paragraph(
                f"Peak #{i}: {peak_info.timestamp:%d.%m.%Y %H:%M} ({fmt_num(peak_info.power_kw, 1, 'kW')})",
                styles["CustomHeading3"]
            ))
            story.append(Spacer(1, DesignTokens.SPACE_S))

            story.append(Paragraph(f"<b>Diagnose:</b> {peak_info.diagnosis}", styles["BodySmall"]))
            story.append(Spacer(1, DesignTokens.SPACE_M))

            fig_12h = make_peak_context_plot(df_15, peak_info.timestamp, 12, cap_kw_sel)
            story.extend(add_chart_with_caption(
                fig_12h,
                "12-Stunden-Fenster: Zeigt unmittelbares Umfeld des Peaks.",
                styles,
                width=170 * mm
            ))
            story.append(Spacer(1, DesignTokens.SPACE_M))

            fig_3d = make_peak_context_plot(df_15, peak_info.timestamp, 72, cap_kw_sel)
            story.extend(add_chart_with_caption(
                fig_3d,
                "3-Tage-Fenster: Zeigt größeren Kontext (Wochenmuster erkennbar).",
                styles,
                width=170 * mm
            ))

            if i < len(top_peaks):
                story.append(Spacer(1, DesignTokens.SPACE_L))

    # === GLOSSAR ===
    if profile.include_glossary:
        story.append(PageBreak())
        glossary_story = build_glossary(styles)
        story.extend(glossary_story)

    update_progress("Erstelle Visualisierungen...", 85)

    # === VISUALISIERUNGEN ===
    chart_captions = []
    if profile.include_heatmap:
        chart_captions.extend([
            "Zeitverlauf der Leistung mit Peak-Shaving Cap. Überschreitungen sind rot markiert.",
            "Sortierte Leistungswerte (Jahresdauerlinie). Zeigt, wie oft welche Leistung erreicht wird.",
            "Heatmap: Durchschnittliche Leistung je Wochentag und Uhrzeit. Zeigt typische Lastmuster.",
            "Peaks pro Monat: Anzahl Überschreitungen (Balken) und maximale Leistung (Linie).",
        ])
    if profile.include_peak_cluster:
        chart_captions.append("Peak-Ereignisse: Dauer vs. benötigte Verschiebeleistung. Größere Punkte = höhere Verschiebeleistung.")
    if blk.available and profile.include_blk:
        chart_captions.append("Blindleistungsverlauf mit cosϕ-Grenzwert (0,9). Überschreitungen zeigen Kompensationsbedarf.")

    viz_header = [
        Paragraph("Visualisierungen", styles["CustomHeading2"]),
        Spacer(1, DesignTokens.SPACE_M)
    ]

    if figs:
        first_chart = add_chart_with_caption(figs[0], chart_captions[0] if len(chart_captions) > 0 else "", styles)
        viz_header.extend(first_chart)
        story.append(KeepTogether(viz_header))

        for i, img_path in enumerate(figs[1:], start=1):
            story.append(Spacer(1, DesignTokens.SPACE_M))
            chart_elements = add_chart_with_caption(
                img_path,
                chart_captions[i] if i < len(chart_captions) else "",
                styles
            )
            story.extend(chart_elements)

    update_progress("Schreibe PDF...", 95)

    doc.build(
        story,
        onFirstPage=lambda c, d: add_page_template(c, d, site_name),
        onLaterPages=lambda c, d: add_page_template(c, d, site_name)
    )

    update_progress("Report fertig!", 100)
