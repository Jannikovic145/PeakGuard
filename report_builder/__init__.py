# report_builder/__init__.py
"""
PeakGuard Report Builder - Modularisierte Lastgang-Analyse

Dieses Modul bietet alle Funktionen zur Analyse von Stromverbrauchsdaten
und zur Generierung professioneller PDF-Reports.

Module:
- config: Konfiguration, Design-Tokens, Profile
- models: Datenmodelle und Dataclasses
- utils: Hilfsfunktionen (Parsing, Formatierung, Validierung)
- analytics: Analyse-Funktionen (Peak-Erkennung, Berechnungen)
- visualization: Matplotlib-Charts
- components: ReportLab-Komponenten (Tabellen, Karten)
- pdf_builder: PDF-Generierung
"""
from __future__ import annotations

# Version
__version__ = "4.1.0"

# Config exports
from .config import (
    DesignTokens,
    PeakGuardColors,
    GOAL_TO_QUANTILE,
    Tariffs,
    ReportConfig,
    ReportProfile,
    PROFILE_LITE,
    PROFILE_STANDARD,
    PROFILE_PRO,
    DEFAULT_CONFIG,
    apply_intelligent_triggers,
)

# Model exports
from .models import (
    NumberLike,
    TableData,
    PeakEventsResult,
    PeakContextInfo,
    UnbalanceResult,
    BlkResult,
    Recommendation,
    Scenario,
    ReportInput,
    AnalysisResult,
)

# Utils exports
from .utils import (
    setup_logging,
    RobustNumericParser,
    fmt_num,
    fmt_pct,
    fmt_date,
    get_temp_path,
    validate_dataframe,
    sanitize_filename,
    infer_resolution_minutes,
    calculate_missing_quote,
)

# Analytics exports
from .analytics import (
    prepare_raw_power_data,
    aggregate_to_interval,
    analyze_top_peaks,
    compute_peak_events,
    compute_unbalance_module,
    compute_blk_metrics,
    compute_cap,
    tariff_for_util_hours,
    compute_scenario,
    build_recommendations,
)

# Visualization exports
from .visualization import (
    make_timeseries_plot,
    make_duration_curve,
    make_heatmap,
    make_monthly_peaks_bar,
    make_events_scatter,
    make_peak_context_plot,
    make_blk_plot,
)

# Components exports
from .components import (
    get_custom_styles,
    TableFactory,
    create_info_table,
    create_data_table,
    create_scenario_table,
    create_recommendations_table,
    create_peaks_table,
    create_kpi_card,
    create_action_card,
    create_scenario_card,
    add_chart_with_caption,
    build_recommendations_table_rows,
    build_top_peaks_rows,
)

# PDF Builder exports
from .pdf_builder import (
    build_pdf_report,
    build_executive_summary,
    build_glossary,
    add_page_template,
)

# Extended Analytics exports
from .extended_analytics import (
    CO2Result,
    ROIResult,
    BatterySpec,
    LoadForecastResult,
    CompensationROIResult,
    compute_co2_analysis,
    compute_battery_roi,
    compute_load_forecast,
    compute_compensation_roi,
)

# Export functions
from .export import (
    ExportConfig,
    export_to_excel,
    export_to_excel_bytes,
    export_to_powerpoint,
    export_to_powerpoint_bytes,
    export_analysis_to_csv,
)

# ML Analytics exports
from .ml_analytics import (
    ForecastResult,
    AnomalyResult,
    PeakPrediction,
    compute_load_forecast_ml,
    detect_anomalies,
    predict_peak_probability,
    create_time_features,
)


# Convenience alias für Rückwärtskompatibilität
# Diese Funktion wurde früher direkt aus report_builder.py importiert
def compute_blk_metrics_15min(df_15):
    """Alias für compute_blk_metrics (Rückwärtskompatibilität)"""
    return compute_blk_metrics(df_15)


__all__ = [
    # Version
    "__version__",

    # Config
    "DesignTokens",
    "PeakGuardColors",
    "GOAL_TO_QUANTILE",
    "Tariffs",
    "ReportConfig",
    "ReportProfile",
    "PROFILE_LITE",
    "PROFILE_STANDARD",
    "PROFILE_PRO",
    "DEFAULT_CONFIG",
    "apply_intelligent_triggers",

    # Models
    "NumberLike",
    "TableData",
    "PeakEventsResult",
    "PeakContextInfo",
    "UnbalanceResult",
    "BlkResult",
    "Recommendation",
    "Scenario",
    "ReportInput",
    "AnalysisResult",

    # Utils
    "setup_logging",
    "RobustNumericParser",
    "fmt_num",
    "fmt_pct",
    "fmt_date",
    "get_temp_path",
    "validate_dataframe",
    "sanitize_filename",
    "infer_resolution_minutes",
    "calculate_missing_quote",

    # Analytics
    "prepare_raw_power_data",
    "aggregate_to_interval",
    "analyze_top_peaks",
    "compute_peak_events",
    "compute_unbalance_module",
    "compute_blk_metrics",
    "compute_blk_metrics_15min",  # Alias
    "compute_cap",
    "tariff_for_util_hours",
    "compute_scenario",
    "build_recommendations",

    # Visualization
    "make_timeseries_plot",
    "make_duration_curve",
    "make_heatmap",
    "make_monthly_peaks_bar",
    "make_events_scatter",
    "make_peak_context_plot",
    "make_blk_plot",

    # Components
    "get_custom_styles",
    "TableFactory",
    "create_info_table",
    "create_data_table",
    "create_scenario_table",
    "create_recommendations_table",
    "create_peaks_table",
    "create_kpi_card",
    "create_action_card",
    "create_scenario_card",
    "add_chart_with_caption",
    "build_recommendations_table_rows",
    "build_top_peaks_rows",

    # PDF Builder
    "build_pdf_report",
    "build_executive_summary",
    "build_glossary",
    "add_page_template",

    # Extended Analytics
    "CO2Result",
    "ROIResult",
    "BatterySpec",
    "LoadForecastResult",
    "CompensationROIResult",
    "compute_co2_analysis",
    "compute_battery_roi",
    "compute_load_forecast",
    "compute_compensation_roi",

    # Export
    "ExportConfig",
    "export_to_excel",
    "export_to_excel_bytes",
    "export_to_powerpoint",
    "export_to_powerpoint_bytes",
    "export_analysis_to_csv",

    # ML Analytics
    "ForecastResult",
    "AnomalyResult",
    "PeakPrediction",
    "compute_load_forecast_ml",
    "detect_anomalies",
    "predict_peak_probability",
    "create_time_features",
]
