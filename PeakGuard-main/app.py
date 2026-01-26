# app.py - PeakGuard v5.1 - Performance Edition
"""
PeakGuard Report Generator - Professional Pitch Dashboard

Features v5.1:
- Modernes, professionelles UI-Design
- Interaktive Plotly-Dashboards
- Animierte KPI-Karten
- Real-time Analyse-Visualisierungen
- Pitch-optimierte Präsentation
- Performance-Optimierung für große Dateien (10MB+)
- Intelligentes Downsampling für Charts
- Session-basiertes Caching
"""
from __future__ import annotations

import io
import tempfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st

# Import aus modularisiertem report_builder
from report_builder import (
    Tariffs,
    build_pdf_report,
    PROFILE_LITE,
    PROFILE_STANDARD,
    PROFILE_PRO,
    RobustNumericParser,
    validate_dataframe,
    sanitize_filename,
    compute_co2_analysis,
    compute_battery_roi,
    ExportConfig,
    export_to_excel_bytes,
    # ML Analytics
    compute_load_forecast_ml,
    detect_anomalies,
    predict_peak_probability,
    ForecastResult,
    AnomalyResult,
)

# Plotly für interaktives Dashboard
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================================
# KONFIGURATION
# ============================================================================
MAX_FILE_SIZE_MB = 100
MAX_ROWS = 500_000
MAX_CHART_POINTS = 5000  # Maximum Datenpunkte für Plotly-Charts
DOWNSAMPLING_THRESHOLD = 10000  # Ab dieser Anzahl Punkte wird gedownsampelt

# Farben (PeakGuard Corporate Design)
COLORS = {
    "primary": "#1E3A5F",      # Dunkelblau
    "secondary": "#E67E22",    # Orange
    "accent": "#27AE60",       # Grün
    "warning": "#E74C3C",      # Rot
    "light": "#ECF0F1",        # Hellgrau
    "dark": "#2C3E50",         # Dunkel
    "gradient_start": "#1E3A5F",
    "gradient_end": "#3498DB",
}


# ============================================================================
# CUSTOM CSS FÜR PROFESSIONELLES DESIGN
# ============================================================================
def inject_custom_css():
    """Injiziert professionelles CSS für Pitch-Präsentation"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1E3A5F 0%, #3498DB 100%);
        padding: 2rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(30, 58, 95, 0.3);
    }

    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .main-header p {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-top: 0.5rem;
    }

    /* KPI Card Styling */
    .kpi-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border: 1px solid rgba(0,0,0,0.05);
        transition: all 0.3s ease;
        height: 100%;
    }

    .kpi-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.12);
    }

    .kpi-value {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A5F;
        margin: 0.5rem 0;
    }

    .kpi-label {
        font-size: 0.9rem;
        color: #7F8C8D;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500;
    }

    .kpi-delta {
        font-size: 0.85rem;
        padding: 0.25rem 0.5rem;
        border-radius: 20px;
        display: inline-block;
        margin-top: 0.5rem;
    }

    .kpi-delta.positive {
        background: rgba(39, 174, 96, 0.15);
        color: #27AE60;
    }

    .kpi-delta.negative {
        background: rgba(231, 76, 60, 0.15);
        color: #E74C3C;
    }

    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #ECF0F1;
    }

    .section-header h2 {
        color: #1E3A5F;
        font-size: 1.5rem;
        font-weight: 600;
        margin: 0;
    }

    .section-icon {
        font-size: 1.5rem;
    }

    /* Chart Container */
    .chart-container {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    .chart-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #2C3E50;
        margin-bottom: 1rem;
    }

    /* Insight Cards */
    .insight-card {
        background: linear-gradient(135deg, #1E3A5F 0%, #2C3E50 100%);
        border-radius: 16px;
        padding: 1.5rem;
        color: white;
        margin: 1rem 0;
    }

    .insight-card h3 {
        color: #E67E22;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .insight-card p {
        font-size: 1.1rem;
        line-height: 1.6;
        margin: 0;
    }

    /* Savings Highlight */
    .savings-highlight {
        background: linear-gradient(135deg, #27AE60 0%, #2ECC71 100%);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        color: white;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(39, 174, 96, 0.3);
    }

    .savings-highlight .amount {
        font-size: 3rem;
        font-weight: 700;
        display: block;
    }

    .savings-highlight .label {
        font-size: 1rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Button Styling */
    .stButton > button {
        background: linear-gradient(135deg, #1E3A5F 0%, #3498DB 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(30, 58, 95, 0.4);
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background: #ECF0F1;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1E3A5F 0%, #3498DB 100%);
        color: white;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A5F 0%, #2C3E50 100%);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
    }

    section[data-testid="stSidebar"] label {
        color: white !important;
        font-weight: 500 !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stRadio label,
    section[data-testid="stSidebar"] .stMultiSelect label,
    section[data-testid="stSidebar"] .stTextInput label,
    section[data-testid="stSidebar"] .stNumberInput label {
        color: white !important;
        font-weight: 500 !important;
    }

    /* Selectbox/Input Felder in Sidebar - besserer Kontrast */
    section[data-testid="stSidebar"] .stSelectbox > div > div,
    section[data-testid="stSidebar"] .stMultiSelect > div > div,
    section[data-testid="stSidebar"] .stTextInput > div > div > input,
    section[data-testid="stSidebar"] .stNumberInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1E3A5F !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
    }

    section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.95) !important;
        color: #1E3A5F !important;
    }

    section[data-testid="stSidebar"] .stRadio > div {
        color: white !important;
    }

    section[data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
        color: white !important;
    }

    /* Info/Success/Warning Boxes in Sidebar */
    section[data-testid="stSidebar"] .stAlert {
        background-color: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
    }

    /* File Uploader */
    .stFileUploader {
        border: 2px dashed #3498DB;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        background: rgba(52, 152, 219, 0.05);
    }

    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1E3A5F 0%, #E67E22 100%);
    }

    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1E3A5F;
    }

    /* Hide Streamlit branding for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .animate-in {
        animation: fadeInUp 0.5s ease-out forwards;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# PERFORMANCE-OPTIMIERUNG: CACHING UND DOWNSAMPLING
# ============================================================================
def detect_csv_separator(file_content: bytes, sample_size: int = 8192) -> str:
    """
    Erkennt den CSV-Separator schnell anhand einer kleinen Stichprobe.
    Prüft Semikolon, Komma und Tab.
    """
    # Nur ersten Teil der Datei lesen für schnelle Erkennung
    sample = file_content[:sample_size].decode('utf-8', errors='ignore')
    lines = sample.split('\n')[:10]  # Erste 10 Zeilen

    # Zähle Vorkommen der häufigsten Separatoren
    separators = {';': 0, ',': 0, '\t': 0}

    for line in lines:
        for sep in separators:
            separators[sep] += line.count(sep)

    # Wähle den häufigsten Separator (Semikolon bevorzugt bei Gleichstand - deutsche CSVs)
    if separators[';'] >= separators[','] and separators[';'] > 0:
        return ';'
    elif separators[','] > 0:
        return ','
    elif separators['\t'] > 0:
        return '\t'
    else:
        return ';'  # Default für deutsche CSVs


def detect_decimal_separator(file_content: bytes, csv_sep: str, sample_size: int = 8192) -> str:
    """
    Erkennt den Dezimaltrenner (Komma oder Punkt).
    """
    sample = file_content[:sample_size].decode('utf-8', errors='ignore')

    # Wenn CSV-Separator Semikolon ist, ist Dezimaltrenner wahrscheinlich Komma
    if csv_sep == ';':
        # Prüfe ob Zahlen mit Komma vorkommen (z.B. "123,45")
        import re
        if re.search(r'\d+,\d+', sample):
            return ','

    return '.'


@st.cache_data(show_spinner=False)
def load_csv_cached(file_content: bytes, file_name: str) -> pd.DataFrame:
    """
    Lädt CSV mit Caching und schneller Separator-Erkennung.
    Optimiert für große Dateien (10MB+).
    """
    # Schnelle Separator-Erkennung statt sep=None
    separator = detect_csv_separator(file_content)
    decimal = detect_decimal_separator(file_content, separator)

    # Schnelles C-Engine verwenden
    try:
        df = pd.read_csv(
            io.BytesIO(file_content),
            sep=separator,
            decimal=decimal,
            engine='c',  # Schnelles C-Engine statt Python
            low_memory=True,  # Weniger RAM-Verbrauch
            on_bad_lines='warn'  # Fehlerhafte Zeilen überspringen
        )
    except Exception:
        # Fallback: Python-Engine für komplexe Fälle
        df = pd.read_csv(
            io.BytesIO(file_content),
            sep=None,
            engine='python',
            on_bad_lines='warn'
        )

    return df


def downsample_for_chart(df: pd.DataFrame, timestamp_col: str,
                         power_col: str, max_points: int = MAX_CHART_POINTS) -> pd.DataFrame:
    """
    Reduziert Datenpunkte für effiziente Chart-Darstellung.
    Verwendet schnelles numpy-basiertes Min/Max-Sampling um Peaks zu erhalten.
    """
    n = len(df)
    if n <= max_points:
        return df

    # Schnelle Methode: numpy-basiertes Sampling
    # Nimm jeden n-ten Punkt plus die globalen Extremwerte
    step = max(1, n // max_points)

    # Gleichmäßig verteilte Indizes
    indices = list(range(0, n, step))

    # Füge globales Maximum und Minimum hinzu (wichtig für Peaks)
    power_values = df[power_col].values
    max_idx = int(np.nanargmax(power_values))
    min_idx = int(np.nanargmin(power_values))

    if max_idx not in indices:
        indices.append(max_idx)
    if min_idx not in indices:
        indices.append(min_idx)

    # Sortiere und entferne Duplikate
    indices = sorted(set(indices))

    # Begrenze auf max_points
    if len(indices) > max_points:
        step = len(indices) // max_points
        indices = indices[::step]
        # Stelle sicher, dass Extremwerte dabei sind
        if max_idx not in indices:
            indices.append(max_idx)
        if min_idx not in indices:
            indices.append(min_idx)
        indices = sorted(set(indices))

    return df.iloc[indices].copy()


def downsample_arrays_for_chart(timestamps: np.ndarray, values: np.ndarray,
                                max_points: int = MAX_CHART_POINTS) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reduziert Arrays für Chart-Darstellung.
    Behält wichtige Peaks bei.
    """
    n = len(values)
    if n <= max_points:
        return timestamps, values

    step = max(1, n // max_points)
    indices = list(range(0, n, step))

    # Globale Extremwerte hinzufügen
    valid_values = np.nan_to_num(values, nan=-np.inf)
    max_idx = int(np.argmax(valid_values))
    valid_values_min = np.nan_to_num(values, nan=np.inf)
    min_idx = int(np.argmin(valid_values_min))

    if max_idx not in indices:
        indices.append(max_idx)
    if min_idx not in indices:
        indices.append(min_idx)

    indices = sorted(set(indices))

    if len(indices) > max_points:
        step = len(indices) // max_points
        indices = indices[::step]
        if max_idx not in indices:
            indices.append(max_idx)
        if min_idx not in indices:
            indices.append(min_idx)
        indices = sorted(set(indices))

    return timestamps[indices], values[indices]


def get_file_hash(file_content: bytes) -> str:
    """Berechnet Hash für Cache-Key."""
    return hashlib.md5(file_content).hexdigest()[:16]


@st.cache_data(show_spinner=False)
def parse_power_column_cached(values_tuple: tuple, col_name: str) -> np.ndarray:
    """
    Parst Leistungswerte gecacht - vermeidet wiederholtes teures Parsing.
    """
    s = pd.Series(values_tuple)

    # Schneller Check ob bereits numerisch
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce').values

    # Sonst RobustNumericParser verwenden
    return RobustNumericParser.parse_series(s).values


@st.cache_data(show_spinner=False)
def compute_kpis_cached(power_values_tuple: tuple, peak_price: float, energy_price: float) -> dict:
    """
    Berechnet alle KPIs gecacht.
    Verwendet numpy für maximale Performance.
    """
    power_values = np.array(power_values_tuple, dtype=float)
    power_values = power_values[~np.isnan(power_values)]  # NaN entfernen

    if len(power_values) == 0:
        return {
            "peak_kw": 0, "avg_kw": 0, "min_kw": 0, "total_kwh": 0,
            "q95": 0, "q90": 0, "q85": 0,
            "savings_bronze": 0, "savings_silver": 0, "savings_gold": 0,
        }

    # Numpy ist deutlich schneller als pandas für diese Operationen
    peak_kw = float(np.nanmax(power_values))
    avg_kw = float(np.nanmean(power_values))
    min_kw = float(np.nanmin(power_values))
    total_kwh = float(np.nansum(power_values) * 0.25)  # 15-min Intervalle

    # Quantile mit numpy
    q95 = float(np.nanpercentile(power_values, 95))
    q90 = float(np.nanpercentile(power_values, 90))
    q85 = float(np.nanpercentile(power_values, 85))

    # Einsparpotenziale
    savings_bronze = (peak_kw - q95) * peak_price
    savings_silver = (peak_kw - q90) * peak_price
    savings_gold = (peak_kw - q85) * peak_price

    return {
        "peak_kw": peak_kw,
        "avg_kw": avg_kw,
        "min_kw": min_kw,
        "total_kwh": total_kwh,
        "q95": q95,
        "q90": q90,
        "q85": q85,
        "savings_bronze": savings_bronze,
        "savings_silver": savings_silver,
        "savings_gold": savings_gold,
    }


# ============================================================================
# DUMMY-DATEN GENERATOR
# ============================================================================
@st.cache_data(show_spinner=False)
def generate_dummy_data(days: int = 30) -> pd.DataFrame:
    """Generiert realistische Dummy-Lastgang-Daten für Demo-Reports."""
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = [start + timedelta(minutes=15 * i) for i in range(days * 96)]

    n = len(timestamps)
    np.random.seed(42)
    power = np.zeros(n)

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        weekday = ts.weekday()
        base = 45.0

        if 6 <= hour < 22:
            day_factor = 1.5 + 0.5 * np.sin((hour - 6) / 16 * np.pi)
        else:
            day_factor = 0.3

        if weekday >= 5:
            day_factor *= 0.4

        if weekday < 5 and 4 <= hour < 8:
            peak_factor = 2.5 + np.random.uniform(0, 1.5)
        elif weekday < 5 and 10 <= hour < 14:
            peak_factor = 1.8 + np.random.uniform(0, 0.8)
        else:
            peak_factor = 1.0

        noise = np.random.normal(0, 5)
        power[i] = base * day_factor * peak_factor + noise

        if np.random.random() < 0.02:
            power[i] += np.random.uniform(30, 60)

    power = np.clip(power, 5, None)

    return pd.DataFrame({
        "Zeitstempel": timestamps,
        "Leistung_kW": power.round(2),
        "Phase_L1_kW": (power * 0.35 + np.random.normal(0, 2, n)).round(2),
        "Phase_L2_kW": (power * 0.33 + np.random.normal(0, 2, n)).round(2),
        "Phase_L3_kW": (power * 0.32 + np.random.normal(0, 2, n)).round(2),
    })


# ============================================================================
# KPI KOMPONENTEN
# ============================================================================
def render_kpi_card(label: str, value: str, delta: Optional[str] = None,
                   delta_positive: bool = True, icon: str = "📊"):
    """Rendert eine animierte KPI-Karte"""
    delta_html = ""
    if delta:
        delta_class = "positive" if delta_positive else "negative"
        delta_html = f'<span class="kpi-delta {delta_class}">{delta}</span>'

    st.markdown(f"""
    <div class="kpi-card animate-in">
        <span style="font-size: 2rem;">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def render_savings_highlight(amount: float, label: str = "Jährliches Einsparpotenzial"):
    """Rendert eine hervorgehobene Einspar-Anzeige"""
    st.markdown(f"""
    <div class="savings-highlight animate-in">
        <span class="amount">{amount:,.0f} €</span>
        <span class="label">{label}</span>
    </div>
    """, unsafe_allow_html=True)


def render_section_header(title: str, icon: str):
    """Rendert einen Section Header"""
    st.markdown(f"""
    <div class="section-header">
        <span class="section-icon">{icon}</span>
        <h2>{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def render_insight_card(title: str, content: str):
    """Rendert eine Insight-Karte"""
    st.markdown(f"""
    <div class="insight-card animate-in">
        <h3>💡 {title}</h3>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# INTERAKTIVE PLOTLY CHARTS
# ============================================================================
def create_interactive_timeseries(df: pd.DataFrame, timestamp_col: str,
                                  power_col: str) -> go.Figure:
    """Erstellt interaktives Zeitreihen-Chart mit Zoom und Hover.
    Nutzt Downsampling für Performance bei großen Datensätzen."""
    if not PLOTLY_AVAILABLE:
        return None

    # Performance: Downsampling für große Datensätze
    n_points = len(df)
    if n_points > DOWNSAMPLING_THRESHOLD:
        df_plot = downsample_for_chart(df, timestamp_col, power_col, MAX_CHART_POINTS)
        is_downsampled = True
    else:
        df_plot = df
        is_downsampled = False

    fig = go.Figure()

    # Hauptlinie (mit gedownsampelten Daten)
    fig.add_trace(go.Scatter(
        x=df_plot[timestamp_col],
        y=df_plot[power_col],
        mode='lines',
        name='Leistung',
        line=dict(color=COLORS["primary"], width=1.5),
        fill='tozeroy',
        fillcolor='rgba(30, 58, 95, 0.1)',
        hovertemplate='<b>%{x}</b><br>Leistung: %{y:.1f} kW<extra></extra>'
    ))

    # Peak-Linie (aus Original-Daten für Genauigkeit)
    peak = df[power_col].max()
    fig.add_hline(y=peak, line_dash="dash", line_color=COLORS["warning"],
                  annotation_text=f"Peak: {peak:.1f} kW",
                  annotation_position="top right")

    # Durchschnittslinie (aus Original-Daten)
    avg = df[power_col].mean()
    fig.add_hline(y=avg, line_dash="dot", line_color=COLORS["accent"],
                  annotation_text=f"Ø: {avg:.1f} kW",
                  annotation_position="bottom right")

    # Titel mit Downsampling-Hinweis
    title_text = "<b>Lastgang-Analyse</b>"
    if is_downsampled:
        title_text += f" <span style='font-size:12px;color:#7F8C8D;'>(optimiert: {len(df_plot):,} von {n_points:,} Punkten)</span>"

    fig.update_layout(
        title=dict(
            text=title_text,
            font=dict(size=18, color=COLORS["dark"])
        ),
        xaxis_title="Zeit",
        yaxis_title="Leistung (kW)",
        hovermode='x unified',
        template='plotly_white',
        height=450,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        ),
        font=dict(family="Inter, sans-serif")
    )

    return fig


def create_peak_distribution_chart(df: pd.DataFrame, power_col: str) -> go.Figure:
    """Erstellt Verteilungs-Chart der Lastspitzen"""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure()

    # Histogram
    fig.add_trace(go.Histogram(
        x=df[power_col],
        nbinsx=50,
        name='Verteilung',
        marker_color=COLORS["primary"],
        opacity=0.7
    ))

    # Quantile markieren
    q95 = df[power_col].quantile(0.95)
    q90 = df[power_col].quantile(0.90)

    fig.add_vline(x=q95, line_dash="dash", line_color=COLORS["secondary"],
                  annotation_text=f"P95: {q95:.1f} kW")
    fig.add_vline(x=q90, line_dash="dash", line_color=COLORS["accent"],
                  annotation_text=f"P90: {q90:.1f} kW")

    fig.update_layout(
        title=dict(
            text="<b>Lastverteilung</b>",
            font=dict(size=18, color=COLORS["dark"])
        ),
        xaxis_title="Leistung (kW)",
        yaxis_title="Häufigkeit",
        template='plotly_white',
        height=350,
        showlegend=False,
        font=dict(family="Inter, sans-serif")
    )

    return fig


def create_heatmap_chart(df: pd.DataFrame, timestamp_col: str,
                         power_col: str) -> go.Figure:
    """Erstellt Heatmap der Last nach Stunde/Wochentag.
    Performance-optimiert: Aggregation reduziert Datenmenge automatisch."""
    if not PLOTLY_AVAILABLE:
        return None

    # Nur benötigte Spalten kopieren (Performance)
    df_temp = df[[timestamp_col, power_col]].copy()
    timestamps = pd.to_datetime(df_temp[timestamp_col])
    df_temp['hour'] = timestamps.dt.hour
    df_temp['weekday'] = timestamps.dt.dayofweek

    # Pivot-Aggregation (reduziert auf max 7x24=168 Werte)
    pivot = df_temp.pivot_table(
        values=power_col,
        index='weekday',
        columns='hour',
        aggfunc='mean'
    )

    weekday_names = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So']

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=list(range(24)),
        y=weekday_names,
        colorscale=[
            [0, '#ECF0F1'],
            [0.5, '#3498DB'],
            [1, '#E74C3C']
        ],
        hovertemplate='%{y}, %{x}:00 Uhr<br>Ø Leistung: %{z:.1f} kW<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="<b>Lastheatmap</b> (Durchschnitt nach Wochentag & Stunde)",
            font=dict(size=18, color=COLORS["dark"])
        ),
        xaxis_title="Stunde",
        yaxis_title="Wochentag",
        template='plotly_white',
        height=350,
        font=dict(family="Inter, sans-serif")
    )

    return fig


def create_duration_curve(df: pd.DataFrame, power_col: str) -> go.Figure:
    """Erstellt Jahresdauerlinie mit Downsampling für Performance."""
    if not PLOTLY_AVAILABLE:
        return None

    sorted_power = df[power_col].sort_values(ascending=False).reset_index(drop=True)
    n_points = len(sorted_power)
    hours = np.arange(n_points) * 0.25  # 15-min Intervalle

    # Downsampling für große Datensätze
    if n_points > MAX_CHART_POINTS:
        # Gleichmäßiges Sampling für Dauerlinie (sie ist monoton, daher kein Min/Max nötig)
        indices = np.linspace(0, n_points - 1, MAX_CHART_POINTS, dtype=int)
        plot_hours = hours[indices]
        plot_power = sorted_power.iloc[indices].values
    else:
        plot_hours = hours
        plot_power = sorted_power.values

    fig = go.Figure()

    # Fläche unter der Kurve
    fig.add_trace(go.Scatter(
        x=plot_hours,
        y=plot_power,
        mode='lines',
        fill='tozeroy',
        fillcolor='rgba(30, 58, 95, 0.2)',
        line=dict(color=COLORS["primary"], width=2),
        hovertemplate='Stunden: %{x:.0f}h<br>Leistung: %{y:.1f} kW<extra></extra>'
    ))

    # Peak-Bereich markieren (Top 5%)
    peak_hours = n_points * 0.05 * 0.25
    fig.add_vrect(x0=0, x1=peak_hours, fillcolor="rgba(231, 76, 60, 0.1)",
                  line_width=0, annotation_text="Peak 5%",
                  annotation_position="top left")

    fig.update_layout(
        title=dict(
            text="<b>Jahresdauerlinie</b>",
            font=dict(size=18, color=COLORS["dark"])
        ),
        xaxis_title="Stunden pro Jahr",
        yaxis_title="Leistung (kW)",
        template='plotly_white',
        height=350,
        font=dict(family="Inter, sans-serif")
    )

    return fig


def create_savings_waterfall(current_peak: float, scenarios: List[dict]) -> go.Figure:
    """Erstellt Waterfall-Chart für Einsparpotenziale"""
    if not PLOTLY_AVAILABLE:
        return None

    fig = go.Figure(go.Waterfall(
        name="Einsparpotenzial",
        orientation="v",
        measure=["absolute"] + ["relative"] * len(scenarios),
        x=["Aktuell"] + [s["name"] for s in scenarios],
        y=[current_peak * 100] + [-s["savings"] for s in scenarios],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": COLORS["accent"]}},
        increasing={"marker": {"color": COLORS["warning"]}},
        totals={"marker": {"color": COLORS["primary"]}},
        textposition="outside",
        text=[f"{current_peak * 100:,.0f} €"] + [f"-{s['savings']:,.0f} €" for s in scenarios],
        hovertemplate='%{x}<br>%{text}<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text="<b>Einsparpotenzial nach Szenarien</b>",
            font=dict(size=18, color=COLORS["dark"])
        ),
        yaxis_title="Jährliche Kosten (€)",
        template='plotly_white',
        height=400,
        showlegend=False,
        font=dict(family="Inter, sans-serif")
    )

    return fig


def create_gauge_chart(value: float, max_value: float, title: str,
                       thresholds: List[float] = None) -> go.Figure:
    """Erstellt Gauge-Chart für KPIs"""
    if not PLOTLY_AVAILABLE:
        return None

    if thresholds is None:
        thresholds = [max_value * 0.6, max_value * 0.8, max_value]

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 16, 'color': COLORS["dark"]}},
        number={'suffix': " kW", 'font': {'size': 28, 'color': COLORS["primary"]}},
        gauge={
            'axis': {'range': [0, max_value], 'tickwidth': 1, 'tickcolor': COLORS["dark"]},
            'bar': {'color': COLORS["primary"]},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': COLORS["light"],
            'steps': [
                {'range': [0, thresholds[0]], 'color': 'rgba(39, 174, 96, 0.3)'},
                {'range': [thresholds[0], thresholds[1]], 'color': 'rgba(230, 126, 34, 0.3)'},
                {'range': [thresholds[1], thresholds[2]], 'color': 'rgba(231, 76, 60, 0.3)'}
            ],
            'threshold': {
                'line': {'color': COLORS["warning"], 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        font=dict(family="Inter, sans-serif")
    )

    return fig


# ============================================================================
# HAUPTANWENDUNG
# ============================================================================
def main():
    st.set_page_config(
        page_title="PeakGuard - Lastanalyse",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    inject_custom_css()

    # -------------------------
    # Header
    # -------------------------
    st.markdown("""
    <div class="main-header">
        <h1>⚡ PeakGuard</h1>
        <p>Intelligente Lastanalyse & Peak-Shaving Optimierung</p>
    </div>
    """, unsafe_allow_html=True)

    # -------------------------
    # Sidebar
    # -------------------------
    with st.sidebar:
        st.markdown("### 📁 Datenquelle")

        data_source = st.radio(
            "Wählen Sie:",
            ["📊 Demo-Daten verwenden", "📤 Eigene Daten hochladen"],
            label_visibility="collapsed"
        )

        # Initialisiere Variablen
        df = None
        timestamp_col = None
        power_col = None
        power_cols = None
        pf_cols = None
        power_unit = "kW"
        include_reactive = False

        if data_source == "📊 Demo-Daten verwenden":
            demo_days = st.slider("Demo-Zeitraum (Tage)", 7, 90, 30)
            df = generate_dummy_data(demo_days)
            timestamp_col = "Zeitstempel"
            power_col = "Leistung_kW"
            power_cols = ["Phase_L1_kW", "Phase_L2_kW", "Phase_L3_kW"]
            power_unit = "kW"
            st.success(f"✓ {len(df):,} Datenpunkte geladen")

        else:
            uploaded_file = st.file_uploader(
                "CSV-Datei hochladen",
                type=["csv", "txt"],
                help="CSV mit Zeitstempel und Leistungsdaten"
            )

            if uploaded_file is None:
                # Zeige freundliche Anleitung statt Fehlermeldung
                st.markdown("""
                <div style="padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px; margin-top: 1rem;">
                    <p style="color: white; margin: 0; font-size: 0.9rem;">
                        📤 Laden Sie eine CSV-Datei mit Lastgangdaten hoch, um zu beginnen.
                    </p>
                </div>
                """, unsafe_allow_html=True)

            else:
                # CSV laden MIT CACHING für Performance
                try:
                    # Dateiinhalt einmal lesen (Streamlit cacht das)
                    file_bytes = uploaded_file.getvalue()
                    file_size_mb = len(file_bytes) / (1024 * 1024)

                    # Prüfe Dateigröße
                    if file_size_mb > MAX_FILE_SIZE_MB:
                        st.error(f"Datei zu groß: {file_size_mb:.1f} MB (Max: {MAX_FILE_SIZE_MB} MB)")
                        df = None
                    else:
                        # Progress-Anzeige für große Dateien
                        if file_size_mb > 5:
                            with st.spinner(f"⏳ Lade {file_size_mb:.1f} MB..."):
                                df = load_csv_cached(file_bytes, uploaded_file.name)
                        else:
                            df = load_csv_cached(file_bytes, uploaded_file.name)

                        # Prüfe Zeilenanzahl
                        if len(df) > MAX_ROWS:
                            st.warning(f"⚠️ Große Datei ({len(df):,} Zeilen). Charts werden optimiert.")

                        st.success(f"✓ {len(df):,} Zeilen geladen ({file_size_mb:.1f} MB)")
                except Exception as e:
                    st.error(f"Fehler beim Laden: {e}")
                    df = None

        # Spalten-Konfiguration nur anzeigen wenn Daten geladen
        if df is not None:
            st.divider()
            st.markdown("### ⚡ Spaltenzuordnung")

            # Zeitstempel
            timestamp_col = st.selectbox(
                "Zeitstempel-Spalte",
                df.columns,
                index=0
            )

            # Leistungsformat
            power_format = st.radio(
                "Leistungsdaten-Format",
                ["Gesamtleistung (1 Spalte)", "3 Phasen (L1, L2, L3)"],
                horizontal=False
            )

            if power_format == "Gesamtleistung (1 Spalte)":
                power_col = st.selectbox(
                    "Leistungs-Spalte",
                    [c for c in df.columns if c != timestamp_col],
                    index=0 if len(df.columns) > 1 else 0
                )
                power_cols = None
            else:
                st.markdown("**Phasen-Spalten auswählen:**")
                available_cols = [c for c in df.columns if c != timestamp_col]
                col1, col2, col3 = st.columns(3)
                with col1:
                    p1_col = st.selectbox("L1", available_cols, index=0 if len(available_cols) > 0 else 0, key="p1")
                with col2:
                    p2_col = st.selectbox("L2", available_cols, index=1 if len(available_cols) > 1 else 0, key="p2")
                with col3:
                    p3_col = st.selectbox("L3", available_cols, index=2 if len(available_cols) > 2 else 0, key="p3")
                power_cols = [p1_col, p2_col, p3_col]
                power_col = None

            # Einheit
            power_unit = st.radio(
                "Einheit der Leistung",
                ["kW", "W", "Auto (erkennen)"],
                horizontal=True,
                index=0
            )
            if power_unit == "Auto (erkennen)":
                power_unit = "Auto"

            # Cos Phi / Blindleistung
            st.divider()
            include_reactive = st.checkbox("Blindleistung / cos φ einbeziehen", value=False)

            if include_reactive:
                pf_format = st.radio(
                    "cos φ Format",
                    ["Gesamt (1 Spalte)", "3 Phasen"],
                    horizontal=True
                )
                available_pf_cols = [c for c in df.columns if c != timestamp_col]

                if pf_format == "Gesamt (1 Spalte)":
                    pf_col_single = st.selectbox("cos φ Spalte", available_pf_cols, key="pf_single")
                    pf_cols = [pf_col_single]
                else:
                    st.markdown("**cos φ Phasen-Spalten:**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        pf1 = st.selectbox("cos φ L1", available_pf_cols, key="pf1")
                    with col2:
                        pf2 = st.selectbox("cos φ L2", available_pf_cols, key="pf2")
                    with col3:
                        pf3 = st.selectbox("cos φ L3", available_pf_cols, key="pf3")
                    pf_cols = [pf1, pf2, pf3]

        st.divider()

        # Standort-Informationen
        st.markdown("### 🏭 Standort")
        site_name = st.text_input("Standortname", value="Produktionswerk Alpha")
        source_name = st.text_input("Unternehmen", value="Muster GmbH")

        st.divider()

        # Tarif-Einstellungen
        st.markdown("### 💰 Tarife")
        peak_price = st.number_input("Leistungspreis (€/kW/Jahr)",
                                     value=100.0, min_value=0.0, step=10.0)
        energy_price = st.number_input("Arbeitspreis (ct/kWh)",
                                       value=25.0, min_value=0.0, step=1.0)

        tariffs = Tariffs(
            switch_hours=2500.0,
            work_ct_low=energy_price,
            demand_eur_kw_a_low=peak_price * 0.2,
            work_ct_high=energy_price * 0.5,
            demand_eur_kw_a_high=peak_price
        )

    # -------------------------
    # Prüfen ob Daten geladen sind
    # -------------------------
    if df is None:
        # Zeige Willkommens-Bildschirm statt Fehlermeldung
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
            <h2 style="color: #1E3A5F; margin-bottom: 1rem;">Willkommen bei PeakGuard</h2>
            <p style="color: #7F8C8D; font-size: 1.1rem; max-width: 500px; margin: 0 auto;">
                Laden Sie Ihre Lastgangdaten hoch oder verwenden Sie die Demo-Daten,
                um mit der Analyse zu beginnen.
            </p>
            <div style="margin-top: 2rem; padding: 1.5rem; background: #F8F9FA; border-radius: 12px; display: inline-block;">
                <p style="margin: 0; color: #1E3A5F;">
                    👈 Wählen Sie links eine Datenquelle aus
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # -------------------------
    # Hauptbereich mit Tabs
    # -------------------------
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard",
        "🔍 Detailanalyse",
        "💰 Einsparpotenziale",
        "🤖 KI-Analyse",
        "📄 Report & Export"
    ])

    # =========================================================================
    # DATEN VORBEREITEN (vor Tabs, damit überall verfügbar)
    # Performance: Parsing wird gecacht und nur einmal ausgeführt
    # =========================================================================

    # Einheitenfaktor bestimmen - nutze Stichprobe für Auto-Erkennung
    unit_factor = 1.0
    if power_unit == "W":
        unit_factor = 1.0 / 1000.0
    elif power_unit == "Auto":
        # Schnelle Auto-Erkennung mit Stichprobe (nur erste 1000 Werte)
        sample_col = power_col if power_col else (power_cols[0] if power_cols else None)
        if sample_col:
            sample_size = min(1000, len(df))
            sample_vals = df[sample_col].head(sample_size)
            # Schnell prüfen ob numerisch
            if pd.api.types.is_numeric_dtype(sample_vals):
                median_val = float(sample_vals.median())
            else:
                # Parse nur die Stichprobe
                parsed = pd.to_numeric(sample_vals.astype(str).str.replace(',', '.', regex=False), errors='coerce')
                median_val = float(parsed.median()) if len(parsed.dropna()) > 0 else 0
            unit_factor = 1.0 / 1000.0 if median_val > 500 else 1.0

    # Leistungswerte berechnen (GECACHT für Performance)
    if power_col is not None:
        # Einzelne Leistungsspalte - verwende gecachtes Parsing
        raw_values = tuple(df[power_col].values)
        parsed_values = parse_power_column_cached(raw_values, power_col)
        power_values = pd.Series(parsed_values * unit_factor, index=df.index)
        display_power_col = power_col
    elif power_cols is not None and len(power_cols) == 3:
        # 3 Phasen - gecachtes Parsing für jede Phase
        p1_raw = tuple(df[power_cols[0]].values)
        p2_raw = tuple(df[power_cols[1]].values)
        p3_raw = tuple(df[power_cols[2]].values)

        p1 = parse_power_column_cached(p1_raw, power_cols[0]) * unit_factor
        p2 = parse_power_column_cached(p2_raw, power_cols[1]) * unit_factor
        p3 = parse_power_column_cached(p3_raw, power_cols[2]) * unit_factor

        # NaN als 0 behandeln für Summe
        p1 = np.nan_to_num(p1, nan=0.0)
        p2 = np.nan_to_num(p2, nan=0.0)
        p3 = np.nan_to_num(p3, nan=0.0)

        power_values = pd.Series(p1 + p2 + p3, index=df.index)
        df['_total_power_kw'] = power_values
        display_power_col = '_total_power_kw'
    else:
        st.error("Keine gültige Leistungsspalte konfiguriert.")
        st.stop()

    # =========================================================================
    # TAB 1: INTERAKTIVES DASHBOARD
    # =========================================================================
    with tab1:

        # KPI-Berechnung (gecacht für Performance)
        # Konvertiere zu Tuple für Cache-Kompatibilität
        kpis = compute_kpis_cached(
            tuple(power_values.dropna().values),
            peak_price,
            energy_price
        )

        peak_kw = kpis["peak_kw"]
        avg_kw = kpis["avg_kw"]
        min_kw = kpis["min_kw"]
        total_kwh = kpis["total_kwh"]
        q95 = kpis["q95"]
        q90 = kpis["q90"]
        q85 = kpis["q85"]
        savings_bronze = kpis["savings_bronze"]
        savings_silver = kpis["savings_silver"]
        savings_gold = kpis["savings_gold"]

        # KPI-Karten
        render_section_header("Kennzahlen auf einen Blick", "📈")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            render_kpi_card("Spitzenlast", f"{peak_kw:.1f} kW",
                           f"Top 5%: {q95:.1f} kW", False, "⚡")
        with col2:
            render_kpi_card("Durchschnitt", f"{avg_kw:.1f} kW",
                           f"Auslastung: {avg_kw/peak_kw*100:.0f}%", True, "📊")
        with col3:
            render_kpi_card("Verbrauch", f"{total_kwh:,.0f} kWh",
                           f"{total_kwh * energy_price / 100:,.0f} €", True, "🔋")
        with col4:
            render_kpi_card("Einsparpotenzial", f"{savings_silver:,.0f} €/a",
                           "Peak-Shaving Silber", True, "💰")

        st.markdown("<br>", unsafe_allow_html=True)

        # DataFrame für Charts vorbereiten
        df_chart = df.copy()
        df_chart['_power_kw'] = power_values

        # Hauptchart
        if PLOTLY_AVAILABLE:
            fig_timeseries = create_interactive_timeseries(df_chart, timestamp_col, '_power_kw')
            if fig_timeseries:
                st.plotly_chart(fig_timeseries, use_container_width=True)
        else:
            st.warning("Für interaktive Charts installieren Sie Plotly: `pip install plotly`")
            st.line_chart(df_chart.set_index(timestamp_col)['_power_kw'])

        # Zwei-Spalten-Layout für weitere Charts
        if PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)

            with col1:
                fig_heatmap = create_heatmap_chart(df_chart, timestamp_col, '_power_kw')
                if fig_heatmap:
                    st.plotly_chart(fig_heatmap, use_container_width=True)

            with col2:
                fig_duration = create_duration_curve(df_chart, '_power_kw')
                if fig_duration:
                    st.plotly_chart(fig_duration, use_container_width=True)

    # =========================================================================
    # TAB 2: DETAILANALYSE
    # =========================================================================
    with tab2:
        render_section_header("Detaillierte Lastanalyse", "🔍")

        if PLOTLY_AVAILABLE:
            col1, col2 = st.columns(2)

            with col1:
                # Gauge für aktuelle Last
                fig_gauge = create_gauge_chart(
                    peak_kw, peak_kw * 1.2, "Aktuelle Spitzenlast",
                    [q95, q90, peak_kw * 1.1]
                )
                if fig_gauge:
                    st.plotly_chart(fig_gauge, use_container_width=True)

            with col2:
                # Verteilung
                fig_dist = create_peak_distribution_chart(df_chart, '_power_kw')
                if fig_dist:
                    st.plotly_chart(fig_dist, use_container_width=True)

        # Insight-Karten
        col1, col2 = st.columns(2)

        with col1:
            peak_hours = power_values[power_values > q95].count() * 0.25
            render_insight_card(
                "Peak-Analyse",
                f"Die Spitzenlast von {peak_kw:.1f} kW wird nur {peak_hours:.0f} Stunden "
                f"pro Jahr überschritten. Eine Reduktion auf {q95:.1f} kW (P95) würde "
                f"{savings_bronze:,.0f} € jährlich einsparen."
            )

        with col2:
            load_factor = avg_kw / peak_kw * 100 if peak_kw > 0 else 0
            render_insight_card(
                "Lastfaktor",
                f"Mit einem Lastfaktor von {load_factor:.0f}% gibt es erhebliches "
                f"Optimierungspotenzial. Branchentypisch sind 60-70%. "
                f"Eine Verbesserung um 10% könnte {savings_silver * 0.3:,.0f} € sparen."
            )

        # Top-10 Peaks Tabelle
        render_section_header("Top 10 Lastspitzen", "🔝")

        df_peaks = df_chart.copy()
        df_peaks['_power_numeric'] = power_values
        df_peaks = df_peaks.nlargest(10, '_power_numeric')

        st.dataframe(
            df_peaks[[timestamp_col, '_power_kw']].rename(columns={
                timestamp_col: "Zeitpunkt",
                '_power_kw': "Leistung (kW)"
            }),
            use_container_width=True,
            hide_index=True
        )

    # =========================================================================
    # TAB 3: EINSPARPOTENZIALE
    # =========================================================================
    with tab3:
        render_section_header("Einsparpotenziale durch Peak-Shaving", "💰")

        # Großes Einspar-Highlight
        render_savings_highlight(savings_silver, "Jährliches Einsparpotenzial (Silber-Szenario)")

        # Szenarien-Vergleich
        scenarios = [
            {"name": "Bronze (P95)", "cap": q95, "savings": savings_bronze, "reduction": (peak_kw - q95) / peak_kw * 100 if peak_kw > 0 else 0},
            {"name": "Silber (P90)", "cap": q90, "savings": savings_silver, "reduction": (peak_kw - q90) / peak_kw * 100 if peak_kw > 0 else 0},
            {"name": "Gold (P85)", "cap": q85, "savings": savings_gold, "reduction": (peak_kw - q85) / peak_kw * 100 if peak_kw > 0 else 0},
        ]

        col1, col2, col3 = st.columns(3)

        for col, scenario in zip([col1, col2, col3], scenarios):
            with col:
                st.markdown(f"""
                <div class="kpi-card" style="text-align: center;">
                    <div class="kpi-label">{scenario['name']}</div>
                    <div class="kpi-value">{scenario['savings']:,.0f} €</div>
                    <div style="color: #7F8C8D; font-size: 0.9rem;">
                        Cap: {scenario['cap']:.0f} kW<br>
                        Reduktion: {scenario['reduction']:.0f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Waterfall-Chart
        if PLOTLY_AVAILABLE:
            fig_waterfall = create_savings_waterfall(peak_kw, scenarios)
            if fig_waterfall:
                st.plotly_chart(fig_waterfall, use_container_width=True)

        # ROI-Rechner
        render_section_header("ROI-Rechner für Batteriespeicher", "🔋")

        col1, col2 = st.columns(2)

        with col1:
            target_scenario = st.selectbox(
                "Ziel-Szenario",
                ["Bronze (P95)", "Silber (P90)", "Gold (P85)"],
                index=1
            )

            if "Bronze" in target_scenario:
                target_cap = q95
                target_savings = savings_bronze
            elif "Silber" in target_scenario:
                target_cap = q90
                target_savings = savings_silver
            else:
                target_cap = q85
                target_savings = savings_gold

            if st.button("💡 ROI berechnen", type="primary", use_container_width=True):
                roi_result = compute_battery_roi(
                    annual_peak_cost=peak_kw * peak_price,
                    current_peak_kw=peak_kw,
                    target_peak_kw=target_cap,
                    peak_price_eur_kw_a=peak_price
                )

                if roi_result.available:
                    with col2:
                        st.markdown(f"""
                        <div class="chart-container">
                            <h3 style="color: {COLORS['primary']};">Investitionsanalyse</h3>
                            <table style="width: 100%; border-collapse: collapse;">
                                <tr>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Batteriegröße</td>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #eee; text-align: right; font-weight: 600;">
                                        {roi_result.battery_size_kwh:.0f} kWh / {roi_result.battery_power_kw:.0f} kW
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Investition</td>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #eee; text-align: right; font-weight: 600;">
                                        {roi_result.investment_cost:,.0f} €
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Jährliche Einsparung</td>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #eee; text-align: right; font-weight: 600; color: {COLORS['accent']};">
                                        {roi_result.annual_savings:,.0f} €
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #eee;">Amortisation</td>
                                    <td style="padding: 0.5rem; border-bottom: 1px solid #eee; text-align: right; font-weight: 600;">
                                        {roi_result.payback_years:.1f} Jahre
                                    </td>
                                </tr>
                                <tr>
                                    <td style="padding: 0.5rem;">NPV (15 Jahre)</td>
                                    <td style="padding: 0.5rem; text-align: right; font-weight: 600; color: {COLORS['accent'] if roi_result.npv_10_years > 0 else COLORS['warning']};">
                                        {roi_result.npv_10_years:,.0f} €
                                    </td>
                                </tr>
                            </table>
                        </div>
                        """, unsafe_allow_html=True)

                        render_insight_card("Empfehlung", roi_result.recommendation)

    # =========================================================================
    # TAB 4: REPORT & EXPORT
    # =========================================================================
    # =========================================================================
    # TAB 4: KI-ANALYSE (Prognose & Anomalie-Erkennung)
    # =========================================================================
    with tab4:
        render_section_header("KI-gestützte Analyse", "🤖")

        st.markdown("""
        <div class="chart-container">
            <p style="color: #7F8C8D;">
                Machine-Learning-basierte Analyse für Lastprognosen,
                Anomalie-Erkennung und Peak-Wahrscheinlichkeiten.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Unterteilt in Spalten für Prognose und Anomalie
        ml_col1, ml_col2 = st.columns(2)

        with ml_col1:
            st.subheader("📈 Lastprognose")

            forecast_horizon = st.slider(
                "Prognosehorizont (Stunden)",
                min_value=6,
                max_value=168,
                value=24,
                step=6,
                help="Wie weit in die Zukunft soll prognostiziert werden?"
            )

            if st.button("🔮 Prognose berechnen", use_container_width=True):
                with st.spinner("KI-Modell berechnet Prognose..."):
                    try:
                        # DataFrame mit Zeitstempel vorbereiten
                        df_ml = df.copy()
                        df_ml['_power_kw'] = power_values.values

                        forecast_result = compute_load_forecast_ml(
                            df_ml,
                            timestamp_col=timestamp_col,
                            power_col='_power_kw',
                            forecast_horizon_hours=forecast_horizon,
                            confidence_level=0.95
                        )

                        if forecast_result.available:
                            st.success(f"✅ Prognose für {forecast_horizon}h berechnet")

                            # KPIs der Prognose
                            fcst_col1, fcst_col2, fcst_col3 = st.columns(3)
                            with fcst_col1:
                                st.metric(
                                    "Prognostizierter Peak",
                                    f"{forecast_result.predicted_peak_kw:.1f} kW",
                                    delta=f"{forecast_result.predicted_peak_kw - peak_kw:.1f} kW vs. historisch"
                                )
                            with fcst_col2:
                                st.metric(
                                    "Peak-Wahrscheinlichkeit",
                                    f"{forecast_result.peak_probability*100:.0f}%",
                                    help="Wahrscheinlichkeit, dass der prognostizierte Peak überschritten wird"
                                )
                            with fcst_col3:
                                trend_emoji = "📈" if forecast_result.trend == "steigend" else ("📉" if forecast_result.trend == "fallend" else "➡️")
                                st.metric("Trend", f"{trend_emoji} {forecast_result.trend.capitalize()}")

                            # Prognose-Chart mit Plotly
                            if PLOTLY_AVAILABLE and len(forecast_result.forecast_values) > 0:
                                # Historische Daten (letzte 48h für Kontext)
                                hist_points = min(48 * 4, len(power_values))  # 15-min Intervalle
                                hist_times = df[timestamp_col].iloc[-hist_points:].values
                                hist_values = power_values.iloc[-hist_points:].values

                                fig_forecast = go.Figure()

                                # Historische Daten
                                fig_forecast.add_trace(go.Scatter(
                                    x=hist_times,
                                    y=hist_values,
                                    mode='lines',
                                    name='Historisch',
                                    line=dict(color=COLORS["primary"], width=2)
                                ))

                                # Prognose
                                fig_forecast.add_trace(go.Scatter(
                                    x=forecast_result.forecast_timestamps,
                                    y=forecast_result.forecast_values,
                                    mode='lines',
                                    name='Prognose',
                                    line=dict(color=COLORS["secondary"], width=2, dash='dash')
                                ))

                                # Konfidenzintervall
                                if len(forecast_result.confidence_lower) > 0:
                                    fig_forecast.add_trace(go.Scatter(
                                        x=list(forecast_result.forecast_timestamps) + list(forecast_result.forecast_timestamps)[::-1],
                                        y=list(forecast_result.confidence_upper) + list(forecast_result.confidence_lower)[::-1],
                                        fill='toself',
                                        fillcolor='rgba(230, 126, 34, 0.2)',
                                        line=dict(color='rgba(255,255,255,0)'),
                                        name='95% Konfidenz',
                                        showlegend=True
                                    ))

                                fig_forecast.update_layout(
                                    title="Lastprognose mit Konfidenzintervall",
                                    xaxis_title="Zeit",
                                    yaxis_title="Leistung (kW)",
                                    hovermode='x unified',
                                    template="plotly_white"
                                )

                                st.plotly_chart(fig_forecast, use_container_width=True)

                            # Modellinfo
                            with st.expander("ℹ️ Modell-Details"):
                                st.write(f"**Modelltyp:** {forecast_result.model_type}")
                                st.write(f"**MAE:** {forecast_result.mae:.2f} kW")
                                st.write(f"**MAPE:** {forecast_result.mape:.1f}%")
                                if forecast_result.seasonality:
                                    st.write("**Saisonalität:**")
                                    for key, val in forecast_result.seasonality.items():
                                        st.write(f"  - {key}: {val:.2f}")

                        else:
                            st.warning(f"⚠️ Prognose nicht möglich: {forecast_result.error_message}")

                    except Exception as e:
                        st.error(f"Fehler bei der Prognose: {e}")

        with ml_col2:
            st.subheader("🔍 Anomalie-Erkennung")

            sensitivity = st.slider(
                "Sensitivität",
                min_value=1.0,
                max_value=4.0,
                value=2.5,
                step=0.5,
                help="Niedrigere Werte = mehr Anomalien werden erkannt"
            )

            detection_method = st.selectbox(
                "Erkennungsmethode",
                ["Ensemble (empfohlen)", "Z-Score", "IQR", "Isolation Forest"],
                help="Ensemble kombiniert mehrere Methoden für robustere Ergebnisse"
            )

            method_map = {
                "Ensemble (empfohlen)": "ensemble",
                "Z-Score": "zscore",
                "IQR": "iqr",
                "Isolation Forest": "isolation_forest"
            }

            if st.button("🔎 Anomalien erkennen", use_container_width=True):
                with st.spinner("Analysiere Datenmuster..."):
                    try:
                        df_ml = df.copy()
                        df_ml['_power_kw'] = power_values.values

                        anomaly_result = detect_anomalies(
                            df_ml,
                            timestamp_col=timestamp_col,
                            power_col='_power_kw',
                            sensitivity=sensitivity,
                            method=method_map[detection_method]
                        )

                        if anomaly_result.available:
                            # Anomalie-Statistiken
                            st.success(f"✅ Analyse abgeschlossen: {anomaly_result.total_anomalies} Anomalien gefunden")

                            anom_col1, anom_col2, anom_col3 = st.columns(3)
                            with anom_col1:
                                st.metric(
                                    "Anomalien gesamt",
                                    f"{anomaly_result.total_anomalies}",
                                    delta=f"{anomaly_result.anomaly_percentage:.2f}% der Daten"
                                )
                            with anom_col2:
                                high_severity = anomaly_result.severity_distribution.get("hoch", 0)
                                st.metric(
                                    "Kritische Anomalien",
                                    f"{high_severity}",
                                    delta="erfordert Prüfung" if high_severity > 0 else None,
                                    delta_color="inverse"
                                )
                            with anom_col3:
                                if len(anomaly_result.anomaly_values) > 0:
                                    max_anomaly = float(np.max(anomaly_result.anomaly_values))
                                    st.metric("Höchste Anomalie", f"{max_anomaly:.1f} kW")

                            # Anomalie-Chart
                            if PLOTLY_AVAILABLE and anomaly_result.total_anomalies > 0:
                                # Downsampling für große Datensätze
                                chart_times, chart_values = downsample_arrays_for_chart(
                                    df[timestamp_col].values,
                                    power_values.values,
                                    MAX_CHART_POINTS
                                )

                                fig_anomaly = go.Figure()

                                # Normale Daten
                                fig_anomaly.add_trace(go.Scatter(
                                    x=chart_times,
                                    y=chart_values,
                                    mode='lines',
                                    name='Lastgang',
                                    line=dict(color=COLORS["primary"], width=1),
                                    opacity=0.7
                                ))

                                # Anomalien hervorheben
                                fig_anomaly.add_trace(go.Scatter(
                                    x=anomaly_result.anomaly_timestamps,
                                    y=anomaly_result.anomaly_values,
                                    mode='markers',
                                    name='Anomalien',
                                    marker=dict(
                                        color=COLORS["warning"],
                                        size=8,
                                        symbol='x'
                                    )
                                ))

                                fig_anomaly.update_layout(
                                    title="Erkannte Anomalien im Lastgang",
                                    xaxis_title="Zeit",
                                    yaxis_title="Leistung (kW)",
                                    hovermode='closest',
                                    template="plotly_white"
                                )

                                st.plotly_chart(fig_anomaly, use_container_width=True)

                            # Empfehlungen
                            if anomaly_result.recommendations:
                                with st.expander("💡 Empfehlungen"):
                                    for rec in anomaly_result.recommendations:
                                        st.write(f"• {rec}")

                            # Anomalie-Details
                            if anomaly_result.total_anomalies > 0:
                                with st.expander("📋 Anomalie-Details"):
                                    # Top 10 Anomalien
                                    n_show = min(10, anomaly_result.total_anomalies)
                                    st.write(f"**Top {n_show} Anomalien nach Schwere:**")

                                    # Sortieren nach Score
                                    if len(anomaly_result.anomaly_scores) > 0:
                                        sorted_idx = np.argsort(anomaly_result.anomaly_scores)[::-1][:n_show]
                                        for i, idx in enumerate(sorted_idx):
                                            ts = anomaly_result.anomaly_timestamps[idx] if idx < len(anomaly_result.anomaly_timestamps) else "N/A"
                                            val = anomaly_result.anomaly_values[idx] if idx < len(anomaly_result.anomaly_values) else 0
                                            score = anomaly_result.anomaly_scores[idx] if idx < len(anomaly_result.anomaly_scores) else 0
                                            atype = anomaly_result.anomaly_types[idx] if idx < len(anomaly_result.anomaly_types) else "unbekannt"
                                            st.write(f"{i+1}. **{ts}**: {val:.1f} kW (Score: {score:.2f}, Typ: {atype})")

                        else:
                            st.info("ℹ️ Keine Anomalien erkannt oder zu wenig Daten.")

                    except Exception as e:
                        st.error(f"Fehler bei der Anomalie-Erkennung: {e}")

        # Peak-Wahrscheinlichkeiten
        st.divider()
        st.subheader("⚡ Peak-Wahrscheinlichkeiten")

        peak_col1, peak_col2 = st.columns([1, 2])

        with peak_col1:
            threshold_kw = st.number_input(
                "Schwellenwert (kW)",
                min_value=float(avg_kw),
                max_value=float(peak_kw * 1.5),
                value=float(q95),
                step=10.0,
                help="Ab welcher Leistung gilt ein Wert als kritischer Peak?"
            )

            forecast_hours_peak = st.slider(
                "Vorhersage-Horizont (h)",
                min_value=1,
                max_value=48,
                value=24,
                help="Für wie viele Stunden soll die Peak-Wahrscheinlichkeit berechnet werden?"
            )

        with peak_col2:
            if st.button("📊 Peak-Wahrscheinlichkeiten berechnen", use_container_width=True):
                with st.spinner("Berechne Peak-Wahrscheinlichkeiten..."):
                    try:
                        df_ml = df.copy()
                        df_ml['_power_kw'] = power_values.values

                        peak_predictions = predict_peak_probability(
                            df_ml,
                            timestamp_col=timestamp_col,
                            power_col='_power_kw',
                            threshold_kw=threshold_kw,
                            forecast_hours=forecast_hours_peak
                        )

                        if peak_predictions:
                            st.success(f"✅ {len(peak_predictions)} Zeitfenster analysiert")

                            # Visualisierung der Wahrscheinlichkeiten
                            if PLOTLY_AVAILABLE:
                                hours = [p.timestamp.hour for p in peak_predictions]
                                probs = [p.probability * 100 for p in peak_predictions]
                                expected = [p.predicted_power_kw for p in peak_predictions]

                                fig_peak_prob = make_subplots(specs=[[{"secondary_y": True}]])

                                fig_peak_prob.add_trace(
                                    go.Bar(
                                        x=hours,
                                        y=probs,
                                        name="Überschreitungs-Wahrscheinlichkeit (%)",
                                        marker_color=[
                                            COLORS["warning"] if p > 50 else (COLORS["secondary"] if p > 25 else COLORS["accent"])
                                            for p in probs
                                        ]
                                    ),
                                    secondary_y=False
                                )

                                fig_peak_prob.add_trace(
                                    go.Scatter(
                                        x=hours,
                                        y=expected,
                                        mode='lines+markers',
                                        name="Erwarteter Peak (kW)",
                                        line=dict(color=COLORS["primary"], width=2)
                                    ),
                                    secondary_y=True
                                )

                                # Schwellenwert-Linie
                                fig_peak_prob.add_hline(
                                    y=threshold_kw,
                                    line_dash="dash",
                                    line_color=COLORS["warning"],
                                    annotation_text=f"Schwelle: {threshold_kw:.0f} kW",
                                    secondary_y=True
                                )

                                fig_peak_prob.update_layout(
                                    title=f"Peak-Wahrscheinlichkeiten (nächste {forecast_hours_peak}h)",
                                    xaxis_title="Stunde",
                                    hovermode='x unified',
                                    template="plotly_white"
                                )
                                fig_peak_prob.update_yaxes(title_text="Wahrscheinlichkeit (%)", secondary_y=False)
                                fig_peak_prob.update_yaxes(title_text="Erwarteter Peak (kW)", secondary_y=True)

                                st.plotly_chart(fig_peak_prob, use_container_width=True)

                            # Warnungen für hohe Wahrscheinlichkeiten
                            high_risk_hours = [p for p in peak_predictions if p.probability > 0.5]
                            if high_risk_hours:
                                st.warning(f"⚠️ **{len(high_risk_hours)} Stunden mit >50% Peak-Risiko:**")
                                for p in high_risk_hours[:5]:
                                    st.write(f"  • Stunde {p.timestamp.hour}: {p.probability*100:.0f}% Wahrscheinlichkeit, erwartet: {p.predicted_power_kw:.1f} kW")

                        else:
                            st.info("ℹ️ Nicht genügend Daten für Peak-Wahrscheinlichkeiten.")

                    except Exception as e:
                        st.error(f"Fehler bei Peak-Analyse: {e}")

    # =========================================================================
    # TAB 5: REPORT & EXPORT
    # =========================================================================
    with tab5:
        render_section_header("Report generieren & exportieren", "📄")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div class="chart-container">
                <h3>📊 PDF-Report</h3>
                <p style="color: #7F8C8D;">Professioneller Analyse-Report mit allen Kennzahlen, Charts und Handlungsempfehlungen.</p>
            </div>
            """, unsafe_allow_html=True)

            profile_choice = st.selectbox(
                "Report-Profil",
                ["Standard (empfohlen)", "Lite (Kurzfassung)", "Pro (Vollversion)"]
            )

            pkg = st.selectbox(
                "Peak-Shaving Ziel",
                ["Silber (P90) - empfohlen", "Bronze (P95)", "Gold (P85)", "Manuell"]
            )

            manual_cap_kw = None
            manual_value = ""

            if "Bronze" in pkg:
                reduction_goal = "Bronze"
            elif "Silber" in pkg:
                reduction_goal = "Silber"
            elif "Gold" in pkg:
                reduction_goal = "Gold"
            else:
                reduction_goal = "Manuell"
                manual_cap_kw = st.number_input("Manueller Cap (kW)", value=q90, step=5.0)
                manual_value = f"{manual_cap_kw:.1f} kW"

            if st.button("📥 PDF generieren", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(step: str, percent: int):
                    progress_bar.progress(percent / 100)
                    status_text.text(f"⏳ {step}")

                tmpdir = Path(tempfile.mkdtemp())
                out_path = tmpdir / "peakguard_report.pdf"

                # Profil auswählen
                if "Lite" in profile_choice:
                    selected_profile = PROFILE_LITE
                    profile_name = "Lite"
                elif "Pro" in profile_choice:
                    selected_profile = PROFILE_PRO
                    profile_name = "Pro"
                else:
                    selected_profile = PROFILE_STANDARD
                    profile_name = "Standard"

                try:
                    df2 = df.copy()

                    # Daten für PDF vorbereiten
                    if power_col is not None:
                        df2["power_total"] = RobustNumericParser.parse_series(df2[power_col]) * unit_factor
                    elif power_cols is not None:
                        p1 = RobustNumericParser.parse_series(df2[power_cols[0]]) * unit_factor
                        p2 = RobustNumericParser.parse_series(df2[power_cols[1]]) * unit_factor
                        p3 = RobustNumericParser.parse_series(df2[power_cols[2]]) * unit_factor
                        df2["power_1"] = p1
                        df2["power_2"] = p2
                        df2["power_3"] = p3
                        df2["power_total"] = p1.fillna(0) + p2.fillna(0) + p3.fillna(0)

                    build_pdf_report(
                        df=df2,
                        out_path=out_path,
                        timestamp_col=timestamp_col,
                        power_col=power_col if power_col else "power_total",
                        power_cols=power_cols,
                        power_unit="kW",  # Bereits konvertiert
                        pf_cols=pf_cols,
                        source_name=source_name,
                        site_name=site_name,
                        data_quality="Gut",
                        meter_type="Standard",
                        reduction_goal=reduction_goal,
                        manual_value=manual_value,
                        manual_cap_kw=manual_cap_kw,
                        tariffs=tariffs,
                        include_reactive=include_reactive,
                        input_resolution_minutes=15,
                        demand_interval_minutes=15,
                        profile=selected_profile,
                        progress_callback=update_progress,
                    )

                    progress_bar.progress(100)
                    status_text.empty()

                    today = datetime.now().strftime("%Y-%m-%d")
                    download_filename = f"{today}-PeakGuard-Report-{profile_name}.pdf"

                    st.success("✅ Report erfolgreich erstellt!")
                    st.download_button(
                        "📥 PDF herunterladen",
                        data=out_path.read_bytes(),
                        file_name=download_filename,
                        mime="application/pdf",
                        use_container_width=True,
                    )

                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"Fehler: {e}")

        with col2:
            st.markdown("""
            <div class="chart-container">
                <h3>📤 Daten-Export</h3>
                <p style="color: #7F8C8D;">Exportieren Sie Rohdaten und Analysen für weitere Verarbeitung.</p>
            </div>
            """, unsafe_allow_html=True)

            # Excel Export
            if st.button("📊 Excel erstellen", use_container_width=True):
                with st.spinner("Excel wird erstellt..."):
                    try:
                        analysis_results = {
                            "kpis": {
                                "Spitzenlast": f"{peak_kw:.1f} kW",
                                "Durchschnitt": f"{avg_kw:.1f} kW",
                                "Verbrauch": f"{total_kwh:,.0f} kWh",
                                "Einsparpotenzial": f"{savings_silver:,.0f} €/a",
                            },
                            "recommendations": [],
                            "scenarios": scenarios,
                        }

                        config = ExportConfig(
                            company_name=source_name,
                            site_name=site_name,
                            report_date=datetime.now().strftime("%d.%m.%Y")
                        )

                        excel_bytes = export_to_excel_bytes(df, analysis_results, config)

                        st.download_button(
                            "📥 Excel herunterladen",
                            data=excel_bytes,
                            file_name=f"{datetime.now().strftime('%Y-%m-%d')}-PeakGuard.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            use_container_width=True
                        )
                    except ImportError:
                        st.error("openpyxl nicht installiert: `pip install openpyxl`")
                    except Exception as e:
                        st.error(f"Fehler: {e}")

            # CSV Export
            csv_bytes = df.to_csv(sep=";", decimal=",", index=False).encode("utf-8-sig")
            st.download_button(
                "📋 CSV herunterladen",
                data=csv_bytes,
                file_name=f"{datetime.now().strftime('%Y-%m-%d')}-PeakGuard-Rohdaten.csv",
                mime="text/csv",
                use_container_width=True
            )


if __name__ == "__main__":
    main()
