# app.py
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from report_builder import (
    Tariffs,
    build_pdf_report,
    PROFILE_LITE,
    PROFILE_STANDARD,
    PROFILE_PRO,
)
import numpy as np
from datetime import datetime, timedelta


# ============================================================================
# DUMMY-DATEN GENERATOR
# ============================================================================
def generate_dummy_data(days: int = 30) -> pd.DataFrame:
    """
    Generiert realistische Dummy-Lastgang-Daten für Demo-Reports
    Simuliert einen typischen Gewerbebetrieb (Bäckerei/Produktion)
    """
    start = datetime(2024, 1, 1, 0, 0)
    timestamps = [start + timedelta(minutes=15*i) for i in range(days * 96)]  # 96 = 24h * 4 (15-min)

    n = len(timestamps)
    power = np.zeros(n)

    for i, ts in enumerate(timestamps):
        hour = ts.hour
        weekday = ts.weekday()

        # Grundlast
        base = 25.0

        # Tagesgang (höher tagsüber)
        if 6 <= hour < 22:
            day_factor = 1.5 + 0.5 * np.sin((hour - 6) / 16 * np.pi)
        else:
            day_factor = 0.3

        # Wochenend-Reduktion
        if weekday >= 5:  # Samstag/Sonntag
            day_factor *= 0.4

        # Produktions-Peaks (Bäckerei: morgens)
        if weekday < 5 and 4 <= hour < 8:
            peak_factor = 2.5 + np.random.uniform(0, 1.5)
        elif weekday < 5 and 10 <= hour < 14:
            peak_factor = 1.8 + np.random.uniform(0, 0.8)
        else:
            peak_factor = 1.0

        # Zufällige Schwankungen
        noise = np.random.normal(0, 5)

        power[i] = base * day_factor * peak_factor + noise

        # Gelegentliche extreme Peaks (Gleichzeitigkeit)
        if np.random.random() < 0.02:  # 2% Chance
            power[i] += np.random.uniform(30, 60)

    # Negative Werte vermeiden
    power = np.maximum(power, 5)

    # DataFrame erstellen
    df = pd.DataFrame({
        'timestamp': timestamps,
        'power_kw': power,
    })

    return df


st.set_page_config(
    page_title="PeakGuard Report Generator",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="collapsed"
)

# === CUSTOM CSS (PeakGuard Design) ===
st.markdown("""
<style>
    /* PeakGuard Farben */
    :root {
        --peakguard-primary: #0f1729;
        --peakguard-accent: #0da2e7;
        --peakguard-success: #28A745;
        --peakguard-gray: #6C757D;
    }

    /* Header styling */
    h1 {
        color: var(--peakguard-primary) !important;
        font-weight: 700 !important;
        padding-bottom: 1rem;
        border-bottom: 3px solid var(--peakguard-accent);
    }

    h2, h3 {
        color: var(--peakguard-primary) !important;
        font-weight: 600 !important;
        margin-top: 2rem !important;
    }

    /* Divider zwischen Sektionen */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #E9ECEF;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed var(--peakguard-accent);
        border-radius: 8px;
        padding: 2rem;
        background: #FAFBFC;
    }

    /* Buttons */
    .stButton > button {
        background: var(--peakguard-accent) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        border: none !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        background: var(--peakguard-primary) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(15, 23, 41, 0.2);
    }

    /* Download Button */
    .stDownloadButton > button {
        background: var(--peakguard-success) !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 6px !important;
        padding: 0.75rem 2rem !important;
    }

    /* Info boxes */
    .stAlert {
        border-radius: 8px !important;
        border-left: 4px solid var(--peakguard-accent) !important;
    }

    /* Selectbox / Input */
    .stSelectbox, .stTextInput {
        border-radius: 6px;
    }

    /* Radio buttons (Profile) */
    .stRadio > label {
        font-weight: 600 !important;
        color: var(--peakguard-primary) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        color: var(--peakguard-primary) !important;
        background: #F8F9FA !important;
        border-radius: 6px !important;
    }
</style>
""", unsafe_allow_html=True)

# === HEADER ===
st.markdown("""
# ⚡ PeakGuard Report Generator
**Version 4.0** | Moderne Lastgang-Analyse mit intelligenten Empfehlungen
""")

# === BEISPIEL-REPORT ANZEIGE (oben rechts) ===
col_main, col_example = st.columns([2, 1])

with col_example:
    st.markdown("### 📄 Beispiel-Report")
    st.markdown("**Dummy-Daten verfügbar:**")

    # Dummy-Report-Buttons
    dummy_profile = st.radio(
        "Profil wählen:",
        ["Lite", "Standard", "Pro"],
        horizontal=True,
        key="dummy_profile",
        help="Wähle ein Profil für den Beispiel-Report"
    )

    if st.button("🎯 Beispiel-Report generieren", use_container_width=True, key="generate_dummy"):
        with st.spinner(f"Generiere {dummy_profile}-Report mit Dummy-Daten..."):
            # Profil-Mapping
            if dummy_profile == "Lite":
                demo_profile = PROFILE_LITE
            elif dummy_profile == "Pro":
                demo_profile = PROFILE_PRO
            else:
                demo_profile = PROFILE_STANDARD

            # Dummy-Daten generieren
            dummy_df = generate_dummy_data(days=60)  # 2 Monate Daten

            # Temporäre Datei für Output
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp_path = Path(tmp.name)

            try:
                # Report generieren
                build_pdf_report(
                    df=dummy_df,
                    out_path=tmp_path,
                    timestamp_col="timestamp",
                    power_col="power_kw",
                    power_unit="kW",
                    source_name="Beispiel-Daten (Demo)",
                    site_name="Demo-Standort Musterfirma GmbH",
                    data_quality="Demo-Daten",
                    meter_type="RLM",
                    reduction_goal="Silber",
                    tariffs=Tariffs(),
                    include_reactive=False,
                    profile=demo_profile,
                )

                # Download-Button
                today = datetime.now().strftime("%Y-%m-%d")
                demo_filename = f"{today}-PeakGuard-Report-{dummy_profile}-DEMO.pdf"

                st.success("✅ Beispiel-Report erfolgreich erstellt!")
                st.download_button(
                    "📥 Beispiel-Report herunterladen",
                    data=tmp_path.read_bytes(),
                    file_name=demo_filename,
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_dummy"
                )

            except Exception as e:
                st.error(f"Fehler bei der Report-Generierung: {e}")
            finally:
                # Cleanup
                if tmp_path.exists():
                    tmp_path.unlink()

        st.stop()  # Stoppt hier, damit nicht beide Flows gleichzeitig laufen

with col_main:
    st.markdown("---")

    uploaded = st.file_uploader(
        "📂 CSV-Datei hochladen",
        type=["csv"],
        help="Lastgang-Daten im CSV-Format (15-min oder 1-min Auflösung)"
    )

if uploaded is None:
    st.info("👆 Bitte eine CSV-Datei hochladen, um zu starten.")
    st.stop()

# -----------------------------
# CSV Import
# -----------------------------
st.markdown("### 1️⃣ CSV Import & Format")

with st.expander("🔧 Import-Einstellungen", expanded=False):
    c1, c2, c3 = st.columns(3)

    sep_label = c1.selectbox(
        "Trennzeichen",
        options=["Auto", "; (Semikolon)", ", (Komma)", "Tab", "| (Pipe)"],
        index=0,
    )
    encoding = c2.selectbox("Encoding", options=["utf-8", "cp1252", "latin1"], index=0)
    num_format = c3.selectbox(
        "Zahlenformat",
        options=[
            "Auto",
            "Deutsch 1.234,56",
            "Englisch 1,234.56",
            "Keins",
        ],
        index=0,
    )

sep_map: dict[str, Optional[str]] = {
    "Auto": None,
    "; (Semikolon)": ";",
    ", (Komma)": ",",
    "Tab": "\t",
    "| (Pipe)": "|",
}
sep: Optional[str] = sep_map[sep_label]

decimal: Optional[str] = None
thousands: Optional[str] = None
if num_format == "Deutsch 1.234,56":
    decimal, thousands = ",", "."
elif num_format == "Englisch 1,234.56":
    decimal, thousands = ".", ","

with st.expander("Rohdaten (erste 5 Zeilen)"):
    raw_text = uploaded.getvalue()[:8000].decode(encoding, errors="replace")
    st.code("\n".join(raw_text.splitlines()[:5]))

try:
    if sep is None:
        df = pd.read_csv(
            io.BytesIO(uploaded.getvalue()),
            sep=None,
            engine="python",
            encoding=encoding,
            decimal=decimal if decimal is not None else ".",
            thousands=thousands,
        )
    else:
        df = pd.read_csv(
            io.BytesIO(uploaded.getvalue()),
            sep=sep,
            engine="c",
            encoding=encoding,
            decimal=decimal if decimal is not None else ".",
            thousands=thousands,
        )
except Exception as e:
    st.error(f"CSV konnte nicht gelesen werden: {e}")
    st.stop()

st.success(f"CSV geladen: {df.shape[0]} Zeilen, {df.shape[1]} Spalten")
st.dataframe(df.head(25), use_container_width=True)
cols = df.columns.tolist()

st.divider()

# -----------------------------
# Metadaten
# -----------------------------
st.markdown("### 📋 Metadaten")
m1, m2, m3, m4 = st.columns(4)
site_name = m1.text_input("Standort (optional)", value="")
data_quality = m2.selectbox("Datenqualität", ["OK", "Unvollständig", "Testdaten"], index=0)
meter_type = m3.selectbox("Zählertyp", ["RLM", "SLP", "Submeter"], index=0)
source_name = m4.text_input("Quelle/Dateiname", value=getattr(uploaded, "name", "upload.csv"))

st.markdown("---")

# -----------------------------
# Zeit / Auflösung
# -----------------------------
st.markdown("### 2️⃣ Zeitstempel & Auflösung")

timestamp_col = st.selectbox(
    "Zeitstempel-Spalte",
    options=cols,
    index=cols.index("date") if "date" in cols else 0,
)

resolution = st.selectbox(
    "Eingangs-Auflösung",
    options=[
        "Auto erkennen",
        "1 Minute (hochaufgelöst)",
        "15 Minuten (RLM-Lastgang)",
    ],
    index=0,
    help="Egal was du einspeist: Die Lastgang-Auswertung (Cap/Cluster/Kosten) läuft immer auf 15-Minuten-Mittelwerten.",
)

resolution_minutes: Optional[int]
if resolution.startswith("1 Minute"):
    resolution_minutes = 1
elif resolution.startswith("15 Minuten"):
    resolution_minutes = 15
else:
    resolution_minutes = None

st.markdown("---")

# -----------------------------
# Leistung: Einheit + Phasen-Format
# -----------------------------
st.markdown("### 3️⃣ Leistungsspalten")

power_unit = st.selectbox(
    "Einheit der Leistungswerte in der CSV",
    options=["Auto", "W", "kW"],
    index=0,
    help="SmartPi liefert häufig W; Stromzähler häufig kW. Auto rät über Größenordnung.",
)

power_format = st.radio(
    "Leistungsdatenformat",
    options=[
        "Gesamtleistung (1 Spalte)",
        "3 Phasen getrennt (3 Spalten → Summe)",
    ],
    horizontal=True,
)

power_col: Optional[str] = None
power_cols: Optional[list[str]] = None

# Canonical mapping targets (für report_builder)
p1_src: Optional[str] = None
p2_src: Optional[str] = None
p3_src: Optional[str] = None

if power_format.startswith("Gesamtleistung"):
    defaults = [c for c in ["Pges", "P_total", "power_total", "power", "P", "leistung"] if c in cols]
    default_idx = cols.index(defaults[0]) if defaults else 0
    power_col = st.selectbox("Gesamtleistung-Spalte", options=cols, index=default_idx)
else:
    cc1, cc2, cc3 = st.columns(3)

    p1_src = cc1.selectbox(
        "Phase L1 / P1",
        options=cols,
        index=cols.index("power_1") if "power_1" in cols else (cols.index("P1") if "P1" in cols else 0),
    )
    p2_src = cc2.selectbox(
        "Phase L2 / P2",
        options=cols,
        index=cols.index("power_2") if "power_2" in cols else (cols.index("P2") if "P2" in cols else 0),
    )
    p3_src = cc3.selectbox(
        "Phase L3 / P3",
        options=cols,
        index=cols.index("power_3") if "power_3" in cols else (cols.index("P3") if "P3" in cols else 0),
    )
    power_cols = [p1_src, p2_src, p3_src]

st.divider()

# -----------------------------
# Optional: cosphi für BLK/Blindleistung
# -----------------------------
st.markdown("### 4️⃣ Optional: cosϕ (Blindleistung)")

pf_cols: Optional[list[str]] = None

c1_src: Optional[str] = None
c2_src: Optional[str] = None
c3_src: Optional[str] = None

if power_format.startswith("Gesamtleistung"):
    pf_single = st.selectbox("cosϕ gesamt (optional)", options=["(keins)"] + cols, index=0)
    pf_cols = None if pf_single == "(keins)" else [pf_single]
else:
    cc1, cc2, cc3 = st.columns(3)
    c1_src = cc1.selectbox("cosϕ L1 / cosϕ1 (optional)", options=["(keins)"] + cols, index=0)
    c2_src = cc2.selectbox("cosϕ L2 / cosϕ2 (optional)", options=["(keins)"] + cols, index=0)
    c3_src = cc3.selectbox("cosϕ L3 / cosϕ3 (optional)", options=["(keins)"] + cols, index=0)

    if c1_src != "(keins)" and c2_src != "(keins)" and c3_src != "(keins)":
        pf_cols = [c1_src, c2_src, c3_src]
    else:
        pf_cols = None

include_reactive = st.checkbox("Blindleistung/BLK-Kennzahlen berechnen (falls möglich)", value=True)

st.divider()

# -----------------------------
# Tarife
# -----------------------------
st.markdown("### 💰 Tarifeingaben")

tc1, tc2, tc3, tc4, tc5 = st.columns(5)
switch_hours = tc1.number_input("Schwellwert (h/a)", min_value=0.0, value=2500.0, step=100.0)
work_ct_low = tc2.number_input("Arbeitspreis < Schwelle (ct/kWh)", min_value=0.0, value=8.27, step=0.1)
demand_low = tc3.number_input("Leistungspreis < Schwelle (€/kW/a)", min_value=0.0, value=19.93, step=1.0)
work_ct_high = tc4.number_input("Arbeitspreis > Schwelle (ct/kWh)", min_value=0.0, value=4.25, step=0.1)
demand_high = tc5.number_input("Leistungspreis > Schwelle (€/kW/a)", min_value=0.0, value=120.43, step=1.0)

tariffs = Tariffs(
    switch_hours=float(switch_hours),
    work_ct_low=float(work_ct_low),
    demand_eur_kw_a_low=float(demand_low),
    work_ct_high=float(work_ct_high),
    demand_eur_kw_a_high=float(demand_high),
)

st.divider()

# -----------------------------
# Report-Profil (v2 NEU)
# -----------------------------
st.markdown("### 5️⃣ Report-Konfiguration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**📊 Report-Profil**")
    profile_choice = st.radio(
        "Welche Variante?",
        ["Lite (2-4 Seiten)", "Standard (6-10 Seiten) ⭐", "Pro (10-16 Seiten)"],
        index=1,  # Standard als Default
        help="Lite: Executive Summary + Top-Peaks\nStandard: + Szenarien + Roadmap\nPro: + Peak-Kontext (12h/3d-Fenster)"
    )

    # Profil-Mapping
    if profile_choice.startswith("Lite"):
        selected_profile = PROFILE_LITE
        profile_name = "Lite"
    elif profile_choice.startswith("Pro"):
        selected_profile = PROFILE_PRO
        profile_name = "Pro"
    else:
        selected_profile = PROFILE_STANDARD
        profile_name = "Standard"

with col2:
    st.markdown("**🎯 Peak-Shaving Ziel**")
    pkg = st.radio(
        "Cap-Paket:",
        ["Bronze (P95)", "Silber (P90)", "Gold (P85)", "Manuell"],
        horizontal=False,
    )

reduction_goal: str
manual_cap_kw: Optional[float] = None
manual_value = ""

if pkg.startswith("Bronze"):
    reduction_goal = "Bronze"
elif pkg.startswith("Silber"):
    reduction_goal = "Silber"
elif pkg.startswith("Gold"):
    reduction_goal = "Gold"
else:
    reduction_goal = "Manuell"
    manual_cap_kw = st.number_input("Manueller Cap (kW)", min_value=0.0, value=50.0, step=1.0)
    manual_value = f"{manual_cap_kw:.1f} kW"

st.divider()

# -----------------------------
# PDF
# -----------------------------
st.markdown("### 🚀 PDF generieren")

if st.button("PDF generieren", type="primary"):
    tmpdir = Path(tempfile.mkdtemp())
    out_path = tmpdir / "peakguard_report.pdf"

    try:
        # --- Canonicalize columns so report_builder findet die Daten robust ---
        # --- Canonicalize columns so report_builder findet die Daten robust ---
        df2 = df.copy()

        import re
        import numpy as np
        import pandas as pd

        def _parse_num_series(s: pd.Series) -> pd.Series:
            """
            Robust gegen:
            - Dezimalkomma (12,34)
            - Tausenderpunkte (1.234,56)
            - Einheiten/Strings ("12,3 kW", "123 W")
            - NBSP/Spaces
            """
            if pd.api.types.is_numeric_dtype(s):
                return pd.to_numeric(s, errors="coerce")

            x = s.astype("string")

            # NBSP & normale Spaces raus
            x = x.str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)

            # Alles außer Zahlen, . , - und + entfernen (Einheiten etc.)
            x = x.str.replace(r"[^0-9\.,\-\+]", "", regex=True)

            # Wenn es wie deutsches Format aussieht: 1.234,56 -> 1234.56
            looks_de = x.str.contains(r"\d{1,3}(\.\d{3})+,\d+", regex=True, na=False).mean() > 0.05
            if looks_de:
                x = x.str.replace(".", "", regex=False)
                x = x.str.replace(",", ".", regex=False)
            else:
                # Sonst: wenn nur Komma als Dezimaltrenner vorkommt -> Komma zu Punkt
                has_comma_decimal = x.str.contains(r"\d+,\d+", regex=True, na=False).mean() > 0.05
                has_dot_decimal = x.str.contains(r"\d+\.\d+", regex=True, na=False).mean() > 0.05
                if has_comma_decimal and not has_dot_decimal:
                    x = x.str.replace(",", ".", regex=False)

            return pd.to_numeric(x, errors="coerce")

        # Einheit -> Faktor (Auto wird später anhand Median geschätzt)
        unit = (power_unit or "Auto").lower().strip()
        unit_factor = 1.0
        if unit == "w":
            unit_factor = 1.0 / 1000.0
        elif unit == "kw":
            unit_factor = 1.0

        if power_format.startswith("Gesamtleistung") and power_col is not None:
            x = _parse_num_series(df2[power_col])

            if unit == "auto":
                med = float(x.dropna().median()) if x.notna().any() else 0.0
                unit_factor = (1.0 / 1000.0) if med > 200.0 else 1.0

            df2["power_total"] = x * unit_factor

        else:
            if power_cols is None or len(power_cols) != 3:
                raise ValueError("Bitte 3 Phasen-Leistungsspalten wählen.")

            p1 = _parse_num_series(df2[power_cols[0]])
            p2 = _parse_num_series(df2[power_cols[1]])
            p3 = _parse_num_series(df2[power_cols[2]])

            if unit == "auto":
                med = float(pd.concat([p1, p2, p3]).dropna().median()) if (p1.notna().any() or p2.notna().any() or p3.notna().any()) else 0.0
                unit_factor = (1.0 / 1000.0) if med > 200.0 else 1.0

            df2["power_1"] = p1 * unit_factor
            df2["power_2"] = p2 * unit_factor
            df2["power_3"] = p3 * unit_factor

        # cosphi mapping (robust parsen + clamp)
        if include_reactive and pf_cols is not None:
            if power_format.startswith("Gesamtleistung") and len(pf_cols) == 1:
                c = _parse_num_series(df2[pf_cols[0]]).clip(lower=0.0, upper=1.0)
                df2["cosphi_total"] = c
            elif (not power_format.startswith("Gesamtleistung")) and len(pf_cols) == 3:
                df2["cosphi_1"] = _parse_num_series(df2[pf_cols[0]]).clip(lower=0.0, upper=1.0)
                df2["cosphi_2"] = _parse_num_series(df2[pf_cols[1]]).clip(lower=0.0, upper=1.0)
                df2["cosphi_3"] = _parse_num_series(df2[pf_cols[2]]).clip(lower=0.0, upper=1.0)


        with st.expander("Debug: Canonical-Spalten vorhanden?"):
            st.write([c for c in ["power_total", "power_1", "power_2", "power_3", "cosphi_total", "cosphi_1", "cosphi_2", "cosphi_3"] if c in df2.columns])
            show_cols = [c for c in ["power_total", "power_1", "power_2", "power_3", "cosphi_total", "cosphi_1", "cosphi_2", "cosphi_3"] if c in df2.columns]
            if show_cols:
                st.dataframe(df2[show_cols].head(10), use_container_width=True)

        build_pdf_report(
            df=df2,
            out_path=out_path,
            timestamp_col=timestamp_col,
            power_col=power_col,
            power_cols=power_cols,
            power_unit=power_unit,
            pf_cols=pf_cols,
            source_name=source_name,
            site_name=site_name,
            data_quality=data_quality,
            meter_type=meter_type,
            reduction_goal=reduction_goal,
            manual_value=manual_value,
            manual_cap_kw=manual_cap_kw,
            tariffs=tariffs,
            include_reactive=include_reactive,
            input_resolution_minutes=resolution_minutes,
            demand_interval_minutes=15,
            profile=selected_profile,  # NEU: Profil übergeben
        )
    except Exception as e:
        st.error(f"Fehler beim Report-Build: {e}")
        st.stop()

    # Dynamischer Dateiname: Datum-PeakGuard-Report-Profil.pdf
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    download_filename = f"{today}-PeakGuard-Report-{profile_name}.pdf"

    st.success("✅ PDF erfolgreich erstellt!")
    st.download_button(
        "📥 PDF herunterladen",
        data=out_path.read_bytes(),
        file_name=download_filename,
        mime="application/pdf",
        use_container_width=True,
    )
