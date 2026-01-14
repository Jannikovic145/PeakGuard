# app.py
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from report_builder import Tariffs, build_pdf_report

st.set_page_config(page_title="PeakGuard Report Generator", layout="wide")
st.title("PeakGuard – Report Generator (MVP)")

uploaded = st.file_uploader("CSV hochladen", type=["csv"])
if uploaded is None:
    st.info("Bitte eine CSV hochladen.")
    st.stop()

# -----------------------------
# CSV Import
# -----------------------------
st.subheader("CSV Import (Trennzeichen/Format)")

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
st.subheader("Metadaten")
m1, m2, m3, m4 = st.columns(4)
site_name = m1.text_input("Standort (optional)", value="")
data_quality = m2.selectbox("Datenqualität", ["OK", "Unvollständig", "Testdaten"], index=0)
meter_type = m3.selectbox("Zählertyp", ["RLM", "SLP", "Submeter"], index=0)
source_name = m4.text_input("Quelle/Dateiname", value=getattr(uploaded, "name", "upload.csv"))

st.divider()

# -----------------------------
# Zeit / Auflösung
# -----------------------------
st.subheader("1) Zeitstempel & Auflösung")

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

st.divider()

# -----------------------------
# Leistung: Einheit + Phasen-Format
# -----------------------------
st.subheader("2) Leistungsspalten")

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
st.subheader("3) Optional: cosϕ (für Blindleistung/BLK)")

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
st.subheader("4) Tarifeingaben")

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
# Paket / Ziel
# -----------------------------
st.subheader("5) Ziel / Paketwahl (Cap basiert auf 15-min Mittelwerten)")

pkg = st.radio(
    "Paket",
    ["Bronze (P95)", "Silber (P90)", "Gold (P85)", "Manuell"],
    horizontal=True,
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
st.subheader("6) PDF generieren")

if st.button("PDF generieren", type="primary"):
    tmpdir = Path(tempfile.mkdtemp())
    out_path = tmpdir / "peakguard_report.pdf"

    try:
        # --- Canonicalize columns so report_builder findet die Daten robust ---
        df2 = df.copy()

        unit = (power_unit or "Auto").lower().strip()
        unit_factor = 1.0
        if unit == "w":
            unit_factor = 1.0 / 1000.0
        elif unit == "kw":
            unit_factor = 1.0

        if power_format.startswith("Gesamtleistung") and power_col is not None:
            x = pd.to_numeric(df2[power_col], errors="coerce")
            if unit == "auto":
                med = float(x.dropna().median()) if x.notna().any() else 0.0
                unit_factor = (1.0 / 1000.0) if med > 200.0 else 1.0
            df2["power_total"] = x * unit_factor
        else:
            if power_cols is None or len(power_cols) != 3:
                raise ValueError("Bitte 3 Phasen-Leistungsspalten wählen.")
            p1 = pd.to_numeric(df2[power_cols[0]], errors="coerce")
            p2 = pd.to_numeric(df2[power_cols[1]], errors="coerce")
            p3 = pd.to_numeric(df2[power_cols[2]], errors="coerce")
            if unit == "auto":
                med = float(pd.concat([p1, p2, p3]).dropna().median()) if (p1.notna().any() or p2.notna().any() or p3.notna().any()) else 0.0
                unit_factor = (1.0 / 1000.0) if med > 200.0 else 1.0
            df2["power_1"] = p1 * unit_factor
            df2["power_2"] = p2 * unit_factor
            df2["power_3"] = p3 * unit_factor

        # cosphi mapping
        if include_reactive and pf_cols is not None:
            if power_format.startswith("Gesamtleistung") and len(pf_cols) == 1:
                df2["cosphi_total"] = pd.to_numeric(df2[pf_cols[0]], errors="coerce")
            elif (not power_format.startswith("Gesamtleistung")) and len(pf_cols) == 3:
                df2["cosphi_1"] = pd.to_numeric(df2[pf_cols[0]], errors="coerce")
                df2["cosphi_2"] = pd.to_numeric(df2[pf_cols[1]], errors="coerce")
                df2["cosphi_3"] = pd.to_numeric(df2[pf_cols[2]], errors="coerce")

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
        )
    except Exception as e:
        st.error(f"Fehler beim Report-Build: {e}")
        st.stop()

    st.success("PDF erstellt.")
    st.download_button(
        "PDF herunterladen",
        data=out_path.read_bytes(),
        file_name="peakguard_report.pdf",
        mime="application/pdf",
    )
