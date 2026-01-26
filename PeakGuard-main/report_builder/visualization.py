# report_builder/visualization.py
"""
Visualisierungs-Funktionen für PeakGuard.
Enthält alle Matplotlib-Charts für PDF und interaktive Darstellung.
"""
from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle

from .config import PeakGuardColors
from .models import PeakEventsResult
from .utils import get_temp_path


# ============================================================================
# ZEITREIHEN-PLOTS
# ============================================================================
def make_timeseries_plot(df_15: pd.DataFrame, cap_kw: float) -> Path:
    """
    Erstellt Zeitreihen-Plot mit Cap-Linie.

    Args:
        df_15: 15-min aggregierter DataFrame
        cap_kw: Cap-Wert in kW

    Returns:
        Path zur generierten PNG-Datei
    """
    tmp = get_temp_path("timeseries.png")
    idx = cast(pd.DatetimeIndex, df_15.index)
    y = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')

    # Hauptlinie
    ax.plot(idx, y, color=PeakGuardColors.MPL['primary'],
            linewidth=2, label='Leistung (15-min)')

    # Cap-Linie
    ax.axhline(y=cap_kw, color=PeakGuardColors.MPL['danger'],
               linestyle='--', linewidth=2, label=f'Cap ({cap_kw:.1f} kW)', alpha=0.8)

    # Überschreitungen farbig markieren
    over_mask = y > cap_kw
    if over_mask.any():
        ax.fill_between(idx, y, cap_kw, where=over_mask.values,
                        color=PeakGuardColors.MPL['danger'], alpha=0.15)

    ax.set_ylabel('Leistung (kW)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Zeitraum', fontweight='bold', fontsize=11)
    ax.set_title('Lastgang-Verlauf mit Peak-Shaving Cap', fontweight='bold', fontsize=12, pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_duration_curve(df_15: pd.DataFrame, cap_kw: float) -> Path:
    """
    Erstellt Jahresdauerlinie mit Percentil-Markierungen.

    Args:
        df_15: 15-min aggregierter DataFrame
        cap_kw: Cap-Wert in kW

    Returns:
        Path zur generierten PNG-Datei
    """
    tmp = get_temp_path("duration.png")
    s = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce").dropna().sort_values(ascending=False).to_numpy(dtype=float)
    n = int(len(s))

    if n == 0:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.text(0.5, 0.5, "Keine Daten verfügbar", ha="center", va="center", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(tmp, dpi=180, bbox_inches='tight')
        plt.close()
        return tmp

    x = 100.0 * (np.arange(1, n + 1) / float(n))

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')

    # Hauptkurve
    ax.plot(x, s, color=PeakGuardColors.MPL['primary'], linewidth=2.5, label='Leistung')

    # Cap-Linie
    ax.axhline(y=cap_kw, color=PeakGuardColors.MPL['danger'],
               linestyle='--', linewidth=2.5, label=f'Cap ({cap_kw:.1f} kW)', alpha=0.9)

    # Percentil-WERTE berechnen
    p95_kw = float(np.percentile(s, 95))
    p90_kw = float(np.percentile(s, 90))
    p85_kw = float(np.percentile(s, 85))

    # Horizontale Linien für Bronze/Silber/Gold
    ax.axhline(p95_kw, color=PeakGuardColors.MPL['bronze'], linestyle=':', linewidth=2,
               alpha=0.8, label=f'P95 Bronze ({p95_kw:.1f} kW)')
    ax.axhline(p90_kw, color=PeakGuardColors.MPL['silver'], linestyle=':', linewidth=2,
               alpha=0.8, label=f'P90 Silber ({p90_kw:.1f} kW)')
    ax.axhline(p85_kw, color=PeakGuardColors.MPL['gold'], linestyle=':', linewidth=2,
               alpha=0.8, label=f'P85 Gold ({p85_kw:.1f} kW)')

    ax.set_ylabel('Leistung (kW)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Zeitanteil (%)', fontweight='bold', fontsize=11)
    ax.set_title('Jahresdauerlinie (sortiert nach Leistung)', fontweight='bold', fontsize=12, pad=15)
    ax.set_xlim(0, 100)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=8)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_heatmap(df_15: pd.DataFrame) -> Path:
    """
    Erstellt Heatmap Wochentag x Stunde mit Ampel-Farben.

    Args:
        df_15: 15-min aggregierter DataFrame

    Returns:
        Path zur generierten PNG-Datei
    """
    tmp = get_temp_path("heatmap.png")
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

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')

    # AMPEL-FARBSKALA: Grün (niedrig) -> Gelb (mittel) -> Rot (hoch)
    colors_list = [
        '#2ECC71',  # Grün (niedrig)
        '#A8E063',  # Hellgrün
        '#F4D03F',  # Gelb (mittel)
        '#F39C12',  # Orange
        '#E74C3C'   # Rot (hoch)
    ]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('ampel', colors_list, N=n_bins)

    im = ax.imshow(vals, aspect='auto', cmap=cmap, interpolation='nearest')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Ø Leistung (kW)', rotation=270, labelpad=20, fontweight='bold', fontsize=10)

    # Werte in Zellen
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                # Weißer Hintergrund-Kreis für bessere Lesbarkeit
                circle = Circle((j, i), 0.35, color='white', alpha=0.8, zorder=10)
                ax.add_patch(circle)

                ax.text(j, i, f"{vals[i, j]:.0f}",
                        ha="center", va="center",
                        color='black', fontsize=9, fontweight='bold', zorder=11)

    ax.set_yticks(range(7))
    ax.set_yticklabels(["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"], fontsize=10)
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}" for h in range(24)], fontsize=9)
    ax.set_xlabel("Stunde", fontweight='bold', fontsize=11)
    ax.set_ylabel("Wochentag", fontweight='bold', fontsize=11)
    ax.set_title("Lastprofil-Heatmap (Ø Leistung je Wochentag/Stunde)",
                 fontweight='bold', fontsize=12, pad=15)

    # Grid für bessere Orientierung
    ax.set_xticks(np.arange(-.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=1.5)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_monthly_peaks_bar(df_15: pd.DataFrame, cap_kw: float) -> Path:
    """
    Erstellt Balkendiagramm: Peaks pro Monat.

    Args:
        df_15: 15-min aggregierter DataFrame
        cap_kw: Cap-Wert in kW

    Returns:
        Path zur generierten PNG-Datei
    """
    tmp = get_temp_path("monthly_peaks.png")
    idx = cast(pd.DatetimeIndex, df_15.index)
    p = pd.to_numeric(cast(pd.Series, df_15["p_kw"]), errors="coerce")

    # Zähle Überschreitungen pro Monat
    over = (p > cap_kw)
    monthly_data = pd.DataFrame({
        'year_month': idx.to_period('M'),
        'over': over.astype(int),
        'peak_kw': p
    })

    # Aggregiere: Count Überschreitungen + Max Peak pro Monat
    monthly_agg = monthly_data.groupby('year_month').agg({
        'over': 'sum',
        'peak_kw': 'max'
    }).reset_index()

    monthly_agg['month_str'] = monthly_agg['year_month'].astype(str)

    fig, ax = plt.subplots(figsize=(12, 3.5), facecolor='white')

    # Balken: Anzahl Überschreitungen
    ax.bar(
        range(len(monthly_agg)),
        monthly_agg['over'],
        color=PeakGuardColors.MPL['danger'],
        alpha=0.7,
        label='Anzahl 15-min-Blöcke > Cap'
    )

    # Zweite Y-Achse: Max Peak pro Monat
    ax2 = ax.twinx()
    ax2.plot(
        range(len(monthly_agg)),
        monthly_agg['peak_kw'],
        color=PeakGuardColors.MPL['primary'],
        marker='o',
        linewidth=2.5,
        markersize=6,
        label='Max. Peak des Monats'
    )

    # Cap-Linie
    ax2.axhline(cap_kw, color=PeakGuardColors.MPL['warning'],
                linestyle='--', linewidth=2, alpha=0.8, label=f'Cap ({cap_kw:.1f} kW)')

    ax.set_xlabel('Monat', fontweight='bold', fontsize=11)
    ax.set_ylabel('Anzahl Überschreitungen', fontweight='bold', fontsize=10,
                  color=PeakGuardColors.MPL['danger'])
    ax2.set_ylabel('Leistung (kW)', fontweight='bold', fontsize=10,
                   color=PeakGuardColors.MPL['primary'])

    ax.set_xticks(range(len(monthly_agg)))
    ax.set_xticklabels(monthly_agg['month_str'], rotation=45, ha='right')

    ax.tick_params(axis='y', labelcolor=PeakGuardColors.MPL['danger'])
    ax2.tick_params(axis='y', labelcolor=PeakGuardColors.MPL['primary'])

    ax.set_title('Peaks pro Monat: Überschreitungen & Max-Werte', fontweight='bold', fontsize=12, pad=15)

    # Legende kombiniert
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_events_scatter(mod1: PeakEventsResult) -> Path:
    """
    Erstellt Scatter-Plot: Peak-Ereignisse Dauer vs. Verschiebe-Leistung.

    Args:
        mod1: PeakEventsResult mit Events-DataFrame

    Returns:
        Path zur generierten PNG-Datei
    """
    tmp = get_temp_path("events.png")
    ev = mod1.events_df

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')

    if ev.empty:
        ax.text(0.5, 0.5, "Keine Peak-Ereignisse über Cap erkannt",
                ha="center", va="center", fontsize=14, color=PeakGuardColors.MPL['text'])
        ax.axis("off")
    else:
        # Scatter mit Größe basierend auf max_shift_kw
        scatter = ax.scatter(
            ev["duration_min"],
            ev["max_shift_kw"],
            s=ev["max_shift_kw"] * 10,
            c=ev["duration_min"],
            cmap='YlOrRd',
            alpha=0.6,
            edgecolors=PeakGuardColors.MPL['text'],
            linewidth=0.5
        )

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Dauer (min)', rotation=270, labelpad=20, fontweight='bold')

        ax.set_xlabel('Dauer des Peak-Ereignisses (min)', fontweight='bold', fontsize=11)
        ax.set_ylabel('Max. benötigte Verschiebe-Leistung (kW)', fontweight='bold', fontsize=11)
        ax.set_title('Peak-Ereignisse: Dauer vs. Verschiebe-Leistung', fontweight='bold', fontsize=12, pad=15)

        # Referenzlinien
        ax.axvline(15, color=PeakGuardColors.MPL['success'],
                   linestyle='--', linewidth=1, alpha=0.5, label='Kurze Peaks (≤15 min)')
        ax.axvline(60, color=PeakGuardColors.MPL['warning'],
                   linestyle='--', linewidth=1, alpha=0.5, label='Lange Peaks (≥60 min)')
        ax.legend(loc='upper right', framealpha=0.95, fontsize=9)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_peak_context_plot(
    df_15: pd.DataFrame,
    peak_timestamp: pd.Timestamp,
    window_hours: int,
    cap_kw: float
) -> Path:
    """
    Erstellt Peak-Kontext-Plot: Zeigt Umfeld eines Peaks.

    Args:
        df_15: 15-min aggregierter DataFrame
        peak_timestamp: Zeitpunkt des Peaks
        window_hours: Fenstergröße in Stunden
        cap_kw: Cap-Wert in kW

    Returns:
        Path zur generierten PNG-Datei
    """
    tmp = get_temp_path(f"peak_context_{window_hours}h.png")

    # Zeitfenster berechnen
    half_window = pd.Timedelta(hours=window_hours / 2)
    start = peak_timestamp - half_window
    end = peak_timestamp + half_window

    # Daten filtern
    mask = (df_15.index >= start) & (df_15.index <= end)
    df_window = df_15[mask].copy()

    if df_window.empty:
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
        ax.text(0.5, 0.5, "Nicht genügend Daten im Zeitfenster",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(tmp, dpi=180, bbox_inches='tight')
        plt.close()
        return tmp

    idx = cast(pd.DatetimeIndex, df_window.index)
    y = pd.to_numeric(cast(pd.Series, df_window["p_kw"]), errors="coerce")

    fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')

    # Hauptlinie
    ax.plot(idx, y, color=PeakGuardColors.MPL['primary'],
            linewidth=2.5, label='Leistung (15-min)', zorder=2)

    # Cap-Linie
    ax.axhline(y=cap_kw, color=PeakGuardColors.MPL['danger'],
               linestyle='--', linewidth=2, label=f'Cap ({cap_kw:.1f} kW)', alpha=0.8, zorder=1)

    # Peak-Marker
    ax.axvline(x=peak_timestamp, color=PeakGuardColors.MPL['warning'],
               linestyle='-', linewidth=3, alpha=0.7, label='Peak-Zeitpunkt', zorder=3)

    # Überschreitungen füllen
    over_mask = y > cap_kw
    if over_mask.any():
        ax.fill_between(idx, y, cap_kw, where=over_mask.values,
                        color=PeakGuardColors.MPL['danger'], alpha=0.15, zorder=0)

    ax.set_ylabel('Leistung (kW)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Zeit', fontweight='bold', fontsize=11)

    window_label = f"{window_hours}h" if window_hours < 48 else f"{window_hours // 24}d"
    ax.set_title(f'Peak-Kontext ({window_label}-Fenster um {peak_timestamp:%d.%m.%Y %H:%M})',
                 fontweight='bold', fontsize=12, pad=15)

    ax.legend(loc='upper right', framealpha=0.95, fontsize=9)
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp


def make_blk_plot(df_15: pd.DataFrame) -> Path:
    """
    Erstellt Blindleistungs-Plot.

    Args:
        df_15: 15-min aggregierter DataFrame (mit q_kvar und q_limit)

    Returns:
        Path zur generierten PNG-Datei
    """
    tmp = get_temp_path("blk.png")

    q = pd.to_numeric(cast(pd.Series, df_15.get("q_kvar", pd.Series(index=df_15.index, dtype=float))), errors="coerce")
    qlim = pd.to_numeric(cast(pd.Series, df_15.get("q_limit", pd.Series(index=df_15.index, dtype=float))), errors="coerce")

    idx = cast(pd.DatetimeIndex, df_15.index)

    fig, ax = plt.subplots(figsize=(12, 4.5), facecolor='white')

    ax.plot(idx, q, color=PeakGuardColors.MPL['primary'],
            linewidth=2, label='Q (kvar)', alpha=0.8)
    ax.plot(idx, qlim, color=PeakGuardColors.MPL['danger'],
            linestyle='--', linewidth=2, label='Q-Limit (cosϕ=0,9)', alpha=0.8)

    # Überschreitungen markieren
    over_mask = q > qlim
    if over_mask.any():
        ax.fill_between(idx, q, qlim, where=over_mask.values,
                        color=PeakGuardColors.MPL['danger'], alpha=0.15)

    ax.set_ylabel('Blindleistung (kvar)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Zeitraum', fontweight='bold', fontsize=11)
    ax.set_title('Blindleistungs-Verlauf mit cosϕ-Grenzwert', fontweight='bold', fontsize=12, pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(tmp, dpi=180, bbox_inches='tight')
    plt.close()
    return tmp
