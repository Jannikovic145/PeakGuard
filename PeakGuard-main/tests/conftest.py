# tests/conftest.py
"""
pytest Konfiguration und Fixtures für PeakGuard Tests.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ============================================================================
# FIXTURES: Test-Daten
# ============================================================================
@pytest.fixture
def sample_timestamps() -> list[datetime]:
    """Generiert eine Liste von Zeitstempeln (30 Tage, 15-min Auflösung)"""
    start = datetime(2024, 1, 1, 0, 0)
    return [start + timedelta(minutes=15 * i) for i in range(30 * 96)]


@pytest.fixture
def sample_power_data() -> np.ndarray:
    """Generiert realistische Leistungsdaten in kW"""
    np.random.seed(42)  # Reproduzierbarkeit
    n = 30 * 96  # 30 Tage, 96 Blöcke pro Tag

    # Grundlast + Tagesgang + Rauschen
    base = 25.0
    power = np.zeros(n)

    for i in range(n):
        hour = (i % 96) // 4  # 15-min -> Stunde
        day_factor = 1.5 + 0.5 * np.sin((hour - 6) / 16 * np.pi) if 6 <= hour < 22 else 0.3
        noise = np.random.normal(0, 3)
        power[i] = base * day_factor + noise

        # Gelegentliche Peaks
        if np.random.random() < 0.02:
            power[i] += np.random.uniform(20, 40)

    return np.maximum(power, 5)


@pytest.fixture
def simple_df(sample_timestamps, sample_power_data) -> pd.DataFrame:
    """Einfacher DataFrame mit einer Leistungsspalte"""
    return pd.DataFrame({
        'timestamp': sample_timestamps,
        'power_kw': sample_power_data,
    })


@pytest.fixture
def three_phase_df(sample_timestamps) -> pd.DataFrame:
    """DataFrame mit 3-Phasen-Daten"""
    np.random.seed(42)
    n = len(sample_timestamps)

    return pd.DataFrame({
        'timestamp': sample_timestamps,
        'power_1': np.random.uniform(8, 15, n),
        'power_2': np.random.uniform(7, 14, n),
        'power_3': np.random.uniform(9, 16, n),
        'cosphi_1': np.random.uniform(0.85, 0.98, n),
        'cosphi_2': np.random.uniform(0.85, 0.98, n),
        'cosphi_3': np.random.uniform(0.85, 0.98, n),
    })


@pytest.fixture
def german_format_df() -> pd.DataFrame:
    """DataFrame mit deutschem Zahlenformat"""
    return pd.DataFrame({
        'timestamp': [
            '01.01.2024 00:00',
            '01.01.2024 00:15',
            '01.01.2024 00:30',
            '01.01.2024 00:45',
        ],
        'leistung': ['12,34', '23,45', '34,56', '45,67'],
        'leistung_mit_einheit': ['12,34 kW', '23,45 kW', '34,56 kW', '45,67 kW'],
    })


@pytest.fixture
def df_with_gaps() -> pd.DataFrame:
    """DataFrame mit Datenlücken"""
    timestamps = [
        datetime(2024, 1, 1, 0, 0),
        datetime(2024, 1, 1, 0, 15),
        # Lücke: 00:30 fehlt
        datetime(2024, 1, 1, 0, 45),
        datetime(2024, 1, 1, 1, 0),
        # Lücke: 01:15 bis 02:00 fehlt
        datetime(2024, 1, 1, 2, 15),
    ]
    return pd.DataFrame({
        'timestamp': timestamps,
        'power_kw': [10.0, 12.0, 11.0, 15.0, 13.0],
    })


@pytest.fixture
def edge_case_df() -> pd.DataFrame:
    """DataFrame mit Randfällen"""
    return pd.DataFrame({
        'timestamp': [
            '2024-01-01 00:00',
            '2024-01-01 00:15',
            '2024-01-01 00:30',
            '2024-01-01 00:45',
        ],
        'power_kw': [0.0, np.nan, -5.0, 1000000.0],  # 0, NaN, negativ, sehr groß
    })


# ============================================================================
# FIXTURES: Konfiguration
# ============================================================================
@pytest.fixture
def default_tariffs():
    """Standard-Tarifkonfiguration"""
    from report_builder import Tariffs
    return Tariffs()


@pytest.fixture
def custom_tariffs():
    """Benutzerdefinierte Tarifkonfiguration"""
    from report_builder import Tariffs
    return Tariffs(
        switch_hours=3000.0,
        work_ct_low=10.0,
        demand_eur_kw_a_low=25.0,
        work_ct_high=5.0,
        demand_eur_kw_a_high=150.0,
    )


@pytest.fixture
def temp_output_dir(tmp_path) -> Path:
    """Temporäres Ausgabeverzeichnis"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
