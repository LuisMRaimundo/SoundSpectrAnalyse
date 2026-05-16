"""Tests for harmonic effective power mass (absolute sum of partial powers)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compile_metrics import DENSITY_METRICS_MAIN_COLUMNS, read_excel_metrics
from density import compute_harmonic_effective_power_density, compute_harmonic_effective_power_mass


def test_mass_basic_math() -> None:
    df = pd.DataFrame({"Amplitude": [2.0, 1.0, 0.5]})
    r = compute_harmonic_effective_power_mass(df, amplitude_col="Amplitude")
    assert r["harmonic_effective_power_mass_status"] == "computed"
    assert r["harmonic_effective_power_mass"] == pytest.approx(5.25)
    assert r["harmonic_effective_power_mean"] == pytest.approx(1.75)
    assert r["harmonic_effective_power_rms"] == pytest.approx(math.sqrt(1.75))
    assert r["harmonic_effective_power_component_count"] == 3


def test_mass_invalid_values_filtered() -> None:
    df = pd.DataFrame({"Amplitude": [1.0, np.nan, np.inf, -1.0, 0.0, 2.0]})
    r = compute_harmonic_effective_power_mass(df, amplitude_col="Amplitude")
    assert r["harmonic_effective_power_mass_status"] == "computed"
    assert r["harmonic_effective_power_component_count"] == 2
    assert r["harmonic_effective_power_mass"] == pytest.approx(1.0 + 4.0)


def test_mass_empty_dataframe() -> None:
    r = compute_harmonic_effective_power_mass(pd.DataFrame())
    assert r["harmonic_effective_power_mass_status"] == "skipped_empty_harmonic_df"
    assert r["harmonic_effective_power_component_count"] == 0


def test_mass_missing_amplitude_column() -> None:
    df = pd.DataFrame({"Frequency (Hz)": [100.0]})
    r = compute_harmonic_effective_power_mass(df, amplitude_col="Amplitude")
    assert r["harmonic_effective_power_mass_status"] == "skipped_missing_Amplitude"
    assert r["harmonic_effective_power_component_count"] == 0


def test_mass_scales_relative_density_invariant_shape() -> None:
    a = np.array([1.0, 0.5, 0.25], dtype=float)
    b = np.array([10.0, 5.0, 2.5], dtype=float)
    d1 = compute_harmonic_effective_power_density(amplitudes=a)
    d2 = compute_harmonic_effective_power_density(amplitudes=b)
    m1 = compute_harmonic_effective_power_mass(pd.DataFrame({"Amplitude": a}))
    m2 = compute_harmonic_effective_power_mass(pd.DataFrame({"Amplitude": b}))
    assert d1["harmonic_effective_power_density"] == pytest.approx(float(d2["harmonic_effective_power_density"]))
    assert float(m2["harmonic_effective_power_mass"]) == pytest.approx(100.0 * float(m1["harmonic_effective_power_mass"]))


def test_compiled_main_columns_list_includes_mass() -> None:
    for c in (
        "harmonic_effective_power_mass",
        "harmonic_effective_power_mean",
        "harmonic_effective_power_rms",
        "harmonic_effective_power_component_count",
        "harmonic_effective_power_mass_status",
    ):
        assert c in DENSITY_METRICS_MAIN_COLUMNS


def test_batch_hepm_fields_extractor() -> None:
    import sys

    root = Path(__file__).resolve().parents[1]
    audio_pkg = root / "audio_analysis"
    p = str(audio_pkg)
    if p not in sys.path:
        sys.path.insert(0, p)
    from batch_audio_analyzer import BatchAudioAnalyzer

    class _A:
        metrics = {
            "harmonic_effective_power_mass": 5.0,
            "harmonic_effective_power_mean": 1.0,
            "harmonic_effective_power_rms": 1.2,
            "harmonic_effective_power_component_count": 5,
            "harmonic_effective_power_mass_status": "computed",
        }

    d = BatchAudioAnalyzer._hepm_fields_from_analyzer_metrics(_A())
    assert d["harmonic_effective_power_mass_status"] == "computed"
    assert d["harmonic_effective_power_component_count"] == 5


def test_metrics_sheet_roundtrip_mass_columns(tmp_path: Path) -> None:
    row = {
        "Note": "D4",
        "harmonic_effective_power_mass": 5.25,
        "harmonic_effective_power_mean": 1.75,
        "harmonic_effective_power_rms": math.sqrt(1.75),
        "harmonic_effective_power_component_count": 3,
        "harmonic_effective_power_mass_status": "computed",
    }
    p = tmp_path / "spectral_analysis.xlsx"
    pd.DataFrame([row]).to_excel(p, sheet_name="Metrics", index=False)
    m = read_excel_metrics(p)
    assert m.get("harmonic_effective_power_mass_status") == "computed"
    assert float(m["harmonic_effective_power_mass"]) == pytest.approx(5.25)


_FORBIDDEN = ("gold-standard", "spurious peaks", "more accurate", "absolute density")


def test_touched_sources_avoid_forbidden_phrases() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "proc_audio.py",
        root / "audio_analysis" / "super_audio_analyzer.py",
        root / "audio_analysis" / "batch_audio_analyzer.py",
        root / "interface.py",
        root / "docs" / "DENSITY_EXPORT_SCHEMA.md",
        root / "publication_metric_columns.py",
    ]
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace").lower()
        for phrase in _FORBIDDEN:
            assert phrase not in text, f"{p.name} must not contain {phrase!r}"
