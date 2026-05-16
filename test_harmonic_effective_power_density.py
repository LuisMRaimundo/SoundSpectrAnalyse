"""Tests for harmonic effective power density (HEpd) and wiring."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compile_metrics import (
    DENSITY_METRICS_MAIN_COLUMNS,
    _write_compiled_excel,
    read_excel_metrics,
    read_super_analysis_metrics,
)
from density import compute_harmonic_effective_power_density


def test_hepd_basic_math() -> None:
    r = compute_harmonic_effective_power_density(amplitudes=np.array([1.0, 0.5, 0.25], dtype=float))
    assert r["harmonic_effective_power_density_status"] == "computed"
    assert r["harmonic_effective_power_density"] == pytest.approx(1.3125)
    assert r["harmonic_effective_power_density_component_count"] == 3


def test_hepd_weak_high_partials() -> None:
    r = compute_harmonic_effective_power_density(amplitudes=np.array([1.0, 0.01, 0.01, 0.01], dtype=float))
    assert r["harmonic_effective_power_density_status"] == "computed"
    assert r["harmonic_effective_power_density"] == pytest.approx(1.0003, rel=0, abs=1e-9)


def test_hepd_strong_spectrum_scores_higher_than_weak_tail() -> None:
    a = compute_harmonic_effective_power_density(amplitudes=np.array([1.0, 0.8, 0.7, 0.6], dtype=float))
    b = compute_harmonic_effective_power_density(amplitudes=np.array([1.0, 0.1, 0.05, 0.02], dtype=float))
    assert a["harmonic_effective_power_density_status"] == "computed"
    assert b["harmonic_effective_power_density_status"] == "computed"
    assert float(a["harmonic_effective_power_density"]) > float(b["harmonic_effective_power_density"]) + 0.5


def test_hepd_invalid_amplitudes_filtered_or_skipped() -> None:
    r = compute_harmonic_effective_power_density(amplitudes=np.array([1.0, np.nan, -1.0, 0.0, 0.5], dtype=float))
    assert r["harmonic_effective_power_density_status"] == "computed"
    assert r["harmonic_effective_power_density_component_count"] == 2

    r2 = compute_harmonic_effective_power_density(amplitudes=np.array([np.nan, np.inf], dtype=float))
    assert r2["harmonic_effective_power_density_status"] == "skipped_no_valid_harmonic_rows"


def test_hepd_empty_dataframe_no_amp_column() -> None:
    df = pd.DataFrame({"Frequency (Hz)": [100.0], "Harmonic Number": [1]})
    r = compute_harmonic_effective_power_density(df)
    assert r["harmonic_effective_power_density_status"] == "skipped_no_valid_amplitude_column"


def test_compiled_density_metrics_includes_hepd_columns(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {
                "Note": "C4",
                "weight_function": "linear",
                "Harmonic Partials sum": 1.0,
                "Inharmonic Partials sum": 0.1,
                "Sub-bass sum": 0.0,
                "Total sum": 1.1,
                "effective_partial_density": 1.0,
                "harmonic_energy_sum": 1.0,
                "inharmonic_energy_sum": 0.1,
                "subbass_energy_sum": 0.0,
                "total_component_energy": 1.1,
                "harmonic_energy_ratio": 0.9,
                "inharmonic_energy_ratio": 0.09,
                "subbass_energy_ratio": 0.01,
                "harmonic_order_count": 3,
                "spectral_entropy": 0.5,
                "harmonic_effective_power_density": 1.2,
                "harmonic_effective_power_density_component_count": 3,
                "harmonic_effective_power_density_status": "computed",
                "harmonic_effective_power_density_normalized_by_harmonic_count": 0.4,
                "harmonic_effective_power_mass": 2.5,
                "harmonic_effective_power_mean": 0.5,
                "harmonic_effective_power_rms": 0.7,
                "harmonic_effective_power_component_count": 5,
                "harmonic_effective_power_mass_status": "computed",
            }
        ]
    )
    outp = tmp_path / "compiled_density_metrics.xlsx"
    _write_compiled_excel(
        outp,
        df,
        {"analysis_version": "test"},
        apply_publication_column_filter=False,
        enable_pca_export=False,
    )
    wide = pd.read_excel(outp, sheet_name="Compiled_Metrics_All")
    for c in (
        "harmonic_effective_power_density",
        "harmonic_effective_power_density_component_count",
        "harmonic_effective_power_density_status",
        "harmonic_effective_power_density_normalized_by_harmonic_count",
        "harmonic_effective_power_mass",
        "harmonic_effective_power_mean",
        "harmonic_effective_power_rms",
        "harmonic_effective_power_component_count",
        "harmonic_effective_power_mass_status",
    ):
        assert c in wide.columns


def test_density_metrics_main_columns_include_hepd_clean_set() -> None:
    for c in (
        "harmonic_effective_power_density",
        "harmonic_effective_power_density_component_count",
        "harmonic_effective_power_density_status",
        "harmonic_effective_power_density_normalized_by_harmonic_count",
    ):
        assert c in DENSITY_METRICS_MAIN_COLUMNS


def test_export_metrics_sheet_columns_roundtrip(tmp_path: Path) -> None:
    row = {
        "Note": "C4",
        "harmonic_effective_power_density": 1.3125,
        "harmonic_effective_power_density_component_count": 3,
        "harmonic_effective_power_density_status": "computed",
        "harmonic_effective_power_density_max_amplitude": 1.0,
        "harmonic_effective_power_density_total_power": 1.3125,
        "harmonic_effective_power_density_normalized_by_harmonic_count": 0.4375,
    }
    p = tmp_path / "spectral_analysis.xlsx"
    pd.DataFrame([row]).to_excel(p, sheet_name="Metrics", index=False)
    m = read_excel_metrics(p)
    assert m.get("harmonic_effective_power_density_status") == "computed"
    assert float(m["harmonic_effective_power_density"]) == pytest.approx(1.3125)


def test_super_analysis_json_spectral_metrics_roundtrip(tmp_path: Path) -> None:
    spec = {
        "Note": "G3",
        "harmonic_effective_power_density": 2.0,
        "harmonic_effective_power_density_component_count": 4.0,
        "harmonic_effective_power_density_status": "computed",
        "harmonic_effective_power_density_max_amplitude": 1.0,
        "harmonic_effective_power_density_total_power": 1.5,
        "harmonic_effective_power_density_normalized_by_harmonic_count": 0.5,
    }
    jp = tmp_path / "super_analysis_results.json"
    jp.write_text(json.dumps({"spectral_metrics": spec}), encoding="utf-8")
    m = read_super_analysis_metrics(jp)
    assert m.get("harmonic_effective_power_density_status") == "computed"
    assert float(m["harmonic_effective_power_density"]) == pytest.approx(2.0)


def test_batch_hepd_fields_extractor() -> None:
    root = Path(__file__).resolve().parents[1]
    audio_pkg = root / "audio_analysis"
    ap = str(audio_pkg)
    if ap not in sys.path:
        sys.path.insert(0, ap)
    from batch_audio_analyzer import BatchAudioAnalyzer

    class _A:
        metrics = {
            "harmonic_effective_power_density": 1.1,
            "harmonic_effective_power_density_component_count": 5,
            "harmonic_effective_power_density_status": "computed",
            "harmonic_effective_power_density_max_amplitude": 0.9,
            "harmonic_effective_power_density_total_power": 2.2,
            "harmonic_effective_power_density_normalized_by_harmonic_count": 0.22,
        }

    d = BatchAudioAnalyzer._hepd_fields_from_analyzer_metrics(_A())
    assert d["harmonic_effective_power_density_status"] == "computed"
    assert d["harmonic_effective_power_density_component_count"] == 5
    assert "harmonic_density_model" not in d


def test_gui_weight_function_row_present() -> None:
    pytest.importorskip("PyQt5.QtWidgets")
    import sys

    from PyQt5.QtWidgets import QApplication

    from interface import SpectrumAnalyzer

    app = QApplication.instance() or QApplication(sys.argv if hasattr(sys, "argv") else ["pytest"])
    w = SpectrumAnalyzer()
    assert "Amplitude weighting function:" in w.label_amplitude_weighting_function.text()
    wf_tip = w.combo_weight_function.toolTip()
    assert "Transforms amplitude" in wf_tip
    assert "D3 (Σlog1p A)" in wf_tip or "D3" in wf_tip


_FORBIDDEN = ("gold-standard", "spurious peaks", "more accurate", "absolute density")


def test_new_hepd_related_sources_avoid_forbidden_marketing_phrases() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "proc_audio.py",
        root / "audio_analysis" / "super_audio_analyzer.py",
        root / "audio_analysis" / "batch_audio_analyzer.py",
        root / "interface.py",
        root / "tests" / "test_harmonic_effective_power_density.py",
    ]
    for p in paths:
        if p.name == "test_harmonic_effective_power_density.py":
            continue
        text = p.read_text(encoding="utf-8", errors="replace").lower()
        for phrase in _FORBIDDEN:
            assert phrase not in text, f"{p.name} must not contain {phrase!r}"
