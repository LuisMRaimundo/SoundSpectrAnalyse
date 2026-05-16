"""Workbook export compliance: density sheet separation, counts, PCA, metadata."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from compile_metrics import _write_compiled_excel, read_excel_metrics
from density import effective_partial_density_from_powers, partial_density_effective_components_bundle


def _density_frame(n: int = 12) -> pd.DataFrame:
    rows = []
    for i in range(n):
        hpc, ihpc, sbpc = 3 + (i % 4), 2 + (i % 3), i % 2
        rows.append(
            {
                "Note": f"T{i}",
                "weight_function": "linear",
                "Harmonic Partials sum": 1.0,
                "Inharmonic Partials sum": 0.2,
                "Sub-bass sum": 0.05,
                "Total sum": 1.25,
                "effective_partial_density": 1.2 + 0.01 * i,
                "harmonic_energy_sum": 1.0,
                "inharmonic_energy_sum": 0.2,
                "subbass_energy_sum": 0.05,
                "total_component_energy": 1.25,
                "harmonic_energy_ratio": 0.8,
                "inharmonic_energy_ratio": 0.16,
                "subbass_energy_ratio": 0.04,
                "harmonic_order_count": hpc,
                "harmonic_peak_count": hpc,
                "inharmonic_peak_count": ihpc,
                "subbass_peak_count": sbpc,
                "total_detected_peak_count": hpc + ihpc + sbpc,
                "harmonic_bin_count": hpc + 100,
                "inharmonic_bin_count": ihpc + 800,
                "subbass_bin_count": 20,
                "n_fft": 2048,
                "hop_length": 256,
                "window": "hann",
                "spectral_entropy": 0.4,
                "sethares_dissonance": 0.05 * i,
                "hutchinson_knopoff_dissonance": 0.04 * i,
                "vassilakis_dissonance": 0.03 * i,
                "selected_dissonance_model": "sethares",
                "selected_dissonance_value": 0.05 * i,
                "f0_estimated": 220.0,
                "f0_source": "test",
                "harmonic_slot_expected_count": 10,
                "harmonic_slot_matched_count": 8,
                "harmonic_slot_missing_count": 2,
                "non_harmonic_candidate_count": 3,
                "outside_harmonic_window_candidate_count": 3,
                "mean_abs_harmonic_deviation_cents": 5.0,
                "median_abs_harmonic_deviation_cents": 4.0,
                "max_abs_harmonic_deviation_cents": 12.0,
                "rms_harmonic_deviation_cents": 6.0,
                "harmonic_validation_status": "ok",
            }
        )
    return pd.DataFrame(rows)


def test_density_metrics_sheet_clean_and_side_sheets(tmp_path: Path) -> None:
    df = _density_frame(12)
    outp = tmp_path / "c.xlsx"
    meta = {
        "analysis_version": "test",
        "dissonance_enabled": True,
        "dissonance_compare_models": True,
        "n_samples": len(df),
    }
    out_meta = _write_compiled_excel(
        outp,
        df,
        meta,
        apply_publication_column_filter=True,
        enable_pca_export=True,
        minimum_samples_for_pca=10,
        pca_include_dissonance=True,
    )
    xl = pd.ExcelFile(outp)
    dens = pd.read_excel(outp, sheet_name="Density_Metrics")
    forbidden = (
        "PC1",
        "PC2",
        "PC3",
        "sethares_dissonance",
        "spectral_masking",
        "n_fft",
        "hop_length",
        "Window",
        "R_norm",
        "P_norm",
        "D_agn",
        "D_harm",
    )
    for col in forbidden:
        assert col not in dens.columns, col
    assert "Dissonance_Metrics" in xl.sheet_names
    assert "Debug_Counts" in xl.sheet_names
    assert "Validation_Metrics" in xl.sheet_names
    assert out_meta.get("spectral_masking_enabled") is False
    assert out_meta.get("pca_export_status") == "exported"
    assert out_meta.get("dissonance_export_status") == "exported"
    assert out_meta.get("debug_counts_export_status") == "exported"
    assert out_meta.get("validation_export_status") == "exported"
    assert out_meta.get("per_note_metadata_export_status") == "exported"
    assert "Per_Note_Processing_Metadata" in xl.sheet_names
    clean = outp.parent / f"{outp.stem}_clean{outp.suffix}"
    assert not clean.exists(), "compiled export must be a single workbook (no *_clean sidecar)"
    assert "Density_Metrics" in xl.sheet_names
    assert "Analysis_Metadata" in xl.sheet_names
    assert "Debug_Counts" in xl.sheet_names
    assert "PCA_Scores" in xl.sheet_names


def test_pca_skipped_small_n(tmp_path: Path) -> None:
    df = _density_frame(4)
    outp = tmp_path / "small.xlsx"
    out_meta = _write_compiled_excel(
        outp,
        df,
        {"analysis_version": "test", "n_samples": len(df)},
        enable_pca_export=True,
        minimum_samples_for_pca=10,
    )
    xl = pd.ExcelFile(outp)
    assert "PCA_Scores" not in xl.sheet_names
    assert out_meta.get("pca_export_note") == "PCA skipped: insufficient number of samples."


def test_effective_density_numeric_properties() -> None:
    assert effective_partial_density_from_powers(np.array([1.0])) == pytest.approx(1.0)
    assert effective_partial_density_from_powers(np.array([1.0, 1.0])) == pytest.approx(2.0)
    p = np.array([1.0, 1.0, 1.0, 1.0])
    assert effective_partial_density_from_powers(p) == pytest.approx(4.0)
    d0 = effective_partial_density_from_powers(np.array([1.0, 0.01, 0.01, 0.01]))
    assert abs(d0 - 1.0) < 0.25
    d1 = effective_partial_density_from_powers(np.array([0.3, 0.7, 0.05]) * 1e6)
    d2 = effective_partial_density_from_powers(np.array([0.3, 0.7, 0.05]))
    assert d1 == pytest.approx(d2)


def test_partial_density_bundle_aggregate_inharmonic() -> None:
    d, diag = partial_density_effective_components_bundle(
        harmonic_amplitudes=np.array([1.0, 0.5]),
        inharmonic_amplitudes=np.array([0.1, 0.2, 0.05]),
        ground_noise_power=0.01,
        inharmonic_mode="aggregate",
    )
    assert d > 0
    assert diag.get("partial_density_inharmonic_mode") == "aggregate"


def test_read_excel_compiled_metrics_sheet(tmp_path: Path) -> None:
    outp = tmp_path / "legacy.xlsx"
    df = pd.DataFrame([{"Note": "C4", "effective_partial_density": 2.5, "harmonic_peak_count": 5}])
    with pd.ExcelWriter(outp, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Compiled Metrics", index=False)
    m = read_excel_metrics(outp)
    assert m.get("effective_partial_density") == pytest.approx(2.5)
    assert m.get("harmonic_peak_count") == pytest.approx(5.0)
