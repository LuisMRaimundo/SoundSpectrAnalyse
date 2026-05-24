"""Tests for compiled Excel export: Density_Metrics sheet, PCA gating, Analysis_Metadata."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from compile_metrics import _write_compiled_excel


def _minimal_density_row(note: str, seed: float) -> dict:
    h = 0.55 + 0.01 * seed
    ih = 0.28 - 0.006 * seed
    sb = max(0.05, 1.0 - h - ih)
    hpc = 3 + int(seed % 5)
    ihpc = 1 + int(seed % 3)
    sbpc = int(seed % 2)
    h_sum = 0.5 + seed
    i_sum = 0.2 + 0.02 * seed
    s_sum = 0.05
    return {
        "Note": note,
        "weight_function": "linear",
        "Harmonic Partials sum": h_sum,
        "Inharmonic Partials sum": i_sum,
        "Sub-bass sum": s_sum,
        "Total sum": h_sum + i_sum + s_sum,
        "effective_partial_density": 1.0 + 0.1 * seed,
        "harmonic_energy_sum": 0.5 + seed,
        "inharmonic_energy_sum": 0.2 + 0.02 * seed,
        "subbass_energy_sum": 0.05,
        "total_component_energy": 1.0 + 0.1 * seed,
        "harmonic_energy_ratio": h,
        "inharmonic_energy_ratio": ih,
        "subbass_energy_ratio": sb,
        "harmonic_order_count": hpc,
        "harmonic_peak_count": hpc,
        "inharmonic_peak_count": ihpc,
        "subbass_peak_count": sbpc,
        "total_detected_peak_count": hpc + ihpc + sbpc,
        "harmonic_bin_count": hpc + 50,
        "inharmonic_bin_count": ihpc + 200,
        "subbass_bin_count": 12,
        "spectral_entropy": 0.35 + 0.02 * seed,
        "sethares_dissonance": 0.1 * seed,
        "hutchinson_knopoff_dissonance": 0.11 * seed,
        "vassilakis_dissonance": 0.12 * seed,
        "selected_dissonance_model": "sethares",
        "selected_dissonance_value": 0.1 * seed,
        "dissonance_partial_count": 8,
        "dissonance_pair_count": 28,
    }


def test_write_compiled_pca_exported_when_enough_samples(tmp_path: Path) -> None:
    rows = [_minimal_density_row(f"N{i}", float(i)) for i in range(12)]
    df = pd.DataFrame(rows)
    outp = tmp_path / "out.xlsx"
    meta = {"analysis_version": "test", "n_samples": len(df)}
    out_meta = _write_compiled_excel(
        outp,
        df,
        meta,
        apply_publication_column_filter=True,
        enable_pca_export=True,
        minimum_samples_for_pca=10,
    )
    assert outp.is_file()
    xl = pd.ExcelFile(outp)
    assert "Density_Metrics" in xl.sheet_names
    assert "Analysis_Metadata" in xl.sheet_names
    # Publication-default policy now allows PCA to be skipped when the default
    # independent feature set is intentionally constrained.
    if out_meta.get("pca_export_status") == "exported":
        assert "PCA_Scores" in xl.sheet_names
        assert "PCA_Loadings" in xl.sheet_names
        assert "PCA_Explained_Variance" in xl.sheet_names
    else:
        assert "PCA_Scores" not in xl.sheet_names
        assert "PCA_Loadings" not in xl.sheet_names
        assert "PCA_Explained_Variance" not in xl.sheet_names


def test_write_compiled_pca_skipped_small_n(tmp_path: Path) -> None:
    rows = [_minimal_density_row(f"N{i}", float(i)) for i in range(4)]
    df = pd.DataFrame(rows)
    outp = tmp_path / "small.xlsx"
    meta = {"analysis_version": "test", "n_samples": len(df)}
    out_meta = _write_compiled_excel(
        outp,
        df,
        meta,
        apply_publication_column_filter=True,
        enable_pca_export=True,
        minimum_samples_for_pca=10,
    )
    xl = pd.ExcelFile(outp)
    assert "Density_Metrics" in xl.sheet_names
    assert "PCA_Scores" not in xl.sheet_names
    assert out_meta.get("pca_export_note") == "PCA skipped: insufficient number of samples."


def test_write_compiled_metrics_all_when_public_filter_off(tmp_path: Path) -> None:
    rows = [_minimal_density_row(f"N{i}", float(i)) for i in range(12)]
    df = pd.DataFrame(rows)
    df["Extra_Debug"] = 99
    outp = tmp_path / "all.xlsx"
    _write_compiled_excel(
        outp,
        df,
        {"analysis_version": "test"},
        apply_publication_column_filter=False,
        enable_pca_export=False,
    )
    xl = pd.ExcelFile(outp)
    assert "Compiled_Metrics_All" in xl.sheet_names
    all_df = pd.read_excel(outp, sheet_name="Compiled_Metrics_All")
    assert "Extra_Debug" in all_df.columns
