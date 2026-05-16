"""Tests for dissonance sheets on compiled Excel (separate from Density_Metrics)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from compile_metrics import _write_compiled_excel
from dissonance_export import dissonance_columns_present_in_density_sheet


def _density_core_row(note: str, seed: float) -> dict:
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
        "spectral_entropy": 0.35 + 0.02 * seed,
    }


def test_density_metrics_has_no_dissonance_columns(tmp_path: Path) -> None:
    rows = [_density_core_row(f"N{i}", float(i)) for i in range(5)]
    rows[0]["Sethares Dissonance"] = 1.23
    df = pd.DataFrame(rows)
    outp = tmp_path / "d.xlsx"
    _write_compiled_excel(
        outp,
        df,
        {"analysis_version": "t", "dissonance_enabled": True},
        apply_publication_column_filter=True,
        enable_pca_export=False,
    )
    dens = pd.read_excel(outp, sheet_name="Density_Metrics")
    assert dissonance_columns_present_in_density_sheet(dens) == []
    assert not any("dissonance" in str(c).lower() for c in dens.columns)


def test_dissonance_metrics_sheet_when_legacy_columns_present(tmp_path: Path) -> None:
    rows = []
    for i in range(12):
        r = _density_core_row(f"N{i}", float(i))
        r["Sethares Dissonance"] = 0.1 * i
        r["Hutchinson-Knopoff Dissonance"] = 0.05 * i
        r["Vassilakis Dissonance"] = 0.02 * i
        rows.append(r)
    df = pd.DataFrame(rows)
    outp = tmp_path / "diss.xlsx"
    meta = {
        "analysis_version": "t",
        "dissonance_enabled": True,
        "dissonance_compare_models": True,
        "selected_dissonance_model": "sethares",
    }
    out = _write_compiled_excel(
        outp,
        df,
        meta,
        apply_publication_column_filter=True,
        enable_pca_export=False,
    )
    xl = pd.ExcelFile(outp)
    assert "Dissonance_Metrics" in xl.sheet_names
    dm = pd.read_excel(outp, sheet_name="Dissonance_Metrics")
    assert "sethares_dissonance" in dm.columns
    assert "hutchinson_knopoff_dissonance" in dm.columns
    assert "vassilakis_dissonance" in dm.columns
    assert "Dissonance_Model_Comparison" in xl.sheet_names
    assert "Dissonance_Model_Correlations" in xl.sheet_names
    assert out.get("dissonance_export_status") == "exported"


def test_dissonance_not_dropped_when_public_columns_true(tmp_path: Path) -> None:
    """Dissonance lives on Dissonance_Metrics, not on the publication-filtered wide sheet."""
    rows = []
    for i in range(3):
        r = _density_core_row(f"M{i}", float(i))
        r["Sethares Dissonance"] = float(i + 1)
        rows.append(r)
    df = pd.DataFrame(rows)
    outp = tmp_path / "pub.xlsx"
    _write_compiled_excel(
        outp,
        df,
        {"dissonance_enabled": True, "dissonance_compare_models": False},
        apply_publication_column_filter=True,
        enable_pca_export=False,
    )
    dm = pd.read_excel(outp, sheet_name="Dissonance_Metrics")
    assert pd.notna(dm["sethares_dissonance"].iloc[0])
