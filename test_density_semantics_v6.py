"""Semantics: measured batch energy ratios vs model weights; Density_Metrics cleanliness."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from compile_metrics import DENSITY_METRICS_ALLOWED_COLUMNS, _write_compiled_excel, validate_compiled_density_workbook


def _row(note: str, i: int) -> dict:
    h, ih, sb = 0.7, 0.2, 0.1
    return {
        "Note": note,
        "weight_function": "linear",
        "Harmonic Partials sum": 1.0 + 0.01 * i,
        "Inharmonic Partials sum": 0.2,
        "Sub-bass sum": 0.1,
        "Total sum": 1.3 + 0.01 * i,
        "effective_partial_density": 2.0 + 0.01 * i,
        "harmonic_energy_sum": 1.0,
        "inharmonic_energy_sum": 0.2,
        "subbass_energy_sum": 0.1,
        "total_component_energy": 1.3,
        "harmonic_energy_ratio": h,
        "inharmonic_energy_ratio": ih,
        "subbass_energy_ratio": sb,
        "harmonic_order_count": 5 + i,
        "spectral_entropy": 0.5,
        "harmonic_bin_count": 400,
        "harmonic_peak_count": 99,
    }


def test_density_metrics_allowlist_excludes_bin_and_peak_columns(tmp_path: Path) -> None:
    df = pd.DataFrame([_row("A4", 0)])
    outp = tmp_path / "clean.xlsx"
    _write_compiled_excel(outp, df, {"analysis_version": "t"}, enable_pca_export=False)
    dm = pd.read_excel(outp, sheet_name="Density_Metrics")
    assert set(dm.columns) <= set(DENSITY_METRICS_ALLOWED_COLUMNS)
    assert "harmonic_bin_count" not in dm.columns
    assert "harmonic_peak_count" not in dm.columns
    assert validate_compiled_density_workbook(outp) == []


def test_validator_rejects_inharmonic_peak_on_density_without_robust_meta(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parent.parent
    script = repo / "scripts" / "validate_density_workbook.py"
    outp = tmp_path / "bad_peak.xlsx"
    rows = [_row(f"N{i}", i) for i in range(4)]
    _write_compiled_excel(outp, pd.DataFrame(rows), {"analysis_version": "t"}, enable_pca_export=False)
    xl = pd.ExcelFile(outp)
    frames = {s: pd.read_excel(outp, sheet_name=s) for s in xl.sheet_names}
    xd = frames["Density_Metrics"].copy()
    xd["inharmonic_peak_count"] = 3
    frames["Density_Metrics"] = xd
    with pd.ExcelWriter(outp, engine="openpyxl") as w:
        for s, d in frames.items():
            d.to_excel(w, sheet_name=s, index=False)
    r = subprocess.run(
        [sys.executable, str(script), str(outp)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode != 0


def test_orchestrator_has_no_external_payload_entrypoint() -> None:
    """The new Stage 1 + Stage 2 orchestrator no longer accepts an external
    energy payload: model weights and component ratios come from
    ``proc_audio`` during Stage 1. The removed
    ``_batch_energy_payload`` helper must not exist."""
    from pipeline_orchestrator_integrated import RobustOrchestrator

    assert not hasattr(RobustOrchestrator, "_batch_energy_payload")
