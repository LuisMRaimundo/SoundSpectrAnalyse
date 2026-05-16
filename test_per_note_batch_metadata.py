"""DEPRECATED tests: per-note batch / model-weight provenance fields on
``Per_Note_Processing_Metadata``. The current Stage 1 + Stage 2 pipeline
emits ``component_*`` provenance only - the ``batch_*`` metadata columns
covered by this module are no longer produced.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "deprecated: removed batch_* provenance columns; new pipeline only "
        "emits component_*_energy_ratio sourced from current_analysis."
    )
)

from compile_metrics import _write_compiled_excel, validate_compiled_density_workbook


def _row(note: str, h: float, i: float, s: float) -> dict:
    t = h + i + s
    bh, bi, bs = h / t, i / t, s / t
    mh, mi = h / (h + i), i / (h + i)
    return {
        "Note": note,
        "weight_function": "linear",
        "Harmonic Partials sum": h,
        "Inharmonic Partials sum": i,
        "Sub-bass sum": s,
        "Total sum": t,
        "effective_partial_density": 1.2,
        "harmonic_energy_sum": h,
        "inharmonic_energy_sum": i,
        "subbass_energy_sum": s,
        "total_component_energy": t,
        "harmonic_energy_ratio": 0.7,
        "inharmonic_energy_ratio": 0.2,
        "subbass_energy_ratio": 0.1,
        "harmonic_order_count": 4,
        "spectral_entropy": 0.4,
        "batch_harmonic_energy_ratio": bh,
        "batch_inharmonic_energy_ratio": bi,
        "batch_subbass_energy_ratio": bs,
        "batch_total_inharmonic_energy_ratio": bi + bs,
        "batch_energy_denominator": "harmonic_plus_inharmonic_plus_subbass",
        "batch_energy_method": "global_energy_sum_H_I_S",
        "batch_ratio_source_explicit": "true",
        "model_harmonic_weight": mh,
        "model_inharmonic_weight": mi,
        "model_weight_denominator": "harmonic_plus_inharmonic",
        "model_weights_source": "batch_empirical_energy_ratios",
        "model_weights_warning": "",
        "model_weights_fallback_reason": "",
        "model_weight_safety_guard_applied": "False",
        "legacy_bounded_harmonic_weight": min(max(mh, 0.05), 0.95),
        "legacy_bounded_inharmonic_weight": 1.0 - min(max(mh, 0.05), 0.95),
        "n_fft": 2048,
        "hop_length": 256,
        "window": "hann",
    }


def test_per_note_batch_provenance_on_metadata_sheet_only(tmp_path: Path) -> None:
    df = pd.DataFrame([_row("A4", 0.7, 0.2, 0.1), _row("B3", 0.8, 0.15, 0.05)])
    outp = tmp_path / "compiled.xlsx"
    _write_compiled_excel(outp, df, {"analysis_version": "test_per_note_meta"}, enable_pca_export=False)
    dm = pd.read_excel(outp, sheet_name="Density_Metrics")
    pn = pd.read_excel(outp, sheet_name="Per_Note_Processing_Metadata")
    for col in (
        "batch_harmonic_energy_ratio",
        "batch_inharmonic_energy_ratio",
        "batch_subbass_energy_ratio",
        "batch_total_inharmonic_energy_ratio",
        "batch_energy_denominator",
        "batch_energy_method",
        "batch_ratio_source_explicit",
        "model_harmonic_weight",
        "model_inharmonic_weight",
        "model_weight_denominator",
        "model_weights_source",
    ):
        assert col in pn.columns, col
        assert col not in dm.columns, f"{col} must not leak to Density_Metrics"


def test_batch_triplet_and_model_weights_numeric_checks(tmp_path: Path) -> None:
    df = pd.DataFrame([_row("C5", 0.65, 0.25, 0.1)])
    outp = tmp_path / "c.xlsx"
    _write_compiled_excel(outp, df, {}, enable_pca_export=False)
    pn = pd.read_excel(outp, sheet_name="Per_Note_Processing_Metadata")
    h = pd.to_numeric(pn["batch_harmonic_energy_ratio"], errors="coerce")
    i_ = pd.to_numeric(pn["batch_inharmonic_energy_ratio"], errors="coerce")
    s_ = pd.to_numeric(pn["batch_subbass_energy_ratio"], errors="coerce")
    assert math.isclose(float((h + i_ + s_).iloc[0]), 1.0, abs_tol=0.02)
    mh = float(pn["model_harmonic_weight"].iloc[0])
    mi = float(pn["model_inharmonic_weight"].iloc[0])
    assert math.isclose(mh + mi, 1.0, abs_tol=0.02)


def test_compiled_workbook_validator_accepts_provenance_columns(tmp_path: Path) -> None:
    df = pd.DataFrame([_row("D4", 0.72, 0.18, 0.1)])
    outp = tmp_path / "v.xlsx"
    _write_compiled_excel(outp, df, {}, enable_pca_export=False)
    errs = validate_compiled_density_workbook(outp)
    assert not errs, errs


def test_per_note_metadata_export_has_no_windows_paths(tmp_path: Path) -> None:
    df = pd.DataFrame([_row("E3", 0.9, 0.05, 0.05)])
    outp = tmp_path / "pub.xlsx"
    _write_compiled_excel(outp, df, {}, enable_pca_export=False)
    pn = pd.read_excel(outp, sheet_name="Per_Note_Processing_Metadata")
    blob = pn.to_csv(index=False)
    assert not re.search(r"[A-Za-z]:\\", blob)
