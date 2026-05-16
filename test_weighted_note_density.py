"""Audit tests for the Stage 2 weighted note-density metric.

These tests exercise ``extract_density_components_from_per_note_workbook``
on synthetic workbooks plus the publication-policy default selector for
``Density_Metrics``. They cover the cases enumerated by the audit:

A. Synthetic fixture with known H / I / S amplitude sums and
   component weights — verify the weighted-density arithmetic.
B. Amplitude_raw is preferred over Power_raw on every band even when
   both columns are present.
C. Missing component_* weights ⇒ ``density_extraction_status =
   missing_component_weights`` and weighted-density fields are None
   (NaN-equivalent).
D. ``density_log_weighted`` is NOT normalised to [0, 1].
E. ``publication_chart_policy.select_default_publication_metric``
   selects ``density_log_weighted`` for ``Density_Metrics`` when
   present, with ``harmonic_log_amplitude_density`` as fallback.
F. Real-workbook smoke check (skipped when no synthesised harmonic
   fixture has been produced).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import compile_metrics as cm
from publication_chart_policy import (
    DEFAULT_PUBLICATION_METRIC_BY_SHEET,
    select_default_publication_metric,
)


# ---------------------------------------------------------------------------
# Synthetic per-note workbook helper.
# ---------------------------------------------------------------------------
def _write_per_note_workbook(
    path: Path,
    *,
    harmonic_rows: List[Dict[str, Any]] | None,
    inharmonic_rows: List[Dict[str, Any]] | None,
    subbass_rows: List[Dict[str, Any]] | None,
    component_weights: Optional[Tuple[float, float, float]] = (0.8, 0.15, 0.05),
    schema_version: str = "single_pass_raw_export_v2",
    extra_meta_rows: Optional[List[Tuple[str, Any]]] = None,
) -> None:
    """Write a minimal per-note workbook compatible with the audit-canonical
    schema (Parameter/Value Analysis_Metadata + spectrum sheets)."""
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        if harmonic_rows is not None:
            pd.DataFrame(harmonic_rows).to_excel(
                w, sheet_name="Harmonic Spectrum", index=False
            )
        if inharmonic_rows is not None:
            pd.DataFrame(inharmonic_rows).to_excel(
                w, sheet_name="Inharmonic Spectrum", index=False
            )
        if subbass_rows is not None:
            pd.DataFrame(subbass_rows).to_excel(
                w, sheet_name="Sub-bass band", index=False
            )
        meta_rows: List[Tuple[str, Any]] = [
            ("analysis_schema_version", schema_version),
        ]
        if component_weights is not None:
            wH, wI, wS = component_weights
            meta_rows.extend(
                [
                    ("component_harmonic_energy_ratio", float(wH)),
                    ("component_inharmonic_energy_ratio", float(wI)),
                    ("component_subbass_energy_ratio", float(wS)),
                ]
            )
        if extra_meta_rows:
            meta_rows.extend(extra_meta_rows)
        meta_df = pd.DataFrame(
            {"Parameter": [k for k, _ in meta_rows], "Value": [v for _, v in meta_rows]}
        )
        meta_df.to_excel(w, sheet_name="Analysis_Metadata", index=False)


def _harmonic_row(
    n: int, freq: float, amp: float, *, include_for_density: bool = True
) -> Dict[str, Any]:
    return {
        "Harmonic Number": n,
        "expected_frequency_hz": float(n) * freq,
        "extracted_frequency_hz": float(n) * freq,
        "frequency_deviation_hz": 0.0,
        "Frequency (Hz)": float(n) * freq,
        "Magnitude (dB)": 20.0 * math.log10(max(amp, 1e-12)),
        "Amplitude": float(amp),
        "Amplitude_raw": float(amp),
        "Power_raw": float(amp) ** 2,
        "snr_db": 20.0,
        "prominence_db": 5.0,
        "local_peak_valid": True,
        "candidate_status": "strict_validated",
        "include_for_density": bool(include_for_density),
        "Note": "A4",
    }


def _band_row(freq: float, amp: float) -> Dict[str, Any]:
    return {
        "Frequency (Hz)": float(freq),
        "Magnitude (dB)": 20.0 * math.log10(max(amp, 1e-12)),
        "Amplitude": float(amp),
        "Amplitude_raw": float(amp),
        "Power_raw": float(amp) ** 2,
    }


# ---------------------------------------------------------------------------
# A. Synthetic workbook — exact weighted-density arithmetic.
# ---------------------------------------------------------------------------
def test_a_synthetic_workbook_weighted_density_arithmetic(tmp_path: Path) -> None:
    # H amplitudes: 5 partials of amplitude 12 → sum = 60
    h_rows = [_harmonic_row(k, 220.0, 12.0) for k in range(1, 6)]
    # I amplitudes: 2 rows of amplitude 5 → sum = 10
    i_rows = [_band_row(800.0 + 50 * k, 5.0) for k in range(2)]
    # S amplitudes: 1 row of amplitude 2 → sum = 2
    s_rows = [_band_row(45.0, 2.0)]

    p = tmp_path / "A4_spectral_analysis.xlsx"
    _write_per_note_workbook(
        p,
        harmonic_rows=h_rows,
        inharmonic_rows=i_rows,
        subbass_rows=s_rows,
        component_weights=(0.8, 0.15, 0.05),
    )
    info = cm.extract_density_components_from_per_note_workbook(p)

    assert info["density_extraction_status"] == "ok", info
    assert info["harmonic_amplitude_sum"] == pytest.approx(60.0)
    assert info["inharmonic_amplitude_sum"] == pytest.approx(10.0)
    assert info["subbass_amplitude_sum"] == pytest.approx(2.0)
    assert info["w_H"] == pytest.approx(0.8)
    assert info["w_I"] == pytest.approx(0.15)
    assert info["w_S"] == pytest.approx(0.05)

    assert info["weighted_harmonic_component"] == pytest.approx(48.0)
    assert info["weighted_inharmonic_component"] == pytest.approx(1.5)
    assert info["weighted_subbass_component"] == pytest.approx(0.1)
    assert info["density_weighted_sum"] == pytest.approx(49.6)
    expected_log = math.log10(1.0 + 49.6)
    assert info["density_log_weighted"] == pytest.approx(expected_log)
    assert info["density_log_formula"] == "log10(1 + density_weighted_sum)"
    assert info["density_component_basis"] == "amplitude_sum"
    assert info["density_weight_basis"] == cm.DENSITY_WEIGHT_BASIS


# ---------------------------------------------------------------------------
# A2. weight_function changes density_weighted_sum (not only density_metric_raw).
# ---------------------------------------------------------------------------
def test_a2_density_weighted_sum_follows_weight_function(tmp_path: Path) -> None:
    h_rows = [_harmonic_row(k, 220.0, 12.0) for k in range(1, 6)]
    i_rows = [_band_row(800.0 + 50 * k, 5.0) for k in range(2)]
    s_rows = [_band_row(45.0, 2.0)]
    p = tmp_path / "wf_sensitivity.xlsx"
    _write_per_note_workbook(
        p,
        harmonic_rows=h_rows,
        inharmonic_rows=i_rows,
        subbass_rows=s_rows,
        component_weights=(0.8, 0.15, 0.05),
    )
    info_lin = cm.extract_density_components_from_per_note_workbook(
        p, weight_function="linear"
    )
    info_log = cm.extract_density_components_from_per_note_workbook(
        p, weight_function="log"
    )
    assert info_lin["density_extraction_status"] == "ok"
    assert info_log["density_extraction_status"] == "ok"
    assert info_lin["density_weighted_sum"] == pytest.approx(49.6)
    assert info_log["density_weighted_sum"] != info_lin["density_weighted_sum"]
    # Same closed form as density_metric_raw in the row builder.
    wsum_log = (
        0.8 * float(info_log["harmonic_density_sum"])
        + 0.15 * float(info_log["inharmonic_density_sum"])
        + 0.05 * float(info_log["subbass_density_sum"])
    )
    assert info_log["density_weighted_sum"] == pytest.approx(wsum_log)
    df = cm._build_density_metrics_sheet_from_per_note_files(
        [(p, "A4", "")], weight_function="log"
    )
    assert df.iloc[0]["density_weighted_sum"] == pytest.approx(
        df.iloc[0]["density_metric_raw"]
    )


# ---------------------------------------------------------------------------
# B. Amplitude_raw is preferred when both columns are present.
# ---------------------------------------------------------------------------
def test_b_amplitude_raw_preferred_over_power_raw(tmp_path: Path) -> None:
    # Build rows where Amplitude_raw == 7 but Power_raw == 7**2 = 49 per row.
    # If the extractor wrongly read Power_raw, the inharmonic_amplitude_sum
    # would be 49*N instead of 7*N — easy to detect.
    inharmonic_rows = [
        {
            "Frequency (Hz)": 1000.0 + 50 * k,
            "Magnitude (dB)": 20.0 * math.log10(7.0),
            "Amplitude_raw": 7.0,
            "Power_raw": 49.0,
        }
        for k in range(4)
    ]
    subbass_rows = [
        {
            "Frequency (Hz)": 40.0 + 5 * k,
            "Magnitude (dB)": 20.0 * math.log10(3.0),
            "Amplitude_raw": 3.0,
            "Power_raw": 9.0,
        }
        for k in range(2)
    ]
    harmonic_rows = [_harmonic_row(k, 220.0, 1.0) for k in range(1, 4)]

    p = tmp_path / "raw_preference.xlsx"
    _write_per_note_workbook(
        p,
        harmonic_rows=harmonic_rows,
        inharmonic_rows=inharmonic_rows,
        subbass_rows=subbass_rows,
    )
    info = cm.extract_density_components_from_per_note_workbook(p)

    assert info["density_extraction_status"] == "ok"
    assert info["inharmonic_amplitude_sum"] == pytest.approx(28.0)
    assert info["subbass_amplitude_sum"] == pytest.approx(6.0)
    # Provenance must reference the *raw* column, not the power column.
    assert "Amplitude_raw" in str(info["inharmonic_amplitude_source"])
    assert "Amplitude_raw" in str(info["subbass_amplitude_source"])
    assert "Power_raw" not in str(info["inharmonic_amplitude_source"])
    assert "Power_raw" not in str(info["subbass_amplitude_source"])


# ---------------------------------------------------------------------------
# C. Missing component_* weights ⇒ missing_component_weights, NaN density.
# ---------------------------------------------------------------------------
def test_c_missing_component_weights_marks_missing_and_no_legacy_fallback(
    tmp_path: Path,
) -> None:
    h_rows = [_harmonic_row(k, 220.0, 12.0) for k in range(1, 6)]
    i_rows = [_band_row(800.0 + 50 * k, 5.0) for k in range(2)]
    s_rows = [_band_row(45.0, 2.0)]

    # Workbook with NO component_* rows. Only legacy alias rows ⇒ caller
    # must NOT fall back to them; status must say missing_component_weights.
    legacy_rows = [
        ("batch_harmonic_energy_ratio", 0.8),
        ("batch_inharmonic_energy_ratio", 0.15),
        ("batch_subbass_energy_ratio", 0.05),
    ]
    p = tmp_path / "missing_weights.xlsx"
    _write_per_note_workbook(
        p,
        harmonic_rows=h_rows,
        inharmonic_rows=i_rows,
        subbass_rows=s_rows,
        component_weights=None,
        extra_meta_rows=legacy_rows,
    )
    info = cm.extract_density_components_from_per_note_workbook(p)

    assert info["density_extraction_status"] == "missing_component_weights"
    # Weighted-density fields must be None (not legacy-fallback values).
    assert info["weighted_harmonic_component"] is None
    assert info["weighted_inharmonic_component"] is None
    assert info["weighted_subbass_component"] is None
    assert info["density_weighted_sum"] is None
    assert info["density_log_weighted"] is None
    # The amplitude sums are still extracted (they don't depend on weights).
    assert info["harmonic_amplitude_sum"] == pytest.approx(60.0)
    assert info["inharmonic_amplitude_sum"] == pytest.approx(10.0)
    assert info["subbass_amplitude_sum"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# D. density_log_weighted is not normalised to [0, 1].
# ---------------------------------------------------------------------------
def test_d_density_log_weighted_is_not_normalised(tmp_path: Path) -> None:
    # Make a per-note workbook where the weighted sum is much > 1 so log10
    # exceeds 1. With H sum = 1000, weight 1.0, the log10(1+1000) ≈ 3.0.
    h_rows = [_harmonic_row(1, 220.0, 1000.0)]
    i_rows = [_band_row(800.0, 0.0)]
    s_rows = [_band_row(40.0, 0.0)]

    p = tmp_path / "large_sum.xlsx"
    _write_per_note_workbook(
        p,
        harmonic_rows=h_rows,
        inharmonic_rows=i_rows,
        subbass_rows=s_rows,
        component_weights=(1.0, 0.0, 0.0),
    )
    info = cm.extract_density_components_from_per_note_workbook(p)
    log_val = info["density_log_weighted"]
    assert log_val is not None
    assert log_val > 1.0
    assert log_val == pytest.approx(math.log10(1.0 + 1000.0))

    # And the row-level builder also publishes the un-normalised value.
    df = cm._build_density_metrics_sheet_from_per_note_files(
        [(p, "A4", "")],
    )
    assert "density_log_weighted" in df.columns
    assert float(df.loc[0, "density_log_weighted"]) > 1.0


# ---------------------------------------------------------------------------
# E. publication policy selects density_log_weighted for Density_Metrics.
# ---------------------------------------------------------------------------
def test_e_publication_policy_selects_density_log_weighted() -> None:
    prefs = DEFAULT_PUBLICATION_METRIC_BY_SHEET["Density_Metrics"]
    assert prefs[0] == "density_log_weighted"
    assert "harmonic_log_amplitude_density" in prefs
    assert "density_metric_normalized" not in prefs
    assert "Total sum" not in prefs
    assert "Harmonic Partials sum" not in prefs

    columns = {
        "Note",
        "Harmonic Partials sum",
        "density_metric_normalized",
        "density_log_weighted",
        "harmonic_log_amplitude_density",
    }
    sel = select_default_publication_metric(
        columns, sheet_name="Density_Metrics"
    )
    assert sel == "density_log_weighted"

    # When the weighted metric is missing, the harmonic-only log is selected.
    columns2 = columns - {"density_log_weighted"}
    sel2 = select_default_publication_metric(
        columns2, sheet_name="Density_Metrics"
    )
    assert sel2 == "harmonic_log_amplitude_density"


# ---------------------------------------------------------------------------
# F. Real-workbook smoke: density_log_weighted is finite, harmonic
#    log unchanged from the Stage 1 audit.
# ---------------------------------------------------------------------------
def test_f_existing_per_note_workbook_smoke(tmp_path: Path) -> None:
    """Build a complete synthetic workbook end-to-end and run the
    row builder. We assert the row carries the weighted-density value
    alongside the unchanged Stage 1 ``harmonic_log_amplitude_density``."""
    h_rows = [_harmonic_row(k, 220.0, 1.0 / k) for k in range(1, 11)]  # 10 partials
    h_sum = sum(1.0 / k for k in range(1, 11))
    i_rows = [_band_row(1000.0 + 30 * k, 0.05) for k in range(3)]
    s_rows = [_band_row(45.0, 0.01)]
    p = tmp_path / "smoke_per_note.xlsx"
    _write_per_note_workbook(
        p,
        harmonic_rows=h_rows,
        inharmonic_rows=i_rows,
        subbass_rows=s_rows,
        component_weights=(0.85, 0.13, 0.02),
    )

    df = cm._build_density_metrics_sheet_from_per_note_files(
        [(p, "A4", "")],
    )
    assert len(df) == 1
    row = df.iloc[0]
    # Stage 1 harmonic-log-amplitude is preserved.
    assert row["harmonic_amplitude_sum"] == pytest.approx(h_sum)
    assert row["harmonic_log_amplitude_density"] == pytest.approx(
        math.log10(1.0 + h_sum)
    )
    # Stage 2 weighted-density is present, finite, and equals the closed-
    # form formula.
    wsum = 0.85 * h_sum + 0.13 * (0.05 * 3) + 0.02 * 0.01
    assert row["density_weighted_sum"] == pytest.approx(wsum)
    assert row["density_log_weighted"] == pytest.approx(math.log10(1.0 + wsum))
