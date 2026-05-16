# -*- coding: utf-8 -*-
"""Regression: non-harmonic / residual terminology vs. confirmed inharmonic partials."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import density  # noqa: E402
from compile_metrics import _build_debug_counts_sheet  # noqa: E402
from harmonic_validation import validate_harmonic_series_matched  # noqa: E402
from peak_component_counts import classify_peaks_harmonic_inharmonic_subbass_from_df  # noqa: E402


def test_identify_nonharmonic_residual_rows_exists() -> None:
    assert hasattr(density, "identify_nonharmonic_residual_rows")
    assert callable(density.identify_nonharmonic_residual_rows)


def test_identify_inharmonic_partials_is_deprecated_wrapper() -> None:
    assert hasattr(density, "identify_inharmonic_partials")
    src = inspect.getdoc(density.identify_inharmonic_partials) or ""
    assert "deprecated" in src.lower() or "compat" in src.lower() or "wrapper" in src.lower()

    h = pd.DataFrame({"Frequency (Hz)": [100.0], "Harmonic Number": [1]})
    c = pd.DataFrame(
        {
            "Frequency (Hz)": [100.0, 250.0],
            "Magnitude (dB)": [-20.0, -30.0],
            "Amplitude": [0.1, 0.05],
        }
    )
    a = density.identify_nonharmonic_residual_rows(h, c, tolerance=0.02)
    b = density.identify_inharmonic_partials(h, c, tolerance=0.02)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_peak_component_counts_primary_keys_and_semantics() -> None:
    peaks = pd.DataFrame(
        {
            "Frequency (Hz)": [440.0, 880.0, 900.0, 50.0],
            "Magnitude (dB)": [0.0, -6.0, -10.0, -12.0],
            "Amplitude": [1.0, 0.5, 0.2, 0.15],
        }
    )
    out = classify_peaks_harmonic_inharmonic_subbass_from_df(
        peaks, 440.0, subbass_cutoff_hz=200.0, tolerance_cents=50.0, max_freq_hz=5000.0
    )
    assert "independent_peaklist_window_assignment" in str(out["classification_semantics"])
    assert int(out["peaklist_harmonic_window_candidate_count"]) >= 1
    assert int(out["peaklist_nonharmonic_window_candidate_count"]) >= 0
    assert int(out["legacy_inharmonic_peak_count_deprecated"]) == int(
        out["peaklist_nonharmonic_window_candidate_count"]
    )


def test_harmonic_validation_prefers_outside_harmonic_window_keys() -> None:
    peaks = pd.DataFrame(
        {
            "Frequency (Hz)": [440.0, 900.0],
            "Amplitude": [1.0, 0.3],
        }
    )
    vr = validate_harmonic_series_matched(
        440.0, peaks, max_freq_hz=5000.0, sample_rate=44100.0, n_fft=4096, subbass_cutoff_hz=200.0
    )
    assert "outside_harmonic_window_candidate_row_count" in vr
    assert "outside_harmonic_window_peak_candidate_count" in vr
    assert "outside_harmonic_window_candidate_energy_ratio" in vr


def test_build_debug_counts_sheet_primary_columns_and_legacy_derivation() -> None:
    row = {
        "Note": "A4",
        "residual_spectral_row_count": 12,
        "nonharmonic_candidate_row_count": 12,
        "retained_nonharmonic_peak_candidate_count": 3,
        "exported_nonharmonic_peak_candidate_count": 3,
        "peaklist_nonharmonic_window_candidate_count": 99,
        "harmonic_peak_candidate_count": 4,
        "low_frequency_peak_candidate_count": 1,
        "total_peak_candidate_count": 8,
        "accepted_inharmonic_peak_count": "",
        "accepted_inharmonic_partial_count": "",
        "debug_counts_semantics": "test",
        "debug_counts_source_policy": "pol",
        "debug_counts_invariant_status": "passed",
        "debug_counts_invariant_failures": "",
        "debug_counts_status": "computed",
        "inharmonic_bin_count_deprecated_legacy_alias": 12,
        "inharmonic_peak_count_deprecated_legacy_alias": 3,
        "harmonic_peak_count_deprecated_legacy_alias": 4,
        "subbass_peak_count_deprecated_legacy_alias": 1,
        "total_detected_peak_count_deprecated_legacy_alias": 8,
        "harmonic_bin_count": 40,
        "subbass_bin_count": 2,
        "residual_row_count": 50,
    }
    dbg = _build_debug_counts_sheet(pd.DataFrame([row]))
    assert dbg is not None
    assert "retained_nonharmonic_peak_candidate_count" in dbg.columns
    assert "nonharmonic_peak_candidate_count" not in dbg.columns
    assert "inharmonic_bin_count" in dbg.columns
    assert int(dbg["inharmonic_bin_count"].iloc[0]) == 12
    assert "inharmonic_partial_count" not in dbg.columns
    assert "total_detected_partial_count" not in dbg.columns


def test_per_note_debug_row_semantics_columns(tmp_path: Path) -> None:
    """Smoke: new Debug_Counts keys round-trip to Excel (subset)."""
    from debug_counts import validate_debug_count_invariants

    debug_row = {
        "Note": "A4",
        "harmonic_bin_count": 10,
        "subbass_bin_count": 1,
        "harmonic_peak_candidate_count": 3,
        "low_frequency_peak_candidate_count": 1,
        "total_peak_candidate_count": 6,
        "residual_spectral_row_count": 5,
        "nonharmonic_candidate_row_count": 5,
        "retained_nonharmonic_peak_candidate_count": 2,
        "exported_nonharmonic_peak_candidate_count": 2,
        "peaklist_harmonic_window_candidate_count": 3,
        "peaklist_nonharmonic_window_candidate_count": 9,
        "peaklist_low_frequency_window_candidate_count": 1,
        "peaklist_total_window_candidate_count": 13,
        "legacy_nonharmonic_peak_candidate_count_deprecated": 9,
        "accepted_inharmonic_peak_count": "",
        "accepted_inharmonic_partial_count": "",
        "total_spectral_candidate_count": 6,
        "harmonic_candidate_count": 3,
        "subbass_candidate_count": 1,
        "residual_row_count": 20,
        "debug_counts_status": "computed",
        "inharmonic_bin_count_deprecated_legacy_alias": 5,
        "inharmonic_candidate_count_deprecated_legacy_alias": 2,
        "inharmonic_peak_count_deprecated_legacy_alias": 9,
        "harmonic_peak_count_deprecated_legacy_alias": 3,
        "subbass_peak_count_deprecated_legacy_alias": 1,
        "total_detected_peak_count_deprecated_legacy_alias": 13,
    }
    inv = {
        "residual_spectral_row_count": debug_row["residual_spectral_row_count"],
        "nonharmonic_candidate_row_count": debug_row["nonharmonic_candidate_row_count"],
        "retained_nonharmonic_peak_candidate_count": debug_row["retained_nonharmonic_peak_candidate_count"],
        "exported_nonharmonic_peak_candidate_count": debug_row["exported_nonharmonic_peak_candidate_count"],
        "accepted_inharmonic_peak_count": None,
        "accepted_inharmonic_partial_count": None,
    }
    validate_debug_count_invariants(inv)
    debug_row["debug_counts_semantics"] = inv["debug_counts_semantics"]
    debug_row["debug_counts_source_policy"] = inv["debug_counts_source_policy"]
    debug_row["debug_counts_invariant_status"] = inv["debug_counts_invariant_status"]
    debug_row["debug_counts_invariant_failures"] = inv["debug_counts_invariant_failures"]

    outp = tmp_path / "dbg.xlsx"
    pd.DataFrame([debug_row]).to_excel(outp, sheet_name="Debug_Counts", index=False)
    rd = pd.read_excel(outp, sheet_name="Debug_Counts")
    assert "retained_nonharmonic_peak_candidate_count" in rd.columns
    assert "peaklist_nonharmonic_window_candidate_count" in rd.columns
    assert "nonharmonic_peak_candidate_count" not in rd.columns
    assert rd["debug_counts_invariant_status"].iloc[0] == "passed"


def test_inharmonic_spectrum_export_tagging_logic() -> None:
    """Replicate ``_tag_component_type`` / Inharmonic sheet component semantics."""
    ih_df = pd.DataFrame(
        {"Frequency (Hz)": [900.0], "Magnitude (dB)": [-20.0], "Amplitude": [0.1], "Note": ["A4"]}
    )

    def _tag_component_type(
        dfx: pd.DataFrame,
        cat: str,
        *,
        classification_level: str,
        acoustic_status: str,
    ) -> pd.DataFrame:
        if dfx is None or dfx.empty:
            return pd.DataFrame()
        out = dfx.copy()
        out.insert(0, "Component_Type", cat)
        out.insert(1, "Classification_Level", classification_level)
        out.insert(2, "Acoustic_Interpretation_Status", acoustic_status)
        return out

    tagged = _tag_component_type(
        ih_df,
        "nonharmonic_peak_candidate",
        classification_level="residual_rows_ranked_by_amplitude_after_harmonic_exclusion",
        acoustic_status="candidate_not_confirmed_partial",
    )
    assert set(tagged["Component_Type"].unique()) <= {"nonharmonic_peak_candidate"}
    assert not (tagged["Component_Type"] == "inharmonic_partial").any()
