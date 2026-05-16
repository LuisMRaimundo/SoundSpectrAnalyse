# -*- coding: utf-8 -*-
"""Debug_Counts hierarchy and peaklist separation."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compile_metrics import _build_debug_counts_sheet  # noqa: E402
from debug_counts import validate_debug_count_invariants  # noqa: E402
from peak_component_counts import classify_peaks_harmonic_inharmonic_subbass_from_df  # noqa: E402


def test_validate_debug_counts_passes_hierarchy() -> None:
    row = {
        "residual_spectral_row_count": 100,
        "nonharmonic_candidate_row_count": 80,
        "retained_nonharmonic_peak_candidate_count": 20,
        "exported_nonharmonic_peak_candidate_count": 20,
        "accepted_inharmonic_peak_count": None,
        "accepted_inharmonic_partial_count": None,
    }
    validate_debug_count_invariants(row)
    assert row["debug_counts_invariant_status"] == "passed"
    assert row["debug_counts_invariant_failures"] == ""


def test_validate_debug_counts_fails_when_retained_exceeds_candidate() -> None:
    row = {
        "residual_spectral_row_count": 10,
        "nonharmonic_candidate_row_count": 5,
        "retained_nonharmonic_peak_candidate_count": 8,
        "exported_nonharmonic_peak_candidate_count": 8,
    }
    validate_debug_count_invariants(row)
    assert row["debug_counts_invariant_status"] == "failed"
    assert "retained_nonharmonic_peak_candidate_count_exceeds" in row["debug_counts_invariant_failures"]


def test_peaklist_nonharmonic_may_exceed_residual_without_hierarchy_violation() -> None:
    """Peaklist counts are independent; validator does not compare them to residual rows."""
    peaks = pd.DataFrame(
        {
            "Frequency (Hz)": [440.0, 880.0, 900.0, 50.0],
            "Amplitude": [1.0, 0.5, 0.2, 0.15],
        }
    )
    out = classify_peaks_harmonic_inharmonic_subbass_from_df(
        peaks, 440.0, subbass_cutoff_hz=200.0, tolerance_cents=50.0, max_freq_hz=5000.0
    )
    pl_nh = int(out["peaklist_nonharmonic_window_candidate_count"])
    row = {
        "residual_spectral_row_count": 5,
        "nonharmonic_candidate_row_count": 5,
        "retained_nonharmonic_peak_candidate_count": 3,
        "exported_nonharmonic_peak_candidate_count": 3,
    }
    validate_debug_count_invariants(row)
    assert row["debug_counts_invariant_status"] == "passed"
    assert pl_nh >= 0


def test_build_debug_counts_sheet_prefers_retained_for_inharmonic_candidate() -> None:
    row = {
        "Note": "A#5",
        "residual_spectral_row_count": 1169,
        "nonharmonic_candidate_row_count": 1169,
        "retained_nonharmonic_peak_candidate_count": 134,
        "exported_nonharmonic_peak_candidate_count": 134,
        "peaklist_nonharmonic_window_candidate_count": 1340,
        "legacy_nonharmonic_peak_candidate_count_deprecated": 1340,
        "harmonic_peak_candidate_count": 4,
        "low_frequency_peak_candidate_count": 1,
        "total_peak_candidate_count": 200,
        "accepted_inharmonic_peak_count": "",
        "accepted_inharmonic_partial_count": "",
        "debug_counts_semantics": "x",
        "debug_counts_source_policy": "y",
        "debug_counts_invariant_status": "passed",
        "debug_counts_invariant_failures": "",
        "debug_counts_status": "computed",
        "inharmonic_bin_count_deprecated_legacy_alias": 1169,
        "inharmonic_peak_count_deprecated_legacy_alias": 1340,
        "harmonic_peak_count_deprecated_legacy_alias": 4,
        "subbass_peak_count_deprecated_legacy_alias": 1,
        "total_detected_peak_count_deprecated_legacy_alias": 200,
        "harmonic_bin_count": 40,
        "subbass_bin_count": 2,
        "residual_row_count": 50,
    }
    dbg = _build_debug_counts_sheet(pd.DataFrame([row]))
    assert dbg is not None
    assert int(pd.to_numeric(dbg["inharmonic_candidate_count"].iloc[0], errors="coerce")) == 134
    assert int(pd.to_numeric(dbg["inharmonic_peak_count"].iloc[0], errors="coerce")) == 1340


def test_no_publication_primary_nonharmonic_peak_candidate_without_legacy_marker() -> None:
    """New-style rows omit ambiguous ``nonharmonic_peak_candidate_count`` unless old export."""
    row = {
        "Note": "C4",
        "residual_spectral_row_count": 10,
        "nonharmonic_candidate_row_count": 10,
        "retained_nonharmonic_peak_candidate_count": 4,
        "exported_nonharmonic_peak_candidate_count": 4,
        "peaklist_harmonic_window_candidate_count": 2,
        "peaklist_nonharmonic_window_candidate_count": 99,
        "peaklist_low_frequency_window_candidate_count": 0,
        "peaklist_total_window_candidate_count": 101,
        "harmonic_peak_candidate_count": 2,
        "low_frequency_peak_candidate_count": 0,
        "total_peak_candidate_count": 101,
        "debug_counts_semantics": "s",
        "debug_counts_source_policy": "p",
        "debug_counts_invariant_status": "passed",
        "debug_counts_invariant_failures": "",
        "debug_counts_status": "computed",
        "harmonic_bin_count": 1,
        "subbass_bin_count": 0,
        "residual_row_count": 20,
    }
    dbg = _build_debug_counts_sheet(pd.DataFrame([row]))
    assert dbg is not None
    assert "nonharmonic_peak_candidate_count" not in dbg.columns
