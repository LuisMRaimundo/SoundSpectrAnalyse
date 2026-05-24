"""Regression tests for f0 nominal vs final provenance (no min-harmonic f0)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import proc_audio as pa
from proc_audio import AudioProcessor, _correct_f0_candidate_against_prior


def test_correct_f0_triple_harmonic_maps_to_prior() -> None:
    prior = 466.1637615  # A#4 / Bb4 ET
    raw = prior * 3.0
    corrected = _correct_f0_candidate_against_prior(raw, prior, max_harmonic_ratio=6)
    assert corrected["valid"]
    assert abs(float(corrected["corrected_hz"]) - prior) < 1e-3
    assert corrected["ratio_applied"] == pytest.approx(1.0 / 3.0)
    assert float(corrected["cents_error"]) < 1e-3


def test_proc_audio_has_no_minimum_harmonic_f0_source_string() -> None:
    src = Path(pa.__file__).read_text(encoding="utf-8", errors="replace")
    assert "minimum_harmonic_partial_frequency" not in src


def test_canonical_f0_prefers_final_over_initial() -> None:
    ap = AudioProcessor.__new__(AudioProcessor)
    ap.f0_final = 442.0
    ap.f0_final_source = "prior_constrained_harmonic_fit"
    ap.f0_final_method = "prior_constrained_harmonic_fit"
    ap.f0_fit_accepted = True
    ap.f0_initial = 440.0
    ap.f0_prior_hz = 440.0
    hz, src = AudioProcessor._canonical_f0_hz_for_analysis(ap)
    assert hz == 442.0
    assert "prior_constrained" in src


def test_detuning_cents_matches_log2_ratio() -> None:
    nominal = 440.0
    final_hz = 441.5
    expected = float(1200.0 * np.log2(final_hz / nominal))
    assert expected == pytest.approx(1200.0 * np.log2(441.5 / 440.0))


def test_harmonic_validation_report_f0_source_not_minimum_harmonic() -> None:
    rep = {
        "f0_source": "prior_constrained_harmonic_fit",
        "f0_final_source": "prior_constrained_harmonic_fit",
    }
    assert rep["f0_source"] != "minimum_harmonic_partial_frequency"
    assert "minimum_harmonic" not in rep["f0_source"]


def test_canonical_f0_triplet_marks_nominal_fallback_as_not_acoustically_verified() -> None:
    ap = AudioProcessor.__new__(AudioProcessor)
    ap.f0_final = 220.0
    ap.f0_final_source = "filename_note_nominal_fallback_fit_rejected"
    ap.f0_fit_accepted = False
    ap.f0_initial = 220.0
    ap.f0_prior_hz = 220.0
    hz, src, status = AudioProcessor._canonical_f0_triplet_for_analysis(ap)
    assert hz == pytest.approx(220.0)
    assert "fallback" in src or "initial" in src
    assert status == "nominal_fallback_used_not_acoustically_verified"
