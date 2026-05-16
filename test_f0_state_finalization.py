"""F0 provenance: single finalizer + hard consistency checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proc_audio import AudioProcessor


def _new_ap() -> AudioProcessor:
    return AudioProcessor.__new__(AudioProcessor)


def test_finalize_f0_accepted_sets_consistent_strings() -> None:
    ap = _new_ap()
    AudioProcessor._finalize_f0_state(
        ap,
        nominal_hz=466.1637615,
        candidate_hz=466.2,
        accept_fit=True,
        fit_quality=0.02,
        residual_std_hz=0.4,
        rejection_reason=None,
    )
    assert ap.f0_fit_accepted is True
    assert ap.f0_final_source == "prior_constrained_harmonic_fit"
    assert ap.f0_source == "prior_constrained_harmonic_fit"
    assert ap.f0_final_method == "prior_constrained_harmonic_fit"
    assert ap.f0_fit_rejection_reason is None
    assert ap.f0_robust_accepted is True


def test_finalize_f0_rejected_sets_nominal_fallback() -> None:
    ap = _new_ap()
    AudioProcessor._finalize_f0_state(
        ap,
        nominal_hz=466.1637615,
        candidate_hz=470.0,
        accept_fit=False,
        fit_quality=0.5,
        residual_std_hz=2.0,
        rejection_reason="fit_quality_exceeds_gate",
    )
    assert ap.f0_fit_accepted is False
    assert ap.f0_final == pytest.approx(466.1637615)
    assert ap.f0_final_source == "filename_note_nominal_fallback_fit_rejected"
    assert ap.f0_source == "filename_note_nominal_fallback_fit_rejected"
    assert ap.f0_final_method == "nominal_or_initial_due_to_bad_fit"
    assert ap.f0_fit_rejection_reason == "fit_quality_exceeds_gate"
    assert ap.f0_robust_accepted is False


def test_assert_f0_raises_on_prior_source_with_rejected_flag() -> None:
    ap = _new_ap()
    ap.f0_final_source = "prior_constrained_harmonic_fit"
    ap.f0_fit_accepted = False
    with pytest.raises(RuntimeError, match="prior_constrained_harmonic_fit"):
        AudioProcessor._assert_f0_state_consistency(ap)


def test_assert_f0_raises_on_accepted_without_prior_source() -> None:
    ap = _new_ap()
    ap.f0_final_source = "filename_note_nominal_fallback_fit_rejected"
    ap.f0_fit_accepted = True
    with pytest.raises(RuntimeError, match="f0_fit_accepted is True"):
        AudioProcessor._assert_f0_state_consistency(ap)


def test_validation_dict_matches_instance_f0_fields() -> None:
    ap = _new_ap()
    AudioProcessor._finalize_f0_state(
        ap,
        nominal_hz=440.0,
        candidate_hz=441.0,
        accept_fit=True,
        fit_quality=0.01,
        residual_std_hz=0.2,
        rejection_reason=None,
    )
    ap.f0_nominal_hz = 440.0
    row = {
        "f0_estimated": float(getattr(ap, "f0_final", float("nan"))),
        "f0_source": str(getattr(ap, "f0_final_source", "unresolved")),
        "f0_nominal_hz": float(getattr(ap, "f0_nominal_hz", float("nan"))),
        "f0_final_hz": float(getattr(ap, "f0_final", float("nan"))),
        "f0_final_method": str(getattr(ap, "f0_final_method", "")),
        "f0_final_source": str(getattr(ap, "f0_final_source", "")),
        "f0_detuning_cents_from_nominal": getattr(ap, "f0_detuning_cents_from_nominal", None),
        "f0_fit_accepted": bool(getattr(ap, "f0_fit_accepted", False)),
        "f0_fit_quality": getattr(ap, "f0_fit_quality", None),
        "f0_fit_residual_std_hz": getattr(ap, "f0_robust_residual_std", None),
        "f0_fit_rejection_reason": getattr(ap, "f0_fit_rejection_reason", None),
    }
    assert row["f0_source"] == row["f0_final_source"]
    assert row["f0_fit_accepted"] is True
    assert row["f0_final_source"] == "prior_constrained_harmonic_fit"
    _rej = row["f0_fit_rejection_reason"]
    assert _rej is None or (isinstance(_rej, float) and np.isnan(_rej)) or str(_rej).strip() == ""


def test_prior_source_implies_accepted_invariant() -> None:
    ap = _new_ap()
    for accept in (True, False):
        AudioProcessor._finalize_f0_state(
            ap,
            nominal_hz=440.0,
            candidate_hz=441.0 if accept else 500.0,
            accept_fit=accept,
            fit_quality=0.01 if accept else 0.9,
            residual_std_hz=0.1,
            rejection_reason=None if accept else "test_reject",
        )
        src = str(getattr(ap, "f0_final_source", "") or "")
        acc = bool(getattr(ap, "f0_fit_accepted", False))
        if src == "prior_constrained_harmonic_fit":
            assert acc is True
        if acc is False:
            assert src != "prior_constrained_harmonic_fit"
