# -*- coding: utf-8 -*-
"""DEPRECATED tests: this module exercises the removed batch/orchestrator
handoff (``_resolve_batch_energy_and_model_weights``) that lived in the
pre-refactor three-phase pipeline. The current Stage 1 + Stage 2 pipeline
derives model weights directly from per-note ``proc_audio`` analysis, so the
batch payload entry point is gone. Kept here for historical reference only.
"""

from __future__ import annotations

import math

import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "deprecated: tests removed Stage 1 / Batch orchestrator handoff; the "
        "new pipeline sources model weights from current_analysis only."
    )
)

from pipeline_orchestrator_integrated import RobustOrchestrator


def _resolve(payload: dict) -> tuple:
    return RobustOrchestrator._resolve_batch_energy_and_model_weights(payload)


def test_valid_his_triplet_accepted_case_a() -> None:
    payload = {
        "harmonic_percentage": 90.0,
        "inharmonic_percentage": 5.0,
        "subbass_energy_percentage_global": 5.0,
        "batch_harmonic_energy_ratio": 0.90,
        "batch_inharmonic_energy_ratio": 0.05,
        "batch_subbass_energy_ratio": 0.05,
        "batch_ratio_source_explicit": True,
    }
    ok, reason, mh, mi, mws, meta = _resolve(payload)
    assert ok and reason is None
    assert math.isclose(mh, 0.90 / 0.95, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mi, 0.05 / 0.95, rel_tol=0, abs_tol=1e-9)
    assert mws == "batch_empirical_energy_ratios"
    assert meta["batch_harmonic_energy_ratio"] == 0.90
    assert meta["batch_subbass_energy_ratio"] == 0.05
    assert meta.get("batch_energy_denominator") == "harmonic_plus_inharmonic_plus_subbass"
    assert math.isclose(float(meta["batch_total_inharmonic_energy_ratio"]), 0.10)


def test_valid_triplet_large_subbass_case_b() -> None:
    payload = {
        "harmonic_percentage": 70.0,
        "inharmonic_percentage": 10.0,
        "subbass_energy_percentage_global": 20.0,
        "batch_harmonic_energy_ratio": 0.70,
        "batch_inharmonic_energy_ratio": 0.10,
        "batch_subbass_energy_ratio": 0.20,
        "batch_ratio_source_explicit": True,
    }
    ok, reason, mh, mi, mws, meta = _resolve(payload)
    assert ok and reason is None
    assert math.isclose(mh, 0.70 / 0.80, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mi, 0.10 / 0.80, rel_tol=0, abs_tol=1e-9)
    assert mws == "batch_empirical_energy_ratios"
    assert meta["batch_harmonic_energy_ratio"] == 0.70
    assert meta["batch_subbass_energy_ratio"] == 0.20


def test_highly_harmonic_flute_like_triplet_case_a() -> None:
    """H+I+S≈1; model harmonic weight may exceed 0.95 — must stay batch_empirical."""
    payload = {
        "harmonic_percentage": 98.3,
        "inharmonic_percentage": 1.46,
        "subbass_energy_percentage_global": 0.24,
        "batch_harmonic_energy_ratio": 0.983,
        "batch_inharmonic_energy_ratio": 0.0146,
        "batch_subbass_energy_ratio": 0.0024,
        "batch_ratio_source_explicit": True,
    }
    ok, reason, mh, mi, mws, meta = _resolve(payload)
    assert ok and reason is None
    assert mws == "batch_empirical_energy_ratios"
    assert mh is not None and mh > 0.95
    assert math.isclose(mh, 0.983 / (0.983 + 0.0146), rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mi, 0.0146 / (0.983 + 0.0146), rel_tol=0, abs_tol=1e-9)
    assert meta["batch_subbass_energy_ratio"] == 0.0024


def test_highly_inharmonic_valid_triplet_case_b() -> None:
    payload = {
        "harmonic_percentage": 20.0,
        "inharmonic_percentage": 70.0,
        "subbass_energy_percentage_global": 10.0,
        "batch_harmonic_energy_ratio": 0.20,
        "batch_inharmonic_energy_ratio": 0.70,
        "batch_subbass_energy_ratio": 0.10,
        "batch_ratio_source_explicit": True,
    }
    ok, reason, mh, mi, mws, meta = _resolve(payload)
    assert ok and reason is None
    assert mws == "batch_empirical_energy_ratios"
    assert math.isclose(mh, 0.20 / 0.90, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mi, 0.70 / 0.90, rel_tol=0, abs_tol=1e-9)
    warn = RobustOrchestrator._model_weights_warnings(mh, mi)
    assert "low_musical_harmonic_fraction" in warn


def test_model_weights_warnings_extreme_high() -> None:
    mh, mi = 0.983 / (0.983 + 0.0146), 0.0146 / (0.983 + 0.0146)
    w = RobustOrchestrator._model_weights_warnings(mh, mi)
    assert "extreme_empirical_model_weight_high" in w


def test_legacy_bounded_weights_clip_harmonic() -> None:
    mh, mi = 0.983 / (0.983 + 0.0146), 0.0146 / (0.983 + 0.0146)
    lb_h, lb_i = RobustOrchestrator._legacy_bounded_model_weights(mh, mi)
    assert lb_h == 0.95
    assert math.isclose(lb_i, 0.05, rel_tol=0, abs_tol=1e-9)


def test_invalid_triplet_rejected_case_c() -> None:
    payload = {
        "harmonic_percentage": 70.0,
        "inharmonic_percentage": 10.0,
        "subbass_energy_percentage_global": 50.0,
        "batch_harmonic_energy_ratio": 0.70,
        "batch_inharmonic_energy_ratio": 0.10,
        "batch_subbass_energy_ratio": 0.50,
        "batch_ratio_source_explicit": True,
    }
    ok, reason, mh, mi, mws, meta = _resolve(payload)
    assert not ok
    assert reason is not None
    assert mh is None and mi is None
    assert mws == "fallback_default"


def test_legacy_hi_percent_only_case_d() -> None:
    payload = {
        "harmonic_percentage": 95.0,
        "inharmonic_percentage": 5.0,
        "batch_harmonic_energy_ratio": 0.95,
        "batch_inharmonic_energy_ratio": 0.05,
        "batch_ratio_source_explicit": False,
    }
    ok, reason, mh, mi, mws, meta = _resolve(payload)
    assert ok
    assert math.isclose(mh, 0.95, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mi, 0.05, rel_tol=0, abs_tol=1e-9)
    assert mws == "legacy_batch_hi_percent"
    assert meta.get("batch_subbass_energy_ratio") is None


def test_legacy_trip_percent_case_e() -> None:
    """Legacy 0–100 columns only (no batch_* ratios): validate H+I+S≈100, then H/(H+I) model weights."""
    payload = {
        "harmonic_percentage": 85.0,
        "inharmonic_percentage": 5.0,
        "subbass_energy_percentage_global": 10.0,
        "batch_ratio_source_explicit": False,
    }
    ok, reason, mh, mi, mws, meta = _resolve(payload)
    assert ok and reason is None
    assert math.isclose(mh, 85.0 / 90.0, rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mi, 5.0 / 90.0, rel_tol=0, abs_tol=1e-9)
    assert mws == "batch_empirical_energy_ratios"


def test_fallback_metadata_case_f() -> None:
    payload = {
        "harmonic_percentage": 40.0,
        "inharmonic_percentage": 50.0,
        "subbass_energy_percentage_global": 30.0,
        "batch_ratio_source_explicit": False,
    }
    ok, reason, mh, mi, mws, meta = _resolve(payload)
    assert not ok
    assert mws == "fallback_default"
    assert reason is not None and len(reason) > 0


def test_explicit_ratio_gt_one_rejected() -> None:
    payload = {
        "harmonic_percentage": 90.0,
        "inharmonic_percentage": 10.0,
        "batch_harmonic_energy_ratio": 1.2,
        "batch_inharmonic_energy_ratio": 0.05,
        "batch_subbass_energy_ratio": 0.05,
        "batch_ratio_source_explicit": True,
    }
    ok, reason, mh, mi, mws, _meta = _resolve(payload)
    assert not ok
    assert "invalid explicit batch ratio" in (reason or "")
    assert mws == "fallback_default"


@pytest.mark.parametrize(
    "H,I,S",
    [
        (0.90, 0.05, 0.05),
        (0.70, 0.10, 0.20),
    ],
)
def test_triplet_path_prefers_ratios_when_percentages_also_present(H: float, I: float, S: float) -> None:
    """Ratio triplet is evaluated first; model weights use H/(H+I) from ratios."""
    payload = {
        "harmonic_percentage": 50.0,
        "inharmonic_percentage": 25.0,
        "subbass_energy_percentage_global": 25.0,
        "batch_harmonic_energy_ratio": H,
        "batch_inharmonic_energy_ratio": I,
        "batch_subbass_energy_ratio": S,
        "batch_ratio_source_explicit": True,
    }
    ok, _r, mh, mi, mws, _meta = _resolve(payload)
    assert ok
    assert mws == "batch_empirical_energy_ratios"
    assert math.isclose(mh, H / (H + I), rel_tol=0, abs_tol=1e-9)
    assert math.isclose(mi, I / (H + I), rel_tol=0, abs_tol=1e-9)
