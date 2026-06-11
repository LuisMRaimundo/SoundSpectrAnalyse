from __future__ import annotations

"""
Helper-level contract tests for subbass_policy.py.

Protects the canonical sub-bass upper-bound policy ``min(f0_hz * 0.5, 80.0)``,
invalid-input fallbacks, determinism, and alignment with the operational
boundary consumed by low_frequency_policy callers.

No production code changes. No audio, GUI, plotting, or pipeline runs.
"""

import math

import pytest

from low_frequency_policy import calculate_adaptive_subfundamental_cutoff_hz
from subbass_policy import SubBassPolicy


def _upper(f0_hz: object, *, sr_hz: float = 44100.0, n_fft: int = 4096) -> float:
    return float(
        SubBassPolicy.upper_bound_hz(
            f0_hz=f0_hz,  # type: ignore[arg-type]
            sr_hz=sr_hz,
            n_fft=n_fft,
        )
    )


# ---------------------------------------------------------------------------
# 1. Canonical cutoff contract
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("f0_hz", "expected"),
    [
        (100.0, 50.0),
        (110.0, 55.0),
        (159.9, 79.95),
        (160.0, 80.0),
        (161.0, 80.0),
        (320.0, 80.0),
        (65.41, pytest.approx(32.705, rel=1e-12)),
        (1.0, 0.5),
        (1e-6, 5e-7),
    ],
)
def test_upper_bound_hz_canonical_min_half_f0_and_80(
    f0_hz: float, expected: float
) -> None:
    assert _upper(f0_hz) == expected


def test_upper_bound_hz_is_deterministic() -> None:
    first = _upper(220.0)
    second = _upper(220.0)
    assert first == second == 80.0


def test_upper_bound_returns_float_type() -> None:
    value = _upper(220.0)
    assert isinstance(value, float)
    assert math.isfinite(value)


# ---------------------------------------------------------------------------
# 2. Formula cap boundary (half-f0 vs 80 Hz perceptual cap)
# ---------------------------------------------------------------------------

def test_half_f0_dominates_below_160_hz_fundamental() -> None:
    assert _upper(159.9) == pytest.approx(79.95, rel=1e-12)
    assert _upper(159.9) < 80.0


def test_perceptual_cap_dominates_at_and_above_160_hz_fundamental() -> None:
    assert _upper(160.0) == 80.0
    assert _upper(160.1) == 80.0
    assert _upper(1000.0) == 80.0


def test_caller_strict_less_than_boundary_semantics_pinned() -> None:
    """peak_component_counts routes f < subbass_cutoff_hz to the sub-bass bucket."""
    cutoff = _upper(110.0)
    assert cutoff == 55.0
    assert 54.999 < cutoff
    assert 55.0 >= cutoff


# ---------------------------------------------------------------------------
# 3. Invalid and degenerate inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "bad_f0",
    [
        None,
        float("nan"),
        float("inf"),
        float("-inf"),
        0.0,
        -1.0,
        -1e-6,
        "not_a_number",
        {},
    ],
)
def test_invalid_f0_falls_back_to_80_hz(bad_f0: object) -> None:
    assert _upper(bad_f0) == 80.0


def test_numeric_string_f0_is_coerced_when_parseable() -> None:
    assert _upper("220") == 80.0


# ---------------------------------------------------------------------------
# 4. sr_hz / n_fft reserved parameters do not affect bound
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("sr_hz", "n_fft"),
    [
        (8000.0, 512),
        (48000.0, 16384),
        (96000.0, 32768),
        (44100.0, 0),
    ],
)
def test_sr_hz_and_n_fft_do_not_change_upper_bound(sr_hz: float, n_fft: int) -> None:
    f0 = 120.0
    baseline = _upper(f0)
    assert _upper(f0, sr_hz=sr_hz, n_fft=n_fft) == baseline
    assert baseline == 60.0


# ---------------------------------------------------------------------------
# 5. Compatibility with low_frequency_policy (minimal; no audit duplication)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f0_hz", [65.41, 110.0, 220.0, 440.0])
def test_low_frequency_policy_uses_subbass_policy_as_single_source_bound(
    f0_hz: float,
) -> None:
    policy_bound = _upper(f0_hz)
    out = calculate_adaptive_subfundamental_cutoff_hz(f0_hz)
    expected = min(policy_bound, f0_hz * float(out["max_fraction_of_f0"]))
    assert float(out["adaptive_subfundamental_cutoff_hz"]) == expected


# ---------------------------------------------------------------------------
# 6. Thesis-critical regression guards
# ---------------------------------------------------------------------------

def test_canonical_policy_values_do_not_drift_from_documented_formula() -> None:
    assert _upper(220.0) == 80.0
    assert _upper(110.0) == 55.0
    assert _upper(40.0) == 20.0


def test_low_register_harmonics_above_cutoff_remain_outside_subbass_bucket() -> None:
    """f0=110 Hz → cutoff 55 Hz; first harmonic at 110 Hz is not sub-bass."""
    cutoff = _upper(110.0)
    first_harmonic_hz = 110.0
    assert first_harmonic_hz >= cutoff


def test_subbass_policy_does_not_return_register_margin_percent() -> None:
    """Diagnostic margin tiers live in low_frequency_policy, not here."""
    value = _upper(30.0)
    assert value == 15.0
    assert isinstance(value, float)
