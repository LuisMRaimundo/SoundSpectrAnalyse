from __future__ import annotations

"""
Additional scientifically-motivated coverage for harmonic_validation.py.

Public API under test: ``validate_harmonic_series_matched`` — the audit-key
wrapper over ``harmonic_alignment.compute_harmonic_alignment_metrics``
(round(f/f0) order assignment, cents tolerance gate, strongest-energy collapse
per order).

Focus areas (no production code changes):
- degenerate inputs (None/empty table, non-finite or non-positive f0,
  missing frequency column) -> documented "invalid" payload;
- pure harmonic series -> "ok" with full slot coverage and ~0-cent deviations;
- cents-tolerance window: inside accepted, outside classified non-harmonic;
- status tier mapping (excellent/good -> ok, warning -> warning,
  failed -> invalid);
- harmonic ORDER counting vs literal peak counting; strongest-peak-wins
  collapse policy for competing peaks on one order;
- register invariance of the cents-based gate; amplitude-scaling invariance;
- sub-bass cutoff partition semantics;
- garbage candidate rows are excluded from classification;
- determinism and row-order invariance;
- the matched<=expected clamp in the slot-count alias helper.

Property-style and metamorphic assertions are preferred. Exact values are
asserted only where canonical: constructed cents offsets, slot counts implied
by floor(max_f/f0), and RMS/mean over known error sets.
"""

import math

import numpy as np
import pandas as pd
import pytest

import harmonic_validation as hv
from harmonic_validation import validate_harmonic_series_matched


def _cents_shift(freq_hz: float, cents: float) -> float:
    return float(freq_hz * 2.0 ** (cents / 1200.0))


def _peaks(freqs: list[float], amps: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": amps})


def _assert_invalid_payload(out: dict, *, error: str) -> None:
    assert out["harmonic_validation_status"] == "invalid"
    assert out["error"] == error
    assert out["is_valid"] is False
    assert out["n_peaks_in_pool"] == 0
    assert out["harmonic_slot_expected_count"] == 0
    assert out["harmonic_slot_matched_count"] == 0
    assert out["harmonic_slot_missing_count"] == 0
    assert out["validation_backend"] == "harmonic_order_alignment_cents_v2"


# ---------------------------------------------------------------------------
# 1. Degenerate / invalid inputs
# ---------------------------------------------------------------------------

def test_none_peak_table_is_invalid() -> None:
    out = validate_harmonic_series_matched(110.0, None, max_freq_hz=1000.0)
    _assert_invalid_payload(out, error="empty_peaks_or_invalid_f0")


def test_empty_peak_table_is_invalid() -> None:
    out = validate_harmonic_series_matched(110.0, pd.DataFrame(), max_freq_hz=1000.0)
    _assert_invalid_payload(out, error="empty_peaks_or_invalid_f0")


@pytest.mark.parametrize("bad_f0", [0.0, -100.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_f0_is_invalid(bad_f0: float) -> None:
    peaks = _peaks([110.0, 220.0], [1.0, 0.5])
    out = validate_harmonic_series_matched(bad_f0, peaks, max_freq_hz=1000.0)
    _assert_invalid_payload(out, error="empty_peaks_or_invalid_f0")


def test_missing_frequency_column_is_invalid() -> None:
    bad = pd.DataFrame({"frequency_hz": [110.0, 220.0], "Amplitude": [1.0, 0.5]})
    out = validate_harmonic_series_matched(110.0, bad, max_freq_hz=1000.0)
    _assert_invalid_payload(out, error="missing_Frequency_Hz_column")


def test_f0_above_frequency_ceiling_yields_invalid_with_zero_slots() -> None:
    # floor(max_f / f0) = 0 expected orders: nothing can be validated.
    peaks = _peaks([500.0], [1.0])
    out = validate_harmonic_series_matched(500.0, peaks, max_freq_hz=400.0)
    assert out["harmonic_validation_status"] == "invalid"
    assert out["is_valid"] is False
    assert out["harmonic_slot_expected_count"] == 0
    assert out["harmonic_slot_matched_count"] == 0
    assert out["harmonic_match_ratio"] == 0.0
    # No matches -> RMS deviation is the documented NaN placeholder.
    assert math.isnan(float(out["rms_harmonic_deviation_cents"]))


def test_non_finite_frequency_ceiling_falls_back_to_20khz() -> None:
    peaks = _peaks([110.0, 220.0, 330.0], [1.0, 0.7, 0.5])
    out = validate_harmonic_series_matched(
        110.0, peaks, max_freq_hz=float("inf")
    )
    # Documented fallback ceiling 20000 Hz: floor(20000/110) = 181 slots.
    assert out["harmonic_slot_expected_count"] == 181
    assert out["harmonic_slot_matched_count"] == 3


# ---------------------------------------------------------------------------
# 2. Pure harmonic series
# ---------------------------------------------------------------------------

def test_pure_harmonic_series_validates_ok_with_full_coverage() -> None:
    f0 = 110.0
    n_orders = 9  # floor(1000 / 110) = 9 expected slots
    freqs = [f0 * n for n in range(1, n_orders + 1)]
    amps = [1.0 / n for n in range(1, n_orders + 1)]
    out = validate_harmonic_series_matched(
        f0, _peaks(freqs, amps), max_freq_hz=1000.0
    )
    assert out["harmonic_validation_status"] == "ok"
    assert out["is_valid"] is True
    assert out["harmonic_slot_expected_count"] == n_orders
    assert out["harmonic_slot_matched_count"] == n_orders
    assert out["harmonic_slot_missing_count"] == 0
    assert out["harmonic_match_ratio"] == pytest.approx(1.0, abs=1e-12)
    assert out["missing_harmonic_count"] == 0
    assert out["non_harmonic_candidate_count"] == 0
    assert float(out["mean_abs_harmonic_deviation_cents"]) == pytest.approx(0.0, abs=1e-9)
    assert float(out["rms_harmonic_deviation_cents"]) == pytest.approx(0.0, abs=1e-9)
    assert out["n_peaks_in_pool"] == n_orders


# ---------------------------------------------------------------------------
# 3. Tolerance-window behaviour and the warning tier
# ---------------------------------------------------------------------------

def test_cents_window_accepts_inside_and_rejects_outside() -> None:
    # f0 = 110, ceiling 350 -> 3 expected orders. Fixed 30-cent tolerance.
    # Order 1 exact, order 2 detuned +20c (inside), order 3 detuned +50c
    # (outside every window -> non-harmonic candidate).
    f0 = 110.0
    freqs = [f0, _cents_shift(2 * f0, 20.0), _cents_shift(3 * f0, 50.0)]
    out = validate_harmonic_series_matched(
        f0,
        _peaks(freqs, [1.0, 0.8, 0.6]),
        max_freq_hz=350.0,
        match_tolerance_cents=30.0,
    )
    assert out["harmonic_slot_expected_count"] == 3
    assert out["harmonic_slot_matched_count"] == 2
    assert out["harmonic_slot_missing_count"] == 1
    assert out["non_harmonic_candidate_count"] == 1
    # 2/3 matched orders is below the good tier ratio (0.70) -> warning.
    assert out["harmonic_validation_status"] == "warning"
    assert out["is_valid"] is False
    # Canonical deviation statistics over the known error set {0, +20} cents.
    assert float(out["mean_abs_harmonic_deviation_cents"]) == pytest.approx(10.0, rel=1e-6)
    assert float(out["rms_harmonic_deviation_cents"]) == pytest.approx(
        math.sqrt((0.0**2 + 20.0**2) / 2.0), rel=1e-6
    )
    # Deprecated and clarified aliases must agree with the canonical count.
    for alias in (
        "inharmonic_candidate_count",
        "outside_harmonic_window_candidate_count",
        "outside_harmonic_window_peak_candidate_count",
        "unmatched_spectral_row_count",
    ):
        assert out[alias] == out["non_harmonic_candidate_count"], alias


# ---------------------------------------------------------------------------
# 4. Register / f0 scaling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f0", [55.0, 880.0])
def test_cents_gate_is_register_invariant_for_same_relative_detuning(f0: float) -> None:
    # Identical +12-cent detuning on orders 1-5 with a 6-slot window: the
    # cents-based gate must produce the same classification at any register.
    freqs = [_cents_shift(n * f0, 12.0) for n in range(1, 6)]
    out = validate_harmonic_series_matched(
        f0,
        _peaks(freqs, [1.0, 0.8, 0.6, 0.4, 0.2]),
        max_freq_hz=6.0 * f0,
        match_tolerance_cents=30.0,
    )
    assert out["harmonic_slot_expected_count"] == 6
    assert out["harmonic_slot_matched_count"] == 5
    # 5/6 = 0.833 with 12-cent mean error: good tier -> "ok".
    assert out["harmonic_validation_status"] == "ok"
    assert float(out["mean_abs_harmonic_deviation_cents"]) == pytest.approx(12.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. Duplicate / competing peaks and order semantics
# ---------------------------------------------------------------------------

def test_strongest_peak_wins_when_competing_for_one_order() -> None:
    # Two candidates inside the order-2 window: exact but weak (amp 0.2) vs
    # +10c but strong (amp 1.0). Documented collapse policy: strongest energy
    # wins, and the order is counted once.
    f0 = 110.0
    freqs = [f0, 2 * f0, _cents_shift(2 * f0, 10.0)]
    out = validate_harmonic_series_matched(
        f0,
        _peaks(freqs, [1.0, 0.2, 1.0]),
        max_freq_hz=250.0,
        match_tolerance_cents=30.0,
    )
    assert out["harmonic_slot_expected_count"] == 2
    # Three peaks, two harmonic orders: order count, not peak count.
    assert out["n_peaks_in_pool"] == 3
    assert out["harmonic_slot_matched_count"] == 2
    match_n2 = next(m for m in out["harmonic_matches"] if int(m["n"]) == 2)
    assert float(match_n2["observed_hz"]) == pytest.approx(
        _cents_shift(2 * f0, 10.0), rel=1e-9
    )
    assert float(match_n2["error_cents"]) == pytest.approx(10.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 6. Invalid candidate rows
# ---------------------------------------------------------------------------

def test_non_finite_and_non_positive_rows_do_not_affect_classification() -> None:
    f0 = 110.0
    clean = _peaks([110.0, 220.0, 330.0], [1.0, 0.7, 0.5])
    dirty = _peaks(
        [float("nan"), float("inf"), -50.0, 0.0, 110.0, 220.0, 330.0],
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.7, 0.5],
    )
    out_clean = validate_harmonic_series_matched(f0, clean, max_freq_hz=350.0)
    out_dirty = validate_harmonic_series_matched(f0, dirty, max_freq_hz=350.0)
    for key in (
        "harmonic_validation_status",
        "harmonic_slot_expected_count",
        "harmonic_slot_matched_count",
        "harmonic_match_ratio",
        "non_harmonic_candidate_count",
        "mean_abs_harmonic_deviation_cents",
    ):
        assert out_clean[key] == out_dirty[key], key
    # Current contract: n_peaks_in_pool reports raw table rows (pre-filter).
    assert out_clean["n_peaks_in_pool"] == 3
    assert out_dirty["n_peaks_in_pool"] == 7


# ---------------------------------------------------------------------------
# 7. Sub-bass cutoff partition
# ---------------------------------------------------------------------------

def test_subbass_cutoff_excludes_fundamental_from_matching() -> None:
    f0 = 110.0
    peaks = _peaks([110.0, 220.0, 330.0], [1.0, 0.7, 0.5])
    no_cutoff = validate_harmonic_series_matched(f0, peaks, max_freq_hz=350.0)
    with_cutoff = validate_harmonic_series_matched(
        f0, peaks, max_freq_hz=350.0, subbass_cutoff_hz=120.0
    )
    # Without cutoff: all three orders match -> ok.
    assert no_cutoff["harmonic_slot_matched_count"] == 3
    assert no_cutoff["harmonic_validation_status"] == "ok"
    # With a 120 Hz cutoff the 110 Hz fundamental is partitioned as subbass:
    # excluded from order matching, but NOT counted as non-harmonic.
    assert with_cutoff["harmonic_slot_matched_count"] == 2
    assert with_cutoff["subbass_candidate_count"] == 1
    assert with_cutoff["non_harmonic_candidate_count"] == 0
    assert with_cutoff["harmonic_slot_expected_count"] == 3


# ---------------------------------------------------------------------------
# 8. Scaling, ordering, determinism
# ---------------------------------------------------------------------------

def test_uniform_amplitude_scaling_preserves_classification() -> None:
    f0 = 110.0
    freqs = [f0, _cents_shift(2 * f0, 20.0), _cents_shift(3 * f0, 50.0)]
    base = _peaks(freqs, [1.0, 0.8, 0.6])
    scaled = base.copy()
    scaled["Amplitude"] = scaled["Amplitude"] * 1e3
    out_base = validate_harmonic_series_matched(
        f0, base, max_freq_hz=350.0, match_tolerance_cents=30.0
    )
    out_scaled = validate_harmonic_series_matched(
        f0, scaled, max_freq_hz=350.0, match_tolerance_cents=30.0
    )
    for key in (
        "harmonic_validation_status",
        "harmonic_slot_matched_count",
        "non_harmonic_candidate_count",
        "harmonic_match_ratio",
        "mean_abs_harmonic_deviation_cents",
        "non_harmonic_candidate_energy_ratio",
    ):
        vb, vs = out_base[key], out_scaled[key]
        if isinstance(vb, float):
            assert vb == pytest.approx(vs, rel=1e-9), key
        else:
            assert vb == vs, key


def test_row_order_is_irrelevant_for_validation() -> None:
    # Matching assigns each row by round(f/f0) independently and collapses by
    # energy, so the peak-table row order is not part of the contract.
    f0 = 110.0
    freqs = [f0, 2 * f0, _cents_shift(2 * f0, 10.0), 3 * f0, _cents_shift(5 * f0, 60.0)]
    amps = [1.0, 0.2, 0.9, 0.5, 0.3]
    base = _peaks(freqs, amps)
    permuted = base.iloc[[4, 1, 3, 0, 2]].reset_index(drop=True)
    out_a = validate_harmonic_series_matched(
        f0, base, max_freq_hz=600.0, match_tolerance_cents=30.0
    )
    out_b = validate_harmonic_series_matched(
        f0, permuted, max_freq_hz=600.0, match_tolerance_cents=30.0
    )
    for key in (
        "harmonic_validation_status",
        "harmonic_slot_matched_count",
        "harmonic_slot_expected_count",
        "non_harmonic_candidate_count",
        "mean_abs_harmonic_deviation_cents",
        "rms_harmonic_deviation_cents",
    ):
        assert out_a[key] == out_b[key], key
    # The collapsed winner per order is identical regardless of row order.
    obs_a = {int(m["n"]): float(m["observed_hz"]) for m in out_a["harmonic_matches"]}
    obs_b = {int(m["n"]): float(m["observed_hz"]) for m in out_b["harmonic_matches"]}
    assert obs_a == obs_b


def test_repeated_identical_calls_are_deterministic() -> None:
    f0 = 110.0
    peaks = _peaks([110.0, 220.0, 330.0], [1.0, 0.7, 0.5])
    a = validate_harmonic_series_matched(f0, peaks, max_freq_hz=350.0)
    b = validate_harmonic_series_matched(f0, peaks, max_freq_hz=350.0)
    for key, va in a.items():
        vb = b[key]
        if isinstance(va, float) and math.isnan(va):
            assert isinstance(vb, float) and math.isnan(vb), key
        elif isinstance(va, (int, float, str, bool)):
            assert va == vb, key


# ---------------------------------------------------------------------------
# 9. Slot-count alias clamp (helper-level; unreachable via the public API,
#    which inherits matched <= expected from the alignment invariant)
# ---------------------------------------------------------------------------

def test_slot_count_aliases_clamp_matched_to_expected() -> None:
    aliases = hv._slot_count_aliases(expected=3, matched=5)
    assert aliases["harmonic_slot_expected_count"] == 3
    assert aliases["harmonic_slot_matched_count"] == 3
    assert aliases["harmonic_slot_missing_count"] == 0


def test_slot_count_aliases_clamp_negative_inputs_to_zero() -> None:
    aliases = hv._slot_count_aliases(expected=-2, matched=-7)
    assert aliases["harmonic_slot_expected_count"] == 0
    assert aliases["harmonic_slot_matched_count"] == 0
    assert aliases["harmonic_slot_missing_count"] == 0
