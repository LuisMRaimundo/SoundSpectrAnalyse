from __future__ import annotations

"""
Additional scientifically-motivated coverage for harmonic_alignment.py.

Public API under test: ``compute_harmonic_alignment_metrics`` — round(f/f0)
harmonic-order assignment, cents tolerance gate (explicit or FFT-adaptive),
strongest-energy collapse per order, and the subbass / harmonic-region /
inharmonic candidate partition.

Focus areas (no production code changes):
- degenerate inputs (None, invalid f0, empty/columnless tables, unsupported
  input types, all-invalid rows, zero expected slots) -> documented failed
  payload with stable keys;
- pure harmonic series -> all orders matched at ~0 cents, "excellent";
- explicit cents window: inside accepted / outside inharmonic; tolerance
  provenance (`harmonic_alignment_tolerance_cents_used`);
- FFT-adaptive tolerance widens with coarse bins (documented
  max(18, 2*bin-halfwidth-cents) rule);
- strongest-energy collapse for competing peaks; order count is not raw
  peak count;
- inharmonic-candidate preview population and its documented 20-entry cap;
- register invariance of the cents gate;
- expected slots = min(max_harmonics, floor(max_f / f0)); above-ceiling
  peaks add no slots;
- min_peak_amplitude threshold on both DataFrame and list-of-dicts pools;
- list-of-dicts input equivalence with the DataFrame path;
- amplitude representation equivalence (Amplitude vs Magnitude (dB)) and
  documented Amplitude_linear precedence;
- energy-status tiers and the low-collapsed-energy diagnostic, which must
  never change the cents-based alignment status;
- weighted vs unweighted status divergence (back-compat primary status is
  the unweighted one);
- row-order invariance and determinism;
- the `_cents` non-positive-input guard (helper-level; unreachable through
  the public API, which pre-filters non-positive frequencies).

Property-style and metamorphic assertions are preferred. Exact values are
asserted only where canonical (constructed cents offsets, floor/cap slot
counts, zero-cent deviations, the documented preview cap).
"""

import math

import numpy as np
import pandas as pd
import pytest

import harmonic_alignment as hal
from harmonic_alignment import compute_harmonic_alignment_metrics


def _cents_shift(freq_hz: float, cents: float) -> float:
    return float(freq_hz * 2.0 ** (cents / 1200.0))


def _peaks(freqs: list[float], amps: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": amps})


_STABLE_KEYS = (
    "harmonic_alignment_status",
    "harmonic_order_alignment_status",
    "harmonic_order_alignment_weighted_status",
    "harmonic_alignment_matched_count",
    "harmonic_alignment_expected_count",
    "harmonic_alignment_coverage_ratio",
    "harmonic_alignment_mean_abs_error_cents",
    "harmonic_alignment_max_abs_error_cents",
    "harmonic_alignment_tolerance_cents_used",
    "non_harmonic_candidate_count",
    "non_harmonic_candidate_energy_ratio",
    "subbass_candidate_count",
    "harmonic_alignment_matches",
    "harmonic_alignment_inharmonic_candidates_preview",
    "harmonic_alignment_non_harmonic_candidates_preview",
    "harmonic_representative_energy_status",
)


def _assert_stable_keys(out: dict) -> None:
    for key in _STABLE_KEYS:
        assert key in out, f"missing stable key: {key}"


def _assert_failed_payload(out: dict) -> None:
    _assert_stable_keys(out)
    assert out["harmonic_alignment_status"] == "failed"
    assert out["harmonic_order_alignment_status"] == "failed"
    assert out["harmonic_alignment_matched_count"] == 0
    assert out["harmonic_alignment_expected_count"] == 0
    assert out["harmonic_alignment_coverage_ratio"] == 0.0
    assert out["harmonic_alignment_matches"] == []
    assert out["harmonic_alignment_inharmonic_candidates_preview"] == []


# ---------------------------------------------------------------------------
# 1. Degenerate / invalid inputs
# ---------------------------------------------------------------------------

def test_none_input_returns_failed_payload() -> None:
    out = compute_harmonic_alignment_metrics(110.0, None)
    _assert_failed_payload(out)


@pytest.mark.parametrize("bad_f0", [0.0, -10.0, float("nan"), float("inf")])
def test_invalid_f0_returns_failed_payload(bad_f0: float) -> None:
    out = compute_harmonic_alignment_metrics(bad_f0, _peaks([110.0], [1.0]))
    _assert_failed_payload(out)


def test_empty_table_and_missing_column_fail() -> None:
    _assert_failed_payload(compute_harmonic_alignment_metrics(110.0, pd.DataFrame()))
    no_freq = pd.DataFrame({"frequency_hz": [110.0], "Amplitude": [1.0]})
    _assert_failed_payload(compute_harmonic_alignment_metrics(110.0, no_freq))


def test_unsupported_input_type_fails() -> None:
    out = compute_harmonic_alignment_metrics(110.0, "not-a-table")  # type: ignore[arg-type]
    _assert_failed_payload(out)


def test_all_invalid_rows_yield_empty_pool_and_fail() -> None:
    bad = _peaks([float("nan"), float("inf"), -50.0, 0.0], [1.0, 1.0, 1.0, 1.0])
    _assert_failed_payload(compute_harmonic_alignment_metrics(110.0, bad))


def test_ceiling_below_f0_yields_zero_slots_and_fails() -> None:
    out = compute_harmonic_alignment_metrics(
        500.0, _peaks([500.0], [1.0]), max_frequency_hz=400.0
    )
    _assert_failed_payload(out)


# ---------------------------------------------------------------------------
# 2. Pure harmonic series
# ---------------------------------------------------------------------------

def test_pure_harmonic_series_matches_every_order_at_zero_cents() -> None:
    f0 = 110.0
    n_orders = 9  # floor(1000 / 110)
    peaks = _peaks(
        [f0 * n for n in range(1, n_orders + 1)],
        [1.0 / n for n in range(1, n_orders + 1)],
    )
    out = compute_harmonic_alignment_metrics(f0, peaks, max_frequency_hz=1000.0)
    _assert_stable_keys(out)
    assert out["harmonic_alignment_expected_count"] == n_orders
    assert out["harmonic_alignment_matched_count"] == n_orders
    assert out["harmonic_order_alignment_status"] == "excellent"
    assert out["harmonic_order_alignment_weighted_status"] == "excellent"
    assert float(out["harmonic_alignment_mean_abs_error_cents"]) == pytest.approx(0.0, abs=1e-9)
    assert float(out["harmonic_alignment_max_abs_error_cents"]) == pytest.approx(0.0, abs=1e-9)
    assert out["non_harmonic_candidate_count"] == 0
    assert float(out["collapsed_representative_energy_ratio"]) == pytest.approx(1.0, abs=1e-12)
    # One match entry per order, in order, with the documented fields.
    orders = [int(m["n"]) for m in out["harmonic_alignment_matches"]]
    assert orders == list(range(1, n_orders + 1))
    for m in out["harmonic_alignment_matches"]:
        for field in ("n", "observed_hz", "expected_hz", "error_cents", "abs_error_cents", "energy", "tolerance_cents"):
            assert field in m


# ---------------------------------------------------------------------------
# 3. Tolerance windows
# ---------------------------------------------------------------------------

def test_explicit_cents_window_inside_matches_outside_does_not() -> None:
    f0 = 110.0
    inside = _cents_shift(2 * f0, 20.0)
    outside = _cents_shift(3 * f0, 40.0)
    out = compute_harmonic_alignment_metrics(
        f0,
        _peaks([f0, inside, outside], [1.0, 0.8, 0.6]),
        max_frequency_hz=350.0,
        tolerance_cents=25.0,
    )
    assert out["harmonic_alignment_matched_count"] == 2
    assert out["non_harmonic_candidate_count"] == 1
    # Tolerance provenance: explicit tolerance is reported verbatim.
    assert float(out["harmonic_alignment_tolerance_cents_used"]) == 25.0
    preview = out["harmonic_alignment_inharmonic_candidates_preview"]
    assert len(preview) == 1
    assert float(preview[0]["f_hz"]) == pytest.approx(outside, rel=1e-12)


def test_adaptive_tolerance_widens_with_coarse_fft_bins() -> None:
    # Documented adaptive rule: tol = max(18, 2 * cents-halfwidth of one FFT
    # bin at the expected frequency). A +30c fundamental fails an 18-cent
    # explicit window but must match under a coarse-bin adaptive window
    # (44100/256 = 172 Hz bins -> hundreds of cents at 110 Hz).
    f0 = 110.0
    detuned = _cents_shift(f0, 30.0)
    strict = compute_harmonic_alignment_metrics(
        f0, _peaks([detuned], [1.0]), max_frequency_hz=350.0, tolerance_cents=18.0
    )
    adaptive = compute_harmonic_alignment_metrics(
        f0,
        _peaks([detuned], [1.0]),
        max_frequency_hz=350.0,
        sample_rate=44100.0,
        n_fft=256,
    )
    assert strict["harmonic_alignment_matched_count"] == 0
    assert adaptive["harmonic_alignment_matched_count"] == 1
    assert float(adaptive["harmonic_alignment_tolerance_cents_used"]) > 18.0


def test_adaptive_tolerance_defaults_to_18_cents_without_fft_info() -> None:
    # No sample_rate/n_fft -> documented 18-cent default window: +15c in, +21c out.
    f0 = 110.0
    out = compute_harmonic_alignment_metrics(
        f0,
        _peaks([_cents_shift(f0, 15.0), _cents_shift(2 * f0, 21.0)], [1.0, 0.8]),
        max_frequency_hz=350.0,
    )
    assert out["harmonic_alignment_matched_count"] == 1
    assert out["non_harmonic_candidate_count"] == 1
    assert float(out["harmonic_alignment_tolerance_cents_used"]) == pytest.approx(18.0)


# ---------------------------------------------------------------------------
# 4/5. Competing peaks, collapse policy, order-count semantics
# ---------------------------------------------------------------------------

def test_strongest_energy_wins_collapse_and_orders_are_not_peak_counts() -> None:
    f0 = 110.0
    exact = 2 * f0
    detuned_strong = _cents_shift(2 * f0, 10.0)
    out = compute_harmonic_alignment_metrics(
        f0,
        _peaks([f0, exact, detuned_strong], [1.0, 0.2, 1.0]),
        max_frequency_hz=250.0,
        tolerance_cents=30.0,
    )
    # Three peaks, two unique harmonic orders.
    assert out["harmonic_alignment_expected_count"] == 2
    assert out["harmonic_alignment_matched_count"] == 2
    assert out["harmonic_representative_count"] == 2
    # All three peaks lie inside harmonic windows (region), none inharmonic.
    assert out["harmonic_region_candidate_count"] == 3
    assert out["non_harmonic_candidate_count"] == 0
    # Documented collapse policy: strongest energy wins, not nearest.
    match_n2 = next(m for m in out["harmonic_alignment_matches"] if int(m["n"]) == 2)
    assert float(match_n2["observed_hz"]) == pytest.approx(detuned_strong, rel=1e-12)
    assert float(match_n2["error_cents"]) == pytest.approx(10.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 6. Inharmonic candidate preview
# ---------------------------------------------------------------------------

def test_inharmonic_preview_contains_unmatched_peaks_and_caps_at_20() -> None:
    # 25 half-integer-order peaks: all outside every harmonic window.
    f0 = 110.0
    inharmonic_freqs = [(n + 0.5) * f0 for n in range(1, 26)]
    out = compute_harmonic_alignment_metrics(
        f0,
        _peaks(inharmonic_freqs, [1.0] * 25),
        max_frequency_hz=1000.0,
        tolerance_cents=18.0,
    )
    assert out["non_harmonic_candidate_count"] == 25
    # No matched orders at all -> failed status with populated diagnostics.
    assert out["harmonic_alignment_matched_count"] == 0
    assert out["harmonic_order_alignment_status"] == "failed"
    assert math.isnan(float(out["harmonic_alignment_mean_abs_error_cents"]))
    # Documented preview cap: 20 entries with f_hz/energy fields.
    preview = out["harmonic_alignment_inharmonic_candidates_preview"]
    assert len(preview) == 20
    input_set = {round(f, 6) for f in inharmonic_freqs}
    for entry in preview:
        assert round(float(entry["f_hz"]), 6) in input_set
        assert float(entry["energy"]) >= 0.0
    # Both preview aliases expose the same content.
    assert out["harmonic_alignment_non_harmonic_candidates_preview"] == preview


# ---------------------------------------------------------------------------
# 7. Register invariance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("f0", [55.0, 880.0])
def test_cents_gate_is_register_invariant(f0: float) -> None:
    freqs = [_cents_shift(n * f0, 12.0) for n in range(1, 6)]
    out = compute_harmonic_alignment_metrics(
        f0,
        _peaks(freqs, [1.0, 0.8, 0.6, 0.4, 0.2]),
        max_frequency_hz=6.0 * f0,
        tolerance_cents=30.0,
    )
    assert out["harmonic_alignment_expected_count"] == 6
    assert out["harmonic_alignment_matched_count"] == 5
    assert float(out["harmonic_alignment_mean_abs_error_cents"]) == pytest.approx(12.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 8. Frequency ceiling semantics
# ---------------------------------------------------------------------------

def test_expected_slots_follow_floor_and_max_harmonics_cap() -> None:
    f0 = 110.0
    peaks = _peaks([110.0, 220.0, 5000.0], [1.0, 0.7, 0.5])
    uncapped = compute_harmonic_alignment_metrics(f0, peaks, max_frequency_hz=1000.0)
    capped = compute_harmonic_alignment_metrics(
        f0, peaks, max_frequency_hz=1000.0, max_harmonics=5
    )
    # floor(1000 / 110) = 9 slots; the cap reduces it to 5.
    assert uncapped["harmonic_alignment_expected_count"] == 9
    assert capped["harmonic_alignment_expected_count"] == 5
    # The 5 kHz peak is above the ceiling: it adds no slots and is classified
    # as a non-harmonic candidate in both runs.
    for out in (uncapped, capped):
        assert out["harmonic_alignment_matched_count"] == 2
        assert out["non_harmonic_candidate_count"] == 1


# ---------------------------------------------------------------------------
# 11. Threshold semantics
# ---------------------------------------------------------------------------

def test_min_peak_amplitude_excludes_weak_peaks_from_both_pools() -> None:
    f0 = 110.0
    df = _peaks([110.0, 220.0], [1.0, 0.05])
    no_thr = compute_harmonic_alignment_metrics(f0, df, max_frequency_hz=350.0)
    with_thr = compute_harmonic_alignment_metrics(
        f0, df, max_frequency_hz=350.0, min_peak_amplitude=0.1
    )
    assert no_thr["harmonic_alignment_matched_count"] == 2
    assert with_thr["harmonic_alignment_matched_count"] == 1
    # The excluded peak vanishes from the pool entirely (not non-harmonic).
    assert with_thr["non_harmonic_candidate_count"] == 0

    as_list = [
        {"f_hz": 110.0, "amplitude_linear": 1.0},
        {"f_hz": 220.0, "amplitude_linear": 0.05},
    ]
    with_thr_list = compute_harmonic_alignment_metrics(
        f0, as_list, max_frequency_hz=350.0, min_peak_amplitude=0.1
    )
    assert with_thr_list["harmonic_alignment_matched_count"] == 1


# ---------------------------------------------------------------------------
# List-of-dicts input path
# ---------------------------------------------------------------------------

def test_list_of_dicts_input_is_equivalent_to_dataframe() -> None:
    f0 = 110.0
    df = _peaks([110.0, 220.0, 330.0], [1.0, 0.5, 0.25])
    as_list = [
        {"f_hz": 110.0, "amplitude_linear": 1.0},
        {"f_hz": 220.0, "amp": 0.5},  # documented fallback amplitude key
        "not-a-dict",  # skipped
        {"f_hz": float("nan"), "amplitude_linear": 1.0},  # skipped
        {"f_hz": -10.0, "amplitude_linear": 1.0},  # skipped
        {"f_hz": 330.0, "amplitude_linear": 0.25},
    ]
    out_df = compute_harmonic_alignment_metrics(f0, df, max_frequency_hz=350.0)
    out_list = compute_harmonic_alignment_metrics(f0, as_list, max_frequency_hz=350.0)
    for key in (
        "harmonic_alignment_matched_count",
        "harmonic_alignment_expected_count",
        "non_harmonic_candidate_count",
        "harmonic_order_alignment_status",
    ):
        assert out_df[key] == out_list[key], key
    assert float(out_df["harmonic_representative_energy"]) == pytest.approx(
        float(out_list["harmonic_representative_energy"]), rel=1e-12
    )


# ---------------------------------------------------------------------------
# Amplitude representations
# ---------------------------------------------------------------------------

def test_db_and_linear_amplitude_representations_are_equivalent() -> None:
    f0 = 110.0
    freqs = [110.0, _cents_shift(220.0, 10.0)]
    amps = [1.0, 0.5]
    dbs = [20.0 * math.log10(a) for a in amps]

    via_amp = compute_harmonic_alignment_metrics(
        f0, _peaks(freqs, amps), max_frequency_hz=350.0, tolerance_cents=30.0
    )
    via_db = compute_harmonic_alignment_metrics(
        f0,
        pd.DataFrame({"Frequency (Hz)": freqs, "Magnitude (dB)": dbs}),
        max_frequency_hz=350.0,
        tolerance_cents=30.0,
    )
    via_db_us = compute_harmonic_alignment_metrics(
        f0,
        pd.DataFrame({"Frequency (Hz)": freqs, "Magnitude_dB": dbs}),
        max_frequency_hz=350.0,
        tolerance_cents=30.0,
    )
    for out in (via_db, via_db_us):
        assert out["harmonic_alignment_matched_count"] == via_amp["harmonic_alignment_matched_count"]
        assert float(out["harmonic_representative_energy"]) == pytest.approx(
            float(via_amp["harmonic_representative_energy"]), rel=1e-9
        )


def test_amplitude_linear_column_takes_precedence_over_amplitude() -> None:
    # Two candidates for order 1; "Amplitude" favours the exact peak while
    # "Amplitude_linear" favours the detuned one. The documented column
    # priority (Amplitude_linear first) must decide the collapse winner.
    f0 = 110.0
    detuned = _cents_shift(f0, 10.0)
    df = pd.DataFrame(
        {
            "Frequency (Hz)": [f0, detuned],
            "Amplitude": [1.0, 0.5],
            "Amplitude_linear": [0.5, 1.0],
        }
    )
    out = compute_harmonic_alignment_metrics(
        f0, df, max_frequency_hz=150.0, tolerance_cents=30.0
    )
    match_n1 = next(m for m in out["harmonic_alignment_matches"] if int(m["n"]) == 1)
    assert float(match_n1["observed_hz"]) == pytest.approx(detuned, rel=1e-12)


# ---------------------------------------------------------------------------
# 12. Energy diagnostics never change the cents-based alignment status
# ---------------------------------------------------------------------------

def test_energy_status_acceptable_tier_with_low_collapsed_share_diagnostic() -> None:
    # 3 exact harmonics (energy 3) + strong off-harmonic content (energy 7):
    # r_rep = 0.3 -> "acceptable"; alignment stays "excellent" (cents+orders
    # only) and the low-collapsed-energy diagnostic fires (r_rep < 0.5).
    f0 = 110.0
    out = compute_harmonic_alignment_metrics(
        f0,
        _peaks([110.0, 220.0, 330.0, 505.0], [1.0, 1.0, 1.0, math.sqrt(7.0)]),
        max_frequency_hz=350.0,
        tolerance_cents=18.0,
    )
    assert out["harmonic_order_alignment_status"] == "excellent"
    assert out["harmonic_representative_energy_status"] == "acceptable"
    assert float(out["collapsed_representative_energy_ratio"]) == pytest.approx(0.3, rel=1e-9)
    assert bool(out["harmonic_alignment_low_collapsed_energy_diagnostic"]) is True
    assert out["harmonic_alignment_energy_diagnostic_message"] is not None


def test_energy_status_warning_tier_does_not_downgrade_alignment() -> None:
    # Harmonic representatives carry ~2.6% of total energy -> energy tier
    # "warning", while the cents-based alignment status remains "excellent".
    f0 = 110.0
    out = compute_harmonic_alignment_metrics(
        f0,
        _peaks([110.0, 220.0, 330.0, 505.0], [0.3, 0.3, 0.3, math.sqrt(10.0)]),
        max_frequency_hz=350.0,
        tolerance_cents=18.0,
    )
    assert out["harmonic_order_alignment_status"] == "excellent"
    assert out["harmonic_representative_energy_status"] == "warning"
    assert float(out["collapsed_representative_energy_ratio"]) < 0.22


def test_weighted_status_can_diverge_while_unweighted_stays_primary() -> None:
    # Errors {0, 0, +30c} with the 30c peak carrying dominant energy:
    # unweighted mean = 10c (good tier, p95 above excellent gate), while the
    # energy-weighted mean ~29c exceeds the good gate -> weighted "warning".
    # Back-compat primary status must follow the unweighted tier.
    f0 = 110.0
    freqs = [110.0, 220.0, _cents_shift(330.0, 30.0)]
    amps = [math.sqrt(0.1), math.sqrt(0.1), math.sqrt(10.0)]
    out = compute_harmonic_alignment_metrics(
        f0, _peaks(freqs, amps), max_frequency_hz=350.0, tolerance_cents=40.0
    )
    assert out["harmonic_order_alignment_status"] == "good"
    assert out["harmonic_order_alignment_weighted_status"] == "warning"
    assert out["harmonic_alignment_status"] == out["harmonic_order_alignment_status"]
    assert float(out["harmonic_order_alignment_weighted_mean_abs_error_cents"]) > 18.0


# ---------------------------------------------------------------------------
# 10. Ordering and determinism
# ---------------------------------------------------------------------------

def test_row_order_is_irrelevant_and_calls_are_deterministic() -> None:
    f0 = 110.0
    freqs = [110.0, 220.0, _cents_shift(220.0, 10.0), 330.0, (4.5) * f0]
    amps = [1.0, 0.2, 0.9, 0.5, 0.3]
    base = _peaks(freqs, amps)
    permuted = base.iloc[[4, 1, 3, 0, 2]].reset_index(drop=True)
    out_a = compute_harmonic_alignment_metrics(
        f0, base, max_frequency_hz=600.0, tolerance_cents=30.0
    )
    out_b = compute_harmonic_alignment_metrics(
        f0, permuted, max_frequency_hz=600.0, tolerance_cents=30.0
    )
    out_c = compute_harmonic_alignment_metrics(
        f0, base, max_frequency_hz=600.0, tolerance_cents=30.0
    )
    for key in (
        "harmonic_alignment_matched_count",
        "harmonic_alignment_expected_count",
        "non_harmonic_candidate_count",
        "harmonic_order_alignment_status",
        "harmonic_alignment_mean_abs_error_cents",
        "harmonic_representative_energy",
    ):
        assert out_a[key] == out_b[key], key
        assert out_a[key] == out_c[key], key
    obs_a = {int(m["n"]): float(m["observed_hz"]) for m in out_a["harmonic_alignment_matches"]}
    obs_b = {int(m["n"]): float(m["observed_hz"]) for m in out_b["harmonic_alignment_matches"]}
    assert obs_a == obs_b


def test_uniform_amplitude_scaling_preserves_matching_and_ratios() -> None:
    f0 = 110.0
    freqs = [110.0, _cents_shift(220.0, 20.0), 505.0]
    base = _peaks(freqs, [1.0, 0.8, 0.6])
    scaled = base.copy()
    scaled["Amplitude"] = scaled["Amplitude"] * 1e3
    out_base = compute_harmonic_alignment_metrics(
        f0, base, max_frequency_hz=350.0, tolerance_cents=30.0
    )
    out_scaled = compute_harmonic_alignment_metrics(
        f0, scaled, max_frequency_hz=350.0, tolerance_cents=30.0
    )
    for key in (
        "harmonic_alignment_matched_count",
        "non_harmonic_candidate_count",
        "harmonic_order_alignment_status",
    ):
        assert out_base[key] == out_scaled[key], key
    for key in (
        "collapsed_representative_energy_ratio",
        "non_harmonic_candidate_energy_ratio",
        "harmonic_alignment_mean_abs_error_cents",
    ):
        assert float(out_base[key]) == pytest.approx(float(out_scaled[key]), rel=1e-9), key


# ---------------------------------------------------------------------------
# Helper-level guard (unreachable via public API, which filters f <= 0)
# ---------------------------------------------------------------------------

def test_cents_helper_guards_non_positive_inputs() -> None:
    assert math.isnan(hal._cents(0.0, 110.0))
    assert math.isnan(hal._cents(110.0, 0.0))
    assert math.isnan(hal._cents(-1.0, 110.0))
    # Canonical: one octave = 1200 cents.
    assert hal._cents(220.0, 110.0) == pytest.approx(1200.0, rel=1e-12)
