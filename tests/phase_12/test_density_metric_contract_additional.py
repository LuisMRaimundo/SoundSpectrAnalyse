from __future__ import annotations

"""
Metric-contract regression layer for density.py.

Complements tests/phase_12/test_density_core_additional.py with focused checks on
rolloff-compensated harmonic density, discrete spectral internals, H/I/S band
aggregation semantics, degenerate numeric inputs, validation helpers, and
thesis-critical distinctions (count-based vs fatness vs weighted sums).

No production code changes. No audio, Excel, GUI, or compile_metrics integration.
"""

import math
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import density as D
from density import (
    _apply_discrete_spectral_metrics,
    _spectral_neff_from_filtered_linear_amplitudes,
    apply_density_metric,
    apply_density_metric_df,
    band_partial_metric_sum,
    calculate_combined_density_metric,
    calculate_harmonic_density,
    compute_rolloff_compensated_harmonic_density,
    compute_subbass_protection_tolerance_hz,
    partial_metric_sums_h_i_s_total,
    validate_spectral_density_metric,
)


# ---------------------------------------------------------------------------
# 1. Rolloff-compensated harmonic density
# ---------------------------------------------------------------------------

def test_rolloff_compensated_ideal_decay_matches_component_count() -> None:
    f0 = 100.0
    orders = np.array([1.0, 2.0])
    amps = orders ** -1.5
    freqs = orders * f0
    out = compute_rolloff_compensated_harmonic_density(amps, freqs, f0, weight_function="linear")
    assert out["rolloff_compensated_harmonic_density_status"] == "computed"
    assert out["rolloff_compensated_harmonic_density_component_count"] == 2
    # Equal compensated contributions under ideal 1/n^1.5 decay -> sum of two unit weights.
    assert out["rolloff_compensated_harmonic_density"] == pytest.approx(2.0, rel=1e-6)


def test_rolloff_compensated_skip_statuses_for_invalid_inputs() -> None:
    assert (
        compute_rolloff_compensated_harmonic_density([], [], 100.0)[
            "rolloff_compensated_harmonic_density_status"
        ]
        == "skipped_no_harmonic_components"
    )
    assert (
        compute_rolloff_compensated_harmonic_density([1.0], [100.0], float("nan"))[
            "rolloff_compensated_harmonic_density_status"
        ]
        == "skipped_invalid_fundamental_frequency"
    )
    assert (
        compute_rolloff_compensated_harmonic_density([1.0], [100.0], 100.0, alpha=-1.0)[
            "rolloff_compensated_harmonic_density_status"
        ]
        == "skipped_invalid_alpha"
    )
    assert (
        compute_rolloff_compensated_harmonic_density(
            [1.0, 1.0], [100.0, 200.0], 100.0, harmonic_orders=np.array([1.0])
        )["rolloff_compensated_harmonic_density_status"]
        == "skipped_harmonic_orders_length_mismatch"
    )


def test_rolloff_compensated_is_scale_invariant_on_amplitudes() -> None:
    f0 = 220.0
    freqs = np.array([220.0, 440.0, 660.0])
    amps = np.array([0.4, 0.2, 0.1])
    base = compute_rolloff_compensated_harmonic_density(amps, freqs, f0)
    scaled = compute_rolloff_compensated_harmonic_density(amps * 50.0, freqs, f0)
    assert base["rolloff_compensated_harmonic_density_status"] == "computed"
    assert scaled["rolloff_compensated_harmonic_density"] == pytest.approx(
        base["rolloff_compensated_harmonic_density"], rel=1e-9
    )


# ---------------------------------------------------------------------------
# 2. Discrete spectral internals (_spectral_neff, _apply_discrete_spectral_metrics)
# ---------------------------------------------------------------------------

def test_spectral_neff_canonical_equal_amplitudes() -> None:
    assert _spectral_neff_from_filtered_linear_amplitudes(np.array([2.0, 2.0, 2.0])) == pytest.approx(
        3.0, abs=1e-12
    )
    assert _spectral_neff_from_filtered_linear_amplitudes(np.array([])) == 0.0
    assert _spectral_neff_from_filtered_linear_amplitudes(np.array([0.0])) == 0.0


def test_apply_discrete_spectral_metrics_degenerate_and_unknown_key() -> None:
    assert _apply_discrete_spectral_metrics("d3", []) == 0.0
    assert _apply_discrete_spectral_metrics("d3", [np.nan, np.inf, -1.0]) == 0.0
    assert _apply_discrete_spectral_metrics("bogus", [1.0, 2.0]) == 0.0
    v = np.array([1.0, 1.0])
    manual_d10 = float(np.sum(np.log1p(v)) * (2.0 / 2.0))
    assert _apply_discrete_spectral_metrics("d10", v) == pytest.approx(manual_d10, rel=1e-12)


def test_apply_discrete_d24_respects_global_amplitude_max_override() -> None:
    # Local max is 0.005 but global override is 1.0 -> 0.005 is below 1 % gate.
    v = np.array([0.005])
    assert _apply_discrete_spectral_metrics("d24", v) == pytest.approx(math.log1p(0.005), rel=1e-12)
    assert _apply_discrete_spectral_metrics(
        "d24", v, d24_amplitude_max_override=1.0
    ) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 3. H / I / S band aggregation contracts
# ---------------------------------------------------------------------------

def test_band_partial_metric_sum_aliases_and_nonfinite_filtering() -> None:
    assert band_partial_metric_sum([], "linear") == 0.0
    assert band_partial_metric_sum([np.nan, np.inf, -2.0], "linear") == 0.0
    # d8 -> d17 alias on a single partial.
    a = 0.8
    expected_d17 = math.log1p(a * a) * math.log1p(1.0)
    assert band_partial_metric_sum([a], "d8") == pytest.approx(expected_d17, rel=1e-12)
    # d2 -> linear alias.
    assert band_partial_metric_sum([1.0, 2.0], "d2") == pytest.approx(3.0, abs=1e-12)


def test_partial_metric_sums_log_weight_collapses_bands_then_adds() -> None:
    h, i, s, t = partial_metric_sums_h_i_s_total([1.0, 2.0], [0.5], [0.1], "log")
    assert h == pytest.approx(math.log1p(3.0), rel=1e-12)
    assert i == pytest.approx(math.log1p(0.5), rel=1e-12)
    assert s == pytest.approx(math.log1p(0.1), rel=1e-12)
    assert t == pytest.approx(h + i + s, abs=1e-12)


def test_partial_metric_sums_preserves_missing_band_as_zero() -> None:
    h, i, s, t = partial_metric_sums_h_i_s_total([2.0], [], [], "linear")
    assert (h, i, s, t) == (2.0, 0.0, 0.0, 2.0)


def test_partial_metric_sums_d3_keeps_per_row_band_vectors() -> None:
    h, i, s, t = partial_metric_sums_h_i_s_total([1.0, 0.5], [0.3], [], "d3")
    assert h == pytest.approx(math.log1p(1.0) + math.log1p(0.5), rel=1e-12)
    assert i == pytest.approx(math.log1p(0.3), rel=1e-12)
    assert s == 0.0
    assert t == pytest.approx(h + i + s, abs=1e-12)


def test_partial_metric_sums_does_not_mutate_input_lists() -> None:
    harm = [1.0, 2.0]
    inharm = [0.5]
    sub = [0.1]
    snap = (deepcopy(harm), deepcopy(inharm), deepcopy(sub))
    partial_metric_sums_h_i_s_total(harm, inharm, sub, "linear")
    assert (harm, inharm, sub) == snap


# ---------------------------------------------------------------------------
# 4. apply_density_metric degenerate / domination / normalization
# ---------------------------------------------------------------------------

def test_apply_density_metric_all_zero_and_single_element_normalize() -> None:
    assert apply_density_metric(np.array([0.0, 0.0]), "linear") == 0.0
    assert apply_density_metric(np.array([4.0]), "linear", normalize=True) == pytest.approx(4.0, abs=1e-12)


def test_apply_density_metric_prevent_domination_reduces_strong_partial_leverage() -> None:
    v = np.array([10.0, 1.0])
    raw_sum = apply_density_metric(v, "linear", prevent_domination=False)
    fair_sum = apply_density_metric(v, "linear", prevent_domination=True)
    assert raw_sum == pytest.approx(11.0, abs=1e-12)
    assert fair_sum == pytest.approx(1.1, abs=1e-12)
    assert fair_sum < raw_sum


def test_apply_density_metric_extreme_magnitude_finite() -> None:
    # Domination normalization makes scale-invariant aggregates for moderate values.
    moderate = apply_density_metric(np.array([1e-6, 1e-6]), "linear")
    huge = apply_density_metric(np.array([1e300, 1e300]), "linear")
    assert np.isfinite(moderate)
    assert np.isfinite(huge)
    assert moderate == pytest.approx(2.0, rel=1e-6)
    assert huge == pytest.approx(2.0, rel=1e-6)


# ---------------------------------------------------------------------------
# 5. apply_density_metric_df contracts
# ---------------------------------------------------------------------------

def test_apply_density_metric_df_d24_uses_frequency_column() -> None:
    df = pd.DataFrame({"Amplitude": [1.0, 0.005], "Frequency (Hz)": [1000.0, 15000.0]})
    out = apply_density_metric_df(df, weight_function="d24")
    assert out == pytest.approx(math.log1p(1.0), rel=1e-12)


def test_apply_density_metric_df_magnitude_db_path_copies_without_mutating_input() -> None:
    df = pd.DataFrame({"Magnitude (dB)": [0.0, 6.0206]})
    snapshot = df.copy()
    val = apply_density_metric_df(df, weight_function="linear")
    pd.testing.assert_frame_equal(df, snapshot)
    assert val == pytest.approx(
        apply_density_metric(np.array([1.0, 2.0]), "linear"), rel=1e-3
    )


# ---------------------------------------------------------------------------
# 6. Count-based vs fatness vs combined metrics
# ---------------------------------------------------------------------------

def test_count_based_density_does_not_reward_missing_high_harmonic_slots() -> None:
    # Only two detected partials against 50 expected slots -> low count ratio.
    count_based = calculate_harmonic_density(
        np.array([0.5, 0.5]), include_amp_factor=False, max_expected_harmonics=50
    )
    fatness_many = apply_density_metric(np.array([0.3] * 10), "linear")
    fatness_few = apply_density_metric(np.array([0.5, 0.5]), "linear")
    assert count_based == pytest.approx(0.04, abs=1e-12)
    assert fatness_many > fatness_few
    assert fatness_many > count_based


def test_harmonic_density_threshold_boundary_and_f0_derived_max() -> None:
    # One partial above -60 dB threshold; f0-derived max = floor(nyq/f0) = 2.
    val = calculate_harmonic_density(
        np.array([1e-13, 1.0]),
        threshold_db=-60.0,
        include_amp_factor=False,
        fundamental_freq=440.0,
        sr=1760.0,
    )
    assert val == pytest.approx(0.5, abs=1e-12)


def test_combined_density_zero_weights_and_nonnegative_log_combination() -> None:
    assert calculate_combined_density_metric(5.0, 1.0, alpha=0.0, beta=0.0) == pytest.approx(0.0, abs=1e-12)
    # Negative inputs are clamped at zero before log1p in preserve_dynamic_range mode.
    neg = calculate_combined_density_metric(-10.0, 2.0, alpha=0.8, beta=0.2)
    zero_h = calculate_combined_density_metric(0.0, 2.0, alpha=0.8, beta=0.2)
    assert neg == pytest.approx(zero_h, rel=1e-12)


# ---------------------------------------------------------------------------
# 7. Sub-bass protection tolerance and validation helper
# ---------------------------------------------------------------------------

def test_subbass_protection_tolerance_window_aware_and_floor() -> None:
    tol = compute_subbass_protection_tolerance_hz(44100.0, 4096)
    assert tol > 12.0
    assert tol == pytest.approx(max(12.0, 4.0 * (44100.0 / 4096.0)), rel=1e-9)
    assert compute_subbass_protection_tolerance_hz(float("nan"), 4096) == pytest.approx(12.0, abs=1e-12)
    assert compute_subbass_protection_tolerance_hz(44100.0, 0) == pytest.approx(12.0, abs=1e-12)


def test_validate_spectral_density_metric_physical_checks() -> None:
    ok = validate_spectral_density_metric(
        5.0, np.array([100.0, 200.0]), np.array([1.0, 1.0])
    )
    assert ok["is_valid"] is True
    assert ok["physical_checks"]["positive"] is True
    assert ok["physical_checks"]["finite"] is True

    bad = validate_spectral_density_metric(-1.0, np.array([100.0]), np.array([1.0]))
    assert bad["is_valid"] is False
    assert any("negative" in e.lower() for e in bad["errors"])

    ref = validate_spectral_density_metric(
        10.0,
        np.array([100.0]),
        np.array([1.0]),
        reference_value=10.0,
        tolerance=0.05,
    )
    assert ref["comparison_with_reference"]["within_tolerance"] is True


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------

def test_density_metric_contract_helpers_are_deterministic() -> None:
    freqs = np.array([100.0, 200.0])
    amps = np.array([0.5, 0.25])
    a = compute_rolloff_compensated_harmonic_density(amps, freqs, 100.0)
    b = compute_rolloff_compensated_harmonic_density(amps, freqs, 100.0)
    assert a == b
    h1, i1, s1, t1 = partial_metric_sums_h_i_s_total([1.0], [0.2], [0.1], "sqrt")
    h2, i2, s2, t2 = partial_metric_sums_h_i_s_total([1.0], [0.2], [0.1], "sqrt")
    assert (h1, i1, s1, t1) == (h2, i2, s2, t2)
