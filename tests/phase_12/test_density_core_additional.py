from __future__ import annotations

"""
Additional scientifically-motivated coverage for density.py.

Scope: density-metric semantics and mathematical invariants of the public
API — weighting functions, the canonical fatness metric
(``apply_density_metric``), discrete spectral metrics (D3/D10/D17/D24),
band partial sums (H/I/S/Total), participation-ratio effective densities,
harmonic occupancy / slot counting, residual log-frequency occupancy,
harmonic power mass / effective power density, the fixed-band low-frequency
residual aggregator (and its deprecated wrapper), non-harmonic residual row
identification (and its deprecated wrapper), the combined density metric,
spectral entropy, and the legacy count-based harmonic density.

No production code changes. Exact assertions are used only where the value
is analytically canonical (equal-power participation ratios, floor slot
counts, log1p identities, exact band sums); everything else uses tolerances
or property/metamorphic assertions (scaling invariance, additive
monotonicity, component closure, inside/outside band membership).
"""

import math

import numpy as np
import pandas as pd
import pytest

import density as D
from density import (
    WeightFunction,
    aggregate_low_frequency_residual_peak_power,
    aggregate_subbass_noise_peak_power,
    apply_density_metric,
    apply_density_metric_df,
    band_partial_metric_sum,
    calculate_combined_density_metric,
    calculate_harmonic_density,
    calculate_inharmonic_density,
    compute_discrete_spectral_metrics_bundle,
    compute_expected_harmonic_slot_count,
    compute_harmonic_effective_power_density,
    compute_harmonic_effective_power_mass,
    compute_harmonic_occupancy_ratio,
    compute_residual_log_frequency_occupancy,
    compute_spectral_entropy,
    effective_partial_density_from_powers,
    get_weight_function,
    identify_inharmonic_partials,
    identify_nonharmonic_residual_rows,
    partial_density_effective_components,
    partial_density_effective_components_bundle,
    partial_metric_sums_h_i_s_total,
)


# ---------------------------------------------------------------------------
# 1. Weight functions
# ---------------------------------------------------------------------------

def test_weight_functions_canonical_values_and_aliases() -> None:
    x = np.array([0.0, 1.0, 4.0])
    assert np.allclose(WeightFunction.linear(x), x)
    assert np.allclose(WeightFunction.squared(x), x**2)
    assert np.allclose(WeightFunction.sqrt(x), np.sqrt(x))
    assert np.allclose(WeightFunction.cubic(x), x**3)
    assert np.allclose(WeightFunction.logarithmic(x), np.log1p(x))
    assert np.allclose(WeightFunction.exponential(x), np.expm1(x))
    # cbrt preserves sign (odd root).
    assert WeightFunction.cbrt(-8.0) == pytest.approx(-2.0)
    assert WeightFunction.cbrt(27.0) == pytest.approx(3.0)
    # log weighting is zero-safe; inverse_log is finite at zero via epsilon.
    assert WeightFunction.logarithmic(0.0) == 0.0
    assert np.isfinite(WeightFunction.inverse_log(0.0))
    # Aliases resolve to the same callables / equivalent functions.
    assert get_weight_function("log") is get_weight_function("logarithmic")
    assert get_weight_function("exp") is get_weight_function("exponential")
    assert get_weight_function("sum") is get_weight_function("linear")
    assert get_weight_function("d2") is get_weight_function("linear")


def test_weight_function_ordering_for_amplitudes_above_one() -> None:
    # For x > 1: sqrt(x) < x < x^2 (canonical ordering).
    x = 3.0
    assert get_weight_function("sqrt")(x) < get_weight_function("linear")(x) < get_weight_function("squared")(x)


def test_unknown_weight_function_raises_value_error() -> None:
    with pytest.raises(ValueError):
        get_weight_function("not-a-weight")


# ---------------------------------------------------------------------------
# 2. apply_density_metric (canonical fatness metric)
# ---------------------------------------------------------------------------

def test_fatness_more_moderate_harmonics_outweigh_one_strong_partial() -> None:
    # Documented fatness contract: with max-normalisation, ten moderate
    # partials contribute 10.0 while a single strong partial contributes 1.0.
    many = apply_density_metric(np.array([0.3] * 10), "linear")
    one = apply_density_metric(np.array([1.0]), "linear")
    assert many == pytest.approx(10.0, abs=1e-12)
    assert one == pytest.approx(1.0, abs=1e-12)
    assert many > one


def test_rolloff_compensation_restores_ideal_decay_to_equal_weight() -> None:
    # Partials following the exact documented 1/n^1.5 rolloff are equalised:
    # the compensated sum approaches the number of partials.
    f0 = 100.0
    orders = np.array([1.0, 2.0])
    amps = orders**-1.5
    result = apply_density_metric(
        amps, "linear", frequencies=orders * f0, fundamental_freq=f0
    )
    assert result == pytest.approx(2.0, rel=1e-6)


def test_apply_density_metric_filters_non_finite_and_negative_values() -> None:
    clean = apply_density_metric(np.array([1.0, 1.0]), "linear")
    dirty = apply_density_metric(np.array([1.0, np.nan, np.inf, 1.0]), "linear")
    assert dirty == pytest.approx(clean, abs=1e-12)
    # Negative amplitudes are folded by absolute value (documented).
    neg = apply_density_metric(np.array([-1.0, 1.0]), "linear")
    assert neg == pytest.approx(clean, abs=1e-12)


def test_apply_density_metric_empty_and_noise_removal_and_normalize() -> None:
    assert apply_density_metric(np.array([]), "linear") == 0.0
    # remove_noise drops partials below 1e-6 of the maximum.
    assert apply_density_metric(
        np.array([1.0, 1e-9]), "linear", remove_noise=True
    ) == pytest.approx(1.0, abs=1e-12)
    # normalize=True divides the aggregate by the partial count.
    assert apply_density_metric(
        np.array([2.0, 2.0]), "linear", normalize=True
    ) == pytest.approx(1.0, abs=1e-12)


def test_apply_density_metric_discrete_route_matches_canonical_d3() -> None:
    v = np.array([0.2, 0.5, 1.5])
    # D3 = sum(log1p(A_i)) bypassing rolloff/domination handling.
    assert apply_density_metric(v, "d3") == pytest.approx(float(np.sum(np.log1p(v))), rel=1e-12)


def test_apply_density_metric_df_paths() -> None:
    assert apply_density_metric_df(pd.DataFrame()) == 0.0
    # Magnitude (dB) fallback: 0 dB == amplitude 1.0 (canonical conversion).
    df_db = pd.DataFrame({"Magnitude (dB)": [0.0, 0.0]})
    assert apply_density_metric_df(df_db, weight_function="linear") == pytest.approx(
        apply_density_metric(np.array([1.0, 1.0]), "linear"), rel=1e-12
    )
    with pytest.raises(ValueError):
        apply_density_metric_df(pd.DataFrame({"other": [1.0]}))


# ---------------------------------------------------------------------------
# 3. Discrete spectral metrics and band sums
# ---------------------------------------------------------------------------

def test_d10_equals_d3_for_equal_amplitudes() -> None:
    # Equal amplitudes -> N_eff = N -> the D10 efficiency factor is exactly 1.
    v = np.array([0.4] * 6)
    assert band_partial_metric_sum(v, "d10") == pytest.approx(
        band_partial_metric_sum(v, "d3"), rel=1e-9
    )


def test_d17_single_partial_canonical_value() -> None:
    # One partial: N_eff = 1 -> D17 = log1p(A^2) * log1p(1).
    a = 0.8
    expected = math.log1p(a * a) * math.log1p(1.0)
    assert band_partial_metric_sum(np.array([a]), "d17") == pytest.approx(expected, rel=1e-12)


def test_d24_one_percent_gate_and_12khz_band_limit() -> None:
    # Amplitude below 1 % of max is excluded.
    v = np.array([1.0, 0.005])
    assert band_partial_metric_sum(v, "d24") == pytest.approx(math.log1p(1.0), rel=1e-12)
    # Frequencies above 12 kHz are excluded when aligned frequencies exist.
    v2 = np.array([1.0, 1.0])
    f2 = np.array([1000.0, 15000.0])
    assert band_partial_metric_sum(v2, "d24", frequencies_hz=f2) == pytest.approx(
        math.log1p(1.0), rel=1e-12
    )


def test_discrete_bundle_empty_returns_nan_payload() -> None:
    bundle = compute_discrete_spectral_metrics_bundle([])
    assert set(bundle.keys()) == {
        "discrete_metric_d3",
        "discrete_metric_d10",
        "discrete_metric_d17",
        "discrete_metric_d24",
    }
    assert all(math.isnan(val) for val in bundle.values())


def test_partial_sums_linear_closure_and_empty_bands() -> None:
    h, i, s, t = partial_metric_sums_h_i_s_total([1.0, 2.0], [0.5], [], "linear")
    # Continuous weights: each band collapses to its plain sum; Total = H+I+S.
    assert (h, i, s) == (3.0, 0.5, 0.0)
    assert t == pytest.approx(h + i + s, abs=1e-12)
    assert partial_metric_sums_h_i_s_total([], [], [], "linear") == (0.0, 0.0, 0.0, 0.0)


def test_partial_sums_d24_uses_global_amplitude_gate_across_bands() -> None:
    # 0.005 in the inharmonic band is below 1 % of the GLOBAL max (1.0 in the
    # harmonic band), so it is gated out even though it is its band's max.
    h, i, s, t = partial_metric_sums_h_i_s_total([1.0], [0.005], [], "d24")
    assert h == pytest.approx(math.log1p(1.0), rel=1e-12)
    assert i == 0.0
    assert t == pytest.approx(h, rel=1e-12)


def test_partial_sums_d17_total_is_global_metric_not_band_sum() -> None:
    h, i, s, t = partial_metric_sums_h_i_s_total(
        [1.0, 0.5], [0.3], [0.1], "d17"
    )
    expected_total = band_partial_metric_sum(np.array([1.0, 0.5, 0.3, 0.1]), "d17")
    assert t == pytest.approx(expected_total, rel=1e-12)
    assert t != pytest.approx(h + i + s, rel=1e-3)


# ---------------------------------------------------------------------------
# 4. Participation-ratio effective densities
# ---------------------------------------------------------------------------

def test_effective_partial_density_canonical_and_scale_invariant() -> None:
    # n equal powers -> exactly n; one component -> exactly 1.
    assert effective_partial_density_from_powers(np.array([2.0] * 5)) == pytest.approx(5.0, abs=1e-12)
    assert effective_partial_density_from_powers(np.array([7.0])) == pytest.approx(1.0, abs=1e-12)
    base = effective_partial_density_from_powers(np.array([1.0, 0.5, 0.25]))
    scaled = effective_partial_density_from_powers(np.array([1.0, 0.5, 0.25]) * 1e3)
    assert base == pytest.approx(scaled, rel=1e-12)


def test_effective_partial_density_zero_fallbacks() -> None:
    for bad in (np.array([]), np.array([0.0, 0.0]), np.array([-1.0]), np.array([np.nan, np.inf])):
        assert effective_partial_density_from_powers(bad) == 0.0


def test_partial_density_bundle_components_and_modes() -> None:
    # Two equal strong harmonics, nothing else: D = 2 exactly.
    d_two, diag_two = partial_density_effective_components_bundle(np.array([1.0, 1.0]))
    assert d_two == pytest.approx(2.0, abs=1e-9)
    assert diag_two["partial_density_component_count_harmonic"] == 2
    # significant_peaks resolves inharmonic peaks individually -> higher D
    # than the single aggregate inharmonic bin.
    d_agg, _ = partial_density_effective_components_bundle(np.array([1.0]), np.array([0.5, 0.5]))
    d_sig, _ = partial_density_effective_components_bundle(
        np.array([1.0]), np.array([0.5, 0.5]), inharmonic_mode="significant_peaks"
    )
    assert d_agg == pytest.approx(1.8, rel=1e-9)   # bins [1, 0.5]: 2.25/1.25
    assert d_sig == pytest.approx(2.0, rel=1e-9)   # bins [1, .25, .25]: 2.25/1.125
    assert d_sig > d_agg
    # Unknown mode falls back to aggregate (documented).
    d_bogus, diag_bogus = partial_density_effective_components_bundle(
        np.array([1.0]), np.array([0.5, 0.5]), inharmonic_mode="bogus"
    )
    assert d_bogus == pytest.approx(d_agg, rel=1e-12)
    assert diag_bogus["partial_density_inharmonic_mode"] == "aggregate"
    # Ground-noise power adds one component bin.
    d_ground, _ = partial_density_effective_components_bundle(
        np.array([1.0]), ground_noise_power=1.0
    )
    assert d_ground == pytest.approx(2.0, abs=1e-9)
    # All-empty input: documented zero with zeroed diagnostics.
    d_zero, diag_zero = partial_density_effective_components_bundle()
    assert d_zero == 0.0
    assert diag_zero["partial_density_component_count_harmonic"] == 0
    # NaN / negative amplitudes are sanitised to zero contribution.
    d_dirty, _ = partial_density_effective_components_bundle(np.array([1.0, np.nan, -1.0]))
    assert d_dirty == pytest.approx(1.0, abs=1e-9)
    # Scalar wrapper agrees with the bundle.
    assert partial_density_effective_components(np.array([1.0, 1.0])) == pytest.approx(
        d_two, rel=1e-12
    )


# ---------------------------------------------------------------------------
# 5. Harmonic slots / occupancy / residual occupancy
# ---------------------------------------------------------------------------

def test_expected_harmonic_slot_count_floor_and_invalid_inputs() -> None:
    assert compute_expected_harmonic_slot_count(110.0, 1000.0) == 9
    assert compute_expected_harmonic_slot_count(110.0, 109.9) == 0
    for bad in ((float("nan"), 1000.0), (0.0, 1000.0), (-1.0, 1000.0), (110.0, float("nan")), ("abc", 1000.0), (None, 1000.0)):
        assert compute_expected_harmonic_slot_count(*bad) == 0  # type: ignore[arg-type]


def test_harmonic_occupancy_counts_unique_slots_not_rows() -> None:
    # 100 and 101 Hz both round to order 1: 4 rows, 3 unique slots of 5.
    df = pd.DataFrame({"Frequency (Hz)": [100.0, 200.0, 300.0, 101.0]})
    out = compute_harmonic_occupancy_ratio(df, f0_hz=100.0, max_frequency_hz=500.0)
    assert out["expected_harmonic_slot_count"] == 5
    assert out["detected_harmonic_slot_count"] == 3
    assert out["harmonic_occupancy_ratio"] == pytest.approx(0.6, abs=1e-12)
    assert out["harmonic_occupancy_status"] == "computed"


def test_harmonic_occupancy_gates_and_fallbacks() -> None:
    # include_for_density=False rows are excluded.
    df_inc = pd.DataFrame(
        {"Frequency (Hz)": [100.0, 200.0], "include_for_density": [True, False]}
    )
    assert (
        compute_harmonic_occupancy_ratio(df_inc, f0_hz=100.0, max_frequency_hz=500.0)[
            "detected_harmonic_slot_count"
        ]
        == 1
    )
    # SNR below threshold rows are excluded.
    df_snr = pd.DataFrame(
        {
            "Frequency (Hz)": [100.0, 200.0],
            "SNR_dB": [20.0, 1.0],
            "SNR Threshold (dB)": [6.0, 6.0],
        }
    )
    assert (
        compute_harmonic_occupancy_ratio(df_snr, f0_hz=100.0, max_frequency_hz=500.0)[
            "detected_harmonic_slot_count"
        ]
        == 1
    )
    # Valid slot grid but no rows: occupancy 0.0 with explicit status.
    empty = compute_harmonic_occupancy_ratio(
        pd.DataFrame(), f0_hz=100.0, max_frequency_hz=500.0
    )
    assert empty["harmonic_occupancy_ratio"] == 0.0
    assert empty["harmonic_occupancy_status"] == "no_harmonic_rows"
    # Invalid f0: NaN ratio with explicit status.
    invalid = compute_harmonic_occupancy_ratio(
        pd.DataFrame({"Frequency (Hz)": [100.0]}), f0_hz=float("nan"), max_frequency_hz=500.0
    )
    assert math.isnan(invalid["harmonic_occupancy_ratio"])
    assert invalid["harmonic_occupancy_status"] == "invalid_f0_or_ceiling"


def test_residual_log_frequency_occupancy_canonical_and_fallbacks() -> None:
    # One octave at 24 bins/octave: rows at 20 and 30 Hz occupy 2 of 24 bins.
    r = pd.DataFrame({"Frequency (Hz)": [20.0, 30.0]})
    out = compute_residual_log_frequency_occupancy(
        r, min_frequency_hz=20.0, max_frequency_hz=40.0, bins_per_octave=24
    )
    assert out["residual_log_frequency_bin_total"] == 24
    assert out["residual_log_frequency_bin_count"] == 2
    assert out["residual_log_frequency_occupancy"] == pytest.approx(2.0 / 24.0, rel=1e-12)
    # Adding a row in a new bin can only increase occupancy (monotonicity).
    r3 = pd.DataFrame({"Frequency (Hz)": [20.0, 30.0, 38.0]})
    out3 = compute_residual_log_frequency_occupancy(
        r3, min_frequency_hz=20.0, max_frequency_hz=40.0, bins_per_octave=24
    )
    assert out3["residual_log_frequency_occupancy"] > out["residual_log_frequency_occupancy"]
    # No data -> documented NaN/no_data payload.
    for bad in (None, pd.DataFrame(), pd.DataFrame({"other": [1.0]})):
        nd = compute_residual_log_frequency_occupancy(bad)
        assert nd["residual_log_frequency_occupancy_status"] == "no_data"
        assert math.isnan(nd["residual_log_frequency_occupancy"])
    # Rows exist but all outside the window -> occupancy 0.0, computed.
    outside = compute_residual_log_frequency_occupancy(
        pd.DataFrame({"Frequency (Hz)": [500.0]}),
        min_frequency_hz=20.0,
        max_frequency_hz=40.0,
    )
    assert outside["residual_log_frequency_occupancy"] == 0.0
    assert outside["residual_log_frequency_occupancy_status"] == "computed"


# ---------------------------------------------------------------------------
# 6. Harmonic power mass and effective power density
# ---------------------------------------------------------------------------

def test_harmonic_effective_power_mass_canonical_and_statuses() -> None:
    df = pd.DataFrame({"Amplitude": [1.0, 2.0]})
    out = compute_harmonic_effective_power_mass(df)
    assert out["harmonic_effective_power_mass"] == pytest.approx(5.0, abs=1e-12)
    assert out["harmonic_effective_power_mean"] == pytest.approx(2.5, abs=1e-12)
    assert out["harmonic_effective_power_rms"] == pytest.approx(math.sqrt(2.5), rel=1e-12)
    assert out["harmonic_effective_power_component_count"] == 2
    assert out["harmonic_effective_power_mass_status"] == "computed"

    assert compute_harmonic_effective_power_mass(None)[
        "harmonic_effective_power_mass_status"
    ] == "skipped_empty_harmonic_df"
    assert compute_harmonic_effective_power_mass(pd.DataFrame({"x": [1.0]}))[
        "harmonic_effective_power_mass_status"
    ] == "skipped_missing_Amplitude"
    assert compute_harmonic_effective_power_mass(pd.DataFrame({"Amplitude": [0.0, -1.0, np.nan]}))[
        "harmonic_effective_power_mass_status"
    ] == "skipped_no_positive_finite_amplitudes"


def test_harmonic_effective_power_density_canonical_and_invariance() -> None:
    # Equal amplitudes: every normalised power is 1 -> density = N exactly.
    out = compute_harmonic_effective_power_density(
        amplitudes=np.array([1.0, 1.0]),
        frequencies_hz=np.array([100.0, 200.0]),
        fundamental_freq_hz=100.0,
    )
    assert out["harmonic_effective_power_density_status"] == "computed"
    assert out["harmonic_effective_power_density"] == pytest.approx(2.0, abs=1e-12)
    assert out["harmonic_effective_power_density_component_count"] == 2
    assert out["harmonic_effective_power_density_normalized_by_harmonic_count"] == pytest.approx(1.0)
    # Expected slots = floor(200/100) = 2 -> normalised-by-slots = 1.
    assert out["harmonic_effective_power_density_normalized_by_expected_slots"] == pytest.approx(1.0)
    # Max-normalised formula -> scale invariant.
    scaled = compute_harmonic_effective_power_density(amplitudes=np.array([10.0, 10.0]))
    assert scaled["harmonic_effective_power_density"] == pytest.approx(2.0, abs=1e-12)


def test_harmonic_effective_power_density_skip_statuses() -> None:
    assert compute_harmonic_effective_power_density()[
        "harmonic_effective_power_density_status"
    ] == "skipped_no_valid_harmonic_rows"
    assert compute_harmonic_effective_power_density(pd.DataFrame({"x": [1.0]}))[
        "harmonic_effective_power_density_status"
    ] == "skipped_no_valid_amplitude_column"
    assert compute_harmonic_effective_power_density(amplitudes=np.array([0.0, -1.0]))[
        "harmonic_effective_power_density_status"
    ] == "skipped_no_valid_harmonic_rows"
    # Mismatched harmonic-order vector length is rejected, not broadcast.
    assert compute_harmonic_effective_power_density(
        amplitudes=np.array([1.0, 1.0]), harmonic_orders=np.array([1.0])
    )["harmonic_effective_power_density_status"] == "skipped_no_valid_harmonic_rows"


# ---------------------------------------------------------------------------
# 7. Fixed-band low-frequency residual aggregator
# ---------------------------------------------------------------------------

def test_low_frequency_aggregator_local_maxima_vs_sum_modes() -> None:
    comp = pd.DataFrame({"Frequency (Hz)": [40.0, 50.0, 60.0], "Amplitude": [0.2, 1.0, 0.3]})
    # local_maxima: only the strict local peak (50 Hz, amp 1.0) contributes.
    assert aggregate_low_frequency_residual_peak_power(comp, None) == pytest.approx(1.0, abs=1e-12)
    # sum_all_bins: all in-band powers contribute.
    assert aggregate_low_frequency_residual_peak_power(
        comp, None, low_band_mode="sum_all_bins"
    ) == pytest.approx(0.04 + 1.0 + 0.09, rel=1e-12)


def test_low_frequency_aggregator_harmonic_protection_and_band_edges() -> None:
    comp = pd.DataFrame({"Frequency (Hz)": [40.0, 50.0, 60.0], "Amplitude": [0.2, 1.0, 0.3]})
    # A harmonic template within the 12 Hz tolerance excludes the peaks.
    harm = pd.DataFrame({"Frequency (Hz)": [52.0]})
    assert aggregate_low_frequency_residual_peak_power(comp, harm) == 0.0
    # Band edges: f <= lower bound (30) excluded; f > subbass_hz (200) excluded.
    edges = pd.DataFrame({"Frequency (Hz)": [30.0, 250.0], "Amplitude": [1.0, 1.0]})
    assert aggregate_low_frequency_residual_peak_power(
        edges, None, low_band_mode="sum_all_bins"
    ) == 0.0
    # Degenerate inputs return the documented 0.0.
    assert aggregate_low_frequency_residual_peak_power(None, None) == 0.0
    assert aggregate_low_frequency_residual_peak_power(pd.DataFrame(), None) == 0.0
    assert aggregate_low_frequency_residual_peak_power(
        pd.DataFrame({"Frequency (Hz)": [50.0]}), None
    ) == 0.0  # no amplitude column


def test_deprecated_subbass_wrapper_agrees_with_canonical_aggregator() -> None:
    freqs = np.array([40.0, 50.0, 60.0])
    amps = np.array([0.2, 1.0, 0.3])
    # The wrapper's DeprecationWarning fires once per process; snapshot and
    # restore the once-only flag so this test does not consume the warning
    # asserted by tests/phase_6/test_density_wrapper_deprecation.py.
    flag_before = D._AGGREGATE_SUBBASS_WRAPPER_WARNED
    try:
        legacy = aggregate_subbass_noise_peak_power(freqs_hz=freqs, amplitudes=amps)
    finally:
        D._AGGREGATE_SUBBASS_WRAPPER_WARNED = flag_before
    canonical = aggregate_low_frequency_residual_peak_power(
        pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": amps}), None
    )
    assert legacy == pytest.approx(canonical, rel=1e-12)


# ---------------------------------------------------------------------------
# 8. Non-harmonic residual row identification
# ---------------------------------------------------------------------------

def test_residual_rows_relative_and_absolute_tolerance() -> None:
    harm = pd.DataFrame({"Frequency (Hz)": [100.0]})
    comp = pd.DataFrame({"Frequency (Hz)": [100.0, 101.0, 150.0]})
    # Relative 2 %: 101 Hz falls inside the +/-2 Hz window around 100.
    rel = identify_nonharmonic_residual_rows(harm, comp, 0.02, spectral_leakage_guard=False)
    assert rel["Frequency (Hz)"].tolist() == [150.0]
    # Absolute 5 Hz: 104 inside, 106 outside.
    abs_out = identify_nonharmonic_residual_rows(
        harm,
        pd.DataFrame({"Frequency (Hz)": [104.0, 106.0]}),
        5.0,
        spectral_leakage_guard=False,
    )
    assert abs_out["Frequency (Hz)"].tolist() == [106.0]


def test_residual_rows_leakage_guard_only_widens_exclusion() -> None:
    harm = pd.DataFrame({"Frequency (Hz)": [100.0]})
    comp = pd.DataFrame({"Frequency (Hz)": [100.0, 101.0, 103.0, 150.0]})
    without = identify_nonharmonic_residual_rows(
        harm, comp, 0.02, spectral_leakage_guard=False
    )
    with_guard = identify_nonharmonic_residual_rows(
        harm, comp, 0.02, sr=44100.0, n_fft=4096, spectral_leakage_guard=True
    )
    assert len(with_guard) <= len(without)
    assert set(with_guard["Frequency (Hz)"]).issubset(set(without["Frequency (Hz)"]))


def test_residual_rows_degenerate_inputs_and_deprecated_wrapper() -> None:
    comp = pd.DataFrame({"Frequency (Hz)": [100.0, 150.0]})
    empty = identify_nonharmonic_residual_rows(pd.DataFrame(), comp, 0.02)
    assert empty.empty
    with pytest.raises(ValueError):
        identify_nonharmonic_residual_rows(
            pd.DataFrame({"x": [1.0]}), comp, 0.02
        )
    harm = pd.DataFrame({"Frequency (Hz)": [100.0]})
    canonical = identify_nonharmonic_residual_rows(harm, comp, 0.02, spectral_leakage_guard=False)
    legacy = identify_inharmonic_partials(harm, comp, 0.02, spectral_leakage_guard=False)
    pd.testing.assert_frame_equal(canonical, legacy)


# ---------------------------------------------------------------------------
# 9. Combined density metric
# ---------------------------------------------------------------------------

def test_combined_density_identity_weight_normalization_and_monotonicity() -> None:
    # Equal component densities: the log combination is an exact identity.
    assert calculate_combined_density_metric(5.0, 5.0) == pytest.approx(5.0, rel=1e-12)
    # Un-normalised weights are rescaled to sum to 1.
    assert calculate_combined_density_metric(5.0, 1.0, alpha=8.0, beta=2.0) == pytest.approx(
        calculate_combined_density_metric(5.0, 1.0, alpha=0.8, beta=0.2), rel=1e-12
    )
    # Linear fallback is the exact weighted sum.
    assert calculate_combined_density_metric(
        5.0, 5.0, preserve_dynamic_range=False
    ) == pytest.approx(0.8 * 5.0 + 0.2 * 5.0, rel=1e-12)
    # Monotone in the harmonic component; dynamic range preserved (ff >> pp).
    assert calculate_combined_density_metric(10.0, 1.0) > calculate_combined_density_metric(5.0, 1.0)
    assert calculate_combined_density_metric(100.0, 10.0) > 10.0 * calculate_combined_density_metric(1.0, 0.1)


# ---------------------------------------------------------------------------
# 10. Spectral entropy and legacy count-based densities
# ---------------------------------------------------------------------------

def test_spectral_entropy_canonical_extremes() -> None:
    # Uniform distribution -> exactly 1; single component -> 0; empty -> 0.
    assert compute_spectral_entropy(np.array([2.0] * 8)) == pytest.approx(1.0, abs=1e-12)
    assert compute_spectral_entropy(np.array([3.0])) == 0.0
    assert compute_spectral_entropy(np.array([])) == 0.0
    assert compute_spectral_entropy(np.array([0.0, 0.0])) == 0.0
    # Concentration lowers entropy below the uniform maximum.
    assert compute_spectral_entropy(np.array([10.0, 0.1, 0.1, 0.1])) < 1.0


def test_count_based_harmonic_density_canonical_ratio_and_clip() -> None:
    # 5 significant partials of 10 expected -> 0.5 exactly (no amp factor).
    val = calculate_harmonic_density(
        np.array([0.5] * 5), include_amp_factor=False, max_expected_harmonics=10
    )
    assert val == pytest.approx(0.5, abs=1e-12)
    # Clipped to [0, 1] even when count exceeds the expected maximum.
    over = calculate_harmonic_density(
        np.array([0.5] * 30), include_amp_factor=False, max_expected_harmonics=10
    )
    assert over == 1.0
    assert calculate_harmonic_density(np.array([])) == 0.0
    # Inharmonic variant delegates to the same counting rule.
    assert calculate_inharmonic_density(
        np.array([0.5] * 5), max_expected_partials=10
    ) == calculate_harmonic_density(
        np.array([0.5] * 5), max_expected_harmonics=10
    )


# ---------------------------------------------------------------------------
# 11. Determinism
# ---------------------------------------------------------------------------

def test_density_functions_are_deterministic() -> None:
    v = np.array([1.0, 0.6, 0.3, 0.1])
    f = np.array([100.0, 200.0, 300.0, 400.0])
    a = apply_density_metric(v, "log", frequencies=f, fundamental_freq=100.0)
    b = apply_density_metric(v, "log", frequencies=f, fundamental_freq=100.0)
    assert a == b
    d1 = partial_density_effective_components_bundle(v)[0]
    d2 = partial_density_effective_components_bundle(v)[0]
    assert d1 == d2
