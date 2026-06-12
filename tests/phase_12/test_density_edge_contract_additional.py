from __future__ import annotations

"""
Third Phase 12 edge-contract layer for density.py.

Complements ``test_density_core_additional.py`` and
``test_density_metric_contract_additional.py`` with narrowly targeted checks on
band-boundary semantics, validation helper branches, discrete-vs-fatness
separation, degenerate numeric paths, and H/I/S aggregation edges.

No production code changes. No audio, Excel, GUI, compile_metrics, or pipeline runs.
"""

import math
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from density import (
    aggregate_low_frequency_residual_peak_power,
    apply_density_metric,
    apply_density_metric_df,
    band_partial_metric_sum,
    calculate_harmonic_density,
    compute_discrete_spectral_metrics_bundle,
    compute_rolloff_compensated_harmonic_density,
    partial_metric_sums_h_i_s_total,
    validate_spectral_density_metric,
)


def _subbass_df(freqs: list[float], amps: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": amps})


# ---------------------------------------------------------------------------
# 1. Canonical fatness vs diagnostic / discrete separation
# ---------------------------------------------------------------------------

def test_discrete_weight_bypasses_rolloff_and_domination_in_apply_density_metric() -> None:
    amps = np.array([1.0, 0.5])
    discrete = apply_density_metric(
        amps,
        "d3",
        frequencies=np.array([1000.0, 20000.0]),
        fundamental_freq=100.0,
        account_for_spectral_rolloff=True,
        prevent_domination=True,
    )
    discrete_off = apply_density_metric(
        amps,
        "d3",
        account_for_spectral_rolloff=False,
        prevent_domination=False,
    )
    assert discrete == pytest.approx(discrete_off, rel=0.0, abs=0.0)
    assert discrete == pytest.approx(math.log1p(1.0) + math.log1p(0.5), rel=1e-12)


def test_rolloff_compensated_density_is_distinct_from_fatness_with_rolloff() -> None:
    f0 = 100.0
    orders = np.array([1.0, 2.0, 3.0])
    amps = orders ** -1.5
    freqs = orders * f0
    compensated = compute_rolloff_compensated_harmonic_density(amps, freqs, f0)
    fatness_with_rolloff = apply_density_metric(
        amps, "linear", frequencies=freqs, fundamental_freq=f0, account_for_spectral_rolloff=True
    )
    undecayed = apply_density_metric(
        amps, "linear", frequencies=freqs, fundamental_freq=f0, account_for_spectral_rolloff=False
    )
    assert compensated["rolloff_compensated_harmonic_density_status"] == "computed"
    assert fatness_with_rolloff == pytest.approx(3.0, rel=1e-5)
    assert compensated["rolloff_compensated_harmonic_density"] != pytest.approx(
        fatness_with_rolloff, rel=1e-3
    )
    assert undecayed < fatness_with_rolloff


def test_count_based_density_stays_below_fatness_for_sparse_high_register_partials() -> None:
    weak = np.array([0.01, 0.01, 0.01])
    count = calculate_harmonic_density(weak, include_amp_factor=False, max_expected_harmonics=50)
    fatness = apply_density_metric(weak, "linear")
    assert count == pytest.approx(0.06, abs=1e-12)
    assert fatness == pytest.approx(3.0, abs=1e-12)
    assert count < fatness


def test_discrete_bundle_keys_are_diagnostic_not_fatness_aliases() -> None:
    bundle = compute_discrete_spectral_metrics_bundle([1.0, 0.5])
    assert set(bundle.keys()) == {
        "discrete_metric_d3",
        "discrete_metric_d10",
        "discrete_metric_d17",
        "discrete_metric_d24",
    }
    assert "apply_density_metric" not in bundle
    assert all(np.isfinite(v) for v in bundle.values())


# ---------------------------------------------------------------------------
# 2. H / I / S aggregation edges
# ---------------------------------------------------------------------------

def test_d10_total_uses_global_concatenated_metric_not_band_sum() -> None:
    h, i, s, t = partial_metric_sums_h_i_s_total([1.0, 0.5], [0.3], [0.1], "d10")
    expected_total = band_partial_metric_sum(np.array([1.0, 0.5, 0.3, 0.1]), "d10")
    assert t == pytest.approx(expected_total, rel=1e-12)
    assert t != pytest.approx(h + i + s, rel=1e-3)


def test_partial_metric_sums_d2_and_d8_aliases_match_documented_targets() -> None:
    h_d2, _, _, t_d2 = partial_metric_sums_h_i_s_total([1.0, 2.0], [0.5], [], "d2")
    h_lin, _, _, t_lin = partial_metric_sums_h_i_s_total([1.0, 2.0], [0.5], [], "linear")
    assert (h_d2, t_d2) == (h_lin, t_lin)

    h_d8, _, _, t_d8 = partial_metric_sums_h_i_s_total([0.8], [0.2], [], "d8")
    h_d17, _, _, t_d17 = partial_metric_sums_h_i_s_total([0.8], [0.2], [], "d17")
    assert h_d8 == pytest.approx(h_d17, rel=1e-12)
    assert t_d8 == pytest.approx(t_d17, rel=1e-12)


def test_partial_metric_sums_subbass_band_contributes_to_total_linear_closure() -> None:
    h, i, s, t = partial_metric_sums_h_i_s_total([2.0], [0.5], [0.25], "sqrt")
    assert s == pytest.approx(math.sqrt(0.25), rel=1e-12)
    assert t == pytest.approx(h + i + s, abs=1e-12)


def test_partial_metric_sums_filters_nonfinite_entries_before_linear_collapse() -> None:
    h, i, s, t = partial_metric_sums_h_i_s_total(
        [1.0, float("nan"), float("inf"), -2.0],
        [],
        [],
        "linear",
    )
    assert (h, i, s, t) == (1.0, 0.0, 0.0, 1.0)


def test_partial_metric_sums_d24_global_gate_applies_to_subbass_band() -> None:
    h, i, s, t = partial_metric_sums_h_i_s_total([1.0], [], [0.005], "d24")
    assert s == 0.0
    assert t == pytest.approx(h, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Sub-bass band boundary semantics (density aggregator only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("freq", "included"),
    [
        (30.0, False),
        (30.01, True),
        (200.0, True),
        (200.01, False),
    ],
)
def test_low_frequency_aggregator_band_edges_are_half_open(freq: float, included: bool) -> None:
    power = aggregate_low_frequency_residual_peak_power(
        _subbass_df([freq], [1.0]),
        None,
        low_band_mode="sum_all_bins",
    )
    if included:
        assert power == pytest.approx(1.0, abs=1e-12)
    else:
        assert power == 0.0


def test_low_frequency_aggregator_accepts_magnitude_db_without_amplitude_column() -> None:
    df = pd.DataFrame({"Frequency (Hz)": [50.0], "Magnitude (dB)": [0.0]})
    assert aggregate_low_frequency_residual_peak_power(df, None, low_band_mode="sum_all_bins") == pytest.approx(
        1.0, abs=1e-12
    )


def test_band_partial_metric_sum_ignores_mismatched_frequency_vector_for_d24() -> None:
    amps = np.array([1.0, 1.0])
    matched = band_partial_metric_sum(amps, "d24", frequencies_hz=np.array([1000.0, 15000.0]))
    mismatched = band_partial_metric_sum(amps, "d24", frequencies_hz=np.array([1000.0]))
    assert matched == pytest.approx(math.log1p(1.0), rel=1e-12)
    assert mismatched == pytest.approx(math.log1p(1.0) + math.log1p(1.0), rel=1e-12)


# ---------------------------------------------------------------------------
# 4. apply_density_metric degenerate / normalization paths
# ---------------------------------------------------------------------------

def test_apply_density_metric_remove_noise_strips_tiny_partials() -> None:
    noisy = apply_density_metric(np.array([1.0, 1e-10, 1e-10]), "linear", remove_noise=True)
    clean = apply_density_metric(np.array([1.0]), "linear")
    assert noisy == pytest.approx(clean, rel=1e-12)


def test_apply_density_metric_normalize_divides_by_component_count() -> None:
    val = apply_density_metric(np.array([2.0, 4.0]), "linear", normalize=True)
    assert val == pytest.approx(0.75, abs=1e-12)


def test_apply_density_metric_df_empty_returns_zero_without_mutation() -> None:
    empty = pd.DataFrame()
    assert apply_density_metric_df(empty) == 0.0


def test_compute_discrete_bundle_with_mixed_nonfinite_still_returns_finite_metrics() -> None:
    bundle = compute_discrete_spectral_metrics_bundle([1.0, float("nan"), 0.5])
    assert all(np.isfinite(v) for v in bundle.values())
    assert bundle["discrete_metric_d3"] == pytest.approx(math.log1p(1.0) + math.log1p(0.5), rel=1e-12)


# ---------------------------------------------------------------------------
# 5. validate_spectral_density_metric branches
# ---------------------------------------------------------------------------

def test_validate_spectral_density_metric_rejects_positive_value_with_zero_energy() -> None:
    out = validate_spectral_density_metric(1.0, np.array([100.0]), np.array([0.0]))
    assert out["is_valid"] is False
    assert any("total energy = 0" in e for e in out["errors"])


def test_validate_spectral_density_metric_warns_on_high_energy_ratio() -> None:
    out = validate_spectral_density_metric(100.0, np.array([100.0]), np.array([1.0]))
    assert out["is_valid"] is True
    assert out["warnings"]
    assert any("ratio is high" in w for w in out["warnings"])


def test_validate_spectral_density_metric_expected_range_failure() -> None:
    out = validate_spectral_density_metric(
        5.0, np.array([]), np.array([]), expected_range=(0.0, 1.0)
    )
    assert out["is_valid"] is False
    assert out["physical_checks"]["in_range"] is False


def test_validate_spectral_density_metric_reference_tolerance_failure() -> None:
    out = validate_spectral_density_metric(
        10.0,
        np.array([100.0]),
        np.array([1.0]),
        reference_value=5.0,
        tolerance=0.1,
    )
    assert out["is_valid"] is False
    assert out["comparison_with_reference"]["within_tolerance"] is False


# ---------------------------------------------------------------------------
# 6. Determinism and non-mutation
# ---------------------------------------------------------------------------

def test_partial_metric_sums_is_deterministic_and_non_mutating() -> None:
    harm = [1.0, 2.0]
    inharm = [0.5]
    sub = [0.1]
    snap = (deepcopy(harm), deepcopy(inharm), deepcopy(sub))
    first = partial_metric_sums_h_i_s_total(harm, inharm, sub, "linear")
    second = partial_metric_sums_h_i_s_total(harm, inharm, sub, "linear")
    assert first == second
    assert (harm, inharm, sub) == snap


def test_low_frequency_aggregator_is_deterministic() -> None:
    df = _subbass_df([50.0, 60.0], [0.4, 1.0])
    a = aggregate_low_frequency_residual_peak_power(df, None)
    b = aggregate_low_frequency_residual_peak_power(df.copy(), None)
    assert a == b
