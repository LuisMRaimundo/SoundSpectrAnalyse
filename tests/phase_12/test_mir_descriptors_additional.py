from __future__ import annotations

"""
Helper-level contract tests for mir_descriptors.py.

Protects peak-spectrum MIR descriptor calculation, numeric robustness,
non-finite filtering, deterministic outputs, and stable result shapes.
No production code changes. No audio, GUI, plotting, or pipeline runs.
"""

import math

import numpy as np
import pytest

from mir_descriptors import (
    _erb_rate_hz,
    _roughness_aures_1985,
    _safe_prob,
    compute_mir_descriptors_from_spectrum,
)


EXPECTED_KEYS = (
    "spectral_centroid_hz",
    "spectral_spread_hz",
    "spectral_skewness",
    "spectral_kurtosis",
    "spectral_irregularity",
    "tristimulus_1_fundamental",
    "tristimulus_2_low_harmonics_2_to_4",
    "tristimulus_3_high_harmonics_5_plus",
    "spectral_flatness",
    "spectral_rolloff_hz_85",
    "spectral_rolloff_hz_95",
    "roughness_aures_1985",
    "erb_weighted_spectral_density",
)


def _assert_all_nan(desc: dict[str, float]) -> None:
    assert set(desc.keys()) == set(EXPECTED_KEYS)
    for key in EXPECTED_KEYS:
        assert math.isnan(desc[key]), key


def _assert_all_finite(desc: dict[str, float]) -> None:
    for key, value in desc.items():
        assert isinstance(value, float)
        assert math.isfinite(value), key


# ---------------------------------------------------------------------------
# 1. Empty and degenerate inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "freqs,amps",
    [
        (np.array([]), np.array([])),
        (np.array([440.0]), np.array([0.0])),
        (np.array([440.0, 880.0]), np.array([0.0, 0.0])),
        (np.array([np.nan]), np.array([1.0])),
        (np.array([440.0]), np.array([-1.0])),
        (np.array([440.0]), np.array([-np.inf])),
    ],
)
def test_empty_or_degenerate_spectrum_returns_all_nan(
    freqs: np.ndarray, amps: np.ndarray
) -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=freqs,
        amplitudes=amps,
    )
    _assert_all_nan(desc)


def test_single_bin_spectrum_has_zero_spread_and_flatness_one() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([1000.0]),
        amplitudes=np.array([1.0]),
    )
    assert desc["spectral_centroid_hz"] == pytest.approx(1000.0)
    assert desc["spectral_spread_hz"] == pytest.approx(0.0)
    assert desc["spectral_skewness"] == pytest.approx(0.0)
    assert desc["spectral_kurtosis"] == pytest.approx(0.0)
    assert desc["spectral_irregularity"] == pytest.approx(0.0)
    assert desc["spectral_flatness"] == pytest.approx(1.0)
    assert desc["spectral_rolloff_hz_85"] == pytest.approx(1000.0)
    assert desc["spectral_rolloff_hz_95"] == pytest.approx(1000.0)
    assert desc["roughness_aures_1985"] == pytest.approx(0.0)
    assert math.isnan(desc["tristimulus_1_fundamental"])


def test_two_bin_symmetric_spectrum_moments() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([100.0, 200.0]),
        amplitudes=np.array([1.0, 1.0]),
    )
    assert desc["spectral_centroid_hz"] == pytest.approx(150.0)
    assert desc["spectral_spread_hz"] == pytest.approx(50.0)
    assert desc["spectral_skewness"] == pytest.approx(0.0)
    assert desc["spectral_kurtosis"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 2. Non-finite handling
# ---------------------------------------------------------------------------

def test_non_finite_values_are_filtered_not_propagated() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([440.0, np.nan, 880.0, np.inf]),
        amplitudes=np.array([1.0, 1.0, 1.0, 1.0]),
    )
    assert desc["spectral_centroid_hz"] == pytest.approx(660.0)
    assert desc["spectral_spread_hz"] == pytest.approx(220.0)
    _assert_all_finite(
        {k: v for k, v in desc.items() if not k.startswith("tristimulus")}
    )


def test_safe_prob_non_finite_sum_returns_zeros() -> None:
    out = _safe_prob(np.array([np.nan, 1.0]))
    assert np.allclose(out, np.array([0.0, 0.0]))


def test_safe_prob_zero_weights_returns_zeros() -> None:
    out = _safe_prob(np.array([0.0, 0.0]))
    assert np.allclose(out, np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# 3. Spectral descriptor contracts
# ---------------------------------------------------------------------------

def test_centroid_is_amplitude_scale_invariant() -> None:
    freqs = np.array([100.0, 200.0, 300.0])
    small = compute_mir_descriptors_from_spectrum(
        frequencies_hz=freqs, amplitudes=np.array([1.0, 2.0, 3.0])
    )
    large = compute_mir_descriptors_from_spectrum(
        frequencies_hz=freqs, amplitudes=np.array([10.0, 20.0, 30.0])
    )
    assert small["spectral_centroid_hz"] == pytest.approx(
        large["spectral_centroid_hz"]
    )
    assert small["spectral_centroid_hz"] == pytest.approx(257.1428571428571)


def test_rolloff_95_is_not_below_rolloff_85() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.linspace(200.0, 4000.0, 12),
        amplitudes=np.linspace(1.0, 0.2, 12),
    )
    assert desc["spectral_rolloff_hz_95"] >= desc["spectral_rolloff_hz_85"]


def test_sparse_spectrum_has_lower_flatness_than_dense_uniform() -> None:
    sparse = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([1000.0, 5000.0]),
        amplitudes=np.array([1.0, 0.001]),
    )
    dense = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.linspace(200.0, 4000.0, 20),
        amplitudes=np.ones(20),
    )
    assert sparse["spectral_flatness"] < dense["spectral_flatness"]
    assert 0.0 <= sparse["spectral_flatness"] <= 1.0


def test_irregularity_increases_with_jagged_amplitudes() -> None:
    smooth = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([110.0, 220.0, 330.0, 440.0]),
        amplitudes=np.array([1.0, 0.8, 0.6, 0.4]),
    )
    jagged = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([110.0, 220.0, 330.0, 440.0]),
        amplitudes=np.array([1.0, 0.1, 0.9, 0.05]),
    )
    assert jagged["spectral_irregularity"] > smooth["spectral_irregularity"]
    assert 0.0 <= jagged["spectral_irregularity"] <= 1.0


def test_tristimulus_partitions_harmonic_energy_when_f0_valid() -> None:
    freqs = np.array([110.0, 220.0, 330.0, 440.0, 550.0, 770.0])
    amps = np.array([1.0, 0.7, 0.45, 0.35, 0.20, 0.10])
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=freqs,
        amplitudes=amps,
        f0_hz=110.0,
    )
    t1 = desc["tristimulus_1_fundamental"]
    t2 = desc["tristimulus_2_low_harmonics_2_to_4"]
    t3 = desc["tristimulus_3_high_harmonics_5_plus"]
    assert t1 == pytest.approx(0.3571428571428571)
    assert t2 == pytest.approx(0.5357142857142857)
    assert t3 == pytest.approx(0.10714285714285715)
    assert abs((t1 + t2 + t3) - 1.0) <= 1e-12


@pytest.mark.parametrize("bad_f0", [None, 0.0, -110.0, float("nan"), float("inf")])
def test_invalid_f0_yields_nan_tristimulus(bad_f0: object) -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([110.0, 220.0]),
        amplitudes=np.array([1.0, 0.5]),
        f0_hz=bad_f0,  # type: ignore[arg-type]
    )
    assert math.isnan(desc["tristimulus_1_fundamental"])
    assert math.isnan(desc["tristimulus_2_low_harmonics_2_to_4"])
    assert math.isnan(desc["tristimulus_3_high_harmonics_5_plus"])


def test_erb_weighted_density_is_finite_for_multi_bin_spectrum() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([220.0, 440.0, 880.0]),
        amplitudes=np.array([1.0, 0.6, 0.3]),
    )
    assert desc["erb_weighted_spectral_density"] >= 0.0
    assert math.isfinite(desc["erb_weighted_spectral_density"])


def test_erb_rate_hz_clamps_negative_frequencies_to_zero() -> None:
    rates = _erb_rate_hz(np.array([-100.0, 0.0, 1000.0]))
    assert rates[0] == pytest.approx(0.0)
    assert rates[1] == pytest.approx(0.0)
    assert rates[2] > 0.0


# ---------------------------------------------------------------------------
# 4. Roughness helper edges (accuracy covered in perf tests)
# ---------------------------------------------------------------------------

def test_roughness_returns_zero_for_single_component() -> None:
    assert _roughness_aures_1985(np.array([440.0]), np.array([1.0])) == 0.0


def test_roughness_returns_zero_for_length_mismatch() -> None:
    assert _roughness_aures_1985(np.array([440.0, 880.0]), np.array([1.0])) == 0.0


def test_roughness_filters_non_positive_frequencies() -> None:
    val = _roughness_aures_1985(
        np.array([0.0, -10.0, 440.0, 445.0]),
        np.array([1.0, 1.0, 1.0, 1.0]),
    )
    assert val >= 0.0
    assert math.isfinite(val)


def test_roughness_returns_zero_when_only_one_valid_frequency_remains() -> None:
    assert (
        _roughness_aures_1985(
            np.array([440.0, -1.0, 0.0]),
            np.array([1.0, 1.0, 1.0]),
        )
        == 0.0
    )


def test_roughness_skips_pairs_beyond_critical_band_cutoff() -> None:
    assert (
        _roughness_aures_1985(
            np.array([440.0, 8000.0]),
            np.array([1.0, 1.0]),
        )
        == 0.0
    )


def test_roughness_is_positive_for_close_partial_pair() -> None:
    val = _roughness_aures_1985(
        np.array([440.0, 445.0]),
        np.array([1.0, 1.0]),
    )
    assert val == pytest.approx(0.09722458223363094, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Shape, dtype, and input stability
# ---------------------------------------------------------------------------

def test_result_dict_keys_are_stable() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([440.0, 880.0]),
        amplitudes=np.array([1.0, 0.5]),
    )
    assert tuple(desc.keys()) == EXPECTED_KEYS


def test_inputs_are_not_mutated() -> None:
    freqs = np.array([110.0, 220.0], dtype=float)
    amps = np.array([1.0, 0.5], dtype=float)
    freqs_copy = freqs.copy()
    amps_copy = amps.copy()
    compute_mir_descriptors_from_spectrum(
        frequencies_hz=freqs,
        amplitudes=amps,
        f0_hz=110.0,
    )
    assert np.array_equal(freqs, freqs_copy)
    assert np.array_equal(amps, amps_copy)


def test_mismatched_frequency_amplitude_lengths_raise_index_error() -> None:
    with pytest.raises(IndexError):
        compute_mir_descriptors_from_spectrum(
            frequencies_hz=np.array([440.0, 880.0]),
            amplitudes=np.array([1.0]),
        )


# ---------------------------------------------------------------------------
# 6. Determinism and order invariance
# ---------------------------------------------------------------------------

def test_compute_mir_descriptors_is_deterministic() -> None:
    freqs = np.array([220.0, 330.0, 440.0, 880.0])
    amps = np.array([1.0, 0.4, 0.7, 0.2])
    first = compute_mir_descriptors_from_spectrum(
        frequencies_hz=freqs, amplitudes=amps, f0_hz=220.0
    )
    second = compute_mir_descriptors_from_spectrum(
        frequencies_hz=freqs, amplitudes=amps, f0_hz=220.0
    )
    for key in EXPECTED_KEYS:
        assert first[key] == pytest.approx(second[key], rel=0.0, abs=0.0)


def test_centroid_and_rolloff_invariant_to_input_order() -> None:
    freqs_a = np.array([880.0, 440.0])
    amps_a = np.array([0.5, 1.0])
    freqs_b = np.array([440.0, 880.0])
    amps_b = np.array([1.0, 0.5])
    desc_a = compute_mir_descriptors_from_spectrum(
        frequencies_hz=freqs_a, amplitudes=amps_a
    )
    desc_b = compute_mir_descriptors_from_spectrum(
        frequencies_hz=freqs_b, amplitudes=amps_b
    )
    assert desc_a["spectral_centroid_hz"] == pytest.approx(
        desc_b["spectral_centroid_hz"]
    )
    assert desc_a["spectral_rolloff_hz_85"] == pytest.approx(
        desc_b["spectral_rolloff_hz_85"]
    )


# ---------------------------------------------------------------------------
# 7. Thesis-critical regression guards
# ---------------------------------------------------------------------------

def test_silence_does_not_yield_finite_spectral_descriptors() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([440.0]),
        amplitudes=np.array([0.0]),
    )
    _assert_all_nan(desc)


def test_single_peak_does_not_imply_broad_spread() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([1000.0]),
        amplitudes=np.array([1.0]),
    )
    assert desc["spectral_spread_hz"] == pytest.approx(0.0)
    assert desc["spectral_centroid_hz"] == pytest.approx(1000.0)


def test_non_finite_artifacts_do_not_yield_plausible_full_spectrum() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([np.nan, np.nan]),
        amplitudes=np.array([1.0, 1.0]),
    )
    _assert_all_nan(desc)


def test_descriptor_dict_contains_only_mir_keys_not_density_metrics() -> None:
    desc = compute_mir_descriptors_from_spectrum(
        frequencies_hz=np.array([220.0, 440.0]),
        amplitudes=np.array([1.0, 0.5]),
    )
    assert set(desc.keys()) == set(EXPECTED_KEYS)
    forbidden_keys = {
        "density",
        "harmonic_energy",
        "inharmonic_energy",
        "effective_partial_density",
    }
    assert forbidden_keys.isdisjoint(desc.keys())
