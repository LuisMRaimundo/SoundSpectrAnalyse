from __future__ import annotations

"""
Helper-level contract tests for harmonic_peak_validation.py.

Complements tests/phase_11/test_cfar_detection.py (CFAR only) and
tests/phase_10/test_harmonic_inclusion_audit.py (two audit-reason cases)
with focused coverage of peak refinement, saddle prominence, local-peak
metrics, harmonic-candidate classification, and audit labelling.

No production code changes. No audio files, GUI, plotting, or full pipeline.
"""

import math

import numpy as np
import pytest

import harmonic_peak_validation as hpv


def _isolated_peak_spectrum(
    *,
    n: int = 200,
    peak_idx: int = 100,
    peak_amp: float = 1.0,
    floor: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    mags = np.full(n, floor, dtype=float)
    mags[peak_idx] = peak_amp
    if 0 < peak_idx < n - 1:
        mags[peak_idx - 1] = peak_amp * 0.8
        mags[peak_idx + 1] = peak_amp * 0.8
    lo = max(0, peak_idx - 20)
    hi = min(n, peak_idx + 21)
    mags[lo:peak_idx - 5] = floor * 0.5
    mags[peak_idx + 6 : hi] = floor * 0.5
    freqs = np.linspace(0.0, 2000.0, n)
    return freqs, mags


# ---------------------------------------------------------------------------
# 1. Status token stability
# ---------------------------------------------------------------------------

def test_harmonic_candidate_status_values_are_stable() -> None:
    assert hpv.HARMONIC_CANDIDATE_STATUS_VALUES == (
        "strict_validated",
        "snr_validated",
        "weak_candidate",
        "below_noise_floor",
        "missing_window",
        "rejected_bad_f0",
        "off_frequency",
    )


# ---------------------------------------------------------------------------
# 2. _classify_harmonic_candidate
# ---------------------------------------------------------------------------

def test_classify_strict_validated_is_only_density_inclusion_path() -> None:
    status, include = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0,
        local_peak_valid=True,
        snr_db=10.0,
        prominence_db=10.0,
    )
    assert status == "strict_validated"
    assert include is True


def test_classify_snr_validated_never_includes_density() -> None:
    status, include = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0,
        local_peak_valid=True,
        snr_db=5.0,
        prominence_db=1.0,
    )
    assert status == "snr_validated"
    assert include is False


def test_classify_weak_and_below_noise_floor_reject_density() -> None:
    weak_status, weak_inc = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0,
        local_peak_valid=True,
        snr_db=1.0,
        prominence_db=1.0,
    )
    assert weak_status == "weak_candidate"
    assert weak_inc is False

    low_status, low_inc = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0,
        local_peak_valid=False,
        snr_db=-5.0,
        prominence_db=-5.0,
    )
    assert low_status == "below_noise_floor"
    assert low_inc is False


@pytest.mark.parametrize(
    "bad_amp",
    [0.0, -1.0, float("nan"), float("inf")],
)
def test_classify_nonpositive_or_non_finite_amplitude_is_missing_window(
    bad_amp: float,
) -> None:
    status, include = hpv._classify_harmonic_candidate(
        amplitude_raw=bad_amp,
        local_peak_valid=True,
        snr_db=20.0,
        prominence_db=20.0,
    )
    assert status == "missing_window"
    assert include is False


def test_classify_cfar_detected_false_blocks_strict_even_with_high_snr() -> None:
    status, include = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0,
        local_peak_valid=True,
        snr_db=20.0,
        prominence_db=20.0,
        cfar_detected=False,
    )
    assert status == "snr_validated"
    assert include is False


def test_classify_cfar_detected_true_allows_strict_when_prominence_met() -> None:
    status, include = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0,
        local_peak_valid=False,
        snr_db=10.0,
        prominence_db=10.0,
        cfar_detected=True,
    )
    assert status == "strict_validated"
    assert include is True


def test_classify_rejected_bad_f0_status() -> None:
    status, include = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0,
        local_peak_valid=True,
        snr_db=20.0,
        prominence_db=20.0,
        rejected_bad_f0=True,
    )
    assert status == "rejected_bad_f0"
    assert include is False


def test_classify_threshold_equality_at_strict_boundary() -> None:
    status, include = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0,
        local_peak_valid=True,
        snr_db=3.0,
        prominence_db=3.0,
        strict_snr_db=3.0,
        strict_prominence_db=3.0,
    )
    assert status == "strict_validated"
    assert include is True


# ---------------------------------------------------------------------------
# 3. Peak refinement and parabolic interpolation
# ---------------------------------------------------------------------------

def test_refine_peak_index_snaps_off_shoulder_to_local_maximum() -> None:
    mags = np.array([0.1, 0.2, 1.0, 0.3, 0.1])
    assert hpv._refine_peak_index(mags, 1, refine_radius=2) == 2
    assert hpv._refine_peak_index(mags, 3, refine_radius=2) == 2


def test_refine_peak_index_empty_array_passthrough() -> None:
    assert hpv._refine_peak_index(np.array([]), 0) == 0


def test_parabolic_interpolation_exact_on_symmetric_peak() -> None:
    mags = np.array([0.1, 1.0, 0.1])
    freq, valid = hpv._parabolic_interpolation_log_magnitude(mags, 1, 1.0, 0.0)
    assert valid is True
    assert freq == pytest.approx(1.0, abs=1e-9)


def test_parabolic_interpolation_edge_index_not_valid() -> None:
    mags = np.array([1.0, 0.5, 0.2])
    freq, valid = hpv._parabolic_interpolation_log_magnitude(mags, 0, 1.0, 0.0)
    assert valid is False
    assert freq == pytest.approx(0.0, abs=1e-12)


def test_infer_bin_spacing_from_freqs_median_and_degenerate() -> None:
    freqs = np.array([0.0, 10.0, 20.0, 30.0])
    assert hpv._infer_bin_spacing_from_freqs(freqs) == pytest.approx(10.0, abs=1e-12)
    assert math.isnan(hpv._infer_bin_spacing_from_freqs(np.array([100.0])))
    assert math.isnan(hpv._infer_bin_spacing_from_freqs(np.array([])))


def test_refine_candidate_to_interpolated_peak_schema_and_non_mutation() -> None:
    freqs, mags = _isolated_peak_spectrum()
    mags_snapshot = mags.copy()
    freqs_snapshot = freqs.copy()
    out = hpv._refine_candidate_to_interpolated_peak(
        candidate_freq_hz=float(freqs[100]),
        complete_magnitudes=mags,
        complete_freqs=freqs,
    )
    assert set(out.keys()) == {
        "peak_bin_index",
        "bin_center_frequency_hz",
        "interpolated_frequency_hz",
        "subbin_offset_bins",
        "subbin_interpolation_valid",
        "peak_amplitude_raw",
        "peak_magnitude_db",
    }
    assert out["peak_amplitude_raw"] == pytest.approx(1.0, abs=1e-12)
    assert out["subbin_interpolation_valid"] is True
    np.testing.assert_array_equal(mags, mags_snapshot)
    np.testing.assert_array_equal(freqs, freqs_snapshot)


def test_refine_candidate_empty_inputs_return_nan_payload() -> None:
    out = hpv._refine_candidate_to_interpolated_peak(
        candidate_freq_hz=440.0,
        complete_magnitudes=np.array([]),
        complete_freqs=np.array([]),
    )
    assert math.isnan(out["peak_amplitude_raw"])
    assert out["subbin_interpolation_valid"] is False


# ---------------------------------------------------------------------------
# 4. Saddle prominence and f0-adaptive window
# ---------------------------------------------------------------------------

def test_saddle_prominence_strong_isolated_peak_exceeds_threshold() -> None:
    _, mags = _isolated_peak_spectrum()
    prom = hpv._saddle_prominence_db(mags, 100, saddle_window=10)
    assert prom > 30.0


def test_saddle_prominence_edge_index_is_negative_infinity() -> None:
    mags = np.array([1.0, 0.5, 0.2])
    assert hpv._saddle_prominence_db(mags, 0, saddle_window=10) == float("-inf")


def test_prominence_saddle_window_scales_with_f0_register() -> None:
    low = hpv._prominence_saddle_window_bins(f0_hz=65.0, bin_spacing_hz=10.0)
    high = hpv._prominence_saddle_window_bins(f0_hz=2000.0, bin_spacing_hz=10.0)
    assert low == 3
    assert high == 100
    assert high > low


def test_local_peak_metrics_agree_with_saddle_prominence() -> None:
    _, mags = _isolated_peak_spectrum()
    local_ok, snr_db, prom_db = hpv._local_peak_metrics(mags, 100, saddle_window=10)
    assert local_ok is True
    assert prom_db == pytest.approx(hpv._saddle_prominence_db(mags, 100, saddle_window=10), rel=1e-9)
    assert snr_db > 3.0


def test_is_local_peak_valid_rejects_flat_neighbourhood() -> None:
    mags = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    valid, snr = hpv._is_local_peak_valid(mags, 2, threshold_db=3.0, saddle_window=1)
    assert valid is False


def test_is_local_peak_valid_accepts_isolated_peak() -> None:
    _, mags = _isolated_peak_spectrum()
    valid, snr = hpv._is_local_peak_valid(mags, 100, threshold_db=3.0, saddle_window=10)
    assert bool(valid) is True
    assert snr > 3.0


# ---------------------------------------------------------------------------
# 5. Harmonic inclusion audit labelling (extended)
# ---------------------------------------------------------------------------

def test_harmonic_inclusion_audit_reason_preserves_status_strings() -> None:
    reason = hpv._harmonic_inclusion_audit_exclusion_reason(
        include_for_density=False,
        expected_frequency_hz=440.0,
        frequency_deviation_hz=0.0,
        candidate_status="snr_validated",
        local_peak_valid=True,
        snr_db=5.0,
        prominence_db=1.0,
    )
    assert isinstance(reason, str)
    assert reason.startswith("prominence_below_3dB")


def test_harmonic_inclusion_audit_reason_high_expected_frequency_ceiling() -> None:
    reason = hpv._harmonic_inclusion_audit_exclusion_reason(
        include_for_density=False,
        expected_frequency_hz=6000.0,
        frequency_deviation_hz=0.0,
        candidate_status="strict_validated",
        local_peak_valid=True,
        snr_db=10.0,
        prominence_db=10.0,
    )
    assert reason.startswith("above_body_density_ceiling_5khz")


def test_harmonic_inclusion_audit_included_label() -> None:
    assert (
        hpv._harmonic_inclusion_audit_exclusion_reason(
            include_for_density=True,
            expected_frequency_hz=440.0,
            frequency_deviation_hz=0.0,
            candidate_status="strict_validated",
            local_peak_valid=True,
            snr_db=6.0,
            prominence_db=6.0,
        )
        == "included"
    )


# ---------------------------------------------------------------------------
# 6. Regression guards and determinism
# ---------------------------------------------------------------------------

def test_high_register_prominence_window_does_not_shrink_to_fixed_ten_bins() -> None:
    # High f0 should widen the saddle search radius beyond the legacy fixed 10.
    win = hpv._prominence_saddle_window_bins(f0_hz=1500.0, bin_spacing_hz=5.0)
    assert win > 10


def test_classify_many_salient_non_strict_candidates_do_not_accumulate_density_flags() -> None:
    included = 0
    for snr, prom in ((8.0, 1.0), (4.0, 2.0), (10.0, 2.9)):
        _, inc = hpv._classify_harmonic_candidate(
            amplitude_raw=1.0,
            local_peak_valid=True,
            snr_db=snr,
            prominence_db=prom,
        )
        included += int(inc)
    assert included == 0


def test_harmonic_peak_validation_helpers_are_deterministic() -> None:
    _, mags = _isolated_peak_spectrum()
    a = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0, local_peak_valid=True, snr_db=10.0, prominence_db=10.0
    )
    b = hpv._classify_harmonic_candidate(
        amplitude_raw=1.0, local_peak_valid=True, snr_db=10.0, prominence_db=10.0
    )
    assert a == b
    m1 = hpv._local_peak_metrics(mags, 100, saddle_window=10)
    m2 = hpv._local_peak_metrics(mags, 100, saddle_window=10)
    assert m1 == m2
