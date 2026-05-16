"""Regression tests for the acoustic-physics corrections to the
sub-bass aggregator and harmonic candidate gate. These fix the
Clarinete_mf findings #1 (sub-bass lower bound), #2 (window-aware
harmonic-protection tolerance) and #3 (off-frequency harmonic
demotion).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from density import (
    SUBBASS_AGGREGATE_LOWER_HZ,
    SUBBASS_PROTECTION_BIN_MULTIPLIER,
    SUBBASS_PROTECTION_MIN_HZ,
    aggregate_subbass_noise_peak_power,
    compute_subbass_protection_tolerance_hz,
)


# ---------------------------------------------------------------------------
# Finding #1 — sub-bass lower-frequency floor
# ---------------------------------------------------------------------------
def test_subbass_lower_bound_drops_dc_and_subaudible_bins() -> None:
    """Bins below ``SUBBASS_AGGREGATE_LOWER_HZ`` (30 Hz) must not
    contribute to the sub-bass aggregate."""
    # A spectrum with energy concentrated below 30 Hz (DC offset, HVAC,
    # room rumble) plus a single legitimate musical sub-bass peak at
    # 80 Hz. With the lower-bound applied, only the 80 Hz peak counts.
    spectrum = pd.DataFrame(
        {
            "Frequency (Hz)": [2.7, 5.4, 10.8, 16.2, 21.5, 27.0,
                               40.0, 60.0, 80.0, 120.0, 150.0],
            "Amplitude": [50.0, 40.0, 25.0, 15.0, 8.0, 4.0,
                          1.0, 1.0, 10.0, 0.5, 0.5],
        }
    )
    # No harmonic protection in play.
    pow_no_lower = aggregate_subbass_noise_peak_power(
        spectrum, None, subbass_hz=200.0, subbass_lower_hz=0.0,
        low_band_mode="sum_all_bins",
    )
    pow_with_lower = aggregate_subbass_noise_peak_power(
        spectrum, None, subbass_hz=200.0,
        subbass_lower_hz=SUBBASS_AGGREGATE_LOWER_HZ,
        low_band_mode="sum_all_bins",
    )
    assert pow_with_lower < pow_no_lower
    # Pure musical-band energy = 1**2 + 1**2 + 10**2 + 0.5**2 + 0.5**2 = 102.5
    assert pow_with_lower == 102.5


def test_subbass_lower_bound_keeps_legitimate_low_bass_peak() -> None:
    """A real 35 Hz tone (e.g. organ pedal C1, contrabassoon range)
    must survive the lower-frequency filter."""
    spectrum = pd.DataFrame(
        {
            "Frequency (Hz)": [35.0, 80.0],
            "Amplitude": [10.0, 5.0],
        }
    )
    pow_filt = aggregate_subbass_noise_peak_power(
        spectrum, None, subbass_hz=200.0,
        subbass_lower_hz=SUBBASS_AGGREGATE_LOWER_HZ,
        low_band_mode="sum_all_bins",
    )
    # Expected: 10**2 + 5**2 = 125.0
    assert pow_filt == 125.0


# ---------------------------------------------------------------------------
# Finding #2 — window-aware harmonic-protection tolerance
# ---------------------------------------------------------------------------
def test_protection_tolerance_scales_with_fft_resolution() -> None:
    """For n_fft = 8192, sr = 44100: bin_hz = 5.385... → tolerance =
    max(12, 4 * 5.385) = 21.54 Hz."""
    tol = compute_subbass_protection_tolerance_hz(44100.0, 8192)
    assert tol > SUBBASS_PROTECTION_MIN_HZ
    bin_hz = 44100.0 / 8192
    expected = max(SUBBASS_PROTECTION_MIN_HZ,
                   SUBBASS_PROTECTION_BIN_MULTIPLIER * bin_hz)
    assert np.isclose(tol, expected)


def test_protection_tolerance_min_floor() -> None:
    """For very high-resolution FFTs (tiny bin_hz) the tolerance floor
    is 12 Hz so we still suppress modest jitter."""
    tol = compute_subbass_protection_tolerance_hz(44100.0, 131072)
    assert tol == SUBBASS_PROTECTION_MIN_HZ


def test_protection_tolerance_handles_bad_inputs() -> None:
    """The helper must never raise on degenerate inputs."""
    assert compute_subbass_protection_tolerance_hz(0.0, 8192) == SUBBASS_PROTECTION_MIN_HZ
    assert compute_subbass_protection_tolerance_hz(-1.0, 8192) == SUBBASS_PROTECTION_MIN_HZ
    assert compute_subbass_protection_tolerance_hz(44100.0, 0) == SUBBASS_PROTECTION_MIN_HZ
    assert compute_subbass_protection_tolerance_hz(float("nan"), 8192) == SUBBASS_PROTECTION_MIN_HZ


def test_wider_protection_suppresses_main_lobe_leakage() -> None:
    """Reproduce the D3 leakage pattern: a strong fundamental at
    146.83 Hz with main-lobe shoulders at ±12 Hz and ±15 Hz.
    With 12 Hz tolerance the ±15 Hz shoulders leak into the sub-bass.
    With the new window-aware 21.5 Hz tolerance they are properly
    excluded.
    """
    spectrum = pd.DataFrame(
        {
            # Fundamental position + window leakage shoulders.
            "Frequency (Hz)": [131.83, 146.83, 161.83, 50.0, 80.0],
            "Amplitude":      [   20.0,  100.0,   25.0,  1.0,  1.0],
        }
    )
    harm = pd.DataFrame({"Frequency (Hz)": [146.83]})

    pow_12hz = aggregate_subbass_noise_peak_power(
        spectrum, harm, subbass_hz=200.0,
        subbass_lower_hz=SUBBASS_AGGREGATE_LOWER_HZ,
        freq_match_tol_hz=12.0,
        low_band_mode="sum_all_bins",
    )
    pow_wide = aggregate_subbass_noise_peak_power(
        spectrum, harm, subbass_hz=200.0,
        subbass_lower_hz=SUBBASS_AGGREGATE_LOWER_HZ,
        freq_match_tol_hz=21.5,  # window-aware tolerance for these FFT settings
        low_band_mode="sum_all_bins",
    )
    # With 12 Hz: shoulders at 131.83 and 161.83 (Δ=15 Hz from f0)
    # escape protection → pow_12hz = 20² + 25² + 1² + 1² = 1027
    # With 21.5 Hz: shoulders are protected → pow_wide = 1² + 1² = 2
    assert pow_12hz == 1027.0
    assert pow_wide == 2.0
