"""Regression tests for the saddle-based prominence definition.

Before this fix, ``_local_peak_metrics`` returned ``prominence_db`` as
``peak - max(left_neighbour, right_neighbour)``. For a Blackman–Harris
window (main lobe ≈ 2 bins wide) that measured main-lobe curvature, not
peak prominence, so every real harmonic of a clean clarinet note was
rejected by a ``prominence >= 3 dB`` strict gate. The fix replaces the
±1-bin comparison with a saddle window (±10 bins by default) and adds a
small peak-refinement step so the metric is evaluated at the actual local
maximum rather than at the bin closest to ``n * f0``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proc_audio import (  # noqa: E402
    _is_local_peak_valid,
    _local_peak_metrics,
    _refine_peak_index,
    _saddle_prominence_db,
)


def _blackmanharris(n: int) -> np.ndarray:
    n = int(n)
    if n <= 1:
        return np.ones(n, dtype=float)
    a0, a1, a2, a3 = 0.35875, 0.48829, 0.14128, 0.01168
    k = np.arange(n)
    return (
        a0
        - a1 * np.cos(2 * np.pi * k / (n - 1))
        + a2 * np.cos(4 * np.pi * k / (n - 1))
        - a3 * np.cos(6 * np.pi * k / (n - 1))
    ).astype(float)


def _windowed_sinusoid_spectrum(
    *,
    f_hz: float,
    sr: int = 44_100,
    n_fft: int = 4096,
    noise_amp: float = 1e-5,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n_fft) / float(sr)
    x = np.sin(2 * np.pi * f_hz * t)
    rng = np.random.default_rng(42)
    x = x + noise_amp * rng.standard_normal(n_fft)
    x = x * _blackmanharris(n_fft)
    spec = np.fft.rfft(x)
    mags = np.abs(spec)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    return freqs, mags


def test_saddle_prominence_clean_sinusoid_passes_strict_gate() -> None:
    """A clean windowed sinusoid must produce prominence >> 3 dB.

    The legacy ``peak - max(±1 bin)`` formula returned ~0.1-1 dB for the
    same signal because both immediate neighbours sat inside the
    Blackman–Harris main lobe.
    """
    freqs, mags = _windowed_sinusoid_spectrum(f_hz=440.0)
    peak_idx = int(np.argmax(mags))
    prom_db = _saddle_prominence_db(mags, peak_idx, saddle_window=10)
    assert prom_db > 40.0, prom_db


def test_saddle_prominence_rejects_pure_noise_floor() -> None:
    """A bin that's only ~1 dB above its saddle must fail the 3 dB gate."""
    n = 4096
    mags = np.full(n, 1.0, dtype=float)
    mags[1000] = 1.1
    prom_db = _saddle_prominence_db(mags, 1000, saddle_window=10)
    assert prom_db < 3.0, prom_db


def test_strict_gate_accepts_real_harmonics_after_fix() -> None:
    """End-to-end: a clean A4 windowed sinusoid produces a strict peak."""
    freqs, mags = _windowed_sinusoid_spectrum(f_hz=440.0)
    approx_idx = int(np.argmin(np.abs(freqs - 440.0)))
    is_valid, snr_db = _is_local_peak_valid(mags, approx_idx)
    assert bool(is_valid) is True
    assert snr_db > 30.0


def test_strict_gate_handles_bin_offset_via_refinement() -> None:
    """When the candidate index is the lobe shoulder (not the lobe top),
    the refinement step snaps to the true local maximum within ±2 bins so
    the local-max + prominence checks still succeed."""
    freqs, mags = _windowed_sinusoid_spectrum(f_hz=440.0)
    true_peak = int(np.argmax(mags))
    shoulder_idx = true_peak + 1
    refined = _refine_peak_index(mags, shoulder_idx, refine_radius=2)
    assert refined == true_peak

    is_valid, snr_db = _is_local_peak_valid(mags, shoulder_idx)
    assert bool(is_valid) is True
    assert snr_db > 30.0


def test_local_peak_metrics_prominence_is_saddle_based() -> None:
    """The ``prominence_db`` field returned by ``_local_peak_metrics`` must
    use the saddle definition, not the ±1-bin comparison."""
    freqs, mags = _windowed_sinusoid_spectrum(f_hz=440.0)
    peak_idx = int(np.argmax(mags))
    is_peak, snr_db, prom_db = _local_peak_metrics(mags, peak_idx)
    assert is_peak is True
    assert snr_db > 30.0
    assert prom_db > 40.0, prom_db


def test_refine_peak_radius_stays_within_window() -> None:
    """Refinement must not pull the index further than ``refine_radius``."""
    mags = np.array([0.1, 0.2, 0.3, 0.4, 1.0, 0.4, 0.3, 0.2, 0.1])
    assert _refine_peak_index(mags, 4, refine_radius=2) == 4
    assert _refine_peak_index(mags, 3, refine_radius=2) == 4
    assert _refine_peak_index(mags, 2, refine_radius=2) == 4
    assert _refine_peak_index(mags, 1, refine_radius=2) == 3
    assert _refine_peak_index(mags, 0, refine_radius=2) == 2
