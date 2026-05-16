"""
Golden / reference checks for the STFT path (energy + bin alignment).

Run as part of the suite, or alone:
  python -m pytest tests/test_stft_reference_goldens.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import librosa
from scipy.signal import get_window

from constants import ENERGY_CONSERVATION_TOLERANCE_STRICT
from proc_audio import _normalize_level, _verify_energy_conservation


def test_parseval_energy_ratio_hann_1024_hop1024_normalized():
    """Librosa STFT (Hann, hop=n_fft, center=True) vs time energy within strict tolerance."""
    sr = 44100
    t = np.linspace(0.0, 1.0, int(sr), endpoint=False)
    y = np.sin(2 * np.pi * 440.0 * t)
    y_norm = _normalize_level(y, target_rms_db=-20.0)
    n_fft = 1024
    hop = 1024
    w = get_window("hann", n_fft, fftbins=True)
    S = librosa.stft(
        y_norm,
        n_fft=n_fft,
        win_length=len(w),
        hop_length=hop,
        window=w,
        center=True,
    )
    r = _verify_energy_conservation(
        y_norm,
        S,
        n_fft,
        hop,
        "hann",
        tolerance=ENERGY_CONSERVATION_TOLERANCE_STRICT,
        window_array=w,
    )
    assert r["is_valid"], (
        f"energy_ratio={r['energy_ratio']:.6f} deviation={r['deviation']:.6f} "
        f"tolerance={r['tolerance']}"
    )
    assert abs(float(r["energy_ratio"]) - 1.0) <= float(ENERGY_CONSERVATION_TOLERANCE_STRICT)


def test_grid_aligned_partial_peak_bin_mean_magnitude():
    """A partial exactly on an FFT bin peaks at that bin (mean |STFT| over time)."""
    sr = 44100
    n_fft = 1024
    hop = 1024
    k_target = 17
    f0 = k_target * sr / float(n_fft)
    n = int(sr * 2)
    t = np.arange(n, dtype=float) / sr
    y = np.sin(2 * np.pi * f0 * t)
    w = get_window("hann", n_fft, fftbins=True)
    S = librosa.stft(
        y,
        n_fft=n_fft,
        win_length=len(w),
        hop_length=hop,
        window=w,
        center=True,
    )
    mag_mean = np.mean(np.abs(S), axis=1)
    peak_bin = int(np.argmax(mag_mean))
    assert peak_bin == k_target
    assert mag_mean[k_target] >= 0.99 * float(np.max(mag_mean))


def test_librosa_stft_matches_proc_audio_window_vector():
    """AudioProcessor uses scipy.signal.get_window(..., fftbins=True) — match explicitly."""
    from proc_audio import AudioProcessor

    sr = 48000
    n_fft = 1024
    hop = 1024
    n = int(sr * 0.5)
    t = np.arange(n, dtype=float) / sr
    y = 0.08 * np.sin(2 * np.pi * 512.0 * t)

    ap = AudioProcessor()
    ap.sr = sr
    ap.y = y.astype(np.float64)
    ap.n_fft = n_fft
    ap.hop_length = hop
    ap.window = "hann"
    win_arg = ap._get_window_arg()

    S_code = librosa.stft(
        y,
        n_fft=n_fft,
        win_length=len(win_arg),
        hop_length=hop,
        window=win_arg,
        center=True,
    )
    S_ref = librosa.stft(
        y,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop,
        window="hann",
        center=True,
    )
    diff_db = 20.0 * np.log10(np.maximum(np.abs(S_code), 1e-20)) - 20.0 * np.log10(
        np.maximum(np.abs(S_ref), 1e-20)
    )
    assert float(np.nanmax(np.abs(diff_db))) < 0.01
