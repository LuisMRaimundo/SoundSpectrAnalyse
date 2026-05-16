"""
Reference signal utilities for validation tests.
Generates deterministic synthetic audio with known harmonic/inharmonic structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np


@dataclass(frozen=True)
class SignalConfig:
    sample_rate: int = 22050
    duration_s: float = 1.0


def _time_vector(cfg: SignalConfig) -> np.ndarray:
    n_samples = int(cfg.sample_rate * cfg.duration_s)
    return np.linspace(0.0, cfg.duration_s, n_samples, endpoint=False)


def sine_wave(freq_hz: float, amplitude: float, cfg: SignalConfig) -> np.ndarray:
    t = _time_vector(cfg)
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)


def harmonic_stack(
    f0_hz: float,
    harmonics: Iterable[int],
    amplitudes: Iterable[float],
    cfg: SignalConfig,
) -> np.ndarray:
    signal = np.zeros(int(cfg.sample_rate * cfg.duration_s), dtype=float)
    for n, amp in zip(harmonics, amplitudes):
        signal += sine_wave(f0_hz * n, amp, cfg)
    return signal


def inharmonic_tone(f_hz: float, amplitude: float, cfg: SignalConfig) -> np.ndarray:
    return sine_wave(f_hz, amplitude, cfg)


def normalize_peak(signal: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    peak = np.max(np.abs(signal)) if signal.size else 0.0
    if peak <= 0:
        return signal
    return (signal / peak) * target_peak
