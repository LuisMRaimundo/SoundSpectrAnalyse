"""Regression tests for harmonic-order alignment (cents + collapse), replacing percent deviation diagnostics."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from constants import HARMONIC_VALIDATION_MAX_HARMONICS
from harmonic_alignment import compute_harmonic_alignment_metrics


def _peaks_df(freqs: list[float], amps: list[float] | None = None) -> pd.DataFrame:
    if amps is None:
        amps = [1.0] * len(freqs)
    return pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude_linear": amps})


def test_synthetic_ladder_low_cents_excellent_or_good() -> None:
    f0 = 440.0
    n_need = 40
    freqs = [f0 * n for n in range(1, n_need + 1)]
    df = _peaks_df(freqs)
    ha = compute_harmonic_alignment_metrics(
        f0,
        df,
        sample_rate=44100.0,
        n_fft=4096,
        max_frequency_hz=20000.0,
        tolerance_cents=None,
        subbass_cutoff_hz=20.0,
    )
    assert ha["harmonic_alignment_matched_count"] >= 3
    assert float(ha["harmonic_alignment_mean_abs_error_cents"]) < 1.0
    assert ha["harmonic_alignment_status"] in ("excellent", "good")


def test_inharmonic_candidates_outside_windows_do_not_inflate_order_error() -> None:
    f0 = 440.0
    base = [f0 * n for n in range(1, 33)]
    extra = [150.0, 233.7, 901.2, 1203.5, 3456.0, 7890.0]
    freqs = base + extra
    amps = [1.0] * len(base) + [0.01] * len(extra)
    df = _peaks_df(freqs, amps)
    ha = compute_harmonic_alignment_metrics(
        f0,
        df,
        sample_rate=44100.0,
        n_fft=4096,
        max_frequency_hz=20000.0,
        subbass_cutoff_hz=20.0,
    )
    assert int(ha["inharmonic_candidate_count"]) > 0
    assert float(ha["inharmonic_candidate_energy_ratio"]) > 0.0
    assert float(ha["harmonic_alignment_mean_abs_error_cents"]) < 2.0


def test_detuned_ladder_higher_cents_error_than_exact() -> None:
    f0 = 440.0
    freqs_exact = [f0 * n for n in range(1, 35)]
    stretch = 2.0 ** (75.0 / 1200.0)
    freqs_detuned = [f0 * n * stretch for n in range(1, 35)]
    ha_ok = compute_harmonic_alignment_metrics(
        f0,
        _peaks_df(freqs_exact),
        sample_rate=44100.0,
        n_fft=4096,
        max_frequency_hz=20000.0,
        subbass_cutoff_hz=20.0,
    )
    ha_bad = compute_harmonic_alignment_metrics(
        f0,
        _peaks_df(freqs_detuned),
        sample_rate=44100.0,
        n_fft=4096,
        max_frequency_hz=20000.0,
        subbass_cutoff_hz=20.0,
    )
    assert float(ha_ok["harmonic_alignment_mean_abs_error_cents"]) < 1.0
    assert float(ha_bad["harmonic_alignment_mean_abs_error_cents"]) > float(
        ha_ok["harmonic_alignment_mean_abs_error_cents"]
    ) + 5.0


def test_forbidden_legacy_strings_absent_in_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "harmonic_alignment.py",
        root / "harmonic_validation.py",
        root / "proc_audio.py",
        root / "audio_analysis" / "super_audio_analyzer.py",
    ]
    _wolf_uc = "".join(chr(c) for c in (87, 111, 108, 102, 114, 97, 109))
    _wolf_lc = "".join(chr(c) for c in (119, 111, 108, 102, 114, 97, 109))
    forbidden_in_sources = (
        "avg deviation",
        "Harmonic series validation warning",
        "Harmonic-series consistency review",
        "spurious",
        _wolf_uc + " Alpha",
        "mcp_" + _wolf_lc + "-alpha_query_" + _wolf_lc,
    )
    for p in paths:
        text = p.read_text(encoding="utf-8", errors="replace")
        low = text.lower()
        assert forbidden_in_sources[0] not in low
        for s in forbidden_in_sources[1:]:
            assert s not in text


def test_operates_on_peak_rows_not_fft_bin_grid() -> None:
    f0 = 100.0
    n_fft = 4096
    df = _peaks_df([f0 * n for n in range(1, 8)])
    ha = compute_harmonic_alignment_metrics(
        f0,
        df,
        sample_rate=44100.0,
        n_fft=n_fft,
        max_frequency_hz=5000.0,
        subbass_cutoff_hz=1.0,
    )
    assert len(df) < n_fft // 2
    assert int(ha["harmonic_alignment_expected_count"]) == int(
        min(HARMONIC_VALIDATION_MAX_HARMONICS, math.floor(5000.0 / f0))
    )
    assert int(ha["harmonic_alignment_matched_count"]) <= int(ha["harmonic_alignment_expected_count"])
    assert int(ha["harmonic_alignment_matched_count"]) <= len(df)


def test_adaptive_tolerance_used_when_sr_or_fft_missing() -> None:
    df = _peaks_df([440.0 * n for n in range(1, 20)])
    ha = compute_harmonic_alignment_metrics(
        440.0, df, sample_rate=None, n_fft=None, max_frequency_hz=20000.0, subbass_cutoff_hz=20.0
    )
    assert float(ha["harmonic_alignment_tolerance_cents_used"]) >= 18.0
