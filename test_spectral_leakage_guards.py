"""Tests for STFT spectral leakage guard helpers and ``identify_inharmonic_partials`` integration."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from spectral_leakage_guards import (  # noqa: E402
    DEFAULT_MAIN_LOBE_WIDTH_BINS,
    filter_inharmonic_peak_candidates,
    leakage_halfwidth_hz,
)
from density import identify_inharmonic_partials  # noqa: E402


def test_leakage_halfwidth_hz_from_bin_width() -> None:
    bw = 10.0
    lh = leakage_halfwidth_hz(bin_width_hz=bw, main_lobe_bins=4.0)
    assert lh == pytest.approx(0.5 * 4.0 * bw)


def test_leakage_halfwidth_hz_from_sr_nfft() -> None:
    lh = leakage_halfwidth_hz(sr=44100.0, n_fft=4096, main_lobe_bins=4.0)
    bw = 44100.0 / 4096.0
    assert lh == pytest.approx(0.5 * DEFAULT_MAIN_LOBE_WIDTH_BINS * bw)


def test_filter_inharmonic_peak_candidates_removes_near_harmonic() -> None:
    cands = [(1000.0, 1.0), (1005.0, 0.5), (3000.0, 0.3)]
    harm = [1000.0]
    out = filter_inharmonic_peak_candidates(cands, harm, leakage_halfwidth_hz=20.0)
    freqs = [f for f, _ in out]
    assert 1000.0 not in freqs  # guard removes near harmonic rep
    assert 1005.0 not in freqs
    assert 3000.0 in freqs


def test_identify_inharmonic_widens_exclusion_with_guard() -> None:
    harmonic_df = pd.DataFrame({"Frequency (Hz)": [1000.0]})
    # 15 Hz from harmonic: outside 0.2% (2 Hz) but inside leakage guard (~20 Hz for bw=10)
    complete_df = pd.DataFrame({"Frequency (Hz)": [1015.0], "Magnitude (dB)": [-40.0]})
    narrow = identify_inharmonic_partials(
        harmonic_df,
        complete_df,
        tolerance=0.002,
        spectral_leakage_guard=False,
    )
    wide = identify_inharmonic_partials(
        harmonic_df,
        complete_df,
        tolerance=0.002,
        bin_width_hz=10.0,
        main_lobe_bins=4.0,
        spectral_leakage_guard=True,
    )
    assert len(narrow) == 1
    assert len(wide) == 0
