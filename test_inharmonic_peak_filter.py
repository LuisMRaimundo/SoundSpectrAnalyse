"""Regression tests for ``AudioProcessor._filter_inharmonic_to_local_peaks``.

The filter collapses a bin-level "inharmonic" dataframe (every FFT bin not
matching an expected harmonic) into a peak-level view (discrete partials
above the faintest harmonic amplitude). The motivating bug was that the
"Inharmonic Partials sum" exceeded the "Harmonic Partials sum" on low
clarinet notes simply because the inharmonic dataframe contained thousands
of low-amplitude background bins, which is acoustically misleading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from proc_audio import AudioProcessor


def _make_proc(harmonic_amps: list[float]) -> AudioProcessor:
    proc = AudioProcessor()
    proc.harmonic_list_df = pd.DataFrame(
        {
            "Frequency (Hz)": [440.0 * (i + 1) for i in range(len(harmonic_amps))],
            "Amplitude": list(harmonic_amps),
            "Magnitude (dB)": [20.0 * np.log10(max(a, 1e-12)) for a in harmonic_amps],
        }
    )
    return proc


def test_filter_keeps_top_n_inharmonic_peaks_matching_harmonic_count():
    """With 4 harmonic peaks, the filter must keep exactly 4 inharmonic
    rows — the ones with the largest amplitudes — and discard the
    thousands of background bins between harmonics."""
    proc = _make_proc(harmonic_amps=[5.0, 3.0, 2.0, 1.0])
    rng = np.random.default_rng(0)
    bg = rng.uniform(0.02, 0.20, size=2000)
    bg[3] = 1.5
    bg[7] = 2.3
    bg[400] = 1.8
    bg[1500] = 0.9
    ih_in = pd.DataFrame(
        {
            "Frequency (Hz)": np.linspace(100.0, 10000.0, bg.size),
            "Amplitude": bg,
            "Magnitude (dB)": 20.0 * np.log10(bg),
        }
    )
    ih_out = proc._filter_inharmonic_to_local_peaks(ih_in)
    assert len(ih_out) == 4, len(ih_out)
    kept = sorted(ih_out["Amplitude"].to_numpy().tolist(), reverse=True)
    # The four largest amplitudes must be the survivors.
    assert kept == pytest.approx([2.3, 1.8, 1.5, 0.9], rel=1e-9)


def test_filter_preserves_dominant_inharmonic_peak():
    """When the inharmonic table is much smaller than the harmonic
    count (here 4 harmonics vs. 3 inharmonic bins), the filter still
    returns the strongest inharmonic peaks — never an empty frame."""
    proc = _make_proc(harmonic_amps=[10.0, 8.0, 6.0, 4.0])
    ih_in = pd.DataFrame(
        {
            "Frequency (Hz)": [220.0, 330.0, 660.0],
            "Amplitude": [0.05, 0.10, 7.0],
            "Magnitude (dB)": 20.0 * np.log10(np.array([0.05, 0.10, 7.0])),
        }
    )
    ih_out = proc._filter_inharmonic_to_local_peaks(ih_in)
    assert not ih_out.empty
    assert 660.0 in ih_out["Frequency (Hz)"].tolist()


def test_filter_quantile_fallback_when_no_harmonics():
    """Without a harmonic list (e.g. ultra-noisy signal where harmonic
    detection failed) the filter falls back to a high-quantile cut on
    the inharmonic amplitudes — it must never explode to keep every
    bin and the dominant peak must always survive."""
    proc = AudioProcessor()
    proc.harmonic_list_df = pd.DataFrame()
    rng = np.random.default_rng(1)
    amps = rng.uniform(0.0, 1.0, size=1000)
    amps[-1] = 5.0
    ih_in = pd.DataFrame(
        {
            "Frequency (Hz)": np.linspace(50.0, 12000.0, amps.size),
            "Amplitude": amps,
            "Magnitude (dB)": 20.0 * np.log10(np.maximum(amps, 1e-12)),
        }
    )
    ih_out = proc._filter_inharmonic_to_local_peaks(ih_in)
    # The fallback collapses ~85 % of the rows; never returns the
    # entire table.
    assert len(ih_out) < len(ih_in) // 2
    # The single bright bin must always survive.
    assert np.any(np.isclose(ih_out["Amplitude"].to_numpy(), 5.0))


def test_filter_keeps_loudest_peak_even_when_smaller_than_harmonics():
    """The count-matching rule keeps top-N inharmonic peaks regardless
    of how loud the harmonics are. If there is 1 harmonic and 50
    candidates, the filter keeps the loudest 1 inharmonic peak."""
    proc = _make_proc(harmonic_amps=[100.0])
    ih_in = pd.DataFrame(
        {
            "Frequency (Hz)": np.linspace(100.0, 10000.0, 50),
            "Amplitude": np.linspace(0.1, 2.0, 50),
            "Magnitude (dB)": 20.0 * np.log10(np.linspace(0.1, 2.0, 50)),
        }
    )
    ih_out = proc._filter_inharmonic_to_local_peaks(ih_in)
    assert len(ih_out) == 1
    assert float(ih_out["Amplitude"].iloc[0]) == pytest.approx(2.0)


def test_filter_handles_dbonly_input():
    """Some upstream callers hand us a dataframe with only a dB column
    and no linear amplitude column. The filter must reconstruct the
    amplitude internally and still produce a peak-level output."""
    proc = _make_proc(harmonic_amps=[5.0, 3.0, 1.0])  # 3 harmonics
    db = np.linspace(-40.0, 0.0, 100)
    db[20] = 10.0
    db[55] = 8.0
    db[75] = 6.0
    ih_in = pd.DataFrame(
        {
            "Frequency (Hz)": np.linspace(100.0, 10000.0, db.size),
            "Magnitude (dB)": db,
        }
    )
    ih_out = proc._filter_inharmonic_to_local_peaks(ih_in)
    # With 3 harmonics the filter keeps exactly the 3 loudest dB-only
    # candidates after reconstructing amplitude from the dB column.
    assert len(ih_out) == 3
    kept_db = sorted(ih_out["Magnitude (dB)"].to_numpy().tolist(), reverse=True)
    assert kept_db == pytest.approx([10.0, 8.0, 6.0], rel=1e-9)


def test_filter_returns_unmodified_on_empty_or_missing_columns():
    proc = _make_proc(harmonic_amps=[1.0])
    assert proc._filter_inharmonic_to_local_peaks(pd.DataFrame()).empty
    assert proc._filter_inharmonic_to_local_peaks(None) is None
    # No amplitude / dB columns → return as-is.
    only_freq = pd.DataFrame({"Frequency (Hz)": [100.0, 200.0]})
    out = proc._filter_inharmonic_to_local_peaks(only_freq)
    pd.testing.assert_frame_equal(out, only_freq)
