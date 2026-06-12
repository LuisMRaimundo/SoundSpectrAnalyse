from __future__ import annotations

"""
Sixth Phase 12 preprocessing / DataFrame-assembly contract layer for proc_audio.py.

Complements test_proc_audio_core_additional.py and
test_proc_audio_helper_contract_additional.py with wide-frame descriptor
routing, FFT parameter fallback, frequency-floor filtering, partial-metric
export assembly, and deterministic synthetic-signal edge cases.

No production code changes. No real audio corpus, GUI, plotting, or batch runs.
"""

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

import proc_audio as PA
from proc_audio import AudioProcessor


def _harmonic_df(freqs: list[float], amps: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": amps})


# ---------------------------------------------------------------------------
# 1. FFT / STFT parameter handling
# ---------------------------------------------------------------------------


def test_get_actual_n_fft_from_freqs_reflects_padded_fft_bins() -> None:
    ap = AudioProcessor()
    ap.freqs = np.linspace(0.0, 22050.0, 4097)
    assert ap._get_actual_n_fft() == 8192


def test_get_actual_n_fft_fallback_uses_n_fft_times_zero_padding() -> None:
    ap = AudioProcessor()
    ap.n_fft = 2048
    ap.zero_padding = 2
    assert ap._get_actual_n_fft() == 4096


def test_fft_analysis_caps_window_for_short_signal_and_actual_n_fft_matches_stft() -> None:
    ap = AudioProcessor()
    ap.sr = 44100
    ap.n_fft = 4096
    ap.hop_length = 1024
    ap.window = "hann"
    ap.y = np.sin(2.0 * np.pi * 440.0 * np.arange(500, dtype=float) / 44100.0)
    ap.fft_analysis(zero_padding=1)
    assert ap.sr == 44100
    assert ap.n_fft == 4096
    assert ap._get_actual_n_fft() == 256
    assert ap.S is not None and ap.S.shape[0] == 129


def test_fft_analysis_rejects_missing_audio_data() -> None:
    ap = AudioProcessor()
    with pytest.raises(ValueError, match="Audio data not loaded"):
        ap.fft_analysis()


# ---------------------------------------------------------------------------
# 2. Synthetic signal edge cases (normalization)
# ---------------------------------------------------------------------------


def test_normalize_level_all_zeros_returns_finite_zeros() -> None:
    y = np.zeros(128, dtype=float)
    out = PA._normalize_level(y)
    assert out.dtype == y.dtype
    assert np.all(out == 0.0)
    assert np.isfinite(out).all()


def test_normalize_level_preserves_float32_dtype() -> None:
    y = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    out = PA._normalize_level(y, target_rms_db=-20.0)
    assert out.dtype == np.float32


def test_normalize_level_constant_signal_hits_target_rms() -> None:
    y = np.full(256, 0.01, dtype=float)
    out = PA._normalize_level(y, target_rms_db=-20.0)
    rms = float(np.sqrt(np.mean(out**2)))
    assert rms == pytest.approx(10 ** (-20.0 / 20.0), rel=1e-6)


def test_normalize_level_does_not_mutate_input_array() -> None:
    y = np.array([0.2, -0.1, 0.05], dtype=float)
    snap = y.copy()
    _ = PA._normalize_level(y)
    np.testing.assert_array_equal(y, snap)


def test_extract_amplitude_column_coerces_nan_to_zero_but_preserves_inf() -> None:
    df = pd.DataFrame({"Amplitude": [1.0, float("nan"), float("inf")]})
    out = PA._extract_amplitude_column(df)
    assert out[0] == pytest.approx(1.0)
    assert out[1] == pytest.approx(0.0)
    assert out[2] == float("inf")


# ---------------------------------------------------------------------------
# 3. Descriptor routing / partial-metric assembly
# ---------------------------------------------------------------------------


def test_partial_metric_sums_continuous_uses_energy_scalars_not_dataframe_rows() -> None:
    ap = AudioProcessor()
    ap.weight_function = "linear"
    ap.harmonic_energy_sum = 2.0
    ap.inharmonic_energy_sum = 1.0
    ap.subbass_energy_sum = 0.5
    harm = _harmonic_df([9999.0], [999.0])
    h, i, s, t = ap._partial_metric_sums_for_metrics_export(harm, pd.DataFrame(), pd.DataFrame())
    assert (h, i, s, t) == (2.0, 1.0, 0.5, 3.5)


def test_partial_metric_sums_discrete_keeps_h_i_s_bands_separate() -> None:
    ap = AudioProcessor()
    ap.weight_function = "d3"
    h_df = _harmonic_df([100.0], [1.0])
    i_df = _harmonic_df([1500.0], [0.5])
    s_df = _harmonic_df([50.0], [0.25])
    h, i, s, t = ap._partial_metric_sums_for_metrics_export(h_df, i_df, s_df)
    assert h > 0.0
    assert i > 0.0
    assert s > 0.0
    assert t == pytest.approx(h + i + s, rel=1e-12)


def test_partial_metric_sums_discrete_missing_frequency_column_yields_zero_band() -> None:
    ap = AudioProcessor()
    ap.weight_function = "d3"
    no_freq = pd.DataFrame({"Amplitude": [1.0]})
    h, i, s, t = ap._partial_metric_sums_for_metrics_export(no_freq, pd.DataFrame(), pd.DataFrame())
    assert (h, i, s, t) == (0.0, 0.0, 0.0, 0.0)


def test_apply_density_metric_sdm_vector_linear_uses_power_vector_not_amplitudes() -> None:
    from density import apply_density_metric

    ap = AudioProcessor()
    ap.weight_function = "linear"
    camp = np.array([2.0, 1.0])
    cpow = np.array([4.0, 1.0])
    out = ap._apply_density_metric_sdm_vector(camp, np.array([100.0, 200.0]), cpow)
    assert out == pytest.approx(float(apply_density_metric(cpow, "linear", normalize=False)))
    assert out != pytest.approx(float(apply_density_metric(camp, "linear", normalize=False)))


def test_apply_density_metric_sdm_vector_discrete_uses_amplitudes_with_frequencies() -> None:
    ap = AudioProcessor()
    ap.weight_function = "d3"
    camp = np.array([1.0, 0.5])
    cpow = np.array([999.0, 999.0])
    out = ap._apply_density_metric_sdm_vector(camp, np.array([1000.0, 15000.0]), cpow)
    from density import apply_density_metric

    expected = float(
        apply_density_metric(
            camp,
            weight_function="d3",
            normalize=False,
            frequencies=np.array([1000.0, 15000.0]),
        )
    )
    assert out == pytest.approx(expected, rel=1e-12)
    assert out != pytest.approx(float(apply_density_metric(cpow, "d3", normalize=False)), rel=1e-6)


def test_component_energy_ratio_triple_prefers_primary_over_component_fallback() -> None:
    ap = AudioProcessor()
    ap.harmonic_energy_ratio = 0.70
    ap.inharmonic_energy_ratio = 0.20
    ap.subbass_energy_ratio = 0.10
    ap.component_harmonic_energy_ratio = 0.99
    trip, primary_missing, fb_missing = ap._component_energy_ratio_triple_for_pie()
    assert trip == (0.70, 0.20, 0.10)
    assert primary_missing == []
    assert fb_missing == []


def test_component_amplitude_mass_triple_prefers_named_amplitude_sums() -> None:
    ap = AudioProcessor()
    ap.harmonic_amplitude_sum = 2.0
    ap.inharmonic_amplitude_sum = 1.0
    ap.subbass_amplitude_sum = 0.5
    ap.linear_sum_amplitude_harmonic = 999.0
    triple, basis, gaps, tech = ap._component_amplitude_mass_triple_for_pie()
    assert triple == (2.0, 1.0, 0.5)
    assert basis == "harmonic_amplitude_sum"
    assert tech == "harmonic_amplitude_sum_triple"
    assert gaps == []


# ---------------------------------------------------------------------------
# 4. Frequency-floor / body-band filtering (diagnostic vs density paths)
# ---------------------------------------------------------------------------


def test_dataframe_for_density_frequency_floor_filters_without_mutating_input() -> None:
    ap = AudioProcessor()
    ap.freq_min = 50.0
    ap.subfundamental_guard_valid = True
    ap.adaptive_subfundamental_cutoff_hz = 80.0
    src = _harmonic_df([30.0, 60.0, 100.0], [1.0, 2.0, 3.0])
    snap = src.copy()
    out = ap._dataframe_for_density_frequency_floor(src)
    pd.testing.assert_frame_equal(src, snap)
    assert out["Frequency (Hz)"].tolist() == [100.0]


def test_apply_density_floor_trims_filtered_list_only_not_complete_list() -> None:
    ap = AudioProcessor()
    ap.freq_min = 50.0
    ap.subfundamental_guard_valid = True
    ap.adaptive_subfundamental_cutoff_hz = 80.0
    df = _harmonic_df([30.0, 60.0, 100.0], [1.0, 2.0, 3.0])
    ap.filtered_list_df = df.copy()
    ap.complete_list_df = df.copy()
    ap._apply_density_relevant_frequency_floor_to_filtered_list()
    assert ap.filtered_list_df["Frequency (Hz)"].tolist() == [100.0]
    assert len(ap.complete_list_df) == 3


# ---------------------------------------------------------------------------
# 5. Temporal evolution / energy verification (synthetic STFT)
# ---------------------------------------------------------------------------


def test_calculate_temporal_evolution_single_frame_returns_zero_flux() -> None:
    freqs = np.array([0.0, 1000.0])
    S_mag = np.array([[1.0], [0.5]])
    times = np.array([0.0])
    out = PA._calculate_temporal_evolution(S_mag, times, freqs, sr=44100)
    assert out["spectral_flux"] == 0.0
    assert out["attack_time"] == pytest.approx(0.0)


def test_calculate_temporal_evolution_detects_positive_flux_on_rising_frame() -> None:
    freqs = np.array([100.0, 200.0])
    S_mag = np.array([[0.0, 1.0], [0.0, 0.5]])
    times = np.array([0.0, 0.01])
    out = PA._calculate_temporal_evolution(S_mag, times, freqs, sr=44100)
    assert out["spectral_flux"] > 0.0


# ---------------------------------------------------------------------------
# 6. Determinism / idempotence
# ---------------------------------------------------------------------------


def test_partial_metric_sums_and_sdm_helpers_are_deterministic() -> None:
    ap = AudioProcessor()
    ap.weight_function = "d3"
    h_df = _harmonic_df([100.0, 200.0], [1.0, 0.5])
    i_df = _harmonic_df([1500.0], [0.3])
    first = ap._partial_metric_sums_for_metrics_export(h_df, i_df, pd.DataFrame())
    second = ap._partial_metric_sums_for_metrics_export(h_df, i_df, pd.DataFrame())
    assert first == second
    camp = np.array([1.0, 0.5])
    cpow = np.array([1.0, 0.25])
    fhz = np.array([100.0, 200.0])
    a = ap._apply_density_metric_sdm_vector(camp, fhz, cpow)
    b = ap._apply_density_metric_sdm_vector(camp, fhz, cpow)
    assert a == b


def test_descriptor_routing_helpers_do_not_mutate_attached_scalars() -> None:
    ap = AudioProcessor()
    ap.harmonic_energy_ratio = 0.7
    ap.inharmonic_energy_ratio = 0.2
    ap.subbass_energy_ratio = 0.1
    snap = (ap.harmonic_energy_ratio, ap.inharmonic_energy_ratio, ap.subbass_energy_ratio)
    _ = ap._component_energy_ratio_triple_for_pie()
    assert (ap.harmonic_energy_ratio, ap.inharmonic_energy_ratio, ap.subbass_energy_ratio) == snap

    ap2 = AudioProcessor()
    ap2.harmonic_amplitude_sum = 2.0
    ap2.inharmonic_amplitude_sum = 1.0
    ap2.subbass_amplitude_sum = 0.5
    snap2 = deepcopy((ap2.harmonic_amplitude_sum, ap2.inharmonic_amplitude_sum, ap2.subbass_amplitude_sum))
    _ = ap2._component_amplitude_mass_triple_for_pie()
    assert (ap2.harmonic_amplitude_sum, ap2.inharmonic_amplitude_sum, ap2.subbass_amplitude_sum) == snap2
