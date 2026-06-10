from __future__ import annotations

"""
Additional scientifically-motivated coverage for proc_audio.py.

Scope: stable audio-processing contracts that feed the density/acoustic
pipeline — module-level DSP helpers (window gains, physical amplitude
calibration, QIFFT parabolic refinement, robust f0 estimation, cents errors,
f0 prior correction, bin spacing, RMS level normalisation, amplitude-column
extraction, note naming) and lightweight ``AudioProcessor`` contracts
(note-name → Hz, filename → note, synthetic WAV loading with DC removal and
mono folding).

Explicitly avoided: GUI paths, plotting, batch orchestration, Excel export
formatting, and long end-to-end per-note runs (already exercised by
tests/perf and the phase_10/phase_11/acoustic_validity suites).

Exact assertions are used only where analytically canonical (parabolic
interpolation is exact on a parabola, Δf = sr/(N·zp), 1200 cents per octave,
equal-tempered note frequencies, the documented k-alignment algebra);
window/STFT-derived quantities use broad physical tolerances.
"""

import math
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import proc_audio as PA
from proc_audio import AudioProcessor


def _write_wav(
    path: Path,
    data: np.ndarray,
    *,
    sr: int = 44100,
    channels: int = 1,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(np.clip(data, -1.0, 1.0) * 32767.0, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())
    return path


def _sine(f0: float, *, sr: int = 44100, seconds: float = 0.5, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(sr * seconds), dtype=float) / sr
    return amp * np.sin(2.0 * np.pi * f0 * t)


# ---------------------------------------------------------------------------
# 1. Window gains and physical amplitude calibration
# ---------------------------------------------------------------------------

def test_window_gain_and_sum_relations() -> None:
    # Hann coherent gain is 1/2 (canonical); window sum = gain * N.
    n = 4096
    cg = PA._coherent_gain("hann", n)
    ws = PA._window_sum("hann", n)
    assert cg == pytest.approx(0.5, rel=1e-6)
    assert ws == pytest.approx(cg * n, rel=1e-9)
    # Unknown window names fall back to Hann in both helpers.
    assert PA._coherent_gain("bogus-window", n) == pytest.approx(cg, rel=1e-12)
    assert PA._window_sum("bogus-window", n) == pytest.approx(ws, rel=1e-12)


def test_physical_peak_amplitude_is_exact_window_sum_scaling() -> None:
    mag = np.array([1.0, 2.0, 0.5])
    out = PA.physical_peak_amplitude(mag, "hann", 4096, is_one_sided=True)
    expected = 2.0 * mag / PA._window_sum("hann", 4096)
    assert np.allclose(out, expected, rtol=1e-12)
    # Two-sided spectra use factor 1.0 (documented).
    out2 = PA.physical_peak_amplitude(mag, "hann", 4096, is_one_sided=False)
    assert np.allclose(out2, expected / 2.0, rtol=1e-12)


def test_amplitude_calibration_recovers_unit_sine_independent_of_n_fft() -> None:
    # Documented self-test contract: the physical amplitude of a unit sine is
    # recovered within ~0.5 dB (factor 0.944..1.059) across n_fft values.
    recovered = PA.audit_amplitude_calibration(n_fft_values=(1024, 4096))
    assert set(recovered.keys()) == {1024, 4096}
    for n_fft, amp in recovered.items():
        assert 10 ** (-0.5 / 20.0) <= amp <= 10 ** (0.5 / 20.0), (n_fft, amp)


def test_window_characteristics_are_physically_coherent() -> None:
    wc = PA._calculate_window_characteristics("hann", 2048)
    assert np.isfinite(wc["main_lobe_width"])
    assert 0.5 <= wc["main_lobe_width"] <= 4.0  # Hann -3 dB width ~1.4-2 bins
    assert np.isfinite(wc["side_lobe_level"])
    assert wc["side_lobe_level"] < wc["peak_level"]  # side lobes below peak
    # Unknown window names fall back to Hann characteristics.
    assert PA._calculate_window_characteristics("bogus", 2048) == wc


# ---------------------------------------------------------------------------
# 2. QIFFT parabolic refinement
# ---------------------------------------------------------------------------

def test_parabolic_peak_is_exact_on_a_parabola() -> None:
    # Quadratic interpolation through three samples of a parabola recovers
    # the vertex exactly: y(x) = 4 - (x - 1.25)^2.
    y = [4.0 - (x - 1.25) ** 2 for x in range(4)]
    xv, yv = PA._parabolic_peak(y, 1)
    assert xv == pytest.approx(1.25, abs=1e-12)
    assert yv == pytest.approx(4.0, abs=1e-12)


def test_parabolic_peak_edge_and_flat_guards() -> None:
    # Edge indices pass through unrefined.
    assert PA._parabolic_peak([3.0, 1.0], 0) == (0, 3.0)
    # Flat neighbourhood (zero curvature) passes through unrefined.
    assert PA._parabolic_peak([2.0, 2.0, 2.0], 1) == (1, 2.0)


# ---------------------------------------------------------------------------
# 3. Robust f0 estimation and prior correction
# ---------------------------------------------------------------------------

def test_robust_f0_recovers_truth_from_biased_seed() -> None:
    # Exact harmonic stack at 110 Hz with a sharp 112 Hz seed: the weighted
    # least-squares fit must return to the true fundamental with ~0 residual.
    fit = PA._estimate_f0_global_robust(
        np.array([110.0, 220.0, 330.0, 440.0]),
        np.array([1.0, 0.5, 0.3, 0.2]),
        initial_f0=112.0,
    )
    assert fit["f0_estimated"] == pytest.approx(110.0, rel=1e-9)
    assert fit["residual_std"] == pytest.approx(0.0, abs=1e-9)
    assert fit["n_harmonics_used"] == 4
    assert fit["fit_quality"] == pytest.approx(0.0, abs=1e-12)


def test_robust_f0_underdetermined_returns_seed_payload() -> None:
    fit = PA._estimate_f0_global_robust(np.array([110.0]), np.array([1.0]), initial_f0=112.0)
    assert fit["f0_estimated"] == 112.0
    assert fit["n_harmonics_used"] == 1
    assert fit["residual_std"] == 0.0


def test_nearest_cents_error_canonical_and_guards() -> None:
    assert PA._nearest_cents_error(220.0, 110.0) == pytest.approx(1200.0, rel=1e-12)
    assert PA._nearest_cents_error(110.0, 110.0) == pytest.approx(0.0, abs=1e-12)
    assert math.isnan(PA._nearest_cents_error(0.0, 110.0))
    assert math.isnan(PA._nearest_cents_error(110.0, -1.0))


@pytest.mark.parametrize(
    ("candidate", "prior", "expected_hz", "expected_ratio"),
    [
        (880.0, 440.0, 440.0, 0.5),    # octave confusion (2*f0)
        (146.67, 440.0, 440.0, 3.0),   # sub-harmonic confusion (f0/3)
        (440.0, 440.0, 440.0, 1.0),    # identity
    ],
)
def test_f0_prior_correction_resolves_integer_ratio_confusions(
    candidate: float, prior: float, expected_hz: float, expected_ratio: float
) -> None:
    out = PA._correct_f0_candidate_against_prior(candidate, prior)
    assert out["valid"] is True
    assert out["ratio_applied"] == expected_ratio
    assert out["corrected_hz"] == pytest.approx(expected_hz, rel=2e-4)
    assert out["cents_error"] is not None and out["cents_error"] < 5.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -100.0, "abc", None])
def test_f0_prior_correction_rejects_invalid_inputs(bad: object) -> None:
    out = PA._correct_f0_candidate_against_prior(bad, 440.0)  # type: ignore[arg-type]
    assert out["valid"] is False
    assert out["corrected_hz"] is None


# ---------------------------------------------------------------------------
# 4. Bin spacing and RMS level normalisation
# ---------------------------------------------------------------------------

def test_bin_spacing_canonical_formula() -> None:
    # Δf = sr / (n_fft * zero_padding), exactly.
    assert PA._calculate_bin_spacing(44100.0, 4096, 1) == pytest.approx(44100.0 / 4096.0, rel=1e-12)
    assert PA._calculate_bin_spacing(44100.0, 4096, 2) == pytest.approx(44100.0 / 8192.0, rel=1e-12)


def test_normalize_level_hits_target_rms_and_is_gain_invariant() -> None:
    y = _sine(440.0, amp=0.5)
    yn = PA._normalize_level(y, target_rms_db=-20.0)
    rms = float(np.sqrt(np.mean(yn**2)))
    assert rms == pytest.approx(10 ** (-20.0 / 20.0), rel=1e-6)  # 0.1
    # Gain invariance: a 20 dB louder copy normalises to the same signal.
    yn_loud = PA._normalize_level(10.0 * y, target_rms_db=-20.0)
    assert np.allclose(yn, yn_loud, rtol=1e-9, atol=1e-12)
    # Empty input passes through unchanged.
    empty = np.asarray([], dtype=float)
    assert PA._normalize_level(empty).size == 0


# ---------------------------------------------------------------------------
# 5. Amplitude-column extraction
# ---------------------------------------------------------------------------

def test_extract_amplitude_column_priority_and_db_conversion() -> None:
    # "Amplitude" has priority over dB columns.
    df = pd.DataFrame({"Amplitude": [1.0, 0.5], "Magnitude (dB)": [-100.0, -100.0]})
    assert np.allclose(PA._extract_amplitude_column(df), [1.0, 0.5])
    # dB-only table converts canonically: 0 dB -> 1.0, -20 dB -> 0.1.
    df_db = pd.DataFrame({"Magnitude (dB)": [0.0, -20.0]})
    assert np.allclose(PA._extract_amplitude_column(df_db), [1.0, 0.1], rtol=1e-12)
    # Empty / None inputs give an empty array.
    assert PA._extract_amplitude_column(pd.DataFrame()).size == 0
    assert PA._extract_amplitude_column(None).size == 0  # type: ignore[arg-type]
    # Fallback: first numeric column.
    df_fb = pd.DataFrame({"foo": [2.0, 3.0]})
    assert np.allclose(PA._extract_amplitude_column(df_fb), [2.0, 3.0])


# ---------------------------------------------------------------------------
# 6. Note naming (Hz -> name, name -> Hz)
# ---------------------------------------------------------------------------

def test_frequency_to_note_name_canonical_and_invalid() -> None:
    assert PA.frequency_to_note_name(440.0) == "A4 (+0.00 cents)"
    detuned = 440.0 * 2.0 ** (40.0 / 1200.0)
    assert PA.frequency_to_note_name(detuned) == "A4 (+40.00 cents)"
    for bad in (0.0, -1.0, float("nan"), float("inf"), "abc"):
        assert PA.frequency_to_note_name(bad) == ""  # type: ignore[arg-type]


def test_note_name_to_frequency_equal_temperament() -> None:
    ap = AudioProcessor()
    assert ap.calculate_fundamental_frequency("A4") == pytest.approx(440.0, rel=1e-9)
    assert ap.calculate_fundamental_frequency("C4") == pytest.approx(261.6256, rel=1e-4)
    # Enharmonic flats resolve to the same pitch class (Bb3 == A#3).
    assert ap.calculate_fundamental_frequency("Bb3") == pytest.approx(
        ap.calculate_fundamental_frequency("A#3"), rel=1e-12
    )
    # Numeric fast path, including decimal-comma input.
    assert ap.calculate_fundamental_frequency("440") == 440.0
    assert ap.calculate_fundamental_frequency("440,5") == 440.5
    # Invalid labels return the documented 0.0 sentinel.
    assert ap.calculate_fundamental_frequency("XYZ") == 0.0
    assert ap.calculate_fundamental_frequency("") == 0.0


def test_round_trip_name_frequency_name() -> None:
    ap = AudioProcessor()
    for note in ("A4", "C4", "G#2", "E5"):
        f = ap.calculate_fundamental_frequency(note)
        label = PA.frequency_to_note_name(f)
        assert label.startswith(note), (note, f, label)
        # The round trip lands within +/-0.01 cents of the nominal pitch.
        cents = float(label.split("(")[1].split(" ")[0])
        assert abs(cents) < 0.01


def test_extract_note_name_filename_patterns() -> None:
    ap = AudioProcessor()
    assert ap.extract_note_name("Viola_C4_mf.wav") == "C4"
    assert ap.extract_note_name(Path("A#3.wav")) == "A#3"
    assert ap.extract_note_name("noname.wav") is None


# ---------------------------------------------------------------------------
# 7. Batch alignment factor k (export bookkeeping algebra)
# ---------------------------------------------------------------------------

def test_batch_alignment_k_canonical_algebra() -> None:
    # k = min(1, min(s_h*(p_i+p_s)/p_h, s_h) / (s_ih+s_sb)):
    # with s_h=10, s_ih+s_sb=10, ratios 0.1+0.1 over 0.8 -> target 2.5 -> 0.25.
    assert PA.linear_export_batch_alignment_k(10.0, 5.0, 5.0, 0.8, 0.1, 0.1) == pytest.approx(0.25, rel=1e-12)
    # Already-aligned exports are not scaled (cap at 1).
    assert PA.linear_export_batch_alignment_k(10.0, 1.0, 0.0, 0.5, 0.4, 0.1) == 1.0


def test_batch_alignment_k_documented_fallbacks() -> None:
    # Missing/invalid harmonic ratio -> no scaling.
    assert PA.linear_export_batch_alignment_k(10.0, 5.0, 5.0, None, 0.1, 0.1) == 1.0
    assert PA.linear_export_batch_alignment_k(10.0, 5.0, 5.0, "abc", 0.1, 0.1) == 1.0
    assert PA.linear_export_batch_alignment_k(10.0, 5.0, 5.0, 0.0, 0.1, 0.1) == 1.0
    # Nothing to scale (no inharmonic/sub-bass mass) -> 1.0.
    assert PA.linear_export_batch_alignment_k(10.0, 0.0, 0.0, 0.8, 0.1, 0.1) == 1.0
    # Zero target budget (p_i = p_s = 0) -> full suppression.
    assert PA.linear_export_batch_alignment_k(10.0, 5.0, 5.0, 0.8, 0.0, 0.0) == 0.0
    # Missing p_i / p_s are treated as 0.0 each (documented).
    assert PA.linear_export_batch_alignment_k(10.0, 5.0, 5.0, 0.8, None, None) == 0.0


def test_energy_ratio_pie_values_contract() -> None:
    assert PA._energy_ratio_pie_values(0.7, 0.2, 0.1) == (0.7, 0.2, 0.1)
    # Missing harmonic or inharmonic ratio -> None (no pie).
    assert PA._energy_ratio_pie_values(None, 0.2, 0.1) is None
    assert PA._energy_ratio_pie_values(0.7, None, 0.1) is None
    # Missing sub-bass is treated as 0.0.
    assert PA._energy_ratio_pie_values(0.7, 0.2, None) == (0.7, 0.2, 0.0)
    # All-zero wedges -> None; negative values clip to 0.
    assert PA._energy_ratio_pie_values(0.0, 0.0, 0.0) is None
    assert PA._energy_ratio_pie_values(-0.5, 0.5, 0.0) == (0.0, 0.5, 0.0)


# ---------------------------------------------------------------------------
# 8. Synthetic WAV loading contracts
# ---------------------------------------------------------------------------

def test_load_mono_wav_preserves_rate_removes_dc_and_extracts_note(tmp_path: Path) -> None:
    sr = 44100
    y = _sine(261.63, sr=sr, seconds=0.5, amp=0.4) + 0.05  # deliberate DC offset
    wav = _write_wav(tmp_path / "Piano_C4_mf.wav", y, sr=sr)

    ap = AudioProcessor()
    ap.load_audio_files([str(wav)])
    assert len(ap.audio_data) == 1
    loaded, loaded_sr, note, path_str = ap.audio_data[0]
    assert loaded_sr == sr
    assert note == "C4"
    assert Path(path_str) == wav.resolve()
    assert loaded.ndim == 1 and loaded.size > 0
    assert np.all(np.isfinite(loaded))
    # DC removal contract: the loaded signal is zero-mean.
    assert float(np.mean(loaded)) == pytest.approx(0.0, abs=1e-9)
    assert ap.dc_removal_applied is True


def test_load_stereo_wav_folds_to_single_channel(tmp_path: Path) -> None:
    sr = 22050
    mono = _sine(440.0, sr=sr, seconds=0.25, amp=0.3)
    stereo = np.empty(mono.size * 2, dtype=float)
    stereo[0::2] = mono  # L
    stereo[1::2] = mono  # R (identical)
    wav = _write_wav(tmp_path / "A4_stereo.wav", stereo, sr=sr, channels=2)

    ap = AudioProcessor()
    ap.load_audio_files([str(wav)])
    assert len(ap.audio_data) == 1
    loaded, loaded_sr, note, _ = ap.audio_data[0]
    assert loaded_sr == sr
    assert note == "A4"
    # Current contract: stereo input is folded to a 1-D mono signal.
    assert loaded.ndim == 1
    assert loaded.size == mono.size


def test_load_missing_file_is_skipped_without_crash(tmp_path: Path) -> None:
    ap = AudioProcessor()
    ap.load_audio_files([str(tmp_path / "does_not_exist.wav")])
    assert ap.audio_data == []


def test_load_silent_wav_is_accepted_as_valid_zero_signal(tmp_path: Path) -> None:
    sr = 22050
    wav = _write_wav(tmp_path / "G3_silence.wav", np.zeros(sr // 4), sr=sr)
    ap = AudioProcessor()
    ap.load_audio_files([str(wav)])
    assert len(ap.audio_data) == 1
    loaded, _, note, _ = ap.audio_data[0]
    assert note == "G3"
    assert np.all(loaded == 0.0)


def test_repeated_loading_is_deterministic(tmp_path: Path) -> None:
    wav = _write_wav(tmp_path / "D4_det.wav", _sine(293.66, seconds=0.25), sr=44100)
    ap1 = AudioProcessor()
    ap1.load_audio_files([str(wav)])
    ap2 = AudioProcessor()
    ap2.load_audio_files([str(wav)])
    y1, sr1, n1, _ = ap1.audio_data[0]
    y2, sr2, n2, _ = ap2.audio_data[0]
    assert sr1 == sr2 and n1 == n2
    assert np.array_equal(y1, y2)
