from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from density import (  # noqa: E402
    compute_harmonic_effective_power_density,
    compute_harmonic_occupancy_ratio,
    compute_residual_log_frequency_occupancy,
)
from harmonic_alignment import compute_harmonic_alignment_metrics  # noqa: E402
from proc_audio import AudioProcessor  # noqa: E402
from acoustic_density_core import compute_acoustic_density_descriptors  # noqa: E402


def _harmonic_df(f0: float, partial_count: int, *, amp_scale: float = 1.0) -> pd.DataFrame:
    rows = []
    for n in range(1, partial_count + 1):
        rows.append(
            {
                "Frequency (Hz)": float(f0 * n),
                "Amplitude": float((1.0 / n) * amp_scale),
                "Harmonic Number": n,
                "include_for_density": True,
                "local_peak_valid": True,
                "SNR_dB": 20.0,
                "SNR Threshold (dB)": 3.0,
            }
        )
    return pd.DataFrame(rows)


def _peaks_from_harmonics(
    f0: float,
    n_harmonics: int,
    *,
    amp_scale: float = 1.0,
    rolloff_exp: float = 1.0,
    add_noise: bool = False,
    noise_amp: float = 1e-3,
    noise_n: int = 24,
    noise_fmin: float = 2500.0,
    noise_fmax: float = 5000.0,
) -> pd.DataFrame:
    freqs = []
    amps = []
    for n in range(1, n_harmonics + 1):
        freqs.append(float(f0 * n))
        amps.append(float(amp_scale / (n ** rolloff_exp)))
    if add_noise:
        nf = np.geomspace(noise_fmin, noise_fmax, noise_n)
        freqs.extend([float(x) for x in nf])
        amps.extend([float(noise_amp)] * len(nf))
    return pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": amps})


def test_pure_sine_low_occupancy_low_entropy_proxy_and_near_zero_residual_occupancy() -> None:
    h = _harmonic_df(220.0, 1)
    occ = compute_harmonic_occupancy_ratio(h, f0_hz=220.0, max_frequency_hz=220.0 * 20.0)
    assert occ["harmonic_occupancy_ratio"] < 0.10
    ent_proxy = compute_harmonic_effective_power_density(h)["harmonic_effective_power_density_normalized_by_harmonic_count"]
    assert ent_proxy == 1.0
    residual = compute_residual_log_frequency_occupancy(pd.DataFrame({"Frequency (Hz)": []}))
    assert residual["residual_log_frequency_occupancy_status"] == "no_data"


def test_more_partials_increase_harmonic_occupancy_and_effective_density() -> None:
    max_f = 220.0 * 20.0
    occ = []
    ed = []
    for n in (5, 10, 20):
        h = _harmonic_df(220.0, n)
        occ.append(compute_harmonic_occupancy_ratio(h, f0_hz=220.0, max_frequency_hz=max_f)["harmonic_occupancy_ratio"])
        ed.append(compute_harmonic_effective_power_density(h)["harmonic_effective_power_density"])
    assert occ[0] < occ[1] <= occ[2]
    assert ed[0] < ed[1] <= ed[2]


def test_transposed_equivalent_harmonic_shape_keeps_pitch_normalized_occupancy_stable() -> None:
    h1 = _harmonic_df(220.0, 10)
    h2 = _harmonic_df(440.0, 10)
    o1 = compute_harmonic_occupancy_ratio(h1, f0_hz=220.0, max_frequency_hz=220.0 * 20.0)["harmonic_occupancy_ratio"]
    o2 = compute_harmonic_occupancy_ratio(h2, f0_hz=440.0, max_frequency_hz=440.0 * 20.0)["harmonic_occupancy_ratio"]
    assert abs(float(o1) - float(o2)) < 1e-9


def test_broadband_residual_increases_log_frequency_occupancy() -> None:
    base = pd.DataFrame({"Frequency (Hz)": [500.0, 1000.0, 2000.0]})
    noisy = pd.DataFrame({"Frequency (Hz)": np.geomspace(80.0, 8000.0, 48)})
    o_base = compute_residual_log_frequency_occupancy(base)["residual_log_frequency_occupancy"]
    o_noisy = compute_residual_log_frequency_occupancy(noisy)["residual_log_frequency_occupancy"]
    assert float(o_noisy) > float(o_base)


def test_detuned_partials_raise_alignment_error_without_false_occupancy_gain() -> None:
    f0 = 220.0
    harm = _harmonic_df(f0, 8)
    detuned = harm.copy()
    detuned["Frequency (Hz)"] = detuned["Frequency (Hz)"] * np.power(2.0, 10.0 / 1200.0)
    ok = compute_harmonic_alignment_metrics(f0, harm, max_frequency_hz=f0 * 12)
    bad = compute_harmonic_alignment_metrics(f0, detuned, max_frequency_hz=f0 * 12)
    assert float(bad["harmonic_alignment_mean_abs_error_cents"]) > float(ok["harmonic_alignment_mean_abs_error_cents"])
    o_ok = compute_harmonic_occupancy_ratio(harm, f0_hz=f0, max_frequency_hz=f0 * 12)["harmonic_occupancy_ratio"]
    o_bad = compute_harmonic_occupancy_ratio(detuned, f0_hz=f0, max_frequency_hz=f0 * 12)["harmonic_occupancy_ratio"]
    assert float(o_bad) <= float(o_ok)


def test_amplitude_scaling_keeps_occupancy_and_normalized_effective_density_stable() -> None:
    h1 = _harmonic_df(220.0, 12, amp_scale=1.0)
    h2 = _harmonic_df(220.0, 12, amp_scale=0.1)
    o1 = compute_harmonic_occupancy_ratio(h1, f0_hz=220.0, max_frequency_hz=220.0 * 20.0)["harmonic_occupancy_ratio"]
    o2 = compute_harmonic_occupancy_ratio(h2, f0_hz=220.0, max_frequency_hz=220.0 * 20.0)["harmonic_occupancy_ratio"]
    d1 = compute_harmonic_effective_power_density(h1)["harmonic_effective_power_density_normalized_by_harmonic_count"]
    d2 = compute_harmonic_effective_power_density(h2)["harmonic_effective_power_density_normalized_by_harmonic_count"]
    assert abs(float(o1) - float(o2)) < 1e-12
    assert abs(float(d1) - float(d2)) < 1e-12


def test_pipeline_path_exports_new_density_descriptors(tmp_path) -> None:
    sr = 22050
    t = np.linspace(0, 0.30, int(sr * 0.30), endpoint=False)
    y = np.sin(2 * np.pi * 220.0 * t).astype(np.float32)
    wav = tmp_path / "A3.wav"
    sf.write(str(wav), y, sr)
    out_dir = tmp_path / "out"

    ap = AudioProcessor()
    ap.load_audio_files([str(wav)])
    ap.apply_filters_and_generate_data(
        freq_min=50.0,
        freq_max=5000.0,
        db_min=-90.0,
        db_max=0.0,
        window="hann",
        n_fft=4096,
        hop_length=512,
        tolerance=10.0,
        use_adaptive_tolerance=True,
        results_directory=str(out_dir),
        dissonance_enabled=False,
        compare_models=False,
        harmonic_weight=0.5,
        inharmonic_weight=0.5,
        auto_model_weights_from_analysis=True,
        weight_function="linear",
        zero_padding=1,
        time_avg="mean",
        density_summation_mode="harmonic_only",
        harmonic_density_weight=1.0,
        inharmonic_density_weight=0.0,
        subbass_density_weight=0.0,
        density_salience_threshold_db=-55.0,
        density_frequency_ceiling_hz=3000.0,
        spectral_masking_enabled=False,
        tier="test",
    )
    xlsx = out_dir / "A3" / "spectral_analysis.xlsx"
    assert xlsx.is_file()
    metrics = pd.read_excel(xlsx, sheet_name="Metrics", engine="openpyxl")
    cols = set(metrics.columns)
    assert "harmonic_occupancy_ratio" in cols
    assert "residual_log_frequency_occupancy" in cols
    assert "f0_used_for_density_hz" in cols
    assert "acoustic_f0_status" in cols
    assert "body_weighted_effective_density" in cols
    assert "low_mid_energy_ratio" in cols
    assert "harmonic_body_density_normalized" in cols
    assert "residual_body_contribution_capped" in cols
    assert "final_note_density_count_based" in cols
    assert "final_note_density_salience_weighted" in cols
    assert "density_summation_mode" in cols
    assert "density_frequency_ceiling_hz" in cols
    row0 = metrics.iloc[0]
    assert str(row0["density_summation_mode"]) == "harmonic_only"
    assert abs(float(row0["density_frequency_ceiling_hz"]) - 3000.0) < 1e-9
    am = pd.read_excel(xlsx, sheet_name="Analysis_Metadata", engine="openpyxl")
    if {"Parameter", "Value"}.issubset(am.columns):
        kv = {str(k): v for k, v in zip(am["Parameter"], am["Value"], strict=False)}
        assert "density_summation_mode" in kv
        assert "density_frequency_ceiling_hz" in kv


def test_body_thickness_family_monotonic_and_gain_invariant() -> None:
    f0 = 220.0
    d5 = compute_acoustic_density_descriptors(_peaks_from_harmonics(f0, 5), f0_hz=f0, freq_max_hz=5000.0)
    d10 = compute_acoustic_density_descriptors(_peaks_from_harmonics(f0, 10), f0_hz=f0, freq_max_hz=5000.0)
    d20 = compute_acoustic_density_descriptors(_peaks_from_harmonics(f0, 20), f0_hz=f0, freq_max_hz=5000.0)
    assert float(d5["body_weighted_effective_density"]) < float(d10["body_weighted_effective_density"]) <= float(
        d20["body_weighted_effective_density"]
    )
    assert float(d5["harmonic_body_density_normalized"]) <= float(d10["harmonic_body_density_normalized"]) <= float(
        d20["harmonic_body_density_normalized"]
    )

    d10_scaled = compute_acoustic_density_descriptors(
        _peaks_from_harmonics(f0, 10, amp_scale=0.05), f0_hz=f0, freq_max_hz=5000.0
    )
    assert abs(float(d10["body_weighted_effective_density"]) - float(d10_scaled["body_weighted_effective_density"])) < 1e-9
    assert abs(float(d10["low_mid_energy_ratio"]) - float(d10_scaled["low_mid_energy_ratio"])) < 1e-9
    assert abs(float(d10["harmonic_body_density_normalized"]) - float(d10_scaled["harmonic_body_density_normalized"])) < 1e-9


def test_body_thickness_transposition_and_noise_robustness() -> None:
    low = compute_acoustic_density_descriptors(_peaks_from_harmonics(110.0, 12), f0_hz=110.0, freq_max_hz=5000.0)
    high = compute_acoustic_density_descriptors(_peaks_from_harmonics(220.0, 12), f0_hz=220.0, freq_max_hz=5000.0)
    assert float(low["body_weighted_effective_density"]) >= float(high["body_weighted_effective_density"]) * 0.8

    clean = compute_acoustic_density_descriptors(_peaks_from_harmonics(220.0, 10), f0_hz=220.0, freq_max_hz=5000.0)
    noisy = compute_acoustic_density_descriptors(
        _peaks_from_harmonics(220.0, 10, add_noise=True, noise_amp=1e-4),
        f0_hz=220.0,
        freq_max_hz=5000.0,
    )
    assert float(noisy["effective_partial_density"]) >= float(clean["effective_partial_density"])
    assert float(noisy["body_weighted_effective_density"]) <= float(clean["body_weighted_effective_density"]) + 1.0


def test_residual_body_contribution_capped_and_low_for_pure_sine() -> None:
    sine = compute_acoustic_density_descriptors(_peaks_from_harmonics(220.0, 1), f0_hz=220.0, freq_max_hz=5000.0)
    assert float(sine["effective_partial_density"]) <= 1.1
    assert float(sine["body_weighted_effective_density"]) <= 1.1

    residual_heavy = compute_acoustic_density_descriptors(
        _peaks_from_harmonics(
            220.0,
            6,
            add_noise=True,
            noise_amp=0.08,
            noise_n=64,
            noise_fmin=800.0,
            noise_fmax=5000.0,
        ),
        f0_hz=220.0,
        freq_max_hz=5000.0,
    )
    assert float(residual_heavy["residual_body_contribution"]) >= 0.0
    assert float(residual_heavy["residual_body_contribution_capped"]) <= 0.25 + 1e-12


def test_expected_harmonic_order_count_up_to_5000hz_decreases_with_higher_f0() -> None:
    low = compute_acoustic_density_descriptors(_peaks_from_harmonics(110.0, 30), f0_hz=110.0, freq_max_hz=5000.0)
    high = compute_acoustic_density_descriptors(_peaks_from_harmonics(220.0, 30), f0_hz=220.0, freq_max_hz=5000.0)
    assert int(low["expected_harmonic_order_count_up_to_5000hz"]) > int(high["expected_harmonic_order_count_up_to_5000hz"])


def test_salient_raw_count_decreases_with_transposition_while_coverage_can_stay_stable() -> None:
    low = compute_acoustic_density_descriptors(_peaks_from_harmonics(110.0, 30), f0_hz=110.0, freq_max_hz=5000.0)
    high = compute_acoustic_density_descriptors(_peaks_from_harmonics(220.0, 30), f0_hz=220.0, freq_max_hz=5000.0)
    assert float(low["salient_harmonic_order_count_up_to_5000hz"]) > float(high["salient_harmonic_order_count_up_to_5000hz"])
    assert 0.0 <= float(low["salient_harmonic_coverage_up_to_5000hz"]) <= 1.0 + 1e-12
    assert 0.0 <= float(high["salient_harmonic_coverage_up_to_5000hz"]) <= 1.0 + 1e-12


def test_peak_leakage_counts_single_harmonic_order_once() -> None:
    f0 = 220.0
    # Two peaks around harmonic order 4 should still count as one salient order.
    freqs = [220.0, 440.0, 880.0, 879.0, 881.0]
    amps = [1.0, 0.7, 0.4, 0.35, 0.33]
    df = pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": amps})
    d = compute_acoustic_density_descriptors(df, f0_hz=f0, freq_max_hz=5000.0)
    assert int(d["salient_harmonic_order_count_up_to_5000hz"]) == 3


def test_weak_noise_below_salience_threshold_does_not_increase_salient_count() -> None:
    clean = compute_acoustic_density_descriptors(_peaks_from_harmonics(220.0, 8), f0_hz=220.0, freq_max_hz=5000.0)
    noisy = compute_acoustic_density_descriptors(
        _peaks_from_harmonics(220.0, 8, add_noise=True, noise_amp=1e-5, noise_n=48),
        f0_hz=220.0,
        freq_max_hz=5000.0,
    )
    assert int(noisy["salient_harmonic_order_count_up_to_5000hz"]) == int(
        clean["salient_harmonic_order_count_up_to_5000hz"]
    )


def test_odd_harmonic_dominant_spectrum_has_odd_count_and_energy_ratio_greater_than_even() -> None:
    f0 = 147.0
    freqs = []
    amps = []
    for n in range(1, 21):
        freqs.append(float(n * f0))
        amps.append(1.0 / n if n % 2 == 1 else 0.08 / n)
    d = compute_acoustic_density_descriptors(
        pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": amps}),
        f0_hz=f0,
        freq_max_hz=5000.0,
    )
    assert int(d["salient_odd_harmonic_count_up_to_5000hz"]) > int(d["salient_even_harmonic_count_up_to_5000hz"])
    assert float(d["odd_even_harmonic_energy_ratio"]) > 1.0


def test_final_density_harmonic_only_mode_equals_salient_harmonic_order_count() -> None:
    d = compute_acoustic_density_descriptors(
        _peaks_from_harmonics(220.0, 10),
        f0_hz=220.0,
        freq_max_hz=5000.0,
        density_summation_mode="harmonic_only",
    )
    assert float(d["final_note_density_count_based"]) == float(d["salient_harmonic_order_count_up_to_5000hz"])


def test_final_density_weighted_count_mode_matches_component_weighted_sum() -> None:
    d = compute_acoustic_density_descriptors(
        _peaks_from_harmonics(220.0, 10, add_noise=True, noise_amp=0.02, noise_n=20, noise_fmin=400.0, noise_fmax=4500.0),
        f0_hz=220.0,
        freq_max_hz=5000.0,
        harmonic_density_weight=1.0,
        inharmonic_density_weight=0.5,
        subbass_density_weight=0.25,
    )
    expected = (
        1.0 * float(d["salient_harmonic_order_count_up_to_5000hz"])
        + 0.5 * float(d["salient_inharmonic_log_bin_count_up_to_5000hz"])
        + 0.25 * float(d["salient_subbass_particle_count"])
    )
    assert abs(float(d["final_note_density_count_based"]) - expected) < 1e-12


def test_salience_weighted_component_caps_single_strong_partial_at_one() -> None:
    d = compute_acoustic_density_descriptors(
        _peaks_from_harmonics(220.0, 1),
        f0_hz=220.0,
        freq_max_hz=5000.0,
        density_summation_mode="harmonic_only",
    )
    assert float(d["harmonic_density_component"]) <= 1.0 + 1e-12
    assert float(d["final_note_density_salience_weighted"]) <= 1.0 + 1e-12


def test_more_salient_harmonic_orders_increase_final_density() -> None:
    d5 = compute_acoustic_density_descriptors(_peaks_from_harmonics(220.0, 5), f0_hz=220.0, freq_max_hz=5000.0)
    d12 = compute_acoustic_density_descriptors(_peaks_from_harmonics(220.0, 12), f0_hz=220.0, freq_max_hz=5000.0)
    assert float(d12["final_note_density_count_based"]) > float(d5["final_note_density_count_based"])
    assert float(d12["final_note_density_salience_weighted"]) > float(d5["final_note_density_salience_weighted"])


def test_weak_noise_below_density_threshold_does_not_raise_final_density() -> None:
    clean = compute_acoustic_density_descriptors(_peaks_from_harmonics(220.0, 8), f0_hz=220.0, freq_max_hz=5000.0)
    noisy = compute_acoustic_density_descriptors(
        _peaks_from_harmonics(220.0, 8, add_noise=True, noise_amp=1e-6, noise_n=30),
        f0_hz=220.0,
        freq_max_hz=5000.0,
    )
    assert float(noisy["final_note_density_count_based"]) == float(clean["final_note_density_count_based"])


def test_leaked_peaks_in_same_harmonic_window_count_once_in_final_density() -> None:
    f0 = 220.0
    df = pd.DataFrame(
        {
            "Frequency (Hz)": [220.0, 440.0, 880.0, 879.0, 881.0, 1320.0],
            "Amplitude": [1.0, 0.7, 0.4, 0.35, 0.33, 0.28],
        }
    )
    d = compute_acoustic_density_descriptors(df, f0_hz=f0, freq_max_hz=5000.0, density_summation_mode="harmonic_only")
    assert int(d["salient_harmonic_order_count_up_to_5000hz"]) == 4
    assert float(d["final_note_density_count_based"]) == 4.0


def test_density_salience_threshold_changes_component_or_final_density() -> None:
    peaks = _peaks_from_harmonics(220.0, 10, add_noise=True, noise_amp=0.03, noise_n=40)
    d35 = compute_acoustic_density_descriptors(peaks, f0_hz=220.0, freq_max_hz=5000.0, density_salience_threshold_db=-35.0)
    d55 = compute_acoustic_density_descriptors(peaks, f0_hz=220.0, freq_max_hz=5000.0, density_salience_threshold_db=-55.0)
    changed = (
        abs(float(d55["final_note_density_salience_weighted"]) - float(d35["final_note_density_salience_weighted"])) > 1e-12
        or abs(float(d55["harmonic_density_component"]) - float(d35["harmonic_density_component"])) > 1e-12
        or abs(float(d55["inharmonic_density_component"]) - float(d35["inharmonic_density_component"])) > 1e-12
        or abs(float(d55["subbass_density_component"]) - float(d35["subbass_density_component"])) > 1e-12
    )
    assert changed


def test_density_ceiling_aliases_change_monotonically_with_higher_ceiling() -> None:
    peaks = _peaks_from_harmonics(146.8, 35, add_noise=True, noise_amp=0.01, noise_n=30)
    d3 = compute_acoustic_density_descriptors(peaks, f0_hz=146.8, freq_max_hz=12000.0, density_frequency_ceiling_hz=3000.0)
    d5 = compute_acoustic_density_descriptors(peaks, f0_hz=146.8, freq_max_hz=12000.0, density_frequency_ceiling_hz=5000.0)
    d8 = compute_acoustic_density_descriptors(peaks, f0_hz=146.8, freq_max_hz=12000.0, density_frequency_ceiling_hz=8000.0)
    assert int(d3["expected_harmonic_order_count_up_to_density_ceiling_hz"]) <= int(
        d5["expected_harmonic_order_count_up_to_density_ceiling_hz"]
    ) <= int(d8["expected_harmonic_order_count_up_to_density_ceiling_hz"])
    assert int(d3["salient_harmonic_order_count_up_to_density_ceiling_hz"]) <= int(
        d5["salient_harmonic_order_count_up_to_density_ceiling_hz"]
    ) <= int(d8["salient_harmonic_order_count_up_to_density_ceiling_hz"])
