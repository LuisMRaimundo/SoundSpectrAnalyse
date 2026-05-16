# -*- coding: utf-8 -*-
"""Low-frequency / subfundamental policy and export semantics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from density import (  # noqa: E402
    aggregate_low_frequency_residual_peak_power,
    aggregate_subbass_noise_peak_power,
)
from low_frequency_policy import (  # noqa: E402
    calculate_adaptive_subfundamental_cutoff_hz,
    calculate_subfundamental_margin_percent,
    classify_low_frequency_row,
)


def test_adaptive_subfundamental_cutoff_220_hz() -> None:
    guard = calculate_adaptive_subfundamental_cutoff_hz(220.0, min_floor_hz=20.0)
    assert guard["subfundamental_guard_valid"]
    assert guard["subfundamental_margin_percent"] == pytest.approx(15.0)
    assert guard["percentage_subfundamental_cutoff_hz"] == pytest.approx(187.0)
    assert guard["adaptive_subfundamental_cutoff_hz"] == pytest.approx(187.0)
    assert guard["effective_subfundamental_margin_percent"] == pytest.approx(15.0)
    assert guard["subfundamental_cutoff_selected_by"] == "percentage_subfundamental_cutoff_hz"
    assert guard["subfundamental_guard_policy"] == "f0_adaptive_register_margin"
    assert guard["low_frequency_policy_version"] == "dc_removed_adaptive_subfundamental_guard_v1"
    assert guard["min_floor_hz"] == pytest.approx(20.0)
    assert guard["max_fraction_of_f0"] == pytest.approx(0.95)


def test_subfundamental_leakage_override_220_hz() -> None:
    """Finite leakage guard above the percentage line raises the final cutoff and lowers effective margin."""
    g = calculate_adaptive_subfundamental_cutoff_hz(
        220.0, min_floor_hz=20.0, leakage_guard_cutoff_hz=194.277
    )
    assert g["percentage_subfundamental_cutoff_hz"] == pytest.approx(187.0)
    assert g["adaptive_subfundamental_cutoff_hz"] == pytest.approx(194.277)
    assert g["effective_subfundamental_margin_percent"] == pytest.approx(100.0 * (1.0 - 194.277 / 220.0))
    assert g["effective_subfundamental_margin_percent"] == pytest.approx(11.692272727272727, rel=1e-9)
    assert g["subfundamental_cutoff_selected_by"] == "leakage_guard_cutoff_hz"


def test_effective_subfundamental_margin_matches_cutover_f0() -> None:
    for f0, floor, leak in (
        (330.0, 20.0, None),
        (82.4, 20.0, 70.0),
        (1000.0, 5.0, None),
    ):
        g = calculate_adaptive_subfundamental_cutoff_hz(f0, min_floor_hz=floor, leakage_guard_cutoff_hz=leak)
        ad = float(g["adaptive_subfundamental_cutoff_hz"])
        ff = float(g["f0_final_hz"])
        eff = float(g["effective_subfundamental_margin_percent"])
        assert eff == pytest.approx(100.0 * (1.0 - ad / ff))


def test_subfundamental_max_fraction_cap_wins_selected_by() -> None:
    """When raw_max exceeds 0.95*f0, the cap binds and ``selected_by`` reports it."""
    g = calculate_adaptive_subfundamental_cutoff_hz(220.0, min_floor_hz=20.0, leakage_guard_cutoff_hz=215.0)
    assert g["adaptive_subfundamental_cutoff_hz"] == pytest.approx(220.0 * 0.95)
    assert g["subfundamental_cutoff_selected_by"] == "max_fraction_of_f0_cap"


def test_adaptive_subfundamental_cutoff_high_register_above_800_hz() -> None:
    guard = calculate_adaptive_subfundamental_cutoff_hz(932.33, min_floor_hz=20.0)
    assert guard["subfundamental_guard_valid"]
    assert guard["adaptive_subfundamental_cutoff_hz"] > 800.0
    assert guard["subfundamental_guard_policy"] == "f0_adaptive_register_margin"
    assert guard["low_frequency_policy_version"] == "dc_removed_adaptive_subfundamental_guard_v1"


def test_margin_percent_brackets() -> None:
    assert calculate_subfundamental_margin_percent(50.0) == pytest.approx(35.0)
    assert calculate_subfundamental_margin_percent(90.0) == pytest.approx(25.0)
    assert calculate_subfundamental_margin_percent(200.0) == pytest.approx(15.0)
    assert calculate_subfundamental_margin_percent(400.0) == pytest.approx(10.0)


def test_classify_low_frequency_row_buckets() -> None:
    assert (
        classify_low_frequency_row(
            25.0,
            dc_floor_hz=30.0,
            physical_low_band_upper_hz=200.0,
            adaptive_subfundamental_cutoff_hz=187.0,
        )
        == "dc_or_subaudible_residual"
    )
    assert (
        classify_low_frequency_row(
            100.0,
            dc_floor_hz=30.0,
            physical_low_band_upper_hz=200.0,
            adaptive_subfundamental_cutoff_hz=187.0,
        )
        == "subfundamental_residual"
    )
    assert (
        classify_low_frequency_row(
            195.0,
            dc_floor_hz=30.0,
            physical_low_band_upper_hz=200.0,
            adaptive_subfundamental_cutoff_hz=187.0,
        )
        == "physical_low_frequency_residual"
    )


def test_aggregate_subbass_is_deprecated_alias() -> None:
    h = pd.DataFrame({"Frequency (Hz)": [440.0], "Amplitude": [1.0]})
    c = pd.DataFrame(
        {
            "Frequency (Hz)": [40.0, 50.0, 60.0],
            "Amplitude": [0.1, 0.5, 0.1],
        }
    )
    a = aggregate_low_frequency_residual_peak_power(c, h, subbass_hz=200.0, subbass_lower_hz=30.0)
    b = aggregate_subbass_noise_peak_power(c, h, subbass_hz=200.0, subbass_lower_hz=30.0)
    assert a == pytest.approx(b)


def test_sub_bass_export_semantics_columns(tmp_path: Path) -> None:
    from low_frequency_policy import classify_low_frequency_row as clf

    sub_df = pd.DataFrame(
        {
            "Frequency (Hz)": [45.0, 150.0],
            "Magnitude (dB)": [-20.0, -15.0],
            "Amplitude": [0.1, 0.2],
        }
    )
    _dc_lo = 30.0
    _phys_hi = 200.0
    _ad_cut = 187.0
    sb_rows = sub_df.copy()
    sb_rows.insert(0, "Component_Type", "low_frequency_residual_row")
    sb_rows.insert(1, "Classification_Level", "diagnostic_fixed_frequency_band_residual")
    _fq_sb = pd.to_numeric(sb_rows["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
    _lf_classes = [
        clf(
            float(f),
            dc_floor_hz=_dc_lo,
            physical_low_band_upper_hz=_phys_hi,
            adaptive_subfundamental_cutoff_hz=_ad_cut,
        )
        for f in _fq_sb
    ]
    sb_rows.insert(2, "Low_Frequency_Class", _lf_classes)
    sb_rows.insert(
        3,
        "Acoustic_Interpretation_Status",
        "diagnostic_low_frequency_residual_not_partial",
    )
    outp = tmp_path / "sub.xlsx"
    sb_rows.to_excel(outp, sheet_name="Sub-bass band", index=False)
    df = pd.read_excel(outp, sheet_name="Sub-bass band")
    assert "Low_Frequency_Class" in df.columns
    assert "Acoustic_Interpretation_Status" in df.columns
    assert not (df["Component_Type"] == "subbass_noise").any()
    assert (df["Acoustic_Interpretation_Status"] == "diagnostic_low_frequency_residual_not_partial").all()


def test_dc_removed_on_load(tmp_path: Path) -> None:
    from proc_audio import AudioProcessor  # noqa: E402

    sr = 8000
    t = np.linspace(0, 0.25, int(sr * 0.25), endpoint=False)
    y = 0.05 * np.sin(2 * np.pi * 440.0 * t) + 0.123
    path = tmp_path / "dc.wav"
    sf.write(str(path), y.astype(np.float32), sr)

    ap = AudioProcessor()
    ap.load_audio_files([str(path)])
    assert ap.dc_removal_applied is True
    assert ap.dc_offset_before_removal is not None
    assert abs(float(ap.dc_offset_after_removal or 0.0)) < abs(float(ap.dc_offset_before_removal or 1.0))

    y2, sr2, _note, _fp = ap.audio_data[0]
    assert abs(float(np.mean(y2))) < 1e-9


def test_adaptive_subfundamental_cutoff_values() -> None:
    g1 = calculate_adaptive_subfundamental_cutoff_hz(58.27047)
    assert g1["subfundamental_guard_valid"]
    assert np.isclose(g1["subfundamental_margin_percent"], 35.0)
    assert np.isclose(g1["adaptive_subfundamental_cutoff_hz"], 37.8758055)

    g2 = calculate_adaptive_subfundamental_cutoff_hz(116.887)
    assert np.isclose(g2["subfundamental_margin_percent"], 25.0)
    assert np.isclose(g2["adaptive_subfundamental_cutoff_hz"], 87.66525)

    g3 = calculate_adaptive_subfundamental_cutoff_hz(233.949)
    assert np.isclose(g3["subfundamental_margin_percent"], 15.0)
    assert np.isclose(g3["adaptive_subfundamental_cutoff_hz"], 198.85665)

    g4 = calculate_adaptive_subfundamental_cutoff_hz(465.447)
    assert np.isclose(g4["subfundamental_margin_percent"], 10.0)
    assert np.isclose(g4["adaptive_subfundamental_cutoff_hz"], 418.9023)


def test_per_note_metadata_exports_adaptive_cutoff(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    from proc_audio import AudioProcessor

    sr = 44100
    duration = 1.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    y = np.zeros_like(t)
    for k in range(1, 9):
        y += np.sin(2.0 * np.pi * (k * 440.0) * t)
    wav = tmp_path / "A4.wav"
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = 0.45 * y / peak
    soundfile.write(str(wav), y.astype(np.float32), sr, subtype="FLOAT")

    out_dir = tmp_path / "results"
    proc = AudioProcessor()
    proc.load_audio_files([str(wav)])
    proc.apply_filters_and_generate_data(
        freq_min=50.0,
        freq_max=12000.0,
        db_min=-90.0,
        db_max=0.0,
        window="blackmanharris",
        n_fft=8192,
        hop_length=1024,
        tolerance=10.0,
        use_adaptive_tolerance=True,
        results_directory=str(out_dir),
        dissonance_enabled=False,
        dissonance_curve=False,
        dissonance_scale=False,
        compare_models=False,
        harmonic_weight=0.5,
        inharmonic_weight=0.5,
        auto_model_weights_from_analysis=True,
        weight_function="linear",
        zero_padding=1,
        time_avg="mean",
        spectral_masking_enabled=False,
        tier="Tier_test_LF",
    )
    outp = out_dir / "A4" / "spectral_analysis.xlsx"
    assert outp.is_file()
    meta = pd.read_excel(outp, sheet_name="Analysis_Metadata")
    values = dict(zip(meta["Parameter"].astype(str), meta["Value"]))

    assert "adaptive_subfundamental_cutoff_hz" in values
    assert "subfundamental_margin_percent" in values
    assert "percentage_subfundamental_cutoff_hz" in values
    assert "effective_subfundamental_margin_percent" in values
    assert "subfundamental_cutoff_selected_by" in values
    assert "subfundamental_guard_valid" in values

    cutoff = float(values["adaptive_subfundamental_cutoff_hz"])
    assert np.isfinite(cutoff)
    assert cutoff > 0.0
    assert str(values["subfundamental_guard_policy"]) == "f0_adaptive_register_margin"

    dbg = pd.read_excel(outp, sheet_name="Debug_Counts")
    assert str(dbg["debug_counts_invariant_status"].iloc[0]) == "passed"
    rs = int(pd.to_numeric(dbg["residual_spectral_row_count"].iloc[0], errors="coerce"))
    nc = int(pd.to_numeric(dbg["nonharmonic_candidate_row_count"].iloc[0], errors="coerce"))
    rt = int(pd.to_numeric(dbg["retained_nonharmonic_peak_candidate_count"].iloc[0], errors="coerce"))
    ex = int(pd.to_numeric(dbg["exported_nonharmonic_peak_candidate_count"].iloc[0], errors="coerce"))
    assert rs >= nc >= rt == ex
    pl = int(pd.to_numeric(dbg["peaklist_nonharmonic_window_candidate_count"].iloc[0], errors="coerce"))
    assert pl >= 0
    ih_rows = pd.read_excel(outp, sheet_name="Inharmonic Spectrum")
    assert int(len(ih_rows)) == ex


def test_compiled_output_has_no_not_available_cutoff_when_f0_exists(tmp_path: Path) -> None:
    from compile_metrics import compile_density_metrics_with_pca

    root = tmp_path / "analysis_results"
    root.mkdir(parents=True, exist_ok=True)
    note_dir = root / "stem" / "A4"
    note_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "Density Metric": 1.0,
        "Spectral Density Metric": 0.95,
        "Filtered Density Metric": 0.9,
        "Spectral Entropy": 0.4,
        "weight_function": "linear",
        "Harmonic Partials sum": 1.0,
        "Inharmonic Partials sum": 0.2,
        "Sub-bass sum": 0.1,
        "Total sum": 1.3,
        "effective_partial_density": 1.5,
        "harmonic_energy_sum": 1.0,
        "inharmonic_energy_sum": 0.2,
        "subbass_energy_sum": 0.1,
        "total_component_energy": 1.3,
        "harmonic_energy_ratio": 0.7,
        "inharmonic_energy_ratio": 0.2,
        "subbass_energy_ratio": 0.1,
        "harmonic_order_count": 4,
        "component_harmonic_energy_ratio": 0.7,
        "component_inharmonic_energy_ratio": 0.2,
        "component_subbass_energy_ratio": 0.1,
        "f0_final_hz": 465.447,
    }
    pd.DataFrame([row]).to_excel(note_dir / "spectral_analysis.xlsx", sheet_name="Metrics", index=False)

    outp = root / "compiled_density_metrics.xlsx"
    compile_density_metrics_with_pca(
        folder_path=str(root),
        output_path=str(outp),
        file_pattern="spectral_analysis.xlsx",
        enable_pca_export=False,
    )
    df = pd.read_excel(outp, sheet_name="Canonical_Metrics")

    assert "f0_final_hz" in df.columns
    assert "adaptive_subfundamental_cutoff_hz" in df.columns
    assert "percentage_subfundamental_cutoff_hz" in df.columns
    assert "effective_subfundamental_margin_percent" in df.columns
    assert "subfundamental_cutoff_selected_by" in df.columns

    rows = df[pd.to_numeric(df["f0_final_hz"], errors="coerce").notna()]
    assert not rows.empty

    for value in rows["adaptive_subfundamental_cutoff_hz"]:
        assert str(value) != "not_available_at_compile_stage"
        assert np.isfinite(float(value))
        assert float(value) > 0.0


def test_compile_stage_can_derive_cutoff_from_f0() -> None:
    from compile_metrics import _ensure_adaptive_subfundamental_cutoff

    row: dict = {"f0_final_hz": 465.447}

    out = _ensure_adaptive_subfundamental_cutoff(row)

    assert np.isclose(out["adaptive_subfundamental_cutoff_hz"], 418.9023)
    assert out["subfundamental_margin_percent"] == 10.0
    assert out["percentage_subfundamental_cutoff_hz"] == pytest.approx(465.447 * 0.9)
    assert out["subfundamental_cutoff_selected_by"] == "percentage_subfundamental_cutoff_hz"
    assert out["effective_subfundamental_margin_percent"] == pytest.approx(
        100.0 * (1.0 - 418.9023 / 465.447)
    )
    assert out["subfundamental_guard_valid"] is True
    assert out["adaptive_subfundamental_cutoff_source"] == "derived_at_compile_stage_from_f0_final_hz"
    assert np.isfinite(float(out["min_floor_hz"]))
    assert float(out["max_fraction_of_f0"]) == pytest.approx(0.95)


def test_harmonic_density_candidates_respect_adaptive_cutoff(tmp_path: Path) -> None:
    soundfile = pytest.importorskip("soundfile")
    from proc_audio import AudioProcessor

    f0_hz = 58.27047
    sr = 44100
    duration = 1.0
    t = np.linspace(0.0, duration, int(sr * duration), endpoint=False)
    y = np.zeros_like(t)
    for k in range(1, 13):
        y += np.sin(2.0 * np.pi * (k * f0_hz) * t)
    wav = tmp_path / "A#1.wav"
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 0:
        y = 0.45 * y / peak
    soundfile.write(str(wav), y.astype(np.float32), sr, subtype="FLOAT")

    out_dir = tmp_path / "results_low"
    proc = AudioProcessor()
    proc.load_audio_files([str(wav)])
    proc.apply_filters_and_generate_data(
        freq_min=15.0,
        freq_max=8000.0,
        db_min=-96.0,
        db_max=0.0,
        window="blackmanharris",
        n_fft=8192,
        hop_length=1024,
        tolerance=15.0,
        use_adaptive_tolerance=True,
        results_directory=str(out_dir),
        dissonance_enabled=False,
        dissonance_curve=False,
        dissonance_scale=False,
        compare_models=False,
        harmonic_weight=0.5,
        inharmonic_weight=0.5,
        auto_model_weights_from_analysis=True,
        weight_function="linear",
        zero_padding=1,
        time_avg="mean",
        spectral_masking_enabled=False,
        tier="Tier_test_LF2",
    )
    assert proc.subfundamental_guard_valid is True
    cut = float(proc.adaptive_subfundamental_cutoff_hz)
    cand = getattr(proc, "harmonic_spectrum_candidates_df", None)
    assert isinstance(cand, pd.DataFrame) and not cand.empty
    fq = pd.to_numeric(cand["Frequency (Hz)"], errors="coerce")
    inc = cand["include_for_density"].astype(bool)
    assert (fq.loc[inc] >= cut - 1e-6).all()
