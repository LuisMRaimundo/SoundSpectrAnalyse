from __future__ import annotations

from pathlib import Path

import pandas as pd

import compile_metrics


PHASE7_FIELDS = (
    "pure_observation_w_h",
    "pure_observation_w_i",
    "pure_observation_w_s",
    "component_strength_h",
    "component_strength_i",
    "component_strength_s",
    "legacy_component_strength_h_v55",
    "legacy_component_strength_i_v55",
    "legacy_component_strength_s_v55",
)


def _write_note_workbook(
    path: Path,
    *,
    obs_h: float,
    obs_i: float,
    obs_s: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    harmonic = pd.DataFrame(
        {
            "Frequency (Hz)": [220.0, 440.0],
            "Amplitude_raw": [1.0, 0.6],
            "Power_raw": [1.0, 0.36],
            "include_for_density": [True, True],
        }
    )
    inharmonic = pd.DataFrame(
        {"Frequency (Hz)": [330.0], "Amplitude_raw": [0.2], "Power_raw": [0.04]}
    )
    subbass = pd.DataFrame(
        {"Frequency (Hz)": [55.0], "Amplitude_raw": [0.1], "Power_raw": [0.01]}
    )
    metadata = pd.DataFrame(
        {
            "Parameter": [
                "analysis_schema_version",
                "component_harmonic_energy_ratio",
                "component_inharmonic_energy_ratio",
                "component_subbass_energy_ratio",
            ],
            "Value": [
                compile_metrics.EXPECTED_ANALYSIS_SCHEMA_VERSION,
                0.7,
                0.2,
                0.1,
            ],
        }
    )
    metrics = pd.DataFrame(
        {
            "obs_w_formula_version": ["v58_full_spectrum_region_energy_gate"],
            "pure_observation_w_h": [obs_h],
            "pure_observation_w_i": [obs_i],
            "pure_observation_w_s": [obs_s],
            "component_strength_h": [obs_h * 10.0],
            "component_strength_i": [obs_i * 10.0],
            "component_strength_s": [obs_s * 10.0],
            "legacy_component_strength_h_v55": [obs_h * 12.0],
            "legacy_component_strength_i_v55": [obs_i * 12.0],
            "legacy_component_strength_s_v55": [obs_s * 12.0],
            "harmonic_energy_sum": [1.0],
            "inharmonic_energy_sum": [0.04],
            "subbass_energy_sum": [0.01],
        }
    )

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        harmonic.to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        inharmonic.to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        subbass.to_excel(writer, sheet_name="Sub-bass band", index=False)
        metadata.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        metrics.to_excel(writer, sheet_name="Metrics", index=False)


def test_phase7_fields_exposed_in_compiled_density_and_diagnostic(tmp_path: Path) -> None:
    # Unit-scope compile plumbing test: synthetic per-note workbook fixtures
    # are injected directly (no Stage 1 audio run). Integration coverage for
    # on-disk Stage 1 -> Stage 2 propagation lives in tests/phase_7_1b/.
    _write_note_workbook(tmp_path / "C4" / "spectral_analysis.xlsx", obs_h=0.70, obs_i=0.20, obs_s=0.10)
    _write_note_workbook(tmp_path / "E4" / "spectral_analysis.xlsx", obs_h=0.65, obs_i=0.25, obs_s=0.10)
    _write_note_workbook(tmp_path / "G4" / "spectral_analysis.xlsx", obs_h=0.60, obs_i=0.30, obs_s=0.10)

    out_xlsx = tmp_path / "compiled_density_metrics.xlsx"
    _ = compile_metrics.compile_density_metrics_with_pca(
        folder_path=tmp_path,
        output_path=out_xlsx,
        file_pattern="spectral_analysis.xlsx",
        include_pca=False,
    )

    density_df = pd.read_excel(out_xlsx, sheet_name="Density_Metrics")
    diagnostic_df = pd.read_excel(out_xlsx, sheet_name="Diagnostic_Metrics")

    for col in PHASE7_FIELDS:
        assert col in density_df.columns
        assert pd.to_numeric(density_df[col], errors="coerce").notna().all()
        assert col in diagnostic_df.columns

    assert "obs_w_formula_version" in density_df.columns
    assert density_df["obs_w_formula_version"].astype(str).eq("v58_full_spectrum_region_energy_gate").all()
