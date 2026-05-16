"""DEPRECATED tests: cover the legacy ``Phase 1`` (super_analysis JSON)
precedence over ``Phase 2`` (spectral_analysis.xlsx) for compiled
``rolloff_compensated_*`` columns. The Stage 1 + Stage 2 pipeline still
honours super_analysis JSON sidecars when present but the per-note
spectral_analysis.xlsx is always emitted, so the explicit precedence
scenarios encoded here no longer exercise current code paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "deprecated: tests removed Stage 1 (super_analysis) vs Stage 2 "
        "(spectral_analysis.xlsx) explicit-precedence pathway."
    )
)

from compile_metrics import compile_density_metrics, validate_compiled_density_workbook


def _minimal_density_row_for_compile() -> dict:
    return {
        "Note": "A3",
        "weight_function": "linear",
        "Harmonic Partials sum": 1.0,
        "Inharmonic Partials sum": 0.5,
        "Sub-bass sum": 0.1,
        "Total sum": 1.6,
        "canonical_density_v5_adapted": 61.698906,
        "density_per_component": 61.698906 / 91.0,
        "effective_partial_density": 2.0,
        "harmonic_energy_sum": 1.0,
        "inharmonic_energy_sum": 0.5,
        "subbass_energy_sum": 0.1,
        "total_component_energy": 1.6,
        "harmonic_energy_ratio": 0.625,
        "inharmonic_energy_ratio": 0.3125,
        "subbass_energy_ratio": 0.0625,
        "harmonic_order_count": 91,
        "spectral_entropy": 0.3,
        "rolloff_compensated_harmonic_density": 61.698906,
        "rolloff_compensated_harmonic_density_alpha": 1.5,
        "rolloff_compensated_harmonic_density_component_count": 91,
        "rolloff_compensated_harmonic_density_status": "",
        "density_metric_per_harmonic": 0.5,
        "legacy_rolloff_compensated_density": 61.698906,
    }


def test_a_phase1_canonical_precedence_over_phase2_spectral(tmp_path: Path) -> None:
    note_dir = tmp_path / "IOWA_sG_arco_ff.A3_Sustains"
    note_dir.mkdir(parents=True)
    row = _minimal_density_row_for_compile()
    pd.DataFrame([row]).to_excel(note_dir / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False)
    super_payload = {
        "spectral_metrics": {
            "rolloff_compensated_harmonic_density": 157.606785,
            "rolloff_compensated_harmonic_density_alpha": 1.5,
            "rolloff_compensated_harmonic_density_component_count": 156,
            "rolloff_compensated_harmonic_density_status": "computed",
            "density_metric_per_harmonic": 1.01,
            "density_metric_normalized": 0.99,
            "harmonic_count": 52,
        }
    }
    (note_dir / "super_analysis_results.json").write_text(json.dumps(super_payload), encoding="utf-8")

    outp = tmp_path / "compiled_density_metrics_ff.xlsx"
    df = compile_density_metrics(
        tmp_path,
        output_path=outp,
        file_pattern="spectral_analysis.xlsx",
        enable_pca_export=False,
    )
    assert df is not None
    r0 = df.iloc[0]
    assert r0["rolloff_compensated_harmonic_density"] == pytest.approx(157.606785)
    assert r0["rolloff_compensated_harmonic_density_component_count"] == pytest.approx(156.0)
    assert str(r0["rolloff_compensated_harmonic_density_status"]).strip() == "computed"
    assert r0["rolloff_density_source_phase"] == "phase1_super_analysis"
    assert r0["rolloff_density_json_discovery_method"] == "same_directory_super_analysis_json"
    assert r0["rolloff_density_json_match_confidence"] == "high"
    assert r0["phase2_rolloff_compensated_harmonic_density"] == pytest.approx(61.698906)
    assert r0["canonical_density_v5_adapted"] == pytest.approx(61.698906)
    assert r0["density_normalized_global"] == pytest.approx(1.0)
    assert r0["density_metric_normalized"] == pytest.approx(1.0)
    assert r0["density_metric_per_harmonic"] == pytest.approx(1.01)
    assert validate_compiled_density_workbook(outp) == []
    dm = pd.read_excel(outp, sheet_name="Density_Metrics")
    assert "legacy_rolloff_compensated_density" not in dm.columns
    assert "phase2_rolloff_compensated_harmonic_density" not in dm.columns
    clean = outp.parent / f"{outp.stem}_clean{outp.suffix}"
    assert not clean.exists()


def test_b_no_silent_overwrite_without_explicit_flag(tmp_path: Path) -> None:
    """Valid Phase 1 canonical must win unless prefer_phase2_rolloff_density=True."""
    note_dir = tmp_path / "Note_B4"
    note_dir.mkdir(parents=True)
    pd.DataFrame([_minimal_density_row_for_compile()]).to_excel(
        note_dir / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False
    )
    (note_dir / "super_analysis_results.json").write_text(
        json.dumps(
            {
                "spectral_metrics": {
                    "rolloff_compensated_harmonic_density": 157.606785,
                    "rolloff_compensated_harmonic_density_status": "computed",
                    "rolloff_compensated_harmonic_density_component_count": 156,
                }
            }
        ),
        encoding="utf-8",
    )
    df = compile_density_metrics(tmp_path, output_path=None, file_pattern="spectral_analysis.xlsx")
    assert df is not None
    assert df.iloc[0]["rolloff_compensated_harmonic_density"] == pytest.approx(157.606785)


def test_b_explicit_prefer_phase2_uses_spectral_public(tmp_path: Path) -> None:
    note_dir = tmp_path / "Note_C5"
    note_dir.mkdir(parents=True)
    pd.DataFrame([_minimal_density_row_for_compile()]).to_excel(
        note_dir / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False
    )
    (note_dir / "super_analysis_results.json").write_text(
        json.dumps(
            {
                "spectral_metrics": {
                    "rolloff_compensated_harmonic_density": 157.606785,
                    "rolloff_compensated_harmonic_density_status": "computed",
                    "rolloff_compensated_harmonic_density_component_count": 156,
                }
            }
        ),
        encoding="utf-8",
    )
    df = compile_density_metrics(
        tmp_path,
        output_path=None,
        file_pattern="spectral_analysis.xlsx",
        prefer_phase2_rolloff_density=True,
    )
    assert df is not None
    assert df.iloc[0]["rolloff_compensated_harmonic_density"] == pytest.approx(61.698906)
    assert df.iloc[0]["rolloff_density_source_phase"] == "phase2_configuration_override"
    assert df.iloc[0]["phase1_rolloff_compensated_harmonic_density"] == pytest.approx(157.606785)


def test_c_missing_phase1_marks_phase2_fallback(tmp_path: Path) -> None:
    note_dir = tmp_path / "Note_D5"
    note_dir.mkdir(parents=True)
    pd.DataFrame([_minimal_density_row_for_compile()]).to_excel(
        note_dir / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False
    )
    df = compile_density_metrics(tmp_path, output_path=None, file_pattern="spectral_analysis.xlsx")
    assert df is not None
    assert df.iloc[0]["rolloff_density_source_phase"] == "phase2_spectral_analysis_fallback"
    assert df.iloc[0]["rolloff_density_json_discovery_method"] == "not_found"


def test_d_finite_rolloff_gets_nonempty_status(tmp_path: Path) -> None:
    note_dir = tmp_path / "Note_E6"
    note_dir.mkdir(parents=True)
    row = _minimal_density_row_for_compile()
    row["rolloff_compensated_harmonic_density_status"] = ""
    pd.DataFrame([row]).to_excel(note_dir / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False)
    df = compile_density_metrics(tmp_path, output_path=None, file_pattern="spectral_analysis.xlsx")
    assert df is not None
    st = str(df.iloc[0]["rolloff_compensated_harmonic_density_status"]).strip()
    assert st == "computed"
    assert df.iloc[0]["rolloff_density_json_discovery_method"] == "not_found"


def test_separated_batch_results_and_analysis_results_layout(tmp_path: Path) -> None:
    """Real layout: Phase 1 under batch_results/<indexed_audio>/; Phase 2 under analysis_results/<audio>/<note>/."""
    root = tmp_path
    batch_dir = root / "batch_results" / "07_IOWA_Flute.mf_A3_Sustains_Sustains"
    batch_dir.mkdir(parents=True)
    super_payload = {
        "metadata": {"audio_file": str(root / "dummy" / "IOWA_Flute.mf_A3.wav")},
        "spectral_metrics": {
            "rolloff_compensated_harmonic_density": 157.606785,
            "rolloff_compensated_harmonic_density_alpha": 1.5,
            "rolloff_compensated_harmonic_density_component_count": 156,
            "rolloff_compensated_harmonic_density_status": "computed",
            "harmonic_count": 52,
        },
    }
    (batch_dir / "super_analysis_results.json").write_text(json.dumps(super_payload), encoding="utf-8")

    note_dir = root / "analysis_results" / "IOWA_Flute.mf_A3_Sustains_Sustains" / "A3"
    note_dir.mkdir(parents=True)
    row = _minimal_density_row_for_compile()
    row["Note"] = "A3"
    pd.DataFrame([row]).to_excel(note_dir / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False)

    df = compile_density_metrics(root, output_path=None, file_pattern="spectral_analysis.xlsx")
    assert df is not None
    r0 = df.iloc[0]
    assert r0["rolloff_compensated_harmonic_density"] == pytest.approx(157.606785)
    assert r0["rolloff_compensated_harmonic_density_component_count"] == pytest.approx(156.0)
    assert str(r0["rolloff_compensated_harmonic_density_status"]).strip() == "computed"
    assert r0["rolloff_density_source_phase"] == "phase1_super_analysis"
    assert r0["rolloff_density_json_discovery_method"] == "batch_results_stem_normalized_match"
    assert r0["rolloff_density_json_match_confidence"] == "medium"
    assert "super_analysis_results.json" in str(r0["rolloff_density_source_file"]).replace("\\", "/")


def test_e_clean_export_forbidden_tokens_and_legacy(tmp_path: Path) -> None:
    note_dir = tmp_path / "Note_F6"
    note_dir.mkdir(parents=True)
    row = _minimal_density_row_for_compile()
    row["note_name"] = row["Note"]
    row["Nota"] = row["Note"]
    pd.DataFrame([row]).to_excel(note_dir / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False)
    (note_dir / "super_analysis_results.json").write_text(
        json.dumps(
            {
                "spectral_metrics": {
                    "rolloff_compensated_harmonic_density": 157.606785,
                    "rolloff_compensated_harmonic_density_status": "computed",
                    "rolloff_compensated_harmonic_density_component_count": 156,
                }
            }
        ),
        encoding="utf-8",
    )
    outp = tmp_path / "out.xlsx"
    df = compile_density_metrics(
        tmp_path, output_path=outp, file_pattern="spectral_analysis.xlsx", enable_pca_export=False
    )
    assert df is not None
    clean = outp.parent / f"{outp.stem}_clean{outp.suffix}"
    assert not clean.exists()
    dmc = pd.read_excel(outp, sheet_name="Density_Metrics")
    cols = [str(c).lower() for c in dmc.columns]
    joined = " ".join(cols)
    for bad in ("legacy_rolloff", "spurious", "gold-standard", "gold standard", "more accurate"):
        assert bad not in joined
    assert "note_name" not in dmc.columns


def test_f_energy_denominators_separate_sums(tmp_path: Path) -> None:
    """Musical-band H+I and global H+I+S each sum to ~100 on their own rows (no cross-mixing in assertions)."""
    note_dir = tmp_path / "Note_G4"
    note_dir.mkdir(parents=True)
    row = _minimal_density_row_for_compile()
    row["harmonic_energy_percentage_musical_band"] = 60.0
    row["inharmonic_energy_percentage_musical_band"] = 40.0
    row["harmonic_energy_percentage_global"] = 55.0
    row["inharmonic_energy_percentage_global"] = 30.0
    row["subbass_energy_percentage_global"] = 15.0
    pd.DataFrame([row]).to_excel(note_dir / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False)
    df = compile_density_metrics(tmp_path, output_path=None, file_pattern="spectral_analysis.xlsx")
    assert df is not None
    r = df.iloc[0]
    assert float(r["harmonic_energy_percentage_musical_band"]) + float(r["inharmonic_energy_percentage_musical_band"]) == pytest.approx(100.0)
    assert (
        float(r["harmonic_energy_percentage_global"])
        + float(r["inharmonic_energy_percentage_global"])
        + float(r["subbass_energy_percentage_global"])
        == pytest.approx(100.0)
    )
