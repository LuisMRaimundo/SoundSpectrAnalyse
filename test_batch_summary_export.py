"""DEPRECATED tests: cover the removed slim batch summary export
(``batch_summary.xlsx`` / JSON rows) produced by the pre-refactor Stage 1 /
Batch layer. The new pipeline does not create that workbook at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "deprecated: removed batch_summary.xlsx export; new pipeline only "
        "emits per-note spectral_analysis.xlsx and a compiled workbook."
    )
)

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIO_PKG = REPO_ROOT / "audio_analysis"
sys.path.insert(0, str(AUDIO_PKG))

from batch_audio_analyzer import BATCH_SUMMARY_COLUMNS, BatchAudioAnalyzer  # noqa: E402


def _row(
    file_name: str,
    note: str,
    *,
    hp: float,
    ip: float,
    sp: float,
) -> dict:
    """Build a consistent batch row: masses scale 0–100, percents match masses / total."""
    h_mass = hp
    i_mass = ip
    s_mass = sp
    ti_mass = i_mass + s_mass
    tot = h_mass + i_mass + s_mass
    return {
        "success": True,
        "file_name": file_name,
        "note": note,
        "harmonic_power_mass": h_mass,
        "inharmonic_residual_power_mass": i_mass,
        "subbass_noise_power_mass": s_mass,
        "total_inharmonic_power_mass": ti_mass,
        "total_power_mass": tot,
        "harmonic_power_percent": hp,
        "inharmonic_residual_power_percent": ip,
        "subbass_noise_power_percent": sp,
        "total_inharmonic_power_percent": ip + sp,
    }


def test_batch_rows_to_summary_dataframe_fixed_columns() -> None:
    rows = [
        _row("b_B3.wav", "B3", hp=85.0, ip=14.0, sp=1.0),
        _row("a_A4.wav", "A4", hp=80.0, ip=15.0, sp=5.0),
    ]
    df = BatchAudioAnalyzer._batch_rows_to_summary_dataframe(rows)
    assert list(df.columns) == list(BATCH_SUMMARY_COLUMNS)
    assert len(df) == 2
    assert df.iloc[0]["note"] == "B3"
    assert df.iloc[1]["note"] == "A4"


def test_phase2_load_percentage_mapping_uses_slim_batch_summary(tmp_path: Path) -> None:
    sys.path.insert(0, str(REPO_ROOT))
    from pipeline_orchestrator_integrated import RobustOrchestrator

    xlsx = tmp_path / "batch_summary.xlsx"
    rows = [
        {
            "file_name": "piano_A4.wav",
            "note": "A4",
            "harmonic_power_mass": 60.0,
            "inharmonic_residual_power_mass": 35.0,
            "subbass_noise_power_mass": 5.0,
            "total_inharmonic_power_mass": 40.0,
            "total_power_mass": 100.0,
            "harmonic_power_percent": 60.0,
            "inharmonic_residual_power_percent": 35.0,
            "subbass_noise_power_percent": 5.0,
            "total_inharmonic_power_percent": 40.0,
        },
    ]
    pd.DataFrame(rows, columns=list(BATCH_SUMMARY_COLUMNS)).to_excel(
        xlsx, sheet_name="Batch Summary", index=False
    )
    super_path = AUDIO_PKG / "super_audio_analyzer.py"
    main_out = tmp_path / "main_out"
    orch = RobustOrchestrator(
        audio_files=[tmp_path / "placeholder.wav"],
        super_analyzer_path=super_path,
        batch_output_dir=tmp_path,
        main_analysis_output_dir=main_out,
        excel_summary_path=xlsx,
    )
    assert orch.load_percentage_mapping()
    assert "A4" in orch.percentage_mapping
    entry = orch.percentage_mapping["A4"]
    assert entry["harmonic_percentage"] == pytest.approx(60.0)
    assert entry["inharmonic_percentage"] == pytest.approx(35.0)
    assert entry.get("batch_harmonic_energy_ratio") == pytest.approx(0.6)
    assert entry.get("batch_total_inharmonic_energy_ratio") == pytest.approx(0.4)
