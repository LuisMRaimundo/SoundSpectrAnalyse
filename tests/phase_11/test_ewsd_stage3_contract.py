from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.ewsd_research_integration import merge_ewsd_stage3
from tools.ewsd_stage3_contract import (
    STAGE3_STATUS_DEGRADED,
    STAGE3_STATUS_FAILED,
    STAGE3_STATUS_OK,
    EwsdWorkbooksNotFound,
    assess_stage3_merge_result,
    build_stage3_diagnostics,
    enforce_fail_closed,
)
from tools.ewsd_stage3_contract import Stage3MergeResult


def test_assess_stage3_ok_when_all_merged() -> None:
    sd = pd.DataFrame(
        {
            "Note": ["D3"],
            "ewsd_merge_status": ["merged_individual_exact"],
            "ewsd_primary_analysis_eligible": [True],
        }
    )
    assert (
        assess_stage3_merge_result(sd, include_ewsd=True, global_status="merged", messages=[])
        == STAGE3_STATUS_OK
    )


def test_assess_stage3_failed_when_no_workbooks() -> None:
    sd = pd.DataFrame({"Note": ["D3"], "ewsd_merge_status": ["no_per_note_workbooks_found"]})
    assert (
        assess_stage3_merge_result(
            sd,
            include_ewsd=True,
            global_status="no_per_note_workbooks_found",
            messages=["skipped"],
        )
        == STAGE3_STATUS_FAILED
    )


def test_fail_closed_raises_on_missing_workbooks() -> None:
    diag, summary = build_stage3_diagnostics(
        pd.DataFrame({"Note": ["D3"], "ewsd_merge_status": ["no_per_note_workbooks_found"]}),
        analysis_root="/tmp",
        frequency_ceiling_hz=20000.0,
        n_workbooks=0,
    )
    result = Stage3MergeResult(
        pd.DataFrame({"Note": ["D3"]}),
        diag,
        summary,
        STAGE3_STATUS_FAILED,
        ("no workbooks",),
    )
    with pytest.raises(EwsdWorkbooksNotFound):
        enforce_fail_closed(result)


def test_merge_stage3_degraded_without_workbooks(tmp_path: Path) -> None:
    compiled = tmp_path / "compiled_density_metrics.xlsx"
    density = pd.DataFrame({"Note": ["D3"], "density_frequency_ceiling_hz": [20000.0]})
    with pd.ExcelWriter(compiled, engine="openpyxl") as writer:
        density.to_excel(writer, sheet_name="Density_Metrics", index=False)
    sd = pd.DataFrame({"Note": ["D3"], "density_metric_raw": [1.0]})
    warnings: list[str] = []
    result = merge_ewsd_stage3(sd, density, compiled, warnings, fail_closed=False)
    assert result.status == STAGE3_STATUS_FAILED
    assert "EWSD_score_total" in result.spectral_density_metrics.columns
    assert not result.diagnostics.empty


def test_research_export_writes_stage3_diagnostics_sheet(tmp_path: Path) -> None:
    from tools import export_research_density_workbook as research_export

    compiled = tmp_path / "compiled_density_metrics.xlsx"
    density = pd.DataFrame(
        {
            "Note": ["D3"],
            "density_metric_raw": [0.5],
            "density_frequency_ceiling_hz": [20000.0],
        }
    )
    with pd.ExcelWriter(compiled, engine="openpyxl") as writer:
        density.to_excel(writer, sheet_name="Density_Metrics", index=False)

    out = tmp_path / "compiled_density_metrics_research.xlsx"
    research_export.export_research_workbook(
        input_path=compiled,
        output_path=out,
        overwrite=True,
        no_charts=True,
        include_ewsd=True,
        ewsd_fail_closed=False,
    )
    sheets = pd.ExcelFile(out).sheet_names
    assert "Stage3_Diagnostics" in sheets
