from __future__ import annotations

"""
Fifth Phase 12 validation/status contract layer for compile_metrics.py.

Complements test_compile_metrics_workbook_contract_additional.py with additional
validate_compiled_density_workbook branches, export-status vs sheet consistency,
canonical/diagnostic/legacy routing under validation, and partial/malformed
workbook-like inputs.

No production code changes. No audio, GUI, plotting, or full compile pipeline.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import compile_metrics as cm


def _write_workbook(
    path: Path,
    *,
    density_df: pd.DataFrame | None = None,
    meta_rows: list[dict[str, object]] | None = None,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
    omit_density: bool = False,
    omit_metadata: bool = False,
) -> None:
    rows = meta_rows or [
        {"Parameter": "analysis_schema_version", "Value": cm.EXPECTED_ANALYSIS_SCHEMA_VERSION},
    ]
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        if not omit_density:
            (density_df if density_df is not None else pd.DataFrame([_density_row()])).to_excel(
                writer, sheet_name="Density_Metrics", index=False
            )
        if not omit_metadata:
            pd.DataFrame(rows).to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        for sheet_name, frame in (extra_sheets or {}).items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)


def _density_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "Note": "C4",
        "density_metric_raw": 1.25,
        "component_harmonic_energy_ratio": 0.70,
        "component_inharmonic_energy_ratio": 0.20,
        "component_subbass_energy_ratio": 0.10,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# 1. validate_compiled_density_workbook — additional branches
# ---------------------------------------------------------------------------


def test_validate_reports_missing_required_sheets(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    _write_workbook(path, omit_density=True)
    errs = cm.validate_compiled_density_workbook(path)
    assert any("Missing required sheet: Density_Metrics" in e for e in errs)

    path2 = tmp_path / "compiled2.xlsx"
    _write_workbook(path2, omit_metadata=True)
    errs2 = cm.validate_compiled_density_workbook(path2)
    assert any("Missing required sheet: Analysis_Metadata" in e for e in errs2)


def test_validate_accepts_structurally_empty_density_metrics_sheet(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    _write_workbook(path, density_df=pd.DataFrame(columns=["Note", "density_metric_raw"]))
    assert cm.validate_compiled_density_workbook(path) == []


def test_validate_rejects_disallowed_non_forbidden_column(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    row = _density_row()
    row["not_on_density_metrics_allow_list"] = 99
    _write_workbook(path, density_df=pd.DataFrame([row]))
    errs = cm.validate_compiled_density_workbook(path)
    assert any("disallowed column" in e and "not_on_density_metrics_allow_list" in e for e in errs)


def test_validate_rejects_energy_ratio_alias_sum_drift_on_density_metrics(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    df = pd.DataFrame(
        [
            {
                **_density_row(),
                "harmonic_energy_ratio": 0.80,
                "inharmonic_energy_ratio": 0.30,
                "subbass_energy_ratio": 0.10,
            }
        ]
    )
    _write_workbook(path, density_df=df)
    errs = cm.validate_compiled_density_workbook(path)
    assert any("Energy ratios do not sum to ~1" in e for e in errs)


def test_validate_rejects_non_finite_effective_partial_density(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    _write_workbook(
        path,
        density_df=pd.DataFrame([{**_density_row(), "effective_partial_density": float("inf")}]),
    )
    errs = cm.validate_compiled_density_workbook(path)
    assert any("effective_partial_density invalid" in e for e in errs)


def test_validate_rejects_all_nan_effective_partial_density(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    _write_workbook(
        path,
        density_df=pd.DataFrame([{**_density_row(), "effective_partial_density": float("nan")}]),
    )
    errs = cm.validate_compiled_density_workbook(path)
    assert any("effective_partial_density invalid (all NaN)" in e for e in errs)


def test_validate_rejects_density_normalized_global_above_one(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    _write_workbook(
        path,
        density_df=pd.DataFrame([{**_density_row(), "density_normalized_global": 1.5}]),
    )
    errs = cm.validate_compiled_density_workbook(path)
    assert any("density_normalized_global exceeds 1.0" in e for e in errs)


def test_validate_rejects_density_metric_normalized_outside_unit_interval(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    _write_workbook(
        path,
        density_df=pd.DataFrame([{**_density_row(), "density_metric_normalized": 1.2}]),
    )
    errs = cm.validate_compiled_density_workbook(path)
    assert any("density_metric_normalized outside [0, 1]" in e for e in errs)


@pytest.mark.parametrize(
    ("status_key", "status_value", "expected_fragment"),
    [
        ("dissonance_export_status", "exported", "Dissonance_Metrics is missing"),
        ("validation_export_status", "exported", "Validation_Metrics is missing"),
        ("pca_export_status", "exported", "PCA exported but missing sheet"),
    ],
)
def test_validate_metadata_export_claim_without_matching_sheet(
    tmp_path: Path,
    status_key: str,
    status_value: str,
    expected_fragment: str,
) -> None:
    path = tmp_path / "compiled.xlsx"
    meta = [
        {"Parameter": "analysis_schema_version", "Value": cm.EXPECTED_ANALYSIS_SCHEMA_VERSION},
        {"Parameter": status_key, "Value": status_value},
    ]
    _write_workbook(path, meta_rows=meta)
    errs = cm.validate_compiled_density_workbook(path)
    assert any(expected_fragment in e for e in errs)


def test_validate_pca_skipped_status_rejects_orphan_pca_sheets(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    meta = [
        {"Parameter": "analysis_schema_version", "Value": cm.EXPECTED_ANALYSIS_SCHEMA_VERSION},
        {"Parameter": "pca_export_status", "Value": "skipped: insufficient samples"},
    ]
    _write_workbook(
        path,
        meta_rows=meta,
        extra_sheets={"PCA_Scores": pd.DataFrame({"Note": ["C4"], "PC1": [0.1]})},
    )
    errs = cm.validate_compiled_density_workbook(path)
    assert any("PCA marked skipped but sheet present" in e for e in errs)


def test_validate_per_note_processing_metadata_ratio_sum_failure(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    pn = pd.DataFrame(
        [
            {
                "Note": "C4",
                "component_harmonic_energy_ratio": 0.80,
                "component_inharmonic_energy_ratio": 0.30,
                "component_subbass_energy_ratio": 0.10,
            }
        ]
    )
    _write_workbook(path, extra_sheets={"Per_Note_Processing_Metadata": pn})
    errs = cm.validate_compiled_density_workbook(path)
    assert any("Per_Note_Processing_Metadata: component H+I+S ratios do not sum" in e for e in errs)


def test_validate_is_deterministic_for_same_workbook(tmp_path: Path) -> None:
    path = tmp_path / "compiled.xlsx"
    _write_workbook(path)
    first = cm.validate_compiled_density_workbook(path)
    second = cm.validate_compiled_density_workbook(path)
    assert first == second


def test_validate_metadata_duplicate_parameter_keys_use_last_row_value(tmp_path: Path) -> None:
    """Analysis_Metadata dict uses last Parameter row; exported PCA claim must win."""
    path = tmp_path / "compiled.xlsx"
    meta = [
        {"Parameter": "analysis_schema_version", "Value": cm.EXPECTED_ANALYSIS_SCHEMA_VERSION},
        {"Parameter": "pca_export_status", "Value": "skipped"},
        {"Parameter": "pca_export_status", "Value": "exported"},
    ]
    _write_workbook(path, meta_rows=meta)
    errs = cm.validate_compiled_density_workbook(path)
    assert any("PCA exported but missing sheet" in e for e in errs)
    assert not any("PCA marked skipped but sheet present" in e for e in errs)


# ---------------------------------------------------------------------------
# 2. Export-status / column partition helpers
# ---------------------------------------------------------------------------


def test_slice_compiled_preserves_row_count_and_status_string_dtype() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4", "D4"],
            "qc_status": ["validated_pipeline", "warning_only"],
            "effective_partial_density": [2.0, 3.0],
            "density_metric_raw": [1.1, 1.2],
        }
    )
    for status in ("canonical", "diagnostic", "legacy"):
        out = cm._slice_compiled_df_by_status(df, status)
        assert len(out) == 2
    canonical = cm._slice_compiled_df_by_status(df, "canonical")
    assert canonical["qc_status"].dtype == object
    assert canonical["qc_status"].iloc[0] == "validated_pipeline"


def test_classify_keeps_strict_energy_aliases_diagnostic_not_canonical() -> None:
    assert cm._classify_compiled_column("harmonic_energy_ratio") == "diagnostic"
    assert cm._classify_compiled_column("component_harmonic_energy_ratio") == "canonical"
    assert cm._classify_compiled_column("Density Metric") == "legacy"


def test_slice_compiled_phase7_inharmonicity_stays_diagnostic_not_canonical() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "effective_partial_density": [2.0],
            "inharmonicity_coefficient_B": [1.2e-4],
            "inharmonicity_fit_status": ["fit_ok"],
            "Density Metric": [9.0],
        }
    )
    canonical = cm._slice_compiled_df_by_status(df, "canonical")
    diagnostic = cm._slice_compiled_df_by_status(df, "diagnostic")
    legacy = cm._slice_compiled_df_by_status(df, "legacy")

    assert "effective_partial_density" in canonical.columns
    assert "inharmonicity_coefficient_B" in diagnostic.columns
    assert "inharmonicity_coefficient_B" not in canonical.columns
    assert "Density Metric" in legacy.columns
    assert "Density Metric" not in canonical.columns


def test_slice_compiled_empty_and_metadata_only_inputs() -> None:
    assert cm._slice_compiled_df_by_status(pd.DataFrame(), "canonical").empty
    meta_only = pd.DataFrame({"Note": ["C4"], "qc_status": ["validated_pipeline"]})
    canonical = cm._slice_compiled_df_by_status(meta_only, "canonical")
    assert list(canonical.columns) == ["Note", "qc_status"]


def test_slice_compiled_duplicate_column_labels_preserve_current_order() -> None:
    """Current contract: duplicate labels are not deduplicated by the slicer."""
    df = pd.DataFrame([[1.0, 2.0]], columns=["effective_partial_density", "effective_partial_density"])
    df.insert(0, "Note", "C4")
    out = cm._slice_compiled_df_by_status(df, "canonical")
    assert out.columns.tolist().count("effective_partial_density") == 2
    assert out.iloc[0, 1] == pytest.approx(1.0)
    assert out.iloc[0, 2] == pytest.approx(2.0)


def test_slice_compiled_is_deterministic_and_non_mutating() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "effective_partial_density": [1.5],
            "harmonic_energy_ratio": [0.7],
            "batch_harmonic_energy_ratio": [0.4],
        }
    )
    snap = deepcopy(df)
    a = cm._slice_compiled_df_by_status(df, "diagnostic")
    b = cm._slice_compiled_df_by_status(df, "diagnostic")
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(df, snap)


# ---------------------------------------------------------------------------
# 3. Per-note schema assessor — duplicate metadata key semantics
# ---------------------------------------------------------------------------


def test_assess_per_note_workbook_schema_first_metadata_key_wins(tmp_path: Path) -> None:
    path = tmp_path / "note.xlsx"
    meta = pd.DataFrame(
        [
            {"Parameter": "analysis_schema_version", "Value": "stale_v1"},
            {"Parameter": "analysis_schema_version", "Value": cm.EXPECTED_ANALYSIS_SCHEMA_VERSION},
            {"Parameter": "model_weights_source", "Value": "current_analysis"},
            {"Parameter": "component_profile_source", "Value": "integrated_single_pass"},
            {"Parameter": "export_alignment_source", "Value": "disabled_integrated_single_pass"},
            {"Parameter": "export_alignment_factor", "Value": 1.0},
        ]
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        meta.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        pd.DataFrame({"Amplitude_raw": [1.0]}).to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        pd.DataFrame({"Amplitude_raw": [0.2]}).to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        pd.DataFrame({"Amplitude_raw": [0.1]}).to_excel(writer, sheet_name="Sub-bass band", index=False)

    result = cm.assess_per_note_workbook_schema(path)
    assert result["schema_version"] == "stale_v1"
    assert result["schema_ok"] is False


# ---------------------------------------------------------------------------
# 4. Write / validate interaction (minimal tmp_path)
# ---------------------------------------------------------------------------


def test_write_then_validate_minimal_workbook_export_status_consistency(tmp_path: Path) -> None:
    outp = tmp_path / "compiled.xlsx"
    raw = 1.23
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "effective_partial_density": [raw],
            "density_metric_raw": [raw],
            "__source_file_path": [str(tmp_path / "secret" / "spectral_analysis.xlsx")],
            "harmonic_density_component": [1.0],
            "inharmonic_density_component": [0.2],
            "subbass_density_component": [0.03],
        }
    )
    cm._write_compiled_excel(
        outp,
        df,
        metadata={"analysis_schema_version": cm.EXPECTED_ANALYSIS_SCHEMA_VERSION},
        compile_file_pattern="results/*/spectral_analysis.xlsx",
        input_schema_validation_status="validated",
    )
    meta = pd.read_excel(outp, sheet_name="Analysis_Metadata")
    flat = meta.iloc[0].to_dict()
    assert flat["export_schema_version"] == cm.EXPORT_SCHEMA_VERSION
    assert flat["density_formula_version"] == cm.DENSITY_FORMULA_VERSION
    assert str(flat.get("canonical_metrics_export_status", "")).startswith("exported")

    with pd.ExcelFile(outp) as xf:
        assert "Canonical_Metrics" in xf.sheet_names
        canonical = xf.parse("Canonical_Metrics")
        assert "__source_file_path" not in canonical.columns
        assert canonical["effective_partial_density"].iloc[0] == pytest.approx(raw)

    errs = cm.validate_compiled_density_workbook(outp)
    assert not any("Missing required sheet" in e for e in errs)


def test_satellite_diagnostic_sheet_does_not_satisfy_canonical_requirement(tmp_path: Path) -> None:
    """Diagnostic_Metrics presence must not mask missing canonical export metadata claims."""
    path = tmp_path / "compiled.xlsx"
    meta = [
        {"Parameter": "analysis_schema_version", "Value": cm.EXPECTED_ANALYSIS_SCHEMA_VERSION},
        {"Parameter": "canonical_metrics_export_status", "Value": "exported"},
    ]
    diagnostic = pd.DataFrame({"Note": ["C4"], "density_metric_raw": [1.2], "harmonic_energy_ratio": [0.7]})
    _write_workbook(path, extra_sheets={"Diagnostic_Metrics": diagnostic}, meta_rows=meta)
    with pd.ExcelFile(path) as xf:
        assert "Diagnostic_Metrics" in xf.sheet_names
        assert "Canonical_Metrics" not in xf.sheet_names
    errs = cm.validate_compiled_density_workbook(path)
    assert errs == [] or all("Canonical_Metrics" not in e for e in errs)
