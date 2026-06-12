from __future__ import annotations

"""
Sixth Phase 12 DataFrame assembly contract layer for compile_metrics.py.

Complements the five prior compile_metrics Phase 12 layers with wide synthetic
frame routing, duplicate-column safety, canonical/diagnostic/legacy assembly,
numeric preservation, and minimal write-path deduplication behavior.

No production code changes. No audio, GUI, plotting, or full compile pipeline.
"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import compile_metrics as cm


def _wide_row(note: str = "C4", **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "Note": note,
        "effective_partial_density": 3.0,
        "canonical_density_v5_adapted": 3.0,
        "density_metric_raw": 1.65,
        "density_metric_normalized": 0.8,
        "density_normalized_global": 0.75,
        "Density Metric": 30.0,
        "component_harmonic_energy_ratio": 0.70,
        "component_inharmonic_energy_ratio": 0.20,
        "component_subbass_energy_ratio": 0.10,
        "harmonic_energy_ratio": 0.99,
        "inharmonic_energy_ratio": 0.01,
        "subbass_energy_ratio": 0.00,
        "Harmonic Partials sum": 2.0,
        "Inharmonic Partials sum": 1.0,
        "Sub-bass sum": 0.5,
        "inharmonicity_coefficient_B": 1.2e-4,
        "inharmonicity_fit_status": "fit_ok",
        "obs_w_formula_version": "ewsd_v1",
        "obs_wS_artifact_flag": False,
        "batch_harmonic_energy_ratio": 0.05,
        "harmonic_amplitude_sum": 2.0,
        "qc_status": "validated_pipeline",
        "__source_file_path": r"C:\secret\note.xlsx",
    }
    base.update(overrides)
    return base


def _wide_frame(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(list(rows) if rows else [_wide_row(), _wide_row("D4", effective_partial_density=4.0, canonical_density_v5_adapted=4.0)])


# ---------------------------------------------------------------------------
# 1. Wide synthetic frame assembly and routing
# ---------------------------------------------------------------------------


def test_wide_frame_assembly_preserves_row_order_and_count() -> None:
    df = _wide_frame(_wide_row("A4"), _wide_row("B4"), _wide_row("C4"))
    out = cm._add_canonical_and_global_density_columns(df)
    assert list(out["Note"]) == ["A4", "B4", "C4"]
    assert len(out) == 3


def test_add_canonical_uses_legacy_density_metric_only_when_canonical_absent() -> None:
    df = pd.DataFrame({"Note": ["C4"], "Density Metric": [30.0]})
    out = cm._add_canonical_and_global_density_columns(df)
    assert out["canonical_density_v5_adapted"].iloc[0] == pytest.approx(3.0)


def test_add_canonical_preserves_existing_canonical_despite_conflicting_legacy_display() -> None:
    df = pd.DataFrame(
        {"Note": ["C4"], "canonical_density_v5_adapted": [2.5], "Density Metric": [99.0]}
    )
    out = cm._add_canonical_and_global_density_columns(df)
    assert out["canonical_density_v5_adapted"].iloc[0] == pytest.approx(2.5)


def test_wide_frame_slice_routes_metric_families_to_expected_partitions() -> None:
    df = _wide_frame()
    assembled = cm._add_canonical_and_global_density_columns(df)
    canonical = cm._slice_compiled_df_by_status(assembled, "canonical")
    diagnostic = cm._slice_compiled_df_by_status(assembled, "diagnostic")
    legacy = cm._slice_compiled_df_by_status(assembled, "legacy")

    assert "effective_partial_density" in canonical.columns
    assert "component_harmonic_energy_ratio" in canonical.columns
    assert "density_metric_raw" in diagnostic.columns
    assert "harmonic_energy_ratio" in diagnostic.columns
    assert "obs_w_formula_version" in diagnostic.columns
    assert "inharmonicity_coefficient_B" in diagnostic.columns
    assert "batch_harmonic_energy_ratio" in legacy.columns
    assert "Density Metric" in legacy.columns

    assert "Density Metric" not in canonical.columns
    assert "density_metric_raw" not in canonical.columns
    assert "obs_w_formula_version" not in canonical.columns
    assert "__source_file_path" not in canonical.columns
    assert "__source_file_path" not in diagnostic.columns


def test_weighted_assembly_uses_component_ratios_not_strict_aliases_on_wide_frame() -> None:
    df = _wide_frame(
        _wide_row(
            harmonic_energy_ratio=0.05,
            component_harmonic_energy_ratio=0.70,
        )
    )
    out = cm._add_canonical_and_global_density_columns(df)
    expected = 2.0 * 0.70 + 1.0 * 0.20 + 0.5 * 0.10
    assert out["density_metric_raw"].iloc[0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 2. Duplicate-column safety
# ---------------------------------------------------------------------------


def test_split_strict_alias_separates_identical_alias_values_from_main() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "component_harmonic_energy_ratio": [0.70],
            "harmonic_energy_ratio": [0.70],
            "subbass_energy_ratio": [0.10],
        }
    )
    main, aliases = cm._split_strict_alias_columns(df)
    assert "harmonic_energy_ratio" not in main.columns
    assert "subbass_energy_ratio" not in aliases.columns or "subbass_energy_ratio" in aliases.columns
    assert main["component_harmonic_energy_ratio"].iloc[0] == pytest.approx(0.70)
    assert aliases["harmonic_energy_ratio"].iloc[0] == pytest.approx(0.70)


def test_split_strict_alias_preserves_conflicting_canonical_and_alias_values() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "component_inharmonic_energy_ratio": [0.20],
            "inharmonic_energy_ratio": [0.88],
        }
    )
    main, aliases = cm._split_strict_alias_columns(df)
    assert main["component_inharmonic_energy_ratio"].iloc[0] == pytest.approx(0.20)
    assert aliases["inharmonic_energy_ratio"].iloc[0] == pytest.approx(0.88)


def test_slice_compiled_duplicate_canonical_labels_keep_both_columns() -> None:
    df = pd.DataFrame([[3.0, 9.0]], columns=["effective_partial_density", "effective_partial_density"])
    df.insert(0, "Note", "C4")
    out = cm._slice_compiled_df_by_status(df, "canonical")
    assert out.columns.tolist().count("effective_partial_density") == 2
    assert out.iloc[0, 1] == pytest.approx(3.0)
    assert out.iloc[0, 2] == pytest.approx(9.0)


def test_write_path_deduplicates_duplicate_labels_first_occurrence_wins(tmp_path: Path) -> None:
    outp = tmp_path / "compiled.xlsx"
    df = pd.DataFrame([[3.0, 99.0]], columns=["effective_partial_density", "effective_partial_density"])
    df.insert(0, "Note", "C4")
    df["density_metric_raw"] = 1.25
    df["harmonic_density_component"] = 1.0
    df["inharmonic_density_component"] = 0.2
    df["subbass_density_component"] = 0.03
    cm._write_compiled_excel(outp, df, metadata={})
    canonical = pd.read_excel(outp, sheet_name="Canonical_Metrics")
    assert canonical["effective_partial_density"].iloc[0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 3. Numeric preservation and type stability
# ---------------------------------------------------------------------------


def test_assembly_preserves_finite_canonical_numeric_values() -> None:
    df = _wide_frame(_wide_row(effective_partial_density=2.75, canonical_density_v5_adapted=2.75))
    snap_canon = float(df["canonical_density_v5_adapted"].iloc[0])
    snap_eff = float(df["effective_partial_density"].iloc[0])
    out = cm._add_canonical_and_global_density_columns(df)
    assert out["canonical_density_v5_adapted"].iloc[0] == pytest.approx(snap_canon)
    assert out["effective_partial_density"].iloc[0] == pytest.approx(snap_eff)
    assert pd.api.types.is_numeric_dtype(out["canonical_density_v5_adapted"])


def test_drop_dead_columns_removes_nan_only_diagnostic_but_keeps_canonical() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "effective_partial_density": [2.0],
            "harmonic_amplitude_sum": [np.nan],
            "unused_diagnostic": [np.nan],
        }
    )
    out = cm._drop_dead_columns(df)
    assert "effective_partial_density" in out.columns
    assert "harmonic_amplitude_sum" not in out.columns
    assert "unused_diagnostic" not in out.columns


def test_assembly_keeps_status_columns_as_object_strings() -> None:
    df = _wide_frame(_wide_row(qc_status="validated_pipeline", inharmonicity_fit_status="fit_ok"))
    out = cm._add_canonical_and_global_density_columns(df)
    canonical = cm._slice_compiled_df_by_status(out, "canonical")
    assert canonical["qc_status"].dtype == object
    assert canonical["qc_status"].iloc[0] == "validated_pipeline"


def test_assembly_with_nan_diagnostic_columns_does_not_change_canonical_density() -> None:
    df = _wide_frame(
        _wide_row(
            harmonic_amplitude_sum=np.nan,
            obs_wS_artifact_flag=np.nan,
            effective_partial_density=2.2,
        )
    )
    before = float(df["effective_partial_density"].iloc[0])
    out = cm._add_canonical_and_global_density_columns(df)
    assert out["effective_partial_density"].iloc[0] == pytest.approx(before)


# ---------------------------------------------------------------------------
# 4. Idempotence, determinism, non-mutation
# ---------------------------------------------------------------------------


def test_add_canonical_and_global_density_columns_is_idempotent() -> None:
    df = _wide_frame()
    once = cm._add_canonical_and_global_density_columns(df)
    twice = cm._add_canonical_and_global_density_columns(once)
    for col in ("canonical_density_v5_adapted", "density_normalized_global", "density_metric_raw"):
        if col in once.columns and col in twice.columns:
            pd.testing.assert_series_equal(
                once[col],
                twice[col],
                check_names=True,
                check_exact=False,
                rtol=0.0,
                atol=1e-12,
            )


def test_double_slice_canonical_partition_is_stable() -> None:
    df = _wide_frame()
    assembled = cm._add_canonical_and_global_density_columns(df)
    first = cm._slice_compiled_df_by_status(assembled, "canonical")
    second = cm._slice_compiled_df_by_status(first, "canonical")
    pd.testing.assert_frame_equal(first, second)


def test_wide_frame_assembly_pipeline_does_not_mutate_input() -> None:
    df = _wide_frame()
    snap = deepcopy(df)
    _ = cm._add_canonical_and_global_density_columns(df)
    _ = cm._slice_compiled_df_by_status(df, "canonical")
    _ = cm._split_strict_alias_columns(df)
    pd.testing.assert_frame_equal(df, snap)


def test_split_and_slice_sequence_is_deterministic() -> None:
    df = _wide_frame()
    main_a, alias_a = cm._split_strict_alias_columns(df)
    canon_a = cm._slice_compiled_df_by_status(main_a, "canonical")
    main_b, alias_b = cm._split_strict_alias_columns(df)
    canon_b = cm._slice_compiled_df_by_status(main_b, "canonical")
    pd.testing.assert_frame_equal(canon_a, canon_b)
    pd.testing.assert_frame_equal(alias_a, alias_b)


# ---------------------------------------------------------------------------
# 5. Workbook-write preparation (minimal, non-duplicative)
# ---------------------------------------------------------------------------


def test_write_wide_frame_keeps_legacy_off_canonical_sheet(tmp_path: Path) -> None:
    outp = tmp_path / "compiled.xlsx"
    df = _wide_frame(
        _wide_row(
            effective_partial_density=2.8,
            canonical_density_v5_adapted=2.8,
            **{"Density Metric": 888.0},
        )
    )
    cm._write_compiled_excel(outp, df, metadata={})
    with pd.ExcelFile(outp) as xf:
        canonical = xf.parse("Canonical_Metrics")
        legacy = xf.parse("Legacy_Compatibility")
    assert canonical["effective_partial_density"].iloc[0] == pytest.approx(2.8)
    assert "Density Metric" not in canonical.columns
    assert "Density Metric" in legacy.columns
    assert legacy["Density Metric"].iloc[0] == pytest.approx(888.0)


def test_write_wide_frame_internal_path_not_on_public_sheets(tmp_path: Path) -> None:
    outp = tmp_path / "compiled.xlsx"
    secret = str(tmp_path / "secret" / "spectral_analysis.xlsx")
    df = _wide_frame(_wide_row(__source_file_path=secret))
    cm._write_compiled_excel(outp, df, metadata={})
    with pd.ExcelFile(outp) as xf:
        for sheet in ("Canonical_Metrics", "Diagnostic_Metrics", "Legacy_Compatibility"):
            if sheet in xf.sheet_names:
                assert "__source_file_path" not in xf.parse(sheet).columns


def test_enrich_metadata_does_not_promote_assembled_component_ratios_from_wide_frame() -> None:
    df = _wide_frame(
        _wide_row(
            component_harmonic_energy_ratio=0.99,
            window="hann",
            n_fft=8192,
        )
    )
    enriched = cm._enrich_compiled_metadata_from_df({}, df)
    assert enriched.get("window") == "hann"
    assert enriched.get("n_fft") == 8192
    assert "component_harmonic_energy_ratio" not in enriched
