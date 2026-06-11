from __future__ import annotations

"""
Export / merge / schema-safety contract tests for compile_metrics.py.

Complements tests/phase_12/test_compile_metrics_contract_additional.py by
exercising DataFrame export preparation, density-sheet assembly, column
resolution, metadata merge keys, and canonical/legacy coexistence helpers.

No audio pipeline, GUI, plotting, or broad workbook integration.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import compile_metrics as cm


# ---------------------------------------------------------------------------
# 1. Alias split: canonical vs legacy coexistence
# ---------------------------------------------------------------------------

def test_split_strict_alias_preserves_differing_canonical_and_alias_values() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "component_harmonic_energy_ratio": [0.70],
            "harmonic_energy_ratio": [0.99],  # strict alias — different value
            "qc_status": ["validated_pipeline"],
        }
    )
    main, aliases = cm._split_strict_alias_columns(df)
    assert main["component_harmonic_energy_ratio"].iloc[0] == pytest.approx(0.70)
    assert aliases["harmonic_energy_ratio"].iloc[0] == pytest.approx(0.99)
    assert "harmonic_energy_ratio" not in main.columns
    assert "component_harmonic_energy_ratio" not in aliases.columns


def test_split_strict_alias_keeps_status_text_on_main_frame() -> None:
    df = pd.DataFrame(
        {
            "Note": ["D4"],
            "density_metric_raw": [1.2],
            "inharmonicity_fit_status": ["fit_ok"],
            "subbass_energy_ratio": [0.05],
        }
    )
    main, aliases = cm._split_strict_alias_columns(df)
    assert main["inharmonicity_fit_status"].iloc[0] == "fit_ok"
    assert "subbass_energy_ratio" in aliases.columns


# ---------------------------------------------------------------------------
# 2. Export preparation (_prepare_df_for_density_export)
# ---------------------------------------------------------------------------

def test_prepare_df_for_density_export_aliases_and_soma_linear_total() -> None:
    src = pd.DataFrame(
        {
            "Note": ["C4", "D4"],
            "Spectral Entropy": [2.5, 3.0],
            "unique_harmonic_order_count": [12, 10],
            "linear_sum_amplitude_harmonic": [1.0, 2.0],
            "linear_sum_amplitude_inharmonic_partial": [0.2, 0.1],
            "linear_sum_amplitude_subbass_band": [0.05, 0.0],
            "density_metric_raw": [3.5, 4.0],
        }
    )
    snapshot = src.copy()
    out = cm._prepare_df_for_density_export(src)
    pd.testing.assert_frame_equal(src, snapshot)
    assert out["spectral_entropy"].tolist() == pytest.approx([2.5, 3.0])
    assert out["harmonic_order_count"].tolist() == pytest.approx([12.0, 10.0])
    assert out["Soma_A_linear_total"].tolist() == pytest.approx([1.25, 2.1])
    assert out["density_body_weighted_sum_body_ceiling"].tolist() == pytest.approx([3.5, 4.0])


def test_prepare_df_for_density_export_is_deterministic() -> None:
    src = pd.DataFrame(
        {
            "Note": ["C4"],
            "spectral_entropy": [1.1],
            "linear_sum_amplitude_harmonic": [0.5],
            "linear_sum_amplitude_inharmonic_partial": [0.1],
            "linear_sum_amplitude_subbass_band": [0.0],
        }
    )
    a = cm._prepare_df_for_density_export(src)
    b = cm._prepare_df_for_density_export(src)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# 3. Column resolution and note_density_final
# ---------------------------------------------------------------------------

def test_resolve_note_density_sum_prefers_canonical_over_legacy_display_sum() -> None:
    df = pd.DataFrame(
        {
            "Harmonic Partials sum": [999.0],
            "harmonic_density_sum": [2.0],
        }
    )
    assert cm._resolve_note_density_sum_column(df, "harmonic") == "harmonic_density_sum"


def test_resolve_note_density_ratio_prefers_component_canonical_over_alias() -> None:
    both = pd.DataFrame(
        {
            "component_harmonic_energy_ratio": [0.7],
            "harmonic_energy_ratio": [0.9],
        }
    )
    assert cm._resolve_note_density_ratio_column(both, "harmonic") == "component_harmonic_energy_ratio"
    alias_only = pd.DataFrame({"inharmonic_energy_ratio": [0.2]})
    assert cm._resolve_note_density_ratio_column(alias_only, "inharmonic") == "inharmonic_energy_ratio"


def test_compute_note_density_final_weighted_closure_and_nan_propagation() -> None:
    df = pd.DataFrame(
        {
            "harmonic_density_sum": [2.0, 3.0],
            "inharmonic_density_sum": [1.0, 1.0],
            "subbass_density_sum": [0.5, np.nan],
            "component_harmonic_energy_ratio": [0.7, 0.6],
            "component_inharmonic_energy_ratio": [0.2, 0.3],
            "component_subbass_energy_ratio": [0.1, 0.1],
        }
    )
    out = cm._compute_note_density_final(df)
    assert out.iloc[0] == pytest.approx(2.0 * 0.7 + 1.0 * 0.2 + 0.5 * 0.1)
    assert np.isnan(out.iloc[1])

    incomplete = pd.DataFrame({"Harmonic Partials sum": [1.0]})
    missing = cm._compute_note_density_final(incomplete)
    assert missing.isna().all()


# ---------------------------------------------------------------------------
# 4. Density_Metrics main sheet export contract
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_build_density_metrics_main_sheet_orders_columns_and_sets_status_token() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "Harmonic Partials sum": [1.0],
            "Inharmonic Partials sum": [0.2],
            "Sub-bass sum": [0.05],
            "Total sum": [1.25],
            "density_metric_raw": [1.25],
            "component_harmonic_energy_ratio": [0.7],
            "component_inharmonic_energy_ratio": [0.2],
            "component_subbass_energy_ratio": [0.1],
            "harmonic_density_sum": [1.0],
            "inharmonic_density_sum": [0.2],
            "subbass_density_sum": [0.05],
            "extra_debug_column": [42],
        }
    )
    out = cm._build_density_metrics_main_sheet(df)
    assert out.columns[0] == "Note"
    assert out.columns[1] == "density_metric_raw"
    assert out["density_extraction_status"].iloc[0] == "ok"
    assert out["density_component_basis"].iloc[0] == cm.DENSITY_COMPONENT_BASIS_DEFAULT
    assert "Window" not in out.columns
    prefix = [c for c in cm.DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS if c in out.columns]
    assert list(out.columns[: len(prefix)]) == prefix


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_build_density_metrics_main_sheet_missing_partial_sums_emits_compilation_error() -> None:
    df = pd.DataFrame({"Note": ["C4"], "density_metric_raw": [1.0]})
    out = cm._build_density_metrics_main_sheet(df, weight_function="log")
    assert list(out.columns) == ["compilation_error"]
    assert "Missing partial-sum columns" in out["compilation_error"].iloc[0]


@pytest.mark.filterwarnings("ignore::pandas.errors.PerformanceWarning")
def test_build_density_metrics_main_sheet_is_deterministic() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "Harmonic Partials sum": [1.0],
            "Inharmonic Partials sum": [0.2],
            "Sub-bass sum": [0.05],
            "Total sum": [1.25],
            "harmonic_density_sum": [1.0],
            "inharmonic_density_sum": [0.2],
            "subbass_density_sum": [0.05],
            "component_harmonic_energy_ratio": [0.7],
            "component_inharmonic_energy_ratio": [0.2],
            "component_subbass_energy_ratio": [0.1],
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=pd.errors.PerformanceWarning)
        a = cm._build_density_metrics_main_sheet(df)
        b = cm._build_density_metrics_main_sheet(df)
    pd.testing.assert_frame_equal(a, b)


# ---------------------------------------------------------------------------
# 5. Allowed columns / Phase 7 protection on export prune path
# ---------------------------------------------------------------------------

def test_density_metrics_allowed_columns_include_phase7_and_weighted_tokens() -> None:
    allowed = cm.DENSITY_METRICS_ALLOWED_COLUMNS
    for col in cm.PHASE7_INHARMONICITY_COMPILED_COLUMNS:
        assert col in allowed
    assert "density_weighted_sum" in allowed
    assert "density_metric_raw" in allowed


def test_drop_dead_columns_keeps_phase7_status_even_when_all_blank() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "inharmonicity_fit_status": [""],
            "inharmonicity_coefficient_B": [np.nan],
            "dead_col": [""],
        }
    )
    out = cm._drop_dead_columns(df)
    assert "inharmonicity_fit_status" in out.columns
    assert "inharmonicity_coefficient_B" in out.columns
    assert "dead_col" not in out.columns


# ---------------------------------------------------------------------------
# 6. Sheet / column preference helpers (additional cases)
# ---------------------------------------------------------------------------

def test_build_spectrum_column_preferences_put_amplitude_last() -> None:
    amp_prefs = cm._harmonic_column_preferences("amplitude_sum")
    power_prefs = cm._harmonic_column_preferences("power_sum")
    assert amp_prefs[-1] == "Amplitude"
    assert power_prefs[-1] == "Amplitude"
    assert "Amplitude_raw" in amp_prefs
    assert "Power_raw" in power_prefs


def test_pick_sheet_prefers_first_matching_preference_in_order() -> None:
    sheets = ["Metrics", "Inharmonic Spectrum", "Harmonic Spectrum"]
    assert (
        cm._pick_sheet_case_insensitive(sheets, cm.INHARMONIC_SPECTRUM_SHEET_PREFERENCES)
        == "Inharmonic Spectrum"
    )


def test_pick_column_returns_none_when_no_preference_matches() -> None:
    assert cm._pick_column_case_insensitive(["Note", "qc_status"], ("missing",)) is None


# ---------------------------------------------------------------------------
# 7. Status slicing and primary-profile restriction
# ---------------------------------------------------------------------------

def test_slice_compiled_df_by_status_empty_frame_returns_empty() -> None:
    assert cm._slice_compiled_df_by_status(pd.DataFrame(), "canonical").empty


def test_restrict_primary_subset_single_profile_is_unchanged() -> None:
    pid = "wf=log|dst=-60.0|ceil=20000.0"
    df = pd.DataFrame(
        {
            "Note": ["A", "B"],
            "analysis_parameter_profile_id": [pid, pid],
            "density_metric_raw": [1.0, 2.0],
        }
    )
    out, restricted, kept = cm._restrict_primary_subset_to_single_profile(df)
    pd.testing.assert_frame_equal(out, df)
    assert restricted is False
    assert kept == pid


# ---------------------------------------------------------------------------
# 8. Metadata merge / enrichment invariants
# ---------------------------------------------------------------------------

def test_merge_canonical_compiled_workbook_metadata_sets_formal_tokens() -> None:
    meta: dict = {}
    cm._merge_canonical_compiled_workbook_metadata(
        meta,
        file_pattern="results/*/spectral_analysis.xlsx",
        allow_legacy_super_json=False,
        input_schema_validation_status="validated",
    )
    assert meta["input_schema_validation_status"] == "validated"
    assert meta["legacy_pipeline_used"] is False
    assert meta["publication_output_allowed"] is True
    assert meta["compiled_by"] == "compile_metrics.compile_density_metrics_with_pca"
    assert meta.get("pipeline_contract_version")


def test_merge_canonical_metadata_disallows_publication_for_legacy_super_json() -> None:
    meta: dict = {}
    cm._merge_canonical_compiled_workbook_metadata(
        meta,
        file_pattern="super_analysis_results.json",
        allow_legacy_super_json=True,
        input_schema_validation_status="legacy_source",
    )
    assert meta["legacy_pipeline_used"] is True
    assert meta["publication_output_allowed"] is False


def test_enrich_compiled_metadata_does_not_promote_row_zero_component_ratios() -> None:
    df = pd.DataFrame(
        {
            "window": ["hann"],
            "N FFT": [8192],
            "component_harmonic_energy_ratio": [0.99],
            "component_inharmonic_energy_ratio": [0.01],
        }
    )
    enriched = cm._enrich_compiled_metadata_from_df({}, df)
    assert enriched["window"] == "hann"
    assert enriched["n_fft"] == 8192
    assert enriched.get("phase2_harmonic_application_weight") is None
    assert "component_harmonic_energy_ratio" not in enriched
    assert enriched["harmonic_density_weight_metadata_semantics"].startswith("use phase2_harmonic")


# ---------------------------------------------------------------------------
# 9. Canonical density columns on wide frames
# ---------------------------------------------------------------------------

def test_add_canonical_and_global_density_columns_prefers_existing_canonical() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4", "D4"],
            "canonical_density_v5_adapted": [4.0, 2.0],
            "Density Metric": [40.0, 20.0],
            "harmonic_order_count": [10, 8],
        }
    )
    out = cm._add_canonical_and_global_density_columns(df)
    assert out["canonical_density_v5_adapted"].tolist() == pytest.approx([4.0, 2.0])
    assert out["density_normalized_global"].tolist() == pytest.approx([1.0, 0.5])
    assert (
        out["density_normalization_scope"].iloc[0]
        == "compiled_dataset_max_canonical_density_v5_adapted"
    )


def test_add_canonical_and_global_density_columns_empty_passthrough() -> None:
    empty = pd.DataFrame()
    assert cm._add_canonical_and_global_density_columns(empty) is empty
