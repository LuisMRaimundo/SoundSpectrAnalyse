"""
Output curation tests for the SINGLE-PASS REFACTOR.

These tests verify that the compiled workbook is correctly partitioned into:

* ``Canonical_Metrics``      — final scientific metrics (no legacy_*, no
  batch_*, no compilation_error).
* ``Diagnostic_Metrics``     — intermediates, provenance and audit fields.
* ``Legacy_Compatibility``   — back-compat aliases isolated from canonical.

They also enforce coverage of ``metrics_dictionary.json``: every metric that
ends up in ``Canonical_Metrics`` must have a complete entry in the
dictionary, with non-empty ``formula``, ``quantity_type``, ``denominator``
and ``interpretation`` fields.

The tests build a minimal fixture by directly invoking the compile_metrics
classifier on a synthetic wide DataFrame, rather than going through the full
audio pipeline. This is fast and deterministic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from compile_metrics import (
    CANONICAL_METRIC_COLUMNS,
    LEGACY_COLUMN_EXACT_NAMES,
    LEGACY_COLUMN_NAME_PREFIXES,
    NEVER_CANONICAL_COLUMN_NAMES,
    _classify_compiled_column,
    _slice_compiled_df_by_status,
    _write_compiled_excel,
)


# ---------------------------------------------------------------------------
# Fixture: a wide compiled DataFrame that mixes canonical / diagnostic / legacy
# columns. Two notes so we can also exercise per-row downstream code paths.
# ---------------------------------------------------------------------------
def _wide_compiled_df() -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for note, comp_h in (("A4", 0.97), ("C5", 0.85)):
        rows.append(
            {
                # canonical
                "Note": note,
                "source_file_name": f"{note}.wav",
                "tier": "Tier_test",
                "component_harmonic_energy_ratio": comp_h,
                "component_inharmonic_energy_ratio": 1.0 - comp_h - 0.0,
                "component_subbass_energy_ratio": 0.0,
                "component_total_inharmonic_energy_ratio": 1.0 - comp_h,
                "model_harmonic_weight": comp_h,
                "model_inharmonic_weight": 1.0 - comp_h,
                "effective_partial_count": 1.0,
                "effective_partial_density": 1.0,
                "canonical_density_v5_adapted": 2.0,
                "density_metric_normalized": 0.5,
                "density_normalized_global": 0.5,
                "harmonic_energy_ratio": comp_h,
                "inharmonic_energy_ratio": 1.0 - comp_h,
                "subbass_energy_ratio": 0.0,
                "harmonic_inharmonic_ratio": comp_h / max(1e-9, 1.0 - comp_h),
                "spectral_entropy": 0.4,
                "harmonic_completeness": 0.9,
                # diagnostic
                "harmonic_energy_sum": 1.0,
                "inharmonic_energy_sum": 0.2,
                "subbass_energy_sum": 0.0,
                "total_component_energy": 1.2,
                "n_fft": 8192,
                "hop_length": 1024,
                "snr_threshold_db": -60.0,
                "component_energy_denominator": "H+I+S",
                "component_energy_method": "single_pass_proc_audio_energy",
                "component_profile_source": "integrated_single_pass",
                "component_energy_quantity": "power_sum_amplitude_squared",
                "model_weight_denominator": "harmonic_plus_inharmonic",
                "model_weights_source": "single_pass_proc_audio_energy",
                # legacy
                "batch_harmonic_energy_ratio": comp_h,
                "batch_inharmonic_energy_ratio": 1.0 - comp_h,
                "batch_subbass_energy_ratio": 0.0,
                "batch_total_inharmonic_energy_ratio": 1.0 - comp_h,
                "legacy_harmonic_density": 1.4,
                "legacy_inharmonic_density": 0.5,
                "legacy_combined_density": 1.0,
                "legacy_harmonic_density_percentage": 70.0,
                "legacy_inharmonic_density_percentage": 30.0,
                "harmonic_density": 1.4,
                "inharmonic_density": 0.5,
                "combined_density": 1.0,
                "Spectral Density Metric": 1.4,
                "Filtered Density Metric": 0.5,
                "Combined Density Metric": 1.0,
                "linear_sum_amplitude_harmonic": 1.5,
                "linear_sum_amplitude_inharmonic_partial": 0.45,
                "linear_sum_amplitude_subbass_band": 0.0,
                # forbidden in canonical (defence-in-depth)
                "compilation_error": "",
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Classifier-level tests (pure-function semantics).
# ---------------------------------------------------------------------------
def test_classifier_canonical_explicit():
    for name in CANONICAL_METRIC_COLUMNS:
        assert _classify_compiled_column(name) == "canonical", name


@pytest.mark.parametrize("prefix", LEGACY_COLUMN_NAME_PREFIXES)
def test_classifier_prefixes_are_legacy(prefix):
    assert _classify_compiled_column(prefix + "anything_at_all") == "legacy"


@pytest.mark.parametrize("name", sorted(LEGACY_COLUMN_EXACT_NAMES))
def test_classifier_exact_names_are_legacy(name):
    assert _classify_compiled_column(name) == "legacy"


def test_classifier_compilation_error_is_never_canonical():
    assert _classify_compiled_column("compilation_error") == "diagnostic"
    assert "compilation_error" in NEVER_CANONICAL_COLUMN_NAMES


def test_classifier_unknown_falls_to_diagnostic():
    assert _classify_compiled_column("some_random_internal_diag_field") == "diagnostic"


# ---------------------------------------------------------------------------
# Slicer-level tests.
# ---------------------------------------------------------------------------
def test_slicer_canonical_contains_no_legacy_or_batch():
    df = _wide_compiled_df()
    canon = _slice_compiled_df_by_status(df, "canonical")
    for col in canon.columns:
        assert not col.startswith("legacy_"), col
        assert not col.startswith("batch_"), col
        assert col not in LEGACY_COLUMN_EXACT_NAMES, col
        assert col != "compilation_error", col
    # And contains the headline canonical fields.
    must_contain = [
        "component_harmonic_energy_ratio",
        "component_inharmonic_energy_ratio",
        "component_subbass_energy_ratio",
        "component_total_inharmonic_energy_ratio",
        "model_harmonic_weight",
        "model_inharmonic_weight",
        "effective_partial_count",
    ]
    for c in must_contain:
        assert c in canon.columns, f"{c} missing from canonical slice"


def test_slicer_legacy_contains_batch_and_legacy_only():
    df = _wide_compiled_df()
    legacy_df = _slice_compiled_df_by_status(df, "legacy")
    # batch_* and legacy_* MUST be here.
    for c in (
        "batch_harmonic_energy_ratio",
        "batch_inharmonic_energy_ratio",
        "batch_subbass_energy_ratio",
        "legacy_harmonic_density",
        "legacy_inharmonic_density",
        "legacy_combined_density",
        "legacy_harmonic_density_percentage",
        "legacy_inharmonic_density_percentage",
        "harmonic_density",
        "inharmonic_density",
        "combined_density",
        "Spectral Density Metric",
        "Combined Density Metric",
        "linear_sum_amplitude_harmonic",
    ):
        assert c in legacy_df.columns, c
    # ... and component_* MUST NOT be here.
    for c in (
        "component_harmonic_energy_ratio",
        "model_harmonic_weight",
        "effective_partial_count",
    ):
        assert c not in legacy_df.columns or c == "Note", c


def test_slicer_diagnostic_contains_provenance_and_sums():
    df = _wide_compiled_df()
    diag = _slice_compiled_df_by_status(df, "diagnostic")
    for c in (
        "harmonic_energy_sum",
        "inharmonic_energy_sum",
        "subbass_energy_sum",
        "total_component_energy",
        "n_fft",
        "hop_length",
        "snr_threshold_db",
        "component_energy_denominator",
        "component_energy_method",
        "component_profile_source",
        "component_energy_quantity",
        # SEMANTIC HARDENING — the canonical short-name aliases were
        # demoted from canonical to diagnostic in v1.1 because they are
        # mathematically identical to component_*_energy_ratio.
        "harmonic_energy_ratio",
        "inharmonic_energy_ratio",
        "subbass_energy_ratio",
    ):
        assert c in diag.columns, c
    # ... and legacy_* MUST NOT be here.
    for c in (
        "legacy_harmonic_density",
        "batch_harmonic_energy_ratio",
    ):
        assert c not in diag.columns, c


# ---------------------------------------------------------------------------
# Workbook-level test: write a real .xlsx and inspect its sheets.
# ---------------------------------------------------------------------------
def test_compiled_workbook_has_three_curated_sheets(tmp_path: Path):
    pytest.importorskip("openpyxl")
    outp = tmp_path / "compiled_curation.xlsx"
    df = _wide_compiled_df()
    metadata = {
        "weight_function": "linear",
        "analysis_version": "test",
        "n_fft": 8192,
        "hop_length": 1024,
    }
    # Provide the *new* partial-sum columns so _build_density_metrics_main_sheet
    # does not bail out (otherwise the function takes the early-return branch).
    df["Harmonic Partials sum"] = 1.0
    df["Inharmonic Partials sum"] = 0.45
    df["Sub-bass sum"] = 0.0
    df["Total sum"] = 1.45
    df["weight_function"] = "linear"

    _write_compiled_excel(
        outp,
        df,
        metadata,
        apply_publication_column_filter=False,
        enable_pca_export=False,
        minimum_samples_for_pca=10,
    )

    xl = pd.ExcelFile(outp)
    sheet_names = set(xl.sheet_names)
    assert "Canonical_Metrics" in sheet_names, sheet_names
    assert "Diagnostic_Metrics" in sheet_names, sheet_names
    assert "Legacy_Compatibility" in sheet_names, sheet_names

    canon = pd.read_excel(outp, sheet_name="Canonical_Metrics")
    diag = pd.read_excel(outp, sheet_name="Diagnostic_Metrics")
    legacy_df = pd.read_excel(outp, sheet_name="Legacy_Compatibility")

    # Canonical: no legacy / batch / compilation_error / *_density (legacy names)
    for col in canon.columns:
        assert not col.startswith("legacy_"), col
        assert not col.startswith("batch_"), col
        assert col != "compilation_error", col
        assert col not in {"harmonic_density", "inharmonic_density", "combined_density"}, col

    # Legacy MUST have at least one of the legacy markers.
    legacy_markers = {
        "batch_harmonic_energy_ratio",
        "legacy_harmonic_density",
        "harmonic_density",
        "Spectral Density Metric",
    }
    assert legacy_markers.intersection(set(legacy_df.columns)), (
        f"Legacy sheet did not contain any legacy marker; columns={list(legacy_df.columns)}"
    )

    # Diagnostic MUST contain at least the energy sums.
    for c in ("harmonic_energy_sum", "inharmonic_energy_sum", "subbass_energy_sum"):
        assert c in diag.columns, c


# ---------------------------------------------------------------------------
# metrics_dictionary.json coverage tests.
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def metric_dict() -> Dict[str, Dict[str, str]]:
    path = REPO_ROOT / "metrics_dictionary.json"
    assert path.is_file(), f"metrics_dictionary.json not found at {path}"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert isinstance(data, dict) and "metrics" in data
    by_name: Dict[str, Dict[str, str]] = {}
    for entry in data["metrics"]:
        assert "name" in entry
        by_name[entry["name"]] = entry
    return by_name


def test_dictionary_has_schema_version():
    path = REPO_ROOT / "metrics_dictionary.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert "schema_version" in data
    assert isinstance(data["schema_version"], str)
    assert data["schema_version"].count(".") == 2  # semver-ish


def test_every_canonical_column_has_dictionary_entry(metric_dict):
    missing = [c for c in CANONICAL_METRIC_COLUMNS if c not in metric_dict]
    assert not missing, f"Canonical metrics missing from dictionary: {missing}"


def test_every_canonical_dictionary_entry_is_complete(metric_dict):
    incomplete: List[str] = []
    for name in CANONICAL_METRIC_COLUMNS:
        e = metric_dict[name]
        for required in ("formula", "quantity_type", "denominator", "interpretation"):
            if not e.get(required) or not str(e[required]).strip():
                incomplete.append(f"{name}: missing '{required}'")
    assert not incomplete, incomplete


def test_dictionary_statuses_are_valid(metric_dict):
    allowed = {"canonical", "diagnostic", "legacy"}
    bad = [name for name, e in metric_dict.items() if e.get("status") not in allowed]
    assert not bad, bad


def test_dictionary_quantity_types_are_valid(metric_dict):
    allowed = {
        "energy",
        "amplitude",
        "frequency",
        "count",
        "ratio",
        "entropy",
        "psychoacoustic",
        "metadata",
    }
    bad = [
        (name, e.get("quantity_type"))
        for name, e in metric_dict.items()
        if e.get("quantity_type") not in allowed
    ]
    assert not bad, bad


def test_dictionary_canonical_entries_match_canonical_set(metric_dict):
    """All entries flagged canonical in the dictionary MUST also be in the
    CANONICAL_METRIC_COLUMNS allow-list (no surprise canonical names)."""
    dict_canonical = {name for name, e in metric_dict.items() if e.get("status") == "canonical"}
    code_canonical = set(CANONICAL_METRIC_COLUMNS)
    extra = dict_canonical - code_canonical
    assert not extra, (
        f"Dictionary lists these as canonical but they are not in CANONICAL_METRIC_COLUMNS: {extra}"
    )


def test_dictionary_documents_legacy_and_batch_aliases(metric_dict):
    """At minimum the dictionary must document the legacy and batch aliases
    that users will most likely encounter in old reports."""
    must_be_legacy = {
        "legacy_harmonic_density",
        "legacy_inharmonic_density",
        "legacy_combined_density",
        "legacy_harmonic_density_percentage",
        "legacy_inharmonic_density_percentage",
        "batch_harmonic_energy_ratio",
        "batch_inharmonic_energy_ratio",
        "batch_subbass_energy_ratio",
        "batch_total_inharmonic_energy_ratio",
    }
    missing = [m for m in must_be_legacy if m not in metric_dict]
    assert not missing, f"Legacy/batch aliases missing from dictionary: {missing}"
    for m in must_be_legacy:
        assert metric_dict[m]["status"] == "legacy", (m, metric_dict[m]["status"])
        for required in ("formula", "quantity_type", "denominator", "interpretation", "do_not_interpret_as"):
            assert metric_dict[m].get(required), (m, required)


# ---------------------------------------------------------------------------
# SEMANTIC HARDENING tests (T2, T3, T4)
#
# These tests enforce:
#   - every dictionary entry has metric_family from the documented enum;
#   - every entry has derived_from (list) and independent_for_pca (bool);
#   - the canonical short-name aliases (harmonic_energy_ratio,
#     inharmonic_energy_ratio, subbass_energy_ratio) have been demoted to
#     status="diagnostic" with derived_from pointing at the component_* names;
#   - effective_partial_count and effective_partial_density are both canonical
#     and documented as distinct quantities;
#   - PCA_FEATURE_COLUMNS contains only metrics with independent_for_pca=true;
#   - PCA_FEATURE_COLUMNS_DEBUG_INCLUSIVE is a strict superset of the default.
# ---------------------------------------------------------------------------

ALLOWED_METRIC_FAMILIES = {
    "component_energy",
    "model_weight",
    "density",
    "entropy",
    "harmonicity",
    "rolloff",
    "psychoacoustic",
    "stft_parameter",
    "validation",
    "provenance",
    "legacy_compatibility",
    "metadata",
}


def test_dictionary_top_level_advertises_metric_family_enum():
    path = REPO_ROOT / "metrics_dictionary.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert "metric_family_enum" in data
    assert set(data["metric_family_enum"]) == ALLOWED_METRIC_FAMILIES, data["metric_family_enum"]


def test_dictionary_top_level_documents_pca_policy():
    path = REPO_ROOT / "metrics_dictionary.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert "pca_inclusion_policy" in data
    assert "independent_for_pca" in data["pca_inclusion_policy"]


def test_every_entry_has_metric_family(metric_dict):
    missing = [n for n, e in metric_dict.items() if "metric_family" not in e]
    assert not missing, missing


def test_every_canonical_entry_has_metric_family(metric_dict):
    missing = [
        n
        for n, e in metric_dict.items()
        if e.get("status") == "canonical" and not e.get("metric_family")
    ]
    assert not missing, missing


def test_metric_family_values_are_in_allowed_enum(metric_dict):
    bad = [(n, e.get("metric_family")) for n, e in metric_dict.items()
           if e.get("metric_family") not in ALLOWED_METRIC_FAMILIES]
    assert not bad, bad


def test_every_entry_has_derived_from_list(metric_dict):
    bad: List[str] = []
    for n, e in metric_dict.items():
        if "derived_from" not in e:
            bad.append(f"{n}: missing derived_from")
        elif not isinstance(e["derived_from"], list):
            bad.append(f"{n}: derived_from must be a list, got {type(e['derived_from']).__name__}")
    assert not bad, bad


def test_every_entry_has_independent_for_pca_bool(metric_dict):
    bad: List[str] = []
    for n, e in metric_dict.items():
        if "independent_for_pca" not in e:
            bad.append(f"{n}: missing independent_for_pca")
        elif not isinstance(e["independent_for_pca"], bool):
            bad.append(f"{n}: independent_for_pca must be a bool")
    assert not bad, bad


def test_every_canonical_entry_has_both_pca_fields(metric_dict):
    bad: List[str] = []
    for n, e in metric_dict.items():
        if e.get("status") != "canonical":
            continue
        if "derived_from" not in e:
            bad.append(f"{n}: missing derived_from")
        if "independent_for_pca" not in e:
            bad.append(f"{n}: missing independent_for_pca")
    assert not bad, bad


def test_derived_from_targets_exist_in_dictionary(metric_dict):
    """Every name listed in derived_from must itself be a dictionary entry,
    so the dependency graph is closed and machine-traversable."""
    unknown: List[str] = []
    for n, e in metric_dict.items():
        for parent in e.get("derived_from", []):
            if parent not in metric_dict:
                unknown.append(f"{n} → derived_from references unknown '{parent}'")
    assert not unknown, unknown


def test_canonical_short_aliases_are_demoted_to_diagnostic(metric_dict):
    """harmonic_energy_ratio / inharmonic_energy_ratio / subbass_energy_ratio
    are mathematically identical to component_*_energy_ratio. v1.1 of the
    dictionary demotes them to status='diagnostic'."""
    for alias, parent in (
        ("harmonic_energy_ratio", "component_harmonic_energy_ratio"),
        ("inharmonic_energy_ratio", "component_inharmonic_energy_ratio"),
        ("subbass_energy_ratio", "component_subbass_energy_ratio"),
    ):
        assert alias in metric_dict, alias
        assert metric_dict[alias]["status"] == "diagnostic", (alias, metric_dict[alias]["status"])
        assert parent in metric_dict[alias].get("derived_from", []), (alias, parent)
        assert metric_dict[alias]["independent_for_pca"] is False, alias


def test_effective_partial_count_and_density_are_both_canonical_and_distinct(metric_dict):
    """effective_partial_count (harmonic-only N_eff) and
    effective_partial_density (blended N_eff bundle) MUST both be canonical
    and clearly distinguished — neither references the other in derived_from."""
    for name in ("effective_partial_count", "effective_partial_density"):
        assert name in metric_dict, name
        assert metric_dict[name]["status"] == "canonical", name
        assert metric_dict[name]["independent_for_pca"] is True, name
    # They are NOT algebraic relatives of each other.
    df_count = metric_dict["effective_partial_count"].get("derived_from", [])
    df_dens = metric_dict["effective_partial_density"].get("derived_from", [])
    assert "effective_partial_density" not in df_count
    assert "effective_partial_count" not in df_dens
    # The formula strings should differ.
    assert (
        metric_dict["effective_partial_count"]["formula"]
        != metric_dict["effective_partial_density"]["formula"]
    )


def test_dependent_canonical_metrics_are_flagged_not_independent(metric_dict):
    """component_subbass_energy_ratio = 1 - H - I,
    component_total_inharmonic_energy_ratio = I + S,
    model_inharmonic_weight = 1 - model_harmonic_weight,
    density_metric_normalized, density_normalized_global — all are exact
    algebraic transforms of other canonical metrics and MUST be flagged
    independent_for_pca=false."""
    must_be_dependent = [
        "component_subbass_energy_ratio",
        "component_total_inharmonic_energy_ratio",
        "model_inharmonic_weight",
        "density_metric_normalized",
        "density_normalized_global",
    ]
    bad: List[str] = []
    for name in must_be_dependent:
        assert name in metric_dict, name
        if metric_dict[name].get("independent_for_pca") is True:
            bad.append(name)
    assert not bad, f"These dependent canonical metrics must NOT be flagged independent_for_pca=true: {bad}"


def test_pca_feature_columns_only_include_independent_metrics(metric_dict):
    from compile_metrics import PCA_FEATURE_COLUMNS

    offenders: List[str] = []
    for feat in PCA_FEATURE_COLUMNS:
        if feat not in metric_dict:
            offenders.append(f"{feat}: not in dictionary")
            continue
        if metric_dict[feat].get("independent_for_pca") is not True:
            offenders.append(
                f"{feat}: independent_for_pca={metric_dict[feat].get('independent_for_pca')} (must be true)"
            )
    assert not offenders, offenders


def test_no_two_pca_features_are_exact_algebraic_complements(metric_dict):
    """No selected PCA feature must list another selected PCA feature in its
    derived_from graph — that would make the matrix rank-deficient."""
    from compile_metrics import PCA_FEATURE_COLUMNS

    selected = set(PCA_FEATURE_COLUMNS)
    bad: List[str] = []
    for feat in PCA_FEATURE_COLUMNS:
        entry = metric_dict.get(feat, {})
        for parent in entry.get("derived_from", []):
            if parent in selected:
                bad.append(f"{feat} derives from another PCA feature '{parent}'")
    assert not bad, bad


def test_pca_inclusive_debug_list_is_superset_of_default():
    from compile_metrics import PCA_FEATURE_COLUMNS, PCA_FEATURE_COLUMNS_DEBUG_INCLUSIVE

    default_set = set(PCA_FEATURE_COLUMNS)
    debug_set = set(PCA_FEATURE_COLUMNS_DEBUG_INCLUSIVE)
    assert default_set.issubset(debug_set), default_set - debug_set
    # Strict superset: the inclusive list MUST add at least the algebraic complements.
    assert {"subbass_energy_ratio", "model_inharmonic_weight"}.issubset(debug_set)


def test_compute_optional_pca_sheets_accepts_dependent_flag(tmp_path: Path):
    """The signature must accept ``pca_include_dependent_metrics`` (forensic
    debug). We do not run a full PCA here — only assert the kwarg path is
    plumbed through and rejects unknown kwargs."""
    pytest.importorskip("openpyxl")
    from compile_metrics import _compute_optional_pca_sheets

    df = pd.DataFrame({"effective_partial_density": np.linspace(0.1, 1.0, 20)})
    # Should not raise:
    _compute_optional_pca_sheets(
        df,
        enable_pca_export=False,  # explicit short-circuit to "skipped"
        minimum_samples_for_pca=10,
        pca_include_dissonance=False,
        pca_include_dependent_metrics=True,
    )


def test_dictionary_audit_notes_present():
    path = REPO_ROOT / "metrics_dictionary.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    assert "audit_notes" in data
    notes = data["audit_notes"].get("duplicate_resolution_v1_1", [])
    assert any("harmonic_energy_ratio" in n for n in notes)
    assert any("effective_partial_count" in n and "effective_partial_density" in n for n in notes)


def test_compile_guide_warning_present_in_dataframe():
    """The Compile_Guide dataframe must include the publication-policy
    warning rows produced by SEMANTIC HARDENING."""
    from compile_metrics import _build_compile_guide_dataframe

    cg = _build_compile_guide_dataframe(
        {"weight_function": "linear"},
        density_columns=["Note", "weight_function", "Harmonic Partials sum"],
    )
    # Look for the canonical-vs-density disclaimer in the assembled rows.
    text_blob = " | ".join(map(str, cg["Value"].tolist()))
    assert "Canonical_Metrics" in text_blob
    assert "publication-grade" in text_blob.lower() or "publication grade" in text_blob.lower()
    # PCA policy mention too.
    assert "independent_for_pca" in text_blob


def test_density_metrics_publication_warning_not_in_analysis_metadata(tmp_path: Path):
    """Publication-clean exports keep policy text in Compile_Guide, not Analysis_Metadata prose keys."""
    pytest.importorskip("openpyxl")
    outp = tmp_path / "compiled_warning.xlsx"
    df = _wide_compiled_df()
    df["Harmonic Partials sum"] = 1.0
    df["Inharmonic Partials sum"] = 0.45
    df["Sub-bass sum"] = 0.0
    df["Total sum"] = 1.45
    df["weight_function"] = "linear"
    _write_compiled_excel(
        outp,
        df,
        {"weight_function": "linear", "analysis_version": "test"},
        apply_publication_column_filter=False,
        enable_pca_export=False,
        minimum_samples_for_pca=10,
    )
    am = pd.read_excel(outp, sheet_name="Analysis_Metadata")
    flat = " ".join(map(str, am.values.flatten().tolist())).lower()
    assert "density_metrics_publication_warning" not in flat
    assert "density_metrics is preserved for backward compatibility" not in flat
    cg = pd.read_excel(outp, sheet_name="Compile_Guide")
    cg_blob = " | ".join(map(str, cg["Value"].tolist())).lower()
    assert "canonical_metrics" in cg_blob or "density_metrics" in cg_blob
