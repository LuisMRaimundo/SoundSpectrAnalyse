"""Tests for ``validate_canonical_metrics.py``.

A small synthetic corpus exercises every contract advertised by the script:

* non-canonical (diagnostic / legacy) metrics are excluded from canonical
  statistics, correlations and PCA;
* ``independent_for_pca=false`` canonical metrics are excluded from PCA;
* high-correlation pairs are reported with the correct *algebraic* vs
  *empirical* redundancy classification;
* missing canonical columns surface as warnings, not silent failures;
* near-constant canonical metrics are surfaced and excluded from PCA;
* the Markdown / Excel reports are produced with the required sections;
* no musicological term sneaks into the report text (the only allowed
  interpretation text is taken verbatim from ``metrics_dictionary.json``).
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

from validate_canonical_metrics import (
    MetricDictionary,
    _is_numeric_non_constant,
    build_corpus_summary,
    compute_canonical_coverage,
    compute_correlation_matrix,
    compute_descriptive_stats,
    compute_missing_values_summary,
    detect_near_constant_metrics,
    detect_outlier_rows,
    find_high_correlations,
    load_canonical_metrics_from_workbook,
    run_pca_on_canonical,
    validate_corpus,
    write_report_excel,
    write_report_markdown,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def dictionary() -> MetricDictionary:
    return MetricDictionary.load(REPO_ROOT / "metrics_dictionary.json")


@pytest.fixture
def synthetic_corpus(dictionary: MetricDictionary) -> pd.DataFrame:
    """A 12-row, 2-instrument synthetic Canonical_Metrics corpus.

    Layout:
      - Two ``Instrument`` groups: synth_clarinet, synth_flute.
      - Six notes per instrument (A3..A4-ish).
      - Canonical metrics populated with values that exercise correlations,
        near-constants, and an algebraic dependency.
      - A few **diagnostic** and **legacy** columns mixed in to verify they
        are ignored by the canonical pipeline.
      - One NaN in a canonical metric to exercise missing-value reporting.
    """
    np.random.seed(42)
    notes = ["A3", "B3", "C4", "D4", "E4", "G4"]
    instruments = ["synth_clarinet", "synth_flute"]
    rows: List[Dict[str, object]] = []
    for inst_idx, inst in enumerate(instruments):
        for note_idx, note in enumerate(notes):
            # Vary harmonic-energy fraction by instrument so canonical
            # metrics carry real signal but in PHYSICAL-acoustic terms only.
            base_h = 0.90 if inst == "synth_flute" else 0.60
            comp_h = float(np.clip(base_h + 0.02 * note_idx, 0.0, 1.0))
            comp_i = float(np.clip(0.95 - comp_h - 0.01 * note_idx, 0.0, 1.0))
            comp_s = float(max(0.0, 1.0 - comp_h - comp_i))
            rows.append(
                {
                    # canonical identifiers
                    "Note": note,
                    "source_file_name": f"{inst}_{note}.wav",
                    "tier": "Tier_test",
                    # grouping aid (NOT a canonical metric in the dictionary)
                    "Instrument": inst,
                    # canonical component_energy family
                    "component_harmonic_energy_ratio": comp_h,
                    "component_inharmonic_energy_ratio": comp_i,
                    "component_subbass_energy_ratio": comp_s,
                    "component_total_inharmonic_energy_ratio": comp_i + comp_s,
                    # canonical model_weight family
                    "model_harmonic_weight": comp_h / max(1e-9, comp_h + comp_i),
                    "model_inharmonic_weight": comp_i / max(1e-9, comp_h + comp_i),
                    # canonical density family
                    "effective_partial_count": 3.0 + inst_idx + 0.1 * note_idx,
                    "effective_partial_density": 5.0 + 0.5 * (inst_idx + note_idx),
                    "canonical_density_v5_adapted": 2.0 + 0.05 * note_idx,
                    "density_metric_normalized": 0.5 + 0.04 * note_idx,
                    "density_normalized_global": 0.5 + 0.04 * note_idx,
                    "density_per_component": 0.25 + 0.005 * note_idx,
                    "rolloff_compensated_harmonic_density": 1.5,  # near-constant on purpose
                    "harmonic_effective_power_density": 0.7 + 0.01 * note_idx,
                    # canonical harmonicity / entropy
                    "harmonic_inharmonic_ratio": comp_h / max(1e-9, comp_i),
                    "spectral_entropy": 0.4 + 0.02 * note_idx + 0.1 * inst_idx,
                    "harmonic_completeness": 0.9 - 0.01 * note_idx,
                    # ----- non-canonical clutter the script MUST ignore
                    "harmonic_energy_ratio": comp_h,         # diagnostic alias
                    "inharmonic_energy_ratio": comp_i,       # diagnostic alias
                    "subbass_energy_ratio": comp_s,          # diagnostic alias
                    "harmonic_energy_sum": 1.0 + 0.1 * note_idx,  # diagnostic
                    "n_fft": 8192,                            # diagnostic
                    "batch_harmonic_energy_ratio": comp_h,    # legacy alias
                    "legacy_harmonic_density": 1.4 + 0.1 * note_idx,  # legacy
                    "compilation_error": "",                  # never canonical
                }
            )
    df = pd.DataFrame(rows)
    # Inject a single NaN in one canonical metric to exercise missing-value
    # reporting.
    df.loc[0, "harmonic_completeness"] = np.nan
    return df


# ---------------------------------------------------------------------------
# Dictionary access
# ---------------------------------------------------------------------------
def test_dictionary_loads_canonical_and_pca_lists(dictionary: MetricDictionary):
    canon = dictionary.canonical_names()
    pca_feats = dictionary.canonical_independent_for_pca()
    assert "component_harmonic_energy_ratio" in canon
    assert "component_harmonic_energy_ratio" in pca_feats
    # Algebraic complements MUST be excluded from PCA.
    assert "component_subbass_energy_ratio" not in pca_feats
    assert "component_total_inharmonic_energy_ratio" not in pca_feats
    assert "model_inharmonic_weight" not in pca_feats
    # The dependent canonical metrics are still canonical:
    assert "component_subbass_energy_ratio" in canon
    assert "model_inharmonic_weight" in canon


def test_metric_family_grouping_uses_dictionary(dictionary: MetricDictionary):
    groups = dictionary.family_groups(
        ["component_harmonic_energy_ratio", "model_harmonic_weight", "spectral_entropy"]
    )
    assert "component_energy" in groups
    assert "model_weight" in groups
    assert "entropy" in groups
    assert "component_harmonic_energy_ratio" in groups["component_energy"]


def test_dictionary_declared_dependency_pairs_includes_known_relations(dictionary: MetricDictionary):
    pairs = dictionary.declared_dependency_pairs()
    # component_subbass_energy_ratio is declared as derived from
    # component_harmonic_energy_ratio + component_inharmonic_energy_ratio.
    assert (
        "component_subbass_energy_ratio",
        "component_harmonic_energy_ratio",
    ) in pairs
    assert (
        "model_inharmonic_weight",
        "model_harmonic_weight",
    ) in pairs


# ---------------------------------------------------------------------------
# Coverage / missing values
# ---------------------------------------------------------------------------
def test_coverage_flags_missing_canonical_metrics(synthetic_corpus, dictionary):
    # Remove one canonical metric on purpose.
    df = synthetic_corpus.drop(columns=["spectral_entropy"])
    coverage = compute_canonical_coverage(df, dictionary.canonical_names())
    row = coverage[coverage["metric"] == "spectral_entropy"].iloc[0]
    # pandas may store this as numpy.bool_, so coerce to a Python bool before
    # the identity check.
    assert bool(row["present_in_workbook"]) is False
    assert int(row["non_null_count"]) == 0


def test_missing_values_table_reports_nan(synthetic_corpus, dictionary):
    table = compute_missing_values_summary(
        synthetic_corpus, dictionary.canonical_names()
    )
    nan_row = table[table["metric"] == "harmonic_completeness"].iloc[0]
    assert nan_row["nan_count"] == 1
    assert nan_row["nan_fraction"] > 0
    # An untouched canonical metric should have zero NaNs reported.
    clean = table[table["metric"] == "component_harmonic_energy_ratio"].iloc[0]
    assert clean["nan_count"] == 0


# ---------------------------------------------------------------------------
# Descriptive stats
# ---------------------------------------------------------------------------
def test_descriptive_stats_groups_by_instrument(synthetic_corpus, dictionary):
    desc = compute_descriptive_stats(
        synthetic_corpus,
        dictionary.canonical_names(),
        group_by=["Instrument"],
    )
    groups_seen = set(desc["group"].unique().tolist())
    assert "ALL" in groups_seen
    assert any("synth_flute" in g for g in groups_seen)
    assert any("synth_clarinet" in g for g in groups_seen)


def test_descriptive_stats_only_lists_canonical_metrics(synthetic_corpus, dictionary):
    desc = compute_descriptive_stats(
        synthetic_corpus, dictionary.canonical_names()
    )
    metrics_listed = set(desc["metric"].unique().tolist())
    # diagnostic columns MUST NOT be listed
    assert "harmonic_energy_sum" not in metrics_listed
    assert "harmonic_energy_ratio" not in metrics_listed
    assert "legacy_harmonic_density" not in metrics_listed


# ---------------------------------------------------------------------------
# Correlation / redundancy
# ---------------------------------------------------------------------------
def test_correlation_matrix_excludes_diagnostic_columns(synthetic_corpus, dictionary):
    corr = compute_correlation_matrix(
        synthetic_corpus, dictionary.canonical_names()
    )
    assert "harmonic_energy_sum" not in corr.columns
    assert "harmonic_energy_ratio" not in corr.columns
    # canonical metrics are present
    assert "component_harmonic_energy_ratio" in corr.columns


def test_find_high_correlations_classifies_algebraic_vs_empirical(synthetic_corpus, dictionary):
    canon = dictionary.canonical_names()
    corr = compute_correlation_matrix(synthetic_corpus, canon)
    declared = dictionary.declared_dependency_pairs()
    high = find_high_correlations(corr, threshold=0.90, declared_dependency_pairs=declared)
    assert not high.empty, "expected at least one high-correlation pair in the synthetic corpus"
    # AUDIT FIX (single-pass weighted density) — density_metric_normalized
    # and density_normalized_global are NO LONGER algebraic aliases.
    # After the refactor:
    #   density_normalized_global   = max-norm of canonical_density_v5_adapted (canonical)
    #   density_metric_normalized   = max-norm of density_metric_raw (diagnostic, Density_Metrics)
    # The remaining declared algebraic pair within the canonical set is
    # model_harmonic_weight / model_inharmonic_weight (model_inharmonic_weight
    # derives_from model_harmonic_weight and they sum to 1).
    alias_row = high[
        (
            (high["metric_a"] == "model_harmonic_weight")
            & (high["metric_b"] == "model_inharmonic_weight")
        )
        | (
            (high["metric_a"] == "model_inharmonic_weight")
            & (high["metric_b"] == "model_harmonic_weight")
        )
    ]
    assert not alias_row.empty, "expected the model_*_weight algebraic pair to surface as a high-correlation pair"
    assert (alias_row["redundancy_type"].iloc[0]) == "algebraic"
    # All reported pearson_r values must be ≥ threshold.
    assert (high["abs_r"] >= 0.90).all()


def test_find_high_correlations_flags_empirical_when_undeclared():
    """Build a tiny corpus where two metrics correlate without being declared
    as related in the dictionary; the report must flag this as ``empirical``."""
    df = pd.DataFrame(
        {
            "spectral_entropy": np.linspace(0.1, 0.9, 20),
            "harmonic_completeness": np.linspace(0.9, 0.1, 20),  # perfectly anti-correlated
        }
    )
    corr = compute_correlation_matrix(df, ["spectral_entropy", "harmonic_completeness"])
    high = find_high_correlations(
        corr, threshold=0.90, declared_dependency_pairs=set()
    )
    assert not high.empty
    assert high["redundancy_type"].iloc[0] == "empirical"
    assert high["pearson_r"].iloc[0] < -0.95


# ---------------------------------------------------------------------------
# Near-constant / outliers
# ---------------------------------------------------------------------------
def test_near_constant_metric_flagged(synthetic_corpus, dictionary):
    near = detect_near_constant_metrics(
        synthetic_corpus,
        dictionary.canonical_names(),
        std_threshold=1e-6,
    )
    assert "rolloff_compensated_harmonic_density" in set(near["metric"].tolist())


def test_outlier_row_detection_uses_iqr_fences():
    df = pd.DataFrame(
        {
            "Note": [f"N{i}" for i in range(10)],
            "spectral_entropy": [0.1, 0.12, 0.11, 0.13, 0.12, 0.11, 0.14, 0.13, 0.12, 9.9],
        }
    )
    outliers = detect_outlier_rows(df, ["spectral_entropy"], iqr_multiplier=1.5)
    assert not outliers.empty
    assert outliers.iloc[0]["Note"] == "N9"
    assert outliers.iloc[0]["value"] > outliers.iloc[0]["upper_fence"]


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------
def test_pca_excludes_non_independent_metrics(synthetic_corpus, dictionary):
    pca_feats = dictionary.canonical_independent_for_pca()
    pca_result = run_pca_on_canonical(
        synthetic_corpus, pca_feats, minimum_samples=4
    )
    if pca_result.loadings.empty:
        pytest.skip(f"PCA skipped: {pca_result.note}")
    used = set(pca_result.feature_list)
    # Hard policy: these MUST NOT appear in the PCA feature list.
    assert "component_subbass_energy_ratio" not in used
    assert "component_total_inharmonic_energy_ratio" not in used
    assert "model_inharmonic_weight" not in used
    # Aliases / diagnostics MUST NOT appear either.
    assert "harmonic_energy_ratio" not in used
    assert "harmonic_energy_sum" not in used
    # At least one independent canonical metric MUST be present.
    assert "component_harmonic_energy_ratio" in used


def test_pca_default_feature_list_excludes_legacy_and_alias_density_metrics(dictionary):
    feats = set(dictionary.canonical_independent_for_pca())
    forbidden = {
        "density_weighted_sum_cdm_mean",
        "Combined Density Metric",
        "Weighted Combined Metric",
        "Total sum",
        "density_metric_raw",
        "density_metric_normalized",
        "energy_weighted_component_density_diagnostic",
    }
    assert feats.isdisjoint(forbidden)


def test_final_density_primary_markers_and_legacy_not_primary(dictionary):
    assert "final_note_density_salience_weighted" in dictionary.metrics
    assert "final_note_density_count_based" in dictionary.metrics
    assert dictionary.metrics["final_note_density_salience_weighted"]["status"] != "legacy"
    assert dictionary.metrics["final_note_density_count_based"]["status"] != "legacy"
    if "Combined Density Metric" in dictionary.metrics:
        assert dictionary.metrics["Combined Density Metric"]["status"] == "legacy"


def test_pca_loadings_shape(synthetic_corpus, dictionary):
    pca_feats = dictionary.canonical_independent_for_pca()
    pca_result = run_pca_on_canonical(synthetic_corpus, pca_feats, minimum_samples=4)
    if pca_result.loadings.empty:
        pytest.skip(f"PCA skipped: {pca_result.note}")
    # One row per used feature; at most 3 PC columns + the Feature column.
    assert "Feature" in pca_result.loadings.columns
    n_pc_cols = sum(1 for c in pca_result.loadings.columns if c.startswith("PC"))
    assert 1 <= n_pc_cols <= 3
    assert len(pca_result.explained_variance) == n_pc_cols


# ---------------------------------------------------------------------------
# End-to-end pipeline + report writers
# ---------------------------------------------------------------------------
def test_validate_corpus_emits_required_sections(synthetic_corpus, dictionary):
    report = validate_corpus(
        synthetic_corpus,
        dictionary,
        group_by=["Instrument"],
        correlation_threshold=0.90,
    )
    # required sections
    assert not report.corpus_summary.empty
    assert not report.canonical_coverage.empty
    assert not report.descriptive_stats.empty
    assert not report.correlation_matrix.empty
    assert not report.interpretation_limits.empty
    assert not report.pca_feature_list.empty
    # diagnostic / legacy presence MUST trigger a warning, not an error
    assert any("diagnostic/legacy" in w for w in report.warnings)


def test_validate_corpus_warns_about_missing_canonical(synthetic_corpus, dictionary):
    df = synthetic_corpus.drop(columns=["spectral_entropy"])
    report = validate_corpus(df, dictionary)
    # The warning must mention the missing metric by name.
    assert any("spectral_entropy" in w and "MISSING" in w for w in report.warnings)


def test_validate_corpus_reports_high_correlation_pairs(synthetic_corpus, dictionary):
    report = validate_corpus(synthetic_corpus, dictionary, correlation_threshold=0.95)
    assert not report.high_correlations.empty
    # The redundancy_type column must be populated.
    types = set(report.high_correlations["redundancy_type"].unique().tolist())
    assert types.issubset({"algebraic", "empirical"})


def test_report_excel_written(tmp_path: Path, synthetic_corpus, dictionary):
    pytest.importorskip("openpyxl")
    report = validate_corpus(synthetic_corpus, dictionary, group_by=["Instrument"])
    out = tmp_path / "report.xlsx"
    write_report_excel(out, report)
    assert out.is_file()
    sheets = set(pd.ExcelFile(out).sheet_names)
    # All required section sheets present.
    for s in (
        "Corpus_Summary",
        "Canonical_Coverage",
        "Descriptive_Stats",
        "Correlation_Matrix",
        "PCA_Feature_List",
        "Interpretation_Limits",
        "Warnings",
        "Settings",
    ):
        assert s in sheets, s


def test_report_markdown_written_with_no_musicological_terms(tmp_path: Path, synthetic_corpus, dictionary):
    report = validate_corpus(synthetic_corpus, dictionary, group_by=["Instrument"])
    out = tmp_path / "report.md"
    write_report_markdown(out, report)
    md_text = out.read_text(encoding="utf-8")
    # Required section headers present.
    assert "# Canonical metrics validation report" in md_text
    assert "## Corpus summary" in md_text
    assert "## Canonical metric coverage" in md_text
    assert "## PCA feature list" in md_text
    assert "## Interpretation limits" in md_text
    # PCA policy disclaimer present.
    assert "independent_for_pca=true" in md_text
    # The script MUST NOT introduce musicological inference language. Strings
    # below MAY only appear if they occur verbatim in metrics_dictionary.json
    # — none of them do, so they must be absent from the report.
    blacklisted = (
        "orchestration function",
        "perceptual tension",
        "timbral salience",
        "implies higher dissonance",
    )
    md_lower = md_text.lower()
    for phrase in blacklisted:
        assert phrase not in md_lower, (
            f"banned musicological phrase {phrase!r} found in validation report"
        )


def test_load_canonical_metrics_from_workbook_roundtrip(tmp_path: Path, synthetic_corpus):
    pytest.importorskip("openpyxl")
    wb = tmp_path / "wb.xlsx"
    # Write a workbook with a Canonical_Metrics sheet, mimicking the
    # compile_metrics.py contract.
    with pd.ExcelWriter(wb, engine="openpyxl") as writer:
        synthetic_corpus.to_excel(writer, sheet_name="Canonical_Metrics", index=False)
    loaded = load_canonical_metrics_from_workbook(wb)
    assert len(loaded) == len(synthetic_corpus)
    assert "component_harmonic_energy_ratio" in loaded.columns


def test_load_canonical_metrics_missing_sheet_raises(tmp_path: Path):
    pytest.importorskip("openpyxl")
    wb = tmp_path / "wb.xlsx"
    with pd.ExcelWriter(wb, engine="openpyxl") as writer:
        pd.DataFrame({"x": [1, 2, 3]}).to_excel(writer, sheet_name="Other", index=False)
    with pytest.raises(KeyError):
        load_canonical_metrics_from_workbook(wb)


def test_validate_corpus_pca_with_too_few_samples_returns_note():
    # 3 rows < default pca_minimum_samples=4 → must produce an empty PCA
    # result with a non-empty note (not crash).
    md = MetricDictionary.load(REPO_ROOT / "metrics_dictionary.json")
    pca_feats = md.canonical_independent_for_pca()
    df = pd.DataFrame(
        {f: np.linspace(0.1, 0.9, 3) for f in pca_feats[:3]}
    )
    df["Note"] = ["A4", "B4", "C5"]
    result = run_pca_on_canonical(df, pca_feats, minimum_samples=4)
    assert result.loadings.empty
    assert "PCA skipped" in result.note


# ---------------------------------------------------------------------------
# Defence-in-depth helpers
# ---------------------------------------------------------------------------
def test_is_numeric_non_constant_handles_strings():
    s = pd.Series(["a", "b", "c"])
    assert _is_numeric_non_constant(s) is False


def test_is_numeric_non_constant_detects_constant_floats():
    s = pd.Series([0.5] * 10)
    assert _is_numeric_non_constant(s) is False


def test_is_numeric_non_constant_passes_real_variation():
    s = pd.Series(np.linspace(0.1, 0.9, 10))
    assert _is_numeric_non_constant(s) is True


def test_build_corpus_summary_records_groups(synthetic_corpus):
    summary = build_corpus_summary(synthetic_corpus, group_by=["Instrument"])
    # one row per group + the global totals
    group_rows = [k for k in summary["Key"].tolist() if k.startswith("group(")]
    assert len(group_rows) >= 2
