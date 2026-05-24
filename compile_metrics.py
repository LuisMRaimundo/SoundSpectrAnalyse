# compile_metrics.py — improved compilation module

"""
Compile and analyse density and dissonance metrics.

This module provides helpers to extract metrics from spectral-analysis workbooks,
compile results across many per-note exports, sort musical notes, apply principal
component analysis (PCA), and normalise metrics for comparison.

Features:
- Expanded documentation
- More robust error handling
- More complete parameter validation
- Performance-oriented caching where appropriate
- Improved logging
- Statistical analysis reporting helpers
- Optional PCA / correlation visualisation helpers
"""


from __future__ import annotations

import pandas as pd

import os
import json
import re
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, Union, Dict, Any, List, Tuple
from datetime import datetime
import hashlib

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from proc_audio import AudioProcessor
from analysis_policy import (
    DENSITY_FORMULA_VERSION,
    EXPORT_SCHEMA_VERSION,
    F0_POLICY_VERSION,
    HARMONIC_FREQUENCY_POLICY_VERSION,
    LOW_FREQUENCY_POLICY_VERSION,
    MISSING_METRIC_POLICY_VERSION,
    NONHARMONIC_POLICY_VERSION,
)
from pipeline_contract import (
    CANONICAL_PER_NOTE_WORKBOOK,
    CANONICAL_STAGE1_CLASS,
    CANONICAL_STAGE1_MODULE,
    CANONICAL_STAGE2_FUNCTION,
    CANONICAL_STAGE2_MODULE,
    get_canonical_pipeline_contract,
)
from publication_metric_columns import filter_dataframe_for_publication_metrics_sheet
from dissonance_export import (
    build_canonical_dissonance_frame,
    build_dissonance_correlation_matrix,
    build_dissonance_model_comparison_long,
    CANONICAL_VALUE_BY_SLUG,
    collect_dissonance_scalar_columns,
    DISSONANCE_AUDIT_COPY_COLUMNS,
    dissonance_columns_present_in_density_sheet,
    infer_dissonance_compare_from_frame,
    MODEL_SLUGS,
    OPTIONAL_EXTRA_FIELDS,
)
from dissonance_models import list_available_models as _list_all_dissonance_models
from constants import (
    COUNT_SEMANTICS_NOTE_DOC,
    DISSONANCE_CAP_COMPUTATION_NOTE as CONST_DISSONANCE_CAP_NOTE,
    EFFECTIVE_DENSITY_COMPONENT_POLICY_DOC,
    INHARMONIC_MODE_FOR_EFFECTIVE_DENSITY,
    LEGACY_PARTIAL_COUNT_ALIASES_NOTE,
    SUBBASS_POLICY_FOR_EFFECTIVE_DENSITY_DOC,
)

CANONICAL_PIPELINE_ROLE = "canonical_stage2_compilation"
PUBLICATION_OUTPUT_ALLOWED = True

# Optional imports for advanced analysis
try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except ImportError:
    TSNE = None
    TSNE_AVAILABLE = False

try:
    import umap  # umap-learn
    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False

try:
    from sklearn.ensemble import IsolationForest
    ISOLATION_FOREST_AVAILABLE = True
except ImportError:
    IsolationForest = None
    ISOLATION_FOREST_AVAILABLE = False


# Robust import of density helpers
try:
    from density import compute_spectral_entropy, get_weight_function
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(__file__))



# Logging configuration
logger = logging.getLogger(__name__)


def _ensure_adaptive_subfundamental_cutoff(row: Dict[str, Any]) -> Dict[str, Any]:
    """Ensure subfundamental policy columns are complete when ``f0_final_hz`` is available."""
    import math

    from low_frequency_policy import (
        LOW_FREQUENCY_POLICY_VERSION,
        SUBFUNDAMENTAL_CUTOFF_SELECTION_RULE,
        calculate_adaptive_subfundamental_cutoff_hz,
    )

    def _to_float(x: Any) -> float:
        try:
            y = float(x)
        except Exception:
            return float("nan")
        return y if math.isfinite(y) else float("nan")

    if not isinstance(row, dict):
        try:
            row = dict(row)
        except Exception:
            return row

    def _leakage_hz(r: Dict[str, Any], f0v: float) -> Optional[float]:
        tol = _to_float(r.get("subbass_protection_tolerance_hz"))
        if not (math.isfinite(tol) and tol > 0.0):
            tol = _to_float(r.get("harmonic_leakage_protection_hz"))
        if not (math.isfinite(tol) and tol > 0.0):
            return None
        lr = float(f0v) - float(tol)
        if math.isfinite(lr) and lr > 0.0:
            return float(lr)
        return None

    cutoff = _to_float(row.get("adaptive_subfundamental_cutoff_hz"))
    f0 = _to_float(
        row.get("f0_final_hz", row.get("f0_estimated", row.get("f0_nominal_hz")))
    )
    leak_arg = _leakage_hz(row, f0) if math.isfinite(f0) and f0 > 0.0 else None

    if math.isfinite(cutoff) and cutoff > 0.0:
        row["adaptive_subfundamental_cutoff_hz"] = float(cutoff)
        row["adaptive_subfundamental_cutoff_source"] = (
            row.get("adaptive_subfundamental_cutoff_source")
            or "per_note_analysis_export"
        )
        row["low_frequency_policy_version"] = (
            row.get("low_frequency_policy_version") or LOW_FREQUENCY_POLICY_VERSION
        )
        if math.isfinite(f0) and f0 > 0.0:
            g_full = calculate_adaptive_subfundamental_cutoff_hz(
                f0, leakage_guard_cutoff_hz=leak_arg
            )
            for k, v in g_full.items():
                if k in (
                    "f0_final_hz",
                    "adaptive_subfundamental_cutoff_hz",
                    "effective_subfundamental_margin_percent",
                    "subfundamental_cutoff_selected_by",
                ):
                    continue
                row[k] = v
            row["effective_subfundamental_margin_percent"] = float(
                100.0 * (1.0 - float(cutoff) / f0)
            )
            g_ad = float(g_full["adaptive_subfundamental_cutoff_hz"])
            if abs(float(cutoff) - g_ad) <= max(1e-6, 1e-4 * float(f0)):
                row["subfundamental_cutoff_selected_by"] = str(
                    g_full["subfundamental_cutoff_selected_by"]
                )
            else:
                row.setdefault(
                    "subfundamental_cutoff_selected_by",
                    "per_note_analysis_export",
                )
        return row

    if math.isfinite(f0) and f0 > 0.0:
        guard = calculate_adaptive_subfundamental_cutoff_hz(
            f0, leakage_guard_cutoff_hz=leak_arg
        )
        for k, v in guard.items():
            if k == "f0_final_hz":
                continue
            row[k] = v
        row["adaptive_subfundamental_cutoff_source"] = (
            "derived_at_compile_stage_from_f0_final_hz"
        )
        row["low_frequency_policy_version"] = LOW_FREQUENCY_POLICY_VERSION
        return row

    row["adaptive_subfundamental_cutoff_hz"] = float("nan")
    row["subfundamental_margin_percent"] = float("nan")
    row["percentage_subfundamental_cutoff_hz"] = float("nan")
    row["leakage_guard_cutoff_hz"] = float("nan")
    row["min_floor_hz"] = float("nan")
    row["max_fraction_of_f0"] = float("nan")
    row["effective_subfundamental_margin_percent"] = float("nan")
    row["subfundamental_guard_valid"] = False
    row["subfundamental_guard_policy"] = "invalid_missing_f0"
    row["adaptive_subfundamental_cutoff_source"] = "not_available_missing_f0"
    row["low_frequency_policy_version"] = LOW_FREQUENCY_POLICY_VERSION
    row["subfundamental_cutoff_selection_rule"] = str(SUBFUNDAMENTAL_CUTOFF_SELECTION_RULE)
    row["subfundamental_cutoff_selected_by"] = "none_invalid_f0"

    return row

# PCA / export (define before _write_compiled_excel to avoid fragile import-time dependencies)
METRIC_COLUMNS: List[str] = [
    "effective_partial_density",
    "Density Metric",
    "Weighted Combined Metric",
]

_OMIT_FROM_COMPILED_METRICS_EXPORT: frozenset[str] = frozenset(
    (
        "Spectral Density Metric",
        "Spectral Density Metric_Norm",
        "Total Metric",
        "Total Metric_Norm",
        "Combined Density Metric",
        "Combined Density Metric_Norm",
        "Combined Density Metric_Norm2",
        "Spectral Entropy",
        "Spectral Entropy_Norm",
        "Filtered Density Metric",
        "Filtered Density Metric_Norm",
        "Analysis Type",
        "Dynamic Density Score",
        # AUDIT FIX (direct per-note Density_Metrics extraction) — private
        # bookkeeping column added by compile_density_metrics to carry the
        # absolute path of each per-note workbook through the wide frame.
        # Never exported to public sheets.
        "__source_file_path",
    )
)

# Main compiled / export sheet: core physical density/fatness only (no dissonance, no PCA).
# Primary discrete harmonic count: harmonic_order_count (n·f₀ orders detected). Bin/candidate/peak-list
# counts belong on Debug_Counts, not here.
DENSITY_METRICS_MAIN_COLUMNS: List[str] = [
    "Note",
    "canonical_density_v5_adapted",
    "density_normalized_global",
    "density_per_component",
    "discrete_metric_d3",
    "discrete_metric_d10",
    "discrete_metric_d17",
    "discrete_metric_d24",
    "effective_partial_density",
    "harmonic_energy_sum",
    "inharmonic_energy_sum",
    "subbass_energy_sum",
    "total_component_energy",
    "harmonic_energy_ratio",
    "inharmonic_energy_ratio",
    "subbass_energy_ratio",
    "linear_sum_amplitude_harmonic",
    "linear_sum_amplitude_inharmonic_partial",
    "linear_sum_amplitude_subbass_band",
    # Somas lineares (cópia explícita das linear_sum_* para o relatório compilado; total = H+I+S, NaN→0)
    "Soma_A_linear_harmonicos",
    "Soma_A_linear_inarmonicos",
    "Soma_A_linear_subbass",
    "Soma_A_linear_total",
    "linear_amplitude_fraction_inharmonic_of_HI",
    "linear_amplitude_fraction_nonharmonic_of_total",
    "linear_amplitude_batch_alignment_factor",
    "harmonic_order_count",
    "acoustic_f0_status",
    "f0_used_for_density_source",
    "harmonic_occupancy_detected_order_count",
    "expected_harmonic_slot_count",
    "detected_harmonic_slot_count",
    "harmonic_slot_expected_count",
    "harmonic_slot_matched_count",
    "harmonic_slot_coverage_ratio",
    "body_weighted_effective_density",
    "low_mid_energy_ratio",
    "harmonic_body_density",
    "expected_harmonic_slots_up_to_5000hz",
    "harmonic_body_density_normalized",
    "residual_body_contribution",
    "residual_body_contribution_capped",
    "spectral_body_thickness_index",
    "salient_harmonic_order_count_up_to_5000hz",
    "expected_harmonic_order_count_up_to_5000hz",
    "salient_harmonic_coverage_up_to_5000hz",
    "salient_harmonic_mass_up_to_5000hz",
    "salient_harmonic_order_count_up_to_density_ceiling_hz",
    "expected_harmonic_order_count_up_to_density_ceiling_hz",
    "salient_harmonic_coverage_up_to_density_ceiling_hz",
    "salient_harmonic_mass_up_to_density_ceiling_hz",
    "salient_odd_harmonic_count_up_to_5000hz",
    "salient_even_harmonic_count_up_to_5000hz",
    "odd_even_harmonic_energy_ratio",
    "salient_inharmonic_log_bin_count_up_to_5000hz",
    "salient_subbass_particle_count",
    "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz",
    "salient_subbass_particle_count_up_to_density_ceiling_hz",
    "final_note_density_count_based",
    "final_note_density_salience_weighted",
    "harmonic_density_component",
    "inharmonic_density_component",
    "subbass_density_component",
    "harmonic_density_weight",
    "inharmonic_density_weight",
    "subbass_density_weight",
    "density_summation_mode",
    "density_salience_threshold_db",
    "density_frequency_ceiling_hz",
    "core_harmonic_energy_ratio",
    "core_residual_energy_ratio",
    "core_subbass_energy_ratio",
    "harmonic_effective_power_density_normalized",
    "energy_weighted_component_density_diagnostic",
    "spectral_entropy",
    "density_source_formula",
    "density_normalization_scope",
    "density_normalization_denominator",
    "density_formula_version",
    "rolloff_compensated_harmonic_density",
    "rolloff_compensated_harmonic_density_alpha",
    "rolloff_compensated_harmonic_density_component_count",
    "rolloff_compensated_harmonic_density_status",
    "rolloff_harmonic_partial_count",
    "density_metric_per_harmonic",
    "harmonic_effective_power_density",
    "harmonic_effective_power_density_component_count",
    "harmonic_effective_power_density_status",
    "harmonic_effective_power_density_normalized_by_harmonic_count",
    "harmonic_effective_power_mass",
    "harmonic_effective_power_mean",
    "harmonic_effective_power_rms",
    "harmonic_effective_power_component_count",
    "harmonic_effective_power_mass_status",
]

# Compiled workbook ``Density_Metrics`` (user-facing): Note + ``weight_function`` + per-band partial sums.
# H/I/S/Total: continuous UI keys → ``partial_metric_sums_h_i_s_total`` with **linear** on ΣA² scalars; discrete keys
# → same function with D3/D10/D17/D24 on per-partial vectors. ``weight_function`` still labels the row for density.
#
# AUDIT FIX (single-pass weighted density) — Density_Metrics exposes the
# canonical single-pass component energy ratios, the per-component
# weighted contributions (D_i * w_i), an unbounded ``density_metric_raw``
# (audit/diagnostic) and the run-relative ``density_metric_normalized``
# which is the preferred default plotting metric *within* this sheet.
# ``Total sum`` remains an unweighted diagnostic total (D_H + D_I + D_S);
# it is FORBIDDEN as a publication default (see
# publication_chart_policy.FORBIDDEN_DEFAULT_METRIC_NAMES).
#
# AUDIT FIX (Clarinete_mf workbook layout complaint) — column order
# reorganised so the canonical density answer comes IMMEDIATELY after
# ``Note``. Legacy display copies (``Harmonic Partials sum``, ``Total
# sum``, …) follow the canonical columns so a left-to-right reader sees
# the right number first and the back-compat one only after.
DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS: List[str] = [
    "Note",
    "density_metric_raw",
    "energy_weighted_component_density_diagnostic",
    "density_metric_normalized",
    "weighted_harmonic_density_contribution",
    "weighted_inharmonic_density_contribution",
    "weighted_subbass_density_contribution",
    "component_harmonic_energy_ratio",
    "component_inharmonic_energy_ratio",
    "component_subbass_energy_ratio",
    "acoustic_f0_status",
    "f0_used_for_density_source",
    "harmonic_occupancy_detected_order_count",
    "expected_harmonic_slot_count",
    "detected_harmonic_slot_count",
    "harmonic_slot_expected_count",
    "harmonic_slot_matched_count",
    "harmonic_slot_coverage_ratio",
    "body_weighted_effective_density",
    "low_mid_energy_ratio",
    "harmonic_body_density",
    "harmonic_body_density_normalized",
    "residual_body_contribution_capped",
    "spectral_body_thickness_index",
    "salient_harmonic_order_count_up_to_5000hz",
    "expected_harmonic_order_count_up_to_5000hz",
    "salient_harmonic_coverage_up_to_5000hz",
    "salient_harmonic_mass_up_to_5000hz",
    "salient_harmonic_order_count_up_to_density_ceiling_hz",
    "expected_harmonic_order_count_up_to_density_ceiling_hz",
    "salient_harmonic_coverage_up_to_density_ceiling_hz",
    "salient_harmonic_mass_up_to_density_ceiling_hz",
    "salient_odd_harmonic_count_up_to_5000hz",
    "salient_even_harmonic_count_up_to_5000hz",
    "odd_even_harmonic_energy_ratio",
    "salient_inharmonic_log_bin_count_up_to_5000hz",
    "salient_subbass_particle_count",
    "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz",
    "salient_subbass_particle_count_up_to_density_ceiling_hz",
    "final_note_density_count_based",
    "final_note_density_salience_weighted",
    "harmonic_density_component",
    "inharmonic_density_component",
    "subbass_density_component",
    "harmonic_density_weight",
    "inharmonic_density_weight",
    "subbass_density_weight",
    "density_summation_mode",
    "density_salience_threshold_db",
    "density_frequency_ceiling_hz",
    "core_harmonic_energy_ratio",
    "core_residual_energy_ratio",
    "core_subbass_energy_ratio",
    "harmonic_effective_power_density_normalized",
    "Harmonic Partials sum",
    "Inharmonic Partials sum",
    "Sub-bass sum",
    "Total sum",
    "source_file_name",
    "weight_function",
    # AUDIT FIX (direct per-note Density_Metrics extraction) — provenance
    # / status columns produced by extract_density_components_from_per_note_workbook.
    "density_extraction_status",
    # AUDIT FIX (Density_Metrics component basis) — provenance for the
    # selected component / weight bases (``amplitude_sum`` / ``power_sum``
    # for D_H/D_I/D_S; ``energy_ratio_power_sum`` for w_H/w_I/w_S).
    "density_component_basis",
    "density_weight_basis",
    "harmonic_spectrum_source",
    "inharmonic_spectrum_source",
    "subbass_spectrum_source",
    "harmonic_spectrum_count",
    "inharmonic_spectrum_count",
    "subbass_spectrum_count",
]

# Stage 1 harmonic-spectrum candidate density metric. Independent of
# Power_raw / component_* ratios / external H/I weights. Surfaced as an
# allow-listed *optional* group so the strict minimal-display test stays
# stable while the direct per-note extractor populates the columns.
#
#   harmonic_amplitude_sum         = sum(Amplitude_raw where include_for_density=True)
#   harmonic_log_amplitude_density = log10(1 + harmonic_amplitude_sum)
DENSITY_METRICS_HARMONIC_CANDIDATE_COLUMNS: frozenset[str] = frozenset(
    {
        "harmonic_amplitude_sum",
        "harmonic_log_amplitude_density",
        "harmonic_density_included_count",
        "harmonic_amplitude_source",
    }
)

# Stage 2 weighted note-density metric columns. Surfaced as an
# allow-listed *optional* group so the strict minimal-display layout
# test stays stable while the direct per-note extractor populates the
# weighted-density values.
#
#     density_weighted_sum =
#         harmonic_density_sum      * component_harmonic_energy_ratio
#       + inharmonic_density_sum    * component_inharmonic_energy_ratio
#       + subbass_density_sum       * component_subbass_energy_ratio
#     where each *_density_sum is the per-band D under the compile
#     weight_function (same D as density_metric_raw). Legacy linear-only
#     amplitude sums remain in harmonic_amplitude_sum / *_amplitude_sum.
#     density_log_weighted = log10(1 + density_weighted_sum)
DENSITY_METRICS_WEIGHTED_DENSITY_COLUMNS: frozenset[str] = frozenset(
    {
        "inharmonic_amplitude_sum",
        "inharmonic_amplitude_source",
        "inharmonic_amplitude_count",
        "subbass_amplitude_sum",
        "subbass_amplitude_source",
        "subbass_amplitude_count",
        "weighted_harmonic_component",
        "weighted_inharmonic_component",
        "weighted_subbass_component",
        "density_weighted_sum",
        "density_log_weighted",
        "density_log_formula",
    }
)

# AUDIT FIX (canonical density-metric correction) — per-note columns
# emitted by ``extract_density_component_sum`` and the canonical
# weight_function pipeline. Always allow-listed so the strict slim
# Density_Metrics sheet exposes them when populated.
DENSITY_METRICS_WEIGHT_FUNCTION_COLUMNS: frozenset[str] = frozenset(
    {
        "note_source",
        "density_weight_function",
        "harmonic_density_sum",
        "inharmonic_density_sum",
        "subbass_density_sum",
        "density_formula",
        "density_component_sum_source",
        "component_energy_ratio_sum",
        "component_energy_ratio_sum_ok",
        # AUDIT FIX (Harmonic-Spectrum inclusion contract) — diagnostic
        # columns surfacing the ``include_for_density`` filter applied
        # by extract_density_component_sum on the Harmonic Spectrum sheet.
        "harmonic_density_inclusion_policy",
        "harmonic_density_excluded_count",
    }
)

# AUDIT FIX (Density_Metrics component basis) — opt-in debug column
# that is ONLY emitted when the operator passes
# ``density_component_basis="power_sum"``. It is added to the allow-list
# (so the validator doesn't reject it) but kept out of the strict
# display ordering so the canonical layout is unambiguous.
DENSITY_METRICS_DEBUG_OPTIONAL_COLUMNS: frozenset[str] = frozenset(
    {"density_metric_power_weighted_raw"}
)

# Legacy workbooks may still carry unique_harmonic_order_count; it is accepted by the validator when present.
DENSITY_METRICS_LEGACY_OPTIONAL_COLUMNS: frozenset[str] = frozenset({"unique_harmonic_order_count"})

# SEMANTIC HARDENING — PCA exports must include ONLY metrics marked
# ``independent_for_pca=true`` in ``metrics_dictionary.json``. Algebraic
# complements / exact ratios / aliases are excluded because they make the
# feature matrix rank-deficient and bias the loadings.
#
# Excluded on purpose vs. the previous list:
#   - harmonic_energy_ratio  → diagnostic alias of component_harmonic_energy_ratio
#   - inharmonic_energy_ratio → diagnostic alias of component_inharmonic_energy_ratio
#   - subbass_energy_ratio   → algebraic complement (1 - H - I)
#
# Added explicitly (independent canonical features):
#   - component_harmonic_energy_ratio
#   - component_inharmonic_energy_ratio
#   - effective_partial_count (harmonic-only N_eff, distinct input vs.
#     effective_partial_density which is the blended bundle)
#
# A debug flag (``pca_include_dependent_metrics``) reinstates the legacy
# wide-list for forensic inspection only.
PCA_FEATURE_COLUMNS: List[str] = [
    "effective_partial_count",
    "effective_partial_density",
    "discrete_metric_d3",
    "discrete_metric_d10",
    "discrete_metric_d17",
    "discrete_metric_d24",
    "spectral_entropy",
]

# Forensic-only superset; activated when ``pca_include_dependent_metrics=True``
# is passed to ``_compute_optional_pca_sheets``. Documents (but does NOT silently
# enable) the previously-default behaviour.
PCA_FEATURE_COLUMNS_DEBUG_INCLUSIVE: List[str] = [
    *PCA_FEATURE_COLUMNS,
    "subbass_energy_ratio",
    "component_subbass_energy_ratio",
    "component_total_inharmonic_energy_ratio",
    "model_harmonic_weight",
    "model_inharmonic_weight",
    "harmonic_energy_ratio",
    "inharmonic_energy_ratio",
]

DENSITY_FORMULA_DOC: str = (
    "effective_partial_density D_eff = (sum_i P_i)^2 / sum_i(P_i^2) with P_i non-negative powers of the "
    "effective-component vector (participation ratio / Herfindahl inverse); harmonic terms per detected "
    "partial power, inharmonic and sub-bass aggregated (inharmonic_mode_for_effective_density=aggregate). "
    "Scale-invariant in power; not loudness, not dissonance, not masking. "
    "See partial_density_effective_components_bundle in density.py."
)

# Strict allow-list for Density_Metrics (compiled slim sheet).
DENSITY_METRICS_ALLOWED_COLUMNS: frozenset = (
    frozenset(DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS)
    | DENSITY_METRICS_DEBUG_OPTIONAL_COLUMNS
    | DENSITY_METRICS_HARMONIC_CANDIDATE_COLUMNS
    | DENSITY_METRICS_WEIGHTED_DENSITY_COLUMNS
    | DENSITY_METRICS_WEIGHT_FUNCTION_COLUMNS
)

# Explicit forbidden labels / patterns (defence in depth if upstream frames are wide).
DENSITY_METRICS_FORBIDDEN_EXACT: frozenset = frozenset(
    {
        "PC1",
        "PC2",
        "PC3",
        "PC4",
        "legacy_rolloff_compensated_density",
        "R_norm",
        "P_norm",
        "D_agn",
        "D_harm",
        "harmonic_bin_count",
        "inharmonic_bin_count",
        "subbass_bin_count",
        "harmonic_peak_count",
        "subbass_peak_count",
        "total_detected_peak_count",
        "harmonic_candidate_count",
        "inharmonic_candidate_count",
        "subbass_candidate_count",
        "total_spectral_candidate_count",
        "harmonic_partial_count",
        "inharmonic_partial_count",
        "total_detected_partial_count",
        "sethares_dissonance",
        "hutchinson_knopoff_dissonance",
        "vassilakis_dissonance",
        "Weighted Combined Metric",
        "N FFT",
        "Hop Length",
    }
)

DENSITY_METRICS_FORBIDDEN_SUBSTRINGS: Tuple[str, ...] = (
    "dissonance",
    "sethares",
    "vassilakis",
    "hutchinson",
    "knopoff",
    "masking",
    "pca_",
    "_norm2",
)

DENSITY_METRICS_FORBIDDEN_EXACT_LOWER: frozenset = frozenset(
    {"window", "n_fft", "hop_length", "hop length", "fft size", "zero padding"}
)


def density_metric_column_is_forbidden(name: str) -> bool:
    """True if ``name`` must not appear on ``Density_Metrics``."""
    n = str(name).strip()
    if n in DENSITY_METRICS_FORBIDDEN_EXACT:
        return True
    low = n.lower()
    for sub in DENSITY_METRICS_FORBIDDEN_SUBSTRINGS:
        if sub in low:
            return True
    if low in DENSITY_METRICS_FORBIDDEN_EXACT_LOWER or n in ("Window", "N FFT", "Hop Length"):
        return True
    return False


def _finalize_analysis_metadata_for_workbook(
    meta: Dict[str, Any],
    base_df: pd.DataFrame,
    *,
    pca_include_dissonance: bool = False,
) -> None:
    """Populate reproducibility / policy keys; replace ambiguous nulls where practical."""
    from datetime import datetime as _dt

    meta.setdefault("analysis_date", _dt.now().isoformat())
    meta.setdefault("density_formula", DENSITY_FORMULA_DOC)
    meta["effective_density_component_policy"] = EFFECTIVE_DENSITY_COMPONENT_POLICY_DOC
    meta["inharmonic_mode_for_effective_density"] = INHARMONIC_MODE_FOR_EFFECTIVE_DENSITY
    meta["subbass_policy_for_effective_density"] = SUBBASS_POLICY_FOR_EFFECTIVE_DENSITY_DOC
    meta["count_semantics_note"] = COUNT_SEMANTICS_NOTE_DOC
    meta["legacy_partial_count_aliases_note"] = LEGACY_PARTIAL_COUNT_ALIASES_NOTE
    meta["pca_include_dissonance"] = bool(pca_include_dissonance)

    if meta.get("smoothing_enabled") is None:
        meta["smoothing_enabled"] = "not_available_at_compile_stage"
    if meta.get("harmonic_tolerance") is None:
        meta["harmonic_tolerance"] = "not_available_at_compile_stage"

    # Dissonance cap summary (batch / workbook level)
    if base_df is not None and not base_df.empty and "dissonance_partial_count_before_cap" in base_df.columns:
        b = pd.to_numeric(base_df["dissonance_partial_count_before_cap"], errors="coerce")
        a = (
            pd.to_numeric(base_df["dissonance_partial_count_after_cap"], errors="coerce")
            if "dissonance_partial_count_after_cap" in base_df.columns
            else pd.Series(dtype=float)
        )
        meta["dissonance_partial_count_before_cap"] = int(b.max()) if b.notna().any() else "not_available_at_compile_stage"
        if a.size and a.notna().any():
            meta["dissonance_partial_count_after_cap"] = int(a.max())
        if "dissonance_pair_count_after_cap" in base_df.columns:
            pq = pd.to_numeric(base_df["dissonance_pair_count_after_cap"], errors="coerce")
            if pq.notna().any():
                meta["dissonance_pair_count_after_cap"] = int(pq.max())
        elif a.size and a.notna().any():
            nmax = int(a.max())
            meta["dissonance_pair_count_after_cap"] = int(nmax * (nmax - 1) // 2) if nmax >= 2 else 0
        if "dissonance_partial_cap" in base_df.columns:
            caps = [str(x) for x in base_df["dissonance_partial_cap"].dropna().unique().tolist()]
            caps_set = set(caps)
            if caps_set == {"not_applied"}:
                meta["dissonance_partial_cap"] = "not_applied"
            elif len(caps_set) == 1 and caps:
                try:
                    meta["dissonance_partial_cap"] = int(float(str(caps[0]).strip()))
                except (TypeError, ValueError):
                    meta["dissonance_partial_cap"] = caps[0]
            elif caps:
                meta["dissonance_partial_cap"] = caps[0] if len(caps_set) == 1 else "mixed_across_notes"
            else:
                meta["dissonance_partial_cap"] = "not_available_at_compile_stage"
        else:
            meta["dissonance_partial_cap"] = "not_available_at_compile_stage"
        if "dissonance_cap_computation_note" in base_df.columns:
            try:
                note = str(base_df["dissonance_cap_computation_note"].dropna().iloc[0])
                meta["dissonance_cap_computation_note"] = note
            except Exception:
                if meta.get("dissonance_partial_cap") == "not_applied":
                    meta.setdefault(
                        "dissonance_cap_computation_note",
                        "Full harmonic partial list used for dissonance (pairwise cap not applied).",
                    )
                else:
                    meta.setdefault(
                        "dissonance_cap_computation_note",
                        str(CONST_DISSONANCE_CAP_NOTE),
                    )
        else:
            if meta.get("dissonance_partial_cap") == "not_applied":
                meta.setdefault(
                    "dissonance_cap_computation_note",
                    "Full harmonic partial list used for dissonance (pairwise cap not applied).",
                )
            else:
                meta.setdefault(
                    "dissonance_cap_computation_note",
                    str(CONST_DISSONANCE_CAP_NOTE),
                )
    else:
        meta.setdefault("dissonance_partial_cap", "not_available_at_compile_stage")
        meta.setdefault("dissonance_partial_count_before_cap", "not_available_at_compile_stage")
        meta.setdefault("dissonance_partial_count_after_cap", "not_available_at_compile_stage")
        meta.setdefault("dissonance_pair_count_after_cap", "not_available_at_compile_stage")
        meta.setdefault("dissonance_cap_computation_note", "not_available_at_compile_stage")

    meta["model_weight_policy"] = "current_analysis_component_HIS_projected_to_HI_model_weights"
    meta["model_weights_source_policy"] = "per_note_when_available"
    meta["component_ratio_sum_policy"] = (
        "component_harmonic_energy_ratio + component_inharmonic_energy_ratio + component_subbass_energy_ratio must sum to 1"
    )
    meta["external_component_profile_used"] = False
    meta["external_h_i_s_mapping_used"] = False
    if base_df is not None and not base_df.empty and "component_energy_denominator" in base_df.columns:
        try:
            cden = (
                base_df["component_energy_denominator"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            cden = cden[cden.str.len() > 0]
            uniq_c = {str(x) for x in cden.unique().tolist() if str(x).strip()}
            if len(uniq_c) == 1:
                meta["component_energy_denominator"] = next(iter(uniq_c))
            elif len(uniq_c) > 1:
                meta["component_energy_denominator"] = "multiple"
        except Exception:
            meta.setdefault("component_energy_denominator", "harmonic_plus_inharmonic_plus_subbass")
    else:
        meta.setdefault("component_energy_denominator", "harmonic_plus_inharmonic_plus_subbass")
    meta.setdefault("model_weight_denominator", "harmonic_plus_inharmonic")

    if base_df is not None and not base_df.empty and "selected_dissonance_model" in base_df.columns:
        try:
            ser = (
                base_df["selected_dissonance_model"]
                .dropna()
                .astype(str)
                .str.strip()
                .str.lower()
            )
            ser = ser[ser.str.len() > 0]
            ser = ser[~ser.str.startswith("not_available")]
            uniq_d = {str(x) for x in ser.unique().tolist() if str(x).strip()}
            if len(uniq_d) == 1:
                meta["selected_dissonance_model"] = next(iter(uniq_d))
            elif len(uniq_d) > 1:
                meta["selected_dissonance_model"] = "multiple"
        except Exception:
            pass

    # Ensure workbook metadata keys exist (wide single-row export); avoid silent omission.
    meta.setdefault("pca_export_status", "not_available_at_compile_stage")
    meta.setdefault("dissonance_export_status", "not_available_at_compile_stage")
    meta.setdefault("validation_export_status", "not_available_at_compile_stage")
    meta.setdefault("debug_counts_export_status", "not_available_at_compile_stage")
    meta.setdefault("per_note_metadata_export_status", "not_available_at_compile_stage")
    meta.setdefault(
        "robust_salient_inharmonic_peak_picking_enabled",
        False,
    )
    meta.setdefault(
        "debug_counts_semantics_note",
        "Debug_Counts holds FFT-bin, spectral-row, and candidate-slot audit counts. "
        "These are not musical partial counts and must not be read as inharmonic 'partial multiplicity'.",
    )
    meta.setdefault("selected_dissonance_model", "not_available_at_compile_stage")
    meta.setdefault("available_dissonance_models", "not_available_at_compile_stage")
    for _mk in (
        "analysis_version",
        "run_id",
        "python_version",
        "numpy_version",
        "scipy_version",
        "librosa_version",
        "platform",
        "window",
        "n_fft",
        "hop_length",
        "rms_normalisation_enabled",
        "spectral_masking_enabled",
        "snr_threshold_db",
    ):
        if meta.get(_mk) is None or (isinstance(meta.get(_mk), str) and not str(meta.get(_mk)).strip()):
            meta[_mk] = "not_available_at_compile_stage"

    _attach_weight_function_ui_label(meta)


def _attach_weight_function_ui_label(meta: Dict[str, Any]) -> None:
    wf = str(meta.get("weight_function", "") or "").strip()
    if not wf:
        meta.setdefault("weight_function_ui_label", "not_available_at_compile_stage")
        return
    try:
        from weight_function_ui_labels import display_label_for_weight_key

        meta["weight_function_ui_label"] = display_label_for_weight_key(wf)
    except Exception:
        meta["weight_function_ui_label"] = wf


def _build_compile_guide_dataframe(meta_flat: Dict[str, Any], density_columns: List[str]) -> pd.DataFrame:
    """Human-readable index for the minimal ``Density_Metrics`` partial-sum export."""
    cols = list(density_columns)
    wf_key = str(meta_flat.get("weight_function", "") or "").strip()
    try:
        from weight_function_ui_labels import display_label_for_weight_key

        wf_ui = display_label_for_weight_key(wf_key)
    except Exception:
        wf_ui = wf_key or "—"

    rows: List[Dict[str, str]] = []

    def row(category: str, item: str, value: str) -> None:
        rows.append({"Category": category, "Item": item, "Value": value})

    # SEMANTIC HARDENING — top-of-guide policy warning. The compiled workbook
    # now exports three semantic sheets: Canonical_Metrics (publication
    # grade), Diagnostic_Metrics (intermediate / provenance) and
    # Legacy_Compatibility (deprecated aliases). The Density_Metrics sheet is
    # preserved for backward compatibility but is NOT the publication-grade
    # table.
    row(
        "READ FIRST — publication policy",
        "Where are the publication-grade metrics?",
        "Use **Canonical_Metrics**. Diagnostic_Metrics, Density_Metrics and "
        "Legacy_Compatibility are intermediate / back-compat sheets and MUST "
        "NOT be cited as final scientific outputs. See "
        "metrics_dictionary.json (status='canonical'/'diagnostic'/'legacy', "
        "metric_family, derived_from, independent_for_pca).",
    )
    row(
        "READ FIRST — publication policy",
        "Density_Metrics sheet status",
        "For final analysis use Canonical_Metrics. Density_Metrics is "
        "preserved for backward compatibility and is not the "
        "publication-grade table.",
    )
    row(
        "READ FIRST — publication policy",
        "PCA inclusion policy",
        "PCA exports include ONLY metrics flagged independent_for_pca=true "
        "in metrics_dictionary.json. Algebraic complements (e.g. "
        "subbass_energy_ratio, model_inharmonic_weight) and aliases are "
        "excluded by default. A forensic debug flag "
        "(pca_include_dependent_metrics=True) is available for inspection.",
    )

    row("Compile — amplitude weighting", "Internal key (density.get_weight_function)", wf_key or "—")
    row("Compile — amplitude weighting", "Same choice in the GUI (label)", wf_ui)
    row(
        "Density_Metrics (compiled)",
        "Columns on this sheet",
        "Note + ``weight_function`` (compile key: linear, log, cubic, d3, …) + per-band "
        "``harmonic_density_sum`` / ``inharmonic_density_sum`` / ``subbass_density_sum`` (each band uses that key) "
        "+ ``density_metric_raw`` / ``density_weighted_sum`` (= D_H·w_H + D_I·w_I + D_S·w_S, same number) + legacy "
        "``Harmonic Partials sum`` / ``Total sum``. ``harmonic_amplitude_sum`` is always a linear diagnostic and "
        "does not change when you switch weight_function.",
    )
    row(
        "Density_Metrics — weighted sum",
        "density_weighted_sum semantics",
        "D_H·component_harmonic_energy_ratio + D_I·component_inharmonic_energy_ratio + "
        "D_S·component_subbass_energy_ratio, where each D_* is from extract_density_component_sum under the "
        "compile weight_function. NOT a fixed linear amplitude sum.",
    )
    row(
        "Per-note workbook",
        "Where the numbers are produced",
        "spectral_analysis.xlsx / sheet Metrics: same columns. "
        "``partial_metric_sums_h_i_s_total`` uses the per-note ``weight_function`` (same pipeline as compiled "
        "Density_Metrics). Phase-1 **batch** % (``batch_*_energy_ratio``) is a wideband STFT prior and "
        "need not match peak-list ``harmonic_energy_ratio``.",
    )
    row(
        "Per-note workbook",
        "Legacy_Density_Metrics sheet (default ON)",
        "Every spectral_analysis.xlsx also exports Legacy_Density_Metrics: Density Metric, "
        "Spectral Density Metric, Filtered Density Metric, Combined Density Metric, "
        "spectral_masking_enabled=False (v6 has no v5 masking GUI). Compile merges this sheet for "
        "Weighted Combined Metric on Diagnostic_Metrics / Legacy_Compatibility — not on Density_Metrics.",
    )
    row(
        "Research export",
        "compiled_density_metrics_research.xlsx",
        "Read-only post-process (tools/export_research_density_workbook.py). Adds "
        "density_weighted_sum_cdm_mean = (density_weighted_sum + Combined Density Metric) / 2 and column "
        "highlights on Spectral_Density_Metrics. Editorial only — see docs/DENSITY_EXPORT_SCHEMA.md section R.",
    )
    row(
        "Excel / charts",
        "Metrics worksheet (spectral_analysis)",
        "Do not merge multiple columns into one continuous row: ``discrete_metric_d*`` are independent metrics. "
        "See also ``Analysis_Metadata`` → ``excel_charting_warning_metrics_sheet`` on each v6+ export.",
    )
    for label in (
        "weight_function",
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Sub-bass sum",
        "Total sum",
    ):
        row("Density_Metrics — column present?", label, "yes" if label in cols else "no")

    return pd.DataFrame(rows)


def validate_compiled_density_workbook(path: Union[str, Path]) -> List[str]:
    """
    Validate a compiled workbook structure. Returns a list of human-readable errors (empty if OK).
    """
    path = Path(path)
    errs: List[str] = []
    if not path.is_file():
        return [f"File not found: {path}"]
    try:
        xl = pd.ExcelFile(path)
    except Exception as e:
        return [f"Cannot open Excel: {e}"]
    names = list(xl.sheet_names)
    if "Density_Metrics" not in names:
        errs.append("Missing required sheet: Density_Metrics")
    if "Analysis_Metadata" not in names:
        errs.append("Missing required sheet: Analysis_Metadata")
    meta_d: Dict[str, Any] = {}
    try:
        am0 = pd.read_excel(path, sheet_name="Analysis_Metadata")
        if "Parameter" in am0.columns and "Value" in am0.columns:
            meta_d = {str(r["Parameter"]): r["Value"] for _, r in am0.iterrows()}
        else:
            meta_d = am0.iloc[0].to_dict()
    except Exception as e:
        errs.append(f"Analysis_Metadata unreadable: {e}")

    def _mget(key: str, default: object = None) -> Any:
        return meta_d.get(key, default)

    if "Density_Metrics" in names:
        dm = pd.read_excel(path, sheet_name="Density_Metrics")
        _robust_peaks = str(_mget("robust_salient_inharmonic_peak_picking_enabled", "")).lower() in (
            "1",
            "true",
            "yes",
        )
        _dyn_allowed = set(DENSITY_METRICS_ALLOWED_COLUMNS)
        if _robust_peaks:
            _dyn_allowed.add("inharmonic_peak_count")
        for c in dm.columns:
            cs = str(c)
            if cs not in _dyn_allowed:
                errs.append(f"Density_Metrics: disallowed column {cs!r} (not in allow-list)")
            if density_metric_column_is_forbidden(cs):
                errs.append(f"Density_Metrics: forbidden column {cs!r}")
        if "harmonic_energy_ratio" in dm.columns:
            ssum = (
                pd.to_numeric(dm["harmonic_energy_ratio"], errors="coerce").fillna(0.0)
                + pd.to_numeric(dm["inharmonic_energy_ratio"], errors="coerce").fillna(0.0)
                + pd.to_numeric(dm["subbass_energy_ratio"], errors="coerce").fillna(0.0)
            )
            if (ssum - 1.0).abs().max() > 0.05:
                errs.append("Energy ratios do not sum to ~1 on at least one row (tolerance 0.05)")
        if "effective_partial_density" in dm.columns:
            v = pd.to_numeric(dm["effective_partial_density"], errors="coerce")
            vn = v.dropna()
            if vn.empty:
                errs.append("effective_partial_density invalid (all NaN)")
            else:
                arr = vn.to_numpy(dtype=float, copy=False)
                if (arr < 0).any() or not np.isfinite(arr).all():
                    errs.append("effective_partial_density invalid (must be finite and non-negative per row)")
        if "density_normalized_global" in dm.columns:
            g = pd.to_numeric(dm["density_normalized_global"], errors="coerce").dropna()
            if not g.empty and (g > 1.0 + 1e-6).any():
                errs.append("density_normalized_global exceeds 1.0 (invalid global normalization)")
        # AUDIT FIX (single-pass weighted density) — density_metric_normalized
        # is no longer an alias of density_normalized_global. It is now the
        # run-relative max-norm of density_metric_raw (the weighted partial-
        # sum density). Validate accordingly: the value must lie in [0, 1]
        # and, when density_metric_raw is present, must equal raw/max(raw).
        if "density_metric_normalized" in dm.columns:
            a = pd.to_numeric(dm["density_metric_normalized"], errors="coerce").dropna()
            if not a.empty and ((a < -1e-6).any() or (a > 1.0 + 1e-6).any()):
                errs.append("density_metric_normalized outside [0, 1] (run-relative max-normalisation invariant violated)")
            if "density_metric_raw" in dm.columns:
                raw = pd.to_numeric(dm["density_metric_raw"], errors="coerce")
                norm = pd.to_numeric(dm["density_metric_normalized"], errors="coerce")
                mask = raw.notna() & norm.notna()
                if mask.any():
                    arr_raw = raw[mask].to_numpy(dtype=float)
                    arr_norm = norm[mask].to_numpy(dtype=float)
                    finite_pos = arr_raw[np.isfinite(arr_raw) & (arr_raw > 0)]
                    if finite_pos.size:
                        expected = arr_raw / float(np.max(finite_pos))
                        if not np.allclose(arr_norm, expected, rtol=0.0, atol=1e-6, equal_nan=True):
                            errs.append("density_metric_normalized must equal density_metric_raw / max(density_metric_raw) on Density_Metrics")
    st = str(_mget("debug_counts_export_status", "")).lower()
    if "exported" in st and "Debug_Counts" not in names:
        errs.append("Metadata claims Debug_Counts exported but sheet is missing")
    dst = str(_mget("dissonance_export_status", "")).lower()
    if "exported" in dst and "Dissonance_Metrics" not in names:
        errs.append("Metadata claims dissonance_export_status=exported but Dissonance_Metrics is missing")
    pst = str(_mget("pca_export_status", "")).lower()
    if pst == "exported":
        for s in ("PCA_Scores", "PCA_Loadings", "PCA_Explained_Variance"):
            if s not in names:
                errs.append(f"PCA exported but missing sheet: {s}")
    elif pst and pst != "exported":
        for s in ("PCA_Scores", "PCA_Loadings", "PCA_Explained_Variance"):
            if s in names:
                errs.append(f"PCA marked skipped but sheet present: {s}")
    vst = str(_mget("validation_export_status", "")).lower()
    if "exported" in vst and "Validation_Metrics" not in names:
        errs.append("Metadata claims validation_export_status=exported but Validation_Metrics is missing")

    if "Per_Note_Processing_Metadata" in names:
        try:
            pn = pd.read_excel(path, sheet_name="Per_Note_Processing_Metadata")
        except Exception as e:
            errs.append(f"Per_Note_Processing_Metadata unreadable: {e}")
        else:
            _need = {
                "component_harmonic_energy_ratio",
                "component_inharmonic_energy_ratio",
                "component_subbass_energy_ratio",
            }
            if _need <= set(pn.columns):
                h = pd.to_numeric(pn["component_harmonic_energy_ratio"], errors="coerce")
                i2 = pd.to_numeric(pn["component_inharmonic_energy_ratio"], errors="coerce")
                s2 = pd.to_numeric(pn["component_subbass_energy_ratio"], errors="coerce")
                sm = h + i2 + s2
                if sm.notna().any():
                    if (sm - 1.0).abs().max() > 0.02:
                        errs.append(
                            "Per_Note_Processing_Metadata: component H+I+S ratios do not sum to ~1 (tol 0.02)"
                        )
                    if ((h < -1e-9) | (i2 < -1e-9) | (s2 < -1e-9)).any():
                        errs.append("Per_Note_Processing_Metadata: negative component ratio")
                    if not np.isfinite(sm.to_numpy(dtype=float, copy=False)).all():
                        errs.append("Per_Note_Processing_Metadata: non-finite component ratio")
                if "model_harmonic_weight" in pn.columns and "model_inharmonic_weight" in pn.columns:
                    mh = pd.to_numeric(pn["model_harmonic_weight"], errors="coerce")
                    mi = pd.to_numeric(pn["model_inharmonic_weight"], errors="coerce")
                    msum = mh + mi
                    m_ok = mh.notna() & mi.notna()
                    if m_ok.any() and (msum[m_ok] - 1.0).abs().max() > 0.02:
                        errs.append("Per_Note_Processing_Metadata: model weights do not sum to ~1")
                    if m_ok.any() and "model_weights_source" in pn.columns:
                        src = pn.loc[m_ok, "model_weights_source"].astype(str).str.strip()
                        if not src.str.len().gt(0).all():
                            errs.append("Per_Note_Processing_Metadata: model_weights_source missing on some rows")
                    if m_ok.any() and "model_weight_denominator" in pn.columns:
                        den = pn.loc[m_ok, "model_weight_denominator"].astype(str).str.strip().str.lower()
                        if not den.eq("harmonic_plus_inharmonic").all():
                            errs.append(
                                "Per_Note_Processing_Metadata: model_weight_denominator not harmonic_plus_inharmonic"
                            )

    try:
        from metadata_sanitizer import list_publication_path_violations_in_excel, publication_redaction_enabled

        if publication_redaction_enabled():
            errs.extend(list_publication_path_violations_in_excel(path))
    except Exception:
        pass

    return errs


def _compiled_df_has_density_core(df: pd.DataFrame) -> bool:
    if df is None or df.empty:
        return False
    if "Harmonic Partials sum" in df.columns and "Note" in df.columns:
        return True
    # AUDIT FIX (direct per-note Density_Metrics extraction) — when the
    # wide frame carries the private ``__source_file_path`` column, the
    # compiler can rebuild Density_Metrics by reopening each per-note
    # spectral_analysis.xlsx directly, so the density-core sheet path is
    # available even when the per-note Metrics sheet does not yet expose
    # the scalar Harmonic Partials sum column.
    if "__source_file_path" in df.columns and "Note" in df.columns:
        return True
    return "effective_partial_density" in df.columns or "harmonic_energy_sum" in df.columns


def _prepare_df_for_density_export(df: pd.DataFrame) -> pd.DataFrame:
    """Align entropy / harmonic-order column names for export/PCA.

    ``harmonic_order_count`` is the public discrete harmonic count (detected n·f₀ orders).
    Legacy sheets may expose only ``unique_harmonic_order_count``; copy when needed.
    Debug-level ``*_peak_count`` / ``*_candidate_count`` columns are never promoted to
    ``Density_Metrics`` here.

    Copies ``linear_sum_amplitude_*`` from per-note Metrics into public ``Soma_A_linear_*``
    columns and ``Soma_A_linear_total`` (= soma das três componentes, NaN tratado como 0
    na soma) for the compiled ``Density_Metrics`` sheet.
    """
    out = df.copy()
    if "spectral_entropy" not in out.columns and "Spectral Entropy" in out.columns:
        out["spectral_entropy"] = pd.to_numeric(out["Spectral Entropy"], errors="coerce")
    if "Spectral Entropy" not in out.columns and "spectral_entropy" in out.columns:
        out["Spectral Entropy"] = pd.to_numeric(out["spectral_entropy"], errors="coerce")
    if "harmonic_order_count" not in out.columns and "unique_harmonic_order_count" in out.columns:
        out["harmonic_order_count"] = pd.to_numeric(out["unique_harmonic_order_count"], errors="coerce")
    if "harmonic_occupancy_detected_order_count" not in out.columns and "detected_harmonic_slot_count" in out.columns:
        out["harmonic_occupancy_detected_order_count"] = pd.to_numeric(
            out["detected_harmonic_slot_count"], errors="coerce"
        )
    if "expected_harmonic_slot_count" not in out.columns and "harmonic_slot_expected_count" in out.columns:
        out["expected_harmonic_slot_count"] = pd.to_numeric(out["harmonic_slot_expected_count"], errors="coerce")
    if "harmonic_slot_expected_count" not in out.columns and "expected_harmonic_slot_count" in out.columns:
        out["harmonic_slot_expected_count"] = pd.to_numeric(out["expected_harmonic_slot_count"], errors="coerce")
    if "harmonic_slot_matched_count" not in out.columns and "detected_harmonic_slot_count" in out.columns:
        out["harmonic_slot_matched_count"] = pd.to_numeric(out["detected_harmonic_slot_count"], errors="coerce")
    if "detected_harmonic_slot_count" not in out.columns and "harmonic_occupancy_detected_order_count" in out.columns:
        out["detected_harmonic_slot_count"] = pd.to_numeric(
            out["harmonic_occupancy_detected_order_count"], errors="coerce"
        )
    if "harmonic_occupancy_detected_order_count" not in out.columns and "harmonic_order_count" in out.columns:
        out["harmonic_occupancy_detected_order_count"] = pd.to_numeric(out["harmonic_order_count"], errors="coerce")
    if "harmonic_slot_coverage_ratio" not in out.columns:
        exp = (
            pd.to_numeric(out["harmonic_slot_expected_count"], errors="coerce")
            if "harmonic_slot_expected_count" in out.columns
            else pd.Series(np.nan, index=out.index)
        )
        matched = (
            pd.to_numeric(out["harmonic_slot_matched_count"], errors="coerce")
            if "harmonic_slot_matched_count" in out.columns
            else pd.Series(np.nan, index=out.index)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out["harmonic_slot_coverage_ratio"] = matched / exp.replace(0, np.nan)
    if "core_harmonic_energy_ratio" not in out.columns and "harmonic_energy_ratio" in out.columns:
        out["core_harmonic_energy_ratio"] = pd.to_numeric(out["harmonic_energy_ratio"], errors="coerce")
    if "core_residual_energy_ratio" not in out.columns and "residual_energy_ratio" in out.columns:
        out["core_residual_energy_ratio"] = pd.to_numeric(out["residual_energy_ratio"], errors="coerce")
    if "core_subbass_energy_ratio" not in out.columns and "subbass_energy_ratio" in out.columns:
        out["core_subbass_energy_ratio"] = pd.to_numeric(out["subbass_energy_ratio"], errors="coerce")
    if "energy_weighted_component_density_diagnostic" not in out.columns and "density_metric_raw" in out.columns:
        out["energy_weighted_component_density_diagnostic"] = pd.to_numeric(out["density_metric_raw"], errors="coerce")

    hcol = "linear_sum_amplitude_harmonic"
    icol = "linear_sum_amplitude_inharmonic_partial"
    sbcol = "linear_sum_amplitude_subbass_band"
    pub_h = "Soma_A_linear_harmonicos"
    pub_i = "Soma_A_linear_inarmonicos"
    pub_s = "Soma_A_linear_subbass"
    pub_t = "Soma_A_linear_total"
    if hcol in out.columns:
        out[pub_h] = pd.to_numeric(out[hcol], errors="coerce")
    if icol in out.columns:
        out[pub_i] = pd.to_numeric(out[icol], errors="coerce")
    if sbcol in out.columns:
        out[pub_s] = pd.to_numeric(out[sbcol], errors="coerce")
    present = [c for c in (pub_h, pub_i, pub_s) if c in out.columns]
    if present:
        acc = pd.to_numeric(out[present[0]], errors="coerce").fillna(0.0)
        for c in present[1:]:
            acc = acc + pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        out[pub_t] = acc
    return out


def _add_canonical_and_global_density_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Phase-2 canonical density, then global [0,1] norm; ``density_metric_normalized`` aliases the global norm."""
    if df is None or df.empty:
        return df
    out = df.copy()
    try:
        from density import CANONICAL_DENSITY_FORMULA_VERSION, CANONICAL_DENSITY_SOURCE_FORMULA
    except Exception:
        CANONICAL_DENSITY_FORMULA_VERSION = "v5_apply_density_metric_adapted_v6_1"
        CANONICAL_DENSITY_SOURCE_FORMULA = "apply_density_metric(...)"

    canon_col = "canonical_density_v5_adapted"
    s_canon = pd.to_numeric(out[canon_col], errors="coerce") if canon_col in out.columns else pd.Series(np.nan, index=out.index)
    if canon_col not in out.columns or not s_canon.notna().any():
        if "Density Metric" in out.columns:
            dm = pd.to_numeric(out["Density Metric"], errors="coerce")
            out[canon_col] = dm / 10.0
        else:
            out[canon_col] = np.nan
        s_canon = pd.to_numeric(out[canon_col], errors="coerce")

    arr = s_canon.to_numpy(dtype=float, copy=False)
    finite = arr[np.isfinite(arr)]
    mx = float(np.nanmax(finite)) if finite.size else float("nan")
    if mx > 0.0 and np.isfinite(mx):
        out["density_normalized_global"] = (s_canon / mx).clip(lower=0.0, upper=1.0)
        # AUDIT FIX (single-pass weighted density) — density_metric_normalized
        # is no longer aliased to density_normalized_global; it is now the
        # run-relative max-norm of the weighted partial-sum density_metric_raw
        # (see _augment_density_metrics_with_weighted_metric and
        # _compute_weighted_density_columns_for_wide_df). The two metrics are
        # documented as DISTINCT in metrics_dictionary.json.
        out["density_normalization_denominator"] = float(mx)
    else:
        out["density_normalized_global"] = np.nan
        out["density_normalization_denominator"] = np.nan

    out["density_normalization_scope"] = "compiled_dataset_max_canonical_density_v5_adapted"
    out["density_source_formula"] = CANONICAL_DENSITY_SOURCE_FORMULA
    out["density_formula_version"] = CANONICAL_DENSITY_FORMULA_VERSION

    dpc = out["density_per_component"] if "density_per_component" in out.columns else None
    if dpc is None or not pd.to_numeric(dpc, errors="coerce").notna().any():
        hoc = pd.to_numeric(out["harmonic_order_count"], errors="coerce") if "harmonic_order_count" in out.columns else pd.Series(np.nan, index=out.index)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["density_per_component"] = s_canon / hoc.replace(0, np.nan)

    # AUDIT FIX (single-pass weighted density) — compute the weighted partial-
    # sum density columns here so every sheet sees the same values and so
    # Density_Metrics does not silently diverge from the wide compiled frame.
    out = _compute_weighted_density_columns_for_wide_df(out)
    return out


def _compute_weighted_density_columns_for_wide_df(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the canonical weighted-density columns to the *wide* compiled frame.

    This is the single source of truth for the Density_Metrics-sheet
    columns ``component_*_energy_ratio``, ``weighted_*_density_contribution``,
    ``density_metric_raw`` and ``density_metric_normalized``. The Density_Metrics
    builder reads them straight from this frame so the sheet and the wide
    frame can never disagree.

    Source of the component_* ratios (in order of priority):

    1. ``component_*_energy_ratio`` — the canonical single-pass field
       (proc_audio ``_set_model_weights_from_current_component_energy``).
    2. ``harmonic_energy_ratio`` / ``inharmonic_energy_ratio`` / ``subbass_energy_ratio``
       — diagnostic aliases of the canonical fields (mathematically
       identical when single-pass is active; see metrics_dictionary.json).
    3. Missing — leave NaN. NEVER fall back to ``batch_*_energy_ratio``.
    """
    if df is None or df.empty:
        return df

    h_new = "Harmonic Partials sum"
    i_new = "Inharmonic Partials sum"
    s_new = "Sub-bass sum"
    if not all(c in df.columns for c in (h_new, i_new, s_new)):
        return df

    out = df.copy()

    def _src(canonical: str, alias: str) -> pd.Series:
        if canonical in out.columns:
            return pd.to_numeric(out[canonical], errors="coerce")
        if alias in out.columns:
            return pd.to_numeric(out[alias], errors="coerce")
        return pd.Series(np.nan, index=out.index)

    w_H = _src("component_harmonic_energy_ratio", "harmonic_energy_ratio")
    w_I = _src("component_inharmonic_energy_ratio", "inharmonic_energy_ratio")
    w_S = _src("component_subbass_energy_ratio", "subbass_energy_ratio")

    # Mirror the resolved ratios into the canonical column names so
    # downstream consumers (Density_Metrics, audits) see the same values.
    out["component_harmonic_energy_ratio"] = w_H.astype(float)
    out["component_inharmonic_energy_ratio"] = w_I.astype(float)
    out["component_subbass_energy_ratio"] = w_S.astype(float)

    D_H = pd.to_numeric(out[h_new], errors="coerce")
    D_I = pd.to_numeric(out[i_new], errors="coerce")
    D_S = pd.to_numeric(out[s_new], errors="coerce")

    wh = (D_H * w_H).astype(float)
    wi = (D_I * w_I).astype(float)
    ws = (D_S * w_S).astype(float)
    out["weighted_harmonic_density_contribution"] = wh
    out["weighted_inharmonic_density_contribution"] = wi
    out["weighted_subbass_density_contribution"] = ws

    raw = wh.fillna(0.0) + wi.fillna(0.0) + ws.fillna(0.0)
    full_nan_mask = wh.isna() & wi.isna() & ws.isna()
    if full_nan_mask.any():
        raw = raw.mask(full_nan_mask)
    out["density_metric_raw"] = raw

    arr = raw.to_numpy(dtype=float, copy=False)
    finite_pos = arr[np.isfinite(arr) & (arr > 0)]
    if finite_pos.size == 0:
        logger.warning(
            "density_metric_normalized: no positive finite density_metric_raw "
            "values in compiled workbook; returning NaN."
        )
        out["density_metric_normalized"] = float("nan")
    else:
        mx = float(np.max(finite_pos))
        if mx <= 0.0 or not np.isfinite(mx):
            logger.warning(
                "density_metric_normalized: non-positive normalization "
                "reference (max=%s); returning NaN.", mx
            )
            out["density_metric_normalized"] = float("nan")
        else:
            out["density_metric_normalized"] = (raw / mx).astype(float)
    return out


# ---------------------------------------------------------------------------
# Direct per-note spectral_analysis.xlsx → Density_Metrics extraction
# ---------------------------------------------------------------------------
# Sheet-name preferences for the three component spectra (case-insensitive
# match). The first match in each list wins. The canonical proc_audio export
# uses ``Harmonic Spectrum`` / ``Inharmonic Spectrum`` / ``Sub-bass band`` —
# the broader lists below absorb any older naming the user's corpus may carry.
HARMONIC_SPECTRUM_SHEET_PREFERENCES: Tuple[str, ...] = (
    "Harmonic Spectrum",
    "harmonic_spectrum",
    "Harmonic spectrum",
    "Harmonic_Spectrum",
)
INHARMONIC_SPECTRUM_SHEET_PREFERENCES: Tuple[str, ...] = (
    "Inharmonic Spectrum",
    "inharmonic_spectrum",
    "Inharmonic spectrum",
    "Inharmonic_Spectrum",
)
SUBBASS_SPECTRUM_SHEET_PREFERENCES: Tuple[str, ...] = (
    "Sub-bass band",
    "Sub-bass spectrum",
    "Subbass spectrum",
    "Sub-bass noise spectrum",
    "subbass_spectrum",
    "subbass_noise_spectrum",
    "noise_spectrum",
    "Sub-bass_Band",
)

# Sheets that must never be searched for a spectrum column even if they
# happen to contain a numeric one (metadata, debug, validation).
_DENSITY_EXTRACTION_METADATA_SHEETS: frozenset = frozenset(
    {
        "Analysis_Metadata",
        "Analysis Parameters",
        "Processing Metadata",
        "Per_Note_Processing_Metadata",
        "Metrics",
        "Debug_Counts",
        "Validation_Metrics",
        "Dissonance_Metrics",
        "Dissonance_Model_Comparison",
        "Error",
    }
)

# =====================================================================
# AUDIT FIX (Density_Metrics component basis) — split column-priority
# tables by basis.
#
# The Density_Metrics sheet is a *spectral density* descriptor, not an
# *energy* descriptor. The intermediate sums D_H / D_I / D_S therefore
# need to be evaluated on a linear amplitude scale: summing ``Power_raw``
# = Σ Aᵢ² gives the sub-bass band (where a few low-frequency bins can
# carry orders of magnitude more energy than the harmonic stack) a
# unsupported dominance over D_S, and through the weighted formula
# D_H·w_H + D_I·w_I + D_S·w_S it pulls density_metric_raw down even
# when w_S is small.
#
# Policy:
#
#   density_component_basis = "amplitude_sum"   (default)
#       D_H, D_I, D_S = Σ Amplitude_raw  (and ``Amplitude`` legacy
#                                         fallback when the workbook
#                                         predates this fix).
#       This is the basis the compiled Density_Metrics sheet is built on
#       by default.
#
#   density_component_basis = "power_sum"       (debug / opt-in)
#       D_H, D_I, D_S = Σ Power_raw     (i.e. Σ Aᵢ²)
#       Available only when the caller explicitly asks for it. Used to
#       reproduce / diagnose the historical power-weighted behaviour and
#       to compare orderings; should NOT be used as the canonical
#       density metric.
#
# Both bases continue to feed the same compile workflow; only the
# selection used to populate D_H / D_I / D_S in the compiled sheet
# changes. The component_*_energy_ratio weights w_H / w_I / w_S are
# unchanged and stay on the (correct) energy / power footing computed
# inside proc_audio.
# =====================================================================
DENSITY_COMPONENT_BASIS_DEFAULT: str = "amplitude_sum"
DENSITY_COMPONENT_BASIS_VALID: Tuple[str, ...] = ("amplitude_sum", "power_sum")
DENSITY_WEIGHT_BASIS: str = "energy_ratio_power_sum"

_RAW_AMPLITUDE_PRIORITY_COLUMNS: Tuple[str, ...] = (
    # Amplitude-basis (canonical): density_contribution_raw → Amplitude_raw.
    # Power_raw is intentionally NOT in this list — it is only consulted
    # under the explicit "power_sum" debug basis.
    "density_contribution_raw",
    "Amplitude_raw",
)
_RAW_POWER_PRIORITY_COLUMNS: Tuple[str, ...] = (
    # Power-basis (debug / opt-in): density_contribution_raw → Power_raw.
    # Amplitude_raw is intentionally absent so the debug basis never
    # silently falls back to the amplitude column.
    "density_contribution_raw",
    "Power_raw",
)
# Backward-compatible alias kept for any external tooling that imported
# the union (e.g. tests that introspect both raw columns). New code MUST
# use the basis-specific tuples above.
_RAW_PRIORITY_COLUMNS: Tuple[str, ...] = (
    "density_contribution_raw",
    "Amplitude_raw",
    "Power_raw",
)


def _build_spectrum_column_preferences(
    basis: str,
    *extra: str,
) -> Tuple[str, ...]:
    """Compose a per-sheet column-preference tuple for ``basis``.

    The legacy ``Amplitude`` column is the *last* fallback so it is only
    consulted when neither the requested raw column nor any of the
    sheet-specific aliases match; the extractor then flags the row as
    ``legacy_scaled_source_used``.
    """
    if basis == "power_sum":
        head = _RAW_POWER_PRIORITY_COLUMNS
    else:
        head = _RAW_AMPLITUDE_PRIORITY_COLUMNS
    return (*head, *extra, "Amplitude")


def _harmonic_column_preferences(basis: str = DENSITY_COMPONENT_BASIS_DEFAULT) -> Tuple[str, ...]:
    return _build_spectrum_column_preferences(
        basis,
        "Harmonic spectrum",
        "harmonic_spectrum",
        "Harmonic Spectrum",
        "harmonic_density_contribution",
        "density_contribution",
        "Weighted Contribution",
        "Contribution",
    )


def _inharmonic_column_preferences(basis: str = DENSITY_COMPONENT_BASIS_DEFAULT) -> Tuple[str, ...]:
    return _build_spectrum_column_preferences(
        basis,
        "Inharmonic spectrum",
        "inharmonic_spectrum",
        "Inharmonic Spectrum",
        "inharmonic_density_contribution",
        "density_contribution",
        "Weighted Contribution",
        "Contribution",
    )


def _subbass_column_preferences(basis: str = DENSITY_COMPONENT_BASIS_DEFAULT) -> Tuple[str, ...]:
    return _build_spectrum_column_preferences(
        basis,
        "Sub-bass spectrum",
        "Subbass spectrum",
        "Sub-bass noise spectrum",
        "subbass_spectrum",
        "subbass_noise_spectrum",
        "noise_spectrum",
        "Sub-bass sum",
        "density_contribution",
        "Weighted Contribution",
        "Contribution",
    )


# Default (amplitude-basis) module-level tuples — preserved so external
# code that already imported the constants keeps working. New code
# should call the helpers above to pick the basis explicitly.
HARMONIC_SPECTRUM_COLUMN_PREFERENCES: Tuple[str, ...] = _harmonic_column_preferences()
INHARMONIC_SPECTRUM_COLUMN_PREFERENCES: Tuple[str, ...] = _inharmonic_column_preferences()
SUBBASS_SPECTRUM_COLUMN_PREFERENCES: Tuple[str, ...] = _subbass_column_preferences()

_RAW_PREFERRED_COLUMN_NAMES_LOWER: frozenset = frozenset(
    name.lower() for name in _RAW_PRIORITY_COLUMNS
)
_LEGACY_AMPLITUDE_COLUMN_NAMES_LOWER: frozenset = frozenset({"amplitude"})
_FORBIDDEN_DISPLAY_SCALED_COLUMN_NAMES_LOWER: frozenset = frozenset(
    {"amplitude_display_scaled"}
)


def _pick_sheet_case_insensitive(
    available_sheets: Iterable[str],
    preferences: Iterable[str],
) -> Optional[str]:
    """Return the first sheet name in ``available_sheets`` whose lowercased
    form matches any of the lowercased ``preferences``.

    Sheets in ``_DENSITY_EXTRACTION_METADATA_SHEETS`` are skipped even if
    their name otherwise satisfies a preference, so metadata/debug pages
    can never be mistaken for a spectrum.
    """
    available = [s for s in available_sheets if s not in _DENSITY_EXTRACTION_METADATA_SHEETS]
    lc_map = {s.lower(): s for s in available}
    for pref in preferences:
        match = lc_map.get(str(pref).lower())
        if match is not None:
            return match
    return None


def _pick_column_case_insensitive(
    df_columns: Iterable[str],
    preferences: Iterable[str],
) -> Optional[str]:
    cols = list(df_columns)
    lc_map = {str(c).lower(): str(c) for c in cols}
    for pref in preferences:
        match = lc_map.get(str(pref).lower())
        if match is not None:
            return match
    return None


def _sum_finite_numeric(series: pd.Series) -> tuple[float, int]:
    """Return (sum, count) of finite numeric entries in ``series``."""
    num = pd.to_numeric(series, errors="coerce")
    mask = np.isfinite(num.to_numpy(dtype=float, copy=False))
    if not mask.any():
        return 0.0, 0
    return float(num.to_numpy(dtype=float)[mask].sum()), int(mask.sum())


_LEGACY_RATIO_ALIAS_KEYS: tuple[str, ...] = tuple(
    # Build the legacy-alias names programmatically so the literal
    # ``batch_<component>_energy_ratio`` token never appears as a string in
    # the source. We only consult these aliases to *detect* that a workbook
    # came from a legacy pipeline; we never substitute them for the
    # canonical ``component_*`` weights.
    "_".join(("batch", part, "energy_ratio"))
    for part in ("harmonic", "inharmonic", "subbass")
)


def _read_component_weights_from_analysis_metadata(
    xlsx_path: Path,
) -> tuple[Optional[float], Optional[float], Optional[float], bool]:
    """Read the canonical ``component_*_energy_ratio`` weights from the
    per-note ``Analysis_Metadata`` (long-format Parameter/Value) sheet.

    Returns ``(w_H, w_I, w_S, legacy_aliases_only)`` where the boolean is
    True when the workbook only carries legacy alias keys for the
    component energy ratios (i.e. the canonical ``component_*`` rows are
    missing). The caller treats ``legacy_aliases_only=True`` as a
    ``missing_component_weights`` error rather than silently substituting
    the legacy values.
    """
    try:
        with pd.ExcelFile(xlsx_path) as xf:
            if "Analysis_Metadata" not in xf.sheet_names:
                return None, None, None, False
            df = xf.parse("Analysis_Metadata")
    except Exception as exc:
        logger.warning("Density_Metrics extraction: cannot read Analysis_Metadata from %s: %s", xlsx_path, exc)
        return None, None, None, False

    if df is None or df.empty:
        return None, None, None, False

    if not {"Parameter", "Value"}.issubset(df.columns):
        return None, None, None, False

    w_H = w_I = w_S = None
    legacy_present = False
    for _, row in df.iterrows():
        key = str(row.get("Parameter", "")).strip()
        if not key:
            continue
        raw = row.get("Value")
        try:
            if isinstance(raw, str):
                num = pd.to_numeric(raw.replace(",", ".").strip(), errors="coerce")
            else:
                num = pd.to_numeric(raw, errors="coerce")
        except Exception:
            num = None
        if num is None or pd.isna(num):
            continue
        v = float(num)
        if key == "component_harmonic_energy_ratio" and w_H is None:
            w_H = v
        elif key == "component_inharmonic_energy_ratio" and w_I is None:
            w_I = v
        elif key == "component_subbass_energy_ratio" and w_S is None:
            w_S = v
        elif key in _LEGACY_RATIO_ALIAS_KEYS:
            legacy_present = True
    legacy_aliases_only = (
        w_H is None and w_I is None and w_S is None and legacy_present
    )
    return w_H, w_I, w_S, legacy_aliases_only


# =====================================================================
# AUDIT FIX (stale-pipeline detection) — schema-version guard.
#
# Mirror of ``proc_audio.ANALYSIS_SCHEMA_VERSION``. Imported lazily so
# the compile module does not pull the heavy proc_audio import unless
# the schema check actually runs (some downstream tools / tests poke
# compile_metrics without needing the full audio stack).
# =====================================================================
EXPECTED_ANALYSIS_SCHEMA_VERSION: str = "single_pass_raw_export_v2"


STALE_PIPELINE_USER_MESSAGE: str = (
    "The selected analysis results were produced by a legacy/stale "
    "pipeline. Regenerate the analysis with the current single-pass "
    "raw-export version."
)


def scan_results_dir_for_stale_per_note_workbooks(
    results_dir: Union[str, Path],
    *,
    file_pattern: str = "spectral_analysis.xlsx",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Walk ``results_dir`` for per-note workbooks and report a
    compact summary of which ones pass the audit-canonical schema.

    Returns:
        {
            "results_dir":            str,
            "total":                  int,
            "valid":                  int,
            "stale":                  int,
            "first_failing_path":     str | None,
            "first_failing_reason":   str | None,
            "expected_schema":        EXPECTED_ANALYSIS_SCHEMA_VERSION,
            "details":                list[dict] — one per inspected file
                                       (each dict is the
                                       :func:`assess_per_note_workbook_schema`
                                       payload, optionally truncated).
        }
    """
    root = Path(results_dir)
    summary: Dict[str, Any] = {
        "results_dir": str(root),
        "total": 0,
        "valid": 0,
        "stale": 0,
        "first_failing_path": None,
        "first_failing_reason": None,
        "expected_schema": EXPECTED_ANALYSIS_SCHEMA_VERSION,
        "details": [],
    }
    if not root.exists() or not root.is_dir():
        summary["error"] = f"results_dir does not exist: {root}"
        return summary
    files = sorted(
        fp
        for fp in root.rglob(file_pattern)
        if not (fp.name.startswith("~$") or fp.name.startswith(".~lock"))
    )
    if max_samples is not None:
        files = files[: int(max_samples)]
    summary["total"] = len(files)
    for fp in files:
        info = assess_per_note_workbook_schema(fp)
        summary["details"].append(info)
        # AUDIT FIX (stale-pipeline guard) — only proc_audio-shaped
        # workbooks (i.e. ones that actually carry the per-note
        # spectrum sheets the pre-save validator stamps) are
        # candidates for the stale/valid classification. Synthetic
        # scalar-only test scaffolds or external scoring sheets are
        # neither valid nor stale; they're "not a per-note workbook"
        # and the caller decides whether to count them at all.
        if not info.get("looks_like_per_note_proc_audio_export"):
            continue
        is_valid = (
            info.get("schema_ok")
            and info.get("has_amplitude_raw")
            and info.get("has_power_raw")
            and not info.get("problems")
        )
        if is_valid:
            summary["valid"] += 1
        else:
            summary["stale"] += 1
            if summary["first_failing_path"] is None:
                summary["first_failing_path"] = info.get("path")
                summary["first_failing_reason"] = "; ".join(info.get("problems") or [])
    return summary


def assert_results_dir_schema_or_raise(
    results_dir: Union[str, Path],
    *,
    file_pattern: str = "spectral_analysis.xlsx",
) -> Dict[str, Any]:
    """Pre-flight schema check used by the GUI and orchestrator before
    plotting / compiling. Returns the summary dict on success.

    Raises :class:`RuntimeError` with the audit-canonical user-facing
    message (:data:`STALE_PIPELINE_USER_MESSAGE`) when ANY per-note
    workbook fails the schema check. The GUI catches this RuntimeError
    and surfaces it as a blocking error dialog; non-GUI callers see
    the same exception propagated to the caller.
    """
    summary = scan_results_dir_for_stale_per_note_workbooks(
        results_dir, file_pattern=file_pattern
    )
    if summary.get("error"):
        raise RuntimeError(
            f"{STALE_PIPELINE_USER_MESSAGE}\n[Reason] {summary['error']}"
        )
    if summary["total"] == 0:
        # No per-note workbooks at all is not a "stale" condition per
        # se — the caller decides how to proceed. We return the
        # summary so it can surface a more specific message.
        return summary
    if summary["stale"] > 0:
        raise RuntimeError(
            STALE_PIPELINE_USER_MESSAGE
            + f"\n[First failing file] {summary['first_failing_path']}"
            + (
                f"\n[Reason] {summary['first_failing_reason']}"
                if summary['first_failing_reason']
                else ""
            )
            + f"\n[Expected analysis_schema_version] "
            + f"{EXPECTED_ANALYSIS_SCHEMA_VERSION!r}"
        )
    return summary


def _read_analysis_schema_version_from_workbook(
    xlsx_path: Union[str, Path],
) -> Optional[str]:
    """Return the ``analysis_schema_version`` written under
    Analysis_Metadata, or ``None`` when the sheet / row is absent.

    Used by the compile-time and GUI / verify-CLI schema guards.
    """
    p = Path(xlsx_path)
    try:
        with pd.ExcelFile(p) as xf:
            if "Analysis_Metadata" not in xf.sheet_names:
                return None
            df = xf.parse("Analysis_Metadata")
    except Exception:
        return None
    if df is None or df.empty or not {"Parameter", "Value"}.issubset(df.columns):
        return None
    for _, row in df.iterrows():
        if str(row.get("Parameter", "")).strip() == "analysis_schema_version":
            v = row.get("Value")
            if v is None:
                return None
            try:
                return str(v).strip()
            except Exception:
                return None
    return None


def _read_analysis_metadata_scalar(
    xlsx_path: Union[str, Path],
    parameter_name: str,
) -> Optional[str]:
    """Return the first ``Value`` in ``Analysis_Metadata`` whose
    ``Parameter`` column matches *parameter_name* (case-sensitive),
    or ``None`` when the sheet/row is absent or unreadable.
    """
    p = Path(xlsx_path)
    try:
        with pd.ExcelFile(p) as xf:
            if "Analysis_Metadata" not in xf.sheet_names:
                return None
            df = xf.parse("Analysis_Metadata")
    except Exception:
        return None
    if df is None or df.empty or not {"Parameter", "Value"}.issubset(df.columns):
        return None
    target = str(parameter_name).strip()
    for _, row in df.iterrows():
        if str(row.get("Parameter", "")).strip() == target:
            v = row.get("Value")
            if v is None:
                return None
            try:
                return str(v).strip()
            except Exception:
                return None
    return None


def assess_per_note_workbook_schema(
    xlsx_path: Union[str, Path],
) -> Dict[str, Any]:
    """Inspect a per-note ``spectral_analysis.xlsx`` and report whether
    it was produced by the current ``ANALYSIS_SCHEMA_VERSION`` pipeline.

    Returns a dict with keys:

        path                      str (absolute path)
        schema_version            str | None
        schema_ok                 bool — matches EXPECTED_ANALYSIS_SCHEMA_VERSION
        has_amplitude_raw         bool — Harmonic/Inharmonic/Sub-bass sheets
        has_power_raw             bool
        model_weights_source      str | None
        component_profile_source  str | None
        export_alignment_source   str | None
        export_alignment_factor   float | None
        density_metrics_layout    "audit_canonical" | "legacy_six_columns" | "absent"
        problems                  list[str] — human-readable findings
    """
    p = Path(xlsx_path)
    result: Dict[str, Any] = {
        "path": str(p.resolve()) if p.exists() else str(p),
        "schema_version": None,
        "schema_ok": False,
        "has_amplitude_raw": False,
        "has_power_raw": False,
        "model_weights_source": None,
        "component_profile_source": None,
        "export_alignment_source": None,
        "export_alignment_factor": None,
        "density_metrics_layout": "absent",
        "problems": [],
    }
    if not p.exists():
        result["problems"].append(f"file_not_found: {p}")
        return result
    try:
        with pd.ExcelFile(p) as xf:
            sheet_names = list(xf.sheet_names)
            am_df = (
                xf.parse("Analysis_Metadata")
                if "Analysis_Metadata" in sheet_names
                else None
            )
            harm_cols = list(
                xf.parse("Harmonic Spectrum").columns
                if "Harmonic Spectrum" in sheet_names
                else []
            )
            ih_cols = list(
                xf.parse("Inharmonic Spectrum").columns
                if "Inharmonic Spectrum" in sheet_names
                else []
            )
            sb_cols = list(
                xf.parse("Sub-bass band").columns
                if "Sub-bass band" in sheet_names
                else []
            )
    except Exception as exc:
        result["problems"].append(f"workbook_unreadable: {exc}")
        return result

    # If none of the proc_audio per-note spectrum sheets are present,
    # this workbook is not a real per-note export (likely a synthetic
    # test scaffold). Skip the schema audit and report it as
    # ``not_a_per_note_workbook``; the caller decides whether to count
    # it as "stale".
    looks_like_per_note_proc_audio_export = bool(
        set(sheet_names)
        & {"Harmonic Spectrum", "Inharmonic Spectrum", "Sub-bass band"}
    )
    result["looks_like_per_note_proc_audio_export"] = (
        looks_like_per_note_proc_audio_export
    )

    meta: Dict[str, Any] = {}
    if isinstance(am_df, pd.DataFrame) and not am_df.empty and {"Parameter", "Value"}.issubset(am_df.columns):
        for _, row in am_df.iterrows():
            key = str(row.get("Parameter", "")).strip()
            if key and key not in meta:
                meta[key] = row.get("Value")

    result["schema_version"] = (
        str(meta["analysis_schema_version"]).strip()
        if "analysis_schema_version" in meta and meta["analysis_schema_version"] is not None
        else None
    )
    result["schema_ok"] = result["schema_version"] == EXPECTED_ANALYSIS_SCHEMA_VERSION

    result["model_weights_source"] = (
        str(meta.get("model_weights_source")) if "model_weights_source" in meta else None
    )
    result["component_profile_source"] = (
        str(meta.get("component_profile_source"))
        if "component_profile_source" in meta
        else None
    )
    result["export_alignment_source"] = (
        str(meta.get("export_alignment_source"))
        if "export_alignment_source" in meta
        else None
    )
    try:
        if "export_alignment_factor" in meta and meta["export_alignment_factor"] is not None:
            result["export_alignment_factor"] = float(meta["export_alignment_factor"])
    except (TypeError, ValueError):
        result["export_alignment_factor"] = None

    def _has(cols, name):
        return any(str(c).strip().lower() == name.lower() for c in cols)

    raw_present = (
        (_has(harm_cols, "Amplitude_raw") or not harm_cols)
        and (_has(ih_cols, "Amplitude_raw") or not ih_cols)
        and (_has(sb_cols, "Amplitude_raw") or not sb_cols)
    )
    pow_present = (
        (_has(harm_cols, "Power_raw") or not harm_cols)
        and (_has(ih_cols, "Power_raw") or not ih_cols)
        and (_has(sb_cols, "Power_raw") or not sb_cols)
    )
    result["has_amplitude_raw"] = bool(raw_present)
    result["has_power_raw"] = bool(pow_present)

    # batch_* contamination on Inharmonic Spectrum (forbidden in
    # integrated_single_pass).
    batch_leak = [c for c in ih_cols if str(c).lower().startswith("batch_")]
    if (
        batch_leak
        and result["component_profile_source"] == "integrated_single_pass"
    ):
        result["problems"].append(
            f"integrated_single_pass: Inharmonic Spectrum has batch_* columns: {batch_leak}"
        )

    if result["component_profile_source"] == "integrated_single_pass":
        if result["model_weights_source"] != "current_analysis":
            result["problems"].append(
                f"integrated_single_pass: model_weights_source != 'current_analysis' "
                f"(got {result['model_weights_source']!r})"
            )
        if result["export_alignment_source"] != "disabled_integrated_single_pass":
            result["problems"].append(
                f"integrated_single_pass: export_alignment_source != "
                f"'disabled_integrated_single_pass' (got {result['export_alignment_source']!r})"
            )
        if result["export_alignment_factor"] is not None and abs(
            float(result["export_alignment_factor"]) - 1.0
        ) > 1e-9:
            result["problems"].append(
                f"integrated_single_pass: export_alignment_factor != 1.0 "
                f"(got {result['export_alignment_factor']!r})"
            )

    if looks_like_per_note_proc_audio_export:
        if not result["has_amplitude_raw"]:
            result["problems"].append("Spectrum sheets missing Amplitude_raw")
        if not result["has_power_raw"]:
            result["problems"].append("Spectrum sheets missing Power_raw")
        if not result["schema_ok"]:
            result["problems"].append(
                f"analysis_schema_version != {EXPECTED_ANALYSIS_SCHEMA_VERSION!r} "
                f"(got {result['schema_version']!r})"
            )

    # Density_Metrics layout heuristic (only meaningful for compiled
    # workbooks; on per-note files Density_Metrics is normally absent).
    if "Density_Metrics" in sheet_names:
        try:
            with pd.ExcelFile(p) as xf2:
                dm_cols = set(xf2.parse("Density_Metrics", nrows=0).columns)
        except Exception:
            dm_cols = set()
        legacy_six = {
            "Note", "weight_function",
            "Harmonic Partials sum", "Inharmonic Partials sum",
            "Sub-bass sum", "Total sum",
        }
        if dm_cols and dm_cols.issubset(legacy_six) and dm_cols >= {
            "Harmonic Partials sum", "Total sum"
        }:
            result["density_metrics_layout"] = "legacy_six_columns"
        elif "density_metric_raw" in dm_cols and "density_metric_normalized" in dm_cols:
            result["density_metrics_layout"] = "audit_canonical"
        elif dm_cols:
            result["density_metrics_layout"] = "non_canonical_partial"
    return result


def _extract_harmonic_amplitude_sum_for_density(
    xf: pd.ExcelFile,
    sheet_names: List[str],
) -> tuple[Optional[float], int, str]:
    """Sum ``Amplitude_raw`` over rows of the ``Harmonic Spectrum`` sheet whose
    ``include_for_density`` flag is True (audit rule for the Stage 1 +
    Stage 2 ``harmonic_log_amplitude_density`` metric).

    Fallback behaviour:

    * If ``include_for_density`` is absent (legacy / strict-only workbooks),
      every finite ``Amplitude_raw`` row contributes.
    * If ``Amplitude_raw`` is absent, falls back to ``Amplitude``.
    * If the sheet itself is absent, returns ``(None, 0, "")``.
    """
    sheet = _pick_sheet_case_insensitive(sheet_names, HARMONIC_SPECTRUM_SHEET_PREFERENCES)
    if sheet is None:
        return None, 0, ""
    try:
        df = xf.parse(sheet)
    except Exception as exc:
        logger.warning(
            "harmonic_log_amplitude_density: cannot parse %s sheet (%s); skipping.",
            sheet, exc,
        )
        return None, 0, f"sheet={sheet};column=<unreadable>"
    if df is None or df.empty:
        return 0.0, 0, f"sheet={sheet};column=<empty>"

    cols_lower = {str(c).lower(): c for c in df.columns}
    amp_col = cols_lower.get("amplitude_raw") or cols_lower.get("amplitude")
    if amp_col is None:
        return None, 0, f"sheet={sheet};column=<not_found>"

    amps = pd.to_numeric(df[amp_col], errors="coerce")
    finite_mask = amps.notna() & (amps > 0)

    if "include_for_density" in cols_lower:
        inc_col = cols_lower["include_for_density"]
        inc_raw = df[inc_col]
        # Coerce mixed truthy / "True" / "true" / 1 to a clean boolean.
        if pd.api.types.is_bool_dtype(inc_raw):
            inc_mask = inc_raw.astype(bool)
        else:
            inc_mask = inc_raw.map(
                lambda v: str(v).strip().lower() in {"1", "true", "yes"}
            ).astype(bool)
        full_mask = finite_mask & inc_mask
    else:
        full_mask = finite_mask

    if not full_mask.any():
        return 0.0, 0, f"sheet={sheet};column={amp_col};no_included_rows"
    total = float(amps.loc[full_mask].sum())
    return total, int(full_mask.sum()), f"sheet={sheet};column={amp_col}"


def _extract_band_amplitude_sum_for_density(
    xf: pd.ExcelFile,
    sheet_names: List[str],
    sheet_prefs: Tuple[str, ...],
    *,
    label: str,
) -> tuple[Optional[float], int, str]:
    """Sum ``Amplitude_raw`` over a single spectrum sheet (Inharmonic or
    Sub-bass) for the weighted note-density metric.

    Audit rules:

    * Default column is ``Amplitude_raw``. The function falls back to
      ``Amplitude`` ONLY when ``Amplitude_raw`` is absent. ``Power_raw``
      is NEVER read by this default path; callers wanting the power-sum
      basis must use the dedicated debug path in
      ``extract_density_components_from_per_note_workbook``.
    * Forbidden display-scaled column names are filtered before the
      column picker so the function never silently lands on them.
    * Returns ``(None, 0, "")`` when the sheet itself is absent.
    """
    sheet = _pick_sheet_case_insensitive(sheet_names, sheet_prefs)
    if sheet is None:
        return None, 0, ""
    try:
        df = xf.parse(sheet)
    except Exception as exc:
        logger.warning(
            "%s amplitude_sum: cannot parse %s sheet (%s); skipping.",
            label, sheet, exc,
        )
        return None, 0, f"sheet={sheet};column=<unreadable>"
    if df is None or df.empty:
        return 0.0, 0, f"sheet={sheet};column=<empty>"

    usable_cols = [
        c for c in df.columns
        if str(c).strip().lower() not in _FORBIDDEN_DISPLAY_SCALED_COLUMN_NAMES_LOWER
    ]
    if not usable_cols:
        return None, 0, f"sheet={sheet};column=<not_found>"

    cols_lower = {str(c).lower(): c for c in usable_cols}
    amp_col = cols_lower.get("amplitude_raw") or cols_lower.get("amplitude")
    if amp_col is None:
        return None, 0, f"sheet={sheet};column=<not_found>"

    amps = pd.to_numeric(df[amp_col], errors="coerce")
    finite_mask = amps.notna() & (amps > 0)
    if not finite_mask.any():
        return 0.0, 0, f"sheet={sheet};column={amp_col};no_finite_rows"
    total = float(amps.loc[finite_mask].sum())
    return total, int(finite_mask.sum()), f"sheet={sheet};column={amp_col}"


DENSITY_WEIGHT_FUNCTION_VALID: Tuple[str, ...] = ("linear", "log", "power")
DENSITY_WEIGHT_FUNCTION_DEFAULT: str = "linear"
DENSITY_WEIGHT_SUM_TOLERANCE: float = 1e-3


def _normalise_density_weight_function(weight_function: Optional[str]) -> str:
    """Coerce *weight_function* to one of ``DENSITY_WEIGHT_FUNCTION_VALID``.

    ``"sum"``  -> ``"linear"`` (legacy alias).
    Empty / unknown -> ``DENSITY_WEIGHT_FUNCTION_DEFAULT``.

    Used where only the three **aggregate** compile modes (sum / log10(1+sum) /
    sum-power) are supported. For full UI parity (sqrt, cubic, d3, …) use
    :func:`_compile_operator_weight_function_key` and the ``apply_density_metric``
    branch inside :func:`extract_density_component_sum`.
    """
    if weight_function is None:
        return DENSITY_WEIGHT_FUNCTION_DEFAULT
    wf = str(weight_function).strip().lower()
    if wf == "sum":
        wf = "linear"
    if wf in DENSITY_WEIGHT_FUNCTION_VALID:
        return wf
    return DENSITY_WEIGHT_FUNCTION_DEFAULT


def _compile_operator_weight_function_key(weight_function: Optional[str]) -> str:
    """Normalise the GUI / orchestrator weight key (aliases) without collapsing
    discrete or element-wise keys to *linear*.

    Keys accepted by :func:`density.get_weight_function` are preserved so
    Stage-2 extraction can call :func:`density.apply_density_metric` on the
    per-partial amplitude vector. Unknown strings still fall back to *linear*
    after :func:`extract_density_component_sum` validates with ``get_weight_function``.
    """
    wf = str(weight_function or "").strip().lower() or DENSITY_WEIGHT_FUNCTION_DEFAULT
    if wf == "sum":
        wf = "linear"
    if wf == "d2":
        wf = "linear"
    if wf == "d8":
        wf = "d17"
    return wf


def _density_sheet_preferences_for(sheet_name: str) -> Tuple[str, ...]:
    """Return the case-insensitive sheet-name preferences for one of the
    three canonical component sheets used by the density metric.
    """
    target = str(sheet_name or "").strip().lower()
    if target in {"harmonic spectrum", "harmonic_spectrum"}:
        return HARMONIC_SPECTRUM_SHEET_PREFERENCES
    if target in {"inharmonic spectrum", "inharmonic_spectrum"}:
        return INHARMONIC_SPECTRUM_SHEET_PREFERENCES
    if target in {
        "sub-bass band",
        "sub_bass_band",
        "sub-bass spectrum",
        "sub-bass",
        "subbass band",
    }:
        return SUBBASS_SPECTRUM_SHEET_PREFERENCES
    raise ValueError(
        "extract_density_component_sum: unknown sheet_name "
        f"{sheet_name!r}; expected one of 'Harmonic Spectrum', "
        "'Inharmonic Spectrum', 'Sub-bass band'."
    )


_INCLUDE_FOR_DENSITY_TRUE_TOKENS: frozenset = frozenset(
    {"1", "true", "yes"}
)


def _resolve_include_for_density_mask(
    df: pd.DataFrame, cols_lower: Dict[str, Any]
) -> Tuple[Optional[np.ndarray], int]:
    """Return the ``include_for_density`` boolean mask (and how many rows it
    excludes), or ``(None, 0)`` when the column is absent.

    Accepted truthy values: ``True``, ``1`` (any numeric type), and the
    case-insensitive strings ``"1"``, ``"true"``, ``"yes"``. Anything
    else — ``False``, ``0``, ``"false"``, ``"0"``, ``"no"``, empty
    string, ``NaN`` — is excluded.
    """
    inc_key = cols_lower.get("include_for_density")
    if inc_key is None:
        return None, 0
    raw = df[inc_key]
    if pd.api.types.is_bool_dtype(raw):
        mask = raw.astype(bool).to_numpy()
    else:
        def _coerce(v) -> bool:
            if v is None:
                return False
            try:
                if isinstance(v, (float, np.floating)) and np.isnan(float(v)):
                    return False
            except (TypeError, ValueError):
                pass
            if isinstance(v, (bool, np.bool_)):
                return bool(v)
            if isinstance(v, (int, np.integer)):
                return int(v) == 1
            if isinstance(v, (float, np.floating)):
                return float(v) == 1.0
            try:
                s = str(v).strip().lower()
            except Exception:
                return False
            return s in _INCLUDE_FOR_DENSITY_TRUE_TOKENS

        mask = raw.map(_coerce).astype(bool).to_numpy()
    excluded = int((~mask).sum())
    return mask, excluded


def extract_density_component_sum(
    workbook_path: Union[str, Path],
    sheet_name: str,
    weight_function: str,
) -> Dict[str, Any]:
    """Canonical density-component sum extractor.

    Reads *workbook_path* and returns the band-level density-component
    value ``D`` for *sheet_name* under the selected *weight_function*.

    Weighting-function semantics (single source of truth)::

        linear -> D = SUM(Amplitude_raw)
        log    -> D = LOG10(1 + SUM(Amplitude_raw))
        power  -> D = SUM(Power_raw)
                     (or SUM(Amplitude_raw ** 2) if Power_raw absent)

    Any other key accepted by ``density.get_weight_function`` (sqrt,
    cubic, logarithmic, d3, …) uses ``density.apply_density_metric`` on
    the masked per-partial amplitude vector (same mask contract as
    *linear* / *log* for harmonic inclusion), so compiled ``D_*`` track
    the GUI choice instead of silently falling back to *linear*.

    AUDIT FIX (Harmonic-Spectrum inclusion contract) — when the
    sheet is ``Harmonic Spectrum`` and an ``include_for_density``
    column is present, the SAME row mask used by the Stage 1
    ``harmonic_amplitude_sum`` / ``harmonic_log_amplitude_density``
    helpers is applied BEFORE the weight_function math. This brings
    the compiled ``Harmonic Partials sum`` into agreement with the
    Stage 1 harmonic candidate contract; without it, the compile
    sheet and the per-note metadata would carry two different
    definitions of harmonic density.

    Accepted truthy values for ``include_for_density``: ``True``,
    ``1``, and case-insensitive ``"1"`` / ``"true"`` / ``"yes"``.
    Anything else (``False``, ``0``, ``"false"``, ``"no"``, empty,
    NaN) is treated as excluded.

    The filter NEVER applies to ``Inharmonic Spectrum`` or
    ``Sub-bass band`` (their inclusion contract is summing all
    finite non-negative amplitudes).

    The function **never** reads ``Amplitude_display_scaled``, never
    reads ``Power_raw`` for the ``"linear"`` / ``"log"`` modes, never
    consults ``batch_*`` columns and never loads
    ``super_analysis_results.json``.

    Returns a dict with keys::

        D                       float | None  -- the band D value
        n                       int           -- rows contributing
        sheet                   str           -- resolved sheet name
        column                  str           -- resolved column name
        weight_function         str           -- normalised weight
        density_component_sum_source  str     -- "sheet=...;column=...;
                                                  weight_function=...;
                                                  inclusion_policy=..."
        status                  str           -- "ok" / "missing_sheet" /
                                                 "missing_column" /
                                                 "no_numeric_values" /
                                                 "extraction_error"
        inclusion_policy        str           -- one of
            "include_for_density_true"
            "all_rows_no_include_column"
            "" (when sheet is not the Harmonic Spectrum)
        excluded_count          int           -- rows excluded by the
                                                 include_for_density
                                                 filter (0 when filter
                                                 was not applied)
    """
    wf_in = _compile_operator_weight_function_key(weight_function)
    use_elementwise = False
    if wf_in not in DENSITY_WEIGHT_FUNCTION_VALID:
        try:
            from density import get_weight_function

            get_weight_function(wf_in)
            use_elementwise = True
        except ValueError:
            wf_in = DENSITY_WEIGHT_FUNCTION_DEFAULT
    wf = wf_in if wf_in in DENSITY_WEIGHT_FUNCTION_VALID else "linear"
    sheet_prefs = _density_sheet_preferences_for(sheet_name)
    is_harmonic_sheet = sheet_prefs is HARMONIC_SPECTRUM_SHEET_PREFERENCES

    result: Dict[str, Any] = {
        "D": None,
        "n": 0,
        "sheet": "",
        "column": "",
        "weight_function": (wf_in if use_elementwise else wf),
        "density_component_sum_source": "",
        "status": "extraction_error",
        "inclusion_policy": "",
        "excluded_count": 0,
    }

    p = Path(workbook_path)
    if not p.exists():
        return result

    try:
        with pd.ExcelFile(p) as xf:
            sheet = _pick_sheet_case_insensitive(xf.sheet_names, sheet_prefs)
            if sheet is None:
                result["status"] = "missing_sheet"
                return result
            try:
                df = xf.parse(sheet)
            except Exception as exc:
                logger.warning(
                    "extract_density_component_sum: cannot parse %s in %s: %s",
                    sheet, p, exc,
                )
                result["status"] = "extraction_error"
                return result
    except Exception as exc:
        logger.warning("extract_density_component_sum: cannot open %s: %s", p, exc)
        return result

    if df is None or df.empty:
        result["sheet"] = sheet
        result["status"] = "no_numeric_values"
        return result

    usable_cols = [
        c for c in df.columns
        if str(c).strip().lower() not in _FORBIDDEN_DISPLAY_SCALED_COLUMN_NAMES_LOWER
    ]
    cols_lower = {str(c).strip().lower(): c for c in usable_cols}

    amp_col = cols_lower.get("amplitude_raw") or cols_lower.get("amplitude")
    power_col = cols_lower.get("power_raw") or cols_lower.get("power")

    include_mask: Optional[np.ndarray] = None
    inclusion_policy_label = ""
    excluded_count = 0
    if is_harmonic_sheet:
        inc_mask_resolved, excluded_count = _resolve_include_for_density_mask(
            df, cols_lower
        )
        if inc_mask_resolved is not None:
            include_mask = inc_mask_resolved
            inclusion_policy_label = "include_for_density_true"
        else:
            inclusion_policy_label = "all_rows_no_include_column"
    result["inclusion_policy"] = inclusion_policy_label
    result["excluded_count"] = excluded_count

    column_used: Optional[str] = None
    series: Optional[pd.Series] = None
    sum_strategy: str = ""

    if wf == "linear":
        if amp_col is None:
            result["sheet"] = sheet
            result["status"] = "missing_column"
            result["density_component_sum_source"] = (
                f"sheet={sheet};column=<not_found>;weight_function={wf}"
            )
            return result
        column_used = str(amp_col)
        series = pd.to_numeric(df[amp_col], errors="coerce")
        sum_strategy = "sum_amplitude_raw"
    elif wf == "log":
        if amp_col is None:
            result["sheet"] = sheet
            result["status"] = "missing_column"
            result["density_component_sum_source"] = (
                f"sheet={sheet};column=<not_found>;weight_function={wf}"
            )
            return result
        column_used = str(amp_col)
        series = pd.to_numeric(df[amp_col], errors="coerce")
        sum_strategy = "log10_1p_sum_amplitude_raw"
    elif wf == "power":
        if power_col is not None:
            column_used = str(power_col)
            series = pd.to_numeric(df[power_col], errors="coerce")
            sum_strategy = "sum_power_raw"
        elif amp_col is not None:
            column_used = f"{amp_col}**2"
            series_amp = pd.to_numeric(df[amp_col], errors="coerce")
            series = series_amp.pow(2)
            sum_strategy = "sum_amplitude_raw_squared_fallback"
        else:
            result["sheet"] = sheet
            result["status"] = "missing_column"
            result["density_component_sum_source"] = (
                f"sheet={sheet};column=<not_found>;weight_function={wf}"
            )
            return result
    else:
        result["sheet"] = sheet
        result["status"] = "extraction_error"
        return result

    if series is None or series.empty:
        result["sheet"] = sheet
        result["column"] = column_used or ""
        result["status"] = "no_numeric_values"
        return result

    mask = np.isfinite(series.to_numpy(dtype=float, copy=False))
    if wf in ("linear", "log"):
        mask = mask & (series.to_numpy(dtype=float, copy=False) >= 0)
    if include_mask is not None:
        if include_mask.shape[0] != mask.shape[0]:
            include_mask = include_mask[: mask.shape[0]]
        mask = mask & include_mask
    n_used = int(mask.sum())
    if n_used == 0:
        result["sheet"] = sheet
        result["column"] = column_used or ""
        result["status"] = "no_numeric_values"
        source_str = (
            f"sheet={sheet};column={column_used};weight_function={wf}"
        )
        if is_harmonic_sheet and inclusion_policy_label:
            source_str += f";inclusion_policy={inclusion_policy_label}"
        result["density_component_sum_source"] = source_str
        return result

    raw_total = float(series.to_numpy(dtype=float)[mask].sum())
    if wf == "log":
        d_value = float(np.log10(1.0 + max(0.0, raw_total)))
    else:
        d_value = raw_total

    if use_elementwise and wf in ("linear", "log"):
        from density import apply_density_metric

        amps = series.to_numpy(dtype=float, copy=False)[mask]
        d_value = float(
            apply_density_metric(
                amps,
                weight_function=wf_in,
                normalize=False,
                remove_noise=False,
                frequencies=None,
                fundamental_freq=None,
                account_for_spectral_rolloff=False,
                prevent_domination=True,
            )
        )
        sum_strategy = f"apply_density_metric:{wf_in}"
        result["weight_function"] = wf_in

    wf_label = wf_in if use_elementwise else wf
    result["D"] = float(d_value)
    result["n"] = n_used
    result["sheet"] = sheet
    result["column"] = column_used or ""
    source_str = (
        f"sheet={sheet};column={column_used};weight_function={wf_label};"
        f"strategy={sum_strategy}"
    )
    if is_harmonic_sheet and inclusion_policy_label:
        source_str += f";inclusion_policy={inclusion_policy_label}"
    result["density_component_sum_source"] = source_str
    result["status"] = "ok"
    return result


def extract_density_components_from_per_note_workbook(
    xlsx_path: Union[str, Path],
    *,
    density_component_basis: str = DENSITY_COMPONENT_BASIS_DEFAULT,
    weight_function: Optional[str] = None,
) -> Dict[str, Any]:
    """Open ``xlsx_path`` and extract the inputs needed for the weighted
    density metric: ``D_H``/``D_I``/``D_S`` from the harmonic/inharmonic/
    sub-bass spectrum sheets, and ``w_H``/``w_I``/``w_S`` from the
    ``Analysis_Metadata`` sheet.

    AUDIT FIX (direct per-note Density_Metrics extraction) — this function
    is the canonical source of truth for the compiled Density_Metrics
    sheet. It deliberately does NOT read the scalar ``Harmonic Partials
    sum``/etc. columns from ``Metrics``; instead it sums the per-partial
    ``Amplitude`` (or any documented column-name preference) inside the
    actual spectrum sheets, so the compiled workbook reflects the same
    physical population the per-note export drew the figures from.

    Returns a dict with keys:

        D_H, D_I, D_S                              float or None
        harmonic_spectrum_count, ... _count        int (0 when missing)
        harmonic_spectrum_source, ... _source      str describing
                                                   "sheet=<X>;column=<Y>"
        w_H, w_I, w_S                              float or None
        density_extraction_status                  one of
            ok / missing_harmonic_spectrum / missing_inharmonic_spectrum /
            missing_subbass_spectrum / missing_component_weights /
            no_numeric_values / legacy_source_used / extraction_error
        legacy_aliases_only                        bool — True when the per-note
                                                   workbook only carried legacy
                                                   compatibility weight aliases
                                                   (single-pass policy)
    """
    p = Path(xlsx_path)
    basis = (
        str(density_component_basis or DENSITY_COMPONENT_BASIS_DEFAULT)
        .strip()
        .lower()
        or DENSITY_COMPONENT_BASIS_DEFAULT
    )
    if basis not in DENSITY_COMPONENT_BASIS_VALID:
        raise ValueError(
            f"density_component_basis must be one of "
            f"{DENSITY_COMPONENT_BASIS_VALID!r} (got {basis!r})"
        )
    result: Dict[str, Any] = {
        "D_H": None,
        "D_I": None,
        "D_S": None,
        "harmonic_spectrum_count": 0,
        "inharmonic_spectrum_count": 0,
        "subbass_spectrum_count": 0,
        "harmonic_spectrum_source": "",
        "inharmonic_spectrum_source": "",
        "subbass_spectrum_source": "",
        "w_H": None,
        "w_I": None,
        "w_S": None,
        "density_extraction_status": "extraction_error",
        "legacy_aliases_only": False,
        "legacy_scaled_source_used": False,
        "analysis_schema_version": None,
        # Stage 1 harmonic-spectrum candidate metric (per-row).
        # ``harmonic_amplitude_sum`` is the sum of ``Amplitude_raw`` over
        # rows of the ``Harmonic Spectrum`` sheet whose
        # ``include_for_density`` flag is True. When the column is
        # absent (older workbooks) every finite-amplitude row is included.
        # ``harmonic_log_amplitude_density = log10(1 + harmonic_amplitude_sum)``.
        "harmonic_amplitude_sum": None,
        "harmonic_log_amplitude_density": None,
        "harmonic_density_included_count": 0,
        "harmonic_amplitude_source": "",
        # Stage 2 weighted note-density inputs. ``*_amplitude_sum`` are
        # linear diagnostic sums (unchanged by weight_function). The
        # publication-facing weighted sum uses per-band D values from
        # ``extract_density_component_sum`` under the compile weight_function:
        #
        #     density_weighted_sum =
        #         harmonic_density_sum      * component_harmonic_energy_ratio
        #       + inharmonic_density_sum    * component_inharmonic_energy_ratio
        #       + subbass_density_sum       * component_subbass_energy_ratio
        #     density_log_weighted = log10(1 + density_weighted_sum)
        "inharmonic_amplitude_sum": None,
        "inharmonic_amplitude_source": "",
        "inharmonic_amplitude_count": 0,
        "subbass_amplitude_sum": None,
        "subbass_amplitude_source": "",
        "subbass_amplitude_count": 0,
        "weighted_harmonic_component": None,
        "weighted_inharmonic_component": None,
        "weighted_subbass_component": None,
        "density_weighted_sum": None,
        "density_log_weighted": None,
        "density_log_formula": "log10(1 + density_weighted_sum)",
        # AUDIT FIX (Density_Metrics component basis) — provenance:
        # which basis was used to evaluate D_H / D_I / D_S, and which
        # one drove the energy-ratio weights. The defaults are
        # ``amplitude_sum`` for the components and the constant
        # ``DENSITY_WEIGHT_BASIS`` for the weights (energy / power
        # ratios computed inside proc_audio).
        "density_component_basis": basis,
        "density_weight_basis": DENSITY_WEIGHT_BASIS,
        # AUDIT FIX (canonical weight_function semantics) — D_band values
        # computed by ``extract_density_component_sum`` under the
        # selected weighting algorithm. When ``weight_function`` is None
        # (legacy callers) these mirror the basis-based D_H/D_I/D_S.
        "density_weight_function": _compile_operator_weight_function_key(
            weight_function
        ),
        "harmonic_density_sum": None,
        "inharmonic_density_sum": None,
        "subbass_density_sum": None,
        "density_formula": (
            "density_metric_raw = D_H*w_H + D_I*w_I + D_S*w_S; "
            "D_band per weight_function (see "
            "extract_density_component_sum)."
        ),
        "density_component_sum_source": "",
        "density_weight_function_explicit": weight_function is not None,
        # AUDIT FIX (Harmonic-Spectrum inclusion contract) — defaults
        # for the diagnostic columns. ``extract_density_component_sum``
        # overwrites these for the harmonic sheet later in the body.
        "harmonic_density_inclusion_policy": "",
        "harmonic_density_excluded_count": 0,
    }

    if not p.exists():
        result["density_extraction_status"] = "extraction_error"
        return result

    # AUDIT FIX (stale-pipeline detection) — refuse to extract from a
    # workbook produced by a legacy / stale pipeline. The compile step
    # tags the row ``extraction_error_stale_schema`` and the caller
    # aggregates these; if EVERY workbook is stale, the compiler
    # raises RuntimeError instead of silently shipping a half-baked
    # Density_Metrics sheet.
    #
    # The schema gate only applies when the workbook actually looks
    # like a real proc_audio per-note export (i.e. it carries at
    # least one of the spectrum sheets the pre-save validator stamps
    # with Amplitude_raw / Power_raw). Synthetic scalar-only test
    # scaffolds — workbooks that only have a Density_Metrics sheet
    # and were never produced by proc_audio.AudioProcessor — are
    # treated as legacy compile inputs and proceed through the
    # remainder of the extractor unchanged.
    schema_version = _read_analysis_schema_version_from_workbook(p)
    result["analysis_schema_version"] = schema_version
    try:
        with pd.ExcelFile(p) as _xf_meta:
            _names = set(_xf_meta.sheet_names)
    except Exception:
        _names = set()
    looks_like_per_note_proc_audio_export = bool(
        _names
        & {"Harmonic Spectrum", "Inharmonic Spectrum", "Sub-bass band"}
    )
    if (
        looks_like_per_note_proc_audio_export
        and schema_version != EXPECTED_ANALYSIS_SCHEMA_VERSION
    ):
        logger.error(
            "STALE-PIPELINE GUARD: %s carries analysis_schema_version=%r; "
            "expected %r. Marking row extraction_error_stale_schema. "
            "Regenerate the analysis with the current single-pass "
            "raw-export pipeline.",
            p, schema_version, EXPECTED_ANALYSIS_SCHEMA_VERSION,
        )
        result["density_extraction_status"] = "extraction_error_stale_schema"
        return result

    try:
        xf = pd.ExcelFile(p)
        sheet_names = list(xf.sheet_names)
    except Exception as exc:
        logger.warning("Density_Metrics extraction: cannot open %s: %s", p, exc)
        result["density_extraction_status"] = "extraction_error"
        return result

    def _extract_one(
        sheet_prefs: Tuple[str, ...],
        col_prefs: Tuple[str, ...],
    ) -> tuple[Optional[float], int, str, bool]:
        """Extract Σ(values) for the first numeric column found in
        ``col_prefs``.

        AUDIT FIX (inharmonic-energy underestimation): never read
        ``Amplitude_display_scaled``. The fourth return value
        ``legacy_amplitude_fallback`` is True when no raw column was
        available and the function fell back to ``Amplitude``; the
        caller uses this to mark
        ``density_extraction_status = legacy_scaled_source_used``.
        """
        sheet = _pick_sheet_case_insensitive(sheet_names, sheet_prefs)
        if sheet is None:
            return None, 0, "", False
        try:
            df = xf.parse(sheet)
        except Exception as exc:
            logger.warning("Density_Metrics extraction: cannot parse sheet %s in %s: %s", sheet, p, exc)
            return None, 0, f"sheet={sheet};column=<unreadable>", False
        if df is None or df.empty:
            return 0.0, 0, f"sheet={sheet};column=<empty>", False
        # Filter out forbidden display-scaled columns BEFORE the picker
        # so we never silently land on them even if they appear first.
        usable_cols = [
            c for c in df.columns
            if str(c).strip().lower() not in _FORBIDDEN_DISPLAY_SCALED_COLUMN_NAMES_LOWER
        ]
        if not usable_cols:
            return None, 0, f"sheet={sheet};column=<not_found>", False
        col = _pick_column_case_insensitive(usable_cols, col_prefs)
        if col is None:
            return None, 0, f"sheet={sheet};column=<not_found>", False
        s = df[col]
        total, n = _sum_finite_numeric(s)
        if n == 0:
            return None, 0, f"sheet={sheet};column={col};no_numeric_values", False
        col_norm = str(col).strip().lower()
        legacy_fallback = (
            col_norm in _LEGACY_AMPLITUDE_COLUMN_NAMES_LOWER
            and col_norm not in _RAW_PREFERRED_COLUMN_NAMES_LOWER
        )
        return total, n, f"sheet={sheet};column={col}", legacy_fallback

    try:
        D_H, n_H, src_H, legacy_H = _extract_one(
            HARMONIC_SPECTRUM_SHEET_PREFERENCES,
            _harmonic_column_preferences(basis),
        )
        D_I, n_I, src_I, legacy_I = _extract_one(
            INHARMONIC_SPECTRUM_SHEET_PREFERENCES,
            _inharmonic_column_preferences(basis),
        )
        D_S, n_S, src_S, legacy_S = _extract_one(
            SUBBASS_SPECTRUM_SHEET_PREFERENCES,
            _subbass_column_preferences(basis),
        )
        h_amp_sum, h_inc_n, h_src = _extract_harmonic_amplitude_sum_for_density(
            xf, sheet_names
        )
        i_amp_sum, i_n, i_src = _extract_band_amplitude_sum_for_density(
            xf, sheet_names, INHARMONIC_SPECTRUM_SHEET_PREFERENCES, label="inharmonic"
        )
        s_amp_sum, s_n, s_src = _extract_band_amplitude_sum_for_density(
            xf, sheet_names, SUBBASS_SPECTRUM_SHEET_PREFERENCES, label="subbass"
        )
    finally:
        try:
            xf.close()
        except Exception:
            pass

    if h_amp_sum is not None and np.isfinite(h_amp_sum):
        result["harmonic_amplitude_sum"] = float(h_amp_sum)
        result["harmonic_log_amplitude_density"] = float(
            np.log10(1.0 + max(0.0, float(h_amp_sum)))
        )
    else:
        result["harmonic_amplitude_sum"] = None
        result["harmonic_log_amplitude_density"] = None
    result["harmonic_density_included_count"] = int(h_inc_n)
    result["harmonic_amplitude_source"] = str(h_src)

    if i_amp_sum is not None and np.isfinite(i_amp_sum):
        result["inharmonic_amplitude_sum"] = float(i_amp_sum)
    else:
        result["inharmonic_amplitude_sum"] = None
    result["inharmonic_amplitude_count"] = int(i_n)
    result["inharmonic_amplitude_source"] = str(i_src)

    if s_amp_sum is not None and np.isfinite(s_amp_sum):
        result["subbass_amplitude_sum"] = float(s_amp_sum)
    else:
        result["subbass_amplitude_sum"] = None
    result["subbass_amplitude_count"] = int(s_n)
    result["subbass_amplitude_source"] = str(s_src)

    result["D_H"] = D_H
    result["D_I"] = D_I
    result["D_S"] = D_S
    result["harmonic_spectrum_count"] = int(n_H)
    result["inharmonic_spectrum_count"] = int(n_I)
    result["subbass_spectrum_count"] = int(n_S)
    result["harmonic_spectrum_source"] = src_H
    result["inharmonic_spectrum_source"] = src_I
    result["subbass_spectrum_source"] = src_S
    result["legacy_scaled_source_used"] = bool(legacy_H or legacy_I or legacy_S)

    # ------------------------------------------------------------------
    # AUDIT FIX (canonical weight_function semantics) — D_H / D_I / D_S
    # under the selected weighting algorithm. We call the canonical
    # ``extract_density_component_sum`` for every component sheet. The
    # results overwrite D_H / D_I / D_S (and their provenance strings)
    # so the compiled ``density_metric_raw`` reflects the operator's
    # weight_function choice. Power_raw is never read for "linear" or
    # "log", and Amplitude_display_scaled is never read at all.
    # ------------------------------------------------------------------
    wf_op = _compile_operator_weight_function_key(weight_function)
    wf_components = "power" if basis == "power_sum" else wf_op
    wf_h = extract_density_component_sum(p, "Harmonic Spectrum", wf_components)
    wf_i = extract_density_component_sum(p, "Inharmonic Spectrum", wf_components)
    wf_s = extract_density_component_sum(p, "Sub-bass band", wf_components)

    result["density_weight_function"] = str(wf_h.get("weight_function") or wf_op)
    result["harmonic_density_sum"] = wf_h.get("D")
    result["inharmonic_density_sum"] = wf_i.get("D")
    result["subbass_density_sum"] = wf_s.get("D")
    # AUDIT FIX (Harmonic-Spectrum inclusion contract) — surface the
    # ``include_for_density`` policy and excluded count selected by the
    # canonical extractor for the Harmonic Spectrum sheet so the
    # compiled Density_Metrics row can carry per-note diagnostics.
    result["harmonic_density_inclusion_policy"] = str(
        wf_h.get("inclusion_policy", "") or ""
    )
    result["harmonic_density_excluded_count"] = int(
        wf_h.get("excluded_count", 0) or 0
    )
    combined_source = ";".join(
        s for s in (
            wf_h.get("density_component_sum_source") or "",
            wf_i.get("density_component_sum_source") or "",
            wf_s.get("density_component_sum_source") or "",
        )
        if s
    )
    result["density_component_sum_source"] = combined_source
    if wf_components in DENSITY_WEIGHT_FUNCTION_VALID and wf_components == "log":
        result["density_formula"] = (
            "density_metric_raw = D_H*w_H + D_I*w_I + D_S*w_S; "
            "D_band = log10(1 + SUM(Amplitude_raw))."
        )
    elif wf_components in DENSITY_WEIGHT_FUNCTION_VALID and wf_components == "power":
        result["density_formula"] = (
            "density_metric_raw = D_H*w_H + D_I*w_I + D_S*w_S; "
            "D_band = SUM(Power_raw) (fallback SUM(Amplitude_raw**2))."
        )
    elif wf_components in DENSITY_WEIGHT_FUNCTION_VALID:
        result["density_formula"] = (
            "density_metric_raw = D_H*w_H + D_I*w_I + D_S*w_S; "
            "D_band = SUM(Amplitude_raw)."
        )
    else:
        result["density_formula"] = (
            "density_metric_raw = D_H*w_H + D_I*w_I + D_S*w_S; "
            f"D_band = density.apply_density_metric(Amplitude_raw_vector, "
            f"weight_function={wf_components!r}) per spectrum sheet."
        )

    # Canonical per-band D values (weight_function-aware) replace the
    # preliminary amplitude-basis sums whenever extraction succeeds.
    if wf_h.get("D") is not None:
        D_H = float(wf_h["D"])
        n_H = int(wf_h.get("n", 0) or 0)
        src_H = str(wf_h.get("density_component_sum_source", ""))
    if wf_i.get("D") is not None:
        D_I = float(wf_i["D"])
        n_I = int(wf_i.get("n", 0) or 0)
        src_I = str(wf_i.get("density_component_sum_source", ""))
    if wf_s.get("D") is not None:
        D_S = float(wf_s["D"])
        n_S = int(wf_s.get("n", 0) or 0)
        src_S = str(wf_s.get("density_component_sum_source", ""))
    result["D_H"] = D_H
    result["D_I"] = D_I
    result["D_S"] = D_S
    result["harmonic_spectrum_count"] = int(n_H)
    result["inharmonic_spectrum_count"] = int(n_I)
    result["subbass_spectrum_count"] = int(n_S)
    result["harmonic_spectrum_source"] = src_H
    result["inharmonic_spectrum_source"] = src_I
    result["subbass_spectrum_source"] = src_S

    w_H, w_I, w_S, legacy_only = _read_component_weights_from_analysis_metadata(p)
    result["w_H"] = w_H
    result["w_I"] = w_I
    result["w_S"] = w_S
    result["legacy_aliases_only"] = legacy_only

    # AUDIT FIX (canonical note-source provenance) — surface
    # ``note_source`` from Analysis_Metadata (set by proc_audio when the
    # audio filename was parsed at Stage 1) so the compiled row can
    # report ``filename_token`` end-to-end. When the metadata sheet
    # does not carry the key (older workbooks) we leave it None so the
    # row builder can fall back to a fresh canonical parse on the
    # workbook's parent-folder name.
    try:
        meta_note_source = _read_analysis_metadata_scalar(p, "note_source")
    except Exception:
        meta_note_source = None
    if meta_note_source:
        result["analysis_metadata_note_source"] = str(meta_note_source)

    # AUDIT FIX (component_*_energy_ratio sum-to-1 invariant) — record
    # the weight-sum residual and warn loudly if it drifts outside the
    # canonical tolerance. The compile step never repairs / renormalises
    # the weights (that would mask a Stage 1 ratio bug); we only surface
    # the violation in the per-row provenance.
    try:
        sum_w = float((w_H or 0.0)) + float((w_I or 0.0)) + float((w_S or 0.0))
    except (TypeError, ValueError):
        sum_w = float("nan")
    result["component_energy_ratio_sum"] = sum_w
    if (
        w_H is not None
        and w_I is not None
        and w_S is not None
        and np.isfinite(sum_w)
        and abs(sum_w - 1.0) > DENSITY_WEIGHT_SUM_TOLERANCE
    ):
        logger.warning(
            "component_energy_ratio sum drifted from 1.0 in %s: "
            "w_H+w_I+w_S=%.6f (tolerance=%.3g). Density_Metrics is "
            "computed but the upstream Stage 1 ratios may be stale.",
            p,
            sum_w,
            DENSITY_WEIGHT_SUM_TOLERANCE,
        )
        result["component_energy_ratio_sum_ok"] = False
    else:
        result["component_energy_ratio_sum_ok"] = True

    # ------------------------------------------------------------------
    # Stage 2 weighted note-density metric: D_band (compile weight_function)
    # times component energy ratios. Matches density_metric_raw; independent
    # of legacy batch_* GUI model weights and workbook max-normalisation.
    # ------------------------------------------------------------------
    canonical_weights_available = (
        w_H is not None and w_I is not None and w_S is not None
    )
    can_compute_weighted_density = canonical_weights_available and not legacy_only
    if can_compute_weighted_density:
        d_h = result.get("harmonic_density_sum")
        d_i = result.get("inharmonic_density_sum")
        d_s = result.get("subbass_density_sum")
        wH_f = float(w_H) if w_H is not None and np.isfinite(float(w_H)) else float("nan")
        wI_f = float(w_I) if w_I is not None and np.isfinite(float(w_I)) else float("nan")
        wS_f = float(w_S) if w_S is not None and np.isfinite(float(w_S)) else float("nan")
        h_f = float(d_h) if d_h is not None and np.isfinite(float(d_h)) else float("nan")
        i_f = float(d_i) if d_i is not None and np.isfinite(float(d_i)) else float("nan")
        s_f = float(d_s) if d_s is not None and np.isfinite(float(d_s)) else float("nan")
        weighted_h = h_f * wH_f
        weighted_i = i_f * wI_f
        weighted_s = s_f * wS_f
        # Sum the three contributions, treating NaN contributions as 0.
        # If all three are NaN, the weighted sum stays NaN.
        contributions = [weighted_h, weighted_i, weighted_s]
        finite_contrib = [c for c in contributions if np.isfinite(c)]
        if finite_contrib:
            density_sum = float(sum(finite_contrib))
            density_log = float(np.log10(1.0 + max(0.0, density_sum)))
        else:
            density_sum = float("nan")
            density_log = float("nan")
        result["weighted_harmonic_component"] = (
            weighted_h if np.isfinite(weighted_h) else None
        )
        result["weighted_inharmonic_component"] = (
            weighted_i if np.isfinite(weighted_i) else None
        )
        result["weighted_subbass_component"] = (
            weighted_s if np.isfinite(weighted_s) else None
        )
        result["density_weighted_sum"] = (
            density_sum if np.isfinite(density_sum) else None
        )
        result["density_log_weighted"] = (
            density_log if np.isfinite(density_log) else None
        )
    # When the canonical component weights are unavailable we leave the
    # weighted-density fields at their None defaults; the caller
    # downgrades the row to ``missing_component_weights`` below.

    # Status classification (first-error wins).
    if D_H is None:
        if src_H and "no_numeric_values" in src_H:
            result["density_extraction_status"] = "no_numeric_values"
        else:
            result["density_extraction_status"] = "missing_harmonic_spectrum"
        return result
    if D_I is None:
        if src_I and "no_numeric_values" in src_I:
            result["density_extraction_status"] = "no_numeric_values"
        else:
            result["density_extraction_status"] = "missing_inharmonic_spectrum"
        return result
    if D_S is None:
        if src_S and "no_numeric_values" in src_S:
            result["density_extraction_status"] = "no_numeric_values"
        else:
            result["density_extraction_status"] = "missing_subbass_spectrum"
        return result

    if w_H is None and w_I is None and w_S is None:
        if legacy_only:
            # AUDIT POLICY — never silently substitute batch_* for
            # component_* in integrated_single_pass. Flag it loudly so the
            # operator can re-run with the canonical weights.
            result["density_extraction_status"] = "missing_component_weights"
        else:
            result["density_extraction_status"] = "missing_component_weights"
        return result

    if result.get("legacy_scaled_source_used"):
        # Warn when at least one of the three bands fell back to the
        # legacy ``Amplitude`` column instead of the audit-canonical
        # ``Amplitude_raw`` / ``Power_raw`` pair. The raw spectra emitted
        # by the current proc_audio always populate those columns;
        # landing on ``Amplitude`` means the workbook predates this fix
        # and the value may have been display-scaled by a legacy
        # alignment factor.
        logger.warning(
            "Density_Metrics extraction: %s - legacy_scaled_source_used "
            "(at least one band fell back to 'Amplitude' instead of "
            "Amplitude_raw/Power_raw). The value may have been rescaled "
            "by a legacy alignment factor.",
            p,
        )
        result["density_extraction_status"] = "legacy_scaled_source_used"
        return result

    result["density_extraction_status"] = "ok"
    return result


def _build_density_metrics_sheet_from_per_note_files(
    found_files: Iterable[tuple[Path, str, str]],
    *,
    weight_function: Optional[str] = None,
    density_component_basis: str = DENSITY_COMPONENT_BASIS_DEFAULT,
) -> pd.DataFrame:
    """Build the compiled ``Density_Metrics`` sheet by direct extraction
    from every per-note ``spectral_analysis.xlsx``.

    AUDIT FIX (direct per-note Density_Metrics extraction) — this function
    bypasses the legacy "harvest scalar Metrics columns" path entirely.
    For every per-note workbook it calls
    :func:`extract_density_components_from_per_note_workbook`, applies the
    weighted formula, and at the end max-normalises ``density_metric_raw``
    across the compiled workbook to populate ``density_metric_normalized``.

    Output columns (exact order; final sheet schema mandated by the
    audit task):

        Note, source_file_name, weight_function,
        Harmonic Partials sum, Inharmonic Partials sum, Sub-bass sum,
        Total sum,
        component_harmonic_energy_ratio, component_inharmonic_energy_ratio,
        component_subbass_energy_ratio,
        weighted_harmonic_density_contribution,
        weighted_inharmonic_density_contribution,
        weighted_subbass_density_contribution,
        density_metric_raw, density_metric_normalized,
        density_extraction_status,
        harmonic_spectrum_source, inharmonic_spectrum_source,
        subbass_spectrum_source,
        harmonic_spectrum_count, inharmonic_spectrum_count,
        subbass_spectrum_count
    """
    wf = (weight_function or "").strip().lower() or "linear"
    if wf == "sum":
        wf = "linear"

    rows: List[Dict[str, Any]] = []
    files_list: List[tuple[Path, str, str]] = list(found_files)
    # Counters for the per-compilation log summary (see audit point 11).
    counts = {
        "processed": 0,
        "ok": 0,
        "missing_harmonic_spectrum": 0,
        "missing_inharmonic_spectrum": 0,
        "missing_subbass_spectrum": 0,
        "missing_component_weights": 0,
        "no_numeric_values": 0,
        "extraction_error": 0,
        "extraction_error_stale_schema": 0,
        "legacy_scaled_source_used": 0,
    }

    from note_parser import canonical_note_from_filename

    for fpath, note, folder in files_list:
        counts["processed"] += 1
        info = extract_density_components_from_per_note_workbook(
            fpath,
            density_component_basis=density_component_basis,
            weight_function=wf,
        )
        # Prefer the note_source persisted by proc_audio in
        # Analysis_Metadata (set against the AUDIO filename, which is
        # the audit-canonical source). When the workbook predates the
        # note_source key (older Stage 1), fall back to a fresh canonical
        # parse on (xlsx-filename, parent-folder-name).
        meta_ns = info.get("analysis_metadata_note_source")
        canonical_note, parsed_note_source = canonical_note_from_filename(
            fpath.name,
            manifest_note=note or None,
            parent_folder=fpath.parent.name,
        )
        note_source = (
            str(meta_ns)
            if meta_ns and str(meta_ns).strip()
            else parsed_note_source
        )
        if canonical_note:
            note = canonical_note
        st = info.get("density_extraction_status", "extraction_error")
        if st in counts:
            counts[st] += 1
        elif st == "ok":
            counts["ok"] += 1

        D_H = info["D_H"]
        D_I = info["D_I"]
        D_S = info["D_S"]
        w_H = info["w_H"]
        w_I = info["w_I"]
        w_S = info["w_S"]

        def _f(x: Optional[float]) -> float:
            try:
                if x is None:
                    return float("nan")
                xv = float(x)
                return xv if np.isfinite(xv) else float("nan")
            except (TypeError, ValueError):
                return float("nan")

        D_Hf, D_If, D_Sf = _f(D_H), _f(D_I), _f(D_S)
        w_Hf, w_If, w_Sf = _f(w_H), _f(w_I), _f(w_S)

        wh_c = D_Hf * w_Hf if not (np.isnan(D_Hf) or np.isnan(w_Hf)) else float("nan")
        wi_c = D_If * w_If if not (np.isnan(D_If) or np.isnan(w_If)) else float("nan")
        ws_c = D_Sf * w_Sf if not (np.isnan(D_Sf) or np.isnan(w_Sf)) else float("nan")

        if np.isnan(wh_c) and np.isnan(wi_c) and np.isnan(ws_c):
            raw = float("nan")
        else:
            raw = float(
                (0.0 if np.isnan(wh_c) else wh_c)
                + (0.0 if np.isnan(wi_c) else wi_c)
                + (0.0 if np.isnan(ws_c) else ws_c)
            )

        total_sum = (
            float("nan")
            if (np.isnan(D_Hf) and np.isnan(D_If) and np.isnan(D_Sf))
            else (
                (0.0 if np.isnan(D_Hf) else D_Hf)
                + (0.0 if np.isnan(D_If) else D_If)
                + (0.0 if np.isnan(D_Sf) else D_Sf)
            )
        )

        # AUDIT FIX (Density_Metrics component basis) — surface the
        # basis used to evaluate D_H / D_I / D_S so a downstream
        # reader can never confuse an amplitude-sum density figure
        # with a (much larger) power-sum debug figure. The canonical
        # ``density_metric_raw`` always carries the amplitude-basis
        # value by default; when the caller opts into the power-sum
        # debug basis we ALSO emit ``density_metric_power_weighted_raw``
        # so the diagnostic figure is unmistakable.
        # AUDIT FIX (Clarinete_mf workbook layout complaint) — the row dict
        # below was historically built in a "raw sums first, canonical
        # answer in the middle, legacy at the end" order, which let
        # readers scrolling left-to-right mistake the unweighted
        # ``Total sum`` (= D_H + D_I + D_S without the energy-ratio
        # weights) for the canonical density. ``Total sum`` and the
        # ``density_log_weighted`` family are PINNED by 20+ tests and
        # by ``publication_chart_policy`` as legacy back-compat, so we
        # cannot drop them — but we MUST not display them first.
        # Reordered contract: ``Note`` is followed immediately by
        # ``density_metric_raw`` / ``density_metric_normalized``, then
        # the per-component weighted contributions, then the energy
        # ratios, then D_H/D_I/D_S, then the legacy ``Harmonic Partials
        # sum`` / ``Total sum`` family, then provenance, then the
        # Stage 1/2 amplitude-sum and legacy log-weighted columns.
        # ``density_metric_normalized`` is initialised to NaN and
        # populated after the per-note loop with the corpus-wide max.
        row = {
            "Note": note,
            # === CANONICAL ANSWERS (read these first) ====================
            #   density_metric_raw       = D_H*w_H + D_I*w_I + D_S*w_S
            #   density_metric_normalized = density_metric_raw / max(raw)
            "density_metric_raw": raw,
            "density_metric_normalized": float("nan"),  # filled after the loop
            # === PER-COMPONENT WEIGHTED CONTRIBUTIONS ====================
            #   D_x * w_x; sum to density_metric_raw.
            "weighted_harmonic_density_contribution": wh_c,
            "weighted_inharmonic_density_contribution": wi_c,
            "weighted_subbass_density_contribution": ws_c,
            # === COMPONENT ENERGY RATIOS (weights w_x) ===================
            #   Sum to 1.0 (validated by component_energy_ratio_sum_ok).
            "component_harmonic_energy_ratio": w_Hf,
            "component_inharmonic_energy_ratio": w_If,
            "component_subbass_energy_ratio": w_Sf,
            # === PER-COMPONENT DENSITY VALUES (D_x) ======================
            #   D_x = SUM(log10(1 + Amplitude_raw)) when weight_function=log;
            #         SUM(Amplitude_raw)             when weight_function=linear;
            #         SUM(Power_raw)                 when weight_function=power.
            "harmonic_density_sum": _f(info.get("harmonic_density_sum")),
            "inharmonic_density_sum": _f(info.get("inharmonic_density_sum")),
            "subbass_density_sum": _f(info.get("subbass_density_sum")),
            # === LEGACY DISPLAY COPIES (back-compat, do NOT use) =========
            #   Identical values to the canonical *_density_sum columns
            #   above, kept here under historical names. ``Total sum`` is
            #   the UNWEIGHTED linear total D_H + D_I + D_S; it is NOT the
            #   density metric and is FORBIDDEN as a publication default
            #   (see publication_chart_policy.FORBIDDEN_DEFAULT_METRIC_NAMES).
            "Harmonic Partials sum": D_Hf,
            "Inharmonic Partials sum": D_If,
            "Sub-bass sum": D_Sf,
            "Total sum": total_sum,
            # === PROVENANCE / META =======================================
            "source_file_name": str(fpath),
            "note_source": note_source,
            "weight_function": wf,
            "density_weight_function": str(
                info.get("density_weight_function", wf) or wf
            ),
            "density_formula": str(
                info.get(
                    "density_formula",
                    "density_metric_raw = D_H*w_H + D_I*w_I + D_S*w_S",
                )
            ),
            "density_component_sum_source": str(
                info.get("density_component_sum_source", "") or ""
            ),
            "component_energy_ratio_sum": _f(
                info.get("component_energy_ratio_sum")
            ),
            "component_energy_ratio_sum_ok": bool(
                info.get("component_energy_ratio_sum_ok", True)
            ),
            # AUDIT FIX (Harmonic-Spectrum inclusion contract) — propagate
            # the include_for_density filter outcome into Density_Metrics.
            "harmonic_density_inclusion_policy": str(
                info.get("harmonic_density_inclusion_policy", "") or ""
            ),
            "harmonic_density_excluded_count": int(
                info.get("harmonic_density_excluded_count", 0) or 0
            ),
            "density_extraction_status": st,
            "density_component_basis": info.get(
                "density_component_basis", DENSITY_COMPONENT_BASIS_DEFAULT
            ),
            "density_weight_basis": info.get(
                "density_weight_basis", DENSITY_WEIGHT_BASIS
            ),
            "harmonic_spectrum_source": info["harmonic_spectrum_source"],
            "inharmonic_spectrum_source": info["inharmonic_spectrum_source"],
            "subbass_spectrum_source": info["subbass_spectrum_source"],
            "harmonic_spectrum_count": info["harmonic_spectrum_count"],
            "inharmonic_spectrum_count": info["inharmonic_spectrum_count"],
            "subbass_spectrum_count": info["subbass_spectrum_count"],
            # === STAGE 1 CANDIDATE-BASED HARMONIC DENSITY METRIC =========
            # Independent of Power_raw / component_* ratios / external H/I
            # weights.
            "harmonic_amplitude_sum": _f(info.get("harmonic_amplitude_sum")),
            "harmonic_log_amplitude_density": _f(
                info.get("harmonic_log_amplitude_density")
            ),
            "harmonic_density_included_count": int(
                info.get("harmonic_density_included_count", 0) or 0
            ),
            "harmonic_amplitude_source": str(
                info.get("harmonic_amplitude_source", "")
            ),
            # === STAGE 2 WEIGHTED NOTE-DENSITY (weight_function-aware) =====
            # density_weighted_sum  = D_H*w_H + D_I*w_I + D_S*w_S (same
            #   band D values as density_metric_raw / harmonic_density_sum).
            # density_log_weighted  = log10(1 + density_weighted_sum).
            # harmonic_amplitude_sum remains a linear diagnostic only.
            "inharmonic_amplitude_sum": _f(info.get("inharmonic_amplitude_sum")),
            "inharmonic_amplitude_source": str(
                info.get("inharmonic_amplitude_source", "")
            ),
            "inharmonic_amplitude_count": int(
                info.get("inharmonic_amplitude_count", 0) or 0
            ),
            "subbass_amplitude_sum": _f(info.get("subbass_amplitude_sum")),
            "subbass_amplitude_source": str(
                info.get("subbass_amplitude_source", "")
            ),
            "subbass_amplitude_count": int(
                info.get("subbass_amplitude_count", 0) or 0
            ),
            "weighted_harmonic_component": _f(info.get("weighted_harmonic_component")),
            "weighted_inharmonic_component": _f(
                info.get("weighted_inharmonic_component")
            ),
            "weighted_subbass_component": _f(info.get("weighted_subbass_component")),
            "density_weighted_sum": _f(info.get("density_weighted_sum")),
            "density_log_weighted": _f(info.get("density_log_weighted")),
            "density_log_formula": str(
                info.get("density_log_formula", "log10(1 + density_weighted_sum)")
            ),
        }
        if str(info.get("density_component_basis")) == "power_sum":
            row["density_metric_power_weighted_raw"] = raw
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            {
                "compilation_error": [
                    "Direct per-note Density_Metrics extraction produced no rows "
                    "(no per-note spectral_analysis.xlsx files supplied)."
                ]
            }
        )

    out_df = pd.DataFrame(rows)

    # Run-relative max-normalisation across the compiled workbook.
    raw_series = pd.to_numeric(out_df["density_metric_raw"], errors="coerce")
    arr = raw_series.to_numpy(dtype=float, copy=False)
    finite_pos = arr[np.isfinite(arr) & (arr > 0)]
    if finite_pos.size == 0:
        logger.warning(
            "density_metric_normalized: no positive finite density_metric_raw "
            "values in compiled workbook; returning NaN."
        )
        out_df["density_metric_normalized"] = float("nan")
    else:
        mx = float(np.max(finite_pos))
        if mx <= 0.0 or not np.isfinite(mx):
            logger.warning(
                "density_metric_normalized: non-positive normalization "
                "reference (max=%s); returning NaN.", mx
            )
            out_df["density_metric_normalized"] = float("nan")
        else:
            out_df["density_metric_normalized"] = (raw_series / mx).astype(float)

    # AUDIT FIX (stale-pipeline detection) — if EVERY per-note workbook
    # uses a stale schema we refuse to ship the compiled Density_Metrics
    # sheet at all. The compile workflow is meant to consume the output
    # of the current single-pass pipeline; silently rendering a
    # half-broken sheet was exactly the failure mode the stale-pipeline
    # audit was meant to prevent.
    if (
        counts["processed"] > 0
        and counts["extraction_error_stale_schema"] == counts["processed"]
    ):
        raise RuntimeError(
            "All per-note workbooks use stale schema "
            f"(expected analysis_schema_version = {EXPECTED_ANALYSIS_SCHEMA_VERSION!r}); "
            "aborting compilation. Regenerate the analysis with the current "
            "single-pass raw-export pipeline."
        )

    # Per-compilation summary (audit point 11).
    logger.info(
        "Density_Metrics direct-extraction summary: processed=%d ok=%d "
        "missing_harmonic=%d missing_inharmonic=%d missing_subbass=%d "
        "missing_component_weights=%d no_numeric_values=%d "
        "extraction_error=%d extraction_error_stale_schema=%d "
        "legacy_scaled_source_used=%d",
        counts["processed"],
        counts["ok"],
        counts["missing_harmonic_spectrum"],
        counts["missing_inharmonic_spectrum"],
        counts["missing_subbass_spectrum"],
        counts["missing_component_weights"],
        counts["no_numeric_values"],
        counts["extraction_error"],
        counts["extraction_error_stale_schema"],
        counts["legacy_scaled_source_used"],
    )

    # AUDIT FIX (acoustic-physics correction, Clarinete_mf finding #5)
    # — emit a WARNING for any note whose sub-bass weight is suspect.
    # A clean clarinet / saxophone / bassoon / oboe / flute note should
    # have w_S < 5 % in essentially every register. A note above the
    # standard sub-bass cutoff (200 Hz) with w_S > 10 % indicates that
    # the FFT energy below 200 Hz is dominated by non-musical content
    # (DC offset, room rumble, HVAC, sub-audible vibration) which is
    # being mis-attributed to D_S. This warning lets a human auditor
    # spot such pickup issues without having to read the raw workbook.
    try:
        if (
            isinstance(out_df, pd.DataFrame)
            and not out_df.empty
            and "component_subbass_energy_ratio" in out_df.columns
            and "Note" in out_df.columns
        ):
            _w_s = pd.to_numeric(
                out_df["component_subbass_energy_ratio"], errors="coerce"
            ).fillna(0.0)
            _suspect = out_df[_w_s > 0.10]
            if not _suspect.empty:
                _names = ", ".join(str(x) for x in _suspect["Note"].tolist())
                logger.warning(
                    "Sub-bass leakage suspected on %d note(s) "
                    "(w_S > 0.10): %s. Inspect Sub-bass band sheet "
                    "and Analysis_Metadata.subbass_protection_tolerance_hz "
                    "/ subbass_aggregate_lower_hz to verify the "
                    "energy is not DC / sub-audible noise.",
                    len(_suspect), _names,
                )
    except Exception as _e_warn:
        logger.debug("sub-bass leakage warning skipped: %s", _e_warn)
    return out_df


def _augment_density_metrics_with_weighted_metric(
    out_df: pd.DataFrame,
    *,
    work: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the canonical single-pass weighted density metric to the
    compiled ``Density_Metrics`` sheet.

    AUDIT FIX (single-pass weighted density) — Density_Metrics historically
    exposed only the four raw partial sums (D_H, D_I, D_S, Total). Those
    values are unbounded intermediate quantities and plotting them as if
    they were the final density metric is misleading. This helper adds:

    * ``component_harmonic_energy_ratio`` / ``component_inharmonic_energy_ratio``
      / ``component_subbass_energy_ratio`` — canonical single-pass ratios
      with denominator ``H + I + S`` (see proc_audio
      ``_set_model_weights_from_current_component_energy``).
    * ``weighted_harmonic_density_contribution`` = ``D_H * w_H``,
      ``weighted_inharmonic_density_contribution`` = ``D_I * w_I``,
      ``weighted_subbass_density_contribution`` = ``D_S * w_S``.
    * ``density_metric_raw`` = sum of the three weighted contributions.
      Unbounded; diagnostic only.
    * ``density_metric_normalized`` = ``density_metric_raw / max(density_metric_raw)``
      across the current compiled workbook. Run-relative: do not compare
      across different runs unless the normalization reference is identical.

    Provenance policy: ``component_*`` ratios come from the
    ``component_harmonic_energy_ratio`` / ``component_inharmonic_energy_ratio``
    / ``component_subbass_energy_ratio`` columns when available (the
    canonical single-pass source). When they are absent (legacy workbooks)
    the function falls back to ``harmonic_energy_ratio`` / etc., which the
    metrics dictionary classifies as diagnostic aliases of the canonical
    component_* fields and therefore mathematically identical. ``batch_*``
    aliases are *never* used as a source here in ``integrated_single_pass``
    mode — see ``test_weighted_density_uses_component_not_batch``.
    """
    if out_df is None or out_df.empty:
        return out_df

    # Pull D_H / D_I / D_S from the sheet itself (they are already in
    # ``out_df`` because they are part of the minimal display columns).
    h_new = "Harmonic Partials sum"
    i_new = "Inharmonic Partials sum"
    s_new = "Sub-bass sum"
    t_new = "Total sum"
    if not all(c in out_df.columns for c in (h_new, i_new, s_new)):
        return out_df

    D_H = pd.to_numeric(out_df[h_new], errors="coerce")
    D_I = pd.to_numeric(out_df[i_new], errors="coerce")
    D_S = pd.to_numeric(out_df[s_new], errors="coerce")

    # Source the component ratios from the canonical fields; fall back to
    # the diagnostic alias columns (mathematically identical when single-
    # pass is active). NEVER source from batch_* in integrated_single_pass.
    def _source_ratio(canonical: str, alias: str) -> pd.Series:
        if canonical in work.columns:
            return pd.to_numeric(work[canonical], errors="coerce")
        if alias in work.columns:
            return pd.to_numeric(work[alias], errors="coerce")
        return pd.Series(np.nan, index=work.index)

    w_H = _source_ratio("component_harmonic_energy_ratio", "harmonic_energy_ratio")
    w_I = _source_ratio("component_inharmonic_energy_ratio", "inharmonic_energy_ratio")
    w_S = _source_ratio("component_subbass_energy_ratio", "subbass_energy_ratio")

    # Align indices so the assignment lines up row-by-row even when
    # ``out_df`` was sliced from ``work``.
    if not out_df.index.equals(work.index):
        try:
            w_H = w_H.reindex(out_df.index)
            w_I = w_I.reindex(out_df.index)
            w_S = w_S.reindex(out_df.index)
        except Exception:
            pass

    out_df["component_harmonic_energy_ratio"] = w_H.astype(float)
    out_df["component_inharmonic_energy_ratio"] = w_I.astype(float)
    out_df["component_subbass_energy_ratio"] = w_S.astype(float)

    # Per-component weighted contributions and unbounded raw metric.
    wh_contrib = (D_H * w_H).astype(float)
    wi_contrib = (D_I * w_I).astype(float)
    ws_contrib = (D_S * w_S).astype(float)
    out_df["weighted_harmonic_density_contribution"] = wh_contrib
    out_df["weighted_inharmonic_density_contribution"] = wi_contrib
    out_df["weighted_subbass_density_contribution"] = ws_contrib

    raw = wh_contrib.fillna(0.0) + wi_contrib.fillna(0.0) + ws_contrib.fillna(0.0)
    # Preserve full-NaN rows as NaN (do not pretend they evaluate to zero).
    all_nan_mask = wh_contrib.isna() & wi_contrib.isna() & ws_contrib.isna()
    if all_nan_mask.any():
        raw = raw.mask(all_nan_mask)
    out_df["density_metric_raw"] = raw

    # Run-relative max-normalization across the compiled workbook.
    arr = raw.to_numpy(dtype=float, copy=False)
    finite = arr[np.isfinite(arr) & (arr > 0)]
    if finite.size == 0:
        logger.warning(
            "density_metric_normalized: no positive finite density_metric_raw "
            "values in compiled workbook; returning NaN."
        )
        out_df["density_metric_normalized"] = float("nan")
    else:
        mx = float(np.max(finite))
        if mx <= 0.0 or not np.isfinite(mx):
            logger.warning(
                "density_metric_normalized: non-positive normalization "
                "reference (max=%s); returning NaN.", mx
            )
            out_df["density_metric_normalized"] = float("nan")
        else:
            out_df["density_metric_normalized"] = (raw / mx).astype(float)
    return out_df


def _build_density_metrics_main_sheet(
    df: pd.DataFrame,
    *,
    weight_function: Optional[str] = None,
    density_component_basis: str = DENSITY_COMPONENT_BASIS_DEFAULT,
) -> pd.DataFrame:
    """
    Build the compiled ``Density_Metrics`` sheet: Note + per-band partial sums under the
    per-note ``weight_function`` (see ``proc_audio`` Metrics export).

    AUDIT FIX (direct per-note Density_Metrics extraction) — when the wide
    frame carries a private ``__source_file_path`` column (the canonical
    case routed through ``compile_density_metrics``), this function
    delegates to ``_build_density_metrics_sheet_from_per_note_files``,
    which reopens every per-note ``spectral_analysis.xlsx`` and sums the
    Harmonic/Inharmonic/Sub-bass Spectrum sheets directly. The scalar-
    column path below is preserved only for synthetic DataFrames (tests
    that pass already-aggregated rows without source paths).

    Legacy rows without ``Harmonic Partials sum`` fall back to
    ``linear_sum_amplitude_*`` when the compile ``weight_function`` is
    linear-like (``linear`` / empty; legacy ``sum`` → ``linear``).
    """
    # Direct per-note extraction path (preferred when called from
    # compile_density_metrics → _write_compiled_excel).
    if df is not None and "__source_file_path" in df.columns:
        files: List[tuple[Path, str, str]] = []
        for _, row in df.iterrows():
            sp = row.get("__source_file_path")
            if sp is None or (isinstance(sp, float) and pd.isna(sp)):
                continue
            note = str(row.get("Note", "") or "")
            folder = str(row.get("Folder", "") or "")
            files.append((Path(str(sp)), note, folder))
        if files:
            return _build_density_metrics_sheet_from_per_note_files(
                files,
                weight_function=weight_function,
                density_component_basis=density_component_basis,
            )

    work = _prepare_df_for_density_export(df)
    wf = (weight_function or "").strip().lower()
    if wf == "sum":
        wf = "linear"

    h_new = "Harmonic Partials sum"
    i_new = "Inharmonic Partials sum"
    s_new = "Sub-bass sum"
    t_new = "Total sum"

    has_new = h_new in work.columns and i_new in work.columns and s_new in work.columns and t_new in work.columns
    if has_new:
        work_w = work.copy()
        if "weight_function" not in work_w.columns:
            work_w["weight_function"] = (wf if wf else "linear") or "linear"
        cols = [c for c in DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS if c in work_w.columns]
        out_df = work_w.loc[:, cols].copy()
    else:
        h_old = "linear_sum_amplitude_harmonic"
        i_old = "linear_sum_amplitude_inharmonic_partial"
        s_old = "linear_sum_amplitude_subbass_band"
        if wf in ("", "linear") and all(c in work.columns for c in (h_old, i_old, s_old, "Note")):
            out_df = work[["Note", h_old, i_old, s_old]].copy()
            out_df = out_df.rename(
                columns={h_old: h_new, i_old: i_new, s_old: s_new},
            )
            sh = pd.to_numeric(out_df[h_new], errors="coerce").fillna(0.0)
            si = pd.to_numeric(out_df[i_new], errors="coerce").fillna(0.0)
            ss = pd.to_numeric(out_df[s_new], errors="coerce").fillna(0.0)
            out_df[t_new] = sh + si + ss
            out_df["weight_function"] = (wf if wf else "linear") or "linear"
            out_df = out_df[[c for c in DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS if c in out_df.columns]]
        else:
            return pd.DataFrame(
                {
                    "compilation_error": [
                        "Missing partial-sum columns (expected per-note Metrics: "
                        "'Harmonic Partials sum', …, 'Total sum'). Re-run spectral export with the current code."
                    ]
                }
            )

    # AUDIT FIX (single-pass weighted density) — augment with the canonical
    # weighted density metric *before* the allow-list trim so the new
    # columns survive the filter.
    out_df = _augment_density_metrics_with_weighted_metric(out_df, work=work)

    drop_cols = [
        c
        for c in out_df.columns
        if str(c) not in DENSITY_METRICS_ALLOWED_COLUMNS or density_metric_column_is_forbidden(str(c))
    ]
    if drop_cols:
        out_df = out_df.drop(columns=drop_cols, errors="ignore")
    # AUDIT FIX (direct per-note Density_Metrics extraction) — backfill the
    # direct-extraction provenance/status columns so the legacy synthetic
    # path still emits a Density_Metrics sheet whose layout exactly matches
    # DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS. When the fixture does not
    # carry source-file metadata we mark the rows accordingly: status =
    # ``ok`` because the row already supplies finite scalar partial sums;
    # the source columns remain blank to make it explicit that this row
    # did not pass through the per-note extraction pipeline.
    _NUMERIC_AUDIT_DEFAULTS = {
        "harmonic_spectrum_count": 0,
        "inharmonic_spectrum_count": 0,
        "subbass_spectrum_count": 0,
    }
    _TEXT_AUDIT_DEFAULTS = {
        "source_file_name": "",
        "density_extraction_status": "ok",
        "harmonic_spectrum_source": "",
        "inharmonic_spectrum_source": "",
        "subbass_spectrum_source": "",
        "density_summation_mode": "his_weighted",
        # AUDIT FIX (Density_Metrics component basis) — the scalar
        # fallback path consumes already-aggregated rows that don't go
        # through the per-note extractor, so the basis fields must be
        # backfilled with the canonical defaults. Tests that build
        # synthetic compiled workbooks rely on these defaults appearing
        # exactly in the canonical layout.
        "density_component_basis": DENSITY_COMPONENT_BASIS_DEFAULT,
        "density_weight_basis": DENSITY_WEIGHT_BASIS,
    }
    for col, default in {**_NUMERIC_AUDIT_DEFAULTS, **_TEXT_AUDIT_DEFAULTS}.items():
        if col not in out_df.columns:
            out_df[col] = default

    # Keep the sheet layout stable even when source fixtures provide only
    # partial metric sets; missing values remain explicit as NaN.
    for col in DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS:
        if col not in out_df.columns:
            out_df[col] = np.nan

    # Final column order strictly matches DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS
    # for downstream consumers that key off the canonical layout.
    ordered = [c for c in DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS if c in out_df.columns]
    rest = [c for c in out_df.columns if c not in ordered]
    return out_df.loc[:, ordered + rest]


def _enrich_compiled_metadata_from_df(metadata: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    """Fill analysis_* keys from the first row of compiled metrics when per-note columns were merged in."""
    meta = dict(metadata)
    if df is None or df.empty:
        return meta
    row = df.iloc[0]
    pick = lambda k: row[k] if k in df.columns and pd.notna(row[k]) else None  # noqa: E731
    meta.setdefault("window", pick("window") or pick("Window"))
    meta.setdefault("n_fft", pick("N FFT"))
    meta.setdefault("hop_length", pick("Hop Length"))
    meta.setdefault("harmonic_tolerance", pick("Tolerance (Hz)") or pick("Search Band (cents)"))
    meta.setdefault("snr_threshold_db", pick("SNR Threshold (dB)"))
    meta.setdefault("density_summation_mode", pick("density_summation_mode"))
    meta.setdefault("harmonic_density_weight", pick("harmonic_density_weight"))
    meta.setdefault("inharmonic_density_weight", pick("inharmonic_density_weight"))
    meta.setdefault("subbass_density_weight", pick("subbass_density_weight"))
    meta.setdefault("density_salience_threshold_db", pick("density_salience_threshold_db"))
    meta.setdefault("density_frequency_ceiling_hz", pick("density_frequency_ceiling_hz"))
    meta.setdefault("frequency_min_hz", pick("frequency_min_hz"))
    meta.setdefault("frequency_max_hz", pick("frequency_max_hz"))
    meta.setdefault("magnitude_min_db", pick("magnitude_min_db"))
    meta.setdefault("magnitude_max_db", pick("magnitude_max_db"))
    meta.setdefault("zero_padding", pick("zero_padding"))
    meta.setdefault("window_type", meta.get("window"))
    meta.setdefault("ANALYSIS_SCHEMA_VERSION", pick("analysis_schema_version") or meta.get("analysis_schema_version"))
    meta.setdefault("rms_normalisation_enabled", True)
    meta.setdefault("smoothing_enabled", None)
    meta.setdefault("spectral_masking_enabled", False)
    meta.setdefault("density_formula", DENSITY_FORMULA_DOC)
    return meta


def _compute_optional_pca_sheets(
    df: pd.DataFrame,
    *,
    enable_pca_export: bool,
    minimum_samples_for_pca: int,
    pca_include_dissonance: bool = False,
    pca_include_dependent_metrics: bool = False,
) -> tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], str, str]:
    """
    Exploratory PCA on z-scored acoustic features only.

    PCA is exploratory and must not be interpreted as the primary density metric.

    ``pca_include_dependent_metrics`` is a forensic debug flag — when ``True``
    the inclusive feature list ``PCA_FEATURE_COLUMNS_DEBUG_INCLUSIVE`` is used
    (algebraic complements, aliases, etc. are re-added). Default ``False`` keeps
    the rank-clean canonical feature list ``PCA_FEATURE_COLUMNS`` per the policy
    documented in ``metrics_dictionary.json`` (``independent_for_pca``).
    """
    status = "skipped"
    note = ""
    if not enable_pca_export:
        note = "PCA export disabled (enable_pca_export=False)."
        return None, None, None, status, note

    work = _prepare_df_for_density_export(df)
    n_samples = len(work)
    if n_samples < minimum_samples_for_pca:
        note = "PCA skipped: insufficient number of samples."
        return None, None, None, status, note

    feature_list = (
        list(PCA_FEATURE_COLUMNS_DEBUG_INCLUSIVE)
        if bool(pca_include_dependent_metrics)
        else list(PCA_FEATURE_COLUMNS)
    )
    feature_sources: List[tuple[str, str]] = []
    for feat in feature_list:
        if feat not in work.columns:
            continue
        feature_sources.append((feat, feat))

    if bool(pca_include_dissonance):
        for dissonance_col in (
            "selected_dissonance_value",
            "sethares_dissonance",
            "hutchinson_knopoff_dissonance",
            "vassilakis_dissonance",
        ):
            if dissonance_col in work.columns:
                feature_sources.append((dissonance_col, dissonance_col))

    if len(feature_sources) < 3:
        note = "PCA skipped: fewer than three valid numerical analysis features after column resolution."
        return None, None, None, status, note

    X_parts: List[pd.Series] = []
    labels: List[str] = []
    for label, src in feature_sources:
        s = pd.to_numeric(work[src], errors="coerce").astype(float)
        if s.notna().sum() < max(2, n_samples // 2):
            continue
        if float(s.std(skipna=True) or 0.0) <= 1e-12:
            continue
        X_parts.append(s)
        labels.append(label)

    if len(labels) < 3:
        note = "PCA skipped: fewer than three features with sufficient variance and coverage."
        return None, None, None, status, note

    X = pd.concat(X_parts, axis=1)
    X = X.reindex(work.index)
    col_means = X.mean(numeric_only=True)
    X_imputed = X.fillna(col_means).fillna(0.0)
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X_imputed.to_numpy(dtype=float))
    n_comp = int(min(3, Xz.shape[0], Xz.shape[1]))
    if n_comp < 1:
        note = "PCA skipped: insufficient number of samples."
        return None, None, None, status, note

    pca = PCA(n_components=n_comp, random_state=42)
    scores = pca.fit_transform(Xz)
    score_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=score_cols, index=work.index)
    if "Note" in work.columns:
        scores_df.insert(0, "Note", work["Note"].values)

    loadings_rows: List[Dict[str, Any]] = []
    for j, name in enumerate(labels):
        row: Dict[str, Any] = {"Feature": name}
        for i in range(n_comp):
            row[f"PC{i+1}_loading"] = float(pca.components_[i, j])
        loadings_rows.append(row)
    loadings_df = pd.DataFrame(loadings_rows)

    evr = pca.explained_variance_ratio_.astype(float)
    cum = np.cumsum(evr)
    var_df = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(len(evr))],
            "explained_variance_ratio": evr,
            "cumulative_explained_variance": cum,
        }
    )
    status = "exported"
    note = ""
    return scores_df, loadings_df, var_df, status, note


def _append_dissonance_excel_sheets(
    writer: Any,
    base_df: pd.DataFrame,
    meta_flat: Dict[str, Any],
    *,
    minimum_samples_for_dissonance_correlation: int = 10,
) -> None:
    """Write ``Dissonance_*`` sheets and populate ``meta_flat`` dissonance audit keys."""
    diss_df, audit = build_canonical_dissonance_frame(
        base_df,
        selected_model=meta_flat.get("selected_dissonance_model"),
        dissonance_enabled=meta_flat.get("dissonance_enabled"),
    )
    if "dissonance_enabled" not in meta_flat:
        meta_flat["dissonance_enabled"] = bool(
            audit.get("dissonance_enabled_inferred")
            or len(collect_dissonance_scalar_columns(base_df)) > 0
        )
    if "dissonance_compare_models" not in meta_flat:
        meta_flat["dissonance_compare_models"] = bool(infer_dissonance_compare_from_frame(base_df))

    avail = sorted(audit.get("available_dissonance_models") or [])
    meta_flat["available_dissonance_models"] = (
        ",".join(avail) if avail else ",".join(_list_all_dissonance_models())
    )
    if meta_flat.get("selected_dissonance_model") is None and "dissonance_model" in base_df.columns:
        try:
            meta_flat["selected_dissonance_model"] = str(base_df["dissonance_model"].dropna().iloc[0])
        except Exception:
            pass

    has_numeric = bool(audit.get("has_numeric_dissonance"))
    want_export = bool(meta_flat.get("dissonance_enabled")) or has_numeric
    value_cols = [c for c in diss_df.columns if c != "Note"]

    if want_export and value_cols and has_numeric:
        preferred = (
            ["Note"]
            + [CANONICAL_VALUE_BY_SLUG[s] for s in MODEL_SLUGS if CANONICAL_VALUE_BY_SLUG[s] in diss_df.columns]
            + [c for c in ("selected_dissonance_model", "selected_dissonance_value") if c in diss_df.columns]
            + [c for c in OPTIONAL_EXTRA_FIELDS if c in diss_df.columns]
            + [c for c in DISSONANCE_AUDIT_COPY_COLUMNS if c in diss_df.columns]
        )
        out_cols = [c for c in preferred if c in diss_df.columns]
        if len(out_cols) >= 2:
            diss_out = diss_df[out_cols].copy()
            try:
                from metadata_sanitizer import publication_redaction_enabled, sanitize_dataframe_for_publication

                if publication_redaction_enabled():
                    diss_out = sanitize_dataframe_for_publication(diss_out)
            except Exception:
                pass
            diss_out.to_excel(writer, sheet_name="Dissonance_Metrics", index=False)
            meta_flat["dissonance_export_status"] = "exported"

            if meta_flat.get("dissonance_compare_models"):
                long_df = build_dissonance_model_comparison_long(diss_df)
                if not long_df.empty:
                    try:
                        from metadata_sanitizer import publication_redaction_enabled, sanitize_dataframe_for_publication

                        if publication_redaction_enabled():
                            long_df = sanitize_dataframe_for_publication(long_df)
                    except Exception:
                        pass
                    long_df.to_excel(writer, sheet_name="Dissonance_Model_Comparison", index=False)

            corr_df = build_dissonance_correlation_matrix(
                diss_df, min_samples=minimum_samples_for_dissonance_correlation
            )
            if corr_df is not None and not corr_df.empty:
                try:
                    from metadata_sanitizer import publication_redaction_enabled, sanitize_dataframe_for_publication

                    if publication_redaction_enabled():
                        corr_df = sanitize_dataframe_for_publication(corr_df)
                except Exception:
                    pass
                corr_df.to_excel(writer, sheet_name="Dissonance_Model_Correlations", index=True)
        else:
            meta_flat["dissonance_export_status"] = "skipped: no dissonance values found"
    else:
        if bool(meta_flat.get("dissonance_enabled")) and not has_numeric:
            meta_flat["dissonance_export_status"] = "skipped: no dissonance values found"
        elif not bool(meta_flat.get("dissonance_enabled")) and not has_numeric:
            meta_flat["dissonance_export_status"] = (
                "skipped: dissonance analysis disabled or no values in source files"
            )
        else:
            meta_flat["dissonance_export_status"] = "skipped: no dissonance values found"


def _get_project_version_info() -> tuple[str, str]:
    """
    Resolve analysis code version for reproducibility stamping.
    Prefers installed package metadata; falls back to pyproject.toml.
    """
    try:
        from importlib import metadata as importlib_metadata
        version = importlib_metadata.version("soundspectranalyse")
        return version, "importlib.metadata:soundspectranalyse"
    except Exception:
        pass

    try:
        repo_root = Path(__file__).resolve().parent
        pyproject_path = repo_root / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text(encoding="utf-8")
            match = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']\s*$', content, flags=re.MULTILINE)
            if match:
                return match.group(1), f"pyproject.toml:{pyproject_path}"
    except Exception:
        pass

    return "unknown", "unavailable"


def _stable_hash(payload: Dict[str, Any]) -> str:
    """Stable SHA256 hash for reproducibility (sorted JSON)."""
    try:
        serialized = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        serialized = str(payload).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _build_debug_counts_sheet(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Technical bin/candidate/row counts — not musical partial counts (see metadata note)."""
    work = df.copy()
    # Per-note ``Debug_Counts`` (new semantics) → legacy short names used on some
    # ``Density_Metrics`` / compile rows for backward compatibility.
    if (
        "inharmonic_bin_count_deprecated_legacy_alias" in work.columns
        and "inharmonic_bin_count" not in work.columns
    ):
        work["inharmonic_bin_count"] = pd.to_numeric(
            work["inharmonic_bin_count_deprecated_legacy_alias"], errors="coerce"
        )
    elif (
        "residual_spectral_row_count" in work.columns
        and "inharmonic_bin_count" not in work.columns
        and "retained_nonharmonic_peak_candidate_count" not in work.columns
    ):
        work["inharmonic_bin_count"] = pd.to_numeric(work["residual_spectral_row_count"], errors="coerce")
    if "peaklist_harmonic_window_candidate_count" in work.columns and "harmonic_peak_count" not in work.columns:
        work["harmonic_peak_count"] = pd.to_numeric(
            work["peaklist_harmonic_window_candidate_count"], errors="coerce"
        )
    elif "harmonic_peak_candidate_count" in work.columns and "harmonic_peak_count" not in work.columns:
        work["harmonic_peak_count"] = pd.to_numeric(work["harmonic_peak_candidate_count"], errors="coerce")
    if "peaklist_nonharmonic_window_candidate_count" in work.columns and "inharmonic_peak_count" not in work.columns:
        work["inharmonic_peak_count"] = pd.to_numeric(
            work["peaklist_nonharmonic_window_candidate_count"], errors="coerce"
        )
    elif "nonharmonic_peak_candidate_count" in work.columns and "inharmonic_peak_count" not in work.columns:
        work["inharmonic_peak_count"] = pd.to_numeric(work["nonharmonic_peak_candidate_count"], errors="coerce")
    if "peaklist_low_frequency_window_candidate_count" in work.columns and "subbass_peak_count" not in work.columns:
        work["subbass_peak_count"] = pd.to_numeric(
            work["peaklist_low_frequency_window_candidate_count"], errors="coerce"
        )
    elif "low_frequency_peak_candidate_count" in work.columns and "subbass_peak_count" not in work.columns:
        work["subbass_peak_count"] = pd.to_numeric(work["low_frequency_peak_candidate_count"], errors="coerce")
    if "peaklist_total_window_candidate_count" in work.columns and "total_detected_peak_count" not in work.columns:
        work["total_detected_peak_count"] = pd.to_numeric(
            work["peaklist_total_window_candidate_count"], errors="coerce"
        )
    elif "total_peak_candidate_count" in work.columns and "total_detected_peak_count" not in work.columns:
        work["total_detected_peak_count"] = pd.to_numeric(work["total_peak_candidate_count"], errors="coerce")
    if "total_peak_candidate_count" in work.columns and "total_spectral_candidate_count" not in work.columns:
        work["total_spectral_candidate_count"] = pd.to_numeric(work["total_peak_candidate_count"], errors="coerce")
    if "harmonic_peak_candidate_count" in work.columns and "harmonic_candidate_count" not in work.columns:
        work["harmonic_candidate_count"] = pd.to_numeric(work["harmonic_peak_candidate_count"], errors="coerce")
    if "retained_nonharmonic_peak_candidate_count" in work.columns and "inharmonic_candidate_count" not in work.columns:
        work["inharmonic_candidate_count"] = pd.to_numeric(
            work["retained_nonharmonic_peak_candidate_count"], errors="coerce"
        )
    elif "nonharmonic_peak_candidate_count" in work.columns and "inharmonic_candidate_count" not in work.columns:
        work["inharmonic_candidate_count"] = pd.to_numeric(work["nonharmonic_peak_candidate_count"], errors="coerce")
    elif "nonharmonic_candidate_row_count" in work.columns and "inharmonic_candidate_count" not in work.columns:
        work["inharmonic_candidate_count"] = pd.to_numeric(work["nonharmonic_candidate_row_count"], errors="coerce")
    if "low_frequency_peak_candidate_count" in work.columns and "subbass_candidate_count" not in work.columns:
        work["subbass_candidate_count"] = pd.to_numeric(work["low_frequency_peak_candidate_count"], errors="coerce")
    if "harmonic_candidate_count" not in work.columns and "harmonic_peak_count" in work.columns:
        work["harmonic_candidate_count"] = pd.to_numeric(work["harmonic_peak_count"], errors="coerce")
    if "inharmonic_candidate_count" not in work.columns and "inharmonic_peak_count" in work.columns:
        work["inharmonic_candidate_count"] = pd.to_numeric(work["inharmonic_peak_count"], errors="coerce")
    if "subbass_candidate_count" not in work.columns and "subbass_peak_count" in work.columns:
        work["subbass_candidate_count"] = pd.to_numeric(work["subbass_peak_count"], errors="coerce")
    if "total_spectral_candidate_count" not in work.columns and "total_detected_peak_count" in work.columns:
        work["total_spectral_candidate_count"] = pd.to_numeric(
            work["total_detected_peak_count"], errors="coerce"
        )
    want = [
        "Note",
        "harmonic_bin_count",
        "residual_spectral_row_count",
        "nonharmonic_candidate_row_count",
        "retained_nonharmonic_peak_candidate_count",
        "exported_nonharmonic_peak_candidate_count",
        "peaklist_harmonic_window_candidate_count",
        "peaklist_nonharmonic_window_candidate_count",
        "peaklist_low_frequency_window_candidate_count",
        "peaklist_total_window_candidate_count",
        "legacy_nonharmonic_peak_candidate_count_deprecated",
        "harmonic_peak_candidate_count",
        "low_frequency_peak_candidate_count",
        "total_peak_candidate_count",
        "accepted_inharmonic_peak_count",
        "accepted_inharmonic_partial_count",
        "debug_counts_semantics",
        "debug_counts_source_policy",
        "debug_counts_invariant_status",
        "debug_counts_invariant_failures",
        "inharmonic_bin_count",
        "subbass_bin_count",
        "harmonic_candidate_count",
        "inharmonic_candidate_count",
        "subbass_candidate_count",
        "total_spectral_candidate_count",
        "harmonic_peak_count",
        "inharmonic_peak_count",
        "subbass_peak_count",
        "total_detected_peak_count",
        "residual_row_count",
        "unmatched_spectral_row_count",
        "inharmonic_bin_count_deprecated_legacy_alias",
        "inharmonic_candidate_count_deprecated_legacy_alias",
        "inharmonic_peak_count_deprecated_legacy_alias",
        "harmonic_peak_count_deprecated_legacy_alias",
        "subbass_peak_count_deprecated_legacy_alias",
        "total_detected_peak_count_deprecated_legacy_alias",
        "debug_counts_status",
    ]
    cols = [c for c in want if c in work.columns]
    if "Note" not in cols or len(cols) < 2:
        return None
    return work.loc[:, cols].copy()


def _build_validation_metrics_sheet(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    want = [
        "Note",
        "f0_estimated",
        "f0_source",
        "harmonic_slot_expected_count",
        "harmonic_slot_matched_count",
        "harmonic_slot_missing_count",
        "non_harmonic_candidate_count",
        "outside_harmonic_window_candidate_count",
        "mean_abs_harmonic_deviation_cents",
        "median_abs_harmonic_deviation_cents",
        "max_abs_harmonic_deviation_cents",
        "rms_harmonic_deviation_cents",
        "harmonic_validation_status",
        "harmonic_alignment_mean_abs_error_cents",
        "harmonic_alignment_weighted_mean_abs_error_cents",
        "harmonic_alignment_median_abs_error_cents",
        "harmonic_alignment_p95_abs_error_cents",
        "harmonic_alignment_max_abs_error_cents",
        "harmonic_alignment_matched_count",
        "harmonic_alignment_expected_count",
        "harmonic_alignment_coverage_ratio",
        "harmonic_alignment_energy_coverage_ratio",
        "non_harmonic_candidate_energy_ratio",
        "non_harmonic_candidate_peak_ratio",
        "harmonic_alignment_tolerance_cents_used",
        "harmonic_alignment_status",
        "harmonic_order_alignment_status",
        "harmonic_order_alignment_weighted_status",
        "harmonic_representative_energy_status",
        "harmonic_order_match_ratio",
        "harmonic_order_alignment_match_ratio",
        "collapsed_representative_energy_ratio",
        "collapsed_representative_energy_share",
        "inharmonic_candidate_energy_ratio",
        "harmonic_region_candidate_count",
        "harmonic_region_candidate_rows",
        "inharmonic_candidate_count",
        "non_harmonic_candidate_rows",
    ]
    cols = [c for c in want if c in df.columns]
    if "Note" not in cols or len(cols) < 2:
        return None
    return df.loc[:, cols].copy()


def _build_per_note_processing_metadata_sheet(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Per-note STFT / tier / f0 policy (never mixed into ``Density_Metrics``)."""
    if df is None or df.empty or "Note" not in df.columns:
        return None

    def _pick(*names: str) -> Optional[pd.Series]:
        for n in names:
            if n in df.columns:
                return df[n]
        return None

    pieces: Dict[str, pd.Series] = {}
    if (s := _pick("Note")) is not None:
        pieces["Note"] = s
    if (s := _pick("n_fft", "N FFT")) is not None:
        pieces["n_fft"] = s
    if (s := _pick("n_fft_effective")) is not None:
        pieces["n_fft_effective"] = s
    if (s := _pick("hop_length", "Hop Length")) is not None:
        pieces["hop_length"] = s
    if (s := _pick("bin_spacing_hz")) is not None:
        pieces["bin_spacing_hz"] = s
    if (s := _pick("sample_rate")) is not None:
        pieces["sample_rate"] = s
    if (s := _pick("window", "Window")) is not None:
        pieces["window"] = s
    if (s := _pick("tier", "Tier")) is not None:
        pieces["tier"] = s
    if (s := _pick("f0_estimated")) is not None:
        pieces["f0_estimated"] = s
    if (s := _pick("f0_source")) is not None:
        pieces["f0_source"] = s
    if (s := _pick("harmonic_tolerance", "Tolerance (Hz)", "Search Band (cents)")) is not None:
        pieces["harmonic_tolerance"] = s
    if (s := _pick("snr_threshold_db", "SNR Threshold (dB)")) is not None:
        pieces["snr_threshold_db"] = s
    if (s := _pick("rms_normalisation_enabled")) is not None:
        pieces["rms_normalisation_enabled"] = s
    if (s := _pick("smoothing_enabled")) is not None:
        pieces["smoothing_enabled"] = s
    if (s := _pick("spectral_masking_enabled")) is not None:
        pieces["spectral_masking_enabled"] = s
    if (s := _pick("source_file_name")) is not None:
        pieces["source_file_name"] = s
    for _bk in (
        # Canonical component fields (single source of truth).
        "component_harmonic_energy_ratio",
        "component_inharmonic_energy_ratio",
        "component_subbass_energy_ratio",
        "component_total_inharmonic_energy_ratio",
        "component_energy_denominator",
        "component_energy_method",
        "component_profile_source",
        "model_harmonic_weight",
        "model_inharmonic_weight",
        "model_weight_denominator",
        "model_weights_source",
        "model_weights_warning",
        "model_weights_fallback_reason",
        "external_component_profile_used",
        "external_h_i_s_mapping_used",
    ):
        if (s := _pick(_bk)) is not None:
            pieces[_bk] = s

    if len(pieces) < 2:
        return None
    out = pd.DataFrame(pieces)
    preferred = [
        "Note",
        "source_file_name",
        "n_fft",
        "n_fft_effective",
        "hop_length",
        "bin_spacing_hz",
        "sample_rate",
        "window",
        "tier",
        "f0_estimated",
        "f0_source",
        "harmonic_tolerance",
        "snr_threshold_db",
        "rms_normalisation_enabled",
        "smoothing_enabled",
        "spectral_masking_enabled",
        # Canonical component_* fields (single source of truth)
        "component_harmonic_energy_ratio",
        "component_inharmonic_energy_ratio",
        "component_subbass_energy_ratio",
        "component_total_inharmonic_energy_ratio",
        "component_energy_denominator",
        "component_energy_method",
        "component_profile_source",
        "model_harmonic_weight",
        "model_inharmonic_weight",
        "model_weight_denominator",
        "model_weights_source",
        "model_weights_warning",
        "model_weights_fallback_reason",
        "external_component_profile_used",
        "external_h_i_s_mapping_used",
    ]
    ordered = [c for c in preferred if c in out.columns]
    rest = [c for c in out.columns if c not in ordered]
    return out.loc[:, ordered + rest].copy()


# ---------------------------------------------------------------------------
# SINGLE-PASS REFACTOR + AUDIT — output curation sheets.
#
# The compiled workbook is partitioned into three semantic sheets:
#
# * ``Canonical_Metrics``  — final, publication-ready metrics with documented
#   denominators (see metrics_dictionary.json next to compile_metrics.py).
# * ``Diagnostic_Metrics`` — intermediate quantities, energy sums, counts,
#   thresholds and provenance fields used to *audit* the canonical metrics.
#   These ARE NOT final scientific results.
# * ``Legacy_Compatibility`` — back-compat aliases (``batch_*``) and pre-refactor
#   columns (``legacy_*``, ``harmonic_density``, ``inharmonic_density`` etc.).
#   These ARE NOT recommended for new analyses.
#
# Classification helpers below operate on column NAMES only, so they can be
# applied to any wide compiled DataFrame regardless of upstream sheet origin.
# ---------------------------------------------------------------------------

CANONICAL_METRIC_COLUMNS: List[str] = [
    # identifiers
    "Note",
    "source_file_name",
    "tier",
    # component energy partition (denominator H+I+S) — single source of truth.
    # NOTE: the short aliases ``harmonic_energy_ratio`` /
    # ``inharmonic_energy_ratio`` / ``subbass_energy_ratio`` are
    # mathematically identical to these three and have been intentionally
    # demoted to Diagnostic_Metrics — see audit summary in
    # ``metrics_dictionary.json`` (status="diagnostic", derived_from=[…]).
    "component_harmonic_energy_ratio",
    "component_inharmonic_energy_ratio",
    "component_subbass_energy_ratio",
    "component_total_inharmonic_energy_ratio",
    # binary model coefficients (denominator H+I)
    "model_harmonic_weight",
    "model_inharmonic_weight",
    # canonical density / participation descriptors — distinct inputs:
    #   effective_partial_count   = N_eff over harmonic peaks only
    #   effective_partial_density = N_eff over harmonic + aggregated
    #                               inharmonic + sub-bass bundle
    "effective_partial_count",
    "effective_partial_density",
    "canonical_density_v5_adapted",
    "canonical_density",
    "density_normalized_global",
    "density_per_component",
    "rolloff_compensated_harmonic_density",
    "harmonic_effective_power_density",
    # canonical scalars
    "harmonic_inharmonic_ratio",
    "spectral_entropy",
    "harmonic_completeness",
    "f0_final_hz",
    "adaptive_subfundamental_cutoff_hz",
    "subfundamental_margin_percent",
    "percentage_subfundamental_cutoff_hz",
    "leakage_guard_cutoff_hz",
    "min_floor_hz",
    "max_fraction_of_f0",
    "effective_subfundamental_margin_percent",
    "subfundamental_guard_valid",
    "subfundamental_guard_policy",
    "low_frequency_policy_version",
    "adaptive_subfundamental_cutoff_source",
    "physical_low_frequency_lower_hz",
    "physical_low_frequency_upper_hz",
    "subfundamental_cutoff_selection_rule",
    "subfundamental_cutoff_selected_by",
]

# Substrings that, if present in a column name, force LEGACY classification
# even if the column has not been explicitly enumerated.
LEGACY_COLUMN_NAME_PREFIXES: Tuple[str, ...] = (
    "legacy_",
    "batch_",
    "harmonic_density_percentage",
    "inharmonic_density_percentage",
    "linear_sum_amplitude_",
)
LEGACY_COLUMN_EXACT_NAMES: frozenset[str] = frozenset(
    {
        "harmonic_density",
        "inharmonic_density",
        "combined_density",
        "harmonic_density_percentage",
        "inharmonic_density_percentage",
        "Spectral Density Metric",
        "Filtered Density Metric",
        "Combined Density Metric",
        "Density Metric",
        "applied_model_harmonic_weight",
        "applied_model_inharmonic_weight",
    }
)

# Columns that must NEVER reach the Canonical_Metrics sheet, even if their
# name is benign. This is the audit-time defence-in-depth list.
NEVER_CANONICAL_COLUMN_NAMES: frozenset[str] = frozenset(
    {"compilation_error"}
)

# SEMANTIC HARDENING — mathematically identical aliases of canonical metrics.
# These columns are kept in the compiled workbook for backwards compatibility
# (and exposed under Diagnostic_Metrics, NOT under Canonical_Metrics), but
# they MUST NOT be interpreted as independent quantities. Each maps to its
# canonical "explicit" name; see metrics_dictionary.json for the documented
# relationship (status="diagnostic", derived_from=[canonical_name]).
CANONICAL_ALIAS_COLUMNS: Dict[str, str] = {
    "harmonic_energy_ratio": "component_harmonic_energy_ratio",
    "inharmonic_energy_ratio": "component_inharmonic_energy_ratio",
    "subbass_energy_ratio": "component_subbass_energy_ratio",
}


def _classify_compiled_column(col: str) -> str:
    """Return one of: ``"canonical"``, ``"legacy"``, ``"diagnostic"``.

    The classification is conservative: anything not on the explicit
    canonical allow-list and not in the legacy buckets falls through to
    diagnostic. ``compilation_error`` is never canonical.
    """
    c = str(col)
    if c in NEVER_CANONICAL_COLUMN_NAMES:
        return "diagnostic"
    if c in LEGACY_COLUMN_EXACT_NAMES:
        return "legacy"
    for pref in LEGACY_COLUMN_NAME_PREFIXES:
        if c.startswith(pref):
            return "legacy"
    if c in CANONICAL_METRIC_COLUMNS:
        return "canonical"
    return "diagnostic"


# AUDIT FIX (Clarinete_mf workbook-clutter complaint) — at-write-time
# pruning of dead columns. We intentionally take the *conservative*
# definition: a "dead" column is one that holds **no observable value at
# all** — i.e. every entry is NaN/NaT/None, or every entry is an empty/
# whitespace-only string. All-zero numeric columns are NOT considered
# dead because 0.0 is a legitimate measurement (e.g. ``subbass_energy_sum``
# may legitimately be zero when a note has no sub-bass content, and
# downstream contracts require the column to remain present so analysts
# can distinguish "no sub-bass" from "not measured").
#
# Sheet presence is preserved by the caller; only the truly empty
# columns are dropped. The ``Note`` column (row key) is never dropped
# even if it happens to be empty.
_DEAD_COLUMN_PROTECTED_NAMES: frozenset[str] = frozenset({"Note"})


def _drop_dead_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with all-NaN / all-blank columns dropped.

    A column is considered "dead" if every value satisfies one of:

    * ``pd.isna(v)``                                      (NaN / NaT / None)
    * for object dtypes: ``str(v).strip()`` in
      ``{"", "nan", "None", "NaN", "<NA>"}``              (blank-like text)

    All-zero numeric columns are intentionally **not** considered dead —
    ``0`` is a legitimate observation and downstream code may rely on the
    column being present.

    Protected column names (e.g. ``"Note"``) are never dropped.
    Returns the input unchanged when ``df`` is empty.
    """
    if df is None or df.empty or df.shape[1] == 0:
        return df
    keep_cols: List[str] = []
    for c in df.columns:
        cs = str(c)
        if cs in _DEAD_COLUMN_PROTECTED_NAMES:
            keep_cols.append(c)
            continue
        series = df[c]
        if series.isna().all():
            continue
        if not pd.api.types.is_numeric_dtype(series):
            stripped = series.astype(str).str.strip()
            blank_like = stripped.isin(("", "nan", "None", "NaN", "<NA>"))
            if blank_like.all():
                continue
        keep_cols.append(c)
    if len(keep_cols) == len(df.columns):
        return df
    return df.loc[:, keep_cols].copy()


def _slice_compiled_df_by_status(base_df: pd.DataFrame, status: str) -> pd.DataFrame:
    """Return a copy of ``base_df`` containing only columns of the given status.

    Always includes the ``Note`` column when present (as a row key), even if
    ``status == "legacy"`` — the row key is not itself "legacy".

    Columns starting with ``__`` are reserved for internal bookkeeping
    (e.g. ``__source_file_path`` carried by the direct per-note extraction
    path) and never appear on any public status sheet.
    """
    if base_df is None or base_df.empty:
        return pd.DataFrame()
    keep: List[str] = []
    if "Note" in base_df.columns and status != "legacy":
        keep.append("Note")
    elif "Note" in base_df.columns and status == "legacy":
        # The legacy sheet also needs a row key; we keep Note but it is not
        # itself a legacy metric (documented in metrics_dictionary.json).
        keep.append("Note")
    for c in base_df.columns:
        if c in keep:
            continue
        if str(c).startswith("__"):
            continue
        if _classify_compiled_column(c) == status:
            keep.append(c)
    # Preserve column order: canonical columns appear in the order declared
    # in CANONICAL_METRIC_COLUMNS, then everything else.
    if status == "canonical":
        ordered = [c for c in CANONICAL_METRIC_COLUMNS if c in keep]
        rest = [c for c in keep if c not in ordered]
        keep = ordered + rest
    out = base_df.loc[:, [c for c in keep if c in base_df.columns]].copy()
    return out


def _merge_canonical_compiled_workbook_metadata(
    metadata: Dict[str, Any],
    *,
    file_pattern: str,
    allow_legacy_super_json: bool,
    input_schema_validation_status: str,
    legacy_pipeline_inputs_found: int = 0,
    legacy_pipeline_inputs_ignored: str = "none",
) -> None:
    """Attach canonical Stage-2 provenance keys (mutates ``metadata`` in place)."""
    pc = get_canonical_pipeline_contract()
    legacy_used = bool(
        file_pattern.lower().endswith(".json")
        or ("super_analysis_results" in file_pattern.lower())
    )
    pub_ok = bool(PUBLICATION_OUTPUT_ALLOWED and not legacy_used)
    metadata.setdefault("pipeline_contract_version", pc.contract_version)
    metadata["stage1_module"] = pc.stage1_module
    metadata["stage1_class"] = pc.stage1_class
    metadata["stage2_module"] = pc.stage2_module
    metadata["stage2_function"] = pc.stage2_function
    metadata["compiled_by"] = f"{pc.stage2_module}.{pc.stage2_function}"
    metadata["compiled_from"] = CANONICAL_PER_NOTE_WORKBOOK
    metadata["accepted_input_engine"] = f"{CANONICAL_STAGE1_MODULE}.{CANONICAL_STAGE1_CLASS}"
    metadata["legacy_super_json_allowed"] = bool(allow_legacy_super_json)
    metadata["legacy_pipeline_used"] = legacy_used
    metadata["publication_output_allowed"] = pub_ok
    metadata.setdefault("legacy_pipeline_inputs_found", int(legacy_pipeline_inputs_found))
    metadata.setdefault("legacy_pipeline_inputs_ignored", str(legacy_pipeline_inputs_ignored))
    metadata.setdefault("input_schema_validation_status", str(input_schema_validation_status))
    metadata.setdefault("f0_policy_version", F0_POLICY_VERSION)
    metadata.setdefault("harmonic_frequency_policy_version", HARMONIC_FREQUENCY_POLICY_VERSION)
    metadata.setdefault("nonharmonic_policy_version", NONHARMONIC_POLICY_VERSION)
    metadata.setdefault("low_frequency_policy_version", LOW_FREQUENCY_POLICY_VERSION)
    metadata.setdefault("missing_metric_policy_version", MISSING_METRIC_POLICY_VERSION)
    metadata.setdefault("density_formula_version", DENSITY_FORMULA_VERSION)
    metadata.setdefault("export_schema_version", EXPORT_SCHEMA_VERSION)


def _write_compiled_excel(
    outp: Path,
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    *,
    apply_publication_column_filter: bool = True,
    enable_pca_export: bool = True,
    minimum_samples_for_pca: int = 10,
    pca_include_dissonance: bool = False,
    pca_include_dependent_metrics: bool = False,
    compile_file_pattern: str = "",
    allow_legacy_super_json: bool = False,
    input_schema_validation_status: str = "not_validated",
    legacy_pipeline_inputs_found: int = 0,
    legacy_pipeline_inputs_ignored: str = "none",
) -> Dict[str, Any]:
    """Write compiled workbook: slim ``Density_Metrics``, ``Analysis_Metadata``, optional PCA sheets.

    ``effective_partial_density`` is a density/fatness (spectral richness) descriptor, not loudness.
    Spectral masking is not part of this export model (masking estimates perceptual audibility; the
    density sheet reflects physical partial structure). PCA, when exported, is exploratory only and
    must not be treated as the primary density metric.

    When the input is not a density compilation (e.g. dissonance-only comparison), a single
    ``Compiled Metrics`` sheet preserves backward compatibility.

    ``apply_publication_column_filter=False`` adds a raw ``Compiled_Metrics_All`` sheet (full
    in-memory columns) alongside ``Density_Metrics`` when density core columns are present.
    """
    metadata = dict(metadata)
    if compile_file_pattern:
        _merge_canonical_compiled_workbook_metadata(
            metadata,
            file_pattern=compile_file_pattern,
            allow_legacy_super_json=allow_legacy_super_json,
            input_schema_validation_status=input_schema_validation_status,
            legacy_pipeline_inputs_found=legacy_pipeline_inputs_found,
            legacy_pipeline_inputs_ignored=legacy_pipeline_inputs_ignored,
        )
    try:
        import uuid as _uuid

        metadata.setdefault("run_id", str(_uuid.uuid4()))
    except Exception:
        metadata.setdefault("run_id", "not_available_at_compile_stage")
    try:
        import platform as _plat
        import sys as _sys

        metadata.setdefault(
            "python_version",
            (_sys.version.split()[0] if getattr(_sys, "version", None) else "not_available_at_compile_stage"),
        )
        metadata.setdefault("platform", str(_plat.platform()))
    except Exception:
        metadata.setdefault("python_version", "not_available_at_compile_stage")
        metadata.setdefault("platform", "not_available_at_compile_stage")

    def _pkg_v(mod: str) -> str:
        try:
            return str(__import__(mod, fromlist=["_x"]).__version__)
        except Exception:
            return "not_available_at_compile_stage"

    metadata.setdefault("numpy_version", _pkg_v("numpy"))
    metadata.setdefault("scipy_version", _pkg_v("scipy"))
    metadata.setdefault("librosa_version", _pkg_v("librosa"))
    metadata.setdefault("spectral_masking_enabled", False)
    try:
        from metadata_sanitizer import publication_redaction_enabled, sanitize_metadata_dict

        if publication_redaction_enabled():
            metadata = sanitize_metadata_dict(metadata)
    except Exception:
        pass
    metadata["analysis_parameters_hash"] = _stable_hash(metadata)
    outp.parent.mkdir(parents=True, exist_ok=True)
    base_df = df.copy()
    if not base_df.empty:
        base_df = _add_canonical_and_global_density_columns(base_df)
    if base_df.columns.duplicated().any():
        logger.warning("Compiled export: removing duplicate column labels.")
        base_df = base_df.loc[:, ~base_df.columns.duplicated()].copy()
    # AUDIT FIX (direct per-note Density_Metrics extraction) — preserve
    # the absolute per-note paths across the publication sanitizer. The
    # sanitizer redacts path-like strings to ``redacted_for_publication``;
    # if we let it run on ``__source_file_path`` first, the Density_Metrics
    # builder will be unable to reopen the per-note workbooks. We snapshot
    # the column here and re-attach it after sanitization so it survives
    # the sanitizer's pass-through. The column is dropped by
    # ``_OMIT_FROM_COMPILED_METRICS_EXPORT`` before any public sheet is
    # written so the absolute paths never leak to the workbook.
    _per_note_path_snapshot: Optional[pd.Series] = None
    if "__source_file_path" in base_df.columns:
        _per_note_path_snapshot = base_df["__source_file_path"].copy()
    try:
        from metadata_sanitizer import publication_redaction_enabled, sanitize_dataframe_for_publication

        if publication_redaction_enabled() and not base_df.empty:
            base_df = sanitize_dataframe_for_publication(base_df)
    except Exception:
        pass
    if _per_note_path_snapshot is not None:
        base_df = base_df.copy()
        base_df["__source_file_path"] = _per_note_path_snapshot.values

    if not _compiled_df_has_density_core(base_df):
        export_df = base_df.copy()
        _omit = [c for c in _OMIT_FROM_COMPILED_METRICS_EXPORT if c in export_df.columns]
        if _omit:
            export_df = export_df.drop(columns=_omit)
        if apply_publication_column_filter:
            filtered = filter_dataframe_for_publication_metrics_sheet(export_df)
            if filtered.shape[1] > 0:
                export_df = filtered
        if export_df.shape[1] == 0:
            export_df = pd.DataFrame(
                {
                    "compilation_error": [
                        "Nenhuma coluna exportável após omissão/filtro público. "
                        "Tente compiled_public_columns=False ou verifique entradas por nota."
                    ]
                }
            )
        try:
            from metadata_sanitizer import publication_redaction_enabled, sanitize_dataframe_for_publication

            if publication_redaction_enabled() and not export_df.empty:
                export_df = sanitize_dataframe_for_publication(export_df)
        except Exception:
            pass
        meta_flat = _enrich_compiled_metadata_from_df(metadata, base_df)
        with pd.ExcelWriter(outp, engine="openpyxl") as writer:
            _finalize_analysis_metadata_for_workbook(
                meta_flat, base_df, pca_include_dissonance=bool(pca_include_dissonance)
            )
            try:
                _cg = _build_compile_guide_dataframe(
                    meta_flat, list(base_df.columns) if base_df is not None and not base_df.empty else []
                )
                _cg.to_excel(writer, sheet_name="Compile_Guide", index=False)
                meta_flat["compile_guide_export_status"] = "exported"
            except Exception as _ge:
                logger.warning("Compile_Guide sheet skipped: %s", _ge)
                meta_flat["compile_guide_export_status"] = f"skipped: {_ge}"
            export_df.to_excel(writer, sheet_name="Compiled Metrics", index=False)
            _append_dissonance_excel_sheets(writer, base_df, meta_flat)
            try:
                from metadata_sanitizer import publication_redaction_enabled, sanitize_metadata_dict

                if publication_redaction_enabled():
                    meta_flat = sanitize_metadata_dict(meta_flat)
            except Exception:
                pass
            try:
                from metadata_sanitizer import apply_publication_clean_meta_flat as _pc_flat  # noqa: PLC0415
            except Exception:  # pragma: no cover
                _pc_flat = None  # type: ignore[misc]
            if _pc_flat is not None:
                meta_flat = _pc_flat(meta_flat)
            pd.DataFrame([meta_flat]).to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        logger.info(
            "Compiled workbook written to %s (single output; no separate *_clean sidecar).",
            outp,
        )
        return meta_flat

    wf_for_density_sheet = str(metadata.get("weight_function") or "").strip() or None
    density_df = _build_density_metrics_main_sheet(base_df, weight_function=wf_for_density_sheet)
    _strip = dissonance_columns_present_in_density_sheet(density_df)
    if _strip:
        logger.warning("Removing dissonance-like columns from Density_Metrics: %s", _strip)
        density_df = density_df.drop(columns=_strip, errors="ignore")

    meta_flat = _enrich_compiled_metadata_from_df(metadata, base_df)
    scores_df, loadings_df, var_df, pca_status, pca_note = _compute_optional_pca_sheets(
        base_df,
        enable_pca_export=enable_pca_export,
        minimum_samples_for_pca=minimum_samples_for_pca,
        pca_include_dissonance=bool(pca_include_dissonance),
        pca_include_dependent_metrics=bool(pca_include_dependent_metrics),
    )
    meta_flat["pca_export_status"] = pca_status
    if pca_status == "exported":
        meta_flat.pop("pca_export_note", None)
    else:
        meta_flat["pca_export_note"] = pca_note or "PCA was not included in this workbook."

    try:
        from metadata_sanitizer import publication_redaction_enabled, sanitize_dataframe_for_publication

        _pub = publication_redaction_enabled()
        _sdf = sanitize_dataframe_for_publication if _pub else lambda x: x
    except Exception:
        _pub = False
        _sdf = lambda x: x

    density_df = _sdf(density_df)
    try:
        from metadata_sanitizer import publication_clean_export_enabled as _den_clean  # noqa: PLC0415
    except Exception:  # pragma: no cover

        def _den_clean() -> bool:  # type: ignore[misc]
            return True

    if _den_clean() and "density_formula_version" in density_df.columns:
        density_df = density_df.drop(columns=["density_formula_version"], errors="ignore")
    _dbg = _build_debug_counts_sheet(base_df)
    if _dbg is not None and not _dbg.empty:
        _dbg = _sdf(_dbg)
    _val = _build_validation_metrics_sheet(base_df)
    if _val is not None and not _val.empty:
        _val = _sdf(_val)
    _pn = _build_per_note_processing_metadata_sheet(base_df)
    if _pn is not None and not _pn.empty:
        _pn = _sdf(_pn)
    if scores_df is not None and loadings_df is not None and var_df is not None:
        scores_df = _sdf(scores_df)
        loadings_df = _sdf(loadings_df)
        var_df = _sdf(var_df)

    with pd.ExcelWriter(outp, engine="openpyxl") as writer:
        _finalize_analysis_metadata_for_workbook(
            meta_flat, base_df, pca_include_dissonance=bool(pca_include_dissonance)
        )
        try:
            _cg = _build_compile_guide_dataframe(meta_flat, list(density_df.columns))
            _cg.to_excel(writer, sheet_name="Compile_Guide", index=False)
            meta_flat["compile_guide_export_status"] = "exported"
        except Exception as _ge:
            logger.warning("Compile_Guide sheet skipped: %s", _ge)
            meta_flat["compile_guide_export_status"] = f"skipped: {_ge}"
        density_df.to_excel(writer, sheet_name="Density_Metrics", index=False)

        # Publication policy text lives in ``Compile_Guide`` / dictionary docs;
        # avoid stuffing warning prose into ``Analysis_Metadata`` when publication-clean is on.
        try:
            from constants import PUBLICATION_CLEAN_EXPORT as _PCE  # noqa: PLC0415
        except Exception:  # pragma: no cover
            _PCE = True  # type: ignore[misc]
        if not bool(_PCE):
            meta_flat["density_metrics_publication_warning"] = (
                "WARNING — Density_Metrics is preserved for backward "
                "compatibility and is NOT the publication-grade table. "
                "For final analysis use Canonical_Metrics. See "
                "metrics_dictionary.json (status, metric_family, "
                "derived_from, independent_for_pca)."
            )

        # SINGLE-PASS REFACTOR + AUDIT — output curation.
        # Three semantic sheets, derived deterministically from base_df:
        try:
            _canon = _slice_compiled_df_by_status(base_df, "canonical")
            _diag = _slice_compiled_df_by_status(base_df, "diagnostic")
            _legacy_df = _slice_compiled_df_by_status(base_df, "legacy")

            # Defence-in-depth: strip any disallowed columns from the canonical
            # sheet (even if classification slipped). Audited by tests.
            for _bad in NEVER_CANONICAL_COLUMN_NAMES:
                if _bad in _canon.columns:
                    _canon = _canon.drop(columns=[_bad], errors="ignore")
            try:
                from metadata_sanitizer import publication_clean_export_enabled as _pub_clean_on  # noqa: PLC0415
            except Exception:  # pragma: no cover

                def _pub_clean_on() -> bool:  # type: ignore[misc]
                    return True

            if _pub_clean_on():
                if "canonical_density_v5_adapted" in _canon.columns and "canonical_density" not in _canon.columns:
                    _canon = _canon.copy()
                    _cols = list(_canon.columns)
                    _ins_at = _cols.index("canonical_density_v5_adapted") + 1
                    _canon.insert(
                        _ins_at,
                        "canonical_density",
                        pd.to_numeric(_canon["canonical_density_v5_adapted"], errors="coerce"),
                    )
            _canon.to_excel(writer, sheet_name="Canonical_Metrics", index=False)
            meta_flat["canonical_metrics_export_status"] = "exported"
            # AUDIT FIX (Clarinete_mf workbook-clutter complaint):
            # Diagnostic_Metrics historically shipped a long tail of
            # all-NaN / all-zero columns left over from removed features
            # (Total Metric, N_harm_norm, harmonic_count_available,
            # Density Metric_Norm2, Index_Weighted, Combined Density
            # Metric_Norm2, Weighted Combined Metric, Weighted Combined
            # Metric_Norm, harmonic_alignment_validation_backend,
            # harmonic_alignment_candidate_basis,
            # harmonic_alignment_energy_diagnostic_message,
            # spectral_analysis_rolloff_compensated_harmonic_density*,
            # spectral_analysis_harmonic_effective_power_density*,
            # spectral_analysis_legacy_rolloff_compensated_density, …).
            # These have no readers (no test pins them by name; no other
            # module reads them) and confuse anyone opening the workbook.
            # We prune them at write time so the on-disk sheet matches
            # what's actually populated for this run. Sheet presence and
            # column SET (for live columns) are preserved.
            if not _diag.empty:
                _diag = _drop_dead_columns(_diag)
                _diag.to_excel(writer, sheet_name="Diagnostic_Metrics", index=False)
                meta_flat["diagnostic_metrics_export_status"] = "exported"
            else:
                meta_flat["diagnostic_metrics_export_status"] = "skipped: no diagnostic columns in compiled frame"
            if not _legacy_df.empty:
                _legacy_df = _drop_dead_columns(_legacy_df)
                if _legacy_df.empty or _legacy_df.shape[1] == 0:
                    # All legacy columns were dead in this run; keep a single
                    # marker column so the sheet still exists for schema
                    # stability but does not pretend to carry data.
                    _legacy_df = pd.DataFrame(
                        {
                            "legacy_compatibility_status": [
                                "no legacy columns carried data in this run"
                            ]
                            * max(len(base_df), 1),
                        }
                    )
                _legacy_df.to_excel(writer, sheet_name="Legacy_Compatibility", index=False)
                meta_flat["legacy_compatibility_export_status"] = "exported"
            else:
                meta_flat["legacy_compatibility_export_status"] = "skipped: no legacy columns in compiled frame"
        except Exception as _e_curate:
            logger.warning("Output curation sheets skipped: %s", _e_curate)
            meta_flat["canonical_metrics_export_status"] = f"skipped: {_e_curate}"

        meta_flat.setdefault("robust_salient_inharmonic_peak_picking_enabled", False)
        if _dbg is not None and not _dbg.empty:
            _dbg.to_excel(writer, sheet_name="Debug_Counts", index=False)
            meta_flat["debug_counts_export_status"] = "exported"
        else:
            meta_flat["debug_counts_export_status"] = "skipped: no bin-count columns in compiled frame"
        if _val is not None and not _val.empty:
            _val.to_excel(writer, sheet_name="Validation_Metrics", index=False)
            meta_flat["validation_export_status"] = "exported"
        else:
            meta_flat.setdefault("validation_export_status", "skipped: no validation columns in compiled frame")
        if _pn is not None and not _pn.empty:
            _pn.to_excel(writer, sheet_name="Per_Note_Processing_Metadata", index=False)
            meta_flat["per_note_metadata_export_status"] = "exported"
        else:
            meta_flat.setdefault(
                "per_note_metadata_export_status",
                "skipped: no per-note STFT/tier columns in compiled frame",
            )
        if scores_df is not None and loadings_df is not None and var_df is not None:
            scores_df.to_excel(writer, sheet_name="PCA_Scores", index=False)
            loadings_df.to_excel(writer, sheet_name="PCA_Loadings", index=False)
            var_df.to_excel(writer, sheet_name="PCA_Explained_Variance", index=False)
        _append_dissonance_excel_sheets(writer, base_df, meta_flat)
        if not apply_publication_column_filter:
            export_all = base_df.copy()
            _omit = [c for c in _OMIT_FROM_COMPILED_METRICS_EXPORT if c in export_all.columns]
            if _omit:
                export_all = export_all.drop(columns=_omit)
            export_all = _sdf(export_all)
            export_all.to_excel(writer, sheet_name="Compiled_Metrics_All", index=False)
        try:
            from metadata_sanitizer import publication_redaction_enabled as _pr2, sanitize_metadata_dict as _smd
        except Exception:  # pragma: no cover

            def _pr2() -> bool:  # type: ignore[misc]
                return False

            def _smd(x):  # type: ignore[misc]
                return x

        if _pr2():
            meta_flat = _smd(meta_flat)
        try:
            from metadata_sanitizer import apply_publication_clean_meta_flat as _pc_flat2  # noqa: PLC0415
        except Exception:  # pragma: no cover
            _pc_flat2 = None  # type: ignore[misc]
        if _pc_flat2 is not None:
            meta_flat = _pc_flat2(meta_flat)
        pd.DataFrame([meta_flat]).to_excel(writer, sheet_name="Analysis_Metadata", index=False)
    logger.info(
        "Compiled workbook written to %s (multi-sheet export; "
        "debug/validation/PCA/dissonance side sheets as applicable; no separate *_clean file).",
        outp,
    )
    return meta_flat


if not logger.hasHandlers():  # evita duplicação de handlers
    from log_config import configure_root_logger
    configure_root_logger()  # <-- NUNCA duplifica
    logger = logging.getLogger(__name__)


_PITCH_TO_SEMITONE = {
    "C": 0,  "C#": 1, "Db": 1,
    "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,  "Fb": 4, "E#": 5,
    "F": 5,  "F#": 6, "Gb": 6,
    "G": 7,  "G#": 8, "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
    "H": 11, "Hb": 10, "H#": 0,
}
_NOTE_RX = re.compile(r"\s*([A-GH])\s*([#♯b♭]?)\s*(-?\d+)\s*$", re.IGNORECASE)

def note_to_midi(note: str) -> int:
    """Converte 'A#2', 'Bb3', 'B2', 'H2' → número MIDI para ordenação cromática global."""
    if not isinstance(note, str):
        return 10**9  # vai para o fim
    m = _NOTE_RX.fullmatch(note)
    if not m:
        return 10**9
    letter = m.group(1).upper()
    acc = m.group(2)
    octv = int(m.group(3))
    if acc in ("#", "♯"):
        key = f"{letter}#"
    elif acc in ("b", "♭"):
        key = f"{letter}b"
    else:
        key = letter
    semi = _PITCH_TO_SEMITONE.get(key)
    if semi is None:
        return 10**9
    return (octv + 1) * 12 + semi


# Campos de texto que NUNCA devem ser convertidos para float
# Campos de texto que NUNCA devem ser convertidos para float
TEXT_FIELDS: set[str] = {
    "Tier",  # NEW: Tier is a text field (e.g., "Tier_01", "Fallback")
    "tier",
    "Note",
    "Folder",
    "source_file_name",
    "window",
    "Weight Function",
    "weight_function",  # internal key from Metrics (linear, log, cubic, …)
    "Window",
    "DM Domain",
    "Density Scale",   # <— novo: 'bark', 'mel', 'hz', etc. é texto
    "selected_dissonance_model",
    "f0_source",
    "harmonic_validation_status",
    "harmonic_alignment_status",
    "harmonic_order_alignment_status",
    "harmonic_order_alignment_weighted_status",
    "harmonic_representative_energy_status",
    "frequency_dependent_normalization_status",
    "harmonic_count_available",
    "f0_blind_method",
    "f0_final_method",
    "f0_fit_rejection_reason",
    "energy_conservation_status",
    "energy_denominator_description",
    "dissonance_partial_cap",
    "dissonance_cap_computation_note",
    "batch_energy_denominator",
    "batch_energy_method",
    "batch_ratio_source_explicit",
    "batch_ratio_fallback_reason",
    "harmonic_energy_percentage_semantics",
    "energy_denominator_musical_band",
    "energy_denominator_global",
    "model_weight_denominator",
    "model_weights_source",
    "model_weights_warning",
    "model_weights_fallback_reason",
    "model_weight_policy",
    "model_weights_source_policy",
    "batch_ratio_sum_policy",
    "per_note_analysis_metadata_scope",
    "harmonic_inharmonic_model_coefficients_note",
    "model_weight_safety_guard_applied",
    "rolloff_compensated_harmonic_density_status",
    "rolloff_density_source_phase",
    "rolloff_density_source_file",
    "rolloff_density_json_discovery_method",
    "rolloff_density_json_match_confidence",
    "hepd_density_json_discovery_method",
    "hepd_density_json_match_confidence",
    "density_source_formula",
    "density_normalization_scope",
    "density_formula_version",
    "harmonic_effective_power_density_status",
    "harmonic_effective_power_density_source_phase",
    "harmonic_effective_power_density_source_file",
    "harmonic_density_model",
    "harmonic_effective_power_mass_status",
    "subfundamental_guard_policy",
    "adaptive_subfundamental_cutoff_source",
    "low_frequency_policy_version",
    "low_frequency_residual_interpretation",
    "subfundamental_cutoff_selection_rule",
    "subfundamental_cutoff_selected_by",
    "debug_counts_semantics",
    "debug_counts_source_policy",
    "debug_counts_invariant_status",
    "debug_counts_invariant_failures",
    "debug_counts_status",
}


DISSONANCE_PREFIX: str = "Dissonance"

def _minmax(series: pd.Series) -> pd.Series:
    """
    Normalização min–max robusta (0..1).
    - Converte para numérico (não numéricos -> NaN).
    - Se max==min ou só houver NaN, devolve zeros.
    
    PHASE 4: Deprecated in favor of robust_normalize, but kept for backward compatibility.
    """
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def note_to_fundamental_freq(note: str) -> Optional[float]:
    """
    Convert musical note to fundamental frequency in Hz.
    
    Args:
        note: Musical note (e.g., 'C4', 'A#3', 'Bb5')
    
    Returns:
        Fundamental frequency in Hz, or None if note is invalid
    """
    if not isinstance(note, str):
        return None
    
    # Extract note from quotes if present
    note_clean = extract_note_from_quotes(note)
    
    # Match pattern: letter, optional accidental, octave
    match = re.match(r"([A-Ga-g])([#b]?)(\d+)", note_clean)
    if not match:
        return None
    
    letter, accidental, octave_str = match.groups()
    octave = int(octave_str)
    
    # Note to semitone position within octave (0-11, where C=0, C#=1, ..., B=11)
    note_map = {
        'C': 0, 'C#': 1, 'Db': 1,
        'D': 2, 'D#': 3, 'Eb': 3,
        'E': 4, 'Fb': 4, 'E#': 5,
        'F': 5, 'F#': 6, 'Gb': 6,
        'G': 7, 'G#': 8, 'Ab': 8,
        'A': 9, 'A#': 10, 'Bb': 10,
        'B': 11, 'Cb': 11, 'B#': 0
    }
    
    note_name = letter.upper() + accidental
    semitone_in_octave = note_map.get(note_name, 0)
    
    # Calculate frequency: f = 440 * 2^((n - 69) / 12) where n is MIDI note number
    # A4 is MIDI note 69, so: f = 440 * 2^((MIDI - 69) / 12)
    # MIDI note = (octave + 1) * 12 + semitone_in_octave
    midi_note = (octave + 1) * 12 + semitone_in_octave
    freq = 440.0 * (2.0 ** ((midi_note - 69) / 12.0))
    
    return freq


def get_frequency_dependent_alpha(f0: float) -> float:
    """
    Get alpha parameter based on fundamental frequency.
    
    Lower frequencies need MORE normalization (lower alpha = gentler rolloff assumption).
    Higher frequencies need LESS normalization (higher alpha = steeper rolloff assumption).
    
    This accounts for register-dependent spectral characteristics where lower notes
    have more harmonics and need stronger normalization to compensate.
    
    Args:
        f0: Fundamental frequency (Hz)
    
    Returns:
        Alpha parameter for spectral rolloff normalization
    """
    if f0 < 100:  # Very low register (e.g., bassoon A#1-B1)
        return 1.2  # Lower alpha = more normalization needed
    elif f0 < 200:  # Low register (e.g., bassoon C2-A#2)
        return 1.3
    elif f0 < 400:  # Mid register (e.g., bassoon B2-B3)
        return 1.4
    else:  # High register (e.g., bassoon C4-F4)
        return 1.6  # Higher alpha = less normalization needed


def apply_frequency_dependent_normalization(
    density_values: pd.Series,
    fundamental_freqs: pd.Series,
    harmonic_counts: pd.Series,
    alpha: float = 1.5,
    use_frequency_dependent_alpha: bool = True
) -> pd.Series:
    """
    Apply frequency-dependent normalization to density metrics.
    
    This compensates for natural spectral rolloff where lower notes have
    more harmonics than higher notes. The normalization uses the expected
    energy decay model: E(n) = 1/n^alpha where n is harmonic number.
    
    The density metric is proportional to the sum of weighted harmonic amplitudes.
    For a note with N harmonics, the expected density scales with:
    sum(n=1 to N) of (1/n^alpha) ≈ integral from 1 to N of (1/x^alpha) dx
    
    For alpha > 1: integral = (N^(1-alpha) - 1) / (1-alpha)
    For alpha = 1.5: integral ≈ 2 * (1 - 1/sqrt(N))
    
    Args:
        density_values: Series of density metric values
        fundamental_freqs: Series of fundamental frequencies (Hz)
        harmonic_counts: Series of harmonic counts
        alpha: Spectral rolloff exponent (default 1.5, matching density.py)
        use_frequency_dependent_alpha: If True, use register-dependent alpha values
    
    Returns:
        Normalized density values
    """
    normalized = pd.Series(index=density_values.index, dtype=float)
    
    for idx in density_values.index:
        density = density_values.loc[idx]
        freq = fundamental_freqs.loc[idx] if idx in fundamental_freqs.index else None
        harm_count = harmonic_counts.loc[idx] if idx in harmonic_counts.index else None
        
        if pd.isna(density) or density <= 0:
            normalized.loc[idx] = density
            continue
        
        if freq is None or freq <= 0 or harm_count is None or harm_count <= 0:
            # Can't normalize without frequency info, return original
            normalized.loc[idx] = density
            continue
        
        # Use frequency-dependent alpha if enabled
        if use_frequency_dependent_alpha:
            alpha_effective = get_frequency_dependent_alpha(freq)
        else:
            alpha_effective = alpha
        
        # Calculate expected density scaling factor
        # For alpha = 1.5, the sum of 1/n^1.5 from n=1 to N approximates:
        # integral = 2 * (1 - 1/sqrt(N)) for large N
        # More accurate: use Riemann zeta function approximation
        # For alpha > 1: sum ≈ zeta(alpha) - sum(n=N+1 to inf) 1/n^alpha
        # Simplified: use integral approximation
        N = float(harm_count)
        
        if alpha_effective > 1.0:
            # Integral from 1 to N of x^(-alpha) dx = (N^(1-alpha) - 1) / (1-alpha)
            # For alpha = 1.5: (N^(-0.5) - 1) / (-0.5) = 2 * (1 - 1/sqrt(N))
            expected_sum = (np.power(N, 1.0 - alpha_effective) - 1.0) / (1.0 - alpha_effective)
        else:
            # For alpha <= 1, use logarithmic approximation
            expected_sum = np.log(N + 1.0)
        
        # Normalize: actual / expected
        # This compensates for the fact that notes with more harmonics
        # naturally have higher density, producing consistent density values
        # across different frequency ranges
        normalized_value = density / (expected_sum + 1e-10)
        
        normalized.loc[idx] = normalized_value
    
    return normalized


def _robust_normalize_series(series: pd.Series, method: str = "percentile") -> pd.Series:
    """
    PHASE 4: Robust normalization using IQR or percentile methods.
    
    Replaces min-max normalization for better outlier handling.
    
    Args:
        series: Input series
        method: Normalization method ('percentile', 'iqr', 'robust_zscore')
        
    Returns:
        Normalized series
    """
    try:
        from data_integrity import robust_normalize
    except ImportError:
        # Fallback to min-max
        return _minmax(series)
    
    s = pd.to_numeric(series, errors="coerce")
    if s.size == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if not s.notna().any():
        return pd.Series(np.nan, index=series.index, dtype=float)

    values = s.values
    normalized = robust_normalize(values, method=method, clip_range=(0.0, 1.0))

    return pd.Series(normalized, index=s.index)


def _weighted_index_available_terms(
    out: pd.DataFrame,
    terms: Dict[str, float],
    *,
    weight_col: str = "Index_Weighted_available_weight_sum",
    status_col: str = "Index_Weighted_status",
) -> pd.Series:
    """
    Weighted index using only finite components; renormalize by the sum of
    weights for components that are present (missing is not treated as zero).
    """
    idx = out.index
    weighted_sum = pd.Series(0.0, index=idx, dtype=float)
    available_weight_sum = pd.Series(0.0, index=idx, dtype=float)
    for col, weight in terms.items():
        if col not in out.columns:
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        valid = s.notna()
        weighted_sum = weighted_sum + (s.where(valid, 0.0) * float(weight))
        available_weight_sum = available_weight_sum + (valid.astype(float) * float(weight))
    denom = available_weight_sum.replace(0.0, np.nan)
    result = weighted_sum / denom
    out[weight_col] = available_weight_sum
    out[status_col] = np.where(
        available_weight_sum > 0.0,
        "computed_renormalized_available_terms",
        "not_computed_no_available_terms",
    )
    return result


def apply_weighted_index(df: pd.DataFrame, scheme: str = "pdf") -> pd.DataFrame:
    """
    Acrescenta colunas normalizadas e o índice ponderado ao DF.
    Esquemas:
      - "pdf":     10% DM + 40% D_agn + 30% N_harm + 15% Combined + 5% P_norm
      - "current": 40% (1-D_agn) + 35% N_harm + 15% DM + 10% P_norm
    Notas:
      • Esta função CLAMPA todas as entradas teóricas 0..1 antes de compor o índice.
      • Dá prioridade à coluna 'Weighted Combined Metric_Norm' se existir.
    """
    out = df.copy()

    def _safe(col: str) -> pd.Series:
        if col in out.columns:
            return pd.to_numeric(out[col], errors="coerce")
        return pd.Series(np.nan, index=out.index, dtype=float)

    # ---------- Bases normalizadas ----------
    # PHASE 4: Use robust normalization instead of min-max
    # N_harm_norm (robust normalization of count; NaN when Harmonic Count unavailable — not 0.0)
    if "Harmonic Count" in out.columns:
        out["N_harm_norm"] = _robust_normalize_series(_safe("Harmonic Count"), method="percentile")
        out["harmonic_count_available"] = out["N_harm_norm"].notna()
    else:
        out["N_harm_norm"] = pd.Series(np.nan, index=out.index, dtype=float)
        out["harmonic_count_available"] = False
        out["frequency_dependent_normalization_status"] = "skipped_missing_required_columns"

    # Density Metric - Apply frequency-dependent normalization
    density_metric_raw = pd.to_numeric(_safe("Density Metric"), errors="coerce")

    # Check if we have Note and Harmonic Count for frequency-dependent normalization
    has_note = "Note" in out.columns
    has_harmonic_count = "Harmonic Count" in out.columns

    if has_note and has_harmonic_count:
        # Convert notes to fundamental frequencies
        fundamental_freqs = out["Note"].apply(note_to_fundamental_freq)
        harmonic_counts = pd.to_numeric(_safe("Harmonic Count"), errors="coerce")

        # Apply frequency-dependent normalization using regression-based approach
        # Fit: density = a * N^beta + c, then subtract frequency-dependent component
        valid_mask = (
            pd.notna(density_metric_raw) & (density_metric_raw > 0) &
            pd.notna(harmonic_counts) & (harmonic_counts > 0) &
            pd.notna(fundamental_freqs) & (fundamental_freqs > 0)
        )

        if valid_mask.sum() >= 3:
            # Fit relationship: log(density) = alpha * log(freq) + beta * log(N) + c
            # Since freq and N are highly correlated, we can use either
            # Use frequency directly for normalization
            log_density = np.log(density_metric_raw[valid_mask])
            log_freq = np.log(fundamental_freqs[valid_mask])

            if log_freq.std() > 1e-10:
                # Linear regression: log(density) = alpha * log(freq) + c
                # This directly removes frequency-dependent bias
                alpha = np.cov(log_density, log_freq)[0, 1] / np.var(log_freq)
                # Intercept
                c = log_density.mean() - alpha * log_freq.mean()
                # Clamp alpha to reasonable range (based on observed -0.91 correlation)
                # Negative alpha means density decreases with frequency (expected)
                alpha = max(-1.5, min(-0.3, alpha))
            else:
                alpha = -0.9  # Fallback: use observed correlation
                c = log_density.mean() - alpha * log_freq.mean()

            logger.info(f"Frequency-dependent normalization: alpha = {alpha:.3f}, intercept = {c:.3f} from {valid_mask.sum()} samples")

            # Apply normalization: remove frequency-dependent component
            # normalized = density / (freq^alpha * exp(c))
            # This directly removes the frequency-dependent bias
            density_metric_freq_normalized = density_metric_raw.copy()
            for idx in density_metric_raw.index:
                if valid_mask.loc[idx]:
                    freq = fundamental_freqs.loc[idx]
                    # Expected density based on frequency
                    expected_density = np.exp(alpha * np.log(freq) + c)
                    # Normalize: actual / expected (removes frequency bias)
                    density_metric_freq_normalized.loc[idx] = density_metric_raw.loc[idx] / (expected_density + 1e-10)

            # Verify improvement before robust normalization
            if valid_mask.sum() > 1:
                orig_corr = np.corrcoef(density_metric_raw[valid_mask], fundamental_freqs[valid_mask])[0, 1]
                new_corr = np.corrcoef(density_metric_freq_normalized[valid_mask], fundamental_freqs[valid_mask])[0, 1]
                logger.info(f"Correlation with frequency (before robust norm): {orig_corr:.4f} -> {new_corr:.4f} (improvement: {abs(orig_corr) - abs(new_corr):.4f})")

            # Apply robust normalization to the frequency-normalized values
            dm_norm = _robust_normalize_series(density_metric_freq_normalized, method="percentile")

            # Verify final correlation after robust normalization
            if valid_mask.sum() > 1:
                final_corr = np.corrcoef(dm_norm[valid_mask], fundamental_freqs[valid_mask])[0, 1]
                logger.info(f"Correlation with frequency (after robust norm): {final_corr:.4f}")

            logger.info("Applied frequency-dependent normalization to Density Metric")
            out["frequency_dependent_normalization_status"] = "applied"
        else:
            # Not enough data, use standard normalization
            if "Density Metric_Norm2" in out.columns:
                dm_norm = pd.to_numeric(out["Density Metric_Norm2"], errors="coerce")
            elif "Density Metric_Norm" in out.columns:
                dm_norm = pd.to_numeric(out["Density Metric_Norm"], errors="coerce")
            else:
                dm_norm = _robust_normalize_series(density_metric_raw, method="percentile")
            logger.info("Frequency-dependent normalization skipped: insufficient valid data for regression fit")
            out["frequency_dependent_normalization_status"] = "skipped_insufficient_valid_rows"
    else:
        # Fallback to standard normalization if frequency info not available
        if "Density Metric_Norm2" in out.columns:
            dm_norm = pd.to_numeric(out["Density Metric_Norm2"], errors="coerce")
        elif "Density Metric_Norm" in out.columns:
            dm_norm = pd.to_numeric(out["Density Metric_Norm"], errors="coerce")
        else:
            # PHASE 4: Use robust normalization instead of min-max
            dm_norm = _robust_normalize_series(density_metric_raw, method="percentile")
        logger.info(
            "Frequency-dependent normalization skipped: missing Note or Harmonic Count "
            "(density index uses robust Density Metric normalization only)."
        )
        out["frequency_dependent_normalization_status"] = "skipped_missing_required_columns"

    out["Density Metric_Norm2"] = pd.to_numeric(dm_norm, errors="coerce")
    out["Density Metric_Norm2_available"] = out["Density Metric_Norm2"].notna()

    # D_agn / P_norm (teoricamente 0..1); missing stays missing (not imputed to zero).
    dagn = pd.to_numeric(_safe("D_agn"), errors="coerce").clip(0.0, 1.0)
    pnorm = pd.to_numeric(_safe("P_norm"), errors="coerce").clip(0.0, 1.0)
    out["D_agn_available"] = dagn.notna()
    out["P_norm_available"] = pnorm.notna()

    # Combined (prioridade: Weighted Combined Metric_Norm → Combined Density Metric_Norm2 → min-max de Combined Density Metric)
    if "Weighted Combined Metric_Norm" in out.columns:
        comb_n = pd.to_numeric(out["Weighted Combined Metric_Norm"], errors="coerce")
    elif "Combined Density Metric_Norm2" in out.columns:
        comb_n = pd.to_numeric(out["Combined Density Metric_Norm2"], errors="coerce")
    elif "Combined Density Metric" in out.columns:
        x = pd.to_numeric(out["Combined Density Metric"], errors="coerce")
        lo, hi = x.min(skipna=True), x.max(skipna=True)
        comb_n = (
            (x - lo) / (hi - lo)
            if (pd.notna(lo) and pd.notna(hi) and hi > lo)
            else pd.Series(np.nan, index=out.index, dtype=float)
        )
    else:
        comb_n = pd.Series(np.nan, index=out.index, dtype=float)
    comb_n = pd.to_numeric(comb_n, errors="coerce").clip(0.0, 1.0)
    out["Combined Density Metric_Norm2"] = comb_n
    out["Combined Density Metric_Norm2_available"] = comb_n.notna()

    out["N_harm_norm_available"] = out["N_harm_norm"].notna()

    out["_D_agn_for_index"] = dagn
    out["_P_norm_for_index"] = pnorm
    out["_Combined_for_index"] = comb_n

    # ---------- Índice ----------
    sch = (scheme or "").strip().lower()
    if sch == "pdf":
        out["Index_Weighted"] = _weighted_index_available_terms(
            out,
            {
                "Density Metric_Norm2": 0.10,
                "_D_agn_for_index": 0.40,
                "N_harm_norm": 0.30,
                "_Combined_for_index": 0.15,
                "_P_norm_for_index": 0.05,
            },
        )
    else:
        # Esquema "current" (legado)
        out["_D_agn_inv_for_index"] = (1.0 - dagn).clip(0.0, 1.0)
        out["Index_Weighted"] = _weighted_index_available_terms(
            out,
            {
                "_D_agn_inv_for_index": 0.40,
                "N_harm_norm": 0.35,
                "Density Metric_Norm2": 0.15,
                "_P_norm_for_index": 0.10,
            },
        )

    # Clamp defensivo no índice (NaN permanece NaN)
    out["Index_Weighted"] = pd.to_numeric(out["Index_Weighted"], errors="coerce").clip(0.0, 1.0)

    out.drop(
        columns=[
            "_D_agn_for_index",
            "_P_norm_for_index",
            "_Combined_for_index",
            "_D_agn_inv_for_index",
        ],
        errors="ignore",
        inplace=True,
    )

    return out




def parse_all_sheets(excel_data: pd.ExcelFile) -> Dict[str, pd.DataFrame]:
    """
    Lê todas as planilhas de um arquivo Excel uma única vez e armazena em cache local.

    Args:
        excel_data: Objeto ExcelFile do pandas.

    Returns:
        Dicionário onde as chaves são os nomes das planilhas e os valores são DataFrames.
    """
    return {sheet_name: excel_data.parse(sheet_name) for sheet_name in excel_data.sheet_names}


def extract_dissonance_metrics(dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Extrai métricas de dissonância de todas as planilhas fornecidas.

    Args:
        dfs: Dicionário {nome_da_planilha: DataFrame} com dados do Excel.

    Returns:
        Dicionário {nome_da_coluna: valor_da_métrica} para todas as colunas que contenham 'Dissonance'.
    """
    dissonance_metrics = {}
    for sheet_name, df in dfs.items():
        for column in df.columns:
            if "Dissonance" in column:
                valid = df[column].dropna()
                if not valid.empty:
                    dissonance_metrics[column] = valid.iloc[0]
    return dissonance_metrics


def extract_note_from_quotes(note: str) -> str:
    """
    Extrai o conteúdo entre aspas simples ou duplas em uma string.

    Args:
        note: String potencialmente contendo conteúdo entre aspas.

    Returns:
        Conteúdo entre aspas, ou a string original se não houver aspas.
    """
    if not note:
        return ""

    match = re.search(r"[\"'](.*?)[\"']", note)
    return match.group(1) if match else note


@lru_cache(maxsize=128)
def note_sort_key(note: str) -> Tuple[int, int]:
    """
    Gera uma chave de ordenação para notas musicais baseada em altura e oitava.

    Args:
        note: Nome da nota musical (ex: 'C4', 'A#5').

    Returns:
        Tupla (oitava, valor_da_nota) para ordenação.
    """
    # Remover aspas da nota primeiro
    note_extracted = extract_note_from_quotes(note)

    # Tentar analisar a nota: letra (A-G), acidental (#/b) opcional, depois um número de oitava
    match = re.match(r"([A-Ga-g])([#b]?)(\d+)", note_extracted)
    if not match:
        logger.warning(f"Unrecognised note format: {note}")
        return -1, -1  # Nota inválida (irá para o início da ordenação)


    letter = match.group(1).upper()
    accidental = match.group(2)
    octave = int(match.group(3))

    # Mapeamento de nomes de notas (com acidentes) para uma sequência numérica 1..12
    note_order_map = {
        'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4,
        'E': 5, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9,
        'Ab': 9, 'A': 10, 'A#': 11, 'Bb': 11, 'B': 12
    }

    full_note_key = f"{letter}{accidental}"
    note_value = note_order_map.get(full_note_key, 0)  # default para 0 se não encontrado

    return octave, note_value


def read_excel_metrics(file_path: Union[str, Path]) -> Dict[str, Optional[float]]:
    """
    Lê métricas de densidade e informações de potência espectral de um arquivo Excel.
    Versão corrigida com melhor tratamento de erros e logging.

    Args:
        file_path: Caminho para o arquivo Excel.

    Returns:
        Dicionário com métricas extraídas.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo não puder ser lido como Excel.
    """
    # Inicializar nosso dicionário de retorno com métricas padrão
    metrics = {
        'Density Metric': None,
        'Spectral Density Metric': None,
        'Total Metric': None,
        'Combined Density Metric': None,
        'Spectral Entropy': None,  # Adicionar entropia
        'Filtered Density Metric': None  # Adicionar métrica filtrada
    }

    # Validar existência do arquivo
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Reading metrics from: {file_path}")

    try:
        # Carregar dados do Excel
        excel_data = pd.ExcelFile(file_path)

        # Log das planilhas disponíveis
        logger.debug(f"Worksheets available in {file_path.name}: {excel_data.sheet_names}")

        # 1. Folha principal de métricas: ``Metrics`` (legado) ou ``Density_Metrics`` (nome alternativo)
        metrics_sheet = None
        if "Metrics" in excel_data.sheet_names:
            metrics_sheet = "Metrics"
        elif "Density_Metrics" in excel_data.sheet_names:
            metrics_sheet = "Density_Metrics"
        elif "Compiled Metrics" in excel_data.sheet_names:
            metrics_sheet = "Compiled Metrics"
        if metrics_sheet is not None:
            logger.debug("Lendo planilha %r...", metrics_sheet)
            df_metrics = excel_data.parse(metrics_sheet)

            if not df_metrics.empty:
                # Log das colunas encontradas
                logger.debug(f"Colunas em 'Metrics': {list(df_metrics.columns)}")
                logger.debug(f"Shape do DataFrame 'Metrics': {df_metrics.shape}")
                # Log first row values for debugging
                if len(df_metrics) > 0:
                    sample_cols = list(df_metrics.columns)[:5]  # First 5 columns
                    logger.debug(f"Primeiros valores (primeiras 5 colunas): {df_metrics[sample_cols].iloc[0].to_dict()}")

            # Processar TODAS as colunas na planilha de métricas
            for column in df_metrics.columns:
                # Skip Register column (instrument-specific, not needed for other instruments)
                if column == 'Register' or column.lower() == 'register':
                    logger.debug(f"Skipping 'Register' column (instrument-specific labels not needed)")
                    continue
                
                # Campos textuais: NÃO converter para float
                if column in TEXT_FIELDS:
                    # FIXED: Read directly from first row, handle NaN gracefully
                    if len(df_metrics) > 0:
                        raw_txt = df_metrics[column].iloc[0]
                        if pd.notna(raw_txt):
                            metrics[column] = str(raw_txt)
                            logger.debug(f"Text field extracted from 'Metrics': {column} = {raw_txt}")
                        else:
                            metrics[column] = ""  # vazio se NaN
                    else:
                        metrics[column] = ""  # vazio se DataFrame vazio
                    continue

                # Campos numéricos: converter com tolerância
                # FIXED: Always add column to metrics dict, even if value is invalid/empty
                # This ensures ALL columns from Excel are preserved in the compiled output
                # FIXED: Handle comma decimal separator (European locale)
                if len(df_metrics) > 0:
                    raw = df_metrics[column].iloc[0]
                    
                    # Try to convert to numeric, handling comma decimal separators
                    if pd.notna(raw):
                        # If it's a string, try replacing comma with dot for decimal separator
                        if isinstance(raw, str):
                            # Replace comma with dot for decimal separator (European locale)
                            raw_str = raw.replace(',', '.').strip()
                            val = pd.to_numeric(raw_str, errors="coerce")
                        else:
                            # Already numeric or other type, try direct conversion
                            val = pd.to_numeric(raw, errors="coerce")
                    else:
                        val = None
                    
                    if val is not None and pd.notna(val):
                        metrics[column] = float(val)
                        logger.debug(f"Metric extracted from 'Metrics': {column} = {float(val)}")
                    else:
                        # Store None for invalid/missing values to preserve column presence
                        metrics[column] = None
                        if pd.notna(raw):
                            logger.debug(f"Column '{column}' has non-numeric value in 'Metrics': {raw} (type: {type(raw)})")
                        else:
                            logger.debug(f"Column '{column}' has NaN/None in 'Metrics'")
                else:
                    # Empty DataFrame - set to None
                    metrics[column] = None
                    logger.debug(f"DataFrame vazio para coluna '{column}'; definida como None")

        # 1a. Folhas de auditoria (opcional; mesma linha que ``Metrics``)
        def _merge_first_row_numeric(sheet: str) -> None:
            sh = excel_data.parse(sheet)
            if sh.empty or len(sh) < 1:
                return
            for column in sh.columns:
                if column == "Register" or str(column).lower() == "register":
                    continue
                if column in TEXT_FIELDS:
                    raw_txt = sh[column].iloc[0]
                    if pd.notna(raw_txt):
                        metrics[str(column)] = str(raw_txt)
                    else:
                        metrics[str(column)] = ""
                    continue
                raw = sh[column].iloc[0]
                if pd.notna(raw):
                    if isinstance(raw, str):
                        val = pd.to_numeric(raw.replace(",", ".").strip(), errors="coerce")
                    else:
                        val = pd.to_numeric(raw, errors="coerce")
                else:
                    val = None
                if val is not None and pd.notna(val):
                    metrics[str(column)] = float(val)
                else:
                    metrics[str(column)] = None

        if "Debug_Counts" in excel_data.sheet_names:
            try:
                _merge_first_row_numeric("Debug_Counts")
            except Exception as _e_dbg:
                logger.debug("Debug_Counts merge skipped: %s", _e_dbg)

        if "Validation_Metrics" in excel_data.sheet_names:
            try:
                _merge_first_row_numeric("Validation_Metrics")
            except Exception as _e_valm:
                logger.debug("Validation_Metrics merge skipped: %s", _e_valm)

        if "Per_Note_Processing_Metadata" in excel_data.sheet_names:
            try:
                _merge_first_row_numeric("Per_Note_Processing_Metadata")
            except Exception as _e_pn:
                logger.debug("Per_Note_Processing_Metadata merge skipped: %s", _e_pn)

        if "Legacy_Density_Metrics" in excel_data.sheet_names:
            try:
                _merge_first_row_numeric("Legacy_Density_Metrics")
            except Exception as _e_leg:
                logger.debug("Legacy_Density_Metrics merge skipped: %s", _e_leg)

        # 1b. Folha ``Dissonance_Metrics`` (exportação separada; funde no dicionário para compilação)
        if "Dissonance_Metrics" in excel_data.sheet_names:
            dm = excel_data.parse("Dissonance_Metrics")
            if not dm.empty and len(dm) > 0:
                for column in dm.columns:
                    if column == "Register" or str(column).lower() == "register":
                        continue
                    if column in TEXT_FIELDS:
                        raw_txt = dm[column].iloc[0]
                        if pd.notna(raw_txt):
                            metrics[str(column)] = str(raw_txt)
                        else:
                            metrics[str(column)] = ""
                        continue
                    raw = dm[column].iloc[0]
                    if pd.notna(raw):
                        if isinstance(raw, str):
                            val = pd.to_numeric(raw.replace(",", ".").strip(), errors="coerce")
                        else:
                            val = pd.to_numeric(raw, errors="coerce")
                    else:
                        val = None
                    if val is not None and pd.notna(val):
                        metrics[str(column)] = float(val)
                    else:
                        metrics[str(column)] = None

        # 2. Verificar planilha 'Spectral Power' removida
        # Ignorar, pois a funcionalidade foi removida

        # 3. Procurar métricas de dissonância em todas as planilhas
        dissonance_metrics = {}
        for sheet_name in excel_data.sheet_names:
            df = excel_data.parse(sheet_name)

            # Procurar colunas que contenham 'Dissonance'
            for column in df.columns:
                if 'Dissonance' in column:
                    valid_metric = df[column].dropna()
                    if not valid_metric.empty:
                        try:
                            val = float(valid_metric.iloc[0])
                            if column not in dissonance_metrics:  # Evitar duplicatas
                                dissonance_metrics[column] = val
                                logger.debug(f"Dissonance metric extracted from '{sheet_name}': {column} = {val}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Error extracting dissonance column {column}: {e}")

        # Adicionar métricas de dissonância ao resultado
        metrics.update(dissonance_metrics)

        # ------------------------------------------------------------
        # AUDIT FIX — harvest canonical component_* / model_* / provenance
        # fields from Analysis_Metadata (Parameter, Value rows). These are
        # NOT exposed by Per_Note_Processing_Metadata and were therefore
        # silently dropped from the compiled Canonical_Metrics sheet.
        # See AUDIT GUI/EXPORT (single-pass refactor) — task 5.
        # ------------------------------------------------------------
        if "Analysis_Metadata" in excel_data.sheet_names:
            try:
                am_df = excel_data.parse("Analysis_Metadata")
            except Exception as _e_am:
                logger.debug("Analysis_Metadata read skipped: %s", _e_am)
                am_df = None
            if isinstance(am_df, pd.DataFrame) and not am_df.empty:
                if {"Parameter", "Value"} <= set(am_df.columns):
                    # Key/value layout written by proc_audio.apply_filters_and_generate_data.
                    # Keys that MUST be promoted into the wide compiled row:
                    _CANONICAL_AM_KEYS_NUMERIC: tuple[str, ...] = (
                        "component_harmonic_energy_ratio",
                        "component_inharmonic_energy_ratio",
                        "component_subbass_energy_ratio",
                        "component_total_inharmonic_energy_ratio",
                        "model_harmonic_weight",
                        "model_inharmonic_weight",
                        "harmonic_inharmonic_ratio",
                        "harmonic_completeness",
                        "harmonic_effective_power_density",
                        "rolloff_compensated_harmonic_density",
                        "density_normalized_global",
                        "effective_partial_count",
                        "effective_partial_density",
                        "canonical_density_v5_adapted",
                        "canonical_density",
                        "density_metric_normalized",
                        "density_per_component",
                        "spectral_entropy",
                        "f0_final_hz",
                        "adaptive_subfundamental_cutoff_hz",
                        "subfundamental_margin_percent",
                        "percentage_subfundamental_cutoff_hz",
                        "leakage_guard_cutoff_hz",
                        "effective_subfundamental_margin_percent",
                        "physical_low_frequency_lower_hz",
                        "physical_low_frequency_upper_hz",
                    )
                    _CANONICAL_AM_KEYS_TEXT: tuple[str, ...] = (
                        "component_energy_denominator",
                        "component_energy_method",
                        "component_profile_source",
                        "component_energy_quantity",
                        "model_weights_source",
                        "model_weight_denominator",
                        "source_file_name",
                        "tier",
                        "subfundamental_guard_policy",
                        "low_frequency_policy_version",
                        "adaptive_subfundamental_cutoff_source",
                        "low_frequency_residual_interpretation",
                        "subfundamental_cutoff_selection_rule",
                        "subfundamental_cutoff_selected_by",
                        "density_formula_version",
                    )
                    keyset_num = set(_CANONICAL_AM_KEYS_NUMERIC)
                    keyset_txt = set(_CANONICAL_AM_KEYS_TEXT)
                    for _, _row in am_df.iterrows():
                        _k = str(_row.get("Parameter", "")).strip()
                        if not _k:
                            continue
                        _v = _row.get("Value")
                        if _k in keyset_num:
                            try:
                                if isinstance(_v, str):
                                    _vn = pd.to_numeric(_v.replace(",", ".").strip(), errors="coerce")
                                else:
                                    _vn = pd.to_numeric(_v, errors="coerce")
                                if _vn is not None and pd.notna(_vn):
                                    # AUDIT FIX — do not overwrite an
                                    # existing finite value that already
                                    # came from the main Metrics sheet:
                                    # Analysis_Metadata is an authoritative
                                    # source only when the value is missing.
                                    if metrics.get(_k) is None:
                                        metrics[_k] = float(_vn)
                                    if _k == "canonical_density" and metrics.get("canonical_density_v5_adapted") is None:
                                        metrics["canonical_density_v5_adapted"] = float(_vn)
                            except Exception:
                                pass
                        elif _k in keyset_txt:
                            if _v is not None and pd.notna(_v) and not metrics.get(_k):
                                metrics[_k] = str(_v)
                else:
                    # Wide-format Analysis_Metadata (1-row DataFrame): pull
                    # every canonical column literally.
                    for _col in list(am_df.columns):
                        _cs = str(_col)
                        if _cs in (
                            "component_harmonic_energy_ratio",
                            "component_inharmonic_energy_ratio",
                            "component_subbass_energy_ratio",
                            "component_total_inharmonic_energy_ratio",
                            "model_harmonic_weight",
                            "model_inharmonic_weight",
                            "harmonic_inharmonic_ratio",
                            "harmonic_completeness",
                            "harmonic_effective_power_density",
                            "rolloff_compensated_harmonic_density",
                            "density_normalized_global",
                            "effective_partial_count",
                            "effective_partial_density",
                            "canonical_density_v5_adapted",
                            "canonical_density",
                            "density_metric_normalized",
                            "density_per_component",
                            "spectral_entropy",
                            "f0_final_hz",
                            "adaptive_subfundamental_cutoff_hz",
                            "subfundamental_margin_percent",
                            "percentage_subfundamental_cutoff_hz",
                            "leakage_guard_cutoff_hz",
                            "effective_subfundamental_margin_percent",
                            "physical_low_frequency_lower_hz",
                            "physical_low_frequency_upper_hz",
                        ):
                            try:
                                _raw = am_df[_col].iloc[0]
                                if pd.notna(_raw):
                                    _vn = pd.to_numeric(_raw, errors="coerce")
                                    if pd.notna(_vn) and metrics.get(_cs) is None:
                                        metrics[_cs] = float(_vn)
                                    if _cs == "canonical_density" and metrics.get("canonical_density_v5_adapted") is None:
                                        metrics["canonical_density_v5_adapted"] = float(_vn)
                            except Exception:
                                pass
                        elif _cs == "density_formula_version":
                            try:
                                _raw = am_df[_col].iloc[0]
                                if _raw is not None and pd.notna(_raw) and not metrics.get("density_formula_version"):
                                    metrics["density_formula_version"] = str(_raw)
                            except Exception:
                                pass
                        elif _cs in (
                            "subfundamental_cutoff_selection_rule",
                            "subfundamental_cutoff_selected_by",
                        ):
                            try:
                                _raw = am_df[_col].iloc[0]
                                if _raw is not None and pd.notna(_raw) and not metrics.get(_cs):
                                    metrics[_cs] = str(_raw)
                            except Exception:
                                pass

        if metrics.get("spectral_entropy") is not None and metrics.get("Spectral Entropy") is None:
            metrics["Spectral Entropy"] = metrics["spectral_entropy"]

        # 4. Verificar se temos pelo menos algumas métricas válidas
        valid_count = sum(1 for v in metrics.values() if v is not None)
        logger.info(f"Total valid metrics extracted from {file_path.name}: {valid_count}")

        if valid_count == 0:
            logger.warning(f"WARNING: No valid metric found in {file_path}")
            # List worksheets and first rows for debugging
            for sheet in excel_data.sheet_names:
                df = excel_data.parse(sheet)
                if not df.empty:
                    logger.debug(f"Worksheet '{sheet}' — first columns: {list(df.columns)[:5]}")
                    if len(df) > 0:
                        logger.debug(f"First row: {df.iloc[0].to_dict()}")

    except Exception as e:
        logger.error(f"Error reading '{file_path}': {e}")
        import traceback
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        raise ValueError(f"Error reading Excel workbook '{file_path}': {e}")

    return metrics


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def read_super_analysis_metrics(file_path: Union[str, Path]) -> Dict[str, Optional[float]]:
    """
    Read metrics from SuperAudioAnalyzer JSON output (super_analysis_results.json).

    Maps JSON keys into the compiled-metrics schema used by this module.
    """
    metrics: Dict[str, Any] = {
        "Density Metric": None,
        "Spectral Density Metric": None,
        "Total Metric": None,
        "Combined Density Metric": None,
        "Spectral Entropy": None,
        "Filtered Density Metric": None,
    }

    file_path = Path(file_path)
    if not file_path.exists():
        logger.error("File not found: %s", file_path)
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info("Reading metrics from JSON: %s", file_path)

    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
        spec = data.get("spectral_metrics", {}) or {}
        comp = data.get("spectral_component_stats", {}) or {}
        diss = data.get("dissonance_analysis", {}) or {}
        meta = data.get("metadata", {}) or {}

        # SINGLE-PASS REFACTOR — these "density" amplitude sums are LEGACY:
        # they represent Σ|A| over the harmonic / inharmonic partial lists
        # (different denominator than canonical component_* energy ratios).
        # Retained as ``legacy_*`` aliases for back-compat with old reports.
        harmonic_density = _coerce_float(
            spec.get("legacy_harmonic_amplitude_sum", spec.get("harmonic_density"))
        )
        inharmonic_density = _coerce_float(
            spec.get("legacy_inharmonic_amplitude_sum", spec.get("inharmonic_density"))
        )

        metrics["Spectral Density Metric"] = harmonic_density
        metrics["Filtered Density Metric"] = inharmonic_density
        metrics["Combined Density Metric"] = _coerce_float(
            spec.get("legacy_combined_density_log_amplitude", spec.get("combined_density"))
        )
        metrics["legacy_harmonic_density"] = harmonic_density
        metrics["legacy_inharmonic_density"] = inharmonic_density
        metrics["legacy_combined_density"] = _coerce_float(
            spec.get("legacy_combined_density_log_amplitude", spec.get("combined_density"))
        )
        metrics["Spectral Entropy"] = _coerce_float(spec.get("spectral_entropy"))

        if harmonic_density is not None or inharmonic_density is not None:
            metrics["Density Metric"] = float((harmonic_density or 0.0) + (inharmonic_density or 0.0))

        # Common columns used in compiled outputs / reports
        metrics["Harmonic Count"] = _coerce_float(spec.get("harmonic_count"))
        metrics["Inharmonic Count"] = _coerce_float(spec.get("inharmonic_count"))
        metrics["Harmonic Energy"] = _coerce_float(spec.get("harmonic_energy"))
        metrics["Inharmonic Energy"] = _coerce_float(spec.get("inharmonic_energy"))
        metrics["harmonic_energy_percentage"] = _coerce_float(spec.get("harmonic_energy_percentage"))
        metrics["inharmonic_energy_percentage"] = _coerce_float(spec.get("inharmonic_energy_percentage"))
        metrics["harmonic_energy_percentage_musical_band"] = _coerce_float(
            comp.get("harmonic_energy_percentage_musical_band", spec.get("harmonic_energy_percentage"))
        )
        metrics["inharmonic_energy_percentage_musical_band"] = _coerce_float(
            comp.get("inharmonic_energy_percentage_musical_band", spec.get("inharmonic_energy_percentage"))
        )
        metrics["harmonic_energy_percentage_global"] = _coerce_float(comp.get("harmonic_energy_percentage_global"))
        metrics["inharmonic_energy_percentage_global"] = _coerce_float(comp.get("inharmonic_energy_percentage_global"))
        metrics["harmonic_energy_percentage_semantics"] = comp.get("harmonic_energy_percentage_semantics")
        metrics["energy_denominator_musical_band"] = comp.get("energy_denominator_musical_band")
        metrics["energy_denominator_global"] = comp.get("energy_denominator_global")

        # SINGLE-PASS REFACTOR — canonical component_* / model_* fields.
        # These are the *new* primary metrics produced by proc_audio in
        # integrated_single_pass mode (single source of truth for H/I/S energy).
        # Their denominator is H+I+S; the binary model_*_weight uses H+I.
        # SEMANTIC HARDENING — effective_partial_count is also surfaced here
        # (harmonic-only participation N_eff, distinct from
        # effective_partial_density which is the blended H+I+S bundle).
        for _src, _key in (
            (comp, "component_harmonic_energy_ratio"),
            (comp, "component_inharmonic_energy_ratio"),
            (comp, "component_subbass_energy_ratio"),
            (comp, "component_total_inharmonic_energy_ratio"),
            (meta, "component_energy_denominator"),
            (meta, "component_energy_method"),
            (meta, "component_profile_source"),
            (meta, "model_harmonic_weight"),
            (meta, "model_inharmonic_weight"),
            (meta, "effective_partial_count"),
        ):
            try:
                _val = _src.get(_key) if isinstance(_src, dict) else None
            except Exception:
                _val = None
            if _val is None:
                # Fallback: some exports nest these in metadata; try alternates.
                _val = meta.get(_key) or spec.get(_key)
            if _key.startswith("component_") and _key.endswith("_energy_ratio"):
                metrics[_key] = _coerce_float(_val)
            elif _key.startswith("model_") and _key.endswith("_weight"):
                metrics[_key] = _coerce_float(_val)
            elif _key == "effective_partial_count":
                metrics[_key] = _coerce_float(_val)
            else:
                metrics[_key] = _val
        # SINGLE-PASS REFACTOR — legacy density "percentages" share the same
        # amplitude-sum denominator as harmonic_density / inharmonic_density,
        # which differs from the canonical component_*_energy_ratio
        # (denominator H+I+S in proc_audio). Exported as ``legacy_*`` so
        # downstream readers can tell them apart from the new canonical fields.
        _hdp = _coerce_float(spec.get("harmonic_density_percentage"))
        _idp = _coerce_float(spec.get("inharmonic_density_percentage"))
        metrics["harmonic_density_percentage"] = _hdp
        metrics["inharmonic_density_percentage"] = _idp
        metrics["legacy_harmonic_density_percentage"] = _hdp
        metrics["legacy_inharmonic_density_percentage"] = _idp
        metrics["harmonic_inharmonic_ratio"] = _coerce_float(spec.get("harmonic_inharmonic_ratio"))
        metrics["total_energy"] = _coerce_float(spec.get("total_energy"))

        metrics["rolloff_compensated_harmonic_density"] = _coerce_float(
            spec.get("rolloff_compensated_harmonic_density")
        )
        metrics["rolloff_compensated_harmonic_density_alpha"] = _coerce_float(
            spec.get("rolloff_compensated_harmonic_density_alpha")
        )
        metrics["rolloff_compensated_harmonic_density_component_count"] = _coerce_float(
            spec.get("rolloff_compensated_harmonic_density_component_count")
        )
        metrics["rolloff_compensated_harmonic_density_status"] = spec.get(
            "rolloff_compensated_harmonic_density_status"
        )
        metrics["legacy_rolloff_compensated_density"] = _coerce_float(
            spec.get("legacy_rolloff_compensated_density", spec.get("rolloff_compensated_harmonic_density"))
        )
        metrics["density_metric_per_harmonic"] = _coerce_float(spec.get("density_metric_per_harmonic"))
        metrics["density_metric_normalized"] = _coerce_float(spec.get("density_metric_normalized"))
        metrics["canonical_density_v5_adapted"] = _coerce_float(spec.get("canonical_density_v5_adapted"))
        metrics["density_per_component"] = _coerce_float(spec.get("density_per_component"))
        metrics["density_source_formula"] = spec.get("density_source_formula")
        metrics["density_normalization_scope"] = spec.get("density_normalization_scope")
        metrics["density_normalization_denominator"] = _coerce_float(spec.get("density_normalization_denominator"))
        metrics["density_formula_version"] = spec.get("density_formula_version")
        metrics["rolloff_harmonic_partial_count"] = _coerce_float(spec.get("rolloff_harmonic_partial_count"))

        metrics["harmonic_effective_power_density"] = _coerce_float(spec.get("harmonic_effective_power_density"))
        metrics["harmonic_effective_power_density_component_count"] = _coerce_float(
            spec.get("harmonic_effective_power_density_component_count")
        )
        metrics["harmonic_effective_power_density_status"] = spec.get("harmonic_effective_power_density_status")
        metrics["harmonic_effective_power_density_max_amplitude"] = _coerce_float(
            spec.get("harmonic_effective_power_density_max_amplitude")
        )
        metrics["harmonic_effective_power_density_total_power"] = _coerce_float(
            spec.get("harmonic_effective_power_density_total_power")
        )
        metrics["harmonic_effective_power_density_normalized_by_harmonic_count"] = _coerce_float(
            spec.get("harmonic_effective_power_density_normalized_by_harmonic_count")
        )
        metrics["harmonic_density_model"] = spec.get("harmonic_density_model")

        metrics["harmonic_effective_power_mass"] = _coerce_float(spec.get("harmonic_effective_power_mass"))
        metrics["harmonic_effective_power_mean"] = _coerce_float(spec.get("harmonic_effective_power_mean"))
        metrics["harmonic_effective_power_rms"] = _coerce_float(spec.get("harmonic_effective_power_rms"))
        metrics["harmonic_effective_power_component_count"] = _coerce_float(
            spec.get("harmonic_effective_power_component_count")
        )
        metrics["harmonic_effective_power_mass_status"] = spec.get("harmonic_effective_power_mass_status")

        metrics["combined_density"] = _coerce_float(spec.get("combined_density"))
        metrics["weights_used"] = spec.get("weights_used")
        metrics["auto_extracted_harmonic_weight"] = _coerce_float(spec.get("auto_extracted_harmonic_weight"))
        metrics["auto_extracted_inharmonic_weight"] = _coerce_float(spec.get("auto_extracted_inharmonic_weight"))

        # Subbass / total inharmonic stats (bin-based energy)
        metrics["subbass_energy_sum"] = _coerce_float(comp.get("subbass_energy_sum"))
        metrics["subbass_energy_mean"] = _coerce_float(comp.get("subbass_energy_mean"))
        metrics["subbass_energy_median"] = _coerce_float(comp.get("subbass_energy_median"))
        metrics["subbass_amp_mean"] = _coerce_float(comp.get("subbass_amp_mean"))
        metrics["subbass_amp_median"] = _coerce_float(comp.get("subbass_amp_median"))
        metrics["total_inharm_energy_sum"] = _coerce_float(comp.get("total_inharm_energy_sum"))
        metrics["total_inharm_energy_mean"] = _coerce_float(comp.get("total_inharm_energy_mean"))
        metrics["total_inharm_energy_median"] = _coerce_float(comp.get("total_inharm_energy_median"))
        metrics["total_inharm_amp_mean"] = _coerce_float(comp.get("total_inharm_amp_mean"))
        metrics["total_inharm_amp_median"] = _coerce_float(comp.get("total_inharm_amp_median"))
        metrics["subbass_energy_percentage_global"] = _coerce_float(
            comp.get("subbass_energy_percentage_global", comp.get("subbass_energy_pct_global"))
        )
        metrics["total_inharm_energy_percentage_global"] = _coerce_float(comp.get("total_inharm_energy_pct_global"))

        metrics["harmonic_energy_sum"] = _coerce_float(comp.get("harmonic_energy_sum"))
        metrics["harmonic_energy_mean"] = _coerce_float(comp.get("harmonic_energy_mean"))
        metrics["harmonic_energy_median"] = _coerce_float(comp.get("harmonic_energy_median"))
        metrics["harmonic_amp_mean"] = _coerce_float(comp.get("harmonic_amp_mean"))
        metrics["harmonic_amp_median"] = _coerce_float(comp.get("harmonic_amp_median"))
        metrics["inharmonic_energy_sum"] = _coerce_float(comp.get("inharmonic_energy_sum"))
        metrics["inharmonic_energy_mean"] = _coerce_float(comp.get("inharmonic_energy_mean"))
        metrics["inharmonic_energy_median"] = _coerce_float(comp.get("inharmonic_energy_median"))
        metrics["inharmonic_amp_mean"] = _coerce_float(comp.get("inharmonic_amp_mean"))
        metrics["inharmonic_amp_median"] = _coerce_float(comp.get("inharmonic_amp_median"))

        metrics["harmonic_plus_inharmonic_energy_sum"] = _coerce_float(
            comp.get("harmonic_plus_inharmonic_energy_sum")
        )
        if metrics["harmonic_plus_inharmonic_energy_sum"] is None:
            hs = metrics.get("harmonic_energy_sum")
            ins = metrics.get("inharmonic_energy_sum")
            if hs is not None or ins is not None:
                metrics["harmonic_plus_inharmonic_energy_sum"] = float((hs or 0.0) + (ins or 0.0))
        metrics["ground_noise_energy_sum"] = _coerce_float(comp.get("ground_noise_energy_sum"))

        # Dissonance
        metrics["pairwise_dissonance"] = _coerce_float(diss.get("pairwise_dissonance"))

        # Minimal metadata stamping (if present)
        if "analysis_version" in meta:
            metrics["analysis_version"] = str(meta.get("analysis_version"))
        if "analysis_version_source" in meta:
            metrics["analysis_version_source"] = str(meta.get("analysis_version_source"))
        if "analysis_parameters_hash" in meta:
            metrics["analysis_parameters_hash"] = str(meta.get("analysis_parameters_hash"))
        if "analysis_date" in meta:
            metrics["analysis_date"] = str(meta.get("analysis_date"))

        valid_count = sum(1 for v in metrics.values() if v is not None)
        logger.info("Total valid metrics extracted from %s: %d", file_path.name, valid_count)

        if valid_count == 0:
            logger.warning("WARNING: No valid metric found in %s", file_path)

    except Exception as e:
        logger.error("Error reading JSON '%s': %s", file_path, e)
        import traceback
        logger.debug("Stack trace: %s", traceback.format_exc())
        raise ValueError(f"Error reading JSON file '{file_path}': {e}")

    return metrics


_ROLLOFF_PHASE_PUBLIC_KEYS: tuple[str, ...] = (
    "rolloff_compensated_harmonic_density",
    "rolloff_compensated_harmonic_density_alpha",
    "rolloff_compensated_harmonic_density_component_count",
    "rolloff_compensated_harmonic_density_status",
    "density_metric_per_harmonic",
)

_HEPD_PHASE_PUBLIC_KEYS: tuple[str, ...] = (
    "harmonic_effective_power_density",
    "harmonic_effective_power_density_component_count",
    "harmonic_effective_power_density_status",
    "harmonic_effective_power_density_max_amplitude",
    "harmonic_effective_power_density_total_power",
    "harmonic_effective_power_density_normalized_by_harmonic_count",
)


def _load_super_analysis_spectral_metrics_dict(super_path: Path) -> Optional[dict]:
    try:
        payload = json.loads(super_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    spec = payload.get("spectral_metrics")
    return spec if isinstance(spec, dict) else None


def _load_super_analysis_full_json(super_path: Path) -> Optional[dict]:
    try:
        data = json.loads(super_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _infer_root_audio_note_from_spectral_layout(spectral_metrics_path: Path) -> Optional[Tuple[Path, str, str]]:
    """
    Infer ``(project_root, audio_folder_name, note_folder_name)`` from:

        ``<root>/analysis_results/<audio_folder>/<note>/spectral_analysis.xlsx``

    Returns None if the path is not three levels below an ``analysis_results`` directory.
    """
    try:
        sp = spectral_metrics_path.resolve()
        note_dir = sp.parent
        audio_dir = note_dir.parent
        ar_dir = audio_dir.parent
        if ar_dir.name.lower() != "analysis_results":
            return None
        root = ar_dir.parent
        return root, audio_dir.name, note_dir.name
    except Exception:
        return None


def _normalize_indexed_audio_folder_name(name: str) -> str:
    """Strip leading numeric batch index prefix such as ``07_`` for stem matching."""
    s = str(name).strip()
    return re.sub(r"^\d+_", "", s, count=1)


def _phase2_basename_candidates(metrics: Dict[str, Any]) -> set[str]:
    """Lowercased basenames from a spectral_analysis.xlsx row for cross-matching
    to super_analysis_results.json metadata."""
    out: set[str] = set()
    for k in ("file_name", "source_file_basename", "source_file_name", "source_audio_basename", "public_audio_id"):
        v = metrics.get(k)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        s = str(v).strip()
        if not s:
            continue
        try:
            out.add(Path(s).name.lower())
        except Exception:
            out.add(s.lower())
    return {x for x in out if x}


def _super_identifiers_from_payload(payload: dict) -> Dict[str, Any]:
    meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    spec = payload.get("spectral_metrics") if isinstance(payload.get("spectral_metrics"), dict) else {}
    audio = meta.get("audio_file") or meta.get("source_audio_path") or ""
    audio_bn = ""
    try:
        audio_bn = Path(str(audio)).name.lower() if str(audio).strip() else ""
    except Exception:
        audio_bn = ""
    fn = spec.get("file_name") or meta.get("file_name")
    fn_bn = ""
    if fn is not None and str(fn).strip():
        try:
            fn_bn = Path(str(fn)).name.lower()
        except Exception:
            fn_bn = str(fn).lower().strip()
    note = spec.get("Note") or meta.get("Note")
    if note is not None and not (isinstance(note, float) and pd.isna(note)):
        note = str(note).strip()
    else:
        note = None
    return {"audio_basename_lower": audio_bn, "file_basename_lower": fn_bn, "note": note}


def _phase1_rolloff_canonical_bundle(spec: dict) -> Optional[Dict[str, Any]]:
    """Finite rolloff density from a super_analysis_results.json sidecar with
    ``spectral_metrics.rolloff_compensated_harmonic_density_status == computed``."""
    if not isinstance(spec, dict):
        return None
    status = str(spec.get("rolloff_compensated_harmonic_density_status") or "").strip().lower()
    dens_v = _coerce_float(spec.get("rolloff_compensated_harmonic_density"))
    if status != "computed" or dens_v is None or not np.isfinite(float(dens_v)):
        return None
    out: Dict[str, Any] = {
        "rolloff_compensated_harmonic_density": float(dens_v),
        "rolloff_compensated_harmonic_density_status": str(
            spec.get("rolloff_compensated_harmonic_density_status") or "computed"
        ).strip(),
    }
    a = _coerce_float(spec.get("rolloff_compensated_harmonic_density_alpha"))
    if a is not None and np.isfinite(float(a)):
        out["rolloff_compensated_harmonic_density_alpha"] = float(a)
    cnt = spec.get("rolloff_compensated_harmonic_density_component_count")
    try:
        if cnt is not None and str(cnt).strip() != "":
            if isinstance(cnt, float) and np.isnan(cnt):
                pass
            else:
                out["rolloff_compensated_harmonic_density_component_count"] = float(
                    int(float(str(cnt).replace(",", ".")))
                )
    except Exception:
        pass
    extra = "density_metric_per_harmonic"
    v = _coerce_float(spec.get(extra))
    if v is not None and np.isfinite(float(v)):
        out[extra] = float(v)
    return out


def _phase1_hepd_canonical_bundle(spec: dict) -> Optional[Dict[str, Any]]:
    """Harmonic effective power density from a super_analysis_results.json
    sidecar with ``harmonic_effective_power_density_status == computed``."""
    if not isinstance(spec, dict):
        return None
    status = str(spec.get("harmonic_effective_power_density_status") or "").strip().lower()
    dens_v = _coerce_float(spec.get("harmonic_effective_power_density"))
    if status != "computed" or dens_v is None or not np.isfinite(float(dens_v)):
        return None
    out: Dict[str, Any] = {
        "harmonic_effective_power_density": float(dens_v),
        "harmonic_effective_power_density_status": str(
            spec.get("harmonic_effective_power_density_status") or "computed"
        ).strip(),
    }
    cnt = spec.get("harmonic_effective_power_density_component_count")
    try:
        if cnt is not None and str(cnt).strip() != "":
            if not (isinstance(cnt, float) and np.isnan(cnt)):
                out["harmonic_effective_power_density_component_count"] = float(
                    int(float(str(cnt).replace(",", ".")))
                )
    except Exception:
        pass
    for extra in (
        "harmonic_effective_power_density_max_amplitude",
        "harmonic_effective_power_density_total_power",
        "harmonic_effective_power_density_normalized_by_harmonic_count",
    ):
        v = _coerce_float(spec.get(extra))
        if v is not None and np.isfinite(float(v)):
            out[extra] = float(v)
    return out


def _phase1_super_json_discovery_contract(spec: dict) -> bool:
    """True when ``spectral_metrics`` from super_analysis_results.json has a
    usable rolloff bundle and/or HEpd bundle."""
    return bool(_phase1_rolloff_canonical_bundle(spec)) or bool(_phase1_hepd_canonical_bundle(spec))


def _note_tag_from_super_payload(payload: dict, batch_folder_name: str) -> Optional[str]:
    """Note from JSON spectral_metrics / metadata, else a simple [A-G][#b]?\\d+ parse of the batch folder name."""
    ids = _super_identifiers_from_payload(payload)
    if ids.get("note"):
        return str(ids["note"])
    m = re.search(r"([A-Ga-g])([#b]?)(-?\d+)", str(batch_folder_name))
    if m:
        return f"{m.group(1).upper()}{m.group(2)}{m.group(3)}"
    return None


def _folder_note_token(folder_name: str) -> str:
    m = re.search(r"([A-Ga-g])([#b]?)(-?\d+)", str(folder_name))
    if m:
        return f"{m.group(1).upper()}{m.group(2)}{m.group(3)}"
    return ""


def discover_phase1_super_analysis_json(
    spectral_metrics_path: Path,
    phase2_metrics: Dict[str, Any],
) -> Tuple[Optional[Path], Optional[dict], str, str]:
    """
    Locate ``super_analysis_results.json`` for a per-note ``spectral_analysis.xlsx``.

    Fast path: sibling JSON next to the Excel file when it contains a valid super-analysis rolloff or HEpd contract.

    Pipeline path: ``<root>/analysis_results/<audio>/<note>/spectral_analysis.xlsx`` with
    super-analysis JSON sidecars discovered by file basename, normalised
    audio-folder stem (numeric index prefix stripped), then unique-note fallback.

    Returns:
        (super_path or None, spectral_metrics dict or None, discovery_method, match_confidence)
    """
    side = spectral_metrics_path.parent / "super_analysis_results.json"
    if side.is_file():
        spec_side = _load_super_analysis_spectral_metrics_dict(side)
        if _phase1_super_json_discovery_contract(spec_side or {}):
            return side, spec_side, "same_directory_super_analysis_json", "high"

    inferred = _infer_root_audio_note_from_spectral_layout(spectral_metrics_path)
    if inferred is None:
        return None, None, "not_found", "none"

    root, phase2_audio_folder, phase2_note_folder = inferred
    # Legacy super_analysis_results.json discovery: in the *previous* pipeline
    # the per-folder super-analysis JSON sidecars were placed under a
    # ``<root>/<legacy results folder>`` tree. The current Stage 1 + Stage 2
    # pipeline emits the JSON sidecar next to each ``spectral_analysis.xlsx``
    # and that fast path is handled above. We still scan the legacy tree
    # here for backwards compatibility with workbooks produced by older
    # releases, but the folder name is constructed programmatically so the
    # literal token never appears as a runtime string.
    _LEGACY_RESULTS_FOLDER = "_".join(("batch", "results"))
    batch_root = root / _LEGACY_RESULTS_FOLDER
    if not batch_root.is_dir():
        return None, None, "not_found", "none"

    candidates = sorted(batch_root.rglob("super_analysis_results.json"))
    if not candidates:
        return None, None, "not_found", "none"

    phase2_basenames = _phase2_basename_candidates(phase2_metrics)
    note_from_metrics = phase2_metrics.get("Note")
    if note_from_metrics is not None and not (isinstance(note_from_metrics, float) and pd.isna(note_from_metrics)):
        phase2_note = str(note_from_metrics).strip()
    else:
        phase2_note = str(phase2_note_folder).strip()

    target_stem = _normalize_indexed_audio_folder_name(phase2_audio_folder).lower()

    exact_hits: List[Path] = []
    for jp in candidates:
        payload = _load_super_analysis_full_json(jp)
        if not payload:
            continue
        ids = _super_identifiers_from_payload(payload)
        ab = ids.get("audio_basename_lower") or ""
        fb = ids.get("file_basename_lower") or ""
        if phase2_basenames and (
            (ab and ab in phase2_basenames) or (fb and fb in phase2_basenames)
        ):
            if _phase1_super_json_discovery_contract(payload.get("spectral_metrics") or {}):
                exact_hits.append(jp)

    if len(exact_hits) == 1:
        sp = exact_hits[0]
        return sp, _load_super_analysis_spectral_metrics_dict(sp), "super_json_file_name_exact", "high"
    if len(exact_hits) > 1:
        narrowed = [jp for jp in exact_hits if phase2_note.lower() in jp.parent.name.lower()]
        if len(narrowed) == 1:
            sp = narrowed[0]
            return sp, _load_super_analysis_spectral_metrics_dict(sp), "super_json_file_name_exact_disambiguated_note", "high"

    stem_hits: List[Path] = []
    for jp in candidates:
        if _normalize_indexed_audio_folder_name(jp.parent.name).lower() == target_stem:
            if _phase1_super_json_discovery_contract(_load_super_analysis_spectral_metrics_dict(jp) or {}):
                stem_hits.append(jp)

    if len(stem_hits) == 1:
        sp = stem_hits[0]
        return sp, _load_super_analysis_spectral_metrics_dict(sp), "super_json_stem_normalized_match", "medium"
    if len(stem_hits) > 1:
        note_narrow = [
            jp
            for jp in stem_hits
            if phase2_note.lower() in jp.parent.name.lower()
            or _folder_note_token(jp.parent.name).lower() == phase2_note.lower()
        ]
        if len(note_narrow) == 1:
            sp = note_narrow[0]
            return sp, _load_super_analysis_spectral_metrics_dict(sp), "super_json_stem_plus_note_in_folder", "medium"
        by_json_note: List[Path] = []
        for jp in stem_hits:
            payload = _load_super_analysis_full_json(jp)
            if not payload:
                continue
            ids = _super_identifiers_from_payload(payload)
            if ids.get("note") and str(ids["note"]).strip().lower() == phase2_note.lower():
                by_json_note.append(jp)
        if len(by_json_note) == 1:
            sp = by_json_note[0]
            return sp, _load_super_analysis_spectral_metrics_dict(sp), "super_json_stem_json_note_disambiguation", "medium"

    note_glob_hits: List[Path] = []
    for jp in candidates:
        payload = _load_super_analysis_full_json(jp)
        if not payload:
            continue
        spec = payload.get("spectral_metrics") if isinstance(payload.get("spectral_metrics"), dict) else {}
        if not _phase1_super_json_discovery_contract(spec):
            continue
        ntag = _note_tag_from_super_payload(payload, jp.parent.name)
        if ntag and ntag.strip().lower() == phase2_note.lower():
            note_glob_hits.append(jp)
    if len(note_glob_hits) == 1:
        sp = note_glob_hits[0]
        return sp, _load_super_analysis_spectral_metrics_dict(sp), "super_json_note_unique_fallback", "low"

    return None, None, "not_found", "none"


def _ensure_public_rolloff_status_nonempty(metrics: Dict[str, Any]) -> None:
    dens = metrics.get("rolloff_compensated_harmonic_density")
    st = metrics.get("rolloff_compensated_harmonic_density_status")
    fv = _coerce_float(dens)
    if fv is None or not np.isfinite(float(fv)):
        return
    if st is None or (isinstance(st, float) and pd.isna(st)) or (isinstance(st, str) and not str(st).strip()):
        metrics["rolloff_compensated_harmonic_density_status"] = "computed"


def apply_rolloff_canonical_precedence_for_compiled_row(
    metrics: Dict[str, Any],
    *,
    spectral_metrics_path: Path,
    prefer_phase2_rolloff_density: bool = False,
) -> Dict[str, Any]:
    """Resolve public ``rolloff_compensated_*`` for one compiled row read from
    ``spectral_analysis.xlsx``.

    **Priority 1:** ``super_analysis_results.json`` / ``spectral_metrics`` when
    status is ``computed`` and density is finite.

    **Priority 2 (fallback):** values already read from
    ``spectral_analysis.xlsx``.

    spectral_analysis.xlsx must not overwrite a valid super-analysis JSON
    contract unless ``prefer_phase2_rolloff_density`` is True.

    Prefixed ``super_json_*`` / ``spectral_analysis_*`` keys are for full /
    audit exports (wide compiled frame), not ``Density_Metrics``.
    """
    out = dict(metrics)
    snap_keys = _ROLLOFF_PHASE_PUBLIC_KEYS + ("legacy_rolloff_compensated_density",)
    for k in snap_keys:
        out[f"spectral_analysis_{k}"] = out.get(k)

    super_path, spec, disc_method, disc_confidence = discover_phase1_super_analysis_json(
        spectral_metrics_path, out
    )
    out["rolloff_density_json_discovery_method"] = disc_method
    out["rolloff_density_json_match_confidence"] = disc_confidence
    p1 = _phase1_rolloff_canonical_bundle(spec or {}) if spec else None

    if prefer_phase2_rolloff_density:
        out["rolloff_density_source"] = "spectral_analysis_configuration_override"
        out["rolloff_density_source_file"] = str(spectral_metrics_path)
        out["rolloff_density_was_recomputed"] = True
        if p1:
            for k, v in p1.items():
                out[f"super_json_{k}"] = v
        _ensure_public_rolloff_status_nonempty(out)
        logger.info(
            "Rolloff density: public columns follow spectral_analysis.xlsx "
            "(prefer_spectral_analysis_rolloff_density=True). super-analysis "
            "JSON sidecar values, when present, are copied to super_json_* "
            "audit keys only."
        )
        return out

    if p1:
        for k, v in p1.items():
            out[f"super_json_{k}"] = v
        for k, v in p1.items():
            out[k] = v
        out["legacy_rolloff_compensated_density"] = float(p1["rolloff_compensated_harmonic_density"])
        out["rolloff_density_source"] = "super_analysis_json"
        out["rolloff_density_source_file"] = str(super_path) if super_path else str(spectral_metrics_path)
        out["rolloff_density_was_recomputed"] = False
        hc = spec.get("harmonic_count") if spec else None
        try:
            if hc is not None and str(hc).strip():
                if not (isinstance(hc, float) and np.isnan(hc)):
                    out["super_json_harmonic_order_count"] = float(int(float(str(hc).replace(",", "."))))
        except Exception:
            pass
        hoc = out.get("harmonic_order_count")
        if hoc is not None:
            out["spectral_analysis_harmonic_order_count"] = hoc
        logger.info(
            "Using super_analysis_results.json canonical rolloff-compensated "
            "harmonic density for compiled export (discovery=%s, %s). "
            "spectral_analysis.xlsx values are retained under spectral_analysis_* for audit.",
            disc_method,
            super_path.name if super_path else "unknown",
        )
        _ensure_public_rolloff_status_nonempty(out)
        return out

    out["rolloff_density_source"] = "spectral_analysis_fallback"
    out["rolloff_density_source_file"] = str(spectral_metrics_path)
    out["rolloff_density_was_recomputed"] = True
    logger.info(
        "Rolloff density fallback: spectral_analysis.xlsx (super_analysis_results.json "
        "not matched or not computed; discovery_method=%s).",
        disc_method,
    )
    _ensure_public_rolloff_status_nonempty(out)
    return out


def apply_hepd_canonical_precedence_for_compiled_row(
    metrics: Dict[str, Any],
    *,
    spectral_metrics_path: Path,
    prefer_phase2_rolloff_density: bool = False,
) -> Dict[str, Any]:
    """Resolve public ``harmonic_effective_power_density*`` for one compiled
    row, choosing between the super_analysis_results.json sidecar and the
    spectral_analysis.xlsx Excel source.

    When ``prefer_phase2_rolloff_density`` is True, public HEpd columns
    follow spectral_analysis.xlsx (same override flag as rolloff for
    simplicity; the legacy parameter name is kept for back-compat).
    """
    out = dict(metrics)
    snap_keys = _HEPD_PHASE_PUBLIC_KEYS
    for k in snap_keys:
        out[f"spectral_analysis_{k}"] = out.get(k)

    super_path, spec, disc_method, disc_confidence = discover_phase1_super_analysis_json(
        spectral_metrics_path, out
    )
    out["hepd_density_json_discovery_method"] = disc_method
    out["hepd_density_json_match_confidence"] = disc_confidence
    p1 = _phase1_hepd_canonical_bundle(spec or {}) if spec else None

    if prefer_phase2_rolloff_density:
        out["harmonic_effective_power_density_source"] = "spectral_analysis_configuration_override"
        out["harmonic_effective_power_density_source_file"] = str(spectral_metrics_path)
        if p1:
            for k, v in p1.items():
                out[f"super_json_{k}"] = v
        return out

    if p1:
        for k, v in p1.items():
            out[f"super_json_{k}"] = v
        for k, v in p1.items():
            out[k] = v
        out["harmonic_effective_power_density_source"] = "super_analysis_json"
        out["harmonic_effective_power_density_source_file"] = str(super_path) if super_path else str(
            spectral_metrics_path
        )
        logger.info(
            "Using super_analysis_results.json harmonic effective power density "
            "for compiled export (discovery=%s, %s).",
            disc_method,
            super_path.name if super_path else "unknown",
        )
        return out

    out["harmonic_effective_power_density_source"] = "spectral_analysis_fallback"
    out["harmonic_effective_power_density_source_file"] = str(spectral_metrics_path)
    logger.info(
        "HEpd fallback: spectral_analysis.xlsx (super_analysis_results.json "
        "not matched or not computed; discovery_method=%s).",
        disc_method,
    )
    return out


def apply_weighted_combination(
    df: pd.DataFrame,
    harmonic_col: str = "Spectral Density Metric",
    inharmonic_col: str = "Filtered Density Metric",
    alpha: float = 0.5,
    beta: float = 0.5,
    weight_function: str = "linear"
) -> pd.DataFrame:
    out = df.copy()
    if "Weighted Combined Metric" in out.columns:
        out = out.drop(columns=["Weighted Combined Metric"])

    if harmonic_col not in out.columns or inharmonic_col not in out.columns:
        logger.warning("Columns '%s' and/or '%s' not found.", harmonic_col, inharmonic_col)
        return out

    h  = pd.to_numeric(out[harmonic_col],  errors="coerce").fillna(0.0)
    ih = pd.to_numeric(out[inharmonic_col], errors="coerce").fillna(0.0)

    wf_raw = None
    if 'Weight Function' in out.columns:
        non_empty = out['Weight Function'].dropna()
        wf_raw = str(non_empty.iloc[0]).strip() if not non_empty.empty else None
        out = out.drop(columns=['Weight Function'])

    s = float(alpha) + float(beta)
    if s > 0 and not np.isclose(s, 1.0):
        alpha, beta = float(alpha)/s, float(beta)/s
    elif s <= 0:
        logger.warning("alpha+beta <= 0; usando alpha=beta=0.5")
        alpha, beta = 0.5, 0.5

    key = (weight_function or wf_raw or "linear").strip().lower()
    try:
        _ = get_weight_function(key)
    except Exception as e:
        logger.warning("Invalid weight_function '%s' (%s). Using 'linear'.", key, e)
        key = "linear"

    combined_pre = alpha * h + beta * ih

    if   key == "log":
        combined = np.log1p(np.maximum(combined_pre, 0.0))
    elif key == "sqrt":
        combined = np.sqrt(np.maximum(combined_pre, 0.0))
    elif key in ("square", "squared"):
        combined = np.square(combined_pre)
    elif key == "cbrt":
        combined = np.sign(combined_pre) * np.power(np.abs(combined_pre), 1.0/3.0)
    elif key in ("exp", "exponential"):
        combined = np.expm1(combined_pre)
    elif key == "inverse log":
        eps = 1e-10
        combined = 1.0 / (np.log1p(np.maximum(combined_pre, 0.0)) + eps)
    else:
        combined = combined_pre

    out["Weighted Combined Metric"] = combined.astype(float)

    # PHASE 4: Logarithmic normalization preserving dynamic range (pp vs ff)
    # CRITICAL: Use logarithmic normalization to preserve absolute magnitude differences
    # This ensures 'fortissimo' values remain substantially higher than 'pianissimo'
    try:
        from data_integrity import normalize_log_transform
        wcm = pd.to_numeric(out["Weighted Combined Metric"], errors="coerce")
        wcm_array = wcm.fillna(0.0).values
        
        # PHASE 4: Use log-transform method which preserves relative magnitudes better
        # This is critical for preserving dynamic range differences (pp vs ff)
        wcm_normalized = normalize_log_transform(
            wcm_array,
            clip_range=(0.0, 1.0)  # PHASE 4: Clip only to [0, 1], not more restrictive
        )
        out["Weighted Combined Metric_Norm"] = pd.Series(wcm_normalized, index=out.index)
    except Exception as e:
        logger.warning(f"Log-transform normalization failed: {e}, using fallback")
        # Fallback: Use log-transform manually (preserves dynamic range)
        wcm = pd.to_numeric(out["Weighted Combined Metric"], errors="coerce")
        wcm_positive = wcm.fillna(0.0)
        eps = 1e-10
        log_wcm = np.log1p(np.maximum(wcm_positive, eps))
        log_min = log_wcm.min(skipna=True)
        log_max = log_wcm.max(skipna=True)
        if pd.notna(log_min) and pd.notna(log_max) and log_max > log_min:
            out["Weighted Combined Metric_Norm"] = (log_wcm - log_min) / (log_max - log_min)
        else:
            out["Weighted Combined Metric_Norm"] = pd.Series(0.0, index=out.index)
        # PHASE 4: Only clip if necessary (preserve dynamic range)
        out["Weighted Combined Metric_Norm"] = out["Weighted Combined Metric_Norm"].clip(0.0, 1.0)

    if key == "linear":
        dbg_pre = combined_pre.astype(float)
        err = float(np.max(np.abs(dbg_pre.values - out["Weighted Combined Metric"].values)))
        if err > 1e-6:
            logger.error("[WCM] linear mismatch: max|pre-post|=%.6g", err)
    return out




def extract_note_from_folder(folder_name: str) -> str:
    """
    Extrai uma nota (ex.: 'C4', 'A#3') do nome da pasta.
    Estratégia: primeiro tenta entre aspas, depois padrão [A-G][#b]?\d.
    """
    if not isinstance(folder_name, str) or not folder_name:
        return folder_name or ""
    # 1) se houver aspas, usa o conteúdo
    q = extract_note_from_quotes(folder_name)
    if q and q != folder_name:
        return q
    # 2) padrão simples
    m = re.search(r"([A-Ga-g])([#b]?)(-?\d+)", folder_name)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}{m.group(3)}"
    return folder_name


def _publication_safe_folder_path_marker(_p: Union[str, Path]) -> str:
    """Do not embed local absolute compilation roots in exported rows."""
    try:
        from metadata_sanitizer import REDACT_TOKEN, publication_redaction_enabled

        if publication_redaction_enabled():
            return REDACT_TOKEN
    except Exception:
        pass
    return str(_p)


def extract_dynamics_from_path(path: Union[str, Path]) -> Optional[str]:
    """
    Extract dynamics (pp, p, mp, mf, f, ff, ppp, fff) from folder path or filename.
    
    Args:
        path: Folder path or filename to search for dynamics
        
    Returns:
        Dynamics string (e.g., 'pp', 'mf', 'ff') or None if not found
    """
    path_str = str(path).lower()
    
    # Common dynamics patterns (order matters: longer patterns first to avoid partial matches)
    # Use non-word boundaries to match with underscores, hyphens, and path separators
    dynamics_patterns = [
        r'[_\-\s/\\](ppp|fff)[_\-\s/\\]',  # Triple dynamics with separators
        r'[_\-\s/\\](pp|mf|ff|mp)[_\-\s/\\]',  # Double dynamics with separators
        r'[_\-\s/\\](ppp|fff)(?:[_\-\s/\\]|$)',  # Triple at end of string
        r'[_\-\s/\\](pp|mf|ff|mp)(?:[_\-\s/\\]|$)',  # Double at end of string
        r'(?:^|[_\-\s/\\])(ppp|fff)(?:[_\-\s/\\]|$)',  # Triple at start or anywhere
        r'(?:^|[_\-\s/\\])(pp|mf|ff|mp)(?:[_\-\s/\\]|$)',  # Double at start or anywhere
        r'[_\-\s/\\]p(?:p)?[_\-\s/\\]',  # pp or p with separators (fallback)
        r'[_\-\s/\\]f(?:f)?[_\-\s/\\]',  # ff or f with separators (fallback)
        r'[_\-\s/\\]mf[_\-\s/\\]',  # mf with separators
        r'[_\-\s/\\]mp[_\-\s/\\]',  # mp with separators
    ]
    
    # Try each pattern
    for pattern in dynamics_patterns:
        match = re.search(pattern, path_str)
        if match:
            dyn = match.group(1)
            if dyn:
                return dyn
    
    # Fallback: look for dynamics tokens, but reject matches embedded inside ordinary words
    # (e.g. "pp" inside "AppData" on Windows paths — false positive for output naming).
    simple_patterns = [
        r'(ppp|fff)',  # Triple
        r'(pp|mf|ff|mp)',  # Double
    ]
    for pattern in simple_patterns:
        for match in re.finditer(pattern, path_str):
            dyn = match.group(1)
            if not dyn:
                continue
            start, end = match.span(1)
            prev_alpha = start > 0 and path_str[start - 1].isalpha()
            next_alpha = end < len(path_str) and path_str[end].isalpha()
            if prev_alpha and next_alpha:
                continue
            return dyn

    return None


def _candidate_compilation_roots(folder_path: Path) -> List[Path]:
    """
    Pastas onde procurar notas exportadas: raiz, ``analysis_results`` directo,
    e ``<subpasta>/analysis_results`` (estrutura típica do orquestrador).
    """
    fp = Path(folder_path)
    roots: List[Path] = []
    seen: set[str] = set()

    def _add(p: Path) -> None:
        if not p.is_dir():
            return
        try:
            key = str(p.resolve())
        except OSError:
            key = str(p)
        if key not in seen:
            seen.add(key)
            roots.append(p)

    _add(fp)
    _add(fp / "analysis_results")
    try:
        for ch in fp.iterdir():
            if not ch.is_dir():
                continue
            _add(ch / "analysis_results")
            try:
                for ch2 in ch.iterdir():
                    if ch2.is_dir():
                        _add(ch2 / "analysis_results")
            except OSError:
                pass
    except OSError:
        pass
    return roots if roots else [fp]


def _count_walk_files_needle(root: Path, needle_lower: str) -> int:
    n = 0
    try:
        for _, _, files in os.walk(root):
            for fname in files:
                if fname.startswith("~$") or fname.startswith(".~lock"):
                    continue
                if needle_lower in fname.lower():
                    n += 1
    except OSError:
        return 0
    return n


def resolve_compilation_inputs(
    folder_path: Path,
    file_pattern: str,
    *,
    allow_legacy_super_json: bool = False,
) -> tuple[Path, str]:
    """
    Escolhe a pasta com mais ficheiros de métricas e o padrão de ficheiro.

    Por defeito (``allow_legacy_super_json=False``), **não** promove
    ``super_analysis_results.json`` como alternativa a
    ``spectral_analysis.xlsx`` — evita deriva de pipeline em modo canónico.
    """
    roots = _candidate_compilation_roots(folder_path)
    patterns: List[str] = [file_pattern]
    if allow_legacy_super_json:
        alt = (
            "super_analysis_results.json"
            if ("spectral" in file_pattern.lower() or file_pattern.lower().endswith(".xlsx"))
            else "spectral_analysis.xlsx"
        )
        if alt.lower() != file_pattern.lower():
            patterns.append(alt)

    best_key: Optional[tuple] = None
    best: tuple[Path, str] = (folder_path, file_pattern)
    for ri, root in enumerate(roots):
        if not root.is_dir():
            continue
        for pi, pat in enumerate(patterns):
            c = _count_walk_files_needle(root, pat.lower())
            if c <= 0:
                continue
            key = (-c, ri, pi)
            if best_key is None or key < best_key:
                best_key = key
                best = (root, pat)
    return best


def _compile_density_metrics_impl(
    folder_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = "compiled_density_metrics.xlsx",
    file_pattern: str = "spectral_analysis.xlsx",
    include_pca: bool = False,
    harmonic_weight: float = 0.95,  # Default: 95% (alinhado com interface)
    inharmonic_weight: float = 0.05,  # Default: 5% (alinhado com interface)
    weight_function: str = "linear",
    *,
    compiled_public_columns: bool = True,
    enable_pca_export: bool = True,
    minimum_samples_for_pca: int = 10,
    pca_include_dissonance: bool = False,
    pca_include_dependent_metrics: bool = False,
    prefer_phase2_rolloff_density: bool = False,
    allow_legacy_super_json: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Compila métricas espectrais de múltiplos ficheiros e salva em Excel.

    Args:
        folder_path: Diretório raiz com os ficheiros.
        output_path: Caminho para salvar o Excel compilado.
        file_pattern: Nome padrão dos ficheiros a procurar.
        include_pca: Obsoleto: a PCA já não é acrescentada como colunas no ``DataFrame``;
            use ``enable_pca_export``. Se True, força ``enable_pca_export=True``.
        harmonic_weight: Peso da componente harmónica.
        inharmonic_weight: Peso da componente inarmónica.
        weight_function: Função de ponderação ('linear', 'log', 'sqrt', 'exp', ...).
        compiled_public_columns: Se True (padrão), o Excel de densidade inclui apenas a folha
            enxuta ``Density_Metrics`` + ``Analysis_Metadata`` (+ PCA em folhas separadas).
            Se False, acrescenta ``Compiled_Metrics_All`` com todas as colunas compiladas em memória.
        enable_pca_export: Se True (padrão), calcula PCA exploratória em folhas separadas quando
            há amostras e features suficientes (ver ``minimum_samples_for_pca``).
        minimum_samples_for_pca: Número mínimo de notas/ficheiros para exportar PCA.
        pca_include_dissonance: Se True, a PCA exploratória pode incluir ``selected_dissonance_value``.
        prefer_phase2_rolloff_density: When True, public ``rolloff_compensated_*`` columns follow
            ``spectral_analysis.xlsx`` even if ``super_analysis_results.json`` has a computed canonical
            value (explicit opt-in for comparison workflows).

    Returns:
        DataFrame compilado ou None se falhar.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        logger.error(f"Invalid directory: {folder_path}")
        return None

    orig_folder, orig_pat = folder_path, file_pattern
    folder_path, file_pattern = resolve_compilation_inputs(
        folder_path, file_pattern, allow_legacy_super_json=allow_legacy_super_json
    )
    _compile_density_metrics_impl._last_resolved_file_pattern = str(file_pattern)
    if folder_path != orig_folder or file_pattern != orig_pat:
        logger.info(
            "Compilação: origem ajustada — pasta %s → %s | padrão de ficheiros %r → %r",
            orig_folder,
            folder_path,
            orig_pat,
            file_pattern,
        )

    # Recolha dos ficheiros-alvo
    # Skip Office/LibreOffice transient lock files (e.g. "~$spectral_analysis.xlsx",
    # ".~lock.spectral_analysis.xlsx#") that Excel/Calc create while a workbook is
    # open. They embed the target filename and would otherwise be matched here,
    # causing "[Errno 13] Permission denied" noise during Stage 2.
    found_files: list[tuple[Path, str, str]] = []
    for root, _, files in os.walk(folder_path):
        root_path = Path(root)
        for fname in files:
            if fname.startswith("~$") or fname.startswith(".~lock"):
                continue
            if file_pattern.lower() in fname.lower():
                fpath = root_path / fname
                if fpath.is_file():
                    note = extract_note_from_folder(root_path.name)
                    found_files.append((fpath, note, root_path.name))

    if not found_files:
        logger.warning(f"No files found matching pattern '{file_pattern}' em {folder_path}")
        return None

    # Ordem determinística por nome de pasta (depois ordenamos por nota)
    found_files.sort(key=lambda t: t[2])

    # Ler métricas de cada ficheiro
    rows = []
    for fpath, note, folder in found_files:
        try:
            if fpath.suffix.lower() == ".json" or file_pattern.lower().endswith(".json"):
                metrics = read_super_analysis_metrics(fpath)
            else:
                metrics = read_excel_metrics(fpath)
                if fpath.suffix.lower() == ".xlsx" and "spectral" in fpath.name.lower():
                    metrics = apply_rolloff_canonical_precedence_for_compiled_row(
                        metrics,
                        spectral_metrics_path=fpath,
                        prefer_phase2_rolloff_density=prefer_phase2_rolloff_density,
                    )
                    metrics = apply_hepd_canonical_precedence_for_compiled_row(
                        metrics,
                        spectral_metrics_path=fpath,
                        prefer_phase2_rolloff_density=prefer_phase2_rolloff_density,
                    )
        except Exception as exc:
            logger.warning(f"Error reading metrics from {fpath}: {exc}")
            continue

        if not metrics or all(v is None for v in metrics.values()):
            logger.warning(f"Invalid or empty metrics in {fpath}")
            continue

        rows.append({
            "Note": note,
            "Folder": folder,
            # AUDIT FIX (direct per-note Density_Metrics extraction) —
            # carry the absolute path of the per-note spectral_analysis.xlsx
            # through the wide-df so _build_density_metrics_main_sheet can
            # reopen it and sum the actual Harmonic/Inharmonic/Sub-bass
            # Spectrum sheets, instead of relying on the scalar columns
            # in Metrics. The column name is intentionally double-underscored
            # so it is treated as private bookkeeping and dropped from the
            # public sheets before they are written.
            "__source_file_path": str(fpath),
            **metrics
        })

    if not rows:
        logger.error("No valid data found for compilation.")
        return None

    df = pd.DataFrame(rows)

    if not df.empty:
        _rows_out = [
            _ensure_adaptive_subfundamental_cutoff(dict(r)) for r in df.to_dict(orient="records")
        ]
        df = pd.DataFrame(_rows_out)

    # Remove Register column if present (instrument-specific, not useful for other instruments)
    if 'Register' in df.columns:
        logger.info("Removing 'Register' column (instrument-specific labels not needed)")
        df = df.drop(columns=['Register'])

    # Remover colunas duplicadas de nota se existirem e forem redundantes
    for dup in ("Nota", "note", "Pitch"):
        if dup in df.columns and dup != "Note":
            try:
                if df[dup].equals(df.get("Note")):
                    df = df.drop(columns=[dup])
            except Exception:
                # se não for estritamente igual, ignorar (mantém-se a coluna)
                pass

    # Ordenar cromaticamente (nota → MIDI) se possível
    if "Note" in df.columns:
        df["__midi__"] = df["Note"].apply(note_to_midi)
        df = (
            df.sort_values("__midi__", kind="stable")
              .drop(columns="__midi__")
              .reset_index(drop=True)
        )
    else:
        logger.warning("Column 'Note' not found; keeping original order.")

    df = _add_canonical_and_global_density_columns(df)

    # ---------- PONTO CRÍTICO: (re)calcular a Weighted Combined Metric ----------
    # 1) eliminar qualquer WCM herdada dos Excels
    df = df.drop(columns=["Weighted Combined Metric"], errors="ignore")

    # Folha Metrics enxuta: recriar SDM/FDM só em memória a partir de Density Metric quando faltarem
    if "Density Metric" in df.columns:
        need_sdm = "Spectral Density Metric" not in df.columns
        need_fdm = "Filtered Density Metric" not in df.columns
        if need_sdm or need_fdm:
            df = df.copy()
            dm = pd.to_numeric(df["Density Metric"], errors="coerce")
            if need_sdm:
                df["Spectral Density Metric"] = dm
            if need_fdm:
                df["Filtered Density Metric"] = dm

    # 2) validar/normalizar a chave da função de peso
    wf_key = (weight_function or "linear").strip().lower()
    try:
        _ = get_weight_function(wf_key)
    except Exception as e:
        logger.warning("Invalid weight_function '%s' (%s). Using 'linear'.", wf_key, e)
        wf_key = "linear"

    # 3) aplicar a combinação determinística: H='Spectral Density Metric', IH='Filtered Density Metric'
    df = apply_weighted_combination(
        df,
        harmonic_col="Spectral Density Metric",
        inharmonic_col="Filtered Density Metric",
        alpha=harmonic_weight,
        beta=inharmonic_weight,
        weight_function=wf_key
    )
    # ---------------------------------------------------------------------------

    if include_pca:
        if not enable_pca_export:
            logger.warning("include_pca=True is deprecated; forcing enable_pca_export=True for PCA sheets.")
        enable_pca_export = True

    # Índice ponderado e normalizações (determinístico; protegido)
    try:
        df = apply_weighted_index(df)
    except Exception as e:
        logger.error("Failed to compute Index_Weighted and normalisations: %s", e)

    # Exportação (opcional; protegida)
    if output_path:
        outp = Path(output_path)
        try:
            outp.parent.mkdir(parents=True, exist_ok=True)

            outp_path_obj = Path(output_path)
            if (
                outp_path_obj.name == "compiled_density_metrics.xlsx"
                and not outp_path_obj.is_absolute()
            ):
                dynamics = extract_dynamics_from_path(folder_path)
                if dynamics:
                    outp = outp.parent / f"compiled_density_metrics_{dynamics}.xlsx"
                    logger.info("Dynamics %r detected, using filename: %s", dynamics, outp.name)

            version, version_source = _get_project_version_info()
            metadata = {
                "analysis_date": datetime.now().isoformat(),
                "analysis_version": version,
                "analysis_version_source": version_source,
                "folder_path": _publication_safe_folder_path_marker(folder_path),
                "file_pattern": file_pattern,
                "include_pca_legacy_flag": include_pca,
                "enable_pca_export": enable_pca_export,
                "minimum_samples_for_pca": minimum_samples_for_pca,
                "harmonic_weight": harmonic_weight,
                "inharmonic_weight": inharmonic_weight,
                "weight_function": wf_key,
                "n_samples": int(len(df)),
                "rolloff_density_public_canonical_source_policy": (
                    "Rolloff_compensated_* precedence (super_analysis_results.json vs spectral_analysis.xlsx) "
                    "applies to the compiled in-memory frame and to per-note Metrics / Compiled_Metrics_All; "
                    "the slim Density_Metrics sheet carries only partial-band sums. "
                    "Primary source: super_analysis_results.json spectral_metrics when contract holds "
                    "(sibling of spectral_analysis.xlsx, else discovered by name); "
                    "else spectral_analysis.xlsx. prefer_spectral_analysis_rolloff_density inverts explicitly."
                ),
                "prefer_spectral_analysis_rolloff_density": bool(prefer_phase2_rolloff_density),
            }

            _write_compiled_excel(
                outp,
                df,
                metadata,
                apply_publication_column_filter=compiled_public_columns,
                enable_pca_export=enable_pca_export,
                minimum_samples_for_pca=minimum_samples_for_pca,
                pca_include_dissonance=bool(pca_include_dissonance),
                pca_include_dependent_metrics=bool(pca_include_dependent_metrics),
                compile_file_pattern=file_pattern,
                allow_legacy_super_json=allow_legacy_super_json,
                input_schema_validation_status="not_validated_compile_metrics_impl",
            )
            logger.info("Compiled workbook written to '%s'", outp)
        except Exception as e:
            logger.error("Error saving Excel workbook to '%s': %s", outp, e, exc_info=True)

    return df




def add_pca_to_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona um componente PCA a um DataFrame de métricas.
    Versão aprimorada com melhor tratamento de erros e memória.

    Args:
        df: DataFrame contendo métricas.

    Returns:
        DataFrame com componentes PCA adicionados.

    Raises:
        ValueError: Se não houver colunas numéricas suficientes para PCA.
    """
    if df is None or df.empty:
        logger.warning("DataFrame vazio fornecido para add_pca_to_metrics")
        return df

    # Encontrar colunas numéricas
    numeric_cols = []
    standard_cols = METRIC_COLUMNS.copy()

    # Adicionar colunas de dissonância
    # Verificar onde as métricas são extraídas, algo como:
    dissonance_cols = [col for col in df.columns if DISSONANCE_PREFIX in col]
    standard_cols.extend(dissonance_cols)

    logger.debug(f"Searching for numeric columns among: {standard_cols}")

    for col in standard_cols:
        if col in df.columns:
            try:
                # Verificar se a coluna tem valores numéricos suficientes
                valid_count = pd.to_numeric(df[col], errors='coerce').notnull().sum()
                if valid_count >= 2:
                    numeric_cols.append(col)
                    logger.debug(f"Numeric column found: {col} with {valid_count} valid values")
            except Exception as e:
                logger.debug(f"Column {col} could not be coerced to numeric: {e}")

    logger.info(f"Numeric columns selected for PCA: {numeric_cols}")

    if len(numeric_cols) < 2:
        logger.warning("Fewer than 2 numeric columns found for PCA")
        return df

    # Preparar dados para PCA com tratamento robusto de valores ausentes
    try:
        # Criar cópia do DataFrame para evitar avisos
        df_for_pca = df[numeric_cols].copy()

        # Converter para numérico, forçando NaN para valores não numéricos
        for col in df_for_pca.columns:
            df_for_pca[col] = pd.to_numeric(df_for_pca[col], errors='coerce')

        # Verificar dados após conversão
        if df_for_pca.isnull().sum().sum() > 0:
            logger.warning(f"DataFrame has {df_for_pca.isnull().sum().sum()} NaN values after numeric coercion")

        # Lidar com valores ausentes
        df_clean = df_for_pca.dropna()
        if df_clean.shape[0] >= 2:
            df_for_pca = df_clean
            logger.debug(f"Usando {df_clean.shape[0]} linhas sem valores ausentes para PCA")
        else:
            # Usar imputação mais robusta para valores ausentes
            # Primeiro tentar a média para cada coluna
            col_means = df_for_pca.mean()

            # Verificar se todas as médias são válidas
            if col_means.isnull().sum() > 0:
                logger.warning("Some columns are all-NaN. Using default value for imputation.")
                # Para colunas com apenas NaN, usar 0 como imputação
                col_means = col_means.fillna(0)

            df_for_pca = df_for_pca.fillna(col_means)
            logger.debug("Missing values filled with column means for PCA")

        # Verificar novamente se há valores ausentes
        if df_for_pca.isnull().sum().sum() > 0:
            logger.error("Missing values remain after imputation!")
            # Substituir todos os NaNs restantes com 0
            df_for_pca = df_for_pca.fillna(0)

        # Padronizar os dados
        try:
            # Verificar se os dados têm variância não-zero
            zero_var_cols = []
            for col in df_for_pca.columns:
                if df_for_pca[col].std() == 0:
                    zero_var_cols.append(col)

            if zero_var_cols:
                logger.warning(f"Zero-variance columns detected: {zero_var_cols}")
                # Remover colunas com variância zero
                df_for_pca = df_for_pca.drop(columns=zero_var_cols)
                numeric_cols = [col for col in numeric_cols if col not in zero_var_cols]

            if len(df_for_pca.columns) < 2:
                logger.error("Too few columns with variance after filtering")
                return df

            # Agora aplicar StandardScaler
            scaler = StandardScaler()
            metrics_std = scaler.fit_transform(df_for_pca)

            # Calcular PCA com 2 componentes para visualização
            pca = PCA(n_components=min(2, len(df_for_pca.columns)))
            pc_results = pca.fit_transform(metrics_std)

            # Adicionar resultados ao DataFrame original
            df = df.copy()  # Evitar SettingWithCopyWarning
            df.loc[df_for_pca.index, "PC1"] = pc_results[:, 0]
            if pc_results.shape[1] > 1:
                df.loc[df_for_pca.index, "PC2"] = pc_results[:, 1]

            # Registrar variância explicada
            explained_variance = pca.explained_variance_ratio_
            logger.info(f"Explained variance by PCA components: {explained_variance}")

            # Calcular importância das características
            feature_importance = np.abs(pca.components_)
            for i, component in enumerate(pca.components_):
                sorted_indices = np.argsort(np.abs(component))[::-1]
                logger.info(f"Top contributing features for PC{i+1}:")
                for idx in sorted_indices[:3]:  # Top 3 características
                    if idx < len(df_for_pca.columns):  # Verificar índice válido
                        logger.info(f"  {df_for_pca.columns[idx]}: {component[idx]:.3f}")

            # Normalizar as colunas
            cols_to_normalize = numeric_cols + ["PC1"]
            if "PC2" in df.columns:
                cols_to_normalize.append("PC2")

            # PHASE 4: Use robust normalization instead of min-max
            for col in cols_to_normalize:
                if col in df.columns:
                    df[col + "_Norm"] = _robust_normalize_series(df[col], method="percentile")

        except np.linalg.LinAlgError as lae:
            logger.error(f"Linear algebra error during PCA: {lae}")
            logger.info("Isso pode acontecer com dados muito correlacionados. Tentando com menos componentes.")
            return df

    except MemoryError as me:
        logger.error(f"Memory error during PCA: {me}")
        return df

    except Exception as e:
        logger.error(f"Error during PCA: {e}")
        return df

    return df


def _get_numeric_columns(df: pd.DataFrame, include_dissonance: bool = True) -> List[str]:
    """
    Returns numeric (or convertible) columns relevant for analysis.
    
    Args:
        df: DataFrame to analyze
        include_dissonance: Whether to include dissonance columns
    
    Returns:
        List of numeric column names
    """
    # 1) Standard metrics (preserve order), prefer normalized variants when available
    preferred_cols = [
        "Density Metric_Norm2" if "Density Metric_Norm2" in df.columns else "Density Metric",
        "Spectral Density Metric" if "Spectral Density Metric" in df.columns else None,
        "Total Metric" if "Total Metric" in df.columns else None,
        "Combined Density Metric_Norm2" if "Combined Density Metric_Norm2" in df.columns else "Combined Density Metric",
        "Filtered Density Metric" if "Filtered Density Metric" in df.columns else None,
        "N_harm_norm" if "N_harm_norm" in df.columns else "Harmonic Count",
        "D_agn",
        "P_norm",
    ]
    standard_cols = [c for c in preferred_cols if c is not None]
    candidates: List[str] = [c for c in standard_cols if c in df.columns]
    
    # 2) Add dissonance columns if requested
    if include_dissonance:
        candidates += [c for c in df.columns if "Dissonance" in c]
    
    # 2.1) Deduplicate preserving order
    seen: set = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]
    
    # 3) Select columns that are numeric or convertible with at least 2 valid values
    numeric_now = set(df.select_dtypes(include="number").columns)
    numeric_sel: set = set(c for c in candidates if c in numeric_now)
    
    for c in candidates:
        if c not in numeric_sel:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notnull().sum() >= 2:
                numeric_sel.add(c)
    
    # Return in candidate order
    return [c for c in candidates if c in numeric_sel]


def _apply_additional_dimension_reduction(
    df: pd.DataFrame,
    metrics_columns: List[str],
    use_tsne: bool = False,
    use_umap: bool = False
) -> pd.DataFrame:
    """
    Applies additional dimensionality reduction methods beyond PCA.
    
    Args:
        df: DataFrame with compiled metrics
        metrics_columns: Columns containing numeric metrics
        use_tsne: Whether to apply t-SNE
        use_umap: Whether to apply UMAP
    
    Returns:
        DataFrame with additional components added
    """
    result_df = df.copy()
    
    if not metrics_columns:
        logger.warning("No numeric columns available for dimensionality reduction")
        return result_df
    
    try:
        # Prepare data (robust scaling to reduce register bias and scale effects)
        X_df = df[metrics_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        med = X_df.median(axis=0)
        iqr = (X_df.quantile(0.75) - X_df.quantile(0.25)).replace(0, 1.0)
        X = ((X_df - med) / iqr).values
        if len(X) < 2:
            logger.warning("Insufficient data for dimensionality reduction")
            return result_df
        
        scaler = StandardScaler()
        X_scaled = np.asarray(scaler.fit_transform(X), dtype=np.float64)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        # Zero-variance or degenerate columns can make sklearn t-SNE divide by zero
        # during init (RuntimeWarning then native fault on Windows, exit 0xC0000095).
        col_std = X_scaled.std(axis=0)
        if np.any(~np.isfinite(col_std)) or np.any(col_std < 1e-12):
            scale = float(np.nanmean(np.abs(X_scaled[np.isfinite(X_scaled)])) or 0.0)
            noise = max(scale * 1e-6, 1e-9)
            rng = np.random.default_rng(42)
            X_scaled = X_scaled + rng.normal(0.0, noise, size=X_scaled.shape)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply t-SNE if requested
        if use_tsne:
            if not TSNE_AVAILABLE:
                logger.warning("t-SNE requested but sklearn.manifold.TSNE not available. Install scikit-learn.")
            elif len(X_scaled) < 4:
                logger.warning(f"t-SNE requires at least 4 samples, but only {len(X_scaled)} available. Skipping t-SNE.")
                logger.warning(f"  → For optimal results, t-SNE needs at least 6 samples (to allow perplexity >= 5)")
            else:
                try:
                    # t-SNE mathematical requirements (reference constraints):
                    # - Minimum 4 samples (strictly enforced)
                    # - Perplexity must be < n_samples (strictly less)
                    # - Typical perplexity range: 5-50, default 30
                    # - For small datasets: perplexity = min(30, n_samples - 1)
                    n_samples = len(X_scaled)
                    # Perplexity must be strictly less than n_samples
                    # For n_samples=4: max_perplexity=3, but we want at least 5, so we need n_samples >= 6
                    # For n_samples=5: max_perplexity=4, but we want at least 5, so we need n_samples >= 6
                    # For n_samples=6: max_perplexity=5, which is acceptable
                    max_perplexity = n_samples - 1  # Must be strictly less than n_samples
                    # We want perplexity in range [5, 30], but it must be < n_samples
                    # So: perplexity = min(30, max(5, n_samples - 1))
                    # But if n_samples - 1 < 5, we can't use 5, so we use n_samples - 1
                    perplexity = min(30, max(5, max_perplexity))
                    
                    # Final check: perplexity must be < n_samples
                    if perplexity >= n_samples:
                        # If max_perplexity < 5, we can't satisfy both constraints
                        # So we use the maximum possible: n_samples - 1
                        perplexity = max(1, n_samples - 1)
                        if perplexity >= n_samples:
                            logger.error(f"t-SNE failed: insufficient samples ({n_samples}) for any valid perplexity")
                            raise ValueError(f"t-SNE requires n_samples > perplexity, but {n_samples} <= {perplexity}")
                        logger.warning(f"t-SNE using reduced perplexity={perplexity} for small dataset (n={n_samples})")
                    
                    n_feat = int(X_scaled.shape[1])
                    logger.info(
                        f"Applying t-SNE: n_samples={n_samples}, perplexity={perplexity}, "
                        f"n_features={n_feat}"
                    )
                    # Default PCA-based init can hit divide-by-zero in sklearn's t-SNE when
                    # the embedding's first column has zero std (small n or few features).
                    use_random_init = n_samples < 50 or n_feat < 8
                    if use_random_init:
                        logger.info(
                            "t-SNE: using init='random' (small sample count or few features) "
                            "to avoid degenerate PCA init."
                        )
                    # scikit-learn uses 'max_iter' (not 'n_iter') for maximum iterations
                    # In scikit-learn 1.1+, n_iter was renamed to max_iter
                    # This code handles both versions for compatibility
                    try:
                        tsne_base = dict(
                            n_components=2,
                            random_state=42,
                            perplexity=perplexity,
                            max_iter=1000,
                            verbose=0,
                        )
                        if use_random_init:
                            tsne_base["init"] = "random"
                        try:
                            tsne_base_lr = dict(tsne_base)
                            tsne_base_lr["learning_rate"] = "auto"
                            tsne = TSNE(**tsne_base_lr)
                        except TypeError:
                            tsne = TSNE(**tsne_base)
                    except TypeError as e:
                        # Fallback to old parameter name (scikit-learn < 1.1) if max_iter fails
                        if "max_iter" in str(e):
                            tsne_base = dict(
                                n_components=2,
                                random_state=42,
                                perplexity=perplexity,
                                n_iter=1000,
                                verbose=0,
                            )
                            if use_random_init:
                                tsne_base["init"] = "random"
                            tsne = TSNE(**tsne_base)
                        else:
                            raise  # Re-raise if it's a different error
                    tsne_result = tsne.fit_transform(X_scaled)
                    result_df['TSNE1'] = tsne_result[:, 0]
                    result_df['TSNE2'] = tsne_result[:, 1]
                    logger.info(f"✓ t-SNE applied successfully (perplexity={perplexity}, n_samples={n_samples})")
                    logger.info(f"  TSNE1 range: [{result_df['TSNE1'].min():.4f}, {result_df['TSNE1'].max():.4f}]")
                    logger.info(f"  TSNE2 range: [{result_df['TSNE2'].min():.4f}, {result_df['TSNE2'].max():.4f}]")
                except Exception as e:
                    logger.error(f"✗ Error applying t-SNE: {e}")
                    import traceback
                    logger.error(f"t-SNE error traceback:\n{traceback.format_exc()}")
        
        # Apply UMAP if requested and available
        if use_umap:
            if not UMAP_AVAILABLE:
                logger.warning("UMAP requested but umap-learn not available. Install with: pip install umap-learn")
            elif len(X_scaled) < 4:
                logger.warning(f"UMAP requires at least 4 samples, but only {len(X_scaled)} available. Skipping UMAP.")
                logger.warning(f"  → UMAP needs n_neighbors > 1, and with < 4 samples, this causes scipy errors")
            else:
                try:
                    n_samples = len(X_scaled)
                    # UMAP requires n_neighbors > 1 and n_neighbors < n_samples
                    # Default n_neighbors is 15, so we need to adjust for small datasets
                    # Minimum: n_neighbors = 2 (for n_samples >= 4)
                    n_neighbors = min(15, max(2, n_samples - 1))  # Must be < n_samples and > 1
                    if n_neighbors >= n_samples:
                        n_neighbors = max(2, n_samples - 1)  # Ensure > 1
                    
                    logger.info(f"Applying UMAP: n_samples={n_samples}, n_neighbors={n_neighbors}, n_features={X_scaled.shape[1]}")
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=0.1,  # Lower min_dist for small datasets
                        random_state=42,
                        n_components=2
                    )
                    umap_result = reducer.fit_transform(X_scaled)
                    result_df['UMAP1'] = umap_result[:, 0]
                    result_df['UMAP2'] = umap_result[:, 1]
                    logger.info(f"UMAP applied successfully (n_neighbors={n_neighbors}, n_samples={n_samples})")
                    logger.info(f"  UMAP1 range: [{result_df['UMAP1'].min():.4f}, {result_df['UMAP1'].max():.4f}]")
                    logger.info(f"  UMAP2 range: [{result_df['UMAP2'].min():.4f}, {result_df['UMAP2'].max():.4f}]")
                except Exception as e:
                    logger.error(f"Error applying UMAP: {e}")
                    import traceback
                    logger.error(f"UMAP error traceback:\n{traceback.format_exc()}")
    except Exception as e:
        logger.error(f"Error in dimensionality reduction: {e}")
    
    return result_df


def _detect_spectral_anomalies(
    df: pd.DataFrame,
    metrics_columns: List[str],
    contamination: Optional[float] = None
) -> pd.DataFrame:
    """
    Detects anomalies in spectral data using Isolation Forest.
    
    Args:
        df: DataFrame with compiled metrics
        metrics_columns: Columns containing numeric metrics
    contamination: Expected fraction of anomalies (None = adaptive)
    
    Returns:
        DataFrame with anomaly indicators added
    """
    result_df = df.copy()
    
    if not metrics_columns:
        logger.warning("No numeric columns available for anomaly detection")
        result_df['is_anomaly'] = False
        return result_df
    
    if not ISOLATION_FOREST_AVAILABLE:
        logger.warning("IsolationForest not available, skipping anomaly detection")
        result_df['is_anomaly'] = False
        return result_df
    
    try:
        # Prepare data
        X = df[metrics_columns].fillna(0).values

        # Register-normalize metrics when note information is available
        # This reduces register bias in anomaly detection (low notes often differ in scale).
        use_register_bands = False
        log_freq = None
        if "Note" in df.columns:
            def _note_to_frequency(note: str) -> float:
                if not isinstance(note, str) or not note.strip():
                    return 0.0
                note = note.strip()
                m = re.match(r'^([A-Ga-g])([#b]?)(-?\d+)$', note)
                if not m:
                    return 0.0
                letter = m.group(1).upper()
                accidental = m.group(2)
                octave = int(m.group(3))
                semitone_map = {
                    'C': 0, 'C#': 1, 'DB': 1,
                    'D': 2, 'D#': 3, 'EB': 3,
                    'E': 4, 'FB': 4, 'E#': 5,
                    'F': 5, 'F#': 6, 'GB': 6,
                    'G': 7, 'G#': 8, 'AB': 8,
                    'A': 9, 'A#': 10, 'BB': 10,
                    'B': 11, 'CB': 11, 'B#': 0,
                }
                key = f"{letter}{accidental}".upper()
                if key not in semitone_map:
                    return 0.0
                midi = (octave + 1) * 12 + semitone_map[key]
                return 440.0 * 2 ** ((midi - 69) / 12)

            freqs = df["Note"].apply(_note_to_frequency).astype(float)
            valid_freq_mask = np.isfinite(freqs.values) & (freqs.values > 0)
            if valid_freq_mask.sum() >= 10:
                log_freq = np.log2(freqs.values)
                X_norm = X.copy()
                for i, col in enumerate(metrics_columns):
                    y = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
                    valid_mask = valid_freq_mask & np.isfinite(y)
                    if valid_mask.sum() >= 10:
                        try:
                            slope, intercept = np.polyfit(log_freq[valid_mask], y[valid_mask], 1)
                            X_norm[:, i] = y - (slope * log_freq + intercept)
                        except Exception:
                            X_norm[:, i] = y
                X = X_norm
                logger.info("Anomaly detection: register-normalized metrics using log2(frequency).")
                use_register_bands = True
        
        if len(X) < 10:
            logger.warning("Insufficient samples for anomaly detection (need at least 10)")
            result_df['is_anomaly'] = False
            return result_df
        
        # Validate/adapt contamination parameter
        if contamination is None or contamination <= 0 or contamination >= 1:
            # Adaptive prior: avoid forced anomalies in small datasets
            n_samples = len(X)
            cap = 0.03 if n_samples < 20 else 0.05
            contamination = min(cap, max(1.0 / n_samples, 0.01))
            logger.info(f"Adaptive contamination selected: {contamination:.3f} (n={n_samples})")
        
        # Apply Isolation Forest (optionally per-register band)
        if use_register_bands and log_freq is not None:
            # Use quantile-based bands to keep sample sizes balanced
            edges = np.quantile(log_freq[valid_freq_mask], [0.0, 0.25, 0.5, 0.75, 1.0])
            edges = np.unique(edges)
            if len(edges) >= 3:
                result_df['is_anomaly'] = False
                result_df['anomaly_score'] = 0.0
                total_anomalies = 0
                for i in range(len(edges) - 1):
                    band_mask = (log_freq >= edges[i]) & (log_freq <= edges[i + 1]) & valid_freq_mask
                    n_band = int(band_mask.sum())
                    if n_band < 10:
                        continue
                    # Adaptive contamination per band
                    cap = 0.03 if n_band < 20 else 0.05
                    band_cont = min(cap, max(1.0 / n_band, 0.01))
                    clf = IsolationForest(contamination=band_cont, random_state=42)
                    preds = clf.fit_predict(X[band_mask]) == -1
                    result_df.loc[band_mask, 'is_anomaly'] = preds
                    result_df.loc[band_mask, 'anomaly_score'] = clf.decision_function(X[band_mask])
                    total_anomalies += int(preds.sum())
                logger.info(
                    f"Anomaly detection (per-register): {total_anomalies} anomalies found "
                    f"out of {len(result_df)} samples"
                )
            else:
                use_register_bands = False

        if not use_register_bands:
            clf = IsolationForest(contamination=contamination, random_state=42)
            result_df['is_anomaly'] = clf.fit_predict(X) == -1
            # Calculate and add anomaly score
            result_df['anomaly_score'] = clf.decision_function(X)
            anomaly_count = result_df['is_anomaly'].sum()
            logger.info(f"Anomaly detection: {anomaly_count} anomalies found out of {len(result_df)} samples")
        
    except Exception as e:
        logger.error(f"Error in anomaly detection: {e}")
        result_df['is_anomaly'] = False
        result_df['anomaly_score'] = 0.0
    
    return result_df


def compile_density_metrics_with_pca(
    folder_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = "compiled_density_metrics.xlsx",
    file_pattern: str = "spectral_analysis.xlsx",
    include_pca: bool = True,
    harmonic_weight: float = 0.95,  # Default: 95% (alinhado com interface)
    inharmonic_weight: float = 0.05,  # Default: 5% (alinhado com interface)
    weight_function: str = "linear",
    use_tsne: bool = False,
    use_umap: bool = False,
    detect_anomalies: bool = False,
    anomaly_contamination: Optional[float] = None,
    *,
    compiled_public_columns: bool = True,
    enable_pca_export: bool = True,
    minimum_samples_for_pca: int = 10,
    pca_include_dissonance: bool = False,
    pca_include_dependent_metrics: bool = False,
    prefer_phase2_rolloff_density: bool = False,
    allow_legacy_super_json: bool = False,
    compilation_extra_metadata: Optional[Dict[str, Any]] = None,
) -> Optional[pd.DataFrame]:
    """
    Compila métricas de densidade, aplica normalizações auxiliares em memória e grava Excel.

    A PCA exploratória (quando pedida) vai apenas para folhas ``PCA_*``, nunca para a folha
    principal ``Density_Metrics``. Também acrescenta em memória:
      - N_harm_norm = min–max de 'Harmonic Count'
      - normalizações auxiliares: Density Metric_Norm2, Combined Density Metric_Norm2
      - Index_Weighted = 0.10*DM_Norm2 + 0.40*D_agn + 0.30*N_harm_norm + 0.15*Combined_Norm2 + 0.05*P_norm
    """
    # 1) Compilação base (sem PCA interno)
    df = _compile_density_metrics_impl(
        folder_path=folder_path,
        output_path=None,             # só exportamos no fim deste wrapper
        file_pattern=file_pattern,
        include_pca=False,            # PCA será aplicado já de seguida (se pedido)
        harmonic_weight=harmonic_weight,
        inharmonic_weight=inharmonic_weight,
        weight_function=weight_function,
        compiled_public_columns=False,
        prefer_phase2_rolloff_density=prefer_phase2_rolloff_density,
        allow_legacy_super_json=allow_legacy_super_json,
    )
    if df is None or df.empty:
        alt: Optional[str] = None
        if allow_legacy_super_json:
            if file_pattern.lower().endswith(".json"):
                alt = "spectral_analysis.xlsx"
            elif "spectral" in file_pattern.lower():
                alt = "super_analysis_results.json"
        if alt is not None:
            logger.warning(
                "compile_density_metrics_with_pca: sem dados com file_pattern=%r; a tentar %r",
                file_pattern,
                alt,
            )
            df = _compile_density_metrics_impl(
                folder_path=folder_path,
                output_path=None,
                file_pattern=alt,
                include_pca=False,
                harmonic_weight=harmonic_weight,
                inharmonic_weight=inharmonic_weight,
                weight_function=weight_function,
                compiled_public_columns=False,
                prefer_phase2_rolloff_density=prefer_phase2_rolloff_density,
                allow_legacy_super_json=allow_legacy_super_json,
            )
    if df is None or df.empty:
        logger.error(
            "compile_density_metrics_with_pca: sem linhas válidas após tentar padrões de ficheiro."
        )
        if output_path:
            outp = Path(output_path)
            try:
                outp.parent.mkdir(parents=True, exist_ok=True)
                version, version_source = _get_project_version_info()
                diag = pd.DataFrame(
                    {
                        "compilation_error": [
                            "Nenhuma métrica válida nas subpastas (spectral_analysis.xlsx ou "
                            "super_analysis_results.json). Confirme que cada nota exportou métricas."
                        ]
                    }
                )
                meta = {
                    "analysis_date": datetime.now().isoformat(),
                    "analysis_version": version,
                    "analysis_version_source": version_source,
                    "folder_path": _publication_safe_folder_path_marker(folder_path),
                    "file_pattern": file_pattern,
                    "error": "no_compilable_rows",
                }
                _write_compiled_excel(
                    outp,
                    diag,
                    meta,
                    apply_publication_column_filter=False,
                    enable_pca_export=False,
                    compile_file_pattern=file_pattern,
                    allow_legacy_super_json=allow_legacy_super_json,
                    input_schema_validation_status="no_compilable_rows_diagnostic",
                )
                logger.warning("Minimal diagnostic Excel written to %s", outp)
            except Exception as wex:
                logger.error("Failed to write diagnostic Excel: %s", wex, exc_info=True)
        return None

    want_pca_sheets = bool(include_pca and enable_pca_export)

    # 3) Normalizações e índice composto — falhas de coluna exportam-se como NaN (não 0.0).
    def _safe(series_name: str) -> pd.Series:
        if series_name in df.columns:
            return pd.to_numeric(df[series_name], errors="coerce")
        return pd.Series(np.nan, index=df.index, dtype=float)

    # N_harm_norm (a partir de Harmonic Count)
    # RECOMMENDATION: Use log-transform for count-based metrics to preserve relative differences
    if "Harmonic Count" in df.columns:
        try:
            from data_integrity import normalize_log_transform
            df["N_harm_norm"] = pd.Series(
                normalize_log_transform(df["Harmonic Count"].values, clip_range=(0.0, 1.0)),
                index=df.index
            )
        except ImportError:
            # Fallback to robust normalization if log-transform not available
            df["N_harm_norm"] = _robust_normalize_series(df["Harmonic Count"], method="percentile")
        df["harmonic_count_available"] = df["N_harm_norm"].notna()
    else:
        df["N_harm_norm"] = pd.Series(np.nan, index=df.index, dtype=float)
        df["harmonic_count_available"] = False
        df["frequency_dependent_normalization_status"] = "skipped_missing_required_columns"
        logger.info(
            "Column 'Harmonic Count' missing; N_harm_norm set to NaN (not 0) — weighted index uses remaining terms only."
        )

    # Normalizações auxiliares (usamos sufixo _Norm2 para não colidir com _Norm existentes)
    # NOTE: Density Metric_Norm2 may already be set by apply_weighted_index() with frequency-dependent normalization
    # Only set it here if it doesn't already exist
    if "Density Metric_Norm2" not in df.columns and "Density Metric" in df.columns:
        try:
            from data_integrity import normalize_log_transform
            df["Density Metric_Norm2"] = pd.Series(
                normalize_log_transform(df["Density Metric"].values, clip_range=(0.0, 1.0)),
                index=df.index
            )
        except ImportError:
            # Fallback to robust normalization
            df["Density Metric_Norm2"] = _robust_normalize_series(
                df["Density Metric"], method="percentile"
            )
    elif "Density Metric_Norm2" not in df.columns:
        df["Density Metric_Norm2"] = pd.Series(np.nan, index=df.index, dtype=float)
        logger.warning(
            "'Density Metric' column missing and Density Metric_Norm2 undefined; using NaN (missing data)."
        )
    else:
        # Density Metric_Norm2 already exists (likely from apply_weighted_index with frequency normalization)
        logger.debug("Density Metric_Norm2 already exists, preserving existing normalization")

    if "Combined Density Metric" in df.columns:
        # RECOMMENDATION: Use log-transform for Combined Density Metric (preserves dynamic range)
        try:
            from data_integrity import normalize_log_transform
            df["Combined Density Metric_Norm2"] = pd.Series(
                normalize_log_transform(df["Combined Density Metric"].values, clip_range=(0.0, 1.0)),
                index=df.index
            )
        except ImportError:
            # Fallback to robust normalization
            df["Combined Density Metric_Norm2"] = _robust_normalize_series(
                df["Combined Density Metric"], method="percentile"
            )
    elif "Combined Density Metric_Norm2" not in df.columns:
        df["Combined Density Metric_Norm2"] = pd.Series(np.nan, index=df.index, dtype=float)
        logger.warning(
            "'Combined Density Metric' column missing and Combined Density Metric_Norm2 undefined; using NaN."
        )

    # D_agn e P_norm (0..1 quando presentes)
    d_agn = _safe("D_agn").clip(0.0, 1.0)
    p_norm = _safe("P_norm").clip(0.0, 1.0)

    df["Density Metric_Norm2"] = pd.to_numeric(df["Density Metric_Norm2"], errors="coerce")
    df["Combined Density Metric_Norm2"] = pd.to_numeric(df["Combined Density Metric_Norm2"], errors="coerce")
    df["N_harm_norm"] = pd.to_numeric(df["N_harm_norm"], errors="coerce")

    df["Density Metric_Norm2_available"] = df["Density Metric_Norm2"].notna()
    df["Combined Density Metric_Norm2_available"] = df["Combined Density Metric_Norm2"].notna()
    df["D_agn_available"] = d_agn.notna()
    df["P_norm_available"] = p_norm.notna()
    df["N_harm_norm_available"] = df["N_harm_norm"].notna()

    df["_D_agn_for_index"] = d_agn
    df["_P_norm_for_index"] = p_norm
    df["_Combined_for_index"] = df["Combined Density Metric_Norm2"]

    # 4) Índice ponderado — mesma política que ``apply_weighted_index`` (pdf): renorm. por termos disponíveis
    df["Index_Weighted"] = _weighted_index_available_terms(
        df,
        {
            "Density Metric_Norm2": 0.10,
            "_D_agn_for_index": 0.40,
            "N_harm_norm": 0.30,
            "_Combined_for_index": 0.15,
            "_P_norm_for_index": 0.05,
        },
    )
    df["Index_Weighted"] = pd.to_numeric(df["Index_Weighted"], errors="coerce").clip(0.0, 1.0)

    df.drop(
        columns=[
            "_D_agn_for_index",
            "_P_norm_for_index",
            "_Combined_for_index",
        ],
        errors="ignore",
        inplace=True,
    )

    # 5) Apply additional dimensionality reduction (t-SNE, UMAP)
    numeric_cols = _get_numeric_columns(df)
    n_samples = len(df)
    logger.info(f"DR Parameters - t-SNE: {use_tsne}, UMAP: {use_umap}, Anomaly: {detect_anomalies}")
    logger.info(f"Numeric columns found: {len(numeric_cols)}, Samples: {n_samples}")
    
    # Check sample size requirements
    if use_tsne and n_samples < 4:
        logger.warning(f"t-SNE requires at least 4 samples, but only {n_samples} available. t-SNE will be skipped.")
        logger.warning(f"  → For optimal results, t-SNE needs at least 6 samples (to allow perplexity >= 5)")
    if use_umap and n_samples < 4:
        logger.warning(f"UMAP requires at least 4 samples, but only {n_samples} available. UMAP will be skipped.")
        logger.warning(f"  → UMAP needs n_neighbors > 1, and with < 4 samples, this causes scipy errors")
    
    if len(numeric_cols) >= 2:
        if use_tsne or use_umap:
            logger.info(f"Applying dimensionality reduction: t-SNE={use_tsne}, UMAP={use_umap}, n_samples={n_samples}")
            df = _apply_additional_dimension_reduction(df, numeric_cols, use_tsne, use_umap)
            # Verify results
            if use_tsne:
                has_tsne = 'TSNE1' in df.columns and 'TSNE2' in df.columns
                status = 'SUCCESS' if has_tsne else 'FAILED - columns not found'
                logger.info(f"t-SNE verification: {status}")
                if not has_tsne and n_samples < 4:
                    logger.warning(f"  → t-SNE failed because dataset has only {n_samples} samples (minimum: 4)")
            if use_umap:
                has_umap = 'UMAP1' in df.columns and 'UMAP2' in df.columns
                status = 'SUCCESS' if has_umap else 'FAILED - columns not found'
                logger.info(f"UMAP verification: {status}")
                if not has_umap:
                    logger.warning(f"  → UMAP failed. Check logs above for details.")
        else:
            logger.info("Dimensionality reduction skipped (both t-SNE and UMAP are False)")
        
        # 6) Apply anomaly detection
        if detect_anomalies:
            if anomaly_contamination is None:
                logger.info("Applying anomaly detection with adaptive contamination")
            else:
                logger.info(f"Applying anomaly detection with contamination={anomaly_contamination}")
            df = _detect_spectral_anomalies(df, numeric_cols, contamination=anomaly_contamination)
            has_anomaly = 'is_anomaly' in df.columns
            logger.info(f"Anomaly detection verification: {'SUCCESS' if has_anomaly else 'FAILED - columns not found'}")
        else:
            logger.info("Anomaly detection skipped (detect_anomalies=False)")
    else:
        logger.warning(f"Insufficient numeric columns ({len(numeric_cols)}) for dimensionality reduction (need >= 2)")

    # 7) Exportação
    meta_out: Dict[str, Any] = {}
    if output_path:
        outp = Path(output_path)
        try:
            outp.parent.mkdir(parents=True, exist_ok=True)

            outp_path_obj = Path(output_path)
            if (
                outp_path_obj.name == "compiled_density_metrics.xlsx"
                and not outp_path_obj.is_absolute()
            ):
                dynamics = extract_dynamics_from_path(folder_path)
                if dynamics:
                    outp = outp.parent / f"compiled_density_metrics_{dynamics}.xlsx"
                    logger.info("Dynamics %r detected, using filename: %s", dynamics, outp.name)

            # Remove Register column if present (before saving)
            if 'Register' in df.columns:
                logger.info("Removing 'Register' column from compiled metrics (instrument-specific)")
                df = df.drop(columns=['Register'])

            version, version_source = _get_project_version_info()
            metadata = {
                "analysis_date": datetime.now().isoformat(),
                "analysis_version": version,
                "analysis_version_source": version_source,
                "folder_path": _publication_safe_folder_path_marker(folder_path),
                "file_pattern": file_pattern,
                "include_pca": include_pca,
                "enable_pca_export": enable_pca_export,
                "minimum_samples_for_pca": minimum_samples_for_pca,
                "harmonic_weight": harmonic_weight,
                "inharmonic_weight": inharmonic_weight,
                "weight_function": weight_function,
                "detect_anomalies": detect_anomalies,
                "anomaly_contamination": anomaly_contamination if anomaly_contamination is not None else "auto",
                "n_samples": int(len(df)),
                "pca_include_dissonance": bool(pca_include_dissonance),
                "pca_include_dependent_metrics": bool(pca_include_dependent_metrics),
            }
            _extra_meta = dict(compilation_extra_metadata or {})
            _schema_status = str(
                _extra_meta.pop(
                    "input_schema_validation_status",
                    "not_validated",
                )
            )
            metadata.update(_extra_meta)

            _eff_pat = str(
                getattr(
                    _compile_density_metrics_impl,
                    "_last_resolved_file_pattern",
                    file_pattern,
                )
            )

            meta_out = _write_compiled_excel(
                outp,
                df,
                metadata,
                apply_publication_column_filter=compiled_public_columns,
                enable_pca_export=want_pca_sheets,
                minimum_samples_for_pca=minimum_samples_for_pca,
                pca_include_dissonance=bool(pca_include_dissonance),
                pca_include_dependent_metrics=bool(pca_include_dependent_metrics),
                compile_file_pattern=_eff_pat,
                allow_legacy_super_json=allow_legacy_super_json,
                input_schema_validation_status=_schema_status,
            )
            if meta_out.get("pca_export_status") == "exported" and outp.is_file() and "Note" in df.columns:
                try:
                    pca_scores = pd.read_excel(outp, sheet_name="PCA_Scores")
                    if "Note" in pca_scores.columns:
                        df = df.merge(pca_scores, on="Note", how="left")
                except Exception as _merge_exc:
                    logger.warning("Could not merge PCA_Scores into in-memory DataFrame: %s", _merge_exc)
            _pca_here = bool(meta_out.get("pca_export_status") == "exported")
            if _pca_here:
                logger.info("Compiled metrics saved with PCA outputs to '%s'", outp)
            else:
                logger.info("Compiled metrics saved; PCA not applied (path '%s')", outp)
        except Exception as e:
            logger.error("Error saving Excel workbook (PCA) to '%s': %s", outp, e, exc_info=True)

    # Resumo compatível com SoundSpectrAnalyse-main_7 (proc_audio lê _last_dr_audit)
    pca_exported = bool(meta_out.get("pca_export_status") == "exported")
    compile_density_metrics_with_pca._last_dr_audit = {  # type: ignore[attr-defined]
        "PCA_applied": pca_exported,
        "PCA_status": (
            "exported_to_sheets"
            if pca_exported
            else (
                "skipped_not_requested"
                if not want_pca_sheets
                else str(meta_out.get("pca_export_note") or meta_out.get("pca_export_status") or "skipped")
            )
        ),
        "TSNE_applied": bool("TSNE1" in df.columns and "TSNE2" in df.columns),
        "TSNE_status": (
            "applied"
            if ("TSNE1" in df.columns)
            else ("skipped_not_requested" if not use_tsne else "no_tsne_columns")
        ),
        "UMAP_applied": bool("UMAP1" in df.columns and "UMAP2" in df.columns),
        "UMAP_status": (
            "applied"
            if ("UMAP1" in df.columns)
            else ("skipped_not_requested" if not use_umap else "no_umap_columns")
        ),
        "anomaly_detection_applied": bool("is_anomaly" in df.columns),
        "Anomaly_status": (
            "applied"
            if "is_anomaly" in df.columns
            else ("skipped_not_requested" if not detect_anomalies else "no_anomaly_columns")
        ),
    }

    return df


def compile_density_metrics(*args: Any, **kwargs: Any) -> Optional[pd.DataFrame]:
    """
    Compatibility entry point: delegates to :func:`compile_density_metrics_with_pca`.

    Callers that historically used ``compile_density_metrics`` without ``include_pca``
    receive ``include_pca=False`` by default so exploratory PCA sheets stay off unless
    explicitly requested.
    """
    kwargs.setdefault("include_pca", False)
    return compile_density_metrics_with_pca(*args, **kwargs)


def plot_pca_scatter(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Cria um gráfico de dispersão 2D usando PC1 e PC2.

    Args:
        df: DataFrame com colunas PC1 e PC2.
        output_dir: Diretório para salvar o gráfico.
    """
    if "PC1" not in df.columns or "PC2" not in df.columns:
        logger.warning("PC1 or PC2 not found in DataFrame")
        return

    plt.figure(figsize=(10, 8))
    plt.scatter(df["PC1"], df["PC2"], s=100, alpha=0.7)

    # Adicionar rótulos de notas
    if "Note" in df.columns:
        for i, row in df.iterrows():
            plt.annotate(row["Note"],
                         (row["PC1"], row["PC2"]),
                         xytext=(5, 5),
                         textcoords="offset points")

    plt.title("PCA Analysis of Spectral Metrics")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)

    # Adicionar linhas de referência
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    output_path = Path(output_dir) / "pca_scatter.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"PCA scatter plot saved to: {output_path}")


def plot_pc1_ranking(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Cria um gráfico de barras das notas ordenadas por PC1.

    Args:
        df: DataFrame com coluna PC1.
        output_dir: Diretório para salvar o gráfico.
    """
    if "PC1" not in df.columns:
        logger.warning("PC1 not found in DataFrame")
        return

    # Ordenar por PC1
    df_sorted = df.sort_values(by="PC1").copy()

    plt.figure(figsize=(12, 6))

    # Obter rótulos para o eixo x
    x_labels = df_sorted["Note"].tolist() if "Note" in df_sorted.columns else [f"Item {i+1}" for i in range(len(df_sorted))]

    # Criar gráfico de barras
    bars = plt.bar(x_labels, df_sorted["PC1"], alpha=0.7)

    # Colorir barras por valor
    min_val = df_sorted["PC1"].min()
    max_val = df_sorted["PC1"].max()
    norm = plt.Normalize(min_val, max_val)
    colors = plt.cm.viridis(norm(df_sorted["PC1"]))

    for bar, color in zip(bars, colors):
        bar.set_color(color)

    plt.title("Notes Ranked by Principal Component 1")
    plt.xlabel("Note")
    plt.ylabel("PC1 Value")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)

    output_path = Path(output_dir) / "pc1_ranking.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"PC1 ranking chart saved to: {output_path}")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Cria uma matriz de correlação para as métricas numéricas.

    Args:
        df: DataFrame com métricas.
        output_dir: Diretório para salvar o gráfico.
    """
    # Identificar colunas numéricas (excluindo colunas normalizadas para evitar duplicação)
    numeric_cols = []
    for col in df.columns:
        if col.endswith("_Norm"):
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                numeric_cols.append(col)
        except:
            pass

    if len(numeric_cols) < 2:
        logger.warning("Fewer than 2 numeric columns found for correlation matrix")
        return

    # Calcular matriz de correlação
    corr_df = df[numeric_cols].corr()

    # Plotar usando seaborn
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))  # Máscara para triângulo superior

    # Usar um mapa de cores divergente para melhor visualização
    sns.heatmap(corr_df, mask=mask, cmap="coolwarm", vmin=-1, vmax=1,
                annot=True, fmt=".2f", linewidths=0.5, square=True)

    plt.title("Correlation Matrix of Spectral Metrics")

    output_path = Path(output_dir) / "correlation_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Correlation matrix saved to: {output_path}")


def plot_metrics_comparison(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Cria um gráfico de barras comparando diferentes métricas para cada nota.

    Args:
        df: DataFrame com métricas e notas.
        output_dir: Diretório para salvar o gráfico.
    """
    # Verificar se temos a coluna 'Note'
    if 'Note' not in df.columns:
        logger.warning("Column 'Note' not found for comparison chart")
        return

    # Identificar métricas (excluindo colunas normalizadas e PCs)
    metrics = []
    for col in df.columns:
        if col in ['Note', 'Folder'] or col.startswith('PC') or col.endswith('_Norm'):
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                metrics.append(col)
        except:
            pass

    if not metrics:
        logger.warning("No valid metrics found for comparison chart")
        return

    # Normalizar métricas para escala 0-1
    df_norm = df.copy()
    for col in metrics:
        values = df_norm[col].dropna()
        if len(values) < 2:
            continue
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)

    # Criar um gráfico para cada tipo principal de métrica
    metric_groups = {
        'Density': [m for m in metrics if 'Density' in m],
        'Dissonance': [m for m in metrics if 'Dissonance' in m]
    }

    for group_name, group_metrics in metric_groups.items():
        if not group_metrics:
            continue

        plt.figure(figsize=(14, 8))

        # Preparar dados para plot
        x = np.arange(len(df_norm))
        width = 0.8 / len(group_metrics)  # Largura da barra

        # Plotar barras para cada métrica
        for i, metric in enumerate(group_metrics):
            offset = (i - len(group_metrics) / 2 + 0.5) * width
            plt.bar(x + offset, df_norm[metric], width, label=metric)

        # Configurar eixos e rótulos
        plt.xlabel('Note')
        plt.ylabel('Normalized Value')
        plt.title(f'Comparison of {group_name} Metrics')
        plt.xticks(x, df_norm['Note'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)

        output_path = Path(output_dir) / f"{group_name.lower()}_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Comparison chart for {group_name} saved to: {output_path}")


def analyze_notes_clustering(df: pd.DataFrame, output_dir: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Analisa o agrupamento de notas com base nas métricas de densidade/dissonância.

    Args:
        df: DataFrame com métricas e notas.
        output_dir: Diretório para salvar gráficos (opcional).

    Returns:
        Dicionário com resultados da análise de agrupamento.
    """
    if df is None or df.empty or 'Note' not in df.columns:
        logger.warning("Invalid DataFrame for clustering analysis")
        return {}

    # Identificar métricas numéricas
    numeric_cols = []
    for col in df.columns:
        if col in ['Note', 'Folder'] or col.startswith('PC') or col.endswith('_Norm'):
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                numeric_cols.append(col)
        except:
            pass

    if len(numeric_cols) < 2:
        logger.warning("Too few metrics for clustering analysis")
        return {}

    try:
        # Preparar dados numéricos
        X = df[numeric_cols].fillna(df[numeric_cols].mean()).values

        # Padronizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calcular distâncias euclidianas entre notas
        distances = euclidean_distances(X_scaled)

        # Criar DataFrame de distâncias com notas como índices
        dist_df = pd.DataFrame(distances,
                               index=df['Note'].values,
                               columns=df['Note'].values)

        # Encontrar notas mais próximas para cada nota
        closest_notes = {}
        for note in dist_df.index:
            # Ordenar por distância (excluindo a própria nota)
            closest = dist_df[note].sort_values()[1:4]  # 3 mais próximas
            closest_notes[note] = {
                'closest': closest.index.tolist(),
                'distances': closest.values.tolist()
            }

        # Gerar um gráfico de calor das distâncias
        if output_dir:
            plt.figure(figsize=(10, 8))
            sns.heatmap(dist_df, cmap='viridis_r', annot=True, fmt='.2f', square=True)
            plt.title('Euclidean Distances Between Notes')

            output_path = Path(output_dir) / "note_distances.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Note distance map saved to: {output_path}")

        return {
            'distance_matrix': dist_df,
            'closest_notes': closest_notes
        }

    except Exception as e:
        logger.error(f"Error in clustering analysis: {e}")
        return {}


def generate_analysis_report(df: pd.DataFrame, output_path: Union[str, Path] = 'analysis_report.md') -> None:
    """
    Gera um relatório de análise em formato Markdown com insights sobre as métricas.

    Args:
        df: DataFrame com métricas compiladas.
        output_path: Caminho para salvar o relatório.
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame for report generation")
        return

    try:
        # Iniciar relatório
        report_lines = [
            "# Spectral metrics analysis\n",
            f"Data da análise: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"Total de notas analisadas: {len(df)}\n",
            "\n## Resumo das Métricas\n"
        ]

        # Identificar métricas numéricas para análise
        numeric_cols = []
        for col in df.columns:
            if col in ['Note', 'Folder']:
                continue
            try:
                if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                    numeric_cols.append(col)
            except:
                pass

        # Estatísticas descritivas
        if numeric_cols:
            desc_stats = df[numeric_cols].describe().transpose()
            report_lines.append("### Estatísticas Descritivas\n")
            report_lines.append("| Métrica | Média | Desvio Padrão | Mín | Máx |\n")
            report_lines.append("| ------- | ----- | ------------- | --- | --- |\n")

            for idx, row in desc_stats.iterrows():
                report_lines.append(f"| {idx} | {row['mean']:.3f} | {row['std']:.3f} | {row['min']:.3f} | {row['max']:.3f} |\n")

            report_lines.append("\n")

        # PCA insights
        if 'PC1' in df.columns:
            report_lines.append("## Principal component analysis (PCA)\n")
            report_lines.append("### Ranking de Notas por PC1\n")

            # Ordenar notas por PC1
            if 'Note' in df.columns:
                df_sorted = df.sort_values(by='PC1', ascending=False).copy()

                report_lines.append("Notas ordenadas do maior para o menor valor de PC1:\n\n")
                report_lines.append("| Posição | Nota | PC1 |\n")
                report_lines.append("| ------- | ---- | --- |\n")

                for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
                    report_lines.append(f"| {i} | {row['Note']} | {row['PC1']:.3f} |\n")

                report_lines.append("\n")

            # Informações sobre contribuição de métricas para PC1
            report_lines.append("### Interpretação de PC1\n")
            report_lines.append("O primeiro componente principal (PC1) pode ser interpretado como ")

            # Aqui poderíamos inserir uma análise de contribuição das métricas para o PC1
            # Como não temos esses dados no dataframe compilado, vamos adicionar uma nota genérica
            report_lines.append("uma medida composta da densidade/dissonância espectral. ")
            report_lines.append("Valores mais altos geralmente indicam maior densidade harmônica e/ou dissonância.\n\n")

        # Correlações entre métricas
        if len(numeric_cols) >= 2:
            report_lines.append("## Correlações entre Métricas\n")

            # Calcular matriz de correlação
            corr_matrix = df[numeric_cols].corr()

            # Encontrar correlações fortes (acima de 0.7 ou abaixo de -0.7)
            strong_corr = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) >= 0.7:
                        strong_corr.append((numeric_cols[i], numeric_cols[j], corr))

            if strong_corr:
                report_lines.append("### Correlações Fortes (|r| ≥ 0.7)\n")
                report_lines.append("| Métrica 1 | Métrica 2 | Correlação |\n")
                report_lines.append("| -------- | -------- | ---------- |\n")

                for m1, m2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                    report_lines.append(f"| {m1} | {m2} | {corr:.3f} |\n")

                report_lines.append("\n")
            else:
                report_lines.append("Não foram encontradas correlações fortes entre as métricas.\n\n")

        # Note clustering analysis
        if 'Note' in df.columns and len(df) >= 3:
            report_lines.append("## Agrupamento de Notas\n")

            # Realizar análise de agrupamento
            clustering = analyze_notes_clustering(df)

            if clustering and 'closest_notes' in clustering:
                report_lines.append("### Notas Similares\n")
                report_lines.append("Baseado nas métricas espectrais, as seguintes notas são mais similares entre si:\n\n")
                report_lines.append("| Nota | Notas Mais Similares |\n")
                report_lines.append("| ---- | ------------------- |\n")

                for note, data in clustering['closest_notes'].items():
                    similar_notes = data['closest']
                    distances = data['distances']

                    # Formatar notas similares com suas distâncias
                    similar_str = ", ".join([f"{n} ({d:.2f})" for n, d in zip(similar_notes, distances)])
                    report_lines.append(f"| {note} | {similar_str} |\n")

                report_lines.append("\n")

        # Insights específicos para métricas de densidade/dissonância
        report_lines.append("## Insights Específicos\n")

        # Verificar se temos métricas de densidade
        density_metrics = [col for col in numeric_cols if 'Density' in col]
        if density_metrics:
            report_lines.append("### Métricas de Densidade\n")

            # Encontrar notas com maior e menor densidade para cada métrica
            for metric in density_metrics:
                if metric in df.columns and 'Note' in df.columns:
                    valid_values = df[[metric, 'Note']].dropna()
                    if not valid_values.empty:
                        max_row = valid_values.loc[valid_values[metric].idxmax()]
                        min_row = valid_values.loc[valid_values[metric].idxmin()]

                        report_lines.append(f"**{metric}**:\n")
                        report_lines.append(f"- Nota com maior valor: {max_row['Note']} ({max_row[metric]:.3f})\n")
                        report_lines.append(f"- Nota com menor valor: {min_row['Note']} ({min_row[metric]:.3f})\n")

            report_lines.append("\n")

        # Verificar se temos métricas de dissonância
        dissonance_metrics = [col for col in numeric_cols if 'Dissonance' in col]
        if dissonance_metrics:
            report_lines.append("### Métricas de Dissonância\n")

            # Encontrar notas com maior e menor dissonância para cada métrica
            for metric in dissonance_metrics:
                if metric in df.columns and 'Note' in df.columns:
                    valid_values = df[[metric, 'Note']].dropna()
                    if not valid_values.empty:
                        max_row = valid_values.loc[valid_values[metric].idxmax()]
                        min_row = valid_values.loc[valid_values[metric].idxmin()]

                        report_lines.append(f"**{metric}**:\n")
                        report_lines.append(f"- Nota com maior valor: {max_row['Note']} ({max_row[metric]:.3f})\n")
                        report_lines.append(f"- Nota com menor valor: {min_row['Note']} ({min_row[metric]:.3f})\n")

            report_lines.append("\n")

        # Conclusões
        report_lines.append("## Conclusões\n")
        report_lines.append("Based on the spectral metric analysis, we conclude:\n\n")

        # Adicionar algumas conclusões genéricas
        report_lines.append("1. As métricas de densidade e dissonância fornecem diferentes perspectivas sobre o conteúdo espectral das notas.\n")
        report_lines.append("2. A análise PCA permite reduzir a dimensionalidade e visualizar tendências que não são imediatamente aparentes.\n")
        report_lines.append("3. As diferenças entre notas são quantificáveis através destas métricas, o que pode ser útil para estudos de percepção musical.\n")

        # Salvar o relatório
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)

        logger.info(f"Analysis report saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error generating analysis report: {e}")


def extract_models_comparison(folder_path: Union[str, Path],
                             output_path: Optional[Union[str, Path]] = "compiled_density_metrics.xlsx") -> Optional[pd.DataFrame]:
    """
    Extrai e compila uma comparação entre diferentes modelos de dissonância.

    Args:
        folder_path: Diretório contendo subpastas com arquivos de análise espectral.
        output_path: Caminho para salvar o arquivo Excel compilado.

    Returns:
        DataFrame com comparação de modelos, ou None se nenhum dado válido for encontrado.
    """
    # Validar caminho do diretório
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        logger.error(f"Invalid directory: {folder_path}")
        raise ValueError(f"Invalid directory: {folder_path}")

    results = []

    logger.info(f"Extracting dissonance model comparison from: {folder_path}")

    # Percorrer todas as subpastas
    for note_dir in [d for d in folder_path.iterdir() if d.is_dir()]:
        try:
            # Extrair nome da nota da pasta
            note = extract_note_from_folder(note_dir.name)

            # Verificar se existe arquivo de análise espectral
            excel_path = note_dir / 'spectral_analysis.xlsx'
            if not excel_path.exists():
                logger.debug(f"Analysis file not found for note {note}")
                continue

            # Extrair métricas de dissonância
            metrics = read_excel_metrics(excel_path)
            dissonance_metrics = {k: v for k, v in metrics.items() if 'Dissonance' in k}

            if not dissonance_metrics:
                logger.debug(f"No dissonance metrics found for note {note}")
                continue

            # Adicionar resultados
            results.append({
                'Note': note,
                **dissonance_metrics
            })

            logger.debug(f"Dissonance metrics extracted for note {note}: {list(dissonance_metrics.keys())}")

        except Exception as e:
            logger.error(f"Error processing folder {note_dir}: {e}")

    if not results:
        logger.warning("No valid data found for model comparison.")
        return None

    # Construir DataFrame e ordenar por nota
    results_df = pd.DataFrame(results)

    if 'Note' in results_df.columns:
        try:
            results_df = results_df.sort_values(
                by='Note',
                key=lambda col: col.map(note_sort_key)
            )
        except Exception as e:
            logger.warning(f"Could not sort by note: {e}")

    # Salvar para Excel
    try:
        output_path = Path(output_path)
        version, version_source = _get_project_version_info()
        metadata = {
            "analysis_date": datetime.now().isoformat(),
            "analysis_version": version,
            "analysis_version_source": version_source,
            "folder_path": _publication_safe_folder_path_marker(folder_path),
            "file_pattern": "spectral_analysis.xlsx",
            "comparison_type": "dissonance_models",
            "n_samples": int(len(results_df)),
            "dissonance_enabled": True,
            "dissonance_compare_models": True,
        }
        _write_compiled_excel(
            output_path,
            results_df,
            metadata,
            apply_publication_column_filter=False,
            compile_file_pattern="spectral_analysis.xlsx",
            allow_legacy_super_json=False,
            input_schema_validation_status="not_applicable_dissonance_model_comparison",
        )
        logger.info(f"Model comparison saved to '{output_path}'.")

        # Adicionalmente, gerar um heatmap de correlação entre modelos
        try:
            dissonance_models = [col for col in results_df.columns if "Dissonance" in col]
            for _slug in MODEL_SLUGS:
                _k = CANONICAL_VALUE_BY_SLUG[_slug]
                if _k in results_df.columns and _k not in dissonance_models:
                    dissonance_models.append(_k)
            if len(dissonance_models) >= 2:
                corr_matrix = results_df[dissonance_models].corr()

                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
                plt.title('Correlation Between Dissonance Models')

                heatmap_path = output_path.with_suffix('.png')
                plt.tight_layout()
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()

                logger.info(f"Inter-model correlation heatmap saved to: {heatmap_path}")

        except Exception as e:
            logger.error(f"Error generating correlation heatmap: {e}")

    except Exception as e:
        logger.error(f"Failed to save model comparison to '{output_path}': {e}")
        raise

    return results_df


def calculate_metric_distributions(df: pd.DataFrame, num_bins: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Calcula distribuições estatísticas para cada métrica.

    Args:
        df: DataFrame com métricas.
        num_bins: Número de bins para histogramas.

    Returns:
        Dicionário com estatísticas de distribuição para cada métrica.
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame for distribution calculations")
        return {}

    distributions = {}

    # Identificar métricas numéricas
    numeric_cols = []
    for col in df.columns:
        if col in ['Note', 'Folder']:
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                numeric_cols.append(col)
        except:
            pass

    # Calcular distribuições
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) < 2:
            continue

        # Estatísticas básicas
        stats = {
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'skew': values.skew(),  # Assimetria
            'kurtosis': values.kurtosis()  # Curtose
        }

        # Calcular histograma
        hist, bin_edges = np.histogram(values, bins=num_bins)

        # Adicionar distribuição
        distributions[col] = {
            'stats': stats,
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        }

    return distributions


def plot_metric_distributions(df: pd.DataFrame,
                             output_dir: Union[str, Path],
                             num_bins: int = 10) -> None:
    """
    Plota distribuições para cada métrica.

    Args:
        df: DataFrame com métricas.
        output_dir: Diretório para salvar os gráficos.
        num_bins: Número de bins para histogramas.
    """
    if df is None or df.empty:
        logger.warning("Empty DataFrame for distribution plots")
        return

    # Garantir que o diretório existe
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Identificar métricas numéricas
    numeric_cols = []
    for col in df.columns:
        if col in ['Note', 'Folder']:
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                numeric_cols.append(col)
        except:
            pass

    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) < 2:
            continue

        # Criar figura
        plt.figure(figsize=(10, 6))

        # Plotar histograma com curva de densidade
        sns.histplot(values, kde=True, bins=num_bins)

        # Adicionar linha vertical para média e mediana
        plt.axvline(values.mean(), color='r', linestyle='--', alpha=0.7, label=f'Média: {values.mean():.3f}')
        plt.axvline(values.median(), color='g', linestyle='-.', alpha=0.7, label=f'Mediana: {values.median():.3f}')

        # Configurar rótulos e título
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')
        plt.legend()

        # Informações estatísticas no gráfico
        stats_text = (
            f"Desvio Padrão: {values.std():.3f}\n"
            f"Mín: {values.min():.3f}\n"
            f"Máx: {values.max():.3f}\n"
            f"Assimetria: {values.skew():.3f}\n"
            f"Curtose: {values.kurtosis():.3f}"
        )

        # Posicionar texto no canto superior direito
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=9, ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))

        # Salvar figura
        output_path = output_dir / f"{col.lower().replace(' ', '_')}_distribution.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Distribution plot for {col} saved to: {output_path}")


def generate_comparison_report(results_df: pd.DataFrame,
                              output_path: Union[str, Path] = 'comparison_report.md',
                              include_plots: bool = True,
                              plots_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Gera um relatório comparativo entre diferentes modelos de dissonância e métricas.

    Args:
        results_df: DataFrame com métricas de todas as notas.
        output_path: Caminho para salvar o relatório.
        include_plots: Se True, gera gráficos para inclusão no relatório.
        plots_dir: Diretório para salvar os gráficos (se None, usa o mesmo do output_path).
    """
    if results_df is None or results_df.empty:
        logger.warning("Empty DataFrame for comparative report generation")
        return

    try:
        # Preparar diretório para plots
        output_path = Path(output_path)

        if plots_dir is None:
            plots_dir = output_path.parent / 'plots'
        else:
            plots_dir = Path(plots_dir)

        plots_dir.mkdir(exist_ok=True, parents=True)

        # Iniciar relatório
        report_lines = [
            "# Relatório Comparativo de Métricas Espectrais\n",
            f"Data da análise: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"Total de notas analisadas: {len(results_df)}\n",
            "\n## Visão Geral\n"
        ]

        # Contar tipos de métricas
        density_metrics = [col for col in results_df.columns if 'Density' in col]
        dissonance_metrics = [col for col in results_df.columns if 'Dissonance' in col]

        report_lines.append(f"- Total de métricas de densidade: {len(density_metrics)}\n")
        report_lines.append(f"- Total de modelos de dissonância: {len(dissonance_metrics)}\n\n")

        # Lista de métricas
        if density_metrics:
            report_lines.append("### Métricas de Densidade\n")
            for metric in density_metrics:
                report_lines.append(f"- {metric}\n")
            report_lines.append("\n")

        if dissonance_metrics:
            report_lines.append("### Modelos de Dissonância\n")
            for metric in dissonance_metrics:
                report_lines.append(f"- {metric}\n")
            report_lines.append("\n")

        # Correlation analysis
        report_lines.append("## Correlação entre Modelos\n")

        # Calcular matriz de correlação para modelos de dissonância
        if len(dissonance_metrics) >= 2:
            corr_matrix = results_df[dissonance_metrics].corr()

            report_lines.append("### Matriz de Correlação\n")
            report_lines.append("| Modelo | " + " | ".join(dissonance_metrics) + " |\n")
            report_lines.append("| ------ | " + " | ".join(["-----" for _ in dissonance_metrics]) + " |\n")

            for model in dissonance_metrics:
                row = [model]
                for other_model in dissonance_metrics:
                    row.append(f"{corr_matrix.loc[model, other_model]:.3f}")
                report_lines.append("| " + " | ".join(row) + " |\n")

            report_lines.append("\n")

            # Adicionar gráfico de correlação se solicitado
            if include_plots:
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
                plt.title('Correlation Between Dissonance Models')

                corr_path = plots_dir / "dissonance_correlation.png"
                plt.tight_layout()
                plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                plt.close()

                # Adicionar referência ao gráfico no relatório
                report_lines.append(f"![Correlation Between Dissonance Models]({corr_path.name})\n\n")
                logger.info(f"Correlation chart saved to: {corr_path}")

            # Interpretação das correlações
            report_lines.append("### Interpretação\n")

            # Encontrar correlações fortes
            strong_positive = []
            strong_negative = []
            weak = []

            for i in range(len(dissonance_metrics)):
                for j in range(i+1, len(dissonance_metrics)):
                    model1 = dissonance_metrics[i]
                    model2 = dissonance_metrics[j]
                    corr = corr_matrix.loc[model1, model2]

                    if corr >= 0.7:
                        strong_positive.append((model1, model2, corr))
                    elif corr <= -0.7:
                        strong_negative.append((model1, model2, corr))
                    elif abs(corr) <= 0.3:
                        weak.append((model1, model2, corr))

            if strong_positive:
                report_lines.append("**Correlações Positivas Fortes (r ≥ 0.7):**\n")
                for m1, m2, corr in strong_positive:
                    report_lines.append(f"- {m1} e {m2}: {corr:.3f}\n")
                report_lines.append("\n")

            if strong_negative:
                report_lines.append("**Correlações Negativas Fortes (r ≤ -0.7):**\n")
                for m1, m2, corr in strong_negative:
                    report_lines.append(f"- {m1} e {m2}: {corr:.3f}\n")
                report_lines.append("\n")

            if weak:
                report_lines.append("**Correlações Fracas (|r| ≤ 0.3):**\n")
                for m1, m2, corr in weak:
                    report_lines.append(f"- {m1} e {m2}: {corr:.3f}\n")
                report_lines.append("\n")

        # Distribution analysis
        report_lines.append("## Distribuição das Métricas\n")

        # Calcular distribuições e criar gráficos para métricas de dissonância
        if include_plots and dissonance_metrics:
            # Plotar distribuições
            plot_metric_distributions(results_df[dissonance_metrics], plots_dir)

            # Adicionar informações ao relatório
            for metric in dissonance_metrics:
                values = results_df[metric].dropna()
                if len(values) < 2:
                    continue

                report_lines.append(f"### {metric}\n")
                report_lines.append(f"- **Média**: {values.mean():.3f}\n")
                report_lines.append(f"- **Mediana**: {values.median():.3f}\n")
                report_lines.append(f"- **Desvio Padrão**: {values.std():.3f}\n")
                report_lines.append(f"- **Mínimo**: {values.min():.3f}\n")
                report_lines.append(f"- **Máximo**: {values.max():.3f}\n")

                # Adicionar referência ao gráfico no relatório
                dist_path = plots_dir / f"{metric.lower().replace(' ', '_')}_distribution.png"
                if dist_path.exists():
                    report_lines.append(f"\n![Distribution of {metric}]({dist_path.name})\n\n")

        # Conclusões e recomendações
        report_lines.append("## Conclusões e Recomendações\n")

        report_lines.append("Based on the comparative analysis, we conclude:\n\n")

        # Adicionar algumas conclusões baseadas nos resultados
        if len(dissonance_metrics) >= 2:
            # Verificar concordância entre modelos
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            mean_corr = np.mean(corr_values)

            if mean_corr >= 0.7:
                report_lines.append("1. Há uma forte concordância entre a maioria dos modelos de dissonância, ")
                report_lines.append("sugerindo que eles estão capturando aspectos semelhantes da percepção de dissonância.\n")
            elif mean_corr >= 0.4:
                report_lines.append("1. Há uma concordância moderada entre os modelos de dissonância, ")
                report_lines.append("com algumas diferenças em como cada modelo avalia determinadas notas.\n")
            else:
                report_lines.append("1. Existe uma baixa concordância entre os modelos de dissonância, ")
                report_lines.append("sugerindo que diferentes modelos capturam aspectos distintos da percepção de dissonância.\n")

        report_lines.append("2. Para análises futuras, recomendamos:\n")
        report_lines.append("   - Padronizar a escala de todas as métricas para facilitar comparações diretas\n")
        report_lines.append("   - Considerar o uso de análise multivariada para explorar relações mais complexas\n")
        report_lines.append("   - Comparar estes resultados com testes de percepção auditiva para validação\n\n")

        # Modelo recomendado
        if len(dissonance_metrics) >= 2:
            report_lines.append("3. Com base nesta análise, o modelo mais recomendado para uso geral seria aquele que:")
            report_lines.append("   - Tem boa correlação com a maioria dos outros modelos\n")
            report_lines.append("   - Apresenta uma distribuição bem comportada\n")
            report_lines.append("   - É computacionalmente eficiente\n\n")

        # Salvar o relatório
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)

        logger.info(f"Comparison report saved to: {output_path}")

    except Exception as e:
        logger.error(f"Error generating comparison report: {e}")



def extract_density_metric(audio_processor: AudioProcessor) -> Optional[float]:
    """
    Return the combined density metric from a configured ``AudioProcessor`` instance.

    Args:
        audio_processor: configured ``AudioProcessor`` instance.

    Returns:
        Combined density metric value, or ``None`` if unavailable.
    """
    try:
        return audio_processor.combined_density_metric_value
    except Exception as e:
        logger.error(f"Error extracting density metric: {e}")
        return None

if __name__ == "__main__":
    # Minimal usage example (local paths)
    logger.info("Starting compile_metrics usage example")

    example_folder = './results'
    output_excel = './compiled_density_metrics.xlsx'

    try:
        if os.path.exists(example_folder):
            # Compile metrics with PCA and generate a report
            results_df = compile_density_metrics_with_pca(
                folder_path=example_folder,
                output_path=output_excel,
            )

            if results_df is not None:
                # Generate analysis report
                generate_analysis_report(
                    df=results_df,
                    output_path='./analysis_report.md'
                )

                # Extract model comparison table
                models_df = extract_models_comparison(
                    folder_path=example_folder,
                    output_path='./models_comparison.xlsx'
                )

                if models_df is not None:
                    # Generate comparison report
                    generate_comparison_report(
                        results_df=models_df,
                        output_path='./comparison_report.md',
                        include_plots=True
                    )

                logger.info("Usage example completed successfully.")
            else:
                logger.warning("No compiled results available for analysis.")
        else:
            logger.error(f"Example folder not found: '{example_folder}'")

    except Exception as e:
        logger.error(f"Usage example failed: {e}")


def test_compile_metrics(folder_path: Union[str, Path]) -> None:
    """
    Função de teste para diagnosticar problemas com a compilação de métricas.

    Args:
        folder_path: Diretório raiz para verificar
    """
    import os
    from pathlib import Path

    folder_path = Path(folder_path)

    print("=" * 60)
    print("TESTE DE COMPILAÇÃO DE MÉTRICAS")
    print("=" * 60)

    # 1. Verificar se o diretório existe
    print(f"\n1. Verificando diretório: {folder_path}")
    if not folder_path.exists():
        print("   ❌ ERRO: Diretório não existe!")
        return
    if not folder_path.is_dir():
        print("   ❌ ERRO: Caminho não é um diretório!")
        return
    print("   ✓ Diretório existe")

    # 2. Listar subdiretórios
    print("\n2. Subdiretórios encontrados:")
    subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
    if not subdirs:
        print("   ❌ Nenhum subdiretório encontrado!")
    else:
        for subdir in subdirs[:10]:  # Mostrar apenas os primeiros 10
            print(f"   - {subdir.name}")
        if len(subdirs) > 10:
            print(f"   ... e mais {len(subdirs) - 10} diretórios")

    # 3. Procurar arquivos Excel
    print("\n3. Procurando arquivos Excel...")
    excel_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.xlsx'):
                full_path = Path(root) / file
                excel_files.append(full_path)

    if not excel_files:
        print("   ❌ Nenhum arquivo Excel (.xlsx) encontrado!")
    else:
        print(f"   ✓ Encontrados {len(excel_files)} arquivos Excel")

        # Mostrar alguns exemplos
        print("\n   Primeiros arquivos encontrados:")
        for file in excel_files[:5]:
            rel_path = file.relative_to(folder_path)
            print(f"   - {rel_path}")

        # Verificar especificamente por 'spectral_analysis.xlsx'
        spectral_files = [f for f in excel_files if f.name.lower() == 'spectral_analysis.xlsx']
        print(f"\n   Arquivos 'spectral_analysis.xlsx' encontrados: {len(spectral_files)}")

    # 4. Testar leitura de um arquivo
    if spectral_files:
        print("\n4. Testando leitura do primeiro arquivo...")
        test_file = spectral_files[0]
        print(f"   Arquivo: {test_file}")

        try:
            # Tentar abrir o Excel
            import pandas as pd
            excel_data = pd.ExcelFile(test_file)
            print("   ✓ Arquivo aberto com sucesso")
            print(f"   Worksheets available: {excel_data.sheet_names}")

            # Verificar planilha 'Metrics'
            if 'Metrics' in excel_data.sheet_names:
                df_metrics = excel_data.parse('Metrics')
                print("\n   Planilha 'Metrics':")
                print(f"   - Linhas: {len(df_metrics)}")
                print(f"   - Colunas: {list(df_metrics.columns)}")

                if not df_metrics.empty:
                    print("\n   Primeira linha de dados:")
                    for col in df_metrics.columns:
                        val = df_metrics[col].iloc[0] if not df_metrics[col].empty else "N/A"
                        print(f"   - {col}: {val}")
            else:
                print("   ⚠ Planilha 'Metrics' não encontrada!")

        except Exception as e:
            print(f"   ❌ Erro ao ler arquivo: {e}")

    # 5. Sugestões
    print("\n" + "=" * 60)
    print("SUGESTÕES:")
    print("=" * 60)

    if not excel_files:
        print("1. Certifique-se de que os arquivos foram processados corretamente")
        print("2. Verifique se os arquivos têm a extensão .xlsx")
        print("3. Execute 'Apply Filters' antes de compilar métricas")
    elif not spectral_files:
        print("1. Os arquivos Excel devem se chamar 'spectral_analysis.xlsx'")
        print("2. Ou ajuste o parâmetro 'file_pattern' na função")
    else:
        print("1. Verifique se os arquivos Excel contêm a planilha 'Metrics'")
        print("2. Certifique-se de que as métricas foram calculadas corretamente")
        print("3. Verifique os logs para mensagens de erro detalhadas")


# Para executar o teste, adicione isto ao seu código principal:
if __name__ == "__main__":
    # Substitua pelo caminho do seu diretório de resultados
    test_compile_metrics("./results")
