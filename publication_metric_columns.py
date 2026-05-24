"""Optional column allowlist for publication-style Excel exports (ported from newer tree).

Default policy: ``compile_metrics._write_compiled_excel`` aplica o filtro (ver ``compile_metrics``).
Para desativar: ``_write_compiled_excel(..., apply_publication_column_filter=False)`` ou
``compile_density_metrics(..., compiled_public_columns=False)``.

The allowlist is the union of (a) compact "public" columns used in newer releases and
(b) columns emitted by this branch's per-note ``Metrics`` sheet so mixed batches do not
lose main-6-native fields when filtering is enabled.
"""

from __future__ import annotations

import re
from typing import List

import pandas as pd

# Newer-release public columns (PCA norms, aggregated norms, etc.)
_V7_STYLE_PUBLIC: frozenset[str] = frozenset(
    (
        "Density Metric",
        "Density Metric_Norm",
        "Density Metric_Norm2",
        "Filtered Components",
        "Highest Harmonic (Hz)",
        "Index_Weighted",
        "Lowest Harmonic (Hz)",
        "N_harm_norm",
        "Note",
        "Tier",
        "Total weighted partial activity (H+IH+noise)",
        "Total weighted partial activity (H+IH+noise)_Norm",
        "Weight P",
        "Weight R",
        "Weighted Combined Metric",
        "Weighted Combined Metric_Norm",
        "f0_blind_confidence",
        "f0_blind_hz",
        "f0_blind_method",
        "f0_prior_hz",
        "harmonic_bin_count",
        "inharmonic_bin_count",
        "partial_density_component_count_harmonic",
        "partial_density_component_count_harmonic_Norm",
        "partial_density_effective_components_Norm",
        "partial_density_ground_noise_power_Norm",
        "partial_density_harmonic_power_total_Norm",
        "partial_density_inharmonic_power_total_Norm",
        "sethares Dissonance",
        "sethares Dissonance_Norm",
        "subbass_bin_count",
    )
)

# Typical Metrics columns from main-6 ``proc_audio`` (single-row export)
_MAIN6_METRICS_SHEET: frozenset[str] = frozenset(
    (
        "canonical_density_v5_adapted",
        "density_normalized_global",
        "density_per_component",
        "discrete_metric_d3",
        "discrete_metric_d10",
        "discrete_metric_d17",
        "discrete_metric_d24",
        "density_source_formula",
        "density_normalization_scope",
        "density_normalization_denominator",
        "density_formula_version",
        "effective_partial_density",
        "harmonic_occupancy_ratio",
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
        "harmonic_effective_power_density_normalized",
        "core_harmonic_energy_ratio",
        "core_residual_energy_ratio",
        "core_subbass_energy_ratio",
        "residual_log_frequency_occupancy",
        "residual_energy_ratio",
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
        "linear_amplitude_fraction_inharmonic_of_HI",
        "linear_amplitude_fraction_nonharmonic_of_total",
        "linear_amplitude_batch_alignment_factor",
        "Soma_A_linear_harmonicos",
        "Soma_A_linear_inarmonicos",
        "Soma_A_linear_subbass",
        "Soma_A_linear_total",
        "harmonic_order_count",
        "density_metric_per_harmonic",
        "density_metric_normalized",
        "energy_weighted_component_density_diagnostic",
        "harmonic_partial_count",
        "inharmonic_partial_count",
        "total_detected_partial_count",
        "unique_harmonic_order_count",
        "component_harmonic_energy_ratio",
        "component_inharmonic_energy_ratio",
        "component_subbass_energy_ratio",
        "acoustic_f0_status",
        "f0_used_for_density_hz",
        "f0_used_for_density_source",
        "f0_used_for_harmonic_validation_hz",
        "f0_fit_accepted",
        "f0_fit_rejection_reason",
        "component_total_inharmonic_energy_ratio",
        "component_energy_denominator",
        "component_energy_method",
        "Coherent Gain",
        "D_agn",
        "D_harm",
        "DM Domain",
        "Density Metric",
        "Density Metric Normalized (D*)",
        "Density Metric Per Harmonic (D/N)",
        "Density Scale",
        "Filtered Components",
        "Harmonic Count",
        "harmonic_plus_inharmonic_energy_sum",
        "ground_noise_energy_sum",
        "Harmonic Count (N)",
        "Harmonic Weight (a)",
        "Highest Harmonic (Hz)",
        "Hop Length",
        "Inharmonic Weight (ß)",
        "Lowest Harmonic (Hz)",
        "Max Peaks per Band",
        "N FFT",
        "Note",
        "P_norm",
        "R_norm",
        "SDM Fallback Used",
        "SNR Threshold (dB)",
        "Search Band (cents)",
        "Sigma (scale units)",
        "Tier",
        "Weight Function",
        "Weight P",
        "Weight R",
        "Window",
    )
)

COMPILED_METRICS_PUBLICATION_COLUMN_ALLOWLIST: frozenset[str] = _V7_STYLE_PUBLIC | _MAIN6_METRICS_SHEET

_PCA_PUBLICATION_COLUMN_RE = re.compile(r"^PC\d+(_Norm2?)?$")
_TSNE_COLUMN_RE = re.compile(r"^TSNE\d+$")
_UMAP_COLUMN_RE = re.compile(r"^UMAP\d+$")


def publication_metrics_sheet_columns(df: pd.DataFrame) -> List[str]:
    """Column names retained when filtering (stable order = first-seen in ``df``)."""
    selected: List[str] = []
    seen: set[str] = set()
    for col in df.columns:
        name = str(col)
        if name in seen:
            continue
        if name in COMPILED_METRICS_PUBLICATION_COLUMN_ALLOWLIST or _PCA_PUBLICATION_COLUMN_RE.match(name):
            selected.append(name)
            seen.add(name)
        elif _TSNE_COLUMN_RE.match(name) or _UMAP_COLUMN_RE.match(name):
            selected.append(name)
            seen.add(name)
        elif name in ("is_anomaly", "anomaly_score"):
            selected.append(name)
            seen.add(name)
        elif "Dissonance" in name:
            selected.append(name)
            seen.add(name)
    return selected


def filter_dataframe_for_publication_metrics_sheet(df: pd.DataFrame) -> pd.DataFrame:
    cols = publication_metrics_sheet_columns(df)
    if not cols:
        return df.copy()
    return df.loc[:, cols].copy()
