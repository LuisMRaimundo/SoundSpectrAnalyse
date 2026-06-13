from __future__ import annotations

"""
Seventh Phase 12 descriptor / export-routing contract layer for proc_audio.py.

Complements test_proc_audio_core_additional.py,
test_proc_audio_helper_contract_additional.py, and
test_proc_audio_preprocessing_contract_additional.py with helper-level
regression tests for H/I/S separation, canonical vs body-band vs
full-spectrum vs diagnostic descriptor routing, and export-row preparation
before compile_metrics.

No production code changes. No real audio, GUI, plotting, or batch pipeline.
"""

from copy import deepcopy

import math

import numpy as np
import pytest

import proc_audio as PA
from proc_audio import AudioProcessor


def _export_ap(**attrs: object) -> AudioProcessor:
    ap = AudioProcessor()
    ap.weight_function = "log"
    ap.density_salience_threshold_db = -40.0
    ap.density_frequency_ceiling_hz = 5000.0
    ap.freq_max = 20000.0
    for key, val in attrs.items():
        setattr(ap, key, val)
    return ap


def _main_row(
    ap: AudioProcessor,
    *,
    note: str = "C4",
    h_psum: float = 2.0,
    i_psum: float = 1.0,
    s_psum: float = 0.5,
    t_psum: float = 3.5,
) -> dict:
    return ap._build_main_metrics_export_row(
        note,
        h_psum=h_psum,
        i_psum=i_psum,
        s_psum=s_psum,
        t_psum=t_psum,
    )


# ---------------------------------------------------------------------------
# 1. Canonical / body-band / full-spectrum separation in export row
# ---------------------------------------------------------------------------


def test_export_row_keeps_canonical_body_and_full_spectrum_scalars_separate() -> None:
    ap = _export_ap(
        canonical_density_v5_adapted=1.25,
        effective_partial_density=3.0,
        harmonic_body_density=0.55,
        density_body_weighted_sum_body_ceiling=99.0,
        density_full_spectrum_weighted_sum_20khz=88.0,
        harmonic_full_spectrum_energy_sum_20khz=7.0,
        harmonic_component_energy_sum_body_ceiling=4.0,
    )
    row = _main_row(ap)
    assert row["canonical_density_v5_adapted"] == pytest.approx(1.25)
    assert row["effective_partial_density"] == pytest.approx(3.0)
    assert row["harmonic_body_density"] == pytest.approx(0.55)
    assert row["density_body_weighted_sum_body_ceiling"] == pytest.approx(99.0)
    assert row["density_full_spectrum_weighted_sum_20khz"] == pytest.approx(88.0)
    assert row["harmonic_full_spectrum_energy_sum_20khz"] == pytest.approx(7.0)
    assert row["harmonic_component_energy_sum_body_ceiling"] == pytest.approx(4.0)
    assert row["density_body_weighted_sum_body_ceiling"] != pytest.approx(row["canonical_density_v5_adapted"])


def test_export_row_body_ceiling_does_not_replace_effective_partial_density() -> None:
    ap = _export_ap(
        effective_partial_density=2.75,
        body_weighted_effective_density=1.1,
        density_body_weighted_sum_body_ceiling=50.0,
        pitch_normalized_component_body_density_body_ceiling=0.33,
    )
    row = _main_row(ap)
    assert row["effective_partial_density"] == pytest.approx(2.75)
    assert row["body_weighted_effective_density"] == pytest.approx(1.1)
    assert row["density_body_weighted_sum_body_ceiling"] == pytest.approx(50.0)
    assert row["pitch_normalized_component_body_density_body_ceiling"] == pytest.approx(0.33)


# ---------------------------------------------------------------------------
# 2. H/I/S routing in export row and pie helpers
# ---------------------------------------------------------------------------


def test_export_row_partial_sums_use_arguments_not_attached_energy_scalars() -> None:
    ap = _export_ap(
        harmonic_energy_sum=999.0,
        inharmonic_energy_sum=888.0,
        subbass_energy_sum=777.0,
    )
    row = _main_row(ap, h_psum=2.0, i_psum=1.0, s_psum=0.25, t_psum=3.25)
    assert row["Harmonic Partials sum"] == pytest.approx(2.0)
    assert row["Inharmonic Partials sum"] == pytest.approx(1.0)
    assert row["Sub-bass sum"] == pytest.approx(0.25)
    assert row["Total sum"] == pytest.approx(3.25)
    assert row["harmonic_energy_sum"] == pytest.approx(999.0)
    assert row["inharmonic_energy_sum"] == pytest.approx(888.0)
    assert row["subbass_energy_sum"] == pytest.approx(777.0)


def test_export_row_his_energy_ratios_remain_distinct_from_component_aliases() -> None:
    ap = _export_ap(
        harmonic_energy_ratio=0.60,
        inharmonic_energy_ratio=0.30,
        subbass_energy_ratio=0.10,
        component_harmonic_energy_ratio=0.99,
        component_inharmonic_energy_ratio=0.01,
        component_subbass_energy_ratio=0.0,
    )
    row = _main_row(ap)
    assert row["harmonic_energy_ratio"] == pytest.approx(0.60)
    assert row["inharmonic_energy_ratio"] == pytest.approx(0.30)
    assert row["subbass_energy_ratio"] == pytest.approx(0.10)
    assert row["component_harmonic_energy_ratio"] == pytest.approx(0.99)
    assert row["component_inharmonic_energy_ratio"] == pytest.approx(0.01)
    assert row["component_subbass_energy_ratio"] == pytest.approx(0.0)


def test_component_energy_ratio_triple_preserves_his_order_and_uses_primary() -> None:
    ap = _export_ap(
        harmonic_energy_ratio=0.60,
        inharmonic_energy_ratio=0.30,
        subbass_energy_ratio=0.10,
        component_harmonic_energy_ratio=0.99,
        component_inharmonic_energy_ratio=0.01,
        component_subbass_energy_ratio=0.0,
    )
    trip, primary_missing, fb_missing = ap._component_energy_ratio_triple_for_pie()
    assert trip == (0.60, 0.30, 0.10)
    assert primary_missing == []
    assert fb_missing == []


def test_component_energy_ratio_triple_falls_back_to_component_aliases_when_primary_missing() -> None:
    ap = _export_ap(
        component_harmonic_energy_ratio=0.50,
        component_inharmonic_energy_ratio=0.40,
        component_subbass_energy_ratio=0.10,
    )
    trip, primary_missing, fb_missing = ap._component_energy_ratio_triple_for_pie()
    assert trip == (0.50, 0.40, 0.10)
    assert "harmonic_energy_ratio" in primary_missing
    assert "inharmonic_energy_ratio" in primary_missing


def test_component_energy_ratio_triple_empty_container_returns_none() -> None:
    ap = AudioProcessor()
    trip, primary_missing, fb_missing = ap._component_energy_ratio_triple_for_pie()
    assert trip is None
    assert "harmonic_energy_ratio" in primary_missing
    assert "component_harmonic_energy_ratio" in fb_missing


# ---------------------------------------------------------------------------
# 3. Canonical vs diagnostic density descriptors
# ---------------------------------------------------------------------------


def test_export_row_diagnostic_density_falls_back_to_density_metric_value() -> None:
    ap = _export_ap(
        canonical_density_v5_adapted=1.50,
        density_metric_value=3.30,
    )
    row = _main_row(ap)
    assert row["canonical_density_v5_adapted"] == pytest.approx(1.50)
    assert row["energy_weighted_component_density_diagnostic"] == pytest.approx(3.30)


def test_export_row_explicit_diagnostic_does_not_replace_canonical_density() -> None:
    ap = _export_ap(
        canonical_density_v5_adapted=1.50,
        energy_weighted_component_density_diagnostic=9.99,
        density_metric_value=3.30,
    )
    row = _main_row(ap)
    assert row["canonical_density_v5_adapted"] == pytest.approx(1.50)
    assert row["energy_weighted_component_density_diagnostic"] == pytest.approx(9.99)


def test_export_row_legacy_component_strength_aliases_remain_separate() -> None:
    ap = _export_ap(
        component_strength_h=0.7,
        component_strength_i=0.2,
        component_strength_s=0.1,
        legacy_component_strength_h_v55=0.99,
        legacy_component_strength_i_v55=0.01,
        legacy_component_strength_s_v55=0.0,
    )
    row = _main_row(ap)
    assert row["component_strength_h"] == pytest.approx(0.7)
    assert row["legacy_component_strength_h_v55"] == pytest.approx(0.99)


# ---------------------------------------------------------------------------
# 4. Legacy export row vs main metrics export row
# ---------------------------------------------------------------------------


def test_build_legacy_density_metrics_row_uses_legacy_keys_and_canonical_preference() -> None:
    ap = _export_ap(
        canonical_density_v5_adapted=2.0,
        density_metric_value=9.0,
        spectral_density_metric_value=1.1,
        filtered_density_metric_value=0.8,
        combined_density_metric_value=1.5,
    )
    legacy = ap._build_legacy_density_metrics_row("D4")
    assert set(legacy.keys()) == {
        "Note",
        "weight_function",
        "Density Metric",
        "Spectral Density Metric",
        "Filtered Density Metric",
        "Combined Density Metric",
        "spectral_masking_enabled",
        "legacy_density_export_version",
    }
    assert legacy["Density Metric"] == pytest.approx(2.0)
    assert legacy["Spectral Density Metric"] == pytest.approx(1.1)
    assert "canonical_density_v5_adapted" not in legacy


def test_build_legacy_density_metrics_row_falls_back_to_density_metric_value() -> None:
    ap = _export_ap(density_metric_value=4.5)
    legacy = ap._build_legacy_density_metrics_row("G4")
    assert legacy["Density Metric"] == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# 5. Amplitude-mass routing priority (diagnostic vs linear fallback)
# ---------------------------------------------------------------------------


def test_preferred_component_amplitude_sum_triple_requires_all_three_finite_positive() -> None:
    ap = _export_ap(
        harmonic_amplitude_sum=2.0,
        inharmonic_amplitude_sum=1.0,
    )
    assert ap._preferred_component_amplitude_sum_triple() is None


def test_component_amplitude_mass_triple_falls_back_to_linear_sums_when_preferred_missing() -> None:
    ap = _export_ap(
        linear_sum_amplitude_harmonic=2.0,
        linear_sum_amplitude_inharmonic_partial=1.0,
        linear_sum_amplitude_subbass_band=0.5,
    )
    triple, basis, gaps, tech = ap._component_amplitude_mass_triple_for_pie()
    assert triple == (2.0, 1.0, 0.5)
    assert basis == "linear_amplitude_sum"
    assert "harmonic_amplitude_sum" in gaps
    assert tech == "linear_sum_amplitude_metrics"


def test_linear_component_density_balance_triple_uses_named_linear_sums() -> None:
    ap = _export_ap(
        linear_sum_amplitude_harmonic=1.0,
        linear_sum_amplitude_inharmonic_partial=0.5,
        linear_sum_amplitude_subbass_band=0.25,
    )
    h, i, s, basis = ap._linear_component_density_balance_triple_with_basis()
    assert (h, i, s) == (1.0, 0.5, 0.25)
    assert basis == "linear_sum_amplitude_metrics"


# ---------------------------------------------------------------------------
# 6. Parameter provenance / metadata in export row
# ---------------------------------------------------------------------------


def test_export_row_f0_provenance_fields_reflect_canonical_triplet() -> None:
    ap = _export_ap(
        f0_final=261.63,
        f0_initial=260.0,
        f0_prior_hz=258.0,
        f0_fit_accepted=True,
        f0_final_source="robust_fit",
        acoustic_f0_status="accepted_robust",
        f0_validation_mode="free_fit",
    )
    row = _main_row(ap)
    assert row["Note"] == "C4"
    assert row["f0_used_for_density_hz"] == pytest.approx(261.63)
    assert row["f0_used_for_density_source"] == "robust_fit"
    assert row["f0_final_hz"] == pytest.approx(261.63)
    assert row["weight_function"] == "log"


def test_export_row_analysis_parameter_profile_id_is_deterministic() -> None:
    ap = _export_ap(
        weight_function="log",
        density_salience_threshold_db=-40.0,
        density_frequency_ceiling_hz=5000.0,
    )
    first = _main_row(ap)["analysis_parameter_profile_id"]
    second = _main_row(ap)["analysis_parameter_profile_id"]
    assert first == second == "wf=log|dst=-40.0|ceil=5000.0"


def test_export_row_includes_density_metric_contract_metadata() -> None:
    row = _main_row(_export_ap())
    assert row["metric_contract_name"] == "density_metric_raw"
    assert row["metric_contract_formula"] == "D_H*w_H + D_I*w_I + D_S*w_S"
    assert row["metric_contract_basis"] == "log-amplitude"


def test_export_row_does_not_leak_runtime_path_keys() -> None:
    ap = _export_ap()
    ap.folder_path = "C:\\Users\\secret\\audio_batch"
    row = _main_row(ap)
    forbidden = {"cwd", "proc_audio_file", "folder_path", "sys_executable"}
    assert forbidden.isdisjoint(row.keys())
    for val in row.values():
        if isinstance(val, str) and ("\\Users\\" in val or ":/" in val):
            pytest.fail(f"unexpected path-like export value: {val!r}")


# ---------------------------------------------------------------------------
# 7. Numeric stability and missing-value fallbacks
# ---------------------------------------------------------------------------


def test_export_row_missing_optional_numeric_fields_become_nan() -> None:
    row = _main_row(_export_ap())
    assert math.isnan(row["harmonic_body_density"])
    assert math.isnan(row["density_full_spectrum_weighted_sum_20khz"])
    assert math.isnan(row["density_metric_normalized"])


def test_export_row_finite_numeric_fields_remain_float() -> None:
    ap = _export_ap(
        harmonic_density_component=1.2,
        inharmonic_density_component=0.4,
        subbass_density_component=0.1,
        harmonic_density_component_on_attack=0.9,
    )
    row = _main_row(ap)
    assert isinstance(row["harmonic_density_component"], float)
    assert row["harmonic_density_component_on_attack"] == pytest.approx(0.9)


def test_export_row_nonfinite_attached_scalars_sanitize_to_nan() -> None:
    ap = _export_ap(
        harmonic_energy_ratio=float("inf"),
        inharmonic_energy_sum=float("nan"),
        harmonic_amplitude_sum=None,
    )
    row = _main_row(ap)
    assert math.isnan(row["harmonic_energy_ratio"])
    assert math.isnan(row["inharmonic_energy_sum"])


# ---------------------------------------------------------------------------
# 8. Determinism / non-mutation
# ---------------------------------------------------------------------------


def test_export_row_builder_is_idempotent_and_preserves_attached_scalars() -> None:
    ap = _export_ap(
        canonical_density_v5_adapted=1.8,
        harmonic_energy_ratio=0.65,
        inharmonic_energy_ratio=0.25,
        subbass_energy_ratio=0.10,
    )
    snap = deepcopy(
        {
            "canonical_density_v5_adapted": ap.canonical_density_v5_adapted,
            "harmonic_energy_ratio": ap.harmonic_energy_ratio,
        }
    )
    first = _main_row(ap)
    second = _main_row(ap)
    assert first == second
    assert ap.canonical_density_v5_adapted == snap["canonical_density_v5_adapted"]
    assert ap.harmonic_energy_ratio == snap["harmonic_energy_ratio"]


def test_publication_clean_export_adds_canonical_density_alias_without_dropping_canonical_v5() -> None:
    ap = _export_ap(canonical_density_v5_adapted=2.2, density_formula_version="v5_test")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "metadata_sanitizer.publication_clean_export_enabled",
            lambda: True,
        )
        row = _main_row(ap)
    assert row["canonical_density_v5_adapted"] == pytest.approx(2.2)
    assert row["canonical_density"] == pytest.approx(2.2)
    assert "density_formula_version" not in row
