# -*- coding: utf-8 -*-
"""Hard runtime schema / version guard tests.

AUDIT FIX (stale-pipeline detection) — these tests pin down the
contract that every per-note ``spectral_analysis.xlsx`` carries:

* ``Analysis_Metadata.analysis_schema_version =
  single_pass_raw_export_v2``
* ``Amplitude_raw`` and ``Power_raw`` on the three spectrum sheets
* ``model_weights_source = current_analysis`` in
  ``integrated_single_pass`` mode
* ``export_alignment_source = disabled_integrated_single_pass`` and
  ``export_alignment_factor = 1.0`` in single-pass mode
* no batch_* contamination on Inharmonic Spectrum in single-pass mode

and that the ``verify_runtime_schema.py`` CLI returns a non-zero
exit code on fixtures matching the legacy / stale outputs the user
reported (Density_Metrics with only the six legacy columns, missing
Amplitude_raw/Power_raw, ``model_weights_source=apply_filters_arguments``
in single-pass mode, etc.).
"""

from __future__ import annotations

import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest


HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

from compile_metrics import (  # noqa: E402
    EXPECTED_ANALYSIS_SCHEMA_VERSION,
    STALE_PIPELINE_USER_MESSAGE,
    assess_per_note_workbook_schema,
    assert_results_dir_schema_or_raise,
    scan_results_dir_for_stale_per_note_workbooks,
    extract_density_components_from_per_note_workbook,
)
from proc_audio import (  # noqa: E402
    ANALYSIS_SCHEMA_VERSION,
    _proc_audio_runtime_signature,
    log_runtime_paths,
)


# ---------------------------------------------------------------------------
# Synthetic workbook builders.
# ---------------------------------------------------------------------------
def _spectrum_df(
    component: str,
    *,
    with_raw: bool = True,
    with_amplitude_only: bool = False,
    extra_columns: Optional[dict] = None,
) -> pd.DataFrame:
    rows = [
        {"Frequency (Hz)": 220.0 + i * 110.0, "Magnitude (dB)": -10.0 - i, "Amplitude": 0.5 - 0.1 * i}
        for i in range(3)
    ]
    df = pd.DataFrame(rows)
    if with_raw and not with_amplitude_only:
        df["Amplitude_raw"] = df["Amplitude"]
        df["Power_raw"] = df["Amplitude"] ** 2
    if extra_columns:
        for k, v in extra_columns.items():
            df[k] = v
    df["Component_Type"] = component
    df["Note"] = "A4"
    return df


def _write_workbook(
    path: Path,
    *,
    schema_version: Optional[str] = EXPECTED_ANALYSIS_SCHEMA_VERSION,
    include_raw_columns: bool = True,
    model_weights_source: str = "current_analysis",
    component_profile_source: str = "integrated_single_pass",
    export_alignment_source: str = "disabled_integrated_single_pass",
    export_alignment_factor: float = 1.0,
    include_inharm_batch_columns: bool = False,
    include_density_metrics_sheet: Optional[str] = None,
    include_analysis_metadata: bool = True,
) -> Path:
    """Write a *minimal* per-note ``spectral_analysis.xlsx``.

    ``include_density_metrics_sheet`` may be one of: ``None`` (no
    Density_Metrics sheet at all), ``"legacy_six"`` (the bad layout
    we want to detect), or ``"audit_canonical"`` (the new layout).
    """
    harm_df = _spectrum_df("Harmonic", with_raw=include_raw_columns)
    extra_ih = (
        {
            "batch_harmonic_energy_ratio": 0.95,
            "batch_inharmonic_energy_ratio": 0.04,
            "batch_subbass_energy_ratio": 0.01,
        }
        if include_inharm_batch_columns
        else None
    )
    ih_df = _spectrum_df(
        "Inharmonic",
        with_raw=include_raw_columns,
        with_amplitude_only=not include_raw_columns,
        extra_columns=extra_ih,
    )
    sb_df = _spectrum_df("Subbass", with_raw=include_raw_columns)

    meta_rows = []
    if include_analysis_metadata:
        if schema_version is not None:
            meta_rows.append(("analysis_schema_version", schema_version))
        meta_rows.extend(
            [
                ("model_weights_source", model_weights_source),
                ("component_profile_source", component_profile_source),
                ("export_alignment_source", export_alignment_source),
                ("export_alignment_factor", float(export_alignment_factor)),
                ("component_harmonic_energy_ratio", 0.8),
                ("component_inharmonic_energy_ratio", 0.15),
                ("component_subbass_energy_ratio", 0.05),
                ("proc_audio_runtime_signature", "test_signature_0001"),
                ("proc_audio_file", "<test>"),
            ]
        )
    meta_df = (
        pd.DataFrame(meta_rows, columns=["Parameter", "Value"])
        if meta_rows
        else pd.DataFrame(columns=["Parameter", "Value"])
    )

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        harm_df.to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        ih_df.to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        sb_df.to_excel(writer, sheet_name="Sub-bass band", index=False)
        if include_analysis_metadata:
            meta_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        if include_density_metrics_sheet == "legacy_six":
            dm = pd.DataFrame(
                [
                    {
                        "Note": "A4",
                        "weight_function": "linear",
                        "Harmonic Partials sum": 177857.0,
                        "Inharmonic Partials sum": 100.0,
                        "Sub-bass sum": 10.0,
                        "Total sum": 177967.0,
                    }
                ]
            )
            dm.to_excel(writer, sheet_name="Density_Metrics", index=False)
        elif include_density_metrics_sheet == "audit_canonical":
            dm = pd.DataFrame(
                [
                    {
                        "Note": "A4",
                        "weight_function": "linear",
                        "Harmonic Partials sum": 1.4,
                        "Inharmonic Partials sum": 0.3,
                        "Sub-bass sum": 0.1,
                        "Total sum": 1.8,
                        "component_harmonic_energy_ratio": 0.8,
                        "component_inharmonic_energy_ratio": 0.15,
                        "component_subbass_energy_ratio": 0.05,
                        "weighted_harmonic_density_contribution": 1.12,
                        "weighted_inharmonic_density_contribution": 0.045,
                        "weighted_subbass_density_contribution": 0.005,
                        "density_metric_raw": 1.17,
                        "density_metric_normalized": 1.0,
                        "density_extraction_status": "ok",
                    }
                ]
            )
            dm.to_excel(writer, sheet_name="Density_Metrics", index=False)
    return path


# ---------------------------------------------------------------------------
# Tests A–F (the audit-task labels)
# ---------------------------------------------------------------------------
def test_A_old_workbook_with_only_amplitude_fails_schema(tmp_path):
    """A) An old workbook that exports only ``Amplitude`` (no raw cols)
    must fail the schema audit."""
    p = tmp_path / "spectral_analysis.xlsx"
    _write_workbook(
        p,
        include_raw_columns=False,
        schema_version=EXPECTED_ANALYSIS_SCHEMA_VERSION,
        include_inharm_batch_columns=True,
    )
    info = assess_per_note_workbook_schema(p)
    assert info["schema_ok"] is True  # the version token is fine
    assert info["has_amplitude_raw"] is False
    assert info["has_power_raw"] is False
    assert any("Amplitude_raw" in pr for pr in info["problems"])
    assert any("Power_raw" in pr for pr in info["problems"])


def test_B_workbook_missing_schema_version_fails(tmp_path):
    """B) A workbook lacking ``analysis_schema_version`` must fail."""
    p = tmp_path / "spectral_analysis.xlsx"
    _write_workbook(p, schema_version=None)
    info = assess_per_note_workbook_schema(p)
    assert info["schema_version"] is None
    assert info["schema_ok"] is False
    assert any("analysis_schema_version" in pr for pr in info["problems"])


def test_C_apply_filters_arguments_in_integrated_mode_fails(tmp_path):
    """C) Integrated mode workbook with
    ``model_weights_source=apply_filters_arguments`` must be flagged."""
    p = tmp_path / "spectral_analysis.xlsx"
    _write_workbook(
        p,
        model_weights_source="apply_filters_arguments",
        component_profile_source="integrated_single_pass",
    )
    info = assess_per_note_workbook_schema(p)
    assert info["model_weights_source"] == "apply_filters_arguments"
    assert any("model_weights_source" in pr for pr in info["problems"])


def test_D_density_metrics_six_column_layout_is_flagged(tmp_path):
    """D) A compiled workbook with the legacy six-column Density_Metrics
    layout must be detected as ``legacy_six_columns``."""
    p = tmp_path / "spectral_analysis.xlsx"
    _write_workbook(
        p,
        include_density_metrics_sheet="legacy_six",
    )
    info = assess_per_note_workbook_schema(p)
    assert info["density_metrics_layout"] == "legacy_six_columns"


def test_D2_density_metrics_audit_canonical_layout_passes(tmp_path):
    """D) Inverse — the canonical layout must NOT be flagged."""
    p = tmp_path / "spectral_analysis.xlsx"
    _write_workbook(
        p,
        include_density_metrics_sheet="audit_canonical",
    )
    info = assess_per_note_workbook_schema(p)
    assert info["density_metrics_layout"] == "audit_canonical"


def test_E_current_workbook_with_raw_columns_passes(tmp_path):
    """E) A workbook produced by the current single-pass pipeline (raw
    columns + correct schema version + correct provenance) passes."""
    p = tmp_path / "spectral_analysis.xlsx"
    _write_workbook(p)
    info = assess_per_note_workbook_schema(p)
    assert info["schema_ok"] is True
    assert info["has_amplitude_raw"] is True
    assert info["has_power_raw"] is True
    assert info["problems"] == []


def test_F_verify_runtime_schema_returns_nonzero_on_stale_fixture(tmp_path):
    """F) ``verify_runtime_schema.py`` must return a non-zero exit code
    on a fixture mimicking the user-uploaded stale output (no raw
    columns + batch_* leakage + legacy Density_Metrics layout)."""
    results_dir = tmp_path / "analysis_results" / "Clarinete_mf"
    results_dir.mkdir(parents=True)
    p = results_dir / "spectral_analysis.xlsx"
    _write_workbook(
        p,
        include_raw_columns=False,
        include_inharm_batch_columns=True,
        model_weights_source="apply_filters_arguments",
        include_density_metrics_sheet="legacy_six",
        export_alignment_source="legacy_batch_alignment",
        export_alignment_factor=0.123,
        schema_version="single_pass_raw_export_v1",  # stale, older token
    )
    cmd = [
        sys.executable,
        str(ROOT / "verify_runtime_schema.py"),
        "--results-dir",
        str(tmp_path / "analysis_results"),
        "--json",
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    counters = payload["counters"]
    assert counters["files_found"] == 1
    assert counters["schema_ok"] == 0
    assert counters["missing_amplitude_raw"] == 1
    assert counters["missing_power_raw"] == 1
    assert counters["model_weights_source_not_current_analysis"] == 1
    assert counters["stale_density_metrics_layout"] == 1
    assert counters["first_failing_path"] is not None


def test_F2_verify_runtime_schema_returns_zero_on_clean_fixture(tmp_path):
    """F) Inverse — on a current-schema fixture the CLI returns 0."""
    results_dir = tmp_path / "analysis_results" / "A4"
    results_dir.mkdir(parents=True)
    p = results_dir / "spectral_analysis.xlsx"
    _write_workbook(p)
    cmd = [
        sys.executable,
        str(ROOT / "verify_runtime_schema.py"),
        "--results-dir",
        str(tmp_path / "analysis_results"),
        "--quiet",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# Additional coverage for the GUI / orchestrator guard.
# ---------------------------------------------------------------------------
def test_assert_results_dir_raises_on_stale(tmp_path):
    """The orchestrator/GUI helper must raise RuntimeError carrying the
    canonical user-facing message."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    p = results_dir / "spectral_analysis.xlsx"
    _write_workbook(p, schema_version=None)
    with pytest.raises(RuntimeError) as excinfo:
        assert_results_dir_schema_or_raise(results_dir)
    assert STALE_PIPELINE_USER_MESSAGE in str(excinfo.value)


def test_assert_results_dir_passes_on_current(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    p = results_dir / "spectral_analysis.xlsx"
    _write_workbook(p)
    summary = assert_results_dir_schema_or_raise(results_dir)
    assert summary["total"] == 1
    assert summary["stale"] == 0
    assert summary["valid"] == 1


# ---------------------------------------------------------------------------
# Extractor schema gate.
# ---------------------------------------------------------------------------
def test_extractor_marks_stale_schema(tmp_path):
    """A stale workbook must be tagged ``extraction_error_stale_schema``
    by the direct Density_Metrics extractor (and the rest of the
    payload must be left untouched)."""
    p = tmp_path / "spectral_analysis.xlsx"
    _write_workbook(p, schema_version="single_pass_raw_export_v1")  # stale
    payload = extract_density_components_from_per_note_workbook(p)
    assert payload["density_extraction_status"] == "extraction_error_stale_schema"
    assert payload["analysis_schema_version"] == "single_pass_raw_export_v1"


def test_extractor_accepts_current_schema(tmp_path):
    p = tmp_path / "spectral_analysis.xlsx"
    _write_workbook(p)
    payload = extract_density_components_from_per_note_workbook(p)
    assert payload["density_extraction_status"] != "extraction_error_stale_schema"
    assert payload["analysis_schema_version"] == EXPECTED_ANALYSIS_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Runtime path / signature plumbing.
# ---------------------------------------------------------------------------
def test_runtime_signature_is_stable_short_hash():
    sig = _proc_audio_runtime_signature()
    assert isinstance(sig, str)
    assert len(sig) >= 8


def test_log_runtime_paths_returns_required_keys():
    info = log_runtime_paths()
    for k in (
        "sys_executable",
        "cwd",
        "proc_audio_file",
        "analysis_schema_version",
        "proc_audio_runtime_signature",
    ):
        assert k in info, f"missing key {k} in {info}"
    assert info["analysis_schema_version"] == ANALYSIS_SCHEMA_VERSION


def test_analysis_schema_version_token_is_audit_canonical():
    """The constant token MUST match the value the audit task pinned."""
    assert ANALYSIS_SCHEMA_VERSION == "single_pass_raw_export_v2"
    assert EXPECTED_ANALYSIS_SCHEMA_VERSION == ANALYSIS_SCHEMA_VERSION
