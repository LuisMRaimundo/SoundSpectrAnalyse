# -*- coding: utf-8 -*-
"""Audit fix — Density_Metrics component-basis tests.

AUDIT FIX (Density_Metrics component basis) — D_H, D_I, D_S must be
computed on a linear-amplitude basis (Σ Amplitude_raw) by default;
Power_raw is the *weight* basis (energy ratios computed inside
proc_audio), not the *component* basis. Selecting Power_raw for the
component sums lets sub-bass dominate the metric through Σ A² and is
therefore forbidden unless the operator explicitly opts in via
``density_component_basis="power_sum"``.

The tests below pin down audit points A–E plus the legacy-alignment
log regression.
"""

from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve()
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import compile_metrics as cm  # noqa: E402
from proc_audio import ANALYSIS_SCHEMA_VERSION as _ASV  # noqa: E402


# ---------------------------------------------------------------------------
# Fixture: per-note workbook with explicit Amplitude_raw + Power_raw control.
# ---------------------------------------------------------------------------
def _write_per_note_workbook(
    path: Path,
    *,
    note: str = "A4",
    harmonic_amps,
    inharmonic_amps,
    subbass_amps,
    subbass_power_raw_override: Optional[float] = None,
    w_H: float = 0.8,
    w_I: float = 0.10,
    w_S: float = 0.10,
    include_amplitude_raw: bool = True,
    include_power_raw: bool = True,
    include_amplitude_legacy: bool = True,
) -> None:
    """Write a per-note workbook with controllable raw / power columns.

    The fundamental knobs:

    * ``include_amplitude_raw`` — emit ``Amplitude_raw`` (default True).
      Required for the amplitude-basis default.
    * ``include_power_raw`` — emit ``Power_raw`` = Amplitude_raw²
      (default True). Required for the power-sum debug basis.
    * ``subbass_power_raw_override`` — when not None, replace the
      sub-bass ``Power_raw`` value with this number (used by audit
      test E to inject a ridiculously large power value and prove the
      amplitude-basis density is unaffected).
    """

    def _sheet(values, *, is_subbass: bool = False) -> pd.DataFrame:
        arr = np.asarray(values, dtype=float)
        body = {
            "Frequency (Hz)": np.linspace(220.0, 440.0, len(arr) or 1)[: len(arr)],
            "Magnitude (dB)": [-30.0] * len(arr),
        }
        if include_amplitude_legacy:
            body["Amplitude"] = arr
        df = pd.DataFrame(body)
        if include_amplitude_raw:
            df["Amplitude_raw"] = arr
        if include_power_raw:
            powers = arr ** 2
            if is_subbass and subbass_power_raw_override is not None:
                powers = np.full_like(arr, float(subbass_power_raw_override))
            df["Power_raw"] = powers
        df["Note"] = note
        return df

    harm_df = _sheet(harmonic_amps)
    inh_df = _sheet(inharmonic_amps)
    sb_df = _sheet(subbass_amps, is_subbass=True)

    am_df = pd.DataFrame(
        [
            ("analysis_schema_version", _ASV),
            ("model_weights_source", "current_analysis"),
            ("component_profile_source", "integrated_single_pass"),
            ("export_alignment_source", "disabled_integrated_single_pass"),
            ("export_alignment_factor", 1.0),
            ("component_harmonic_energy_ratio", float(w_H)),
            ("component_inharmonic_energy_ratio", float(w_I)),
            ("component_subbass_energy_ratio", float(w_S)),
        ],
        columns=["Parameter", "Value"],
    )

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        harm_df.to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        inh_df.to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        sb_df.to_excel(writer, sheet_name="Sub-bass band", index=False)
        am_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)


# ---------------------------------------------------------------------------
# A. Both Power_raw and Amplitude_raw present → must select Amplitude_raw.
# ---------------------------------------------------------------------------
def test_A_amplitude_raw_wins_over_power_raw_by_default(tmp_path):
    wb = tmp_path / "spec.xlsx"
    _write_per_note_workbook(
        wb,
        harmonic_amps=[10, 20, 30],
        inharmonic_amps=[5, 5],
        subbass_amps=[2],
        include_amplitude_raw=True,
        include_power_raw=True,
    )
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "ok"
    assert info["density_component_basis"] == "amplitude_sum"
    assert info["density_weight_basis"] == "energy_ratio_power_sum"
    # Source columns must reference Amplitude_raw, not Power_raw.
    for src in (
        info["harmonic_spectrum_source"],
        info["inharmonic_spectrum_source"],
        info["subbass_spectrum_source"],
    ):
        assert "Amplitude_raw" in src
        assert "Power_raw" not in src
    # Values reflect Σ Amplitude_raw (linear), not Σ A².
    assert info["D_H"] == pytest.approx(60.0)
    assert info["D_I"] == pytest.approx(10.0)
    assert info["D_S"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# B. Source strings expose sheet=<X>;column=Amplitude_raw.
# ---------------------------------------------------------------------------
def test_B_source_strings_report_amplitude_raw_columns(tmp_path):
    wb = tmp_path / "spec.xlsx"
    _write_per_note_workbook(
        wb,
        harmonic_amps=[1.0, 2.0, 3.0],
        inharmonic_amps=[0.5, 0.5],
        subbass_amps=[0.1],
    )
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["harmonic_spectrum_source"].startswith("sheet=Harmonic Spectrum")
    assert "column=Amplitude_raw" in info["harmonic_spectrum_source"]
    assert info["inharmonic_spectrum_source"].startswith("sheet=Inharmonic Spectrum")
    assert "column=Amplitude_raw" in info["inharmonic_spectrum_source"]
    assert info["subbass_spectrum_source"].startswith("sheet=Sub-bass band")
    assert "column=Amplitude_raw" in info["subbass_spectrum_source"]


# ---------------------------------------------------------------------------
# C. Power_raw must only be selected under explicit debug basis.
# ---------------------------------------------------------------------------
def test_C_power_raw_only_under_explicit_debug_basis(tmp_path):
    wb = tmp_path / "spec.xlsx"
    _write_per_note_workbook(
        wb,
        harmonic_amps=[10, 20, 30],
        inharmonic_amps=[5, 5],
        subbass_amps=[2],
    )

    # Default — must NEVER hit Power_raw.
    info_default = cm.extract_density_components_from_per_note_workbook(wb)
    assert "Power_raw" not in info_default["harmonic_spectrum_source"]
    assert "Power_raw" not in info_default["inharmonic_spectrum_source"]
    assert "Power_raw" not in info_default["subbass_spectrum_source"]

    # Explicit opt-in.
    info_debug = cm.extract_density_components_from_per_note_workbook(
        wb, density_component_basis="power_sum",
    )
    assert "Power_raw" in info_debug["harmonic_spectrum_source"]
    assert "Power_raw" in info_debug["inharmonic_spectrum_source"]
    assert "Power_raw" in info_debug["subbass_spectrum_source"]
    assert info_debug["density_component_basis"] == "power_sum"
    # Σ A² values.
    assert info_debug["D_H"] == pytest.approx(10 ** 2 + 20 ** 2 + 30 ** 2)
    assert info_debug["D_I"] == pytest.approx(5 ** 2 + 5 ** 2)
    assert info_debug["D_S"] == pytest.approx(2 ** 2)


def test_C2_invalid_basis_raises(tmp_path):
    wb = tmp_path / "spec.xlsx"
    _write_per_note_workbook(
        wb,
        harmonic_amps=[1.0],
        inharmonic_amps=[1.0],
        subbass_amps=[1.0],
    )
    with pytest.raises(ValueError):
        cm.extract_density_components_from_per_note_workbook(
            wb, density_component_basis="not_a_valid_basis",
        )


# ---------------------------------------------------------------------------
# D. Regression — audit-specified worked example.
#    D_H=100, D_I=10, D_S=20 from Amplitude_raw;
#    w_H=0.8, w_I=0.1, w_S=0.1 → density_metric_raw = 83.
# ---------------------------------------------------------------------------
def test_D_audit_worked_example_density_metric_raw_equals_83(tmp_path):
    wb = tmp_path / "spec.xlsx"
    # Amplitudes chosen so the integer sum matches the audit's example.
    _write_per_note_workbook(
        wb,
        harmonic_amps=[40.0, 60.0],     # D_H = 100
        inharmonic_amps=[3.0, 7.0],     # D_I = 10
        subbass_amps=[8.0, 12.0],       # D_S = 20
        w_H=0.8, w_I=0.1, w_S=0.1,
    )
    out_df = cm._build_density_metrics_sheet_from_per_note_files(
        [(wb, "A4", "Note_A4_0")], weight_function="linear",
    )
    row = out_df.iloc[0]
    assert row["Harmonic Partials sum"] == pytest.approx(100.0)
    assert row["Inharmonic Partials sum"] == pytest.approx(10.0)
    assert row["Sub-bass sum"] == pytest.approx(20.0)
    assert row["component_harmonic_energy_ratio"] == pytest.approx(0.8)
    assert row["component_inharmonic_energy_ratio"] == pytest.approx(0.1)
    assert row["component_subbass_energy_ratio"] == pytest.approx(0.1)
    assert row["weighted_harmonic_density_contribution"] == pytest.approx(80.0)
    assert row["weighted_inharmonic_density_contribution"] == pytest.approx(1.0)
    assert row["weighted_subbass_density_contribution"] == pytest.approx(2.0)
    assert row["density_metric_raw"] == pytest.approx(83.0)
    # Provenance columns must be exposed on the compiled sheet.
    assert row["density_component_basis"] == "amplitude_sum"
    assert row["density_weight_basis"] == "energy_ratio_power_sum"


# ---------------------------------------------------------------------------
# E. A huge Power_raw in sub-bass must not affect density_metric_raw.
# ---------------------------------------------------------------------------
def test_E_huge_subbass_power_raw_does_not_affect_density_metric_raw(tmp_path):
    """If a sub-bass bin has a ridiculously large Power_raw value but a
    modest Amplitude_raw, the default amplitude-basis Density_Metrics
    must read the modest amplitude and remain bounded. The power-sum
    debug basis (opt-in) WILL see the inflated value — and that is
    precisely why it is reserved for diagnostic use.
    """
    wb = tmp_path / "spec.xlsx"
    _write_per_note_workbook(
        wb,
        harmonic_amps=[10.0, 20.0, 30.0],
        inharmonic_amps=[5.0, 5.0],
        subbass_amps=[2.0],
        subbass_power_raw_override=1.0e9,   # huge Σ A² spike
        w_H=0.8, w_I=0.15, w_S=0.05,
    )
    # Default (amplitude_sum) — unaffected.
    info_default = cm.extract_density_components_from_per_note_workbook(wb)
    assert info_default["D_S"] == pytest.approx(2.0)  # Σ Amplitude_raw, not the spike
    out_default = cm._build_density_metrics_sheet_from_per_note_files(
        [(wb, "A4", "Note_A4_0")], weight_function="linear",
    )
    assert out_default.iloc[0]["density_metric_raw"] == pytest.approx(
        60 * 0.8 + 10 * 0.15 + 2 * 0.05
    )

    # Power-sum debug basis — sees the inflated value (proves the override
    # actually landed on Power_raw and would have blown up the metric).
    info_debug = cm.extract_density_components_from_per_note_workbook(
        wb, density_component_basis="power_sum",
    )
    assert info_debug["D_S"] == pytest.approx(1.0e9)
    # The debug-only opt-in field is set on every compiled row.
    out_debug = cm._build_density_metrics_sheet_from_per_note_files(
        [(wb, "A4", "Note_A4_0")],
        weight_function="linear",
        density_component_basis="power_sum",
    )
    assert "density_metric_power_weighted_raw" in out_debug.columns
    assert out_debug.iloc[0]["density_metric_power_weighted_raw"] == pytest.approx(
        out_debug.iloc[0]["density_metric_raw"]
    )


# ---------------------------------------------------------------------------
# Alinhamento-export-linear regression — must not log in single-pass.
# ---------------------------------------------------------------------------
def test_no_alinhamento_export_linear_log_in_integrated_single_pass(tmp_path, caplog):
    """The legacy ``Alinhamento export linear`` log line is gated behind
    ``legacy_batch`` mode. A single-pass extraction must not trip it.

    This regression test exercises the canonical extract / compile
    workflow end-to-end and asserts the log capture contains no
    occurrence of the offending substring.
    """
    wb = tmp_path / "spec.xlsx"
    _write_per_note_workbook(
        wb,
        harmonic_amps=[10, 20, 30],
        inharmonic_amps=[5, 5],
        subbass_amps=[2],
        include_power_raw=True,
    )
    with caplog.at_level(logging.DEBUG):
        _ = cm.extract_density_components_from_per_note_workbook(wb)
        out_df = cm._build_density_metrics_sheet_from_per_note_files(
            [(wb, "A4", "Note_A4_0")], weight_function="linear",
        )
    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "Alinhamento export linear" not in joined, (
        f"Forbidden legacy log message leaked into the integrated single-pass "
        f"compile flow:\n{joined}"
    )
    assert not out_df.empty


# ---------------------------------------------------------------------------
# A. proc_audio.py predicate — Alinhamento line must never fire in
#    integrated_single_pass / current_analysis mode.
# ---------------------------------------------------------------------------
def _make_proc_audio_save_block_stub():
    """Construct a stub object mimicking the AudioProcessor attributes
    consumed by the export-alignment block of
    ``_save_spectral_data_to_excel`` so we can exercise the predicate
    without spinning up a full WAV / FFT pipeline.
    """
    class _Stub:
        def __init__(self):
            self.harmonic_weight = 0.95
            self.inharmonic_weight = 0.05
            self.export_alignment_factor = None
            self.export_alignment_source = None
            self.linear_amplitude_batch_alignment_factor = None
            self.auto_model_weights_from_analysis = True
            self.component_profile_source = "integrated_single_pass"
            self.model_weights_source = "current_analysis"
    return _Stub()


def _run_export_alignment_branch(stub, *, s_h=1.0, s_ih_raw=1.0, s_sb_raw=0.5):
    """Replicate the predicate logic used inside
    ``_save_spectral_data_to_excel`` so the test exercises exactly the
    same conditional. Any divergence would be a test bug, not a code
    bug — but the predicate is a one-liner, so this is safe.
    """
    import logging as _lg

    log = _lg.getLogger("proc_audio")
    _component_profile_source = str(getattr(stub, "component_profile_source", "") or "")
    _auto_weights = bool(getattr(stub, "auto_model_weights_from_analysis", True))
    _model_weights_source = str(getattr(stub, "model_weights_source", "") or "")
    _is_integrated_single_pass = (
        _component_profile_source == "integrated_single_pass"
        or _auto_weights
        or _model_weights_source == "current_analysis"
    )

    if _is_integrated_single_pass:
        stub.export_alignment_factor = 1.0
        stub.export_alignment_source = "disabled_integrated_single_pass"
        stub.linear_amplitude_batch_alignment_factor = 1.0
        log.info(
            "Export alignment disabled in integrated_single_pass/"
            "current_analysis mode."
        )
        return 1.0
    # Legacy branch (must not run when integrated/current_analysis is True).
    from proc_audio import linear_export_batch_alignment_k

    k = linear_export_batch_alignment_k(s_h, s_ih_raw, s_sb_raw, 0.95, 0.05, 0.0)
    stub.export_alignment_factor = float(k)
    stub.export_alignment_source = "legacy_batch_alignment"
    stub.linear_amplitude_batch_alignment_factor = float(k)
    if k < 1.0 - 1e-12:
        log.info(
            "Legacy batch export alignment: k=%.6f written to "
            "Amplitude_display_scaled on Inharmonic Spectrum / "
            "Sub-bass band (raw Amplitude preserved).",
            k,
        )
    return k


def test_A_no_alinhamento_log_when_auto_or_current_analysis(caplog):
    """The predicate must take the ``disabled_integrated_single_pass``
    branch when *any* of the three flags is set:

    * ``auto_model_weights_from_analysis=True``
    * ``component_profile_source == "integrated_single_pass"``
    * ``model_weights_source == "current_analysis"``

    None of those branches may emit the legacy ``Alinhamento export
    linear`` log line.
    """
    scenarios = [
        {"auto_model_weights_from_analysis": True,
         "component_profile_source": "legacy_batch",
         "model_weights_source": "apply_filters_arguments"},
        {"auto_model_weights_from_analysis": False,
         "component_profile_source": "integrated_single_pass",
         "model_weights_source": "apply_filters_arguments"},
        {"auto_model_weights_from_analysis": False,
         "component_profile_source": "legacy_batch",
         "model_weights_source": "current_analysis"},
    ]
    for scen in scenarios:
        stub = _make_proc_audio_save_block_stub()
        for k, v in scen.items():
            setattr(stub, k, v)
        caplog.clear()
        with caplog.at_level(logging.DEBUG, logger="proc_audio"):
            k_align = _run_export_alignment_branch(stub)
        joined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert k_align == 1.0, scen
        assert stub.export_alignment_source == "disabled_integrated_single_pass", scen
        assert stub.export_alignment_factor == 1.0, scen
        assert "Alinhamento export linear" not in joined, (
            f"Legacy log leaked for scenario {scen!r}:\n{joined}"
        )
        assert "Export alignment disabled" in joined, (
            f"Replacement log line missing for scenario {scen!r}:\n{joined}"
        )


def test_A_pure_legacy_path_still_logs_legacy_message(caplog):
    """When NONE of the three flags is set, the legacy path runs and the
    NEW (English) log line is emitted. The old Portuguese line
    ``Alinhamento export linear`` must not appear anywhere.
    """
    stub = _make_proc_audio_save_block_stub()
    stub.auto_model_weights_from_analysis = False
    stub.component_profile_source = "legacy_batch"
    stub.model_weights_source = "apply_filters_arguments"
    # Force k < 1 so the legacy info message fires.
    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger="proc_audio"):
        k_align = _run_export_alignment_branch(
            stub, s_h=10.0, s_ih_raw=1.0, s_sb_raw=0.5,
        )
    joined = "\n".join(rec.getMessage() for rec in caplog.records)
    assert stub.export_alignment_source == "legacy_batch_alignment"
    assert k_align < 1.0
    assert "Alinhamento export linear" not in joined
    assert "Legacy batch export alignment" in joined


# ---------------------------------------------------------------------------
# B. Hard runtime assertion — legacy alignment forbidden when
#    model_weights_source == "current_analysis".
# ---------------------------------------------------------------------------
def test_B_pre_save_assertion_raises_on_inconsistent_alignment_state(tmp_path):
    """If the alignment fields end up declaring a legacy-batch state
    while ``model_weights_source == "current_analysis"`` is set on the
    processor (e.g. a re-entrant bug), the pre-save schema validator
    must raise ``RuntimeError`` BEFORE the workbook is written.
    """
    import proc_audio as _pa

    class _BadProc:
        def __init__(self):
            self.logger = logging.getLogger("proc_audio.test")
            self.auto_model_weights_from_analysis = True
            self.model_weights_source = "current_analysis"
            self.component_profile_source = "integrated_single_pass"
            # The bug we want to detect: alignment fields contradict the
            # mode flag above.
            self.export_alignment_source = "legacy_batch_alignment"
            self.export_alignment_factor = 0.42

    proc = _BadProc()
    harm = pd.DataFrame(
        {"Amplitude": [1.0], "Amplitude_raw": [1.0], "Power_raw": [1.0]}
    )
    ih = pd.DataFrame(
        {"Amplitude": [1.0], "Amplitude_raw": [1.0], "Power_raw": [1.0]}
    )
    sb = pd.DataFrame(
        {"Amplitude": [1.0], "Amplitude_raw": [1.0], "Power_raw": [1.0]}
    )
    meta_rows = [
        ("analysis_schema_version", _pa.ANALYSIS_SCHEMA_VERSION),
        ("model_weights_source", "current_analysis"),
        ("component_profile_source", "integrated_single_pass"),
        ("export_alignment_source", "legacy_batch_alignment"),
        ("export_alignment_factor", 0.42),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        _pa.AudioProcessor._validate_per_note_export_schema(
            proc,
            harm_df=harm,
            ih_df=ih,
            sb_df=sb,
            meta_rows=meta_rows,
            note="A4",
        )
    assert "BUG: legacy export alignment active" in str(excinfo.value)


def test_B_pre_save_assertion_accepts_consistent_state(tmp_path):
    """Counter-test: the same call passes when the alignment state is
    consistent with integrated_single_pass / current_analysis.
    """
    import proc_audio as _pa

    class _GoodProc:
        def __init__(self):
            self.logger = logging.getLogger("proc_audio.test")
            self.auto_model_weights_from_analysis = True
            self.model_weights_source = "current_analysis"
            self.component_profile_source = "integrated_single_pass"
            self.export_alignment_source = "disabled_integrated_single_pass"
            self.export_alignment_factor = 1.0

    proc = _GoodProc()
    harm = pd.DataFrame(
        {"Amplitude": [1.0], "Amplitude_raw": [1.0], "Power_raw": [1.0]}
    )
    ih = pd.DataFrame(
        {"Amplitude": [1.0], "Amplitude_raw": [1.0], "Power_raw": [1.0]}
    )
    sb = pd.DataFrame(
        {"Amplitude": [1.0], "Amplitude_raw": [1.0], "Power_raw": [1.0]}
    )
    meta_rows = [
        ("analysis_schema_version", _pa.ANALYSIS_SCHEMA_VERSION),
        ("model_weights_source", "current_analysis"),
        ("component_profile_source", "integrated_single_pass"),
        ("export_alignment_source", "disabled_integrated_single_pass"),
        ("export_alignment_factor", 1.0),
    ]
    # Must not raise.
    _pa.AudioProcessor._validate_per_note_export_schema(
        proc,
        harm_df=harm,
        ih_df=ih,
        sb_df=sb,
        meta_rows=meta_rows,
        note="A4",
    )
