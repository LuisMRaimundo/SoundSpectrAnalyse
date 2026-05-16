"""Audit tests for the direct per-note ``spectral_analysis.xlsx`` →
``Density_Metrics`` extraction path.

Covers the audit's points A–E and the hard contract on the compiled
``Density_Metrics`` sheet layout:

A. Direct extraction from a per-note workbook.
B. Compiled workbook shape (new weighted columns present, old six-column
   layout no longer acceptable on its own).
C. No batch_* leakage in integrated_single_pass mode.
D. Run-relative max-normalisation.
E. GUI default plot metric for Density_Metrics.

Hard contract test:
   ``Density_Metrics`` must always carry ``density_metric_raw``,
   ``density_metric_normalized`` and the three
   ``weighted_*_density_contribution`` columns. A regression that
   resurrects the six-column layout fails immediately.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import compile_metrics as cm
import publication_chart_policy as pcp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _write_per_note_workbook(
    path: Path,
    *,
    note: str,
    harmonic_values,
    inharmonic_values,
    subbass_values,
    w_H: float | None = 0.8,
    w_I: float | None = 0.15,
    w_S: float | None = 0.05,
    inject_batch_only: bool = False,
    inject_misleading_batch: bool = False,
    write_raw_columns: bool = True,
    write_power_raw: bool = False,
    write_display_scaled_only: bool = False,
) -> None:
    """Write a minimal but realistic per-note ``spectral_analysis.xlsx``.

    Sheet layout mirrors what ``proc_audio._save_spectral_data_to_excel``
    emits: ``Harmonic Spectrum`` / ``Inharmonic Spectrum`` / ``Sub-bass band``
    carry ``Amplitude_raw`` + ``Power_raw`` (preferred by the audit-canonical
    direct extractor) and an ``Amplitude`` column for legacy compatibility.
    ``Analysis_Metadata`` is in the canonical (Parameter, Value) long form.

    Knobs:
    * ``write_raw_columns`` (default True) — emit Amplitude_raw / Power_raw.
      Disable to simulate pre-audit workbooks: the direct extractor must
      then fall back to ``Amplitude`` and tag the row
      ``legacy_scaled_source_used``.
    * ``write_display_scaled_only`` (default False) — only emit
      ``Amplitude_display_scaled`` (no ``Amplitude_raw`` / ``Amplitude``)
      to verify the extractor never silently lands on the forbidden
      display-scaled column.
    """
    def _build(rows_dict, amps_list):
        df = pd.DataFrame(rows_dict)
        arr = np.asarray(amps_list, dtype=float)
        if write_display_scaled_only:
            # Forbidden column only — the extractor must skip it.
            df["Amplitude_display_scaled"] = arr * 0.5
            df.drop(columns=["Amplitude"], inplace=True, errors="ignore")
            return df
        if write_raw_columns:
            df["Amplitude_raw"] = arr
            if write_power_raw:
                df["Power_raw"] = arr ** 2
        return df

    harm_df = _build(
        {
            "Harmonic Number": list(range(1, len(harmonic_values) + 1)),
            "Frequency (Hz)": np.linspace(220.0, 220.0 * (len(harmonic_values) + 1), len(harmonic_values) or 1)[: len(harmonic_values)],
            "Magnitude (dB)": [-30.0] * len(harmonic_values),
            "Amplitude": list(harmonic_values),
            "Note": [note] * len(harmonic_values),
        },
        harmonic_values,
    )
    inharm_df = _build(
        {
            "Component_Type": ["inharmonic_partial"] * len(inharmonic_values),
            "Frequency (Hz)": np.linspace(330.0, 330.0 + 10.0 * len(inharmonic_values), len(inharmonic_values) or 1)[: len(inharmonic_values)],
            "Magnitude (dB)": [-35.0] * len(inharmonic_values),
            "Amplitude": list(inharmonic_values),
            "Note": [note] * len(inharmonic_values),
        },
        inharmonic_values,
    )
    subbass_df = _build(
        {
            "Component_Type": ["subbass_noise"] * len(subbass_values),
            "Frequency (Hz)": np.linspace(40.0, 40.0 + 5.0 * len(subbass_values), len(subbass_values) or 1)[: len(subbass_values)],
            "Magnitude (dB)": [-45.0] * len(subbass_values),
            "Amplitude": list(subbass_values),
            "Note": [note] * len(subbass_values),
        },
        subbass_values,
    )

    # AUDIT FIX (stale-pipeline guard) — the schema-version token
    # must be present on every per-note workbook the canonical
    # extractor consumes; tests that exercise the extractor inherit
    # the same contract.
    from proc_audio import ANALYSIS_SCHEMA_VERSION as _ASV

    am_rows: list[tuple[str, object]] = [
        ("analysis_schema_version", _ASV),
        ("export_alignment_source", "disabled_integrated_single_pass"),
        ("export_alignment_factor", 1.0),
        ("source_file_name", str(path)),
        ("tier", "Tier_test"),
        ("model_weights_source", "current_analysis"),
        ("component_profile_source", "integrated_single_pass"),
        ("component_energy_method", "single_pass_partial_amplitude_sums"),
        ("component_energy_denominator", "H + I + S"),
    ]
    if not inject_batch_only:
        if w_H is not None:
            am_rows.append(("component_harmonic_energy_ratio", float(w_H)))
        if w_I is not None:
            am_rows.append(("component_inharmonic_energy_ratio", float(w_I)))
        if w_S is not None:
            am_rows.append(("component_subbass_energy_ratio", float(w_S)))
    if inject_misleading_batch or inject_batch_only:
        am_rows.append(("batch_harmonic_energy_ratio", 0.1))
        am_rows.append(("batch_inharmonic_energy_ratio", 0.1))
        am_rows.append(("batch_subbass_energy_ratio", 0.1))
    am_df = pd.DataFrame(am_rows, columns=["Parameter", "Value"])

    # A small Metrics sheet is also produced by proc_audio; the extraction
    # path must NOT depend on its scalar columns. We intentionally omit
    # the per-band sums here to prove the new path reads from the spectrum
    # sheets directly.
    metrics_df = pd.DataFrame(
        [
            {
                "Note": note,
                "source_file_name": str(path),
            }
        ]
    )

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        harm_df.to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        inharm_df.to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        subbass_df.to_excel(writer, sheet_name="Sub-bass band", index=False)
        am_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        metrics_df.to_excel(writer, sheet_name="Metrics", index=False)


# ---------------------------------------------------------------------------
# A. Direct extraction from a per-note workbook
# ---------------------------------------------------------------------------
def test_direct_extraction_matches_audit_specification(tmp_path: Path) -> None:
    """For the audit's example synthetic workbook:

        Harmonic spectrum values   = [10, 20, 30]   →  D_H = 60
        Inharmonic spectrum values = [5, 5]         →  D_I = 10
        Sub-bass spectrum values   = [2]            →  D_S = 2
        w_H = 0.8, w_I = 0.15, w_S = 0.05
        density_metric_raw = 60*0.8 + 10*0.15 + 2*0.05 = 49.6
    """
    wb = tmp_path / "audit_example_spectral_analysis.xlsx"
    _write_per_note_workbook(
        wb,
        note="A4",
        harmonic_values=[10, 20, 30],
        inharmonic_values=[5, 5],
        subbass_values=[2],
        w_H=0.8, w_I=0.15, w_S=0.05,
    )
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "ok"
    assert info["D_H"] == pytest.approx(60.0)
    assert info["D_I"] == pytest.approx(10.0)
    assert info["D_S"] == pytest.approx(2.0)
    assert info["w_H"] == pytest.approx(0.8)
    assert info["w_I"] == pytest.approx(0.15)
    assert info["w_S"] == pytest.approx(0.05)
    assert info["harmonic_spectrum_count"] == 3
    assert info["inharmonic_spectrum_count"] == 2
    assert info["subbass_spectrum_count"] == 1
    assert "sheet=Harmonic Spectrum" in info["harmonic_spectrum_source"]
    assert "sheet=Inharmonic Spectrum" in info["inharmonic_spectrum_source"]
    assert "sheet=Sub-bass band" in info["subbass_spectrum_source"]

    # End-to-end through the compiler: build Density_Metrics from this single file.
    out_df = cm._build_density_metrics_sheet_from_per_note_files(
        [(wb, "A4", "Note_A4_0")], weight_function="linear"
    )
    row = out_df.iloc[0]
    assert row["Harmonic Partials sum"] == pytest.approx(60.0)
    assert row["Inharmonic Partials sum"] == pytest.approx(10.0)
    assert row["Sub-bass sum"] == pytest.approx(2.0)
    assert row["Total sum"] == pytest.approx(72.0)
    assert row["weighted_harmonic_density_contribution"] == pytest.approx(48.0)
    assert row["weighted_inharmonic_density_contribution"] == pytest.approx(1.5)
    assert row["weighted_subbass_density_contribution"] == pytest.approx(0.1)
    assert row["density_metric_raw"] == pytest.approx(49.6)
    # A single-row workbook normalises to exactly 1.0.
    assert row["density_metric_normalized"] == pytest.approx(1.0)


def test_extraction_status_when_harmonic_sheet_missing(tmp_path: Path) -> None:
    wb = tmp_path / "no_harmonic_spectral_analysis.xlsx"
    from proc_audio import ANALYSIS_SCHEMA_VERSION as _ASV

    am_df = pd.DataFrame(
        [
            ("analysis_schema_version", _ASV),
            ("model_weights_source", "current_analysis"),
            ("export_alignment_source", "disabled_integrated_single_pass"),
            ("export_alignment_factor", 1.0),
            ("component_profile_source", "integrated_single_pass"),
            ("component_harmonic_energy_ratio", 0.8),
            ("component_inharmonic_energy_ratio", 0.15),
            ("component_subbass_energy_ratio", 0.05),
        ],
        columns=["Parameter", "Value"],
    )
    inharm_df = pd.DataFrame({"Amplitude": [5, 5]})
    subbass_df = pd.DataFrame({"Amplitude": [2]})
    with pd.ExcelWriter(wb, engine="xlsxwriter") as writer:
        # Intentionally NO ``Harmonic Spectrum`` sheet.
        inharm_df.to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        subbass_df.to_excel(writer, sheet_name="Sub-bass band", index=False)
        am_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "missing_harmonic_spectrum"
    assert info["D_H"] is None


def test_extraction_status_when_component_weights_missing(tmp_path: Path) -> None:
    wb = tmp_path / "no_weights_spectral_analysis.xlsx"
    _write_per_note_workbook(
        wb,
        note="A4",
        harmonic_values=[10, 20, 30],
        inharmonic_values=[5, 5],
        subbass_values=[2],
        w_H=None, w_I=None, w_S=None,
    )
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "missing_component_weights"
    # Spectra were readable…
    assert info["D_H"] == pytest.approx(60.0)
    assert info["D_I"] == pytest.approx(10.0)
    assert info["D_S"] == pytest.approx(2.0)
    # …but weights are not set.
    assert info["w_H"] is None
    assert info["w_I"] is None
    assert info["w_S"] is None


# ---------------------------------------------------------------------------
# B. Compiled workbook shape — hard contract
# ---------------------------------------------------------------------------
REQUIRED_DENSITY_METRICS_COLUMNS = (
    "density_metric_raw",
    "density_metric_normalized",
    "weighted_harmonic_density_contribution",
    "weighted_inharmonic_density_contribution",
    "weighted_subbass_density_contribution",
)


def test_compiled_density_metrics_sheet_contains_audit_mandated_columns(tmp_path: Path) -> None:
    """HARD CONTRACT — the compiled Density_Metrics sheet must carry the
    new weighted columns. A regression that reverts to the old six-column
    layout fails this test immediately.
    """
    # Build a small synthetic corpus: 3 notes, each in its own folder so
    # compile_density_metrics' folder-walk discovers them.
    notes = [("A3", [10.0, 20.0, 30.0], [5.0, 5.0], [2.0], 0.80, 0.15, 0.05),
             ("B3", [5.0, 10.0, 15.0], [3.0, 3.0], [1.0], 0.70, 0.20, 0.10),
             ("C4", [2.0, 4.0, 6.0],  [1.0, 1.0], [0.5], 0.60, 0.30, 0.10)]
    root = tmp_path / "corpus"
    root.mkdir()
    for i, (note, h, ih, sb, w_H, w_I, w_S) in enumerate(notes):
        d = root / f"Note_{note}_{i}"
        d.mkdir()
        _write_per_note_workbook(
            d / "spectral_analysis.xlsx",
            note=note,
            harmonic_values=h, inharmonic_values=ih, subbass_values=sb,
            w_H=w_H, w_I=w_I, w_S=w_S,
        )
    out_path = tmp_path / "compiled.xlsx"
    df = cm.compile_density_metrics(
        root,
        output_path=out_path,
        file_pattern="spectral_analysis.xlsx",
        enable_pca_export=False,
    )
    assert df is not None and not df.empty

    dm = pd.read_excel(out_path, sheet_name="Density_Metrics")

    # 1. Must NOT be the bare six-column legacy layout.
    legacy_six = {
        "Note",
        "weight_function",
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Sub-bass sum",
        "Total sum",
    }
    assert set(dm.columns) != legacy_six, (
        "Density_Metrics regressed to the old six-column layout. "
        "The weighted density columns must be present in every compiled workbook."
    )

    # 2. All audit-mandated columns must be present.
    missing = [c for c in REQUIRED_DENSITY_METRICS_COLUMNS if c not in dm.columns]
    assert not missing, (
        f"Density_Metrics is missing audit-mandated columns: {missing}. "
        f"Density_Metrics columns: {list(dm.columns)}"
    )

    # 3. The provenance / status columns must be present too.
    for c in (
        "density_extraction_status",
        "harmonic_spectrum_source",
        "inharmonic_spectrum_source",
        "subbass_spectrum_source",
        "harmonic_spectrum_count",
        "inharmonic_spectrum_count",
        "subbass_spectrum_count",
        "source_file_name",
    ):
        assert c in dm.columns, c

    # 4. Numerical sanity: density_metric_raw matches the formula.
    h = pd.to_numeric(dm["Harmonic Partials sum"], errors="coerce").to_numpy(dtype=float)
    i = pd.to_numeric(dm["Inharmonic Partials sum"], errors="coerce").to_numpy(dtype=float)
    s = pd.to_numeric(dm["Sub-bass sum"], errors="coerce").to_numpy(dtype=float)
    w_H = pd.to_numeric(dm["component_harmonic_energy_ratio"], errors="coerce").to_numpy(dtype=float)
    w_I = pd.to_numeric(dm["component_inharmonic_energy_ratio"], errors="coerce").to_numpy(dtype=float)
    w_S = pd.to_numeric(dm["component_subbass_energy_ratio"], errors="coerce").to_numpy(dtype=float)
    expected = h * w_H + i * w_I + s * w_S
    raw = pd.to_numeric(dm["density_metric_raw"], errors="coerce").to_numpy(dtype=float)
    np.testing.assert_allclose(raw, expected, rtol=0.0, atol=1e-9)

    # 5. density_metric_normalized = raw / max(raw); in [0, 1].
    norm = pd.to_numeric(dm["density_metric_normalized"], errors="coerce").to_numpy(dtype=float)
    mx = float(np.nanmax(raw))
    np.testing.assert_allclose(norm, raw / mx, rtol=0.0, atol=1e-9)
    assert np.nanmin(norm) >= 0.0 - 1e-9
    assert np.nanmax(norm) <= 1.0 + 1e-9

    # 6. Status is "ok" for every row that has all three spectra + weights.
    assert (dm["density_extraction_status"] == "ok").all()

    # 7. All rows record finite spectrum counts (>0 here because each
    # synthetic workbook ships harmonic/inharmonic/subbass amplitudes).
    assert (dm["harmonic_spectrum_count"] > 0).all()
    assert (dm["inharmonic_spectrum_count"] > 0).all()
    assert (dm["subbass_spectrum_count"] > 0).all()


# ---------------------------------------------------------------------------
# C. No batch_* leakage in integrated_single_pass mode
# ---------------------------------------------------------------------------
def test_extraction_prefers_component_weights_over_misleading_batch(tmp_path: Path) -> None:
    """When BOTH ``component_*`` AND ``batch_*`` weight rows exist, the
    extractor must pick the canonical ``component_*`` values regardless of
    how misleading the batch_* values are."""
    wb = tmp_path / "spec.xlsx"
    _write_per_note_workbook(
        wb,
        note="A4",
        harmonic_values=[10, 20, 30],
        inharmonic_values=[5, 5],
        subbass_values=[2],
        w_H=0.8, w_I=0.15, w_S=0.05,
        inject_misleading_batch=True,  # batch_* = 0.1/0.1/0.1
    )
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "ok"
    assert info["w_H"] == pytest.approx(0.8)
    assert info["w_I"] == pytest.approx(0.15)
    assert info["w_S"] == pytest.approx(0.05)
    # The numerical answer corresponds to the canonical weights only.
    out_df = cm._build_density_metrics_sheet_from_per_note_files(
        [(wb, "A4", "Note_A4_0")], weight_function="linear"
    )
    assert out_df.iloc[0]["density_metric_raw"] == pytest.approx(49.6)


def test_extraction_does_not_silently_use_batch_only_weights(tmp_path: Path) -> None:
    """If only legacy ``batch_*`` alias rows are present (no ``component_*``),
    the extractor must flag ``missing_component_weights`` rather than
    silently substituting the legacy values."""
    wb = tmp_path / "spec_batch_only.xlsx"
    _write_per_note_workbook(
        wb,
        note="A4",
        harmonic_values=[10, 20, 30],
        inharmonic_values=[5, 5],
        subbass_values=[2],
        inject_batch_only=True,
    )
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "missing_component_weights"
    # ``legacy_aliases_only`` is the renamed flag in the Stage 1 + Stage 2
    # extractor; ``legacy_batch_only`` is kept here as a deprecated key check
    # but allowed to be absent in the new schema.
    assert info.get("legacy_aliases_only", info.get("legacy_batch_only")) is True
    assert info["w_H"] is None and info["w_I"] is None and info["w_S"] is None


# ---------------------------------------------------------------------------
# D. Run-relative max-normalisation
# ---------------------------------------------------------------------------
def test_run_relative_normalisation_two_files(tmp_path: Path) -> None:
    """For two workbooks whose raw values are [49.6, 24.8], the normalised
    values must be [1.0, 0.5]."""
    wb1 = tmp_path / "n1_spec.xlsx"
    _write_per_note_workbook(
        wb1, note="A4",
        harmonic_values=[10, 20, 30],     # D_H = 60
        inharmonic_values=[5, 5],         # D_I = 10
        subbass_values=[2],               # D_S = 2
        w_H=0.8, w_I=0.15, w_S=0.05,
    )
    wb2 = tmp_path / "n2_spec.xlsx"
    _write_per_note_workbook(
        wb2, note="B4",
        harmonic_values=[5, 10, 15],      # D_H = 30
        inharmonic_values=[2.5, 2.5],     # D_I = 5
        subbass_values=[1],               # D_S = 1
        w_H=0.8, w_I=0.15, w_S=0.05,
    )
    out_df = cm._build_density_metrics_sheet_from_per_note_files(
        [(wb1, "A4", "f1"), (wb2, "B4", "f2")], weight_function="linear"
    )
    raw = out_df["density_metric_raw"].astype(float).tolist()
    norm = out_df["density_metric_normalized"].astype(float).tolist()
    assert raw == [pytest.approx(49.6), pytest.approx(24.8)]
    assert norm == [pytest.approx(1.0), pytest.approx(0.5)]


def test_no_per_note_files_returns_compilation_error(tmp_path: Path) -> None:
    out_df = cm._build_density_metrics_sheet_from_per_note_files(
        [], weight_function="linear"
    )
    assert "compilation_error" in out_df.columns


# ---------------------------------------------------------------------------
# E. GUI default plot metric for Density_Metrics
# ---------------------------------------------------------------------------
def test_density_metrics_default_plot_is_density_log_weighted() -> None:
    """When the active sheet is ``Density_Metrics``, the publication
    policy must return ``density_log_weighted`` as the default (Stage 2
    weighted note-density audit) and must NOT return any of the raw
    partial sums or the workbook-relative ``density_metric_normalized``."""
    cols = [
        "Note",
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Sub-bass sum",
        "Total sum",
        "density_metric_raw",
        "density_metric_normalized",
        "density_log_weighted",
        "harmonic_log_amplitude_density",
    ]
    chosen = pcp.select_default_publication_metric(cols, sheet_name="Density_Metrics")
    assert chosen == "density_log_weighted"
    for forbidden in ("Harmonic Partials sum", "Inharmonic Partials sum",
                      "Sub-bass sum", "Total sum"):
        assert pcp.classify_metric_for_publication(forbidden) == "legacy"
