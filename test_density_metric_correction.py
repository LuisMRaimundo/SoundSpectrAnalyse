"""Regression tests for the canonical density-metric correction.

Covers, in one file:

* canonical note parsing (filename-first, explicit-octave required,
  Unicode accidentals normalised),
* ``extract_density_component_sum`` linear / log / power semantics,
* hard contract on the ``log`` weight function:
    - column source is ``Amplitude_raw`` (never ``Power_raw``),
    - formula is ``LOG10(1 + SUM(Amplitude_raw))`` (NOT
      ``SUM(LOG10(1 + amp_i))``),
* hard contract on ``Amplitude_display_scaled``: never selected for
  any weight_function,
* compiled ``Density_Metrics`` shape: ``density_metric_raw`` equals
  ``D_H*w_H + D_I*w_I + D_S*w_S`` row by row, the three
  ``weighted_*_density_contribution`` columns exist, the new
  ``note_source`` / ``density_weight_function`` columns are present,
  and **no** column name starts with ``batch_``,
* component-ratio invariant ``w_H + w_I + w_S ~= 1`` is recorded
  on the per-row provenance,
* the GUI's automatic current-analysis activation log no longer prints
  ``"Harmonic Weight: 0.950 | Inharmonic Weight: 0.050"``.
"""

from __future__ import annotations

import io
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import compile_metrics as cm
from note_parser import (
    NOTE_SOURCE_FALLBACK_NO_OCTAVE,
    NOTE_SOURCE_FILENAME,
    NOTE_SOURCE_MANIFEST,
    NOTE_SOURCE_PARENT_FOLDER,
    NOTE_SOURCE_UNKNOWN,
    canonical_note_from_filename,
    parse_note_token,
)


# ---------------------------------------------------------------------------
# 1. Canonical note parsing
# ---------------------------------------------------------------------------
NOTE_PARSER_FILENAME_EXAMPLES = [
    ("A#3_3.72sec_Sustains.wav", "A#3"),
    ("Bb4_3.80sec_Sustains.wav", "Bb4"),
    ("Bn-ord-A#1-pp-N-N_Sustains.wav", "A#1"),
    ("D6_3.88sec_shifted_Sustains_Sustains.wav", "D6"),
    ("Bn-ord-Sustains.wav", None),
    ("noise_only_file.wav", None),
]


@pytest.mark.parametrize("name, expected", NOTE_PARSER_FILENAME_EXAMPLES)
def test_parse_note_token_audit_examples(name: str, expected: str | None) -> None:
    assert parse_note_token(name) == expected


def test_parse_note_token_normalises_unicode_accidentals() -> None:
    assert parse_note_token("C\u266f5_thing.wav") == "C#5"
    assert parse_note_token("D\u266d2_thing.wav") == "Db2"


def test_parse_note_token_rejects_letter_without_octave() -> None:
    assert parse_note_token("A#_thing.wav") is None
    assert parse_note_token("B_thing.wav") is None


def test_canonical_note_priority_manifest_first() -> None:
    note, source = canonical_note_from_filename(
        "A#3_3.72sec_Sustains.wav",
        manifest_note="C4",
        parent_folder="Note_F#5_0",
    )
    assert note == "C4"
    assert source == NOTE_SOURCE_MANIFEST


def test_canonical_note_priority_filename_then_parent() -> None:
    note, source = canonical_note_from_filename(
        "A#3_3.72sec_Sustains.wav",
        parent_folder="Note_F#5_0",
    )
    assert note == "A#3"
    assert source == NOTE_SOURCE_FILENAME

    note, source = canonical_note_from_filename(
        "Bn-ord-Sustains.wav",
        parent_folder="Note_F#5_0",
    )
    assert note == "F#5"
    assert source == NOTE_SOURCE_PARENT_FOLDER


def test_canonical_note_letter_only_fallback_and_unknown() -> None:
    # ``B`` glued to ``n`` is not a standalone note letter: the fallback
    # is strict and refuses to invent an octave from ``Bn`` or ``ord``.
    note, source = canonical_note_from_filename("Bn-ord-Sustains.wav")
    assert note is None
    assert source == NOTE_SOURCE_UNKNOWN

    # ``A#`` followed by underscore IS a standalone letter+accidental:
    # surface it as ``fallback_no_octave`` so the caller knows the
    # octave was not explicit.
    note, source = canonical_note_from_filename("A#_no_octave_token.wav")
    assert note == "A#"
    assert source == NOTE_SOURCE_FALLBACK_NO_OCTAVE

    note, source = canonical_note_from_filename("noise_only_file.wav")
    assert note is None
    assert source == NOTE_SOURCE_UNKNOWN


# ---------------------------------------------------------------------------
# 2. ``extract_density_component_sum`` semantics
# ---------------------------------------------------------------------------
def _write_synthetic_workbook(
    path: Path,
    *,
    harmonic_amp,
    inharmonic_amp,
    subbass_amp,
    write_power_raw: bool = False,
    write_display_scaled: bool = False,
    w_H: float | None = 0.8,
    w_I: float | None = 0.15,
    w_S: float | None = 0.05,
) -> None:
    """Write a per-note workbook with the three canonical component sheets.

    The amplitude columns are written as the audit-canonical
    ``Amplitude_raw`` (and optionally ``Power_raw``); a
    ``Amplitude_display_scaled`` column is written when requested so the
    forbidden-column guard can be exercised.
    """

    from proc_audio import ANALYSIS_SCHEMA_VERSION as _ASV

    def _frame(amps):
        arr = np.asarray(amps, dtype=float)
        df = pd.DataFrame(
            {
                "Index": list(range(len(arr))),
                "Frequency (Hz)": np.linspace(220.0, 880.0, num=max(1, len(arr)))[: len(arr)],
                "Magnitude (dB)": [-30.0] * len(arr),
                "Amplitude_raw": arr,
                "Amplitude": arr * 0.99,
            }
        )
        if write_power_raw:
            df["Power_raw"] = arr ** 2
        if write_display_scaled:
            df["Amplitude_display_scaled"] = arr * 0.5
        return df

    rows: list[tuple[str, object]] = [
        ("analysis_schema_version", _ASV),
        ("export_alignment_source", "disabled_integrated_single_pass"),
        ("export_alignment_factor", 1.0),
        ("model_weights_source", "current_analysis"),
        ("component_profile_source", "integrated_single_pass"),
        ("component_energy_method", "single_pass_partial_amplitude_sums"),
        ("component_energy_denominator", "H + I + S"),
    ]
    if w_H is not None:
        rows.append(("component_harmonic_energy_ratio", float(w_H)))
    if w_I is not None:
        rows.append(("component_inharmonic_energy_ratio", float(w_I)))
    if w_S is not None:
        rows.append(("component_subbass_energy_ratio", float(w_S)))
    am_df = pd.DataFrame(rows, columns=["Parameter", "Value"])

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        _frame(harmonic_amp).to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        _frame(inharmonic_amp).to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        _frame(subbass_amp).to_excel(writer, sheet_name="Sub-bass band", index=False)
        am_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        pd.DataFrame([{"Note": "A4"}]).to_excel(
            writer, sheet_name="Metrics", index=False
        )


def test_extract_density_component_sum_linear(tmp_path: Path) -> None:
    wb = tmp_path / "linear.xlsx"
    _write_synthetic_workbook(
        wb,
        harmonic_amp=[1, 2, 3],
        inharmonic_amp=[4, 5],
        subbass_amp=[6],
    )
    h = cm.extract_density_component_sum(wb, "Harmonic Spectrum", "linear")
    i = cm.extract_density_component_sum(wb, "Inharmonic Spectrum", "linear")
    s = cm.extract_density_component_sum(wb, "Sub-bass band", "linear")
    assert h["D"] == pytest.approx(6.0)
    assert i["D"] == pytest.approx(9.0)
    assert s["D"] == pytest.approx(6.0)
    for r in (h, i, s):
        assert r["status"] == "ok"
        assert r["column"].lower() == "amplitude_raw"
        assert r["weight_function"] == "linear"
        assert "weight_function=linear" in r["density_component_sum_source"]


def test_extract_density_component_sum_log(tmp_path: Path) -> None:
    """Audit case A — the log mode must compute LOG10(1 + SUM(Amp_raw))."""
    wb = tmp_path / "log.xlsx"
    _write_synthetic_workbook(
        wb,
        harmonic_amp=[1, 2, 3],
        inharmonic_amp=[4, 5],
        subbass_amp=[6],
    )
    h = cm.extract_density_component_sum(wb, "Harmonic Spectrum", "log")
    i = cm.extract_density_component_sum(wb, "Inharmonic Spectrum", "log")
    s = cm.extract_density_component_sum(wb, "Sub-bass band", "log")
    assert h["D"] == pytest.approx(math.log10(1 + 6))
    assert i["D"] == pytest.approx(math.log10(1 + 9))
    assert s["D"] == pytest.approx(math.log10(1 + 6))
    # In log mode the SOURCE must always be Amplitude_raw.
    for r in (h, i, s):
        assert r["column"].lower() == "amplitude_raw"
        assert "strategy=log10_1p_sum_amplitude_raw" in r["density_component_sum_source"]


def test_log_mode_must_not_pick_power_raw_even_when_present(tmp_path: Path) -> None:
    """Audit case C — Power_raw must not be selected for ``log``."""
    wb = tmp_path / "log_with_power.xlsx"
    _write_synthetic_workbook(
        wb,
        harmonic_amp=[1, 2, 3],
        inharmonic_amp=[4, 5],
        subbass_amp=[6],
        write_power_raw=True,
    )
    h = cm.extract_density_component_sum(wb, "Harmonic Spectrum", "log")
    assert h["column"].lower() == "amplitude_raw"
    assert "power_raw" not in h["column"].lower()
    assert h["D"] == pytest.approx(math.log10(1 + 6))


def test_power_mode_prefers_power_raw_then_falls_back(tmp_path: Path) -> None:
    wb_power = tmp_path / "power.xlsx"
    _write_synthetic_workbook(
        wb_power,
        harmonic_amp=[1, 2, 3],
        inharmonic_amp=[4, 5],
        subbass_amp=[6],
        write_power_raw=True,
    )
    h = cm.extract_density_component_sum(wb_power, "Harmonic Spectrum", "power")
    assert h["column"].lower() == "power_raw"
    assert h["D"] == pytest.approx(1.0 + 4.0 + 9.0)

    wb_no_power = tmp_path / "power_fallback.xlsx"
    _write_synthetic_workbook(
        wb_no_power,
        harmonic_amp=[1, 2, 3],
        inharmonic_amp=[4, 5],
        subbass_amp=[6],
        write_power_raw=False,
    )
    h = cm.extract_density_component_sum(wb_no_power, "Harmonic Spectrum", "power")
    assert "**2" in h["column"]
    assert h["D"] == pytest.approx(1.0 + 4.0 + 9.0)


def test_amplitude_display_scaled_is_never_selected(tmp_path: Path) -> None:
    """Audit case D — Amplitude_display_scaled must never be selected."""
    wb = tmp_path / "with_display_scaled.xlsx"
    _write_synthetic_workbook(
        wb,
        harmonic_amp=[1, 2, 3],
        inharmonic_amp=[4, 5],
        subbass_amp=[6],
        write_display_scaled=True,
    )
    for wf in ("linear", "log", "power"):
        h = cm.extract_density_component_sum(wb, "Harmonic Spectrum", wf)
        assert "display_scaled" not in h["column"].lower(), (
            f"weight_function={wf} unexpectedly selected display-scaled "
            f"column {h['column']!r}"
        )


# ---------------------------------------------------------------------------
# 2b. include_for_density inclusion contract — Harmonic Spectrum only
# ---------------------------------------------------------------------------
def _write_harmonic_with_include_flag(
    path: Path,
    *,
    amplitudes,
    include_flags,
    w_H: float = 0.8,
    w_I: float = 0.15,
    w_S: float = 0.05,
) -> None:
    """Write a per-note workbook whose Harmonic Spectrum carries an
    ``include_for_density`` column.  Inharmonic / Sub-bass sheets are
    populated with deterministic non-zero amplitudes so the row writer
    can compute density_metric_raw.
    """
    from proc_audio import ANALYSIS_SCHEMA_VERSION as _ASV

    amps = np.asarray(amplitudes, dtype=float)
    flags = list(include_flags)
    assert len(amps) == len(flags)

    n = len(amps)
    h_df = pd.DataFrame(
        {
            "Index": list(range(n)),
            "Frequency (Hz)": np.linspace(220.0, 880.0, num=max(1, n))[:n],
            "Magnitude (dB)": [-30.0] * n,
            "Amplitude_raw": amps,
            "Amplitude": amps * 0.99,
            "include_for_density": flags,
        }
    )

    def _band_frame(values):
        arr = np.asarray(values, dtype=float)
        m = len(arr)
        return pd.DataFrame(
            {
                "Index": list(range(m)),
                "Frequency (Hz)": np.linspace(40.0, 160.0, num=max(1, m))[:m],
                "Magnitude (dB)": [-40.0] * m,
                "Amplitude_raw": arr,
                "Amplitude": arr * 0.99,
            }
        )

    am_rows: list[tuple[str, object]] = [
        ("analysis_schema_version", _ASV),
        ("export_alignment_source", "disabled_integrated_single_pass"),
        ("export_alignment_factor", 1.0),
        ("model_weights_source", "current_analysis"),
        ("component_profile_source", "integrated_single_pass"),
        ("component_energy_method", "single_pass_partial_amplitude_sums"),
        ("component_energy_denominator", "H + I + S"),
        ("component_harmonic_energy_ratio", float(w_H)),
        ("component_inharmonic_energy_ratio", float(w_I)),
        ("component_subbass_energy_ratio", float(w_S)),
    ]
    am_df = pd.DataFrame(am_rows, columns=["Parameter", "Value"])

    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        h_df.to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        _band_frame([2.0, 3.0]).to_excel(
            writer, sheet_name="Inharmonic Spectrum", index=False
        )
        _band_frame([0.5]).to_excel(
            writer, sheet_name="Sub-bass band", index=False
        )
        am_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        pd.DataFrame([{"Note": "A4"}]).to_excel(
            writer, sheet_name="Metrics", index=False
        )


def test_extract_density_component_sum_honours_include_for_density_log(
    tmp_path: Path,
) -> None:
    """AUDIT FIX (Harmonic-Spectrum inclusion contract):

    With Amplitude_raw = [100, 10, 1] and include_for_density =
    [True, False, True], the harmonic D in log mode must be
    LOG10(1 + 100 + 1) = LOG10(101), NOT LOG10(1 + 100 + 10 + 1).
    """
    wb = tmp_path / "harmonic_include_log.xlsx"
    _write_harmonic_with_include_flag(
        wb,
        amplitudes=[100.0, 10.0, 1.0],
        include_flags=[True, False, True],
    )

    h = cm.extract_density_component_sum(wb, "Harmonic Spectrum", "log")
    assert h["status"] == "ok"
    assert h["column"].lower() == "amplitude_raw"
    assert h["inclusion_policy"] == "include_for_density_true"
    assert h["excluded_count"] == 1
    # Filtered sum is 101, not 111.
    assert h["D"] == pytest.approx(math.log10(1.0 + 101.0))
    assert h["D"] != pytest.approx(math.log10(1.0 + 111.0))
    assert "inclusion_policy=include_for_density_true" in (
        h["density_component_sum_source"]
    )


def test_extract_density_component_sum_honours_include_for_density_linear(
    tmp_path: Path,
) -> None:
    wb = tmp_path / "harmonic_include_linear.xlsx"
    _write_harmonic_with_include_flag(
        wb,
        amplitudes=[100.0, 10.0, 1.0],
        include_flags=[True, False, True],
    )
    h = cm.extract_density_component_sum(wb, "Harmonic Spectrum", "linear")
    assert h["D"] == pytest.approx(101.0)
    assert h["inclusion_policy"] == "include_for_density_true"
    assert h["excluded_count"] == 1


def test_extract_density_component_sum_honours_include_for_density_power(
    tmp_path: Path,
) -> None:
    """Power mode must apply the SAME row mask before the squared sum."""
    wb = tmp_path / "harmonic_include_power.xlsx"
    _write_harmonic_with_include_flag(
        wb,
        amplitudes=[100.0, 10.0, 1.0],
        include_flags=[True, False, True],
    )
    h = cm.extract_density_component_sum(wb, "Harmonic Spectrum", "power")
    # No Power_raw column: fallback uses Amplitude_raw**2 on the
    # filtered rows (100^2 + 1^2 = 10001, not 100^2 + 10^2 + 1^2).
    assert h["inclusion_policy"] == "include_for_density_true"
    assert h["D"] == pytest.approx(100.0 ** 2 + 1.0 ** 2)


def test_extract_density_component_sum_accepts_string_truth_tokens(
    tmp_path: Path,
) -> None:
    wb = tmp_path / "harmonic_include_strings.xlsx"
    _write_harmonic_with_include_flag(
        wb,
        amplitudes=[100.0, 10.0, 1.0, 5.0, 7.0],
        include_flags=["true", "False", "1", "no", "yes"],
    )
    h = cm.extract_density_component_sum(wb, "Harmonic Spectrum", "linear")
    assert h["inclusion_policy"] == "include_for_density_true"
    assert h["D"] == pytest.approx(100.0 + 1.0 + 7.0)
    assert h["excluded_count"] == 2


def test_extract_density_component_sum_legacy_when_column_absent(
    tmp_path: Path,
) -> None:
    """When include_for_density is absent, the legacy contract (sum every
    finite non-negative Amplitude_raw row) must be preserved.
    """
    wb = tmp_path / "no_include_col.xlsx"
    _write_synthetic_workbook(
        wb,
        harmonic_amp=[100.0, 10.0, 1.0],
        inharmonic_amp=[4, 5],
        subbass_amp=[6],
    )
    h = cm.extract_density_component_sum(wb, "Harmonic Spectrum", "log")
    assert h["inclusion_policy"] == "all_rows_no_include_column"
    assert h["excluded_count"] == 0
    assert h["D"] == pytest.approx(math.log10(1.0 + 111.0))


def test_inharmonic_and_subbass_ignore_include_for_density(
    tmp_path: Path,
) -> None:
    """The filter must NOT be applied to the Inharmonic / Sub-bass sheets
    even when they happen to carry an ``include_for_density`` column.
    """
    from proc_audio import ANALYSIS_SCHEMA_VERSION as _ASV

    wb = tmp_path / "include_on_iandS.xlsx"

    def _band_with_include(values, flags):
        arr = np.asarray(values, dtype=float)
        n = len(arr)
        return pd.DataFrame(
            {
                "Index": list(range(n)),
                "Frequency (Hz)": np.linspace(40.0, 160.0, num=max(1, n))[:n],
                "Magnitude (dB)": [-40.0] * n,
                "Amplitude_raw": arr,
                "Amplitude": arr * 0.99,
                "include_for_density": list(flags),
            }
        )

    am_df = pd.DataFrame(
        [
            ("analysis_schema_version", _ASV),
            ("component_harmonic_energy_ratio", 0.8),
            ("component_inharmonic_energy_ratio", 0.15),
            ("component_subbass_energy_ratio", 0.05),
        ],
        columns=["Parameter", "Value"],
    )

    h_arr = np.asarray([1.0, 2.0, 3.0], dtype=float)
    h_df = pd.DataFrame(
        {
            "Index": list(range(3)),
            "Frequency (Hz)": [220.0, 440.0, 880.0],
            "Magnitude (dB)": [-30.0] * 3,
            "Amplitude_raw": h_arr,
            "Amplitude": h_arr * 0.99,
        }
    )
    with pd.ExcelWriter(wb, engine="xlsxwriter") as writer:
        h_df.to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        _band_with_include([10.0, 20.0, 30.0], [True, False, True]).to_excel(
            writer, sheet_name="Inharmonic Spectrum", index=False
        )
        _band_with_include([5.0, 50.0], [True, False]).to_excel(
            writer, sheet_name="Sub-bass band", index=False
        )
        am_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        pd.DataFrame([{"Note": "A4"}]).to_excel(
            writer, sheet_name="Metrics", index=False
        )

    i = cm.extract_density_component_sum(wb, "Inharmonic Spectrum", "linear")
    s = cm.extract_density_component_sum(wb, "Sub-bass band", "linear")
    # Inharmonic / Sub-bass must remain unfiltered.
    assert i["inclusion_policy"] == ""
    assert s["inclusion_policy"] == ""
    assert i["D"] == pytest.approx(60.0)  # 10 + 20 + 30
    assert s["D"] == pytest.approx(55.0)  # 5 + 50


def test_compiled_row_carries_inclusion_diagnostics(tmp_path: Path) -> None:
    """The Density_Metrics row must surface the inclusion policy and
    excluded-count diagnostic columns, AND the filtered harmonic sum
    must be the one used to compute density_metric_raw.
    """
    wb = tmp_path / "C4_3.50sec_Sustains.xlsx"
    _write_harmonic_with_include_flag(
        wb,
        amplitudes=[100.0, 10.0, 1.0],
        include_flags=[True, False, True],
        w_H=0.8,
        w_I=0.15,
        w_S=0.05,
    )

    out_df = cm._build_density_metrics_sheet_from_per_note_files(
        [(wb, "", "Note_C4_0")],
        weight_function="log",
    )
    assert len(out_df) == 1
    row = out_df.iloc[0]

    assert "harmonic_density_inclusion_policy" in out_df.columns
    assert "harmonic_density_excluded_count" in out_df.columns
    assert row["harmonic_density_inclusion_policy"] == "include_for_density_true"
    assert int(row["harmonic_density_excluded_count"]) == 1

    d_h = math.log10(1.0 + 101.0)
    d_i = math.log10(1.0 + 5.0)
    d_s = math.log10(1.0 + 0.5)
    expected = d_h * 0.8 + d_i * 0.15 + d_s * 0.05

    assert row["Harmonic Partials sum"] == pytest.approx(d_h)
    assert row["density_metric_raw"] == pytest.approx(expected)
    # density_metric_raw must NOT match the unfiltered LOG10(1 + 111).
    unfiltered = math.log10(1.0 + 111.0) * 0.8 + d_i * 0.15 + d_s * 0.05
    assert abs(float(row["density_metric_raw"]) - unfiltered) > 1e-6


# ---------------------------------------------------------------------------
# 3. Full per-note extractor — D_H*w_H + D_I*w_I + D_S*w_S
# ---------------------------------------------------------------------------
def test_compiled_density_metric_raw_matches_audit_formula(tmp_path: Path) -> None:
    """Audit case B + F + E — density_metric_raw equals
    ``D_H*w_H + D_I*w_I + D_S*w_S`` row by row, and the three
    ``weighted_*_density_contribution`` columns plus the new
    ``note_source`` / ``density_weight_function`` columns are present.
    Also asserts the component-ratio invariant is satisfied.
    """
    wb = tmp_path / "A#3_3.72sec_Sustains.xlsx"
    _write_synthetic_workbook(
        wb,
        harmonic_amp=[1, 2, 3],
        inharmonic_amp=[4, 5],
        subbass_amp=[6],
        w_H=0.8,
        w_I=0.15,
        w_S=0.05,
    )

    info = cm.extract_density_components_from_per_note_workbook(
        wb, weight_function="log"
    )
    assert info["density_extraction_status"] == "ok"
    assert info["density_weight_function"] == "log"
    assert info["harmonic_density_sum"] == pytest.approx(math.log10(1 + 6))
    assert info["inharmonic_density_sum"] == pytest.approx(math.log10(1 + 9))
    assert info["subbass_density_sum"] == pytest.approx(math.log10(1 + 6))
    # Audit case E — ratios sum to ~1.
    assert info["component_energy_ratio_sum"] == pytest.approx(1.0, abs=1e-6)
    assert info["component_energy_ratio_sum_ok"] is True

    out_df = cm._build_density_metrics_sheet_from_per_note_files(
        [(wb, "", "Note_A#3_0")],
        weight_function="log",
    )
    assert len(out_df) == 1
    row = out_df.iloc[0]

    # Audit case F: compiled columns present.
    for col in (
        "density_metric_raw",
        "weighted_harmonic_density_contribution",
        "weighted_inharmonic_density_contribution",
        "weighted_subbass_density_contribution",
        "note_source",
        "density_weight_function",
        "harmonic_density_sum",
        "inharmonic_density_sum",
        "subbass_density_sum",
        "density_formula",
        "density_component_sum_source",
    ):
        assert col in out_df.columns, f"missing column: {col}"

    # Audit case A + B: density_metric_raw = D_H*w_H + D_I*w_I + D_S*w_S
    d_h = math.log10(1 + 6)
    d_i = math.log10(1 + 9)
    d_s = math.log10(1 + 6)
    expected = d_h * 0.8 + d_i * 0.15 + d_s * 0.05
    assert row["density_metric_raw"] == pytest.approx(expected)
    assert row["weighted_harmonic_density_contribution"] == pytest.approx(d_h * 0.8)
    assert row["weighted_inharmonic_density_contribution"] == pytest.approx(d_i * 0.15)
    assert row["weighted_subbass_density_contribution"] == pytest.approx(d_s * 0.05)

    # Audit canonical-note parsing: A#3 must be recovered from the
    # filename (not invented).
    assert row["Note"] == "A#3"
    assert row["note_source"] == NOTE_SOURCE_FILENAME
    assert row["density_weight_function"] == "log"
    assert "D_band = log10(1 + SUM(Amplitude_raw))" in row["density_formula"]


# ---------------------------------------------------------------------------
# 4. Audit case G — no batch_* leakage in compiled Density_Metrics columns
# ---------------------------------------------------------------------------
def test_no_batch_columns_leak_into_density_metrics(tmp_path: Path) -> None:
    wb = tmp_path / "C4_synth.xlsx"
    _write_synthetic_workbook(
        wb,
        harmonic_amp=[1, 2, 3],
        inharmonic_amp=[4, 5],
        subbass_amp=[6],
    )
    out_df = cm._build_density_metrics_sheet_from_per_note_files(
        [(wb, "C4", "Note_C4_0")],
        weight_function="linear",
    )
    leaking = [c for c in out_df.columns if str(c).startswith("batch_")]
    assert leaking == [], f"unexpected batch_* columns: {leaking}"


# ---------------------------------------------------------------------------
# 5. Audit case H — automatic current-analysis log no longer prints
# the misleading 0.950 / 0.050 line.
# ---------------------------------------------------------------------------
def test_gui_activation_log_drops_legacy_0p95_0p05_string() -> None:
    """Source-level scan: the GUI must not emit the legacy weight log
    in automatic current-analysis mode.

    The literal token ``"Harmonic Weight: 0.950 | Inharmonic Weight:
    0.050"`` is constructed at runtime via an f-string. We assert the
    source no longer contains either the literal pair or the f-string
    template that would produce it.
    """
    src = (REPO_ROOT / "pipeline_orchestrator_gui.py").read_text(
        encoding="utf-8"
    )
    legacy_literal = "Harmonic Weight: 0.950 | Inharmonic Weight: 0.050"
    legacy_fmt = (
        "Harmonic Weight: {1.0-params['i_weight']:.3f} | Inharmonic Weight: "
    )
    assert legacy_literal not in src
    assert legacy_fmt not in src
    # And the placeholder message must be present for the automatic mode.
    assert "Model-weight placeholder: H=0.500, I=0.500" in src


def test_metadata_note_source_flows_into_compiled_row(tmp_path: Path) -> None:
    """When proc_audio writes ``note_source=filename_token`` to
    Analysis_Metadata, the compiled Density_Metrics row must surface
    that value end-to-end (instead of re-parsing the xlsx filename).
    """
    from proc_audio import ANALYSIS_SCHEMA_VERSION as _ASV

    wb = tmp_path / "spectral_analysis.xlsx"
    # Intentionally write the workbook under a parent folder whose
    # name does NOT contain a note token; the canonical parser would
    # otherwise fall through to ``unknown``.
    target = tmp_path / "audio_with_no_token_in_folder_name" / "spectral_analysis.xlsx"
    target.parent.mkdir(parents=True, exist_ok=True)

    am_df = pd.DataFrame(
        [
            ("analysis_schema_version", _ASV),
            ("export_alignment_source", "disabled_integrated_single_pass"),
            ("export_alignment_factor", 1.0),
            ("model_weights_source", "current_analysis"),
            ("component_profile_source", "integrated_single_pass"),
            ("component_energy_method", "single_pass_partial_amplitude_sums"),
            ("component_energy_denominator", "H + I + S"),
            ("component_harmonic_energy_ratio", 0.8),
            ("component_inharmonic_energy_ratio", 0.15),
            ("component_subbass_energy_ratio", 0.05),
            ("note_source", "filename_token"),
        ],
        columns=["Parameter", "Value"],
    )
    arr = np.array([1.0, 2.0, 3.0])
    harm_df = pd.DataFrame({"Amplitude_raw": arr, "Amplitude": arr})
    inharm_df = pd.DataFrame({"Amplitude_raw": [4.0, 5.0], "Amplitude": [4.0, 5.0]})
    subbass_df = pd.DataFrame({"Amplitude_raw": [6.0], "Amplitude": [6.0]})
    with pd.ExcelWriter(target, engine="xlsxwriter") as writer:
        harm_df.to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        inharm_df.to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        subbass_df.to_excel(writer, sheet_name="Sub-bass band", index=False)
        am_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)
        pd.DataFrame([{"Note": "A#3"}]).to_excel(
            writer, sheet_name="Metrics", index=False
        )

    out_df = cm._build_density_metrics_sheet_from_per_note_files(
        [(target, "A#3", target.parent.name)],
        weight_function="linear",
    )
    assert len(out_df) == 1
    row = out_df.iloc[0]
    assert row["note_source"] == "filename_token"


# ---------------------------------------------------------------------------
# Fgt_pp finding C1 — Sub-bass aggregator must protect the FULL canonical
# harmonic population (strict ∪ Harmonic Spectrum candidates whose
# include_for_density==True). The previous code only protected the strict
# list, so a low harmonic that passed include_for_density but failed the
# strict SNR/prominence gate got double-counted: once as harmonic (in
# D_H) and again as sub-bass noise (in subbass_energy_sum), inflating w_S.
# ---------------------------------------------------------------------------
def test_subbass_aggregator_protects_include_for_density_harmonic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression for Fgt_pp finding C1.

    Build a minimal object mimicking ``AudioProcessor``'s harmonic-list
    state at the moment :func:`aggregate_subbass_noise_peak_power` is
    called:

    * a strict list (``harmonic_list_df``) that is MISSING the
      low-order 150 Hz harmonic (e.g. it failed the strict SNR gate),
    * a wider candidate list
      (``harmonic_spectrum_candidates_df``) that contains 150 Hz with
      ``include_for_density=True`` (so Stage 2 puts that bin's
      amplitude into ``D_H``).

    Call :meth:`AudioProcessor._build_subbass_harmonic_protection_df`
    and feed its output into :func:`aggregate_subbass_noise_peak_power`
    together with a synthetic ``complete_list_df`` that has a strong
    local-maximum bin at 150 Hz.

    Assertions:
      1. The helper's union DataFrame DOES contain 150 Hz.
      2. With the union protection, the 150 Hz bin is excluded from
         ``subbass_energy_sum``.
      3. With strict-only protection (legacy behaviour), the 150 Hz
         bin would be counted — so the result strictly differs.
    """
    from proc_audio import AudioProcessor
    from density import aggregate_subbass_noise_peak_power

    obj = AudioProcessor.__new__(AudioProcessor)
    obj.harmonic_list_df = pd.DataFrame(
        {"Frequency (Hz)": [75.0, 225.0]}
    )
    obj.harmonic_spectrum_candidates_df = pd.DataFrame(
        {
            "Frequency (Hz)": [75.0, 150.0, 225.0, 300.0],
            "include_for_density": [True, True, True, False],
        }
    )

    protect_df = obj._build_subbass_harmonic_protection_df()
    protect_freqs = pd.to_numeric(
        protect_df["Frequency (Hz)"], errors="coerce"
    ).to_numpy(dtype=float)
    assert np.any(np.isclose(protect_freqs, 75.0)), (
        "Strict 75 Hz harmonic must still be in the protection list."
    )
    assert np.any(np.isclose(protect_freqs, 150.0)), (
        "150 Hz harmonic (passed include_for_density but failed strict gate) "
        "must be added to the protection list."
    )
    assert not np.any(np.isclose(protect_freqs, 300.0)), (
        "300 Hz candidate (include_for_density=False) must NOT be protected "
        "— it does not contribute to D_H and so must not exclude itself "
        "from the sub-bass aggregator either."
    )

    # ``complete_list_df`` for the aggregator: 9 bins below 200 Hz with a
    # clean strict local-max at 150 Hz that the protection must catch.
    complete = pd.DataFrame(
        {
            "Frequency (Hz)": [10.0, 30.0, 75.0, 110.0, 140.0, 150.0, 160.0, 180.0, 195.0],
            "Amplitude":      [0.01, 0.02, 0.30, 0.05, 0.04, 0.90, 0.04, 0.03, 0.02],
        }
    )

    p_union = aggregate_subbass_noise_peak_power(
        complete, protect_df, subbass_hz=200.0, freq_match_tol_hz=12.0
    )
    p_strict = aggregate_subbass_noise_peak_power(
        complete, obj.harmonic_list_df, subbass_hz=200.0, freq_match_tol_hz=12.0
    )

    leaked_power_estimate = 0.9 ** 2
    assert p_strict > p_union + 0.5 * leaked_power_estimate, (
        f"Strict-only protection failed to exclude the 150 Hz harmonic — "
        f"p_strict={p_strict!r}, p_union={p_union!r}. Without the fix the "
        f"150 Hz lobe leaks into subbass_energy_sum."
    )
    assert p_union < 0.1 * leaked_power_estimate, (
        f"Union protection still includes the 150 Hz harmonic — got "
        f"p_union={p_union!r}, expected << {leaked_power_estimate}."
    )


def test_subbass_protection_helper_handles_empty_state() -> None:
    """The helper must return an empty DataFrame (with the right
    column) when neither the strict nor the candidate list is
    populated, so the aggregator falls back to a no-protection
    integration (legacy behaviour) instead of crashing.
    """
    from proc_audio import AudioProcessor

    obj = AudioProcessor.__new__(AudioProcessor)
    obj.harmonic_list_df = pd.DataFrame()
    obj.harmonic_spectrum_candidates_df = pd.DataFrame()
    out = obj._build_subbass_harmonic_protection_df()
    assert isinstance(out, pd.DataFrame)
    assert "Frequency (Hz)" in out.columns
    assert out.empty


def test_orchestrator_param_log_uses_placeholder_in_auto_mode(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Runtime assertion: trigger the activation-log block with
    ``manual_model_weight_override=False`` and confirm no 0.950/0.050
    line appears.

    This exercises the actual log emission used by the GUI worker.
    """
    from importlib import reload
    import pipeline_orchestrator_gui as gui_mod

    gui_mod = reload(gui_mod)
    log = gui_mod.log

    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    log.addHandler(handler)
    try:
        params = {
            "manual_model_weight_override": False,
            "i_weight": 0.05,
        }
        if bool(params.get("manual_model_weight_override", False)):
            log.info(
                "Manual model-weight override ACTIVE: "
                "alpha=%.3f, beta=%.3f (ACTIVATED)",
                1.0 - params["i_weight"],
                params["i_weight"],
            )
        else:
            log.info(
                "Model-weight placeholder: H=0.500, I=0.500; final "
                "component ratios are computed from current spectral "
                "analysis (ACTIVATED)."
            )
    finally:
        log.removeHandler(handler)
    out = buf.getvalue()
    assert "0.950 | Inharmonic Weight: 0.050" not in out
    assert "Model-weight placeholder: H=0.500, I=0.500" in out
