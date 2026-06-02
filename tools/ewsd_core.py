#!/usr/bin/env python3
"""
ewsd_core.py — Effective Weighted Spectral Density (EWSD-R v18 core).

Embedded computation module for SoundSpectrAnalyse Stage 3 research export.
Source logic: apply_effective_weighted_spectral_density_gui_v18.py (EWSD-R v18).
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import queue
import re
import tempfile
import threading
import traceback
import zipfile
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from tools.ewsd_pure import (
    ACOUSTIC_BALANCE_ALPHA_DEFAULT as _PURE_ACOUSTIC_BALANCE_ALPHA_DEFAULT,
    CompartmentInputs,
    EWSD_PURE_REVISION,
    canonical_weight_key,
    compute_compartment_metrics,
    d24_strength_mask,
    original_elementwise_weight,
    original_sum_metric,
    spectral_neff_from_linear_amplitudes,
)


INDIVIDUAL_SHEETS = {"Harmonic Spectrum", "Inharmonic Spectrum"}
COMPILED_SHEETS = {
    "Canonical_Metrics",
    "Density_Metrics",
    "Spectral_Density_Metrics",
    "Primary_Statistics_Eligible",
    "Canonical_Primary_Filtered",
    "Diagnostic_Metrics",
}

# Existing EWSD outputs can be upgraded with the v16 acoustic-alignment layer.
# This is not a recomputation from component spectra; it is a safe post-processing
# upgrade when only ewsd_ratio_respecting_results.xlsx is present in an analysis folder.
EWSD_RESULT_SHEETS = {"EWSD_All_Columns", "EWSD_Main"}

WEIGHT_ALGORITHMS = [
    "auto_from_excel",
    "linear",
    "sum",
    "sqrt",
    "squared",
    "cbrt",
    "cubic",
    "logarithmic",
    "log",
    "exponential",
    "exp",
    "inverse log",
    "d3",
    "d10",
    "d17",
    "d24",
    "d2",
    "d8",
]

RATIO_SOURCES = [
    "auto_excel_required",
    "density_weight",
    "pure_observation",
    "component_energy_ratio",
    "core_energy_ratio",
    "energy_ratio",
    "manual_override",
]

RATIO_COLUMN_SETS: dict[str, tuple[str, str, str]] = {
    # Best match for the existing research exports when present.
    "density_weight": (
        "harmonic_density_weight",
        "inharmonic_density_weight",
        "subbass_density_weight",
    ),
    # Present in individual Metrics sheet and usually expresses the per-note
    # observational H/I/S balance uncovered by the analysis.
    "pure_observation": (
        "pure_observation_w_h",
        "pure_observation_w_i",
        "pure_observation_w_s",
    ),
    # Direct component-energy ratios; useful fallback and explicit H/I/S ratio.
    "component_energy_ratio": (
        "component_harmonic_energy_ratio",
        "component_inharmonic_energy_ratio",
        "component_subbass_energy_ratio",
    ),
    "core_energy_ratio": (
        "core_harmonic_energy_ratio",
        "core_residual_energy_ratio",
        "core_subbass_energy_ratio",
    ),
    "energy_ratio": (
        "harmonic_energy_ratio",
        "inharmonic_energy_ratio",
        "subbass_energy_ratio",
    ),
}

AUTO_RATIO_PRIORITY = [
    "density_weight",
    "pure_observation",
    "component_energy_ratio",
    "core_energy_ratio",
    "energy_ratio",
]

SCRIPT_VERSION = "EWSD-R v18.1"
ACOUSTIC_BALANCE_ALPHA_DEFAULT = _PURE_ACOUSTIC_BALANCE_ALPHA_DEFAULT
BIBLIOGRAPHIC_ALIGNMENT_FILL_EXPONENT_DEFAULT = 1.0
PRIMARY_OUTPUT_FILENAME = "ewsd_ratio_respecting_results.xlsx"
THESIS_SAFE_WEIGHT_FUNCTIONS = {"log", "d3", "d10", "d17", "d24", "sqrt", "cbrt", "linear"}
AGGRESSIVE_WEIGHT_FUNCTIONS = {"exponential", "cubic", "squared", "inverse log"}


@dataclass
class HISWeights:
    harmonic: float = math.nan
    nonharmonic_residual: float = math.nan
    noise_subbass: float = math.nan
    source: str = "missing"
    columns: str = ""
    input_sum: float = math.nan
    normalised: bool = False
    warning: str = ""

    def is_valid(self) -> bool:
        vals = [self.harmonic, self.nonharmonic_residual, self.noise_subbass]
        return all(np.isfinite(v) and v >= 0.0 for v in vals) and sum(vals) > 0.0


@dataclass
class ComponentSet:
    source_file: str
    note: str
    components: pd.DataFrame
    weight_function: str
    basis: str
    mode: str
    his_weights: HISWeights = field(default_factory=HISWeights)
    warning: str = ""
    weight_function_source: str = ""
    source_sha256: str = ""
    n_components_raw_harmonic: int = 0
    n_components_raw_nonharmonic_residual: int = 0
    n_components_raw_noise_subbass: int = 0



def safe_numeric_scalar(value: Any) -> float:
    try:
        out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(out) if np.isfinite(out) else math.nan
    except Exception:
        return math.nan


def safe_float_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def list_excel_sheets(path: Path) -> list[str]:
    try:
        return list(pd.ExcelFile(path).sheet_names)
    except Exception:
        return []


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable checksum for provenance. Empty string on failure."""
    try:
        h = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def normalise_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def first_existing_column(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    # exact first, then case-insensitive.
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {str(c).lower(): c for c in df.columns}
    for c in candidates:
        got = lower_map.get(c.lower())
        if got is not None:
            return got
    return None


NOTE_CHROMATIC_ORDER = {
    "C": 0,
    "C#": 1, "DB": 1,
    "D": 2,
    "D#": 3, "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "GB": 6,
    "G": 7,
    "G#": 8, "AB": 8,
    "A": 9,
    "A#": 10, "BB": 10,
    "B": 11,
}

NOTE_CANONICAL_SHARP = {
    0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F",
    6: "F#", 7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B",
}


def infer_note_from_filename(path: Path) -> str:
    name = path.stem
    m = re.search(r"([A-Ga-g](?:#|b|♯|♭)?)(-?\d{1,2})", name)
    if m:
        return normalise_note_name(m.group(1) + m.group(2))
    return name


def normalise_note_name(note: Any) -> str:
    """Return a compact note spelling such as C#4 or Bb2, preserving flat spelling when supplied."""
    s = str(note).strip() if note is not None else ""
    if not s or s.lower() == "nan":
        return ""
    s = s.replace("♯", "#").replace("♭", "b")
    m = re.search(r"([A-Ga-g])\s*([#b]?)\s*(-?\d{1,2})", s)
    if not m:
        return s
    letter = m.group(1).upper()
    accidental = m.group(2)
    octave = m.group(3)
    return f"{letter}{accidental}{octave}"


def parse_note_sort_fields(note: Any) -> dict[str, Any]:
    """Parse note names and generate octave/chromatic sort fields.

    Sorting convention: C, C#/Db, D, D#/Eb, E, F, F#/Gb, G, G#/Ab, A, A#/Bb, B inside each octave;
    octave ascending. Enharmonic spellings receive the same MIDI-like index.
    """
    raw = normalise_note_name(note)
    m = re.search(r"^([A-G])([#b]?)(-?\d{1,2})$", raw)
    if not m:
        return {
            "Note_normalised": raw,
            "Note_pitch_class": "",
            "Note_octave": np.nan,
            "Note_chromatic_index": np.nan,
            "Note_midi_sort": np.nan,
            "Note_sort_warning": "unparsed_note_for_chromatic_sort" if raw else "missing_note",
        }
    pc_text = (m.group(1) + m.group(2)).upper()
    octave = int(m.group(3))
    pc = NOTE_CHROMATIC_ORDER.get(pc_text)
    if pc is None:
        return {
            "Note_normalised": raw,
            "Note_pitch_class": "",
            "Note_octave": np.nan,
            "Note_chromatic_index": np.nan,
            "Note_midi_sort": np.nan,
            "Note_sort_warning": "unknown_pitch_class_for_chromatic_sort",
        }
    return {
        "Note_normalised": raw,
        "Note_pitch_class": NOTE_CANONICAL_SHARP[pc],
        "Note_octave": octave,
        "Note_chromatic_index": pc,
        "Note_midi_sort": (octave + 1) * 12 + pc,
        "Note_sort_warning": "",
    }


def add_chromatic_sort_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Note" not in out.columns:
        out["Note"] = ""
    parsed = pd.DataFrame([parse_note_sort_fields(v) for v in out["Note"].tolist()])
    for col in parsed.columns:
        out[col] = parsed[col].values
    return out


def sort_chromatically_by_octave(df: pd.DataFrame) -> pd.DataFrame:
    """Sort output rows in strict musical order.

    v6 deliberately sorts note-first, not instrument-first and not alphabetically.
    Required order: octave ascending, and within each octave
    C, C#/Db, D, D#/Eb, E, F, F#/Gb, G, G#/Ab, A, A#/Bb, B.

    This avoids the Excel-looking alphabetical order A#, A, B, C#, C... that can occur
    if any text column is allowed to dominate the sort key.
    """
    out = add_chromatic_sort_columns(df)

    # Explicit audit column: 36=C2, 37=C#2, etc.  Unparsed notes remain last.
    out["Note_order"] = out["Note_midi_sort"]

    # The note sort key must be first.  Other fields only break ties among repeated notes.
    sort_cols = []
    for col in [
        "Note_midi_sort",
        "Note_normalised",
        "Instrument",
        "mode",
        "source_file",
    ]:
        if col in out.columns:
            sort_cols.append(col)

    if sort_cols:
        out = out.sort_values(
            sort_cols,
            ascending=[True] * len(sort_cols),
            na_position="last",
            kind="mergesort",
        ).reset_index(drop=True)
    return out


def read_first_row_as_dict(path: Path, sheet_name: str) -> dict[str, Any]:
    try:
        df = pd.read_excel(path, sheet_name=sheet_name, nrows=1)
    except Exception:
        return {}
    if df.empty:
        return {}
    return {str(c): df.iloc[0][c] for c in df.columns}


def choose_excel_weight_function(meta: dict[str, Any], fallback: str, use_excel_weight_function: bool) -> tuple[str, str]:
    if use_excel_weight_function:
        for col in ["weight_function", "density_weight_function", "Weight Function", "weighting_function"]:
            val = meta.get(col)
            if isinstance(val, str) and val.strip():
                return val.strip(), col
    if fallback == "auto_from_excel":
        return "log", "fallback_missing_excel_weight_function_log"
    return fallback, "manual_or_cli"


def _extract_ratio_set(meta: dict[str, Any], source: str) -> Optional[HISWeights]:
    cols = RATIO_COLUMN_SETS.get(source)
    if not cols:
        return None
    h_col, i_col, s_col = cols
    # exact and case-insensitive lookup in a dict.
    lower = {str(k).lower(): k for k in meta.keys()}
    actual_cols = []
    vals = []
    for wanted in [h_col, i_col, s_col]:
        actual = wanted if wanted in meta else lower.get(wanted.lower())
        if actual is None:
            return None
        actual_cols.append(str(actual))
        vals.append(safe_numeric_scalar(meta.get(actual)))
    if not all(np.isfinite(v) and v >= 0.0 for v in vals):
        return None
    s = float(sum(vals))
    if s <= 0.0:
        return None
    normalised = False
    warning = ""
    # Ratios should normally sum to 1. If they do not, preserving relative ratios
    # requires normalisation; the original sum is exported.
    if not (0.999 <= s <= 1.001):
        vals = [float(v) / s for v in vals]
        normalised = True
        warning = f"ratio columns found but sum={s:.12g}; normalised preserving H/I/S proportions"
    return HISWeights(
        harmonic=float(vals[0]),
        nonharmonic_residual=float(vals[1]),
        noise_subbass=float(vals[2]),
        source=source,
        columns="|".join(actual_cols),
        input_sum=s,
        normalised=normalised,
        warning=warning,
    )


def extract_his_weights(
    meta: dict[str, Any],
    ratio_source: str,
    manual_h: Optional[float] = None,
    manual_i: Optional[float] = None,
    manual_s: Optional[float] = None,
) -> HISWeights:
    ratio_source = (ratio_source or "auto_excel_required").strip().lower()
    if ratio_source == "manual_override":
        vals = [manual_h, manual_i, manual_s]
        if not all(v is not None and np.isfinite(float(v)) and float(v) >= 0.0 for v in vals):
            return HISWeights(source="manual_override_missing", warning="manual override requested but H/I/S weights are invalid")
        s = float(sum(float(v) for v in vals))
        if s <= 0.0:
            return HISWeights(source="manual_override_missing", warning="manual override requested but H/I/S sum is zero")
        return HISWeights(
            harmonic=float(manual_h) / s,
            nonharmonic_residual=float(manual_i) / s,
            noise_subbass=float(manual_s) / s,
            source="manual_override_normalised",
            columns="manual_h|manual_i|manual_s",
            input_sum=s,
            normalised=True,
            warning="manual override used; not analysis-derived",
        )

    if ratio_source == "auto_excel_required":
        for src in AUTO_RATIO_PRIORITY:
            w = _extract_ratio_set(meta, src)
            if w is not None and w.is_valid():
                return w
        return HISWeights(source="missing", warning="no complete per-note H/I/noise ratio set found in Excel; row not computed with defaults")

    w = _extract_ratio_set(meta, ratio_source)
    if w is not None and w.is_valid():
        return w
    return HISWeights(source=f"missing_{ratio_source}", warning=f"requested ratio source not found or invalid: {ratio_source}")


# -----------------------------------------------------------------------------
# Original SoundSpectrAnalyse weighting/sum family (delegates to ewsd_pure)
# -----------------------------------------------------------------------------

_spectral_neff_from_linear_amplitudes = spectral_neff_from_linear_amplitudes


# -----------------------------------------------------------------------------
# Reading component distributions
# -----------------------------------------------------------------------------


def choose_component_basis(df: pd.DataFrame, basis: str) -> pd.Series:
    basis = basis.lower().strip()
    if basis == "amplitude":
        col = first_existing_column(df, ["Amplitude_raw", "Amplitude", "Amplitude_linear", "Linear Amplitude", "amplitude_raw", "amplitude"])
        if col is not None:
            return safe_float_series(df[col])
        pcol = first_existing_column(df, ["Power_raw", "Power", "power_raw", "power"])
        if pcol is not None:
            return np.sqrt(safe_float_series(df[pcol]).clip(lower=0))
    elif basis == "power":
        col = first_existing_column(df, ["Power_raw", "Power", "power_raw", "power"])
        if col is not None:
            return safe_float_series(df[col])
        acol = first_existing_column(df, ["Amplitude_raw", "Amplitude", "Amplitude_linear", "Linear Amplitude", "amplitude_raw", "amplitude"])
        if acol is not None:
            return safe_float_series(df[acol]).clip(lower=0) ** 2
    raise ValueError("No usable amplitude/power column found in component table.")


def standardise_component_table(
    df: pd.DataFrame,
    component_type: str,
    basis: str,
    frequency_ceiling_hz: Optional[float],
    include_only_for_density: bool = False,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["component_type", "frequency_hz", "magnitude_db", "basis_value"])

    out = df.copy()
    fcol = first_existing_column(out, ["Frequency (Hz)", "Frequency", "frequency_hz", "freq_hz"])
    out["frequency_hz"] = safe_float_series(out[fcol]) if fcol is not None else np.nan

    mag_col = first_existing_column(out, ["Magnitude (dB)", "Magnitude", "magnitude_db", "mag_db"])
    out["magnitude_db"] = safe_float_series(out[mag_col]) if mag_col is not None else np.nan

    if include_only_for_density and "include_for_density" in out.columns:
        out = out[normalise_bool_series(out["include_for_density"])]

    if frequency_ceiling_hz is not None:
        out = out[(out["frequency_hz"].isna()) | (out["frequency_hz"] <= float(frequency_ceiling_hz))]

    if out.empty:
        return pd.DataFrame(columns=["component_type", "frequency_hz", "magnitude_db", "basis_value"])

    out["basis_value"] = choose_component_basis(out, basis)
    out["component_type"] = component_type
    out = out[["component_type", "frequency_hz", "magnitude_db", "basis_value"]]
    out = out[np.isfinite(out["basis_value"]) & (out["basis_value"] > 0)]
    return out.reset_index(drop=True)


def aggregate_subbass_rows(subbass: pd.DataFrame, basis: str) -> pd.DataFrame:
    if subbass is None or subbass.empty:
        return pd.DataFrame(columns=["component_type", "frequency_hz", "magnitude_db", "basis_value"])
    vals = safe_float_series(subbass["basis_value"]).clip(lower=0)
    if vals.dropna().empty:
        return pd.DataFrame(columns=["component_type", "frequency_hz", "magnitude_db", "basis_value"])
    if basis == "amplitude":
        energy_equiv = float(np.sum(vals.to_numpy(dtype=float) ** 2))
        basis_value = math.sqrt(max(energy_equiv, 0.0))
    else:
        basis_value = float(np.sum(vals.to_numpy(dtype=float)))
    freq = float(np.nanmean(subbass["frequency_hz"])) if "frequency_hz" in subbass else np.nan
    mag = float(np.nanmax(subbass["magnitude_db"])) if "magnitude_db" in subbass else np.nan
    return pd.DataFrame([
        {
            "component_type": "subbass_aggregated_band",
            "frequency_hz": freq,
            "magnitude_db": mag,
            "basis_value": basis_value,
        }
    ])


def read_individual_workbook(
    path: Path,
    requested_weight_function: str,
    use_excel_weight_function: bool,
    ratio_source: str,
    manual_h: Optional[float],
    manual_i: Optional[float],
    manual_s: Optional[float],
    basis: str,
    frequency_ceiling_hz: Optional[float],
    aggregate_subbass: bool,
) -> Optional[ComponentSet]:
    sheets = list_excel_sheets(path)
    if not INDIVIDUAL_SHEETS.issubset(set(sheets)):
        return None
    meta = read_first_row_as_dict(path, "Metrics") if "Metrics" in sheets else {}
    wf, wf_source = choose_excel_weight_function(meta, requested_weight_function, use_excel_weight_function)
    his = extract_his_weights(meta, ratio_source, manual_h, manual_i, manual_s)

    try:
        h = pd.read_excel(path, sheet_name="Harmonic Spectrum")
        i = pd.read_excel(path, sheet_name="Inharmonic Spectrum") if "Inharmonic Spectrum" in sheets else pd.DataFrame()
        s = pd.read_excel(path, sheet_name="Sub-bass band") if "Sub-bass band" in sheets else pd.DataFrame()
    except Exception as exc:
        return ComponentSet(
            str(path), infer_note_from_filename(path), pd.DataFrame(), wf, basis, "individual_exact", his,
            f"read_error: {exc}; weight_function_source={wf_source}",
            wf_source, file_sha256(path), 0, 0, 0
        )

    h2 = standardise_component_table(h, "harmonic", basis, frequency_ceiling_hz, include_only_for_density=True)
    i2 = standardise_component_table(i, "nonharmonic_residual", basis, frequency_ceiling_hz, include_only_for_density=False)
    s2 = standardise_component_table(s, "subbass_residual", basis, frequency_ceiling_hz, include_only_for_density=False)
    raw_h_count = int(len(h2))
    raw_i_count = int(len(i2))
    raw_s_count = int(len(s2))
    if aggregate_subbass:
        s2 = aggregate_subbass_rows(s2, basis)

    components = pd.concat([h2, i2, s2], ignore_index=True)

    note = infer_note_from_filename(path)
    if meta.get("Note") is not None and str(meta.get("Note")).strip():
        note = str(meta.get("Note")).strip()

    warning_parts = []
    if wf_source.startswith("fallback_missing"):
        warning_parts.append(wf_source)
    if his.warning:
        warning_parts.append(his.warning)
    return ComponentSet(
        str(path), note, components, wf, basis, "individual_exact", his, "; ".join(warning_parts),
        wf_source, file_sha256(path), raw_h_count, raw_i_count, raw_s_count
    )


def apply_relative_db_threshold(components: pd.DataFrame, threshold_db_relative: Optional[float]) -> pd.DataFrame:
    if components.empty or threshold_db_relative is None:
        return components
    if "magnitude_db" not in components.columns or components["magnitude_db"].dropna().empty:
        return components
    max_db = float(np.nanmax(components["magnitude_db"].to_numpy(dtype=float)))
    cutoff = max_db + float(threshold_db_relative)
    return components[(components["magnitude_db"].isna()) | (components["magnitude_db"] >= cutoff)].reset_index(drop=True)


def _component_family_mask(component_types: pd.Series, family: str) -> pd.Series:
    t = component_types.astype(str)
    if family == "harmonic":
        return t.eq("harmonic")
    if family == "nonharmonic_residual":
        return t.eq("nonharmonic_residual")
    if family == "noise_subbass":
        return t.str.startswith("subbass") | t.eq("noise") | t.eq("subnoise")
    return pd.Series(False, index=component_types.index)


def _d24_strength_mask(values: np.ndarray, freqs: Optional[np.ndarray]) -> np.ndarray:
    return d24_strength_mask(values, freqs)


def _compute_family_metrics(
    family_df: pd.DataFrame,
    family_name: str,
    cset: ComponentSet,
    analysis_ratio_weight: float,
    apply_anti_concentration: bool,
) -> dict[str, Any]:
    values = (
        family_df["basis_value"].to_numpy(dtype=float)
        if not family_df.empty and "basis_value" in family_df.columns
        else np.array([], dtype=float)
    )
    freqs = (
        family_df["frequency_hz"].to_numpy(dtype=float)
        if not family_df.empty and "frequency_hz" in family_df.columns
        else None
    )
    try:
        metrics = compute_compartment_metrics(
            CompartmentInputs(
                values=values,
                analysis_ratio=float(analysis_ratio_weight),
                frequencies_hz=freqs,
                weight_function=cset.weight_function,
                apply_anti_concentration=apply_anti_concentration,
            )
        )
    except Exception as exc:
        return {
            f"count_{family_name}": 0,
            f"original_sum_metric_{family_name}": 0.0,
            f"analysis_ratio_weight_{family_name}": float(analysis_ratio_weight)
            if np.isfinite(float(analysis_ratio_weight))
            else math.nan,
            f"ratio_weighted_metric_{family_name}": 0.0,
            f"weighted_mass_{family_name}": 0.0,
            f"effective_component_count_{family_name}": 0.0,
            f"concentration_penalty_{family_name}": 0.0,
            f"entropy_normalized_{family_name}": 0.0,
            f"ewsd_score_{family_name}": 0.0,
            f"warning_{family_name}": f"compartment_error:{exc}",
        }

    return {
        f"count_{family_name}": metrics.count,
        f"original_sum_metric_{family_name}": metrics.original_sum_metric,
        f"analysis_ratio_weight_{family_name}": metrics.analysis_ratio_weight,
        f"ratio_weighted_metric_{family_name}": metrics.ratio_weighted_metric,
        f"weighted_mass_{family_name}": metrics.weighted_mass,
        f"effective_component_count_{family_name}": metrics.effective_component_count,
        f"concentration_penalty_{family_name}": metrics.concentration_penalty,
        f"entropy_normalized_{family_name}": metrics.entropy_normalized,
        f"ewsd_score_{family_name}": metrics.ewsd_score,
    }


def _empty_row(cset: ComponentSet, reason: str, apply_anti_concentration: bool) -> dict[str, Any]:
    rows: dict[str, Any] = {
        "source_file": cset.source_file,
        "mode": cset.mode,
        "Note": cset.note,
        "weight_function": cset.weight_function,
        "weight_function_canonical": canonical_weight_key(cset.weight_function),
        "weight_function_source": cset.weight_function_source,
        "source_sha256": cset.source_sha256,
        "basis": cset.basis,
        "raw_component_count_harmonic_before_threshold": cset.n_components_raw_harmonic,
        "raw_component_count_nonharmonic_before_threshold": cset.n_components_raw_nonharmonic_residual,
        "raw_component_count_noise_subbass_before_threshold": cset.n_components_raw_noise_subbass,
        "his_ratio_source": cset.his_weights.source,
        "his_ratio_columns": cset.his_weights.columns,
        "his_ratio_input_sum": cset.his_weights.input_sum,
        "his_ratio_normalised": cset.his_weights.normalised,
        "analysis_ratio_weight_harmonic": cset.his_weights.harmonic,
        "analysis_ratio_weight_nonharmonic_residual": cset.his_weights.nonharmonic_residual,
        "analysis_ratio_weight_noise_subbass": cset.his_weights.noise_subbass,
        "component_count_salient": 0,
        "original_sum_metric": math.nan,
        "original_sum_metric_unweighted_HIS_total": math.nan,
        "ewsd_weighted_mass": math.nan,
        "ewsd_effective_component_count": math.nan,
        "ewsd_concentration_penalty": math.nan,
        "ewsd_entropy_normalized": math.nan,
        "ewsd_score": math.nan,
        "anti_concentration_applied": bool(apply_anti_concentration),
        "warning": reason,
    }
    for fam in ["harmonic", "nonharmonic_residual", "noise_subbass"]:
        rows.update({
            f"count_{fam}": 0,
            f"original_sum_metric_{fam}": 0.0,
            f"analysis_ratio_weight_{fam}": getattr(cset.his_weights, fam if fam != "noise_subbass" else "noise_subbass", math.nan),
            f"ratio_weighted_metric_{fam}": 0.0,
            f"weighted_mass_{fam}": 0.0,
            f"effective_component_count_{fam}": 0.0,
            f"concentration_penalty_{fam}": 0.0,
            f"entropy_normalized_{fam}": 0.0,
            f"ewsd_score_{fam}": 0.0,
            f"ratio_raw_metric_{fam}": math.nan,
            f"ratio_ratio_weighted_metric_{fam}": math.nan,
            f"ratio_ewsd_score_{fam}": math.nan,
            f"ratio_count_{fam}": math.nan,
        })
    return rows


def compute_ewsd(
    cset: ComponentSet,
    threshold_db_relative: Optional[float],
    apply_anti_concentration: bool,
) -> dict[str, Any]:
    if not cset.his_weights.is_valid():
        return _empty_row(cset, cset.his_weights.warning or "missing per-note H/I/noise ratios", apply_anti_concentration)

    components = apply_relative_db_threshold(cset.components.copy(), threshold_db_relative)
    if components.empty:
        return _empty_row(cset, cset.warning or "no_components", apply_anti_concentration)

    rows: dict[str, Any] = {
        "source_file": cset.source_file,
        "mode": cset.mode,
        "Note": cset.note,
        "weight_function": cset.weight_function,
        "weight_function_canonical": canonical_weight_key(cset.weight_function),
        "weight_function_source": cset.weight_function_source,
        "source_sha256": cset.source_sha256,
        "basis": cset.basis,
        "raw_component_count_harmonic_before_threshold": cset.n_components_raw_harmonic,
        "raw_component_count_nonharmonic_before_threshold": cset.n_components_raw_nonharmonic_residual,
        "raw_component_count_noise_subbass_before_threshold": cset.n_components_raw_noise_subbass,
        "his_ratio_source": cset.his_weights.source,
        "his_ratio_columns": cset.his_weights.columns,
        "his_ratio_input_sum": cset.his_weights.input_sum,
        "his_ratio_normalised": cset.his_weights.normalised,
        "analysis_ratio_weight_harmonic": cset.his_weights.harmonic,
        "analysis_ratio_weight_nonharmonic_residual": cset.his_weights.nonharmonic_residual,
        "analysis_ratio_weight_noise_subbass": cset.his_weights.noise_subbass,
        "anti_concentration_applied": bool(apply_anti_concentration),
        "warning": "; ".join(x for x in [cset.warning, cset.his_weights.warning] if x),
    }

    if canonical_weight_key(cset.weight_function) in AGGRESSIVE_WEIGHT_FUNCTIONS:
        existing_warning = str(rows.get("warning", "")).strip()
        aggressive_warning = f"aggressive_weight_function_not_recommended_for_primary_density:{canonical_weight_key(cset.weight_function)}"
        rows["warning"] = "; ".join(x for x in [existing_warning, aggressive_warning] if x)

    family_defs = [
        ("harmonic", cset.his_weights.harmonic),
        ("nonharmonic_residual", cset.his_weights.nonharmonic_residual),
        ("noise_subbass", cset.his_weights.noise_subbass),
    ]

    for family_name, family_weight in family_defs:
        fam_mask = _component_family_mask(components["component_type"], family_name)
        family_df = components.loc[fam_mask].copy().reset_index(drop=True)
        rows.update(
            _compute_family_metrics(
                family_df=family_df,
                family_name=family_name,
                cset=cset,
                analysis_ratio_weight=family_weight,
                apply_anti_concentration=apply_anti_concentration,
            )
        )

    original_total_raw = float(sum(rows.get(f"original_sum_metric_{f}", 0.0) for f, _ in family_defs))
    original_total_weighted = float(sum(rows.get(f"ratio_weighted_metric_{f}", 0.0) for f, _ in family_defs))
    score_total = float(sum(rows.get(f"ewsd_score_{f}", 0.0) for f, _ in family_defs))
    mass_total = float(sum(rows.get(f"weighted_mass_{f}", 0.0) for f, _ in family_defs))
    count_total = int(sum(int(rows.get(f"count_{f}", 0) or 0) for f, _ in family_defs))
    neff_sum_by_class = float(sum(rows.get(f"effective_component_count_{f}", 0.0) for f, _ in family_defs))
    entropy_vals = [float(rows.get(f"entropy_normalized_{f}", 0.0)) for f, _ in family_defs if rows.get(f"count_{f}", 0)]
    entropy_mean_by_class = float(np.mean(entropy_vals)) if entropy_vals else 0.0
    global_penalty = float(score_total / original_total_weighted) if original_total_weighted > 0 else 0.0

    rows.update({
        "component_count_salient": count_total,
        "original_sum_metric": original_total_weighted,
        "original_sum_metric_unweighted_HIS_total": original_total_raw,
        "ewsd_weighted_mass": mass_total,
        "ewsd_effective_component_count": neff_sum_by_class,
        "ewsd_effective_component_count_sum_by_class": neff_sum_by_class,
        "ewsd_concentration_penalty": global_penalty,
        "ewsd_entropy_normalized": entropy_mean_by_class,
        "ewsd_score": score_total,
        "ewsd_HIS_compartment_policy": "per-note Excel H/I/noise ratios; separate H/I/noise compartments; no default weights; no mixed global p_i distribution",
    })

    for f, _ in family_defs:
        raw = float(rows.get(f"original_sum_metric_{f}", 0.0))
        wraw = float(rows.get(f"ratio_weighted_metric_{f}", 0.0))
        sc = float(rows.get(f"ewsd_score_{f}", 0.0))
        cnt = float(rows.get(f"count_{f}", 0.0))
        rows[f"ratio_raw_metric_{f}"] = float(raw / original_total_raw) if original_total_raw > 0 else 0.0
        rows[f"ratio_ratio_weighted_metric_{f}"] = float(wraw / original_total_weighted) if original_total_weighted > 0 else 0.0
        rows[f"ratio_ewsd_score_{f}"] = float(sc / score_total) if score_total > 0 else 0.0
        rows[f"ratio_count_{f}"] = float(cnt / count_total) if count_total > 0 else 0.0

    return rows

def add_quality_columns(result: pd.DataFrame) -> pd.DataFrame:
    """Add row-level thesis safety gates and reliability diagnostics.

    These columns do not change EWSD itself. They make the output safer to use:
    only rows marked primary_analysis_eligible should enter final thesis statistics.
    """
    out = result.copy()
    n = len(out)
    if n == 0:
        out["row_quality_score_0_100"] = []
        out["primary_analysis_eligible"] = []
        out["row_quality_grade"] = []
        return out

    def col(name: str, default: Any = np.nan) -> pd.Series:
        if name in out.columns:
            return out[name]
        return pd.Series([default] * n, index=out.index)

    mode = col("mode", "").astype(str)
    warning = col("warning", "").fillna("").astype(str).str.strip()
    ratio_sum = (
        pd.to_numeric(col("analysis_ratio_weight_harmonic"), errors="coerce")
        + pd.to_numeric(col("analysis_ratio_weight_nonharmonic_residual"), errors="coerce")
        + pd.to_numeric(col("analysis_ratio_weight_noise_subbass"), errors="coerce")
    )
    score = pd.to_numeric(col("ewsd_score"), errors="coerce")
    count = pd.to_numeric(col("component_count_salient"), errors="coerce")
    note_sort_warning = col("Note_sort_warning", "").fillna("").astype(str).str.strip()
    weight_key = col("weight_function_canonical", "").astype(str).str.lower().str.strip()

    exact = mode.eq("individual_exact")
    ratios_ok = np.isfinite(ratio_sum) & (ratio_sum.between(0.999, 1.001))
    score_ok = np.isfinite(score) & (score >= 0)
    count_ok = np.isfinite(count) & (count > 0)
    no_warning = warning.eq("") | warning.eq("nan")
    note_ok = note_sort_warning.eq("") | note_sort_warning.eq("nan")
    weight_safe = weight_key.isin(THESIS_SAFE_WEIGHT_FUNCTIONS)

    # Penalised additive score. It is intentionally conservative: proxy rows and
    # aggressive/explosive algorithms lose substantial reliability by construction.
    quality = pd.Series(100.0, index=out.index)
    quality -= np.where(exact, 0.0, 35.0)
    quality -= np.where(ratios_ok, 0.0, 25.0)
    quality -= np.where(score_ok, 0.0, 20.0)
    quality -= np.where(count_ok, 0.0, 15.0)
    quality -= np.where(no_warning, 0.0, 10.0)
    quality -= np.where(note_ok, 0.0, 5.0)
    quality -= np.where(weight_safe, 0.0, 15.0)
    quality = quality.clip(lower=0.0, upper=100.0)

    eligible = exact & ratios_ok & score_ok & count_ok & no_warning & note_ok & weight_safe
    out["HIS_ratio_sum_check"] = ratio_sum
    out["row_quality_score_0_100"] = quality.round(2)
    out["primary_analysis_eligible"] = eligible.astype(bool)
    out["row_quality_grade"] = pd.cut(
        quality,
        bins=[-0.1, 59.999, 74.999, 84.999, 92.999, 100.1],
        labels=["reject", "diagnostic", "usable_with_caution", "good", "strong"],
    ).astype(str)
    out["quality_gate_rule"] = (
        "primary=True requires individual_exact + H/I/S sum≈1 + finite EWSD + positive component count "
        "+ no warning + parsed note + non-aggressive weight function"
    )
    return out


def _safe_power01(values: pd.Series, exponent: float) -> pd.Series:
    """Raise a [0, 1] penalty to a finite exponent with safe clipping."""
    x = pd.to_numeric(values, errors="coerce").clip(lower=0.0, upper=1.0)
    try:
        a = float(exponent)
    except Exception:
        a = ACOUSTIC_BALANCE_ALPHA_DEFAULT
    if not np.isfinite(a) or a < 0.0:
        a = ACOUSTIC_BALANCE_ALPHA_DEFAULT
    return np.power(x, a)


def add_acoustic_alignment_columns(
    result: pd.DataFrame,
    frequency_ceiling_hz: Optional[float],
    acoustic_balance_alpha: float = ACOUSTIC_BALANCE_ALPHA_DEFAULT,
    bibliographic_fill_exponent: float = BIBLIOGRAPHIC_ALIGNMENT_FILL_EXPONENT_DEFAULT,
) -> pd.DataFrame:
    """Add comparative acoustic-alignment columns without changing EWSD_score_total.

    Rationale: strict EWSD uses penalty=(N_eff/N)^1.  The previous review showed that
    this is useful as an anti-concentration metric, but it can over-penalise spectra
    where a real instrument naturally concentrates energy in a smaller set of strong
    partials.  The acoustic-balanced companion score keeps the same H/I/S ratios and
    original sum algorithm, but uses penalty^alpha, default alpha=0.5.  This is
    deliberately reported as a companion metric; it does not overwrite ewsd_score.
    """
    out = sort_chromatically_by_octave(result.copy()) if len(result) else result.copy()
    try:
        alpha = float(acoustic_balance_alpha)
    except Exception:
        alpha = ACOUSTIC_BALANCE_ALPHA_DEFAULT
    if not np.isfinite(alpha) or alpha < 0.0:
        alpha = ACOUSTIC_BALANCE_ALPHA_DEFAULT

    try:
        fill_beta = float(bibliographic_fill_exponent)
    except Exception:
        fill_beta = BIBLIOGRAPHIC_ALIGNMENT_FILL_EXPONENT_DEFAULT
    if not np.isfinite(fill_beta) or fill_beta < 0.0:
        fill_beta = BIBLIOGRAPHIC_ALIGNMENT_FILL_EXPONENT_DEFAULT

    # Estimate fundamental from the parsed chromatic/MIDI-like note index.
    midi = pd.to_numeric(out.get("Note_midi_sort"), errors="coerce")
    f0 = 440.0 * np.power(2.0, (midi - 69.0) / 12.0)
    f0 = pd.Series(f0, index=out.index).where(np.isfinite(f0) & (f0 > 0.0), np.nan)
    out["ewsd_estimated_f0_hz"] = f0

    if frequency_ceiling_hz is not None and np.isfinite(float(frequency_ceiling_hz)) and float(frequency_ceiling_hz) > 0:
        capacity = np.floor(float(frequency_ceiling_hz) / f0)
        capacity = pd.Series(capacity, index=out.index).where(np.isfinite(capacity) & (capacity > 0.0), np.nan)
    else:
        capacity = pd.Series(np.nan, index=out.index)
    out["ewsd_capacity_harmonic_slots_to_ceiling"] = capacity

    families = ["harmonic", "nonharmonic_residual", "noise_subbass"]
    balanced_cols = []
    for fam in families:
        mass_col = f"ratio_weighted_metric_{fam}"
        pen_col = f"concentration_penalty_{fam}"
        if mass_col not in out.columns:
            out[mass_col] = np.nan
        if pen_col not in out.columns:
            out[pen_col] = np.nan
        mass = pd.to_numeric(out[mass_col], errors="coerce")
        pen_a = _safe_power01(out[pen_col], alpha)
        bcol = f"ewsd_score_acoustic_balanced_{fam}"
        out[bcol] = (mass * pen_a).where(np.isfinite(mass), np.nan)
        balanced_cols.append(bcol)

    out["ewsd_acoustic_balance_alpha"] = alpha
    out["ewsd_score_strict_original"] = pd.to_numeric(out.get("ewsd_score"), errors="coerce")
    out["ewsd_score_acoustic_balanced"] = out[balanced_cols].sum(axis=1, min_count=1)

    # Capacity-normalized companion: useful to separate timbral density from simple register capacity.
    cap_denom = np.log1p(pd.to_numeric(out["ewsd_capacity_harmonic_slots_to_ceiling"], errors="coerce"))
    bal = pd.to_numeric(out["ewsd_score_acoustic_balanced"], errors="coerce")
    out["ewsd_score_acoustic_balanced_capacity_normalized"] = (bal / cap_denom).where(np.isfinite(cap_denom) & (cap_denom > 0.0), np.nan)

    neff = pd.to_numeric(out.get("ewsd_effective_component_count"), errors="coerce")
    raw_fill = (neff / capacity).where(np.isfinite(capacity) & (capacity > 0.0), np.nan)
    out["ewsd_effective_fill_ratio_raw_to_harmonic_slots"] = raw_fill
    out["ewsd_effective_fill_fraction_to_ceiling"] = raw_fill.clip(lower=0.0, upper=1.0)
    out["ewsd_fill_fraction_capped"] = raw_fill.where(raw_fill <= 1.0, 1.0)
    out["ewsd_fill_fraction_overflow_warning"] = raw_fill.where(raw_fill <= 1.0, np.nan).isna() & raw_fill.notna()

    # v18 correction: there is no separate "bibliographic-aligned score".
    # The previous v17 compatibility alias duplicated ewsd_score_acoustic_balanced
    # and made the output misleading.  The bibliography-facing comparative metric
    # is explicitly ewsd_score_acoustic_balanced.  Spectral fill remains a
    # diagnostic/context variable only, not a multiplier and not a duplicate score.
    for deprecated_col in [
        "ewsd_score_bibliographic_aligned",
        "ewsd_score_bibliographic_aligned_log1p",
        "ewsd_score_bibliographic_aligned_norm_global_0_1",
        "ewsd_bibliographic_fill_exponent",
    ]:
        if deprecated_col in out.columns:
            out = out.drop(columns=[deprecated_col])

    out["ewsd_score_log1p"] = np.log1p(pd.to_numeric(out.get("ewsd_score"), errors="coerce").clip(lower=0.0))
    out["ewsd_score_acoustic_balanced_log1p"] = np.log1p(bal.clip(lower=0.0))

    for score_col, norm_col in [
        ("ewsd_score_acoustic_balanced", "ewsd_score_acoustic_balanced_norm_global_0_1"),
        ("ewsd_score_acoustic_balanced_capacity_normalized", "ewsd_score_acoustic_balanced_capacity_normalized_norm_global_0_1"),
    ]:
        vals = pd.to_numeric(out.get(score_col), errors="coerce")
        finite = vals[np.isfinite(vals)]
        if not finite.empty and float(finite.max()) > float(finite.min()):
            out[norm_col] = (vals - float(finite.min())) / (float(finite.max()) - float(finite.min()))
        else:
            out[norm_col] = np.nan

    out["acoustic_alignment_metric_policy"] = (
        "EWSD_score_total is unchanged strict EWSD. EWSD_score_acoustic_balanced uses the same H/I/S ratios "
        "and weighting algorithm but applies concentration_penalty^alpha, default alpha=0.5. "
        "Use EWSD_score_acoustic_balanced for cross-instrument bibliographic-distance diagnostics, "
        "and compare instruments only in pitch-matched windows. EWSD_effective_fill_fraction_to_ceiling is diagnostic only."
    )
    return out


