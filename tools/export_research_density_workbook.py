#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Post-process ``compiled_density_metrics.xlsx`` into a reduced research workbook.

Does **not** alter compilation logic or the source workbook; read-only input.

**Output Excel behaviour:** the writer uses **worksheet-level AutoFilter** and
**frozen header rows** on data sheets only. It does **not** create formal Excel
**Table** / ListObject parts (no ``xl/tables/table*.xml``), so Microsoft Excel
should open the file without repair prompts. **README** and **Dashboard** sheets
are not auto-filtered. DataFrame column names are sanitised before export (no
blank headers; duplicate names receive ``_2``, ``_3``, …).

**Metadata CLI:** optional ``--instrument``, ``--dynamic``, and
``--force-metadata`` (see ``README.md`` and the research workbook README sheet).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import CellIsRule, ColorScaleRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.utils.dataframe import dataframe_to_rows

from metadata_sanitizer import (
    apply_publication_clean_research_metadata_fields,
    drop_publication_noise_columns_from_dataframe,
    format_utc_publication_timestamp,
    publication_clean_drop_known_sparse_columns,
    publication_clean_export_enabled,
    publication_research_canonical_density_columns,
)

SCRIPT_NAME = "export_research_density_workbook.py"
SCRIPT_VERSION = "1.1.2"


@dataclass
class ResearchExportMetadata:
    """Optional CLI overrides for Instrument / Dynamic columns."""

    instrument: Optional[str] = None
    dynamic: Optional[str] = None
    force_metadata: bool = False


CHART_AMPLITUDE_NAME = "component_amplitude_mass_pie.png"
CHART_ENERGY_RATIO_NAME = "component_energy_ratio_pie.png"
CHART_ENERGY_LEGACY_NAME = "component_energy_pie.png"

# --- Sheet merge priority (first wins for non-null when coalescing) ---
MERGE_SHEETS: Tuple[str, ...] = (
    "Density_Metrics",
    "Canonical_Metrics",
    "Diagnostic_Metrics",
    "Validation_Metrics",
    "Debug_Counts",
    "Per_Note_Processing_Metadata",
)

# Canonical output column -> ordered list of source aliases (first match wins per sheet).
COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "harmonic_energy_ratio": ("harmonic_energy_ratio", "component_harmonic_energy_ratio"),
    "inharmonic_energy_ratio": ("inharmonic_energy_ratio", "component_inharmonic_energy_ratio"),
    "subbass_energy_ratio": ("subbass_energy_ratio", "component_subbass_energy_ratio"),
    "harmonic_energy_sum": ("harmonic_energy_sum", "component_harmonic_energy_sum"),
    "inharmonic_energy_sum": ("inharmonic_energy_sum", "component_inharmonic_energy_sum"),
    "subbass_energy_sum": ("subbass_energy_sum", "component_subbass_energy_sum"),
    "harmonic_density_sum": (
        "harmonic_density_sum",
        "Harmonic Partials sum",
        "linear_sum_amplitude_harmonic",
    ),
    "inharmonic_density_sum": (
        "inharmonic_density_sum",
        "Inharmonic Partials sum",
        "linear_sum_amplitude_inharmonic_partial",
    ),
    "subbass_density_sum": (
        "subbass_density_sum",
        "Sub-bass sum",
        "linear_sum_amplitude_subbass_band",
    ),
    "Source_File": (
        "Source_File",
        "source_file_name",
        "filename",
        "file_name",
        "source_file",
        "input_file",
    ),
    "Source_Workbook": ("Source_Workbook", "source_workbook", "workbook_path", "compiled_from"),
    "f0_final_hz": ("f0_final_hz", "f0_estimated", "f0_final"),
    "Instrument": ("Instrument", "instrument", "instrument_detected"),
    "Dynamic": ("Dynamic", "dynamic", "dynamic_detected"),
}

# Do not add dangerous reverse aliases (user forbade mapping Total sum -> density_metric_raw, etc.)

HEADER_FILL = PatternFill("solid", fgColor="1F4E79")
HEADER_FONT = Font(bold=True, color="FFFFFF", size=11)
TITLE_FILL = PatternFill("solid", fgColor="1F4E79")
TITLE_FONT = Font(bold=True, color="FFFFFF", size=16)
SUBHEADER_FONT = Font(bold=True, size=12)
KPI_LABEL_FONT = Font(bold=True, size=10)
THIN = Side(style="thin", color="B4B4B4")
KPI_BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
PASS_FILL = PatternFill("solid", fgColor="C6EFCE")
WARN_FILL = PatternFill("solid", fgColor="FCE4D6")
FAIL_FILL = PatternFill("solid", fgColor="FFC7CE")
NEUTRAL_KPI_FILL = PatternFill("solid", fgColor="E7E6E6")
MISSING_WARN_FILL = PatternFill("solid", fgColor="F8CBAD")
F0_FALSE_FILL = PatternFill("solid", fgColor="FFEB9C")


PITCH_CLASS_MAP: Dict[str, int] = {
    "C": 0,
    "C#": 1,
    "DB": 1,
    "D": 2,
    "D#": 3,
    "EB": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "GB": 6,
    "G": 7,
    "G#": 8,
    "AB": 8,
    "A": 9,
    "A#": 10,
    "BB": 10,
    "B": 11,
}


def parse_note(note_string: str | None) -> Tuple[str | None, int | None, int | None, str | None]:
    """
    Parse scientific pitch notation into (token, pitch_class, octave, letter).

    Returns (None, None, None, None) if parsing fails.
    """
    if note_string is None or (isinstance(note_string, float) and np.isnan(note_string)):
        return None, None, None, None
    s = str(note_string).strip()
    if not s:
        return None, None, None, None
    m = re.match(r"^([A-Ga-g])([#bB]?)(\d+)$", s)
    if not m:
        return None, None, None, None
    letter = m.group(1).upper()
    acc = m.group(2)
    if acc == "b" or acc == "B":
        acc_token = "b"
    elif acc == "#":
        acc_token = "#"
    else:
        acc_token = ""
    token_key = letter + acc_token
    if acc_token == "b":
        token_key = letter + "b"
    elif acc_token == "#":
        token_key = letter + "#"
    else:
        token_key = letter
    pc_key = token_key.replace("b", "B") if "b" in token_key else token_key
    pc_key = pc_key.upper()
    if pc_key not in PITCH_CLASS_MAP:
        return None, None, None, None
    pc = PITCH_CLASS_MAP[pc_key]
    octave = int(m.group(3))
    display = letter + (acc if acc in ("#", "b", "B") else "") + str(octave)
    if acc == "B":
        display = letter + "b" + str(octave)
    return display, pc, octave, letter


def note_to_midi(note_string: str | None) -> float | None:
    _tok, pc, octave, _ = parse_note(note_string)
    if pc is None or octave is None:
        return None
    return float(12 * (octave + 1) + pc)


def register_from_midi(midi: float | None) -> str | None:
    if midi is None or (isinstance(midi, float) and np.isnan(midi)):
        return None
    m = int(round(midi))
    if m < 36:
        return "Very low"
    if m < 48:
        return "Low"
    if m < 60:
        return "Middle"
    if m < 72:
        return "High"
    return "Very high"


def pitch_class_name(note_string: str | None) -> str | None:
    _t, pc, octave, letter = parse_note(note_string)
    if pc is None:
        return None
    m = re.match(r"^([A-Ga-g])([#b]?)", str(note_string).strip())
    if not m:
        return None
    base = m.group(1).upper() + (m.group(2) if m.group(2) in ("#", "b") else "")
    return base


def find_note_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        if str(c).strip().lower() == "note":
            return str(c)
    return None


def _first_matching_column(df: pd.DataFrame, names: Sequence[str]) -> str | None:
    colset = {str(c): str(c) for c in df.columns}
    lower_map = {str(c).lower(): str(c) for c in df.columns}
    for name in names:
        if name in colset:
            return name
        low = name.lower()
        if low in lower_map:
            return lower_map[low]
    return None


def _rename_frame_to_canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases to canonical names when the canonical column is absent."""
    out = df.copy()
    renames: Dict[str, str] = {}
    for col in list(out.columns):
        sc = str(col)
        if sc in renames:
            continue
        for canon, aliases in COLUMN_ALIASES.items():
            if sc == canon:
                break
            if sc in aliases or sc.lower() in {a.lower() for a in aliases}:
                if canon not in out.columns:
                    renames[col] = canon
                break
    return out.rename(columns=renames) if renames else out


def _coalesce_series(a: pd.Series, b: pd.Series) -> pd.Series:
    return a.where(a.notna() & (a.astype(str) != ""), b)


def merge_workbook_frames(path: Path, warnings: List[str]) -> pd.DataFrame:
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Cannot read workbook: {path}: {e}") from e
    names = set(xl.sheet_names)
    if "Density_Metrics" not in names:
        raise ValueError("Required sheet 'Density_Metrics' is missing from the compiled workbook.")

    merged: Optional[pd.DataFrame] = None
    note_key = "Note"

    for sheet in MERGE_SHEETS:
        if sheet not in names:
            warnings.append(f"Optional sheet '{sheet}' not found; skipped.")
            continue
        raw = pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
        if raw.empty:
            warnings.append(f"Sheet '{sheet}' is empty; skipped.")
            continue
        nc = find_note_column(raw)
        if nc is None:
            warnings.append(f"Sheet '{sheet}' has no 'Note' column; skipped.")
            continue
        frame = raw.rename(columns={nc: note_key})
        frame = _rename_frame_to_canonical(frame)
        if merged is None:
            merged = frame
            continue
        # Outer merge on Note
        merged = merged.merge(frame, on=note_key, how="outer", suffixes=("", "_y"))
        drop_y: List[str] = []
        for col in list(merged.columns):
            if not col.endswith("_y"):
                continue
            base = col[:-2]
            if base == note_key:
                drop_y.append(col)
                continue
            if base in merged.columns:
                merged[base] = _coalesce_series(merged[base], merged[col])
            else:
                merged[base] = merged[col]
            drop_y.append(col)
        merged = merged.drop(columns=drop_y, errors="ignore")
        # Drop duplicate Note columns if any
        dup_note = [c for c in merged.columns if str(c).lower() == "note" and c != note_key]
        merged = merged.drop(columns=dup_note, errors="ignore")

    if merged is None or merged.empty:
        raise ValueError("No per-note rows could be loaded from Density_Metrics.")

    return merged


def _series_or_nan(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def _series_str(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name].astype(str).replace({"nan": np.nan, "None": np.nan})
    return pd.Series(np.nan, index=df.index)


def min_max_normalize(s: pd.Series) -> Tuple[pd.Series, bool]:
    """Returns (normalized, warn_constant)."""
    v = pd.to_numeric(s, errors="coerce")
    vmin = v.min(skipna=True)
    vmax = v.max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
        return pd.Series(np.nan, index=s.index), True
    return (v - vmin) / (vmax - vmin), False


def _pick_series(df: pd.DataFrame, canonical: str) -> pd.Series:
    """First matching column among aliases, else canonical name, else NaN column."""
    if canonical in COLUMN_ALIASES:
        for a in COLUMN_ALIASES[canonical]:
            if a in df.columns:
                return df[a]
    if canonical in df.columns:
        return df[canonical]
    return pd.Series(np.nan, index=df.index)


def _note_to_source_file_lookup(merged: pd.DataFrame) -> Dict[str, Any]:
    """Last row per note for ``Source_File`` (or aliases); used for chart path tie-break without exporting the column."""
    if merged.empty or "Note" not in merged.columns:
        return {}
    src = _pick_series(merged, "Source_File")
    m = merged[["Note"]].copy()
    m["_src"] = src.reindex(m.index).to_numpy()
    last = m.groupby("Note", as_index=False, sort=False).last()
    return {str(r["Note"]).strip(): r["_src"] for _, r in last.iterrows()}


def _all_blank_or_nan(series: pd.Series) -> bool:
    if series.isna().all():
        return True
    t = series.dropna().astype(str).str.strip()
    return len(t) == 0 or t.eq("").all()


# Longest dynamic first (prefer pp over p)
_DYNAMIC_ORDER: Tuple[str, ...] = ("fff", "ppp", "ff", "pp", "mp", "mf", "p", "f")
_DYNAMIC_TOKEN_OK = frozenset(_DYNAMIC_ORDER)


def _ascii_fold(text: str) -> str:
    s = unicodedata.normalize("NFKD", text)
    return "".join(c for c in s if not unicodedata.combining(c))


def _tokenize_metadata_text(text: str | None) -> List[str]:
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return []
    s = _ascii_fold(str(text).strip()).lower()
    if not s:
        return []
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    parts = re.split(r"[_\s\.\/\\-]+", s)
    return [p for p in parts if len(p) >= 2 or p in _DYNAMIC_TOKEN_OK]


# (canonical name, substring phrases (longest match wins within rule), whole-token abbrev set)
_INSTRUMENT_RULES: Tuple[Tuple[str, Tuple[str, ...], Tuple[str, ...]], ...] = (
    ("English Horn", ("englishhorn", "english horn", "cor anglais", "coranglais", "cornoingles", "corno ingles"), ("eh",)),
    ("Clarinet", ("clarinet", "clarinete", "clarinetto"), ("clar", "cl")),
    ("Bassoon", ("bassoon", "basson", "fagote", "fagott", "fagotto", "fgt"), ("bn",)),
    ("Flute", ("flute", "flauta", "flûte"), ("fl",)),
    ("Oboe", ("oboe", "oboé"), ("ob",)),
    ("Violin", ("violin", "violino"), ("vln",)),
    ("Viola", ("viola",), ("vla",)),
    ("Cello", ("violoncello", "violoncelo", "cello"), ("vc",)),
    ("Contrabass", ("contrabass", "contrabaixo", "double bass", "doublebass"), ("cb",)),
    ("Trumpet", ("trumpet", "trompete"), ("tpt",)),
    ("Horn", ("frenchhorn", "french horn", "trompa"), ("hn",)),
    ("Trombone", ("trombone",), ("tbn",)),
    ("Tuba", ("tuba",), ("tba",)),
)


def infer_instrument_conservative(text: str | None) -> Optional[str]:
    """Infer instrument from path/filename text using token and phrase rules."""
    if not text:
        return None
    folded = _ascii_fold(str(text)).lower()
    compact = re.sub(r"[_\s\.\/\\-]+", "", folded)
    tokens = set(_tokenize_metadata_text(text))
    for canon, phrases, abbrevs in _INSTRUMENT_RULES:
        for ph in sorted(phrases, key=len, reverse=True):
            ph_c = re.sub(r"\s+", "", ph)
            if ph in folded or ph_c in compact:
                return canon
        for ab in abbrevs:
            if ab in tokens:
                return canon
    return None


def infer_dynamic_conservative(text: str | None) -> Optional[str]:
    if not text:
        return None
    folded = _ascii_fold(str(text)).lower()
    tokens = _tokenize_metadata_text(text)
    hits: List[str] = []
    for tok in tokens:
        if tok in _DYNAMIC_ORDER:
            hits.append(tok)
    for dyn in _DYNAMIC_ORDER:
        for sep in ("-", "_", " "):
            if f"{sep}{dyn}{sep}" in f"{sep}{folded}{sep}":
                hits.append(dyn)
                break
    if not hits:
        return None
    hits.sort(key=lambda d: (-len(d), _DYNAMIC_ORDER.index(d) if d in _DYNAMIC_ORDER else 99))
    return hits[0]


def infer_instrument_from_text(text: str | None) -> Optional[str]:
    return infer_instrument_conservative(text)


def infer_dynamic_from_text(text: str | None) -> Optional[str]:
    return infer_dynamic_conservative(text)


def _cell_str(row: pd.Series, col: str) -> Optional[str]:
    if col not in row.index:
        return None
    v = row.get(col)
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip()
    return s or None


def _dedupe_preserve(seq: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in seq:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _instrument_inference_sources(merged: pd.DataFrame, i: int, compiled_workbook: Path) -> List[str]:
    """
    Priority-ordered strings for conservative instrument inference (first hit wins).

    Skips canonical ``Instrument`` / ``Dynamic`` here: those are handled as explicit workbook
    columns in ``_build_instrument_dynamic_series`` before inference runs.
    """
    row = merged.iloc[i]
    texts: List[str] = []
    hint_cols = (
        "instrument_detected",
        "instrument_family",
        "instrument_guess",
        "instrument_name",
        "instrument_label",
        "selected_instrument",
        "inferred_instrument",
        "per_note_instrument",
    )
    for col in hint_cols:
        s = _cell_str(row, col)
        if s:
            texts.append(s)
    for col in merged.columns:
        lc = str(col).lower()
        if lc in {"instrument", "note"} or col in hint_cols:
            continue
        if "instrument" in lc and "instrumentation" not in lc:
            s = _cell_str(row, str(col))
            if s and s.lower() not in {"unknown", "nan", "none"}:
                texts.append(s)
    src = _pick_series(merged, "Source_File")
    if len(src) > i:
        v = src.iloc[i]
        if pd.notna(v) and str(v).strip():
            texts.append(str(v))
    sw = _pick_series(merged, "Source_Workbook")
    if len(sw) > i:
        v = sw.iloc[i]
        if pd.notna(v) and str(v).strip():
            texts.append(str(v))
    if publication_clean_export_enabled():
        texts.append(compiled_workbook.name)
    else:
        texts.append(str(compiled_workbook.resolve()))
        p = compiled_workbook.resolve()
        cur: Optional[Path] = p.parent
        for _ in range(6):
            if cur is None:
                break
            texts.append(str(cur))
            if cur.name:
                texts.append(cur.name)
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
    return _dedupe_preserve(texts)


def _dynamic_inference_sources(merged: pd.DataFrame, i: int, compiled_workbook: Path) -> List[str]:
    row = merged.iloc[i]
    texts: List[str] = []
    hint_cols = (
        "dynamic_detected",
        "dynamics",
        "written_dynamic",
        "dynamic_marking",
        "notated_dynamic",
        "per_note_dynamic",
    )
    for col in hint_cols:
        s = _cell_str(row, col)
        if s:
            texts.append(s)
    for col in merged.columns:
        lc = str(col).lower()
        if lc == "dynamic" or col in hint_cols:
            continue
        if "dynamic" in lc:
            s = _cell_str(row, str(col))
            if s and s.lower() not in {"unknown", "nan", "none"}:
                texts.append(s)
    src = _pick_series(merged, "Source_File")
    if len(src) > i:
        v = src.iloc[i]
        if pd.notna(v) and str(v).strip():
            texts.append(str(v))
    sw = _pick_series(merged, "Source_Workbook")
    if len(sw) > i:
        v = sw.iloc[i]
        if pd.notna(v) and str(v).strip():
            texts.append(str(v))
    if publication_clean_export_enabled():
        texts.append(compiled_workbook.name)
    else:
        texts.append(str(compiled_workbook.resolve()))
        p = compiled_workbook.resolve()
        cur: Optional[Path] = p.parent
        for _ in range(6):
            if cur is None:
                break
            texts.append(str(cur))
            if cur.name:
                texts.append(cur.name)
            parent = cur.parent
            if parent == cur:
                break
            cur = parent
    return _dedupe_preserve(texts)


def _infer_first_from_sources(sources: Sequence[str], infer_fn) -> Optional[str]:
    for raw in sources:
        g = infer_fn(raw)
        if g:
            return g
    return None


def _build_instrument_dynamic_series(
    merged: pd.DataFrame,
    compiled_workbook: Path,
    warnings: List[str],
    meta: ResearchExportMetadata,
) -> Tuple[pd.Series, pd.Series]:
    """Resolve Instrument and Dynamic per priority rules and CLI overrides."""
    n = len(merged)
    orig_inst = _pick_series(merged, "Instrument").copy()
    orig_dyn = _pick_series(merged, "Dynamic").copy()

    inst_out = orig_inst.astype(object)
    dyn_out = orig_dyn.astype(object)

    for i in range(n):
        if not (pd.isna(inst_out.iloc[i]) or str(inst_out.iloc[i]).strip() == ""):
            continue
        guess = _infer_first_from_sources(
            _instrument_inference_sources(merged, i, compiled_workbook),
            infer_instrument_conservative,
        )
        if guess:
            inst_out.iloc[i] = guess

    for i in range(n):
        if not (pd.isna(dyn_out.iloc[i]) or str(dyn_out.iloc[i]).strip() == ""):
            continue
        guess = _infer_first_from_sources(
            _dynamic_inference_sources(merged, i, compiled_workbook),
            infer_dynamic_conservative,
        )
        if guess:
            dyn_out.iloc[i] = guess

    if meta.instrument:
        for i in range(n):
            row_has = pd.notna(orig_inst.iloc[i]) and str(orig_inst.iloc[i]).strip() != ""
            if meta.force_metadata or not row_has:
                inst_out.iloc[i] = meta.instrument
    if meta.dynamic:
        for i in range(n):
            row_has = pd.notna(orig_dyn.iloc[i]) and str(orig_dyn.iloc[i]).strip() != ""
            if meta.force_metadata or not row_has:
                dyn_out.iloc[i] = meta.dynamic

    if _all_blank_or_nan(inst_out):
        warnings.append("Instrument column missing and could not be inferred; left blank.")
    if _all_blank_or_nan(dyn_out):
        warnings.append("Dynamic column missing or ambiguous; could not be inferred confidently; left blank.")

    return inst_out, dyn_out


def _collect_note_named_directories(analysis_root: Path) -> Dict[str, List[Path]]:
    """Map note folder name -> directories under ``analysis_root`` whose final component parses as a note."""
    out: Dict[str, List[Path]] = defaultdict(list)
    if not analysis_root.is_dir():
        return dict(out)
    try:
        for dirpath, dirnames, _filenames in os.walk(analysis_root):
            for d in dirnames:
                if parse_note(d)[0]:
                    out[d].append(Path(dirpath) / d)
    except OSError:
        pass
    return dict(out)


def _pick_note_folder_for_row(
    note: str,
    candidates: List[Path],
    source_file_val: Any,
    warnings: List[str],
    warn_key: str,
) -> Optional[Path]:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    stem = None
    if source_file_val is not None and pd.notna(source_file_val):
        stem = Path(str(source_file_val)).stem.lower()
    scored: List[Tuple[int, str, Path]] = []
    for c in candidates:
        pnorm = str(c).lower().replace("\\", "/")
        score = 1 if stem and stem in pnorm else 0
        scored.append((score, pnorm, c))
    scored.sort(key=lambda t: (-t[0], t[1]))
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        warnings.append(
            f"{warn_key}: multiple folders for note {note!r}; using {scored[0][2]} (tie-break: sorted path)."
        )
    return scored[0][2]


def _rel_under_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def apply_per_note_chart_paths(
    sd: pd.DataFrame,
    compiled_workbook: Path,
    merged: pd.DataFrame,
    warnings: List[str],
) -> None:
    """Fill ``amplitude_mass_chart_file`` / ``energy_ratio_chart_file`` using filesystem search."""
    analysis_root = compiled_workbook.parent.resolve()
    note_index = _collect_note_named_directories(analysis_root)
    note_src = _note_to_source_file_lookup(merged)
    any_amp_still_missing = False
    any_ratio_still_missing = False

    for pos in range(len(sd)):
        idx = sd.index[pos]
        note = sd.iloc[pos].get("Note")
        if note is None or (isinstance(note, float) and np.isnan(note)):
            continue
        note_s = str(note).strip()
        src = note_src.get(note_s)
        amp_col = "amplitude_mass_chart_file"
        erg_col = "energy_ratio_chart_file"
        if amp_col not in sd.columns:
            sd[amp_col] = np.nan
        if erg_col not in sd.columns:
            sd[erg_col] = np.nan
        cur_amp = sd.at[idx, amp_col]
        cur_erg = sd.at[idx, erg_col]
        if pd.isna(cur_amp) or str(cur_amp).strip() == "":
            cands = note_index.get(note_s, [])
            folder = _pick_note_folder_for_row(
                note_s,
                cands,
                src,
                warnings,
                "Chart path",
            )
            if folder and (folder / CHART_AMPLITUDE_NAME).is_file():
                sd.at[idx, amp_col] = _rel_under_root(folder / CHART_AMPLITUDE_NAME, analysis_root)
            else:
                any_amp_still_missing = True
        if pd.isna(cur_erg) or str(cur_erg).strip() == "":
            cands = note_index.get(note_s, [])
            folder = _pick_note_folder_for_row(
                note_s,
                cands,
                src,
                warnings,
                "Chart path",
            )
            rel = None
            if folder:
                if (folder / CHART_ENERGY_RATIO_NAME).is_file():
                    rel = _rel_under_root(folder / CHART_ENERGY_RATIO_NAME, analysis_root)
                elif (folder / CHART_ENERGY_LEGACY_NAME).is_file():
                    rel = _rel_under_root(folder / CHART_ENERGY_LEGACY_NAME, analysis_root)
            if rel:
                sd.at[idx, erg_col] = rel
            else:
                any_ratio_still_missing = True

    if any_amp_still_missing:
        warnings.append(
            "Per-note component_amplitude_mass_pie.png not found under analysis folder for at least one note "
            "(see amplitude_mass_chart_file on Spectral_Density_Metrics)."
        )
    if any_ratio_still_missing:
        warnings.append(
            "Per-note component_energy_ratio_pie.png / component_energy_pie.png not found under analysis folder "
            "for at least one note (see energy_ratio_chart_file on Spectral_Density_Metrics)."
        )


def build_spectral_density_metrics(
    merged: pd.DataFrame,
    warnings: List[str],
    compiled_workbook: Path,
    meta: Optional[ResearchExportMetadata] = None,
) -> pd.DataFrame:
    meta = meta or ResearchExportMetadata()
    note_col = "Note"
    notes = merged[note_col] if note_col in merged.columns else pd.Series(np.nan, index=merged.index)

    midi_list = [note_to_midi(x) for x in notes]
    midi = pd.Series(midi_list, index=merged.index, dtype=float)

    norm_warns: Dict[str, bool] = {}

    instrument, dynamic = _build_instrument_dynamic_series(merged, compiled_workbook, warnings, meta)

    out = pd.DataFrame(
        {
            "Instrument": instrument,
            "Note": notes,
            "MIDI": midi,
            "Pitch_Class": [pitch_class_name(x) for x in notes],
            "Octave": [parse_note(x)[2] for x in notes],
            "Register": [register_from_midi(m) for m in midi_list],
            "Dynamic": dynamic,
            "f0_nominal_hz": _series_or_nan(merged, "f0_nominal_hz"),
            "f0_final_hz": _pick_series(merged, "f0_final_hz"),
            "f0_source": _series_str(merged, "f0_source"),
            "f0_fit_accepted": merged["f0_fit_accepted"]
            if "f0_fit_accepted" in merged.columns
            else pd.Series(np.nan, index=merged.index),
            "f0_detuning_cents_from_nominal": _series_or_nan(merged, "f0_detuning_cents_from_nominal"),
            "density_metric_raw": _series_or_nan(merged, "density_metric_raw"),
            "density_metric_normalized": _series_or_nan(merged, "density_metric_normalized"),
            "density_weighted_sum": _series_or_nan(merged, "density_weighted_sum"),
            "density_log_weighted": _series_or_nan(merged, "density_log_weighted"),
            "Total sum": _series_or_nan(merged, "Total sum"),
            "effective_partial_density": _series_or_nan(merged, "effective_partial_density"),
            "spectral_entropy": _series_or_nan(merged, "spectral_entropy"),
            "harmonic_density_sum": _pick_series(merged, "harmonic_density_sum"),
            "inharmonic_density_sum": _pick_series(merged, "inharmonic_density_sum"),
            "subbass_density_sum": _pick_series(merged, "subbass_density_sum"),
            "weighted_harmonic_density_contribution": _series_or_nan(merged, "weighted_harmonic_density_contribution"),
            "weighted_inharmonic_density_contribution": _series_or_nan(merged, "weighted_inharmonic_density_contribution"),
            "weighted_subbass_density_contribution": _series_or_nan(merged, "weighted_subbass_density_contribution"),
            "harmonic_energy_sum": _pick_series(merged, "harmonic_energy_sum"),
            "inharmonic_energy_sum": _pick_series(merged, "inharmonic_energy_sum"),
            "subbass_energy_sum": _pick_series(merged, "subbass_energy_sum"),
            "total_component_energy": _series_or_nan(merged, "total_component_energy"),
            "harmonic_energy_ratio": _pick_series(merged, "harmonic_energy_ratio"),
            "inharmonic_energy_ratio": _pick_series(merged, "inharmonic_energy_ratio"),
            "subbass_energy_ratio": _pick_series(merged, "subbass_energy_ratio"),
            "harmonic_order_count": _series_or_nan(merged, "harmonic_order_count"),
            "harmonic_alignment_status": _series_str(merged, "harmonic_alignment_status"),
            "harmonic_alignment_coverage_ratio": _series_or_nan(merged, "harmonic_alignment_coverage_ratio"),
            "mean_abs_harmonic_deviation_cents": _series_or_nan(merged, "mean_abs_harmonic_deviation_cents"),
            "max_abs_harmonic_deviation_cents": _series_or_nan(merged, "max_abs_harmonic_deviation_cents"),
            "debug_counts_invariant_status": _series_str(merged, "debug_counts_invariant_status"),
            "publication_output_allowed": merged["publication_output_allowed"]
            if "publication_output_allowed" in merged.columns
            else pd.Series(np.nan, index=merged.index),
        }
    )

    for extra in (
        "harmonic_amplitude_sum",
        "inharmonic_amplitude_sum",
        "subbass_amplitude_sum",
        "amplitude_mass_chart_file",
        "energy_ratio_chart_file",
    ):
        out[extra] = merged[extra] if extra in merged.columns else np.nan

    if "canonical_density" in merged.columns:
        out["canonical_density"] = pd.to_numeric(merged["canonical_density"], errors="coerce")

    for col in (
        "density_metric_raw",
        "density_weighted_sum",
        "Total sum",
        "effective_partial_density",
        "spectral_entropy",
    ):
        s = pd.to_numeric(out[col], errors="coerce")
        n, w = min_max_normalize(s)
        out[f"{col}_norm_for_chart"] = n
        if w:
            norm_warns[col] = True

    if norm_warns:
        warnings.append(
            "Min-max chart normalization undefined for: "
            + ", ".join(sorted(norm_warns.keys()))
            + " (constant or all-missing); chart columns set to NaN."
        )

    out = out.sort_values("MIDI", na_position="last", kind="mergesort")
    return out


def build_component_balance(sd: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    cols = [
        "Instrument",
        "Note",
        "MIDI",
        "Register",
        "Dynamic",
        "harmonic_density_sum",
        "inharmonic_density_sum",
        "subbass_density_sum",
        "Total sum",
        "harmonic_energy_ratio",
        "inharmonic_energy_ratio",
        "subbass_energy_ratio",
        "weighted_harmonic_density_contribution",
        "weighted_inharmonic_density_contribution",
        "weighted_subbass_density_contribution",
        "density_metric_raw",
        "harmonic_amplitude_sum",
        "inharmonic_amplitude_sum",
        "subbass_amplitude_sum",
        "density_weighted_sum",
        "density_log_weighted",
        "amplitude_mass_chart_file",
        "energy_ratio_chart_file",
    ]
    cb = pd.DataFrame({c: sd[c] if c in sd.columns else np.nan for c in cols})

    w_h = pd.to_numeric(cb["weighted_harmonic_density_contribution"], errors="coerce")
    w_i = pd.to_numeric(cb["weighted_inharmonic_density_contribution"], errors="coerce")
    w_s = pd.to_numeric(cb["weighted_subbass_density_contribution"], errors="coerce")
    if w_h.isna().all() and w_i.isna().all() and w_s.isna().all():
        warnings.append("Component_Balance: weighted density contributions missing; recomputed checks set to NaN.")

    cb["energy_ratio_sum"] = (
        pd.to_numeric(cb["harmonic_energy_ratio"], errors="coerce")
        + pd.to_numeric(cb["inharmonic_energy_ratio"], errors="coerce")
        + pd.to_numeric(cb["subbass_energy_ratio"], errors="coerce")
    )
    cb["density_metric_raw_recomputed"] = w_h + w_i + w_s
    raw = pd.to_numeric(cb["density_metric_raw"], errors="coerce")
    rec = pd.to_numeric(cb["density_metric_raw_recomputed"], errors="coerce")
    cb["density_metric_raw_difference"] = raw - rec

    h_d = pd.to_numeric(cb["harmonic_density_sum"], errors="coerce")
    i_d = pd.to_numeric(cb["inharmonic_density_sum"], errors="coerce")
    s_d = pd.to_numeric(cb["subbass_density_sum"], errors="coerce")
    if h_d.isna().all() and i_d.isna().all() and s_d.isna().all():
        warnings.append("Component_Balance: component density sums missing; total_sum_recomputed set to NaN.")

    cb["total_sum_recomputed"] = h_d + i_d + s_d
    tot = pd.to_numeric(cb["Total sum"], errors="coerce")
    cb["total_sum_difference"] = tot - pd.to_numeric(cb["total_sum_recomputed"], errors="coerce")

    def row_status(row: pd.Series) -> str:
        ers = row["energy_ratio_sum"]
        dmd = row["density_metric_raw_difference"]
        tsd = row["total_sum_difference"]
        try:
            er_ok = bool(pd.isna(ers)) or abs(float(ers) - 1.0) <= 0.01
        except (TypeError, ValueError):
            er_ok = bool(pd.isna(ers))

        dm_raw = row.get("density_metric_raw")
        dm_rec = row.get("density_metric_raw_recomputed")
        try:
            if pd.isna(dmd):
                dm_ok = bool(pd.isna(dm_raw) and pd.isna(dm_rec))
            else:
                dm_ok = abs(float(dmd)) <= 1e-6
        except (TypeError, ValueError):
            dm_ok = False

        tot = row.get("Total sum")
        tre = row.get("total_sum_recomputed")
        try:
            if pd.isna(tsd):
                ts_ok = pd.isna(tot) and pd.isna(tre)
            else:
                ts_ok = abs(float(tsd)) <= 1e-6
        except (TypeError, ValueError):
            ts_ok = False

        if er_ok and dm_ok and ts_ok:
            return "passed"
        return "warning"

    cb["component_balance_status"] = cb.apply(row_status, axis=1)
    return cb.sort_values("MIDI", na_position="last", kind="mergesort")


def build_validation_summary(merged: pd.DataFrame, sd: pd.DataFrame, warnings: List[str]) -> pd.DataFrame:
    cols = [
        "Instrument",
        "Note",
        "MIDI",
        "Register",
        "f0_nominal_hz",
        "f0_final_hz",
        "f0_source",
        "f0_final_source",
        "f0_fit_accepted",
        "f0_fit_quality",
        "f0_fit_residual_std_hz",
        "f0_fit_rejection_reason",
        "f0_detuning_cents_from_nominal",
        "harmonic_alignment_status",
        "harmonic_alignment_coverage_ratio",
        "harmonic_alignment_energy_coverage_ratio",
        "mean_abs_harmonic_deviation_cents",
        "max_abs_harmonic_deviation_cents",
        "rms_harmonic_deviation_cents",
        "debug_counts_invariant_status",
        "debug_counts_invariant_failures",
        "input_schema_validation_status",
        "publication_output_allowed",
    ]
    base_prefix = ("Instrument", "Note", "MIDI", "Register")
    base_cols = [c for c in base_prefix if c in sd.columns]
    vs_base = sd[base_cols].copy()
    m = merged.copy()
    if "Note" not in m.columns:
        nc = find_note_column(m)
        if nc:
            m = m.rename(columns={nc: "Note"})
    m = m.groupby("Note", as_index=False).last()
    rest = [c for c in cols if c not in base_cols and c in m.columns]
    vs = vs_base.merge(m[["Note"] + rest], on="Note", how="left")
    for c in cols:
        if c not in vs.columns:
            vs[c] = np.nan
    vs = vs[cols]

    def f0_contradiction(row: pd.Series) -> bool:
        src = str(row.get("f0_source", "") or "")
        fsrc = str(row.get("f0_final_source", "") or "")
        if "prior_constrained_harmonic_fit" in (src, fsrc):
            acc = row.get("f0_fit_accepted")
            if acc is False or str(acc).lower() == "false":
                return True
        return False

    ok_align = {"ok", "excellent", "good"}

    def val_status(row: pd.Series) -> str:
        if f0_contradiction(row):
            return "warning"
        dci = str(row.get("debug_counts_invariant_status", "") or "").lower().strip()
        if dci in ("failed", "fail", "warning"):
            return "warning"
        ha = str(row.get("harmonic_alignment_status", "") or "").lower().strip()
        if ha and ha not in ok_align:
            return "warning"
        return "passed"

    vs["validation_summary_status"] = vs.apply(val_status, axis=1)
    if vs["f0_final_source"].isna().all():
        warnings.append("Validation_Summary: f0_final_source column missing from source workbook.")
    return vs.sort_values("MIDI", na_position="last", kind="mergesort")


def build_charts_data(sd: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Note",
        "MIDI",
        "density_weighted_sum",
        "density_metric_raw",
        "Total sum",
        "effective_partial_density",
        "spectral_entropy",
        "density_weighted_sum_norm_for_chart",
        "density_metric_raw_norm_for_chart",
        "Total sum_norm_for_chart",
        "effective_partial_density_norm_for_chart",
        "spectral_entropy_norm_for_chart",
        "weighted_harmonic_density_contribution",
        "weighted_inharmonic_density_contribution",
        "weighted_subbass_density_contribution",
        "harmonic_energy_ratio",
        "inharmonic_energy_ratio",
        "subbass_energy_ratio",
    ]
    cd = pd.DataFrame({c: sd[c] for c in cols if c in sd.columns})
    for c in cols:
        if c not in cd.columns:
            cd[c] = np.nan
    return cd[cols].sort_values("MIDI", na_position="last", kind="mergesort")


def load_analysis_metadata(path: Path, warnings: List[str]) -> Dict[str, Any]:
    try:
        xl = pd.ExcelFile(path, engine="openpyxl")
        if "Analysis_Metadata" not in xl.sheet_names:
            warnings.append("Analysis_Metadata sheet missing; Metadata sheet will have blanks.")
            return {}
        am = pd.read_excel(path, sheet_name="Analysis_Metadata", engine="openpyxl", header=0)
        if am is None or am.empty:
            return {}
        cols_lower = {str(c).strip().lower() for c in am.columns}
        if "parameter" in cols_lower and "value" in cols_lower:
            pcol = next(c for c in am.columns if str(c).strip().lower() == "parameter")
            vcol = next(c for c in am.columns if str(c).strip().lower() == "value")
            out: Dict[str, Any] = {}
            for _, r in am.iterrows():
                k = str(r.get(pcol, "")).strip()
                if k:
                    out[k] = r.get(vcol)
            return out
        if len(am) == 1:
            row = am.iloc[0]
            return {str(c).strip(): row.iloc[i] for i, c in enumerate(am.columns)}
        am2 = pd.read_excel(path, sheet_name="Analysis_Metadata", engine="openpyxl", header=None)
        if am2.shape[1] < 2:
            return {}
        keys = am2.iloc[:, 0].astype(str)
        vals = am2.iloc[:, 1]
        return {str(k).strip(): v for k, v in zip(keys, vals, strict=False)}
    except Exception:
        warnings.append("Could not parse Analysis_Metadata.")
        return {}


def build_metadata_rows(
    path: Path,
    meta: Mapping[str, Any],
    sd: pd.DataFrame,
    warnings: List[str],
) -> pd.DataFrame:
    now = format_utc_publication_timestamp()

    meta_missing: set[str] = set()

    def mget(key: str) -> Any:
        if key in meta:
            return meta[key]
        lk = key.lower()
        for k, v in meta.items():
            if str(k).strip().lower() == lk:
                return v
        meta_missing.add(key)
        return np.nan

    pitch_range = np.nan
    if sd["MIDI"].notna().any():
        pitch_range = f"{int(sd['MIDI'].min())}-{int(sd['MIDI'].max())}"

    rows = {
        "source_compiled_workbook": str(path.resolve()),
        "research_export_created_at": now,
        "research_export_script": SCRIPT_NAME,
        "research_export_version": SCRIPT_VERSION,
        "pipeline_contract_version": mget("pipeline_contract_version"),
        "analysis_schema_version": mget("ANALYSIS_SCHEMA_VERSION"),
        "stage1_module": mget("stage1_module"),
        "stage1_class": mget("stage1_class"),
        "stage2_module": mget("stage2_module"),
        "stage2_function": mget("stage2_function"),
        "compiled_from": mget("compiled_from"),
        "accepted_input_engine": mget("accepted_input_engine"),
        "legacy_pipeline_used": mget("legacy_pipeline_used"),
        "legacy_super_json_allowed": mget("legacy_super_json_allowed"),
        "publication_output_allowed": mget("publication_output_allowed"),
        "input_schema_validation_status": mget("input_schema_validation_status"),
        "weight_function": mget("weight_function"),
        "window_type": mget("window_type"),
        "frequency_min_hz": mget("frequency_min_hz"),
        "frequency_max_hz": mget("frequency_max_hz"),
        "magnitude_min_db": mget("magnitude_min_db"),
        "magnitude_max_db": mget("magnitude_max_db"),
        "notes_count": len(sd),
        "pitch_range": pitch_range,
        "instrument_detected": (
            sd["Instrument"].dropna().iloc[0]
            if "Instrument" in sd.columns and sd["Instrument"].notna().any()
            else np.nan
        ),
        "dynamic_detected": (
            sd["Dynamic"].dropna().iloc[0]
            if "Dynamic" in sd.columns and sd["Dynamic"].notna().any()
            else np.nan
        ),
    }
    rows = apply_publication_clean_research_metadata_fields(rows, workbook_basename=path.name)
    if meta_missing:
        warnings.append(
            "Metadata sheet missing keys (left blank in Metadata sheet): " + ", ".join(sorted(meta_missing))
        )
    return pd.DataFrame([{"Field": k, "Value": rows[k]} for k in rows])


def readme_lines(
    source: Path,
    warnings: List[str],
    n_notes: int,
    pitch_range: str,
    instrument: str,
    dynamic: str,
    generated: str,
) -> List[str]:
    if publication_clean_export_enabled():
        lines = [
            "Spectral density — research export workbook",
            "",
            "Source compiled workbook (filename only):",
            source.name,
            "",
            f"Generated: {generated}",
            f"Script: {SCRIPT_NAME}  version {SCRIPT_VERSION}",
            f"Number of notes in export: {n_notes}",
            "",
            "Purpose",
            "-------",
            "This workbook is a reduced, professionally formatted export intended for secondary research,",
            "plotting, thesis writing, and visual inspection.",
            "",
            "The file compiled_density_metrics.xlsx remains the complete technical and audit export.",
            "This workbook (compiled_density_metrics_research.xlsx) is the recommended research and reporting export.",
            "",
            "Metric hierarchy (read carefully)",
            "-----------------------------------",
            "density_metric_raw:",
            "    D_H·w_H + D_I·w_I + D_S·w_S where each D_* uses the compile weight_function (linear, log, sqrt,",
            "    d3, …) and w_* are measured component_harmonic / inharmonic / subbass energy ratios.",
            "",
            "density_weighted_sum:",
            "    Numerically equal to density_metric_raw. Changes when you change the compile weight_function.",
            "    harmonic_amplitude_sum (if present) is a separate linear diagnostic and does not follow that key.",
            "",
            "Total sum:",
            "    Unweighted sum of per-band D values (D_H + D_I + D_S); diagnostic, not energy-ratio-weighted.",
            "",
            "effective_partial_density:",
            "    Effective participation descriptor; not total spectral mass.",
            "",
            "spectral_entropy:",
            "    Distributional spread of spectral power.",
            "",
            "harmonic_energy_ratio / inharmonic_energy_ratio / subbass_energy_ratio:",
            "    Measured component energy ratios, not full psychoacoustic perceptual weights.",
            "",
            "nonharmonic / inharmonic fields:",
            "    Interpret as nonharmonic candidate material unless stricter validation is explicitly present.",
            "",
            "subbass / low-frequency fields:",
            "    Interpret as low-frequency residual material, not automatically 'ground noise'.",
            "",
            "Instrument and dynamic metadata",
            "-------------------------------",
            "Instrument and dynamic may be read from source metadata, inferred conservatively from filenames,",
            "folder paths, and the compiled workbook location, or supplied manually with --instrument and --dynamic.",
            "Use --force-metadata to override non-empty workbook values with the CLI arguments.",
            "",
            "Sheets",
            "------",
            "README — This sheet.",
            "Dashboard — Overview KPIs and charts (charts omitted if export was run with --no-charts).",
            "Spectral_Density_Metrics — Main clean per-note table for analysis.",
            "Component_Balance — How density_metric_raw and totals relate to components and energy ratios.",
            "Validation_Summary — Compact QC fields including f0 and alignment.",
            "Charts_Data — Chart source data (sorted by MIDI); Dashboard charts reference this sheet.",
            "Metadata — Pipeline and export metadata extracted from Analysis_Metadata plus export audit fields.",
            "",
            "Warnings / Missing Fields",
            "-------------------------",
        ]
    else:
        lines = [
            "Spectral density — research export workbook",
            "",
            "Source compiled workbook:",
            str(source.resolve()),
            "",
            f"Generated: {generated}",
            f"Script: {SCRIPT_NAME}  version {SCRIPT_VERSION}",
            f"Number of notes in export: {n_notes}",
            "",
            "Purpose",
            "-------",
            "This workbook is a reduced, professionally formatted export intended for secondary research,",
            "plotting, thesis writing, and visual inspection.",
            "",
            "The file compiled_density_metrics.xlsx remains the complete technical and audit export.",
            "This workbook (compiled_density_metrics_research.xlsx) is the recommended research and reporting export.",
            "",
            "Metric hierarchy (read carefully)",
            "-----------------------------------",
            "density_metric_raw:",
            "    D_H·w_H + D_I·w_I + D_S·w_S where each D_* uses the compile weight_function (linear, log, sqrt,",
            "    d3, …) and w_* are measured component_harmonic / inharmonic / subbass energy ratios.",
            "",
            "density_weighted_sum:",
            "    Numerically equal to density_metric_raw. Changes when you change the compile weight_function.",
            "    harmonic_amplitude_sum (if present) is a separate linear diagnostic and does not follow that key.",
            "",
            "Total sum:",
            "    Unweighted sum of per-band D values (D_H + D_I + D_S); diagnostic, not energy-ratio-weighted.",
            "",
            "effective_partial_density:",
            "    Effective participation descriptor; not total spectral mass.",
            "",
            "spectral_entropy:",
            "    Distributional spread of spectral power.",
            "",
            "harmonic_energy_ratio / inharmonic_energy_ratio / subbass_energy_ratio:",
            "    Measured component energy ratios, not full psychoacoustic perceptual weights.",
            "",
            "nonharmonic / inharmonic fields:",
            "    Interpret as nonharmonic candidate material unless stricter validation is explicitly present.",
            "",
            "subbass / low-frequency fields:",
            "    Interpret as low-frequency residual material, not automatically 'ground noise'.",
            "",
            "Instrument and dynamic metadata",
            "-------------------------------",
            "Instrument and dynamic may be read from source metadata, inferred conservatively from filenames,",
            "folder paths, and the compiled workbook location, or supplied manually with --instrument and --dynamic.",
            "Use --force-metadata to override non-empty workbook values with the CLI arguments.",
            "",
            "Sheets",
            "------",
            "README — This sheet.",
            "Dashboard — Overview KPIs and charts (charts omitted if export was run with --no-charts).",
            "Spectral_Density_Metrics — Main clean per-note table for analysis.",
            "Component_Balance — How density_metric_raw and totals relate to components and energy ratios.",
            "Validation_Summary — Compact QC fields including f0 and alignment.",
            "Charts_Data — Chart source data (sorted by MIDI); Dashboard charts reference this sheet.",
            "Metadata — Pipeline and export metadata extracted from Analysis_Metadata plus export audit fields.",
            "",
            "Warnings / Missing Fields",
            "-------------------------",
        ]
    if warnings:
        for w in warnings:
            lines.append(f"- {w}")
    else:
        lines.append("(none)")
    return lines


def _autosize_columns(ws, max_width: float = 48.0) -> None:
    for col in ws.columns:
        letter = get_column_letter(col[0].column)
        maxlen = 0
        for cell in col:
            try:
                v = str(cell.value) if cell.value is not None else ""
            except Exception:
                v = ""
            maxlen = max(maxlen, min(len(v), 80))
        ws.column_dimensions[letter].width = min(max(10, maxlen + 2), max_width)


def _style_header_row(ws, row: int = 1, ncol: int | None = None) -> None:
    ncol = ncol or ws.max_column
    for c in range(1, ncol + 1):
        cell = ws.cell(row=row, column=c)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(vertical="center", wrap_text=True)


def _sanitize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure non-blank, unique column names for Excel (duplicates get ``_2``, ``_3``, …).
    """
    out = df.copy()
    new_names: List[str] = []
    seen_count: Dict[str, int] = {}
    for i, col in enumerate(out.columns):
        if col is None or (isinstance(col, float) and pd.isna(col)):
            base = f"column_{i + 1}"
        else:
            base = str(col).strip()
        if not base:
            base = f"column_{i + 1}"
        n = seen_count.get(base, 0)
        seen_count[base] = n + 1
        if n == 0:
            new_names.append(base)
        else:
            new_names.append(f"{base}_{n + 1}")
    out.columns = new_names
    return out


def apply_simple_autofilter(ws) -> None:
    """Worksheet AutoFilter only (does not create ``xl/tables/table*.xml`` Table parts)."""
    if ws.max_row >= 2 and ws.max_column >= 1:
        ws.auto_filter.ref = f"A1:{get_column_letter(ws.max_column)}{ws.max_row}"


def _apply_number_formats(ws, headers: Sequence[str], formats: Mapping[str, str]) -> None:
    if ws.max_row < 2:
        return
    hdr = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    col_index = {str(h): i + 1 for i, h in enumerate(hdr) if h is not None}
    for name, fmt in formats.items():
        if name not in col_index:
            continue
        ci = col_index[name]
        for r in range(2, ws.max_row + 1):
            ws.cell(r, ci).number_format = fmt


def _write_data_sheet(
    wb: Workbook,
    title: str,
    df: pd.DataFrame,
    ratio_cols: Tuple[str, ...],
    metric_cols: Tuple[str, ...],
) -> None:
    df = _sanitize_dataframe_columns(df)
    ws = wb.create_sheet(title)
    for row in dataframe_to_rows(df, index=False, header=True):
        ws.append(row)
    ncol = max(1, df.shape[1])
    _style_header_row(ws, 1, ncol)
    ws.freeze_panes = "A2"
    apply_simple_autofilter(ws)
    fmt_map: Dict[str, str] = {}
    for c in metric_cols:
        fmt_map[c] = "0.000000"
    for c in ratio_cols:
        fmt_map[c] = "0.00%"
    _apply_number_formats(ws, list(df.columns), fmt_map)
    _autosize_columns(ws)


def _apply_sdm_conditional(ws, headers: List[str | None]) -> None:
    def col_idx(name: str) -> int | None:
        for i, h in enumerate(headers, start=1):
            if h == name:
                return i
        return None

    last = ws.max_row
    if last < 2:
        return

    def add_rule(col_name: str, rule: Any) -> None:
        ci = col_idx(col_name)
        if not ci:
            return
        letter = get_column_letter(ci)
        ws.conditional_formatting.add(f"{letter}2:{letter}{last}", rule)

    green = PatternFill("solid", fgColor="C6EFCE")

    add_rule(
        "harmonic_energy_ratio",
        ColorScaleRule(
            start_type="num",
            start_value=0,
            start_color="F8696B",
            mid_type="num",
            mid_value=0.85,
            mid_color="FFEB84",
            end_type="num",
            end_value=1,
            end_color="63BE7B",
        ),
    )
    ci_d = col_idx("debug_counts_invariant_status")
    if ci_d:
        letter = get_column_letter(ci_d)
        ws.conditional_formatting.add(
            f"{letter}2:{letter}{last}",
            CellIsRule(operator="equal", formula=["passed"], fill=green, stopIfTrue=False),
        )
    ci_f = col_idx("f0_fit_accepted")
    if ci_f:
        letter = get_column_letter(ci_f)
        ws.conditional_formatting.add(f"{letter}2:{letter}{last}", CellIsRule(operator="equal", formula=["TRUE"], fill=green))
        ws.conditional_formatting.add(
            f"{letter}2:{letter}{last}", CellIsRule(operator="equal", formula=["FALSE"], fill=F0_FALSE_FILL)
        )


def _write_dashboard_layout(
    dash,
    source: Path,
    sd: pd.DataFrame,
    vs: pd.DataFrame,
    *,
    generated: str,
    pr: str,
    ins: str,
    dyn: str,
) -> int:
    """Two-column dashboard header + KPIs; returns suggested first row for charts."""
    widths = {1: 22.0, 2: 54.0, 3: 3.0, 4: 26.0, 5: 16.0, 6: 3.0, 7: 26.0, 8: 16.0}
    for ci, wi in widths.items():
        dash.column_dimensions[get_column_letter(ci)].width = wi

    dash.merge_cells("A1:B2")
    c0 = dash["A1"]
    c0.value = "Spectral Density Research Dashboard"
    c0.fill = TITLE_FILL
    c0.font = TITLE_FONT
    c0.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)

    dash.merge_cells("D1:H2")
    c1 = dash["D1"]
    c1.value = "Summary statistics"
    c1.font = Font(bold=True, size=14, color="FFFFFF")
    c1.fill = PatternFill("solid", fgColor="44546A")
    c1.alignment = Alignment(horizontal="center", vertical="center")

    lbl_fill = PatternFill("solid", fgColor="E7E6E6")
    r = 4
    dash.cell(r, 1, "Run context").font = SUBHEADER_FONT
    dash.merge_cells(f"D{r}:H{r}")
    dash.cell(r, 4, "Key performance indicators").font = SUBHEADER_FONT
    r += 1
    fields = [
        ("Source workbook", str(source.resolve())),
        ("Number of notes", len(sd)),
        ("Pitch range (MIDI)", pr or "n/a"),
        ("Instrument", ins or "n/a"),
        ("Dynamic", dyn or "n/a"),
        ("Generated", generated),
    ]
    if publication_clean_export_enabled():
        fields = [
            ("Compiled workbook", source.name),
            ("Number of notes", len(sd)),
            ("Pitch range (MIDI)", pr or "n/a"),
        ]
        if ins:
            fields.append(("Instrument", ins))
        if dyn:
            fields.append(("Dynamic", dyn))
        fields.append(("Generated", generated))
    for lab, val in fields:
        dash.cell(r, 1, lab).font = KPI_LABEL_FONT
        dash.cell(r, 1).fill = lbl_fill
        dash.cell(r, 1).border = KPI_BORDER
        vv = dash.cell(r, 2, val)
        vv.border = KPI_BORDER
        vv.alignment = Alignment(wrap_text=True, vertical="top")
        r += 1

    def mean_col(name: str) -> float:
        if name not in sd.columns:
            return float("nan")
        return float(pd.to_numeric(sd[name], errors="coerce").mean())

    kpis: List[Tuple[Any, Any]] = [
        ("Mean density_metric_raw", mean_col("density_metric_raw")),
        ("Mean density_weighted_sum", mean_col("density_weighted_sum")),
        ("Mean effective_partial_density", mean_col("effective_partial_density")),
        ("Mean spectral_entropy", mean_col("spectral_entropy")),
        ("Mean harmonic_energy_ratio", mean_col("harmonic_energy_ratio")),
        ("Validation passed count", int((vs["validation_summary_status"] == "passed").sum())),
        (
            "f0 accepted count",
            int(sd["f0_fit_accepted"].apply(lambda x: str(x).lower() in ("true", "1")).sum())
            if "f0_fit_accepted" in sd.columns
            else 0,
        ),
    ]
    r0 = 5
    half = (len(kpis) + 1) // 2
    for i, (label, val) in enumerate(kpis):
        if i < half:
            rr = r0 + i
            lc = 4
        else:
            rr = r0 + (i - half)
            lc = 7
        dash.cell(rr, lc, label).font = KPI_LABEL_FONT
        dash.cell(rr, lc).fill = NEUTRAL_KPI_FILL
        dash.cell(rr, lc).border = KPI_BORDER
        cell_v = dash.cell(rr, lc + 1, val)
        cell_v.fill = NEUTRAL_KPI_FILL
        cell_v.border = KPI_BORDER
        if isinstance(val, float) and not np.isnan(val):
            if "ratio" in str(label).lower():
                cell_v.number_format = "0.00%"
            elif "Mean" in str(label):
                cell_v.number_format = "0.000000"
    bottom = r0 + max(half, len(kpis) - half)
    return bottom + 3


def _dashboard_charts(wb: Workbook, ws, charts_df: pd.DataFrame, data_start_row: int) -> None:
    """Append charts below KPI block. Charts reference Charts_Data sheet."""
    cd_sheet = wb["Charts_Data"]
    n = len(charts_df)
    if n == 0:
        return
    data_end = 1 + n
    anchor_row = data_start_row

    def ref_col(col_name: str) -> int:
        headers = list(charts_df.columns)
        return headers.index(col_name) + 1

    # Line chart 1: Note vs density_weighted_sum
    chart1 = LineChart()
    chart1.title = "Register-dependent weighted spectral-mass profile"
    chart1.y_axis.title = "density_weighted_sum"
    chart1.x_axis.title = "Note"
    cats = Reference(cd_sheet, min_col=1, min_row=2, max_row=data_end)
    v1 = Reference(cd_sheet, min_col=ref_col("density_weighted_sum"), min_row=1, max_row=data_end)
    chart1.add_data(v1, titles_from_data=True)
    chart1.set_categories(cats)
    chart1.height = 8
    chart1.width = 18
    ws.add_chart(chart1, f"A{anchor_row}")
    anchor_row += 20

    chart2 = LineChart()
    chart2.title = "Algorithm-weighted spectral-density metric"
    cats = Reference(cd_sheet, min_col=1, min_row=2, max_row=data_end)
    v2 = Reference(cd_sheet, min_col=ref_col("density_metric_raw"), min_row=1, max_row=data_end)
    chart2.add_data(v2, titles_from_data=True)
    chart2.set_categories(cats)
    chart2.height = 8
    chart2.width = 18
    ws.add_chart(chart2, f"A{anchor_row}")
    anchor_row += 20

    chart3 = LineChart()
    chart3.title = "Normalized descriptor comparison"
    for col in (
        "density_metric_raw_norm_for_chart",
        "density_weighted_sum_norm_for_chart",
        "Total sum_norm_for_chart",
        "effective_partial_density_norm_for_chart",
        "spectral_entropy_norm_for_chart",
    ):
        v = Reference(cd_sheet, min_col=ref_col(col), min_row=1, max_row=data_end)
        chart3.add_data(v, titles_from_data=True)
    chart3.set_categories(Reference(cd_sheet, min_col=1, min_row=2, max_row=data_end))
    chart3.height = 9
    chart3.width = 20
    ws.add_chart(chart3, f"A{anchor_row}")
    anchor_row += 22

    chart4 = BarChart()
    chart4.type = "col"
    chart4.grouping = "stacked"
    chart4.title = "Weighted component contributions"
    chart4.overlap = 100
    for col in (
        "weighted_harmonic_density_contribution",
        "weighted_inharmonic_density_contribution",
        "weighted_subbass_density_contribution",
    ):
        v = Reference(cd_sheet, min_col=ref_col(col), min_row=1, max_row=data_end)
        chart4.add_data(v, titles_from_data=True)
    chart4.set_categories(Reference(cd_sheet, min_col=1, min_row=2, max_row=data_end))
    chart4.height = 10
    chart4.width = 18
    ws.add_chart(chart4, f"A{anchor_row}")
    anchor_row += 24

    chart5 = BarChart()
    chart5.type = "col"
    chart5.grouping = "percentStacked"
    chart5.title = "Component energy ratios"
    for col in ("harmonic_energy_ratio", "inharmonic_energy_ratio", "subbass_energy_ratio"):
        v = Reference(cd_sheet, min_col=ref_col(col), min_row=1, max_row=data_end)
        chart5.add_data(v, titles_from_data=True)
    chart5.set_categories(Reference(cd_sheet, min_col=1, min_row=2, max_row=data_end))
    chart5.height = 10
    chart5.width = 18
    ws.add_chart(chart5, f"A{anchor_row}")


def build_workbook(
    source: Path,
    output: Path,
    *,
    no_charts: bool,
    overwrite: bool,
    research_metadata: Optional[ResearchExportMetadata] = None,
) -> List[str]:
    warnings: List[str] = []
    if not source.is_file():
        raise FileNotFoundError(f"Input file does not exist: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(
            f"Output file already exists: {output}\n"
            "Pass --overwrite to replace it, or choose a different --output path."
        )

    meta = research_metadata or ResearchExportMetadata()
    merged = merge_workbook_frames(source, warnings)
    merged = _rename_frame_to_canonical(merged)
    merged = publication_research_canonical_density_columns(merged)
    sd = build_spectral_density_metrics(merged, warnings, source, meta)
    apply_per_note_chart_paths(sd, source, merged, warnings)
    if publication_clean_export_enabled():
        sd = publication_clean_drop_known_sparse_columns(sd)
        sd = drop_publication_noise_columns_from_dataframe(sd)
    cb = build_component_balance(sd, warnings)
    vs = build_validation_summary(merged, sd, warnings)
    cd = build_charts_data(sd)
    if publication_clean_export_enabled():
        cb = drop_publication_noise_columns_from_dataframe(cb)
        vs = drop_publication_noise_columns_from_dataframe(vs)
        cd = drop_publication_noise_columns_from_dataframe(cd)
    meta_map = load_analysis_metadata(source, warnings)
    meta_df = build_metadata_rows(source, meta_map, sd, warnings)
    generated = format_utc_publication_timestamp()
    pr = ""
    if sd["MIDI"].notna().any():
        pr = f"{int(sd['MIDI'].min())}–{int(sd['MIDI'].max())}"
    ins = (
        str(sd["Instrument"].dropna().iloc[0])
        if "Instrument" in sd.columns and sd["Instrument"].notna().any()
        else ""
    )
    dyn = (
        str(sd["Dynamic"].dropna().iloc[0])
        if "Dynamic" in sd.columns and sd["Dynamic"].notna().any()
        else ""
    )

    wb = Workbook()
    # README
    rm = wb.active
    rm.title = "README"
    for line in readme_lines(source, warnings, len(sd), pr, ins, dyn, generated):
        rm.append([line])
    for row in range(1, rm.max_row + 1):
        v = rm.cell(row, 1).value
        if isinstance(v, str) and v and v[0].isupper() and not v.startswith(" "):
            if v in (
                "Purpose",
                "Metric hierarchy (read carefully)",
                "Instrument and dynamic metadata",
                "Sheets",
                "Warnings / Missing Fields",
            ):
                rm.cell(row, 1).font = SUBHEADER_FONT
    rm.column_dimensions["A"].width = 108

    # Dashboard (charts added after Charts_Data exists)
    dash = wb.create_sheet("Dashboard")
    chart_anchor = _write_dashboard_layout(
        dash, source, sd, vs, generated=generated, pr=pr, ins=ins, dyn=dyn
    )

    # Data sheets order: create Charts_Data before Dashboard charts
    ratio_cols = (
        "harmonic_energy_ratio",
        "inharmonic_energy_ratio",
        "subbass_energy_ratio",
        "harmonic_alignment_coverage_ratio",
    )
    metric_cols_tuple = (
        "density_metric_raw",
        "density_weighted_sum",
        "density_log_weighted",
        "Total sum",
        "effective_partial_density",
        "spectral_entropy",
        "harmonic_density_sum",
        "inharmonic_density_sum",
        "subbass_density_sum",
        "weighted_harmonic_density_contribution",
        "weighted_inharmonic_density_contribution",
        "weighted_subbass_density_contribution",
        "harmonic_energy_sum",
        "inharmonic_energy_sum",
        "subbass_energy_sum",
        "total_component_energy",
        "f0_nominal_hz",
        "f0_final_hz",
        "f0_detuning_cents_from_nominal",
        "mean_abs_harmonic_deviation_cents",
        "max_abs_harmonic_deviation_cents",
        "canonical_density",
    )

    _write_data_sheet(wb, "Spectral_Density_Metrics", sd, ratio_cols, metric_cols_tuple)
    sdm_ws = wb["Spectral_Density_Metrics"]
    hdrs = [sdm_ws.cell(1, c).value for c in range(1, sdm_ws.max_column + 1)]
    _apply_sdm_conditional(sdm_ws, hdrs)

    cb_ratios = (
        "harmonic_energy_ratio",
        "inharmonic_energy_ratio",
        "subbass_energy_ratio",
        "energy_ratio_sum",
    )
    cb_metrics = (
        "harmonic_density_sum",
        "inharmonic_density_sum",
        "subbass_density_sum",
        "Total sum",
        "weighted_harmonic_density_contribution",
        "weighted_inharmonic_density_contribution",
        "weighted_subbass_density_contribution",
        "density_metric_raw",
        "density_metric_raw_recomputed",
        "density_metric_raw_difference",
        "total_sum_recomputed",
        "total_sum_difference",
        "harmonic_amplitude_sum",
        "inharmonic_amplitude_sum",
        "subbass_amplitude_sum",
        "density_weighted_sum",
        "density_log_weighted",
    )
    _write_data_sheet(wb, "Component_Balance", cb, cb_ratios, cb_metrics)

    vs_ratios = ("harmonic_alignment_coverage_ratio", "harmonic_alignment_energy_coverage_ratio")
    vs_metrics = (
        "f0_nominal_hz",
        "f0_final_hz",
        "f0_fit_residual_std_hz",
        "f0_detuning_cents_from_nominal",
        "mean_abs_harmonic_deviation_cents",
        "max_abs_harmonic_deviation_cents",
        "rms_harmonic_deviation_cents",
    )
    _write_data_sheet(wb, "Validation_Summary", vs, vs_ratios, vs_metrics)

    _write_data_sheet(
        wb,
        "Charts_Data",
        cd,
        ("harmonic_energy_ratio", "inharmonic_energy_ratio", "subbass_energy_ratio"),
        (
            "density_weighted_sum",
            "density_metric_raw",
            "Total sum",
            "effective_partial_density",
            "spectral_entropy",
            "density_weighted_sum_norm_for_chart",
            "density_metric_raw_norm_for_chart",
            "Total sum_norm_for_chart",
            "effective_partial_density_norm_for_chart",
            "spectral_entropy_norm_for_chart",
            "weighted_harmonic_density_contribution",
            "weighted_inharmonic_density_contribution",
            "weighted_subbass_density_contribution",
        ),
    )

    # Metadata sheet (worksheet AutoFilter only; no formal Table)
    meta_df_out = _sanitize_dataframe_columns(meta_df)
    mws = wb.create_sheet("Metadata")
    for row in dataframe_to_rows(meta_df_out, index=False, header=True):
        mws.append(row)
    _style_header_row(mws, 1, max(1, meta_df_out.shape[1]))
    mws.freeze_panes = "A2"
    apply_simple_autofilter(mws)
    _autosize_columns(mws)

    if not no_charts:
        try:
            _dashboard_charts(wb, dash, cd, chart_anchor)
        except Exception as e:  # noqa: BLE001
            warnings.append(f"Dashboard charts could not be fully created ({e}); data sheets are still valid.")
            dash.cell(chart_anchor, 1, f"Chart generation note: {e}")

    # Validation chart/table on dashboard
    last_row = dash.max_row + 3
    dash.cell(last_row, 1, "Validation snapshot (from Spectral_Density_Metrics)").font = SUBHEADER_FONT
    tbl_r = last_row + 1
    headers = ("Note", "f0_fit_accepted", "debug_counts_invariant_status", "harmonic_alignment_status")
    for i, h in enumerate(headers, start=1):
        dash.cell(tbl_r, i, h)
        dash.cell(tbl_r, i).fill = HEADER_FILL
        dash.cell(tbl_r, i).font = HEADER_FONT
    sdm = wb["Spectral_Density_Metrics"]
    col_map = {}
    for c in range(1, sdm.max_column + 1):
        val = sdm.cell(1, c).value
        if val in headers:
            col_map[str(val)] = c
    for ridx, (_, srow) in enumerate(sd.iterrows(), start=1):
        for j, h in enumerate(headers, start=1):
            ci = col_map.get(h)
            if ci:
                dash.cell(tbl_r + ridx, j, sdm.cell(ridx + 1, ci).value)

    wb.save(output)
    return warnings


def export_research_workbook(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    overwrite: bool = False,
    no_charts: bool = False,
    instrument: Optional[str] = None,
    dynamic: Optional[str] = None,
    force_metadata: bool = False,
    research_metadata: Optional[ResearchExportMetadata] = None,
) -> Path:
    """
    Build ``compiled_density_metrics_research.xlsx`` from a compiled workbook.

    Parameters
    ----------
    input_path:
        Path to ``compiled_density_metrics.xlsx``.
    output_path:
        If ``None``, writes ``compiled_density_metrics_research.xlsx`` next to the input.
    overwrite:
        If ``True``, replace an existing output file.
    no_charts:
        If ``True``, omit chart objects from the Dashboard sheet.
    instrument, dynamic, force_metadata:
        Optional CLI-style metadata (ignored if ``research_metadata`` is passed explicitly).
    research_metadata:
        If set, overrides the individual metadata keyword arguments.

    Returns
    -------
    Path
        Resolved path to the written research workbook.

    Raises
    ------
    FileNotFoundError
        If ``input_path`` is not an existing file.
    FileExistsError
        If the output exists and ``overwrite`` is ``False``.
    ValueError
        If the workbook cannot be read or merged (e.g. missing ``Density_Metrics``).
    """
    src = Path(input_path).expanduser().resolve()
    if output_path is None:
        out = src.parent / "compiled_density_metrics_research.xlsx"
    else:
        out = Path(output_path).expanduser().resolve()
    if research_metadata is not None:
        meta = research_metadata
    else:
        meta = ResearchExportMetadata(
            instrument=instrument.strip() if instrument else None,
            dynamic=dynamic.strip() if dynamic else None,
            force_metadata=force_metadata,
        )
    warns = build_workbook(
        src,
        out,
        no_charts=no_charts,
        overwrite=overwrite,
        research_metadata=meta,
    )
    for w in warns:
        print(f"WARNING: {w}", file=sys.stderr)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Export reduced research workbook from compiled_density_metrics.xlsx")
    p.add_argument("input", type=Path, help="Path to compiled_density_metrics.xlsx")
    p.add_argument("--output", type=Path, default=None, help="Output path (default: sibling compiled_density_metrics_research.xlsx)")
    p.add_argument("--no-charts", action="store_true", help="Skip chart objects on Dashboard")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    p.add_argument("--instrument", type=str, default=None, help="Override Instrument for all rows (see --force-metadata)")
    p.add_argument("--dynamic", type=str, default=None, help="Override Dynamic for all rows (see --force-metadata)")
    p.add_argument(
        "--force-metadata",
        action="store_true",
        help="When set with --instrument/--dynamic, replace non-empty workbook values too",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    src = args.input.expanduser().resolve()
    out_arg = args.output
    if out_arg is None:
        out_resolved: Path | None = None
    else:
        out_resolved = out_arg.expanduser().resolve()

    try:
        out = export_research_workbook(
            src,
            out_resolved,
            overwrite=args.overwrite,
            no_charts=args.no_charts,
            instrument=args.instrument,
            dynamic=args.dynamic,
            force_metadata=args.force_metadata,
        )
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 1
    except FileExistsError as e:
        print(str(e), file=sys.stderr)
        return 1
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    print(f"Wrote: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
