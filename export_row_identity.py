"""Stable per-row identifiers for compiled and research workbook exports."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Optional

import pandas as pd

__all__ = [
    "compute_sample_id",
    "assign_sample_ids",
    "dedupe_identical_columns",
    "primary_merge_keys",
]


def compute_sample_id(
    *,
    note: str,
    source_file_name: str = "",
    row_index: int = 0,
) -> str:
    """Stable slug for one compiled/research row (handles duplicate Note keys)."""
    stem = Path(str(source_file_name)).stem if source_file_name else ""
    note_s = str(note or "").strip()
    key = f"{note_s}|{stem}|{int(row_index)}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    slug_base = stem or note_s or "sample"
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", slug_base).strip("._")
    if len(slug) > 80:
        slug = slug[:80].rstrip("._")
    if not slug:
        slug = "sample"
    return f"{slug}__{digest}"


def assign_sample_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``sample_id`` when missing; preserve existing values."""
    if df is None or df.empty:
        return df
    out = df.copy()
    if "sample_id" in out.columns and out["sample_id"].astype(str).str.strip().ne("").any():
        return out
    note_col = "Note" if "Note" in out.columns else None
    src_col = next(
        (c for c in ("source_file_name", "Source_File", "filename", "file_name") if c in out.columns),
        None,
    )
    ids: list[str] = []
    for i, row in out.iterrows():
        note = str(row[note_col]).strip() if note_col else ""
        src = str(row[src_col]).strip() if src_col and pd.notna(row.get(src_col)) else ""
        ids.append(compute_sample_id(note=note, source_file_name=src, row_index=int(i)))
    out["sample_id"] = ids
    return out


def primary_merge_keys(df: pd.DataFrame) -> list[str]:
    """Prefer ``sample_id``; fall back to ``Note`` only when keys are unique."""
    if df is not None and "sample_id" in df.columns:
        sid = df["sample_id"].astype(str).str.strip()
        if sid.ne("").all() and sid.is_unique:
            return ["sample_id"]
    if df is not None and "Note" in df.columns:
        notes = df["Note"].astype(str).str.strip()
        if notes.is_unique:
            return ["Note"]
    if df is not None and "sample_id" in df.columns:
        return ["sample_id"]
    return ["Note"]


def dedupe_identical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop ``col_2``-style duplicates when values match the base column exactly."""
    if df is None or df.empty:
        return df
    out = df.copy()
    drop: list[str] = []
    for col in list(out.columns):
        if not re.match(r"^(.+)_(\d+)$", str(col)):
            continue
        base = re.match(r"^(.+)_(\d+)$", str(col)).group(1)  # type: ignore[union-attr]
        if base not in out.columns:
            continue
        left = pd.to_numeric(out[base], errors="coerce")
        right = pd.to_numeric(out[col], errors="coerce")
        if left.equals(right) or (
            left.fillna(-999999.0).equals(right.fillna(-999999.0))
            and left.isna().equals(right.isna())
        ):
            drop.append(col)
            continue
        # Non-numeric exact match
        if out[base].astype(str).equals(out[col].astype(str)):
            drop.append(col)
    if drop:
        out = out.drop(columns=sorted(set(drop)), errors="ignore")
    return out
