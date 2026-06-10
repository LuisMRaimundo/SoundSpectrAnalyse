"""Stable per-row identifiers for compiled and research workbook exports."""

from __future__ import annotations

import hashlib
import re
from pathlib import PurePosixPath
from typing import Optional

import pandas as pd

DEAD_COLUMN_PROTECTED_NAMES: frozenset[str] = frozenset({"Note", "sample_id"})

__all__ = [
    "compute_sample_id",
    "assign_sample_ids",
    "attach_sample_id_from_density",
    "dedupe_identical_columns",
    "drop_dead_columns",
    "merge_keys_for_frames",
    "primary_merge_keys",
    "sample_id_fully_populated",
    "DEAD_COLUMN_PROTECTED_NAMES",
]


def _source_file_stem(source_file_name: str) -> str:
    """Basename without extension, independent of host path conventions.

    Source paths may arrive as POSIX (``/dir/sample.wav``) or Windows
    (``C:\\dir\\sample.wav``) strings regardless of the runtime OS. Normalise
    separators to ``/`` and parse with :class:`~pathlib.PurePosixPath` so the
    same logical file yields the same stem on Linux and Windows.
    """
    raw = str(source_file_name or "").strip()
    if not raw:
        return ""
    normalized = raw.replace("\\", "/").rstrip("/")
    if not normalized:
        return ""
    return PurePosixPath(normalized).stem


def compute_sample_id(
    *,
    note: str,
    source_file_name: str = "",
    row_index: int = 0,
) -> str:
    """Stable slug for one compiled/research row (handles duplicate Note keys)."""
    stem = _source_file_stem(source_file_name)
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


def sample_id_fully_populated(df: pd.DataFrame) -> bool:
    """True when every row has a non-blank ``sample_id`` (NaN / ``nan`` text counts as missing)."""
    if df is None or df.empty or "sample_id" not in df.columns:
        return False
    for v in df["sample_id"]:
        if pd.isna(v):
            return False
        s = str(v).strip().lower()
        if s in ("", "nan", "none", "<na>"):
            return False
    return True


def attach_sample_id_from_density(
    df: pd.DataFrame,
    density_df: pd.DataFrame,
) -> pd.DataFrame:
    """Copy authoritative ``sample_id`` from ``Density_Metrics`` onto satellite sheets."""
    if df is None or df.empty or density_df is None or density_df.empty:
        return df
    if sample_id_fully_populated(df):
        return df
    if "sample_id" not in density_df.columns or "Note" not in df.columns or "Note" not in density_df.columns:
        return df
    sid_map = density_df[["Note", "sample_id"]].drop_duplicates(subset=["Note"], keep="last")
    out = df.merge(sid_map, on="Note", how="left", suffixes=("", "__sid"))
    if "sample_id__sid" not in out.columns:
        return out
    if "sample_id" not in out.columns:
        out["sample_id"] = out["sample_id__sid"]
    else:
        need = out["sample_id"].isna() | out["sample_id"].astype(str).str.strip().str.lower().isin(
            ("", "nan", "none", "<na>")
        )
        out.loc[need, "sample_id"] = out.loc[need, "sample_id__sid"]
    out = out.drop(columns=["sample_id__sid"], errors="ignore")
    if "Note" in out.columns and "sample_id" in out.columns:
        cols = list(out.columns)
        cols.remove("sample_id")
        note_idx = cols.index("Note")
        cols.insert(note_idx + 1, "sample_id")
        out = out.loc[:, cols]
    return out


def drop_dead_columns(
    df: pd.DataFrame,
    *,
    protected: frozenset[str] = DEAD_COLUMN_PROTECTED_NAMES,
) -> pd.DataFrame:
    """Drop columns that are entirely NaN or blank-like text (never all-zero numerics)."""
    if df is None or df.empty or df.shape[1] == 0:
        return df
    keep_indices: list[int] = []
    for col_idx, c in enumerate(df.columns):
        cs = str(c)
        if cs in protected:
            keep_indices.append(col_idx)
            continue
        series = df.iloc[:, col_idx]
        if bool(series.isna().all()):
            continue
        if not pd.api.types.is_numeric_dtype(series):
            stripped = series.astype(str).str.strip()
            blank_like = stripped.isin(("", "nan", "None", "NaN", "<NA>"))
            if bool(blank_like.all()):
                continue
        keep_indices.append(col_idx)
    if len(keep_indices) == len(df.columns):
        return df
    return df.iloc[:, keep_indices].copy()


def merge_keys_for_frames(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    note_key: str = "Note",
) -> list[str]:
    """Pick merge keys where anchor and satellite rows actually align."""
    if left is None or left.empty:
        return primary_merge_keys(left)
    if (
        right is not None
        and not right.empty
        and "sample_id" in left.columns
        and "sample_id" in right.columns
    ):
        left_sid = left["sample_id"].astype(str).str.strip()
        right_sid = right["sample_id"].astype(str).str.strip()
        if (
            left_sid.ne("").all()
            and right_sid.ne("").all()
            and left_sid.is_unique
            and right_sid.is_unique
        ):
            overlap = set(left_sid) & set(right_sid)
            if len(overlap) >= len(left):
                return ["sample_id"]
    if right is not None and note_key in left.columns and note_key in right.columns:
        return [note_key]
    return primary_merge_keys(left)


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
        # Pure-string pairs coerce to all-NaN on BOTH sides; aligned NaNs
        # compare equal under Series.equals, which would wrongly drop a
        # suffixed text column whose content differs from the base. The
        # numeric-equivalence branch therefore only applies when the
        # coercion retains numeric information on at least one side;
        # otherwise the exact string comparison below decides.
        has_numeric_info = bool(left.notna().any() or right.notna().any())
        if has_numeric_info and (
            left.equals(right)
            or (
                left.fillna(-999999.0).equals(right.fillna(-999999.0))
                and left.isna().equals(right.isna())
            )
        ):
            drop.append(col)
            continue
        # Non-numeric exact match
        if out[base].astype(str).equals(out[col].astype(str)):
            drop.append(col)
    if drop:
        out = out.drop(columns=sorted(set(drop)), errors="ignore")
    return out
