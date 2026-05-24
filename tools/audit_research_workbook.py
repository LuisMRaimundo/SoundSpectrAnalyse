from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _note_to_midi(note: Any) -> float:
    if note is None:
        return float("nan")
    s = str(note).strip()
    if not s:
        return float("nan")
    names = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5, "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10, "BB": 10, "B": 11}
    pitch = s[:-1].upper()
    try:
        octave = int(s[-1])
    except ValueError:
        return float("nan")
    if pitch not in names:
        return float("nan")
    return float((octave + 1) * 12 + names[pitch])


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    def _b(v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False
        return str(v).strip().lower() in {"true", "1", "yes"}

    return s.apply(_b)


def _choose_main_sheet(xl: pd.ExcelFile) -> str:
    for name in ("Spectral_Density_Metrics", "Density_Metrics", "Canonical_Metrics"):
        if name in xl.sheet_names:
            return name
    return xl.sheet_names[0]


def _metric_like_columns(df: pd.DataFrame) -> List[str]:
    keys = (
        "density",
        "entropy",
        "occupancy",
        "roughness",
        "dissonance",
        "harmonic_effective_power",
        "residual_energy_ratio",
    )
    cols: List[str] = []
    for c in df.columns:
        cs = str(c)
        lc = cs.lower()
        if any(k in lc for k in keys) and pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce")):
            cols.append(cs)
    return cols


def _series_stats(s: pd.Series) -> Dict[str, float]:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if x.empty:
        return {
            "count": 0,
            "min": float("nan"),
            "max": float("nan"),
            "mean": float("nan"),
            "median": float("nan"),
            "cv": float("nan"),
        }
    mean = float(x.mean())
    std = float(x.std(ddof=0))
    cv = float(std / mean) if np.isfinite(mean) and abs(mean) > 1e-12 else float("nan")
    return {
        "count": int(x.size),
        "min": float(x.min()),
        "max": float(x.max()),
        "mean": mean,
        "median": float(x.median()),
        "cv": cv,
    }


def _corr(x: pd.Series, y: pd.Series) -> Dict[str, float]:
    a = pd.to_numeric(x, errors="coerce")
    b = pd.to_numeric(y, errors="coerce")
    m = a.notna() & b.notna()
    if int(m.sum()) < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "n": int(m.sum())}
    ap = a[m].to_numpy(dtype=float)
    bp = b[m].to_numpy(dtype=float)
    pearson = float(np.corrcoef(ap, bp)[0, 1]) if ap.size >= 3 else float("nan")
    spear = float(spearmanr(ap, bp, nan_policy="omit").correlation)
    return {"pearson": pearson, "spearman": spear, "n": int(ap.size)}


def audit_workbook(path: Path) -> Dict[str, Any]:
    xl = pd.ExcelFile(path, engine="openpyxl")
    main_sheet = _choose_main_sheet(xl)
    df = pd.read_excel(path, sheet_name=main_sheet, engine="openpyxl")
    if "MIDI" not in df.columns:
        if "Note" in df.columns:
            df["MIDI"] = df["Note"].apply(_note_to_midi)
        else:
            df["MIDI"] = np.nan

    hoc = pd.to_numeric(df["harmonic_order_count"], errors="coerce") if "harmonic_order_count" in df.columns else pd.Series(np.nan, index=df.index)
    metric_cols = _metric_like_columns(df)
    metric_stats: Dict[str, Any] = {}
    for c in metric_cols:
        metric_stats[c] = {
            "stats": _series_stats(df[c]),
            "corr_with_midi": _corr(df[c], df["MIDI"]),
            "corr_with_harmonic_order_count": _corr(df[c], hoc),
        }

    f0_fit = _coerce_bool_series(df["f0_fit_accepted"]) if "f0_fit_accepted" in df.columns else pd.Series(False, index=df.index)
    af0 = df["acoustic_f0_status"].astype(str) if "acoustic_f0_status" in df.columns else pd.Series("", index=df.index)
    fallback = af0.eq("nominal_fallback_used_not_acoustically_verified")
    acoustically_verified = af0.eq("fit_accepted_acoustically_verified")

    ratio_summary = {}
    for c in ("component_harmonic_energy_ratio", "component_inharmonic_energy_ratio", "component_subbass_energy_ratio"):
        ratio_summary[c] = _series_stats(df[c]) if c in df.columns else _series_stats(pd.Series(dtype=float))

    report: Dict[str, Any] = {
        "input_workbook": str(path),
        "sheet_used": main_sheet,
        "row_count": int(len(df)),
        "density_like_metrics": metric_stats,
        "f0_fit_accepted_rows": int(f0_fit.sum()),
        "f0_fit_accepted_ratio": float(f0_fit.mean()) if len(df) else float("nan"),
        "f0_fallback_rows": int(fallback.sum()),
        "f0_fallback_ratio": float(fallback.mean()) if len(df) else float("nan"),
        "acoustically_verified_rows": int(acoustically_verified.sum()),
        "acoustically_verified_ratio": float(acoustically_verified.mean()) if len(df) else float("nan"),
        "nominal_or_fallback_only_rows": int(fallback.sum()),
        "density_weighted_sum_cdm_mean_present": bool("density_weighted_sum_cdm_mean" in df.columns),
        "combined_density_metric_present": bool("Combined Density Metric" in df.columns),
        "density_metric_raw_labelled_diagnostic": bool(
            "energy_weighted_component_density_diagnostic" in df.columns
            or "density_metric_raw" in df.columns
        ),
        "arithmetic_acoustic_validation_separated": bool(
            "arithmetic_validation_status" in df.columns and "acoustic_validation_status" in df.columns
        ),
        "energy_ratio_summary": ratio_summary,
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit workbook metrics as regression artifact")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Input workbook not found: {args.input}")
    report = audit_workbook(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
