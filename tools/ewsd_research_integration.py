#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 3 — EWSD integration for ``compiled_density_metrics_research.xlsx``.

Recomputes EWSD-R v18 from per-note ``spectral_analysis.xlsx`` workbooks under the
analysis folder and left-joins scores into ``Spectral_Density_Metrics``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd

from tools.ewsd_core import (
    ACOUSTIC_BALANCE_ALPHA_DEFAULT,
    INDIVIDUAL_SHEETS,
    SCRIPT_VERSION,
    add_acoustic_alignment_columns,
    add_quality_columns,
    apply_relative_db_threshold,
    compute_ewsd,
    list_excel_sheets,
    read_individual_workbook,
    sort_chromatically_by_octave,
)
from tools.ewsd_stage3_contract import (
    STAGE3_STATUS_DEGRADED,
    STAGE3_STATUS_FAILED,
    STAGE3_STATUS_OK,
    Stage3MergeResult,
    assess_stage3_merge_result,
    build_stage3_diagnostics,
    enforce_fail_closed,
)
from tools.ewsd_uncertainty import CompartmentBootstrapData, bootstrap_ewsd_from_compartments

__all__ = (
    "EWSD_RESEARCH_SCORE_COLUMNS",
    "EWSD_RESEARCH_PROVENANCE_COLUMNS",
    "EWSD_RESEARCH_UNCERTAINTY_COLUMNS",
    "EWSD_RESEARCH_ALL_COLUMNS",
    "Stage3MergeResult",
    "compute_ewsd_dataframe_from_analysis_root",
    "discover_individual_exact_workbooks",
    "merge_ewsd_into_spectral_density_metrics",
    "merge_ewsd_stage3",
)

EWSD_RESEARCH_SCORE_COLUMNS: tuple[str, ...] = (
    "EWSD_score_total",
    "EWSD_score_acoustic_balanced",
)

EWSD_RESEARCH_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "ewsd_mode",
    "ewsd_primary_analysis_eligible",
    "ewsd_his_ratio_source",
    "ewsd_H_ratio",
    "ewsd_I_ratio",
    "ewsd_S_noise_ratio",
    "ewsd_weight_function_canonical",
    "ewsd_acoustic_balance_alpha",
    "ewsd_stage3_version",
    "ewsd_merge_status",
)

EWSD_RESEARCH_UNCERTAINTY_COLUMNS: tuple[str, ...] = (
    "EWSD_score_total_ci_low",
    "EWSD_score_total_ci_high",
    "EWSD_score_total_rel_uncertainty",
    "EWSD_score_acoustic_balanced_ci_low",
    "EWSD_score_acoustic_balanced_ci_high",
    "EWSD_score_acoustic_balanced_rel_uncertainty",
    "ewsd_uncertainty_sources",
)

EWSD_RESEARCH_ALL_COLUMNS: tuple[str, ...] = (
    EWSD_RESEARCH_SCORE_COLUMNS
    + EWSD_RESEARCH_PROVENANCE_COLUMNS
    + EWSD_RESEARCH_UNCERTAINTY_COLUMNS
)

_SKIP_XLSX_NAMES = frozenset(
    {
        "compiled_density_metrics.xlsx",
        "compiled_density_metrics_research.xlsx",
        "ewsd_ratio_respecting_results.xlsx",
    }
)


def discover_individual_exact_workbooks(analysis_root: Path) -> list[Path]:
    """Return per-note workbooks that contain the full H/I component sheets."""
    root = analysis_root.expanduser().resolve()
    if not root.is_dir():
        return []

    found: list[Path] = []
    seen: set[Path] = set()

    def _maybe_add(path: Path) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        sheets = set(list_excel_sheets(resolved))
        if INDIVIDUAL_SHEETS.issubset(sheets):
            seen.add(resolved)
            found.append(resolved)

    for candidate in root.rglob("spectral_analysis.xlsx"):
        if candidate.name.lower() not in _SKIP_XLSX_NAMES:
            _maybe_add(candidate)

    if not found:
        for candidate in root.rglob("*.xlsx"):
            if candidate.name.lower() in _SKIP_XLSX_NAMES:
                continue
            _maybe_add(candidate)

    return sorted(found, key=lambda p: str(p).lower())


def _first_finite_from_frame(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[float]:
    for col in candidates:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        v = s.dropna()
        if not v.empty and np.isfinite(float(v.iloc[0])):
            return float(v.iloc[0])
    return None


def _component_family_mask(component_types, family: str):
    t = component_types.astype(str)
    if family == "harmonic":
        return t.eq("harmonic")
    if family == "nonharmonic_residual":
        return t.eq("nonharmonic_residual")
    if family == "noise_subbass":
        return t.str.startswith("subbass") | t.eq("noise") | t.eq("subnoise")
    import pandas as pd

    return pd.Series(False, index=component_types.index)


def _compartment_bootstrap_data_from_cset(
    cset,
    threshold_db_relative: Optional[float],
) -> list[CompartmentBootstrapData]:
    """Extract H/I/S salient partial arrays for EWSD bootstrap."""
    components = apply_relative_db_threshold(cset.components.copy(), threshold_db_relative)
    families = [
        ("harmonic", cset.his_weights.harmonic),
        ("nonharmonic_residual", cset.his_weights.nonharmonic_residual),
        ("noise_subbass", cset.his_weights.noise_subbass),
    ]
    out: list[CompartmentBootstrapData] = []
    for family_name, ratio in families:
        fam_mask = _component_family_mask(components["component_type"], family_name)
        family_df = components.loc[fam_mask]
        amps = (
            family_df["basis_value"].to_numpy(dtype=float)
            if not family_df.empty and "basis_value" in family_df.columns
            else np.array([], dtype=float)
        )
        freqs = (
            family_df["frequency_hz"].to_numpy(dtype=float)
            if not family_df.empty and "frequency_hz" in family_df.columns
            else None
        )
        out.append(
            CompartmentBootstrapData(
                amplitudes=amps,
                analysis_ratio=float(ratio),
                frequencies_hz=freqs,
                weight_function=cset.weight_function,
                apply_anti_concentration=True,
            )
        )
    return out


def compute_ewsd_dataframe_from_analysis_root(
    analysis_root: Path,
    *,
    ratio_source: str = "auto_excel_required",
    use_excel_weight_function: bool = True,
    weight_function: str = "auto_from_excel",
    basis: str = "amplitude",
    frequency_ceiling_hz: Optional[float] = None,
    threshold_db_relative: Optional[float] = None,
    aggregate_subbass: bool = True,
    apply_anti_concentration: bool = True,
    acoustic_balance_alpha: float = ACOUSTIC_BALANCE_ALPHA_DEFAULT,
    include_uncertainty: bool = True,
    bootstrap_n: int = 800,
    bootstrap_ci: float = 0.95,
    bootstrap_seed: int = 0,
) -> pd.DataFrame:
    """Run EWSD-R v18 on all individual-exact workbooks under ``analysis_root``."""
    workbooks = discover_individual_exact_workbooks(analysis_root)
    rows: list[dict[str, Any]] = []
    for wb_path in workbooks:
        cset = read_individual_workbook(
            wb_path,
            requested_weight_function=weight_function,
            use_excel_weight_function=use_excel_weight_function,
            ratio_source=ratio_source,
            manual_h=None,
            manual_i=None,
            manual_s=None,
            basis=basis,
            frequency_ceiling_hz=frequency_ceiling_hz,
            aggregate_subbass=aggregate_subbass,
        )
        if cset is not None:
            row = compute_ewsd(cset, threshold_db_relative, apply_anti_concentration)
            if include_uncertainty and cset.his_weights.is_valid():
                try:
                    boot = bootstrap_ewsd_from_compartments(
                        _compartment_bootstrap_data_from_cset(cset, threshold_db_relative),
                        acoustic_balance_alpha=acoustic_balance_alpha,
                        n_boot=bootstrap_n,
                        ci=bootstrap_ci,
                        seed=bootstrap_seed + len(rows),
                        propagate_ratio_uncertainty=True,
                    )
                    row.update(
                        {
                            "EWSD_score_total_ci_low": boot["ewsd_score_total_ci_low"],
                            "EWSD_score_total_ci_high": boot["ewsd_score_total_ci_high"],
                            "EWSD_score_total_rel_uncertainty": boot["ewsd_score_total_rel_uncertainty"],
                            "EWSD_score_acoustic_balanced_ci_low": boot[
                                "ewsd_score_acoustic_balanced_ci_low"
                            ],
                            "EWSD_score_acoustic_balanced_ci_high": boot[
                                "ewsd_score_acoustic_balanced_ci_high"
                            ],
                            "EWSD_score_acoustic_balanced_rel_uncertainty": boot[
                                "ewsd_score_acoustic_balanced_rel_uncertainty"
                            ],
                            "ewsd_uncertainty_sources": boot["uncertainty_sources"],
                        }
                    )
                except Exception:
                    row.update(
                        {
                            "EWSD_score_total_ci_low": np.nan,
                            "EWSD_score_total_ci_high": np.nan,
                            "EWSD_score_total_rel_uncertainty": np.nan,
                            "EWSD_score_acoustic_balanced_ci_low": np.nan,
                            "EWSD_score_acoustic_balanced_ci_high": np.nan,
                            "EWSD_score_acoustic_balanced_rel_uncertainty": np.nan,
                            "ewsd_uncertainty_sources": "unavailable",
                        }
                    )
            rows.append(row)

    if not rows:
        return pd.DataFrame()

    result = sort_chromatically_by_octave(pd.DataFrame(rows))
    result = add_acoustic_alignment_columns(
        result,
        frequency_ceiling_hz,
        acoustic_balance_alpha=acoustic_balance_alpha,
    )
    result = add_quality_columns(result)
    if "ewsd_score" in result.columns and "EWSD_score_total" not in result.columns:
        result["EWSD_score_total"] = pd.to_numeric(result["ewsd_score"], errors="coerce")
    return result


def _dedupe_ewsd_by_note(ewsd: pd.DataFrame) -> pd.DataFrame:
    if ewsd.empty or "Note" not in ewsd.columns:
        return ewsd
    out = ewsd.copy()
    out["Note"] = out["Note"].astype(str).str.strip()
    out = out[out["Note"].ne("") & out["Note"].str.lower().ne("nan")]

    mode = out.get("mode", pd.Series("", index=out.index)).astype(str)
    eligible = out.get("primary_analysis_eligible", pd.Series(False, index=out.index)).astype(bool)
    quality = pd.to_numeric(out.get("row_quality_score_0_100"), errors="coerce").fillna(-1.0)

    out["_ewsd_pick_priority"] = (
        eligible.astype(int) * 1000
        + mode.eq("individual_exact").astype(int) * 100
        + quality
    )
    out = out.sort_values(["Note", "_ewsd_pick_priority"], ascending=[True, False], kind="mergesort")
    out = out.drop_duplicates(subset=["Note"], keep="first")
    return out.drop(columns=["_ewsd_pick_priority"], errors="ignore")


def _prepare_ewsd_merge_frame(ewsd: pd.DataFrame) -> pd.DataFrame:
    if ewsd.empty:
        return pd.DataFrame(columns=["Note", *EWSD_RESEARCH_ALL_COLUMNS])

    frame = _dedupe_ewsd_by_note(ewsd)
    merge = pd.DataFrame({"Note": frame["Note"].astype(str).str.strip()})
    merge["EWSD_score_total"] = pd.to_numeric(frame.get("ewsd_score"), errors="coerce")
    merge["EWSD_score_acoustic_balanced"] = pd.to_numeric(
        frame.get("ewsd_score_acoustic_balanced"), errors="coerce"
    )
    merge["ewsd_mode"] = frame.get("mode", pd.Series(np.nan, index=frame.index)).astype(str)
    merge["ewsd_primary_analysis_eligible"] = frame.get(
        "primary_analysis_eligible", pd.Series(False, index=frame.index)
    ).astype(bool)
    merge["ewsd_his_ratio_source"] = frame.get("his_ratio_source", np.nan)
    merge["ewsd_H_ratio"] = pd.to_numeric(frame.get("analysis_ratio_weight_harmonic"), errors="coerce")
    merge["ewsd_I_ratio"] = pd.to_numeric(
        frame.get("analysis_ratio_weight_nonharmonic_residual"), errors="coerce"
    )
    merge["ewsd_S_noise_ratio"] = pd.to_numeric(
        frame.get("analysis_ratio_weight_noise_subbass"), errors="coerce"
    )
    merge["ewsd_weight_function_canonical"] = frame.get("weight_function_canonical", np.nan)
    merge["ewsd_acoustic_balance_alpha"] = pd.to_numeric(
        frame.get("ewsd_acoustic_balance_alpha"), errors="coerce"
    ).fillna(ACOUSTIC_BALANCE_ALPHA_DEFAULT)
    merge["ewsd_stage3_version"] = SCRIPT_VERSION
    merge["EWSD_score_total_ci_low"] = pd.to_numeric(
        frame.get("EWSD_score_total_ci_low"), errors="coerce"
    )
    merge["EWSD_score_total_ci_high"] = pd.to_numeric(
        frame.get("EWSD_score_total_ci_high"), errors="coerce"
    )
    merge["EWSD_score_total_rel_uncertainty"] = pd.to_numeric(
        frame.get("EWSD_score_total_rel_uncertainty"), errors="coerce"
    )
    merge["EWSD_score_acoustic_balanced_ci_low"] = pd.to_numeric(
        frame.get("EWSD_score_acoustic_balanced_ci_low"), errors="coerce"
    )
    merge["EWSD_score_acoustic_balanced_ci_high"] = pd.to_numeric(
        frame.get("EWSD_score_acoustic_balanced_ci_high"), errors="coerce"
    )
    merge["EWSD_score_acoustic_balanced_rel_uncertainty"] = pd.to_numeric(
        frame.get("EWSD_score_acoustic_balanced_rel_uncertainty"), errors="coerce"
    )
    merge["ewsd_uncertainty_sources"] = frame.get("ewsd_uncertainty_sources", np.nan)
    merge["ewsd_merge_status"] = np.where(
        merge["ewsd_mode"].eq("individual_exact"),
        "merged_individual_exact",
        "merged_non_exact_mode",
    )
    return merge


def _init_empty_ewsd_columns(sd: pd.DataFrame, status: str) -> pd.DataFrame:
    out = sd.copy()
    for col in EWSD_RESEARCH_ALL_COLUMNS:
        if col not in out.columns:
            if col == "ewsd_primary_analysis_eligible":
                out[col] = False
            elif col == "ewsd_stage3_version":
                out[col] = SCRIPT_VERSION
            elif col == "ewsd_acoustic_balance_alpha":
                out[col] = ACOUSTIC_BALANCE_ALPHA_DEFAULT
            elif col == "ewsd_uncertainty_sources":
                out[col] = "unavailable"
            else:
                out[col] = np.nan
    out["ewsd_merge_status"] = status
    return out


def merge_ewsd_stage3(
    sd: pd.DataFrame,
    merged: pd.DataFrame,
    compiled_workbook: Path,
    warnings: List[str],
    *,
    include_ewsd: bool = True,
    acoustic_balance_alpha: float = ACOUSTIC_BALANCE_ALPHA_DEFAULT,
    include_uncertainty: bool = True,
    bootstrap_n: int = 800,
    fail_closed: bool = False,
) -> Stage3MergeResult:
    """Left-join EWSD into ``Spectral_Density_Metrics`` with diagnostics."""
    if not include_ewsd or sd is None or sd.empty or "Note" not in sd.columns:
        empty = sd if sd is not None else pd.DataFrame()
        diag, diag_summary = build_stage3_diagnostics(
            empty, analysis_root="", frequency_ceiling_hz=None, n_workbooks=0
        )
        return Stage3MergeResult(empty, diag, diag_summary, STAGE3_STATUS_OK, tuple(warnings))

    analysis_root = Path(compiled_workbook).expanduser().resolve().parent
    freq_ceiling = _first_finite_from_frame(
        sd if "density_frequency_ceiling_hz" in sd.columns else merged,
        ("density_frequency_ceiling_hz",),
    )
    if freq_ceiling is None:
        freq_ceiling = _first_finite_from_frame(
            merged,
            ("density_frequency_ceiling_hz", "density_frequency_ceiling"),
        )

    global_status = "merged"
    stage_messages: list[str] = list(warnings)

    try:
        ewsd_raw = compute_ewsd_dataframe_from_analysis_root(
            analysis_root,
            frequency_ceiling_hz=freq_ceiling,
            acoustic_balance_alpha=acoustic_balance_alpha,
            include_uncertainty=include_uncertainty,
            bootstrap_n=bootstrap_n,
        )
    except (OSError, ValueError, KeyError, pd.errors.ParserError) as exc:
        stage_messages.append(f"Stage 3 EWSD computation failed: {exc}")
        out = _init_empty_ewsd_columns(sd, "ewsd_computation_failed")
        global_status = "ewsd_computation_failed"
    else:
        if ewsd_raw.empty:
            stage_messages.append(
                "Stage 3 EWSD skipped: no per-note workbooks with Harmonic Spectrum + "
                "Inharmonic Spectrum found under analysis folder."
            )
            out = _init_empty_ewsd_columns(sd, "no_per_note_workbooks_found")
            global_status = "no_per_note_workbooks_found"
        else:
            ewsd_merge = _prepare_ewsd_merge_frame(ewsd_raw)
            out = sd.copy()
            out["Note"] = out["Note"].astype(str).str.strip()

            for col in EWSD_RESEARCH_ALL_COLUMNS:
                if col in out.columns:
                    out = out.drop(columns=[col])

            out = out.merge(ewsd_merge, on="Note", how="left", validate="m:1")
            missing = out["EWSD_score_total"].isna()
            out.loc[missing, "ewsd_merge_status"] = "note_not_in_ewsd_output"

            n_merged = int((~missing).sum())
            n_eligible = int(out.get("ewsd_primary_analysis_eligible", pd.Series(False)).astype(bool).sum())
            n_workbooks = len(discover_individual_exact_workbooks(analysis_root))
            stage_messages.append(
                f"Stage 3 EWSD merged {n_merged}/{len(out)} notes from {n_workbooks} "
                f"per-note workbook(s); {n_eligible} row(s) ewsd_primary_analysis_eligible=True."
            )

    n_workbooks = len(discover_individual_exact_workbooks(analysis_root))
    diagnostics, diagnostics_summary = build_stage3_diagnostics(
        out,
        analysis_root=str(analysis_root),
        frequency_ceiling_hz=freq_ceiling,
        n_workbooks=n_workbooks,
    )
    status = assess_stage3_merge_result(
        out,
        include_ewsd=True,
        global_status=global_status,
        messages=stage_messages,
    )
    result = Stage3MergeResult(out, diagnostics, diagnostics_summary, status, tuple(stage_messages))
    if fail_closed:
        enforce_fail_closed(result)
    return result


def merge_ewsd_into_spectral_density_metrics(
    sd: pd.DataFrame,
    merged: pd.DataFrame,
    compiled_workbook: Path,
    warnings: List[str],
    *,
    include_ewsd: bool = True,
    acoustic_balance_alpha: float = ACOUSTIC_BALANCE_ALPHA_DEFAULT,
    include_uncertainty: bool = True,
    bootstrap_n: int = 800,
    fail_closed: bool = False,
) -> pd.DataFrame:
    """Backward-compatible wrapper returning only ``Spectral_Density_Metrics``."""
    result = merge_ewsd_stage3(
        sd,
        merged,
        compiled_workbook,
        warnings,
        include_ewsd=include_ewsd,
        acoustic_balance_alpha=acoustic_balance_alpha,
        include_uncertainty=include_uncertainty,
        bootstrap_n=bootstrap_n,
        fail_closed=fail_closed,
    )
    warnings[:] = list(result.messages)
    return result.spectral_density_metrics
