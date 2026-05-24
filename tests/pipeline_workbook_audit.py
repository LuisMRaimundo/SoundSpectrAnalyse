# -*- coding: utf-8 -*-
"""
Compiled workbook invariant checks (Stage 2 output).

Used by ``tests/test_final_pipeline_invariants.py`` and ``tools/audit_compiled_workbook.py``.
Does not import ``compile_metrics`` or ``proc_audio`` — only reads Excel artefacts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

RTOL = 1e-5
ATOL_HZ = 0.02
ATOL_PCT_POINTS = 0.05  # percent points on effective_subfundamental_margin_percent

PRIOR_FIT = "prior_constrained_harmonic_fit"


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _to_bool(x: Any) -> Optional[bool]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        if int(x) == 1:
            return True
        if int(x) == 0:
            return False
    s = str(x).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no", ""):
        return False
    return None


def _to_str(x: Any) -> str:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return ""
    return str(x).strip()


def _is_na_string_metric(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    s = str(x).strip().lower()
    return s in (
        "",
        "nan",
        "none",
        "not_available_at_compile_stage",
        "not_computed",
        "not_applicable",
        "skipped",
    )


def load_compiled_workbook(path: Path) -> Dict[str, pd.DataFrame]:
    path = Path(path)
    xl = pd.ExcelFile(path)
    return {name: pd.read_excel(path, sheet_name=name) for name in xl.sheet_names}


def analysis_metadata_as_dict(am: pd.DataFrame) -> Dict[str, Any]:
    """Compiled ``Analysis_Metadata`` is one wide row."""
    if am is None or am.empty:
        return {}
    row = am.iloc[0]
    return {str(k): row[k] for k in am.columns}


def _diag_by_note(diag: pd.DataFrame) -> Dict[str, pd.Series]:
    if diag is None or diag.empty or "Note" not in diag.columns:
        return {}
    out: Dict[str, pd.Series] = {}
    for _, r in diag.iterrows():
        out[str(r["Note"])] = r
    return out


def audit_f0_provenance(canonical: pd.DataFrame, diagnostic: pd.DataFrame) -> List[str]:
    failures: List[str] = []
    dmap = _diag_by_note(diagnostic)
    if canonical is None or canonical.empty or "Note" not in canonical.columns:
        return ["blocker:Canonical_Metrics missing or has no Note column"]
    for _, row in canonical.iterrows():
        note = str(row.get("Note", ""))
        dr = dmap.get(note)
        src = _to_str(row.get("f0_source", ""))
        if not src and dr is not None:
            src = _to_str(dr.get("f0_source", ""))
        ffinal = _to_str(row.get("f0_final_source", ""))
        if not ffinal and dr is not None:
            ffinal = _to_str(dr.get("f0_final_source", ""))
        # Compiled workbooks may omit ``f0_final_source``; diagnostic ``f0_source`` is authoritative.
        effective_final = ffinal or src
        acc = _to_bool(row.get("f0_fit_accepted"))
        if acc is None and dr is not None:
            acc = _to_bool(dr.get("f0_fit_accepted"))

        if not src and not ffinal and acc is None:
            continue
        if src == PRIOR_FIT and acc is not True:
            failures.append(f"blocker:f0_source prior fit but f0_fit_accepted not True (Note={note!r})")
        if ffinal and ffinal == PRIOR_FIT and acc is not True:
            failures.append(f"blocker:f0_final_source prior fit but f0_fit_accepted not True (Note={note!r})")
        if acc is False:
            if src == PRIOR_FIT:
                failures.append(f"blocker:f0_fit_accepted False but f0_source is prior fit (Note={note!r})")
            if ffinal == PRIOR_FIT:
                failures.append(f"blocker:f0_fit_accepted False but f0_final_source is prior fit (Note={note!r})")
        if acc is False:
            f0f = row.get("f0_final_hz")
            f0n = row.get("f0_nominal_hz")
            af0 = _to_str(row.get("acoustic_f0_status", ""))
            if not af0 and dr is not None:
                af0 = _to_str(dr.get("acoustic_f0_status", ""))
            if dr is not None:
                if f0n is None or (isinstance(f0n, float) and pd.isna(f0n)):
                    f0n = dr.get("f0_nominal_hz")
            if _finite(f0f) and _finite(f0n):
                if abs(float(f0f) - float(f0n)) > max(ATOL_HZ, RTOL * abs(float(f0n))):
                    exc = effective_final.lower()
                    if "filename_note_nominal_fallback" not in exc and "nominal" not in exc:
                        failures.append(
                            f"blocker:rejected fit but f0_final_hz != f0_nominal_hz without nominal fallback "
                            f"(Note={note!r} f0_final={f0f!r} f0_nominal={f0n!r} effective_final_source={effective_final!r})"
                        )
            if af0 and af0 != "nominal_fallback_used_not_acoustically_verified":
                failures.append(
                    f"blocker:f0_fit_accepted False but acoustic_f0_status is {af0!r} "
                    f"(expected nominal_fallback_used_not_acoustically_verified) (Note={note!r})"
                )
    return failures


def audit_subfundamental(canonical: pd.DataFrame) -> List[str]:
    failures: List[str] = []
    if canonical is None or canonical.empty:
        return failures
    need = (
        "f0_final_hz",
        "adaptive_subfundamental_cutoff_hz",
        "subfundamental_margin_percent",
        "percentage_subfundamental_cutoff_hz",
        "effective_subfundamental_margin_percent",
        "subfundamental_cutoff_selected_by",
        "subfundamental_cutoff_selection_rule",
        "min_floor_hz",
        "max_fraction_of_f0",
    )
    for c in need:
        if c not in canonical.columns:
            failures.append(f"blocker:Canonical_Metrics missing column {c!r}")
            return failures

    for _, row in canonical.iterrows():
        note = str(row.get("Note", ""))
        f0 = row.get("f0_final_hz")
        if not _finite(f0) or float(f0) <= 0:
            continue
        f0f = float(f0)
        ad = row.get("adaptive_subfundamental_cutoff_hz")
        ads = _to_str(ad)
        if _is_na_string_metric(ad) or "not_available" in ads.lower():
            failures.append(f"blocker:invalid adaptive_subfundamental_cutoff for Note={note!r} ({ad!r})")
            continue
        if not _finite(ad) or float(ad) <= 0:
            failures.append(f"blocker:adaptive_subfundamental_cutoff_hz not positive (Note={note!r})")
            continue
        adf = float(ad)

        margin = row.get("subfundamental_margin_percent")
        if not _finite(margin):
            failures.append(f"blocker:subfundamental_margin_percent not numeric (Note={note!r})")
            continue
        pct = row.get("percentage_subfundamental_cutoff_hz")
        if not _finite(pct):
            failures.append(f"blocker:percentage_subfundamental_cutoff_hz not numeric (Note={note!r})")
            continue
        pct_expected = f0f * (1.0 - float(margin) / 100.0)
        if abs(float(pct) - pct_expected) > max(ATOL_HZ, RTOL * max(abs(pct_expected), 1.0)):
            failures.append(
                f"blocker:percentage_subfundamental_cutoff mismatch Note={note!r} "
                f"got {pct} expected ~{pct_expected}"
            )

        eff = row.get("effective_subfundamental_margin_percent")
        eff_exp = 100.0 * (1.0 - adf / f0f)
        if not _finite(eff) or abs(float(eff) - eff_exp) > max(ATOL_PCT_POINTS, RTOL * max(abs(eff_exp), 1.0)):
            failures.append(
                f"blocker:effective_subfundamental_margin_percent mismatch Note={note!r} "
                f"got {eff} expected ~{eff_exp}"
            )

        sel = _to_str(row.get("subfundamental_cutoff_selected_by"))
        if not sel:
            failures.append(f"blocker:subfundamental_cutoff_selected_by empty (Note={note!r})")
        rule = _to_str(row.get("subfundamental_cutoff_selection_rule"))
        if not rule:
            failures.append(f"blocker:subfundamental_cutoff_selection_rule empty (Note={note!r})")

        leak = row.get("leakage_guard_cutoff_hz")
        if sel == "leakage_guard_cutoff_hz":
            if not _finite(leak):
                failures.append(f"blocker:leakage selected but leakage_guard_cutoff_hz not numeric (Note={note!r})")
            elif abs(adf - float(leak)) > max(ATOL_HZ, RTOL * abs(float(leak))):
                failures.append(
                    f"blocker:leakage_guard selected but adaptive != leakage (Note={note!r}) "
                    f"ad={adf} leak={leak}"
                )
        elif sel == "percentage_subfundamental_cutoff_hz":
            if abs(adf - float(pct)) > max(ATOL_HZ, RTOL * abs(float(pct))):
                failures.append(
                    f"blocker:percentage selected but adaptive != percentage line (Note={note!r}) "
                    f"ad={adf} pct={pct}"
                )
        elif sel == "min_floor_hz":
            floor_v = row.get("min_floor_hz")
            if not _finite(floor_v):
                failures.append(f"blocker:min_floor selected but min_floor_hz not numeric (Note={note!r})")
            elif abs(adf - float(floor_v)) > ATOL_HZ:
                failures.append(
                    f"blocker:min_floor selected but adaptive != min_floor_hz (Note={note!r}) "
                    f"ad={adf} floor={floor_v}"
                )
        elif sel == "max_fraction_of_f0_cap":
            mfrac = row.get("max_fraction_of_f0")
            if not _finite(mfrac):
                failures.append(f"blocker:cap selected but max_fraction_of_f0 not numeric (Note={note!r})")
            else:
                cap = f0f * float(mfrac)
                if abs(adf - cap) > max(ATOL_HZ, RTOL * abs(cap)):
                    failures.append(
                        f"blocker:cap selected but adaptive != f0*max_fraction (Note={note!r}) "
                        f"ad={adf} cap={cap}"
                    )

    return failures


def audit_debug_counts(debug_df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Returns (hard_failures, warnings)."""
    hard: List[str] = []
    warn: List[str] = []
    if debug_df is None or debug_df.empty:
        warn.append("warning:Debug_Counts sheet missing or empty")
        return hard, warn
    if "nonharmonic_peak_candidate_count" in debug_df.columns:
        if "legacy_nonharmonic_peak_candidate_count_deprecated" not in debug_df.columns:
            hard.append(
                "blocker:ambiguous column nonharmonic_peak_candidate_count present without "
                "legacy_nonharmonic_peak_candidate_count_deprecated"
            )
    need = (
        "residual_spectral_row_count",
        "nonharmonic_candidate_row_count",
        "retained_nonharmonic_peak_candidate_count",
        "exported_nonharmonic_peak_candidate_count",
        "debug_counts_invariant_status",
    )
    for c in need:
        if c not in debug_df.columns:
            hard.append(f"blocker:Debug_Counts missing {c!r}")
            return hard, warn

    for _, row in debug_df.iterrows():
        note = str(row.get("Note", ""))
        rs = pd.to_numeric(row.get("residual_spectral_row_count"), errors="coerce")
        nc = pd.to_numeric(row.get("nonharmonic_candidate_row_count"), errors="coerce")
        rt = pd.to_numeric(row.get("retained_nonharmonic_peak_candidate_count"), errors="coerce")
        ex = pd.to_numeric(row.get("exported_nonharmonic_peak_candidate_count"), errors="coerce")
        if pd.notna(rs) and pd.notna(nc) and float(nc) > float(rs) + 1e-9:
            hard.append(f"blocker:Debug_Counts candidate>residual Note={note!r}")
        if pd.notna(nc) and pd.notna(rt) and float(rt) > float(nc) + 1e-9:
            hard.append(f"blocker:Debug_Counts retained>candidate Note={note!r}")
        if pd.notna(rt) and pd.notna(ex) and int(rt) != int(ex):
            hard.append(f"blocker:Debug_Counts retained!=exported Note={note!r} rt={rt} ex={ex}")
        st = _to_str(row.get("debug_counts_invariant_status")).lower()
        if st and st != "passed":
            hard.append(f"blocker:debug_counts_invariant_status={row.get('debug_counts_invariant_status')!r} Note={note!r}")
        if st == "passed":
            fail = _to_str(row.get("debug_counts_invariant_failures"))
            if fail:
                hard.append(f"blocker:debug_counts_invariant_failures non-empty but passed Note={note!r}")
    return hard, warn


def audit_ambiguous_column_names(sheets: Mapping[str, pd.DataFrame]) -> List[str]:
    hard: List[str] = []
    for name, df in sheets.items():
        if df is None or df.empty:
            continue
        if "nonharmonic_peak_candidate_count" in df.columns:
            if "legacy_nonharmonic_peak_candidate_count_deprecated" not in df.columns:
                hard.append(
                    f"blocker:sheet {name!r} exposes ambiguous nonharmonic_peak_candidate_count "
                    f"without legacy_nonharmonic_peak_candidate_count_deprecated"
                )
    return hard


def audit_missing_value_policy(
    density_df: pd.DataFrame,
    diagnostic_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[str], List[str]]:
    hard: List[str] = []
    warn: List[str] = []
    if density_df is None or density_df.empty:
        warn.append("warning:Density_Metrics missing or empty")
    elif "Harmonic Count" in density_df.columns and "N_harm_norm" in density_df.columns:
        hc = pd.to_numeric(density_df["Harmonic Count"], errors="coerce")
        nh = pd.to_numeric(density_df["N_harm_norm"], errors="coerce")
        bad = hc.isna() & nh.notna()
        if bad.any():
            hard.append("blocker:N_harm_norm finite where Harmonic Count is NaN (silent zeroing risk)")

    if diagnostic_df is not None and not diagnostic_df.empty:
        if "Harmonic Count" in diagnostic_df.columns and "N_harm_norm" in diagnostic_df.columns:
            hc = pd.to_numeric(diagnostic_df["Harmonic Count"], errors="coerce")
            nh = pd.to_numeric(diagnostic_df["N_harm_norm"], errors="coerce")
            bad = hc.isna() & nh.notna()
            if bad.any():
                hard.append("blocker:Diagnostic N_harm_norm finite where Harmonic Count is NaN")
        elif "N_harm_norm" in diagnostic_df.columns:
            for _, r in diagnostic_df.iterrows():
                hca = _to_bool(r.get("harmonic_count_available"))
                nh_val = pd.to_numeric(r.get("N_harm_norm"), errors="coerce")
                if hca is False and pd.notna(nh_val) and float(nh_val) == 0.0:
                    hard.append(
                        "blocker:N_harm_norm is 0.0 while harmonic_count_available is False "
                        "(expected NaN, not numeric zero)"
                    )
                    break
        if "Index_Weighted_status" in diagnostic_df.columns:
            for _, r in diagnostic_df.iterrows():
                st = _to_str(r.get("Index_Weighted_status")).lower()
                if not st:
                    warn.append("warning:Index_Weighted_status empty on a diagnostic row")
                elif "zero_fill" in st or "missing_as_zero" in st:
                    hard.append(f"blocker:Index_Weighted_status suggests zero-fill: {r.get('Index_Weighted_status')!r}")

    frames = [df for df in (density_df, diagnostic_df) if df is not None and not df.empty]
    for df in frames:
        status_cols = [c for c in df.columns if str(c).endswith("_available")]
        for c in status_cols:
            s = df[c].dropna()
            if s.empty:
                continue
            for v in s.head(50):
                if v is None or (isinstance(v, float) and pd.isna(v)):
                    continue
                if not isinstance(v, (bool, np.bool_)):
                    if str(v).lower() not in ("true", "false", "0", "1"):
                        warn.append(f"warning:column {c!r} may not be boolean (sample={v!r})")

        suspicious = ("not_computed", "not_applicable", "skipped")
        obj_cols = list(df.select_dtypes(include=["object"]).columns)
        n_warn = 0
        for col in obj_cols:
            if col == "Note" or n_warn > 12:
                break
            for token in suspicious:
                mask = df[col].astype(str).str.lower().eq(token)
                if mask.any():
                    warn.append(
                        f"documentation-only:column {col!r} contains status token {token!r} "
                        f"({int(mask.sum())} rows) — confirm consumers expect strings not NaN"
                    )
                    n_warn += 1
                    break

    return hard, warn


def _series_bool_valid(s: pd.Series) -> pd.Series:
    out = []
    for v in s:
        b = _to_bool(v)
        out.append(True if b is True else False)
    return pd.Series(out, index=s.index)


def audit_harmonic_spectrum_sheet(hs: pd.DataFrame) -> Tuple[List[str], List[str]]:
    hard: List[str] = []
    warn: List[str] = []
    if hs is None or hs.empty:
        warn.append("warning:Harmonic Spectrum sheet missing or empty (interpolation audit skipped)")
        return hard, warn
    req = (
        "bin_center_frequency_hz",
        "interpolated_frequency_hz",
        "subbin_offset_bins",
        "subbin_interpolation_valid",
    )
    for c in req:
        if c not in hs.columns:
            hard.append(f"blocker:Harmonic Spectrum missing {c!r}")
            return hard, warn

    bc = pd.to_numeric(hs["bin_center_frequency_hz"], errors="coerce")
    fi = pd.to_numeric(hs["interpolated_frequency_hz"], errors="coerce")
    if not (bc.notna() & fi.notna() & (bc != fi)).any():
        warn.append(
            "documentation-only:Harmonic Spectrum has no row where interpolated_frequency_hz "
            "differs from bin_center_frequency_hz (may be normal for this clip)"
        )

    if "subbin_interpolation_valid" in hs.columns and "extracted_frequency_hz" in hs.columns:
        valid = _series_bool_valid(hs["subbin_interpolation_valid"])
        ext = pd.to_numeric(hs["extracted_frequency_hz"], errors="coerce")
        if valid.any():
            diff = (ext.loc[valid] - fi.loc[valid]).abs()
            if (diff > 1e-3).any():
                hard.append(
                    "blocker:extracted_frequency_hz != interpolated_frequency_hz when subbin_interpolation_valid"
                )

    if all(c in hs.columns for c in ("frequency_deviation_hz", "extracted_frequency_hz", "expected_frequency_hz")):
        fd = pd.to_numeric(hs["frequency_deviation_hz"], errors="coerce")
        ex = pd.to_numeric(hs["extracted_frequency_hz"], errors="coerce")
        ef = pd.to_numeric(hs["expected_frequency_hz"], errors="coerce")
        exp_fd = ex - ef
        m = fd.notna() & exp_fd.notna()
        if m.any() and (fd.loc[m] - exp_fd.loc[m]).abs().max() > 1e-2:
            hard.append("blocker:frequency_deviation_hz != extracted - expected")

    return hard, warn


def _meta_value_equals_bool(actual: Any, expected: bool) -> bool:
    b = _to_bool(actual)
    if b is not None:
        return b is expected
    return str(actual).strip().lower() == str(expected).lower()


def audit_canonical_metadata(meta: Mapping[str, Any]) -> Tuple[List[str], List[str], List[str]]:
    """Returns (hard, warnings, documentation)."""
    hard: List[str] = []
    warn: List[str] = []
    doc: List[str] = []
    required = (
        "pipeline_contract_version",
        "stage1_module",
        "stage1_class",
        "stage2_module",
        "stage2_function",
        "compiled_from",
        "accepted_input_engine",
        "legacy_pipeline_used",
        "publication_output_allowed",
        "legacy_super_json_allowed",
    )
    for k in required:
        if k not in meta or meta[k] is None or (isinstance(meta[k], float) and pd.isna(meta[k])):
            hard.append(f"blocker:Analysis_Metadata missing {k!r}")

    def _eq_str(key: str, expected: str) -> None:
        if key not in meta:
            return
        v = meta[key]
        if str(v).strip() != str(expected).strip():
            hard.append(f"blocker:Analysis_Metadata {key}={v!r} expected {expected!r}")

    def _eq_bool(key: str, expected: bool) -> None:
        if key not in meta:
            return
        if not _meta_value_equals_bool(meta[key], expected):
            hard.append(f"blocker:Analysis_Metadata {key}={meta[key]!r} expected {expected!r}")

    _eq_str("stage1_module", "proc_audio")
    _eq_str("stage1_class", "AudioProcessor")
    _eq_str("stage2_module", "compile_metrics")
    _eq_str("stage2_function", "compile_density_metrics_with_pca")
    _eq_str("compiled_from", "spectral_analysis.xlsx")
    _eq_str("accepted_input_engine", "proc_audio.AudioProcessor")
    _eq_bool("legacy_pipeline_used", False)
    _eq_bool("publication_output_allowed", True)
    _eq_bool("legacy_super_json_allowed", False)

    schema = _to_str(meta.get("input_schema_validation_status", ""))
    if schema.startswith("not_validated"):
        warn.append(
            f"warning:input_schema_validation_status={schema!r} "
            f"(TODO: wire orchestrator schema validator — not a release blocker per audit policy)"
        )
    return hard, warn, doc


def run_audit_on_workbook(
    path: Path,
    *,
    per_note_workbook: Optional[Path] = None,
) -> "AuditReport":
    """Run all checks on ``compiled_density_metrics.xlsx``."""
    path = Path(path)
    sheets = load_compiled_workbook(path)
    canonical = sheets.get("Canonical_Metrics", pd.DataFrame())
    diagnostic = sheets.get("Diagnostic_Metrics", pd.DataFrame())
    density = sheets.get("Density_Metrics", pd.DataFrame())
    debug_df = sheets.get("Debug_Counts", pd.DataFrame())
    am = sheets.get("Analysis_Metadata", pd.DataFrame())
    meta = analysis_metadata_as_dict(am)

    rep = AuditReport(row_count=int(len(canonical)) if not canonical.empty else 0)
    rep.hard_failures.extend(audit_f0_provenance(canonical, diagnostic))
    rep.hard_failures.extend(audit_subfundamental(canonical))
    rep.hard_failures.extend(audit_ambiguous_column_names(sheets))
    dh, dw = audit_debug_counts(debug_df)
    rep.hard_failures.extend(dh)
    rep.warnings.extend(dw)
    mh, mw = audit_missing_value_policy(density, diagnostic)
    rep.hard_failures.extend(mh)
    rep.warnings.extend(mw)

    ch, cw, cd = audit_canonical_metadata(meta)
    rep.hard_failures.extend(ch)
    rep.warnings.extend(cw)
    rep.documentation.extend(cd)

    hs_path = per_note_workbook
    if hs_path is not None and Path(hs_path).is_file():
        try:
            hs = pd.read_excel(hs_path, sheet_name="Harmonic Spectrum")
            hh, hw = audit_harmonic_spectrum_sheet(hs)
            rep.hard_failures.extend(hh)
            rep.warnings.extend(hw)
        except Exception as e:
            rep.warnings.append(f"warning:Harmonic Spectrum read failed: {e}")
    else:
        rep.documentation.append(
            "documentation-only:per-note workbook not supplied — harmonic interpolation sheet not audited"
        )

    return rep


@dataclass
class AuditReport:
    row_count: int = 0
    hard_failures: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    documentation: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.hard_failures

    def print_summary(self) -> None:
        print(f"Workbook rows (Canonical_Metrics): {self.row_count}")
        print(f"f0 provenance contradictions (blockers): {sum(1 for x in self.hard_failures if 'f0_' in x.lower())}")
        print(f"Adaptive cutoff inconsistencies (blockers): {sum(1 for x in self.hard_failures if 'subfundamental' in x.lower() or 'percentage_subfundamental' in x.lower())}")
        print(f"Debug_Counts invariant failures (blockers): {sum(1 for x in self.hard_failures if 'debug_counts' in x.lower())}")
        print(f"Missing-value policy issues (blockers): {sum(1 for x in self.hard_failures if 'n_harm' in x.lower() or 'index_weighted' in x.lower())}")
        print(f"Canonical metadata / sheet blockers: {sum(1 for x in self.hard_failures if 'metadata' in x.lower() or 'sheet' in x.lower() or 'ambiguous' in x.lower())}")
        print(f"Total hard failures: {len(self.hard_failures)}")
        print("--- detail: hard failures ---")
        for x in self.hard_failures:
            print(f"  [BLOCKER] {x}")
        print(f"Warnings: {len(self.warnings)}")
        for x in self.warnings[:50]:
            print(f"  [WARN] {x}")
        if len(self.warnings) > 50:
            print(f"  ... ({len(self.warnings) - 50} more warnings truncated)")
        print(f"Documentation notes: {len(self.documentation)}")
        for x in self.documentation[:30]:
            print(f"  [DOC] {x}")
        print("---")
        if self.ok:
            print("FINAL: PASS (no hard invariant failures)")
        else:
            print("FINAL: FAIL")


def audit_cli_main(argv: Optional[List[str]] = None) -> int:
    import sys

    argv = argv if argv is not None else sys.argv
    if len(argv) < 2:
        print("Usage: python tools/audit_compiled_workbook.py path/to/compiled_density_metrics.xlsx [per_note_spectral_analysis.xlsx]")
        return 2
    path = Path(argv[1])
    if not path.is_file():
        print(f"Not a file: {path}")
        return 2
    per_note: Optional[Path] = Path(argv[2]) if len(argv) > 2 and str(argv[2]).strip() else None
    if per_note is not None and not per_note.is_file():
        print(f"Not a file (per-note workbook ignored): {per_note}")
        per_note = None
    rep = run_audit_on_workbook(path, per_note_workbook=per_note)
    rep.print_summary()
    return 0 if rep.ok else 1
