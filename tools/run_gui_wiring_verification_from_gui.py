from __future__ import annotations

import json
import os
import shutil
import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from note_parser import canonical_note_from_filename
from pipeline_orchestrator_gui import (
    DENSITY_MODE_INTERNAL_TO_LABEL,
    RobustOrchestratorApp,
)
from tools.export_research_density_workbook import export_research_workbook
from weight_function_ui_labels import resolve_weight_key_from_user_label


CORPUS_DIR = Path(os.environ.get("SSA_AUDIT_CORPUS_DIR", "<corpus_path>"))
OUT_ROOT = CORPUS_DIR / "analysis_results_gui_wiring_verification"
MD_OUT = REPO_ROOT / "docs" / "GUI_OPTION_EFFECT_AUDIT.md"
JSON_OUT = REPO_ROOT / "audit_gui_option_effects.json"

KEY_COLS = [
    "salient_harmonic_order_count_up_to_5000hz",
    "salient_inharmonic_log_bin_count_up_to_5000hz",
    "salient_subbass_particle_count",
    "final_note_density_count_based",
    "final_note_density_salience_weighted",
]

REQUIRED_META_KEYS = [
    "density_summation_mode",
    "harmonic_density_weight",
    "inharmonic_density_weight",
    "subbass_density_weight",
    "density_salience_threshold_db",
    "density_frequency_ceiling_hz",
    "window_type",
    "n_fft",
    "hop_length",
    "zero_padding",
    "harmonic_tolerance",
    "frequency_min_hz",
    "frequency_max_hz",
    "magnitude_min_db",
]

PROPAGATION_FIELDS = [
    "final_note_density_count_based",
    "final_note_density_salience_weighted",
    "salient_harmonic_order_count_up_to_density_ceiling_hz",
    "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz",
    "salient_subbass_particle_count",
    "harmonic_density_component",
    "inharmonic_density_component",
    "subbass_density_component",
    "harmonic_density_weight",
    "inharmonic_density_weight",
    "subbass_density_weight",
    "density_summation_mode",
    "density_salience_threshold_db",
    "density_frequency_ceiling_hz",
]


@dataclass
class ScenarioResult:
    name: str
    scenario_dir: Path
    analysis_dir: Path
    compiled_path: Path
    research_path: Path
    worker_log: Path
    per_note_path: Path
    sdm: pd.DataFrame
    charts: pd.DataFrame
    dashboard: pd.DataFrame
    meta_map: dict[str, Any]


def _select_subset_files(corpus_dir: Path) -> list[Path]:
    wanted = ["D3_", "F4_", "A4_", "G5_", "C6_"]
    files = sorted(
        p for p in corpus_dir.glob("*") if p.suffix.lower() in {".wav", ".aif", ".aiff"}
    )
    out: list[Path] = []
    for tok in wanted:
        m = next((p for p in files if tok in p.name), None)
        if m is not None:
            out.append(m)
    if len(out) != len(wanted):
        raise RuntimeError("Could not build deterministic 5-note subset.")
    return out


def _set_entry(widget: Any, value: Any) -> None:
    widget.delete(0, tk.END)
    widget.insert(0, str(value))


def _apply_gui_overrides(app: RobustOrchestratorApp, overrides: dict[str, Any]) -> None:
    app.combo_window.set(str(overrides.get("window", "blackmanharris")))
    app.combo_weight.set(str(overrides.get("weight_label", app.combo_weight.get())))
    app.combo_dissonance.set(str(overrides.get("dissonance", "sethares")))
    _mode_internal = str(overrides.get("density_summation_mode", "his_weighted"))
    app.combo_density_mode.set(DENSITY_MODE_INTERNAL_TO_LABEL.get(_mode_internal, DENSITY_MODE_INTERNAL_TO_LABEL["his_weighted"]))
    app.combo_avg.set(str(overrides.get("time_avg", "mean")))

    _set_entry(app.entry_min_db, overrides.get("db_min", -90.0))
    _set_entry(app.entry_max_db, overrides.get("db_max", 0.0))
    _set_entry(app.entry_min_freq, overrides.get("freq_min", 20.0))
    _set_entry(app.entry_max_freq, overrides.get("freq_max", 20000.0))
    _set_entry(app.entry_tolerance, overrides.get("tolerance", 5.0))
    _set_entry(app.entry_density_w_h, overrides.get("harmonic_density_weight", 1.0))
    _set_entry(app.entry_density_w_i, overrides.get("inharmonic_density_weight", 0.5))
    _set_entry(app.entry_density_w_s, overrides.get("subbass_density_weight", 0.25))
    _set_entry(
        app.entry_density_salience_threshold_db,
        overrides.get("density_salience_threshold_db", -45.0),
    )
    _set_entry(
        app.entry_density_frequency_ceiling_hz,
        overrides.get("density_frequency_ceiling_hz", 5000.0),
    )

    app.var_smart.set(bool(overrides.get("smart", False)))
    app._update_fixed_fft_visibility()
    _set_entry(app.entry_n_fft, overrides.get("n_fft", 4096))
    _set_entry(app.entry_hop_length, overrides.get("hop_length", 1024))
    _set_entry(app.entry_zero_padding, overrides.get("zero_padding", 2))

    app.var_adaptive_tolerance.set(bool(overrides.get("use_adaptive_tolerance", False)))
    app.var_compile.set(True)
    app.var_use_tsne.set(False)
    app.var_use_umap.set(False)
    app.var_detect_anomalies.set(False)


def _collect_params_from_gui(app: RobustOrchestratorApp) -> dict[str, Any]:
    wf_raw = app.combo_weight.get().strip()
    weight_function = resolve_weight_key_from_user_label(wf_raw)
    return {
        "i_weight": 0.05,
        "manual_model_weight_override": False,
        "avg": app.combo_avg.get(),
        "win": app.combo_window.get().strip().lower(),
        "wf": weight_function,
        "diss": app.combo_dissonance.get(),
        "db_min": float(app.entry_min_db.get() or "-90"),
        "db_max": float(app.entry_max_db.get() or "0"),
        "freq_min": float(app.entry_min_freq.get() or "20"),
        "freq_max": float(app.entry_max_freq.get() or "20000"),
        "tolerance": float(app.entry_tolerance.get() or "5.0"),
        "use_adaptive_tolerance": bool(app.var_adaptive_tolerance.get()),
        "kaiser_beta": None,
        "gaussian_std": None,
        "spectral_masking_enabled": False,
        "density_summation_mode": app._density_mode_internal(),
        "harmonic_density_weight": float(app.entry_density_w_h.get() or "1.0"),
        "inharmonic_density_weight": float(app.entry_density_w_i.get() or "0.5"),
        "subbass_density_weight": float(app.entry_density_w_s.get() or "0.25"),
        "density_salience_threshold_db": float(app.entry_density_salience_threshold_db.get() or "-45.0"),
        "density_frequency_ceiling_hz": float(app.entry_density_frequency_ceiling_hz.get() or "5000.0"),
        "compile": True,
        "smart": bool(app.var_smart.get()),
        "use_tsne": False,
        "use_umap": False,
        "detect_anomalies": False,
        "anomaly_contamination": None,
    }


def _meta_map(path: Path) -> dict[str, Any]:
    md = pd.read_excel(path, sheet_name="Metadata", engine="openpyxl")
    key_col = "Parameter" if "Parameter" in md.columns else "Field"
    out: dict[str, Any] = {}
    for _, row in md.iterrows():
        k = str(row.get(key_col, "")).strip()
        if k:
            out[k] = row.get("Value")
    return out


def _run_gui_scenario(
    app: RobustOrchestratorApp,
    name: str,
    source_files: list[Path],
    overrides: dict[str, Any],
) -> ScenarioResult:
    scenario_dir = OUT_ROOT / name
    if scenario_dir.exists():
        shutil.rmtree(scenario_dir)
    scenario_dir.mkdir(parents=True, exist_ok=True)
    for src in source_files:
        shutil.copy2(src, scenario_dir / src.name)

    _apply_gui_overrides(app, overrides)
    params = _collect_params_from_gui(app)
    ok, err = app._validate_parameters(params)
    if not ok:
        raise RuntimeError(f"Scenario {name} invalid parameters: {err}")

    app.stop_requested = False
    app._process_folder_complete_pipeline(scenario_dir, params)

    analysis_dir = scenario_dir / "analysis_results"
    compiled_path = analysis_dir / "compiled_density_metrics.xlsx"
    if not compiled_path.is_file():
        raise RuntimeError(f"Missing compiled workbook for scenario {name}: {compiled_path}")

    research_path = analysis_dir / "compiled_density_metrics_research.xlsx"
    export_research_workbook(compiled_path, research_path, overwrite=True)

    sdm = pd.read_excel(research_path, sheet_name="Spectral_Density_Metrics", engine="openpyxl")
    charts = pd.read_excel(research_path, sheet_name="Charts_Data", engine="openpyxl")
    dashboard = pd.read_excel(research_path, sheet_name="Dashboard", engine="openpyxl")
    meta_map = _meta_map(research_path)

    first_file = source_files[0]
    note, _ = canonical_note_from_filename(first_file.name, parent_folder=first_file.parent.name)
    per_note_path = analysis_dir / first_file.stem / (note if note else "") / "spectral_analysis.xlsx"
    worker_log = analysis_dir / "gui_worker.log"
    return ScenarioResult(
        name=name,
        scenario_dir=scenario_dir,
        analysis_dir=analysis_dir,
        compiled_path=compiled_path,
        research_path=research_path,
        worker_log=worker_log,
        per_note_path=per_note_path,
        sdm=sdm,
        charts=charts,
        dashboard=dashboard,
        meta_map=meta_map,
    )


def _changed_columns(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> list[str]:
    ai = a.set_index("Note") if "Note" in a.columns else a
    bi = b.set_index("Note") if "Note" in b.columns else b
    changed: list[str] = []
    for c in cols:
        if c not in ai.columns or c not in bi.columns:
            continue
        xa = pd.to_numeric(ai[c], errors="coerce")
        xb = pd.to_numeric(bi[c], errors="coerce")
        idx = xa.index.intersection(xb.index)
        if len(idx) and not np.allclose(xa.loc[idx].fillna(0.0), xb.loc[idx].fillna(0.0), atol=1e-9):
            changed.append(c)
    return changed


def _allclose_series(df: pd.DataFrame, a: str, b: str) -> bool:
    if a not in df.columns or b not in df.columns:
        return False
    xa = pd.to_numeric(df[a], errors="coerce")
    xb = pd.to_numeric(df[b], errors="coerce")
    m = xa.notna() & xb.notna()
    return bool(m.any() and np.allclose(xa[m], xb[m], atol=1e-9))


def _weighted_formula_ok(df: pd.DataFrame, w_h: float, w_i: float, w_s: float) -> bool:
    needed = [
        "final_note_density_count_based",
        "salient_harmonic_order_count_up_to_5000hz",
        "salient_inharmonic_log_bin_count_up_to_5000hz",
        "salient_subbass_particle_count",
    ]
    if any(c not in df.columns for c in needed):
        return False
    lhs = pd.to_numeric(df["final_note_density_count_based"], errors="coerce")
    rhs = (
        w_h * pd.to_numeric(df["salient_harmonic_order_count_up_to_5000hz"], errors="coerce")
        + w_i * pd.to_numeric(df["salient_inharmonic_log_bin_count_up_to_5000hz"], errors="coerce")
        + w_s * pd.to_numeric(df["salient_subbass_particle_count"], errors="coerce")
    )
    m = lhs.notna() & rhs.notna()
    return bool(m.any() and np.allclose(lhs[m], rhs[m], atol=1e-9))


def _dashboard_has_fields(dashboard: pd.DataFrame, fields: list[str]) -> bool:
    vals = [
        str(v).strip().lower()
        for v in dashboard.fillna("").to_numpy().ravel()
        if str(v).strip()
    ]
    return all(any(f.lower() in v for v in vals) for f in fields)


def _propagation_check(res: ScenarioResult) -> tuple[bool, dict[str, Any]]:
    per_note_metrics = pd.read_excel(res.per_note_path, sheet_name="Metrics", engine="openpyxl")
    compiled_xl = pd.ExcelFile(res.compiled_path)
    compiled_cols: set[str] = set()
    for s in compiled_xl.sheet_names:
        compiled_cols.update(pd.read_excel(res.compiled_path, sheet_name=s, nrows=1, engine="openpyxl").columns)
    sdm_cols = set(res.sdm.columns)
    charts_cols = set(res.charts.columns)

    details: dict[str, Any] = {}
    all_ok = True
    for f in PROPAGATION_FIELDS:
        equivalent = [f]
        if f == "salient_harmonic_order_count_up_to_density_ceiling_hz":
            equivalent.append("salient_harmonic_order_count_up_to_5000hz")
        if f == "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz":
            equivalent.append("salient_inharmonic_log_bin_count_up_to_5000hz")
        in_per_note = any(e in per_note_metrics.columns for e in equivalent)
        in_compiled = any(e in compiled_cols for e in equivalent)
        in_research = any(e in sdm_cols for e in equivalent)
        in_charts = any(e in charts_cols for e in equivalent)
        in_meta = True
        if f in {
            "harmonic_density_weight",
            "inharmonic_density_weight",
            "subbass_density_weight",
            "density_summation_mode",
            "density_salience_threshold_db",
            "density_frequency_ceiling_hz",
        }:
            val = res.meta_map.get(f, "")
            in_meta = bool(str(val).strip()) and not pd.isna(val)
        status = in_per_note and in_compiled and in_research and in_charts and in_meta
        all_ok = all_ok and status
        details[f] = {
            "per_note": in_per_note,
            "compiled": in_compiled,
            "research": in_research,
            "charts": in_charts,
            "metadata_if_parameter": in_meta,
            "status": "PASS" if status else "FAIL",
        }
    dash_ok = _dashboard_has_fields(
        res.dashboard,
        [
            "final_note_density_salience_weighted",
            "final_note_density_count_based",
            "salient_harmonic_order_count_up_to_5000hz",
        ],
    )
    all_ok = all_ok and dash_ok
    details["dashboard"] = {"status": "PASS" if dash_ok else "FAIL"}
    return all_ok, details


def _metadata_check(meta: dict[str, Any]) -> tuple[str, str, list[str]]:
    missing: list[str] = []
    for k in REQUIRED_META_KEYS:
        v = meta.get(k, "")
        if pd.isna(v) or str(v).strip() == "":
            missing.append(k)
    if missing:
        return "FAIL", f"missing={missing}", missing
    return "PASS", "all required keys present", []


def _log_config_check(log_path: Path) -> tuple[str, str]:
    txt = log_path.read_text(encoding="utf-8", errors="ignore")
    required = [
        "Final density config:",
        "density_summation_mode =",
        "wH =",
        "wI =",
        "wS =",
        "density_salience_threshold_db =",
        "density_frequency_ceiling_hz =",
    ]
    old_phrase = "Model-weight placeholder: H=0.500, I=0.500"
    ok = all(t in txt for t in required) and old_phrase not in txt
    return ("PASS" if ok else "FAIL"), ("config block present and old phrase removed" if ok else "log format mismatch")


def main() -> None:
    if not CORPUS_DIR.is_dir():
        raise RuntimeError(
            "Set SSA_AUDIT_CORPUS_DIR to a valid corpus folder before running this wiring verification."
        )
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    subset_files = _select_subset_files(CORPUS_DIR)
    full_files = sorted(
        p for p in CORPUS_DIR.glob("*") if p.suffix.lower() in {".wav", ".aif", ".aiff"}
    )

    root = tk.Tk()
    root.withdraw()
    app = RobustOrchestratorApp(root)
    app.master.withdraw()

    scenarios: dict[str, ScenarioResult] = {}
    base = {
        "window": "blackmanharris",
        "weight_label": "Linear",
        "dissonance": "sethares",
        "freq_min": 20.0,
        "freq_max": 20000.0,
        "db_min": -90.0,
        "db_max": 0.0,
        "tolerance": 5.0,
        "use_adaptive_tolerance": False,
        "smart": False,
        "n_fft": 4096,
        "hop_length": 1024,
        "zero_padding": 2,
        "time_avg": "mean",
        "density_summation_mode": "his_weighted",
        "harmonic_density_weight": 1.0,
        "inharmonic_density_weight": 0.5,
        "subbass_density_weight": 0.25,
        "density_salience_threshold_db": -45.0,
        "density_frequency_ceiling_hz": 5000.0,
    }

    def run(name: str, files: list[Path], extra: dict[str, Any]) -> ScenarioResult:
        cfg = dict(base)
        cfg.update(extra)
        res = _run_gui_scenario(app, name, files, cfg)
        scenarios[name] = res
        return res

    baseline = run("subset_baseline_his_weighted", subset_files, {})
    mode_h = run("subset_mode_harmonic_only", subset_files, {"density_summation_mode": "harmonic_only"})
    mode_i = run("subset_mode_inharmonic_only", subset_files, {"density_summation_mode": "inharmonic_only"})
    mode_s = run("subset_mode_subbass_only", subset_files, {"density_summation_mode": "subbass_only"})
    mode_w = run(
        "subset_mode_his_weighted_1_0_0_5_0_25",
        subset_files,
        {
            "density_summation_mode": "his_weighted",
            "harmonic_density_weight": 1.0,
            "inharmonic_density_weight": 0.5,
            "subbass_density_weight": 0.25,
        },
    )
    w_b = run(
        "subset_weights_1_0_0_0_0_0",
        subset_files,
        {"density_summation_mode": "his_weighted", "harmonic_density_weight": 1.0, "inharmonic_density_weight": 0.0, "subbass_density_weight": 0.0},
    )
    w_c = run(
        "subset_weights_0_0_1_0_0_0",
        subset_files,
        {"density_summation_mode": "his_weighted", "harmonic_density_weight": 0.0, "inharmonic_density_weight": 1.0, "subbass_density_weight": 0.0},
    )
    w_d = run(
        "subset_weights_0_0_0_0_1_0",
        subset_files,
        {"density_summation_mode": "his_weighted", "harmonic_density_weight": 0.0, "inharmonic_density_weight": 0.0, "subbass_density_weight": 1.0},
    )
    w_e = run(
        "subset_weights_2_0_0_5_0_25",
        subset_files,
        {"density_summation_mode": "his_weighted", "harmonic_density_weight": 2.0, "inharmonic_density_weight": 0.5, "subbass_density_weight": 0.25},
    )
    th35 = run("subset_threshold_-35", subset_files, {"density_salience_threshold_db": -35.0})
    th45 = run("subset_threshold_-45", subset_files, {"density_salience_threshold_db": -45.0})
    th55 = run("subset_threshold_-55", subset_files, {"density_salience_threshold_db": -55.0})
    c3 = run("subset_ceiling_3000", subset_files, {"density_frequency_ceiling_hz": 3000.0})
    c5 = run("subset_ceiling_5000", subset_files, {"density_frequency_ceiling_hz": 5000.0})
    c8 = run("subset_ceiling_8000", subset_files, {"density_frequency_ceiling_hz": 8000.0})
    full = run("full_clarinet_his_weighted", full_files, {})

    rows: list[dict[str, Any]] = []
    propagation_summary: dict[str, Any] = {}

    def add_row(option: str, tested: str, expected: str, observed: str, status: str, affected: list[str], notes: str) -> None:
        rows.append(
            {
                "GUI option": option,
                "tested values": tested,
                "expected effect": expected,
                "observed effect": observed,
                "pass/fail": status,
                "affected columns": affected,
                "notes": notes,
            }
        )

    # Test 1
    t1a = _allclose_series(mode_h.sdm, "final_note_density_count_based", "salient_harmonic_order_count_up_to_5000hz")
    t1b = _allclose_series(mode_i.sdm, "final_note_density_count_based", "salient_inharmonic_log_bin_count_up_to_5000hz")
    t1c = _allclose_series(mode_s.sdm, "final_note_density_count_based", "salient_subbass_particle_count")
    t1d = _weighted_formula_ok(mode_w.sdm, 1.0, 0.5, 0.25)
    add_row(
        "density_summation_mode",
        "harmonic_only / inharmonic_only / subbass_only / his_weighted",
        "Mode-specific formula equalities hold",
        f"harmonic_only={t1a}, inharmonic_only={t1b}, subbass_only={t1c}, his_weighted_formula={t1d}",
        "PASS" if all([t1a, t1b, t1c, t1d]) else "FAIL",
        ["final_note_density_count_based"],
        "Executed via GUI orchestrator path (_process_folder_complete_pipeline).",
    )

    # Test 2
    changed_b = _changed_columns(baseline.sdm, w_b.sdm, ["final_note_density_count_based", "final_note_density_salience_weighted"])
    changed_c = _changed_columns(baseline.sdm, w_c.sdm, ["final_note_density_count_based", "final_note_density_salience_weighted"])
    changed_d = _changed_columns(baseline.sdm, w_d.sdm, ["final_note_density_count_based", "final_note_density_salience_weighted"])
    changed_e = _changed_columns(baseline.sdm, w_e.sdm, ["final_note_density_count_based", "final_note_density_salience_weighted"])
    wf_all = all(
        [
            _weighted_formula_ok(w_b.sdm, 1.0, 0.0, 0.0),
            _weighted_formula_ok(w_c.sdm, 0.0, 1.0, 0.0),
            _weighted_formula_ok(w_d.sdm, 0.0, 0.0, 1.0),
            _weighted_formula_ok(w_e.sdm, 2.0, 0.5, 0.25),
        ]
    )
    add_row(
        "density weights (wH,wI,wS)",
        "A(1,0.5,0.25), B(1,0,0), C(0,1,0), D(0,0,1), E(2,0.5,0.25)",
        "Changing weights changes final densities per formula",
        f"formula_ok={wf_all}; deltas_B={changed_b}; deltas_C={changed_c}; deltas_D={changed_d}; deltas_E={changed_e}",
        "PASS" if wf_all and any([changed_b, changed_c, changed_d, changed_e]) else "FAIL",
        sorted(set(changed_b + changed_c + changed_d + changed_e)),
        "His-weighted mode with GUI-entered weights.",
    )

    # Test 3
    means = []
    for r in (th35, th45, th55):
        means.append(
            (
                float(pd.to_numeric(r.sdm["final_note_density_salience_weighted"], errors="coerce").mean()),
                float(pd.to_numeric(r.sdm["harmonic_density_component"], errors="coerce").mean()),
                float(pd.to_numeric(r.sdm["inharmonic_density_component"], errors="coerce").mean()),
                float(pd.to_numeric(r.sdm["subbass_density_component"], errors="coerce").mean()),
            )
        )
    mono = all(means[i][0] <= means[i + 1][0] + 1e-9 for i in range(len(means) - 1))
    add_row(
        "density_salience_threshold_db",
        "-35 / -45 / -55",
        "More permissive threshold increases or preserves salience-based means globally",
        f"means(final,H,I,S)={means}",
        "PASS" if mono else "FAIL",
        ["final_note_density_salience_weighted", "harmonic_density_component", "inharmonic_density_component", "subbass_density_component"],
        "Threshold sweep via GUI controls.",
    )

    # Test 4
    ceil_col_h = "salient_harmonic_order_count_up_to_density_ceiling_hz"
    ceil_col_i = "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz"
    ceil_cols_ok = all(c in df.sdm.columns for c in [ceil_col_h, ceil_col_i] for df in (c3, c5, c8))
    h_means = [float(pd.to_numeric(x.sdm[ceil_col_h], errors="coerce").mean()) for x in (c3, c5, c8)] if ceil_cols_ok else [float("nan")] * 3
    i_means = [float(pd.to_numeric(x.sdm[ceil_col_i], errors="coerce").mean()) for x in (c3, c5, c8)] if ceil_cols_ok else [float("nan")] * 3
    ceil_mono = bool(ceil_cols_ok and h_means[0] <= h_means[1] + 1e-9 and h_means[1] <= h_means[2] + 1e-9 and i_means[0] <= i_means[1] + 1e-9 and i_means[1] <= i_means[2] + 1e-9)
    add_row(
        "density_frequency_ceiling_hz",
        "3000 / 5000 / 8000",
        "Ceiling-aware counts increase or remain stable with higher ceiling",
        f"mean(H_ceiling_alias)={h_means}; mean(I_ceiling_alias)={i_means}",
        "PASS" if ceil_mono else "FAIL",
        [ceil_col_h, ceil_col_i],
        "Ceiling-aware aliases checked; no reinterpretation in *_up_to_5000hz columns.",
    )

    # Test 5
    meta_status, meta_obs, missing_keys = _metadata_check(full.meta_map)
    add_row(
        "Metadata propagation",
        ", ".join(REQUIRED_META_KEYS),
        "Required GUI settings present and non-blank (or unavailable_not_recorded)",
        meta_obs,
        meta_status,
        REQUIRED_META_KEYS,
        "Checked on full clarinet run.",
    )

    # Test 6
    log_status, log_obs = _log_config_check(full.worker_log)
    add_row(
        "log density config",
        "gui_worker.log run header",
        "Final density config block logged; old confusing placeholder line removed/relabelled",
        log_obs,
        log_status,
        [],
        str(full.worker_log),
    )

    # Test 7
    prop_ok_subset, prop_subset = _propagation_check(baseline)
    prop_ok_full, prop_full = _propagation_check(full)
    propagation_summary["subset_baseline_his_weighted"] = prop_subset
    propagation_summary["full_clarinet_his_weighted"] = prop_full
    add_row(
        "workbook propagation",
        "subset baseline + full clarinet",
        "Fields populated in per-note, compiled, research, Charts_Data, Dashboard, Metadata",
        f"subset={prop_ok_subset}, full={prop_ok_full}",
        "PASS" if prop_ok_subset and prop_ok_full else "FAIL",
        PROPAGATION_FIELDS,
        "Dashboard check includes presence of final/control/salient metrics.",
    )

    status_map = {
        "density_summation_mode": next((r["pass/fail"] for r in rows if r["GUI option"] == "density_summation_mode"), "MISSING"),
        "density weights": next((r["pass/fail"] for r in rows if r["GUI option"] == "density weights (wH,wI,wS)"), "MISSING"),
        "density_salience_threshold_db": next((r["pass/fail"] for r in rows if r["GUI option"] == "density_salience_threshold_db"), "MISSING"),
        "density_frequency_ceiling_hz": next((r["pass/fail"] for r in rows if r["GUI option"] == "density_frequency_ceiling_hz"), "MISSING"),
        "Metadata propagation": next((r["pass/fail"] for r in rows if r["GUI option"] == "Metadata propagation"), "MISSING"),
        "log density config": next((r["pass/fail"] for r in rows if r["GUI option"] == "log density config"), "MISSING"),
    }

    payload = {
        "repo_root": str(REPO_ROOT),
        "corpus_dir": str(CORPUS_DIR),
        "subset_files": [str(p) for p in subset_files],
        "rows": rows,
        "expected_final_statuses": status_map,
        "scenarios": {
            name: {
                "scenario_dir": str(r.scenario_dir),
                "analysis_dir": str(r.analysis_dir),
                "compiled_path": str(r.compiled_path),
                "research_path": str(r.research_path),
                "worker_log": str(r.worker_log),
            }
            for name, r in scenarios.items()
        },
        "propagation_summary": propagation_summary,
    }
    JSON_OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    lines = [
        "# GUI Option Effect Audit (Current GUI Wiring Verification)",
        "",
        f"- Repo root: `{REPO_ROOT}`",
        f"- Corpus: `{CORPUS_DIR}`",
        f"- Deterministic subset: `{', '.join(p.name for p in subset_files)}`",
        "- Execution path: GUI orchestrator (`pipeline_orchestrator_gui.RobustOrchestratorApp._process_folder_complete_pipeline`)",
        "",
        "## Audit Table",
        "",
        "| GUI option | tested values | expected effect | observed effect | pass/fail | affected columns | notes |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        affected = ", ".join(row["affected columns"]) if row["affected columns"] else "-"
        lines.append(
            f"| {row['GUI option']} | {row['tested values']} | {row['expected effect']} | {row['observed effect']} | {row['pass/fail']} | {affected} | {row['notes']} |"
        )

    lines.extend(
        [
            "",
            "## Expected Final Statuses",
            "",
            f"- density_summation_mode: **{status_map['density_summation_mode']}**",
            f"- density weights: **{status_map['density weights']}**",
            f"- density_salience_threshold_db: **{status_map['density_salience_threshold_db']}**",
            f"- density_frequency_ceiling_hz: **{status_map['density_frequency_ceiling_hz']}**",
            f"- Metadata propagation: **{status_map['Metadata propagation']}**",
            f"- log density config: **{status_map['log density config']}**",
            "",
            "## Notes",
            "",
            "- The old ambiguous line `Model-weight placeholder: H=0.500, I=0.500` is no longer emitted as-is; logs now include explicit final density config keys.",
            "- Ceiling behavior is validated on `*_up_to_density_ceiling_hz` columns to avoid overloading `*_up_to_5000hz` names.",
        ]
    )
    MD_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {MD_OUT}")
    print(f"wrote {JSON_OUT}")


if __name__ == "__main__":
    main()
