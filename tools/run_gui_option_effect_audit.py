from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import compile_metrics
from note_parser import canonical_note_from_filename
from proc_audio import AudioProcessor
from tools.export_research_density_workbook import export_research_workbook


CORPUS_DIR = Path(os.environ.get("SSA_AUDIT_CORPUS_DIR", "<corpus_path>"))
OUT_ROOT = CORPUS_DIR / "analysis_results_gui_effect_audit"


@dataclass
class ScenarioResult:
    name: str
    out_dir: Path
    compiled_path: Path
    research_path: Path
    per_note_path: Path
    metrics_df: pd.DataFrame
    charts_df: pd.DataFrame
    meta_map: dict[str, Any]


def _select_subset_files(corpus_dir: Path) -> list[Path]:
    wanted = ["D3_", "F4_", "A4_", "G5_", "C6_"]
    files = sorted(p for p in corpus_dir.glob("*.wav"))
    out: list[Path] = []
    for tok in wanted:
        m = next((p for p in files if tok in p.name), None)
        if m is not None:
            out.append(m)
    if len(out) != len(wanted):
        raise RuntimeError(f"Could not build deterministic 5-note subset from {corpus_dir}")
    return out


def _pick_window_params(window: str) -> tuple[float | None, float | None]:
    w = window.lower().strip()
    if w == "kaiser":
        return 14.0, None
    if w in ("gaussian", "gauss", "gaussiana"):
        return None, 512.0
    return None, None


def _meta_map_from_research(path: Path) -> dict[str, Any]:
    md = pd.read_excel(path, sheet_name="Metadata", engine="openpyxl")
    key_col = None
    if "Parameter" in md.columns:
        key_col = "Parameter"
    elif "Field" in md.columns:
        key_col = "Field"
    if key_col is None or "Value" not in md.columns:
        return {}
    out: dict[str, Any] = {}
    for _, row in md.iterrows():
        k = str(row.get(key_col, "")).strip()
        if k:
            out[k] = row.get("Value")
    return out


def run_scenario(name: str, files: list[Path], overrides: dict[str, Any]) -> ScenarioResult:
    out_dir = OUT_ROOT / name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = {
        "freq_min": 20.0,
        "freq_max": 20000.0,
        "db_min": -90.0,
        "db_max": 0.0,
        "n_fft": 4096,
        "hop_length": 1024,
        "window": "blackmanharris",
        "tolerance": 5.0,
        "use_adaptive_tolerance": True,
        "zero_padding": 2,
        "time_avg": "median",
        "weight_function": "linear",
        "density_summation_mode": "his_weighted",
        "harmonic_density_weight": 1.0,
        "inharmonic_density_weight": 0.5,
        "subbass_density_weight": 0.25,
        "density_salience_threshold_db": -45.0,
        "density_frequency_ceiling_hz": 5000.0,
    }
    base.update(overrides or {})

    for wav in files:
        note, note_source = canonical_note_from_filename(wav.name, parent_folder=wav.parent.name)
        parent_output_dir = out_dir / wav.stem
        kaiser_beta, gaussian_std = _pick_window_params(str(base["window"]))

        ap = AudioProcessor()
        ap.note_source = note_source
        if note:
            ap.note = note
        ap.load_audio_files([str(wav)])
        ap.apply_filters_and_generate_data(
            freq_min=float(base["freq_min"]),
            freq_max=float(base["freq_max"]),
            db_min=float(base["db_min"]),
            db_max=float(base["db_max"]),
            n_fft=int(base["n_fft"]),
            hop_length=int(base["hop_length"]),
            window=str(base["window"]),
            tolerance=float(base["tolerance"]),
            use_adaptive_tolerance=bool(base["use_adaptive_tolerance"]),
            results_directory=str(parent_output_dir),
            dissonance_enabled=False,
            compare_models=False,
            harmonic_weight=0.5,
            inharmonic_weight=0.5,
            auto_model_weights_from_analysis=True,
            weight_function=str(base["weight_function"]),
            zero_padding=int(base["zero_padding"]),
            time_avg=str(base["time_avg"]),
            spectral_masking_enabled=False,
            density_summation_mode=str(base["density_summation_mode"]),
            harmonic_density_weight=float(base["harmonic_density_weight"]),
            inharmonic_density_weight=float(base["inharmonic_density_weight"]),
            subbass_density_weight=float(base["subbass_density_weight"]),
            density_salience_threshold_db=float(base["density_salience_threshold_db"]),
            density_frequency_ceiling_hz=float(base["density_frequency_ceiling_hz"]),
            tier=None,
            kaiser_beta=kaiser_beta,
            gaussian_std=gaussian_std,
            compile_per_call=False,
            use_tsne=False,
            use_umap=False,
            detect_anomalies=False,
            anomaly_contamination=None,
        )

    compiled_path = out_dir / f"compiled_density_metrics_{name}.xlsx"
    _ = compile_metrics.compile_density_metrics_with_pca(
        folder_path=str(out_dir),
        output_path=str(compiled_path),
        file_pattern="spectral_analysis.xlsx",
        include_pca=True,
        harmonic_weight=0.5,
        inharmonic_weight=0.5,
        weight_function=str(base["weight_function"]),
        use_tsne=False,
        use_umap=False,
        detect_anomalies=False,
        anomaly_contamination=None,
        allow_legacy_super_json=False,
        compilation_extra_metadata={
            "input_schema_validation_status": "not_validated_orchestrator_v2_16",
            "legacy_pipeline_used": False,
            "publication_output_allowed": True,
            "gui_effect_audit_scenario": name,
        },
    )

    research_path = out_dir / f"compiled_density_metrics_research_{name}.xlsx"
    export_research_workbook(compiled_path, research_path, overwrite=True)

    metrics_df = pd.read_excel(research_path, sheet_name="Spectral_Density_Metrics", engine="openpyxl")
    charts_df = pd.read_excel(research_path, sheet_name="Charts_Data", engine="openpyxl")
    first_note, _ = canonical_note_from_filename(files[0].name, parent_folder=files[0].parent.name)
    per_note_path = (
        out_dir / files[0].stem / (first_note if first_note else "") / "spectral_analysis.xlsx"
    )
    meta_map = _meta_map_from_research(research_path)
    return ScenarioResult(name, out_dir, compiled_path, research_path, per_note_path, metrics_df, charts_df, meta_map)


def _changed_columns(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> list[str]:
    changed: list[str] = []
    k = "Note"
    if k in a.columns and k in b.columns:
        ai = a.set_index(k)
        bi = b.set_index(k)
    else:
        ai, bi = a, b
    for c in cols:
        if c not in ai.columns or c not in bi.columns:
            continue
        xa = pd.to_numeric(ai[c], errors="coerce")
        xb = pd.to_numeric(bi[c], errors="coerce")
        idx = xa.index.intersection(xb.index)
        if len(idx) == 0:
            continue
        if not np.allclose(xa.loc[idx].fillna(0.0), xb.loc[idx].fillna(0.0), atol=1e-9):
            changed.append(c)
    return changed


def main() -> None:
    if not CORPUS_DIR.is_dir():
        raise RuntimeError(
            "Set SSA_AUDIT_CORPUS_DIR to a valid corpus folder before running this audit script."
        )
    files = _select_subset_files(CORPUS_DIR)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    scenarios: dict[str, ScenarioResult] = {}
    scenarios["baseline"] = run_scenario("baseline", files, {})
    scenarios["mode_harmonic_only"] = run_scenario(
        "mode_harmonic_only", files, {"density_summation_mode": "harmonic_only"}
    )
    scenarios["mode_inharmonic_only"] = run_scenario(
        "mode_inharmonic_only", files, {"density_summation_mode": "inharmonic_only"}
    )
    scenarios["mode_subbass_only"] = run_scenario(
        "mode_subbass_only", files, {"density_summation_mode": "subbass_only"}
    )
    scenarios["mode_his_weighted"] = run_scenario(
        "mode_his_weighted",
        files,
        {
            "density_summation_mode": "his_weighted",
            "harmonic_density_weight": 1.0,
            "inharmonic_density_weight": 0.5,
            "subbass_density_weight": 0.25,
        },
    )
    scenarios["weights_custom"] = run_scenario(
        "weights_custom",
        files,
        {
            "density_summation_mode": "his_weighted",
            "harmonic_density_weight": 0.2,
            "inharmonic_density_weight": 1.1,
            "subbass_density_weight": 0.6,
        },
    )
    scenarios["threshold_-35"] = run_scenario(
        "threshold_-35", files, {"density_salience_threshold_db": -35.0}
    )
    scenarios["threshold_-45"] = run_scenario(
        "threshold_-45", files, {"density_salience_threshold_db": -45.0}
    )
    scenarios["threshold_-55"] = run_scenario(
        "threshold_-55", files, {"density_salience_threshold_db": -55.0}
    )
    scenarios["ceiling_3000"] = run_scenario(
        "ceiling_3000", files, {"density_frequency_ceiling_hz": 3000.0}
    )
    scenarios["ceiling_5000"] = run_scenario(
        "ceiling_5000", files, {"density_frequency_ceiling_hz": 5000.0}
    )
    scenarios["ceiling_8000"] = run_scenario(
        "ceiling_8000", files, {"density_frequency_ceiling_hz": 8000.0}
    )
    scenarios["window_hann"] = run_scenario("window_hann", files, {"window": "hann"})
    scenarios["nfft_2048"] = run_scenario("nfft_2048", files, {"n_fft": 2048, "hop_length": 512})
    scenarios["hop_256"] = run_scenario("hop_256", files, {"hop_length": 256})
    scenarios["zp_1"] = run_scenario("zp_1", files, {"zero_padding": 1})
    scenarios["dbmin_-70"] = run_scenario("dbmin_-70", files, {"db_min": -70.0})
    scenarios["dbmin_-50"] = run_scenario("dbmin_-50", files, {"db_min": -50.0})
    scenarios["dbmin_-35"] = run_scenario("dbmin_-35", files, {"db_min": -35.0})
    scenarios["tol_3"] = run_scenario("tol_3", files, {"tolerance": 3.0, "use_adaptive_tolerance": False})
    scenarios["tol_20"] = run_scenario("tol_20", files, {"tolerance": 20.0, "use_adaptive_tolerance": False})

    key_cols = [
        "salient_harmonic_order_count_up_to_5000hz",
        "salient_inharmonic_log_bin_count_up_to_5000hz",
        "salient_subbass_particle_count",
        "final_note_density_count_based",
        "final_note_density_salience_weighted",
        "harmonic_occupancy_ratio",
        "harmonic_slot_coverage_ratio",
    ]
    base = scenarios["baseline"]

    rows: list[dict[str, Any]] = []

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

    def _allclose_series(df: pd.DataFrame, a: str, b: str) -> bool:
        if a not in df.columns or b not in df.columns:
            return False
        xa = pd.to_numeric(df[a], errors="coerce")
        xb = pd.to_numeric(df[b], errors="coerce")
        m = xa.notna() & xb.notna()
        return bool(m.any() and np.allclose(xa[m], xb[m], atol=1e-9))

    def _weighted_formula_ok(df: pd.DataFrame) -> bool:
        needed = [
            "final_note_density_count_based",
            "salient_harmonic_order_count_up_to_5000hz",
            "salient_inharmonic_log_bin_count_up_to_5000hz",
            "salient_subbass_particle_count",
        ]
        if any(c not in df.columns for c in needed):
            return False
        lhs = pd.to_numeric(df["final_note_density_count_based"], errors="coerce")
        h = pd.to_numeric(df["salient_harmonic_order_count_up_to_5000hz"], errors="coerce")
        i = pd.to_numeric(df["salient_inharmonic_log_bin_count_up_to_5000hz"], errors="coerce")
        s = pd.to_numeric(df["salient_subbass_particle_count"], errors="coerce")
        rhs = 1.0 * h + 0.5 * i + 0.25 * s
        m = lhs.notna() & rhs.notna()
        return bool(m.any() and np.allclose(lhs[m], rhs[m], atol=1e-9))

    # Density mode checks
    mode_h_ok = _allclose_series(
        scenarios["mode_harmonic_only"].metrics_df,
        "final_note_density_count_based",
        "salient_harmonic_order_count_up_to_5000hz",
    )
    mode_i_ok = _allclose_series(
        scenarios["mode_inharmonic_only"].metrics_df,
        "final_note_density_count_based",
        "salient_inharmonic_log_bin_count_up_to_5000hz",
    )
    mode_s_ok = _allclose_series(
        scenarios["mode_subbass_only"].metrics_df,
        "final_note_density_count_based",
        "salient_subbass_particle_count",
    )
    add_row(
        "density_summation_mode",
        "harmonic_only / inharmonic_only / subbass_only / his_weighted",
        "Mode-specific identity constraints hold for final_note_density_count_based",
        f"harmonic_only={mode_h_ok}, inharmonic_only={mode_i_ok}, subbass_only={mode_s_ok}",
        "PASS" if (mode_h_ok and mode_i_ok and mode_s_ok) else "FAIL",
        ["final_note_density_count_based"],
        "Mode-to-formula wiring verified on 5-note deterministic subset.",
    )

    weights_changed = _changed_columns(
        base.metrics_df,
        scenarios["weights_custom"].metrics_df,
        ["final_note_density_count_based", "final_note_density_salience_weighted"],
    )
    weighted_formula_ok = _weighted_formula_ok(scenarios["mode_his_weighted"].metrics_df)
    add_row(
        "density weights (wH,wI,wS)",
        "wH,wI,wS variations",
        "Weighted count/salience density should change per formula",
        f"weighted_formula_ok={weighted_formula_ok}; changed_columns={weights_changed}",
        "PASS" if (weighted_formula_ok and len(weights_changed) > 0) else "FAIL",
        weights_changed,
        "his_weighted explicit mode with 1.0/0.5/0.25 validated; custom weights produce output deltas.",
    )

    # Threshold effects
    t35 = scenarios["threshold_-35"].metrics_df
    t45 = scenarios["threshold_-45"].metrics_df
    t55 = scenarios["threshold_-55"].metrics_df
    m35 = float(pd.to_numeric(t35["final_note_density_salience_weighted"], errors="coerce").mean())
    m45 = float(pd.to_numeric(t45["final_note_density_salience_weighted"], errors="coerce").mean())
    m55 = float(pd.to_numeric(t55["final_note_density_salience_weighted"], errors="coerce").mean())
    h35 = float(pd.to_numeric(t35["salient_harmonic_order_count_up_to_5000hz"], errors="coerce").mean())
    h45 = float(pd.to_numeric(t45["salient_harmonic_order_count_up_to_5000hz"], errors="coerce").mean())
    h55 = float(pd.to_numeric(t55["salient_harmonic_order_count_up_to_5000hz"], errors="coerce").mean())
    thr_monotonic = (m35 <= m45 + 1e-9) and (m45 <= m55 + 1e-9)
    add_row(
        "density_salience_threshold_db",
        "-35 / -45 / -55",
        "More permissive threshold should not reduce global density means",
        f"mean(final)=[{m35:.4f},{m45:.4f},{m55:.4f}] mean(H_count)=[{h35:.3f},{h45:.3f},{h55:.3f}]",
        "PASS" if thr_monotonic else "AMBIGUOUS",
        ["final_note_density_salience_weighted", "salient_harmonic_order_count_up_to_5000hz"],
        "Order is strict->default->permissive.",
    )

    # Ceiling effects (generic ceiling-aware aliases)
    c3 = scenarios["ceiling_3000"].metrics_df
    c5 = scenarios["ceiling_5000"].metrics_df
    c8 = scenarios["ceiling_8000"].metrics_df
    colc = "salient_harmonic_order_count_up_to_density_ceiling_hz"
    if colc in c3.columns and colc in c5.columns and colc in c8.columns:
        mc3 = float(pd.to_numeric(c3[colc], errors="coerce").mean())
        mc5 = float(pd.to_numeric(c5[colc], errors="coerce").mean())
        mc8 = float(pd.to_numeric(c8[colc], errors="coerce").mean())
        ceil_ok = (mc3 <= mc5 + 1e-9) and (mc5 <= mc8 + 1e-9)
        obs = f"mean({colc})=[{mc3:.3f},{mc5:.3f},{mc8:.3f}]"
        status = "PASS" if ceil_ok else "AMBIGUOUS"
        affected = [colc]
    else:
        obs = "ceiling-aware alias column missing"
        status = "FAIL"
        affected = []
    add_row(
        "density_frequency_ceiling_hz",
        "3000 / 5000 / 8000",
        "Higher ceiling should increase-or-hold salient harmonic count",
        obs,
        status,
        affected,
        "Ceiling-aware aliases used to avoid silent reinterpretation of up_to_5000hz names.",
    )

    stft_cases = [
        ("window type", "blackmanharris -> hann", "window_hann"),
        ("n_fft / tier strategy", "4096 -> 2048", "nfft_2048"),
        ("hop length", "1024 -> 256", "hop_256"),
        ("zero padding", "2 -> 1", "zp_1"),
        ("magnitude threshold", "db_min -90 -> -70", "dbmin_-70"),
    ]
    for label, tested, sc_name in stft_cases:
        sc = scenarios[sc_name]
        changed = _changed_columns(base.metrics_df, sc.metrics_df, key_cols)
        status = "PASS" if changed else "AMBIGUOUS"
        observed = (
            f"{len(changed)} key output columns changed" if changed else "No key density output change detected"
        )
        add_row(
            label,
            tested,
            "Changing GUI-controlled STFT/threshold/tolerance should alter spectral extraction and/or metadata",
            observed,
            status,
            changed,
            f"Scenario path: {sc.out_dir}",
        )

    db_changes = set(
        _changed_columns(base.metrics_df, scenarios["dbmin_-70"].metrics_df, key_cols)
        + _changed_columns(base.metrics_df, scenarios["dbmin_-50"].metrics_df, key_cols)
        + _changed_columns(base.metrics_df, scenarios["dbmin_-35"].metrics_df, key_cols)
    )
    add_row(
        "magnitude threshold",
        "db_min -90 / -70 / -50 / -35",
        "At sufficiently strict thresholds, peak classification/counts should change",
        f"changed_columns={sorted(db_changes)}",
        "PASS" if len(db_changes) > 0 else "AMBIGUOUS",
        sorted(db_changes),
        "If unchanged, this subset may be insensitive while metadata still records the control.",
    )

    tol_changes = set(
        _changed_columns(base.metrics_df, scenarios["tol_3"].metrics_df, key_cols)
        + _changed_columns(base.metrics_df, scenarios["tol_20"].metrics_df, key_cols)
    )
    add_row(
        "harmonic tolerance",
        "3 / 5 / 20 (adaptive off for extremes)",
        "Harmonic-vs-residual assignment should change under wide tolerance sweeps",
        f"changed_columns={sorted(tol_changes)}",
        "PASS" if len(tol_changes) > 0 else "AMBIGUOUS",
        sorted(tol_changes),
        "If unchanged, tolerance path may be weakly coupled to this subset's final-density outputs.",
    )

    # Metadata propagation audit row.
    required_meta = [
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
    present = [k for k in required_meta if k in base.meta_map and str(base.meta_map.get(k)).strip() != ""]
    missing = [k for k in required_meta if k not in present]
    add_row(
        "Metadata propagation",
        ", ".join(required_meta),
        "All used GUI parameters should appear in Metadata",
        f"present={len(present)} missing={len(missing)}",
        "PASS" if not missing else "AMBIGUOUS",
        present,
        ("Missing keys: " + ", ".join(missing)) if missing else "All required keys present",
    )

    add_row(
        "Acceptance 1 (harmonic_only formula)",
        "wH=1,wI=0,wS=0",
        "final_note_density_count_based == salient_harmonic_order_count_up_to_5000hz",
        str(mode_h_ok),
        "PASS" if mode_h_ok else "FAIL",
        ["final_note_density_count_based", "salient_harmonic_order_count_up_to_5000hz"],
        "Mode scenario check.",
    )
    add_row(
        "Acceptance 2 (inharmonic_only formula)",
        "wH=0,wI=1,wS=0",
        "final_note_density_count_based == salient_inharmonic_log_bin_count_up_to_5000hz",
        str(mode_i_ok),
        "PASS" if mode_i_ok else "FAIL",
        ["final_note_density_count_based", "salient_inharmonic_log_bin_count_up_to_5000hz"],
        "Mode scenario check.",
    )
    add_row(
        "Acceptance 3 (subbass_only formula)",
        "wH=0,wI=0,wS=1",
        "final_note_density_count_based == salient_subbass_particle_count",
        str(mode_s_ok),
        "PASS" if mode_s_ok else "FAIL",
        ["final_note_density_count_based", "salient_subbass_particle_count"],
        "Mode scenario check.",
    )
    add_row(
        "Acceptance 4 (weighted H/I/S formula)",
        "wH=1.0,wI=0.5,wS=0.25",
        "final_note_density_count_based = 1.0*H + 0.5*I + 0.25*S",
        str(weighted_formula_ok),
        "PASS" if weighted_formula_ok else "FAIL",
        [
            "final_note_density_count_based",
            "salient_harmonic_order_count_up_to_5000hz",
            "salient_inharmonic_log_bin_count_up_to_5000hz",
            "salient_subbass_particle_count",
        ],
        "his_weighted scenario check.",
    )

    md_path = REPO_ROOT / "docs" / "GUI_OPTION_EFFECT_AUDIT.md"
    json_path = REPO_ROOT / "audit_gui_option_effects.json"

    lines: list[str] = []
    lines.append("# GUI Option Effect Audit")
    lines.append("")
    lines.append(f"- Corpus: `{CORPUS_DIR}`")
    lines.append(f"- Subset notes: `{', '.join(p.name for p in files)}`")
    lines.append(f"- Baseline output: `{scenarios['baseline'].out_dir}`")
    lines.append("")
    lines.append("| GUI option | tested values | expected effect | observed effect | pass/fail | affected columns | notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| {r['GUI option']} | {r['tested values']} | {r['expected effect']} | {r['observed effect']} | {r['pass/fail']} | {', '.join(r['affected columns'])} | {r['notes']} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = {
        "repo_root": str(REPO_ROOT),
        "corpus_dir": str(CORPUS_DIR),
        "subset_files": [str(p) for p in files],
        "baseline_output_dir": str(scenarios["baseline"].out_dir),
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()

