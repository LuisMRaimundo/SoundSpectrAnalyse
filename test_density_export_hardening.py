"""Hardening tests: dissonance cap metadata, Density_Metrics naming, validator script, docs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

from compile_metrics import DENSITY_METRICS_ALLOWED_COLUMNS, _write_compiled_excel
from constants import DISSONANCE_CAP_COMPUTATION_NOTE, DISSONANCE_PAIRWISE_PARTIAL_CAP


def _density_core_row(note: str, seed: float) -> dict:
    h = 0.55 + 0.01 * seed
    ih = 0.28 - 0.006 * seed
    sb = max(0.05, 1.0 - h - ih)
    hpc = 3 + int(seed % 5)
    ihpc = 1 + int(seed % 3)
    sbpc = int(seed % 2)
    h_sum = 0.5 + seed
    i_sum = 0.2 + 0.02 * seed
    s_sum = 0.05
    t_sum = h_sum + i_sum + s_sum
    return {
        "Note": note,
        "weight_function": "linear",
        "Harmonic Partials sum": h_sum,
        "Inharmonic Partials sum": i_sum,
        "Sub-bass sum": s_sum,
        "Total sum": t_sum,
        "effective_partial_density": 1.0 + 0.1 * seed,
        "harmonic_energy_sum": 0.5 + seed,
        "inharmonic_energy_sum": 0.2 + 0.02 * seed,
        "subbass_energy_sum": 0.05,
        "total_component_energy": 1.0 + 0.1 * seed,
        "harmonic_energy_ratio": h,
        "inharmonic_energy_ratio": ih,
        "subbass_energy_ratio": sb,
        "harmonic_order_count": hpc,
        "harmonic_peak_count": hpc,
        "inharmonic_peak_count": ihpc,
        "subbass_peak_count": sbpc,
        "total_detected_peak_count": hpc + ihpc + sbpc,
        "spectral_entropy": 0.35 + 0.02 * seed,
    }


def test_analysis_metadata_dissonance_cap_when_capped(tmp_path: Path) -> None:
    cap = int(DISSONANCE_PAIRWISE_PARTIAL_CAP)
    n_before = cap + 40
    n_after = cap
    pair_after = n_after * (n_after - 1) // 2
    rows = []
    for i in range(12):
        r = _density_core_row(f"N{i}", float(i))
        r["Sethares Dissonance"] = 0.01 * i
        r["dissonance_partial_cap"] = cap
        r["dissonance_partial_count_before_cap"] = n_before
        r["dissonance_partial_count_after_cap"] = n_after
        r["dissonance_pair_count_after_cap"] = pair_after
        r["dissonance_cap_computation_note"] = DISSONANCE_CAP_COMPUTATION_NOTE
        rows.append(r)
    df = pd.DataFrame(rows)
    outp = tmp_path / "cap.xlsx"
    _write_compiled_excel(
        outp,
        df,
        {"dissonance_enabled": True, "selected_dissonance_model": "sethares"},
        enable_pca_export=False,
    )
    am = pd.read_excel(outp, sheet_name="Analysis_Metadata")
    meta = am.iloc[0].to_dict()
    assert int(float(meta.get("dissonance_partial_cap", 0))) == cap
    assert int(float(meta.get("dissonance_partial_count_after_cap", 0))) <= cap
    assert str(meta.get("dissonance_cap_computation_note", "")).strip() != ""


def test_analysis_metadata_dissonance_not_applied(tmp_path: Path) -> None:
    rows = []
    for i in range(8):
        r = _density_core_row(f"M{i}", float(i))
        r["Sethares Dissonance"] = 0.02 * i
        r["dissonance_partial_cap"] = "not_applied"
        r["dissonance_partial_count_before_cap"] = 6
        r["dissonance_partial_count_after_cap"] = 6
        r["dissonance_pair_count_after_cap"] = 15
        r["dissonance_cap_computation_note"] = "Full list."
        rows.append(r)
    df = pd.DataFrame(rows)
    outp = tmp_path / "nocap.xlsx"
    _write_compiled_excel(outp, df, {"dissonance_enabled": True}, enable_pca_export=False)
    am = pd.read_excel(outp, sheet_name="Analysis_Metadata")
    meta = am.iloc[0].to_dict()
    assert meta.get("dissonance_partial_cap") == "not_applied"


def test_density_metrics_sheet_only_partial_sums_no_debug_counts(tmp_path: Path) -> None:
    from compile_metrics import DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS

    rows = []
    for i in range(5):
        r = _density_core_row(f"P{i}", float(i))
        r["harmonic_bin_count"] = 999
        rows.append(r)
    df = pd.DataFrame(rows)
    outp = tmp_path / "peak.xlsx"
    _write_compiled_excel(outp, df, {}, enable_pca_export=False)
    dm = pd.read_excel(outp, sheet_name="Density_Metrics")
    assert list(dm.columns) == DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS
    assert "harmonic_peak_count" not in dm.columns
    assert "harmonic_bin_count" not in dm.columns
    assert "harmonic_partial_count" not in dm.columns


def test_density_metrics_forbidden_pc1_fails_validation(tmp_path: Path) -> None:
    from compile_metrics import validate_compiled_density_workbook

    rows = [_density_core_row("A0", 1.0)]
    rows[0]["PC1"] = 0.1
    df = pd.DataFrame(rows)
    outp = tmp_path / "bad.xlsx"
    # Need valid Analysis_Metadata for validator
    _write_compiled_excel(outp, df, {}, enable_pca_export=False)
    # Inject forbidden column by rewriting Density_Metrics only
    xl = pd.ExcelFile(outp)
    frames = {s: pd.read_excel(outp, sheet_name=s) for s in xl.sheet_names}
    xd = frames["Density_Metrics"].copy()
    xd["PC1"] = 0.0
    frames["Density_Metrics"] = xd
    with pd.ExcelWriter(outp, engine="openpyxl") as w:
        for s, d in frames.items():
            d.to_excel(w, sheet_name=s, index=False)
    errs = validate_compiled_density_workbook(outp)
    assert any("forbidden" in e.lower() or "disallow" in e.lower() for e in errs)


def test_validate_density_workbook_script_good_and_bad(tmp_path: Path) -> None:
    from compile_metrics import validate_compiled_density_workbook

    repo = Path(__file__).resolve().parent.parent
    script = repo / "scripts" / "validate_density_workbook.py"

    good = tmp_path / "good.xlsx"
    rows = [_density_core_row(f"G{i}", float(i)) for i in range(5)]
    _write_compiled_excel(good, pd.DataFrame(rows), {}, enable_pca_export=False)
    assert validate_compiled_density_workbook(good) == []

    r = subprocess.run(
        [sys.executable, str(script), str(good)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "passed" in r.stdout.lower()

    bad = tmp_path / "badscript.xlsx"
    _write_compiled_excel(bad, pd.DataFrame(rows), {}, enable_pca_export=False)
    xl2 = pd.ExcelFile(bad)
    fr2 = {s: pd.read_excel(bad, sheet_name=s) for s in xl2.sheet_names}
    xd2 = fr2["Density_Metrics"].copy()
    xd2["sethares_dissonance"] = 1.0
    fr2["Density_Metrics"] = xd2
    with pd.ExcelWriter(bad, engine="openpyxl") as w:
        for s, d in fr2.items():
            d.to_excel(w, sheet_name=s, index=False)
    r2 = subprocess.run(
        [sys.executable, str(script), str(bad)],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=False,
    )
    assert r2.returncode != 0
    assert "fail" in r2.stdout.lower() or "failed" in r2.stdout.lower()


def test_density_export_schema_doc_exists() -> None:
    repo = Path(__file__).resolve().parent.parent
    doc = repo / "docs" / "DENSITY_EXPORT_SCHEMA.md"
    assert doc.is_file()
    text = doc.read_text(encoding="utf-8")
    for term in (
        "effective_partial_density",
        "harmonic_order_count",
        "Debug_Counts",
        "Dissonance_Metrics",
        "PCA_Scores",
        "Analysis_Metadata",
        "spectral masking",
        "batch_harmonic_energy_ratio",
    ):
        assert term in text


def test_allowed_columns_are_minimal_partial_sums_only() -> None:
    assert "Note" in DENSITY_METRICS_ALLOWED_COLUMNS
    assert "weight_function" in DENSITY_METRICS_ALLOWED_COLUMNS
    assert "Harmonic Partials sum" in DENSITY_METRICS_ALLOWED_COLUMNS
    assert "harmonic_order_count" not in DENSITY_METRICS_ALLOWED_COLUMNS
