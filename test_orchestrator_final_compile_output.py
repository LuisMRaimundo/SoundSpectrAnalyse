# -*- coding: utf-8 -*-
"""Regression: Stage 2 must write the root-level compiled_density_metrics.xlsx
under ``main_analysis_output_dir`` after the Stage 1 per-note workbooks have
been produced.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline_orchestrator_integrated import RobustOrchestrator


def _write_minimal_spectral_analysis_xlsx(path: Path, *, seed: float) -> None:
    """Minimal per-note workbook readable by compile_metrics.read_excel_metrics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    h, ih, sb = 0.7, 0.2, 0.1
    row = {
        "Density Metric": 1.0 + 0.1 * seed,
        "Spectral Density Metric": 0.95 + 0.02 * seed,
        "Filtered Density Metric": 0.9 + 0.01 * seed,
        "Spectral Entropy": 0.4 + 0.01 * seed,
        "weight_function": "linear",
        "Harmonic Partials sum": 1.0,
        "Inharmonic Partials sum": 0.2,
        "Sub-bass sum": 0.1,
        "Total sum": 1.3,
        "effective_partial_density": 1.5 + 0.1 * seed,
        "harmonic_energy_sum": 1.0,
        "inharmonic_energy_sum": 0.2,
        "subbass_energy_sum": 0.1,
        "total_component_energy": 1.3,
        "harmonic_energy_ratio": h,
        "inharmonic_energy_ratio": ih,
        "subbass_energy_ratio": sb,
        "harmonic_order_count": 4 + int(seed % 3),
    }
    pd.DataFrame([row]).to_excel(path, sheet_name="Metrics", index=False)


def test_run_stage2_compilation_writes_main_output_workbook(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    analysis_results = tmp_path / "analysis_results"
    analysis_results.mkdir(parents=True, exist_ok=True)

    (corpus / "FileA_A4.wav").write_bytes(b"")
    (corpus / "FileB_A5.wav").write_bytes(b"")

    # Simulate Stage 1 output: <main_output>/<stem>/<note>/spectral_analysis.xlsx
    _write_minimal_spectral_analysis_xlsx(analysis_results / "FileA_A4" / "A4" / "spectral_analysis.xlsx", seed=1.0)
    _write_minimal_spectral_analysis_xlsx(analysis_results / "FileB_A5" / "A5" / "spectral_analysis.xlsx", seed=2.0)

    orch = RobustOrchestrator(
        audio_files=[corpus / "FileA_A4.wav", corpus / "FileB_A5.wav"],
        main_analysis_output_dir=analysis_results,
    )
    assert orch.run_stage2_compilation() is True

    final_wb = analysis_results / "compiled_density_metrics.xlsx"
    assert final_wb.is_file(), f"missing root workbook: {final_wb}"

    xl = pd.ExcelFile(final_wb)
    assert "Density_Metrics" in xl.sheet_names
    assert "Analysis_Metadata" in xl.sheet_names
    dm = pd.read_excel(final_wb, sheet_name="Density_Metrics")
    assert len(dm) >= 2


def test_compile_nested_spectral_under_analysis_results(tmp_path: Path) -> None:
    """Direct compile_metrics call: nested spectral_analysis.xlsx under one root."""
    from compile_metrics import compile_density_metrics_with_pca

    root = tmp_path / "analysis_results"
    root.mkdir(parents=True, exist_ok=True)
    _write_minimal_spectral_analysis_xlsx(root / "stem1" / "A4" / "spectral_analysis.xlsx", seed=0.5)
    _write_minimal_spectral_analysis_xlsx(root / "stem2" / "C3" / "spectral_analysis.xlsx", seed=1.5)
    outp = root / "compiled_density_metrics.xlsx"
    df = compile_density_metrics_with_pca(
        folder_path=root,
        output_path=str(outp),
        file_pattern="spectral_analysis.xlsx",
        include_pca=False,
        enable_pca_export=False,
        minimum_samples_for_pca=99,
    )
    assert df is not None and not df.empty
    assert outp.is_file()
