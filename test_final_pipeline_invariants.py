# -*- coding: utf-8 -*-
"""Final compiled-workbook invariant tests (canonical Stage 2)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from compile_metrics import compile_density_metrics_with_pca  # noqa: E402
from proc_audio import AudioProcessor  # noqa: E402
from tests.pipeline_workbook_audit import run_audit_on_workbook  # noqa: E402


def _build_minimal_canonical_pipeline(tmp_path: Path) -> tuple[Path, Path]:
    """proc_audio → spectral_analysis.xlsx → compile → compiled_density_metrics.xlsx."""
    sr = 44100
    t = np.linspace(0, 0.35, int(sr * 0.35), endpoint=False)
    y = sum(np.sin(2 * np.pi * k * 440.0 * t) for k in range(1, 5))
    y = 0.45 * y / (np.max(np.abs(y)) + 1e-12)
    wav = tmp_path / "A4.wav"
    sf.write(str(wav), y.astype(np.float32), sr)

    out_dir = tmp_path / "results"
    ap = AudioProcessor()
    ap.load_audio_files([str(wav)])
    ap.apply_filters_and_generate_data(
        freq_min=50.0,
        freq_max=12000.0,
        db_min=-90.0,
        db_max=0.0,
        window="blackmanharris",
        n_fft=8192,
        hop_length=1024,
        tolerance=10.0,
        use_adaptive_tolerance=True,
        results_directory=str(out_dir),
        dissonance_enabled=False,
        dissonance_curve=False,
        dissonance_scale=False,
        compare_models=False,
        harmonic_weight=0.5,
        inharmonic_weight=0.5,
        auto_model_weights_from_analysis=True,
        weight_function="linear",
        zero_padding=1,
        time_avg="mean",
        spectral_masking_enabled=False,
        tier="t",
    )
    per_note = out_dir / "A4" / "spectral_analysis.xlsx"
    assert per_note.is_file()
    compiled = tmp_path / "compiled_density_metrics.xlsx"
    compile_density_metrics_with_pca(
        folder_path=str(out_dir),
        output_path=str(compiled),
        file_pattern="spectral_analysis.xlsx",
        enable_pca_export=False,
        minimum_samples_for_pca=99,
        allow_legacy_super_json=False,
    )
    assert compiled.is_file()
    return compiled, per_note


def test_fixture_compiled_workbook_passes_audit(tmp_path: Path) -> None:
    compiled, per_note = _build_minimal_canonical_pipeline(tmp_path)
    rep = run_audit_on_workbook(compiled, per_note_workbook=per_note)
    assert rep.ok, rep.hard_failures


@pytest.mark.skipif(not os.environ.get("SSA_COMPILED_WORKBOOK"), reason="SSA_COMPILED_WORKBOOK not set")
def test_env_compiled_workbook_audit() -> None:
    """Optional integration check: ``SSA_COMPILED_WORKBOOK`` → compiled xlsx.

    Set ``SSA_PER_NOTE_WORKBOOK`` to one ``spectral_analysis.xlsx`` to include
    Harmonic Spectrum interpolation audits (same checks as the CLI second argument).
    """
    p = Path(os.environ["SSA_COMPILED_WORKBOOK"])
    assert p.is_file(), p
    per = os.environ.get("SSA_PER_NOTE_WORKBOOK")
    per_p = Path(per) if per else None
    rep = run_audit_on_workbook(p, per_note_workbook=per_p)
    assert rep.ok, rep.hard_failures
