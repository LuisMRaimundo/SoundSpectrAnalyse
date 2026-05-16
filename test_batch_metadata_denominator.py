# -*- coding: utf-8 -*-
"""DEPRECATED tests: exercise legacy Stage 1 / Batch metadata denominators
and external H/I/S projections. The current pipeline derives the
``component_*_energy_ratio`` and model weights from the current per-note
spectrum exclusively, so these checks no longer reflect runtime behaviour.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

pytestmark = pytest.mark.skip(
    reason=(
        "deprecated: tests removed Stage 1 / Batch metadata pathway; the new "
        "pipeline uses component_*_energy_ratio sourced from current_analysis."
    )
)

from compile_metrics import _write_compiled_excel
from pipeline_orchestrator_integrated import RobustOrchestrator


def _global_his_ratios(h_e: float, i_e: float, s_e: float) -> tuple[float, float, float, float]:
    t = h_e + i_e + s_e
    assert t > 0
    return h_e / t, i_e / t, s_e / t, (i_e + s_e) / t


def test_case_a_batch_ratio_denominator() -> None:
    bh, bi, bs, tih = _global_his_ratios(95.0, 5.0, 5.0)
    assert math.isclose(bh, 95.0 / 105.0)
    assert math.isclose(bi, 5.0 / 105.0)
    assert math.isclose(bs, 5.0 / 105.0)
    assert math.isclose(bh + bi + bs, 1.0)
    assert math.isclose(tih, bi + bs)


def test_case_b_tiny_subbass_sum() -> None:
    bh, bi, bs, _ = _global_his_ratios(956.49, 43.51, 0.064)
    assert math.isclose(bh + bi + bs, 1.0, rel_tol=0, abs_tol=1e-9)


def test_case_c_model_projection_from_global_ratios() -> None:
    H, I, S = 0.90, 0.05, 0.05
    mh, mi = RobustOrchestrator._derive_model_weights_from_batch_energy(H, I, S)
    assert mh is not None and mi is not None
    assert math.isclose(mh, 0.90 / 0.95)
    assert math.isclose(mi, 0.05 / 0.95)
    assert math.isclose(mh + mi, 1.0)


def test_case_d_per_note_metadata_not_on_density_sheet(tmp_path: Path) -> None:
    rows = []
    for i in range(3):
        rows.append(
            {
                "Note": f"N{i}",
                "weight_function": "linear",
                "Harmonic Partials sum": 1.0,
                "Inharmonic Partials sum": 0.2,
                "Sub-bass sum": 0.05,
                "Total sum": 1.25,
                "effective_partial_density": 1.1 + 0.01 * i,
                "harmonic_energy_sum": 1.0,
                "inharmonic_energy_sum": 0.2,
                "subbass_energy_sum": 0.05,
                "total_component_energy": 1.25,
                "harmonic_energy_ratio": 0.8,
                "inharmonic_energy_ratio": 0.15,
                "subbass_energy_ratio": 0.05,
                "harmonic_order_count": 3,
                "batch_harmonic_energy_ratio": 0.8,
                "batch_inharmonic_energy_ratio": 0.15,
                "batch_subbass_energy_ratio": 0.05,
                "model_harmonic_weight": 0.8 / 0.95,
                "model_inharmonic_weight": 0.15 / 0.95,
                "model_weights_source": "batch_empirical_energy_ratios",
                "n_fft": 2048,
                "hop_length": 256,
                "window": "hann",
            }
        )
    df = pd.DataFrame(rows)
    outp = tmp_path / "m.xlsx"
    _write_compiled_excel(outp, df, {"analysis_version": "test"}, enable_pca_export=False)
    dm = pd.read_excel(outp, sheet_name="Density_Metrics")
    pn = pd.read_excel(outp, sheet_name="Per_Note_Processing_Metadata")
    assert "batch_harmonic_energy_ratio" not in dm.columns
    assert "batch_harmonic_energy_ratio" in pn.columns
    assert "model_harmonic_weight" in pn.columns


def test_case_e_selected_dissonance_model_in_analysis_metadata(tmp_path: Path) -> None:
    rows = []
    for i in range(5):
        rows.append(
            {
                "Note": f"D{i}",
                "weight_function": "linear",
                "Harmonic Partials sum": 1.0,
                "Inharmonic Partials sum": 0.2,
                "Sub-bass sum": 0.05,
                "Total sum": 1.25,
                "effective_partial_density": 1.0 + 0.01 * i,
                "harmonic_energy_sum": 1.0,
                "inharmonic_energy_sum": 0.2,
                "subbass_energy_sum": 0.05,
                "total_component_energy": 1.25,
                "harmonic_energy_ratio": 0.8,
                "inharmonic_energy_ratio": 0.15,
                "subbass_energy_ratio": 0.05,
                "harmonic_order_count": 3,
                "selected_dissonance_model": "sethares",
            }
        )
    df = pd.DataFrame(rows)
    outp = tmp_path / "d.xlsx"
    meta = _write_compiled_excel(outp, df, {}, enable_pca_export=False)
    assert str(meta.get("selected_dissonance_model", "")).lower() == "sethares"
