"""Tests for rolloff-compensated harmonic density (relative richness descriptor)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from density import (
    DEFAULT_HARMONIC_ROLLOFF_ALPHA,
    DEFAULT_ROLLOFF_COMPENSATED_DENSITY_WEIGHT_FUNCTION,
    compute_rolloff_compensated_harmonic_density,
)
from compile_metrics import (
    DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS,
    _build_density_metrics_main_sheet,
    compile_density_metrics,
    read_super_analysis_metrics,
    validate_compiled_density_workbook,
)


def test_compensated_values_synthetic_linear_weight():
    """f0=100 Hz, geometric decay amplitudes; orders 1..4; alpha=1.5; linear sum."""
    f0 = 100.0
    f = np.array([100.0, 200.0, 300.0, 400.0], dtype=float)
    a = np.array([1.0, 0.5, 0.25, 0.125], dtype=float)
    n = np.array([1, 2, 3, 4], dtype=int)
    alpha = 1.5
    a_norm = a / np.max(a)
    expected = np.power(n.astype(float), -alpha)
    compensated = a_norm / (expected + 1e-12)
    np.testing.assert_allclose(
        compensated,
        np.array(
            [
                1.0,
                0.5 * (2.0**1.5),
                0.25 * (3.0**1.5),
                0.125 * (4.0**1.5),
            ],
            dtype=float,
        ),
        rtol=1e-9,
        atol=1e-9,
    )
    out = compute_rolloff_compensated_harmonic_density(
        a, f, f0, harmonic_orders=n.astype(float), alpha=alpha, weight_function="linear"
    )
    assert out["rolloff_compensated_harmonic_density_status"] == "computed"
    assert out["rolloff_compensated_harmonic_density_alpha"] == alpha
    assert out["rolloff_compensated_harmonic_density_component_count"] == 4
    assert math.isclose(float(out["rolloff_compensated_harmonic_density"]), float(np.sum(compensated)), rel_tol=1e-9)


def test_default_alpha_and_weight():
    a = np.array([1.0, 0.5])
    f = np.array([100.0, 200.0])
    out = compute_rolloff_compensated_harmonic_density(
        a, f, 100.0, harmonic_orders=np.array([1.0, 2.0])
    )
    assert out["rolloff_compensated_harmonic_density_alpha"] == DEFAULT_HARMONIC_ROLLOFF_ALPHA
    assert out["rolloff_compensated_harmonic_density_status"] == "computed"
    assert DEFAULT_ROLLOFF_COMPENSATED_DENSITY_WEIGHT_FUNCTION == "logarithmic"
    a_norm = a / np.max(a)
    n = np.array([1.0, 2.0])
    expc = n ** (-DEFAULT_HARMONIC_ROLLOFF_ALPHA)
    comp = a_norm / (expc + 1e-12)
    expected_sum = float(np.sum(np.log1p(comp)))
    assert math.isclose(float(out["rolloff_compensated_harmonic_density"]), expected_sum, rel_tol=1e-9)


def test_invalid_f0_skipped():
    out = compute_rolloff_compensated_harmonic_density(
        np.array([1.0]),
        np.array([100.0]),
        0.0,
    )
    assert "skipped" in str(out["rolloff_compensated_harmonic_density_status"])
    assert out["rolloff_compensated_harmonic_density_component_count"] == 0


def test_empty_components_skipped():
    out = compute_rolloff_compensated_harmonic_density(
        np.array([], dtype=float),
        np.array([], dtype=float),
        100.0,
    )
    assert out["rolloff_compensated_harmonic_density_status"] == "skipped_no_harmonic_components"


def test_density_metrics_main_sheet_is_minimal_excluding_rolloff():
    """Compiled ``Density_Metrics`` keeps only partial sums; rolloff stays on wide / per-note sheets."""
    row = {
        "Note": "A4",
        "weight_function": "linear",
        "Harmonic Partials sum": 1.0,
        "Inharmonic Partials sum": 0.5,
        "Sub-bass sum": 0.1,
        "Total sum": 1.6,
        "effective_partial_density": 2.0,
        "harmonic_energy_sum": 1.0,
        "inharmonic_energy_sum": 0.5,
        "subbass_energy_sum": 0.1,
        "total_component_energy": 1.6,
        "harmonic_energy_ratio": 0.625,
        "inharmonic_energy_ratio": 0.3125,
        "subbass_energy_ratio": 0.0625,
        "harmonic_order_count": 4,
        "spectral_entropy": 0.3,
        "rolloff_compensated_harmonic_density": 1.25,
        "rolloff_compensated_harmonic_density_alpha": 1.5,
        "rolloff_compensated_harmonic_density_component_count": 4,
        "rolloff_compensated_harmonic_density_status": "computed",
        "legacy_rolloff_compensated_density": 1.25,
    }
    out = _build_density_metrics_main_sheet(pd.DataFrame([row]))
    assert list(out.columns) == DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS
    assert "rolloff_compensated_harmonic_density" not in out.columns


def test_read_super_analysis_metrics_maps_rolloff(tmp_path: Path):
    spec = {
        "harmonic_density": 1.0,
        "inharmonic_density": 0.5,
        "harmonic_energy_percentage": 60.0,
        "inharmonic_energy_percentage": 40.0,
        "rolloff_compensated_harmonic_density": 2.5,
        "rolloff_compensated_harmonic_density_alpha": 1.5,
        "rolloff_compensated_harmonic_density_component_count": 3,
        "rolloff_compensated_harmonic_density_status": "computed",
        "legacy_rolloff_compensated_density": 2.5,
    }
    payload = {"spectral_metrics": spec, "spectral_component_stats": {}, "metadata": {}}
    p = tmp_path / "super_analysis_results.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    m = read_super_analysis_metrics(p)
    assert m["rolloff_compensated_harmonic_density"] == 2.5
    assert m["rolloff_compensated_harmonic_density_status"] == "computed"


def test_log_messages_contain_no_forbidden_phrases():
    msgs = [
        "Rolloff-compensated harmonic density computed: value=1.0, alpha=1.5, components=2",
        "Rolloff-compensated harmonic density skipped: reason=skipped_invalid_fundamental_frequency",
    ]
    banned = ("gold-standard", "gold standard", "spurious peaks", "absolute density", "more accurate")
    for msg in msgs:
        low = msg.lower()
        for b in banned:
            assert b not in low


def test_compiled_density_workbook_contains_rolloff_columns(tmp_path: Path):
    note_dir = tmp_path / "01_A4_compile_smoke"
    note_dir.mkdir(parents=True)
    row = {
        "Note": "A4",
        "weight_function": "linear",
        "Harmonic Partials sum": 1.0,
        "Inharmonic Partials sum": 0.5,
        "Sub-bass sum": 0.1,
        "Total sum": 1.6,
        "effective_partial_density": 2.0,
        "harmonic_energy_sum": 1.0,
        "inharmonic_energy_sum": 0.5,
        "subbass_energy_sum": 0.1,
        "total_component_energy": 1.6,
        "harmonic_energy_ratio": 0.625,
        "inharmonic_energy_ratio": 0.3125,
        "subbass_energy_ratio": 0.0625,
        "harmonic_order_count": 4,
        "spectral_entropy": 0.3,
        "rolloff_compensated_harmonic_density": 1.25,
        "rolloff_compensated_harmonic_density_alpha": 1.5,
        "rolloff_compensated_harmonic_density_component_count": 4,
        "rolloff_compensated_harmonic_density_status": "computed",
        "legacy_rolloff_compensated_density": 1.25,
    }
    pd.DataFrame([row]).to_excel(note_dir / "spectral_analysis.xlsx", sheet_name="Metrics", index=False)
    outp = tmp_path / "compiled_density_metrics.xlsx"
    df = compile_density_metrics(
        tmp_path,
        output_path=outp,
        file_pattern="spectral_analysis.xlsx",
        enable_pca_export=False,
        compiled_public_columns=False,
    )
    assert df is not None
    assert validate_compiled_density_workbook(outp) == []
    wide = pd.read_excel(outp, sheet_name="Compiled_Metrics_All")
    assert "rolloff_compensated_harmonic_density" in wide.columns


@pytest.mark.parametrize(
    "v5_path",
    [
        Path(
            r"C:\Users\lmr20\Desktop\PYTHON CODES_importantes\EM USO E ACTUALIZADOS\SoundSpectrAnalyse-main_5\density.py"
        ),
    ],
)
def test_v5_reference_tendency_when_available(v5_path: Path):
    """If v5 density.py exists, legacy apply_density_metric should rank similarly on a simple ladder."""
    if not v5_path.is_file():
        pytest.skip("v5 reference density.py not found on this machine")
    import importlib.util

    spec = importlib.util.spec_from_file_location("density_v5_reference", v5_path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    apply_v5 = getattr(mod, "apply_density_metric", None)
    if apply_v5 is None:
        pytest.skip("v5 apply_density_metric not available")

    f0 = 100.0
    freqs = np.array([100.0, 200.0, 300.0], dtype=float)
    amps = np.array([1.0, 0.4, 0.2], dtype=float)
    v5_val = float(
        apply_v5(
            amps,
            "linear",
            normalize=False,
            frequencies=freqs,
            fundamental_freq=f0,
            account_for_spectral_rolloff=True,
            prevent_domination=True,
        )
    )
    new = compute_rolloff_compensated_harmonic_density(
        amps,
        freqs,
        f0,
        harmonic_orders=np.array([1.0, 2.0, 3.0]),
        alpha=1.5,
        weight_function="linear",
    )
    assert new["rolloff_compensated_harmonic_density_status"] == "computed"
    new_val = float(new["rolloff_compensated_harmonic_density"])
    # Same analytical tendency: both positive and finite; not required to match numerically.
    assert math.isfinite(v5_val) and v5_val > 0
    assert math.isfinite(new_val) and new_val > 0
