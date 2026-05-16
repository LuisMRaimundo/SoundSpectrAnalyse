"""Data-integrity helpers: missing vs real-zero semantics (export / compile)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_metric_float_or_nan_preserves_missing_and_zero() -> None:
    from data_integrity import metric_float_or_nan

    assert metric_float_or_nan(0.0) == 0.0
    assert math.isnan(metric_float_or_nan(None))
    assert math.isnan(metric_float_or_nan(float("nan")))
    assert math.isnan(metric_float_or_nan("not_computed"))
    assert metric_float_or_nan("3.5") == 3.5


def test_metric_int_or_nan() -> None:
    from data_integrity import metric_int_or_nan

    assert pd.isna(metric_int_or_nan(None))
    assert metric_int_or_nan(0) == 0
    assert metric_int_or_nan(3.7) == 3


def test_robust_normalize_all_non_finite() -> None:
    from data_integrity import robust_normalize

    out = robust_normalize(np.array([float("nan"), float("inf"), float("-inf")]))
    assert out.shape == (3,)
    assert all(math.isnan(float(x)) for x in out)


def test_apply_weighted_index_renormalizes_missing_terms() -> None:
    from compile_metrics import apply_weighted_index

    df = pd.DataFrame(
        {
            "Note": ["A#4"],
            "Density Metric_Norm2": [1.0],
            "N_harm_norm": [1.0],
            "P_norm": [1.0],
        }
    )
    out = apply_weighted_index(df, scheme="pdf")
    # Available weights: 0.10 + 0.30 + 0.05 = 0.45 → renormalized index 1.0
    assert np.isclose(float(out.loc[0, "Index_Weighted"]), 1.0)
    assert out.loc[0, "Index_Weighted_status"] == "computed_renormalized_available_terms"


def test_missing_density_metric_norm2_is_nan_not_zero() -> None:
    from compile_metrics import apply_weighted_index

    df2 = pd.DataFrame({"Note": ["A#4"], "D_agn": [0.5], "P_norm": [0.5]})
    out2 = apply_weighted_index(df2, scheme="pdf")
    assert math.isnan(float(out2.loc[0, "Density Metric_Norm2"]))
    assert bool(out2.loc[0, "Density Metric_Norm2_available"]) is False


def test_build_main_metrics_export_row_nan_when_not_computed() -> None:
    from proc_audio import AudioProcessor

    p = AudioProcessor()
    p.weight_function = "linear"
    p.effective_partial_density = None
    p.harmonic_energy_sum = None
    p.density_formula_version = None
    p.density_source_formula = None
    p.density_normalization_scope = None
    row = p._build_main_metrics_export_row(
        "A#4", h_psum=None, i_psum=None, s_psum=None, t_psum=None
    )
    assert math.isnan(float(row["effective_partial_density"]))
    assert math.isnan(float(row["harmonic_energy_sum"]))
    assert row["component_energy_status"] == "not_computed"


def test_debug_counts_uses_missing_not_zero() -> None:
    from data_integrity import metric_int_or_nan

    assert pd.isna(metric_int_or_nan(None))
    v = metric_int_or_nan(0)
    assert v == 0
