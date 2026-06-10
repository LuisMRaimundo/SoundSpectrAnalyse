from __future__ import annotations

"""
Additional contract-level coverage for data_integrity.py.

Public APIs under test:
- metric export sanitizers (``metric_float_or_nan``, ``metric_int_or_nan``,
  ``metric_ratio_or_nan``, ``metric_series_or_nan``);
- robust statistics (IQR bounds, outlier detection, ``robust_normalize``);
- ``GlobalReferenceScaler`` fit/transform semantics;
- metric and audio-parameter validation helpers;
- ``normalize_log_transform`` for wide-dynamic-range columns.

Focus areas (no production code changes):
- missing/non-finite/invalid scalar coercion;
- empty and all-non-finite array degeneracy;
- normalization method branches and clip/no-clip contracts;
- validation diagnostics (formal error tokens, stats dict keys);
- determinism and input non-mutation;
- boundary audio-parameter rejection paths.

Exact assertions are used only for canonical arithmetic implied directly by
the implementation (IQR multipliers, ratio guards, clip ranges).
"""

import math

import numpy as np
import pandas as pd
import pytest

from data_integrity import (
    MISSING_FLOAT,
    GlobalReferenceScaler,
    calculate_iqr_bounds,
    detect_outliers,
    metric_float_or_nan,
    metric_int_or_nan,
    metric_ratio_or_nan,
    metric_series_or_nan,
    normalize_log_transform,
    robust_normalize,
    validate_audio_parameters,
    validate_metric_array,
    validate_metric_value,
)


_STATS_KEYS = frozenset(
    {"mean", "median", "std", "min", "max", "nan_count", "inf_count"}
)


def _assert_all_nan(series: pd.Series) -> None:
    assert len(series) == len(series.index)
    assert series.isna().all()


# ---------------------------------------------------------------------------
# 1. Metric export sanitizers
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        (None, True),
        ("not-a-number", True),
        (float("inf"), True),
        (float("-inf"), True),
        ("", True),
    ],
)
def test_metric_float_or_nan_maps_missing_and_nonfinite_to_nan(
    raw: object, expected: bool
) -> None:
    out = metric_float_or_nan(raw)
    assert math.isnan(out) is expected


def test_metric_float_or_nan_preserves_real_zero_and_finite_values() -> None:
    assert metric_float_or_nan(0) == 0.0
    assert metric_float_or_nan(0.0) == 0.0
    assert metric_float_or_nan("3.5") == pytest.approx(3.5)
    assert metric_float_or_nan(-2.25) == pytest.approx(-2.25)


@pytest.mark.parametrize(
    "raw",
    [None, "bad", float("inf"), float("nan")],
)
def test_metric_int_or_nan_maps_invalid_to_pd_na(raw: object) -> None:
    assert pd.isna(metric_int_or_nan(raw))


def test_metric_int_or_nan_truncates_toward_zero() -> None:
    assert metric_int_or_nan(3.9) == 3
    assert metric_int_or_nan(-3.9) == -3
    assert metric_int_or_nan(0.0) == 0


def test_metric_ratio_or_nan_requires_finite_positive_denominator() -> None:
    assert math.isnan(metric_ratio_or_nan(2.0, 0.0))
    assert math.isnan(metric_ratio_or_nan(2.0, -1.0))
    assert math.isnan(metric_ratio_or_nan(float("nan"), 1.0))
    assert math.isnan(metric_ratio_or_nan(1.0, float("inf")))
    assert metric_ratio_or_nan(1.0, 4.0) == pytest.approx(0.25)


def test_metric_series_or_nan_missing_column_is_all_nan() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out = metric_series_or_nan(df, "missing")
    _assert_all_nan(out)
    assert list(out.index) == list(df.index)


def test_metric_series_or_nan_coerces_present_column() -> None:
    df = pd.DataFrame({"x": ["1.0", "bad", "3"]})
    out = metric_series_or_nan(df, "x")
    assert out.iloc[0] == pytest.approx(1.0)
    assert math.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# 2. IQR bounds and outlier detection
# ---------------------------------------------------------------------------

def test_calculate_iqr_bounds_empty_and_all_nonfinite_return_zeros() -> None:
    assert calculate_iqr_bounds(np.array([])) == (0.0, 0.0, 0.0, 0.0)
    assert calculate_iqr_bounds(np.array([np.nan, np.inf])) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )


def test_calculate_iqr_bounds_tukey_1_5_multiplier() -> None:
    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    q1, q3, lower, upper = calculate_iqr_bounds(data, iqr_multiplier=1.5)
    assert q1 == pytest.approx(2.0)
    assert q3 == pytest.approx(4.0)
    iqr = q3 - q1
    assert lower == pytest.approx(q1 - 1.5 * iqr)
    assert upper == pytest.approx(q3 + 1.5 * iqr)


def test_detect_outliers_empty_arrays_and_mask_mode() -> None:
    assert detect_outliers(np.array([])).size == 0
    outliers, mask = detect_outliers(np.array([]), return_mask=True)
    assert outliers.size == 0
    assert mask.size == 0
    assert mask.dtype == bool

    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    flagged, mask = detect_outliers(values, return_mask=True)
    assert flagged.tolist() == [100.0]
    assert mask.tolist() == [False, False, False, False, False, True]


def test_detect_outliers_value_mode_returns_only_outlier_values() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    outliers = detect_outliers(values)
    assert outliers.tolist() == [100.0]


# ---------------------------------------------------------------------------
# 3. robust_normalize — methods, degeneracy, copy semantics
# ---------------------------------------------------------------------------

def test_robust_normalize_empty_returns_empty_array() -> None:
    out = robust_normalize(np.array([]))
    assert out.size == 0


def test_robust_normalize_all_nonfinite_returns_nan_positions() -> None:
    src = np.array([np.nan, np.inf, -np.inf])
    out = robust_normalize(src)
    assert out.shape == src.shape
    assert np.isnan(out).all()


def test_robust_normalize_identical_finite_values_map_to_zero() -> None:
    src = np.array([7.0, 7.0, 7.0])
    out = robust_normalize(src, method="iqr")
    assert np.allclose(out, 0.0)


@pytest.mark.parametrize("method", ["iqr", "percentile", "robust_zscore"])
def test_robust_normalize_methods_produce_finite_clipped_outputs_in_unit_interval(
    method: str,
) -> None:
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = robust_normalize(src, method=method)
    finite = out[np.isfinite(src)]
    assert np.all(np.isfinite(finite))
    assert finite.min() >= 0.0
    assert finite.max() <= 1.0
    assert finite[0] < finite[-1]


def test_robust_normalize_robust_zscore_identical_finite_values_map_to_zero() -> None:
    src = np.array([4.0, 4.0, 4.0])
    out = robust_normalize(src, method="robust_zscore")
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_robust_normalize_unknown_method_identical_values_map_to_zero() -> None:
    src = np.array([2.0, 2.0, 2.0])
    out = robust_normalize(src, method="legacy-minmax")
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_global_reference_scaler_iqr_collapsed_bounds_map_to_zero() -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([3.0, 3.0, 3.0]), method="iqr")
    out = scaler.transform(np.array([3.0, 4.0]))
    assert out.tolist() == [0.0, 0.0]


def test_global_reference_scaler_mean_std_positive_std_transform() -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), method="mean_std")
    out = scaler.transform(np.array([2.0, 5.0]))
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[1] <= 1.0
    assert out[0] < out[1]


def test_global_reference_scaler_minmax_transform_and_collapsed_reference() -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([1.0, 2.0, 3.0]), method="minmax")
    out = scaler.transform(np.array([2.0]))
    assert out.tolist() == pytest.approx([0.5])

    collapsed = GlobalReferenceScaler()
    collapsed.fit(np.array([2.0, 2.0, 2.0]), method="minmax")
    assert collapsed.transform(np.array([2.0, 3.0])).tolist() == [0.0, 0.0]


def test_robust_normalize_percentile_identical_finite_values_map_to_zero() -> None:
    src = np.array([6.0, 6.0, 6.0])
    out = robust_normalize(src, method="percentile")
    assert out.tolist() == [0.0, 0.0, 0.0]


def test_robust_normalize_percentile_method_spans_observed_range_to_unit_interval() -> None:
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = robust_normalize(src, method="percentile")
    finite = out[np.isfinite(src)]
    assert finite.min() == pytest.approx(0.0, abs=1e-6)
    assert finite.max() == pytest.approx(1.0, abs=1e-6)


def test_robust_normalize_unknown_method_falls_back_to_minmax() -> None:
    src = np.array([0.0, 5.0, 10.0])
    out = robust_normalize(src, method="not-a-method")
    assert out.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_robust_normalize_preserves_nonfinite_positions_and_does_not_mutate_input() -> None:
    src = np.array([1.0, np.nan, 3.0, 100.0])
    snapshot = src.copy()
    out = robust_normalize(src, method="iqr")
    assert np.array_equal(src, snapshot, equal_nan=True)
    assert math.isnan(out[1])
    assert out[0] <= out[2] <= out[3]


def test_robust_normalize_clip_none_allows_values_outside_unit_interval() -> None:
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    out = robust_normalize(src, method="iqr", clip_range=None)
    assert out.max() > 1.0


def test_robust_normalize_is_deterministic() -> None:
    src = np.array([0.1, 0.5, 0.9, 1.1, 4.0])
    a = robust_normalize(src, method="percentile")
    b = robust_normalize(src, method="percentile")
    assert np.array_equal(a, b, equal_nan=True)


# ---------------------------------------------------------------------------
# 4. GlobalReferenceScaler
# ---------------------------------------------------------------------------

def test_global_reference_scaler_unfitted_transform_delegates_to_local_iqr() -> None:
    scaler = GlobalReferenceScaler()
    src = np.array([1.0, 2.0, 3.0])
    out = scaler.transform(src)
    expected = robust_normalize(src, method="iqr")
    assert np.allclose(out, expected, equal_nan=True)


def test_global_reference_scaler_fit_empty_or_all_nonfinite_leaves_stats_none() -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([]))
    assert scaler.reference_stats is None

    scaler2 = GlobalReferenceScaler()
    scaler2.fit(np.array([np.nan, np.inf]))
    assert scaler2.reference_stats is None


@pytest.mark.parametrize(
    "method, expected_keys",
    [
        ("percentile", {"p5", "p95", "median", "mean"}),
        ("iqr", {"Q1", "Q3", "lower_bound", "upper_bound", "median"}),
        ("mean_std", {"mean", "std", "median"}),
        ("minmax", {"min", "max", "mean"}),
    ],
)
def test_global_reference_scaler_fit_records_method_specific_stats(
    method: str, expected_keys: set[str]
) -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), method=method)
    assert scaler.method == method
    assert scaler.reference_stats is not None
    assert set(scaler.reference_stats.keys()) == expected_keys


def test_global_reference_scaler_transform_empty_returns_empty() -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([1.0, 2.0, 3.0]))
    assert scaler.transform(np.array([])).size == 0


def test_global_reference_scaler_transform_all_nonfinite_returns_zeros() -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), method="percentile")
    out = scaler.transform(np.array([np.nan, np.inf]))
    assert out.tolist() == [0.0, 0.0]


def test_global_reference_scaler_transform_collapsed_reference_maps_to_zero() -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([5.0, 5.0, 5.0]), method="percentile")
    out = scaler.transform(np.array([5.0, 6.0]))
    assert out.tolist() == [0.0, 0.0]


def test_global_reference_scaler_mean_std_zero_std_branch() -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([2.0, 2.0, 2.0]), method="mean_std")
    out = scaler.transform(np.array([2.0, 3.0]))
    assert out.tolist() == [0.0, 0.0]


def test_global_reference_scaler_fit_transform_matches_fit_then_transform() -> None:
    src = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    scaler = GlobalReferenceScaler()
    combined = scaler.fit_transform(src, method="percentile")
    reference = GlobalReferenceScaler()
    reference.fit(src, method="percentile")
    split = reference.transform(src)
    assert np.allclose(combined, split, equal_nan=True)


def test_global_reference_scaler_transform_preserves_nonfinite_positions() -> None:
    scaler = GlobalReferenceScaler()
    scaler.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), method="iqr")
    src = np.array([2.0, np.nan, 4.0])
    out = scaler.transform(src)
    assert math.isnan(out[1])
    assert 0.0 <= out[0] <= 1.0
    assert 0.0 <= out[2] <= 1.0


# ---------------------------------------------------------------------------
# 5. validate_metric_value / validate_metric_array
# ---------------------------------------------------------------------------

def test_validate_metric_value_nan_and_inf_diagnostics() -> None:
    ok, err = validate_metric_value(float("nan"), "density_metric")
    assert ok is False
    assert err == "density_metric is NaN or None"

    ok_inf, err_inf = validate_metric_value(float("inf"), "density_metric")
    assert ok_inf is False
    assert err_inf == "density_metric is Inf"


def test_validate_metric_value_allow_flags_and_range() -> None:
    assert validate_metric_value(float("nan"), "m", allow_nan=True) == (True, None)
    assert validate_metric_value(float("inf"), "m", allow_inf=True) == (True, None)

    ok, err = validate_metric_value(1.5, "m", expected_range=(0.0, 1.0))
    assert ok is False
    assert "outside expected range" in err
    assert validate_metric_value(0.5, "m", expected_range=(0.0, 1.0)) == (True, None)


def test_validate_metric_array_empty_and_all_nonfinite() -> None:
    ok, err, stats = validate_metric_array(np.array([]), "m")
    assert ok is False
    assert err == "m array is empty"
    assert stats == {}

    ok2, err2, stats2 = validate_metric_array(np.array([np.nan, np.inf]), "m")
    assert ok2 is False
    assert err2 == "m array has no finite values"
    assert stats2 == {}


def test_validate_metric_array_success_stats_schema_and_outlier_gate() -> None:
    values = np.array([0.1, 0.2, 0.3, 0.4])
    ok, err, stats = validate_metric_array(values, "m", expected_range=(0.0, 1.0))
    assert ok is True
    assert err is None
    assert set(stats.keys()) == _STATS_KEYS
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0

    noisy = np.array([0.1, 0.2, 0.3, 0.4, 10.0])
    bad, msg, stats_bad = validate_metric_array(
        noisy, "m", max_outlier_fraction=0.1
    )
    assert bad is False
    assert msg is not None
    assert "outliers" in msg
    assert set(stats_bad.keys()) == _STATS_KEYS

    ok_loose, _, _ = validate_metric_array(noisy, "m", max_outlier_fraction=0.5)
    assert ok_loose is True


def test_validate_metric_array_range_violation_reports_count() -> None:
    values = np.array([0.1, 0.2, 1.5])
    ok, err, stats = validate_metric_array(
        values, "m", expected_range=(0.0, 1.0), max_outlier_fraction=0.0
    )
    assert ok is False
    assert err == "m has 1 values outside range [0.0, 1.0]"
    assert set(stats.keys()) == _STATS_KEYS


def test_validate_metric_array_counts_nan_and_inf_in_stats() -> None:
    values = np.array([0.1, np.nan, np.inf, 0.2])
    ok, _, stats = validate_metric_array(values, "m", max_outlier_fraction=1.0)
    assert ok is True
    assert stats["nan_count"] == 1
    assert stats["inf_count"] == 1


# ---------------------------------------------------------------------------
# 6. validate_audio_parameters
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "n_fft, hop, sr, length, token",
    [
        (32, 512, 44100, 10000, "too small"),
        (70000, 512, 44100, 80000, "too large"),
        (0, 512, 44100, 10000, "too small"),
        (2048, 0, 44100, 10000, "must be positive"),
        (2048, 3000, 44100, 10000, "hop_length"),
        (2048, 512, 7000, 10000, "too low"),
        (2048, 512, 200000, 10000, "too high"),
        (2048, 512, 44100, 100, "Signal length"),
    ],
)
def test_validate_audio_parameters_rejects_out_of_contract_values(
    n_fft: int, hop: int, sr: int, length: int, token: str
) -> None:
    ok, err = validate_audio_parameters(n_fft, hop, sr, length)
    assert ok is False
    assert err is not None
    assert token in err


def test_validate_audio_parameters_accepts_canonical_fft_block() -> None:
    ok, err = validate_audio_parameters(2048, 512, 44100, 44100)
    assert ok is True
    assert err is None


def test_validate_audio_parameters_non_power_of_two_still_passes() -> None:
    ok, err = validate_audio_parameters(1000, 256, 44100, 20000)
    assert ok is True
    assert err is None


# ---------------------------------------------------------------------------
# 7. normalize_log_transform
# ---------------------------------------------------------------------------

def test_validate_audio_parameters_coarse_resolution_emits_warning_but_passes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        ok, err = validate_audio_parameters(128, 64, 44100, 10000)
    assert ok is True
    assert err is None
    assert any("Frequency resolution" in rec.message for rec in caplog.records)


def test_normalize_log_transform_empty_returns_empty() -> None:
    assert normalize_log_transform(np.array([])).size == 0


def test_normalize_log_transform_coerces_object_and_preserves_nonfinite() -> None:
    src = np.array(["1", "2", "bad"], dtype=object)
    out = normalize_log_transform(src)
    assert out.shape == src.shape
    assert out[0] == pytest.approx(0.0)
    assert out[1] == pytest.approx(1.0)
    assert math.isnan(out[2])


def test_normalize_log_transform_all_nonfinite_or_zero_maps_to_zeros() -> None:
    zeros = normalize_log_transform(np.array([0.0, 0.0, 0.0]))
    assert zeros.tolist() == [0.0, 0.0, 0.0]

    all_bad = normalize_log_transform(np.array([np.nan, "x"]))
    assert all_bad.tolist() == [0.0, 0.0]


def test_normalize_log_transform_preserves_dynamic_range_order() -> None:
    src = np.array([1.0, 10.0, 100.0])
    out = normalize_log_transform(src)
    assert out[0] < out[1] < out[2]
    assert out[0] == pytest.approx(0.0)
    assert out[-1] == pytest.approx(1.0)


def test_normalize_log_transform_does_not_mutate_input() -> None:
    src = np.array([1.0, 2.0, 3.0])
    snapshot = src.copy()
    _ = normalize_log_transform(src)
    assert np.array_equal(src, snapshot)


def test_normalize_log_transform_is_deterministic() -> None:
    src = np.array([0.0, 1.0, np.nan, 5.0, 50.0])
    a = normalize_log_transform(src)
    b = normalize_log_transform(src)
    assert np.array_equal(a, b, equal_nan=True)


def test_missing_float_constant_is_nan() -> None:
    assert math.isnan(MISSING_FLOAT)
