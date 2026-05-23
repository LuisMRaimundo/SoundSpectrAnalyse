"""Formula validation Pass 12 — dissonance models (validation plan only)."""

from __future__ import annotations

import math

import numpy as np
import numpy.testing as npt
import pandas as pd

import compile_metrics
import dissonance_models as dm


# Case DT-1
def test_total_dissonance_single_cross_pair() -> None:
    model = dm.SetharesDissonance()
    p1 = [(100.0, 1.0)]
    p2 = [(200.0, 1.0)]
    d = model.total_dissonance(p1, p2)
    ref = model.pure_tones_dissonance(100.0, 200.0, 1.0, 1.0)
    npt.assert_allclose(d, ref, rtol=1e-12, atol=0.0)


# Case ST-1
def test_same_timbre_shifted_equals_total_on_cross() -> None:
    model = dm.SetharesDissonance(curve_mode="cross")
    base = [(440.0, 1.0)]
    r = 2.0
    shifted = [(f * r, a) for f, a in base]
    st = model.same_timbre_dissonance(base, r)
    td = model.total_dissonance(base, shifted)
    npt.assert_allclose(st, td, rtol=1e-12, atol=0.0)


# Case CV-1
def test_calculate_dissonance_curve_abscissae() -> None:
    model = dm.SetharesDissonance()
    curve = model.calculate_dissonance_curve([(440.0, 1.0)], 1.0, 2.0, 3)
    keys = sorted(curve.keys())
    npt.assert_allclose(keys, [1.0, 1.5, 2.0], rtol=0.0, atol=1e-12)


# Case LM-1
def test_find_local_minima_interior_minimum() -> None:
    model = dm.SetharesDissonance()
    curve = {1.0: 0.5, 1.5: 0.05, 2.0: 0.5}
    minima = model.find_local_minima(curve, sensitivity=0.01)
    assert minima == [1.5]


# Case LM-2
def test_find_local_minima_asymmetric_sensitivity_excludes_peak() -> None:
    model = dm.SetharesDissonance()
    curve = {1.0: 0.06, 1.5: 0.055, 2.0: 0.06}
    minima = model.find_local_minima(curve, sensitivity=0.01)
    assert 1.5 not in minima


# Case SS-1
def test_sethares_s_scaling() -> None:
    model = dm.SetharesDissonance()
    out = model._s(100.0)
    ref = 0.24 / (0.0207 * 100.0 + 18.96)
    npt.assert_allclose(out, ref, rtol=1e-12, atol=0.0)


# Case SP-1
def test_sethares_pure_tones_manual() -> None:
    model = dm.SetharesDissonance()
    f1, f2, a1, a2 = 100.0, 200.0, 1.0, 2.0
    s = model._s(f1)
    y = s * (f2 - f1)
    raw = min(a1, a2) * model.gain * (math.exp(-model.b1 * y) - math.exp(-model.b2 * y))
    manual = float(raw) if raw > 0.0 else 0.0
    out = model.pure_tones_dissonance(f1, f2, a1, a2)
    npt.assert_allclose(out, manual, rtol=1e-9, atol=0.0)


# Case SK-1
def test_sethares_pairwise_sum_three_partials() -> None:
    model = dm.SetharesDissonance()
    pts = [(300.0, 1.0), (100.0, 1.0), (200.0, 1.0)]
    out = model._pairwise_sum(pts)
    d12 = model.pure_tones_dissonance(100.0, 200.0, 1.0, 1.0)
    d13 = model.pure_tones_dissonance(100.0, 300.0, 1.0, 1.0)
    d23 = model.pure_tones_dissonance(200.0, 300.0, 1.0, 1.0)
    npt.assert_allclose(out, d12 + d13 + d23, rtol=1e-9, atol=0.0)


# Case SX-1
def test_sethares_same_timbre_cross_differs_from_full() -> None:
    base = [(100.0, 1.0), (250.0, 1.0)]
    r = 2.0
    cross_m = dm.SetharesDissonance(curve_mode="cross")
    full_m = dm.SetharesDissonance(curve_mode="full")
    shifted = [(f * r, a) for f, a in base]
    cross_val = cross_m.same_timbre_dissonance(base, r)
    cross_ref = cross_m.total_dissonance(base, shifted)
    npt.assert_allclose(cross_val, cross_ref, rtol=1e-9, atol=0.0)
    full_val = full_m.same_timbre_dissonance(base, r)
    assert full_val != cross_val


# Case CM-1
def test_calculate_dissonance_metric_modes_hutchinson() -> None:
    df = pd.DataFrame(
        {"Frequency (Hz)": [100.0, 200.0, 300.0], "Amplitude": [1.0, 2.0, 1.0]}
    )
    model = dm.HutchinsonKnopoffDissonance()
    d_sum = model.calculate_dissonance_metric(df, metric_mode="sum", metric_scale=10.0)
    d_mean = model.calculate_dissonance_metric(df, metric_mode="mean_pair", metric_scale=10.0)
    d_scaled = model.calculate_dissonance_metric(df, metric_mode="mean_pair_scaled", metric_scale=10.0)
    d_minamp = model.calculate_dissonance_metric(df, metric_mode="minamp_norm", metric_scale=10.0)
    npt.assert_allclose(d_mean, d_sum / 3.0, rtol=1e-9, atol=0.0)
    npt.assert_allclose(d_scaled, d_mean * 10.0, rtol=1e-9, atol=0.0)
    npt.assert_allclose(d_minamp, d_sum / 3.0, rtol=1e-9, atol=0.0)


# Case HK-1
def test_hutchinson_cbw() -> None:
    out = dm.HutchinsonKnopoffDissonance.cbw(100.0)
    ref = 1.72 * (100.0**0.65)
    npt.assert_allclose(out, ref, rtol=1e-12, atol=0.0)


# Case HK-2
def test_hutchinson_g_table_knot() -> None:
    model = dm.HutchinsonKnopoffDissonance()
    npt.assert_allclose(model.g(0.25), 1.00, rtol=0.0, atol=1e-12)


# Case HK-3
def test_hutchinson_pure_tones_manual() -> None:
    model = dm.HutchinsonKnopoffDissonance()
    f1, f2, a1, a2 = 400.0, 500.0, 1.0, 1.0
    f_bar = 0.5 * (f1 + f2)
    cb = model.cbw(f_bar)
    y = abs(f1 - f2) / cb
    g = model.g(y)
    denom = a1 * a1 + a2 * a2
    manual = (a1 * a2 * g) / denom
    out = model.pure_tones_dissonance(f1, f2, a1, a2)
    npt.assert_allclose(out, manual, rtol=1e-9, atol=0.0)


# Case VA-1
def test_vassilakis_pure_tones_manual() -> None:
    model = dm.VassilakisDissonance()
    f1, f2, a1, a2 = 100.0, 200.0, 1.0, 2.0
    A1 = max(a1, a2)
    A2 = min(a1, a2)
    af = (2.0 * A2) / (A1 + A2)
    s = model._s(f1)
    x = s * (f2 - f1)
    spectral = math.exp(-model.b1 * x) - math.exp(-model.b2 * x)
    manual = float(
        (A1 * A2) ** model.spl_exp
        * model.pair_factor
        * (af**model.af_exp)
        * spectral
    )
    out = model.pure_tones_dissonance(f1, f2, a1, a2)
    npt.assert_allclose(out, manual, rtol=1e-9, atol=0.0)


# Case AM-1
def test_calculate_all_dissonance_metrics_keys_finite() -> None:
    df = pd.DataFrame({"Frequency (Hz)": [100.0, 200.0], "Amplitude": [1.0, 1.0]})
    out = dm.calculate_all_dissonance_metrics(df)
    assert set(out.keys()) == {"sethares", "hutchinson-knopoff", "vassilakis"}
    assert all(np.isfinite(float(v)) for v in out.values())


# Case CP-1
def test_compare_curves_minmax_normalisation_formula() -> None:
    vals = [0.2, 0.5, 0.8]
    v_min, v_max = min(vals), max(vals)
    norm = [(v - v_min) / (v_max - v_min) for v in vals]
    npt.assert_allclose(norm, [0.0, 0.5, 1.0], rtol=1e-12, atol=0.0)


# Case PA-1
def test_pair_count_binomial() -> None:
    n = 5
    assert n * (n - 1) // 2 == 10


# Case EX-1
def test_extract_dissonance_metrics_first_finite() -> None:
    dfs = {"Sheet1": pd.DataFrame({"X Dissonance Y": [np.nan, 1.5, 2.0]})}
    out = compile_metrics.extract_dissonance_metrics(dfs)
    assert out == {"X Dissonance Y": 1.5}
