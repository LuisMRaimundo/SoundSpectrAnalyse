"""Tests for discrete spectral metrics D3, D10, D17, D24 in density.apply_density_metric."""
from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from density import (
    DISCRETE_SPECTRAL_METRIC_KEYS,
    _apply_discrete_spectral_metrics,
    _spectral_neff_from_filtered_linear_amplitudes,
    apply_density_metric,
    band_partial_metric_sum,
    compute_discrete_spectral_metrics_bundle,
    get_weight_function,
    partial_metric_sums_h_i_s_total,
)


class DiscreteSpectralMetricsTests(unittest.TestCase):
    def test_get_weight_function_accepts_discrete_keys(self) -> None:
        for k in ("d3", "d10", "d17", "d24"):
            fn = get_weight_function(k)
            self.assertTrue(callable(fn))

    def test_legacy_d2_maps_to_linear_weight(self) -> None:
        self.assertIs(get_weight_function("d2"), get_weight_function("linear"))

    def test_legacy_d8_maps_to_d17_weight(self) -> None:
        self.assertIs(get_weight_function("d8"), get_weight_function("d17"))

    def test_legacy_sum_weight_alias_is_linear(self) -> None:
        f_sum = get_weight_function("sum")
        f_lin = get_weight_function("linear")
        x = np.array([0.25, 0.5, 1.0])
        np.testing.assert_array_equal(f_sum(x), f_lin(x))

    def test_neff_helper_uniform_amplitudes(self) -> None:
        a = np.array([1.0, 1.0, 1.0])
        self.assertAlmostEqual(_spectral_neff_from_filtered_linear_amplitudes(a), 3.0)

    def test_d3_log_sum_natural_log1p(self) -> None:
        a = np.array([0.0, 1.0])
        expected = math.log(2.0)
        self.assertAlmostEqual(_apply_discrete_spectral_metrics("d3", a, None), expected)

    def test_d10_uses_neff_over_n(self) -> None:
        a = np.array([1.0, 1.0, 1.0])
        want = float(3.0 * math.log(2.0))  # N_eff=N=3, sum log1p(1)=3*ln2
        self.assertAlmostEqual(_apply_discrete_spectral_metrics("d10", a, None), want)

    def test_d17_log_energy_times_log_neff(self) -> None:
        a = np.array([1.0, 1.0, 1.0])
        want = float(math.log1p(3.0) * math.log1p(3.0))
        self.assertAlmostEqual(_apply_discrete_spectral_metrics("d17", a, None), want)

    def test_d24_frequency_and_amplitude_gate(self) -> None:
        a = np.array([1.0, 0.005, 0.5, 1.0])
        f = np.array([100.0, 200.0, 5000.0, 15000.0])
        v = a[[0, 2]]
        want = float(np.sum(np.log1p(v)))
        got = _apply_discrete_spectral_metrics("d24", a, f)
        self.assertAlmostEqual(got, want)

    def test_discrete_bypasses_rolloff_path(self) -> None:
        a = np.array([10.0, 1.0])
        f = np.array([100.0, 200.0])
        want = float(math.log1p(10.0) + math.log1p(1.0))
        d3 = apply_density_metric(
            a,
            "d3",
            frequencies=f,
            fundamental_freq=100.0,
            account_for_spectral_rolloff=True,
            prevent_domination=True,
        )
        self.assertAlmostEqual(d3, want)

    def test_keys_frozen_set(self) -> None:
        self.assertEqual(
            DISCRETE_SPECTRAL_METRIC_KEYS,
            frozenset({"d3", "d10", "d17", "d24"}),
        )

    def test_compute_bundle_matches_individual(self) -> None:
        a = np.array([1.0, 2.0, 3.0])
        f = np.array([100.0, 200.0, 8000.0])
        b = compute_discrete_spectral_metrics_bundle(a, f)
        self.assertAlmostEqual(b["discrete_metric_d3"], _apply_discrete_spectral_metrics("d3", a, None))
        self.assertAlmostEqual(b["discrete_metric_d10"], _apply_discrete_spectral_metrics("d10", a, None))
        self.assertAlmostEqual(b["discrete_metric_d17"], _apply_discrete_spectral_metrics("d17", a, None))
        self.assertAlmostEqual(b["discrete_metric_d24"], _apply_discrete_spectral_metrics("d24", a, f))

    def test_weight_ui_display_labels_resolve_to_internal_keys(self) -> None:
        from weight_function_ui_labels import resolve_weight_key_from_user_label

        self.assertEqual(resolve_weight_key_from_user_label("D3 (Σlog1p A)"), "d3")
        self.assertEqual(resolve_weight_key_from_user_label("D10 (Σlog1p·N_eff/N)"), "d10")
        self.assertEqual(resolve_weight_key_from_user_label("D17 (log1p E · log1p N_eff)"), "d17")
        self.assertEqual(resolve_weight_key_from_user_label("Logarithmic"), "log")
        self.assertEqual(resolve_weight_key_from_user_label("log"), "log")
        # Legacy removed UI labels
        self.assertEqual(resolve_weight_key_from_user_label("d2 (σa²)"), "linear")
        self.assertEqual(resolve_weight_key_from_user_label("d8 (n_eff)"), "d17")

    def test_density_metrics_sheet_is_minimal_partial_sums(self) -> None:
        from compile_metrics import DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS, _build_density_metrics_main_sheet

        row = {
            "Note": "A4",
            "weight_function": "log",
            "Harmonic Partials sum": 1.0,
            "Inharmonic Partials sum": 0.2,
            "Sub-bass sum": 0.05,
            "Total sum": 1.25,
        }
        out = _build_density_metrics_main_sheet(pd.DataFrame([row]), weight_function="log")
        self.assertEqual(list(out.columns), DENSITY_METRICS_MINIMAL_DISPLAY_COLUMNS)

    def test_density_metrics_sheet_legacy_linear_fallback(self) -> None:
        from compile_metrics import _build_density_metrics_main_sheet

        row = {
            "Note": "A4",
            "linear_sum_amplitude_harmonic": 1.0,
            "linear_sum_amplitude_inharmonic_partial": 0.2,
            "linear_sum_amplitude_subbass_band": 0.05,
        }
        out = _build_density_metrics_main_sheet(pd.DataFrame([row]), weight_function="linear")
        self.assertEqual(float(out["Harmonic Partials sum"].iloc[0]), 1.0)
        self.assertAlmostEqual(float(out["Total sum"].iloc[0]), 1.25)

    def test_partial_metric_sums_h_i_s_total_matches_band_helper(self) -> None:
        ah = np.array([1.0, 1.0])
        ai = np.array([0.5])
        asb = np.array([0.25])
        fh = np.array([100.0, 200.0])
        fi = np.array([150.0])
        fsb = np.array([30.0])
        for wf in ("linear", "sqrt", "log"):
            h, i, s, t = partial_metric_sums_h_i_s_total(
                ah, ai, asb, wf, harmonic_frequencies_hz=fh, inharmonic_frequencies_hz=fi, subbass_frequencies_hz=fsb
            )
            sh = float(np.sum(ah))
            si = float(np.sum(ai))
            ss = float(np.sum(asb))
            eh = band_partial_metric_sum(np.array([sh], dtype=float), wf, frequencies_hz=None)
            ei = band_partial_metric_sum(np.array([si], dtype=float), wf, frequencies_hz=None)
            es = band_partial_metric_sum(np.array([ss], dtype=float), wf, frequencies_hz=None)
            self.assertAlmostEqual(h, eh)
            self.assertAlmostEqual(i, ei)
            self.assertAlmostEqual(s, es)
            self.assertAlmostEqual(t, eh + ei + es)
        ah8 = np.array([1.0, 1.0])
        ai8 = np.array([1.0])
        asb8 = np.array([1.0])
        for wf8 in ("d10", "d17"):
            h8, i8, s8, t8 = partial_metric_sums_h_i_s_total(ah8, ai8, asb8, wf8)
            concat = np.concatenate([ah8, ai8, asb8])
            want_t = band_partial_metric_sum(concat, wf8)
            self.assertAlmostEqual(t8, want_t)
            self.assertAlmostEqual(h8, band_partial_metric_sum(ah8, wf8))
            self.assertAlmostEqual(i8, band_partial_metric_sum(ai8, wf8))
            self.assertAlmostEqual(s8, band_partial_metric_sum(asb8, wf8))


if __name__ == "__main__":
    unittest.main()
