"""Unit tests for effective partial density (participation ratio on powers)."""

import math
import unittest

import numpy as np

from density import (
    effective_partial_density_from_powers,
    partial_density_effective_components_bundle,
)


class TestEffectivePartialDensityFromPowers(unittest.TestCase):
    def test_single_component(self):
        self.assertAlmostEqual(effective_partial_density_from_powers(np.array([1.0])), 1.0, places=9)

    def test_two_equal(self):
        self.assertAlmostEqual(effective_partial_density_from_powers(np.array([1.0, 1.0])), 2.0, places=9)

    def test_four_equal(self):
        self.assertAlmostEqual(
            effective_partial_density_from_powers(np.array([1.0, 1.0, 1.0, 1.0])),
            4.0,
            places=9,
        )

    def test_one_dominant_many_tiny(self):
        d = effective_partial_density_from_powers(np.array([1.0, 0.01, 0.01, 0.01]))
        self.assertLess(abs(d - 1.0), 0.2, msg=f"expected near 1, got {d}")

    def test_scale_invariant(self):
        p = np.array([0.3, 0.7, 0.05])
        d0 = effective_partial_density_from_powers(p)
        d1 = effective_partial_density_from_powers(p * 1e6)
        self.assertAlmostEqual(d0, d1, places=9)

    def test_nan_zero_negative_ignored(self):
        p = np.array([1.0, np.nan, 0.0, -1.0, 1.0])
        self.assertAlmostEqual(effective_partial_density_from_powers(p), 2.0, places=9)

    def test_empty_returns_zero(self):
        self.assertEqual(effective_partial_density_from_powers(np.array([])), 0.0)


class TestPartialDensityBundle(unittest.TestCase):
    def test_bundle_returns_float_and_diag(self):
        d, diag = partial_density_effective_components_bundle(
            harmonic_amplitudes=np.array([1.0, 0.5]),
            inharmonic_amplitudes=np.array([0.2]),
            ground_noise_power=0.01,
        )
        self.assertTrue(math.isfinite(d))
        self.assertIn("partial_density_harmonic_power_total", diag)


if __name__ == "__main__":
    unittest.main()
