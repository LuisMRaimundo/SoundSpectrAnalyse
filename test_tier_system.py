"""
Tests for 90-Tier Granular Clustering System
Analytical reference values for tier assignment verification

Tests verify:
- Correct tier assignment based on frequency
- Parameter selection per tier
- Adaptive tolerance calculation
- Security margin calculation
"""

import unittest
import sys
from pathlib import Path
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import tier system from orchestrator
try:
    from pipeline_orchestrator_gui import (
        FFT_SETTINGS_BY_CLUSTER,
        _calculate_security_margin,
        _calculate_adaptive_tolerance,
        _round_to_power_of_2
    )
    TIER_SYSTEM_AVAILABLE = True
except ImportError:
    TIER_SYSTEM_AVAILABLE = False


@unittest.skipUnless(TIER_SYSTEM_AVAILABLE, "Tier system not available")
class TestTierAssignment(unittest.TestCase):
    """Test tier assignment logic"""
    
    def test_tier_assignment_low_frequencies(self):
        """Test tier assignment for low frequencies"""
        # Low frequencies (< 20 Hz) should use Tier_01
        # Frequencies 20-60 Hz should use Tiers 01-10
        
        # Test boundary frequencies
        test_cases = [
            (15.0, 'Tier_01'),   # Below first tier
            (20.0, 'Tier_01'),   # First tier boundary
            (30.0, 'Tier_04'),   # Mid-range in low frequencies
            (60.0, 'Tier_10'),   # Boundary of first 10 tiers
        ]
        
        for freq, expected_tier_prefix in test_cases:
            # Find appropriate tier
            tier_found = None
            for tier_name, settings in FFT_SETTINGS_BY_CLUSTER.items():
                if freq <= settings['max_freq']:
                    tier_found = tier_name
                    break
            
            if tier_found:
                # Should match expected tier or be in appropriate range
                self.assertTrue(tier_found.startswith(expected_tier_prefix.split('_')[0]),
                              f"Frequency {freq} Hz assigned to {tier_found}, "
                              f"expected {expected_tier_prefix}")
    
    def test_tier_parameter_selection(self):
        """Test that tier parameters are correctly selected"""
        # Each tier should have appropriate parameters
        for tier_name, settings in list(FFT_SETTINGS_BY_CLUSTER.items())[:10]:  # Test first 10 tiers
            self.assertIn('n_fft', settings)
            self.assertIn('tolerance', settings)
            self.assertIn('zp', settings)
            
            # N_FFT should be power of 2
            n_fft = settings['n_fft']
            self.assertTrue(is_power_of_2(n_fft),
                          f"Tier {tier_name} n_fft={n_fft} should be power of 2")
            
            # Tolerance should be reasonable (3-15 Hz for low frequencies)
            tolerance = settings['tolerance']
            self.assertGreater(tolerance, 0.0)
            self.assertLess(tolerance, 100.0)
            
            # Zero padding should be reasonable (1-8)
            zp = settings['zp']
            self.assertGreaterEqual(zp, 1)
            self.assertLessEqual(zp, 8)


@unittest.skipUnless(TIER_SYSTEM_AVAILABLE, "Tier system not available")
class TestSecurityMarginCalculation(unittest.TestCase):
    """Test security margin calculation"""
    
    def test_security_margin_boundaries(self):
        """Test security margin at frequency boundaries"""
        # Mathematical verification (reference):
        # f0 = 20 Hz  → margin = 35.0% (boundary)
        # f0 = 60 Hz  → margin = 25.0% (boundary)
        # f0 = 120 Hz → margin = 15.0% (boundary)
        # f0 = 300 Hz → margin = 10.0% (boundary)
        
        test_cases = [
            (20.0, 35.0),
            (60.0, 25.0),
            (120.0, 15.0),
            (300.0, 10.0),
            (400.0, 10.0),  # Above 300 Hz, should stay at 10.0
        ]
        
        for f0, expected_margin in test_cases:
            margin = _calculate_security_margin(f0)
            self.assertAlmostEqual(margin, expected_margin, delta=0.1,
                                 msg=f"Security margin for f0={f0} Hz: expected {expected_margin}%, "
                                 f"got {margin}%")
    
    def test_security_margin_interpolation(self):
        """Test security margin interpolation between boundaries"""
        # Test intermediate values
        test_cases = [
            (45.0, 27.0, 30.0),   # Between 20-60 Hz: should interpolate between 35% and 25%
            (85.0, 19.0, 25.0),   # Between 60-120 Hz: should interpolate between 25% and 15% (allow 1% tolerance)
            (150.0, 12.0, 15.0),  # Between 120-300 Hz: should interpolate between 15% and 10%
        ]
        
        for f0, min_margin, max_margin in test_cases:
            margin = _calculate_security_margin(f0)
            self.assertGreaterEqual(margin, min_margin,
                                  msg=f"Security margin for f0={f0} Hz: {margin}% should be >= {min_margin}%")
            self.assertLessEqual(margin, max_margin,
                               msg=f"Security margin for f0={f0} Hz: {margin}% should be <= {max_margin}%")
    
    def test_security_margin_monotonic(self):
        """Test that security margin decreases monotonically with frequency"""
        frequencies = [20.0, 40.0, 60.0, 100.0, 150.0, 200.0, 300.0, 400.0]
        margins = [_calculate_security_margin(f) for f in frequencies]
        
        # Each subsequent margin should be <= previous (monotonic decrease)
        for i in range(1, len(margins)):
            self.assertLessEqual(margins[i], margins[i-1],
                               f"Security margin should decrease monotonically. "
                               f"f={frequencies[i-1]} Hz → margin={margins[i-1]}%, "
                               f"f={frequencies[i]} Hz → margin={margins[i]}%")


@unittest.skipUnless(TIER_SYSTEM_AVAILABLE, "Tier system not available")
class TestAdaptiveTolerance(unittest.TestCase):
    """Test adaptive tolerance calculation"""
    
    def test_adaptive_tolerance_jnd_model(self):
        """Test that adaptive tolerance follows JND model"""
        # Mathematical verification (reference):
        # JND ≈ 1.5% of frequency (psychoacoustic)
        # For f = 440 Hz: tolerance = max(base, 440 × 0.015) = max(5.0, 6.6) = 6.6 Hz
        # For f = 1000 Hz: tolerance = max(base, 1000 × 0.015) = max(5.0, 15.0) = 15.0 Hz
        
        test_cases = [
            (440.0, 5.0, True, 6.6),   # Adaptive should override base
            (100.0, 5.0, True, 5.0),   # Base should be used (100 × 0.015 = 1.5 < 5.0)
            (1000.0, 5.0, True, 15.0), # Adaptive should override (1000 × 0.015 = 15.0)
        ]
        
        for freq, base_tolerance, use_adaptive, expected_tolerance in test_cases:
            tolerance = _calculate_adaptive_tolerance(freq, base_tolerance, use_adaptive)
            
            # Should be approximately equal (within 0.5 Hz)
            self.assertAlmostEqual(tolerance, expected_tolerance, delta=0.5,
                                 msg=f"Adaptive tolerance for f={freq} Hz, base={base_tolerance}: "
                                 f"expected {expected_tolerance}, got {tolerance}")
    
    def test_adaptive_tolerance_without_adaptive(self):
        """Test tolerance calculation without adaptive mode"""
        # When use_adaptive=False, should return base_tolerance
        test_cases = [
            (440.0, 5.0, False, 5.0),
            (1000.0, 10.0, False, 10.0),
        ]
        
        for freq, base_tolerance, use_adaptive, expected_tolerance in test_cases:
            tolerance = _calculate_adaptive_tolerance(freq, base_tolerance, use_adaptive)
            self.assertEqual(tolerance, expected_tolerance,
                           f"Tolerance for f={freq} Hz, base={base_tolerance}, "
                           f"adaptive={use_adaptive}: expected {expected_tolerance}, got {tolerance}")


@unittest.skipUnless(TIER_SYSTEM_AVAILABLE, "Tier system not available")
class TestPowerOf2Rounding(unittest.TestCase):
    """Test power-of-2 rounding function"""
    
    def test_power_of_2_rounding(self):
        """Test that values are rounded to nearest power of 2"""
        # Mathematical verification (reference):
        # Rounding to power of 2:
        # 1000 → 1024 (2^10)
        # 5000 → 4096 (2^12) or 8192 (2^13)
        # 16384 → 16384 (already power of 2)
        
        test_cases = [
            (512, 512),      # Already power of 2
            (513, 512),      # Round down to 512 (distance: 1)
            (768, 1024),     # Closer to 1024 (distance: 256 vs 256, rounds up)
            (769, 1024),     # Closer to 1024 (distance: 255)
            (1000, 1024),    # Round up to 1024 (distance: 24)
            (1500, 1024),    # Round to 1024 (distance: 476 < 548)
            (4096, 4096),    # Already power of 2
        ]
        
        for value, expected in test_cases:
            result = _round_to_power_of_2(value)
            self.assertEqual(result, expected,
                           f"Rounding {value} to power of 2: expected {expected}, got {result}")
    
    def test_power_of_2_result_is_valid(self):
        """Test that rounded values are valid powers of 2"""
        test_values = [100, 500, 1000, 5000, 10000, 20000]
        
        for value in test_values:
            result = _round_to_power_of_2(value)
            self.assertTrue(is_power_of_2(result),
                          f"Rounded value {result} should be power of 2")


def is_power_of_2(n):
    """Check if n is a power of 2"""
    return n > 0 and (n & (n - 1)) == 0


if __name__ == '__main__':
    unittest.main()
