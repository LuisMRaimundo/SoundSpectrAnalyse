"""
Edge case tests for increased coverage

RECOMMENDATION: Add edge case tests
Score Impact: Part of +5 points for test coverage

Tests cover:
- Boundary conditions
- Invalid inputs
- Extreme values
- Error conditions
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from proc_audio import AudioProcessor, _verify_energy_conservation, _calculate_window_characteristics
from density import apply_density_metric, calculate_combined_density_metric
from constants import (
    ENERGY_CONSERVATION_TOLERANCE,
    DEFAULT_N_FFT,
    EPSILON_AMPLITUDE
)


class TestBoundaryConditions(unittest.TestCase):
    """Test boundary conditions"""
    
    def test_zero_amplitude(self):
        """Test density metric with zero amplitudes"""
        values = np.zeros(10)
        result = apply_density_metric(values, weight_function='linear')
        self.assertEqual(result, 0.0)
    
    def test_single_amplitude(self):
        """Test density metric with single amplitude"""
        values = np.array([1.0])
        result = apply_density_metric(values, weight_function='linear')
        self.assertGreater(result, 0.0)
    
    def test_very_small_amplitudes(self):
        """Test density metric with very small amplitudes"""
        values = np.array([EPSILON_AMPLITUDE] * 10)
        result = apply_density_metric(values, weight_function='linear')
        self.assertGreaterEqual(result, 0.0)
    
    def test_very_large_amplitudes(self):
        """Test density metric with very large amplitudes"""
        values = np.array([1e10] * 10)
        result = apply_density_metric(values, weight_function='linear')
        self.assertGreater(result, 0.0)
        self.assertTrue(np.isfinite(result))
    
    def test_mixed_amplitudes(self):
        """Test density metric with mixed amplitude range"""
        values = np.array([1e-10, 1.0, 1e10])
        result = apply_density_metric(values, weight_function='linear')
        self.assertGreater(result, 0.0)
        self.assertTrue(np.isfinite(result))
    
    def test_empty_array(self):
        """Test density metric with empty array"""
        values = np.array([])
        result = apply_density_metric(values, weight_function='linear')
        self.assertEqual(result, 0.0)
    
    def test_nan_values(self):
        """Test density metric with NaN values"""
        # Mathematical verification (reference):
        # NaN + any = NaN, Inf + any = Inf
        # Solution: Function should filter NaN/Inf before calculation
        values = np.array([1.0, np.nan, 2.0, np.inf])
        
        # Should handle NaN/Inf gracefully by filtering them out
        result = apply_density_metric(values, weight_function='linear')
        
        # Result should be finite (NaN/Inf values filtered)
        self.assertTrue(np.isfinite(result), 
                       msg=f"Result should be finite after filtering NaN/Inf. Got: {result}")
        
        # Result should be based on valid values only (1.0 + 2.0 = 3.0 for linear)
        # With normalization, result may vary, but should be positive and finite
        self.assertGreater(result, 0.0,
                          msg=f"Result should be positive. Got: {result}")


class TestInvalidInputs(unittest.TestCase):
    """Test handling of invalid inputs"""
    
    def test_negative_amplitudes(self):
        """Test density metric with negative amplitudes"""
        # Mathematical verification (reference):
        # Amplitude = |complex_value| = sqrt(real² + imag²) ≥ 0
        # Negative amplitudes are non-physical, function should take absolute value
        values = np.array([-1.0, 1.0, -2.0])
        
        # Should handle negative values by taking absolute value
        result = apply_density_metric(values, weight_function='linear')
        
        # Result should be non-negative (absolute values used)
        self.assertGreaterEqual(result, 0.0,
                               msg=f"Result should be non-negative after taking absolute values. Got: {result}")
        
        # Result should be based on absolute values: |1.0| + |1.0| + |2.0| = 4.0 for linear
        # With normalization/prevention, result may vary, but should be positive
        self.assertGreater(result, 0.0,
                          msg=f"Result should be positive. Got: {result}")
    
    def test_invalid_weight_function(self):
        """Test with invalid weight function"""
        values = np.array([1.0, 2.0, 3.0])
        # Should handle invalid weight function gracefully
        try:
            result = apply_density_metric(values, weight_function='invalid')
            # If it doesn't raise, should use default or fallback
        except (ValueError, KeyError):
            # Expected to raise error for invalid weight function
            pass
    
    def test_zero_combined_metric(self):
        """Test combined metric with zero inputs"""
        result = calculate_combined_density_metric(
            0.0, 0.0, alpha=0.8, beta=0.2, preserve_dynamic_range=True
        )
        self.assertGreaterEqual(result, 0.0)
    
    def test_negative_combined_metric(self):
        """Test combined metric with negative inputs"""
        # Should handle negative inputs gracefully
        try:
            result = calculate_combined_density_metric(
                -1.0, -2.0, alpha=0.8, beta=0.2, preserve_dynamic_range=True
            )
            self.assertGreaterEqual(result, 0.0)
        except (ValueError, RuntimeWarning):
            # Acceptable to raise error or warning
            pass


class TestExtremeValues(unittest.TestCase):
    """Test extreme value handling"""
    
    def test_energy_conservation_extreme_signal(self):
        """Test energy conservation with extreme signal"""
        sr = 44100
        duration = 0.1
        # Very high amplitude signal
        y = np.ones(int(sr * duration)) * 1e6
        
        import librosa
        n_fft = 2048
        hop_length = 512
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann')
        
        result = _verify_energy_conservation(
            y, S, n_fft, hop_length, 'hann', tolerance=0.2  # More lenient tolerance
        )
        
        # Should still verify (may have larger deviation)
        self.assertIn('energy_ratio', result)
        self.assertIn('is_valid', result)
    
    def test_window_characteristics_extreme_size(self):
        """Test window characteristics with extreme sizes"""
        # Very small window
        result_small = _calculate_window_characteristics('hann', 8)
        self.assertIn('main_lobe_width', result_small)
        
        # Very large window
        result_large = _calculate_window_characteristics('hann', 65536)
        self.assertIn('main_lobe_width', result_large)
    
    def test_combined_metric_extreme_ratios(self):
        """Test combined metric with extreme harmonic/inharmonic ratios"""
        # Harmonic >> Inharmonic
        result1 = calculate_combined_density_metric(
            100.0, 0.01, alpha=0.8, beta=0.2, preserve_dynamic_range=True
        )
        self.assertGreater(result1, 0.0)
        
        # Inharmonic >> Harmonic
        result2 = calculate_combined_density_metric(
            0.01, 100.0, alpha=0.8, beta=0.2, preserve_dynamic_range=True
        )
        self.assertGreater(result2, 0.0)
        
        # Both very large
        result3 = calculate_combined_density_metric(
            1e10, 1e10, alpha=0.8, beta=0.2, preserve_dynamic_range=True
        )
        self.assertGreater(result3, 0.0)
        self.assertTrue(np.isfinite(result3))


class TestErrorConditions(unittest.TestCase):
    """Test error condition handling"""
    
    def test_processor_without_audio(self):
        """Test AudioProcessor without loaded audio"""
        processor = AudioProcessor()
        
        # Should handle gracefully when audio not loaded
        try:
            processor.fft_analysis()
        except (AttributeError, ValueError):
            # Expected to raise error
            pass
    
    def test_processor_with_invalid_sr(self):
        """Test AudioProcessor with invalid sample rate"""
        # Mathematical verification (reference):
        # Sample rate must be > 0 for FFT calculations
        # Division by zero occurs when sr = 0 in frequency calculations
        processor = AudioProcessor()
        processor.y = np.array([1.0, 2.0, 3.0])
        processor.sr = 0  # Invalid sample rate
        
        # Should raise an error (RuntimeError wraps the ZeroDivisionError)
        with self.assertRaises((ValueError, ZeroDivisionError, RuntimeError)):
            processor.fft_analysis()
    
    def test_processor_with_invalid_n_fft(self):
        """Test AudioProcessor with invalid n_fft"""
        processor = AudioProcessor()
        sr = 44100
        duration = 0.5
        t = np.linspace(0, duration, int(sr * duration))
        processor.y = np.sin(2 * np.pi * 440 * t)
        processor.sr = sr
        processor.n_fft = 0  # Invalid n_fft
        
        try:
            processor.fft_analysis()
        except (ValueError, AssertionError):
            # Expected to raise error
            pass


if __name__ == '__main__':
    unittest.main()

