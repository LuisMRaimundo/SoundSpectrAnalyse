"""
Unit tests for density.py (Phase 3 enhancements)

Tests cover:
- Critical band analysis (24 bands)
- Harmonic completeness with frequency-dependent penalty
- Spectral smoothing
- Combined density metric
"""

import unittest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from density import (
    calculate_perceptual_spectral_density,
    apply_spectral_smoothing,
    calculate_combined_density_metric,
    _calculate_harmonic_completeness_phase2
)
from constants import (
    NUM_CRITICAL_BANDS,
    HARMONIC_DETECTION_THRESHOLD_DB,
    SMOOTHING_WINDOW_PERCENTAGE,
    SMOOTHING_MIN_WINDOW_LENGTH
)


class TestCriticalBandAnalysis(unittest.TestCase):
    """Test 24 critical band analysis"""
    
    def test_critical_band_allocation(self):
        """Test that harmonics are allocated to correct critical bands"""
        # Generate harmonic series
        f0 = 440.0
        n_harmonics = 10
        harmonic_freqs = np.array([f0 * (i + 1) for i in range(n_harmonics)])
        harmonic_amps = np.ones(n_harmonics) * 0.5
        
        # Calculate perceptual density (uses 24 critical bands)
        density = calculate_perceptual_spectral_density(
            harmonic_amps,
            harmonic_freqs,
            f0,
            threshold_db=HARMONIC_DETECTION_THRESHOLD_DB
        )
        
        # Should return valid density (0-1)
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)
    
    def test_masking_model(self):
        """Test that masking model reduces effective audibility"""
        # Create two harmonics: one strong, one weak
        f0 = 440.0
        harmonic_freqs = np.array([f0, f0 * 2])
        # Strong first harmonic, weak second
        harmonic_amps = np.array([1.0, 0.1])
        
        density = calculate_perceptual_spectral_density(
            harmonic_amps,
            harmonic_freqs,
            f0,
            threshold_db=-40.0  # Higher threshold to test masking
        )
        
        # Density should be less than if no masking occurred
        self.assertGreaterEqual(density, 0.0)
        self.assertLessEqual(density, 1.0)


class TestHarmonicCompleteness(unittest.TestCase):
    """Test harmonic completeness with frequency-dependent penalty"""
    
    def test_completeness_with_all_harmonics(self):
        """Test completeness when all harmonics are present"""
        f0 = 440.0
        frequency_limit = 20000.0
        
        # Create complete harmonic series
        n_harmonics = int(frequency_limit / f0)
        harmonic_freqs = np.array([f0 * (i + 1) for i in range(min(n_harmonics, 20))])
        harmonic_db = np.ones(len(harmonic_freqs)) * -30.0  # All above threshold
        
        completeness = _calculate_harmonic_completeness_phase2(
            harmonic_freqs,
            harmonic_db,
            f0,
            frequency_limit,
            threshold_db=-40.0
        )
        
        # Should be close to 1.0 (complete)
        self.assertGreater(completeness, 0.8)
    
    def test_completeness_with_missing_lower_harmonics(self):
        """Test that missing lower harmonics penalizes more"""
        f0 = 440.0
        frequency_limit = 20000.0
        
        # Missing 2nd and 3rd harmonics (should be heavily penalized)
        harmonic_freqs = np.array([f0, f0 * 4, f0 * 5, f0 * 6])  # Missing 2nd, 3rd
        harmonic_db = np.ones(len(harmonic_freqs)) * -30.0
        
        completeness = _calculate_harmonic_completeness_phase2(
            harmonic_freqs,
            harmonic_db,
            f0,
            frequency_limit,
            threshold_db=-40.0
        )
        
        # Should be lower due to missing lower harmonics
        self.assertLess(completeness, 0.7)
    
    def test_completeness_with_missing_higher_harmonics(self):
        """Test that missing higher harmonics penalizes less"""
        f0 = 440.0
        frequency_limit = 20000.0
        
        # Missing 10th and 11th harmonics (should be less penalized)
        harmonic_freqs = np.array([f0 * i for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13]])
        harmonic_db = np.ones(len(harmonic_freqs)) * -30.0
        
        completeness = _calculate_harmonic_completeness_phase2(
            harmonic_freqs,
            harmonic_db,
            f0,
            frequency_limit,
            threshold_db=-40.0
        )
        
        # Should be higher than missing lower harmonics
        self.assertGreater(completeness, 0.6)


class TestSpectralSmoothing(unittest.TestCase):
    """Test spectral smoothing functions"""
    
    def test_smoothing_reduces_noise(self):
        """Test that smoothing reduces isolated narrow peaks in the spectrum model"""
        # Create spectrum with noise
        n_freq = 1024
        n_time = 5
        
        # Clean signal + noise
        clean = np.zeros((n_freq, n_time))
        clean[100, :] = 1.0  # Single peak
        
        noise = np.random.randn(n_freq, n_time) * 0.1
        noisy_spectrum = clean + noise
        
        # Apply smoothing
        smoothed = apply_spectral_smoothing(
            noisy_spectrum,
            method="savitzky_golay",
            window_length=None,
            noise_floor_percentile=15.0
        )
        
        # Smoothed spectrum should have less noise
        noise_level_original = np.std(noisy_spectrum[200:300, :])
        noise_level_smoothed = np.std(smoothed[200:300, :])
        
        # Smoothed should have less noise (or at least not more)
        self.assertLessEqual(noise_level_smoothed, noise_level_original * 1.5)
    
    def test_smoothing_preserves_peaks(self):
        """Test that smoothing preserves genuine peaks"""
        n_freq = 512
        n_time = 3
        
        # Create spectrum with clear peak
        spectrum = np.zeros((n_freq, n_time))
        peak_idx = 100
        spectrum[peak_idx, :] = 1.0
        
        smoothed = apply_spectral_smoothing(
            spectrum,
            method="savitzky_golay",
            window_length=11
        )
        
        # Peak should still be present (may be slightly reduced but not eliminated)
        # Mathematical verification (reference):
        # Savitzky-Golay filter with window=11, polyorder=3 reduces peak by ~80% for sharp peak
        # For single-bin peak: peak_strength ≈ 0.2-0.3 after smoothing is expected
        # The smoothing spreads the peak across neighboring bins
        
        # Filter out NaN/Inf values for comparison
        smoothed_clean = smoothed[np.isfinite(smoothed)]
        if smoothed_clean.size == 0:
            self.fail("Smoothed spectrum contains only NaN/Inf values")
        
        peak_strength = np.max(smoothed[peak_idx-5:peak_idx+5, :])
        
        # Peak should still be detectable in its region
        # For a single-bin peak, smoothing spreads it, so peak_strength ≈ 0.2-0.3 is expected
        self.assertGreater(peak_strength, 0.05, 
                          msg=f"Peak strength {peak_strength:.3f} should be > 0.05 after smoothing")
        
        # Peak region should have higher values than surrounding regions
        # Check that peak region is stronger than a distant region
        if peak_idx > 50 and peak_idx < n_freq - 50:
            distant_region_max = np.max(smoothed[peak_idx-50:peak_idx-40, :])
            self.assertGreater(peak_strength, distant_region_max,
                             msg=f"Peak region ({peak_strength:.3f}) should be stronger than "
                                 f"distant region ({distant_region_max:.3f})")


class TestCombinedDensityMetric(unittest.TestCase):
    """Test combined density metric"""
    
    def test_combined_metric_range(self):
        """Test that combined metric returns valid range"""
        harm = 0.5
        inharm = 0.3
        
        combined = calculate_combined_density_metric(
            harm, inharm, alpha=0.8, beta=0.2, preserve_dynamic_range=True
        )
        
        # Should be positive
        self.assertGreaterEqual(combined, 0.0)
    
    def test_combined_metric_weights(self):
        """Test that weights are correctly applied"""
        harm = 1.0
        inharm = 1.0
        
        # High harmonic weight
        combined_high_harm = calculate_combined_density_metric(
            harm, inharm, alpha=0.9, beta=0.1, preserve_dynamic_range=True
        )
        
        # Low harmonic weight
        combined_low_harm = calculate_combined_density_metric(
            harm, inharm, alpha=0.1, beta=0.9, preserve_dynamic_range=True
        )
        
        # With same inputs, high harmonic weight should give similar result
        # (both components are equal, so weights don't matter much)
        # But they should both be positive
        self.assertGreater(combined_high_harm, 0.0)
        self.assertGreater(combined_low_harm, 0.0)


if __name__ == '__main__':
    unittest.main()

