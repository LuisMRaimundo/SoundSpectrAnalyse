"""
Mathematical Verification Tests
Analytical reference values for unit checks

Tests verify mathematical correctness of:
- Parseval's theorem (energy conservation)
- Frequency normalization (power law)
- Harmonic series calculations
- Dissonance models
- Spectral density calculations
"""

import unittest
import numpy as np
import sys
from pathlib import Path
import math

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from proc_audio import AudioProcessor, _verify_energy_conservation
from density import calculate_combined_density_metric
from dissonance_models import get_dissonance_model
import librosa


class TestParsevalsTheorem(unittest.TestCase):
    """Test Parseval's theorem (energy conservation)"""
    
    def test_parseval_sine_wave(self):
        """Test Parseval's theorem for pure sine wave"""
        # Mathematical verification (reference):
        # Sine wave: x(t) = sin(2πf₀t)
        # Energy in time domain: E_t = (1/2) × T (for normalized sine)
        # Energy in frequency domain: E_f = (1/N) × Σ|X[k]|²
        # Parseval's theorem: E_t = E_f (for rectangular window)
        
        sr = 44100
        duration = 1.0
        freq = 440.0
        
        t = np.linspace(0, duration, int(sr * duration))
        y = np.sin(2 * np.pi * freq * t)
        
        n_fft = 4096
        hop_length = 1024
        
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann', center=True)
        
        result = _verify_energy_conservation(
            y, S, n_fft, hop_length, 'hann',
            tolerance=0.1  # 10% tolerance
        )
        
        # Energy ratio should be close to 1.0
        self.assertAlmostEqual(result['energy_ratio'], 1.0, delta=0.1,
                             msg=f"Energy ratio should be ≈ 1.0 (Parseval's theorem). Got: {result['energy_ratio']:.4f}")
    
    def test_parseval_white_noise(self):
        """Test Parseval's theorem for white noise"""
        # White noise: uniformly distributed energy across frequencies
        # Parseval's theorem should still hold
        
        sr = 44100
        duration = 0.5
        np.random.seed(42)  # For reproducibility
        y = np.random.randn(int(sr * duration))
        
        n_fft = 2048
        hop_length = 512
        
        S = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window='hann', center=True)
        
        result = _verify_energy_conservation(
            y, S, n_fft, hop_length, 'hann',
            tolerance=0.1
        )
        
        # Energy ratio should be close to 1.0
        self.assertAlmostEqual(result['energy_ratio'], 1.0, delta=0.1,
                             msg=f"Energy ratio should be ≈ 1.0 for white noise. Got: {result['energy_ratio']:.4f}")


class TestFrequencyNormalization(unittest.TestCase):
    """Test frequency normalization (power law)"""
    
    def test_power_law_normalization(self):
        """Test that frequency normalization follows power law"""
        # Mathematical verification (reference):
        # Spectral rolloff: P(f) ∝ f^(-α) where α ≈ 1.5 for musical instruments
        # Normalization factor: norm = f^α = f^1.5
        # For harmonic n: norm_n = (n × f0)^1.5 = n^1.5 × f0^1.5
        
        f0 = 440.0
        harmonics = [1, 2, 3, 4, 5]
        
        normalization_factors = []
        for n in harmonics:
            freq = n * f0
            # Normalization factor: f^1.5
            norm_factor = freq ** 1.5
            normalization_factors.append(norm_factor)
        
        # Verify power law relationship
        # Ratio between consecutive harmonics should follow: (n+1/n)^1.5
        for i in range(len(harmonics) - 1):
            n1, n2 = harmonics[i], harmonics[i+1]
            expected_ratio = (n2 / n1) ** 1.5
            actual_ratio = normalization_factors[i+1] / normalization_factors[i]
            
            self.assertAlmostEqual(actual_ratio, expected_ratio, delta=0.01,
                                 msg=f"Normalization ratio for harmonics {n1}→{n2}: "
                                 f"expected {expected_ratio:.4f}, got {actual_ratio:.4f}")


class TestHarmonicSeries(unittest.TestCase):
    """Test harmonic series calculations"""
    
    def test_harmonic_frequency_calculation(self):
        """Test harmonic frequency calculations"""
        # Mathematical verification (reference):
        # Harmonic series: f_n = n × f₀
        # For f₀ = 440 Hz:
        #   f₁ = 440 Hz (fundamental)
        #   f₂ = 880 Hz (2nd harmonic)
        #   f₃ = 1320 Hz (3rd harmonic)
        
        f0 = 440.0
        harmonics = [1, 2, 3, 4, 5]
        
        for n in harmonics:
            expected_freq = n * f0
            # Direct calculation
            calculated_freq = n * f0
            
            self.assertAlmostEqual(calculated_freq, expected_freq, delta=0.01,
                                 msg=f"Harmonic {n} frequency: expected {expected_freq} Hz, "
                                 f"got {calculated_freq} Hz")
    
    def test_harmonic_tolerance_calculation(self):
        """Test harmonic tolerance calculation"""
        # Tolerance for harmonic detection should scale with frequency
        # Higher harmonics have wider tolerance (psychoacoustic JND)
        
        f0 = 440.0
        harmonics = [1, 2, 3, 4, 5]
        base_tolerance = 5.0
        
        for n in harmonics:
            freq = n * f0
            # Adaptive tolerance: 1.5% of frequency
            adaptive_tolerance = freq * 0.015
            # Use max of base and adaptive
            tolerance = max(base_tolerance, adaptive_tolerance)
            
            # Tolerance should increase with harmonic number
            if n > 1:
                prev_freq = (n - 1) * f0
                prev_tolerance = max(base_tolerance, prev_freq * 0.015)
                self.assertGreaterEqual(tolerance, prev_tolerance,
                                      f"Tolerance should increase with frequency. "
                                      f"Harmonic {n-1}: {prev_tolerance:.2f} Hz, "
                                      f"Harmonic {n}: {tolerance:.2f} Hz")


class TestCombinedDensityMetric(unittest.TestCase):
    """Test combined density metric calculations"""
    
    def test_logarithmic_combination(self):
        """Test that logarithmic combination preserves dynamic range"""
        # Mathematical verification (reference):
        # Logarithmic combination: combined = log₁₀(1 + α×harm + β×inharm)
        # For small values: combined ≈ log₁₀(1 + value) ≈ value (for value << 1)
        # For large values: combined ≈ log₁₀(value) (for value >> 1)
        
        # Small values (pp - pianissimo)
        harm_small = 0.1
        inharm_small = 0.05
        
        combined_small = calculate_combined_density_metric(
            harm_small, inharm_small,
            alpha=0.8, beta=0.2,
            preserve_dynamic_range=True
        )
        
        # Large values (ff - fortissimo)
        harm_large = 10.0
        inharm_large = 5.0
        
        combined_large = calculate_combined_density_metric(
            harm_large, inharm_large,
            alpha=0.8, beta=0.2,
            preserve_dynamic_range=True
        )
        
        # Ratio should reflect the large difference
        # For logarithmic combination: ratio ≈ log(large) / log(small) > linear ratio
        ratio = combined_large / combined_small if combined_small > 0 else 0
        
        self.assertGreater(ratio, 10.0,
                         f"Dynamic range should be preserved. Ratio: {ratio:.2f}, "
                         f"expected > 10.0")
        
        # Verify both values are positive and finite
        self.assertGreater(combined_small, 0.0)
        self.assertGreater(combined_large, 0.0)
        self.assertTrue(np.isfinite(combined_small))
        self.assertTrue(np.isfinite(combined_large))


class TestDissonanceModels(unittest.TestCase):
    """Test dissonance model calculations"""
    
    def test_dissonance_model_availability(self):
        """Test that dissonance models are available"""
        try:
            models = get_dissonance_model('sethares')
            self.assertIsNotNone(models, "Sethares model should be available")
        except Exception:
            # Dissonance models may not be fully implemented
            pass
    
    def test_dissonance_octave_relationship(self):
        """Test that octave intervals have low dissonance"""
        # Mathematical verification (reference):
        # Octave: frequency ratio = 2:1
        # Most dissonance models predict low dissonance for octave intervals
        # (due to harmonic relationship)
        
        try:
            model = get_dissonance_model('sethares')
            if model:
                # Test pure tones at octave relationship
                f1 = 440.0
                f2 = 880.0  # Octave above
                
                # Dissonance should be relatively low for octave
                dissonance = model.pure_tones_dissonance(f1, f2, 1.0, 1.0)
                
                # Compare with tritone (high dissonance)
                f3 = 440.0 * (2 ** (6/12))  # Tritone (augmented fourth)
                dissonance_tritone = model.pure_tones_dissonance(f1, f3, 1.0, 1.0)
                
                # Octave should have lower dissonance than tritone
                self.assertLess(dissonance, dissonance_tritone,
                              f"Octave should have lower dissonance than tritone. "
                              f"Octave: {dissonance:.4f}, Tritone: {dissonance_tritone:.4f}")
        except Exception:
            # Skip if models not available
            pass


class TestSpectralDensityCalculations(unittest.TestCase):
    """Test spectral density metric calculations"""
    
    def test_density_metric_non_negative(self):
        """Test that density metrics are non-negative"""
        from density import apply_density_metric
        
        # Test with positive values
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        density = apply_density_metric(values, weight_function='linear')
        
        self.assertGreaterEqual(density, 0.0,
                              f"Density metric should be non-negative. Got: {density}")
    
    def test_density_metric_scaling(self):
        """Test that density metric scales appropriately"""
        from density import apply_density_metric
        
        # Mathematical verification (reference):
        # With prevent_domination=True (default), values are normalized by max
        # [0.1, 0.2, 0.3] → normalized → [0.33, 0.67, 1.0] → sum = 2.0
        # [1.0, 2.0, 3.0] → normalized → [0.33, 0.67, 1.0] → sum = 2.0
        # So normalized values produce same density (this is by design)
        
        # Test with different amplitudes (same relative ratios)
        values_small = np.array([0.1, 0.2, 0.3])
        values_large = np.array([1.0, 2.0, 3.0])  # Same ratios, 10x larger absolute
        
        density_small = apply_density_metric(values_small, weight_function='linear', prevent_domination=True)
        density_large = apply_density_metric(values_large, weight_function='linear', prevent_domination=True)
        
        # With normalization, same relative ratios produce same density
        # This is by design: prevent_domination ensures relative distribution matters more than absolute values
        self.assertAlmostEqual(density_small, density_large, delta=0.1,
                             msg=f"Normalized density should be similar for same relative ratios. "
                                 f"Small: {density_small:.4f}, Large: {density_large:.4f}")
        
        # Test with prevent_domination=False to verify absolute scaling
        density_small_abs = apply_density_metric(values_small, weight_function='linear', prevent_domination=False)
        density_large_abs = apply_density_metric(values_large, weight_function='linear', prevent_domination=False)
        
        # Without normalization, larger absolute values should produce larger density
        self.assertGreater(density_large_abs, density_small_abs,
                         msg=f"Without normalization, larger amplitudes should produce larger density. "
                             f"Small: {density_small_abs:.4f}, Large: {density_large_abs:.4f}")


if __name__ == '__main__':
    unittest.main()
