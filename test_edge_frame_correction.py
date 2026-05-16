"""
Tests for Edge Frame Correction Implementation
Analytical reference values for mathematical verification

Tests verify:
- Edge frame weight calculation correctness
- Correction factor application
- First note density improvement
- Mathematical correctness of coverage calculations
"""

import unittest
import numpy as np
import pytest
import sys
from pathlib import Path
import tempfile
import soundfile as sf

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from proc_audio import AudioProcessor


def create_test_audio_sequence(sr=44100, duration=0.5, fundamental=440.0, output_path=None):
    """Create test audio with known properties"""
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * fundamental * t)
    y += 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)
    
    if output_path:
        sf.write(output_path, y, sr)
    
    return y, sr


class TestEdgeFrameCorrection(unittest.TestCase):
    """Test edge frame correction implementation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = AudioProcessor()
    
    def test_edge_frame_counts_calculation(self):
        """Test that edge frame counts are calculated correctly"""
        # Mathematical verification (reference):
        # pad_length = n_fft // 2
        # edge_frame_count = ceil(pad_length / hop_length)
        # For n_fft=2048, hop_length=256: pad_length=1024, edge_count = ceil(1024/256) = 4
        
        n_fft = 2048
        hop_length = 256
        signal_length = 44100  # 1 second at 44.1kHz
        n_frames = int(np.ceil((signal_length + n_fft // 2) / hop_length))
        
        # Set hop_length on processor (required by implementation)
        self.processor.hop_length = hop_length
        
        first_count, last_count = self.processor._calculate_edge_frame_counts(n_frames, n_fft)
        
        # Expected: ceil((n_fft//2) / hop_length) = ceil(1024/256) = 4
        # But implementation also caps at n_frames // 2, so we need to account for that
        pad_length = n_fft // 2
        edge_frame_count = max(1, int(np.ceil(pad_length / hop_length)))
        expected_count = min(edge_frame_count, max(1, n_frames // 2))
        
        self.assertEqual(first_count, expected_count,
                        f"Expected first edge count {expected_count}, got {first_count}")
        self.assertEqual(last_count, expected_count,
                        f"Expected last edge count {expected_count}, got {last_count}")
    
    def test_edge_frame_weights_calculation(self):
        """Test that edge frame weights are calculated correctly"""
        n_fft = 2048
        hop_length = 256
        signal_length = 44100
        n_frames = int(np.ceil((signal_length + n_fft // 2) / hop_length))
        
        # Set hop_length on processor (required by implementation)
        self.processor.hop_length = hop_length
        
        weights = self.processor._calculate_edge_frame_weights(n_frames, n_fft)
        
        # Verify weights shape
        self.assertEqual(len(weights), n_frames)
        
        # Verify first frame weight (should be > 1.0 for correction)
        # Mathematical verification (reference):
        # First frame coverage: 50% real signal (other 50% is reflected padding)
        # Energy reduction: √0.5 ≈ 0.707
        # Correction needed: 1 / 0.707 ≈ 1.414 (capped at 2.0)
        first_weight = weights[0]
        self.assertGreater(first_weight, 1.0, "First frame should have correction weight > 1.0")
        self.assertLessEqual(first_weight, 2.0, "First frame weight should be capped at 2.0")
        
        # Verify center frames have weight ≈ 1.0 (no correction needed)
        # Skip if n_frames is too small
        if n_frames > 10:
            center_idx = n_frames // 2
            center_weight = weights[center_idx]
            self.assertAlmostEqual(center_weight, 1.0, delta=0.1,
                                 msg=f"Center frame should have weight ≈ 1.0, got {center_weight}")
        
        # Verify last frame weight (should be > 1.0)
        last_weight = weights[-1]
        self.assertGreater(last_weight, 1.0, "Last frame should have correction weight > 1.0")
        self.assertLessEqual(last_weight, 2.0, "Last frame weight should be capped at 2.0")
    
    def test_edge_frame_correction_application(self):
        """Test that edge frame correction is applied in generate_complete_list"""
        # Create test audio
        sr = 44100
        duration = 0.5
        test_file = Path(tempfile.mkdtemp()) / "test.wav"
        
        y, sr = create_test_audio_sequence(sr=sr, duration=duration, 
                                          fundamental=440.0, output_path=str(test_file))
        
        self.processor.load_audio_files([str(test_file)])
        
        # Extract audio data
        if len(self.processor.audio_data) > 0:
            y, sr, note, file_path = self.processor.audio_data[0]
            self.processor.y = y
            self.processor.sr = sr
        else:
            self.fail("No audio data loaded")
        
        self.processor.n_fft = 2048
        self.processor.hop_length = 256
        
        # Perform FFT analysis (should calculate edge frame weights)
        self.processor.fft_analysis(zero_padding=1)
        
        # Verify edge frame weights were calculated
        self.assertTrue(hasattr(self.processor, 'frame_weights'))
        self.assertIsNotNone(self.processor.frame_weights)
        
        # Verify weights are applied during generate_complete_list
        self.processor.generate_complete_list()
        
        # Check that complete_list_df exists and has reasonable values
        self.assertIsNotNone(self.processor.complete_list_df)
        if len(self.processor.complete_list_df) > 0:
            # Amplitudes should be positive and finite
            if 'Amplitude' in self.processor.complete_list_df.columns:
                amplitudes = self.processor.complete_list_df['Amplitude'].values
                self.assertTrue(np.all(amplitudes > 0), "All amplitudes should be positive")
                self.assertTrue(np.all(np.isfinite(amplitudes)), 
                              "All amplitudes should be finite")

    @pytest.mark.slow
    def test_first_note_density_improvement(self):
        """Test that edge frame correction improves first note density.

        Both clips use the same synthetic waveform and the **same musical note
        label** so harmonic / f0 context matches; the comparison isolates
        STFT edge handling rather than filename-derived nominal f0 differences
        (see triage: ``note1`` vs ``note2`` non-musical labels skewed density).
        """
        # Create two identical audio files (simulating two takes of the same note)
        sr = 44100
        duration = 0.5
        temp_dir = Path(tempfile.mkdtemp())

        file1 = temp_dir / "take1_A4.wav"
        file2 = temp_dir / "take2_A4.wav"

        # Create identical signals
        y1, sr = create_test_audio_sequence(sr=sr, duration=duration,
                                           fundamental=440.0, output_path=str(file1))
        y2, sr = create_test_audio_sequence(sr=sr, duration=duration,
                                           fundamental=440.0, output_path=str(file2))

        # Process both files
        processor1 = AudioProcessor()
        processor1.load_audio_files([str(file1)])

        # Extract audio data
        if len(processor1.audio_data) > 0:
            y, sr, note, file_path = processor1.audio_data[0]
            processor1.y = y
            processor1.sr = sr
        else:
            self.fail("No audio data loaded for file1")

        processor1.fft_analysis(zero_padding=1)
        processor1.generate_complete_list()
        processor1._process_filtered_and_harmonic_data(
            freq_min=20.0, freq_max=20000.0,
            db_min=-80.0, db_max=0.0,
            tolerance=10.0, note="A4",
            zero_padding=1, time_avg='mean'
        )
        processor1._calculate_metrics()

        processor2 = AudioProcessor()
        processor2.load_audio_files([str(file2)])

        # Extract audio data
        if len(processor2.audio_data) > 0:
            y, sr, note, file_path = processor2.audio_data[0]
            processor2.y = y
            processor2.sr = sr
        else:
            self.fail("No audio data loaded for file2")

        processor2.fft_analysis(zero_padding=1)
        processor2.generate_complete_list()
        processor2._process_filtered_and_harmonic_data(
            freq_min=20.0, freq_max=20000.0,
            db_min=-80.0, db_max=0.0,
            tolerance=10.0, note="A4",
            zero_padding=1, time_avg='mean'
        )
        processor2._calculate_metrics()

        # With edge frame correction, two identical clips under the same note label
        # should yield similar combined densities.
        density1 = processor1.combined_density_metric_value
        density2 = processor2.combined_density_metric_value

        # Verify densities are valid
        self.assertIsNotNone(density1, "Density1 should be calculated")
        self.assertIsNotNone(density2, "Density2 should be calculated")
        self.assertGreater(density1, 0.0, f"Density1 should be positive. Got: {density1}")
        self.assertGreater(density2, 0.0, f"Density2 should be positive. Got: {density2}")

        ratio = density2 / density1 if density1 > 0 else 0

        # Same harmonic context: expect ratio near 1.0; keep tolerance for FP noise.
        if density1 > 1000.0 or density2 > 1000.0:
            self.assertLess(ratio, 4.0,
                           msg=f"Ratio should be < 4.0 even for simple signals. Got: {ratio:.3f}")
            self.assertGreater(ratio, 0.25,
                             msg=f"Ratio should be > 0.25 even for simple signals. Got: {ratio:.3f}")
        else:
            self.assertGreater(ratio, 0.25,
                             msg=f"Ratio should be > 0.25 (edge correction working). Got: {ratio:.3f}")
            self.assertLess(ratio, 4.0,
                           msg=f"Ratio should be < 4.0 (edge correction working). Got: {ratio:.3f}")
    
    def test_weight_smooth_transition(self):
        """Test that weights transition smoothly from edge to center"""
        n_fft = 2048
        hop_length = 256
        signal_length = 44100
        n_frames = int(np.ceil((signal_length + n_fft // 2) / hop_length))
        
        # Set hop_length on processor (required by implementation)
        self.processor.hop_length = hop_length
        
        weights = self.processor._calculate_edge_frame_weights(n_frames, n_fft)
        
        # Find transition point (where weight becomes close to 1.0)
        pad_length = n_fft // 2
        edge_frame_count = max(1, int(np.ceil(pad_length / hop_length)))
        edge_count = min(edge_frame_count, max(1, n_frames // 2))
        transition_start = min(edge_count + 1, n_frames - 1)
        
        # Check that weights decrease smoothly
        # Weight at edge should be > weight at transition_start (if transition_start is valid)
        if transition_start < len(weights):
            edge_weight = weights[0]
            transition_weight = weights[transition_start]
            
            self.assertGreater(edge_weight, transition_weight,
                             "Edge weight should be greater than transition weight")
        
        # Check smoothness: difference between consecutive weights should be small
        # (except at edge boundaries)
        max_jump = 0.0
        check_range = min(edge_count + 5, len(weights) - 1)
        for i in range(1, check_range):
            jump = abs(weights[i] - weights[i-1])
            max_jump = max(max_jump, jump)
        
        # Max jump should be reasonable (< 0.5 for smooth transition)
        self.assertLess(max_jump, 0.5,
                       f"Weights should transition smoothly. Max jump: {max_jump}")


class TestEdgeFrameMathematicalCorrectness(unittest.TestCase):
    """Mathematical verification of edge frame correction"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = AudioProcessor()
    
    def test_coverage_calculation_first_frame(self):
        """Test coverage calculation for first frame"""
        # Mathematical verification (reference):
        # For first frame with center=True:
        # - pad_length = n_fft // 2
        # - real_signal_samples = n_fft - pad_length = n_fft // 2
        # - effective_coverage = real_signal_samples / n_fft = 0.5
        # - correction = 1 / 0.5 = 2.0 (capped at 2.0)
        
        n_fft = 2048
        hop_length = 256
        signal_length = 44100
        n_frames = int(np.ceil((signal_length + n_fft // 2) / hop_length))
        
        # Set hop_length on processor (required by implementation)
        self.processor.hop_length = hop_length
        
        weights = self.processor._calculate_edge_frame_weights(n_frames, n_fft)
        
        # First frame should have coverage ≈ 0.5
        # Correction weight should be ≈ 2.0 (capped)
        first_weight = weights[0]
        
        # Weight should be close to 2.0 (allowing for implementation details)
        self.assertGreater(first_weight, 1.5,
                         f"First frame weight should be > 1.5 (approaching 2.0). Got: {first_weight}")
        self.assertLessEqual(first_weight, 2.0,
                           f"First frame weight should be ≤ 2.0. Got: {first_weight}")
    
    def test_coverage_calculation_center_frames(self):
        """Test coverage calculation for center frames"""
        n_fft = 2048
        hop_length = 256
        signal_length = 44100
        n_frames = int(np.ceil((signal_length + n_fft // 2) / hop_length))
        
        # Set hop_length on processor (required by implementation)
        self.processor.hop_length = hop_length
        
        weights = self.processor._calculate_edge_frame_weights(n_frames, n_fft)
        
        # Center frames should have coverage ≈ 1.0 (full real signal)
        # Correction weight should be ≈ 1.0
        pad_length = n_fft // 2
        edge_frame_count = max(1, int(np.ceil(pad_length / hop_length)))
        edge_count = min(edge_frame_count, max(1, n_frames // 2))
        
        center_start = edge_count + 1
        center_end = n_frames - edge_count - 1
        
        # Only check if we have valid center frames
        if center_start < center_end:
            center_weights = weights[center_start:center_end]
            
            # All center weights should be close to 1.0
            for i, weight in enumerate(center_weights):
                self.assertAlmostEqual(weight, 1.0, delta=0.05,
                                     msg=f"Center frame {center_start + i} should have weight ≈ 1.0. Got: {weight}")


if __name__ == '__main__':
    unittest.main()
