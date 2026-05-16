"""
Reference signal tests for harmonic/inharmonic/subbass energy partitioning.
These tests validate the "mass/fatness" metrics on controlled synthetic inputs.
"""

import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent
ANALYZER_DIR = ROOT / "audio_analysis"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ANALYZER_DIR))

from super_audio_analyzer import SuperAudioAnalyzer  # type: ignore
from tests.reference_signal_utils import (
    SignalConfig,
    harmonic_stack,
    inharmonic_tone,
    normalize_peak,
    sine_wave,
)


def _run_minimal_analysis(audio_path: Path, sample_rate: int) -> dict:
    analyzer = SuperAudioAnalyzer(
        audio_path=audio_path,
        output_dir=audio_path.parent / "analysis_out",
        sample_rate=sample_rate,
        use_90_tier=False,
        harmonic_tolerance=0.03,
        window="hann",
        use_adaptive_tolerance=True,
        auto_extract_weights=True,
    )
    analyzer.load_audio()
    analyzer.compute_spectrogram()
    analyzer.detect_fundamental_frequency()
    analyzer.separate_harmonic_inharmonic()
    analyzer.calculate_spectral_metrics()
    return analyzer.results


class TestReferenceSignals(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SignalConfig(sample_rate=22050, duration_s=1.0)

    def _write_and_analyze(self, signal: np.ndarray) -> dict:
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "ref.wav"
            sf.write(wav_path, signal.astype(np.float32), self.cfg.sample_rate)
            return _run_minimal_analysis(wav_path, self.cfg.sample_rate)

    def test_pure_sine_is_mostly_harmonic(self) -> None:
        signal = sine_wave(440.0, 1.0, self.cfg)
        signal = normalize_peak(signal)
        results = self._write_and_analyze(signal)
        stats = results.get("spectral_component_stats", {})

        harmonic_pct = float(stats.get("harmonic_energy_pct_musical", 0.0))
        inharm_pct = float(stats.get("inharmonic_energy_pct_musical", 0.0))
        subbass_pct = float(stats.get("subbass_energy_pct_global", 0.0))

        self.assertGreater(harmonic_pct, 95.0)
        self.assertLess(inharm_pct, 5.0)
        self.assertLess(subbass_pct, 2.0)

    def test_harmonic_stack_is_mostly_harmonic(self) -> None:
        signal = harmonic_stack(220.0, [1, 2, 3], [1.0, 0.6, 0.3], self.cfg)
        signal = normalize_peak(signal)
        results = self._write_and_analyze(signal)
        stats = results.get("spectral_component_stats", {})

        harmonic_pct = float(stats.get("harmonic_energy_pct_musical", 0.0))
        inharm_pct = float(stats.get("inharmonic_energy_pct_musical", 0.0))

        self.assertGreater(harmonic_pct, 90.0)
        self.assertLess(inharm_pct, 10.0)

    def test_inharmonic_component_increases_inharmonic_energy(self) -> None:
        base = harmonic_stack(220.0, [1, 2], [1.0, 0.5], self.cfg)
        inharm = inharmonic_tone(220.0 * 1.6, 0.7, self.cfg)
        signal = normalize_peak(base + inharm)
        results = self._write_and_analyze(signal)
        stats = results.get("spectral_component_stats", {})

        inharm_pct = float(stats.get("inharmonic_energy_pct_musical", 0.0))
        self.assertGreater(inharm_pct, 8.0)

    def test_subbass_increases_subbass_energy(self) -> None:
        base = harmonic_stack(440.0, [1, 2], [1.0, 0.5], self.cfg)
        subbass = sine_wave(110.0, 0.3, self.cfg)
        signal = normalize_peak(base + subbass)
        results = self._write_and_analyze(signal)
        stats = results.get("spectral_component_stats", {})

        f0 = float(stats.get("f0_hz", 0.0) or 0.0)
        subbass_pct = float(stats.get("subbass_energy_pct_global", 0.0))

        # Ensure fundamental is not pulled into subbass region
        self.assertGreater(f0, 300.0)
        self.assertGreater(subbass_pct, 3.0)


if __name__ == "__main__":
    unittest.main()
