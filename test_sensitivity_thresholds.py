"""
Lightweight sensitivity checks to ensure metrics respond to parameter changes.
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
from tests.reference_signal_utils import SignalConfig, harmonic_stack, normalize_peak


def _run_minimal_analysis(audio_path: Path, sample_rate: int, noise_floor_db: float, window: str) -> dict:
    analyzer = SuperAudioAnalyzer(
        audio_path=audio_path,
        output_dir=audio_path.parent / "analysis_out",
        sample_rate=sample_rate,
        use_90_tier=False,
        harmonic_tolerance=0.03,
        window=window,
        use_adaptive_tolerance=True,
        auto_extract_weights=True,
    )
    analyzer.noise_floor_db = noise_floor_db
    analyzer.load_audio()
    analyzer.compute_spectrogram()
    analyzer.detect_fundamental_frequency()
    analyzer.separate_harmonic_inharmonic()
    analyzer.calculate_spectral_metrics()
    return analyzer.results


class TestSensitivityThresholds(unittest.TestCase):
    def test_noise_floor_impacts_subbass(self) -> None:
        cfg = SignalConfig(sample_rate=22050, duration_s=1.0)
        signal = harmonic_stack(220.0, [1, 2, 3], [1.0, 0.6, 0.3], cfg)
        signal = normalize_peak(signal)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sens.wav"
            sf.write(wav_path, signal.astype(np.float32), cfg.sample_rate)
            low_nf = _run_minimal_analysis(wav_path, cfg.sample_rate, noise_floor_db=-80.0, window="hann")
            high_nf = _run_minimal_analysis(wav_path, cfg.sample_rate, noise_floor_db=-40.0, window="hann")

        low_stats = low_nf.get("spectral_component_stats", {})
        high_stats = high_nf.get("spectral_component_stats", {})

        low_inharm = int(low_stats.get("inharmonic_peak_count", 0) or 0)
        high_inharm = int(high_stats.get("inharmonic_peak_count", 0) or 0)

        # Expect a reduction in detected inharmonic components with a higher noise floor
        self.assertGreater(low_inharm - high_inharm, 20)

    def test_window_impacts_energy_distribution(self) -> None:
        cfg = SignalConfig(sample_rate=22050, duration_s=1.0)
        signal = harmonic_stack(220.0, [1, 2, 3], [1.0, 0.6, 0.3], cfg)
        signal = normalize_peak(signal)

        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = Path(tmpdir) / "sens_win.wav"
            sf.write(wav_path, signal.astype(np.float32), cfg.sample_rate)
            hann = _run_minimal_analysis(wav_path, cfg.sample_rate, noise_floor_db=-60.0, window="hann")
            bh = _run_minimal_analysis(wav_path, cfg.sample_rate, noise_floor_db=-60.0, window="blackmanharris")

        hann_stats = hann.get("spectral_component_stats", {})
        bh_stats = bh.get("spectral_component_stats", {})

        hann_harm = float(hann_stats.get("harmonic_energy_pct_musical", 0.0))
        bh_harm = float(bh_stats.get("harmonic_energy_pct_musical", 0.0))

        # Expect a measurable change across windows
        self.assertNotAlmostEqual(hann_harm, bh_harm, delta=0.5)


if __name__ == "__main__":
    unittest.main()
