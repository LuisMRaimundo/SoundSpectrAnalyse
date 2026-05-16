"""
Benchmark cases based on synthetic signals with expected metric ranges.
"""

import json
import sys
import unittest
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).parent.parent
ANALYZER_DIR = ROOT / "audio_analysis"
BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"
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


def _generate_signal(case: dict, cfg: SignalConfig) -> np.ndarray:
    sig = case["signal"]
    if sig["type"] == "sine":
        signal = sine_wave(sig["freq_hz"], sig["amplitude"], cfg)
    elif sig["type"] == "harmonic_stack":
        signal = harmonic_stack(sig["f0_hz"], sig["harmonics"], sig["amplitudes"], cfg)
    elif sig["type"] == "inharmonic_mix":
        base = harmonic_stack(sig["f0_hz"], sig["harmonics"], sig["amplitudes"], cfg)
        inharm = inharmonic_tone(sig["inharmonic_freq_hz"], sig["inharmonic_amp"], cfg)
        signal = base + inharm
    elif sig["type"] == "subbass_mix":
        base = harmonic_stack(sig["f0_hz"], sig["harmonics"], sig["amplitudes"], cfg)
        subbass = sine_wave(sig["subbass_freq_hz"], sig["subbass_amp"], cfg)
        signal = base + subbass
    else:
        raise ValueError(f"Unknown benchmark signal type: {sig['type']}")
    return normalize_peak(signal)


class TestBenchmarks(unittest.TestCase):
    def setUp(self) -> None:
        self.cfg = SignalConfig(sample_rate=22050, duration_s=1.0)
        cases_path = BENCHMARKS_DIR / "benchmark_cases.json"
        self.cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]

    def test_benchmarks(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["name"]):
                audio_path = case.get("audio_path")
                if audio_path:
                    wav_path = (ROOT / audio_path).resolve()
                    if not wav_path.is_file():
                        pytest.skip(
                            "Optional benchmark audio fixture is absent in this curated GitHub export: "
                            f"{audio_path}"
                        )
                    results = _run_minimal_analysis(wav_path, self.cfg.sample_rate)
                else:
                    signal = _generate_signal(case, self.cfg)
                    with tempfile.TemporaryDirectory() as tmpdir:
                        wav_path = Path(tmpdir) / f"{case['name']}.wav"
                        sf.write(wav_path, signal.astype(np.float32), self.cfg.sample_rate)
                        results = _run_minimal_analysis(wav_path, self.cfg.sample_rate)

                stats = results.get("spectral_component_stats", {})
                expected = case["expected"]

                if "harmonic_energy_pct_musical_min" in expected:
                    self.assertGreater(
                        float(stats.get("harmonic_energy_pct_musical", 0.0)),
                        expected["harmonic_energy_pct_musical_min"],
                    )
                if "inharmonic_energy_pct_musical_max" in expected:
                    self.assertLess(
                        float(stats.get("inharmonic_energy_pct_musical", 0.0)),
                        expected["inharmonic_energy_pct_musical_max"],
                    )
                if "inharmonic_energy_pct_musical_min" in expected:
                    self.assertGreater(
                        float(stats.get("inharmonic_energy_pct_musical", 0.0)),
                        expected["inharmonic_energy_pct_musical_min"],
                    )
                if "subbass_energy_pct_global_max" in expected:
                    self.assertLess(
                        float(stats.get("subbass_energy_pct_global", 0.0)),
                        expected["subbass_energy_pct_global_max"],
                    )
                if "subbass_energy_pct_global_min" in expected:
                    self.assertGreater(
                        float(stats.get("subbass_energy_pct_global", 0.0)),
                        expected["subbass_energy_pct_global_min"],
                    )
                if "fundamental_freq_min" in expected:
                    self.assertGreater(
                        float(stats.get("f0_hz", 0.0) or 0.0),
                        expected["fundamental_freq_min"],
                    )


if __name__ == "__main__":
    unittest.main()
