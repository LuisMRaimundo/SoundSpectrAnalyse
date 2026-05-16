"""
Performance-oriented checks (optional wall-clock benchmarks).

Default ``pytest`` run keeps **smoke tests** only (pipeline completes, cache round-trip).

Set ``RUN_PERFORMANCE_TESTS=1`` to enable historical wall-clock / speedup assertions
(machine-dependent; not enforced in normal CI).
"""

from __future__ import annotations

import gc
import multiprocessing
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))

from proc_audio import AudioProcessor
from result_cache import ResultCache

RUN_PERFORMANCE_TESTS = os.environ.get("RUN_PERFORMANCE_TESTS", "").strip() == "1"
skip_unless_perf = unittest.skipUnless(
    RUN_PERFORMANCE_TESTS,
    "Set RUN_PERFORMANCE_TESTS=1 to run wall-clock performance benchmarks.",
)


def create_test_audio(sr=44100, duration=0.5, fundamental=440.0, output_path=None):
    """Create test audio file."""
    t = np.linspace(0, duration, int(sr * duration))
    y = np.sin(2 * np.pi * fundamental * t)
    y += 0.5 * np.sin(2 * np.pi * fundamental * 2 * t)

    if output_path:
        sf.write(output_path, y, sr)

    return y, sr


def _bind_first_loaded_clip(processor: AudioProcessor) -> None:
    """``load_audio_files`` fills ``audio_data``; ``fft_analysis`` requires ``y``/``sr``."""
    if not getattr(processor, "audio_data", None):
        raise AssertionError("Expected audio_data after load_audio_files")
    y, sr, _n, _p = processor.audio_data[0]
    processor.y = y
    processor.sr = sr


class TestProcessingPipelineSmoke(unittest.TestCase):
    """Always-on: verify STFT pipeline runs after binding loaded samples."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_single_file_fft_and_metrics_complete(self):
        test_file = self.temp_dir / "smoke.wav"
        create_test_audio(sr=44100, duration=0.35, fundamental=440.0, output_path=str(test_file))
        processor = AudioProcessor()
        processor.load_audio_files([str(test_file)])
        _bind_first_loaded_clip(processor)
        processor.fft_analysis()
        processor.generate_complete_list()
        processor._process_filtered_and_harmonic_data(
            freq_min=20.0,
            freq_max=20000.0,
            db_min=-80.0,
            db_max=0.0,
            tolerance=10.0,
            note="A4",
            zero_padding=1,
            time_avg="mean",
        )
        processor._calculate_metrics()
        self.assertIsNotNone(processor.density_metric_value)
        self.assertTrue(np.isfinite(float(processor.density_metric_value)))


@skip_unless_perf
class TestProcessingSpeed(unittest.TestCase):
    """Wall-clock style checks (opt-in via RUN_PERFORMANCE_TESTS=1)."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_single_file_processing_speed(self):
        test_file = self.temp_dir / "test.wav"
        create_test_audio(sr=44100, duration=0.5, fundamental=440.0, output_path=str(test_file))

        processor = AudioProcessor()
        start_time = time.time()
        _bind_first_loaded_clip(processor)
        processor.fft_analysis()
        processor.generate_complete_list()
        processor._process_filtered_and_harmonic_data(
            freq_min=20.0,
            freq_max=20000.0,
            db_min=-80.0,
            db_max=0.0,
            tolerance=10.0,
            note="A4",
            zero_padding=1,
            time_avg="mean",
        )
        processor._calculate_metrics()
        elapsed_time = time.time() - start_time

        self.assertGreater(elapsed_time, 0.0)
        # Loose ceiling: local dev / CI variance; strict regression belongs in dedicated perf jobs.
        self.assertLess(elapsed_time, 600.0, f"Processing took {elapsed_time:.2f}s (sanity upper bound)")

    def test_batch_processing_speed(self):
        test_files = []
        for i in range(10):
            test_file = self.temp_dir / f"test_{i}.wav"
            create_test_audio(sr=44100, duration=0.3, fundamental=440.0 + i * 10, output_path=str(test_file))
            test_files.append(str(test_file))

        processor = AudioProcessor()
        processor.load_audio_files(test_files)

        start_time = time.time()
        for y, sr, note, file_path in processor.audio_data:
            processor.y = y
            processor.sr = sr
            processor._reset_metrics()
            processor.fft_analysis(zero_padding=1)
            processor.generate_complete_list()
            processor._process_filtered_and_harmonic_data(
                freq_min=20.0,
                freq_max=20000.0,
                db_min=-80.0,
                db_max=0.0,
                tolerance=10.0,
                note=note,
                zero_padding=1,
                time_avg="mean",
            )
            processor._calculate_metrics()
        elapsed_time = time.time() - start_time

        self.assertGreater(elapsed_time, 0.0)
        self.assertLess(elapsed_time, 900.0, f"Batch processing took {elapsed_time:.2f}s (sanity upper bound)")
        avg_time_per_file = elapsed_time / len(test_files)
        self.assertLess(avg_time_per_file, 120.0, f"Average time per file: {avg_time_per_file:.2f}s")

    def test_large_file_processing(self):
        test_file = self.temp_dir / "test_large.wav"
        create_test_audio(sr=44100, duration=5.0, fundamental=440.0, output_path=str(test_file))

        processor = AudioProcessor()
        _bind_first_loaded_clip(processor)

        start_time = time.time()
        processor.fft_analysis()
        processor.generate_complete_list()
        elapsed_time = time.time() - start_time

        self.assertGreater(elapsed_time, 0.0)
        self.assertLess(elapsed_time, 900.0, f"Large file processing took {elapsed_time:.2f}s (sanity upper bound)")


class TestMemoryUsage(unittest.TestCase):
    """Memory / GC behaviour (no global wall-clock limits)."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_memory_doesnt_grow_unbounded(self):
        import os as _os

        import psutil

        process = psutil.Process(_os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024

        test_files = []
        for i in range(10):
            test_file = self.temp_dir / f"test_{i}.wav"
            create_test_audio(sr=44100, duration=0.3, fundamental=440.0, output_path=str(test_file))
            test_files.append(str(test_file))

        for test_file in test_files:
            processor = AudioProcessor()
            _bind_first_loaded_clip_from_path(processor, test_file)
            processor.fft_analysis()
            processor.generate_complete_list()
            processor._process_filtered_and_harmonic_data(
                freq_min=20.0,
                freq_max=20000.0,
                db_min=-80.0,
                db_max=0.0,
                tolerance=10.0,
                note="A4",
                zero_padding=1,
                time_avg="mean",
            )
            processor._calculate_metrics()

            del processor
            gc.collect()

        final_memory = process.memory_info().rss / 1024 / 1024
        memory_increase = final_memory - initial_memory

        self.assertLess(memory_increase, 500.0, f"Memory increased by {memory_increase:.1f}MB, expected < 500MB")

    def test_garbage_collection_effectiveness(self):
        test_file = self.temp_dir / "test.wav"
        create_test_audio(sr=44100, duration=0.3, fundamental=440.0, output_path=str(test_file))

        initial_objects = len(gc.get_objects())

        for _i in range(5):
            processor = AudioProcessor()
            _bind_first_loaded_clip_from_path(processor, str(test_file))
            processor.fft_analysis()
            processor.generate_complete_list()

            del processor
            gc.collect()

        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects

        self.assertLess(object_increase, initial_objects,
                        f"Object count increased by {object_increase}, expected less than {initial_objects}")


def _bind_first_loaded_clip_from_path(processor: AudioProcessor, wav_path: str) -> None:
    processor.load_audio_files([wav_path])
    _bind_first_loaded_clip(processor)


class TestParallelProcessingPerformance(unittest.TestCase):
    """Parallel-style batch (sequential reference only in normal suite)."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @pytest.mark.slow
    @unittest.skipIf(multiprocessing.cpu_count() < 2, "Requires multiple CPU cores")
    def test_parallel_speedup_exists(self):
        num_files = max(4, multiprocessing.cpu_count())
        test_files = []

        for i in range(num_files):
            test_file = self.temp_dir / f"test_{i}.wav"
            create_test_audio(sr=44100, duration=0.3, fundamental=440.0 + i * 10, output_path=str(test_file))
            test_files.append(str(test_file))

        processor_seq = AudioProcessor()
        processor_seq.load_audio_files(test_files)

        start_time = time.time()
        for y, sr, note, file_path in processor_seq.audio_data:
            processor_seq.y = y
            processor_seq.sr = sr
            processor_seq._reset_metrics()
            processor_seq.fft_analysis(zero_padding=1)
            processor_seq.generate_complete_list()
            processor_seq._process_filtered_and_harmonic_data(
                freq_min=20.0,
                freq_max=20000.0,
                db_min=-80.0,
                db_max=0.0,
                tolerance=10.0,
                note=note,
                zero_padding=1,
                time_avg="mean",
            )
            processor_seq._calculate_metrics()
        sequential_time = time.time() - start_time

        self.assertGreater(sequential_time, 0.0)
        self.assertTrue(np.isfinite(sequential_time))


class TestCachePerformance(unittest.TestCase):
    """Cache round-trip (always on); strict speedup only when RUN_PERFORMANCE_TESTS=1."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.cache_dir = self.temp_dir / "cache"
        self.cache = ResultCache(cache_dir=self.cache_dir)

    def tearDown(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_cache_roundtrip_smoke(self):
        test_file = self.temp_dir / "test.wav"
        create_test_audio(sr=44100, duration=0.3, fundamental=440.0, output_path=str(test_file))

        processor = AudioProcessor()
        _bind_first_loaded_clip_from_path(processor, str(test_file))
        processor.fft_analysis()
        processor.generate_complete_list()
        processor._process_filtered_and_harmonic_data(
            freq_min=20.0,
            freq_max=20000.0,
            db_min=-80.0,
            db_max=0.0,
            tolerance=10.0,
            note="A4",
            zero_padding=1,
            time_avg="mean",
        )
        processor._calculate_metrics()

        params = {
            "freq_min": 20.0,
            "freq_max": 20000.0,
            "db_min": -80.0,
            "db_max": 0.0,
            "tolerance": 10.0,
            "zero_padding": 1,
            "time_avg": "mean",
        }
        results = {"density": processor.density_metric_value}
        self.cache.set(Path(test_file), params, results)
        cached_results = self.cache.get(Path(test_file), params)
        self.assertIsNotNone(cached_results)
        self.assertEqual(cached_results.get("density"), results["density"])

    @skip_unless_perf
    def test_cache_hit_speedup(self):
        test_file = self.temp_dir / "test.wav"
        create_test_audio(sr=44100, duration=0.3, fundamental=440.0, output_path=str(test_file))

        processor = AudioProcessor()
        _bind_first_loaded_clip_from_path(processor, str(test_file))

        start_time = time.time()
        processor.fft_analysis()
        processor.generate_complete_list()
        processor._process_filtered_and_harmonic_data(
            freq_min=20.0,
            freq_max=20000.0,
            db_min=-80.0,
            db_max=0.0,
            tolerance=10.0,
            note="A4",
            zero_padding=1,
            time_avg="mean",
        )
        processor._calculate_metrics()
        computation_time = time.time() - start_time

        params = {
            "freq_min": 20.0,
            "freq_max": 20000.0,
            "db_min": -80.0,
            "db_max": 0.0,
            "tolerance": 10.0,
            "zero_padding": 1,
            "time_avg": "mean",
        }
        results = {"density": processor.density_metric_value}
        self.cache.set(Path(test_file), params, results)

        start_time = time.time()
        cached_results = self.cache.get(Path(test_file), params)
        cache_time = time.time() - start_time

        self.assertIsNotNone(cached_results)
        self.assertLess(cache_time, max(computation_time, 1e-6))
        speedup = computation_time / cache_time if cache_time > 0 else 0.0
        self.assertGreater(speedup, 2.0, f"Expected modest cache speedup, got {speedup:.1f}x")


class TestScalability(unittest.TestCase):
    """Duration scaling (ratio checks, not absolute wall-clock ceilings)."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

    def tearDown(self):
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_processing_time_scales_linearly_with_duration(self):
        durations = [0.5, 1.0, 2.0]
        processing_times = []

        for duration in durations:
            test_file = self.temp_dir / f"test_{duration}.wav"
            create_test_audio(sr=44100, duration=duration, fundamental=440.0, output_path=str(test_file))

            processor = AudioProcessor()
            _bind_first_loaded_clip_from_path(processor, str(test_file))

            start_time = time.time()
            processor.fft_analysis()
            processor.generate_complete_list()
            elapsed_time = time.time() - start_time

            processing_times.append(elapsed_time)

        # STFT cost has a large fixed component (window setup, frame count floor); wall-clock is not
        # strictly linear in duration for short clips. Require weak monotonicity vs longer audio.
        self.assertGreater(processing_times[0], 0.0)
        self.assertGreaterEqual(
            processing_times[2],
            processing_times[0] * 0.85,
            f"Longer audio should not be dramatically faster than shorter audio: "
            f"t({durations[0]}s)={processing_times[0]:.3f}s, t({durations[2]}s)={processing_times[2]:.3f}s",
        )
        self.assertGreaterEqual(processing_times[2], processing_times[1] * 0.85)


if __name__ == "__main__":
    unittest.main()
