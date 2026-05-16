"""Regression: user-visible reporting language, denominators, PCA wording."""

from __future__ import annotations

from pathlib import Path

import json
import tempfile
from compile_metrics import read_super_analysis_metrics
import pandas as pd

from harmonic_alignment import compute_harmonic_alignment_metrics


def _repo_roots() -> list[Path]:
    root = Path(__file__).resolve().parents[1]
    return [
        root / "harmonic_alignment.py",
        root / "harmonic_validation.py",
        root / "proc_audio.py",
        root / "compile_metrics.py",
        root / "audio_analysis" / "super_audio_analyzer.py",
        root / "audio_analysis" / "batch_audio_analyzer.py",
    ]


def test_no_spurious_token_in_core_sources() -> None:
    for p in _repo_roots():
        t = p.read_text(encoding="utf-8", errors="replace").lower()
        assert "spurious" not in t, p.name


def test_no_gold_standard_marketing_in_core_sources() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("proc_audio.py", "pyproject.toml"):
        t = (root / rel).read_text(encoding="utf-8", errors="replace").lower()
        assert "gold-standard" not in t and "gold standard" not in t, rel


def test_harmonic_alignment_emits_order_and_weighted_status_keys() -> None:
    f0 = 440.0
    df = pd.DataFrame({"Frequency (Hz)": [f0 * n for n in range(1, 35)], "Amplitude_linear": [1.0] * 34})
    ha = compute_harmonic_alignment_metrics(
        f0, df, sample_rate=44100.0, n_fft=4096, max_frequency_hz=20000.0, subbass_cutoff_hz=20.0
    )
    assert "harmonic_order_alignment_status" in ha
    assert "harmonic_order_alignment_weighted_status" in ha
    assert "harmonic_representative_energy_status" in ha
    assert "non_harmonic_candidate_energy_ratio" in ha


def test_musical_and_global_energy_percentages_sum_to_100_in_json_roundtrip() -> None:
    """spectral_component_stats uses distinct denominators; each triple sums to 100."""
    payload = {
        "spectral_metrics": {
            "harmonic_energy_percentage": 93.0,
            "inharmonic_energy_percentage": 7.0,
        },
        "spectral_component_stats": {
            "harmonic_energy_percentage_musical_band": 93.0,
            "inharmonic_energy_percentage_musical_band": 7.0,
            "harmonic_energy_percentage_global": 80.0,
            "inharmonic_energy_percentage_global": 15.0,
            "subbass_energy_percentage_global": 5.0,
            "harmonic_energy_percentage_semantics": "test",
            "harmonic_energy_sum": 80.0,
            "inharmonic_energy_sum": 15.0,
            "subbass_energy_sum": 5.0,
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as fh:
        fh.write(json.dumps(payload))
        path = fh.name
    try:
        m = read_super_analysis_metrics(path)
        assert abs(float(m["harmonic_energy_percentage_musical_band"] or 0) + float(m["inharmonic_energy_percentage_musical_band"] or 0) - 100.0) < 1e-6
        g = (
            float(m["harmonic_energy_percentage_global"] or 0)
            + float(m["inharmonic_energy_percentage_global"] or 0)
            + float(m["subbass_energy_percentage_global"] or 0)
        )
        assert abs(g - 100.0) < 1e-6
    finally:
        Path(path).unlink(missing_ok=True)


def test_compile_metrics_source_pca_log_not_unconditional_com_phrase() -> None:
    p = Path(__file__).resolve().parents[1] / "compile_metrics.py"
    text = p.read_text(encoding="utf-8", errors="replace")
    assert "Resultados (com PCA)" not in text


def test_peak_diagnostic_log_template_has_no_accuracy_superlative() -> None:
    p = Path(__file__).resolve().parents[1] / "audio_analysis" / "super_audio_analyzer.py"
    t = p.read_text(encoding="utf-8", errors="replace").lower()
    assert "more accurate" not in t
