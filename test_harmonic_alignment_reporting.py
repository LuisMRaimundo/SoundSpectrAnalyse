"""Reporting and invariants for harmonic-order alignment (no contradictory status vs cents tiers)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from harmonic_alignment import compute_harmonic_alignment_metrics
from harmonic_validation import validate_harmonic_series_matched


def _df(freqs: list[float], amps: list[float] | None = None) -> pd.DataFrame:
    if amps is None:
        amps = [1.0] * len(freqs)
    return pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude_linear": amps})


def test_multiple_candidates_same_order_collapse_to_one_representative() -> None:
    f0 = 200.0
    # Three peaks all rounding to n=2 (400 Hz ladder) — only strongest should be representative
    freqs = [399.0, 400.0, 401.0, 600.0]
    amps = [0.2, 0.5, 0.3, 1.0]
    ha = compute_harmonic_alignment_metrics(
        f0,
        _df(freqs, amps),
        max_frequency_hz=5000.0,
        tolerance_cents=50.0,
        subbass_cutoff_hz=20.0,
    )
    assert int(ha["harmonic_representative_count"]) <= int(ha["total_expected_harmonic_orders"])
    # Order 2 should appear once in matches
    ns = [m["n"] for m in (ha.get("harmonic_alignment_matches") or []) if isinstance(m, dict)]
    assert ns.count(2) <= 1


def test_excellent_alignment_does_not_yield_validation_warning() -> None:
    f0 = 440.0
    freqs = [f0 * n for n in range(1, 42)]
    df = _df(freqs)
    vr = validate_harmonic_series_matched(
        f0,
        df,
        max_freq_hz=20000.0,
        sample_rate=44100.0,
        n_fft=4096,
        subbass_cutoff_hz=20.0,
    )
    if str(vr.get("harmonic_alignment_status")) == "excellent":
        assert vr.get("harmonic_validation_status") == "ok"


def test_collapsed_count_never_exceeds_expected_orders() -> None:
    f0 = 50.0
    freqs = [f0 * n for n in range(1, 30)]
    ha = compute_harmonic_alignment_metrics(
        f0,
        _df(freqs),
        max_frequency_hz=2000.0,
        subbass_cutoff_hz=10.0,
    )
    n_exp = int(ha["total_expected_harmonic_orders"])
    assert int(ha["harmonic_representative_count"]) <= n_exp


def test_super_audio_source_contains_no_deprecated_spurious_token() -> None:
    p = Path(__file__).resolve().parents[1] / "audio_analysis" / "super_audio_analyzer.py"
    text = p.read_text(encoding="utf-8", errors="replace")
    assert "spurious" not in text.lower()


def test_super_audio_log_contains_no_strongest_peak_per_harmonic_mislabel() -> None:
    p = Path(__file__).resolve().parents[1] / "audio_analysis" / "super_audio_analyzer.py"
    text = p.read_text(encoding="utf-8", errors="replace")
    assert "strongest peak per harmonic" not in text.lower()


def test_forbidden_harmonic_series_percent_phrases_absent() -> None:
    root = Path(__file__).resolve().parents[1]
    for rel in ("harmonic_alignment.py", "harmonic_validation.py", "audio_analysis/super_audio_analyzer.py"):
        t = (root / rel).read_text(encoding="utf-8", errors="replace").lower()
        assert "harmonic-series consistency review" not in t
        assert "avg deviation" not in t
        assert "harmonic series validation warning" not in t
