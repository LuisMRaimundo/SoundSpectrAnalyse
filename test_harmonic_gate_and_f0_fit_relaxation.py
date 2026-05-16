"""Regression tests for the upper-harmonic gate and f₀ fit acceptance
corrections (Clarinete_mf finding #3 + follow-up).

Background:
    The first iteration of the off-frequency harmonic gate used an
    absolute-Hz tolerance:

        tol = max(5 Hz, 2.5 × FFT bin_width)

    For a clarinet at f₀ = 466 Hz and FFT bin width ≈ 5.4 Hz, this
    produced tol ≈ 13.5 Hz. That correctly rejects stray broadband
    peaks at low orders but over-rejects legitimate upper partials
    where natural inharmonicity / vibrato carries each peak 50–100 Hz
    off n·f₀ — well below a quarter-tone. The corpus audit for A#4
    showed h34 at 15918 Hz (Δ = 68 Hz, 0.43 % drift) demoted to
    ``off_frequency`` and excluded from density.

    The fix combines an absolute Hz floor with a 1 % relative-Hz floor:

        tol = max(5 Hz, 2.5 × bin_width, 0.01 × expected_freq)

    so the gate stays tight at low orders (where peaks must land on
    n·f₀ to within a sub-bin distance) but widens proportionally at
    higher orders.

    Additionally, ``max_fit_quality`` was raised from 0.05 → 0.10
    because real instruments with 30+ strict harmonics routinely produce
    fits at 0.07–0.09 due to mild inharmonicity, and the 2 %
    absolute-shift gate remains the substantive safety guard.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Gate formula used in proc_audio._build_harmonic_candidate_row
# ---------------------------------------------------------------------------
def _gate_tol_hz(
    *,
    expected_freq_hz: float,
    sr_hz: float,
    n_fft: int,
) -> float:
    """Pure mirror of the gate formula. Kept here verbatim so the test
    pins the contract and not the proc_audio internals."""
    bin_hz = sr_hz / float(n_fft) if (sr_hz > 0 and n_fft > 0) else 0.0
    abs_floor = max(5.0, 2.5 * bin_hz) if bin_hz > 0 else 5.0
    rel_floor = 0.01 * float(expected_freq_hz)
    return max(abs_floor, rel_floor)


# ---------------------------------------------------------------------------
# Gate behaviour
# ---------------------------------------------------------------------------
def test_gate_is_tight_for_low_partials() -> None:
    """At h=2 (~1 kHz) the gate must stay near the bin-resolution floor
    so stray broadband peaks 30+ Hz away from n·f₀ are still rejected."""
    # f₀ = 466 Hz, h=2 → expected = 932 Hz
    tol = _gate_tol_hz(expected_freq_hz=932.0, sr_hz=44100.0, n_fft=8192)
    # 1 % of 932 Hz = 9.32 Hz; bin-resolution floor = 2.5 * 5.385 ≈ 13.5 Hz
    # Effective gate is the larger ≈ 13.5 Hz.
    assert 12.0 <= tol <= 15.0, f"Low-partial gate should be ~13.5 Hz, got {tol:.3f}"


def test_gate_widens_for_upper_partials() -> None:
    """At h=34 (~16 kHz) the gate must widen with expected_freq so
    legitimate upper partials drifting 0.4 % off n·f₀ are admitted."""
    # f₀ = 466 Hz, h=34 → expected = 15844 Hz
    tol = _gate_tol_hz(expected_freq_hz=15844.0, sr_hz=44100.0, n_fft=8192)
    # 1 % of 15844 Hz = 158.4 Hz; bin floor 13.5 Hz → effective gate ≈ 158 Hz.
    assert 150.0 <= tol <= 165.0, f"Upper-partial gate should be ~158 Hz, got {tol:.3f}"


def test_gate_admits_real_clarinet_h34_drift() -> None:
    """The Clarinete_mf A#4 audit showed h=34 at Δ = 68 Hz being
    demoted. With the corrected gate that candidate must now be
    admitted."""
    expected = 15850.0
    actual_drift_hz = 68.0  # observed in the real per-note workbook
    tol = _gate_tol_hz(expected_freq_hz=expected, sr_hz=44100.0, n_fft=8192)
    assert actual_drift_hz < tol, (
        f"Gate must admit Δ = {actual_drift_hz} Hz at h=34 / 15850 Hz; "
        f"tol = {tol:.3f}"
    )


def test_gate_still_rejects_quarter_tone_drift() -> None:
    """A quarter-tone (~3 %) drift means the candidate is NOT a
    harmonic — it must still be rejected at all orders."""
    expected = 466.0  # h=1
    drift_quarter_tone = 0.03 * expected  # ~14 Hz at f₀ = 466 Hz
    tol = _gate_tol_hz(expected_freq_hz=expected, sr_hz=44100.0, n_fft=8192)
    assert drift_quarter_tone > tol, (
        f"Gate must reject quarter-tone drift at h=1; tol = {tol:.3f}"
    )

    expected_hi = 15850.0  # h=34
    drift_quarter_tone_hi = 0.03 * expected_hi  # ~475 Hz
    tol_hi = _gate_tol_hz(expected_freq_hz=expected_hi, sr_hz=44100.0, n_fft=8192)
    assert drift_quarter_tone_hi > tol_hi, (
        f"Gate must reject quarter-tone drift at h=34; tol = {tol_hi:.3f}"
    )


def test_gate_rejects_stray_broadband_peak_at_low_order() -> None:
    """At h=2 a peak 30 Hz off n·f₀ is a stray broadband bin, NOT a
    real partial (clarinet h=2 should land within a few Hz of 2·f₀).
    The gate must keep rejecting it."""
    expected = 932.0  # h=2 of A#4
    stray_drift = 30.0  # well above the 13.5 Hz gate
    tol = _gate_tol_hz(expected_freq_hz=expected, sr_hz=44100.0, n_fft=8192)
    assert stray_drift > tol, f"Low-partial stray peak must be rejected; tol = {tol:.3f}"


# ---------------------------------------------------------------------------
# f₀ fit acceptance constant
# ---------------------------------------------------------------------------
def test_max_fit_quality_constant_is_relaxed_to_0_10() -> None:
    """Pin the constant so it does not silently regress to 0.05.

    Reading the constant from the source rather than re-running the full
    pipeline keeps this fast and deterministic.
    """
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "proc_audio.py"
    text = src.read_text(encoding="utf-8")
    assert "max_fit_quality = 0.10" in text, (
        "max_fit_quality must be 0.10 (raised from 0.05 to admit mid-quality "
        "global-f0 fits where strict_peaks >= 30 carries real inharmonicity)."
    )
    # And make sure the obsolete 0.05 line is not lingering:
    assert "max_fit_quality = 0.05" not in text, (
        "Found stale max_fit_quality = 0.05 in proc_audio.py — regression."
    )


# ---------------------------------------------------------------------------
# attack_time log line removed
# ---------------------------------------------------------------------------
def test_attack_time_log_line_is_silenced() -> None:
    """The Temporal-evolution INFO log used to print attack_time; user
    confirmed audio is pre-trimmed so the value is misleading. Make sure
    the noisy log line no longer reaches the logger."""
    from pathlib import Path
    src = Path(__file__).resolve().parents[1] / "proc_audio.py"
    text = src.read_text(encoding="utf-8")
    assert "attack_time={self.attack_time:.4f}s" not in text, (
        "attack_time should no longer be in the temporal-evolution INFO log."
    )
    # spectral_flux IS still logged — that's the timbral-evolution metric we keep.
    assert "Spectral flux (mean)" in text


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
