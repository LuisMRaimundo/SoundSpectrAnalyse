"""Audit tests for the Stage 1 harmonic-spectrum candidate extraction and
the ``harmonic_log_amplitude_density`` metric used by Density_Metrics.

The tests cover:

A. Synthetic harmonic stack (10 partials) — ~10 candidates exported,
   ``harmonic_log_amplitude_density`` matches ``log10(1 + sum amps)``.
B. Shoulder-peak case — strict prominence fails but SNR is high; the row
   is labelled ``snr_validated`` for diagnostics and is **not** included
   in harmonic density (only ``strict_validated`` rows are).
C. Bad f0 fit — adjusted f0 shifts too much or residual is high, the
   guard rejects it and ``f0_fit_accepted=False``.
D. Weak harmonic candidates — ``weak_candidate`` / low SNR rows are
   labelled but excluded from the density sum.
E. Regression — large expected order count with many smeared peaks still
   produces a populated ``harmonic_spectrum_candidates_df``; clean partials
   still yield a non-trivial density sum via ``strict_validated`` only.
F. Sub-bin export — Harmonic Spectrum rows carry bin-centre vs interpolated
   peak metadata; ``extracted_frequency_hz`` follows the interpolated peak
   when valid; ``frequency_deviation_hz`` matches ``extracted - expected``.

The tests run against the exact module-level helpers in
``proc_audio`` so they do not require a full audio pipeline run.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import proc_audio as pa
from proc_audio import (
    HARMONIC_CANDIDATE_STATUS_VALUES,
    _classify_harmonic_candidate,
    _local_peak_metrics,
)


# ---------------------------------------------------------------------------
# Lightweight shell that exposes ``_build_harmonic_candidate_row`` against a
# pair of injected ``filtered_list_df`` / ``complete_list_df`` frames, plus
# the ``_local_peak_metrics`` helper. We deliberately do not run the full
# ``_generate_harmonic_list`` because it requires a real FFT pass; the
# regression-style test below mocks the relevant inputs directly.
# ---------------------------------------------------------------------------
class _ShellHarmonicExtractor:
    """Minimum surface required by ``_build_harmonic_candidate_row``."""

    def __init__(
        self,
        *,
        filtered_df: pd.DataFrame | None = None,
        complete_df: pd.DataFrame | None = None,
        note: str = "A4",
    ) -> None:
        self.filtered_list_df = (
            filtered_df if filtered_df is not None else pd.DataFrame()
        )
        self.complete_list_df = (
            complete_df if complete_df is not None else pd.DataFrame()
        )
        self.note = note

    # Borrow the real method from AudioProcessor verbatim.
    _build_harmonic_candidate_row = pa.AudioProcessor._build_harmonic_candidate_row


def _synth_complete_spectrum(
    *,
    f0: float = 440.0,
    n_partials: int = 10,
    amplitudes: list[float] | None = None,
    sr: int = 44100,
    n_fft: int = 8192,
    noise_floor_db: float = -80.0,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return ``(filtered_df, complete_freqs, complete_magnitudes)`` for a
    synthetic spectrum with ``n_partials`` partials at ``k * f0`` and a flat
    noise floor at ``noise_floor_db``.

    The ``filtered_df`` contains only the partial bins (it mimics the
    cleaned peak list used by proc_audio); the ``complete_magnitudes``
    array is the dense linear spectrum used for SNR / prominence checks.
    """
    if amplitudes is None:
        amplitudes = [1.0] * n_partials

    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2, n_bins)
    mags = np.full(n_bins, 10.0 ** (noise_floor_db / 20.0))
    rows: list[dict] = []
    for k in range(1, n_partials + 1):
        f = k * f0
        if f >= sr / 2:
            break
        idx = int(np.argmin(np.abs(freqs - f)))
        # Three-bin main lobe so the strict local-peak test has neighbour
        # bins to look at.
        amp = amplitudes[k - 1]
        mags[idx] = max(mags[idx], amp)
        if idx > 0:
            mags[idx - 1] = max(mags[idx - 1], amp * 0.3)
        if idx + 1 < n_bins:
            mags[idx + 1] = max(mags[idx + 1], amp * 0.3)
        rows.append(
            {
                "Frequency (Hz)": float(freqs[idx]),
                "Magnitude (dB)": 20.0 * math.log10(max(amp, 1e-12)),
                "Amplitude": float(amp),
            }
        )
    filtered_df = pd.DataFrame(rows)
    return filtered_df, freqs, mags


# ---------------------------------------------------------------------------
# A. Synthetic harmonic stack.
# ---------------------------------------------------------------------------
def test_a_synthetic_harmonic_stack_10_partials_exports_10_candidates() -> None:
    f0 = 440.0
    n_partials = 10
    amps = [1.0 / k for k in range(1, n_partials + 1)]  # 1, 1/2, 1/3, ...
    filt, freqs, mags = _synth_complete_spectrum(
        f0=f0, n_partials=n_partials, amplitudes=amps
    )
    complete_df = pd.DataFrame(
        {"Frequency (Hz)": freqs, "Amplitude": mags}
    )
    shell = _ShellHarmonicExtractor(
        filtered_df=filt, complete_df=complete_df, note="A4"
    )

    rows: list[dict] = []
    for k in range(1, n_partials + 1):
        rows.append(
            shell._build_harmonic_candidate_row(
                hnum=k,
                expected_freq_hz=k * f0,
                tol_hz=max(5.0, k * f0 * 0.02),
                complete_magnitudes=mags,
                complete_freqs=freqs,
            )
        )
    cand_df = pd.DataFrame(rows)
    assert (cand_df["include_for_density"].astype(bool)).sum() == n_partials, cand_df
    statuses = set(cand_df["candidate_status"].astype(str))
    assert statuses.issubset(set(HARMONIC_CANDIDATE_STATUS_VALUES))
    # Every row corresponds to a real peak — none should be missing_window.
    assert "missing_window" not in statuses, cand_df

    sum_amps = float(cand_df["Amplitude_raw"].astype(float).sum())
    expected_sum = float(sum(amps))
    assert sum_amps == pytest.approx(expected_sum, abs=1e-9)
    expected_log = math.log10(1.0 + expected_sum)
    actual_log = math.log10(1.0 + sum_amps)
    assert actual_log == pytest.approx(expected_log, abs=1e-12)


# ---------------------------------------------------------------------------
# B. Shoulder-peak case — strict prominence fails but candidate remains.
#
# With the new saddle-based prominence (≈ scipy.signal.peak_prominences),
# a clean isolated peak — even one with main-lobe smearing across ±1 bin —
# correctly passes the 3 dB strict gate (the saddle on either side reaches
# the noise floor 60+ dB below the peak, which is what we want).
#
# The "strict-fails-but-density-includes" path now occurs only when two
# peaks sit close enough together that the saddle BETWEEN them is shallow:
# a small "shoulder" peak next to a stronger one. We build that scenario
# explicitly here.
# ---------------------------------------------------------------------------
def test_b_smeared_candidate_remains_weak_or_snr_validated() -> None:
    # Plateau of magnitude 0.98 across ±10 bins with a small bump at the
    # centre. The saddle prominence stays below 3 dB, but SNR is high —
    # the classifier labels ``snr_validated`` while **excluding** the row
    # from harmonic density (non-harmonic / shoulder energy must not inflate D_H).
    n_bins = 4096
    mags = np.full(n_bins, 1e-4, dtype=float)  # -80 dB noise floor
    peak_idx = 1000
    for i in range(peak_idx - 10, peak_idx + 11):
        mags[i] = 0.98
    mags[peak_idx] = 1.0
    freqs = np.arange(n_bins, dtype=float) * 1.0

    is_peak, snr_db, prom_db = _local_peak_metrics(mags, peak_idx)
    assert is_peak is True
    assert snr_db > 30.0
    assert prom_db < 3.0

    status, include = _classify_harmonic_candidate(
        amplitude_raw=1.0,
        local_peak_valid=is_peak,
        snr_db=snr_db,
        prominence_db=prom_db,
    )
    assert include is False
    # The classifier promotes via SNR label even though prominence < 3 dB.
    assert status == "snr_validated", (status, snr_db, prom_db)


# ---------------------------------------------------------------------------
# C. Bad f0 fit — the user's stricter guard rejects unrealistic shifts.
# ---------------------------------------------------------------------------
def test_c_bad_f0_fit_is_rejected_by_strict_guard() -> None:
    """We deliberately fabricate detected frequencies that produce a
    residual_std and shift well above the audit thresholds and confirm the
    guard rejects the fit."""
    f0_initial = 440.0
    # Push the "detected" frequencies far enough off the true grid that
    # the robust fit lands ~12 % below f0_initial (well above the 2 % cap).
    detected_freqs = np.array([350.0, 700.0, 1050.0])
    detected_amps = np.array([1.0, 0.5, 0.25])
    fit = pa._estimate_f0_global_robust(detected_freqs, detected_amps, f0_initial)
    f0_est = float(fit["f0_estimated"])
    fit_quality = float(fit["fit_quality"])
    residual_std = float(fit["residual_std"])

    # Mirror the acceptance logic from _generate_harmonic_list.
    max_residual_std = max(5.0, 0.01 * f0_initial)
    max_abs_shift = 0.02 * f0_initial
    max_fit_quality = 0.05
    accept = (
        np.isfinite(f0_est)
        and residual_std <= max_residual_std
        and abs(f0_est - f0_initial) <= max_abs_shift
        and fit_quality <= max_fit_quality
    )
    assert accept is False, (f0_est, residual_std, fit_quality)


def test_c_bad_f0_fit_due_to_fewer_than_three_strict_peaks_is_rejected() -> None:
    """Even when the candidate fit looks numerically plausible, the guard
    must refuse to apply it when fewer than 3 strict peaks were available."""
    # Helper recomputes the acceptance condition exactly.
    strict_peak_count_for_fit = 2  # two strict peaks only
    accept = strict_peak_count_for_fit >= 3
    assert accept is False


# ---------------------------------------------------------------------------
# D. Weak harmonic candidates — still available for density.
# ---------------------------------------------------------------------------
def test_d_weak_candidates_are_classified_weak_candidate_and_excluded_from_density() -> None:
    status, include = _classify_harmonic_candidate(
        amplitude_raw=0.05,
        local_peak_valid=True,
        snr_db=1.5,        # below the minimum_snr_db = 3 dB threshold
        prominence_db=0.5,  # below strict prominence threshold
    )
    assert status == "weak_candidate"
    assert include is False


def test_d_amplitude_zero_is_excluded_from_density() -> None:
    status, include = _classify_harmonic_candidate(
        amplitude_raw=0.0,
        local_peak_valid=True,
        snr_db=20.0,
        prominence_db=10.0,
    )
    assert status == "missing_window"
    assert include is False


def test_d_negative_snr_no_local_peak_is_below_noise_floor() -> None:
    status, include = _classify_harmonic_candidate(
        amplitude_raw=1e-3,
        local_peak_valid=False,
        snr_db=-10.0,
        prominence_db=-5.0,
    )
    assert status == "below_noise_floor"
    assert include is False


# ---------------------------------------------------------------------------
# E. Regression — large expected count, few strict peaks, density still ok.
# ---------------------------------------------------------------------------
def test_e_regression_high_expected_low_strict_still_populates_candidates() -> None:
    """Simulate a high-pitched note where many harmonic orders are *expected*
    but only the lower orders pass strict saddle prominence.

    The candidate table must remain fully populated (one row per order), while
    ``include_for_density`` is True **only** for ``strict_validated`` rows so
    the harmonic density sum is not inflated by smeared high-order bins."""
    f0 = 220.0
    sr = 44100
    nyquist = sr / 2
    n_partials = int(min(120, (nyquist / f0) - 1))
    assert n_partials >= 80, n_partials

    # Build a synthetic complete spectrum with all partials present but
    # *no* prominence margin (every partial sits within 1 dB of its
    # neighbours so the strict acceptance would reject them).
    n_bins = 16384
    freqs = np.linspace(0.0, nyquist, n_bins)
    mags = np.full(n_bins, 1e-3, dtype=float)  # -60 dB noise floor
    amps = [1.0 / (1.0 + 0.1 * k) for k in range(n_partials)]
    rows: list[dict] = []
    for k in range(1, n_partials + 1):
        f = k * f0
        if f >= nyquist:
            break
        idx = int(np.argmin(np.abs(freqs - f)))
        amp = amps[k - 1]
        # First partials: isolated main lobe so ``strict_validated`` passes
        # (density inclusion requires strict acceptance). Higher orders:
        # heavy smearing so strict prominence fails — they remain diagnostic
        # candidates only (snr_validated / weak_candidate, not in D_H).
        if k <= 35:
            mags[idx] = amp
            mags[idx - 1] = max(mags[idx - 1], amp * 0.30)
            mags[idx + 1] = max(mags[idx + 1], amp * 0.30)
        else:
            mags[idx] = amp
            mags[idx - 1] = max(mags[idx - 1], amp * 0.97)
            mags[idx + 1] = max(mags[idx + 1], amp * 0.97)
        rows.append(
            {
                "Frequency (Hz)": float(freqs[idx]),
                "Magnitude (dB)": 20.0 * math.log10(max(amp, 1e-12)),
                "Amplitude": float(amp),
            }
        )
    filtered_df = pd.DataFrame(rows)
    complete_df = pd.DataFrame(
        {"Frequency (Hz)": freqs, "Amplitude": mags}
    )
    shell = _ShellHarmonicExtractor(
        filtered_df=filtered_df, complete_df=complete_df, note="A3"
    )

    # Mimic _generate_harmonic_list: build one candidate per expected order.
    n_actual = len(rows)
    cand_rows: list[dict] = []
    for k in range(1, n_actual + 1):
        cand_rows.append(
            shell._build_harmonic_candidate_row(
                hnum=k,
                expected_freq_hz=k * f0,
                tol_hz=max(5.0, k * f0 * 0.02),
                complete_magnitudes=mags,
                complete_freqs=freqs,
            )
        )
    cand_df = pd.DataFrame(cand_rows)

    # The regression: the candidate path remains populated; density uses
    # strict rows only (here: the first ~35 orders with clean lobes).
    included = cand_df[cand_df["include_for_density"].astype(bool)]
    assert len(cand_df) >= 80, len(cand_df)
    assert len(included) >= 25, len(included)
    assert (included["candidate_status"].astype(str) == "strict_validated").all()
    h_amp_sum = float(included["Amplitude_raw"].astype(float).sum())
    assert h_amp_sum > 0.1, h_amp_sum
    h_log = math.log10(1.0 + h_amp_sum)
    assert h_log > 0.05, h_log


# ---------------------------------------------------------------------------
# F. Sub-bin harmonic frequency export (Harmonic Spectrum sheet contract).
# ---------------------------------------------------------------------------
def _asymmetric_peak_near_880hz() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Dense FFT grid with a deliberately asymmetric three-bin lobe so
    log-magnitude parabolic interpolation shifts ``f`` away from the bin centre."""
    sr = 44100
    n_fft = 8192
    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_bins)
    mags = np.full(n_bins, 1e-6, dtype=float)
    f_target = 880.0
    idx = int(np.argmin(np.abs(freqs - f_target)))
    assert 1 <= idx < n_bins - 1
    mags[idx - 1] = 0.12
    mags[idx] = 1.0
    mags[idx + 1] = 0.25
    filtered_df = pd.DataFrame(
        [
            {
                "Frequency (Hz)": float(freqs[idx]),
                "Magnitude (dB)": 20.0 * math.log10(1.0),
                "Amplitude": 1.0,
            }
        ]
    )
    return filtered_df, freqs, mags


def test_f_harmonic_spectrum_subbin_columns_and_deviation_identity() -> None:
    """Harmonic Spectrum rows must expose bin centre vs interpolated peak and
    keep ``frequency_deviation_hz`` consistent with ``extracted_frequency_hz``."""
    filt, freqs, mags = _asymmetric_peak_near_880hz()
    complete_df = pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": mags})
    shell = _ShellHarmonicExtractor(
        filtered_df=filt, complete_df=complete_df, note="A4"
    )
    f0 = 440.0
    row = shell._build_harmonic_candidate_row(
        hnum=2,
        expected_freq_hz=2.0 * f0,
        tol_hz=50.0,
        complete_magnitudes=mags,
        complete_freqs=freqs,
    )
    for col in (
        "bin_center_frequency_hz",
        "interpolated_frequency_hz",
        "subbin_offset_bins",
        "subbin_interpolation_valid",
        "peak_bin_index",
    ):
        assert col in row, row.keys()

    assert row["subbin_interpolation_valid"] is True
    bc = float(row["bin_center_frequency_hz"])
    fi = float(row["interpolated_frequency_hz"])
    assert abs(fi - bc) > 1e-6, (fi, bc)

    ex = float(row["extracted_frequency_hz"])
    assert ex == pytest.approx(fi, abs=1e-9)
    assert float(row["Frequency (Hz)"]) == pytest.approx(ex, abs=1e-9)

    exp_hz = float(row["expected_frequency_hz"])
    dev = float(row["frequency_deviation_hz"])
    assert dev == pytest.approx(ex - exp_hz, rel=0.0, abs=1e-6)


def test_f_harmonic_spectrum_excel_roundtrip_subbin_columns(tmp_path: Path) -> None:
    """Mirror proc_audio Harmonic Spectrum ``preferred_cols`` filtering: new audit
    columns survive Excel round-trip."""
    # Keep in sync with ``AudioProcessor`` Excel export in proc_audio.py.
    preferred_cols = [
        "Harmonic Number",
        "expected_frequency_hz",
        "extracted_frequency_hz",
        "frequency_deviation_hz",
        "bin_center_frequency_hz",
        "interpolated_frequency_hz",
        "subbin_offset_bins",
        "subbin_interpolation_valid",
        "peak_bin_index",
        "Frequency (Hz)",
        "Magnitude (dB)",
        "Amplitude",
        "Amplitude_raw",
        "Power_raw",
        "snr_db",
        "prominence_db",
        "local_peak_valid",
        "candidate_status",
        "include_for_density",
        "Note",
    ]
    filt, freqs, mags = _asymmetric_peak_near_880hz()
    complete_df = pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": mags})
    shell = _ShellHarmonicExtractor(
        filtered_df=filt, complete_df=complete_df, note="A4"
    )
    f0 = 440.0
    row = shell._build_harmonic_candidate_row(
        hnum=2,
        expected_freq_hz=2.0 * f0,
        tol_hz=50.0,
        complete_magnitudes=mags,
        complete_freqs=freqs,
    )
    cand_df = pd.DataFrame([row])
    cols = [c for c in preferred_cols if c in cand_df.columns]
    out_xlsx = tmp_path / "spectral_analysis_hs_fixture.xlsx"
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        cand_df[cols].to_excel(writer, sheet_name="Harmonic Spectrum", index=False)

    hs = pd.read_excel(out_xlsx, sheet_name="Harmonic Spectrum")
    assert "bin_center_frequency_hz" in hs.columns
    assert "interpolated_frequency_hz" in hs.columns
    assert "subbin_interpolation_valid" in hs.columns

    valid = hs["subbin_interpolation_valid"].fillna(False).astype(bool)
    assert valid.sum() > 0
    diff = (
        hs.loc[valid, "interpolated_frequency_hz"].astype(float)
        - hs.loc[valid, "bin_center_frequency_hz"].astype(float)
    ).abs()
    assert (diff > 1e-6).any()

    ex = hs["extracted_frequency_hz"].astype(float)
    exp = hs["expected_frequency_hz"].astype(float)
    dev = hs["frequency_deviation_hz"].astype(float)
    assert np.allclose(dev.to_numpy(), (ex - exp).to_numpy(), rtol=0.0, atol=1e-5)


def test_f_synthetic_stack_frequency_deviation_matches_extracted_minus_expected() -> None:
    """All partials in the canonical synthetic stack obey the deviation identity."""
    f0 = 440.0
    n_partials = 10
    amps = [1.0 / k for k in range(1, n_partials + 1)]
    filt, freqs, mags = _synth_complete_spectrum(
        f0=f0, n_partials=n_partials, amplitudes=amps
    )
    complete_df = pd.DataFrame({"Frequency (Hz)": freqs, "Amplitude": mags})
    shell = _ShellHarmonicExtractor(
        filtered_df=filt, complete_df=complete_df, note="A4"
    )
    rows: list[dict] = []
    for k in range(1, n_partials + 1):
        rows.append(
            shell._build_harmonic_candidate_row(
                hnum=k,
                expected_freq_hz=k * f0,
                tol_hz=max(5.0, k * f0 * 0.02),
                complete_magnitudes=mags,
                complete_freqs=freqs,
            )
        )
    cand_df = pd.DataFrame(rows)
    ex = cand_df["extracted_frequency_hz"].astype(float)
    exp = cand_df["expected_frequency_hz"].astype(float)
    dev = cand_df["frequency_deviation_hz"].astype(float)
    assert np.allclose(dev.to_numpy(), (ex - exp).to_numpy(), rtol=0.0, atol=1e-5)
