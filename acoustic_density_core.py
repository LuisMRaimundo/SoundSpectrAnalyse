#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
acoustic_density_core.py

A small, explicit acoustic-density core for pitched instrumental spectra.

Purpose
-------
This module separates acoustically different constructs instead of collapsing
them into one unstable scalar:

1. harmonic_occupancy_ratio
2. harmonic_effective_power_density_normalized
3. residual_log_frequency_occupancy
4. residual_energy_ratio
5. spectral_entropy
6. effective_partial_density
7. f0 provenance / acoustic verification status

It is designed to be called from proc_audio.py, compile_metrics.py, or an
Excel-export stage. It deliberately does not use "lowest detected harmonic" as
f0.

Inputs
------
A pandas DataFrame with at least one frequency column and one amplitude/power
column. Accepted aliases:

frequency:
    "Frequency (Hz)", "frequency_hz", "freq_hz", "frequency"

amplitude:
    "Amplitude", "amplitude", "amp", "magnitude_linear"

dB magnitude:
    "Magnitude (dB)", "magnitude_db", "db"

power:
    "Power", "power", "power_raw"

No external audio I/O is performed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

import math
import numpy as np
import pandas as pd


EPS = 1e-12


@dataclass(frozen=True)
class F0Triplet:
    """Authoritative f0 provenance for downstream acoustic descriptors."""
    f0_hz: float
    f0_source: str
    acoustic_f0_status: str
    f0_fit_accepted: bool


def _finite_positive(x: Any) -> bool:
    try:
        xf = float(x)
        return math.isfinite(xf) and xf > 0.0
    except Exception:
        return False


def canonical_f0_triplet(
    *,
    f0_final_hz: Optional[float] = None,
    f0_initial_hz: Optional[float] = None,
    f0_prior_hz: Optional[float] = None,
    f0_fit_accepted: Optional[bool] = None,
    f0_source: Optional[str] = None,
) -> F0Triplet:
    """
    Select f0 without ever using the lowest detected spectral peak.

    Policy
    ------
    - If the fitted/acoustic f0 was accepted and f0_final_hz is valid, use it.
    - Otherwise use nominal/prior fallback if available, but mark it as not
      acoustically verified.
    - If no valid f0 exists, return NaN and an explicit invalid status.
    """

    accepted = bool(f0_fit_accepted)

    if accepted and _finite_positive(f0_final_hz):
        return F0Triplet(
            f0_hz=float(f0_final_hz),
            f0_source=str(f0_source or "f0_final_hz"),
            acoustic_f0_status="fit_accepted_acoustically_verified",
            f0_fit_accepted=True,
        )

    # If the fit was rejected, f0_final_hz may contain a nominal fallback.
    # That can be useful for slot construction, but it is NOT acoustic proof.
    for value, source_name in (
        (f0_initial_hz, "f0_initial_hz_nominal_or_initial"),
        (f0_prior_hz, "f0_prior_hz_nominal"),
        (f0_final_hz, "f0_final_hz_fallback"),
    ):
        if _finite_positive(value):
            return F0Triplet(
                f0_hz=float(value),
                f0_source=str(f0_source or source_name),
                acoustic_f0_status="nominal_fallback_used_not_acoustically_verified",
                f0_fit_accepted=False,
            )

    return F0Triplet(
        f0_hz=float("nan"),
        f0_source="missing",
        acoustic_f0_status="missing_invalid_f0",
        f0_fit_accepted=False,
    )


def _first_existing_column(df: pd.DataFrame, names: tuple[str, ...]) -> Optional[str]:
    lower_to_original = {str(c).strip().lower(): c for c in df.columns}
    for name in names:
        c = lower_to_original.get(name.lower())
        if c is not None:
            return c
    return None


def _extract_peak_vectors(peaks_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return frequency_hz and power vectors from a permissive peak table."""
    if peaks_df is None or peaks_df.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    f_col = _first_existing_column(
        peaks_df,
        ("Frequency (Hz)", "frequency_hz", "freq_hz", "frequency", "freq"),
    )
    if f_col is None:
        raise ValueError("No frequency column found in peaks_df.")

    freq = pd.to_numeric(peaks_df[f_col], errors="coerce").to_numpy(float)

    p_col = _first_existing_column(peaks_df, ("Power", "power", "power_raw"))
    if p_col is not None:
        power = pd.to_numeric(peaks_df[p_col], errors="coerce").to_numpy(float)
    else:
        a_col = _first_existing_column(
            peaks_df,
            ("Amplitude", "amplitude", "amp", "magnitude_linear"),
        )
        db_col = _first_existing_column(
            peaks_df,
            ("Magnitude (dB)", "magnitude_db", "db", "level_db"),
        )

        if a_col is not None:
            amp = pd.to_numeric(peaks_df[a_col], errors="coerce").to_numpy(float)
        elif db_col is not None:
            db = pd.to_numeric(peaks_df[db_col], errors="coerce").to_numpy(float)
            # Treat dB values as linear-amplitude reference conversion.
            amp = np.power(10.0, db / 20.0)
        else:
            raise ValueError("No amplitude, dB, or power column found in peaks_df.")

        power = np.square(np.maximum(amp, 0.0))

    ok = np.isfinite(freq) & np.isfinite(power) & (freq > 0.0) & (power > 0.0)
    return freq[ok].astype(float), power[ok].astype(float)


def _normalized_entropy(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=float)
    p = p[np.isfinite(p) & (p > 0.0)]
    if p.size <= 1:
        return 0.0
    p = p / max(float(np.sum(p)), EPS)
    h = -float(np.sum(p * np.log2(np.maximum(p, EPS))))
    hmax = math.log2(p.size)
    return float(np.clip(h / hmax if hmax > 0 else 0.0, 0.0, 1.0))


def _effective_count(power: np.ndarray) -> float:
    p = np.asarray(power, dtype=float)
    p = p[np.isfinite(p) & (p > 0.0)]
    if p.size == 0:
        return 0.0
    total = float(np.sum(p))
    if total <= 0.0:
        return 0.0
    return float((total * total) / max(float(np.sum(p * p)), EPS))


def _expected_harmonic_orders(
    f0_hz: float,
    *,
    freq_min_hz: float,
    freq_max_hz: float,
) -> np.ndarray:
    if not _finite_positive(f0_hz):
        return np.array([], dtype=int)
    n0 = max(1, int(math.ceil(freq_min_hz / f0_hz)))
    n1 = max(0, int(math.floor(freq_max_hz / f0_hz)))
    if n1 < n0:
        return np.array([], dtype=int)
    return np.arange(n0, n1 + 1, dtype=int)


def compute_acoustic_density_descriptors(
    peaks_df: pd.DataFrame,
    *,
    f0_hz: float,
    f0_source: str = "",
    acoustic_f0_status: str = "",
    f0_fit_accepted: bool = False,
    freq_min_hz: float = 20.0,
    freq_max_hz: float = 20000.0,
    harmonic_tolerance_cents: float = 35.0,
    min_relative_db: float = -60.0,
    residual_log_bin_cents: float = 100.0,
    subbass_upper_ratio: float = 0.75,
    body_freq_min_hz: float = 20.0,
    body_freq_max_hz: float = 5000.0,
    body_peak_relative_db: float = -45.0,
    body_weight_knee_hz: float = 1800.0,
    low_mid_upper_hz: float = 2000.0,
    residual_body_contribution_cap: float = 0.25,
    salient_harmonic_relative_db: float = -45.0,
    salient_harmonic_ceiling_hz: float = 5000.0,
    density_summation_mode: str = "his_weighted",
    harmonic_density_weight: float = 1.0,
    inharmonic_density_weight: float = 0.5,
    subbass_density_weight: float = 0.25,
    density_salience_threshold_db: float = -45.0,
    density_frequency_ceiling_hz: float = 5000.0,
) -> dict[str, Any]:
    """
    Compute separated acoustic descriptors from a peak/component table.

    The returned descriptors are designed for export. No descriptor here should
    be silently averaged with legacy "Combined Density Metric" fields.
    """
    freq, power = _extract_peak_vectors(peaks_df)

    out: dict[str, Any] = {
        "f0_used_for_density_hz": float(f0_hz) if _finite_positive(f0_hz) else float("nan"),
        "f0_used_for_density_source": str(f0_source or ""),
        "acoustic_f0_status": str(acoustic_f0_status or ""),
        "f0_fit_accepted": bool(f0_fit_accepted),
        "expected_harmonic_slot_count": 0,
        "detected_harmonic_slot_count": 0,
        "harmonic_occupancy_ratio": 0.0,
        "harmonic_effective_partial_count": 0.0,
        "harmonic_effective_power_density_normalized": 0.0,
        "residual_log_frequency_occupancy": 0.0,
        "residual_energy_ratio": 0.0,
        "subbass_energy_ratio": 0.0,
        "harmonic_energy_ratio": 0.0,
        "spectral_entropy": 0.0,
        "effective_partial_density": 0.0,
        "body_weighted_effective_density": 0.0,
        "low_mid_energy_ratio": 0.0,
        "harmonic_body_density": 0.0,
        "expected_harmonic_slots_up_to_5000hz": 0,
        "harmonic_body_density_normalized": 0.0,
        "residual_body_contribution": 0.0,
        "residual_body_contribution_capped": 0.0,
        "salient_harmonic_order_count_up_to_5000hz": 0,
        "expected_harmonic_order_count_up_to_5000hz": 0,
        "salient_harmonic_coverage_up_to_5000hz": 0.0,
        "salient_harmonic_mass_up_to_5000hz": 0.0,
        "salient_harmonic_order_count_up_to_density_ceiling_hz": 0,
        "expected_harmonic_order_count_up_to_density_ceiling_hz": 0,
        "salient_harmonic_coverage_up_to_density_ceiling_hz": 0.0,
        "salient_harmonic_mass_up_to_density_ceiling_hz": 0.0,
        "salient_odd_harmonic_count_up_to_5000hz": 0,
        "salient_even_harmonic_count_up_to_5000hz": 0,
        "odd_even_harmonic_energy_ratio": 0.0,
        "salient_inharmonic_log_bin_count_up_to_5000hz": 0,
        "salient_subbass_particle_count": 0,
        "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz": 0,
        "salient_subbass_particle_count_up_to_density_ceiling_hz": 0,
        "final_note_density_count_based": 0.0,
        "final_note_density_salience_weighted": 0.0,
        "harmonic_density_component": 0.0,
        "inharmonic_density_component": 0.0,
        "subbass_density_component": 0.0,
        "harmonic_density_weight": float(harmonic_density_weight),
        "inharmonic_density_weight": float(inharmonic_density_weight),
        "subbass_density_weight": float(subbass_density_weight),
        "density_summation_mode": str(density_summation_mode or "his_weighted"),
        "density_salience_threshold_db": float(density_salience_threshold_db),
        "density_frequency_ceiling_hz": float(density_frequency_ceiling_hz),
        "density_metric_raw": float("nan"),
        "energy_weighted_component_density_diagnostic": float("nan"),
        "arithmetic_validation_status": "passed",
        "acoustic_validation_status": (
            "passed" if bool(f0_fit_accepted) else "nominal_fallback_used_not_acoustically_verified"
        ),
    }

    if freq.size == 0 or power.size == 0 or not _finite_positive(f0_hz):
        out["arithmetic_validation_status"] = "failed_missing_spectrum_or_f0"
        out["acoustic_validation_status"] = (
            "failed_missing_f0" if not _finite_positive(f0_hz) else out["acoustic_validation_status"]
        )
        return out

    freq_min_hz = float(max(freq_min_hz, 1e-6))
    freq_max_hz = float(max(freq_max_hz, freq_min_hz))
    in_range = (freq >= freq_min_hz) & (freq <= freq_max_hz)
    freq = freq[in_range]
    power = power[in_range]

    if freq.size == 0:
        out["arithmetic_validation_status"] = "failed_no_peaks_in_frequency_range"
        return out

    # Relative thresholding by power, using dB relative to the strongest retained peak.
    pmax = float(np.max(power))
    rel_power_threshold = pmax * (10.0 ** (float(min_relative_db) / 10.0))
    significant = power >= rel_power_threshold
    freq_sig = freq[significant]
    power_sig = power[significant]

    if freq_sig.size == 0:
        out["arithmetic_validation_status"] = "failed_no_significant_peaks"
        return out

    orders_expected = _expected_harmonic_orders(
        float(f0_hz),
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
    )
    expected_count = int(orders_expected.size)
    out["expected_harmonic_slot_count"] = expected_count

    # Classify each significant peak by nearest harmonic order in cents.
    nearest_order = np.rint(freq_sig / float(f0_hz)).astype(int)
    valid_order = nearest_order >= 1
    predicted = nearest_order.astype(float) * float(f0_hz)
    cents_error = 1200.0 * np.log2(np.maximum(freq_sig, EPS) / np.maximum(predicted, EPS))
    harmonic_peak_mask = valid_order & (np.abs(cents_error) <= float(harmonic_tolerance_cents))

    subbass_upper_hz = max(freq_min_hz, float(subbass_upper_ratio) * float(f0_hz))
    subbass_mask = freq_sig < subbass_upper_hz
    harmonic_peak_mask = harmonic_peak_mask & ~subbass_mask
    residual_mask = ~(harmonic_peak_mask | subbass_mask)

    detected_orders = np.unique(nearest_order[harmonic_peak_mask])
    if expected_count > 0:
        detected_orders = detected_orders[np.isin(detected_orders, orders_expected)]

    detected_count = int(detected_orders.size)
    out["detected_harmonic_slot_count"] = detected_count
    out["harmonic_occupancy_ratio"] = (
        float(detected_count / expected_count) if expected_count > 0 else 0.0
    )

    harmonic_power = power_sig[harmonic_peak_mask]
    residual_power = power_sig[residual_mask]
    subbass_power = power_sig[subbass_mask]
    total_power = float(np.sum(power_sig))

    h_energy = float(np.sum(harmonic_power))
    r_energy = float(np.sum(residual_power))
    s_energy = float(np.sum(subbass_power))

    if total_power > 0.0:
        out["harmonic_energy_ratio"] = h_energy / total_power
        out["residual_energy_ratio"] = r_energy / total_power
        out["subbass_energy_ratio"] = s_energy / total_power

    h_eff = _effective_count(harmonic_power)
    out["harmonic_effective_partial_count"] = h_eff
    out["harmonic_effective_power_density_normalized"] = (
        float(h_eff / expected_count) if expected_count > 0 else 0.0
    )

    # Residual occupancy on a log-frequency grid.
    residual_freq = freq_sig[residual_mask]
    if residual_freq.size > 0 and freq_max_hz > freq_min_hz:
        total_bins = int(math.ceil(1200.0 * math.log2(freq_max_hz / freq_min_hz) / residual_log_bin_cents))
        total_bins = max(total_bins, 1)
        bin_idx = np.floor(
            1200.0 * np.log2(np.maximum(residual_freq, freq_min_hz) / freq_min_hz)
            / residual_log_bin_cents
        ).astype(int)
        bin_idx = bin_idx[(bin_idx >= 0) & (bin_idx < total_bins)]
        out["residual_log_frequency_occupancy"] = float(len(np.unique(bin_idx)) / total_bins)

    out["spectral_entropy"] = _normalized_entropy(power_sig)
    out["effective_partial_density"] = _effective_count(power_sig)

    # Body-focused thickness descriptors (salient peaks, 20..5000 Hz default).
    bmin = float(max(body_freq_min_hz, freq_min_hz, 1e-6))
    bmax = float(max(bmin, min(body_freq_max_hz, freq_max_hz)))
    bmask = (freq_sig >= bmin) & (freq_sig <= bmax)
    body_freq = freq_sig[bmask]
    body_power = power_sig[bmask]
    if body_power.size > 0:
        bpmax = float(np.max(body_power))
        body_rel_thr = bpmax * (10.0 ** (float(body_peak_relative_db) / 10.0))
        salient_mask = body_power >= body_rel_thr
        body_freq = body_freq[salient_mask]
        body_power = body_power[salient_mask]

    if body_power.size > 0:
        salience = np.sqrt(np.maximum(body_power, 0.0))
        knee = float(max(body_weight_knee_hz, 1e-6))
        w_body = 1.0 / (1.0 + np.square(body_freq / knee))
        wx = w_body * salience
        out["body_weighted_effective_density"] = _effective_count(wx)

        low_mid_mask = body_freq <= float(max(low_mid_upper_hz, bmin))
        low_mid_salience = float(np.sum(salience[low_mid_mask]))
        total_body_salience = float(np.sum(salience))
        if total_body_salience > 0.0:
            out["low_mid_energy_ratio"] = low_mid_salience / total_body_salience

    body_orders = _expected_harmonic_orders(float(f0_hz), freq_min_hz=bmin, freq_max_hz=bmax)
    out["expected_harmonic_slots_up_to_5000hz"] = int(body_orders.size)
    harmonic_body_mask = harmonic_peak_mask & (freq_sig >= bmin) & (freq_sig <= bmax)
    harmonic_body_power = power_sig[harmonic_body_mask]
    if harmonic_body_power.size > 0:
        harmonic_salience = np.sqrt(np.maximum(harmonic_body_power, 0.0))
        harmonic_body_freq = freq_sig[harmonic_body_mask]
        knee = float(max(body_weight_knee_hz, 1e-6))
        w_harm_body = 1.0 / (1.0 + np.square(harmonic_body_freq / knee))
        out["harmonic_body_density"] = _effective_count(w_harm_body * harmonic_salience)
    if out["expected_harmonic_slots_up_to_5000hz"] > 0:
        out["harmonic_body_density_normalized"] = float(
            out["harmonic_body_density"] / out["expected_harmonic_slots_up_to_5000hz"]
        )

    # Register-dependent salient raw harmonic-count family (up to 5000 Hz by default).
    salient_ceiling_hz = float(max(salient_harmonic_ceiling_hz, 1e-6))
    expected_harmonic_order_count = int(math.floor(salient_ceiling_hz / float(f0_hz))) if _finite_positive(f0_hz) else 0
    expected_harmonic_order_count = max(expected_harmonic_order_count, 0)
    out["expected_harmonic_order_count_up_to_5000hz"] = expected_harmonic_order_count

    if expected_harmonic_order_count > 0:
        harmonic_orders = nearest_order[harmonic_peak_mask]
        harmonic_pow = power_sig[harmonic_peak_mask]
        in_salient_band = harmonic_orders * float(f0_hz) <= salient_ceiling_hz + EPS
        harmonic_orders = harmonic_orders[in_salient_band]
        harmonic_pow = harmonic_pow[in_salient_band]

        order_power_max: dict[int, float] = {}
        for n, p in zip(harmonic_orders.tolist(), harmonic_pow.tolist(), strict=False):
            ni = int(n)
            if ni < 1 or ni > expected_harmonic_order_count:
                continue
            pf = float(p)
            if not np.isfinite(pf) or pf <= 0.0:
                continue
            prev = order_power_max.get(ni)
            if prev is None or pf > prev:
                order_power_max[ni] = pf

        salient_threshold = pmax * (10.0 ** (float(salient_harmonic_relative_db) / 10.0))
        salient_orders = sorted(n for n, p in order_power_max.items() if p >= salient_threshold)
        salient_count = int(len(salient_orders))
        out["salient_harmonic_order_count_up_to_5000hz"] = salient_count
        out["salient_harmonic_coverage_up_to_5000hz"] = float(salient_count / expected_harmonic_order_count)

        salient_powers = np.array([order_power_max[n] for n in salient_orders], dtype=float)
        if salient_powers.size > 0:
            out["salient_harmonic_mass_up_to_5000hz"] = float(np.sum(np.sqrt(np.maximum(salient_powers, 0.0))))

        odd_orders = [n for n in salient_orders if (n % 2) == 1]
        even_orders = [n for n in salient_orders if (n % 2) == 0]
        out["salient_odd_harmonic_count_up_to_5000hz"] = int(len(odd_orders))
        out["salient_even_harmonic_count_up_to_5000hz"] = int(len(even_orders))
        odd_power = float(np.sum([order_power_max[n] for n in odd_orders])) if odd_orders else 0.0
        even_power = float(np.sum([order_power_max[n] for n in even_orders])) if even_orders else 0.0
        out["odd_even_harmonic_energy_ratio"] = float(odd_power / max(even_power, EPS))

    # Final user-facing density family (count-based and salience-weighted).
    d_ceiling_hz = float(max(density_frequency_ceiling_hz, 1e-6))
    d_thr_db = float(density_salience_threshold_db)
    mode = str(density_summation_mode or "his_weighted").strip().lower()
    w_h = float(harmonic_density_weight)
    w_i = float(inharmonic_density_weight)
    w_s = float(subbass_density_weight)
    if mode in ("harmonic_only", "harmonic-only", "h_only"):
        w_h, w_i, w_s = 1.0, 0.0, 0.0
    elif mode in ("inharmonic_only", "inharmonic-only", "i_only"):
        w_h, w_i, w_s = 0.0, 1.0, 0.0
    elif mode in ("subbass_only", "subbass-only", "s_only"):
        w_h, w_i, w_s = 0.0, 0.0, 1.0
    out["harmonic_density_weight"] = w_h
    out["inharmonic_density_weight"] = w_i
    out["subbass_density_weight"] = w_s
    out["density_summation_mode"] = mode
    out["density_salience_threshold_db"] = d_thr_db
    out["density_frequency_ceiling_hz"] = d_ceiling_hz

    def _salience_from_power(power_values: np.ndarray) -> np.ndarray:
        pv = np.asarray(power_values, dtype=float)
        if pv.size == 0 or not np.isfinite(pmax) or pmax <= 0.0:
            return np.array([], dtype=float)
        rel_db = 10.0 * np.log10(np.maximum(pv, EPS) / max(pmax, EPS))
        denom = max(0.0 - d_thr_db, EPS)
        return np.clip((rel_db - d_thr_db) / denom, 0.0, 1.0)

    # Harmonic component: one contribution per harmonic order (strongest peak per order).
    harmonic_orders = nearest_order[harmonic_peak_mask]
    harmonic_pow = power_sig[harmonic_peak_mask]
    in_density_band = harmonic_orders * float(f0_hz) <= d_ceiling_hz + EPS
    harmonic_orders = harmonic_orders[in_density_band]
    harmonic_pow = harmonic_pow[in_density_band]
    harmonic_order_power_max: dict[int, float] = {}
    for n, p in zip(harmonic_orders.tolist(), harmonic_pow.tolist(), strict=False):
        ni = int(n)
        if ni < 1:
            continue
        pf = float(p)
        if not np.isfinite(pf) or pf <= 0.0:
            continue
        prev = harmonic_order_power_max.get(ni)
        if prev is None or pf > prev:
            harmonic_order_power_max[ni] = pf
    harmonic_order_ids = sorted(harmonic_order_power_max.keys())
    harmonic_order_powers = np.array([harmonic_order_power_max[n] for n in harmonic_order_ids], dtype=float)
    harmonic_order_salience = _salience_from_power(harmonic_order_powers)
    salient_harmonic_orders = [n for n, s in zip(harmonic_order_ids, harmonic_order_salience, strict=False) if s > 0.0]
    h_count = float(len(salient_harmonic_orders))
    h_density = float(np.sum(harmonic_order_salience)) if harmonic_order_salience.size > 0 else 0.0

    # Inharmonic component: one contribution per occupied log-frequency bin.
    inharmonic_freq = freq_sig[residual_mask]
    inharmonic_pow = power_sig[residual_mask]
    inharmonic_in_band = inharmonic_freq <= d_ceiling_hz + EPS
    inharmonic_freq = inharmonic_freq[inharmonic_in_band]
    inharmonic_pow = inharmonic_pow[inharmonic_in_band]
    salient_inharmonic_bin_count = 0
    inharmonic_density = 0.0
    if inharmonic_freq.size > 0:
        i_bin_idx = np.floor(
            1200.0 * np.log2(np.maximum(inharmonic_freq, freq_min_hz) / max(freq_min_hz, 1e-6))
            / float(residual_log_bin_cents)
        ).astype(int)
        inharmonic_sal = _salience_from_power(inharmonic_pow)
        bin_salience_max: dict[int, float] = {}
        for b, s in zip(i_bin_idx.tolist(), inharmonic_sal.tolist(), strict=False):
            bi = int(b)
            sf = float(s)
            prev = bin_salience_max.get(bi)
            if prev is None or sf > prev:
                bin_salience_max[bi] = sf
        if bin_salience_max:
            _vals = np.array(list(bin_salience_max.values()), dtype=float)
            salient_inharmonic_bin_count = int(np.count_nonzero(_vals > 0.0))
            inharmonic_density = float(np.sum(_vals))

    # Subbass component: one contribution per salient subbass particle.
    subbass_freq = freq_sig[subbass_mask]
    subbass_pow = power_sig[subbass_mask]
    subbass_in_band = subbass_freq <= d_ceiling_hz + EPS
    subbass_pow = subbass_pow[subbass_in_band]
    subbass_sal = _salience_from_power(subbass_pow)
    salient_subbass_particle_count = int(np.count_nonzero(subbass_sal > 0.0))
    subbass_density = float(np.sum(subbass_sal)) if subbass_sal.size > 0 else 0.0

    out["salient_inharmonic_log_bin_count_up_to_5000hz"] = int(salient_inharmonic_bin_count)
    out["salient_subbass_particle_count"] = int(salient_subbass_particle_count)
    out["salient_harmonic_order_count_up_to_density_ceiling_hz"] = int(h_count)
    out["expected_harmonic_order_count_up_to_density_ceiling_hz"] = int(
        max(0, int(math.floor(d_ceiling_hz / float(f0_hz))) if _finite_positive(f0_hz) else 0)
    )
    if out["expected_harmonic_order_count_up_to_density_ceiling_hz"] > 0:
        out["salient_harmonic_coverage_up_to_density_ceiling_hz"] = float(
            h_count / out["expected_harmonic_order_count_up_to_density_ceiling_hz"]
        )
    out["salient_harmonic_mass_up_to_density_ceiling_hz"] = float(
        np.sum(np.sqrt(np.maximum(np.array([harmonic_order_power_max[n] for n in salient_harmonic_orders], dtype=float), 0.0)))
        if len(salient_harmonic_orders) > 0
        else 0.0
    )
    out["salient_inharmonic_log_bin_count_up_to_density_ceiling_hz"] = int(salient_inharmonic_bin_count)
    out["salient_subbass_particle_count_up_to_density_ceiling_hz"] = int(salient_subbass_particle_count)
    out["harmonic_density_component"] = float(h_density)
    out["inharmonic_density_component"] = float(inharmonic_density)
    out["subbass_density_component"] = float(subbass_density)
    out["final_note_density_count_based"] = float(
        w_h * h_count + w_i * float(salient_inharmonic_bin_count) + w_s * float(salient_subbass_particle_count)
    )
    out["final_note_density_salience_weighted"] = float(
        w_h * h_density + w_i * inharmonic_density + w_s * subbass_density
    )

    out["residual_body_contribution"] = float(
        out["residual_energy_ratio"] * out["residual_log_frequency_occupancy"]
    )
    out["residual_body_contribution_capped"] = float(
        min(out["residual_body_contribution"], float(residual_body_contribution_cap))
    )

    # Retain the old scalar only as a diagnostic alias. This is intentionally
    # not a publication-safe "spectral density" construct.
    D_H = h_eff
    D_R = float(len(np.unique(residual_freq))) if residual_freq.size else 0.0
    D_S = float(np.count_nonzero(subbass_mask))
    w_H = out["harmonic_energy_ratio"]
    w_R = out["residual_energy_ratio"]
    w_S = out["subbass_energy_ratio"]
    diagnostic = D_H * w_H + D_R * w_R + D_S * w_S
    out["density_metric_raw"] = float(diagnostic)
    out["energy_weighted_component_density_diagnostic"] = float(diagnostic)

    return out


def compute_descriptors_from_row_and_peaks(
    row: Mapping[str, Any],
    peaks_df: pd.DataFrame,
    *,
    freq_min_hz: float = 20.0,
    freq_max_hz: float = 20000.0,
) -> dict[str, Any]:
    """
    Convenience wrapper for workbook/pipeline rows.

    This makes f0 status explicit and prevents the common bug:
    f0 = min(harmonic_list_df["Frequency (Hz)"])
    """
    triplet = canonical_f0_triplet(
        f0_final_hz=row.get("f0_final_hz", row.get("f0_final")),
        f0_initial_hz=row.get("f0_initial_hz", row.get("f0_initial")),
        f0_prior_hz=row.get("f0_prior_hz", row.get("nominal_f0_hz")),
        f0_fit_accepted=row.get("f0_fit_accepted", False),
        f0_source=row.get("f0_source", ""),
    )
    return compute_acoustic_density_descriptors(
        peaks_df,
        f0_hz=triplet.f0_hz,
        f0_source=triplet.f0_source,
        acoustic_f0_status=triplet.acoustic_f0_status,
        f0_fit_accepted=triplet.f0_fit_accepted,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
    )
