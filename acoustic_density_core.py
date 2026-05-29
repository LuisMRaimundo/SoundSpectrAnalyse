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

Module changelog
----------------
2026-05-26
- Phase 1 semantic change: ``harmonic_density_weight``,
  ``inharmonic_density_weight`` and ``subbass_density_weight`` now expose the
  note-local pure observation (no prior mixing). Legacy smoothed values remain
  available as explicit deprecated fields.
- Phase 7 semantic change: register-invariant strength formula now normalizes
  harmonic-order / residual-log-bin / subbass-particle terms by their
  available slot capacities before computing pure observation weights.
- Phase 7.1 housekeeping: canonical runtime path now calls
  ``SubBassPolicy.upper_bound_hz`` directly (deprecated wrapper retained for
  external callers only), removing operational deprecation warnings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Mapping, Optional

import math
import warnings
import numpy as np
import pandas as pd
from subbass_policy import SubBassPolicy
from inharmonicity_model import fit_inharmonicity_coefficient
from constants import (
    ADAPTIVE_HARMONIC_TOLERANCE_POLICY_DOC,
    BODY_DENSITY_MAX_HZ,
    FULL_SPECTRUM_MAX_HZ,
    INHARMONICITY_B_ENABLE_THRESHOLD,
    INHARMONICITY_FIT_CENTS_WINDOW,
    INHARMONICITY_FIT_ORDER_CAP,
    STRENGTH_OCCUPANCY_WEIGHT_HARMONIC,
    STRENGTH_OCCUPANCY_WEIGHT_INHARMONIC,
    STRENGTH_OCCUPANCY_WEIGHT_SUBBASS,
)


EPS = 1e-12
_SUBBASS_RATIO_SHIM_WARNED = False


"""
deprecated, see pure_observation_w_{h,i,s}

Historical feed-forward blend retained only for backward-compatible reporting
of legacy smoothed values. Mixing current evidence with a running prior at the
observation stage can bias stochastic updates; online methods should ingest the
pure per-sample evidence and apply Bayesian / stochastic updates downstream
(Bottou, 2010).
"""
DEPRECATED_LEGACY_OBSERVATION_BLEND_WEIGHT = 0.55
DEPRECATED_LEGACY_PRIOR_BLEND_WEIGHT = 0.45

OBS_W_FORMULA_VERSION_CURRENT: Final[str] = "v58_full_spectrum_region_energy_gate"
"""
obs_w formula semantic tag used in exported descriptors.

Allowed values:
- ``v50_prior_mixed``: historical prior-contaminated blend
  (0.55 observation / 0.45 prior).
- ``v55_incommensurate_strength``: pure data ratio from the Phase-6/v55
  incommensurate strength formulation.
- ``v56_occupancy_ratio``: pure data ratio from the Phase-7 register-invariant
  occupancy-ratio strength formulation.
- ``v57_energy_anchored_occupancy``: occupancy strength gated by BODY-ceiling
  band energy share (superseded — body truncation made the non-harmonic energy
  share instrument-dependent).
- ``v58_full_spectrum_region_energy_gate``: occupancy strength gated by the
  FULL-SPECTRUM, total-power-normalised region energy triple
  (harmonic-peak / non-harmonic-residual / sub-bass). Energy-conserving and
  instrument-agnostic. The non-harmonic band is the inter-harmonic RESIDUAL
  (broadband noise + non-n*f0 content); partial inharmonicity (B, inharmonic
  peaks) is reported separately and is not part of this density gate.
"""


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


def _local_maxima_peak_centers(
    freq: np.ndarray,
    power: np.ndarray,
    rel_power_threshold: float,
) -> np.ndarray:
    """Return parabola-refined frequencies of local-maximum peaks above threshold.

    A "peak" is a bin whose power strictly exceeds both neighbours and is at or
    above ``rel_power_threshold``. The peak frequency is refined with a 3-point
    parabolic interpolation in power (standard QIFFT), giving a sub-bin centre
    that preserves any harmonic stretch instead of snapping to the nearest bin.
    """
    f = np.asarray(freq, dtype=float).ravel()
    p = np.asarray(power, dtype=float).ravel()
    n = f.size
    if n < 3 or p.size != n:
        return np.asarray([], dtype=float)
    # Local maxima (interior bins) at/above the threshold.
    interior = np.arange(1, n - 1)
    is_peak = (
        (p[interior] >= float(rel_power_threshold))
        & (p[interior] > p[interior - 1])
        & (p[interior] > p[interior + 1])
    )
    idx = interior[is_peak]
    if idx.size == 0:
        return np.asarray([], dtype=float)
    centers = []
    for k in idx:
        y0, y1, y2 = float(p[k - 1]), float(p[k]), float(p[k + 1])
        denom = (y0 - 2.0 * y1 + y2)
        delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
        if not np.isfinite(delta) or abs(delta) > 0.5:
            delta = 0.0
        # Local bin spacing around k (handles non-uniform axes defensively).
        df = float(f[k + 1] - f[k]) if (f[k + 1] - f[k]) > 0 else float(f[k] - f[k - 1])
        centers.append(float(f[k]) + delta * df)
    return np.asarray(centers, dtype=float)


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


def _expected_residual_bin_count(
    *,
    freq_min_hz: float,
    freq_max_hz: float,
    residual_log_bin_cents: float,
) -> int:
    lo = float(max(freq_min_hz, 1e-6))
    hi = float(max(freq_max_hz, lo))
    step = float(max(residual_log_bin_cents, 1e-6))
    if hi <= lo:
        return 0
    span_cents = float(1200.0 * math.log2(hi / lo))
    if not np.isfinite(span_cents) or span_cents <= 0.0:
        return 0
    return int(max(0, math.ceil(span_cents / step)))


def _append_qc_status(existing: str, token: str) -> str:
    base = str(existing or "").strip()
    if not base:
        return token
    parts = [p.strip() for p in base.split(",") if p.strip()]
    if token in parts:
        return ",".join(parts)
    parts.append(token)
    return ",".join(parts)


def deprecated_subbass_upper_bound_hz_from_ratio(
    *,
    f0_hz: float,
    sr_hz: float,
    n_fft: int,
    subbass_upper_ratio: float = 0.75,
) -> float:
    """deprecated, see SubBassPolicy.upper_bound_hz"""
    del subbass_upper_ratio
    global _SUBBASS_RATIO_SHIM_WARNED
    if not _SUBBASS_RATIO_SHIM_WARNED:
        warnings.warn(
            "deprecated, see SubBassPolicy.upper_bound_hz",
            DeprecationWarning,
            stacklevel=2,
        )
        _SUBBASS_RATIO_SHIM_WARNED = True
    return float(SubBassPolicy.upper_bound_hz(f0_hz=f0_hz, sr_hz=sr_hz, n_fft=n_fft))


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
    body_freq_max_hz: Optional[float] = None,
    body_peak_relative_db: Optional[float] = None,
    body_weight_knee_hz: float = 1800.0,
    low_mid_upper_hz: float = 2000.0,
    residual_body_contribution_cap: float = 0.25,
    salient_harmonic_relative_db: Optional[float] = None,
    salient_harmonic_ceiling_hz: Optional[float] = None,
    density_summation_mode: str = "his_note_adaptive",
    harmonic_density_weight: float = 1.0,
    inharmonic_density_weight: float = 0.5,
    subbass_density_weight: float = 0.25,
    density_salience_threshold_db: Optional[float] = None,
    density_frequency_ceiling_hz: Optional[float] = None,
    full_spectrum_max_hz: float = FULL_SPECTRUM_MAX_HZ,
    sr_hz: float = 44100.0,
    n_fft: int = 4096,
) -> dict[str, Any]:
    """
    Compute separated acoustic descriptors from a peak/component table.

    The returned descriptors are designed for export. No descriptor here should
    be silently averaged with legacy "Combined Density Metric" fields.

    deprecated, see pure_observation_w_{h,i,s}
    ``smoothed_w_h_legacy``, ``smoothed_w_i_legacy`` and
    ``smoothed_w_s_legacy`` preserve the historical prior-mixed semantics for
    backward compatibility only.
    """
    # Legacy compatibility argument kept for external callers; canonical runtime
    # path resolves sub-bass boundary via SubBassPolicy directly.
    del subbass_upper_ratio
    freq, power = _extract_peak_vectors(peaks_df)

    _freq_max_runtime = float(freq_max_hz) if np.isfinite(float(freq_max_hz)) else float(FULL_SPECTRUM_MAX_HZ)
    _body_ceiling_default_hz = float(max(1.0, min(_freq_max_runtime, float(BODY_DENSITY_MAX_HZ))))
    _body_freq_max_hz = (
        float(body_freq_max_hz)
        if body_freq_max_hz is not None and np.isfinite(float(body_freq_max_hz))
        else _body_ceiling_default_hz
    )
    _body_peak_relative_db = (
        float(body_peak_relative_db)
        if body_peak_relative_db is not None and np.isfinite(float(body_peak_relative_db))
        else float(min_relative_db)
    )
    _salient_harmonic_relative_db = (
        float(salient_harmonic_relative_db)
        if salient_harmonic_relative_db is not None and np.isfinite(float(salient_harmonic_relative_db))
        else float(min_relative_db)
    )
    _salient_harmonic_ceiling_hz = (
        float(salient_harmonic_ceiling_hz)
        if salient_harmonic_ceiling_hz is not None and np.isfinite(float(salient_harmonic_ceiling_hz))
        else _body_freq_max_hz
    )
    _density_salience_threshold_db = (
        float(density_salience_threshold_db)
        if density_salience_threshold_db is not None and np.isfinite(float(density_salience_threshold_db))
        else float(min_relative_db)
    )
    _density_frequency_ceiling_hz = (
        float(density_frequency_ceiling_hz)
        if density_frequency_ceiling_hz is not None and np.isfinite(float(density_frequency_ceiling_hz))
        else _body_ceiling_default_hz
    )

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
        "expected_harmonic_slots_up_to_body_ceiling": 0,
        "harmonic_body_density_normalized": 0.0,
        "residual_body_contribution": 0.0,
        "residual_body_contribution_capped": 0.0,
        "salient_harmonic_order_count_up_to_body_ceiling": 0,
        "expected_harmonic_order_count_up_to_body_ceiling": 0,
        "salient_harmonic_coverage_up_to_body_ceiling": 0.0,
        "theoretical_harmonic_order_count_up_to_body_ceiling": 0,
        "detected_salient_harmonic_order_count_up_to_body_ceiling": 0,
        "salient_harmonic_coverage_ratio_up_to_body_ceiling": 0.0,
        "salient_harmonic_mass_up_to_body_ceiling": 0.0,
        "salient_harmonic_order_count_up_to_density_ceiling_hz": 0,
        "expected_harmonic_order_count_up_to_density_ceiling_hz": 0,
        "salient_harmonic_coverage_up_to_density_ceiling_hz": 0.0,
        "salient_harmonic_mass_up_to_density_ceiling_hz": 0.0,
        "salient_odd_harmonic_count_up_to_body_ceiling": 0,
        "salient_even_harmonic_count_up_to_body_ceiling": 0,
        "odd_even_harmonic_energy_ratio": 0.0,
        "salient_inharmonic_log_bin_count_up_to_body_ceiling": 0,
        "salient_subbass_particle_count": 0,
        "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz": 0,
        "salient_subbass_particle_count_up_to_density_ceiling_hz": 0,
        "harmonic_body_energy_sum_body_ceiling": 0.0,
        "inharmonic_body_energy_sum_body_ceiling": 0.0,
        "subbass_rumble_energy_sum": 0.0,
        "density_body_weighted_sum_body_ceiling": 0.0,
        "harmonic_full_spectrum_energy_sum_20khz": 0.0,
        "inharmonic_full_spectrum_energy_sum_20khz": 0.0,
        "density_full_spectrum_weighted_sum_20khz": 0.0,
        "high_frequency_spectral_activity_sum": 0.0,
        "spectral_extension_index_20khz": 0.0,
        "brightness_or_upper_spectral_activity_index_20khz": 0.0,
        "full_spectrum_harmonic_candidate_count_20khz": 0,
        "final_note_density_count_based": 0.0,
        "final_note_density_salience_weighted": 0.0,
        "harmonic_density_component": 0.0,
        "inharmonic_density_component": 0.0,
        "subbass_density_component": 0.0,
        "harmonic_density_weight": float(harmonic_density_weight),
        "inharmonic_density_weight": float(inharmonic_density_weight),
        "subbass_density_weight": float(subbass_density_weight),
        "pure_observation_w_h": float("nan"),
        "pure_observation_w_i": float("nan"),
        "pure_observation_w_s": float("nan"),
        "smoothed_w_h_legacy": float("nan"),
        "smoothed_w_i_legacy": float("nan"),
        "smoothed_w_s_legacy": float("nan"),
        "legacy_component_strength_h_v55": float("nan"),
        "legacy_component_strength_i_v55": float("nan"),
        "legacy_component_strength_s_v55": float("nan"),
        "component_strength_h": float("nan"),
        "component_strength_i": float("nan"),
        "component_strength_s": float("nan"),
        "density_summation_mode": str(density_summation_mode or "his_note_adaptive"),
        "density_salience_threshold_db": float(_density_salience_threshold_db),
        "density_frequency_ceiling_hz": float(_density_frequency_ceiling_hz),
        "density_metric_raw": float("nan"),
        "effective_components_weighted_diagnostic": float("nan"),
        "diagnostic_effective_components_h": float("nan"),
        "diagnostic_effective_components_r": float("nan"),
        "diagnostic_effective_components_s": float("nan"),
        "energy_weighted_component_density_diagnostic": float("nan"),
        "inharmonicity_coefficient_B": float(0.0),
        "inharmonicity_fit_residual_std_cents": float("nan"),
        "inharmonicity_fit_status": "insufficient_partials",
        "inharmonicity_fit_method": "",
        "inharmonicity_stretch_applied": False,
        "adaptive_harmonic_tolerance_policy": ADAPTIVE_HARMONIC_TOLERANCE_POLICY_DOC,
        "arithmetic_validation_status": "passed",
        "acoustic_validation_status": (
            "passed" if bool(f0_fit_accepted) else "nominal_fallback_used_not_acoustically_verified"
        ),
        "qc_status": "",
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

    # Inharmonicity fit on PEAK CENTERS, not the raw significant-bin cloud.
    # ``freq``/``power`` arrive as the full spectrum (one row per FFT bin), so a
    # partial spans several adjacent bins. Feeding all of them to the fit lets
    # the order matcher pick the bin nearest the *nominal* n*f0 (a lobe flank),
    # which biases the joint (f0, B) fit toward B≈0 and destroys end-to-end
    # inharmonicity recovery. Reducing to local-maximum centers (with parabolic
    # sub-bin refinement) gives the fit the true — possibly stretched — partial
    # frequencies.
    inharm_freqs = _local_maxima_peak_centers(freq, power, rel_power_threshold)
    if inharm_freqs.size < 3:
        inharm_freqs = freq_sig
    fit_result = fit_inharmonicity_coefficient(
        candidate_freqs_hz=inharm_freqs,
        f0_hz=float(f0_hz),
        order_cap=int(INHARMONICITY_FIT_ORDER_CAP),
        cents_window=float(INHARMONICITY_FIT_CENTS_WINDOW),
    )
    fit_B = float(fit_result.get("inharmonicity_coefficient_B", 0.0) or 0.0)
    fit_status = str(fit_result.get("fit_status", "insufficient_partials") or "insufficient_partials")
    out["inharmonicity_coefficient_B"] = fit_B
    out["inharmonicity_fit_residual_std_cents"] = float(
        fit_result.get("fit_residual_std_cents", float("nan"))
    )
    out["inharmonicity_fit_status"] = fit_status
    out["inharmonicity_fit_method"] = str(fit_result.get("method", "") or "")
    out["inharmonicity_fit_result"] = fit_result
    # Explicit boolean export semantic:
    # False => fit computed but stretched prediction not used in assignment.
    out["inharmonicity_stretch_applied"] = False

    freq_sorted = np.sort(freq_sig.astype(float))
    freq_diffs = np.diff(freq_sorted) if freq_sorted.size >= 2 else np.asarray([], dtype=float)
    freq_diffs = freq_diffs[np.isfinite(freq_diffs) & (freq_diffs > EPS)]
    bin_spacing_hz = float(np.median(freq_diffs)) if freq_diffs.size > 0 else float(f0_hz / 8.0)
    out["bin_spacing_hz_estimate"] = float(bin_spacing_hz)

    # Classify each significant peak by nearest harmonic order in cents.
    nearest_order = np.rint(freq_sig / float(f0_hz)).astype(int)
    valid_order = nearest_order >= 1
    predicted = nearest_order.astype(float) * float(f0_hz)
    if fit_status == "ok" and fit_B > float(INHARMONICITY_B_ENABLE_THRESHOLD):
        n_float = np.maximum(nearest_order.astype(float), 1.0)
        predicted = n_float * float(f0_hz) * np.sqrt(1.0 + fit_B * (n_float**2))
        out["inharmonicity_stretch_applied"] = True
    cents_error = 1200.0 * np.log2(np.maximum(freq_sig, EPS) / np.maximum(predicted, EPS))
    n_safe = np.maximum(nearest_order.astype(float), 1.0)
    tol_floor_cents = 1200.0 * float(bin_spacing_hz) / np.maximum(n_safe * float(f0_hz), EPS)
    tol_per_partial = np.maximum(float(harmonic_tolerance_cents), tol_floor_cents)
    harmonic_peak_mask = valid_order & (np.abs(cents_error) <= tol_per_partial)

    try:
        sr_guess = float(max(freq_max_hz * 2.0, 1.0))
    except (TypeError, ValueError):
        sr_guess = 44100.0
    subbass_upper_hz = max(
        freq_min_hz,
        SubBassPolicy.upper_bound_hz(
            f0_hz=float(f0_hz),
            sr_hz=sr_guess,
            n_fft=int(n_fft),
        ),
    )
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

    # Body-focused thickness descriptors (salient peaks up to configured body ceiling).
    bmin = float(max(body_freq_min_hz, freq_min_hz, 1e-6))
    bmax = float(max(bmin, min(_body_freq_max_hz, freq_max_hz)))
    bmask = (freq_sig >= bmin) & (freq_sig <= bmax)
    body_freq = freq_sig[bmask]
    body_power = power_sig[bmask]
    if body_power.size > 0:
        bpmax = float(np.max(body_power))
        body_rel_thr = bpmax * (10.0 ** (float(_body_peak_relative_db) / 10.0))
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
    out["expected_harmonic_slots_up_to_body_ceiling"] = int(body_orders.size)
    harmonic_body_mask = harmonic_peak_mask & (freq_sig >= bmin) & (freq_sig <= bmax)
    harmonic_body_power = power_sig[harmonic_body_mask]
    if harmonic_body_power.size > 0:
        harmonic_salience = np.sqrt(np.maximum(harmonic_body_power, 0.0))
        harmonic_body_freq = freq_sig[harmonic_body_mask]
        knee = float(max(body_weight_knee_hz, 1e-6))
        w_harm_body = 1.0 / (1.0 + np.square(harmonic_body_freq / knee))
        out["harmonic_body_density"] = _effective_count(w_harm_body * harmonic_salience)
    if out["expected_harmonic_slots_up_to_body_ceiling"] > 0:
        out["harmonic_body_density_normalized"] = float(
            out["harmonic_body_density"] / out["expected_harmonic_slots_up_to_body_ceiling"]
        )

    # Register-dependent salient raw harmonic-count family (up to body ceiling).
    salient_ceiling_hz = float(max(_salient_harmonic_ceiling_hz, 1e-6))
    expected_harmonic_order_count = int(math.floor(salient_ceiling_hz / float(f0_hz))) if _finite_positive(f0_hz) else 0
    expected_harmonic_order_count = max(expected_harmonic_order_count, 0)
    out["expected_harmonic_order_count_up_to_body_ceiling"] = expected_harmonic_order_count
    out["theoretical_harmonic_order_count_up_to_body_ceiling"] = expected_harmonic_order_count

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

        salient_threshold = pmax * (10.0 ** (float(_salient_harmonic_relative_db) / 10.0))
        salient_orders = sorted(n for n, p in order_power_max.items() if p >= salient_threshold)
        salient_count = int(len(salient_orders))
        out["salient_harmonic_order_count_up_to_body_ceiling"] = salient_count
        out["detected_salient_harmonic_order_count_up_to_body_ceiling"] = salient_count
        out["salient_harmonic_coverage_up_to_body_ceiling"] = float(salient_count / expected_harmonic_order_count)
        out["salient_harmonic_coverage_ratio_up_to_body_ceiling"] = out["salient_harmonic_coverage_up_to_body_ceiling"]

        salient_powers = np.array([order_power_max[n] for n in salient_orders], dtype=float)
        if salient_powers.size > 0:
            out["salient_harmonic_mass_up_to_body_ceiling"] = float(np.sum(np.sqrt(np.maximum(salient_powers, 0.0))))

        odd_orders = [n for n in salient_orders if (n % 2) == 1]
        even_orders = [n for n in salient_orders if (n % 2) == 0]
        out["salient_odd_harmonic_count_up_to_body_ceiling"] = int(len(odd_orders))
        out["salient_even_harmonic_count_up_to_body_ceiling"] = int(len(even_orders))
        odd_power = float(np.sum([order_power_max[n] for n in odd_orders])) if odd_orders else 0.0
        even_power = float(np.sum([order_power_max[n] for n in even_orders])) if even_orders else 0.0
        out["odd_even_harmonic_energy_ratio"] = float(odd_power / max(even_power, EPS))

    # Final user-facing density family (count-based and salience-weighted).
    d_ceiling_hz = float(max(_density_frequency_ceiling_hz, 1e-6))
    d_thr_db = float(_density_salience_threshold_db)
    mode = str(density_summation_mode or "his_note_adaptive").strip().lower()
    manual_w_h = float(harmonic_density_weight)
    manual_w_i = float(inharmonic_density_weight)
    manual_w_s = float(subbass_density_weight)
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

    out["salient_inharmonic_log_bin_count_up_to_body_ceiling"] = int(salient_inharmonic_bin_count)
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

    body_ceiling_hz = float(max(BODY_DENSITY_MAX_HZ, 1.0))
    full_ceiling_hz = float(max(full_spectrum_max_hz, body_ceiling_hz))
    body_harmonic_band = harmonic_orders * float(f0_hz) <= body_ceiling_hz + EPS
    body_harmonic_energy = float(np.sum(harmonic_pow[body_harmonic_band])) if harmonic_pow.size > 0 else 0.0
    body_inharmonic_band = inharmonic_freq <= body_ceiling_hz + EPS
    body_inharmonic_energy = float(np.sum(inharmonic_pow[body_inharmonic_band])) if inharmonic_pow.size > 0 else 0.0
    body_subbass_energy = float(np.sum(subbass_pow)) if subbass_pow.size > 0 else 0.0
    out["harmonic_body_energy_sum_body_ceiling"] = body_harmonic_energy
    out["inharmonic_body_energy_sum_body_ceiling"] = body_inharmonic_energy
    out["subbass_rumble_energy_sum"] = body_subbass_energy

    harmonic_freq_all = nearest_order[harmonic_peak_mask] * float(f0_hz)
    harmonic_pow_all = power_sig[harmonic_peak_mask]
    full_harmonic_band = harmonic_freq_all <= full_ceiling_hz + EPS
    full_harmonic_energy = float(np.sum(harmonic_pow_all[full_harmonic_band])) if harmonic_pow_all.size > 0 else 0.0
    full_inharmonic_band = inharmonic_freq <= full_ceiling_hz + EPS
    full_inharmonic_energy = float(np.sum(inharmonic_pow[full_inharmonic_band])) if inharmonic_pow.size > 0 else 0.0
    out["harmonic_full_spectrum_energy_sum_20khz"] = full_harmonic_energy
    out["inharmonic_full_spectrum_energy_sum_20khz"] = full_inharmonic_energy

    above_body_harmonic_energy = max(0.0, full_harmonic_energy - body_harmonic_energy)
    above_body_inharmonic_energy = max(0.0, full_inharmonic_energy - body_inharmonic_energy)
    out["high_frequency_spectral_activity_sum"] = float(
        above_body_harmonic_energy + above_body_inharmonic_energy
    )
    full_orders = nearest_order[harmonic_peak_mask]
    if full_orders.size > 0:
        full_orders = full_orders[
            (full_orders.astype(float) * float(f0_hz) <= full_ceiling_hz + EPS) & (full_orders >= 1)
        ]
    out["full_spectrum_harmonic_candidate_count_20khz"] = int(np.unique(full_orders).size)

    w_h, w_i, w_s = manual_w_h, manual_w_i, manual_w_s
    pure_observation_triplet: tuple[float, float, float] | None = None
    smoothed_legacy_triplet: tuple[float, float, float] | None = None
    if mode in ("harmonic_only", "harmonic-only", "h_only"):
        w_h, w_i, w_s = 1.0, 0.0, 0.0
        out["density_weight_origin"] = "mode_forced_component"
    elif mode in ("inharmonic_only", "inharmonic-only", "i_only"):
        w_h, w_i, w_s = 0.0, 1.0, 0.0
        out["density_weight_origin"] = "mode_forced_component"
    elif mode in ("subbass_only", "subbass-only", "s_only"):
        w_h, w_i, w_s = 0.0, 0.0, 1.0
        out["density_weight_origin"] = "mode_forced_component"
    elif mode in ("his_note_adaptive", "his_adaptive", "note_adaptive", "adaptive", "auto_note"):
        # Phase 1 canonical behavior: expose pure note-local observation and
        # keep prior-smoothed output only as explicit legacy compatibility.
        legacy_component_strength = np.array(
            [
                max(float(h_density), 0.0) + 0.25 * max(float(h_count), 0.0),
                max(float(inharmonic_density), 0.0) + 0.25 * max(float(salient_inharmonic_bin_count), 0.0),
                max(float(subbass_density), 0.0) + 0.25 * max(float(salient_subbass_particle_count), 0.0),
            ],
            dtype=float,
        )
        out["legacy_component_strength_h_v55"] = float(legacy_component_strength[0])
        out["legacy_component_strength_i_v55"] = float(legacy_component_strength[1])
        out["legacy_component_strength_s_v55"] = float(legacy_component_strength[2])

        expected_h_slots = int(
            out.get("expected_harmonic_order_count_up_to_density_ceiling_hz", 0) or 0
        )
        subbass_upper_hz = float(
            SubBassPolicy.upper_bound_hz(
                f0_hz=float(f0_hz),
                sr_hz=float(sr_hz),
                n_fft=int(max(1, n_fft)),
            )
        )
        residual_bin_min_hz = max(float(freq_min_hz), subbass_upper_hz)
        residual_bin_max_hz = max(float(residual_bin_min_hz), float(d_ceiling_hz))
        expected_i_bins = int(
            _expected_residual_bin_count(
                freq_min_hz=residual_bin_min_hz,
                freq_max_hz=residual_bin_max_hz,
                residual_log_bin_cents=float(residual_log_bin_cents),
            )
        )
        bin_spacing_hz_estimate = float(out.get("bin_spacing_hz_estimate", float("nan")))
        if not np.isfinite(bin_spacing_hz_estimate) or bin_spacing_hz_estimate <= 0.0:
            bin_spacing_hz_estimate = 1.0
        raw_expected_s_particles = int(
            math.ceil(
                max(0.0, subbass_upper_hz - float(freq_min_hz))
                / max(1.0, float(bin_spacing_hz_estimate))
            )
        )
        expected_s_particles = int(max(1, raw_expected_s_particles))

        h_term = i_term = s_term = 0.0
        if expected_h_slots <= 0:
            out["qc_status"] = _append_qc_status(
                str(out.get("qc_status", "")),
                "register_normalization_denominator_zero_harmonic",
            )
        else:
            h_occupancy = float(
                np.clip(float(h_count) / float(expected_h_slots), 0.0, 1.0)
            )
            h_density_per_slot = float(
                np.clip(max(float(h_density), 0.0) / float(expected_h_slots), 0.0, 1.0)
            )
            h_term = float(
                h_density_per_slot
                + float(STRENGTH_OCCUPANCY_WEIGHT_HARMONIC) * h_occupancy
            )

        if expected_i_bins <= 0:
            out["qc_status"] = _append_qc_status(
                str(out.get("qc_status", "")),
                "register_normalization_denominator_zero_inharmonic",
            )
        else:
            i_occupancy = float(
                np.clip(
                    float(salient_inharmonic_bin_count) / float(expected_i_bins), 0.0, 1.0
                )
            )
            i_density_per_bin = float(
                np.clip(
                    max(float(inharmonic_density), 0.0) / float(expected_i_bins),
                    0.0,
                    1.0,
                )
            )
            i_term = float(
                i_density_per_bin
                + float(STRENGTH_OCCUPANCY_WEIGHT_INHARMONIC) * i_occupancy
            )

        if raw_expected_s_particles <= 0:
            out["qc_status"] = _append_qc_status(
                str(out.get("qc_status", "")),
                "register_normalization_denominator_zero_subbass",
            )
            s_term = 0.0
        else:
            s_occupancy = float(
                np.clip(
                    float(salient_subbass_particle_count) / float(expected_s_particles),
                    0.0,
                    1.0,
                )
            )
            s_density_per_particle = float(
                np.clip(
                    max(float(subbass_density), 0.0) / float(expected_s_particles),
                    0.0,
                    1.0,
                )
            )
            s_term = float(
                s_density_per_particle
                + float(STRENGTH_OCCUPANCY_WEIGHT_SUBBASS) * s_occupancy
            )

        # === ENERGY-CONSISTENCY GATE (obs formula v58: full-spectrum region) ==
        # The structural strength above (occupancy + density-per-slot) is
        # normalised by each band's EXPECTED slot count, and those counts differ
        # by orders of magnitude across bands (harmonic ~ hundreds of orders;
        # sub-bass ~ a handful of bins). At a permissive salience threshold a
        # spectrally narrow, energetically negligible band can therefore
        # saturate its occupancy on the noise floor and dominate the learned
        # density profile. Observed defect: cello C2 sub-bass carried 3.6e-5 of
        # the spectral energy yet drew w_s=0.52, and the wide non-harmonic band
        # out-weighted the (99.7%-energy) harmonic band.
        #
        # Fix: weight each band's structural strength by its MEASURED energy
        # presence (band energy share). A band with ~0 energy then contributes
        # ~0 to the adaptive observation, while structural richness still
        # modulates the weight AMONG bands that genuinely carry energy.
        #
        # v57 -> v58: the gate previously used BODY-ceiling-truncated band
        # energies. That made the non-harmonic band's energy share arbitrary and
        # INSTRUMENT-DEPENDENT: for bright/noisy tones (bowed strings, winds,
        # cymbals) most broadband/residual energy lies ABOVE the body ceiling, so
        # the truncated gate under-counted it and silently collapsed toward a
        # peak-only basis; for dark tones it did not. v58 uses the FULL-SPECTRUM,
        # total-power-normalised region triple — harmonic-peak / NON-HARMONIC
        # RESIDUAL / sub-bass — whose three energies partition every spectral bin
        # and therefore sum to the total measured power (energy-conserving) and
        # are instrument-agnostic.
        #
        # TERMINOLOGY: the middle band is the NON-HARMONIC / inter-harmonic
        # RESIDUAL (broadband bow/breath/attack noise plus any non-n*f0 tonal
        # content). It is NOT the same as partial INHARMONICITY (piano/bell
        # stretch), which is a separate physics descriptor — the inharmonicity
        # coefficient B and the inharmonic-PEAK energy — and must not be conflated
        # with this density gate. ``h_energy``/``r_energy``/``s_energy`` are the
        # full-spectrum region powers behind ``harmonic_energy_ratio`` /
        # ``residual_energy_ratio`` / ``subbass_energy_ratio``.
        _e_h = max(0.0, float(h_energy))
        _e_i = max(0.0, float(r_energy))
        _e_s = max(0.0, float(s_energy))
        _e_total = _e_h + _e_i + _e_s
        if _e_total > EPS:
            _gate_h, _gate_i, _gate_s = (
                _e_h / _e_total,
                _e_i / _e_total,
                _e_s / _e_total,
            )
        else:
            # No measurable energy in any band: fall back to ungated structure.
            _gate_h = _gate_i = _gate_s = 1.0
        out["component_strength_energy_gate_h"] = float(_gate_h)
        out["component_strength_energy_gate_i"] = float(_gate_i)
        out["component_strength_energy_gate_s"] = float(_gate_s)
        # Clearer, terminology-correct aliases (density basis = non-harmonic
        # residual region; back-compat names retained above).
        out["component_strength_energy_gate_harmonic"] = float(_gate_h)
        out["component_strength_energy_gate_non_harmonic_residual"] = float(_gate_i)
        out["component_strength_energy_gate_subbass"] = float(_gate_s)
        out["density_band_energy_basis"] = "full_spectrum_region_total_power_v58"
        h_term = float(h_term) * _gate_h
        i_term = float(i_term) * _gate_i
        s_term = float(s_term) * _gate_s

        component_strength = np.array([h_term, i_term, s_term], dtype=float)
        out["component_strength_h"] = float(component_strength[0])
        out["component_strength_i"] = float(component_strength[1])
        out["component_strength_s"] = float(component_strength[2])
        total_strength = float(np.sum(component_strength))
        if np.isfinite(total_strength) and total_strength > EPS:
            data_ratio = component_strength / total_strength
            # Feed-forward prior from previous analyses (GUI/orchestrator can pass
            # running weights learned from earlier notes). This makes later notes
            # depend on extracted earlier-note metrics instead of a fixed constant.
            prior_ratio = np.array([manual_w_h, manual_w_i, manual_w_s], dtype=float)
            prior_ratio = np.maximum(prior_ratio, 0.0)
            prior_sum = float(np.sum(prior_ratio))
            if prior_sum <= EPS:
                prior_ratio = np.array([0.45, 0.35, 0.20], dtype=float)
                prior_sum = float(np.sum(prior_ratio))
            prior_ratio = prior_ratio / max(prior_sum, EPS)
            legacy_smoothed_ratio = (
                DEPRECATED_LEGACY_OBSERVATION_BLEND_WEIGHT * data_ratio
                + DEPRECATED_LEGACY_PRIOR_BLEND_WEIGHT * prior_ratio
            )
            legacy_smoothed_ratio = legacy_smoothed_ratio / max(float(np.sum(legacy_smoothed_ratio)), EPS)

            pure_observation_triplet = (
                float(data_ratio[0]),
                float(data_ratio[1]),
                float(data_ratio[2]),
            )
            smoothed_legacy_triplet = (
                float(legacy_smoothed_ratio[0]),
                float(legacy_smoothed_ratio[1]),
                float(legacy_smoothed_ratio[2]),
            )
            # Canonical alias fields now expose pure observation.
            w_h, w_i, w_s = pure_observation_triplet
            out["density_weight_origin"] = "per_note_adaptive_pure_observation"
        else:
            fallback = np.array([1.0, 0.5, 0.25], dtype=float)
            fallback = fallback / max(float(np.sum(fallback)), EPS)
            pure_observation_triplet = (
                float(fallback[0]),
                float(fallback[1]),
                float(fallback[2]),
            )
            smoothed_legacy_triplet = pure_observation_triplet
            w_h, w_i, w_s = pure_observation_triplet
            out["density_weight_origin"] = "adaptive_fallback_default"
    else:
        out["density_weight_origin"] = "manual_or_mode_default"

    if pure_observation_triplet is None:
        pure_observation_triplet = (float(w_h), float(w_i), float(w_s))
    if smoothed_legacy_triplet is None:
        smoothed_legacy_triplet = (float(w_h), float(w_i), float(w_s))

    out["pure_observation_w_h"] = float(pure_observation_triplet[0])
    out["pure_observation_w_i"] = float(pure_observation_triplet[1])
    out["pure_observation_w_s"] = float(pure_observation_triplet[2])
    out["obs_w_formula_version"] = OBS_W_FORMULA_VERSION_CURRENT
    out["smoothed_w_h_legacy"] = float(smoothed_legacy_triplet[0])
    out["smoothed_w_i_legacy"] = float(smoothed_legacy_triplet[1])
    out["smoothed_w_s_legacy"] = float(smoothed_legacy_triplet[2])
    out["harmonic_density_weight"] = float(w_h)
    out["inharmonic_density_weight"] = float(w_i)
    out["subbass_density_weight"] = float(w_s)
    out["density_body_weighted_sum_body_ceiling"] = float(
        w_h * out["harmonic_body_energy_sum_body_ceiling"]
        + w_i * out["inharmonic_body_energy_sum_body_ceiling"]
        + w_s * out["subbass_rumble_energy_sum"]
    )
    out["density_full_spectrum_weighted_sum_20khz"] = float(
        w_h * out["harmonic_full_spectrum_energy_sum_20khz"]
        + w_i * out["inharmonic_full_spectrum_energy_sum_20khz"]
        + w_s * out["subbass_rumble_energy_sum"]
    )
    out["spectral_extension_index_20khz"] = float(
        out["density_full_spectrum_weighted_sum_20khz"]
        / max(out["density_body_weighted_sum_body_ceiling"], EPS)
    )
    out["brightness_or_upper_spectral_activity_index_20khz"] = float(
        out["high_frequency_spectral_activity_sum"]
        / max(
            out["harmonic_full_spectrum_energy_sum_20khz"] + out["inharmonic_full_spectrum_energy_sum_20khz"],
            EPS,
        )
    )
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

    # Unit-coherent diagnostic alias: all terms are effective-component counts
    # (inverse Herfindahl / participation-ratio family), not mixed counts.
    D_H = float(_effective_count(harmonic_power))
    D_R = float(_effective_count(residual_power))
    D_S = float(_effective_count(subbass_power))
    w_H = out["harmonic_energy_ratio"]
    w_R = out["residual_energy_ratio"]
    w_S = out["subbass_energy_ratio"]
    diagnostic = D_H * w_H + D_R * w_R + D_S * w_S
    out["effective_components_weighted_diagnostic"] = float(diagnostic)
    out["diagnostic_effective_components_h"] = float(D_H)
    out["diagnostic_effective_components_r"] = float(D_R)
    out["diagnostic_effective_components_s"] = float(D_S)
    # deprecated diagnostic alias, see effective_components_weighted_diagnostic
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
