# -*- coding: utf-8 -*-
from __future__ import annotations
from audio_utils import amp_to_db_mag, db_mag_to_amp, power_to_db, db_to_power

"""
Audio processing, spectral analysis (FFT), and metric extraction
(density, dissonance). Used by the PyQt GUI and batch execution.
"""

# ====================================================
# IMPORTS — standard, third-party, local
# ====================================================
import gc
import json
import logging
import os
import re
import shutil
import time
from functools import lru_cache
import multiprocessing
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
from collections import defaultdict
import threading
from density import spectral_density
from audio_utils import harmonic_tolerance_hz
import math
from constants import (
    ROBUST_SALIENT_INHARMONIC_PEAK_PICKING_ENABLED,
    DEFAULT_N_FFT, DEFAULT_HOP_LENGTH, DEFAULT_WINDOW,
    ENERGY_CONSERVATION_TOLERANCE,
    NORMALIZATION_TARGET_RMS_DB,
    MAX_SIGNAL_LENGTH, SIGNAL_TRUNCATION_FACTOR, LARGE_SIGNAL_THRESHOLD,
    FFT_DOWNGRADE_FACTOR, FFT_MIN_SIZE,
    ATTACK_TIME_THRESHOLD, SPECTRAL_ROLLOFF_PERCENTILE,
    MAIN_LOBE_THRESHOLD_DB, WINDOW_CHAR_FFT_PADDING,
    SMOOTHING_WINDOW_PERCENTAGE, SMOOTHING_MIN_WINDOW_LENGTH,
    SMOOTHING_POLYORDER, SMOOTHING_NOISE_FLOOR_PERCENTILE, SMOOTHING_NOISE_FLOOR_MULTIPLIER,
    DEFAULT_STFT_MAGNITUDE_SMOOTHING_ENABLED,
    SUBBASS_AGGREGATE_CUTOFF_HZ,
    MAX_ABS_DENSITY, DENSITY_METRIC_WEIGHT_D, DENSITY_METRIC_WEIGHT_S,
    DENSITY_METRIC_WEIGHT_E, DENSITY_METRIC_WEIGHT_C, TOTAL_METRIC_SCALE,
    EPSILON_AMPLITUDE,     SNR_THRESHOLD_DB,
    DISSONANCE_PAIRWISE_PARTIAL_CAP,
    DISSONANCE_CAP_COMPUTATION_NOTE,
    EFFECTIVE_DENSITY_COMPONENT_POLICY_DOC,
    INHARMONIC_MODE_FOR_EFFECTIVE_DENSITY,
    SUBBASS_POLICY_FOR_EFFECTIVE_DENSITY_DOC,
    COUNT_SEMANTICS_NOTE_DOC,
    LEGACY_PARTIAL_COUNT_ALIASES_NOTE,
)

from analysis_policy import (
    DENSITY_FORMULA_VERSION,
    EXPORT_SCHEMA_VERSION,
    F0_POLICY_VERSION,
    HARMONIC_FREQUENCY_POLICY_VERSION,
    LOW_FREQUENCY_POLICY_VERSION,
    MISSING_METRIC_POLICY_VERSION,
    NONHARMONIC_POLICY_VERSION,
)
from pipeline_contract import get_canonical_pipeline_contract

CANONICAL_PIPELINE_ROLE = "canonical_stage1_per_note_analysis"
PUBLICATION_OUTPUT_ALLOWED = True


# =====================================================================
# AUDIT FIX (stale-pipeline detection) — runtime schema version.
#
# Every per-note ``spectral_analysis.xlsx`` MUST persist this constant
# under the Analysis_Metadata ``analysis_schema_version`` key so that
# the compile / GUI / verify_runtime_schema tools can reject any
# workbook that was produced by a stale legacy pipeline.
#
# Bump this token whenever the per-note workbook layout changes in a
# way that downstream consumers must observe (new raw columns, new
# provenance fields, new component_energy ratios, etc.). Stale
# workbooks made before this bump will be rejected at compile / plot
# time with a clear "regenerate analysis" message.
# =====================================================================
ANALYSIS_SCHEMA_VERSION = "single_pass_raw_export_v2"


def _proc_audio_runtime_signature() -> str:
    """Short stable hash of the proc_audio.py source file.

    Used to detect cases where a GUI / orchestrator is importing a
    different (older) copy of proc_audio than the one currently
    sitting on disk. The hash is recomputed lazily from the on-disk
    file path so a Python session that picked up a stale .pyc still
    reports the *current* source signature, making divergence
    obvious in Analysis_Metadata.
    """
    import hashlib as _hl
    try:
        p = Path(__file__).resolve()
        data = p.read_bytes()
        return _hl.sha1(data).hexdigest()[:12]
    except Exception:
        return "unknown"


def _spectral_sheet_has_raw_columns(df: "pd.DataFrame") -> bool:
    """Return True when a per-note spectrum DataFrame has the
    audit-canonical ``Amplitude_raw`` + ``Power_raw`` pair.
    """
    if df is None or getattr(df, "empty", True):
        # Empty sheets are tolerated (no partials of that type); their
        # absence of columns is not a schema regression.
        return True
    cols = {str(c).strip() for c in df.columns}
    return ("Amplitude_raw" in cols) and ("Power_raw" in cols)


def log_runtime_paths(logger: "logging.Logger | None" = None) -> Dict[str, str]:
    """Log and return a dict of runtime paths + schema version.

    Used by the orchestrator / verify CLI to surface the *actually
    loaded* module file paths so a stale legacy install is obvious.
    """
    import sys as _sys
    import platform as _platform
    info: Dict[str, str] = {
        "sys_executable": str(_sys.executable),
        "cwd": str(Path.cwd()),
        "platform": _platform.platform(),
        "python_version": _sys.version.split()[0],
        "proc_audio_file": str(Path(__file__).resolve()),
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "proc_audio_runtime_signature": _proc_audio_runtime_signature(),
    }
    try:
        import compile_metrics as _cm

        info["compile_metrics_file"] = str(Path(_cm.__file__).resolve())
    except Exception:
        info["compile_metrics_file"] = "<not_importable>"
    try:
        import publication_chart_policy as _pcp

        info["publication_chart_policy_file"] = str(Path(_pcp.__file__).resolve())
    except Exception:
        info["publication_chart_policy_file"] = "<not_importable>"

    lg = logger or logging.getLogger(__name__)
    try:
        lg.info(
            "Runtime schema/path audit:\n"
            "  sys.executable                = %s\n"
            "  cwd                           = %s\n"
            "  proc_audio.__file__           = %s\n"
            "  compile_metrics.__file__      = %s\n"
            "  publication_chart_policy.__file__ = %s\n"
            "  ANALYSIS_SCHEMA_VERSION       = %s\n"
            "  proc_audio_runtime_signature  = %s",
            info["sys_executable"], info["cwd"],
            info["proc_audio_file"],
            info["compile_metrics_file"],
            info["publication_chart_policy_file"],
            info["analysis_schema_version"],
            info["proc_audio_runtime_signature"],
        )
    except Exception:
        pass
    return info


def linear_export_batch_alignment_k(
    s_h: float,
    s_ih: float,
    s_sb: float,
    p_h: Any,
    p_i: Any,
    p_s: Any,
) -> float:
    """
    Factor k in (0, 1] applied equally to inharmonic-partial and sub-bass export amplitudes so that
    k * (s_ih + s_sb) <= min(((p_i + p_s) / p_h) * s_h, s_h), using batch GUI energy ratios as the
    linear-mass budget (export bookkeeping; STFT energy sums are unchanged).

    Missing ``p_s`` / ``p_i`` (e.g. legacy H+I batch handoff with no sub-bass column) are treated as
    0.0 so a valid H-only denominator is still used; if ``p_h`` is missing, returns 1.0 (no scaling).
    """
    if p_h is None:
        return 1.0
    try:
        ph = float(p_h)
    except (TypeError, ValueError):
        return 1.0
    try:
        pi = 0.0 if p_i is None else float(p_i)
    except (TypeError, ValueError):
        return 1.0
    try:
        ps = 0.0 if p_s is None else float(p_s)
    except (TypeError, ValueError):
        return 1.0
    try:
        sh = max(0.0, float(s_h))
        si = max(0.0, float(s_ih))
        ss = max(0.0, float(s_sb))
    except (TypeError, ValueError):
        return 1.0
    if not (math.isfinite(sh) and math.isfinite(si) and math.isfinite(ss)):
        return 1.0
    t_raw = si + ss
    if t_raw <= 1e-30:
        return 1.0
    if not (math.isfinite(ph) and math.isfinite(pi) and math.isfinite(ps)):
        return 1.0
    if ph <= 1e-30:
        return 1.0
    r_in_sb = (pi + ps) / ph
    if not math.isfinite(r_in_sb) or r_in_sb < 0.0:
        return 1.0
    t_target = min(sh * r_in_sb, sh)
    if not math.isfinite(t_target):
        return 1.0
    if t_target <= 1e-30:
        return 0.0
    return float(min(1.0, t_target / t_raw))


# PHASE 4: Data Integrity imports
try:
    from data_integrity import (
        robust_normalize, normalize_log_transform,
        validate_metric_value, validate_metric_array, validate_audio_parameters,
        GlobalReferenceScaler, calculate_iqr_bounds
    )
except ImportError:
    # Fallback if data_integrity not available
    # Logger will be defined later, so we'll create fallback functions without logging
    pass
    def robust_normalize(data, method="iqr", clip_range=(0.0, 1.0), **kwargs):
        # Fallback to simple min-max
        data_clean = data[np.isfinite(data)]
        if data_clean.size == 0:
            return np.zeros_like(data)
        data_min, data_max = np.min(data_clean), np.max(data_clean)
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)
        if clip_range:
            normalized = np.clip(normalized, clip_range[0], clip_range[1])
        return normalized
    
    def normalize_log_transform(data, clip_range=(0.0, 1.0), epsilon=1e-10):
        # Fallback implementation
        data_clean = data[np.isfinite(data)]
        if data_clean.size == 0:
            return np.zeros_like(data)
        data_positive = np.maximum(data_clean, epsilon)
        log_data = np.log1p(data_positive)
        log_min, log_max = np.min(log_data), np.max(log_data)
        if log_max > log_min:
            normalized = (log_data - log_min) / (log_max - log_min)
        else:
            normalized = np.zeros_like(data_clean)
        if clip_range:
            normalized = np.clip(normalized, clip_range[0], clip_range[1])
        result = np.full_like(data, np.nan)
        result[np.isfinite(data)] = normalized
        return result
    
    def validate_metric_value(value, metric_name, expected_range=None, **kwargs):
        return (True, None)
    
    def validate_metric_array(values, metric_name, expected_range=None, **kwargs):
        return (True, None, {})
    
    def validate_audio_parameters(n_fft, hop_length, sr, signal_length):
        return (True, None)
    
    class GlobalReferenceScaler:
        def fit(self, *args, **kwargs): pass
        def transform(self, data, **kwargs): return data
        def fit_transform(self, data, **kwargs): return data
    
    def calculate_iqr_bounds(data, iqr_multiplier=1.5):
        return (0.0, 0.0, 0.0, 0.0)

import librosa
import librosa.display
import numpy as np

import pandas as pd

from density import (
    apply_density_metric,
    partial_metric_sums_h_i_s_total,
    compute_discrete_spectral_metrics_bundle,
    get_weight_function,
    compute_spectral_entropy,
    calculate_combined_density_metric,
    identify_nonharmonic_residual_rows,
    partial_density_effective_components_bundle,
    aggregate_low_frequency_residual_peak_power,
    aggregate_subbass_noise_peak_power,
    compute_harmonic_occupancy_ratio,
    compute_residual_log_frequency_occupancy,
    compute_expected_harmonic_slot_count,
    # AUDIT FIX (acoustic-physics, Clarinete_mf findings #1 + #2) — the
    # sub-bass aggregator now respects a lower-frequency floor and a
    # window-aware harmonic-protection tolerance to suppress DC bins,
    # sub-audible noise, and main-lobe leakage from strong harmonics.
    SUBBASS_AGGREGATE_LOWER_HZ,
    compute_subbass_protection_tolerance_hz,
    DISCRETE_SPECTRAL_METRIC_KEYS,
    CANONICAL_DENSITY_FORMULA_VERSION,
    CANONICAL_DENSITY_SOURCE_FORMULA,
    # usadas em alguns cálculos
    # (se não existirem todas no seu módulo, ajuste aqui conforme a sua API real)
    # calculate_harmonic_density,
    # calculate_inharmonic_density,
)

from dissonance_models import (
    get_dissonance_model,
    list_available_models,
)

from peak_component_counts import classify_peaks_harmonic_inharmonic_subbass_from_df
from energy_accounting import describe_component_energy_balance
from data_integrity import metric_float_or_nan, metric_int_or_nan
from acoustic_density_core import (
    canonical_f0_triplet,
    compute_acoustic_density_descriptors,
)

# logging base
logger = logging.getLogger(__name__)
if not logger.handlers:
    try:
        from log_config import configure_root_logger
        configure_root_logger()
        logger = logging.getLogger(__name__)
    except Exception:
        logging.basicConfig(level=logging.INFO)

# ====================================================
# CONFIGURAÇÃO GLOBAL
# ====================================================
DEFAULT_N_FFT: int = 4096
DEFAULT_HOP_LENGTH: int = 1024
DEFAULT_WINDOW: str = "hann"
DEFAULT_PLOT_DPI: int = 300

# --- Component balance pie charts (visualisation only; linear ΣA vs energy ratios) ---
COMPONENT_AMPLITUDE_MASS_PIE_FILENAME = "component_amplitude_mass_pie.png"
COMPONENT_ENERGY_RATIO_PIE_FILENAME = "component_energy_ratio_pie.png"
COMPONENT_ENERGY_PIE_LEGACY_ALIAS_FILENAME = "component_energy_pie.png"
COMPONENT_AMPLITUDE_MASS_PIE_TITLE_PREFIX = "Candidate amplitude balance"
COMPONENT_AMPLITUDE_MASS_PIE_BASIS_FOOTNOTE = (
    "Basis: linear amplitude sums; not power/energy ratios."
)
COMPONENT_ENERGY_RATIO_PIE_BASIS_FOOTNOTE = "Basis: power/energy ratios."
_COMPONENT_AMPLITUDE_MASS_PIE_LEGEND_LABELS: Tuple[str, str, str] = (
    "Harmonic candidate amplitude mass",
    "Nonharmonic candidate amplitude mass",
    "Low-frequency residual amplitude mass",
)
_COMPONENT_ENERGY_RATIO_PIE_LEGEND_LABELS: Tuple[str, str, str] = (
    "Harmonic energy",
    "Nonharmonic candidate energy",
    "Low-frequency residual energy",
)


def _energy_ratio_pie_values(
    harmonic_energy_ratio: Optional[float],
    inharmonic_energy_ratio: Optional[float],
    subbass_energy_ratio: Optional[float],
) -> Optional[Tuple[float, float, float]]:
    """Raw (H, I, S) wedges for the component **power/energy** pie.

    Requires both ``harmonic_energy_ratio`` and ``inharmonic_energy_ratio`` to
    be present. ``subbass_energy_ratio`` may be ``None`` and is treated as
    ``0.0``. Values are **not** re-scaled here so they match stored metrics when
    those already sum to 1.
    """
    if harmonic_energy_ratio is None or inharmonic_energy_ratio is None:
        return None

    def _fr01(x: float) -> float:
        if not math.isfinite(x) or x < 0.0:
            return 0.0
        return float(x)

    try:
        hf = _fr01(float(harmonic_energy_ratio))
        inf = _fr01(float(inharmonic_energy_ratio))
    except (TypeError, ValueError):
        return None

    if subbass_energy_ratio is None:
        sf = 0.0
    else:
        try:
            sf = _fr01(float(subbass_energy_ratio))
        except (TypeError, ValueError):
            return None

    tot = hf + inf + sf
    if tot <= 1e-18:
        return None
    return hf, inf, sf


def _coherent_gain(win: str, n_fft: int) -> float:
    """Ganho coerente da janela: G = (1/N)*sum w[n]."""
    try:
        import numpy as np
        try:
            from scipy.signal import windows as _win
            wname = (win or "").lower()
            if wname in ("flattop","flat-top","flat_top"):
                w = _win.flattop(n_fft, sym=False)
            elif wname in ("blackmanharris","blackmanharris","bh92","bh-92"):
                w = _win.blackmanharris(n_fft, sym=False)
            elif wname in ("hann","hanning"):
                w = _win.hann(n_fft, sym=False)
            elif wname in ("hamming",):
                w = _win.hamming(n_fft, sym=False)
            else:
                w = _win.hann(n_fft, sym=False)
        except Exception:
            if (win or "").lower() in ("hann","hanning"):
                w = np.hanning(n_fft)
            elif (win or "").lower() in ("hamming",):
                w = np.hamming(n_fft)
            else:
                w = np.hanning(n_fft)
        return float(np.sum(w) / float(n_fft))
    except Exception:
        return 1.0


def _window_sum(win: str, n_fft: int) -> float:
    """
    FIX 4 — Σ w[n] of the analysis window (NOT divided by N).

    For a real sinusoid of amplitude ``A`` centred on bin ``k``, the
    one-sided STFT magnitude obeys ``A ≈ 2 * |X[k]| / Σ w[n]``. The legacy
    ``_coherent_gain`` returns ``G = (Σ w) / N``, so dividing magnitudes by
    ``G`` leaves a residual dependence on ``N``. This helper returns
    ``Σ w[n]`` directly, used by ``physical_peak_amplitude`` below.
    """
    try:
        import numpy as np
        try:
            from scipy.signal import windows as _win
            wname = (win or "").lower()
            if wname in ("flattop", "flat-top", "flat_top"):
                w = _win.flattop(n_fft, sym=False)
            elif wname in ("blackmanharris", "bh92", "bh-92"):
                w = _win.blackmanharris(n_fft, sym=False)
            elif wname in ("hann", "hanning"):
                w = _win.hann(n_fft, sym=False)
            elif wname in ("hamming",):
                w = _win.hamming(n_fft, sym=False)
            elif wname in ("bartlett",):
                w = _win.bartlett(n_fft, sym=False)
            elif wname in ("kaiser",):
                w = _win.kaiser(n_fft, beta=6.5, sym=False)
            elif wname in ("gaussian", "gauss", "gaussiana"):
                w = _win.gaussian(n_fft, std=n_fft / 8.0, sym=False)
            else:
                w = _win.hann(n_fft, sym=False)
        except Exception:
            if (win or "").lower() in ("hann", "hanning"):
                w = np.hanning(n_fft)
            elif (win or "").lower() in ("hamming",):
                w = np.hamming(n_fft)
            else:
                w = np.hanning(n_fft)
        return float(np.sum(w))
    except Exception:
        return float(n_fft)


def physical_peak_amplitude(
    stft_magnitude: "np.ndarray",
    window_name: str,
    n_fft: int,
    *,
    is_one_sided: bool = True,
) -> "np.ndarray":
    """
    FIX 4 — Convert raw STFT magnitudes |X[k]| to physical peak amplitudes.

    For a real sinusoid concentrated on a bin:

        A_peak ≈ ``one_sided_factor`` * |X[k]| / Σ w[n]

    where ``one_sided_factor`` is ``2.0`` for the one-sided spectrum
    (excluding DC and Nyquist bins) and ``1.0`` otherwise. The result is
    **independent of n_fft** (provided ``is_one_sided`` is set correctly).

    Notes
    -----
    The existing pipeline applies a different correction
    (``amp / coherent_gain``, where ``coherent_gain = Σw / N``) which leaves
    an unintended factor of ``N/2``. This function is the physically correct
    version and should be preferred for any new amplitude-calibrated
    metric. Existing values are left untouched so legacy comparisons remain
    reproducible.
    """
    import numpy as _np
    sw = _window_sum(window_name, int(n_fft))
    if not (_np.isfinite(sw) and sw > 0.0):
        return _np.asarray(stft_magnitude, dtype=float)
    factor = 2.0 if is_one_sided else 1.0
    return factor * _np.asarray(stft_magnitude, dtype=float) / sw


def audit_amplitude_calibration(
    test_amplitude: float = 1.0,
    freq_hz: float = 1000.0,
    sr: int = 44100,
    n_fft_values: "Iterable[int] | None" = None,
    window_name: str = "hann",
) -> "dict":
    """
    FIX 4 — Self-test for STFT amplitude calibration.

    Generates a pure sinusoid at ``freq_hz`` and verifies that
    ``physical_peak_amplitude`` recovers ``test_amplitude`` to within ~0.5 dB
    across several ``n_fft`` values, demonstrating that the calibration is
    n_fft-independent. Returns a dict ``{n_fft: recovered_amplitude}``.
    """
    import numpy as _np
    if n_fft_values is None:
        n_fft_values = (1024, 2048, 4096, 8192, 16384)

    out: "dict[int, float]" = {}
    duration_s = 1.0
    t = _np.arange(int(sr * duration_s)) / float(sr)
    x = test_amplitude * _np.sin(2.0 * _np.pi * freq_hz * t)

    for n_fft in n_fft_values:
        n_fft = int(n_fft)
        hop = n_fft // 4
        try:
            import librosa  # type: ignore
            S = librosa.stft(x, n_fft=n_fft, hop_length=hop, window=window_name, center=True)
            mag = _np.abs(S).max(axis=1)
        except Exception:
            from scipy.signal import stft as _stft  # type: ignore
            _, _, S = _stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop, window=window_name)
            mag = _np.abs(S).max(axis=1)
        a_phys = physical_peak_amplitude(mag, window_name, n_fft, is_one_sided=True)
        out[n_fft] = float(_np.max(a_phys))
    return out


def _calculate_window_characteristics(win: str, n_fft: int) -> Dict[str, float]:
    """
    Calculate window characteristics: main-lobe width and side-lobe level.

    Spectral-leakage quantification helper.
    
    Args:
        win: Window type name
        n_fft: FFT size
        
    Returns:
        Dictionary with 'main_lobe_width' (in bins) and 'side_lobe_level' (in dB)
    """
    try:
        import numpy as np
        from scipy.signal import windows as _win
        from scipy.fft import fft, fftshift
        
        wname = (win or "").lower()
        
        # Get window function
        if wname in ("flattop", "flat-top", "flat_top"):
            w = _win.flattop(n_fft, sym=False)
        elif wname in ("blackmanharris", "blackmanharris", "bh92", "bh-92"):
            w = _win.blackmanharris(n_fft, sym=False)
        elif wname in ("hann", "hanning"):
            w = _win.hann(n_fft, sym=False)
        elif wname in ("hamming",):
            w = _win.hamming(n_fft, sym=False)
        elif wname in ("bartlett",):
            w = _win.bartlett(n_fft, sym=False)
        elif wname in ("kaiser",):
            # Default beta for Kaiser
            beta = 6.5
            w = _win.kaiser(n_fft, beta=beta, sym=False)
        elif wname in ("gaussian", "gauss", "gaussiana"):
            # Default std for Gaussian
            std = n_fft / 8.0
            w = _win.gaussian(n_fft, std=std, sym=False)
        else:
            w = _win.hann(n_fft, sym=False)
        
        # Compute FFT of window (zero-padded for better resolution)
        n_pad = n_fft * WINDOW_CHAR_FFT_PADDING  # WINDOW_CHAR_FFT_PADDING = 8
        w_padded = np.zeros(n_pad)
        w_padded[:n_fft] = w
        W = fftshift(fft(w_padded))
        W_mag_db = 20 * np.log10(np.abs(W) + 1e-12)
        
        # Find main lobe (peak at center)
        center = n_pad // 2
        peak_idx = center
        
        # Main lobe width: find -3dB points
        peak_db = W_mag_db[peak_idx]
        threshold_db = peak_db + MAIN_LOBE_THRESHOLD_DB  # MAIN_LOBE_THRESHOLD_DB = -3.0
        
        # Find left and right -3dB points
        left_idx = peak_idx
        while left_idx > 0 and W_mag_db[left_idx] > threshold_db:
            left_idx -= 1
        
        right_idx = peak_idx
        while right_idx < n_pad - 1 and W_mag_db[right_idx] > threshold_db:
            right_idx += 1
        
        # Convert to bins (normalize by n_fft)
        main_lobe_width_bins = (right_idx - left_idx) * (n_fft / n_pad)
        
        # Side-lobe level: maximum level outside main lobe
        # Exclude main lobe region (±2 bins around peak)
        exclude_region = 4 * (n_fft / n_pad)  # ±2 bins in original scale
        exclude_start = int(center - exclude_region * (n_pad / n_fft))
        exclude_end = int(center + exclude_region * (n_pad / n_fft))
        
        side_lobes = np.concatenate([
            W_mag_db[:exclude_start],
            W_mag_db[exclude_end:]
        ])
        
        if len(side_lobes) > 0:
            side_lobe_level_db = float(np.max(side_lobes))
        else:
            side_lobe_level_db = -np.inf
        
        return {
            'main_lobe_width': float(main_lobe_width_bins),
            'side_lobe_level': float(side_lobe_level_db),
            'peak_level': float(peak_db)
        }
        
    except Exception as e:
        logger.warning(f"Failed to calculate window characteristics: {e}")
        return {
            'main_lobe_width': float('nan'),
            'side_lobe_level': float('nan'),
            'peak_level': float('nan')
        }


def _calculate_temporal_evolution(
    S_mag: np.ndarray,
    times: np.ndarray,
    freqs: np.ndarray,
    sr: int
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Calculate temporal evolution metrics: spectral flux, attack time, etc.
    
    Phase 2 Implementation: Temporal Evolution Analysis
    
    Args:
        S_mag: Magnitude spectrogram (freq x time)
        times: Time array for each frame
        freqs: Frequency array
        sr: Sample rate
        
    Returns:
        Dictionary with temporal metrics
    """
    try:
        import numpy as np
        
        n_freq, n_time = S_mag.shape
        
        if n_time < 2:
            return {
                'spectral_flux': 0.0,
                'attack_time': 0.0,
                'spectral_centroid_evolution': None,
                'spectral_rolloff_evolution': None
            }
        
        # 1. Spectral Flux: rate of change of spectral magnitude
        # Formula: flux[t] = Σ |X[t, k] - X[t-1, k]| for positive differences only
        spectral_flux = np.zeros(n_time - 1)
        
        for t in range(1, n_time):
            diff = S_mag[:, t] - S_mag[:, t-1]
            # Only count increases (positive differences)
            positive_diff = np.maximum(diff, 0.0)
            spectral_flux[t-1] = np.sum(positive_diff)
        
        # Average spectral flux
        avg_spectral_flux = float(np.mean(spectral_flux))
        
        # 2. Attack Time: time to reach 90% of maximum energy
        # Calculate total energy per frame
        frame_energies = np.sum(S_mag ** 2, axis=0)
        max_energy = np.max(frame_energies)
        energy_threshold = ATTACK_TIME_THRESHOLD * max_energy  # ATTACK_TIME_THRESHOLD = 0.9
        
        # Find first frame above threshold
        attack_frame = None
        for t in range(n_time):
            if frame_energies[t] >= energy_threshold:
                attack_frame = t
                break
        
        if attack_frame is not None and len(times) > attack_frame:
            attack_time = float(times[attack_frame])
        else:
            attack_time = float(times[-1]) if len(times) > 0 else 0.0
        
        # 3. Spectral Centroid Evolution: how the "brightness" changes over time
        spectral_centroid_evolution = np.zeros(n_time)
        for t in range(n_time):
            frame_mag = S_mag[:, t]
            total_mag = np.sum(frame_mag)
            if total_mag > 0:
                # Weighted average frequency
                centroid = np.sum(freqs * frame_mag) / total_mag
                spectral_centroid_evolution[t] = centroid
            else:
                spectral_centroid_evolution[t] = 0.0
        
        # 4. Spectral Rolloff Evolution: frequency below which 85% of energy is contained
        rolloff_percentile = SPECTRAL_ROLLOFF_PERCENTILE  # SPECTRAL_ROLLOFF_PERCENTILE = 0.85
        spectral_rolloff_evolution = np.zeros(n_time)
        
        for t in range(n_time):
            frame_mag = S_mag[:, t]
            frame_power = frame_mag ** 2
            total_power = np.sum(frame_power)
            
            if total_power > 0:
                # Cumulative power
                cumsum_power = np.cumsum(frame_power)
                threshold_power = rolloff_percentile * total_power
                
                # Find frequency where cumulative power exceeds threshold
                rolloff_idx = np.searchsorted(cumsum_power, threshold_power)
                rolloff_idx = min(rolloff_idx, len(freqs) - 1)
                spectral_rolloff_evolution[t] = freqs[rolloff_idx]
            else:
                spectral_rolloff_evolution[t] = 0.0
        
        return {
            'spectral_flux': avg_spectral_flux,
            'attack_time': attack_time,
            'spectral_centroid_evolution': spectral_centroid_evolution,
            'spectral_rolloff_evolution': spectral_rolloff_evolution
        }
        
    except Exception as e:
        logger.warning(f"Failed to calculate temporal evolution: {e}")
        return {
            'spectral_flux': float('nan'),
            'attack_time': float('nan'),
            'spectral_centroid_evolution': None,
            'spectral_rolloff_evolution': None
        }


def _verify_energy_conservation(
    y_time: np.ndarray,
    S_freq: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: Union[str, np.ndarray],
    tolerance: float = 0.1,
    window_array: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Verify energy conservation using Parseval's theorem.

    Energy-conservation verification helper.
    
    Parseval's theorem: Σ|x[n]|² = (1/N) * Σ|X[k]|²
    
    For STFT with overlap, we need to account for windowing and overlap.
    
    CORRECTED: Proper normalization for librosa.stft output.
    
    librosa.stft returns STFT with specific normalization:
    - No 1/n_fft normalization by default
    - Energy needs to account for window power and overlap
    
    Theoretical formula for windowed STFT with overlap:
    Energy_time = Energy_freq / (window_power * overlap_factor)
    
    Where:
    - window_power = Σw[n]² (energy of window function)
    - overlap_factor = window_length / hop_length (samples counted multiple times)
    
    Args:
        y_time: Time-domain signal
        S_freq: STFT matrix (complex) from librosa.stft
        n_fft: FFT size
        hop_length: Hop length
        window: Window type name
        tolerance: Acceptable deviation (default 10%)
        window_array: Actual window array used (for accurate calculation)
        
    Returns:
        Dictionary with energy ratio and verification status
    """
    try:
        import numpy as np
        
        # Time-domain energy (per sample, before windowing)
        energy_time = np.sum(np.abs(y_time) ** 2)
        
        # Frequency-domain energy (from STFT)
        # For STFT: energy = sum over all time frames and frequency bins
        energy_freq = np.sum(np.abs(S_freq) ** 2)
        
        # FIXED: Calculate window function's power (energy reduction factor)
        # Window power = sum(window²) - this is the energy scaling factor
        if window_array is not None:
            # Use provided window array (most accurate)
            w = np.asarray(window_array, dtype=float)
            if len(w) != n_fft:
                # If lengths don't match, use the actual window length
                # This handles zero-padding cases
                w_full = np.zeros(n_fft)
                w_full[:len(w)] = w
                w = w_full
        elif isinstance(window, np.ndarray):
            # Window is already an array
            w = np.asarray(window, dtype=float)
            if len(w) != n_fft:
                w_full = np.zeros(n_fft)
                w_full[:len(w)] = w
                w = w_full
        else:
            # Generate window from name (fallback - may not match exact parameters)
            try:
                from scipy.signal import windows as _win
                wname = (window or "").lower()
                
                # Get window function (matching the logic in _calculate_window_characteristics)
                if wname in ("flattop", "flat-top", "flat_top"):
                    w = _win.flattop(n_fft, sym=False)
                elif wname in ("blackmanharris", "blackmanharris", "bh92", "bh-92"):
                    w = _win.blackmanharris(n_fft, sym=False)
                elif wname in ("hann", "hanning"):
                    w = _win.hann(n_fft, sym=False)
                elif wname in ("hamming",):
                    w = _win.hamming(n_fft, sym=False)
                elif wname in ("bartlett",):
                    w = _win.bartlett(n_fft, sym=False)
                elif wname in ("kaiser",):
                    # Default beta for Kaiser (should match what's used in STFT)
                    beta = 6.5
                    w = _win.kaiser(n_fft, beta=beta, sym=False)
                elif wname in ("gaussian", "gauss", "gaussiana"):
                    # Default std for Gaussian (should match what's used in STFT)
                    std = n_fft / 8.0
                    w = _win.gaussian(n_fft, std=std, sym=False)
                else:
                    w = _win.hann(n_fft, sym=False)
            except Exception:
                # Fallback: assume rectangular window (no energy reduction)
                w = np.ones(n_fft)
        
        # Calculate window power (sum of window squared)
        # This represents the total energy of the window function
        window_power = float(np.sum(w ** 2))
        window_length = len(w)

        # FIX 7: define `overlap_factor` unconditionally, before the try/except.
        # The original code only assigned it inside the `except` branch but read
        # it again in the success-path return dict, which silently raised
        # NameError and was swallowed by the outer except — so the conservation
        # audit always returned NaNs when librosa was available.
        overlap_factor = (window_length / hop_length) if hop_length > 0 else 1.0

        # ====================================================================
        # PARSEVAL ENERGY-CONSERVATION NORMALISATION — canonical closed form.
        # ====================================================================
        # Mathematical foundation (Allen & Rabiner 1977; Smith, "Spectral
        # Audio Signal Processing", §STFT-Parseval; librosa istft source):
        #
        #     Σ_t Σ_k |S_t[k]|²_two_sided    =    N · Σ_m |y[m]|² · W[m]
        #
        # where N = n_fft, W[m] = Σ_t |w[m − t·hop]|² is the window
        # sum-of-squares envelope (a.k.a. ``window_sumsquare``), and the
        # one-sided / two-sided conversion accounts for DC and Nyquist not
        # being mirrored.
        #
        # In the "interior" of a sufficiently long signal W[m] is constant
        # at the value ``K = window_power · overlap_factor / N``, so the
        # energy-recovery factor reduces to the closed form
        #
        #     energy_time  ≈  energy_freq_one_sided / (N · K)
        #                   =  energy_freq_one_sided / (window_power · overlap_factor)
        #
        # AUDIT FIX (librosa.window_sumsquare warning, May 2026) — the
        # previous code path tried to call ``librosa.util.window_sumsquare``
        # (which moved to ``librosa.filters.window_sumsquare`` in librosa
        # ≥ 0.10) and on success divided by ``Σ_m W[m]`` instead of the
        # required ``N · mean(W) = window_power · overlap_factor``. The
        # ``Σ_m W[m] ≈ n_frames · window_power`` denominator under-
        # estimates the energy by a factor of roughly ``n_frames /
        # overlap_factor``, which would have produced energy_ratio ≈ 0.09
        # on the hann/2 s sine reference. The bug was masked for years
        # because the ``librosa.util.window_sumsquare`` import failed on
        # librosa ≥ 0.10 and the (correct) approximate branch always ran.
        # We now use the closed-form factor unconditionally — it is the
        # interior limit of the per-sample formula above and is what every
        # existing energy-conservation regression test (Parseval sine,
        # Parseval white noise, STFT golden) is calibrated against.
        #
        # Librosa is still required (and used heavily) elsewhere in this
        # module — for ``librosa.stft``, ``librosa.load``, window resampling
        # and other STFT operations — so removing this one mis-call does
        # not weaken the librosa dependency.
        # ====================================================================

        n_bins = S_freq.shape[0]
        dc_energy = float(np.sum(np.abs(S_freq[0, :]) ** 2))
        if n_fft % 2 == 0:
            nyquist_energy = float(np.sum(np.abs(S_freq[n_bins - 1, :]) ** 2))
            other_energy = float(np.sum(np.abs(S_freq[1:n_bins - 1, :]) ** 2))
        else:
            nyquist_energy = 0.0
            other_energy = float(np.sum(np.abs(S_freq[1:, :]) ** 2))

        _parseval_denominator = float(window_power * overlap_factor)
        if not np.isfinite(_parseval_denominator) or _parseval_denominator <= 0.0:
            # Pathological inputs (zero-amplitude window, hop_length<=0).
            # Returning NaN here lets the outer ``is_valid`` flag correctly
            # mark the audit as inconclusive.
            energy_freq_norm = float("nan")
        else:
            energy_freq_norm = (
                dc_energy + nyquist_energy + 2.0 * other_energy
            ) / _parseval_denominator
        
        # Calculate ratio (should be close to 1.0)
        if energy_time > 0:
            energy_ratio = energy_freq_norm / energy_time
        else:
            energy_ratio = float('nan')
        
        # Check if within tolerance
        deviation = abs(energy_ratio - 1.0)
        is_valid = deviation <= tolerance
        
        return {
            'energy_time': float(energy_time),
            'energy_freq': float(energy_freq),
            'energy_freq_norm': float(energy_freq_norm),
            'energy_ratio': float(energy_ratio),
            'deviation': float(deviation),
            'is_valid': bool(is_valid),
            'tolerance': float(tolerance),
            'window_power': float(window_power),  # Added for debugging
            'overlap_factor': float(overlap_factor)  # Added for debugging
        }
        
    except Exception as e:
        logger.warning(f"Failed to verify energy conservation: {e}")
        return {
            'energy_time': float('nan'),
            'energy_freq': float('nan'),
            'energy_freq_norm': float('nan'),
            'energy_ratio': float('nan'),
            'deviation': float('nan'),
            'is_valid': False,
            'tolerance': float(tolerance),
            'window_power': float('nan'),
            'overlap_factor': float('nan')
        }



def _parabolic_peak(y, x):
    """
    Parabolic interpolation (simplified QIFFT) around index x.
    Returns (xv, yv) — sub-bin position and amplitude.
    """
    if x <= 0 or x >= len(y) - 1:
        return x, float(y[x])
    alpha, beta, gamma = float(y[x-1]), float(y[x]), float(y[x+1])
    denom = (alpha - 2 * beta + gamma)
    if denom == 0.0:
        return x, beta
    p = 0.5 * (alpha - gamma) / denom
    xv = x + p
    yv = beta - 0.25 * (alpha - gamma) * p
    return xv, yv


# ============================================================================
# Harmonic detection: order-based grouping and local peak refinement
# Mathematical validation: standard reference
# ============================================================================

def _parabolic_interpolation_log_magnitude(
    magnitudes: np.ndarray,
    peak_idx: int,
    bin_spacing: float,
    freq_base: float
) -> Tuple[float, bool]:
    """
    Interpolação parabólica em log-magnitude para estimação sub-bin.
    
    Fundamentação Matemática (standard reference):
    - Para um pico localizado no bin k: y[k-1], y[k], y[k+1]
    - Assumindo forma parabólica em log-escala: log(y) = a·x² + b·x + c
    - Posição do pico (máximo): x_peak = -b/(2a) = -(Y₃ - Y₁)/(2·(Y₁ - 2Y₂ + Y₃))
    - Correção: f_corrected = f_bin + x_peak × Δf
    
    Args:
        magnitudes: Array de magnitudes (linear)
        peak_idx: Índice do bin de máximo
        bin_spacing: Espaçamento entre bins (Hz)
        freq_base: Frequência base (Hz) - para calcular frequência absoluta
    
    Returns:
        Tuple (frequência corrigida, validade)
    """
    if peak_idx <= 0 or peak_idx >= len(magnitudes) - 1:
        freq_corrected = freq_base + peak_idx * bin_spacing
        return freq_corrected, False
    
    # Converter para log-magnitude (mais linear que magnitude linear)
    log_mags = 20 * np.log10(np.maximum(magnitudes, 1e-10))
    
    # Amostras: k-1, k, k+1
    y1, y2, y3 = log_mags[peak_idx-1], log_mags[peak_idx], log_mags[peak_idx+1]
    
    # Parâmetros da parábola: y = a·x² + b·x + c
    # Resolvendo: a = (Y₁ - 2Y₂ + Y₃)/2, b = (Y₃ - Y₁)/2, c = Y₂
    a = (y1 - 2*y2 + y3) / 2.0
    b = (y3 - y1) / 2.0
    
    # Posição do pico: x_peak = -b/(2a)
    if abs(a) < 1e-10:  # Parábola degenerada (linha reta)
        x_peak = 0.0
    else:
        x_peak = -b / (2 * a)
    
    # Validar: x_peak deve estar em [-0.5, +0.5]
    if abs(x_peak) > 0.5:
        freq_corrected = freq_base + peak_idx * bin_spacing
        return freq_corrected, False
    
    # Calcular frequência corrigida
    freq_bin = freq_base + peak_idx * bin_spacing
    freq_corrected = freq_bin + x_peak * bin_spacing
    
    return freq_corrected, True


def _refine_peak_index(
    magnitudes: np.ndarray,
    approx_idx: int,
    *,
    refine_radius: int = 2,
) -> int:
    """Snap ``approx_idx`` to the nearest local maximum within
    ±``refine_radius`` bins.

    When the candidate index comes from ``argmin(|freqs - n·f0|)`` the
    expected harmonic frequency does not generally land on an FFT bin;
    the true peak can sit one or two bins away. Without this refinement,
    the ``mag[idx] > mag[idx±1]`` local-max check would fail for the
    majority of legitimate harmonics simply because the index was
    pointing at the lobe shoulder rather than the lobe top.
    """
    n = len(magnitudes)
    if n == 0:
        return int(approx_idx)
    lo = max(0, int(approx_idx) - int(refine_radius))
    hi = min(n, int(approx_idx) + int(refine_radius) + 1)
    if hi <= lo:
        return int(approx_idx)
    return int(lo + int(np.argmax(magnitudes[lo:hi])))


def _infer_bin_spacing_from_freqs(freqs: np.ndarray) -> float:
    """Infer FFT-bin spacing (Hz) from the frequency axis."""
    freqs = np.asarray(freqs, dtype=float)
    if freqs.size < 2:
        return float("nan")
    diffs = np.diff(freqs)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        return float("nan")
    return float(np.median(diffs))


def _refine_candidate_to_interpolated_peak(
    *,
    candidate_freq_hz: float,
    complete_magnitudes: np.ndarray,
    complete_freqs: np.ndarray,
    refine_radius: int = 2,
) -> Dict[str, Any]:
    """Snap a bin-centred candidate to a local magnitude peak and apply
    log-magnitude parabolic interpolation (:func:`_parabolic_interpolation_log_magnitude`).

    Returns audit fields for Harmonic Spectrum export: FFT bin index and
    centre frequency, sub-bin interpolated frequency, and peak linear / dB
    magnitude at the refined bin.
    """
    nan = float("nan")
    result: Dict[str, Any] = {
        "peak_bin_index": nan,
        "bin_center_frequency_hz": float(candidate_freq_hz),
        "interpolated_frequency_hz": float(candidate_freq_hz),
        "subbin_offset_bins": 0.0,
        "subbin_interpolation_valid": False,
        "peak_amplitude_raw": nan,
        "peak_magnitude_db": nan,
    }
    if complete_magnitudes is None or complete_freqs is None:
        return result

    mags = np.asarray(complete_magnitudes, dtype=float)
    freqs = np.asarray(complete_freqs, dtype=float)

    if mags.size == 0 or freqs.size == 0 or mags.size != freqs.size:
        return result

    idx0 = int(np.argmin(np.abs(freqs - float(candidate_freq_hz))))
    idx_peak = _refine_peak_index(mags, idx0, refine_radius=int(refine_radius))

    if idx_peak < 0 or idx_peak >= mags.size:
        return result

    bin_center = float(freqs[idx_peak])
    amp = float(mags[idx_peak])
    mag_db = float(20.0 * np.log10(max(amp, 1e-12)))

    bin_spacing = _infer_bin_spacing_from_freqs(freqs)
    if not np.isfinite(bin_spacing) or bin_spacing <= 0:
        result.update(
            {
                "peak_bin_index": int(idx_peak),
                "bin_center_frequency_hz": bin_center,
                "interpolated_frequency_hz": bin_center,
                "subbin_offset_bins": 0.0,
                "subbin_interpolation_valid": False,
                "peak_amplitude_raw": amp,
                "peak_magnitude_db": mag_db,
            }
        )
        return result

    if idx_peak <= 0 or idx_peak >= mags.size - 1:
        result.update(
            {
                "peak_bin_index": int(idx_peak),
                "bin_center_frequency_hz": bin_center,
                "interpolated_frequency_hz": bin_center,
                "subbin_offset_bins": 0.0,
                "subbin_interpolation_valid": False,
                "peak_amplitude_raw": amp,
                "peak_magnitude_db": mag_db,
            }
        )
        return result

    freq_base = float(freqs[0])
    freq_interp, valid = _parabolic_interpolation_log_magnitude(
        mags,
        idx_peak,
        bin_spacing,
        freq_base,
    )

    if not valid or not np.isfinite(freq_interp):
        freq_interp = bin_center
        valid = False

    offset_bins = float((float(freq_interp) - bin_center) / bin_spacing)

    result.update(
        {
            "peak_bin_index": int(idx_peak),
            "bin_center_frequency_hz": bin_center,
            "interpolated_frequency_hz": float(freq_interp),
            "subbin_offset_bins": offset_bins,
            "subbin_interpolation_valid": bool(valid),
            "peak_amplitude_raw": amp,
            "peak_magnitude_db": mag_db,
        }
    )
    return result


def _saddle_prominence_db(
    magnitudes: np.ndarray,
    peak_idx: int,
    *,
    saddle_window: int = 10,
) -> float:
    """Saddle-based peak prominence (in dB).

    Returns the peak height above the HIGHEST of the two local minima
    found within ±``saddle_window`` bins on either side of ``peak_idx``.

    Why a saddle window — and not the legacy ±1-bin comparison:

        For a Blackman–Harris window the main lobe is ≈ 2 bins wide
        between the −3 dB points, so even a perfectly clean sinusoid
        produces immediate neighbours that sit only ~0.1–1.5 dB below
        the peak. The legacy ``peak − max(left, right)`` formula
        therefore measured *main-lobe curvature*, not prominence:
        every real harmonic of a clarinet A♯3 came out at
        ``prominence_db`` ≈ 0.04–0.63 dB and was rejected by a
        ``> 3 dB`` strict gate — including the fundamental at SNR 75 dB.

        Saddle-based prominence (the standard definition used by
        ``scipy.signal.peak_prominences``) compares the peak to the
        local minimum on each flank, which for an isolated harmonic
        sits 20–40 dB below the peak. Real harmonics now pass the 3 dB
        gate; only side-lobes / noise excursions fail.

        ``saddle_window = 10`` bins is wide enough to step past the
        Blackman–Harris main lobe and side lobes on every tier used
        by this pipeline (smallest inter-harmonic spacing in the
        Clarinete_mf corpus is 27 bins at D3, n_fft = 8192), so the
        window never crosses into an adjacent harmonic.

    Returns ``-inf`` on array edges or empty windows.
    """
    mags = np.asarray(magnitudes, dtype=float)
    n = len(mags)
    if n < 3 or peak_idx <= 0 or peak_idx >= n - 1:
        return float("-inf")
    ls = max(0, peak_idx - int(saddle_window))
    rs = min(n, peak_idx + int(saddle_window) + 1)
    left_slice = mags[ls:peak_idx] if peak_idx > ls else None
    right_slice = mags[peak_idx + 1 : rs] if rs > peak_idx + 1 else None
    if (
        left_slice is None
        or right_slice is None
        or len(left_slice) == 0
        or len(right_slice) == 0
    ):
        return float("-inf")
    peak_db = 20.0 * np.log10(max(float(mags[peak_idx]), 1e-10))
    left_min_db = 20.0 * np.log10(max(float(np.min(left_slice)), 1e-10))
    right_min_db = 20.0 * np.log10(max(float(np.min(right_slice)), 1e-10))
    return float(peak_db - max(left_min_db, right_min_db))


def _is_local_peak_valid(
    magnitudes: np.ndarray,
    peak_idx: int,
    threshold_db: float = 3.0,
    noise_floor_percentile: float = 15.0,
    window_size: int = 50,
    saddle_window: int = 10,
) -> Tuple[bool, float]:
    """
    Valida se índice corresponde a pico local válido (não lóbulo lateral).

    Three independent checks:
      * local maximum:      mag[k] > mag[k±1]
      * saddle prominence:  peak − max(left_saddle, right_saddle) ≥ threshold_db
                            (saddle defined via ``_saddle_prominence_db``)
      * SNR vs noise floor: peak_db − percentile(local_window) ≥ 3 dB

    Args:
        magnitudes: Linear magnitude array.
        peak_idx: Index to evaluate.
        threshold_db: Minimum prominence above the saddle.
        noise_floor_percentile: Percentile for noise-floor estimation.
        window_size: Bin width of the noise-floor window (±).
        saddle_window: Bin width used to locate the prominence saddle (±).

    Returns:
        Tuple (is_valid, snr_db).
    """
    if peak_idx <= 0 or peak_idx >= len(magnitudes) - 1:
        return False, -np.inf

    peak_idx = _refine_peak_index(magnitudes, peak_idx, refine_radius=2)
    if peak_idx <= 0 or peak_idx >= len(magnitudes) - 1:
        return False, -np.inf

    log_mags = 20 * np.log10(np.maximum(magnitudes, 1e-10))

    peak_mag_db = log_mags[peak_idx]
    left_db = log_mags[peak_idx - 1]
    right_db = log_mags[peak_idx + 1]

    is_peak = peak_mag_db > left_db and peak_mag_db > right_db

    prom_db = _saddle_prominence_db(
        magnitudes, peak_idx, saddle_window=saddle_window
    )
    threshold_met = prom_db >= float(threshold_db)

    start_idx = max(0, peak_idx - window_size)
    end_idx = min(len(magnitudes), peak_idx + window_size)
    local_magnitudes = magnitudes[start_idx:end_idx]
    noise_floor_mag = np.percentile(local_magnitudes, noise_floor_percentile)
    noise_floor_db = 20 * np.log10(noise_floor_mag + 1e-10)

    snr_db = peak_mag_db - noise_floor_db
    snr_valid = snr_db >= 3.0

    is_valid = is_peak and threshold_met and snr_valid

    return is_valid, snr_db


# ---------------------------------------------------------------------------
# Harmonic-spectrum candidate helpers (see _generate_harmonic_list).
#
# The strict harmonic acceptance test (``_is_local_peak_valid``) requires a
# local maximum AND a ≥ 3 dB margin above both neighbours AND a SNR ≥ 3 dB
# vs. the local noise floor. That criterion is appropriate for diagnostics
# but rejects too many legitimate FFT-smeared harmonics for the density
# metric. The two helpers below split harmonic acceptance into:
#
#   * strict_harmonic_peaks       — what ``_is_local_peak_valid`` accepts
#                                   (drives ``harmonic_list_df`` / inharmonic
#                                   classification / robust f0 fit);
#   * harmonic_spectrum_candidates — one entry per expected harmonic order
#                                   that drives the per-note ``Harmonic
#                                   Spectrum`` sheet and the
#                                   ``harmonic_log_amplitude_density`` metric
#                                   used by Density_Metrics.
# ---------------------------------------------------------------------------

# Public list of candidate-status codes. Kept as a module-level tuple so
# downstream tests / metadata writers can reference the canonical names
# without re-implementing the enum.
HARMONIC_CANDIDATE_STATUS_VALUES: Tuple[str, ...] = (
    "strict_validated",
    "snr_validated",
    "weak_candidate",
    "below_noise_floor",
    "missing_window",
    "rejected_bad_f0",
)


def _local_peak_metrics(
    magnitudes: np.ndarray,
    peak_idx: int,
    *,
    noise_floor_percentile: float = 15.0,
    window_size: int = 50,
    saddle_window: int = 10,
) -> Tuple[bool, float, float]:
    """Compute ``(local_peak_valid, snr_db, prominence_db)`` for ``peak_idx``.

    * ``local_peak_valid``: ``mag[idx] > mag[idx-1] AND mag[idx] > mag[idx+1]``
      (strict local maximum on linear magnitudes).
    * ``snr_db``: ``20 * log10(mag[idx])`` minus the noise-floor level
      estimated as ``percentile(noise_floor_percentile)`` over a ±``window_size``
      bin window around ``peak_idx``.
    * ``prominence_db``: saddle-based prominence (see
      :func:`_saddle_prominence_db`). The legacy "peak − max(left, right)"
      formula measured Blackman–Harris main-lobe curvature, NOT prominence,
      which caused every real harmonic to fall below a 3 dB strict gate.

    Returns ``(False, -inf, -inf)`` when ``peak_idx`` is at an array edge.
    """
    mags = np.asarray(magnitudes, dtype=float)
    if peak_idx <= 0 or peak_idx >= len(mags) - 1:
        return False, float("-inf"), float("-inf")
    peak_idx = _refine_peak_index(mags, peak_idx, refine_radius=2)
    if peak_idx <= 0 or peak_idx >= len(mags) - 1:
        return False, float("-inf"), float("-inf")
    peak_lin = float(max(mags[peak_idx], 1e-10))
    left_lin = float(max(mags[peak_idx - 1], 1e-10))
    right_lin = float(max(mags[peak_idx + 1], 1e-10))
    peak_db = 20.0 * np.log10(peak_lin)
    left_db = 20.0 * np.log10(left_lin)
    right_db = 20.0 * np.log10(right_lin)
    is_local_peak = bool((peak_db > left_db) and (peak_db > right_db))
    prominence_db = _saddle_prominence_db(
        mags, peak_idx, saddle_window=saddle_window
    )
    s = max(0, peak_idx - window_size)
    e = min(len(mags), peak_idx + window_size)
    if e <= s:
        return is_local_peak, float("-inf"), prominence_db
    nf_mag = float(np.percentile(mags[s:e], noise_floor_percentile))
    nf_db = 20.0 * np.log10(max(nf_mag, 1e-10))
    snr_db = float(peak_db - nf_db)
    return is_local_peak, snr_db, prominence_db


def _classify_harmonic_candidate(
    *,
    amplitude_raw: float,
    local_peak_valid: bool,
    snr_db: float,
    prominence_db: float,
    strict_snr_db: float = 3.0,
    strict_prominence_db: float = 3.0,
    minimum_snr_db: float = 3.0,
    rejected_bad_f0: bool = False,
) -> Tuple[str, bool]:
    """Classify a per-order harmonic candidate.

    Returns ``(candidate_status, include_for_density)``.

    Density inclusion is deliberately stricter than candidate labelling:

        include_for_density == True iff candidate_status == ``strict_validated``.

    SNR can say that a spectral component is salient; it cannot by itself prove
    that the component is a valid harmonic partial. Therefore ``snr_validated``
    and ``weak_candidate`` remain useful diagnostic labels but are excluded from
    the harmonic density sum.

    This prevents harmonic density from being inflated by side-lobes, broadband
    shoulders, or non-harmonic peaks that happen to fall inside a wide harmonic
    search window.

    The prominence + local-peak + SNR gate that promotes ``strict_validated``
    is the only path that sets ``include_for_density`` to True.
    """
    if rejected_bad_f0:
        return "rejected_bad_f0", False
    if (
        amplitude_raw is None
        or not np.isfinite(amplitude_raw)
        or float(amplitude_raw) <= 0.0
    ):
        return "missing_window", False
    snr = float(snr_db) if np.isfinite(snr_db) else float("-inf")
    prom = float(prominence_db) if np.isfinite(prominence_db) else float("-inf")
    if local_peak_valid and snr >= strict_snr_db and prom >= strict_prominence_db:
        return "strict_validated", True
    if snr >= minimum_snr_db:
        return "snr_validated", False
    if snr > 0.0 or local_peak_valid:
        return "weak_candidate", False
    return "below_noise_floor", False


def _estimate_f0_global_robust(
    detected_freqs: np.ndarray,
    detected_amplitudes: np.ndarray,
    initial_f0: float,
    max_n: int = 15
) -> Dict[str, float]:
    """
    Estima f₀ por ajuste robusto usando múltiplos parciais.
    
    Fundamentação Matemática (standard reference):
    - Para cada harmónico n: f_n = n × f₀ + ε_n
    - Minimizar SSE ponderado: SSE = Σ w_n × (f_n - n×f₀)²
    - Solução analítica: f₀ = (Σ w_n × n × f_n) / (Σ w_n × n²)
    - Pesos: w_n = A_n² / max(A_n²) (ponderar por energia)
    
    Args:
        detected_freqs: Frequências detetadas (Hz)
        detected_amplitudes: Amplitudes correspondentes
        initial_f0: Estimativa inicial de f₀
        max_n: Número máximo de harmónicos a considerar
    
    Returns:
        Dict com f0_estimated, residual_std, residual_median, fit_quality, n_harmonics_used
    """
    if len(detected_freqs) < 2:
        return {
            'f0_estimated': initial_f0,
            'residual_std': 0.0,
            'residual_median': 0.0,
            'fit_quality': 0.0,
            'n_harmonics_used': len(detected_freqs)
        }
    
    # Atribuir cada frequência a um n provável
    n_assignments = np.round(detected_freqs / initial_f0).astype(int)
    n_assignments = np.clip(n_assignments, 1, max_n)
    
    # Pesos: normalizar amplitudes (harmónicos mais fortes têm mais peso)
    weights = detected_amplitudes / (np.max(detected_amplitudes) + 1e-10)
    weights = weights ** 2  # Ponderar por energia (A²)
    
    # Solução analítica (mínimos quadrados ponderados)
    # f₀ = (Σ w_n × n × f_n) / (Σ w_n × n²)
    numerator = np.sum(weights * n_assignments * detected_freqs)
    denominator = np.sum(weights * n_assignments ** 2)
    
    if denominator > 1e-10:
        f0_robust = numerator / denominator
    else:
        f0_robust = initial_f0
    
    # Calcular resíduos
    expected_freqs = n_assignments * f0_robust
    residuals = detected_freqs - expected_freqs
    
    # Métricas de qualidade
    weighted_sse = np.sum(weights * residuals ** 2)
    weight_sum = np.sum(weights)
    residual_std = np.sqrt(weighted_sse / weight_sum) if weight_sum > 0 else 0.0
    residual_median = np.median(np.abs(residuals))
    
    return {
        'f0_estimated': float(f0_robust),
        'residual_std': float(residual_std),
        'residual_median': float(residual_median),
        'fit_quality': float(residual_std / f0_robust) if f0_robust > 0 else 0.0,
        'n_harmonics_used': len(detected_freqs)
    }


def _correct_f0_candidate_against_prior(
    candidate_hz: float,
    prior_hz: float,
    *,
    max_harmonic_ratio: int = 6,
) -> Dict[str, Any]:
    """Correct a raw f0 detector estimate against a note-name prior.

    Tests identity and integer harmonic/subharmonic ratios (÷n, ×n) for
    n in ``2 .. max_harmonic_ratio`` to resolve common confusions (2·f0,
    3·f0, f0/2, …). Returns the corrected frequency with minimum absolute
    cent deviation from ``prior_hz``.
    """
    result: Dict[str, Any] = {
        "raw_hz": candidate_hz,
        "corrected_hz": None,
        "ratio_applied": None,
        "cents_error": None,
        "valid": False,
    }
    try:
        cand = float(candidate_hz)
        prior = float(prior_hz)
    except (TypeError, ValueError):
        return result

    if not np.isfinite(cand) or not np.isfinite(prior) or cand <= 0 or prior <= 0:
        return result

    candidates: List[Tuple[float, float]] = [(cand, 1.0)]
    for r in range(2, int(max_harmonic_ratio) + 1):
        candidates.append((cand / float(r), 1.0 / float(r)))
        candidates.append((cand * float(r), float(r)))

    best_hz: Optional[float] = None
    best_ratio: Optional[float] = None
    best_err = float("inf")

    for hz, ratio in candidates:
        if hz <= 0 or not np.isfinite(hz):
            continue
        err = abs(1200.0 * np.log2(hz / prior))
        if err < best_err:
            best_err = err
            best_hz = float(hz)
            best_ratio = float(ratio)

    if best_hz is None or best_ratio is None:
        return result

    result.update(
        {
            "corrected_hz": best_hz,
            "ratio_applied": best_ratio,
            "cents_error": float(best_err),
            "valid": True,
        }
    )
    return result


def _calculate_bin_spacing(sr: float, n_fft: int, zero_padding: int) -> float:
    """
    Calcula bin spacing real: Δf = SR / (N_FFT × ZP)
    
    Fundamentação Matemática (standard reference):
    - Bin spacing: Δf = SR / (N_FFT × ZP)
    - N_FFT efetivo: N_FFT_effective = N_FFT × ZP
    
    Args:
        sr: Sample rate (Hz)
        n_fft: Tamanho da FFT (sem padding)
        zero_padding: Fator de zero padding
    
    Returns:
        Bin spacing (Hz)
    """
    n_fft_effective = n_fft * zero_padding
    bin_spacing = sr / n_fft_effective
    return bin_spacing

# ----------------- Normalização de nível (RMS) global -----------------
def _normalize_level(y: np.ndarray, target_rms_db: float = -20.0) -> np.ndarray:
    """
    Normaliza o nível do sinal para um RMS alvo (dB), tornando métricas invariantes a ganho.
    
    Fundamentação Matemática (standard reference):
    - RMS: RMS = √(Σ y² / N)
    - dB: dB = 20 × log₁₀(RMS)
    - Gain: G = 10^((target_db - current_db) / 20)
    - Normalizado: y_norm = y × G
    """
    if y is None or len(y) == 0:
        return y
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    cur_db = 20.0 * np.log10(rms)
    gain = 10.0 ** ((target_rms_db - cur_db) / 20.0)
    return (y * gain).astype(y.dtype, copy=False)



# ====================================================
# MODULE-LEVEL UTILITIES (OUTSIDE AudioProcessor)
# ====================================================
def _extract_amplitude_column(df: pd.DataFrame) -> np.ndarray:
    """
    Extract an amplitude column robustly; if none exists, try converting from dB.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return np.asarray([], dtype=float)

    # candidatos usuais
    for col in ["Amplitude", "amplitude", "Amp", "amp"]:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(float)

    # converter de dB se existir
    for col in ["Magnitude (dB)", "Mag(dB)", "Mag_db", "Mag", "magnitude", "Magnitude"]:
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").fillna(-120.0)
            return np.power(10.0, v / 20.0).to_numpy(float)

    # fallback: first numeric column
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return s.fillna(0.0).to_numpy(float)
    return np.asarray([], dtype=float)


_NOTE_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def frequency_to_note_name(freq_hz: float, a4: float = 440.0) -> str:
    """
    Converte frequência (Hz) para nome de nota temperado igual (A4=440), com cents.
    Retorna string tipo: 'A#3 (+12.3 cents)'.
    """
    try:
        f = float(freq_hz)
    except Exception:
        return ""

    if not math.isfinite(f) or f <= 0.0:
        return ""

    # MIDI float (A4=69)
    midi = 69.0 + 12.0 * math.log2(f / a4)
    midi_round = int(round(midi))

    name = _NOTE_NAMES_SHARP[midi_round % 12]
    octave = (midi_round // 12) - 1

    # cents relativo à nota arredondada
    f_ref = a4 * (2.0 ** ((midi_round - 69) / 12.0))
    cents = 1200.0 * math.log2(f / f_ref)

    return f"{name}{octave} ({cents:+.2f} cents)"


# ====================================================
# MAIN CLASS
# ====================================================
class AudioProcessor:


    @lru_cache(maxsize=128)
    def calculate_fundamental_frequency(self, note: str) -> float:
        """
        Converte nome de nota (ex.: A4, A#4, Bb3, Ab4, C-1, C#5, etc.) em Hz (A4=440).
        Também aceita entrada numérica em Hz (ex.: "440", "440.0", 440).
        """
        s = (note or "").strip()
        if not s:
            self.logger.warning("Empty/None note in calculate_fundamental_frequency()")
            return 0.0

        # 1) FAST PATH: accept frequency in Hz
        try:
            s_num = s.replace(",", ".")
            f_hz = float(s_num)
            if f_hz > 0.0 and math.isfinite(f_hz):
                return float(f_hz)
        except Exception:
            pass

        # 2) Note parsing: letter + optional accidental + octave (accepts ♯ ♭)
        s0 = s.split()[0]
        m = re.search(r'([A-Ga-g])\s*([#b♯♭]?)\s*[-_]?(\-?\d+)', s0)
        if not m:
            self.logger.warning(f"Invalid note format: {note}")
            return 0.0

        letter = m.group(1).upper()
        acc = m.group(2)
        if acc == "♯":
            acc = "#"
        elif acc == "♭":
            acc = "b"

        octave = int(m.group(3))
        pitch = letter + acc

        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        flats = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#', 'Cb': 'B', 'Fb': 'E'}
        sharps = {'E#': 'F', 'B#': 'C'}

        if pitch in flats:
            pitch = flats[pitch]
        if pitch in sharps:
            pitch = sharps[pitch]

        if pitch not in names:
            self.logger.warning(f"Unrecognised pitch class: {pitch} (from {note})")
            return 0.0

        # 3) Frequency from semitones relative to C0
        freq_A4 = 440.0
        freq_C0 = freq_A4 * 2 ** (-4.75)  # C0 when A4=440

        idx = names.index(pitch)
        h = idx + 12 * octave
        f = freq_C0 * (2 ** (h / 12.0))

        self.logger.debug(f"F0({note}) = {f:.6f} Hz")
        return float(f)

    def _canonical_f0_hz_for_analysis(self) -> Tuple[float, str]:
        """Backward-compatible wrapper returning ``(f0_hz, f0_source)``."""
        f0_hz, f0_source, _ = self._canonical_f0_triplet_for_analysis()
        return float(f0_hz), str(f0_source)

    def _canonical_f0_triplet_for_analysis(self) -> Tuple[float, str, str]:
        """Authoritative fundamental-provenance path for acoustic analysis.

        Priority:
        ``f0_final`` (accepted fit or explicit fallback) →
        ``f0_initial`` / nominal →
        ``f0_prior_hz`` →
        unresolved NaN.
        """
        _acc_raw = getattr(self, "f0_fit_accepted", False)
        _acc = bool(_acc_raw is True or str(_acc_raw).strip().lower() in ("true", "1"))
        _triplet = canonical_f0_triplet(
            f0_final_hz=(getattr(self, "f0_final", None) if _acc else None),
            f0_initial_hz=getattr(self, "f0_initial", None),
            f0_prior_hz=getattr(self, "f0_prior_hz", None),
            f0_fit_accepted=_acc,
            f0_source=(
                str(getattr(self, "f0_final_source", None) or "").strip()
                or str(getattr(self, "f0_final_method", None) or "").strip()
                or str(getattr(self, "f0_source", None) or "").strip()
                or "f0_final"
            ),
        )
        return float(_triplet.f0_hz), str(_triplet.f0_source), str(_triplet.acoustic_f0_status)

    def _finalize_f0_state(
        self,
        *,
        nominal_hz: float,
        candidate_hz: float,
        accept_fit: bool,
        fit_quality: Optional[float] = None,
        residual_std_hz: Optional[float] = None,
        rejection_reason: Optional[str] = None,
    ) -> None:
        """Canonical f0 provenance finalizer.

        Single assignment point for ``f0_final``, ``f0_final_source``, ``f0_source``,
        ``f0_final_method``, ``f0_fit_accepted``, ``f0_robust_accepted``,
        ``f0_fit_quality``, ``f0_robust_residual_std``, ``f0_fit_rejection_reason``,
        and ``f0_detuning_cents_from_nominal`` after robust harmonic-series fitting.
        """
        nominal = float(nominal_hz)
        try:
            candidate = float(candidate_hz)
        except (TypeError, ValueError):
            candidate = float("nan")

        accepted = bool(
            accept_fit
            and np.isfinite(candidate)
            and candidate > 0.0
            and np.isfinite(nominal)
            and nominal > 0.0
        )

        rq: Optional[float] = None
        if fit_quality is not None:
            try:
                fq = float(fit_quality)
                rq = fq if np.isfinite(fq) else None
            except (TypeError, ValueError):
                rq = None
        self.f0_fit_quality = rq

        rs: Optional[float] = None
        if residual_std_hz is not None:
            try:
                rv = float(residual_std_hz)
                rs = rv if np.isfinite(rv) else None
            except (TypeError, ValueError):
                rs = None
        self.f0_robust_residual_std = rs
        self.f0_fit_residual_std_hz = rs

        if accepted:
            self.f0_final = float(candidate)
            self.f0_final_source = "prior_constrained_harmonic_fit"
            self.f0_source = "prior_constrained_harmonic_fit"
            self.f0_final_method = "prior_constrained_harmonic_fit"
            self.f0_fit_accepted = True
            self.f0_robust_accepted = True
            self.f0_fit_rejection_reason = None
        else:
            self.f0_final = float(nominal)
            self.f0_final_source = "filename_note_nominal_fallback_fit_rejected"
            self.f0_source = "filename_note_nominal_fallback_fit_rejected"
            self.f0_final_method = "nominal_or_initial_due_to_bad_fit"
            self.f0_fit_accepted = False
            self.f0_robust_accepted = False
            self.f0_fit_rejection_reason = (
                str(rejection_reason).strip()
                if rejection_reason is not None and str(rejection_reason).strip()
                else "robust_fit_rejected_or_unavailable"
            )

        if (
            np.isfinite(float(self.f0_final))
            and float(self.f0_final) > 0.0
            and np.isfinite(nominal)
            and nominal > 0.0
        ):
            self.f0_detuning_cents_from_nominal = float(
                1200.0 * np.log2(float(self.f0_final) / float(nominal))
            )
        else:
            self.f0_detuning_cents_from_nominal = None

        self._assert_f0_state_consistency()

    def _assert_f0_state_consistency(self) -> None:
        """Fail fast if f0 provenance fields contradict one another."""
        source = str(getattr(self, "f0_final_source", "") or "")
        accepted = bool(getattr(self, "f0_fit_accepted", False))

        if source == "prior_constrained_harmonic_fit" and not accepted:
            raise RuntimeError(
                "Inconsistent f0 state: f0_final_source is "
                "'prior_constrained_harmonic_fit' but f0_fit_accepted is False."
            )
        if accepted and source != "prior_constrained_harmonic_fit":
            raise RuntimeError(
                "Inconsistent f0 state: f0_fit_accepted is True but "
                f"f0_final_source is {source!r}."
            )
        if not accepted and source == "prior_constrained_harmonic_fit":
            raise RuntimeError(
                "Inconsistent f0 state: rejected fit cannot publish "
                "'prior_constrained_harmonic_fit' as final source."
            )

    # ----------------- janela p/ STFT -----------------
    def _get_window_arg(self):
        """
        Resolve the window vector passed to the STFT (``scipy.signal.get_window`` when applicable).
        """
        import numpy as _np
        from scipy import signal as _sig

        n = int(getattr(self, 'n_fft', DEFAULT_N_FFT) or DEFAULT_N_FFT)
        w = getattr(self, 'window', DEFAULT_WINDOW)

        def _info(msg): self.logger.info(msg)
        def _warn(msg): self.logger.warning(msg)
        def _err(msg): self.logger.error(msg)

        if isinstance(w, (list, tuple, _np.ndarray)):
            arr = _np.asarray(w, dtype=float).ravel()
            if arr.ndim != 1:
                _err(f"Provided window is not 1D (ndim={arr.ndim}).")
                raise ValueError("Provided window must be one-dimensional.")
            if arr.size != n:
                _err(f"Window length ({arr.size}) != n_fft ({n}).")
                raise ValueError("Window length must equal n_fft.")
            _info(f"STFT: n_fft={n}, window=array(len={arr.size})")
            return arr

        if isinstance(w, str):
            name = w.strip().lower()
            if name == 'kaiser':
                beta = float(getattr(self, 'kaiser_beta', 6.5))
                if beta < 0:
                    _warn(f"Negative kaiser_beta ({beta}); clamping to 6.5.")
                    beta = 6.5
                win = _sig.get_window(('kaiser', beta), n, fftbins=True)
                _info(f"STFT: n_fft={n}, window=kaiser(beta={beta})")
                return win

            if name in ('gaussian', 'gauss', 'gaussiana'):
                std = float(getattr(self, 'gaussian_std', n / 8.0))
                if std <= 0:
                    _warn(f"Non-positive gaussian_std ({std}); using n_fft/8.")
                    std = n / 8.0
                win = _sig.get_window(('gaussian', std), n, fftbins=True)
                _info(f"STFT: n_fft={n}, window=gaussian(std={std})")
                return win

            try:
                win = _sig.get_window(name, n, fftbins=True)
                _info(f"STFT: n_fft={n}, window={name}")
                return win
            except Exception as e:
                _warn(f"Window '{w}' unavailable ({e}); falling back to 'hann'.")
                return _sig.get_window('hann', n, fftbins=True)

        _err(f"Unsupported window type: {type(w).__name__}.")
        raise TypeError("Parameter 'window' must be str, list/tuple, or numpy.ndarray.")

    # ----------------- init -----------------
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # PHASE 3: Thread Safety
        # Lock for thread-safe access to instance variables
        # NOTE: AudioProcessor is designed for single-threaded use per instance.
        # This lock provides safety if multiple threads access the same instance.
        # For multi-threaded processing, create separate AudioProcessor instances per thread.
        self._lock = threading.Lock()

        # Dados de Ã¡udio
        self.audio_data: List[Tuple[np.ndarray, int, str, str]] = []
        self.y: Optional[np.ndarray] = None
        self.sr: Optional[int] = None

        # Resultados transformadas
        self.S: Optional[np.ndarray] = None
        self.db_S: Optional[np.ndarray] = None
        self.freqs: Optional[np.ndarray] = None
        self.times: Optional[np.ndarray] = None
        # --- flags de coerência de amplitude ---
        self._filtered_amp_corrected = False
        self._harmonic_amp_corrected = False
        self._complete_amp_corrected = False
        self._harmonic_amp_corrected = False




        # DataFrames
        self.complete_list_df: Optional[pd.DataFrame] = None
        self.filtered_list_df: Optional[pd.DataFrame] = None
        self.harmonic_list_df: Optional[pd.DataFrame] = None

        # MÃ©tricas
        self.density_metric_value: Optional[float] = None
        self.scaled_density_metric_value: Optional[float] = None
        self.filtered_density_metric_value: Optional[float] = None
        self.entropy_spectral_value: Optional[float] = None
        self.combined_density_metric_value: Optional[float] = None
        self.spectral_density_metric_value: Optional[float] = None
        self.total_metric_value: Optional[float] = None

        # Stage 1 harmonic-extraction surfaces (see _generate_harmonic_list).
        # ``harmonic_list_df`` continues to hold the strict-validated peaks
        # used by the inharmonic classifier and the robust f0 fit. The new
        # ``harmonic_spectrum_candidates_df`` carries one row per expected
        # harmonic order with the classification fields documented in
        # ``HARMONIC_CANDIDATE_STATUS_VALUES`` and is the source of truth
        # for the per-note ``Harmonic Spectrum`` sheet and for the
        # ``harmonic_log_amplitude_density`` metric used by Density_Metrics.
        self.harmonic_spectrum_candidates_df: Optional[pd.DataFrame] = None
        self.expected_harmonic_count: Optional[int] = None
        self.strict_harmonic_count: Optional[int] = None
        self.harmonic_candidate_count_density: Optional[int] = None
        self.harmonic_amplitude_sum: Optional[float] = None
        self.harmonic_log_amplitude_density: Optional[float] = None
        self.f0_initial: Optional[float] = None
        self.f0_prior_note: Optional[str] = None
        self.f0_prior_source: Optional[str] = None
        self.f0_nominal_hz: Optional[float] = None
        self.f0_prior_hz: Optional[float] = None
        self.f0_final: Optional[float] = None
        self.f0_final_source: Optional[str] = None
        self.f0_source: Optional[str] = None
        self.f0_detuning_cents_from_nominal: Optional[float] = None
        self.f0_fit_residual_std_hz: Optional[float] = None
        self.f0_fit_accepted: Optional[bool] = None
        self.f0_final_method: Optional[str] = None

        # DC removal (mean-centring) — not a high-pass filter; see ``Analysis_Metadata``.
        self.dc_offset_before_removal: Optional[float] = None
        self.dc_offset_after_removal: Optional[float] = None
        self.dc_removal_applied: Optional[bool] = None
        # Adaptive subfundamental guard (canonical; separate from fixed 30–200 Hz diagnostic band).
        self.adaptive_subfundamental_cutoff_hz: Optional[float] = None
        self.subfundamental_margin_percent: Optional[float] = None
        self.percentage_subfundamental_cutoff_hz: Optional[float] = None
        self.leakage_guard_cutoff_hz: Optional[float] = None
        self.effective_subfundamental_margin_percent: Optional[float] = None
        self.subfundamental_cutoff_selection_rule: str = ""
        self.subfundamental_cutoff_selected_by: str = ""
        self.subfundamental_guard_valid: bool = False
        self.subfundamental_guard_policy: str = "invalid_f0"
        self.low_frequency_policy_version: str = LOW_FREQUENCY_POLICY_VERSION
        self.physical_low_frequency_lower_hz: Optional[float] = None
        self.physical_low_frequency_upper_hz: Optional[float] = None
        self.spectral_density_freq_floor_hz: Optional[float] = None
        self.harmonic_leakage_protection_hz: Optional[float] = None
        self.low_frequency_aggregate_mode: str = "local_maxima"
        self.low_frequency_residual_interpretation: str = (
            "diagnostic residual; not automatically sub-bass, "
            "not automatically noise, not partial"
        )

        # DissonÃ¢ncia
        available_models = list_available_models()
        self.dissonance_values: Dict[str, Optional[float]] = {m: None for m in available_models}
        self.dissonance_curves: Dict[str, Optional[Dict]] = {m: None for m in available_models}
        self.dissonance_scales: Dict[str, Optional[List]] = {m: None for m in available_models}

        # ParÃ¢metros FFT
        self.n_fft: int = DEFAULT_N_FFT
        self.hop_length: Optional[int] = DEFAULT_HOP_LENGTH
        self.window: str = DEFAULT_WINDOW
        self.weight_function: str = 'linear'

        # DissonÃ¢ncia â€“ opÃ§Ãµes
        self.dissonance_enabled: bool = True
        self.dissonance_model: str = 'Sethares'
        self.dissonance_curve_enabled: bool = True
        self.dissonance_scale_enabled: bool = True
        self.dissonance_compare_models: bool = False

        # Pesos a/ß (alinhado com interface default: 95% / 5%)
        self.harmonic_weight: float = 0.95
        self.inharmonic_weight: float = 0.05

        # Outros
        self.results_directory: Path = Path("./results")
        # Preenchido em _compile_metrics (alinhado com SoundSpectrAnalyse-main_7 / compile_metrics DR audit)
        self.last_density_dr_audit: Dict[str, Any] = {}
        self.freq_min = 20.0
        self.freq_max = 20000.0
        self.db_min = -90.0
        self.db_max = 0.0
        self.tolerance = 10.0
        self.use_adaptive_tolerance = True
        self.zero_padding = 1
        self.time_avg = "mean"

        # --- Effective partial density (primary “fatness” metric) + energy accounting ---
        self.spectral_magnitude_smoothing_enabled: bool = bool(DEFAULT_STFT_MAGNITUDE_SMOOTHING_ENABLED)
        self.subbass_aggregate_hz: float = float(SUBBASS_AGGREGATE_CUTOFF_HZ)
        self.effective_partial_density: Optional[float] = None
        self.partial_density_effective_components: Optional[float] = None  # alias of effective_partial_density
        self.harmonic_energy_sum: Optional[float] = None
        self.inharmonic_energy_sum: Optional[float] = None
        self.subbass_energy_sum: Optional[float] = None
        self.total_component_energy: Optional[float] = None
        self.harmonic_energy_ratio: Optional[float] = None
        self.inharmonic_energy_ratio: Optional[float] = None
        self.subbass_energy_ratio: Optional[float] = None
        # Same partial vectors as harmonic_energy_sum / inharmonic_energy_sum (linear ΣA, not ΣA²).
        self.linear_sum_amplitude_harmonic: Optional[float] = None
        self.linear_sum_amplitude_inharmonic_partial: Optional[float] = None
        self.linear_sum_amplitude_subbass_band: Optional[float] = None
        self.linear_amplitude_fraction_inharmonic_of_HI: Optional[float] = None
        self.linear_amplitude_fraction_nonharmonic_of_total: Optional[float] = None
        self.linear_amplitude_batch_alignment_factor: Optional[float] = None
        # Inharmonic *energy* path uses the same harmonic-window residual mask as export
        # (``identify_nonharmonic_residual_rows``); rows are not confirmed inharmonic partials.
        self._metrics_ih_amps_eff: np.ndarray = np.asarray([], dtype=float)
        self._metrics_ih_freqs_eff: Optional[np.ndarray] = None
        self.harmonic_partial_count: Optional[int] = None
        self.inharmonic_partial_count: Optional[int] = None
        self.total_detected_partial_count: Optional[int] = None
        self.unique_harmonic_order_count: Optional[int] = None
        self.harmonic_order_count: Optional[int] = None
        # Classified spectral rows (debug semantics; not verified local-max peak picking on full spectrum)
        self.harmonic_peak_count: Optional[int] = None
        self.inharmonic_peak_count: Optional[int] = None
        self.subbass_peak_count: Optional[int] = None
        self.total_detected_peak_count: Optional[int] = None
        self.harmonic_peak_candidate_count: Optional[int] = None
        self.nonharmonic_peak_candidate_count: Optional[int] = None
        self.low_frequency_peak_candidate_count: Optional[int] = None
        self.total_peak_candidate_count: Optional[int] = None
        self.residual_spectral_row_count: Optional[int] = None
        self.nonharmonic_candidate_row_count: Optional[int] = None
        self.retained_nonharmonic_peak_candidate_count: Optional[int] = None
        self.exported_nonharmonic_peak_candidate_count: Optional[int] = None
        self.peaklist_harmonic_window_candidate_count: Optional[int] = None
        self.peaklist_nonharmonic_window_candidate_count: Optional[int] = None
        self.peaklist_low_frequency_window_candidate_count: Optional[int] = None
        self.peaklist_total_window_candidate_count: Optional[int] = None
        self.debug_counts_invariant_status: str = ""
        self.debug_counts_invariant_failures: str = ""
        self.accepted_inharmonic_peak_count: Optional[int] = None
        self.accepted_inharmonic_partial_count: Optional[int] = None
        self.harmonic_candidate_count: Optional[int] = None
        self.inharmonic_candidate_count: Optional[int] = None
        self.subbass_candidate_count: Optional[int] = None
        self.total_spectral_candidate_count: Optional[int] = None
        self.residual_row_count: Optional[int] = None
        self.harmonic_bin_count: Optional[int] = None
        self.inharmonic_bin_count: Optional[int] = None
        self.subbass_bin_count: Optional[int] = None
        self.energy_conservation_status: Optional[str] = None
        self.energy_conservation_error: Optional[float] = None
        self.energy_denominator_description: Optional[str] = None
        self.dissonance_partial_count: Optional[int] = None
        self.dissonance_pair_count: Optional[int] = None
        self.dissonance_partial_cap: Union[int, str, None] = None
        self.dissonance_partial_count_before_cap: Optional[int] = None
        self.dissonance_partial_count_after_cap: Optional[int] = None
        self.dissonance_pair_count_after_cap: Optional[int] = None
        self.dissonance_cap_computation_note: Optional[str] = None
        self.harmonic_validation_report: Optional[Dict[str, Any]] = None
        self.f0_prior_available: Optional[bool] = None
        self.f0_blind_method: Optional[str] = None
        self.f0_fit_accepted: Optional[bool] = None
        self.f0_fit_quality: Optional[float] = None
        self.f0_fit_rejection_reason: Optional[str] = None

        self.canonical_density_v5_adapted: Optional[float] = None
        self.discrete_metric_d3: Optional[float] = None
        self.discrete_metric_d10: Optional[float] = None
        self.discrete_metric_d17: Optional[float] = None
        self.discrete_metric_d24: Optional[float] = None
        self.density_per_component: Optional[float] = None
        self.density_formula_version: Optional[str] = None
        self.density_source_formula: Optional[str] = None
        self.density_normalization_scope: Optional[str] = None
        self.density_normalization_denominator: Optional[float] = None
        self.density_metric_ratio_over_fundamental_legacy: Optional[float] = None

        self.component_energy_status: str = "not_computed"
        self.component_energy_pie_basis: str = "not_written"
        self.amplitude_mass_chart_file: str = ""
        self.amplitude_mass_chart_basis: str = "linear_amplitude_sum"
        self.amplitude_mass_chart_interpretation: str = (
            "diagnostic_candidate_mass_not_energy"
        )
        self.energy_ratio_chart_file: str = ""
        self.energy_ratio_chart_basis: str = "component_power_energy_ratios"
        self.energy_ratio_chart_interpretation: str = "acoustic_energy_balance"
        self.amplitude_mass_chart_status: str = "not_attempted"
        self.energy_ratio_chart_status: str = "not_attempted"
        self.component_energy_pie_file: str = ""
        self.component_energy_pie_alias_basis: str = ""
        self.effective_partial_density_status: str = "not_computed"
        self.density_metric_status: str = "not_computed"
        self.normalization_status: str = "not_computed"
        self.debug_counts_status: str = "not_computed"
        self.model_weight_status: str = "not_computed"
        self.model_weight_fallback_applied: bool = False

        self.logger.info("AudioProcessor initialised")

    # ----------------- audio loading -----------------
    def _load_audio_with_fallback(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
        p = str(file_path).replace('"', '').replace("'", "")
        if not os.path.exists(p):
            self.logger.error(f"File not found: {p}")
            return None, None

        try:
            y, sr = librosa.load(p, sr=None)
            if y is not None and len(y) > 0:
                return y, sr
        except Exception:
            pass

        try:
            import soundfile as sf
            if p.lower().endswith(('.aiff', '.aif')):
                data, samplerate = sf.read(p)
                return data, samplerate
        except Exception:
            pass

        try:
            from scipy.io import wavfile
            if p.lower().endswith('.wav'):
                samplerate, data = wavfile.read(p)
                if getattr(data, "dtype", None) is not None and data.dtype.kind not in 'fc':
                    data = data.astype(np.float32) / np.iinfo(data.dtype).max
                return data, samplerate
        except Exception:
            pass

        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(p)
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            samples = samples / float(1 << (8 * audio.sample_width - 1))
            if audio.channels > 1:
                samples = samples.reshape((-1, audio.channels)).mean(axis=1)
            return samples, audio.frame_rate
        except Exception as e:
            self.logger.error(f"All loading methods failed: {e}")
            return None, None

    def load_audio_files(self, file_paths: List[Union[str, Path]]) -> None:
        start = time.time()
        self.audio_data.clear()
        for p in file_paths:
            try:
                p = Path(str(p).replace('"', '').replace("'", "")).resolve()
                if not p.exists():
                    self.logger.error(f"File not found: {p}")
                    continue
                y, sr = self._load_audio_with_fallback(p)
                if y is None or sr is None or len(y) == 0:
                    self.logger.warning(f"Invalid audio data: {p}")
                    continue
                y = np.asarray(y, dtype=np.float64)
                dc_offset_before = float(np.mean(y))
                y = y - dc_offset_before
                dc_offset_after = float(np.mean(y))
                self.dc_offset_before_removal = dc_offset_before
                self.dc_offset_after_removal = dc_offset_after
                self.dc_removal_applied = True
                note = self.extract_note_name(p) or p.stem
                self.audio_data.append((y, sr, note, str(p)))
                self.logger.debug(f"Loaded: {p} (note: {note})")
            except Exception as e:
                self.logger.error(f"Error loading {p}: {e}")
        self.logger.info(f"Loaded {len(self.audio_data)} file(s) in {time.time() - start:.2f}s.")

    def extract_note_name(self, file_path: Union[str, Path]) -> Optional[str]:
        name = file_path.name if isinstance(file_path, Path) else os.path.basename(str(file_path))
        name = name.replace('"', '').replace("'", "")
        patterns = [r"([A-G][#b]?)[-_]?(\d)", r"([A-G][#b]?)(\d)"]
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                return m.group(1) + m.group(2)
        return None

    def _spectral_leakage_guard_kwargs(self) -> Dict[str, Any]:
        """Optional kwargs for ``density.identify_nonharmonic_residual_rows`` (STFT leakage widening)."""
        try:
            sr = float(self.sr)
            zp = int(getattr(self, "zero_padding", 1) or 1)
            n_base = int(self.n_fft)
            if not (np.isfinite(sr) and sr > 0.0 and n_base > 0):
                return {}
            bw = float(_calculate_bin_spacing(sr, n_base, zp))
            ml = getattr(self, "window_main_lobe_width", None)
            try:
                ml_f = float(ml) if ml is not None and np.isfinite(float(ml)) and float(ml) > 0.0 else None
            except (TypeError, ValueError):
                ml_f = None
            out: Dict[str, Any] = {
                "bin_width_hz": bw,
                "spectral_leakage_guard": True,
            }
            if ml_f is not None:
                out["main_lobe_bins"] = ml_f
            return out
        except Exception:
            return {}

    def _finalize_low_frequency_policy_state(self) -> None:
        """
        Canonical low-frequency / subfundamental policy finalizer.

        Must be called after ``f0_final`` has been established (``_finalize_f0_state``).
        """
        from low_frequency_policy import (
            LOW_FREQUENCY_POLICY_VERSION,
            calculate_adaptive_subfundamental_cutoff_hz,
        )

        try:
            min_floor = float(getattr(self, "freq_min", 20.0) or 20.0)
        except (TypeError, ValueError):
            min_floor = 20.0
        f0 = getattr(self, "f0_final", None)

        leak_arg: Optional[float] = None
        try:
            f0f = float(f0) if f0 is not None else float("nan")
            sr = float(getattr(self, "sr", None) or getattr(self, "sample_rate", None) or 0.0)
            nff = int(getattr(self, "n_fft", 0) or 0)
            if np.isfinite(f0f) and f0f > 0.0 and np.isfinite(sr) and sr > 0.0 and nff > 0:
                tol = float(compute_subbass_protection_tolerance_hz(sr, nff))
                if np.isfinite(tol) and tol > 0.0:
                    lr = float(f0f - tol)
                    if np.isfinite(lr) and lr > 0.0:
                        leak_arg = lr
        except Exception:
            pass

        guard = calculate_adaptive_subfundamental_cutoff_hz(
            f0,
            min_floor_hz=min_floor,
            max_fraction_of_f0=0.95,
            leakage_guard_cutoff_hz=leak_arg,
        )

        self.low_frequency_policy_version = LOW_FREQUENCY_POLICY_VERSION
        self.adaptive_subfundamental_cutoff_hz = guard["adaptive_subfundamental_cutoff_hz"]
        self.subfundamental_margin_percent = guard["subfundamental_margin_percent"]
        self.percentage_subfundamental_cutoff_hz = guard["percentage_subfundamental_cutoff_hz"]
        self.leakage_guard_cutoff_hz = guard["leakage_guard_cutoff_hz"]
        self.effective_subfundamental_margin_percent = guard["effective_subfundamental_margin_percent"]
        self.subfundamental_cutoff_selection_rule = str(
            guard.get("subfundamental_cutoff_selection_rule") or ""
        )
        self.subfundamental_cutoff_selected_by = str(guard.get("subfundamental_cutoff_selected_by") or "")
        self.subfundamental_guard_valid = bool(guard["subfundamental_guard_valid"])
        self.subfundamental_guard_policy = str(guard["subfundamental_guard_policy"])

        self.physical_low_frequency_lower_hz = 30.0
        self.physical_low_frequency_upper_hz = 200.0
        self.low_frequency_residual_interpretation = (
            "diagnostic residual; not automatically sub-bass, "
            "not automatically noise, not partial"
        )

    def _density_relevant_frequency_floor_hz(self) -> float:
        """Lower frequency bound for density / nonharmonic masks (max of user floor and adaptive guard)."""
        try:
            fm = float(getattr(self, "freq_min", 20.0) or 20.0)
        except (TypeError, ValueError):
            fm = 20.0
        if not getattr(self, "subfundamental_guard_valid", False):
            return fm
        ad = getattr(self, "adaptive_subfundamental_cutoff_hz", None)
        if ad is None:
            return fm
        try:
            adf = float(ad)
        except (TypeError, ValueError):
            return fm
        if not np.isfinite(adf):
            return fm
        return max(fm, adf)

    def _apply_density_relevant_frequency_floor_to_filtered_list(self) -> None:
        """Trim ``filtered_list_df`` for density paths; ``complete_list_df`` stays full for diagnostics."""
        floor = float(self._density_relevant_frequency_floor_hz())
        self.spectral_density_freq_floor_hz = floor
        fl = getattr(self, "filtered_list_df", None)
        if fl is None or getattr(fl, "empty", True) or "Frequency (Hz)" not in fl.columns:
            return
        fq = pd.to_numeric(fl["Frequency (Hz)"], errors="coerce")
        self.filtered_list_df = fl.loc[fq >= floor].copy().reset_index(drop=True)
        self.logger.debug(
            "Density-relevant frequency floor applied to filtered_list_df: floor=%.4f Hz rows=%d",
            floor,
            int(len(self.filtered_list_df)),
        )

    def _dataframe_for_density_frequency_floor(self, df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Return a copy of ``df`` restricted to f >= density floor, or ``df`` if unusable."""
        if df is None or getattr(df, "empty", True) or "Frequency (Hz)" not in df.columns:
            return df
        floor = float(self._density_relevant_frequency_floor_hz())
        fq = pd.to_numeric(df["Frequency (Hz)"], errors="coerce")
        out = df.loc[fq >= floor].copy()
        return out if not out.empty else df.iloc[0:0].copy()

    # ----------------- FFT -----------------
    def fft_analysis(self, zero_padding: int = 1) -> None:
        """
        STFT with basic memory handling and parameter restoration.
        - Preserves the existing processing flow.
        - Adds universal coherent-gain calculation for the analysis window.
        
        PHASE 3: Thread-safe implementation
        NOTE: This method uses a lock to ensure thread safety when accessing
        instance variables. For multi-threaded processing, create separate
        AudioProcessor instances per thread.
        """
        # PHASE 3: Thread safety - acquire lock for entire critical section
        with self._lock:
            if self.y is None or self.sr is None:
                raise ValueError("Audio data not loaded.")

            start_time = time.time()
            orig_n_fft = int(self.n_fft)
            # FIXED: Preserve hop_length exactly as passed - don't override tier-specific values
            orig_hop = self.hop_length if self.hop_length is not None else None

            # hop por omissÃ£o (only if not already set - preserves tier-specific values)
            if self.hop_length is None:
                self.hop_length = self.n_fft // 4

            # Cap n_fft for short signals (prevents oversized windows on short/low notes)
            sig_len = len(self.y)
            if sig_len > 0:
                zp = int(zero_padding) if zero_padding and zero_padding > 0 else 1
                max_n_fft = sig_len // zp
                if max_n_fft <= 0:
                    max_n_fft = sig_len
                if self.n_fft * zp > sig_len:
                    # Prefer power-of-two cap for FFT efficiency when possible
                    capped_pow2 = 2 ** int(np.floor(np.log2(max_n_fft))) if max_n_fft >= 2 else max_n_fft
                    capped_n_fft = capped_pow2 if capped_pow2 > 0 else max_n_fft
                    # If signal is very short, allow n_fft to be the raw length
                    if capped_n_fft < 256:
                        capped_n_fft = max_n_fft
                    if capped_n_fft < self.n_fft:
                        self.logger.warning(
                            f"n_fft ({self.n_fft}) * zp({zp}) > signal length ({sig_len}); "
                            f"capping n_fft to {capped_n_fft}"
                        )
                        self.n_fft = int(capped_n_fft)
                        if orig_hop is None:
                            self.hop_length = self.n_fft // 4
            if self.hop_length is not None and self.hop_length > self.n_fft:
                self.logger.warning(
                    f"hop_length ({self.hop_length}) > n_fft ({self.n_fft}); "
                    f"capping hop_length to n_fft//4"
                )
                self.hop_length = max(1, self.n_fft // 4)

            # janela (argumento para librosa)
            win_arg = self._get_window_arg()
            win_length = len(win_arg)
        
            # ---------- PHASE 1: Spectral Leakage Quantification ----------
            # Calculate and log window characteristics (main-lobe width, side-lobe level)
            try:
                window_chars = _calculate_window_characteristics(self.window, int(self.n_fft))
                self.logger.info(
                    f"Window characteristics: main_lobe_width={window_chars['main_lobe_width']:.2f} bins, "
                    f"side_lobe_level={window_chars['side_lobe_level']:.2f} dB, "
                    f"peak_level={window_chars['peak_level']:.2f} dB"
                )
                # Store for potential use in analysis
                self.window_main_lobe_width = window_chars['main_lobe_width']
                self.window_side_lobe_level = window_chars['side_lobe_level']
            except Exception as e:
                self.logger.warning(f"Failed to calculate window characteristics: {e}")
                self.window_main_lobe_width = float('nan')
                self.window_side_lobe_level = float('nan')
            # -----------------------------------------------------------------

            # aplica zero padding se necessÃ¡rio
            n_fft_padded = win_length * zero_padding

            # ajuste leve p/ sinais gigantes
            if sig_len > LARGE_SIGNAL_THRESHOLD:
                adj = min(self.n_fft, FFT_MIN_SIZE * 8)  # 8192 = FFT_MIN_SIZE * 8
                if adj != self.n_fft:
                    self.logger.info(f"Reducing n_fft {self.n_fft} -> {adj} (long signal)")
                    self.n_fft = adj
                    if orig_hop is None:
                        self.hop_length = self.n_fft // 4

            # possÃ­vel amostragem parcial (proteÃ§Ã£o de memÃ³ria)
            y_work = (self.y[:LARGE_SIGNAL_THRESHOLD * SIGNAL_TRUNCATION_FACTOR] 
                     if sig_len > MAX_SIGNAL_LENGTH else self.y)

            tried_downgrade = False
            try:
                gc.collect()
                # PHASE 1: Fix duplicate normalization (removed duplicate line)
                # PHASE 3: Use constant instead of magic number
                # RMS normalisation: downstream density metrics describe spectral *shape* at a
                # reference level, not absolute concert loudness.
                y_norm = _normalize_level(y_work, target_rms_db=NORMALIZATION_TARGET_RMS_DB)
                self.S = librosa.stft(
                    y_norm,
                    n_fft=n_fft_padded,
                    win_length=win_length,
                    hop_length=self.hop_length,
                    window=win_arg,
                    center=True,
                )


                S_mag = np.abs(self.S)
                
                # ---------- Optional STFT magnitude smoothing (default from constants; v5 used always-on) ----------
                # When enabled, Savitzky–Golay runs on |STFT| before peak lists / density.
                # Default OFF in v6 so partial-based density is not silently reshaped; set True for v5-like spectra.
                if getattr(self, "spectral_magnitude_smoothing_enabled", False):
                    try:
                        from density import apply_spectral_smoothing

                        S_mag_smoothed = apply_spectral_smoothing(
                            S_mag,
                            method="savitzky_golay",
                            window_length=None,
                            polyorder=SMOOTHING_POLYORDER,
                            noise_floor_percentile=SMOOTHING_NOISE_FLOOR_PERCENTILE,
                            noise_floor_multiplier=SMOOTHING_NOISE_FLOOR_MULTIPLIER,
                        )
                        S_mag = S_mag_smoothed
                        self.logger.debug("Applied spectral magnitude smoothing (optional).")
                    except Exception as e:
                        self.logger.warning("Spectral magnitude smoothing requested but failed: %s", e)
                else:
                    self.logger.debug(
                        "STFT magnitude smoothing disabled; |STFT| unchanged before peaks/density."
                    )
                # -------------------------------------------------------------------------
                
                # dB de MAGNITUDE para visualizaÃ§Ã£o/thresholds; cÃ¡lculos mÃ©dios serÃ£o em POTÃŠNCIA noutro passo
                # FIX: Use absolute reference (ref=1.0) instead of per-file normalization (ref=np.max)
                # This ensures consistent dB scaling across all files, eliminating per-file normalization
                # artifacts that cause first notes to have different density values.
                # RMS normalization (already applied via _normalize_level) handles gain differences.
                self.db_S = librosa.amplitude_to_db(S_mag, ref=1.0)
                self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft_padded)
                frame_idx = np.arange(self.S.shape[1])
                self.times = librosa.frames_to_time(frame_idx, sr=self.sr, hop_length=self.hop_length)
                
                # ---------- EDGE FRAME HANDLING: Calculate correction weights for STFT edge effects ----------
                # With center=True, librosa.stft pads signal with reflected padding (length = n_fft//2)
                # First and last frames have reduced effective signal coverage, causing energy reduction
                # This correction applies weights to compensate for edge effects
                # Mathematical basis: Effective coverage = (pad_length - distance) / pad_length
                # Correction factor = 1 / max(effective_coverage, 0.5) to restore energy
                n_frames = self.S.shape[1]
                self.frame_weights = self._calculate_edge_frame_weights(n_frames, n_fft_padded)
                
                # Log edge frame information for debugging
                first_edge, last_edge = self._calculate_edge_frame_counts(n_frames, n_fft_padded)
                if first_edge > 0 or last_edge > 0:
                    avg_weight_first = np.mean(self.frame_weights[:first_edge]) if first_edge > 0 else 1.0
                    avg_weight_last = np.mean(self.frame_weights[-last_edge:]) if last_edge > 0 else 1.0
                    # Changed to INFO level for visibility - edge frame correction is important to verify
                    self.logger.info(
                        f"Edge frame correction: {first_edge} first frames (avg weight={avg_weight_first:.2f}), "
                        f"{last_edge} last frames (avg weight={avg_weight_last:.2f}), "
                        f"first frame weight={self.frame_weights[0]:.2f}x"
                    )
                # --------------------------------------------------------------------------------------------

                # ---------- coherent gain: calcular e guardar universalmente ----------
                cg_val = 1.0
                try:
                    # Preferir helper, se existir no mÃ³dulo
                    cg_val = float(_coherent_gain(self.window, int(self.n_fft)))  # type: ignore[name-defined]
                except Exception:
                    try:
                        # Fallback robusto: calcular mÃ©dia da janela efetiva
                        # Se win_arg for especificaÃ§Ã£o (str/tuple), obter vetor da janela
                        try:
                            w_vec = librosa.filters.get_window(win_arg, int(self.n_fft), fftbins=True)
                        except Exception:
                            # se win_arg jÃ¡ for vetor NumPy
                            w_vec = np.array(win_arg, dtype=float) if hasattr(win_arg, "__len__") else np.hanning(int(self.n_fft))
                        cg_val = float(np.mean(w_vec))
                    except Exception:
                        cg_val = 1.0
                # evitar zero/negativos
                self.coherent_gain_value = cg_val if cg_val > 0.0 else 1.0
                # ---------------------------------------------------------------------
                
                # ---------- PHASE 1: Energy Conservation Verification (Parseval's Theorem) ----------
                # Verify that energy is conserved: Σ|x[n]|² ≈ (1/N) * Σ|X[k]|²
                try:
                    # Pass the actual window array for accurate energy calculation
                    energy_result = _verify_energy_conservation(
                        y_norm, self.S, n_fft_padded, self.hop_length, self.window, 
                        tolerance=ENERGY_CONSERVATION_TOLERANCE,
                        window_array=win_arg  # Pass actual window array for accuracy
                    )
                    
                    if energy_result['is_valid']:
                        self.logger.info(
                            f"Energy conservation: OK (ratio={energy_result['energy_ratio']:.4f}, "
                            f"deviation={energy_result['deviation']:.2%})"
                        )
                    else:
                        self.logger.warning(
                            f"Energy conservation: VIOLATION (ratio={energy_result['energy_ratio']:.4f}, "
                            f"deviation={energy_result['deviation']:.2%}, tolerance={energy_result['tolerance']:.2%})"
                        )
                        self.logger.warning(
                            f"  Time-domain energy: {energy_result['energy_time']:.6e}, "
                            f"Frequency-domain energy (normalized): {energy_result['energy_freq_norm']:.6e}"
                        )
                    
                    # Store for potential use in analysis
                    self.energy_conservation_ratio = energy_result['energy_ratio']
                    self.energy_conservation_valid = energy_result['is_valid']
                except Exception as e:
                    self.logger.warning(f"Failed to verify energy conservation: {e}")
                    self.energy_conservation_ratio = float('nan')
                    self.energy_conservation_valid = False
                # ------------------------------------------------------------------------------------

                # ---------- PHASE 2: Temporal Evolution Analysis ----------
                # Calculate temporal metrics. ``attack_time`` is still
                # computed for backward-compatibility with downstream
                # consumers / tests, but it is NOT logged: the analysis
                # corpus is assumed to be pre-trimmed (silence and
                # transient leading edge already removed by the user
                # before feeding the pipeline), so an attack-time
                # measurement on these signals would just report the
                # first-non-silent-frame artifact, which is misleading.
                try:
                    temporal_metrics = _calculate_temporal_evolution(
                        S_mag, self.times, self.freqs, self.sr
                    )

                    self.spectral_flux = temporal_metrics['spectral_flux']
                    self.attack_time = temporal_metrics['attack_time']
                    self.spectral_centroid_evolution = temporal_metrics['spectral_centroid_evolution']
                    self.spectral_rolloff_evolution = temporal_metrics['spectral_rolloff_evolution']

                    self.logger.info(
                        f"Spectral flux (mean): {self.spectral_flux:.4f}"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to calculate temporal evolution: {e}")
                    self.spectral_flux = float('nan')
                    self.attack_time = float('nan')
                    self.spectral_centroid_evolution = None
                    self.spectral_rolloff_evolution = None
                # -----------------------------------------------------------------
                
                self.logger.info(f"FFT completed in {time.time()-start_time:.3f}s (shape={self.S.shape})")
            except MemoryError:
                self.logger.error("MemoryError in STFT")
                if not tried_downgrade:
                    tried_downgrade = True
                    # PHASE 3: Use constant instead of magic number
                    self.n_fft = max(FFT_MIN_SIZE, self.n_fft // FFT_DOWNGRADE_FACTOR)
                    self.hop_length = self.n_fft // 4
                    self.S = None
                    self.db_S = None
                    gc.collect()
                    # Release lock before recursive call
                    # Lock will be re-acquired in recursive call
                    return self.fft_analysis()
                else:
                    raise
            except Exception as e:
                raise RuntimeError(f"Error in FFT: {e}")
            finally:
                # PHASE 3: Restore original values (thread-safe, still in lock)
                # FIXED: Always restore orig_hop if it was set, to preserve tier-specific hop_length
                self.n_fft = orig_n_fft
                if orig_hop is not None:
                    self.hop_length = orig_hop  # Restore exact value passed (preserves tier settings)
                elif self.hop_length is None:
                    self.hop_length = self.n_fft // 4  # Only set default if still None
        # PHASE 3: Lock released automatically when exiting 'with' block

    def _calculate_edge_frame_counts(self, n_frames: int, n_fft: int) -> tuple[int, int]:
        """
        Calculate number of edge frames affected by STFT padding.
        
        With center=True, librosa.stft pads signal with reflected padding of length n_fft//2.
        Frames within pad_length/hop_length of the start/end are affected by edge effects.
        
        Args:
            n_frames: Total number of STFT frames
            n_fft: FFT size (including zero padding)
            
        Returns:
            (first_edge_count, last_edge_count): Number of edge frames at start and end
        """
        pad_length = n_fft // 2  # Padding length for center=True
        edge_frame_count = max(1, int(np.ceil(pad_length / self.hop_length)))
        
        # Ensure we don't exceed half the total frames
        first_edge_count = min(edge_frame_count, max(1, n_frames // 2))
        last_edge_count = min(edge_frame_count, max(1, n_frames // 2))
        
        return first_edge_count, last_edge_count

    def _calculate_edge_frame_weights(self, n_frames: int, n_fft: int) -> np.ndarray:
        """
        Calculate correction weights for each frame to compensate for STFT edge effects.
        
        Mathematical basis (standard reference):
        - With center=True, padding length = n_fft // 2
        - First frame: effective coverage = pad_length / n_fft = 0.5 (50%)
        - Energy reduction factor = sqrt(0.5) ≈ 0.707 (for amplitude)
        - Correction factor needed = 1 / 0.707 ≈ 1.414
        
        For frame i at distance d from signal start/end:
        - effective_coverage = max(pad_length - d, pad_length * 0.5) / pad_length
        - correction = 1 / max(effective_coverage, 0.5)
        - Weight = min(correction, 2.0) to avoid over-correction
        
        Args:
            n_frames: Total number of STFT frames
            n_fft: FFT size (including zero padding)
            
        Returns:
            weights: Array of correction weights (typically 0.7 to 2.0) for each frame
        """
        pad_length = n_fft // 2
        first_edge_count, last_edge_count = self._calculate_edge_frame_counts(n_frames, n_fft)
        
        weights = np.ones(n_frames, dtype=float)
        
        # Calculate weights for first edge frames
        # standard reference formula:
        # Frame i at distance d = i * hop_length from signal start
        # - Frame covers samples [d, d + n_fft] in padded signal
        # - Real signal starts at pad_length in padded signal
        # - Real signal portion = (d + n_fft - pad_length) / n_fft for d < pad_length
        # - Correction factor = 1 / real_signal_portion
        for i in range(first_edge_count):
            distance = i * self.hop_length
            # Calculate how much of the frame window actually covers real signal
            if distance < pad_length:
                # Frame still partially overlaps with reflected padding
                # Real signal portion: portion of frame that covers [pad_length, d + n_fft]
                real_signal_samples = max(0.0, distance + n_fft - pad_length)
                real_signal_portion = real_signal_samples / n_fft
                # Ensure minimum coverage of 0.5 to avoid extreme correction factors
                effective_coverage = max(real_signal_portion, 0.5)
            else:
                # Frame is fully in real signal region
                effective_coverage = 1.0
            
            # Correction factor: inverse of effective coverage
            # Frame 0: coverage = 0.5 → correction = 2.0x (restores 50% energy loss)
            correction = 1.0 / effective_coverage
            # Cap at 2.0x to prevent over-correction and numerical instability
            weights[i] = min(correction, 2.0)
        
        # Calculate weights for last edge frames (similar logic for signal end)
        # Last frames have reflected padding at the end of the signal
        for i in range(n_frames - last_edge_count, n_frames):
            # Distance from signal end: how many frames from the last frame
            frame_index_from_end = n_frames - 1 - i
            distance_from_end = frame_index_from_end * self.hop_length
            
            # Similar calculation: how much of the frame covers real signal (not end padding)
            # For end frames, we need to account for padding at signal end
            # The last pad_length samples are reflected padding
            if distance_from_end < pad_length:
                # Frame partially overlaps with end padding
                # Real signal portion: portion before padding starts
                # Assuming signal ends at some length L, padding is in [L-pad_length, L]
                # Frame i covers [distance_i, distance_i + n_fft]
                # Real signal portion = (n_fft - overlap_with_end_padding) / n_fft
                overlap_with_end_padding = pad_length - distance_from_end
                real_signal_samples = max(0.0, n_fft - overlap_with_end_padding)
                real_signal_portion = real_signal_samples / n_fft
                effective_coverage = max(real_signal_portion, 0.5)
            else:
                # Frame is fully in real signal region (away from end)
                effective_coverage = 1.0
            
            correction = 1.0 / effective_coverage
            weights[i] = min(correction, 2.0)
        
        return weights

    # ----------------- lista completa -----------------
    def generate_complete_list(self) -> None:
        if self.db_S is None or self.freqs is None:
            raise ValueError("Run fft_analysis() before generate_complete_list().")

        self.logger.info("Generating complete partial list")
        start = time.time()
        complete_list = []

        # função de agregação temporal
        agg_func = {"median": np.median, "max": np.max}.get(self.time_avg, np.mean)

        # garantir que self.db_S é array NumPy
        db_S = np.asarray(self.db_S, dtype=float)

        for i, f in enumerate(self.freqs):
            if f <= 0:
                continue

            try:
                # dB de amplitude -> amplitude linear
                amp_t = np.power(10.0, db_S[i] / 20.0)

                # limpeza numérica
                amp_t = np.nan_to_num(amp_t, nan=0.0, posinf=0.0, neginf=0.0)

                # ---------- EDGE FRAME CORRECTION: Apply weights to compensate for STFT edge effects ----------
                # Edge frames (first and last) have reduced effective signal coverage due to reflected padding
                # Apply correction weights to restore energy before temporal aggregation
                if hasattr(self, 'frame_weights') and self.frame_weights is not None:
                    if len(self.frame_weights) == len(amp_t):
                        # Apply weights to each frame: weighted_amp = amp * weight
                        # This compensates for energy reduction in edge frames
                        amp_t_weighted = amp_t * self.frame_weights
                        # Use weighted amplitudes for aggregation
                        amp_t = amp_t_weighted
                        # Log application for first frequency bin only (verification)
                        if i == 0 and not getattr(self, '_edge_correction_applied_logged', False):
                            first_frame_amp_before = amp_t[0] / self.frame_weights[0]  # Original (before correction)
                            first_frame_amp_after = amp_t[0]  # After correction
                            self.logger.info(
                                f"Edge frame correction applied: first frame amp {first_frame_amp_before:.6e} -> "
                                f"{first_frame_amp_after:.6e} (weight={self.frame_weights[0]:.2f}x, "
                                f"increase={first_frame_amp_after/first_frame_amp_before:.2f}x)"
                            )
                            self._edge_correction_applied_logged = True
                    else:
                        # Frame weights not matching - log warning but continue without correction
                        if i == 0:  # Only log once per frequency
                            self.logger.warning(
                                f"Frame weights shape mismatch: {len(self.frame_weights)} != {len(amp_t)}. "
                                f"Edge frame correction not applied."
                            )
                # ---------------------------------------------------------------------------------------------

                # agregação temporal (mean/median/max)
                amp_lin = float(agg_func(amp_t))

                # PHASE 3: Use constant instead of magic number
                # evitar log(0)
                amp_lin = max(amp_lin, EPSILON_AMPLITUDE)

                # amplitude -> dB de amplitude
                mag_db = 20.0 * np.log10(amp_lin)

                note_str = frequency_to_note_name(f)
                complete_list.append((f, mag_db, note_str))

            except Exception as e:
                self.logger.warning(f"generate_complete_list failed (i={i}, f={f:.2f} Hz): {e}")
                continue

        self.complete_list_df = pd.DataFrame(
            complete_list,
            columns=["Frequency (Hz)", "Magnitude (dB)", "Note"]
        )

        self.logger.info(f"Complete list: {len(complete_list)} partials in {time.time()-start:.3f}s")

    # ----------------- pipeline principal (GUI) -----------------
    def apply_filters_and_generate_data(
        self, *,
        freq_min: float = 200.0,
        freq_max: float = 8000.0,
        db_min: float = -80.0,
        db_max: float = 0.0,
        n_fft: int = DEFAULT_N_FFT,
        hop_length: Optional[int] = None,
        window: str = DEFAULT_WINDOW,
        tolerance: float = 0.02,
        use_adaptive_tolerance: bool = True,
        results_directory: Union[str, Path] = "./results",
        dissonance_enabled: bool = True,
        dissonance_model: str = "Sethares",
        dissonance_curve: bool = True,
        dissonance_scale: bool = True,
        compare_models: bool = False,
        harmonic_weight: float = 0.95,  # Default: 95% (alinhado com interface)
        inharmonic_weight: float = 0.05,  # Default: 5% (alinhado com interface)
        auto_model_weights_from_analysis: bool = True,  # SINGLE-PASS REFACTOR — default canonical path: derive H/(H+I) and I/(H+I) from the current spectral analysis itself, not from a pre-computed Batch.
        weight_function: str = "linear",
        zero_padding: int = 1,
        time_avg: str = "mean",
        density_summation_mode: str = "his_weighted",
        harmonic_density_weight: float = 1.0,
        inharmonic_density_weight: float = 0.5,
        subbass_density_weight: float = 0.25,
        density_salience_threshold_db: float = -45.0,
        density_frequency_ceiling_hz: float = 5000.0,
        spectral_masking_enabled: bool = False,  # NEW: Control spectral masking (default: OFF for physical model)
        spectral_magnitude_smoothing_enabled: bool = DEFAULT_STFT_MAGNITUDE_SMOOTHING_ENABLED,
        parallel_processing: bool = False,
        export_data_format: str = "json",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        use_tsne: bool = False,  # NEW: t-SNE dimensionality reduction
        use_umap: bool = False,  # NEW: UMAP dimensionality reduction
        detect_anomalies: bool = False,  # NEW: Anomaly detection
        anomaly_contamination: Optional[float] = None,  # NEW: Adaptive if None
        tier: Optional[str] = None,  # NEW: Tier name for tracking processing tier
        # AUDIT FIX (per-file Stage 2 leakage / Clarinete_mf "jumped off" bug):
        # The per-file ``_compile_metrics`` call at the end of this method
        # was designed for the standalone use case where ``results_directory``
        # already holds a fully-populated batch of per-note workbooks.
        # When this method is invoked from the GUI worker or the CLI
        # orchestrator on a SINGLE audio file, ``results_directory`` is the
        # per-note folder and the compile becomes (a) wasted PCA work over
        # one row, (b) a write of an unwanted ``compiled_density_metrics*.xlsx``
        # next to each ``spectral_analysis.xlsx``, and (c) an extra failure
        # surface that can abort the worker before the corpus-wide compile
        # runs. Callers that perform their own corpus-wide Stage 2 must pass
        # ``compile_per_call=False`` to suppress this.
        compile_per_call: bool = True,
        **kwargs: object,
    ) -> None:
        # Store main parameters
        self.window = str(window)
        self.n_fft = int(n_fft)
        # FIXED: Preserve exact hop_length value passed (critical for tier-specific settings)
        if hop_length is not None:
            self.hop_length = int(hop_length)
            self.logger.debug(f"AudioProcessor: Set hop_length={self.hop_length} from parameter (n_fft={self.n_fft})")
        else:
            self.hop_length = self.n_fft // 2
            self.logger.debug(f"AudioProcessor: Set hop_length={self.hop_length} from default (n_fft={self.n_fft})")
        self.tolerance = float(tolerance) if isinstance(tolerance, (int, float)) else tolerance
        self.use_adaptive_tolerance = bool(use_adaptive_tolerance)
        self.results_directory = Path(results_directory)
        self.freq_min = float(freq_min)
        self.freq_max = float(freq_max)
        self.db_min = float(db_min)
        self.db_max = float(db_max)
        self.dissonance_enabled = bool(dissonance_enabled)
        self.dissonance_model = str(dissonance_model)
        self.dissonance_curve_enabled = bool(dissonance_curve)
        self.dissonance_scale_enabled = bool(dissonance_scale)
        self.dissonance_compare_models = bool(compare_models)
        self.harmonic_weight = float(harmonic_weight)
        self.inharmonic_weight = float(inharmonic_weight)
        # SINGLE-PASS REFACTOR — when True (the new canonical default), proc_audio
        # becomes the single source of truth for harmonic/inharmonic/sub-bass
        # energy and overrides ``harmonic_weight`` / ``inharmonic_weight`` after
        # spectral classification using the H/(H+I), I/(H+I) coefficients derived
        # from the current analysis (see _set_model_weights_from_current_component_energy).
        self.auto_model_weights_from_analysis = bool(auto_model_weights_from_analysis)
        self.tier = str(tier) if tier is not None else None  # NEW: Store tier name
        self.weight_function = str(weight_function)
        self.spectral_masking_enabled = bool(spectral_masking_enabled)  # NEW: Store spectral masking flag
        self.spectral_magnitude_smoothing_enabled = bool(spectral_magnitude_smoothing_enabled)

        # Critical validation
        self._validate_and_store_parameters(
            n_fft, hop_length, window, weight_function,
            harmonic_weight, inharmonic_weight,
            dissonance_enabled, dissonance_model, dissonance_curve, dissonance_scale, compare_models
        )

        # STFT parameters: zero padding and time averaging are standard STFT options
        self.zero_padding = int(zero_padding)
        self.time_avg = str(time_avg)
        if self.time_avg not in {"mean", "median", "max"}:
            self.time_avg = "mean"
        self.density_summation_mode = str(density_summation_mode or "his_weighted").strip().lower()
        self.harmonic_density_weight = float(harmonic_density_weight)
        self.inharmonic_density_weight = float(inharmonic_density_weight)
        self.subbass_density_weight = float(subbass_density_weight)
        self.density_salience_threshold_db = float(density_salience_threshold_db)
        self.density_frequency_ceiling_hz = float(density_frequency_ceiling_hz)

        # Directories
        results_directory = self.results_directory
        interactive_dir = results_directory / "interactive_visualizations"
        results_directory.mkdir(parents=True, exist_ok=True)
        interactive_dir.mkdir(parents=True, exist_ok=True)

        # Tolerance sanity check
        if not (0 < float(self.tolerance) < 100):
            self.logger.warning(f"Tolerance {self.tolerance} outside recommended range (0-100 Hz).")

        # Pre-analysis log. When auto_model_weights_from_analysis is True
        # (canonical default), the supplied harmonic_weight / inharmonic_weight
        # are neutral placeholders that proc_audio overwrites after spectral
        # classification using the current-analysis component energies.
        if bool(getattr(self, "auto_model_weights_from_analysis", True)):
            self.logger.info(
                "Initial neutral/placeholder model weights supplied for API "
                "compatibility (harmonic_weight=%.4f, inharmonic_weight=%.4f). "
                "Final model weights will be derived from the current per-note "
                "spectral analysis.",
                float(self.harmonic_weight),
                float(self.inharmonic_weight),
            )
        else:
            self.logger.info(
                "Externally supplied final model weights "
                "(harmonic_weight=%.4f, inharmonic_weight=%.4f). "
                "Component ratios remain sourced from the current analysis.",
                float(self.harmonic_weight),
                float(self.inharmonic_weight),
            )

        # Process files (sequential or parallel based on parallel_processing flag)
        total_files = len(self.audio_data)
        self.logger.info(f"Processing {total_files} file(s) ...")
        
        if parallel_processing and total_files > 1:
            self.logger.info("Using parallel processing mode")
            self._parallel_process_audio_files(
                self.freq_min, self.freq_max, self.db_min, self.db_max, self.tolerance,
                results_directory, interactive_dir, export_data_format, progress_callback,
                zero_padding=self.zero_padding, time_avg=self.time_avg,
                use_tsne=use_tsne,
                use_umap=use_umap,
                detect_anomalies=detect_anomalies,
                anomaly_contamination=anomaly_contamination,
            )
        else:
            self.logger.info("Using sequential processing mode")
            self._sequential_process_audio_files(
                self.freq_min, self.freq_max, self.db_min, self.db_max, self.tolerance,
                results_directory, interactive_dir, export_data_format, progress_callback,
                zero_padding=self.zero_padding, time_avg=self.time_avg
            )

        # Compile metrics and export combined results.
        #
        # AUDIT FIX (per-file Stage 2 leakage): only run the per-call
        # ``_compile_metrics`` when the caller has not explicitly opted out
        # via ``compile_per_call=False``. The GUI worker
        # (``pipeline_orchestrator_gui``) and the CLI orchestrator
        # (``pipeline_orchestrator_integrated``) both perform their own
        # corpus-wide compile after looping over all audio files, so the
        # per-call compile would write ``compiled_density_metrics*.xlsx``
        # files inside every per-note folder (wasted I/O, confusing
        # output, and an unnecessary failure point that can drop the
        # worker before the corpus-wide compile runs).
        if total_files > 0 and bool(compile_per_call):
            # SINGLE-PASS REFACTOR — when auto_model_weights_from_analysis is
            # True, _set_model_weights_from_current_component_energy() has
            # already overridden self.harmonic_weight / self.inharmonic_weight
            # during _calculate_metrics. Forwarding the *original* function
            # args here would leak placeholder weights (e.g. 0.5/0.5) into the
            # compiled output, defeating the refactor. Use the live
            # attributes instead.
            _hw_for_compile = (
                float(self.harmonic_weight)
                if bool(getattr(self, "auto_model_weights_from_analysis", True))
                else float(harmonic_weight)
            )
            _ihw_for_compile = (
                float(self.inharmonic_weight)
                if bool(getattr(self, "auto_model_weights_from_analysis", True))
                else float(inharmonic_weight)
            )
            try:
                self._compile_metrics(
                    results_directory,
                    use_tsne=use_tsne,
                    use_umap=use_umap,
                    detect_anomalies=detect_anomalies,
                    anomaly_contamination=anomaly_contamination,
                    harmonic_weight=_hw_for_compile,
                    inharmonic_weight=_ihw_for_compile,
                    weight_function=weight_function
                )
            except Exception as e:
                self.logger.error(f"Metrics compilation failed: {e}")
            try:
                self._export_combined_data_for_visualization(results_directory, interactive_dir, export_data_format)
            except Exception as e:
                self.logger.error(f"Combined data export failed: {e}")
        elif total_files > 0:
            self.logger.info(
                "Per-call Stage 2 compile suppressed (compile_per_call=False); "
                "caller is expected to run a corpus-wide compile after the "
                "per-note loop."
            )

    def _validate_and_store_parameters(
        self,
        n_fft: int,
        hop_length: Optional[int],
        window: str,
        weight_function: str,
        harmonic_weight: float,
        inharmonic_weight: float,
        dissonance_enabled: bool,
        dissonance_model: str,
        dissonance_curve: bool,
        dissonance_scale: bool,
        compare_models: bool
    ) -> None:
        if n_fft <= 0:
            raise ValueError("n_fft must be positive.")
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4

        valid_windows = [
            'hann','hamming','bartlett','blackmanharris','flattop',
            'bohman','kaiser','gaussian','gauss','gaussiana'
        ]

        name = (window or '').lower()
        if name not in valid_windows:
            self.logger.warning(f"Window '{window}' may not be supported. Recommended: {valid_windows}")
        self.window = name

        weight_name = (weight_function or "linear").strip().lower()
        _ = get_weight_function(weight_name)  # validate
        self.weight_function = weight_name

        try:
            hw = float(harmonic_weight)
            ihw = float(inharmonic_weight)
        except Exception:
            hw, ihw = 1.0, 0.0
        if hw < 0 or ihw < 0:
            hw = max(hw, 0.0)
            ihw = max(ihw, 0.0)
        total = hw + ihw
        if total == 0:
            hw, ihw = 1.0, 0.0
        elif not np.isclose(total, 1.0, atol=1e-5):
            hw /= total
            ihw /= total
        self.harmonic_weight, self.inharmonic_weight = hw, ihw

        self.dissonance_enabled = bool(dissonance_enabled)
        if self.dissonance_enabled:
            available = list_available_models()
            wanted = str(dissonance_model).strip()
            chosen = next((m for m in available if m.lower() == wanted.lower()), None)
            if not chosen:
                raise ValueError(f"Unknown dissonance model: {wanted}. Available: {available}")
            self.dissonance_model = chosen
            self.dissonance_curve_enabled = bool(dissonance_curve)
            self.dissonance_scale_enabled = bool(dissonance_scale)
            self.dissonance_compare_models = bool(compare_models)

        self.logger.debug("Parameters validated.")

    def _sequential_process_audio_files(
        self,
        freq_min: float,
        freq_max: float,
        db_min: float,
        db_max: float,
        tolerance: float,
        results_directory: Path,
        interactive_dir: Path,
        export_data_format: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        zero_padding: int = 1,
        time_avg: str = 'mean'
    ) -> None:
        import time
        import gc  # Important for memory cleanup in large loops

        start = time.time()

        for i, (y, sr, note, file_path) in enumerate(self.audio_data, 1):
            try:
                # 1. Notify progress in interface
                if progress_callback:
                    progress_callback(i, len(self.audio_data), note)

                # 2. Load data into class state
                self.y, self.sr = y, sr
                self._reset_metrics()
                
                # CRITICAL: Reset all DataFrames to prevent state persistence
                # This ensures each file starts with a clean state
                self.complete_list_df = None
                self.filtered_list_df = None
                self.harmonic_list_df = None
                self.S = None
                self.db_S = None
                
                # CRITICAL: Reset amplitude correction flags to prevent state persistence
                # These flags control whether coherent gain is applied, and if not reset,
                # the first file gets different treatment than subsequent files
                self._filtered_amp_corrected = False
                self._harmonic_amp_corrected = False
                self._complete_amp_corrected = False
                self.freqs = None
                self.times = None
                
                # CRITICAL: Reset frame weights to ensure each file gets fresh edge frame calculation
                # Frame weights are recalculated in fft_analysis() based on actual signal length
                self.frame_weights = None
                self._edge_correction_applied_logged = False  # Reset logging flag for each file

                # 3. Execute Analysis (FFT)
                self.fft_analysis(zero_padding=zero_padding)

                # 4. Generate list of partials
                self.generate_complete_list()

                # 5. Process Harmonics and Calculate Metrics
                # (Log/Linear correction acts here)
                self._process_filtered_and_harmonic_data(
                    freq_min, freq_max, db_min, db_max, tolerance, note,
                    zero_padding=zero_padding, time_avg=time_avg
                )

                # 6. Save Results
                out_folder = results_directory / note
                out_folder.mkdir(parents=True, exist_ok=True)

                self.save_results(out_folder, note)
                self._export_data_for_visualization(note, out_folder, interactive_dir, export_data_format)

                self.logger.info(f"{i}/{len(self.audio_data)} processed: {note}")

                # 7. Memory cleanup (Critical for batch processing)
                # Release matplotlib graphics memory and large arrays
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                except Exception:
                    pass
                gc.collect()

            except Exception as e:
                self.logger.error(f"Error processing {note}: {e}")
                continue

        self.logger.info(f"Processing completed in {time.time()-start:.2f}s")

    def _parallel_process_audio_files(
        self,
        freq_min: float,
        freq_max: float,
        db_min: float,
        db_max: float,
        tolerance: float,
        results_directory: Path,
        interactive_dir: Path,
        export_data_format: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        zero_padding: int = 1,
        time_avg: str = 'mean',
        *,
        use_tsne: bool = False,
        use_umap: bool = False,
        detect_anomalies: bool = False,
        anomaly_contamination: Optional[float] = None,
    ) -> None:
        """
        Parallel processing of audio files using multiprocessing.
        
        Mathematical verification (reference):
        - Speedup: S = T_sequential / T_parallel
        - Ideal: S = N_cores (linear speedup)
        - Actual: S = N_cores × efficiency (70-80% typical due to overhead)
        - For 8 cores: S ≈ 5.6-6.4x (realistic expectation)
        
        Args:
            freq_min: Minimum frequency (Hz)
            freq_max: Maximum frequency (Hz)
            db_min: Minimum magnitude (dB)
            db_max: Maximum magnitude (dB)
            tolerance: Harmonic tolerance (Hz)
            results_directory: Output directory for results
            interactive_dir: Directory for interactive visualizations
            export_data_format: Export format ('json', 'csv', etc.)
            progress_callback: Optional callback for progress updates
            zero_padding: Zero padding factor
            time_avg: Time averaging method ('mean', 'median', 'max')
        """
        import time
        from multiprocessing import Pool, cpu_count
        from functools import partial
        
        start = time.time()
        total_files = len(self.audio_data)
        
        # Determine number of workers (leave 1 core free for system)
        max_workers = max(1, cpu_count() - 1)
        num_workers = min(max_workers, total_files)
        
        self.logger.info(f"Parallel processing: {num_workers} workers for {total_files} files")
        
        # Prepare parameters for worker function
        worker_params = {
            'freq_min': freq_min,
            'freq_max': freq_max,
            'db_min': db_min,
            'db_max': db_max,
            'tolerance': tolerance,
            'results_directory': str(results_directory),
            'interactive_dir': str(interactive_dir),
            'export_data_format': export_data_format,
            'zero_padding': zero_padding,
            'time_avg': time_avg,
            # Pass all class parameters needed for processing
            'n_fft': self.n_fft,
            'hop_length': self.hop_length,
            'window': self.window,
            'weight_function': self.weight_function,
            'harmonic_weight': self.harmonic_weight,
            'inharmonic_weight': self.inharmonic_weight,
            'dissonance_enabled': self.dissonance_enabled,
            'dissonance_model': self.dissonance_model,
            'dissonance_curve_enabled': self.dissonance_curve_enabled,
            'dissonance_scale_enabled': self.dissonance_scale_enabled,
            'dissonance_compare_models': self.dissonance_compare_models,
            'use_adaptive_tolerance': self.use_adaptive_tolerance,
            'spectral_masking_enabled': self.spectral_masking_enabled,
            'tier': self.tier,
            'use_tsne': use_tsne,
            'use_umap': use_umap,
            'detect_anomalies': detect_anomalies,
            'anomaly_contamination': anomaly_contamination,
            'spectral_magnitude_smoothing_enabled': bool(
                getattr(self, "spectral_magnitude_smoothing_enabled", False)
            ),
        }
        
        # Import cache module
        try:
            from result_cache import get_cache
            cache = get_cache()
            use_cache = True
        except ImportError:
            cache = None
            use_cache = False
            self.logger.warning("Cache module not available, proceeding without caching")
        
        # Prepare file data for workers
        file_data_list = [
            (y, sr, note, file_path) 
            for y, sr, note, file_path in self.audio_data
        ]
        
        # Process files in parallel
        args_list = [(data, worker_params) for data in file_data_list]
        
        results = []
        with Pool(processes=num_workers) as pool:
            # Use imap for progress tracking
            completed = 0
            for result in pool.imap(_process_file_worker_parallel, args_list):
                completed += 1
                results.append(result)
                
                # Update progress
                if progress_callback and result['success']:
                    progress_callback(completed, total_files, result['note'])
                
                if result['success']:
                    cache_status = " (cached)" if result.get('cached', False) else ""
                    self.logger.info(f"{completed}/{total_files} processed: {result['note']}{cache_status}")
                else:
                    self.logger.error(f"Error: {result.get('error', 'Unknown error')}")
        
        # Log statistics
        successful = sum(1 for r in results if r['success'])
        cached = sum(1 for r in results if r.get('cached', False))
        elapsed = time.time() - start
        
        self.logger.info(f"Parallel processing completed in {elapsed:.2f}s")
        self.logger.info(f"Success: {successful}/{total_files}, Cached: {cached}/{total_files}")
        
        if use_cache:
            cache_stats = cache.get_stats()
            self.logger.info(f"Cache statistics: {cache_stats}")

    def _process_filtered_and_harmonic_data(
        self,
        freq_min: float, freq_max: float,
        db_min: float, db_max: float,
        tolerance: float, note: str,
        zero_padding: int = 1, time_avg: str = 'mean'  # STFT parameters: zero padding and time averaging
    ) -> None:
        """
        Filtra a lista completa de parciais, aplica a correção de Ganho Coerente (Gc),
        gera a lista de harmónicos e inicia o cálculo de métricas (FFT).
        """
        import numpy as np
        import pandas as pd

        self.note = note
        if self.complete_list_df is None or self.complete_list_df.empty:
            self.logger.error("Complete list not generated.")
            self.filtered_list_df = pd.DataFrame()
            self.harmonic_list_df = pd.DataFrame()
            return

        # ===== 1. BLOCO DE FILTROS E CORREÇÃO DE AMPLITUDE =====

        # Aplica filtros de frequência e magnitude (dB)
        fdf = self.complete_list_df[
            (self.complete_list_df["Frequency (Hz)"] >= freq_min) &
            (self.complete_list_df["Frequency (Hz)"] <= freq_max) &
            (self.complete_list_df["Magnitude (dB)"] >= db_min) &
            (self.complete_list_df["Magnitude (dB)"] <= db_max)
        ].copy()

        # ---------- NOVO: fallback se filtros esvaziam ----------
        if fdf.empty:
            self.logger.warning(
                f"No components within filters for {note}. "
                f"Trying to include the bin closest to f0."
            )

            # try to obtain f0 (note label or Hz)
            try:
                f0 = float(self.calculate_fundamental_frequency(note) or 0.0)
            except Exception:
                f0 = 0.0

            if (
                f0 > 0.0 and
                self.complete_list_df is not None and
                not self.complete_list_df.empty and
                "Frequency (Hz)" in self.complete_list_df.columns and
                "Magnitude (dB)" in self.complete_list_df.columns
            ):
                try:
                    # pick the bin closest to f0 (no filters)
                    idx = (self.complete_list_df["Frequency (Hz)"] - f0).abs().idxmin()
                    row = self.complete_list_df.loc[[idx]].copy()

                    # convert dB -> amplitude and correct coherent gain (same rule as elsewhere)
                    amps = np.power(10.0, row["Magnitude (dB)"].to_numpy(float) / 20.0)

                    cg = float(getattr(self, "coherent_gain_value", 1.0) or 1.0)
                    if cg <= 0.0:
                        cg = 1.0

                    row["Amplitude"] = amps / cg
                    self._filtered_amp_corrected = True

                    # use this fallback as the minimal filtered list
                    fdf = row

                    self.logger.info(
                        f"Fallback applied: reinserted 1 component near f0={f0:.3f} Hz "
                        f"(bin={float(fdf['Frequency (Hz)'].iloc[0]):.6f} Hz)."
                    )
                except Exception as e:
                    self.logger.warning(f"f0 fallback failed: {e}")
                    self.filtered_list_df = pd.DataFrame()
                    self.harmonic_list_df = pd.DataFrame()
                    self._set_default_metrics()
                    return
            else:
                # sem f0 ou sem complete_list_df: mantém comportamento antigo
                self.filtered_list_df = pd.DataFrame()
                self.harmonic_list_df = pd.DataFrame()
                self._set_default_metrics()
                return
        # ---------- FIM NOVO ----------

        # dB (Magnitude) -> Amplitude Linear (apenas se ainda não existir amplitude)
        if "Amplitude" in fdf.columns:
            # garantir float
            fdf["Amplitude"] = fdf["Amplitude"].astype(float)
        else:
            amps = np.power(10.0, fdf["Magnitude (dB)"].to_numpy(float) / 20.0)

            # Coherent Gain (Gc) Correction
            cg = float(getattr(self, "coherent_gain_value", 1.0) or 1.0)
            if cg <= 0.0:
                cg = 1.0

            fdf["Amplitude"] = amps / cg
            self._filtered_amp_corrected = True  # filtered_list_df já está corrigida por ganho coerente

        # Limpeza de valores extremos
        fdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        fdf.dropna(subset=["Amplitude"], inplace=True)

        # Ordenar por amplitude desc (mantém coerência)
        fdf = fdf.sort_values(by="Amplitude", ascending=False).reset_index(drop=True)

        self.filtered_list_df = fdf
        self.logger.info(f"Filtered list: {len(fdf)} partials")

        # ===== 2. GERAÇÃO DE HARMÓNICOS =====
        self._generate_harmonic_list(
            note, freq_max, tolerance,
            use_adaptive_tolerance=getattr(self, "use_adaptive_tolerance", True)
        )
        self._apply_density_relevant_frequency_floor_to_filtered_list()

        # Inicialização de métricas (Reset)
        self.density_metric_value = None
        self.scaled_density_metric_value = None
        self.entropy_spectral_value = None
        self.combined_density_metric_value = None
        self.spectral_density_metric_value = None
        self.total_metric_value = None
        # ROBUSTNESS FIX: Initialize new metrics
        self.density_metric_per_harmonic = None
        self.density_metric_normalized = None
        self.canonical_density_v5_adapted = None
        self.density_per_component = None
        self.density_formula_version = None
        self.density_source_formula = None
        self.density_normalization_scope = None
        self.density_normalization_denominator = None
        self.density_metric_ratio_over_fundamental_legacy = None
        self.discrete_metric_d3 = None
        self.discrete_metric_d10 = None
        self.discrete_metric_d17 = None
        self.discrete_metric_d24 = None
        self.effective_partial_density = None
        self.partial_density_effective_components = None
        self.harmonic_energy_sum = None
        self.inharmonic_energy_sum = None
        self.subbass_energy_sum = None
        self.total_component_energy = None
        self.harmonic_energy_ratio = None
        self.inharmonic_energy_ratio = None
        self.subbass_energy_ratio = None
        self.linear_sum_amplitude_harmonic = None
        self.linear_sum_amplitude_inharmonic_partial = None
        self.linear_sum_amplitude_subbass_band = None
        self.linear_amplitude_fraction_inharmonic_of_HI = None
        self.linear_amplitude_fraction_nonharmonic_of_total = None
        self.linear_amplitude_batch_alignment_factor = None
        self.harmonic_partial_count = None
        self.inharmonic_partial_count = None
        self.total_detected_partial_count = None
        self.unique_harmonic_order_count = None
        self.harmonic_order_count = None
        self.harmonic_peak_count = None
        self.inharmonic_peak_count = None
        self.subbass_peak_count = None
        self.total_detected_peak_count = None
        self.harmonic_peak_candidate_count = None
        self.nonharmonic_peak_candidate_count = None
        self.low_frequency_peak_candidate_count = None
        self.total_peak_candidate_count = None
        self.residual_spectral_row_count = None
        self.nonharmonic_candidate_row_count = None
        self.retained_nonharmonic_peak_candidate_count = None
        self.exported_nonharmonic_peak_candidate_count = None
        self.peaklist_harmonic_window_candidate_count = None
        self.peaklist_nonharmonic_window_candidate_count = None
        self.peaklist_low_frequency_window_candidate_count = None
        self.peaklist_total_window_candidate_count = None
        self.debug_counts_invariant_status = ""
        self.debug_counts_invariant_failures = ""
        self.accepted_inharmonic_peak_count = None
        self.accepted_inharmonic_partial_count = None
        self.harmonic_candidate_count = None
        self.inharmonic_candidate_count = None
        self.subbass_candidate_count = None
        self.total_spectral_candidate_count = None
        self.residual_row_count = None
        self.harmonic_bin_count = None
        self.inharmonic_bin_count = None
        self.subbass_bin_count = None
        self.energy_conservation_status = None
        self.energy_conservation_error = None
        self.energy_denominator_description = None
        self.dissonance_partial_count = None
        self.dissonance_pair_count = None
        self.harmonic_validation_report = None
        self.f0_prior_available = None
        self.f0_blind_method = None

        self.component_energy_status = "not_computed"
        self.component_energy_pie_basis = "not_written"
        self.amplitude_mass_chart_file = ""
        self.energy_ratio_chart_file = ""
        self.amplitude_mass_chart_status = "not_attempted"
        self.energy_ratio_chart_status = "not_attempted"
        self.component_energy_pie_file = ""
        self.component_energy_pie_alias_basis = ""
        self.effective_partial_density_status = "not_computed"
        self.density_metric_status = "not_computed"
        self.normalization_status = "not_computed"
        self.debug_counts_status = "not_computed"
        self.model_weight_status = "not_computed"
        self.model_weight_fallback_applied = False

        # ===== 3. CÁLCULO DE MÉTRICAS (CHAMADA CRÍTICA) =====
        self._calculate_metrics()

        # ===== 4. VERIFICAÇÃO FINAL DE DM (Fallback) =====
        # Este bloco só actua se DM não foi calculado (None/NaN), não por ser 0.
        dm_invalid = (
            self.scaled_density_metric_value is None
            or (isinstance(self.scaled_density_metric_value, float) and not np.isfinite(self.scaled_density_metric_value))
        )

        if dm_invalid:
            self.logger.warning(
                "Density metric missing (None/NaN). Attempting fallback recomputation."
            )
            try:
                # Escolher a melhor fonte de amplitudes disponíveis:
                # 1) harmónicos; 2) filtrada; 3) completa (último recurso)
                src_df = None
                if getattr(self, "harmonic_list_df", None) is not None and not self.harmonic_list_df.empty:
                    src_df = self.harmonic_list_df
                elif getattr(self, "filtered_list_df", None) is not None and not self.filtered_list_df.empty:
                    src_df = self.filtered_list_df
                elif getattr(self, "complete_list_df", None) is not None and not self.complete_list_df.empty:
                    src_df = self.complete_list_df

                if src_df is None:
                    self.density_metric_value = 0.0
                    self.scaled_density_metric_value = 0.0
                else:
                    # Garantir Amplitude linear (SEM reaplicar coherent gain se já foi aplicado)
                    if "Amplitude" in src_df.columns:
                        amps = src_df["Amplitude"].to_numpy(float)
                    elif "Magnitude (dB)" in src_df.columns:
                        amps = np.power(10.0, src_df["Magnitude (dB)"].to_numpy(float) / 20.0)

                        # aplicar coherent gain apenas se ainda não corrigiu amplitudes antes
                        cg = float(getattr(self, "coherent_gain_value", 1.0) or 1.0)
                        if cg <= 0.0:
                            cg = 1.0
                        amps = amps / cg
                    else:
                        amps = np.asarray([], dtype=float)

                    # limpar amps
                    amps = np.asarray(amps, dtype=float)
                    amps = amps[np.isfinite(amps)]
                    amps = amps[amps > 0.0]

                    # caso limite: 0 ou 1 componente -> DM definido por convenção (não "falha")
                    if amps.size <= 1:
                        self.density_metric_value = 0.0
                        self.scaled_density_metric_value = 0.0
                        self.logger.info(
                            f"Fallback DM: amps.size={amps.size} -> density metric set to 0.0 (edge case)."
                        )
                    else:
                        density = float(apply_density_metric(amps, self.weight_function, normalize=False))
                        if not np.isfinite(density):
                            density = 0.0

                        self.density_metric_value = density
                        self.scaled_density_metric_value = density
                        self.logger.info(f"Density metric recomputed (fallback): {density:.6f}")

            except Exception as e:
                self.logger.error(f"Fallback density-metric recomputation failed: {e}", exc_info=True)
                self.density_metric_value = 0.0
                self.scaled_density_metric_value = 0.0



    def _reset_metrics(self) -> None:
        self.density_metric_value = None
        self.scaled_density_metric_value = None
        self.filtered_density_metric_value = None
        self.entropy_spectral_value = None
        self.combined_density_metric_value = None
        self.total_metric_value = None
        self.spectral_density_metric_value = None
        self.effective_partial_density = None
        self.partial_density_effective_components = None
        self.harmonic_energy_sum = None
        self.inharmonic_energy_sum = None
        self.subbass_energy_sum = None
        self.total_component_energy = None
        self.harmonic_energy_ratio = None
        self.inharmonic_energy_ratio = None
        self.subbass_energy_ratio = None
        self.linear_sum_amplitude_harmonic = None
        self.linear_sum_amplitude_inharmonic_partial = None
        self.linear_sum_amplitude_subbass_band = None
        self.linear_amplitude_fraction_inharmonic_of_HI = None
        self.linear_amplitude_fraction_nonharmonic_of_total = None
        self.linear_amplitude_batch_alignment_factor = None
        self.harmonic_partial_count = None
        self.inharmonic_partial_count = None
        self.total_detected_partial_count = None
        self.unique_harmonic_order_count = None
        self.harmonic_order_count = None
        self.harmonic_peak_count = None
        self.inharmonic_peak_count = None
        self.subbass_peak_count = None
        self.total_detected_peak_count = None
        self.harmonic_peak_candidate_count = None
        self.nonharmonic_peak_candidate_count = None
        self.low_frequency_peak_candidate_count = None
        self.total_peak_candidate_count = None
        self.residual_spectral_row_count = None
        self.nonharmonic_candidate_row_count = None
        self.retained_nonharmonic_peak_candidate_count = None
        self.exported_nonharmonic_peak_candidate_count = None
        self.peaklist_harmonic_window_candidate_count = None
        self.peaklist_nonharmonic_window_candidate_count = None
        self.peaklist_low_frequency_window_candidate_count = None
        self.peaklist_total_window_candidate_count = None
        self.debug_counts_invariant_status = ""
        self.debug_counts_invariant_failures = ""
        self.accepted_inharmonic_peak_count = None
        self.accepted_inharmonic_partial_count = None
        self.harmonic_candidate_count = None
        self.inharmonic_candidate_count = None
        self.subbass_candidate_count = None
        self.total_spectral_candidate_count = None
        self.residual_row_count = None
        self.harmonic_bin_count = None
        self.inharmonic_bin_count = None
        self.subbass_bin_count = None
        self.energy_conservation_status = None
        self.energy_conservation_error = None
        self.energy_denominator_description = None
        self.dissonance_partial_count = None
        self.dissonance_pair_count = None
        self.harmonic_validation_report = None
        self.f0_prior_available = None
        self.f0_blind_method = None
        self.f0_final = None
        self.f0_final_source = None
        self.f0_source = None
        self.f0_detuning_cents_from_nominal = None
        self.f0_robust_residual_std = None
        self.f0_fit_residual_std_hz = None
        self.f0_robust_accepted = None
        self.f0_final_method = None
        self.f0_fit_accepted = None
        self.f0_fit_quality = None
        self.f0_fit_rejection_reason = None
        try:
            if hasattr(self, "f0_robust"):
                delattr(self, "f0_robust")
        except Exception:
            pass
        self.density_metric_per_harmonic = None
        self.density_metric_normalized = None
        self.canonical_density_v5_adapted = None
        self.discrete_metric_d3 = None
        self.discrete_metric_d10 = None
        self.discrete_metric_d17 = None
        self.discrete_metric_d24 = None
        self.density_per_component = None
        self.density_formula_version = None
        self.density_source_formula = None
        self.density_normalization_scope = None
        self.density_normalization_denominator = None
        self.density_metric_ratio_over_fundamental_legacy = None

        self.adaptive_subfundamental_cutoff_hz = None
        self.subfundamental_margin_percent = None
        self.percentage_subfundamental_cutoff_hz = None
        self.leakage_guard_cutoff_hz = None
        self.effective_subfundamental_margin_percent = None
        self.subfundamental_cutoff_selection_rule = ""
        self.subfundamental_cutoff_selected_by = ""
        self.subfundamental_guard_valid = False
        self.subfundamental_guard_policy = "invalid_f0"
        self.low_frequency_policy_version = LOW_FREQUENCY_POLICY_VERSION
        self.physical_low_frequency_lower_hz = None
        self.physical_low_frequency_upper_hz = None
        self.low_frequency_residual_interpretation = (
            "diagnostic residual; not automatically sub-bass, "
            "not automatically noise, not partial"
        )

        self.component_energy_status = "not_computed"
        self.component_energy_pie_basis = "not_written"
        self.amplitude_mass_chart_file = ""
        self.energy_ratio_chart_file = ""
        self.amplitude_mass_chart_status = "not_attempted"
        self.energy_ratio_chart_status = "not_attempted"
        self.component_energy_pie_file = ""
        self.component_energy_pie_alias_basis = ""
        self.effective_partial_density_status = "not_computed"
        self.density_metric_status = "not_computed"
        self.normalization_status = "not_computed"
        self.debug_counts_status = "not_computed"
        self.model_weight_status = "not_computed"
        self.model_weight_fallback_applied = False

        for model_name in self.dissonance_values:
            self.dissonance_values[model_name] = None
            self.dissonance_curves[model_name] = None
            self.dissonance_scales[model_name] = None

    # ----------------- gerar lista de harmÃ³nicos -----------------
    def _generate_harmonic_list(
        self,
        note: str,
        freq_max: float,
        tolerance: float,
        use_adaptive_tolerance: bool = True
    ) -> None:
        """
        Generate harmonic list with harmonic-order grouping and local peak refinement:
        - Sub-bin interpolation (parabolic in log-magnitude)
        - Local peak validation (exclude side lobes)
        - Global f₀ estimation (robust least squares)
        - Bin spacing-based tolerances
        """

        f0_from_note = float(self.calculate_fundamental_frequency(note))
        self.f0_prior_note = str(note or "").strip()
        self.f0_prior_source = "filename_or_parsed_note"
        if np.isfinite(f0_from_note) and f0_from_note > 0.0:
            self.f0_nominal_hz = float(f0_from_note)
        else:
            self.f0_nominal_hz = float("nan")
        self.f0_prior_hz = self.f0_nominal_hz
        if np.isfinite(self.f0_nominal_hz) and self.f0_nominal_hz > 0.0:
            self.f0_initial = float(self.f0_nominal_hz)
        else:
            self.f0_initial = float("nan")

        f0 = float(f0_from_note)
        if f0 <= 0:
            self.logger.warning(f"Invalid f0 for {note}. Attempting estimate.")
            if self.filtered_list_df is not None and not self.filtered_list_df.empty:
                low = self.filtered_list_df.nsmallest(20, 'Frequency (Hz)')
                if not low.empty:
                    max_amp_idx = low['Amplitude'].idxmax()
                    f0 = float(self.filtered_list_df.loc[max_amp_idx, 'Frequency (Hz)'])
                else:
                    self.harmonic_list_df = pd.DataFrame()
                    return
            else:
                self.harmonic_list_df = pd.DataFrame()
                return

        self.logger.info(f"F0({note}) = {f0:.2f} Hz (nominal_prior_hz={self.f0_prior_hz})")

        if self.filtered_list_df is None or self.filtered_list_df.empty:
            self.harmonic_list_df = pd.DataFrame()
            return

        # Calculate bin spacing for tolerances
        # Mathematical validation (standard reference): Δf = SR / (N_FFT × ZP)
        bin_spacing = None
        has_sub_bin_interpolation = False
        if hasattr(self, 'sr') and self.sr and hasattr(self, 'n_fft') and hasattr(self, 'zero_padding'):
            bin_spacing = _calculate_bin_spacing(self.sr, self.n_fft, self.zero_padding)
            has_sub_bin_interpolation = True  # Enable sub-bin interpolation if bin spacing available
            self.logger.debug(f"Bin spacing calculated: {bin_spacing:.4f} Hz (SR={self.sr}, N_FFT={self.n_fft}, ZP={self.zero_padding})")

        # Get complete spectrum magnitudes for interpolation and validation
        # If db_S is available (time-averaged), use it; otherwise fallback to complete_list_df
        complete_magnitudes = None
        complete_freqs = None
        if hasattr(self, 'db_S') and self.db_S is not None and hasattr(self, 'freqs') and self.freqs is not None:
            # Match generate_complete_list: linear amplitude per frame, optional edge weights,
            # then temporal aggregation (mean/median/max). Using mean(dB)->linear biases peak
            # validation vs the exported complete spectrum and can reject real harmonics or
            # distort SNR / parabolic refinement (visible as stepped or inconsistent roll-offs).
            agg_func = {"median": np.median, "max": np.max}.get(
                getattr(self, "time_avg", "mean"), np.mean
            )
            db_S_arr = np.asarray(self.db_S, dtype=float)
            if db_S_arr.ndim == 2:
                lin = np.power(10.0, db_S_arr / 20.0)
                lin = np.nan_to_num(lin, nan=0.0, posinf=0.0, neginf=0.0)
                fw = getattr(self, "frame_weights", None)
                if fw is not None and len(fw) == lin.shape[1]:
                    lin = lin * np.asarray(fw, dtype=float)[np.newaxis, :]
                complete_magnitudes = agg_func(lin, axis=1)
                complete_magnitudes = np.asarray(complete_magnitudes, dtype=float)
                complete_freqs = self.freqs
            else:
                complete_magnitudes = np.power(10.0, db_S_arr / 20.0)
                complete_freqs = self.freqs
        elif self.complete_list_df is not None and not self.complete_list_df.empty:
            # Fallback: use complete_list_df (already time-averaged)
            if 'Amplitude' in self.complete_list_df.columns:
                complete_magnitudes = self.complete_list_df['Amplitude'].values
            elif 'Magnitude (dB)' in self.complete_list_df.columns:
                complete_magnitudes = np.power(10.0, self.complete_list_df['Magnitude (dB)'].values / 20.0)
            else:
                complete_magnitudes = None
            
            if 'Frequency (Hz)' in self.complete_list_df.columns:
                complete_freqs = self.complete_list_df['Frequency (Hz)'].values
            else:
                complete_freqs = None

        harmonic_list = []
        candidate_rows: list = []
        max_harm = int(freq_max / f0) + 1
        expected = [f0 * n for n in range(1, max_harm + 1)]
        # AUDIT NOTE — this is the count of *expected* harmonic ORDERS up to
        # freq_max (an upper bound). Acceptance requires a validated local
        # peak inside the tolerance window (see `_is_local_peak_valid`); the
        # actual identified count is logged after the loop. The companion
        # ``candidate_rows`` list captures one entry per expected order
        # (regardless of strict acceptance) so the per-note Harmonic Spectrum
        # sheet can populate the ``harmonic_log_amplitude_density`` metric
        # from a much richer population than the strict diagnostics list.
        self.logger.info(
            f"Searching up to {len(expected)} harmonic orders (f0={f0:.2f} Hz, "
            f"freq_max={freq_max:.1f} Hz). Strict acceptance requires a validated local peak; "
            f"density candidates use more permissive criteria."
        )
        self.expected_harmonic_count = int(len(expected))
        
        # Mathematical reasoning (standard reference):
        # Harmonic series: f_n = n * f0 (where n = 1, 2, 3, ...)
        # For correct harmonic identification, we must select the frequency CLOSEST to the expected harmonic,
        # NOT the one with maximum amplitude. This is because:
        # 1. Harmonic series is defined by frequency relationships, not amplitude
        # 2. Frequency accuracy is more critical than amplitude for harmonic identification
        # 3. Selecting by amplitude can incorrectly choose non-harmonic components that happen to be louder
        # 4. The closest frequency minimizes the error: |f_detected - f_expected| which is the mathematical criterion
        
        last_tol_hz: float = float(tolerance) if tolerance else 0.0
        for hnum, ef in enumerate(expected, 1):
            # Tolerance based on bin spacing
            if bin_spacing and has_sub_bin_interpolation:
                # With sub-bin interpolation: 0.5 bin tolerance
                # Without: 1.0 bin tolerance
                tolerance_bins = 0.5 if has_sub_bin_interpolation else 1.0
                tol_hz_from_bin = bin_spacing * tolerance_bins
                # Adaptive tolerance (2% or bin-based, use the more restrictive)
                tol_hz_adaptive = ef * 0.02 if use_adaptive_tolerance else tolerance
                tol_hz = max(tol_hz_from_bin, tol_hz_adaptive)
            else:
                # Fallback: adaptive tolerance only
                tol_hz = max(tolerance, ef * 0.02) if use_adaptive_tolerance else tolerance
            last_tol_hz = float(tol_hz)
            # ------------------------------------------------------------
            # CANDIDATE PATH — populate harmonic_spectrum_candidates with
            # one row per expected order regardless of strict acceptance.
            # The candidate is picked by maximum finite Amplitude inside
            # the tolerance window from filtered_list_df first, falling
            # back to complete_list_df.
            # ------------------------------------------------------------
            candidate_rows.append(
                self._build_harmonic_candidate_row(
                    hnum=hnum,
                    expected_freq_hz=float(ef),
                    tol_hz=float(tol_hz),
                    complete_magnitudes=complete_magnitudes,
                    complete_freqs=complete_freqs,
                )
            )
            
            candidates = self.filtered_list_df[
                (self.filtered_list_df['Frequency (Hz)'] >= ef - tol_hz) &
                (self.filtered_list_df['Frequency (Hz)'] <= ef + tol_hz)
            ]
            
            if not candidates.empty:
                # Select frequency closest to expected harmonic
                # Mathematical criterion: argmin(|f_detected - f_expected|)
                candidates_copy = candidates.copy()
                candidates_copy['FreqError'] = abs(candidates_copy['Frequency (Hz)'] - ef)
                best_idx = candidates_copy['FreqError'].idxmin()
                best = candidates.loc[best_idx].copy()
                
                # Validate local peak
                # Check if selected frequency corresponds to a valid local peak (not side lobe)
                is_valid_peak = True
                snr_db = -np.inf
                
                if complete_magnitudes is not None and complete_freqs is not None:
                    # Find index in complete spectrum
                    freq_val = best['Frequency (Hz)']
                    freq_idx = np.argmin(np.abs(complete_freqs - freq_val))
                    
                    # Validate local peak
                    is_valid_peak, snr_db = _is_local_peak_valid(
                        complete_magnitudes, freq_idx,
                        threshold_db=3.0, noise_floor_percentile=15.0, window_size=50
                    )
                    
                    if not is_valid_peak:
                        self.logger.debug(f"Harmonic {hnum} (f={freq_val:.2f} Hz): Invalid peak (SNR={snr_db:.1f} dB) - may be side lobe")
                    else:
                        self.logger.debug(f"Harmonic {hnum} (f={freq_val:.2f} Hz): Valid peak (SNR={snr_db:.1f} dB)")
                        
                        # Apply sub-bin interpolation
                        if bin_spacing and has_sub_bin_interpolation:
                            # Find frequency base (first bin frequency)
                            freq_base = complete_freqs[0] if len(complete_freqs) > 0 else 0.0
                            
                            # Apply parabolic interpolation in log-magnitude
                            freq_corrected, is_valid = _parabolic_interpolation_log_magnitude(
                                complete_magnitudes, freq_idx, bin_spacing, freq_base
                            )
                            
                            if is_valid:
                                # Update frequency with sub-bin correction
                                old_freq = best['Frequency (Hz)']
                                best['Frequency (Hz)'] = freq_corrected
                                correction_hz = freq_corrected - old_freq
                                self.logger.debug(f"Harmonic {hnum}: Sub-bin correction {correction_hz:.4f} Hz ({correction_hz/bin_spacing:.3f} bins)")
                                best['SubBinCorrected'] = True
                            else:
                                best['SubBinCorrected'] = False
                        else:
                            best['SubBinCorrected'] = False
                
                if is_valid_peak:
                    best['Harmonic Number'] = hnum
                    best['SNR_dB'] = snr_db
                    exists = any(abs(r['Frequency (Hz)'] - best['Frequency (Hz)']) < 0.1 for _, r in pd.DataFrame(harmonic_list).iterrows()) if harmonic_list else False
                    if not exists:
                        harmonic_list.append(best)
                else:
                    # AUDIT FIX (harmonic over-classification) — old
                    # behaviour: silently promoted the nearest-frequency
                    # bin even when `_is_local_peak_valid` rejected it
                    # (no local maximum, no SNR margin, no prominence).
                    # On signals with several hundred expected orders
                    # that produced the symptomatic
                    # ``Searching N orders → N identified`` log line.
                    # New behaviour: a candidate from complete_list_df
                    # IS still examined, but the SAME local-peak +
                    # SNR ≥ snr_threshold_db + prominence ≥ threshold_db
                    # acceptance criteria are enforced. Orders whose
                    # candidate fails are left MISSING so residual /
                    # inharmonic energy is attributed to the correct
                    # bucket.
                    # AUDIT FIX (harmonic over-classification) — Tolerant
                    # SNR-gated fallback.
                    #
                    # The previous behaviour silently promoted any nearest
                    # bin from complete_list_df, regardless of SNR. The
                    # strict ``_is_local_peak_valid`` (threshold_db=3.0)
                    # is too tight in the presence of FFT main-lobe
                    # smearing on windowed sinusoids: the bins immediately
                    # adjacent to the true peak can sit within 1–2 dB of
                    # it, so the 3 dB-above-neighbours requirement
                    # rejects perfectly legitimate harmonics. The
                    # acceptance criteria the audit requires (local
                    # maximum + SNR ≥ snr_threshold_db) ARE enforced via
                    # a SNR check against the local noise floor; the
                    # 3 dB-above-neighbours requirement is relaxed for
                    # the fallback to avoid main-lobe false negatives.
                    accepted_fallback = self._accept_harmonic_via_snr_gated_fallback(
                        ef=ef,
                        tol_hz=tol_hz,
                        complete_magnitudes=complete_magnitudes,
                        complete_freqs=complete_freqs,
                        harmonic_list=harmonic_list,
                        hnum=hnum,
                    )
                    if not accepted_fallback:
                        self.logger.debug(
                            f"Harmonic {hnum} (f≈{ef:.2f} Hz): "
                            f"rejected (no SNR-validated candidate; SNR={snr_db:.1f} dB)."
                        )
            else:
                # No candidate inside tolerance window in filtered_list_df.
                # AUDIT FIX — still try complete_list_df via the same
                # SNR-gated fallback (the previous behaviour silently
                # promoted the nearest bin without ANY validation).
                accepted_fallback = self._accept_harmonic_via_snr_gated_fallback(
                    ef=ef,
                    tol_hz=tol_hz,
                    complete_magnitudes=complete_magnitudes,
                    complete_freqs=complete_freqs,
                    harmonic_list=harmonic_list,
                    hnum=hnum,
                )
                if not accepted_fallback:
                    self.logger.debug(
                        f"Harmonic {hnum} (f≈{ef:.2f} Hz): "
                        f"no SNR-validated candidate in tolerance window; order left missing."
                    )

        # Robust f0 global fit + acceptance guard.
        #
        # Accept the fitted f0 only when ALL of the following hold:
        #   * at least 3 strict (local-peak + SNR + prominence) harmonics
        #     were available to drive the fit;
        #   * residual_std <= max(harmonic tolerance window, 1% of f0_initial);
        #   * abs(f0_adjusted - f0_initial) <= 2% of f0_initial
        #     (this is the strong guard against runaway fits — keeps the
        #     fitted f0 within a quarter-tone of the nominal value);
        #   * fit_quality (residual_std / f0_adjusted) <= 0.10.
        #
        # NOTE on max_fit_quality: previously 0.05, but the Clarinete_mf
        # corpus audit showed many otherwise-fine fits with strict_peaks
        # ≥ 30 sitting at fit_quality 0.07–0.10 (e.g. B3, C4, D#4, E4).
        # Those fits ARE useful (the fitted f0 is within a fraction of a
        # cent of nominal); they were only being rejected because
        # high-order partials carry real, mild inharmonicity that inflates
        # residual_std. The 2% absolute-shift gate is the substantive
        # guard against bad fits — fit_quality is a soft tie-break. Raising
        # to 0.10 keeps the strict cases (F#3 at 0.37, G#3 at 0.18) out
        # while letting through the mid-quality fits that improve the
        # downstream harmonic alignment.
        #
        # On rejection we publish ``f0_final_method = nominal_or_initial_due_to_bad_fit``
        # and keep f0_initial for downstream code.
        strict_peak_count_for_fit = len(harmonic_list)
        max_residual_std_hz = max(float(last_tol_hz), 0.01 * float(f0))
        max_abs_shift_hz = 0.02 * float(f0)
        max_fit_quality = 0.10
        f0_est = float(f0)
        fit_quality = 0.0
        residual_std = 0.0
        if strict_peak_count_for_fit >= 3:
            detected_freqs = np.array([h['Frequency (Hz)'] for h in harmonic_list])
            detected_amps = np.array([h['Amplitude'] for h in harmonic_list])

            f0_robust = _estimate_f0_global_robust(detected_freqs, detected_amps, f0)

            f0_est = float(f0_robust.get('f0_estimated', f0))
            fit_quality = float(f0_robust.get('fit_quality', 0.0))
            residual_std = float(f0_robust.get('residual_std', 0.0))
            accept_f0 = (
                np.isfinite(f0_est)
                and residual_std <= max_residual_std_hz
                and abs(f0_est - float(f0)) <= max_abs_shift_hz
                and fit_quality <= max_fit_quality
            )
        else:
            accept_f0 = False

        fit_rej: Optional[str] = None
        if not accept_f0:
            if strict_peak_count_for_fit < 3:
                fit_rej = "insufficient_strict_harmonic_peaks"
            elif strict_peak_count_for_fit >= 3:
                if not np.isfinite(f0_est):
                    fit_rej = "non_finite_f0_estimate"
                elif residual_std > max_residual_std_hz:
                    fit_rej = "residual_std_exceeds_gate"
                elif abs(f0_est - float(f0)) > max_abs_shift_hz:
                    fit_rej = "f0_shift_exceeds_gate"
                elif fit_quality > max_fit_quality:
                    fit_rej = "fit_quality_exceeds_gate"
                else:
                    fit_rej = "harmonic_fit_rejected"

        self.f0_robust_fit_quality = float(fit_quality)

        _fq_pass: Optional[float] = (
            float(fit_quality) if strict_peak_count_for_fit >= 3 else None
        )
        if _fq_pass is not None and not np.isfinite(_fq_pass):
            _fq_pass = None

        _nominal_for_finalize = float(f0)
        try:
            _fn = getattr(self, "f0_nominal_hz", None)
            if _fn is not None:
                _fnf = float(_fn)
                if np.isfinite(_fnf) and _fnf > 0.0:
                    _nominal_for_finalize = _fnf
        except (TypeError, ValueError):
            pass

        self._finalize_f0_state(
            nominal_hz=_nominal_for_finalize,
            candidate_hz=float(f0_est),
            accept_fit=bool(accept_f0),
            fit_quality=_fq_pass,
            residual_std_hz=float(residual_std),
            rejection_reason=None if accept_f0 else fit_rej,
        )
        self._finalize_low_frequency_policy_state()

        if self.f0_fit_accepted:
            self.f0_robust = float(self.f0_final)
            self.logger.info(
                f"Global-fit f0 accepted: {f0:.4f} -> {float(self.f0_final):.4f} Hz "
                f"(residual_std={residual_std:.4f} Hz, "
                f"fit_quality={fit_quality:.6f}, strict_peaks={strict_peak_count_for_fit})"
            )
        else:
            try:
                if hasattr(self, "f0_robust"):
                    delattr(self, "f0_robust")
            except Exception:
                pass
            self.logger.warning(
                f"Robust f0 rejected: keeping nominal/initial f0={f0:.4f} Hz. "
                f"fit_attempted_f0={f0_est:.4f} Hz, strict_peaks={strict_peak_count_for_fit}, "
                f"residual_std={residual_std:.4f} (limit={max_residual_std_hz:.4f}), "
                f"|df0|={abs(f0_est - float(f0)):.4f} (limit={max_abs_shift_hz:.4f}), "
                f"fit_quality={fit_quality:.6f} (limit={max_fit_quality:.4f})."
            )

        self.harmonic_list_df = pd.DataFrame(harmonic_list).reset_index(drop=True) if harmonic_list else pd.DataFrame()
        strict_n = int(len(self.harmonic_list_df))
        self.strict_harmonic_count = strict_n
        if not self.harmonic_list_df.empty:
            self.logger.info(
                "%d strict_harmonic_peaks accepted out of up to %d expected orders "
                "(local-peak + SNR ≥ 3 dB + prominence > 3 dB; "
                "orders without a validated peak are left out of the strict list).",
                strict_n,
                len(expected),
            )

        # Materialise the harmonic_spectrum_candidates_df and surface the
        # density-bound aggregates. The Harmonic Spectrum sheet exported
        # for downstream tools (and the Density_Metrics harmonic-amplitude
        # sum) draws from this DataFrame, NOT from the strict diagnostics
        # list.
        if candidate_rows:
            cand_df = pd.DataFrame(candidate_rows).reset_index(drop=True)
        else:
            cand_df = pd.DataFrame()
        if (
            not cand_df.empty
            and getattr(self, "subfundamental_guard_valid", False)
            and "Frequency (Hz)" in cand_df.columns
            and "include_for_density" in cand_df.columns
        ):
            try:
                _cut_h = float(self.adaptive_subfundamental_cutoff_hz)
            except (TypeError, ValueError):
                _cut_h = float("nan")
            if np.isfinite(_cut_h):
                _fq_h = pd.to_numeric(cand_df["Frequency (Hz)"], errors="coerce")
                cand_df.loc[_fq_h < _cut_h, "include_for_density"] = False
        self.harmonic_spectrum_candidates_df = cand_df
        candidate_n = int(len(cand_df))
        if candidate_n > 0 and "include_for_density" in cand_df.columns:
            inc_mask = cand_df["include_for_density"].astype(bool)
            density_included_n = int(inc_mask.sum())
            amp_for_density = pd.to_numeric(
                cand_df.loc[inc_mask, "Amplitude_raw"], errors="coerce"
            )
            harm_amp_sum = float(np.nansum(amp_for_density.to_numpy(dtype=float)))
        else:
            density_included_n = 0
            harm_amp_sum = 0.0
        self.harmonic_candidate_count_density = density_included_n
        self.harmonic_amplitude_sum = float(harm_amp_sum)
        self.harmonic_log_amplitude_density = float(
            np.log10(1.0 + max(0.0, harm_amp_sum))
        )

        self.logger.info(
            "Harmonic extraction summary: expected=%d strict=%d candidates=%d "
            "density_included=%d f0_fit_accepted=%s",
            int(len(expected)),
            strict_n,
            candidate_n,
            density_included_n,
            str(bool(accept_f0)),
        )
        if strict_n == 0:
            self.logger.warning(
                "No strict harmonics accepted (searched up to %d orders; "
                "strict criterion: local peak + SNR ≥ 3 dB + prominence > 3 dB). "
                "Density candidates: %d / %d included.",
                len(expected),
                density_included_n,
                candidate_n,
            )

    # ------------------------------------------------------------------
    # AUDIT FIX (Fgt_pp finding C1) — canonical harmonic-protection
    # population for the Sub-bass aggregator and the Sub-bass band
    # sheet. Returns a small DataFrame carrying only the
    # ``Frequency (Hz)`` column (the only column the aggregator and the
    # sheet-construction mask read), built as the union of:
    #   * the strict-harmonic list (``self.harmonic_list_df``), AND
    #   * the wider Harmonic Spectrum candidates whose
    #     ``include_for_density`` flag is True.
    # The union guarantees we never protect FEWER bins than the legacy
    # strict-only behaviour, while adding protection for low harmonics
    # that pass the density filter but fail the SNR/prominence gate.
    # ------------------------------------------------------------------
    def _select_nonharmonic_peak_candidates_from_residual_rows(
        self,
        residual_df: pd.DataFrame,
        *,
        amp_quantile_when_no_harmonics: float = 0.85,
        min_kept_when_harmonics_present: int = 1,
    ) -> pd.DataFrame:
        """Reduce residual spectral rows to amplitude-ranked non-harmonic peak candidates.

        ``identify_nonharmonic_residual_rows`` returns every FFT bin in the
        filtered spectrum that lies **outside** widened harmonic exclusion
        windows. That set is dominated by spectral background between
        harmonics, not discrete inharmonic partials.

        This helper keeps the **top N** rows by amplitude (``N ≈`` strict
        harmonic count, or a high quantile when no harmonics exist) so
        downstream linear sums are not inflated by row-count effects.

        **Important:** this does **not** prove local-maximum structure, SNR,
        prominence, or temporal stability. The output classification level is
        **nonharmonic_peak_candidate** (ranked residual rows), not
        ``accepted_inharmonic_peak`` or ``inharmonic_partial``.
        """
        ih_df = residual_df
        if ih_df is None or ih_df.empty:
            return ih_df

        # Resolve an amplitude column.
        amp_col: Optional[str] = None
        for c in ("Amplitude", "Amplitude_raw"):
            if c in ih_df.columns:
                amp_col = c
                break
        if amp_col is None and "Magnitude (dB)" in ih_df.columns:
            db_v = pd.to_numeric(ih_df["Magnitude (dB)"], errors="coerce").to_numpy(float)
            amp_v = np.power(10.0, db_v / 20.0)
            ih_df = ih_df.copy()
            ih_df["_amp_tmp"] = amp_v
            amp_col = "_amp_tmp"
        if amp_col is None:
            return ih_df

        ih_amps = pd.to_numeric(ih_df[amp_col], errors="coerce").to_numpy(float)
        ih_amps = np.nan_to_num(ih_amps, nan=0.0, posinf=0.0, neginf=0.0)
        ih_amps = np.maximum(ih_amps, 0.0)
        if ih_amps.size == 0:
            return ih_df

        # Determine how many inharmonic peaks to keep.
        n_keep: Optional[int] = None
        hdf = getattr(self, "harmonic_list_df", None)
        if isinstance(hdf, pd.DataFrame) and not hdf.empty:
            n_keep = int(len(hdf))
            if n_keep < max(1, int(min_kept_when_harmonics_present)):
                n_keep = max(1, int(min_kept_when_harmonics_present))
        else:
            try:
                cut = float(np.quantile(ih_amps, float(amp_quantile_when_no_harmonics)))
            except Exception:
                cut = 0.0
            if cut > 0.0:
                n_keep = int(np.count_nonzero(ih_amps >= cut))

        if not n_keep or n_keep <= 0:
            return ih_df
        n_keep = int(min(n_keep, ih_amps.size))

        # Argpartition for top-n selection (O(N), stable enough for
        # downstream pandas reset_index). The resulting order is
        # arbitrary inside the kept set, so we re-sort the output by
        # frequency at the end for a tidy export.
        order = np.argpartition(-ih_amps, n_keep - 1)[:n_keep]
        out = ih_df.iloc[order].copy()
        if "Frequency (Hz)" in out.columns:
            out = out.sort_values("Frequency (Hz)", kind="mergesort")
        if amp_col == "_amp_tmp":
            out = out.drop(columns=["_amp_tmp"], errors="ignore")
        return out.reset_index(drop=True)

    def _filter_inharmonic_to_local_peaks(
        self,
        ih_df: pd.DataFrame,
        *,
        amp_quantile_when_no_harmonics: float = 0.85,
        min_kept_when_harmonics_present: int = 1,
    ) -> pd.DataFrame:
        """Deprecated wrapper — use ``_select_nonharmonic_peak_candidates_from_residual_rows``."""
        return self._select_nonharmonic_peak_candidates_from_residual_rows(
            ih_df,
            amp_quantile_when_no_harmonics=amp_quantile_when_no_harmonics,
            min_kept_when_harmonics_present=min_kept_when_harmonics_present,
        )

    def _nonharmonic_residual_pipeline_dataframes(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Same ``peak_src`` + identify + top-N select as ``Inharmonic Spectrum`` export."""
        peak_src = getattr(self, "filtered_list_df", None)
        if peak_src is None or getattr(peak_src, "empty", True):
            peak_src = getattr(self, "complete_list_df", None)
        peak_src = self._dataframe_for_density_frequency_floor(peak_src)
        empty = pd.DataFrame()
        hl = getattr(self, "harmonic_list_df", None)
        if (
            not isinstance(hl, pd.DataFrame)
            or hl.empty
            or not isinstance(peak_src, pd.DataFrame)
            or peak_src.empty
        ):
            return empty.copy(), empty.copy()
        try:
            ih_full = identify_nonharmonic_residual_rows(
                hl,
                peak_src,
                tolerance=0.02,
                **self._spectral_leakage_guard_kwargs(),
            )
        except Exception:
            ih_full = pd.DataFrame()
        if ih_full is None:
            ih_full = empty.copy()
        ih_sel = ih_full.copy()
        if not ih_sel.empty:
            try:
                ih_sel = self._select_nonharmonic_peak_candidates_from_residual_rows(ih_sel)
            except Exception:
                pass
        return ih_full, ih_sel

    def _assign_hierarchical_residual_debug_counts(
        self,
        ih_identified_full: pd.DataFrame,
        ih_retained: pd.DataFrame,
    ) -> None:
        """Populate residual-row hierarchy counts (no peaklist mixing)."""
        n_full = (
            int(len(ih_identified_full))
            if isinstance(ih_identified_full, pd.DataFrame) and not ih_identified_full.empty
            else 0
        )
        n_ret = (
            int(len(ih_retained))
            if isinstance(ih_retained, pd.DataFrame) and not ih_retained.empty
            else 0
        )
        self.residual_spectral_row_count = int(n_full)
        self.nonharmonic_candidate_row_count = int(n_full)
        self.retained_nonharmonic_peak_candidate_count = int(n_ret)
        self.exported_nonharmonic_peak_candidate_count = int(n_ret)

    def _build_subbass_harmonic_protection_df(self) -> pd.DataFrame:
        freq_arrays: list[np.ndarray] = []
        strict = getattr(self, "harmonic_list_df", None)
        if isinstance(strict, pd.DataFrame) and not strict.empty and "Frequency (Hz)" in strict.columns:
            arr = pd.to_numeric(strict["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
            freq_arrays.append(arr[np.isfinite(arr)])
        cand = getattr(self, "harmonic_spectrum_candidates_df", None)
        if (
            isinstance(cand, pd.DataFrame)
            and not cand.empty
            and "Frequency (Hz)" in cand.columns
            and "include_for_density" in cand.columns
        ):
            mask = cand["include_for_density"].astype(bool).to_numpy()
            arr = pd.to_numeric(cand["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
            keep = arr[mask & np.isfinite(arr)]
            freq_arrays.append(keep)
        if freq_arrays:
            combined = np.concatenate(freq_arrays)
            combined = combined[np.isfinite(combined)]
            combined = np.unique(np.round(combined, 6))
            return pd.DataFrame({"Frequency (Hz)": combined})
        return pd.DataFrame({"Frequency (Hz)": pd.Series(dtype=float)})

    # ----------------- mÃ©tricas (FFT) -----------------
    def _get_actual_n_fft(self) -> int:
        """
        Get the actual N_FFT size used in the FFT analysis.
        
        This is important because self.n_fft may differ from the actual FFT size
        if a downgrade occurred during processing (e.g., due to memory constraints).
        
        Returns:
            Actual N_FFT size
            
        Note:
            - STFT output has shape (n_fft_padded//2 + 1, n_frames) for real FFT
            - If zero_padding is used: n_fft_padded = n_fft * zero_padding
            - Example: n_fft=4096, zero_padding=2 → n_fft_padded=8192 → shape[0]=4097
            - Formula: n_fft_padded = (shape[0] - 1) * 2
            - For normalization, we use n_fft_padded (not original n_fft) because that's
              what determines the number of frequency bins and components detected.
        """
        if hasattr(self, 'freqs') and self.freqs is not None and len(self.freqs) > 0:
            # Actual FFT size = (number of frequency bins - 1) * 2
            # For real FFT: n_fft bins → n_fft//2 + 1 unique frequency bins
            # So: n_fft = (len(freqs) - 1) * 2
            # This gives the PADDED n_fft (if zero_padding was used)
            return (len(self.freqs) - 1) * 2
        elif hasattr(self, 'S') and self.S is not None and self.S.shape[0] > 0:
            # Actual FFT size from STFT shape
            # S.shape[0] = n_fft//2 + 1 (for real FFT)
            # So: n_fft = (S.shape[0] - 1) * 2
            # This gives the PADDED n_fft (if zero_padding was used)
            return (self.S.shape[0] - 1) * 2
        else:
            # Fallback to self.n_fft if FFT data not available
            # But this doesn't account for zero_padding!
            n_fft = getattr(self, "n_fft", 2048)
            zero_padding = getattr(self, "zero_padding", 1)
            return n_fft * zero_padding

    @staticmethod
    def _normalize_weight_function_ui_key(weight_function: Optional[str]) -> str:
        key = str(weight_function or "linear").strip().lower()
        if key == "d2":
            return "linear"
        if key == "d8":
            return "d17"
        return key

    # ---------------------------------------------------------------------
    # AUDIT FIX (stale-pipeline guard) — pre-save schema validator.
    # ---------------------------------------------------------------------
    def _validate_per_note_export_schema(
        self,
        *,
        harm_df: "pd.DataFrame",
        ih_df: "pd.DataFrame",
        sb_df: "pd.DataFrame",
        meta_rows: list,
        note: str,
    ) -> None:
        """Last-chance schema audit for a per-note ``spectral_analysis.xlsx``.

        Raises ``RuntimeError`` (and the surrounding writer is rolled
        back by the caller) when any of the following audit-canonical
        conditions are violated:

        * harmonic / inharmonic / sub-bass spectrum sheets must carry
          ``Amplitude_raw`` and ``Power_raw`` (empty sheets are
          tolerated because they hold no partials to characterise);
        * the metadata rows must declare the current
          ``ANALYSIS_SCHEMA_VERSION``;
        * in ``integrated_single_pass`` mode the Inharmonic Spectrum
          must NOT carry ``batch_*`` columns; ``model_weights_source``
          must be ``current_analysis``; ``export_alignment_source``
          must be ``disabled_integrated_single_pass`` and
          ``export_alignment_factor`` must be ``1.0``.

        These are the same predicates the compile / GUI guards use, so
        catching them HERE means a stale workbook can never reach disk
        in the first place.
        """
        meta_dict = {str(k): v for (k, v) in meta_rows}
        errors: list[str] = []

        # AUDIT FIX (current_analysis hardening) — hard runtime assertion
        # against any silent reactivation of the legacy export alignment.
        # In integrated_single_pass / current_analysis mode the
        # export_alignment_* fields must declare the disabled state
        # exactly; if not, we crash before the workbook can be written
        # so the operator immediately sees the bug.
        _self_mws = str(getattr(self, "model_weights_source", "") or "")
        _self_auto = bool(getattr(self, "auto_model_weights_from_analysis", True))
        if _self_auto or _self_mws == "current_analysis":
            _eas = str(getattr(self, "export_alignment_source", "") or "")
            try:
                _eaf = float(getattr(self, "export_alignment_factor", 1.0) or 1.0)
            except (TypeError, ValueError):
                _eaf = float("nan")
            if _eas != "disabled_integrated_single_pass" or not (
                abs(_eaf - 1.0) < 1e-12
            ):
                raise RuntimeError(
                    "BUG: legacy export alignment active during integrated "
                    "single-pass analysis "
                    f"(export_alignment_source={_eas!r}, "
                    f"export_alignment_factor={_eaf!r}, "
                    f"model_weights_source={_self_mws!r}, "
                    f"auto_model_weights_from_analysis={_self_auto!r})."
                )

        if str(meta_dict.get("analysis_schema_version", "")) != ANALYSIS_SCHEMA_VERSION:
            errors.append(
                f"analysis_schema_version != {ANALYSIS_SCHEMA_VERSION} "
                f"(got {meta_dict.get('analysis_schema_version')!r})"
            )

        for label, dfx in (
            ("Harmonic Spectrum", harm_df),
            ("Inharmonic Spectrum", ih_df),
            ("Sub-bass band", sb_df),
        ):
            if not _spectral_sheet_has_raw_columns(dfx):
                cols = (
                    [str(c) for c in dfx.columns]
                    if isinstance(dfx, pd.DataFrame)
                    else []
                )
                errors.append(
                    f"{label} is missing Amplitude_raw / Power_raw "
                    f"(columns: {cols})"
                )

        # Current-analysis invariants (formerly integrated_single_pass).
        cps = str(meta_dict.get("component_profile_source", ""))
        is_current_analysis = (
            cps in ("current_analysis", "integrated_single_pass")
            or bool(getattr(self, "auto_model_weights_from_analysis", True))
        )
        if is_current_analysis:
            mws = str(meta_dict.get("model_weights_source", ""))
            if mws != "current_analysis":
                errors.append(
                    f"current_analysis mode requires model_weights_source == "
                    f"'current_analysis' (got {mws!r})"
                )
            eas = str(meta_dict.get("export_alignment_source", ""))
            if eas != "disabled_integrated_single_pass":
                errors.append(
                    f"current_analysis mode requires export_alignment_source == "
                    f"'disabled_integrated_single_pass' (got {eas!r})"
                )
            try:
                eaf = float(meta_dict.get("export_alignment_factor", 0.0))
            except (TypeError, ValueError):
                eaf = float("nan")
            if not (abs(eaf - 1.0) < 1e-12):
                errors.append(
                    f"current_analysis mode requires export_alignment_factor == 1.0 "
                    f"(got {eaf!r})"
                )

            # Internal-compatibility aliases must not leak into the
            # user-facing Inharmonic Spectrum sheet in current-analysis mode.
            if isinstance(ih_df, pd.DataFrame) and not ih_df.empty:
                forbidden_cols = [
                    c for c in ih_df.columns
                    if str(c).lower().startswith("batch_")
                ]
                if forbidden_cols:
                    errors.append(
                        f"current_analysis mode: Inharmonic Spectrum still carries "
                        f"internal-only compatibility columns: {forbidden_cols}"
                    )

        if errors:
            msg = (
                f"[STALE-PIPELINE GUARD] Per-note workbook for note {note!r} "
                f"would have been written with a stale / legacy schema. "
                f"Refusing to save. Violations:\n  - "
                + "\n  - ".join(errors)
                + f"\nExpected analysis_schema_version = {ANALYSIS_SCHEMA_VERSION!r}. "
                + "Regenerate the analysis with the current single-pass raw-export pipeline."
            )
            try:
                self.logger.error(msg)
            except Exception:
                pass
            raise RuntimeError(msg)

    # ---------------------------------------------------------------------
    # Harmonic spectrum candidate row builder.
    # ---------------------------------------------------------------------
    def _build_harmonic_candidate_row(
        self,
        *,
        hnum: int,
        expected_freq_hz: float,
        tol_hz: float,
        complete_magnitudes: Optional[np.ndarray],
        complete_freqs: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        """Build one ``harmonic_spectrum_candidates`` row for harmonic order
        ``hnum`` (expected frequency = ``expected_freq_hz``).

        The candidate search picks the strongest finite-amplitude bin inside
        ``[ef ± tol_hz]`` from ``filtered_list_df`` first, falling back to
        ``complete_list_df``. The candidate is then classified via
        :func:`_classify_harmonic_candidate`, which controls whether the row
        contributes to the harmonic-amplitude sum used by Density_Metrics.

        The returned dict ALWAYS carries the audit-canonical column set so
        the per-note workbook schema is stable regardless of whether a
        candidate was found:

            Harmonic Number, expected_frequency_hz, extracted_frequency_hz,
            frequency_deviation_hz, bin_center_frequency_hz,
            interpolated_frequency_hz, subbin_offset_bins,
            subbin_interpolation_valid, peak_bin_index,
            Frequency (Hz),
            Amplitude_raw, Power_raw, snr_db, prominence_db,
            local_peak_valid, candidate_status, include_for_density, Note.
        """
        nan = float("nan")
        row: Dict[str, Any] = {
            "Harmonic Number": int(hnum),
            "expected_frequency_hz": float(expected_freq_hz),
            "extracted_frequency_hz": nan,
            "frequency_deviation_hz": nan,
            "bin_center_frequency_hz": nan,
            "interpolated_frequency_hz": nan,
            "subbin_offset_bins": nan,
            "subbin_interpolation_valid": False,
            "peak_bin_index": nan,
            "Frequency (Hz)": nan,
            "Amplitude": nan,
            "Amplitude_raw": nan,
            "Power_raw": nan,
            "snr_db": nan,
            "prominence_db": nan,
            "local_peak_valid": False,
            "candidate_status": "missing_window",
            "include_for_density": False,
            "Magnitude (dB)": nan,
            "Note": getattr(self, "note", None),
        }

        def _search(df_opt) -> Optional[pd.DataFrame]:
            if df_opt is None or not isinstance(df_opt, pd.DataFrame) or df_opt.empty:
                return None
            if "Frequency (Hz)" not in df_opt.columns:
                return None
            mask = (
                (df_opt["Frequency (Hz)"] >= expected_freq_hz - tol_hz)
                & (df_opt["Frequency (Hz)"] <= expected_freq_hz + tol_hz)
            )
            cand = df_opt[mask]
            return cand if not cand.empty else None

        cand = _search(getattr(self, "filtered_list_df", None))
        if cand is None:
            cand = _search(getattr(self, "complete_list_df", None))
        if cand is None:
            return row

        # Pick the strongest finite-amplitude bin inside the window
        # (audit rule: "extract strongest finite amplitude candidate").
        amp_col: Optional[str] = None
        if "Amplitude" in cand.columns:
            amp_col = "Amplitude"
        elif "Magnitude (dB)" in cand.columns:
            cand = cand.copy()
            cand["Amplitude"] = np.power(
                10.0,
                pd.to_numeric(cand["Magnitude (dB)"], errors="coerce").fillna(-120.0)
                / 20.0,
            )
            amp_col = "Amplitude"
        if amp_col is None:
            return row

        amp_series = pd.to_numeric(cand[amp_col], errors="coerce")
        finite_mask = amp_series.notna() & (amp_series > 0)
        if not finite_mask.any():
            return row
        best_idx = amp_series[finite_mask].idxmax()
        best = cand.loc[best_idx]
        bin_candidate_freq = float(best["Frequency (Hz)"])
        amplitude_raw = float(amp_series.loc[best_idx])
        peak_refinement: Dict[str, Any] = {
            "peak_bin_index": nan,
            "bin_center_frequency_hz": bin_candidate_freq,
            "interpolated_frequency_hz": bin_candidate_freq,
            "subbin_offset_bins": 0.0,
            "subbin_interpolation_valid": False,
            "peak_amplitude_raw": amplitude_raw,
            "peak_magnitude_db": float(
                best["Magnitude (dB)"]
                if "Magnitude (dB)" in best.index
                else 20.0 * float(np.log10(max(amplitude_raw, 1e-12)))
            ),
        }
        if (
            complete_magnitudes is not None
            and complete_freqs is not None
            and len(complete_freqs) > 0
        ):
            peak_refinement = _refine_candidate_to_interpolated_peak(
                candidate_freq_hz=bin_candidate_freq,
                complete_magnitudes=complete_magnitudes,
                complete_freqs=complete_freqs,
                refine_radius=2,
            )

        extracted_freq = (
            float(peak_refinement["interpolated_frequency_hz"])
            if peak_refinement["subbin_interpolation_valid"]
            else float(peak_refinement["bin_center_frequency_hz"])
        )
        if np.isfinite(peak_refinement.get("peak_amplitude_raw", nan)):
            amplitude_raw = float(peak_refinement["peak_amplitude_raw"])

        # Local-peak / SNR / prominence at the refined peak bin.
        local_peak_valid = False
        snr_db = float("nan")
        prominence_db = float("nan")
        if (
            complete_magnitudes is not None
            and complete_freqs is not None
            and len(complete_freqs) > 0
        ):
            pbi = peak_refinement.get("peak_bin_index", nan)
            if np.isfinite(pbi):
                idx = int(pbi)
            else:
                idx = int(np.argmin(np.abs(complete_freqs - extracted_freq)))
            local_peak_valid, snr_db, prominence_db = _local_peak_metrics(
                complete_magnitudes, idx
            )

        # AUDIT FIX (acoustic-physics correction, Clarinete_mf finding
        # #3, revised) — frequency-deviation gate. The candidate search
        # window uses ``2% * expected_freq``; without a deviation gate,
        # ANY spectral peak within that window gets promoted to a
        # "harmonic" candidate, contaminating density with bins that
        # are NOT periodic at n·f₀.
        #
        # The gate must satisfy two physical regimes simultaneously:
        #   * LOW partials (n ≤ ~10): peaks should be within an
        #     FFT-resolution / sub-bin distance of n·f₀; an
        #     absolute-Hz floor (max(5 Hz, 2.5 × bin width)) is the
        #     correct discriminator there.
        #   * UPPER partials (n > 10): real instruments naturally
        #     drift from strict n·f₀ harmonicity by O(0.1–1%) of the
        #     expected frequency (slight inharmonicity, vibrato,
        #     register effects). An absolute-Hz floor that worked at
        #     h=2 (10 Hz) would over-reject at h=34 (where 0.5% of
        #     16 kHz is ~80 Hz — well within musical normality).
        #
        # We therefore combine both into a SINGLE tolerance:
        #     tol = max(5 Hz, 2.5 × bin_width, 0.01 × expected_freq)
        # which is ~1 % relative — comfortably below a quarter-tone
        # (~3 %) so the n·f₀ harmonic-identity constraint still
        # rejects pure broadband peaks, but does not punish real
        # upper-partial drift.
        try:
            _sr_hz_gate = float(
                getattr(self, "sr", None)
                or getattr(self, "sample_rate", 0.0)
                or 0.0
            )
            _n_fft_gate = int(getattr(self, "n_fft", 0) or 0)
            if _sr_hz_gate > 0.0 and _n_fft_gate > 0:
                _bin_hz_gate = _sr_hz_gate / float(_n_fft_gate)
            else:
                _bin_hz_gate = 0.0
        except Exception:
            _bin_hz_gate = 0.0
        _abs_floor_hz = max(5.0, 2.5 * _bin_hz_gate) if _bin_hz_gate > 0 else 5.0
        _rel_floor_hz = 0.01 * float(expected_freq_hz)
        _max_dev_hz_gate = max(_abs_floor_hz, _rel_floor_hz)
        _freq_deviation_abs = abs(float(extracted_freq) - float(expected_freq_hz))
        if _freq_deviation_abs > _max_dev_hz_gate:
            status = "off_frequency"
            include = False
        else:
            status, include = _classify_harmonic_candidate(
                amplitude_raw=amplitude_raw,
                local_peak_valid=local_peak_valid,
                snr_db=snr_db,
                prominence_db=prominence_db,
            )

        row.update(
            {
                "extracted_frequency_hz": extracted_freq,
                "frequency_deviation_hz": float(extracted_freq - expected_freq_hz),
                "Frequency (Hz)": extracted_freq,
                "bin_center_frequency_hz": float(
                    peak_refinement["bin_center_frequency_hz"]
                ),
                "interpolated_frequency_hz": float(
                    peak_refinement["interpolated_frequency_hz"]
                ),
                "subbin_offset_bins": float(peak_refinement["subbin_offset_bins"]),
                "subbin_interpolation_valid": bool(
                    peak_refinement["subbin_interpolation_valid"]
                ),
                "peak_bin_index": (
                    int(peak_refinement["peak_bin_index"])
                    if np.isfinite(peak_refinement.get("peak_bin_index", nan))
                    else nan
                ),
                "Amplitude": amplitude_raw,
                "Amplitude_raw": amplitude_raw,
                "Power_raw": amplitude_raw ** 2,
                "snr_db": float(snr_db),
                "prominence_db": float(prominence_db),
                "local_peak_valid": bool(local_peak_valid),
                "candidate_status": str(status),
                "include_for_density": bool(include),
                "Magnitude (dB)": float(
                    peak_refinement.get(
                        "peak_magnitude_db",
                        20.0 * float(np.log10(max(amplitude_raw, 1e-12))),
                    )
                ),
            }
        )
        return row

    # ---------------------------------------------------------------------
    # AUDIT FIX (harmonic over-classification) — SNR-gated fallback helper.
    # ---------------------------------------------------------------------
    def _accept_harmonic_via_snr_gated_fallback(
        self,
        *,
        ef: float,
        tol_hz: float,
        complete_magnitudes,
        complete_freqs,
        harmonic_list: list,
        hnum: int,
        snr_threshold_db: float = 3.0,
    ) -> bool:
        """Attempt to accept a harmonic order from ``complete_list_df`` when
        the strict ``_is_local_peak_valid`` check failed (or there was no
        candidate in ``filtered_list_df``).

        Acceptance criteria (audit-canonical):
        * candidate frequency lies inside an expanded tolerance window
          (``1.5 × tol_hz``);
        * the candidate's bin in ``complete_magnitudes`` is a local maximum
          (peak > both immediate neighbours);
        * SNR vs. the local noise floor (15th percentile of a ±50-bin
          window) is ≥ ``snr_threshold_db``.

        The strict 3 dB-above-neighbours requirement is INTENTIONALLY
        relaxed here: on windowed FFTs the bins adjacent to the true
        peak sit within 1–2 dB of it (main-lobe smearing), so requiring
        3 dB margin rejects legitimate harmonics. SNR vs. the local
        noise floor still keeps unsupported "nearest-bin in empty
        window" candidates out.

        Returns True when a candidate was accepted (and appended to
        ``harmonic_list``), False otherwise.
        """
        if (
            getattr(self, "complete_list_df", None) is None
            or self.complete_list_df.empty
            or complete_magnitudes is None
            or complete_freqs is None
        ):
            return False
        wtol = float(tol_hz) * 1.5
        cand2 = self.complete_list_df[
            (self.complete_list_df['Frequency (Hz)'] >= ef - wtol) &
            (self.complete_list_df['Frequency (Hz)'] <= ef + wtol)
        ]
        if cand2.empty:
            return False
        cand2_copy = cand2.copy()
        cand2_copy['FreqError'] = abs(cand2_copy['Frequency (Hz)'] - ef)
        best_idx2 = cand2_copy['FreqError'].idxmin()
        best_fb = cand2.loc[best_idx2].copy()
        f2 = float(best_fb['Frequency (Hz)'])
        freq_idx2 = int(np.argmin(np.abs(complete_freqs - f2)))

        if freq_idx2 <= 0 or freq_idx2 >= len(complete_magnitudes) - 1:
            return False
        mags = np.asarray(complete_magnitudes, dtype=float)
        # Snap to the actual local maximum within ±2 bins. ``argmin(|f-ef|)``
        # generally lands on the bin nearest to the expected frequency, but
        # the true FFT peak can sit 1-2 bins away due to sub-bin offset. The
        # legacy ``peak > immediate-neighbour`` test then failed for almost
        # every fallback candidate.
        freq_idx2 = _refine_peak_index(mags, freq_idx2, refine_radius=2)
        if freq_idx2 <= 0 or freq_idx2 >= len(mags) - 1:
            return False
        log_mags = 20.0 * np.log10(np.maximum(mags, 1e-10))
        peak_db = float(log_mags[freq_idx2])
        left_db = float(log_mags[freq_idx2 - 1])
        right_db = float(log_mags[freq_idx2 + 1])
        is_local_max = (peak_db > left_db) and (peak_db > right_db)
        # Local noise floor (15th percentile over ±50-bin window).
        ws = 50
        s = max(0, freq_idx2 - ws)
        e = min(len(mags), freq_idx2 + ws)
        local_mags = mags[s:e]
        if local_mags.size == 0:
            return False
        nf_mag = float(np.percentile(local_mags, 15.0))
        nf_db = 20.0 * np.log10(max(nf_mag, 1e-10))
        snr2 = peak_db - nf_db
        if not (is_local_max and snr2 >= float(snr_threshold_db)):
            return False
        # Reflect the refinement back into the row so the harmonic_list
        # stores the actual peak frequency, not the candidate's snapped
        # bin. This also ensures the de-duplication check in the caller
        # (``abs(freq - existing.freq) < 0.1``) operates on the same
        # spectrum coordinate the strict path uses.
        peak_refinement_fb = _refine_candidate_to_interpolated_peak(
            candidate_freq_hz=float(complete_freqs[freq_idx2]),
            complete_magnitudes=mags,
            complete_freqs=complete_freqs,
            refine_radius=0,
        )
        best_fb["bin_center_frequency_hz"] = float(
            peak_refinement_fb["bin_center_frequency_hz"]
        )
        best_fb["interpolated_frequency_hz"] = float(
            peak_refinement_fb["interpolated_frequency_hz"]
        )
        best_fb["subbin_offset_bins"] = float(
            peak_refinement_fb["subbin_offset_bins"]
        )
        best_fb["subbin_interpolation_valid"] = bool(
            peak_refinement_fb["subbin_interpolation_valid"]
        )
        _pbi_fb = peak_refinement_fb.get("peak_bin_index", float("nan"))
        best_fb["peak_bin_index"] = (
            int(_pbi_fb) if np.isfinite(_pbi_fb) else float("nan")
        )
        _f_extract_fb = (
            float(peak_refinement_fb["interpolated_frequency_hz"])
            if peak_refinement_fb["subbin_interpolation_valid"]
            else float(peak_refinement_fb["bin_center_frequency_hz"])
        )
        best_fb["Frequency (Hz)"] = _f_extract_fb

        if 'Amplitude' not in best_fb:
            best_fb['Amplitude'] = float(np.power(
                10.0, float(best_fb.get('Magnitude (dB)', -120.0)) / 20.0
            ))
        best_fb['Harmonic Number'] = hnum
        best_fb['SNR_dB'] = float(snr2)
        best_fb['SubBinCorrected'] = bool(
            peak_refinement_fb["subbin_interpolation_valid"]
        )
        exists = any(
            abs(r['Frequency (Hz)'] - _f_extract_fb) < 0.1
            for _, r in pd.DataFrame(harmonic_list).iterrows()
        ) if harmonic_list else False
        if exists:
            return False
        harmonic_list.append(best_fb)
        return True

    # ---------------------------------------------------------------------
    # SINGLE-PASS REFACTOR — canonical component / model weights helper.
    # ---------------------------------------------------------------------
    def _set_model_weights_from_current_component_energy(self) -> None:
        """
        Single source of truth for harmonic / inharmonic / sub-bass energies.

        Reads the energies already computed during spectral classification
        (``self.harmonic_energy_sum``, ``self.inharmonic_energy_sum``,
        ``self.subbass_energy_sum``) and writes:

        Energy semantics (audited)
        --------------------------
        All three input quantities are **true power / energy**, i.e. ``Σ A²``
        on linear amplitudes:

        * ``self.harmonic_energy_sum``   ← ``Σ harmonic_amps²``
          (see _calculate_metrics: ``h_energy = np.sum(np.square(harmonic_amps))``)
        * ``self.inharmonic_energy_sum`` ← ``Σ ih_amps_eff²``
          (same site: ``ih_energy = np.sum(np.square(ih_amps_eff))``)
        * ``self.subbass_energy_sum``    ← ``aggregate_low_frequency_residual_peak_power``
          which is documented in ``density.py`` to return ``A²`` aggregates,
          not raw amplitude sums.

        Raw amplitude sums (``Σ |A|``) are intentionally NOT used here and
        are kept under explicitly legacy names (e.g. ``self.linear_sum_amplitude_*``
        or ``legacy_*_amplitude_sum``) so they cannot be confused with the
        canonical energy partition.

        * canonical ``component_*`` ratios with denominator ``H + I + S``
          (intentionally three-way; documents acoustic partition);
        * binary model coefficients ``model_harmonic_weight`` /
          ``model_inharmonic_weight`` with denominator ``H + I`` (these are
          dimensionless coefficients used by dissonance / weighting models;
          their denominator is intentionally different from the three-way
          partition — see policy note);
        * legacy ``batch_*`` aliases assigned **from the canonical values**
          (never read from old Batch Excel files in this path);
        * provenance fields ``component_energy_denominator``,
          ``component_energy_method``, ``component_profile_source``.

        When ``self.auto_model_weights_from_analysis`` is True (the new
        default), ``self.harmonic_weight`` / ``self.inharmonic_weight`` are
        also overwritten with the freshly computed coefficients, so that the
        rest of the pipeline (dissonance models, downstream weighting) uses
        the current-audio-derived weights rather than externally injected
        Batch percentages.

        Failure modes
        -------------
        - Any of ``H``, ``I``, ``S`` missing (never computed): ``component_*`` and
          ``model_*`` export as NaN with ``not_computed`` / ``skipped_missing_required_columns``-style
          statuses; ``harmonic_weight`` / ``inharmonic_weight`` are not overwritten in auto-mode.
        - ``H + I + S == 0`` (all energies reported as finite zeros): ``component_*`` ratios are
          **NaN** (undefined partition) with ``undefined_zero_total_energy``.
        - ``H + I == 0`` but ``H+I+S>0`` (e.g. pure sub-bass): ``model_*`` export as **NaN** with
          ``fallback_equal_weights_zero_HI_energy`` while ``harmonic_weight`` / ``inharmonic_weight``
          still receive ``0.5`` / ``0.5`` as a documented algorithmic fallback for dissonance models.
        """
        def _to_non_negative(v: Any) -> Optional[float]:
            if v is None:
                return None
            try:
                x = float(v)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(x):
                return None
            return float(max(0.0, x))

        H = _to_non_negative(getattr(self, "harmonic_energy_sum", None))
        I = _to_non_negative(getattr(self, "inharmonic_energy_sum", None))
        S = _to_non_negative(getattr(self, "subbass_energy_sum", None))
        missing_energy = H is None or I is None or S is None

        self.model_weight_fallback_applied = False
        model_h_rt = 0.5
        model_i_rt = 0.5

        if missing_energy:
            self.component_energy_status = "skipped_missing_required_columns"
            comp_h = comp_i = comp_s = float("nan")
            cti = float("nan")
            Hn = In = Sn = 0.0
            self.model_weight_status = "not_computed"
            model_h = float("nan")
            model_i = float("nan")
        else:
            Hn, In, Sn = float(H), float(I), float(S)
            T = Hn + In + Sn
            if T > 1e-30:
                comp_h = Hn / T
                comp_i = In / T
                comp_s = Sn / T
                self.component_energy_status = "computed"
            else:
                comp_h = comp_i = comp_s = float("nan")
                self.component_energy_status = "undefined_zero_total_energy"
                try:
                    self.logger.warning(
                        "single_pass: H+I+S == 0; canonical component energy ratios undefined (NaN)."
                    )
                except Exception:
                    pass

            if math.isfinite(comp_i) and math.isfinite(comp_s):
                cti = float(comp_i + comp_s)
            else:
                cti = float("nan")

            HI = Hn + In
            if HI > 1e-30:
                model_h = Hn / HI
                model_i = In / HI
                self.model_weight_status = "computed"
                model_h_rt, model_i_rt = model_h, model_i
            else:
                model_h = float("nan")
                model_i = float("nan")
                model_h_rt = model_i_rt = 0.5
                self.model_weight_status = "fallback_equal_weights_zero_HI_energy"
                self.model_weight_fallback_applied = True
                try:
                    self.logger.warning(
                        "single_pass: H+I == 0; model_*_weight undefined for export (NaN); "
                        "dissonance runtime weights use 0.5/0.5 fallback (see model_weight_status)."
                    )
                except Exception:
                    pass

        # Provenance flag — readable by tests / external auditors to confirm
        # that the canonical fields below were derived from POWER (Σ A²) and
        # not from amplitude sums (Σ |A|).
        self.component_energy_quantity = "power_sum_amplitude_squared"

        self.component_harmonic_energy_ratio = comp_h
        self.component_inharmonic_energy_ratio = comp_i
        self.component_subbass_energy_ratio = comp_s
        self.component_total_inharmonic_energy_ratio = cti
        self.component_energy_denominator = "H+I+S"
        self.component_energy_method = "single_pass_proc_audio_energy"

        # AUDIT FIX (inharmonic-energy underestimation) — publish the
        # diffuse non-harmonic residual as a canonical-candidate ratio.
        # Denominator is intentionally ``H+I+S+residual`` (the total
        # surviving spectral energy after noise-floor rejection), so the
        # new ratios are bounded in [0, 1] and component_nonharmonic +
        # component_harmonic_over_extended = 1 by construction. This is
        # distinct from the historical ``H+I+S`` denominator used by the
        # original component_* triplet and is documented as such in
        # metrics_dictionary.json.
        _residual = float(max(0.0, float(getattr(self, "residual_noise_energy_sum", 0.0) or 0.0)))
        if missing_energy:
            self.component_residual_noise_energy_ratio = float("nan")
            self.component_nonharmonic_energy_ratio = float("nan")
        else:
            _Tn = Hn + In + Sn
            _T_ext = _Tn + _residual
            if _T_ext > 1e-30:
                self.component_residual_noise_energy_ratio = float(_residual / _T_ext)
                self.component_nonharmonic_energy_ratio = float((In + Sn + _residual) / _T_ext)
            else:
                self.component_residual_noise_energy_ratio = float("nan")
                self.component_nonharmonic_energy_ratio = float("nan")
        self.component_residual_energy_denominator = "H+I+S+residual"
        # Component-profile provenance. The canonical value is
        # ``current_analysis`` (computed from the current per-note spectrum).
        # ``current_analysis_legacy_weights`` is a back-compat marker emitted
        # only when the caller forwards externally supplied model weights
        # instead of letting the spectrum drive them.
        self.component_profile_source = (
            "current_analysis"
            if bool(getattr(self, "auto_model_weights_from_analysis", True))
            else "current_analysis_legacy_weights"
        )

        self.model_harmonic_weight = model_h
        self.model_inharmonic_weight = model_i
        # Denominators are intentionally different; documented for readers.
        self.model_weight_denominator = "harmonic_plus_inharmonic"

        # AUDIT FIX — canonical scalar derived from the freshly computed
        # energies. Kept in sync with metrics_dictionary.json (status="canonical",
        # derived_from=[harmonic_energy_sum, inharmonic_energy_sum]).
        if missing_energy:
            self.harmonic_inharmonic_ratio = float("nan")
        else:
            _eps_hi = 1e-12
            self.harmonic_inharmonic_ratio = float(Hn / max(In, _eps_hi))

        # Internal compatibility aliases (private, never exported in
        # current-analysis user-facing sheets). They mirror the canonical
        # ``component_*`` ratios so legacy in-memory consumers continue to
        # work without re-reading the per-note workbook.
        self.batch_harmonic_energy_ratio = self.component_harmonic_energy_ratio
        self.batch_inharmonic_energy_ratio = self.component_inharmonic_energy_ratio
        self.batch_subbass_energy_ratio = self.component_subbass_energy_ratio
        self.batch_total_inharmonic_energy_ratio = self.component_total_inharmonic_energy_ratio
        self.batch_energy_denominator = "harmonic_plus_inharmonic_plus_subbass"
        self.batch_energy_method = "current_analysis_proc_audio_energy"

        # When auto-mode is active, override the harmonic_weight /
        # inharmonic_weight used by downstream dissonance / weighting models.
        if bool(getattr(self, "auto_model_weights_from_analysis", True)):
            if not missing_energy:
                self.harmonic_weight = float(model_h_rt)
                self.inharmonic_weight = float(model_i_rt)
                # AUDIT FIX — provenance: when the single-pass helper actually
                # overwrites the placeholder weights, ``model_weights_source``
                # MUST report ``current_analysis``. The misleading
                # ``apply_filters_arguments`` label is only correct when
                # ``auto_model_weights_from_analysis=False`` *and* no rewrite
                # happened. We surface this as an instance attribute so the
                # downstream Excel export (Per_Note_Processing_Metadata,
                # Analysis_Metadata) can prefer it over ``_gwm``.
                self.model_weights_source = "current_analysis"
                self.component_profile_source = "current_analysis"
                try:
                    _gwm = getattr(self, "gui_weight_resolution_meta", None)
                    if isinstance(_gwm, dict):
                        _gwm["model_weights_source"] = "current_analysis"
                        _gwm["component_profile_source"] = "current_analysis"
                except Exception:
                    pass
                try:
                    if math.isfinite(model_h) and math.isfinite(model_i):
                        self.logger.info(
                            "current_analysis: final model weights derived from current "
                            "per-note spectrum (source=current_analysis): "
                            "H=%.4f I=%.4f S=%.4f -> "
                            "model_harmonic_weight=%.4f, model_inharmonic_weight=%.4f",
                            comp_h, comp_i, comp_s, model_h, model_i,
                        )
                    else:
                        self.logger.info(
                            "current_analysis: model weights undefined for export (NaN); "
                            "runtime dissonance weights=%.4f/%.4f (status=%s).",
                            model_h_rt,
                            model_i_rt,
                            getattr(self, "model_weight_status", ""),
                        )
                except Exception:
                    pass

    def _partial_metric_sums_for_metrics_export(
        self,
        h_df: pd.DataFrame,
        ih_df: pd.DataFrame,
        sub_df: pd.DataFrame,
    ) -> Tuple[float, float, float, float]:
        """
        H/I/S/Total sums for Metrics / compiled Density_Metrics.

        **Continuous** UI keys (``linear``, ``log``, ``sqrt``, …): each exported band is **one ΣA² scalar**
        from ``*_energy_sum``. Those columns are **always** aggregated with ``weight_key="linear"`` so they stay
        true energy partitions (comparable to ``harmonic_energy_ratio`` / pie / batch STFT proportions). Applying
        ``log`` / ``sqrt`` etc. to ΣA² scalars was misleading (sub-bass could look as large as harmonic).

        **Discrete** keys (``d3``/``d10``/``d17``/``d24``): native per-partial sums on export-aligned linear
        amplitudes + Hz (same ``weight_function`` as the UI).
        """
        wf_ui = self._normalize_weight_function_ui_key(getattr(self, "weight_function", None))
        if wf_ui not in DISCRETE_SPECTRAL_METRIC_KEYS:
            _h_en = float(max(0.0, float(getattr(self, "harmonic_energy_sum", 0.0) or 0.0)))
            _i_en = float(max(0.0, float(getattr(self, "inharmonic_energy_sum", 0.0) or 0.0)))
            _s_en = float(max(0.0, float(getattr(self, "subbass_energy_sum", 0.0) or 0.0)))
            return partial_metric_sums_h_i_s_total(
                np.array([_h_en], dtype=float),
                np.array([_i_en], dtype=float),
                np.array([_s_en], dtype=float),
                "linear",
                harmonic_frequencies_hz=None,
                inharmonic_frequencies_hz=None,
                subbass_frequencies_hz=None,
            )

        def _af(dfx: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
            if dfx is None or getattr(dfx, "empty", True) or "Amplitude" not in dfx.columns:
                return np.array([], dtype=float), np.array([], dtype=float)
            if "Frequency (Hz)" in dfx.columns:
                fq_col = "Frequency (Hz)"
            elif "Frequency" in dfx.columns:
                fq_col = "Frequency"
            else:
                return np.array([], dtype=float), np.array([], dtype=float)
            a = pd.to_numeric(dfx["Amplitude"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            f = pd.to_numeric(dfx[fq_col], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(a) & np.isfinite(f) & (a >= 0.0)
            return a[m], f[m]

        ah, fh = _af(h_df)
        ai, fi = _af(ih_df)
        asb, fsb = _af(sub_df)
        return partial_metric_sums_h_i_s_total(
            ah,
            ai,
            asb,
            wf_ui,
            harmonic_frequencies_hz=fh if fh.size else None,
            inharmonic_frequencies_hz=fi if fi.size else None,
            subbass_frequencies_hz=fsb if fsb.size else None,
        )

    def _apply_density_metric_sdm_vector(
        self,
        camp_vec: np.ndarray,
        f_hz_vec: Optional[np.ndarray],
        cpow_vec: np.ndarray,
    ) -> float:
        """Apply UI ``weight_function`` to SDM legacy / fallback vectors (amplitudes + Hz for discrete keys)."""
        wf_ui = self._normalize_weight_function_ui_key(getattr(self, "weight_function", None))
        camp_vec = np.asarray(camp_vec, dtype=float).reshape(-1)
        cpow_vec = np.asarray(cpow_vec, dtype=float).reshape(-1)
        if wf_ui in DISCRETE_SPECTRAL_METRIC_KEYS:
            f_arr = np.asarray(f_hz_vec, dtype=float).reshape(-1) if f_hz_vec is not None else None
            if f_arr is not None and f_arr.size == camp_vec.size:
                return float(
                    apply_density_metric(
                        camp_vec,
                        weight_function=wf_ui,
                        normalize=False,
                        frequencies=f_arr,
                    )
                )
            return float(apply_density_metric(camp_vec, weight_function=wf_ui, normalize=False))
        return float(apply_density_metric(cpow_vec, weight_function=wf_ui, normalize=False))

    def _calculate_metrics(self) -> None:
        """
        Calcula as mÃ©tricas principais (modo FFT) a partir das listas harmÃ³nica/filtrada/completa.
        VersÃ£o robusta em escala absoluta:
          - corrige amplitudes pelo ganho coerente da janela;
          - DM e FDM por soma ponderada de amplitudes (normalize=False);
          - SDM a partir de potÃªncia (sem fallback DM*10);
          - mantÃ©m Combined e Total Metric como no design original.
        
        NOTA IMPORTANTE: Normalização por N_FFT
        =========================================
        Spectral Density Metric e R_norm são normalizados por N_FFT para permitir comparações
        entre diferentes tamanhos de FFT. A normalização usa N_FFT=1024 como referência.
        
        Sem normalização:
        - Valores com N_FFT=2048 seriam ~2x maiores que N_FFT=1024
        - Valores com N_FFT=512 seriam ~0.5x menores que N_FFT=1024
        
        Com normalização:
        - Todos os valores são normalizados para N_FFT=1024 equivalente
        - Permite comparações diretas entre diferentes configurações de FFT
        - Fórmula: normalized_value = original_value / (n_fft / 1024.0)
        
        Validação de valores extremos:
        - Spectral Density Metric > 200.0 gera warning
        - Pode indicar ruído, artefatos ou problemas de filtragem
        - Verificar filtragem de noise floor e spectral masking se valores extremos aparecerem
        """
        try:
            import numpy as np

            # --- helper local (auto-contido; remove se jÃ¡ tiveres versÃ£o global) ---
            def _coherent_gain_local(win: str, n_fft: int) -> float:
                """Ganho coerente da janela: G = (1/N) * sum w[n]."""
                try:
                    import numpy as _np
                    try:
                        from scipy.signal import windows as _win
                        wname = (win or "").lower()
                        if wname in ("flattop", "flat-top", "flat_top"):
                            w = _win.flattop(n_fft, sym=False)
                        elif wname in ("blackmanharris", "blackmanharris", "bh92", "bh-92"):
                            w = _win.blackmanharris(n_fft, sym=False)
                        elif wname in ("hann", "hanning"):
                            w = _win.hann(n_fft, sym=False)
                        elif wname in ("hamming",):
                            w = _win.hamming(n_fft, sym=False)
                        else:
                            w = _win.hann(n_fft, sym=False)
                    except Exception:
                        # fallback sem SciPy
                        wname = (win or "").lower()
                        if wname in ("hann", "hanning"):
                            w = _np.hanning(n_fft)
                        elif wname in ("hamming",):
                            w = _np.hamming(n_fft)
                        else:
                            w = _np.hanning(n_fft)
                    return float(_np.sum(w) / float(n_fft))
                except Exception:
                    return 1.0

            # ------------------- validaÃ§Ã£o de entradas -------------------
            if self.harmonic_list_df is None or self.complete_list_df is None:
                self._set_default_metrics()
                return
            if self.harmonic_list_df.empty:
                self._set_default_metrics()
                return

            # ------------------- amplitudes harmónicas -------------------
            harmonic_amps = np.asarray([], dtype=float)

            if self.harmonic_list_df is not None and not self.harmonic_list_df.empty:

                if "Amplitude" in self.harmonic_list_df.columns:
                    harmonic_amps = self.harmonic_list_df["Amplitude"].to_numpy(dtype=float)

                elif "Magnitude (dB)" in self.harmonic_list_df.columns:
                    db_vals = self.harmonic_list_df["Magnitude (dB)"].to_numpy(dtype=float)

                    # dB -> amplitude linear (referência A_ref = 1)
                    harmonic_amps = np.power(10.0, db_vals / 20.0)

                    # Guardar também no DF (útil para auditoria/export)
                    self.harmonic_list_df["Amplitude"] = harmonic_amps

                # Limpeza numérica: remove NaN/inf e força não-negatividade
                harmonic_amps = np.nan_to_num(harmonic_amps, nan=0.0, posinf=0.0, neginf=0.0)
                harmonic_amps = np.maximum(harmonic_amps, 0.0)

            # ------------------- Density Metric — legacy weighted partial activity (not SPL) ---
            # Effective multiplicity descriptor: see ``effective_partial_density`` (exported).
            # Note thickness/body is now represented by spectral_body_thickness_index family.
            # ACOUSTIC FIX: Account for natural frequency-dependent energy decay
            # This produces a smooth descending curve instead of irregular patterns
            # Higher frequencies naturally have less energy (spectral rolloff), so we normalize
            # by expected energy at each frequency to get consistent density values
            
            # Extract frequencies and fundamental for acoustic normalization
            harmonic_freqs = None
            fundamental_freq = None
            
            if self.harmonic_list_df is not None and not self.harmonic_list_df.empty:
                if "Frequency (Hz)" in self.harmonic_list_df.columns:
                    harmonic_freqs = self.harmonic_list_df["Frequency (Hz)"].to_numpy(dtype=float)

            fundamental_freq, _fund_src = self._canonical_f0_hz_for_analysis()
            if not np.isfinite(fundamental_freq) or fundamental_freq <= 0:
                fundamental_freq = None

            # Fallback: nominal from note name if canonical f0 unavailable
            if fundamental_freq is None or fundamental_freq <= 0:
                if hasattr(self, 'note') and self.note:
                    try:
                        fundamental_freq = float(self.calculate_fundamental_frequency(self.note) or 0.0)
                    except Exception:
                        fundamental_freq = None
                if fundamental_freq is not None and fundamental_freq <= 0:
                    fundamental_freq = None
            
            # Apply density metric with acoustic normalization
            self.density_metric_value = float(
                apply_density_metric(
                    harmonic_amps, 
                    self.weight_function,
                    frequencies=harmonic_freqs,
                    fundamental_freq=fundamental_freq,
                    account_for_spectral_rolloff=True  # Enable smooth descending curve
                )
            ) if harmonic_amps.size > 0 else 0.0

            # Scale to 0-10 range (matching older version)
            self.scaled_density_metric_value = self.density_metric_value * 10.0

            # Métricas discretas D2/D3/D8/D10/D17/D24 (definições fixas; export / menus da GUI)
            if harmonic_amps.size > 0:
                _disc = compute_discrete_spectral_metrics_bundle(harmonic_amps, harmonic_freqs)
                self.discrete_metric_d3 = _disc["discrete_metric_d3"]
                self.discrete_metric_d10 = _disc["discrete_metric_d10"]
                self.discrete_metric_d17 = _disc["discrete_metric_d17"]
                self.discrete_metric_d24 = _disc["discrete_metric_d24"]
            else:
                self.discrete_metric_d3 = float("nan")
                self.discrete_metric_d10 = float("nan")
                self.discrete_metric_d17 = float("nan")
                self.discrete_metric_d24 = float("nan")

            harmonic_count = len(harmonic_amps) if harmonic_amps.size > 0 else 0
            self.canonical_density_v5_adapted = float(self.density_metric_value)
            self.density_formula_version = CANONICAL_DENSITY_FORMULA_VERSION
            self.density_source_formula = CANONICAL_DENSITY_SOURCE_FORMULA
            self.density_normalization_scope = "none_per_note_absolute_canonical"
            self.density_normalization_denominator = float("nan")
            self.density_per_component = (
                float(self.canonical_density_v5_adapted / harmonic_count)
                if harmonic_count > 0
                else float("nan")
            )
            self.density_metric_per_harmonic = self.density_per_component
            self.density_metric_normalized = float("nan")
            self.density_metric_status = "computed" if harmonic_amps.size > 0 else "computed_zero"
            self.normalization_status = "not_applicable"

            if harmonic_amps.size > 0 and fundamental_freq is not None:
                if self.harmonic_list_df is not None and not self.harmonic_list_df.empty:
                    if "Amplitude" in self.harmonic_list_df.columns:
                        fundamental_amplitude = float(self.harmonic_list_df["Amplitude"].iloc[0])
                    elif harmonic_freqs is not None and len(harmonic_freqs) > 0:
                        fund_idx = np.argmin(harmonic_freqs)
                        fundamental_amplitude = float(harmonic_amps[fund_idx]) if fund_idx < len(harmonic_amps) else 1.0
                    else:
                        fundamental_amplitude = float(harmonic_amps[0]) if len(harmonic_amps) > 0 else 1.0
                elif harmonic_freqs is not None and len(harmonic_freqs) > 0:
                    fund_idx = np.argmin(harmonic_freqs)
                    fundamental_amplitude = float(harmonic_amps[fund_idx]) if fund_idx < len(harmonic_amps) else 1.0
                else:
                    fundamental_amplitude = float(harmonic_amps[0]) if len(harmonic_amps) > 0 else 1.0

                if fundamental_amplitude > 1e-10:
                    self.density_metric_ratio_over_fundamental_legacy = float(
                        self.density_metric_value / fundamental_amplitude
                    )
                else:
                    self.density_metric_ratio_over_fundamental_legacy = float("nan")
                    self.logger.warning(
                        f"Fundamental amplitude too small ({fundamental_amplitude:.2e}); "
                        f"legacy D/A1 ratio not calculated for note {getattr(self, 'note', 'unknown')}"
                    )
            else:
                self.density_metric_ratio_over_fundamental_legacy = float("nan")

            # ------------------- Spectral Density Metric (potência; com noise floor por bandas críticas e masking) -------------------
            self.spectral_density_metric_value = 0.0

            camp = None
            if self.complete_list_df is not None and not self.complete_list_df.empty:

                if "Amplitude" in self.complete_list_df.columns:
                    camp = self.complete_list_df["Amplitude"].to_numpy(dtype=float)

                elif "Magnitude (dB)" in self.complete_list_df.columns:
                    db_vals = self.complete_list_df["Magnitude (dB)"].to_numpy(dtype=float)

                    # dB -> amplitude linear (A_ref = 1)
                    camp = np.power(10.0, db_vals / 20.0)

                    # aplicar ganho coerente (se e só se a sua convenção for "amplitude corrigida pela janela")
                    cg = float(getattr(self, "coherent_gain_value", 1.0) or 1.0)
                    if cg > 0.0 and not getattr(self, "_complete_amp_corrected", False):
                        camp = camp / cg
                        self._complete_amp_corrected = True

                    # guardar uma vez
                    self.complete_list_df["Amplitude"] = camp

                # limpeza numérica (aplica-se a ambos os ramos)
                if camp is not None:
                    camp = np.nan_to_num(camp, nan=0.0, posinf=0.0, neginf=0.0)
                    camp = np.maximum(camp, 0.0)
                    # manter DF coerente, se existir coluna Amplitude
                    try:
                        self.complete_list_df["Amplitude"] = camp
                    except Exception:
                        pass

                #
                if camp is not None and camp.size > 0:
                    # Track fallback usage for reporting
                    self.sdm_fallback_used = False
                    # PHASE 4: Enhanced Spectral Density Metric with:
                    # 1. Noise floor estimation by critical bands
                    # 2. Spectral masking filter
                    # 3. Ground truth validation
                    
                    try:
                        from density import (
                            estimate_noise_floor_by_critical_bands,  # NOTE: Function name is legacy, but uses Hz bands internally
                            apply_spectral_masking_filter,
                            physical_spectral_density
                        )
                        # Alias for clarity (function already uses Hz bands, not Bark)
                        estimate_noise_floor_by_hz_bands = estimate_noise_floor_by_critical_bands
                        from constants import (
                            MASKING_ABSOLUTE_THRESHOLD_DB,
                            SMOOTHING_NOISE_FLOOR_PERCENTILE,
                            SMOOTHING_NOISE_FLOOR_MULTIPLIER
                        )
                        
                        # Get frequencies and magnitudes
                        if "Frequency (Hz)" in self.complete_list_df.columns:
                            f_hz = pd.to_numeric(
                                self.complete_list_df["Frequency (Hz)"], 
                                errors="coerce"
                            ).to_numpy(float)
                        else:
                            f_hz = None
                        
                        if "Magnitude (dB)" in self.complete_list_df.columns:
                            db_vals = self.complete_list_df["Magnitude (dB)"].to_numpy(dtype=float)
                        else:
                            # Convert amplitude to dB
                            db_vals = 20.0 * np.log10(np.maximum(camp, EPSILON_AMPLITUDE))
                        
                        # Filter out invalid values
                        valid_mask = (
                            np.isfinite(f_hz) & np.isfinite(db_vals) & np.isfinite(camp) &
                            (f_hz > 0) & (camp > 1e-12)
                        )
                        
                        # PHASE 4: Additional filtering - Remove noise below fundamental with adaptive safety margin
                        # This matches the pipeline orchestrator's adaptive HPF logic
                        # to prevent irregular Spectral Density Metric values from sub-fundamental noise
                        if np.sum(valid_mask) > 0:
                            # Try to get F0 from note name (if available)
                            f0_hz = None
                            note = getattr(self, "note", None)
                            if note:
                                try:
                                    f0_hz = float(self.calculate_fundamental_frequency(note) or 0.0)
                                except Exception:
                                    pass
                            
                            # If F0 is available, apply adaptive safety margin (same logic as orchestrator)
                            if f0_hz and f0_hz > 0:
                                # Adaptive margin by register (matching pipeline_orchestrator_gui.py:468-482)
                                if f0_hz < 60:
                                    margin_percent = 35.0  # High margin for unstable sub-bass
                                elif f0_hz < 120:
                                    margin_percent = 25.0  # Medium margin for bass
                                elif f0_hz < 300:
                                    margin_percent = 15.0  # Lower margin for low-mids
                                else:
                                    margin_percent = 10.0  # Default margin for higher registers
                                
                                # Calculate cutoff: F0 * (1 - margin%)
                                # This ensures we cut sub-fundamental noise but keep the fundamental intact
                                f0_cutoff = f0_hz * (1.0 - margin_percent / 100.0)
                                
                                # FIX: Use individual file cutoff only (don't override with orchestrator freq_min)
                                # The orchestrator freq_min is for initial broad filtering (line 1796),
                                # but for precise HPF per note, use each file's optimal cutoff based on its F0.
                                # This prevents lower notes from getting more aggressive filtering than ideal.
                                # Previous code: final_cutoff = max(f0_cutoff, user_freq_min)  # Caused override issue
                                final_cutoff = f0_cutoff
                                
                                # Count components before filtering (for logging)
                                n_before_f0_filter = np.sum(valid_mask)
                                
                                # Filter out components below cutoff
                                valid_mask = valid_mask & (f_hz >= final_cutoff)
                                
                                n_after_f0_filter = np.sum(valid_mask)
                                n_filtered_by_f0 = n_before_f0_filter - n_after_f0_filter
                                
                                # Log cutoff info (now using individual file optimal cutoff, no override)
                                self.logger.debug(
                                    f"Spectral Density Metric: F0={f0_hz:.1f} Hz, "
                                    f"margin={margin_percent:.1f}%, cutoff={final_cutoff:.1f} Hz (optimal, no override), "
                                    f"filtered {n_filtered_by_f0}/{n_before_f0_filter} components below fundamental"
                                )
                        
                        if np.sum(valid_mask) > 0:
                            f_hz_valid = f_hz[valid_mask]
                            db_vals_valid = db_vals[valid_mask]
                            camp_valid = camp[valid_mask]
                            
                            # 1. Estimate noise floor by Hz frequency bands (adaptive to signal level)
                            # CORRECTED: Use Hz bands instead of critical bands for physical-acoustic model
                            noise_floors_db = estimate_noise_floor_by_hz_bands(
                                f_hz_valid,
                                db_vals_valid,
                                noise_floor_percentile=SMOOTHING_NOISE_FLOOR_PERCENTILE,  # 15th percentile
                                noise_floor_multiplier=1.0  # apply adaptive multiplier below
                            )

                            # Adaptive thresholds by register (Hz bands)
                            # Lower registers need more permissive thresholds to preserve fundamentals.
                            freq_band_snr_db = np.where(
                                f_hz_valid < 120.0,
                                3.0,
                                np.where(f_hz_valid < 300.0, 6.0, float(SNR_THRESHOLD_DB))
                            )
                            freq_band_nf_mult = np.where(
                                f_hz_valid < 120.0,
                                1.0,
                                np.where(f_hz_valid < 300.0, 1.2, float(SMOOTHING_NOISE_FLOOR_MULTIPLIER))
                            )

                            # Apply adaptive multiplier to noise floor (per-frequency)
                            noise_floors_db = noise_floors_db * freq_band_nf_mult
                            
                            # ROBUSTNESS FIX: Explicit SNR threshold for consistent detection
                            # Combine adaptive threshold with absolute threshold (psychoacoustic limit)
                            # For fortissimo sounds, use a higher absolute threshold to filter more noise
                            # -80 dB is too low for fortissimo (almost everything passes)
                            # Use -60 dB as minimum for fortissimo, or adaptive based on signal level
                            max_db = np.max(db_vals_valid) if len(db_vals_valid) > 0 else 0.0
                            # Adaptive absolute threshold: higher for louder sounds
                            if max_db > -20.0:  # Fortissimo
                                absolute_threshold_db = -60.0  # More aggressive filtering
                            elif max_db > -40.0:  # Mezzo-forte
                                absolute_threshold_db = -70.0
                            else:  # Pianissimo
                                absolute_threshold_db = MASKING_ABSOLUTE_THRESHOLD_DB  # -80 dB (original)
                            
                            # ROBUSTNESS FIX: Apply explicit SNR threshold above noise floor
                            # This ensures consistent detection criterion across all tiers/N_FFT settings
                            # Formula: threshold = noise_floor + SNR_THRESHOLD_DB
                            snr_threshold_db = noise_floors_db + freq_band_snr_db
                            
                            # Use maximum of: SNR threshold, absolute threshold
                            # Note: noise_floors_db is already included in snr_threshold_db (snr_threshold_db = noise_floors_db + SNR_THRESHOLD_DB)
                            # Since SNR_THRESHOLD_DB > 0, snr_threshold_db > noise_floors_db always
                            # Therefore, we only need to compare SNR threshold and absolute threshold
                            threshold_db = np.maximum(
                                snr_threshold_db,  # Explicit SNR threshold above noise floor (includes noise_floor + 6.0 dB)
                                absolute_threshold_db  # Adaptive absolute threshold (psychoacoustic limit)
                            )
                            
                            # ROBUSTNESS FIX: Log explicit SNR threshold for transparency
                            note_name = getattr(self, "note", "unknown")
                            min_noise_floor = np.min(noise_floors_db) if len(noise_floors_db) > 0 else 0.0
                            max_noise_floor = np.max(noise_floors_db) if len(noise_floors_db) > 0 else 0.0
                            mean_noise_floor = np.mean(noise_floors_db) if len(noise_floors_db) > 0 else 0.0
                            mean_threshold = np.mean(threshold_db) if len(threshold_db) > 0 else 0.0
                            self.logger.debug(
                                f"Spectral Density Metric ({note_name}): "
                                f"max_db={max_db:.1f}, absolute_threshold={absolute_threshold_db:.1f}, "
                                f"noise_floor_range=[{min_noise_floor:.1f}, {max_noise_floor:.1f}], "
                                f"mean_noise_floor={mean_noise_floor:.1f}, SNR_threshold={SNR_THRESHOLD_DB:.1f}dB, "
                                f"mean_threshold={mean_threshold:.1f}, components_before={len(db_vals_valid)}"
                            )
                            
                            # Filter by noise floor threshold
                            noise_floor_mask = db_vals_valid >= threshold_db

                            # Fundamental safeguard: ensure f0 bin is kept if above absolute threshold
                            # This preserves the perceptual identity of low-register notes.
                            if f0_hz and f0_hz > 0 and noise_floor_mask.size > 0:
                                idx_f0 = int(np.argmin(np.abs(f_hz_valid - f0_hz)))
                                if np.isfinite(db_vals_valid[idx_f0]) and db_vals_valid[idx_f0] >= absolute_threshold_db:
                                    noise_floor_mask[idx_f0] = True
                            
                            if np.sum(noise_floor_mask) > 0:
                                f_hz_filtered = f_hz_valid[noise_floor_mask]
                                db_vals_filtered = db_vals_valid[noise_floor_mask]
                                camp_filtered = camp_valid[noise_floor_mask]
                                
                                # DIAGNOSTIC: Log filtering effectiveness
                                n_filtered = len(db_vals_valid) - len(db_vals_filtered)
                                filter_ratio = len(db_vals_filtered) / len(db_vals_valid) if len(db_vals_valid) > 0 else 0.0
                                self.logger.debug(
                                    f"Spectral Density Metric ({note_name}): "
                                    f"filtered {n_filtered}/{len(db_vals_valid)} components "
                                    f"({(1.0-filter_ratio)*100:.1f}% removed), "
                                    f"{len(db_vals_filtered)} remaining"
                                )
                                
                                # 2. Apply spectral masking filter (OPTIONAL - only if enabled)
                                # For physical-acoustic model, masking should be OFF (default)
                                # Masking removes components that exist physically but are "inaudible"
                                if getattr(self, "spectral_masking_enabled", False):
                                    f_hz_final, db_vals_final, camp_final, is_audible = apply_spectral_masking_filter(
                                        f_hz_filtered,
                                        db_vals_filtered,
                                        camp_filtered,
                                        mask_components=True  # Remove masked components
                                    )
                                    self.logger.debug("Spectral masking ENABLED (perceptual mode)")
                                else:
                                    # Physical mode: keep all components above noise floor
                                    f_hz_final = f_hz_filtered
                                    db_vals_final = db_vals_filtered
                                    camp_final = camp_filtered
                                    is_audible = np.ones(len(camp_filtered), dtype=bool)
                                    self.logger.debug("Spectral masking DISABLED (physical-acoustic mode)")
                                
                                if len(camp_final) > 0:
                                    # Calculate Spectral Density Metric from audible components
                                    # Use physical, bandwidth-normalized density (robust to FFT resolution)
                                    use_physical_sdm = True
                                    cpow = camp_final ** 2  # potência
                                    
                                    # DIAGNOSTIC: Check for potential issues before calculation
                                    note_name = getattr(self, "note", "unknown")
                                    n_components_original = len(camp) if camp is not None else 0
                                    n_components_final = len(camp_final)
                                    
                                    # Check for outliers in power values that could cause peaks
                                    if len(cpow) > 0:
                                        max_power = np.max(cpow)
                                        mean_power = np.mean(cpow)
                                        median_power = np.median(cpow)
                                        std_power = np.std(cpow)
                                        power_ratio = max_power / mean_power if mean_power > 0 else 0.0
                                        total_power = np.sum(cpow)
                                        
                                        # Log detailed power statistics for diagnosis
                                        self.logger.debug(
                                            f"Spectral Density Metric ({note_name}): power stats - "
                                            f"max={max_power:.2e}, mean={mean_power:.2e}, median={median_power:.2e}, "
                                            f"std={std_power:.2e}, total={total_power:.2e}, ratio={power_ratio:.1f}x"
                                        )
                                        
                                        # If there's a very large outlier, it could cause a peak
                                        if power_ratio > 100.0:  # One component has 100x more power than mean
                                            self.logger.warning(
                                                f"Spectral Density Metric: outlier detected for {note_name} "
                                                f"(max_power/mean_power={power_ratio:.1f}x). "
                                                f"This may cause a peak in the metric."
                                            )
                                    
                                    if use_physical_sdm:
                                        # Physical spectral density: integrates power over bandwidth and normalizes by range
                                        self.spectral_density_metric_value = float(
                                            physical_spectral_density(camp_final, f_hz_final)
                                        )
                                    else:
                                        # Legacy component-based density (kept for fallback)
                                        self.spectral_density_metric_value = self._apply_density_metric_sdm_vector(
                                            camp_final, f_hz_final, cpow
                                        )
                                        
                                        # Normalize by number of components to get "average density"
                                        if n_components_final > 0:
                                            original_sum = self.spectral_density_metric_value
                                            self.spectral_density_metric_value = self.spectral_density_metric_value / n_components_final
                                            self.logger.debug(
                                                f"Spectral Density Metric ({note_name}): "
                                                f"sum={original_sum:.2f}, components={n_components_final}, "
                                                f"average_density={self.spectral_density_metric_value:.4f}"
                                            )
                                    
                                    # NORMALIZATION: Remove N_FFT and hop_length dependency
                                    # Spectral Density Metric varies with:
                                    # 1. N_FFT: larger FFTs detect more frequency components (higher resolution)
                                    # 2. hop_length: smaller hops create more time frames (more temporal averaging)
                                    # 3. Resolution: Δf = sr / n_fft (smaller Δf = more components per Hz)
                                    #
                                    # Normalize to reference: N_FFT=1024, hop=128 (standard tier settings)
                                    # This ensures comparability across different tiers with different FFT settings
                                    # IMPORTANT: Use BASE N_FFT (before zero-padding), not padded N_FFT
                                    # Zero-padding increases frequency resolution but doesn't change the fundamental
                                    # number of components detected - it just interpolates between bins
                                    n_fft_base = getattr(self, "n_fft", 2048)  # Base N_FFT before zero-padding
                                    hop_length = getattr(self, "hop_length", n_fft_base // 8) or (n_fft_base // 8)
                                    sr = getattr(self, "sr", 44100) or 44100
                                    
                                    # Reference values (standard tier: N_FFT=1024, hop=128)
                                    ref_n_fft = 1024.0
                                    ref_hop = 128.0
                                    
                                    # Normalization factors:
                                    # 1. Frequency resolution: Δf_ref / Δf_actual = (sr/ref_n_fft) / (sr/n_fft) = n_fft / ref_n_fft
                                    # Use BASE N_FFT, not padded N_FFT, because zero-padding doesn't change fundamental resolution
                                    freq_resolution_factor = n_fft_base / ref_n_fft
                                    
                                    # 2. Temporal resolution: more frames = more averaging = higher values
                                    # Normalize by hop ratio: smaller hop = more frames = divide by more
                                    # hop_ratio = ref_hop / hop_length (if hop < ref_hop, ratio > 1, so we divide more)
                                    hop_ratio = ref_hop / hop_length if hop_length > 0 else 1.0
                                    
                                    # Combined normalization: accounts for both frequency and temporal resolution
                                    # The square root accounts for the fact that density scales with sqrt of resolution
                                    normalization_factor = np.sqrt(freq_resolution_factor * hop_ratio)
                                    
                                    if (not use_physical_sdm) and normalization_factor > 0 and self.spectral_density_metric_value > 0:
                                        original_value = self.spectral_density_metric_value
                                        self.spectral_density_metric_value = self.spectral_density_metric_value / normalization_factor
                                        # Get actual padded N_FFT for logging (informational only)
                                        n_fft_actual = self._get_actual_n_fft()
                                        note_name = getattr(self, "note", "unknown")
                                        
                                        # Enhanced logging for high values (especially first file)
                                        log_level = "WARNING" if original_value > 200.0 else "INFO"
                                        log_msg = (
                                            f"Spectral Density Metric: normalized by tier factor {normalization_factor:.3f} "
                                            f"(N_FFT_base={n_fft_base}, N_FFT_actual={n_fft_actual}, hop={hop_length}, "
                                            f"freq_res={freq_resolution_factor:.2f}x, hop_res={hop_ratio:.2f}x, "
                                            f"original={original_value:.2f}, normalized={self.spectral_density_metric_value:.2f}, "
                                            f"components: {n_components_original}→{n_components_final}, note={note_name})"
                                        )
                                        
                                        if original_value > 200.0:
                                            self.logger.warning(log_msg + " [HIGH VALUE - check filtering]")
                                        else:
                                            self.logger.info(log_msg)
                                    else:
                                        n_fft_actual = self._get_actual_n_fft()
                                        if bool(use_physical_sdm):
                                            self.logger.info(
                                                "Spectral Density Metric: tier N_FFT/hop scaling not applied on physical-SDM path "
                                                "(metric definition already comparable across tiers). "
                                                "N_FFT_base=%s, N_FFT_actual=%s, factor=%.3f, value=%s, components %s→%s.",
                                                n_fft_base,
                                                n_fft_actual,
                                                normalization_factor,
                                                self.spectral_density_metric_value,
                                                n_components_original,
                                                n_components_final,
                                            )
                                        elif self.spectral_density_metric_value <= 0:
                                            self.logger.info(
                                                "Spectral Density Metric: no positive value after filtering; "
                                                "N_FFT_base=%s, N_FFT_actual=%s.",
                                                n_fft_base,
                                                n_fft_actual,
                                            )
                                        else:
                                            self.logger.info(
                                                "Spectral Density Metric: tier scaling factor <= 0 or non-finite; "
                                                "leaving value unchanged. N_FFT_base=%s, N_FFT_actual=%s, factor=%.3f.",
                                                n_fft_base,
                                                n_fft_actual,
                                                normalization_factor,
                                            )
                                    
                                    # VALIDATION: Check for extreme values
                                    # Values > 200 may indicate noise, artifacts, or filtering issues
                                    # Also check if this is significantly higher than typical values
                                    if self.spectral_density_metric_value > 200.0:
                                        self.logger.warning(
                                            f"Spectral Density Metric very high ({self.spectral_density_metric_value:.2f}) "
                                            f"for note {note_name} (N_FFT_base={n_fft_base}). "
                                            f"May indicate noise or artefacts; check filtering. "
                                            f"Components: {n_components_original}->{n_components_final}"
                                        )
                                    elif self.spectral_density_metric_value > 100.0:
                                        # Log info for values > 100 (high but not extreme)
                                        self.logger.info(
                                            f"Spectral Density Metric high ({self.spectral_density_metric_value:.2f}) "
                                            f"for note {note_name} (N_FFT_base={n_fft_base}). "
                                            f"Components: {n_components_original}->{n_components_final}"
                                        )
                                    
                                    # 3. Ground truth validation
                                    try:
                                        from density import validate_spectral_density_metric
                                        
                                        # Validate against physical constraints
                                        validation_result = validate_spectral_density_metric(
                                            self.spectral_density_metric_value,
                                            f_hz_final,
                                            camp_final,
                                            expected_range=(0.0, 1000.0),  # Reasonable range for scaled values
                                            reference_value=None,  # Can be set if reference available
                                            tolerance=0.2
                                        )
                                        
                                        if not validation_result['is_valid']:
                                            self.logger.warning(
                                                f"Spectral Density Metric validation failed: "
                                                f"{'; '.join(validation_result['errors'])}"
                                            )
                                        
                                        if validation_result['warnings']:
                                            self.logger.debug(
                                                f"Spectral Density Metric warnings: "
                                                f"{'; '.join(validation_result['warnings'])}"
                                            )
                                        
                                        # Log validation details
                                        n_original = len(camp)
                                        n_after_noise_floor = np.sum(noise_floor_mask)
                                        n_after_masking = len(camp_final)
                                        
                                        self.logger.debug(
                                            f"Spectral Density Metric: "
                                            f"original={n_original}, "
                                            f"after noise floor={n_after_noise_floor}, "
                                            f"after masking={n_after_masking}, "
                                            f"value={self.spectral_density_metric_value:.2f}, "
                                            f"valid={validation_result['is_valid']}"
                                        )
                                    except ImportError:
                                        # Fallback if validation function not available
                                        n_original = len(camp)
                                        n_after_noise_floor = np.sum(noise_floor_mask)
                                        n_after_masking = len(camp_final)
                                        
                                        self.logger.debug(
                                            f"Spectral Density Metric: "
                                            f"original={n_original}, "
                                            f"after noise floor={n_after_noise_floor}, "
                                            f"after masking={n_after_masking}, "
                                            f"value={self.spectral_density_metric_value:.2f}"
                                        )
                                else:
                                    # All components were masked
                                    self.spectral_density_metric_value = 0.0
                                    self.logger.warning(
                                        "Spectral Density Metric: all components were masked or below noise floor"
                                    )
                            else:
                                # All components below noise floor
                                self.spectral_density_metric_value = 0.0
                                self.logger.warning(
                                    "Spectral Density Metric: all components below noise floor threshold"
                                )
                        else:
                            # No valid components
                            self.spectral_density_metric_value = 0.0
                            
                    except ImportError as e:
                        # Fallback to simple calculation if new functions not available
                        self.logger.warning(
                            f"Spectral Density Metric: enhanced filtering not available ({e}), "
                            f"using simple calculation"
                        )
                        self.sdm_fallback_used = True
                        cpow = camp ** 2  # potência
                        self.spectral_density_metric_value = self._apply_density_metric_sdm_vector(camp, f_hz, cpow)
                        
                        # CRITICAL FIX: Normalize by number of components (fallback path)
                        # Count components used in calculation
                        n_components_fallback = len(cpow) if cpow.size > 0 else 0
                        if n_components_fallback > 0:
                            original_sum = self.spectral_density_metric_value
                            self.spectral_density_metric_value = self.spectral_density_metric_value / n_components_fallback
                            self.logger.debug(
                                f"Spectral Density Metric (ImportError fallback): "
                                f"sum={original_sum:.2f}, components={n_components_fallback}, "
                                f"average_density={self.spectral_density_metric_value:.4f}"
                            )
                        
                        # NORMALIZATION: Remove N_FFT dependency (fallback path)
                        # IMPORTANT: Use BASE N_FFT (before zero-padding), not padded N_FFT
                        n_fft_base = getattr(self, "n_fft", 2048)  # Base N_FFT before zero-padding
                        hop_length = getattr(self, "hop_length", n_fft_base // 8) or (n_fft_base // 8)
                        
                        # Reference values (standard tier: N_FFT=1024, hop=128)
                        ref_n_fft = 1024.0
                        ref_hop = 128.0
                        
                        # Normalization factors (same as main path)
                        freq_resolution_factor = n_fft_base / ref_n_fft
                        hop_ratio = ref_hop / hop_length if hop_length > 0 else 1.0
                        normalization_factor = np.sqrt(freq_resolution_factor * hop_ratio)
                        if normalization_factor > 0 and self.spectral_density_metric_value > 0:
                            original_value = self.spectral_density_metric_value
                            self.spectral_density_metric_value = self.spectral_density_metric_value / normalization_factor
                            n_fft_actual = self._get_actual_n_fft()  # For logging only
                            self.logger.info(
                                f"Spectral Density Metric (error fallback): normalized by tier factor {normalization_factor:.3f} "
                                f"(N_FFT_base={n_fft_base}, N_FFT_actual={n_fft_actual}, hop={hop_length}, "
                                f"original={original_value:.2f}, normalized={self.spectral_density_metric_value:.2f})"
                            )
                        else:
                            n_fft_actual = self._get_actual_n_fft()
                            self.logger.info(
                                "Spectral Density Metric (error fallback): tier factor not applied "
                                "(value=%s, N_FFT_base=%s, N_FFT_actual=%s, factor=%.3f).",
                                self.spectral_density_metric_value,
                                n_fft_base,
                                n_fft_actual,
                                normalization_factor,
                            )
                        
                        # VALIDATION: Check for extreme values (fallback path)
                        if self.spectral_density_metric_value > 200.0:
                            note_name = getattr(self, "note", "unknown")
                            self.logger.warning(
                                f"Spectral Density Metric very high ({self.spectral_density_metric_value:.2f}) "
                                f"for note {note_name} (N_FFT_base={n_fft_base}, fallback path). "
                                f"May indicate noise or artefacts."
                            )
                    except Exception as e:
                        # Fallback on any error
                        self.logger.warning(
                            f"Spectral Density Metric: error in enhanced filtering ({e}), "
                            f"using simple calculation"
                        )
                        self.sdm_fallback_used = True
                        cpow = camp ** 2  # potência
                        self.spectral_density_metric_value = self._apply_density_metric_sdm_vector(camp, f_hz, cpow)
                        
                        # CRITICAL FIX: Normalize by number of components (error fallback path)
                        # Count components used in calculation
                        n_components_error_fallback = len(cpow) if cpow.size > 0 else 0
                        if n_components_error_fallback > 0:
                            original_sum = self.spectral_density_metric_value
                            self.spectral_density_metric_value = self.spectral_density_metric_value / n_components_error_fallback
                            self.logger.debug(
                                f"Spectral Density Metric (Exception fallback): "
                                f"sum={original_sum:.2f}, components={n_components_error_fallback}, "
                                f"average_density={self.spectral_density_metric_value:.4f}"
                            )
                        
                        # NORMALIZATION: Remove N_FFT dependency (error fallback path)
                        # IMPORTANT: Use BASE N_FFT (before zero-padding), not padded N_FFT
                        n_fft_base = getattr(self, "n_fft", 2048)  # Base N_FFT before zero-padding
                        hop_length = getattr(self, "hop_length", n_fft_base // 8) or (n_fft_base // 8)
                        
                        # Reference values (standard tier: N_FFT=1024, hop=128)
                        ref_n_fft = 1024.0
                        ref_hop = 128.0
                        
                        # Normalization factors (same as main path)
                        freq_resolution_factor = n_fft_base / ref_n_fft
                        hop_ratio = ref_hop / hop_length if hop_length > 0 else 1.0
                        normalization_factor = np.sqrt(freq_resolution_factor * hop_ratio)
                        if normalization_factor > 0 and self.spectral_density_metric_value > 0:
                            original_value = self.spectral_density_metric_value
                            self.spectral_density_metric_value = self.spectral_density_metric_value / normalization_factor
                            n_fft_actual = self._get_actual_n_fft()  # For logging only
                            self.logger.info(
                                f"Spectral Density Metric (error fallback): normalized by tier factor {normalization_factor:.3f} "
                                f"(N_FFT_base={n_fft_base}, N_FFT_actual={n_fft_actual}, hop={hop_length}, "
                                f"original={original_value:.2f}, normalized={self.spectral_density_metric_value:.2f})"
                            )
                        else:
                            n_fft_actual = self._get_actual_n_fft()
                            self.logger.info(
                                "Spectral Density Metric (error fallback): tier factor not applied "
                                "(value=%s, N_FFT_base=%s, N_FFT_actual=%s, factor=%.3f).",
                                self.spectral_density_metric_value,
                                n_fft_base,
                                n_fft_actual,
                                normalization_factor,
                            )
                        
                        # VALIDATION: Check for extreme values (error fallback path)
                        if self.spectral_density_metric_value > 200.0:
                            note_name = getattr(self, "note", "unknown")
                            self.logger.warning(
                                f"Spectral Density Metric very high ({self.spectral_density_metric_value:.2f}) "
                                f"for note {note_name} (N_FFT_base={n_fft_base}, error fallback path). "
                                f"May indicate noise or artefacts."
                            )
            # --- NOVO: métricas de densidade (R_norm, P_norm, D_agn, D_harm) ---
            try:
                if self.complete_list_df is not None and not self.complete_list_df.empty:
                    if "Frequency (Hz)" in self.complete_list_df.columns:
                        f_hz = pd.to_numeric(self.complete_list_df["Frequency (Hz)"], errors="coerce").to_numpy(float)
                    else:
                        f_hz = None

                    if "Amplitude" in self.complete_list_df.columns:
                        a_lin = pd.to_numeric(self.complete_list_df["Amplitude"], errors="coerce").to_numpy(float)
                    elif "Magnitude (dB)" in self.complete_list_df.columns:
                        # Amplitude linear (coerente com coluna Amplitude e com ``spectral_density``, que usa amps**gamma em amplitude)
                        a_lin = np.power(
                            10.0,
                            pd.to_numeric(self.complete_list_df["Magnitude (dB)"], errors="coerce").to_numpy(float) / 20.0,
                        )
                    else:
                        a_lin = None

                    if f_hz is not None and a_lin is not None:
                        # threshold relativo a -40 dB (~ 0.01 em amplitude; 1e-4 em potência)
                        if "Magnitude (dB)" in self.complete_list_df.columns:
                            mask = self.complete_list_df["Magnitude (dB)"].max() - 40.0
                            mask = self.complete_list_df["Magnitude (dB)"] >= mask
                            f_hz = f_hz[mask.to_numpy()]
                            a_lin = a_lin[mask.to_numpy()]

                        # Authoritative f0 path from acoustic core: never infer from
                        # lowest detected harmonic / peak rows.
                        f0_est = None
                        try:
                            _acc_raw = getattr(self, "f0_fit_accepted", False)
                            _acc = bool(_acc_raw is True or str(_acc_raw).strip().lower() in ("true", "1"))
                            _triplet = canonical_f0_triplet(
                                f0_final_hz=(getattr(self, "f0_final", None) if _acc else None),
                                f0_initial_hz=getattr(self, "f0_initial", None),
                                f0_prior_hz=getattr(self, "f0_prior_hz", None),
                                f0_fit_accepted=_acc,
                                f0_source=getattr(self, "f0_source", None),
                            )
                            if np.isfinite(_triplet.f0_hz) and _triplet.f0_hz > 0.0:
                                f0_est = float(_triplet.f0_hz)
                            self.f0_used_for_density_hz = (
                                float(_triplet.f0_hz) if np.isfinite(_triplet.f0_hz) else float("nan")
                            )
                            self.f0_used_for_density_source = str(_triplet.f0_source)
                            self.acoustic_f0_status = str(_triplet.acoustic_f0_status)

                            _peak_cols = [c for c in ("Frequency (Hz)", "Amplitude", "Magnitude (dB)", "Power") if c in self.complete_list_df.columns]
                            _peaks_df = self.complete_list_df[_peak_cols].copy()
                            _desc = compute_acoustic_density_descriptors(
                                _peaks_df,
                                f0_hz=float(_triplet.f0_hz),
                                f0_source=str(_triplet.f0_source),
                                acoustic_f0_status=str(_triplet.acoustic_f0_status),
                                f0_fit_accepted=bool(_triplet.f0_fit_accepted),
                                freq_min_hz=20.0,
                                freq_max_hz=float(getattr(self, "freq_max", 20000.0) or 20000.0),
                                density_summation_mode=str(
                                    getattr(self, "density_summation_mode", "his_weighted") or "his_weighted"
                                ),
                                harmonic_density_weight=float(
                                    getattr(self, "harmonic_density_weight", 1.0)
                                    if getattr(self, "harmonic_density_weight", 1.0) is not None
                                    else 1.0
                                ),
                                inharmonic_density_weight=float(
                                    getattr(self, "inharmonic_density_weight", 0.5)
                                    if getattr(self, "inharmonic_density_weight", 0.5) is not None
                                    else 0.5
                                ),
                                subbass_density_weight=float(
                                    getattr(self, "subbass_density_weight", 0.25)
                                    if getattr(self, "subbass_density_weight", 0.25) is not None
                                    else 0.25
                                ),
                                density_salience_threshold_db=float(
                                    getattr(self, "density_salience_threshold_db", -45.0)
                                    if getattr(self, "density_salience_threshold_db", -45.0) is not None
                                    else -45.0
                                ),
                                density_frequency_ceiling_hz=float(
                                    getattr(self, "density_frequency_ceiling_hz", 5000.0)
                                    if getattr(self, "density_frequency_ceiling_hz", 5000.0) is not None
                                    else 5000.0
                                ),
                            )
                            self._acoustic_density_desc = dict(_desc)
                            for _k, _v in _desc.items():
                                setattr(self, _k, _v)
                        except Exception:
                            f0_est = None
                            self._acoustic_density_desc = {}

                        from density import spectral_density
                        import numpy as np

                        # --- CORREÇÃO: CÁLCULO DINÂMICO DOS PESOS ---
                        # Lê os valores definidos na interface em vez de usar fixos
                        h_weight = float(getattr(self, "harmonic_weight", 0.95))
                        w_func = str(getattr(self, "weight_function", "linear")).lower()

                        if w_func == "log":
                            # Constant Power / Equal Power (Seno/Cosseno)
                            theta = h_weight * (np.pi / 2)
                            calc_wp = np.sin(theta)  # Peso Harmónico (Pitch)
                            calc_wr = np.cos(theta)  # Peso Inharmónico (Roughness)
                        else:
                            # Linear
                            calc_wp = h_weight
                            calc_wr = 1.0 - h_weight
                        # --------------------------------------------

                        dens = spectral_density(
                            f_hz, a_lin,
                            f0_hz=f0_est,
                            proximity_axis="hz",  # CHANGED: Use Hz (physical) instead of Bark (perceptual)
                            sigma=500.0,  # CHANGED: sigma in Hz (was 0.5 Bark ≈ 500 Hz at mid-frequencies)
                            hz_window=4000.0,  # CHANGED: window in Hz (was 8.0 Bark ≈ 4000 Hz)
                            max_peaks_per_band=4,
                            weight_r=calc_wr, weight_p=calc_wp, # <--- USAR VARIÁVEIS CALCULADAS
                            lambda_low=0.35,  # injeta "peso" de graves
                            low_hz_cut=1000.0,  # CHANGED: cutoff in Hz (was 8 Bark ≈ 1000 Hz)
                            weight_function=w_func,
                        )
                        # Exporta: D_peso (novo), além de R_norm/P_norm/D_agn
                        
                        # NORMALIZATION: Remove N_FFT and hop_length dependency from R_norm
                        # R_norm (richness) varies with:
                        # 1. N_FFT: larger FFTs detect more frequency components
                        # 2. hop_length: smaller hops create more time frames (more averaging)
                        # Normalize to reference: N_FFT=1024, hop=128 (standard tier settings)
                        # IMPORTANT: Use BASE N_FFT (before zero-padding), not padded N_FFT
                        # Zero-padding increases frequency resolution but doesn't change the fundamental
                        # number of components detected - it just interpolates between bins
                        n_fft_base = getattr(self, "n_fft", 2048)  # Base N_FFT before zero-padding
                        hop_length = getattr(self, "hop_length", n_fft_base // 8) or (n_fft_base // 8)
                        
                        # Reference values (standard tier: N_FFT=1024, hop=128)
                        ref_n_fft = 1024.0
                        ref_hop = 128.0
                        
                        # Normalization factors (same as Spectral Density Metric)
                        # Use BASE N_FFT, not padded N_FFT
                        freq_resolution_factor = n_fft_base / ref_n_fft
                        hop_ratio = ref_hop / hop_length if hop_length > 0 else 1.0
                        normalization_factor = np.sqrt(freq_resolution_factor * hop_ratio)
                        
                        # Guardar valores originais antes de normalizar
                        r_norm_original = float(dens["R_norm"])
                        p_norm_original = float(dens["P_norm"])
                        d_agn_original = float(dens["D_agn"])
                        
                        # Normalizar R_norm por N_FFT
                        if normalization_factor > 0 and r_norm_original > 0:
                            self.R_norm = r_norm_original / normalization_factor
                            # Só não-negatividade: ``spectral_density`` já devolve R_norm∈[0,1]; o factor de tier
                            # pode legitimar R>1 — cortar a 1.0 criava patamares artificiais em D_agn.
                            self.R_norm = max(0.0, float(self.R_norm))
                            # Get actual padded N_FFT for logging (informational only)
                            n_fft_actual = self._get_actual_n_fft()
                            self.logger.debug(
                                f"R_norm: normalized by tier factor {normalization_factor:.3f} "
                                f"(N_FFT_base={n_fft_base}, N_FFT_actual={n_fft_actual}, "
                                f"original={r_norm_original:.4f}, normalized={self.R_norm:.4f})"
                            )
                        else:
                            self.R_norm = r_norm_original
                        
                        # P_norm e D_agn não precisam normalização (já são normalizados 0-1 e não dependem diretamente de N_FFT)
                        # mas D_agn depende de R_norm, então precisa ser recalculado se R_norm foi normalizado
                        self.P_norm = p_norm_original
                        if normalization_factor > 0 and r_norm_original > 0:
                            # Recalcular D_agn com R_norm normalizado
                            # D_agn = weight_r * R_norm + weight_p * P_norm
                            calc_wr = float(getattr(self, "inharmonic_weight", 0.05))
                            calc_wp = float(getattr(self, "harmonic_weight", 0.95))
                            self.D_agn = calc_wr * self.R_norm + calc_wp * self.P_norm
                            self.logger.debug(
                                f"D_agn: recalculated with normalized R_norm "
                                f"(original={d_agn_original:.4f}, recalculated={self.D_agn:.4f})"
                            )
                        else:
                            self.D_agn = d_agn_original
                        
                        # Prefer D_harm from harmonic_list_df when available (partial-track estimate)
                        if dens["D_harm"] is not None:
                            self.D_harm = float(dens["D_harm"])
                        elif self.harmonic_list_df is not None and not self.harmonic_list_df.empty:
                            try:
                                if "Amplitude" in self.harmonic_list_df.columns:
                                    harmonic_amps = pd.to_numeric(
                                        self.harmonic_list_df["Amplitude"], 
                                        errors="coerce"
                                    ).dropna().to_numpy(float)
                                    if harmonic_amps.size > 0:
                                        _wf_d = self._normalize_weight_function_ui_key(
                                            getattr(self, "weight_function", None)
                                        )
                                        _dens_kw: Dict[str, Any] = {
                                            "weight_function": _wf_d,
                                            "normalize": False,
                                            "remove_noise": False,
                                            "prevent_domination": True,
                                        }
                                        if _wf_d in DISCRETE_SPECTRAL_METRIC_KEYS and "Frequency (Hz)" in self.harmonic_list_df.columns:
                                            _fq_d = pd.to_numeric(
                                                self.harmonic_list_df["Frequency (Hz)"],
                                                errors="coerce",
                                            ).to_numpy(dtype=float)
                                            if _fq_d.size == harmonic_amps.size:
                                                _dens_kw["frequencies"] = _fq_d
                                        self.D_harm = float(apply_density_metric(harmonic_amps, **_dens_kw))
                                        # No clipping - D_harm can be > 1.0 for rich harmonic sounds
                                        # More harmonics = higher D_harm (preserves variation)
                                        # Only ensure non-negative
                                        self.D_harm = max(0.0, self.D_harm)
                                    else:
                                        self.D_harm = None
                                else:
                                    self.D_harm = None
                            except Exception as e:
                                self.logger.debug(f"Error calculating D_harm from harmonic_list_df: {e}")
                                self.D_harm = None
                        else:
                            self.D_harm = None

            except Exception as _e:
                self.R_norm = self.P_norm = self.D_agn = 0.0
                self.D_harm = None
            # --- FIM NOVO ---

            # ------------------- Filtered Density (absoluta; sem contagem) -------------------
            self.filtered_density_metric_value = 0.0
            if self.filtered_list_df is not None and not self.filtered_list_df.empty:
                if "Amplitude" in self.filtered_list_df.columns:
                    famps = self.filtered_list_df["Amplitude"].to_numpy(float)
                elif "Magnitude (dB)" in self.filtered_list_df.columns:
                    famps = np.power(10.0, self.filtered_list_df["Magnitude (dB)"].to_numpy(float) / 20.0)
                    try:
                        self.filtered_list_df["Amplitude"] = famps
                    except Exception:
                        pass
                else:
                    famps = None

                if famps is not None and famps.size > 0:
                    _wf_fd = str(self.weight_function or "linear").strip().lower()
                    _freq_fd = None
                    if _wf_fd == "d24" and "Frequency (Hz)" in self.filtered_list_df.columns:
                        _freq_fd = pd.to_numeric(
                            self.filtered_list_df["Frequency (Hz)"], errors="coerce"
                        ).to_numpy(dtype=float)
                    self.filtered_density_metric_value = float(
                        apply_density_metric(
                            famps, self.weight_function, normalize=False, frequencies=_freq_fd
                        )
                    )

            # ------------------- Effective partial density (primary) + energy accounting -------------------
            # Participation ratio on **powers** of detected peaks only (not FFT bins).
            # Use the **same** peak table as ``Inharmonic Spectrum`` export: filtered peaks when available,
            # otherwise ``complete_list_df`` (harmonic_list_df alone is not enough to classify inharmonic rows).
            ih_amps_eff = np.asarray([], dtype=float)
            ih_freqs_eff: Optional[np.ndarray] = None
            try:
                _peak_src_energy = (
                    self.filtered_list_df
                    if self.filtered_list_df is not None and not self.filtered_list_df.empty
                    else self.complete_list_df
                )
                _peak_for_density = self._dataframe_for_density_frequency_floor(_peak_src_energy)
                if (
                    _peak_for_density is not None
                    and not _peak_for_density.empty
                    and self.harmonic_list_df is not None
                    and not self.harmonic_list_df.empty
                ):
                    ih_df_eff = identify_nonharmonic_residual_rows(
                        self.harmonic_list_df,
                        _peak_for_density,
                        tolerance=0.02,
                        **self._spectral_leakage_guard_kwargs(),
                    )
                    # Demote the bin-level inharmonic dataframe to true
                    # spectral peaks before accumulating energy. Without
                    # this step, ``ih_amps_eff`` was the sum over every
                    # filtered bin between harmonics — i.e. the spectral
                    # background — which on a sustained clarinet note
                    # contains thousands of mid-level bins and inflates
                    # the inharmonic energy / amplitude sums into the
                    # acoustically absurd "I > H" regime (see audit on
                    # Clarinete_mf D3/D#3/E3).
                    if ih_df_eff is not None and not ih_df_eff.empty:
                        try:
                            ih_df_eff = self._select_nonharmonic_peak_candidates_from_residual_rows(ih_df_eff)
                        except Exception as _e_ih_pk:
                            self.logger.warning(
                                "Inharmonic peak filtering failed (%s); using bin-level data.",
                                _e_ih_pk,
                            )
                    if ih_df_eff is not None and not ih_df_eff.empty:
                        if "Amplitude" in ih_df_eff.columns:
                            ih_amps_eff = pd.to_numeric(ih_df_eff["Amplitude"], errors="coerce").to_numpy(dtype=float)
                        elif "Magnitude (dB)" in ih_df_eff.columns:
                            ih_amps_eff = np.power(
                                10.0,
                                pd.to_numeric(ih_df_eff["Magnitude (dB)"], errors="coerce").to_numpy(dtype=float) / 20.0,
                            )
                        ih_amps_eff = np.nan_to_num(ih_amps_eff, nan=0.0, posinf=0.0, neginf=0.0)
                        ih_amps_eff = np.maximum(ih_amps_eff, 0.0)
                        if "Frequency (Hz)" in ih_df_eff.columns:
                            ih_freqs_eff = pd.to_numeric(
                                ih_df_eff["Frequency (Hz)"], errors="coerce"
                            ).to_numpy(dtype=float)
            except Exception as _e_eff:
                self.logger.debug("Effective partial density: inharmonic list failed: %s", _e_eff)

            self._metrics_ih_amps_eff = np.asarray(ih_amps_eff, dtype=float).copy()
            self._metrics_ih_freqs_eff = (
                None
                if ih_freqs_eff is None
                else np.asarray(ih_freqs_eff, dtype=float).copy()
            )

            # AUDIT FIX (Fgt_pp finding C1) — align the harmonic-protection
            # population used by ``aggregate_low_frequency_residual_peak_power`` with
            # the population Stage 2 uses for ``D_H``: the union of the
            # strict-harmonic list and the wider Harmonic Spectrum
            # candidates where ``include_for_density == True``. Without
            # this, low harmonics that pass the density filter but fail
            # the strict SNR/prominence gate (typical at pp dynamic and
            # in the bottom register) get DOUBLE-COUNTED — once into
            # ``D_H`` via Harmonic Spectrum, again into
            # ``subbass_energy_sum`` because the 12 Hz protection band
            # never fires around them. The union is at least as protective
            # as the previous strict-only list, so it never regresses
            # notes whose strict list was complete.
            harm_protection_df = self._build_subbass_harmonic_protection_df()
            # AUDIT FIX (acoustic-physics, Clarinete_mf findings #1 + #2)
            # — derive the harmonic-protection tolerance from the active
            # FFT configuration so the protection band covers the
            # actual window main-lobe (the legacy 12 Hz floor is
            # narrower than every realistic window's main lobe and
            # lets fundamental leakage shoulders escape into D_S).
            # Apply the new sub-bass lower-frequency floor (30 Hz) to
            # exclude DC / sub-audible content. NOTE: the canonical
            # sample-rate attribute on ``Analyser`` is ``self.sr``
            # (librosa naming); ``self.sample_rate`` is only set on a
            # few code paths. Read both so we are robust to either.
            try:
                _sr_hz = float(
                    getattr(self, "sr", None)
                    or getattr(self, "sample_rate", 0.0)
                    or 0.0
                )
                _n_fft_eff = int(getattr(self, "n_fft", 0) or 0)
            except Exception:
                _sr_hz = 0.0
                _n_fft_eff = 0
            try:
                _tol_hz_eff = float(
                    compute_subbass_protection_tolerance_hz(_sr_hz, _n_fft_eff)
                )
            except Exception:
                _tol_hz_eff = 12.0
            self.subbass_protection_tolerance_hz = _tol_hz_eff
            self.harmonic_leakage_protection_hz = float(_tol_hz_eff)
            self.subbass_aggregate_lower_hz = float(SUBBASS_AGGREGATE_LOWER_HZ)
            try:
                gpow_eff = float(
                    aggregate_low_frequency_residual_peak_power(
                        self.complete_list_df,
                        harm_protection_df,
                        subbass_hz=float(getattr(self, "subbass_aggregate_hz", SUBBASS_AGGREGATE_CUTOFF_HZ)),
                        subbass_lower_hz=float(SUBBASS_AGGREGATE_LOWER_HZ),
                        freq_match_tol_hz=_tol_hz_eff,
                    )
                )
            except Exception as _e_sub:
                self.logger.debug("Sub-bass aggregate power failed: %s", _e_sub)
                gpow_eff = 0.0

            try:
                d_eff, _diag_eff = partial_density_effective_components_bundle(
                    harmonic_amplitudes=harmonic_amps,
                    inharmonic_amplitudes=ih_amps_eff,
                    ground_noise_power=gpow_eff,
                    inharmonic_mode="aggregate",
                    min_db_relative=-60.0,
                )
            except Exception as _e_bundle:
                self.logger.warning("Effective partial density bundle failed: %s", _e_bundle)
                self.effective_partial_density_status = "failed_exception"
                d_eff = float("nan")

            h_energy = float(np.sum(np.square(harmonic_amps))) if harmonic_amps.size else 0.0
            ih_energy = float(np.sum(np.square(ih_amps_eff))) if ih_amps_eff.size else 0.0
            sub_energy = float(max(0.0, gpow_eff))
            tot_energy = float(h_energy + ih_energy + sub_energy)

            sum_lin_h = float(np.sum(harmonic_amps)) if harmonic_amps.size else 0.0
            sum_lin_ih = float(np.sum(ih_amps_eff)) if ih_amps_eff.size else 0.0
            den_lin_hi = sum_lin_h + sum_lin_ih
            frac_lin_ih = float(sum_lin_ih / den_lin_hi) if den_lin_hi > 1e-30 else 0.0
            self.linear_sum_amplitude_harmonic = sum_lin_h
            self.linear_sum_amplitude_inharmonic_partial = sum_lin_ih
            self.linear_amplitude_fraction_inharmonic_of_HI = frac_lin_ih

            if np.isfinite(d_eff):
                self.effective_partial_density = float(d_eff)
                if getattr(self, "effective_partial_density_status", "") != "failed_exception":
                    self.effective_partial_density_status = "computed"
            else:
                self.effective_partial_density = float("nan")
                if getattr(self, "effective_partial_density_status", "") != "failed_exception":
                    self.effective_partial_density_status = "not_computed"
            self.partial_density_effective_components = self.effective_partial_density

            # SEMANTIC HARDENING — ``effective_partial_count`` is computed
            # explicitly here as the participation ratio (Hill q=2 / inverse
            # Herfindahl) over the **harmonic peaks only**:
            #     N_eff = (Σ P_i)² / Σ P_i²   with P_i = A_h_i²
            # This is **NOT** the same quantity as
            # ``effective_partial_density``, which is the blended bundle
            # over harmonic + aggregated inharmonic + sub-bass. Keeping both
            # as canonical metrics is intentional and audited; the
            # ``derived_from`` field in metrics_dictionary.json documents
            # the distinct inputs.
            try:
                _amp_h = np.asarray(harmonic_amps, dtype=float)
                _amp_h = _amp_h[np.isfinite(_amp_h) & (_amp_h > 0.0)]
                if _amp_h.size > 0:
                    _p_h = np.square(_amp_h)
                    _sp = float(np.sum(_p_h))
                    _spp = float(np.sum(_p_h * _p_h))
                    if _spp > 0.0:
                        self.effective_partial_count = float((_sp * _sp) / _spp)
                    else:
                        self.effective_partial_count = 0.0
                else:
                    self.effective_partial_count = 0.0
            except Exception as _e_epc:
                self.logger.debug("effective_partial_count computation failed: %s", _e_epc)
                self.effective_partial_count = 0.0
            self.harmonic_energy_sum = h_energy
            self.inharmonic_energy_sum = ih_energy
            self.subbass_energy_sum = sub_energy
            self.total_component_energy = tot_energy
            if tot_energy > 1e-30:
                self.harmonic_energy_ratio = float(h_energy / tot_energy)
                self.inharmonic_energy_ratio = float(ih_energy / tot_energy)
                self.subbass_energy_ratio = float(sub_energy / tot_energy)
            else:
                self.harmonic_energy_ratio = 0.0
                self.inharmonic_energy_ratio = 0.0
                self.subbass_energy_ratio = 0.0

            # AUDIT FIX (inharmonic-energy underestimation) — compute the
            # diffuse non-harmonic residual: spectral power that survived
            # the noise-floor rejection (filtered_list_df) but is NOT in
            # the accepted harmonic peaks, NOT in the discrete inharmonic
            # peaks, and NOT in the sub-bass region. Mathematically
            #
            #     residual = max(0, ΣA²|filtered  - (H + I + S))
            #
            # so that residual ⩾ 0 by construction even when round-off /
            # mask overlaps would otherwise produce a small negative
            # number. This is the "diffuse non-harmonic" bucket that the
            # legacy I-only accounting collapsed into noise, which made
            # ``component_inharmonic_energy_ratio`` artificially small on
            # signals with broadband residual (breath noise, bowing
            # noise, etc.).
            try:
                _e_filtered = 0.0
                if (
                    isinstance(getattr(self, "filtered_list_df", None), pd.DataFrame)
                    and not self.filtered_list_df.empty
                    and "Amplitude" in self.filtered_list_df.columns
                ):
                    _af = pd.to_numeric(
                        self.filtered_list_df["Amplitude"], errors="coerce"
                    ).fillna(0.0).to_numpy(dtype=float)
                    _af = _af[np.isfinite(_af)]
                    if _af.size > 0:
                        _e_filtered = float(np.sum(_af * _af))
                _residual_energy = float(
                    max(0.0, _e_filtered - (h_energy + ih_energy + sub_energy))
                )
                self.total_filtered_spectral_energy = float(_e_filtered)
                self.residual_noise_energy_sum = _residual_energy
            except Exception as _e_res:
                self.logger.debug(
                    "residual_noise_energy_sum computation failed: %s", _e_res
                )
                self.total_filtered_spectral_energy = 0.0
                self.residual_noise_energy_sum = 0.0

            # SINGLE-PASS REFACTOR — derive canonical ``component_*`` ratios and
            # ``model_*`` weights here (single source of truth). When
            # ``auto_model_weights_from_analysis`` is True (the canonical path)
            # this also overrides ``self.harmonic_weight`` / ``self.inharmonic_weight``
            # *before* the dissonance models / further metrics are computed.
            try:
                self._set_model_weights_from_current_component_energy()
            except Exception as _e_smw:
                self.logger.warning(
                    "single_pass: _set_model_weights_from_current_component_energy failed: %s",
                    _e_smw,
                )

            try:
                _eb = describe_component_energy_balance(
                    float(self.harmonic_energy_sum or 0.0),
                    float(self.inharmonic_energy_sum or 0.0),
                    float(self.subbass_energy_sum or 0.0),
                    float(self.total_component_energy or 0.0),
                    float(self.harmonic_energy_ratio or 0.0),
                    float(self.inharmonic_energy_ratio or 0.0),
                    float(self.subbass_energy_ratio or 0.0),
                )
                self.energy_denominator_description = str(_eb.get("energy_denominator_description"))
                self.energy_conservation_error = float(_eb.get("energy_conservation_error", 0.0))
                self.energy_conservation_status = str(_eb.get("energy_conservation_status", "unknown"))
            except Exception as _e_ea:
                self.logger.debug("Energy accounting audit failed: %s", _e_ea)
                self.energy_denominator_description = "not_available_at_compile_stage"
                self.energy_conservation_error = None
                self.energy_conservation_status = "not_available_at_compile_stage"

            # Residual-row counts on the full detected list (not "partial" counts; audit/debug only)
            ih_complete_df = pd.DataFrame()
            try:
                if (
                    self.complete_list_df is not None
                    and not self.complete_list_df.empty
                    and self.harmonic_list_df is not None
                    and not self.harmonic_list_df.empty
                ):
                    _comp_for_bins = self._dataframe_for_density_frequency_floor(self.complete_list_df)
                    if _comp_for_bins is not None and not _comp_for_bins.empty:
                        ih_complete_df = identify_nonharmonic_residual_rows(
                            self.harmonic_list_df,
                            _comp_for_bins,
                            tolerance=0.02,
                            **self._spectral_leakage_guard_kwargs(),
                        )
            except Exception as _e_bin:
                self.logger.debug("Inharmonic bin count (complete list) failed: %s", _e_bin)
            self.inharmonic_bin_count = int(len(ih_complete_df)) if ih_complete_df is not None else 0
            n_complete_rows = (
                int(len(self.complete_list_df))
                if self.complete_list_df is not None and not self.complete_list_df.empty
                else 0
            )
            self.residual_row_count = int(n_complete_rows)
            self.harmonic_bin_count = int(max(0, n_complete_rows - int(self.inharmonic_bin_count or 0)))
            try:
                if self.complete_list_df is not None and not self.complete_list_df.empty:
                    _fq = pd.to_numeric(self.complete_list_df["Frequency (Hz)"], errors="coerce")
                    _cut = float(getattr(self, "subbass_aggregate_hz", SUBBASS_AGGREGATE_CUTOFF_HZ))
                    self.subbass_bin_count = int((_fq < _cut).sum())
                else:
                    self.subbass_bin_count = 0
            except Exception:
                self.subbass_bin_count = 0

            # Peak-based harmonic / inharmonic / sub-bass counts (v7-style on peak list)
            f0_hz, f0_source, acoustic_f0_status = self._canonical_f0_triplet_for_analysis()
            self.f0_used_for_harmonic_validation_hz = (
                float(f0_hz) if np.isfinite(float(f0_hz)) else float("nan")
            )
            self.f0_used_for_harmonic_validation_source = str(f0_source)
            self.acoustic_f0_status = str(acoustic_f0_status)

            peaks_for_class = self.filtered_list_df
            if peaks_for_class is None or peaks_for_class.empty:
                peaks_for_class = self._dataframe_for_density_frequency_floor(self.complete_list_df)

            pc: Dict[str, Any] = {
                "peaklist_harmonic_window_candidate_count": 0,
                "peaklist_nonharmonic_window_candidate_count": 0,
                "peaklist_low_frequency_window_candidate_count": 0,
                "peaklist_total_window_candidate_count": 0,
                "classification_valid": False,
                "classification_semantics": (
                    "independent_peaklist_window_assignment; not part of residual-row hierarchy"
                ),
            }
            try:
                if np.isfinite(f0_hz) and f0_hz > 0.0 and peaks_for_class is not None and not peaks_for_class.empty:
                    pc = classify_peaks_harmonic_inharmonic_subbass_from_df(
                        peaks_for_class,
                        f0_hz,
                        subbass_cutoff_hz=float(getattr(self, "subbass_aggregate_hz", SUBBASS_AGGREGATE_CUTOFF_HZ)),
                        tolerance_cents=18.0,
                        max_freq_hz=float(getattr(self, "freq_max", 20000.0) or 20000.0),
                    )
            except Exception as _e_pc:
                self.logger.debug("Peak classification for counts failed: %s", _e_pc)

            self.peaklist_harmonic_window_candidate_count = int(
                pc.get("peaklist_harmonic_window_candidate_count", 0) or 0
            )
            self.peaklist_nonharmonic_window_candidate_count = int(
                pc.get("peaklist_nonharmonic_window_candidate_count", 0) or 0
            )
            self.peaklist_low_frequency_window_candidate_count = int(
                pc.get("peaklist_low_frequency_window_candidate_count", 0) or 0
            )
            self.peaklist_total_window_candidate_count = int(
                pc.get("peaklist_total_window_candidate_count", 0) or 0
            )
            self.harmonic_peak_candidate_count = int(self.peaklist_harmonic_window_candidate_count or 0)
            self.nonharmonic_peak_candidate_count = int(self.peaklist_nonharmonic_window_candidate_count or 0)
            self.low_frequency_peak_candidate_count = int(self.peaklist_low_frequency_window_candidate_count or 0)
            self.total_peak_candidate_count = int(self.peaklist_total_window_candidate_count or 0)
            self.harmonic_peak_count = int(self.peaklist_harmonic_window_candidate_count or 0)
            self.inharmonic_peak_count = int(self.peaklist_nonharmonic_window_candidate_count or 0)
            self.subbass_peak_count = int(self.peaklist_low_frequency_window_candidate_count or 0)
            self.total_detected_peak_count = int(self.peaklist_total_window_candidate_count or 0)
            if not bool(pc.get("classification_valid")) and harmonic_amps.size:
                self.harmonic_peak_candidate_count = int(harmonic_amps.size)
                self.harmonic_peak_count = int(self.harmonic_peak_candidate_count)
                self.peaklist_harmonic_window_candidate_count = int(harmonic_amps.size)

            self.harmonic_candidate_count = int(self.harmonic_peak_candidate_count or 0)
            self.subbass_candidate_count = int(self.low_frequency_peak_candidate_count or 0)
            self.total_spectral_candidate_count = int(self.total_peak_candidate_count or 0)

            _ih_id, _ih_rt = self._nonharmonic_residual_pipeline_dataframes()
            self._assign_hierarchical_residual_debug_counts(_ih_id, _ih_rt)
            self.inharmonic_candidate_count = int(self.retained_nonharmonic_peak_candidate_count or 0)

            self.debug_counts_status = "computed"

            self.accepted_inharmonic_peak_count = None
            self.accepted_inharmonic_partial_count = None

            # Deprecated internal names: legacy alias for candidate-slot counts (not harmonic_order_count)
            self.harmonic_partial_count = int(self.harmonic_peak_count or 0)
            self.inharmonic_partial_count = int(self.inharmonic_peak_count or 0)
            self.total_detected_partial_count = int(self.total_detected_peak_count or 0)

            self.harmonic_validation_report = None
            try:
                from harmonic_validation import validate_harmonic_series_matched

                _pool = self.filtered_list_df
                if _pool is None or _pool.empty:
                    _pool = self.complete_list_df
                if np.isfinite(f0_hz) and f0_hz > 0.0 and _pool is not None and not _pool.empty:
                    _f0_validate = float(f0_hz)
                    _sr_v = getattr(self, "sample_rate", None)
                    if _sr_v is None:
                        _sr_v = getattr(self, "sr", None)
                    try:
                        _sr_f = float(_sr_v) if _sr_v is not None else None
                        if _sr_f is not None and (not np.isfinite(_sr_f) or _sr_f <= 0):
                            _sr_f = None
                    except (TypeError, ValueError):
                        _sr_f = None
                    try:
                        _nfft_v = int(getattr(self, "n_fft", 0) or 0)
                        _nfft_v = _nfft_v if _nfft_v > 0 else None
                    except (TypeError, ValueError):
                        _nfft_v = None
                    _f_sub_val = getattr(self, "subbass_aggregate_hz", None)
                    try:
                        _f_sub = float(_f_sub_val) if _f_sub_val is not None else float(SUBBASS_AGGREGATE_CUTOFF_HZ)
                    except (TypeError, ValueError):
                        _f_sub = float(SUBBASS_AGGREGATE_CUTOFF_HZ)
                    _vr = validate_harmonic_series_matched(
                        _f0_validate,
                        _pool,
                        max_freq_hz=float(getattr(self, "freq_max", 20000.0) or 20000.0),
                        sample_rate=_sr_f,
                        n_fft=_nfft_v,
                        subbass_cutoff_hz=_f_sub,
                    )
                    _spc = int(
                        _vr.get("non_harmonic_candidate_count", _vr.get("inharmonic_candidate_count", 0)) or 0
                    )
                    self.harmonic_slot_expected_count = int(_vr.get("harmonic_slot_expected_count", 0) or 0)
                    self.harmonic_slot_matched_count = int(_vr.get("harmonic_slot_matched_count", 0) or 0)
                    self.harmonic_slot_missing_count = int(_vr.get("harmonic_slot_missing_count", 0) or 0)
                    try:
                        _f0_rep = float(getattr(self, "f0_final", float("nan")))
                    except (TypeError, ValueError):
                        _f0_rep = float("nan")
                    if not np.isfinite(_f0_rep) or _f0_rep <= 0.0:
                        _f0_rep = float(f0_hz) if np.isfinite(float(f0_hz)) else float("nan")

                    _res_vm = getattr(self, "f0_robust_residual_std", None)
                    try:
                        _res_vm_f = (
                            float(_res_vm)
                            if _res_vm is not None and np.isfinite(float(_res_vm))
                            else float("nan")
                        )
                    except (TypeError, ValueError):
                        _res_vm_f = float("nan")

                    self.harmonic_validation_report = {
                        "f0_estimated": _f0_rep,
                        "f0_source": str(getattr(self, "f0_final_source", "") or "unresolved"),
                        "f0_nominal_hz": (
                            float(x)
                            if (x := getattr(self, "f0_nominal_hz", None)) is not None
                            and np.isfinite(float(x))
                            else float("nan")
                        ),
                        "f0_final_hz": _f0_rep,
                        "f0_final_method": str(getattr(self, "f0_final_method", "") or ""),
                        "f0_final_source": str(getattr(self, "f0_final_source", "") or "unresolved"),
                        "f0_detuning_cents_from_nominal": getattr(
                            self, "f0_detuning_cents_from_nominal", None
                        ),
                        "f0_fit_accepted": bool(getattr(self, "f0_fit_accepted", False)),
                        "f0_fit_quality": getattr(self, "f0_fit_quality", None),
                        "f0_fit_residual_std_hz": _res_vm_f,
                        "f0_fit_rejection_reason": getattr(
                            self, "f0_fit_rejection_reason", None
                        ),
                        "f0_used_for_density_hz": getattr(
                            self, "f0_used_for_density_hz", float("nan")
                        ),
                        "f0_used_for_density_source": str(
                            getattr(self, "f0_used_for_density_source", "")
                        ),
                        "f0_used_for_harmonic_validation_hz": float(_f0_validate),
                        "f0_used_for_harmonic_validation_source": str(f0_source),
                        "acoustic_f0_status": str(acoustic_f0_status),
                        "harmonic_slot_expected_count": int(_vr.get("harmonic_slot_expected_count", 0) or 0),
                        "harmonic_slot_matched_count": int(_vr.get("harmonic_slot_matched_count", 0) or 0),
                        "harmonic_slot_missing_count": int(_vr.get("harmonic_slot_missing_count", 0) or 0),
                        "non_harmonic_candidate_count": _spc,
                        "unmatched_spectral_row_count": int(
                            _vr.get("unmatched_spectral_row_count", _spc) or 0
                        ),
                        "outside_harmonic_window_candidate_count": int(_spc),
                        "outside_harmonic_window_candidate_row_count": int(
                            _vr.get("outside_harmonic_window_candidate_row_count", _spc) or 0
                        ),
                        "outside_harmonic_window_peak_candidate_count": int(
                            _vr.get("outside_harmonic_window_peak_candidate_count", _spc) or 0
                        ),
                        "outside_harmonic_window_candidate_energy_ratio": float(
                            _vr.get(
                                "outside_harmonic_window_candidate_energy_ratio",
                                _vr.get("non_harmonic_candidate_energy_ratio", 0.0),
                            )
                            or 0.0
                        ),
                        "mean_abs_harmonic_deviation_cents": _vr.get("mean_abs_harmonic_deviation_cents"),
                        "median_abs_harmonic_deviation_cents": _vr.get("median_abs_harmonic_deviation_cents"),
                        "max_abs_harmonic_deviation_cents": _vr.get("max_abs_harmonic_deviation_cents"),
                        "rms_harmonic_deviation_cents": _vr.get("rms_harmonic_deviation_cents"),
                        "harmonic_validation_status": str(_vr.get("harmonic_validation_status", "unknown")),
                    }
                    for __k, __v in _vr.items():
                        if not isinstance(__k, str) or not __k.startswith("harmonic_alignment_"):
                            continue
                        if __k in (
                            "harmonic_alignment_matches",
                            "harmonic_alignment_non_harmonic_candidates_preview",
                            "harmonic_alignment_inharmonic_candidates_preview",
                        ):
                            continue
                        self.harmonic_validation_report[__k] = __v
            except Exception as _e_val:
                self.logger.debug("Harmonic validation skipped/failed: %s", _e_val)

            _f0_prior = getattr(self, "f0_prior_hz", None)
            try:
                self.f0_prior_available = bool(
                    _f0_prior is not None
                    and np.isfinite(float(_f0_prior))
                    and float(_f0_prior) > 0.0
                )
            except (TypeError, ValueError):
                self.f0_prior_available = False
            self.f0_blind_method = "not_available_at_compile_stage"

            try:
                if (
                    self.harmonic_list_df is not None
                    and not self.harmonic_list_df.empty
                    and "Harmonic Number" in self.harmonic_list_df.columns
                ):
                    self.unique_harmonic_order_count = int(
                        pd.to_numeric(self.harmonic_list_df["Harmonic Number"], errors="coerce").dropna().nunique()
                    )
                else:
                    self.unique_harmonic_order_count = None
            except Exception:
                self.unique_harmonic_order_count = None

            self.harmonic_order_count = self.unique_harmonic_order_count

            # AUDIT FIX — canonical ``harmonic_completeness`` is the ratio of
            # unique harmonic orders effectively detected to the maximum
            # number of harmonic orders that fit below ``freq_max`` for the
            # current f0. Capped at 1.0 to bound overshoot when the
            # detector emits more harmonics than the geometric bound (e.g.
            # under aggressive search bands).
            try:
                _uniq = int(self.unique_harmonic_order_count or 0)
                _f0_hc, _ = self._canonical_f0_hz_for_analysis()
                _fmax_hc = float(getattr(self, "freq_max", 20000.0) or 20000.0)
                if np.isfinite(_f0_hc) and _f0_hc > 0.0 and _fmax_hc > 0.0:
                    _expected = max(1, int(_fmax_hc // _f0_hc))
                    self.harmonic_completeness = float(min(1.0, _uniq / _expected))
                else:
                    self.harmonic_completeness = 0.0
            except Exception as _e_hc:
                self.logger.debug("harmonic_completeness computation failed: %s", _e_hc)
                self.harmonic_completeness = 0.0

            _ac_desc = getattr(self, "_acoustic_density_desc", {}) or {}
            if _ac_desc:
                self.harmonic_occupancy_ratio = float(_ac_desc.get("harmonic_occupancy_ratio", float("nan")))
                self.expected_harmonic_slot_count = int(_ac_desc.get("expected_harmonic_slot_count", 0) or 0)
                self.detected_harmonic_slot_count = int(_ac_desc.get("detected_harmonic_slot_count", 0) or 0)
                self.harmonic_occupancy_detected_order_count = int(
                    _ac_desc.get("detected_harmonic_slot_count", 0) or 0
                )
                self.harmonic_effective_partial_count = float(
                    _ac_desc.get("harmonic_effective_partial_count", float("nan"))
                )
                self.harmonic_effective_power_density_normalized = float(
                    _ac_desc.get("harmonic_effective_power_density_normalized", float("nan"))
                )
                self.residual_log_frequency_occupancy = float(
                    _ac_desc.get("residual_log_frequency_occupancy", float("nan"))
                )
                self.residual_energy_ratio = float(_ac_desc.get("residual_energy_ratio", float("nan")))
                self.subbass_energy_ratio = float(_ac_desc.get("subbass_energy_ratio", float("nan")))
                self.harmonic_energy_ratio = float(_ac_desc.get("harmonic_energy_ratio", float("nan")))
                self.spectral_entropy = float(_ac_desc.get("spectral_entropy", float("nan")))
                self.effective_partial_density = float(_ac_desc.get("effective_partial_density", float("nan")))
                self.energy_weighted_component_density_diagnostic = float(
                    _ac_desc.get("energy_weighted_component_density_diagnostic", float("nan"))
                )
                self.arithmetic_validation_status = str(
                    _ac_desc.get("arithmetic_validation_status", getattr(self, "arithmetic_validation_status", "passed"))
                )
                self.acoustic_validation_status = str(
                    _ac_desc.get("acoustic_validation_status", getattr(self, "acoustic_validation_status", "passed"))
                )
                self.harmonic_occupancy_status = "from_acoustic_density_core"
                self.residual_log_frequency_occupancy_status = "from_acoustic_density_core"
            else:
                try:
                    _fmax_occ = float(getattr(self, "freq_max", 20000.0) or 20000.0)
                    _occ = compute_harmonic_occupancy_ratio(
                        self.harmonic_list_df,
                        f0_hz=float(f0_hz),
                        max_frequency_hz=_fmax_occ,
                    )
                    self.harmonic_occupancy_ratio = float(
                        _occ.get("harmonic_occupancy_ratio", float("nan"))
                    )
                    self.expected_harmonic_slot_count = int(
                        _occ.get(
                            "expected_harmonic_slot_count",
                            compute_expected_harmonic_slot_count(float(f0_hz), _fmax_occ),
                        )
                        or 0
                    )
                    self.detected_harmonic_slot_count = int(
                        _occ.get("detected_harmonic_slot_count", 0) or 0
                    )
                    self.harmonic_occupancy_detected_order_count = int(
                        _occ.get("detected_harmonic_slot_count", 0) or 0
                    )
                    self.harmonic_occupancy_status = str(
                        _occ.get("harmonic_occupancy_status", "unknown")
                    )
                except Exception as _e_occ:
                    self.logger.debug("harmonic_occupancy_ratio computation failed: %s", _e_occ)
                    self.harmonic_occupancy_ratio = float("nan")
                    self.expected_harmonic_slot_count = 0
                    self.detected_harmonic_slot_count = 0
                    self.harmonic_occupancy_detected_order_count = 0
                    self.harmonic_occupancy_status = "failed_exception"

                try:
                    _res_occ = compute_residual_log_frequency_occupancy(
                        ih_complete_df,
                        min_frequency_hz=20.0,
                        max_frequency_hz=float(getattr(self, "freq_max", 20000.0) or 20000.0),
                        bins_per_octave=24,
                    )
                    self.residual_log_frequency_occupancy = float(
                        _res_occ.get("residual_log_frequency_occupancy", float("nan"))
                    )
                    self.residual_log_frequency_bin_count = int(
                        _res_occ.get("residual_log_frequency_bin_count", 0) or 0
                    )
                    self.residual_log_frequency_bin_total = int(
                        _res_occ.get("residual_log_frequency_bin_total", 0) or 0
                    )
                    self.residual_log_frequency_occupancy_status = str(
                        _res_occ.get("residual_log_frequency_occupancy_status", "unknown")
                    )
                except Exception as _e_res_occ:
                    self.logger.debug(
                        "residual_log_frequency_occupancy computation failed: %s", _e_res_occ
                    )
                    self.residual_log_frequency_occupancy = float("nan")
                    self.residual_log_frequency_bin_count = 0
                    self.residual_log_frequency_bin_total = 0
                    self.residual_log_frequency_occupancy_status = "failed_exception"

            # ------------------- Entropia espectral -------------------
            if harmonic_amps.size > 0:
                powers = harmonic_amps ** 2  # entropia sobre potÃªncia (normalizada internamente)
                self.entropy_spectral_value = float(compute_spectral_entropy(powers))
            else:
                self.entropy_spectral_value = 0.0

            # ------------------- Combined (H/IH) - Amplitude-weighted + inharmonic from partial list -------------------
            harm_density = float(self.density_metric_value if self.density_metric_value is not None else 0.0)
            inharm_density = float(self._compute_inharmonic_density_for_combined())

            # CRITICAL: Use logarithmic combination to preserve dynamic range
            # This ensures 'ff' (high amplitudes) produces substantially higher values than 'pp'
            # Formula: log(1 + energy) preserves relative differences while handling wide dynamic range
            try:
                import math
                # Apply logarithmic scaling to preserve dynamic range
                # Use log1p to handle small values gracefully: log(1 + x) ≈ x for small x
                harm_log = math.log1p(max(0.0, harm_density))
                inharm_log = math.log1p(max(0.0, inharm_density))
                
                # Weighted combination in log space (preserves relative magnitudes)
                combined_log = self.harmonic_weight * harm_log + self.inharmonic_weight * inharm_log
                
                # Convert back to linear scale: exp(x) - 1, but preserve the dynamic range
                # This maintains the absolute differences between 'pp' and 'ff'
                self.combined_density_metric_value = float(math.expm1(combined_log))
                
            except Exception as e:
                self.logger.warning(f"Logarithmic combination failed, using linear: {e}")
                # Fallback to linear combination (still amplitude-weighted, not count-based)
                # But use preserve_dynamic_range=False to get linear combination
                self.combined_density_metric_value = float(
                    calculate_combined_density_metric(
                        harm_density, inharm_density, 
                        self.harmonic_weight, self.inharmonic_weight,
                        preserve_dynamic_range=False  # Linear fallback
                    )
                )

            # PHASE 3: Use constants instead of magic numbers
            # Definir um limite empírico seguro para a Densidade Absoluta (DM/SDM).
            # Este valor deve ser ajustado com base na sua base de dados, mas MAX_ABS_DENSITY
            # é um valor seguro para métricas DM/SDM absolutas corrigidas (após Coherent Gain).
            # Se o seu sinal de entrada for normalizado para -20dB RMS, MAX_ABS_DENSITY deve ser um limite superior conservador.
            wD, wS, wE, wC = (DENSITY_METRIC_WEIGHT_D, DENSITY_METRIC_WEIGHT_S, 
                            DENSITY_METRIC_WEIGHT_E, DENSITY_METRIC_WEIGHT_C)

            # ------------------- Total Metric (PHASE 4: Robust Normalization) -------------------
            
            # PHASE 4: Use robust normalization instead of hardcoded limits
            # Validate metrics before normalization
            metrics_to_validate = [
                (self.scaled_density_metric_value, "Density Metric"),
                (self.spectral_density_metric_value, "Spectral Density Metric"),
                (self.entropy_spectral_value, "Spectral Entropy"),
                (self.combined_density_metric_value, "Combined Density Metric")
            ]
            
            for value, name in metrics_to_validate:
                if value is not None:
                    # UPDATED: Realistic maximums based on mathematical analysis
                    # Density Metric (scaled by 10.0): With frequency normalization (n^1.5 boost)
                    # and many harmonics (100+), values can legitimately reach 1000-2000
                    # Combined Metric: Logarithmic combination (expm1) can produce values
                    # up to 500-1000 for very rich sounds with many harmonics
                    if name == "Density Metric":
                        # Scaled density metric: base * 10.0, with frequency boost can reach 2000+
                        expected_range = (0.0, 2000.0)  # Realistic maximum for rich sounds
                    elif name == "Spectral Density Metric":
                        # Spectral density: normalized by component count, typically lower
                        expected_range = (0.0, 1000.0)  # Already validated at line 2255
                    elif name == "Combined Density Metric":
                        # Logarithmic combination: expm1(log(1+H) + log(1+I)) can exceed 100
                        expected_range = (0.0, 1000.0)  # Realistic maximum for rich sounds
                    else:
                        expected_range = (0.0, 100.0)  # Other metrics use original range
                    is_valid, error_msg = validate_metric_value(value, name, expected_range=expected_range)
                    if not is_valid:
                        # Only warn if significantly outside expected range (not just slightly over)
                        # Values slightly over are acceptable for rich sounds
                        if name == "Density Metric" and value > 2000.0:
                            self.logger.warning(f"Metric validation: {error_msg} (very high value, may indicate issue)")
                        elif name == "Combined Density Metric" and value > 1000.0:
                            self.logger.warning(f"Metric validation: {error_msg} (very high value, may indicate issue)")
                        else:
                            # Values in reasonable range for rich sounds - log as info, not warning
                            self.logger.debug(f"Metric value ({name}={value:.2f}) exceeds typical range but is acceptable for rich sounds")

            # 1. Normalização da Densidade (DM) - Restored from older version
            # Simple division by 10.0 to normalize from 0-10 range to 0-1
            norm_density = self.scaled_density_metric_value / 10.0 if self.scaled_density_metric_value is not None else 0.0

            # 2. Normalização da Densidade Espectral (SDM) - Restored from older version
            # Simple division by 10.0 to normalize from 0-10 range to 0-1
            norm_spectral = self.spectral_density_metric_value / 10.0 if self.spectral_density_metric_value is not None else 0.0

            # 3. Entropia Espectral (já é normalizada 0-1)
            norm_entropy = self.entropy_spectral_value or 0.0
            # PHASE 4: Validate instead of just clipping
            is_valid, error_msg = validate_metric_value(norm_entropy, "Entropy", expected_range=(0.0, 1.0))
            if not is_valid:
                self.logger.warning(f"Entropy validation: {error_msg}, clipping to [0, 1]")
                norm_entropy = np.clip(norm_entropy, 0.0, 1.0)

            # 4. Métrica Combinada - PHASE 4: Remove unnecessary clipping
            # Combined metric is already designed to preserve dynamic range (log-transform)
            # FIXED: Current version uses amplitude-weighted densities with log-transform,
            # which can produce values > 10.0 (e.g., 22.92). This is expected behavior.
            # The older version used count-based densities (0-1), but current preserves dynamic range.
            norm_combined = self.combined_density_metric_value or 0.0
            # PHASE 4: Use realistic range for amplitude-weighted + log-transform metrics
            # Values can exceed 10.0 due to log1p/expm1 transformation of large amplitude values
            # With frequency normalization and many harmonics, values can reach 100-1000
            # UPDATED: Increased range to (0.0, 1000.0) to accommodate rich sounds
            is_valid, error_msg = validate_metric_value(norm_combined, "Combined Metric", expected_range=(0.0, 1000.0))
            if not is_valid:
                # Only warn if significantly outside expected range
                if norm_combined > 1000.0:
                    self.logger.warning(f"Combined metric validation: {error_msg} (very high value, may indicate issue)")
                else:
                    self.logger.debug(f"Combined metric value ({norm_combined:.2f}) exceeds typical range but is acceptable for rich sounds")
                # Don't clip - preserve dynamic range for normalization later
                # Values will be normalized in compile_metrics.py using robust methods

            # PHASE 3: Use constant instead of magic number
            # Cálculo Final da Total Metric (mantém-se a multiplicação por TOTAL_METRIC_SCALE para escala 0-10)
            self.total_metric_value = (wD*norm_density + wS*norm_spectral + wE*norm_entropy + wC*norm_combined) * TOTAL_METRIC_SCALE


            # Dynamic Density Score (Acoustic Momentum): Structure (CDM) * Energy (Log FDM)
            try:
                import math
                if self.filtered_density_metric_value > 0:
                    self.dynamic_density_score = self.combined_density_metric_value * math.log10(self.filtered_density_metric_value)
                else:
                    self.dynamic_density_score = 0.0
            except Exception:
                self.dynamic_density_score = 0.0

            # --------------------
            # ------------------- DissonÃ¢ncia -------------------
            if getattr(self, "dissonance_enabled", False):
                self.calculate_dissonance_metrics()

        except Exception as e:
            self.logger.error(f"Critical error in _calculate_metrics: {e}", exc_info=True)
            self._set_default_metrics()

    def _compute_inharmonic_density_for_combined(self) -> float:
        """
        Inharmonic contribution for Combined Density Metric.

        Priority:
        1) ``identify_nonharmonic_residual_rows`` on the same peak table as energy metrics (cached vectors).
        2) Residual ``max(0, filtered_density - harmonic_density)`` (same ``apply_density_metric`` units).
        3) Fallback: density on complete-list rows farther than ~1 STFT bin from any harmonic frequency.
        """
        wf = str(getattr(self, "weight_function", "linear") or "linear").strip().lower()
        ih_amp = np.asarray(getattr(self, "_metrics_ih_amps_eff", []), dtype=float).reshape(-1)
        ih_fr = getattr(self, "_metrics_ih_freqs_eff", None)

        if ih_amp.size == 0:
            try:
                _peak_src = (
                    self.filtered_list_df
                    if self.filtered_list_df is not None and not self.filtered_list_df.empty
                    else self.complete_list_df
                )
                _peak_src = self._dataframe_for_density_frequency_floor(_peak_src)
                if (
                    _peak_src is not None
                    and not _peak_src.empty
                    and self.harmonic_list_df is not None
                    and not self.harmonic_list_df.empty
                ):
                    ih_df = identify_nonharmonic_residual_rows(
                        self.harmonic_list_df,
                        _peak_src,
                        tolerance=0.02,
                        **self._spectral_leakage_guard_kwargs(),
                    )
                    if ih_df is not None and not ih_df.empty:
                        if "Amplitude" in ih_df.columns:
                            ih_amp = pd.to_numeric(ih_df["Amplitude"], errors="coerce").to_numpy(dtype=float)
                        elif "Magnitude (dB)" in ih_df.columns:
                            ih_amp = np.power(
                                10.0,
                                pd.to_numeric(ih_df["Magnitude (dB)"], errors="coerce").to_numpy(dtype=float) / 20.0,
                            )
                        ih_amp = np.nan_to_num(ih_amp, nan=0.0, posinf=0.0, neginf=0.0)
                        ih_amp = np.maximum(ih_amp, 0.0)
                        if "Frequency (Hz)" in ih_df.columns:
                            ih_fr = pd.to_numeric(ih_df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
            except Exception as _e_onfly:
                self.logger.debug("_compute_inharmonic_density_for_combined on-the-fly partial list: %s", _e_onfly)

        if ih_amp.size > 0:
            freq_arg = None
            if wf == "d24" and ih_fr is not None:
                ih_fr_a = np.asarray(ih_fr, dtype=float).reshape(-1)
                if ih_fr_a.size == ih_amp.size:
                    freq_arg = ih_fr_a
            return float(
                apply_density_metric(
                    ih_amp, self.weight_function, normalize=False, frequencies=freq_arg
                )
            )

        fd = float(self.filtered_density_metric_value or 0.0)
        hd = float(self.density_metric_value or 0.0)
        if fd > 0.0:
            return max(0.0, fd - hd)

        if (
            self.complete_list_df is None
            or self.complete_list_df.empty
            or self.harmonic_list_df is None
            or self.harmonic_list_df.empty
            or "Frequency (Hz)" not in self.complete_list_df.columns
            or "Frequency (Hz)" not in self.harmonic_list_df.columns
        ):
            return 0.0

        try:
            compf = pd.to_numeric(self.complete_list_df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
            harf = pd.to_numeric(self.harmonic_list_df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
            harf = harf[np.isfinite(harf)]
            zp = max(1, int(getattr(self, "zero_padding", 1) or 1))
            nfb = max(1, int(getattr(self, "n_fft", 2048) or 2048))
            sr_v = float(getattr(self, "sr", 44100.0) or 44100.0)
            tol_hz = max(0.5, sr_v / float(nfb * zp))
            if harf.size == 0:
                mask_inharm = np.ones(compf.shape[0], dtype=bool)
            else:
                dmin = np.min(np.abs(compf[:, None] - harf[None, :]), axis=1)
                mask_inharm = np.isfinite(dmin) & (dmin > tol_hz)
            sub = self.complete_list_df.loc[mask_inharm]
            if sub.empty:
                return 0.0
            if "Amplitude" in sub.columns:
                inharm_amps = pd.to_numeric(sub["Amplitude"], errors="coerce").to_numpy(dtype=float)
            elif "Magnitude (dB)" in sub.columns:
                inharm_amps = np.power(
                    10.0, pd.to_numeric(sub["Magnitude (dB)"], errors="coerce").to_numpy(dtype=float) / 20.0
                )
            else:
                return 0.0
            inharm_amps = np.nan_to_num(inharm_amps, nan=0.0, posinf=0.0, neginf=0.0)
            inharm_amps = np.maximum(inharm_amps, 0.0)
            if inharm_amps.size == 0:
                return 0.0
            _wf_i = self._normalize_weight_function_ui_key(getattr(self, "weight_function", None))
            inharm_freqs = pd.to_numeric(sub["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
            if _wf_i in DISCRETE_SPECTRAL_METRIC_KEYS and inharm_freqs.size == inharm_amps.size:
                return float(
                    apply_density_metric(
                        inharm_amps,
                        weight_function=_wf_i,
                        normalize=False,
                        frequencies=inharm_freqs,
                    )
                )
            inharm_power = inharm_amps ** 2
            return float(apply_density_metric(inharm_power, weight_function=_wf_i, normalize=False))
        except Exception as e:
            self.logger.warning("Inharmonic density (complete-list fallback) failed: %s", e)
            return 0.0

    # ----------------- defaults mÃ©tricas -----------------
    def _set_default_metrics(self):
        self.density_metric_value = 0.0
        self.scaled_density_metric_value = 0.0
        self.filtered_density_metric_value = 0.0
        self.entropy_spectral_value = 0.0
        self.combined_density_metric_value = 0.0
        self.total_metric_value = 0.0
        self.spectral_density_metric_value = 0.0
        self.effective_partial_density = None
        self.partial_density_effective_components = None
        self.harmonic_energy_sum = None
        self.inharmonic_energy_sum = None
        self.subbass_energy_sum = None
        self.total_component_energy = None
        self.harmonic_energy_ratio = None
        self.inharmonic_energy_ratio = None
        self.subbass_energy_ratio = None
        self.linear_sum_amplitude_harmonic = None
        self.linear_sum_amplitude_inharmonic_partial = None
        self.linear_sum_amplitude_subbass_band = None
        self.linear_amplitude_fraction_inharmonic_of_HI = None
        self.linear_amplitude_fraction_nonharmonic_of_total = None
        self.linear_amplitude_batch_alignment_factor = None
        self.harmonic_partial_count = None
        self.inharmonic_partial_count = None
        self.total_detected_partial_count = None
        self.unique_harmonic_order_count = None
        self.harmonic_order_count = None
        self.harmonic_peak_count = None
        self.inharmonic_peak_count = None
        self.subbass_peak_count = None
        self.total_detected_peak_count = None
        self.harmonic_peak_candidate_count = None
        self.nonharmonic_peak_candidate_count = None
        self.low_frequency_peak_candidate_count = None
        self.total_peak_candidate_count = None
        self.residual_spectral_row_count = None
        self.nonharmonic_candidate_row_count = None
        self.retained_nonharmonic_peak_candidate_count = None
        self.exported_nonharmonic_peak_candidate_count = None
        self.peaklist_harmonic_window_candidate_count = None
        self.peaklist_nonharmonic_window_candidate_count = None
        self.peaklist_low_frequency_window_candidate_count = None
        self.peaklist_total_window_candidate_count = None
        self.debug_counts_invariant_status = ""
        self.debug_counts_invariant_failures = ""
        self.accepted_inharmonic_peak_count = None
        self.accepted_inharmonic_partial_count = None
        self.harmonic_candidate_count = None
        self.inharmonic_candidate_count = None
        self.subbass_candidate_count = None
        self.total_spectral_candidate_count = None
        self.residual_row_count = None
        self.harmonic_bin_count = None
        self.inharmonic_bin_count = None
        self.subbass_bin_count = None
        self.energy_conservation_status = "not_available_at_compile_stage"
        self.energy_conservation_error = None
        self.energy_denominator_description = "not_available_at_compile_stage"
        self.dissonance_partial_count = None
        self.dissonance_pair_count = None
        self.harmonic_validation_report = None
        self.adaptive_subfundamental_cutoff_hz = None
        self.subfundamental_margin_percent = None
        self.percentage_subfundamental_cutoff_hz = None
        self.leakage_guard_cutoff_hz = None
        self.effective_subfundamental_margin_percent = None
        self.subfundamental_cutoff_selection_rule = ""
        self.subfundamental_cutoff_selected_by = ""
        self.subfundamental_guard_valid = False
        self.subfundamental_guard_policy = "invalid_f0"
        self.spectral_density_freq_floor_hz = None
        self.harmonic_leakage_protection_hz = None
        self.canonical_density_v5_adapted = None
        self.discrete_metric_d3 = float("nan")
        self.discrete_metric_d10 = float("nan")
        self.discrete_metric_d17 = float("nan")
        self.discrete_metric_d24 = float("nan")
        self.density_per_component = float("nan")
        self.density_formula_version = CANONICAL_DENSITY_FORMULA_VERSION
        self.density_source_formula = CANONICAL_DENSITY_SOURCE_FORMULA
        self.density_normalization_scope = "none_per_note_absolute_canonical"
        self.density_normalization_denominator = float("nan")
        self.density_metric_normalized = float("nan")
        self.density_metric_per_harmonic = None
        self.density_metric_ratio_over_fundamental_legacy = float("nan")
        self._metrics_ih_amps_eff = np.asarray([], dtype=float)
        self._metrics_ih_freqs_eff = None
        self.component_energy_status = "not_computed"
        self.component_energy_pie_basis = "not_written"
        self.amplitude_mass_chart_file = ""
        self.energy_ratio_chart_file = ""
        self.amplitude_mass_chart_status = "not_attempted"
        self.energy_ratio_chart_status = "not_attempted"
        self.component_energy_pie_file = ""
        self.component_energy_pie_alias_basis = ""
        self.effective_partial_density_status = "not_computed"
        self.density_metric_status = "not_computed"
        self.normalization_status = "not_computed"
        self.debug_counts_status = "not_computed"
        self.model_weight_status = "not_computed"
        self.model_weight_fallback_applied = False
        self.logger.warning("Metrics reset to default values (0.0)")

    # ----------------- helpers mÃ©tricas -----------------
    def _validate_amplitude_data(self, data_name: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
        if amplitudes.size == 0:
            return amplitudes
        amps = amplitudes[np.isfinite(amplitudes)]
        if amps.size == 0:
            return np.asarray([], dtype=float)
        return np.maximum(amps, 1e-12)  # sem reescala



    def _ensure_all_metrics_calculated(self) -> None:
        try:
            import numpy as np

            # ---- helper para ganho coerente (usa o nome que tiveres no ficheiro) ----
            def _cg():
                try:
                    # tenta _coherent_gain_local; se nÃ£o existir, usa _coherent_gain
                    fn = globals().get("_coherent_gain_local") or globals().get("_coherent_gain")
                    cg = float(fn(getattr(self, "window", "hann"), int(getattr(self, "n_fft", 4096))))
                    return cg if cg > 0 else 1.0
                except Exception:
                    return 1.0

            # ---- checks mÃ­nimos ----
            if self.harmonic_list_df is None or self.harmonic_list_df.empty:
                self._set_default_metrics()
                return

            # ---- garantir coluna Amplitude (harmÃ³nicos) ----
            if "Amplitude" not in self.harmonic_list_df.columns:
                if "Magnitude (dB)" in self.harmonic_list_df.columns:
                    self.harmonic_list_df["Amplitude"] = np.power(
                        10.0, self.harmonic_list_df["Magnitude (dB)"].to_numpy(float) / 20.0
                    )
                else:
                    self._set_default_metrics()
                    return

            amps_c = self.harmonic_list_df["Amplitude"].to_numpy(float)

            # ---- Density Metric (absoluta; restored from older version) ----
            if self.density_metric_value is None:
                _wf_dm = str(self.weight_function or "linear").strip().lower()
                _freq_dm = None
                if _wf_dm == "d24" and "Frequency (Hz)" in self.harmonic_list_df.columns:
                    _freq_dm = pd.to_numeric(
                        self.harmonic_list_df["Frequency (Hz)"], errors="coerce"
                    ).to_numpy(dtype=float)
                self.density_metric_value = float(
                    apply_density_metric(amps_c, self.weight_function, frequencies=_freq_dm)
                )
                # Scale to 0-10 range (matching older version)
                self.scaled_density_metric_value = self.density_metric_value * 10.0

            _freq_disc = None
            if "Frequency (Hz)" in self.harmonic_list_df.columns:
                _freq_disc = pd.to_numeric(
                    self.harmonic_list_df["Frequency (Hz)"], errors="coerce"
                ).to_numpy(dtype=float)
            _disc2 = compute_discrete_spectral_metrics_bundle(amps_c, _freq_disc)
            self.discrete_metric_d3 = _disc2["discrete_metric_d3"]
            self.discrete_metric_d10 = _disc2["discrete_metric_d10"]
            self.discrete_metric_d17 = _disc2["discrete_metric_d17"]
            self.discrete_metric_d24 = _disc2["discrete_metric_d24"]

            # ---- Filtered Density (absoluta; por amplitudes; sem normalize) ----
            if self.filtered_density_metric_value is None:
                if self.filtered_list_df is not None and not self.filtered_list_df.empty:
                    if "Amplitude" not in self.filtered_list_df.columns:
                        if "Magnitude (dB)" in self.filtered_list_df.columns:
                            self.filtered_list_df["Amplitude"] = np.power(
                                10.0, self.filtered_list_df["Magnitude (dB)"].to_numpy(float) / 20.0
                            )
                    if "Amplitude" in self.filtered_list_df.columns:
                        famps_c = self.filtered_list_df["Amplitude"].to_numpy(float)
                        _wf_fdc = str(self.weight_function or "linear").strip().lower()
                        _freq_fdc = None
                        if _wf_fdc == "d24" and "Frequency (Hz)" in self.filtered_list_df.columns:
                            _freq_fdc = pd.to_numeric(
                                self.filtered_list_df["Frequency (Hz)"], errors="coerce"
                            ).to_numpy(dtype=float)
                        self.filtered_density_metric_value = float(
                            apply_density_metric(
                                famps_c, self.weight_function, normalize=False, frequencies=_freq_fdc
                            )
                        )
                    else:
                        self.filtered_density_metric_value = 0.0
                else:
                    self.filtered_density_metric_value = 0.0

            # ---- Entropia espectral (sobre potÃªncia) ----
            if self.entropy_spectral_value is None:
                self.entropy_spectral_value = float(compute_spectral_entropy((amps_c ** 2)))

            # ---- Combined (H/IH) - Amplitude-weighted + inharmonic from partial list / residual ----
            if self.combined_density_metric_value is None:
                harm_density = float(self.density_metric_value if self.density_metric_value is not None else 0.0)
                inharm_density = float(self._compute_inharmonic_density_for_combined())

                # Use logarithmic combination to preserve dynamic range
                try:
                    import math
                    harm_log = math.log1p(max(0.0, harm_density))
                    inharm_log = math.log1p(max(0.0, inharm_density))
                    combined_log = self.harmonic_weight * harm_log + self.inharmonic_weight * inharm_log
                    self.combined_density_metric_value = float(math.expm1(combined_log))
                except Exception:
                    # Fallback to linear (but still amplitude-weighted, not count-based)
                    self.combined_density_metric_value = float(
                        calculate_combined_density_metric(
                            harm_density, inharm_density, 
                            self.harmonic_weight, self.inharmonic_weight,
                            preserve_dynamic_range=False  # Linear fallback
                        )
                    )

            # ---- Total Metric (idÃªntica ao mÃ©todo principal) ----
            if self.total_metric_value is None:
                # PHASE 3: Use constants instead of magic numbers
                wD, wS, wE, wC = (DENSITY_METRIC_WEIGHT_D, DENSITY_METRIC_WEIGHT_S, 
                                DENSITY_METRIC_WEIGHT_E, DENSITY_METRIC_WEIGHT_C)
                # usa as mesmas normalizaÃ§Ãµes do principal
                sd = self.scaled_density_metric_value or 0.0
                sdm = self.spectral_density_metric_value or 0.0
                norm_density  = sd / 10.0
                norm_spectral = sdm / 10.0
                norm_entropy  = self.entropy_spectral_value or 0.0
                norm_combined = self.combined_density_metric_value or 0.0
                self.total_metric_value = (wD*norm_density + wS*norm_spectral + wE*norm_entropy + wC*norm_combined) * 10.0

        except Exception as e:
            self.logger.error(f"Error verifying metrics: {e}", exc_info=True)
            self._set_default_metrics()


    # ----------------- dissonance -----------------
    def calculate_dissonance_metrics(self) -> None:
        if not self.dissonance_enabled or self.harmonic_list_df is None or self.harmonic_list_df.empty:
            return
        try:
            # Garantir que a coluna Amplitude existe
            if 'Amplitude' not in self.harmonic_list_df.columns:
                self.harmonic_list_df['Amplitude'] = np.power(10.0, self.harmonic_list_df['Magnitude (dB)'] / 20.0)

            # --- OTIMIZAÇÃO CRÍTICA (Limitador de Picos) ---
            # Pairwise dissonance scales as O(n^2); cap the harmonic partial list for stability.
            _cap = int(DISSONANCE_PAIRWISE_PARTIAL_CAP)
            df_calc = self.harmonic_list_df.copy()
            n_before = int(len(df_calc))
            self.dissonance_partial_count_before_cap = n_before

            if n_before > _cap:
                df_calc = df_calc.nlargest(_cap, "Amplitude")
                self.dissonance_partial_cap = _cap
                self.dissonance_cap_computation_note = str(DISSONANCE_CAP_COMPUTATION_NOTE)
            else:
                self.dissonance_partial_cap = "not_applied"
                self.dissonance_cap_computation_note = (
                    "Full harmonic partial list used for dissonance (pairwise cap not applied)."
                )

            n_after = int(len(df_calc))
            self.dissonance_partial_count_after_cap = n_after
            self.dissonance_pair_count_after_cap = int(n_after * (n_after - 1) // 2) if n_after >= 2 else 0

            # Gerar lista de tuplos baseada APENAS nestes parciais (pós-cap)
            partials = [(row['Frequency (Hz)'], row['Amplitude']) for _, row in df_calc.iterrows()]

            # Decidir quais modelos calcular
            models_to_calc = list_available_models() if self.dissonance_compare_models else [self.dissonance_model]

            for mname in models_to_calc:
                try:
                    model = get_dissonance_model(mname)

                    # 1. Cálculo do valor escalar (usando o DataFrame reduzido)
                    self.dissonance_values[mname] = model.calculate_dissonance_metric(df_calc)

                    # 2. Cálculo da Curva (se ativado)
                    if self.dissonance_curve_enabled:
                        # Usa a lista 'partials' que já está reduzida a 50 itens
                        self.dissonance_curves[mname] = model.calculate_dissonance_curve(partials, 1.0, 2.0, 200)

                        # 3. Escalas e Mínimos Locais
                        if self.dissonance_scale_enabled and self.dissonance_curves[mname] is not None:
                            self.dissonance_scales[mname] = model.find_local_minima(self.dissonance_curves[mname])
                            if 1.0 not in self.dissonance_scales[mname]:
                                self.dissonance_scales[mname].insert(0, 1.0)
                            if 2.0 not in self.dissonance_scales[mname]:
                                self.dissonance_scales[mname].append(2.0)
                            self.dissonance_scales[mname] = sorted(self.dissonance_scales[mname])

                except Exception as e:
                    self.logger.error(f"Dissonance model {mname} failed: {e}")
                    self.dissonance_values[mname] = None
                    self.dissonance_curves[mname] = None
                    self.dissonance_scales[mname] = None

            self.dissonance_partial_count = int(n_after)
            self.dissonance_pair_count = int(self.dissonance_pair_count_after_cap or 0)

        except Exception as e:
            self.logger.error(f"Error in calculate_dissonance_metrics: {e}")
            # Limpeza de segurança
            self.dissonance_partial_cap = None
            self.dissonance_partial_count_before_cap = None
            self.dissonance_partial_count_after_cap = None
            self.dissonance_pair_count_after_cap = None
            self.dissonance_cap_computation_note = None
            self.dissonance_partial_count = None
            self.dissonance_pair_count = None
            if hasattr(self, 'dissonance_values') and self.dissonance_values:
                for m in self.dissonance_values:
                    self.dissonance_values[m] = None
                    self.dissonance_curves[m] = None
                    self.dissonance_scales[m] = None

    # ----------------- compilar mÃ©tricas / exportaÃ§Ãµes -----------------
    @staticmethod
    def _metrics_sources_present(root: Path) -> tuple[bool, bool]:
        """Return (excel_present, json_present) under root (recursive)."""
        root = Path(root)
        return (
            any(root.rglob("spectral_analysis.xlsx")),
            any(root.rglob("super_analysis_results.json")),
        )

    def _resolve_compile_metrics_root(self, results_directory: Path) -> Path:
        """
        Directory containing per-note metrics (aligned with ``pipeline_orchestrator_gui``).

        - If workbooks live directly under ``results_directory``, that directory is used.
        - Otherwise, if ``results_directory / "analysis_results"`` contains
          ``spectral_analysis.xlsx`` or ``super_analysis_results.json``, compile from
          that subfolder (orchestrator-style layout).
        """
        rd = Path(results_directory)
        ex_root, js_root = self._metrics_sources_present(rd)
        if ex_root or js_root:
            return rd
        nested = rd / "analysis_results"
        if nested.is_dir():
            ex_n, js_n = self._metrics_sources_present(nested)
            if ex_n or js_n:
                self.logger.info(
                    "Analysis artefacts found under %s; compiling from that folder.",
                    nested,
                )
                return nested
        return rd

    def _find_compiled_metrics_excel(self, results_directory: Path) -> Optional[Path]:
        """
        Localiza o Excel compilado (main_7 grava em compile_root; pode haver sufixo de dinâmica).
        """
        compile_root = self._resolve_compile_metrics_root(Path(results_directory))
        for fn in ("compiled_density_metrics.xlsx", "compiled_metrics.xlsx"):
            p = compile_root / fn
            if p.is_file():
                return p
        dynamics = list(compile_root.glob("compiled_density_metrics_*.xlsx"))
        if dynamics:
            return max(dynamics, key=lambda x: x.stat().st_mtime)
        return None

    def _compile_metrics(
        self, 
        results_directory: Path,
        use_tsne: bool = False,
        use_umap: bool = False,
        detect_anomalies: bool = False,
        anomaly_contamination: Optional[float] = None,
        harmonic_weight: float = 0.95,
        inharmonic_weight: float = 0.05,
        weight_function: str = "linear"
    ) -> None:
        try:
            from compile_metrics import compile_density_metrics_with_pca, extract_dynamics_from_path
            rd_abs = Path(results_directory).resolve()
            self.logger.info("Starting _compile_metrics: results_directory=%s", rd_abs)
            compile_root = self._resolve_compile_metrics_root(results_directory)
            if compile_root.resolve() != rd_abs:
                self.logger.info("Resolved compile root directory: %s", compile_root.resolve())
            excel_present, json_present = self._metrics_sources_present(compile_root)
            allow_legacy = bool(
                getattr(self, "allow_legacy_super_json_for_compile", False)
            )
            if excel_present:
                file_pattern = "spectral_analysis.xlsx"
            elif allow_legacy and json_present:
                file_pattern = "super_analysis_results.json"
                self.logger.warning(
                    "Legacy compile mode: using super_analysis_results.json under %s "
                    "(non-canonical publication path).",
                    compile_root,
                )
            else:
                self.logger.warning(
                    "No canonical spectral_analysis.xlsx under %s. "
                    "super_analysis_results.json is not accepted unless allow_legacy_super_json_for_compile=True.",
                    compile_root,
                )
                return
            # Extract dynamics from results directory path and add to filename
            dynamics = extract_dynamics_from_path(compile_root)
            if dynamics:
                outp = compile_root / f'compiled_density_metrics_{dynamics}.xlsx'
                self.logger.info(f"Dynamics '{dynamics}' detected, using filename: {outp.name}")
            else:
                outp = compile_root / 'compiled_density_metrics.xlsx'
            compile_density_metrics_with_pca(
                folder_path=compile_root, 
                output_path=outp,
                file_pattern=file_pattern,
                include_pca=True,
                use_tsne=use_tsne,
                use_umap=use_umap,
                detect_anomalies=detect_anomalies,
                anomaly_contamination=anomaly_contamination,
                harmonic_weight=harmonic_weight,
                inharmonic_weight=inharmonic_weight,
                weight_function=weight_function,
                allow_legacy_super_json=allow_legacy,
                compilation_extra_metadata={
                    "input_schema_validation_status": "not_validated_proc_audio_auto_compile",
                },
            )
            if not outp.is_file():
                self.logger.warning(
                    "compiled_density_metrics did not appear after PCA; trying simple compile (without PCA)."
                )
                from compile_metrics import compile_density_metrics
                compile_density_metrics(
                    compile_root,
                    outp,
                    file_pattern=file_pattern,
                    include_pca=False,
                    harmonic_weight=harmonic_weight,
                    inharmonic_weight=inharmonic_weight,
                    weight_function=weight_function,
                    enable_pca_export=False,
                )
            if not outp.is_file():
                self.logger.warning(
                    "compiled_density_metrics still missing; trying export without publication "
                    "column filtering (compiled_public_columns=False)."
                )
                from compile_metrics import compile_density_metrics as _cdm

                _cdm(
                    compile_root,
                    outp,
                    file_pattern=file_pattern,
                    include_pca=False,
                    harmonic_weight=harmonic_weight,
                    inharmonic_weight=inharmonic_weight,
                    weight_function=weight_function,
                    compiled_public_columns=False,
                    enable_pca_export=False,
                )
            if outp.is_file():
                self.logger.info("Compiled workbook confirmed: %s", outp.resolve())
            else:
                self.logger.error(
                    "Final failure: could not create %s after PCA, simple fallback, and "
                    "non-publication-filter fallback. Check write permissions and compile_metrics logs "
                    "(Excel save error).",
                    outp,
                )
            audit = getattr(compile_density_metrics_with_pca, "_last_dr_audit", {}) or {}
            self.last_density_dr_audit = audit
            pca_done = bool(audit.get("PCA_applied"))
            self.logger.info(
                "Compiled metrics saved to %s | PCA_applied=%s PCA_status=%s",
                outp,
                pca_done,
                audit.get("PCA_status"),
            )
            if pca_done:
                self.logger.info(
                    "PCA was applied to compiled density metrics (see Methodological Warnings / DR audit when available)."
                )
            else:
                self.logger.info(
                    "PCA was not applied for this compilation (status=%s); do not describe outputs as 'with PCA'.",
                    audit.get("PCA_status"),
                )
            if use_tsne:
                self.logger.info(
                    "t-SNE: applied=%s status=%s",
                    audit.get("TSNE_applied"),
                    audit.get("TSNE_status"),
                )
            if use_umap:
                self.logger.info(
                    "UMAP: applied=%s status=%s",
                    audit.get("UMAP_applied"),
                    audit.get("UMAP_status"),
                )
            if detect_anomalies:
                self.logger.info(
                    "Anomaly detection: applied=%s status=%s",
                    audit.get("anomaly_detection_applied"),
                    audit.get("Anomaly_status"),
                )
        except ImportError:
            self.logger.error("compile_density_metrics_with_pca is not available.")
            try:
                from compile_metrics import compile_density_metrics
                compile_root_fb = self._resolve_compile_metrics_root(results_directory)
                outp = compile_root_fb / 'compiled_metrics.xlsx'
                compile_density_metrics(compile_root_fb, outp)
                self.last_density_dr_audit = {}
                self.logger.info(
                    "Metrics compiled via legacy compile_density_metrics (no compile_density_metrics_with_pca DR/PCA path): %s",
                    outp,
                )
            except ImportError:
                self.logger.error("compile_density_metrics is not available.")
        except Exception as e:
            self.logger.error("Error compiling metrics: %s", e, exc_info=True)

    def _export_data_for_visualization(
        self,
        note: str,
        output_folder: Path,
        interactive_dir: Path,
        export_format: str
    ) -> None:
        try:
            if self.db_S is not None and self.freqs is not None and self.times is not None:
                MAX_FREQ_BINS = 128
                MAX_TIME_FRAMES = 200
                freq_step = max(1, len(self.freqs) // MAX_FREQ_BINS + 1)
                time_step = max(1, len(self.times) // MAX_TIME_FRAMES + 1)
                freqs_reduced = self.freqs[::freq_step]
                times_reduced = self.times[::time_step]
                idx_f = np.arange(0, len(self.freqs), freq_step)
                idx_t = np.arange(0, len(self.times), time_step)
                idx_f = idx_f[idx_f < self.db_S.shape[0]]
                idx_t = idx_t[idx_t < self.db_S.shape[1]]
                if len(idx_f) > 0 and len(idx_t) > 0:
                    try:
                        spectro_reduced = self.db_S[np.ix_(idx_f, idx_t)]
                        data = {
                            'note': note,
                            'freqs': freqs_reduced.tolist(),
                            'times': times_reduced.tolist(),
                            'values': spectro_reduced.tolist()
                        }
                        if export_format.lower() == 'json':
                            spath = interactive_dir / f"{note}_spectrogram_data.json"
                            with open(spath, 'w') as f:
                                json.dump(data, f)
                        elif export_format.lower() == 'csv':
                            pass
                    except MemoryError:
                        spath = interactive_dir / f"{note}_spectrogram_data.json"
                        data = {
                            'note': note,
                            'freqs': freqs_reduced[::2].tolist(),
                            'times': times_reduced[::2].tolist(),
                            'error': 'reduced due to memory'
                        }
                        with open(spath, 'w') as f:
                            json.dump(data, f)
        except Exception as e:
            self.logger.error(f"Error exporting data for visualisation: {e}")
            try:
                ep = interactive_dir / f"{note}_error.json"
                with open(ep, 'w') as f:
                    json.dump({'note': note, 'error': str(e)}, f)
            except Exception:
                pass

    def _export_combined_data_for_visualization(
        self,
        results_directory: Path,
        interactive_dir: Path,
        export_format: str
    ) -> None:
        try:
            compiled_file = self._find_compiled_metrics_excel(results_directory)
            if compiled_file is None:
                self.logger.warning("No compiled workbook found; skipping combined export.")
                return
            df = pd.read_excel(compiled_file)
            if df.empty:
                self.logger.warning("Compiled workbook is empty.")
                return

            if export_format.lower() == 'json':
                path = interactive_dir / "combined_metrics.json"
                try:
                    from metadata_sanitizer import publication_redaction_enabled, sanitize_dataframe_for_publication

                    if publication_redaction_enabled():
                        df = sanitize_dataframe_for_publication(df)
                except Exception:
                    pass
                with open(path, 'w') as f:
                    json.dump(df.to_dict(orient='records'), f)
            elif export_format.lower() == 'csv':
                path = interactive_dir / "combined_metrics.csv"
                df.to_csv(path, index=False)

            audit = getattr(self, "last_density_dr_audit", {}) or {}
            pca_columns_present = "PC1" in df.columns
            pca_applied_in_compilation = bool(audit.get("PCA_applied", False))
            cfg = {
                'metrics_available': [c for c in df.columns if c not in ['Note', 'Folder']],
                'notes': sorted(df['Note'].unique().tolist() if 'Note' in df.columns else []),
                'model_names': list_available_models(),
                'interactive_visualizations': {
                    'spectrogram_3d': True,
                    'dissonance_curves': self.dissonance_curve_enabled,
                    'pca_scatter': bool(pca_columns_present and pca_applied_in_compilation),
                    'pca_columns_present_without_dr_audit': bool(pca_columns_present and not audit),
                    'tsne_scatter': 'TSNE1' in df.columns and 'TSNE2' in df.columns,
                    'umap_scatter': 'UMAP1' in df.columns and 'UMAP2' in df.columns,
                    'anomaly_detection': 'is_anomaly' in df.columns
                },
                'stage2_compile_dr_audit': audit,
            }
            try:
                from metadata_sanitizer import publication_redaction_enabled, sanitize_metadata_dict

                if publication_redaction_enabled():
                    cfg = sanitize_metadata_dict(cfg)
            except Exception:
                pass
            with open(interactive_dir / "visualization_config.json", 'w') as f:
                json.dump(cfg, f)
            try:
                from analysis_reporting import build_compile_warnings

                wr = build_compile_warnings(df, audit, include_pca=True)
                try:
                    from metadata_sanitizer import publication_redaction_enabled, sanitize_metadata_dict

                    if publication_redaction_enabled() and isinstance(wr, dict):
                        wr = sanitize_metadata_dict(wr)
                except Exception:
                    pass
                with open(interactive_dir / "stage2_compiled_warnings.json", "w", encoding="utf-8") as wf:
                    json.dump(wr, wf, indent=2, ensure_ascii=False, default=str)
            except Exception:
                pass
        except Exception as e:
            self.logger.error(f"Error exporting combined data: {e}")

    # ----------------- visualizaÃ§Ãµes -----------------
    def plot_spectrograms(
        self,
        path: Optional[Union[str, Path]] = None,
        note: str = ""
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception as e:
            self.logger.error(f"Matplotlib not available: {e}")
            return
        if any(v is None for v in (self.db_S, self.freqs, self.times, getattr(self, "S", None))):
            self.logger.error("Insufficient data to plot (db_S/freqs/times/S).")
            return

        S_mag = np.abs(self.S)
        S_db = np.asarray(self.db_S, dtype=float)

        fig = plt.figure(figsize=(12, 10))
        try:
            ax1 = plt.subplot(3, 1, 1)
            librosa.display.specshow(
                S_db, sr=self.sr, x_axis="time", y_axis="log", cmap="coolwarm"
            )
            plt.colorbar(format="%+2.0f dB", ax=ax1)
            ax1.set_title(f"Spectrogram (dB) — Note: {note}")

            ax2 = plt.subplot(3, 1, 2)
            # Use mean of dB along time so this curve matches a vertical average of the
            # spectrogram above. dB(mean(|S|)) inflates weak bands (esp. sub-bass / noise)
            # versus mean(dB), which looked like an inconsistent "high floor" vs the 2D plot.
            mean_spectrum_db = np.nanmean(S_db, axis=1)
            n = min(len(self.freqs), len(mean_spectrum_db))
            ax2.plot(self.freqs[:n], mean_spectrum_db[:n])
            ax2.set_title(f"Frequency spectrum (mean over time) — Note: {note}")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Magnitude (dB)")
            ax2.set_xscale("log")

            try:
                ax3 = plt.subplot(3, 1, 3)
                S_power = S_mag**2
                S_mel = librosa.feature.melspectrogram(S=S_power, sr=self.sr, n_mels=128)
                S_db_mel = librosa.power_to_db(S_mel, ref=np.max)
                librosa.display.specshow(S_db_mel, sr=self.sr, x_axis="time", y_axis="mel", cmap="magma")
                plt.colorbar(format="%+2.0f dB", ax=ax3)
                ax3.set_title(f"Mel spectrogram (dB) — Note: {note}")
            except Exception as _mel_e:
                self.logger.warning("Mel spectrogram (third panel) skipped: %s", _mel_e)

            plt.tight_layout()

            if path:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=150, bbox_inches="tight")
                self.logger.info(f"Spectrogram saved to: {path}")
                plt.close(fig)

                path_3d = path.with_name(path.stem + "_3d").with_suffix(".html")
                try:
                    self.plot_3d_spectrogram(path=path_3d, note=note)
                except Exception as _3d_e:
                    self.logger.warning("Interactive 3D spectrogram (HTML) not written: %s", _3d_e)
            else:
                plt.show()
                plt.close(fig)
        except Exception as e:
            self.logger.error(f"Error plotting spectrograms: {e}")
            plt.close(fig)

    def plot_3d_spectrogram(
        self,
        path: Optional[Union[str, Path]] = None,
        note: str = ""
    ) -> None:
        try:
            import plotly.graph_objects as go
        except Exception as e:
            self.logger.error(f"Plotly not available: {e}")
            return
        if any(v is None for v in (self.db_S, self.freqs, self.times)):
            self.logger.error("Insufficient data for 3D plot.")
            return
        try:
            Z_raw = np.asarray(self.db_S, dtype=float)
            finite = np.isfinite(Z_raw)
            if finite.any():
                z_hi = float(np.nanpercentile(Z_raw[finite], 99.5))
                z_lo = float(np.nanpercentile(Z_raw[finite], 0.5))
                if not math.isfinite(z_hi) or not math.isfinite(z_lo):
                    z_lo, z_hi = -120.0, 0.0
                z_hi = min(0.0, max(z_lo + 1.0, z_hi))
                z_lo = max(-200.0, min(z_hi - 1.0, z_lo))
            else:
                z_lo, z_hi = -120.0, 0.0
            Z = np.nan_to_num(Z_raw, nan=z_lo, posinf=z_hi, neginf=-200.0)
            X = np.asarray(self.times, dtype=float)
            Y = np.asarray(self.freqs, dtype=float)
            Y = np.maximum(Y, 1e-12)
            if Z.shape != (len(Y), len(X)):
                ny = min(Z.shape[0], len(Y))
                nx = min(Z.shape[1], len(X))
                Z = Z[:ny, :nx]
                Y = Y[:ny]
                X = X[:nx]

            z_pad = 6.0
            z_min_ax = max(-200.0, z_lo - z_pad)
            z_max_ax = min(0.0, z_hi + z_pad)
            sr_v = float(getattr(self, "sr", 44100.0) or 44100.0)
            f_y_hi = min(sr_v * 0.499, 16000.0)
            if Y.size:
                f_y_lo = max(20.0, float(np.min(Y)))
                f_y_hi = min(f_y_hi, max(50.0, float(np.max(Y))))
            else:
                f_y_lo = 20.0
            if f_y_lo >= f_y_hi * 0.995:
                f_y_hi = max(f_y_lo * 2.0, 100.0)

            surface = go.Surface(
                z=Z,
                x=X,
                y=Y,
                colorscale="Viridis",
                showscale=True,
                cmin=z_lo,
                cmax=z_hi,
            )
            layout = go.Layout(
                title=f"3D Spectrogram (dB) — Note: {note}",
                scene=dict(
                    xaxis=dict(title="Time (s)"),
                    yaxis=dict(
                        title="Frequency (Hz)",
                        type="log",
                        range=[math.log10(f_y_lo), math.log10(f_y_hi)],
                    ),
                    zaxis=dict(title="Magnitude (dB)", range=[z_min_ax, z_max_ax]),
                ),
                width=900, height=700, margin=dict(l=65, r=50, b=65, t=90),
            )
            fig = go.Figure(data=[surface], layout=layout)
            fig.update_layout(scene_camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8), up=dict(x=0, y=0, z=1)))
            if path:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(path))
                self.logger.info(f"3D spectrogram saved to: {path}")
            else:
                fig.show()
        except Exception as e:
            self.logger.error(f"Error plotting 3D spectrogram: {e}")

    def plot_dissonance_curve(
        self,
        model_name: str,
        path: Optional[Union[str, Path]] = None,
        note: str = ""
    ) -> None:
        if not getattr(self, "dissonance_enabled", False) or model_name not in self.dissonance_curves:
            self.logger.warning(f"Dissonance model {model_name} is not available.")
            return
        curve = self.dissonance_curves.get(model_name)
        scale = self.dissonance_scales.get(model_name)
        if curve is None or scale is None:
            self.logger.warning(f"No dissonance curve for {model_name}.")
            return
        try:
            model = get_dissonance_model(model_name)
            title = f"{model_name} Dissonance Curve — Note: {note}"
            model.visualize_dissonance_curve(curve, scale, title=title, save_file=path)
            if path:
                self.logger.info(f"Dissonance curve for {model_name} saved to: {path}")
        except Exception as e:
            self.logger.error(f"Error plotting dissonance curve: {e}")

    def plot_dissonance_comparison(
        self,
        path: Optional[Union[str, Path]] = None,
        note: str = ""
    ) -> None:
        if not getattr(self, "dissonance_enabled", False) or not getattr(self, "dissonance_compare_models", False):
            return
        models_to_include = [m for m, v in self.dissonance_curves.items() if v is not None]
        if len(models_to_include) < 2:
            self.logger.warning("Not enough dissonance curves for comparison.")
            return
        try:
            from dissonance_models import compare_dissonance_models
            if "Amplitude" not in self.harmonic_list_df.columns:
                self.harmonic_list_df["Amplitude"] = np.power(10.0, self.harmonic_list_df["Magnitude (dB)"] / 20.0)
            partials = [(row["Frequency (Hz)"], row["Amplitude"]) for _, row in self.harmonic_list_df.iterrows()]
            compare_dissonance_models(partials, 1.0, 2.0, 200, save_file=path, models_to_include=models_to_include)
            if path:
                self.logger.info(f"Dissonance model comparison saved to: {path}")
        except Exception as e:
            self.logger.error(f"Error in dissonance model comparison: {e}")


    def _reconstruct_linear_component_density_balance_triple(self) -> Tuple[float, float, float]:
        """Rebuild linear Σ amplitudes when ``linear_sum_amplitude_*`` are all zero.

        Harmonic: Σ ``Amplitude_raw`` (else ``Amplitude``) on harmonic spectrum
        candidates with ``include_for_density`` true — same population as the
        Harmonic Spectrum density mask.

        Inharmonic / sub-bass: same discrete-peak and band masks as the exported
        ``Inharmonic Spectrum`` / ``Sub-bass band`` sheets (linear amplitude, not ΣA²).
        """
        def _ensure_amp_column(df: pd.DataFrame) -> pd.DataFrame:
            if df is None or df.empty:
                return df
            if "Amplitude" not in df.columns:
                if "Magnitude (dB)" in df.columns:
                    df = df.copy()
                    df["Amplitude"] = np.power(
                        10.0,
                        pd.to_numeric(df["Magnitude (dB)"], errors="coerce").fillna(-120.0) / 20.0,
                    )
            return df

        h_sum = 0.0
        cand = getattr(self, "harmonic_spectrum_candidates_df", None)
        if isinstance(cand, pd.DataFrame) and not cand.empty:
            if "include_for_density" in cand.columns:
                inc = cand["include_for_density"].astype(bool).to_numpy()
            else:
                inc = np.ones(len(cand), dtype=bool)
            amp_col = (
                "Amplitude_raw"
                if "Amplitude_raw" in cand.columns
                else ("Amplitude" if "Amplitude" in cand.columns else None)
            )
            if amp_col is not None:
                vals = (
                    pd.to_numeric(cand.loc[inc, amp_col], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                h_sum = float(np.nansum(vals))
        elif isinstance(getattr(self, "harmonic_list_df", None), pd.DataFrame) and not self.harmonic_list_df.empty:
            hl = _ensure_amp_column(self.harmonic_list_df.copy())
            if "Amplitude" in hl.columns:
                h_sum = float(pd.to_numeric(hl["Amplitude"], errors="coerce").fillna(0.0).sum())

        _, ih_sel = self._nonharmonic_residual_pipeline_dataframes()
        ih_raw = ih_sel.copy()

        ih_df = _ensure_amp_column(ih_raw.copy()) if ih_raw is not None and not ih_raw.empty else pd.DataFrame()

        _cut_sb = float(getattr(self, "subbass_aggregate_hz", SUBBASS_AGGREGATE_CUTOFF_HZ))
        _lo_sb = float(getattr(self, "subbass_aggregate_lower_hz", SUBBASS_AGGREGATE_LOWER_HZ))
        sub_df = pd.DataFrame()
        if isinstance(getattr(self, "complete_list_df", None), pd.DataFrame) and not self.complete_list_df.empty:
            compf = _ensure_amp_column(self.complete_list_df.copy())
            if "Frequency (Hz)" in compf.columns:
                ff = pd.to_numeric(compf["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
                mask_sb = np.isfinite(ff) & (ff > _lo_sb) & (ff <= _cut_sb)
                sub_df = compf.loc[mask_sb].copy()

        if not ih_df.empty and not sub_df.empty and "Frequency (Hz)" in ih_df.columns and "Frequency (Hz)" in sub_df.columns:
            ih_f = pd.to_numeric(ih_df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
            ih_f = ih_f[np.isfinite(ih_f)]
            sf = pd.to_numeric(sub_df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
            keep_sb = np.zeros(sf.shape[0], dtype=bool)
            for _j, f_hz in enumerate(sf):
                if not np.isfinite(f_hz):
                    continue
                thr_m = max(1e-6, abs(f_hz) * 1e-6)
                if ih_f.size == 0 or not np.any(np.abs(ih_f - f_hz) < thr_m):
                    keep_sb[_j] = True
            sub_df = sub_df.loc[keep_sb].copy()

        try:
            harm_protect_df = self._build_subbass_harmonic_protection_df()
        except Exception:
            harm_protect_df = pd.DataFrame({"Frequency (Hz)": pd.Series(dtype=float)})
        if (
            not sub_df.empty
            and "Frequency (Hz)" in sub_df.columns
            and not harm_protect_df.empty
            and "Frequency (Hz)" in harm_protect_df.columns
        ):
            hp_f = pd.to_numeric(
                harm_protect_df["Frequency (Hz)"], errors="coerce"
            ).to_numpy(dtype=float)
            hp_f = hp_f[np.isfinite(hp_f)]
            if hp_f.size > 0:
                sf2 = pd.to_numeric(
                    sub_df["Frequency (Hz)"], errors="coerce"
                ).to_numpy(dtype=float)
                keep_no_harm = np.ones(sf2.shape[0], dtype=bool)
                _sb_tol_hz = float(getattr(self, "subbass_protection_tolerance_hz", 12.0) or 12.0)
                for _j, f_hz in enumerate(sf2):
                    if not np.isfinite(f_hz):
                        keep_no_harm[_j] = False
                        continue
                    if np.any(np.abs(hp_f - f_hz) <= _sb_tol_hz):
                        keep_no_harm[_j] = False
                sub_df = sub_df.loc[keep_no_harm].copy()

        i_sum = (
            float(pd.to_numeric(ih_df["Amplitude"], errors="coerce").fillna(0.0).sum())
            if not ih_df.empty and "Amplitude" in ih_df.columns
            else 0.0
        )
        s_sum = (
            float(pd.to_numeric(sub_df["Amplitude"], errors="coerce").fillna(0.0).sum())
            if not sub_df.empty and "Amplitude" in sub_df.columns
            else 0.0
        )
        return float(max(0.0, h_sum)), float(max(0.0, i_sum)), float(max(0.0, s_sum))

    def _linear_component_density_balance_triple_with_basis(self) -> Tuple[float, float, float, str]:
        """Linear ΣA triple for the component-density pie (workbook-aligned)."""
        h = float(getattr(self, "linear_sum_amplitude_harmonic", 0.0) or 0.0)
        i = float(getattr(self, "linear_sum_amplitude_inharmonic_partial", 0.0) or 0.0)
        s = float(getattr(self, "linear_sum_amplitude_subbass_band", 0.0) or 0.0)
        basis = "linear_sum_amplitude_metrics"
        if not (math.isfinite(h) and math.isfinite(i) and math.isfinite(s)):
            h2, i2, s2 = self._reconstruct_linear_component_density_balance_triple()
            return h2, i2, s2, "reconstructed_from_internal_frames"
        tot = h + i + s
        if tot > 1e-18:
            return h, i, s, basis
        h2, i2, s2 = self._reconstruct_linear_component_density_balance_triple()
        tot2 = h2 + i2 + s2
        if tot2 > 1e-18:
            return h2, i2, s2, "reconstructed_from_internal_frames"
        return h, i, s, basis

    def _preferred_component_amplitude_sum_triple(self) -> Optional[Tuple[float, float, float]]:
        """(H, I, S) from ``harmonic_amplitude_sum`` / ``inharmonic_amplitude_sum`` / ``subbass_amplitude_sum`` when all exist."""
        names = ("harmonic_amplitude_sum", "inharmonic_amplitude_sum", "subbass_amplitude_sum")
        for n in names:
            if not hasattr(self, n):
                return None
        vals: List[float] = []
        for n in names:
            raw = getattr(self, n, None)
            if raw is None:
                return None
            try:
                vals.append(float(raw))
            except (TypeError, ValueError):
                return None
        h, i, s = vals[0], vals[1], vals[2]
        if not (math.isfinite(h) and math.isfinite(i) and math.isfinite(s)):
            return None
        if h < 0.0 or i < 0.0 or s < 0.0:
            return None
        if h + i + s <= 1e-18:
            return None
        return h, i, s

    def _component_amplitude_mass_triple_for_pie(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float]], str, List[str], str]:
        """Return (triple, metadata_basis_token, preferred_field_gaps, internal_tech_basis)."""
        preferred_missing: List[str] = []
        for n in ("harmonic_amplitude_sum", "inharmonic_amplitude_sum", "subbass_amplitude_sum"):
            if not hasattr(self, n) or getattr(self, n, None) is None:
                preferred_missing.append(n)
        preferred = self._preferred_component_amplitude_sum_triple()
        if preferred is not None:
            return preferred, "harmonic_amplitude_sum", preferred_missing, "harmonic_amplitude_sum_triple"
        h, i, s, basis = self._linear_component_density_balance_triple_with_basis()
        tot = float(h + i + s)
        if tot > 1e-18:
            return (h, i, s), "linear_amplitude_sum", preferred_missing, basis
        return None, "linear_amplitude_sum", preferred_missing, basis

    def _component_energy_ratio_triple_for_pie(
        self,
    ) -> Tuple[Optional[Tuple[float, float, float]], List[str], List[str]]:
        """Return (H, I, S) energy ratios or ``None``, with missing-field diagnostics."""
        primary_missing: List[str] = []
        if getattr(self, "harmonic_energy_ratio", None) is None:
            primary_missing.append("harmonic_energy_ratio")
        if getattr(self, "inharmonic_energy_ratio", None) is None:
            primary_missing.append("inharmonic_energy_ratio")
        trip = _energy_ratio_pie_values(
            getattr(self, "harmonic_energy_ratio", None),
            getattr(self, "inharmonic_energy_ratio", None),
            getattr(self, "subbass_energy_ratio", None),
        )
        if trip is not None:
            return trip, primary_missing, []
        fb_missing: List[str] = []
        if getattr(self, "component_harmonic_energy_ratio", None) is None:
            fb_missing.append("component_harmonic_energy_ratio")
        if getattr(self, "component_inharmonic_energy_ratio", None) is None:
            fb_missing.append("component_inharmonic_energy_ratio")
        trip2 = _energy_ratio_pie_values(
            getattr(self, "component_harmonic_energy_ratio", None),
            getattr(self, "component_inharmonic_energy_ratio", None),
            getattr(self, "component_subbass_energy_ratio", None),
        )
        if trip2 is not None:
            return trip2, primary_missing, fb_missing
        return None, primary_missing, fb_missing

    def _save_component_balance_pies(self, output_folder: Path, note: str) -> None:
        """Export component pies: amplitude-mass diagnostic vs power/energy ratios.

        Amplitude wedges prefer ``*_amplitude_sum`` when all three exist; otherwise
        ``linear_sum_amplitude_*``. Energy wedges prefer ``harmonic_energy_ratio`` /
        ``inharmonic_energy_ratio`` / ``subbass_energy_ratio``, else ``component_*``.
        """
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        self.component_energy_pie_basis = "not_written"
        self.amplitude_mass_chart_file = ""
        self.energy_ratio_chart_file = ""
        self.amplitude_mass_chart_basis = "linear_amplitude_sum"
        self.amplitude_mass_chart_interpretation = "diagnostic_candidate_mass_not_energy"
        self.energy_ratio_chart_basis = "component_power_energy_ratios"
        self.energy_ratio_chart_interpretation = "acoustic_energy_balance"
        self.amplitude_mass_chart_status = "not_attempted"
        self.energy_ratio_chart_status = "not_attempted"
        self.component_energy_pie_file = ""
        self.component_energy_pie_alias_basis = ""

        # --- Amplitude-mass (diagnostic; linear ΣA / preferred amplitude sums) ---
        try:
            triple, amp_meta_basis, preferred_gaps, pie_tech_basis = (
                self._component_amplitude_mass_triple_for_pie()
            )
            if triple is None:
                lh = getattr(self, "linear_sum_amplitude_harmonic", None)
                li = getattr(self, "linear_sum_amplitude_inharmonic_partial", None)
                ls = getattr(self, "linear_sum_amplitude_subbass_band", None)
                self.amplitude_mass_chart_file = ""
                self.amplitude_mass_chart_basis = ""
                if lh is None and li is None and ls is None:
                    self.amplitude_mass_chart_status = "skipped_missing_component_amplitude_sums"
                    self.logger.warning(
                        "Candidate amplitude-mass pie skipped: missing values "
                        "[linear_sum_amplitude_harmonic, linear_sum_amplitude_inharmonic_partial, "
                        "linear_sum_amplitude_subbass_band]",
                    )
                else:
                    self.amplitude_mass_chart_status = "skipped_no_positive_finite_values"
                    self.logger.warning(
                        "Candidate amplitude-mass pie skipped: no positive finite amplitude-mass values",
                    )
                if preferred_gaps:
                    self.logger.info(
                        "Candidate amplitude-mass pie: preferred amplitude-sum fields incomplete %s; "
                        "linear fallback was unavailable or summed to zero.",
                        preferred_gaps,
                    )
            else:
                h, i, s = triple
                import matplotlib.pyplot as plt

                sizes = [h, i, s]
                colors = ("#2ecc71", "#e74c3c", "#3498db")
                fig, ax = plt.subplots(figsize=(6.0, 5.6))
                wedges, _texts, autotexts = ax.pie(
                    sizes,
                    colors=colors,
                    startangle=90,
                    autopct="%1.1f%%",
                    pctdistance=0.72,
                    wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
                )
                for t in autotexts:
                    t.set_fontsize(9)
                    t.set_color("0.15")
                ax.legend(
                    wedges,
                    list(_COMPONENT_AMPLITUDE_MASS_PIE_LEGEND_LABELS),
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=10,
                    title_fontsize=10,
                    frameon=True,
                    fancybox=False,
                    edgecolor="0.75",
                    facecolor="0.98",
                )
                ax.set_title(
                    f"{COMPONENT_AMPLITUDE_MASS_PIE_TITLE_PREFIX} — {note}",
                    fontsize=11,
                )
                ax.set_aspect("equal")
                fig.subplots_adjust(bottom=0.14)
                fig.text(
                    0.5,
                    0.02,
                    COMPONENT_AMPLITUDE_MASS_PIE_BASIS_FOOTNOTE,
                    ha="center",
                    fontsize=8,
                    style="italic",
                    color="0.35",
                )
                amp_path = output_folder / COMPONENT_AMPLITUDE_MASS_PIE_FILENAME
                legacy_path = output_folder / COMPONENT_ENERGY_PIE_LEGACY_ALIAS_FILENAME
                fig.savefig(amp_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                self.component_energy_pie_basis = pie_tech_basis
                self.amplitude_mass_chart_basis = amp_meta_basis
                self.amplitude_mass_chart_file = COMPONENT_AMPLITUDE_MASS_PIE_FILENAME
                self.amplitude_mass_chart_status = "saved"
                self.logger.info("Candidate amplitude-mass pie saved: %s", amp_path)
                self.logger.info(
                    "Basis: linear amplitude sums; not power/energy ratios.",
                )
                try:
                    shutil.copyfile(amp_path, legacy_path)
                    self.component_energy_pie_file = COMPONENT_ENERGY_PIE_LEGACY_ALIAS_FILENAME
                    self.component_energy_pie_alias_basis = "legacy_alias_of_amplitude_mass_chart"
                    self.logger.info(
                        "Legacy component_energy_pie.png copied from component_amplitude_mass_pie.png: %s",
                        legacy_path,
                    )
                except OSError as copy_exc:
                    self.logger.warning(
                        "Legacy component_energy_pie.png copy failed (source=%s target=%s): %s",
                        amp_path,
                        legacy_path,
                        copy_exc,
                    )
        except Exception as exc:
            self.component_energy_pie_basis = "error"
            self.amplitude_mass_chart_status = "error"
            self.logger.warning("Could not save component amplitude-mass pie: %s", exc)

        # --- Energy / power ratios ---
        try:
            trip, pmiss, fmiss = self._component_energy_ratio_triple_for_pie()
            if trip is None:
                union = sorted({*pmiss, *fmiss})
                self.energy_ratio_chart_file = ""
                self.energy_ratio_chart_basis = ""
                if union:
                    self.energy_ratio_chart_status = "skipped_missing_component_energy_ratios"
                    self.logger.warning(
                        "Component energy-ratio pie skipped: missing values %s",
                        union,
                    )
                else:
                    self.energy_ratio_chart_status = "skipped_no_positive_finite_values"
                    self.logger.warning(
                        "Component energy-ratio pie skipped: no positive finite energy-ratio values",
                    )
            else:
                hf, inf, sf = trip
                import matplotlib.pyplot as plt

                sizes = [hf, inf, sf]
                colors = ("#2ecc71", "#e74c3c", "#3498db")
                fig, ax = plt.subplots(figsize=(6.0, 5.6))
                wedges, _texts, autotexts = ax.pie(
                    sizes,
                    colors=colors,
                    startangle=90,
                    autopct="%1.1f%%",
                    pctdistance=0.72,
                    wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
                )
                for t in autotexts:
                    t.set_fontsize(9)
                    t.set_color("0.15")
                ax.legend(
                    wedges,
                    list(_COMPONENT_ENERGY_RATIO_PIE_LEGEND_LABELS),
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    fontsize=10,
                    title_fontsize=10,
                    frameon=True,
                    fancybox=False,
                    edgecolor="0.75",
                    facecolor="0.98",
                )
                ax.set_title(f"Component energy balance — {note}", fontsize=11)
                ax.set_aspect("equal")
                fig.subplots_adjust(bottom=0.14)
                fig.text(
                    0.5,
                    0.02,
                    COMPONENT_ENERGY_RATIO_PIE_BASIS_FOOTNOTE,
                    ha="center",
                    fontsize=8,
                    style="italic",
                    color="0.35",
                )
                en_path = output_folder / COMPONENT_ENERGY_RATIO_PIE_FILENAME
                fig.savefig(en_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                self.energy_ratio_chart_file = COMPONENT_ENERGY_RATIO_PIE_FILENAME
                self.energy_ratio_chart_status = "saved"
                self.logger.info("Component energy-ratio pie saved: %s", en_path)
                self.logger.info(
                    "Basis: harmonic_energy_ratio / inharmonic_energy_ratio / subbass_energy_ratio.",
                )
        except Exception as exc:
            self.energy_ratio_chart_status = "error"
            self.logger.warning("Could not save component energy-ratio pie: %s", exc)

    # ----------------- salvar resultados (grÃ¡ficos + excel) -----------------
    def save_results(self, output_folder: Union[str, Path], note: str) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)

        analysis_method = "STFT"  # Always STFT (zero padding and time averaging are STFT options)
        self.logger.info(f"Saving results ({analysis_method}) for '{note}' to {output_folder}")

        # garantir mÃ©tricas completas
        try:
            self._ensure_all_metrics_calculated()
        except Exception as e:
            self.logger.warning(f"Failed to close metrics before save: {e}")

        # grÃ¡ficos
        try:
            spectrogram_png_path = output_folder / "spectrogram.png"
            self.plot_spectrograms(path=spectrogram_png_path, note=note)
        except Exception as e:
            self.logger.error(f"Error saving spectrograms: {e}")

        if self.dissonance_enabled and self.dissonance_curve_enabled:
            try:
                if self.dissonance_compare_models:
                    comp = output_folder / "dissonance_comparison.png"
                    self.plot_dissonance_comparison(path=comp, note=note)
                # Only process models that are actually available (exclude removed models)
                from dissonance_models import list_available_models
                available_models = list_available_models()
                models_to_process = list(self.dissonance_values.keys()) if self.dissonance_compare_models else [self.dissonance_model]
                # Filter to only include available models (safety check for removed models like stolzenburg, spectral-autocorrelation)
                models_to_process = [m for m in models_to_process if m in available_models]
                for m in models_to_process:
                    if self.dissonance_curves.get(m) is not None:
                        cpath = output_folder / f"{str(m).lower()}_dissonance_curve.png"
                        self.plot_dissonance_curve(m, path=cpath, note=note)
            except Exception as e:
                self.logger.error(f"Error saving dissonance curves: {e}")

        # Excel (single workbook per note; no spectral_analysis_clean.xlsx sidecar)
        excel_path = output_folder / "spectral_analysis.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
                self._save_spectral_data_to_excel(writer, note, export_output_dir=output_folder)
            self.logger.info(f"Spectral analysis saved to: {excel_path}")
            legacy_clean = output_folder / "spectral_analysis_clean.xlsx"
            if legacy_clean.is_file():
                try:
                    legacy_clean.unlink()
                    self.logger.info("Removed legacy sidecar %s (single-workbook export only).", legacy_clean)
                except OSError as _unlink_exc:
                    self.logger.warning("Could not remove legacy sidecar %s: %s", legacy_clean, _unlink_exc)
        except PermissionError as pe:
            self.logger.error(f"Permission denied: {pe}")
        except RuntimeError:
            # AUDIT FIX (stale-pipeline guard) — schema-version /
            # raw-column / provenance validation failures MUST surface
            # to the caller. The old behaviour silently logged and
            # continued, which is exactly how the GUI ended up
            # consuming half-written legacy workbooks. Best-effort
            # clean-up of the partial file, then re-raise.
            try:
                if excel_path.exists():
                    excel_path.unlink()
            except Exception as _unlink_exc:
                self.logger.warning(
                    "Could not remove partial workbook %s after schema failure: %s",
                    excel_path, _unlink_exc,
                )
            raise
        except Exception as exc:
            self.logger.error(f"Error saving results: {exc}")

    def _get_interval_name(self, cents: float) -> Optional[str]:
        try:
            c = float(cents) % 1200.0
            intervals = {
                0: "Unison", 100: "Minor 2nd", 200: "Major 2nd", 300: "Minor 3rd",
                400: "Major 3rd", 500: "Perfect 4th", 600: "Tritone", 700: "Perfect 5th",
                800: "Minor 6th", 900: "Major 6th", 1000: "Minor 7th", 1100: "Major 7th", 1200: "Octave"
            }
            target = min(intervals.keys(), key=lambda k: abs(c - k))
            return intervals[target] if abs(c - target) <= 10.0 else None
        except Exception:
            return None

    def _build_main_metrics_export_row(
        self,
        note: str,
        *,
        h_psum: Any,
        i_psum: Any,
        s_psum: Any,
        t_psum: Any,
    ) -> Dict[str, Any]:
        """Build the single-row ``Metrics`` export dict (NaN-safe; includes provenance status columns)."""
        _cd0 = getattr(self, "canonical_density_v5_adapted", None)
        if _cd0 is None:
            _cd0 = getattr(self, "density_metric_value", None)
        main_metrics: Dict[str, Any] = {
            "Note": note,
            "weight_function": str(getattr(self, "weight_function", "linear") or "linear"),
            "canonical_density_v5_adapted": metric_float_or_nan(_cd0),
            "density_per_component": metric_float_or_nan(getattr(self, "density_per_component", None)),
            "discrete_metric_d3": metric_float_or_nan(getattr(self, "discrete_metric_d3", None)),
            "discrete_metric_d10": metric_float_or_nan(getattr(self, "discrete_metric_d10", None)),
            "discrete_metric_d17": metric_float_or_nan(getattr(self, "discrete_metric_d17", None)),
            "discrete_metric_d24": metric_float_or_nan(getattr(self, "discrete_metric_d24", None)),
            "density_formula_version": (self.density_formula_version or CANONICAL_DENSITY_FORMULA_VERSION),
            "density_source_formula": (self.density_source_formula or CANONICAL_DENSITY_SOURCE_FORMULA),
            "density_normalization_scope": (
                self.density_normalization_scope or "none_per_note_absolute_canonical"
            ),
            "density_normalization_denominator": metric_float_or_nan(
                getattr(self, "density_normalization_denominator", None)
            ),
            "effective_partial_density": metric_float_or_nan(getattr(self, "effective_partial_density", None)),
            "body_weighted_effective_density": metric_float_or_nan(
                getattr(self, "body_weighted_effective_density", None)
            ),
            "low_mid_energy_ratio": metric_float_or_nan(getattr(self, "low_mid_energy_ratio", None)),
            "harmonic_body_density": metric_float_or_nan(getattr(self, "harmonic_body_density", None)),
            "expected_harmonic_slots_up_to_5000hz": metric_int_or_nan(
                getattr(self, "expected_harmonic_slots_up_to_5000hz", None)
            ),
            "harmonic_body_density_normalized": metric_float_or_nan(
                getattr(self, "harmonic_body_density_normalized", None)
            ),
            "residual_body_contribution": metric_float_or_nan(
                getattr(self, "residual_body_contribution", None)
            ),
            "residual_body_contribution_capped": metric_float_or_nan(
                getattr(self, "residual_body_contribution_capped", None)
            ),
            "salient_harmonic_order_count_up_to_5000hz": metric_int_or_nan(
                getattr(self, "salient_harmonic_order_count_up_to_5000hz", None)
            ),
            "expected_harmonic_order_count_up_to_5000hz": metric_int_or_nan(
                getattr(self, "expected_harmonic_order_count_up_to_5000hz", None)
            ),
            "salient_harmonic_coverage_up_to_5000hz": metric_float_or_nan(
                getattr(self, "salient_harmonic_coverage_up_to_5000hz", None)
            ),
            "salient_harmonic_mass_up_to_5000hz": metric_float_or_nan(
                getattr(self, "salient_harmonic_mass_up_to_5000hz", None)
            ),
            "salient_harmonic_order_count_up_to_density_ceiling_hz": metric_int_or_nan(
                getattr(self, "salient_harmonic_order_count_up_to_density_ceiling_hz", None)
            ),
            "expected_harmonic_order_count_up_to_density_ceiling_hz": metric_int_or_nan(
                getattr(self, "expected_harmonic_order_count_up_to_density_ceiling_hz", None)
            ),
            "salient_harmonic_coverage_up_to_density_ceiling_hz": metric_float_or_nan(
                getattr(self, "salient_harmonic_coverage_up_to_density_ceiling_hz", None)
            ),
            "salient_harmonic_mass_up_to_density_ceiling_hz": metric_float_or_nan(
                getattr(self, "salient_harmonic_mass_up_to_density_ceiling_hz", None)
            ),
            "salient_odd_harmonic_count_up_to_5000hz": metric_int_or_nan(
                getattr(self, "salient_odd_harmonic_count_up_to_5000hz", None)
            ),
            "salient_even_harmonic_count_up_to_5000hz": metric_int_or_nan(
                getattr(self, "salient_even_harmonic_count_up_to_5000hz", None)
            ),
            "odd_even_harmonic_energy_ratio": metric_float_or_nan(
                getattr(self, "odd_even_harmonic_energy_ratio", None)
            ),
            "salient_inharmonic_log_bin_count_up_to_5000hz": metric_int_or_nan(
                getattr(self, "salient_inharmonic_log_bin_count_up_to_5000hz", None)
            ),
            "salient_subbass_particle_count": metric_int_or_nan(
                getattr(self, "salient_subbass_particle_count", None)
            ),
            "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz": metric_int_or_nan(
                getattr(self, "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz", None)
            ),
            "salient_subbass_particle_count_up_to_density_ceiling_hz": metric_int_or_nan(
                getattr(self, "salient_subbass_particle_count_up_to_density_ceiling_hz", None)
            ),
            "final_note_density_count_based": metric_float_or_nan(
                getattr(self, "final_note_density_count_based", None)
            ),
            "final_note_density_salience_weighted": metric_float_or_nan(
                getattr(self, "final_note_density_salience_weighted", None)
            ),
            "harmonic_density_component": metric_float_or_nan(
                getattr(self, "harmonic_density_component", None)
            ),
            "inharmonic_density_component": metric_float_or_nan(
                getattr(self, "inharmonic_density_component", None)
            ),
            "subbass_density_component": metric_float_or_nan(
                getattr(self, "subbass_density_component", None)
            ),
            "harmonic_density_weight": metric_float_or_nan(
                getattr(self, "harmonic_density_weight", None)
            ),
            "inharmonic_density_weight": metric_float_or_nan(
                getattr(self, "inharmonic_density_weight", None)
            ),
            "subbass_density_weight": metric_float_or_nan(
                getattr(self, "subbass_density_weight", None)
            ),
            "density_summation_mode": str(getattr(self, "density_summation_mode", "") or ""),
            "density_salience_threshold_db": metric_float_or_nan(
                getattr(self, "density_salience_threshold_db", None)
            ),
            "density_frequency_ceiling_hz": metric_float_or_nan(
                getattr(self, "density_frequency_ceiling_hz", None)
            ),
            "spectral_body_thickness_index": metric_float_or_nan(
                getattr(self, "spectral_body_thickness_index", None)
            ),
            "harmonic_occupancy_ratio": metric_float_or_nan(
                getattr(self, "harmonic_occupancy_ratio", None)
            ),
            "harmonic_occupancy_detected_order_count": metric_int_or_nan(
                getattr(self, "harmonic_occupancy_detected_order_count", None)
            ),
            "expected_harmonic_slot_count": metric_int_or_nan(
                getattr(self, "expected_harmonic_slot_count", None)
            ),
            "detected_harmonic_slot_count": metric_int_or_nan(
                getattr(self, "detected_harmonic_slot_count", None)
            ),
            "harmonic_slot_expected_count": metric_int_or_nan(
                getattr(self, "harmonic_slot_expected_count", getattr(self, "expected_harmonic_slot_count", None))
            ),
            "harmonic_slot_matched_count": metric_int_or_nan(
                getattr(self, "harmonic_slot_matched_count", None)
            ),
            "harmonic_slot_coverage_ratio": metric_float_or_nan(
                (
                    float(getattr(self, "harmonic_slot_matched_count", np.nan))
                    / float(
                        getattr(
                            self,
                            "harmonic_slot_expected_count",
                            getattr(self, "expected_harmonic_slot_count", np.nan),
                        )
                    )
                )
                if (
                    getattr(
                        self,
                        "harmonic_slot_expected_count",
                        getattr(self, "expected_harmonic_slot_count", None),
                    )
                    is not None
                    and float(
                        getattr(
                            self,
                            "harmonic_slot_expected_count",
                            getattr(self, "expected_harmonic_slot_count", 0),
                        )
                        or 0
                    )
                    > 0
                )
                else None
            ),
            "harmonic_effective_power_density_normalized": metric_float_or_nan(
                getattr(
                    self,
                    "harmonic_effective_power_density_normalized",
                    getattr(self, "harmonic_effective_power_density_normalized_by_harmonic_count", None),
                )
            ),
            "residual_log_frequency_occupancy": metric_float_or_nan(
                getattr(self, "residual_log_frequency_occupancy", None)
            ),
            "residual_energy_ratio": metric_float_or_nan(
                getattr(
                    self,
                    "residual_energy_ratio",
                    getattr(self, "component_residual_noise_energy_ratio", None),
                )
            ),
            "harmonic_energy_sum": metric_float_or_nan(getattr(self, "harmonic_energy_sum", None)),
            "inharmonic_energy_sum": metric_float_or_nan(getattr(self, "inharmonic_energy_sum", None)),
            "subbass_energy_sum": metric_float_or_nan(getattr(self, "subbass_energy_sum", None)),
            "total_component_energy": metric_float_or_nan(getattr(self, "total_component_energy", None)),
            "harmonic_energy_ratio": metric_float_or_nan(getattr(self, "harmonic_energy_ratio", None)),
            "inharmonic_energy_ratio": metric_float_or_nan(getattr(self, "inharmonic_energy_ratio", None)),
            "subbass_energy_ratio": metric_float_or_nan(getattr(self, "subbass_energy_ratio", None)),
            "core_harmonic_energy_ratio": metric_float_or_nan(getattr(self, "harmonic_energy_ratio", None)),
            "core_residual_energy_ratio": metric_float_or_nan(getattr(self, "residual_energy_ratio", None)),
            "core_subbass_energy_ratio": metric_float_or_nan(getattr(self, "subbass_energy_ratio", None)),
            "linear_sum_amplitude_harmonic": metric_float_or_nan(
                getattr(self, "linear_sum_amplitude_harmonic", None)
            ),
            "linear_sum_amplitude_inharmonic_partial": metric_float_or_nan(
                getattr(self, "linear_sum_amplitude_inharmonic_partial", None)
            ),
            "linear_sum_amplitude_subbass_band": metric_float_or_nan(
                getattr(self, "linear_sum_amplitude_subbass_band", None)
            ),
            "linear_amplitude_fraction_inharmonic_of_HI": metric_float_or_nan(
                getattr(self, "linear_amplitude_fraction_inharmonic_of_HI", None)
            ),
            "linear_amplitude_fraction_nonharmonic_of_total": metric_float_or_nan(
                getattr(self, "linear_amplitude_fraction_nonharmonic_of_total", None)
            ),
            "linear_amplitude_batch_alignment_factor": metric_float_or_nan(
                getattr(self, "linear_amplitude_batch_alignment_factor", None)
            ),
            "Harmonic Partials sum": metric_float_or_nan(h_psum),
            "Inharmonic Partials sum": metric_float_or_nan(i_psum),
            "Sub-bass sum": metric_float_or_nan(s_psum),
            "Total sum": metric_float_or_nan(t_psum),
            "component_energy_status": getattr(self, "component_energy_status", "not_computed"),
            "effective_partial_density_status": getattr(
                self, "effective_partial_density_status", "not_computed"
            ),
            "density_metric_status": getattr(self, "density_metric_status", "not_computed"),
            "energy_weighted_component_density_diagnostic": metric_float_or_nan(
                getattr(
                    self,
                    "energy_weighted_component_density_diagnostic",
                    getattr(self, "density_metric_value", None),
                )
            ),
            "arithmetic_validation_status": getattr(self, "arithmetic_validation_status", "passed"),
            "acoustic_validation_status": getattr(self, "acoustic_validation_status", "passed"),
            "normalization_status": getattr(self, "normalization_status", "not_computed"),
            "model_weight_status": getattr(self, "model_weight_status", "not_computed"),
        }
        hoc = getattr(self, "harmonic_order_count", None)
        if hoc is None:
            hoc = getattr(self, "unique_harmonic_order_count", None)
        if hoc is not None:
            try:
                main_metrics["harmonic_order_count"] = int(hoc)
            except (TypeError, ValueError):
                pass
        uhq = getattr(self, "unique_harmonic_order_count", None)
        if uhq is not None:
            try:
                main_metrics["unique_harmonic_order_count"] = int(uhq)
            except (TypeError, ValueError):
                pass
        ent = getattr(self, "entropy_spectral_value", None)
        if ent is not None and np.isfinite(float(ent)):
            main_metrics["spectral_entropy"] = float(ent)

        main_metrics["density_metric_per_harmonic"] = metric_float_or_nan(
            getattr(self, "density_metric_per_harmonic", None)
        )
        main_metrics["density_metric_normalized"] = metric_float_or_nan(
            getattr(self, "density_metric_normalized", None)
        )

        for _opt in ("effective_partial_count", "harmonic_completeness", "harmonic_inharmonic_ratio"):
            _v = metric_float_or_nan(getattr(self, _opt, None))
            if np.isfinite(_v):
                main_metrics[_opt] = float(_v)

        for _canon in (
            "component_harmonic_energy_ratio",
            "component_inharmonic_energy_ratio",
            "component_subbass_energy_ratio",
            "component_total_inharmonic_energy_ratio",
            "model_harmonic_weight",
            "model_inharmonic_weight",
        ):
            main_metrics[_canon] = metric_float_or_nan(getattr(self, _canon, None))

        try:
            _f0m = getattr(self, "f0_final", None)
            main_metrics["f0_final_hz"] = (
                float(_f0m)
                if _f0m is not None and np.isfinite(float(_f0m)) and float(_f0m) > 0.0
                else metric_float_or_nan(None)
            )
        except (TypeError, ValueError):
            main_metrics["f0_final_hz"] = metric_float_or_nan(None)
        _f0_used_hz, _f0_used_src, _f0_used_status = self._canonical_f0_triplet_for_analysis()
        main_metrics["f0_used_for_density_hz"] = (
            float(_f0_used_hz) if np.isfinite(float(_f0_used_hz)) else metric_float_or_nan(None)
        )
        main_metrics["f0_used_for_density_source"] = str(_f0_used_src)
        main_metrics["f0_used_for_harmonic_validation_hz"] = (
            metric_float_or_nan(getattr(self, "f0_used_for_harmonic_validation_hz", _f0_used_hz))
        )
        main_metrics["acoustic_f0_status"] = str(
            getattr(self, "acoustic_f0_status", _f0_used_status) or _f0_used_status
        )
        main_metrics["f0_fit_accepted"] = bool(getattr(self, "f0_fit_accepted", False))
        main_metrics["f0_fit_rejection_reason"] = str(
            getattr(self, "f0_fit_rejection_reason", "") or ""
        )

        main_metrics["low_frequency_policy_version"] = str(
            getattr(self, "low_frequency_policy_version", "") or LOW_FREQUENCY_POLICY_VERSION
        )
        main_metrics["adaptive_subfundamental_cutoff_hz"] = metric_float_or_nan(
            getattr(self, "adaptive_subfundamental_cutoff_hz", None)
        )
        main_metrics["subfundamental_margin_percent"] = metric_float_or_nan(
            getattr(self, "subfundamental_margin_percent", None)
        )
        main_metrics["percentage_subfundamental_cutoff_hz"] = metric_float_or_nan(
            getattr(self, "percentage_subfundamental_cutoff_hz", None)
        )
        main_metrics["leakage_guard_cutoff_hz"] = metric_float_or_nan(
            getattr(self, "leakage_guard_cutoff_hz", None)
        )
        main_metrics["effective_subfundamental_margin_percent"] = metric_float_or_nan(
            getattr(self, "effective_subfundamental_margin_percent", None)
        )
        main_metrics["subfundamental_cutoff_selection_rule"] = str(
            getattr(self, "subfundamental_cutoff_selection_rule", "") or ""
        )
        main_metrics["subfundamental_cutoff_selected_by"] = str(
            getattr(self, "subfundamental_cutoff_selected_by", "") or ""
        )
        main_metrics["subfundamental_guard_valid"] = bool(
            getattr(self, "subfundamental_guard_valid", False)
        )
        main_metrics["subfundamental_guard_policy"] = str(
            getattr(self, "subfundamental_guard_policy", "") or ""
        )
        main_metrics["physical_low_frequency_lower_hz"] = metric_float_or_nan(
            getattr(self, "physical_low_frequency_lower_hz", None)
        )
        main_metrics["physical_low_frequency_upper_hz"] = metric_float_or_nan(
            getattr(self, "physical_low_frequency_upper_hz", None)
        )

        try:
            from metadata_sanitizer import publication_clean_export_enabled as _pub_clean_metrics  # noqa: PLC0415
        except Exception:  # pragma: no cover

            def _pub_clean_metrics() -> bool:  # type: ignore[misc]
                return True

        if _pub_clean_metrics():
            _cdv = main_metrics.get("canonical_density_v5_adapted")
            try:
                if _cdv is not None and np.isfinite(float(_cdv)):
                    main_metrics["canonical_density"] = float(_cdv)
            except (TypeError, ValueError):
                pass
            main_metrics.pop("density_formula_version", None)

        return main_metrics

    def _build_legacy_density_metrics_row(self, note: str) -> Dict[str, Any]:
        """Per-note legacy scalars (SDM / FDM / CDM) for compile WCM and v5 comparison — default export."""
        _dm = getattr(self, "canonical_density_v5_adapted", None)
        if _dm is None:
            _dm = getattr(self, "density_metric_value", None)
        return {
            "Note": note,
            "weight_function": str(getattr(self, "weight_function", "linear") or "linear"),
            "Density Metric": metric_float_or_nan(_dm),
            "Spectral Density Metric": metric_float_or_nan(
                getattr(self, "spectral_density_metric_value", None)
            ),
            "Filtered Density Metric": metric_float_or_nan(
                getattr(self, "filtered_density_metric_value", None)
            ),
            "Combined Density Metric": metric_float_or_nan(
                getattr(self, "combined_density_metric_value", None)
            ),
            "spectral_masking_enabled": False,
            "legacy_density_export_version": "1",
        }

    def _save_spectral_data_to_excel(
        self,
        writer: pd.ExcelWriter,
        note: str,
        export_output_dir: Optional[Path] = None,
    ) -> None:
        import numpy as np
        import pandas as pd

        from low_frequency_policy import classify_low_frequency_row

        log = self.logger
        try:
            try:
                from metadata_sanitizer import publication_redaction_enabled, sanitize_dataframe_for_publication

                def _pub_df(dfx):
                    if dfx is None or getattr(dfx, "empty", True):
                        return dfx
                    if publication_redaction_enabled():
                        return sanitize_dataframe_for_publication(dfx)
                    return dfx

            except Exception:

                def _pub_df(dfx):
                    return dfx

            # ===== 1. ESPECTROS (DADOS BRUTOS) =====
            def _ensure_amp_column(df: pd.DataFrame) -> pd.DataFrame:
                if df is None or df.empty:
                    return df
                if "Amplitude" not in df.columns:
                    if "Magnitude (dB)" in df.columns:
                        df = df.copy()
                        # Fórmula física correta: A = 10^(dB / 20)
                        df["Amplitude"] = np.power(10.0, pd.to_numeric(df["Magnitude (dB)"], errors="coerce").fillna(-120.0) / 20.0)
                return df

            if isinstance(self.complete_list_df, pd.DataFrame) and not self.complete_list_df.empty:
                df_complete = _ensure_amp_column(self.complete_list_df)
                cols = [c for c in ["Frequency (Hz)", "Magnitude (dB)", "Amplitude", "Note"] if c in df_complete.columns]
                (df_complete[cols] if cols else df_complete).to_excel(writer, sheet_name="Complete Spectrum", index=False)
                log.debug(f"Espectro completo salvo: {len(df_complete)}")

            if isinstance(self.filtered_list_df, pd.DataFrame) and not self.filtered_list_df.empty:
                df_filt = _ensure_amp_column(self.filtered_list_df)
                cols = [c for c in ["Frequency (Hz)", "Magnitude (dB)", "Amplitude", "Note"] if c in df_filt.columns]
                (df_filt[cols] if cols else df_filt).to_excel(writer, sheet_name="Filtered Spectrum", index=False)
                log.debug(f"Espectro filtrado salvo: {len(df_filt)}")

            # AUDIT FIX (stale-pipeline guard) — hoist the harmonic
            # spectrum DataFrame to method scope so the pre-metadata
            # schema validator can inspect the *exact* columns we are
            # writing.
            harm_export_for_validation: pd.DataFrame = pd.DataFrame()
            # Stage 1 harmonic-spectrum split:
            #
            #   * ``Harmonic Spectrum`` sheet is built from the
            #     ``harmonic_spectrum_candidates_df`` (one row per expected
            #     harmonic order, classified by candidate_status) so the
            #     Density_Metrics ``harmonic_log_amplitude_density`` metric
            #     has the dense per-order population it needs.
            #   * ``Strict_Harmonic_Peaks`` diagnostics sheet mirrors the
            #     historic strict list (``harmonic_list_df``) and feeds
            #     inharmonic classification / the robust f0 fit.
            cand_src = getattr(self, "harmonic_spectrum_candidates_df", None)
            harmonic_sheet_df = (
                cand_src.copy()
                if isinstance(cand_src, pd.DataFrame) and not cand_src.empty
                else pd.DataFrame()
            )
            if harmonic_sheet_df.empty and isinstance(
                self.harmonic_list_df, pd.DataFrame
            ) and not self.harmonic_list_df.empty:
                # Fall back to the strict list if the candidate dataframe
                # is empty for any reason; the schema guard still requires
                # Amplitude_raw / Power_raw to be present.
                harmonic_sheet_df = _ensure_amp_column(self.harmonic_list_df).copy()
            if not harmonic_sheet_df.empty:
                # Ensure Amplitude_raw / Power_raw are present even on the
                # strict-only fallback path. Harmonic Spectrum is never
                # subject to k_align scaling, so Amplitude == Amplitude_raw
                # by construction.
                if (
                    "Amplitude_raw" not in harmonic_sheet_df.columns
                    and "Amplitude" in harmonic_sheet_df.columns
                ):
                    _amps_raw_h = (
                        pd.to_numeric(
                            harmonic_sheet_df["Amplitude"], errors="coerce"
                        )
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
                    harmonic_sheet_df["Amplitude_raw"] = _amps_raw_h
                    harmonic_sheet_df["Power_raw"] = _amps_raw_h ** 2
                # Stable column order with the candidate-only metadata
                # appended after the historical core columns.
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
                cols = [c for c in preferred_cols if c in harmonic_sheet_df.columns]
                _harm_to_write = (
                    harmonic_sheet_df[cols] if cols else harmonic_sheet_df
                )
                _harm_to_write.to_excel(
                    writer, sheet_name="Harmonic Spectrum", index=False
                )
                harm_export_for_validation = _harm_to_write
                log.debug(
                    "Harmonic Spectrum candidates exported: rows=%d",
                    len(harmonic_sheet_df),
                )

            # Strict diagnostics sheet (always written when at least one
            # strict-validated peak survived; the sheet is *not* read by
            # Density_Metrics).
            if isinstance(self.harmonic_list_df, pd.DataFrame) and not self.harmonic_list_df.empty:
                strict_df = _ensure_amp_column(self.harmonic_list_df).copy()
                if "Amplitude" in strict_df.columns:
                    _amps_raw_s = (
                        pd.to_numeric(strict_df["Amplitude"], errors="coerce")
                        .fillna(0.0)
                        .to_numpy(dtype=float)
                    )
                    strict_df["Amplitude_raw"] = _amps_raw_s
                    strict_df["Power_raw"] = _amps_raw_s ** 2
                strict_cols = [
                    c
                    for c in [
                        "Harmonic Number",
                        "Frequency (Hz)",
                        "Magnitude (dB)",
                        "Amplitude",
                        "Amplitude_raw",
                        "Power_raw",
                        "SNR_dB",
                        "SubBinCorrected",
                        "Note",
                    ]
                    if c in strict_df.columns
                ]
                _strict_to_write = strict_df[strict_cols] if strict_cols else strict_df
                _strict_to_write.to_excel(
                    writer, sheet_name="Strict_Harmonic_Peaks", index=False
                )
                log.debug(
                    "Strict_Harmonic_Peaks exported: rows=%d", len(strict_df)
                )

            # ===== 2. GARANTIR MÉTRICAS (antes de export inarmónico / sub-bass alinhado) =====
            try:
                self._ensure_all_metrics_calculated()
            except Exception as e:
                log.warning(f"_ensure_all_metrics_calculated falhou: {e}")
                self._set_default_metrics()

            # Acoustic / bookkeeping separation (export):
            # - ``Inharmonic Spectrum`` (sheet name retained): non-harmonic peak **candidates**
            #   after harmonic-window exclusion and amplitude ranking — **not** confirmed inharmonic partials.
            #   Same population drives ``inharmonic_energy_sum`` bookkeeping; linear sums on ``Metrics`` are re-derived
            #   here after optional batch-aligned scaling of export amplitudes.
            # - ``Sub-bass band``: low-frequency spectral bins for context only; energy accounting uses
            #   ``subbass_energy_sum`` (aggregate ground-noise power), not a row-sum of this sheet.
            # Linear amplitudes on both sheets are scaled by a common k so
            # k*(ΣA_IH+ΣA_SB) <= min(((p_I+p_S)/p_H)*ΣA_H, ΣA_H) using batch GUI ratios when present.
            # Per-note batch / GUI energy ratios are repeated on each row of both exports when available.
            def _gwm_batch_energy(key: str) -> Any:
                md = getattr(self, "gui_weight_resolution_meta", None) or {}
                if isinstance(md, dict):
                    return md.get(key)
                return None

            _ih_identified, _ih_selected = self._nonharmonic_residual_pipeline_dataframes()
            self._assign_hierarchical_residual_debug_counts(_ih_identified, _ih_selected)
            ih_raw = _ih_selected.copy()

            ih_df = _ensure_amp_column(ih_raw.copy()) if ih_raw is not None and not ih_raw.empty else pd.DataFrame()

            _cut_sb = float(getattr(self, "subbass_aggregate_hz", SUBBASS_AGGREGATE_CUTOFF_HZ))
            # AUDIT FIX (acoustic-physics, Clarinete_mf finding #1) —
            # apply the same lower-frequency floor that the energy
            # aggregator uses. Bins below this floor are DC / sub-audible
            # (room rumble, HVAC, mic DC offset, FFT-leakage from the
            # DC bin) and have no musical content. Showing them on the
            # Sub-bass band sheet misleads analysts into thinking the
            # corresponding ``subbass_energy_sum`` reflects real audio.
            _lo_sb = float(getattr(
                self, "subbass_aggregate_lower_hz", SUBBASS_AGGREGATE_LOWER_HZ
            ))
            sub_df = pd.DataFrame()
            if isinstance(getattr(self, "complete_list_df", None), pd.DataFrame) and not self.complete_list_df.empty:
                compf = _ensure_amp_column(self.complete_list_df.copy())
                if "Frequency (Hz)" in compf.columns:
                    ff = pd.to_numeric(compf["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
                    mask_sb = np.isfinite(ff) & (ff > _lo_sb) & (ff <= _cut_sb)
                    sub_df = compf.loc[mask_sb].copy()

            if not ih_df.empty and not sub_df.empty and "Frequency (Hz)" in ih_df.columns and "Frequency (Hz)" in sub_df.columns:
                ih_f = pd.to_numeric(ih_df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
                ih_f = ih_f[np.isfinite(ih_f)]
                sf = pd.to_numeric(sub_df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
                keep_sb = np.zeros(sf.shape[0], dtype=bool)
                for i, f in enumerate(sf):
                    if not np.isfinite(f):
                        continue
                    thr_m = max(1e-6, abs(f) * 1e-6)
                    if ih_f.size == 0 or not np.any(np.abs(ih_f - f) < thr_m):
                        keep_sb[i] = True
                sub_df = sub_df.loc[keep_sb].copy()

            # AUDIT FIX (Fgt_pp finding L1 + acoustic-physics Clarinete_mf
            # finding #2) — apply the SAME harmonic-exclusion that the
            # energy aggregator uses (canonical population: strict
            # harmonics ∪ Harmonic Spectrum candidates with
            # include_for_density==True), AND derive the tolerance
            # window from the current FFT configuration via
            # ``compute_subbass_protection_tolerance_hz`` instead of the
            # legacy 12 Hz constant. The 12 Hz constant was narrower
            # than every realistic FFT window's main-lobe and let
            # fundamental leakage shoulders (typically 2–5 FFT bins
            # from the peak) survive on the sheet AND in the
            # aggregator, inflating ``subbass_energy_sum``.
            try:
                harm_protect_df = self._build_subbass_harmonic_protection_df()
            except Exception:
                harm_protect_df = pd.DataFrame({"Frequency (Hz)": pd.Series(dtype=float)})
            if (
                not sub_df.empty
                and "Frequency (Hz)" in sub_df.columns
                and not harm_protect_df.empty
                and "Frequency (Hz)" in harm_protect_df.columns
            ):
                hp_f = pd.to_numeric(
                    harm_protect_df["Frequency (Hz)"], errors="coerce"
                ).to_numpy(dtype=float)
                hp_f = hp_f[np.isfinite(hp_f)]
                if hp_f.size > 0:
                    sf2 = pd.to_numeric(
                        sub_df["Frequency (Hz)"], errors="coerce"
                    ).to_numpy(dtype=float)
                    keep_no_harm = np.ones(sf2.shape[0], dtype=bool)
                    _sb_tol_hz = float(getattr(
                        self, "subbass_protection_tolerance_hz", 12.0
                    ) or 12.0)
                    for i, f in enumerate(sf2):
                        if not np.isfinite(f):
                            keep_no_harm[i] = False
                            continue
                        if np.any(np.abs(hp_f - f) <= _sb_tol_hz):
                            keep_no_harm[i] = False
                    sub_df = sub_df.loc[keep_no_harm].copy()

            h_for_sum = (
                _ensure_amp_column(self.harmonic_list_df.copy())
                if isinstance(getattr(self, "harmonic_list_df", None), pd.DataFrame)
                and not self.harmonic_list_df.empty
                else pd.DataFrame()
            )
            s_h = (
                float(pd.to_numeric(h_for_sum["Amplitude"], errors="coerce").fillna(0.0).sum())
                if not h_for_sum.empty and "Amplitude" in h_for_sum.columns
                else 0.0
            )
            s_ih_raw = (
                float(pd.to_numeric(ih_df["Amplitude"], errors="coerce").fillna(0.0).sum())
                if not ih_df.empty and "Amplitude" in ih_df.columns
                else 0.0
            )
            s_sb_raw = (
                float(pd.to_numeric(sub_df["Amplitude"], errors="coerce").fillna(0.0).sum())
                if not sub_df.empty and "Amplitude" in sub_df.columns
                else 0.0
            )

            # AUDIT FIX (inharmonic-energy underestimation +
            # current_analysis hardening) — the legacy
            # ``linear_export_batch_alignment_k`` step shrank the exported
            # Inharmonic / Sub-bass amplitudes towards a batch energy
            # budget computed from harmonic_weight=0.95 / inharmonic_weight=0.05,
            # so any real signal whose actual inharmonic share exceeded 5 %
            # of the harmonic mass had its non-harmonic export amplitudes
            # silently divided by a factor ``k << 1``. In single-pass mode
            # this is wrong by construction: the canonical component
            # ratios are derived from the raw energies of this very
            # analysis, so any rescaling biases them.
            #
            # The audit further hardens the predicate so the legacy path
            # is *also* disabled whenever the per-note helper
            # ``_set_model_weights_from_current_component_energy`` has set
            # ``self.model_weights_source = "current_analysis"`` —
            # otherwise the orchestrator log would still emit the legacy
            # alignment message after the canonical weights have already
            # been written, which is the failure mode we observed in the
            # field.
            #
            # Policy:
            # * integrated_single_pass / current_analysis (default):
            #   NEVER apply k_align; export the raw amplitudes; tag
            #   provenance with ``disabled_integrated_single_pass``.
            # * legacy_batch (opt-in only): still compute k_align but
            #   write the scaled values to a separate
            #   ``Amplitude_display_scaled`` column instead of
            #   overwriting the raw ``Amplitude``.
            # Export alignment is always disabled in the current-analysis
            # pipeline. There is no external energy mapping to align to:
            # the per-note spectrum is itself the source of truth.
            k_align = 1.0
            self.export_alignment_factor = 1.0
            self.export_alignment_source = "disabled_integrated_single_pass"
            self.linear_amplitude_batch_alignment_factor = 1.0
            log.info(
                "Export alignment disabled (current_analysis mode): "
                "factor=1.0; raw amplitudes preserved as Amplitude_raw."
            )

            def _attach_raw_and_display_amplitude_columns(
                df: pd.DataFrame, k: float
            ) -> pd.DataFrame:
                """Promote the per-row ``Amplitude`` to the audit-canonical
                ``Amplitude_raw`` / ``Power_raw`` pair and, when a non-unit
                ``k`` is provided (legacy_batch only), expose the scaled
                values as ``Amplitude_display_scaled`` without
                overwriting the raw column.

                ``Power_raw = Amplitude_raw ** 2`` keeps the per-row power
                accessible to the Density_Metrics direct extractor and to
                any audit that wants ``Σ A²`` from a single source of
                truth.
                """
                if df is None or getattr(df, "empty", True):
                    return df
                out = df.copy()
                if "Amplitude" not in out.columns:
                    return out
                amps_raw = (
                    pd.to_numeric(out["Amplitude"], errors="coerce")
                    .fillna(0.0)
                    .to_numpy(dtype=float)
                )
                out["Amplitude_raw"] = amps_raw
                out["Power_raw"] = amps_raw ** 2
                if (k is not None) and (float(k) < 1.0 - 1e-15):
                    scaled = amps_raw * float(k)
                    out["Amplitude_display_scaled"] = scaled
                return out

            ih_df = _attach_raw_and_display_amplitude_columns(ih_df, k_align)
            sub_df = _attach_raw_and_display_amplitude_columns(sub_df, k_align)

            sum_ih = float(s_ih_raw)
            sum_sb = float(s_sb_raw)
            nh = sum_ih + sum_sb
            den_all = s_h + nh
            self.linear_sum_amplitude_harmonic = float(s_h)
            self.linear_sum_amplitude_inharmonic_partial = float(sum_ih)
            self.linear_sum_amplitude_subbass_band = float(sum_sb)
            self.linear_amplitude_fraction_nonharmonic_of_total = (
                float(nh / den_all) if den_all > 1e-30 else 0.0
            )
            den_hi = s_h + sum_ih
            self.linear_amplitude_fraction_inharmonic_of_HI = (
                float(sum_ih / den_hi) if den_hi > 1e-30 else 0.0
            )

            def _tag_component_type(
                dfx: pd.DataFrame,
                cat: str,
                *,
                classification_level: str,
                acoustic_status: str,
            ) -> pd.DataFrame:
                if dfx is None or dfx.empty:
                    return pd.DataFrame()
                out = dfx.copy()
                out.insert(0, "Component_Type", cat)
                out.insert(1, "Classification_Level", classification_level)
                out.insert(2, "Acoustic_Interpretation_Status", acoustic_status)
                return out

            ih_partials = _tag_component_type(
                ih_df,
                "nonharmonic_peak_candidate",
                classification_level="residual_rows_ranked_by_amplitude_after_harmonic_exclusion",
                acoustic_status="candidate_not_confirmed_partial",
            )
            _dc_lo_sb = float(getattr(self, "subbass_aggregate_lower_hz", SUBBASS_AGGREGATE_LOWER_HZ))
            _phys_hi_sb = float(getattr(self, "subbass_aggregate_hz", SUBBASS_AGGREGATE_CUTOFF_HZ))
            try:
                _ad_raw = getattr(self, "adaptive_subfundamental_cutoff_hz", None)
                _adf = float(_ad_raw) if _ad_raw is not None else float("nan")
            except (TypeError, ValueError):
                _adf = float("nan")
            _ad_cut_sb = _adf if np.isfinite(_adf) and _adf > 0.0 else float(_dc_lo_sb)
            if sub_df is None or sub_df.empty:
                sb_rows = pd.DataFrame()
            else:
                sb_rows = sub_df.copy()
                sb_rows.insert(0, "Component_Type", "low_frequency_residual_row")
                sb_rows.insert(1, "Classification_Level", "diagnostic_fixed_frequency_band_residual")
                _fq_sb = pd.to_numeric(sb_rows["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float)
                _lf_classes: list[str] = []
                for _f in _fq_sb:
                    if not np.isfinite(float(_f)):
                        _lf_classes.append("not_low_frequency_residual")
                    else:
                        _lf_classes.append(
                            classify_low_frequency_row(
                                float(_f),
                                dc_floor_hz=_dc_lo_sb,
                                physical_low_band_upper_hz=_phys_hi_sb,
                                adaptive_subfundamental_cutoff_hz=_ad_cut_sb,
                            )
                        )
                sb_rows.insert(2, "Low_Frequency_Class", _lf_classes)
                sb_rows.insert(
                    3,
                    "Acoustic_Interpretation_Status",
                    "diagnostic_low_frequency_residual_not_partial",
                )

            def _attach_note_column(dfx: pd.DataFrame) -> pd.DataFrame:
                """Ensure the per-row spectrum sheet carries the Note column.

                In current-analysis mode the user-facing Inharmonic Spectrum
                and Sub-bass band sheets never carry internal-only
                compatibility columns; they expose Amplitude_raw / Power_raw
                only.
                """
                out = dfx.copy() if dfx is not None else pd.DataFrame()
                if "Note" not in out.columns:
                    out["Note"] = note
                elif not out.empty:
                    out["Note"] = out["Note"].fillna(note)
                return out

            ih_export = _attach_note_column(ih_partials)
            sb_export = _attach_note_column(sb_rows)

            if sb_export is None or sb_export.empty:
                sb_export = pd.DataFrame(
                    columns=[
                        "Component_Type",
                        "Classification_Level",
                        "Low_Frequency_Class",
                        "Acoustic_Interpretation_Status",
                        "Frequency (Hz)",
                        "Magnitude (dB)",
                        "Amplitude",
                        "Note",
                    ]
                )
                sb_export = _attach_note_column(sb_export)

            if ih_export.empty:
                ih_export = pd.DataFrame(
                    columns=[
                        "Component_Type",
                        "Classification_Level",
                        "Acoustic_Interpretation_Status",
                        "Frequency (Hz)",
                        "Magnitude (dB)",
                        "Amplitude",
                        "Note",
                    ]
                )
                ih_export = _attach_note_column(ih_export)

            _pub_df(ih_export).to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
            log.debug("Inharmonic Spectrum (parciais inarmónicos): %s rows", len(ih_export))

            if sb_export is not None and not sb_export.empty:
                _pub_df(sb_export).to_excel(writer, sheet_name="Sub-bass band", index=False)
                log.debug("Sub-bass band (bins espectrais): %s rows", len(sb_export))

            # ===== 3. MÉTRICAS CONSOLIDADAS =====
            # FIXED: Use actual hop_length value, don't recalculate from n_fft
            # This ensures tier-specific hop_length values are preserved
            hl = getattr(self, "hop_length", None)
            if hl is None:
                # Fallback: use n_fft // 2 only if hop_length was never set
                hl = int(getattr(self, "n_fft", 4096)) // 2
            else:
                hl = int(hl)

            # ``Harmonic Partials sum`` … ``Total sum``: ``partial_metric_sums_h_i_s_total`` — **linear** on per-band
            # ΣA² scalars when the UI key is continuous (``log``/``sqrt``/… still use linear here for energy bookkeeping).
            # Discrete UI keys use native partial vectors. ``weight_function`` continues to drive canonical density / SDM.
            _h_psum, _i_psum, _s_psum, _t_psum = self._partial_metric_sums_for_metrics_export(
                h_for_sum, ih_df, sub_df
            )

            # Folha ``Metrics``: apenas núcleo do modelo de densidade efetiva (sem PCA, dissonância,
            # parâmetros STFT, nem campos de masking). Parâmetros técnicos → ``Analysis_Metadata``.
            # effective_partial_density: número efetivo de componentes espectrais energeticamente relevantes
            # (riqueza espectral / “gordura”), não loudness. Masking perceptivo fica desativado no fluxo
            # principal (ver ``spectral_masking_enabled`` em Analysis_Metadata).
            main_metrics = self._build_main_metrics_export_row(
                note,
                h_psum=_h_psum,
                i_psum=_i_psum,
                s_psum=_s_psum,
                t_psum=_t_psum,
            )

            metrics_df = _pub_df(pd.DataFrame([main_metrics]))
            # Column order UX: keep ``discrete_metric_d*`` at the **end**. They are independent
            # descriptors (different scales); placing them early (e.g. cols G–H) caused Excel line
            # charts across A–H to show a false "cliff" between D17 and D24 for every instrument.
            try:
                from metadata_sanitizer import publication_clean_export_enabled as _pub_col_order  # noqa: PLC0415
            except Exception:  # pragma: no cover

                def _pub_col_order() -> bool:  # type: ignore[misc]
                    return True

            _metrics_sheet_column_order = [
                "Note",
                "weight_function",
                "canonical_density_v5_adapted",
            ]
            if _pub_col_order():
                _metrics_sheet_column_order.append("canonical_density")
            _metrics_sheet_column_order.extend(
                [
                    "density_per_component",
                ]
            )
            if not _pub_col_order():
                _metrics_sheet_column_order.append("density_formula_version")
            _metrics_sheet_column_order.extend(
                [
                    "density_source_formula",
                    "density_normalization_scope",
                    "density_normalization_denominator",
                    "effective_partial_density",
                    "harmonic_energy_sum",
                    "inharmonic_energy_sum",
                    "subbass_energy_sum",
                    "total_component_energy",
                    "harmonic_energy_ratio",
                    "inharmonic_energy_ratio",
                    "subbass_energy_ratio",
                    "linear_sum_amplitude_harmonic",
                    "linear_sum_amplitude_inharmonic_partial",
                    "linear_sum_amplitude_subbass_band",
                    "linear_amplitude_fraction_inharmonic_of_HI",
                    "linear_amplitude_fraction_nonharmonic_of_total",
                    "linear_amplitude_batch_alignment_factor",
                    "Harmonic Partials sum",
                    "Inharmonic Partials sum",
                    "Sub-bass sum",
                    "Total sum",
                    "harmonic_order_count",
                    "unique_harmonic_order_count",
                    "spectral_entropy",
                    "harmonic_completeness",
                    "harmonic_inharmonic_ratio",
                    "effective_partial_count",
                    "component_harmonic_energy_ratio",
                    "component_inharmonic_energy_ratio",
                    "component_subbass_energy_ratio",
                    "component_total_inharmonic_energy_ratio",
                    "model_harmonic_weight",
                    "model_inharmonic_weight",
                    "component_energy_status",
                    "effective_partial_density_status",
                    "density_metric_status",
                    "normalization_status",
                    "model_weight_status",
                    "density_metric_per_harmonic",
                    "density_metric_normalized",
                    "f0_final_hz",
                    "low_frequency_policy_version",
                    "adaptive_subfundamental_cutoff_hz",
                    "subfundamental_margin_percent",
                    "percentage_subfundamental_cutoff_hz",
                    "leakage_guard_cutoff_hz",
                    "effective_subfundamental_margin_percent",
                    "subfundamental_cutoff_selection_rule",
                    "subfundamental_cutoff_selected_by",
                    "subfundamental_guard_valid",
                    "subfundamental_guard_policy",
                    "physical_low_frequency_lower_hz",
                    "physical_low_frequency_upper_hz",
                    "discrete_metric_d3",
                    "discrete_metric_d10",
                    "discrete_metric_d17",
                    "discrete_metric_d24",
                ]
            )
            _ordered = [c for c in _metrics_sheet_column_order if c in metrics_df.columns]
            _rest = [c for c in metrics_df.columns if c not in _ordered]
            metrics_df = metrics_df[_ordered + _rest]
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

            legacy_row = self._build_legacy_density_metrics_row(note)
            legacy_df = _pub_df(pd.DataFrame([legacy_row]))
            _legacy_order = [
                "Note",
                "weight_function",
                "Density Metric",
                "Spectral Density Metric",
                "Filtered Density Metric",
                "Combined Density Metric",
                "spectral_masking_enabled",
                "legacy_density_export_version",
            ]
            _leg_ord = [c for c in _legacy_order if c in legacy_df.columns]
            _leg_rest = [c for c in legacy_df.columns if c not in _leg_ord]
            legacy_df = legacy_df[_leg_ord + _leg_rest]
            legacy_df.to_excel(writer, sheet_name="Legacy_Density_Metrics", index=False)

            self.exported_nonharmonic_peak_candidate_count = int(len(ih_export))
            from debug_counts import (
                DEBUG_COUNTS_SEMANTICS,
                DEBUG_COUNTS_SOURCE_POLICY,
                validate_debug_count_invariants,
            )

            _dbg_inv: dict = {
                "residual_spectral_row_count": getattr(self, "residual_spectral_row_count", None),
                "nonharmonic_candidate_row_count": getattr(self, "nonharmonic_candidate_row_count", None),
                "retained_nonharmonic_peak_candidate_count": getattr(
                    self, "retained_nonharmonic_peak_candidate_count", None
                ),
                "exported_nonharmonic_peak_candidate_count": getattr(
                    self, "exported_nonharmonic_peak_candidate_count", None
                ),
                "accepted_inharmonic_peak_count": getattr(self, "accepted_inharmonic_peak_count", None),
                "accepted_inharmonic_partial_count": getattr(self, "accepted_inharmonic_partial_count", None),
            }
            validate_debug_count_invariants(_dbg_inv)
            self.debug_counts_invariant_status = str(_dbg_inv.get("debug_counts_invariant_status", "") or "")
            self.debug_counts_invariant_failures = str(_dbg_inv.get("debug_counts_invariant_failures", "") or "")

            debug_row = {
                "Note": note,
                "harmonic_bin_count": metric_int_or_nan(getattr(self, "harmonic_bin_count", None)),
                "subbass_bin_count": metric_int_or_nan(getattr(self, "subbass_bin_count", None)),
                "harmonic_peak_candidate_count": metric_int_or_nan(
                    getattr(self, "harmonic_peak_candidate_count", None)
                ),
                "low_frequency_peak_candidate_count": metric_int_or_nan(
                    getattr(self, "low_frequency_peak_candidate_count", None)
                ),
                "total_peak_candidate_count": metric_int_or_nan(
                    getattr(self, "total_peak_candidate_count", None)
                ),
                "residual_spectral_row_count": metric_int_or_nan(
                    getattr(self, "residual_spectral_row_count", None)
                ),
                "nonharmonic_candidate_row_count": metric_int_or_nan(
                    getattr(self, "nonharmonic_candidate_row_count", None)
                ),
                "retained_nonharmonic_peak_candidate_count": metric_int_or_nan(
                    getattr(self, "retained_nonharmonic_peak_candidate_count", None)
                ),
                "exported_nonharmonic_peak_candidate_count": metric_int_or_nan(
                    getattr(self, "exported_nonharmonic_peak_candidate_count", None)
                ),
                "peaklist_harmonic_window_candidate_count": metric_int_or_nan(
                    getattr(self, "peaklist_harmonic_window_candidate_count", None)
                ),
                "peaklist_nonharmonic_window_candidate_count": metric_int_or_nan(
                    getattr(self, "peaklist_nonharmonic_window_candidate_count", None)
                ),
                "peaklist_low_frequency_window_candidate_count": metric_int_or_nan(
                    getattr(self, "peaklist_low_frequency_window_candidate_count", None)
                ),
                "peaklist_total_window_candidate_count": metric_int_or_nan(
                    getattr(self, "peaklist_total_window_candidate_count", None)
                ),
                "legacy_nonharmonic_peak_candidate_count_deprecated": metric_int_or_nan(
                    getattr(self, "peaklist_nonharmonic_window_candidate_count", None)
                ),
                "accepted_inharmonic_peak_count": metric_int_or_nan(
                    getattr(self, "accepted_inharmonic_peak_count", None)
                ),
                "accepted_inharmonic_partial_count": metric_int_or_nan(
                    getattr(self, "accepted_inharmonic_partial_count", None)
                ),
                "total_spectral_candidate_count": metric_int_or_nan(
                    getattr(self, "total_spectral_candidate_count", None)
                ),
                "harmonic_candidate_count": metric_int_or_nan(
                    getattr(self, "harmonic_candidate_count", None)
                ),
                "subbass_candidate_count": metric_int_or_nan(
                    getattr(self, "subbass_candidate_count", None)
                ),
                "residual_row_count": metric_int_or_nan(getattr(self, "residual_row_count", None)),
                "debug_counts_semantics": str(_dbg_inv.get("debug_counts_semantics") or DEBUG_COUNTS_SEMANTICS),
                "debug_counts_source_policy": str(
                    _dbg_inv.get("debug_counts_source_policy") or DEBUG_COUNTS_SOURCE_POLICY
                ),
                "debug_counts_invariant_status": str(_dbg_inv.get("debug_counts_invariant_status", "") or ""),
                "debug_counts_invariant_failures": str(_dbg_inv.get("debug_counts_invariant_failures", "") or ""),
                "debug_counts_status": getattr(self, "debug_counts_status", "not_computed"),
                # Deprecated legacy column names — same integers as historically emitted; do not
                # interpret as confirmed inharmonic partials or acoustic peak validation.
                "inharmonic_bin_count_deprecated_legacy_alias": metric_int_or_nan(
                    getattr(self, "inharmonic_bin_count", None)
                ),
                "inharmonic_candidate_count_deprecated_legacy_alias": metric_int_or_nan(
                    getattr(self, "inharmonic_candidate_count", None)
                ),
                "inharmonic_peak_count_deprecated_legacy_alias": metric_int_or_nan(
                    getattr(self, "inharmonic_peak_count", None)
                ),
                "harmonic_peak_count_deprecated_legacy_alias": metric_int_or_nan(
                    getattr(self, "harmonic_peak_count", None)
                ),
                "subbass_peak_count_deprecated_legacy_alias": metric_int_or_nan(
                    getattr(self, "subbass_peak_count", None)
                ),
                "total_detected_peak_count_deprecated_legacy_alias": metric_int_or_nan(
                    getattr(self, "total_detected_peak_count", None)
                ),
            }
            _pub_df(pd.DataFrame([debug_row])).to_excel(writer, sheet_name="Debug_Counts", index=False)

            _hv = getattr(self, "harmonic_validation_report", None)
            if isinstance(_hv, dict) and _hv:
                _val_row = {"Note": note, **_hv}
                _pub_df(pd.DataFrame([_val_row])).to_excel(writer, sheet_name="Validation_Metrics", index=False)

            tol_hz = float(getattr(self, "tolerance", 10.0) or 10.0)
            _sr = float(getattr(self, "sr", None) or getattr(self, "sample_rate", None) or 44100.0)
            _nff = int(getattr(self, "n_fft", 4096))
            _zp = int(getattr(self, "zero_padding", 1) or 1)
            try:
                _nff_eff = int(self._get_actual_n_fft())
            except Exception:
                _nff_eff = int(_nff * max(1, _zp))
            _f0e = None
            if isinstance(_hv, dict) and _hv.get("f0_estimated") is not None:
                try:
                    _f0e = float(_hv["f0_estimated"])
                except (TypeError, ValueError):
                    _f0e = None
            _f0src = None
            if isinstance(_hv, dict) and _hv.get("f0_source"):
                _f0src = str(_hv["f0_source"])

            def _window_str_for_export() -> str:
                w = getattr(self, "window", None)
                if w is None:
                    return str(DEFAULT_WINDOW)
                ws = str(w).strip()
                if not ws or ws.lower() == "none":
                    return str(DEFAULT_WINDOW)
                return ws

            _src_name = None
            for _y, _sr_ad, _n, _fp in getattr(self, "audio_data", []) or []:
                if str(_n) == str(note):
                    try:
                        _src_name = Path(str(_fp)).name
                    except Exception:
                        _src_name = str(_fp)
                    break

            _gwm = getattr(self, "gui_weight_resolution_meta", None)
            if not isinstance(_gwm, dict):
                _gwm = {}

            per_note_row: Dict[str, Any] = {
                "Note": note,
                "source_file_name": _src_name,
                "n_fft": _nff,
                "n_fft_effective": _nff_eff,
                "hop_length": int(hl),
                "bin_spacing_hz": float(_calculate_bin_spacing(_sr, _nff, _zp)),
                "sample_rate": _sr,
                "window": _window_str_for_export(),
                "tier": getattr(self, "tier", None),
                "f0_estimated": _f0e,
                "f0_source": _f0src,
                "f0_used_for_density_hz": getattr(self, "f0_used_for_density_hz", None),
                "f0_used_for_density_source": getattr(self, "f0_used_for_density_source", None),
                "f0_used_for_harmonic_validation_hz": getattr(
                    self, "f0_used_for_harmonic_validation_hz", None
                ),
                "f0_used_for_harmonic_validation_source": getattr(
                    self, "f0_used_for_harmonic_validation_source", None
                ),
                "f0_fit_accepted": getattr(self, "f0_fit_accepted", None),
                "f0_fit_rejection_reason": getattr(self, "f0_fit_rejection_reason", None),
                "acoustic_f0_status": getattr(self, "acoustic_f0_status", None),
                "harmonic_tolerance": float(tol_hz),
                "snr_threshold_db": float(SNR_THRESHOLD_DB),
                "rms_normalisation_enabled": True,
                "smoothing_enabled": bool(getattr(self, "spectral_magnitude_smoothing_enabled", False)),
                "spectral_masking_enabled": bool(getattr(self, "spectral_masking_enabled", False)),
            }
            for _k in (
                "component_harmonic_energy_ratio",
                "component_inharmonic_energy_ratio",
                "component_subbass_energy_ratio",
                "component_total_inharmonic_energy_ratio",
                "component_energy_denominator",
                "component_energy_method",
                "model_weight_denominator",
                "model_weights_source",
                "model_weights_warning",
                "model_weights_fallback_reason",
            ):
                if _k not in _gwm:
                    continue
                _vv = _gwm.get(_k)
                if _vv is None:
                    continue
                try:
                    if isinstance(_vv, (int, float)) and np.isfinite(float(_vv)):
                        per_note_row[_k] = float(_vv)
                    elif isinstance(_vv, str):
                        per_note_row[_k] = str(_vv)
                except Exception:
                    per_note_row[_k] = str(_vv)
            _mh_ap = float(getattr(self, "harmonic_weight", 0.95) or 0.95)
            _mi_ap = float(getattr(self, "inharmonic_weight", 0.05) or 0.05)
            per_note_row["model_harmonic_weight"] = _mh_ap
            per_note_row["model_inharmonic_weight"] = _mi_ap
            # AUDIT FIX — override ``model_weights_source`` when the single-
            # pass helper actually overwrote the placeholder weights. The
            # ``_gwm`` payload still announces the original API source even
            # though the rewrite happened in
            # ``_set_model_weights_from_current_component_energy``.
            _self_mws_pn = getattr(self, "model_weights_source", None)
            if _self_mws_pn:
                per_note_row["model_weights_source"] = str(_self_mws_pn)
            _self_cps_pn = getattr(self, "component_profile_source", None)
            if _self_cps_pn:
                per_note_row["component_profile_source"] = str(_self_cps_pn)
            _pub_df(pd.DataFrame([per_note_row])).to_excel(
                writer, sheet_name="Per_Note_Processing_Metadata", index=False
            )

            try:
                from compile_metrics import _get_project_version_info as _gv
                _ver, _ = _gv()
            except Exception:
                _ver = "unknown"

            import datetime as _dtmod
            import platform as _plat
            import sys as _sys
            import uuid as _uuid

            def _pkg_ver(mod: str) -> str:
                try:
                    m = __import__(mod, fromlist=["_dummy"])
                    v = getattr(m, "__version__", None)
                    return str(v) if v is not None else "not_available_at_compile_stage"
                except Exception:
                    return "not_available_at_compile_stage"

            _val_stat = "exported" if isinstance(_hv, dict) and _hv else "skipped"

            def _meta_atom(v: Any) -> Any:
                if v is None:
                    return "not_available_at_compile_stage"
                return v

            def _dissonance_model_slug() -> str:
                raw = str(getattr(self, "dissonance_model", "") or "").strip()
                if not raw:
                    return ""
                return raw.split()[0].lower()

            _selected_dm_for_meta = _dissonance_model_slug()

            _component_energy_denominator = (
                _gwm.get("component_energy_denominator")
                or _gwm.get("batch_energy_denominator")
                or "harmonic_plus_inharmonic_plus_subbass"
            )
            _mwden = _gwm.get("model_weight_denominator") or "harmonic_plus_inharmonic"
            # AUDIT FIX — provenance: ``model_weights_source`` MUST reflect
            # whether ``_set_model_weights_from_current_component_energy``
            # actually overwrote the placeholder API weights. The single-
            # pass helper sets ``self.model_weights_source = "current_analysis"``
            # whenever it does so; prefer that over any pre-existing _gwm
            # label (which would still announce ``apply_filters_arguments``
            # despite the actual rewrite).
            _self_mws = getattr(self, "model_weights_source", None)
            if _self_mws:
                _mws_src = str(_self_mws)
            else:
                _mws_src = str(
                    getattr(self, "gui_model_weights_source", None)
                    or _gwm.get("model_weights_source")
                    or "apply_filters_arguments"
                )
            _mws_warn = str(getattr(self, "gui_model_weights_warning", "") or _gwm.get("model_weights_warning") or "")
            _mws_fb = _gwm.get("model_weights_fallback_reason")
            _mh_ap = float(getattr(self, "harmonic_weight", 0.95) or 0.95)
            _mi_ap = float(getattr(self, "inharmonic_weight", 0.05) or 0.05)

            # AUDIT FIX (stale-pipeline detection) — every per-note
            # workbook MUST carry the schema-version token plus the
            # resolved runtime paths so a downstream consumer can
            # reject stale exports without ambiguity.
            try:
                _runtime_paths = log_runtime_paths(self.logger)
            except Exception:
                _runtime_paths = {
                    "sys_executable": str(_sys.executable),
                    "cwd": str(Path.cwd()),
                    "proc_audio_file": str(Path(__file__).resolve()),
                    "compile_metrics_file": "<not_importable>",
                    "publication_chart_policy_file": "<not_importable>",
                    "proc_audio_runtime_signature": _proc_audio_runtime_signature(),
                }
            _export_align_factor = float(
                getattr(self, "export_alignment_factor", 1.0) or 1.0
            )
            _export_align_source = str(
                getattr(self, "export_alignment_source", "")
                or "disabled_integrated_single_pass"
            )

            _pipe_contract = get_canonical_pipeline_contract()
            analysis_meta_rows = [
                ("analysis_schema_version", ANALYSIS_SCHEMA_VERSION),
                ("ANALYSIS_SCHEMA_VERSION", ANALYSIS_SCHEMA_VERSION),
                ("pipeline_contract_version", _pipe_contract.contract_version),
                ("analysis_engine", "proc_audio.AudioProcessor"),
                ("analysis_engine_role", CANONICAL_PIPELINE_ROLE),
                ("canonical_output", True),
                ("legacy_pipeline_used", False),
                ("f0_policy_version", F0_POLICY_VERSION),
                ("harmonic_frequency_policy_version", HARMONIC_FREQUENCY_POLICY_VERSION),
                ("nonharmonic_policy_version", NONHARMONIC_POLICY_VERSION),
                ("low_frequency_policy_version", LOW_FREQUENCY_POLICY_VERSION),
                ("missing_metric_policy_version", MISSING_METRIC_POLICY_VERSION),
                ("density_formula_version", DENSITY_FORMULA_VERSION),
                ("export_schema_version", EXPORT_SCHEMA_VERSION),
                ("proc_audio_file", _runtime_paths.get("proc_audio_file", "")),
                (
                    "proc_audio_runtime_signature",
                    _runtime_paths.get("proc_audio_runtime_signature", "unknown"),
                ),
                (
                    "compile_metrics_file",
                    _runtime_paths.get("compile_metrics_file", "<not_importable>"),
                ),
                (
                    "publication_chart_policy_file",
                    _runtime_paths.get(
                        "publication_chart_policy_file", "<not_importable>"
                    ),
                ),
                ("sys_executable", _runtime_paths.get("sys_executable", "")),
                ("cwd", _runtime_paths.get("cwd", "")),
                ("export_alignment_factor", _export_align_factor),
                ("export_alignment_source", _export_align_source),
                # AUDIT FIX — also surface the residual / non-harmonic
                # ratios so the compile guard can see them at the
                # metadata layer (in addition to the Metrics sheet).
                (
                    "component_residual_noise_energy_ratio",
                    float(
                        getattr(self, "component_residual_noise_energy_ratio", 0.0)
                        or 0.0
                    ),
                ),
                (
                    "component_nonharmonic_energy_ratio",
                    float(
                        getattr(self, "component_nonharmonic_energy_ratio", 0.0)
                        or 0.0
                    ),
                ),
                (
                    "component_residual_energy_denominator",
                    str(
                        getattr(
                            self,
                            "component_residual_energy_denominator",
                            "H+I+S+residual",
                        )
                    ),
                ),
                ("analysis_version", _ver),
                ("run_id", str(_uuid.uuid4())),
                ("analysis_date", _dtmod.datetime.now().isoformat()),
                ("python_version", _meta_atom(_sys.version.split()[0] if _sys.version else None)),
                ("platform", _meta_atom(_plat.platform())),
                ("numpy_version", _pkg_ver("numpy")),
                ("scipy_version", _pkg_ver("scipy")),
                ("librosa_version", _pkg_ver("librosa")),
                ("window", _window_str_for_export()),
                ("window_type", _window_str_for_export()),
                ("n_fft", int(getattr(self, "n_fft", 4096))),
                ("n_fft_effective", int(_nff_eff)),
                ("hop_length", int(hl)),
                ("zero_padding", int(getattr(self, "zero_padding", 1) or 1)),
                ("frequency_min_hz", float(getattr(self, "freq_min", float("nan")))),
                ("frequency_max_hz", float(getattr(self, "freq_max", float("nan")))),
                ("magnitude_min_db", float(getattr(self, "db_min", float("nan")))),
                ("magnitude_max_db", float(getattr(self, "db_max", float("nan")))),
                ("rms_normalisation_enabled", True),
                ("smoothing_enabled", bool(getattr(self, "spectral_magnitude_smoothing_enabled", False))),
                ("spectral_masking_enabled", bool(getattr(self, "spectral_masking_enabled", False))),
                ("snr_threshold_db", float(SNR_THRESHOLD_DB)),
                ("harmonic_tolerance", tol_hz),
                (
                    "density_summation_mode",
                    str(getattr(self, "density_summation_mode", "his_weighted") or "his_weighted"),
                ),
                (
                    "harmonic_density_weight",
                    float(
                        getattr(self, "harmonic_density_weight", 1.0)
                        if getattr(self, "harmonic_density_weight", 1.0) is not None
                        else 1.0
                    ),
                ),
                (
                    "inharmonic_density_weight",
                    float(
                        getattr(self, "inharmonic_density_weight", 0.5)
                        if getattr(self, "inharmonic_density_weight", 0.5) is not None
                        else 0.5
                    ),
                ),
                (
                    "subbass_density_weight",
                    float(
                        getattr(self, "subbass_density_weight", 0.25)
                        if getattr(self, "subbass_density_weight", 0.25) is not None
                        else 0.25
                    ),
                ),
                (
                    "density_salience_threshold_db",
                    float(
                        getattr(self, "density_salience_threshold_db", -45.0)
                        if getattr(self, "density_salience_threshold_db", -45.0) is not None
                        else -45.0
                    ),
                ),
                (
                    "density_frequency_ceiling_hz",
                    float(
                        getattr(self, "density_frequency_ceiling_hz", 5000.0)
                        if getattr(self, "density_frequency_ceiling_hz", 5000.0) is not None
                        else 5000.0
                    ),
                ),
                (
                    "legacy_up_to_5000hz_columns_alias_density_ceiling",
                    bool(
                        abs(
                            float(
                                getattr(self, "density_frequency_ceiling_hz", 5000.0)
                                if getattr(self, "density_frequency_ceiling_hz", 5000.0) is not None
                                else 5000.0
                            )
                            - 5000.0
                        )
                        > 1e-9
                    ),
                ),
                ("per_note_analysis_metadata_scope", "this_note_single_file_export"),
                (
                    "sheet_Inharmonic_Spectrum_sheet_semantics",
                    "nonharmonic_peak_candidates_not_confirmed_partials",
                ),
                ("sheet_Inharmonic_Spectrum_inharmonic_partial_claim", False),
                (
                    "sheet_Inharmonic_Spectrum_classification_level",
                    "nonharmonic_peak_candidate",
                ),
                (
                    "sheet_Sub_bass_band_sheet_semantics",
                    "diagnostic_low_frequency_residuals_not_confirmed_subbass_or_noise",
                ),
                ("sheet_Sub_bass_band_subbass_noise_claim", False),
                ("sheet_Sub_bass_band_partial_claim", False),
                ("dc_offset_before_removal", _meta_atom(getattr(self, "dc_offset_before_removal", None))),
                ("dc_offset_after_removal", _meta_atom(getattr(self, "dc_offset_after_removal", None))),
                ("dc_removal_applied", bool(getattr(self, "dc_removal_applied", False))),
                (
                    "adaptive_subfundamental_cutoff_hz",
                    (
                        float(x)
                        if (x := getattr(self, "adaptive_subfundamental_cutoff_hz", None)) is not None
                        and np.isfinite(float(x))
                        else float("nan")
                    ),
                ),
                (
                    "subfundamental_margin_percent",
                    (
                        float(x)
                        if (x := getattr(self, "subfundamental_margin_percent", None)) is not None
                        and np.isfinite(float(x))
                        else float("nan")
                    ),
                ),
                ("subfundamental_guard_valid", bool(getattr(self, "subfundamental_guard_valid", False))),
                (
                    "subfundamental_guard_policy",
                    str(getattr(self, "subfundamental_guard_policy", "") or ""),
                ),
                (
                    "percentage_subfundamental_cutoff_hz",
                    (
                        float(x)
                        if (x := getattr(self, "percentage_subfundamental_cutoff_hz", None)) is not None
                        and np.isfinite(float(x))
                        else float("nan")
                    ),
                ),
                (
                    "leakage_guard_cutoff_hz",
                    (
                        float(x)
                        if (x := getattr(self, "leakage_guard_cutoff_hz", None)) is not None
                        and np.isfinite(float(x))
                        else float("nan")
                    ),
                ),
                (
                    "effective_subfundamental_margin_percent",
                    (
                        float(x)
                        if (x := getattr(self, "effective_subfundamental_margin_percent", None)) is not None
                        and np.isfinite(float(x))
                        else float("nan")
                    ),
                ),
                (
                    "subfundamental_cutoff_selection_rule",
                    str(getattr(self, "subfundamental_cutoff_selection_rule", "") or ""),
                ),
                (
                    "subfundamental_cutoff_selected_by",
                    str(getattr(self, "subfundamental_cutoff_selected_by", "") or ""),
                ),
                (
                    "physical_low_frequency_lower_hz",
                    float(
                        getattr(self, "physical_low_frequency_lower_hz", SUBBASS_AGGREGATE_LOWER_HZ)
                        or SUBBASS_AGGREGATE_LOWER_HZ
                    ),
                ),
                (
                    "physical_low_frequency_upper_hz",
                    float(
                        getattr(self, "physical_low_frequency_upper_hz", SUBBASS_AGGREGATE_CUTOFF_HZ)
                        or SUBBASS_AGGREGATE_CUTOFF_HZ
                    ),
                ),
                (
                    "low_frequency_aggregate_mode",
                    str(getattr(self, "low_frequency_aggregate_mode", "local_maxima") or "local_maxima"),
                ),
                (
                    "harmonic_leakage_protection_hz",
                    _meta_atom(getattr(self, "harmonic_leakage_protection_hz", None)),
                ),
                (
                    "spectral_density_freq_floor_hz",
                    _meta_atom(getattr(self, "spectral_density_freq_floor_hz", None)),
                ),
                (
                    "low_frequency_residual_interpretation",
                    str(getattr(self, "low_frequency_residual_interpretation", "") or ""),
                ),
                ("model_weight_policy", "current_analysis_component_HIS_projected_to_HI_model_weights"),
                ("component_energy_denominator", str(_component_energy_denominator)),
                ("model_weight_denominator", str(_mwden)),
                ("model_weights_source_policy", "per_note_when_available"),
                (
                    "component_ratio_sum_policy",
                    "component_harmonic_energy_ratio + component_inharmonic_energy_ratio + component_subbass_energy_ratio must sum to 1",
                ),
                ("external_component_profile_used", False),
                ("external_h_i_s_mapping_used", False),
                # AUDIT FIX (Fgt_pp finding M1) — stamp the actual note
                # label and the active weight_function alongside
                # ``note_source`` so the per-note workbook is internally
                # self-describing. Previously only ``note_source`` was
                # written, leaving ``Note`` and ``weight_function``
                # blank in Analysis_Metadata while the per-folder name
                # implied them.
                (
                    "Note",
                    str(getattr(self, "note", "") or ""),
                ),
                (
                    "weight_function",
                    str(getattr(self, "weight_function", "") or ""),
                ),
                # AUDIT FIX (canonical note-source provenance) — record
                # how the runtime ``self.note`` was resolved so the
                # compiled Density_Metrics row can surface
                # ``note_source`` end-to-end.
                (
                    "note_source",
                    str(
                        getattr(self, "note_source", "")
                        or "unknown"
                    ),
                ),
                ("f0_prior_note", _meta_atom(getattr(self, "f0_prior_note", None))),
                ("f0_prior_source", _meta_atom(getattr(self, "f0_prior_source", None))),
                ("f0_nominal_hz", _meta_atom(getattr(self, "f0_nominal_hz", None))),
                ("f0_prior_hz", _meta_atom(getattr(self, "f0_prior_hz", None))),
                # Stage 1 harmonic-extraction provenance (audit task).
                (
                    "expected_harmonic_count",
                    int(getattr(self, "expected_harmonic_count", 0) or 0),
                ),
                (
                    "strict_harmonic_count",
                    int(getattr(self, "strict_harmonic_count", 0) or 0),
                ),
                (
                    "harmonic_candidate_count",
                    int(
                        len(getattr(self, "harmonic_spectrum_candidates_df", None))
                        if isinstance(
                            getattr(self, "harmonic_spectrum_candidates_df", None),
                            pd.DataFrame,
                        )
                        else 0
                    ),
                ),
                (
                    "harmonic_density_included_count",
                    int(getattr(self, "harmonic_candidate_count_density", 0) or 0),
                ),
                (
                    "harmonic_amplitude_sum",
                    float(getattr(self, "harmonic_amplitude_sum", 0.0) or 0.0),
                ),
                (
                    "harmonic_log_amplitude_density",
                    float(
                        getattr(self, "harmonic_log_amplitude_density", 0.0) or 0.0
                    ),
                ),
                (
                    "f0_initial",
                    float(getattr(self, "f0_initial", 0.0) or 0.0),
                ),
                (
                    "f0_final",
                    float(getattr(self, "f0_final", 0.0) or 0.0),
                ),
                (
                    "f0_final_hz",
                    (
                        float(z)
                        if (z := getattr(self, "f0_final", None)) is not None
                        and np.isfinite(float(z))
                        and float(z) > 0.0
                        else float("nan")
                    ),
                ),
                (
                    "f0_final_source",
                    str(getattr(self, "f0_final_source", "") or ""),
                ),
                (
                    "f0_detuning_cents_from_nominal",
                    _meta_atom(getattr(self, "f0_detuning_cents_from_nominal", None)),
                ),
                # AUDIT FIX (Fgt_pp finding H2) — surface the global-fit
                # residual standard deviation so downstream consumers
                # can tell HOW bad the rejected fit was. Together with
                # the already-exported ``f0_final_method``,
                # ``f0_fit_accepted`` and ``f0_fit_quality`` (emitted
                # in the analysis-grade block further below) this
                # lets an operator decide whether the fallback to
                # ``nominal_or_initial_due_to_bad_fit`` is acceptable
                # for the use case (small residual but |Δf₀| > 2 %
                # vs. residual far above the threshold).
                (
                    "f0_fit_residual_std_hz",
                    float(
                        getattr(self, "f0_fit_residual_std_hz", None)
                        or getattr(self, "f0_robust_residual_std", 0.0)
                        or 0.0
                    ),
                ),
                (
                    "f0_fit_accepted",
                    bool(getattr(self, "f0_fit_accepted", False)),
                ),
                (
                    "f0_fit_quality",
                    _meta_atom(getattr(self, "f0_fit_quality", None)),
                ),
                (
                    "f0_fit_rejection_reason",
                    _meta_atom(getattr(self, "f0_fit_rejection_reason", None)),
                ),
                (
                    "f0_final_method",
                    str(
                        getattr(
                            self,
                            "f0_final_method",
                            "nominal_or_initial_due_to_bad_fit",
                        )
                    ),
                ),
                # AUDIT FIX (acoustic-physics correction, Clarinete_mf
                # findings #1 + #2) — surface the parameters actually
                # used by ``aggregate_low_frequency_residual_peak_power`` so any
                # downstream auditor can verify what window of frequency
                # space was treated as "sub-bass" and how wide the
                # harmonic-protection band around each harmonic was.
                # Without these the workbook is silent about two
                # critical sub-bass classification choices.
                (
                    "subbass_aggregate_cutoff_hz",
                    float(
                        getattr(self, "subbass_aggregate_hz",
                                SUBBASS_AGGREGATE_CUTOFF_HZ)
                    ),
                ),
                (
                    "subbass_aggregate_lower_hz",
                    float(
                        getattr(self, "subbass_aggregate_lower_hz",
                                SUBBASS_AGGREGATE_LOWER_HZ)
                    ),
                ),
                (
                    "subbass_protection_tolerance_hz",
                    float(
                        getattr(self, "subbass_protection_tolerance_hz", 12.0)
                        or 12.0
                    ),
                ),
                (
                    "density_formula",
                    "effective_partial_density = participation-ratio style effective number of "
                    "energetically relevant partials (harmonic + aggregated inharmonic + sub-bass aggregate); "
                    "effective component participation descriptor (not the primary perceived thickness metric).",
                ),
                ("effective_density_component_policy", EFFECTIVE_DENSITY_COMPONENT_POLICY_DOC),
                ("inharmonic_mode_for_effective_density", INHARMONIC_MODE_FOR_EFFECTIVE_DENSITY),
                ("subbass_policy_for_effective_density", SUBBASS_POLICY_FOR_EFFECTIVE_DENSITY_DOC),
                ("count_semantics_note", COUNT_SEMANTICS_NOTE_DOC),
                ("legacy_partial_count_aliases_note", LEGACY_PARTIAL_COUNT_ALIASES_NOTE),
                ("robust_salient_inharmonic_peak_picking_enabled", bool(ROBUST_SALIENT_INHARMONIC_PEAK_PICKING_ENABLED)),
                (
                    "harmonic_inharmonic_model_coefficients_note",
                    "model_harmonic_weight and model_inharmonic_weight are H/(H+I) and I/(H+I) coefficients for this note; "
                    "they differ from the H+I+S component ratios when sub-bass is nonzero.",
                ),
                ("applied_model_harmonic_weight", _mh_ap),
                ("applied_model_inharmonic_weight", _mi_ap),
                ("model_harmonic_weight", _mh_ap),
                ("model_inharmonic_weight", _mi_ap),
                # Canonical component_* / model_* metadata for current-analysis mode.
                # ``component_*`` use H+I+S as denominator (three-way partition);
                # ``model_*_weight`` use H+I as denominator (binary coefficients).
                # These keys are the single source of truth.
                (
                    "component_harmonic_energy_ratio",
                    float(getattr(self, "component_harmonic_energy_ratio", 0.0) or 0.0),
                ),
                (
                    "component_inharmonic_energy_ratio",
                    float(getattr(self, "component_inharmonic_energy_ratio", 0.0) or 0.0),
                ),
                (
                    "component_subbass_energy_ratio",
                    float(getattr(self, "component_subbass_energy_ratio", 0.0) or 0.0),
                ),
                (
                    "component_total_inharmonic_energy_ratio",
                    float(getattr(self, "component_total_inharmonic_energy_ratio", 0.0) or 0.0),
                ),
                (
                    "component_energy_denominator",
                    str(getattr(self, "component_energy_denominator", "H+I+S")),
                ),
                (
                    "component_energy_method",
                    str(
                        getattr(
                            self,
                            "component_energy_method",
                            "single_pass_proc_audio_energy",
                        )
                    ),
                ),
                (
                    "component_profile_source",
                    str(
                        getattr(
                            self,
                            "component_profile_source",
                            "current_analysis",
                        )
                    ),
                ),
                # SEMANTIC HARDENING — harmonic-only participation count.
                # Distinct from effective_partial_density (which blends
                # harmonic + inharmonic + sub-bass).
                (
                    "effective_partial_count",
                    float(getattr(self, "effective_partial_count", 0.0) or 0.0),
                ),
                # AUDIT FIX — canonical scalars computed during this single
                # analysis call. Persisted in Analysis_Metadata so the
                # compile step can harvest them into ``Canonical_Metrics``
                # when the per-note Metrics sheet does not carry them
                # explicitly (older workbooks).
                (
                    "harmonic_completeness",
                    float(getattr(self, "harmonic_completeness", 0.0) or 0.0),
                ),
                (
                    "harmonic_inharmonic_ratio",
                    float(getattr(self, "harmonic_inharmonic_ratio", 0.0) or 0.0),
                ),
                ("model_weights_source", _mws_src),
                ("model_weights_warning", _mws_warn),
                ("model_weights_fallback_reason", "" if _mws_fb is None else str(_mws_fb)),
                ("energy_conservation_status", _meta_atom(getattr(self, "energy_conservation_status", None))),
                ("energy_conservation_error", _meta_atom(getattr(self, "energy_conservation_error", None))),
                ("energy_denominator_description", _meta_atom(getattr(self, "energy_denominator_description", None))),
                ("validation_export_status", _val_stat),
                ("pca_export_status", "not_available_at_compile_stage"),
                ("f0_prior_note", _meta_atom(getattr(self, "f0_prior_note", None))),
                ("f0_prior_source", _meta_atom(getattr(self, "f0_prior_source", None))),
                ("f0_nominal_hz", _meta_atom(getattr(self, "f0_nominal_hz", None))),
                ("f0_prior_hz", _meta_atom(getattr(self, "f0_prior_hz", None))),
                ("f0_final_source", _meta_atom(getattr(self, "f0_final_source", None))),
                ("f0_used_for_density_hz", _meta_atom(getattr(self, "f0_used_for_density_hz", None))),
                (
                    "f0_used_for_density_source",
                    _meta_atom(getattr(self, "f0_used_for_density_source", None)),
                ),
                (
                    "f0_used_for_harmonic_validation_hz",
                    _meta_atom(getattr(self, "f0_used_for_harmonic_validation_hz", None)),
                ),
                (
                    "f0_used_for_harmonic_validation_source",
                    _meta_atom(getattr(self, "f0_used_for_harmonic_validation_source", None)),
                ),
                ("acoustic_f0_status", _meta_atom(getattr(self, "acoustic_f0_status", None))),
                (
                    "f0_detuning_cents_from_nominal",
                    _meta_atom(getattr(self, "f0_detuning_cents_from_nominal", None)),
                ),
                (
                    "f0_fit_residual_std_hz",
                    _meta_atom(
                        getattr(self, "f0_fit_residual_std_hz", None)
                        or getattr(self, "f0_robust_residual_std", None)
                    ),
                ),
                ("f0_prior_available", _meta_atom(getattr(self, "f0_prior_available", None))),
                ("f0_blind_method", _meta_atom(getattr(self, "f0_blind_method", None))),
                ("f0_final_method", _meta_atom(getattr(self, "f0_final_method", None))),
                ("f0_fit_accepted", _meta_atom(getattr(self, "f0_fit_accepted", None))),
                ("f0_fit_quality", _meta_atom(getattr(self, "f0_fit_quality", None))),
                ("f0_fit_rejection_reason", _meta_atom(getattr(self, "f0_fit_rejection_reason", None))),
                (
                    "harmonic_occupancy_ratio",
                    _meta_atom(getattr(self, "harmonic_occupancy_ratio", None)),
                ),
                (
                    "expected_harmonic_slot_count",
                    _meta_atom(getattr(self, "expected_harmonic_slot_count", None)),
                ),
                (
                    "detected_harmonic_slot_count",
                    _meta_atom(getattr(self, "detected_harmonic_slot_count", None)),
                ),
                (
                    "residual_log_frequency_occupancy",
                    _meta_atom(getattr(self, "residual_log_frequency_occupancy", None)),
                ),
                (
                    "residual_log_frequency_bin_count",
                    _meta_atom(getattr(self, "residual_log_frequency_bin_count", None)),
                ),
                (
                    "residual_log_frequency_bin_total",
                    _meta_atom(getattr(self, "residual_log_frequency_bin_total", None)),
                ),
            ]
            # No legacy batch_* aliases are emitted into Analysis_Metadata in
            # current-analysis mode. The canonical component_* keys above are
            # the single source of truth.
            if bool(getattr(self, "gui_manual_override_active", False)):
                _mh = getattr(self, "gui_manual_model_harmonic_weight", None)
                _mi = getattr(self, "gui_manual_model_inharmonic_weight", None)
                if _mh is not None:
                    analysis_meta_rows.append(("manual_model_harmonic_weight", float(_mh)))
                if _mi is not None:
                    analysis_meta_rows.append(("manual_model_inharmonic_weight", float(_mi)))
            diss_status = "skipped: no dissonance values found"
            try:
                from dissonance_models import list_available_models as _list_dm
            except Exception:
                _list_dm = lambda: []  # type: ignore[misc, assignment]

            if bool(getattr(self, "dissonance_enabled", False)) and isinstance(
                getattr(self, "dissonance_values", None), dict
            ):
                try:
                    from dissonance_export import (
                        CANONICAL_VALUE_BY_SLUG,
                        DISSONANCE_AUDIT_COPY_COLUMNS,
                        MODEL_SLUGS,
                        build_dissonance_model_comparison_long,
                    )

                    drow: Dict[str, Any] = {
                        "Note": note,
                        "selected_dissonance_model": _selected_dm_for_meta
                        or str(getattr(self, "dissonance_model", "") or ""),
                    }
                    _dpc = getattr(self, "dissonance_partial_count", None)
                    if _dpc is not None:
                        drow["dissonance_partial_count"] = int(_dpc)
                    _dpp = getattr(self, "dissonance_pair_count", None)
                    if _dpp is not None:
                        drow["dissonance_pair_count"] = int(_dpp)
                    for _ak in DISSONANCE_AUDIT_COPY_COLUMNS:
                        _v = getattr(self, _ak, None)
                        if _v is None:
                            continue
                        drow[_ak] = _v
                    for slug in MODEL_SLUGS:
                        v = self.dissonance_values.get(slug)
                        key = CANONICAL_VALUE_BY_SLUG[slug]
                        if v is not None and np.isfinite(float(v)):
                            drow[key] = float(v)
                    _dm_key = getattr(self, "dissonance_model", None)
                    sv = self.dissonance_values.get(_dm_key) if _dm_key is not None else None
                    if sv is None and _selected_dm_for_meta:
                        for _cand in (
                            self.dissonance_model,
                            _selected_dm_for_meta.title(),
                            _selected_dm_for_meta.upper(),
                        ):
                            if _cand in self.dissonance_values:
                                sv = self.dissonance_values.get(_cand)
                                break
                    if sv is not None and np.isfinite(float(sv)):
                        drow["selected_dissonance_value"] = float(sv)
                    ddf = pd.DataFrame([drow])
                    hasv = any(
                        CANONICAL_VALUE_BY_SLUG[s] in ddf.columns
                        and pd.notna(ddf[CANONICAL_VALUE_BY_SLUG[s]].iloc[0])
                        for s in MODEL_SLUGS
                    )
                    if hasv:
                        pref = (
                            ["Note"]
                            + [CANONICAL_VALUE_BY_SLUG[s] for s in MODEL_SLUGS if CANONICAL_VALUE_BY_SLUG[s] in ddf.columns]
                            + [
                                c
                                for c in (
                                    "selected_dissonance_model",
                                    "selected_dissonance_value",
                                    "dissonance_partial_count",
                                    "dissonance_pair_count",
                                )
                                if c in ddf.columns
                            ]
                            + [c for c in DISSONANCE_AUDIT_COPY_COLUMNS if c in ddf.columns]
                        )
                        _pub_df(ddf[[c for c in pref if c in ddf.columns]]).to_excel(
                            writer, sheet_name="Dissonance_Metrics", index=False
                        )
                        diss_status = "exported"
                        if bool(getattr(self, "dissonance_compare_models", False)):
                            long_df = build_dissonance_model_comparison_long(ddf)
                            if not long_df.empty:
                                _pub_df(long_df).to_excel(
                                    writer, sheet_name="Dissonance_Model_Comparison", index=False
                                )
                except Exception as _diss_ex:
                    log.warning("Dissonance sheet export failed: %s", _diss_ex)

            analysis_meta_rows.extend(
                [
                    ("dissonance_enabled", bool(getattr(self, "dissonance_enabled", False))),
                    ("dissonance_compare_models", bool(getattr(self, "dissonance_compare_models", False))),
                    ("available_dissonance_models", ",".join(_list_dm())),
                    (
                        "selected_dissonance_model",
                        _selected_dm_for_meta or "not_available_at_compile_stage",
                    ),
                    ("dissonance_export_status", diss_status),
                ]
            )
            if bool(getattr(self, "dissonance_enabled", False)):
                _capv = getattr(self, "dissonance_partial_cap", None)
                analysis_meta_rows.extend(
                    [
                        (
                            "dissonance_partial_cap",
                            _capv if _capv is not None else "not_available_at_compile_stage",
                        ),
                        (
                            "dissonance_partial_count_before_cap",
                            _meta_atom(getattr(self, "dissonance_partial_count_before_cap", None)),
                        ),
                        (
                            "dissonance_partial_count_after_cap",
                            _meta_atom(getattr(self, "dissonance_partial_count_after_cap", None)),
                        ),
                        (
                            "dissonance_pair_count_after_cap",
                            _meta_atom(getattr(self, "dissonance_pair_count_after_cap", None)),
                        ),
                        (
                            "dissonance_cap_computation_note",
                            _meta_atom(getattr(self, "dissonance_cap_computation_note", None)),
                        ),
                    ]
                )
            else:
                analysis_meta_rows.extend(
                    [
                        ("dissonance_partial_cap", "not_available_at_compile_stage"),
                        ("dissonance_partial_count_before_cap", "not_available_at_compile_stage"),
                        ("dissonance_partial_count_after_cap", "not_available_at_compile_stage"),
                        ("dissonance_pair_count_after_cap", "not_available_at_compile_stage"),
                        ("dissonance_cap_computation_note", "not_available_at_compile_stage"),
                    ]
                )
            analysis_meta_rows.extend(
                [
                    (
                        "excel_charting_warning_metrics_sheet",
                        "Do not draw one contiguous LINE chart across multiple Metrics sheet columns: "
                        "each column is a different quantity; ``discrete_metric_d3/d10/d17/d24`` use independent "
                        "scales. For trends across notes: X = Note (or folder), Y = **one** metric. "
                        "To compare D3…D24: use a **column** chart (categories = metric names).",
                    ),
                    (
                        "metrics_sheet_discrete_columns_position",
                        "Export v6+: ``discrete_metric_*`` columns appear at the **end** of the Metrics sheet (not in A–G).",
                    ),
                ]
            )
            try:
                _pie_dir: Optional[Path] = None
                if export_output_dir is not None:
                    _pie_dir = Path(export_output_dir)
                else:
                    _wpath = getattr(writer, "path", None)
                    if _wpath:
                        _pie_dir = Path(str(_wpath)).parent
                if _pie_dir is not None:
                    _pie_dir.mkdir(parents=True, exist_ok=True)
                    self._save_component_balance_pies(_pie_dir, note)
                else:
                    log.warning(
                        "Component balance pies skipped: could not resolve output directory "
                        "(export_output_dir is None and writer.path is %r).",
                        getattr(writer, "path", None),
                    )
            except Exception as _pie_exc:
                log.warning("Component balance pies (pre-Analysis_Metadata): %s", _pie_exc)
            analysis_meta_rows.extend(
                [
                    (
                        "amplitude_mass_chart_file",
                        str(getattr(self, "amplitude_mass_chart_file", "") or ""),
                    ),
                    (
                        "amplitude_mass_chart_basis",
                        str(getattr(self, "amplitude_mass_chart_basis", "") or ""),
                    ),
                    (
                        "amplitude_mass_chart_interpretation",
                        str(getattr(self, "amplitude_mass_chart_interpretation", "") or ""),
                    ),
                    (
                        "energy_ratio_chart_file",
                        str(getattr(self, "energy_ratio_chart_file", "") or ""),
                    ),
                    (
                        "energy_ratio_chart_basis",
                        str(getattr(self, "energy_ratio_chart_basis", "") or ""),
                    ),
                    (
                        "energy_ratio_chart_interpretation",
                        str(getattr(self, "energy_ratio_chart_interpretation", "") or ""),
                    ),
                    (
                        "amplitude_mass_chart_status",
                        str(getattr(self, "amplitude_mass_chart_status", "") or ""),
                    ),
                    (
                        "energy_ratio_chart_status",
                        str(getattr(self, "energy_ratio_chart_status", "") or ""),
                    ),
                    (
                        "component_energy_pie_file",
                        str(getattr(self, "component_energy_pie_file", "") or ""),
                    ),
                    (
                        "component_energy_pie_basis",
                        str(getattr(self, "component_energy_pie_alias_basis", "") or ""),
                    ),
                ]
            )
            try:
                from metadata_sanitizer import filter_analysis_meta_rows_publication_clean as _filt_am  # noqa: PLC0415
            except Exception:  # pragma: no cover

                def _filt_am(rows):  # type: ignore[misc]
                    return rows

            analysis_meta_rows_for_disk = _filt_am(analysis_meta_rows)
            meta_df = _pub_df(pd.DataFrame(analysis_meta_rows_for_disk, columns=["Parameter", "Value"]))

            # AUDIT FIX (stale-pipeline guard) — final pre-save schema
            # check. If anything is off, raise RuntimeError so the
            # outer try/except can clean up the partial workbook
            # instead of silently shipping a half-written legacy
            # export. The validator never lies-silently; it raises.
            self._validate_per_note_export_schema(
                harm_df=harm_export_for_validation,
                ih_df=ih_export,
                sb_df=sb_export,
                meta_rows=analysis_meta_rows,
                note=note,
            )

            meta_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)

            # ===== 4. PARÂMETROS =====
            # FIXED: Ensure hop_length in parameters matches the actual value used
            hop_length_param = getattr(self, "hop_length", None)
            if hop_length_param is None:
                hop_length_param = int(getattr(self, "n_fft", 4096)) // 2  # Fallback
            else:
                hop_length_param = int(hop_length_param)
            
            params_data = {
                "Parameter": [
                    "Note", "Sample Rate (Hz)", "FFT Size", "Hop Length", "Window Type", "Weight Function",
                    "Frequency Range (Hz)", "Magnitude Range (dB)", "Tolerance (Hz)", "Adaptive Tolerance", "Analysis Method",
                    "Tier",  # NEW: Add tier to parameters
                ],
                "Value": [
                    note,
                    int(getattr(self, "sr", 0) or 0),
                    int(getattr(self, "n_fft", 0) or 0),
                    hop_length_param,  # FIXED: Use calculated value
                    str(getattr(self, "window", "")),
                    str(getattr(self, "weight_function", "linear")),
                    f"{float(getattr(self, 'freq_min', 20.0) or 20.0)} - {float(getattr(self, 'freq_max', 20000.0) or 20000.0)}",
                    f"{float(getattr(self, 'db_min', -90.0) or -90.0)} - {float(getattr(self, 'db_max', 0.0) or 0.0)}",
                    float(getattr(self, "tolerance", 10.0) or 10.0),
                    bool(getattr(self, "use_adaptive_tolerance", True)),
                    "STFT",
                    str(getattr(self, "tier", "")) if getattr(self, "tier", None) is not None else "",  # NEW: Add tier value
                ],
            }
            _pub_df(pd.DataFrame(params_data)).to_excel(writer, sheet_name="Analysis Parameters", index=False)

            # Processing Metadata Sheet
            # Mathematical validation (standard reference):
            # Bin spacing: Δf = SR / (N_FFT × ZP)
            # N_FFT efetivo: N_FFT_effective = N_FFT × ZP
            # Frame duration: T = N_FFT / SR
            # Nyquist frequency: f_nyquist = SR / 2
            
            processing_metadata = {
                'Parameter': [
                    'Sample Rate (Hz)', 'N_FFT', 'Zero Padding', 'N_FFT Effective',
                    'Bin Spacing (Hz)', 'Bin Spacing (bins)', 'Frame Duration (s)',
                    'Hop Length', 'Window Type', 'Window Length',
                    'Nyquist Frequency (Hz)', 'Tier',
                    'Sub-Bin Interpolation', 'Global f₀ Estimation', 'Local Peak Validation'
                ],
                'Value': []
            }
            
            # Calculate values
            sr_val = int(getattr(self, 'sr', 0) or 0)
            n_fft_val = int(getattr(self, 'n_fft', 0) or 0)
            zp_val = int(getattr(self, 'zero_padding', 1) or 1)
            n_fft_effective = n_fft_val * zp_val
            bin_spacing_val = _calculate_bin_spacing(sr_val, n_fft_val, zp_val) if sr_val > 0 else 0.0
            frame_duration_val = n_fft_val / sr_val if sr_val > 0 else 0.0
            nyquist_freq_val = sr_val / 2.0 if sr_val > 0 else 0.0
            hop_length_val = int(getattr(self, 'hop_length', 0) or 0)
            window_val = str(getattr(self, 'window', ''))
            window_length_val = n_fft_val
            tier_val = str(getattr(self, 'tier', '')) if getattr(self, 'tier', None) is not None else ''
            
            # Harmonic detection metadata fields
            has_sub_bin = hasattr(self, 'f0_robust') or (sr_val > 0 and n_fft_val > 0 and zp_val > 0)
            has_global_f0 = bool(getattr(self, "f0_fit_accepted", False))
            has_peak_validation = has_sub_bin  # Peak validation requires complete spectrum
            
            processing_metadata['Value'] = [
                sr_val,
                n_fft_val,
                zp_val,
                n_fft_effective,
                bin_spacing_val,
                1.0,  # Bin spacing in bins (always 1.0 by definition)
                frame_duration_val,
                hop_length_val,
                window_val,
                window_length_val,
                nyquist_freq_val,
                tier_val,
                'Enabled' if has_sub_bin else 'Disabled',
                'Enabled' if has_global_f0 else 'Disabled',
                'Enabled' if has_peak_validation else 'Disabled'
            ]
            
            # Add robust f₀ estimates if available
            if hasattr(self, 'f0_robust_fit_quality') or hasattr(self, 'f0_robust_accepted'):
                processing_metadata['Parameter'].extend([
                    'f₀ Robust (Hz)', 'f₀ Robust Residual Std (Hz)', 'f₀ Robust Fit Quality', 'f₀ Robust Accepted'
                ])
                processing_metadata['Value'].extend([
                    float(getattr(self, 'f0_robust', 0.0)),
                    float(getattr(self, 'f0_robust_residual_std', 0.0)),
                    float(getattr(self, 'f0_robust_fit_quality', 0.0)),
                    bool(getattr(self, "f0_fit_accepted", False))
                ])
            
            _pub_df(pd.DataFrame(processing_metadata)).to_excel(writer, sheet_name="Processing Metadata", index=False)
            log.debug("Processing Metadata sheet saved")

        except Exception as e:
            log.error(f"Error in _save_spectral_data_to_excel: {e}", exc_info=True)
            try:
                pd.DataFrame([{"note": note, "error": str(e)}]).to_excel(writer, sheet_name="Error", index=False)
            except Exception:
                pass


def _process_file_worker_parallel(
    task: Tuple[Tuple[np.ndarray, int, str, str], Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Ponto de entrada picklável para ``multiprocessing.Pool`` (restaurado a partir de main_7).
    Espera ``((y, sr, note, file_path), worker_params)``.
    """
    file_tuple, params = task
    y, sr, note, file_path = file_tuple
    y = np.asarray(y, dtype=np.float64)
    _dcb = float(np.mean(y))
    y = y - _dcb
    _dca = float(np.mean(y))
    ap = AudioProcessor()
    ap.dc_offset_before_removal = _dcb
    ap.dc_offset_after_removal = _dca
    ap.dc_removal_applied = True
    ap.audio_data = [(y, sr, note, str(file_path))]
    rd = Path(params["results_directory"])
    hop = params.get("hop_length")
    if hop is not None:
        hop = int(hop)
    try:
        ap.apply_filters_and_generate_data(
            freq_min=float(params["freq_min"]),
            freq_max=float(params["freq_max"]),
            db_min=float(params["db_min"]),
            db_max=float(params["db_max"]),
            tolerance=float(params["tolerance"]),
            use_adaptive_tolerance=bool(params["use_adaptive_tolerance"]),
            results_directory=rd,
            n_fft=int(params["n_fft"]),
            hop_length=hop,
            window=str(params["window"]),
            weight_function=str(params["weight_function"]),
            harmonic_weight=float(params["harmonic_weight"]),
            inharmonic_weight=float(params["inharmonic_weight"]),
            dissonance_enabled=bool(params["dissonance_enabled"]),
            dissonance_model=str(params["dissonance_model"]),
            dissonance_curve=bool(params["dissonance_curve_enabled"]),
            dissonance_scale=bool(params["dissonance_scale_enabled"]),
            compare_models=bool(params["dissonance_compare_models"]),
            zero_padding=int(params["zero_padding"]),
            time_avg=str(params["time_avg"]),
            spectral_masking_enabled=bool(params["spectral_masking_enabled"]),
            parallel_processing=False,
            export_data_format=str(params["export_data_format"]),
            progress_callback=None,
            use_tsne=bool(params.get("use_tsne", False)),
            use_umap=bool(params.get("use_umap", False)),
            detect_anomalies=bool(params.get("detect_anomalies", False)),
            anomaly_contamination=params.get("anomaly_contamination"),
            tier=params.get("tier"),
            spectral_magnitude_smoothing_enabled=bool(
                params.get("spectral_magnitude_smoothing_enabled", False)
            ),
        )
        return {"success": True, "note": note, "cached": False, "error": None}
    except Exception as e:
        return {"success": False, "note": note, "cached": False, "error": str(e)}
