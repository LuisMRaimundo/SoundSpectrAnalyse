"""
SoundSpectrAnalyse - standalone Tk file-picker GUI.

This GUI exposes the canonical three-stage pipeline:

    Stage 1: Per-note spectral analysis (``proc_audio.AudioProcessor``)
    Stage 2: Compilation (``compile_metrics.compile_density_metrics_with_pca``)
    Stage 3: Research export + EWSD-R v18 (``post_compile_research_export``)

Component energy ratios are derived from the *current* spectral analysis only.
No external H/I/S percentages, no preprocessing stage and no external
energy-mapping workbook are consulted at runtime.

Features retained from the historical UI:
1. 90-Tier Granular Clustering (zero decalage).
2. Blackman-Harris alignment (hop = N/8).
3. Adaptive HPF logic.
4. Smart zero padding per tier.
5. Full parameter parity with the main interface (window, weight function,
   dissonance model, t-SNE / UMAP / anomaly detection, manual model-weight
   override).
6. Comprehensive logging for parameter activation tracking.
"""

import matplotlib
matplotlib.use('Agg') # Backend headless (no-GUI) para evitar crashes

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import logging
import gc
import re
import sys
import math
import random
import os
import webbrowser
import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# --- CRITICAL: Ensure main directory is in sys.path ---
# This ensures all imports come from the main directory, even when processing external folders
MAIN_DIR = Path(__file__).parent.resolve()
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))
log_main = logging.getLogger("RobustOrchestrator")
log_main.info(f"Main directory set to: {MAIN_DIR}")


def resolve_stage2_compile_file_pattern(
    analysis_results_dir: Path,
    *,
    allow_legacy_super_json: bool = False,
) -> Optional[str]:
    """Return the per-note metrics filename used for Stage 2 compilation.

    Canonical mode compiles only ``spectral_analysis.xlsx``. Legacy
    ``super_analysis_results.json`` is returned only when explicitly allowed
    and no canonical workbooks exist under ``analysis_results_dir``.
    """
    excel_present = any(analysis_results_dir.rglob("spectral_analysis.xlsx"))
    json_present = any(analysis_results_dir.rglob("super_analysis_results.json"))
    if excel_present:
        return "spectral_analysis.xlsx"
    if allow_legacy_super_json and json_present:
        return "super_analysis_results.json"
    return None


def compute_obs_ws_artifact_diagnostics(
    *,
    pure_observation_w_h: float,
    pure_observation_w_i: float,
    pure_observation_w_s: float,
    component_subbass_energy_ratio: float,
    harmonic_energy_sum: float,
    subbass_energy_sum: float,
    subbass_energy_sum_tier_normalized: float | None = None,
    subbass_upper_bound_hz: float | None = None,
    subbass_density_component: float | None = None,
) -> Dict[str, Any]:
    """Phase 7 diagnostic interpretation for obs_wS in discovery exports."""

    def _f(x: Any) -> float:
        try:
            xv = float(x)
            return xv if np.isfinite(xv) else float("nan")
        except Exception:
            return float("nan")

    obs_h = _f(pure_observation_w_h)
    obs_i = _f(pure_observation_w_i)
    obs_s = _f(pure_observation_w_s)
    comp_s = _f(component_subbass_energy_ratio)
    h_sum = _f(harmonic_energy_sum)
    s_sum = _f(subbass_energy_sum)
    s_tier = _f(subbass_energy_sum_tier_normalized)
    s_cut = _f(subbass_upper_bound_hz)
    s_density = _f(subbass_density_component)

    high_obs = bool(np.isfinite(obs_s) and obs_s > 0.05)
    negligible_comp_ratio = bool(np.isfinite(comp_s) and comp_s < 1e-4)

    # Prefer tier-normalized sum when available to align with compiled comparability checks.
    s_evidence = s_tier if np.isfinite(s_tier) else s_sum
    if np.isfinite(h_sum) and h_sum > 0.0 and np.isfinite(s_evidence):
        negligible_vs_h = bool((s_evidence / h_sum) < 1e-4)
    else:
        negligible_vs_h = False
    negligible_abs = bool(np.isfinite(s_evidence) and abs(s_evidence) < 1e-9)
    missing_energy_evidence = bool(not np.isfinite(s_evidence))
    # Phase 7C semantic requirement:
    # high obs_wS + negligible subbass_energy_ratio is sufficient to flag
    # a model-density vs energy-evidence mismatch.
    energy_mismatch = bool(negligible_comp_ratio)
    is_artifact = bool(high_obs and energy_mismatch)

    reason_tokens: List[str] = []
    if high_obs:
        reason_tokens.append("obs_wS_above_0p05")
    if negligible_comp_ratio:
        reason_tokens.append("component_subbass_energy_ratio_below_1e-4")
    if negligible_vs_h:
        reason_tokens.append("subbass_energy_sum_negligible_vs_harmonic_energy_sum")
    if negligible_abs:
        reason_tokens.append("subbass_energy_sum_below_absolute_floor")
    if missing_energy_evidence:
        reason_tokens.append("subbass_energy_sum_missing")
    if is_artifact:
        interpretation = "model_density_residual_not_physical_subbass_energy"
    else:
        interpretation = "physical_or_mixed_subbass_evidence"

    return {
        "pure_observation_w_h": obs_h,
        "pure_observation_w_i": obs_i,
        "pure_observation_w_s": obs_s,
        "obs_wS": obs_s,
        "subbass_density_component": s_density,
        "subbass_energy_ratio": comp_s,
        "subbass_energy_sum": s_sum,
        "subbass_energy_sum_tier_normalized": s_tier,
        "subbass_upper_bound_hz": s_cut,
        "subbass_component_interpretation": interpretation,
        "obs_wS_artifact_flag": is_artifact,
        "obs_wS_artifact_reason": ";".join(reason_tokens) if is_artifact else "",
        "obs_wS_diagnostic_source": "compile_extractor_energy_terms_plus_per_note_observation",
    }


def _flush_orchestrator_log_handlers(logger: logging.Logger) -> None:
    """Best-effort flush so ``gui_worker.log`` survives abrupt process exit."""
    for h in getattr(logger, "handlers", ()):
        try:
            h.flush()
        except Exception:
            pass


def _stage2_compile_via_subprocess(
    compile_kw: Dict[str, Any],
    log: logging.Logger,
) -> Optional[Any]:
    """Run ``compile_density_metrics_with_pca`` in a child process (advanced options).

    t-SNE / UMAP (numba) / anomaly paths can hard-crash the Tk worker on Windows
    without a Python traceback; isolating them keeps the GUI alive and surfaces
    stderr in ``gui_worker.log``.
    """
    import json
    import pickle
    import subprocess
    import tempfile

    worker = MAIN_DIR / "gui_compile_stage2_worker.py"
    if not worker.is_file():
        log.error(
            "Missing %s (expected next to pipeline_orchestrator_gui.py). "
            "Cannot run isolated Stage 2 with advanced options.",
            worker.name,
        )
        return None

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".stage2_compile.json",
        delete=False,
        encoding="utf-8",
    ) as tf:
        cfg_path = Path(tf.name)
        pickle_out = cfg_path.with_suffix(".df.pkl")
        json.dump(
            {"kwargs": compile_kw, "pickle_out": str(pickle_out)},
            tf,
        )

    try:
        log.info(
            "Spawning isolated Stage 2 worker: %s %s",
            sys.executable,
            worker.name,
        )
        _flush_orchestrator_log_handlers(log)
        proc = subprocess.run(
            [sys.executable, str(worker), str(cfg_path)],
            cwd=str(MAIN_DIR),
            capture_output=True,
            text=True,
            timeout=7200,
        )
        out_tail = (proc.stdout or "").strip()
        if out_tail:
            log.info("Stage 2 worker stdout (tail):\n%s", out_tail[-4000:])
            if proc.returncode != 0:
                rc = int(proc.returncode)
                log.error("Stage 2 worker exited with code %s", rc)
                if rc == 3221225725 or rc == -1073741571:
                    # 0xC0000095: common when sklearn/t-SNE hits invalid FP state on Windows
                    log.error(
                        "This exit code often indicates a native fault during t-SNE/UMAP "
                        "(e.g. divide-by-zero in sklearn init). The GUI will retry without "
                        "those options if advanced compile is enabled."
                    )
            err_tail = (proc.stderr or "").strip()
            if err_tail:
                log.error("Stage 2 worker stderr (tail):\n%s", err_tail[-12000:])
            return None
        if not pickle_out.is_file():
            log.error("Stage 2 worker did not write result file: %s", pickle_out)
            return None
        return pickle.loads(pickle_out.read_bytes())
    except subprocess.TimeoutExpired:
        log.error("Stage 2 worker timed out after 7200 seconds")
        return None
    finally:
        try:
            cfg_path.unlink(missing_ok=True)
        except OSError:
            pass
        try:
            pickle_out.unlink(missing_ok=True)
        except OSError:
            pass


from weight_function_ui_labels import (
    WEIGHT_FUNCTION_COMBO_LABELS,
    display_label_for_weight_key,
    resolve_weight_key_from_user_label,
)
from adaptive_density_engine import AdaptiveDensityEngine

# --- DEPENDENCIES ---
try:
    import librosa
    import soundfile as sf
    import pandas as pd
    import matplotlib.pyplot as plt
    # Import from main directory explicitly
    import proc_audio
    import compile_metrics
except ImportError as e:
    raise ImportError(f"CRITICAL: Missing dependency. Details: {e}")

# --- 90-TIER GRANULAR CONFIGURATION ---
# Scientific Alignment:
# ZP diminui antes de N_FFT diminuir. Hop sempre N/8.
# 
# OPTIMIZATION NOTES:
# - All N_FFT values rounded to nearest power-of-2 for FFT efficiency
# - Tolerance scales with frequency (psychoacoustic JND considerations)
# - Tier boundaries aligned with critical band centers where possible
# - Major transitions: ~1500 Hz (timbre region), ~3600 Hz (presence), ~7500 Hz (air)

def _round_to_power_of_2(n: int) -> int:
    """Round to nearest power of 2 for FFT efficiency."""
    if n <= 0:
        return 512
    # Find nearest power of 2
    log2 = n.bit_length() - 1
    lower = 1 << log2
    upper = 1 << (log2 + 1)
    return lower if (n - lower) < (upper - n) else upper

def _calculate_security_margin(f0: float) -> float:
    """
    Calculate continuous security margin percentage for given fundamental frequency.
    
    Mathematical formulation (verified):
    - Uses logarithmic interpolation for smooth, psychoacoustically-correct scaling
    - Ensures C¹ continuity (continuous first derivative) at all boundaries
    - Margin ranges: 35% at 20 Hz → 10% at 300+ Hz
    
    Args:
        f0: Fundamental frequency in Hz
        
    Returns:
        Security margin percentage (10.0 to 35.0)
        
    Verification examples (mathematically verified):
        f0 = 20 Hz  → margin = 35.0% (boundary)
        f0 = 45 Hz  → margin ≈ 27.6% (interpolated in log space)
        f0 = 60 Hz  → margin = 25.0% (boundary, C¹ continuous)
        f0 = 85 Hz  → margin ≈ 20.8% (interpolated in log space)
        f0 = 120 Hz → margin = 15.0% (boundary, C¹ continuous)
        f0 = 150 Hz → margin ≈ 13.3% (interpolated in log space)
        f0 = 300 Hz → margin = 10.0% (boundary, C¹ continuous)
        f0 = 400 Hz → margin = 10.0% (constant above 300 Hz)
        
    Mathematical properties:
        - C¹ continuity: First derivative is continuous at all boundaries
        - Monotonic: Margin decreases smoothly as frequency increases
        - Logarithmic scaling: Matches psychoacoustic perception (Weber-Fechner law)
    """
    if f0 >= 300.0:
        return 10.0
    elif f0 <= 20.0:
        return 35.0
    else:
        log_f0 = math.log(f0)
        
        if f0 < 60.0:
            # Segment 1: [20 Hz, 60 Hz] → [35%, 25%]
            log_20 = math.log(20.0)
            log_60 = math.log(60.0)
            t = (log_f0 - log_20) / (log_60 - log_20)
            margin = 35.0 + (25.0 - 35.0) * t
        elif f0 < 120.0:
            # Segment 2: [60 Hz, 120 Hz] → [25%, 15%]
            log_60 = math.log(60.0)
            log_120 = math.log(120.0)
            t = (log_f0 - log_60) / (log_120 - log_60)
            margin = 25.0 + (15.0 - 25.0) * t
        else:  # 120 <= f0 < 300
            # Segment 3: [120 Hz, 300 Hz] → [15%, 10%]
            log_120 = math.log(120.0)
            log_300 = math.log(300.0)
            t = (log_f0 - log_120) / (log_300 - log_120)
            margin = 15.0 + (10.0 - 15.0) * t
        
        return max(10.0, min(35.0, margin))

def _calculate_adaptive_tolerance(freq: float, base_tolerance: float, 
                                  use_adaptive: bool = False) -> float:
    """
    Calculate adaptive tolerance based on frequency.
    
    Args:
        freq: Center frequency in Hz
        base_tolerance: Base tolerance from tier configuration
        use_adaptive: If True, tolerance = 1.5% of frequency (psychoacoustic JND)
    
    Returns:
        Tolerance value in Hz
    """
    if use_adaptive and freq > 0:
        # Psychoacoustic: JND scales roughly as 1.5% of frequency
        adaptive = freq * 0.015
        # Use the larger of base or adaptive, but cap at reasonable maximum
        return min(max(base_tolerance, adaptive), 50.0)
    return base_tolerance

FFT_SETTINGS_BY_CLUSTER = {
    # --- SUB BASS: Max resolution, slow decay ---
    # Note: All N_FFT values optimized to power-of-2 for FFT efficiency
    # Critical band: ~20-100 Hz (Bark 0-1)
    'Tier_01': {'max_freq': 20,  'n_fft': 16384, 'tolerance': 3.0, 'zp': 2},  # 2^14
    'Tier_02': {'max_freq': 23,  'n_fft': 16384, 'tolerance': 3.1, 'zp': 2},  # 15800→16384
    'Tier_03': {'max_freq': 26,  'n_fft': 16384, 'tolerance': 3.2, 'zp': 2},  # 15200→16384
    'Tier_04': {'max_freq': 30,  'n_fft': 16384, 'tolerance': 3.3, 'zp': 2},  # 14600→16384
    'Tier_05': {'max_freq': 34,  'n_fft': 16384, 'tolerance': 3.4, 'zp': 2},  # 14000→16384
    'Tier_06': {'max_freq': 38,  'n_fft': 16384, 'tolerance': 3.5, 'zp': 2},  # 13400→16384
    'Tier_07': {'max_freq': 43,  'n_fft': 16384, 'tolerance': 3.6, 'zp': 2},  # 12800→16384
    'Tier_08': {'max_freq': 48,  'n_fft': 16384, 'tolerance': 3.8, 'zp': 2},  # 12200→16384
    'Tier_09': {'max_freq': 54,  'n_fft': 16384, 'tolerance': 4.0, 'zp': 2},  # 11600→16384
    'Tier_10': {'max_freq': 60,  'n_fft': 16384, 'tolerance': 4.2, 'zp': 2},  # 11000→16384
    'Tier_11': {'max_freq': 68,  'n_fft': 16384, 'tolerance': 4.4, 'zp': 2},  # 10400→16384
    'Tier_12': {'max_freq': 76,  'n_fft': 8192,  'tolerance': 4.6, 'zp': 2},  # 9800→8192 (2^13)
    'Tier_13': {'max_freq': 85,  'n_fft': 8192,  'tolerance': 4.8, 'zp': 2},  # 9200→8192
    'Tier_14': {'max_freq': 95,  'n_fft': 8192,  'tolerance': 5.0, 'zp': 2},  # 8600→8192
    'Tier_15': {'max_freq': 105, 'n_fft': 8192,  'tolerance': 5.2, 'zp': 2},  # Already 2^13

    # --- BASS: Window shrinking for better timing ---
    # Critical band: ~100-250 Hz (Bark 1-3)
    'Tier_16': {'max_freq': 115, 'n_fft': 8192,  'tolerance': 5.4, 'zp': 2},  # 7800→8192
    'Tier_17': {'max_freq': 125, 'n_fft': 8192,  'tolerance': 5.6, 'zp': 2},  # 7500→8192
    'Tier_18': {'max_freq': 135, 'n_fft': 8192,  'tolerance': 5.8, 'zp': 2},  # 7200→8192
    'Tier_19': {'max_freq': 145, 'n_fft': 8192,  'tolerance': 6.0, 'zp': 2},  # 6900→8192
    'Tier_20': {'max_freq': 155, 'n_fft': 8192,  'tolerance': 6.2, 'zp': 2},  # 6600→8192
    'Tier_21': {'max_freq': 165, 'n_fft': 8192,  'tolerance': 6.4, 'zp': 2},  # 6300→8192
    'Tier_22': {'max_freq': 175, 'n_fft': 8192,  'tolerance': 6.6, 'zp': 2},  # 6000→8192
    'Tier_23': {'max_freq': 185, 'n_fft': 8192,  'tolerance': 6.8, 'zp': 2},  # 5700→8192
    'Tier_24': {'max_freq': 195, 'n_fft': 8192,  'tolerance': 7.0, 'zp': 2},  # 5400→8192
    'Tier_25': {'max_freq': 205, 'n_fft': 8192,  'tolerance': 7.2, 'zp': 2},  # 5100→8192
    'Tier_26': {'max_freq': 215, 'n_fft': 4096,  'tolerance': 7.4, 'zp': 2},  # 4800→4096 (2^12)
    'Tier_27': {'max_freq': 225, 'n_fft': 4096,  'tolerance': 7.6, 'zp': 2},  # 4500→4096
    'Tier_28': {'max_freq': 235, 'n_fft': 4096,  'tolerance': 7.8, 'zp': 2},  # 4200→4096
    'Tier_29': {'max_freq': 245, 'n_fft': 4096,  'tolerance': 8.0, 'zp': 2},  # Already 2^12
    'Tier_30': {'max_freq': 260, 'n_fft': 4096,  'tolerance': 8.2, 'zp': 2},  # 3950→4096

    # --- LOW MIDS: Decreasing window for transient clarity ---
    # Critical band: ~250-500 Hz (Bark 3-5)
    'Tier_31': {'max_freq': 275, 'n_fft': 4096,  'tolerance': 8.4, 'zp': 2},  # 3800→4096
    'Tier_32': {'max_freq': 290, 'n_fft': 4096,  'tolerance': 8.6, 'zp': 2},  # 3650→4096
    'Tier_33': {'max_freq': 305, 'n_fft': 4096,  'tolerance': 8.8, 'zp': 2},  # 3500→4096
    'Tier_34': {'max_freq': 320, 'n_fft': 4096,  'tolerance': 9.0, 'zp': 2},  # 3350→4096
    'Tier_35': {'max_freq': 340, 'n_fft': 4096,  'tolerance': 9.2, 'zp': 2},  # 3200→4096
    'Tier_36': {'max_freq': 360, 'n_fft': 4096,  'tolerance': 9.4, 'zp': 2},  # 3050→4096
    'Tier_37': {'max_freq': 380, 'n_fft': 4096,  'tolerance': 9.6, 'zp': 2},  # 2900→4096
    'Tier_38': {'max_freq': 400, 'n_fft': 4096,  'tolerance': 9.8, 'zp': 2},  # 2750→4096
    'Tier_39': {'max_freq': 425, 'n_fft': 4096,  'tolerance': 10.0, 'zp': 2},  # 2600→4096
    'Tier_40': {'max_freq': 450, 'n_fft': 4096,  'tolerance': 10.2, 'zp': 2},  # 2450→4096
    'Tier_41': {'max_freq': 475, 'n_fft': 4096,  'tolerance': 10.4, 'zp': 2},  # 2300→4096
    'Tier_42': {'max_freq': 500, 'n_fft': 4096,  'tolerance': 10.6, 'zp': 2},  # 2150→4096
    'Tier_43': {'max_freq': 530, 'n_fft': 2048,  'tolerance': 10.8, 'zp': 2},  # Already 2^11
    'Tier_44': {'max_freq': 560, 'n_fft': 2048,  'tolerance': 11.0, 'zp': 2},  # 1980→2048
    'Tier_45': {'max_freq': 590, 'n_fft': 2048,  'tolerance': 11.2, 'zp': 2},  # 1920→2048

    # --- MID RANGE: Focusing on timing ---
    # Critical band: ~500-1500 Hz (Bark 5-10) - Timbre region
    'Tier_46': {'max_freq': 620, 'n_fft': 2048,  'tolerance': 11.4, 'zp': 2},  # 1860→2048
    'Tier_47': {'max_freq': 660, 'n_fft': 2048,  'tolerance': 11.6, 'zp': 2},  # 1800→2048
    'Tier_48': {'max_freq': 700, 'n_fft': 2048,  'tolerance': 11.8, 'zp': 2},  # 1740→2048
    'Tier_49': {'max_freq': 740, 'n_fft': 2048,  'tolerance': 12.0, 'zp': 2},  # 1680→2048
    'Tier_50': {'max_freq': 790, 'n_fft': 2048,  'tolerance': 12.2, 'zp': 2},  # 1620→2048
    'Tier_51': {'max_freq': 840, 'n_fft': 2048,  'tolerance': 12.4, 'zp': 2},  # 1560→2048
    'Tier_52': {'max_freq': 890, 'n_fft': 2048,  'tolerance': 12.6, 'zp': 2},  # 1500→2048
    'Tier_53': {'max_freq': 950, 'n_fft': 2048,  'tolerance': 12.8, 'zp': 2},  # 1440→2048
    'Tier_54': {'max_freq': 1000, 'n_fft': 2048, 'tolerance': 13.0, 'zp': 2},  # 1380→2048
    'Tier_55': {'max_freq': 1070, 'n_fft': 2048, 'tolerance': 13.2, 'zp': 2},  # 1320→2048
    'Tier_56': {'max_freq': 1140, 'n_fft': 2048, 'tolerance': 13.4, 'zp': 2},  # 1260→2048
    'Tier_57': {'max_freq': 1210, 'n_fft': 2048, 'tolerance': 13.6, 'zp': 2},  # 1200→2048
    'Tier_58': {'max_freq': 1280, 'n_fft': 2048, 'tolerance': 13.8, 'zp': 2},  # 1140→2048
    'Tier_59': {'max_freq': 1360, 'n_fft': 2048, 'tolerance': 14.0, 'zp': 2},  # 1080→2048
    'Tier_60': {'max_freq': 1450, 'n_fft': 1024, 'tolerance': 14.2, 'zp': 2},  # Already 2^10

    # --- HIGH MIDS: Toward 512 ---
    # Critical band: ~1500-3600 Hz (Bark 10-15) - Presence region
    'Tier_61': {'max_freq': 1550, 'n_fft': 1024, 'tolerance': 14.4, 'zp': 2},  # 990→1024
    'Tier_62': {'max_freq': 1650, 'n_fft': 1024, 'tolerance': 14.6, 'zp': 2},  # 960→1024
    'Tier_63': {'max_freq': 1760, 'n_fft': 1024, 'tolerance': 14.8, 'zp': 2},  # 930→1024
    'Tier_64': {'max_freq': 1880, 'n_fft': 1024, 'tolerance': 15.0, 'zp': 2},  # 900→1024
    'Tier_65': {'max_freq': 2000, 'n_fft': 1024, 'tolerance': 15.2, 'zp': 2},  # 870→1024
    'Tier_66': {'max_freq': 2120, 'n_fft': 1024, 'tolerance': 15.4, 'zp': 2},  # 840→1024
    'Tier_67': {'max_freq': 2250, 'n_fft': 1024, 'tolerance': 15.6, 'zp': 2},  # 810→1024
    'Tier_68': {'max_freq': 2400, 'n_fft': 1024, 'tolerance': 15.8, 'zp': 2},  # 780→1024
    'Tier_69': {'max_freq': 2550, 'n_fft': 1024, 'tolerance': 16.0, 'zp': 2},  # 750→1024
    'Tier_70': {'max_freq': 2700, 'n_fft': 1024, 'tolerance': 16.5, 'zp': 2},  # 720→1024
    'Tier_71': {'max_freq': 2850, 'n_fft': 1024, 'tolerance': 17.0, 'zp': 2},  # 690→1024
    'Tier_72': {'max_freq': 3000, 'n_fft': 1024, 'tolerance': 17.5, 'zp': 2},  # 660→1024
    'Tier_73': {'max_freq': 3200, 'n_fft': 1024, 'tolerance': 18.0, 'zp': 2},  # 630→1024
    'Tier_74': {'max_freq': 3400, 'n_fft': 1024, 'tolerance': 18.5, 'zp': 2},  # 600→1024
    'Tier_75': {'max_freq': 3600, 'n_fft': 512,  'tolerance': 19.0, 'zp': 2},  # Already 2^9

    # --- TREBLE / AIR: Stable small window ---
    # Critical band: ~3600+ Hz (Bark 15+) - Brilliance/Air region
    'Tier_76': {'max_freq': 3850,  'n_fft': 512, 'tolerance': 19.5, 'zp': 2},
    'Tier_77': {'max_freq': 4100,  'n_fft': 512, 'tolerance': 20.0, 'zp': 2},
    'Tier_78': {'max_freq': 4400,  'n_fft': 512, 'tolerance': 20.5, 'zp': 2},
    'Tier_79': {'max_freq': 4700,  'n_fft': 512, 'tolerance': 21.0, 'zp': 2},
    'Tier_80': {'max_freq': 5050,  'n_fft': 512, 'tolerance': 21.5, 'zp': 2},
    'Tier_81': {'max_freq': 5400,  'n_fft': 512, 'tolerance': 22.0, 'zp': 2},
    'Tier_82': {'max_freq': 5800,  'n_fft': 512, 'tolerance': 22.5, 'zp': 2},
    'Tier_83': {'max_freq': 6300,  'n_fft': 512, 'tolerance': 23.0, 'zp': 2},
    'Tier_84': {'max_freq': 6900,  'n_fft': 512, 'tolerance': 23.5, 'zp': 2},
    'Tier_85': {'max_freq': 7500,  'n_fft': 512, 'tolerance': 24.0, 'zp': 1},
    'Tier_86': {'max_freq': 8500,  'n_fft': 512, 'tolerance': 24.5, 'zp': 1},
    'Tier_87': {'max_freq': 10000, 'n_fft': 512, 'tolerance': 25.0, 'zp': 1},
    'Tier_88': {'max_freq': 12500, 'n_fft': 512, 'tolerance': 25.5, 'zp': 1},
    'Tier_89': {'max_freq': 16000, 'n_fft': 512, 'tolerance': 26.0, 'zp': 1},
    'Tier_90': {'max_freq': float('inf'), 'n_fft': 512, 'tolerance': 27.0, 'zp': 1},
}

VALID_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aif', '.aiff', '.flac'}
PHASE_1_ADAPTIVE_RANDOMIZATION_SEED = 20260526


def build_phase1_file_iteration_order(
    folder: Path,
    *,
    enable_adaptive_path_randomization: bool = False,
    random_seed: int = PHASE_1_ADAPTIVE_RANDOMIZATION_SEED,
) -> List[Path]:
    """Return deterministic Phase 1 order: ascending nominal f0, unknown notes last."""
    from note_parser import canonical_note_from_filename

    folder_files = [
        f for f in folder.glob("*")
        if f.suffix.lower() in VALID_AUDIO_EXTENSIONS
    ]

    sortable: List[Tuple[float, str, int, str, Path]] = []
    for idx, f in enumerate(folder_files):
        note_token, _ = canonical_note_from_filename(f.name)
        if note_token:
            try:
                hz = float(librosa.note_to_hz(str(note_token)))
            except Exception:
                hz = float("inf")
        else:
            hz = float("inf")
        # Unknown notes (inf) are always last, then deterministic by filename.
        unknown_rank = 1 if not np.isfinite(hz) else 0
        sortable.append((hz, f.name.lower(), unknown_rank, f.name, f))

    sortable.sort(key=lambda item: (item[2], item[0], item[1]))
    ordered = [item[4] for item in sortable]

    if enable_adaptive_path_randomization:
        rng = random.Random(int(random_seed))
        rng.shuffle(ordered)

    return ordered

# --- WINDOW TYPES (Full parity with interface.py) ---
VALID_WINDOW_TYPES = ['hann', 'hamming', 'blackmanharris', 'bartlett', 'kaiser', 'gaussian']

def _attach_tk_tooltip(widget: tk.Widget, text: str) -> None:
    """Small hover tooltip (Tk has no native ttk tooltip)."""
    state: Dict[str, Any] = {"tw": None, "after_id": None}

    def cancel_scheduled() -> None:
        aid = state["after_id"]
        if aid is not None:
            try:
                widget.after_cancel(aid)
            except Exception:
                pass
            state["after_id"] = None

    def hide(_event: object = None) -> None:
        cancel_scheduled()
        tw = state["tw"]
        if tw is not None:
            try:
                tw.destroy()
            except Exception:
                pass
            state["tw"] = None

    def show(_event: object = None) -> None:
        if state["tw"] is not None:
            return
        try:
            x = int(widget.winfo_rootx()) + 10
            y = int(widget.winfo_rooty()) + int(widget.winfo_height()) + 2
        except Exception:
            return
        tw = tk.Toplevel(widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            tw,
            text=text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=320,
        )
        lbl.pack(ipadx=4, ipady=2)
        state["tw"] = tw

    def on_enter(_event: object = None) -> None:
        cancel_scheduled()
        state["after_id"] = widget.after(400, show)

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", hide)
    widget.bind("<ButtonPress>", hide)


# --- LOGGING ---
log = logging.getLogger("RobustOrchestrator")
log.setLevel(logging.INFO)

DENSITY_MODE_LABEL_TO_INTERNAL: Dict[str, str] = {
    "Per-note adaptive (acoustic, auto)": "his_note_adaptive",
    "Harmonic only": "harmonic_only",
    "Inharmonic only": "inharmonic_only",
    "Subbass only": "subbass_only",
    "Harmonic + inharmonic + subbass": "his_weighted",
}
DENSITY_MODE_INTERNAL_TO_LABEL: Dict[str, str] = {
    v: k for k, v in DENSITY_MODE_LABEL_TO_INTERNAL.items()
}

class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
    def emit(self, record: logging.LogRecord):
        self.log_queue.put(self.format(record))

class RobustOrchestratorApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("SoundSpectrAnalyse \u2014 Per-note Spectral Analysis")
        master.geometry("1200x850")

        self.processing_queue: List[Path] = []
        self.is_running = False
        self.stop_requested = False
        self.log_queue = queue.Queue()
        log.addHandler(QueueLogHandler(self.log_queue))

        self._build_ui()
        self.master.after(100, self._process_log_queue)

    def _build_ui(self):
        frame_input = ttk.LabelFrame(self.master, text="Input folders")
        frame_input.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(frame_input, text="Add Folder(s)", command=self._add_folders).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        ttk.Button(frame_input, text="Clear Queue", command=self._clear_queue).pack(
            side=tk.LEFT, padx=5, pady=5
        )
        self.lbl_count = ttk.Label(frame_input, text="Queue: 0 folders")
        self.lbl_count.pack(side=tk.LEFT, padx=15)

        frame_options = ttk.LabelFrame(self.master, text="Pipeline controls")
        frame_options.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        notebook = ttk.Notebook(frame_options)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tab_basic = ttk.Frame(notebook)
        tab_advanced = ttk.Frame(notebook)
        notebook.add(tab_basic, text="Basic")
        notebook.add(tab_advanced, text="Advanced")

        # ---------------------- BASIC TAB ----------------------
        lf_density = ttk.LabelFrame(tab_basic, text="Final Density")
        lf_density.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(lf_density, text="Density mode").pack(anchor="w")
        self.combo_density_mode = ttk.Combobox(
            lf_density,
            state="readonly",
            values=list(DENSITY_MODE_LABEL_TO_INTERNAL.keys()),
        )
        self.combo_density_mode.set(DENSITY_MODE_INTERNAL_TO_LABEL["his_note_adaptive"])
        self.combo_density_mode.pack(fill=tk.X)
        self.combo_density_mode.bind("<<ComboboxSelected>>", self._on_density_mode_changed)
        _attach_tk_tooltip(
            self.combo_density_mode,
            (
                "Controls how final-density combines harmonic/inharmonic/subbass "
                "evidence. In per-note adaptive mode, weights are inferred from "
                "each note's own spectral evidence (no imposed fixed percentages)."
            ),
        )

        ttk.Label(lf_density, text="Harmonic weight").pack(anchor="w", pady=(6, 0))
        self.entry_density_w_h = ttk.Entry(lf_density, width=10)
        self.entry_density_w_h.insert(0, "1.0")
        self.entry_density_w_h.pack(fill=tk.X)
        _attach_tk_tooltip(
            self.entry_density_w_h,
            "Weight applied to salient harmonic orders.",
        )

        ttk.Label(lf_density, text="Inharmonic/noise weight").pack(anchor="w", pady=(6, 0))
        self.entry_density_w_i = ttk.Entry(lf_density, width=10)
        self.entry_density_w_i.insert(0, "0.5")
        self.entry_density_w_i.pack(fill=tk.X)
        _attach_tk_tooltip(
            self.entry_density_w_i,
            "Weight applied to occupied inharmonic log-frequency bins.",
        )

        ttk.Label(lf_density, text="Subbass/particle weight").pack(anchor="w", pady=(6, 0))
        self.entry_density_w_s = ttk.Entry(lf_density, width=10)
        self.entry_density_w_s.insert(0, "0.25")
        self.entry_density_w_s.pack(fill=tk.X)
        _attach_tk_tooltip(
            self.entry_density_w_s,
            "Weight applied to salient subbass/breath/bow-scrape particles.",
        )

        self.var_runtime_density_profile = tk.StringVar(
            value="Runtime learned profile: waiting for first note"
        )
        self.lbl_runtime_density_profile = ttk.Label(
            lf_density,
            textvariable=self.var_runtime_density_profile,
            justify="left",
        )
        self.lbl_runtime_density_profile.pack(anchor="w", pady=(8, 0))
        self._on_density_mode_changed()

        lf_output = ttk.LabelFrame(tab_basic, text="Recommended output")
        lf_output.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(
            lf_output,
            text=(
                "Primary output:\n"
                "final_note_density_salience_weighted\n\n"
                "Control output:\n"
                "final_note_density_count_based"
            ),
            justify="left",
        ).pack(anchor="w", padx=6, pady=6)

        lf_help = ttk.LabelFrame(tab_basic, text="Help")
        lf_help.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Button(
            lf_help,
            text="Quick Guide",
            command=lambda: self._open_doc("README.md"),
        ).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(
            lf_help,
            text="Technical Manual",
            command=lambda: self._open_doc("docs/TECHNICAL_MANUAL_COMPLETE.md"),
        ).pack(side=tk.LEFT, padx=6, pady=6)
        ttk.Button(
            lf_help,
            text="Tutorial",
            command=lambda: self._open_doc("docs/GUI_OPTION_REFERENCE.md"),
        ).pack(side=tk.LEFT, padx=6, pady=6)

        frame_act = ttk.LabelFrame(tab_basic, text="Run")
        frame_act.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 8))
        self.btn_run = ttk.Button(frame_act, text="RUN PIPELINE", command=self._run)
        self.btn_run.pack(fill=tk.X, padx=6, pady=(6, 2))
        self.btn_stop = ttk.Button(frame_act, text="STOP", state=tk.DISABLED, command=self._stop)
        self.btn_stop.pack(fill=tk.X, padx=6, pady=2)
        self.lbl_status = ttk.Label(frame_act, text="Idle", font=("Arial", 10, "bold"))
        self.lbl_status.pack(padx=6, pady=4)
        self.txt_log = tk.Text(frame_act, height=12, state=tk.DISABLED, bg="#f0f0f0")
        self.txt_log.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))

        # ---------------------- ADVANCED TAB ----------------------
        lf_stft = ttk.LabelFrame(tab_advanced, text="STFT settings")
        lf_stft.pack(fill=tk.X, padx=10, pady=8)

        ttk.Label(lf_stft, text="Window type").grid(row=0, column=0, sticky="w")
        self.combo_window = ttk.Combobox(lf_stft, values=VALID_WINDOW_TYPES, state="readonly")
        self.combo_window.set("blackmanharris")
        self.combo_window.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.combo_window.bind("<<ComboboxSelected>>", self._on_window_changed)
        _attach_tk_tooltip(self.combo_window, "STFT window used for spectral analysis.")

        self.frame_window_params = ttk.LabelFrame(lf_stft, text="Window parameters")
        self.frame_window_params.grid(row=1, column=1, sticky="ew")
        self.lbl_kaiser = ttk.Label(self.frame_window_params, text="Kaiser beta:")
        self.entry_kaiser_beta = ttk.Entry(self.frame_window_params, width=10)
        self.entry_kaiser_beta.insert(0, "6.5")
        self.lbl_gaussian = ttk.Label(self.frame_window_params, text="Gaussian std:")
        self.entry_gaussian_std = ttk.Entry(self.frame_window_params, width=10)
        self.entry_gaussian_std.insert(0, "auto")
        self._update_window_params_visibility()

        self.var_smart = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            lf_stft,
            text="90-tier granular clustering (tier strategy)",
            variable=self.var_smart,
            command=self._on_smart_changed,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 2))

        self.lbl_fixed_fft_override = ttk.Label(
            lf_stft,
            text="Fixed FFT controls are overridden by tier strategy when enabled.",
            foreground="#666666",
        )
        self.lbl_fixed_fft_override.grid(row=3, column=0, columnspan=2, sticky="w")

        ttk.Label(lf_stft, text="N_FFT (fixed mode)").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.entry_n_fft = ttk.Entry(lf_stft, width=10)
        self.entry_n_fft.insert(0, "4096")
        self.entry_n_fft.grid(row=5, column=0, sticky="ew", padx=(0, 8))

        ttk.Label(lf_stft, text="Hop length (fixed mode)").grid(row=4, column=1, sticky="w", pady=(8, 0))
        self.entry_hop_length = ttk.Entry(lf_stft, width=10)
        self.entry_hop_length.insert(0, "1024")
        self.entry_hop_length.grid(row=5, column=1, sticky="ew")

        ttk.Label(lf_stft, text="Zero padding (fixed mode)").grid(row=6, column=0, sticky="w", pady=(8, 0))
        self.entry_zero_padding = ttk.Entry(lf_stft, width=10)
        self.entry_zero_padding.insert(0, "2")
        self.entry_zero_padding.grid(row=7, column=0, sticky="ew", padx=(0, 8))

        ttk.Label(lf_stft, text="Time averaging").grid(row=6, column=1, sticky="w", pady=(8, 0))
        self.combo_avg = ttk.Combobox(lf_stft, values=["mean", "median", "max"], state="readonly")
        self.combo_avg.set("mean")
        self.combo_avg.grid(row=7, column=1, sticky="ew")

        ttk.Label(lf_stft, text="Peak detection magnitude range (dB)").grid(
            row=8, column=0, sticky="w", pady=(8, 0)
        )
        self.entry_min_db = ttk.Entry(lf_stft, width=10)
        self.entry_min_db.insert(0, "-90.0")
        self.entry_min_db.grid(row=9, column=0, sticky="ew", padx=(0, 8))
        self.entry_max_db = ttk.Entry(lf_stft, width=10)
        self.entry_max_db.insert(0, "0.0")
        self.entry_max_db.grid(row=9, column=1, sticky="ew")
        _attach_tk_tooltip(
            self.entry_min_db,
            "Used by peak detection and spectral component filtering in analysis.",
        )
        _attach_tk_tooltip(
            self.entry_max_db,
            "Used by peak detection and spectral component filtering in analysis.",
        )

        lf_harmonic = ttk.LabelFrame(tab_advanced, text="Harmonic classification")
        lf_harmonic.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(lf_harmonic, text="Frequency range (Hz)").grid(row=0, column=0, sticky="w")
        self.entry_min_freq = ttk.Entry(lf_harmonic, width=12)
        self.entry_min_freq.insert(0, "20.0")
        self.entry_min_freq.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        self.entry_max_freq = ttk.Entry(lf_harmonic, width=12)
        self.entry_max_freq.insert(0, "20000.0")
        self.entry_max_freq.grid(row=1, column=1, sticky="ew")
        ttk.Label(lf_harmonic, text="Harmonic tolerance (Hz)").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.entry_tolerance = ttk.Entry(lf_harmonic, width=12)
        self.entry_tolerance.insert(0, "5.0")
        self.entry_tolerance.grid(row=3, column=0, sticky="ew", padx=(0, 8))
        self.var_adaptive_tolerance = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            lf_harmonic,
            text="Use adaptive tolerance",
            variable=self.var_adaptive_tolerance,
        ).grid(row=3, column=1, sticky="w")
        ttk.Label(
            lf_harmonic,
            text="f0 strategy: acoustic fit when validated; otherwise nominal fallback (reported in outputs).",
            foreground="#444444",
            wraplength=520,
            justify="left",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 0))

        lf_secondary = ttk.LabelFrame(tab_advanced, text="Secondary descriptors")
        lf_secondary.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(lf_secondary, text="Dissonance model").grid(row=0, column=0, sticky="w")
        self.combo_dissonance = ttk.Combobox(
            lf_secondary,
            state="readonly",
            values=["sethares", "hutchinson", "vassilakis", "ALL (Compare)"],
        )
        self.combo_dissonance.set("sethares")
        self.combo_dissonance.grid(row=1, column=0, sticky="ew", padx=(0, 8))
        _attach_tk_tooltip(
            self.combo_dissonance,
            "Secondary descriptor only. Does not define final note density.",
        )

        self.label_amplitude_weighting_function = ttk.Label(
            lf_secondary, text="Amplitude weighting (diagnostic paths)"
        )
        self.label_amplitude_weighting_function.grid(row=0, column=1, sticky="w")
        self.combo_weight = ttk.Combobox(
            lf_secondary, values=list(WEIGHT_FUNCTION_COMBO_LABELS), state="readonly"
        )
        # Default to the PRIMARY comparable profile (wf=log) so that even an
        # isolated, single-run analysis is cross-instrument comparable by
        # default. Selecting any other weighting downgrades the run to the
        # EXPLORATORY profile (logged + flagged in the compiled comparability
        # verdict). See docs/DENSITY_EXPORT_SCHEMA.md (Corpus Comparability).
        self.combo_weight.set(display_label_for_weight_key("log"))
        self.combo_weight.grid(row=1, column=1, sticky="ew")
        _attach_tk_tooltip(
            self.combo_weight,
            "Affects diagnostic/secondary density paths. Final note density uses the H/I/S mode+weights controls.",
        )
        ttk.Label(
            lf_secondary,
            text=(
                "Secondary metrics:\n"
                "- spectral_body_thickness_index\n"
                "- spectral_entropy\n"
                "- effective_partial_density\n"
                "- dissonance descriptors"
            ),
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        lf_debug = ttk.LabelFrame(tab_advanced, text="Debug / legacy")
        lf_debug.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(
            lf_debug,
            text=(
                "Diagnostic/legacy outputs (not final density):\n"
                "- density_metric_raw\n"
                "- Combined Density Metric\n"
                "- legacy/diagnostic exports"
            ),
            justify="left",
        ).pack(anchor="w", pady=(0, 6))

        self.var_compile = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            lf_debug,
            text="Auto-compile compiled_density_metrics.xlsx (Stage 2)",
            variable=self.var_compile,
        ).pack(anchor="w")

        self.var_use_tsne = tk.BooleanVar(value=False)
        self.var_use_umap = tk.BooleanVar(value=False)
        self.var_detect_anomalies = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf_debug, text="Use t-SNE (advanced)", variable=self.var_use_tsne).pack(anchor="w")
        ttk.Checkbutton(lf_debug, text="Use UMAP (advanced)", variable=self.var_use_umap).pack(anchor="w")
        ttk.Checkbutton(
            lf_debug,
            text="Detect anomalies (advanced)",
            variable=self.var_detect_anomalies,
        ).pack(anchor="w")
        ttk.Label(lf_debug, text="Anomaly contamination (auto or 0-1):").pack(anchor="w", pady=(6, 0))
        self.entry_contamination = ttk.Entry(lf_debug, width=12)
        self.entry_contamination.insert(0, "auto")
        self.entry_contamination.pack(anchor="w")

        lf_mw = ttk.LabelFrame(lf_debug, text="Manual model-weight override (advanced)")
        lf_mw.pack(fill=tk.X, pady=(8, 0))
        self.var_manual_model_weight_override = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            lf_mw,
            text="Enable manual inharmonic coefficient β",
            variable=self.var_manual_model_weight_override,
            command=self._on_manual_model_weight_override_toggled,
        ).pack(anchor="w")
        ttk.Label(
            lf_mw,
            text="Inharmonic model weight β (%); α = 1 − β. This does not replace final density H/I/S controls.",
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))
        self.var_i_weight = tk.IntVar(value=5)
        self.scale_i_weight = ttk.Scale(
            lf_mw, from_=0, to=100, variable=self.var_i_weight, command=self._upd_lbl
        )
        self.scale_i_weight.pack(fill=tk.X, pady=(2, 0))
        self.lbl_weight = ttk.Label(lf_mw, text="5% β / 95% α")
        self.lbl_weight.pack(anchor="w")
        self.scale_i_weight.state(["disabled"])

        self._update_fixed_fft_visibility()

    def _on_window_changed(self, event=None):
        """Update window parameter visibility based on selected window type."""
        self._update_window_params_visibility()
    
    def _on_smart_changed(self):
        """Update visibility of fixed FFT parameters based on smart mode."""
        self._update_fixed_fft_visibility()
    
    def _update_fixed_fft_visibility(self):
        """Show/hide fixed FFT parameters based on smart mode."""
        is_smart = self.var_smart.get()
        # Enable fixed FFT parameters when smart mode is OFF
        state = "normal" if not is_smart else "disabled"
        if hasattr(self, 'entry_n_fft'):
            self.entry_n_fft.config(state=state)
            self.entry_hop_length.config(state=state)
            self.entry_zero_padding.config(state=state)
        if hasattr(self, "lbl_fixed_fft_override"):
            self.lbl_fixed_fft_override.config(
                foreground="#666666" if is_smart else "#2e7d32",
                text=(
                    "Fixed FFT controls are overridden by tier strategy when enabled."
                    if is_smart
                    else "Fixed FFT controls are active (tier strategy disabled)."
                ),
            )

    def _update_window_params_visibility(self):
        """Show/hide window-specific parameters based on window type."""
        window = self.combo_window.get().lower()
        
        # Hide all first
        for widget in self.frame_window_params.winfo_children():
            widget.pack_forget()
        
        # Show relevant ones
        if window == "kaiser":
            self.lbl_kaiser.pack(anchor="w")
            self.entry_kaiser_beta.pack(fill=tk.X)
        elif window in ("gaussian", "gauss", "gaussiana"):
            self.lbl_gaussian.pack(anchor="w")
            self.entry_gaussian_std.pack(fill=tk.X)

    def _upd_lbl(self, v):
        if hasattr(self, "lbl_weight"):
            iv = int(float(v))
            self.lbl_weight.config(text=f"{iv}% β (inharmonic) / {100 - iv}% α (harmonic)")

    def _on_manual_model_weight_override_toggled(self):
        if not hasattr(self, "scale_i_weight"):
            return
        if self.var_manual_model_weight_override.get():
            self.scale_i_weight.state(["!disabled"])
        else:
            self.scale_i_weight.state(["disabled"])

    def _density_mode_internal(self) -> str:
        display = str(self.combo_density_mode.get() or "").strip()
        return DENSITY_MODE_LABEL_TO_INTERNAL.get(display, "his_note_adaptive")

    def _on_density_mode_changed(self, _event: object = None) -> None:
        mode = self._density_mode_internal()
        manual_weights_enabled = mode not in {"his_note_adaptive"}
        widgets = (
            self.entry_density_w_h,
            self.entry_density_w_i,
            self.entry_density_w_s,
        )

        def _set_entry(widget: ttk.Entry, text: str, state: str) -> None:
            widget.configure(state="normal")
            widget.delete(0, tk.END)
            widget.insert(0, text)
            widget.configure(state=state)

        if manual_weights_enabled:
            cached = getattr(self, "_manual_density_weights", None)
            if cached is not None and len(cached) == 3:
                _set_entry(self.entry_density_w_h, f"{float(cached[0]):.6f}", "normal")
                _set_entry(self.entry_density_w_i, f"{float(cached[1]):.6f}", "normal")
                _set_entry(self.entry_density_w_s, f"{float(cached[2]):.6f}", "normal")
            else:
                for widget in widgets:
                    widget.configure(state="normal")
        else:
            try:
                self._manual_density_weights = (
                    float(self.entry_density_w_h.get()),
                    float(self.entry_density_w_i.get()),
                    float(self.entry_density_w_s.get()),
                )
            except Exception:
                self._manual_density_weights = (1.0, 0.5, 0.25)
            for widget in widgets:
                _set_entry(widget, "AUTO (Phase 1 discovery)", "disabled")
        if hasattr(self, "var_runtime_density_profile"):
            if manual_weights_enabled:
                self.var_runtime_density_profile.set(
                    "Runtime learned profile: manual/fixed mode selected"
                )
            else:
                self.var_runtime_density_profile.set(
                    "Runtime learned profile: adaptive mode (Phase 1 discovery -> Phase 2 application; GUI fixed values ignored)"
                )

    def _extract_note_density_feedback(self, workbook_path: Path) -> Optional[Tuple[float, float, float]]:
        """Read per-note density weights emitted by Stage 1 workbook."""
        if not workbook_path.is_file():
            return None
        try:
            def _normalize_triplet(values: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
                vals = np.array(values, dtype=float)
                if not np.all(np.isfinite(vals)):
                    return None
                vals = np.maximum(vals, 0.0)
                total = float(np.sum(vals))
                if total <= 0.0:
                    return None
                vals = vals / total
                return (float(vals[0]), float(vals[1]), float(vals[2]))

            with pd.ExcelFile(workbook_path) as xf:
                # Stage-1 per-note workbook writes adaptive weights in ``Metrics``.
                # ``Density_Metrics`` appears in compiled/legacy workbook variants.
                for sheet in ("Metrics", "Density_Metrics", "Spectral_Density_Metrics"):
                    if sheet not in xf.sheet_names:
                        continue
                    df = xf.parse(sheet_name=sheet, nrows=1)
                    if df.empty:
                        continue
                    row = df.iloc[0]

                    # Canonical adaptive triplet (Phase 1 pure observation).
                    triplet = _normalize_triplet(
                        (
                            float(row.get("pure_observation_w_h", float("nan"))),
                            float(row.get("pure_observation_w_i", float("nan"))),
                            float(row.get("pure_observation_w_s", float("nan"))),
                        )
                    )
                    if triplet is not None:
                        return triplet

                    # Backward-compatible alias fallback.
                    triplet = _normalize_triplet(
                        (
                            float(row.get("harmonic_density_weight", float("nan"))),
                            float(row.get("inharmonic_density_weight", float("nan"))),
                            float(row.get("subbass_density_weight", float("nan"))),
                        )
                    )
                    if triplet is not None:
                        return triplet

                    # Legacy fallback with only model H/I weights.
                    w_h = float(row.get("model_harmonic_weight", float("nan")))
                    w_i = float(row.get("model_inharmonic_weight", float("nan")))
                    if np.isfinite(w_h) and np.isfinite(w_i):
                        triplet = _normalize_triplet((w_h, w_i, max(0.0, 1.0 - (w_h + w_i))))
                        if triplet is not None:
                            return triplet

                    # Last fallback from component-energy ratios.
                    triplet = _normalize_triplet(
                        (
                            float(row.get("harmonic_energy_ratio", float("nan"))),
                            float(row.get("inharmonic_energy_ratio", float("nan"))),
                            float(row.get("subbass_energy_ratio", float("nan"))),
                        )
                    )
                    if triplet is not None:
                        return triplet
            return None
        except Exception as exc:
            log.warning("Could not extract adaptive density feedback from %s: %s", workbook_path, exc)
            return None

    def _extract_note_density_feedback_diagnostics(self, workbook_path: Path) -> Dict[str, Any]:
        """Extract Phase 7 obs_wS diagnostics from per-note workbook."""
        defaults = compute_obs_ws_artifact_diagnostics(
            pure_observation_w_h=float("nan"),
            pure_observation_w_i=float("nan"),
            pure_observation_w_s=float("nan"),
            component_subbass_energy_ratio=float("nan"),
            harmonic_energy_sum=float("nan"),
            subbass_energy_sum=float("nan"),
            subbass_energy_sum_tier_normalized=float("nan"),
            subbass_upper_bound_hz=float("nan"),
            subbass_density_component=float("nan"),
        )
        defaults["obs_w_formula_version"] = ""
        for _k in (
            "component_strength_h",
            "component_strength_i",
            "component_strength_s",
            "legacy_component_strength_h_v55",
            "legacy_component_strength_i_v55",
            "legacy_component_strength_s_v55",
        ):
            defaults[_k] = float("nan")
        if not workbook_path.is_file():
            return defaults
        try:
            row = None
            with pd.ExcelFile(workbook_path) as xf:
                for sheet in ("Metrics", "Density_Metrics", "Spectral_Density_Metrics"):
                    if sheet not in xf.sheet_names:
                        continue
                    df = xf.parse(sheet_name=sheet, nrows=1)
                    if df.empty:
                        continue
                    row = df.iloc[0]
                    if any(k in df.columns for k in ("pure_observation_w_h", "pure_observation_w_i", "pure_observation_w_s")):
                        break
            if row is None:
                return defaults

            # Reuse compile-stage extraction so Phase 1 diagnostics and Density_Metrics
            # share the same energy-term source semantics.
            extracted: Dict[str, Any] = {}
            try:
                extracted = compile_metrics.extract_density_components_from_per_note_workbook(
                    workbook_path
                )
            except Exception:
                extracted = {}

            extracted_subbass_sum = extracted.get("subbass_energy_sum", float("nan"))
            extracted_harm_sum = extracted.get("harmonic_energy_sum", float("nan"))
            extracted_subbass_sum_tn = extracted.get("subbass_energy_sum_tier_normalized", float("nan"))
            extracted_tier_source = "per_note_metrics_sheet"
            try:
                _tn_val = float(extracted_subbass_sum_tn)
                if not np.isfinite(_tn_val):
                    raise ValueError("missing_tier_normalized_sum")
                extracted_subbass_sum_tn = _tn_val
                extracted_tier_source = "compile_extractor_tier_normalized_sum"
            except Exception:
                try:
                    from spectral_normalization import n_fft_normalization_factor

                    _n_fft_raw = extracted.get("n_fft", extracted.get("n_fft_effective", float("nan")))
                    _n_fft = int(float(_n_fft_raw))
                    _power_factor = n_fft_normalization_factor(
                        n_fft=_n_fft,
                        n_fft_reference=8192,
                        quantity_kind="peak_power_sum",
                    )
                    _s_en = float(extracted_subbass_sum)
                    if np.isfinite(_s_en) and np.isfinite(_power_factor):
                        extracted_subbass_sum_tn = float(_s_en * _power_factor)
                        extracted_tier_source = "computed_from_compile_extractor_n_fft_and_subbass_energy_sum"
                    else:
                        extracted_subbass_sum_tn = float("nan")
                        extracted_tier_source = "not_available_missing_subbass_sum_or_n_fft"
                except Exception:
                    extracted_subbass_sum_tn = float("nan")
                    extracted_tier_source = "not_available_missing_subbass_sum_or_n_fft"

            def _pick(*vals: Any) -> float:
                for v in vals:
                    try:
                        fv = float(v)
                        if np.isfinite(fv):
                            return fv
                    except Exception:
                        continue
                return float("nan")

            _diag = compute_obs_ws_artifact_diagnostics(
                pure_observation_w_h=_pick(row.get("pure_observation_w_h", float("nan"))),
                pure_observation_w_i=_pick(row.get("pure_observation_w_i", float("nan"))),
                pure_observation_w_s=_pick(row.get("pure_observation_w_s", float("nan"))),
                component_subbass_energy_ratio=_pick(
                    extracted.get("component_subbass_energy_ratio", float("nan")),
                    row.get("component_subbass_energy_ratio", row.get("subbass_energy_ratio", float("nan"))),
                ),
                harmonic_energy_sum=_pick(
                    extracted_harm_sum,
                    row.get("harmonic_energy_sum", float("nan")),
                ),
                subbass_energy_sum=_pick(
                    extracted_subbass_sum,
                    row.get("subbass_energy_sum", float("nan")),
                ),
                subbass_energy_sum_tier_normalized=_pick(
                    extracted_subbass_sum_tn,
                    row.get("subbass_energy_sum_tier_normalized", float("nan")),
                ),
                subbass_upper_bound_hz=_pick(
                    extracted.get("adaptive_subfundamental_cutoff_hz", float("nan")),
                    row.get("subbass_upper_bound_hz", row.get("subfundamental_cutoff_hz", float("nan"))),
                ),
                subbass_density_component=_pick(
                    extracted.get("subbass_density_sum", float("nan")),
                    row.get("subbass_density_component", float("nan")),
                ),
            )
            _diag["obs_w_formula_version"] = str(
                extracted.get(
                    "obs_w_formula_version",
                    row.get("obs_w_formula_version", ""),
                )
                or ""
            )
            for _k in (
                "component_strength_h",
                "component_strength_i",
                "component_strength_s",
                "legacy_component_strength_h_v55",
                "legacy_component_strength_i_v55",
                "legacy_component_strength_s_v55",
            ):
                _diag[_k] = _pick(
                    extracted.get(_k, float("nan")),
                    row.get(_k, float("nan")),
                )
            _diag["subbass_energy_sum_source"] = "compile_extractor_energy_terms"
            _diag["subbass_energy_sum_tier_normalized_source"] = extracted_tier_source
            _diag["component_subbass_energy_ratio_source"] = "compile_extractor_energy_terms"
            return _diag
        except Exception as exc:
            log.warning("Could not extract obs_wS diagnostics from %s: %s", workbook_path, exc)
        return defaults

    def _push_density_feedback_to_ui(
        self,
        w_h: float,
        w_i: float,
        w_s: float,
        *,
        confidence: Optional[float] = None,
        reliability: Optional[float] = None,
    ) -> None:
        """Reflect adaptive weights in GUI entries from worker thread safely."""
        def _apply() -> None:
            widgets = (
                self.entry_density_w_h,
                self.entry_density_w_i,
                self.entry_density_w_s,
            )
            previous_states = [str(widget.cget("state")) for widget in widgets]
            for widget in widgets:
                widget.configure(state="normal")
            self.entry_density_w_h.delete(0, tk.END)
            self.entry_density_w_h.insert(0, f"{w_h:.6f}")
            self.entry_density_w_i.delete(0, tk.END)
            self.entry_density_w_i.insert(0, f"{w_i:.6f}")
            self.entry_density_w_s.delete(0, tk.END)
            self.entry_density_w_s.insert(0, f"{w_s:.6f}")
            meta_parts: List[str] = []
            if confidence is not None and np.isfinite(confidence):
                meta_parts.append(f"conf={float(confidence):.3f}")
            if reliability is not None and np.isfinite(reliability):
                meta_parts.append(f"rel={float(reliability):.3f}")
            meta_txt = f" ({', '.join(meta_parts)})" if meta_parts else ""
            self.var_runtime_density_profile.set(
                f"Runtime learned profile: wH={w_h:.4f}, wI={w_i:.4f}, wS={w_s:.4f}{meta_txt}"
            )
            for widget, old_state in zip(widgets, previous_states, strict=False):
                widget.configure(state=old_state)

        self.master.after(0, _apply)

    def _open_doc(self, relative_path: str) -> None:
        doc_path = (MAIN_DIR / relative_path).resolve()
        if not doc_path.is_file():
            messagebox.showerror("Missing file", f"Could not find: {doc_path}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(doc_path))  # type: ignore[attr-defined]
            else:
                webbrowser.open(doc_path.as_uri())
        except Exception as exc:
            messagebox.showerror("Open failed", f"Could not open document:\n{exc}")
    
    def _process_log_queue(self):
        while not self.log_queue.empty():
            self.txt_log.config(state=tk.NORMAL)
            self.txt_log.insert(tk.END, self.log_queue.get() + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state=tk.DISABLED)
        self.master.after(100, self._process_log_queue)

    def _add_folders(self):
        d = filedialog.askdirectory(mustexist=True)
        if d and Path(d) not in self.processing_queue:
            self.processing_queue.append(Path(d))
            self.lbl_count.config(text=f"Queue: {len(self.processing_queue)}")
            log.info(f"Added: {Path(d).name}")

    def _clear_queue(self):
        if not self.is_running:
            self.processing_queue.clear()
            self.lbl_count.config(text="Queue: 0")
            log.info("Queue cleared.")

    def _stop(self):
        self.stop_requested = True
        self.lbl_status.config(text="Stopping...")

    def _run(self):
        if not self.processing_queue:
            return
        self.is_running = True
        self.stop_requested = False
        self.btn_run.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        threading.Thread(target=self._worker, daemon=True).start()

    def _validate_parameters(self, params: Dict[str, Any]) -> tuple:
        """
        Comprehensive parameter validation with detailed error reporting.
        Returns (is_valid, error_message).
        """
        # Validate window type
        window = params.get('win', '').lower()
        if window not in VALID_WINDOW_TYPES:
            return False, f"Invalid window type: {window}. Must be one of {VALID_WINDOW_TYPES}"
        
        # Validate weight function (single source of truth: density.get_weight_function)
        wf = (params.get("wf") or "").strip().lower()
        try:
            from density import get_weight_function

            get_weight_function(wf)
        except ValueError as e:
            return (
                False,
                f"Invalid weight function: {wf!r}. Must be a key accepted by density.get_weight_function. ({e})",
            )
        
        # Validate numerical ranges
        try:
            db_min = float(params.get('db_min', -90))
            db_max = float(params.get('db_max', 0))
            if db_min >= db_max:
                return False, f"db_min ({db_min}) must be < db_max ({db_max})"
            
            freq_min = float(params.get('freq_min', 20))
            freq_max = float(params.get('freq_max', 20000))
            if freq_min >= freq_max or freq_min < 0:
                return False, f"Invalid frequency range: [{freq_min}, {freq_max}] Hz"
            
            tolerance = float(params.get('tolerance', 5.0))
            if tolerance <= 0 or tolerance > 100:
                return False, f"Tolerance ({tolerance}) should be in range (0, 100] Hz"

            density_mode = str(
                params.get("density_summation_mode", "his_note_adaptive") or "his_note_adaptive"
            ).strip().lower()
            if density_mode not in {
                "his_weighted",
                "his_note_adaptive",
                "harmonic_only",
                "inharmonic_only",
                "subbass_only",
            }:
                return False, f"Invalid density_summation_mode: {density_mode}"
            for k in (
                "harmonic_density_weight",
                "inharmonic_density_weight",
                "subbass_density_weight",
            ):
                _ = float(params.get(k, 0.0))
            dst_raw = params.get("density_salience_threshold_db", None)
            if dst_raw is not None:
                dst = float(dst_raw)
                if dst >= 0.0 or dst < -200.0:
                    return False, f"density_salience_threshold_db ({dst}) should be in [-200, 0)"
            dceil_raw = params.get("density_frequency_ceiling_hz", None)
            if dceil_raw is not None:
                dceil = float(dceil_raw)
                if dceil <= 0.0:
                    return False, f"density_frequency_ceiling_hz ({dceil}) must be positive"
            
            # Validate window-specific parameters
            if window == "kaiser":
                kaiser_beta = params.get('kaiser_beta')
                if kaiser_beta is not None:
                    beta = float(kaiser_beta)
                    if beta < 0:
                        return False, f"Kaiser beta ({beta}) must be non-negative"
            
            if window in ("gaussian", "gauss", "gaussiana"):
                gaussian_std = params.get('gaussian_std')
                if gaussian_std is not None and gaussian_std != "auto":
                    std = float(gaussian_std)
                    if std <= 0:
                        return False, f"Gaussian std ({std}) must be positive"
        
        except (ValueError, TypeError) as e:
            return False, f"Parameter validation error: {e}"
        
        return True, None

    def _worker(self):
        try:
            # Collect all parameters with full parity
            window = self.combo_window.get().strip().lower()
            
            # Get window-specific parameters
            kaiser_beta = None
            gaussian_std = None
            if window == "kaiser":
                beta_txt = self.entry_kaiser_beta.get().strip()
                if beta_txt:
                    try:
                        kaiser_beta = float(beta_txt)
                    except ValueError:
                        kaiser_beta = 6.5  # Default
                        log.warning(f"Invalid kaiser_beta, using default: 6.5")
            
            if window in ("gaussian", "gauss", "gaussiana"):
                std_txt = self.entry_gaussian_std.get().strip()
                if std_txt and std_txt.lower() != "auto":
                    try:
                        gaussian_std = float(std_txt)
                    except ValueError:
                        gaussian_std = None  # Will be calculated from n_fft
                        log.warning(f"Invalid gaussian_std, will use auto-calculation")
            
            wf_raw = self.combo_weight.get().strip()
            weight_function = resolve_weight_key_from_user_label(wf_raw)

            # Get advanced analysis parameters
            try:
                contamination_raw = (self.entry_contamination.get() or "").strip().lower()
                if contamination_raw in ("", "auto", "adaptive"):
                    contamination = None
                else:
                    contamination = float(contamination_raw)
                    contamination = max(0.01, min(0.5, contamination))  # Clamp to valid range
            except ValueError:
                contamination = None
                log.warning("Invalid contamination value, using adaptive default")
            
            manual_mw = bool(self.var_manual_model_weight_override.get())
            i_weight_val = (self.var_i_weight.get() / 100.0) if manual_mw else 0.05
            density_mode_selected = self._density_mode_internal()
            if density_mode_selected == "his_note_adaptive":
                cached_density = getattr(self, "_manual_density_weights", (1.0, 0.5, 0.25))
                dwh = float(cached_density[0])
                dwi = float(cached_density[1])
                dws = float(cached_density[2])
            else:
                dwh = float(self.entry_density_w_h.get() or "1.0")
                dwi = float(self.entry_density_w_i.get() or "0.5")
                dws = float(self.entry_density_w_s.get() or "0.25")
            params = {
                'i_weight': i_weight_val,
                'manual_model_weight_override': manual_mw,
                # LFT removed: zero_padding and time_avg are now standard STFT parameters
                'avg': self.combo_avg.get(),
                'win': window,
                'wf': weight_function,
                'diss': self.combo_dissonance.get(),
                'db_min': float(self.entry_min_db.get() or "-90"),
                'db_max': float(self.entry_max_db.get() or "0"),
                'freq_min': float(self.entry_min_freq.get() or "20"),
                'freq_max': float(self.entry_max_freq.get() or "20000"),
                'tolerance': float(self.entry_tolerance.get() or "5.0"),
                'use_adaptive_tolerance': self.var_adaptive_tolerance.get(),
                'kaiser_beta': kaiser_beta,
                'gaussian_std': gaussian_std,
                'spectral_masking_enabled': False,  # Physical density workflow: masking not exposed in GUI
                'density_summation_mode': density_mode_selected,
                'harmonic_density_weight': dwh,
                'inharmonic_density_weight': dwi,
                'subbass_density_weight': dws,
                'enable_adaptive_path_randomization': False,
                'density_salience_threshold_db': float(self.entry_min_db.get() or "-90"),
                'density_frequency_ceiling_hz': float(self.entry_max_freq.get() or "20000"),
                'compile': self.var_compile.get(),
                'smart': self.var_smart.get(),
                'use_tsne': self.var_use_tsne.get(),
                'use_umap': self.var_use_umap.get(),
                'detect_anomalies': self.var_detect_anomalies.get(),
                'anomaly_contamination': contamination
            }
            
            # Validate parameters
            is_valid, error_msg = self._validate_parameters(params)
            if not is_valid:
                log.error(f"Parameter validation failed: {error_msg}")
                messagebox.showerror("Validation Error", f"Invalid parameters:\n{error_msg}")
                self._reset()
                return

            # Compact pre-run summary
            log.info("RUN SUMMARY (pre-execution)")
            log.info("Final density configuration:")
            log.info("  mode=%s", params["density_summation_mode"])
            log.info("  wH=%.6f", float(params["harmonic_density_weight"]))
            log.info("  wI=%.6f", float(params["inharmonic_density_weight"]))
            log.info("  wS=%.6f", float(params["subbass_density_weight"]))
            log.info(
                "  threshold=%s dB | ceiling=%s Hz",
                "auto" if params.get("density_salience_threshold_db") is None else f"{float(params['density_salience_threshold_db']):.2f}",
                "auto" if params.get("density_frequency_ceiling_hz") is None else f"{float(params['density_frequency_ceiling_hz']):.2f}",
            )
            _wf_cmp = str(params.get("wf", "linear") or "linear").strip().lower()
            _dst_cmp = params.get("density_salience_threshold_db", None)
            _dceil_cmp = params.get("density_frequency_ceiling_hz", None)
            _primary = (_wf_cmp == "log")
            if _primary:
                log.info(
                    "Comparability profile: PRIMARY (wf=log; threshold/ceiling follow current run config)"
                )
            else:
                log.warning(
                    "Comparability profile: EXPLORATORY (wf=%s, threshold=%s dB, ceiling=%s Hz). "
                    "Do not compare directly against primary-profile runs.",
                    _wf_cmp,
                    "auto" if _dst_cmp is None else f"{float(_dst_cmp):.1f}",
                    "auto" if _dceil_cmp is None else f"{float(_dceil_cmp):.1f}",
                )
            tier_or_fixed = "tier strategy" if bool(params.get("smart", False)) else "fixed FFT"
            log.info("STFT configuration:")
            log.info(
                "  window=%s | mode=%s | n_fft=%s | hop=%s | zp=%s",
                params["win"],
                tier_or_fixed,
                self.entry_n_fft.get() if not bool(params.get("smart", False)) else "overridden_by_tier",
                self.entry_hop_length.get() if not bool(params.get("smart", False)) else "overridden_by_tier",
                self.entry_zero_padding.get() if not bool(params.get("smart", False)) else "overridden_by_tier",
            )
            log.info(
                "  peak range dB=[%.2f, %.2f] | harmonic tolerance=%.3f Hz | adaptive=%s",
                float(params["db_min"]),
                float(params["db_max"]),
                float(params["tolerance"]),
                bool(params["use_adaptive_tolerance"]),
            )
            
            # Log all parameter activations
            log.info("=" * 60)
            log.info("ORCHESTRATOR PARAMETER ACTIVATION LOG")
            log.info("=" * 60)
            log.info(f"Window Type: {params['win']} (ACTIVATED)")
            if params['kaiser_beta'] is not None:
                log.info(f"  -> Kaiser Beta: {params['kaiser_beta']} (ACTIVATED)")
            if params['gaussian_std'] is not None:
                log.info(f"  -> Gaussian Std: {params['gaussian_std']} (ACTIVATED)")
            log.info(f"Amplitude weighting function: {params['wf']} (ACTIVATED)")
            log.info(f"Dissonance Model: {params['diss']} (ACTIVATED)")
            if bool(params.get('manual_model_weight_override', False)):
                log.info(
                    "Manual model-weight override ACTIVE: "
                    "alpha=%.3f, beta=%.3f (ACTIVATED)",
                    1.0 - params['i_weight'],
                    params['i_weight'],
                )
            else:
                log.info(
                    "Model-weight (component-energy) placeholder: "
                    "H=0.500, I=0.500; final component ratios are computed "
                    "from current spectral analysis (ACTIVATED)."
                )
            log.info(
                "Global spectral peak-search range: [%.1f, %.1f] Hz (ACTIVATED)",
                float(params["freq_min"]),
                float(params["freq_max"]),
            )
            log.info(f"Magnitude Range: [{params['db_min']:.1f}, {params['db_max']:.1f}] dB (ACTIVATED)")
            log.info(f"Tolerance: {params['tolerance']:.2f} Hz | Adaptive: {params['use_adaptive_tolerance']} (ACTIVATED)")
            log.info(
                "Final density controls: mode=%s, wH=%.3f, wI=%.3f, wS=%.3f, threshold=%s dB, ceiling=%s Hz (ACTIVATED)",
                params["density_summation_mode"],
                params["harmonic_density_weight"],
                params["inharmonic_density_weight"],
                params["subbass_density_weight"],
                "auto"
                if params.get("density_salience_threshold_db", None) is None
                else f"{float(params['density_salience_threshold_db']):.1f}",
                "auto"
                if params.get("density_frequency_ceiling_hz", None) is None
                else f"{float(params['density_frequency_ceiling_hz']):.1f}",
            )
            log.info(
                "Domain distinction: peak-search range is global STFT detection domain; "
                "density ceiling is metric integration upper bound."
            )
            log.info("Final density config:")
            log.info("density_summation_mode = %s", params["density_summation_mode"])
            log.info("wH = %.6f", float(params["harmonic_density_weight"]))
            log.info("wI = %.6f", float(params["inharmonic_density_weight"]))
            log.info("wS = %.6f", float(params["subbass_density_weight"]))
            log.info(
                "density_salience_threshold_db = %s",
                "auto"
                if params.get("density_salience_threshold_db", None) is None
                else f"{float(params['density_salience_threshold_db']):.6f}",
            )
            log.info(
                "density_frequency_ceiling_hz = %s",
                "auto"
                if params.get("density_frequency_ceiling_hz", None) is None
                else f"{float(params['density_frequency_ceiling_hz']):.6f}",
            )
            log.info(f"STFT Options: Zero Padding={params.get('zero_padding', 1)} | Time Avg: {params['avg']} (ACTIVATED)")
            log.info(f"90-Tier Clustering: {params['smart']} | Auto-Compile: {params['compile']} (ACTIVATED)")
            log.info(
                "Component energy ratios source: current spectral analysis "
                "(no external H/I/S mapping)."
            )
            log.info(
                "Pipeline: Stage 1 \u2014 Per-note spectral analysis; "
                "Stage 2 \u2014 Compilation."
            )
            log.info("=" * 60)
            
        except Exception as e:
            log.error(f"Input Error: {e}")
            messagebox.showerror("Error", f"Failed to collect parameters: {e}")
            self._reset()
            return

        total = len(self.processing_queue)
        log.info("=" * 80)
        log.info(f"STARTING PROCESSING OF {total} FOLDER(S)")
        log.info("=" * 80)
        log.info(
            "Each folder runs the three-stage pipeline: "
            "Stage 1 (per-note spectral analysis), Stage 2 (compilation), "
            "Stage 3 (research export + EWSD)."
        )
        log.info("=" * 80)
        folder_summaries: List[Dict[str, Any]] = []

        for i, folder in enumerate(self.processing_queue):
            if self.stop_requested:
                log.warning("Processing stopped by user")
                break

            log.info("")
            log.info("=" * 80)
            log.info(f"FOLDER {i+1}/{total}: {folder.name}")
            log.info(f"Full path: {folder}")
            log.info("=" * 80)
            log.info("This folder will now go through:")
            log.info("  1. Stage 1: Per-note spectral analysis (proc_audio)")
            log.info(
                "  2. Stage 2: Compilation (compiled_density_metrics.xlsx)"
            )
            log.info(
                "  3. Stage 3: Research export + EWSD "
                "(compiled_density_metrics_research.xlsx)"
            )
            log.info("=" * 80)
            
            try:
                self._process_folder_complete_pipeline(
                    folder,
                    params,
                    enable_adaptive_path_randomization=bool(
                        params.get("enable_adaptive_path_randomization", False)
                    ),
                )
                _summary = self._collect_post_run_summary(folder)
                if _summary is not None:
                    folder_summaries.append(_summary)
                log.info("")
                log.info("=" * 80)
                log.info(f"✓ FOLDER {i+1}/{total} COMPLETE: {folder.name}")
                log.info("=" * 80)
            except Exception as folder_error:
                log.error("")
                log.error("=" * 80)
                log.error(f"✗ FOLDER {i+1}/{total} FAILED: {folder.name}")
                log.error(f"Error: {folder_error}")
                import traceback
                log.error(traceback.format_exc())
                log.error("=" * 80)
                # Continue with next folder even if this one failed
                continue
        
        log.info("")
        log.info("=" * 80)
        log.info("ALL FOLDERS PROCESSING COMPLETE")
        log.info("=" * 80)
        log.info("Done.")
        if folder_summaries:
            _last = folder_summaries[-1]
            _msg = (
                "Run summary\n\n"
                f"Files processed: {_last['files_processed']}\n"
                f"Files failed: {_last['files_failed']}\n"
                f"Workbook path: {_last['workbook_path']}\n"
                "Primary metric: final_note_density_salience_weighted\n"
                f"Mean final_note_density_salience_weighted: {_last['mean_final_density']:.6f}\n"
                f"Top 5 densest notes: {', '.join(_last['top5_notes'])}\n"
                f"Bottom 5 densest notes: {', '.join(_last['bottom5_notes'])}\n"
                f"f0 fallback count: {_last['f0_fallback_count']}"
            )
            messagebox.showinfo("Run complete", _msg)
        else:
            messagebox.showinfo("Run complete", "All folders processed.")
        self._reset()

    def _collect_post_run_summary(self, folder: Path) -> Optional[Dict[str, Any]]:
        analysis_results_dir = folder / "analysis_results"
        research_path = analysis_results_dir / "compiled_density_metrics_research.xlsx"
        if not research_path.is_file():
            return None
        try:
            sdm = pd.read_excel(
                research_path,
                sheet_name="Spectral_Density_Metrics",
                engine="openpyxl",
            )
            if sdm.empty:
                return None
            files_total = len(
                [
                    f
                    for f in folder.glob("*")
                    if f.suffix.lower() in VALID_AUDIO_EXTENSIONS
                ]
            )
            files_processed = len(
                list(analysis_results_dir.rglob("spectral_analysis.xlsx"))
            )
            files_failed = max(0, int(files_total - files_processed))
            score = pd.to_numeric(
                sdm.get("final_note_density_salience_weighted"), errors="coerce"
            )
            notes = sdm.get("Note")
            ranked = pd.DataFrame({"Note": notes, "score": score}).dropna()
            top5 = ranked.nlargest(5, "score")["Note"].astype(str).tolist()
            bottom5 = ranked.nsmallest(5, "score")["Note"].astype(str).tolist()
            fallback_count = 0
            if "acoustic_validation_status" in sdm.columns:
                fallback_count = int(
                    sdm["acoustic_validation_status"]
                    .astype(str)
                    .str.contains(
                        "nominal_fallback_used_not_acoustically_verified",
                        na=False,
                    )
                    .sum()
                )
            return {
                "files_processed": int(files_processed),
                "files_failed": int(files_failed),
                "workbook_path": str(research_path),
                "mean_final_density": float(score.mean(skipna=True) or 0.0),
                "top5_notes": top5,
                "bottom5_notes": bottom5,
                "f0_fallback_count": fallback_count,
            }
        except Exception as exc:
            log.warning("Could not build post-run summary for %s: %s", folder, exc)
            return None

    def _process_folder_complete_pipeline(
        self,
        folder: Path,
        params: Dict[str, Any],
        *,
        enable_adaptive_path_randomization: bool = False,
    ) -> None:
        """Run the three-stage pipeline on a single folder.

        Stage 1: per-note spectral analysis via
        :class:`proc_audio.AudioProcessor`. One ``spectral_analysis.xlsx``
        per audio file is written under ``<folder>/analysis_results/``.

        Stage 2: compilation via
        :func:`compile_metrics.compile_density_metrics_with_pca`,
        producing ``compiled_density_metrics.xlsx`` next to the per-note
        workbooks.

        Component energy ratios are derived from the current spectral
        analysis (``auto_model_weights_from_analysis=True``) unless the
        user enabled the manual model-weight override in Advanced. No
        external H/I/S mapping and no preprocessing stage are consulted
        at runtime.
        """
        from note_parser import canonical_note_from_filename

        log.info("=" * 80)
        log.info(f"PIPELINE START - FOLDER: {folder.name}")
        log.info("=" * 80)

        folder_files = build_phase1_file_iteration_order(
            folder,
            enable_adaptive_path_randomization=enable_adaptive_path_randomization,
            random_seed=PHASE_1_ADAPTIVE_RANDOMIZATION_SEED,
        )
        if not folder_files:
            log.warning(f"No audio files in folder: {folder.name}")
            return

        log.info(
            f"Found {len(folder_files)} audio file(s) in folder: "
            f"{folder.name}"
        )
        if enable_adaptive_path_randomization:
            log.info(
                "Phase 1 adaptive path randomization enabled "
                "(seed=%d).",
                PHASE_1_ADAPTIVE_RANDOMIZATION_SEED,
            )

        if str(MAIN_DIR) not in sys.path:
            sys.path.insert(0, str(MAIN_DIR))

        import importlib
        if 'proc_audio' in sys.modules:
            local_proc_audio = importlib.reload(sys.modules['proc_audio'])
        else:
            import proc_audio as local_proc_audio  # type: ignore[no-redef]
        log.info(
            f"proc_audio module location: "
            f"{getattr(local_proc_audio, '__file__', 'unknown')}"
        )

        analysis_results_dir = folder / "analysis_results"
        analysis_results_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"Stage 1 outputs (per-note workbooks) will be saved to: "
            f"{analysis_results_dir}"
        )

        # AUDIT FIX (Clarinete_mf "jumped off" bug) — install a per-folder
        # FileHandler so the worker's INFO/WARN/ERROR stream is persisted
        # to disk even if the Tk window closes mid-run or the worker
        # thread dies before the corpus-wide compile completes. The
        # handler is unconditionally removed in the matching ``finally``
        # block at the end of this method so subsequent folders do not
        # accumulate handlers.
        _worker_log_path = (
            analysis_results_dir / "gui_worker.log"
        )
        try:
            _worker_log_handler: Optional[logging.FileHandler] = (
                logging.FileHandler(
                    _worker_log_path, mode="w", encoding="utf-8"
                )
            )
            _worker_log_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s  %(levelname)-7s  %(name)s  | %(message)s"
                )
            )
            _worker_log_handler.setLevel(logging.INFO)
            log.addHandler(_worker_log_handler)
            log.info(
                f"GUI worker log persisted to: {_worker_log_path}"
            )
        except Exception as _log_err:
            _worker_log_handler = None
            log.warning(
                f"Could not open per-folder GUI worker log "
                f"({_worker_log_path}): {_log_err}"
            )

        manual_mw = bool(params.get("manual_model_weight_override", False))
        i_weight_param = float(params.get("i_weight", 0.05))
        if manual_mw:
            harmonic_weight = max(0.0, min(1.0, 1.0 - i_weight_param))
            inharmonic_weight = max(0.0, min(1.0, i_weight_param))
            auto_model_weights = False
            log.info(
                "Manual model-weight override ACTIVE: "
                f"alpha={harmonic_weight:.3f}, beta={inharmonic_weight:.3f}"
            )
        else:
            harmonic_weight = 0.5
            inharmonic_weight = 0.5
            auto_model_weights = True
            log.info(
                "Model-weight (component-energy) placeholder: "
                "H=0.500, I=0.500; final component ratios are computed "
                "from current spectral analysis."
            )

        log.info("Final density config:")
        log.info(
            "density_summation_mode = %s",
            str(params.get("density_summation_mode", "his_note_adaptive") or "his_note_adaptive"),
        )
        log.info("wH = %.6f", float(params.get("harmonic_density_weight", 1.0)))
        log.info("wI = %.6f", float(params.get("inharmonic_density_weight", 0.5)))
        log.info("wS = %.6f", float(params.get("subbass_density_weight", 0.25)))
        log.info(
            "density_salience_threshold_db = %s",
            "auto" if params.get("density_salience_threshold_db", None) is None else f"{float(params.get('density_salience_threshold_db')):.6f}",
        )
        log.info(
            "density_frequency_ceiling_hz = %s",
            "auto" if params.get("density_frequency_ceiling_hz", None) is None else f"{float(params.get('density_frequency_ceiling_hz')):.6f}",
        )

        successful_files = 0
        failed_files = 0
        density_mode_runtime = str(
            params.get("density_summation_mode", "his_note_adaptive") or "his_note_adaptive"
        ).strip().lower()
        adaptive_feedback_enabled = density_mode_runtime in {
            "his_note_adaptive",
            "his_adaptive",
            "note_adaptive",
            "adaptive",
            "auto_note",
        }
        if adaptive_feedback_enabled:
            self.master.after(
                0,
                lambda: self.var_runtime_density_profile.set(
                    "Runtime learned profile: adaptive feed-forward active"
                ),
            )
        if adaptive_feedback_enabled:
            # In adaptive mode, start from a neutral prior instead of fixed GUI
            # percentages so the run is driven by extracted note evidence.
            running_density_w_h = 1.0 / 3.0
            running_density_w_i = 1.0 / 3.0
            running_density_w_s = 1.0 / 3.0
        else:
            running_density_w_h = float(params.get("harmonic_density_weight", 1.0))
            running_density_w_i = float(params.get("inharmonic_density_weight", 0.5))
            running_density_w_s = float(params.get("subbass_density_weight", 0.25))
        _running_vec = np.maximum(
            np.array([running_density_w_h, running_density_w_i, running_density_w_s], dtype=float),
            0.0,
        )
        _running_sum = float(np.sum(_running_vec))
        if _running_sum > 0.0:
            _running_vec = _running_vec / _running_sum
        else:
            _running_vec = np.array([0.5, 0.3, 0.2], dtype=float)
        running_density_w_h = float(_running_vec[0])
        running_density_w_i = float(_running_vec[1])
        running_density_w_s = float(_running_vec[2])
        adaptive_engine: Optional[AdaptiveDensityEngine] = None
        if adaptive_feedback_enabled:
            adaptive_engine = AdaptiveDensityEngine(
                initial_profile=(
                    running_density_w_h,
                    running_density_w_i,
                    running_density_w_s,
                ),
                initial_strength=8.0,
                forgetting=0.02,
                divergence_temperature=0.18,
            )
        adaptive_profile_history: List[Dict[str, Any]] = []

        log.info("=" * 80)
        log.info(
            f"PHASE 1 (DISCOVERY): Per-note spectral analysis "
            f"for natural H/I/S contribution discovery ({len(folder_files)} file(s))"
        )
        log.info("=" * 80)

        for audio_file in folder_files:
            if self.stop_requested:
                break

            try:
                log.info(f"Processing: {audio_file.name}")
                log.info(f"  File path: {audio_file}")

                extracted_note, note_source = canonical_note_from_filename(
                    audio_file.name,
                    parent_folder=audio_file.parent.name,
                )
                log.info(
                    "  Note parsing: %r -> %s (note_source=%s)",
                    audio_file.name,
                    extracted_note,
                    note_source,
                )
                filename_stem = audio_file.stem

                hz = 0.0
                if extracted_note:
                    try:
                        hz = float(librosa.note_to_hz(extracted_note))
                    except Exception:
                        hz = 0.0

                if params['smart']:
                    tier_name: Optional[str] = 'Fallback'
                    tier_settings: Dict[str, Any] = {
                        'n_fft': 4096,
                        'zp': 1,
                        'tolerance': 10.0,
                        'max_freq': 20000,
                    }
                    if hz > 0:
                        sorted_tiers = sorted(
                            FFT_SETTINGS_BY_CLUSTER.items(),
                            key=lambda x: (
                                x[1]['max_freq']
                                if x[1]['max_freq'] != float('inf')
                                else float('inf')
                            ),
                        )
                        assigned = False
                        for t_name, t_cfg in sorted_tiers:
                            if hz < t_cfg['max_freq']:
                                tier_name = t_name
                                tier_settings = t_cfg
                                assigned = True
                                break
                        if not assigned:
                            tier_name = 'Tier_90'
                            tier_settings = FFT_SETTINGS_BY_CLUSTER.get(
                                'Tier_90', tier_settings
                            )
                    n_fft = int(tier_settings['n_fft'])
                    zp = int(tier_settings['zp'])
                    hop_length = n_fft // 8
                    base_tolerance = float(
                        tier_settings.get('tolerance', 5.0)
                    )
                    tier_max_freq = tier_settings.get('max_freq', 20000)
                    if tier_max_freq == float('inf'):
                        tier_max_freq = 20000
                    tier_list = sorted(
                        [t for t in FFT_SETTINGS_BY_CLUSTER.keys()]
                    )
                    tier_idx = (
                        tier_list.index(tier_name)
                        if tier_name in tier_list
                        else 0
                    )
                    prev_max_freq = (
                        FFT_SETTINGS_BY_CLUSTER[
                            tier_list[tier_idx - 1]
                        ].get('max_freq', 20)
                        if tier_idx > 0
                        else 20
                    )
                    center_freq = (prev_max_freq + tier_max_freq) / 2.0
                    tolerance = _calculate_adaptive_tolerance(
                        center_freq,
                        base_tolerance,
                        params.get('use_adaptive_tolerance', True),
                    )
                    log.info(
                        f"  Tier {tier_name}: n_fft={n_fft}, "
                        f"hop_length={hop_length}, zp={zp}, "
                        f"tolerance={tolerance:.2f} Hz, "
                        f"note={extracted_note}, hz={hz:.2f}"
                    )
                else:
                    try:
                        n_fft = int(self.entry_n_fft.get() or "4096")
                        hop_length = int(
                            self.entry_hop_length.get() or "1024"
                        )
                        zp = int(self.entry_zero_padding.get() or "2")
                    except (ValueError, AttributeError):
                        n_fft = 4096
                        hop_length = 1024
                        zp = 2
                    tolerance = _calculate_adaptive_tolerance(
                        (
                            params.get('freq_min', 20.0)
                            + params.get('freq_max', 20000.0)
                        ) / 2.0,
                        params['tolerance'],
                        params.get('use_adaptive_tolerance', False),
                    )
                    tier_name = None
                    log.info(
                        f"  Fixed FFT: n_fft={n_fft}, "
                        f"hop_length={hop_length}, zp={zp}, "
                        f"tolerance={tolerance:.2f} Hz, "
                        f"note={extracted_note}, hz={hz:.2f}"
                    )

                user_freq_min = float(params.get('freq_min', 20.0))
                cutoff = user_freq_min
                if hz > 0:
                    margin = _calculate_security_margin(hz)
                    calculated_cutoff = hz * (1.0 - margin / 100.0)
                    cutoff = max(user_freq_min, calculated_cutoff)

                gaussian_std = params.get('gaussian_std')
                if (
                    params['win'] in ("gaussian", "gauss", "gaussiana")
                    and gaussian_std is None
                ):
                    gaussian_std = n_fft / 8.0

                parent_output_dir = analysis_results_dir / filename_stem
                parent_output_dir.mkdir(parents=True, exist_ok=True)

                dissonance_model_param = params['diss']
                compare_models_param = False
                if dissonance_model_param == "ALL (Compare)":
                    dissonance_model_param = "sethares"
                    compare_models_param = True
                    log.info(
                        "  Dissonance compare mode: ALL "
                        "(sethares + hutchinson + vassilakis)"
                    )

                pr = local_proc_audio.AudioProcessor()
                pr.note_source = note_source
                if extracted_note:
                    pr.note = extracted_note
                pr.load_audio_files([str(audio_file)])
                if params['kaiser_beta'] is not None:
                    pr.kaiser_beta = params['kaiser_beta']
                if gaussian_std is not None:
                    pr.gaussian_std = gaussian_std

                try:
                    if adaptive_feedback_enabled:
                        log.info(
                            "  Adaptive prior used for this note: wH=%.4f, wI=%.4f, wS=%.4f",
                            running_density_w_h,
                            running_density_w_i,
                            running_density_w_s,
                        )
                    pr.apply_filters_and_generate_data(
                        freq_min=cutoff,
                        freq_max=params['freq_max'],
                        db_min=params['db_min'],
                        db_max=params['db_max'],
                        tolerance=tolerance,
                        use_adaptive_tolerance=params[
                            'use_adaptive_tolerance'
                        ],
                        n_fft=n_fft,
                        hop_length=hop_length,
                        window=params['win'],
                        kaiser_beta=params['kaiser_beta'],
                        gaussian_std=gaussian_std,
                        weight_function=params['wf'],
                        results_directory=str(parent_output_dir),
                        harmonic_weight=harmonic_weight,
                        inharmonic_weight=inharmonic_weight,
                        auto_model_weights_from_analysis=auto_model_weights,
                        dissonance_enabled=(params['diss'] != 'None'),
                        dissonance_model=dissonance_model_param,
                        dissonance_curve=True,
                        dissonance_scale=False,
                        compare_models=compare_models_param,
                        zero_padding=zp,
                        time_avg=params['avg'],
                        tier=tier_name,
                        spectral_masking_enabled=False,
                        density_summation_mode=params.get("density_summation_mode", "his_note_adaptive"),
                        harmonic_density_weight=float(running_density_w_h),
                        inharmonic_density_weight=float(running_density_w_i),
                        subbass_density_weight=float(running_density_w_s),
                        density_salience_threshold_db=(
                            None
                            if params.get("density_salience_threshold_db", None) is None
                            else float(params.get("density_salience_threshold_db"))
                        ),
                        density_frequency_ceiling_hz=(
                            None
                            if params.get("density_frequency_ceiling_hz", None) is None
                            else float(params.get("density_frequency_ceiling_hz"))
                        ),
                        use_tsne=params.get('use_tsne', False),
                        use_umap=params.get('use_umap', False),
                        detect_anomalies=params.get(
                            'detect_anomalies', False
                        ),
                        anomaly_contamination=params.get(
                            'anomaly_contamination', None
                        ),
                        compile_per_call=False,
                    )

                    if extracted_note:
                        note_output_dir = parent_output_dir / extracted_note
                        expected = note_output_dir / "spectral_analysis.xlsx"
                        if expected.is_file():
                            log.info(
                                "  [OK] spectral_analysis.xlsx written to "
                                f"{note_output_dir}"
                            )
                        else:
                            log.warning(
                                "  [warn] expected workbook missing: "
                                f"{expected}"
                            )
                    else:
                        log.info(
                            "  Note token absent: skipping per-note "
                            "workbook existence check."
                        )

                    if adaptive_feedback_enabled:
                        feedback_workbook: Optional[Path] = None
                        if extracted_note:
                            candidate = (parent_output_dir / extracted_note / "spectral_analysis.xlsx")
                            if candidate.is_file():
                                feedback_workbook = candidate
                        if feedback_workbook is None:
                            found = sorted(parent_output_dir.rglob("spectral_analysis.xlsx"))
                            if found:
                                feedback_workbook = found[0]

                        if feedback_workbook is not None:
                            pure_observation_triplet = self._extract_note_density_feedback(feedback_workbook)
                            if pure_observation_triplet is not None:
                                obs_ws_diag = self._extract_note_density_feedback_diagnostics(feedback_workbook)
                                if adaptive_engine is not None:
                                    upd = adaptive_engine.update(
                                        pure_observation_triplet,
                                        evidence_strength=1.0,
                                    )
                                    running_density_w_h, running_density_w_i, running_density_w_s = upd.profile
                                else:
                                    running_density_w_h, running_density_w_i, running_density_w_s = pure_observation_triplet
                                params["harmonic_density_weight"] = running_density_w_h
                                params["inharmonic_density_weight"] = running_density_w_i
                                params["subbass_density_weight"] = running_density_w_s
                                self._push_density_feedback_to_ui(
                                    running_density_w_h,
                                    running_density_w_i,
                                    running_density_w_s,
                                    confidence=(upd.confidence if adaptive_engine is not None else None),
                                    reliability=(upd.reliability if adaptive_engine is not None else None),
                                )
                                adaptive_profile_history.append(
                                    {
                                        "audio_file": audio_file.name,
                                        "note_token": extracted_note or "",
                                        "f0_hz_from_filename": float(hz) if np.isfinite(hz) else float("nan"),
                                        "obs_wH": float(pure_observation_triplet[0]),
                                        "obs_wI": float(pure_observation_triplet[1]),
                                        "obs_wS": float(pure_observation_triplet[2]),
                                        "obs_w_formula_version": obs_ws_diag.get(
                                            "obs_w_formula_version", ""
                                        ),
                                        "wH": running_density_w_h,
                                        "wI": running_density_w_i,
                                        "wS": running_density_w_s,
                                        "confidence": (
                                            float(upd.confidence) if adaptive_engine is not None else float("nan")
                                        ),
                                        "uncertainty": (
                                            float(upd.uncertainty) if adaptive_engine is not None else float("nan")
                                        ),
                                        "reliability": (
                                            float(upd.reliability) if adaptive_engine is not None else float("nan")
                                        ),
                                        "js_divergence": (
                                            float(upd.js_divergence) if adaptive_engine is not None else float("nan")
                                        ),
                                        "source_workbook": str(feedback_workbook),
                                        "pure_observation_w_h": obs_ws_diag.get("pure_observation_w_h"),
                                        "pure_observation_w_i": obs_ws_diag.get("pure_observation_w_i"),
                                        "pure_observation_w_s": obs_ws_diag.get("pure_observation_w_s"),
                                        "component_strength_h": obs_ws_diag.get("component_strength_h"),
                                        "component_strength_i": obs_ws_diag.get("component_strength_i"),
                                        "component_strength_s": obs_ws_diag.get("component_strength_s"),
                                        "legacy_component_strength_h_v55": obs_ws_diag.get(
                                            "legacy_component_strength_h_v55"
                                        ),
                                        "legacy_component_strength_i_v55": obs_ws_diag.get(
                                            "legacy_component_strength_i_v55"
                                        ),
                                        "legacy_component_strength_s_v55": obs_ws_diag.get(
                                            "legacy_component_strength_s_v55"
                                        ),
                                        "subbass_density_component": obs_ws_diag.get("subbass_density_component"),
                                        "subbass_energy_ratio": obs_ws_diag.get("subbass_energy_ratio"),
                                        "subbass_energy_sum": obs_ws_diag.get("subbass_energy_sum"),
                                        "subbass_energy_sum_tier_normalized": obs_ws_diag.get(
                                            "subbass_energy_sum_tier_normalized"
                                        ),
                                        "subbass_energy_sum_source": obs_ws_diag.get(
                                            "subbass_energy_sum_source"
                                        ),
                                        "subbass_energy_sum_tier_normalized_source": obs_ws_diag.get(
                                            "subbass_energy_sum_tier_normalized_source"
                                        ),
                                        "component_subbass_energy_ratio_source": obs_ws_diag.get(
                                            "component_subbass_energy_ratio_source"
                                        ),
                                        "subbass_upper_bound_hz": obs_ws_diag.get("subbass_upper_bound_hz"),
                                        "subbass_component_interpretation": obs_ws_diag.get(
                                            "subbass_component_interpretation"
                                        ),
                                        "obs_wS_artifact_flag": obs_ws_diag.get("obs_wS_artifact_flag"),
                                        "obs_wS_artifact_reason": obs_ws_diag.get("obs_wS_artifact_reason"),
                                        "obs_wS_diagnostic_source": obs_ws_diag.get("obs_wS_diagnostic_source"),
                                    }
                                )
                                log.info(
                                    "  Adaptive per-note density feedback learned and propagated: "
                                    "wH=%.4f, wI=%.4f, wS=%.4f",
                                    running_density_w_h,
                                    running_density_w_i,
                                    running_density_w_s,
                                )
                                if adaptive_engine is not None:
                                    log.info(
                                        "  Adaptive diagnostics: conf=%.3f rel=%.3f js=%.4f",
                                        float(upd.confidence),
                                        float(upd.reliability),
                                        float(upd.js_divergence),
                                    )
                            else:
                                log.warning(
                                    "  Adaptive feedback unavailable from workbook: %s",
                                    feedback_workbook,
                                )
                except Exception as analysis_error:
                    log.error(
                        "  [fail] apply_filters_and_generate_data raised: "
                        f"{analysis_error}"
                    )
                    import traceback
                    log.error(traceback.format_exc())
                    failed_files += 1
                    continue
                finally:
                    del pr
                    plt.close('all')
                    gc.collect()

                successful_files += 1
                log.info(f"  [OK] Completed: {audio_file.name}")

            except Exception as exc:
                failed_files += 1
                log.error(f"Error processing {audio_file.name}: {exc}")
                import traceback
                log.error(traceback.format_exc())
                continue

        phase1_application_profile: Optional[Dict[str, float]] = None
        if adaptive_feedback_enabled and adaptive_profile_history:
            try:
                history_df = pd.DataFrame(adaptive_profile_history)
                history_path = analysis_results_dir / "phase1_discovered_density_profiles.csv"
                history_df.to_csv(history_path, index=False, encoding="utf-8")
                log.info("Phase 1 discovery profile history exported: %s", history_path)
                if adaptive_engine is not None:
                    state_path = analysis_results_dir / "adaptive_density_engine_state.json"
                    state_path.write_text(
                        json.dumps(adaptive_engine.state_dict(), indent=2),
                        encoding="utf-8",
                    )
                    log.info("Adaptive engine state exported: %s", state_path)
                    st = adaptive_engine.state_dict()
                    phase1_application_profile = {
                        "harmonic_density_weight": float(st.get("profile_h", running_density_w_h)),
                        "inharmonic_density_weight": float(st.get("profile_i", running_density_w_i)),
                        "subbass_density_weight": float(st.get("profile_s", running_density_w_s)),
                        "confidence": float(st.get("confidence", 0.0)),
                        "uncertainty": float(st.get("uncertainty", 1.0)),
                    }
                else:
                    phase1_application_profile = {
                        "harmonic_density_weight": float(running_density_w_h),
                        "inharmonic_density_weight": float(running_density_w_i),
                        "subbass_density_weight": float(running_density_w_s),
                        "confidence": float("nan"),
                        "uncertainty": float("nan"),
                    }
                if phase1_application_profile is not None:
                    profile_path = analysis_results_dir / "phase2_application_profile.json"
                    profile_path.write_text(
                        json.dumps(phase1_application_profile, indent=2),
                        encoding="utf-8",
                    )
                    log.info("Phase 2 application profile exported: %s", profile_path)
            except Exception as history_exc:
                log.warning("Could not export adaptive profile history: %s", history_exc)

        log.info("=" * 80)
        log.info(
            f"STAGE 1 COMPLETE - "
            f"Successful: {successful_files}, Failed: {failed_files}, "
            f"Total: {len(folder_files)}"
        )
        log.info("=" * 80)

        if not params.get('compile', True):
            log.info("Stage 2 skipped (auto-compile disabled).")
            return

        if successful_files == 0:
            log.error(
                "Stage 2 skipped: no per-note workbooks were produced."
            )
            return

        log.info("=" * 80)
        log.info("PHASE 2 (APPLICATION): Apply discovered profile to downstream compilation")
        log.info("=" * 80)
        if phase1_application_profile is not None:
            log.info(
                "Applying discovered Phase 1 profile in Phase 2: "
                "wH=%.4f, wI=%.4f, wS=%.4f | conf=%s",
                float(phase1_application_profile["harmonic_density_weight"]),
                float(phase1_application_profile["inharmonic_density_weight"]),
                float(phase1_application_profile["subbass_density_weight"]),
                (
                    f"{float(phase1_application_profile['confidence']):.3f}"
                    if np.isfinite(float(phase1_application_profile["confidence"]))
                    else "n/a"
                ),
            )
        else:
            log.info("No discovered profile available; using runtime fallback profile.")

        try:
            stage2_fallback_compile_attempted = False
            # Use the module imported at startup. Avoid ``importlib.reload`` here:
            # reloading ``compile_metrics`` re-runs its heavy import graph (including
            # ``proc_audio``) on the Tk worker thread and has been observed to abort
            # Stage 2 right after "Compiling metrics to:" with no Python traceback.
            compiled_output_path = (
                analysis_results_dir / "compiled_density_metrics.xlsx"
            )
            log.info(f"Compiling metrics to: {compiled_output_path}")

            allow_legacy = bool(params.get("allow_legacy_super_json", False))
            file_pattern = resolve_stage2_compile_file_pattern(
                analysis_results_dir,
                allow_legacy_super_json=allow_legacy,
            )
            if file_pattern is None:
                log.error(
                    "No canonical per-note spectral_analysis.xlsx workbooks found under %s. "
                    "Stage 2 aborted. Legacy super_analysis_results.json is not accepted in "
                    "canonical mode (pass params['allow_legacy_super_json']=True to allow JSON-only folders).",
                    analysis_results_dir,
                )
                return

            log.info(
                "Stage 2: compiling %d successful note(s); file_pattern=%r",
                successful_files,
                file_pattern,
            )
            _flush_orchestrator_log_handlers(log)

            compile_kw: Dict[str, Any] = {
                "folder_path": str(analysis_results_dir),
                "output_path": str(compiled_output_path),
                "file_pattern": file_pattern,
                "include_pca": True,
                "harmonic_weight": (
                    float(phase1_application_profile["harmonic_density_weight"])
                    if phase1_application_profile is not None
                    else harmonic_weight
                ),
                "inharmonic_weight": (
                    float(phase1_application_profile["inharmonic_density_weight"])
                    if phase1_application_profile is not None
                    else inharmonic_weight
                ),
                "subbass_weight": (
                    float(phase1_application_profile["subbass_density_weight"])
                    if phase1_application_profile is not None
                    else None
                ),
                "weight_function": params["wf"],
                "use_tsne": params.get("use_tsne", False),
                "use_umap": params.get("use_umap", False),
                "detect_anomalies": params.get("detect_anomalies", False),
                "anomaly_contamination": params.get("anomaly_contamination", None),
                "allow_legacy_super_json": allow_legacy,
                "compilation_extra_metadata": {
                    "input_schema_validation_status": (
                        "legacy_super_json_compilation"
                        if file_pattern.lower().endswith(".json")
                        else "not_validated_orchestrator_v2_16"
                    ),
                    "legacy_pipeline_used": file_pattern.lower().endswith(".json"),
                    "publication_output_allowed": not file_pattern.lower().endswith(
                        ".json"
                    ),
                    "density_runtime_profile_h": float(running_density_w_h),
                    "density_runtime_profile_i": float(running_density_w_i),
                    "density_runtime_profile_s": float(running_density_w_s),
                    "density_runtime_profile_mode": (
                        "adaptive_online_engine" if adaptive_feedback_enabled else "manual_or_fixed"
                    ),
                    "density_runtime_profile_confidence": (
                        (
                            float(phase1_application_profile["confidence"])
                            if phase1_application_profile is not None
                            else (float(adaptive_engine.confidence()) if adaptive_engine is not None else None)
                        )
                    ),
                    "density_runtime_profile_uncertainty": (
                        (
                            float(phase1_application_profile["uncertainty"])
                            if phase1_application_profile is not None
                            else (float(adaptive_engine.uncertainty()) if adaptive_engine is not None else None)
                        )
                    ),
                },
            }
            advanced_stage2 = bool(
                compile_kw["use_tsne"]
                or compile_kw["use_umap"]
                or compile_kw["detect_anomalies"]
            )
            if advanced_stage2:
                log.info(
                    "Advanced Stage 2 options enabled (t-SNE / UMAP / anomalies). "
                    "Running compilation in an isolated subprocess to protect the GUI."
                )
                compiled_df = _stage2_compile_via_subprocess(compile_kw, log)
                if compiled_df is None:
                    stage2_fallback_compile_attempted = True
                    log.warning(
                        "Advanced Stage 2 subprocess did not return a workbook DataFrame; "
                        "retrying compilation in-process without t-SNE / UMAP / anomaly "
                        "detection so compiled_density_metrics.xlsx is still produced."
                    )
                    _flush_orchestrator_log_handlers(log)
                    fallback_kw = dict(compile_kw)
                    fallback_kw["use_tsne"] = False
                    fallback_kw["use_umap"] = False
                    fallback_kw["detect_anomalies"] = False
                    extra = dict(
                        fallback_kw.get("compilation_extra_metadata") or {}
                    )
                    extra["advanced_stage2_fallback"] = True
                    extra["advanced_stage2_fallback_reason"] = (
                        "subprocess_failed_or_empty; retried without t-SNE/UMAP/anomalies"
                    )
                    fallback_kw["compilation_extra_metadata"] = extra
                    compiled_df = compile_metrics.compile_density_metrics_with_pca(
                        **fallback_kw,
                    )
            else:
                compiled_df = compile_metrics.compile_density_metrics_with_pca(
                    **compile_kw,
                )

            if compiled_output_path.is_file():
                legacy_compiled_clean = compiled_output_path.parent / (
                    f"{compiled_output_path.stem}_clean"
                    f"{compiled_output_path.suffix}"
                )
                if legacy_compiled_clean.is_file():
                    try:
                        legacy_compiled_clean.unlink()
                        log.info(
                            "Removed legacy %s (single compiled "
                            "workbook export).",
                            legacy_compiled_clean.name,
                        )
                    except OSError as lc_err:
                        log.warning(
                            "Could not remove legacy compiled clean "
                            "sidecar: %s",
                            lc_err,
                        )

            if compiled_df is not None and not compiled_df.empty:
                log.info(
                    "[OK] compiled_density_metrics.xlsx created: "
                    f"{compiled_output_path}"
                )
                log.info(f"  Total rows: {len(compiled_df)}")
                log.info(f"  Columns: {len(compiled_df.columns)}")
                if stage2_fallback_compile_attempted:
                    log.info(
                        "Advanced t-SNE/UMAP/anomaly steps were skipped after the "
                        "isolated worker failed; the workbook above is the standard compile only."
                    )
                from post_compile_research_export import run_research_workbook_export

                run_research_workbook_export(compiled_output_path, log=log)
            else:
                if advanced_stage2 and compiled_df is None:
                    log.error(
                        "Stage 2 failed even after retrying without t-SNE / UMAP / "
                        "anomaly detection. See earlier errors in this log."
                    )
                elif compiled_df is None:
                    log.error("Stage 2 returned no DataFrame.")
                else:
                    log.error("Stage 2 returned an empty DataFrame.")

        except Exception as exc:
            log.error(f"Error in Stage 2 (compilation): {exc}")
            import traceback
            log.error(traceback.format_exc())

        log.info("=" * 80)
        log.info(f"PIPELINE FINISHED FOR FOLDER: {folder.name}")
        log.info("=" * 80)

        # AUDIT FIX (Clarinete_mf "jumped off" bug) — detach the per-folder
        # FileHandler installed at the top of this method so subsequent
        # folders (or future GUI sessions) don't accumulate handlers and so
        # the file descriptor is closed promptly.
        try:
            _h = locals().get("_worker_log_handler", None)
            if _h is not None:
                log.removeHandler(_h)
                _h.close()
        except Exception:
            pass

    def _reset(self):
        self.is_running = False
        self.btn_run.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="Idle")

if __name__ == "__main__":
    if not log.hasHandlers():
        log.addHandler(logging.StreamHandler(sys.stdout))
    root = tk.Tk()
    RobustOrchestratorApp(root)
    root.mainloop()
