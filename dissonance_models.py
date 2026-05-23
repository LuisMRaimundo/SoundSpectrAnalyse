# dissonance_models.py — dissonance metric implementations

"""
Dissonance models for audio spectral analysis.
Visual comparison helpers used by the orchestrator GUI.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import math
import logging
from functools import lru_cache
import os
from pathlib import Path
import scipy.signal

# Logging configuration
logger = logging.getLogger(__name__)

# Global constants
DEFAULT_PLOT_DPI = 300
CENTS_PER_OCTAVE = 1200
HK_G_TABLE_PROVENANCE = (
    "Hutchinson & Knopoff (1978), Interface 7(1):1-29, Fig. 1. "
    "Re-digitised with WebPlotDigitizer; calibration parameters "
    "and source archived in data/hk1978_g_table_wpd_project.tar "
    "and data/hk1978_g_table_provenance.txt."
)
_HK_G_TABLE_CSV = Path(__file__).resolve().parent / "data" / "hk1978_g_table.csv"


def _load_hk_default_g_table() -> list[tuple[float, float]]:
    """Load the default Hutchinson-Knopoff g(y) table from CSV."""
    skiprows = 0
    for line in _HK_G_TABLE_CSV.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            skiprows += 1
            continue
        if stripped.startswith("#"):
            skiprows += 1
            continue
        if stripped.lower().replace(" ", "") == "y,g":
            skiprows += 1
        break

    table = np.loadtxt(
        _HK_G_TABLE_CSV,
        delimiter=",",
        comments="#",
        dtype=float,
        skiprows=skiprows,
    )
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] != 2:
        raise ValueError(f"Invalid HK g-table shape: {table.shape}")
    return [(float(y), float(g)) for y, g in table]

# -----------------------------------------------------------------------------
# BASE CLASS
# -----------------------------------------------------------------------------

class DissonanceModel(ABC):
    """Abstract base class for dissonance models."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        logger.debug("Dissonance model initialized: %s", name)
    
    @abstractmethod
    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        """Compute dissonance between two pure tones (pairwise)."""
        pass
    
    def total_dissonance(self, partials1: List[Tuple[float, float]], 
                        partials2: List[Tuple[float, float]]) -> float:
        """Compute total dissonance (pairwise summation)."""
        if not partials1 or not partials2:
            return 0.0
        
        try:
            total_diss = 0.0
            for f1, a1 in partials1:
                for f2, a2 in partials2:
                    total_diss += self.pure_tones_dissonance(f1, f2, a1, a2)
            return total_diss
        except Exception as e:
            logger.error(f"Error computing total dissonance: {e}")
            raise
    
    def same_timbre_dissonance(self, base_partials: List[Tuple[float, float]], 
                              interval: float) -> float:
        """Compute dissonance of a timbre shifted by an interval."""
        if not base_partials: 
            return 0.0
        if interval <= 0:
            raise ValueError(f"Interval must be positive: {interval}")
        
        # Default implementation for pairwise models
        shifted_partials = [(f * interval, a) for f, a in base_partials]
        return self.total_dissonance(base_partials, shifted_partials)
    
    def calculate_dissonance_curve(self, partials: List[Tuple[float, float]], 
                                  min_interval: float = 1.0,
                                  max_interval: float = 2.0,
                                  num_points: int = 100) -> Dict[float, float]:
        """Compute the dissonance curve for a timbre across an interval span."""
        if not partials: return {}
        intervals = np.linspace(min_interval, max_interval, num_points)
        curve = {}
        for interval in intervals:
            curve[interval] = self.same_timbre_dissonance(partials, interval)
        return curve
    
    def find_local_minima(self, curve: Dict[float, float], sensitivity: float = 0.01) -> List[float]:
        """Find local minima in the curve (consonances)."""
        if not curve: return []
        intervals = sorted(list(curve.keys()))
        minima = []
        for i in range(1, len(intervals) - 1):
            interval = intervals[i]
            val = curve[interval]
            if (val < curve[intervals[i-1]] and 
                val < curve[intervals[i+1]] and
                val < curve[intervals[i-1]] - sensitivity):
                minima.append(interval)
        return minima

    def visualize_dissonance_curve(self, curve: Dict[float, float], 
                                 scale: Optional[List[float]] = None,
                                 title: Optional[str] = None,
                                 save_file: Optional[str] = None,
                                 show_cents: bool = True,
                                 highlight_minima: bool = True,
                                 dpi: int = DEFAULT_PLOT_DPI):
        """Plot the dissonance curve."""
        if not curve: return
        intervals = sorted(list(curve.keys()))
        vals = [curve[i] for i in intervals]
        
        plt.figure(figsize=(12, 6))
        plt.plot(intervals, vals, 'b-', linewidth=2)
        
        if highlight_minima and not scale:
            minima = self.find_local_minima(curve)
            if minima:
                my = [curve[m] for m in minima]
                plt.plot(minima, my, 'go', markersize=6)
                
        if scale:
            sy = [curve.get(r, 0) for r in scale]
            plt.plot(scale, sy, 'ro', markersize=8)
            
        plt.title(title or f"{self.name} Dissonance Curve")
        plt.xlabel('Frequency Ratio')
        plt.ylabel('Dissonance')
        plt.grid(True, alpha=0.3)
        
        if show_cents:
            ax1 = plt.gca()
            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
            ax2.set_xticks([2**(c/1200) for c in ticks])
            ax2.set_xticklabels([f"{c}¢" for c in ticks])
            ax2.set_xlabel('Cents')
            
        if save_file:
            plt.savefig(save_file, dpi=dpi, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            plt.close()

    @staticmethod
    def _pairwise_arrays(
        freqs: np.ndarray,
        amps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build upper-triangular frequency/amplitude pair arrays."""
        if freqs.size < 2 or amps.size < 2:
            empty = np.array([], dtype=float)
            return empty, empty, empty, empty
        i_idx, j_idx = np.triu_indices(freqs.size, k=1)
        return freqs[i_idx], freqs[j_idx], amps[i_idx], amps[j_idx]

    def _pairwise_dissonance_values(
        self,
        fi: np.ndarray,
        fj: np.ndarray,
        ai: np.ndarray,
        aj: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate pairwise dissonance values with narrow exception handling."""
        values = np.zeros(fi.size, dtype=float)
        valid_mask = np.ones(fi.size, dtype=bool)
        for idx, (f1, f2, a1, a2) in enumerate(zip(fi, fj, ai, aj)):
            try:
                values[idx] = self.pure_tones_dissonance(f1, f2, a1, a2)
            except (ValueError, TypeError, ZeroDivisionError, FloatingPointError) as exc:
                valid_mask[idx] = False
                values[idx] = 0.0
                logger.warning(
                    "Skipping invalid dissonance pair (%s, %s): %s",
                    f1,
                    f2,
                    exc,
                )
        return values, valid_mask

    def _dissonance_total_and_pairs(self, df: pd.DataFrame) -> tuple[float, int]:
        """Compute raw pairwise dissonance sum."""
        if df is None or df.empty: return 0.0, 0
        
        # Safe data preparation
        if "Frequency (Hz)" not in df.columns or ("Amplitude" not in df.columns and "Magnitude (dB)" not in df.columns):
            return 0.0, 0
            
        df = df.copy()
        if "Amplitude" not in df.columns:
            df["Amplitude"] = 10 ** (df["Magnitude (dB)"] / 20)
            
        df = df[df["Frequency (Hz)"] > 0]
        freqs = df["Frequency (Hz)"].to_numpy(dtype=float)
        amps = df["Amplitude"].to_numpy(dtype=float)
        
        n = len(freqs)
        if n < 2: return 0.0, 0
        
        fi, fj, ai, aj = self._pairwise_arrays(freqs, amps)
        if fi.size == 0:
            return 0.0, 0
        pair_values, valid_mask = self._pairwise_dissonance_values(fi, fj, ai, aj)
        total = float(np.sum(pair_values))
        n_pairs = int(np.sum(valid_mask))
        return total, n_pairs


    def _dissonance_total_pairs_and_minamp(
        self,
        df: pd.DataFrame,
        *,
        apply_amp_compensation: bool = False,
        win_length: int | None = None,
    ) -> tuple[float, int, float]:
        """
        Retorna:
          total       : Σ d_ij
          n_pairs     : número de pares
          sum_minamp  : Σ min(a_i,a_j) (para normalização robusta)
        """
        if df is None or df.empty:
            return 0.0, 0, 0.0

        if "Frequency (Hz)" not in df.columns:
            return 0.0, 0, 0.0

        dfx = df.copy()

        # Linear amplitude
        if "Amplitude" not in dfx.columns:
            if "Magnitude (dB)" not in dfx.columns:
                return 0.0, 0, 0.0
            dfx["Amplitude"] = 10.0 ** (dfx["Magnitude (dB)"].astype(float) / 20.0)

        dfx = dfx[(dfx["Frequency (Hz)"] > 0) & (dfx["Amplitude"] > 0)]
        freqs = dfx["Frequency (Hz)"].to_numpy(dtype=float)
        amps = dfx["Amplitude"].to_numpy(dtype=float)

        n = len(freqs)
        if n < 2:
            return 0.0, 0, 0.0

        # Optional compensation (legacy proc_audio): amps *= 2/N
        if apply_amp_compensation:
            N = int(win_length or getattr(self, "win_length", 0) or getattr(self, "n_fft", 0) or 0)
            if N > 0:
                amps = amps * (2.0 / N)

        fi, fj, ai, aj = self._pairwise_arrays(freqs, amps)
        if fi.size == 0:
            return 0.0, 0, 0.0
        pair_values, valid_mask = self._pairwise_dissonance_values(fi, fj, ai, aj)
        min_amplitudes = np.minimum(ai, aj)
        total = float(np.sum(pair_values))
        n_pairs = int(np.sum(valid_mask))
        sum_minamp = float(np.sum(min_amplitudes[valid_mask]))

        return total, n_pairs, sum_minamp


    def calculate_dissonance_metric(
        self,
        df: pd.DataFrame,
        *,
        metric_mode: str = "mean_pair_scaled",
        metric_scale: float = 10.0,
        # compensação opcional para proc_audio antigo (amplitudes ~N/2 inflacionadas)
        apply_amp_compensation: bool = False,
        win_length: int | None = None,
    ) -> float:
        """
        Calcula uma métrica Sethares a partir de um DF (freq, amplitude).

        metric_mode:
          - "sum"              : soma bruta Σ d_ij
          - "mean_pair"        : média por par Σ d_ij / n_pairs
          - "mean_pair_scaled" : média por par × metric_scale (legado: ~0–10)
          - "minamp_norm"      : Σ d_ij / Σ min(a_i,a_j)  (normalização robusta à escala)
        """
        total, n_pairs, sum_minamp = self._dissonance_total_pairs_and_minamp(
            df,
            apply_amp_compensation=apply_amp_compensation,
            win_length=win_length,
        )
        if n_pairs <= 0:
            return 0.0

        mode = (metric_mode or "mean_pair_scaled").strip().lower()

        if mode == "sum":
            return float(total)

        if mode == "mean_pair":
            return float(total / n_pairs)

        if mode == "minamp_norm":
            return float(total / sum_minamp) if sum_minamp > 0 else 0.0

        # Default legacy behavior
        return float((total / n_pairs) * float(metric_scale))


    def generate_scale(
        self,
        partials: List[Tuple[float, float]],
        min_interval: float = 1.0,
        max_interval: float = 2.0,
        num_points: int = 100,
        include_endpoints: bool = True,
        endpoint_eps: float = 1e-12,
    ) -> List[float]:
        """
        Gera uma escala baseada nos mínimos locais da curva de dissonância.

        - include_endpoints=True garante inclusão de min_interval e max_interval.
        - endpoint_eps evita problemas de comparação float.
        """
        curve = self.calculate_dissonance_curve(partials, min_interval, max_interval, num_points)
        minima = list(self.find_local_minima(curve))  # presume que devolve lista de intervalos (floats)

        if include_endpoints:
            if all(abs(m - min_interval) > endpoint_eps for m in minima):
                minima.append(min_interval)
            if all(abs(m - max_interval) > endpoint_eps for m in minima):
                minima.append(max_interval)

        # limpar duplicados "quase iguais" e ordenar
        minima_sorted = sorted(minima)
        cleaned: list[float] = []
        for m in minima_sorted:
            if not cleaned or abs(m - cleaned[-1]) > 1e-9:
                cleaned.append(float(m))

        return cleaned



# -----------------------------------------------------------------------------
# IMPLEMENTAÇÕES DOS MODELOS
# -----------------------------------------------------------------------------

class SetharesDissonance(DissonanceModel):
    """Sethares (TTSS, 2nd ed., 2005) robust implementation.

    Elementar (dois parciais):
        d(f1,f2,a1,a2) = min(a1,a2) * gain * (exp(-b1*y) - exp(-b2*y))
        y = s(f1) * (f2 - f1)
        s(f1) = x_star / (s1*f1 + s2)

    Dissonance curve for timbre F at ratio 'interval':
      - mode='cross': sum only cross-interactions F vs interval·F (legacy module behaviour)
      - mode='full' : sum over union {F} ∪ {interval·F} (book form)

    Note: 'gain' is a global rescale (does not alter curve shape).
          For compatibility with older scaling (C1=5, C2=-5), use gain=5.0.
    """

    def __init__(
        self,
        *,
        b1: float = 3.5,      # Sethares (2005, 2nd ed., Eq. 3.8), spectral term coefficient b1.
        b2: float = 5.75,     # Sethares (2005, 2nd ed., Eq. 3.8), spectral term coefficient b2.
        x_star: float = 0.24, # Sethares (2005, 2nd ed., Eq. 3.9), critical-band scaling x*.
        s1: float = 0.0207,   # Sethares (2005, 2nd ed., Eq. 3.9), denominator slope s1.
        s2: float = 18.96,    # Sethares (2005, 2nd ed., Eq. 3.9), denominator intercept s2.
        gain: float = 1.0,
        curve_mode: str = "full",          # 'full' (book) or 'cross' (legacy)
        subtract_intrinsic: bool = False,  # if True, return full - intrinsic terms
        metric_mode: str = "mean_pair_scaled",  # 'sum'|'mean_pair'|'mean_pair_scaled'|'minamp_norm'
        metric_scale: float = 10.0,
    ):
        super().__init__("Sethares-Revised", "Based on Plomp-Levelt curves (Sethares, 2005)")

        self.b1 = float(b1)
        self.b2 = float(b2)
        self.x_star = float(x_star)
        self.s1 = float(s1)
        self.s2 = float(s2)
        self.gain = float(gain)

        self.curve_mode = str(curve_mode).strip().lower()
        self.subtract_intrinsic = bool(subtract_intrinsic)

        self.metric_mode = str(metric_mode).strip().lower()
        self.metric_scale = float(metric_scale)

        # Legacy attributes (unused internally; retained for debug compatibility).
        # Older expression: min(a1,a2) * (5*exp(-3.51*x) - 5*exp(-5.75*x))
        self.C1, self.C2, self.A1, self.A2 = 1.0, -1.0, -self.b1, -self.b2
        self.d_star = self.x_star  # legacy naming

    def _s(self, f1: float) -> float:
        f1 = max(float(f1), 1e-12)
        return self.x_star / (self.s1 * f1 + self.s2)

    def pure_tones_dissonance(self, f1, f2, a1, a2) -> float:
        f1 = float(f1); f2 = float(f2)
        a1 = float(a1); a2 = float(a2)

        if f1 <= 0.0 or f2 <= 0.0 or a1 <= 0.0 or a2 <= 0.0:
            return 0.0

        if f1 > f2:
            f1, f2, a1, a2 = f2, f1, a2, a1

        y = self._s(f1) * (f2 - f1)
        d = min(a1, a2) * self.gain * (np.exp(-self.b1 * y) - np.exp(-self.b2 * y))

        # Numerical robustness
        return float(d) if d > 0.0 else 0.0

    def _pairwise_sum(self, partials: List[Tuple[float, float]]) -> float:
        if not partials:
            return 0.0
        ps = [(float(f), float(a)) for f, a in partials if f > 0 and a > 0]
        if len(ps) < 2:
            return 0.0
        ps.sort(key=lambda x: x[0])

        freqs = np.asarray([p[0] for p in ps], dtype=float)
        amps = np.asarray([p[1] for p in ps], dtype=float)
        i_idx, j_idx = np.triu_indices(freqs.size, k=1)
        f1 = freqs[i_idx]
        f2 = freqs[j_idx]
        a1 = amps[i_idx]
        a2 = amps[j_idx]
        s = self.x_star / (self.s1 * np.maximum(f1, 1e-12) + self.s2)
        y = s * (f2 - f1)
        pair_values = np.minimum(a1, a2) * self.gain * (
            np.exp(-self.b1 * y) - np.exp(-self.b2 * y)
        )
        pair_values = np.where(pair_values > 0.0, pair_values, 0.0)
        return float(np.sum(pair_values))

    def same_timbre_dissonance(self, base_partials: List[Tuple[float, float]], interval: float) -> float:
        if not base_partials:
            return 0.0
        if interval <= 0:
            raise ValueError(f"Interval must be positive: {interval}")

        shifted = [(f * interval, a) for (f, a) in base_partials]

        if self.curve_mode == "cross":
            # Legacy module behavior: cross-interactions only
            return float(self.total_dissonance(base_partials, shifted))

        # Book form: sum over {F} ∪ {interval·F}
        full = self._pairwise_sum(base_partials + shifted)

        if self.subtract_intrinsic:
            full -= self._pairwise_sum(base_partials)
            full -= self._pairwise_sum(shifted)
            if full < 0.0:
                full = 0.0

        return float(full)

    def calculate_dissonance_metric(self, df: pd.DataFrame) -> float:
        """Per-note metric (for Excel export).

        Modos:
          - 'sum'              : Σ d_ij (soma bruta)
          - 'mean_pair'        : média por par = Σ d_ij / n_pairs
          - 'mean_pair_scaled' : (Σ d_ij / n_pairs) * metric_scale [module-default compatible]
          - 'minamp_norm'      : Σ d_ij / Σ min(a_i,a_j) (robust to global amplitude scale)
        """
        if df is None or df.empty:
            return 0.0

        if "Frequency (Hz)" not in df.columns or ("Amplitude" not in df.columns and "Magnitude (dB)" not in df.columns):
            return 0.0

        dfx = df.copy()
        if "Amplitude" not in dfx.columns:
            dfx["Amplitude"] = 10 ** (dfx["Magnitude (dB)"] / 20)

        dfx = dfx[(dfx["Frequency (Hz)"] > 0) & (dfx["Amplitude"] > 0)]
        freqs = dfx["Frequency (Hz)"].to_numpy(dtype=float)
        amps = dfx["Amplitude"].to_numpy(dtype=float)

        n = len(freqs)
        if n < 2:
            return 0.0

        total = 0.0
        n_pairs = 0
        sum_minamp = 0.0

        for i in range(n - 1):
            for j in range(i + 1, n):
                a_min = amps[i] if amps[i] < amps[j] else amps[j]
                sum_minamp += a_min
                total += self.pure_tones_dissonance(freqs[i], freqs[j], amps[i], amps[j])
                n_pairs += 1

        if n_pairs <= 0:
            return 0.0

        if self.metric_mode == "sum":
            return float(total)

        if self.metric_mode == "mean_pair":
            return float(total / n_pairs)

        if self.metric_mode == "minamp_norm":
            return float(total / sum_minamp) if sum_minamp > 0 else 0.0

        # Default module-compatible behavior (≈0–10)
        return float((total / n_pairs) * self.metric_scale)

class HutchinsonKnopoffDissonance(DissonanceModel):
    """
    Hutchinson & Knopoff (1978), following eqs. (1)-(3).

    CRITICAL NOTE:
    - g(y) is not provided as an analytical formula; it is supplied as a lookup
      curve (Figure 1), so this implementation requires a g_table.
    """

    DEFAULT_G_TABLE = _load_hk_default_g_table()

    def __init__(self, g_table=None):
        super().__init__("Hutchinson-Knopoff", "CBW( f̄ ) and g(y) lookup (1978)")
        # g_table: list of points [(y0,g0), (y1,g1), ...] typically spanning [0, 1.2]
        # If omitted, use the default CSV table.
        self.g_table = sorted(g_table) if g_table else sorted(self.DEFAULT_G_TABLE)

    @staticmethod
    def cbw(f_bar: float) -> float:
        # CBW = 1.72 * (f̄)^0.65 (Fig. 2; empirical fit)
        return 1.72 * (f_bar ** 0.65)

    def g(self, y: float) -> float:
        # y = |fi - fj| / CBW(f̄)
        if y <= 0.0 or y > 1.2:
            return 0.0

        if not self.g_table:
            # This should not happen if DEFAULT_G_TABLE is used, but handle gracefully
            logger.warning("g_table not provided, using zero roughness")
            return 0.0

        ys = np.array([p[0] for p in self.g_table], dtype=float)
        gs = np.array([p[1] for p in self.g_table], dtype=float)

        # Linear interpolation (table look-up).
        # Assumes the table covers relevant y; out-of-range is handled above.
        return float(np.interp(y, ys, gs))

    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        # Eq. (1) with normalization N = A1^2 + A2^2
        denom = (a1 * a1) + (a2 * a2)
        if denom <= 0.0:
            return 0.0

        f_bar = 0.5 * (f1 + f2)
        cb = self.cbw(f_bar)
        if cb <= 0.0:
            return 0.0

        y = abs(f1 - f2) / cb
        return (a1 * a2 * self.g(y)) / denom

    def total_dissonance(self, partials1, partials2) -> float:
        """
        Implements eq. (3) over all components of the composite sound:
            D = [ (1/2) Σ_i Σ_j Ai Aj g_ij ] / [ Σ_i Ai^2 ]

        Equivalent implementation (avoids double counting):
            numerator = Σ_{i<j} Ai Aj g_ij   (assuming g_ii=0)
            D = numerator / Σ_i Ai^2
        """
        partials = list(partials1 or []) + list(partials2 or [])
        if not partials:
            return 0.0

        denom = sum((a * a) for _, a in partials)
        if denom <= 0.0:
            return 0.0

        num = 0.0
        n = len(partials)
        for i in range(n):
            fi, ai = partials[i]
            for j in range(i + 1, n):
                fj, aj = partials[j]

                f_bar = 0.5 * (fi + fj)
                cb = self.cbw(f_bar)
                if cb <= 0.0:
                    continue

                y = abs(fi - fj) / cb
                g_ij = self.g(y)
                num += (ai * aj * g_ij)

        return num / denom


class VassilakisDissonance(DissonanceModel):
    def __init__(self):
        super().__init__("Vassilakis", "Eq. (6.23): AF-degree + SPL + Sethares spectral term")

        # Vassilakis (2001, eqs. 6.20-6.23), shared spectral coefficients with Sethares term.
        self.b1 = 3.5
        self.b2 = 5.75
        self.x_star = 0.24
        self.s1 = 0.0207
        self.s2 = 18.96

        # Vassilakis (2001, eqs. 6.20-6.23; see also Vassilakis & Fitz, 2007).
        self.af_exp = 3.11     # AF_degree exponent
        self.spl_exp = 0.1     # SPL exponent (1/10)
        self.pair_factor = 0.5 # Eq. (6.22) pair factor

    def _s(self, f_low: float) -> float:
        # s = x* / (s1*f1 + s2)
        return self.x_star / (self.s1 * f_low + self.s2)

    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        # Enforce f1 < f2 as required by the definition of s(f1)
        if f1 > f2:
            f1, f2 = f2, f1

        # Basic domain safety (not specified by the equation, but prevents invalid math)
        if f1 <= 0.0 or f2 <= 0.0 or a1 <= 0.0 or a2 <= 0.0:
            return 0.0

        A1 = max(a1, a2)  # text assumes A1 >= A2
        A2 = min(a1, a2)

        # AFdegree = 2A2 / (A1 + A2)
        af_degree = (2.0 * A2) / (A1 + A2)

        # spectral term: exp(-b1*s*(f2-f1)) - exp(-b2*s*(f2-f1))
        s = self._s(f1)
        x = s * (f2 - f1)
        spectral = np.exp(-self.b1 * x) - np.exp(-self.b2 * x)

        # Eq. (6.23)
        R = (A1 * A2) ** self.spl_exp
        R *= self.pair_factor * (af_degree ** self.af_exp) * spectral
        return float(R)


# -----------------------------------------------------------------------------
# UTILITY AND COMPARISON FUNCTIONS
# -----------------------------------------------------------------------------

_MODELS = {
    "sethares": SetharesDissonance,
    "hutchinson-knopoff": HutchinsonKnopoffDissonance,
    "vassilakis": VassilakisDissonance,
}

def get_dissonance_model(name: str, *, allow_harmonicity: bool = True) -> DissonanceModel:
    key = name.strip().lower()
    if key in _MODELS: return _MODELS[key]()
    raise ValueError(f"Unknown model: {name}")

def list_available_models(*, include_harmonicity: bool = True) -> List[str]:
    return list(_MODELS.keys())

def calculate_all_dissonance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    results = {}
    for name in _MODELS:
        try:
            model = get_dissonance_model(name)
            results[name] = model.calculate_dissonance_metric(df)
        except Exception as e:
            logger.error("Error in %s: %s", name, e)
            results[name] = 0.0
    return results

def compare_dissonance_models(partials: List[Tuple[float, float]],
                             min_interval: float = 1.0,
                             max_interval: float = 2.0,
                             num_points: int = 100,
                             save_file: Optional[str] = None,
                             models_to_include: Optional[List[str]] = None,
                             normalize_curves: bool = True,
                             show_minima: bool = True,
                             add_cent_axis: bool = True,
                             dpi: int = DEFAULT_PLOT_DPI) -> Dict[str, Dict]:
    """Compare dissonance curves across different models."""
    if not partials: return {}
    
    models = [get_dissonance_model(name) for name in (models_to_include or list_available_models())]
    curves = {}
    
    for model in models:
        curves[model.name] = model.calculate_dissonance_curve(partials, min_interval, max_interval, num_points)
        
    plt.figure(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for i, (model_name, curve) in enumerate(curves.items()):
        intervals = sorted(list(curve.keys()))
        vals = [curve[inter] for inter in intervals]
        
        if normalize_curves:
            v_min, v_max = min(vals), max(vals)
            if v_max > v_min:
                vals = [(v - v_min) / (v_max - v_min) for v in vals]
        
        plt.plot(intervals, vals, color=colors[i], label=model_name, linewidth=2)
        
        if show_minima:
            model = models[i]
            minima = model.find_local_minima(curve)
            if minima:
                if normalize_curves and v_max > v_min:
                    my = [(curve[m] - v_min) / (v_max - v_min) for m in minima]
                else:
                    my = [curve[m] for m in minima]
                plt.plot(minima, my, 'o', color=colors[i], markersize=6)

    plt.title("Comparison of Dissonance Models")
    plt.xlabel('Frequency Ratio')
    plt.ylabel('Normalized Dissonance' if normalize_curves else 'Dissonance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if add_cent_axis:
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
        ax2.set_xticks([2**(c/1200) for c in ticks])
        ax2.set_xticklabels([f"{c}¢" for c in ticks])
        ax2.set_xlabel('Cents')

    if save_file:
        plt.savefig(save_file, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()
        
    return curves

def analyze_real_timbre(df: pd.DataFrame, 
                       note_name: str = "",
                       include_models: Optional[List[str]] = None,
                       save_directory: Optional[str] = None) -> Dict[str, Any]:
    """Analyze a real timbre and save metrics/plots."""
    if df is None or df.empty or "Frequency (Hz)" not in df.columns: return {}
    
    amps = df["Amplitude"] if "Amplitude" in df.columns else 10**(df["Magnitude (dB)"]/20)
    partials = list(zip(df["Frequency (Hz)"], amps))
    
    models = [get_dissonance_model(name) for name in (include_models or list_available_models())]
    if save_directory: os.makedirs(save_directory, exist_ok=True)
    
    results = {"metrics": {}, "curves": {}, "scales": {}}
    
    for model in models:
        metric = model.calculate_dissonance_metric(df)
        results["metrics"][model.name] = metric
        
        curve = model.calculate_dissonance_curve(partials, 1.0, 2.0, 200)
        results["curves"][model.name] = curve
        
        scale = model.find_local_minima(curve)
        if 1.0 not in scale: scale.insert(0, 1.0)
        if 2.0 not in scale: scale.append(2.0)
        results["scales"][model.name] = sorted(scale)
        
        if save_directory:
            title = f"{model.name} Dissonance Curve - {note_name}"
            path = os.path.join(save_directory, f"{model.name.lower()}_dissonance_curve.png")
            model.visualize_dissonance_curve(curve, scale, title=title, save_file=path)
            
    if save_directory and len(models) > 1:
        path = os.path.join(save_directory, "dissonance_comparison.png")
        compare_dissonance_models(partials, save_file=path, models_to_include=[m.name for m in models])
        
        # Save metrics
        m_df = pd.DataFrame({"Model": list(results["metrics"].keys()), "Dissonance": list(results["metrics"].values())})
        m_df.to_csv(os.path.join(save_directory, "dissonance_metrics.csv"), index=False)

    return results

# Compatibility exports
__all__ = [
    'DissonanceModel',
    'SetharesDissonance',
    'HutchinsonKnopoffDissonance',
    'VassilakisDissonance',
    'HK_G_TABLE_PROVENANCE',
    'get_dissonance_model',
    'list_available_models',
    'compare_dissonance_models',
    'calculate_all_dissonance_metrics',
    'analyze_real_timbre'
]