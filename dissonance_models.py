# dissonance_models.py - Versão Completa e Corrigida

"""
Módulo de modelos de dissonância para análise de áudio.
funções de comparação visual exigidas pelo orquestrador.
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
import scipy.signal

# Configuração de logging
logger = logging.getLogger(__name__)

# Constantes globais
DEFAULT_PLOT_DPI = 300
CENTS_PER_OCTAVE = 1200

# -----------------------------------------------------------------------------
# CLASSE BASE
# -----------------------------------------------------------------------------

class DissonanceModel(ABC):
    """Classe base abstrata para modelos de dissonância."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        logger.debug(f"Modelo de dissonância inicializado: {name}")
    
    @abstractmethod
    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        """Calcula a dissonância entre dois tons puros (pairwise)."""
        pass
    
    def total_dissonance(self, partials1: List[Tuple[float, float]], 
                        partials2: List[Tuple[float, float]]) -> float:
        """Calcula dissonância total (pairwise summation)."""
        if not partials1 or not partials2:
            return 0.0
        
        try:
            total_diss = 0.0
            for f1, a1 in partials1:
                for f2, a2 in partials2:
                    total_diss += self.pure_tones_dissonance(f1, f2, a1, a2)
            return total_diss
        except Exception as e:
            logger.error(f"Erro ao calcular dissonância total: {e}")
            raise
    
    def same_timbre_dissonance(self, base_partials: List[Tuple[float, float]], 
                              interval: float) -> float:
        """Calcula dissonância de um timbre deslocado por um intervalo."""
        if not base_partials: 
            return 0.0
        if interval <= 0:
            raise ValueError(f"Intervalo deve ser positivo: {interval}")
        
        # Implementação padrão para modelos pairwise
        shifted_partials = [(f * interval, a) for f, a in base_partials]
        return self.total_dissonance(base_partials, shifted_partials)
    
    def calculate_dissonance_curve(self, partials: List[Tuple[float, float]], 
                                  min_interval: float = 1.0,
                                  max_interval: float = 2.0,
                                  num_points: int = 100) -> Dict[float, float]:
        """Calcula a curva de dissonância para um timbre em um intervalo."""
        if not partials: return {}
        intervals = np.linspace(min_interval, max_interval, num_points)
        curve = {}
        for interval in intervals:
            curve[interval] = self.same_timbre_dissonance(partials, interval)
        return curve
    
    def find_local_minima(self, curve: Dict[float, float], sensitivity: float = 0.01) -> List[float]:
        """Encontra mínimos locais na curva (consonâncias)."""
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
        """Plota a curva de dissonância."""
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

    def _dissonance_total_and_pairs(self, df: pd.DataFrame) -> tuple[float, int]:
        """Calcula soma bruta de dissonância pairwise."""
        if df is None or df.empty: return 0.0, 0
        
        # Preparação segura dos dados
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
        
        total = 0.0
        n_pairs = 0
        # Loop otimizado (mas ainda Python puro)
        for i in range(n - 1):
            for j in range(i + 1, n):
                try:
                    total += self.pure_tones_dissonance(freqs[i], freqs[j], amps[i], amps[j])
                    n_pairs += 1
                except: continue
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

        # amplitude linear
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

        # compensação opcional (proc_audio antigo): amps *= 2/N
        if apply_amp_compensation:
            N = int(win_length or getattr(self, "win_length", 0) or getattr(self, "n_fft", 0) or 0)
            if N > 0:
                amps = amps * (2.0 / N)

        total = 0.0
        n_pairs = 0
        sum_minamp = 0.0

        for i in range(n - 1):
            f1 = freqs[i]; a1 = amps[i]
            for j in range(i + 1, n):
                f2 = freqs[j]; a2 = amps[j]
                a_min = a1 if a1 < a2 else a2
                sum_minamp += a_min
                total += self.pure_tones_dissonance(f1, f2, a1, a2)
                n_pairs += 1

        return float(total), int(n_pairs), float(sum_minamp)


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

        # default/legado
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
    """Sethares (TTSS, 2.ª ed., 2005) — implementação robusta.

    Elementar (dois parciais):
        d(f1,f2,a1,a2) = min(a1,a2) * gain * (exp(-b1*y) - exp(-b2*y))
        y = s(f1) * (f2 - f1)
        s(f1) = x_star / (s1*f1 + s2)

    Curva de dissonância para um timbre F num intervalo 'interval' (razão):
      - mode='cross': soma apenas interacções F vs interval·F (comportamento antigo do módulo)
      - mode='full' : soma sobre o conjunto {F} ∪ {interval·F} (forma do livro)

    Nota: 'gain' é um reescale global (não altera a forma da curva).
          Para compatibilidade com a versão antiga (C1=5, C2=-5), use gain=5.0.
    """

    def __init__(
        self,
        *,
        b1: float = 3.5,
        b2: float = 5.75,
        x_star: float = 0.24,
        s1: float = 0.0207,
        s2: float = 18.96,
        gain: float = 1.0,
        curve_mode: str = "full",          # 'full' (livro) ou 'cross' (legado)
        subtract_intrinsic: bool = False,  # se True, devolve full - (intrínsecas)
        metric_mode: str = "mean_pair_scaled",  # 'sum'|'mean_pair'|'mean_pair_scaled'|'minamp_norm'
        metric_scale: float = 10.0,
    ):
        super().__init__("Sethares-Revised", "Baseado em curvas Plomp-Levelt (Sethares, 2005)")

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

        # Atributos "legados" (não usados internamente; mantidos para evitar confusão em debug)
        # Versão antiga: min(a1,a2) * (5*exp(-3.51*x) - 5*exp(-5.75*x))
        self.C1, self.C2, self.A1, self.A2 = 1.0, -1.0, -self.b1, -self.b2
        self.d_star = self.x_star  # naming legado

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

        # robustez numérica
        return float(d) if d > 0.0 else 0.0

    def _pairwise_sum(self, partials: List[Tuple[float, float]]) -> float:
        if not partials:
            return 0.0
        ps = [(float(f), float(a)) for f, a in partials if f > 0 and a > 0]
        if len(ps) < 2:
            return 0.0
        ps.sort(key=lambda x: x[0])

        total = 0.0
        for i in range(len(ps) - 1):
            f1, a1 = ps[i]
            for j in range(i + 1, len(ps)):
                f2, a2 = ps[j]
                total += self.pure_tones_dissonance(f1, f2, a1, a2)
        return float(total)

    def same_timbre_dissonance(self, base_partials: List[Tuple[float, float]], interval: float) -> float:
        if not base_partials:
            return 0.0
        if interval <= 0:
            raise ValueError(f"Intervalo deve ser positivo: {interval}")

        shifted = [(f * interval, a) for (f, a) in base_partials]

        if self.curve_mode == "cross":
            # comportamento legado do módulo: apenas interacções cruzadas
            return float(self.total_dissonance(base_partials, shifted))

        # forma do livro: soma sobre {F} ∪ {interval·F}
        full = self._pairwise_sum(base_partials + shifted)

        if self.subtract_intrinsic:
            full -= self._pairwise_sum(base_partials)
            full -= self._pairwise_sum(shifted)
            if full < 0.0:
                full = 0.0

        return float(full)

    def calculate_dissonance_metric(self, df: pd.DataFrame) -> float:
        """Métrica por nota (para export para Excel).

        Modos:
          - 'sum'              : Σ d_ij (soma bruta)
          - 'mean_pair'        : média por par = Σ d_ij / n_pairs
          - 'mean_pair_scaled' : (Σ d_ij / n_pairs) * metric_scale  [compatível com o default do módulo]
          - 'minamp_norm'      : Σ d_ij / Σ min(a_i,a_j)  (robusto a escala global de amplitude)
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

        # default: compatível com o módulo (≈0–10)
        return float((total / n_pairs) * self.metric_scale)

class HutchinsonKnopoffDissonance(DissonanceModel):
    def __init__(self):
        super().__init__("Hutchinson-Knopoff", "Baseado em largura de banda crítica (1978)")
        self.a, self.b = 3.5, 5.75

    def pure_tones_dissonance(self, f1, f2, a1, a2) -> float:
        if f1 > f2: f1, f2, a1, a2 = f2, f1, a2, a1
        cb = 1.2 * (f1 ** 0.76) # Critical bandwidth
        x = (f2 - f1) / cb
        return (a1 * a2) * (np.exp(-self.a * x) - np.exp(-self.b * x))

class VassilakisDissonance(DissonanceModel):
    def __init__(self):
        super().__init__("Vassilakis", "Baseado em flutuação espectral (2005)")
        self.alpha, self.beta = 3.11, 5.09
        self.gamma, self.delta = 0.5, 1.0

    def pure_tones_dissonance(self, f1, f2, a1, a2) -> float:
        if f1 > f2: f1, f2, a1, a2 = f2, f1, a2, a1
        f1 = max(f1, 20.0)
        freq_diff = f2 - f1
        amp_fluct = (2 * min(a1, a2)) / (a1 + a2)
        degree = amp_fluct**self.gamma
        roughness = np.exp(-self.alpha * freq_diff / f1) - np.exp(-self.beta * freq_diff / f1)
        return degree * (a1 * a2)**self.delta * roughness

class AuresZwickerDissonance(DissonanceModel):
    def __init__(self):
        super().__init__("Aures-Zwicker", "Rugosidade sensorial (Zwicker & Fastl)")
        self.k, self.gamma = 0.25, 1.25

    def pure_tones_dissonance(self, f1, f2, a1, a2) -> float:
        if f1 == f2: return 0.0
        f_mean = (f1 + f2) / 2
        cbw = 25 + 75 * (1 + 1.4 * (f_mean / 1000) ** 2) ** 0.69
        s = abs(f2 - f1) / cbw
        return self.k * (min(a1, a2) ** 0.6) * s * math.exp(-self.gamma * s)


# -----------------------------------------------------------------------------
# FUNÇÕES DE UTILIDADE E COMPARADORAS (RESTAURADAS)
# -----------------------------------------------------------------------------

_MODELS = {
    "sethares": SetharesDissonance,
    "hutchinson-knopoff": HutchinsonKnopoffDissonance,
    "vassilakis": VassilakisDissonance,
    "aures-zwicker": AuresZwickerDissonance,
}

def get_dissonance_model(name: str, *, allow_harmonicity: bool = True) -> DissonanceModel:
    key = name.strip().lower()
    if key in _MODELS: return _MODELS[key]()
    raise ValueError(f"Modelo desconhecido: {name}")

def list_available_models(*, include_harmonicity: bool = True) -> List[str]:
    return list(_MODELS.keys())

def calculate_all_dissonance_metrics(df: pd.DataFrame) -> Dict[str, float]:
    results = {}
    for name in _MODELS:
        try:
            model = get_dissonance_model(name)
            results[name] = model.calculate_dissonance_metric(df)
        except Exception as e:
            logger.error(f"Erro em {name}: {e}")
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
    """Compara curvas de dissonância de diferentes modelos."""
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
    """Analisa um timbre real e salva métricas/gráficos."""
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
        
        # Salva métricas
        m_df = pd.DataFrame({"Model": list(results["metrics"].keys()), "Dissonance": list(results["metrics"].values())})
        m_df.to_csv(os.path.join(save_directory, "dissonance_metrics.csv"), index=False)

    return results

# Exports para compatibilidade
__all__ = [
    'DissonanceModel',
    'SetharesDissonance',
    'HutchinsonKnopoffDissonance',
    'VassilakisDissonance',
    'AuresZwickerDissonance',
    'get_dissonance_model',
    'list_available_models',
    'compare_dissonance_models',
    'calculate_all_dissonance_metrics',
    'analyze_real_timbre'
]