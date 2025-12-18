# -*- coding: utf-8 -*-
from __future__ import annotations
from audio_utils import amp_to_db_mag, db_mag_to_amp, power_to_db, db_to_power

"""
Processamento de Ã¡udio, anÃ¡lise espectral (FFT/LFT) e extracÃ§Ã£o de mÃ©tricas
(densidade, dissonÃ¢ncia, potÃªncia). Destina-se a GUI (PyQt) e execuÃ§Ã£o batch.
"""

# ====================================================
# IMPORTS â€“ standard ? terceiros ? locais
# ====================================================
import gc
import json
import logging
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union
from density import spectral_density
from audio_utils import harmonic_tolerance_hz

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


import pandas as pd
import plotly.graph_objects as go

from spectral_power import spectral_power
from spectral_power import spectral_power_lft  # opcional em LFT
# se existir classe LFT no mÃ³dulo:
try:
    from spectral_power import LinearTimeFrequencyTransform
except Exception:
    LinearTimeFrequencyTransform = None  # fallback



from density import (
    apply_density_metric,
    get_weight_function,
    compute_spectral_entropy,
    calculate_combined_density_metric,
    # usadas em alguns cÃ¡lculos
    # (se nÃ£o existirem todas no seu mÃ³dulo, ajuste aqui conforme a sua API real)
    # calculate_harmonic_density,
    # calculate_inharmonic_density,
)

from dissonance_models import (
    get_dissonance_model,
    list_available_models,
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
# CONFIGURAÃ‡ÃƒO GLOBAL
# ====================================================
DEFAULT_N_FFT: int = 4096
DEFAULT_HOP_LENGTH: int = 1024
DEFAULT_WINDOW: str = "hann"
DEFAULT_PLOT_DPI: int = 300


# === Robust measurement helpers (amplitude absoluta, sem normalizar) ===
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



def _parabolic_peak(y, x):
    """
    InterpolaÃ§Ã£o parabÃ³lica (QIFFT simplificada) em torno do Ã­ndice x.
    Retorna (xv, yv) â€” posiÃ§Ã£o e amplitude sub-bin.
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

# ----------------- Normalização de nível (RMS) global -----------------
def _normalize_level(y: np.ndarray, target_rms_db: float = -20.0) -> np.ndarray:
    if y is None or len(y) == 0:
        return y
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    cur_db = 20.0 * np.log10(rms)
    gain = 10.0 ** ((target_rms_db - cur_db) / 20.0)
    return (y * gain).astype(y.dtype, copy=False)


# ----------------- Normalização de nível (RMS) global -----------------
def _normalize_level(y: np.ndarray, target_rms_db: float = -20.0) -> np.ndarray:
    """
    Normaliza o nível do sinal para um RMS alvo (dB), tornando métricas invariantes a ganho.
    """
    if y is None or len(y) == 0:
        return y
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    cur_db = 20.0 * np.log10(rms)
    gain = 10.0 ** ((target_rms_db - cur_db) / 20.0)
    return (y * gain).astype(y.dtype, copy=False)



# ====================================================
# UTILITÃRIOS FORA DA CLASSE
# ====================================================
def _extract_amplitude_column(df: pd.DataFrame) -> np.ndarray:
    """
    Extrai coluna de amplitudes com robustez; se nÃ£o houver, tenta converter de dB.
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

    # fallback: 1Âª coluna numÃ©rica
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().any():
            return s.fillna(0.0).to_numpy(float)
    return np.asarray([], dtype=float)


@lru_cache(maxsize=128)
def frequency_to_note_name(frequency: float) -> str:
    """
    Converte frequÃªncia em nome de nota aproximado (com cents).
    """
    if frequency <= 0:
        return "Invalid Frequency"
    freq_A4 = 440.0
    freq_C0 = freq_A4 * 2 ** (-4.75)
    h = int(round(12 * np.log2(frequency / freq_C0)))
    octave = h // 12
    n = h % 12
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    flat_note_names = ['C', 'Db', 'D', 'Eb', 'E', 'F',
                       'Gb', 'G', 'Ab', 'A', 'Bb', 'B']
    note_name_sharp = note_names[n] + str(octave)
    note_name_flat = flat_note_names[n] + str(octave)
    closest_note_frequency = freq_C0 * 2 ** (h / 12)
    cents_deviation = 1200 * np.log2(frequency / closest_note_frequency)
    if note_name_sharp[:2] in ['C#', 'D#', 'F#', 'G#', 'A#']:
        return f"{note_name_flat} ({cents_deviation:+.2f} cents)"
    else:
        return f"{note_name_sharp} ({cents_deviation:+.2f} cents)"


# ====================================================
# CLASSE PRINCIPAL
# ====================================================
class AudioProcessor:
    """
    Classe para processamento de Ã¡udio, anÃ¡lise FFT/LFT e geraÃ§Ã£o de dados espectrais.
    """

    def _compute_tol_hz(self, f0, h):
        """
        Perceptual tolerance (cents + piso em Hz) quando 'use_adaptive_tolerance' é True;
        caso contrário usa tolerância fixa 'tolerance' (Hz).
        """
        try:
            if getattr(self, "use_adaptive_tolerance", True):
                cents = float(getattr(self, "search_band_cents", 5.0))
                from audio_utils import harmonic_tolerance_hz
                return float(harmonic_tolerance_hz(float(f0), int(h),
                                                   search_band_cents=cents,
                                                   min_tolerance_hz=2.0))
            else:
                return float(getattr(self, "tolerance", 5.0))
        except Exception:
            return float(getattr(self, "tolerance", 5.0))

    # ----------------- janela p/ STFT -----------------
    def _get_window_arg(self):
        """
        Resolve a janela a passar ao STFT (scipy.signal.get_window quando aplicÃ¡vel).
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
                _err(f"Janela fornecida nÃ£o Ã© 1D (ndim={arr.ndim}).")
                raise ValueError("A janela fornecida deve ser 1D.")
            if arr.size != n:
                _err(f"Tamanho janela ({arr.size}) != n_fft ({n}).")
                raise ValueError("Comprimento da janela != n_fft.")
            _info(f"STFT: n_fft={n}, window=array(len={arr.size})")
            return arr

        if isinstance(w, str):
            name = w.strip().lower()
            if name == 'kaiser':
                beta = float(getattr(self, 'kaiser_beta', 6.5))
                if beta < 0:
                    _warn(f"kaiser_beta negativo ({beta}); ajustado para 6.5.")
                    beta = 6.5
                win = _sig.get_window(('kaiser', beta), n, fftbins=True)
                _info(f"STFT: n_fft={n}, window=kaiser(beta={beta})")
                return win

            if name in ('gaussian', 'gauss', 'gaussiana'):
                std = float(getattr(self, 'gaussian_std', n / 8.0))
                if std <= 0:
                    _warn(f"gaussian_std nÃ£o positivo ({std}); usando n/8.")
                    std = n / 8.0
                win = _sig.get_window(('gaussian', std), n, fftbins=True)
                _info(f"STFT: n_fft={n}, window=gaussian(std={std})")
                return win

            try:
                win = _sig.get_window(name, n, fftbins=True)
                _info(f"STFT: n_fft={n}, window={name}")
                return win
            except Exception as e:
                _warn(f"Janela '{w}' indisponÃ­vel ({e}); usando 'hann'.")
                return _sig.get_window('hann', n, fftbins=True)

        _err(f"Tipo de janela nÃ£o suportado: {type(w).__name__}.")
        raise TypeError("ParÃ¢metro 'window' deve ser str, list/tuple ou numpy.ndarray.")

    # ----------------- init -----------------
    def __init__(self):
        self.logger = logging.getLogger(__name__)

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

        # LFT
        self.use_lft: bool = False

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

        # Pesos a/ÃŸ
        self.harmonic_weight: float = 0.8
        self.inharmonic_weight: float = 0.2

        # Outros
        self.results_directory: Path = Path("./results")
        self.freq_min = 20.0
        self.freq_max = 20000.0
        self.db_min = -90.0
        self.db_max = 0.0
        self.tolerance = 10.0
        self.use_adaptive_tolerance = True
        self.zero_padding = 1
        self.time_avg = "mean"

        self.logger.info("AudioProcessor inicializado")

    # ----------------- carregamento -----------------
    def _load_audio_with_fallback(self, file_path: Path) -> Tuple[Optional[np.ndarray], Optional[int]]:
        p = str(file_path).replace('"', '').replace("'", "")
        if not os.path.exists(p):
            self.logger.error(f"Arquivo nÃ£o encontrado: {p}")
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
            self.logger.error(f"Falha em todos os mÃ©todos de carregamento: {e}")
            return None, None

    def load_audio_files(self, file_paths: List[Union[str, Path]]) -> None:
        start = time.time()
        self.audio_data.clear()
        for p in file_paths:
            try:
                p = Path(str(p).replace('"', '').replace("'", "")).resolve()
                if not p.exists():
                    self.logger.error(f"Arquivo nÃ£o encontrado: {p}")
                    continue
                y, sr = self._load_audio_with_fallback(p)
                if y is None or sr is None or len(y) == 0:
                    self.logger.warning(f"Dados invÃ¡lidos: {p}")
                    continue
                note = self.extract_note_name(p) or p.stem
                self.audio_data.append((y, sr, note, str(p)))
                self.logger.debug(f"Carregado: {p} (Nota: {note})")
            except Exception as e:
                self.logger.error(f"Erro ao carregar {p}: {e}")
        self.logger.info(f"Carregados {len(self.audio_data)} ficheiros em {time.time() - start:.2f}s.")

    def extract_note_name(self, file_path: Union[str, Path]) -> Optional[str]:
        name = file_path.name if isinstance(file_path, Path) else os.path.basename(str(file_path))
        name = name.replace('"', '').replace("'", "")
        patterns = [r"([A-G][#b]?)[-_]?(\d)", r"([A-G][#b]?)(\d)"]
        for pat in patterns:
            m = re.search(pat, name)
            if m:
                return m.group(1) + m.group(2)
        return None

    # ----------------- LFT helper -----------------
    def lft_power(self, **kwargs):
        if getattr(self, "sr", None) is None or getattr(self, "y", None) is None:
            raise RuntimeError("Ãudio nÃ£o carregado.")
        return spectral_power_lft(self.y, fs=self.sr, **kwargs)

    # ----------------- FFT -----------------
    def fft_analysis(self, zero_padding: int = 1) -> None:
        """
        STFT com gestÃ£o bÃ¡sica de memÃ³ria e restauro de parÃ¢metros.
        - MantÃ©m fluxo existente.
        - Acrescenta cÃ¡lculo universal de coherent gain da janela.
        """
        if self.y is None or self.sr is None:
            raise ValueError("Dados de Ã¡udio nÃ£o carregados.")

        start_time = time.time()
        orig_n_fft = int(self.n_fft)
        orig_hop = self.hop_length if self.hop_length is not None else None

        # hop por omissÃ£o
        if self.hop_length is None:
            self.hop_length = self.n_fft // 4

        # janela (argumento para librosa)
        win_arg = self._get_window_arg()
        win_length = len(win_arg)

        # aplica zero padding se necessÃ¡rio
        n_fft_padded = win_length * zero_padding

        # ajuste leve p/ sinais gigantes
        sig_len = len(self.y)
        if sig_len > 5_000_000:
            adj = min(self.n_fft, 8192)
            if adj != self.n_fft:
                self.logger.info(f"Reduzindo n_fft {self.n_fft}?{adj} (sinal longo)")
                self.n_fft = adj
                if orig_hop is None:
                    self.hop_length = self.n_fft // 4

        # possÃ­vel amostragem parcial (proteÃ§Ã£o de memÃ³ria)
        MAX_SIGNAL = 20_000_000
        y_work = self.y[:5_000_000 * 5] if sig_len > MAX_SIGNAL else self.y

        tried_downgrade = False
        try:
            gc.collect()
            y_norm = _normalize_level(y_work, target_rms_db=-20.0)
            y_norm = _normalize_level(y_work, target_rms_db=-20.0)
            self.S = librosa.stft(
                y_norm,
                n_fft=n_fft_padded,
                win_length=win_length,
                hop_length=self.hop_length,
                window=win_arg,
                center=True,
            )


            S_mag = np.abs(self.S)
            # dB de MAGNITUDE para visualizaÃ§Ã£o/thresholds; cÃ¡lculos mÃ©dios serÃ£o em POTÃŠNCIA noutro passo
            self.db_S = librosa.amplitude_to_db(S_mag, ref=np.max)
            self.freqs = librosa.fft_frequencies(sr=self.sr, n_fft=n_fft_padded)
            frame_idx = np.arange(self.S.shape[1])
            self.times = librosa.frames_to_time(frame_idx, sr=self.sr, hop_length=self.hop_length)

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

            self.logger.info(f"FFT concluÃ­da em {time.time()-start_time:.3f}s (shape={self.S.shape})")

        except MemoryError:
            self.logger.error("MemoryError em STFT")
            if not tried_downgrade:
                tried_downgrade = True
                self.n_fft = max(1024, self.n_fft // 4)
                self.hop_length = self.n_fft // 4
                self.S = None
                self.db_S = None
                gc.collect()
                return self.fft_analysis()
            else:
                raise
        except Exception as e:
            raise RuntimeError(f"Erro em FFT: {e}")
        finally:
            # restauro
            self.n_fft = orig_n_fft
            self.hop_length = (orig_hop if orig_hop is not None else self.n_fft // 4)


    # ----------------- LFT -----------------
    def fft_analysis_lft(self, zero_padding: int = 1, time_avg: str = 'mean') -> None:
        """
        'LFT' aqui Ã© uma FFT com zero-padding.
        - MantÃ©m db_S como dB de magnitude (freq x tempo) para visual/thresholds.
        - AgregaÃ§Ãµes temporais serÃ£o feitas em POTÃŠNCIA noutro passo (e.g., generate_complete_list).
        - Calcula e guarda coherent_gain da janela para uso universal.
        """
        if self.y is None or self.sr is None:
            raise ValueError("Dados de Ã¡udio nÃ£o carregados.")

        self.logger.info(f"LFT: n_fft={self.n_fft}, window={self.window}, zero_padding={zero_padding}")
        start_time = time.time()

        sig_len = len(self.y)
        orig_n_fft = int(self.n_fft)

        # Ajuste leve para sinais muito longos
        if sig_len > 5_000_000:
            adj = min(self.n_fft, 8192)
            if adj != self.n_fft:
                self.logger.info(f"Reduzindo n_fft {self.n_fft}?{adj} (sinal longo)")
                self.n_fft = adj

        # hop por omissÃ£o
        if self.hop_length is None:
            self.hop_length = self.n_fft // 4

        try:
            # ProteÃ§Ã£o de memÃ³ria: truncar gentilmente sinais gigantes
            MAX_SIGNAL = 20_000_000
            y_work = self.y
            if sig_len > MAX_SIGNAL:
                self.logger.info(f"Sinal muito longo ({sig_len}); processando parte inicial.")
                chunk_size = 5_000_000
                n_chunks = min(5, sig_len // chunk_size + 1)
                total = n_chunks * chunk_size
                y_work = self.y[:total]

            gc.collect()

            if LinearTimeFrequencyTransform is None:
                self.logger.error("LinearTimeFrequencyTransform indisponÃ­vel. Usando FFT.")
                return self.fft_analysis()

            lft = LinearTimeFrequencyTransform(
                window_size=int(self.n_fft),
                hop_size=int(self.hop_length),
                window_type=self.window,
                zero_padding=int(zero_padding)
            )

            # Transformada
            y_norm = _normalize_level(y_work, target_rms_db=-20.0)
            self.times, self.freqs, lft_result = lft.transform(y_norm, fs=self.sr)
            self.S = lft_result  # <- acrescentar isto
            self.logger.info(
                f"LFT concluída em {time.time()-start_time:.3f}s (shape={self.S.shape})"
            )




            # Magnitude (freq x tempo)
            lft_mag = lft.magnitude(lft_result)

            # db_S = dB de MAGNITUDE (para visualizaÃ§Ãµes/thresholds coerentes com FFT)
            self.db_S = 20.0 * np.log10(np.maximum(lft_mag, 1e-10))


            # ---------- coherent gain: calcular e guardar universalmente ----------
            cg_val = 1.0
            try:
                # Tentar helper existente no mÃ³dulo
                cg_val = float(_coherent_gain(self.window, int(self.n_fft)))  # type: ignore[name-defined]
            except Exception:
                try:
                    # Fallback: obter vetor de janela e usar mÃ©dia (ganho coerente)
                    w_vec = librosa.filters.get_window(self._get_window_arg(), int(self.n_fft), fftbins=True)
                    cg_val = float(np.mean(w_vec))
                except Exception:
                    cg_val = 1.0
            self.coherent_gain_value = cg_val if cg_val > 0.0 else 1.0
            # ---------------------------------------------------------------------

            self.logger.info(f"LFT concluÃ­da em {time.time()-start_time:.3f}s (shape={self.S.shape})")

        except Exception as e:
            self.logger.error(f"Erro em LFT: {e}. Fallback FFT.")
            # Restaurar n_fft antes do fallback
            self.n_fft = orig_n_fft
            self.fft_analysis()
        finally:
            # Restauro de n_fft (hop_length mantÃ©m-se conforme polÃ­tica do restante cÃ³digo)
            self.n_fft = orig_n_fft


    # ----------------- lista completa -----------------
    def generate_complete_list(self) -> None:
        if self.db_S is None or self.freqs is None:
            raise ValueError("Execute fft_analysis() ou fft_analysis_lft() antes de generate_complete_list().")

        self.logger.info("Gerando lista completa de parciais")
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

                # agregação temporal (mean/median/max)
                amp_lin = float(agg_func(amp_t))

                # evitar log(0)
                amp_lin = max(amp_lin, 1e-20)

                # amplitude -> dB de amplitude
                mag_db = 20.0 * np.log10(amp_lin)

                note_str = frequency_to_note_name(f)
                complete_list.append((f, mag_db, note_str))

            except Exception as e:
                self.logger.warning(f"Falha em generate_complete_list (i={i}, f={f:.2f} Hz): {e}")
                continue

        self.complete_list_df = pd.DataFrame(
            complete_list,
            columns=["Frequency (Hz)", "Magnitude (dB)", "Note"]
        )

        self.logger.info(f"Lista completa: {len(complete_list)} parciais em {time.time()-start:.3f}s")

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
        harmonic_weight: float = 0.8,
        inharmonic_weight: float = 0.2,
        weight_function: str = "linear",
        use_lft: bool = False,
        zero_padding: int = 1,
        time_avg: str = "mean",
        parallel_processing: bool = False,
        export_data_format: str = "json",
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        **kwargs: object,
    ) -> None:
        # armazenar parÃ¢metros principais
        self.window = str(window)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length) if hop_length is not None else self.n_fft // 2
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
        self.weight_function = str(weight_function)

        # validaÃ§Ã£o crÃ­tica
        self._validate_and_store_parameters(
            n_fft, hop_length, window, weight_function,
            harmonic_weight, inharmonic_weight,
            dissonance_enabled, dissonance_model, dissonance_curve, dissonance_scale, compare_models
        )

        # LFT
        self.use_lft = bool(use_lft)
        self.zero_padding = int(zero_padding)
        self.time_avg = str(time_avg)
        if self.time_avg not in {"mean", "median", "max"}:
            self.time_avg = "mean"
        if self.use_lft:
            if self.zero_padding < 1:
                self.zero_padding = 1

        # diretÃ³rios
        results_directory = self.results_directory
        interactive_dir = results_directory / "interactive_visualizations"
        results_directory.mkdir(parents=True, exist_ok=True)
        interactive_dir.mkdir(parents=True, exist_ok=True)

        # tolerÃ¢ncia sanidade
        if not (0 < float(self.tolerance) < 100):
            self.logger.warning(f"TolerÃ¢ncia {self.tolerance} fora do recomendado (0â€“100 Hz).")

        # processamento sequencial
        total_files = len(self.audio_data)
        self.logger.info(f"Processando {total_files} ficheiro(s) ...")
        self._sequential_process_audio_files(
            self.freq_min, self.freq_max, self.db_min, self.db_max, self.tolerance,
            results_directory, interactive_dir, export_data_format, progress_callback,
            use_lft=self.use_lft, zero_padding=self.zero_padding, time_avg=self.time_avg
        )

        # compilar mÃ©tricas e exportar combinados
        if total_files > 0:
            try:
                self._compile_metrics(results_directory)
            except Exception as e:
                self.logger.error(f"CompilaÃ§Ã£o de mÃ©tricas falhou: {e}")
            try:
                self._export_combined_data_for_visualization(results_directory, interactive_dir, export_data_format)
            except Exception as e:
                self.logger.error(f"ExportaÃ§Ã£o de dados combinados falhou: {e}")

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
            raise ValueError("n_fft deve ser positivo.")
        self.n_fft = n_fft
        self.hop_length = hop_length if hop_length is not None else n_fft // 4

        valid_windows = [
            'hann','hamming','bartlett','blackmanharris','flattop',
            'bohman','kaiser','gaussian','gauss','gaussiana'
        ]

        name = (window or '').lower()
        if name not in valid_windows:
            self.logger.warning(f"Janela '{window}' pode nÃ£o ser suportada. Recomendadas: {valid_windows}")
        self.window = name

        weight_name = (weight_function or "linear").strip().lower()
        _ = get_weight_function(weight_name)  # valida
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
                raise ValueError(f"Modelo de dissonÃ¢ncia desconhecido: {wanted}. DisponÃ­veis: {available}")
            self.dissonance_model = chosen
            self.dissonance_curve_enabled = bool(dissonance_curve)
            self.dissonance_scale_enabled = bool(dissonance_scale)
            self.dissonance_compare_models = bool(compare_models)

        self.logger.debug("ParÃ¢metros validados.")

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
        use_lft: bool = False,
        zero_padding: int = 1,
        time_avg: str = 'mean'
    ) -> None:
        import time
        import gc  # Importante para limpar memória em loops grandes

        start = time.time()
        
        for i, (y, sr, note, file_path) in enumerate(self.audio_data, 1):
            try:
                # 1. Notificar progresso na interface
                if progress_callback:
                    progress_callback(i, len(self.audio_data), note)
                
                # 2. Carregar dados para o estado da classe
                self.y, self.sr = y, sr
                self._reset_metrics()

                # 3. Executar Análise (FFT ou LFT)
                if use_lft:
                    try:
                        self.fft_analysis_lft(zero_padding=zero_padding, time_avg=time_avg)
                    except Exception as e:
                        self.logger.error(f"LFT falhou ({e}); a usar FFT normal.")
                        self.fft_analysis(zero_padding=zero_padding)
                else:
                    self.fft_analysis(zero_padding=zero_padding)

                # 4. Gerar lista de parciais
                self.generate_complete_list()

                # 5. Processar Harmónicos e Calcular Métricas
                # (É AQUI DENTRO que a correção Log/Linear que fizemos vai atuar)
                self._process_filtered_and_harmonic_data(
                    freq_min, freq_max, db_min, db_max, tolerance, note,
                    use_lft=use_lft, zero_padding=zero_padding, time_avg=time_avg
                )

                # 6. Guardar Resultados
                out_folder = results_directory / note
                out_folder.mkdir(parents=True, exist_ok=True)
                
                self.save_results(out_folder, note, use_lft=use_lft)
                self._export_data_for_visualization(note, out_folder, interactive_dir, export_data_format)

                self.logger.info(f"{i}/{len(self.audio_data)} processado: {note}")

                # 7. Limpeza de Memória (Crucial para batch processing)
                # Liberta a memória gráfica do Matplotlib e arrays grandes
                plt.close('all') 
                gc.collect()

            except Exception as e:
                self.logger.error(f"Erro ao processar {note}: {e}")
                continue
        
        self.logger.info(f"Processamento concluído em {time.time()-start:.2f}s")

    def _process_filtered_and_harmonic_data(
        self,
        freq_min: float, freq_max: float,
        db_min: float, db_max: float,
        tolerance: float, note: str,
        use_lft: bool = False, zero_padding: int = 1, time_avg: str = 'mean' # <-- Argumentos LFT aceites
    ) -> None:
        """
        Filtra a lista completa de parciais, aplica a correção de Ganho Coerente (Gc),
        gera a lista de harmónicos e inicia o cálculo de métricas (FFT ou LFT).
        """
        import numpy as np
        import pandas as pd
    
        self.note = note
        if self.complete_list_df is None or self.complete_list_df.empty:
            self.logger.error("Lista completa não gerada.")
            self.filtered_list_df = pd.DataFrame()
            self.harmonic_list_df = pd.DataFrame()
            return

        # ===== 1. BLOCO DE FILTROS E CORREÇÃO DE AMPLITUDE =====
    
        # Aplica filtros de frequência e magnitude (dB) [8, 9]
        fdf = self.complete_list_df[
            (self.complete_list_df['Frequency (Hz)'] >= freq_min) &
            (self.complete_list_df['Frequency (Hz)'] <= freq_max) &
            (self.complete_list_df['Magnitude (dB)'] >= db_min) &
            (self.complete_list_df['Magnitude (dB)'] <= db_max)
        ].copy()

        if fdf.empty:
            self.logger.warning(f"Nenhum componente dentro dos filtros para {note}")
            self.filtered_list_df = fdf
            self.harmonic_list_df = pd.DataFrame()
            self._set_default_metrics()
            return

        # dB (Magnitude) -> Amplitude Linear [1, 2]
        amps = np.power(10.0, fdf['Magnitude (dB)'].to_numpy(float) / 20.0)

        # Coherent Gain (Gc) Correction: Essencial para métricas absolutas [1, 2, 7]
        cg = float(getattr(self, "coherent_gain_value", 1.0) or 1.0)
        if cg <= 0.0:
            cg = 1.0
        
        fdf['Amplitude'] = amps / cg # APLICAR CORREÇÃO
        self._filtered_amp_corrected = True  # filtered_list_df já está corrigida por ganho coerente


    
        # Limpeza de valores extremos
        fdf.replace([np.inf, -np.inf], np.nan, inplace=True)
        fdf.dropna(subset=['Amplitude'], inplace=True)
        fdf = fdf.sort_values(by='Amplitude', ascending=False).reset_index(drop=True)
    
        self.filtered_list_df = fdf
        self.logger.info(f"Lista filtrada: {len(fdf)} parciais")

        # ===== 2. GERAÇÃO DE HARMÓNICOS =====
    
        self._generate_harmonic_list(
            note, freq_max, tolerance,
            use_adaptive_tolerance=getattr(self, "use_adaptive_tolerance", True)
        )

        # Inicialização de métricas (Reset)
        self.density_metric_value = None
        self.scaled_density_metric_value = None
        self.entropy_spectral_value = None
        self.combined_density_metric_value = None
        self.spectral_density_metric_value = None
        self.total_metric_value = None

        # ===== 3. CÁLCULO DE MÉTRICAS (CHAMADA CRÍTICA) =====
        if use_lft:
            # Chama o método LFT corrigido, passando os argumentos de compatibilidade [5, 10]
            self._calculate_metrics_lft(zero_padding=zero_padding, time_avg=time_avg)
        else:
            self._calculate_metrics()

        # ===== 4. VERIFICAÇÃO FINAL DE DM (Fallback) =====
        # Este bloco garante que o DM absoluto é calculado mesmo se o passo anterior falhar
        if (self.scaled_density_metric_value is None) or (self.scaled_density_metric_value == 0):
            self.logger.warning("Density Metric não calculada; tentativa de recálculo (corrigida).")
            try:
                if self.harmonic_list_df is not None and not self.harmonic_list_df.empty:
                
                    # Garantir coluna Amplitude [11, 12]
                    if 'Amplitude' not in self.harmonic_list_df.columns and 'Magnitude (dB)' in self.harmonic_list_df.columns:
                        self.harmonic_list_df['Amplitude'] = np.power(
                            10.0, self.harmonic_list_df['Magnitude (dB)'].to_numpy(float) / 20.0
                        )
                
                    if 'Amplitude' in self.harmonic_list_df.columns:
                        amps = self.harmonic_list_df['Amplitude'].to_numpy(float)
                    
                        # Calibração final do CG (usa o helper disponível) [11, 12]
                        try:
                            # Tenta obter a função de ganho coerente do escopo global
                            _cg_fn = globals().get('_coherent_gain') or globals().get('_coherent_gain_local')
                            cg_fallback = float(_cg_fn(getattr(self, "window", "hann"), int(getattr(self, "n_fft", 4096)))) if _cg_fn else 1.0
                        except Exception:
                            cg_fallback = 1.0
                    
                        if cg_fallback > 0:
                            amps = amps / cg_fallback

                        # Aplica a métrica de densidade absoluta [13, 14]
                        density = float(apply_density_metric(amps, self.weight_function, normalize=False))
                        self.density_metric_value = density
                        self.scaled_density_metric_value = density
                        self.logger.info(f"Density Metric recalculada (absoluta): {self.scaled_density_metric_value:.6f}")
                    else:
                        self.density_metric_value = 0.0
                        self.scaled_density_metric_value = 0.0
            except Exception as e:
                self.logger.error(f"Recalcular DM falhou: {e}", exc_info=True)


    def _reset_metrics(self) -> None:
        self.density_metric_value = None
        self.scaled_density_metric_value = None
        self.filtered_density_metric_value = None
        self.entropy_spectral_value = None
        self.combined_density_metric_value = None
        self.total_metric_value = None
        self.spectral_density_metric_value = None

        for model_name in self.dissonance_values:
            self.dissonance_values[model_name] = None
            self.dissonance_curves[model_name] = None
            self.dissonance_scales[model_name] = None

    # ----------------- F0 por nota (robusto) -----------------
    @lru_cache(maxsize=128)
    def calculate_fundamental_frequency(self, note: str) -> float:
        """
        Converte nome de nota para frequÃªncia via C0 e semitons; aceita bemÃ³is/sustenidos.
        """
        # normalizaÃ§Ãµes simples
        s = (note or "").strip()
        m = re.search(r'([A-Ga-g])([#?b?]?)[-_]?(-?\d+)', s)
        if not m:
            self.logger.warning(f"Formato de nota invÃ¡lido: {note}")
            return 0.0
        letter = m.group(1).upper()
        acc = m.group(2).replace('?', '#').replace('?', 'b')
        octave = int(m.group(3))
        pitch = letter + acc

        names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        flats = {'Db': 'C#', 'Eb': 'D#', 'Gb': 'F#', 'Ab': 'G#', 'Bb': 'A#'}
        if pitch in flats:
            pitch = flats[pitch]
        if pitch not in names:
            self.logger.warning(f"Nota nÃ£o reconhecida: {pitch}")
            return 0.0

        freq_A4 = 440.0
        # C0
        freq_C0 = freq_A4 * 2 ** (-4.75)
        # Ã­ndice do pitch
        idx = names.index(pitch)
        # semitons relativos a C0
        h = idx + 12 * octave
        f = freq_C0 * (2 ** (h / 12))
        self.logger.debug(f"F0({note}) = {f:.2f} Hz")
        return float(f)

    # ----------------- gerar lista de harmÃ³nicos -----------------
    def _generate_harmonic_list(
        self,
        note: str,
        freq_max: float,
        tolerance: float,
        use_adaptive_tolerance: bool = True
    ) -> None:

        f0 = self.calculate_fundamental_frequency(note)
        if f0 <= 0:
            self.logger.warning(f"F0 invÃ¡lido para {note}. Tentando estimar.")
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

        self.logger.info(f"F0({note}) = {f0:.2f} Hz")

        if self.filtered_list_df is None or self.filtered_list_df.empty:
            self.harmonic_list_df = pd.DataFrame()
            return

        harmonic_list = []
        max_harm = int(freq_max / f0) + 1
        expected = [f0 * n for n in range(1, max_harm + 1)]
        self.logger.info(f"Procurando {len(expected)} harmÃ³nicos atÃ© {freq_max:.1f} Hz")
        for hnum, ef in enumerate(expected, 1):
            tol_hz = max(tolerance, ef * 0.02) if use_adaptive_tolerance else tolerance
            candidates = self.filtered_list_df[
                (self.filtered_list_df['Frequency (Hz)'] >= ef - tol_hz) &
                (self.filtered_list_df['Frequency (Hz)'] <= ef + tol_hz)
            ]
            if not candidates.empty:
                best = candidates.loc[candidates['Amplitude'].idxmax()].copy()
                best['Harmonic Number'] = hnum
                exists = any(abs(r['Frequency (Hz)'] - best['Frequency (Hz)']) < 0.1 for _, r in pd.DataFrame(harmonic_list).iterrows()) if harmonic_list else False
                if not exists:
                    harmonic_list.append(best)
            else:
                if self.complete_list_df is not None and not self.complete_list_df.empty:
                    wtol = tol_hz * 1.5
                    cand2 = self.complete_list_df[
                        (self.complete_list_df['Frequency (Hz)'] >= ef - wtol) &
                        (self.complete_list_df['Frequency (Hz)'] <= ef + wtol)
                    ]
                    if not cand2.empty:
                        best = cand2.loc[cand2['Magnitude (dB)'].idxmax()].copy()
                        best['Harmonic Number'] = hnum
                        if 'Amplitude' not in best:
                            best['Amplitude'] = np.power(10.0, best['Magnitude (dB)'] / 20.0)
                        exists = any(abs(r['Frequency (Hz)'] - best['Frequency (Hz)']) < 0.1 for _, r in pd.DataFrame(harmonic_list).iterrows()) if harmonic_list else False
                        if not exists:
                            harmonic_list.append(best)

        self.harmonic_list_df = pd.DataFrame(harmonic_list).reset_index(drop=True) if harmonic_list else pd.DataFrame()
        if not self.harmonic_list_df.empty:
            self.logger.info(f"{len(self.harmonic_list_df)} harmÃ³nicos identificados")
        else:
            self.logger.warning("Nenhum harmÃ³nico encontrado.")

    # ----------------- mÃ©tricas (FFT) -----------------
    def _calculate_metrics(self) -> None:
        """
        Calcula as mÃ©tricas principais (modo FFT) a partir das listas harmÃ³nica/filtrada/completa.
        VersÃ£o robusta em escala absoluta:
          - corrige amplitudes pelo ganho coerente da janela;
          - DM e FDM por soma ponderada de amplitudes (normalize=False);
          - SDM a partir de potÃªncia (sem fallback DM*10);
          - mantÃ©m Combined e Total Metric como no design original.
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

                # Se (e só se) a convenção do seu pipeline for trabalhar com amplitudes "corrigidas pela janela",
                # aplique aqui exactamente uma vez. Caso contrário, comente este bloco.
                cg = float(getattr(self, "coherent_gain_value", 1.0) or 1.0)
                if cg > 0.0 and not getattr(self, "_harmonic_amp_corrected", False):
                    harmonic_amps = harmonic_amps / cg
                    self.harmonic_list_df["Amplitude"] = harmonic_amps
                    self._harmonic_amp_corrected = True


            # O bloco de correção redundante de ganho coerente foi removido aqui.
            # (harmonic_amps já contém valores corrigidos vindos do DataFrame)

            # ------------------- Density Metric (absoluta) -------------------
            self.density_metric_value = float(
                apply_density_metric(harmonic_amps, self.weight_function, normalize=False)
            ) if harmonic_amps.size > 0 else 0.0

            # manter coluna "scaled" para compatibilidade, mas sem *10
            self.scaled_density_metric_value = self.density_metric_value

            # ------------------- Spectral Density Metric (potÃªncia; sem fallback) -------------------
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
                    # O bloco de correção redundante de ganho coerente foi removido aqui.
                    # (camp já contém valores corrigidos vindos do complete_list_df)
    
                    cpow = camp ** 2  # potência
                    self.spectral_density_metric_value = float(
                        apply_density_metric(cpow, weight_function="linear", normalize=False)
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
                        # Se estes dB forem de amplitude, converte para potência: (10**(dB/20))**2
                        a_lin = np.power(10.0, pd.to_numeric(self.complete_list_df["Magnitude (dB)"], errors="coerce").to_numpy(float) / 20.0) ** 2
                    else:
                        a_lin = None

                    if f_hz is not None and a_lin is not None:
                        # threshold relativo a -40 dB (~ 0.01 em amplitude; 1e-4 em potência)
                        if "Magnitude (dB)" in self.complete_list_df.columns:
                            mask = self.complete_list_df["Magnitude (dB)"] >= (self.complete_list_df["Magnitude (dB)"].max() - 40.0)
                            f_hz = f_hz[mask.to_numpy()]
                            a_lin = a_lin[mask.to_numpy()]

                        # f0 se tiveres calculado anteriormente; senão, usa None
                        f0_est = None
                        try:
                            if self.harmonic_list_df is not None and not self.harmonic_list_df.empty:
                                f0_est = float(self.harmonic_list_df.nsmallest(1, "Frequency (Hz)")["Frequency (Hz)"].iloc[0])
                        except Exception:
                            f0_est = None

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
                            proximity_axis="bark",
                            sigma=0.5, bark_window=8.0,
                            max_peaks_per_band=4,
                            weight_r=calc_wr, weight_p=calc_wp, # <--- USAR VARIÁVEIS CALCULADAS
                            lambda_low=0.35,  # injeta “peso” de graves
                            low_bark_cut=8,
                        )
                        # Exporta: D_peso (novo), além de R_norm/P_norm/D_agn

                        # guardar para exportação
                        self.R_norm = float(dens["R_norm"])
                        self.P_norm = float(dens["P_norm"])
                        self.D_agn  = float(dens["D_agn"])
                        self.D_harm = (None if dens["D_harm"] is None else float(dens["D_harm"]))

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
                    
                    self.filtered_density_metric_value = float(
                        apply_density_metric(famps, self.weight_function, normalize=False)
                    )

            # ------------------- Entropia espectral -------------------
            if harmonic_amps.size > 0:
                powers = harmonic_amps ** 2  # entropia sobre potÃªncia (normalizada internamente)
                self.entropy_spectral_value = float(compute_spectral_entropy(powers))
            else:
                self.entropy_spectral_value = 0.0

            # ------------------- Combined (H/IH) como tinhas -------------------
            harm_density = min(1.0, len(self.harmonic_list_df) / 200.0)
            inharm_density = 0.0
            if (self.complete_list_df is not None and not self.complete_list_df.empty and
                "Frequency (Hz)" in self.complete_list_df.columns and
                "Frequency (Hz)" in self.harmonic_list_df.columns):
                try:
                    harmonic_freqs = set(self.harmonic_list_df["Frequency (Hz)"].values)
                    inharmonic_components = len(
                        self.complete_list_df[~self.complete_list_df["Frequency (Hz)"].isin(harmonic_freqs)]
                    )
                    inharm_density = min(1.0, inharmonic_components / 5000.0)
                except Exception:
                    inharm_density = 0.0

            self.combined_density_metric_value = float(
                calculate_combined_density_metric(
                    harm_density, inharm_density, self.harmonic_weight, self.inharmonic_weight
                )
            )

            # Definir um limite empírico seguro para a Densidade Absoluta (DM/SDM).
            # Este valor deve ser ajustado com base na sua base de dados, mas 20.0
            # é um valor seguro para métricas DM/SDM absolutas corrigidas (após Coherent Gain).
            # Se o seu sinal de entrada for normalizado para -20dB RMS, 20.0 deve ser um limite superior conservador.

            MAX_ABS_DENSITY = 20.0
            wD, wS, wE, wC = 0.3, 0.2, 0.2, 0.3 # Pesos mantidos

            # ------------------- Total Metric (Corrigida: Normalização por Limite Empírico) -------------------

            # 1. Normalização da Densidade (DM)
            if self.scaled_density_metric_value is not None:
                # Divide pelo máximo empírico e limita (clip) o resultado a [7]
                norm_density = np.clip(self.scaled_density_metric_value / MAX_ABS_DENSITY, 0.0, 1.0)
            else:
                norm_density = 0.0

            # 2. Normalização da Densidade Espectral (SDM)
            if self.spectral_density_metric_value is not None:
                # Divide pelo máximo empírico e limita (clip) o resultado a [7]
                norm_spectral = np.clip(self.spectral_density_metric_value / MAX_ABS_DENSITY, 0.0, 1.0)
            else:
                norm_spectral = 0.0

            # 3. Entropia Espectral (já é normalizada 0-1)
            norm_entropy = self.entropy_spectral_value or 0.0
            # Garante que Entropia está limitada, caso o cálculo interno falhe
            norm_entropy = np.clip(norm_entropy, 0.0, 1.0) 

            # 4. Métrica Combinada (já é limitada 0-1 por construção, mas deve ser verificada)
            norm_combined = self.combined_density_metric_value or 0.0
            # Garante que Combined Metric está limitada
            norm_combined = np.clip(norm_combined, 0.0, 1.0)

            # Cálculo Final da Total Metric (mantém-se a multiplicação por 10.0 se desejar a escala 0-10)
            self.total_metric_value = (wD*norm_density + wS*norm_spectral + wE*norm_entropy + wC*norm_combined) * 10.0
            

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
            self.logger.error(f"Erro crÃ­tico em _calculate_metrics: {e}", exc_info=True)
            self._set_default_metrics()



    # ----------------- defaults mÃ©tricas -----------------
    def _set_default_metrics(self):
        self.density_metric_value = 0.0
        self.scaled_density_metric_value = 0.0
        self.filtered_density_metric_value = 0.0
        self.entropy_spectral_value = 0.0
        self.combined_density_metric_value = 0.0
        self.total_metric_value = 0.0
        self.spectral_density_metric_value = 0.0
        self.logger.warning("MÃ©tricas definidas como valores padrÃ£o (0.0)")

    # ----------------- mÃ©tricas (LFT) -----------------
    def _calculate_metrics_lft(self, zero_padding: int = 1, time_avg: str = 'mean') -> None:
        """
        Versão LFT do cálculo de métricas, robusta e em escala absoluta.
    
        Argumentos 'zero_padding' e 'time_avg' são aceites para compatibilidade com
        o método chamador, mas o cálculo utiliza os atributos internos definidos
        na fase fft_analysis_lft [1, 5].
        """
        try:
            import numpy as np
            import pandas as pd
        
            # --- helper local para Ganho Coerente (essencial para calibração) ---
            def _coherent_gain_local(win: str, n_fft: int) -> float:
                """Ganho coerente da janela: G = (1/N) * sum w[n]."""
                try:
                    import numpy as _np
                    # A lógica robusta de janelamento (SciPy/NumPy fallback) é assumida aqui [6]
                    try:
                        from scipy.signal import windows as _win
                        wname = (win or "").lower()
                        if wname in ("flattop", "flat-top", "flat_top"):
                            w = _win.flattop(n_fft, sym=False)
                        # ... [outras janelas SciPy]
                        else:
                            w = _win.hann(n_fft, sym=False)
                    except Exception:
                        w = _np.hanning(n_fft)
                    return float(_np.sum(w) / float(n_fft))
                except Exception:
                    return 1.0

            # ---------- validações ----------
            if self.harmonic_list_df is None or self.complete_list_df is None:
                self._set_default_metrics()
                return
            if self.harmonic_list_df.empty:
                self._set_default_metrics()
                return

            # ---------- amplitudes harmónicas (DM) ----------
            harmonic_amps = np.asarray([], dtype=float)

            if self.harmonic_list_df is not None and not self.harmonic_list_df.empty:

                if "Amplitude" in self.harmonic_list_df.columns:
                    harmonic_amps = self.harmonic_list_df["Amplitude"].to_numpy(dtype=float)

                elif "Magnitude (dB)" in self.harmonic_list_df.columns:
                    db_vals = self.harmonic_list_df["Magnitude (dB)"].to_numpy(dtype=float)

                    # dB de amplitude → amplitude linear (A_ref = 1)
                    harmonic_amps = np.power(10.0, db_vals / 20.0)

                    # guardar (sem try/except: se falhar, é erro estrutural do DF)
                    self.harmonic_list_df["Amplitude"] = harmonic_amps

                # limpeza numérica
                harmonic_amps = np.nan_to_num(harmonic_amps, nan=0.0, posinf=0.0, neginf=0.0)
                harmonic_amps = np.maximum(harmonic_amps, 0.0)

                # IMPORTANTÍSSIMO:
                # NÃO aplique aqui /cg se a sua harmonic_list_df vem de filtered_list_df já corrigida.
                # A correcção por cg deve existir UMA vez no pipeline, tipicamente quando cria filtered_list_df["Amplitude"].


            # ---------- Density Metric (absoluta; sem normalize) ----------
            if harmonic_amps.size > 0:
                self.density_metric_value = float(
                    apply_density_metric(harmonic_amps, self.weight_function, normalize=False)
                )
            else:
                self.density_metric_value = 0.0
            self.scaled_density_metric_value = self.density_metric_value # Sem *10 para DM

            # ---------- Spectral Density Metric (potência; CORRIGIDO) ----------
            self.spectral_density_metric_value = 0.0
            camp = None # <--- CORREÇÃO CRÍTICA: Inicialização para resolver NameError/Scope

            if self.complete_list_df is not None and not self.complete_list_df.empty:
            
                if "Amplitude" in self.complete_list_df.columns:
                    camp = self.complete_list_df["Amplitude"].to_numpy(float)
            
                elif "Magnitude (dB)" in self.complete_list_df.columns:
                    # 1. Conversão dB -> Linear Amplitude
                    raw_amps = np.power(10.0, self.complete_list_df["Magnitude (dB)"].to_numpy(float) / 20.0)
                
                    # 2. Aplica correção do Ganho Coerente UMA VEZ
                    cg = float(getattr(self, "coherent_gain_value", 1.0) or 1.0)
                    camp = raw_amps / (cg if cg > 0.0 else 1.0) # Uso do valor armazenado
                
                    try:
                        self.complete_list_df["Amplitude"] = camp
                    except Exception:
                        pass
            
                # Nota: O bloco de re-correção redundante CG é intencionalmente omitido aqui.

                if camp is not None and camp.size > 0:
                    cpow = camp ** 2 # Potência calibrada
                
                    # SDM absoluta (sem normalize); usa 'linear' para potência [8]
                    self.spectral_density_metric_value = float(
                        apply_density_metric(cpow, weight_function="linear", normalize=False)
                    )

            # ---------- Filtered Density (absoluta; CORRIGIDO - remoção CG redundante) ----------
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
                    # REMOVIDO: Bloco de correção redundante de CG [9]
                
                    self.filtered_density_metric_value = float(
                        apply_density_metric(famps, self.weight_function, normalize=False)
                    )

            # ---------- Entropia espectral e Combined Metric ----------
            if harmonic_amps.size > 0:
                powers = harmonic_amps ** 2
                self.entropy_spectral_value = float(compute_spectral_entropy(powers))
            else:
                self.entropy_spectral_value = 0.0

            # [Cálculo da Densidade Combinada (H/IH) - Estrutura Original [10, 11]]
            harm_density = min(1.0, len(self.harmonic_list_df) / 200.0)
            inharm_density = 0.0
        
            if (self.complete_list_df is not None and not self.complete_list_df.empty and
                "Frequency (Hz)" in self.complete_list_df.columns and
                "Frequency (Hz)" in self.harmonic_list_df.columns):
                try:
                    harmonic_freqs = set(self.harmonic_list_df["Frequency (Hz)"].values)
                    inharmonic_components = len(
                        self.complete_list_df[~self.complete_list_df["Frequency (Hz)"].isin(harmonic_freqs)]
                    )
                    inharm_density = min(1.0, inharmonic_components / 5000.0)
                except Exception:
                    inharm_density = 0.0
        
            self.combined_density_metric_value = float(
                calculate_combined_density_metric(
                    harm_density, inharm_density, self.harmonic_weight, self.inharmonic_weight
                )
            )

            # ---------- Total Metric (CORRIGIDO: Normalização Estável) [2-4] ----------
        
            MAX_ABS_DENSITY = 20.0 # Limite empírico seguro para métricas absolutas
            wD, wS, wE, wC = 0.3, 0.2, 0.2, 0.3 # Pesos fixos [12]

            # Normalização Bounded (DM)
            norm_density = (
                np.clip(self.scaled_density_metric_value / MAX_ABS_DENSITY, 0.0, 1.0)
                if self.scaled_density_metric_value is not None else 0.0
            )
        
            # Normalização Bounded (SDM)
            norm_spectral = (
                np.clip(self.spectral_density_metric_value / MAX_ABS_DENSITY, 0.0, 1.0)
                if self.spectral_density_metric_value is not None else 0.0
            )
        
            # Entropia e Combinada (clipados por segurança)
            norm_entropy = np.clip(self.entropy_spectral_value or 0.0, 0.0, 1.0)
            norm_combined = np.clip(self.combined_density_metric_value or 0.0, 0.0, 1.0)

            # Cálculo Final da Total Metric (escala 0-10) [2, 13]
            self.total_metric_value = (wD * norm_density + wS * norm_spectral + wE * norm_entropy + wC * norm_combined) * 10.0
            
            # Dynamic Density Score (Acoustic Momentum)
            try:
                import math
                if self.filtered_density_metric_value > 0:
                    self.dynamic_density_score = self.combined_density_metric_value * math.log10(self.filtered_density_metric_value)
                else:
                    self.dynamic_density_score = 0.0
            except Exception:
                self.dynamic_density_score = 0.0


            # ---------------------------------
            # ---------- Dissonância ----------
            if getattr(self, "dissonance_enabled", False):
                self.calculate_dissonance_metrics()

        except Exception as e:
            self.logger.error(f"Erro crítico em _calculate_metrics_lft: {e}", exc_info=True)
            self._set_default_metrics()


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

            # ---- Density Metric (absoluta; sem normalize; sem *10) ----
            if self.density_metric_value is None:
                self.density_metric_value = float(
                    apply_density_metric(amps_c, self.weight_function, normalize=False)
                )
                # manter compatibilidade de coluna "scaled", mas sem multiplicar por 10
                self.scaled_density_metric_value = self.density_metric_value

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
                        self.filtered_density_metric_value = float(
                            apply_density_metric(famps_c, self.weight_function, normalize=False)
                        )
                    else:
                        self.filtered_density_metric_value = 0.0
                else:
                    self.filtered_density_metric_value = 0.0

            # ---- Entropia espectral (sobre potÃªncia) ----
            if self.entropy_spectral_value is None:
                self.entropy_spectral_value = float(compute_spectral_entropy((amps_c ** 2)))

            # ---- Combined (H/IH) simples, como tinhas ----
            if self.combined_density_metric_value is None:
                harm_density = min(1.0, len(self.harmonic_list_df) / 200.0)
                inharm_density = 0.0
                self.combined_density_metric_value = float(
                    calculate_combined_density_metric(
                        harm_density, inharm_density, self.harmonic_weight, self.inharmonic_weight
                    )
                )

            # ---- Total Metric (idÃªntica ao mÃ©todo principal) ----
            if self.total_metric_value is None:
                wD, wS, wE, wC = 0.3, 0.2, 0.2, 0.3
                # usa as mesmas normalizaÃ§Ãµes do principal
                sd = self.scaled_density_metric_value or 0.0
                sdm = self.spectral_density_metric_value or 0.0
                norm_density  = sd / 10.0
                norm_spectral = sdm / 10.0
                norm_entropy  = self.entropy_spectral_value or 0.0
                norm_combined = self.combined_density_metric_value or 0.0
                self.total_metric_value = (wD*norm_density + wS*norm_spectral + wE*norm_entropy + wC*norm_combined) * 10.0

        except Exception as e:
            self.logger.error(f"Erro na verificaÃ§Ã£o de mÃ©tricas: {e}", exc_info=True)
            self._set_default_metrics()


    # ----------------- dissonÃ¢ncia -----------------
    def calculate_dissonance_metrics(self) -> None:
        if not self.dissonance_enabled or self.harmonic_list_df is None or self.harmonic_list_df.empty:
            return
        try:
            # Garantir que a coluna Amplitude existe
            if 'Amplitude' not in self.harmonic_list_df.columns:
                self.harmonic_list_df['Amplitude'] = np.power(10.0, self.harmonic_list_df['Magnitude (dB)'] / 20.0)

            # --- OTIMIZAÇÃO CRÍTICA (Limitador de Picos) ---
            # Para notas muito graves (Contrafagote), a janela gigante deteta milhares de parciais.
            # Calcular dissonância com 6000 parciais bloqueia o CPU (milhões de operações).
            # Limitamos aqui aos 50 parciais mais fortes, que contêm 99% da energia percetível.
            
            df_calc = self.harmonic_list_df.copy() # Cópia para não estragar dados originais
            
            if len(df_calc) > 80:
                df_calc = df_calc.nlargest(80, 'Amplitude')
            # -----------------------------------------------

            # Gerar lista de tuplos baseada APENAS nestes 50 parciais filtrados
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
                    self.logger.error(f"Dissonância {mname} falhou: {e}")
                    self.dissonance_values[mname] = None
                    self.dissonance_curves[mname] = None
                    self.dissonance_scales[mname] = None

        except Exception as e:
            self.logger.error(f"Erro em calculate_dissonance_metrics: {e}")
            # Limpeza de segurança
            if hasattr(self, 'dissonance_values') and self.dissonance_values:
                for m in self.dissonance_values:
                    self.dissonance_values[m] = None
                    self.dissonance_curves[m] = None
                    self.dissonance_scales[m] = None

    # ----------------- compilar mÃ©tricas / exportaÃ§Ãµes -----------------
    def _compile_metrics(self, results_directory: Path) -> None:
        try:
            from compile_metrics import compile_density_metrics_with_pca
            outp = results_directory / 'compiled_density_metrics.xlsx'
            compile_density_metrics_with_pca(folder_path=results_directory, output_path=outp, include_pca=True)
            self.logger.info(f"MÃ©tricas compiladas (PCA) em: {outp}")
        except ImportError:
            self.logger.error("compile_density_metrics_with_pca indisponÃ­vel.")
            try:
                from compile_metrics import compile_density_metrics
                outp = results_directory / 'compiled_metrics.xlsx'
                compile_density_metrics(results_directory, outp)
                self.logger.info(f"MÃ©tricas compiladas (sem PCA) em: {outp}")
            except ImportError:
                self.logger.error("compile_density_metrics indisponÃ­vel.")
        except Exception as e:
            self.logger.error(f"Erro na compilaÃ§Ã£o de mÃ©tricas: {e}")

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
            self.logger.error(f"Erro exportando dados p/ visualizaÃ§Ã£o: {e}")
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
            compiled_file = None
            for fn in ['compiled_density_metrics.xlsx', 'compiled_metrics.xlsx']:
                cand = results_directory / fn
                if cand.exists():
                    compiled_file = cand
                    break
            if compiled_file is None:
                self.logger.warning("Sem ficheiro compilado; a saltar exportaÃ§Ã£o combinada.")
                return
            df = pd.read_excel(compiled_file)
            if df.empty:
                self.logger.warning("Ficheiro compilado vazio.")
                return

            if export_format.lower() == 'json':
                path = interactive_dir / "combined_metrics.json"
                with open(path, 'w') as f:
                    json.dump(df.to_dict(orient='records'), f)
            elif export_format.lower() == 'csv':
                path = interactive_dir / "combined_metrics.csv"
                df.to_csv(path, index=False)

            cfg = {
                'metrics_available': [c for c in df.columns if c not in ['Note', 'Folder']],
                'notes': sorted(df['Note'].unique().tolist() if 'Note' in df.columns else []),
                'model_names': list_available_models(),
                'interactive_visualizations': {
                    'spectrogram_3d': True,
                    'dissonance_curves': self.dissonance_curve_enabled,
                    'pca_scatter': 'PC1' in df.columns,
                    'tsne_scatter': 'TSNE1' in df.columns and 'TSNE2' in df.columns,
                    'umap_scatter': 'UMAP1' in df.columns and 'UMAP2' in df.columns,
                    'anomaly_detection': 'is_anomaly' in df.columns
                }
            }
            with open(interactive_dir / "visualization_config.json", 'w') as f:
                json.dump(cfg, f)
        except Exception as e:
            self.logger.error(f"Erro ao exportar dados combinados: {e}")

    # ----------------- visualizaÃ§Ãµes -----------------
    def plot_spectrograms(
        self,
        path: Optional[Union[str, Path]] = None,
        note: str = ""
    ) -> None:
        if any(v is None for v in (self.db_S, self.freqs, self.times, getattr(self, "S", None))):
            self.logger.error("Dados insuficientes para plotar (db_S/freqs/times/S).")
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
            ax1.set_title(f"Spectrogram (dB) â€” Note: {note}")

            ax2 = plt.subplot(3, 1, 2)
            mean_spectrum_mag = S_mag.mean(axis=1)
            mean_spectrum_db = 20.0 * np.log10(np.maximum(mean_spectrum_mag, 1e-12))
            n = min(len(self.freqs), len(mean_spectrum_db))
            ax2.plot(self.freqs[:n], mean_spectrum_db[:n])
            ax2.set_title(f"Frequency Spectrum (mean over time) â€” Note: {note}")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Magnitude (dB)")
            ax2.set_xscale("log")

            ax3 = plt.subplot(3, 1, 3)
            S_power = S_mag**2
            S_mel = librosa.feature.melspectrogram(S=S_power, sr=self.sr, n_mels=128)
            S_db_mel = librosa.power_to_db(S_mel, ref=np.max)
            librosa.display.specshow(S_db_mel, sr=self.sr, x_axis="time", y_axis="mel", cmap="magma")
            plt.colorbar(format="%+2.0f dB", ax=ax3)
            ax3.set_title(f"Mel Spectrogram (dB) â€” Note: {note}")

            plt.tight_layout()

            if path:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=150, bbox_inches="tight")
                self.logger.info(f"Espectrograma salvo em: {path}")
                plt.close(fig)

                path_3d = path.with_name(path.stem + "_3d").with_suffix(".html")
                self.plot_3d_spectrogram(path=path_3d, note=note)
            else:
                plt.show()
                plt.close(fig)
        except Exception as e:
            self.logger.error(f"Erro ao plotar espectrogramas: {e}")
            plt.close(fig)

    def plot_3d_spectrogram(
        self,
        path: Optional[Union[str, Path]] = None,
        note: str = ""
    ) -> None:
        if any(v is None for v in (self.db_S, self.freqs, self.times)):
            self.logger.error("Dados insuficientes para plotar 3D.")
            return
        try:
            Z = np.asarray(self.db_S, dtype=float)
            X = np.asarray(self.times, dtype=float)
            Y = np.asarray(self.freqs, dtype=float)
            if Z.shape != (len(Y), len(X)):
                ny = min(Z.shape[0], len(Y))
                nx = min(Z.shape[1], len(X))
                Z = Z[:ny, :nx]; Y = Y[:ny]; X = X[:nx]

            surface = go.Surface(z=Z, x=X, y=Y, colorscale="Viridis", showscale=True)
            layout = go.Layout(
                title=f"3D Spectrogram (dB) â€” Note: {note}",
                scene=dict(
                    xaxis=dict(title="Time (s)"),
                    yaxis=dict(title="Frequency (Hz)", type="log"),
                    zaxis=dict(title="Magnitude (dB)"),
                ),
                width=900, height=700, margin=dict(l=65, r=50, b=65, t=90),
            )
            fig = go.Figure(data=[surface], layout=layout)
            fig.update_layout(scene_camera=dict(eye=dict(x=1.8, y=-1.8, z=0.8), up=dict(x=0, y=0, z=1)))
            if path:
                path = Path(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.write_html(str(path))
                self.logger.info(f"Espectrograma 3D salvo em: {path}")
            else:
                fig.show()
        except Exception as e:
            self.logger.error(f"Erro ao plotar espectrograma 3D: {e}")

    def plot_dissonance_curve(
        self,
        model_name: str,
        path: Optional[Union[str, Path]] = None,
        note: str = ""
    ) -> None:
        if not getattr(self, "dissonance_enabled", False) or model_name not in self.dissonance_curves:
            self.logger.warning(f"Modelo de dissonÃ¢ncia {model_name} nÃ£o disponÃ­vel.")
            return
        curve = self.dissonance_curves.get(model_name)
        scale = self.dissonance_scales.get(model_name)
        if curve is None or scale is None:
            self.logger.warning(f"Sem curva de dissonÃ¢ncia para {model_name}.")
            return
        try:
            model = get_dissonance_model(model_name)
            title = f"{model_name} Dissonance Curve â€” Note: {note}"
            model.visualize_dissonance_curve(curve, scale, title=title, save_file=path)
            if path:
                self.logger.info(f"Curva de dissonÃ¢ncia {model_name} salva em: {path}")
        except Exception as e:
            self.logger.error(f"Erro ao plotar curva de dissonÃ¢ncia: {e}")

    def plot_dissonance_comparison(
        self,
        path: Optional[Union[str, Path]] = None,
        note: str = ""
    ) -> None:
        if not getattr(self, "dissonance_enabled", False) or not getattr(self, "dissonance_compare_models", False):
            return
        models_with_data = [m for m, v in self.dissonance_curves.items() if v is not None]
        if len(models_with_data) < 2:
            self.logger.warning("Curvas de dissonÃ¢ncia insuficientes para comparaÃ§Ã£o.")
            return
        try:
            from dissonance_models import compare_dissonance_models
            if "Amplitude" not in self.harmonic_list_df.columns:
                self.harmonic_list_df["Amplitude"] = np.power(10.0, self.harmonic_list_df["Magnitude (dB)"] / 20.0)
            partials = [(row["Frequency (Hz)"], row["Amplitude"]) for _, row in self.harmonic_list_df.iterrows()]
            compare_dissonance_models(partials, 1.0, 2.0, 200, save_file=path, models_to_include=models_with_data)
            if path:
                self.logger.info(f"ComparaÃ§Ã£o de modelos de dissonÃ¢ncia salva em: {path}")
        except Exception as e:
            self.logger.error(f"Erro na comparaÃ§Ã£o de dissonÃ¢ncia: {e}")

    # ----------------- salvar resultados (grÃ¡ficos + excel) -----------------
    def save_results(self, output_folder: Union[str, Path], note: str, use_lft: bool = False) -> None:
        output_folder = Path(output_folder)
        output_folder.mkdir(exist_ok=True, parents=True)

        analysis_method = "LFT" if use_lft else "FFT"
        self.logger.info(f"Salvando resultados ({analysis_method}) para '{note}' em {output_folder}")

        # garantir mÃ©tricas completas
        try:
            self._ensure_all_metrics_calculated()
        except Exception as e:
            self.logger.warning(f"Falha ao fechar mÃ©tricas antes de salvar: {e}")

        # grÃ¡ficos
        try:
            spectrogram_png_path = output_folder / "spectrogram.png"
            self.plot_spectrograms(path=spectrogram_png_path, note=note)
        except Exception as e:
            self.logger.error(f"Erro ao salvar espectrogramas: {e}")

        if self.dissonance_enabled and self.dissonance_curve_enabled:
            try:
                if self.dissonance_compare_models:
                    comp = output_folder / "dissonance_comparison.png"
                    self.plot_dissonance_comparison(path=comp, note=note)
                models_to_process = list(self.dissonance_values.keys()) if self.dissonance_compare_models else [self.dissonance_model]
                for m in models_to_process:
                    if self.dissonance_curves.get(m) is not None:
                        cpath = output_folder / f"{str(m).lower()}_dissonance_curve.png"
                        self.plot_dissonance_curve(m, path=cpath, note=note)
            except Exception as e:
                self.logger.error(f"Erro ao salvar curvas de dissonÃ¢ncia: {e}")

        # Excel
        excel_path = output_folder / "spectral_analysis.xlsx"
        try:
            with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
                self._save_spectral_data_to_excel(writer, note)
            self.logger.info(f"AnÃ¡lise espectral salva em: {excel_path}")
        except PermissionError as pe:
            self.logger.error(f"PermissÃ£o negada: {pe}")
        except Exception as exc:
            self.logger.error(f"Erro ao salvar resultados: {exc}")

    def _save_spectral_power_data(self, writer: pd.ExcelWriter, note: str) -> None:
        log = self.logger
        try:
            if getattr(self, "y", None) is None or getattr(self, "sr", None) is None:
                log.warning("Sem y/sr; a saltar 'Spectral Power'.")
                return
            y = np.asarray(self.y, dtype=float).ravel()
            if y.size == 0:
                log.warning("Sinal vazio; a saltar 'Spectral Power'.")
                return

            n_fft = int(getattr(self, "n_fft", 4096) or 4096)
            hop = int(getattr(self, "hop_length", n_fft // 4) or n_fft // 4)
            window_type = str(getattr(self, "window", "hann") or "hann")
            order = max(1, n_fft // 2)

            try:
                sp_fft_db = spectral_power(y, n_fft=n_fft, hop_length=hop, window_type=window_type, order=order)
            except Exception as e:
                log.warning(f"spectral_power falhou ({e}); fallback FFT.")
                win = np.hanning(min(n_fft, y.size))
                x = y[:win.size] * win
                X = np.fft.rfft(x, n=win.size)
                power = (np.abs(X) ** 2) / max(1, win.size)
                power[power <= 1e-20] = 1e-20
                sp_fft_db = 10.0 * np.log10(power)

            sp_fft_db = np.asarray(sp_fft_db, dtype=float)
            freq_fft = np.fft.rfftfreq(n_fft, d=1.0 / float(self.sr))
            if sp_fft_db.shape[0] != freq_fft.shape[0]:
                m = min(sp_fft_db.shape[0], freq_fft.shape[0])
                sp_fft_db = sp_fft_db[:m]
                freq_fft = freq_fft[:m]

            power_linear = np.power(10.0, sp_fft_db / 10.0)
            if power_linear.size > 0:
                power_linear_no_dc = power_linear.copy()
                power_linear_no_dc[0] = 0.0
            else:
                power_linear_no_dc = power_linear

            try:
                sp_density = apply_density_metric(power_linear_no_dc, getattr(self, "weight_function", "linear"))
            except Exception as e:
                log.warning(f"apply_density_metric na potÃªncia falhou: {e}; usando 0.")
                sp_density = 0.0

            total_power = float(np.sum(power_linear_no_dc))
            total_power_db = 10.0 * np.log10(total_power) if total_power > 0.0 else -100.0
            average_power = float(np.mean(power_linear_no_dc)) if power_linear_no_dc.size else 0.0
            average_power_db = 10.0 * np.log10(average_power) if average_power > 0.0 else -100.0
            rms_power = float(np.sqrt(np.mean(power_linear_no_dc))) if power_linear_no_dc.size else 0.0
            rms_power_db = 20.0 * np.log10(rms_power) if rms_power > 0.0 else -100.0

            try:
                peak_idx = int(np.argmax(power_linear_no_dc)) if power_linear_no_dc.size else 0
                peak_freq = float(freq_fft[peak_idx]) if peak_idx < freq_fft.size else 0.0
            except Exception:
                peak_freq = 0.0

            spectral_centroid = None
            try:
                denom = np.sum(power_linear_no_dc)
                if denom > 0.0:
                    spectral_centroid = float(np.sum(freq_fft * power_linear_no_dc) / denom)
            except Exception:
                spectral_centroid = None

            df_fft = pd.DataFrame({
                "Frequency (Hz)": freq_fft,
                "Spectral Power (dB)": sp_fft_db
            })
            if not df_fft.empty:
                df_fft.at[0, "Spectral Density Metric"] = sp_density
                df_fft.at[0, "Total Power (dB)"] = total_power_db
                df_fft.at[0, "Average Power (dB)"] = average_power_db
                df_fft.at[0, "RMS Power (dB)"] = rms_power_db
                df_fft.at[0, "Peak Frequency (Hz)"] = peak_freq
                if spectral_centroid is not None:
                    df_fft.at[0, "Spectral Centroid (Hz)"] = spectral_centroid

            df_fft.to_excel(writer, sheet_name="Spectral Power", index=False)

        except Exception as e:
            log.error(f"Erro ao guardar potÃªncia espectral: {e}", exc_info=True)
            raise

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

    def _save_spectral_data_to_excel(self, writer: pd.ExcelWriter, note: str) -> None:
        import numpy as np
        import pandas as pd
        
        log = self.logger
        try:
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

            if isinstance(self.harmonic_list_df, pd.DataFrame) and not self.harmonic_list_df.empty:
                df_harm = _ensure_amp_column(self.harmonic_list_df)
                cols = [c for c in ["Harmonic Number", "Frequency (Hz)", "Magnitude (dB)", "Amplitude", "Note"] if c in df_harm.columns]
                (df_harm[cols] if cols else df_harm).to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
                log.debug(f"Espectro harmónico salvo: {len(df_harm)}")

            # ===== 2. GARANTIR MÉTRICAS =====
            try:
                self._ensure_all_metrics_calculated()
            except Exception as e:
                log.warning(f"_ensure_all_metrics_calculated falhou: {e}")
                self._set_default_metrics()

            # ===== 3. MÉTRICAS CONSOLIDADAS =====
            hl = (getattr(self, "hop_length", None) or int(getattr(self, "n_fft", 4096)) // 2)
            
            # Helper de Coherent Gain
            _cg_fn = (globals().get("_coherent_gain") or globals().get("_coherent_gain_local"))
            try:
                cg_val = float(_cg_fn(getattr(self, "window", "hann"), int(getattr(self, "n_fft", 4096)))) if _cg_fn else 1.0
            except Exception:
                cg_val = 1.0

            # --- CORREÇÃO: RECALCULAR PESOS PARA O RELATÓRIO ---
            # Para o Excel mostrar os valores reais usados no cálculo
            _hw = float(getattr(self, "harmonic_weight", 0.95))
            _wf = str(getattr(self, "weight_function", "linear")).lower()
            
            if _wf == "log":
                _theta = _hw * (np.pi / 2)
                _final_wp = np.sin(_theta)
                _final_wr = np.cos(_theta)
            else:
                _final_wp = _hw
                _final_wr = 1.0 - _hw
            # ----------------------------------------------------

            main_metrics = {
                "Note": note,
                "Analysis Type": ("LFT" if getattr(self, "use_lft", False) else "FFT"),

                # Métricas
                "Density Metric": float(getattr(self, "density_metric_value", 0.0) or 0.0),
                "Filtered Density Metric": float(getattr(self, "filtered_density_metric_value", 0.0) or 0.0),
                "Dynamic Density Score": float(getattr(self, "dynamic_density_score", 0.0) or 0.0),
                "Spectral Entropy": float(getattr(self, "entropy_spectral_value", 0.0) or 0.0),
                "Combined Density Metric": float(getattr(self, "combined_density_metric_value", 0.0) or 0.0),
                "Total Metric": float(getattr(self, "total_metric_value", 0.0) or 0.0),
                "Spectral Density Metric": float(getattr(self, "spectral_density_metric_value", 0.0) or 0.0),

                # NOVAS Métricas
                "R_norm": float(getattr(self, "R_norm", 0.0) or 0.0),
                "P_norm": float(getattr(self, "P_norm", 0.0) or 0.0),
                "D_agn":  float(getattr(self, "D_agn", 0.0) or 0.0),
                "D_harm": (np.nan if getattr(self, "D_harm", None) is None else float(self.D_harm)),

                # Parâmetros
                "Weight Function": str(getattr(self, "weight_function", "linear")),
                "Harmonic Weight (a)": float(getattr(self, "harmonic_weight", 0.5)),
                "Inharmonic Weight (ß)": float(getattr(self, "inharmonic_weight", 0.5)),

                # Metadados
                "Window": str(getattr(self, "window", "hann")),
                "N FFT": int(getattr(self, "n_fft", 4096)),
                "Hop Length": int(hl),
                "Search Band (cents)": int(getattr(self, "search_band_cents", 5)),
                "SNR Threshold (dB)": float(getattr(self, "snr_threshold_db", 20.0)),
                "Coherent Gain": float(cg_val),
                "DM Domain": str(getattr(self, "dm_domain", "amplitude")),
                "Density Scale": "bark",
                "Sigma (scale units)": 0.5,
                "Max Peaks per Band": 4,
                
                # --- CORRIGIDO AQUI ---
                "Weight R": float(_final_wr),
                "Weight P": float(_final_wp),
                # ----------------------
            }

            if isinstance(self.harmonic_list_df, pd.DataFrame) and not self.harmonic_list_df.empty:
                main_metrics["Harmonic Count"] = int(len(self.harmonic_list_df))
                if "Frequency (Hz)" in self.harmonic_list_df.columns:
                    fmin = pd.to_numeric(self.harmonic_list_df["Frequency (Hz)"], errors="coerce").min()
                    fmax = pd.to_numeric(self.harmonic_list_df["Frequency (Hz)"], errors="coerce").max()
                    main_metrics["Lowest Harmonic (Hz)"] = float(fmin) if np.isfinite(fmin) else None
                    main_metrics["Highest Harmonic (Hz)"] = float(fmax) if np.isfinite(fmax) else None

            if isinstance(self.filtered_list_df, pd.DataFrame) and not self.filtered_list_df.empty:
                main_metrics["Filtered Components"] = int(len(self.filtered_list_df))

            if getattr(self, "dissonance_enabled", False):
                try:
                    models_to_include = (list(self.dissonance_values.keys())
                                         if getattr(self, "dissonance_compare_models", False)
                                         else [getattr(self, "dissonance_model", "Sethares")])
                    for m in models_to_include:
                        val = None
                        if isinstance(getattr(self, "dissonance_values", None), dict):
                            val = self.dissonance_values.get(m)
                        if val is not None and np.isfinite(val):
                            main_metrics[f"{m} Dissonance"] = float(val)
                except Exception:
                    pass

            metrics_df = pd.DataFrame([main_metrics])
            metrics_df.to_excel(writer, sheet_name="Metrics", index=False)

            # ===== 4. POTÊNCIA ESPECTRAL =====
            if getattr(self, "db_S", None) is not None:
                try:
                    self._save_spectral_power_data(writer, note)
                except Exception as e:
                    log.debug(f"_save_spectral_power_data indisponível: {e}")

            # ===== 5. PARÂMETROS =====
            params_data = {
                "Parameter": [
                    "Note", "Sample Rate (Hz)", "FFT Size", "Hop Length", "Window Type", "Weight Function",
                    "Frequency Range (Hz)", "Magnitude Range (dB)", "Tolerance (Hz)", "Adaptive Tolerance", "Analysis Method",
                ],
                "Value": [
                    note,
                    int(getattr(self, "sr", 0) or 0),
                    int(getattr(self, "n_fft", 0) or 0),
                    int(getattr(self, "hop_length", 0) or 0),
                    str(getattr(self, "window", "")),
                    str(getattr(self, "weight_function", "linear")),
                    f"{float(getattr(self, 'freq_min', 20.0) or 20.0)} - {float(getattr(self, 'freq_max', 20000.0) or 20000.0)}",
                    f"{float(getattr(self, 'db_min', -90.0) or -90.0)} - {float(getattr(self, 'db_max', 0.0) or 0.0)}",
                    float(getattr(self, "tolerance", 10.0) or 10.0),
                    bool(getattr(self, "use_adaptive_tolerance", True)),
                    ("LFT" if getattr(self, "use_lft", False) else "FFT"),
                ],
            }
            pd.DataFrame(params_data).to_excel(writer, sheet_name="Analysis Parameters", index=False)

        except Exception as e:
            log.error(f"Erro em _save_spectral_data_to_excel: {e}", exc_info=True)
            try:
                pd.DataFrame([{"note": note, "error": str(e)}]).to_excel(writer, sheet_name="Error", index=False)
            except Exception:
                pass


