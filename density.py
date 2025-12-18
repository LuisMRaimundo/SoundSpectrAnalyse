# density.py - Corrected Version

"""
Module for calculating spectral density metrics for musical audio analysis.
Implements weight functions, density calculations, and combined metrics for
harmonic and inharmonic components.

Improvements:
- Expanded and standardized documentation
- Reinforced parameter validation
- More robust error handling
- Performance optimization in critical functions
- Consistent naming in English
"""

import numpy as np
import pandas as pd
from typing import Callable, Union, Optional, Dict, Tuple
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
#  Spectral-Density metrics (restaurado)
# ----------------------------------------------------------------------
class SpectralDensityMetrics:
    """
    Conjunto de métricas espectrais clássicas.
    Referências:
        * Krimphoff et al., 1994 – sparsity / concentration
        * Peeters et al., 2011 – timbre toolbox
        * Zwicker & Fastl, 1999  – densidade perceptual por bandas Bark
    """

    # ---------- 1) Sparsity (0 = denso ; 1 = esparso)
    @staticmethod
    def spectral_sparsity(amplitudes: np.ndarray,
                          frequencies: Optional[np.ndarray] = None) -> float:
        """
        Mede quão 'esparso' é o espectro. Valores altos indicam poucos bins
        acima de um limiar relativo; valores baixos indicam ocupação densa.
        """
        if amplitudes.size == 0:
            return 1.0

        # Normalização por pico para invariância a ganho
        amps = amplitudes.astype(float)
        amax = float(np.max(amps)) if amps.size else 0.0
        if amax > 0.0:
            amps = amps / amax

        # Limiar relativo (~ -40 dB)
        threshold = 0.01
        significant = int(np.sum(amps > threshold))

        if frequencies is None or frequencies.size == 0:
            return float(np.clip(1.0 - significant / max(amps.size, 1), 0.0, 1.0))

        # Se houver frequências, corrige a expectativa pelo espaçamento efetivo
        f = frequencies.astype(float)
        w = amps
        f_mean = float(np.average(f, weights=w)) if np.sum(w) > 0 else float(np.mean(f))
        f_std = float(np.sqrt(np.average((f - f_mean) ** 2, weights=w))) if np.sum(w) > 0 else float(np.std(f))
        bw_eff = 4.0 * f_std
        bw_nom = float(f[-1] - f[0]) if f.size > 1 else 0.0
        expected = (bw_eff / (bw_nom / f.size)) if (bw_nom > 0 and f.size > 0) else float(amps.size)
        return float(np.clip(1.0 - significant / max(expected, 1.0), 0.0, 1.0))

    # ---------- 2) Concentration (0 = difuso ; 1 = concentrado)
    @staticmethod
    def spectral_concentration(amplitudes: np.ndarray, n_peaks: int = 5) -> float:
        """
        Fração de energia nos n picos principais (com pequena correção por dimensão).
        """
        if amplitudes.size == 0:
            return 0.0
        a = amplitudes.astype(float)
        if not np.isfinite(a).any() or np.sum(a) <= 0:
            return 0.0

        # Ordenar por amplitude/energia
        sorted_amps = np.sort(a)[::-1]
        peak_e = float(np.sum(sorted_amps[:max(1, n_peaks)]))
        total_e = float(np.sum(sorted_amps))
        conc_raw = peak_e / total_e if total_e > 0 else 0.0

        # Penalização suave por dimensionalidade (evita triviais com poucos bins)
        if a.size > n_peaks:
            conc_raw *= (1.0 - n_peaks / float(a.size))
        return float(np.clip(conc_raw, 0.0, 1.0))

    # ---------- 3) Densidade espectral perceptual (0–1)
    @staticmethod
    def perceptual_spectral_density(amplitudes: np.ndarray,
                                    frequencies: np.ndarray) -> float:
        if amplitudes.size == 0 or frequencies.size == 0:
            return 0.0
        f = np.maximum(frequencies.astype(float), 1.0)

        def _equal_loudness_weight(freq_hz: np.ndarray) -> np.ndarray:
            w = np.ones_like(freq_hz, dtype=float)
            low  = freq_hz < 1000.0
            mid  = (freq_hz >= 1000.0) & (freq_hz <= 5000.0)
            high = freq_hz > 5000.0
            w[low]  = 0.5 + 0.5 * (freq_hz[low] / 1000.0)
            w[mid]  = 1.0
            w[high] = 1.0 - 0.5 * np.minimum((freq_hz[high] - 5000.0) / 15000.0, 1.0)
            return w

        w_eq = _equal_loudness_weight(f)
        amp = amplitudes.astype(float)
        pow_like  = np.square(amp)
        loud_spec = np.power(pow_like, 0.3) * w_eq  # |X|^0.6 ponderado

        bark = 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)
        bmin, bmax = int(np.floor(bark.min())), int(np.ceil(bark.max()))
        band_vals = []
        for b in range(bmin, bmax + 1):
            m = (bark >= b) & (bark < b + 1)
            if m.any():
                band_vals.append(loud_spec[m].sum() / (m.sum()))
        if not band_vals:
            return 0.0
        pd = float(np.mean(band_vals))
        denom = float(np.mean(loud_spec)) if np.mean(loud_spec) > 0 else 1.0
        return float(np.clip(pd / denom, 0.0, 1.0))



def calculate_harmonic_density(
    harmonic_amplitudes: np.ndarray,
    threshold_db: float = -60.0,
    fundamental_freq: float | None = None,
    sr: float | None = None,
    include_amp_factor: bool = True,
    amp_weight: float = 0.20,
    max_expected_harmonics: int | None = None
) -> float:
    if harmonic_amplitudes.size == 0:
        return 0.0

    # 1) máximo teórico dependente de f0
    if max_expected_harmonics is None and fundamental_freq and fundamental_freq > 0:
        nyq = (sr/2.0) if sr else 20000.0
        max_expected_harmonics = max(1, int(nyq // fundamental_freq))
    max_expected_harmonics = max_expected_harmonics or 50  # fallback

    # 2) contagem acima do threshold (em dB)
    amps_db = 20*np.log10(np.maximum(harmonic_amplitudes, 1e-12))
    significant = amps_db > threshold_db
    density_count = significant.sum() / max_expected_harmonics

    # 3) fator de amplitude (opcional e fraco)
    if include_amp_factor:
        avg_amp = np.mean(harmonic_amplitudes[significant]) if significant.any() else 0.0
        amp_factor = np.tanh(avg_amp)
        density = (1.0-amp_weight)*density_count + amp_weight*amp_factor
    else:
        density = density_count

    return float(np.clip(density, 0.0, 1.0))



def calculate_inharmonic_density(
    inharmonic_amplitudes: np.ndarray,
    threshold_db: float = -60.0,
    max_expected_partials: int = 50 # CORRECTED: Parameterized
) -> float:
    """
    Same as harmonic density, but for inharmonic components.
    """
    return calculate_harmonic_density(inharmonic_amplitudes, threshold_db=threshold_db, max_expected_harmonics=max_expected_partials)


def compute_spectral_entropy(power: np.ndarray) -> float:
    """
    Calcula a entropia espectral normalizada (Shannon entropy).
    
    Args:
        power: vetor de potências espectrais (amplitude^2 ou magnitude em dB convertido)
        
    Returns:
        Entropia espectral normalizada (0 = máximo foco, 1 = máxima dispersão)
    """
    if len(power) == 0:
        logger.warning("Array de potências vazio para entropia")
        return 0.0
    
    # Garantir que temos potências (valores positivos)
    power = np.abs(power)
    
    # Remover zeros e valores muito pequenos
    power = power[power > 1e-12]
    
    if len(power) == 0:
        logger.warning("Todas as potências são zero ou muito pequenas")
        return 0.0
    
    # Calcular soma total
    total_power = np.sum(power)
    
    if total_power <= 0:
        logger.warning("Potência total <= 0")
        return 0.0
    
    # Normalizar para distribuição de probabilidade
    p = power / total_power
    
    # Calcular entropia de Shannon
    entropy = -np.sum(p * np.log2(p))
    
    # Normalizar pela entropia máxima (distribuição uniforme)
    max_entropy = np.log2(len(power))
    
    if max_entropy > 0:
        normalized_entropy = entropy / max_entropy
    else:
        normalized_entropy = 0.0
    
    # Garantir intervalo [0, 1]
    normalized_entropy = np.clip(normalized_entropy, 0.0, 1.0)
    
    logger.debug(f"Entropia espectral: {normalized_entropy:.4f} (entropia: {entropy:.4f}, max: {max_entropy:.4f})")
    
    return normalized_entropy

def calculate_perceptual_spectral_density(
    harmonic_amplitudes: np.ndarray,
    harmonic_frequencies: np.ndarray,
    fundamental_freq: float,
    threshold_db: float = -60.0,
    frequency_limit: float = 20000.0
) -> float:
    """
    Calcula a densidade espectral perceptual baseada em princípios psicoacústicos.
    
    Esta métrica considera:
    1. Número de harmônicos audíveis presentes vs. possíveis
    2. Distribuição de energia ao longo do espectro
    3. Ponderação perceptual (curva de Fletcher-Munson simplificada)
    4. Mascaramento espectral
    
    Args:
        harmonic_amplitudes: Amplitudes dos harmônicos
        harmonic_frequencies: Frequências dos harmônicos
        fundamental_freq: Frequência fundamental
        threshold_db: Limiar de audibilidade
        frequency_limit: Limite superior de frequência (tipicamente 20kHz)
        
    Returns:
        Densidade espectral perceptual normalizada (0-1)
    """
    if len(harmonic_amplitudes) == 0 or fundamental_freq <= 0:
        return 0.0
    
    # 1. Converter amplitudes para dB se necessário
    if np.all(harmonic_amplitudes >= 0):
        harmonic_db = 20 * np.log10(np.maximum(harmonic_amplitudes, 1e-12))
    else:
        harmonic_db = harmonic_amplitudes
    
    # 2. Calcular número máximo teórico de harmônicos
    max_possible_harmonics = int(frequency_limit / fundamental_freq)
    
    # 3. Aplicar ponderação perceptual (A-weighting simplificado)
    # Mais peso para frequências entre 1-5 kHz onde o ouvido é mais sensível
    perceptual_weights = np.ones_like(harmonic_frequencies)
    for i, freq in enumerate(harmonic_frequencies):
        if freq < 1000:
            # Reduz peso para frequências graves
            perceptual_weights[i] = 0.5 + 0.5 * (freq / 1000)
        elif 1000 <= freq <= 5000:
            # Peso máximo na faixa mais sensível
            perceptual_weights[i] = 1.0
        else:
            # Reduz peso para frequências muito agudas
            perceptual_weights[i] = 1.0 - 0.5 * min((freq - 5000) / 15000, 1.0)
    
    # 4. Calcular harmônicos efetivos (acima do threshold e ponderados)
    effective_harmonics = 0
    total_weighted_energy = 0
    
    for i, (amp_db, weight) in enumerate(zip(harmonic_db, perceptual_weights)):
        if amp_db > threshold_db:
            # Contribuição ponderada para a contagem
            contribution = weight * (1 + (amp_db - threshold_db) / 60)  # Bônus por amplitude
            effective_harmonics += contribution
            
            # Energia ponderada
            total_weighted_energy += (10 ** (amp_db / 20)) * weight
    
    # 5. Calcular métricas componentes
    # a) Densidade de ocupação: proporção de harmônicos presentes
    occupancy_density = effective_harmonics / max_possible_harmonics
    
    # b) Uniformidade espectral: quão uniformemente distribuída está a energia
    if len(harmonic_db) > 1:
        # Calcular desvio padrão das amplitudes ponderadas
        weighted_amps = harmonic_db[harmonic_db > threshold_db] * perceptual_weights[harmonic_db > threshold_db]
        if len(weighted_amps) > 1:
            uniformity = 1.0 / (1.0 + np.std(weighted_amps) / 20)  # Normalizado
        else:
            uniformity = 0.5
    else:
        uniformity = 0.5
    
    # c) Fator de completude: penaliza gaps na série harmônica
    completeness = 1.0
    if len(harmonic_frequencies) > 1:
        # Verificar gaps entre harmônicos consecutivos
        expected_harmonics = np.arange(1, len(harmonic_frequencies) + 1) * fundamental_freq
        actual_harmonics = sorted(harmonic_frequencies)
        
        gaps = 0
        for i in range(1, min(len(expected_harmonics), 10)):  # Verificar primeiros 10
            expected = i * fundamental_freq
            # Procurar o harmônico mais próximo
            closest = min(actual_harmonics, key=lambda x: abs(x - expected))
            if abs(closest - expected) > fundamental_freq * 0.1:  # 10% de tolerância
                gaps += 1
        
        completeness = 1.0 - (gaps / 10)
    
    # 6. Combinar métricas com pesos baseados em pesquisa psicoacústica
    # Pesos derivados de estudos sobre percepção de riqueza tímbrica
    final_density = (
        0.5 * occupancy_density +      # Quantidade de harmônicos
        0.3 * uniformity +              # Distribuição de energia
        0.2 * completeness              # Completude da série
    )
    
    # 7. Aplicar correção logarítmica (Weber-Fechner)
    # A percepção de densidade não é linear
    perceptual_density = 1.0 - np.exp(-3 * final_density)
    
    return np.clip(perceptual_density, 0.0, 1.0)

def calculate_spectral_complexity(
    complete_spectrum_df: pd.DataFrame,
    fundamental_freq: float,
    bandwidth: Tuple[float, float] = (20, 20000)
) -> float:
    """
    Calcula a complexidade espectral total, incluindo componentes inarmônicos.
    
    Baseado em:
    - Krimphoff et al. (1994) - Caracterização do timbre
    - McAdams et al. (1995) - Espaço perceptual do timbre
    """
    if complete_spectrum_df.empty or fundamental_freq <= 0:
        return 0.0
    
    # Filtrar espectro na banda de interesse
    mask = (
        (complete_spectrum_df['Frequency (Hz)'] >= bandwidth[0]) & 
        (complete_spectrum_df['Frequency (Hz)'] <= bandwidth[1])
    )
    spectrum = complete_spectrum_df[mask].copy()
    
    if spectrum.empty:
        return 0.0
    
    # 1. Irregularidade espectral (Krimphoff)
    # Desvio do envelope espectral suave
    if 'Amplitude' in spectrum.columns:
        amps = spectrum['Amplitude'].values
    else:
        amps = 10 ** (spectrum['Magnitude (dB)'].values / 20)
    
    # Suavizar espectro com média móvel
    window_size = max(3, len(amps) // 20)
    if len(amps) > window_size:
        smooth_amps = np.convolve(amps, np.ones(window_size)/window_size, mode='same')
        irregularity = np.mean(np.abs(amps - smooth_amps)) / np.mean(amps)
    else:
        irregularity = 0.0
    
    # 2. Inharmonicidade
    # Proporção de energia em componentes não-harmônicos
    total_energy = np.sum(amps ** 2)
    
    # Identificar componentes harmônicos (dentro de 3% da série harmônica)
    harmonic_energy = 0
    for n in range(1, int(bandwidth[1] / fundamental_freq) + 1):
        expected_freq = n * fundamental_freq
        tolerance = expected_freq * 0.03
        
        harmonic_mask = (
            (spectrum['Frequency (Hz)'] >= expected_freq - tolerance) &
            (spectrum['Frequency (Hz)'] <= expected_freq + tolerance)
        )
        
        if harmonic_mask.any():
            harmonic_energy += np.sum(amps[harmonic_mask] ** 2)
    
    inharmonicity = 1.0 - (harmonic_energy / total_energy) if total_energy > 0 else 0.0
    
    # 3. Entropia espectral normalizada
    if total_energy > 0:
        probs = (amps ** 2) / total_energy
        probs = probs[probs > 1e-10]  # Evitar log(0)
        entropy = -np.sum(probs * np.log2(probs)) / np.log2(len(probs))
    else:
        entropy = 0.0
    
    # Combinar métricas
    complexity = (
        0.4 * irregularity +
        0.4 * inharmonicity +
        0.2 * entropy
    )
    
    return np.clip(complexity, 0.0, 1.0)


def calculate_harmonic_richness(
    harmonic_df: pd.DataFrame,
    max_expected_harmonics: int = 100, # CORRECTED: Parameterized
    amplitude_weight: float = 0.2
) -> float:
    """
    Calculates harmonic richness based mainly on the NUMBER of harmonics.

    Args:
        harmonic_df: DataFrame with harmonic data.
        max_expected_harmonics: The maximum expected number of harmonics for normalization.
        amplitude_weight: Weight to give to amplitude consideration (0-1).

    Returns:
        A value between 0 and 1, where 1 indicates a full and strong harmonic spectrum.
    """
    if harmonic_df is None or harmonic_df.empty:
        return 0.0

    # 1. Count factor (primary)
    num_harmonics = len(harmonic_df)
    count_factor = min(1.0, num_harmonics / max_expected_harmonics)

    # 2. Amplitude factor (secondary)
    amplitude_factor = 0.0
    if 'Amplitude' in harmonic_df.columns:
        # Use geometric mean of amplitudes (less sensitive to outliers)
        amps = harmonic_df['Amplitude'].values
        amps_positive = amps[amps > 0]
        if len(amps_positive) > 0:
            geometric_mean = np.exp(np.mean(np.log(amps_positive)))
            # Normalize assuming a reasonable max amplitude is 1.0
            amplitude_factor = np.tanh(geometric_mean)  # Saturate between 0-1

    # Combine factors
    count_weight = 1.0 - amplitude_weight
    richness = count_weight * count_factor + amplitude_weight * amplitude_factor

    logger.debug(f"Harmonic richness: {richness:.4f} (count: {count_factor:.4f}, amplitude: {amplitude_factor:.4f})")

    return richness


def calculate_spectral_density_corrected(
    spectrum_df: pd.DataFrame,
    freq_min: float = 20.0,
    freq_max: float = 20000.0,
    bin_width: float = 100.0
) -> float:
    """
    Calcula densidade espectral como número de bins ocupados no espectro.
    
    Args:
        spectrum_df: DataFrame com espectro completo
        freq_min: Frequência mínima
        freq_max: Frequência máxima
        bin_width: Largura de cada bin em Hz
        
    Returns:
        Densidade normalizada (0-1)
    """
    if spectrum_df is None or spectrum_df.empty:
        return 0.0
    
    # Filtrar espectro na faixa de interesse
    if 'Frequency (Hz)' in spectrum_df.columns:
        mask = (spectrum_df['Frequency (Hz)'] >= freq_min) & (spectrum_df['Frequency (Hz)'] <= freq_max)
        filtered = spectrum_df[mask]
    else:
        filtered = spectrum_df
    
    if filtered.empty:
        return 0.0
    
    # Calcular número de bins
    total_bins = int((freq_max - freq_min) / bin_width)
    
    # Contar bins ocupados
    occupied_bins = 0
    for bin_start in np.arange(freq_min, freq_max, bin_width):
        bin_end = bin_start + bin_width
        bin_mask = (filtered['Frequency (Hz)'] >= bin_start) & (filtered['Frequency (Hz)'] < bin_end)
        
        if bin_mask.any():
            # Verificar se há energia significativa no bin
            if 'Amplitude' in filtered.columns:
                bin_energy = filtered.loc[bin_mask, 'Amplitude'].sum()
                if bin_energy > 1e-6:  # Threshold mínimo
                    occupied_bins += 1
            else:
                occupied_bins += 1
    
    # Normalizar
    density = occupied_bins / total_bins if total_bins > 0 else 0.0
    
    logger.debug(f"Densidade espectral: {density:.4f} ({occupied_bins}/{total_bins} bins ocupados)")
    
    return density


class WeightFunction:
    @staticmethod
    def linear(x):
        return x

    @staticmethod
    def squared(x):
        return np.square(x)  # x^2

    @staticmethod
    def sqrt(x):
        return np.sqrt(x)

    @staticmethod
    def cbrt(x):
        return np.sign(x) * (np.abs(x) ** (1.0 / 3.0))

    @staticmethod
    def cubic(x):
        return x ** 3

    @staticmethod
    def logarithmic(x):
        return np.log1p(x)

    @staticmethod
    def exponential(x):
        return np.expm1(x)

    @staticmethod
    def inverse_log(x):
        eps = 1e-10
        return 1.0 / (np.log1p(x) + eps)

    @staticmethod
    def sum(x):
        # Identidade → a agregação final faz a soma
        return x


def get_weight_function(name: str) -> Callable:
    """
    Obtém a função de ponderação pelo nome.

    Args:
        name: Nome da função de ponderação (ex.: 'linear', 'sqrt', 'cbrt', 'exp').

    Returns:
        Função de ponderação correspondente.

    Raises:
        ValueError: Se o nome da função não for reconhecido.
    """
    weight_functions = {
        'linear':      WeightFunction.linear,
        'sqrt':        WeightFunction.sqrt,
        'squared':     WeightFunction.squared,
        'cbrt':        WeightFunction.cbrt,
        'cubic':       WeightFunction.cubic,
        'logarithmic': WeightFunction.logarithmic,
        'log':         WeightFunction.logarithmic,   # alias
        'exponential': WeightFunction.exponential,
        'exp':         WeightFunction.exponential,   # alias
        'inverse log': WeightFunction.inverse_log,
        'sum':         WeightFunction.sum,
    }

    key = (name or '').strip().lower()
    if key not in weight_functions:
        raise ValueError(f"Função de ponderação '{name}' não encontrada.")
    return weight_functions[key]



def apply_density_metric(values, weight_function='linear',
                        normalize=False, remove_noise=False):
    """
    Applies a weighting function to a set of values and aggregates them.

    Args:
        values: The input numpy array of amplitudes.
        weight_function: The name of the weighting function to apply.
        normalize: If True, normalizes the result by the number of values.
        remove_noise: If True, filters out low-level noise before calculation.
    """
    if remove_noise:
        # This could be configurable
        noise_threshold = 1e-6  # or passed as a parameter
        values = values[values > np.max(values) * noise_threshold]

    if values.size == 0:
        return 0.0

    # Apply weight function
    weight_func = get_weight_function(weight_function)
    weighted = weight_func(values)

    if normalize and len(values) > 0:
        return np.sum(weighted) / len(values)
    else:
        return np.sum(weighted)


def apply_density_metric_df(
    df: pd.DataFrame, 
    amplitude_column: str = 'Amplitude',
    weight_function: str = 'linear'
) -> float:
    """
    Calcula a métrica de densidade para um DataFrame de dados espectrais.
    
    Args:
        df: DataFrame contendo dados espectrais.
        amplitude_column: Nome da coluna contendo valores de amplitude.
        weight_function: Nome da função de ponderação a aplicar.
        
    Returns:
        Métrica de densidade calculada.
        
    Raises:
        ValueError: Se a coluna de amplitude não for encontrada ou a função de
                    ponderação não for válida.
    """
    if df is None or df.empty:
        logger.warning("DataFrame vazio ou None fornecido para apply_density_metric_df")
        return 0.0
    
    # Verificar se a coluna de amplitude existe
    if amplitude_column not in df.columns:
        # Tentar calcular a partir da magnitude (dB) se disponível
        if 'Magnitude (dB)' in df.columns:
            df = df.copy()
            df[amplitude_column] = 10 ** (df['Magnitude (dB)'] / 20)
            logger.info("Coluna de amplitude calculada a partir de 'Magnitude (dB)'")
        else:
            msg = f"Coluna '{amplitude_column}' não encontrada no DataFrame e 'Magnitude (dB)' também não está disponível"
            logger.error(msg)
            raise ValueError(msg)
    
    # Extrair valores de amplitude
    amplitude_values = df[amplitude_column].values
    
    # Calcular métrica de densidade
    return apply_density_metric(amplitude_values, weight_function)


def identify_inharmonic_partials(
    harmonic_df: pd.DataFrame,
    complete_df: pd.DataFrame,
    tolerance: Union[float, int] = 0.02
) -> pd.DataFrame:
    """
    Identifica parciais inarmônicos em um espectro completo.
    
    A tolerância é interpretada da seguinte forma:
    * Se tolerance >= 1.0: limiar absoluto em Hertz (ex.: 10 → ±10 Hz)
    * Se tolerance < 1.0: limiar relativo (proporção) (ex.: 0.02 → ±2%)
    
    Args:
        harmonic_df: DataFrame com os parciais harmônicos (coluna 'Frequency (Hz)').
        complete_df: DataFrame com todos os parciais.
        tolerance: Tolerância para considerar um parcial como harmônico.
        
    Returns:
        DataFrame contendo apenas os parciais inarmônicos.
        
    Raises:
        ValueError: Se os DataFrames não contiverem a coluna 'Frequency (Hz)'.
    """
    # Validação de entrada
    if harmonic_df is None or harmonic_df.empty or complete_df is None or complete_df.empty:
        logger.warning("DataFrame harmônico ou completo vazio em identify_inharmonic_partials")
        empty_df = pd.DataFrame(columns=complete_df.columns if complete_df is not None 
                                and not complete_df.empty else ['Frequency (Hz)'])
        return empty_df

    # Verificar presença da coluna de frequência
    for df, name in [(harmonic_df, "harmônico"), (complete_df, "completo")]:
        if 'Frequency (Hz)' not in df.columns:
            msg = f"Coluna 'Frequency (Hz)' não encontrada no DataFrame {name}"
            logger.error(msg)
            raise ValueError(msg)

    # Extrair arrays de frequência
    try:
        harm_freqs = harmonic_df["Frequency (Hz)"].to_numpy()
        all_freqs = complete_df["Frequency (Hz)"].to_numpy()
    except Exception as e:
        logger.error(f"Erro ao extrair frequências dos DataFrames: {e}")
        raise

    # Máscara booleana: começa por assumir que TODOS são inarmônicos
    inharmonic_mask = np.ones_like(all_freqs, dtype=bool)

    # Itera sobre cada harmônico e "desmarca" quem cair dentro da tolerância
    for f0 in harm_freqs:
        if tolerance < 1.0:  # limiar relativo
            thr = np.maximum(f0 * tolerance, 1e-6)  # piso 1 mHz
        else:  # limiar absoluto
            thr = tolerance
        inharmonic_mask &= np.abs(all_freqs - f0) > thr

    # Aplicar a máscara e retornar apenas os parciais inarmônicos
    return complete_df.loc[inharmonic_mask].reset_index(drop=True)


def calculate_combined_density_metric(
    harmonic_density: float,
    inharmonic_density: float,
    alpha: float = 0.8,
    beta: float = 0.2
) -> float:
    """
    Combines harmonic and inharmonic densities transparently.
    """
    # Normalize weights to sum to 1
    total_weight = alpha + beta
    if total_weight > 0 and not np.isclose(total_weight, 1.0):
        alpha = alpha / total_weight
        beta = beta / total_weight

    # Calculate combined metric
    combined = alpha * harmonic_density + beta * inharmonic_density

    return combined


def compare_with_sethares_dissonance(
    harmonic_df: pd.DataFrame,
    sethares_dissonance: float,
    density_metric: float,
    output_path: Optional[str] = None
) -> Dict[str, float]:
    """
    Compara a métrica de densidade tradicional com a dissonância de Sethares.
    
    Args:
        harmonic_df: DataFrame com parciais harmônicos.
        sethares_dissonance: Valor de dissonância de Sethares calculado.
        density_metric: Valor de métrica de densidade tradicional.
        output_path: Caminho para salvar o gráfico de comparação.
        
    Returns:
        Dicionário com métricas de comparação.
    """
    if harmonic_df is None or harmonic_df.empty or sethares_dissonance is None or density_metric is None:
        logger.warning("Dados inválidos fornecidos para compare_with_sethares_dissonance")
        return {'correlation': 0.0, 'ratio': 0.0}
    
    # Normalizar ambas as métricas para a faixa 0-1 para comparação
    norm_sethares = sethares_dissonance / 10  # Assumindo que Sethares é escalado por 10
    norm_density = density_metric / 10  # Assumindo que a densidade é escalada por 10
    
    # Calcular relação entre métricas
    ratio = norm_sethares / norm_density if norm_density > 0 else 0.0
    
    # Criar gráfico de comparação se o caminho for fornecido
    if output_path:
        plt.figure(figsize=(10, 6))
        
        # Gráfico de barras comparando métricas
        metrics = ['Density Metric', 'Sethares Dissonance']
        values = [norm_density, norm_sethares]
        
        plt.bar(metrics, values, color=['blue', 'red'])
        plt.title('Comparison of Density Metric and Sethares Dissonance')
        plt.ylabel('Normalized Value (0-1)')
        plt.ylim(0, 1.1)  # Adicionar algum espaço
        
        # Adicionar valores acima das barras
        for i, v in enumerate(values):
            plt.text(i, v + 0.05, f"{v:.3f}", ha='center')
        
        plt.tight_layout()
        
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico de comparação salvo em {output_path}")
        except Exception as e:
            logger.error(f"Erro ao salvar o gráfico de comparação: {e}")
        finally:
            plt.close()
    
    return {
        'normalized_density': norm_density,
        'normalized_sethares': norm_sethares,
        'ratio': ratio
    }


def plot_harmonic_spectrum(
    harmonic_df: pd.DataFrame,
    density_metric: float,
    sethares_dissonance: Optional[float] = None,
    output_path: Optional[str] = None,
    note_name: str = ""
) -> None:
    """
    Plota o espectro harmônico com métricas de densidade e dissonância.
    
    Args:
        harmonic_df: DataFrame com parciais harmônicos.
        density_metric: Valor de métrica de densidade tradicional.
        sethares_dissonance: Valor de dissonância de Sethares.
        output_path: Caminho para salvar o gráfico.
        note_name: Nome da nota para o título do gráfico.
        
    Raises:
        ValueError: Se o DataFrame for inválido.
    """
    if harmonic_df is None or harmonic_df.empty:
        logger.warning("DataFrame vazio ou None fornecido para plot_harmonic_spectrum")
        return
    
    plt.figure(figsize=(12, 6))
    
    try:
        # Extrair frequências e amplitudes
        frequencies = harmonic_df['Frequency (Hz)'].values
        
        if 'Amplitude' in harmonic_df.columns:
            amplitudes = harmonic_df['Amplitude'].values
        elif 'Magnitude (dB)' in harmonic_df.columns:
            # Converter de dB para amplitude linear
            amplitudes = 10 ** (harmonic_df['Magnitude (dB)'].values / 20)
        else:
            msg = "Nem 'Amplitude' nem 'Magnitude (dB)' encontrados no DataFrame"
            logger.error(msg)
            raise ValueError(msg)
        
        # Plotar harmônicos como linhas verticais (stem plot)
        plt.stem(frequencies, amplitudes, basefmt=' ')
        
        # Configurar rótulos de eixos e título
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        
        title = f"Harmonic Spectrum - {note_name}" if note_name else "Harmonic Spectrum"
        
        # Adicionar métricas ao título
        metrics_text = f"Density: {density_metric:.3f}"
        if sethares_dissonance is not None:
            metrics_text += f", Sethares Dissonance: {sethares_dissonance:.3f}"
        
        plt.title(f"{title}\n{metrics_text}")
        
        # Configurar escala logarítmica no eixo x para melhor visualização
        plt.xscale('log')
        plt.grid(True, alpha=0.3)
        
        # Salvar ou mostrar
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Espectro harmônico salvo em {output_path}")
            plt.close()
        else:
            plt.show()
            plt.close()
            
    except Exception as e:
        logger.error(f"Erro ao plotar espectro harmônico: {e}")
        plt.close()
        raise

# Funções auxiliares necessárias (caso não estejam importadas no escopo)
def _hz_to_bark(f):
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

def spectral_density(
    freqs_hz, amps, f0_hz=None,
    # proximidade
    proximity_axis="bark",   # "bark" recomendado
    sigma=0.5,               # 0.5 Bark
    bark_window=8.0,         # usar [f0, f0+8 Bark]
    max_peaks_per_band=4,    # cap por banda Bark
    # pesos R/P
    weight_r=0.45, weight_p=0.55,
    # termo de “peso” (baixo-freq)
    lambda_low=0.35,         # mistura com W
    low_bark_cut=8,          # Bark ≤ 8 contam como “baixos”
    q=1.0, gamma=1.0,
):
    freqs_hz = np.asarray(freqs_hz, float)
    amps = np.asarray(amps, float)
    mask = (freqs_hz > 0) & (amps > 0) & np.isfinite(freqs_hz) & np.isfinite(amps)
    freqs_hz, amps = freqs_hz[mask], amps[mask]
    
    if freqs_hz.size == 0:
        return dict(R_norm=0.0, P_norm=0.0, W_low=0.0, D_agn=0.0, D_peso=0.0, D_harm=None)

    # pesos normalizados (potência^gamma)
    p = (amps**gamma)
    p_sum = p.sum()
    if p_sum > 0:
        p = p / p_sum
    else:
        return dict(R_norm=0.0, P_norm=0.0, W_low=0.0, D_agn=0.0, D_peso=0.0, D_harm=None)

    # --- coordenada Bark e janela relativa a f0 ---
    u_bark = _hz_to_bark(freqs_hz)
    if (f0_hz is not None) and np.isfinite(f0_hz) and f0_hz > 0:
        u0_arr = _hz_to_bark(np.array([f0_hz]))
        if u0_arr.size > 0:
            u0 = float(u0_arr[0])
            win = (u_bark >= u0) & (u_bark <= u0 + float(bark_window))
            if win.any():
                u_bark, p, freqs_hz = u_bark[win], p[win], freqs_hz[win]
                # Re-normalizar p após janela? Geralmente sim para métricas de distribuição
                p = p / p.sum()

    # cap por banda Bark
    if max_peaks_per_band and max_peaks_per_band > 0:
        bands = np.floor(u_bark + 0.5)
        keep = np.zeros_like(bands, dtype=bool)
        unique_bands = np.unique(bands)
        for b in unique_bands:
            idx = np.where(bands == b)[0]
            if idx.size > max_peaks_per_band:
                # Manter os picos com maior amplitude
                idx = idx[np.argsort(p[idx])[::-1][:max_peaks_per_band]]
            keep[idx] = True
        
        if keep.any():
            u_bark, p, freqs_hz = u_bark[keep], p[keep], freqs_hz[keep]
            p = p / p.sum()

    M = p.size
    # --- R (riqueza efetiva, Hill q=1) ---
    if M <= 1:
        R_norm = 0.0
    else:
        # Proteção numérica no log
        p_safe = np.clip(p, 1e-12, 1.0)
        if abs(q - 1.0) < 1e-12:
            H = -np.sum(p * np.log(p_safe))
            N_eff = np.exp(H)
        else:
            denom = 1.0 - q
            if denom == 0: denom = 1e-12 # Should correspond to q=1 case, but safeguard
            N_eff = np.power(np.sum(np.power(p, q)), 1.0 / denom)
            
        N_eff = float(N_eff)
        # R_norm pode ser NaN se M=1, mas já tratámos M<=1
        R_norm = (N_eff - 1.0) / (M - 1.0)
        R_norm = max(0.0, min(1.0, R_norm))

    # --- P (proximidade em Bark) ---
    if M <= 1:
        P_norm = 0.0
    else:
        # Matriz de distâncias
        d = np.abs(u_bark[:, None] - u_bark[None, :])
        # Ignorar auto-distância na soma ou usar exp(0)=1? 
        # Fórmula original usava fill_diagonal infinity para zerar K
        np.fill_diagonal(d, np.inf)
        
        K = np.exp(-(d**2) / (2.0 * float(sigma)**2))
        
        # Numerador: soma ponderada das proximidades
        # Mask diagonal already handled by d=inf -> K=0
        P_num = float(np.sum((p[:, None] * p[None, :]) * K))
        
        # Denominador: Máximo possível (Simpsons index complement)
        P_den = float(1.0 - np.sum(p**2))
        
        if P_den <= 1e-12:
            P_norm = 0.0
        else:
            P_norm = min(P_num / P_den, 1.0)

    # --- W (peso baixo-freq: partilha em Bark baixos) ---
    bark_idx = np.clip(np.floor(u_bark + 0.5).astype(int), 1, 24)
    E_band = {}
    for bi, pi in zip(bark_idx, p):
        E_band[bi] = E_band.get(bi, 0.0) + float(pi)
        
    num = sum(v for k, v in E_band.items() if k <= int(low_bark_cut))
    den = sum(E_band.values())
    if den == 0: den = 1.0
    W_low = float(num / den)  # 0..1

    # --- COMBINAÇÕES (CORRIGIDO) ---
    wr, wp = float(weight_r), float(weight_p)
    
    # [FIX] REMOVIDA A NORMALIZAÇÃO FORÇADA
    # Antes: s = wr + wp; wr = wr/s... (Isto matava o "Equal Power")
    # Agora: Aceitamos os pesos como vêm. Se a soma for > 1 (Log mode), 
    # o resultado D_core aumenta, compensando a queda perceptiva.
    
    # Proteção básica apenas contra negativos
    wr = max(0.0, wr)
    wp = max(0.0, wp)
    
    # Se ambos forem zero (erro de input), usamos 0.5 default
    if wr == 0 and wp == 0:
        wr, wp = 0.5, 0.5

    D_core = wr * R_norm + wp * P_norm
    
    # Lambda mistura o resultado core com o peso de graves
    lam = float(lambda_low)
    lam = max(0.0, min(1.0, lam)) if np.isfinite(lam) else 0.0
    
    D_peso = (1.0 - lam) * D_core + lam * W_low

    return dict(
        R_norm=float(R_norm), 
        P_norm=float(P_norm),
        W_low=float(W_low), 
        D_agn=float(D_core), 
        D_peso=float(D_peso),
        D_harm=None
    )

# --- FIM NOVO ---


# Exportar funções públicas do módulo
__all__ = [
    # Classes
    'SpectralDensityMetrics',
    'WeightFunction',
    
    # Funções principais
        # Funções principais
    'apply_density_metric',
    'apply_density_metric_df',
    'calculate_harmonic_density',
    'calculate_inharmonic_density',
    'compute_spectral_entropy',
    'calculate_combined_density_metric',
    'calculate_perceptual_spectral_density',
    'calculate_spectral_complexity',
    'calculate_harmonic_richness',
    'calculate_spectral_density_corrected',
    'spectral_density',              # <-- NOVO

    
    # Funções auxiliares
    'get_weight_function',
    'identify_inharmonic_partials',
    'compare_with_sethares_dissonance',
    'plot_harmonic_spectrum'
]

