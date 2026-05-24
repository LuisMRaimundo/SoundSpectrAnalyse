# density.py - Corrected Version

from __future__ import annotations

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
from typing import Callable, Union, Optional, Dict, Tuple, List, Any, Literal
import logging
import warnings

logger = logging.getLogger(__name__)

# PHASE 3: Import constants
try:
    from constants import (
        SPARSITY_THRESHOLD_RELATIVE, SPECTRAL_CONCENTRATION_DEFAULT_PEAKS,
        PERCEPTUAL_DENSITY_POWER_EXPONENT, PERCEPTUAL_DENSITY_OCCUPANCY_WEIGHT,
        PERCEPTUAL_DENSITY_UNIFORMITY_WEIGHT, PERCEPTUAL_DENSITY_COMPLETENESS_WEIGHT,
        PERCEPTUAL_DENSITY_LOG_SCALE_FACTOR,
        HARMONIC_DETECTION_THRESHOLD_DB, HARMONIC_TOLERANCE_BASE,
        HARMONIC_TOLERANCE_ADAPTIVE_FACTOR, HARMONIC_MAX_CHECK,
        NUM_CRITICAL_BANDS, CRITICAL_BAND_MASKING_STRONG_THRESHOLD,
        CRITICAL_BAND_MASKING_MODERATE_THRESHOLD, CRITICAL_BAND_MASKING_WEAK_THRESHOLD,
        MASKING_WITHIN_BAND_OFFSET_DB, MASKING_ADJACENT_BAND_OFFSET_DB,
        MASKING_ADJACENT_BAND_SLOPE_DB, MASKING_NEARBY_BAND_OFFSET_DB,
        MASKING_NEARBY_BAND_SLOPE_DB, MASKING_FAR_BAND_OFFSET_DB,
        MASKING_FAR_BAND_SLOPE_DB, MASKING_ABSOLUTE_THRESHOLD_DB,
        # REMOVIDO: Equal loudness constants (não mais usadas - densidade física)
        # EQUAL_LOUDNESS_LOW_WEIGHT_MIN, EQUAL_LOUDNESS_HIGH_WEIGHT_MAX,
        # EQUAL_LOUDNESS_HIGH_WEIGHT_DECAY, EQUAL_LOUDNESS_HIGH_FREQ_RANGE,
        FREQ_MID_LOW_HZ, FREQ_MID_HIGH_HZ,  # Ainda usadas em outras funções
        SMOOTHING_WINDOW_PERCENTAGE, SMOOTHING_MIN_WINDOW_LENGTH,
        SMOOTHING_POLYORDER, SMOOTHING_NOISE_FLOOR_PERCENTILE, SMOOTHING_NOISE_FLOOR_MULTIPLIER,
        EPSILON, EPSILON_POWER, EPSILON_AMPLITUDE, EPSILON_FREQUENCY,
        HARMONIC_COMPLETENESS_WEIGHT_BASE
    )
except ImportError:
    # Fallback if constants.py not available
    SPARSITY_THRESHOLD_RELATIVE = 0.01
    SPECTRAL_CONCENTRATION_DEFAULT_PEAKS = 5
    PERCEPTUAL_DENSITY_POWER_EXPONENT = 0.3
    PERCEPTUAL_DENSITY_OCCUPANCY_WEIGHT = 0.5
    PERCEPTUAL_DENSITY_UNIFORMITY_WEIGHT = 0.3
    PERCEPTUAL_DENSITY_COMPLETENESS_WEIGHT = 0.2
    PERCEPTUAL_DENSITY_LOG_SCALE_FACTOR = 3.0
    HARMONIC_DETECTION_THRESHOLD_DB = -60.0
    HARMONIC_TOLERANCE_BASE = 0.1
    HARMONIC_TOLERANCE_ADAPTIVE_FACTOR = 0.1
    HARMONIC_MAX_CHECK = 100
    NUM_CRITICAL_BANDS = 24
    CRITICAL_BAND_MASKING_STRONG_THRESHOLD = 0.5
    CRITICAL_BAND_MASKING_MODERATE_THRESHOLD = 1.0
    CRITICAL_BAND_MASKING_WEAK_THRESHOLD = 2.0
    MASKING_WITHIN_BAND_OFFSET_DB = -10.0
    MASKING_ADJACENT_BAND_OFFSET_DB = -15.0
    MASKING_ADJACENT_BAND_SLOPE_DB = -10.0
    MASKING_NEARBY_BAND_OFFSET_DB = -20.0
    MASKING_NEARBY_BAND_SLOPE_DB = -5.0
    MASKING_FAR_BAND_OFFSET_DB = -30.0
    MASKING_FAR_BAND_SLOPE_DB = -2.0
    MASKING_ABSOLUTE_THRESHOLD_DB = -80.0
    # REMOVIDO: Equal loudness constants (não mais usadas - densidade física)
    # EQUAL_LOUDNESS_LOW_WEIGHT_MIN = 0.5
    # EQUAL_LOUDNESS_HIGH_WEIGHT_MAX = 1.0
    # EQUAL_LOUDNESS_HIGH_WEIGHT_DECAY = 0.5
    # EQUAL_LOUDNESS_HIGH_FREQ_RANGE = 15000.0
    # FREQ_MID_LOW_HZ = 1000.0
    # FREQ_MID_HIGH_HZ = 5000.0
    SMOOTHING_WINDOW_PERCENTAGE = 0.05
    SMOOTHING_MIN_WINDOW_LENGTH = 11
    SMOOTHING_POLYORDER = 3
    SMOOTHING_NOISE_FLOOR_PERCENTILE = 15.0
    SMOOTHING_NOISE_FLOOR_MULTIPLIER = 1.5
    EPSILON = 1e-12
    EPSILON_POWER = 1e-12
    EPSILON_AMPLITUDE = 1e-20
    EPSILON_FREQUENCY = 1e-6
    HARMONIC_COMPLETENESS_WEIGHT_BASE = 1.0

# ======================================================================
# PHASE 1: Spectral Smoothing Functions
# ======================================================================

def apply_spectral_smoothing(
    spectrum_magnitude: np.ndarray,
    method: str = "savitzky_golay",
    window_length: Optional[int] = None,
    polyorder: int = 3,
    noise_floor_percentile: float = 15.0,
    noise_floor_multiplier: float = 1.5
) -> np.ndarray:
    """
    Apply spectral smoothing to reduce narrow high-frequency noise peaks and noise.
    
    Spectral-smoothing helper.
    
    This function applies smoothing to the magnitude spectrum before
    temporal aggregation to reduce isolated narrow peaks that contradict
    expected harmonic spectrum behavior.
    
    Args:
        spectrum_magnitude: 2D array of magnitude spectrum (freq x time)
        method: Smoothing method ('savitzky_golay' or 'moving_average')
        window_length: Smoothing window length (auto-calculated if None)
        polyorder: Polynomial order for Savitzky-Golay (must be < window_length)
        noise_floor_percentile: Percentile for noise floor estimation
        noise_floor_multiplier: Multiplier for noise floor threshold
        
    Returns:
        Smoothed magnitude spectrum (same shape as input)
    """
    if spectrum_magnitude.size == 0:
        return spectrum_magnitude
    
    spectrum = np.asarray(spectrum_magnitude, dtype=float)
    
    # Handle 1D case (single time frame)
    if spectrum.ndim == 1:
        spectrum = spectrum[:, np.newaxis]
        was_1d = True
    else:
        was_1d = False
    
    n_freq, n_time = spectrum.shape
    
    # Auto-calculate window length if not provided
    if window_length is None:
        # Use 5% of spectrum length, minimum 11, must be odd
        window_length = max(11, int(n_freq * 0.05))
        if window_length % 2 == 0:
            window_length += 1
        window_length = min(window_length, n_freq - 1)
    
    # Ensure window_length is odd and valid
    if window_length % 2 == 0:
        window_length += 1
    window_length = max(3, min(window_length, n_freq - 1))
    
    # PHASE 3: Use constant instead of magic number
    # Ensure polyorder < window_length
    polyorder = min(SMOOTHING_POLYORDER, window_length - 1)
    
    smoothed = np.zeros_like(spectrum)
    
    try:
        if method == "savitzky_golay":
            # Use Savitzky-Golay filter (preferred for preserving peaks)
            try:
                from scipy.signal import savgol_filter
                
                for t in range(n_time):
                    # FIX: Apply double-pass smoothing to reduce artifacts
                    # First pass: standard Savitzky-Golay
                    smoothed_pass1 = savgol_filter(
                        spectrum[:, t],
                        window_length=window_length,
                        polyorder=polyorder,
                        mode='nearest'  # Handle boundaries
                    )
                    # Second pass: lighter smoothing to remove residual artifacts
                    # Use smaller window for second pass (half size, must be odd)
                    window_length_2 = max(5, (window_length // 2) | 1)  # Ensure odd
                    polyorder_2 = min(2, window_length_2 - 1)
                    smoothed[:, t] = savgol_filter(
                        smoothed_pass1,
                        window_length=window_length_2,
                        polyorder=polyorder_2,
                        mode='nearest'
                    )
                
                logger.debug(
                    f"Applied Savitzky-Golay smoothing: window={window_length}, "
                    f"polyorder={polyorder}"
                )
            except ImportError:
                logger.warning("scipy not available, falling back to moving average")
                method = "moving_average"
        
        if method == "moving_average":
            # Use moving average (fallback if scipy not available)
            try:
                from scipy.ndimage import uniform_filter1d
                
                for t in range(n_time):
                    smoothed[:, t] = uniform_filter1d(
                        spectrum[:, t],
                        size=window_length,
                        mode='nearest'
                    )
                
                logger.debug(f"Applied moving average smoothing: window={window_length}")
            except ImportError:
                # Pure NumPy implementation if scipy not available
                logger.warning("scipy.ndimage not available, using pure NumPy moving average")
                for t in range(n_time):
                    # Simple moving average using convolution
                    kernel = np.ones(window_length) / window_length
                    smoothed[:, t] = np.convolve(spectrum[:, t], kernel, mode='same')
        
        # PHASE 3: Use constants instead of magic numbers (if not provided)
        # PHASE 3: Use constants if not provided
        # Noise floor removal
        if noise_floor_percentile is None or noise_floor_percentile <= 0:
            noise_floor_percentile = SMOOTHING_NOISE_FLOOR_PERCENTILE
        if noise_floor_multiplier is None or noise_floor_multiplier <= 0:
            noise_floor_multiplier = SMOOTHING_NOISE_FLOOR_MULTIPLIER
            
        if noise_floor_percentile > 0:
            # FIX: Improved noise floor removal with adaptive threshold
            # This prevents removal of valid components at specific frequencies
            # Divide spectrum into bands and estimate noise floor per band
            n_bands = min(10, n_freq // 100)  # Adaptive number of bands
            if n_bands > 1:
                band_size = n_freq // n_bands
                noise_floors = []
                
                for b in range(n_bands):
                    start_idx = b * band_size
                    end_idx = (b + 1) * band_size if b < n_bands - 1 else n_freq
                    band_data = smoothed[start_idx:end_idx, :]
                    if band_data.size > 0:
                        band_noise = np.percentile(band_data, noise_floor_percentile)
                        noise_floors.append(band_noise)
                
                if len(noise_floors) > 0:
                    # Use median of band noise floors (more robust than global)
                    global_noise_floor = np.median(noise_floors)
                else:
                    global_noise_floor = np.percentile(smoothed, noise_floor_percentile)
            else:
                # Fallback to global estimation
                global_noise_floor = np.percentile(smoothed, noise_floor_percentile)
            
            # FIX: Use adaptive threshold with smooth rolloff instead of hard cutoff
            noise_threshold = global_noise_floor * noise_floor_multiplier
            
            # FIX: Apply smooth rolloff instead of hard cutoff to prevent artifacts
            # Use sigmoid-like function for smooth transition
            # Mathematical verification (reference):
            # Avoid division by zero: when noise_threshold ≈ 0, use direct clipping
            excess = smoothed - noise_threshold
            if noise_threshold > 1e-10:  # Avoid division by zero
                # Smooth rolloff: keep everything above threshold, gradual reduction below
                rolloff_factor = np.where(
                    excess > 0,
                    1.0,  # Above threshold: keep
                    np.maximum(0.0, 1.0 + excess / (noise_threshold * 0.5))  # Below: gradual reduction
                )
                smoothed = smoothed * rolloff_factor
            else:
                # If noise threshold is too small, just clip negative values
                smoothed = np.maximum(smoothed, 0.0)
            
            logger.debug(
                f"Noise floor removal: percentile={noise_floor_percentile}%, "
                f"threshold={noise_threshold:.6e}, bands={n_bands}"
            )
        
        # Restore original shape
        if was_1d:
            smoothed = smoothed[:, 0]
        
        return smoothed
        
    except Exception as e:
        logger.warning(f"Spectral smoothing failed: {e}, returning original spectrum")
        return spectrum_magnitude


def estimate_noise_floor(
    psd: np.ndarray,
    percentile: float = 15.0
) -> float:
    """
    Estimate noise floor from PSD using percentile method.
    
    Noise-floor estimation helper.
    
    Args:
        psd: Power spectral density array
        percentile: Percentile to use for noise floor estimation
        
    Returns:
        Estimated noise floor value
    """
    if psd.size == 0:
        return 0.0
    
    psd_flat = np.asarray(psd).flatten()
    psd_positive = psd_flat[psd_flat > 0]
    
    if len(psd_positive) == 0:
        return 0.0
    
    noise_floor = np.percentile(psd_positive, percentile)
    return float(noise_floor)

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

        # PHASE 3: Use constant instead of magic number
        # Limiar relativo (~ -40 dB)
        threshold = SPARSITY_THRESHOLD_RELATIVE
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
    def spectral_concentration(amplitudes: np.ndarray, n_peaks: int = SPECTRAL_CONCENTRATION_DEFAULT_PEAKS) -> float:
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

    # ---------- 3) Physical Spectral Density (PSD-integrated bandwidth occupancy)
    @staticmethod
    def physical_spectral_density(amplitudes: np.ndarray,
                                  frequencies: np.ndarray,
                                  bin_width_hz: Optional[float] = None) -> float:  # noqa: ARG001
        """
        Computes effective partial density as ``N_eff / N``.

        This is the Hill diversity index with q = 2 (inverse Herfindahl),
        normalized by component count N (Hill, 1973; Jost, 2006).
        It quantifies the effective number of active partials relative to
        the observed component count.

        This diverges conceptually from "classical" spectral-density framings
        associated with Krimphoff et al. (1994) and Peeters et al. (2011),
        which the previous naming could suggest.

        See: docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md
        ("Naming caveat: effective_partial_density vs. classical density").
        """
        if amplitudes is None or amplitudes.size == 0:
            return 0.0
        if bin_width_hz is not None:
            warnings.warn(
                "'bin_width_hz' is deprecated and ignored in physical_spectral_density; "
                "it will be removed in a 4.x release.",
                DeprecationWarning,
                stacklevel=2,
            )

        amp = np.asarray(amplitudes, dtype=float)
        amp = amp[np.isfinite(amp) & (amp > 0.0)]
        if amp.size == 0:
            return 0.0

        power = np.square(amp)
        total_power = float(np.sum(power))
        if total_power <= 0.0:
            return 0.0

        # Hill q=2 = 1 / Σ p_i² ; equivalently (Σ p)^2 / Σ p^2 (numerically safer).
        n_eff = float((total_power ** 2) / float(np.sum(power * power)))
        n_components = float(amp.size)
        score = n_eff / n_components if n_components > 0 else 0.0
        return float(np.clip(score, 0.0, 1.0))

    # ---------- 4) Perceptual Spectral Density (Bark scale with justified parameters)
    @staticmethod
    def perceptual_spectral_density(amplitudes: np.ndarray,
                                    frequencies: np.ndarray) -> float:
        """
        FIX 2 (re-attached): this method was previously orphaned inside the
        module-level wrapper for `physical_spectral_density` (after its
        `return`), so `hasattr(SpectralDensityMetrics, "perceptual_spectral_density")`
        was False. Re-anchored as a real `@staticmethod` of the class.

        Mathematical foundation (Zwicker & Terhardt, 1980):
            B(f) = 13*arctan(0.00076*f) + 3.5*arctan((f/7500)**2)

        Combines occupancy (fraction of active Bark bands) with the entropy
        of the energy distribution across the active bands.
        """
        if amplitudes.size == 0 or frequencies.size == 0:
            return 0.0

        f = np.maximum(frequencies.astype(float), 1.0)
        amp = amplitudes.astype(float)
        power = np.square(amp)

        bark = 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

        bmin, bmax = int(np.floor(bark.min())), int(np.ceil(bark.max()))
        bmin = max(0, bmin)
        bmax = min(24, bmax)

        band_energies = np.zeros(bmax - bmin + 1)
        for i, b in enumerate(range(bmin, bmax + 1)):
            band_center = b + 0.5
            distances = np.abs(bark - band_center)
            weights = np.maximum(0.0, 1.0 - distances)
            band_energies[i] = np.sum(power * weights)

        active_bands = int(np.sum(band_energies > 0))
        total_bands = len(band_energies)
        occupancy = active_bands / total_bands if total_bands > 0 else 0.0

        total_energy = float(np.sum(band_energies))
        if total_energy > 0:
            band_fractions = band_energies[band_energies > 0] / total_energy
            entropy = -np.sum(band_fractions * np.log2(band_fractions + 1e-10))
            max_entropy = np.log2(len(band_fractions)) if len(band_fractions) > 0 else 1.0
            uniformity = entropy / max_entropy if max_entropy > 0 else 0.0
        else:
            uniformity = 0.0

        # Unconstrained design choice: occupancy/uniformity blend has no direct literature fit yet.
        # Weights (0.6, 0.4) should be treated as sensitivity-analysis candidates.
        density = 0.6 * occupancy + 0.4 * uniformity
        return float(np.clip(density, 0.0, 1.0))


def physical_spectral_density(
    amplitudes: np.ndarray,
    frequencies: np.ndarray,
    bin_width_hz: Optional[float] = None
) -> float:
    """
    Module-level wrapper for physical_spectral_density.

    Keeps backward compatibility with imports like:
    `from density import physical_spectral_density`.
    """
    return SpectralDensityMetrics.physical_spectral_density(
        amplitudes=amplitudes,
        frequencies=frequencies,
        bin_width_hz=bin_width_hz
    )


def perceptual_spectral_density(
    amplitudes: np.ndarray,
    frequencies: np.ndarray,
) -> float:
    """Module-level wrapper for perceptual_spectral_density (FIX 2)."""
    return SpectralDensityMetrics.perceptual_spectral_density(
        amplitudes=amplitudes,
        frequencies=frequencies,
    )



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

def _critical_band_masking(
    masker_freq: float,
    masker_level_db: float,
    probe_freq: float,
    probe_level_db: float
) -> float:
    """
    Calculate masking threshold using Parncutt (1989) model.
    
    Critical-band analysis - masking model.
    
    Args:
        masker_freq: Frequency of masking tone (Hz)
        masker_level_db: Level of masking tone (dB)
        probe_freq: Frequency of probe tone (Hz)
        probe_level_db: Level of probe tone (dB)
        
    Returns:
        Masking threshold in dB (probe is masked if probe_level < threshold)
    """
    # Convert to Bark scale (using existing function)
    masker_bark = _hz_to_bark(np.array([masker_freq]))[0]
    probe_bark = _hz_to_bark(np.array([probe_freq]))[0]
    
    # Bark distance
    bark_distance = abs(probe_bark - masker_bark)
    
    # Parncutt (1989) masking model
    # Threshold increases with masker level and decreases with distance
    # PHASE 3: Use constants instead of magic numbers
    if bark_distance < CRITICAL_BAND_MASKING_STRONG_THRESHOLD:
        # Within same critical band: strong masking
        threshold_db = masker_level_db + MASKING_WITHIN_BAND_OFFSET_DB
    elif bark_distance < CRITICAL_BAND_MASKING_MODERATE_THRESHOLD:
        # Adjacent critical band: moderate masking
        threshold_db = masker_level_db + MASKING_ADJACENT_BAND_OFFSET_DB + MASKING_ADJACENT_BAND_SLOPE_DB * bark_distance
    elif bark_distance < CRITICAL_BAND_MASKING_WEAK_THRESHOLD:
        # Nearby critical bands: weak masking
        threshold_db = masker_level_db + MASKING_NEARBY_BAND_OFFSET_DB + MASKING_NEARBY_BAND_SLOPE_DB * bark_distance
    else:
        # Far critical bands: minimal masking
        threshold_db = masker_level_db + MASKING_FAR_BAND_OFFSET_DB + MASKING_FAR_BAND_SLOPE_DB * bark_distance
    
    # Ensure threshold is reasonable (not below absolute threshold)
    threshold_db = max(threshold_db, MASKING_ABSOLUTE_THRESHOLD_DB)
    
    return threshold_db


def estimate_noise_floor_by_critical_bands(
    frequencies_hz: np.ndarray,
    magnitudes_db: np.ndarray,
    noise_floor_percentile: float = 5.0,
    noise_floor_multiplier: float = 1.5,
    noise_floor_margin_db: Optional[float] = None,
) -> np.ndarray:
    """
    Estimate noise floor per frequency band using percentile method.
    
    PHASE 5: Physical-Acoustic Model - Uses Hz-based frequency bands (not Bark)
    
    This function estimates noise floor separately for each frequency band (Hz scale),
    providing more accurate noise estimation that accounts for frequency-dependent
    characteristics of noise and signal.
    
    CHANGED: Now uses physical frequency bands (Hz) instead of perceptual critical bands (Bark)
    to maintain physical-acoustic consistency.
    
    Args:
        frequencies_hz: Array of frequencies in Hz
        magnitudes_db: Array of magnitudes in dB
        noise_floor_percentile: Percentile to use for noise floor estimation (default 5.0)
        noise_floor_multiplier: Linear-domain factor applied to the noise floor.
            FIX 3 — interpreted as a *linear amplitude factor* and converted to
            an additive dB margin via ``20 * log10(multiplier)`` so the threshold
            actually moves up, not down. (Multiplying a negative dB value by 1.5
            yields a *lower*, more permissive threshold, which was the bug.)
        noise_floor_margin_db: Optional explicit additive margin in dB. When
            provided it takes precedence over ``noise_floor_multiplier``.

    Returns:
        Array of noise floor thresholds in dB (one per frequency)
        
    References:
        - Moore, B. C. J. (2012). An Introduction to the Psychology of Hearing (6th ed.)
        - Zwicker, E., & Fastl, H. (1999). Psychoacoustics: Facts and Models (2nd ed.)
    """
    if frequencies_hz.size == 0 or magnitudes_db.size == 0:
        return np.array([])
    
    # PHYSICAL MODEL: Use frequency bands in Hz (not Bark)
    # Define frequency bands based on physical frequency ranges
    # These bands cover the audible spectrum (20-20000 Hz) with logarithmic spacing
    frequency_bands_hz = [
        (20.0, 200.0),      # Sub-bass / Bass
        (200.0, 1000.0),    # Low-mid
        (1000.0, 5000.0),   # Mid-high
        (5000.0, 20000.0)   # High
    ]
    
    # Allocate frequencies to bands
    band_indices = np.zeros(len(frequencies_hz), dtype=int)
    for i, (f_low, f_high) in enumerate(frequency_bands_hz):
        mask = (frequencies_hz >= f_low) & (frequencies_hz < f_high)
        band_indices[mask] = i
    
    # Handle frequencies above highest band
    band_indices[frequencies_hz >= frequency_bands_hz[-1][1]] = len(frequency_bands_hz) - 1
    
    # Estimate noise floor per frequency band
    num_bands = len(frequency_bands_hz)
    noise_floors_per_band = np.full(num_bands, MASKING_ABSOLUTE_THRESHOLD_DB)
    
    # FIX 3 — additive dB margin instead of dB * multiplier.
    # If `noise_floor_margin_db` is given, use it directly. Otherwise convert
    # `noise_floor_multiplier` (interpreted as a linear amplitude factor) to dB.
    if noise_floor_margin_db is not None and np.isfinite(noise_floor_margin_db):
        margin_db = float(noise_floor_margin_db)
    else:
        try:
            mult = float(noise_floor_multiplier)
        except (TypeError, ValueError):
            mult = 1.5
        if not np.isfinite(mult) or mult <= 0.0:
            mult = 1.5
        margin_db = 20.0 * float(np.log10(mult))

    for band_idx in range(num_bands):
        band_mask = (band_indices == band_idx)
        if np.sum(band_mask) > 0:
            band_magnitudes = magnitudes_db[band_mask]
            band_noise_floor_db = float(np.percentile(band_magnitudes, noise_floor_percentile))
            noise_floors_per_band[band_idx] = max(
                band_noise_floor_db + margin_db,
                MASKING_ABSOLUTE_THRESHOLD_DB,
            )
    
    # Map noise floor back to each frequency based on its frequency band
    # SMOOTHING: Add smooth transitions at band boundaries to avoid discontinuities
    # This prevents systematic peaks at boundaries (200 Hz, 1000 Hz, 5000 Hz)
    noise_floors = noise_floors_per_band[band_indices].astype(float)
    
    # Add smooth transitions at boundaries (±20% of band width, minimum 50 Hz)
    # This prevents abrupt changes that cause systematic peaks
    # Increased from 10% to 20% to better handle notes near boundaries (e.g., A#3 at 233 Hz near 200 Hz boundary)
    transition_width_factor = 0.2  # 20% of band width for transition
    min_transition_width = 50.0  # Minimum 50 Hz transition to ensure smoothness
    
    for i in range(len(frequency_bands_hz) - 1):
        boundary = frequency_bands_hz[i][1]  # Upper boundary of band i
        band_width = boundary - frequency_bands_hz[i][0]
        transition_width = max(band_width * transition_width_factor, min_transition_width)
        
        # Find frequencies near boundary
        near_boundary_mask = (frequencies_hz >= boundary - transition_width) & (frequencies_hz <= boundary + transition_width)
        
        if np.sum(near_boundary_mask) > 0:
            # Interpolate between adjacent band noise floors
            noise_floor_low = noise_floors_per_band[i]
            noise_floor_high = noise_floors_per_band[i + 1]
            
            # Linear interpolation based on distance from boundary
            distances = frequencies_hz[near_boundary_mask] - boundary
            normalized_distances = distances / transition_width  # -1 to +1
            weights = 0.5 * (1.0 - normalized_distances)  # 1.0 at boundary-transition_width, 0.0 at boundary+transition_width
            
            # Smooth interpolation
            noise_floors[near_boundary_mask] = (
                weights * noise_floor_low + (1.0 - weights) * noise_floor_high
            )
    
    return noise_floors


def apply_spectral_masking_filter(
    frequencies_hz: np.ndarray,
    magnitudes_db: np.ndarray,
    amplitudes: np.ndarray,
    mask_components: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply spectral masking filter to remove masked components.

    Optional advanced psychoacoustic module: not part of the physical effective-density /
    fatness pipeline (masking estimates perceptual audibility, not spectral component richness).

    PHASE 4: Enhanced Spectral Masking - Parncutt (1989) Model
    
    This function applies psychoacoustic masking to identify and optionally remove
    components that are masked by stronger components in nearby critical bands.
    
    Args:
        frequencies_hz: Array of frequencies in Hz
        magnitudes_db: Array of magnitudes in dB
        amplitudes: Array of amplitudes (linear)
        mask_components: If True, remove masked components; if False, only mark them
        
    Returns:
        Tuple of (filtered_frequencies, filtered_magnitudes_db, filtered_amplitudes, mask)
        where mask indicates which components are audible (not masked)
        
    References:
        - Parncutt, R. (1989). Harmony: A Psychoacoustical Approach.
    """
    if frequencies_hz.size == 0:
        return np.array([]), np.array([]), np.array([]), np.array([], dtype=bool)
    
    n_components = len(frequencies_hz)
    is_audible = np.ones(n_components, dtype=bool)
    
    # Sort by magnitude (descending) to process strongest components first
    sort_indices = np.argsort(magnitudes_db)[::-1]
    
    # For each component, check if it's masked by stronger components
    for i, idx in enumerate(sort_indices):
        probe_freq = frequencies_hz[idx]
        probe_level_db = magnitudes_db[idx]
        
        # Check masking by all stronger components (already processed)
        is_masked = False
        for j in range(i):  # Only check components processed before (stronger)
            masker_idx = sort_indices[j]
            if not is_audible[masker_idx]:  # Skip if masker was already masked
                continue
                
            masker_freq = frequencies_hz[masker_idx]
            masker_level_db = magnitudes_db[masker_idx]
            
            # Calculate masking threshold
            masking_threshold = _critical_band_masking(
                masker_freq, masker_level_db,
                probe_freq, probe_level_db
            )
            
            # If probe level is below masking threshold, it's masked
            if probe_level_db < masking_threshold:
                is_masked = True
                break
        
        # Mark as masked if below threshold
        if is_masked:
            is_audible[idx] = False
    
    # Apply filter if requested
    if mask_components:
        filtered_freqs = frequencies_hz[is_audible]
        filtered_mags_db = magnitudes_db[is_audible]
        filtered_amps = amplitudes[is_audible]
    else:
        filtered_freqs = frequencies_hz
        filtered_mags_db = magnitudes_db
        filtered_amps = amplitudes
    
    return filtered_freqs, filtered_mags_db, filtered_amps, is_audible


def validate_spectral_density_metric(
    calculated_value: float,
    frequencies_hz: np.ndarray,
    amplitudes: np.ndarray,
    expected_range: Optional[Tuple[float, float]] = None,
    reference_value: Optional[float] = None,
    tolerance: float = 0.2
) -> Dict[str, Union[bool, float, str]]:
    """
    Validate Spectral Density Metric against ground truth or expected values.
    
    PHASE 4: Ground Truth Validation
    
    This function validates the calculated Spectral Density Metric against:
    1. Expected range (if provided)
    2. Reference value (if provided, e.g., from known test signals)
    3. Physical constraints (energy conservation, positive values)
    
    Args:
        calculated_value: Calculated Spectral Density Metric value
        frequencies_hz: Array of frequencies used in calculation
        amplitudes: Array of amplitudes used in calculation
        expected_range: Optional tuple (min, max) for expected range
        reference_value: Optional reference value for comparison
        tolerance: Tolerance for comparison with reference (default 20%)
        
    Returns:
        Dictionary with validation results:
        - 'is_valid': bool - Overall validation status
        - 'errors': list - List of error messages
        - 'warnings': list - List of warning messages
        - 'comparison_with_reference': dict - Comparison with reference if provided
        - 'physical_checks': dict - Physical constraint checks
        
    References:
        - Parseval's theorem for energy conservation
        - Psychoacoustic limits for spectral density
    """
    errors = []
    warnings = []
    physical_checks = {}
    comparison = {}
    
    # 1. Physical constraint checks
    # Value should be positive (sum of powers)
    if calculated_value < 0:
        errors.append(f"Spectral Density Metric is negative: {calculated_value}")
        physical_checks['positive'] = False
    else:
        physical_checks['positive'] = True
    
    # Value should be finite
    if not np.isfinite(calculated_value):
        errors.append(f"Spectral Density Metric is not finite: {calculated_value}")
        physical_checks['finite'] = False
    else:
        physical_checks['finite'] = True
    
    # Energy conservation check: metric should be related to total spectral energy
    if frequencies_hz.size > 0 and amplitudes.size > 0:
        total_energy = np.sum(amplitudes ** 2)
        if total_energy > 0:
            # Spectral Density Metric should be proportional to total energy
            # (exact relationship depends on weight function, but should be correlated)
            energy_ratio = calculated_value / total_energy
            if energy_ratio > 10.0:  # Unusually high ratio
                warnings.append(
                    f"Spectral Density Metric / Total Energy ratio is high: {energy_ratio:.2f}"
                )
            physical_checks['energy_ratio'] = energy_ratio
        else:
            if calculated_value > 0:
                errors.append(
                    "Spectral Density Metric > 0 but total energy = 0 (inconsistent)"
                )
    
    # 2. Expected range check
    if expected_range is not None:
        min_val, max_val = expected_range
        if calculated_value < min_val or calculated_value > max_val:
            errors.append(
                f"Spectral Density Metric ({calculated_value:.2f}) outside expected range "
                f"[{min_val:.2f}, {max_val:.2f}]"
            )
            physical_checks['in_range'] = False
        else:
            physical_checks['in_range'] = True
    
    # 3. Reference value comparison
    if reference_value is not None:
        if reference_value > 0:
            relative_error = abs(calculated_value - reference_value) / reference_value
            comparison['relative_error'] = relative_error
            comparison['absolute_error'] = abs(calculated_value - reference_value)
            
            if relative_error > tolerance:
                errors.append(
                    f"Spectral Density Metric ({calculated_value:.2f}) differs from reference "
                    f"({reference_value:.2f}) by {relative_error*100:.1f}% (tolerance: {tolerance*100:.1f}%)"
                )
                comparison['within_tolerance'] = False
            else:
                comparison['within_tolerance'] = True
        else:
            warnings.append("Reference value is zero or negative, skipping comparison")
    
    # Overall validation status
    is_valid = len(errors) == 0
    
    return {
        'is_valid': is_valid,
        'errors': errors,
        'warnings': warnings,
        'comparison_with_reference': comparison,
        'physical_checks': physical_checks,
        'calculated_value': calculated_value
    }


def calculate_perceptual_spectral_density(
    harmonic_amplitudes: np.ndarray,
    harmonic_frequencies: np.ndarray,
    fundamental_freq: float,
    threshold_db: float = -60.0,
    frequency_limit: float = 20000.0
) -> float:
    """
    Calcula a densidade espectral perceptual baseada em princípios psicoacústicos.
    
    PHASE 2: Enhanced with 24 Critical Bands and Masking Model
    
    Esta métrica considera:
    1. Número de harmônicos audíveis presentes vs. possíveis
    2. Distribuição de energia ao longo do espectro
    3. Ponderação perceptual usando 24 bandas críticas (Moore, 2012)
    4. Mascaramento espectral (Parncutt, 1989)
    
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
    
    # 3. PHASE 2: Use 24 Critical Bands (Moore, 2012) instead of 3-band approximation
    # PHASE 3: Use constant instead of magic number
    # Critical band boundaries in Bark: 0, 1, 2, ..., 23 (24 bands)
    critical_bands = np.arange(0, NUM_CRITICAL_BANDS)  # NUM_CRITICAL_BANDS = 24
    
    # Convert frequencies to Bark (using existing function)
    harmonic_bark = _hz_to_bark(harmonic_frequencies)
    
    # PHASE 3: Use constant instead of magic number
    # Allocate harmonics to critical bands
    band_energies = np.zeros(NUM_CRITICAL_BANDS)
    band_counts = np.zeros(NUM_CRITICAL_BANDS)
    
    for i, (freq, amp_db, bark) in enumerate(zip(harmonic_frequencies, harmonic_db, harmonic_bark)):
        if amp_db > threshold_db:
            # Find which critical band this harmonic belongs to
            band_idx = int(np.clip(np.floor(bark), 0, NUM_CRITICAL_BANDS - 1))
            
            # Convert dB to linear amplitude for energy calculation
            amp_linear = 10 ** (amp_db / 20)
            band_energies[band_idx] += amp_linear ** 2  # Power
            band_counts[band_idx] += 1
    
    # 4. PHASE 2: Apply masking model (Parncutt, 1989)
    # PHASE 3: Use constant instead of magic number
    # Calculate effective audibility considering masking
    effective_band_energies = np.zeros(NUM_CRITICAL_BANDS)
    
    for band_idx in range(NUM_CRITICAL_BANDS):
        if band_energies[band_idx] > 0:
            # Convert back to dB for masking calculation
            band_level_db = 10 * np.log10(band_energies[band_idx] + 1e-12)
            
            # Center frequency of this critical band (in Hz)
            # Approximate: Bark to Hz conversion (inverse)
            band_center_bark = band_idx + 0.5
            # Use approximate inverse Bark-to-Hz conversion
            # For low Bark: f ≈ (bark/13) * 1000
            # For high Bark: use iterative approximation or lookup table
            if band_center_bark < 2:
                band_center_hz = (band_center_bark / 13.0) * 1000.0
            elif band_center_bark < 10:
                # Mid-range: approximate linear relationship
                band_center_hz = 100.0 * (band_center_bark - 1.0) + 200.0
            else:
                # High range: approximate exponential
                band_center_hz = 1000.0 * np.exp((band_center_bark - 10.0) / 3.0)
            
            # PHASE 3: Use constant instead of magic number
            # Check if this band is masked by other bands
            is_masked = False
            for other_band_idx in range(NUM_CRITICAL_BANDS):
                if other_band_idx != band_idx and band_energies[other_band_idx] > 0:
                    other_level_db = 10 * np.log10(band_energies[other_band_idx] + 1e-12)
                    other_center_bark = other_band_idx + 0.5
                    if other_center_bark < 2:
                        other_center_hz = (other_center_bark / 13.0) * 1000.0
                    elif other_center_bark < 10:
                        other_center_hz = 100.0 * (other_center_bark - 1.0) + 200.0
                    else:
                        other_center_hz = 1000.0 * np.exp((other_center_bark - 10.0) / 3.0)
                    
                    # Calculate masking threshold
                    masking_threshold = _critical_band_masking(
                        other_center_hz, other_level_db,
                        band_center_hz, band_level_db
                    )
                    
                    # If probe level is below masking threshold, it's masked
                    if band_level_db < masking_threshold:
                        is_masked = True
                        break
            
            # Only count unmasked energy
            if not is_masked:
                effective_band_energies[band_idx] = band_energies[band_idx]
    
    # 5. Calculate metrics using critical bands
    # PHASE 3: Use constant instead of magic number
    # a) Occupancy: number of active critical bands
    active_bands = np.sum(effective_band_energies > 0)
    occupancy_density = active_bands / float(NUM_CRITICAL_BANDS)  # Normalized by NUM_CRITICAL_BANDS
    
    # b) Energy distribution: how evenly energy is distributed across bands
    total_energy = np.sum(effective_band_energies)
    if total_energy > 0:
        band_energy_fractions = effective_band_energies / total_energy
        # PHASE 3: Use constant instead of magic number
        # Entropy of distribution (higher = more uniform)
        entropy = -np.sum(band_energy_fractions[band_energy_fractions > 0] * 
                         np.log2(band_energy_fractions[band_energy_fractions > 0]))
        max_entropy = np.log2(float(NUM_CRITICAL_BANDS))
        uniformity = entropy / max_entropy if max_entropy > 0 else 0.0
    else:
        uniformity = 0.0
    
    # c) Completeness: check harmonic series gaps (enhanced extractor).
    completeness = _calculate_harmonic_completeness_phase2(
        harmonic_frequencies, harmonic_db, fundamental_freq, frequency_limit, threshold_db
    )
    
    # 6. PHASE 3: Use constants instead of magic numbers
    # Combine metrics with weights based on psychoacoustic research
    final_density = (
        PERCEPTUAL_DENSITY_OCCUPANCY_WEIGHT * occupancy_density +      # Number of active critical bands
        PERCEPTUAL_DENSITY_UNIFORMITY_WEIGHT * uniformity +              # Energy distribution uniformity
        PERCEPTUAL_DENSITY_COMPLETENESS_WEIGHT * completeness              # Harmonic series completeness
    )
    
    # 7. Apply logarithmic correction (Weber-Fechner law)
    perceptual_density = 1.0 - np.exp(-PERCEPTUAL_DENSITY_LOG_SCALE_FACTOR * final_density)
    
    return np.clip(perceptual_density, 0.0, 1.0)


def _calculate_harmonic_completeness_phase2(
    harmonic_frequencies: np.ndarray,
    harmonic_db: np.ndarray,
    fundamental_freq: float,
    frequency_limit: float,
    threshold_db: float
) -> float:
    """
    Calculate harmonic series completeness with frequency-dependent gap penalty.
    
    Enhanced harmonic-series analysis.
    
    Lower harmonics (more audible) have higher penalty for gaps.
    
    Args:
        harmonic_frequencies: Frequencies of detected harmonics
        harmonic_db: Amplitudes in dB
        fundamental_freq: Fundamental frequency
        frequency_limit: Maximum frequency (Nyquist)
        threshold_db: Detection threshold
        
    Returns:
        Completeness score (0-1)
    """
    if fundamental_freq <= 0:
        return 0.0
    
    # PHASE 3: Use constant instead of magic number
    # Calculate maximum possible harmonics up to Nyquist
    max_harmonic_number = int(frequency_limit / fundamental_freq)
    max_harmonic_number = min(max_harmonic_number, HARMONIC_MAX_CHECK)  # Cap at HARMONIC_MAX_CHECK for practicality
    
    if max_harmonic_number < 1:
        return 0.0
    
    # Expected harmonic frequencies
    expected_freqs = np.arange(1, max_harmonic_number + 1) * fundamental_freq
    
    # Tolerance for harmonic detection (adaptive: tighter for lower harmonics)
    base_tolerance = 0.1  # 10% base tolerance
    
    # Check each expected harmonic
    gap_penalty = 0.0
    total_weight = 0.0
    
    # PHASE 3: Use constants instead of magic numbers
    for n, expected_freq in enumerate(expected_freqs, start=1):
        # Frequency-dependent tolerance: tighter for lower harmonics
        tolerance = HARMONIC_TOLERANCE_BASE * (1.0 + HARMONIC_TOLERANCE_ADAPTIVE_FACTOR * n)
        
        # Find closest detected harmonic
        if len(harmonic_frequencies) > 0:
            freq_diffs = np.abs(harmonic_frequencies - expected_freq)
            closest_idx = np.argmin(freq_diffs)
            closest_freq = harmonic_frequencies[closest_idx]
            closest_db = harmonic_db[closest_idx]
            
            # Check if harmonic is present and above threshold
            freq_error = abs(closest_freq - expected_freq) / expected_freq
            
            if freq_error <= tolerance and closest_db > threshold_db:
                # PHASE 3: Use constant instead of magic number
                # Harmonic present: no penalty
                weight = HARMONIC_COMPLETENESS_WEIGHT_BASE / n  # Lower harmonics weighted more
                total_weight += weight
            else:
                # Harmonic missing: apply penalty
                # Penalty increases for lower harmonics (more audible)
                weight = 1.0 / n
                gap_penalty += weight
                total_weight += weight
        else:
            # PHASE 3: Use constant instead of magic number
            # No harmonics detected: full penalty
            weight = HARMONIC_COMPLETENESS_WEIGHT_BASE / n
            gap_penalty += weight
            total_weight += weight
    
    # Completeness = 1 - (gap_penalty / total_weight)
    if total_weight > 0:
        completeness = 1.0 - (gap_penalty / total_weight)
    else:
        completeness = 0.0
    
    return np.clip(completeness, 0.0, 1.0)

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
        # Registered for UI / validation; ``apply_density_metric`` short-circuits to
        # ``_apply_discrete_spectral_metrics`` (no element-wise path).
        'd3': WeightFunction.logarithmic,
        'd10': WeightFunction.logarithmic,
        'd17': WeightFunction.linear,
        'd24': WeightFunction.logarithmic,
    }

    key = (name or '').strip().lower()
    # Legacy alias removed from UI: identical to ``linear`` (element-wise identity, then Σ).
    if key == "sum":
        key = "linear"
    # Removed metrics D2/D8 from UI; old presets still resolve.
    if key == "d2":
        key = "linear"
    if key == "d8":
        key = "d17"
    if key not in weight_functions:
        raise ValueError(f"Função de ponderação '{name}' não encontrada.")
    return weight_functions[key]


# --- Rolloff-compensated harmonic density (relative descriptor; not SPL / physical power) ---
DEFAULT_HARMONIC_ROLLOFF_ALPHA: float = 1.5
DEFAULT_ROLLOFF_COMPENSATED_DENSITY_WEIGHT_FUNCTION: str = "logarithmic"

# --- Canonical harmonic "fatness" (version-5 core, adapted to v6 pipeline) ---
# Same numerical path as ``apply_density_metric`` with default flags used from ProcAudio / SuperAudioAnalyzer.
CANONICAL_DENSITY_FORMULA_VERSION: str = "v5_apply_density_metric_adapted_v6_1"
CANONICAL_DENSITY_SOURCE_FORMULA: str = (
    "canonical_density_v5_adapted = apply_density_metric("
    "linear_harmonic_amplitudes, weight_function, normalize=False, remove_noise=False, "
    "frequencies=Hz_per_row, fundamental_freq=f0, account_for_spectral_rolloff=True, "
    "prevent_domination=True): max-normalize amplitudes, divide each by expected n^(-alpha) "
    "rolloff with n=max(f/f0,1), alpha=1.5, then sum weight_function(partial_values). "
    "Unbounded positive index (not a probability); cross-note comparison uses raw values; "
    "[0,1] scaling only via density_normalized_global at compile time."
)


def compute_harmonic_effective_power_density(
    harmonic_df: Optional[pd.DataFrame] = None,
    *,
    amplitudes: Optional[np.ndarray] = None,
    frequencies_hz: Optional[np.ndarray] = None,
    harmonic_orders: Optional[np.ndarray] = None,
    fundamental_freq_hz: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Harmonic effective power density (additive descriptor; not masking / loudness / roughness).

    For valid harmonic partial rows with linear amplitudes A_i > 0:
        P_i = A_i^2
        p_i = P_i / max(P)
        harmonic_effective_power_density = sum_i p_i

    Optional diagnostic:
        harmonic_effective_power_density_normalized_by_harmonic_count = value / N
    """
    out: Dict[str, Any] = {
        "harmonic_effective_power_density": float("nan"),
        "harmonic_effective_power_density_component_count": 0,
        "harmonic_effective_power_density_status": "skipped_uninitialized",
        "harmonic_effective_power_density_max_amplitude": float("nan"),
        "harmonic_effective_power_density_total_power": float("nan"),
        "harmonic_effective_power_density_normalized_by_harmonic_count": float("nan"),
        "harmonic_effective_power_density_normalized_by_expected_slots": float("nan"),
    }

    def _finish(status: str) -> Dict[str, Any]:
        out["harmonic_effective_power_density_status"] = status
        return out

    amp_priority = ("Amplitude", "Amplitude_linear", "Linear Amplitude", "magnitude_linear")

    def _resolve_amp_series(df: pd.DataFrame) -> Optional[pd.Series]:
        for col in amp_priority:
            if col in df.columns:
                return pd.to_numeric(df[col], errors="coerce")
        return None

    a_valid: Optional[np.ndarray] = None

    if harmonic_df is not None and isinstance(harmonic_df, pd.DataFrame) and not harmonic_df.empty:
        df = harmonic_df.copy()
        amp_s = _resolve_amp_series(df)
        if amp_s is None:
            return _finish("skipped_no_valid_amplitude_column")

        mask = np.ones(len(df), dtype=bool)
        a_raw = amp_s.to_numpy(dtype=float, copy=False)
        mask &= np.isfinite(a_raw) & (a_raw > 0.0)

        if "Harmonic Number" in df.columns:
            ho = pd.to_numeric(df["Harmonic Number"], errors="coerce").to_numpy(dtype=float, copy=False)
            n_int = np.round(ho).astype(int)
            mask &= np.isfinite(ho) & (n_int >= 1)
        elif "Frequency (Hz)" in df.columns and fundamental_freq_hz is not None:
            fhz = pd.to_numeric(df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float, copy=False)
            f0 = float(fundamental_freq_hz)
            if np.isfinite(f0) and f0 > 0.0:
                n_est = np.round(fhz / f0).astype(int)
                mask &= np.isfinite(fhz) & (fhz > 0.0) & (n_est >= 1)
            else:
                fhz = pd.to_numeric(df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float, copy=False)
                mask &= np.isfinite(fhz) & (fhz > 0.0)
        elif "Frequency (Hz)" in df.columns:
            fhz = pd.to_numeric(df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float, copy=False)
            mask &= np.isfinite(fhz) & (fhz > 0.0)

        a_valid = a_raw[mask]
        if a_valid.size == 0:
            return _finish("skipped_no_valid_harmonic_rows")
    else:
        if amplitudes is None:
            return _finish("skipped_no_valid_harmonic_rows")
        a = np.asarray(amplitudes, dtype=float).reshape(-1)
        if a.size == 0:
            return _finish("skipped_no_valid_harmonic_rows")
        mask = np.isfinite(a) & (a > 0.0)
        if harmonic_orders is not None:
            ho = np.asarray(harmonic_orders, dtype=float).reshape(-1)
            if ho.size != a.size:
                return _finish("skipped_no_valid_harmonic_rows")
            n_int = np.round(ho).astype(int)
            mask &= np.isfinite(ho) & (n_int >= 1)
        elif frequencies_hz is not None and fundamental_freq_hz is not None:
            f = np.asarray(frequencies_hz, dtype=float).reshape(-1)
            if f.size != a.size:
                return _finish("skipped_no_valid_harmonic_rows")
            f0 = float(fundamental_freq_hz)
            if not (np.isfinite(f0) and f0 > 0.0):
                mask &= np.isfinite(f) & (f > 0.0)
            else:
                n_est = np.round(f / f0).astype(int)
                mask &= np.isfinite(f) & (f > 0.0) & (n_est >= 1)
        a_valid = a[mask]
        if a_valid.size == 0:
            return _finish("skipped_no_valid_harmonic_rows")

    pwr = np.square(a_valid.astype(float, copy=False))
    max_p = float(np.max(pwr))
    if not np.isfinite(max_p) or max_p <= 0.0:
        return _finish("skipped_zero_or_negative_max_amplitude")

    p_norm = pwr / max_p
    dens = float(np.sum(p_norm))
    if not np.isfinite(dens):
        return _finish("skipped_nonfinite_result")

    n_comp = int(a_valid.size)
    out["harmonic_effective_power_density"] = dens
    out["harmonic_effective_power_density_component_count"] = n_comp
    out["harmonic_effective_power_density_max_amplitude"] = float(np.max(a_valid))
    out["harmonic_effective_power_density_total_power"] = float(np.sum(pwr))
    out["harmonic_effective_power_density_normalized_by_harmonic_count"] = float(dens / n_comp) if n_comp > 0 else float("nan")
    try:
        expected_slots = int(
            compute_expected_harmonic_slot_count(
                float(fundamental_freq_hz) if fundamental_freq_hz is not None else float("nan"),
                float(np.nanmax(np.asarray(frequencies_hz, dtype=float)))
                if frequencies_hz is not None and np.asarray(frequencies_hz).size
                else float("nan"),
            )
        )
    except Exception:
        expected_slots = 0
    if expected_slots > 0:
        out["harmonic_effective_power_density_normalized_by_expected_slots"] = float(
            dens / expected_slots
        )
    out["harmonic_effective_power_density_status"] = "computed"
    return out


def compute_expected_harmonic_slot_count(
    f0_hz: float,
    max_frequency_hz: float,
) -> int:
    """Return how many integer harmonic slots can exist up to ``max_frequency_hz``."""
    try:
        f0 = float(f0_hz)
        fmax = float(max_frequency_hz)
    except (TypeError, ValueError):
        return 0
    if not np.isfinite(f0) or f0 <= 0.0 or not np.isfinite(fmax) or fmax <= 0.0:
        return 0
    return int(max(0, np.floor(fmax / f0)))


def compute_harmonic_occupancy_ratio(
    harmonic_df: Optional[pd.DataFrame],
    *,
    f0_hz: float,
    max_frequency_hz: float,
) -> Dict[str, Any]:
    """Compute harmonic occupancy as detected-valid slots / expected slots."""
    expected = int(compute_expected_harmonic_slot_count(f0_hz, max_frequency_hz))
    out: Dict[str, Any] = {
        "harmonic_occupancy_ratio": float("nan"),
        "expected_harmonic_slot_count": int(expected),
        "detected_harmonic_slot_count": 0,
        "harmonic_occupancy_status": "invalid_f0_or_ceiling",
    }
    if expected <= 0 or harmonic_df is None or harmonic_df.empty:
        if expected > 0:
            out["harmonic_occupancy_ratio"] = 0.0
            out["harmonic_occupancy_status"] = "no_harmonic_rows"
        return out

    df = harmonic_df.copy()
    if "Frequency (Hz)" not in df.columns:
        return out
    freq = pd.to_numeric(df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float, copy=False)
    mask = np.isfinite(freq) & (freq > 0.0) & (freq <= float(max_frequency_hz))

    if "include_for_density" in df.columns:
        inc = df["include_for_density"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
        mask &= inc.to_numpy(dtype=bool, copy=False)
    if "local_peak_valid" in df.columns:
        lp = df["local_peak_valid"].astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
        mask &= lp.to_numpy(dtype=bool, copy=False)
    if "SNR_dB" in df.columns and "SNR Threshold (dB)" in df.columns:
        snr = pd.to_numeric(df["SNR_dB"], errors="coerce").to_numpy(dtype=float, copy=False)
        thr = pd.to_numeric(df["SNR Threshold (dB)"], errors="coerce").to_numpy(dtype=float, copy=False)
        mask &= np.isfinite(snr) & np.isfinite(thr) & (snr >= thr)

    n_est = np.round(freq / float(f0_hz)).astype(int)
    n_est = n_est[mask]
    n_est = n_est[(n_est >= 1) & (n_est <= expected)]
    detected = int(np.unique(n_est).size) if n_est.size else 0
    out["detected_harmonic_slot_count"] = detected
    out["harmonic_occupancy_ratio"] = float(min(1.0, detected / expected)) if expected > 0 else float("nan")
    out["harmonic_occupancy_status"] = "computed"
    return out


def compute_residual_log_frequency_occupancy(
    residual_df: Optional[pd.DataFrame],
    *,
    min_frequency_hz: float = 20.0,
    max_frequency_hz: Optional[float] = None,
    bins_per_octave: int = 24,
) -> Dict[str, Any]:
    """Compute log-frequency occupancy of residual rows outside harmonic windows."""
    out: Dict[str, Any] = {
        "residual_log_frequency_occupancy": float("nan"),
        "residual_log_frequency_bin_count": 0,
        "residual_log_frequency_bin_total": 0,
        "residual_log_frequency_occupancy_status": "no_data",
    }
    if residual_df is None or residual_df.empty or "Frequency (Hz)" not in residual_df.columns:
        return out
    f = pd.to_numeric(residual_df["Frequency (Hz)"], errors="coerce").to_numpy(dtype=float, copy=False)
    fmin = float(min_frequency_hz) if np.isfinite(min_frequency_hz) and min_frequency_hz > 0 else 20.0
    finite_f = f[np.isfinite(f) & (f > 0.0)]
    if finite_f.size == 0:
        return out
    fmax = (
        float(max_frequency_hz)
        if max_frequency_hz is not None and np.isfinite(max_frequency_hz) and max_frequency_hz > fmin
        else float(np.max(finite_f))
    )
    if not np.isfinite(fmax) or fmax <= fmin:
        return out
    valid = finite_f[(finite_f >= fmin) & (finite_f <= fmax)]
    if valid.size == 0:
        out["residual_log_frequency_occupancy"] = 0.0
        out["residual_log_frequency_occupancy_status"] = "computed"
        return out
    bpo = max(1, int(bins_per_octave))
    total_bins = max(1, int(np.ceil(np.log2(fmax / fmin) * bpo)))
    log_pos = np.log2(valid / fmin) * bpo
    idx = np.clip(np.floor(log_pos).astype(int), 0, total_bins - 1)
    occupied = int(np.unique(idx).size)
    out["residual_log_frequency_bin_count"] = occupied
    out["residual_log_frequency_bin_total"] = total_bins
    out["residual_log_frequency_occupancy"] = float(occupied / total_bins)
    out["residual_log_frequency_occupancy_status"] = "computed"
    return out


def compute_harmonic_effective_power_mass(
    harmonic_df: Optional[pd.DataFrame] = None,
    amplitude_col: str = "Amplitude",
) -> Dict[str, Any]:
    """
    Compute absolute harmonic power mass from harmonic partial linear amplitudes.

    Sum of squared amplitudes preserves energetic scale across notes/instruments (not max-normalized).

    Parameters
    ----------
    harmonic_df:
        DataFrame containing harmonic partial rows (read-only; not mutated).
    amplitude_col:
        Column containing linear amplitudes.

    Returns
    -------
    dict with:
        harmonic_effective_power_mass
        harmonic_effective_power_mean
        harmonic_effective_power_rms
        harmonic_effective_power_component_count
        harmonic_effective_power_mass_status
    """
    if harmonic_df is None or len(harmonic_df) == 0:
        return {
            "harmonic_effective_power_mass": float("nan"),
            "harmonic_effective_power_mean": float("nan"),
            "harmonic_effective_power_rms": float("nan"),
            "harmonic_effective_power_component_count": 0,
            "harmonic_effective_power_mass_status": "skipped_empty_harmonic_df",
        }

    if amplitude_col not in harmonic_df.columns:
        return {
            "harmonic_effective_power_mass": float("nan"),
            "harmonic_effective_power_mean": float("nan"),
            "harmonic_effective_power_rms": float("nan"),
            "harmonic_effective_power_component_count": 0,
            "harmonic_effective_power_mass_status": f"skipped_missing_{amplitude_col}",
        }

    amplitudes = pd.to_numeric(harmonic_df[amplitude_col], errors="coerce").to_numpy(dtype=float, copy=False)
    amplitudes = amplitudes[np.isfinite(amplitudes)]
    amplitudes = amplitudes[amplitudes > 0]

    if amplitudes.size == 0:
        return {
            "harmonic_effective_power_mass": float("nan"),
            "harmonic_effective_power_mean": float("nan"),
            "harmonic_effective_power_rms": float("nan"),
            "harmonic_effective_power_component_count": 0,
            "harmonic_effective_power_mass_status": "skipped_no_positive_finite_amplitudes",
        }

    power = np.square(amplitudes.astype(float, copy=False))

    return {
        "harmonic_effective_power_mass": float(np.sum(power)),
        "harmonic_effective_power_mean": float(np.mean(power)),
        "harmonic_effective_power_rms": float(np.sqrt(np.mean(power))),
        "harmonic_effective_power_component_count": int(power.size),
        "harmonic_effective_power_mass_status": "computed",
    }


def compute_rolloff_compensated_harmonic_density(
    amplitudes: np.ndarray,
    frequencies_hz: np.ndarray,
    fundamental_freq_hz: float,
    *,
    harmonic_orders: Optional[np.ndarray] = None,
    alpha: float = DEFAULT_HARMONIC_ROLLOFF_ALPHA,
    weight_function: str = DEFAULT_ROLLOFF_COMPENSATED_DENSITY_WEIGHT_FUNCTION,
    epsilon: float = 1e-12,
) -> Dict[str, Any]:
    """
    Rolloff-compensated relative harmonic richness (legacy-inspired, explicitly named).

    For each detected harmonic-row partial i (linear amplitude A_i, harmonic order n_i >= 1):
        A_norm_i = A_i / max(A)
        expected_rolloff_i = n_i ** (-alpha)
        compensated_i = A_norm_i / expected_rolloff_i
        weighted_i = w(compensated_i)   with w from ``get_weight_function``
        metric = sum_i weighted_i

    This is not an absolute physical density, not SPL, and not a substitute for bin-based
    energy fractions. Integer orders default to round(f_i / f0) when ``harmonic_orders``
    is not supplied; invalid rows are dropped (not coerced).

    Returns keys:
        rolloff_compensated_harmonic_density, rolloff_compensated_harmonic_density_alpha,
        rolloff_compensated_harmonic_density_component_count, rolloff_compensated_harmonic_density_status,
        and when status == ``computed``: optional ``rolloff_harmonic_partial_count``,
        ``rolloff_density_metric_per_harmonic``, ``rolloff_density_metric_normalized`` (density divided
        by linear amplitude of the partial whose order is 1, if present and positive). The ratio
        ``rolloff_density_metric_normalized`` is **not** confined to [0, 1] and must **not** be used as
        a normalized density metric; use ``density_normalized_global`` from ``compile_metrics`` instead.
    """
    out: Dict[str, Any] = {
        "rolloff_compensated_harmonic_density": float("nan"),
        "rolloff_compensated_harmonic_density_alpha": float(alpha),
        "rolloff_compensated_harmonic_density_component_count": 0,
        "rolloff_compensated_harmonic_density_status": "skipped_uninitialized",
    }

    def _finish(status: str, reason: Optional[str] = None) -> Dict[str, Any]:
        out["rolloff_compensated_harmonic_density_status"] = status if not reason else f"{status}:{reason}"
        return out

    if not (isinstance(alpha, (int, float)) and np.isfinite(float(alpha)) and float(alpha) > 0):
        return _finish("skipped_invalid_alpha")

    f0 = float(fundamental_freq_hz)
    if not (np.isfinite(f0) and f0 > 0):
        return _finish("skipped_invalid_fundamental_frequency")

    a = np.asarray(amplitudes, dtype=float).reshape(-1)
    f = np.asarray(frequencies_hz, dtype=float).reshape(-1)
    if a.size == 0 or f.size == 0 or a.size != f.size:
        return _finish("skipped_no_harmonic_components")

    if harmonic_orders is not None:
        ho = np.asarray(harmonic_orders, dtype=float).reshape(-1)
        if ho.size != a.size:
            return _finish("skipped_harmonic_orders_length_mismatch")
        n_orders = np.round(ho).astype(int)
    else:
        n_orders = np.round(f / f0).astype(int)

    mask = (
        np.isfinite(a)
        & np.isfinite(f)
        & (a >= 0.0)
        & (f > 0.0)
        & (n_orders >= 1)
    )
    a = a[mask]
    f = f[mask]
    n_orders = n_orders[mask]
    if a.size == 0:
        return _finish("skipped_no_valid_partial_after_filters")

    a_max = float(np.max(a))
    if not np.isfinite(a_max) or a_max <= 0.0:
        return _finish("skipped_all_amplitudes_nonpositive")

    a_norm = a / a_max
    expected = np.power(np.maximum(n_orders.astype(float), 1.0), -float(alpha))
    compensated = a_norm / (expected + epsilon)

    try:
        wfn = get_weight_function(weight_function)
        weighted = np.asarray(wfn(compensated), dtype=float)
    except Exception as exc:
        return _finish("skipped_weight_function_error", str(exc))

    if not np.all(np.isfinite(weighted)):
        return _finish("skipped_nonfinite_weighted_values")

    density = float(np.sum(weighted))
    count = int(a.size)
    out["rolloff_compensated_harmonic_density"] = density
    out["rolloff_compensated_harmonic_density_component_count"] = count
    out["rolloff_compensated_harmonic_density_status"] = "computed"

    out["rolloff_harmonic_partial_count"] = count
    out["rolloff_density_metric_per_harmonic"] = float(density / count) if count > 0 else float("nan")
    ord1 = np.where(n_orders == 1)[0]
    if ord1.size > 0:
        a1 = float(a[ord1[0]])
        if np.isfinite(a1) and a1 > epsilon:
            out["rolloff_density_metric_normalized"] = float(density / a1)
        else:
            out["rolloff_density_metric_normalized"] = float("nan")
    else:
        out["rolloff_density_metric_normalized"] = float("nan")

    return out


DISCRETE_SPECTRAL_METRIC_KEYS = frozenset({"d3", "d10", "d17", "d24"})


def _spectral_neff_from_filtered_linear_amplitudes(v: np.ndarray) -> float:
    """
    N_eff = 1 / Σ p_i² with p_i = A_i² / Σ A_j² on nonnegative linear amplitudes.

    ``v`` must already be finite, nonnegative, and 1-D (same filtered vector as discrete metrics).
    """
    v = np.asarray(v, dtype=float).reshape(-1)
    if v.size == 0:
        return 0.0
    pwr = np.square(v.astype(float, copy=False))
    s = float(np.sum(pwr))
    if s <= 1e-30:
        return 0.0
    p = pwr / s
    den = float(np.sum(np.square(p)))
    if den <= 1e-30:
        return 0.0
    return float(1.0 / den)


def _apply_discrete_spectral_metrics(
    weight_key: str,
    values: Union[np.ndarray, List[float]],
    frequencies: Optional[Union[np.ndarray, List[float]]] = None,
    *,
    d24_amplitude_max_override: Optional[float] = None,
) -> float:
    """
    Atomic spectral summaries on linear amplitudes A_i (no rolloff / max-normalization).

    D3:  Σ log(1 + A_i)  (log natural via ``np.log1p``)
    D10: (Σ log(1 + A_i)) · (N_eff / N)  com N = número de parciais; N_eff = 1/Σ p_i², p_i = A_i²/Σ A_j²
    D17: log(1 + Σ A_i²) · log(1 + N_eff)  (log natural via ``np.log1p``)
    D24: como D3 só em parciais com A_i ≥ 0.01·max(A) e f ≤ 12000 Hz (se ``frequencies`` alinhado);
         sem frequências aplica-se só o limiar de 1 % do máximo.
    """
    key = (weight_key or "").strip().lower()
    v = np.asarray(values, dtype=float).reshape(-1)
    if v.size == 0:
        return 0.0

    f: Optional[np.ndarray] = None
    if frequencies is not None:
        f = np.asarray(frequencies, dtype=float).reshape(-1)
        if f.size != v.size:
            f = None

    m = np.isfinite(v) & (v >= 0.0)
    if f is not None:
        m &= np.isfinite(f)
    v = v[m]
    if f is not None:
        f = f[m]
    if v.size == 0:
        return 0.0

    if key == "d3":
        return float(np.sum(np.log1p(v)))

    if key == "d10":
        n_eff = float(_spectral_neff_from_filtered_linear_amplitudes(v))
        n = float(v.size)
        if n <= 0.0:
            return 0.0
        s_log = float(np.sum(np.log1p(v)))
        return float(s_log * (n_eff / n))

    if key == "d17":
        n_eff = float(_spectral_neff_from_filtered_linear_amplitudes(v))
        e_sum = float(np.sum(np.square(v)))
        return float(np.log1p(e_sum) * np.log1p(n_eff))

    if key == "d24":
        if f is not None:
            mb = f <= 12000.0
        else:
            mb = np.ones(v.shape[0], dtype=bool)
        if d24_amplitude_max_override is not None and np.isfinite(float(d24_amplitude_max_override)):
            a_max = float(d24_amplitude_max_override)
        else:
            a_max = float(np.max(v))
        if a_max <= 0.0:
            return 0.0
        mb &= v >= (0.01 * a_max)
        v24 = v[mb]
        if v24.size == 0:
            return 0.0
        return float(np.sum(np.log1p(v24)))

    return 0.0


def band_partial_metric_sum(
    amplitudes: Union[np.ndarray, List[float]],
    weight_key: str,
    *,
    frequencies_hz: Optional[Union[np.ndarray, List[float]]] = None,
    d24_global_amplitude_max: Optional[float] = None,
) -> float:
    """
    Aggregate one component list (harmonic, inharmonic, or sub-bass) using the same
    ``weight_key`` semantics as the main analyser (``get_weight_function`` or discrete D3/D10/D17/D24).

    For ``d24``, ``d24_global_amplitude_max`` (global max amplitude across H+I+S) reproduces the
    1 %-of-max gate used on the full vector while summing per band.
    """
    key = (weight_key or "linear").strip().lower()
    if key == "d2":
        key = "linear"
    elif key == "d8":
        key = "d17"
    v = np.asarray(amplitudes, dtype=float).reshape(-1)
    v = v[np.isfinite(v) & (v >= 0.0)]
    if v.size == 0:
        return 0.0
    f = None
    if frequencies_hz is not None:
        f = np.asarray(frequencies_hz, dtype=float).reshape(-1)
        if f.size != v.size:
            f = None
    if key in DISCRETE_SPECTRAL_METRIC_KEYS:
        if key == "d24":
            return float(
                _apply_discrete_spectral_metrics(
                    "d24",
                    v,
                    f,
                    d24_amplitude_max_override=d24_global_amplitude_max,
                )
            )
        return float(_apply_discrete_spectral_metrics(key, v, None))
    fn = get_weight_function(key)
    return float(np.sum(fn(v)))


def partial_metric_sums_h_i_s_total(
    harmonic_amplitudes: Union[np.ndarray, List[float]],
    inharmonic_amplitudes: Union[np.ndarray, List[float]],
    subbass_amplitudes: Union[np.ndarray, List[float]],
    weight_key: str,
    *,
    harmonic_frequencies_hz: Optional[Union[np.ndarray, List[float]]] = None,
    inharmonic_frequencies_hz: Optional[Union[np.ndarray, List[float]]] = None,
    subbass_frequencies_hz: Optional[Union[np.ndarray, List[float]]] = None,
) -> Tuple[float, float, float, float]:
    """
    Single pipeline for per-note ``Metrics`` / compiled ``Density_Metrics`` partial sums (H, I, S, Total).

    Each band is aggregated only through ``band_partial_metric_sum`` with the same ``weight_key``.

    **Continuous** weights (``linear``, ``sqrt``, ``log``, …): each band vector is first reduced to a
    **single nonnegative scalar** (sum of the supplied nonnegative entries). For ``Metrics`` export,
    ``proc_audio`` passes one ΣA² total per band and **``weight_key="linear"``** so H/I/S/Total remain plain
    energy sums (``log``/``sqrt`` on those scalars would not be interpretable as shares of total energy).

    **Discrete** ``d3``/``d10``/``d17``/``d24``: native definitions on the **per-row** linear-amplitude
    vectors supplied for each band (unchanged).

    **Total:** for ``d10`` and ``d17``, ``Total`` is the metric on the concatenated H+I+S vector
    (one global ``band_partial_metric_sum(..., wf)``). Otherwise ``Total`` is ``H + I + S`` (additive across
    the three disjoint lists). For ``d24``, the 1 %-of-max gate uses the maximum amplitude across H+I+S
    (``d24_global_amplitude_max``), matching ``band_partial_metric_sum`` contract.
    """
    wf = (weight_key or "linear").strip().lower()
    if wf == "d2":
        wf = "linear"
    elif wf == "d8":
        wf = "d17"

    def _vec(x: Union[np.ndarray, List[float]]) -> np.ndarray:
        return np.asarray(x, dtype=float).reshape(-1)

    ah_raw = _vec(harmonic_amplitudes)
    ai_raw = _vec(inharmonic_amplitudes)
    asb_raw = _vec(subbass_amplitudes)
    parts_raw = [v for v in (ah_raw, ai_raw, asb_raw) if v.size > 0]
    all_a_raw = np.concatenate(parts_raw) if parts_raw else np.zeros(0, dtype=float)
    gmax = float(np.nanmax(all_a_raw)) if all_a_raw.size and np.isfinite(all_a_raw).any() else None

    def _freq_opt(
        freqs: Optional[Union[np.ndarray, List[float]]], n: int
    ) -> Optional[np.ndarray]:
        if freqs is None or n <= 0:
            return None
        f = np.asarray(freqs, dtype=float).reshape(-1)
        if f.size != n:
            return None
        return f

    if wf in DISCRETE_SPECTRAL_METRIC_KEYS:
        ah, ai, asb = ah_raw, ai_raw, asb_raw
        fh = _freq_opt(harmonic_frequencies_hz, ah.size)
        fi = _freq_opt(inharmonic_frequencies_hz, ai.size)
        fsb = _freq_opt(subbass_frequencies_hz, asb.size)
    else:

        def _band_linear_total(a: np.ndarray) -> np.ndarray:
            if a.size == 0:
                return np.array([0.0], dtype=float)
            s = float(np.sum(a[np.isfinite(a) & (a >= 0.0)]))
            return np.array([s], dtype=float)

        ah = _band_linear_total(ah_raw)
        ai = _band_linear_total(ai_raw)
        asb = _band_linear_total(asb_raw)
        fh = fi = fsb = None

    def _one_band(amps: np.ndarray, freqs: Optional[np.ndarray]) -> float:
        return float(
            band_partial_metric_sum(
                amps,
                wf,
                frequencies_hz=freqs,
                d24_global_amplitude_max=gmax,
            )
        )

    h_sum = _one_band(ah, fh)
    i_sum = _one_band(ai, fi)
    s_sum = _one_band(asb, fsb)

    if wf in ("d10", "d17"):
        chunks_a: List[np.ndarray] = []
        chunks_f: List[np.ndarray] = []
        for a_part, f_part in ((ah, fh), (ai, fi), (asb, fsb)):
            if a_part.size == 0:
                continue
            chunks_a.append(a_part)
            if f_part is not None and f_part.size == a_part.size:
                chunks_f.append(np.asarray(f_part, dtype=float).reshape(-1))
            else:
                chunks_f.append(np.full(a_part.shape[0], np.nan, dtype=float))
        if chunks_a:
            af = np.concatenate(chunks_a)
            ff = np.concatenate(chunks_f)
            t_sum = float(band_partial_metric_sum(af, wf, frequencies_hz=ff))
        else:
            t_sum = 0.0
    else:
        t_sum = float(h_sum + i_sum + s_sum)

    return h_sum, i_sum, s_sum, t_sum


def compute_discrete_spectral_metrics_bundle(
    amplitudes: Union[np.ndarray, List[float]],
    frequencies_hz: Optional[Union[np.ndarray, List[float]]] = None,
) -> Dict[str, float]:
    """
    Métricas espectrais discretas D3/D10/D17/D24 sobre o mesmo vector de amplitudes lineares dos parciais
    harmónicos (definições fixas; ver ``_apply_discrete_spectral_metrics``).

    Chaves devolvidas: ``discrete_metric_d3`` … ``discrete_metric_d24`` (para export / GUI / compilação).
    """
    keys = (
        "discrete_metric_d3",
        "discrete_metric_d10",
        "discrete_metric_d17",
        "discrete_metric_d24",
    )
    nan_bundle = {k: float("nan") for k in keys}
    a = np.asarray(amplitudes, dtype=float).reshape(-1)
    if a.size == 0:
        return nan_bundle
    return {
        "discrete_metric_d3": float(_apply_discrete_spectral_metrics("d3", a, None)),
        "discrete_metric_d10": float(_apply_discrete_spectral_metrics("d10", a, None)),
        "discrete_metric_d17": float(_apply_discrete_spectral_metrics("d17", a, None)),
        "discrete_metric_d24": float(_apply_discrete_spectral_metrics("d24", a, frequencies_hz)),
    }


def apply_density_metric(values, weight_function='linear',
                        normalize=False, remove_noise=False,
                        frequencies=None, fundamental_freq=None,
                        account_for_spectral_rolloff=True,
                        prevent_domination=True):
    """
    Applies a weighting function to a set of values and aggregates them.
    
    FATNESS METRIC: Measures spectral "fatness" - more harmonics with considerable energy = more density.
    Prevents single strong partials from dominating the metric.

    Discrete spectral metrics (bypass rolloff, domination, and ``remove_noise``):
    ``d3``, ``d10``, ``d17``, ``d24`` — see ``_apply_discrete_spectral_metrics``.
    
    Args:
        values: The input numpy array of amplitudes (harmonic partials).
        weight_function: The name of the weighting function to apply ('linear', 'log', 'sqrt', etc.).
        normalize: If True, normalizes the result by the number of values.
        remove_noise: If True, filters out low-level noise before calculation.
        frequencies: Optional array of frequencies corresponding to values (Hz).
        fundamental_freq: Optional fundamental frequency (Hz) for normalization.
        account_for_spectral_rolloff: If True, applies frequency-dependent normalization
            to account for natural acoustic decay (produces smooth descending curve).
        prevent_domination: If True, normalizes amplitudes to prevent single strong partials
            from dominating the metric. This ensures "more harmonics = more density" logic.
    """
    wf_key = (weight_function or "linear").strip().lower()
    if wf_key == "d2":
        wf_key = "linear"
    elif wf_key == "d8":
        wf_key = "d17"
    if wf_key in DISCRETE_SPECTRAL_METRIC_KEYS:
        return float(_apply_discrete_spectral_metrics(wf_key, values, frequencies))

    if remove_noise:
        # This could be configurable
        noise_threshold = 1e-6  # or passed as a parameter
        values = values[values > np.max(values) * noise_threshold]

    if values.size == 0:
        return 0.0
    
    values = np.asarray(values, dtype=float)
    
    # Handle NaN and Inf values (filter invalid values)
    # Mathematical verification: NaN + any = NaN, Inf + any = Inf
    # Solution: Filter out NaN/Inf before processing
    valid_mask = np.isfinite(values)
    if not np.all(valid_mask):
        # Filter out invalid values
        values = values[valid_mask]
        if values.size == 0:
            return 0.0
    
    # Handle negative values (amplitudes should be non-negative)
    # Mathematical verification: |amplitude| = sqrt(real² + imag²) ≥ 0
    # Solution: Take absolute value to ensure non-negative
    if np.any(values < 0):
        values = np.abs(values)
    
    # FATNESS FIX: Normalize to prevent single strong partials from dominating
    # This ensures that "more harmonics with considerable energy = more density"
    # Without this, a single very strong partial can dominate the sum
    original_max = None
    if prevent_domination and len(values) > 1:
        # Normalize by maximum amplitude to prevent domination
        # Each partial is relative to the strongest (0-1 range)
        # This ensures all harmonics contribute proportionally
        original_max = np.max(values)
        if original_max > 1e-10:
            # Normalize by max: each partial relative to strongest
            # Note with 1 strong partial (1.0): normalized=[1.0] → contributes 1.0
            # Note with 10 moderate partials (0.3 each): normalized=[0.3, 0.3, ...] → contributes 3.0
            # Result: More harmonics = more density ✅
            values = values / original_max

    # ACOUSTIC FIX: Apply frequency-dependent normalization for smooth descending curve
    if account_for_spectral_rolloff and frequencies is not None and fundamental_freq is not None:
        frequencies = np.asarray(frequencies, dtype=float)
        
        # Ensure fundamental is valid
        if fundamental_freq > 0 and len(frequencies) > 0:
            # Normalize frequencies by fundamental (harmonic number)
            harmonic_numbers = frequencies / fundamental_freq
            
            # Expected energy decay: 1/n^alpha where n is harmonic number
            # Typical spectral rolloff: alpha ≈ 1.0-1.5 (energy decays as 1/n to 1/n^1.5)
            # INCREASED alpha to 1.5 for stronger normalization (better compensation)
            alpha = 1.5  # Increased from 1.2 for stronger compensation
            expected_energy = np.power(np.maximum(harmonic_numbers, 1.0), -alpha)
            
            # Normalize by expected energy: actual / expected
            # This compensates for natural decay, producing consistent density values
            # Low frequencies (n≈1): expected_energy≈1.0, no change
            # High frequencies (n≈10): expected_energy≈0.032, values boosted by ~31x
            # Result: Smooth descending curve instead of irregular pattern
            normalized_values = values / (expected_energy + 1e-10)  # Add small epsilon to avoid division by zero
            
            # Use normalized values for calculation
            values = normalized_values
    
    # Apply weight function (log, sqrt, linear, etc.)
    weight_func = get_weight_function(weight_function)
    weighted = weight_func(values)

    # Sum all weighted harmonics
    # More harmonics = more density (each contributes to the sum)
    # With normalization, no single partial dominates
    
    result = np.sum(weighted)
    
    # If we normalized to prevent domination, the sum already reflects proportional contributions
    # More harmonics = more terms in the sum = higher density
    # The normalization ensures no single partial dominates, so all harmonics contribute fairly
    
    if normalize and len(values) > 0:
        return result / len(values)
    else:
        return result


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
    
    wf_key = (weight_function or "linear").strip().lower()
    if wf_key == "d24" and "Frequency (Hz)" in df.columns:
        sub = pd.DataFrame(
            {
                "a": pd.to_numeric(df[amplitude_column], errors="coerce"),
                "f": pd.to_numeric(df["Frequency (Hz)"], errors="coerce"),
            }
        ).dropna()
        if sub.empty:
            return 0.0
        return apply_density_metric(
            sub["a"].to_numpy(dtype=float),
            weight_function,
            frequencies=sub["f"].to_numpy(dtype=float),
        )

    amplitude_values = df[amplitude_column].values
    return apply_density_metric(amplitude_values, weight_function)


# ======================================================================
# Effective partial density (participation ratio on powers; v7-inspired, v6-primary)
# ======================================================================
#
# Audio is already RMS-normalised upstream (proc_audio._normalize_level) before the STFT.
# Density metrics here therefore describe spectral *shape* after level normalisation, not
# concert-hall loudness.
# ======================================================================


def effective_partial_density_from_powers(
    powers: np.ndarray,
    *,
    eps: float = 1e-30,
) -> float:
    """
    Inverse participation ratio (effective number of equal-power components).

        D_eff = (sum_i P_i)^2 / sum_i(P_i^2)

    with P_i **linear power** (non-negative). Scale-invariant: scaling all P_i by a
    positive constant leaves D_eff unchanged. One dominant component → D_eff ≈ 1;
    n equal components → n; many tiny residuals pooled with one large peak do not
    inflate D_eff like a raw FFT-bin count would.

    Returns 0.0 when there is no strictly positive finite power.
    """
    p = np.asarray(powers, dtype=float).ravel()
    p = p[np.isfinite(p) & (p > float(eps))]
    if p.size == 0:
        return 0.0
    s = float(np.sum(p))
    ss = float(np.sum(p * p))
    if s <= float(eps) or ss <= float(eps):
        return 0.0
    d = (s * s) / ss
    return float(d) if np.isfinite(d) else 0.0


def _inverse_herfindahl_effective_components(powers: np.ndarray, eps: float = 1e-30) -> float:
    """Alias for ``effective_partial_density_from_powers`` (Herfindahl / participation ratio)."""
    return effective_partial_density_from_powers(powers, eps=eps)


# AUDIT FIX (acoustic-physics correction, Clarinete_mf finding #1) — the
# sub-bass band aggregator must reject sub-audible content. Bins below
# the lower bound below carry DC offset, room rumble, HVAC, structural
# vibration, breath noise and FFT-leakage from the DC bin — none of
# which is musical sub-bass energy. The default 30 Hz floor sits a
# comfortable margin below the lowest musical fundamental in standard
# orchestral practice (piano A0 = 27.5 Hz) and well below the lowest
# clarinet fundamental (Bb clarinet sounding pitch D3 = 146.83 Hz), so
# it never excludes legitimate harmonic content.
SUBBASS_AGGREGATE_LOWER_HZ: float = 30.0


# AUDIT FIX (acoustic-physics correction, Clarinete_mf finding #2) — the
# legacy 12 Hz harmonic-protection band is narrower than the main-lobe
# of every realistic FFT window (Blackman-Harris main lobe ~9.4 bins;
# Hann ~4 bins; Hamming ~4 bins). When a strong fundamental sits inside
# or near the sub-bass band, the window leakage shoulder (typically
# 2–5 FFT bins from the peak) escapes the protection band and inflates
# ``subbass_energy_sum``. The corrected tolerance scales with FFT
# resolution: ``max(12 Hz, k * FFT bin width)`` with k = 4 by default
# (≈ blackman-harris main lobe -60 dB extent in bins).
SUBBASS_PROTECTION_BIN_MULTIPLIER: float = 4.0
SUBBASS_PROTECTION_MIN_HZ: float = 12.0


def compute_subbass_protection_tolerance_hz(
    sample_rate_hz: float,
    n_fft: int,
    *,
    bin_multiplier: float = SUBBASS_PROTECTION_BIN_MULTIPLIER,
    minimum_hz: float = SUBBASS_PROTECTION_MIN_HZ,
) -> float:
    """Return the window-aware harmonic-protection tolerance for the
    sub-bass aggregator.

    The Blackman-Harris main lobe occupies ~9.4 FFT bins between -3 dB
    points; its -60 dB extent is in the 4-bin range. We use that
    -60 dB extent as the protection half-width, with a 12 Hz floor for
    very narrow analyses.

    Returns the legacy 12 Hz floor when inputs are unusable.
    """
    try:
        sr = float(sample_rate_hz)
        nfft = int(n_fft)
    except (TypeError, ValueError):
        return float(minimum_hz)
    if not np.isfinite(sr) or sr <= 0.0 or nfft <= 0:
        return float(minimum_hz)
    bin_hz = sr / float(nfft)
    if not np.isfinite(bin_hz) or bin_hz <= 0.0:
        return float(minimum_hz)
    return max(float(minimum_hz), float(bin_multiplier) * bin_hz)


def aggregate_low_frequency_residual_peak_power(
    complete_list_df: Optional[pd.DataFrame],
    harmonic_list_df: Optional[pd.DataFrame],
    *,
    subbass_hz: float = 200.0,
    subbass_lower_hz: float = SUBBASS_AGGREGATE_LOWER_HZ,
    freq_match_tol_hz: float = 12.0,
    low_band_mode: Literal["local_maxima", "sum_all_bins"] = "local_maxima",
) -> float:
    """
    Aggregate **power** (amplitude^2) for **fixed-band low-frequency residual** peaks in
    ``(subbass_lower_hz, subbass_hz]``, excluding bins matched to harmonic templates within
    ``freq_match_tol_hz``.

    This is **not** the adaptive subfundamental guard (see ``low_frequency_policy``). It does
    **not** prove physical sub-bass energy, acoustic noise, or independent partials — only a
    bounded-band residual peak-power statistic for energy bookkeeping / diagnostics.
    """
    if complete_list_df is None or complete_list_df.empty:
        return 0.0
    if "Frequency (Hz)" not in complete_list_df.columns:
        return 0.0
    f_all = pd.to_numeric(complete_list_df["Frequency (Hz)"], errors="coerce").to_numpy(float)
    harm_freqs = np.asarray([], dtype=float)
    if harmonic_list_df is not None and not harmonic_list_df.empty and "Frequency (Hz)" in harmonic_list_df.columns:
        harm_freqs = pd.to_numeric(harmonic_list_df["Frequency (Hz)"], errors="coerce").dropna().to_numpy(float)

    if "Amplitude" in complete_list_df.columns:
        amp = pd.to_numeric(complete_list_df["Amplitude"], errors="coerce").to_numpy(float)
    elif "Magnitude (dB)" in complete_list_df.columns:
        amp = np.power(10.0, pd.to_numeric(complete_list_df["Magnitude (dB)"], errors="coerce").to_numpy(float) / 20.0)
    else:
        return 0.0

    n = int(min(len(f_all), len(amp)))
    if n <= 0:
        return 0.0
    f_all = f_all[:n]
    amp = amp[:n]

    # Defensive sanitisation of the lower-bound parameter.
    try:
        _lo = float(subbass_lower_hz)
    except (TypeError, ValueError):
        _lo = SUBBASS_AGGREGATE_LOWER_HZ
    if not np.isfinite(_lo) or _lo < 0.0:
        _lo = SUBBASS_AGGREGATE_LOWER_HZ

    def _harmonic_mask(fi: float) -> bool:
        if harm_freqs.size <= 0:
            return False
        return float(np.min(np.abs(harm_freqs - fi))) <= float(freq_match_tol_hz)

    def _in_sub_band(i: int) -> bool:
        fi = float(f_all[i])
        if not np.isfinite(fi) or fi <= _lo or fi > float(subbass_hz):
            return False
        if _harmonic_mask(fi):
            return False
        ai = float(amp[i])
        return np.isfinite(ai) and ai > 0.0

    mode = str(low_band_mode or "local_maxima").strip().lower()
    if mode not in ("local_maxima", "sum_all_bins"):
        mode = "local_maxima"

    tot = 0.0
    if mode == "sum_all_bins":
        for i in range(n):
            if not _in_sub_band(i):
                continue
            ai = float(amp[i])
            tot += ai * ai
        return float(tot)

    def _is_strict_local_max(i: int) -> bool:
        ai = float(amp[i])
        if not np.isfinite(ai) or ai <= 0.0:
            return False
        if i == 0:
            if n < 2:
                return False
            ar = float(amp[1])
            return np.isfinite(ar) and ai > ar
        if i == n - 1:
            al = float(amp[n - 2])
            return np.isfinite(al) and ai > al
        al, ar = float(amp[i - 1]), float(amp[i + 1])
        if not (np.isfinite(al) and np.isfinite(ar)):
            return False
        return ai > al and ai > ar

    for i in range(n):
        if not _in_sub_band(i):
            continue
        if not _is_strict_local_max(i):
            continue
        ai = float(amp[i])
        tot += ai * ai
    return float(tot)


def aggregate_subbass_noise_peak_power(*args: Any, **kwargs: Any) -> float:
    """
    Deprecated compatibility wrapper.

    Use :func:`aggregate_low_frequency_residual_peak_power` instead. This aggregate
    measures fixed-band low-frequency **residual** peak power, not proven physical
    sub-bass and not proven noise.
    """
    return aggregate_low_frequency_residual_peak_power(*args, **kwargs)


def partial_density_effective_components_bundle(
    harmonic_amplitudes: Optional[np.ndarray] = None,
    inharmonic_amplitudes: Optional[np.ndarray] = None,
    ground_noise_power: Optional[float] = None,
    inharmonic_mode: str = "aggregate",
    min_db_relative: float = -60.0,
    eps: float = 1e-30,
) -> Tuple[float, Dict[str, Any]]:
    """
    Participation-ratio effective component count from a **small** power vector.

    Harmonics: each partial above ``min_db_relative`` dB vs the peak is its own P_i;
    weaker harmonic power is merged into one bin. Inharmonics: one aggregate bin by
    default. Ground / sub-bass aggregate: one bin when ``ground_noise_power > 0``.

    See module docstring above for RMS / shape interpretation.
    """
    mode = str(inharmonic_mode or "aggregate").strip().lower()
    if mode not in ("aggregate", "significant_peaks"):
        mode = "aggregate"

    h = np.asarray(harmonic_amplitudes if harmonic_amplitudes is not None else [], dtype=float).ravel()
    ih = np.asarray(inharmonic_amplitudes if inharmonic_amplitudes is not None else [], dtype=float).ravel()
    h = np.maximum(np.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    ih = np.maximum(np.nan_to_num(ih, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

    ph = h * h
    pih_peak = ih * ih

    gpow = float(ground_noise_power) if ground_noise_power is not None else 0.0
    if not np.isfinite(gpow) or gpow < 0.0:
        gpow = 0.0

    harm_tot = float(np.sum(ph)) if ph.size else 0.0
    ih_tot = float(np.sum(pih_peak)) if pih_peak.size else 0.0

    ref = max(
        float(np.max(ph)) if ph.size else 0.0,
        float(np.max(pih_peak)) if pih_peak.size else 0.0,
        gpow,
    )
    if ref <= float(eps) or not np.isfinite(ref):
        diag: Dict[str, Any] = {
            "partial_density_harmonic_power_total": harm_tot,
            "partial_density_inharmonic_power_total": ih_tot,
            "partial_density_ground_noise_power": gpow,
            "partial_density_component_count_harmonic": 0,
            "partial_density_component_count_inharmonic_significant": 0,
            "partial_density_inharmonic_mode": mode,
            "partial_density_min_db_relative": float(min_db_relative),
        }
        return 0.0, diag

    thresh = ref * (10.0 ** (float(min_db_relative) / 10.0))

    p_list: List[float] = []

    if ph.size:
        strong = ph[ph >= thresh]
        weak_sum = float(np.sum(ph[ph < thresh]))
        for v in strong.tolist():
            if v > float(eps):
                p_list.append(float(v))
        if weak_sum > float(eps):
            p_list.append(weak_sum)

    ih_sig_count = 0
    if mode == "aggregate":
        if ih_tot > float(eps):
            p_list.append(ih_tot)
    else:
        if pih_peak.size:
            sig = pih_peak[pih_peak >= thresh]
            weak_ih = float(np.sum(pih_peak[pih_peak < thresh]))
            for v in sig.tolist():
                if v > float(eps):
                    p_list.append(float(v))
            if weak_ih > float(eps):
                p_list.append(weak_ih)
        ih_sig_count = int(np.sum(pih_peak >= thresh)) if pih_peak.size else 0

    if gpow > float(eps):
        p_list.append(gpow)

    p_arr = np.asarray(p_list, dtype=float)
    n_h_model = 0
    if ph.size:
        n_h_model = int(np.sum(ph >= thresh)) + (
            1 if float(np.sum(ph[ph < thresh])) > float(eps) else 0
        )

    d = _inverse_herfindahl_effective_components(p_arr, eps)

    diag = {
        "partial_density_harmonic_power_total": harm_tot,
        "partial_density_inharmonic_power_total": ih_tot,
        "partial_density_ground_noise_power": gpow,
        "partial_density_component_count_harmonic": int(n_h_model),
        "partial_density_component_count_inharmonic_significant": int(ih_sig_count),
        "partial_density_inharmonic_mode": mode,
        "partial_density_min_db_relative": float(min_db_relative),
    }
    return float(d), diag


def partial_density_effective_components(
    harmonic_amplitudes: Optional[np.ndarray] = None,
    inharmonic_amplitudes: Optional[np.ndarray] = None,
    ground_noise_power: Optional[float] = None,
    inharmonic_mode: str = "aggregate",
    min_db_relative: float = -60.0,
    eps: float = 1e-30,
) -> float:
    """Return only D_eff; see ``partial_density_effective_components_bundle``."""
    d, _ = partial_density_effective_components_bundle(
        harmonic_amplitudes=harmonic_amplitudes,
        inharmonic_amplitudes=inharmonic_amplitudes,
        ground_noise_power=ground_noise_power,
        inharmonic_mode=inharmonic_mode,
        min_db_relative=min_db_relative,
        eps=eps,
    )
    return float(d)


def identify_nonharmonic_residual_rows(
    harmonic_df: pd.DataFrame,
    complete_df: pd.DataFrame,
    tolerance: Union[float, int] = 0.02,
    *,
    sr: Optional[float] = None,
    n_fft: Optional[int] = None,
    bin_width_hz: Optional[float] = None,
    main_lobe_bins: Optional[float] = None,
    spectral_leakage_guard: bool = True,
) -> pd.DataFrame:
    """
    Return spectral rows **outside** the protected harmonic exclusion windows.

    This function does **not** by itself identify acoustic inharmonic partials.
    It returns **residual spectral rows** (complete-list bins/rows that are not
    within tolerance — plus optional leakage widening — of any frequency in
    ``harmonic_df``).

    A row returned here may be: spectral background, leakage outside the
    exclusion window, noise, a side-lobe shoulder, or a genuine non-harmonic
    component. Further local-peak, SNR, prominence, leakage, and (ideally)
    temporal-stability checks are required before calling any row an
    **accepted inharmonic peak** or **inharmonic partial**.

    Tolerance semantics (unchanged):
    * If ``tolerance >= 1.0``: absolute Hz band per harmonic.
    * If ``tolerance < 1.0``: relative proportion (e.g. 0.02 → ±2%).

    When ``spectral_leakage_guard`` is true and STFT geometry is available,
    exclusion half-width per harmonic is ``max(tolerance, main-lobe half-width)``.

    Args:
        harmonic_df: Harmonic reference rows (``Frequency (Hz)``).
        complete_df: Full spectrum table to mask against those windows.
        tolerance: Match / exclusion width (relative or absolute Hz).
        sr, n_fft, bin_width_hz, main_lobe_bins: optional STFT geometry for leakage guard.
        spectral_leakage_guard: widen windows when true.

    Returns:
        Subset of ``complete_df`` rows outside all harmonic windows.

    Raises:
        ValueError: If ``Frequency (Hz)`` is missing from either frame.
    """
    # Validação de entrada
    if harmonic_df is None or harmonic_df.empty or complete_df is None or complete_df.empty:
        logger.warning("DataFrame harmônico ou completo vazio em identify_nonharmonic_residual_rows")
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
        logger.error(f"Error extracting frequencies from DataFrames: {e}")
        raise

    leak_hw = 0.0
    if spectral_leakage_guard:
        try:
            from spectral_leakage_guards import leakage_halfwidth_hz as _leakage_hw

            leak_hw = float(
                _leakage_hw(
                    sr=sr,
                    n_fft=n_fft,
                    bin_width_hz=bin_width_hz,
                    main_lobe_bins=main_lobe_bins,
                )
            )
        except Exception as _e_leak:
            logger.debug("Spectral leakage guard disabled (import/geometry): %s", _e_leak)
            leak_hw = 0.0

    # Máscara booleana: começa por assumir que TODOS são inarmônicos
    inharmonic_mask = np.ones_like(all_freqs, dtype=bool)

    # Itera sobre cada harmônico e "desmarca" quem cair dentro da tolerância (+ guard STFT)
    for f0 in harm_freqs:
        if tolerance < 1.0:  # limiar relativo
            # PHASE 3: Use constant instead of magic number
            thr_match = np.maximum(float(f0) * float(tolerance), EPSILON_FREQUENCY)  # piso 1 mHz
        else:  # limiar absoluto
            thr_match = float(tolerance)
        thr = float(max(thr_match, leak_hw)) if spectral_leakage_guard and leak_hw > 0.0 else float(thr_match)
        inharmonic_mask &= np.abs(all_freqs.astype(float) - float(f0)) > thr

    # Aplicar a máscara e retornar apenas as linhas residuais fora das janelas harmónicas
    return complete_df.loc[inharmonic_mask].reset_index(drop=True)


def identify_inharmonic_partials(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Deprecated compatibility wrapper.

    Use :func:`identify_nonharmonic_residual_rows` instead. The returned rows are
    **residual spectral rows outside harmonic windows**, not confirmed
    inharmonic partials.
    """
    return identify_nonharmonic_residual_rows(*args, **kwargs)


def calculate_combined_density_metric(
    harmonic_density: float,
    inharmonic_density: float,
    alpha: float = 0.8,
    beta: float = 0.2,
    preserve_dynamic_range: bool = True
) -> float:
    """
    Combines harmonic and inharmonic densities with dynamic range preservation.
    
    CRITICAL FIX: This function now preserves absolute magnitude differences between
    different dynamic levels (e.g., 'pp' vs 'ff'). The logarithmic combination
    ensures that 'fortissimo' produces substantially higher values than 'pianissimo'.
    
    Args:
        harmonic_density: Harmonic density value (amplitude/energy-weighted, NOT count-based)
        inharmonic_density: Inharmonic density value (amplitude/energy-weighted, NOT count-based)
        alpha: Weight for harmonic component
        beta: Weight for inharmonic component
        preserve_dynamic_range: If True, uses logarithmic combination to preserve dynamic range
        
    Returns:
        Combined density metric (preserves dynamic range, no per-sample normalization)
    """
    # Normalize weights to sum to 1
    total_weight = alpha + beta
    if total_weight > 0 and not np.isclose(total_weight, 1.0):
        alpha = alpha / total_weight
        beta = beta / total_weight

    # CRITICAL FIX: Use logarithmic combination to preserve dynamic range
    # This ensures 'ff' (high amplitudes) produces substantially higher values than 'pp'
    if preserve_dynamic_range:
        try:
            # Apply log1p to handle wide dynamic range gracefully
            # log1p(x) = log(1 + x) ≈ x for small x, preserves relative differences
            harm_log = np.log1p(max(0.0, harmonic_density))
            inharm_log = np.log1p(max(0.0, inharmonic_density))
            
            # Weighted combination in log space (preserves relative magnitudes)
            combined_log = alpha * harm_log + beta * inharm_log
            
            # Convert back to linear scale: expm1(x) = exp(x) - 1
            # This maintains the absolute differences between different dynamic levels
            combined = float(np.expm1(combined_log))
            
            return combined
        except (ValueError, TypeError, FloatingPointError, ZeroDivisionError) as exc:
            logger.warning("Falling back to linear combination after log-combination failure: %s", exc)
    
    # Linear combination (fallback or if preserve_dynamic_range=False)
    # Still preserves differences, but may compress very high values
    combined = alpha * harmonic_density + beta * inharmonic_density
    
    # Note: NO per-sample normalization here - that would erase dynamic differences
    # Normalization should only be applied at the dataset level for comparison purposes
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
    import matplotlib.pyplot as plt
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
            logger.error(f"Error saving comparison plot: {e}")
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
    import matplotlib.pyplot as plt
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
        
        # Save or display
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Harmonic spectrum plot saved to {output_path}")
            plt.close()
        else:
            plt.show()
            plt.close()
            
    except Exception as e:
        logger.error(f"Error plotting harmonic spectrum: {e}")
        plt.close()
        raise

# Helper functions (imported here if not already in scope)
def _hz_to_bark(f):
    return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

def spectral_density(
    freqs_hz, amps, f0_hz=None,
    # proximidade
    proximity_axis="hz",      # CHANGED: "hz" (physical) instead of "bark" (perceptual)
    sigma=500.0,             # CHANGED: sigma in Hz (was 0.5 Bark ≈ 500 Hz at mid-frequencies)
    hz_window=4000.0,        # CHANGED: window in Hz (was 8.0 Bark ≈ 4000 Hz)
    max_peaks_per_band=4,    # FIX 8 — soft cap (kept for back-compat). See `hard_cap_peaks_per_band`.
    hard_cap_peaks_per_band: bool = False,  # FIX 8 — when True, components beyond max are *dropped* (not just down-weighted to 0.1).
    # pesos R/P
    weight_r=0.45, weight_p=0.55,
    # termo de "peso" (baixo-freq)
    lambda_low=0.35,         # mistura com W
    low_hz_cut=1000.0,       # CHANGED: cutoff in Hz (was 8 Bark ≈ 1000 Hz)
    q=1.0, gamma=2.0,        # FIX 8 — default is now `2` (power), not `1` (amplitude).
    weight_function: str = "linear",
):
    """
    FIX 8 — default ``gamma`` is now ``2.0``. With linear amplitudes this
    produces ``p_i = A_i**2``, i.e. power weighting, which matches the
    docstring intent and Parseval-style energy reasoning.

    A new keyword ``hard_cap_peaks_per_band`` toggles a *real* per-band cap:
    when ``True``, components beyond ``max_peaks_per_band`` are removed from
    ``p`` (mass and frequency vector are both filtered). When ``False`` (the
    historical behaviour) excess components are only multiplied by
    ``weights * 0.1`` and renormalised — useful for smoother metrics but a
    misleading name for a "cap". Callers who want the old behaviour need no
    change; callers who relied on the parameter as a real cap should pass
    ``hard_cap_peaks_per_band=True``.
    """
    freqs_hz = np.asarray(freqs_hz, float)
    amps = np.asarray(amps, float)
    mask = (freqs_hz > 0) & (amps > 0) & np.isfinite(freqs_hz) & np.isfinite(amps)
    freqs_hz, amps = freqs_hz[mask], amps[mask]
    
    if freqs_hz.size == 0:
        return dict(R_norm=0.0, P_norm=0.0, W_low=0.0, D_agn=0.0, D_peso=0.0, D_harm=None)

    # pesos normalizados (potência^gamma) — default gamma=2 → P_i = A_i**2.
    p = (amps**gamma)
    p_sum = p.sum()
    if p_sum > 0:
        p = p / p_sum
    else:
        return dict(R_norm=0.0, P_norm=0.0, W_low=0.0, D_agn=0.0, D_peso=0.0, D_harm=None)

    # --- PHYSICAL MODEL: Use Hz directly (no Bark conversion) ---
    # Window relative to f0 in Hz
    if (f0_hz is not None) and np.isfinite(f0_hz) and f0_hz > 0:
        # Window in Hz: [f0, f0 + hz_window]
        win = (freqs_hz >= f0_hz) & (freqs_hz <= f0_hz + float(hz_window))
        if win.any():
            p, freqs_hz = p[win], freqs_hz[win]
            # Re-normalize p after windowing
            p = p / p.sum()

    # Cap by frequency band (PHYSICAL: Hz-based bands, not Bark)
    if max_peaks_per_band and max_peaks_per_band > 0:
        # PHYSICAL MODEL: Use frequency bands in Hz (not Bark)
        # Define frequency bands based on physical frequency ranges
        frequency_bands_hz = [
            (20.0, 200.0),      # Sub-bass / Bass
            (200.0, 1000.0),    # Low-mid
            (1000.0, 5000.0),   # Mid-high
            (5000.0, 20000.0)   # High
        ]
        
        # Allocate frequencies to bands
        bands_discrete = np.zeros(len(freqs_hz), dtype=int)
        for i, (f_low, f_high) in enumerate(frequency_bands_hz):
            mask = (freqs_hz >= f_low) & (freqs_hz < f_high)
            bands_discrete[mask] = i
        
        # Handle frequencies above highest band
        bands_discrete[freqs_hz >= frequency_bands_hz[-1][1]] = len(frequency_bands_hz) - 1
        
        # Calculate weights based on distance from band center (smooth transition)
        # For each frequency, find its band center
        band_centers = np.zeros_like(freqs_hz)
        for i, (f_low, f_high) in enumerate(frequency_bands_hz):
            mask = (bands_discrete == i)
            if np.sum(mask) > 0:
                band_centers[mask] = (f_low + f_high) / 2.0
        
        # Distance from band center (normalized by band width)
        distances_from_center = np.abs(freqs_hz - band_centers)
        # Normalize by approximate band width (for smooth weighting)
        band_widths = np.zeros_like(freqs_hz)
        for i, (f_low, f_high) in enumerate(frequency_bands_hz):
            mask = (bands_discrete == i)
            if np.sum(mask) > 0:
                band_widths[mask] = (f_high - f_low) / 2.0
        
        normalized_distances = distances_from_center / (band_widths + 1e-12)
        # Weight: 1.0 at center, decreases to 0.5 at boundaries
        weights = 1.0 - 0.5 * np.minimum(normalized_distances, 1.0)
        
        keep = np.ones_like(bands_discrete, dtype=bool)
        unique_bands = np.unique(bands_discrete)
        
        for b in unique_bands:
            idx = np.where(bands_discrete == b)[0]
            if idx.size > max_peaks_per_band:
                # Use weighted selection instead of hard cutoff
                # Combine amplitude and position weight for smoother selection
                weighted_scores = p[idx] * weights[idx]
                # Keep top max_peaks_per_band, but with smooth weighting
                top_idx = idx[np.argsort(weighted_scores)[::-1][:max_peaks_per_band]]
                # Mark others for removal
                keep[idx] = False
                keep[top_idx] = True
        
        if keep.any():
            if hard_cap_peaks_per_band:
                # FIX 8 — drop components beyond the cap entirely so the
                # parameter name actually matches the behaviour.
                p, freqs_hz = p[keep], freqs_hz[keep]
                p_sum_hard = p.sum()
                if p_sum_hard > 0:
                    p = p / p_sum_hard
                else:
                    return dict(R_norm=0.0, P_norm=0.0, W_low=0.0, D_agn=0.0, D_peso=0.0, D_harm=None)
            else:
                # Historical behaviour: smooth down-weighting (excess * 0.1).
                p_weighted = p * np.where(keep, 1.0, weights * 0.1)
                p_sum_weighted = p_weighted.sum()
                if p_sum_weighted > 0:
                    p = p_weighted / p_sum_weighted
                else:
                    p, freqs_hz = p[keep], freqs_hz[keep]
                    if p.size > 0:
                        p = p / p.sum()
                    else:
                        return dict(R_norm=0.0, P_norm=0.0, W_low=0.0, D_agn=0.0, D_peso=0.0, D_harm=None)

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
        # Só corrigir negativos por erro numérico; não saturar a 1.0 (evita patamar artificial).
        R_norm = max(0.0, float(R_norm))

    # --- P (proximidade em Hz - PHYSICAL MODEL) ---
    if M <= 1:
        P_norm = 0.0
    else:
        # PHYSICAL MODEL: Calculate proximity in Hz (not Bark)
        # Optimize O(N²) distance calculation
        # For large spectra, use approximate method to avoid O(N²) memory
        if M > 1000:
            # Use improved sparse sampling with smooth kernel
            # This reduces complexity while maintaining smoothness
            P_num = 0.0
            sigma_sq = 2.0 * float(sigma) ** 2
            
            # Increase cutoff distance for smoother results (4*sigma instead of 3*sigma)
            # This reduces edge effects and discontinuities
            max_distance = 4.0 * float(sigma)
            
            # Use sorted Hz values for more efficient neighbor search
            sorted_indices = np.argsort(freqs_hz)
            freqs_hz_sorted = freqs_hz[sorted_indices]
            p_sorted = p[sorted_indices]
            
            for i, freq_i in enumerate(freqs_hz_sorted):
                # Binary search for nearby points (more efficient)
                left_idx = np.searchsorted(freqs_hz_sorted, freq_i - max_distance, side='left')
                right_idx = np.searchsorted(freqs_hz_sorted, freq_i + max_distance, side='right')
                
                # Get nearby indices (exclude self)
                nearby_indices = np.concatenate([
                    np.arange(left_idx, i),
                    np.arange(i + 1, right_idx)
                ])
                
                if nearby_indices.size > 0:
                    d_nearby = np.abs(freqs_hz_sorted[nearby_indices] - freq_i)
                    # Use smoother kernel with better numerical stability
                    K_nearby = np.exp(-np.clip(d_nearby ** 2 / sigma_sq, 0, 50))  # Clip to prevent underflow
                    p_nearby = p_sorted[nearby_indices]
                    P_num += float(np.sum(p_sorted[i] * p_nearby * K_nearby))
        else:
            # For small spectra, use full O(N²) calculation (more accurate)
            # Improved numerical stability and smoothness
            # Distance matrix in Hz
            d = np.abs(freqs_hz[:, None] - freqs_hz[None, :])
            
            # Use smooth diagonal handling instead of infinity
            # Set diagonal to small value instead of inf for numerical stability
            np.fill_diagonal(d, 0.0)  # Self-distance is 0
            
            # Improved kernel calculation with numerical stability
            sigma_sq = 2.0 * float(sigma) ** 2
            # Clip exponent to prevent underflow/overflow
            exponent = np.clip(-(d**2) / sigma_sq, -50, 50)
            K = np.exp(exponent)
            
            # Set diagonal to 0 (self-similarity excluded)
            np.fill_diagonal(K, 0.0)
            
            # Numerator: weighted sum of proximities
            P_num = float(np.sum((p[:, None] * p[None, :]) * K))
        
        # Denominator: Maximum possible (Simpson's index complement)
        P_den = float(1.0 - np.sum(p**2))
        
        if P_den <= 1e-12:
            P_norm = 0.0
        else:
            P_norm = min(P_num / P_den, 1.0)

    # --- W (peso baixo-freq: partilha em frequências baixas - PHYSICAL MODEL) ---
    # PHYSICAL MODEL: Use Hz-based frequency bands (not Bark)
    # Calculate energy in low frequencies (below low_hz_cut)
    # Use smooth weighting based on distance from cutoff frequency
    low_freq_mask = freqs_hz <= float(low_hz_cut)
    
    if np.sum(low_freq_mask) > 0:
        # Calculate energy in low frequencies
        # Use smooth transition near cutoff (not hard cutoff)
        # Weight decreases gradually as frequency approaches cutoff
        cutoff_freq = float(low_hz_cut)
        transition_width = cutoff_freq * 0.1  # 10% transition width
        
        # Calculate weights: 1.0 for frequencies well below cutoff, decreasing near cutoff
        freq_distances = cutoff_freq - freqs_hz
        weights_low = np.clip(freq_distances / transition_width, 0.0, 1.0)
        weights_low = np.where(freq_distances > 0, weights_low, 0.0)  # Only below cutoff
        
        # Weighted energy in low frequencies
        E_low = float(np.sum(p * weights_low))
    else:
        E_low = 0.0
    
    # Total energy
    E_total = float(np.sum(p))
    
    if E_total <= 1e-12:
        W_low = 0.0
    else:
        W_low = float(E_low / E_total)  # 0..1

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

    # --- Calculate D_harm (harmonic density) ---
    D_harm = None
    if (f0_hz is not None) and np.isfinite(f0_hz) and f0_hz > 0:
        try:
            # Identify harmonic frequencies (multiples of f0)
            # Tolerance: ±2% of harmonic frequency (adaptive tolerance)
            tolerance_factor = 0.02
            harmonic_indices = []
            
            for n in range(1, int(20000.0 / f0_hz) + 1):  # Up to Nyquist
                harmonic_freq = n * f0_hz
                tolerance = harmonic_freq * tolerance_factor
                # Find frequencies within tolerance of this harmonic
                mask = np.abs(freqs_hz - harmonic_freq) <= tolerance
                if np.any(mask):
                    # Use the closest frequency to the harmonic
                    idx = np.argmin(np.abs(freqs_hz[mask] - harmonic_freq))
                    harmonic_indices.append(np.where(mask)[0][idx])
            
            if len(harmonic_indices) > 0:
                harmonic_amps = amps[np.array(harmonic_indices)]
                wf_h = (weight_function or "linear").strip().lower()
                if wf_h == "d2":
                    wf_h = "linear"
                elif wf_h == "d8":
                    wf_h = "d17"
                harm_freqs = freqs_hz[np.array(harmonic_indices)]
                _d_kw: Dict[str, Any] = {
                    "weight_function": wf_h,
                    "normalize": False,
                    "remove_noise": False,
                    "prevent_domination": True,
                }
                if wf_h in DISCRETE_SPECTRAL_METRIC_KEYS and harm_freqs.size == harmonic_amps.size:
                    _d_kw["frequencies"] = harm_freqs
                D_harm = float(apply_density_metric(harmonic_amps, **_d_kw))
                # No clipping - D_harm can be > 1.0 for rich harmonic sounds
                # More harmonics = higher D_harm (preserves variation)
                # Only ensure non-negative
                D_harm = max(0.0, D_harm)
        except Exception as e:
            # If calculation fails, leave as None
            logger.debug(f"Error calculating D_harm: {e}")
            D_harm = None

    return dict(
        R_norm=float(R_norm), 
        P_norm=float(P_norm),
        W_low=float(W_low), 
        D_agn=float(D_core), 
        D_peso=float(D_peso),
        D_harm=D_harm
    )

# --- FIM NOVO ---


# Exportar funções públicas do módulo
__all__ = [
    # Classes
    'SpectralDensityMetrics',
    'WeightFunction',

    "DEFAULT_HARMONIC_ROLLOFF_ALPHA",
    "DEFAULT_ROLLOFF_COMPENSATED_DENSITY_WEIGHT_FUNCTION",
    "compute_rolloff_compensated_harmonic_density",
    "compute_harmonic_effective_power_density",
    "compute_harmonic_effective_power_mass",
    "compute_expected_harmonic_slot_count",
    "compute_harmonic_occupancy_ratio",
    "compute_residual_log_frequency_occupancy",
    
    # Funções principais
    'apply_density_metric',
    'apply_density_metric_df',
    'compute_discrete_spectral_metrics_bundle',
    'band_partial_metric_sum',
    'partial_metric_sums_h_i_s_total',
    'DISCRETE_SPECTRAL_METRIC_KEYS',
    'calculate_harmonic_density',
    'calculate_inharmonic_density',
    'compute_spectral_entropy',
    'calculate_combined_density_metric',
    'calculate_perceptual_spectral_density',
    'calculate_spectral_complexity',
    'calculate_harmonic_richness',
    'calculate_spectral_density_corrected',
    'spectral_density',              # <-- NOVO
    'physical_spectral_density',
    
    # PHASE 1: Spectral Smoothing Functions
    'apply_spectral_smoothing',
    'estimate_noise_floor',
    'effective_partial_density_from_powers',
    'aggregate_low_frequency_residual_peak_power',
    'aggregate_subbass_noise_peak_power',
    'partial_density_effective_components_bundle',
    'partial_density_effective_components',
    
    # Funções auxiliares
    'get_weight_function',
    'identify_nonharmonic_residual_rows',
    'identify_inharmonic_partials',
    'compare_with_sethares_dissonance',
    'plot_harmonic_spectrum'
]

