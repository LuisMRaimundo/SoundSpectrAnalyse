# spectral_power.py - Módulo melhorado

"""
Módulo para análise de potência espectral.

Este módulo fornece funções para carregar arquivos de áudio,
calcular a potência espectral usando FFT com várias opções de janelamento,
e visualizar os resultados como gráficos 2D e 3D.

Melhorias:
- Documentação expandida com exemplos
- Validação de parâmetros mais robusta
- Tratamento de erros aprimorado
- Otimização de desempenho
- Suporte a mais tipos de janelas
- Visualizações melhoradas
- Normalização e filtragem opcionais
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict, Union, Any
from pathlib import Path
import soundfile as sf
from scipy import signal as sp_sig

# Configurar logging
from log_config import configure_root_logger
configure_root_logger()            # <-- NUNCA duplifica
logger = logging.getLogger(__name__)


def get_window_function(name: str, n: int):
    try:
        from scipy.signal import get_window
        return get_window(name, n, fftbins=True)
    except Exception:
        import numpy as _np
        return _np.hanning(n)


# Constantes
DEFAULT_N_FFT = 8192
DEFAULT_DPI = 300
VALID_WINDOW_TYPES = [
    'hann', 'hamming', 'bartlett', 'blackmanharris',
    'bohman', 'boxcar', 'cosine', 'flattop', 'kaiser', 'nuttall',
    'parzen', 'triang', 'tukey'
]
# Adicione esta classe ao início do arquivo spectral_power.py (após as importações)
class LinearTimeFrequencyTransform:
    """
    A Linear Time-Frequency Transform (LFT) implementation based on principles
    from Kristoffer Jensen's work on sinusoidal modeling.
    
    This implementation combines elements of windowing and sinusoidal analysis
    to provide time-frequency representations of signals with improved time resolution
    compared to standard FFT-based approaches while maintaining frequency accuracy.
    """
    
    def __init__(self, window_size=1024, hop_size=256, window_type='hann', 
                 zero_padding=1, min_freq=0, max_freq=None):
        """
        Initialize the LFT parameters.
        
        Args:
            window_size: Size of the analysis window in samples
            hop_size: Number of samples between consecutive frames
            window_type: Type of window function ('hann', 'hamming', etc.)
            zero_padding: Factor for zero padding (1 = no padding)
            min_freq: Minimum frequency to analyze (Hz)
            max_freq: Maximum frequency to analyze (Hz, default=Nyquist)
        """
        self.window_size = window_size
        self.hop_size = hop_size
        self.window_type = window_type
        self.zero_padding = zero_padding
        self.min_freq = min_freq
        self.max_freq = max_freq
        
        # Create window function
        self.window = get_window_function(window_type, window_size)
        
        # Calculate FFT size with zero padding
        self.fft_size = window_size * zero_padding
        
    def transform(self, x, fs):
        """
        Transform a signal using the Linear Time-Frequency Transform.
    
        Args:
            x: Input signal (1D array)
            fs: Sampling frequency (Hz)
    
        Returns:
            tuple containing:
            - time vector (1D array)
            - frequency vector (1D array)
            - time-frequency representation (2D complex array: freq x time)
        """
        # Calculate number of frames
        num_frames = max(1, 1 + (len(x) - self.window_size) // self.hop_size)

        # Create time and frequency vectors
        time = np.arange(num_frames) * self.hop_size / fs

        # Calculate frequency bins - use actual fs, not a fixed value
        freq = np.fft.rfftfreq(self.fft_size, 1/fs)
    
        # MODIFICAÇÃO: Adicionar log para debug do range de frequências
        logger.debug(f"LFT range de frequências: {freq[0]:.2f} - {freq[-1]:.2f} Hz, resolução: {freq[1] - freq[0]:.2f} Hz")
    
        # Filter frequency range if needed
        if self.max_freq is None:
            self.max_freq = fs / 2
    
        # MODIFICAÇÃO: Garantir que a frequência mínima seja baixa o suficiente
        if self.min_freq > 20:
            original_min = self.min_freq
            self.min_freq = 20
            logger.warning(f"Ajustando min_freq de {original_min} para {self.min_freq} Hz para melhor detecção de baixas frequências")
    
        # MODIFICAÇÃO: Verificar se o zero_padding está adequado para baixas frequências
        min_resolution_needed = 1.0  # 1 Hz de resolução desejada
        current_resolution = freq[1] - freq[0]
        if current_resolution > min_resolution_needed and self.zero_padding == 1:
            logger.warning(f"Resolução de frequência atual ({current_resolution:.2f} Hz) pode ser insuficiente para baixas frequências. Considere aumentar zero_padding.")

        freq_mask = (freq >= self.min_freq) & (freq <= self.max_freq)
        freq = freq[freq_mask]

        # Initialize output matrix
        S = np.zeros((len(freq), num_frames), dtype=complex)

        # Process each frame
        for i in range(num_frames):
            # Extract frame
            start = i * self.hop_size
            end = start + self.window_size
    
            # Handle boundary conditions
            if end > len(x):
                frame = np.zeros(self.window_size)
                frame[:len(x)-start] = x[start:]
            else:
                frame = x[start:end]
    
            # Apply window
            frame_windowed = frame * self.window
    
            # Zero padding
            if self.zero_padding > 1:
                frame_padded = np.zeros(self.fft_size)
                frame_padded[:self.window_size] = frame_windowed
            else:
                frame_padded = frame_windowed
    
            # Compute FFT
            frame_fft = np.fft.rfft(frame_padded)
    
            # Store filtered frequency components
            S[:, i] = frame_fft[freq_mask]
    
        # MODIFICAÇÃO: Adicionar log com informações sobre os dados processados
        logger.debug(f"LFT processada: {num_frames} frames, {len(freq)} bins de frequência, faixa: {freq[0]:.2f}-{freq[-1]:.2f} Hz")
    
        return time, freq, S
    
    def magnitude(self, S):
        """
        Compute magnitude of the transform.
        
        Args:
            S: Time-frequency representation from transform()
            
        Returns:
            Magnitude spectrogram
        """
        return np.abs(S)
    
    def phase(self, S):
        """
        Compute phase of the transform.
        
        Args:
            S: Time-frequency representation from transform()
            
        Returns:
            Phase spectrogram
        """
        return np.angle(S)
    
    def power(self, S):
        """
        Compute power of the transform.
        
        Args:
            S: Time-frequency representation from transform()
            
        Returns:
            Power spectrogram
        """
        return np.abs(S)**2


# Adicione esta função para cálculo de potência espectral usando LFT
def spectral_power_lft(
    signal: np.ndarray,
    fs: int,                                  # <-- OBRIGATÓRIO: taxa de amostragem real
    n_fft: int = DEFAULT_N_FFT,
    hop_length: Optional[int] = None,
    window_type: str = 'hann',
    order: int = 30,
    normalize: bool = False,
    remove_dc: bool = True,
    zero_padding: int = 1,
    time_avg: str = 'mean',
    window_kwargs: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Calcula a potência espectral de um sinal usando Linear Time-Frequency Transform (LFT)
    e retorna a potência dos primeiros 'order' componentes.

    Args:
        signal: array 1D com o sinal de áudio (mono).
        fs: taxa de amostragem (Hz). Tem de ser > 0.
        n_fft: tamanho da janela (amostras).
        hop_length: passo entre janelas; se None, usa n_fft//4.
        window_type: tipo de janela ('hann', 'hamming', ...).
        order: nº de componentes a retornar.
        normalize: normaliza magnitude se True.
        remove_dc: remove componente DC se True.
        zero_padding: fator de zero-padding (1 = sem padding).
        time_avg: agregação temporal ('mean'|'median').
        window_kwargs: parâmetros extra para a janela (se aplicável).

    Returns:
        np.ndarray com 'order' valores (dB) da potência espectral LFT.

    Raises:
        ValueError: se fs <= 0 ou parâmetros inválidos.
    """
    # --- validação de entrada
    if fs is None or fs <= 0:
        raise ValueError("spectral_power_lft: 'fs' é obrigatório e tem de ser > 0.")
    if signal is None:
        logger.warning("Sinal None fornecido para spectral_power_lft")
        return np.zeros(order)
    if len(signal) == 0:
        logger.warning("Sinal vazio fornecido para spectral_power_lft")
        return np.zeros(order)
    if np.isnan(signal).any() or np.isinf(signal).any():
        logger.warning("Sinal contém NaN/Inf; retornando zeros.")
        return np.zeros(order)

    # parâmetros derivados
    if hop_length is None:
        hop_length = max(1, n_fft // 4)
    if order <= 0:
        logger.warning("Parâmetro 'order' <= 0; ajustado para 1.")
        order = 1

    try:
        # Inicializa LFT com os parâmetros pedidos
        lft = LinearTimeFrequencyTransform(
            window_size=n_fft,
            hop_size=hop_length,
            window_type=window_type,
            zero_padding=zero_padding
        )

        # Transformada LFT usa a taxa de amostragem REAL
        time, freq, S = lft.transform(signal, fs=fs)

        # Magnitude em dB
        magnitude = lft.magnitude(S)
        if normalize:
            mag_max = np.max(magnitude) if magnitude.size else 1.0
            if mag_max > 0:
                magnitude = magnitude / mag_max

        # Agregação temporal
        if time_avg == 'median':
            spec = np.median(magnitude, axis=1)
        else:
            spec = np.mean(magnitude, axis=1)

        # Remoção de DC se pedido
        if remove_dc and spec.size > 0:
            spec[0] = 0.0

        # Ordenar por potência decrescente e devolver top 'order'
        idx_sorted = np.argsort(spec)[::-1]
        top = spec[idx_sorted[:order]]

        # Converter para dB de forma robusta
        eps = 1e-12
        result = 10.0 * np.log10(np.maximum(top, eps))

        # Garantir comprimento 'order'
        if result.shape[0] < order:
            result = np.pad(result, (0, order - result.shape[0]),
                            mode='constant', constant_values=-100.0)

        logger.debug(f"Potência espectral LFT calculada com fs={fs} Hz e {order} componentes")
        return result

    except Exception as e:
        logger.error(f"Erro ao calcular potência espectral LFT: {e}")
        import traceback as _tb
        logger.debug(f"Stack trace: {_tb.format_exc()}")
        return np.zeros(order)


# Adicione esta função para plotar espectrograma LFT
def plot_lft_spectrogram(
    signal: np.ndarray,
    fs: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: Optional[int] = None,
    window_type: str = 'hann',
    zero_padding: int = 1,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = DEFAULT_DPI,
    freq_range: Optional[Tuple[float, float]] = None,
    cmap: str = 'viridis'
) -> None:
    """
    Plota um espectrograma usando a Transformada Linear de Tempo-Frequência (LFT).
    
    Args:
        signal: O sinal de áudio (array NumPy 1D).
        fs: Taxa de amostragem em Hz.
        n_fft: Número de pontos na janela de análise.
        hop_length: Tamanho do salto (default=None => n_fft//4).
        window_type: Tipo de janela a aplicar.
        zero_padding: Fator de zero padding para aumentar resolução de frequência.
        save_path: Caminho para salvar o gráfico.
        show_plot: Se True, exibe o gráfico na tela.
        title: Título do gráfico.
        figsize: Tamanho da figura (largura, altura) em polegadas.
        dpi: Resolução do gráfico em pontos por polegada.
        freq_range: Intervalo de frequências a mostrar (min, max) em Hz.
        cmap: Mapa de cores para o espectrograma.
    """
    if signal is None or len(signal) == 0:
        logger.error("Sinal vazio ou None fornecido para plot_lft_spectrogram")
        return
        
    # Definir hop_length padrão se não fornecido
    if hop_length is None:
        hop_length = n_fft // 4
        
    try:
        # Inicializar o objeto LFT
        lft = LinearTimeFrequencyTransform(
            window_size=n_fft,
            hop_size=hop_length,
            window_type=window_type,
            zero_padding=zero_padding
        )
        
        # Calcular transformada LFT
        time, freq, S = lft.transform(signal, fs=fs)
        
        # Converter para magnitude em dB
        magnitude = lft.magnitude(S)
        magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
        
        # Configurar limites de frequência
        freq_min, freq_max = 0, fs/2
        if freq_range is not None:
            freq_min, freq_max = freq_range
            if freq_min < 0:
                freq_min = 0
            if freq_max > fs/2:
                freq_max = fs/2
                
        # Encontrar índices que correspondem ao intervalo de frequência
        freq_indices = np.logical_and(freq >= freq_min, freq <= freq_max)
        
        # Filtrar dados para plotagem
        freq_plot = freq[freq_indices]
        magnitude_db_plot = magnitude_db[freq_indices, :]
        
        # Criar figura
        plt.figure(figsize=figsize)
        
        # Plotar espectrograma
        plt.pcolormesh(time, freq_plot, magnitude_db_plot, cmap=cmap, shading='gouraud')
        
        # Configurar rótulos e título
        plt.xlabel('Tempo (s)')
        plt.ylabel('Frequência (Hz)')
        
        if title is None:
            title = 'Espectrograma LFT'
        plt.title(title)
        
        # Adicionar barra de cores
        plt.colorbar(label='Magnitude (dB)')
        
        # Escala logarítmica para o eixo y (frequência)
        plt.yscale('log')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar o gráfico, se caminho fornecido
        if save_path:
            save_path = Path(save_path)
            
            # Garantir que o diretório existe
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Espectrograma LFT salvo em: {save_path}")

        # Mostrar o gráfico, se solicitado
        if show_plot:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Erro ao plotar espectrograma LFT: {e}")
        plt.close()


# Adicione esta função para comparar FFT e LFT
def compare_fft_lft(
    x: np.ndarray,
    fs: int,
    n_fft: int = DEFAULT_N_FFT,
    window_type: str = 'hann',
    zero_padding: int = 1,
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = DEFAULT_DPI
) -> None:
    """
    Compara resultados de análise FFT (STFT) e LFT para o mesmo sinal.

    Args:
        x: Sinal de áudio mono (array NumPy 1D).
        fs: Taxa de amostragem em Hz.
        n_fft: Tamanho da janela/FFT.
        window_type: Tipo de janela.
        zero_padding: Fator de zero padding para LFT.
        save_path: Caminho para guardar a figura (opcional).
        show_plot: Se True, mostra a figura na tela.
        figsize: Tamanho da figura.
        dpi: Resolução (DPI) do ficheiro guardado.
    """
    # validações básicas
    if x is None or len(x) == 0:
        logger.error("Sinal vazio ou None fornecido para compare_fft_lft")
        return
    if fs is None or fs <= 0:
        logger.error("Taxa de amostragem (fs) inválida em compare_fft_lft")
        return

    try:
        # --- Espectros (1D) estáticos ---
        # NOTA: assegure-se de que spectral_power(...) e spectral_power_lft(...)
        #       devolvem valores coerentes. Se devolverem magnitudes lineares,
        #       aqui convertemos para dB. Se já devolverem dB, remova a conversão.
        sp_fft = spectral_power(x, n_fft=n_fft, window_type=window_type)  # 1D
        sp_lft = spectral_power_lft(
            x,
            fs=fs,                               # <-- PASSAR fs correto (não fixo)
            n_fft=n_fft,
            window_type=window_type,
            zero_padding=zero_padding
        )  # 1D

        # Converter para dB se estiverem em linear (ajuste se já vier em dB)
        eps = np.finfo(float).eps
        if np.any(sp_fft > 0) and np.any(sp_lft > 0):
            sp_fft_db = 20 * np.log10(np.maximum(sp_fft, eps))
            sp_lft_db = 20 * np.log10(np.maximum(sp_lft, eps))
        else:
            # fallback robusto: não assumir linear se não souber
            sp_fft_db = sp_fft
            sp_lft_db = sp_lft

        # --- Espectrogramas (2D) ---
        hop_length = max(1, n_fft // 4)

        # janela: use utilitário local se existir; caso contrário, SciPy
        try:
            window = get_window_function(window_type, n_fft)  # se existir no seu código
        except NameError:
            window = sp_sig.get_window(window_type, n_fft, fftbins=True)

        # STFT (SciPy) — usar alias sp_sig (evita sombra com variável 'x')
        f_fft, t_fft, Zxx = sp_sig.stft(
            x,
            fs=fs,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            window=window,
            return_onesided=True,
            boundary='zeros',
            padded=True
        )
        spec_fft_db = 20 * np.log10(np.maximum(np.abs(Zxx), eps))  # magnitude→dB

        # LFT
        lft = LinearTimeFrequencyTransform(
            window_size=n_fft,
            hop_size=hop_length,
            window_type=window_type,
            zero_padding=zero_padding
        )
        t_lft, f_lft, S_lft = lft.transform(x, fs=fs)

        # Magnitude LFT → dB
        # Se lft.magnitude() existir e devolver |.|, pode usar: mag_lft = lft.magnitude(S_lft)
        mag_lft = np.abs(S_lft)
        spec_lft_db = 20 * np.log10(np.maximum(mag_lft, eps))

        # Garantir orientação [freq, tempo] para pcolormesh
        if spec_lft_db.shape == (len(t_lft), len(f_lft)):
            spec_lft_db = spec_lft_db.T  # transpor para [freq, tempo]
        elif spec_lft_db.shape != (len(f_lft), len(t_lft)):
            logger.warning(
                f"Dimensões inesperadas do espectrograma LFT: {spec_lft_db.shape} "
                f"(esperado {(len(f_lft), len(t_lft))} ou {(len(t_lft), len(f_lft))})"
            )

        # --- Plotagem ---
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # 1) Espectro FFT
        axes[0, 0].plot(sp_fft_db)
        axes[0, 0].set_title('Espectro FFT')
        axes[0, 0].set_xlabel('Bin de Frequência')
        axes[0, 0].set_ylabel('Magnitude (dB)')
        axes[0, 0].grid(True, alpha=0.3)

        # 2) Espectro LFT
        axes[0, 1].plot(sp_lft_db)
        axes[0, 1].set_title('Espectro LFT')
        axes[0, 1].set_xlabel('Bin de Frequência')
        axes[0, 1].set_ylabel('Magnitude (dB)')
        axes[0, 1].grid(True, alpha=0.3)

        # 3) Espectrograma FFT
        im1 = axes[1, 0].pcolormesh(t_fft, f_fft, spec_fft_db, shading='gouraud', cmap='viridis')
        axes[1, 0].set_title('Espectrograma FFT (STFT)')
        axes[1, 0].set_xlabel('Tempo (s)')
        axes[1, 0].set_ylabel('Frequência (Hz)')
        axes[1, 0].set_yscale('log')
        fig.colorbar(im1, ax=axes[1, 0], label='Magnitude (dB)')

        # 4) Espectrograma LFT
        im2 = axes[1, 1].pcolormesh(t_lft, f_lft, spec_lft_db, shading='gouraud', cmap='viridis')
        axes[1, 1].set_title('Espectrograma LFT')
        axes[1, 1].set_xlabel('Tempo (s)')
        axes[1, 1].set_ylabel('Frequência (Hz)')
        axes[1, 1].set_yscale('log')
        fig.colorbar(im2, ax=axes[1, 1], label='Magnitude (dB)')

        # título global e layout
        fig.suptitle('Comparação FFT vs. LFT', fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        # guardar/mostrar
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Comparação FFT vs. LFT salva em: {save_path}")

        if show_plot:
            plt.show()

        plt.close()

    except Exception as e:
        logger.error(f"Erro ao comparar FFT e LFT: {e}")
        import traceback
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        plt.close()



def load_sound(file_path: Union[str, Path]) -> Tuple[Optional[np.ndarray], Optional[int]]:
    """
    Carrega um arquivo de áudio e retorna o sinal (como mono) e taxa de amostragem.

    Args:
        file_path: Caminho para o arquivo de áudio.

    Returns:
        Tupla (sinal, fs) onde 'sinal' é um array NumPy 1D contendo as amostras
        de áudio, e 'fs' é a taxa de amostragem. Se o carregamento falhar,
        retorna (None, None).
        
    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    file_path = Path(file_path)
    
    # Verificar se o arquivo existe
    if not file_path.exists():
        logger.error(f"Arquivo não encontrado: {file_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    
    try:
        # Tentar carregar o arquivo
        signal_data, fs = sf.read(file_path)
        
        # Verificar se temos dados válidos
        if len(signal_data) == 0:
            logger.warning(f"Arquivo de áudio vazio: {file_path}")
            return None, None
            
        # Converter para mono se for multicanal
        if signal_data.ndim > 1:
            # Média dos canais, ajustada por sqrt(2) para preservar energia
            signal_data = np.mean(signal_data, axis=1) / np.sqrt(2)
            logger.debug(f"Arquivo multicanal convertido para mono: {file_path}")
            
        logger.info(f"Arquivo de áudio carregado: {file_path} (taxa: {fs} Hz, duração: {len(signal_data)/fs:.2f}s)")
        return signal_data, fs
        
    except Exception as e:
        logger.error(f"Erro ao carregar áudio de {file_path}: {e}")
        return None, None


import numpy as np
import logging

logger = logging.getLogger(__name__)

# Conjunto com as janelas mais usuais (apenas para mensagem de erro)
_COMMON_WINDOWS = {
    "hann", "hamming", "bartlett", "boxcar",
    "blackmanharris", "bohman", "cosine", "flattop",
    "kaiser", "nuttall", "parzen", "triang", "tukey"
}

def get_window_function(window_type: str, n_samples: int, **kwargs) -> np.ndarray:
    """
    Devolve uma janela de comprimento ``n_samples``.

    O nome da janela é indiferente a maiúsculas/minúsculas.  
    Valida os argumentos e gera mensagens de erro explícitas.

    Parameters
    ----------
    window_type : str
        Nome da janela (e.g. 'hann', 'hamming', 'kaiser' …).
    n_samples : int
        Número de amostras.
    **kwargs :
        Argumentos extra aceites por :pyfunc:`scipy.sp_sig.get_window`
        (p.ex. ``beta`` para 'kaiser', ``alpha`` para 'tukey').

    Returns
    -------
    np.ndarray
        Vector 1-D ``float64`` com a janela requerida.

    Raises
    ------
    ValueError
        Se ``window_type`` for desconhecido ou ``n_samples`` < 1.
    """
    if n_samples < 1:
        raise ValueError("n_samples tem de ser ≥ 1")

    name = (window_type or "hann").lower()
    # Definir defaults seguros para janelas com parâmetros
    if window_type is None:
        name = 'hann'
    name = (window_type or 'hann').lower()
    if name in ('kaiser',):
        # SciPy aceita ('kaiser', beta) ou string 'kaiser' com beta por defeito=14
        kwargs = {} if kwargs is None else dict(kwargs)
        kwargs.setdefault('beta', 14.0)
        window = sp_sig.get_window(('kaiser', float(kwargs['beta'])), n_samples, fftbins=True)
    elif name in ('gaussian','gauss','gaussiana'):
        # Gaussian requer std obrigatória; usar std = n_samples/8 como default pragmático
        kwargs = {} if kwargs is None else dict(kwargs)
        std = float(kwargs.get('std', n_samples/8.0))
        window = sp_sig.get_window(('gaussian', std), n_samples, fftbins=True)
    else:
        window = sp_sig.get_window(name, n_samples, fftbins=True)
    logger.debug("Janela '%s' criada com %d amostras", name, n_samples)
    return window.astype(np.float64, copy=False)



def spectral_power(
    signal: np.ndarray,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: Optional[int] = None,
    window_type: str = "hann",
    order: int = 30,
    normalize: bool = False,
    remove_dc: bool = True,
    window_kwargs: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Potência espectral média por STFT:
      - calcula |X|^2 em cada frame (rfft)
      - faz média temporal em potência linear
      - converte para dB com 10*log10

    Nota: esta função devolve potência relativa (não calibrada).
    Para comparações, use sempre a mesma parametrização (janela/n_fft/hop).
    """

    # ---------- validações básicas ----------
    if signal is None:
        return np.zeros(order, dtype=float)

    signal = np.asarray(signal, dtype=float)
    if signal.size == 0:
        return np.zeros(order, dtype=float)

    # limpar NaN/Inf
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    if n_fft <= 0:
        raise ValueError("n_fft deve ser positivo")

    if hop_length is None or hop_length <= 0:
        hop_length = n_fft // 4

    if window_kwargs is None:
        window_kwargs = {}

    # ---------- opcional: ajustar n_fft para potência de 2 ----------
    if n_fft & (n_fft - 1) != 0:
        n_fft = 2 ** (n_fft.bit_length())

    # bins do rfft (inclui Nyquist)
    max_bins = (n_fft // 2) + 1
    if order <= 0:
        raise ValueError("order deve ser positivo")
    order = min(order, max_bins)

    # ---------- janela ----------
    window = get_window_function(window_type, n_fft, **window_kwargs)
    window = np.asarray(window, dtype=float)

    # energia da janela (normalização crucial)
    window_energy = float(np.sum(window ** 2))
    if window_energy <= 0.0 or not np.isfinite(window_energy):
        window_energy = 1.0  # fallback seguro (não deveria acontecer)

    # ---------- garantir que há pelo menos 1 frame ----------
    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size), mode="constant")

    # opcional: incluir último frame com padding (mais estável)
    remainder = (signal.size - n_fft) % hop_length
    if remainder != 0:
        pad = hop_length - remainder
        signal = np.pad(signal, (0, pad), mode="constant")

    # ---------- STFT manual: potência por frame ----------
    frames_power = []
    last_start = signal.size - n_fft
    for start in range(0, last_start + 1, hop_length):
        frame = signal[start:start + n_fft]
        # frame.size deve ser n_fft, mas deixo segurança:
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size), mode="constant")

        frame = frame * window
        X = np.fft.rfft(frame, n=n_fft)

        # potência normalizada pela energia da janela
        p = (np.abs(X) ** 2) / window_energy
        frames_power.append(p)

    if len(frames_power) == 0:
        return np.zeros(order, dtype=float)

    # ---------- média temporal em potência ----------
    power_mean = np.mean(np.stack(frames_power, axis=0), axis=0)

    # remover DC
    if remove_dc and power_mean.size > 0:
        power_mean[0] = 0.0

    # ---------- converter para dB (potência) ----------
    eps = 1e-20
    spectral_power_db = 10.0 * np.log10(np.maximum(power_mean, eps))

    spectral_power_db = np.nan_to_num(
        spectral_power_db, nan=-200.0, posinf=0.0, neginf=-200.0
    )

    # normalização opcional (apenas para visualização)
    if normalize:
        lo = float(np.min(spectral_power_db))
        hi = float(np.max(spectral_power_db))
        if hi > lo:
            spectral_power_db = (spectral_power_db - lo) / (hi - lo)
        else:
            spectral_power_db = np.zeros_like(spectral_power_db)

    return spectral_power_db[:order]




def compute_spectral_centroid(
    signal: np.ndarray,
    fs: int,
    n_fft: int = DEFAULT_N_FFT,
    window_type: str = 'hann'
) -> float:
    """
    Calcula o centroide espectral de um sinal de áudio.
    
    O centroide espectral é uma medida que indica onde está o "centro de massa"
    do espectro, ou seja, a frequência média ponderada pela amplitude.
    
    Args:
        signal: O sinal de áudio (array NumPy 1D).
        fs: Taxa de amostragem em Hz.
        n_fft: Número de pontos FFT.
        window_type: Tipo de janela a aplicar.
        
    Returns:
        Centroide espectral em Hz.
    """
    if signal is None or len(signal) == 0:
        logger.warning("Sinal vazio fornecido para compute_spectral_centroid")
        return 0.0
        
    try:
        # Aplicar janela
        window = get_window_function(window_type, min(len(signal), n_fft))
        if len(signal) > len(window):
            windowed_signal = signal[:len(window)] * window
        else:
            windowed_signal = signal * window[:len(signal)]
            
        # Calcular FFT
        magnitude_spectrum = np.abs(np.fft.rfft(windowed_signal, n=n_fft))
        
        # Frequências correspondentes
        freqs = np.fft.rfftfreq(n_fft, d=1/fs)
        
        # Calcular centroide (média ponderada)
        if np.sum(magnitude_spectrum) > 0:
            centroid = np.sum(freqs * magnitude_spectrum) / np.sum(magnitude_spectrum)
            return centroid
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Erro ao calcular centroide espectral: {e}")
        return 0.0


def compute_spectral_flatness(
    signal: np.ndarray,
    n_fft: int = DEFAULT_N_FFT,
    window_type: str = 'hann'
) -> float:
    """
    Calcula a planura espectral (spectral flatness) de um sinal.
    
    A planura espectral é a razão entre a média geométrica e a média aritmética
    do espectro de potência, indicando quão similar o sinal é a um ruído branco.
    Valores próximos a 1 indicam ruído, valores próximos a 0 indicam tons.
    
    Args:
        signal: O sinal de áudio (array NumPy 1D).
        n_fft: Número de pontos FFT.
        window_type: Tipo de janela a aplicar.
        
    Returns:
        Valor de planura espectral entre 0 e 1.
    """
    if signal is None or len(signal) == 0:
        logger.warning("Sinal vazio fornecido para compute_spectral_flatness")
        return 0.0
        
    try:
        # Aplicar janela
        window = get_window_function(window_type, min(len(signal), n_fft))
        if len(signal) > len(window):
            windowed_signal = signal[:len(window)] * window
        else:
            windowed_signal = signal * window[:len(signal)]
            
        # Calcular espectro de potência
        power_spectrum = np.abs(np.fft.rfft(windowed_signal, n=n_fft)) ** 2
        
        # Remover valores muito próximos de zero para evitar problemas com log
        power_spectrum = np.maximum(power_spectrum, 1e-10)
        
        # Calcular média geométrica e aritmética
        geometric_mean = np.exp(np.mean(np.log(power_spectrum)))
        arithmetic_mean = np.mean(power_spectrum)
        
        # Calcular planura espectral
        if arithmetic_mean > 0:
            flatness = geometric_mean / arithmetic_mean
            return flatness
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Erro ao calcular planura espectral: {e}")
        return 0.0


def compute_spectral_rolloff(
    signal: np.ndarray,
    fs: int,
    n_fft: int = DEFAULT_N_FFT,
    window_type: str = 'hann',
    rolloff_threshold: float = 0.85
) -> float:
    """
    Calcula a frequência de rolloff espectral.
    
    A frequência de rolloff é o ponto no qual uma certa porcentagem da 
    energia espectral total está contida abaixo dessa frequência.
    
    Args:
        signal: O sinal de áudio (array NumPy 1D).
        fs: Taxa de amostragem em Hz.
        n_fft: Número de pontos FFT.
        window_type: Tipo de janela a aplicar.
        rolloff_threshold: Limiar de energia (padrão: 0.85 = 85%).
        
    Returns:
        Frequência de rolloff em Hz.
    """
    if signal is None or len(signal) == 0:
        logger.warning("Sinal vazio fornecido para compute_spectral_rolloff")
        return 0.0
        
    # Validar threshold
    if rolloff_threshold <= 0 or rolloff_threshold >= 1:
        logger.error(f"Limiar de rolloff inválido: {rolloff_threshold}")
        raise ValueError("O limiar de rolloff deve estar entre 0 e 1")
        
    try:
        # Aplicar janela
        window = get_window_function(window_type, min(len(signal), n_fft))
        if len(signal) > len(window):
            windowed_signal = signal[:len(window)] * window
        else:
            windowed_signal = signal * window[:len(signal)]
            
        # Calcular espectro de potência
        power_spectrum = np.abs(np.fft.rfft(windowed_signal, n=n_fft)) ** 2
        
        # Frequências correspondentes
        freqs = np.fft.rfftfreq(n_fft, d=1/fs)
        
        # Calcular energia cumulativa
        cumulative_power = np.cumsum(power_spectrum)
        total_power = np.sum(power_spectrum)
        
        # Encontrar frequência de rolloff
        if total_power > 0:
            threshold_energy = rolloff_threshold * total_power
            rolloff_index = np.where(cumulative_power >= threshold_energy)[0][0]
            return freqs[rolloff_index]
        else:
            return 0.0
            
    except Exception as e:
        logger.error(f"Erro ao calcular rolloff espectral: {e}")
        return 0.0


def compute_spectral_contrast(
    signal: np.ndarray,
    fs: int,
    n_fft: int = DEFAULT_N_FFT,
    window_type: str = 'hann',
    n_bands: int = 6,
    quantile: float = 0.02
) -> np.ndarray:
    """
    Calcula o contraste espectral em várias bandas de frequência.
    
    O contraste espectral estima a diferença entre picos e vales
    no espectro para diferentes bandas de frequência.
    
    Args:
        signal: O sinal de áudio (array NumPy 1D).
        fs: Taxa de amostragem em Hz.
        n_fft: Número de pontos FFT.
        window_type: Tipo de janela a aplicar.
        n_bands: Número de bandas de frequência.
        quantile: Quantil para considerar como vale (0 a 1).
        
    Returns:
        Array com os valores de contraste para cada banda.
    """
    if signal is None or len(signal) == 0:
        logger.warning("Sinal vazio fornecido para compute_spectral_contrast")
        return np.zeros(n_bands)
        
    # Validar parâmetros
    if n_bands <= 0:
        logger.error(f"Número de bandas inválido: {n_bands}")
        raise ValueError("O número de bandas deve ser positivo")
        
    if quantile <= 0 or quantile >= 1:
        logger.error(f"Valor de quantil inválido: {quantile}")
        raise ValueError("O quantil deve estar entre 0 e 1")
        
    try:
        # Aplicar janela
        window = get_window_function(window_type, min(len(signal), n_fft))
        if len(signal) > len(window):
            windowed_signal = signal[:len(window)] * window
        else:
            windowed_signal = signal * window[:len(signal)]
            
        # Calcular espectro de potência
        power_spectrum = np.abs(np.fft.rfft(windowed_signal, n=n_fft)) ** 2
        
        # Frequências correspondentes
        freqs = np.fft.rfftfreq(n_fft, d=1/fs)
        
        # Definir as bandas de frequência (em escala mel ou log)
        max_freq = min(fs/2, 16000)  # Limitar a 16kHz ou Nyquist
        min_freq = 20  # Começar em 20Hz (audível)
        
        # Bandas em escala logarítmica
        band_edges = np.logspace(np.log10(min_freq), np.log10(max_freq), n_bands + 1)
        
        contrasts = np.zeros(n_bands)
        
        # Calcular contraste em cada banda
        for i in range(n_bands):
            # Índices de frequência dentro desta banda
            band_mask = (freqs >= band_edges[i]) & (freqs < band_edges[i+1])
            if np.sum(band_mask) == 0:
                contrasts[i] = 0
                continue
                
            band_powers = power_spectrum[band_mask]
            
            # Encontrar os picos (média dos maiores valores)
            peaks = np.mean(np.sort(band_powers)[-int(len(band_powers) * 0.15):])
            
            # Encontrar os vales (média dos menores valores)
            valleys = np.mean(np.sort(band_powers)[:int(len(band_powers) * quantile)])
            
            # Calcular contraste (em dB)
            if valleys > 0 and peaks > 0:
                contrasts[i] = 10 * np.log10(peaks / valleys)
            else:
                contrasts[i] = 0
                
        return contrasts
        
    except Exception as e:
        logger.error(f"Erro ao calcular contraste espectral: {e}")
        return np.zeros(n_bands)


def plot_spectral_power(
    spectral_power_values: np.ndarray, 
    label: str, 
    save_path: Optional[Union[str, Path]] = None, 
    show_plot: bool = True,
    x_axis_label: str = 'Harmonic Order',
    y_axis_label: str = 'Power (dB)',
    title: Optional[str] = None,
    color: str = 'blue',
    grid: bool = True,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = DEFAULT_DPI,
    additional_text: Optional[str] = None
) -> None:
    """
    Plota a potência espectral de um único sinal.

    Args:
        spectral_power_values: Array 1D com a potência espectral em dB.
        label: Rótulo para a legenda do gráfico.
        save_path: Caminho para salvar o gráfico (PNG, JPG, etc.).
        show_plot: Se True, exibe o gráfico na tela.
        x_axis_label: Rótulo para o eixo x.
        y_axis_label: Rótulo para o eixo y.
        title: Título do gráfico. Se None, usa 'Spectral Power - {label}'.
        color: Cor para a linha do gráfico.
        grid: Se True, exibe uma grade.
        figsize: Tamanho da figura (largura, altura) em polegadas.
        dpi: Resolução do gráfico em pontos por polegada.
        additional_text: Texto adicional a exibir no gráfico.
    """
    if spectral_power_values is None or len(spectral_power_values) == 0:
        logger.error(f"Dados de potência espectral vazios ou inválidos para rótulo '{label}'.")
        return

    try:
        plt.figure(figsize=figsize)
        plt.plot(spectral_power_values, label=label, color=color, linewidth=2)
        
        # Configurar título
        if title is None:
            title = f'Spectral Power - {label}'
        plt.title(title)
        
        # Configurar rótulos de eixos
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        
        # Configurar grade
        if grid:
            plt.grid(True, alpha=0.3)
            
        plt.legend()
        
        # Adicionar texto, se fornecido
        if additional_text:
            plt.annotate(additional_text, xy=(0.02, 0.97), xycoords='axes fraction',
                       fontsize=9, ha='left', va='top',
                       bbox=dict(boxstyle='round', fc='white', alpha=0.7))

        # Salvar o gráfico, se caminho fornecido
        if save_path:
            save_path = Path(save_path)
            
            # Garantir que o diretório existe
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Gráfico salvo em: {save_path}")

        # Mostrar o gráfico, se solicitado
        if show_plot:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Erro ao plotar potência espectral: {e}")
        plt.close()


def plot_multiple_spectral_powers(
    spectral_powers: List[np.ndarray], 
    labels: List[str], 
    save_path: Optional[Union[str, Path]] = None, 
    show_plot: bool = True,
    x_axis_label: str = 'Harmonic Order',
    y_axis_label: str = 'Power (dB)',
    title: str = 'Spectral Powers Comparison',
    grid: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    dpi: int = DEFAULT_DPI,
    normalize: bool = False,
    colors: Optional[List[str]] = None,
    linestyles: Optional[List[str]] = None,
    use_markers: bool = False
) -> None:
    """
    Plota múltiplas curvas de potência espectral no mesmo gráfico.

    Args:
        spectral_powers: Lista de arrays 1D representando potência espectral em dB.
        labels: Lista de rótulos correspondentes para cada curva.
        save_path: Caminho para salvar o gráfico combinado.
        show_plot: Se True, exibe o gráfico na tela.
        x_axis_label: Rótulo para o eixo x.
        y_axis_label: Rótulo para o eixo y.
        title: Título do gráfico.
        grid: Se True, exibe uma grade.
        figsize: Tamanho da figura (largura, altura) em polegadas.
        dpi: Resolução do gráfico em pontos por polegada.
        normalize: Se True, normaliza todas as curvas para facilitar comparação.
        colors: Lista opcional de cores para as curvas.
        linestyles: Lista opcional de estilos de linha para as curvas.
        use_markers: Se True, adiciona marcadores às linhas.
    """
    if not spectral_powers or not labels or len(spectral_powers) != len(labels):
        logger.error("Dados de potência espectral ou rótulos ausentes ou com comprimentos incompatíveis.")
        return

    try:
        plt.figure(figsize=figsize)
        
        # Definir cores e estilos padrão se não fornecidos
        if colors is None:
            colors = plt.cm.tab10(np.linspace(0, 1, len(spectral_powers)))
            
        if linestyles is None:
            linestyles = ['-'] * len(spectral_powers)
            
        # Garantir que temos estilos de linha suficientes
        if len(linestyles) < len(spectral_powers):
            linestyles = linestyles * (len(spectral_powers) // len(linestyles) + 1)
            
        # Marcadores para as linhas
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', '+', 'x'] if use_markers else [None] * len(spectral_powers)
        
        # Normalizar curvas, se solicitado
        if normalize:
            normalized_powers = []
            for sp in spectral_powers:
                if sp is None or len(sp) == 0:
                    normalized_powers.append(np.array([]))
                    continue
                    
                min_val = np.min(sp)
                max_val = np.max(sp)
                if max_val > min_val:
                    normalized_powers.append((sp - min_val) / (max_val - min_val))
                else:
                    normalized_powers.append(sp)
            
            plotting_data = normalized_powers
            if normalize:
                y_axis_label = 'Normalized Power'
        else:
            plotting_data = spectral_powers

        # Plotar cada curva
        for i, (sp_values, lbl) in enumerate(zip(plotting_data, labels)):
            if sp_values is None or len(sp_values) == 0:
                logger.warning(f"Dados de potência espectral para '{lbl}' vazios ou inválidos. Ignorando.")
                continue
                
            marker = markers[i % len(markers)] if use_markers else None
            marker_stride = max(1, len(sp_values) // 10) if use_markers else 1
                
            plt.plot(
                sp_values,
                label=lbl,
                color=colors[i % len(colors)],
                linestyle=linestyles[i % len(linestyles)],
                marker=marker,
                markevery=marker_stride
            )

        plt.title(title)
        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        
        if grid:
            plt.grid(True, alpha=0.3)
            
        plt.legend()

        # Salvar o gráfico, se caminho fornecido
        if save_path:
            save_path = Path(save_path)
            
            # Garantir que o diretório existe
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Gráfico de múltiplas potências espectrais salvo em: {save_path}")

        # Mostrar o gráfico, se solicitado
        if show_plot:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Erro ao plotar múltiplas potências espectrais: {e}")
        plt.close()


def plot_spectrogram(
    signal: np.ndarray,
    fs: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: Optional[int] = None,
    window_type: str = 'hann',
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    dpi: int = DEFAULT_DPI,
    freq_range: Optional[Tuple[float, float]] = None,
    time_range: Optional[Tuple[float, float]] = None,
    cmap: str = 'viridis',
    colorbar: bool = True,
    y_axis: str = 'log'
) -> np.ndarray:
    """
    Plota um espectrograma do sinal de áudio.
    
    Args:
        signal: O sinal de áudio (array NumPy 1D).
        fs: Taxa de amostragem em Hz.
        n_fft: Número de pontos FFT.
        hop_length: Tamanho do salto (default=None => n_fft//4).
        window_type: Tipo de janela a aplicar.
        save_path: Caminho para salvar o gráfico.
        show_plot: Se True, exibe o gráfico na tela.
        title: Título do gráfico.
        figsize: Tamanho da figura (largura, altura) em polegadas.
        dpi: Resolução do gráfico em pontos por polegada.
        freq_range: Intervalo de frequências a mostrar (min, max) em Hz.
        time_range: Intervalo de tempo a mostrar (min, max) em segundos.
        cmap: Mapa de cores para o espectrograma.
        colorbar: Se True, exibe uma barra de cores.
        y_axis: Tipo de escala para o eixo y ('log' ou 'linear').
        
    Returns:
        Array NumPy 2D com o espectrograma calculado.
    """
    if signal is None or len(signal) == 0:
        logger.error("Sinal vazio ou None fornecido para plot_spectrogram")
        return np.array([[]])
        
    # Definir hop_length padrão se não fornecido
    if hop_length is None:
        hop_length = n_fft // 4
        
    # Validar y_axis
    if y_axis not in ['log', 'linear']:
        logger.warning(f"Valor inválido para y_axis: {y_axis}. Usando 'log'.")
        y_axis = 'log'
        
    try:
        # Calcular STFT
        window = get_window_function(window_type, n_fft)
        f, t, stft = sp_sig.stft(signal, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length, 
                               window=window, return_onesided=True)
        
        # Converter para espectrograma (magnitude em dB)
        spectrogram = np.abs(stft)
        spectrogram_db = 20 * np.log10(np.maximum(spectrogram, 1e-10))
        
        # Criar figura
        plt.figure(figsize=figsize)
        
        # Configurar limites de frequência e tempo
        freq_min, freq_max = 0, fs/2
        if freq_range is not None:
            freq_min, freq_max = freq_range
            if freq_min < 0:
                freq_min = 0
            if freq_max > fs/2:
                freq_max = fs/2
                
        time_min, time_max = 0, len(signal) / fs
        if time_range is not None:
            time_min, time_max = time_range
            if time_min < 0:
                time_min = 0
            if time_max > len(signal) / fs:
                time_max = len(signal) / fs
        
        # Índices correspondentes aos limites
        freq_indices = np.logical_and(f >= freq_min, f <= freq_max)
        time_indices = np.logical_and(t >= time_min, t <= time_max)
        
        # Extrair região de interesse
        f_plot = f[freq_indices]
        t_plot = t[time_indices]
        spec_plot = spectrogram_db[np.ix_(freq_indices, time_indices)]
        
        # Plotar espectrograma
        if y_axis == 'log':
            # Escala logarítmica para o eixo y (frequências)
            plt.pcolormesh(t_plot, f_plot, spec_plot, cmap=cmap, shading='gouraud')
            plt.yscale('log')
        else:
            # Escala linear
            plt.pcolormesh(t_plot, f_plot, spec_plot, cmap=cmap, shading='gouraud')
        
        # Configurar rótulos e título
        plt.xlabel('Tempo (s)')
        plt.ylabel('Frequência (Hz)')
        
        if title is None:
            title = 'Espectrograma'
        plt.title(title)
        
        # Adicionar barra de cores
        if colorbar:
            cbar = plt.colorbar()
            cbar.set_label('Magnitude (dB)')
        
        # Ajustar limites dos eixos
        plt.xlim(time_min, time_max)
        plt.ylim(freq_min, freq_max)
        
        # Salvar o gráfico, se caminho fornecido
        if save_path:
            save_path = Path(save_path)
            
            # Garantir que o diretório existe
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Espectrograma salvo em: {save_path}")

        # Mostrar o gráfico, se solicitado
        if show_plot:
            plt.show()
            
        plt.close()
        
        return spectrogram
        
    except Exception as e:
        logger.error(f"Erro ao plotar espectrograma: {e}")
        plt.close()
        return np.array([[]])


def plot_spectral_features(
    signal: np.ndarray,
    fs: int,
    n_fft: int = DEFAULT_N_FFT,
    window_type: str = 'hann',
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (12, 10),
    dpi: int = DEFAULT_DPI
) -> Dict[str, float]:
    """
    Calcula e plota múltiplas características espectrais de um sinal.
    
    Args:
        signal: O sinal de áudio (array NumPy 1D).
        fs: Taxa de amostragem em Hz.
        n_fft: Número de pontos FFT.
        window_type: Tipo de janela a aplicar.
        save_path: Caminho para salvar o gráfico.
        show_plot: Se True, exibe o gráfico na tela.
        figsize: Tamanho da figura (largura, altura) em polegadas.
        dpi: Resolução do gráfico em pontos por polegada.
        
    Returns:
        Dicionário com as características espectrais calculadas.
    """
    if signal is None or len(signal) == 0:
        logger.error("Sinal vazio ou None fornecido para plot_spectral_features")
        return {}
        
    try:
        # Calcular características espectrais
        centroid = compute_spectral_centroid(signal, fs, n_fft, window_type)
        flatness = compute_spectral_flatness(signal, n_fft, window_type)
        rolloff = compute_spectral_rolloff(signal, fs, n_fft, window_type)
        contrast = compute_spectral_contrast(signal, fs, n_fft, window_type)
        
        # Calcular espectro de potência
        sp_values = spectral_power(signal, n_fft, window_type=window_type)
        
        # Criar figura com subplots
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Espectro de potência com centroide e rolloff marcados
        axs[0, 0].plot(sp_values, linewidth=2)
        axs[0, 0].set_title('Espectro de Potência')
        axs[0, 0].set_xlabel('Ordem Harmônica')
        axs[0, 0].set_ylabel('Potência (dB)')
        axs[0, 0].grid(True, alpha=0.3)
        
        # Adicionar texto com centroide e rolloff
        info_text = (f"Centroide: {centroid:.1f} Hz\n"
                    f"Rolloff: {rolloff:.1f} Hz")
        axs[0, 0].annotate(info_text, xy=(0.02, 0.95), xycoords='axes fraction',
                         fontsize=9, ha='left', va='top',
                         bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # 2. Planura espectral
        axs[0, 1].bar(['Flatness'], [flatness], color='green')
        axs[0, 1].set_title('Planura Espectral')
        axs[0, 1].set_ylabel('Valor (0-1)')
        axs[0, 1].set_ylim(0, 1)
        axs[0, 1].grid(True, alpha=0.3, axis='y')
        
        # Adicionar interpretação
        if flatness < 0.1:
            interpretation = "Timbre tonal"
        elif flatness < 0.5:
            interpretation = "Mistura tonal/ruidosa"
        else:
            interpretation = "Timbre ruidoso"
            
        axs[0, 1].annotate(f"Valor: {flatness:.3f}\n{interpretation}",
                         xy=(0.02, 0.85), xycoords='axes fraction',
                         fontsize=9, ha='left', va='top',
                         bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # 3. Espectrograma
        f, t, stft = sp_sig.stft(signal, fs=fs, nperseg=n_fft, window=window_type)
        spectrogram = 20 * np.log10(np.maximum(np.abs(stft), 1e-10))
        
        im = axs[1, 0].pcolormesh(t, f, spectrogram, cmap='viridis', shading='gouraud')
        axs[1, 0].set_title('Espectrograma')
        axs[1, 0].set_xlabel('Tempo (s)')
        axs[1, 0].set_ylabel('Frequência (Hz)')
        axs[1, 0].set_yscale('log')
        fig.colorbar(im, ax=axs[1, 0], label='Magnitude (dB)')
        
        # 4. Contraste espectral
        band_edges = np.logspace(np.log10(20), np.log10(min(fs/2, 16000)), len(contrast) + 1)
        band_centers = np.sqrt(band_edges[:-1] * band_edges[1:])
        
        axs[1, 1].bar(range(len(contrast)), contrast, color='purple')
        axs[1, 1].set_title('Contraste Espectral')
        axs[1, 1].set_xlabel('Banda de Frequência')
        axs[1, 1].set_ylabel('Contraste (dB)')
        axs[1, 1].set_xticks(range(len(contrast)))
        axs[1, 1].set_xticklabels([f"{b:.0f}" for b in band_centers])
        axs[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar o gráfico, se caminho fornecido
        if save_path:
            save_path = Path(save_path)
            
            # Garantir que o diretório existe
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Gráfico de características espectrais salvo em: {save_path}")

        # Mostrar o gráfico, se solicitado
        if show_plot:
            plt.show()
            
        plt.close()
        
        # Retornar características calculadas
        features = {
            'centroid': centroid,
            'flatness': flatness,
            'rolloff': rolloff,
            'contrast': contrast.tolist()
        }
        
        return features
        
    except Exception as e:
        logger.error(f"Erro ao plotar características espectrais: {e}")
        plt.close()
        return {}


def plot_3d_spectrogram(
    signal: np.ndarray,
    fs: int,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: Optional[int] = None,
    window_type: str = 'hann',
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    dpi: int = DEFAULT_DPI,
    freq_range: Optional[Tuple[float, float]] = None,
    elev: float = 30,
    azim: float = -45,
    cmap: str = 'viridis'
) -> None:
    """
    Cria um gráfico 3D do espectrograma.
    
    Args:
        signal: O sinal de áudio (array NumPy 1D).
        fs: Taxa de amostragem em Hz.
        n_fft: Número de pontos FFT.
        hop_length: Tamanho do salto (default=None => n_fft//4).
        window_type: Tipo de janela a aplicar.
        save_path: Caminho para salvar o gráfico.
        show_plot: Se True, exibe o gráfico na tela.
        title: Título do gráfico.
        figsize: Tamanho da figura (largura, altura) em polegadas.
        dpi: Resolução do gráfico em pontos por polegada.
        freq_range: Intervalo de frequências a mostrar (min, max) em Hz.
        elev: Elevação para a visualização 3D.
        azim: Azimute para a visualização 3D.
        cmap: Mapa de cores para o espectrograma.
    """
    if signal is None or len(signal) == 0:
        logger.error("Sinal vazio ou None fornecido para plot_3d_spectrogram")
        return
        
    # Definir hop_length padrão se não fornecido
    if hop_length is None:
        hop_length = n_fft // 4
        
    try:
        # Calcular STFT
        window = get_window_function(window_type, n_fft)
        f, t, stft = sp_sig.stft(signal, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length, 
                               window=window, return_onesided=True)
        
        # Converter para espectrograma (magnitude em dB)
        spectrogram = np.abs(stft)
        spectrogram_db = 10 * np.log10(np.maximum(spectrogram, 1e-10))
        
        # Aplicar limites de frequência, se fornecidos
        if freq_range is not None:
            freq_min, freq_max = freq_range
            if freq_min < 0:
                freq_min = 0
            if freq_max > fs/2:
                freq_max = fs/2
                
            # Encontrar índices correspondentes
            freq_indices = np.logical_and(f >= freq_min, f <= freq_max)
            
            # Filtrar dados
            f = f[freq_indices]
            spectrogram_db = spectrogram_db[freq_indices, :]
        
        # Criar figura 3D
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Criar grade de tempo e frequência
        T, F = np.meshgrid(t, f)
        
        # Plotar superfície 3D
        surf = ax.plot_surface(T, F, spectrogram_db, cmap=cmap, antialiased=True)
        
        # Configurar ângulo de visão
        ax.view_init(elev=elev, azim=azim)
        
        # Configurar rótulos e título
        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Frequência (Hz)')
        ax.set_zlabel('Magnitude (dB)')
        
        if title is None:
            title = 'Espectrograma 3D'
        ax.set_title(title)
        
        # Adicionar barra de cores
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        # Configurar escala logarítmica para o eixo de frequência
        ax.set_yscale('log')
        
        # Salvar o gráfico, se caminho fornecido
        if save_path:
            save_path = Path(save_path)
            
            # Garantir que o diretório existe
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Espectrograma 3D salvo em: {save_path}")

        # Mostrar o gráfico, se solicitado
        if show_plot:
            plt.show()
            
        plt.close()
        
    except Exception as e:
        logger.error(f"Erro ao plotar espectrograma 3D: {e}")
        plt.close()


def compare_spectral_profiles(
    signals: List[np.ndarray],
    fs_list: List[int],
    labels: List[str],
    n_fft: int = DEFAULT_N_FFT,
    window_type: str = 'hann',
    save_path: Optional[Union[str, Path]] = None,
    show_plot: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    dpi: int = DEFAULT_DPI,
    freq_scale: str = 'log'
) -> Dict[str, Dict[str, float]]:
    """
    Compara perfis espectrais de múltiplos sinais.
    
    Args:
        signals: Lista de sinais de áudio.
        fs_list: Lista de taxas de amostragem correspondentes.
        labels: Lista de rótulos para cada sinal.
        n_fft: Número de pontos FFT.
        window_type: Tipo de janela a aplicar.
        save_path: Caminho para salvar o gráfico.
        show_plot: Se True, exibe o gráfico na tela.
        figsize: Tamanho da figura (largura, altura) em polegadas.
        dpi: Resolução do gráfico em pontos por polegada.
        freq_scale: Escala para o eixo de frequência ('log' ou 'linear').
        
    Returns:
        Dicionário com características de cada sinal.
    """
    if not signals or not fs_list or not labels:
        logger.error("Dados inválidos para compare_spectral_profiles")
        return {}
        
    if len(signals) != len(fs_list) or len(signals) != len(labels):
        logger.error("Número de sinais, taxas de amostragem e rótulos deve ser igual")
        return {}
        
    # Validar freq_scale
    if freq_scale not in ['log', 'linear']:
        logger.warning(f"Valor inválido para freq_scale: {freq_scale}. Usando 'log'.")
        freq_scale = 'log'
        
    try:
        # Calcular características para cada sinal
        features = {}
        
        for i, (signal, fs, label) in enumerate(zip(signals, fs_list, labels)):
            if signal is None or len(signal) == 0:
                logger.warning(f"Sinal vazio para rótulo: {label}")
                continue
                
            # Calcular características
            centroid = compute_spectral_centroid(signal, fs, n_fft, window_type)
            flatness = compute_spectral_flatness(signal, n_fft, window_type)
            rolloff = compute_spectral_rolloff(signal, fs, n_fft, window_type)
            
            features[label] = {
                'centroid': centroid,
                'flatness': flatness,
                'rolloff': rolloff
            }
            
        # Criar figura comparativa
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Magnitudes espectrais
        for i, (signal, fs, label) in enumerate(zip(signals, fs_list, labels)):
            if signal is None or len(signal) == 0:
                continue
                
            # Calcular FFT
            window = get_window_function(window_type, min(len(signal), n_fft))
            if len(signal) > len(window):
                windowed_signal = signal[:len(window)] * window
            else:
                windowed_signal = signal * window[:len(signal)]
                
            fft_result = np.fft.rfft(windowed_signal, n=n_fft)
            magnitude = np.abs(fft_result)
            
            # Converter para dB
            magnitude_db = 20 * np.log10(np.maximum(magnitude, 1e-10))
            
            # Frequências correspondentes
            freqs = np.fft.rfftfreq(n_fft, d=1/fs)
            
            # Limitar a 20Hz-20kHz (aproximadamente audível)
            valid_indices = np.logical_and(freqs >= 20, freqs <= 20000)
            plot_freqs = freqs[valid_indices]
            plot_magnitude = magnitude_db[valid_indices]
            
            # Escolher uma cor diferente para cada sinal
            color = plt.cm.tab10(i % 10)
            
            # Plotar espectro
            axs[0, 0].plot(plot_freqs, plot_magnitude, label=label, color=color, alpha=0.7)
            
            # Marcar centroide e rolloff
            centroid = features[label]['centroid']
            rolloff = features[label]['rolloff']
            
            # Encontrar os valores mais próximos no eixo de frequência
            idx_centroid = np.argmin(np.abs(plot_freqs - centroid))
            idx_rolloff = np.argmin(np.abs(plot_freqs - rolloff))
            
            # Adicionar marcadores
            axs[0, 0].scatter(centroid, plot_magnitude[idx_centroid], 
                            marker='o', color=color, s=50, alpha=0.8)
            axs[0, 0].scatter(rolloff, plot_magnitude[idx_rolloff], 
                            marker='x', color=color, s=50, alpha=0.8)
            
        # Configurar gráfico de espectro
        axs[0, 0].set_title('Magnitude Espectral')
        axs[0, 0].set_xlabel('Frequência (Hz)')
        axs[0, 0].set_ylabel('Magnitude (dB)')
        if freq_scale == 'log':
            axs[0, 0].set_xscale('log')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend()
        
        # 2. Comparação de centroides
        centroid_values = [features[label]['centroid'] for label in labels if label in features]
        axs[0, 1].bar(range(len(centroid_values)), centroid_values, color=plt.cm.tab10.colors[:len(centroid_values)])
        axs[0, 1].set_title('Centroide Espectral')
        axs[0, 1].set_ylabel('Frequência (Hz)')
        axs[0, 1].set_xticks(range(len(centroid_values)))
        axs[0, 1].set_xticklabels([label for label in labels if label in features], rotation=45, ha='right')
        axs[0, 1].grid(True, alpha=0.3, axis='y')
        
        # 3. Comparação de planura espectral
        flatness_values = [features[label]['flatness'] for label in labels if label in features]
        axs[1, 0].bar(range(len(flatness_values)), flatness_values, color=plt.cm.tab10.colors[:len(flatness_values)])
        axs[1, 0].set_title('Planura Espectral')
        axs[1, 0].set_ylabel('Valor (0-1)')
        axs[1, 0].set_ylim(0, 1)
        axs[1, 0].set_xticks(range(len(flatness_values)))
        axs[1, 0].set_xticklabels([label for label in labels if label in features], rotation=45, ha='right')
        axs[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Comparação de rolloff
        rolloff_values = [features[label]['rolloff'] for label in labels if label in features]
        axs[1, 1].bar(range(len(rolloff_values)), rolloff_values, color=plt.cm.tab10.colors[:len(rolloff_values)])
        axs[1, 1].set_title('Frequência de Rolloff (85%)')
        axs[1, 1].set_ylabel('Frequência (Hz)')
        axs[1, 1].set_xticks(range(len(rolloff_values)))
        axs[1, 1].set_xticklabels([label for label in labels if label in features], rotation=45, ha='right')
        axs[1, 1].grid(True, alpha=0.3, axis='y')
        
        # Ajustar layout
        plt.tight_layout()
        
        # Salvar o gráfico, se caminho fornecido
        if save_path:
            save_path = Path(save_path)
            
            # Garantir que o diretório existe
            save_path.parent.mkdir(exist_ok=True, parents=True)
            
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            logger.info(f"Comparação de perfis espectrais salva em: {save_path}")

        # Mostrar o gráfico, se solicitado
        if show_plot:
            plt.show()
            
        plt.close()
        
        return features
        
    except Exception as e:
        logger.error(f"Erro ao comparar perfis espectrais: {e}")
        plt.close()
        return {}


def generate_spectral_summary(
    signal: np.ndarray,
    fs: int,
    output_dir: Union[str, Path],
    file_name: str = 'spectral_summary',
    n_fft: int = DEFAULT_N_FFT,
    window_type: str = 'hann',
    dpi: int = DEFAULT_DPI,
    alpha: float = 0.5,
    beta: float = 0.5
) -> Dict[str, Any]:
    """
    Gera um conjunto completo de análises espectrais para um sinal.

    Args:
        signal: O sinal de áudio (array NumPy 1D).
        fs: Taxa de amostragem em Hz.
        output_dir: Diretório para salvar os resultados.
        file_name: Prefixo para nomes de arquivos.
        n_fft: Número de pontos FFT.
        window_type: Tipo de janela a aplicar.
        dpi: Resolução dos gráficos em pontos por polegada.
        alpha: Peso da componente harmónica (0.0 a 1.0).
        beta: Peso da componente inarmónica (0.0 a 1.0).

    Returns:
        Dicionário com métricas e caminhos para os gráficos gerados.
    """
    if signal is None or len(signal) == 0:
        logger.error("Sinal vazio ou None fornecido para generate_spectral_summary")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    results = {}

    try:
        # 1. Espectrograma 2D
        spectrogram_path = output_dir / f"{file_name}_spectrogram.png"
        plot_spectrogram(signal, fs, n_fft, window_type=window_type,
                         save_path=spectrogram_path, show_plot=False, dpi=dpi)
        results['spectrogram_path'] = str(spectrogram_path)

        # 2. Espectrograma 3D
        spectrogram3d_path = output_dir / f"{file_name}_spectrogram3d.png"
        plot_3d_spectrogram(signal, fs, n_fft, window_type=window_type,
                            save_path=spectrogram3d_path, show_plot=False, dpi=dpi)
        results['spectrogram3d_path'] = str(spectrogram3d_path)

        # 3. Potência espectral (FFT)
        sp_values = spectral_power(signal, n_fft, window_type=window_type)
        sp_path = output_dir / f"{file_name}_power.png"
        plot_spectral_power(sp_values, label='Potência Espectral',
                            save_path=sp_path, show_plot=False, dpi=dpi)
        results['power_path'] = str(sp_path)

        # 4. Potência espectral inarmónica (LFT)
        ip_values = spectral_power_lft(signal, n_fft, window_type=window_type)
        ip_path = output_dir / f"{file_name}_inharmonic_power.png"
        plot_spectral_power(ip_values, label='Potência Inarmónica',
                            save_path=ip_path, show_plot=False, dpi=dpi)
        results['inharmonic_power_path'] = str(ip_path)

        # 5. Potência espectral ponderada
        weighted_values = weighted_spectral_metric(sp_values, ip_values, alpha=alpha, beta=beta)
        wp_path = output_dir / f"{file_name}_weighted_power.png"
        plot_spectral_power(weighted_values, label='Potência Ponderada',
                            save_path=wp_path, show_plot=False, dpi=dpi)
        results['weighted_power_path'] = str(wp_path)

        # 6. Características espectrais
        features_path = output_dir / f"{file_name}_features.png"
        features = plot_spectral_features(signal, fs, n_fft, window_type=window_type,
                                          save_path=features_path, show_plot=False, dpi=dpi)
        results['features_path'] = str(features_path)
        results['features'] = features

        # 7. Métricas adicionais
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        crest_factor = peak / rms if rms > 0 else 0
        zero_crossings = np.sum(np.abs(np.diff(np.signbit(signal))))
        zcr = zero_crossings / len(signal)
        energy = np.sum(signal**2)
        duration = len(signal) / fs

        results.update({
            'rms': float(rms),
            'crest_factor': float(crest_factor),
            'zero_crossing_rate': float(zcr),
            'energy': float(energy),
            'duration': float(duration),
            'sample_rate': int(fs),
            'harmonic_weight': float(alpha),
            'inharmonic_weight': float(beta)
        })

        logger.info(f"Resumo espectral gerado em: {output_dir}")
        return results

    except Exception as e:
        logger.error(f"Erro ao gerar resumo espectral: {e}")
        return {}



def batch_process_audio_files(
    file_paths: List[Union[str, Path]],
    output_dir: Union[str, Path],
    n_fft: int = DEFAULT_N_FFT,
    window_type: str = 'hann',
    generate_summaries: bool = True,
    compare_files: bool = True,
    dpi: int = DEFAULT_DPI,
    alpha: float = 0.5,
    beta: float = 0.5
) -> Dict[str, Any]:
    """
    Processa em lote múltiplos arquivos de áudio com análise espectral.

    Args:
        file_paths: Lista de caminhos para arquivos de áudio.
        output_dir: Diretório para salvar os resultados.
        n_fft: Número de pontos FFT.
        window_type: Tipo de janela a aplicar.
        generate_summaries: Se True, gera resumos espectrais individuais.
        compare_files: Se True, gera comparações entre arquivos.
        dpi: Resolução dos gráficos em pontos por polegada.
        alpha: Peso da componente harmónica (0.0 a 1.0).
        beta: Peso da componente inarmónica (0.0 a 1.0).

    Returns:
        Dicionário com resultados e caminhos para os gráficos gerados.
    """
    if not file_paths:
        logger.error("Nenhum arquivo fornecido para processamento em lote")
        return {}

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    results = {
        'files': {},
        'comparisons': {}
    }

    signals = []
    fs_list = []
    labels = []
    valid_files = []

    for file_path in file_paths:
        file_path = Path(file_path)
        try:
            signal, fs = load_sound(file_path)
            if signal is not None and fs is not None:
                signals.append(signal)
                fs_list.append(fs)
                labels.append(file_path.stem)
                valid_files.append(file_path)
                logger.info(f"Arquivo carregado: {file_path}")
            else:
                logger.warning(f"Não foi possível carregar: {file_path}")
        except Exception as e:
            logger.error(f"Erro ao carregar {file_path}: {e}")

    if not signals:
        logger.error("Nenhum arquivo válido para processar")
        return results

    try:
        # Processar cada arquivo individualmente
        if generate_summaries:
            for signal, fs, label, file_path in zip(signals, fs_list, labels, valid_files):
                file_dir = output_dir / label
                file_dir.mkdir(exist_ok=True, parents=True)

                summary = generate_spectral_summary(
                    signal, fs, file_dir,
                    file_name=label,
                    n_fft=n_fft,
                    window_type=window_type,
                    dpi=dpi,
                    alpha=alpha,
                    beta=beta
                )

                results['files'][label] = {
                    'path': str(file_path),
                    'sample_rate': fs,
                    'summary': summary
                }

        # Comparar arquivos
        if compare_files and len(signals) >= 2:
            comp_dir = output_dir / 'comparisons'
            comp_dir.mkdir(exist_ok=True, parents=True)

            # Comparar potência espectral FFT
            sp_values_list = [spectral_power(sig, n_fft, window_type=window_type) for sig in signals]
            sp_comp_path = comp_dir / 'spectral_power_comparison.png'
            plot_multiple_spectral_powers(
                sp_values_list, labels,
                save_path=sp_comp_path,
                show_plot=False,
                normalize=True,
                dpi=dpi,
                use_markers=True
            )
            results['comparisons']['power_comparison'] = str(sp_comp_path)

            # Comparar potência espectral ponderada (FFT + LFT)
            wp_values_list = []
            for sig in signals:
                sp = spectral_power(sig, n_fft, window_type=window_type)
                ip = spectral_power_lft(sig, n_fft, window_type=window_type)
                wp = weighted_spectral_metric(sp, ip, alpha=alpha, beta=beta)
                wp_values_list.append(wp)

            wp_comp_path = comp_dir / 'weighted_power_comparison.png'
            plot_multiple_spectral_powers(
                wp_values_list, labels,
                save_path=wp_comp_path,
                show_plot=False,
                normalize=True,
                dpi=dpi,
                use_markers=True
            )
            results['comparisons']['weighted_power_comparison'] = str(wp_comp_path)

            # Comparar perfis espectrais
            profile_comp_path = comp_dir / 'spectral_profile_comparison.png'
            features = compare_spectral_profiles(
                signals, fs_list, labels,
                n_fft=n_fft,
                window_type=window_type,
                save_path=profile_comp_path,
                show_plot=False,
                dpi=dpi
            )
            results['comparisons']['profile_comparison'] = str(profile_comp_path)
            results['comparisons']['features'] = features

        logger.info(f"Processamento em lote concluído em: {output_dir}")
        return results

    except Exception as e:
        logger.error(f"Erro no processamento em lote: {e}")
        return results



def weighted_spectral_metric(
    harmonic_power: np.ndarray,
    inharmonic_power: np.ndarray,
    alpha: float = 0.5,
    beta: float = 0.5
) -> np.ndarray:
    """
    Combina potência harmónica e inarmónica com pesos definidos.

    Args:
        harmonic_power: Array com potência espectral harmónica.
        inharmonic_power: Array com potência espectral inarmónica.
        alpha: Peso da componente harmónica (0.0 a 1.0).
        beta: Peso da componente inarmónica (0.0 a 1.0).

    Returns:
        Array com métrica ponderada.
    """
    if harmonic_power.shape != inharmonic_power.shape:
        logger.warning("Dimensões incompatíveis entre harmónico e inarmónico. Ajustando...")
        min_len = min(len(harmonic_power), len(inharmonic_power))
        harmonic_power = harmonic_power[:min_len]
        inharmonic_power = inharmonic_power[:min_len]

    combined = alpha * harmonic_power + beta * inharmonic_power
    return combined

