# dissonance_models.py - Módulo melhorado

"""
Implementação de múltiplos modelos de dissonância para análise de áudio:
- Modelo de Sethares (baseado nas curvas de Plomp-Levelt)
- Modelo de Hutchinson-Knopoff (extensão de Plomp-Levelt com largura de banda crítica)
- Modelo de Vassilakis (modelo de flutuação espectral)
- Modelo de Aures-Zwicker (baseado em rugosidade sensorial)
- Modelo de Stolzenburg (baseado em harmonicidade)

Cada modelo fornece métodos para calcular dissonância entre tons puros,
sons complexos, e para gerar curvas de dissonância e escalas ótimas.

Melhorias:
- Documentação expandida com referências bibliográficas
- Validação de parâmetros mais robusta
- Tratamento de erros aprimorado
- Implementação de cache para cálculos repetitivos
- Métodos adicionais para análise comparativa
- Nomenclatura consistente
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Any
from numpy.fft import rfft, irfft
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
import math
import logging
from functools import lru_cache
import os
import librosa
import scipy.signal
from scipy.fft import rfft, irfft





logger = logging.getLogger(__name__)


# Configuração de logging
from log_config import configure_root_logger
configure_root_logger()            # <-- NUNCA duplifica
logger = logging.getLogger(__name__)



# Constantes globais
DEFAULT_PLOT_DPI = 300
CENTS_PER_OCTAVE = 1200



class DissonanceModel(ABC):
    """
    Classe base abstrata para modelos de dissonância.
    
    Fornece uma interface comum para todos os modelos de dissonância e
    implementa funcionalidade compartilhada.
    """
    
    def __init__(self, name: str, description: str = ""):
        """
        Inicializa o modelo de dissonância com um nome e descrição.
        
        Args:
            name: Nome do modelo.
            description: Descrição breve do modelo.
        """
        self.name = name
        self.description = description
        logger.debug(f"Modelo de dissonância inicializado: {name}")
    
    @abstractmethod
    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        """
        Calcula a dissonância entre dois tons puros.
        
        Args:
            f1: Frequência do primeiro tom puro (Hz)
            f2: Frequência do segundo tom puro (Hz)
            a1: Amplitude do primeiro tom
            a2: Amplitude do segundo tom
            
        Returns:
            Valor de dissonância entre os dois tons
        """
        pass
    
    def total_dissonance(self, partials1: List[Tuple[float, float]], 
                        partials2: List[Tuple[float, float]]) -> float:
        """
        Calcula a dissonância total entre dois sons complexos.
        Cada som é representado como uma lista de tuplas (frequência, amplitude).
        
        Args:
            partials1: Lista de tuplas (frequência, amplitude) para o primeiro som
            partials2: Lista de tuplas (frequência, amplitude) para o segundo som
            
        Returns:
            Valor total de dissonância entre os dois sons complexos
            
        Raises:
            ValueError: Se as listas de parciais estiverem vazias ou inválidas.
        """
        # Validação de entrada
        if not partials1 or not partials2:
            logger.warning("Listas de parciais vazias fornecidas para total_dissonance")
            return 0.0
        
        try:
            total_diss = 0.0
            
            # Para todos os pares de parciais entre os dois sons
            for f1, a1 in partials1:
                for f2, a2 in partials2:
                    # Adicionar dissonância de cada par ao total
                    pair_diss = self.pure_tones_dissonance(f1, f2, a1, a2)
                    total_diss += pair_diss
                    
            return total_diss
            
        except Exception as e:
            logger.error(f"Erro ao calcular dissonância total: {e}")
            raise
    
    def same_timbre_dissonance(self, base_partials: List[Tuple[float, float]], 
                              interval: float) -> float:
        """
        Calcula a dissonância quando o mesmo timbre é deslocado por um intervalo.
        
        Args:
            base_partials: Lista de tuplas (frequência, amplitude) para o timbre base
            interval: Razão de frequência do intervalo (ex.: 2.0 para oitava, 1.5 para quinta)
            
        Returns:
            Dissonância do intervalo
            
        Raises:
            ValueError: Se a lista de parciais estiver vazia ou o intervalo for inválido.
        """
        # Validação de entrada
        if not base_partials:
            logger.warning("Lista de parciais vazia fornecida para same_timbre_dissonance")
            return 0.0
            
        if interval <= 0:
            logger.error(f"Intervalo inválido: {interval}. Deve ser positivo.")
            raise ValueError(f"Intervalo deve ser positivo, obtido: {interval}")
        
        try:
            # Criar timbre deslocado pelo intervalo
            shifted_partials = [(f * interval, a) for f, a in base_partials]
            
            # Calcular dissonância total
            return self.total_dissonance(base_partials, shifted_partials)
            
        except Exception as e:
            logger.error(f"Erro ao calcular dissonância do mesmo timbre: {e}")
            raise
    
    def calculate_dissonance_curve(self, partials: List[Tuple[float, float]], 
                                  min_interval: float = 1.0,
                                  max_interval: float = 2.0,
                                  num_points: int = 100) -> Dict[float, float]:
        """
        Calcula a curva de dissonância para um timbre em um intervalo.
        
        Args:
            partials: Lista de tuplas (frequência, amplitude) para o timbre
            min_interval: Menor intervalo a analisar (proporção)
            max_interval: Maior intervalo a analisar (proporção)
            num_points: Número de pontos na curva
            
        Returns:
            Dicionário com intervalos como chaves e valores de dissonância como valores
            
        Raises:
            ValueError: Se os parâmetros forem inválidos.
        """
        # Validação de entrada
        if not partials:
            logger.warning("Lista de parciais vazia fornecida para calculate_dissonance_curve")
            return {}
            
        if min_interval <= 0 or max_interval <= min_interval:
            logger.error(f"Intervalos inválidos: min={min_interval}, max={max_interval}")
            raise ValueError("Intervalo mínimo deve ser > 0 e máximo > mínimo")
            
        if num_points < 2:
            logger.error(f"Número de pontos inválido: {num_points}")
            raise ValueError(f"Número de pontos deve ser >= 2, obtido: {num_points}")
        
        try:
            # Calcular intervalos linearmente espaçados
            intervals = np.linspace(min_interval, max_interval, num_points)
            curve = {}
            
            # Calcular dissonância para cada intervalo
            for interval in intervals:
                curve[interval] = self.same_timbre_dissonance(partials, interval)
                
            return curve
            
        except Exception as e:
            logger.error(f"Erro ao calcular curva de dissonância: {e}")
            raise
    
    def find_local_minima(self, curve: Dict[float, float], 
                         sensitivity: float = 0.01) -> List[float]:
        """
        Encontra mínimos locais na curva de dissonância.
        Estes representam os intervalos mais consonantes para o timbre.
        
        Args:
            curve: Dicionário com intervalos como chaves e valores de dissonância como valores
            sensitivity: Determina quão pronunciado um mínimo deve ser para ser detectado
            
        Returns:
            Lista de intervalos representando mínimos locais de dissonância
            
        Raises:
            ValueError: Se a curva estiver vazia ou inválida.
        """
        # Validação de entrada
        if not curve:
            logger.warning("Curva vazia fornecida para find_local_minima")
            return []
            
        if sensitivity <= 0:
            logger.warning(f"Sensibilidade inválida: {sensitivity}, usando valor padrão 0.01")
            sensitivity = 0.01
        
        try:
            # Ordenar intervalos
            intervals = sorted(list(curve.keys()))
            minima = []
            
            # Ignorar extremos da curva
            for i in range(1, len(intervals) - 1):
                interval = intervals[i]
                value = curve[interval]
                
                # Verificar se é um mínimo local
                if (value < curve[intervals[i-1]] and 
                    value < curve[intervals[i+1]] and
                    value < curve[intervals[i-1]] - sensitivity):
                    minima.append(interval)
                    logger.debug(f"Mínimo local encontrado em intervalo {interval} (valor: {value})")
            
            return minima
            
        except Exception as e:
            logger.error(f"Erro ao encontrar mínimos locais: {e}")
            raise
    
    def generate_scale(self, partials: List[Tuple[float, float]], 
                     min_interval: float = 1.0, max_interval: float = 2.0,
                     num_points: int = 100,
                     include_endpoints: bool = True) -> List[float]:
        """
        Gera uma escala baseada nos mínimos da curva de dissonância.
        
        Args:
            partials: Lista de tuplas (frequência, amplitude) para o timbre base
            min_interval: Intervalo mínimo para a escala (ex., 1.0)
            max_interval: Intervalo máximo para a escala (ex., 2.0 para uma oitava)
            num_points: Número de pontos a calcular na curva
            include_endpoints: Se True, inclui sempre os intervalos mínimo e máximo na escala
            
        Returns:
            Lista de proporções de frequência que formam a escala
            
        Raises:
            ValueError: Se os parâmetros ou parciais forem inválidos.
        """
        # Validação de entrada
        if not partials:
            logger.warning("Lista de parciais vazia fornecida para generate_scale")
            return []
        
        try:
            # Calcular curva de dissonância
            curve = self.calculate_dissonance_curve(partials, min_interval, max_interval, num_points)
            
            # Encontrar mínimos locais
            minima = self.find_local_minima(curve)
            
            # Adicionar intervalo final para completar a escala, se solicitado
            if include_endpoints:
                if max_interval not in minima:
                    minima.append(max_interval)
                
                # Garantir que o intervalo inicial esteja incluído
                if min_interval not in minima:
                    minima.insert(0, min_interval)
            
            # Ordenar e remover duplicatas
            minima = sorted(set(minima))
            
            # Registrar a escala gerada
            cents = [1200 * np.log2(interval) for interval in minima]
            logger.info(f"Escala gerada com {len(minima)} notas: {', '.join([f'{c:.1f}¢' for c in cents])}")
            
            return minima
            
        except Exception as e:
            logger.error(f"Erro ao gerar escala: {e}")
            raise
    
    def visualize_dissonance_curve(self, curve: Dict[float, float], 
                                 scale: Optional[List[float]] = None,
                                 title: Optional[str] = None,
                                 save_file: Optional[str] = None,
                                 show_cents: bool = True,
                                 highlight_minima: bool = True,
                                 dpi: int = DEFAULT_PLOT_DPI):
        """
        Visualiza a curva de dissonância e opcionalmente os pontos da escala.
        
        Args:
            curve: Dicionário com intervalos como chaves e valores de dissonância como valores
            scale: Lista opcional de proporções que formam uma escala
            title: Título do gráfico
            save_file: Se fornecido, salva o gráfico neste arquivo
            show_cents: Se deve mostrar um eixo secundário com centésimos
            highlight_minima: Se deve destacar automaticamente mínimos locais
            dpi: Resolução do gráfico salvo (pontos por polegada)
            
        Raises:
            ValueError: Se a curva estiver vazia ou inválida.
        """
        # Validação de entrada
        if not curve:
            logger.warning("Curva vazia fornecida para visualize_dissonance_curve")
            return
        
        try:
            # Preparar dados
            intervals = sorted(list(curve.keys()))
            dissonance_values = [curve[i] for i in intervals]
            
            # Criar gráfico
            plt.figure(figsize=(12, 6))
            plt.plot(intervals, dissonance_values, 'b-', linewidth=2)
            
            # Encontrar e destacar mínimos locais automaticamente
            if highlight_minima and not scale:
                minima = self.find_local_minima(curve)
                if minima:
                    minima_y = [curve[m] for m in minima]
                    plt.plot(minima, minima_y, 'go', markersize=6)
            
            # Adicionar pontos da escala se fornecidos
            if scale:
                scale_y = [curve.get(ratio, 0) for ratio in scale]
                plt.plot(scale, scale_y, 'ro', markersize=8)
                
                # Rotular pontos da escala
                for i, ratio in enumerate(scale):
                    if i < len(scale_y):  # Segurança extra
                        cent_value = CENTS_PER_OCTAVE * np.log2(ratio)
                        plt.annotate(f"{ratio:.3f} ({cent_value:.0f}¢)", 
                                    (ratio, scale_y[i]),
                                    xytext=(0, 10), 
                                    textcoords='offset points',
                                    ha='center')
            
            # Configurar gráfico
            if title is None:
                title = f"{self.name} Dissonance Curve"
                
            plt.title(title)
            plt.xlabel('Frequency Ratio')
            plt.ylabel('Dissonance')
            plt.grid(True, alpha=0.3)
            
            # Adicionar eixo x secundário com centésimos
            if show_cents:
                ax1 = plt.gca()
                ax2 = ax1.twiny()
                ax2.set_xlim(ax1.get_xlim())
                
                # Configurar marcações de centésimos
                cent_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
                ratio_ticks = [2**(c/CENTS_PER_OCTAVE) for c in cent_ticks]
                ax2.set_xticks(ratio_ticks)
                ax2.set_xticklabels([f"{c}¢" for c in cent_ticks])
                ax2.set_xlabel('Cents')
            
            # Salvar ou mostrar
            if save_file:
                plt.savefig(save_file, dpi=dpi, bbox_inches='tight')
                logger.info(f"Curva de dissonância salva em: {save_file}")
                plt.close()
            else:
                plt.show()
                plt.close()
                
        except Exception as e:
            logger.error(f"Erro ao visualizar curva de dissonância: {e}")
            plt.close()
            raise
    
    # Linha ~300: Melhorar método calculate_dissonance_metric da classe DissonanceModel
    def calculate_dissonance_metric(self, df: pd.DataFrame) -> float:
        """
        Calcula uma métrica de dissonância para um conjunto de parciais em um DataFrame.
        Versão aprimorada com tratamento de erros e otimização de memória.
    
        Args:
            df: DataFrame com colunas 'Frequency (Hz)' e 'Amplitude'
        
        Returns:
            Valor total da métrica de dissonância
        
        Raises:
            ValueError: Se o DataFrame não contiver as colunas necessárias.
        """
        # Validação robusta de entrada
        if df is None or df.empty:
            logger.warning("DataFrame vazio ou None fornecido para calculate_dissonance_metric")
            return 0.0
        
        # Verificar colunas necessárias
        if 'Frequency (Hz)' not in df.columns:
            msg = "DataFrame deve conter a coluna 'Frequency (Hz)'"
            logger.error(msg)
            raise ValueError(msg)
    
        try:
            # Verificar e preparar a coluna de Amplitude
            if 'Amplitude' not in df.columns:
                if 'Magnitude (dB)' in df.columns:
                    # Criar uma cópia para evitar modificar o original
                    df = df.copy()
                    # Converter dB para amplitude linear com tratamento de erros para valores extremos
                    with np.errstate(over='ignore', under='ignore'):  # Suprimir avisos de overflow/underflow
                        df['Amplitude'] = 10 ** (df['Magnitude (dB)'] / 20)
                    # Verificar e limitar valores extremos
                    df['Amplitude'] = np.clip(df['Amplitude'], 1e-10, 1e10)
                    logger.debug("Amplitude calculada a partir de 'Magnitude (dB)'")
                else:
                    msg = "DataFrame deve conter coluna 'Amplitude' ou 'Magnitude (dB)'"
                    logger.error(msg)
                    raise ValueError(msg)
        
            # Verificar valores inválidos
            if df['Frequency (Hz)'].isnull().any() or df['Amplitude'].isnull().any():
                logger.warning("Encontrados valores nulos nas colunas. Removendo linhas inválidas.")
                df = df.dropna(subset=['Frequency (Hz)', 'Amplitude'])
            
            if df.empty:
                logger.warning("DataFrame vazio após remoção de valores nulos")
                return 0.0
            
            # Verificar frequências não-positivas
            if (df['Frequency (Hz)'] <= 0).any():
                logger.warning("Encontradas frequências zero ou negativas. Removendo.")
                df = df[df['Frequency (Hz)'] > 0]
            
            if df.empty:
                logger.warning("DataFrame vazio após remoção de frequências inválidas")
                return 0.0
            
            # Verificar amplitudes não-positivas
            if (df['Amplitude'] <= 0).any():
                logger.warning("Encontradas amplitudes zero ou negativas. Substituindo por valor mínimo.")
                min_valid_amp = df[df['Amplitude'] > 0]['Amplitude'].min()
                if np.isnan(min_valid_amp):
                    min_valid_amp = 1e-10
                df.loc[df['Amplitude'] <= 0, 'Amplitude'] = min_valid_amp
        
            # Extrair parciais como tuplas (frequência, amplitude)
            partials = list(zip(df['Frequency (Hz)'].values, df['Amplitude'].values))
        
            # Usar processamento em lotes para grandes conjuntos de dados
            total_diss = 0.0
            n = len(partials)
        
            if n > 100:  # Para conjuntos grandes, processamento em lotes
                # Tamanho do lote - ajustar conforme necessário
                batch_size = 50
            
                for i in range(0, n, batch_size):
                    batch_end = min(i + batch_size, n)
                
                    # Processar este lote
                    for j in range(i, batch_end):
                        for k in range(j+1, n):  # Comparar com todos os outros parciais
                            f1, a1 = partials[j]
                            f2, a2 = partials[k]
                            try:
                                pair_diss = self.pure_tones_dissonance(f1, f2, a1, a2)
                                total_diss += pair_diss
                            except (OverflowError, ZeroDivisionError, FloatingPointError) as calc_error:
                                logger.warning(f"Erro numérico no cálculo de dissonância para f1={f1}, f2={f2}: {calc_error}")
                                continue
                            except Exception as e:
                                logger.error(f"Erro no cálculo de dissonância: {e}")
                                continue
            else:
                # Cálculo normal para conjuntos pequenos
                for i in range(n):
                    for j in range(i+1, n):  # Apenas calcular para pares únicos
                        f1, a1 = partials[i]
                        f2, a2 = partials[j]
                        try:
                            pair_diss = self.pure_tones_dissonance(f1, f2, a1, a2)
                            total_diss += pair_diss
                        except Exception as e:
                            logger.warning(f"Erro no cálculo de dissonância para par: {e}")
                            continue
        
            # Escalar o resultado para facilitar comparação com outras métricas
            # Escala adaptativa baseada no número de parciais
            scale = 10.0 / (n * (n-1) / 2) if n > 1 else 10.0  # Normalizar pelo número de pares
            scaled_diss = total_diss * scale
        
            logger.debug(f"Métrica de dissonância calculada: {scaled_diss}")
            return scaled_diss
        
        except MemoryError as me:
            logger.error(f"Erro de memória ao calcular métrica de dissonância: {me}")
            # Tentar uma abordagem mais conservadora com menos parciais
            try:
                # Reduzir o número de parciais para os 50 mais significativos
                df_reduced = df.nlargest(50, 'Amplitude')
                logger.warning(f"Tentando novamente com apenas {len(df_reduced)} parciais devido a restrições de memória")
                return self.calculate_dissonance_metric(df_reduced)
            except Exception:
                return 0.0
            
        except Exception as e:
            logger.error(f"Erro ao calcular métrica de dissonância: {e}")
            return 0.0


    def calculate_perceptual_weight(self, frequency: float, amplitude: float) -> float:
        """
        Calcula um peso perceptual para um parcial baseado em curvas de ponderação auditiva.
        
        Args:
            frequency: Frequência do parcial em Hz
            amplitude: Amplitude do parcial
            
        Returns:
            Peso perceptual entre 0 e 1
        """
        # Implementação básica de uma curva de ponderação auditiva simplificada baseada em A-weighting
        # Enfatiza frequências médias (1-5 kHz) e reduz baixas e altas frequências
        
        # Valores de referência
        f_ref = 1000.0  # Frequência de referência em Hz
        
        # Para frequências muito baixas ou muito altas, reduzir o peso
        if frequency < 20.0 or frequency > 20000.0:
            return 0.0
            
        # Aproximação simplificada de A-weighting
        r_a = ((12200**2) * (frequency**4)) / (
            (frequency**2 + 20.6**2) * 
            (frequency**2 + 12200**2) * 
            np.sqrt((frequency**2 + 107.7**2) * (frequency**2 + 737.9**2))
        )
        
        # Normalizar para 0-1
        weight = r_a / (r_a + 0.1)
        
        # Considerar também a amplitude (parciais mais fortes têm mais influência)
        weight *= amplitude
        
        return weight


class SetharesDissonance(DissonanceModel):
    """
    Modelo de dissonância de Sethares (versão revisada, 2ª ed., 2005).
    
    Baseado nas curvas de Plomp-Levelt, este modelo calcula a dissonância
    como função da diferença de frequência e das amplitudes dos parciais.
    
    Referência:
        Sethares, W. A. (2005). Tuning, Timbre, Spectrum, Scale (2nd ed.).
        Springer-Verlag.
    """
    def __init__(self):
        """Inicializa o modelo de dissonância de Sethares com parâmetros da versão revisada."""
        super().__init__("Sethares-Revised", 
                         "Modelo de dissonância baseado nas curvas de Plomp-Levelt (Sethares, 2005)")
        # Constantes Plomp–Levelt (revisadas)
        self.C1 = 5.0
        self.C2 = -5.0
        self.A1 = -3.51
        self.A2 = -5.75
        # Parâmetros de S(f)
        self.d_star = 0.24
        self.s1 = 0.0207
        self.s2 = 18.96

    def _S(self, f: float) -> float:
        """
        Calcula o fator S dependente da frequência.
        
        Args:
            f: Frequência em Hz
            
        Returns:
            Fator de escala S(f)
        """
        return self.d_star / (self.s1 * f + self.s2)

    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        """
        Calcula a dissonância entre dois tons puros usando o modelo de Sethares.
        
        Args:
            f1: Frequência do primeiro tom puro (Hz)
            f2: Frequência do segundo tom puro (Hz)
            a1: Amplitude do primeiro tom
            a2: Amplitude do segundo tom
            
        Returns:
            Dissonância entre os dois tons
        """
        # Ordenar pares (garantir f1 <= f2)
        if f1 > f2:
            f1, f2 = f2, f1
            a1, a2 = a2, a1

        # Calcular o fator S(f) e a diferença normalizada
        S = self._S(f1)
        x = S * (f2 - f1)
        
        # Usar amplitude mínima para ponderação (modelo de batimento)
        a = min(a1, a2)

        # Fórmula de dissonância Sethares
        return a * (
            self.C1 * np.exp(self.A1 * x) +
            self.C2 * np.exp(self.A2 * x)
        )


class HutchinsonKnopoffDissonance(DissonanceModel):
    """
    Implementação do modelo de dissonância de Hutchinson-Knopoff.
    
    Este modelo é uma extensão de Plomp-Levelt que usa explicitamente
    a largura de banda crítica para calcular dissonância.
    
    Referência:
        Hutchinson, W., & Knopoff, L. (1978). The acoustic component of
        Western consonance. Interface, 7, 1-29.
    """
    
    def __init__(self):
        """Inicializa o modelo de dissonância de Hutchinson-Knopoff."""
        super().__init__("Hutchinson-Knopoff", 
                         "Modelo de dissonância baseado em largura de banda crítica (Hutchinson & Knopoff, 1978)")
        
        # Parâmetros do modelo
        self.a = 3.5
        self.b = 5.75
        
        # Parâmetros de largura de banda crítica (de Hutchinson & Knopoff, 1978)
        self.cb_factor_1 = 1.2  # Fator de largura de banda crítica 1
        self.cb_factor_2 = 0.76  # Fator de largura de banda crítica 2
    
    @lru_cache(maxsize=1024)
    def critical_bandwidth(self, f: float) -> float:
        """
        Calcula a largura de banda crítica na frequência f.
        
        Usa a fórmula de Hutchinson & Knopoff (1978).
        
        Args:
            f: Frequência em Hz
            
        Returns:
            Largura de banda crítica em Hz
        """
        return self.cb_factor_1 * (f ** self.cb_factor_2)
    
    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        """
        Calcula a dissonância entre dois tons puros usando o modelo de Hutchinson-Knopoff.
        
        Args:
            f1: Frequência do primeiro tom puro (Hz)
            f2: Frequência do segundo tom puro (Hz)
            a1: Amplitude do primeiro tom
            a2: Amplitude do segundo tom
            
        Returns:
            Dissonância entre os dois tons
        """
        # Garantir f1 < f2
        if f1 > f2:
            f1, f2 = f2, f1
            a1, a2 = a2, a1
        
        # Calcular largura de banda crítica na frequência mais baixa
        cb = self.critical_bandwidth(f1)
        
        # Diferença de frequência expressa em termos de largura de banda crítica
        x = (f2 - f1) / cb
        
        # Calcular fator dependente da amplitude
        amplitude_factor = a1 * a2
        
        # Calcular dissonância usando largura de banda crítica
        return amplitude_factor * (np.exp(-self.a * x) - np.exp(-self.b * x))


class VassilakisDissonance(DissonanceModel):
    """
    Implementação do modelo de dissonância de Vassilakis (Flutuação Espectral).
    
    Este modelo concentra-se na flutuação de amplitude como o principal
    determinante da dissonância.
    
    Referência:
        Vassilakis, P. N. (2005). Auditory roughness as a means of musical
        expression. Selected Reports in Ethnomusicology, 12, 119-144.
    """
    
    def __init__(self):
        """Inicializa o modelo de dissonância de Vassilakis."""
        super().__init__("Vassilakis", 
                         "Modelo de dissonância baseado em flutuação espectral (Vassilakis, 2001, 2005)")
        
        # Parâmetros do modelo de Vassilakis (2001, 2005)
        self.alpha = 3.11  # Parâmetro de rugosidade
        self.beta = 5.09   # Parâmetro de rugosidade
        self.gamma = 0.5   # Parâmetro de flutuação de amplitude
        self.delta = 1.0   # Parâmetro de escala de amplitude
    
    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        """
        Calcula a dissonância entre dois tons puros usando o modelo de Vassilakis.
        
        Args:
            f1: Frequência do primeiro tom puro (Hz)
            f2: Frequência do segundo tom puro (Hz)
            a1: Amplitude do primeiro tom
            a2: Amplitude do segundo tom
            
        Returns:
            Dissonância entre os dois tons
        """
        # Garantir f1 < f2
        if f1 > f2:
            f1, f2 = f2, f1
            a1, a2 = a2, a1
            
        # Frequência mínima para evitar divisão por zero
        f1 = max(f1, 20.0)
        
        # Diferença de frequência
        freq_diff = f2 - f1
        
        # Fator de profundidade de flutuação de amplitude
        # Calcula a quantidade de flutuação de amplitude entre os dois tons
        amp_fluct = (2 * min(a1, a2)) / (a1 + a2)
        
        # Grau de flutuação de amplitude
        degree = amp_fluct**self.gamma
        
        # Calcular rugosidade a partir da diferença de frequência
        roughness = np.exp(-self.alpha * freq_diff / f1) - np.exp(-self.beta * freq_diff / f1)
        
        # Valor de dissonância combinado
        return degree * (a1 * a2)**self.delta * roughness


class AuresZwickerDissonance(DissonanceModel):
    """
    Rugosidade segundo Zwicker & Fastl (Handbook of Psychoacoustics, 1999, cap. 12)
    implementada nos moldes simplificados de Aures (1985).
    
    Referências:
        Zwicker, E., & Fastl, H. (1999). Psychoacoustics: Facts and models (2nd ed.).
        Springer-Verlag.
        
        Aures, W. (1985). Ein Berechnungsverfahren der Rauhigkeit. 
        Acustica, 58, 268-281.
    """
    def __init__(self):
        """Inicializa o modelo de dissonância de Aures-Zwicker."""
        super().__init__("Aures-Zwicker", 
                         "Modelo de rugosidade baseado em Zwicker & Fastl (1999) e Aures (1985)")
        self.gamma = 1.25       # Declive empírico da curva de rugosidade
        self.k = 0.25           # Fator de escala (ajustado para ~0–10)

    @staticmethod
    @lru_cache(maxsize=1024)
    def critical_bw(freq_hz: float) -> float:
        """
        Calcula a largura de banda crítica aproximada (Bark) em Hz.
        
        Args:
            freq_hz: Frequência em Hz
            
        Returns:
            Largura de banda crítica em Hz
        """
        return 25 + 75 * (1 + 1.4 * (freq_hz / 1000) ** 2) ** 0.69

    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        """
        Calcula a dissonância entre dois tons puros usando o modelo Aures-Zwicker.
        
        Args:
            f1: Frequência do primeiro tom puro (Hz)
            f2: Frequência do segundo tom puro (Hz)
            a1: Amplitude do primeiro tom
            a2: Amplitude do segundo tom
            
        Returns:
            Dissonância entre os dois tons
        """
        # Tons idênticos não têm dissonância
        if f1 == f2:
            return 0.0
            
        # Calcular frequência média e largura de banda crítica
        f_mean = (f1 + f2) / 2
        cbw = self.critical_bw(f_mean)
        
        # Diferença de frequência absoluta
        df = abs(f2 - f1)
        
        # Normalizar pela largura de banda crítica
        s = df / cbw
        
        # Forma simplificada: R = k · (a_min)^0.6 · s · exp(-γ s)
        return self.k * (min(a1, a2) ** 0.6) * s * math.exp(-self.gamma * s)


class StolzenburgHarmonicity(DissonanceModel):
    """
    Índice de harmonicidade global inspirado em Stolzenburg (2015, 2020).
    Mede quão bem os parciais se alinham com múltiplos inteiros de um f0 implícito.
    Devolve 1 para espectros perfeitamente harmônicos, 0 para altamente inarmônicos.
    
    Referências:
        Stolzenburg, F. (2015). Harmony perception by periodicity detection.
        Journal of Mathematics and Music, 9(3), 215-238.
        
        Stolzenburg, F. (2020). Periodicity detection by neural transformation.
        Front. Comput. Neurosci., 14, 62.
    """
    def __init__(self):
        """Inicializa o modelo de harmonicidade de Stolzenburg."""
        super().__init__("Stolzenburg-Harmonicity", 
                         "Índice de harmonicidade baseado em alinhamento de parciais (Stolzenburg, 2015)")

    # Não faz sentido par-a-par → devolve 0 para compatibilidade
    def pure_tones_dissonance(self, f1: float, f2: float, a1: float, a2: float) -> float:
        """
        Para compatibilidade com a interface DissonanceModel.
        A harmonicidade de Stolzenburg não é calculada entre pares de tons.
        
        Returns:
            Sempre retorna 0.0
        """
        return 0.0

    def calculate_dissonance_metric(self, df: pd.DataFrame) -> float:
        """
        Calcula o índice de harmonicidade para um conjunto de parciais.
        
        Args:
            df: DataFrame com colunas 'Frequency (Hz)' e 'Amplitude'
            
        Returns:
            Índice de harmonicidade (1.0 = perfeitamente harmônico, 0.0 = inarmônico)
            
        Raises:
            ValueError: Se o DataFrame não contiver as colunas necessárias.
        """
        # Validação de entrada
        if df is None or df.empty:
            logger.warning("DataFrame vazio ou None fornecido para calculate_dissonance_metric")
            return 0.0
            
        if 'Frequency (Hz)' not in df.columns:
            msg = "DataFrame deve conter a coluna 'Frequency (Hz)'"
            logger.error(msg)
            raise ValueError(msg)

        try:
            # Verificar a coluna de Amplitude ou converter de Magnitude (dB)
            if 'Amplitude' not in df.columns:
                if 'Magnitude (dB)' in df.columns:
                    df = df.copy()
                    df['Amplitude'] = 10 ** (df['Magnitude (dB)'] / 20)
                    logger.debug("Amplitude calculada a partir de 'Magnitude (dB)'")
                else:
                    msg = "DataFrame deve conter coluna 'Amplitude' ou 'Magnitude (dB)'"
                    logger.error(msg)
                    raise ValueError(msg)

            # Estimar f0 pelo mínimo comum divisor (robusto para espectros monofônicos)
            f0 = df['Frequency (Hz)'].min()
            amps = df['Amplitude'].values
            freqs = df['Frequency (Hz)'].values

            # Erro relativo de cada parcial ao múltiplo inteiro mais próximo
            k = np.round(freqs / f0)
            err = np.abs(freqs - k * f0) / freqs
            weighted_err = np.sum(err * amps) / np.sum(amps)

            # Índice de harmonicidade: 1 / (1 + erro médio)
            harmonicity = 1.0 / (1.0 + weighted_err)
            
            logger.debug(f"Índice de harmonicidade calculado: {harmonicity}")
            return harmonicity
            
        except Exception as e:
            logger.error(f"Erro ao calcular índice de harmonicidade: {e}")
            raise


class SpectralAutocorrelationHarmonicity(DissonanceModel):
    """
    Métrica de harmonicidade global baseada na Autocorrelação Espectral (SACf).
    Valores:
        •  ≈1.0  → espectro altamente harmónico
        •  <0.2 → espectro inarmónico / ruidoso
    """

    def __init__(self) -> None:
        super().__init__(
            name="Spectral-Autocorrelation",
            description=(
                "Harmonicidade calculada pela autocorrelação do espectro de "
                "magnitude em escala logarítmica de frequência (SACf)."
            ),
        )

    # -------------------------------------------------------------- #
    #  Interface exigida pelo AudioProcessor
    # -------------------------------------------------------------- #
    def pure_tones_dissonance(
        self, f1: float, f2: float, a1: float, a2: float
    ) -> float:
        # A SACf é global; não faz sentido par-a-par  
        return 0.0

    def analyze_raw_audio(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
        window_size: int = 2048,
        hop_length: Optional[int] = None,
        plot_results: bool = False,
    ) -> Tuple[float, Dict[str, Any]]:
        if hop_length is None:
            hop_length = window_size // 4

        harmonicity, results = self.spectral_autocorrelation_function(
            audio_signal=audio_signal,
            sample_rate=sample_rate,
            window_size=window_size,
            hop_length=hop_length,
            plot_results=plot_results,
        )
        results["harmonicity"] = harmonicity
        return harmonicity, results

    # -------------------------------------------------------------- #
    #  Compatibilidade com rotinas que fornecem apenas parciais
    # -------------------------------------------------------------- #
    def calculate_dissonance_metric(self, df: pd.DataFrame) -> float:
        """
        Reconstrói um espectro sintético a partir dos parciais e calcula a SACf.
        """
        if (
            df is None
            or df.empty
            or "Frequency (Hz)" not in df.columns
        ):
            logger.warning("DataFrame inválido em calculate_dissonance_metric()")
            return 0.0

        if "Amplitude" not in df.columns:
            if "Magnitude (dB)" in df.columns:
                df = df.copy()
                df["Amplitude"] = 10 ** (df["Magnitude (dB)"] / 20)
            else:
                raise ValueError(
                    "DataFrame deve conter 'Amplitude' ou 'Magnitude (dB)'."
                )

        sr = 48_000          # amostragem fictícia
        n_fft = 4096
        spectrum = np.zeros(n_fft // 2 + 1)
        freqs = np.fft.rfftfreq(n_fft, 1 / sr)

        for f, a in zip(df["Frequency (Hz)"], df["Amplitude"]):
            if f <= 0:
                continue
            bin_idx = np.argmin(np.abs(freqs - f))
            spectrum[bin_idx] += a

        harmonicity, _ = self._sacf_core(
            magnitude_vector=spectrum,
            frequencies=freqs,
            min_freq=20.0,
            max_freq=min(8000.0, freqs[-1]),
        )
        return harmonicity

    # -------------------------------------------------------------- #
    #  Núcleo SACf reutilizável
    # -------------------------------------------------------------- #
    @staticmethod
    def _sacf_core(
        magnitude_vector: np.ndarray,
        frequencies: np.ndarray,
        min_freq: float,
        max_freq: float,
        freq_resolution: int = 128,
    ) -> Tuple[float, np.ndarray]:
        mask = (frequencies >= min_freq) & (frequencies <= max_freq)
        if not np.any(mask):
            return 0.0, np.zeros(freq_resolution)

        mag = magnitude_vector[mask]
        freqs = frequencies[mask]

        log_freqs = np.logspace(
            np.log10(min_freq), np.log10(max_freq), freq_resolution
        )
        log_mag = np.interp(log_freqs, freqs, mag)

        ac = irfft(np.abs(rfft(log_mag)) ** 2)[:freq_resolution]
        if ac[0] == 0:
            return 0.0, ac

        rho = ac / ac[0]
        peak = int(np.argmax(rho[1:]) + 1)
        return float(rho[peak]), rho

    # -------------------------------------------------------------- #
    #  Função “oficial” chamada a partir do sinal completo
    # -------------------------------------------------------------- #
    def spectral_autocorrelation_function(
        self,
        audio_signal: np.ndarray,
        sample_rate: int,
        window_size: int = 2048,
        hop_length: int = 512,
        min_freq: float = 20.0,
        max_freq: float = 8000.0,
        freq_resolution: int = 128,
        plot_results: bool = False,
        save_path: Optional[str] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        S = librosa.stft(
            audio_signal,
            n_fft=window_size,
            hop_length=hop_length,
            window="hann",
        )
        mean_mag = np.mean(np.abs(S), axis=1)
        freqs = librosa.fft_frequencies(sr=sample_rate, n_fft=window_size)

        harmonicity, rho = self._sacf_core(
            mean_mag,
            freqs,
            min_freq,
            max_freq,
            freq_resolution,
        )

        # Frequência fundamental estimada (opcional)
        res_ratio = np.log2(max_freq / min_freq) / freq_resolution
        peak_idx = int(np.argmax(rho[1:]) + 1)
        est_f0 = min_freq * 2 ** (peak_idx * res_ratio) if peak_idx else 0.0

        results: Dict[str, Any] = {
            "autocorrelation": rho,
            "peak_index": peak_idx,
            "harmonicity": harmonicity,
            "log_frequencies": np.logspace(
                np.log10(min_freq), np.log10(max_freq), freq_resolution
            ),
            "mean_spectrum": mean_mag,
            "estimated_f0": est_f0,
            "peaks": scipy.signal.find_peaks(rho, height=0.2)[0],
        }

        if plot_results:
            self._plot_sacf_results(results, save_path)

        return harmonicity, results

    # -------------------------------------------------------------- #
    #  Gráfico opcional
    # -------------------------------------------------------------- #
    @staticmethod
    def _plot_sacf_results(
        results: Dict[str, Any],
        save_path: Optional[str] = None,
    ) -> None:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

        ax1.semilogx(
            results["log_frequencies"],
            results["mean_spectrum"],
        )
        ax1.set(
            title="Espectro médio (escala log-Hz)",
            xlabel="Frequência (Hz)",
            ylabel="Magnitude",
        )
        ax1.grid(True, which="both", ls="--", alpha=0.3)

        ax2.plot(results["autocorrelation"], label="ρ[d]")
        ax2.scatter(
            results["peak_index"],
            results["harmonicity"],
            label=f"H = {results['harmonicity']:.3f}",
        )
        ax2.set(
            xlabel="Lag (amostras)",
            ylabel="Autocorrelação normalizada",
        )
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()


# Registro centralizado dos modelos disponíveis
_DISSONANCE_MODELS = {
    'sethares': SetharesDissonance,
    'hutchinson-knopoff': HutchinsonKnopoffDissonance,
    'vassilakis': VassilakisDissonance,
    'aures-zwicker': AuresZwickerDissonance,
    'stolzenburg': StolzenburgHarmonicity,
    'spectral-autocorrelation': SpectralAutocorrelationHarmonicity,
}


def get_dissonance_model(name: str) -> DissonanceModel:
    """
    Retorna uma instância do modelo solicitado (key insensível a maiúsculas).
    
    Args:
        name: Nome do modelo de dissonância.
        
    Returns:
        Uma instância do modelo de dissonância solicitado.
        
    Raises:
        ValueError: Se o nome do modelo não for reconhecido.
    """
    key = name.strip().lower()
    if key not in _DISSONANCE_MODELS:
        valid_models = list(_DISSONANCE_MODELS.keys())
        logger.error(f"Modelo de dissonância desconhecido: {name}")
        raise ValueError(f"Modelo de dissonância desconhecido: {name}. Modelos válidos: {valid_models}")
    
    try:
        model_instance = _DISSONANCE_MODELS[key]()
        logger.debug(f"Modelo de dissonância instanciado: {name}")
        return model_instance
    except Exception as e:
        logger.error(f"Erro ao instanciar modelo de dissonância {name}: {e}")
        raise


def list_available_models() -> List[str]:
    """
    Lista todos os modelos de dissonância disponíveis.
    
    Returns:
        Lista de nomes de modelos disponíveis.
    """
    return list(_DISSONANCE_MODELS.keys())


def get_model_description(name: str) -> str:
    """
    Obtém a descrição de um modelo de dissonância.
    
    Args:
        name: Nome do modelo de dissonância.
        
    Returns:
        Descrição do modelo.
        
    Raises:
        ValueError: Se o nome do modelo não for reconhecido.
    """
    model = get_dissonance_model(name)
    return model.description


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
    """
    Compara curvas de dissonância de diferentes modelos para o mesmo timbre.
    
    Args:
        partials: Lista de tuplas (frequência, amplitude) para o timbre
        min_interval: Menor intervalo a analisar (proporção)
        max_interval: Maior intervalo a analisar (proporção)
        num_points: Número de pontos na curva
        save_file: Se fornecido, salva o gráfico neste arquivo
        models_to_include: Lista de nomes de modelos a incluir (se None, usa todos)
        normalize_curves: Se True, normaliza cada curva para faixa 0-1 para melhor comparação
        show_minima: Se True, marca os mínimos locais nas curvas
        add_cent_axis: Se True, adiciona um eixo secundário com centésimos
        dpi: Resolução do gráfico salvo (pontos por polegada)
        
    Returns:
        Dicionário com as curvas de dissonância calculadas por cada modelo
        
    Raises:
        ValueError: Se os parâmetros ou parciais forem inválidos.
    """
    # Validação de entrada
    if not partials:
        logger.warning("Lista de parciais vazia fornecida para compare_dissonance_models")
        return {}
        
    if min_interval <= 0 or max_interval <= min_interval:
        logger.error(f"Intervalos inválidos: min={min_interval}, max={max_interval}")
        raise ValueError("Intervalo mínimo deve ser > 0 e máximo > mínimo")
        
    if num_points < 2:
        logger.error(f"Número de pontos inválido: {num_points}")
        raise ValueError(f"Número de pontos deve ser >= 2, obtido: {num_points}")
    
    try:
        # Instanciar os modelos a serem comparados
        if models_to_include is None:
            models = [get_dissonance_model(name) for name in list_available_models()]
        else:
            models = [get_dissonance_model(name) for name in models_to_include]
            
        logger.info(f"Comparando {len(models)} modelos de dissonância: {[m.name for m in models]}")
            
        # Calcular curvas de dissonância para cada modelo
        curves = {}
        for model in models:
            curves[model.name] = model.calculate_dissonance_curve(
                partials, min_interval, max_interval, num_points)
            
        # Criar gráfico
        plt.figure(figsize=(14, 8))
        
        # Plotar cada curva
        colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k', 'orange', 'purple', 'brown']  
        markers = ['o', 's', '^', 'v', 'D', 'p', 'h', '8', '*', '+']
        
        # Garantir que temos cores e marcadores suficientes
        while len(colors) < len(models):
            colors.extend(colors)
        while len(markers) < len(models):
            markers.extend(markers)
        
        for i, (model_name, curve) in enumerate(curves.items()):
            intervals = sorted(list(curve.keys()))
            dissonance_values = [curve[i] for i in intervals]
            
            # Normalizar cada curva para faixa 0-1 para melhor comparação, se solicitado
            if normalize_curves:
                max_diss = max(dissonance_values)
                min_diss = min(dissonance_values)
                if max_diss > min_diss:
                    dissonance_values = [(d - min_diss) / (max_diss - min_diss) for d in dissonance_values]
            
            plt.plot(intervals, dissonance_values, f'{colors[i]}-', 
                    linewidth=2, label=model_name)
            
            # Encontrar e marcar mínimos locais, se solicitado
            if show_minima:
                model = models[i] 
                minima = model.find_local_minima(curve)
                
                # Se normalizamos a curva, precisamos normalizar os valores dos mínimos também
                if normalize_curves and max_diss > min_diss:
                    minima_y = [(curve[m] - min_diss) / (max_diss - min_diss) for m in minima]
                else:
                    minima_y = [curve[m] for m in minima]
                    
                plt.plot(minima, minima_y, f'{colors[i]}{markers[i]}', markersize=6)
                
                # Adicionar anotações para cada mínimo em cents
                if len(minima) > 0:
                    logger.info(f"Mínimos para modelo {model_name}: {minima}")
                    # Anotar apenas o primeiro e último mínimo para não sobrecarregar o gráfico
                    for idx, m in enumerate([minima[0], minima[-1]]):
                        if idx < len(minima):
                            cents = int(1200 * np.log2(m))
                            plt.annotate(f"{cents}¢", 
                                       (m, minima_y[idx]),
                                       xytext=(0, 10 if idx == 0 else -15), 
                                       textcoords='offset points',
                                       ha='center',
                                       fontsize=8,
                                       color=colors[i])
        
        # Configurar gráfico
        plt.title("Comparison of Dissonance Models")
        plt.xlabel('Frequency Ratio')
        plt.ylabel('Normalized Dissonance' if normalize_curves else 'Dissonance')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Adicionar eixo secundário com centésimos, se solicitado
        if add_cent_axis:
            ax1 = plt.gca()
            ax2 = ax1.twiny()
            ax2.set_xlim(ax1.get_xlim())
            
            # Configurar marcações de centésimos
            cent_ticks = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
            ratio_ticks = [2**(c/CENTS_PER_OCTAVE) for c in cent_ticks]
            ax2.set_xticks(ratio_ticks)
            ax2.set_xticklabels([f"{c}¢" for c in cent_ticks])
            ax2.set_xlabel('Cents')
        
        # Salvar ou mostrar
        if save_file:
            plt.savefig(save_file, dpi=dpi, bbox_inches='tight')
            logger.info(f"Comparação de modelos de dissonância salva em: {save_file}")
            plt.close()
        else:
            plt.show()
            plt.close()
            
        return curves
        
    except Exception as e:
        logger.error(f"Erro ao comparar modelos de dissonância: {e}")
        plt.close()
        raise


def calculate_all_dissonance_metrics(df: pd.DataFrame, 
                                    normalize: bool = True,
                                    include_models: Optional[List[str]] = None) -> Dict[str, float]:
    """
    Calcula métricas de dissonância para um DataFrame usando todos os modelos disponíveis.
    
    Args:
        df: DataFrame com as colunas 'Frequency (Hz)' e 'Amplitude' ou 'Magnitude (dB)'
        normalize: Se True, normaliza os valores para uma faixa comum
        include_models: Lista de nomes de modelos a incluir (se None, usa todos)
        
    Returns:
        Dicionário com nomes de modelos como chaves e valores de dissonância como valores
        
    Raises:
        ValueError: Se o DataFrame não contiver as colunas necessárias.
    """
    # Validação de entrada
    if df is None or df.empty:
        logger.warning("DataFrame vazio ou None fornecido para calculate_all_dissonance_metrics")
        return {}
    
    try:
        # Determinar quais modelos calcular
        if include_models is None:
            models = [get_dissonance_model(name) for name in list_available_models()]
        else:
            models = [get_dissonance_model(name) for name in include_models]
        
        # Calcular métricas para cada modelo
        results = {}
        for model in models:
            results[model.name] = model.calculate_dissonance_metric(df)
            
        # Normalizar valores, se solicitado
        if normalize and results:
            max_value = max(results.values())
            min_value = min(results.values())
            if max_value > min_value:
                for model_name in results:
                    results[model_name] = (results[model_name] - min_value) / (max_value - min_value)
        
        return results
        
    except Exception as e:
        logger.error(f"Erro ao calcular métricas de dissonância: {e}")
        raise


def compare_common_intervals(models: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compara os mínimos de dissonância (intervalos consonantes) previstos por diferentes modelos
    para um espectro harmônico simples.
    
    Args:
        models: Lista de nomes de modelos a incluir (se None, usa todos)
        
    Returns:
        DataFrame com intervalos musicais comuns e suas dissonâncias para cada modelo
        
    Raises:
        ValueError: Se algum nome de modelo não for reconhecido.
    """
    # Criar um espectro harmônico simples com 8 harmônicos, amplitudes decrescentes
    f0 = 440.0  # Hz (A4)
    harmonics = [(f0 * i, 1.0 / i) for i in range(1, 9)]
    
    # Intervalos musicais comuns (proporções de frequência)
    intervals = {
        "Unison": 1.0,
        "Minor Second": 16/15,
        "Major Second": 9/8,
        "Minor Third": 6/5,
        "Major Third": 5/4,
        "Perfect Fourth": 4/3,
        "Tritone": 45/32,
        "Perfect Fifth": 3/2,
        "Minor Sixth": 8/5,
        "Major Sixth": 5/3,
        "Minor Seventh": 9/5,
        "Major Seventh": 15/8,
        "Octave": 2.0
    }
    
    try:
        # Determinar quais modelos usar
        if models is None:
            model_instances = [get_dissonance_model(name) for name in list_available_models()]
        else:
            model_instances = [get_dissonance_model(name) for name in models]
            
        # Calcular dissonância para cada intervalo e modelo
        results = {}
        
        # Adicionar colunas para nomes de intervalos e razões
        results["Interval"] = list(intervals.keys())
        results["Ratio"] = [intervals[name] for name in results["Interval"]]
        results["Cents"] = [1200 * np.log2(ratio) for ratio in results["Ratio"]]
        
        # Calcular dissonâncias
        for model in model_instances:
            dissonances = []
            for interval_name, ratio in intervals.items():
                # Calcular dissonância do mesmo timbre deslocado pelo intervalo
                dissonance = model.same_timbre_dissonance(harmonics, ratio)
                dissonances.append(dissonance)
                
            # Normalizar para faixa 0-1
            max_diss = max(dissonances)
            min_diss = min(dissonances)
            if max_diss > min_diss:
                dissonances = [(d - min_diss) / (max_diss - min_diss) for d in dissonances]
                
            results[model.name] = dissonances
            
        # Criar DataFrame com os resultados
        df = pd.DataFrame(results)
        
        return df
        
    except Exception as e:
        logger.error(f"Erro ao comparar intervalos comuns: {e}")
        raise


def plot_interval_comparison(df: pd.DataFrame, 
                           save_file: Optional[str] = None,
                           sort_by_interval: bool = True,
                           highlight_consonances: bool = True,
                           dpi: int = DEFAULT_PLOT_DPI):
    """
    Plota uma comparação visual da dissonância de intervalos comuns
    entre diferentes modelos.
    
    Args:
        df: DataFrame retornado por compare_common_intervals
        save_file: Se fornecido, salva o gráfico neste arquivo
        sort_by_interval: Se True, ordena por tamanho do intervalo; se False, por nome
        highlight_consonances: Se True, destaca intervalos tradicionalmente consonantes
        dpi: Resolução do gráfico salvo (pontos por polegada)
    """
    if df is None or df.empty:
        logger.warning("DataFrame vazio ou None fornecido para plot_interval_comparison")
        return
    
    try:
        # Criar cópia para não modificar o original
        plot_df = df.copy()
        
        # Ordenar por tamanho do intervalo ou por nome
        if sort_by_interval:
            plot_df = plot_df.sort_values(by="Cents")
        
        # Configurar o gráfico
        plt.figure(figsize=(14, 8))
        
        # Obter nomes dos modelos
        model_names = [col for col in plot_df.columns 
                      if col not in ["Interval", "Ratio", "Cents"]]
        
        # Número de modelos e intervalos
        n_models = len(model_names)
        n_intervals = len(plot_df)
        
        # Criar posições x para as barras
        x = np.arange(n_intervals)
        width = 0.8 / n_models  # Largura de cada barra
        
        # Cores para os modelos
        colors = plt.cm.tab10(np.linspace(0, 1, n_models))
        
        # Plotar barras para cada modelo
        for i, model_name in enumerate(model_names):
            offset = (i - n_models / 2 + 0.5) * width
            bars = plt.bar(x + offset, plot_df[model_name], width, label=model_name, color=colors[i])
        
        # Destacar intervalos tradicionalmente consonantes
        if highlight_consonances:
            consonant_intervals = ["Unison", "Perfect Fourth", "Perfect Fifth", "Octave", 
                                 "Major Third", "Minor Third", "Major Sixth"]
            
            highlight_indices = []
            for idx, interval in enumerate(plot_df["Interval"]):
                if interval in consonant_intervals:
                    highlight_indices.append(idx)
            
            if highlight_indices:
                plt.axhspan(0, 0.3, alpha=0.1, color='green', zorder=0)
                for idx in highlight_indices:
                    plt.axvspan(idx - 0.5, idx + 0.5, alpha=0.1, color='green', zorder=0)
        
        # Configurar rótulos e título
        plt.ylabel('Normalized Dissonance')
        plt.title('Comparison of Dissonance Models for Common Musical Intervals')
        plt.xticks(x, plot_df["Interval"], rotation=45, ha='right')
        
        # Adicionar rótulos de centésimos abaixo dos nomes dos intervalos
        ax1 = plt.gca()
        ax2 = ax1.twiny()
        ax2.set_xticks(x)
        ax2.set_xticklabels([f"{c:.0f}¢" for c in plot_df["Cents"]], rotation=45, ha='left')
        
        # Ajustar layout e adicionar legenda
        plt.tight_layout()
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=n_models, frameon=False)
        
        # Salvar ou mostrar
        if save_file:
            plt.savefig(save_file, dpi=dpi, bbox_inches='tight')
            logger.info(f"Comparação de intervalos salva em: {save_file}")
            plt.close()
        else:
            plt.show()
            plt.close()
            
    except Exception as e:
        logger.error(f"Erro ao plotar comparação de intervalos: {e}")
        plt.close()
        raise


def analyze_real_timbre(df: pd.DataFrame, 
                       note_name: str = "",
                       include_models: Optional[List[str]] = None,
                       save_directory: Optional[str] = None) -> Dict[str, Any]:
    """
    Realiza uma análise completa de dissonância para um timbre real,
    calculando métricas de dissonância, gerando curvas e escalas ótimas.
    
    Args:
        df: DataFrame com as colunas 'Frequency (Hz)' e 'Amplitude' ou 'Magnitude (dB)'
        note_name: Nome da nota para identificação nos gráficos
        include_models: Lista de nomes de modelos a incluir (se None, usa todos)
        save_directory: Se fornecido, salva os resultados neste diretório
        
    Returns:
        Dicionário com resultados da análise (métricas, curvas, escalas)
        
    Raises:
        ValueError: Se o DataFrame não contiver as colunas necessárias.
    """
    # Validação de entrada
    if df is None or df.empty:
        logger.warning("DataFrame vazio ou None fornecido para analyze_real_timbre")
        return {}
        
    # Verificar colunas necessárias
    if 'Frequency (Hz)' not in df.columns:
        msg = "DataFrame deve conter a coluna 'Frequency (Hz)'"
        logger.error(msg)
        raise ValueError(msg)
        
    if 'Amplitude' not in df.columns:
        if 'Magnitude (dB)' in df.columns:
            df = df.copy()
            df['Amplitude'] = 10 ** (df['Magnitude (dB)'] / 20)
            logger.debug("Amplitude calculada a partir de 'Magnitude (dB)'")
        else:
            msg = "DataFrame deve conter coluna 'Amplitude' ou 'Magnitude (dB)'"
            logger.error(msg)
            raise ValueError(msg)
    
    try:
        # Determinar quais modelos usar
        if include_models is None:
            models = [get_dissonance_model(name) for name in list_available_models()]
        else:
            models = [get_dissonance_model(name) for name in include_models]
            
        # Preparar diretório para salvar resultados
        if save_directory:
            os.makedirs(save_directory, exist_ok=True)
        
        # Extrair parciais como tuplas (frequência, amplitude)
        partials = list(zip(df['Frequency (Hz)'].values, df['Amplitude'].values))
        
        # Resultados
        results = {
            "note_name": note_name,
            "metrics": {},
            "curves": {},
            "scales": {},
            "scale_cents": {}
        }
        
        # Calcular métricas, curvas e escalas para cada modelo
        for model in models:
            model_name = model.name
            
            # 1. Calcular métrica de dissonância
            metric = model.calculate_dissonance_metric(df)
            results["metrics"][model_name] = metric
            
            # 2. Calcular curva de dissonância
            curve = model.calculate_dissonance_curve(partials, 1.0, 2.0, 200)
            results["curves"][model_name] = curve
            
            # 3. Gerar escala ótima
            scale = model.find_local_minima(curve)
            
            # Garantir que 1.0 e 2.0 estejam na escala
            if 1.0 not in scale:
                scale.insert(0, 1.0)
            if 2.0 not in scale:
                scale.append(2.0)
                
            # Ordenar a escala
            scale = sorted(scale)
            results["scales"][model_name] = scale
            
            # Converter para centésimos
            scale_cents = [1200 * np.log2(ratio) for ratio in scale]
            results["scale_cents"][model_name] = scale_cents
            
            # 4. Visualizar curva de dissonância com a escala
            if save_directory:
                title = f"{model_name} Dissonance Curve - {note_name}" if note_name else f"{model_name} Dissonance Curve"
                curve_path = os.path.join(save_directory, f"{model_name.lower()}_dissonance_curve.png")
                
                model.visualize_dissonance_curve(
                    curve, scale, title=title, save_file=curve_path
                )
        
        # 5. Comparar todos os modelos em um único gráfico
        if save_directory and len(models) > 1:
            comparison_path = os.path.join(save_directory, "dissonance_comparison.png")
            compare_dissonance_models(
                partials, save_file=comparison_path,
                models_to_include=[m.name for m in models]
            )
        
        # 6. Criar tabela comparativa de escalas
        if save_directory:
            # Preparar dados para a tabela
            scale_data = []
            
            # Encontrar o número máximo de notas em qualquer escala
            max_notes = max(len(scale) for scale in results["scales"].values())
            
            # Preencher a tabela
            for i in range(max_notes):
                row = {"Note": i + 1}
                
                for model_name in results["scales"].keys():
                    scale = results["scales"][model_name]
                    scale_cents = results["scale_cents"][model_name]
                    
                    if i < len(scale):
                        row[f"{model_name} (ratio)"] = f"{scale[i]:.3f}"
                        row[f"{model_name} (cents)"] = f"{scale_cents[i]:.1f}¢"
                    else:
                        row[f"{model_name} (ratio)"] = ""
                        row[f"{model_name} (cents)"] = ""
                        
                scale_data.append(row)
                
            # Criar DataFrame e salvar
            scale_df = pd.DataFrame(scale_data)
            scale_path = os.path.join(save_directory, "optimal_scales.csv")
            scale_df.to_csv(scale_path, index=False)
            
            # Também salvar métricas
            metrics_df = pd.DataFrame({"Model": list(results["metrics"].keys()),
                                       "Dissonance": list(results["metrics"].values())})
            metrics_path = os.path.join(save_directory, "dissonance_metrics.csv")
            metrics_df.to_csv(metrics_path, index=False)
        
        return results
        
    except Exception as e:
        logger.error(f"Erro ao analisar timbre real: {e}")
        raise


# Exportar funções e classes úteis
__all__ = [
    'DissonanceModel',
    'SetharesDissonance',
    'HutchinsonKnopoffDissonance',
    'VassilakisDissonance',
    'AuresZwickerDissonance',
    'StolzenburgHarmonicity',
    'get_dissonance_model',
    'list_available_models',
    'compare_dissonance_models',
    'calculate_all_dissonance_metrics',
    'compare_common_intervals',
    'plot_interval_comparison',
    'analyze_real_timbre'
]


# Exemplos de uso
if __name__ == "__main__":
    print("Módulo de modelos de dissonância")
    print("Modelos disponíveis:")
    for model_name in list_available_models():
        model = get_dissonance_model(model_name)
        print(f"  - {model.name}: {model.description}")
    
    # Exemplo: Gerar uma escala para um som harmônico simples
    f0 = 440.0  # A4
    harmonics = [(f0 * i, 1.0 / i) for i in range(1, 9)]  # 8 harmônicos
    
    model = get_dissonance_model("sethares")
    scale = model.generate_scale(harmonics)
    scale_cents = [1200 * np.log2(ratio) for ratio in scale]
    
    print("\nEscala gerada para um espectro harmônico simples (Sethares):")
    for i, (ratio, cents) in enumerate(zip(scale, scale_cents)):
        print(f"  Nota {i+1}: Razão = {ratio:.3f}, Cents = {cents:.1f}¢")
