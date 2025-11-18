# density_corrections_integration.py
"""
Módulo de integração das correções de densidade espectral.
Este arquivo facilita a aplicação das correções ao código existente.
"""

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class DensityMetricsCorrector:
    """
    Classe que encapsula todas as correções das métricas de densidade.
    """
    
    def __init__(self, audio_processor):
        """
        Inicializa o corretor com uma referência ao AudioProcessor.
        
        Args:
            audio_processor: Instância do AudioProcessor a ser corrigida
        """
        self.ap = audio_processor
        self.original_methods = {}
        self.corrections_applied = False
        
    def apply_corrections(self):
        """
        Aplica todas as correções ao AudioProcessor.
        """
        if self.corrections_applied:
            logger.warning("Correções já foram aplicadas anteriormente")
            return
            
        # Salvar métodos originais
        self.original_methods['_calculate_metrics'] = self.ap._calculate_metrics
        self.original_methods['_generate_harmonic_list'] = self.ap._generate_harmonic_list
        
        # Aplicar correções
        self.ap._calculate_metrics = self._calculate_metrics_corrected
        self.ap._generate_harmonic_list = self._generate_harmonic_list_corrected
        
        # Adicionar novos atributos
        if not hasattr(self.ap, 'fundamental_frequency'):
            self.ap.fundamental_frequency = None
            
        self.corrections_applied = True
        logger.info("Correções de densidade espectral aplicadas com sucesso")
        
    def revert_corrections(self):
        """
        Reverte as correções, restaurando os métodos originais.
        """
        if not self.corrections_applied:
            logger.warning("Nenhuma correção foi aplicada")
            return
            
        # Restaurar métodos originais
        for method_name, original_method in self.original_methods.items():
            setattr(self.ap, method_name, original_method)
            
        self.corrections_applied = False
        logger.info("Correções revertidas - métodos originais restaurados")
        
    def get_metrics_comparison(self) -> Dict[str, Any]:
        """
        Compara métricas antes e depois das correções.
        
        Returns:
            Dicionário com comparação das métricas
        """
        if not self.corrections_applied:
            logger.error("Correções não foram aplicadas ainda")
            return {}
            
        comparison = {
            'density_metric': {
                'original': getattr(self.ap, 'density_metric_value_original', None),
                'corrected': self.ap.density_metric_value
            },
            'combined_density': {
                'original': getattr(self.ap, 'combined_density_metric_value_original', None),
                'corrected': self.ap.combined_density_metric_value
            },
            'harmonic_count': len(self.ap.harmonic_list_df) if hasattr(self.ap, 'harmonic_list_df') else 0
        }
        
        return comparison
        
    def _calculate_metrics_corrected(self):
        """
        Método corrigido para cálculo de métricas.
        """
        # Implementação do calculate_metrics_corrected
        # (código da função calculate_metrics_corrected aqui)
        
    def _generate_harmonic_list_corrected(self, note, freq_max, tolerance):
        """
        Método corrigido para geração da lista harmônica.
        """
        # Implementação do generate_harmonic_list_corrected
        # (código da função generate_harmonic_list_corrected aqui)


def validate_corrections(test_cases: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Valida as correções com casos de teste específicos.
    
    Args:
        test_cases: Dicionário com nome do teste e array de amplitudes
        
    Returns:
        DataFrame com resultados da validação
    """
    results = []
    
    for test_name, amplitudes in test_cases.items():
        # Calcular métricas corrigidas
        from spectral_metrics_corrections import (
            calculate_harmonic_density_corrected,
            calculate_spectral_concentration_corrected,
            compute_spectral_entropy_corrected
        )
        
        density = calculate_harmonic_density_corrected(amplitudes, max_harmonics=20)
        concentration = calculate_spectral_concentration_corrected(amplitudes)
        entropy = compute_spectral_entropy_corrected(amplitudes**2, normalize_by_length=False)
        effective_partials = 2**entropy if entropy > 0 else 1
        
        results.append({
            'Test': test_name,
            'N_Partials': len(amplitudes),
            'Density': density,
            'Concentration': concentration,
            'Entropy': entropy,
            'Effective_Partials': effective_partials
        })
        
    return pd.DataFrame(results)


# Exemplo de uso
if __name__ == "__main__":
    # Criar casos de teste
    test_cases = {
        'Single_Partial': np.array([1.0]),
        'Two_Equal': np.array([0.5, 0.5]),
        'Five_Equal': np.ones(5) / 5,
        'Ten_Equal': np.ones(10) / 10,
        'Dominant_Partial': np.array([0.9, 0.05, 0.05]),
        'Decaying_Series': np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    }
    
    # Validar correções
    validation_df = validate_corrections(test_cases)
    print("\nResultados da Validação:")
    print(validation_df.to_string(index=False))
