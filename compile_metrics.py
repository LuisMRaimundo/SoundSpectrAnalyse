# compile_metrics.py - Módulo melhorado

"""
Módulo para compilação e análise de métricas de densidade e dissonância.

Este módulo oferece funções para extrair métricas de arquivos de análise espectral,
compilar resultados de múltiplos arquivos, ordenar notas musicais, aplicar análise
de componentes principais (PCA) e normalizar métricas para comparação.

Melhorias:
- Documentação expandida
- Tratamento de erros mais robusto
- Validação de parâmetros mais completa
- Otimização de desempenho com cache
- Sistema de registro (logging) aprimorado
- Relatórios de análise estatística
- Visualizações dinâmicas de PCA e correlação
"""


from __future__ import annotations

import pandas as pd

import os
import re
import logging
from pathlib import Path
from functools import lru_cache
from typing import Optional, Union, Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from proc_audio import AudioProcessor


# Importação robusta da função de densidade
try:
    from density import compute_spectral_entropy, get_weight_function
except ImportError:
    import sys, os
    sys.path.append(os.path.dirname(__file__))



# Configuração de logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():  # evita duplicação de handlers
    from log_config import configure_root_logger
    configure_root_logger()            # <-- NUNCA duplifica
    logger = logging.getLogger(__name__)


_PITCH_TO_SEMITONE = {
    "C": 0,  "C#": 1, "Db": 1,
    "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,  "Fb": 4, "E#": 5,
    "F": 5,  "F#": 6, "Gb": 6,
    "G": 7,  "G#": 8, "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
    "H": 11, "Hb": 10, "H#": 0,
}
_NOTE_RX = re.compile(r"\s*([A-GH])\s*([#♯b♭]?)\s*(-?\d+)\s*$", re.IGNORECASE)

def note_to_midi(note: str) -> int:
    """Converte 'A#2', 'Bb3', 'B2', 'H2' → número MIDI para ordenação cromática global."""
    if not isinstance(note, str):
        return 10**9  # vai para o fim
    m = _NOTE_RX.fullmatch(note)
    if not m:
        return 10**9
    letter = m.group(1).upper()
    acc = m.group(2)
    octv = int(m.group(3))
    if acc in ("#", "♯"):
        key = f"{letter}#"
    elif acc in ("b", "♭"):
        key = f"{letter}b"
    else:
        key = letter
    semi = _PITCH_TO_SEMITONE.get(key)
    if semi is None:
        return 10**9
    return (octv + 1) * 12 + semi



# Constantes
METRIC_COLUMNS: List[str] = [
    "Density Metric",
    "Spectral Density Metric",
    "Total Metric",
    "Combined Density Metric",
    "Filtered Density Metric",
    "Spectral Entropy",
]

# Campos de texto que NUNCA devem ser convertidos para float
# Campos de texto que NUNCA devem ser convertidos para float
TEXT_FIELDS: set[str] = {
    "Note",
    "Folder",
    "Analysis Type",
    "Weight Function",
    "Window",
    "DM Domain",
    "Density Scale",   # <— novo: 'bark', 'mel', 'hz', etc. é texto
}


DISSONANCE_PREFIX: str = "Dissonance"

def _minmax(series: pd.Series) -> pd.Series:
    """
    Normalização min–max robusta (0..1).
    - Converte para numérico (não numéricos -> NaN).
    - Se max==min ou só houver NaN, devolve zeros.
    """
    s = pd.to_numeric(series, errors="coerce")
    lo, hi = s.min(skipna=True), s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - lo) / (hi - lo)


def apply_weighted_index(df: pd.DataFrame, scheme: str = "pdf") -> pd.DataFrame:
    """
    Acrescenta colunas normalizadas e o índice ponderado ao DF.
    Esquemas:
      - "pdf":     10% DM + 40% D_agn + 30% N_harm + 15% Combined + 5% P_norm
      - "current": 40% (1-D_agn) + 35% N_harm + 15% DM + 10% P_norm
    Notas:
      • Esta função CLAMPA todas as entradas teóricas 0..1 antes de compor o índice.
      • Dá prioridade à coluna 'Weighted Combined Metric_Norm' se existir.
    """
    out = df.copy()

    def _safe(col: str) -> pd.Series:
        return out[col] if col in out.columns else pd.Series(0.0, index=out.index, dtype=float)

    # ---------- Bases normalizadas ----------
    # N_harm_norm (min-max do count)
    out["N_harm_norm"] = _minmax(_safe("Harmonic Count")).fillna(0.0).clip(0.0, 1.0)

    # Density Metric (preferir _Norm2 se já existir e for numérica; senão min-max de 'Density Metric')
    if "Density Metric_Norm2" in out.columns:
        dm_norm = pd.to_numeric(out["Density Metric_Norm2"], errors="coerce")
    elif "Density Metric_Norm" in out.columns:
        dm_norm = pd.to_numeric(out["Density Metric_Norm"], errors="coerce")
    else:
        dm_norm = _minmax(pd.to_numeric(_safe("Density Metric"), errors="coerce"))
    out["Density Metric_Norm2"] = dm_norm.fillna(0.0).clip(0.0, 1.0)

    # D_agn / P_norm (teoricamente 0..1)
    dagn  = pd.to_numeric(_safe("D_agn"),  errors="coerce").fillna(0.0).clip(0.0, 1.0)
    pnorm = pd.to_numeric(_safe("P_norm"), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Combined (prioridade: Weighted Combined Metric_Norm → Combined Density Metric_Norm2 → min-max de Combined Density Metric)
    if "Weighted Combined Metric_Norm" in out.columns:
        comb_n = pd.to_numeric(out["Weighted Combined Metric_Norm"], errors="coerce")
    elif "Combined Density Metric_Norm2" in out.columns:
        comb_n = pd.to_numeric(out["Combined Density Metric_Norm2"], errors="coerce")
    elif "Combined Density Metric" in out.columns:
        x = pd.to_numeric(out["Combined Density Metric"], errors="coerce")
        lo, hi = x.min(skipna=True), x.max(skipna=True)
        comb_n = ((x - lo) / (hi - lo)) if (pd.notna(lo) and pd.notna(hi) and hi > lo) else pd.Series(0.0, index=out.index, dtype=float)
    else:
        comb_n = pd.Series(0.0, index=out.index, dtype=float)
    comb_n = comb_n.fillna(0.0).clip(0.0, 1.0)

    # ---------- Índice ----------
    sch = (scheme or "").strip().lower()
    if sch == "pdf":
        out["Index_Weighted"] = (
            0.10 * out["Density Metric_Norm2"] +
            0.40 * dagn +
            0.30 * out["N_harm_norm"] +
            0.15 * comb_n +
            0.05 * pnorm
        )
        out["scheme_used"] = "pdf"
    else:
        # Esquema "current" (legado)
        out["D_agn_inv"] = (1.0 - dagn).clip(0.0, 1.0)
        out["Index_Weighted"] = (
            0.40 * out["D_agn_inv"] +
            0.35 * out["N_harm_norm"] +
            0.15 * out["Density Metric_Norm2"] +
            0.10 * pnorm
        )
        out["scheme_used"] = "current"

    # Clamp defensivo no índice (deve já estar em [0,1] por construção)
    out["Index_Weighted"] = pd.to_numeric(out["Index_Weighted"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    return out




def parse_all_sheets(excel_data: pd.ExcelFile) -> Dict[str, pd.DataFrame]:
    """
    Lê todas as planilhas de um arquivo Excel uma única vez e armazena em cache local.

    Args:
        excel_data: Objeto ExcelFile do pandas.

    Returns:
        Dicionário onde as chaves são os nomes das planilhas e os valores são DataFrames.
    """
    return {sheet_name: excel_data.parse(sheet_name) for sheet_name in excel_data.sheet_names}


def extract_dissonance_metrics(dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Extrai métricas de dissonância de todas as planilhas fornecidas.

    Args:
        dfs: Dicionário {nome_da_planilha: DataFrame} com dados do Excel.

    Returns:
        Dicionário {nome_da_coluna: valor_da_métrica} para todas as colunas que contenham 'Dissonance'.
    """
    dissonance_metrics = {}
    for sheet_name, df in dfs.items():
        for column in df.columns:
            if "Dissonance" in column:
                valid = df[column].dropna()
                if not valid.empty:
                    dissonance_metrics[column] = valid.iloc[0]
    return dissonance_metrics


def extract_note_from_quotes(note: str) -> str:
    """
    Extrai o conteúdo entre aspas simples ou duplas em uma string.
    
    Args:
        note: String potencialmente contendo conteúdo entre aspas.
        
    Returns:
        Conteúdo entre aspas, ou a string original se não houver aspas.
    """
    if not note:
        return ""
        
    match = re.search(r"[\"'](.*?)[\"']", note)
    return match.group(1) if match else note


@lru_cache(maxsize=128)
def note_sort_key(note: str) -> Tuple[int, int]:
    """
    Gera uma chave de ordenação para notas musicais baseada em altura e oitava.
    
    Args:
        note: Nome da nota musical (ex: 'C4', 'A#5').
        
    Returns:
        Tupla (oitava, valor_da_nota) para ordenação.
    """
    # Remover aspas da nota primeiro
    note_extracted = extract_note_from_quotes(note)

    # Tentar analisar a nota: letra (A-G), acidental (#/b) opcional, depois um número de oitava
    match = re.match(r"([A-Ga-g])([#b]?)(\d+)", note_extracted)
    if not match:
        logger.warning(f"Formato de nota não reconhecido: {note}")
        return -1, -1  # Nota inválida (irá para o início da ordenação)


    letter = match.group(1).upper()
    accidental = match.group(2)
    octave = int(match.group(3))

    # Mapeamento de nomes de notas (com acidentes) para uma sequência numérica 1..12
    note_order_map = {
        'C': 1, 'C#': 2, 'Db': 2, 'D': 3, 'D#': 4, 'Eb': 4,
        'E': 5, 'F': 6, 'F#': 7, 'Gb': 7, 'G': 8, 'G#': 9,
        'Ab': 9, 'A': 10, 'A#': 11, 'Bb': 11, 'B': 12
    }

    full_note_key = f"{letter}{accidental}"
    note_value = note_order_map.get(full_note_key, 0)  # default para 0 se não encontrado
    
    return octave, note_value


def read_excel_metrics(file_path: Union[str, Path]) -> Dict[str, Optional[float]]:
    """
    Lê métricas de densidade e informações de potência espectral de um arquivo Excel.
    Versão corrigida com melhor tratamento de erros e logging.
    
    Args:
        file_path: Caminho para o arquivo Excel.
        
    Returns:
        Dicionário com métricas extraídas.
        
    Raises:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se o arquivo não puder ser lido como Excel.
    """
    # Inicializar nosso dicionário de retorno com métricas padrão
    metrics = {
        'Density Metric': None,
        'Spectral Density Metric': None,
        'Total Metric': None,
        'Combined Density Metric': None,
        'Spectral Entropy': None,  # Adicionar entropia
        'Filtered Density Metric': None  # Adicionar métrica filtrada
    }

    # Validar existência do arquivo
    file_path = Path(file_path)
    if not file_path.exists():
        logger.error(f"Arquivo não encontrado: {file_path}")
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    logger.info(f"Lendo métricas de: {file_path}")

    try:
        # Carregar dados do Excel
        excel_data = pd.ExcelFile(file_path)
        
        # Log das planilhas disponíveis
        logger.debug(f"Planilhas disponíveis em {file_path.name}: {excel_data.sheet_names}")

        # 1. Tentar ler a planilha 'Metrics' primeiro (prioridade máxima)
        if 'Metrics' in excel_data.sheet_names:
            logger.debug("Lendo planilha 'Metrics'...")
            df_metrics = excel_data.parse('Metrics')
            
            if not df_metrics.empty:
                # Log das colunas encontradas
                logger.debug(f"Colunas em 'Metrics': {list(df_metrics.columns)}")
                
            # Processar TODAS as colunas na planilha de métricas
            for column in df_metrics.columns:
                # Campos textuais: NÃO converter para float
                if column in TEXT_FIELDS:
                    if not df_metrics[column].dropna().empty:
                        txt = df_metrics[column].dropna().iloc[0]
                        metrics[column] = str(txt)
                        logger.debug(f"Campo textual extraído de 'Metrics': {column} = {txt}")
                    else:
                        metrics[column] = ""  # vazio
                    continue

                # Campos numéricos: converter com tolerância
                if not df_metrics[column].dropna().empty:
                    raw = df_metrics[column].dropna().iloc[0]
                    val = pd.to_numeric(raw, errors="coerce")
                    if pd.notna(val):
                        metrics[column] = float(val)
                        logger.debug(f"Métrica extraída de 'Metrics': {column} = {float(val)}")
                    else:
                        logger.warning(f"Valor inválido para métrica '{column}': {raw} - não numérico")
                else:
                    logger.debug(f"Coluna '{column}' está vazia em 'Metrics'; ignorada")
    

        # 2. Verificar planilha 'Spectral Power' para métricas adicionais
        if 'Spectral Power' in excel_data.sheet_names:
            logger.debug("Lendo planilha 'Spectral Power'...")
            df_spectral = excel_data.parse('Spectral Power')
            
            if not df_spectral.empty:
                # Log das colunas encontradas
                logger.debug(f"Colunas em 'Spectral Power': {list(df_spectral.columns)}")
                
                # Procurar por métricas específicas
                metrics_to_find = [
                    'Spectral Density Metric',
                    'Filtered Density Metric',
                    'Total Power (dB)',
                    'Average Power (dB)',
                    'RMS Power (dB)'
                ]
                
                for metric_name in metrics_to_find:
                    if metric_name in df_spectral.columns:
                        valid_metric = df_spectral[metric_name].dropna()
                        if not valid_metric.empty:
                            try:
                                val = float(valid_metric.iloc[0])
                                metrics[metric_name] = val
                                logger.debug(f"Métrica extraída de 'Spectral Power': {metric_name} = {val}")
                            except (ValueError, TypeError) as e:
                                logger.warning(f"Erro ao extrair {metric_name}: {e}")

        # 3. Procurar métricas de dissonância em todas as planilhas
        dissonance_metrics = {}
        for sheet_name in excel_data.sheet_names:
            df = excel_data.parse(sheet_name)
            
            # Procurar colunas que contenham 'Dissonance'
            for column in df.columns:
                if 'Dissonance' in column:
                    valid_metric = df[column].dropna()
                    if not valid_metric.empty:
                        try:
                            val = float(valid_metric.iloc[0])
                            if column not in dissonance_metrics:  # Evitar duplicatas
                                dissonance_metrics[column] = val
                                logger.debug(f"Métrica de dissonância extraída de '{sheet_name}': {column} = {val}")
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Erro ao extrair dissonância {column}: {e}")
        
        # Adicionar métricas de dissonância ao resultado
        metrics.update(dissonance_metrics)
        
        # 4. Verificar se temos pelo menos algumas métricas válidas
        valid_count = sum(1 for v in metrics.values() if v is not None)
        logger.info(f"Total de métricas válidas extraídas de {file_path.name}: {valid_count}")
        
        if valid_count == 0:
            logger.warning(f"AVISO: Nenhuma métrica válida encontrada em {file_path}")
            # Listar as planilhas e suas primeiras linhas para debug
            for sheet in excel_data.sheet_names:
                df = excel_data.parse(sheet)
                if not df.empty:
                    logger.debug(f"Planilha '{sheet}' - primeiras colunas: {list(df.columns)[:5]}")
                    if len(df) > 0:
                        logger.debug(f"Primeira linha: {df.iloc[0].to_dict()}")

    except Exception as e:
        logger.error(f"Erro ao ler '{file_path}': {e}")
        import traceback
        logger.debug(f"Stack trace: {traceback.format_exc()}")
        raise ValueError(f"Erro ao ler arquivo Excel '{file_path}': {e}")

    return metrics

    
def apply_weighted_combination(
    df: pd.DataFrame,
    harmonic_col: str = "Spectral Density Metric",
    inharmonic_col: str = "Filtered Density Metric",
    alpha: float = 0.5,
    beta: float = 0.5,
    weight_function: str = "linear"
) -> pd.DataFrame:
    out = df.copy()
    if "Weighted Combined Metric" in out.columns:
        out = out.drop(columns=["Weighted Combined Metric"])

    if harmonic_col not in out.columns or inharmonic_col not in out.columns:
        logger.warning("Colunas '%s' e/ou '%s' não encontradas.", harmonic_col, inharmonic_col)
        return out

    h  = pd.to_numeric(out[harmonic_col],  errors="coerce").fillna(0.0)
    ih = pd.to_numeric(out[inharmonic_col], errors="coerce").fillna(0.0)

    wf_raw = None
    if 'Weight Function' in out.columns:
        non_empty = out['Weight Function'].dropna()
        wf_raw = str(non_empty.iloc[0]).strip() if not non_empty.empty else None
        out = out.drop(columns=['Weight Function'])

    s = float(alpha) + float(beta)
    if s > 0 and not np.isclose(s, 1.0):
        alpha, beta = float(alpha)/s, float(beta)/s
    elif s <= 0:
        logger.warning("alpha+beta <= 0; usando alpha=beta=0.5")
        alpha, beta = 0.5, 0.5

    key = (weight_function or wf_raw or "linear").strip().lower()
    try:
        _ = get_weight_function(key)
    except Exception as e:
        logger.warning("weight_function inválida '%s' (%s). A usar 'linear'.", key, e)
        key = "linear"

    combined_pre = alpha * h + beta * ih

    if   key == "log":
        combined = np.log1p(np.maximum(combined_pre, 0.0))
    elif key == "sqrt":
        combined = np.sqrt(np.maximum(combined_pre, 0.0))
    elif key in ("square", "squared"):
        combined = np.square(combined_pre)
    elif key == "cbrt":
        combined = np.sign(combined_pre) * np.power(np.abs(combined_pre), 1.0/3.0)
    elif key in ("exp", "exponential"):
        combined = np.expm1(combined_pre)
    elif key == "inverse log":
        eps = 1e-10
        combined = 1.0 / (np.log1p(np.maximum(combined_pre, 0.0)) + eps)
    elif key == "sum":
        combined = combined_pre
    else:
        combined = combined_pre

    out["Weighted Combined Metric"] = combined.astype(float)

    # >>> INSERTAR DAQUI >>>
    # Normalized mirror for downstream consumers (bounded in [0,1])
    wcm = pd.to_numeric(out["Weighted Combined Metric"], errors="coerce")
    lo, hi = wcm.min(skipna=True), wcm.max(skipna=True)
    if pd.notna(lo) and pd.notna(hi) and hi > lo:
        out["Weighted Combined Metric_Norm"] = (wcm - lo) / (hi - lo)
    else:
        out["Weighted Combined Metric_Norm"] = 0.0
    out["Weighted Combined Metric_Norm"] = out["Weighted Combined Metric_Norm"].clip(0.0, 1.0)
    # <<< ATÉ AQUI <<<

    out["WF_used"]  = key if wf_raw is None else (wf_raw if wf_raw else key)
    out["_DBG_WF"]  = key
    out["_DBG_PRE"] = combined_pre.astype(float)


    if key == "linear":
        err = float(np.max(np.abs(out["_DBG_PRE"].values - out["Weighted Combined Metric"].values)))
        if err > 1e-6:
            logger.error("[WCM] linear mismatch: max|pre-post|=%.6g", err)
    return out




def extract_note_from_folder(folder_name: str) -> str:
    """
    Extrai uma nota (ex.: 'C4', 'A#3') do nome da pasta.
    Estratégia: primeiro tenta entre aspas, depois padrão [A-G][#b]?\d.
    """
    if not isinstance(folder_name, str) or not folder_name:
        return folder_name or ""
    # 1) se houver aspas, usa o conteúdo
    q = extract_note_from_quotes(folder_name)
    if q and q != folder_name:
        return q
    # 2) padrão simples
    m = re.search(r"([A-Ga-g])([#b]?)(-?\d+)", folder_name)
    if m:
        return f"{m.group(1).upper()}{m.group(2)}{m.group(3)}"
    return folder_name


def compile_density_metrics(
    folder_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = "compiled_density_metrics.xlsx",
    file_pattern: str = "spectral_analysis.xlsx",
    include_pca: bool = False,
    harmonic_weight: float = 0.5,
    inharmonic_weight: float = 0.5,
    weight_function: str = "linear"
) -> Optional[pd.DataFrame]:
    """
    Compila métricas espectrais de múltiplos ficheiros e salva em Excel.

    Args:
        folder_path: Diretório raiz com os ficheiros.
        output_path: Caminho para salvar o Excel compilado.
        file_pattern: Nome padrão dos ficheiros a procurar.
        include_pca: Se True, aplica PCA.
        harmonic_weight: Peso da componente harmónica.
        inharmonic_weight: Peso da componente inarmónica.
        weight_function: Função de ponderação ('linear', 'log', 'sqrt', 'exp', ...).

    Returns:
        DataFrame compilado ou None se falhar.
    """
    folder_path = Path(folder_path)
    if not folder_path.is_dir():
        logger.error(f"Diretório inválido: {folder_path}")
        return None

    # Recolha dos ficheiros-alvo
    found_files: list[tuple[Path, str, str]] = []
    for root, _, files in os.walk(folder_path):
        root_path = Path(root)
        for fname in files:
            if file_pattern.lower() in fname.lower():
                fpath = root_path / fname
                if fpath.is_file():
                    note = extract_note_from_folder(root_path.name)
                    found_files.append((fpath, note, root_path.name))

    if not found_files:
        logger.warning(f"Nenhum ficheiro encontrado com padrão '{file_pattern}' em {folder_path}")
        return None

    # Ordem determinística por nome de pasta (depois ordenamos por nota)
    found_files.sort(key=lambda t: t[2])

    # Ler métricas de cada ficheiro
    rows = []
    for fpath, note, folder in found_files:
        try:
            metrics = read_excel_metrics(fpath)
        except Exception as exc:
            logger.warning(f"Erro ao ler métricas de {fpath}: {exc}")
            continue

        if not metrics or all(v is None for v in metrics.values()):
            logger.warning(f"Métricas inválidas ou vazias em {fpath}")
            continue

        rows.append({
            "Note": note,
            "Folder": folder,
            **metrics
        })

    if not rows:
        logger.error("Nenhum dado válido encontrado para compilação.")
        return None

    df = pd.DataFrame(rows)

    # Remover colunas duplicadas de nota se existirem e forem redundantes
    for dup in ("Nota", "note", "Pitch"):
        if dup in df.columns and dup != "Note":
            try:
                if df[dup].equals(df.get("Note")):
                    df = df.drop(columns=[dup])
            except Exception:
                # se não for estritamente igual, ignorar (mantém-se a coluna)
                pass

    # Ordenar cromaticamente (nota → MIDI) se possível
    if "Note" in df.columns:
        df["__midi__"] = df["Note"].apply(note_to_midi)
        df = (
            df.sort_values("__midi__", kind="stable")
              .drop(columns="__midi__")
              .reset_index(drop=True)
        )
    else:
        logger.warning("Coluna 'Note' não encontrada; ordem original mantida.")

    # ---------- PONTO CRÍTICO: (re)calcular a Weighted Combined Metric ----------
    # 1) eliminar qualquer WCM herdada dos Excels
    df = df.drop(columns=["Weighted Combined Metric"], errors="ignore")

    # 2) validar/normalizar a chave da função de peso
    wf_key = (weight_function or "linear").strip().lower()
    try:
        _ = get_weight_function(wf_key)
    except Exception as e:
        logger.warning("weight_function inválida '%s' (%s). A usar 'linear'.", wf_key, e)
        wf_key = "linear"

    # 3) aplicar a combinação determinística: H='Spectral Density Metric', IH='Filtered Density Metric'
    df = apply_weighted_combination(
        df,
        harmonic_col="Spectral Density Metric",
        inharmonic_col="Filtered Density Metric",
        alpha=harmonic_weight,
        beta=inharmonic_weight,
        weight_function=wf_key
    )
    # ---------------------------------------------------------------------------

    # PCA (opcional; protegido)
    if include_pca:
        try:
            df = add_pca_to_metrics(df)
            logger.info("PCA aplicado às métricas.")
        except Exception as e:
            logger.error("Falha ao aplicar PCA: %s", e)

    # Índice ponderado e normalizações (determinístico; protegido)
    try:
        df = apply_weighted_index(df)
    except Exception as e:
        logger.error("Falha ao calcular Index_Weighted e normalizações: %s", e)

    # Exportação (opcional; protegida)
    if output_path:
        try:
            outp = Path(output_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(outp, index=False)
            logger.info("Resultados guardados em '%s'", outp)
        except Exception as e:
            logger.error("Erro ao salvar Excel em '%s': %s", outp, e)

    return df




def add_pca_to_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona um componente PCA a um DataFrame de métricas.
    Versão aprimorada com melhor tratamento de erros e memória.
    
    Args:
        df: DataFrame contendo métricas.
        
    Returns:
        DataFrame com componentes PCA adicionados.
        
    Raises:
        ValueError: Se não houver colunas numéricas suficientes para PCA.
    """
    if df is None or df.empty:
        logger.warning("DataFrame vazio fornecido para add_pca_to_metrics")
        return df
        
    # Encontrar colunas numéricas
    numeric_cols = []
    standard_cols = METRIC_COLUMNS.copy()
    
    # Adicionar colunas de dissonância
    # Verificar onde as métricas são extraídas, algo como:
    dissonance_cols = [col for col in df.columns if DISSONANCE_PREFIX in col]
    standard_cols.extend(dissonance_cols)
    
    logger.debug(f"Procurando colunas numéricas entre: {standard_cols}")
    
    for col in standard_cols:
        if col in df.columns:
            try:
                # Verificar se a coluna tem valores numéricos suficientes
                valid_count = pd.to_numeric(df[col], errors='coerce').notnull().sum()
                if valid_count >= 2:
                    numeric_cols.append(col)
                    logger.debug(f"Coluna numérica encontrada: {col} com {valid_count} valores válidos")
            except Exception as e:
                logger.debug(f"Coluna {col} não pôde ser convertida para numérica: {e}")
    
    logger.info(f"Colunas numéricas encontradas para PCA: {numeric_cols}")
    
    if len(numeric_cols) < 2:
        logger.warning("Menos de 2 colunas numéricas encontradas para PCA")
        return df
    
    # Preparar dados para PCA com tratamento robusto de valores ausentes
    try:
        # Criar cópia do DataFrame para evitar avisos
        df_for_pca = df[numeric_cols].copy()
        
        # Converter para numérico, forçando NaN para valores não numéricos
        for col in df_for_pca.columns:
            df_for_pca[col] = pd.to_numeric(df_for_pca[col], errors='coerce')
        
        # Verificar dados após conversão
        if df_for_pca.isnull().sum().sum() > 0:
            logger.warning(f"DataFrame contém {df_for_pca.isnull().sum().sum()} valores NaN após conversão numérica")
        
        # Lidar com valores ausentes
        df_clean = df_for_pca.dropna()
        if df_clean.shape[0] >= 2:
            df_for_pca = df_clean
            logger.debug(f"Usando {df_clean.shape[0]} linhas sem valores ausentes para PCA")
        else:
            # Usar imputação mais robusta para valores ausentes
            # Primeiro tentar a média para cada coluna
            col_means = df_for_pca.mean()
            
            # Verificar se todas as médias são válidas
            if col_means.isnull().sum() > 0:
                logger.warning("Algumas colunas têm apenas NaN. Usando valor padrão para imputação.")
                # Para colunas com apenas NaN, usar 0 como imputação
                col_means = col_means.fillna(0)
                
            df_for_pca = df_for_pca.fillna(col_means)
            logger.debug("Valores ausentes preenchidos com a média para PCA")
            
        # Verificar novamente se há valores ausentes
        if df_for_pca.isnull().sum().sum() > 0:
            logger.error("Ainda há valores ausentes após imputação!")
            # Substituir todos os NaNs restantes com 0
            df_for_pca = df_for_pca.fillna(0)
        
        # Padronizar os dados
        try:
            # Verificar se os dados têm variância não-zero
            zero_var_cols = []
            for col in df_for_pca.columns:
                if df_for_pca[col].std() == 0:
                    zero_var_cols.append(col)
                    
            if zero_var_cols:
                logger.warning(f"Colunas com variância zero detectadas: {zero_var_cols}")
                # Remover colunas com variância zero
                df_for_pca = df_for_pca.drop(columns=zero_var_cols)
                numeric_cols = [col for col in numeric_cols if col not in zero_var_cols]
                
            if len(df_for_pca.columns) < 2:
                logger.error("Insuficientes colunas com variância após filtragem")
                return df
            
            # Agora aplicar StandardScaler
            scaler = StandardScaler()
            metrics_std = scaler.fit_transform(df_for_pca)
            
            # Calcular PCA com 2 componentes para visualização
            pca = PCA(n_components=min(2, len(df_for_pca.columns)))
            pc_results = pca.fit_transform(metrics_std)
            
            # Adicionar resultados ao DataFrame original
            df = df.copy()  # Evitar SettingWithCopyWarning
            df.loc[df_for_pca.index, "PC1"] = pc_results[:, 0]
            if pc_results.shape[1] > 1:
                df.loc[df_for_pca.index, "PC2"] = pc_results[:, 1]
                
            # Registrar variância explicada
            explained_variance = pca.explained_variance_ratio_
            logger.info(f"Variância explicada pelos componentes PCA: {explained_variance}")
            
            # Calcular importância das características
            feature_importance = np.abs(pca.components_)
            for i, component in enumerate(pca.components_):
                sorted_indices = np.argsort(np.abs(component))[::-1]
                logger.info(f"Características mais importantes para PC{i+1}:")
                for idx in sorted_indices[:3]:  # Top 3 características
                    if idx < len(df_for_pca.columns):  # Verificar índice válido
                        logger.info(f"  {df_for_pca.columns[idx]}: {component[idx]:.3f}")
            
            # Normalizar as colunas
            cols_to_normalize = numeric_cols + ["PC1"]
            if "PC2" in df.columns:
                cols_to_normalize.append("PC2")
                
            for col in cols_to_normalize:
                if col in df.columns:
                    valid_values = df[col].dropna()
                    if not valid_values.empty:
                        col_min = valid_values.min()
                        col_max = valid_values.max()
                        if col_max > col_min:
                            df[col + "_Norm"] = (df[col] - col_min) / (col_max - col_min)
                        else:
                            df[col + "_Norm"] = 0
            
        except np.linalg.LinAlgError as lae:
            logger.error(f"Erro de álgebra linear no cálculo de PCA: {lae}")
            logger.info("Isso pode acontecer com dados muito correlacionados. Tentando com menos componentes.")
            return df
        
    except MemoryError as me:
        logger.error(f"Erro de memória durante cálculo de PCA: {me}")
        return df
        
    except Exception as e:
        logger.error(f"Erro durante cálculo de PCA: {e}")
        return df
    
    return df

def compile_density_metrics_with_pca(
    folder_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = "compiled_density_metrics.xlsx",
    file_pattern: str = "spectral_analysis.xlsx",
    include_pca: bool = True,
    harmonic_weight: float = 0.5,
    inharmonic_weight: float = 0.5,
    weight_function: str = "linear",
) -> Optional[pd.DataFrame]:
    """
    Compila métricas de densidade, (opcionalmente) adiciona PCA e grava Excel.
    Também acrescenta:
      - N_harm_norm = min–max de 'Harmonic Count'
      - normalizações auxiliares: Density Metric_Norm2, Combined Density Metric_Norm2
      - Index_Weighted = 0.10*DM_Norm2 + 0.40*D_agn + 0.30*N_harm_norm + 0.15*Combined_Norm2 + 0.05*P_norm
    """
    # 1) Compilação base (sem PCA interno)
    df = compile_density_metrics(
        folder_path=folder_path,
        output_path=None,             # só exportamos no fim deste wrapper
        file_pattern=file_pattern,
        include_pca=False,            # PCA será aplicado já de seguida (se pedido)
        harmonic_weight=harmonic_weight,
        inharmonic_weight=inharmonic_weight,
        weight_function=weight_function,
    )
    if df is None or df.empty:
        return None

    # 2) PCA opcional
    if include_pca:
        try:
            df = add_pca_to_metrics(df)
        except Exception as e:
            logger.error("Falha ao aplicar PCA: %s", e)

    # 3) Normalizações determinísticas (min–max) e índice composto
    #    – só criamos se as colunas existirem; faltas viram 0.0 para não bloquear.
    def _safe(series_name: str) -> pd.Series:
        return df[series_name] if series_name in df.columns else pd.Series(0.0, index=df.index, dtype=float)

    # N_harm_norm (a partir de Harmonic Count)
    if "Harmonic Count" in df.columns:
        df["N_harm_norm"] = _minmax(df["Harmonic Count"])
    else:
        df["N_harm_norm"] = 0.0
        logger.warning("Coluna 'Harmonic Count' ausente; N_harm_norm definido a 0.")

    # Normalizações auxiliares (usamos sufixo _Norm2 para não colidir com _Norm existentes)
    if "Density Metric" in df.columns:
        df["Density Metric_Norm2"] = _minmax(df["Density Metric"])
    else:
        df["Density Metric_Norm2"] = 0.0
        logger.warning("Coluna 'Density Metric' ausente; Density Metric_Norm2 definido a 0.")

    if "Combined Density Metric" in df.columns:
        df["Combined Density Metric_Norm2"] = _minmax(df["Combined Density Metric"])
    else:
        df["Combined Density Metric_Norm2"] = 0.0
        logger.warning("Coluna 'Combined Density Metric' ausente; Combined Density Metric_Norm2 definido a 0.")

    # D_agn e P_norm já estão em [0,1] pelo pipeline
    d_agn = _safe("D_agn").fillna(0.0)
    p_norm = _safe("P_norm").fillna(0.0)

    # 4) Índice ponderado (0..1) — pesos acordados: 10% + 40% + 30% + 15% + 5%
    df["Index_Weighted"] = (
        0.10 * df["Density Metric_Norm2"].fillna(0.0) +
        0.40 * d_agn +
        0.30 * df["N_harm_norm"].fillna(0.0) +
        0.15 * df["Combined Density Metric_Norm2"].fillna(0.0) +
        0.05 * p_norm
    )

    # 5) Exportação
    if output_path:
        try:
            outp = Path(output_path)
            outp.parent.mkdir(parents=True, exist_ok=True)
            df.to_excel(outp, index=False)
            logger.info("Resultados (com PCA) guardados em '%s'", outp)
        except Exception as e:
            logger.error("Erro ao salvar Excel (PCA): %s", e)

    return df


def plot_pca_scatter(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Cria um gráfico de dispersão 2D usando PC1 e PC2.
    
    Args:
        df: DataFrame com colunas PC1 e PC2.
        output_dir: Diretório para salvar o gráfico.
    """
    if "PC1" not in df.columns or "PC2" not in df.columns:
        logger.warning("PC1 ou PC2 não encontrados no DataFrame")
        return
        
    plt.figure(figsize=(10, 8))
    plt.scatter(df["PC1"], df["PC2"], s=100, alpha=0.7)
    
    # Adicionar rótulos de notas
    if "Note" in df.columns:
        for i, row in df.iterrows():
            plt.annotate(row["Note"], 
                         (row["PC1"], row["PC2"]),
                         xytext=(5, 5),
                         textcoords="offset points")
    
    plt.title("PCA Analysis of Spectral Metrics")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(True, alpha=0.3)
    
    # Adicionar linhas de referência
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    output_path = Path(output_dir) / "pca_scatter.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de dispersão PCA salvo em: {output_path}")


def plot_pc1_ranking(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Cria um gráfico de barras das notas ordenadas por PC1.
    
    Args:
        df: DataFrame com coluna PC1.
        output_dir: Diretório para salvar o gráfico.
    """
    if "PC1" not in df.columns:
        logger.warning("PC1 não encontrado no DataFrame")
        return
        
    # Ordenar por PC1
    df_sorted = df.sort_values(by="PC1").copy()
    
    plt.figure(figsize=(12, 6))
    
    # Obter rótulos para o eixo x
    x_labels = df_sorted["Note"].tolist() if "Note" in df_sorted.columns else [f"Item {i+1}" for i in range(len(df_sorted))]
    
    # Criar gráfico de barras
    bars = plt.bar(x_labels, df_sorted["PC1"], alpha=0.7)
    
    # Colorir barras por valor
    min_val = df_sorted["PC1"].min()
    max_val = df_sorted["PC1"].max()
    norm = plt.Normalize(min_val, max_val)
    colors = plt.cm.viridis(norm(df_sorted["PC1"]))
    
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.title("Notes Ranked by Principal Component 1")
    plt.xlabel("Note")
    plt.ylabel("PC1 Value")
    plt.xticks(rotation=45, ha="right")
    plt.grid(True, alpha=0.3)
    
    output_path = Path(output_dir) / "pc1_ranking.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Gráfico de ranking PC1 salvo em: {output_path}")


def plot_correlation_matrix(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Cria uma matriz de correlação para as métricas numéricas.
    
    Args:
        df: DataFrame com métricas.
        output_dir: Diretório para salvar o gráfico.
    """
    # Identificar colunas numéricas (excluindo colunas normalizadas para evitar duplicação)
    numeric_cols = []
    for col in df.columns:
        if col.endswith("_Norm"):
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                numeric_cols.append(col)
        except:
            pass
    
    if len(numeric_cols) < 2:
        logger.warning("Menos de 2 colunas numéricas encontradas para matriz de correlação")
        return
    
    # Calcular matriz de correlação
    corr_df = df[numeric_cols].corr()
    
    # Plotar usando seaborn
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))  # Máscara para triângulo superior
    
    # Usar um mapa de cores divergente para melhor visualização
    sns.heatmap(corr_df, mask=mask, cmap="coolwarm", vmin=-1, vmax=1, 
                annot=True, fmt=".2f", linewidths=0.5, square=True)
    
    plt.title("Correlation Matrix of Spectral Metrics")
    
    output_path = Path(output_dir) / "correlation_matrix.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Matriz de correlação salva em: {output_path}")


def plot_metrics_comparison(df: pd.DataFrame, output_dir: Union[str, Path]) -> None:
    """
    Cria um gráfico de barras comparando diferentes métricas para cada nota.
    
    Args:
        df: DataFrame com métricas e notas.
        output_dir: Diretório para salvar o gráfico.
    """
    # Verificar se temos a coluna 'Note'
    if 'Note' not in df.columns:
        logger.warning("Coluna 'Note' não encontrada para gráfico de comparação")
        return
    
    # Identificar métricas (excluindo colunas normalizadas e PCs)
    metrics = []
    for col in df.columns:
        if col in ['Note', 'Folder'] or col.startswith('PC') or col.endswith('_Norm'):
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                metrics.append(col)
        except:
            pass
    
    if not metrics:
        logger.warning("Nenhuma métrica válida encontrada para gráfico de comparação")
        return
    
    # Normalizar métricas para escala 0-1
    df_norm = df.copy()
    for col in metrics:
        values = df_norm[col].dropna()
        if len(values) < 2:
            continue
        min_val = values.min()
        max_val = values.max()
        if max_val > min_val:
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val)
    
    # Criar um gráfico para cada tipo principal de métrica
    metric_groups = {
        'Density': [m for m in metrics if 'Density' in m],
        'Dissonance': [m for m in metrics if 'Dissonance' in m]
    }
    
    for group_name, group_metrics in metric_groups.items():
        if not group_metrics:
            continue
            
        plt.figure(figsize=(14, 8))
        
        # Preparar dados para plot
        x = np.arange(len(df_norm))
        width = 0.8 / len(group_metrics)  # Largura da barra
        
        # Plotar barras para cada métrica
        for i, metric in enumerate(group_metrics):
            offset = (i - len(group_metrics) / 2 + 0.5) * width
            plt.bar(x + offset, df_norm[metric], width, label=metric)
        
        # Configurar eixos e rótulos
        plt.xlabel('Note')
        plt.ylabel('Normalized Value')
        plt.title(f'Comparison of {group_name} Metrics')
        plt.xticks(x, df_norm['Note'], rotation=45, ha='right')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = Path(output_dir) / f"{group_name.lower()}_comparison.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico de comparação de {group_name} salvo em: {output_path}")


def analyze_notes_clustering(df: pd.DataFrame, output_dir: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Analisa o agrupamento de notas com base nas métricas de densidade/dissonância.
    
    Args:
        df: DataFrame com métricas e notas.
        output_dir: Diretório para salvar gráficos (opcional).
        
    Returns:
        Dicionário com resultados da análise de agrupamento.
    """
    if df is None or df.empty or 'Note' not in df.columns:
        logger.warning("DataFrame inválido para análise de agrupamento")
        return {}
    
    # Identificar métricas numéricas
    numeric_cols = []
    for col in df.columns:
        if col in ['Note', 'Folder'] or col.startswith('PC') or col.endswith('_Norm'):
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                numeric_cols.append(col)
        except:
            pass
    
    if len(numeric_cols) < 2:
        logger.warning("Métricas insuficientes para análise de agrupamento")
        return {}
    
    try:
        # Preparar dados numéricos
        X = df[numeric_cols].fillna(df[numeric_cols].mean()).values
        
        # Padronizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Calcular distâncias euclidianas entre notas
        distances = euclidean_distances(X_scaled)
        
        # Criar DataFrame de distâncias com notas como índices
        dist_df = pd.DataFrame(distances, 
                               index=df['Note'].values,
                               columns=df['Note'].values)
        
        # Encontrar notas mais próximas para cada nota
        closest_notes = {}
        for note in dist_df.index:
            # Ordenar por distância (excluindo a própria nota)
            closest = dist_df[note].sort_values()[1:4]  # 3 mais próximas
            closest_notes[note] = {
                'closest': closest.index.tolist(),
                'distances': closest.values.tolist()
            }
        
        # Gerar um gráfico de calor das distâncias
        if output_dir:
            plt.figure(figsize=(10, 8))
            sns.heatmap(dist_df, cmap='viridis_r', annot=True, fmt='.2f', square=True)
            plt.title('Euclidean Distances Between Notes')
            
            output_path = Path(output_dir) / "note_distances.png"
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Mapa de distâncias entre notas salvo em: {output_path}")
        
        return {
            'distance_matrix': dist_df,
            'closest_notes': closest_notes
        }
        
    except Exception as e:
        logger.error(f"Erro na análise de agrupamento: {e}")
        return {}


def generate_analysis_report(df: pd.DataFrame, output_path: Union[str, Path] = 'analysis_report.md') -> None:
    """
    Gera um relatório de análise em formato Markdown com insights sobre as métricas.
    
    Args:
        df: DataFrame com métricas compiladas.
        output_path: Caminho para salvar o relatório.
    """
    if df is None or df.empty:
        logger.warning("DataFrame vazio para geração de relatório")
        return
    
    try:
        # Iniciar relatório
        report_lines = [
            "# Análise de Métricas Espectrais\n",
            f"Data da análise: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"Total de notas analisadas: {len(df)}\n",
            "\n## Resumo das Métricas\n"
        ]
        
        # Identificar métricas numéricas para análise
        numeric_cols = []
        for col in df.columns:
            if col in ['Note', 'Folder']:
                continue
            try:
                if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                    numeric_cols.append(col)
            except:
                pass
        
        # Estatísticas descritivas
        if numeric_cols:
            desc_stats = df[numeric_cols].describe().transpose()
            report_lines.append("### Estatísticas Descritivas\n")
            report_lines.append("| Métrica | Média | Desvio Padrão | Mín | Máx |\n")
            report_lines.append("| ------- | ----- | ------------- | --- | --- |\n")
            
            for idx, row in desc_stats.iterrows():
                report_lines.append(f"| {idx} | {row['mean']:.3f} | {row['std']:.3f} | {row['min']:.3f} | {row['max']:.3f} |\n")
            
            report_lines.append("\n")
            
        # PCA insights
        if 'PC1' in df.columns:
            report_lines.append("## Análise de Componentes Principais\n")
            report_lines.append("### Ranking de Notas por PC1\n")
            
            # Ordenar notas por PC1
            if 'Note' in df.columns:
                df_sorted = df.sort_values(by='PC1', ascending=False).copy()
                
                report_lines.append("Notas ordenadas do maior para o menor valor de PC1:\n\n")
                report_lines.append("| Posição | Nota | PC1 |\n")
                report_lines.append("| ------- | ---- | --- |\n")
                
                for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
                    report_lines.append(f"| {i} | {row['Note']} | {row['PC1']:.3f} |\n")
                
                report_lines.append("\n")
            
            # Informações sobre contribuição de métricas para PC1
            report_lines.append("### Interpretação de PC1\n")
            report_lines.append("O primeiro componente principal (PC1) pode ser interpretado como ")
            
            # Aqui poderíamos inserir uma análise de contribuição das métricas para o PC1
            # Como não temos esses dados no dataframe compilado, vamos adicionar uma nota genérica
            report_lines.append("uma medida composta da densidade/dissonância espectral. ")
            report_lines.append("Valores mais altos geralmente indicam maior densidade harmônica e/ou dissonância.\n\n")
        
        # Correlações entre métricas
        if len(numeric_cols) >= 2:
            report_lines.append("## Correlações entre Métricas\n")
            
            # Calcular matriz de correlação
            corr_matrix = df[numeric_cols].corr()
            
            # Encontrar correlações fortes (acima de 0.7 ou abaixo de -0.7)
            strong_corr = []
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr = corr_matrix.iloc[i, j]
                    if abs(corr) >= 0.7:
                        strong_corr.append((numeric_cols[i], numeric_cols[j], corr))
            
            if strong_corr:
                report_lines.append("### Correlações Fortes (|r| ≥ 0.7)\n")
                report_lines.append("| Métrica 1 | Métrica 2 | Correlação |\n")
                report_lines.append("| -------- | -------- | ---------- |\n")
                
                for m1, m2, corr in sorted(strong_corr, key=lambda x: abs(x[2]), reverse=True):
                    report_lines.append(f"| {m1} | {m2} | {corr:.3f} |\n")
                    
                report_lines.append("\n")
            else:
                report_lines.append("Não foram encontradas correlações fortes entre as métricas.\n\n")
        
        # Análise de Agrupamento de Notas
        if 'Note' in df.columns and len(df) >= 3:
            report_lines.append("## Agrupamento de Notas\n")
            
            # Realizar análise de agrupamento
            clustering = analyze_notes_clustering(df)
            
            if clustering and 'closest_notes' in clustering:
                report_lines.append("### Notas Similares\n")
                report_lines.append("Baseado nas métricas espectrais, as seguintes notas são mais similares entre si:\n\n")
                report_lines.append("| Nota | Notas Mais Similares |\n")
                report_lines.append("| ---- | ------------------- |\n")
                
                for note, data in clustering['closest_notes'].items():
                    similar_notes = data['closest']
                    distances = data['distances']
                    
                    # Formatar notas similares com suas distâncias
                    similar_str = ", ".join([f"{n} ({d:.2f})" for n, d in zip(similar_notes, distances)])
                    report_lines.append(f"| {note} | {similar_str} |\n")
                
                report_lines.append("\n")
        
        # Insights específicos para métricas de densidade/dissonância
        report_lines.append("## Insights Específicos\n")
        
        # Verificar se temos métricas de densidade
        density_metrics = [col for col in numeric_cols if 'Density' in col]
        if density_metrics:
            report_lines.append("### Métricas de Densidade\n")
            
            # Encontrar notas com maior e menor densidade para cada métrica
            for metric in density_metrics:
                if metric in df.columns and 'Note' in df.columns:
                    valid_values = df[[metric, 'Note']].dropna()
                    if not valid_values.empty:
                        max_row = valid_values.loc[valid_values[metric].idxmax()]
                        min_row = valid_values.loc[valid_values[metric].idxmin()]
                        
                        report_lines.append(f"**{metric}**:\n")
                        report_lines.append(f"- Nota com maior valor: {max_row['Note']} ({max_row[metric]:.3f})\n")
                        report_lines.append(f"- Nota com menor valor: {min_row['Note']} ({min_row[metric]:.3f})\n")
            
            report_lines.append("\n")
        
        # Verificar se temos métricas de dissonância
        dissonance_metrics = [col for col in numeric_cols if 'Dissonance' in col]
        if dissonance_metrics:
            report_lines.append("### Métricas de Dissonância\n")
            
            # Encontrar notas com maior e menor dissonância para cada métrica
            for metric in dissonance_metrics:
                if metric in df.columns and 'Note' in df.columns:
                    valid_values = df[[metric, 'Note']].dropna()
                    if not valid_values.empty:
                        max_row = valid_values.loc[valid_values[metric].idxmax()]
                        min_row = valid_values.loc[valid_values[metric].idxmin()]
                        
                        report_lines.append(f"**{metric}**:\n")
                        report_lines.append(f"- Nota com maior valor: {max_row['Note']} ({max_row[metric]:.3f})\n")
                        report_lines.append(f"- Nota com menor valor: {min_row['Note']} ({min_row[metric]:.3f})\n")
            
            report_lines.append("\n")
        
        # Conclusões
        report_lines.append("## Conclusões\n")
        report_lines.append("Com base na análise das métricas espectrais, podemos concluir que:\n\n")
        
        # Adicionar algumas conclusões genéricas
        report_lines.append("1. As métricas de densidade e dissonância fornecem diferentes perspectivas sobre o conteúdo espectral das notas.\n")
        report_lines.append("2. A análise PCA permite reduzir a dimensionalidade e visualizar tendências que não são imediatamente aparentes.\n")
        report_lines.append("3. As diferenças entre notas são quantificáveis através destas métricas, o que pode ser útil para estudos de percepção musical.\n")
        
        # Salvar o relatório
        output_path = Path(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
            
        logger.info(f"Relatório de análise salvo em: {output_path}")
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório de análise: {e}")


def extract_models_comparison(folder_path: Union[str, Path], 
                             output_path: Optional[Union[str, Path]] = "compiled_density_metrics.xlsx") -> Optional[pd.DataFrame]:
    """
    Extrai e compila uma comparação entre diferentes modelos de dissonância.
    
    Args:
        folder_path: Diretório contendo subpastas com arquivos de análise espectral.
        output_path: Caminho para salvar o arquivo Excel compilado.
        
    Returns:
        DataFrame com comparação de modelos, ou None se nenhum dado válido for encontrado.
    """
    # Validar caminho do diretório
    folder_path = Path(folder_path)
    if not folder_path.exists() or not folder_path.is_dir():
        logger.error(f"Diretório inválido: {folder_path}")
        raise ValueError(f"Diretório inválido: {folder_path}")
    
    results = []
    
    logger.info(f"Extraindo comparação de modelos de dissonância de: {folder_path}")
    
    # Percorrer todas as subpastas
    for note_dir in [d for d in folder_path.iterdir() if d.is_dir()]:
        try:
            # Extrair nome da nota da pasta
            note = extract_note_from_folder(note_dir.name)
            
            # Verificar se existe arquivo de análise espectral
            excel_path = note_dir / 'spectral_analysis.xlsx'
            if not excel_path.exists():
                logger.debug(f"Arquivo de análise não encontrado para nota {note}")
                continue
            
            # Extrair métricas de dissonância
            metrics = read_excel_metrics(excel_path)
            dissonance_metrics = {k: v for k, v in metrics.items() if 'Dissonance' in k}
            
            if not dissonance_metrics:
                logger.debug(f"Nenhuma métrica de dissonância encontrada para nota {note}")
                continue
            
            # Adicionar resultados
            results.append({
                'Note': note,
                **dissonance_metrics
            })
            
            logger.debug(f"Métricas de dissonância extraídas para nota {note}: {list(dissonance_metrics.keys())}")
            
        except Exception as e:
            logger.error(f"Erro processando pasta {note_dir}: {e}")
    
    if not results:
        logger.warning("Nenhum dado válido encontrado para comparação de modelos.")
        return None
    
    # Construir DataFrame e ordenar por nota
    results_df = pd.DataFrame(results)
    
    if 'Note' in results_df.columns:
        try:
            results_df = results_df.sort_values(
                by='Note',
                key=lambda col: col.map(note_sort_key)
            )
        except Exception as e:
            logger.warning(f"Não foi possível ordenar por nota: {e}")
    
    # Salvar para Excel
    try:
        output_path = Path(output_path)
        results_df.to_excel(output_path, index=False)
        logger.info(f"Comparação de modelos salva em '{output_path}'.")
        
        # Adicionalmente, gerar um heatmap de correlação entre modelos
        try:
            dissonance_models = [col for col in results_df.columns if 'Dissonance' in col]
            if len(dissonance_models) >= 2:
                corr_matrix = results_df[dissonance_models].corr()
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
                plt.title('Correlation Between Dissonance Models')
                
                heatmap_path = output_path.with_suffix('.png')
                plt.tight_layout()
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Heatmap de correlação entre modelos salvo em: {heatmap_path}")
                
        except Exception as e:
            logger.error(f"Erro ao gerar heatmap de correlação: {e}")
        
    except Exception as e:
        logger.error(f"Falha ao salvar comparação de modelos em '{output_path}': {e}")
        raise
    
    return results_df


def calculate_metric_distributions(df: pd.DataFrame, num_bins: int = 10) -> Dict[str, Dict[str, Any]]:
    """
    Calcula distribuições estatísticas para cada métrica.
    
    Args:
        df: DataFrame com métricas.
        num_bins: Número de bins para histogramas.
        
    Returns:
        Dicionário com estatísticas de distribuição para cada métrica.
    """
    if df is None or df.empty:
        logger.warning("DataFrame vazio para cálculo de distribuições")
        return {}
    
    distributions = {}
    
    # Identificar métricas numéricas
    numeric_cols = []
    for col in df.columns:
        if col in ['Note', 'Folder']:
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                numeric_cols.append(col)
        except:
            pass
    
    # Calcular distribuições
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) < 2:
            continue
            
        # Estatísticas básicas
        stats = {
            'mean': values.mean(),
            'median': values.median(),
            'std': values.std(),
            'min': values.min(),
            'max': values.max(),
            'skew': values.skew(),  # Assimetria
            'kurtosis': values.kurtosis()  # Curtose
        }
        
        # Calcular histograma
        hist, bin_edges = np.histogram(values, bins=num_bins)
        
        # Adicionar distribuição
        distributions[col] = {
            'stats': stats,
            'histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            }
        }
    
    return distributions


def plot_metric_distributions(df: pd.DataFrame, 
                             output_dir: Union[str, Path],
                             num_bins: int = 10) -> None:
    """
    Plota distribuições para cada métrica.
    
    Args:
        df: DataFrame com métricas.
        output_dir: Diretório para salvar os gráficos.
        num_bins: Número de bins para histogramas.
    """
    if df is None or df.empty:
        logger.warning("DataFrame vazio para plotagem de distribuições")
        return
    
    # Garantir que o diretório existe
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Identificar métricas numéricas
    numeric_cols = []
    for col in df.columns:
        if col in ['Note', 'Folder']:
            continue
        try:
            if pd.to_numeric(df[col], errors='coerce').notnull().sum() >= 2:
                numeric_cols.append(col)
        except:
            pass
    
    for col in numeric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        if len(values) < 2:
            continue
            
        # Criar figura
        plt.figure(figsize=(10, 6))
        
        # Plotar histograma com curva de densidade
        sns.histplot(values, kde=True, bins=num_bins)
        
        # Adicionar linha vertical para média e mediana
        plt.axvline(values.mean(), color='r', linestyle='--', alpha=0.7, label=f'Média: {values.mean():.3f}')
        plt.axvline(values.median(), color='g', linestyle='-.', alpha=0.7, label=f'Mediana: {values.median():.3f}')
        
        # Configurar rótulos e título
        plt.title(f'Distribuição de {col}')
        plt.xlabel(col)
        plt.ylabel('Frequência')
        plt.legend()
        
        # Informações estatísticas no gráfico
        stats_text = (
            f"Desvio Padrão: {values.std():.3f}\n"
            f"Mín: {values.min():.3f}\n"
            f"Máx: {values.max():.3f}\n"
            f"Assimetria: {values.skew():.3f}\n"
            f"Curtose: {values.kurtosis():.3f}"
        )
        
        # Posicionar texto no canto superior direito
        plt.annotate(stats_text, xy=(0.95, 0.95), xycoords='axes fraction',
                    fontsize=9, ha='right', va='top',
                    bbox=dict(boxstyle='round', fc='white', alpha=0.7))
        
        # Salvar figura
        output_path = output_dir / f"{col.lower().replace(' ', '_')}_distribution.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráfico de distribuição para {col} salvo em: {output_path}")


def generate_comparison_report(results_df: pd.DataFrame, 
                              output_path: Union[str, Path] = 'comparison_report.md',
                              include_plots: bool = True,
                              plots_dir: Optional[Union[str, Path]] = None) -> None:
    """
    Gera um relatório comparativo entre diferentes modelos de dissonância e métricas.
    
    Args:
        results_df: DataFrame com métricas de todas as notas.
        output_path: Caminho para salvar o relatório.
        include_plots: Se True, gera gráficos para inclusão no relatório.
        plots_dir: Diretório para salvar os gráficos (se None, usa o mesmo do output_path).
    """
    if results_df is None or results_df.empty:
        logger.warning("DataFrame vazio para geração de relatório comparativo")
        return
    
    try:
        # Preparar diretório para plots
        output_path = Path(output_path)
        
        if plots_dir is None:
            plots_dir = output_path.parent / 'plots'
        else:
            plots_dir = Path(plots_dir)
            
        plots_dir.mkdir(exist_ok=True, parents=True)
        
        # Iniciar relatório
        report_lines = [
            "# Relatório Comparativo de Métricas Espectrais\n",
            f"Data da análise: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n",
            f"Total de notas analisadas: {len(results_df)}\n",
            "\n## Visão Geral\n"
        ]
        
        # Contar tipos de métricas
        density_metrics = [col for col in results_df.columns if 'Density' in col]
        dissonance_metrics = [col for col in results_df.columns if 'Dissonance' in col]
        
        report_lines.append(f"- Total de métricas de densidade: {len(density_metrics)}\n")
        report_lines.append(f"- Total de modelos de dissonância: {len(dissonance_metrics)}\n\n")
        
        # Lista de métricas
        if density_metrics:
            report_lines.append("### Métricas de Densidade\n")
            for metric in density_metrics:
                report_lines.append(f"- {metric}\n")
            report_lines.append("\n")
            
        if dissonance_metrics:
            report_lines.append("### Modelos de Dissonância\n")
            for metric in dissonance_metrics:
                report_lines.append(f"- {metric}\n")
            report_lines.append("\n")
        
        # Análise de correlação
        report_lines.append("## Correlação entre Modelos\n")
        
        # Calcular matriz de correlação para modelos de dissonância
        if len(dissonance_metrics) >= 2:
            corr_matrix = results_df[dissonance_metrics].corr()
            
            report_lines.append("### Matriz de Correlação\n")
            report_lines.append("| Modelo | " + " | ".join(dissonance_metrics) + " |\n")
            report_lines.append("| ------ | " + " | ".join(["-----" for _ in dissonance_metrics]) + " |\n")
            
            for model in dissonance_metrics:
                row = [model]
                for other_model in dissonance_metrics:
                    row.append(f"{corr_matrix.loc[model, other_model]:.3f}")
                report_lines.append("| " + " | ".join(row) + " |\n")
            
            report_lines.append("\n")
            
            # Adicionar gráfico de correlação se solicitado
            if include_plots:
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.3f')
                plt.title('Correlation Between Dissonance Models')
                
                corr_path = plots_dir / "dissonance_correlation.png"
                plt.tight_layout()
                plt.savefig(corr_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Adicionar referência ao gráfico no relatório
                report_lines.append(f"![Correlation Between Dissonance Models]({corr_path.name})\n\n")
                logger.info(f"Gráfico de correlação salvo em: {corr_path}")
            
            # Interpretação das correlações
            report_lines.append("### Interpretação\n")
            
            # Encontrar correlações fortes
            strong_positive = []
            strong_negative = []
            weak = []
            
            for i in range(len(dissonance_metrics)):
                for j in range(i+1, len(dissonance_metrics)):
                    model1 = dissonance_metrics[i]
                    model2 = dissonance_metrics[j]
                    corr = corr_matrix.loc[model1, model2]
                    
                    if corr >= 0.7:
                        strong_positive.append((model1, model2, corr))
                    elif corr <= -0.7:
                        strong_negative.append((model1, model2, corr))
                    elif abs(corr) <= 0.3:
                        weak.append((model1, model2, corr))
            
            if strong_positive:
                report_lines.append("**Correlações Positivas Fortes (r ≥ 0.7):**\n")
                for m1, m2, corr in strong_positive:
                    report_lines.append(f"- {m1} e {m2}: {corr:.3f}\n")
                report_lines.append("\n")
                
            if strong_negative:
                report_lines.append("**Correlações Negativas Fortes (r ≤ -0.7):**\n")
                for m1, m2, corr in strong_negative:
                    report_lines.append(f"- {m1} e {m2}: {corr:.3f}\n")
                report_lines.append("\n")
                
            if weak:
                report_lines.append("**Correlações Fracas (|r| ≤ 0.3):**\n")
                for m1, m2, corr in weak:
                    report_lines.append(f"- {m1} e {m2}: {corr:.3f}\n")
                report_lines.append("\n")
        
        # Análise de distribuição
        report_lines.append("## Distribuição das Métricas\n")
        
        # Calcular distribuições e criar gráficos para métricas de dissonância
        if include_plots and dissonance_metrics:
            # Plotar distribuições
            plot_metric_distributions(results_df[dissonance_metrics], plots_dir)
            
            # Adicionar informações ao relatório
            for metric in dissonance_metrics:
                values = results_df[metric].dropna()
                if len(values) < 2:
                    continue
                    
                report_lines.append(f"### {metric}\n")
                report_lines.append(f"- **Média**: {values.mean():.3f}\n")
                report_lines.append(f"- **Mediana**: {values.median():.3f}\n")
                report_lines.append(f"- **Desvio Padrão**: {values.std():.3f}\n")
                report_lines.append(f"- **Mínimo**: {values.min():.3f}\n")
                report_lines.append(f"- **Máximo**: {values.max():.3f}\n")
                
                # Adicionar referência ao gráfico no relatório
                dist_path = plots_dir / f"{metric.lower().replace(' ', '_')}_distribution.png"
                if dist_path.exists():
                    report_lines.append(f"\n![Distribution of {metric}]({dist_path.name})\n\n")
        
        # Conclusões e recomendações
        report_lines.append("## Conclusões e Recomendações\n")
        
        report_lines.append("Com base na análise comparativa, podemos concluir que:\n\n")
        
        # Adicionar algumas conclusões baseadas nos resultados
        if len(dissonance_metrics) >= 2:
            # Verificar concordância entre modelos
            corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            mean_corr = np.mean(corr_values)
            
            if mean_corr >= 0.7:
                report_lines.append("1. Há uma forte concordância entre a maioria dos modelos de dissonância, ")
                report_lines.append("sugerindo que eles estão capturando aspectos semelhantes da percepção de dissonância.\n")
            elif mean_corr >= 0.4:
                report_lines.append("1. Há uma concordância moderada entre os modelos de dissonância, ")
                report_lines.append("com algumas diferenças em como cada modelo avalia determinadas notas.\n")
            else:
                report_lines.append("1. Existe uma baixa concordância entre os modelos de dissonância, ")
                report_lines.append("sugerindo que diferentes modelos capturam aspectos distintos da percepção de dissonância.\n")
        
        report_lines.append("2. Para análises futuras, recomendamos:\n")
        report_lines.append("   - Padronizar a escala de todas as métricas para facilitar comparações diretas\n")
        report_lines.append("   - Considerar o uso de análise multivariada para explorar relações mais complexas\n")
        report_lines.append("   - Comparar estes resultados com testes de percepção auditiva para validação\n\n")
        
        # Modelo recomendado
        if len(dissonance_metrics) >= 2:
            report_lines.append("3. Com base nesta análise, o modelo mais recomendado para uso geral seria aquele que:")
            report_lines.append("   - Tem boa correlação com a maioria dos outros modelos\n")
            report_lines.append("   - Apresenta uma distribuição bem comportada\n")
            report_lines.append("   - É computacionalmente eficiente\n\n")
        
        # Salvar o relatório
        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(report_lines)
            
        logger.info(f"Relatório comparativo salvo em: {output_path}")
        
    except Exception as e:
        logger.error(f"Erro ao gerar relatório comparativo: {e}")



def extract_density_metric(audio_processor: AudioProcessor) -> Optional[float]:
    """
    Extrai a métrica de densidade combinada do processador de áudio já configurado.

    Args:
        audio_processor: Instância configurada do AudioProcessor.

    Returns:
        Valor da métrica combinada de densidade.
    """
    try:
        return audio_processor.combined_density_metric_value
    except Exception as e:
        logger.error(f"Erro ao extrair métrica de densidade: {e}")
        return None

if __name__ == "__main__":
    # Exemplo de uso
    logger.info("Iniciando exemplo de uso do módulo compile_metrics")
    
    example_folder = './results'
    output_excel = './compiled_density_metrics.xlsx'
    
    try:
        if os.path.exists(example_folder):
            # Compilar métricas com PCA e gerar relatório
            results_df = compile_density_metrics_with_pca(
                folder_path=example_folder,
                output_path=output_excel,
                generate_plots=True
            )
            
            if results_df is not None:
                # Gerar relatório de análise
                generate_analysis_report(
                    df=results_df,
                    output_path='./analysis_report.md'
                )
                
                # Extrair comparação de modelos
                models_df = extract_models_comparison(
                    folder_path=example_folder,
                    output_path='./models_comparison.xlsx'
                )
                
                if models_df is not None:
                    # Gerar relatório comparativo
                    generate_comparison_report(
                        results_df=models_df,
                        output_path='./comparison_report.md',
                        include_plots=True
                    )
                
                logger.info("Exemplo de uso concluído com sucesso!")
            else:
                logger.warning("Nenhum resultado compilado para análise")
        else:
            logger.error(f"Diretório de exemplo '{example_folder}' não encontrado")
            
    except Exception as e:
        logger.error(f"Erro no exemplo de uso: {e}")


def test_compile_metrics(folder_path: Union[str, Path]) -> None:
    """
    Função de teste para diagnosticar problemas com a compilação de métricas.
    
    Args:
        folder_path: Diretório raiz para verificar
    """
    import os
    from pathlib import Path
    
    folder_path = Path(folder_path)
    
    print("=" * 60)
    print("TESTE DE COMPILAÇÃO DE MÉTRICAS")
    print("=" * 60)
    
    # 1. Verificar se o diretório existe
    print(f"\n1. Verificando diretório: {folder_path}")
    if not folder_path.exists():
        print("   ❌ ERRO: Diretório não existe!")
        return
    if not folder_path.is_dir():
        print("   ❌ ERRO: Caminho não é um diretório!")
        return
    print("   ✓ Diretório existe")
    
    # 2. Listar subdiretórios
    print("\n2. Subdiretórios encontrados:")
    subdirs = [d for d in folder_path.iterdir() if d.is_dir()]
    if not subdirs:
        print("   ❌ Nenhum subdiretório encontrado!")
    else:
        for subdir in subdirs[:10]:  # Mostrar apenas os primeiros 10
            print(f"   - {subdir.name}")
        if len(subdirs) > 10:
            print(f"   ... e mais {len(subdirs) - 10} diretórios")
    
    # 3. Procurar arquivos Excel
    print("\n3. Procurando arquivos Excel...")
    excel_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.xlsx'):
                full_path = Path(root) / file
                excel_files.append(full_path)
                
    if not excel_files:
        print("   ❌ Nenhum arquivo Excel (.xlsx) encontrado!")
    else:
        print(f"   ✓ Encontrados {len(excel_files)} arquivos Excel")
        
        # Mostrar alguns exemplos
        print("\n   Primeiros arquivos encontrados:")
        for file in excel_files[:5]:
            rel_path = file.relative_to(folder_path)
            print(f"   - {rel_path}")
            
        # Verificar especificamente por 'spectral_analysis.xlsx'
        spectral_files = [f for f in excel_files if f.name.lower() == 'spectral_analysis.xlsx']
        print(f"\n   Arquivos 'spectral_analysis.xlsx' encontrados: {len(spectral_files)}")
        
    # 4. Testar leitura de um arquivo
    if spectral_files:
        print("\n4. Testando leitura do primeiro arquivo...")
        test_file = spectral_files[0]
        print(f"   Arquivo: {test_file}")
        
        try:
            # Tentar abrir o Excel
            import pandas as pd
            excel_data = pd.ExcelFile(test_file)
            print("   ✓ Arquivo aberto com sucesso")
            print(f"   Planilhas disponíveis: {excel_data.sheet_names}")
            
            # Verificar planilha 'Metrics'
            if 'Metrics' in excel_data.sheet_names:
                df_metrics = excel_data.parse('Metrics')
                print("\n   Planilha 'Metrics':")
                print(f"   - Linhas: {len(df_metrics)}")
                print(f"   - Colunas: {list(df_metrics.columns)}")
                
                if not df_metrics.empty:
                    print("\n   Primeira linha de dados:")
                    for col in df_metrics.columns:
                        val = df_metrics[col].iloc[0] if not df_metrics[col].empty else "N/A"
                        print(f"   - {col}: {val}")
            else:
                print("   ⚠ Planilha 'Metrics' não encontrada!")
                
        except Exception as e:
            print(f"   ❌ Erro ao ler arquivo: {e}")
            
    # 5. Sugestões
    print("\n" + "=" * 60)
    print("SUGESTÕES:")
    print("=" * 60)
    
    if not excel_files:
        print("1. Certifique-se de que os arquivos foram processados corretamente")
        print("2. Verifique se os arquivos têm a extensão .xlsx")
        print("3. Execute 'Apply Filters' antes de compilar métricas")
    elif not spectral_files:
        print("1. Os arquivos Excel devem se chamar 'spectral_analysis.xlsx'")
        print("2. Ou ajuste o parâmetro 'file_pattern' na função")
    else:
        print("1. Verifique se os arquivos Excel contêm a planilha 'Metrics'")
        print("2. Certifique-se de que as métricas foram calculadas corretamente")
        print("3. Verifique os logs para mensagens de erro detalhadas")


# Para executar o teste, adicione isto ao seu código principal:
if __name__ == "__main__":
    # Substitua pelo caminho do seu diretório de resultados
    test_compile_metrics("./results")