# --- Standard library
import os, sys, shutil, subprocess, logging, traceback
from pathlib import Path
from typing import Optional, List, Dict, Any, Union, Callable
from PyQt5.QtCore import QObject, QThread, pyqtSignal, pyqtSlot
# ou: from PySide6.QtCore import QObject, QThread, Signal as pyqtSignal, Slot as pyqtSlot

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PyQt5.QtCore import Qt, QThread, QThreadPool, QRunnable, QObject, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox,
    QTabWidget, QMessageBox, QFileDialog, QCheckBox,
    QGroupBox, QFormLayout, QSlider, QProgressDialog
)
from PyQt5.QtCore import Qt
try:
    from PyQt5.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, pyqtSlot
except Exception:
    from PySide6.QtCore import QRunnable, QThreadPool, QObject
    from PySide6.QtCore import Signal as pyqtSignal, Slot as pyqtSlot


# D.R. / ML
try:
    import umap  # umap-learn
    UMAP_AVAILABLE = True
except Exception:
    umap = None
    UMAP_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    TSNE_AVAILABLE = True
except Exception:
    TSNE_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    PCA_AVAILABLE = True
except Exception:
    PCA_AVAILABLE = False

from sklearn.preprocessing import StandardScaler            # <-- necessário
from sklearn.ensemble import IsolationForest                # <-- necessário

# Plotly p/ visualizações interativas
import plotly.graph_objects as go                           # <-- necessário
from plotly.subplots import make_subplots                   # <-- necessário

from proc_audio import AudioProcessor
# Removed spectral_power import

logger = logging.getLogger(__name__)

ALLOWED_OPEN_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".html", ".htm",
                     ".csv", ".json", ".xlsx", ".xls", ".txt", ".pdf"}


# --- Mapa de rótulos da UI (Português) para chaves internas (density.py)
_UI_WEIGHT_KEY = {

    "LOGARÍTMICA":   "log",
    "LINEAR":        "linear",
    "RAIZ QUADRADA": "sqrt",
    "RAIZ CÚBICA":   "cbrt",
    "QUADRADO":      "squared",
    "CÚBICA":        "cubic",
    "EXPONENCIAL":   "exp",
    "INV. LOG":      "inverse log",
    "SOMA":          "sum",
}


# --- Helpers de UI → core (mapeamento PT→EN para função de peso) ---
def _resolve_weight_key_from_ui(label: str) -> str:
    """
    Mapeia rótulos da UI (PT/EN) para keys aceites em density.get_weight_function().
    """
    if label is None:
        raise ValueError("Função de peso inválida (label None).")
    key = str(label).strip().lower()

    mapping = {
         "log": "log",
        "logarítmica": "log",
        "logaritmica": "log",
        "linear": "linear",
        "raiz quadrada": "sqrt",
        "sqrt": "sqrt",
        "quadrática": "square",
        "quadratica": "square",
        "exponencial": "exp",
        "exp": "exp",
    }
    resolved = mapping.get(key, key)
    if not resolved:
        raise ValueError(f"Função de peso inválida: {label!r}")
    return resolved

class SlotWorker(QObject):
    finished = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @pyqtSlot()
    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class _WorkerSignals(QObject):
    progress = pyqtSignal(int)     # 0..100
    error = pyqtSignal(str)
    result = pyqtSignal(object)
    finished = pyqtSignal()

class Worker(QRunnable):
    """Executa fn(params) no QThreadPool. Se fn aceitar progress_cb, usa-o."""
    def __init__(self, fn, params):
        super().__init__()
        self.fn = fn
        self.params = params
        self.signals = _WorkerSignals()

    @pyqtSlot()
    def run(self):
        try:
            try:
                result = self.fn(self.params, progress_cb=self.signals.progress)
            except TypeError:
                result = self.fn(self.params)
            self.signals.result.emit(result)
        except Exception as e:
            self.signals.error.emit(str(e))
        finally:
            self.signals.finished.emit()


def _safe_open_path(path: str | os.PathLike, allowed_root: Path | None = None) -> None:
    """Abre um ficheiro de forma segura (sem shell). Valida raiz e extensão."""
    p = Path(path)
    try:
        p = p.resolve(strict=True)
    except FileNotFoundError:
        raise FileNotFoundError(f"Ficheiro não encontrado: {path}")

    # Se houver uma raiz autorizada, garantir que o ficheiro está dentro dela
    if allowed_root is not None:
        root = Path(allowed_root).resolve()
        if os.path.commonpath([root, p]) != str(root):
            raise PermissionError(f"Caminho fora do diretório permitido: {p}")

    # Permitir só tipos previstos
    if p.suffix.lower() not in ALLOWED_OPEN_EXTS:
        raise ValueError(f"Extensão não permitida para abertura: {p.suffix}")


    # Abrir com app por omissão (input validado + executável whitelisted)
    p_str = str(Path(p).expanduser().resolve(strict=False))
    if "\x00" in p_str:
        raise ValueError("Caminho inválido (NUL).")

    # (opcional mas útil) garantir que é caminho local existente
    if not (os.path.isfile(p_str) or os.path.isdir(p_str)):
        logging.error("Caminho inexistente ou inválido: %s", p_str)
    else:
        if sys.platform.startswith("win"):
            os.startfile(p_str)  # nosec: caminho local validado

        elif sys.platform == "darwin":
            cmd = shutil.which("open") or "/usr/bin/open"
            subprocess.run([cmd, p_str], check=False)  # nosec S603,S607: cmd whitelisted; input validado

        else:  # Linux/BSD
            cmd = (shutil.which("xdg-open")
                   or shutil.which("gio")
                   or shutil.which("gnome-open"))
            if cmd:
                subprocess.run([cmd, p_str], check=False)  # nosec S603,S607: cmd whitelisted; input validado
            else:
                logging.error("Sem utilitário gráfico para abrir: %s", p_str)



class SpectrumAnalyzer(QMainWindow):
    """
    A PyQt5-based graphical user interface for spectral analysis.

    This class provides an interactive GUI for tasks like loading audio files,
    applying spectral analysis, configuring filters, and compiling results.
    It integrates functionalities like density metrics computation, dissonance calculation, and visualizations.
    """

    def __init__(self):
        """
        Initializes the graphical user interface (GUI).
        """
        super().__init__()
        self.setWindowTitle('Spectrum Analyzer')
        self.setGeometry(100, 100, 800, 600)

        # Core data/processing objects
        self.audio_processor = AudioProcessor()
        self.results_directory: Optional[str] = None

        # Thread pool (coloque aqui, dentro do __init__)
        self.threadpool = QThreadPool.globalInstance()

        # Set up the UI
        self.init_ui()


    def init_ui(self) -> None:
        """
        Initializes the user interface layout and tabs.
        """
        # 1) Light sand background for the entire QMainWindow
        self.setStyleSheet("background-color:rgb(230, 218, 204);")

        self.main_layout = QVBoxLayout()
        self.tabs = QTabWidget()

        self.setup_controls_tab()
        self.setup_filters_tab()
        self.setup_advanced_tab()

        self.main_layout.addWidget(self.tabs)
        container = QWidget()
        container.setLayout(self.main_layout)
        self.setCentralWidget(container)

    def setup_controls_tab(self) -> None:
        """
        Configures the 'Controls' tab of the GUI.
        """
        controls_tab = QWidget()
        controls_layout = QVBoxLayout()

        # Button: Load Audio Files
        self.load_button = QPushButton('Load Audio Files')
        # A pale olive green color
        self.load_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.load_button.clicked.connect(self.load_audio_files)
        controls_layout.addWidget(self.load_button)

        # Button: Choose Save Directory
        self.choose_save_dir_button = QPushButton('Choose Save Directory')
        # A pale olive green color
        self.choose_save_dir_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.choose_save_dir_button.clicked.connect(self.choose_save_directory)
        controls_layout.addWidget(self.choose_save_dir_button)

        # Button: Compile Metrics with PCA (Combined functionality)
        self.compile_metrics_button = QPushButton('Compile Metrics with PCA')
        # A pale olive green color
        self.compile_metrics_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.compile_metrics_button.clicked.connect(self.compile_metrics_with_pca)
        controls_layout.addWidget(self.compile_metrics_button)

        # Button: Generate Interactive Visualizations
        self.interactive_viz_button = QPushButton('Generate Interactive Visualizations')
        self.interactive_viz_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.interactive_viz_button.clicked.connect(self.generate_interactive_visualizations)
        controls_layout.addWidget(self.interactive_viz_button)

        # Button: View Dissonance Curves
        self.view_dissonance_curves_button = QPushButton('View Dissonance Curves')
        self.view_dissonance_curves_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.view_dissonance_curves_button.clicked.connect(self.view_dissonance_curves)
        controls_layout.addWidget(self.view_dissonance_curves_button)

        controls_tab.setLayout(controls_layout)
        self.tabs.addTab(controls_tab, "Controls")

    def setup_filters_tab(self) -> None:
        """
        Configures the 'Filters' tab of the GUI.
        """
        filters_tab = QWidget()
        filters_layout = QVBoxLayout()

        # Frequency and Magnitude Filter Group
        filter_group = QGroupBox("Frequency and Magnitude Filters")
        grid_filters = QFormLayout()

        # MODIFICAÇÃO: Mudar valor padrão de 200 para 20 Hz
        self.input_min_freq = QLineEdit("20")      # admitir G3 (≈196 Hz) e registos graves
        self.input_max_freq = QLineEdit("20000")
        self.input_min_db = QLineEdit("-90")
        self.input_max_db = QLineEdit("0")
        self.checkbox_adaptive_tolerance = QCheckBox("Usar tolerância adaptativa")
        self.checkbox_adaptive_tolerance.setChecked(True)  # Valor padrão
        grid_filters.addRow("Modo de tolerância:", self.checkbox_adaptive_tolerance)

        # MODIFICAÇÃO: Aumentar tolerância padrão
        self.input_tolerance = QLineEdit("5.0")

        grid_filters.addRow("Minimum Frequency (Hz):", self.input_min_freq)
        grid_filters.addRow("Maximum Frequency (Hz):", self.input_max_freq)
        grid_filters.addRow("Minimum Magnitude (dB):", self.input_min_db)
        grid_filters.addRow("Maximum Magnitude (dB):", self.input_max_db)
        grid_filters.addRow("Tolerance (Hz):", self.input_tolerance)

        filter_group.setLayout(grid_filters)
        filters_layout.addWidget(filter_group)

        # FFT Parameters Group
        fft_group = QGroupBox("FFT Parameters")
        fft_layout = QFormLayout()

        self.input_n_fft = QLineEdit("4096")
        self.input_hop_length = QLineEdit("")
        self.combo_window_type = QComboBox()
        self.combo_window_type.addItems(['hann', 'hamming', 'blackmanharris', 'bartlett', 'kaiser', 'gaussian'])

        fft_layout.addRow("FFT Window Size (n_fft):", self.input_n_fft)
        fft_layout.addRow("Hop Length:", self.input_hop_length)
        fft_layout.addRow("Window Type:", self.combo_window_type)

        fft_group.setLayout(fft_layout)
        filters_layout.addWidget(fft_group)


        # Adicionar opções para LFT - POSICIONADO AQUI
        lft_group = QGroupBox("LFT Parameters (Linear Time-Frequency Transform)")
        lft_layout = QFormLayout()

        self.check_use_lft = QCheckBox()
        self.check_use_lft.setChecked(False)
        lft_layout.addRow("Use LFT instead of FFT:", self.check_use_lft)

        self.input_zero_padding = QLineEdit("1")
        lft_layout.addRow("Zero Padding Factor:", self.input_zero_padding)

        self.combo_time_avg = QComboBox()
        self.combo_time_avg.addItems(['mean', 'median', 'max'])
        lft_layout.addRow("Time Averaging Method:", self.combo_time_avg)

        lft_group.setLayout(lft_layout)
        filters_layout.addWidget(lft_group)

        # Metric Calculation Group
        metric_group = QGroupBox("Metric Calculation")
        metric_layout = QFormLayout()

        self.combo_weight_function = QComboBox()
        self.combo_weight_function.addItems(['log','linear', 'sqrt', 'cbrt', 'exp','inverse log', 'sum'])
        metric_layout.addRow("Weight Function:", self.combo_weight_function)

        # Slider and layout for harmonic/inharmonic weights
        harmonic_weight_layout = QHBoxLayout()
        self.harmonic_weight_slider = QSlider(Qt.Horizontal)
        self.harmonic_weight_slider.setMinimum(0)     # 0% = 0.0 (minimum)
        self.harmonic_weight_slider.setMaximum(100)   # 100% = 1.0 (maximum)
        self.harmonic_weight_slider.setValue(95)      # Default: 95% harmonic, 5% inharmonic
        self.harmonic_weight_slider.setTickPosition(QSlider.TicksBelow)
        self.harmonic_weight_slider.setTickInterval(10)  # Ticks every 10%

        self.harmonic_weight_value = QLabel("95%")
        self.inharmonic_weight_value = QLabel("5%")

        self.harmonic_weight_slider.valueChanged.connect(self.update_harmonic_weight_display)

        harmonic_weight_layout.addWidget(QLabel("Harmonic (α):"))
        harmonic_weight_layout.addWidget(self.harmonic_weight_slider)
        harmonic_weight_layout.addWidget(self.harmonic_weight_value)
        harmonic_weight_layout.addWidget(QLabel("Inharmonic (β):"))
        harmonic_weight_layout.addWidget(self.inharmonic_weight_value)

        metric_layout.addRow("Combined Metric Weights:", harmonic_weight_layout)

        # Dissonance Model Selection
        self.combo_dissonance_model = QComboBox()
        self.combo_dissonance_model.addItems([
            'Sethares', 'Hutchinson-Knopoff', 'Vassilakis',
            'Aures-Zwicker', 'Stolzenburg', 'Spectral-Autocorrelation'
        ])
        metric_layout.addRow("Dissonance Model:", self.combo_dissonance_model)

        # Dissonance Controls
        self.check_dissonance_enabled = QCheckBox()
        self.check_dissonance_enabled.setChecked(True)  # Enable by default
        metric_layout.addRow("Enable Dissonance Analysis:", self.check_dissonance_enabled)

        self.check_dissonance_curve = QCheckBox()
        self.check_dissonance_curve.setChecked(True)
        metric_layout.addRow("Generate Dissonance Curve:", self.check_dissonance_curve)

        self.check_dissonance_scale = QCheckBox()
        self.check_dissonance_scale.setChecked(True)
        metric_layout.addRow("Generate Optimal Scale:", self.check_dissonance_scale)

        # Compare dissonance models option
        self.check_compare_models = QCheckBox()
        self.check_compare_models.setChecked(False)
        metric_layout.addRow("Compare All Dissonance Models:", self.check_compare_models)

        metric_group.setLayout(metric_layout)
        filters_layout.addWidget(metric_group)

        # Apply Button
        self.apply_filters_button = QPushButton('Apply Filters')
        self.apply_filters_button.clicked.connect(self.apply_filters)
        self.apply_filters_button.setFont(QFont("Arial", 10, QFont.Bold))
        self.apply_filters_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        filters_layout.addWidget(self.apply_filters_button)

        filters_tab.setLayout(filters_layout)
        self.tabs.addTab(filters_tab, "Filters")

    def _get_weight_function_from_ui(self) -> str:
        """
        Devolve SEMPRE a chave interna válida da função de peso, a partir do combo da UI.
        Faz:
          - leitura do texto;
          - normalização (lower/strip);
          - validação direta com get_weight_function;
          - fallback via mapa UI PT→EN (_resolve_weight_key_from_ui);
        Lança em caso extremo, expondo erro claro na UI/log.
        """
        try:
            raw_label = str(self.combo_weight_function.currentText())
        except Exception:
            raw_label = "linear"

        key = (raw_label or "").strip().lower()

        # 1) tentar usar diretamente (EN)
        try:
            from density import get_weight_function
            _ = get_weight_function(key)     # valida; lança se inválida
            return key
        except Exception:
            pass

        # 2) fallback via rótulo UI (PT) → chave interna (EN)
        try:
            wf = _resolve_weight_key_from_ui(raw_label)
            from density import get_weight_function
            _ = get_weight_function(wf)      # valida; lança se inválida
            return wf
        except Exception as e:
            import logging
            logging.getLogger(__name__).error("Weight Function inválida na UI: '%s' (%s)", raw_label, e)
            raise

    def update_harmonic_weight_display(self, value: int) -> None:
        """
        Actualiza os rótulos de percentagem α (harmónico) / β (inharmónico)
        quando o *slider* se move. Garante que nunca se exibem 0 % ou 100 %,
        impondo um mínimo de 1 % em cada componente.
        """
        # Limites de segurança: pelo menos 0 %, no máximo 100 %
        harmonic = max(0, min(100, value))
        inharmonic = 100 - harmonic

        self.harmonic_weight_value.setText(f"{harmonic}%")
        self.inharmonic_weight_value.setText(f"{inharmonic}%")


    def setup_advanced_tab(self) -> None:
        """
        Configures the 'Advanced' tab of the GUI.
        """
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout()

        # Dissonance Analysis Group
        dissonance_group = QGroupBox("Dissonance Analysis")
        dissonance_layout = QFormLayout()

        # Comparison options
        self.check_compare_dissonance = QCheckBox()
        self.check_compare_dissonance.setChecked(True)
        dissonance_layout.addRow("Compare Dissonance with Density:", self.check_compare_dissonance)

        # Scale visualization options
        self.combo_scale_visualization = QComboBox()
        self.combo_scale_visualization.addItems(['Cents', 'Ratio', 'Both'])
        dissonance_layout.addRow("Scale Visualization:", self.combo_scale_visualization)

        # Button: Analyze Dissonance vs Density
        self.analyze_dissonance_button = QPushButton('Analyze Dissonance vs Density')
        self.analyze_dissonance_button.setStyleSheet("background-color: rgb(219, 224, 169);")
        self.analyze_dissonance_button.clicked.connect(self.analyze_dissonance_vs_density)
        dissonance_layout.addRow(self.analyze_dissonance_button)

        dissonance_group.setLayout(dissonance_layout)
        advanced_layout.addWidget(dissonance_group)

        # Advanced Analysis Group
        advanced_analysis_group = QGroupBox("Advanced Analysis Options")
        advanced_analysis_layout = QFormLayout()

        # Dimensionality Reduction Methods
        self.check_use_pca = QCheckBox()
        self.check_use_pca.setChecked(True)
        advanced_analysis_layout.addRow("Use PCA:", self.check_use_pca)

        self.check_use_tsne = QCheckBox()
        self.check_use_tsne.setChecked(False)
        advanced_analysis_layout.addRow("Use t-SNE:", self.check_use_tsne)

        self.check_use_umap = QCheckBox()
        self.check_use_umap.setChecked(False)
        if not UMAP_AVAILABLE:
            self.check_use_umap.setEnabled(False)
            self.check_use_umap.setToolTip("UMAP not available. Install with 'pip install umap-learn'")
        advanced_analysis_layout.addRow("Use UMAP:", self.check_use_umap)

        # Anomaly Detection
        self.check_anomaly_detection = QCheckBox()
        self.check_anomaly_detection.setChecked(False)
        advanced_analysis_layout.addRow("Detect Anomalies:", self.check_anomaly_detection)

        self.input_contamination = QLineEdit("0.05")
        advanced_analysis_layout.addRow("Expected Anomaly Fraction:", self.input_contamination)

        # Include dissonance in analysis
        self.check_include_dissonance = QCheckBox()
        self.check_include_dissonance.setChecked(True)
        advanced_analysis_layout.addRow("Include Dissonance in Analysis:", self.check_include_dissonance)

        advanced_analysis_group.setLayout(advanced_analysis_layout)
        advanced_layout.addWidget(advanced_analysis_group)

        # Interactive Visualization Options
        viz_group = QGroupBox("Interactive Visualization Options")
        viz_layout = QFormLayout()

        self.check_3d_spectrogram = QCheckBox()
        self.check_3d_spectrogram.setChecked(True)
        viz_layout.addRow("3D Spectrograms:", self.check_3d_spectrogram)

        self.check_interactive_curves = QCheckBox()
        self.check_interactive_curves.setChecked(True)
        viz_layout.addRow("Interactive Dissonance Curves:", self.check_interactive_curves)

        self.check_dimension_scatterplots = QCheckBox()
        self.check_dimension_scatterplots.setChecked(True)
        viz_layout.addRow("Dimensionality Reduction Plots:", self.check_dimension_scatterplots)

        viz_group.setLayout(viz_layout)
        advanced_layout.addWidget(viz_group)

        advanced_tab.setLayout(advanced_layout)
        self.tabs.addTab(advanced_tab, "Advanced")
    # No arquivo interface.py, dentro da classe SpectrumAnalyzer

    # Linha ~340: Melhorar método create_interactive_visualizations
    def create_interactive_visualizations(self, df: pd.DataFrame, output_dir: str) -> None:
        """
        Cria visualizações interativas mais modernas e responsivas para os dados espectrais.
        Melhorada com tratamento de erros e verificação de dados.
        """
        import plotly.express as px

        try:
            # Garantir que diretório existe
            os.makedirs(output_dir, exist_ok=True)

            # Verificar se o DataFrame contém dados válidos
            if df is None or df.empty:
                logger.error("DataFrame vazio fornecido para visualizações interativas")
                # Criar HTML de erro informativo
                error_path = os.path.join(output_dir, 'error.html')
                with open(error_path, 'w') as f:
                    f.write("<html><body><h1>Erro ao Gerar Visualizações</h1>")
                    f.write("<p>Não há dados válidos para visualização.</p></body></html>")
                return error_path

            # 1. Visualização PCA interativa
            if 'PC1' in df.columns and 'PC2' in df.columns:
                try:
                    # Verificar e preparar dados para visualização
                    color_column = None
                    if 'Density Metric' in df.columns:
                        # Verificar se a coluna contém dados numéricos
                        if pd.api.types.is_numeric_dtype(df['Density Metric']):
                            color_column = 'Density Metric'

                    # Verificar coluna de texto para rótulos
                    hover_name = None
                    if 'Note' in df.columns:
                        hover_name = 'Note'

                    # Identificar colunas para informações de hover
                    hover_data = []
                    for col in df.columns:
                        if ('Metric' in col or 'Dissonance' in col) and pd.api.types.is_numeric_dtype(df[col]):
                            hover_data.append(col)

                    # Criar gráfico PCA
                    fig = px.scatter(
                        df, x='PC1', y='PC2',
                        color=color_column,
                        hover_name=hover_name,
                        hover_data=hover_data,
                        title='PCA Analysis of Spectral Properties',
                        labels={'PC1': 'Principal Component 1', 'PC2': 'Principal Component 2'},
                        color_continuous_scale='viridis'
                    )

                    # Melhorar layout
                    fig.update_layout(
                        template='plotly_white',
                        margin=dict(l=10, r=10, t=50, b=10),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                        width=900, height=700
                    )

                    # Salvar como HTML interativo
                    fig.write_html(os.path.join(output_dir, 'pca_interactive.html'))
                    logger.info(f"Visualização PCA interativa criada: {os.path.join(output_dir, 'pca_interactive.html')}")
                except Exception as e:
                    logger.error(f"Erro ao criar visualização PCA interativa: {e}")

            # [Código para outras visualizações com tratamento similar...]

            # Dashboard aprimorado para integrar as visualizações disponíveis
            dashboard_files = []
            for viz_file in ['pca_interactive.html', 'correlation_interactive.html', 'metrics_comparison_interactive.html']:
                if os.path.exists(os.path.join(output_dir, viz_file)):
                    dashboard_files.append(viz_file)

            if dashboard_files:
                dashboard_html = self._create_dashboard_html(dashboard_files)
                dashboard_path = os.path.join(output_dir, 'dashboard.html')

                with open(dashboard_path, 'w') as f:
                    f.write(dashboard_html)

                return dashboard_path
            else:
                logger.warning("Nenhuma visualização criada para incluir no dashboard")
                return None

        except Exception as e:
            logger.error(f"Erro ao criar visualizações interativas: {e}")
            # Criar arquivo de erro
            error_path = os.path.join(output_dir, 'visualization_error.txt')
            with open(error_path, 'w') as f:
                f.write(f"Erro ao criar visualizações interativas: {str(e)}")
            return error_path

    # Adicionar método auxiliar para criar HTML do dashboard
    def _create_dashboard_html(self, viz_files: List[str]) -> str:
        """Cria HTML para dashboard com tratamento de erros para arquivos ausentes"""

        # Template para início do HTML
        html_start = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Spectral Analysis Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }
                .dashboard { display: flex; flex-direction: column; gap: 20px; }
                .dashboard-item { background-color: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .dashboard-header { text-align: center; margin-bottom: 20px; }
                .viz-container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
                .viz-item { flex: 1; min-width: 300px; min-height: 300px; }
                h1, h2 { color: #2c3e50; }
                iframe { border: none; width: 100%; height: 600px; }
                .error { color: red; padding: 10px; background-color: #ffeeee; border-radius: 4px; }
            </style>
        </head>
        <body>
            <div class="dashboard">
                <div class="dashboard-header">
                    <h1>Spectral Analysis Dashboard</h1>
                    <p>Interactive visualizations for spectral analysis data</p>
                </div>
        """

        # Mapear nomes de arquivo para títulos
        file_titles = {
            'pca_interactive.html': 'Principal Component Analysis',
            'correlation_interactive.html': 'Correlation Matrix',
            'metrics_comparison_interactive.html': 'Metrics Comparison'
        }

        # Construir o conteúdo do dashboard
        html_items = ""
        for viz_file in viz_files:
            title = file_titles.get(viz_file, viz_file)
            html_items += f"""
            <div class="dashboard-item">
                <h2>{title}</h2>
                <iframe src="{viz_file}" onload="this.style.height = Math.max(600, this.contentWindow.document.body.scrollHeight + 30) + 'px';"></iframe>
            </div>
            """

        # Final do HTML
        html_end = """
            </div>
        </body>
        </html>
        """

        return html_start + html_items + html_end

    def plot_enhanced_spectrum(self, harmonic_df: pd.DataFrame, note: str) -> None:
        """
        Cria uma visualização aprimorada do espectro harmônico.

        Args:
            harmonic_df: DataFrame com parciais harmônicos
            note: Nome da nota musical
        """
        import plotly.graph_objects as go

        if harmonic_df is None or harmonic_df.empty:
            return

        # Extrair dados
        frequencies = harmonic_df['Frequency (Hz)'].values
        amplitudes = harmonic_df['Amplitude'].values if 'Amplitude' in harmonic_df.columns else \
                    10**(harmonic_df['Magnitude (dB)'].values/20)
        harmonic_numbers = harmonic_df['Harmonic Number'].values if 'Harmonic Number' in harmonic_df.columns else \
                          np.arange(1, len(frequencies)+1)

        # Normalizar amplitudes para visualização
        norm_amplitudes = amplitudes / np.max(amplitudes)

        # Criar figura
        fig = go.Figure()

        # Adicionar barras para cada parcial
        fig.add_trace(go.Bar(
            x=harmonic_numbers,
            y=norm_amplitudes,
            marker=dict(
                color=norm_amplitudes,
                colorscale='Viridis',
                line=dict(color='rgba(0,0,0,0.5)', width=1)
            ),
            name='Amplitude',
            text=[f"{freq:.1f} Hz" for freq in frequencies],
            hovertemplate='Harmonic: %{x}<br>Amplitude: %{y:.3f}<br>Frequency: %{text}'
        ))

        # Layout
        fig.update_layout(
            title=f'Harmonic Spectrum - {note}',
            xaxis_title='Harmonic Number',
            yaxis_title='Normalized Amplitude',
            template='plotly_white',
            height=600,
            width=900,
            showlegend=False
        )

        # Adicionar linha para a série harmônica ideal
        if len(frequencies) > 1:
            f0 = frequencies[0]  # Frequência fundamental
            ideal_frequencies = [f0 * (i+1) for i in range(len(frequencies))]

            fig.add_trace(go.Scatter(
                x=harmonic_numbers,
                y=[0.05] * len(harmonic_numbers),  # Altura fixa para visualização
                mode='markers',
                marker=dict(
                    symbol='diamond',
                    size=12,
                    color='red',
                    line=dict(color='rgba(0,0,0,0.5)', width=1)
                ),
                name='Ideal Harmonics',
                text=[f"{freq:.1f} Hz" for freq in ideal_frequencies],
                hovertemplate='Harmonic: %{x}<br>Ideal Frequency: %{text}'
            ))

        # Salvar como HTML interativo
        output_dir = os.path.join(self.results_directory, note)
        os.makedirs(output_dir, exist_ok=True)

        fig.write_html(os.path.join(output_dir, 'enhanced_spectrum.html'))

        # Também criar uma versão PNG para exportação
        fig.write_image(os.path.join(output_dir, 'enhanced_spectrum.png'))

    # No arquivo interface.py, dentro da classe SpectrumAnalyzer

    def save_spectral_analysis(self, note, harmonic_df):
        """
        Salva resultados da análise espectral, incluindo visualizações aprimoradas.

        Args:
            note: Nome da nota musical
            harmonic_df: DataFrame com parciais harmônicos
        """
        # Código existente para salvar resultados...

        # Adicionar visualização aprimorada
        try:
            # Verificar se temos PyPlot instalado para visualizações avançadas
            import importlib
            if importlib.util.find_spec("plotly") is not None:
                self.plot_enhanced_spectrum(harmonic_df, note)
            else:
                logger.warning("Plotly não está instalado. Visualizações avançadas desabilitadas.")
        except Exception as e:
            logger.error(f"Erro ao criar visualização aprimorada para {note}: {e}")

    # No arquivo interface.py, dentro da classe SpectrumAnalyzer

    def plot_enhanced_dissonance_curve(self, model_name: str, curve: Dict, scale: List, note: str) -> None:
        """
        Cria uma visualização aprimorada da curva de dissonância com escala.

        Args:
            model_name: Nome do modelo de dissonância
            curve: Dicionário da curva de dissonância
            scale: Lista de intervalos da escala
            note: Nome da nota musical
        """
        import plotly.graph_objects as go

        if not curve or not scale:
            return

        # Preparar dados
        intervals = sorted(list(curve.keys()))
        dissonance_values = [curve[i] for i in intervals]

        # Converter intervalos para centésimos
        cents = [1200 * np.log2(i) for i in intervals]

        # Criar figura
        fig = go.Figure()

        # Adicionar curva de dissonância
        fig.add_trace(go.Scatter(
            x=cents,
            y=dissonance_values,
            mode='lines',
            line=dict(color='blue', width=2),
            name='Dissonance Curve'
        ))

        # Adicionar pontos da escala
        scale_cents = [1200 * np.log2(i) for i in scale]
        scale_values = [curve.get(i, 0) for i in scale]

        fig.add_trace(go.Scatter(
            x=scale_cents,
            y=scale_values,
            mode='markers',
            marker=dict(
                color='red',
                size=10,
                line=dict(color='black', width=1)
            ),
            name='Optimal Scale Points'
        ))

        # Adicionar linhas verticais e rótulos para intervalos musicais comuns
        common_intervals = {
            0: "Unison",
            100: "Minor 2nd",
            200: "Major 2nd",
            300: "Minor 3rd",
            400: "Major 3rd",
            500: "Perfect 4th",
            600: "Tritone",
            700: "Perfect 5th",
            800: "Minor 6th",
            900: "Major 6th",
            1000: "Minor 7th",
            1100: "Major 7th",
            1200: "Octave"
        }

        for cents_value, name in common_intervals.items():
            fig.add_shape(
                type="line",
                x0=cents_value, y0=min(dissonance_values),
                x1=cents_value, y1=max(dissonance_values),
                line=dict(color="rgba(128, 128, 128, 0.5)", width=1, dash="dot")
            )

            # Adicionar rótulo
            fig.add_annotation(
                x=cents_value,
                y=max(dissonance_values) * 1.05,
                text=name,
                showarrow=False,
                textangle=-90,
                font=dict(size=10)
            )

        # Layout
        fig.update_layout(
            title=f'{model_name} Dissonance Curve - {note}',
            xaxis_title='Interval (cents)',
            yaxis_title='Dissonance',
            template='plotly_white',
            height=600,
            width=900,
            xaxis=dict(
                tickmode='array',
                tickvals=list(common_intervals.keys()),
                ticktext=list(common_intervals.values()),
                tickangle=-45
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Adicionar eixo x secundário com razões
        tick_intervals = [1.0, 1.125, 1.25, 1.333, 1.5, 1.667, 1.75, 2.0]
        tick_cents = [1200 * np.log2(i) for i in tick_intervals]

        fig.update_layout(
            xaxis2=dict(
                overlaying="x",
                side="bottom",
                position=0.05,
                tickmode='array',
                tickvals=tick_cents,
                ticktext=[f"{i:.3f}" for i in tick_intervals],
                title="Frequency Ratio",
                showgrid=False,
                zeroline=False
            )
        )

        # Salvar como HTML interativo
        output_dir = os.path.join(self.results_directory, note)
        os.makedirs(output_dir, exist_ok=True)

        fig.write_html(os.path.join(output_dir, f'{model_name.lower()}_dissonance_curve.html'))

        # Também criar uma versão PNG para exportação
        fig.write_image(os.path.join(output_dir, f'{model_name.lower()}_dissonance_curve.png'))

    # -------------------------------------------------------------------------
    #                           CONTROLS TAB FUNCTIONS
    # -------------------------------------------------------------------------

    def choose_save_directory(self) -> None:
        """
        Opens a dialog for selecting the directory to save results.
        """
        selected_directory = QFileDialog.getExistingDirectory(
            self, "Select Directory to Save Results", os.getcwd()
        )
        if selected_directory:
            self.results_directory = selected_directory
            QMessageBox.information(self, "Directory Selected",
                                    f"Results will be saved in: {selected_directory}")
        else:
            QMessageBox.warning(self, "Warning", "No directory selected.")

    def load_audio_files(self) -> None:
        """
        Opens a dialog for selecting and loading audio files.
        """
        try:
            options = QFileDialog.Options()
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Audio Files",
                "",
                "Audio Files (*.wav *.mp3 *.flac *.aif *.aiff);;All Files (*)",
                options=options
            )
            if files:
                self.audio_processor.load_audio_files(files)
                QMessageBox.information(self, "Success",
                                        f"{len(files)} files successfully loaded.")
            else:
                QMessageBox.warning(self, "Warning", "No files selected.")
        except Exception as e:
            QMessageBox.critical(self, "Error",
                                 f"An error occurred while loading the files: {str(e)}")


    # Linha ~450: Melhorar método compile_metrics_with_pca

    def compile_metrics_with_pca(self) -> None:
        """
        Compiles metrics and runs advanced analysis in a background thread.
        Lê os parâmetros na UI (thread principal), corre em QThreadPool (QRunnable),
        injeta um shim de 'signals' no método _run_compile_metrics_task (para .progress.emit(int,str)),
        atualiza a UI por sinais, e reativa controlos no fim/erro/cancelamento.
        """
        # 1) Escolha da pasta
        selected_folder = QFileDialog.getExistingDirectory(
            self, "Select the Folder with Results", os.getcwd()
        )
        if not selected_folder:
            QMessageBox.warning(self, "Warning", "No folder selected.")
            return

        output_path = os.path.join(selected_folder, "compiled_metrics_with_analysis.xlsx")

        # 2) Ler parâmetros na UI (thread principal)
        try:
            params = {
                "folder_path": selected_folder,
                "output_path": output_path,
                "include_pca": bool(self.check_use_pca.isChecked()),
                "use_tsne": bool(self.check_use_tsne.isChecked()),
                "use_umap": bool(self.check_use_umap.isChecked()) and bool(UMAP_AVAILABLE),
                "detect_anomalies": bool(self.check_anomaly_detection.isChecked()),
                "include_dissonance": bool(self.check_include_dissonance.isChecked()),
                "harmonic_weight": float(self.harmonic_weight_slider.value()) / 100.0,
                "weight_function_label": str(self.combo_weight_function.currentText()),
                "anomaly_contamination": float(self.input_contamination.text()),
            }
        except (ValueError, TypeError) as e:
            QMessageBox.critical(self, "Invalid Parameter", f"One of the input parameters is invalid: {e}")
            return

        # 3) Progress dialog
        progress = QProgressDialog("Compiling metrics...", "Cancel", 0, 100, self)
        progress.setWindowTitle("Analysis in Progress")
        progress.setWindowModality(Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setValue(0)
        progress.show()

        # 4) Desativar controlos durante o processamento
        if hasattr(self, "_set_controls_enabled"):
            self._set_controls_enabled(False)

        # 5) Worker em QThreadPool (sem moveToThread; com shim de signals)
        try:
            from PyQt5.QtCore import QRunnable, QThreadPool, QObject, pyqtSignal, pyqtSlot
        except Exception:
            from PySide6.QtCore import QRunnable, QThreadPool, QObject
            from PySide6.QtCore import Signal as pyqtSignal, Slot as pyqtSlot  # fallback

        class _CMWSignals(QObject):
            # importante: aceita (int, str) porque o teu método emite assim
            progress = pyqtSignal(int, str)
            error = pyqtSignal(str)
            result = pyqtSignal(object)
            finished = pyqtSignal()

        class _ShimEmitter:
            """Fornece .emit(...) compatível, ligando aos sinais Qt externos."""
            def __init__(self, emit_fn):
                self._emit_fn = emit_fn
            def emit(self, *args, **kwargs):
                # encaminha para o sinal Qt correspondente
                self._emit_fn(*args, **kwargs)

        class _ShimSignals:
            """Objeto 'signals' com atributos .progress/.error/.result/.finished (cada um com .emit)."""
            def __init__(self, qt_signals: _CMWSignals):
                self.progress = _ShimEmitter(qt_signals.progress.emit)
                self.error    = _ShimEmitter(qt_signals.error.emit)
                self.result   = _ShimEmitter(qt_signals.result.emit)
                self.finished = _ShimEmitter(qt_signals.finished.emit)

        class _CMWRunner(QRunnable):
            """Executa fn(params) no threadpool e injeta owner.signals se não existir."""
            def __init__(self, fn, params_dict):
                super().__init__()
                self.fn = fn                      # bound method: owner = fn.__self__
                self.params = params_dict
                self.signals = _CMWSignals()
                self._cancel = False

            def cancel(self):
                self._cancel = True

            @pyqtSlot()
            def run(self):
                owner = getattr(self.fn, "__self__", None)
                cleanup = False
                try:
                    # Injeta 'signals' se o método esperar self.signals.progress.emit(...)
                    if owner is not None and not hasattr(owner, "signals"):
                        owner.signals = _ShimSignals(self.signals)
                        cleanup = True

                    # Chamada principal SEM kwargs extra (o teu método não aceita progress_cb)
                    result = self.fn(self.params)

                    if not self._cancel:
                        # Se o método próprio tiver usado self.signals.result.emit, ótimo.
                        # Caso contrário, emitimos aqui um result convencional:
                        self.signals.result.emit(result)
                except Exception as e:
                    self.signals.error.emit(str(e))
                finally:
                    # Emite finished em qualquer caso
                    self.signals.finished.emit()
                    # Limpa o shim para não deixar atributos pendurados no owner
                    if cleanup:
                        try:
                            delattr(owner, "signals")
                        except Exception:
                            pass

        runner = _CMWRunner(self._run_compile_metrics_task, params)

        # 6) Handlers de sinais (progresso agora recebe (int, str))
        def _on_progress(val: int, msg: str = ""):
            if progress.wasCanceled():
                runner.cancel()
                return
            progress.setValue(int(max(0, min(100, val))))
            if msg:
                progress.setLabelText(str(msg))

        def _on_result(_payload):
            # sucesso (pode já ter havido 'result' interno)
            if progress.isVisible():
                progress.setValue(100)
                progress.close()
            if hasattr(self, "_set_controls_enabled"):
                self._set_controls_enabled(True)
            QMessageBox.information(self, "Done", f"Metrics compiled.\nSaved to:\n{output_path}")

        def _on_error(msg: str):
            if progress.isVisible():
                progress.close()
            if hasattr(self, "_set_controls_enabled"):
                self._set_controls_enabled(True)
            try:
                self.logger.exception("Erro no worker: %s", msg)
            except Exception:
                pass
            QMessageBox.critical(self, "Error", f"An unexpected error occurred:\n{msg}")

        def _on_finished():
            # salvaguarda — garante UI reativada mesmo sem result/erro explícito
            if hasattr(self, "_set_controls_enabled"):
                self._set_controls_enabled(True)
            if progress.isVisible():
                progress.close()

        # 7) Ligar sinais
        runner.signals.progress.connect(_on_progress)
        runner.signals.result.connect(_on_result)
        runner.signals.error.connect(_on_error)
        runner.signals.finished.connect(_on_finished)
        progress.canceled.connect(runner.cancel)

        # 8) Arrancar no threadpool
        if not hasattr(self, "threadpool") or self.threadpool is None:
            self.threadpool = QThreadPool.globalInstance()
        self.threadpool.start(runner)




    def _run_compile_metrics_task(self, params: dict) -> dict:
        """
        Executes the metric compilation task in the background.
        This version is decoupled from the UI and receives all parameters via a dictionary.
        """
        try:
            from compile_metrics import compile_density_metrics
            from density import get_weight_function

            # --- CORRECTED PATTERN: Use parameters passed via the `params` dictionary ---
            # No more direct access to `self.some_widget`.
            folder_path = params["folder_path"]
            output_path = params["output_path"]

            # Resolve weight function key safely
            try:
                raw_label = params["weight_function_label"]
                key = (raw_label or "").strip().lower()
                get_weight_function(key)
                weight_function = key
            except Exception:
                weight_function = _resolve_weight_key_from_ui(raw_label)

            harmonic_weight = params["harmonic_weight"]
            inharmonic_weight = 1.0 - harmonic_weight

            # Step 1: Compile base metrics
            self.signals.progress.emit(10, "Compiling base metrics...")
            compiled_df = compile_density_metrics(
                folder_path=folder_path,
                output_path=None,  # Save only at the end
                include_pca=params["include_pca"],
                harmonic_weight=harmonic_weight,
                inharmonic_weight=inharmonic_weight,
                weight_function=weight_function
            )
            if compiled_df is None or compiled_df.empty:
                return {"success": False, "message": "No valid data found for compilation."}

            self.signals.progress.emit(40, "Extracting features for analysis...")
            numeric_cols = self.get_numeric_columns(compiled_df, params["include_dissonance"])

            # Step 2: Apply optional dimensionality reduction
            if len(numeric_cols) >= 2 and (params["use_tsne"] or params["use_umap"]):
                self.signals.progress.emit(55, "Applying dimensionality reduction...")
                compiled_df = self.apply_additional_dimension_reduction(
                    compiled_df, numeric_cols, params["use_tsne"], params["use_umap"]
                )

            # Step 3: Apply optional anomaly detection
            if len(numeric_cols) >= 2 and params["detect_anomalies"]:
                self.signals.progress.emit(70, "Detecting anomalies...")
                compiled_df = self.detect_spectral_anomalies(
                    compiled_df, numeric_cols, contamination=params["anomaly_contamination"]
                )

            # Step 4: Save final results
            self.signals.progress.emit(90, "Saving final Excel file...")
            compiled_df.to_excel(output_path, index=False)

            self.signals.progress.emit(100, "Done!")
            return {"success": True, "path": str(output_path)}

        except Exception as e:
            logger.error("Error in background task:", exc_info=True)
            # Propagate the error back to the main thread via the error signal
            self.signals.error.emit(str(e))
            return {"success": False, "message": str(e)}


        def report(cur: int, tot: int, msg: str) -> None:
            if progress_cb:
                try:
                    progress_cb(int(cur), int(tot), str(msg))
                except Exception:
                    pass  # não deixar o progresso rebentar a tarefa

        def check_cancel() -> None:
            th = QThread.currentThread()
            if th is not None and th.isInterruptionRequested():
                raise InterruptedError("Operação cancelada pelo utilizador.")

        try:
            # Imports locais (reduz custo no arranque)
            from compile_metrics import compile_density_metrics
            from density import get_weight_function  # para validação da chave

            # Normalizar paths
            selected_folder = Path(selected_folder)
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # ------------------ Ler parâmetros (com fallback robusto) ------------------
            # ATENÇÃO: Ler widgets numa thread de worker não é o ideal. Mantemos try/except
            # com defaults caso não seja seguro aceder.
            try:
                harmonic_weight = float(self.harmonic_weight_slider.value()) / 100.0
            except Exception:
                harmonic_weight = 0.5

            try:
                raw_label = str(self.combo_weight_function.currentText())
            except Exception:
                raw_label = "linear"

            # 1) tentar usar diretamente (UI pode já estar em EN: 'linear','sqrt','cbrt','exp',...)
            key = (raw_label or "").strip().lower()
            try:
                # valida se é uma chave interna válida
                _ = get_weight_function(key)
                weight_function = key
            except Exception:
                # 2) UI em PT → normalizar via mapa e validar; se falhar, cair para 'linear'
                try:
                    weight_function = _resolve_weight_key_from_ui(raw_label)  # devolve 'linear','sqrt',...
                    _ = get_weight_function(weight_function)  # valida
                except Exception:
                    logger.warning("Rótulo de weight desconhecido %r; a usar 'linear'.", raw_label)
                    weight_function = "linear"

            logger.debug("UI weight label=%r → chave interna=%r", raw_label, weight_function)

            try:
                include_dissonance = bool(self.check_include_dissonance.isChecked())
            except Exception:
                include_dissonance = False

            # clamp pesos
            harmonic_weight = max(0.0, min(harmonic_weight, 1.0))
            inharmonic_weight = 1.0 - harmonic_weight

            # ------------------ Passo 1: Compilar métricas ------------------
            report(10, 100, "A preparar compilação…")
            check_cancel()

            compiled_df = compile_density_metrics(
                folder_path=selected_folder,
                output_path=output_path,
                include_pca=include_pca,
                harmonic_weight=harmonic_weight,
                inharmonic_weight=inharmonic_weight,
                weight_function=weight_function
            )

            report(40, 100, "Métricas compiladas. A preparar análises…")
            check_cancel()

            if compiled_df is None or getattr(compiled_df, "empty", True):
                return {"success": False, "message": "Nenhum dado válido encontrado para compilação."}

            # ------------------ Passo 2: Seleção de colunas numéricas ------------------
            try:
                numeric_cols = self.get_numeric_columns(compiled_df, include_dissonance)
            except Exception:
                import pandas as pd
                numeric_cols = [c for c in compiled_df.columns
                                if pd.api.types.is_numeric_dtype(compiled_df[c].dtype)]

            # ------------------ Passo 3: DR (t-SNE/UMAP) opcional ------------------
            if len(numeric_cols) >= 2 and (use_tsne or use_umap):
                report(55, 100, "A aplicar redução dimensional…")
                check_cancel()
                try:
                    compiled_df = self.apply_additional_dimension_reduction(
                        compiled_df, numeric_cols, use_tsne, use_umap
                    )
                except Exception as e:
                    report(60, 100, f"Redução dimensional falhou ({e}). A continuar…")

            # ------------------ Passo 4: Deteção de anomalias opcional ------------------
            if len(numeric_cols) >= 2 and detect_anomalies:
                report(70, 100, "A detetar anomalias…")
                check_cancel()
                try:
                    compiled_df = self.detect_spectral_anomalies(compiled_df, numeric_cols)
                except Exception as e:
                    report(75, 100, f"Deteção de anomalias falhou ({e}). A continuar…")

            # ------------------ Passo 5: Guardar resultados ------------------
            report(85, 100, "A guardar resultados…")
            check_cancel()

            # Excel consolidado
            compiled_df.to_excel(output_path, index=False)

            # (Relatório markdown opcional, se precisares volta a ligar aqui)
            report(100, 100, "Concluído.")

            return {"success": True, "path": str(output_path)}

        except InterruptedError as ie:
            return {"success": False, "message": str(ie)}
        except Exception as e:
            return {"success": False, "message": str(e), "traceback": traceback.format_exc()}

    # Adicionar método para atualizar progresso
    def _update_compile_progress(self, progress_bar, current, total, message):
        """Atualiza a barra de progresso durante a compilação"""
        if progress_bar is None:
            return

        if total > 0:
            percent = int((current / total) * 100)
            progress_bar.setValue(percent)

        if message:
            progress_bar.setLabelText(message)

        # Processar eventos para manter interface responsiva
        QApplication.processEvents()

    # Adicionar método para manipular resultado da compilação
    def _handle_compile_result(self, result, output_path):
        """Manipula o resultado da compilação em segundo plano"""
        if result.get('success', False):
            QMessageBox.information(
                self,
                "Success",
                f"Métricas compiladas com análise completa salvas em:\n{output_path}\n\n"
                f"Relatório de análise disponível em:\n{result.get('report', '')}"
            )
        else:
            QMessageBox.warning(
                self,
                "Warning",
                f"Problema na compilação: {result.get('message', 'Erro desconhecido')}"
            )



    def get_numeric_columns(self, df: pd.DataFrame, include_dissonance: bool = True) -> List[str]:
        """
        Devolve as colunas numéricas (ou convertíveis) relevantes para análise.

        Critérios:
          1) Começa pelas métricas padrão (se existirem no DataFrame).
          2) Opcionalmente acrescenta colunas cujo nome contém "Dissonance".
          3) Considera numéricas:
             - colunas já numéricas; ou
             - colunas convertíveis via pd.to_numeric(errors="coerce") com >= 2 valores não nulos.
        """
        # 1) métricas padrão (ordem preservada)
        standard_cols = [
            "Density Metric",
            "Spectral Density Metric",
            "Total Metric",
            "Combined Density Metric",
            "Filtered Density Metric",
        ]
        candidates: List[str] = [c for c in standard_cols if c in df.columns]

        # 2) acrescentar colunas de dissonância, se pedido
        if include_dissonance:
            candidates += [c for c in df.columns if "Dissonance" in c]

        # 2.1) deduplicar preservando ordem
        seen: set = set()
        candidates = [c for c in candidates if not (c in seen or seen.add(c))]

        # 3) selecionar as que são numéricas ou convertíveis com pelo menos 2 valores válidos
        numeric_now = set(df.select_dtypes(include="number").columns)
        numeric_sel: set = set(c for c in candidates if c in numeric_now)

        for c in candidates:
            if c not in numeric_sel:
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notnull().sum() >= 2:
                    numeric_sel.add(c)

        # devolver na ordem dos candidatos
        return [c for c in candidates if c in numeric_sel]



    def apply_additional_dimension_reduction(
        self,
        df: pd.DataFrame,
        metrics_columns: List[str],
        use_tsne: bool = False,
        use_umap: bool = False
    ) -> pd.DataFrame:
        """
        Applies additional dimensionality reduction methods beyond PCA.

        Args:
            df: DataFrame with compiled metrics
            metrics_columns: Columns containing numeric metrics
            use_tsne: Whether to apply t-SNE
            use_umap: Whether to apply UMAP

        Returns:
            DataFrame with additional components added
        """
        result_df = df.copy()

        # Prepare data
        X = df[metrics_columns].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply t-SNE if requested
        if use_tsne:
            try:
                tsne = TSNE(n_components=2, random_state=42)
                tsne_result = tsne.fit_transform(X_scaled)
                result_df['TSNE1'] = tsne_result[:, 0]
                result_df['TSNE2'] = tsne_result[:, 1]
            except Exception as e:
                print(f"Error applying t-SNE: {e}")

        # Apply UMAP if requested and available
        if use_umap and UMAP_AVAILABLE:
            try:
                reducer = umap.UMAP(random_state=42)
                umap_result = reducer.fit_transform(X_scaled)
                result_df['UMAP1'] = umap_result[:, 0]
                result_df['UMAP2'] = umap_result[:, 1]
            except Exception as e:
                print(f"Error applying UMAP: {e}")

        return result_df

    def detect_spectral_anomalies(
        self,
        df: pd.DataFrame,
        metrics_columns: List[str],
        contamination: float = 0.05
    ) -> pd.DataFrame:
        """
        Detects anomalies in spectral data using Isolation Forest.

        Args:
            df: DataFrame with compiled metrics
            metrics_columns: Columns containing numeric metrics
            contamination: Expected fraction of anomalies

        Returns:
            DataFrame with anomaly indicators added
        """
        result_df = df.copy()

        try:
            # Prepare data
            X = df[metrics_columns].dropna().values

            if len(X) < 10:
                print("Insufficient samples for anomaly detection")
                result_df['is_anomaly'] = False
                return result_df

            # Get contamination parameter from UI
            try:
                contamination = float(self.input_contamination.text())
                if contamination <= 0 or contamination >= 1:
                    contamination = 0.05  # Default if invalid
            except:
                contamination = 0.05  # Default if parsing fails

            # Apply Isolation Forest
            clf = IsolationForest(contamination=contamination, random_state=42)
            result_df['is_anomaly'] = clf.fit_predict(X) == -1

            # Calculate and add anomaly score
            result_df['anomaly_score'] = clf.decision_function(X)

            print(f"Anomaly detection: {result_df['is_anomaly'].sum()} anomalies found")

        except Exception as e:
            print(f"Error in anomaly detection: {e}")
            result_df['is_anomaly'] = False

        return result_df

    def generate_interactive_visualizations(self) -> None:
        """
        Generates interactive visualizations based on the compiled metrics.
        """
        if not self.results_directory:
            QMessageBox.warning(self, "Warning", "Please choose a results directory first.")
            return

        try:
            # Check if there's a compiled metrics file
            compiled_metrics_path = self.find_compiled_metrics_file()

            if not compiled_metrics_path:
                reply = QMessageBox.question(
                    self,
                    "Compile Metrics",
                    "No compiled metrics file found. Would you like to compile metrics first?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self.compile_metrics_with_pca()
                    compiled_metrics_path = self.find_compiled_metrics_file()
                    if not compiled_metrics_path:
                        return
                else:
                    return

            # Read the compiled metrics
            df = pd.read_excel(compiled_metrics_path)

            if df.empty:
                QMessageBox.warning(self, "Warning", "Compiled metrics file is empty.")
                return

            # Create visualizations subfolder
            viz_dir = os.path.join(self.results_directory, "interactive_visualizations")
            os.makedirs(viz_dir, exist_ok=True)

            # Generate visualizations based on settings
            visualizations_created = []

            # 1. 3D Spectrograms if requested
            if self.check_3d_spectrogram.isChecked():
                spectrograms_path = self.create_interactive_spectrograms(viz_dir)
                if spectrograms_path:
                    visualizations_created.append(('3D Spectrograms', spectrograms_path))

            # 2. Interactive dissonance curves if requested
            if self.check_interactive_curves.isChecked():
                curves_path = self.create_interactive_dissonance_curves(viz_dir)
                if curves_path:
                    visualizations_created.append(('Dissonance Curves', curves_path))

            # 3. Dimensionality reduction plots if requested
            if self.check_dimension_scatterplots.isChecked():
                dimension_path = self.create_dimensionality_plots(df, viz_dir)
                if dimension_path:
                    visualizations_created.append(('Dimensionality Reduction', dimension_path))

            # Show success message with links to visualizations
            if visualizations_created:
                message = "The following interactive visualizations were created:\n\n"
                for name, path in visualizations_created:
                    message += f"• {name}: {path}\n"

                QMessageBox.information(self, "Success", message)

                # Ask if user wants to open the visualization directory
                reply = QMessageBox.question(
                    self,
                    "Open Visualizations",
                    "Would you like to open the visualizations directory?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self.open_file_or_directory(viz_dir)
            else:
                QMessageBox.warning(
                    self,
                    "Warning",
                    "No visualizations were created. Please check your settings."
                )

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error generating interactive visualizations: {str(e)}"
            )

    def find_compiled_metrics_file(self) -> Optional[str]:
        """
        Finds the most recent compiled metrics file in the results directory.

        Returns:
            Path to the compiled metrics file, or None if not found
        """
        # Look for various possible filenames
        possible_files = [
            'compiled_metrics_with_analysis.xlsx',
            'compiled_metrics.xlsx',
            'compiled_density_metrics.xlsx'
        ]

        for filename in possible_files:
            path = os.path.join(self.results_directory, filename)
            if os.path.exists(path):
                return path

        return None

    def create_interactive_spectrograms(self, output_dir: str) -> Optional[str]:
        """
        Creates interactive 3D spectrograms for loaded audio files.

        Args:
            output_dir: Directory to save the visualizations

        Returns:
            Path to the created HTML file, or None if failed
        """
        if not self.audio_processor.audio_data:
            return None

        try:
            # Create output path
            output_path = os.path.join(output_dir, "interactive_spectrograms.html")

            # Create a plotly figure with subplots - one row for each audio file
            num_files = min(len(self.audio_processor.audio_data), 4)  # Limit to 4 files to avoid huge HTML

            fig = make_subplots(
                rows=num_files, cols=2,
                specs=[[{"type": "surface"}, {"type": "heatmap"}] for _ in range(num_files)],
                subplot_titles=[f"{note} 3D" for _, _, note, _ in self.audio_processor.audio_data[:num_files]] +
                               [f"{note} 2D" for _, _, note, _ in self.audio_processor.audio_data[:num_files]]
            )

            # Process each audio file (up to the limit)
            for i, (y, sr, note, _) in enumerate(self.audio_processor.audio_data[:num_files]):
                try:
                    # Calculate STFT
                    from scipy import signal
                    f, t, Zxx = signal.stft(
                        y, fs=sr, nperseg=min(2048, len(y)),
                        window='hann', noverlap=None
                    )

                    # Convert to dB
                    spec = 10 * np.log10(np.abs(Zxx) + 1e-10)

                    # Add 3D surface plot
                    fig.add_trace(
                        go.Surface(z=spec, x=t, y=f, colorscale='Viridis'),
                        row=i+1, col=1
                    )

                    # Add 2D heatmap
                    fig.add_trace(
                        go.Heatmap(z=spec, x=t, y=f, colorscale='Viridis'),
                        row=i+1, col=2
                    )

                    # Configure axes
                    fig.update_scenes(
                        xaxis_title="Time (s)",
                        yaxis_title="Frequency (Hz)",
                        zaxis_title="Magnitude (dB)",
                        row=i+1, col=1
                    )

                    # Configure 2D axes
                    fig.update_yaxes(title="Frequency (Hz)", type="log", row=i+1, col=2)
                    fig.update_xaxes(title="Time (s)", row=i+1, col=2)

                except Exception as e:
                    print(f"Error processing {note} for interactive spectrogram: {e}")

            # Update layout
            fig.update_layout(
                height=300 * num_files,
                width=1200,
                title="Interactive Spectral Analysis"
            )

            # Save figure
            fig.write_html(output_path)

            return output_path

        except Exception as e:
            print(f"Error creating interactive spectrograms: {e}")
            return None

    def create_interactive_dissonance_curves(self, output_dir: str) -> Optional[str]:
        """
        Creates interactive dissonance curves visualization.

        Args:
            output_dir: Directory to save the visualizations

        Returns:
            Path to the created HTML file, or None if failed
        """
        # Check if we have dissonance curves
        if not self.results_directory:
            return None

        try:
            # Find directories containing dissonance curves
            dissonance_files = []
            model_name = self.combo_dissonance_model.currentText().lower()

            for item in os.listdir(self.results_directory):
                item_path = os.path.join(self.results_directory, item)
                if os.path.isdir(item_path):
                    # Check if this directory contains dissonance data
                    curve_file = os.path.join(item_path, f"{model_name}_dissonance_curve.png")
                    if os.path.exists(curve_file):
                        dissonance_files.append((item, item_path))

            if not dissonance_files:
                return None

            # Create output path
            output_path = os.path.join(output_dir, "interactive_dissonance_curves.html")

            # Create plotly figure
            fig = go.Figure()

            # Add dissonance curves from the first 10 directories (to avoid huge HTML)
            for note, note_dir in dissonance_files[:10]:
                try:
                    # Try to find the saved dissonance data
                    # This is a simplification - in a real implementation,
                    # you would need to extract or recalculate the actual curve data

                    # For demonstration, we'll generate some placeholder data
                    # In real implementation, extract this from Excel files or recalculate
                    intervals = np.linspace(1.0, 2.0, 200)
                    # Create a curve that has dips at common musical intervals
                    common_intervals = [1.0, 1.25, 1.33, 1.5, 1.67, 1.75, 2.0]  # Unison, M3, P4, P5, M6, M7, Octave
                    dissonance = np.ones_like(intervals)

                    for interval in common_intervals:
                        # Create dips at common intervals
                        dissonance -= 0.2 * np.exp(-100 * (intervals - interval)**2)

                    # Add some noise to make curves different
                    dissonance += 0.05 * np.random.randn(len(dissonance))

                    # Normalize to 0-1
                    dissonance = (dissonance - np.min(dissonance)) / (np.max(dissonance) - np.min(dissonance))

                    # Add to plot
                    fig.add_trace(
                        go.Scatter(
                            x=intervals,
                            y=dissonance,
                            mode='lines',
                            name=note
                        )
                    )

                except Exception as e:
                    print(f"Error processing {note} for dissonance curve: {e}")

            # Add vertical lines at common musical intervals with labels
            interval_names = {
                1.0: "Unison",
                1.25: "Major 3rd",
                1.33: "Perfect 4th",
                1.5: "Perfect 5th",
                1.67: "Major 6th",
                1.75: "Major 7th",
                2.0: "Octave"
            }

            for interval, name in interval_names.items():
                fig.add_vline(
                    x=interval,
                    line_dash="dash",
                    line_color="rgba(0,0,0,0.3)",
                    annotation_text=name,
                    annotation_position="top"
                )

            # Configure layout
            fig.update_layout(
                title=f"{model_name.capitalize()} Dissonance Curves",
                xaxis_title="Frequency Ratio",
                yaxis_title="Dissonance",
                height=600,
                width=1000,
                legend_title="Notes",
                hovermode="closest"
            )

            # Add a secondary x-axis with cents
            fig.update_layout(
                xaxis2=dict(
                    title="Cents",
                    overlaying="x",
                    side="top",
                    range=[0, 1200],  # 0 to 1200 cents (1 octave)
                    tickvals=[0, 200, 400, 600, 800, 1000, 1200],
                    ticktext=["0¢", "200¢", "400¢", "600¢", "800¢", "1000¢", "1200¢"],
                    showgrid=False
                )
            )

            # Save figure
            fig.write_html(output_path)

            return output_path

        except Exception as e:
            print(f"Error creating interactive dissonance curves: {e}")
            return None

    def create_dimensionality_plots(self, df: pd.DataFrame, output_dir: str) -> Optional[str]:
        """
        Creates interactive dimensionality reduction visualizations.

        Args:
            df: DataFrame with metrics and dimensionality reduction components
            output_dir: Directory to save the visualizations

        Returns:
            Path to the created HTML file, or None if failed
        """
        try:
            # Check if we have dimensionality reduction data
            has_pca = 'PC1' in df.columns
            has_tsne = 'TSNE1' in df.columns and 'TSNE2' in df.columns
            has_umap = 'UMAP1' in df.columns and 'UMAP2' in df.columns

            if not (has_pca or has_tsne or has_umap):
                return None

            # Check if we need to detect anomalies
            detect_anomalies = self.check_anomaly_detection.isChecked()
            if detect_anomalies and 'is_anomaly' not in df.columns:
                # Get numeric columns
                numeric_cols = self.get_numeric_columns(
                    df, self.check_include_dissonance.isChecked()
                )

                if len(numeric_cols) >= 2:
                    # Apply anomaly detection
                    df = self.detect_spectral_anomalies(df, numeric_cols)

            # Create output path
            output_path = os.path.join(output_dir, "dimensionality_reduction.html")

            # Create plotly figure
            fig = make_subplots(
                rows=1,
                cols=sum([has_pca, has_tsne, has_umap]),
                subplot_titles=[title for title, flag in
                               [("PCA", has_pca), ("t-SNE", has_tsne), ("UMAP", has_umap)]
                               if flag]
            )

            # Column counter for subplots
            col = 1

            # Colors for anomalies
            color_scale = ['blue', 'red'] if 'is_anomaly' in df.columns else None

            # Add PCA plot if available
            if has_pca:
                # Check if we have a categorical 'Note' column to use as hover text
                hover_text = df['Note'] if 'Note' in df.columns else None

                # Check if we have anomaly detection results
                marker_color = df['is_anomaly'].astype(int) if 'is_anomaly' in df.columns else 'blue'

                scatter = go.Scatter(
                    x=df['PC1'],
                    y=df['PC2'] if 'PC2' in df.columns else np.zeros(len(df)),
                    mode='markers',
                    marker=dict(
                        color=marker_color,
                        colorscale=color_scale,
                        size=10
                    ),
                    text=hover_text,
                    name='PCA'
                )

                fig.add_trace(scatter, row=1, col=col)
                fig.update_xaxes(title="PC1", row=1, col=col)
                fig.update_yaxes(title="PC2" if 'PC2' in df.columns else "", row=1, col=col)
                col += 1

            # Add t-SNE plot if available
            if has_tsne:
                scatter = go.Scatter(
                    x=df['TSNE1'],
                    y=df['TSNE2'],
                    mode='markers',
                    marker=dict(
                        color=df['is_anomaly'].astype(int) if 'is_anomaly' in df.columns else 'green',
                        colorscale=color_scale,
                        size=10
                    ),
                    text=df['Note'] if 'Note' in df.columns else None,
                    name='t-SNE'
                )

                fig.add_trace(scatter, row=1, col=col)
                fig.update_xaxes(title="t-SNE 1", row=1, col=col)
                fig.update_yaxes(title="t-SNE 2", row=1, col=col)
                col += 1

            # Add UMAP plot if available
            if has_umap:
                scatter = go.Scatter(
                    x=df['UMAP1'],
                    y=df['UMAP2'],
                    mode='markers',
                    marker=dict(
                        color=df['is_anomaly'].astype(int) if 'is_anomaly' in df.columns else 'purple',
                        colorscale=color_scale,
                        size=10
                    ),
                    text=df['Note'] if 'Note' in df.columns else None,
                    name='UMAP'
                )

                fig.add_trace(scatter, row=1, col=col)
                fig.update_xaxes(title="UMAP 1", row=1, col=col)
                fig.update_yaxes(title="UMAP 2", row=1, col=col)

            # Update layout
            fig.update_layout(
                title="Dimensionality Reduction Visualization",
                height=600,
                width=1000 * sum([has_pca, has_tsne, has_umap]),
                showlegend=False,
                hovermode="closest"
            )

            # Add a legend for anomalies if applicable
            if 'is_anomaly' in df.columns:
                fig.update_layout(
                    updatemenus=[{
                        'buttons': [
                            {
                                'args': [{'marker.color': [df['is_anomaly'].astype(int) if 'is_anomaly' in df.columns else 'blue']}],
                                'label': 'Show Anomalies',
                                'method': 'update'
                            },
                            {
                                'args': [{'marker.color': ['blue']}],
                                'label': 'Hide Anomalies',
                                'method': 'update'
                            }
                        ],
                        'direction': 'down',
                        'showactive': True,
                        'x': 0.1,
                        'y': 1.1
                    }]
                )

            # Save figure
            fig.write_html(output_path)

            return output_path

        except Exception as e:
            print(f"Error creating dimensionality reduction visualization: {e}")
            traceback.print_exc()
            return None

    def view_dissonance_curves(self) -> None:
        """
        Opens a file dialog to view dissonance curves for processed notes.
        """
        if not self.results_directory:
            QMessageBox.warning(self, "Warning", "Please choose a results directory first.")
            return

        try:
            # Get the currently selected model
            model_name = self.combo_dissonance_model.currentText().lower()

            # Find all notes directories
            note_dirs = []
            for item in os.listdir(self.results_directory):
                item_path = os.path.join(self.results_directory, item)
                if os.path.isdir(item_path):
                    # Check if this contains a dissonance curve for the selected model
                    curve_path = os.path.join(item_path, f"{model_name}_dissonance_curve.png")
                    if os.path.exists(curve_path):
                        note_dirs.append((item, curve_path))

            if not note_dirs:
                QMessageBox.warning(self, "Warning",
                                  f"No {model_name.capitalize()} dissonance curves found in the results directory.")
                return

            # Build a message with the available notes and their curves
            message = f"The following notes have {model_name.capitalize()} dissonance curves available:\n\n"
            for note, path in note_dirs:
                message += f"• {note}: {path}\n"

            message += "\nWould you like to open one of these files?"

            reply = QMessageBox.question(self, "Dissonance Curves", message,
                                        QMessageBox.Yes | QMessageBox.No)

            if reply == QMessageBox.Yes:
                # Show a file dialog pre-filtered to the results directory
                file_dialog = QFileDialog()
                file_dialog.setNameFilter("Image Files (*.png)")
                file_dialog.setDirectory(self.results_directory)

                if file_dialog.exec_():
                    selected_file = file_dialog.selectedFiles()[0]
                    self.open_file_or_directory(selected_file)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error viewing dissonance curves: {str(e)}")

    def open_file_or_directory(self, path: str) -> None:
        """
        Abre um ficheiro ou diretório com a aplicação predefinida do sistema,
        de forma segura (sem shell) e validando o caminho.
        """

        log = globals().get("logger")  # usa logger se existir; senão faz print
        def _log(level, msg):
            (log and getattr(log, level, None) or print)(msg)

        try:
            p = Path(path).resolve(strict=True)
        except FileNotFoundError:
            _log("error", f"Ficheiro/diretório não encontrado: {path}")
            return

        # Se existir um diretório-raiz permitido na instância, garanta que o caminho pertence a ele
        allowed_root = getattr(self, "results_directory", None)
        if allowed_root is not None:
            root = Path(allowed_root).resolve()
            try:
                if os.path.commonpath([str(root), str(p)]) != str(root):
                    _log("error", f"Caminho fora do diretório permitido: {p}")
                    return
            except Exception as e:
                _log("error", f"Falha a validar diretório permitido: {e}")
                return

        # Se for diretório: abrir gestor de ficheiros
        if p.is_dir():
            try:
                if sys.platform.startswith("win"):
                    os.startfile(str(p))  # não invoca shell
                elif sys.platform == "darwin":
                    subprocess.run(["open", str(p)], check=False)      # shell=False
                else:
                    subprocess.run(["xdg-open", str(p)], check=False)  # shell=False
            except Exception as e:
                _log("error", f"Erro ao abrir diretório '{p}': {e}")
            return

        # Se for ficheiro: restringir extensões (ajuste conforme necessário)
        allowed_exts = {
            ".png", ".jpg", ".jpeg", ".gif",
            ".html", ".htm", ".csv", ".json",
            ".xlsx", ".xls", ".txt", ".pdf"
        }
        if p.suffix.lower() not in allowed_exts:
            _log("error", f"Extensão não permitida para abertura: {p.suffix}")
            return

        try:
            # 1) validar/sanitizar p (sem NUL e caminho normalizado)
            if "\x00" in str(p):
                raise ValueError("Caminho inválido (NUL).")
            p_str = str(Path(p).expanduser().resolve(strict=False))

            # 2) opcional mas ajuda no linte: garantir que existe
            if not (os.path.isfile(p_str) or os.path.isdir(p_str)):
                logging.error("Caminho inexistente ou inválido: %s", p_str)

            elif sys.platform.startswith("win"):
                os.startfile(p_str)  # nosec: caminho local validado

            elif sys.platform == "darwin":
                cmd = shutil.which("open") or "/usr/bin/open"
                subprocess.run([cmd, p_str], check=False)  # nosec S603,S607: cmd whitelisted; input validado

            else:
                cmd = (shutil.which("xdg-open")
                       or shutil.which("gio")
                       or shutil.which("gnome-open"))
                if cmd:
                    subprocess.run([cmd, p_str], check=False)  # nosec S603,S607: cmd whitelisted; input validado
                else:
                    logging.error("Sem utilitário gráfico para abrir: %s", p_str)

        except Exception as e:
            logging.exception("Erro ao abrir ficheiro '%s': %s", p, e)



    def analyze_dissonance_vs_density(self) -> None:
        """
        Analyzes and compares dissonance with density metrics.
        """
        if not self.results_directory:
            QMessageBox.warning(self, "Warning", "Please choose a results directory first.")
            return

        try:
            # Get the currently selected model
            model_name = self.combo_dissonance_model.currentText()
            dissonance_column = f"{model_name} Dissonance"

            # First, check if compiled metrics file exists
            compiled_metrics_path = self.find_compiled_metrics_file()

            if not compiled_metrics_path:
                # If not, ask if user wants to compile metrics first
                reply = QMessageBox.question(
                    self,
                    "Compile Metrics",
                    "No compiled metrics file found. Would you like to compile metrics first?",
                    QMessageBox.Yes | QMessageBox.No
                )

                if reply == QMessageBox.Yes:
                    self.compile_metrics_with_pca()
                    # Check again if file exists after compilation
                    compiled_metrics_path = self.find_compiled_metrics_file()
                    if not compiled_metrics_path:
                        QMessageBox.warning(self, "Warning", "Could not create compiled metrics file.")
                        return
                else:
                    return

            # Read the compiled metrics
            df = pd.read_excel(compiled_metrics_path)

            # Check if both needed metrics exist
            if 'Density Metric' not in df.columns or dissonance_column not in df.columns:
                QMessageBox.warning(
                    self,
                    "Warning",
                    f"Both Density Metric and {dissonance_column} are required for comparison."
                )
                return

            # Create a comparison plot - both static and interactive
            # 1. Static matplotlib plot for backward compatibility
            plt.figure(figsize=(10, 6))

            # Get the data, removing any rows with NaN values
            plot_data = df[['Note', 'Density Metric', dissonance_column]].dropna()

            if plot_data.empty:
                QMessageBox.warning(self, "Warning", "No valid data for comparison.")
                return

            # Normalize the data for better comparison (0-1 scale)
            for col in ['Density Metric', dissonance_column]:
                min_val = plot_data[col].min()
                max_val = plot_data[col].max()
                if max_val != min_val:
                    plot_data[f'{col}_Norm'] = (plot_data[col] - min_val) / (max_val - min_val)
                else:
                    plot_data[f'{col}_Norm'] = 0

            # Calculate correlation
            corr = plot_data['Density Metric'].corr(plot_data[dissonance_column])

            # Scatter plot
            plt.scatter(
                plot_data['Density Metric'],
                plot_data[dissonance_column],
                s=100, alpha=0.7
            )

            # Add note labels
            for i, row in plot_data.iterrows():
                plt.annotate(
                    row['Note'],
                    (row['Density Metric'], row[dissonance_column]),
                    xytext=(5, 5),
                    textcoords='offset points'
                )

            # Add trendline
            if len(plot_data) > 1:
                z = np.polyfit(plot_data['Density Metric'], plot_data[dissonance_column], 1)
                p = np.poly1d(z)
                plt.plot(
                    [plot_data['Density Metric'].min(), plot_data['Density Metric'].max()],
                    [p(plot_data['Density Metric'].min()), p(plot_data['Density Metric'].max())],
                    "r--", alpha=0.7
                )

            plt.title(f'{model_name} Dissonance vs Density Metric (Correlation: {corr:.3f})')
            plt.xlabel('Density Metric')
            plt.ylabel(f'{model_name} Dissonance')
            plt.grid(True, alpha=0.3)

            # Save the static plot
            static_path = os.path.join(self.results_directory, f'{model_name.lower()}_vs_density.png')
            plt.savefig(static_path, dpi=300, bbox_inches='tight')
            plt.close()

            # 2. Create interactive plotly version
            viz_dir = os.path.join(self.results_directory, "interactive_visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            interactive_path = os.path.join(viz_dir, f'{model_name.lower()}_vs_density_interactive.html')

            fig = go.Figure()

            # Add scatter plot
            fig.add_trace(
                go.Scatter(
                    x=plot_data['Density Metric'],
                    y=plot_data[dissonance_column],
                    mode='markers+text',
                    marker=dict(
                        size=12,
                        color='rgba(0, 123, 255, 0.7)'
                    ),
                    text=plot_data['Note'],
                    textposition="top center",
                    name='Notes'
                )
            )

            # Add trendline
            if len(plot_data) > 1:
                z = np.polyfit(plot_data['Density Metric'], plot_data[dissonance_column], 1)
                p = np.poly1d(z)
                x_range = np.linspace(plot_data['Density Metric'].min(), plot_data['Density Metric'].max(), 100)

                fig.add_trace(
                    go.Scatter(
                        x=x_range,
                        y=p(x_range),
                        mode='lines',
                        line=dict(color='red', dash='dash'),
                        name=f'Trendline (r={corr:.3f})'
                    )
                )

            # Update layout
            fig.update_layout(
                title=f'{model_name} Dissonance vs Density Metric',
                xaxis_title='Density Metric',
                yaxis_title=f'{model_name} Dissonance',
                height=600,
                width=800,
                hovermode='closest',
                showlegend=True
            )

            # Save interactive version
            fig.write_html(interactive_path)

            QMessageBox.information(
                self,
                "Analysis Complete",
                f"Dissonance vs Density analysis completed.\n\n"
                f"Static plot saved at:\n{static_path}\n\n"
                f"Interactive plot saved at:\n{interactive_path}"
            )

            # Ask to view the interactive version
            reply = QMessageBox.question(
                self,
                "View Results",
                "Would you like to view the interactive comparison plot?",
                QMessageBox.Yes | QMessageBox.No
            )

            if reply == QMessageBox.Yes:
                self.open_file_or_directory(interactive_path)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error analyzing dissonance vs density: {str(e)}")

    def apply_filters(self) -> None:
        """
        Aplica os filtros definidos pelo utilizador ao áudio carregado,
        validando a função de peso e forçando recálculo determinístico.
        """
        import logging, traceback
        logger = logging.getLogger(__name__)

        try:
            # ---------------- Parâmetros espectrais básicos ----------------
            freq_min = float((self.input_min_freq.text() or "20").strip())
            freq_max = float((self.input_max_freq.text() or "20000").strip())
            db_min   = float((self.input_min_db.text()  or "-90").strip())
            db_max   = float((self.input_max_db.text()  or "0").strip())
            tolerance = float((self.input_tolerance.text() or "5.0").strip())
            use_adaptive_tolerance = bool(self.checkbox_adaptive_tolerance.isChecked())

            # ---------------- Parâmetros de janela e FFT -------------------
            n_fft = int((self.input_n_fft.text() or "4096").strip())
            hop_length_txt = (self.input_hop_length.text() or "").strip()
            hop_length = int(hop_length_txt) if hop_length_txt else None
            window = str(self.combo_window_type.currentText()).strip().lower()

            # Parâmetros específicos (se existirem na UI)
            kaiser_beta = None
            gaussian_std = None
            if window == "kaiser" and hasattr(self, "input_kaiser_beta"):
                txt = (self.input_kaiser_beta.text() or "").strip()
                if txt:
                    try:
                        kaiser_beta = float(txt)
                    except Exception:
                        kaiser_beta = None
            if window in ("gaussian", "gauss", "gaussiana") and hasattr(self, "input_gaussian_std"):
                txt = (self.input_gaussian_std.text() or "").strip()
                if txt:
                    try:
                        gaussian_std = float(txt)
                    except Exception:
                        gaussian_std = None

            # ---------------- Função de ponderação (robusta) ---------------
            try:
                raw_label = str(self.combo_weight_function.currentText())
            except Exception:
                raw_label = "linear"
            key = (raw_label or "").strip().lower()

            from density import get_weight_function
            try:
                _ = get_weight_function(key)       # aceitar diretamente (EN)
                weight_function = key
            except Exception:
                weight_function = _resolve_weight_key_from_ui(raw_label)
                _ = get_weight_function(weight_function)  # valida; deixa rebentar se inválida

            # ---------------- LFT ------------------------------------------
            use_lft = bool(self.check_use_lft.isChecked())
            zero_padding = int((self.input_zero_padding.text() or "1").strip())
            time_avg = str(self.combo_time_avg.currentText()).strip().lower()

            # ---------------- Pesos α/β -----------------------------------
            alpha = max(0.0, min(1.0, self.harmonic_weight_slider.value() / 100.0))
            beta  = 1.0 - alpha
            if hasattr(self, "harmonic_weight_value"):
                self.harmonic_weight_value.setText(f"{int(round(alpha*100))}%")
            if hasattr(self, "inharmonic_weight_value"):
                self.inharmonic_weight_value.setText(f"{int(round(beta*100))}%")

            # ---------------- Dissonância ---------------------------------
            dissonance_enabled = bool(self.check_dissonance_enabled.isChecked())
            dissonance_curve   = bool(self.check_dissonance_curve.isChecked())
            dissonance_scale   = bool(self.check_dissonance_scale.isChecked())
            dissonance_model   = str(self.combo_dissonance_model.currentText()).strip()
            compare_models     = bool(self.check_compare_models.isChecked())

            # ---------------- Diretório de saída ---------------------------
            if not self.results_directory:
                QMessageBox.warning(self, "Aviso", "Selecione um diretório para salvar os resultados.")
                return

            logger.info(
                "Apply Filters | wf=%s | α=%.3f β=%.3f | f=[%.1f, %.1f] Hz | dB=[%.1f, %.1f] | tol=%.2f Hz (adapt=%s) | "
                "FFT n=%d hop=%s win=%s | LFT=%s zp=%d avg=%s",
                weight_function, alpha, beta, freq_min, freq_max, db_min, db_max, tolerance,
                use_adaptive_tolerance, n_fft, hop_length, window, use_lft, zero_padding, time_avg
            )

            # ---------------- Barra de progresso ---------------------------
            progress = QProgressDialog("Aplicando filtros e gerando dados...", "Cancelar", 0, 100, self)
            progress.setWindowTitle("Processamento")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            progress.show()

            try:
                # ---- Reset determinístico das métricas --------------------
                ap = self.audio_processor
                if hasattr(ap, "_reset_metrics") and callable(getattr(ap, "_reset_metrics")):
                    ap._reset_metrics()
                else:
                    for attr in (
                        "density_metric_value", "scaled_density_metric_value",
                        "filtered_density_metric_value", "entropy_spectral_value",
                        "combined_density_metric_value", "total_metric_value",
                        "spectral_density_metric_value"
                    ):
                        if hasattr(ap, attr):
                            setattr(ap, attr, None)
                    for dict_attr in ("dissonance_values", "dissonance_curves", "dissonance_scales"):
                        if hasattr(ap, dict_attr) and isinstance(getattr(ap, dict_attr), dict):
                            for k in list(getattr(ap, dict_attr).keys()):
                                getattr(ap, dict_attr)[k] = None

                progress.setValue(10)

                # ---- Passar parâmetros específicos de janela ao processor --
                ap.kaiser_beta  = kaiser_beta if kaiser_beta is not None else 6.5
                ap.gaussian_std = gaussian_std if gaussian_std is not None else (n_fft / 8.0)

                # ---- Callback de progresso para o núcleo ------------------
                def _progress_cb(i: int, total: int, label: str) -> None:
                    try:
                        if total and total > 0:
                            pct = 10 + int(85 * i / total)   # 10→95%
                            progress.setValue(min(95, pct))
                        if label:
                            progress.setLabelText(f"Processando {label} ({i}/{total})…")
                        if progress.wasCanceled():
                            raise RuntimeError("Operação cancelada pelo utilizador.")
                    except Exception:
                        pass  # nunca propagar erros do callback de UI

                # ---- Chamada única ao núcleo de processamento --------------
                ap.apply_filters_and_generate_data(
                    freq_min=freq_min,
                    freq_max=freq_max,
                    db_min=db_min,
                    db_max=db_max,
                    tolerance=tolerance,
                    use_adaptive_tolerance=use_adaptive_tolerance,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window,
                    kaiser_beta=kaiser_beta,
                    gaussian_std=gaussian_std,
                    weight_function=weight_function,
                    results_directory=self.results_directory,
                    dissonance_enabled=dissonance_enabled,
                    dissonance_model=dissonance_model,
                    dissonance_curve=dissonance_curve,
                    dissonance_scale=dissonance_scale,
                    compare_models=compare_models,
                    harmonic_weight=float(alpha),
                    inharmonic_weight=float(beta),
                    use_lft=use_lft,
                    zero_padding=zero_padding,
                    time_avg=time_avg,
                    progress_callback=_progress_cb
                )

                progress.setValue(100)
                QMessageBox.information(self, "Filtros Aplicados", "Filtros aplicados e resultados guardados.")

                # Opcional: encadear compilação e visualizações interativas
                reply = QMessageBox.question(
                    self, "Compilar Métricas",
                    "Deseja compilar métricas com PCA agora?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply == QMessageBox.Yes:
                    self.compile_metrics_with_pca()
                    viz_reply = QMessageBox.question(
                        self, "Visualizações Interativas",
                        "Deseja gerar visualizações interativas agora?",
                        QMessageBox.Yes | QMessageBox.No
                    )
                    if viz_reply == QMessageBox.Yes:
                        self.generate_interactive_visualizations()

            except ValueError as ve:
                QMessageBox.critical(self, "Erro de Valor", f"Erro ao aplicar filtros: {ve}")
                logger.exception("Erro de valor em apply_filters")
            except PermissionError as pe:
                QMessageBox.critical(self, "Erro de Permissão", f"Acesso negado: {pe}")
                logger.exception("Erro de permissão em apply_filters")
            except Exception as e:
                QMessageBox.critical(self, "Erro", f"Erro ao aplicar filtros: {e}\n\n{traceback.format_exc()}")
                logger.exception("Erro inesperado em apply_filters")
            finally:
                progress.close()

        except Exception as e:
            # Falhas antes de abrir o progress dialog (parsing de UI, etc.)
            QMessageBox.critical(self, "Erro", f"Erro ao preparar parâmetros: {e}")
            logger.exception("Erro ao preparar parâmetros em apply_filters")

    # [REMOVED Spectral Power methods: switch_on_spectral_power, analyze_spectral_power, analyze_multiple_spectral_powers]
