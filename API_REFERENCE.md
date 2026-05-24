# API Reference
## SoundSpectrAnalyse - Complete API Documentation

**Package version:** 3.7.0 (`soundspectranalyse` in `pyproject.toml`)  
**Last updated:** May 2026

**Scope:** This page summarises the **classes and modules used by the active pipeline** (`proc_audio.AudioProcessor`, `density.py`, `compile_metrics`, orchestrator). The current final note-density architecture uses **`final_note_density_salience_weighted`** (primary final scalar) and **`final_note_density_count_based`** (control scalar). **`effective_partial_density`** remains a secondary/diagnostic participation descriptor, not the final note-density scalar. Older metric names (**Density Metric**, **Combined Density Metric**, **Spectral Density Metric**, **Filtered Density Metric**, **Weighted Combined Metric**, **R_norm**, **P_norm**, **D_agn**, **D_harm**) appear on per-note **`Legacy_Density_Metrics`**, **`Legacy_Compatibility`**, and **`Diagnostic_Metrics`**; they are **not** the canonical final-density contract. Research-only **`density_weighted_sum_cdm_mean`** is legacy opt-in and documented in **`docs/DENSITY_EXPORT_SCHEMA.md`** §R.

---

## Table of Contents

1. [Core Modules](#core-modules)
2. [Audio Processing](#audio-processing)
3. [Density Metrics](#density-metrics)
4. [Data Integrity](#data-integrity)
5. [Utilities](#utilities)

---

## Core Modules

### `proc_audio.py`

#### `AudioProcessor`

Main class for audio processing and spectral analysis.

**Initialization:**
```python
processor = AudioProcessor()
```

**Key Methods:**

##### `load_audio(file_path: str) -> None`
Load audio file for processing.

**Parameters:**
- `file_path` (str): Path to audio file (WAV, MP3, FLAC, etc.)

**Example:**
```python
processor.load_audio("path/to/audio.wav")
```

##### `fft_analysis(zero_padding: int = 1) -> None`
Perform FFT analysis on loaded audio.

**Parameters:**
- `zero_padding` (int): Zero padding factor (default: 1)

**Attributes Set:**
- `self.S`: Complex STFT matrix
- `self.freqs`: Frequency array
- `self.times`: Time array
- `self.spectral_flux`: Spectral flux metric
- `self.attack_time`: Attack time in seconds

**Example:**
```python
processor.fft_analysis(zero_padding=2)
```

##### `generate_complete_list() -> None`
Generate complete list of spectral components.

**Attributes Set:**
- `self.complete_list_df`: DataFrame with all spectral components

##### `_process_filtered_and_harmonic_data(...) -> None`
Process filtered and harmonic data.

**Parameters:**
- `freq_min` (float): Minimum frequency in Hz
- `freq_max` (float): Maximum frequency in Hz
- `db_min` (float): Minimum magnitude in dB
- `db_max` (float): Maximum magnitude in dB
- `tolerance` (float): Harmonic tolerance in Hz
- `note` (str): Note name (e.g., "A4")
- `zero_padding` (int): Zero padding factor
- `time_avg` (str): Time averaging method ('mean', 'max', etc.)

##### `_calculate_metrics() -> None`
Calculate all density metrics (canonical + legacy).

**Attributes Set (selection):**
- `self.effective_partial_density`: participation-ratio fatness \(D_{\mathrm{eff}}\) on the effective-component power vector (exported on **`Density_Metrics`**)
- `self.harmonic_energy_sum`, `self.inharmonic_energy_sum`, `self.subbass_energy_sum`, `self.total_component_energy`, energy ratios, `harmonic_order_count`, `spectral_entropy` (see `docs/DENSITY_EXPORT_SCHEMA.md`)
- `self.density_metric_value`: legacy Density Metric (amplitude-weighted)
- `self.spectral_density_metric_value`: legacy spectral-density scalar (wide metrics)
- `self.combined_density_metric_value`: legacy Combined Density Metric
- `self.total_metric_value`: legacy Total Metric

##### `save_results() -> None`
Save analysis results for a single note folder: **`spectral_analysis.xlsx`**, standard plots (e.g. **`spectrogram.png`**), and **component balance** PNGs.

**Output (typical artefacts next to the workbook):**
- **`spectral_analysis.xlsx`** — per-note multi-sheet export (includes **`Analysis_Metadata`**).
- **`component_amplitude_mass_pie.png`** — diagnostic pie built from **`linear_sum_amplitude_*`** (linear amplitude sums; **not** the same as power/energy ratios; title and footnote state the basis).
- **`component_energy_ratio_pie.png`** — pie built from **`harmonic_energy_ratio`**, **`inharmonic_energy_ratio`**, **`subbass_energy_ratio`** when available.
- **`component_energy_pie.png`** — legacy **copy** of the amplitude-mass chart (backward-compatible filename only).

See **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`** and **`docs/DENSITY_EXPORT_SCHEMA.md`** §J.

---

### `density.py`

#### `apply_density_metric(values, weight_function='linear', prevent_domination=True, ...) -> float`

Calculate density metric from amplitude values.

**Parameters:**
- `values` (np.ndarray): Array of amplitude values
- `weight_function` (str): Weight function ('linear', 'log', 'sqrt', 'cbrt', 'exp', 'inverse log', 'sum', 'quadratic')
- `prevent_domination` (bool): Normalize by max to prevent single partial domination (default: True)
- `frequencies` (np.ndarray, optional): Frequency array for frequency-dependent normalization
- `fundamental_freq` (float, optional): Fundamental frequency for frequency-dependent normalization
- `account_for_spectral_rolloff` (bool): Account for natural spectral rolloff (default: False)

**Returns:**
- `float`: Density metric value

**Example:**
```python
from density import apply_density_metric
import numpy as np

amplitudes = np.array([0.5, 0.3, 0.2, 0.1])
density = apply_density_metric(amplitudes, weight_function='log')
```

**Note:** programmatic defaults and docs use `linear` first; Tk/Qt
analysis GUIs also initialise the combo to **Linear**. The `log`
example above is for comparing dynamic-range compression on the same
array.

#### `calculate_combined_density_metric(harmonic_density, inharmonic_density, alpha=0.8, beta=0.2, preserve_dynamic_range=True) -> float`

Calculate combined density metric from harmonic and inharmonic components.

**Parameters:**
- `harmonic_density` (float): Harmonic density value
- `inharmonic_density` (float): Inharmonic density value
- `alpha` (float): Weight for harmonic component (default: 0.8)
- `beta` (float): Weight for inharmonic component (default: 0.2)
- `preserve_dynamic_range` (bool): Use logarithmic combination (default: True)

**Returns:**
- `float`: Combined density metric value

**Example:**
```python
from density import calculate_combined_density_metric

combined = calculate_combined_density_metric(
    harmonic_density=5.0,
    inharmonic_density=1.0,
    alpha=0.8,
    beta=0.2,
    preserve_dynamic_range=True
)
```

#### `calculate_perceptual_spectral_density(amplitudes, frequencies, fundamental_freq, threshold_db=-60.0) -> float`

Calculate perceptual spectral density using 24 critical bands.

**Parameters:**
- `amplitudes` (np.ndarray): Array of amplitudes
- `frequencies` (np.ndarray): Array of frequencies in Hz
- `fundamental_freq` (float): Fundamental frequency in Hz
- `threshold_db` (float): Detection threshold in dB (default: -60.0)

**Returns:**
- `float`: Perceptual spectral density (0-1)

**Example:**
```python
from density import calculate_perceptual_spectral_density
import numpy as np

amplitudes = np.array([0.5, 0.3, 0.2])
frequencies = np.array([440, 880, 1320])
density = calculate_perceptual_spectral_density(
    amplitudes, frequencies, fundamental_freq=440.0
)
```

---

### `data_integrity.py`

#### `robust_normalize(data, method='iqr', clip_range=(0.0, 1.0), ...) -> np.ndarray`

Robust normalization using IQR or percentile methods.

**Parameters:**
- `data` (np.ndarray): Input data array
- `method` (str): Normalization method ('iqr', 'percentile', 'robust_zscore')
- `clip_range` (tuple): Clipping range (default: (0.0, 1.0))
- `iqr_multiplier` (float): IQR multiplier (default: 1.5)
- `percentile_low` (float): Lower percentile (default: 5.0)
- `percentile_high` (float): Upper percentile (default: 95.0)

**Returns:**
- `np.ndarray`: Normalized data array

**Example:**
```python
from data_integrity import robust_normalize
import numpy as np

data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0])  # Contains outlier
normalized = robust_normalize(data, method='iqr')
```

#### `normalize_log_transform(data, clip_range=(0.0, 1.0), epsilon=1e-10) -> np.ndarray`

Log-transform normalization preserving dynamic range.

**Parameters:**
- `data` (np.ndarray): Input data array
- `clip_range` (tuple): Clipping range (default: (0.0, 1.0))
- `epsilon` (float): Small value to avoid log(0) (default: 1e-10)

**Returns:**
- `np.ndarray`: Normalized data array

**Example:**
```python
from data_integrity import normalize_log_transform
import numpy as np

data = np.array([0.01, 0.1, 1.0, 10.0, 100.0])  # Wide dynamic range
normalized = normalize_log_transform(data)
```

#### `detect_outliers(data, iqr_multiplier=1.5, return_mask=False) -> np.ndarray`

Detect outliers using IQR method.

**Parameters:**
- `data` (np.ndarray): Input data array
- `iqr_multiplier` (float): IQR multiplier (default: 1.5)
- `return_mask` (bool): Return boolean mask instead of values (default: False)

**Returns:**
- `np.ndarray`: Outlier values or (outliers, mask) if return_mask=True

**Example:**
```python
from data_integrity import detect_outliers
import numpy as np

data = np.array([1.0, 2.0, 3.0, 100.0, 4.0, 5.0])
outliers = detect_outliers(data)
# Returns: array([100.0])
```

---

### `compile_metrics.py`

#### `compile_density_metrics_with_pca(folder_path, output_path=None, file_pattern='spectral_analysis.xlsx', ...) -> pd.DataFrame`

**Canonical Stage 2** compiler: builds the compiled workbook (multi-sheet, including `Density_Metrics`, `Canonical_Metrics`, …) from per-note **`spectral_analysis.xlsx`** files. This is the function the orchestrator and tests treat as the normative export path.

Legacy **`compile_density_metrics(...)`** may still exist for compatibility; new code and documentation should call **`compile_density_metrics_with_pca`** unless there is a specific reason not to.

**Parameters (representative):**
- `folder_path` (str | Path): Folder tree containing per-note `spectral_analysis.xlsx`
- `output_path` (str | Path, optional): Output Excel path
- `file_pattern` (str): Glob/file pattern (default `'spectral_analysis.xlsx'`)
- PCA / DR flags as implemented in `compile_metrics.py`

**Returns:** `pd.DataFrame` — wide compiled frame in memory (sheets are written via the internal Excel writer).

**`weight_function` (compile):** passed through to per-note extraction. Sets per-band `harmonic_density_sum` / `inharmonic_density_sum` / `subbass_density_sum` and therefore **`density_weighted_sum`** (= **`density_metric_raw`** = \(D_H w_H + D_I w_I + D_S w_S\)). Linear **`harmonic_amplitude_sum`** columns are independent of this key. See **`docs/DENSITY_EXPORT_SCHEMA.md`** §C.1.

**Example:**
```python
from compile_metrics import compile_density_metrics_with_pca
from pathlib import Path

df = compile_density_metrics_with_pca(
    folder_path=Path("./results"),
    output_path="compiled_density_metrics.xlsx",
    file_pattern="spectral_analysis.xlsx",
    enable_pca_export=False,
)
```

#### `compile_density_metrics(...)` (legacy)

Older entry point with a smaller default feature set; prefer **`compile_density_metrics_with_pca`** for documentation and new scripts.

---

## Utilities

### `_verify_energy_conservation(y_time, S_freq, n_fft, hop_length, window, tolerance=0.1, ...) -> dict`

Verify energy conservation using Parseval's theorem.

**Parameters:**
- `y_time` (np.ndarray): Time-domain signal
- `S_freq` (np.ndarray): STFT matrix (complex)
- `n_fft` (int): FFT size
- `hop_length` (int): Hop length
- `window` (str | np.ndarray): Window type or array
- `tolerance` (float): Acceptable deviation (default: 0.1)
- `window_array` (np.ndarray, optional): Actual window array

**Returns:**
- `dict`: Dictionary with:
  - `energy_ratio` (float): Energy ratio (should be ~1.0)
  - `is_valid` (bool): Whether within tolerance
  - `deviation` (float): Deviation percentage

**Example:**
```python
from proc_audio import _verify_energy_conservation
import librosa
import numpy as np

y = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 44100))
S = librosa.stft(y, n_fft=4096, hop_length=1024, window='hann')
result = _verify_energy_conservation(y, S, 4096, 1024, 'hann')
print(f"Energy ratio: {result['energy_ratio']:.4f}, Valid: {result['is_valid']}")
```

### `_calculate_window_characteristics(win, n_fft) -> dict`

Calculate window characteristics: main-lobe width and side-lobe level.

**Parameters:**
- `win` (str): Window type name
- `n_fft` (int): FFT size

**Returns:**
- `dict`: Dictionary with:
  - `main_lobe_width` (float): Main lobe width in bins
  - `side_lobe_level` (float): Side lobe level in dB
  - `peak_level` (float): Peak level in dB

**Example:**
```python
from proc_audio import _calculate_window_characteristics

chars = _calculate_window_characteristics('hann', 4096)
print(f"Main lobe width: {chars['main_lobe_width']:.2f} bins")
print(f"Side lobe level: {chars['side_lobe_level']:.2f} dB")
```

### Research export: `tools.export_research_density_workbook.export_research_workbook`

Build **`compiled_density_metrics_research.xlsx`** from an existing compiled workbook (read-only input).

**Signature (summary):** `export_research_workbook(input_path, output_path=None, *, overwrite=False, no_charts=False, instrument=None, dynamic=None, force_metadata=False, research_metadata=None) -> pathlib.Path`

**Behaviour notes**

- Optional **`--instrument`**, **`--dynamic`**, **`--force-metadata`** (CLI) or equivalent keyword arguments.  
- Output uses **worksheet AutoFilter** on tabular sheets only—not Excel **Table** XML—so desktop Excel should open without removing table/autofilter features.  
- Merges compiled **`Legacy_Compatibility`** so **`Combined Density Metric`** is available when Stage 1 wrote **`Legacy_Density_Metrics`**.  
- On **`Spectral_Density_Metrics`**: adds **`density_weighted_sum_cdm_mean`** = \((\texttt{density\_weighted\_sum}+\texttt{Combined Density Metric})/2\); soft column highlights on **`density_weighted_sum`**, **`Combined Density Metric`**, and the mean (research workbook only).  
- See **`README.md`**, **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`** §9, and **`docs/DENSITY_EXPORT_SCHEMA.md`** §R.

### Per-note export: `Legacy_Density_Metrics` (default ON)

Written by **`AudioProcessor._build_legacy_density_metrics_row`** into every **`spectral_analysis.xlsx`**:

| Column | Attribute |
|--------|-----------|
| `Density Metric` | `canonical_density_v5_adapted` / `density_metric_value` |
| `Spectral Density Metric` | `spectral_density_metric_value` |
| `Filtered Density Metric` | `filtered_density_metric_value` |
| `Combined Density Metric` | `combined_density_metric_value` |

Merged at compile by **`compile_metrics.read_excel_metrics`**. Feeds **`apply_weighted_combination`** → **`Weighted Combined Metric`** on **`Diagnostic_Metrics`** / **`Legacy_Compatibility`**.

---

## Constants

### `constants.py`

Key constants used throughout the codebase:

- `DEFAULT_N_FFT`: Default FFT size (4096)
- `DEFAULT_HOP_LENGTH`: Default hop length (1024)
- `DEFAULT_WINDOW`: Default window type ('hann')
- `ENERGY_CONSERVATION_TOLERANCE`: Energy conservation tolerance (0.1)
- `NUM_CRITICAL_BANDS`: Number of critical bands (24)
- `MAX_ABS_DENSITY`: Maximum absolute density (20.0)
- `NORMALIZATION_TARGET_RMS_DB`: Target RMS level (-20.0 dB)

---

## Usage Examples

### Complete Workflow

```python
from proc_audio import AudioProcessor
from pathlib import Path

# Initialize processor
processor = AudioProcessor()

# Load audio
processor.load_audio("path/to/audio.wav")

# Configure parameters
processor.n_fft = 4096
processor.hop_length = 1024
processor.window = 'hann'
processor.freq_min = 20.0
processor.freq_max = 20000.0
processor.db_min = -80.0
processor.db_max = 0.0
processor.tolerance = 10.0

# Perform analysis
processor.fft_analysis(zero_padding=2)
processor.generate_complete_list()
processor._process_filtered_and_harmonic_data(
    freq_min=20.0, freq_max=20000.0,
    db_min=-80.0, db_max=0.0,
    tolerance=10.0, note="A4"
)
processor._calculate_metrics()

# Save results
processor.results_directory = Path("./results")
processor.save_results()

# Access metrics
print(f"Density Metric: {processor.density_metric_value}")
print(f"Spectral Density Metric: {processor.spectral_density_metric_value}")
print(f"Combined Density Metric: {processor.combined_density_metric_value}")
```

---

**Last updated:** May 2026 (package 3.7.0; includes research export API note)

