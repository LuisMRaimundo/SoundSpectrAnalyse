# Formula Validation Plan Index

## 1. Purpose

These documents define **small, hand-checkable numerical examples** so each case can confirm that **Python outputs match the formulas** recorded in the formula-extraction tables (**Passes 1–15**). The markdown files are **validation plans only** (they do not execute tests or modify application code). The corresponding **pytest** modules live under **`tests/formula_validation/`** and implement the plans for **Passes 1–15**.

**Cautious interpretation:** the formula-validation corpus supports **internal consistency** between the documented mathematical formulas and the tested Python implementations. It verifies formula/code agreement for **selected numerical fixtures**. It does **not**, by itself, prove scientific optimality, universal correctness, or full acoustic validity of the models. See **`METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md`** (repo root) and **`VALIDATION_STATUS_812_PASSED_PASSES_1_15.md`** for a recorded cycle summary.

## 2. Documents

| Pass | File | Topic | Priority | Notes |
|---|---|---|---|---|
| 1 | [`FORMULA_VALIDATION_PLAN_PASS_01_DENSITY_METRICS.md`](FORMULA_VALIDATION_PLAN_PASS_01_DENSITY_METRICS.md) | Density metrics | Highest | Core `density.py` metrics: entropy, \(D_{\mathrm{eff}}\), \(N_{\mathrm{eff}}\), d3/d10/d17, rolloff density. |
| 2 | [`FORMULA_VALIDATION_PLAN_PASS_02_WEIGHT_FUNCTIONS.md`](FORMULA_VALIDATION_PLAN_PASS_02_WEIGHT_FUNCTIONS.md) | Weight functions | Highest | `WeightFunction` and `get_weight_function` aliases. |
| 3 | [`FORMULA_VALIDATION_PLAN_PASS_03_PARTIAL_SUMS_AND_BUNDLES.md`](FORMULA_VALIDATION_PLAN_PASS_03_PARTIAL_SUMS_AND_BUNDLES.md) | Partial sums and metric bundles | High | `band_partial_metric_sum`, H/I/S totals, d10 concatenation rule, discrete bundle. |
| 4 | [`FORMULA_VALIDATION_PLAN_PASS_04_RESIDUAL_AND_INHARMONIC_CLASSIFICATION.md`](FORMULA_VALIDATION_PLAN_PASS_04_RESIDUAL_AND_INHARMONIC_CLASSIFICATION.md) | Residual and inharmonic classification | Medium | `identify_nonharmonic_residual_rows` masking. |
| 5 | [`FORMULA_VALIDATION_PLAN_PASS_05_HARMONIC_ALIGNMENT.md`](FORMULA_VALIDATION_PLAN_PASS_05_HARMONIC_ALIGNMENT.md) | Harmonic alignment | Medium | Cents, adaptive tolerance, expected slot count. |
| 6 | [`FORMULA_VALIDATION_PLAN_PASS_06_PEAK_COMPONENT_COUNTS.md`](FORMULA_VALIDATION_PLAN_PASS_06_PEAK_COMPONENT_COUNTS.md) | Peak component counts | Medium | dB→linear, Hz tolerance from cents, peak-list classification counts. |
| 7 | [`FORMULA_VALIDATION_PLAN_PASS_07_LOW_FREQUENCY_POLICY.md`](FORMULA_VALIDATION_PLAN_PASS_07_LOW_FREQUENCY_POLICY.md) | Low-frequency policy | Medium | Register margins (35 / 25 / 15 / 10% for \(f_0<\)60 / 120 / 300 Hz), adaptive cutoff dict, row labels. |
| 8 | [`FORMULA_VALIDATION_PLAN_PASS_08_SPECTRAL_LEAKAGE_GUARDS.md`](FORMULA_VALIDATION_PLAN_PASS_08_SPECTRAL_LEAKAGE_GUARDS.md) | Spectral leakage guards | Medium | Half-width Hz and candidate filtering. |
| 9 | [`FORMULA_VALIDATION_PLAN_PASS_09_COMPILE_TIME_NORMALISATION.md`](FORMULA_VALIDATION_PLAN_PASS_09_COMPILE_TIME_NORMALISATION.md) | Compile-time normalisation | Medium / high | Canonical fallback, global norm, per-component and weighted raw density columns. |
| 10 | [`FORMULA_VALIDATION_PLAN_PASS_10_PROC_AUDIO_SELECTED_FORMULAS.md`](FORMULA_VALIDATION_PLAN_PASS_10_PROC_AUDIO_SELECTED_FORMULAS.md) | Selected proc_audio formulas | Lower (for now) | RMS/gain, window calibration, Parseval audit, \(f_0\) LS—more **environment-sensitive** (STFT / `librosa`). |
| 11 | [`FORMULA_VALIDATION_PLAN_PASS_11_DENSITY_EXTENDED_METRICS.md`](FORMULA_VALIDATION_PLAN_PASS_11_DENSITY_EXTENDED_METRICS.md) | Extended density metrics | Medium / high | Selected `density.py` extended metrics and bundles per plan. |
| 12 | [`FORMULA_VALIDATION_PLAN_PASS_12_DISSONANCE_MODELS.md`](FORMULA_VALIDATION_PLAN_PASS_12_DISSONANCE_MODELS.md) | Dissonance models | Medium | Toy partials and model kernels (`dissonance_models.py`). |
| 13 | [`FORMULA_VALIDATION_PLAN_PASS_13_PEAK_DETECTION_AND_F0_REFINEMENT.md`](FORMULA_VALIDATION_PLAN_PASS_13_PEAK_DETECTION_AND_F0_REFINEMENT.md) | Peak detection and f₀ refinement | Medium | `proc_audio.py` parabolic / prominence / cents / spacing cases per plan. |
| 14 | [`FORMULA_VALIDATION_PLAN_PASS_14_COMPILE_EXTRACTION_AND_BATCH_MASS.md`](FORMULA_VALIDATION_PLAN_PASS_14_COMPILE_EXTRACTION_AND_BATCH_MASS.md) | Compile extraction and batch mass | Medium | `compile_metrics.py` and batch mass helpers per plan. |
| 15 | [`FORMULA_VALIDATION_PLAN_PASS_15_DATA_INTEGRITY_NORMALISATION.md`](FORMULA_VALIDATION_PLAN_PASS_15_DATA_INTEGRITY_NORMALISATION.md) | Data integrity normalisation | Medium | `data_integrity.py` IQR, robust normalisation, validation helpers per plan. |

**Suggested implementation order (historical):** Passes 1 → 2 → 3, then 4–8, then 9, then 10–15 as fixture complexity grows.

## 3. Boundary

The plan markdown files **do not** run tests and **do not** modify source code; they list validation examples and suggested assertions. **Executable tests** for Passes **1–15** are maintained under **`tests/formula_validation/`** (see **`TESTING.md`**).

The consolidated source split from the extraction tree is still available as [`../formula_extraction/FORMULA_VALIDATION_PLAN.md`](../formula_extraction/FORMULA_VALIDATION_PLAN.md).

## 4. Related (repo root)

- [`VALIDATION_STATUS_812_PASSED_PASSES_1_15.md`](../../VALIDATION_STATUS_812_PASSED_PASSES_1_15.md) — recorded pytest counts (full suite and formula-validation subset).  
- [`METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md`](../../METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md) — methodological note.  
- [`CODE_FORMULA_TRACEABILITY_TABLE.md`](../../CODE_FORMULA_TRACEABILITY_TABLE.md) — optional traceability table.  
- [`../formula_extraction/FORMULA_EXTRACTION_INDEX.md`](../formula_extraction/FORMULA_EXTRACTION_INDEX.md) — formula-extraction index.
