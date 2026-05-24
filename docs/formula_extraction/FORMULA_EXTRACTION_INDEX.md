# Formula Extraction Index

## 1. Purpose

These documents are **Step 1** of the mathematical workflow: map **Python expressions → mathematical formulas** in compact tables. They record what the code computes in symbolic form; they are **not** full mathematical validation reports, proof obligations, or literature reviews.

## 2. Formula-extraction documents

| Pass | File | Topic | Main content | Status |
|---|---|---|---|---|
| 1 | [`FORMULA_EXTRACTION_TABLE_DENSITY_FIRST_PASS.md`](FORMULA_EXTRACTION_TABLE_DENSITY_FIRST_PASS.md) | Density metrics, first pass | Six core `density.py` metrics (entropy, \(D_{\mathrm{eff}}\), \(N_{\mathrm{eff}}\), discrete d3–d24, `apply_density_metric`, rolloff-compensated harmonic density). | Available |
| 2 | [`FORMULA_EXTRACTION_TABLE_PASS_02_WEIGHT_FUNCTIONS.md`](FORMULA_EXTRACTION_TABLE_PASS_02_WEIGHT_FUNCTIONS.md) | Weight functions | `WeightFunction` mappings (`linear`, `sqrt`, `log1p`, etc.) and `get_weight_function` key normalisation plus default rolloff constants. | Available |
| 3 | [`FORMULA_EXTRACTION_TABLE_PASS_03_PARTIAL_SUMS_AND_BUNDLES.md`](FORMULA_EXTRACTION_TABLE_PASS_03_PARTIAL_SUMS_AND_BUNDLES.md) | Partial sums and metric bundles | `band_partial_metric_sum`, `partial_metric_sums_h_i_s_total` (H/I/S/Total rules), and `compute_discrete_spectral_metrics_bundle`. | Available |
| 4 | [`FORMULA_EXTRACTION_TABLE_PASS_04_RESIDUAL_AND_INHARMONIC_CLASSIFICATION.md`](FORMULA_EXTRACTION_TABLE_PASS_04_RESIDUAL_AND_INHARMONIC_CLASSIFICATION.md) | Residual and inharmonic classification | Exclusion windows and masks in `identify_nonharmonic_residual_rows`; wrapper `identify_inharmonic_partials` has no standalone formulas. | Available |
| 5 | [`FORMULA_EXTRACTION_TABLE_PASS_05_HARMONIC_ALIGNMENT.md`](FORMULA_EXTRACTION_TABLE_PASS_05_HARMONIC_ALIGNMENT.md) | Harmonic alignment | Cents error, adaptive tolerance, harmonic windows, collapse, and energy-weighted summaries in `harmonic_alignment.py`. | Available |
| 6 | [`FORMULA_EXTRACTION_TABLE_PASS_06_PEAK_COMPONENT_COUNTS.md`](FORMULA_EXTRACTION_TABLE_PASS_06_PEAK_COMPONENT_COUNTS.md) | Peak component counts | Peak-list linear amplitude, tuples, and Hz-window harmonic vs inharmonic vs subbass counts in `peak_component_counts.py`. | Available |
| 7 | [`FORMULA_EXTRACTION_TABLE_PASS_07_LOW_FREQUENCY_POLICY.md`](FORMULA_EXTRACTION_TABLE_PASS_07_LOW_FREQUENCY_POLICY.md) | Low-frequency policy | Register margins, adaptive subfundamental cutoff, and row labels in `low_frequency_policy.py`. | Available |
| 8 | [`FORMULA_EXTRACTION_TABLE_PASS_08_SPECTRAL_LEAKAGE_GUARDS.md`](FORMULA_EXTRACTION_TABLE_PASS_08_SPECTRAL_LEAKAGE_GUARDS.md) | Spectral leakage guards | Leakage half-width in Hz and filtering of inharmonic peak candidates in `spectral_leakage_guards.py`. | Available |
| 9 | [`FORMULA_EXTRACTION_TABLE_PASS_09_COMPILE_TIME_NORMALISATION.md`](FORMULA_EXTRACTION_TABLE_PASS_09_COMPILE_TIME_NORMALISATION.md) | Compile-time normalisation | Canonical density fallback, global \([0,1]\) norm, per-component density, and weighted raw/normalised density columns in `compile_metrics.py`. | Available |
| 10 | [`FORMULA_EXTRACTION_TABLE_PASS_10_PROC_AUDIO_SELECTED_FORMULAS.md`](FORMULA_EXTRACTION_TABLE_PASS_10_PROC_AUDIO_SELECTED_FORMULAS.md) | Selected proc_audio formulas | RMS level scaling, STFT magnitude and frequency grid, coherent gain / window sum, peak amplitude calibration, Parseval audit, edge weights, f₀ robust fit, and component energy ratios (selected sites only). | Available |
| 11 | [`FORMULA_EXTRACTION_TABLE_PASS_11_DENSITY_EXTENDED_METRICS.md`](FORMULA_EXTRACTION_TABLE_PASS_11_DENSITY_EXTENDED_METRICS.md) | Extended density metrics | Additional `density.py` metrics (Bark / masking paths, complexity, combined metrics, `spectral_density`, etc.). | Available |
| 12 | [`FORMULA_EXTRACTION_TABLE_PASS_12_DISSONANCE_MODELS.md`](FORMULA_EXTRACTION_TABLE_PASS_12_DISSONANCE_MODELS.md) | Dissonance models | `dissonance_models.py` and related call sites (pairwise kernels, curves, compile export hooks). | Available |
| 13 | [`FORMULA_EXTRACTION_TABLE_PASS_13_PEAK_DETECTION_AND_F0_REFINEMENT.md`](FORMULA_EXTRACTION_TABLE_PASS_13_PEAK_DETECTION_AND_F0_REFINEMENT.md) | Peak detection and f₀ refinement | Selected `proc_audio.py` peak / parabolic / SNR / cents / note-frequency formulas. | Available |
| 14 | [`FORMULA_EXTRACTION_TABLE_PASS_14_COMPILE_EXTRACTION_AND_BATCH_MASS.md`](FORMULA_EXTRACTION_TABLE_PASS_14_COMPILE_EXTRACTION_AND_BATCH_MASS.md) | Compile extraction and batch mass | `compile_metrics.py` extraction/normalisation helpers; `finalize_batch_power_mass_summary` in `super_audio_analyzer.py`. | Available |
| 15 | [`FORMULA_EXTRACTION_TABLE_PASS_15_DATA_INTEGRITY_NORMALISATION.md`](FORMULA_EXTRACTION_TABLE_PASS_15_DATA_INTEGRITY_NORMALISATION.md) | Data integrity normalisation | `data_integrity.py` (IQR, robust normalisation, validation helpers, log transform). | Available |

## 3. Recommended order of use

1. **Pass 1** and **Pass 2** together define the core density constructions and the element-wise weight functions used in weighted sums.
2. **Pass 3** shows how per-component amplitudes are aggregated into H/I/S (and Total) and how discrete metric bundles are keyed for export.
3. **Passes 4–8** cover residual masking, alignment, peak-list classification, low-frequency policy, and leakage-aware gating—read in pass order when tracing a classification decision.
4. **Pass 9** applies after pipeline metrics exist: compile-time scaling and weighted-density columns on the compiled frame.
5. **Pass 10** then **Passes 11–15**: upstream `proc_audio.py` selections (10), extended density (11), dissonance (12), peak/f₀ refinement in `proc_audio` (13), compile/batch mass (14), and `data_integrity` normalisation (15).

## 4. Known notation clean-up

In `FORMULA_EXTRACTION_TABLE_DENSITY_FIRST_PASS.md`, the `np.abs(power)` row uses scalar absolute-value notation \(u_i = |x_i|\) (equivalent to \(\lVert x_i\rVert\) for real scalars).

## 5. What these files do not yet provide

- Literature justification or citations for modelling choices  
- Sensitivity or uncertainty analysis tied to every extracted row  
- Exhaustive, line-by-line coverage of `proc_audio.py` (Pass 10 and 13 remain partial by design)  
- Final thesis-ready prose (tables are reference artefacts, not narrative)

**Executable regression checks** for Passes **1–15** live under **`tests/formula_validation/`** (see **`docs/formula_validation/FORMULA_VALIDATION_PLAN_INDEX.md`**, **`TESTING.md`**, and **`docs/validation/VALIDATION_STATUS.md`**). Those tests implement the plans; they do not replace domain review.

## 6. Related documentation

- **`docs/formula_validation/FORMULA_VALIDATION_PLAN_INDEX.md`** — validation-plan index (fixtures / assertions).  
- **`tests/formula_validation/`** — pytest implementation.  
- **`docs/validation/VALIDATION_STATUS.md`** — formula-validation status note for Passes 1–15.  
- **`docs/validation/METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md`** — workflow and cautious interpretation.  
- **`CODE_FORMULA_TRACEABILITY_TABLE.md`** (repo root) — optional code ↔ formula traceability.  
- Legacy consolidated split: [`FORMULA_VALIDATION_PLAN.md`](FORMULA_VALIDATION_PLAN.md) (older umbrella; per-pass plans are authoritative for new work).
