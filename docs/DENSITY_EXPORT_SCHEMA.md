# Density export schema (scientific reference)

Compiled workbooks separate **spectral density / fatness** (`Density_Metrics`), **measured energy accounting**, **dissonance**, **exploratory PCA**, **per-note processing provenance**, and **technical audit counts** (`Debug_Counts`, `Validation_Metrics`). This document matches the multi-sheet Excel layout produced by `compile_metrics._write_compiled_excel` and per-note `proc_audio` exports.

**Production path:** `run_orchestrator.py` (or `soundspectranalyse` after `pip install -e .`) → `pipeline_orchestrator_integrated.py` → per-note `spectral_analysis.xlsx` → aggregate **`compiled_density_metrics.xlsx`** with the sheets described here.

**Canonical vs legacy inputs:** **Stage 2** publication compilation is defined on folders of **`spectral_analysis.xlsx`** produced by **`proc_audio.AudioProcessor`**. Some **rolloff** and **harmonic effective power density (HEpd)** public columns may still be **filled from a sibling `super_analysis_results.json`** when that JSON exists and satisfies the Phase‑1 contract — this is an **optional discovery path**, not the definition of canonical Stage 1. When JSON is absent or invalid, values already stored in **`spectral_analysis.xlsx`** are used (`Analysis_Metadata` records discovery status). See **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`** for the normative pipeline narrative.

---

## A. Main metric — `effective_partial_density`

**`effective_partial_density`** \(D_{\mathrm{eff}}\) estimates how many energetically relevant spectral components contribute to the measured **power** distribution, using the participation-ratio (Herfindahl-inverse) form on the **effective-component power vector**:

\[
D_{\mathrm{eff}} = \frac{\left(\sum_i P_i\right)^2}{\sum_i P_i^2}
\]

with \(P_i \ge 0\) in the **power** domain.

**Component policy (must match `density.partial_density_effective_components_bundle`):**

- **Harmonics:** individual detected harmonic powers (per partial where supported).
- **Inharmonic:** default **single aggregated** power term (`inharmonic_mode_for_effective_density = aggregate`); the metric must **not** treat every inharmonic FFT row as an independent partial.
- **Sub-bass / sub-noise:** default **single aggregated** term (`subbass_policy_for_effective_density`).
- **Scale invariance:** multiplying all \(P_i\) by a positive constant leaves \(D_{\mathrm{eff}}\) unchanged — it is **not** a loudness meter.
- **Not:** sensory dissonance, PCA, spectral masking, or rhythmic “density”.

Metadata keys: `density_formula`, `effective_density_component_policy`, `inharmonic_mode_for_effective_density`, `subbass_policy_for_effective_density`, `rms_normalisation_enabled`.

---

## B. Energy ratios vs model weights

**Measured acoustic ratios** (per analysed note, from the spectral energy decomposition):

- `harmonic_energy_ratio`, `inharmonic_energy_ratio`, `subbass_energy_ratio` — partition of **component energy** among harmonic / inharmonic / sub-bass pools (approximately sum to 1).
- `harmonic_energy_sum`, `inharmonic_energy_sum`, `subbass_energy_sum`, `total_component_energy` — unnormalised power sums in the same decomposition frame.

**Batch layer (`batch_summary.xlsx`):** empirical profiles for instrument / note / dynamic contexts. **Canonical batch ratios** — **`batch_harmonic_energy_ratio`**, **`batch_inharmonic_energy_ratio`**, **`batch_subbass_energy_ratio`** — are **global H+I+S energy fractions** from **`harmonic_energy_sum + inharmonic_energy_sum + subbass_energy_sum`** (same denominator for all three; they **sum to 1** when that total is positive). Legacy **`harmonic_energy_percentage` / `inharmonic_energy_percentage`** remain display / alignment columns and may follow different conventions; see batch row **`harmonic_energy_percentage_semantics`**. **`batch_total_inharmonic_energy_ratio`** is **(I+S)/total** on that same energy-sum denominator.

**Model weights** (formal coefficients in weighted combinations, e.g. log-blend of harmonic vs inharmonic density **metrics**):

- `model_harmonic_weight`, `model_inharmonic_weight` — passed into `AudioProcessor` as `harmonic_weight` / `inharmonic_weight` for the combined metric path. These are **`H/(H+I)`** and **`I/(H+I)`** projections from the validated global profile — **not** the same numbers as the global **H/(H+I+S)** fractions when subbass is nonzero.
- `model_weights_source` — documents origin, e.g. `apply_filters_arguments`, `batch_empirical_energy_ratios` (when the orchestrator derives weights from validated batch ratios), or `fixed_fallback`.

**Rule:** `harmonic_energy_ratio = 0.983` means “about 98.3% of classified component energy is harmonic.” That is **not** the same statement as `model_harmonic_weight = 0.95`.

### GUI controls and model weights

- **Spectral masking** is **not** part of the physical / spectral density model. The desktop **Spectrum Analyzer** main workflow keeps **`spectral_masking_enabled = False`**; per-note metadata records that value. Masking does not affect **`effective_partial_density`**, measured per-note energy ratios, count semantics, or **`Density_Metrics`**.
- **`harmonic_weight` / `inharmonic_weight`** (also exported as **`model_harmonic_weight` / `model_inharmonic_weight`** in `Analysis_Metadata`) are **model coefficients** for legacy / combined metric paths (e.g. log-blends), **not** the raw empirical **`batch_*_energy_ratio`** columns.
- When a valid **`batch_summary.xlsx`** profile is loaded for the current note/file, the GUI derives **α = H/(H+I)**, **β = I/(H+I)** (same rule as **`RobustOrchestrator`**) and uses those as the applied weights unless the user enables **advanced manual override**.
- **Manual override** (optional checkbox) sets **`model_weights_source = manual_override`**, records **`manual_model_harmonic_weight` / `manual_model_inharmonic_weight`**, and emits a clear **warning** that the run is **not** the empirical batch-derived workflow. Manual coefficients must **not** be labelled as empirical H/I/S ratios.
- **`effective_partial_density`** is computed from the **effective-component power vector** only; it does **not** follow the harmonic/inharmonic **slider** or manual weight UI.

---

## C. `Density_Metrics` (public density sheet)

Preferred columns:

- `Note`
- `canonical_density_v5_adapted`, `density_normalized_global`, `density_metric_normalized` (alias), `density_per_component`, `density_source_formula`, `density_normalization_scope`, `density_normalization_denominator`, `density_formula_version`
- `effective_partial_density`
- `harmonic_energy_sum`, `inharmonic_energy_sum`, `subbass_energy_sum`, `total_component_energy`
- `harmonic_energy_ratio`, `inharmonic_energy_ratio`, `subbass_energy_ratio`
- `harmonic_order_count` — number of detected harmonic orders \(n \cdot f_0\) on the harmonic list (same integer as `unique_harmonic_order_count` when both are present).
- `spectral_entropy`
- `harmonic_effective_power_density`, `harmonic_effective_power_density_component_count`, `harmonic_effective_power_density_status`, `harmonic_effective_power_density_normalized_by_harmonic_count` (see §M)
- `harmonic_effective_power_mass`, `harmonic_effective_power_mean`, `harmonic_effective_power_rms`, `harmonic_effective_power_component_count`, `harmonic_effective_power_mass_status` (absolute-scale harmonic power; see §M)

**Optional compatibility column:** `unique_harmonic_order_count` (same semantics as `harmonic_order_count`).

**Optional (only if `robust_salient_inharmonic_peak_picking_enabled` is true in metadata):** `salient_inharmonic_peak_count`, `salient_subbass_peak_count` (not implemented in v6 by default).

**Do not place on `Density_Metrics`:** PC scores, dissonance scalars, Sethares/Hutchinson–Knopoff/Vassilakis, spectral masking outputs, STFT/window/hop/bin counts, candidate/peak-slot debug counts, legacy `R_norm` / `P_norm` / `D_agn` / `D_harm`, or `inharmonic_peak_count` unless robust salient peak picking is explicitly enabled and documented.

### C.1 Stage 2 weighted density (`density_weighted_sum`, `density_log_weighted`)

These columns are populated at **compile time** from per-note spectrum sheets and **`component_*_energy_ratio`** metadata (measured H+I+S energy fractions from Stage 1). They are **not** the same object as GUI **model weights** (`model_harmonic_weight` / `model_inharmonic_weight` = H/(H+I) for legacy combined metrics).

**Per-band density** \(D_H, D_I, D_S\) comes from `compile_metrics.extract_density_component_sum` under the **compile** `weight_function` (the same key passed to `compile_density_metrics_with_pca` / the orchestrator GUI: linear, log, sqrt, cubic, d3, d10, d17, d24, …). Examples:

| `weight_function` | Harmonic band \(D_H\) (typical) |
|-------------------|----------------------------------|
| `linear` | \(\sum \texttt{Amplitude\_raw}\) on included harmonic rows |
| `log` | \(\log_{10}(1 + \sum \texttt{Amplitude\_raw})\) |
| `sqrt`, `cubic`, … | `density.apply_density_metric` on the masked amplitude vector |

**Weighted sum (May 2026 semantics):**

\[
\text{density\_weighted\_sum} = D_H \cdot w_H + D_I \cdot w_I + D_S \cdot w_S
\]

with \(w_H, w_I, w_S\) = `component_harmonic_energy_ratio`, `component_inharmonic_energy_ratio`, `component_subbass_energy_ratio`.

- **`density_metric_raw`** on the compiled row uses the **same formula** and is numerically equal to **`density_weighted_sum`** when extraction status is `ok`.
- **`density_log_weighted`** = \(\log_{10}(1 + \text{density\_weighted\_sum})\).
- **`harmonic_amplitude_sum`**, **`inharmonic_amplitude_sum`**, **`subbass_amplitude_sum`** remain **linear** diagnostic sums of `Amplitude_raw`; they do **not** change when you switch from linear to log weighting. Use **`harmonic_density_sum`** (and the weighted sum above) for weight-function sensitivity.
- **`density_metric_normalized`** = `density_metric_raw / max(density_metric_raw)` **within the current compiled workbook** only — do not compare normalized values across runs that used different `weight_function` keys unless you re-normalise externally.

**Tests:** `tests/test_weighted_note_density.py` (including weight-function sensitivity).

---

## D. Count semantics

- **`harmonic_order_count`:** defensible public **harmonic multiplicity** — distinct harmonic numbers detected on the harmonic list.
- **`harmonic_candidate_count` / `inharmonic_candidate_count` / `subbass_candidate_count` / `total_spectral_candidate_count`:** counts of **classified rows** on the detected-peak table (slot assignment / sub-bass band split). **Not** verified local maxima on a frequency-sorted full spectrum unless a dedicated robust peak picker is enabled.
- **FFT / table row counts:** `harmonic_bin_count`, `inharmonic_bin_count`, `subbass_bin_count`, `residual_row_count`, `unmatched_spectral_row_count` — **debug/audit only** on `Debug_Counts` / `Validation_Metrics`.

**Why inharmonic energy is often aggregated for \(D_{\mathrm{eff}}\):** without frequency-ordered local-max peak picking, treating every inharmonic table row as an independent “partial” massively overstates musical inharmonic multiplicity.

---

## E. `Debug_Counts`

Technical bin, candidate-slot, and **residual-row hierarchy** counts. **These are not musical partial counts.** See `debug_counts_semantics_note` in `Analysis_Metadata`.

**Hierarchy (invariant tests):** `residual_spectral_row_count` ≥ `nonharmonic_candidate_row_count` ≥ `retained_nonharmonic_peak_candidate_count` = `exported_nonharmonic_peak_candidate_count`. **`peaklist_*` window counts** (`peaklist_harmonic_window_candidate_count`, `peaklist_nonharmonic_window_candidate_count`, `peaklist_low_frequency_window_candidate_count`, `peaklist_total_window_candidate_count`) are **independent** assignment counts — **do not** compare them to the residual hierarchy. **`debug_counts_invariant_status`** should read **`passed`** for normal runs; **`debug_counts_invariant_failures`** must be empty when status is passed.

Legacy column names `harmonic_peak_count` / … may appear here as aliases aligned with candidate-slot semantics.

---

## F. `Per_Note_Processing_Metadata`

Per-note STFT and policy fields (`source_file_name`, `n_fft`, `n_fft_effective`, `hop_length`, `bin_spacing_hz`, `sample_rate`, `window`, `tier`, `f0_estimated`, `f0_source`, `harmonic_tolerance`, `snr_threshold_db`, `rms_normalisation_enabled`, `smoothing_enabled`, `spectral_masking_enabled`). When the batch/orchestrator handoff is available, this sheet also carries **per-note** **global batch energy profile** (`batch_harmonic_energy_ratio`, `batch_inharmonic_energy_ratio`, `batch_subbass_energy_ratio`, `batch_total_inharmonic_energy_ratio`, `batch_energy_denominator`, `batch_energy_method`, `batch_ratio_source_explicit`) and **model projection / provenance** (`model_harmonic_weight`, `model_inharmonic_weight` as **H/(H+I)** and **I/(H+I)** on the musical band, `model_weight_denominator`, `model_weights_source`, `model_weights_warning`, `model_weights_fallback_reason`, `model_weight_safety_guard_applied`, optional `legacy_bounded_*`). These are **metadata / reproducibility**, not the primary harmonic-fatness readout — **`canonical_density_v5_adapted` on `Density_Metrics`** is the primary harmonic-row “fatness” index (see §K); **`effective_partial_density`** remains the participation-ratio descriptor on the same sheet. **Bin and candidate counts depend strongly on these settings** — they belong here, not on `Density_Metrics`.

---

## G. `Validation_Metrics`

QC for harmonic tracking / slot matching. **`non_harmonic_candidate_count`** / **`outside_harmonic_window_candidate_count`** count **spectral rows outside all harmonic tolerance windows** after slot assignment — a structural diagnostic, not a judgment that those peaks are invalid. This sheet is **quality control**, not a source of primary density.

---

## H. `Dissonance_Metrics` (and related sheets)

Per-note Sethares / Hutchinson–Knopoff / Vassilakis scalars, `selected_dissonance_model`, `selected_dissonance_value`, cap audit columns. **`dissonance_partial_count`** refers to partials **fed to the dissonance model** (possibly capped) — document separately from `harmonic_order_count`.

---

## I. PCA sheets (`PCA_Scores`, `PCA_Loadings`, `PCA_Explained_Variance`)

Exploratory only. Default PCA features include `effective_partial_density`, energy ratios, `harmonic_order_count`, `spectral_entropy`. **`inharmonic_peak_count` / `total_detected_peak_count` are excluded** unless robust peak features exist. Optional `selected_dissonance_value` when `pca_include_dissonance=True`.

---

## J. `Analysis_Metadata`

Reproducibility, export statuses, `robust_salient_inharmonic_peak_picking_enabled` (default false), density formula text, dissonance cap notes, `debug_counts_semantics_note`, `per_note_metadata_export_status`, and **workbook-level policy strings** such as `model_weight_policy`, `model_weights_source_policy`, `batch_ratio_sum_policy`, `batch_energy_denominator`, `model_weight_denominator`, and `selected_dissonance_model` (aligned from per-note rows when a single model is used). Per-note coefficient snapshots in per-file exports are scoped by `per_note_analysis_metadata_scope`; compiled workbooks describe **policies** here while **per-note values** live on `Per_Note_Processing_Metadata`.

**Per-note workbook only — component balance chart provenance:** after the linear amplitude sums used for `Metrics` alignment are finalised, Stage 1 may write two pie PNGs beside **`spectral_analysis.xlsx`**. The metadata keys below are populated on the **per-note** `Analysis_Metadata` sheet (filenames are basenames only):

- **`amplitude_mass_chart_file`** — typically **`component_amplitude_mass_pie.png`** when the diagnostic chart is written (empty when skipped, e.g. zero total linear sums).
- **`amplitude_mass_chart_basis`** — `linear_amplitude_sum` (linear ΣA-style workbook triple, **not** power ratios).
- **`amplitude_mass_chart_interpretation`** — `diagnostic_candidate_mass_not_energy`.
- **`energy_ratio_chart_file`** — typically **`component_energy_ratio_pie.png`** when harmonic and inharmonic energy ratios are available (empty when skipped).
- **`energy_ratio_chart_basis`** — `component_power_energy_ratios`.
- **`energy_ratio_chart_interpretation`** — `acoustic_energy_balance`.

A legacy copy **`component_energy_pie.png`** duplicates the amplitude-mass image for backward compatibility; interpret it using **`amplitude_mass_chart_*`** keys, not the old filename alone. Normative narrative: **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`** (per-note chart table).

---

## K. Canonical harmonic fatness (`canonical_density_v5_adapted`) and global normalization

**Why version 5 logic was restored:** Earlier v6 exports sometimes wrote **`density_metric_normalized`** as the rolloff sum divided by the fundamental linear amplitude, or other ratios that are **not bounded by 1** and are easily misread as a “probability” of density. That breaks cross-note comparability and confuses secondary rolloff descriptors with a primary density readout. The **primary** cross-note / cross-instrument harmonic-fatness scalar is again the **same core pipeline as legacy v5**: `density.apply_density_metric` on the harmonic-row linear amplitudes (max-normalization to limit single-partial domination, optional rolloff compensation vs \(n^{-\alpha}\) with fixed \(\alpha=1.5\) in the current implementation, then sum of the chosen weight function). This value is **unbounded on purpose** so that absolute spectral richness differences (register, bore, bow noise, etc.) are not collapsed by per-note ad hoc rescaling.

**`canonical_density_v5_adapted`:** Exported from Phase 2 `spectral_analysis.xlsx` / `proc_audio` (and mirrored in `super_analysis_results.json` when that path fills `spectral_metrics`). It is **not** recomputed from Phase 1 JSON during compilation, so **Phase 1 and Phase 2 are not mixed** into this column.

**`density_normalized_global`:** After all notes in a compilation are loaded, `compile_metrics` sets  
`density_normalized_global = canonical_density_v5_adapted / max(canonical_density_v5_adapted)`  
over the current compiled frame (finite max only). This is the **only** density field intended to be in **[0, 1]** by construction (aside from NaNs).

**`density_metric_normalized`:** Strict **alias** of `density_normalized_global` on compiled **`Density_Metrics`** (same numeric column policy). Per-note workbooks may leave it blank (NaN) until compilation.

**`density_per_component`:** Diagnostic only: `canonical_density_v5_adapted / harmonic_order_count` when the per-note export did not already supply it; at compile time a fill is applied only if the column is missing. **Never** treat this as the primary density metric for comparing instruments.

**`density_metric_per_harmonic` (on `Density_Metrics`):** Per-note **harmonic-density model** selector (`harmonic_density_model` on the `Metrics` sheet): default **`rolloff_compensated`** uses the mean rolloff-compensated contribution per harmonic row when `rolloff_compensated_harmonic_density_status == "computed"`; **`weight_function`** uses `canonical_density_v5_adapted / harmonic_row_count`; **`harmonic_effective_power_density`** uses the diagnostic `harmonic_effective_power_density_normalized_by_harmonic_count` when that metric is computed (see §M); **`harmonic_effective_power_mass`** uses `harmonic_effective_power_rms` when `harmonic_effective_power_mass_status == "computed"` (see §M). Phase precedence for **rolloff** public columns still follows §L below; HEpd precedence follows §M. Neither precedence applies to `canonical_density_v5_adapted`.

**Metadata:** `density_source_formula`, `density_formula_version`, `density_normalization_scope`, `density_normalization_denominator` (the global max used for normalization, when defined).

**Legacy workbook support:** If `canonical_density_v5_adapted` is absent but legacy `Density Metric` is present, compilation reconstructs `canonical_density_v5_adapted ≈ Density Metric / 10` (because `Density Metric` in older per-note sheets carried the scaled display form).

---

## L. Rolloff-compensated harmonic density (Phase 1 vs Phase 2)

**Definition:** `rolloff_compensated_harmonic_density` is the sum of weighted, rolloff-expected–compensated harmonic partial strengths from Phase 1 harmonic-component analysis (see `density.compute_rolloff_compensated_harmonic_density`: per-partial normalization to the local harmonic amplitude maximum, expected rolloff \(n^{-\alpha}\) with default \(\alpha = 1.5\), weight function default `logarithmic`).

**Primary source for compiled public columns:** values already recorded in per-note **`spectral_analysis.xlsx`** (Phase 2 / `proc_audio` export). **Optional upgrade path:** when a sibling **`super_analysis_results.json`** exists and satisfies the Phase‑1 contract (`status == "computed"` and finite density), compilation may copy **rolloff** / **HEpd** public fields from JSON → `spectral_metrics` instead of the Excel-only values — see `rolloff_density_json_discovery_method` and related metadata. Discovery order remains: (1) sibling JSON; (2) optional `batch_results/**` candidate match; (3) otherwise **not_found** and Excel values stand. This does **not** make `super_audio_analyzer` the canonical Stage 1 engine; it is a **legacy sidecar** used only when present and contract-valid.

**Phase 2-only / canonical path:** If JSON is missing or the contract fails, public rolloff / HEpd columns follow **`spectral_analysis.xlsx`** only. `Analysis_Metadata` records `rolloff_density_public_canonical_source_policy` and per-row `rolloff_density_source_phase` (`phase1_super_analysis`, `phase2_spectral_analysis_fallback`, or `phase2_configuration_override`), `rolloff_density_source_file`, `rolloff_density_json_discovery_method` (e.g. `same_directory_super_analysis_json`, `batch_results_stem_normalized_match`, `not_found`), and `rolloff_density_json_match_confidence` (`high` / `medium` / `low` / `none`).

**Explicit Phase 2 preference:** Callers may set `prefer_phase2_rolloff_density=True` on `compile_density_metrics_with_pca` (and legacy `compile_density_metrics`) so the public columns follow Phase 2 even when Phase 1 JSON is valid (comparison / debugging only).

**Count semantics:** `rolloff_compensated_harmonic_density_component_count` is the **component population used in the Phase 1 rolloff sum** (same row set as the rolloff sum). It must **not** be equated with `harmonic_order_count` from Phase 2 (detected orders on the harmonic list). When Phase 1 canonical rolloff is applied, optional audit columns `phase1_harmonic_order_count` (from super `harmonic_count` when present) and `phase2_harmonic_order_count` (copy of compiled `harmonic_order_count`) clarify the split. Wide compiled frames also carry `phase1_*` / `phase2_*` rolloff snapshots for audit.

**`legacy_rolloff_compensated_density`:** Kept only as a backward-compatible alias on the **wide** in-memory / `Compiled_Metrics_All` path when present; it is **not** written to **`Density_Metrics`** or publication-filtered sheets (it duplicates the public scalar and was a common source of semantic confusion).

**Absolute vs normalized descriptors:** `effective_partial_density` is scale-invariant in power (participation-ratio / “effective number of components”). Rolloff-compensated harmonic density uses normalized partial amplitudes within the harmonic row set but aggregates in a way that preserves relative partial structure; it is **not** the same object as min–max “Index_Weighted” style normalizations applied across the corpus in some legacy PCA helper paths. Treat cross-note min–max of auxiliary indices as **comparative**, not as replacements for absolute instrument-sensitive readouts unless explicitly documented.

---

## M. Harmonic effective power density

**Harmonic effective power density** is an additive harmonic-row descriptor computed **only** from valid harmonic partial rows (finite frequency and linear amplitude \(A_i>0\), harmonic number \(\ge 1\) when available, otherwise inferred from \(f/f_0\) when \(f_0\) is valid). It is **not** a perceptual loudness model: it does **not** include masking, equal-loudness contours, critical-band aggregation, or psychoacoustic roughness.

For each included partial \(i\), with linear amplitude \(A_i\):

\[
P_i = A_i^2,\qquad p_i = \frac{P_i}{\max_j P_j},\qquad D_{\mathrm{eff}} = \sum_i p_i .
\]

Weak high partials contribute proportionally less because \(p_i\) scales with **power** relative to the strongest harmonic. The **primary exported scalar** is **`harmonic_effective_power_density`** \(= D_{\mathrm{eff}}\). A **secondary diagnostic only** is

\[
\frac{D_{\mathrm{eff}}}{N}
\]

exported as **`harmonic_effective_power_density_normalized_by_harmonic_count`** when the valid-component count \(N>0\). **Do not** treat \(D_{\mathrm{eff}}/N\) as the main density value.

**Supporting fields:** `harmonic_effective_power_density_component_count`, `harmonic_effective_power_density_status` (e.g. `computed`, `skipped_no_valid_amplitude_column`, `skipped_no_valid_harmonic_rows`, `skipped_nonfinite_result`, `skipped_zero_or_negative_max_amplitude`), `harmonic_effective_power_density_max_amplitude`, `harmonic_effective_power_density_total_power`.

**Relative vs absolute harmonic power (same harmonic-row amplitude basis):** `harmonic_effective_power_density` is **intra-note relative** richness: each note is normalized by its own strongest harmonic partial, so it does not preserve raw energetic scale across instruments. **`harmonic_effective_power_mass`** is \(\sum_i P_i\) over **positive finite** linear amplitudes in the chosen amplitude column (no max-normalization); it preserves **absolute-scale** harmonic power and is suited to comparing harmonic energetic weight across instruments or notes. **`harmonic_effective_power_rms`** is \(\sqrt{\mathrm{mean}(P_i)}\), an absolute-scale companion less directly tied to partial count than the raw sum. **`harmonic_effective_power_mean`** is \(\mathrm{mean}(P_i)\). Status `harmonic_effective_power_mass_status` records `computed` or neutral skip reasons (`skipped_empty_harmonic_df`, `skipped_missing_<column>`, `skipped_no_positive_finite_amplitudes`). **`harmonic_effective_power_component_count`** counts the amplitudes used in the mass statistics (distinct from `harmonic_effective_power_density_component_count` when row filters differ).

**Warning:** **`density_metric_normalized`** (global alias of `density_normalized_global` after compilation, §K) is **not** the primary cross-instrument absolute harmonic-power readout; use **`harmonic_effective_power_mass`** / **`harmonic_effective_power_rms`** when absolute harmonic energy scale is required.

**Phase precedence (compiled workbooks):** Public HEpd columns on **`Density_Metrics`** follow the same **Excel-first / JSON-optional** rule as §L: **`spectral_analysis.xlsx`** is authoritative when JSON is absent or fails the Phase‑1 contract; otherwise JSON may supply values when valid. The flag `prefer_phase2_rolloff_density=True` forces Phase 2 values for HEpd (same override as rolloff). Wide rows may include `harmonic_effective_power_density_source_phase`, `harmonic_effective_power_density_source_file`, and `phase1_*` / `phase2_*` snapshots for audit. Mass fields follow the same per-file ingestion path as other `spectral_metrics` / `Metrics` columns (no separate Phase-1 override bundle today).

---

## Publication path redaction

- **Local absolute paths** (for example `C:\Users\...`, drive-letter paths containing `Desktop\`, `/home/...`, `/Users/...`, `/mnt/...`) are **not** written into publication-oriented Excel, JSON, CSV, or text exports when `REDACT_LOCAL_PATHS_FOR_PUBLICATION` is **True** (the default in `constants.py`).
- Exports prefer **`public_audio_id`**, **`source_file_basename`**, and **`source_file_hash_short`** for traceability without exposing the analyst’s directory layout. Path-like keys (`folder_path`, `*_path`, `*_dir`, `output_directory`, etc.) are emitted as **`redacted_for_publication`** unless a safe **`<DATASET_ROOT>/relative`** form applies inside a controlled dataset tree.
- **Runtime logs** may still print full paths for local debugging; only **deposable scientific metadata** is sanitised (`metadata_sanitizer`, `compile_metrics`, `proc_audio`, batch writers, orchestrator JSON).
- **Numerical density metrics** (`effective_partial_density`, energy ratios, `Density_Metrics` column allow-list) are unchanged by this policy; only string metadata and path-bearing cells are redacted. Use `python scripts/check_publication_paths.py <folder>` before Zenodo upload and `python scripts/validate_density_workbook.py <compiled_density_metrics.xlsx>` for workbook gates.
