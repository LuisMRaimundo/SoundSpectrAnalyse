# Canonical pipeline and export semantics

**Audit date:** 2026-05-13  
**Audience:** researchers, reviewers, and developers citing SoundSpectrAnalyse in publications or downstream tools.

This page summarises what the **current code and tests** implement. For column allow-lists and sheet layout, see **`DENSITY_EXPORT_SCHEMA.md`**. For batch row semantics, see **`BATCH_ANALYSIS_AUDIT.md`**.

---

## 1. Canonical pipeline

| Stage | Component | Output |
|--------|-----------|--------|
| **Stage 1 (canonical)** | `proc_audio.AudioProcessor` | Per-note **`spectral_analysis.xlsx`** (slim **`Metrics`** + default **`Legacy_Density_Metrics`**) |
| **Stage 2 (canonical)** | `compile_metrics.compile_density_metrics_with_pca` | **`compiled_density_metrics.xlsx`** |

**Legacy / diagnostic (not required for the canonical chain):**

- `audio_analysis/super_audio_analyzer.py`, **`SuperAudioAnalyzer`**, and **`super_analysis_results.json`** — optional Phase‑1 batch path and sidecar JSON used for some **rolloff / HEpd discovery** fallbacks when present; they are **not** the canonical Stage‑1 engine.
- Older Tk entry points (`pipeline_orchestrator_gui.py`, `run.bat`) remain available but are not the normative description of publication metrics.

**Per-note chart PNGs** (same directory as each **`spectral_analysis.xlsx`**, written during Stage 1 `AudioProcessor` export when chart generation succeeds):

| File | Quantity plotted | Interpretation |
| --- | --- | --- |
| **`component_amplitude_mass_pie.png`** | Linear sums **`linear_sum_amplitude_harmonic`**, **`linear_sum_amplitude_inharmonic_partial`**, **`linear_sum_amplitude_subbass_band`** | **Diagnostic amplitude-mass** partition aligned with the workbook export columns — **not** measured \(\sum A^2\) energy fractions. The chart title and footnote state this basis explicitly. |
| **`component_energy_ratio_pie.png`** | **`harmonic_energy_ratio`**, **`inharmonic_energy_ratio`**, **`subbass_energy_ratio`** | **Measured component energy / power** partition (approximately sum to 1 on valid rows). |
| **`component_energy_pie.png`** | Identical pixels to **`component_amplitude_mass_pie.png`** | **Legacy filename only** (byte copy for backward compatibility). Older tooling that assumes this path still works; readers must **not** infer from the legacy name that the wedges are the same object as the energy-ratio chart. |

Per-note **`Analysis_Metadata`** (sheet in **`spectral_analysis.xlsx`**) includes **`amplitude_mass_chart_file`**, **`amplitude_mass_chart_basis`** (`linear_amplitude_sum`), **`amplitude_mass_chart_interpretation`** (`diagnostic_candidate_mass_not_energy`), **`energy_ratio_chart_file`**, **`energy_ratio_chart_basis`** (`component_power_energy_ratios`), **`energy_ratio_chart_interpretation`** (`acoustic_energy_balance`). Regression tests: **`tests/test_component_balance_pies.py`**.

Canonical compilation ingests **`spectral_analysis.xlsx`** (pattern configurable; default matches the per-note export from `AudioProcessor`).

---

## 2. f0 policy (field semantics)

Exports use a **single coherent f0 decision** per note. Important fields (names may appear on per-note and/or compiled diagnostic sheets):

| Field | Role |
|--------|------|
| **`f0_prior_note`** / filename-derived label | Strong **nominal pitch prior** from the parsed note — **not** automatically the measured acoustic f0. |
| **`f0_nominal_hz`** | Nominal frequency in Hz corresponding to that prior (e.g. from note-to-Hz conversion). |
| **`f0_final_hz`** | Final frequency used for harmonic spacing and exports after fitting / rejection policy. |
| **`f0_source`** / **`f0_final_source`** | Provenance of the chosen f0 (e.g. prior-constrained harmonic fit vs nominal fallback). |
| **`f0_final_method`** | Method tag aligned with the decision. |
| **`f0_fit_accepted`** | Whether a prior-constrained harmonic fit was **accepted**. |
| **`f0_fit_quality`**, **`f0_fit_residual_std_hz`** | Fit diagnostics when a fit was attempted. |
| **`f0_fit_rejection_reason`** | Text reason when a fit is rejected. |
| **`f0_detuning_cents_from_nominal`** | Detuning of final f0 vs nominal when both are defined. |

**Invariant (tests):** `prior_constrained_harmonic_fit` cannot be exported as the source while **`f0_fit_accepted`** is false. A rejected fit falls back toward nominal behaviour; see `tests/pipeline_workbook_audit.py` and `tests/test_final_pipeline_invariants.py`.

---

## 3. Harmonic frequency export (Harmonic Spectrum sheet)

| Column | Meaning |
|--------|---------|
| **`expected_frequency_hz`** | Template frequency \(k \cdot f_0\) for the row. |
| **`bin_center_frequency_hz`** | FFT bin centre — **audit / reference** value, not the preferred peak frequency. |
| **`interpolated_frequency_hz`** | **Preferred** sub-bin acoustic peak estimate when interpolation is used. |
| **`extracted_frequency_hz`** | Must match **`interpolated_frequency_hz`** when **`subbin_interpolation_valid`** is true. |
| **`subbin_offset_bins`**, **`subbin_interpolation_valid`** | Sub-bin offset and validity flag. |
| **`peak_bin_index`** | Peak bin index when applicable. |
| **`frequency_deviation_hz`** | **`extracted_frequency_hz - expected_frequency_hz`**. |

---

## 4. Nonharmonic terminology

**Hierarchy (residual / candidate semantics):**

1. **`residual_spectral_row`** — row in the residual-oriented spectral table.  
2. **`nonharmonic_candidate_row`** — row classified as a nonharmonic **candidate**.  
3. **`nonharmonic_peak_candidate`** — peak-level candidate before retention filters.  
4. **`retained_nonharmonic_peak_candidate`** / **`exported_nonharmonic_peak_candidate`** — counts aligned with what is retained and exported.

**Stricter physical labels** such as **`accepted_inharmonic_peak`** / **`accepted_inharmonic_partial`** apply only when the pipeline’s **strict validation** path says so. Default exported nonharmonic table rows are **candidates**, not guaranteed physical inharmonic partials. Do **not** use the label **`inharmonic_partial`** unless strict physical validation applies.

**Publication-facing counts:** use **`retained_nonharmonic_peak_candidate_count`** and **`exported_nonharmonic_peak_candidate_count`**. The bare name **`nonharmonic_peak_candidate_count`** must not appear as an ambiguous publication column unless paired with an explicit deprecated legacy alias (see invariant tests).

---

## 5. Low-frequency / subfundamental policy

These are **distinct** concepts:

- **DC offset** — mean removal / DC handling in time domain (not the same as subfundamental guard).  
- **Physical low-frequency band** — fixed diagnostic band for low-frequency residual labelling (see policy exports).  
- **Subfundamental residual** — classification relative to **`adaptive_subfundamental_cutoff_hz`**, not “all sub-bass energy”.  
- **Adaptive subfundamental cutoff** — register-dependent guard derived from **`low_frequency_policy.py`**.  
- **Leakage / main-lobe guard** — optional **`leakage_guard_cutoff_hz`** candidate in the max stack.  
- **Low-frequency diagnostic exports** — sheets such as sub-bass / low-frequency residual tables for audit.

**Key exported fields** (see compiled **Canonical_Metrics** / metadata when present):

- **`low_frequency_policy_version`**
- **`f0_final_hz`**
- **`subfundamental_margin_percent`** — **nominal** register-dependent margin (percent below f0 defining the percentage line).  
- **`percentage_subfundamental_cutoff_hz`** — \(f_{0,\mathrm{final}} \times (1 - \texttt{subfundamental_margin_percent}/100)\).  
- **`leakage_guard_cutoff_hz`**
- **`adaptive_subfundamental_cutoff_hz`** — **numeric** when f0 is valid at compile time (not `not_available_at_compile_stage` for valid-f0 rows).  
- **`effective_subfundamental_margin_percent`** — **actual** margin after all guards: \(100 \times (1 - \texttt{adaptive}/f_{0,\mathrm{final}})\).  
- **`subfundamental_cutoff_selection_rule`**, **`subfundamental_cutoff_selected_by`**
- **`min_floor_hz`**, **`max_fraction_of_f0`**
- **`physical_low_frequency_lower_hz`**, **`physical_low_frequency_upper_hz`**
- **`low_frequency_residual_interpretation`** (when exported)

**Selection:** **`subfundamental_cutoff_selected_by`** records which rule set the final adaptive cutoff among at least: **`percentage_subfundamental_cutoff_hz`**, **`leakage_guard_cutoff_hz`**, **`min_floor_hz`**, **`max_fraction_of_f0_cap`**.

**Sub-bass energy (`subbass_energy_sum`)** is a **wideband aggregated** component for the H+I+S energy model; it is **not** the same object as the **subfundamental residual** classification band.

---

## 6. Debug_Counts

**Residual hierarchy (must hold on invariant-passing rows):**

`residual_spectral_row_count` ≥ `nonharmonic_candidate_row_count` ≥ `retained_nonharmonic_peak_candidate_count` = `exported_nonharmonic_peak_candidate_count`

**Independent peaklist window counts** (do **not** compare to the hierarchy above):

- `peaklist_harmonic_window_candidate_count`  
- `peaklist_nonharmonic_window_candidate_count`  
- `peaklist_low_frequency_window_candidate_count`  
- `peaklist_total_window_candidate_count`

**QC fields:** `debug_counts_invariant_status` (expect **`passed`** for normal runs), `debug_counts_invariant_failures` (should be empty when status is passed), `debug_counts_semantics`, `debug_counts_source_policy` when present on **`Analysis_Metadata`**.

---

## 7. Missing metric policy

- **`NaN`** means missing / not computed / not applicable for a numeric metric.  
- **`0.0`** means a **real computed zero**, not “unknown”.  
- **String statuses** (`not_computed`, `not_applicable`, `skipped`, …) may appear on object columns; interpret via status columns.  
- **`Index_Weighted`** uses **available-term renormalisation**; missing components are **not** silently treated as zero in the weighted index (see compiler logs and `Index_Weighted_status`).  
- **`N_harm_norm`** is **`NaN`** when **`Harmonic Count`** is unavailable — not filled with 0.0.

---

## 8. Final audit workflow

**CLI**

```bash
python tools/audit_compiled_workbook.py path/to/compiled_density_metrics.xlsx
```

Optional second path for Harmonic Spectrum checks:

```bash
python tools/audit_compiled_workbook.py path/to/compiled_density_metrics.xlsx path/to/spectral_analysis.xlsx
```

Exit code **0** = no hard invariant failures; **1** = blocker failures; **2** = usage or missing file.

**Tests**

```bash
pytest tests/test_final_pipeline_invariants.py tests/test_low_frequency_policy.py tests/test_validate_canonical_metrics.py
```

**Optional environment integration**

PowerShell:

```powershell
$env:SSA_COMPILED_WORKBOOK="C:\path\to\compiled_density_metrics.xlsx"
$env:SSA_PER_NOTE_WORKBOOK="C:\path\to\spectral_analysis.xlsx"
pytest tests/test_final_pipeline_invariants.py
```

CMD:

```bat
set SSA_COMPILED_WORKBOOK=C:\path\to\compiled_density_metrics.xlsx
set SSA_PER_NOTE_WORKBOOK=C:\path\to\spectral_analysis.xlsx
pytest tests/test_final_pipeline_invariants.py
```

**`input_schema_validation_status`:** values beginning with `not_validated` are treated as a **non-blocking documentation warning** until an orchestrator schema validator is wired.

---

## 9. Research export workbook (`compiled_density_metrics_research.xlsx`)

After Stage 2 produces **`compiled_density_metrics.xlsx`**, the optional **research** workbook is a **read-only, reduced** Excel file for plotting and thesis tables. It is built by **`tools/export_research_density_workbook.py`** (also invoked automatically after successful compile from **`post_compile_research_export`** when wired in the orchestrator).

**Semantics**

- Does **not** modify Stage 1/2 numeric pipelines or rewrite the compiled workbook.  
- Merges **`Legacy_Compatibility`** (among other sheets) so **`Combined Density Metric`** and other legacy columns from per-note **`Legacy_Density_Metrics`** are available when present.  
- Keeps `density_metric_raw` as a diagnostic energy-weighted component sum and does **not** export `density_weighted_sum_cdm_mean` by default.  
- `density_weighted_sum_cdm_mean` is available only with `--include-legacy-cdm-mean` and remains a deprecated editorial blend, not a canonical density (see **`docs/DENSITY_EXPORT_SCHEMA.md`** §R).  
- May **infer or override** `Instrument` / `Dynamic` metadata (CLI: `--instrument`, `--dynamic`, `--force-metadata`); see the research workbook **README** sheet.  
- May **resolve** per-note component chart paths under the compiled workbook’s parent folder when filenames are missing from the source sheet.

**Excel file format**

- Uses **worksheet-level `AutoFilter`** on data sheets only; **no** formal **Table** / ListObject parts (avoids `xl/tables/table*.xml` and Microsoft Excel “repair” prompts). **README** and **Dashboard** sheets are not auto-filtered.  
- Column headers written from DataFrames are **sanitised** (non-blank names; duplicate bases receive `_2`, `_3`, …).  
- **`Spectral_Density_Metrics`** only: soft fills on **`density_weighted_sum`** (blue), **`Combined Density Metric`** (yellow), **`density_weighted_sum_cdm_mean`** (lavender) for quick visual separation in thesis tables.

**Legacy density at Stage 1 (context)**

- Every new **`spectral_analysis.xlsx`** includes **`Legacy_Density_Metrics`** (SDM, FDM, CDM, `Density Metric`) so compile can rebuild **`Weighted Combined Metric`** on **`Diagnostic_Metrics`**. v6 does **not** expose the v5 “Enable Spectral Masking” GUI; **`spectral_masking_enabled`** is recorded as **`False`**.

**Further reading**

- **`README.md`** — CLI examples.  
- **`docs/DENSITY_EXPORT_SCHEMA.md`** §F1, §R — normative sheet and research-column definitions.  
- **`docs/COMPUTATIONAL_METRICS_CODE_REVIEW_REPORT.md`** — separate read-only survey of project-owned **computational** code (not the Excel exporter).

---

## 10. Formula extraction, validation, and traceability (engineering)

This section is **orthogonal** to the Stage 1/2 Excel contract above: it documents how **project-owned mathematics** in code is made explicit and regression-checked.

| Artefact | Role |
|----------|------|
| **`docs/formula_extraction/FORMULA_EXTRACTION_INDEX.md`** | Index to Pass **1–15** formula-extraction tables (Python expressions → notation). |
| **`docs/formula_validation/FORMULA_VALIDATION_PLAN_INDEX.md`** | Index to Pass **1–15** validation plans (fixtures and suggested assertions). |
| **`tests/formula_validation/`** | Pytest modules implementing those plans for Passes **1–15**. |
| **`docs/validation/VALIDATION_STATUS.md`** | Formula-validation status summary (Passes 1–15) without fixed full-suite pass/fail assertions. |
| **`docs/validation/METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md`** | Methodological note on the extraction → validation workflow. |
| **`CODE_FORMULA_TRACEABILITY_TABLE.md`** (repo root) | Optional code ↔ formula traceability table. |

**Cautious interpretation:** the formula-validation corpus supports **internal consistency** between the documented mathematical formulas and the tested Python implementations. It verifies formula/code agreement for **selected numerical fixtures**. It does **not**, by itself, prove scientific optimality, universal correctness, or full acoustic validity of the models.

---

## 11. Naming caveat: effective_partial_density vs. classical density

`effective_partial_density` currently computes `N_eff / N`, where `N_eff` is the Hill diversity index for `q = 2` (inverse Herfindahl) on partial power weights (`Hill, 1973`; `Jost, 2006`).
This is an effective-participation statistic, not a bandwidth-occupancy density in the classical sense.
Readers should avoid mapping it directly to density framings associated with Krimphoff et al. (1994) or Peeters et al. (2011).
The exported column name is retained for backward compatibility and workbook stability.
