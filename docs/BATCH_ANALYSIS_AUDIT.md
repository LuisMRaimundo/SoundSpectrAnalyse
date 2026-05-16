# Batch-analysis layer — static audit (SoundSpectrAnalyse-main_6)

> **Canonical pipeline reminder:** Publication-grade per-note analysis and compilation are **`proc_audio.AudioProcessor` → `spectral_analysis.xlsx` → `compile_metrics.compile_density_metrics_with_pca` → `compiled_density_metrics.xlsx`** (see **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`**). This document describes the **optional Phase 1** batch layer that runs **`SuperAudioAnalyzer`** to populate **`batch_summary.xlsx`** and related artefacts — useful for empirical **H+I+S** priors, **not** a substitute for the canonical Stage 1 engine.

**Scope:** Batch folder **`audio_analysis`** (ASCII directory name; legacy checkouts used a non-ASCII UTF-8 path segment and have been migrated for portability).  
**Path resolution in code:** `Path(__file__).parent / "audio_analysis"` (e.g. in `pipeline_orchestrator_integrated.py`, `run_orchestrator.py`). The repository expects UTF-8 source files; path literals for this folder are ASCII-only.

---

## A. Executive summary

The batch layer runs **one full `SuperAudioAnalyzer` pipeline per audio file** in parallel (`ProcessPoolExecutor`), writes **per-file artefacts** under `batch_results/<NN>_<stem>/`, then aggregates a **tabular summary** (`batch_summary.xlsx` / `.csv`) plus **`batch_results.json`** and **`batch_statistics.txt`**.

Empirical **harmonic / inharmonic / subbass energy shares** are ultimately derived from **time-averaged STFT power** (`Power` = mean over time of `|STFT|²`) summed over bin-level masks after harmonic vs inharmonic vs subbass separation. That is **scientifically aligned with an energy (power) interpretation**, not raw dB and not raw linear amplitude for the percentage numerators.

However, several issues reduce interpretability without extra documentation in downstream tools:

1. **Mixed denominators before batch renormalisation:** `spectral_metrics` exports `harmonic_energy_percentage` / `inharmonic_energy_percentage` as **100 × H / (H + I)** in the **musical band only** (subbass excluded). Batch then adds `subbass_energy_pct_global` from `spectral_component_stats`, which is defined on **H + I + S** over the full classified spectrum. Summing the three raw values can deviate from 100%; batch **rescales** H, I, S so the triplet sums to 100%. The exported legacy column names still read like generic “energy percentage” without stating the **post hoc** three-way denominator.

2. **Count semantics are not partial counts:** `spectral_component_stats['harmonic_peak_count']` is **`len(harmonic_df)`** — bins/rows retained after masking and dB noise-floor filtering — not “peaks” in the strict sense, nor `harmonic_order_count` from the main per-note workbook. Batch prefers these keys when present, so **`harmonic_count` / `inharmonic_count` in the batch table are often FFT-bin/row-like counts**, with a fallback to unique harmonic numbers for harmonic only when peak stats are missing.

3. **Orchestrator handoff (updated):** `RobustOrchestrator` validates **H + I + S ≈ 1** (or legacy **H + I ≈ 100%** / **H + I + S ≈ 100%** when only percentage columns exist) and derives **`harmonic_weight` / `inharmonic_weight` as model coefficients** via **H / (H + I)** and **I / (H + I)** on the musical band, **not** by using global H and I as weights when subbass is present. **Subbass is still not a third `proc_audio` weight**, but measured **`batch_subbass_energy_ratio`** is preserved in result metadata. **Extreme empirical model weights** (e.g. flute-like **H/(H+I) ≫ 0.95**) are **accepted**, optionally flagged via **`model_weights_warning`**; **`legacy_bounded_*`** weights exist only for optional legacy consumers. **Fixed numeric priors** (`0.90`/`0.10` on invalid batch data; **`0.95`/`0.05`** in `SuperAudioAnalyzer` / compilation defaults) must be tagged **`fallback_default`** when used — never as empirical batch ratios. See **§Tier 1 denominator/handoff fix** below.

**Bottom line:** The batch layer **can** support empirical energy-profile estimation in the **power domain**, but **column naming, mixed-definition percentages before rescaling, and “peak” count labels** create real ambiguity versus the main pipeline’s `harmonic_order_count` / bin-count exports. Treat batch outputs as **measured energy ratios plus debug counts** unless provenance fields (`batch_energy_denominator`, `batch_energy_method`, `f0_detection_method`) are carried through to every consumer.

---

## B. Pipeline call graph (batch → super → export → orchestrator)

```
CLI / GUI / RobustOrchestrator.run_preprocessing_phase()
  └─► batch_audio_analyzer.BatchAudioAnalyzer.run_batch_analysis()
        └─► ProcessPoolExecutor: batch_audio_analyzer._analyze_single_file(file, output_subdir, **kwargs)
              └─► super_audio_analyzer.SuperAudioAnalyzer(audio_path, output_dir=output_subdir, **kwargs)
                    └─► run_complete_analysis()
                          1. load_audio()              [librosa.load, optional HPF, RMS metadata]
                          2. compute_spectrogram()     [librosa.stft → |X|; time-mean → spectrum DF]
                          3. detect_fundamental_frequency()  [pYIN → YIN → autocorr; filename note as prior]
                          4. separate_harmonic_inharmonic()  [masks → harmonic_df, inharmonic_df, subbass_df;
                                                              spectral_component_stats (+ peak-based validation)]
                          5. calculate_spectral_metrics()    [harmonic_energy, inharmonic_energy, H/(H+I) %]
                          6. calculate_dissonance_metrics()
                          7. perform_statistical_analysis()
                          8. perform_dimensionality_reduction()
                          9. run_internal_consistency_checks()
                          10. generate_comprehensive_plots() → super_comprehensive_analysis.png
                          11. save_results() → JSON, CSVs, metrics_summary.txt, analysis_metadata.json
              └─► Build flat dict row from results["spectral_metrics"], results["spectral_component_stats"],
                    metrics, dissonance, metadata (f0, RMS, …)

  └─► BatchAudioAnalyzer._save_batch_results()
        ├─► batch_results.json  (full paths + summary_stats)
        ├─► batch_summary.xlsx / batch_summary.csv  (_batch_summary_export_dataframe drops file_path from Excel)
        └─► batch_statistics.txt

RobustOrchestrator (post batch)
  └─► load_percentage_mapping()  ← reads batch_summary.xlsx
  └─► run_main_analysis_phase()  ← get_percentages_for_file(); validates; sets harmonic_weight/inharmonic_weight;
        └─► proc_audio.AudioProcessor.apply_filters_and_generate_data(..., harmonic_weight=..., inharmonic_weight=...)

run_orchestrator.py
  └─► Thin CLI wrapper: same default excel path under audio_analysis/batch_results/batch_summary.xlsx

compile_metrics.py (secondary path, not batch writer)
  └─► May read harmonic_energy_percentage / inharmonic_energy_percentage from per-note super JSON when compiling
```

---

## C. Files and functions inspected

| File | Role |
|------|------|
| `audio_analysis/batch_audio_analyzer.py` | `BatchAudioAnalyzer`, `_analyze_single_file`, `run_batch_analysis`, `_save_batch_results`, `_calculate_summary_stats`, Excel/JSON export helpers |
| `audio_analysis/super_audio_analyzer.py` | `SuperAudioAnalyzer`, `load_audio`, `compute_spectrogram`, `detect_fundamental_frequency`, `separate_harmonic_inharmonic`, `calculate_spectral_metrics`, `_detect_harmonics_peak_based`, `save_results`, `run_complete_analysis` |
| `audio_analysis/super_audio_analyzer_gui.py` | UI references to batch outputs (not altering pipeline logic) |
| `audio_analysis/BATCH_PROCESSING_README.md`, `BATCH_CONFIG_README.md` | Existing user-facing batch docs |
| `pipeline_orchestrator_integrated.py` | Batch run, `load_percentage_mapping`, `get_percentages_for_file`, `run_main_analysis_phase`, `run_compilation_phase` |
| `run_orchestrator.py` | Default paths to `audio_analysis/batch_results/batch_summary.xlsx` |
| `compile_metrics.py` | Coercion of `harmonic_energy_percentage` from super JSON into compiled metrics (secondary consumer) |

---

## D. Output files and fields

### D.1 Batch-level (under `audio_analysis/batch_results/` by default)

| Output | Created in | Function | Contents (high level) |
|--------|------------|----------|------------------------|
| `batch_results.json` | `_save_batch_results` | Full `results` list + `summary` + paths | **Audit / debug**; includes `file_path` |
| `batch_summary.xlsx` / `batch_summary.csv` | `_save_batch_results` | One row per file + optional MEAN/MEDIAN/TIER rows | Mixed: **energy %**, **ratios**, **sums of power**, **counts**, **dissonance**, **metadata**; `file_path` dropped from Excel via `_batch_summary_export_dataframe` |
| `batch_statistics.txt` | `_save_batch_results` | Human-readable summary | **Audit** |
| `batch_analyzer.log` | module logging config | Stream + file handler | **Debug** (cwd-dependent) |

### D.2 Per-file subdirectory `batch_results/<NN>_<stem>/`

| Output | Created in | Function | Field semantics |
|--------|------------|----------|-----------------|
| `super_analysis_results.json` | `save_results` | Full `self.results` tree | **Mixed:** metrics dicts, `spectral_component_stats`, frequency analysis, dissonance, metadata — **energy** and **debug** |
| `analysis_metadata.json` | `save_results` | Subset of metadata | **Audit** |
| `harmonic_components.csv` | `save_results` | Rows from `harmonic_df`: `Frequency (Hz)`, `Amplitude_linear`, `Power`, `Magnitude_dB`, … | **Per-bin / per-row** harmonic-classified spectrum samples — **not** “one row per partial” |
| `inharmonic_components.csv` | `save_results` | `inharmonic_df` | Same — **inharmonic-classified bins** |
| `complete_spectrum.csv` | `save_results` | Full averaged spectrum | **Amplitude**, **power**, **dB** columns |
| `metrics_summary.txt` | `save_results` | Dump of `self.metrics` (+ dissonance) | **Mixed** human-readable; starts with **interpretation notes** (bin vs peak energy labels, diagnostic density vs `effective_partial_density`, component-count semantics). **`dissonance_curve`** is a **numeric summary** in this file; the full interval→dissonance map stays in **`super_analysis_results.json`** so `txt` writers never need unsafe formatting on float keys. |
| `super_comprehensive_analysis.png` | `generate_comprehensive_plots` | Multi-panel figure | Plots include **dB** axis for spectrum; pie uses **bin-based export** energy % from `spectral_metrics` when available else **bin/candidate count** fallback (**labelled**, but still weaker than energy when fallback triggers). |

---

## E. Energy-ratio formulas

### E.1 Bin-based power in `complete_spectrum_df`

- `avg_power = mean(|STFT|², axis=time)`  
- `Power` column = **power** (energy per bin in the averaged-spectrum representation)  
- `harmonic_energy` / `inharmonic_energy` in `calculate_spectral_metrics`: **Σ Power** over respective DataFrames  

### E.2 `spectral_metrics` percentages (used first for batch H/I unless peak override)

\[
\text{harmonic\_energy\_percentage} = 100 \cdot \frac{\sum_{b \in H} P_b}{\sum_{b \in H} P_b + \sum_{b \in I} P_b}
\]

Inharmonic analogous. **Denominator: H + I only (musical band). Subbass S excluded.**

### E.3 `spectral_component_stats` global percentages

Harmonic / inharmonic / subbass global percentages are computed from **class energy sums** over **H, I, S** partitions of the full spectrum (post mask, post dB noise floor), then **normalised** so **H + I + S = 100%** when needed. `total_inharm_energy_pct_global` combines **inharmonic + subbass** shares for a “total inharmonicity” style readout.

### E.4 Batch row (`_analyze_single_file`)

1. Read `harmonic_energy_percentage` / `inharmonic_energy_percentage` from `spectral_metrics` (musical **H/(H+I)** style).  
2. Read `subbass_energy_pct_global` from `spectral_component_stats`.  
3. `global_sum = H + I + S`; if `|global_sum - 100| > 0.01`, **multiply each by `100/global_sum`**.  
4. Emit `batch_*_energy_ratio` = `pct / 100` after that rescaling.  
5. `batch_total_inharmonic_energy_ratio` = `(inharmonic_pct + subbass_pct) / 100` after rescaling (equals **1 − harmonic_ratio** in the three-way partition).

**Explicit batch provenance strings (current code):** `batch_energy_denominator = "harmonic_plus_inharmonic_plus_subbass_percent_scaled"`; `batch_energy_method = "spectral_component_stats_bin_or_peak_fallback"` (wording reflects fallback hierarchy in the batch row builder).

---

## F. Energy domain: amplitude / power / dB / normalised

| Quantity | Domain |
|----------|--------|
| Energy sums in `spectral_component_stats` | **Power** (`Power` or `Amplitude_linear²`) |
| `spectral_metrics` harmonic/inharmonic **energy** and **percentages** | **Power** |
| `harmonic_density` / `inharmonic_density` in `spectral_metrics` | **Sum of linear amplitudes** (not power) — **not** an energy ratio |
| dB columns / noise-floor filter | **dB** used for **thresholding only**; not used as energy for ratios |
| STFT magnitude | Linear **\|X\|**; power uses **\|X\|²** |

**Assessment:** Primary batch percentages (after choosing `spectral_metrics` + global subbass) are **power-based** at the source, but the **stitching** of musical-band H/I with global S and the **optional peak-based override** mean the row-level method is **acceptable but under-documented** unless consumers read `batch_energy_method` / `batch_energy_denominator`.

---

## G. Denominator analysis

| Stage | Denominator | Includes subbass? |
|-------|-------------|-----------------|
| `spectral_metrics` H% / I% | H + I | **No** |
| `spectral_component_stats` global H/I/S | H + I + S (after their internal normalisation) | **Yes** |
| Batch `global_sum` rescaling | H + I + S (after mixing sources, then scaled) | **Yes** |
| Orchestrator validation (post-fix) | **`H + I + S ≈ 1`** on 0–1 ratios, or **legacy `H+I≈100%` / `H+I+S≈100%`** on percentage columns | **Fixed:** valid global triplets are **no longer rejected** because `H + I < 1` (or `< 100%`). Model weights use **`H/(H+I)`**, **`I/(H+I)`** (musical-band denominator). |

**Risk:** Treating `harmonic_energy_percentage` + `inharmonic_energy_percentage` as a self-contained pair **without** reading `subbass_energy_percentage_global` **or** checking rescaling is **ambiguous**.

---

## H. Count-semantics table

| Field / key | Typical meaning | Classification |
|-------------|-----------------|----------------|
| `spectral_component_stats['harmonic_peak_count']` | `len(harmonic_df)` | **FFT-bin / row count** (misleading name “peak”) |
| `spectral_component_stats['inharmonic_peak_count']` | `len(inharmonic_df)` | **FFT-bin / row count** |
| `spectral_component_stats['subbass_peak_count']` | `len(subbass_df)` | **FFT-bin / row count** |
| `*_peak_count_peak_based` | From `_calculate_peak_based_energy` | **True local peak / partial-style** counts (validation track) |
| Batch `harmonic_count` | Prefers `harmonic_peak_count` from comp | Usually **bin-like**; fallback: **unique harmonic order** count capped by Nyquist |
| Batch `inharmonic_count` | `inharmonic_peak_count` or `len(inharmonic_df)` | **Bin-like** |
| Batch `subbass_count` | `len(subbass_df)` | **Bin-like** |
| Batch `total_inharm_count` | `len(total_inharmonic_df)` | **Bin-like** (I + S concat) |
| Main workbook `harmonic_order_count` (from `compile_metrics` / `proc_audio`) | Discrete n·f₀ detection for per-note analysis | **Unique harmonic order** semantics (documented in `compile_metrics.py`) |

**Misleading names:** `harmonic_peak_count` etc. in batch/super stats **≠** Salient peak count in the strict DSP sense for the bin-based track.

---

## I. Batch-to-main handoff analysis

**Reader:** `RobustOrchestrator.load_percentage_mapping()` — `pd.read_excel(batch_summary_path)`; filters rows where `file_name` contains TIER/MEAN/MEDIAN.

**Columns read:** `Note`, `file_name`, `harmonic_energy_percentage`, `inharmonic_energy_percentage`, `subbass_energy_percentage_global`, optional `batch_harmonic_energy_ratio` / `batch_inharmonic_energy_ratio` / `batch_subbass_energy_ratio`, `batch_energy_denominator`, `batch_energy_method`, `batch_total_inharmonic_energy_ratio`, optional `batch_ratio_fallback_reason` / `harmonic_energy_percentage_semantics`, plus `batch_ratio_source_explicit` (whether any `batch_*_ratio` cell was populated in Excel).

**Canonical batch ratios in `batch_summary`:** when **`batch_energy_denominator = harmonic_plus_inharmonic_plus_subbass`**, the three **`batch_*_energy_ratio`** fields are computed from **energy sums** `harmonic_energy_sum + inharmonic_energy_sum + subbass_energy_sum` so they **share one denominator and sum to 1** (not H/I on H+I with S appended separately).

**Model weights vs global profile:** Global batch ratios **H, I, S** (sum ≈ 1) are **preserved** in `applied_weights` (`batch_*_energy_ratio`). **`harmonic_weight` / `inharmonic_weight`** passed to `apply_filters_and_generate_data` are **model coefficients**: **`H/(H+I)`** and **`I/(H+I)`** when `H+I>0` — **not** raw global H and I. **Subbass is not** a third `proc_audio` weight.

**Validation (`_resolve_batch_energy_and_model_weights`):** (1) Full **0–1 triplet** when all three `batch_*_ratio` are present — **`|H+I+S−1| ≤ 0.02`**; explicit Excel ratios **> 1** rejected. (2) Else if **`subbass_energy_percentage_global`** present — legacy **`H+I+S ≈ 100%`**. (3) Else — legacy **`H+I ≈ 100%`**. On success: `model_weights_source` is **`batch_empirical_energy_ratios`** or **`legacy_batch_hi_percent`**. **Extreme derived weights** (e.g. **`model_harmonic_weight > 0.95`**, **`< 0.05`**, or **`< 0.5`** musical-band fraction) are **not** replaced: they are **logged** and surfaced as **`model_weights_warning`** (e.g. `extreme_empirical_model_weight_high`, `low_musical_harmonic_fraction`). Optional **`legacy_bounded_harmonic_weight` / `legacy_bounded_inharmonic_weight`** clip harmonic to **[0.05, 0.95]** for **legacy metrics only** — **`proc_audio` still receives the empirical `harmonic_weight` / `inharmonic_weight`**. **`model_weight_safety_guard_applied`** is **`false`** for this path. **`fallback_default`** is reserved for **invalid data** (missing fields, non-finite/negative values, bad triplet sum, **`H+I==0`**, explicit **`batch_* > 1`**).

**Fixed 5% / 0.95 references still present**

- `SuperAudioAnalyzer.__init__`: default `harmonic_weight=0.95`, `inharmonic_weight=0.05` (used when `auto_extract_weights` is false for internal combined metrics).  
- `run_compilation_phase`: defaults `harmonic_weight = 0.95`, `inharmonic_weight = 0.05` before overwriting from `first_result['applied_weights']` if present.  
- CLI `--inharmonic-weight` default `0.05` in `super_audio_analyzer.py` argparse.

So: **a fixed 5% inharmonic prior is still embedded as library defaults and compilation defaults**, even though the orchestrator’s **happy path** now uses **denominator-aware** batch validation and **`H/(H+I)`** model weights when the batch profile validates.

---

## J. Normalisation analysis

**Batch / super pipeline**

- **No RMS or peak normalisation** of the waveform for STFT: `librosa.load` returns float32 in approximately **[-1, 1]** per common practice; energy scales with **recording level**.  
- **dB for spectrogram display** uses `ref=np.max` on the spectrogram — **display / thresholding**, not the power sum.  
- Optional **high-pass** for very low notes (filename-derived f0 &lt; 30 Hz) — documented in metadata (`hpf_applied`).  
- **Per-frame** STFT only; time average reduces variance but does not remove level differences across files.

**Cross-file comparability:** Batch percentages are **within-spectrum partitions** (after mixing + rescaling). They are **not automatically comparable across different recording gains** as absolute physical measures unless recordings are calibrated.

---

## K. f0 / note-detection analysis

**Primary f0:** `detect_fundamental_frequency()` — **pYIN** (voiced, high confidence), **YIN**, **autocorrelation**; filename note gives **prior** for octave correction (`_select_best_f0`, ±octave tests, **50 cent** tolerance vs prior; if diverges, **prior wins**).

**Fallbacks:** If signal-based f0 missing, batch sets `fundamental_freq` from **filename note** (`filename_fallback`). If both fail, `0.0` with `f0_detection_method='failed'`.

**Batch audit fields:** `expected_freq_from_note`, `f0_error_cents`, `f0_detection_method`, `note_name` — **reasonable** for audit; still depend on **parseable filenames** (`test_0` style names → **no note**, weaker audit).

**STFT tier selection:** `_select_stft_parameters` uses **filename note → librosa.note_to_hz** to pick **90-tier** `n_fft` / hop — **not** the signal-based f0.

---

## L. Risks and ambiguities

1. ~~**Orchestrator `H+I≈100` gate**~~ — **addressed** in code (see §Tier 1 denominator/handoff fix).  
2. **Mixed-definition H/I vs S** before batch rescaling (see §G).  
3. **`harmonic_peak_count` naming** vs bin reality (§H).  
4. **Pie chart fallback** to **count-based** pie if energy metrics missing — visually confuses energy vs count.  
5. **`harmonic_density` / `inharmonic_density`** in batch table are **amplitude sums**, not energy — **easy to misread** as energy-related.  
6. **compile_metrics** default weight path **0.95/0.05** if no `applied_weights` on first result.  
7. **Main vs batch terminology gap:** `harmonic_order_count` (main) vs `harmonic_count` (batch) — **dangerously similar** without reading code.

---

## M. Scientific usability assessment

| Statement | Verdict | Brief justification |
|-----------|---------|---------------------|
| A. Batch estimates empirical H/I/S energy ratios | **Partially** | Power-based masks + global S, but musical-band H/I mixed with global S before rescaling complicates interpretation. |
| B. Batch can replace a fixed 5% inharmonic prior | **Partially** | **Orchestrator triplet + extreme-weight handling fixed**; **0.95/0.05** remains in defaults/compile path only. |
| C. Batch counts real harmonic partials | **No** | Bin-row counts dominate; peak-based partials exist mainly as **validation**. |
| D. Batch counts real inharmonic partials | **No** | Residual **bins** above cutoff, not salient inharmonic peak partials. |
| E. Batch counts FFT bins / candidates | **Yes** | Primary `*_peak_count` bin-based semantics. |
| F. Outputs safe to use as model weights verbatim | **Partially** | Use **`H/(H+I)`** mapping; read **`model_weights_source`** / **`model_weights_fallback_reason`**; global **H,I,S** are not the same as the two proc_audio weights. |
| G. Only naming/documentation fixes needed | **No** | Remaining issues: **mixed batch row construction**, **count naming**, compile defaults — beyond the orchestrator handoff fix. |
| H. Algorithmic changes needed | **Partially** | Core power STFT is fine; **denominator consistency** and **count naming** need algorithm-level clarity or pipeline alignment for “final” science use. |

---

## N. Ranked recommendations

### Tier 1 — must fix before final scientific use

1. ~~**Orchestrator validation vs batch triplet**~~ — **Implemented** (see §Tier 1 denominator/handoff fix): `H+I+S` validation, **`H/(H+I)`** model weights, explicit fallback metadata.  
2. **Document or remove 0.95/0.05 defaults** on every path that influences published metrics; ensure compilation cannot silently revert to **5% inharmonic** when `applied_weights` missing.  
3. **Single denominator policy** for batch export: either export **musical-band H/I** and **separate S**, or export **only** the triplet from one coherent source — avoid implicit rescaling without a published equation in the spreadsheet header.

### Tier 2 — should fix for clarity

1. Rename `harmonic_peak_count` (bin-based) to e.g. `harmonic_bin_row_count` in `spectral_component_stats`, keep true peak counts under `*_peak_based`.  
2. Rename batch `harmonic_count` → e.g. `harmonic_bin_row_count_export` or split **order count** vs **bin count**.  
3. Carry **`batch_energy_denominator`**, **`batch_energy_method`**, **`f0_detection_method`**, **`subbass_energy_percentage_global`** into orchestrator JSON next to `applied_weights`.  
4. Add **`model_weights_source`** and **`batch_row_file_name`** to every compiled row when weights originate from batch.

### Tier 3 — optional

1. Salient inharmonic **peak** pipeline parity with main analysis.  
2. JSONL export of batch summary for long-term deposition.  
3. GUI field tooltips referencing this audit.

---

## Tier 1 denominator/handoff fix (implemented)

**Date:** 2026-05-09 (code + targeted tests).

**Problem:** `run_main_analysis_phase` treated **`harmonic_energy_percentage + inharmonic_energy_percentage ≈ 100`** as a validity gate. After batch triplet normalisation, **global H + I + S ≈ 1** (or 100% with subbass), so **H + I is intentionally &lt; 1** when **S &gt; 0**. Valid empirical profiles were rejected and replaced by **`0.90` / `0.10`** fallbacks.

**Fix ( `pipeline_orchestrator_integrated.py` ):**

- **`_validate_batch_energy_triplet`**, **`_validate_legacy_hi_percent`**, **`_validate_legacy_trip_percent`**, **`_derive_model_weights_from_batch_energy`**, **`_resolve_batch_energy_and_model_weights`** — centralised validation and derivation.
- **Global batch denominator:** **`H + I + S ≈ 1`** (tolerance **0.02**) when all three **`batch_*_energy_ratio`** values are present (evaluated **first**). **Legacy:** **`H+I+S≈100%`** when **`subbass_energy_percentage_global`** is present (percentage path); **`H+I≈100%`** when there is no subbass percentage column.
- **Model weight denominator:** **`H + I`** (musical band only for the two-weight API): **`model_harmonic_weight = H/(H+I)`**, **`model_inharmonic_weight = I/(H+I)`** — **not** raw global H and I as `proc_audio` weights.
- **Subbass preserved** in **`applied_weights`**: **`batch_harmonic_energy_ratio`**, **`batch_inharmonic_energy_ratio`**, **`batch_subbass_energy_ratio`**, **`batch_energy_denominator`**, **`model_weight_denominator = "harmonic_plus_inharmonic"`**.
- **Fallbacks** use **`model_weights_source = "fallback_default"`** only for **invalid batch data** (not for extreme empirical weights). **`model_weights_warning`** documents extreme but **accepted** profiles. **`batch_empirical_energy_ratios`** / **`legacy_batch_hi_percent`** are never applied to fallback numeric priors.
- **Excel load:** skips rows with explicit **`batch_*_ratio &gt; 1`**; passes **`subbass_energy_percentage_global`**, **`batch_energy_denominator`**, provenance flags into the mapping payload.

**Batch export:** `batch_energy_denominator` string in `batch_audio_analyzer.py` aligned to **`harmonic_plus_inharmonic_plus_subbass`**.

**Tests:** `tests/test_batch_orchestrator_handoff.py`.

**Follow-up (extreme weights):** Removed orchestrator fallbacks that replaced empirical weights when **`model_harmonic_weight < 0.5`** or outside **`[0.05, 0.95]`**. Those cases now emit **`model_weights_warning`** and optional **`legacy_bounded_*`** fields only.

---

## GUI controls and model weights (desktop Spectrum Analyzer)

- The **Filters** tab shows **read-only** batch empirical ratios (when `batch_summary.xlsx` matches the first loaded audio file) and the **derived model coefficients** α/β. **Spectral masking** is not offered as a normal control; the main analysis path forces **`spectral_masking_enabled = False`** in metadata.
- **Harmonic / inharmonic sliders** are **not** primary scientific controls: default behaviour uses **batch-derived** **H/(H+I)** and **I/(H+I)** when the batch profile validates; otherwise documented fallbacks (**0.95 / 0.05**, tagged **`fallback_default`**). An **optional advanced checkbox** enables **manual** α/β for sensitivity work only, with **`model_weights_source = manual_override`** and explicit manual fields in export metadata.
- **`effective_partial_density`** and the **`Density_Metrics`** column set are unchanged by this GUI policy; batch vs manual weights affect **combined / legacy metric paths** that still consume **`harmonic_weight` / `inharmonic_weight`** in `proc_audio`.

---

## Appendix: related documentation

- `audio_analysis/BATCH_PROCESSING_README.md`  
- `audio_analysis/BATCH_CONFIG_README.md`  
- `docs/DENSITY_EXPORT_SCHEMA.md` (main workbook / density semantics; **not** batch-specific but needed for cross-reference)

---

### Audit completion checklist (for downstream tools)

| Question | Answer |
|----------|--------|
| `docs/BATCH_ANALYSIS_AUDIT.md` created? | **Yes** |
| `docs/BATCH_ANALYSIS_FIELD_MAP.md` created? | **Yes** (optional companion) |
| Fixed 5% / 0.95 fallback still exists? | **Yes** in super/compile defaults; orchestrator uses **0.90/0.10** only when **batch handoff invalid** — **not** for extreme empirical weights |
| Batch percentages amplitude / power / dB? | **Power-based** for core energy sums; **dB** for thresholds; **density** uses **linear amplitude sum** — **mixed** |
| Denominators clear? | **Partially** — **ambiguous** at H/I vs S junction without reading rescaling code |
| Batch values used as model weights? | **Derived** — `harmonic_weight`/`inharmonic_weight` = **`H/(H+I)`**, **`I/(H+I)`** from global profile; raw **H,I,S** stored separately; **subbass not** a third proc_audio weight |
| Count names misleading? | **Yes** for `*_peak_count` when bin-based |
| Scientifically usable as empirical energy profiles? | **Partially** — orchestrator handoff **fixed**; batch row construction / compile defaults still need care |
| Recommended next implementation pass | **Tier 1 (remaining):** silent **0.95/0.05** compile path; **batch row denominator** documentation in Excel headers |
| Full test suite run? | **No** (per audit constraints) |

---

## Publication path redaction

Batch tabular and JSON outputs (`batch_summary.xlsx`, `batch_results.json`, `super_analysis_results.json`, `analysis_metadata.json`, `metrics_summary.txt`, and related artefacts) **strip local absolute paths** by default (`REDACT_LOCAL_PATHS_FOR_PUBLICATION = True`). Rows include **`public_audio_id`**, **`source_file_basename`**, and **`source_file_hash_short`** where feasible. **`file_path`**, **`output_dir`**, and other `*_path` / `*_dir` fields must not contain `C:\Users\...` or POSIX home paths in publication bundles. Local paths may still appear in **console logs** during a desktop run. Verification: `python scripts/check_publication_paths.py <batch_results_folder>`.
