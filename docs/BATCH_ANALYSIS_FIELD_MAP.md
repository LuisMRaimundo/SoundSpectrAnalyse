# Batch-analysis field map (quick reference)

> **Canonical pipeline:** `proc_audio` → `spectral_analysis.xlsx` → `compile_density_metrics_with_pca` → `compiled_density_metrics.xlsx` (**`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`**). This page maps **batch / SuperAudioAnalyzer** outputs only.

Companion to `docs/BATCH_ANALYSIS_AUDIT.md`. Semantics are **inferred from code** in `batch_audio_analyzer.py` and `super_audio_analyzer.py`.

## batch_summary.xlsx / batch_summary.csv (per data row)

| Column / key | Origin | Type | Notes |
|--------------|--------|------|-------|
| `file_name` | Input path | **Audit** | Primary key for de-duplication |
| `file_path` | Input path | **Audit** | **Publication exports:** value **`redacted_for_publication`**; use **`public_audio_id`** / **`source_file_basename`** / **`source_file_hash_short`** instead |
| `Note` | Regex on `file_name` | **Audit** | May be empty if no pitch in name |
| `fundamental_freq` | `SuperAudioAnalyzer` + batch fallbacks | **Hz** | Signal-based; filename fallback |
| `expected_freq_from_note` | librosa `note_to_hz` | **Hz** | Validation |
| `f0_error_cents` | 1200·log₂(detected/expected) | **Audit** | |
| `f0_detection_method` | `results['frequency_analysis']['method_used']` | **Audit** | |
| `harmonic_energy_percentage` | `spectral_metrics` then optional peak override; + triplet normalise | **Percentage** | Historically **H/(H+I)** source mixed with S |
| `inharmonic_energy_percentage` | same | **Percentage** | |
| `subbass_energy_percentage_global` | `spectral_component_stats` | **Percentage** | Global band |
| `total_inharm_energy_percentage_global` | `spectral_component_stats` | **Percentage** | Inharmonic + subbass (global readout) |
| `batch_harmonic_energy_ratio` | **`harmonic_energy_sum / (H+I+S)`** when energy sums &gt; 0 | **Ratio** | **Canonical global H+I+S** fractions; **sum = 1** by construction |
| `batch_inharmonic_energy_ratio` | **`inharmonic_energy_sum / (H+I+S)`** | **Ratio** | Same denominator as `batch_harmonic_energy_ratio` |
| `batch_subbass_energy_ratio` | **`subbass_energy_sum / (H+I+S)`** | **Ratio** | Same denominator |
| `batch_total_inharmonic_energy_ratio` | **`(I+S) / (H+I+S)`** (energy sums) | **Ratio** | `(inharmonic+subbass) / total` |
| `batch_ratio_fallback_reason` | text when energy-sum path unavailable | **Audit** | Set when **`H+I+S` energy sum ≤ 0** and legacy percentages are used instead |
| `harmonic_energy_percentage_semantics` | documentation string | **Audit** | Clarifies legacy `%` columns vs canonical **`batch_*_energy_ratio`** |
| `batch_energy_denominator` | constant string | **Audit** | Canonical value: **`harmonic_plus_inharmonic_plus_subbass`** |
| `batch_energy_method` | string | **Audit** | e.g. **`global_energy_sum_H_I_S`** or fallback label when sums are zero |
| `batch_ratio_source_explicit` | batch writer | **Audit** | **True** when canonical **`batch_*_energy_ratio`** cells were populated (vs legacy `%`-only inference) |
| `harmonic_count` | comp `harmonic_peak_count` or DF logic | **Count** | Usually **bin rows**, not orders |
| `inharmonic_count` | comp / `len(inharmonic_df)` | **Count** | **Bin rows** |
| `subbass_count` | `len(subbass_df)` | **Count** | **Bin rows** |
| `total_inharm_count` | `len(total_inharmonic_df)` | **Count** | **Bin rows** |
| `sum_*` / `mean_*` / `median_*` | `spectral_component_stats` | **Energy** (power domain sums / moments) | |
| `harmonic_plus_inharmonic_energy_sum` | comp | **Energy** | Σ partials H+I energy |
| `ground_noise_energy_sum` | comp | **Energy** | Residual not in H/I/S |
| `harmonic_density` | `spectral_metrics` | **Unclear / not energy** | Sum of **linear amplitudes** |
| `inharmonic_density` | `spectral_metrics` | **Unclear / not energy** | Sum of **linear amplitudes** |
| `spectral_entropy` | metrics | **Model / stat** | Normalised distribution metric |
| `pairwise_dissonance` | dissonance block | **Model / stat** | |
| `rms_stationary` | metadata | **Amplitude RMS** | Middle 70% segment |
| `energy_conservation_error_pct` | comp | **Debug** | |
| `analysis_date` | batch | **Audit** | |
| `success` | batch | **Audit** | |

**Summary rows:** `MEAN`, `MEDIAN`, `TIER_*` rows — **aggregates**, not raw files.

## Per-file `harmonic_components.csv` / `inharmonic_components.csv`

| Column | Type |
|--------|------|
| `Frequency (Hz)` | STFT **bin centre** for this CSV row (batch harmonic/inharmonic component tables). **Not** the same column set as per-note **`spectral_analysis.xlsx`** `Harmonic Spectrum` exports, which carry **`bin_center_frequency_hz`**, **`interpolated_frequency_hz`**, **`extracted_frequency_hz`**, and validity flags — see **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`** §3. |
| `Amplitude_linear` | **Amplitude** (RMS per bin) |
| `Power` | **Energy** ∝ amplitude² |
| `Magnitude_dB` | **dB** (display / threshold) |
| `Harmonic Number` (harmonic only) | **Order index** derived from `round(f/f0)` — multiple bins can share an order |

## `spectral_component_stats` (selected keys used by batch)

| Key | Typical semantics |
|-----|-------------------|
| `harmonic_energy_sum` | **Energy** |
| `inharmonic_energy_sum` | **Energy** |
| `subbass_energy_sum` | **Energy** |
| `harmonic_energy_pct_global` | **Percentage** |
| `inharmonic_energy_pct_global` | **Percentage** |
| `subbass_energy_pct_global` | **Percentage** |
| `harmonic_energy_pct_musical` | **Percentage** (H vs I, no S) |
| `harmonic_peak_count` | **FFT-bin row count** (name misleading) |
| `*_peak_based` variants | **Peak-based validation track** |

## Orchestrator mapping → `proc_audio`

| Batch / Excel | Role |
|---------------|------|
| `batch_harmonic_energy_ratio`, `batch_inharmonic_energy_ratio`, `batch_subbass_energy_ratio` | **Global measured profile** (0–1), **H+I+S≈1** when all present |
| `harmonic_energy_percentage`, `inharmonic_energy_percentage`, `subbass_energy_percentage_global` | **Legacy 0–100 columns**; used when full ratio triplet absent — **H+I+S≈100%** if subbass present |
| **`harmonic_weight` / `inharmonic_weight`** (API names) | **Model coefficients** = **`H/(H+I)`**, **`I/(H+I)`** with **H,I** from validated global profile — **not** raw global H, I |
| `subbass_energy_percentage_global` | Drives **H+I+S** validation; **not** a third weight argument to `apply_filters_and_generate_data` |

### `applied_weights` metadata (per main-analysis result)

| Key | Meaning |
|-----|---------|
| `model_harmonic_weight` / `model_inharmonic_weight` | Same as `harmonic` / `inharmonic` — **model** blend coefficients |
| `model_weights_source` | **`batch_empirical_energy_ratios`** \| **`legacy_batch_hi_percent`** \| **`fallback_default`** |
| `model_weights_fallback_reason` | Non-empty only when `fallback_default` (**invalid batch data**), not for extreme empirical weights |
| `model_weights_warning` | Optional pipe-separated codes (e.g. **`extreme_empirical_model_weight_high`**, **`low_musical_harmonic_fraction`**) when weights are unusual but **still empirical** |
| `model_weight_safety_guard_applied` | **`false`** — empirical weights are **not** overwritten for being extreme |
| `legacy_bounded_harmonic_weight` / `legacy_bounded_inharmonic_weight` | Harmonic clipped to **[0.05, 0.95]** with partner summing to 1 — **for legacy metrics only**, not passed as main `proc_audio` weights |
| `batch_energy_denominator` | **`harmonic_plus_inharmonic_plus_subbass`** (or from batch row) |
| `model_weight_denominator` | **`harmonic_plus_inharmonic`** |
| `batch_*_energy_ratio` | Preserved global measured ratios |
| `batch_ratio_source_explicit` | **Whether** canonical **`batch_*_ratio`** cells were populated in **`batch_summary.xlsx`** (vs legacy `%`-only inference) |

## Compiled workbook `Per_Note_Processing_Metadata` (per note row)

Mirrors the per-file `spectral_analysis.xlsx` sheet of the same name. When Phase 3 ran with a validated batch handoff, expect the same keys as **`applied_weights`** above plus **`batch_ratio_source_explicit`** (stored as **`true`/`false`** strings in Excel for stable text merge). **`Density_Metrics`** intentionally omits these columns — use this sheet for **H+I+S global profile** vs **H/(H+I)** model coefficients audit.

---

## Publication path redaction

- **`REDACT_LOCAL_PATHS_FOR_PUBLICATION`** defaults to **True**; batch JSON/Excel/text must not embed **`C:\Users\...`**, **`/home/...`**, **`/Users/...`**, or **`/mnt/...`** in publication deposits.
- Prefer **`public_audio_id`**, **`source_file_basename`**, **`source_file_hash_short`**, **`file_name`**, **`Note`**, **`instrument`**, **`dynamic`** for cross-run identity.
- **`output_dir`**, **`file_path`**, and similar keys are **`redacted_for_publication`** in exported rows. See **`metadata_sanitizer.enrich_and_redact_batch_audio_result`** and **`scripts/check_publication_paths.py`**.
