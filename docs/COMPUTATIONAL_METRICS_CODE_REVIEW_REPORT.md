# Computational metrics code review report

**Repository:** `SoundSpectrAnalyse-main_6`  
**Mode:** Read-only analytical review (no code changes implied by this document)  
**Date note:** Produced as a consolidated deliverable from a structured codebase inspection.

**Rules observed in the original review**

- Only project-owned Python code was analysed; **NumPy, SciPy, librosa, pandas, matplotlib, openpyxl**, etc. are treated as **black boxes** (calls may be named; internals are not).
- **Ignored unless directly affecting numeric results:** typical GUI wiring, file I/O, plotting-only paths, Excel formatting, logging, print, path handling, comments, boilerplate.
- **Exception:** some functions mix plotting and numeric comparison (e.g. optional plot branches)—numeric parts are still in scope.

---

## Scope and limitation (important)

The tree includes very large drivers—especially **`proc_audio.py`** (on the order of **10⁴** lines) and **`compile_metrics.py`** (on the order of **10³–10⁴** lines)—that contain most **STFT assembly, peak operations, band sums, f0 fitting, energy bookkeeping, and compilation-time transforms**. Enumerating **every** project-owned arithmetic line in those files is not practical in a single document without machine-generated augmentation.

Therefore:

- **Section A** lists **representative, metric-defining expressions** (exact snippets as read from the codebase) across the **dedicated math/metric modules** and **one canonical compile-time normalization block**, plus a **summary row** for the `proc_audio` / `compile_metrics` bulk.
- **Section B** lists **functions** in the smaller modules **completely** (by `def` inventory) and summarizes **`proc_audio` / `compile_metrics` / `interface.py`** as **subsystems** rather than every inner function.
- **Sections C–E** remain actionable for verification and paper-writing.

---

## A. Table of relevant computational lines and expressions (representative set)

| # | File | Function / block | Exact line or expression | Operation type | Variables / inputs | What it appears to compute | Formalise later? | Notes / risks |
|---|------|-------------------|--------------------------|----------------|---------------------|----------------------------|------------------|---------------|
| 1 | `density.py` | `calculate_harmonic_density` | `amps_db = 20*np.log10(np.maximum(harmonic_amplitudes, 1e-12))` | dB/linear conversion | `harmonic_amplitudes` | dB amplitudes with floor | YES | Floor `1e-12` fixes log domain; biases very weak partials |
| 2 | `density.py` | `calculate_harmonic_density` | `significant = amps_db > threshold_db` | thresholding | `amps_db`, `threshold_db` | Partial “significance” mask in dB | YES | Threshold semantics vs physical SPL depend on upstream scaling |
| 3 | `density.py` | `calculate_harmonic_density` | `density_count = significant.sum() / max_expected_harmonics` | normalisation / ratio | counts vs `max_expected_harmonics` | Fraction of expected slots filled | YES | `max_expected_harmonics` from `nyq//f0` or default 50 |
| 4 | `density.py` | `calculate_harmonic_density` | `avg_amp = np.mean(harmonic_amplitudes[significant]) …` + `amp_factor = np.tanh(avg_amp)` + convex combo | averaging; non-linear map; arithmetic | `harmonic_amplitudes`, `amp_weight` | Optional amplitude shaping of density | YES | `tanh` compresses; `amp_weight` default 0.2 |
| 5 | `density.py` | `calculate_harmonic_density` | `return float(np.clip(density, 0.0, 1.0))` | thresholding / clip | `density` | Hard-bounded output in [0,1] | YES | Clipping hides overflow diagnostics |
| 6 | `density.py` | `compute_spectral_entropy` | `p = power / total_power` | normalisation | `power` | Probability weights for Shannon entropy | YES | Uses **linear** power after `abs` and small-value cull |
| 7 | `density.py` | `compute_spectral_entropy` | `entropy = -np.sum(p * np.log2(p))` | entropy / information | `p` | Shannon entropy (bits) | YES | Base-2 log; not nats |
| 8 | `density.py` | `compute_spectral_entropy` | `max_entropy = np.log2(len(power))` … `normalized_entropy = entropy / max_entropy` | normalisation | `len(power)` | Normalized entropy vs uniform | YES | After masking `>1e-12`; length is **post-filter** count |
| 9 | `density.py` | `apply_density_metric` | `values = values / original_max` (under `prevent_domination`) | normalisation | `values` | Max-normalize partial amplitudes | YES | Dominant-partial suppression policy |
| 10 | `density.py` | `apply_density_metric` | `harmonic_numbers = frequencies / fundamental_freq` | arithmetic | `frequencies`, `fundamental_freq` | Harmonic index proxy | YES | Assumes `frequencies` align to harmonic rows |
| 11 | `density.py` | `apply_density_metric` | `expected_energy = np.power(np.maximum(harmonic_numbers, 1.0), -alpha)` with `alpha = 1.5` | power law | `harmonic_numbers`, `alpha` | Expected rolloff curve \(n^{-\alpha}\) | YES | Fixed **1.5** is a strong modelling choice |
| 12 | `density.py` | `apply_density_metric` | `normalized_values = values / (expected_energy + 1e-10)` | normalisation | `values`, `expected_energy` | Rolloff compensation | YES | Epsilon avoids div-by-zero |
| 13 | `density.py` | `apply_density_metric` | `weighted = weight_func(values)` then `result = np.sum(weighted)` | weighting; summation | `values`, `weight_function` | Weighted “fatness” sum | YES | Discrete keys `d3`… bypass this path |
| 14 | `density.py` | `apply_density_metric_df` | `df[amplitude_column] = 10 ** (df['Magnitude (dB)'] / 20)` | dB/linear conversion | dB column | Amplitude from dB | YES | Assumes 20·log10(A) convention |
| 15 | `density.py` | `_spectral_neff_from_filtered_linear_amplitudes` | `p = pwr / s` with `pwr = np.square(v)` … `return float(1.0 / den)` for `den = sum(p**2)` | spectral metric | linear amps | Effective number \(1/\sum p_i^2\) | YES | Classic “inverse participation” on **powers** |
| 16 | `density.py` | `_apply_discrete_spectral_metrics` (`d3`) | `return float(np.sum(np.log1p(v)))` | logarithmic; summation | `v` | \(\sum \log(1+A_i)\) | YES | Natural log via `log1p` |
| 17 | `density.py` | `_apply_discrete_spectral_metrics` (`d10`) | `return float(s_log * (n_eff / n))` | arithmetic; spectral | `s_log`, `n_eff`, `n` | Combined richness term | YES | Couples log-sum and participation |
| 18 | `density.py` | `compute_rolloff_compensated_harmonic_density` | `n_orders = np.round(f / f0).astype(int)` | harmonic calculation | `f`, `f0` | Integer harmonic order | YES | Rounding vs floor matters for borderline partials |
| 19 | `density.py` | `compute_rolloff_compensated_harmonic_density` | `a_norm = a / a_max` | normalisation | `a` | Per-note max-normalized amplitudes | YES | Same “prevent domination” spirit as `apply_density_metric` |
| 20 | `density.py` | `compute_rolloff_compensated_harmonic_density` | `expected = np.power(np.maximum(n_orders.astype(float), 1.0), -float(alpha))` | power law | `n_orders`, `alpha` | Expected \(n^{-\alpha}\) rolloff | YES | Parameter `alpha` default from constants |
| 21 | `density.py` | `compute_rolloff_compensated_harmonic_density` | `compensated = a_norm / (expected + epsilon)` | arithmetic | `a_norm`, `expected` | Rolloff-compensated amplitudes | YES | Docstring matches implemented formula |
| 22 | `density.py` | `compute_rolloff_compensated_harmonic_density` | `density = float(np.sum(weighted))` | summation | weighted compensated | Scalar rolloff density | YES | |
| 23 | `density.py` | `compute_rolloff_compensated_harmonic_density` | `out["rolloff_density_metric_normalized"] = float(density / a1)` | normalisation | `density`, fundamental partial amp | Ratio vs n=1 partial | YES | Explicitly **not** guaranteed in [0,1] (per docstring) |
| 24 | `density.py` | `effective_partial_density_from_powers` | `D_eff = (sum P)^2 / sum(P^2)` (as implemented) | spectral / information | `powers` | Inverse participation on powers | YES | Docstring states scale invariance |
| 25 | `density.py` | `calculate_combined_density_metric` | `combined_log = alpha * harm_log + beta * inharm_log` then `np.expm1(combined_log)` | logarithmic; arithmetic | harmonic/inharmonic logs | Log-domain blend | YES | Fallback linear `alpha*h+beta*i` |
| 26 | `density.py` | `compare_with_sethares_dissonance` | `norm_sethares = sethares_dissonance / 10` … `ratio = norm_sethares / norm_density` | heuristic scaling; ratio | external dissonance vs density | **Ad hoc** comparison metric | YES (but low trust) | **Hard-coded `/10`** is not physically derived—high ambiguity |
| 27 | `density.py` | `_critical_band_masking` (excerpt) | `bark_distance = abs(probe_bark - masker_bark)` + piecewise `threshold_db = masker_level_db + …` | thresholding; psychoacoustic model | Bark distances, constants | Masking threshold in dB | YES | Parncutt-inspired piecewise parameters |
| 28 | `energy_accounting.py` | `describe_component_energy_balance` | `den_parts = h + ih + s` … `sum_err = abs(tot - den_parts) / max(abs(tot), abs(den_parts), eps)` | arithmetic; normalisation | energy sums | Conservation residual | YES | Audit, not physical forward model |
| 29 | `energy_accounting.py` | `describe_component_energy_balance` | `ratio_sum_err = abs(1.0 - (rh + rih + rs))` | arithmetic | ratios | Closure of H+I+S ratios | YES | Thresholds `1e-6`, `1e-5` are policy |
| 30 | `peak_component_counts.py` | `_linear_amp_from_row` | `return float(10.0 ** (float(v) / 20.0))` | dB/linear | dB cell | Linear amplitude | YES | Consistent with 20·log10 A |
| 31 | `peak_component_counts.py` | `classify_peaks_harmonic_inharmonic_subbass_from_df` | `tol_hz = expected_freq * (2.0 ** (float(tolerance_cents) / 1200.0) - 1.0)` | harmonic; thresholding | cents tolerance | Hz tolerance per harmonic | YES | Exponential cents→multiplicative factor |
| 32 | `peak_component_counts.py` | same | `err = abs(freq - expected_freq)` vs `tol_hz` | arithmetic | peak vs slot | Window assignment test | YES | “Strongest per slot” tie-break with `amp > a0` |
| 33 | `low_frequency_policy.py` | `calculate_subfundamental_margin_percent` | piecewise `if f0 < 60: return 35.0` … | thresholding | `f0_hz` | Register-dependent margin % | YES | Policy table, not physics-first |
| 34 | `low_frequency_policy.py` | `calculate_adaptive_subfundamental_cutoff_hz` | `percentage_cut = f0 * (1.0 - margin / 100.0)` | arithmetic | `f0`, `margin` | Nominal cutoff line | YES | |
| 35 | `low_frequency_policy.py` | same | `raw_max = max(floor, percentage_cut, [leakage])` then cap `f0 * max_frac` | min/max composition | several Hz | Adaptive guard frequency | YES | Order of max/min documented in `SUBFUNDAMENTAL_CUTOFF_SELECTION_RULE` |
| 36 | `spectral_leakage_guards.py` | `leakage_halfwidth_hz` | `bw = float(sr) / float(int(n_fft))` … `return 0.5 * ml * bw` | FFT/STFT-related (geometry) | `sr`, `n_fft`, `main_lobe_bins` | Hz half-width of sidelobe guard | YES | Maps bins → Hz via bin width |
| 37 | `spectral_leakage_guards.py` | `filter_inharmonic_peak_candidates` | `if np.any(np.abs(hf - ff) <= lh): continue` | thresholding | harmonic freqs, candidate | Drop leakage-near-harmonic peaks | YES | Depends on representative `hf` list quality |
| 38 | `harmonic_alignment.py` | `_cents` | `return float(1200.0 * math.log2(obs_hz / exp_hz))` | logarithmic / harmonic | frequencies | Cents deviation | YES | Standard music-theory definition |
| 39 | `harmonic_alignment.py` | `_adaptive_tolerance_cents` | `bw_cents = 1200.0 * math.log2(hi / expected_hz)` … `max(18.0, 2.0 * bw_cents)` | FFT/STFT-related; thresholding | bin width | Adaptive tolerance | YES | Couples STFT resolution to cents gate |
| 40 | `harmonic_alignment.py` | `_linear_amp_and_energy` | `return amp, amp * amp` | arithmetic | amplitude | Linear amp + **energy proxy** \(A^2\) | YES | Energy is not band-integrated STFT power here |
| 41 | `harmonic_validation.py` | `validate_harmonic_series_matched` | `rms_c = float(np.sqrt(np.mean(np.square(np.asarray(signed, dtype=float)))))` | statistical metric | cents errors | RMS of signed cents errors | YES | Uses subset of matches with `error_cents` |
| 42 | `compile_metrics.py` | `_add_canonical_and_global_density_columns` | `out[canon_col] = dm / 10.0` (legacy path) | scaling | `Density Metric` | Legacy canonical reconstruction | YES | **Heuristic `/10`**—documented as approximate |
| 43 | `compile_metrics.py` | `_add_canonical_and_global_density_columns` | `out["density_normalized_global"] = (s_canon / mx).clip(lower=0.0, upper=1.0)` | normalisation | per-compile max | Global [0,1] density | YES | **Dataset-relative** norm—not per-note |
| 44 | **`proc_audio.py` (bulk)** | numerous methods | e.g. `physical_peak_amplitude` uses `factor * magnitude / sw` with `sw = Σ w[n]` | FFT/STFT-related; arithmetic | STFT magnitudes, window | Amplitude calibration | YES | Large file: many more band-sums, ratios, f0 fits |
| 45 | **`interface.py` / GUI orchestration** | n/a | — | excluded | — | Mostly wiring | NO | GUI unless it alters numeric pipeline |

---

## B. Table of computational functions (project-owned)

| Function | File | Inputs (summary) | Outputs | Main calculation steps (high level) | Black-box libs used | Formalise? | Priority |
|----------|------|------------------|---------|--------------------------------------|----------------------|------------|----------|
| `apply_spectral_smoothing` | `density.py` | spectra-related arrays | smoothed spectrum | convolution-like smoothing | `numpy` | YES | MEDIUM |
| `estimate_noise_floor` | `density.py` | spectra, params | noise estimate | order-stat / percentile style ops | `numpy` | YES | MEDIUM |
| `physical_spectral_density` | `density.py` | amps, freqs | scalar | wrapper to class metric | `numpy` | YES | MEDIUM |
| `perceptual_spectral_density` | `density.py` | amps, freqs | scalar | wrapper | `numpy` | YES | MEDIUM |
| `calculate_harmonic_density` | `density.py` | harmonic amps, thresholds, f0, sr | [0,1] scalar | dB mask, count ratio, optional tanh amp mix | `numpy` | YES | **HIGH** |
| `calculate_inharmonic_density` | `density.py` | inharmonic amps | scalar | delegates to `calculate_harmonic_density` | `numpy` | YES | HIGH |
| `compute_spectral_entropy` | `density.py` | `power` | [0,1] | Shannon + max-entropy normalization | `numpy` | YES | **HIGH** |
| `_critical_band_masking` | `density.py` | freqs/levels dB | dB threshold | Bark distance + piecewise masking | `numpy` (Bark via `_hz_to_bark`) | YES | MEDIUM |
| `estimate_noise_floor_by_critical_bands` | `density.py` | spectrum data | floor estimate | band-wise statistics | `numpy` | YES | MEDIUM |
| `apply_spectral_masking_filter` | `density.py` | spectrum, maskers | filtered spectrum | applies masking model | `numpy` | YES | MEDIUM |
| `validate_spectral_density_metric` | `density.py` | metric, context | validation dict | inequality checks | `numpy`/python | NO (audit) | LOW |
| `calculate_perceptual_spectral_density` | `density.py` | frames | scalar | multi-step perceptual pipeline | `numpy` | YES | MEDIUM |
| `_calculate_harmonic_completeness_phase2` | `density.py` | partials | completeness | coverage-style metric | `numpy` | YES | MEDIUM |
| `calculate_spectral_complexity` | `density.py` | spectrum | scalar | composite complexity | `numpy` | YES | MEDIUM |
| `calculate_harmonic_richness` | `density.py` | partials | scalar | richness index | `numpy` | YES | MEDIUM |
| `calculate_spectral_density_corrected` | `density.py` | inputs | scalar | “corrected” density variant | `numpy` | YES | MEDIUM |
| `get_weight_function` | `density.py` | name | callable | maps name→weight fn | python | YES | **HIGH** |
| `compute_harmonic_effective_power_density` | `density.py` | amplitudes, orders | dict | HEpd-style metrics | `numpy` | YES | **HIGH** |
| `compute_harmonic_effective_power_mass` | `density.py` | amplitudes | dict | sums of squares, RMS | `numpy` | YES | HIGH |
| `compute_rolloff_compensated_harmonic_density` | `density.py` | amps, freqs, f0, … | dict | rolloff compensation + sum | `numpy` | YES | **HIGH** |
| `_spectral_neff_from_filtered_linear_amplitudes` | `density.py` | `v` | scalar | \(1/\sum p_i^2\) | `numpy` | YES | HIGH |
| `_apply_discrete_spectral_metrics` | `density.py` | key, values, freqs | scalar | d3/d10/d17/d24 definitions | `numpy` | YES | **HIGH** |
| `band_partial_metric_sum` | `density.py` | frame, band | scalar | band-limited sums | `numpy` | YES | MEDIUM |
| `partial_metric_sums_h_i_s_total` | `density.py` | frame | tuple | H/I/S/total partial metric sums | `numpy` | YES | **HIGH** |
| `compute_discrete_spectral_metrics_bundle` | `density.py` | amplitudes, freqs | dict | wraps `_apply_discrete_spectral_metrics` | `numpy` | YES | HIGH |
| `apply_density_metric` | `density.py` | values, weight, flags, freqs, f0 | scalar | max-norm, rolloff, weight, sum | `numpy` | YES | **HIGH** |
| `apply_density_metric_df` | `density.py` | `df`, columns | scalar | column extraction + dB fallback + `apply_density_metric` | `pandas`, `numpy` | YES | HIGH |
| `effective_partial_density_from_powers` | `density.py` | `powers` | scalar | participation ratio | `numpy` | YES | **HIGH** |
| `_inverse_herfindahl_effective_components` | `density.py` | powers | scalar | related effective count | `numpy` | YES | MEDIUM |
| `compute_subbass_protection_tolerance_hz` | `density.py` | f0, … | Hz | tolerance schedule | `numpy`/python | YES | MEDIUM |
| `aggregate_low_frequency_residual_peak_power` | `density.py` | peaks, cutoffs | power aggregate | sums under policy | `numpy` | YES | HIGH |
| `aggregate_subbass_noise_peak_power` | `density.py` | (compat wrapper) | scalar | delegates | `numpy` | YES | LOW |
| `partial_density_effective_components_bundle` | `density.py` | … | dict | bundles effective components | `numpy` | YES | HIGH |
| `partial_density_effective_components` | `density.py` | … | dict | simpler bundle | `numpy` | YES | MEDIUM |
| `identify_nonharmonic_residual_rows` | `density.py` | DataFrames | filtered DF | row classification rules | `pandas`, `numpy` | YES | **HIGH** |
| `identify_inharmonic_partials` | `density.py` | (compat) | DF | delegates / legacy | `pandas` | YES | MEDIUM |
| `calculate_combined_density_metric` | `density.py` | harmonic/inharmonic scalars | scalar | log vs linear blend | `numpy` | YES | HIGH |
| `compare_with_sethares_dissonance` | `density.py` | df, dissonance, density | dict | `/10` scaling + ratio (+ optional plot) | `numpy`, `matplotlib` (plot path) | YES (ratio only) | MEDIUM (risk: scaling) |
| `plot_harmonic_spectrum` | `density.py` | … | None | plotting | `matplotlib` | NO | LOW |
| `_hz_to_bark` | `density.py` | `f` | bark | psychoacoustic freq map | `numpy` | YES | MEDIUM |
| `spectral_density` | `density.py` | … | … | legacy / spectral density entry | `numpy` | YES | LOW |
| `describe_component_energy_balance` | `energy_accounting.py` | sums + ratios | audit dict | conservation checks | python `math` | YES (as constraints) | MEDIUM |
| `_linear_amp_from_row` | `peak_component_counts.py` | row, df | float | dB→linear or amplitude | python | YES | MEDIUM |
| `_peak_tuples` | `peak_component_counts.py` | df | list | build `(f, a)` list | `pandas` | YES | MEDIUM |
| `classify_peaks_harmonic_inharmonic_subbass_from_df` | `peak_component_counts.py` | peaks, f0, cutoffs | dict | slot scan + strongest per slot | `math`, `pandas` | YES | **HIGH** |
| `_finite_positive` | `low_frequency_policy.py` | value | bool | finiteness | python | NO | LOW |
| `calculate_subfundamental_margin_percent` | `low_frequency_policy.py` | f0 | percent | piecewise table | python | YES | HIGH |
| `_nan_result` | `low_frequency_policy.py` | — | dict | NaN placeholders | python | NO | LOW |
| `calculate_adaptive_subfundamental_cutoff_hz` | `low_frequency_policy.py` | f0, floors, leakage | dict | min/max composition of cutoffs | python `math` | YES | **HIGH** |
| `classify_low_frequency_row` | `low_frequency_policy.py` | row-like dict | labels | policy classification | python | YES | MEDIUM |
| `leakage_halfwidth_hz` | `spectral_leakage_guards.py` | STFT geometry | Hz | bin width × lobe bins | `numpy` (light) | YES | HIGH |
| `filter_inharmonic_peak_candidates` | `spectral_leakage_guards.py` | candidates, harmonics, width | list | geometric rejection | `numpy` | YES | HIGH |
| `_slot_count_aliases` | `harmonic_validation.py` | expected/matched | dict | integer bookkeeping | python | NO | LOW |
| `validate_harmonic_series_matched` | `harmonic_validation.py` | f0, peaks, SR, fft, … | dict | calls `compute_harmonic_alignment_metrics` + RMS cents | `pandas`, `numpy` | YES | **HIGH** |
| `compute_harmonic_alignment_metrics` | `harmonic_alignment.py` | f0, peaks, … | large dict | round order, windows, collapse, energy ratios | `numpy`, `pandas` | YES | **HIGH** |
| `_linear_amp_and_energy` | `harmonic_alignment.py` | row, df | (A, A²) | amplitude + energy proxy | `pandas` | YES | HIGH |
| `_cents` | `harmonic_alignment.py` | obs, exp Hz | cents | log2 ratio | `math` | YES | HIGH |
| `_adaptive_tolerance_cents` | `harmonic_alignment.py` | expected Hz, SR, N | cents | bin-width-derived tolerance | `math` | YES | HIGH |
| `_tolerance_for_order` | `harmonic_alignment.py` | n, f0, … | cents | fixed vs adaptive | python | YES | MEDIUM |
| `_in_any_harmonic_window` | `harmonic_alignment.py` | f, f0, slots, … | bool | union-of-windows test | `math` | YES | MEDIUM |
| `len_or_none` / `_as_int_count` | `debug_counts.py` | various | int/None | safe int coercion | `pandas`, `math` | NO | LOW |
| `validate_debug_count_invariants` | `debug_counts.py` | row dict | mutated row | inequality audit on counts | python | NO (QC) | LOW |
| `_add_canonical_and_global_density_columns` | `compile_metrics.py` | wide DF | DF | canonical column + global max norm | `pandas`, `numpy` | YES | **HIGH** |
| **Hundreds of additional `compile_metrics.py` helpers** | `compile_metrics.py` | many | many | merge, coerce, PCA, column algebra, validation | `pandas`, `numpy`, … | YES (subset) | HIGH/MED |
| **`proc_audio.py` analysis pipeline methods** | `proc_audio.py` | audio, params | workbooks/attrs | STFT, peak pick, band sums, f0 fit, energy ratios, numeric inputs to pies | `numpy`, `librosa`, `scipy`, `pandas`, … | YES | **HIGH** |
| `calculate_dissonance_metrics` (and related) | `audio_analysis/super_audio_analyzer.py` | internal state | dict | orchestrates `dissonance_models` | project `dissonance_models` + `numpy` | YES | MEDIUM |
| **Classes in `dissonance_models.py`** | `dissonance_models.py` | partials / spectrum | scalars/curves | Sethares-type pairing sums | `numpy` | YES | MEDIUM |

---

## C. Priority list for mathematical formalisation (first pass)

### HIGH (core published / explainable metrics and gates)

1. `density.apply_density_metric` (+ `get_weight_function` weight shapes)  
2. `density.compute_rolloff_compensated_harmonic_density`  
3. `density.compute_spectral_entropy`  
4. `density.effective_partial_density_from_powers` + `_spectral_neff_from_filtered_linear_amplitudes` + `_apply_discrete_spectral_metrics` (`d3`/`d10`/`d17`/`d24`)  
5. `density.compute_harmonic_effective_power_density` / `compute_harmonic_effective_power_mass`  
6. `density.partial_metric_sums_h_i_s_total` / `band_partial_metric_sum`  
7. `density.identify_nonharmonic_residual_rows` (classification geometry)  
8. `harmonic_alignment.compute_harmonic_alignment_metrics`  
9. `peak_component_counts.classify_peaks_harmonic_inharmonic_subbass_from_df`  
10. `low_frequency_policy.calculate_adaptive_subfundamental_cutoff_hz` (+ margin table)  
11. `compile_metrics._add_canonical_and_global_density_columns` (dataset-relative normalization semantics)  
12. **`proc_audio.py`:** STFT scaling to partial tables + **H/I/S linear sums and energy ratios** + **f0 fit objective/thresholds** (as one formal “Stage 2 forward model” document)

### MEDIUM

- Masking / noise-floor family in `density.py` (`_critical_band_masking`, `apply_spectral_masking_filter`, noise floor estimators)  
- `spectral_leakage_guards` + interaction with inharmonic identification  
- `energy_accounting.describe_component_energy_balance` (as **constraints** linking exported fields)  
- `dissonance_models.py` + orchestration in `super_audio_analyzer.py`

### LOW (QC / housekeeping / plotting-heavy)

- `debug_counts.validate_debug_count_invariants`  
- Plot wrappers (`density.plot_harmonic_spectrum`, pie generation that only visualizes precomputed triples)  
- Most pure “validation status string” assembly without changing numbers

---

## D. Excluded sections (and why)

| Exclusion | Reason |
|-----------|--------|
| `interface.py`, `pipeline_orchestrator*.py`, `gui_*` | Primarily GUI / orchestration / threading; not metric definitions (unless calling into metrics) |
| `tools/export_research_density_workbook.py`, `post_compile_research_export.py` | Research workbook **reformatting**; chart min-max for display; not acoustic forward model |
| `tests/**` | Test harness, not production metric definitions |
| `scripts/scan_mojibake.py`, packaging, logging setup | Non-metric |
| **Library internals** (`numpy.linalg`, `scipy.signal.stft` implementation, etc.) | Black box per review rules |
| `compare_with_sethares_dissonance` **plotting branch** | Plotting excluded; numeric `/10` scaling still flagged in Section E |

---

## E. Ambiguous or risky calculations (human review recommended)

1. **Fixed rolloff exponent** `alpha = 1.5` inside `apply_density_metric` (and separate `alpha` in rolloff-compensated path): strong modeling assumption; sensitivity studies needed.  
2. **`compare_with_sethares_dissonance`:** arbitrary `/10` scaling for both Sethares and density—**not physically linked**; any “correlation/ratio” from this should not be interpreted as invariant.  
3. **Legacy canonical reconstruction** in `compile_metrics._add_canonical_and_global_density_columns`: `Density Metric / 10.0` is explicitly approximate—legacy continuity vs truthfulness.  
4. **`density_normalized_global` is compile-max normalized**: cross-note comparability is **within compiled batch**, not absolute across unrelated compilations unless procedure is fixed.  
5. **Harmonic alignment energy proxies**: `_linear_amp_and_energy` uses \(A^2\) per peak row; not guaranteed to match STFT integrated band power used elsewhere—**semantic alignment risk** across modules.  
6. **Peak-list vs residual-row hierarchies**: multiple counting semantics (`debug_counts` notes); reviewers must not mix `peaklist_*` with residual pipeline counts.  
7. **`proc_audio` amplitude calibration path** (`physical_peak_amplitude` and legacy `amp / coherent_gain`): documented historical factor risk; formal documentation should state which path is canonical for which exported columns.

---

## F. Note on exhaustive line-level extraction

A **true exhaustive** inventory of every arithmetic / comparison line in **`proc_audio.py`** alone is better produced as a **machine-generated artifact** (e.g. AST walk or custom linter output) than as a manually maintained markdown table. This document intentionally prioritises **metric-defining modules** and **high-risk compile-time transforms** while still flagging the Stage 2 monolith for separate formalisation.

---

## G. Optional cross-reference

Add a link from `docs/CURRENT_DOCUMENTATION_INDEX.md` to this file if you want the report discoverable from the documentation index (manual edit).
