# Code–Formula Traceability Table

## 1. Purpose

This document links **project-owned Python** expressions to the **mathematical formulas** recorded in the Pass 1–15 formula-extraction tables (`docs/formula_extraction/FORMULA_EXTRACTION_TABLE_*.md`). It supports **auditability** (code ↔ notation) and **PDF-friendly** technical documentation. Third-party library internals are **not** traced beyond the call boundary unless the extraction table explicitly mirrors a public API (for example `numpy.percentile` used with the same arguments as in project code).

**Line numbers** were taken from the repository at the time this note was generated; they are **approximate** for long functions. **Line numbers should be rechecked after code changes.**

**Mathematics (web / StackEdit / Stack.io):** Inline math in this document uses **paired dollar delimiters** (`$ … $`), which MathJax and KaTeX treat as inline math by default. If your site expects additional delimiter pairs (for example other LaTeX inline delimiters), extend the MathJax `tex.inlineMath` option per the MathJax documentation.

## 2. Coverage

Formula-extraction and validation work spans **Passes 1–15**:

- Pass 1 — density metrics (first-pass table)
- Pass 2 — weight functions
- Pass 3 — partial sums and metric bundles
- Pass 4 — residual / inharmonic classification
- Pass 5 — harmonic alignment
- Pass 6 — peak component counts
- Pass 7 — low-frequency policy
- Pass 8 — spectral leakage guards
- Pass 9 — compile-time normalisation
- Pass 10 — selected `proc_audio` formulas
- Pass 11 — extended density metrics
- Pass 12 — dissonance models
- Pass 13 — peak detection and f₀ refinement (`proc_audio.py`)
- Pass 14 — compile extraction and batch mass (`compile_metrics.py`, `super_audio_analyzer.py`)
- Pass 15 — data integrity normalisation (`data_integrity.py`)

Entries below **merge** multiple extraction-table rows for the same function where the mathematics is sequential in one implementation block; **sub-expressions** omitted here remain listed in the corresponding extraction table.

## 3. Traceability entries

### Pass 1 — `density.py` (density metrics, first pass)

### 1. `compute_spectral_entropy` — normalised Shannon entropy on spectral mass

| Field | Content |
|---|---|
| Function | `compute_spectral_entropy` |
| What it does | Builds a probability mass on strictly positive powers, computes Shannon entropy in bits, normalises by uniform entropy, clips to $[0,1]$. |
| Script | `density.py` |
| Line(s) | approx. 559–609 |
| Python code | `power = np.abs(power)`; `power = power[power > 1e-12]`; `total_power = np.sum(power)`; `p = power / total_power`; `entropy = -np.sum(p * np.log2(p))`; `max_entropy = np.log2(len(power))`; `normalized_entropy = entropy / max_entropy` (else `0`); `np.clip(normalized_entropy, 0.0, 1.0)` |
| Mathematical formula | Let $u_i=\lvert x_i\rvert$, $\mathcal{I}=\{i\mid u_i>10^{-12}\}$, retain masses $P_j=u_{i_j}$, $S=\sum_j P_j$, $p_j=P_j/S$ with $\sum_j p_j=1$. Then $H=-\sum_j p_j\log_2 p_j$, $H_{\max}=\log_2 N$ for $N=\lvert\mathcal{I}\rvert$, $\tilde H=H/H_{\max}$ if $H_{\max}>0$ else $0$, and $H_{\mathrm{out}}=\operatorname{clip}(\tilde H,0,1)$. |
| Symbol definitions | $x_i$: input `power` elements; $p_j$: normalised masses; $H$: Shannon entropy (base 2). |
| Notes | Empty or all-subthreshold inputs return `0.0` early (guards). |

### 2. `effective_partial_density_from_powers` — effective number / IPR on powers

| Field | Content |
|---|---|
| Function | `effective_partial_density_from_powers` |
| What it does | Computes inverse-participation-style effective density from nonnegative powers. |
| Script | `density.py` |
| Line(s) | approx. 2267–2320 |
| Python code | `d = (s * s) / ss` with $s=\sum P_i$, $ss=\sum P_i^2$ (see extraction table; finite-positive guards in code). |
| Mathematical formula | $D_{\mathrm{eff}} = S_1^2/S_2$ with $S_1=\sum_i P_i$, $S_2=\sum_i P_i^2$ over finite $P_i>\varepsilon$. |
| Symbol definitions | $P_i$: per-bin power masses. |
| Notes | Extraction Pass 1: inverse participation ratio; code variables `s` and `ss` correspond to $S_1$ and $S_2$. |

### 3. `_spectral_neff_from_filtered_linear_amplitudes` — $N_{\mathrm{eff}}$ from linear amplitudes

| Field | Content |
|---|---|
| Function | `_spectral_neff_from_filtered_linear_amplitudes` |
| What it does | Converts linear amplitudes to normalised squared weights and returns inverse Herfindahl $1/\sum p_i^2$. |
| Script | `density.py` |
| Line(s) | approx. 1808–1825 |
| Python code | `w = v * v`, `p = w / s`, `1.0 / np.sum(p * p)` (with empty-sum guards). |
| Mathematical formula | $W_i=A_i^2$, $S=\sum_i W_i$, $p_i=W_i/S$, $N_{\mathrm{eff}}=1/\sum_i p_i^2$. |
| Symbol definitions | $A_i\ge 0$: filtered linear amplitudes. |
| Notes | Used by discrete metric paths (Pass 1 extraction). |

### 4. `_apply_discrete_spectral_metrics` — discrete keys d3, d10, d17, d24

| Field | Content |
|---|---|
| Function | `_apply_discrete_spectral_metrics` |
| What it does | Dispatches discrete spectral metrics (`d3`, `d10`, `d17`, `d24`) per Pass 1 extraction. |
| Script | `density.py` |
| Line(s) | approx. 1828–2078 |
| Python code | Examples: `np.sum(np.log1p(values))`; `np.sum(np.log1p(values)) * (neff / n)`; `np.log1p(energy) * np.log1p(neff)`; `np.sum(np.log1p(masked_values))` for d24 subset rule. |
| Mathematical formula | $\mathrm{d3}=\sum_i \ln(1+A_i)$; $\mathrm{d10}=S_{\ln}\cdot N_{\mathrm{eff}}/N$; $\mathrm{d17}=\ln(1+E)\cdot\ln(1+N_{\mathrm{eff}})$; $\mathrm{d24}=\sum_j\ln(1+A^{(24)}_j)$ on the frequency/amplitude subset defined in extraction. |
| Symbol definitions | $A_i$: masked nonnegative amplitudes; $E=\sum_i A_i^2$; $N_{\mathrm{eff}}$ as in traceability entry 3 where applicable. |
| Notes | d24 uses optional `d24_global_amplitude_max`; see extraction table for mask. |

### 5. `apply_density_metric` — continuous harmonic-index weighted sum

| Field | Content |
|---|---|
| Function | `apply_density_metric` |
| What it does | Harmonic index $n_i=f_i/f_0$, rolloff $E_i=(\max(n_i,1))^{-\alpha}$ (default $\alpha=1.5$), amplitude rescaling `a_i / (E_i + 1e-10)`, then weighted sum `np.sum(w(a_i))` with optional mean normalisation. |
| Script | `density.py` |
| Line(s) | approx. 2080–2198 |
| Python code | `n_i = frequencies / fundamental_freq`; rolloff factor; `a_i / (E_i + 1e-10)`; `np.sum(w(a_i))`; optional `result / len(values)`. |
| Mathematical formula | $n_i=f_i/f_0$, $E_i=(\max(n_i,1))^{-\alpha}$, $a_i\leftarrow a_i/(E_i+10^{-10})$, $R=\sum_i w(a_i)$, optional $R/N$ if `normalize`. |
| Symbol definitions | $w$: weight from `get_weight_function`; $f_0>0$ required on continuous path. |
| Notes | Discrete keys short-circuit to `_apply_discrete_spectral_metrics` (entry 4). |

### 6. `compute_rolloff_compensated_harmonic_density` — rolloff-compensated weighted density

| Field | Content |
|---|---|
| Function | `compute_rolloff_compensated_harmonic_density` |
| What it does | Rounds harmonic orders, max-normalises amplitudes, applies rolloff divisor, applies weight $w$ (default `logarithmic`), normalises by first-partial amplitude when available. |
| Script | `density.py` |
| Line(s) | approx. 1690–1805 |
| Python code | `np.round(f / f0)`; max-normalise; `E_i = np.maximum(n_i, 1.0) ** (-alpha)`; `C_i = A_norm / (E_i + epsilon)`; `np.sum(w(C))`; `D / A_first_partial`. |
| Mathematical formula | $n_i=\mathrm{round}(f_i/f_0)$, $A^{\mathrm{norm}}_i=A_i/\max_j A_j$, $E_i=(\max(n_i,1))^{-\alpha}$, $C_i=A^{\mathrm{norm}}_i/(E_i+\varepsilon)$, $D=\sum_i w(C_i)$, $D_{\mathrm{norm}}=D/A^{(1)}$ (first raw partial with $n_i=1$) else `nan` per extraction. |
| Symbol definitions | $\alpha$, $\varepsilon$: parameters (`DEFAULT_HARMONIC_ROLLOFF_ALPHA`, epsilon in signature). |
| Notes | Weight function name defaults to `DEFAULT_ROLLOFF_COMPENSATED_DENSITY_WEIGHT_FUNCTION`. |

### Pass 2 — `density.py` (weight functions)

### 7. `WeightFunction` — static element-wise maps

| Field | Content |
|---|---|
| Function | `WeightFunction` (`linear`, `squared`, `sqrt`, `cbrt`, `cubic`, `logarithmic`, `exponential`, `inverse_log`) |
| What it does | Registry of element-wise transforms used in weighted spectral sums. |
| Script | `density.py` |
| Line(s) | approx. 1405–1437 |
| Python code | `return x`; `np.square(x)`; `np.sqrt(x)`; `np.sign(x) * (np.abs(x) ** (1.0 / 3.0))`; `x ** 3`; `np.log1p(x)`; `np.expm1(x)`; `1.0 / (np.log1p(x) + eps)`. |
| Mathematical formula | $w_{\mathrm{lin}}(x)=x$; $w_{\mathrm{sq}}(x)=x^2$; $w_{\mathrm{sqrt}}(x)=\sqrt{x}$; $w_{\mathrm{cbrt}}(x)=\operatorname{sign}(x)\,\lvert x\rvert^{1/3}$; $w_{\mathrm{cub}}(x)=x^3$; $w_{\ln}(x)=\ln(1+x)$; $w_{\mathrm{expm1}}(x)=e^x-1$; $w_{\mathrm{invlog}}(x)=1/(\ln(1+x)+\varepsilon)$. |
| Symbol definitions | $\varepsilon=10^{-10}$ in `inverse_log`. |
| Notes | Discrete keys `d3`/`d10`/`d24`→`logarithmic`, `d17`→`linear` also documented in `get_weight_function`. |

### 8. `get_weight_function` — normalised key lookup

| Field | Content |
|---|---|
| Function | `get_weight_function` |
| What it does | Normalises string key and maps aliases (`sum`→`linear`, `d2`→`linear`, `d8`→`d17`) before registry dispatch. |
| Script | `density.py` |
| Line(s) | approx. 1440–1484 |
| Python code | `key = (name or '').strip().lower()`; alias branches; return registered callable. |
| Mathematical formula | (Lookup / identity on key space; no continuous transform.) |
| Symbol definitions | — |
| Notes | Formula-bearing only insofar as it fixes which $w(\cdot)$ is used downstream. |

### 9. Rolloff defaults — named constants

| Field | Content |
|---|---|
| Function | `DEFAULT_HARMONIC_ROLLOFF_ALPHA`, `DEFAULT_ROLLOFF_COMPENSATED_DENSITY_WEIGHT_FUNCTION` |
| What it does | Default rolloff exponent $\alpha$ and default weight-function name for rolloff-compensated density. |
| Script | `density.py` |
| Line(s) | approx. 1487–1488 |
| Python code | `DEFAULT_HARMONIC_ROLLOFF_ALPHA: float = 1.5`; `DEFAULT_ROLLOFF_COMPENSATED_DENSITY_WEIGHT_FUNCTION: str = "logarithmic"` |
| Mathematical formula | $\alpha_{\mathrm{default}}=1.5$; default weight label `"logarithmic"`. |
| Symbol definitions | — |
| Notes | Policy constants; not a runtime computation. |

### Pass 3 — `density.py` (partial sums and bundles)

### 10. `band_partial_metric_sum` — band partial aggregation

| Field | Content |
|---|---|
| Function | `band_partial_metric_sum` |
| What it does | Filters nonnegative finite amplitudes; discrete branch calls `_apply_discrete_spectral_metrics`; else `np.sum(fn(v))` with `fn=get_weight_function(key)`. |
| Script | `density.py` |
| Line(s) | approx. 1900–1940 |
| Python code | `v = v[np.isfinite(v) & (v >= 0.0)]`; discrete vs `np.sum(fn(v))`. |
| Mathematical formula | Continuous: $\sum_i w(v_i)$ with $w=\texttt{get\_weight\_function}(\texttt{key})$. Discrete: extraction-table shorthand $M=\texttt{\_apply\_discrete\_spectral\_metrics}(\texttt{key},v,f)$. |
| Symbol definitions | $v_i$: nonnegative finite samples. |
| Notes | `d24` may pass `d24_global_amplitude_max`. |

### 11. `partial_metric_sums_h_i_s_total` — H / I / S / Total rules

| Field | Content |
|---|---|
| Function | `partial_metric_sums_h_i_s_total` |
| What it does | Concatenates H/I/S for global max; computes `h_sum`,`i_sum`,`s_sum` via `band_partial_metric_sum`; `t_sum = h_sum + i_sum + s_sum` except `d10`/`d17` use concatenated path per extraction. |
| Script | `density.py` |
| Line(s) | approx. 1943–2049 |
| Python code | `gmax = float(np.nanmax(all_a_raw))`; `_band_linear_total`; per-band sums; `t_sum` composition per `wf`. |
| Mathematical formula | $A_{\max}$ over concatenated raw amplitudes; $H,I,S$ band sums; $T=H+I+S$ unless `wf` in `("d10","d17")` where $T$ applies `band_partial_metric_sum` on concatenated $(a_H,a_I,a_S)$. |
| Symbol definitions | Band raw vectors per extraction. |
| Notes | See Pass 3 extraction for `d10`/`d17` concatenation rule. |

### 12. `compute_discrete_spectral_metrics_bundle` — per-key discrete floats

| Field | Content |
|---|---|
| Function | `compute_discrete_spectral_metrics_bundle` |
| What it does | Evaluates `d3`,`d10`,`d17`,`d24` via `_apply_discrete_spectral_metrics` into a dict; empty → all `nan`. |
| Script | `density.py` |
| Line(s) | approx. 2052–2080 |
| Python code | `float(_apply_discrete_spectral_metrics(...))` per key. |
| Mathematical formula | $m_{\texttt{d3}}=\mathrm{d3}(a)$, $m_{\texttt{d10}}=\mathrm{d10}(a)$, … as in traceability entry 4. |
| Symbol definitions | $a$: harmonic partial amplitudes; frequencies for `d24`. |
| Notes | Composition only; core formulas in entry 4. |

### Pass 4 — `density.py` (residual rows)

### 13. `identify_nonharmonic_residual_rows` — harmonic-window exclusion mask

| Field | Content |
|---|---|
| Function | `identify_nonharmonic_residual_rows` |
| What it does | Iterates harmonic reference frequencies; tightens/ widens match threshold; ANDs residual mask requiring $|f_k-f_h|>\tau$ for every harmonic $f_h$. |
| Script | `density.py` |
| Line(s) | approx. 2594–2692 |
| Python code | `thr_match = np.maximum(float(f0) * float(tolerance), EPSILON_FREQUENCY)` if `tolerance < 1.0` else `float(tolerance)`; `thr = float(max(thr_match, leak_hw))` when the spectral-leakage guard supplies `leak_hw`; `inharmonic_mask &= np.abs(all_freqs.astype(float) - float(f0)) > thr`. |
| Mathematical formula | $\tau_{\mathrm{match}}=\max(f_h\tau,\varepsilon_f)$ if $\tau<1$ else $\tau_{\mathrm{match}}=\tau$; with guard active, $\tau=\max(\tau_{\mathrm{match}},w_\ell)$ (Pass 4 extraction: widening when $w_\ell>0$); row $k$ residual iff $\forall h:\ |f_k-f_h|>\tau$ (AND across loop). |
| Symbol definitions | $f_h$: harmonic reference; $\tau$: `tolerance`; $\varepsilon_f$: `EPSILON_FREQUENCY`; $w_\ell$: `leakage_halfwidth_hz` from the guard path when used. |
| Notes | Returns **residual spectrum rows**, not confirmed inharmonic partials (docstring). See extraction Pass 4 for full guard/import branch logic. |

### Pass 5 — `harmonic_alignment.py`

### 14. `_linear_amp_and_energy` — linear amplitude and energy

| Field | Content |
|---|---|
| Function | `_linear_amp_and_energy` |
| What it does | Reads linear amplitude from row (dB→linear if needed), clips nonnegative, returns $(A,A^2)$. |
| Script | `harmonic_alignment.py` |
| Line(s) | approx. 37–49 |
| Python code | dB path `10**(L/20)`; `max(amp,0)`; `return amp, amp * amp`. |
| Mathematical formula | If dB: $A=10^{L/20}$; $A\leftarrow\max(A,0)$; energy $E=A^2$. |
| Symbol definitions | $L$: dB column when used. |
| Notes | Column priority per extraction Pass 5. |

### 15. `_cents` — log-frequency cents deviation

| Field | Content |
|---|---|
| Function | `_cents` |
| What it does | Twelve-tone cents map between two positive frequencies. |
| Script | `harmonic_alignment.py` |
| Line(s) | approx. 51–55 |
| Python code | `1200.0 * math.log2(obs_hz / exp_hz)` |
| Mathematical formula | $c=1200\log_2(f_{\mathrm{obs}}/f_{\mathrm{exp}})$. |
| Symbol definitions | Frequencies in Hz; `nan` if $\le 0` per code. |
| Notes | Used by harmonic-window tests. |

### 16. `_adaptive_tolerance_cents` and `_tolerance_for_order` — tolerance schedule

| Field | Content |
|---|---|
| Function | `_adaptive_tolerance_cents`, `_tolerance_for_order` |
| What it does | Maps expected bin width to cents bandwidth; returns $\max(18,2\tau_{\mathrm{bw}})$ when STFT geometry valid; else `18.0`. `_tolerance_for_order` chooses fixed override vs adaptive at $n f_0$. |
| Script | `harmonic_alignment.py` |
| Line(s) | approx. 57–88 |
| Python code | `bin_w = sample_rate / n_fft`; `bw_cents = 1200.0 * math.log2(hi / expected_hz)`; `return max(18.0, 2.0 * bw_cents)`. |
| Mathematical formula | $\Delta f=f_s/N_{\mathrm{fft}}$, $f^+=f_{\mathrm{exp}}+\Delta f/2$, $\tau_{\mathrm{bw}}=1200\log_2(f^+/f_{\mathrm{exp}})$, $\tau=\max(18,2\tau_{\mathrm{bw}})$. |
| Symbol definitions | STFT parameters as in extraction Pass 5. |
| Notes | Default `18.0` cents if geometry missing. |

### 17. `_in_any_harmonic_window` — cents-gated membership

| Field | Content |
|---|---|
| Function | `_in_any_harmonic_window` |
| What it does | Tests whether observed Hz lies within cents tolerance of any harmonic $n f_0$ up to Nyquist-related cap. |
| Script | `harmonic_alignment.py` |
| Line(s) | approx. 90–108 |
| Python code | Loop `n`, `f_exp = n * f0`, compare `abs(_cents(f_hz, f_exp)) <= tolerance_cents`. |
| Mathematical formula | Exists $n$ with $|c(f,f_{\mathrm{exp},n})|\le \tau_n$ for $f_{\mathrm{exp},n}=n f_0$, $c=\texttt{\_cents}$, $\tau_n=\texttt{\_tolerance\_for\_order}(n,\ldots)$. |
| Symbol definitions | $f$: observed frequency. |
| Notes | Stops when $n f_0$ exceeds max frequency in loop. |

### 18. `compute_harmonic_alignment_metrics` — slot counts, buckets, weighted means

| Field | Content |
|---|---|
| Function | `compute_harmonic_alignment_metrics` |
| What it does | Computes expected slot count, partitions energies into sub/reg/inh buckets with collapse/winner rules, energy-weighted mean absolute cents error, matched-order ratio. |
| Script | `harmonic_alignment.py` |
| Line(s) | approx. 110–400 (approx.; large function) |
| Python code | `n_slots = max(0, min(int(max_harmonics), int(math.floor(max_f / f0))))`; `n_round = int(round(f_hz / f0))`; energy sums; `w_mean = np.sum(w * e) / np.sum(w)`; `ratio_orders = matched_count / n_slots`. |
| Mathematical formula | $N_{\mathrm{slots}}=\min(N_{\max},\lfloor f_{\max}/f_0\rfloor)$; $n=\mathrm{round}(f/f_0)$; energy partition and $\bar e_w=\sum_j E_j e_j/\sum_j E_j$; $\rho=N_{\mathrm{match}}/N_{\mathrm{slots}}$; region ratios $E_x/(E_{\mathrm{sub}}+E_{\mathrm{reg}}+E_{\mathrm{inh}})$ per helper. |
| Symbol definitions | Energies $E$, cents errors $e$, weights $w$ as in extraction Pass 5. |
| Notes | **Line numbers should be rechecked after code changes** (long function). |

### Pass 6 — `peak_component_counts.py`

### 19. `_linear_amp_from_row` — row to linear amplitude

| Field | Content |
|---|---|
| Function | `_linear_amp_from_row` |
| What it does | Reads nonnegative linear amplitude or converts dB column to linear. |
| Script | `peak_component_counts.py` |
| Line(s) | approx. 20–30 |
| Python code | `max(0.0, float(v))` or `10.0 ** (float(v) / 20.0)`. |
| Mathematical formula | $A=\max(0,a)$ or $A=10^{L/20}$. |
| Symbol definitions | $L$: `Magnitude (dB)` when used. |
| Notes | Skips invalid rows upstream. |

### 20. `_peak_tuples` — valid $(f,A)$ list

| Field | Content |
|---|---|
| Function | `_peak_tuples` |
| What it does | Builds list of strictly positive finite frequency/amplitude pairs. |
| Script | `peak_component_counts.py` |
| Line(s) | approx. 32–45 |
| Python code | Append `(f, a)` with validity checks. |
| Mathematical formula | Sequence of $(f,A)$ with $f>0$, $A>0$ finite. |
| Symbol definitions | — |
| Notes | Feeds classifier. |

### 21. `classify_peaks_harmonic_inharmonic_subbass_from_df` — Hz tolerance and counts

| Field | Content |
|---|---|
| Function | `classify_peaks_harmonic_inharmonic_subbass_from_df` |
| What it does | Builds expected harmonic grid, cents-derived Hz tolerance, assigns best slot per peak, counts harmonic/inharmonic/subbass. |
| Script | `peak_component_counts.py` |
| Line(s) | approx. 47–200 (approx.) |
| Python code | `tol_hz = expected_freq * (2.0 ** (tolerance_cents / 1200.0) - 1.0)`; `err = abs(freq - expected_freq)`; dict updates; counts. |
| Mathematical formula | $\Delta f_n=f^{\mathrm{exp}}_n(2^{\tau/1200}-1)$, $e=|f-f^{\mathrm{exp}}_n|$, smallest-error assignment; $N_h,N_i,N_s$ counts per extraction. |
| Symbol definitions | $\tau$: `tolerance_cents`; subbass uses cutoff $f<f_{\mathrm{cut}}$. |
| Notes | **Line numbers should be rechecked after code changes.** |

### Pass 7 — `low_frequency_policy.py`

### 22. `calculate_subfundamental_margin_percent` — piecewise margin

| Field | Content |
|---|---|
| Function | `calculate_subfundamental_margin_percent` |
| What it does | Returns register margin percent vs $f_0$ bands. |
| Script | `low_frequency_policy.py` |
| Line(s) | approx. 33–72 |
| Python code | Piecewise thresholds on `f0_hz` returning `35`,`25`,`15`,`10`. |
| Mathematical formula | $m\in\{35,25,15,10\}\%$ by $f_0<60$, $<120$, $<300$, else; invalid $f_0\Rightarrow 10$. |
| Symbol definitions | $f_0$: Hz. |
| Notes | Policy table. |

### 23. `calculate_adaptive_subfundamental_cutoff_hz` — adaptive cutoff

| Field | Content |
|---|---|
| Function | `calculate_adaptive_subfundamental_cutoff_hz` |
| What it does | Combines percent cutoff, floor, leakage candidate, caps by fraction of $f_0$, reports effective margin. |
| Script | `low_frequency_policy.py` |
| Line(s) | approx. 74–152 |
| Python code | `percentage_cut = f0 * (1.0 - margin / 100.0)`; `raw_max` over finite candidates; `adaptive = min(raw_max, cap_hz)`; `eff_margin = 100.0 * (1.0 - adaptive / f0)`. |
| Mathematical formula | $f_\%=f_0(1-m/100)$; $F_{\mathrm{raw}}=\max(\ldots)$ over finite candidates; $f_{\mathrm{ad}}=\min(F_{\mathrm{raw}},\gamma f_0)$; $m_{\mathrm{eff}}=100(1-f_{\mathrm{ad}}/f_0)$. |
| Symbol definitions | $\gamma=$ `max_fraction_of_f0`; optional leakage cutoff. |
| Notes | See extraction for candidate list. |

### 24. `classify_low_frequency_row` — categorical buckets

| Field | Content |
|---|---|
| Function | `classify_low_frequency_row` |
| What it does | Labels row by DC floor, adaptive subfundamental cutoff, physical low band, else not residual. |
| Script | `low_frequency_policy.py` |
| Line(s) | approx. 154–250 (approx.) |
| Python code | Comparisons `f <= dc_floor_hz`, `f < ad`, `f <= physical_low_band_upper_hz`, else. |
| Mathematical formula | Piecewise region tests on $f$ vs thresholds (extraction Pass 7). |
| Symbol definitions | `ad`: adaptive cutoff from entry 23. |
| Notes | Label-only policy logic. |

### Pass 8 — `spectral_leakage_guards.py`

### 25. `leakage_halfwidth_hz` — main-lobe half-width in Hz

| Field | Content |
|---|---|
| Function | `leakage_halfwidth_hz` |
| What it does | Computes bin width then half-width in Hz from main-lobe bins. |
| Script | `spectral_leakage_guards.py` |
| Line(s) | approx. 22–50 |
| Python code | `bw = sr / n_fft`; `return 0.5 * ml * bw` |
| Mathematical formula | $\Delta f=f_s/N_{\mathrm{fft}}$; $w_\ell=\tfrac12 B\Delta f$ with $B$ main-lobe bins. |
| Symbol definitions | Defaults per module constants when not passed. |
| Notes | `bin_width_hz` can override $\Delta f$ when provided. |

### 26. `filter_inharmonic_peak_candidates` — harmonic proximity veto

| Field | Content |
|---|---|
| Function | `filter_inharmonic_peak_candidates` |
| What it does | Drops peak candidates within leakage half-width of any harmonic representative frequency. |
| Script | `spectral_leakage_guards.py` |
| Line(s) | approx. 52–120 (approx.) |
| Python code | `np.any(np.abs(hf - ff) <= lh)` style test in loop/logic per extraction. |
| Mathematical formula | Remove $(f,a)$ if $\exists h:\ |h-f|\le w_\ell$. |
| Symbol definitions | $h\in$ harmonic rep set; $w_\ell$ from entry 25. |
| Notes | Pairwise geometric test; see extraction. |

### Pass 9 — `compile_metrics.py` (compile-time normalisation)

### 27. `_add_canonical_and_global_density_columns` — canonical fallback and global $[0,1]$ norm

| Field | Content |
|---|---|
| Function | `_add_canonical_and_global_density_columns` |
| What it does | Fills canonical density from `Density Metric`/10 when missing; global max-normalises finite canonical values with clip. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 1058–1110 |
| Python code | `out[canon_col] = dm / 10.0`; `mx = np.nanmax(finite)`; `(s_canon / mx).clip(0,1)`; `density_per_component = s_canon / hoc.replace(0, nan)`. |
| Mathematical formula | $d_{\mathrm{canon}}=d_{\mathrm{DM}}/10$ fallback; $\hat d_k=\mathrm{clip}(d_k/M,0,1)$ with $M=\max_k d_{\mathrm{canon},k}$; $d^{(\mathrm{pc})}_k=d_k/N_k$ with harmonic order count $N_k$. |
| Symbol definitions | $d_{\mathrm{DM}}$: `Density Metric` column; $N_k$: `harmonic_order_count`. |
| Notes | Division-by-zero masked on per-component path. |

### 28. `_compute_weighted_density_columns_for_wide_df` — weighted raw and run-relative norm

| Field | Content |
|---|---|
| Function | `_compute_weighted_density_columns_for_wide_df` |
| What it does | Forms weighted harmonic/inharmonic/subbass contributions, sums to raw metric, max-normalises positive finite raw to $[0,\infty)$ scale per extraction. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 1112–1200 (approx.) |
| Python code | `wh = (D_H * w_H)` etc.; `raw = wh.fillna(0) + ...`; `mx = np.max(finite_pos)`; `raw / mx`. |
| Mathematical formula | $c_{H,k}=D_{H,k}w_{H,k}$ (and $I,S$); $r_k$ sum with NaN-as-zero rules per extraction; $\tilde r_k=r_k/M_r$, $M_r=\max_{k:\ r_k>0,\ \mathrm{finite}} r_k$. |
| Symbol definitions | $D_H,w_H$: harmonic partial sum and component ratio columns. |
| Notes | **Line numbers should be rechecked after code changes.** |

### Pass 10 — `proc_audio.py` (selected sites)

### 29. `_normalize_level` — RMS-based gain

| Field | Content |
|---|---|
| Function | `_normalize_level` |
| What it does | RMS in time domain, convert to dB, apply scalar gain for target RMS dB. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1600–1648 |
| Python code | `rms = sqrt(mean(square(y)) + 1e-12)`; `cur_db = 20 * log10(rms)`; `gain = 10 ** ((target_rms_db - cur_db) / 20)`; `y * gain`. |
| Mathematical formula | $\mathrm{RMS}=\sqrt{\overline{y^2}+\varepsilon}$, $L_{\mathrm{RMS}}=20\log_{10}(\mathrm{RMS})$, $G=10^{(L_{\mathrm{tgt}}-L_{\mathrm{RMS}})/20}$, $y'=Gy$. |
| Symbol definitions | $\varepsilon=10^{-12}$. |
| Notes | Before STFT in pipeline. |

### 30. STFT magnitude, dB, frequency grid — `librosa` delegation

| Field | Content |
|---|---|
| Function / context | STFT pipeline (`librosa.stft`, `np.abs`, `librosa.amplitude_to_db`, `librosa.fft_frequencies`) |
| What it does | Builds complex STFT, magnitude, dB display, frequency bin centres via library conventions. |
| Script | `proc_audio.py` |
| Line(s) | approx. varies by class (search `librosa.stft` / `amplitude_to_db` / `fft_frequencies`) |
| Python code | `S_mag = np.abs(self.S)`; `librosa.amplitude_to_db(S_mag, ref=1.0)`; `librosa.fft_frequencies(sr=..., n_fft=...)`. |
| Mathematical formula | $M_{k,t}=\lvert S_{k,t}\rvert$; $L_{k,t}=20\log_{10}(M_{k,t})$ with `ref=1`; $f_k=k f_s/N_{\mathrm{fft}}$ per `librosa` bin convention. |
| Symbol definitions | Black-box details inside `librosa` / `numpy`. |
| Notes | **Third-party internals not proven here**—only call-level correspondence. |

### 31. `_coherent_gain` and `_window_sum` — window sums

| Field | Content |
|---|---|
| Function | `_coherent_gain`, `_window_sum` |
| What it does | Window sample mean vs sum used in calibration bookkeeping. |
| Script | `proc_audio.py` |
| Line(s) | approx. 407–477 |
| Python code | `np.sum(w) / float(n_fft)`; `np.sum(w)`. |
| Mathematical formula | $G=\frac{1}{N}\sum_n w[n]$; $S_w=\sum_n w[n]$. |
| Symbol definitions | $N=N_{\mathrm{fft}}$; $w$: window samples. |
| Notes | Used with `physical_peak_amplitude`. |

### 32. `physical_peak_amplitude` — calibrated peak amplitude

| Field | Content |
|---|---|
| Function | `physical_peak_amplitude` |
| What it does | Scales STFT magnitude by window sum and one-sided/two-sided factor. |
| Script | `proc_audio.py` |
| Line(s) | approx. 479–520 (approx.) |
| Python code | `factor * mag / sw` |
| Mathematical formula | $A_{\mathrm{peak}}=\gamma M/S_w$ with $\gamma\in\{1,2\}$ per one-sided flag in extraction. |
| Symbol definitions | $M$: STFT magnitude sample; $S_w$: window sum. |
| Notes | See extraction Pass 10 for branch details. |

### 33. `_verify_energy_conservation` — Parseval-style ratio

| Field | Content |
|---|---|
| Function | `_verify_energy_conservation` |
| What it does | Compares time-domain energy to frequency-domain energy with window/overlap correction per implementation. |
| Script | `proc_audio.py` |
| Line(s) | approx. 768–970 (approx.) |
| Python code | `energy_time = sum(abs(y)**2)`; `energy_freq = sum(abs(S)**2)`; `window_power = sum(w**2)`; `overlap_factor = window_length / hop_length`; DC/Nyquist handling then `energy_ratio = energy_freq_norm / energy_time`. |
| Mathematical formula | $E_t=\sum_n \lvert y[n]\rvert^2$; $E_f=\sum_{k,t}\lvert S_{k,t}\rvert^2$; composite $E_{\mathrm{fn}}$ per code’s one-sided normalisation; $R=E_{\mathrm{fn}}/E_t$. |
| Symbol definitions | Window power $P_w=\sum w^2$; overlap factor $O=N/h$. |
| Notes | Full branch algebra in extraction Pass 10; **line range approximate**. |

### 34. `AudioProcessor._calculate_edge_frame_weights` — edge frame correction

| Field | Content |
|---|---|
| Function | `AudioProcessor._calculate_edge_frame_weights` |
| What it does | Infers partial real-signal occupancy of first/last frames and applies bounded correction factor. |
| Script | `proc_audio.py` |
| Line(s) | approx. 2701–2800 (approx.) |
| Python code | `correction = 1 / max(portion, 0.5)` capped per extraction. |
| Mathematical formula | $c=\min(2,\,1/\max(p,0.5))$ with $p$ inferred portion. |
| Symbol definitions | $p$: real-signal fraction of frame length. |
| Notes | Heuristic edge handling. |

### 35. `_estimate_f0_global_robust` — weighted harmonic LS on detected peaks

| Field | Content |
|---|---|
| Function | `_estimate_f0_global_robust` |
| What it does | Assigns integer harmonic indices, quadratic weights from amplitudes, closed-form $f_0$ and residual RMS. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1448–1515 |
| Python code | `n_assignments = round(detected_freqs / initial_f0)`; `weights = (A/A_max)**2`; weighted ratio for `f0_robust`; `residuals = detected_freqs - n_assignments * f0_robust`; `sqrt(weighted_sse / weight_sum)`. |
| Mathematical formula | $n_i=\mathrm{round}(f_i/f_0^{(\mathrm{init})})$ (clipped); $w_i\propto A_i^2$; $f_0=\sum_i w_i n_i f_i/\sum_i w_i n_i^2$; $\varepsilon_i=f_i-n_i f_0$; $\sigma=\sqrt{\sum_i w_i\varepsilon_i^2/\sum_i w_i}$. |
| Symbol definitions | Detected peaks $(f_i,A_i)$. |
| Notes | Clipping of $n_i$ per code. |

### 36. `AudioProcessor._calculate_metrics` and `_set_model_weights_from_current_component_energy` — energy ratios

| Field | Content |
|---|---|
| Function | `AudioProcessor._calculate_metrics`, `AudioProcessor._set_model_weights_from_current_component_energy` |
| What it does | Sums squared harmonic/inharmonic/subbass amplitudes; forms $E_H/(E_H+E_I+E_S)$ and related ratios. |
| Script | `proc_audio.py` |
| Line(s) | approx. 4992–5050 and 5316–5400 (approx.) |
| Python code | `h_energy = sum(square(harmonic_amps))`; `tot_energy = h_energy + ih_energy + sub_energy`; ratios; `comp_h = Hn / T`, `model_h = Hn / HI`. |
| Mathematical formula | $E_H=\sum A_{H,i}^2$, $E_{\mathrm{tot}}=E_H+E_I+E_S$, $r_H=E_H/E_{\mathrm{tot}}$; $w_H=H/(H+I+S)$; $m_H=H/(H+I)$. |
| Symbol definitions | Linear amplitudes in each band. |
| Notes | Guards on tiny `tot_energy` in code. |

### 37. Export path — raw power and summed amplitudes

| Field | Content |
|---|---|
| Function / context | `_attach_raw_and_display` (nested) and integrated amplitude sums |
| What it does | Stores `Power_raw = amps_raw ** 2` and summed linear totals per extraction Pass 10. |
| Script | `proc_audio.py` |
| Line(s) | approx. 8969+ (approx.; search `Power_raw` / `linear_sum_amplitude`) |
| Python code | `Power_raw = amps_raw ** 2`; assignments to `self.linear_sum_amplitude_*` per extraction. |
| Mathematical formula | $P=A^2$; $\Sigma_H A$, $\Sigma_I A$, $\Sigma_S A$ from column totals as implemented. |
| Symbol definitions | Per-row amplitudes in export frame. |
| Notes | **Line numbers should be rechecked after code changes** (large file). |

### Pass 11 — `density.py` (extended metrics; condensed)

The Pass 11 extraction table lists **many** functions; each row’s mathematics is preserved there. Below: **one entry per named function** with the **primary** closed-form or algorithmic contract. Sub-branches marked “partial / ambiguous” in `FORMULA_EXTRACTION_TABLE_PASS_11_DENSITY_EXTENDED_METRICS.md` are not expanded beyond that document.

### 38. `estimate_noise_floor` — PSD percentile floor

| Field | Content |
|---|---|
| Function | `estimate_noise_floor` |
| What it does | Positive PSD flatten → `numpy` percentile. |
| Script | `density.py` |
| Line(s) | approx. 280–480 (approx.) |
| Python code | `np.percentile(psd_positive, percentile)` |
| Mathematical formula | $F=\mathrm{Percentile}_{p\%}(\mathrm{PSD}^+)$. |
| Symbol definitions | $p=$ `percentile`. |
| Notes | `numpy.percentile` treated as black box for interpolation details. |

### 39. `physical_spectral_density` — participation score

| Field | Content |
|---|---|
| Function | `physical_spectral_density` (via `SpectralDensityMetrics`) |
| What it does | Power from amplitudes, $N_{\mathrm{eff}}$, score clipped to $[0,1]$ per extraction. |
| Script | `density.py` |
| Line(s) | approx. 483–510 |
| Python code | See `SpectralDensityMetrics.physical_spectral_density` body. |
| Mathematical formula | $P_i=A_i^2$, $N_{\mathrm{eff}}=(\sum P_i)^2/\sum P_i^2$, score $=(N_{\mathrm{eff}}/N)$ clipped per extraction. |
| Symbol definitions | $A_i>0$ finite. |
| Notes | `bin_width_hz` unused in formula (extraction note). |

### 40. `perceptual_spectral_density` — Bark map and blend

| Field | Content |
|---|---|
| Function | `perceptual_spectral_density` |
| What it does | Bark $B(f)$, triangular band weights, entropy-based uniformity, linear blend $0.6O+0.4U$. |
| Script | `density.py` |
| Line(s) | approx. 501–545 |
| Python code | `bark = 13*arctan(0.00076*f)+3.5*arctan((f/7500)**2)`; band occupancy/uniformity; `0.6 * occupancy + 0.4 * uniformity`. |
| Mathematical formula | $B(f)=13\arctan(0.00076f)+3.5\arctan((f/7500)^2)$; band energies $E_b$; $D=0.6O+0.4U$ clipped (extraction Pass 11). |
| Symbol definitions | $f\ge 1$ Hz clamp in code path per extraction. |
| Notes | Full entropy details in extraction table. |

### 41. `calculate_harmonic_density` — threshold count density (+ optional tanh blend)

| Field | Content |
|---|---|
| Function | `calculate_harmonic_density` |
| What it does | Caps expected harmonics, dB levels, significant partial fraction, optional amplitude blend. |
| Script | `density.py` |
| Line(s) | approx. 513–547 |
| Python code | `max_expected_harmonics`; `20*log10(max(amps,1e-12))`; `significant.sum() / max_expected_harmonics`; optional `tanh(mean)` blend. |
| Mathematical formula | Extraction Pass 11: $N_{\max}$ cap; $L_i=20\log_{10}\max(A_i,\varepsilon)$; $\rho=\frac{1}{N_{\max}}\sum_i \mathbf{1}[L_i>L_{\mathrm{thr}}]$; optional $\tanh$ blend with `amp_weight`. |
| Symbol definitions | $L_{\mathrm{thr}}=$ `threshold_db`. |
| Notes | — |

### 42. `calculate_inharmonic_density` — delegate

| Field | Content |
|---|---|
| Function | `calculate_inharmonic_density` |
| What it does | Delegates to `calculate_harmonic_density` with partial-count parameter. |
| Script | `density.py` |
| Line(s) | approx. 548–560 |
| Python code | `return calculate_harmonic_density(..., max_expected_harmonics=max_expected_partials)` |
| Mathematical formula | Same as entry 41 with different cap parameter. |
| Symbol definitions | — |
| Notes | **No separate formula**—trace entry 41. |

### 43. `calculate_perceptual_spectral_density` — band energies and final blend (partial)

| Field | Content |
|---|---|
| Function | `calculate_perceptual_spectral_density` |
| What it does | dB mapping, Bark bin accumulation, masking loop, final weighted squash `1-exp(-k D_fin)` per extraction. |
| Script | `density.py` |
| Line(s) | approx. 986–1140 (approx.) |
| Python code | Multiple branches: `max_possible_harmonics`; band loop; `_critical_band_masking`; constants `PERCEPTUAL_DENSITY_*`. |
| Mathematical formula | See extraction Pass 11 (long piecewise/heuristic sections). |
| Symbol definitions | Import constants from `constants`. |
| Notes | **Partial / ambiguous** segments per extraction “human review” flags. |

### 44. `_critical_band_masking` — dB masking threshold

| Field | Content |
|---|---|
| Function | `_critical_band_masking` |
| What it does | Piecewise slope on Bark distance; absolute floor. |
| Script | `density.py` |
| Line(s) | approx. 611–658 |
| Python code | Piecewise on `bark_distance`; `max(threshold_db, MASKING_ABSOLUTE_THRESHOLD_DB)`. |
| Mathematical formula | $T=L_{\mathrm{masker}}+\Delta(L_{\mathrm{masker}},\Delta b)$ with constants; $T\leftarrow\max(T,L_{\min})$. |
| Symbol definitions | $\Delta b=\lvert b(f_p)-b(f_m)\rvert$. |
| Notes | Parncutt-style model per docstring. |

### 45. `estimate_noise_floor_by_critical_bands` — per-band floors (partial)

| Field | Content |
|---|---|
| Function | `estimate_noise_floor_by_critical_bands` |
| What it does | Margin dB from multiplier; per-band percentile floors; boundary interpolation per extraction. |
| Script | `density.py` |
| Line(s) | approx. 660–784 |
| Python code | `margin_db = 20*log10(mult)`; per-band `np.percentile`; interpolation `weights` mix. |
| Mathematical formula | Extraction Pass 11: $\tau_b=\max(P_{p\%}(L_b)+M_{\mathrm{dB}},L_{\min})$; boundary blend. |
| Symbol definitions | Bands in fixed Hz ranges. |
| Notes | Boundary weights flagged ambiguous in extraction. |

### 46. `apply_spectral_masking_filter` — iterative audibility (algorithmic)

| Field | Content |
|---|---|
| Function | `apply_spectral_masking_filter` |
| What it does | Double loop: probe audible iff above `_critical_band_masking` threshold vs all stronger maskers (extraction). |
| Script | `density.py` |
| Line(s) | approx. 785–985 (approx.) |
| Python code | Sort by dB; nested comparisons calling `_critical_band_masking`. |
| Mathematical formula | Predicate $\forall$ stronger maskers: $L_{\mathrm{probe}}\ge T(f_m,L_m,f_p)$ per extraction. |
| Symbol definitions | dB levels $L$. |
| Notes | Algorithmic, not single closed form. |

### 47. `calculate_spectral_complexity` — irregularity, inharmonicity, entropy blend

| Field | Content |
|---|---|
| Function | `calculate_spectral_complexity` |
| What it does | Moving-average irregularity; harmonic tube energy; inharmonicity; normalised Shannon; linear blend. |
| Script | `density.py` |
| Line(s) | approx. 1227–1303 |
| Python code | Moving average ratio; harmonic tube sums; `inharmonicity = 1 - harmonic_energy/total_energy`; entropy `/log2(len(probs))`; `0.4*irregularity + 0.4*inharmonicity + 0.2*entropy`. |
| Mathematical formula | Extraction Pass 11: $\mathrm{irr}$, $E_H$, $I=1-E_H/E$, normalised $H$, clipped blend. |
| Symbol definitions | Linear `Amplitude` or dB→linear per code. |
| Notes | — |

### 48. `calculate_harmonic_richness` — count factor and geometric mean blend

| Field | Content |
|---|---|
| Function | `calculate_harmonic_richness` |
| What it does | $C=\min(1,N/N_{\max})$; geometric mean of positive amps with `tanh`; blend with `amplitude_weight`. |
| Script | `density.py` |
| Line(s) | approx. 1305–1403 |
| Python code | Extraction-aligned: `exp(mean(log(A_i))))`, `tanh`, weighted sum. |
| Mathematical formula | $C=\min(1,N/N_{\max})$; $G=\exp(\mean(\ln A_i))$; $R=(1-w)C+w\tanh(G)$. |
| Symbol definitions | $w=$ `amplitude_weight`. |
| Notes | — |

### 49. `_calculate_harmonic_completeness_phase2` — tolerance ramp and gap penalty

| Field | Content |
|---|---|
| Function | `_calculate_harmonic_completeness_phase2` |
| What it does | Order-dependent tolerance $\tau_n=\tau_0(1+cn)$; completeness from weighted gap penalty (extraction). |
| Script | `density.py` |
| Line(s) | approx. 1141–1225 |
| Python code | `tolerance = BASE*(1+ADAPTIVE*n)`; gap weights $\propto 1/n$. |
| Mathematical formula | Extraction Pass 11. |
| Symbol definitions | Constants `HARMONIC_TOLERANCE_*`. |
| Notes | — |

### 50. `compute_harmonic_effective_power_density` — max-normalised power sum

| Field | Content |
|---|---|
| Function | `compute_harmonic_effective_power_density` |
| What it does | Normalises powers by max, sums, optional divide by $N$. |
| Script | `density.py` |
| Line(s) | approx. 1504–1621 |
| Python code | `p_norm = pwr / max_p`; `dens = sum(p_norm)`. |
| Mathematical formula | $\tilde P_i=A_i^2/\max_j A_j^2$, $D=\sum_i \tilde P_i$ (and $D/N$ variant per extraction). |
| Symbol definitions | Amplitudes $A_i$. |
| Notes | — |

### 51. `compute_harmonic_effective_power_mass` — total and mean-square RMS

| Field | Content |
|---|---|
| Function | `compute_harmonic_effective_power_mass` |
| What it does | Sums squares of amplitudes; mean power; RMS. |
| Script | `density.py` |
| Line(s) | approx. 1623–1688 |
| Python code | `power = square(amplitudes)` sums/means; `sqrt` of mean. |
| Mathematical formula | $E=\sum A_i^2$, $\bar P=\mean(A_i^2)$, $\mathrm{RMS}=\sqrt{\bar P}$. |
| Symbol definitions | — |
| Notes | — |

### 52. `compute_subbass_protection_tolerance_hz` — bin-based protection width

| Field | Content |
|---|---|
| Function | `compute_subbass_protection_tolerance_hz` |
| What it does | Lower bound on tolerance from bins × sample rate / FFT size. |
| Script | `density.py` |
| Line(s) | approx. 2326–2354 |
| Python code | `max(minimum_hz, bin_multiplier * sr/n_fft)` |
| Mathematical formula | $\tau=\max(\tau_{\min},\,k f_s/N_{\mathrm{fft}})$. |
| Symbol definitions | Defaults per extraction (`minimum_hz`, `bin_multiplier`). |
| Notes | — |

### 53. `aggregate_low_frequency_residual_peak_power` — band power with harmonic mask

| Field | Content |
|---|---|
| Function | `aggregate_low_frequency_residual_peak_power` |
| What it does | dB→linear; harmonic proximity mask; sums power in band (sum-all vs strict-local-max modes). |
| Script | `density.py` |
| Line(s) | approx. 2356–2467 |
| Python code | `10**(L/20)`; `_harmonic_mask`; sums of $A^2$ per mode. |
| Mathematical formula | Extraction Pass 11: exclude if $\min_h |f_i-h|\le\tau$; band integrals $\sum A_i^2$. |
| Symbol definitions | Band $(f_{\mathrm{lo}},f_{\mathrm{hi}}]$. |
| Notes | Two aggregation modes. |

### 54. `partial_density_effective_components_bundle` — relative power threshold and $D_{\mathrm{eff}}$

| Field | Content |
|---|---|
| Function | `partial_density_effective_components_bundle` |
| What it does | Threshold on powers; `_inverse_herfindahl_effective_components` on merged list. |
| Script | `density.py` |
| Line(s) | approx. 2469–2572 |
| Python code | `thresh = ref * (10**(min_db_relative/10.0))`; `_inverse_herfindahl_effective_components`. |
| Mathematical formula | $T=R\cdot 10^{r/10}$; $D_{\mathrm{eff}}=(\sum p_k)^2/\sum p_k^2$ on merged powers (extraction). |
| Symbol definitions | $R$: reference max power across H/I/S per extraction. |
| Notes | Same $D_{\mathrm{eff}}$ structure as IPR (entry 2). |

### 55. `partial_density_effective_components` — scalar wrapper

| Field | Content |
|---|---|
| Function | `partial_density_effective_components` |
| What it does | Calls bundle; returns scalar only. |
| Script | `density.py` |
| Line(s) | approx. 2574–2592 |
| Python code | Delegates to `partial_density_effective_components_bundle`. |
| Mathematical formula | Same as entry 54 (scalar return). |
| Symbol definitions | — |
| Notes | No new math. |

### 56. `calculate_combined_density_metric` — log vs linear blend

| Field | Content |
|---|---|
| Function | `calculate_combined_density_metric` |
| What it does | Weight renormalisation; `expm1` of weighted `log1p` branch or linear $\alpha h+\beta i$. |
| Script | `density.py` |
| Line(s) | approx. 2706–2904 |
| Python code | Weight renorm; `expm1(alpha * log1p(...) + beta * log1p(...))` or linear. |
| Mathematical formula | Extraction Pass 11: $\alpha\leftarrow \alpha/(\alpha+\beta)$ if needed; $\mathrm{expm1}(\alpha\ln(1+h_+)+\beta\ln(1+i_+))$ or $\alpha h+\beta i$. |
| Symbol definitions | $h,i$: harmonic/inharmonic inputs. |
| Notes | — |

### 57. `_hz_to_bark` — Bark mapping

| Field | Content |
|---|---|
| Function | `_hz_to_bark` |
| What it does | Same closed form as perceptual Bark helper. |
| Script | `density.py` |
| Line(s) | approx. 2906–2908 |
| Python code | `13*arctan(0.00076*f)+3.5*arctan((f/7500)**2)` |
| Mathematical formula | $B(f)=13\arctan(0.00076f)+3.5\arctan((f/7500)^2)$. |
| Symbol definitions | $f$ in Hz (array-safe). |
| Notes | Matches entry 40’s Bark term. |

### 58. `spectral_density` — Hill/Rényi, Gaussian proximity, low-band weight (partial)

| Field | Content |
|---|---|
| Function | `spectral_density` |
| What it does | Power-mass normalisation $p_i\propto A_i^\gamma$; optional window; Hill/Rényi $N_{\mathrm{eff}}$; Gaussian proximity $P_{\mathrm{num}}$, $P_{\mathrm{norm}}$; low-band weight blend; large-$M$ approximate path per extraction. |
| Script | `density.py` |
| Line(s) | approx. 2909–3250 (approx.) |
| Python code | See long function body in `density.py`. |
| Mathematical formula | Extraction Pass 11 summary equations for mass, Hill/Rényi, Gaussian proximity, $P_{\mathrm{norm}}$, $D_{\mathrm{peso}}$; harmonic-bin block partial. |
| Symbol definitions | $\gamma$ default 2; $\sigma$ from `sigma_hz`; many constants. |
| Notes | **Partial / ambiguous** per extraction; thesis should cite extraction table for full branch list. |

### Pass 12 — `dissonance_models.py` (+ `proc_audio.py` call sites)

### 59. `DissonanceModel.total_dissonance` — cross-partial pairwise sum

| Field | Content |
|---|---|
| Function | `DissonanceModel.total_dissonance` |
| What it does | Double loop adds `pure_tones_dissonance` over all cross pairs. |
| Script | `dissonance_models.py` |
| Line(s) | approx. 44–58 |
| Python code | Nested `for` with `total_diss += self.pure_tones_dissonance(f1, f2, a1, a2)`. |
| Mathematical formula | $D=\sum_{(f_1,a_1)\in\mathcal P_1}\sum_{(f_2,a_2)\in\mathcal P_2} d(f_1,f_2,a_1,a_2)$. |
| Symbol definitions | $d$: model-specific pairwise kernel. |
| Notes | Abstract `pure_tones_dissonance` implemented in subclasses. |

### 60. `DissonanceModel.same_timbre_dissonance` and `calculate_dissonance_curve`

| Field | Content |
|---|---|
| Function | `same_timbre_dissonance`, `calculate_dissonance_curve` |
| What it does | Default shift $(rf,a)$ then `total_dissonance`; curve samples interval ratios. |
| Script | `dissonance_models.py` |
| Line(s) | approx. 60–82 |
| Python code | `shifted_partials = [(f * interval, a) for ...]`; `np.linspace(min_interval, max_interval, num_points)` loop. |
| Mathematical formula | $\tilde{\mathcal P}=\{(rf,a)\}$; $D(r)=\texttt{total\_dissonance}(\mathcal P,\tilde{\mathcal P})$; curve samples $r_k\in[r_{\min},r_{\max}]$. |
| Symbol definitions | $r>0$: interval ratio. |
| Notes | Sethares overrides in subclass. |

### 61. `DissonanceModel.find_local_minima` — discrete local minima rule

| Field | Content |
|---|---|
| Function | `find_local_minima` |
| What it does | Sorted-interval scan with asymmetric left-neighbour `- sensitivity` test. |
| Script | `dissonance_models.py` |
| Line(s) | approx. 84–96 |
| Python code | `val < curve[intervals[i-1]] - sensitivity` among neighbour inequalities. |
| Mathematical formula | Extraction Pass 12 (non-symmetric local-min test). |
| Symbol definitions | $\varepsilon=$ `sensitivity` (default `0.01`). |
| Notes | Extraction flags possible typo vs strict local minimum. |

### 62. `DissonanceModel._dissonance_total_and_pairs` and `_dissonance_total_pairs_and_minamp`

| Field | Content |
|---|---|
| Function | `_dissonance_total_and_pairs`, `_dissonance_total_pairs_and_minamp` |
| What it does | dB→linear; pairwise upper-triangle sums; optional amplitude compensation; `S_min` sum of mins. |
| Script | `dissonance_models.py` |
| Line(s) | approx. 144–312 |
| Python code | `10 ** (Magnitude (dB) / 20)`; nested `i<j` loops; optional `amps * (2.0 / N)`. |
| Mathematical formula | Pairwise $d_{ij}$; $S_{\min}=\sum_{i<j}\min(a_i,a_j)$; modes in `calculate_dissonance_metric` (sum / mean / scaled / minamp_norm) per extraction. |
| Symbol definitions | `N`: `win_length` or `n_fft` per model attrs for compensation hook. |
| Notes | See extraction for mode formulas. |

### 63. `SetharesDissonance` — `_s`, `pure_tones_dissonance`, `same_timbre_dissonance` branches

| Field | Content |
|---|---|
| Function | `SetharesDissonance._s`, `pure_tones_dissonance`, `_pairwise_sum`, `same_timbre_dissonance` |
| What it does | Critical-band-style scale $s(f)$; pairwise Plomp–Levelt / Sethares difference of exponentials on $y=s(f_{\min})(f_{\max}-f_{\min})$; union-spectrum and subtract-intrinsic branches. |
| Script | `dissonance_models.py` |
| Line(s) | approx. 313–474 |
| Python code | `x_star / (s1 * f1 + s2)`; `d = min(a1,a2)*gain*(exp(-b1*y)-exp(-b2*y))` with nonpositivity clip; `_pairwise_sum`; `full`, `cross`, `subtract_intrinsic` branches per extraction. |
| Mathematical formula | Extraction Pass 12: $s(f)=x^\star/(s_1 f+s_2)$; pairwise $d$ with $\min$ amplitude; self sums $D_{\mathcal P},D_{r\mathcal P}$; $D=\max(0,D_{\mathrm{full}}-D_{\mathcal P}-D_{r\mathcal P})$ when enabled. |
| Symbol definitions | Defaults $b_1,b_2,x^\star,s_1,s_2,g$ per class. |
| Notes | **Model-dependent** constants. |

### 64. `HutchinsonKnopoffDissonance` — CBW, tabular $g$, normalised product sum

| Field | Content |
|---|---|
| Function | `cbw`, `g`, `pure_tones_dissonance`, `total_dissonance` |
| What it does | $\mathrm{CBW}(\bar f)=1.72 \bar f^{0.65}$; linear interpolation table `np.interp`; pairwise $d=a_1a_2 g(y)/N$ with $y=\Delta f/\mathrm{CBW}$; global ratio sum. |
| Script | `dissonance_models.py` |
| Line(s) | approx. 475–590 |
| Python code | See class methods in `dissonance_models.py`. |
| Mathematical formula | Extraction Pass 12 equations (1)-style normalisation and double-sum ratio form. |
| Symbol definitions | $\bar f=\tfrac12(f_i+f_j)$. |
| Notes | Tabular $g$ is **model data**, not derived here. |

### 65. `VassilakisDissonance.pure_tones_dissonance` — Vassilakis–Sethares variant

| Field | Content |
|---|---|
| Function | `VassilakisDissonance.pure_tones_dissonance` |
| What it does | Amplitude factor AF, spectral difference of exponentials, stated constants. |
| Script | `dissonance_models.py` |
| Line(s) | approx. 592–654 |
| Python code | `af_degree = 2*A2/(A1+A2)`; `spectral = exp(-b1*x)-exp(-b2*x)`; `R = (A1*A2)**spl_exp * pair_factor * af_degree**af_exp * spectral`. |
| Mathematical formula | Extraction Pass 12 closed form with $\mathrm{AF}=2A_2/(A_1+A_2)$, $x=s(f_{\min})(f_{\max}-f_{\min})$, constants `pair_factor`, `af_exp`, `spl_exp`. |
| Symbol definitions | $A_1=\max(a_i,a_j)$, $A_2=\min(a_i,a_j)$ after ordering. |
| Notes | **Model-dependent**. |

### 66. `calculate_all_dissonance_metrics` and `compare_dissonance_models`

| Field | Content |
|---|---|
| Function | `calculate_all_dissonance_metrics`, `compare_dissonance_models` |
| What it does | Loops registered models; compares curves with per-curve min–max normalisation for plotting. |
| Script | `dissonance_models.py` |
| Line(s) | approx. 656–720 |
| Python code | `m_k = model.calculate_dissonance_metric(df)`; `(v - v_min) / (v_max - v_min)`. |
| Mathematical formula | $m_k$ per model; plotting normalisation $\tilde D=(D-D_{\min})/(D_{\max}-D_{\min})$ per curve. |
| Symbol definitions | — |
| Notes | `compare_dissonance_models` is visualisation-oriented. |

### 67. `AudioProcessor.calculate_dissonance_metrics` — cap, pair count, curve wiring

| Field | Content |
|---|---|
| Function | `AudioProcessor.calculate_dissonance_metrics` |
| What it does | `nlargest` partial cap; pair count; calls `calculate_dissonance_curve` and `find_local_minima`. |
| Script | `proc_audio.py` |
| Line(s) | approx. 7371–7500 (approx.) |
| Python code | `nlargest(_cap, "Amplitude")`; `n_after * (n_after - 1) // 2`; `calculate_dissonance_curve(partials, 1.0, 2.0, 200)`. |
| Mathematical formula | $N_{\mathrm{pairs}}=\binom{n}{2}$; curve samples $r\in[1,2]$ with `num_points=200$ per extraction. |
| Symbol definitions | $K=$ `DISSONANCE_PAIRWISE_PARTIAL_CAP`. |
| Notes | **Heuristic** complexity cap. |

### Pass 13 — `proc_audio.py` (peak refinement, f₀, note mapping)

### 68. `_parabolic_peak` — quadratic interpolation on linear magnitudes

| Field | Content |
|---|---|
| Function | `_parabolic_peak` |
| What it does | Three-point parabola; sub-bin offset and interpolated value. |
| Script | `proc_audio.py` |
| Line(s) | approx. 982–1002 |
| Python code | `denom = alpha - 2*beta + gamma`; `p = 0.5 * (alpha - gamma) / denom`; `xv = x + p`; `yv = beta - 0.25 * (alpha - gamma) * p`. |
| Mathematical formula | Extraction Pass 13: $p=\dfrac{\alpha-\gamma}{2(\alpha-2\beta+\gamma)}$, $x_{\mathrm v}=x+p$, $y_{\mathrm v}=\beta-\tfrac14(\alpha-\gamma)p$ on relative coordinates. |
| Symbol definitions | $(\alpha,\beta,\gamma)=(y_{x-1},y_x,y_{x+1})$. |
| Notes | Edge and `denom==0` guards in code. |

### 69. `_parabolic_interpolation_log_magnitude` — log-domain parabola and Hz correction

| Field | Content |
|---|---|
| Function | `_parabolic_interpolation_log_magnitude` |
| What it does | `log10` magnitudes; fit $ax^2+bx+c$ on $\{-1,0,1\}$; vertex; reject if $|x_{\mathrm{peak}}|>0.5$; else Hz correction. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1004–1059 |
| Python code | `log_mags = 20 * np.log10(np.maximum(magnitudes, 1e-10))`; `a = (y1 - 2*y2 + y3) / 2.0`; `b = (y3 - y1) / 2.0`; `x_peak = -b / (2 * a)`; `freq_corrected = freq_bin + x_peak * bin_spacing`. |
| Mathematical formula | Extraction Pass 13: $Y_k=20\log_{10}\max(M_k,\varepsilon)$; parabola coefficients; $f_{\mathrm{corr}}=f_{\mathrm{bin}}+x_{\mathrm{peak}}\Delta f$. |
| Symbol definitions | $\Delta f=$ `bin_spacing`. |
| Notes | Rejection cap $|x_{\mathrm{peak}}|\le \tfrac12$ bin. |

### 70. `_refine_peak_index` — windowed `argmax`

| Field | Content |
|---|---|
| Function | `_refine_peak_index` |
| What it does | Restricts to `[lo,hi)` window around approximate index; returns global argmax index. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1061–1085 |
| Python code | `lo = max(0, approx_idx - refine_radius)`; `hi = min(n, approx_idx + refine_radius + 1)`; `return lo + int(np.argmax(magnitudes[lo:hi]))`. |
| Mathematical formula | $k^\*=\arg\max_{i\in[\ell,h)} M_i$ with $\ell,h$ from extraction Pass 13. |
| Symbol definitions | `refine_radius` default 2. |
| Notes | Tie-break: first max (`argmax`). |

### 71. `_infer_bin_spacing_from_freqs` — median positive diffs

| Field | Content |
|---|---|
| Function | `_infer_bin_spacing_from_freqs` |
| What it does | Positive finite first-differences → `numpy.median`. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1087–1097 |
| Python code | `diffs = np.diff(freqs)`; `float(np.median(diffs[...]))` |
| Mathematical formula | $\Delta f=\mathrm{median}\{f_{i+1}-f_i : \Delta_i>0,\ \text{finite}\}$. |
| Symbol definitions | — |
| Notes | **`numpy.median` black box** for interpolation definition. |

### 72. `_refine_candidate_to_interpolated_peak` — nearest bin anchor and offset bins

| Field | Content |
|---|---|
| Function | `_refine_candidate_to_interpolated_peak` |
| What it does | Nearest frequency bin to candidate; dB magnitude; sub-bin offset in bins. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1099–1197 |
| Python code | `idx0 = int(np.argmin(np.abs(freqs - candidate_freq_hz)))`; `mag_db = float(20.0 * np.log10(max(amp, 1e-12)))`; `offset_bins = (freq_interp - bin_center) / bin_spacing`. |
| Mathematical formula | Extraction Pass 13. |
| Symbol definitions | $\hat f$: interpolated frequency from log-parabola path. |
| Notes | Composes helpers 69–71. |

### 73. `_saddle_prominence_db` — dB prominence vs flank minima

| Field | Content |
|---|---|
| Function | `_saddle_prominence_db` |
| What it does | Peak dB minus max of left/right window minima in dB with floors. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1199–1254 |
| Python code | `peak_db - max(left_min_db, right_min_db)` after `20*log10(max(...))` floors. |
| Mathematical formula | Extraction Pass 13 saddle prominence definition. |
| Symbol definitions | Window half-width `saddle_window` default 10. |
| Notes | Returns $-\infty$ on bad geometry. |

### 74. `_is_local_peak_valid` and `_local_peak_metrics` — local max, prominence, SNR gates

| Field | Content |
|---|---|
| Function | `_is_local_peak_valid`, `_local_peak_metrics` |
| What it does | Strict dB local maximum; prominence threshold; percentile noise floor; hard-coded `snr_db >= 3.0` gate in `_is_local_peak_valid` per extraction. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1256–1446 |
| Python code | `log_mags = 20 * np.log10(np.maximum(magnitudes, 1e-10))`; neighbour tests; `np.percentile` on windows; SNR differences. |
| Mathematical formula | Extraction Pass 13 (percentile via `numpy` black box). |
| Symbol definitions | `noise_floor_percentile`, `window_size`. |
| Notes | Policy constants (3 dB SNR) are **heuristic**. |

### 75. `_correct_f0_candidate_against_prior` — harmonic/subharmonic hypothesis set + cents error

| Field | Content |
|---|---|
| Function | `_correct_f0_candidate_against_prior` |
| What it does | Builds candidates $c/r, cr$ for integer $r$; minimises $1200|\log_2(f/f_{\mathrm{prior}})|$. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1518–1577 |
| Python code | `candidates.append((cand / float(r), 1.0 / float(r)))`; `err = abs(1200.0 * np.log2(hz / prior))`. |
| Mathematical formula | $\mathcal C=\{c\}\cup\{c/r,cr:r=2,\ldots,R\}$; choose $\arg\min_{f\in\mathcal C} 1200|\log_2(f/f_{\mathrm{prior}})|$. |
| Symbol definitions | $R=$ `max_harmonic_ratio` (default 6). |
| Notes | — |

### 76. `_calculate_bin_spacing` — zero-padding effective FFT length

| Field | Content |
|---|---|
| Function | `_calculate_bin_spacing` |
| What it does | Effective FFT size includes zero-padding factor. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1579–1598 |
| Python code | `n_fft_effective = n_fft * zero_padding`; `bin_spacing = sr / n_fft_effective` |
| Mathematical formula | $\Delta f = f_s/(N_{\mathrm{fft}} Z)$. |
| Symbol definitions | $Z=$ `zero_padding`. |
| Notes | — |

### 77. `frequency_to_note_name` — MIDI, tempered reference, cents offset

| Field | Content |
|---|---|
| Function | `frequency_to_note_name` |
| What it does | Continuous MIDI from Hz; rounds to nearest chroma; reference Hz; signed cents. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1650–1682 |
| Python code | `midi = 69.0 + 12.0 * math.log2(f / a4)`; `midi_round = int(round(midi))`; `f_ref = a4 * (2.0 ** ((midi_round - 69) / 12.0))`; `cents = 1200.0 * math.log2(f / f_ref)`. |
| Mathematical formula | $m=69+12\log_2(f/f_{A4})$; $\hat m=\mathrm{round}(m)$; $f_{\mathrm{ref}}=f_{A4}2^{(\hat m-69)/12}$; $\mathrm{cents}=1200\log_2(f/f_{\mathrm{ref}})$. |
| Symbol definitions | Default $f_{A4}=440$ Hz. |
| Notes | `.5` rounding ambiguity per extraction. |

### 78. `AudioProcessor.calculate_fundamental_frequency` — numeric Hz and tempered note path

| Field | Content |
|---|---|
| Function | `AudioProcessor.calculate_fundamental_frequency` |
| What it does | Parses numeric Hz; else builds $f_{C_0}$ anchor and $f=f_{C_0}2^{h/12}$ from note/octave per extraction. |
| Script | `proc_audio.py` |
| Line(s) | approx. 1684–1900 (approx.) |
| Python code | `freq_C0 = freq_A4 * 2 ** (-4.75)`; `f = freq_C0 * (2 ** (h / 12.0))` with `h = idx + 12 * octave`. |
| Mathematical formula | Extraction Pass 13: $f_{C_0}=440\cdot2^{-4.75}$; semitone index $h$; $f=f_{C_0}2^{h/12}$; equivalently $f=440\cdot2^{(m-69)/12}$. |
| Symbol definitions | Parsed pitch class and octave from string. |
| Notes | **Line numbers should be rechecked after code changes** (long method). |

### Pass 14 — `compile_metrics.py` and `audio_analysis/super_audio_analyzer.py`

### 79. `_sum_finite_numeric` — finite-only sum helper

| Field | Content |
|---|---|
| Function | `_sum_finite_numeric` |
| What it does | Sums numeric-coerced finite entries and counts them. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 1415–1435 |
| Python code | `mask = np.isfinite(num)`; `num[mask].sum()`, `mask.sum()` |
| Mathematical formula | $S=\sum_{j\in\mathcal J} x_j$, $n=|\mathcal J|$, $\mathcal J=\{j: x_j\ \text{finite}\}$ after `to_numeric`. |
| Symbol definitions | — |
| Notes | Used by legacy workbook extraction paths. |

### 80. `extract_density_component_sum` — linear / log / power / elementwise density

| Field | Content |
|---|---|
| Function | `extract_density_component_sum` |
| What it does | Masked finite sums; `log10(1+max(0,raw_total))` branch; power column or squared fallback; optional `apply_density_metric` elementwise path. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 2108–2368 |
| Python code | `raw_total = float(series[mask].sum())`; `np.log10(1.0 + max(0.0, raw_total))`; power modes; `apply_density_metric(..., normalize=False, ...)`. |
| Mathematical formula | Extraction Pass 14: $D=S$, $D=\log_{10}(1+\max(S,0))$, $D=\sum p_j$ or $\sum a_j^2$; elementwise path delegates to `density.apply_density_metric` (**partial black box**). |
| Symbol definitions | Masks per extraction (`include_for_density`, nonnegativity). |
| Notes | — |

### 81. `extract_density_components_from_per_note_workbook` — weighted logs and audits

| Field | Content |
|---|---|
| Function | `extract_density_components_from_per_note_workbook` |
| What it does | Harmonic log density; weighted products $H_{\mathrm{sum}}w_H$ etc.; weighted-sum log; weight-sum tolerance audit; per-band `extract_density_component_sum` refresh. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 2371–3650 (approx.) |
| Python code | `np.log10(1+max(0,H_sum))`; `weighted_h = h_f * wH_f`; `density_log = np.log10(1.0 + max(0.0, density_sum))`; `sum_w` tolerance check. |
| Mathematical formula | Extraction Pass 14: $L_H=\log_{10}(1+\max(0,H_{\mathrm{sum}}))$; $c_H=H_{\mathrm{sum}}w_H$; $L_w=\log_{10}(1+\max(0,S_w))$; $|w_H+w_I+w_S-1|>\tau$ audit. |
| Symbol definitions | $\tau=$ `DENSITY_WEIGHT_SUM_TOLERANCE`. |
| Notes | **Line numbers should be rechecked after code changes** (very long function). |

### 82. `get_frequency_dependent_alpha` and `apply_frequency_dependent_normalization`

| Field | Content |
|---|---|
| Function | `get_frequency_dependent_alpha`, `apply_frequency_dependent_normalization` |
| What it does | Piecewise $\alpha(f_0)$; expected partial-sum proxy $E(N)=(N^{1-\alpha}-1)/(1-\alpha)$ if $\alpha>1$ else $\log(N+1)$; `normalized_value = density / (expected_sum + 1e-10)`. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 4834–4938 |
| Python code | Piecewise returns `1.2,1.3,1.4,1.6`; `expected_sum` branches; division with `1e-10`. |
| Mathematical formula | Extraction Pass 14. |
| Symbol definitions | $N=$ `harmonic_count`. |
| Notes | Register policy for $\alpha(f_0)$. |

### 83. `_minmax` and `_robust_normalize_series`

| Field | Content |
|---|---|
| Function | `_minmax`, `_robust_normalize_series` |
| What it does | Min–max on series; or `data_integrity.robust_normalize` with clip $[0,1]$, ImportError fallback to `_minmax`. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 4772–5001 |
| Python code | `(s - lo) / (hi - lo)`; `robust_normalize(values, method=method, clip_range=(0.0, 1.0))`. |
| Mathematical formula | $z=(x-x_{\min})/(x_{\max}-x_{\min})$; robust path delegates to Pass 15 (`data_integrity`). |
| Symbol definitions | — |
| Notes | `robust_normalize` detailed in entries 90–93. |

### 84. `note_to_midi` and `note_to_fundamental_freq`

| Field | Content |
|---|---|
| Function | `note_to_midi`, `note_to_fundamental_freq` |
| What it does | Sorting MIDI-like index $(o+1)12+s$ with invalid sentinel; equal-temperament Hz from parsed note. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 4672–4832 |
| Python code | `(octv + 1) * 12 + semi`; `440.0 * (2.0 ** ((midi_note - 69) / 12.0))`. |
| Mathematical formula | $m=(o+1)12+s$, $f=440\cdot2^{(m-69)/12}$. |
| Symbol definitions | Invalid notes → large sentinel per extraction. |
| Notes | Mapping differs from strict MIDI in other modules (extraction warning). |

### 85. `apply_weighted_index` — log–log slope, debiasing, available-term index

| Field | Content |
|---|---|
| Function | `apply_weighted_index` |
| What it does | Robust norm on harmonic count; OLS $\hat\beta=\mathrm{Cov}(U,V)/\mathrm{Var}(V)$ on logs with clamp; $\hat c=\bar U-\hat\beta\bar V$; multiplicative debias; `_weighted_index_available_terms` rational; scheme weights; clips. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 5003–5216 |
| Python code | `alpha = np.cov(log_density, log_freq)[0,1]/np.var(log_freq)`; `alpha = max(-1.5, min(-0.3, alpha))`; `expected_density = np.exp(alpha * np.log(freq) + c)`; weighted sums with masks. |
| Mathematical formula | Extraction Pass 14: regression, clamp, $d^{\mathrm{freq}}_i=d_i/(\exp(\alpha\ln f_i+c)+\varepsilon)$; index $I_i$ as weighted sum over available terms only; schemes `"pdf"` and `"current"`. |
| Symbol definitions | $U=\ln d$, $V=\ln f$ on `valid_mask`. |
| Notes | **`numpy.cov` / `var` black boxes** for finite-sample definitions. |

### 86. `extract_dissonance_metrics` — substring column export contract

| Field | Content |
|---|---|
| Function | `extract_dissonance_metrics` |
| What it does | For each sheet/column whose name contains `"Dissonance"`, exports first finite row scalar. |
| Script | `compile_metrics.py` |
| Line(s) | approx. 5218–5250 (approx.) |
| Python code | `"Dissonance" in column`; `valid.iloc[0]`. |
| Mathematical formula | (Heuristic selection; no physics formula.) |
| Symbol definitions | — |
| Notes | Same export contract summarised in Pass 12 extraction; **different file/context** from entry 67. |

### 87. `finalize_batch_power_mass_summary` — nonnegative masses and percent renorm

| Field | Content |
|---|---|
| Function | `finalize_batch_power_mass_summary` |
| What it does | Clips masses; totals; raw percents; rescales to sum exactly 100. |
| Script | `audio_analysis/super_audio_analyzer.py` |
| Line(s) | approx. 156–250 (approx.) |
| Python code | `h = max(0.0, float(harmonic_power_mass))`; `total_power_mass = h + i + s`; `hp = 100.0 * h / total_power_mass`; `scale = 100.0 / ssum`; multiply percents. |
| Mathematical formula | $T=h+i+s$; $p_k=100 x_k/T$; if $s_{\%}=\sum p_k>0$, $\tilde p_k=100 p_k/s_{\%}$ so $\sum \tilde p_k=100$ (extraction Pass 14). |
| Symbol definitions | $h,i,s$: harmonic/inharmonic/subbass power masses. |
| Notes | **Line numbers should be rechecked after code changes.** |

### Pass 15 — `data_integrity.py`

### 88. `metric_float_or_nan`, `metric_int_or_nan`, `metric_ratio_or_nan`

| Field | Content |
|---|---|
| Function | `metric_float_or_nan`, `metric_int_or_nan`, `metric_ratio_or_nan` |
| What it does | Coerce to float/int with NaN/`pd.NA` sentinels; ratio with positive finite denominator guard. |
| Script | `data_integrity.py` |
| Line(s) | approx. 25–66 |
| Python code | `float(value)` with finite check; `int(x)` after float; `n/d` when finite and `d>0`. |
| Mathematical formula | Extraction Pass 15: missing/invalid → NaN or `pd.NA`; ratio $n/d$ if valid. |
| Symbol definitions | `MISSING_FLOAT = nan`. |
| Notes | — |

### 89. `calculate_iqr_bounds` — Tukey fences

| Field | Content |
|---|---|
| Function | `calculate_iqr_bounds` |
| What it does | Finite-only percentiles; IQR; $L=Q_1-k\,\mathrm{IQR}$, $U=Q_3+k\,\mathrm{IQR}$; empty → zeros tuple. |
| Script | `data_integrity.py` |
| Line(s) | approx. 80–110 |
| Python code | `Q1 = np.percentile(data_clean, 25)`; `Q3 = np.percentile(data_clean, 75)`; `lower_bound = Q1 - iqr_multiplier * IQR`; `upper_bound = Q3 + iqr_multiplier * IQR`. |
| Mathematical formula | $Q_1=P_{25}(\mathcal X)$, $Q_3=P_{75}(\mathcal X)$, $\mathrm{IQR}=Q_3-Q_1$, $L=Q_1-k\,\mathrm{IQR}$, $U=Q_3+k\,\mathrm{IQR}$. |
| Symbol definitions | $k=$ `iqr_multiplier` (default 1.5). |
| Notes | **`numpy.percentile` black box** beyond call parity. |

### 90. `detect_outliers` — Tukey mask on raw array

| Field | Content |
|---|---|
| Function | `detect_outliers` |
| What it does | Bounds from `calculate_iqr_bounds` on finite subset; compares **original** `data` elements to $L,U$. |
| Script | `data_integrity.py` |
| Line(s) | approx. 113–142 |
| Python code | `outlier_mask = (data < lower_bound) | (data > upper_bound)` |
| Mathematical formula | Outlier if $x_i<L$ or $x_i>U$ (finite comparisons; non-finite positions not flagged as outliers by inequality). |
| Symbol definitions | $L,U$ from entry 89 on `data[np.isfinite(data)]`. |
| Notes | Documented subtlety in extraction Pass 15. |

### 91. `robust_normalize` — affine maps (IQR / percentile / robust z / min–max) + clip + NaN preservation

| Field | Content |
|---|---|
| Function | `robust_normalize` |
| What it does | IQR affine on Tukey bounds; percentile affine; robust MAD z mapped $(z'+3)/6$; unknown method min–max on clean data; optional clip; preserves non-finite inputs as NaN in output slots. |
| Script | `data_integrity.py` |
| Line(s) | approx. 145–227 |
| Python code | `(data - lower_bound) / (upper_bound - lower_bound)`; `(data - p_low) / (p_high - p_low)`; `(data - median) / (1.4826 * mad)` then `(z_scores + 3.0) / 6.0`; else min–max; `np.clip`; `result[np.isfinite(data)] = normalized[np.isfinite(data)]`. |
| Mathematical formula | Extraction Pass 15 branches: $z_i=(x_i-L)/(U-L)$ with $L,U$ from Tukey fences; percentile $z_i=(x_i-P_\ell)/(P_u-P_\ell)$; $z_i'=(x_i-\tilde m)/(1.4826\,\mathrm{MAD})$, $u_i=(z_i'+3)/6$; fallback $z_i=(x_i-x_{\min})/(x_{\max}-x_{\min})$; clip; NaN restore. |
| Symbol definitions | Defaults $\ell=5$, $u=95$; `clip_range` default $[0,1]$. |
| Notes | Empty finite set → all-NaN output (shape preserved). |

### 92. `GlobalReferenceScaler.fit` / `GlobalReferenceScaler.transform`

| Field | Content |
|---|---|
| Function | `GlobalReferenceScaler.fit`, `GlobalReferenceScaler.transform` |
| What it does | Stores reference percentiles / IQR bounds / mean-std; affine transform using stored bounds; unfitted delegates to `robust_normalize(..., method="iqr")`; empty-clean transform returns zeros (differs from `robust_normalize` in edge case). |
| Script | `data_integrity.py` |
| Line(s) | approx. 234–366 |
| Python code | `fit`: percentile / iqr / mean_std branches; `transform`: affine + clip + finite mask restore; `if self.reference_stats is None: return robust_normalize(...)`. |
| Mathematical formula | Same affine forms as entry 91 but with **fitted** reference statistics (extraction Pass 15). |
| Symbol definitions | — |
| Notes | Policy inconsistency on empty-clean branch per extraction “ambiguous” flag. |

### 93. `validate_metric_value`, `validate_metric_array`, `validate_audio_parameters`

| Field | Content |
|---|---|
| Function | `validate_metric_value`, `validate_metric_array`, `validate_audio_parameters` |
| What it does | Range and NaN/Inf policy checks; array stats + outlier fraction vs `detect_outliers`; audio parameter thresholds; records $f_{\mathrm{Nyq}}=f_s/2$, $\Delta f=f_s/N_{\mathrm{fft}}$. |
| Script | `data_integrity.py` |
| Line(s) | approx. 383–531 |
| Python code | Comparisons on `value`; `outlier_fraction = np.sum(outlier_mask) / values_clean.size`; `nyquist = sr / 2.0`; `freq_resolution = sr / n_fft`. |
| Mathematical formula | Extraction Pass 15: $\rho$ outlier fraction; sampling relations for Nyquist and bin spacing. |
| Symbol definitions | — |
| Notes | Threshold policy beyond formulas is **heuristic** (warnings). |

### 94. `normalize_log_transform` — log1p min–max with reshape

| Field | Content |
|---|---|
| Function | `normalize_log_transform` |
| What it does | `pd.to_numeric` ravel; finite mask; $\ln(1+\max(x,\varepsilon))$; min–max; optional clip; scatter back to shape; all-nonfinite → zeros (policy differs from `robust_normalize`). |
| Script | `data_integrity.py` |
| Line(s) | approx. 538–591 |
| Python code | `log_data = np.log1p(data_positive)`; `(log_data - log_min) / (log_max - log_min)`; `result_flat.reshape(shape)`. |
| Mathematical formula | $u_j=\ln(1+\max(x_j,\varepsilon))$; $z_j=(u_j-u_{\min})/(u_{\max}-u_{\min})$; optional clip. |
| Symbol definitions | Default $\varepsilon=10^{-10}$. |
| Notes | Extraction Pass 15 flags empty-clean inconsistency vs other helpers. |

## 4. Limitations

- This document records **code–formula correspondence** for audit and documentation; it does **not** prove **scientific optimality** of the underlying acoustic or psychoacoustic models.
- **Third-party package internals** (`numpy`, `pandas`, `librosa`, `scipy`, etc.) are **not validated** here beyond noting which API calls the project makes; numerical details (percentile interpolation, `median`, `cov`, STFT definition) follow the upstream implementation.
- **Line numbers** are approximate for long functions and **will change** whenever source files are edited; always reconcile against the live file and the formula-extraction tables in `docs/formula_extraction/`.
- Entries marked **partial** or **algorithmic** defer full branch lists to the corresponding **`FORMULA_EXTRACTION_TABLE_PASS_*.md`** files, which remain the authoritative row-by-row decomposition for ambiguous regions.
