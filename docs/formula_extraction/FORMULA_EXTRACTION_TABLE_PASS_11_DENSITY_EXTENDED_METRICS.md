# Formula Extraction Table — Pass 11 — Density Extended Metrics

Project-owned `density.py` only (Passes 1–10 elsewhere). Third-party primitives are black boxes.

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `estimate_noise_floor` | `np.percentile(psd_positive, percentile)` | \(F=\mathrm{Percentile}_{p\%}(\mathrm{PSD}^+)\) | Positive flattened PSD; \(p=\) `percentile` | Power domain; **extracted** |
| `physical_spectral_density` | `SpectralDensityMetrics.physical_spectral_density` | \(P_i=A_i^2\), \(N_{\mathrm{eff}}=(\sum P_i)^2/\sum P_i^2\), score \(=N_{\mathrm{eff}}/N\) clipped to \([0,1]\) | \(A_i>0\) finite | Max-normalised participation / \(N\); `bin_width_hz` unused; **extracted** |
| `perceptual_spectral_density` | `bark = 13*arctan(0.00076*f)+3.5*arctan((f/7500)**2)` | \(B(f)=13\arctan(0.00076f)+3.5\arctan((f/7500)^2)\) | \(f=\max(f_{\mathrm{Hz}},1)\) | Zwicker–Terhardt-style Bark; **extracted** |
| `perceptual_spectral_density` | band weights & entropy | \(E_b=\sum_i P_i w_{b}(B(f_i))\), occupancy \(=\#\{E_b>0\}/\#\text{bands}\), uniformity \(=H/\log_2 K\) for \(q_k=E_k/\sum E\) | \(P_i=A_i^2\); triangular weight \(w_b=\max(0,1-|B-b_{\mathrm{center}}|)\) | Entropy base 2; **extracted** |
| `perceptual_spectral_density` | `0.6 * occupancy + 0.4 * uniformity` | \(D=0.6\,O+0.4\,U\) clipped \([0,1]\) | | Linear blend; **extracted** |
| `calculate_harmonic_density` | `max_expected_harmonics` | \(N_{\max}=\max(1,\lfloor f_{\mathrm{Nyq}}/f_0\rfloor)\) or fallback 50 | \(f_{\mathrm{Nyq}}=sr/2\) or 20 kHz | Count cap; **extracted** |
| `calculate_harmonic_density` | `20*log10(max(amps,1e-12))` | \(L_i=20\log_{10}\max(A_i,\varepsilon)\) | dB from linear amplitude | **extracted** |
| `calculate_harmonic_density` | `significant.sum() / max_expected_harmonics` | \(\rho=\frac{1}{N_{\max}}\sum_i \mathbf{1}[L_i>L_{\mathrm{thr}}]\) | \(L_{\mathrm{thr}}=\) `threshold_db` | Relative count density; **extracted** |
| `calculate_harmonic_density` | optional `tanh(mean)` blend | \(D=\mathrm{clip}((1-w_\rho)\rho+w_\rho\tanh(\bar A_{\mathrm{sig}}),0,1)\) | \(w_\rho=\) `amp_weight` | **extracted** |
| `calculate_inharmonic_density` | body | Delegates to `calculate_harmonic_density(..., max_expected_harmonics=max_expected_partials)` | | **No new formula** — same as harmonic row |
| `calculate_perceptual_spectral_density` | `harmonic_db = 20*log10(max(amps,1e-12))` | Same dB map as above | | If input already dB branch skips; **partial** |
| `calculate_perceptual_spectral_density` | `max_possible_harmonics` | \(N_{\mathrm{cap}}=\lfloor f_{\lim}/f_0\rfloor\) | `frequency_limit` | **extracted** |
| `calculate_perceptual_spectral_density` | band energies from Bark index | \(k_i=\lfloor B(f_i)\rfloor\) clipped to 24 bands; power \(\propto (10^{L_i/20})^2\) accumulated | | dB→linear then square; **extracted** |
| `calculate_perceptual_spectral_density` | masking loop on bands | Uses `_critical_band_masking` in dB between band centers (piecewise Bark→Hz proxy) | | Inverse Bark→Hz is **heuristic / piecewise**; **ambiguous — human review** |
| `calculate_perceptual_spectral_density` | `occupancy_density`, `uniformity`, `completeness` | \(D_{\mathrm{fin}}=w_O O+w_U U+w_C C\) then `1-exp(-k D_fin)` | Constants `PERCEPTUAL_DENSITY_*`, `PERCEPTUAL_DENSITY_LOG_SCALE_FACTOR` | Nonlinear squash; **partial** (constants imported) |
| `_critical_band_masking` | piecewise on `bark_distance` | \(T=L_{\mathrm{masker}}+\Delta(L_{\mathrm{masker}},\Delta b)\) with slope/offset constants | \(\Delta b=|b(f_p)-b(f_m)|\); outputs dB threshold | Constants from `constants`; **extracted** |
| `_critical_band_masking` | `max(threshold_db, MASKING_ABSOLUTE_THRESHOLD_DB)` | \(T\leftarrow\max(T,L_{\min})\) | | Floor; **extracted** |
| `estimate_noise_floor_by_critical_bands` | `margin_db = 20*log10(mult)` | \(M_{\mathrm{dB}}=20\log_{10}k\) when margin not explicit | \(k=\) `noise_floor_multiplier` (linear factor → dB) | FIX 3; **extracted** |
| `estimate_noise_floor_by_critical_bands` | per-band percentile | \(\tau_b=\max(P_{p\%}(L_b)+M_{\mathrm{dB}},L_{\min})\) | \(L_b\): dB magnitudes in Hz band | Bands fixed Hz ranges; **extracted** |
| `estimate_noise_floor_by_critical_bands` | boundary interpolation | Linear blend of \(\tau_i,\tau_{i+1}\) vs normalised distance near band edge | `weights` formula mixes adjacent floors | **ambiguous — human review** (weight line mixes low/high) |
| `apply_spectral_masking_filter` | double loop threshold | Audible iff \(\forall\) stronger maskers: \(L_{\mathrm{probe}}\ge T(f_m,L_m,f_p)\) with \(T=\) `_critical_band_masking` | dB levels | Sort by descending dB; **partial** (algorithm, not closed form) |
| `calculate_spectral_complexity` | moving-average irregularity | \(\mathrm{irr}=\mean(|a-\tilde a|)/\mean(a)\) with \(\tilde a=\) conv boxcar | Linear `Amplitude` or dB→linear | **extracted** |
| `calculate_spectral_complexity` | harmonic energy tube | \(E_H=\sum_{n}\sum_{f\in T_n} a^2\), \(T_n\) ±3% of \(n f_0\) | | **extracted** |
| `calculate_spectral_complexity` | `inharmonicity = 1 - harmonic_energy/total_energy` | \(I=1-E_H/E\) | \(E=\sum a^2\) | **extracted** |
| `calculate_spectral_complexity` | entropy term | \(H=-\sum p_i\log_2 p_i\) with \(p_i=a_i^2/E\), then `/log2(len(probs))` | | Normalised entropy; **extracted** |
| `calculate_spectral_complexity` | `0.4*irregularity + 0.4*inharmonicity + 0.2*entropy` | Clip \([0,1]\) | | **extracted** |
| `calculate_harmonic_richness` | count factor | \(C=\min(1,N/N_{\max})\) | \(N=\) row count | **extracted** |
| `calculate_harmonic_richness` | geometric mean | \(G=\exp(\mean(\ln A_i))\) on \(A_i>0\); `tanh(G)` | | **extracted** |
| `calculate_harmonic_richness` | blend | \(R=(1-w)C+w\tanh(G)\) | `amplitude_weight` \(=w\) | **extracted** |
| `_calculate_harmonic_completeness_phase2` | `tolerance = BASE*(1+ADAPTIVE*n)` | \(\tau_n=\tau_0(1+c n)\) | Constants `HARMONIC_TOLERANCE_*` | Relative freq match; **extracted** |
| `_calculate_harmonic_completeness_phase2` | gap weights | Completeness \(=1-\mathrm{gap\_penalty}/\mathrm{total\_weight}\) with weights \(\propto 1/n\) | | Clipped; **extracted** |
| `compute_harmonic_effective_power_density` | `p_norm = pwr / max_p`; `dens = sum(p_norm)` | \(\tilde P_i=A_i^2/\max_j A_j^2\), \(D=\sum_i \tilde P_i\) | | Additive descriptor; also `dens/N`; **extracted** |
| `compute_harmonic_effective_power_mass` | `power = square(amplitudes)` | \(E=\sum A_i^2\), \(\bar P=\mean(A_i^2)\), \(\mathrm{RMS}=\sqrt{\bar P}\) | | **extracted** |
| `compute_subbass_protection_tolerance_hz` | `max(minimum_hz, bin_multiplier * sr/n_fft)` | \(\tau=\max(\tau_{\min},\,k\,f_s/N_{\mathrm{fft}})\) | Defaults 12 Hz, \(k=4\) | **extracted** |
| `aggregate_low_frequency_residual_peak_power` | dB→linear | \(A_i=10^{L_i/20}\) if dB column | | **extracted** |
| `aggregate_low_frequency_residual_peak_power` | `_harmonic_mask` | Exclude if \(\min_h|f_i-h|\le \tau\) | \(\tau=\) `freq_match_tol_hz` | **extracted** |
| `aggregate_low_frequency_residual_peak_power` | `sum_all_bins` mode | \(\sum_{i\in\mathcal{B}} A_i^2\) | Band \((f_{\mathrm{lo}},f_{\mathrm{hi}}]\) minus harmonic mask | Power; **extracted** |
| `aggregate_low_frequency_residual_peak_power` | strict local max mode | Same sum restricted to strict local maxima of \(A\) on grid | | **extracted** |
| `partial_density_effective_components_bundle` | `thresh = ref * (10**(min_db_relative/10.0))` | \(T=R\cdot 10^{r/10}\) with \(r=\) `min_db_relative` | \(R=\max(\max P_{H,i},\max P_{I,i},g)\) powers | dB-like **relative power** threshold on power; **extracted** |
| `partial_density_effective_components_bundle` | `d = _inverse_herfindahl_effective_components(p_arr)` | \(D_{\mathrm{eff}}=(\sum p_k)^2/\sum p_k^2\) on merged power bins | Same as `effective_partial_density_from_powers` on constructed list | **extracted** |
| `partial_density_effective_components` | calls bundle | Returns scalar only | | **No new formula** |
| `calculate_combined_density_metric` | weight renorm | \(\alpha\leftarrow\alpha/(\alpha+\beta)\) if sum \(\neq 1\) | | **extracted** |
| `calculate_combined_density_metric` | log branch | \(\mathrm{expm1}(\alpha\ln(1+h_+)+\beta\ln(1+i_+))\) with \(h_+,i_+=\max(0,\cdot)\) | Natural log via `log1p`/`expm1` | **extracted** |
| `calculate_combined_density_metric` | linear fallback | \(\alpha h+\beta i\) | | **extracted** |
| `_hz_to_bark` | return expression | \(B(f)=13\arctan(0.00076f)+3.5\arctan((f/7500)^2)\) | \(f\) in Hz (array-safe) | Matches perceptual Bark; **extracted** |
| `spectral_density` | `p = (amps**gamma); p/=p.sum()` | \(p_i\propto A_i^{\gamma}\), \(\sum p_i=1\) | Default \(\gamma=2\) → power | **extracted** |
| `spectral_density` | optional window on \(f_0\) | Restrict to \(f_0\le f\le f_0+h_w\); renormalise \(p\) | `hz_window` | **extracted** |
| `spectral_density` | Hill / Rényi entropy step | \(N_{\mathrm{eff}}=\exp(H)\) if \(q=1\) else \((\sum p_i^q)^{1/(1-q)}\); \(R=(N_{\mathrm{eff}}-1)/(M-1)\) | \(M\) after capping | \(q\) default 1; **extracted** |
| `spectral_density` | Gaussian proximity (small \(M\)) | \(P_{\mathrm{num}}=\sum_{i\neq j} p_i p_j \exp(-d_{ij}^2/\sigma^2)\) with \(\sigma^2=2\sigma_{\mathrm{Hz}}^2\) | \(d_{ij}=|f_i-f_j|\) | Diagonal zeroed; **extracted** |
| `spectral_density` | `P_norm = P_num / (1 - sum(p**2))` | Normalised proximity vs Simpson complement | | Denom small → 0; **extracted** |
| `spectral_density` | `W_low`, `D_peso` | \(W=E_{\mathrm{low}}/E\), \(D_{\mathrm{peso}}=(1-\lambda)D_{\mathrm{core}}+\lambda W\) with \(D_{\mathrm{core}}=w_r R+w_p P\) | Low band via `low_hz_cut` smooth weight | **partial** (many branches / large \(M\) path) |
| `spectral_density` | `D_harm` block | Harmonic-bin mass near \(n f_0\) (±2% tolerance) | | **partial** — remainder of function long; **ambiguous — human review** for full thesis |

## Summary

- **Functions fully extracted (core closed-form rows):** `estimate_noise_floor`; module-level **`physical_spectral_density`** / **`perceptual_spectral_density`** (via `SpectralDensityMetrics`); **`calculate_harmonic_density`**; **`_critical_band_masking`**; **`compute_subbass_protection_tolerance_hz`**; **`aggregate_low_frequency_residual_peak_power`** (both modes); **`partial_density_effective_components_bundle`** (threshold + \(D_{\mathrm{eff}}\)); **`compute_harmonic_effective_power_density`**; **`compute_harmonic_effective_power_mass`**; **`calculate_combined_density_metric`**; **`_hz_to_bark`**; **`calculate_harmonic_richness`**; **`_calculate_harmonic_completeness_phase2`**; **`calculate_spectral_complexity`**; key blocks of **`spectral_density`** (mass, Hill/Rényi, Gaussian proximity, \(P_{\mathrm{norm}}\)).

- **Functions partially extracted:** **`calculate_perceptual_spectral_density`** (masking + inverse-Bark centres heuristic); **`estimate_noise_floor_by_critical_bands`** (boundary smoothing); **`apply_spectral_masking_filter`** (iterative audibility); **`spectral_density`** (large \(M\) approximate proximity path and full **`D_harm`** tail not fully tabulated).

- **Functions not found / N/A:** none from the requested list (all symbols exist in `density.py`).

- **Ambiguous / needs human review:** (1) **`calculate_perceptual_spectral_density`** — piecewise Bark→Hz centre for masking comparison vs band-level dB. (2) **`estimate_noise_floor_by_critical_bands`** — boundary interpolation weights vs adjacent band floors (verify intent vs code). (3) **`spectral_density`** — optional **`D_harm`** harmonic-mass aggregation and sparse \(M>1000\) proximity approximation vs full \(O(N^2)\) path.

- **Recommendation:** **Yes — add a Pass 11 validation-plan document later** (`FORMULA_VALIDATION_PLAN_PASS_11_…`) with hand-checks for: `physical_spectral_density` two-partial toys; `calculate_harmonic_density` threshold count; `partial_density_effective_components_bundle` threshold splitting; `aggregate_low_frequency_residual_peak_power` toy grid vs harmonic mask; and a **small** `spectral_density` fixture with \(M\le 100\) to pin \(R_{\mathrm{norm}}\) and \(P_{\mathrm{norm}}\). Skip full Monte Carlo of masking until ambiguities are resolved.
