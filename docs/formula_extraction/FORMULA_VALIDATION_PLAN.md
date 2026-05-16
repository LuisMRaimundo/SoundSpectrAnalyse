# Formula Validation Plan

## 1. Purpose

This document lists **small, hand-checkable numerical examples** so that, when tests are implemented later, each extracted formula can be checked against the **same inputs** in Python. It does **not** assert that the physics or modelling choices are correct—only that **implementation matches the formula-extraction tables** (Passes 1–10).

## 2. Validation method

For each row below:

| Field | Content |
|--------|---------|
| **Formula / expression** | What is being checked (from the extraction tables). |
| **Input example** | Concrete numbers or small arrays / DataFrames. |
| **Manual expected result** | Value computed by hand or with a calculator / short symbolic derivation. |
| **Python target** | Callable or expression to invoke in the codebase (same semantics as the table). |
| **Assertion** | Typically `numpy.testing.assert_allclose(actual, expected, rtol=..., atol=...)`. |
| **Tolerance** | Recommend `rtol=1e-9`, `atol=1e-12` for closed-form float algebra; use `atol=1e-6` when the path involves `librosa`/window samples or iterative edge weights; use `atol=0` only for exact integers after rounding. |

**Workflow:** run the Python target on the input → compare to manual expected result with the given assertion type. Extend with parametrized cases only after the smoke examples pass.

## 3. Pass-by-pass validation plan

| Pass | Formula / expression | Input example | Manual expected result | Python target | Assertion |
|------|------------------------|---------------|------------------------|----------------|-----------|
| 1 | Spectral entropy (uniform two bins) | `power = np.array([1.0, 1.0])` | \(p=[0.5,0.5]\), \(H=1\), \(H_{\max}=1\) → **1.0** after clip | `density.compute_spectral_entropy` | `assert_allclose(..., 1.0, rtol=0, atol=1e-15)` |
| 1 | Spectral entropy (single survivor) | `power = np.array([1.0, 0.0, 0.0])` | One mass survives; code branch → **0.0** | `density.compute_spectral_entropy` | `assert_allclose(..., 0.0)` |
| 1 | \(D_{\mathrm{eff}}=S_1^2/S_2\) | `powers = np.array([1.0, 1.0, 1.0])`, default `eps` | \(S_1=3\), \(S_2=3\) → **3.0** | `density.effective_partial_density_from_powers` | `assert_allclose(..., 3.0)` |
| 1 | \(N_{\mathrm{eff}}=1/\sum p_i^2\) on \(A^2\) | `v = np.array([1.0, 1.0])` | \(W=[1,1]\), \(p=[0.5,0.5]\), \(\sum p^2=0.5\) → **2.0** | `density._spectral_neff_from_filtered_linear_amplitudes` | `assert_allclose(..., 2.0)` |
| 1 | d3 \(=\sum \ln(1+A_i)\) | `values = np.array([1.0, 2.0])`, no freqs | \(\ln 2 + \ln 3\) ≈ **1.791759** | `density._apply_discrete_spectral_metrics("d3", values, None)` | `assert_allclose(..., rtol=1e-12)` |
| 1 | d10 | `values = np.array([1.0, 1.0])` | \(S_{\ln}=2\ln 2\), \(N_{\mathrm{eff}}=2\), \(N=2\) → **\(2\ln 2\)** ≈ **1.386294** | `density._apply_discrete_spectral_metrics("d10", values, None)` | `assert_allclose` |
| 1 | d17 | `values = np.array([1.0, 1.0])` | \(E=2\), \(N_{\mathrm{eff}}=2\) → \((\ln 3)^2\) ≈ **1.206949** | `density._apply_discrete_spectral_metrics("d17", values, None)` | `assert_allclose` |
| 1 | Rolloff density (minimal harmonic) | `amplitudes=[1.0]`, `frequencies_hz=[100.0]`, `fundamental_freq_hz=100.0`, default `alpha`, `weight_function="logarithmic"` | Max-norm 1, \(n=1\), \(E=1\), \(C\approx 1/(1+\varepsilon)\), \(D\approx \ln(1+C)\approx \ln 2\) | `density.compute_rolloff_compensated_harmonic_density` (compare `rolloff_density_metric` or equivalent scalar in return dict) | `assert_allclose`, `atol=1e-9` |
| 2 | `WeightFunction.linear` | `x = 4.0` | **4.0** | `density.WeightFunction.linear(4.0)` | exact |
| 2 | `sqrt` / `squared` / `cubic` | `x = 4.0` | **2.0**, **16.0**, **64.0** | same class staticmethods | exact |
| 2 | `cbrt` | `x = -8.0` | **-2.0** | `WeightFunction.cbrt(-8.0)` | exact |
| 2 | `logarithmic` | `x = 4.0` | \(\ln 5\) ≈ **1.609438** | `WeightFunction.logarithmic(4.0)` | `assert_allclose` |
| 2 | `exponential` | `x = 4.0` | \(e^4-1\) | `WeightFunction.exponential(4.0)` | `assert_allclose` |
| 2 | `inverse_log` | `x = 4.0` | \(1/(\ln 5 + 10^{-10})\) | `WeightFunction.inverse_log(4.0)` | `assert_allclose`, `atol=1e-12` |
| 2 | `get_weight_function` alias | `name="sum"` or `"d2"` | resolves to same as `linear` | `density.get_weight_function("sum")(3.0)` vs `linear` | equal |
| 3 | `band_partial_metric_sum` linear | `amplitudes=[1.0, 2.0]`, `weight_key="linear"` | \(\sum A_i=\) **3.0** | `density.band_partial_metric_sum` | `assert_allclose` |
| 3 | H/I/S scalars + Total additive | `H=[1]`, `I=[2]`, `S=[3]`, `weight_key="linear"` | **H=1, I=2, S=3, T=6** | `density.partial_metric_sums_h_i_s_total` | `assert_allclose` each |
| 3 | d10/d17 Total = concatenated metric | Small `ah`, `ai` with known `d10` on concat | **T** equals `band_partial_metric_sum(concat(ah,ai,asb), "d10", ff)` not **H+I+S** | `density.partial_metric_sums_h_i_s_total(..., weight_key="d10")` | compare fourth return to direct `band_partial_metric_sum` |
| 3 | `compute_discrete_spectral_metrics_bundle` | `amplitudes=[1.0,1.0]` | `discrete_metric_d3` = \(2\ln 2\) | `density.compute_discrete_spectral_metrics_bundle` | key-wise `assert_allclose` |
| 4 | Exclusion half-width (relative) | `harmonic_df` one row 100 Hz; `complete_df` rows 100.0 and 103.0; `tolerance=0.02`; no leakage | \(\tau=2\) Hz; **100** masked in; **103** in residual subset | `density.identify_nonharmonic_residual_rows` | row counts / membership |
| 4 | Same with `spectral_leakage_guard=False` | as above | Same \(\tau\) if `leak_hw=0` | same | membership |
| 5 | Cents unison | `obs_hz=440`, `exp_hz=440` | **0.0** | `harmonic_alignment._cents(440.0, 440.0)` | `assert_allclose(..., 0.0)` |
| 5 | Cents octave | `obs_hz=880`, `exp_hz=440` | **1200.0** | `harmonic_alignment._cents(880.0, 440.0)` | `assert_allclose` |
| 5 | Adaptive tolerance floor | `sample_rate=None` | **18.0** | `harmonic_alignment._adaptive_tolerance_cents(1000.0, None, None)` | exact |
| 5 | Adaptive tolerance numeric | `expected_hz=100`, `sample_rate=44100`, `n_fft=4096` | Hand-compute \(\Delta f=f_s/N\), \(h^+=100+\Delta f/2\), \(\tau_{\mathrm{bw}}=1200\log_2(h^+/100)\), \(\tau=\max(18,2\tau_{\mathrm{bw}})\) | `harmonic_alignment._adaptive_tolerance_cents` | `assert_allclose`, `atol=1e-6` |
| 5 | Slot count | `f0_hz=100`, `max_frequency_hz=500` | \(N=\lfloor 500/100\rfloor=\) **5** (capped by `max_harmonics` in real call) | `compute_harmonic_alignment_metrics` → `total_expected_harmonic_orders` | match |
| 6 | dB → linear | Row with `Magnitude (dB)` = **20.0** | **10.0** | `peak_component_counts._linear_amp_from_row` with minimal `DataFrame` | `assert_allclose` |
| 6 | Hz tolerance from cents | `f0=100`, `n=2` → expected 200 Hz, `tolerance_cents=18` | \(\Delta f = 200\,(2^{18/1200}-1)\) | same formula as in `classify_peaks_harmonic_inharmonic_subbass_from_df` loop | numeric check vs **tol_hz** |
| 6 | Subbass vs harmonic | One peak at **50 Hz** (`<200`), one at **300 Hz** near **3×100** within tol | **s_n≥1**, harmonic slot filled if within window | `peak_component_counts.classify_peaks_harmonic_inharmonic_subbass_from_df` | integer counts |
| 7 | Margin piecewise | `f0_hz = 50`, `90`, `150`, `400` | **35**, **25**, **15**, **10** (percent) | `low_frequency_policy.calculate_subfundamental_margin_percent` | exact |
| 7 | Adaptive cutoff | `f0_hz=100`, defaults, no leakage | \(m=25\%\) (\(f_0<120\)); \(f_\%=100(1-25/100)=75\) Hz nominal; compare dict fields `raw_max`, `adaptive`, `effective_subfundamental_margin_percent` | `low_frequency_policy.calculate_adaptive_subfundamental_cutoff_hz` | dict fields `assert_allclose` |
| 7 | `classify_low_frequency_row` boundaries | `dc_floor_hz=30`, `ad=40`, `phys_hi=200` | **f=20** → dc; **f=35** → subfundamental; **f=150** → physical low; **f=300** → not low | `low_frequency_policy.classify_low_frequency_row` | string equality |
| 8 | `leakage_halfwidth_hz` | `sr=44100`, `n_fft=4096`, default main lobe | **\(0.5\times 4\times 44100/4096\)** Hz ≈ **21.533** | `spectral_leakage_guards.leakage_halfwidth_hz(sr=44100, n_fft=4096)` | `assert_allclose`, `atol=1e-3` |
| 8 | Filter candidates | `inharmonic_candidates=[(100.5, 1.0)]`, `harmonic_rep=[100.0]`, `lh=2.0` | Candidate **dropped** (\(|100.5-100|\le 2\)) | `spectral_leakage_guards.filter_inharmonic_peak_candidates` | `len(out)==0` |
| 9 | Canonical fallback | One-row `DataFrame`: `Density Metric` = **50**, missing canonical column | `canonical_density_v5_adapted` = **5.0** | `compile_metrics._add_canonical_and_global_density_columns` (fixture frame) | `assert_allclose` |
| 9 | Global normalisation | Two rows canonical **2** and **8** | `density_normalized_global` → **0.25**, **1.0**; denominator **8** | same helper | column-wise |
| 9 | Per-component density | `s_canon=6`, `harmonic_order_count=2` | **3.0** where division defined | same (after helper fills column) | `assert_allclose` |
| 9 | Weighted raw | \(D_H=10,w_H=0.2\Rightarrow c_H=2\); analogous **\(c_I,c_S\)** | `density_metric_raw` = sum contributions | `compile_metrics._compute_weighted_density_columns_for_wide_df` | `assert_allclose` on constructed wide row |
| 10 | RMS | `y = np.ones(4)` | \(\mathrm{RMS}=1\), `cur_db=0` if \(\log_{10}(1)=0\) | `proc_audio._normalize_level(y, target_rms_db=0.0)` | output RMS **1** (gain 1) |
| 10 | Gain | `y` with RMS **0.1** (`-20` dB), target **0** dB | \(G=10^{(0-(-20))/20}=10\), scaled max abs **1** | `_normalize_level` | `assert_allclose` max abs |
| 10 | Coherent gain Hann | `win="hann"`, `n_fft=4096` | \(G=(\sum w)/N\) — compare to independent NumPy `hanning` sum / N | `proc_audio._coherent_gain` | `assert_allclose`, `atol=1e-10` |
| 10 | `physical_peak_amplitude` | `mag=np.array([1.0])`, Hann, `n_fft` matching `_window_sum` | **\(2\cdot 1/S_w\)** one-sided | `proc_audio.physical_peak_amplitude` | `assert_allclose` |
| 10 | Energy ratio (Parseval helper) | Short synthetic `y` + `S` from `librosa.stft` on same `y` | `energy_ratio` near **1.0** within doc tolerance | `proc_audio._verify_energy_conservation` | `assert_allclose(..., 1.0, atol=0.1)` or stricter if stable |
| 10 | Weighted \(f_0\) LS | `detected_freqs=[200,400]`, `amps=[1,1]`, `initial_f0=100`, `max_n=10` | **\(f_0=100\)** (exact fit) | `proc_audio._estimate_f0_global_robust` | `assert_allclose` on `f0_estimated` |

## 4. Tests not yet created

This file **only plans** checks. It does **not** add `tests/`, does **not** run pytest, and does **not** modify application code. Implementation belongs in a follow-up change set (e.g. `tests/test_formula_validation_pass_01.py`, …).

## 5. Priority order for implementation

1. **Pass 1** — density metrics (core spectral summaries).  
2. **Pass 2** — weight functions (pure, fast, no I/O).  
3. **Pass 3** — partial sums and bundles (depends on 1–2).  
4. **Passes 4–8** — classification and policy gates (DataFrame / peak fixtures).  
5. **Pass 9** — compile-time normalisation (column wiring on small `DataFrame`s).  
6. **Pass 10** — selected `proc_audio` formulas (signal / STFT fixtures; slowest and most environment-sensitive).
