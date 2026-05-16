# Formula Validation Plan — Pass 1 — Density metrics

## 1. Scope

Pass 1 of the formula-extraction workflow: **`density.py`** density metrics covered in `FORMULA_EXTRACTION_TABLE_DENSITY_FIRST_PASS.md`—spectral entropy, effective partial density, \(N_{\mathrm{eff}}\), discrete d3/d10/d17, and rolloff-compensated harmonic density.

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 1-01 | Spectral entropy (uniform two bins) | `power = np.array([1.0, 1.0])` | \(p=[0.5,0.5]\), \(H=1\), \(H_{\max}=1\) → **1.0** after clip | `density.compute_spectral_entropy` | `assert_allclose(..., 1.0, rtol=0, atol=1e-15)` | From consolidated plan §2: `rtol=1e-9`, `atol=1e-12` for closed-form algebra unless noted. |
| 1-02 | Spectral entropy (single survivor) | `power = np.array([1.0, 0.0, 0.0])` | One mass survives; code branch → **0.0** | `density.compute_spectral_entropy` | `assert_allclose(..., 0.0)` | |
| 1-03 | \(D_{\mathrm{eff}}=S_1^2/S_2\) | `powers = np.array([1.0, 1.0, 1.0])`, default `eps` | \(S_1=3\), \(S_2=3\) → **3.0** | `density.effective_partial_density_from_powers` | `assert_allclose(..., 3.0)` | |
| 1-04 | \(N_{\mathrm{eff}}=1/\sum p_i^2\) on \(A^2\) | `v = np.array([1.0, 1.0])` | \(W=[1,1]\), \(p=[0.5,0.5]\), \(\sum p^2=0.5\) → **2.0** | `density._spectral_neff_from_filtered_linear_amplitudes` | `assert_allclose(..., 2.0)` | |
| 1-05 | d3 \(=\sum \ln(1+A_i)\) | `values = np.array([1.0, 2.0])`, no freqs | \(\ln 2 + \ln 3\) ≈ **1.791759** | `density._apply_discrete_spectral_metrics("d3", values, None)` | `assert_allclose(..., rtol=1e-12)` | |
| 1-06 | d10 | `values = np.array([1.0, 1.0])` | \(S_{\ln}=2\ln 2\), \(N_{\mathrm{eff}}=2\), \(N=2\) → **\(2\ln 2\)** ≈ **1.386294** | `density._apply_discrete_spectral_metrics("d10", values, None)` | `assert_allclose` | |
| 1-07 | d17 | `values = np.array([1.0, 1.0])` | \(E=2\), \(N_{\mathrm{eff}}=2\) → \((\ln 3)^2\) ≈ **1.206949** | `density._apply_discrete_spectral_metrics("d17", values, None)` | `assert_allclose` | |
| 1-08 | Rolloff density (minimal harmonic) | `amplitudes=[1.0]`, `frequencies_hz=[100.0]`, `fundamental_freq_hz=100.0`, default `alpha`, `weight_function="logarithmic"` | Max-norm 1, \(n=1\), \(E=1\), \(C\approx 1/(1+\varepsilon)\), \(D\approx \ln(1+C)\approx \ln 2\) | `density.compute_rolloff_compensated_harmonic_density` (compare `rolloff_density_metric` or equivalent scalar in return dict) | `assert_allclose`, `atol=1e-9` | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
