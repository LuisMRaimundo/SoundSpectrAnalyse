# Formula Extraction Table — Density Metrics, First Pass

Extracted from `MATHEMATICAL_FORMALISATION_VERIFICATION_REPORT_FIRST_PASS.md` (no new formulas).

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `compute_spectral_entropy` | `np.abs(power)` | \(u_i = |x_i|\) | \(x_i\): input elements | |
| `compute_spectral_entropy` | `power[power > 1e-12]` | \(\mathcal{I}=\{i\mid u_i>10^{-12}\}\), retain \(P_j=u_{i_j}\) | \(N\): count after filter | |
| `compute_spectral_entropy` | `total_power = np.sum(power)` | \(S=\sum_{j=1}^{N} P_j\) | \(P_j\): retained masses | |
| `compute_spectral_entropy` | `p = power / total_power` | \(p_j=P_j/S\), \(\sum p_j=1\) | | |
| `compute_spectral_entropy` | `-np.sum(p * np.log2(p))` | \(H=-\sum_{j=1}^{N} p_j\log_2 p_j\) | | Shannon entropy, base 2 |
| `compute_spectral_entropy` | `max_entropy = np.log2(len(power))` | \(H_{\max}=\log_2 N\) | \(N\): post-filter length | |
| `compute_spectral_entropy` | `normalized_entropy = entropy / max_entropy` | \(\tilde H=H/H_{\max}\) if \(H_{\max}>0\), else \(0\) | | Code branch for \(N=1\) |
| `compute_spectral_entropy` | `np.clip(..., 0.0, 1.0)` | \(H_{\mathrm{out}}=\operatorname{clip}(\tilde H,0,1)\) | | |
| `effective_partial_density_from_powers` | `d = (s * s) / ss` | \(D_{\mathrm{eff}}=S_1^2/S_2\) | \(S_1=\sum_i P_i\), \(S_2=\sum_i P_i^2\); \(P_i>\varepsilon\), finite | Inverse participation ratio |
| `_spectral_neff_from_filtered_linear_amplitudes` | `w = v * v`, `p = w / s`, `1.0 / np.sum(p * p)` | \(W_i=A_i^2\), \(S=\sum_i W_i\), \(p_i=W_i/S\), \(N_{\mathrm{eff}}=1/\sum_i p_i^2\) | \(A_i\ge 0\) amplitudes | Guards on empty \(S\) in code |
| `_apply_discrete_spectral_metrics` (d3) | `np.sum(np.log1p(values))` | \(\mathrm{d3}=\sum_{i=1}^{N}\ln(1+A_i)\) | \(A_i\): masked nonnegative finite amplitudes | Natural log |
| `_apply_discrete_spectral_metrics` (d10) | `np.sum(np.log1p(values)) * (neff / n)` | \(\mathrm{d10}=S_{\ln}\cdot N_{\mathrm{eff}}/N\) | \(S_{\ln}=\sum_i\ln(1+A_i)\); \(N_{\mathrm{eff}}\) from §3.3 | |
| `_apply_discrete_spectral_metrics` (d17) | `np.log1p(energy) * np.log1p(neff)` | \(\mathrm{d17}=\ln(1+E)\cdot\ln(1+N_{\mathrm{eff}})\) | \(E=\sum_i A_i^2\) | |
| `_apply_discrete_spectral_metrics` (d24) | `np.sum(np.log1p(masked_values))` | \(\mathrm{d24}=\sum_j\ln(1+A^{(24)}_j)\) | Subset with \(f_i\le12000\) and \(A_i\ge0.01A_{\max}\) | \(A_{\max}\) or override |
| `apply_density_metric` (discrete) | `_apply_discrete_spectral_metrics(key, ...)` | Return discrete metric for `key` \(\in\{\texttt{d3},\texttt{d10},\texttt{d17},\texttt{d24}\}\) | Same as `_apply_discrete_spectral_metrics` | Short-circuit path |
| `apply_density_metric` (generic) | `n_i = frequencies / fundamental_freq` | \(n_i=f_i/f_0\) | Continuous harmonic index | With rolloff branch |
| `apply_density_metric` (generic) | rolloff factor | \(E_i=(\max(n_i,1))^{-\alpha}\), \(\alpha=1.5\) literal | \(f_0>0\) | As in report |
| `apply_density_metric` (generic) | `a_i / (E_i + 1e-10)` | \(a_i\leftarrow a_i/(E_i+10^{-10})\) | Post–max-norm amplitudes \(a_i\) | |
| `apply_density_metric` (generic) | `np.sum(w(a_i))` | \(R=\sum_i w(a_i)\) | \(w\): weight from `get_weight_function` | |
| `apply_density_metric` (generic) | `result / len(values)` | \(R/N\) if `normalize` | \(N>0\) | Optional mean |
| `compute_rolloff_compensated_harmonic_density` | `np.round(f / f0)` (or orders) | \(n_i=\mathrm{round}(f_i/f_0)\) or \(\mathrm{round}(h_i)\) | Integer harmonic order | |
| `compute_rolloff_compensated_harmonic_density` | max-normalise | \(A^{\mathrm{norm}}_i=A_i/\max_j A_j\) | Retained nonnegative finite rows | |
| `compute_rolloff_compensated_harmonic_density` | `E_i = np.maximum(n_i, 1.0) ** (-alpha)` | \(E_i=(\max(n_i,1))^{-\alpha}\) | \(\alpha\) parameter (default constant in code) | |
| `compute_rolloff_compensated_harmonic_density` | `C_i = A_norm / (E_i + epsilon)` | \(C_i=A^{\mathrm{norm}}_i/(E_i+\varepsilon)\) | Default \(\varepsilon=10^{-12}\) | |
| `compute_rolloff_compensated_harmonic_density` | `np.sum(w(C))` | \(D=\sum_i w(C_i)\) | Default \(w(C)=\ln(1+C)\) | |
| `compute_rolloff_compensated_harmonic_density` | `D / A_first_partial` | \(D_{\mathrm{norm}}=D/A^{(1)}\) | \(A^{(1)}\): first raw amplitude with \(n_i=1\) | Else `nan` per report |
