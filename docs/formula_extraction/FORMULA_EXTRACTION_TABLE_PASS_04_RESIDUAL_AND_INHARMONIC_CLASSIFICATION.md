# Formula Extraction Table — Pass 4 — Residual and Inharmonic Classification

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `identify_nonharmonic_residual_rows` | `thr_match` if `tolerance < 1.0` | \(\tau_{\mathrm{match}}=\max(f_h\,\tau,\,\varepsilon_f)\) | \(f_h\): harmonic reference frequency; \(\tau\): `tolerance`; \(\varepsilon_f\): `EPSILON_FREQUENCY` | Relative band |
| `identify_nonharmonic_residual_rows` | `thr_match` else | \(\tau_{\mathrm{match}}=\tau\) | | Absolute Hz if \(\tau\ge 1\) |
| `identify_nonharmonic_residual_rows` | `thr = max(thr_match, leak_hw)` when guard | \(\tau=\max(\tau_{\mathrm{match}},w_{\ell})\) | \(w_{\ell}\): `leakage_halfwidth_hz` | When guard on and \(w_\ell>0\) |
| `identify_nonharmonic_residual_rows` | `inharmonic_mask &= np.abs(all_freqs - f0) > thr` | Row \(k\) stays residual iff \(|f_k-f_h|>\tau\) for **every** harmonic \(f_h\) | \(f_k\): row frequencies | AND across loop |
| `identify_inharmonic_partials` | — | No formula-bearing expression identified. | | Thin wrapper to `identify_nonharmonic_residual_rows` |
