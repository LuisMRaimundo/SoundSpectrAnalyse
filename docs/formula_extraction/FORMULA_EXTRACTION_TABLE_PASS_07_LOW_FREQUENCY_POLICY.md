# Formula Extraction Table — Pass 7 — Low-Frequency Policy

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `calculate_subfundamental_margin_percent` | piecewise on `f0` | \(m\in\{35,25,15,10\}\) percent by \(f_0\) bands \(<60\), \(<120\), \(<300\), else | \(m\): margin % | Invalid \(f_0\) \(\Rightarrow 10\) |
| `calculate_adaptive_subfundamental_cutoff_hz` | `percentage_cut = f0 * (1.0 - margin / 100.0)` | \(f_\%=f_0(1-m/100)\) | | |
| `calculate_adaptive_subfundamental_cutoff_hz` | `parts` / `raw_max` | \(F_{\mathrm{raw}}=\max(f_{\mathrm{floor}},f_\%,f_{\ell})\) | \(f_{\ell}\): optional leakage cutoff | Finite candidates only |
| `calculate_adaptive_subfundamental_cutoff_hz` | `adaptive = min(raw_max, cap_hz)` | \(f_{\mathrm{ad}}=\min(F_{\mathrm{raw}},\,\gamma f_0)\) | \(\gamma=\) `max_fraction_of_f0` | |
| `calculate_adaptive_subfundamental_cutoff_hz` | `eff_margin = 100.0 * (1.0 - adaptive / f0)` | \(m_{\mathrm{eff}}=100(1-f_{\mathrm{ad}}/f_0)\) | | |
| `classify_low_frequency_row` | `f <= dc_floor_hz` | DC/sub-audible bucket | \(f\): row frequency | |
| `classify_low_frequency_row` | `f < ad` | Subfundamental band | `ad`: adaptive cutoff | |
| `classify_low_frequency_row` | `f <= physical_low_band_upper_hz` | Physical low-frequency residual | | |
| `classify_low_frequency_row` | else | `not_low_frequency_residual` | | Label only |
