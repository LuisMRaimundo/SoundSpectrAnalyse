# Formula Extraction Table — Pass 3 — Partial Sums and Metric Bundles

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `band_partial_metric_sum` | `v = v[np.isfinite(v) & (v >= 0.0)]` | Retain nonnegative finite amplitudes | \(v\): flattened vector | |
| `band_partial_metric_sum` | discrete branch | \(M=\texttt{\_apply\_discrete\_spectral\_metrics}(\texttt{key},v,f)\) | | `d24` passes optional `d24_global_amplitude_max` |
| `band_partial_metric_sum` | `np.sum(fn(v))` | \(\sum_i w(v_i)\) | \(w=\texttt{get\_weight\_function}(\texttt{key})\) | Continuous weights |
| `partial_metric_sums_h_i_s_total` | `gmax = float(np.nanmax(all_a_raw))` | \(A_{\max}=\max\) over concatenated H/I/S raw amplitudes | | For `d24` global gate |
| `partial_metric_sums_h_i_s_total` | `_band_linear_total` | \(s=\sum_{i:\,\mathrm{finite},\,a_i\ge 0} a_i\); band vector \([s]\) | | Non-discrete path only |
| `partial_metric_sums_h_i_s_total` | `h_sum`, `i_sum`, `s_sum` | \(H,I,S=\texttt{band\_partial\_metric\_sum}\) per band | | |
| `partial_metric_sums_h_i_s_total` | `t_sum = h_sum + i_sum + s_sum` | \(T=H+I+S\) | | Unless `wf in ("d10","d17")` |
| `partial_metric_sums_h_i_s_total` | `t_sum` for `d10`/`d17` | \(T=\texttt{band\_partial\_metric\_sum}(\texttt{concat}(a_H,a_I,a_S),\texttt{wf},\texttt{ff})\) | Concatenated amplitudes/frequencies | |
| `compute_discrete_spectral_metrics_bundle` | per-key `float(_apply_discrete_spectral_metrics(...))` | \(m_{\texttt{d3}}=\mathrm{d3}(a)\), \(m_{\texttt{d10}}=\mathrm{d10}(a)\), … | \(a\): harmonic partial amplitudes | `d24` uses `frequencies_hz` |
| `compute_discrete_spectral_metrics_bundle` | empty input | All keys `nan` | | |
