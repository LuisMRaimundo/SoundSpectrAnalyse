# Formula Extraction Table — Pass 9 — Compile-Time Normalisation

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `_add_canonical_and_global_density_columns` | fallback `out[canon_col] = dm / 10.0` | \(d_{\mathrm{canon}}=d_{\mathrm{DM}}/10\) | \(d_{\mathrm{DM}}\): `Density Metric` | When canonical column missing/empty |
| `_add_canonical_and_global_density_columns` | `mx = np.nanmax(finite)` | \(M=\max_k d_{\mathrm{canon},k}\) over finite values | | |
| `_add_canonical_and_global_density_columns` | `density_normalized_global = (s_canon / mx).clip(0,1)` | \(\hat d_k=\operatorname{clip}(d_k/M,0,1)\) | | |
| `_add_canonical_and_global_density_columns` | `density_per_component = s_canon / hoc.replace(0, nan)` | \(d^{(\mathrm{pc})}_k=d_k/N_k\) | \(N_k\): `harmonic_order_count` | Division-by-zero masked |
| `_compute_weighted_density_columns_for_wide_df` | `wh = (D_H * w_H)` | \(c_{H,k}=D_{H,k}\,w_{H,k}\) | \(D_H\): `Harmonic Partials sum`; \(w_H\): component ratio | |
| `_compute_weighted_density_columns_for_wide_df` | `raw = wh.fillna(0) + ...` | \(r_k=c_{H,k}+c_{I,k}+c_{S,k}\) with NaN-as-zero unless all NaN | | `density_metric_raw` |
| `_compute_weighted_density_columns_for_wide_df` | `mx = np.max(finite_pos)` | \(M_r=\max_k r_k\) over finite \(r_k>0\) | | For weighted norm |
| `_compute_weighted_density_columns_for_wide_df` | `density_metric_normalized = raw / mx` | \(\tilde r_k=r_k/M_r\) | | Run-relative max norm |
