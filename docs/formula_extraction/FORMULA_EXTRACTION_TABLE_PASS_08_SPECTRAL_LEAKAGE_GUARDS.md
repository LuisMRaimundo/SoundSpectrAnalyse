# Formula Extraction Table — Pass 8 — Spectral Leakage Guards

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `leakage_halfwidth_hz` | `bw = sr / n_fft` | \(\Delta f=f_s/N_{\mathrm{fft}}\) | | When `bin_width_hz` absent |
| `leakage_halfwidth_hz` | `return 0.5 * ml * bw` | \(w_\ell=\tfrac{1}{2}\,B\,\Delta f\) | \(B\): `main_lobe_bins` or default `DEFAULT_MAIN_LOBE_WIDTH_BINS` | Half-width Hz |
| `filter_inharmonic_peak_candidates` | `np.any(np.abs(hf - ff) <= lh)` | Drop candidate \((f,a)\) if \(\exists h:\ |h-f|\le w_\ell\) | \(h\in\) harmonic rep frequencies | |
