# Formula Extraction Table — Pass 6 — Peak Component Counts

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `_linear_amp_from_row` | `max(0.0, float(v))` from `Amplitude` | \(A=\max(0,a_{\mathrm{col}})\) | | |
| `_linear_amp_from_row` | `10.0 ** (float(v) / 20.0)` from dB | \(A=10^{L/20}\) | \(L\): `Magnitude (dB)` | |
| `_peak_tuples` | append `(f, a)` | List of \((f,A)\) pairs | \(f>0\), finite; \(A>0\), finite | Skips invalid rows |
| `classify_peaks_harmonic_inharmonic_subbass_from_df` | `n_slots = max(0, min(HARMONIC_MAX_CHECK, floor(max_f / f0)))` | \(N=\min(N_{\mathrm{cap}},\lfloor f_{\max}/f_0\rfloor)\) | \(N_{\mathrm{cap}}=\texttt{HARMONIC\_MAX\_CHECK}\) | |
| `classify_peaks_harmonic_inharmonic_subbass_from_df` | `expected = [f0 * float(n) for n in range(1, n_slots + 1)]` | \(f^{\mathrm{exp}}_n=n f_0\), \(n=1,\ldots,N\) | | |
| `classify_peaks_harmonic_inharmonic_subbass_from_df` | `tol_hz = expected_freq * (2.0 ** (tolerance_cents / 1200.0) - 1.0)` | \(\Delta f_n=f^{\mathrm{exp}}_n\,(2^{\tau/1200}-1)\) | \(\tau\): `tolerance_cents` | Hz tolerance |
| `classify_peaks_harmonic_inharmonic_subbass_from_df` | `err = abs(freq - expected_freq)` | \(e=|f-f^{\mathrm{exp}}_n|\) | | Smallest-error slot wins |
| `classify_peaks_harmonic_inharmonic_subbass_from_df` | harmonic dict update | One peak per \(n\) with largest \(A\) among Hz-valid hits | | |
| `classify_peaks_harmonic_inharmonic_subbass_from_df` | counts | \(N_h=|\mathrm{dict}|,\ N_i=|\mathrm{inharmonic}|,\ N_s=|\mathrm{subbass}|\) | Subbass: \(f<f_{\mathrm{cut}}\) | |
