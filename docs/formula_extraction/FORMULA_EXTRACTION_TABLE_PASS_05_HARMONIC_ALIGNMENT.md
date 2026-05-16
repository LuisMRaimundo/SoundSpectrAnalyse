# Formula Extraction Table — Pass 5 — Harmonic Alignment

| Function | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `_linear_amp_and_energy` | `amp = float(row["Amplitude_linear"])` etc. | \(A\): linear amplitude from column priority | | dB columns: \(A=10^{L/20}\) |
| `_linear_amp_and_energy` | `amp = float(max(amp, 0.0))` | \(A\leftarrow\max(A,0)\) | | |
| `_linear_amp_and_energy` | `return amp, amp * amp` | \((A,E)=(A,A^2)\) | | Energy |
| `_cents` | `1200.0 * math.log2(obs_hz / exp_hz)` | \(c=1200\log_2(f_{\mathrm{obs}}/f_{\mathrm{exp}})\) | Cents deviation | Returns `nan` if frequencies \(\le 0\) |
| `_adaptive_tolerance_cents` | default branch | \(\tau=18.0\) | cents | Missing/invalid STFT geometry |
| `_adaptive_tolerance_cents` | `bin_w = sample_rate / n_fft` | \(\Delta f=f_s/N_{\mathrm{fft}}\) | | |
| `_adaptive_tolerance_cents` | `hi = expected_hz + bin_w / 2.0` | \(f^+=f_{\mathrm{exp}}+\Delta f/2\) | | |
| `_adaptive_tolerance_cents` | `bw_cents = 1200.0 * math.log2(hi / expected_hz)` | \(\tau_{\mathrm{bw}}=1200\log_2(f^+/f_{\mathrm{exp}})\) | | Clamped \(\ge 0\) |
| `_adaptive_tolerance_cents` | `return max(18.0, 2.0 * bw_cents)` | \(\tau=\max(18,\,2\tau_{\mathrm{bw}})\) | | |
| `_tolerance_for_order` | `if tolerance_cents is not None` | \(\tau=\texttt{tolerance\_cents}\) | | Fixed override |
| `_tolerance_for_order` | else | \(\tau=\texttt{\_adaptive\_tolerance\_cents}(n f_0,\ldots)\) | \(n\): harmonic order | |
| `_in_any_harmonic_window` | loop over `n` | \(f_{\mathrm{exp},n}=n f_0\); test \(|c(f,f_{\mathrm{exp},n})|\le \tau_n\) | \(c=\texttt{\_cents}\); \(\tau_n=\texttt{\_tolerance\_for\_order}(n,\ldots)\) | Stops when \(n f_0>f_{\max}\) |
| `compute_harmonic_alignment_metrics` | `n_slots = max(0, min(int(max_harmonics), int(math.floor(max_f / f0))))` | \(N_{\mathrm{slots}}=\min(N_{\max},\lfloor f_{\max}/f_0\rfloor)\) | | |
| `compute_harmonic_alignment_metrics` | partition energies | \(E_{\mathrm{sub}},E_{\mathrm{reg}},E_{\mathrm{inh}}=\sum A^2\) per class | | Mutually exclusive buckets |
| `compute_harmonic_alignment_metrics` | `n_round = int(round(f_hz / f0))` | \(n=\mathrm{round}(f/f_0)\) | | Assignment |
| `compute_harmonic_alignment_metrics` | collapse per `n` | Winner \(=\arg\max A^2\) within cents gate per bucket | | |
| `compute_harmonic_alignment_metrics` | `w_mean = np.sum(w * e) / np.sum(w)` | \(\bar{e}_w=\sum_j E_j e_j/\sum_j E_j\) | \(E_j\): matched energies; \(e_j\): abs cents errors | Energy-weighted mean |
| `compute_harmonic_alignment_metrics` | `ratio_orders = matched_count / n_slots` | \(\rho=N_{\mathrm{match}}/N_{\mathrm{slots}}\) | | |
| `compute_harmonic_alignment_metrics` | `_ratio(e_region, e_total)` etc. | \(r_x=E_x/(E_{\mathrm{sub}}+E_{\mathrm{reg}}+E_{\mathrm{inh}})\) | | Denominator guard in helper |
