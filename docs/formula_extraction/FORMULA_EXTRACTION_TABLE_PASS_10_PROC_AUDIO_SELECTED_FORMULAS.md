# Formula Extraction Table — Pass 10 — Proc Audio (Selected Formulas)

Subset of `proc_audio.py` only (STFT scaling path, calibration, energies, f₀ fit, export semantics).

| Function / context | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `_normalize_level` | `rms = sqrt(mean(square(y)) + 1e-12)` | \(\mathrm{RMS}=\sqrt{\overline{y^2}+\varepsilon}\) | \(\varepsilon=10^{-12}\) | Before STFT |
| `_normalize_level` | `cur_db = 20 * log10(rms)` | \(L_{\mathrm{RMS}}=20\log_{10}(\mathrm{RMS})\) | | |
| `_normalize_level` | `gain = 10 ** ((target_rms_db - cur_db) / 20)` | \(G=10^{(L_{\mathrm{tgt}}-L_{\mathrm{RMS}})/20}\) | | |
| `_normalize_level` | `y * gain` | \(y' = G y\) | | |
| STFT pipeline | `self.S = librosa.stft(...)` | Complex STFT \(S[k,t]\) | | `librosa` as implementation |
| STFT pipeline | `S_mag = np.abs(self.S)` | \(M_{k,t}=|S_{k,t}|\) | | Magnitude |
| STFT pipeline | `librosa.amplitude_to_db(S_mag, ref=1.0)` | \(L_{k,t}=20\log_{10}(M_{k,t})\) (librosa `ref=1`) | | Display/threshold dB |
| STFT pipeline | `librosa.fft_frequencies(sr=..., n_fft=...)` | \(f_k=k\,f_s/N_{\mathrm{fft}}\) (librosa bin convention) | \(k\): bin index | Exported `self.freqs` |
| `_coherent_gain` | `np.sum(w) / float(n_fft)` | \(G=\frac{1}{N}\sum_n w[n]\) | \(N=N_{\mathrm{fft}}\) | Stored as `coherent_gain_value` |
| `_window_sum` | `np.sum(w)` | \(S_w=\sum_n w[n]\) | | No \(1/N\) |
| `physical_peak_amplitude` | `factor * mag / sw` | \(A_{\mathrm{peak}}=\gamma\,M/S_w\) | \(\gamma=2\) one-sided else \(1\); \(M\): STFT magnitude | |
| `_verify_energy_conservation` | `energy_time = sum(abs(y)**2)` | \(E_t=\sum_n |y[n]|^2\) | | |
| `_verify_energy_conservation` | `energy_freq = sum(abs(S)**2)` | \(E_f=\sum_{k,t}|S_{k,t}|^2\) | | Raw STFT energy sum |
| `_verify_energy_conservation` | `window_power = sum(w**2)` | \(P_w=\sum_n w[n]^2\) | | |
| `_verify_energy_conservation` | `overlap_factor = window_length / hop_length` | \(O=N/h\) | \(h\): hop | |
| `_verify_energy_conservation` | `energy_freq_norm = (dc + nyq + 2*other) / (window_power * overlap_factor)` | One-sided Parseval-style normalisation per code | DC/Nyquist terms as implemented | |
| `_verify_energy_conservation` | `energy_ratio = energy_freq_norm / energy_time` | \(R=E_{\mathrm{fn}}/E_t\) | | |
| `_calculate_edge_frame_weights` | `real_signal_samples / n_fft` then `correction = 1 / max(portion, 0.5)` | \(c=\min(2,\,1/\max(p,0.5))\) | \(p\): inferred real-signal portion of frame | First/last edge loops |
| `_estimate_f0_global_robust` | `n_assignments = round(detected_freqs / initial_f0)` | \(n_i=\mathrm{round}(f_i/f_0^{(\mathrm{init})})\) | Clipped to \([1,N_{\max}]\) | |
| `_estimate_f0_global_robust` | `weights = (A/A_max)**2` | \(w_i\propto A_i^2\) | | |
| `_estimate_f0_global_robust` | `numerator / denominator` | \(f_0=\sum_i w_i n_i f_i\,/\,\sum_i w_i n_i^2\) | | Weighted LS closed form |
| `_estimate_f0_global_robust` | `residuals = detected_freqs - n_assignments * f0_robust` | \(\varepsilon_i=f_i-n_i f_0\) | | |
| `_estimate_f0_global_robust` | `sqrt(weighted_sse / weight_sum)` | \(\sigma=\sqrt{\sum_i w_i\varepsilon_i^2/\sum_i w_i}\) | | `residual_std` |
| `_calculate_metrics` (site) | `h_energy = sum(square(harmonic_amps))` | \(E_H=\sum_i A_{H,i}^2\) | | Harmonic energy |
| `_calculate_metrics` (site) | `tot_energy = h_energy + ih_energy + sub_energy` | \(E_{\mathrm{tot}}=E_H+E_I+E_S\) | | |
| `_calculate_metrics` (site) | ratios if `tot_energy > 1e-30` | \(r_H=E_H/E_{\mathrm{tot}}\) (and \(I,S\)) | | Mirrors `_set_model_weights` inputs |
| `_set_model_weights_from_current_component_energy` | `T = Hn + In + Sn` then `comp_h = Hn / T` | \(w_H=H/(H+I+S)\) | | Component ratios |
| `_set_model_weights_from_current_component_energy` | `model_h = Hn / HI` | \(m_H=H/(H+I)\) | | Binary model weights |
| export `_attach_raw_and_display` | `Power_raw = amps_raw ** 2` | \(P=A^2\) per row | | |
| export path | `self.linear_sum_amplitude_*` assignments | \(\Sigma_H A\), \(\Sigma_I A\), \(\Sigma_S A\) from summed column totals | | Integrated path |
