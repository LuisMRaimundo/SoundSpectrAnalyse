# Formula Validation Plan — Pass 10 — Selected proc_audio formulas

## 1. Scope

Pass 10: **Selected helpers in `proc_audio.py`** (RMS/gain, coherent gain, physical peak amplitude, Parseval audit, weighted \(f_0\) least squares), as in `FORMULA_EXTRACTION_TABLE_PASS_10_PROC_AUDIO_SELECTED_FORMULAS.md`.

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 10-01 | RMS | `y = np.ones(4)` | \(\mathrm{RMS}=1\), `cur_db=0` if \(\log_{10}(1)=0\) | `proc_audio._normalize_level(y, target_rms_db=0.0)` | output RMS **1** (gain 1) | |
| 10-02 | Gain | `y` with RMS **0.1** (`-20` dB), target **0** dB | \(G=10^{(0-(-20))/20}=10\), scaled max abs **1** | `_normalize_level` | `assert_allclose` max abs | |
| 10-03 | Coherent gain Hann | `win="hann"`, `n_fft=4096` | \(G=(\sum w)/N\) — compare to independent NumPy `hanning` sum / N | `proc_audio._coherent_gain` | `assert_allclose`, `atol=1e-10` | |
| 10-04 | `physical_peak_amplitude` | `mag=np.array([1.0])`, Hann, `n_fft` matching `_window_sum` | **\(2\cdot 1/S_w\)** one-sided | `proc_audio.physical_peak_amplitude` | `assert_allclose` | |
| 10-05 | Energy ratio (Parseval helper) | Short synthetic `y` + `S` from `librosa.stft` on same `y` | `energy_ratio` near **1.0** within doc tolerance | `proc_audio._verify_energy_conservation` | `assert_allclose(..., 1.0, atol=0.1)` or stricter if stable | |
| 10-06 | Weighted \(f_0\) LS | `detected_freqs=[200,400]`, `amps=[1,1]`, `initial_f0=100`, `max_n=10` | **\(f_0=100\)** (exact fit) | `proc_audio._estimate_f0_global_robust` | `assert_allclose` on `f0_estimated` | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
