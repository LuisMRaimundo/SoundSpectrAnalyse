# Formula Validation Plan — Pass 13 — Peak Detection and f0 Refinement

## 1. Scope

This plan defines small, hand-checkable numerical fixtures for formulas documented in `docs/formula_extraction/FORMULA_EXTRACTION_TABLE_PASS_13_PEAK_DETECTION_AND_F0_REFINEMENT.md` (Pass 13): project-owned **`proc_audio.py`** helpers listed below. It is intended for later `pytest` implementation; **no tests are created here**. Imports under test should call the named functions / `AudioProcessor` methods **directly** — **no** full STFT / `proc_audio` audio pipeline.

## 2. Included validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| PP-1 | `_parabolic_peak` three-point vertex | `y = [1.,2.,1.,1.]`, `x=1` | \(\alpha=1,\beta=2,\gamma=1\Rightarrow p=0\), \(x_{\mathrm v}=1\), \(y_{\mathrm v}=2\) | `proc_audio._parabolic_peak(y, 1)` | `assert_allclose([xv,yv],[1.,2.], rtol=0, atol=1e-15)` | Symmetric maximum at centre bin. |
| PP-2 | `_parabolic_peak` non-zero offset | `y=[2.,3.,0.]`, `x=1` | \(\alpha=2,\beta=3,\gamma=0\Rightarrow \mathrm{denom}=-4\), \(p=-0.25\), \(x_{\mathrm v}=0.75\), \(y_{\mathrm v}=3.125\) | `_parabolic_peak(y,1)` | `assert_allclose(xv,0.75)`; `assert_allclose(yv,3.125)` | Hand: \(p=\tfrac12(\alpha-\gamma)/(\alpha-2\beta+\gamma)\), \(y_{\mathrm v}=\beta-\tfrac14(\alpha-\gamma)p\). |
| PI-1 | `_parabolic_interpolation_log_magnitude` flat log-peak | `magnitudes=np.ones(5)`, `peak_idx=2`, `bin_spacing=10.0`, `freq_base=1000.0` | \(Y_1=Y_2=Y_3=0\Rightarrow a=b=0\Rightarrow x_{\mathrm{peak}}=0\) (degenerate branch); \(f_{\mathrm{corr}}=1000+2\cdot10=1020\); `valid` is `True` | `_parabolic_interpolation_log_magnitude(mags,2,10.,1000.)` | `assert_allclose(freq,1020.)`; `assert valid is True` | Bin-centre frequency when log-mags are flat at the triplet. |
| PI-2 | `_parabolic_interpolation_log_magnitude` interior offset | Choose linear mags `[0.1,1.0,0.1]` at `peak_idx=1`, spacing `1.0`, `freq_base=0.0` | \(Y=[-20,0,-20]\) dB \(\Rightarrow a=-20\), \(b=0\), \(x_{\mathrm{peak}}=0\); \(f_{\mathrm{corr}}=1.0\) | Same function | `assert_allclose(freq,1.)`; `assert valid` | Keeps \(\|x_{\mathrm{peak}}\|\le\tfrac12\). |
| RI-1 | `_refine_peak_index` window argmax | `magnitudes=np.array([0.,1.,0.,5.,4.])`, `approx_idx=2`, `refine_radius=2` | Window indices `0..4`, max at index `3` | `_refine_peak_index(mags,2,refine_radius=2)` | `assert out == 3` | No ties. |
| IB-1 | `_infer_bin_spacing_from_freqs` uniform grid | `freqs=np.array([100.,110.,120.])` | Positive gaps `{10,10}\)`, median `10` | `_infer_bin_spacing_from_freqs(freqs)` | `assert_allclose(out,10.)` | |
| IB-2 | `_infer_bin_spacing_from_freqs` odd median | `freqs=np.array([0.,10.,25.,40.])` | Positive gaps `10,15,15` → median `15` | `_infer_bin_spacing_from_freqs(freqs)` | `assert_allclose(out,15.)` | |
| RC-1 | `_refine_candidate_to_interpolated_peak` keys + nearest snap | Uniform `freqs=np.linspace(440.,460.,11)` (step `2` Hz), `mags` small positive floor `1e-6` everywhere except index `5` value `2.0`, `candidate_freq_hz=451.0`, `refine_radius=2` | Nearest bin to `451` is `450` (index `5`); `peak_bin_index==5`; `bin_center_frequency_hz==450`; `peak_magnitude_db==20*log10(2)`; with symmetric linear magnitudes around the peak, `interpolated_frequency_hz` matches `bin_center_frequency_hz` and `subbin_interpolation_valid` is `False` if \(\|a\|<10^{-10}\) on the log triplet | `_refine_candidate_to_interpolated_peak(...)` | `assert out["peak_bin_index"]==5`; `assert_allclose(out["bin_center_frequency_hz"],450.)`; `assert_allclose(out["peak_magnitude_db"],20*np.log10(2))`; optional `assert_allclose(out["interpolated_frequency_hz"],out["bin_center_frequency_hz"])` | Avoid exact `0` linear mags (log branch); tie-break `argmin` picks the lower-frequency bin at equal distance (`450` before `452`). |
| SP-1 | `_saddle_prominence_db` symmetric shoulders | `magnitudes`: length `11`, index `5` value `100`, others `1`, `peak_idx=5`, `saddle_window=3` | Flank minima linear `1` → \(L_p=40\) dB, \(L_\ell=L_r=0\) dB → prominence `40` dB | `_saddle_prominence_db(mags,5,saddle_window=3)` | `assert_allclose(out,40., rtol=0, atol=1e-9)` | \(20\log_{10}100-20\log_{10}1\). |
| SP-2 | `_saddle_prominence_db` edge guard | `magnitudes=np.ones(3)`, `peak_idx=0` | Code returns `-inf` | `_saddle_prominence_db(mags,0)` | `assert out == float("-inf")` | |
| LV-1 | `_is_local_peak_valid` synthetic pass | `magnitudes`: length `21`, index `10` value `100`, all other entries `0.01`, defaults `threshold_db=3`, `noise_floor_percentile=15`, `window_size=5`, `saddle_window=3` | After refinement, strict dB local max at `10`; saddle prominence \(\gg 3\) dB; noise floor \(\approx 0.01\) → SNR \(\gg 3\) dB → `(True, snr_db)` with large positive `snr_db` | `_is_local_peak_valid(mags,10,...)` | `assert is_valid is True`; `assert snr_db > 3` | Percentile on almost-constant floor is hand-stable; **defer** optimality of `15` / `3` dB policy. |
| LM-1 | `_local_peak_metrics` same fixture as LV-1 | Same `magnitudes`, `peak_idx=10`, default kwargs | `local_peak_valid is True`; `snr_db` finite and \(\gg 0\); `prominence_db` matches `_saddle_prominence_db` on same inputs | `_local_peak_metrics(mags,10)` | `assert is_lp`; `assert_allclose(prom, _saddle_prominence_db(mags,10,saddle_window=10), rtol=1e-12)` | Cross-check prominence delegate. |
| CF-1 | `_correct_f0_candidate_against_prior` octave drop | `candidate_hz=880.0`, `prior_hz=440.0`, defaults | Best is \(440\) Hz with ratio `0.5`, cents error `0` | `_correct_f0_candidate_against_prior(880.,440.)` | `assert out["valid"]`; `assert_allclose(out["corrected_hz"],440.)`; `assert_allclose(out["cents_error"],0., atol=1e-9)` | |
| CF-2 | `_correct_f0_candidate_against_prior` identity | `candidate_hz=261.6255653005988` (scientific C4), `prior_hz` same | Corrected = prior, ratio `1`, cents `0` | Same | `assert out["corrected_hz"]==prior` (or `assert_allclose`) | |
| BS-1 | `_calculate_bin_spacing` | `sr=48000`, `n_fft=4096`, `zero_padding=2` | \(\Delta f=48000/8192=375/64\) Hz | `_calculate_bin_spacing(48000.,4096,2)` | `assert_allclose(out,48000./(4096*2), rtol=1e-15)` | |
| FN-1 | `frequency_to_note_name` A4 | `freq_hz=440.0`, `a4=440.0` | `midi=69`; cents `0`; string contains `A4` and `+0.00` or `0.00` cents | `proc_audio.frequency_to_note_name(440.,440.)` | `assert "A4" in s`; `assert "+0.00" in s or "-0.00" in s` (or parse cents substring) | **Defer** half-semitone `round` semantics. |
| FN-2 | `frequency_to_note_name` one semitone up | `f=440*2**(1/12)`, `a4=440` | Continuous MIDI \(69+1=70\); cents vs rounded ref \(\approx +0\) to tight tolerance | `frequency_to_note_name(f,440.)` | `assert "A#4" in s or "Bb4" in s` per sharp spelling in code (`_NOTE_NAMES_SHARP`) → expect `A#4` | Code uses sharp names. |
| CF0-1 | `AudioProcessor.calculate_fundamental_frequency` numeric Hz | `proc = AudioProcessor()` (no-args constructor used in existing tests) | Input `"440"` | Returns `440.0` | `assert_allclose(proc.calculate_fundamental_frequency("440"),440.)` | Numeric string fast path. |
| CF0-2 | `AudioProcessor.calculate_fundamental_frequency` note A4 | Input `"A4"` | \(440\) Hz | `calculate_fundamental_frequency("A4")` | `assert_allclose(out,440., rtol=1e-9)` | Uses internal `freq_C0` chain; cross-check \(440\cdot2^{(69-69)/12}\). |
| CF0-3 | `AudioProcessor.calculate_fundamental_frequency` C4 | Input `"C4"` | \(f=f_{C_0}2^{48/12}=f_{C_0}\cdot16\) with \(f_{C_0}=440\cdot2^{-4.75}\) | Same | `assert_allclose(out,440.*2**((60-69)/12.), rtol=1e-9)` | MIDI C4 = 60. |

## 3. Deferred / human-review cases

Do **not** encode these as settled scientific requirements in automated tests:

- Whether **`np.argmax` tie-breaking** (first maximum) is acoustically or statistically preferable.
- **`float.__round__` / Python `round`** behaviour at exact half-semitone boundaries for `frequency_to_note_name`.
- Whether defaults **`saddle_window`**, **`refine_radius`**, **noise-floor percentile rank**, or the **hard-coded 3 dB SNR gate** are optimal for all corpora.
- **Full audio-pipeline** validation (STFT, tracking, export) — out of scope for Pass 13 unit fixtures.

## 4. Implementation status

No tests are created by this document. This is a validation plan only.
