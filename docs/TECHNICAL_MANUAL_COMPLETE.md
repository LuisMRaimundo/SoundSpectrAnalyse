# SoundSpectrAnalyse — Complete Technical Manual

## 0. Scope and epistemic status

This manual documents the current implementation in `SoundSpectrAnalyse` as code exists today.

- The software analyses individual note recordings.
- It is not restricted to clarinet.
- Clarinet sustain corpus behavior is used as a control/validation context for approximately harmonic sustained wind tones, not as a universal reference target.
- Instrument-family context is mandatory for interpretation.
- High inharmonicity can be physically valid for piano, harp, stiff/plucked strings, idiophones, and extended techniques.
- High residual content can be physically valid for transient/noisy or percussive sounds.
- High `obs_wS` is only an artifact interpretation when low-frequency energy evidence is negligible and artifact criteria are met.

Epistemic distinction used throughout this manual:

1. measured or computed spectrum quantities;
2. modelled density quantities;
3. per-note energy ratios;
4. corpus-level weighting profiles;
5. diagnostics and warnings;
6. GUI labels and convenience mappings.

The H/I/S decomposition is an operational computational model:

- `H`: harmonic-aligned content;
- `I`: inharmonic/residual content;
- `S`: low-frequency/subfundamental residual diagnostic content.

It is not a universal ontology of acoustic sources.

---

## 1. Pipeline overview

Primary runtime path:

1. Audio loading and conditioning in `proc_audio.py`.
2. STFT/magnitude extraction and smoothing options.
3. Tier-dependent STFT policy (or fixed-mode STFT if enabled).
4. Peak candidate selection and local-peak filtering.
5. Nominal-guided and robust/global F0 estimation logic.
6. Harmonic/inharmonic/subbass candidate partitioning.
7. Subbass boundary via `SubBassPolicy.upper_bound_hz(...)`.
8. Inharmonicity fit (`B`) using stiff-string approximation.
9. Acoustic density descriptors and salience-based component metrics.
10. Phase 1 adaptive observation triplets (`pure_observation_w_*`) and online profile updates.
11. Phase 2 profile application during compilation.
12. Tier normalization of amplitude/power sums.
13. MIR/timbral descriptors extraction.
14. Optional temporal segmentation descriptors (attack/sustain/release).
15. Stage 2 compile/export to `compiled_density_metrics.xlsx`.
16. Research workbook post-export to `compiled_density_metrics_research.xlsx`.
17. Validation sheets, diagnostic flags, and metadata summaries.

Flow diagram:

```text
Audio file
  -> Stage 1 (proc_audio.AudioProcessor)
      -> STFT / peaks / F0 / H-I-S decomposition
      -> per-note workbook (spectral_analysis.xlsx)
      -> Phase 1 obs triplets + adaptive updates (optional runtime learning)
  -> Stage 2 (compile_metrics.compile_density_metrics_with_pca)
      -> direct per-note extraction
      -> density_metric_raw + profile application + tier normalization
      -> compiled_density_metrics.xlsx (multi-sheet)
  -> Post-process research export (tools/export_research_density_workbook.py)
      -> compiled_density_metrics_research.xlsx
```

---

## 2. Notation and units

- $x[n]$: discrete-time signal (unitless PCM amplitude).
- $f_s$: sampling rate (Hz).
- $N$: FFT length (`n_fft`).
- $H$: hop length (samples).
- $w[n]$: analysis window.
- $X(k,t)$: STFT coefficient.
- $A_{k,t} = |X(k,t)|$: magnitude.
- $P_{k,t} = |X(k,t)|^2$: power.
- $f_k = \frac{k f_s}{N}$: FFT-bin frequency.
- $f_0$: fundamental frequency estimate.
- $n$: harmonic index/order.
- $B$: inharmonicity coefficient in stiff-string model.
- $D_H,D_I,D_S$: component density sums.
- $w_H,w_I,w_S$: weights applied in weighted density sums.
- $r_H,r_I,r_S$: per-note component energy ratios.
- $\theta_{\mathrm{dB}}$: salience threshold in dB.
- $F_{\max}$: density ceiling in Hz.
- `S`: subbass residual component (not sustain).

Metric-type interpretation:

- linear amplitude sums: `*_amplitude_sum`;
- linear power/energy sums: `*_energy_sum`;
- log-amplitude-based densities: many `*_density_*` terms when `weight_function=log`;
- ratio metrics: occupancy, energy-ratio, completeness-like fractions;
- count metrics: harmonic orders, bins, particles;
- normalized metrics: tier-normalized, min-max normalized, or max-relative normalized.

---

## 3. STFT, windowing, and tier strategy

### STFT core equations

**Purpose in the software.**
Provide time-frequency representation for peak extraction and all downstream metrics.

**Code location.**
- Module: `proc_audio.py`
- Function(s): `fft_analysis(...)` and dependent peak-processing routines.
- GUI controls: window type, `n_fft`, hop length, zero padding, time averaging.
- Exported columns: `n_fft`, `n_fft_effective`, `hop_length`, `bin_spacing_hz`.
- Workbook sheets: per-note `Per_Note_Processing_Metadata`, compiled metadata sheets.

**Inputs.**
Audio vector, sampling rate, window type, `n_fft`, hop length, zero-padding mode.

**Mathematical definition.**

$$
X(k,t)=\sum_{n=0}^{N-1} x[n+tH]\,w[n]\,e^{-j2\pi kn/N}
$$

$$
A(k,t)=|X(k,t)|,\qquad P(k,t)=|X(k,t)|^2
$$

$$
f_k = \frac{k f_s}{N}
$$

**Algorithmic implementation.**
The code computes STFT-derived magnitudes/powers, optionally applies smoothing, then uses local-peak and harmonic classification logic to build analysis tables.

**Interpretation.**
Higher magnitude/power values indicate stronger spectral components in the selected frame aggregation context.

**Limitations and non-claims.**
This does not by itself represent physical source separation; it is a transform-domain representation sensitive to window and FFT choices.

**Relevant bibliography or provenance.**
implementation-defined; requires sensitivity analysis or bibliographic justification (for defaults not explicitly sourced).

**Edge cases.**
Empty/invalid arrays fall back to safe defaults or NaN outputs depending on stage.

**Example.**
For `n_fft=4096`, `f_s=44100`, bin spacing is $\approx 10.7666$ Hz.

### Tier strategy and normalization

Tier strategy chooses per-note effective FFT settings (when enabled) and later provides cross-note normalization via `spectral_normalization.py`:

$$
\alpha_{\mathrm{peak\_amp}}(N)=\frac{N_{\mathrm{ref}}}{N},\qquad
\alpha_{\mathrm{peak\_pow}}(N)=\left(\frac{N_{\mathrm{ref}}}{N}\right)^2.
$$

where default $N_{\mathrm{ref}}=8192$.

Backward-compatibility branches for broadband L2 quantities remain available in code:

$$
\alpha_{\mathrm{broadband\_amp\_l2}}(N)=\sqrt{\frac{N_{\mathrm{ref}}}{N}},\qquad
\alpha_{\mathrm{broadband\_pow\_l2}}(N)=\frac{N_{\mathrm{ref}}}{N}.
$$

---

## 4. F0 estimation and note parsing

### Canonical note token parsing

**Purpose in the software.**
Obtain robust note labels from manifests/filenames/folders without accidental token corruption.

**Code location.**
- Module: `note_parser.py`
- Function(s): `parse_note_token`, `canonical_note_from_filename`.
- Exported column(s): `Note`, `note_source` in compile path.

**Inputs.**
Manifest note, filename, parent-folder name.

**Mathematical definition.**
Regex-based parsing with normalization, not a floating-point formula.

**Algorithmic implementation.**
Priority order: `manifest` -> `filename_token` -> `parent_folder` -> `fallback_no_octave` -> `unknown`.

**Interpretation.**
`fallback_no_octave` is diagnostic and should not drive frequency conversion blindly.

**Limitations and non-claims.**
Does not infer missing octave.

**Relevant bibliography or provenance.**
implementation-defined; requires sensitivity analysis or bibliographic justification.

**Edge cases.**
Malformed tokens return `unknown`.

**Example.**
`Bn-ord-A#1-pp-N-N_Sustains.wav` -> `A#1`, source `filename_token`.

### F0 provenance and cents deviation

**Purpose in the software.**
Track whether final F0 is acoustically accepted or nominal fallback.

**Code location.**
- Module: `acoustic_density_core.py`
- Function(s): `canonical_f0_triplet`.
- Exported columns: `f0_final_hz`, `f0_source`, `f0_final_source`, `acoustic_f0_status`, `f0_fit_accepted`, `f0_deviation_cents`.

**Mathematical definition.**

$$
\Delta c = 1200\log_2\left(\frac{f_{\mathrm{candidate}}}{f_{\mathrm{nominal}}}\right)
$$

**Algorithmic implementation.**
Use fitted final F0 only when accepted; otherwise explicit nominal-guided fallback status.

**Limitations and non-claims.**
Nominal fallback is not acoustic verification.

---

## 5. Harmonic, inharmonic, and subbass decomposition

### 5.1 Harmonic candidate prediction

$$
f_n = n f_0
$$

and, when model-applied in fit context:

$$
f_n = n f_0\sqrt{1+B n^2}
$$

### 5.2 Harmonic tolerance

Adaptive tolerance policy is documented in constants (`ADAPTIVE_HARMONIC_TOLERANCE_POLICY_DOC`):

$$
\tau_n = \max\left(\tau_{\mathrm{cents}}, 1200 \cdot \frac{\Delta f_{\mathrm{bin}}}{n f_0}\right)
$$

### 5.3 Inharmonic/residual classification

Residual assignment occurs after harmonic matching and candidate filters; counts and occupancy diagnostics are exported.

### 5.4 Per-order candidate classification and density inclusion

**Code location.** `harmonic_peak_validation.py`
(`_local_peak_metrics`, `cfar_peak_detection`, `_classify_harmonic_candidate`,
`_saddle_prominence_db`, `_prominence_saddle_window_bins`), re-exported by
`proc_audio` and driven from `proc_audio._generate_harmonic_list`.

For each expected order, the candidate nearest to `n·f0` (within the tolerance
window) is refined to the local spectral peak and classified. A candidate is
promoted to `strict_validated` (and only then `include_for_density = True`)
when it is **CFAR-detected AND clears the saddle-prominence criterion**. The
noise-significance gate is a cell-averaging **CFAR** (constant false-alarm-rate)
test (`cfar_peak_detection`): the peak power must exceed an adaptive threshold
derived from a stated false-alarm probability (`Pfa`, default `1e-2`) against a
locally-estimated, peak-trimmed noise floor — replacing the previous ad-hoc
fixed `SNR ≥ 3 dB` margin with a detection-theoretic, register- and
noise-adaptive criterion (the same significance-gate philosophy used for the
inharmonicity coefficient `B`; see §F3/F7 in `FORMULA_VALIDATION_STATUS.md`).
`cfar_margin_db = 10·log10(peak_power / threshold)` and `cfar_detected` are
exported in the audit. Prominence is measured against a saddle window scaled to
the inter-harmonic half-spacing (±f0/2), not a fixed bin count — a fixed window
collapses prominence on low-pitched, densely packed spectra (e.g. cello C2) and
was the cause of severe harmonic under-counting in the low register. The ±1-bin
"local maximum" flag is reported (`local_peak_valid`) but is **not** a strict
gate, because on windowed FFTs it measures main-lobe curvature rather than
partial validity.

Candidate-status taxonomy (`candidate_status`): `strict_validated`,
`snr_validated`, `weak_candidate`, `below_noise_floor`, `missing_window`,
`rejected_bad_f0`, `off_frequency`. Candidates are re-aligned to the fitted f0
before final classification so detuned partials are not mislabelled
`off_frequency`. Every order's decision (and its reason) is exported, read-only,
to the per-note `Harmonic_Inclusion_Audit` sheet (see §14).

### 5.5 Subbass policy

**Code location.**
- Module: `subbass_policy.py`
- Function: `SubBassPolicy.upper_bound_hz`.
- Exported column(s): `subbass_upper_bound_hz` (diagnostic path), subfundamental-policy columns.

**Mathematical definition.**

$$
f_{\mathrm{sub,max}} = \min\left(0.5f_0,\;80\right)
$$

**Interpretation.**
Operational low-frequency residual boundary, not automatic physical subbass claim.

**Limitations and non-claims.**
High `obs_wS` can exist with negligible subbass energy; see artifact flags.

---

## 6. Inharmonicity model

### Inharmonicity coefficient fit

**Purpose in the software.**
Quantify stretch relative to ideal harmonic series.

**Code location.**
- Module: `inharmonicity_model.py`
- Function(s): `fit_inharmonicity_coefficient`.
- Export columns: `inharmonicity_coefficient_B`, `inharmonicity_fit_status`, `inharmonicity_fit_residual_std_cents`, `inharmonicity_fit_method`, `inharmonicity_model_applied`, `inharmonicity_fit_source`, `inharmonicity_validation_warning`.

**Inputs.**
Candidate peak frequencies, `f0_hz`, order cap, cents window.

**Mathematical definition.**

$$
f_n = n f_0\sqrt{1+B n^2}
$$

Least-squares relation used in code:

$$
y_n = \left(\frac{f_n}{n f_0}\right)^2 - 1,\quad y_n \approx B n^2.
$$

**Algorithmic implementation.**
Match near-harmonic candidates per order, estimate $B$, compute residual spread in cents, assign `ok` or rejection status.

**Interpretation.**
Low $B$ is typical for many sustained winds; non-zero/higher $B$ can be physically correct in stiff/plucked systems.

**Limitations and non-claims.**
Not an instrument classifier; context required.

**Edge cases.**
Insufficient matched partials -> `insufficient_partials`.

---

## 7. Density metrics

### 7.1 Harmonic density component

**Purpose in the software.**
Summarize harmonic-band contribution under selected weight function.

**Code location.**
- Module: `compile_metrics.py`, `density.py`, `acoustic_density_core.py`
- Functions: extraction and weighted composition pipeline.
- Export columns: `harmonic_density_sum`, `harmonic_density_component`, `harmonic_amplitude_sum`, `harmonic_energy_sum`, normalized variants.

**Mathematical definition.**
General weighted sum form:

$$
D_H = \sum_{i\in H} \phi(A_i)
$$

where $\phi$ is selected by weight function (linear/log/etc.).

**Algorithmic implementation.**
Per-note extraction from harmonic sheets with canonical source prioritization and inclusion policy flags.

**Interpretation.**
Higher values indicate stronger or denser harmonic occupancy/mass under chosen $\phi$.

**Limitations and non-claims.**
Not direct loudness; depends on weighting function and thresholds.

### 7.2 Inharmonic density component

Same structure for inharmonic/residual population:

$$
D_I = \sum_{i\in I} \phi(A_i)
$$

Export: `inharmonic_density_sum`, `inharmonic_density_component`.

### 7.3 Subbass density component

Same structure over subbass residual band:

$$
D_S = \sum_{i\in S} \phi(A_i)
$$

Export: `subbass_density_sum`, `subbass_density_component`, `subbass_energy_ratio`, diagnostic artifact fields.

### 7.4 Raw density metric (canonical compiled metric)

If Phase 2 profile applied:

$$
D_{\mathrm{raw}} = w_H^{(c)} D_H + w_I^{(c)} D_I + w_S^{(c)} D_S
$$

with $w_H^{(c)}+w_I^{(c)}+w_S^{(c)}=1$.

If not, per-note energy-ratio weights are used.

### 7.5 Per-note balance density

$$
D_{\mathrm{per-note}} = r_H D_H + r_I D_I + r_S D_S
$$

$$
r_H=\frac{E_H}{E_H+E_I+E_S},\quad
r_I=\frac{E_I}{E_H+E_I+E_S},\quad
r_S=\frac{E_S}{E_H+E_I+E_S}.
$$

Export: `density_metric_raw_per_note_balance`.

### 7.6 Weight source flags

- `density_weights_source`: `phase2_corpus_profile` or `per_note_energy_ratio`.
- `density_metric_raw` remains canonical compile output.

### 7.7 Principled per-note scalar density (`note_density_final`)

$$
D_{\mathrm{final}} = r_H D_H + r_I D_I + r_S D_S
$$

where $r_H, r_I, r_S$ are the per-note **measured** component energy ratios
(`component_*_energy_ratio`, from `Component_Balance`, summing to 1) and
$D_H, D_I, D_S$ are the per-band density sums (`*_density_sum`) under the active
GUI amplitude weight function.

**Code location.** `compile_metrics._compute_note_density_final` (compiled
`Density_Metrics`); mirrored in
`tools/export_research_density_workbook.build_spectral_density_metrics`
(research `Spectral_Density_Metrics`, highlighted light blue).

**Properties.**
- Combines the GUI weight function (already inside each $D_*$) with the per-note
  **physical** component balance — it does **not** use the Bayesian adaptive
  weights.
- Absolute (not corpus-normalized). Input audio is RMS-referenced, so it
  describes spectral shape at a reference level; cross-instrument comparison is
  valid only under an identical analysis profile.
- NaN-propagating: if any of the six inputs is NaN, the result is NaN.
- Algebraically identical in form to §7.5 but sourced explicitly from the
  canonical `*_density_sum` + `component_*_energy_ratio` columns and exported as
  a distinct, clearly-named primary column. It does not modify
  `density_metric_raw`, `density_metric_normalized`, or
  `final_note_density_salience_weighted`.

**Uncertainty quantification.** Each note carries a transform-aware,
non-parametric bootstrap confidence interval for `note_density_final`
(`density_uncertainty.bootstrap_note_density_final`): per-partial contributions
are resampled within each band **and** the component energy ratios are
recomputed inside each resample from the bootstrapped band energies
(`propagate_ratio_uncertainty=True`), so both the band-sum and the ratio
uncertainty are propagated jointly. Exported columns:
`note_density_final_ci_low`, `note_density_final_ci_high`,
`note_density_final_rel_uncertainty` (std/|point|), and
`note_density_final_uncertainty_sources` (`partials+ratios`). The window/n_fft
sensitivity component is an **opt-in** study tool
(`tools/note_density_nfft_sensitivity.py`, built on
`density_uncertainty.nfft_sensitivity`), kept out of the hot path because
re-analysis at multiple resolutions multiplies per-note runtime.

Export: `note_density_final` (+ the four uncertainty columns above).

---

## 8. Adaptive Phase 1 and Phase 2

### 8.1 Pure observation triplet

For component strengths $s_H,s_I,s_S$:

$$
o_H=\frac{s_H}{s_H+s_I+s_S},\quad
o_I=\frac{s_I}{s_H+s_I+s_S},\quad
o_S=\frac{s_S}{s_H+s_I+s_S}.
$$

Export: `pure_observation_w_h`, `pure_observation_w_i`, `pure_observation_w_s`.

**Energy-consistency gate (`obs_w_formula_version = v58_full_spectrum_region_energy_gate`).**
Each structural strength is weighted by its band's energy presence before
forming the triplet: $s_x \leftarrow s_x \cdot g_x$, where the gate
$g_x = E_x / (E_H+E_I+E_S)$ uses the **full-spectrum, total-power-normalised
region energy triple** — harmonic-peak $E_H$, **non-harmonic residual** $E_I$,
sub-bass $E_S$ (`harmonic_energy_ratio` / `residual_energy_ratio` /
`subbass_energy_ratio`). These three powers partition every spectral bin, so the
gate conserves energy and is instrument-agnostic. A band with ~0 energy
contributes ~0 (this is what suppresses the empty sub-bass band). Audit:
`component_strength_energy_gate_{harmonic,non_harmonic_residual,subbass}`,
`density_band_energy_basis`.

**Terminology.** The middle (non-harmonic) band is the inter-harmonic RESIDUAL
(broadband bow/breath/attack noise plus any non-`n·f0` content). It is *not* the
same as partial INHARMONICITY (piano/bell stretch), which is reported separately
as the inharmonicity coefficient $B$ and the inharmonic-peak energy, and is *not*
part of this density gate. v57 (`v57_energy_anchored_occupancy`) used
body-ceiling-truncated energies and is superseded because that truncation made
the non-harmonic share instrument-dependent.

### 8.2 Adaptive engine

**Code location.**
- Module: `adaptive_density_engine.py`
- Class: `AdaptiveDensityEngine`.

Dirichlet-style concentration update with forgetting and JS-divergence reliability gate:

- prior mean from concentration vector $\alpha$;
- reliability $\propto \exp(-\mathrm{JSD}/T)$, clipped;
- gain from evidence strength;
- update:
  $$
  \alpha \leftarrow (1-\lambda)\alpha + g\cdot o.
  $$

Outputs: profile, confidence, uncertainty, reliability, JS divergence.

### 8.3 Phase 2 corpus profile

Phase 2 exports and applies:

- `phase2_harmonic_weight`
- `phase2_inharmonic_weight`
- `phase2_subbass_weight`
- confidence metadata.

These are corpus-level model-density weights, not per-note physical energy fractions.

---

## 9. Tier normalization

**Code location.**
- Module: `spectral_normalization.py`
- Function: `n_fft_normalization_factor`.

Formulas (`n_fft_normalization_factor(..., quantity_kind=...)`):

$$
\alpha_{\mathrm{peak\_amp}}(N)=\frac{N_{\mathrm{ref}}}{N},
\qquad
\alpha_{\mathrm{peak\_pow}}(N)=\left(\frac{N_{\mathrm{ref}}}{N}\right)^2.
$$

Backward-compatible broadband-L2 branches:

$$
\alpha_{\mathrm{broadband\_amp\_l2}}(N)=\sqrt{\frac{N_{\mathrm{ref}}}{N}},
\qquad
\alpha_{\mathrm{broadband\_pow\_l2}}(N)=\frac{N_{\mathrm{ref}}}{N}.
$$

Applied columns include:

- `harmonic_amplitude_sum_tier_normalized`
- `inharmonic_amplitude_sum_tier_normalized`
- `subbass_amplitude_sum_tier_normalized`
- `harmonic_energy_sum_tier_normalized`
- `inharmonic_energy_sum_tier_normalized`
- `subbass_energy_sum_tier_normalized`
- `tier_consistency_status`.

---

## 10. Spectral entropy and distributional descriptors

### Normalized spectral entropy

**Code location.**
- Module: `acoustic_density_core.py`
- Function: `_normalized_entropy`.
- Export: `spectral_entropy`.

Definition:

$$
p_i=\frac{P_i}{\sum_j P_j},\quad
H=-\sum_i p_i\log_2 p_i,\quad
H_{\mathrm{norm}}=\frac{H}{\log_2 K}.
$$

Returned value clipped to $[0,1]$.

---

## 11. Dissonance and roughness

Project includes dissonance/roughness infrastructure (`dissonance_models.py`, `dissonance_export.py`, `sethares_*` status fields).

### Roughness approximation in MIR module

`mir_descriptors.py` computes `roughness_aures_1985` using a pairwise interaction shape:

$$
x=\frac{|f_i-f_j|}{0.25\min(f_i,f_j)+24.7},\quad
g(x)=x\,e^{1-x},
$$

and sums amplitude-product weighted interactions.

Interpretation should be described as implementation-level roughness proxy unless full model assumptions are validated for the use case.

---

## 12. MIR and timbral descriptors

### Spectral centroid

$$
C=\frac{\sum_i f_i p_i}{\sum_i p_i}
$$

with $p_i$ from normalized power in current implementation.

### Spectral spread

$$
\sigma=\sqrt{\sum_i (f_i-C)^2p_i}
$$

### Spectral skewness and kurtosis

$$
\mathrm{skew}=\sum_i\left(\frac{f_i-C}{\sigma}\right)^3p_i,\quad
\mathrm{kurt}=\sum_i\left(\frac{f_i-C}{\sigma}\right)^4p_i.
$$

### Spectral flatness

$$
F=\frac{\exp\left(\frac{1}{K}\sum_i\ln P_i\right)}
{\frac{1}{K}\sum_i P_i}.
$$

### Spectral rolloff

Find $R_p$ such that:

$$
\sum_{f_i\le R_p}P_i = p\sum_i P_i,\quad p\in\{0.85,0.95\}.
$$

### Spectral irregularity

Adjacent-amplitude variation:

$$
I=\frac{\sum_i |A_{i+1}-A_i|}{\sum_i A_i}
$$

clipped to $[0,1]$ in implementation.

### Tristimulus

$$
T_1=\frac{A_1}{\sum_i A_i},\quad
T_2=\frac{A_2+A_3+A_4}{\sum_i A_i},\quad
T_3=\frac{\sum_{i\ge5}A_i}{\sum_i A_i}.
$$

### ERB-weighted spectral density

ERB-rate transform:

$$
\mathrm{ERB}(f)=21.4\log_{10}(1+0.00437 f).
$$

Grouped power-mass by integer ERB bins, then effective density:

$$
D_{\mathrm{ERB}}=\frac{1}{\sum_b q_b^2},
\quad q_b=\frac{m_b}{\sum_j m_j}.
$$

### MIR availability transparency

Compiled export now includes:

- `mir_descriptors_available`
- `mir_descriptors_source`
- `mir_descriptors_missing_reason`.

---

## 13. Temporal segmentation

**Code location.**
- Module: `temporal_segmentation.py`
- Function: `segment_attack_sustain_release`.

Envelope smoothing then threshold-based segmentation:

- attack onset at $0.1$ of peak envelope;
- attack end at $0.9$ of peak;
- release starts near $0.2$ of peak after sustain.

Log attack time:

$$
\mathrm{LAT}=\log_{10}(t_{\mathrm{attack,end}}-t_{\mathrm{attack,start}})
$$

Export family includes segmented descriptor suffixes:

- `*_on_attack`
- `*_on_sustain`
- `*_on_release`
- and `*_on_sustain_segment`.

---

## 14. Compilation and Excel exports

### Primary compiled workbook

`compiled_density_metrics.xlsx` major sheets:

- `Density_Metrics` (includes `density_metric_raw`, `density_metric_normalized`, `density_metric_raw_per_note_balance`, and `note_density_final` — see §7.7)
- `Canonical_Metrics`
- `Canonical_Primary_Filtered`
- `Diagnostic_Metrics`
- `Legacy_Compatibility`
- `Legacy_Aliases`
- `Debug_Counts`
- `Validation_Metrics`
- `Validation_Summary`
- `Per_Note_Processing_Metadata`
- `Analysis_Metadata`
- optional PCA and dissonance sheets.

### Research workbook

`compiled_density_metrics_research.xlsx` major sheets:

- `Spectral_Density_Metrics` (includes `note_density_final` — see §7.7)
- `Primary_Statistics_Filtered`
- `Component_Balance`
- `Validation_Summary`
- `Charts_Data`
- `Legacy_Compatibility`
- `Analysis_Settings_By_Note`
- `Metadata`
- `Dashboard`, `README`.

### Per-note workbook (`spectral_analysis.xlsx`)

Beyond the spectral sheets (`Harmonic Spectrum`, `Strict_Harmonic_Peaks`,
`Inharmonic Spectrum`, `Sub-bass band`, `Metrics`, `Analysis_Metadata`), each
per-note workbook carries a read-only **`Harmonic_Inclusion_Audit`** sheet: one
row per harmonic order exposing exactly why each candidate is included in or
excluded from the density computation (`exclusion_reason`, `snr_db`,
`prominence_db`, `cfar_margin_db`, `cfar_detected`, `local_peak_valid`,
`candidate_status`, `include_for_density`, `included_in_strict_peaks`,
`included_in_body_density_5khz`, deviation in Hz and cents, and the search/body
ceilings). The count of `included` rows equals
`harmonic_density_included_count` in `Analysis_Metadata`. See §5.4 and
`docs/DENSITY_EXPORT_SCHEMA.md` §2b.

---

## 15. GUI option tutorial (advanced user)

Primary operational GUI: `pipeline_orchestrator_gui.py` (Tk).  
Legacy/reference GUI: `interface.py` (PyQt) — archived to `Backup/root_modules/` (no longer in the active package).

Core controls (Tk):

- Density mode (`density_summation_mode`): adaptive/harmonic-only/inharmonic-only/subbass-only/HIS weighted modes.
- Component weights (`harmonic_density_weight`, `inharmonic_density_weight`, `subbass_density_weight`).
- Salience threshold (`density_salience_threshold_db`).
- Density ceiling (`density_frequency_ceiling_hz`).
- Frequency range (`freq_min`, `freq_max`).
- Harmonic tolerance (`tolerance`) + adaptive tolerance toggle.
- STFT controls: window, `n_fft`, hop, zero padding, averaging.
- Amplitude weighting function (`wf`) mapped by label. **Defaults to
  `Logarithmic` (`log`, the PRIMARY comparable profile)**, so an isolated single
  run is cross-instrument comparable by default; any other choice downgrades the
  run to `EXPLORATORY`.
- Dissonance model (`diss`).
- Compile/advanced toggles: auto-compile, t-SNE, UMAP, anomaly detection, contamination, manual model-weight override.

Comparability-critical controls (primary comparable profile):

- `wf=log`
- `density_salience_threshold_db` = runtime-configured (no hardcoded value)
- `density_frequency_ceiling_hz` = runtime-configured (no hardcoded value)

Per current policy the primary comparable profile is
`wf=log|dst=runtime_configured|ceil=runtime_configured`
(`primary_comparable_profile_definition`). Runs on other profiles are flagged
`EXPLORATORY` and must not be compared directly against primary-profile runs.

---

## 16. Instrument-family interpretation guide

- Sustained woodwinds: often low `B`, strong harmonic occupancy, lower residual occupancy.
- Brass: harmonic dominant but potentially stronger nonlinear/transient residuals.
- Bowed strings: potential structured inharmonicity; harmonic and residual both informative.
- Piano/harp/plucked strings: elevated `B` can be physically correct due to stiffness.
- Tuned percussion/idiophones: residual/inharmonic dominance can be expected.
- Voice/extended techniques: broad variability; avoid harmonic-only assumptions.
- Synthetic/electronic tones: interpretation depends on synthesis architecture; model outputs remain computational descriptors.

Do not interpret high `B` or high residual occupancy as automatic error without instrument and technique context.

---

## 17. Validation and warnings

Important flags include:

- comparability profile (`PRIMARY`/`EXPLORATORY`);
- `tier_consistency_status`;
- `inharmonicity_validation_warning`;
- `obs_wS_artifact_flag`, `obs_wS_artifact_reason`;
- MIR availability flags;
- f0 provenance flags (`acoustic_f0_status`, `f0_fit_accepted`, etc.);
- debug invariants and arithmetic/acoustic validation statuses;
- metadata inference and git-status reasons in research export.

Operational guidance:

- Missing fit status/residual with valid `B`: treat as partial export diagnostics.
- `obs_wS_artifact_flag=True`: treat subbass contribution as model residual, not physical subbass claim.
- Non-primary comparability profile: avoid direct cross-run inference without profile harmonization.

---

## 18. Deprecated / legacy fields

Common legacy/deprecated families:

- `harmonic_energy_ratio`, `inharmonic_energy_ratio`, `subbass_energy_ratio` (alias family split into legacy/diagnostic usage contexts).
- stage-1 legacy scalars: `Density Metric`, `Combined Density Metric`, `Total Metric`, and normalized variants.
- deprecated subbass cutoff shims in `low_frequency_policy.py` and `acoustic_density_core.py`.
- strict alias columns routed to `Legacy_Aliases` sheet.

New work should prioritize canonical and explicitly versioned columns.

---

## 19. Bibliography and parameter provenance

Primary provenance files:

- `docs/CONSTANTS_PROVENANCE.md`
- `docs/validation/FORMULA_VALIDATION_STATUS.md`
- `metrics_dictionary.json`
- inline source comments in `constants.py`, `mir_descriptors.py`, `inharmonicity_model.py`, `subbass_policy.py`.

Examples of cited families in code:

- Zwicker & Fastl (psychoacoustics/subbass region context).
- Fletcher / stiff-string inharmonicity model.
- Peeters et al. descriptor conventions.
- Pollard & Jansson tristimulus.
- Krimphoff et al. irregularity.
- Aures roughness approximation.
- Moore & Glasberg ERB-rate transform.

Numeric constants are now explicitly classified as `primary_source`, `derived`, `convention`, or `internal_default` in `docs/CONSTANTS_PROVENANCE.md`. `internal_default` entries are documented engineering defaults (not hidden defects).

---

## 19A. Known documentation and provenance limitations

1. **Constants provenance is now registry-based rather than TODO-based.**  
   `docs/CONSTANTS_PROVENANCE.md` is canonical for constants exported by `constants.py`. Historical TODO-style entries in `docs/parameter_provenance.md` are retained as legacy Phase-6 context and should not be treated as the primary provenance source for current runs.

2. **Registry semantics are still partly code-defined.**  
   `metrics_dictionary.json` now documents canonical/diagnostic/validation and multi-phase families, but some enforcement semantics (sheet splitting, omission filters, dynamic feature gates) still live authoritatively in `compile_metrics.py` and related export code.

3. **Dynamic auxiliary sheets are not stably enumerable.**  
   Sheets such as `Compiled Metrics`, `Compiled_Metrics_All`, PCA outputs, and optional dissonance/debug/correlation sheets are generated conditionally. Their exact column sets vary with runtime options, data sufficiency, and enabled analysis branches.

4. **GUI surface mismatch remains by design.**  
   The Tk orchestrator (`pipeline_orchestrator_gui.py`) is the canonical operational interface. The PyQt interface (`interface.py`) was a legacy/reference surface and has been archived to `Backup/root_modules/`.

5. **Research workbook inharmonicity family gap.**  
   `compiled_density_metrics_research.xlsx` currently does not expose the full `inharmonicity_*` diagnostic family present in compiled `Density_Metrics`; inharmonicity audit should therefore reference the compiled workbook until research mapping is extended.

6. **Formula-validation scope is intentionally proportionate.**  
   `tests/formula_validation/` and `docs/validation/FORMULA_VALIDATION_STATUS.md` currently canonicalise six high-impact formula families (F1–F6). Internal helper expressions are still primarily governed by numerical regression under `tests/phase_*`.

---

## 20. Complete formula index

See `docs/METRIC_FORMULA_INDEX.md` for indexed formulas (`F-001` onward), mapped to functions and exported columns, and `docs/validation/FORMULA_VALIDATION_STATUS.md` for the AST-based canonical validation status of F1–F6.

---

## 21. Complete export column dictionary

See `docs/EXPORT_COLUMN_DICTIONARY.md` for exhaustive sheet-by-sheet exported column coverage for compiled and research workbooks.

---

## Additional explicit Phase 7 interpretations

- `obs_wS` is not a physical energy ratio.
- `phase2_*_weight` values are corpus-level model-density weights, not direct per-note energy fractions.
- High inharmonicity may be physically valid for multiple instrument families.
- Clarinet corpus usage is validation/control context, not universal target behavior.
