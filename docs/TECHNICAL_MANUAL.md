# Technical Manual (Final Accepted Architecture)

## 1) Purpose and Scope

SoundSpectrAnalyse processes pitched instrumental note recordings and produces note-level spectral metrics, validation metadata, and research workbooks.

The software:
- reads pitched note files (for example `.wav` and `.aif`);
- performs spectral analysis and peak extraction;
- resolves f0 provenance (acoustic fit or explicit fallback);
- classifies harmonic, inharmonic/residual, and subbass/particle content;
- computes final note-density metrics;
- exports compiled and research workbooks with `Dashboard`, `Charts_Data`, `Metadata`, `Validation_Summary`, and `Analysis_Settings_By_Note`.

The software does **not** claim:
- to be a psychoacoustic loudness model;
- to be a universal timbre model;
- to provide a single "truth metric" for all musical density;
- that fallback f0 equals acoustic verification;
- that chart-normalized values are absolute across runs.

Current final-density architecture:
- primary metric: `final_note_density_salience_weighted`
- control metric: `final_note_density_count_based`
- harmonic raw-count sanity metric: `salient_harmonic_order_count_up_to_density_ceiling_hz` (or fixed alias `salient_harmonic_order_count_up_to_5000hz` when ceiling is 5000 Hz)

## 2) Pipeline Overview

### Stage 0 - GUI / Configuration
- Input folder and output paths.
- Final-density controls: mode, H/I/S weights, salience threshold, frequency ceiling.
- STFT settings (window, FFT strategy, hop strategy, zero padding, magnitude range).
- Harmonic classification settings (tolerance strategy/value, frequency band).
- Secondary/diagnostic controls.

### Stage 1 - Per-note analysis
- Audio load and note parsing.
- f0 acquisition and provenance selection.
- STFT and peak detection.
- Harmonic/inharmonic/subbass classification.
- Metric computation in acoustic core.
- Per-note workbook write (`spectral_analysis.xlsx`).

### Stage 2 - Compilation
- Merge per-note workbooks into `compiled_density_metrics.xlsx`.
- Preserve final-density fields and validation status fields.
- Preserve canonical/diagnostic separation.
- Optional exploratory PCA/diagnostic outputs.

### Stage 3 - Research export
- Build `compiled_density_metrics_research*.xlsx`.
- Produce `Spectral_Density_Metrics`, `Charts_Data`, `Dashboard`, `Metadata`, `Analysis_Settings_By_Note`, `Legacy_Compatibility`, and `Validation_Summary`.
- Keep legacy metrics out of primary final-density interpretation.

## 3) STFT and Spectral Analysis

STFT definition:

\[
X_m[k] = \sum_{n=0}^{N-1} x[n+mR]\,w[n]\,e^{-j2\pi kn/N}
\]

where:
- \(x[n]\): signal
- \(w[n]\): analysis window
- \(N\): `n_fft`
- \(R\): hop length
- \(k\): frequency bin index
- \(m\): frame index

Frequency bin mapping:

\[
f_k = \frac{k\,F_s}{N}
\]

Main analysis parameters:
- `window_type`
- `n_fft` (or tier strategy)
- `hop_length` (or tier strategy)
- `zero_padding`
- `frequency_min_hz`, `frequency_max_hz`
- `magnitude_min_db`, `magnitude_max_db`
- harmonic tolerance strategy/value

Tier strategy versus fixed mode:
- in 90-tier granular mode, per-note analysis settings can vary by tier;
- in fixed FFT mode, scalar settings are shared.

When tier-dependent:
- Metadata should record `tier_dependent_see_Analysis_Settings_By_Note` for tier-varying fields;
- `Analysis_Settings_By_Note` stores actual per-note values.

`Analysis_Settings_By_Note` is required for reproducibility whenever settings vary by note or tier.

## 4) f0 Provenance and Validation

Canonical f0 path used for density:
- `f0_final_hz` (if valid and accepted),
- else `f0_initial_hz`,
- else `f0_prior_hz`,
- else invalid/NaN.

Key fields:
- `f0_used_for_density_hz`
- `f0_used_for_density_source`
- `acoustic_f0_status`
- `f0_fit_accepted`
- `f0_fit_rejection_reason`
- `arithmetic_validation_status`
- `acoustic_validation_status`

Status meaning:
- `fit_accepted_acoustically_verified`: acoustic fit accepted.
- `nominal_fallback_used_not_acoustically_verified`: fallback f0 used for deterministic computation, not acoustic confirmation.
- `missing_invalid_f0`: no valid f0 available.

Arithmetic validation and acoustic validation are intentionally separate.

Canonical f0 provenance path (verbatim):
`f0_final(valid) -> f0_initial(valid) -> f0_prior_hz(valid) -> NaN`

## 5) Harmonic Extraction Algorithm

Harmonic order model:
\[
f_n = n\,f_0
\]

Expected order count up to ceiling \(F_c\):
\[
N_{\text{expected}}(F_c) = \left\lfloor \frac{F_c}{f_0}\right\rfloor
\]

For candidate peak \(f_i\):
\[
n_i = \operatorname{round}(f_i/f_0)
\]

Acceptance can be represented as either:
- cents tolerance:
\[
\left|1200\log_2\left(\frac{f_i}{n_i f_0}\right)\right|\le \text{tolerance}_{\text{cents}}
\]
- or Hz tolerance:
\[
|f_i-n_i f_0|\le \text{tolerance}_{\text{hz}}
\]

Implementation detail: both tolerance representations exist in project paths; exported tolerance provenance should be read from `harmonic_tolerance` and per-note settings.

Unique-order counting policy:
- multiple nearby peaks mapping to same harmonic order count once;
- one representative per order is used for salience contribution;
- leakage must not multiply harmonic-order counts.

Key metrics:
- `salient_harmonic_order_count_up_to_density_ceiling_hz`
- `salient_harmonic_order_count_up_to_5000hz` (fixed alias)
- `expected_harmonic_order_count_up_to_density_ceiling_hz`
- `salient_harmonic_coverage_up_to_density_ceiling_hz`
- `salient_harmonic_mass_up_to_density_ceiling_hz`
- clarinet-oriented descriptors:
  - `salient_odd_harmonic_count_up_to_density_ceiling_hz` (or fixed-5000 equivalent),
  - `salient_even_harmonic_count_up_to_density_ceiling_hz` (or fixed-5000 equivalent),
  - `odd_even_harmonic_energy_ratio`

## 6) Inharmonic / Residual Extraction

Inharmonic/residual candidates are peaks not assigned to accepted harmonic windows and not subbass-classified.

Rationale:
- raw residual peak counts are unstable (leakage/noise inflation);
- occupied log-frequency bins are used to stabilize residual counting.

Core metric:
- `salient_inharmonic_log_bin_count_up_to_density_ceiling_hz`
- fixed alias: `salient_inharmonic_log_bin_count_up_to_5000hz`

Residual occupancy descriptor:
- `residual_log_frequency_occupancy` (occupied residual bins / total residual bins).

Residual energy descriptor:
- `core_residual_energy_ratio` in the core peak-classification energy family.

## 7) Subbass / Particle Extraction

Subbass/particle components represent low-frequency residual events (for example bow/breath/noise-like low events), not normal harmonic partials.

Primary count:
- `salient_subbass_particle_count`

Salience contribution:
- `subbass_density_component`

Default weighting in final-density mode:
- `wS = 0.25` (lower than harmonic default to avoid domination by low-frequency particle bursts).

## 8) Salience Mapping

For component/peak \(i\):
\[
dB^{\text{rel}}_i = dB_i - dB_{\max,\text{note}}
\]

With threshold \(T = \) `density_salience_threshold_db` (default \(-45\) dB):
\[
\text{salience}_i = \operatorname{clip}\left(\frac{dB^{\text{rel}}_i - T}{0-T},\,0,\,1\right)
\]

With \(T=-45\):
\[
\text{salience}_i = \operatorname{clip}\left(\frac{dB^{\text{rel}}_i+45}{45},\,0,\,1\right)
\]

Equivalent implementation token:
`salience_i = clip((dB_rel_i + 45) / 45, 0, 1)`

Interpretation:
- \(dB^{\text{rel}}_i \le T\) contributes 0;
- \(dB^{\text{rel}}_i = 0\) contributes 1;
- linear interpolation between threshold and maximum;
- capping prevents one very strong peak from being misread as higher count/density.

## 9) Final Density Metrics

Define component counts:
- \(H =\) `salient_harmonic_order_count_up_to_density_ceiling_hz`
- \(I =\) `salient_inharmonic_log_bin_count_up_to_density_ceiling_hz`
- \(S =\) `salient_subbass_particle_count`

Count-based metric:
\[
\text{final\_note\_density\_count\_based}=w_HH+w_II+w_SS
\]

Verbatim control formula:
`final_note_density_count_based = wH*H + wI*I + wS*S`

Defaults:
- \(w_H=1.0\)
- \(w_I=0.5\)
- \(w_S=0.25\)

Salience-weighted components:
- \(H_s\): summed harmonic salience over unique harmonic orders
- \(I_s\): summed salient inharmonic-bin salience
- \(S_s\): summed salient subbass salience

Primary final metric:
\[
\text{final\_note\_density\_salience\_weighted}=w_HH_s+w_II_s+w_SS_s
\]

Chart-only normalization:
\[
x_{\text{norm}}=\frac{x-\min(x)}{\max(x)-\min(x)}
\]

for `final_note_density_salience_weighted_norm_for_chart` (run-relative; not absolute).

## 10) Density Summation Modes

- `harmonic_only`:
  - \(w_H=1,w_I=0,w_S=0\)
  - expected: `final_note_density_count_based == H`
- `inharmonic_only`:
  - \(w_H=0,w_I=1,w_S=0\)
  - expected: `final_note_density_count_based == I`
- `subbass_only`:
  - \(w_H=0,w_I=0,w_S=1\)
  - expected: `final_note_density_count_based == S`
- `his_weighted`:
  - GUI weights are used directly.

Recorded in workbook:
- `density_summation_mode`
- `harmonic_density_weight`
- `inharmonic_density_weight`
- `subbass_density_weight`
- `density_salience_threshold_db`
- `density_frequency_ceiling_hz`

## 11) Harmonic Coverage and Occupancy Distinction

Validation coverage:
\[
\text{harmonic\_slot\_coverage\_ratio}=\frac{\text{harmonic\_slot\_matched\_count}}{\text{harmonic\_slot\_expected\_count}}
\]

Acoustic occupancy:
\[
\text{harmonic\_occupancy\_ratio}=\frac{\text{harmonic\_occupancy\_detected\_order\_count}}{\text{expected\_harmonic\_slot\_count}}
\]

These are related but not identical constructs and must not be conflated.

## 12) Body/Thickness Secondary Metrics

Body-weighted effective density:
\[
\text{body\_weighted\_effective\_density}=
\frac{\left(\sum_i w_{\text{body}}(f_i)\sqrt{P_i}\right)^2}
{\sum_i\left(w_{\text{body}}(f_i)\sqrt{P_i}\right)^2}
\]
with
\[
w_{\text{body}}(f)=\frac{1}{1+(f/1800)^2}
\]

Residual contribution:
\[
\text{residual\_body\_contribution}=
\text{core\_residual\_energy\_ratio}\cdot\text{residual\_log\_frequency\_occupancy}
\]
\[
\text{residual\_body\_contribution\_capped}=\min(\text{residual\_body\_contribution},0.25)
\]

Composite thickness index:
\[
\text{spectral\_body\_thickness\_index}=
0.45\,z(\text{body\_weighted\_effective\_density})+
0.25\,z(\text{low\_mid\_energy\_ratio})+
0.20\,z(\text{harmonic\_body\_density\_normalized})+
0.10\,z(\text{residual\_body\_contribution\_capped})
\]

This is secondary (body/thickness), not the final H/I/S density metric.

## 13) Entropy and Participation Metrics

If \(p_i=P_i/\sum_jP_j\):
\[
H=-\sum_i p_i\log_2(p_i),\quad
\text{spectral\_entropy}=H/\log_2(N)
\]

Participation ratio:
\[
\text{effective\_partial\_density}=\frac{(\sum_iP_i)^2}{\sum_iP_i^2}
\]

Interpretation:
- `spectral_entropy`: spread/dispersion descriptor;
- `effective_partial_density`: effective participation count;
- neither is the final note density.

## 14) Energy Families

Core peak-classification family:
- `core_harmonic_energy_ratio`
- `core_residual_energy_ratio`
- `core_subbass_energy_ratio`
- expected sum approximately 1.

Component-balance family:
- `component_harmonic_energy_ratio`
- `component_inharmonic_energy_ratio`
- `component_subbass_energy_ratio`
- expected sum approximately 1.

Do not mix ratios from different families in one sum interpretation.

## 15) Legacy and Diagnostic Metrics

Not final-density metrics:
- `density_metric_raw`
- `density_weighted_sum`
- `energy_weighted_component_density_diagnostic`
- `Combined Density Metric`
- `Weighted Combined Metric`
- `Total Metric`
- `density_weighted_sum_cdm_mean`
- `effective_partial_density`
- `spectral_body_thickness_index`

Policy:
- `Legacy_Compatibility` is for back-compatibility;
- final-density interpretation should use H/I/S final metrics.

## 16) Workbook Schema

- `Spectral_Density_Metrics`: primary per-note research table.
- `Charts_Data`: plotting table (includes chart-normalized fields).
- `Dashboard`: KPIs and trends.
- `Metadata`: run-level provenance (settings, paths, hash, commit, version).
- `Analysis_Settings_By_Note`: per-note analysis settings and f0 provenance.
- `Legacy_Compatibility`: legacy-only metrics.
- `Validation_Summary`: arithmetic/acoustic validation view.
- optional PCA sheets: exploratory only unless explicitly validated.

`Analysis_Settings_By_Note` columns include:
- `Note`, `MIDI`
- `f0_used_for_density_hz`, `f0_used_for_density_source`, `acoustic_f0_status`
- `tier_name`, `n_fft`, `hop_length`, `zero_padding`, `window_type`, `harmonic_tolerance_hz`
- `frequency_min_hz`, `frequency_max_hz`, `magnitude_min_db`, `magnitude_max_db`
- `density_summation_mode`, `harmonic_density_weight`, `inharmonic_density_weight`, `subbass_density_weight`
- `density_salience_threshold_db`, `density_frequency_ceiling_hz`

## 17) GUI Options and Effects

Each control should be interpreted as either final-density, spectral-analysis, secondary, or diagnostic/legacy.

Minimum documented controls:
- Density mode (`density_summation_mode`, default `his_weighted`, final-density).
- Harmonic weight (`harmonic_density_weight`, default `1.0`, final-density).
- Inharmonic/noise weight (`inharmonic_density_weight`, default `0.5`, final-density).
- Subbass/particle weight (`subbass_density_weight`, default `0.25`, final-density).
- Salience threshold (`density_salience_threshold_db`, default `-45`, final-density).
- Density ceiling (`density_frequency_ceiling_hz`, default `5000`, final-density).
- STFT controls (`window_type`, `n_fft`/tier strategy, `hop_length`/tier strategy, `zero_padding`, magnitude range, tolerance).
- Secondary controls (for example dissonance model) are not final-density definers.

Verification evidence is documented in:
- `docs/GUI_OPTION_EFFECT_AUDIT.md`
- `audit_gui_option_effects.json`

## 18) Quality Control and Tests

Release acceptance checks include:
- required final-density columns populated;
- formula max errors within tolerance (count-based exact in tested cases);
- GUI effect audit PASS for mode/weights/threshold/ceiling/metadata/log;
- metadata propagation present and non-blank where required;
- no Excel formula errors;
- fallback rows not marked acoustically passed;
- `Combined Density Metric` absent from primary research sheet;
- `density_weighted_sum_cdm_mean` absent by default;
- no new failures relative to baseline-known failures.

Expected formula identities:
- harmonic-only: `final_note_density_count_based == H`
- inharmonic-only: `final_note_density_count_based == I`
- subbass-only: `final_note_density_count_based == S`
- weighted: `final_note_density_count_based == wH*H + wI*I + wS*S`

## 19) Interpretation Guide

Typical readings:
- high H, low I/S: harmonically dense and coherent.
- low H, high I: stronger residual/noisy/inharmonic structure.
- high S: strong low-frequency particle contribution.
- high `final_note_density_salience_weighted`: many salient weighted contributors.
- high `spectral_body_thickness_index`: thicker/body-rich spectrum, not necessarily higher final H/I/S density.
- high `spectral_entropy`: more distributed spectral power.

## 20) Register Interpretation

Raw harmonic counts are intentionally register-dependent under fixed ceiling:
\[
N_{\text{expected}}=\left\lfloor\frac{F_c}{f_0}\right\rfloor
\]

Lower f0 implies more harmonic opportunities below \(F_c\). This is expected behavior, not a bug. Use normalized coverage or residualized analyses for register-controlled comparisons.

## 21) Instrument Notes

Clarinet:
- odd/even harmonic structure can be informative;
- harmonic-order trends can decline in upper register under fixed ceiling.

Cello:
- residual/subbass content may track bow/noise behavior;
- low notes have more harmonic opportunities under fixed ceiling;
- inspect `acoustic_f0_status` and fallback counts before interpretation.

## 22) Limitations

- Metrics are corpus- and setting-dependent.
- Threshold/ceiling settings materially affect outputs.
- f0 fallback is computationally valid but not acoustically verified.
- STFT/tolerance settings influence classification.
- Cross-instrument comparisons require matched settings.
- `*_norm_for_chart` values are run-relative only.
- Final density is not loudness, roughness, or a complete timbre model.

## 23) Reproducibility Checklist

For publication/reporting, record:
- corpus path and curation;
- instrument/dynamic/articulation conventions;
- sampling rate and preprocessing assumptions;
- `window_type`;
- `n_fft` / tier strategy;
- `hop_length` / tier strategy;
- `zero_padding`;
- harmonic tolerance strategy/value;
- `density_summation_mode`;
- `harmonic_density_weight`, `inharmonic_density_weight`, `subbass_density_weight`;
- `density_salience_threshold_db`;
- `density_frequency_ceiling_hz`;
- f0 validation and fallback counts;
- `git_commit`, `git_branch`;
- workbook hash (`source_workbook_sha256`);
- run timestamp;
- final metric used (`final_note_density_salience_weighted`).

