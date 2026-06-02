# Stage 3 EWSD-R v18 integration in research export (2026-06-02)

Integrates Effective Weighted Spectral Density (EWSD-R v18) into the canonical
research workbook export so bibliography-facing density comparisons no longer
require a separate post-processing GUI step.

- **New modules:** `tools/ewsd_core.py` (EWSD-R v18 computation core),
  `tools/ewsd_research_integration.py` (Stage 3 discovery, compute, left-join).
- **Pipeline hook:** `post_compile_research_export.run_research_workbook_export`
  now triggers EWSD inside `tools/export_research_density_workbook.build_workbook`.
  Per-note `spectral_analysis.xlsx` workbooks under the analysis folder are
  recomputed with `individual_exact` mode; H/I/S ratios are read from each
  note's Metrics sheet (`auto_excel_required`) — no silent H=I=S=1 defaults.
- **Research columns added to `Spectral_Density_Metrics`:**
  `EWSD_score_total`, `EWSD_score_acoustic_balanced`, `ewsd_mode`,
  `ewsd_primary_analysis_eligible`, `ewsd_his_ratio_source`, `ewsd_H_ratio`,
  `ewsd_I_ratio`, `ewsd_S_noise_ratio`, `ewsd_weight_function_canonical`,
  `ewsd_acoustic_balance_alpha`, `ewsd_stage3_version`, `ewsd_merge_status`.
- **Publication gate:** use only rows with `ewsd_primary_analysis_eligible == True`
  for final thesis statistics. For cross-instrument bibliographic distance,
  prefer `EWSD_score_acoustic_balanced`; keep `EWSD_score_total` as strict EWSD.
- **Tests:** `tests/phase_11/test_research_export_includes_ewsd.py`.
- **Docs:** README, `CHANGES.md`, `TECHNICAL_MANUAL_COMPLETE.md`,
  `EXPORT_COLUMN_DICTIONARY.md`, `DENSITY_EXPORT_SCHEMA.md` §R,
  `CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md` §9, `METRIC_FORMULA_INDEX.md`,
  `pipeline.md`, `pipeline_runtime.md`.

## Documentation sweep — orchestrator, contract, metadata (2026-06-02)

Aligns remaining entry points and metadata with the three-stage pipeline:

- **Orchestrator / entry points:** `run_orchestrator.py`, `pipeline_orchestrator_integrated.py`,
  `pipeline_orchestrator_gui.py`, `main.py` — Stage 3 post-compile EWSD messaging.
- **Contract:** `pipeline_contract.py` — Stage 3 constants, EWSD module paths,
  contract version `SSA_CANONICAL_PIPELINE_2026_06_STAGE1_PROC_AUDIO_STAGE2_COMPILE_STAGE3_EWSD`.
- **Metadata:** `metrics_dictionary.json` (EWSD columns), `docs/GUI_OPTION_REFERENCE.md` §A4,
  `docs/parameter_provenance.md`, `docs/validation/FORMULA_VALIDATION_STATUS.md` F7/F8,
  `installers/README.md`.

# Density energy gate: full-spectrum region basis + non-harmonic terminology (2026-05-29)

Resolves the "band-vs-peak" inconsistency between the inharmonic density weight
(`wI`) and the reported inharmonic energy, generalised for ANY instrument.

Root issue (terminology + basis): the code conflated two physically distinct
things under "inharmonic" — (a) partial INHARMONICITY (discrete non-`n·f0`
tonal peaks: piano stretch, bells; coefficient B) and (b) the inter-harmonic
RESIDUAL (broadband bow/breath/attack noise + any non-`n·f0` content). The v57
density gate also used BODY-ceiling-truncated band energies, which made the
non-harmonic energy share arbitrary and instrument-dependent (bright/noisy tones
carry most residual energy ABOVE the body ceiling, so the gate silently
collapsed toward a peak-only basis).

- **`acoustic_density_core.py` energy gate → v58
  (`v58_full_spectrum_region_energy_gate`).** The gate now weights each band's
  structural strength by the FULL-SPECTRUM, total-power-normalised region energy
  triple (`harmonic_energy_ratio` / `residual_energy_ratio` /
  `subbass_energy_ratio`). The three region powers partition every spectral bin,
  so they conserve energy (sum to total power) and are instrument-agnostic:
  discrete inharmonic peaks (piano/bell) and broadband noise (bowed/wind) both
  land in the non-harmonic residual band and both correctly contribute to
  perceived spectral density. New audit fields
  `component_strength_energy_gate_{harmonic,non_harmonic_residual,subbass}` and
  `density_band_energy_basis` are exported; the legacy
  `component_strength_energy_gate_{h,i,s}` names are retained as aliases.
- **Terminology corrected.** The density middle band is now documented as the
  NON-HARMONIC / inter-harmonic RESIDUAL, not "inharmonic". Partial
  inharmonicity (coefficient B + inharmonic-peak energy) is a SEPARATE physics
  descriptor and is no longer conflated with the density gate. The component
  energy-ratio pie is relabelled as the peak/physics view, explicitly distinct
  from the density residual basis.
- Test expectations and README/docs updated `v57 → v58`. Full suite green
  (112 passed, 2 skipped), including the subbass-suppression regression and the
  strength-formula-units contract.

# Unified single-scalar note density across H+I+S (2026-05-30)

Adds `note_effective_component_density` — ONE per-note density that spans all
three bands (harmonic + inharmonic + sub-bass), separates instruments, and is
designed as the per-note basis for downstream chord/aggregate density.

- **Definition.** Energy-weighted participation ratio (effective number of
  energy-bearing spectral components) pooled over the validated harmonic peaks,
  inharmonic peaks, and sub-bass particles using raw amplitudes:
  `N_eff = (Σ Aᵢ²)² / Σ Aᵢ⁴`. One scalar; covers harmonics + inharmonics +
  sub-noise; computed in `compile_metrics._energy_distribution_density`
  (Density_Metrics column).
- **Why this one.** It is timbre-discriminating (Orchidea pooled means
  Trombone 5.08 > Cello 2.74 > Clarinet 1.82; clarinet lowest at matched pitch),
  far less register-bound than `note_density_final` (pooled r≈−0.63 vs −0.96),
  and **aggregates for chords**: applying the same formula to the pooled partials
  of several notes yields the chord's effective component count, with coincident
  partials fusing (modelling masking) rather than double-counting. 131/131 notes
  populated across the three corpora.
- `note_density_final` and the harmonic-only "fatness" columns are retained
  unchanged; this is the unified cross-band density for aggregate work.

# Energy-distribution density — timbral "fatness" restored (2026-05-30)

Adds register-robust, energy-based density descriptors that separate timbres,
addressing the finding that `note_density_final` (log-weighted, per-note
normalised) behaves as a partial-COUNT measure dominated by register: across
130 Orchidea notes it correlated r≈−0.96 with pitch and barely separated the
three instruments at matched pitch (mean spread 0.11). The historical objective
of the code — *more harmonics carrying considerable energy ⇒ denser* — is
recovered as explicit first-class columns rather than by mutating the validated
`note_density_final`.

- **New compiled `Density_Metrics` columns** (computed in
  `compile_metrics._energy_distribution_density` from the validated harmonic
  peaks of each note's Harmonic Spectrum sheet, so no Stage-1 re-run is needed):
  - `harmonic_effective_partial_count` — participation ratio
    `(Σ Aₙ²)² / Σ Aₙ⁴` (effective number of partials carrying energy).
  - `harmonic_energy_above_fundamental_ratio` — fraction of harmonic energy not
    in the fundamental (0 = concentrated at f0; →1 = spread across partials).
  - `harmonic_energy_centroid_order` — energy-weighted mean harmonic order
    (brightness in harmonic-order units).
  - `effective_partial_density` — full-spectrum participation ratio (surfaced).
- **Validated on the cello/clarinet/trombone Orchidea corpora.** Pooled Neff
  Trombone 5.04 > Cello 2.64 > Clarinet 1.80; energy-above-f0 0.82 / 0.54 / 0.31.
  At matched pitch the new density separates the instruments ~16× more than
  `note_density_final` (mean spread 1.69 vs 0.11 for Neff), with the
  acoustically-correct ordering (brass spread > bowed string > closed-tube reed).
- Note: `note_density_final` is intentionally unchanged (count/register density,
  with its bootstrap CI / UQ contract intact); the new columns are the
  complementary energy-distribution density.

# Robust f0 global-fit order-clipping bugfix (2026-05-29)

Fixes the high f0-rejection rate (and consequent suppression of the
inharmonicity-B fit) on low-pitched, harmonic-rich tones (cello C2–C4).

- **`_estimate_f0_global_robust` (proc_audio.py) now FILTERS partials to
  rounded order `[1, max_n]` instead of CLIPPING the order to `max_n`.** The
  weighted least-squares estimator `f0 = Σ(w·n·f)/Σ(w·n²)` was fed every
  detected strict peak (cello C2 carries ~100+ peaks up to order ~300), but the
  order label was clipped to `max_n=15`. Each high-order partial was therefore
  relabelled as order 15, so e.g. a 6500 Hz partial contributed 6500/15 ≈ 433 Hz
  to the f0 estimate, dragging it far above the truth (C2: 65 Hz → ~114 Hz,
  |Δf0| ≈ 48 Hz). The acceptance gate (|Δf0| ≤ 2 % f0) then correctly rejected
  the garbage fit on almost every low note (`f0_fit_accepted=False`), which also
  starved the downstream inharmonicity (B) estimation. Low-order partials are
  both reliably labelable and the least inharmonic, so they are the correct
  basis for the pure `f = n·f0` model; the fix restricts the fit to them.
  Verified on synthetic cello combs: C2/F2/A2 now estimate within |Δf0| < 0.5 Hz
  (fit_quality ≈ 0.011 ≪ 0.10 gate) and are accepted. All 21 f0 / inharmonicity
  / FFT-invariance / ground-truth regression tests pass.

# density_metric_raw_per_note_balance reconciliation (2026-05-29)

Closes the last cross-sheet scale residual found while auditing the cello run.
`density_metric_raw_per_note_balance` (the per-note, energy-ratio-weighted
comparator `r_H·D_H + r_I·D_I + r_S·D_S`) was computed on the canonical
log-weighted band density in the `Density_Metrics` sheet (cello C2 = 3.22) but
on the raw wide-frame band sums in the harvested `Diagnostic_Metrics` sheet
(cello C2 = 173624). The single-source-of-truth reconciliation in
`compile_metrics._write_compiled_excel` only propagated `density_metric_raw` /
`density_metric_normalized`, leaving this column on the stale raw basis.

- **`density_metric_raw_per_note_balance` added to
  `_CANONICAL_DENSITY_COLS_TO_PROPAGATE`.** The reconciliation now overwrites the
  wide frame with the canonical `Density_Metrics` value, so every derived sheet
  reports the same log-weighted figure. Reconciled-column count goes 9 → 10.
  Verified on the cello corpus: `Density_Metrics` and `Diagnostic_Metrics` now
  both report `density_metric_raw_per_note_balance` = 3.2199 (matching the
  `density_metric_raw` = 2.75 log basis); the raw 173624 leak is gone.
  `tests/phase_2/test_phase2_profile_actually_applied.py` and
  `tests/phase_6/test_density_metrics_excel_export_phase2_profile.py` still pass
  (they read the authoritative `Density_Metrics` value, unchanged).

# note_density_final scale-consistency fix (2026-05-29)

Fixes a cross-sheet scale inconsistency in `note_density_final` surfaced on the
cello run, plus log-noise polish:

- **note_density_final no longer uses the raw display-sum fallback
  (`compile_metrics._resolve_note_density_sum_column`).** The resolver had a
  third-priority fallback to the legacy display columns `Harmonic Partials sum`
  / `Inharmonic Partials sum` / `Sub-bass sum`. In the per-note `Metrics` sheet
  (and therefore in the harvested wide compiled frame) those columns carry a RAW
  partial sum on a different scale than the GUI-weighted band density
  `*_density_sum` (cello C2: `Harmonic Partials sum` = 174178 raw vs
  `harmonic_density_sum` = 3.22 log). The fallback produced a wrong-scale
  `note_density_final` (173624) in the wide-frame-derived `Diagnostic_Metrics`
  sheet, while the canonical `Density_Metrics` sheet and the research workbook —
  which carry the true `*_density_sum` columns — stayed correct (3.22). The
  fallback is removed: `note_density_final` is now computed **only** from the
  canonical weighted band sums, so it is correct wherever present and simply
  absent from the diagnostic sheet (which lacks those columns) rather than wrong.
  Verified end-to-end: `Density_Metrics` and research `note_density_final` =
  3.2199, 26/26 non-NaN; `Diagnostic_Metrics` no longer carries the column.
- **Wide-frame skip logged as INFO, not WARNING.** When the wide compiled frame
  cannot compute `note_density_final` (no canonical `*_density_sum` columns —
  the expected, correct outcome), the message is now an INFO noting it is
  computed authoritatively in `Density_Metrics`; a WARNING is kept only for any
  other (unexpected) context.

Known, deferred (documented for transparency): the wide-frame / `Diagnostic_Metrics`
density family (`density_metric_raw`, `weighted_*_density_contribution`) is still
computed from the raw `Harmonic Partials sum` and therefore reported on a raw
scale in that diagnostic sheet, inconsistent with the canonical `Density_Metrics`
values. Publication outputs (`Density_Metrics`, research `Spectral_Density_Metrics`)
are unaffected. A full reconciliation of the per-note `Metrics` `Partials sum`
semantics with the canonical weighted band sums is planned as a dedicated phase.

Full suite: 112 passed, 2 skipped.

---

# Adaptive observation energy-anchoring + CFAR log fix (2026-05-29)

Fixes a real defect surfaced by a cello (IOWA arco mf, C string) run and a
stale log string:

- **Energy-anchored adaptive observation (`obs_w_formula_version`
  `v56_occupancy_ratio` → `v57_energy_anchored_occupancy`).** The per-note
  adaptive observation that feeds `AdaptiveDensityEngine` (and hence
  `density_metric_raw` / the Phase-2 profile) was computed purely from
  structural *occupancy + density-per-slot*, each normalised by the band's
  expected slot count. Because bands have wildly different expected counts
  (harmonic ~ hundreds of orders; sub-bass ~ a handful of bins), at permissive
  salience thresholds a spectrally narrow, energetically negligible band could
  saturate its occupancy on the noise floor and dominate the learned profile.
  Observed: cello **C2 sub-bass carried 3.6e-5 of the spectral energy yet drew
  `pure_observation_w_s = 0.52`**, and the wide inharmonic band out-weighted the
  99.7%-energy harmonic band. Fix (`acoustic_density_core.py`): each band's
  structural strength is now weighted by its **measured energy share**
  (`component_strength_energy_gate_{h,i,s}`, exported for audit), so a band with
  ~0 energy contributes ~0 to the observation while richness still modulates the
  weight among energetically-present bands. This makes the adaptive observation
  physically coherent with the measured component energy ratios that
  `note_density_final` already uses. New regression:
  `tests/phase_6/test_subbass_observation_cap.py::test_energy_anchoring_suppresses_noise_floor_subbass_band`.
- **CFAR log string corrected (`proc_audio.py`).** The per-note harmonic summary
  no longer announces the obsolete "SNR ≥ 3 dB" criterion; it now states the
  active gate: "(CFAR detection [Pfa-based] + saddle prominence ≥ 3 dB)".

Note: `note_density_final` (energy-ratio based) is unaffected by the first fix —
it never used the adaptive weights. Full suite: 112 passed, 2 skipped.

---

# Methodological closure: CFAR acceptance, primary-by-default, full UQ (2026-05-29)

Closing the final three methodological inconsistencies flagged in the doctoral
re-evaluation:

- **Harmonic acceptance is now detection-theoretic (CFAR).** The ad-hoc fixed
  3 dB SNR margin is replaced by a cell-averaging CFAR test
  (`harmonic_peak_validation.cfar_peak_detection`): at each refined peak bin the
  power must exceed a threshold derived from a stated false-alarm probability
  (`pfa`, default `1e-2`) against a locally-estimated, peak-trimmed noise floor.
  A candidate becomes `strict_validated` only when CFAR-detected **and** clearing
  saddle prominence — the same significance-gate philosophy already applied to
  the inharmonicity coefficient `B`. New audit columns `cfar_margin_db` /
  `cfar_detected` in `Harmonic_Inclusion_Audit`. Calibrated to preserve the
  validated acoustic chain (dense low-register recovery, FFT invariance,
  ground-truth accuracy all green).
- **Primary comparable profile is the analysis default.** The orchestrator GUI
  now defaults the amplitude weighting to `Logarithmic` (the PRIMARY profile,
  `wf=log`), so even an isolated single run is cross-instrument comparable by
  default; any other choice downgrades the run to EXPLORATORY (logged and
  flagged). Per-note `Analysis_Metadata` already self-declares
  `is_primary_comparable_profile` / `analysis_parameter_profile_id`.
- **Full uncertainty quantification.** `bootstrap_note_density_final` gains
  `propagate_ratio_uncertainty` (default used by the pipeline = True): component
  energy ratios are recomputed inside each resample from the bootstrapped band
  energies, so the uncertainty of BOTH the band sums and the ratios is propagated
  jointly. New column `note_density_final_uncertainty_sources`
  (`partials+ratios`). The window/n_fft sensitivity is exposed as an opt-in study
  tool `tools/note_density_nfft_sensitivity.py` (re-analysis at multiple FFT
  sizes is intentionally kept out of the hot path).

Full suite: 111 passed, 2 skipped.

---

# Scientific-robustness closure blocks (2026-05-29)

Closing the three open robustness gaps surfaced by the earlier phases:

- **Block 1 — joint (f0, B) inharmonicity fit.** `inharmonicity_model.fit_inharmonicity_coefficient`
  now estimates `f0` and `B` jointly (linear OLS on `f_n^2 = a·n^2 + c·n^4`,
  `f0=sqrt(a)`, `B=c/a`, iterative order reassignment) with a **t-test
  significance gate** on the `n^4` term (keeps `B` only when `|t|>=2`). The
  inharmonicity fit is now fed **local-maximum peak centers** (parabolic sub-bin,
  `acoustic_density_core._local_maxima_peak_centers`) instead of the raw
  significant-bin cloud. This closes the end-to-end B-magnitude under-recovery:
  `tests/phase_11` now asserts recovery of a known `B=3e-4` within `[0.4x,2.5x]`
  and `B≈0` (no false positive) on a pure-harmonic stack. New export:
  `inharmonicity_fit_f0_hz`. FORMULA_VALIDATION_STATUS.md F3 updated (resolved).
- **Block 2 — refuse cross-profile aggregation.** `Canonical_Primary_Filtered`
  is now hard-restricted to a single analysis profile
  (`_restrict_primary_subset_to_single_profile`; dominant profile kept, others
  dropped), and a single authoritative `corpus_comparable_for_statistics`
  boolean is written to `Analysis_Metadata`. Primary statistics can no longer be
  silently computed across mixed profiles.
- **Block 3 — uncertainty emitted with the metric.** Each compiled note now
  carries a per-note non-parametric bootstrap CI for `note_density_final`
  (`note_density_final_ci_low`, `note_density_final_ci_high`,
  `note_density_final_rel_uncertainty`), computed transform-aware via
  `density_uncertainty.bootstrap_note_density_final` (guarded; NaN if a per-note
  workbook is unreadable). Surfaced on both `Density_Metrics` and research
  `Spectral_Density_Metrics`.

Full suite: 106 passed, 2 skipped.

---

# Scientific-robustness phases (2026-05-29)

Three phased additions to harden acoustic/scientific robustness:

- **Phase 1 — cross-profile comparability guard.** `compile_metrics._corpus_comparability_audit`
  surfaces a corpus verdict (`corpus_comparability_status`, profile count,
  primary-comparable row count) into `Analysis_Metadata` and WARNS when a
  compiled workbook mixes analysis profiles or is single-but-EXPLORATORY.
  Density metrics are only comparable within one primary profile;
  `Canonical_Primary_Filtered` remains the physically isolated comparable subset.
  Tests: `tests/phase_11/test_corpus_comparability_guard.py`.
- **Phase 2 — end-to-end ground-truth accuracy.** New
  `tests/phase_11/test_ground_truth_accuracy.py` synthesises signals with known
  content and asserts the full pipeline recovers harmonic **frequencies** (<25
  cents) and **amplitude ratios** (1/n within 35%), and produces **no false
  inharmonicity** on a pure-harmonic stack. FINDING: end-to-end recovery of a
  non-zero inharmonicity **B magnitude** is unreliable (the fit is anchored to
  the stretch-absorbing robust-fitted f0; a joint f0–B fit is required). Logged
  as an open limitation in `docs/validation/FORMULA_VALIDATION_STATUS.md` (F3).
- **Phase 3 — uncertainty quantification.** New `density_uncertainty.py`
  (`bootstrap_density_ci`, `nfft_sensitivity`) gives a non-parametric bootstrap
  CI for `note_density_final` (resampling per-partial contributions; ratios held
  fixed) and an n_fft/window sensitivity band (CV, relative range). Tests:
  `tests/phase_11/test_density_uncertainty.py`, including an end-to-end check
  that `note_density_final` is bounded-stable across n_fft on a fixed signal.

Full suite: 102 passed, 2 skipped.

---

# Code + documentation synchronization (2026-05-29)

## Functional changes

- **`note_density_final`** (new primary per-note scalar density) on the compiled
  `Density_Metrics` sheet and the research `Spectral_Density_Metrics` sheet:
  `r_H·harmonic_density_sum + r_I·inharmonic_density_sum + r_S·subbass_density_sum`,
  where `r_*` are the per-note **measured** `component_*_energy_ratio` values (not the
  Bayesian adaptive weights) and each `*_density_sum` already carries the GUI amplitude
  weight function. NaN-propagating. Source: `compile_metrics._compute_note_density_final`.
  Highlighted light blue on the research sheet.
- **`Harmonic_Inclusion_Audit`** read-only per-note sheet in each `spectral_analysis.xlsx`:
  one row per harmonic order with `exclusion_reason` and the SNR/prominence/ceiling/
  deviation diagnostics that explain density inclusion/exclusion.
- **Harmonic validation correctness:** f0-adaptive saddle-prominence window (±f0/2),
  removal of the asymmetric `n>10` gate, candidate re-alignment to the fitted f0, and
  a tolerance-scaled refine radius (restores FFT-tier amplitude invariance). Fixes the
  dense low-register (cello) under-counting.
- **Performance:** `mir_descriptors._roughness_aures_1985` vectorised with a
  critical-band window (per-note runtime ~333 s → ~20 s on cello C2; result unchanged
  to ~1e-7).
- **Module decomposition:** extracted the pure peak-validation cluster from
  `proc_audio.py` into the new `harmonic_peak_validation.py` (re-exported by
  `proc_audio`). Note: `harmonic_validation.py` is a DISTINCT pre-existing module
  (`validate_harmonic_series_matched`) and is unchanged.
- **New test guards:** `tests/perf/` (roughness + per-note budget) and
  `tests/acoustic_validity/` (instrument-family harmonic-richness contracts).
- **Cleanup:** unused modules archived to `Backup/` and removed from `pyproject.toml`
  py-modules (`interface`, `export_paths`, `public_audio_identifiers`,
  `reference_signal_utils`, `runtime_versions`, `audio_analysis/batch_example.py`,
  `scripts/harmonic_count_audit.py`). See `Backup/README.md`.

## Documentation updated for the above

- `metrics_dictionary.json`: added `note_density_final` (canonical).
- `docs/METRIC_FORMULA_INDEX.md`: added F-042 (`note_density_final`).
- `docs/EXPORT_COLUMN_DICTIONARY.md`: interpretation row + column-inventory entries for
  `note_density_final`.
- `docs/DENSITY_EXPORT_SCHEMA.md`: normative §2.1 (`note_density_final`) and §2b
  (per-note `Harmonic_Inclusion_Audit`).
- `pipeline.md`: added `harmonic_peak_validation.py`, corrected the `harmonic_validation.py`
  description, and added an "Archived modules (moved to `Backup/`)" section.
- `docs/GUI_OPTION_REFERENCE.md`, `docs/TECHNICAL_MANUAL_COMPLETE.md`,
  `docs/MANUAL_COVERAGE_REPORT.md`: annotated `interface.py` as archived to `Backup/`.

---

# Documentation synchronization to current code state (2026-05-27)

## Files touched and rationale

- `docs/TECHNICAL_MANUAL_COMPLETE.md`
  Updated provenance and limitations sections to reflect current canonical documentation sources: constants registry (`docs/CONSTANTS_PROVENANCE.md`) and formula-validation status (`docs/validation/FORMULA_VALIDATION_STATUS.md`), replacing stale TODO-era framing. Also corrected FFT-tier normalization equations to match the current Phase-8 `quantity_kind` contract (`peak_amplitude_sum: N_ref/N`, `peak_power_sum: (N_ref/N)^2`) and documented backward-compatible broadband-L2 branches explicitly.

- `docs/MANUAL_COVERAGE_REPORT.md`
  Synchronized scope/caveat language with current repository state; added explicit resolved rows for constants-provenance registry completion and formula-validation baseline (F1-F6).

- `docs/parameter_provenance.md`
  Added a current-state note clarifying this file is now a historical Phase-6 ledger for signature defaults, while constants provenance is canonicalized in `docs/CONSTANTS_PROVENANCE.md`.

- `pipeline.md`
  Updated Phase-6 provenance row to reference the current constants-provenance registry path and removed stale generator-path wording. Updated P3 normalization row to the current Phase-8 peak-sum normalization law.

- `pipeline_runtime.md`
  Updated runtime-path P3 normalization row to the current Phase-8 peak-sum normalization law (`N_ref/N` and `(N_ref/N)^2`) instead of the legacy broadband-L2 form.

- `docs/METRIC_FORMULA_INDEX.md`
  Corrected F-021 and F-022 to the current Phase-8 tier-normalization formulas for exported peak-amplitude and peak-power sums.

- `README.md`, `pipeline_runtime.md`
  Applied terminology/notation consistency polish only (e.g., `tier-normalized`, `normalization`, `N_ref`) with no change to technical claims, formulas, or export semantics.

- `metrics_dictionary.json`
  Bumped `registry_version` to `phase_8_docs_sync_v1` to mark documentation/registry synchronization to the current code state.

## Scope and non-scope

- No computational logic changed.
- No `.py` implementation module modified.
- No tests modified.
- Documentation and metadata synchronization only.

# README.md technical body restoration (2026-05-27)

## Files touched and rationale

- `README.md`
  Inserted the technical scaffolding sections (Status; What this software does; Installation; Usage; Outputs; Scientific governance) at named positions, between the pre-existing authorial sections. The opening paragraph, Theoretical anchoring, License, Citation, and Acknowledgements sections were preserved byte-for-byte. The README now functions as a doctoral-grade entry document and links to `REFERENCES.md`, `CITATION.cff`, `CHANGES.md`, `docs/TECHNICAL_MANUAL_COMPLETE.md`, `docs/EXPORT_COLUMN_DICTIONARY.md`, `docs/METRIC_FORMULA_INDEX.md`, `docs/CONSTANTS_PROVENANCE.md`, `docs/validation/FORMULA_VALIDATION_STATUS.md`, `tests/formula_validation/`, and `LICENSE`.

## Scope and non-scope

- No `.py` file modified.
- No test modified.
- No file under `docs/` modified.
- No authorial content altered.
- `REFERENCES.md`, `pyproject.toml`, `CITATION.cff`, and `LICENSE` untouched.

# Proportionate formula-validation suite (2026-05-27)

## Files touched and rationale

- `tests/formula_validation/` (new directory, six tests plus conftest)
  Six AST-based formula-extraction tests covering the canonical form of: the H/I/S weighted density formula (F1), the Phase-8 FFT-length normalisation factor (F2), the stiff-string inharmonicity fit (F3), the sub-bass upper bound (F4), the effective partial density (F5), and the Jensen-Shannon divergence (F6). The suite complements the numerical regression tests under `tests/phase_*`: those detect drift in numerical outputs, these detect drift in the symbolic structure of the formulae themselves.

- `docs/validation/FORMULA_VALIDATION_STATUS.md` (new)
  Per-formula record of canonical form, module, reference, test path, and status. This document is the citation target for any methodology-chapter reference to the formula-validation suite.

## Scope and non-scope

- No existing test modified.
- No `.py` module under the implementation tree modified.
- The suite is deliberately proportionate (six formulae), not a full mirror of `SoundSpectrAnalyse-main/tests/formula_validation/`, because v55 has scientific modules (`adaptive_density_engine.py`, `inharmonicity_model.py`, `subbass_policy.py`, `spectral_normalization.py`, etc.) that have no counterpart in `main`. A direct mirror would be incoherent.
- Tests are conservative: they assert structural invariants (presence of `sqrt`, polynomial degree, additive structure) rather than exact AST identity, to avoid firing on legitimate refactoring.

# Authorial completion of README.md and CITATION.cff; LICENSE file and pyproject.toml license synchronisation (2026-05-27)

## Files touched and rationale

- `README.md`
  Filled the four `[AUTHOR: ...]` placeholder blocks with content supplied by the author (opening framing paragraph, dissertation reference sentence, full proprietary licence summary, FCT funding citation, and acknowledgements). Technical sections were not altered.

- `CITATION.cff`
  Filled author identity (Luís Raimundo, NOVA University of Lisbon), version and release date, proprietary licence reference (`LicenseRef-Proprietary`), and added a structured `references` entry citing the FCT doctoral grant DOI `10.54499/2020.08817.BD`. ORCID and project URL fields were omitted as the author has not supplied them.

- `LICENSE`
  Created at the repository root with the full proprietary copyright notice supplied by the author.

- `pyproject.toml`
  `license = {text = "Scientific Research Use"}` was replaced with `license = {text = "Proprietary - All Rights Reserved"}` to synchronise the package metadata with the LICENSE file and the README. The `"License :: Other/Proprietary License"` classifier was added if absent. No other field was modified.

## Scope and non-scope

- No `.py` file modified.
- No test modified.
- No file under `docs/` modified.
- `REFERENCES.md` not modified.
- `py-modules` list in `pyproject.toml` not modified.
- No licence was selected on the author's behalf — the proprietary posture was supplied by the author.

# Constants provenance registry and pyproject.toml module manifest completion (2026-05-27)

## Files touched and rationale

- `docs/CONSTANTS_PROVENANCE.md`
  Created per-constant provenance registry classifying every numeric constant in `constants.py` as `primary_source`, `derived`, `convention`, or `internal_default`. Honest classification preferred over fabricated provenance.

- `constants.py`
  Extended `_PROVENANCE_SOURCED_CONSTANTS` to cover every constant classified as `primary_source`, `derived`, or `convention` in the registry. Softened the unsourced-constants notification from `RuntimeWarning` to `logging.INFO`, in keeping with the fact that `internal_default` constants are a documented design choice rather than a defect. Module-level docstring updated to reference `docs/CONSTANTS_PROVENANCE.md` and `REFERENCES.md`.

- `REFERENCES.md`
  Extended only if a new primary source was cited that was not already present.

- `pyproject.toml`
  Completed the `py-modules` manifest to include all 48 top-level Python modules. Previously, 23 modules — including all of the new scientific modules (`adaptive_density_engine`, `inharmonicity_model`, `metric_contract`, `mir_descriptors`, `spectral_normalization`, `subbass_policy`, `temporal_segmentation`, etc.) — were absent from the manifest and would not have been packaged into an installed wheel.

## Scope and non-scope

- No numeric constant value was altered.
- No constant was renamed or removed.
- No test was modified.
- No computational logic was changed.
- No exported metric or schema was changed.

# Bibliographic provenance pass — REFERENCES.md and inline docstring references (2026-05-27)

## Files touched and rationale

- `REFERENCES.md`
  Created canonical APA-7 bibliography at repository root; serves as the single source of truth for the theoretical anchors used in the scientific modules and as the bridge document between source code and dissertation.

- `inharmonicity_model.py`, `mir_descriptors.py`, `adaptive_density_engine.py`, `metric_contract.py`, `temporal_segmentation.py`
  Added short-form inline `References` blocks to module-level docstrings, mirroring the convention already used in `spectral_normalization.py` and `subbass_policy.py`. No computational logic, signatures, or exported metrics were modified.

## Scope and non-scope

- No tests modified.
- No exported metric names, schemas, or formula versions altered.
- `spectral_normalization.py` and `subbass_policy.py` were intentionally left untouched as they already carry correct inline references.

# Phase 8 - FFT-Length Normalization for Peak Sums (2026-05-26)

## Files touched and rationale

- `spectral_normalization.py`  
  Refactored `n_fft_normalization_factor(...)` to make scaling assumptions explicit via `quantity_kind`, adding peak-sum laws (`peak_amplitude_sum -> N_ref/N`, `peak_power_sum -> (N_ref/N)^2`) and preserving legacy `kind="amplitude"/"power"` as deprecated aliases to the broadband-L2 laws for backward compatibility.

- `compile_metrics.py`  
  Updated tier-normalization mapping to use peak-sum semantics for production exported sums (`harmonic/inharmonic/subbass_amplitude_sum` and `harmonic/inharmonic/subbass_energy_sum`) and switched all normalization call sites to `quantity_kind=`; this removes FFT-length bias from the compiled `*_tier_normalized` peak-sum columns.

- `pipeline_orchestrator_gui.py`  
  Updated the fallback normalization call used by Phase-1 diagnostics to the explicit peak-power law (`quantity_kind="peak_power_sum"`), avoiding legacy deprecation usage and keeping Phase-1 derived tier-normalized sub-bass energy aligned with the new Phase-8 semantics.

- `tests/phase_8/test_normalization_factor_peak_amplitude.py`  
  Added unit regression for `peak_amplitude_sum` factors at `n_fft=4096` and `n_fft=16384`.

- `tests/phase_8/test_normalization_factor_peak_power.py`  
  Added unit regression for `peak_power_sum` factors at `n_fft=4096` and `n_fft=16384`.

- `tests/phase_8/test_peak_amplitude_invariance_in_pipeline.py`  
  Added critical integration regression: synthetic sinusoid processed through real Stage-1/Stage-2 paths at two FFT lengths, asserting `harmonic_amplitude_sum_tier_normalized` invariance within 5%.

- `tests/phase_8/test_legacy_kind_keyword_still_works.py`  
  Added backward-compatibility regression ensuring legacy `kind=` still returns prior broadband-L2 scaling and emits `DeprecationWarning`.

- `tests/phase_8/test_phase_3_test_still_passes.py`  
  Added compatibility regression asserting the existing Phase-3 invariance test still passes under legacy alias behavior.

## Acoustic / methodological justification

- Peak-bin sums and broadband L2 quantities have different N-dependence under the DFT/window model; applying broadband-L2 normalization (`sqrt(N_ref/N)` for amplitude-like quantities) to peak-amplitude sums leaves a systematic cross-tier bias.
- For fixed-window harmonic peaks, coherent gain causes peak magnitudes to scale linearly with N, so peak-amplitude sums require `N_ref/N` and peak-power sums require `(N_ref/N)^2` for cross-tier comparability.

## Cross-tier discontinuity measurement (synthetic benchmark)

- Synthetic 1 kHz sinusoid benchmark (Stage 1 + Stage 2, `n_fft=4096` vs `8192`):  
  pre-Phase-8 (legacy broadband amplitude factor) step discontinuity = **29.903588%**;  
  post-Phase-8 (peak-amplitude factor) step discontinuity = **0.868704%**.

## References (APA)

- Harris, F. J. (1978). On the use of windows for harmonic analysis with the discrete Fourier transform. *Proceedings of the IEEE, 66*(1), 51-83.
- Heinzel, G., Rudiger, A., & Schilling, R. (2002). *Spectrum and spectral density estimation by the Discrete Fourier transform (DFT), including a comprehensive list of window functions and some new at-top windows* (Technical report). Max-Planck-Institut fur Gravitationsphysik.

# Phase 7.1 - Version Tagging, Compiled Exposure, and Warning Cleanup (2026-05-26)

## Files touched and rationale

- `acoustic_density_core.py`  
  Added `OBS_W_FORMULA_VERSION_CURRENT = "v56_occupancy_ratio"` and exported `obs_w_formula_version` with the pure-observation triplet so v56 audit cross-version `obs_w*` comparisons have explicit semantics; switched the canonical runtime sub-bass bound call site from `deprecated_subbass_upper_bound_hz_from_ratio(...)` to `SubBassPolicy.upper_bound_hz(...)` to remove the per-run deprecation warning flagged by the v56 audit while preserving numeric behavior.

- `pipeline_orchestrator_gui.py`  
  Added `obs_w_formula_version` to Phase 1 discovery diagnostics and `phase1_discovered_density_profiles.csv` rows so the same v56 audit traceability tag present per note is propagated into corpus-level adaptive history exports.

- `compile_metrics.py`  
  Extended compiled allow-lists and extraction plumbing to ingest/export `obs_w_formula_version`, `pure_observation_w_{h,i,s}`, `component_strength_{h,i,s}`, and `legacy_component_strength_{h,i,s}_v55` from per-note `Metrics` into `Density_Metrics`; classification logic was updated so these Phase-7 fields are visible in `Diagnostic_Metrics` and remain excluded from `Canonical_Metrics`, addressing the v56 workbook column-audit visibility gap.

- `tests/phase_7_1/test_formula_version_tagged.py`  
  Added regression ensuring `compute_acoustic_density_descriptors(...)` always tags outputs with `obs_w_formula_version == "v56_occupancy_ratio"`.

- `tests/phase_7_1/test_phase7_fields_in_compiled.py`  
  Added compile-path regression ensuring the nine Phase-7 observation/strength fields are present and populated in `Density_Metrics` and available in `Diagnostic_Metrics`.

- `tests/phase_7_1/test_no_operational_deprecation_warning.py`  
  Added runtime warning regression asserting no operational `DeprecationWarning` referencing `SubBassPolicy.upper_bound_hz` is emitted during descriptor computation.

- `tests/phase_7_1/test_numeric_invariance.py`  
  Added invariance regression that emulates the pre-7.1 deprecated-call path and asserts numeric outputs are bit-identical to the direct-policy path, guarding against silent semantic drift.

## Phase 7.1B - Per-note Metrics Serialization Gap (2026-05-26)

- `proc_audio.py`  
  Fixed the per-note `Metrics` writer gap by explicitly serializing `pure_observation_w_{h,i,s}`, `component_strength_{h,i,s}`, `legacy_component_strength_{h,i,s}_v55`, and `obs_w_formula_version` from the in-memory `compute_acoustic_density_descriptors(...)` output state into the per-note `Metrics` sheet row.

- `pipeline_orchestrator_gui.py`  
  Extended `_extract_note_density_feedback_diagnostics(...)` and `phase1_discovered_density_profiles.csv` history rows to carry the same Phase-7.1 field family (`obs_w_formula_version`, `component_strength_*`, `legacy_component_strength_*_v55`) alongside `pure_observation_w_*`, so Phase 1 CSV exports no longer drop these values.

- `tests/phase_7_1/test_phase7_fields_in_compiled.py`  
  Clarified scope as a unit/plumbing test (synthetic workbook fixture path), keeping it as valid narrow coverage.

- `tests/phase_7_1b/test_per_note_metrics_writes_all_phase7_fields.py`  
  Added integration regression that runs Stage 1 on a synthetic WAV, opens on-disk per-note `Metrics`, and verifies all ten Phase-7.1 fields are present/populated with expected semantics.

- `tests/phase_7_1b/test_compiled_has_phase7_fields_populated.py`  
  Added integration regression that runs synthetic Stage 1 + Stage 2 and asserts on-disk `compiled_density_metrics.xlsx` `Density_Metrics` contains all ten Phase-7.1 fields populated for all rows.

- `tests/phase_7_1b/test_phase1_csv_has_phase7_fields_populated.py`  
  Added integration regression validating Phase 1 discovery CSV generation path writes non-NaN `pure_observation_w_*` and propagates the full Phase-7.1 field family, including `obs_w_formula_version`.

# Phase 1 - Decouple Prior from Observation and Fix File Ordering (2026-05-26)

## Files touched and rationale

- `acoustic_density_core.py`  
  Added explicit Phase 1 changelog note, introduced pure observation outputs (`pure_observation_w_h`, `pure_observation_w_i`, `pure_observation_w_s`), preserved legacy prior-smoothed outputs (`smoothed_w_h_legacy`, `smoothed_w_i_legacy`, `smoothed_w_s_legacy`), and changed canonical compatibility aliases (`harmonic_density_weight`, `inharmonic_density_weight`, `subbass_density_weight`) to expose pure observation. Replaced hard-coded `0.55/0.45` with named deprecated constants for traceability and auditability of prior-contaminated behavior.

- `pipeline_orchestrator_gui.py`  
  Added deterministic Phase 1 file ordering utility (`build_phase1_file_iteration_order`) that sorts by parsed note f0 via `canonical_note_from_filename` + `librosa.note_to_hz`, with unparseable names sorted last by filename. Added orchestrator entry-point parameter `enable_adaptive_path_randomization: bool = False` and deterministic seed logging, with default behavior remaining sorted-by-f0. Updated adaptive feedback extraction to prioritize pure observation fields and ensured adaptive engine updates consume pure observations.

- `adaptive_density_engine.py`  
  Added explicit `update()` docstring assertion that `observation` must be pure data ratio input (not prior-mixed), clarifying methodological contract between Stage 1 evidence and online Bayesian-style update.

- `tests/phase_1/conftest.py`  
  Added test bootstrap path setup so Phase 1 tests import project modules consistently.

- `tests/phase_1/test_no_prior_contamination.py`  
  Added regression test proving note-level pure observation output is invariant to strongly different priors for identical spectral evidence.

- `tests/phase_1/test_deterministic_ordering.py`  
  Added regression test proving Phase 1 folder iteration order is monotonic in nominal f0 and deterministic for unparseable note tokens.

## Acoustic / methodological justification

- Observation and prior should be separated in online learning pipelines: the likelihood term should reflect current data, while prior influence should be applied in the posterior update step. Mixing prior into the observation channel causes biased updates and can over-propagate early-run conditions across later notes.
- Deterministic f0-ordered traversal improves reproducibility and interpretability of adaptive trajectories in per-note spectral pipelines; optional seeded randomization provides controlled robustness checks without changing default publication behavior.

## References (APA)

- Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. In Y. Lechevallier & G. Saporta (Eds.), *Proceedings of COMPSTAT'2010* (pp. 177-186). Springer. https://doi.org/10.1007/978-3-7908-2604-3_16
- Bishop, C. M. (2006). *Pattern recognition and machine learning*. Springer.

# Phase 2 - Unify Sub-Bass Semantics and Fix Phase-2 Application Path (2026-05-26)

## Files touched and rationale

- `subbass_policy.py`  
  Added canonical `SubBassPolicy.upper_bound_hz(f0_hz, sr_hz, n_fft)` implementing `min(f0_hz * 0.5, 80.0)` as the single operational sub-bass definition.

- `constants.py`  
  Kept legacy constant compatibility and added deprecated shim `deprecated_subbass_aggregate_cutoff_hz(...)` that routes to `SubBassPolicy` and emits one deprecation warning per process.

- `acoustic_density_core.py`  
  Replaced operational use of legacy `subbass_upper_ratio` with `SubBassPolicy` resolution while retaining a deprecated ratio shim (`deprecated_subbass_upper_bound_hz_from_ratio`) for backward compatibility and testability.

- `low_frequency_policy.py`  
  Kept `calculate_adaptive_subfundamental_cutoff_hz(...)` as a deprecated compatibility shim with one-time warning, internally mapped to `SubBassPolicy` so legacy callers converge on the same cutoff value.

- `proc_audio.py`  
  Replaced all operational sub-bass fallback call sites that used fixed aggregate constants or legacy low-frequency cutoff routine with unified `SubBassPolicy` via `_current_subbass_upper_bound_hz()`.

- `run_real_corpus_validation.py`  
  Updated validation-side cutoff derivation to use `SubBassPolicy`, removing direct dependency on legacy adaptive cutoff computation.

- `pipeline_orchestrator_gui.py`  
  Fixed Phase-2 application path by forwarding `subbass_weight` in Stage-2 `compile_kw` so discovered profile triplets are fully transmitted.

- `compile_metrics.py`  
  Extended `compile_density_metrics_with_pca` and `_compile_density_metrics_impl` with `subbass_weight: float = None` behavior. Updated weighted-density computation so explicit corpus profile (`harmonic_weight`, `inharmonic_weight`, `subbass_weight`) is actually applied when all three are provided; added `density_weights_source` and invariant comparator `density_metric_raw_per_note_balance`.

- `tests/phase_2/conftest.py`  
  Added Phase-2 test import bootstrap.

- `tests/phase_2/test_subbass_policy_single_source.py`  
  Added regression test ensuring legacy entry points resolve to the same sub-bass upper bound for identical `(f0, sr, n_fft)`.

- `tests/phase_2/test_phase2_profile_actually_applied.py`  
  Added regression test ensuring Phase-2 fixed corpus profile is truly applied to `density_metric_raw`, while `density_metric_raw_per_note_balance` remains per-note.

## Acoustic / methodological justification

- A single sub-bass boundary definition prevents semantic drift between Stage-1 extraction, low-frequency guards, and Stage-2 compilation. This improves reproducibility and interpretability of low-frequency component accounting.
- Explicitly surfacing both corpus-profile and per-note-balance raw density scores removes ambiguity about which weighting regime generated a result, preventing silent methodological mismatch in downstream analysis.

## References (APA)

- Zwicker, E., & Fastl, H. (1990). *Psychoacoustics: Facts and models*. Springer.

# Phase 7 - Register-Invariant Strength Formula (2026-05-26)

## Files touched and rationale

- `acoustic_density_core.py`  
  Replaced the Phase-6/v55 incommensurate strength blend (harmonic-order count + residual log-bin count + sub-bass particle count with fixed scalar 0.25) with a register-invariant occupancy-ratio formulation. Added `_expected_residual_bin_count(...)`, denominator guards with `qc_status` append semantics (`register_normalization_denominator_zero_*`), and explicit deprecated exports for the prior v55 strengths:
  `legacy_component_strength_h_v55`, `legacy_component_strength_i_v55`, `legacy_component_strength_s_v55`.  
  Canonical `pure_observation_w_{h,i,s}` now expose the new register-normalized data ratio; `smoothed_w_*_legacy` remain prior-mixed legacy compatibility fields built on top of the new data ratio.

- `constants.py`  
  Added Phase-7 neutral symmetry constants:
  `STRENGTH_OCCUPANCY_WEIGHT_HARMONIC = 1.0`,  
  `STRENGTH_OCCUPANCY_WEIGHT_INHARMONIC = 1.0`,  
  `STRENGTH_OCCUPANCY_WEIGHT_SUBBASS = 1.0`,  
  each documented as Phase-7 equal-weight occupancy defaults. Added the three constants to `_PROVENANCE_SOURCED_CONSTANTS`.

- `compile_metrics.py`  
  Extended compiled `Density_Metrics` allow-lists to carry inharmonicity fit outputs:  
  `inharmonicity_coefficient_B`, `inharmonicity_fit_residual_std_cents`, `inharmonicity_fit_status`, `inharmonicity_fit_method`.  
  Updated per-note extraction to ingest these values from per-note `Metrics` when available and to fall back to the per-note `Inharmonicity_Fit` sheet (first row) otherwise.

- `tests/phase_7/conftest.py`  
  Added phase bootstrap import path setup.

- `tests/phase_7/test_register_invariant_on_synthetic_odd_harmonic.py`  
  Added low/high-register odd-harmonic clarinet-like synthetic regression for harmonic-majority observation constraints.

- `tests/phase_7/test_register_invariance_across_f0.py`  
  Added cross-register invariance regression (`f0` sweep) for `pure_observation_w_h`.

- `tests/phase_7/test_strength_formula_units_match.py`  
  Added dimensionless-scale regression for Phase-7 strength terms (`component_strength_*`) under near-uniform occupancy scenarios.

- `tests/phase_7/test_inharmonicity_columns_in_compiled.py`  
  Added compiled-workbook regression ensuring inharmonicity fit columns propagate into compiled `Density_Metrics`.

- `tests/phase_7/test_clarinet_corpus_wh_is_majority.py`  
  Added corpus-gated clarinet adaptive-profile sanity test (`profile_h >= 0.50`) using `CLARINET_SUSTAINS_DIR`.

- `docs/parameter_provenance.md` and `tools/generate_parameter_provenance.py`  
  Updated provenance ledger generation and entries for the three Phase-7 occupancy symmetry constants.

## Acoustic / methodological justification

- The previous v55 strength formula combined unlike counting alphabets (harmonic order slots vs 100-cent residual bins vs sub-bass particle slots) without normalizing by each alphabet’s available capacity. As f0 rises, this causes systematic register drift independent of actual spectral balance.  
- Register-invariant occupancy normalization enforces commensurate comparison across H/I/S by dividing each density/count term by the number of available slots in that band before combination.  
- Equal occupancy weights (1.0, 1.0, 1.0) are the neutral symmetry point; non-equal settings encode deliberate prior preference and must be explicitly documented.

## References (APA)

- Backus, J. (1974). *The acoustical foundations of music* (2nd ed.). W. W. Norton.
- Benade, A. H. (1976). *Fundamentals of musical acoustics* (2nd ed.). Oxford University Press.
- Cogan, R. (1984). *New images of musical sound*. Harvard University Press.
- Dickens, P., Smith, J., & Wolfe, J. (2007). Improved precision of resonance frequency measurements in musical wind instruments. *The Journal of the Acoustical Society of America, 121*(4), 2020-2026.
- Fletcher, N. H., & Rossing, T. D. (1998). *The physics of musical instruments* (2nd ed.). Springer.
- Bottou, L. (2010). Large-scale machine learning with stochastic gradient descent. In Y. Lechevallier & G. Saporta (Eds.), *Proceedings of COMPSTAT'2010* (pp. 177-186). Springer. https://doi.org/10.1007/978-3-7908-2604-3_16

# Phase 3 - Normalise Amplitudes Across FFT Tiers; Repair Diagnostic Alias (2026-05-26)

## Files touched and rationale

- `spectral_normalization.py`  
  Added canonical FFT-tier normalization helper `n_fft_normalization_factor(n_fft, n_fft_reference=8192, kind=...)` with amplitude (`sqrt(N_ref/N)`) and power (`N_ref/N`) modes.

- `compile_metrics.py`  
  Added `_tier_normalized` companions for cross-note absolute sums in the compiled path:
  `harmonic_amplitude_sum`, `inharmonic_amplitude_sum`, `subbass_amplitude_sum`,
  `harmonic_energy_sum`, `inharmonic_energy_sum`, `subbass_energy_sum`.  
  Added `tier_consistency_status` with explicit row status values including
  `all_tiers_normalised`.  
  Added these normalization/status columns to canonical sheet column sets so exported
  rows expose both raw and normalized forms.

- `acoustic_density_core.py`  
  Replaced mixed-unit diagnostic alias terms with coherent participation-ratio terms:
  `D_H`, `D_R`, `D_S` now all use inverse-Herfindahl (`_effective_count`) over
  harmonic/residual/subbass power arrays, respectively.  
  Added explicit diagnostic term fields:
  `diagnostic_effective_components_h`, `diagnostic_effective_components_r`,
  `diagnostic_effective_components_s`.  
  Added canonical diagnostic output
  `effective_components_weighted_diagnostic` and kept
  `energy_weighted_component_density_diagnostic` as deprecated alias.

- `tests/phase_3/conftest.py`  
  Added Phase-3 test import bootstrap.

- `tests/phase_3/test_tier_normalisation_invariance.py`  
  Added regression test with a synthetic sustained sinusoid analyzed at `n_fft=4096`
  and `n_fft=8192`, asserting tier-normalized agreement within 2%.

- `tests/phase_3/test_diagnostic_unit_coherence.py`  
  Added regression test validating finite, non-negative float outputs for each
  effective-component term and the combined weighted diagnostic.

## Acoustic / methodological justification

- FFT-size changes alter absolute spectral sums in ways that confound cross-note
  comparisons unless a reference normalization is explicitly applied.
- Participation-ratio style diagnostics must combine unit-coherent terms; mixing
  effective-density quantities with integer count surrogates leads to inconsistent
  interpretation and unstable cross-note ranking.

## References (APA)

- Cogan, R. (1984). *New images of musical sound*. Harvard University Press.
- Edwards, J. T., & Thouless, D. J. (1972). Numerical studies of localization in disordered systems. *Journal of Physics C: Solid State Physics, 5*(8), 807-820. https://doi.org/10.1088/0022-3719/5/8/007

# Phase 4 - Parameterise Inharmonicity Instead of Merely Gating It (2026-05-26)

## Files touched and rationale

- `inharmonicity_model.py`  
  Added `fit_inharmonicity_coefficient(...)` with least-squares estimation of stiff-string inharmonicity coefficient `B` in `f_n = n*f0*sqrt(1 + B*n^2)`, including fit status, residual spread, and predicted stretched harmonic grid.

- `constants.py`  
  Added dedicated inharmonicity / adaptive-harmonic-tolerance constants and policy documentation, including references for FFT-bin-aware tolerance floor.

- `acoustic_density_core.py`  
  Integrated inharmonicity fit before harmonic-mask construction.  
  When fit succeeds and `B > 1e-5`, harmonic prediction uses stretched partials; otherwise behavior remains mathematically equivalent to the legacy gate (`B=0` path).  
  Added adaptive per-partial tolerance:
  `max(harmonic_tolerance_cents, 1200 * bin_spacing_hz / (n * f0_hz))`.  
  Exported inharmonicity fit fields and status into descriptor output.

- `proc_audio.py`  
  Added export of inharmonicity fit payload as new workbook sheet `Inharmonicity_Fit` (coefficient, residual, status, method, and stretched predicted frequencies by order).

- `tests/phase_4/conftest.py`  
  Added Phase-4 test import bootstrap.

- `tests/phase_4/test_inharmonicity_zero_for_pure_harmonic.py`  
  Added regression test asserting near-zero `B` for exact harmonic synthetic spectrum.

- `tests/phase_4/test_inharmonicity_recovers_known_B.py`  
  Added regression test for synthetic stiff-string spectrum with known `B=1e-4` and ±20% recovery tolerance.

- `tests/phase_4/test_clarinet_corpus_B_is_small.py`  
  Added clarinet corpus sanity test (`mean B < 1e-5`) gated by `CLARINET_SUSTAINS_DIR` environment variable for environments where corpus audio is available.

## Acoustic / methodological justification

- A fixed ±35-cent harmonic gate can incorrectly label stretched stiff-string partials as inharmonic noise at higher orders.  
- Explicit `B` estimation provides a physically grounded correction that preserves harmonic classification under known inharmonic instruments while remaining backward-compatible for acoustically harmonic cases (`B≈0`).  
- Adaptive tolerance floor tied to FFT-bin spacing prevents deterministic spectral quantization error from being treated as structural inharmonicity.

## References (APA)

- Fletcher, H. (1962). *The physics of musical instruments*. Dover.
- Fletcher, H., Blackham, E. D., & Stratton, R. (1962). Quality of piano tones. *The Journal of the Acoustical Society of America, 34*(6), 749-761.
- Galembo, A., & Askenfelt, A. (1994). Signal representation and estimation of spectral parameters by inharmonic comb filtering. *IEEE Transactions on Speech and Audio Processing, 2*(2), 197-203.
- Järveläinen, H., Karjalainen, M., & Tolonen, T. (2001). Computationally efficient analysis of beating and inharmonicity in musical tones. *Journal of the Audio Engineering Society, 49*(7/8), 695-708.
- McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE Transactions on Acoustics, Speech, and Signal Processing, 34*(4), 744-754.
- Serra, X., & Smith, J. O. (1990). Spectral modeling synthesis: A sound analysis/synthesis system based on a deterministic plus stochastic decomposition. *Computer Music Journal, 14*(4), 12-24.
- Fletcher, N. H., & Rossing, T. D. (1998). *The physics of musical instruments* (2nd ed.). Springer.

# Phase 5 - Extended Timbral Descriptors and Temporal Segmentation (2026-05-26)

## Files touched and rationale

- `mir_descriptors.py`  
  Added Phase-5 MIR descriptor engine covering spectral moments, irregularity, tristimulus family, flatness, rolloff (85/95), Aures roughness, and ERB-weighted spectral density.

- `temporal_segmentation.py`  
  Added envelope-follower segmentation (`attack`, `sustain`, `release`) with MPEG-7 style log-attack-time output.

- `proc_audio.py`  
  Integrated descriptor extraction into per-note export workflow; added whole-note descriptor fields plus segmented variants (`_on_attack`, `_on_sustain`, `_on_release`) and sustain-focused aliases (`_on_sustain_segment`).  
  Exported segmented density component columns and `log_attack_time_s` into `Metrics`.

- `compile_metrics.py`  
  Extended compiled-sheet allowlists to include Phase-5 descriptor columns and segmented columns so they propagate into `compiled_density_metrics.xlsx`.  
  Updated direct per-note extractor to ingest these columns from per-note `Metrics` and carry them into compiled rows.

- `metrics_dictionary.json`  
  Added citation-backed entries for all new Phase-5 descriptors.

- `tests/phase_5/conftest.py`  
  Added Phase-5 test bootstrap.

- `tests/phase_5/test_descriptor_ranges.py`  
  Added regression test asserting valid descriptor-domain ranges on synthetic spectra.

- `tests/phase_5/test_segmentation_on_pluck_synth.py`  
  Added regression test asserting pluck segmentation behavior (`attack < 50 ms`, sustain dominant).

## Acoustic / methodological justification

- MPEG-7/Timbre-Toolbox-compatible descriptors improve external comparability across studies and instrument families.  
- Attack/sustain/release decomposition prevents sustained-reed assumptions from being baked into descriptors when analyzing percussive/plucked instruments with attack-dominant timbre.

## References (APA)

- Aures, W. (1985). Ein Berechnungsverfahren der Rauhigkeit. *Acustica, 58*(5), 268-281.
- Krimphoff, J., McAdams, S., & Winsberg, S. (1994). Caractérisation du timbre des sons complexes. II. Analyses acoustiques et quantification psychophysique. *Journal de Physique IV*, 4(C5), 625-628.
- Moore, B. C. J., & Glasberg, B. R. (1983). Suggested formulae for calculating auditory-filter bandwidths and excitation patterns. *The Journal of the Acoustical Society of America, 74*(3), 750-753.
- Peeters, G. (2004). *A large set of audio features for sound description (similarity and classification) in the CUIDADO project*. IRCAM.
- Peeters, G., Giordano, B., Susini, P., Misdariis, N., & McAdams, S. (2011). The Timbre Toolbox: Extracting audio descriptors from musical signals. *The Journal of the Acoustical Society of America, 130*(5), 2902-2916.
- Pollard, H. F., & Jansson, E. V. (1982). A tristimulus method for the specification of musical timbre. *Acta Acustica united with Acustica, 51*(3), 162-171.

# Phase 6 - Document Magic Numbers; Final Consolidation (2026-05-26)

## Files touched and rationale

- `docs/parameter_provenance.md`  
  Added a generated provenance ledger that enumerates all numeric constants in `constants.py` and all numeric defaults in function signatures for modules touched in Phases 1-5. Each entry now records canonical name, current value, one-line acoustic meaning, source status, qualitative stability range, and an anchoring test file.

- `constants.py`  
  Added a one-time import warning (`RuntimeWarning`) that lists numeric constants lacking bibliographic provenance and marks them explicitly as `TODO: bibliographic justification required`, fulfilling traceability requirements without silently changing analytical behavior.

- `density.py`  
  Kept `aggregate_subbass_noise_peak_power` because live imports still exist; hardened it into an explicit deprecation wrapper (one-time `DeprecationWarning`) with backward-compatible argument mapping to `aggregate_low_frequency_residual_peak_power`.

- `compile_metrics.py`  
  Added strict alias partitioning (`_split_strict_alias_columns`) and wrote strict aliases to a dedicated `Legacy_Aliases` sheet, while keeping the primary metrics sheet narrower and canonical for downstream analyses.

- `tests/phase_6/conftest.py`  
  Added import bootstrap for Phase 6 tests.

- `tests/phase_6/test_parameter_provenance_doc.py`  
  Added regression test asserting that the parameter provenance document exists and exposes required ledger fields.

- `tests/phase_6/test_legacy_aliases_sheet_split.py`  
  Added regression test proving strict aliases move to `Legacy_Aliases` and are excluded from the primary metrics sheet.

- `tests/phase_6/test_density_wrapper_deprecation.py`  
  Added regression test proving legacy sub-bass wrapper still works but emits `DeprecationWarning`.

- `tests/phase_6/test_density_order_invariance.py`  
  Added regression test showing `density_metric_raw` values are permutation-invariant under row reordering.

- `tests/phase_6/test_subbass_observation_cap.py`  
  Added regression test verifying low sub-bass observation weight (`pure_observation_w_s < 0.05`) for a harmonic-like synthetic note.

- `tools/generate_parameter_provenance.py`  
  Added utility script to regenerate `docs/parameter_provenance.md` deterministically from AST-level extraction of constants/defaults.

## Migration guide for downstream analyses (Phases 1-6)

- **Phase 1**  
  Treat `pure_observation_w_h|w_i|w_s` as canonical observation weights. Legacy blended fields are explicitly labeled legacy/deprecated and should not be used for adaptive updates.

- **Phase 2**  
  Sub-bass semantics are unified under `SubBassPolicy.upper_bound_hz`. In compiled outputs, interpret `density_weights_source` to distinguish explicit corpus profile application (`phase2_corpus_profile`) from per-note weighting. Use `density_metric_raw_per_note_balance` for per-note-only comparators.

- **Phase 3**  
  Cross-note absolute sums should use `_tier_normalized` columns for FFT-size comparability. `effective_components_weighted_diagnostic` is the coherent diagnostic; `energy_weighted_component_density_diagnostic` remains as deprecated alias only.

- **Phase 4**  
  Consume inharmonicity fields (`inharmonicity_coefficient_B`, fit status/residual/method) for harmonicity diagnostics. Clarinet-like harmonic corpora should remain near `B≈0`; stretched instruments are now handled physically rather than by fixed gate alone.

- **Phase 5**  
  New MIR and temporal descriptors are available as whole-note and segmented forms. Segment suffixes (`_on_attack`, `_on_sustain`, `_on_release`, `_on_sustain_segment`) are semantically distinct and should not be collapsed without explicit modeling intent.

- **Phase 6**  
  Strict aliases have moved to the `Legacy_Aliases` sheet; primary analysis should use canonical fields from `Density_Metrics` / `Canonical_Metrics`. Treat import-time provenance warnings as a signal to prioritize bibliographic completion before publication.

## References (APA)

- Fletcher, N. H., & Rossing, T. D. (1998). *The physics of musical instruments* (2nd ed.). Springer.
- Galembo, A., & Askenfelt, A. (1994). Signal representation and estimation of spectral parameters by inharmonic comb filtering. *IEEE Transactions on Speech and Audio Processing, 2*(2), 197-203.
- Järveläinen, H., Karjalainen, M., & Tolonen, T. (2001). Computationally efficient analysis of beating and inharmonicity in musical tones. *Journal of the Audio Engineering Society, 49*(7/8), 695-708.
- McAulay, R. J., & Quatieri, T. F. (1986). Speech analysis/synthesis based on a sinusoidal representation. *IEEE Transactions on Acoustics, Speech, and Signal Processing, 34*(4), 744-754.
- Moore, B. C. J. (2012). *An introduction to the psychology of hearing* (6th ed.). Brill.
- Zwicker, E., & Fastl, H. (1990). *Psychoacoustics: Facts and models*. Springer.
