# Formula Validation Status

This file records the canonical form of each formula tested by
`tests/formula_validation/`. The tests verify that the implementation
matches the canonical form at the symbolic-structure level. Numerical
correctness is verified separately under `tests/phase_*`.

The status of each formula is one of:

- **canonicalised_and_tested** — canonical form documented here and verified by a test under `tests/formula_validation/`.
- **canonicalised_only** — canonical form documented here but not yet verified by a structural test.
- **not_yet_canonicalised** — formula exists in the implementation but has no canonical-form documentation.

## F1 — Canonical H/I/S weighted density formula

- **Canonical form**: `density_metric_raw = D_H * w_H + D_I * w_I + D_S * w_S`
- **Module**: `metric_contract.py` (declaration); `density.py`, `compile_metrics.py` (application).
- **Version tag**: `density_formula_version = "v5_apply_density_metric_adapted_v6_1"`.
- **Test**: `tests/formula_validation/test_density_formula_canonical.py`.
- **Status**: canonicalised_and_tested.

## F2 — FFT-length normalisation factor (Phase 8)

- **Canonical form**:
  - `peak_amplitude_sum`: factor = `N_ref / N`
  - `peak_power_sum`: factor = `(N_ref / N)^2`
  - `broadband_amplitude_l2`: factor = `sqrt(N_ref / N)`
  - `broadband_power_l2`: factor = `N_ref / N`
- **Module**: `spectral_normalization.py` (function `n_fft_normalization_factor`).
- **Reference**: Harris (1978); Heinzel et al. (2002). See `REFERENCES.md`.
- **Test**: `tests/formula_validation/test_n_fft_normalization_factor_canonical.py`.
- **Status**: canonicalised_and_tested.

## F3 — Stiff-string inharmonicity fit (Fletcher 1962)

- **Canonical form**: `f_n = n * f0 * sqrt(1 + B * n^2)`.
- **Module**: `inharmonicity_model.py` (function `fit_inharmonicity_coefficient`).
- **Reference**: Fletcher (1962); Fletcher & Rossing (1998). See `REFERENCES.md`.
- **Test**: `tests/formula_validation/test_stiff_string_fit_canonical.py`.
- **Status**: canonicalised_and_tested.
- **Estimator (current)**: **joint `(f0, B)`** least squares. The model squared
  is linear in two parameters, `f_n^2 = a·n^2 + c·n^4` with `f0 = sqrt(a)` and
  `B = c/a`, solved by OLS on the squared peak-center frequencies with iterative
  harmonic-order reassignment. The `f0_hz` argument only seeds order assignment.
  A **significance gate** keeps `B` only when the `n^4` coefficient is
  statistically distinguishable from zero (`|t| >= 2`), which suppresses the
  spurious small `B` that a 2-parameter fit would otherwise read from sub-bin
  frequency-measurement noise. The fit now also returns `inharmonicity_fit_f0_hz`.
- **Isolated accuracy**: `tests/phase_4/test_inharmonicity_recovers_known_B.py`
  recovers a known `B` within 20%; `test_inharmonicity_zero_for_pure_harmonic.py`
  returns `B ≈ 0` on exact harmonics.
- **End-to-end accuracy (resolved)**: previously the magnitude of a non-zero `B`
  collapsed to ≈0 through the pipeline because (i) the fit was anchored to the
  stretch-absorbing robust-fitted `f0` and (ii) it was fed the raw significant-bin
  cloud, so order matching picked lobe-flank bins nearest the nominal grid. Both
  are fixed: the inharmonicity fit is now fed **local-maximum peak centers**
  (parabolic sub-bin refinement, `acoustic_density_core._local_maxima_peak_centers`)
  and uses the **joint `(f0, B)`** estimator above.
  `tests/phase_11/test_ground_truth_accuracy.py` verifies, end-to-end, that a
  stiff-string synthetic with `B = 3e-4` is recovered within `[0.4x, 2.5x]` and
  that a pure-harmonic stack yields `B ≈ 0` (no false inharmonicity).

## F4 — Sub-bass upper bound (Zwicker & Fastl 1990)

- **Canonical form**: `upper_bound_hz = min(f0_hz * 0.5, 80.0)`.
- **Module**: `subbass_policy.py` (`SubBassPolicy.upper_bound_hz`).
- **Reference**: Zwicker & Fastl (1990). See `REFERENCES.md`.
- **Test**: `tests/formula_validation/test_subbass_policy_canonical.py`.
- **Status**: canonicalised_and_tested.

## F5 — Effective partial density (participation ratio / inverse Herfindahl)

- **Canonical form**: `D_eff = (sum_i P_i)^2 / sum_i(P_i^2)`.
- **Module**: `density.py` (documented in module-level docstring or canonical-formula constant).
- **Interpretation**: scale-invariant in power; equivalent to the inverse Herfindahl concentration index of the power-component vector.
- **Test**: `tests/formula_validation/test_effective_partial_density_canonical.py`.
- **Status**: canonicalised_and_tested.

## F6 — Jensen–Shannon divergence (Lin 1991)

- **Canonical form**:
```
  m = 0.5 * (p + q)
  JS(p, q) = 0.5 * (KL(p, m) + KL(q, m))
```
- **Module**: `adaptive_density_engine.py` (function `_js_divergence`).
- **Reference**: Lin (1991). See `REFERENCES.md`.
- **Test**: `tests/formula_validation/test_js_divergence_canonical.py`.
- **Status**: canonicalised_and_tested.

## F7 — Harmonic acceptance: cell-averaging CFAR (constant false-alarm rate)

- **Canonical form**: under noise-only, squared-magnitude FFT bins are ~exponential
  (chi-square, 2 dof). Cell-averaging CFAR threshold
  `T = alpha * noise_mean`, with `alpha = N·(Pfa^(-1/N) − 1)` for `N` training
  cells; detect iff `peak_power >= T`,
  `cfar_margin_db = 10·log10(peak_power / T)`.
- **Module**: `harmonic_peak_validation.py` (`cfar_peak_detection`), gated in
  `_classify_harmonic_candidate` (CFAR-detected **and** saddle-prominence ⇒
  `strict_validated`).
- **Reference**: constant false-alarm-rate detection (Rohling 1983, CA-CFAR);
  spectral peak detection under exponential noise. See `REFERENCES.md`.
- **Test**: `tests/phase_11/test_cfar_detection.py` (strong-peak detection,
  floor-level rejection, bounded empirical false-alarm rate, Pfa monotonicity);
  acceptance behaviour guarded by `tests/phase_10` (dense low-register),
  `tests/acoustic_validity`, and `tests/phase_8` (FFT invariance).
- **Status**: canonicalised_and_tested.
- **Rationale**: replaces the previous fixed 3 dB SNR margin (ad-hoc, register-
  and noise-blind) with an adaptive, stated-false-alarm criterion, extending the
  same detection-theoretic significance-gate philosophy used for `B` (F3). The
  noise estimate trims the strongest training cells so neighbouring partials in
  dense spectra do not inflate the local floor. Default `Pfa = 1e-2`, calibrated
  to preserve the validated acoustic chain.

## F7 — Strict EWSD (Stage 3, EWSD-R v18)

- **Canonical form**: `EWSD_score_total = sum_k r_k D_k (N_eff,k / N_k)` over H/I/S compartments.
- **Module**: `tools/ewsd_core.py` (`compute_ewsd`); integration in `tools/ewsd_research_integration.py`.
- **Reference**: participation-ratio anti-concentration; see `docs/TECHNICAL_MANUAL_COMPLETE.md` §7.8 and `docs/METRIC_FORMULA_INDEX.md` F-048.
- **Test**: `tests/phase_11/test_research_export_includes_ewsd.py`.
- **Status**: canonicalised_only (numerical regression; no symbolic AST test yet).

## F8 — Acoustic-balanced EWSD companion (Stage 3)

- **Canonical form**: `EWSD_score_acoustic_balanced = sum_k r_k D_k (N_eff,k / N_k)^alpha`, default `alpha = 0.5`.
- **Module**: `tools/ewsd_core.py` (`add_acoustic_alignment_columns`).
- **Reference**: see `docs/METRIC_FORMULA_INDEX.md` F-049.
- **Test**: `tests/phase_11/test_research_export_includes_ewsd.py`.
- **Status**: canonicalised_only.

## Coverage notes

This suite is proportionate, not exhaustive. The six formulae covered are those that are either explicitly version-tagged or the subject of an explicit phase entry in `CHANGES.md`. Internal helper formulae are governed by the numerical regression tests under `tests/phase_*` rather than by symbolic-structure tests, since the cost of false positives from over-specified AST tests is greater than the benefit at that level of granularity.

Future passes may canonicalise additional formulae as they become version-tagged or methodologically significant. The status of each addition should be recorded in this file before any corresponding test is added.
