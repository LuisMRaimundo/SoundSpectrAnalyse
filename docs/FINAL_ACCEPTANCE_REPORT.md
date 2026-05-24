# FINAL ACCEPTANCE REPORT (Blocker-Fix Pass)

## 1) Blocker Status

| Blocker | Status | Evidence |
|---|---|---|
| Blocker 1: final-density columns populated | PASS | 37/37 (clarinet), 26/26 (cello) for all required columns |
| Blocker 2: GUI control wiring/propagation | PASS | `audit_gui_option_effects.json` central controls = PASS, Metadata propagation = PASS, workbook propagation = PASS on final acceptance workbooks |
| Blocker 3: ceiling-aware naming consistency | PASS | ceiling audit row = PASS using `_up_to_density_ceiling_hz` aliases |
| Blocker 4: metadata completeness | PASS | all required metadata fields non-blank (value or `unavailable_not_recorded`) |
| Blocker 5: 5 new failures beyond baseline | PASS | all 5 regressions fixed; current failures are subset of true baseline failures |
| Blocker 6: GUI option audit rerun | PASS | refreshed `docs/GUI_OPTION_EFFECT_AUDIT.md` + `audit_gui_option_effects.json` |
| Blocker 7: regenerate from audio | PASS | full stage1+stage2+stage3 rerun completed for both corpora |

## 2) GUI Option Audit Before/After

| GUI option | Before (previous rejected run) | After (this blocker-fix run) |
|---|---|---|
| density_summation_mode | NOT EXPOSED | PASS |
| density weights | NOT EXPOSED | PASS |
| density_salience_threshold_db | NOT EXPOSED | PASS |
| density_frequency_ceiling_hz | NOT EXPOSED | PASS |
| Metadata propagation | present=0, missing=14 | PASS |
| workbook propagation | subset=False, full=False (stale check) | PASS (rerun on final acceptance clarinet+cello workbooks) |
| magnitude threshold | prior ambiguous | PASS |
| harmonic tolerance | prior ambiguous | PASS |

Policy note: `Charts_Data` keeps fixed `salient_inharmonic_log_bin_count_up_to_5000hz` as the release plotting field when `density_frequency_ceiling_hz=5000`; the ceiling-aware alias remains available in `Spectral_Density_Metrics`.

## 3) Trace Table (Final-Density Columns)

### Clarinet Trace

| metric | computed in core | present in per-note workbook | present in compiled workbook | present in research workbook | present in Charts_Data | present in Metadata if parameter | status |
|---|---|---|---|---|---|---|---|
| `final_note_density_count_based` | yes | yes | yes | yes | yes | n/a | PASS |
| `final_note_density_salience_weighted` | yes | yes | yes | yes | yes | n/a | PASS |
| `final_note_density_salience_weighted_norm_for_chart` | no | no | no | yes | yes | n/a | PASS |
| `salient_harmonic_order_count_up_to_5000hz` | yes | yes | yes | yes | yes | n/a | PASS |
| `salient_inharmonic_log_bin_count_up_to_5000hz` | yes | yes | yes | yes | yes | n/a | PASS |
| `salient_subbass_particle_count` | yes | yes | yes | yes | yes | n/a | PASS |
| `harmonic_density_component` | yes | yes | yes | yes | yes | n/a | PASS |
| `inharmonic_density_component` | yes | yes | yes | yes | yes | n/a | PASS |
| `subbass_density_component` | yes | yes | yes | yes | yes | n/a | PASS |
| `harmonic_density_weight` | yes | yes | yes | yes | yes | yes | PASS |
| `inharmonic_density_weight` | yes | yes | yes | yes | yes | yes | PASS |
| `subbass_density_weight` | yes | yes | yes | yes | yes | yes | PASS |
| `density_summation_mode` | yes | yes | yes | yes | yes | yes | PASS |
| `density_salience_threshold_db` | yes | yes | yes | yes | yes | yes | PASS |
| `density_frequency_ceiling_hz` | yes | yes | yes | yes | yes | yes | PASS |

### Cello Trace

| metric | computed in core | present in per-note workbook | present in compiled workbook | present in research workbook | present in Charts_Data | present in Metadata if parameter | status |
|---|---|---|---|---|---|---|---|
| `final_note_density_count_based` | yes | yes | yes | yes | yes | n/a | PASS |
| `final_note_density_salience_weighted` | yes | yes | yes | yes | yes | n/a | PASS |
| `final_note_density_salience_weighted_norm_for_chart` | no | no | no | yes | yes | n/a | PASS |
| `salient_harmonic_order_count_up_to_5000hz` | yes | yes | yes | yes | yes | n/a | PASS |
| `salient_inharmonic_log_bin_count_up_to_5000hz` | yes | yes | yes | yes | yes | n/a | PASS |
| `salient_subbass_particle_count` | yes | yes | yes | yes | yes | n/a | PASS |
| `harmonic_density_component` | yes | yes | yes | yes | yes | n/a | PASS |
| `inharmonic_density_component` | yes | yes | yes | yes | yes | n/a | PASS |
| `subbass_density_component` | yes | yes | yes | yes | yes | n/a | PASS |
| `harmonic_density_weight` | yes | yes | yes | yes | yes | yes | PASS |
| `inharmonic_density_weight` | yes | yes | yes | yes | yes | yes | PASS |
| `subbass_density_weight` | yes | yes | yes | yes | yes | yes | PASS |
| `density_summation_mode` | yes | yes | yes | yes | yes | yes | PASS |
| `density_salience_threshold_db` | yes | yes | yes | yes | yes | yes | PASS |
| `density_frequency_ceiling_hz` | yes | yes | yes | yes | yes | yes | PASS |

## 4) Corpus Audit

### Clarinet

- row_count: `37`
- compiled workbook: `<clarinet_output_path>/compiled_density_metrics_final_density_acceptance.xlsx`
- research workbook: `<clarinet_output_path>/compiled_density_metrics_research_final_density_acceptance_metadatafix_provenance.xlsx`
- log file: `<clarinet_output_path>/gui_worker_final_density_acceptance.log`
- count-based formula max error: `0`
- salience-weighted formula max error: `7.1054273576e-15`
- Charts_Data contains H/I/S + final density: `PASS`
- Combined Density Metric absent in Spectral_Density_Metrics: `PASS`
- density_weighted_sum_cdm_mean absent by default: `PASS`
- fallback rows marked acoustically passed: `0` (must be 0)
- Excel formula error cells: `0`
- metadata missing required fields: `0`
- required final-density population:
  - `final_note_density_count_based`: `37/37`
  - `final_note_density_salience_weighted`: `37/37`
  - `final_note_density_salience_weighted_norm_for_chart`: `37/37`
  - `salient_harmonic_order_count_up_to_5000hz`: `37/37`
  - `salient_inharmonic_log_bin_count_up_to_5000hz`: `37/37`
  - `salient_subbass_particle_count`: `37/37`
  - `harmonic_density_component`: `37/37`
  - `inharmonic_density_component`: `37/37`
  - `subbass_density_component`: `37/37`
  - `harmonic_density_weight`: `37/37`
  - `inharmonic_density_weight`: `37/37`
  - `subbass_density_weight`: `37/37`
  - `density_summation_mode`: `37/37`
  - `density_salience_threshold_db`: `37/37`
  - `density_frequency_ceiling_hz`: `37/37`

### Cello

- row_count: `26`
- compiled workbook: `<cello_output_path>/compiled_density_metrics_final_density_acceptance.xlsx`
- research workbook: `<cello_output_path>/compiled_density_metrics_research_final_density_acceptance_metadatafix_provenance.xlsx`
- log file: `<cello_output_path>/gui_worker_final_density_acceptance.log`
- count-based formula max error: `0`
- salience-weighted formula max error: `1.42108547152e-14`
- Charts_Data contains H/I/S + final density: `PASS`
- Combined Density Metric absent in Spectral_Density_Metrics: `PASS`
- density_weighted_sum_cdm_mean absent by default: `PASS`
- fallback rows marked acoustically passed: `0` (must be 0)
- Excel formula error cells: `0`
- metadata missing required fields: `0`
- required final-density population:
  - `final_note_density_count_based`: `26/26`
  - `final_note_density_salience_weighted`: `26/26`
  - `final_note_density_salience_weighted_norm_for_chart`: `26/26`
  - `salient_harmonic_order_count_up_to_5000hz`: `26/26`
  - `salient_inharmonic_log_bin_count_up_to_5000hz`: `26/26`
  - `salient_subbass_particle_count`: `26/26`
  - `harmonic_density_component`: `26/26`
  - `inharmonic_density_component`: `26/26`
  - `subbass_density_component`: `26/26`
  - `harmonic_density_weight`: `26/26`
  - `inharmonic_density_weight`: `26/26`
  - `subbass_density_weight`: `26/26`
  - `density_summation_mode`: `26/26`
  - `density_salience_threshold_db`: `26/26`
  - `density_frequency_ceiling_hz`: `26/26`

## 5) Full-Suite Failure Matrix

- baseline (publication gate baseline): `15 failed, 850 passed, 40 skipped`
- current (after blocker fixes): `15 failed, 850 passed, 40 skipped`
- new failures introduced: `no`
- baseline failures remaining: `15`
- final density tests: `passed`
- export tests: `passed`
- documentation tests: `passed`

| test name | baseline status | current status | new? | cause | fix/action |
|---|---|---|---|---|---|
| `tests/test_density_export_hardening.py::test_density_metrics_sheet_only_partial_sums_no_debug_counts` | PASS | PASS | no | regression introduced in blocker run candidate set | fixed in this pass; verified PASS in baseline and current |
| `tests/test_discrete_spectral_metrics.py::DiscreteSpectralMetricsTests::test_density_metrics_sheet_is_minimal_partial_sums` | PASS | PASS | no | regression introduced in blocker run candidate set | fixed in this pass; verified PASS in baseline and current |
| `tests/test_export_compliance_v6.py::test_density_metrics_sheet_clean_and_side_sheets` | PASS | PASS | no | regression introduced in blocker run candidate set | fixed in this pass; verified PASS in baseline and current |
| `tests/test_output_curation.py::test_metric_family_values_are_in_allowed_enum` | PASS | PASS | no | regression introduced in blocker run candidate set | fixed in this pass; verified PASS in baseline and current |
| `tests/test_rolloff_compensated_harmonic_density.py::test_density_metrics_main_sheet_is_minimal_excluding_rolloff` | PASS | PASS | no | regression introduced in blocker run candidate set | fixed in this pass; verified PASS in baseline and current |

## 6) Release Gate Decision

Release accepted: `YES`

Provenance-only metadata patch confirmation:
- final density deltas vs previous metadatafix workbooks:
  - clarinet: `count_based=0.0`, `salience_weighted=0.0`
  - cello: `count_based=0.0`, `salience_weighted=0.0`
- frequency/magnitude unresolved fields are explicitly flagged as source-limited with:
  - `frequency_magnitude_fields_recovery_status = partially_unavailable_in_compiled_source`

Gate checklist:
- final density columns populated in real corpus workbooks: `PASS`
- GUI controls exposed and effective: `PASS`
- metadata records required settings: `PASS`
- new failures introduced: `PASS`
- final density formulas pass: `PASS`
- workbook hygiene checks pass: `PASS`

