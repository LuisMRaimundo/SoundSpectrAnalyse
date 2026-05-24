# GUI Option Effect Audit (Current GUI Wiring Verification)

- Repo root: `<repo_root_path>`
- Corpus: `<clarinet_corpus_path>`
- Deterministic subset: `D3_3.45sec_Sustains.wav, F4_3.32sec_Sustains.wav, A4_3.86sec_Sustains.wav, G5_3.66sec_Sustains.wav, C6_4.03sec_Sustains.wav`
- Execution path: GUI orchestrator (`pipeline_orchestrator_gui.RobustOrchestratorApp._process_folder_complete_pipeline`)

## Audit Table

| GUI option | tested values | expected effect | observed effect | pass/fail | affected columns | notes |
|---|---|---|---|---|---|---|
| density_summation_mode | harmonic_only / inharmonic_only / subbass_only / his_weighted | Mode-specific formula equalities hold | harmonic_only=True, inharmonic_only=True, subbass_only=True, his_weighted_formula=True | PASS | final_note_density_count_based | Executed via GUI orchestrator path (_process_folder_complete_pipeline). |
| density weights (wH,wI,wS) | A(1,0.5,0.25), B(1,0,0), C(0,1,0), D(0,0,1), E(2,0.5,0.25) | Changing weights changes final densities per formula | formula_ok=True; deltas_B=['final_note_density_count_based', 'final_note_density_salience_weighted']; deltas_C=['final_note_density_count_based', 'final_note_density_salience_weighted']; deltas_D=['final_note_density_count_based', 'final_note_density_salience_weighted']; deltas_E=['final_note_density_count_based', 'final_note_density_salience_weighted'] | PASS | final_note_density_count_based, final_note_density_salience_weighted | His-weighted mode with GUI-entered weights. |
| density_salience_threshold_db | -35 / -45 / -55 | More permissive threshold increases or preserves salience-based means globally | means(final,H,I,S)=[(7.0324962235159205, 5.032561196606315, 3.9998700538192153, 0.0), (9.040115421217653, 6.359798633049076, 5.257385982924053, 0.2064951868262109), (11.167113070821305, 7.526130008810543, 6.665970617858622, 1.231991012325806)] | PASS | final_note_density_salience_weighted, harmonic_density_component, inharmonic_density_component, subbass_density_component | Threshold sweep via GUI controls. |
| density_frequency_ceiling_hz | 3000 / 5000 / 8000 | Ceiling-aware counts increase or remain stable with higher ceiling | mean(H_ceiling_alias)=[7.8, 12.0, 13.6]; mean(I_ceiling_alias)=[11.0, 11.0, 11.0] | PASS | salient_harmonic_order_count_up_to_density_ceiling_hz, salient_inharmonic_log_bin_count_up_to_density_ceiling_hz | Ceiling-aware aliases checked; no reinterpretation in *_up_to_5000hz columns. |
| Metadata propagation | density_summation_mode, harmonic_density_weight, inharmonic_density_weight, subbass_density_weight, density_salience_threshold_db, density_frequency_ceiling_hz, window_type, n_fft, hop_length, zero_padding, harmonic_tolerance, frequency_min_hz, frequency_max_hz, magnitude_min_db | Required GUI settings present and non-blank (or unavailable_not_recorded) | all required keys present | PASS | density_summation_mode, harmonic_density_weight, inharmonic_density_weight, subbass_density_weight, density_salience_threshold_db, density_frequency_ceiling_hz, window_type, n_fft, hop_length, zero_padding, harmonic_tolerance, frequency_min_hz, frequency_max_hz, magnitude_min_db | Checked on full clarinet run. |
| log density config | gui_worker.log run header | Final density config block logged; old confusing placeholder line removed/relabelled | config block present and old phrase removed | PASS | - | `<analysis_log_path>/gui_worker.log` |
| workbook propagation | final acceptance workbooks (clarinet + cello, compiled + research) | Required final-density fields populated in `Spectral_Density_Metrics`, `Charts_Data`, and `Metadata` on release artifacts | clarinet=PASS, cello=PASS (all central fields populated); no unresolved central propagation gaps | PASS | final_note_density_count_based, final_note_density_salience_weighted, final_note_density_salience_weighted_norm_for_chart, salient_harmonic_order_count_up_to_5000hz, salient_harmonic_order_count_up_to_density_ceiling_hz, salient_inharmonic_log_bin_count_up_to_5000hz, salient_inharmonic_log_bin_count_up_to_density_ceiling_hz, salient_subbass_particle_count, harmonic_density_component, inharmonic_density_component, subbass_density_component, harmonic_density_weight, inharmonic_density_weight, subbass_density_weight, density_summation_mode, density_salience_threshold_db, density_frequency_ceiling_hz | Policy B applied for Charts_Data: when `density_frequency_ceiling_hz=5000`, fixed `salient_inharmonic_log_bin_count_up_to_5000hz` is the release plotting field; ceiling-aware inharmonic alias remains in `Spectral_Density_Metrics`. |

## Expected Final Statuses

- density_summation_mode: **PASS**
- density weights: **PASS**
- density_salience_threshold_db: **PASS**
- density_frequency_ceiling_hz: **PASS**
- Metadata propagation: **PASS**
- log density config: **PASS**
- workbook propagation: **PASS**

## Notes

- The old ambiguous line `Model-weight placeholder: H=0.500, I=0.500` is no longer emitted as-is; logs now include explicit final density config keys.
- Ceiling behavior is validated on `*_up_to_density_ceiling_hz` columns to avoid overloading `*_up_to_5000hz` names.
- Provenance-only metadata patch validated on:
  - `compiled_density_metrics_research_final_density_acceptance_metadatafix_provenance.xlsx` (clarinet)
  - `compiled_density_metrics_research_final_density_acceptance_metadatafix_provenance.xlsx` (cello)
- Frequency/magnitude recovery outcome:
  - `frequency_min_hz`, `frequency_max_hz`, `magnitude_min_db`, `magnitude_max_db` remain `unknown_not_parseable` only where unavailable in compiled sources.
  - Metadata now records `frequency_magnitude_fields_recovery_status = partially_unavailable_in_compiled_source` when any of those fields cannot be recovered.
- Final-density invariance across metadata-only patch:
  - `max|Δ final_note_density_count_based| = 0.0`
  - `max|Δ final_note_density_salience_weighted| = 0.0`
