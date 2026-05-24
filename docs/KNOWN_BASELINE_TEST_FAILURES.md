# Known Baseline Test Failures

Baseline reference: publication-gate baseline for this repaired release.
Current verification run: `python -m pytest tests -q` -> `15 failed, 850 passed, 40 skipped`.

| Test | Baseline | Repaired | Why unrelated to V4 repair | Blocks publication/export use |
|---|---:|---:|---|---:|
| `tests/formula_validation/test_formula_validation_pass_14_compile_extraction_and_batch_mass.py::test_extract_density_component_sum_log` | fail | fail | Existing compile extraction formula guard unrelated to final-density wiring/provenance patch | no |
| `tests/test_benchmarks.py::TestBenchmarks::test_benchmarks` | fail | fail | Benchmark fixture/environment dependency | no |
| `tests/test_density_metric_correction.py::test_extract_density_component_sum_log` | fail | fail | Existing density extraction correction path | no |
| `tests/test_density_metric_correction.py::test_log_mode_must_not_pick_power_raw_even_when_present` | fail | fail | Existing log extraction selector behavior | no |
| `tests/test_density_metric_correction.py::test_extract_density_component_sum_honours_include_for_density_log` | fail | fail | Existing include-for-density handling | no |
| `tests/test_density_metric_correction.py::test_extract_density_component_sum_legacy_when_column_absent` | fail | fail | Existing legacy-column fallback behavior | no |
| `tests/test_density_metric_correction.py::test_compiled_row_carries_inclusion_diagnostics` | fail | fail | Existing compile diagnostics contract | no |
| `tests/test_density_metric_correction.py::test_compiled_density_metric_raw_matches_audit_formula` | fail | fail | Existing weighted extraction parity check | no |
| `tests/test_density_metric_correction.py::test_gui_activation_log_drops_legacy_0p95_0p05_string` | fail | fail | Existing legacy GUI log text policy regression | no |
| `tests/test_density_metrics_component_basis.py::test_C_power_raw_only_under_explicit_debug_basis` | fail | fail | Existing component-basis policy regression | no |
| `tests/test_density_metrics_component_basis.py::test_E_huge_subbass_power_raw_does_not_affect_density_metric_raw` | fail | fail | Existing power-vs-amplitude basis behavior | no |
| `tests/test_external_validation_marketing_ban.py::test_batch_super_analysis_json_samples_clean` | fail | fail | Existing content policy fixture text mismatch | no |
| `tests/test_external_validation_marketing_ban.py::test_batch_metrics_summary_txt_samples_clean` | fail | fail | Existing content policy fixture text mismatch | no |
| `tests/test_forbidden_legacy_tokens.py::test_pipeline_orchestrator_gui_live_widget_text_has_no_legacy_strings` | fail | fail | Existing legacy token cleanup debt in live widget text | no |
| `tests/test_inharmonic_energy_audit.py::test_extractor_power_sum_debug_basis_selects_power_raw` | fail | fail | Existing extractor basis selection issue | no |

No newly introduced full-suite failures are accepted beyond this documented baseline set.
