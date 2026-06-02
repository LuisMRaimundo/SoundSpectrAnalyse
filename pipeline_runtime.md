# Pipeline Runtime (Strict Path)

This is the filtered version containing only scripts used in the default runtime execution path (GUI/orchestrator), excluding tests and non-runtime helper scripts.

## Runtime entry paths

| Entry mode | Entrypoint | Immediate orchestrator |
|---|---|---|
| CLI (default orchestrator path) | `run_orchestrator.py:main()` | `pipeline_orchestrator_integrated.py:RobustOrchestrator.run_complete_pipeline()` |
| GUI orchestrator path | `pipeline_orchestrator_gui.py:RobustOrchestratorApp._process_folder_complete_pipeline()` | in-file stage orchestration |

## Phase flow in strict runtime path

| Runtime phase | What executes | Output artifact |
|---|---|---|
| Stage 1A Input/tier assignment | file discovery, note parsing, FFT tier assignment | in-memory file plan |
| Stage 1B Per-note analysis | `proc_audio.AudioProcessor` + core acoustic/density modules | per-note `spectral_analysis.xlsx` |
| Stage 1C Adaptive update | `adaptive_density_engine.AdaptiveDensityEngine.update` fed by pure observation | `adaptive_density_engine_state.json`, phase profile CSV/JSON |
| Stage 2 Compilation | `compile_metrics.compile_density_metrics_with_pca` and `_write_compiled_excel` | `compiled_density_metrics.xlsx` |
| Stage 2B Publication/research curation | metadata/column policies applied in compile/export helpers | curated sheets (`Density_Metrics`, `Canonical_Metrics`, etc.) |
| Stage 3 EWSD + research export | `post_compile_research_export` → `export_research_workbook` + `ewsd_research_integration` | `compiled_density_metrics_research.xlsx` with EWSD columns |

## Scripts actually traversed

| Script | Runtime functions/classes on path | What happens there | Core formulas (code) | Mathematical form (LaTeX) |
|---|---|---|---|---|
| `run_orchestrator.py` | `main`, `_reject_legacy_cli_flags` | CLI arg parsing and orchestration start | `orchestrator.run_complete_pipeline()` | N/A (control flow) |
| `pipeline_orchestrator_integrated.py` | `RobustOrchestrator.run_stage1_per_note_analysis`, `run_stage2_compilation`, `run_complete_pipeline` | Three-stage orchestration and handoff to Stage 1/2/3 engines | `compiled_df = compile_density_metrics_with_pca(...)`; `run_research_workbook_export(...)` | N/A (pipeline control) |
| `pipeline_orchestrator_gui.py` | `build_phase1_file_iteration_order`, `_process_folder_complete_pipeline` | GUI path orchestration, deterministic ordering, adaptive profile management | `sorted(... note_to_hz(...))` | Deterministic ordering by ascending \(f_0\) |
| `proc_audio.py` | `AudioProcessor` methods (peak extraction, f0 refine, per-note export) | Per-note spectral analysis and workbook writing | `density_raw = D_H*w_H + D_I*w_I + D_S*w_S` | \(D_{raw}=D_Hw_H + D_Iw_I + D_Sw_S\) |
| `acoustic_density_core.py` | `compute_acoustic_density_descriptors`, `canonical_f0_triplet`, `_expected_harmonic_orders`, `_expected_residual_bin_count` | Computes H/I/S observation triplet, density components, inharmonicity-aware descriptors | `data_ratio = component_strength / sum(component_strength)` | \(w_k = s_k/\sum_j s_j\) |
| `acoustic_density_core.py` | `compute_acoustic_density_descriptors` (Phase 7 block) | Register-invariant strength normalization by slot capacity | `h_term = clip(h_density/N_h,0,1)+λ_H*clip(h_count/N_h,0,1)` | \(s_H = \mathrm{clip}(D_H/N_H)+\lambda_H\mathrm{clip}(C_H/N_H)\) |
| `adaptive_density_engine.py` | `AdaptiveDensityEngine.update`, `_js_divergence` | Online posterior-like update with divergence reliability gate | `reliability = exp(-jsd/temp)` | \(r=e^{-\mathrm{JSD}(o,p)/T}\) |
| `subbass_policy.py` | `SubBassPolicy.upper_bound_hz` | Canonical sub-bass upper limit | `min(f0_hz*0.5, 80.0)` | \(f_{SB}^{max}=\min(0.5f_0,80)\) |
| `inharmonicity_model.py` | `fit_inharmonicity_coefficient` | Fits stiff-string \(B\) for stretched harmonic grid | `f_n = n*f0*sqrt(1+B*n*n)` | \(f_n=nf_0\sqrt{1+Bn^2}\) |
| `density.py` | `apply_density_metric`, `partial_metric_sums_h_i_s_total`, `partial_density_effective_components_bundle` | Weight-function spectral density operators and effective-component metrics | `n_eff = (sum(p)^2)/sum(p*p)` | \(N_{eff}=\frac{(\sum p_i)^2}{\sum p_i^2}\) |
| `harmonic_alignment.py` | `compute_harmonic_alignment_metrics` | Cents-domain alignment and slot/tolerance checks | `cents = 1200*log2(f/f_ref)` | \(\Delta_c = 1200\log_2(f/f_{ref})\) |
| `mir_descriptors.py` | `compute_mir_descriptors_from_spectrum` | Whole-note MIR descriptors | `centroid = sum(f*p)` | \(\mu_f=\sum_i f_i p_i\) |
| `temporal_segmentation.py` | `segment_attack_sustain_release` | Envelope segmentation and log attack time | `log_attack_time_s = log10(max(t_attack,eps))` | \(LAT=\log_{10}(t_{attack})\) |
| `spectral_normalization.py` | `n_fft_normalization_factor` | Tier normalization factors used in compile output (`quantity_kind` contract) | `peak_amplitude_sum: N_ref/N; peak_power_sum: (N_ref/N)^2` | \(k_{peak\_amp}=N_{ref}/N,\;k_{peak\_pow}=(N_{ref}/N)^2\) |
| `compile_metrics.py` | `compile_density_metrics_with_pca`, `_compile_density_metrics_impl`, `_write_compiled_excel`, `_build_density_metrics_main_sheet`, `_build_density_metrics_sheet_from_per_note_files` | Stage-2 aggregation, direct per-note extraction, canonical + legacy sheet writing | `raw_per_note = D_H*wH_per + D_I*wI_per + D_S*wS_per` | \(D_{per}=D_Hw_H^{(n)}+D_Iw_I^{(n)}+D_Sw_S^{(n)}\) |
| `post_compile_research_export.py` | `run_research_workbook_export` | Stage 3 hook after compile; triggers research workbook + EWSD merge | delegates to `export_research_workbook` | N/A (control flow) |
| `tools/export_research_density_workbook.py` | `export_research_workbook`, `build_workbook`, `merge_ewsd_into_spectral_density_metrics` (via integration) | Research workbook assembly and EWSD left-join | EWSD: \(\sum_k r_k D_k (N_{eff,k}/N_k)\) | `EWSD_score_total`, `EWSD_score_acoustic_balanced` |
| `tools/ewsd_core.py` | `compute_ewsd`, `add_acoustic_alignment_columns`, `add_quality_columns` | EWSD-R v18 core from per-note component spectra | participation-ratio penalty per H/I/S compartment | Stage 3 research columns |
| `tools/ewsd_research_integration.py` | `discover_individual_exact_workbooks`, `merge_ewsd_into_spectral_density_metrics` | Discover workbooks, compute EWSD, merge on Note | left join + `ewsd_primary_analysis_eligible` gate | provenance columns |
| `metadata_sanitizer.py` | publication redaction/clean functions used during write | Removes private paths/noise from published exports | `sanitize_dataframe_for_publication(df)` | N/A (policy transform) |
| `publication_metric_columns.py` | `filter_dataframe_for_publication_metrics_sheet` | Final column filtering for publication sheet | allow-list filter | N/A (deterministic set projection) |
| `note_parser.py` | `canonical_note_from_filename`, `parse_note_token` | Canonical note parsing used by orchestrator and compile | parser token rules | N/A (grammar mapping) |

## Included vs excluded in this strict report

- Included: modules traversed by default stage execution (CLI orchestrator and GUI orchestrator paths) and their directly invoked computational dependencies.
- Excluded: test modules, standalone validation/report CLIs not required by default run (`validate_canonical_metrics.py`, `verify_runtime_schema.py`, etc.), and utility/example scripts not used during normal execution.
