# Pipeline

## Scope

This document covers operational scripts in `SoundSpectrAnalyse` (runtime, orchestration, validation, export, GUI worker, and support modules), excluding test files.

## Phase map (end-to-end runtime)

| Phase | What happens | Main scripts |
|---|---|---|
| P0 Intake/IO | Audio and metadata are loaded, IDs/paths are normalized/sanitized | `audio_utils.py`, `public_audio_identifiers.py`, `export_paths.py`, `metadata_sanitizer.py` |
| P1 Per-note spectral extraction | Peaks, harmonic/non-harmonic split, f0 provenance, component energies, density descriptors are computed | `proc_audio.py`, `acoustic_density_core.py`, `density.py`, `harmonic_alignment.py`, `harmonic_validation.py`, `spectral_leakage_guards.py`, `peak_component_counts.py` |
| P2 Adaptive profile | Pure per-note observation triplet updates adaptive H/I/S profile | `adaptive_density_engine.py`, `pipeline_orchestrator_gui.py` |
| P3 Tier normalization | Cross-note FFT-tier normalization for absolute amplitude/power sums | `spectral_normalization.py`, `compile_metrics.py` |
| P4 Inharmonicity model | Stiff-string `B` coefficient fit and adaptive harmonic tolerance behavior | `inharmonicity_model.py`, `acoustic_density_core.py`, `proc_audio.py`, `compile_metrics.py` |
| P5 MIR + temporal | MPEG-7-like descriptors and attack/sustain/release segmentation | `mir_descriptors.py`, `temporal_segmentation.py`, `proc_audio.py`, `compile_metrics.py` |
| P6 Provenance + consolidation | Constants provenance registry, alias consolidation, curated exports | `constants.py`, `docs/CONSTANTS_PROVENANCE.md`, `compile_metrics.py`, `post_compile_research_export.py` |
| P7 Register-invariant strength | Occupancy-normalized H/I/S strength to remove register drift | `acoustic_density_core.py`, `constants.py`, `compile_metrics.py` |
| Final compile/export | Build compiled workbook and side sheets; publication/research variants | `compile_metrics.py`, `publication_metric_columns.py`, `publication_chart_policy.py`, `post_compile_research_export.py` |

## Core computational scripts (functions, summaries, formulas)

| Script | Functions (key names) | Brief function summary | Code formula (snippet) | Mathematical form (LaTeX) | Phase behavior |
|---|---|---|---|---|---|
| `acoustic_density_core.py` | `canonical_f0_triplet`, `_expected_harmonic_orders`, `_expected_residual_bin_count`, `compute_acoustic_density_descriptors` | Builds per-note acoustic descriptors, f0 epistemic state, H/I/S observation weights, diagnostics, inharmonicity outputs | `data_ratio = component_strength / sum(component_strength)` | `w_k = \\frac{s_k}{\\sum_j s_j}` | P1/P4/P7: main per-note descriptor engine |
| `acoustic_density_core.py` | `compute_acoustic_density_descriptors` | Phase 7 register-invariant strength using occupancy-normalized terms | `h_term = clip(h_density/expected_h_slots,0,1) + wH_occ*clip(h_count/expected_h_slots,0,1)` | `s_H = \\frac{D_H}{N_H}+\\lambda_H\\frac{C_H}{N_H}` (clipped to [0,1] per component term) | P7: removes register bias from incommensurate alphabets |
| `adaptive_density_engine.py` | `_normalize_triplet`, `_js_divergence`, `AdaptiveDensityEngine.update` | Online profile update from pure observation with divergence-based reliability gate | `reliability = exp(-jsd/temp); alpha=(1-forgetting)*alpha + gain*obs` | `r=e^{-\\mathrm{JSD}(o,p)/T},\\;\\alpha'=(1-\\rho)\\alpha+g\\,o` | P2: adaptive corpus-profile learning |
| `subbass_policy.py` | `SubBassPolicy.upper_bound_hz` | Canonical sub-bass upper boundary policy | `min(f0_hz*0.5, 80.0)` | `f_{SB}^{max}=\\min(0.5f_0,80\\,\\mathrm{Hz})` | P2: single source for sub-bass semantics |
| `spectral_normalization.py` | `n_fft_normalization_factor` | FFT-tier normalization factor with quantity semantics | `peak_amplitude_sum: n_ref/n_fft; peak_power_sum: (n_ref/n_fft)^2` | `k_{peak\\_amp}=N_{ref}/N,\\;k_{peak\\_pow}=(N_{ref}/N)^2` | P3: cross-note tier invariance (Phase 8 contract) |
| `inharmonicity_model.py` | `fit_inharmonicity_coefficient` | Fits stiff-string inharmonicity coefficient `B` and residual spread | `f_n = n*f0*sqrt(1+B*n*n)` | `f_n=nf_0\\sqrt{1+Bn^2}` | P4: physically grounded inharmonicity modeling |
| `mir_descriptors.py` | `compute_mir_descriptors_from_spectrum`, `_roughness_aures_1985`, `_erb_rate_hz` | Computes spectral moments, tristimulus, flatness, rolloff, roughness, ERB-weighted descriptors | `centroid = sum(f*p)` | `\\mu_f=\\sum_i f_i p_i` | P5: descriptor extension |
| `temporal_segmentation.py` | `segment_attack_sustain_release` | Envelope-based segmentation and log attack time extraction | `log_attack_time_s = log10(max(attack_s,eps))` | `LAT=\\log_{10}(t_{attack})` | P5: temporal decomposition |
| `density.py` | `apply_density_metric`, `partial_metric_sums_h_i_s_total`, `compute_harmonic_occupancy_ratio`, `compute_rolloff_compensated_harmonic_density`, `partial_density_effective_components_bundle` | Canonical density operators, H/I/S sums, occupancy, effective-component calculations, weight-function family | `n_eff = (sum(p)^2)/sum(p*p)` | `N_{eff}=\\frac{(\\sum_i p_i)^2}{\\sum_i p_i^2}` | P1/P3: foundational density math |
| `harmonic_alignment.py` | `compute_harmonic_alignment_metrics` | Cents-domain harmonic-order alignment with adaptive tolerance windowing | `cents = 1200*log2(f_meas/f_ref)` | `\\Delta_c=1200\\log_2\\left(\\frac{f}{f_{ref}}\\right)` | P1/P4: harmonic-fit quality metrics |
| `low_frequency_policy.py` | `calculate_adaptive_subfundamental_cutoff_hz`, `classify_low_frequency_row` | Low-frequency guard policy and row-level LF classification | `cutoff = SubBassPolicy.upper_bound_hz(...)` | `f_{LF}^{cut}=f_{SB}^{max}` (delegated) | P2: compatibility shim around canonical sub-bass policy |
| `proc_audio.py` | `AudioProcessor` methods (peak extraction, f0 refine, per-note exports) | End-to-end per-note analysis pipeline, writes spectral/workbook outputs including inharmonicity fit sheet | `weighted = D_H*w_H + D_I*w_I + D_S*w_S` | `D_{raw}=D_Hw_H+D_Iw_I+D_Sw_S` | P1/P4/P5: per-note runtime execution |
| `compile_metrics.py` | `_compile_density_metrics_impl`, `_write_compiled_excel`, `_build_density_metrics_main_sheet`, `_build_density_metrics_sheet_from_per_note_files`, `_compute_weighted_density_columns_for_wide_df` | Corpus compile path, direct per-note extraction, weighted density reconciliation, tier-normalized companion columns, workbook assembly | `density_metric_raw_per_note_balance = D_H*w_H_per + D_I*w_I_per + D_S*w_S_per` | `D_{per}=D_Hw_H^{(n)}+D_Iw_I^{(n)}+D_Sw_S^{(n)}` | P2/P3/P6/P7 + final export |
| `run_real_corpus_validation.py` | `run_real_corpus_validation`, `analyze_audio_file`, `build_canonical_dataframe`, `write_canonical_workbook` | Batch/corpus validation runner with canonical output and reporting | `coverage = non_null/required` | `\\mathrm{coverage}=\\frac{n_{valid}}{n_{required}}` | Validation phase after pipeline runs |

## Operational scripts inventory (all non-test operational modules)

| Script | Main functions/classes | What it does | Key calc/algorithm |
|---|---|---|---|
| `acoustic_data_analysis_suite.py` | `AcousticDataAnalyzer`, `MultiFileComparator`, `run_gui`, `main` | GUI analysis suite entrypoint and multi-file comparison | Statistical comparison/orchestration |
| `analysis_policy.py` | constants-only module | Shared analysis policy flags/metadata | No core numeric formula |
| `audio_utils.py` | `load_audio_with_fallback`, unit conversions | Audio loading, dB/power/amplitude/cents conversion helpers | `db=20log10(A)`; `P=A^2` |
| `constants.py` | constants + shims | Numerical policy constants, deprecations, provenance warnings | Constant definitions |
| `data_integrity.py` | normalization/validation helpers | Metric coercion, outlier detection, robust scaling | IQR/outlier and robust normalization |
| `debug_counts.py` | `validate_debug_count_invariants` | Debug count consistency checks | Invariant checks |
| `dissonance_export.py` | canonical dissonance frame builders | Export harmonized dissonance comparison sheets | Correlation/reshape ops |
| `dissonance_models.py` | `SetharesDissonance`, `HutchinsonKnopoffDissonance`, `VassilakisDissonance` | Pairwise sensory dissonance models | Model-specific pairwise kernels |
| `energy_accounting.py` | `describe_component_energy_balance` | Summarizes component energy balance | Ratio/accounting |
| `export_paths.py` | path redaction/sanitization | Publication-safe path handling | String/path sanitation |
| `gui_compile_stage2_worker.py` | `main` | Stage-2 compile worker entry | Orchestration |
| `gui_model_weight_policy.py` | `resolve_analysis_model_weights` | Resolves GUI model-weight policy | Rule resolution |
| `harmonic_validation.py` | `validate_harmonic_series_matched` | Harmonic series validation utilities | Tolerance-window checks |
| `interface.py` | `SpectrumAnalyzer`, workers | Main desktop UI implementation | UI/worker orchestration |
| `log_config.py` | `configure_root_logger` | Logging bootstrap | Logging config |
| `main.py` | `main` | Primary app entrypoint | Bootstrapping |
| `metadata_sanitizer.py` | publication sanitation functions | Redaction, clean exports, leakage checks | Policy-driven transformations |
| `metric_contract.py` | contract builders/classifiers | Canonical metric definitions and metadata contracts | Contract mapping |
| `note_parser.py` | note token parsing functions | Canonical note parsing from filenames | Token parsing/state machine |
| `peak_component_counts.py` | `classify_peaks_harmonic_inharmonic_subbass_from_df` | Peak-table H/I/S classification counts | Frequency-to-class mapping |
| `pipeline_contract.py` | `PipelineContract` | Canonical pipeline contract container | Contract serialization |
| `pipeline_orchestrator_gui.py` | `RobustOrchestratorApp` + helpers | Full GUI orchestration over Phase 1/2/compile/export | Stage orchestration and adaptive handoff |
| `pipeline_orchestrator_integrated.py` | `RobustOrchestrator`, `main` | Integrated orchestrator runner | CLI orchestration |
| `post_compile_research_export.py` | `run_research_workbook_export` | Generates research workbook variant after compile | Export transform |
| `public_audio_identifiers.py` | hashing and ID builders | Stable public IDs and sample identifiers | SHA-256 based ID construction |
| `publication_chart_policy.py` | chart metric policies | Publication-safe metric selection and warnings | Policy rules |
| `publication_metric_columns.py` | metrics-sheet filters | Column allow-listing for publication sheets | Deterministic filtering |
| `reference_signal_utils.py` | synthetic signal generators | Reference sine/harmonic/inharmonic synthetic signals | Signal synthesis formulas |
| `result_cache.py` | `ResultCache`, `get_cache` | Runtime caching utility | Key/value cache behavior |
| `run_orchestrator.py` | `main` | CLI orchestrator entry script | Runner bootstrapping |
| `runtime_versions.py` | runtime fingerprint functions | Runtime package/version fingerprinting | Metadata generation |
| `spectral_leakage_guards.py` | leakage guard functions | Leakage halfwidth and candidate filtering | Frequency-window rejection |
| `validate_canonical_metrics.py` | validation/report generators | Canonical workbook validation and report generation | Coverage/stats/PCA checks |
| `verify_runtime_schema.py` | `_main` | Runtime schema verifier CLI | Contract verification |
| `weight_function_ui_labels.py` | label mapping functions | UI label <-> weight-key mapping | Mapping logic |
| `audio_analysis/super_audio_analyzer.py` | `SuperAudioAnalyzer`, export helpers | Super-analysis engine and summaries | Aggregation and summary formulas |
| `audio_analysis/super_audio_analyzer_gui.py` | `SuperAudioAnalyzerGUI`, workers | GUI front-end for super analyzer | GUI orchestration |
| `audio_analysis/batch_audio_analyzer.py` | `BatchAudioAnalyzer`, `main` | Batch analysis runner and row deduplication | Batch traversal |
| `audio_analysis/batch_example.py` | `main` | Example batch execution script | Example orchestration |

## Notes on formula coverage

For scripts that are orchestration-only, policy-only, or metadata-only, no core independent mathematical kernel exists in that file; formulas are delegated to core computational modules (`acoustic_density_core.py`, `density.py`, `inharmonicity_model.py`, `compile_metrics.py`, `mir_descriptors.py`, `temporal_segmentation.py`).
