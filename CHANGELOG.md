# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Hutchinson-Knopoff provenance artefacts:** moved the default `g(y)` lookup knots to `data/hk1978_g_table.csv` (import-time `numpy.loadtxt`), exposed `dissonance_models.HK_G_TABLE_PROVENANCE`, and added `tests/test_hk_g_table_provenance.py` (provenance non-empty + piecewise monotonic checks).
- **Dissonance vectorisation guard test:** added `tests/test_dissonance_vectorisation_equivalence.py` to assert numerical equivalence (`atol=1e-12`, `rtol=1e-10`) between broadcast/index-based pair evaluation and the previous nested-loop reference.
- **Public API reachability test:** added `tests/test_public_api_importable.py` to validate exported names in `density.__all__` and `dissonance_models.__all__`.
- **Reference register:** added root `REFERENCES.md` with full APA entries used by in-code short citations and naming disclosures.

### Changed

- **Pairwise dissonance internals:** replaced `for i / for j` nested loops in `dissonance_models.py` (`_dissonance_total_and_pairs`, `_dissonance_total_pairs_and_minamp`, `SetharesDissonance._pairwise_sum`) with upper-triangular index arrays (`np.triu_indices`) and pair-array evaluation.
- **Citation metadata alignment:** updated `CITATION.cff` author metadata/ORCID, funding DOI identifier, and set `version: "3.7.0"` to match the current package version and README acknowledgements block.
- **Repository spelling hygiene:** normalized installer path naming to `installers/` and updated installer README path references accordingly.

### Deprecated

- **`bin_width_hz` in `physical_spectral_density`:** parameter is retained for signature compatibility but now emits `DeprecationWarning` when non-`None`; removal is scheduled for a future **4.x** release.

### Documentation

- Added explicit naming disclosure for `effective_partial_density` (`N_eff / N`, Hill q=2 / inverse Herfindahl) in `density.py` and `docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md` under **"Naming caveat: effective_partial_density vs. classical density"**, including divergence note from Krimphoff/Peeters-style density framings.
- Added model-constant citation comments in `dissonance_models.py` (Sethares; Vassilakis) and explicit "unconstrained design choice" annotation for perceptual density mixing weights (`0.6/0.4`).

### Added

- **`Legacy_Density_Metrics` (per-note export, default ON):** every **`spectral_analysis.xlsx`** now includes a dedicated sheet with **`Density Metric`**, **`Spectral Density Metric`**, **`Filtered Density Metric`**, **`Combined Density Metric`**, and **`spectral_masking_enabled`** (`False` — no v5 masking GUI in v6). Stage 2 **`read_excel_metrics`** merges this sheet so **`Weighted Combined Metric`** is recomputed from real SDM/FDM on **`Diagnostic_Metrics`** / **`Legacy_Compatibility`**, not from zero placeholders.
- **Research workbook (`compiled_density_metrics_research.xlsx`) historical note:** this changelog entry describes the behavior at the time of that release. In the current contract, **`Combined Density Metric`** remains legacy-only (not primary `Spectral_Density_Metrics`), and **`density_weighted_sum_cdm_mean`** is legacy opt-in only.

### Fixed

- **`density_weighted_sum` (Stage 2 compile):** now uses per-band **`harmonic_density_sum` / `inharmonic_density_sum` / `subbass_density_sum`** under the compile **`weight_function`** (same formula as **`density_metric_raw`**), instead of always using linear **`harmonic_amplitude_sum`** × energy ratios. **`harmonic_amplitude_sum`** remains an unchanged linear diagnostic. Tests: **`tests/test_weighted_note_density.py::test_a2_density_weighted_sum_follows_weight_function`**. Docs: **`docs/DENSITY_EXPORT_SCHEMA.md`** §C.1, Pass 14 formula tables, **`metrics_dictionary.json`**, **`Compile_Guide`** text, research export README prose.

### Portability

- **Repository paths & orchestrator entry scripts:** renamed legacy batch directory to **`audio_analysis/`** (ASCII); renamed **`robust_orchestrator_v2_16.py`** → **`pipeline_orchestrator_gui.py`** and **`robust_orchestrator_integrated.py`** → **`pipeline_orchestrator_integrated.py`** with imports, docs, `run.bat`, and `pyproject.toml` updated. User-facing copy now refers to the **pipeline orchestrator** where appropriate. See **`REPOSITORY_NAMING_AND_LANGUAGE_SYNC_REPORT.md`**.

### Documentation

- **Formula extraction & formula validation (Passes 1–15):** completed pytest coverage under **`tests/formula_validation/`** aligned with **`docs/formula_extraction/`** and **`docs/formula_validation/`** plans. Status: **`docs/validation/VALIDATION_STATUS.md`**. Methodology: **`docs/validation/METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md`**. Optional traceability: **`CODE_FORMULA_TRACEABILITY_TABLE.md`**. Indexes: **`docs/formula_extraction/FORMULA_EXTRACTION_INDEX.md`**, **`docs/formula_validation/FORMULA_VALIDATION_PLAN_INDEX.md`**. Cross-linked **`README.md`**, **`docs/CURRENT_DOCUMENTATION_INDEX.md`**, **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`** §10, **`TECHNICAL_MANUAL.md`** §10.3, **`TESTING.md`**.
- **Mathematical formalisation (first pass):** added **`docs/MATHEMATICAL_FORMALISATION_VERIFICATION_REPORT_FIRST_PASS.md`** (LaTeX formulas, code-grounded verification for six `density.py` functions) and linked it from **`docs/CURRENT_DOCUMENTATION_INDEX.md`**.

- **Research export (`compiled_density_metrics_research.xlsx`):** documented Excel-safe behaviour (worksheet **AutoFilter** only—no ListObject **Table** XML), optional CLI metadata, and column-header sanitisation. Cross-linked **`README.md`**, **`docs/CURRENT_DOCUMENTATION_INDEX.md`**, **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`**, **`TECHNICAL_MANUAL.md`** §5.3, **`API_REFERENCE.md`**, **`TESTING.md`**, and the **`tools/export_research_density_workbook.py`** module docstring.

- **Computational metrics inventory:** added **`docs/COMPUTATIONAL_METRICS_CODE_REVIEW_REPORT.md`** (read-only survey of project-owned maths/metrics code; black-boxes NumPy/SciPy/librosa internals) and indexed it from **`docs/CURRENT_DOCUMENTATION_INDEX.md`** and **`README.md`**.

- **Component balance pies (Stage 1):** documented the split between **`component_amplitude_mass_pie.png`** (linear amplitude-sum diagnostic, with explicit basis in title/footnote), **`component_energy_ratio_pie.png`** (`harmonic_energy_ratio` / `inharmonic_energy_ratio` / `subbass_energy_ratio`), and legacy-alias **`component_energy_pie.png`**; per-note **`Analysis_Metadata`** keys `amplitude_mass_chart_*` and `energy_ratio_chart_*`. Updated **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`**, **`docs/DENSITY_EXPORT_SCHEMA.md`** §J, **`TECHNICAL_MANUAL.md`**, **`API_REFERENCE.md`**, **`README.md`**, **`TESTING.md`**, **`docs/CURRENT_DOCUMENTATION_INDEX.md`**, and **`audio_analysis/README_SUPER_ANALYZER.md`**. Tests: **`tests/test_component_balance_pies.py`**.

- Synchronised documentation after pipeline stabilisation (audit 2026-05-13): canonical **Stage 1** `proc_audio.AudioProcessor` and **Stage 2** `compile_metrics.compile_density_metrics_with_pca`; per-note **`spectral_analysis.xlsx`** and compiled **`compiled_density_metrics.xlsx`**; **f0** provenance and nominal-vs-final distinction; **sub-bin harmonic frequency** export semantics (`bin_center_frequency_hz` vs `interpolated_frequency_hz`); **adaptive subfundamental** metadata (`subfundamental_margin_percent` vs `effective_subfundamental_margin_percent`, selection fields, `min_floor_hz`, `max_fraction_of_f0`); **nonharmonic** terminology and candidate vs strict partial wording; **Debug_Counts** hierarchy vs independent `peaklist_*` counts; **missing-value** policy (NaN vs 0.0, `Index_Weighted` available-term renormalisation); **legacy** `super_audio_analyzer` / `super_analysis_results.json` scoped as optional/diagnostic; **final workbook audit** tool `tools/audit_compiled_workbook.py` and tests `tests/test_final_pipeline_invariants.py` / `tests/pipeline_workbook_audit.py`. Added **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`**, **`docs/CURRENT_DOCUMENTATION_INDEX.md`**, **`docs/DOCUMENTATION_AUDIT_REPORT.md`**; superseded **`PIPELINE_FUNCTIONING_REPORT`** narrative was later removed from the tree (root stub + git history). See **`docs/DOCUMENTATION_AUDIT_REPORT.md`** for per-file classification.

- **Retired snapshot docs:** removed all **`*.md`** / **`*.txt`** that lived under **`Backup/`**, removed the now-empty **`Backup/`** tree, and removed **`docs/archive_obsolete/`** (housekeeping). Updated **`README.md`**, **`PIPELINE_FUNCTIONING_REPORT.md`**, **`docs/CURRENT_DOCUMENTATION_INDEX.md`**, **`docs/DOCUMENTATION_AUDIT_REPORT.md`**, **`audio_analysis/README_SUPER_ANALYZER.md`** so links and the audit register match the slimmer tree; use **git history** for old snapshot wording.
