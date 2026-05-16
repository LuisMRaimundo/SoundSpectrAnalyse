# Current documentation index

**Last audit:** 2026-05-14 (index + research-export / metrics-report supplement; prior full pass 2026-05-13). **Formula validation:** Passes **1–15** completed in pytest (`tests/formula_validation/`); see **`VALIDATION_STATUS_812_PASSED_PASSES_1_15.md`** and **`METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md`**.  
**Pipeline contract (high level):** `proc_audio.AudioProcessor` → per-note `spectral_analysis.xlsx` → `compile_metrics.compile_density_metrics_with_pca` → `compiled_density_metrics.xlsx`  
**Schema / export version strings:** see per-workbook `Analysis_Metadata` (`ANALYSIS_SCHEMA_VERSION`, `pipeline_contract_version`, `export_schema_version`, etc.) — this index does not duplicate those runtime values.

---

## Safe to cite for academic or software documentation (canonical)

| Document | Purpose |
|----------|---------|
| [docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md](CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md) | **Normative** summary: canonical stages, f0 fields, harmonic interpolation columns, nonharmonic hierarchy, subfundamental vs sub-bass, **per-note component pie filenames and bases** (amplitude-mass vs energy-ratio vs legacy alias), Debug_Counts, missing metrics, audit CLI/tests. |
| [docs/DENSITY_EXPORT_SCHEMA.md](DENSITY_EXPORT_SCHEMA.md) | **Authoritative** workbook layout: `Density_Metrics`, `Canonical_Metrics`, `Diagnostic_Metrics`, `Debug_Counts`, PCA/dissonance separation, redaction. **§C.1** documents `density_weighted_sum` / `weight_function` compile semantics (May 2026). |
| [docs/BATCH_ANALYSIS_AUDIT.md](BATCH_ANALYSIS_AUDIT.md) | Batch / `SuperAudioAnalyzer` **optional Phase 1** semantics and H+I+S handoff (read banner at top). |
| [docs/BATCH_ANALYSIS_FIELD_MAP.md](BATCH_ANALYSIS_FIELD_MAP.md) | Short column map for `batch_summary.xlsx` and related identifiers. |
| [README.md](../README.md) | Install, entry points, documentation table, tests. |
| [TECHNICAL_MANUAL.md](../TECHNICAL_MANUAL.md) | Long-form architecture, equations, historical sections explicitly labelled. |
| [QUICK_START_ORCHESTRATOR.md](../QUICK_START_ORCHESTRATOR.md) | CLI examples for `run_orchestrator.py`. |
| [API_REFERENCE.md](../API_REFERENCE.md) | API surface overview (verify signatures against code when citing). |
| [TESTING.md](../TESTING.md) | Pytest policy and recommended invariant commands. |
| [docs/DOCUMENTATION_AUDIT_REPORT.md](DOCUMENTATION_AUDIT_REPORT.md) | What was reviewed, updated, or archived in the 2026-05-13 documentation pass. |
| [docs/MATHEMATICAL_FORMALISATION_VERIFICATION_REPORT_FIRST_PASS.md](MATHEMATICAL_FORMALISATION_VERIFICATION_REPORT_FIRST_PASS.md) | LaTeX formalisation of six core `density.py` metrics (`compute_spectral_entropy`, `effective_partial_density_from_powers`, `_spectral_neff_*`, discrete `d3`–`d24`, `apply_density_metric`, `compute_rolloff_compensated_harmonic_density`); read-only verification vs code. |
| [docs/COMPUTATIONAL_METRICS_CODE_REVIEW_REPORT.md](COMPUTATIONAL_METRICS_CODE_REVIEW_REPORT.md) | Read-only inventory of **project-owned** computational lines and functions (density, alignment, policy, compile hooks); prioritises mathematical formalisation. External libraries are black-boxed. |
| [docs/formula_extraction/FORMULA_EXTRACTION_INDEX.md](formula_extraction/FORMULA_EXTRACTION_INDEX.md) | **Formula extraction (Passes 1–15):** tables mapping project-owned Python → mathematical notation. |
| [docs/formula_validation/FORMULA_VALIDATION_PLAN_INDEX.md](formula_validation/FORMULA_VALIDATION_PLAN_INDEX.md) | **Formula-validation plans (Passes 1–15):** hand-checkable fixtures and suggested assertions (plans only; executable tests live under `tests/formula_validation/`). |
| [VALIDATION_STATUS_812_PASSED_PASSES_1_15.md](../VALIDATION_STATUS_812_PASSED_PASSES_1_15.md) | **Recorded test cycle:** full suite **812 passed** (39 skipped, 0 failed); formula-validation **149 passed** (0 failed); Passes **1–15** completed. |
| [METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md](../METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md) | Methodological note: extraction → validation workflow; what is and is not established by automated formula checks. |
| [CODE_FORMULA_TRACEABILITY_TABLE.md](../CODE_FORMULA_TRACEABILITY_TABLE.md) | Optional **code ↔ formula** traceability (function, script, line ranges, expressions). |

**Formula-validation (methodological, cautious):** the formula-validation corpus supports **internal consistency** between the documented mathematical formulas and the tested Python implementations. It verifies formula/code agreement for **selected numerical fixtures**. It does **not**, by itself, prove scientific optimality, universal correctness, or full acoustic validity of the models.

**Validation tools (cite as engineering artefacts):**

- `tools/audit_compiled_workbook.py` — compiled workbook invariant audit.  
- `tools/export_research_density_workbook.py` — post-process `compiled_density_metrics.xlsx` into a reduced **research** workbook (`compiled_density_metrics_research.xlsx`) for plotting and thesis tables; does not alter compilation or the source workbook. Optional CLI metadata: `--instrument`, `--dynamic`, `--force-metadata`. Uses **worksheet `AutoFilter` only** (no formal Excel **Table** / `xl/tables/table*.xml`) so Microsoft Excel opens without repair prompts; README/Dashboard sheets are not filtered.  
- `tests/pipeline_workbook_audit.py`, `tests/test_final_pipeline_invariants.py` — automated checks.

---

## Legacy / diagnostic only (do not describe as the sole canonical path)

| Document | Note |
|----------|------|
| [audio_analysis/README_SUPER_ANALYZER.md](../audio_analysis/README_SUPER_ANALYZER.md) | **Super Audio Analyzer** CLI and `super_analysis_results.json` — optional batch Phase 1; bannered as legacy relative to `proc_audio`. |
| [ORCHESTRATOR_INTEGRATION_GUIDE.md](../ORCHESTRATOR_INTEGRATION_GUIDE.md) | Integrated orchestrator including **optional** batch preprocessing; read overview for canonical vs optional. |
| [TROUBLESHOOTING_ORCHESTRATOR.md](../TROUBLESHOOTING_ORCHESTRATOR.md) | Tk `pipeline_orchestrator_gui.py` startup only. |
| `audio_analysis/BATCH_*` READMEs | Batch folder layout and GUI config paths. |

---

## Obsolete / retired snapshots

Historical Markdown under **`Backup/**` and the old **`docs/archive_obsolete/`** tree was **removed from the working tree** (housekeeping). Retrieve specific retired files from **git history** if needed. The canonical pipeline description is **[docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md](CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md)**; root **[PIPELINE_FUNCTIONING_REPORT.md](../PIPELINE_FUNCTIONING_REPORT.md)** remains a short pointer stub only.

---

## Changelog for documentation

See [CHANGELOG.md](../CHANGELOG.md) for synchronised documentation entries (including the formula-validation Passes 1–15 documentation pass).
