# SoundSpectrAnalyse

Spectral analysis for acoustic research. **Canonical publication pipeline:** **`proc_audio.AudioProcessor`** (Stage 1) writes per-note **`spectral_analysis.xlsx`** plus standard PNGs (**`spectrogram.png`**, two **semantically distinct** component pies — linear **amplitude-mass** vs **energy-ratio** — and a legacy-alias **`component_energy_pie.png`**; see **`docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md`**); **`compile_metrics.compile_density_metrics_with_pca`** (Stage 2) builds **`compiled_density_metrics.xlsx`** with multi-sheet exports (`Density_Metrics`, `Canonical_Metrics`, `Diagnostic_Metrics`, `Debug_Counts`, …). The primary public spectral-fatness scalar on **`Density_Metrics`** is **`effective_partial_density`**.

Optional **batch preprocessing** (`batch_audio_analyzer` / `super_audio_analyzer`) may supply **`batch_summary.xlsx`** for empirical **H+I+S** profiles and **H/(H+I)** model coefficients; it is **not** required for the canonical chain above. Legacy Tk / PyQt entry points remain ancillary.

**Package version:** 3.7.0 (`pyproject.toml`; at runtime: `importlib.metadata.version("soundspectranalyse")`)  
**Python:** ≥ 3.8  

## Install

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # optional: pytest, coverage, linters
pip install -e .                      # editable install + console scripts (see Entry points)
```

Optional reproducible stack (same bounds as `requirements.txt`; does not alter export columns):

```bash
pip install --upgrade --force-reinstall -r requirements-pins.txt
```

## Entry points (what is actually used)

| Command | Role |
|---------|------|
| **`python run_orchestrator.py`** / **`soundspectranalyse`** (`pip install -e .`) | **Canonical full pipeline:** optional preprocessing → `batch_summary.xlsx` → per-note `spectral_analysis.xlsx` → **`compiled_density_metrics.xlsx`**. |
| **`run.bat`** (Windows) | Starts **`pipeline_orchestrator_gui.py`** (Tk / tier orchestrator). For the integrated batch→compile path use **`python run_orchestrator.py`** above. |
| **`python pipeline_orchestrator_integrated.py`** | Same backend as `run_orchestrator.py` when you pass audio paths explicitly (no wrapper discovery). |
| **`python main.py`** / **`soundspectranalyse-legacy-gui`** | Forwards to **`pipeline_orchestrator_integrated.py --gui`** (Tk; typically subprocess **`pipeline_orchestrator_gui.py`**). Not the old PyQt **`interface.py`** window. |
| **`python pipeline_orchestrator_gui.py`** | Same entry as **`run.bat`**; **`FFT_SETTINGS_BY_CLUSTER`** is imported from here by **`pipeline_orchestrator_integrated.py`**. |

## Documentation (aligned with the canonical code path)

| Document | Purpose |
|----------|---------|
| **[docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md](docs/CANONICAL_PIPELINE_AND_EXPORT_SEMANTICS.md)** | **Normative** pipeline, f0, harmonics, nonharmonics, subfundamental, Debug_Counts, missing metrics, audit CLI. |
| **[docs/CURRENT_DOCUMENTATION_INDEX.md](docs/CURRENT_DOCUMENTATION_INDEX.md)** | What is safe to cite vs legacy vs archived. |
| **[docs/DOCUMENTATION_AUDIT_REPORT.md](docs/DOCUMENTATION_AUDIT_REPORT.md)** | 2026-05-13 documentation audit register. |
| **[docs/MATHEMATICAL_FORMALISATION_VERIFICATION_REPORT_FIRST_PASS.md](docs/MATHEMATICAL_FORMALISATION_VERIFICATION_REPORT_FIRST_PASS.md)** | LaTeX formalisation of six core `density.py` metrics (read-only vs code). |
| **[docs/formula_extraction/FORMULA_EXTRACTION_INDEX.md](docs/formula_extraction/FORMULA_EXTRACTION_INDEX.md)** | Pass 1–15 **formula-extraction** tables (Python → notation); companion to validation plans and tests. |
| **[docs/formula_validation/FORMULA_VALIDATION_PLAN_INDEX.md](docs/formula_validation/FORMULA_VALIDATION_PLAN_INDEX.md)** | Pass 1–15 **formula-validation plans** (fixtures and assertions; plans only). |
| **[VALIDATION_STATUS_812_PASSED_PASSES_1_15.md](VALIDATION_STATUS_812_PASSED_PASSES_1_15.md)** | Recorded pytest counts: full suite **812 passed** (39 skipped, 0 failed); **149** formula-validation tests passed; Passes **1–15** completed. |
| **[METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md](METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md)** | Methodological note on extraction → validation workflow and limits of what automated checks establish. |
| **[CODE_FORMULA_TRACEABILITY_TABLE.md](CODE_FORMULA_TRACEABILITY_TABLE.md)** | Optional code ↔ formula traceability (audit / PDF-oriented). |
| **[docs/COMPUTATIONAL_METRICS_CODE_REVIEW_REPORT.md](docs/COMPUTATIONAL_METRICS_CODE_REVIEW_REPORT.md)** | Project-owned computational code inventory for formalisation / peer review (not a user guide). |
| **`docs/DENSITY_EXPORT_SCHEMA.md`** | **Authoritative** export schema: `Density_Metrics`, `Per_Note_Processing_Metadata`, dissonance/PCA separation, redaction notes. |
| **`docs/BATCH_ANALYSIS_AUDIT.md`** | Batch row semantics, H+I+S validation, model weights **H/(H+I)** (optional Phase 1). |
| **`docs/BATCH_ANALYSIS_FIELD_MAP.md`** | Short field map for `batch_summary.xlsx` and orchestrator handoff. |
| [TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md) | Long-form architecture (includes **historical** sections; canonical export semantics in **`docs/`**). |
| [TESTING.md](TESTING.md) | Pytest policy, slow-marker contract, pipeline invariants, **formula-validation** command. |
| [QUICK_START_ORCHESTRATOR.md](QUICK_START_ORCHESTRATOR.md) | CLI examples for **`run_orchestrator.py`**. |
| [ORCHESTRATOR_INTEGRATION_GUIDE.md](ORCHESTRATOR_INTEGRATION_GUIDE.md) | Optional preprocessing → main analysis integration. |
| [API_REFERENCE.md](API_REFERENCE.md) | **`AudioProcessor`** / **`density`** / compile overview. |
| [audio_analysis/README_SUPER_ANALYZER.md](audio_analysis/README_SUPER_ANALYZER.md) | **Legacy** Super Audio Analyzer CLI when batch Phase 1 is used — not the canonical Stage 1 engine. |

### Legacy or out-of-repo (not part of the default pipeline)

Older Markdown snapshots (integrated vs Tk `v2_16` notes, external **`split_audio_segments`** manual, pre-cleanup inventories) lived under **`Backup/`** and were **removed from the working tree**; use **git history** if you need them.

| Document | Note |
|----------|------|
| [TROUBLESHOOTING_ORCHESTRATOR.md](TROUBLESHOOTING_ORCHESTRATOR.md) | **Tk `pipeline_orchestrator_gui.py` startup** only — not `run_orchestrator.py`. |

## Tests

```bash
python -m pytest tests -v
```

**Formula-validation (Passes 1–15):** executable checks under **`tests/formula_validation/`** implement the per-pass validation plans against selected numerical fixtures. As recorded in **`VALIDATION_STATUS_812_PASSED_PASSES_1_15.md`**, the formula-validation suite reports **149 passed**, **0 failed**, and the full repository suite **812 passed**, **39 skipped**, **0 failed** for that recorded run. The formula-validation corpus supports **internal consistency** between the documented formulas and the tested Python implementations; it verifies formula/code agreement for those fixtures and **does not**, by itself, prove scientific optimality, universal correctness, or full acoustic validity of the models. See **`METHODOLOGICAL_NOTE_FORMULA_VALIDATION.md`** for scope and limitations.

```bash
python -m pytest tests/formula_validation/ -q
```

**Pipeline / workbook invariants (recommended):**

```bash
pytest tests/test_final_pipeline_invariants.py tests/test_low_frequency_policy.py tests/test_validate_canonical_metrics.py
```

**Compiled workbook audit (CLI):**

```bash
python tools/audit_compiled_workbook.py path/to/compiled_density_metrics.xlsx
# optional: second path to a per-note spectral_analysis.xlsx for Harmonic Spectrum checks
```

**Research export (reduced workbook for plotting / thesis tables):**

After you have the full compiled workbook, generate a separate, professionally formatted research export (does not modify the source file or change Stage-2 compilation):

```bash
python tools/export_research_density_workbook.py path/to/compiled_density_metrics.xlsx
# optional: --output path/to/compiled_density_metrics_research.xlsx
# optional: --no-charts   --overwrite
# optional: --instrument Clarinet --dynamic pp   (metadata; use --force-metadata to override non-empty workbook cells)
```

Instrument and dynamic columns may be filled from existing workbook fields, inferred conservatively from filenames and folder paths, or set explicitly with ``--instrument`` / ``--dynamic`` (see the research workbook README sheet for details).

The research workbook is written for **Microsoft Excel compatibility**: it does **not** embed formal **Table** objects (no `xl/tables/table*.xml`). Data sheets use **worksheet-level AutoFilter** and frozen header rows; **README** and **Dashboard** are not auto-filtered. Exported column headers are sanitised (no blank names; duplicates get `_2`, `_3`, …).

The full **`compiled_density_metrics.xlsx`** remains the complete technical and audit export; **`compiled_density_metrics_research.xlsx`** is the recommended workbook for analysis, plotting, and thesis-ready tables.

When you run **`run_orchestrator.py`**, **`pipeline_orchestrator_integrated.py`**, or the Tk **`pipeline_orchestrator_gui.py`** pipeline, the research workbook is generated **automatically** after each successful Stage 2 compile (same folder as the compiled workbook). Failures there are logged only and do not fail the acoustic pipeline.

Continuous integration runs the full `tests/` suite on Ubuntu (`.github/workflows/ci.yml`).

## Copyright and use

This repository is proprietary research software.

No open-source licence is granted. All rights are reserved by Luís Raimundo.

Access to this repository does not imply permission to copy, reuse, modify, redistribute, publish, sublicense, incorporate, train on, or derive works from the code, documentation, mathematical formulations, tests, validation reports, data structures, or associated materials.

Authorised users may run the software only for the specific purpose for which access has been granted. Any other use requires prior written permission.

Do not add an open-source licence.

## Acknowledgements

This project was developed by **Luís Raimundo** with the support and funding of the **Fundação para a Ciência e a Tecnologia (FCT)** and **Universidade NOVA de Lisboa**.

The FCT funding should be cited through its corresponding DOI in scientific publications, technical reports, datasets, software documentation, and other research outputs related to this project.

**Funding DOI:** `https://doi.org/10.54499/2020.08817.BD`

The author also gratefully acknowledges **Isabel Pires** for her support throughout the development of this work.
