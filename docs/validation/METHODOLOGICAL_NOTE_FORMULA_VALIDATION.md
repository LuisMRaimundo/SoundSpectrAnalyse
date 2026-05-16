# Methodological Note — Formula Extraction and Validation

## 1. Purpose

The goal of this work was to make the **computational mathematics** of the codebase **explicit**, **auditable**, and **testable**: to move from implicit arithmetic in project-owned Python to a traceable chain from code to notation to executable checks.

The process followed **three stages**:

1. **Extraction** of project-owned, formula-bearing expressions from the implementation (recorded in per-pass formula-extraction tables, indexed in `docs/formula_extraction/FORMULA_EXTRACTION_INDEX.md`).
2. **Translation** of those expressions into **mathematical notation** (compact symbolic summaries aligned with the code paths they document).
3. **Validation** through **small, executable numerical tests** specified in per-pass validation plans (`docs/formula_validation/FORMULA_VALIDATION_PLAN_PASS_*.md`, indexed in `docs/formula_validation/FORMULA_VALIDATION_PLAN_INDEX.md`) and realised as pytest modules under `tests/formula_validation/`.

Third-party libraries are used where the project delegates computation; their **internals are treated as black boxes** unless a test deliberately reproduces the same public call pattern as the project code (for example matching `numpy` percentile conventions where the implementation explicitly uses them).

## 2. Scope of the validation

**Formula-validation** coverage extends over **Passes 1–15**, corresponding to the following thematic areas (each with a paired extraction table and validation plan, except Pass 1 which uses the density first-pass extraction table name):

- **Pass 1** — density metrics  
- **Pass 2** — weight functions  
- **Pass 3** — partial sums and metric bundles  
- **Pass 4** — residual / inharmonic classification  
- **Pass 5** — harmonic alignment  
- **Pass 6** — peak component counts  
- **Pass 7** — low-frequency policy  
- **Pass 8** — spectral leakage guards  
- **Pass 9** — compile-time normalisation  
- **Pass 10** — selected `proc_audio` formulas  
- **Pass 11** — extended density metrics  
- **Pass 12** — dissonance models  
- **Pass 13** — peak detection and f₀ refinement  
- **Pass 14** — compile extraction and batch mass  
- **Pass 15** — data integrity normalisation  

Primary artefacts include, among others: `FORMULA_EXTRACTION_TABLE_DENSITY_FIRST_PASS.md` and `FORMULA_EXTRACTION_TABLE_PASS_02_WEIGHT_FUNCTIONS.md` through `FORMULA_EXTRACTION_TABLE_PASS_15_DATA_INTEGRITY_NORMALISATION.md`; and `FORMULA_VALIDATION_PLAN_PASS_01_DENSITY_METRICS.md` through `FORMULA_VALIDATION_PLAN_PASS_15_DATA_INTEGRITY_NORMALISATION.md`.

## 3. Method

The method combined **static documentation** with **targeted dynamic checks**:

- **Project-owned Python** was inspected; extraction tables record the **intended computational meaning** of selected expressions (not every line of every module).
- **Third-party package internals** were treated as **black boxes** unless a validation case explicitly mirrors the same API call the project uses (so tests check **agreement with the documented project formula** and the **delegated primitive**, not a full proof of the library).
- Each relevant expression was translated into **mathematical notation** in the extraction tables.
- **Small, hand-checkable examples** (inputs, manual expectations, suggested assertions) were defined in the validation plans; deferred or policy-heavy items were marked for human review rather than encoded as settled scientific requirements where the plans so indicate.
- **pytest** tests compared **Python outputs** with **manual expected values** (floating-point comparisons via `numpy.testing.assert_allclose` where appropriate; exact checks for discrete or sentinel outcomes).
- When a check disagreed with expectations, outcomes were triaged as **implementation errors**, **test-expectation errors**, **documentation inconsistencies**, or **metadata/provenance issues**, without treating a single passing fixture as universal proof.

## 4. Validation result

As recorded for the completed cycle:

- **Full pytest result:** 812 passed, 39 skipped, 0 failed.  
- **Formula-validation suite:** 149 passed, 0 failed.  
- **Formula-validation passes completed:** 1–15.  

(Full-suite runs may still emit **warnings**—for example dependency deprecations or expected edge-case warnings—as summarised in the status document cited below; those warnings did not cause failures in the recorded run.)

## 5. What was established

The validation establishes **internal consistency** between the **documented mathematical formulas** (extraction tables and validation plans) and the **tested Python implementations** for the **selected numerical fixtures**.

It supports **software reliability**, **reproducibility** of documented computations on those fixtures, **auditability** of the formula-to-code mapping, and **regression stability** when the implementation changes, provided the test suite is re-run and maintained.

## 6. What was not established

This validation does **not** prove:

- **scientific optimality** of the models;  
- **universal correctness** for all possible inputs;  
- **literature validity** of all modelling choices;  
- **full physical/acoustic adequacy** of the metrics as measures of sound or timbre;  
- **correctness after future code changes** without renewed testing;  
- **adequacy** of all thresholds, heuristics, or policy constants used outside the fixed fixtures.  

It also does not replace independent replication, measurement campaigns, or domain-expert sign-off.

## 7. Remaining scientific work

Typical **remaining scientific tasks**—outside the scope of formula extraction/validation as executed here—include:

- **Literature justification** of model choices and parameter conventions;  
- **Sensitivity analysis** and uncertainty quantification for key pipelines;  
- **External validation** on reference signals, corpora, or controlled acoustic conditions;  
- **Comparison with alternative models** or implementations;  
- **Domain-expert review** of whether the documented quantities answer the intended research questions;  
- **Thesis or technical-manual prose** integrating mathematical definitions with acoustic interpretation and experimental design.  

## 8. Recommended wording for thesis or technical manual

The computational metrics were subjected to a formula-extraction and formula-validation procedure. Project-owned formula-bearing expressions were translated into explicit mathematical notation and validated through executable numerical tests. The resulting formula-validation suite covers Passes 1–15 and passed fully, supporting internal consistency between the documented formulas and their Python implementations. This procedure verifies implementation/formula agreement for selected numerical fixtures; it does not, by itself, prove scientific optimality or universal acoustic validity.

## 9. Status note

The file **`VALIDATION_STATUS_812_PASSED_PASSES_1_15.md`** (repository root) is the **final status record** for this validation cycle: it summarises the recorded pytest counts, lists Passes 1–15, states what the validation supports and does not prove, and notes warning behaviour during the full-suite run. **It is a static human-readable status note and does not execute tests.**
