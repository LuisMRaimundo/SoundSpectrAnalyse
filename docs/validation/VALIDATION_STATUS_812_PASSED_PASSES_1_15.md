# Validation Status — 812 Passed — Formula Passes 1–15

## 1. Full test-suite result

- **Full pytest result:** 812 passed, 39 skipped, 0 failed.
- **Formula-validation suite:** 149 passed, 0 failed.
- **Formula-validation passes completed:** Passes 1–15.

## 2. Formula-validation coverage

Formula-validation tests now cover:

- **Pass 1** — density metrics
- **Pass 2** — weight functions
- **Pass 3** — partial sums and metric bundles
- **Pass 4** — residual and inharmonic classification
- **Pass 5** — harmonic alignment
- **Pass 6** — peak component counts
- **Pass 7** — low-frequency policy
- **Pass 8** — spectral leakage guards
- **Pass 9** — compile-time normalisation
- **Pass 10** — selected proc_audio formulas
- **Pass 11** — extended density metrics
- **Pass 12** — dissonance models
- **Pass 13** — peak detection and f0 refinement
- **Pass 14** — compile extraction and batch mass
- **Pass 15** — data integrity normalisation

## 3. What this validation supports

The formula-validation corpus supports **internal consistency** between the documented mathematical formulas and the tested Python implementations.

It verifies that, for the selected numerical fixtures, the implementation outputs agree with the extracted formulas.

## 4. What this validation does not prove

This does **not** prove:

- scientific optimality of models;
- universal correctness for all possible inputs;
- literature validity of all modelling choices;
- complete physical/acoustic adequacy;
- future correctness after later code changes.

## 5. Warning status

The full suite still emits **warnings**, mainly dependency deprecation warnings from matplotlib / pyparsing and expected edge-case warnings from selected tests. These warnings did not cause test failures.

## 6. Repository state note

This document records the validated state after completing formula-validation Passes 1–15. It is a static human-readable status note and does not execute tests.
