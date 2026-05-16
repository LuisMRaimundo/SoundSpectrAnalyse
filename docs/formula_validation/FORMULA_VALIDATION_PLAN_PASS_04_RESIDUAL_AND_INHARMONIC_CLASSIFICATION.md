# Formula Validation Plan — Pass 4 — Residual and inharmonic classification

## 1. Scope

Pass 4: **`identify_nonharmonic_residual_rows`** in `density.py`, as in `FORMULA_EXTRACTION_TABLE_PASS_04_RESIDUAL_AND_INHARMONIC_CLASSIFICATION.md`.

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 4-01 | Exclusion half-width (relative) | `harmonic_df` one row 100 Hz; `complete_df` rows 100.0 and 103.0; `tolerance=0.02`; no leakage | \(\tau=2\) Hz; **100** masked in; **103** in residual subset | `density.identify_nonharmonic_residual_rows` | row counts / membership | |
| 4-02 | Same with `spectral_leakage_guard=False` | as above | Same \(\tau\) if `leak_hw=0` | same | membership | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
