# Formula Validation Plan — Pass 6 — Peak component counts

## 1. Scope

Pass 6: **`peak_component_counts.py`** (dB→linear, Hz tolerance from cents, peak-list classification), as in `FORMULA_EXTRACTION_TABLE_PASS_06_PEAK_COMPONENT_COUNTS.md`.

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 6-01 | dB → linear | Row with `Magnitude (dB)` = **20.0** | **10.0** | `peak_component_counts._linear_amp_from_row` with minimal `DataFrame` | `assert_allclose` | |
| 6-02 | Hz tolerance from cents | `f0=100`, `n=2` → expected 200 Hz, `tolerance_cents=18` | \(\Delta f = 200\,(2^{18/1200}-1)\) | same formula as in `classify_peaks_harmonic_inharmonic_subbass_from_df` loop | numeric check vs **tol_hz** | |
| 6-03 | Subbass vs harmonic | One peak at **50 Hz** (`<200`), one at **300 Hz** near **3×100** within tol | **s_n≥1**, harmonic slot filled if within window | `peak_component_counts.classify_peaks_harmonic_inharmonic_subbass_from_df` | integer counts | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
