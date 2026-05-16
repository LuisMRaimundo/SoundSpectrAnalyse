# Formula Validation Plan — Pass 3 — Partial sums and metric bundles

## 1. Scope

Pass 3: **`band_partial_metric_sum`**, **`partial_metric_sums_h_i_s_total`**, **`compute_discrete_spectral_metrics_bundle`** in `density.py`, as in `FORMULA_EXTRACTION_TABLE_PASS_03_PARTIAL_SUMS_AND_BUNDLES.md`.

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 3-01 | `band_partial_metric_sum` linear | `amplitudes=[1.0, 2.0]`, `weight_key="linear"` | \(\sum A_i=\) **3.0** | `density.band_partial_metric_sum` | `assert_allclose` | |
| 3-02 | H/I/S scalars + Total additive | `H=[1]`, `I=[2]`, `S=[3]`, `weight_key="linear"` | **H=1, I=2, S=3, T=6** | `density.partial_metric_sums_h_i_s_total` | `assert_allclose` each | |
| 3-03 | d10/d17 Total = concatenated metric | Small `ah`, `ai` with known `d10` on concat | **T** equals `band_partial_metric_sum(concat(ah,ai,asb), "d10", ff)` not **H+I+S** | `density.partial_metric_sums_h_i_s_total(..., weight_key="d10")` | compare fourth return to direct `band_partial_metric_sum` | |
| 3-04 | `compute_discrete_spectral_metrics_bundle` | `amplitudes=[1.0,1.0]` | `discrete_metric_d3` = \(2\ln 2\) | `density.compute_discrete_spectral_metrics_bundle` | key-wise `assert_allclose` | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
