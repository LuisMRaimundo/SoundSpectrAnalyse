# Formula Validation Plan — Pass 9 — Compile-time normalisation

## 1. Scope

Pass 9: **`compile_metrics.py`** helpers **`_add_canonical_and_global_density_columns`** and **`_compute_weighted_density_columns_for_wide_df`**, as in `FORMULA_EXTRACTION_TABLE_PASS_09_COMPILE_TIME_NORMALISATION.md`.

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 9-01 | Canonical fallback | One-row `DataFrame`: `Density Metric` = **50**, missing canonical column | `canonical_density_v5_adapted` = **5.0** | `compile_metrics._add_canonical_and_global_density_columns` (fixture frame) | `assert_allclose` | |
| 9-02 | Global normalisation | Two rows canonical **2** and **8** | `density_normalized_global` → **0.25**, **1.0**; denominator **8** | same helper | column-wise | |
| 9-03 | Per-component density | `s_canon=6`, `harmonic_order_count=2` | **3.0** where division defined | same (after helper fills column) | `assert_allclose` | |
| 9-04 | Weighted raw | \(D_H=10,w_H=0.2\Rightarrow c_H=2\); analogous **\(c_I,c_S\)** | `density_metric_raw` = sum contributions | `compile_metrics._compute_weighted_density_columns_for_wide_df` | `assert_allclose` on constructed wide row | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
