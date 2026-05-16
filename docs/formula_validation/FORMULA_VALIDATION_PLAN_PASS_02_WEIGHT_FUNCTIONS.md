# Formula Validation Plan — Pass 2 — Weight functions

## 1. Scope

Pass 2: **`WeightFunction`** and **`get_weight_function`** in `density.py`, as in `FORMULA_EXTRACTION_TABLE_PASS_02_WEIGHT_FUNCTIONS.md`.

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 2-01 | `WeightFunction.linear` | `x = 4.0` | **4.0** | `density.WeightFunction.linear(4.0)` | exact | |
| 2-02 | `sqrt` / `squared` / `cubic` | `x = 4.0` | **2.0**, **16.0**, **64.0** | same class staticmethods | exact | |
| 2-03 | `cbrt` | `x = -8.0` | **-2.0** | `WeightFunction.cbrt(-8.0)` | exact | |
| 2-04 | `logarithmic` | `x = 4.0` | \(\ln 5\) ≈ **1.609438** | `WeightFunction.logarithmic(4.0)` | `assert_allclose` | |
| 2-05 | `exponential` | `x = 4.0` | \(e^4-1\) | `WeightFunction.exponential(4.0)` | `assert_allclose` | |
| 2-06 | `inverse_log` | `x = 4.0` | \(1/(\ln 5 + 10^{-10})\) | `WeightFunction.inverse_log(4.0)` | `assert_allclose`, `atol=1e-12` | |
| 2-07 | `get_weight_function` alias | `name="sum"` or `"d2"` | resolves to same as `linear` | `density.get_weight_function("sum")(3.0)` vs `linear` | equal | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
