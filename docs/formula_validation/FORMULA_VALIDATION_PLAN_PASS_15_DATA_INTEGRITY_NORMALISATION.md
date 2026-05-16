# Formula Validation Plan — Pass 15 — Data Integrity Normalisation

## 1. Scope

This plan defines small, hand-checkable numerical fixtures for formulas documented in `docs/formula_extraction/FORMULA_EXTRACTION_TABLE_PASS_15_DATA_INTEGRITY_NORMALISATION.md` (Pass 15): project-owned **`data_integrity.py`**. It is intended for later `pytest` implementation; **no tests are created here**. Use **direct function calls** and **`numpy` / `pandas`** only for reproducing expected numbers where the implementation delegates to a black box (e.g. match **`numpy.percentile`** defaults used by the code).

## 2. Included validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| MF-1 | `metric_float_or_nan` | `3.14`, `None`, `"not_a_number"` | \(3.14\); `nan`; `nan` | `metric_float_or_nan(...)` | `assert_allclose(v1, 3.14)`; `assert math.isnan(v2)`; `assert math.isnan(v3)` | |
| MI-1 | `metric_int_or_nan` | `8.9`, `None` | \(8\); `pd.NA` | `metric_int_or_nan(...)` | `assert v1 == 8`; `assert v2 is pd.NA` (or `pd.isna(v2)`) | Truncates toward zero like `int(float(x))`. |
| MR-1 | `metric_ratio_or_nan` | `(10, 2)`, `(1.0, 0.0)`, `(1.0, float("nan"))` | \(5\); `nan`; `nan` | `metric_ratio_or_nan(...)` | `assert_allclose(r1, 5.0)`; `assert math.isnan(r2)` and `r3` | |
| IQR-1 | `calculate_iqr_bounds` | `data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])`, `iqr_multiplier=1.5` | \(Q_1=P_{25}\), \(Q_3=P_{75}\), \(\mathrm{IQR}=Q_3-Q_1\), \(L=Q_1-1.5\,\mathrm{IQR}\), \(U=Q_3+1.5\,\mathrm{IQR}\) — **match** `numpy.percentile` on `data` in the test with the same arguments the implementation uses | `calculate_iqr_bounds(data, 1.5)` | `assert_allclose(Q1, np.percentile(data,25))` (and likewise for `Q3`, bounds) | **Defer** alternative percentile definitions. |
| IQR-2 | `calculate_iqr_bounds` empty | `np.array([])` | `(0.0, 0.0, 0.0, 0.0)` | `calculate_iqr_bounds(...)` | `assert tuple == (0.0,0.0,0.0,0.0)` | |
| DO-1 | `detect_outliers` | `data = np.array([0.0,1.0,2.0,3.0,100.0])`, default multiplier | Values outside Tukey fences from `calculate_iqr_bounds` on finite subset; `100.0` typically flagged | `detect_outliers(data, return_mask=False)` | `assert 100.0 in outliers` (or compare mask with manual fence check) | Mask uses **raw** `data` vs bounds from **finite-only** stats. |
| RN-IQR-1 | `robust_normalize` `method="iqr"` | e.g. `np.array([0.0, 5.0, 10.0])` with bounds \(L,U\) from `calculate_iqr_bounds` on clean data | For finite \(x_i\): \((x_i-L)/(U-L)\); interior point hand-recomputable once \(L,U\) fixed | `robust_normalize(data, method="iqr", clip_range=None)` | `assert_allclose(out[finite], manual, rtol=1e-12)` | Use `clip_range=None` to isolate affine step before clip. |
| RN-PCT-1 | `robust_normalize` `method="percentile"` | `np.linspace(0.0, 100.0, 21)` (21 points), `percentile_low=5`, `percentile_high=95`, `clip_range=None` | \(P_5,P_{95}\) from `numpy`; midpoint \(50\) maps to \((50-P_5)/(P_{95}-P_5)\) | `robust_normalize(..., method="percentile", ...)` | Recompute \(P_5,P_{95}\) in test with `np.percentile`; assert midpoint | |
| RN-RZ-1 | `robust_normalize` `method="robust_zscore"` | `np.array([-1.0, 0.0, 1.0])` | \(\tilde m=0\), \(\mathrm{MAD}=1\); at \(x=0\): \(z'=0\), \(u=(z'+3)/6=0.5\); at \(x=1\): \(z'=1/(1.4826\cdot1)\), \(u=(z'+3)/6\) | `robust_normalize(arr, method="robust_zscore", clip_range=None)` | `assert_allclose(out[1], 0.5, rtol=1e-9)` for index of `0.0` | **Defer** optimality of `1.4826` and \((z+3)/6\). |
| RN-FB-1 | `robust_normalize` unknown method → min–max | `method="not_a_real_method"`, `data=np.array([1.0, 2.0, 4.0])`, `clip_range=None` | \((2-1)/(4-1)=1/3\) at \(x=2\) | `robust_normalize(...)` | `assert_allclose(out[idx], 1.0/3.0, rtol=1e-12)` | |
| RN-CLIP-1 | `robust_normalize` clipping | Same array as **RN-FB-1** but first scale values **above** 1 without clip (e.g. extend range), then call with default `clip_range=(0,1)` **or** use inputs where affine output exceeds 1 before clip | All finite outputs in \([0,1]\) | `robust_normalize(..., clip_range=(0.0,1.0))` | `assert (out[np.isfinite(out)] <= 1.0 + 1e-12).all()` and lower bound \(\ge -1e-12\) | Construct explicit overshoot if needed. |
| RN-NAN-1 | `robust_normalize` all non-finite | `np.array([np.nan, np.inf, -np.inf])` | Entire output **NaN** (shape preserved) | `robust_normalize(..., method="iqr")` | `assert np.isnan(out).all()` | Documents “no finite values” branch. |
| GRS-1 | `GlobalReferenceScaler.fit` / `transform` percentile | `fit` on `reference = np.linspace(0.0, 10.0, 11)` with `method="percentile"`; `transform` on `np.array([5.0])` | Stored `p5`,`p95` equal `np.percentile(reference,5)` and `95`; transformed \(=(5-p_5)/(p_{95}-p_5)\) | `GlobalReferenceScaler()` then `fit` / `transform` | `assert_allclose(transformed[0], (5.0-p5)/(p95-p5), rtol=1e-9)` | |
| GRS-2 | `GlobalReferenceScaler.transform` unfitted | `transform` before `fit` on any finite array | Delegates to `robust_normalize(..., method="iqr", ...)` | `transform` | `assert_allclose(out, robust_normalize(data, method="iqr", clip_range=(0.0,1.0)), rtol=1e-12)` | |
| VMV-1 | `validate_metric_value` range | `value=0.5`, `expected_range=(0.0,1.0)` | `(True, None)` | `validate_metric_value(0.5, "m", expected_range=(0.0,1.0))` | `assert ok is True` and message `None` | |
| VMV-2 | `validate_metric_value` out of range | `value=1.5`, same range | `(False, msg)` | same | `assert ok is False` | |
| VMA-1 | `validate_metric_array` outlier fraction | `values = np.array([0,0,0,0,0,1000.0])`, `max_outlier_fraction=0.1` | IQR fences degenerate or tight so `1000` is outlier; fraction \(1/6 > 0.1\) → invalid | `validate_metric_array(values, "m", max_outlier_fraction=0.1)` | `assert ok is False` | Tune counts so fraction **exceeds** threshold under current `detect_outliers`. |
| VMA-2 | `validate_metric_array` pass | `np.linspace(1.0, 5.0, 20)`, same threshold | Valid | same | `assert ok is True` | |
| VAP-1 | `validate_audio_parameters` + Nyquist / bin spacing | `n_fft=4096`, `hop_length=256`, `sr=48000`, `signal_length=8192` | `(True, None)`; manual \(f_{\mathrm{Nyq}}=24000\) Hz, \(\Delta f=48000/4096\) Hz (duplicate formulas in test for documentation) | `validate_audio_parameters(4096,256,48000,8192)` | `assert out == (True, None)`; optional `assert_allclose(sr/2.0, 24000)` and `assert_allclose(sr/n_fft, 48000/4096)` | Warning branch when \(\Delta f > f_{\mathrm{Nyq}}/100\) is **log-only**; still returns `True`. **Defer** policy interpretation. |
| VAP-2 | `validate_audio_parameters` rejection | `n_fft=32` (below minimum) | `(False, ...)` | `validate_audio_parameters(32,16,44100,65536)` | `assert out[0] is False` | |
| NL-1 | `normalize_log_transform` | `np.array([[1.0, 3.0]])` (or 1-D `[1.0, 3.0]`), default `epsilon`, `clip_range=None` | \(u_j=\ln(1+\max(x_j,\varepsilon))\); min–max on \(\{u_j\}\) gives endpoints **0** and **1** | `normalize_log_transform(data, clip_range=None)` | `assert_allclose(finite outputs, [0.0, 1.0], atol=1e-9)` after selecting finite entries | Shape preserved. |

## 3. Deferred / human-review cases

Do **not** encode these as settled scientific requirements in automated tests:

- Whether **`1.4826`** is the best MAD scale factor for **`robust_zscore`**.
- Whether mapping \(z \mapsto (z+3)/6\) is **scientifically optimal**.
- Whether default **`percentile_low` / `percentile_high`** (5 / 95) are optimal.
- Whether **empty** or **all-NaN** outputs should be **NaN vs zero** consistently across **`robust_normalize`**, **`GlobalReferenceScaler.transform`**, and **`normalize_log_transform`** (known cross-function inconsistency in extraction).
- **`numpy`** internals for **`percentile`**, **`median`**, **`std`** (including interpolation / `ddof`).
- Broad **QA policy** interpretation beyond the explicit thresholds in **`validate_audio_parameters`**.

## 4. Implementation status

No tests are created by this document. This is a validation plan only.
