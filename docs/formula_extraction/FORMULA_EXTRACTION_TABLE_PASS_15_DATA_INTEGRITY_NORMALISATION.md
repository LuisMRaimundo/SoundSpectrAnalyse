# Formula Extraction Table — Pass 15 — Data Integrity Normalisation

Project-owned **`data_integrity.py`** only. `numpy` / `pandas` primitives (`percentile`, `median`, `clip`, `nan`, etc.) are treated as **black boxes** except where this file combines them with explicit closed-form expressions.

| Function / context | Python expression | Mathematical formula | Symbol definitions | Notes |
|---|---|---|---|---|
| `metric_float_or_nan` | `float(value)`; `not math.isfinite(x)` | \(y=x\) if \(x\in\mathbb{R}\) finite; else NaN sentinel | `MISSING_FLOAT` = `nan` | No continuous transform. **extracted** |
| `metric_int_or_nan` | `int(x)` after float | \(y=\lfloor x\rfloor\) toward zero for finite \(x\); else `pd.NA` | | **extracted** |
| `metric_ratio_or_nan` | `return float(n / d)` when finite and \(d>0\) | \(r=n/d\) if both finite and \(d>0\); else NaN | \(n,d\) coerced via `metric_float_or_nan` | **extracted** |
| `calculate_iqr_bounds` | `data_clean = data[np.isfinite(data)]`; `Q1 = np.percentile(data_clean, 25)`; `Q3 = np.percentile(data_clean, 75)`; `IQR = Q3 - Q1` | \(Q_1=P_{25}(\mathcal{X})\), \(Q_3=P_{75}(\mathcal{X})\), \(\mathrm{IQR}=Q_3-Q_1\) over finite multiset \(\mathcal{X}\) | \(k=\) `iqr_multiplier` (default `1.5`) | Percentiles via `numpy`; **partially extracted**. |
| `calculate_iqr_bounds` | `lower_bound = Q1 - iqr_multiplier * IQR`; `upper_bound = Q3 + iqr_multiplier * IQR` | \(L=Q_1-k\cdot\mathrm{IQR}\), \(U=Q_3+k\cdot\mathrm{IQR}\) | Tukey-style fences | **extracted** |
| `calculate_iqr_bounds` | empty or no finite data | Returns `(0,0,0,0)` | | **heuristic** sentinel. **extracted** |
| `detect_outliers` | `outlier_mask = (data < lower_bound) \| (data > upper_bound)` | Outlier if \(x_i<L\) or \(x_i>U\) using bounds from `calculate_iqr_bounds` on finite subset | Bounds from **clean** finite data; mask evaluated on **original** `data` (non-finite positions not flagged as outliers by comparison) | **extracted** |
| `robust_normalize` | `data_clean = data[np.isfinite(data)]`; empty clean → `np.full_like(data, np.nan)` | If no finite samples, output all-NaN (shape preserved) | | Differs from `GlobalReferenceScaler.transform` empty-clean branch (**zeros**); **ambiguous / needs human review** (inconsistent policy). **extracted** |
| `robust_normalize` method `"iqr"` | `normalized = (data - lower_bound) / (upper_bound - lower_bound)` when `upper_bound > lower_bound` | \(z_i=(x_i-L)/(U-L)\) for all \(x_i\) in array (then non-finite slots overwritten) | \(L,U\) from `calculate_iqr_bounds(data_clean, iqr_multiplier)` | Linear affine map using Tukey fences as range. **extracted** |
| `robust_normalize` method `"iqr"` degenerate | `normalized = np.zeros_like(data, dtype=float)` | All mapped to \(0\) when \(U=L\) on finite clean data | | **extracted** |
| `robust_normalize` method `"percentile"` | `p_low = np.percentile(data_clean, percentile_low)`; `p_high = np.percentile(data_clean, percentile_high)`; `(data - p_low) / (p_high - p_low)` | \(z_i=(x_i-P_{\ell})/(P_u-P_{\ell})\) with defaults \(\ell=5\), \(u=95\) | | **partially extracted** (percentile black box). **extracted** |
| `robust_normalize` method `"robust_zscore"` | `median = np.median(data_clean)`; `mad = np.median(np.abs(data_clean - median))` | \(\tilde m=\mathrm{median}(\mathcal{X})\), \(\mathrm{MAD}=\mathrm{median}_j|x_j-\tilde m|\) | \(\mathcal{X}=\) finite values | MAD standard. **partially extracted** |
| `robust_normalize` method `"robust_zscore"` | `z_scores = (data - median) / (1.4826 * mad)`; `normalized = (z_scores + 3.0) / 6.0` | \(z_i'=\dfrac{x_i-\tilde m}{1.4826\,\mathrm{MAD}}\); then \(u_i=(z_i'+3)/6\) mapping nominal \([-3,3]\rightarrow[0,1]\) before clip | Factor `1.4826` = Gaussian consistency factor (comment in code) | **model-dependent** scale choice. **extracted** |
| `robust_normalize` method `"robust_zscore"` degenerate | `mad > 0` else zeros | If \(\mathrm{MAD}=0\), all zeros | | **extracted** |
| `robust_normalize` else branch (unknown `method`) | `data_min = np.min(data_clean)`; `(data - data_min) / (data_max - data_min)` | Min–max on finite clean subset: \(z_i=(x_i-x_{\min})/(x_{\max}-x_{\min})\) | | Fallback path. **extracted** |
| `robust_normalize` | `np.clip(normalized, clip_range[0], clip_range[1])` when `clip_range` not `None` | \(z_i'=\mathrm{clip}(z_i,c_{\min},c_{\max})\) with default \([0,1]\) | | **extracted** |
| `robust_normalize` | `result = np.full_like(data, np.nan)`; `result[np.isfinite(data)] = normalized[np.isfinite(data)]` | Preserve non-finite input positions as NaN in output | | **extracted** |
| `GlobalReferenceScaler.fit` | `percentile` branch stores `p5`, `p95`, `median`, `mean` | Records \(P_5, P_{95}, \tilde m, \bar x\) of clean reference | | **partially extracted** |
| `GlobalReferenceScaler.fit` | `iqr` branch | Stores `Q1,Q3,lower_bound,upper_bound,median` from `calculate_iqr_bounds` | | **extracted** |
| `GlobalReferenceScaler.fit` | `mean_std` branch | `mean`, `std`, `median` of clean reference | \(\sigma=\) population `np.std` on clean array (default ddof) | **partially extracted** |
| `GlobalReferenceScaler.transform` | percentile: `(data - p_low) / (p_high - p_low)` | Same affine form as local percentile norm but using **fitted** reference bounds | | **extracted** |
| `GlobalReferenceScaler.transform` | `data_clean.size == 0` | `return np.zeros_like(data)` | | Differs from `robust_normalize` all-NaN → NaN; **ambiguous / needs human review**. **extracted** |
| `GlobalReferenceScaler.transform` | unfitted scaler | `return robust_normalize(data, method="iqr", clip_range=clip_range)` | Delegates | **extracted** |
| `GlobalReferenceScaler.transform` | clip + finite mask restore | Same pattern as `robust_normalize` | | **extracted** |
| `validate_metric_value` | range check | Fails if \(v<v_{\min}\) or \(v>v_{\max}\) when range provided | | **extracted** (logic). |
| `validate_metric_array` | `stats` dict | Means, medians, std, min, max, NaN/Inf counts on finite clean subset | | Mostly aggregation; **partially extracted** |
| `validate_metric_array` | `outlier_fraction = np.sum(outlier_mask) / values_clean.size` | \(\rho=\frac{1}{|\mathcal{X}|}\sum_j \mathbf{1}_{\mathrm{outlier}}(x_j)\) vs `max_outlier_fraction` | Uses `detect_outliers(..., return_mask=True)` on **clean** array | **extracted** |
| `validate_audio_parameters` | `nyquist = sr / 2.0`; `freq_resolution = sr / n_fft` | \(f_{\mathrm{Nyq}}=f_s/2\), \(\Delta f=f_s/N_{\mathrm{fft}}\) | \(f_s=\) `sr`; \(N_{\mathrm{fft}}=\) `n_fft` | Physical sampling relations; threshold checks heuristic. **extracted** |
| `normalize_log_transform` | `flat = pd.to_numeric(...).to_numpy()`; `m = np.isfinite(flat)` | Coerce to numeric; finite mask \(\mathcal{M}\) | | **extracted** |
| `normalize_log_transform` | `data_positive = np.maximum(data_clean, epsilon)`; `log_data = np.log1p(data_positive)` | \(u_j=\ln(1+\max(x_j,\varepsilon))\) with \(\varepsilon=\) `epsilon` (default \(10^{-10}\)) | | \(\log(1+x)\) stabilisation. **extracted** |
| `normalize_log_transform` | `(log_data - log_min) / (log_max - log_min)` | Min–max on \(\{u_j\}\): \(z_j=(u_j-u_{\min})/(u_{\max}-u_{\min})\) | | **extracted** |
| `normalize_log_transform` | `np.clip(normalized, clip_range[0], clip_range[1])` | Clip to \([0,1]\) by default | | **extracted** |
| `normalize_log_transform` | no finite values after coerce | `return np.zeros_like(orig, dtype=float)` | | Differs from `robust_normalize` empty-clean → NaN; **ambiguous / needs human review**. **extracted** |
| `normalize_log_transform` | reshape to `orig.shape` | Scatter normalized values back to ravel indices | | **extracted** |

## Summary

- **Fully extracted:** `calculate_iqr_bounds` (IQR and Tukey fences); `detect_outliers` (threshold mask); `robust_normalize` for all documented `method` branches including affine maps, degenerate zero maps, optional `np.clip`, and finite-position preservation; `GlobalReferenceScaler.fit` / `transform` affine formulas and clip/mask pattern; `metric_ratio_or_nan`; `validate_metric_array` outlier-fraction rule; `validate_audio_parameters` Nyquist and bin-spacing formulas; `normalize_log_transform` (\(\max(x,\varepsilon)\), \(\log(1+\cdot)\), min–max on log values, clip, index scatter).

- **Partially extracted:** Any use of `np.percentile`, `np.median`, `np.mean`, `np.std` as black-box estimators; `GlobalReferenceScaler` `mean_std` \(\sigma\) definition tied to `numpy` default `ddof`.

- **Functions not found in `data_integrity.py`:** **`robust_normalize_array`**, **`log_transform_normalize`** (closest implemented name is **`normalize_log_transform`**), **`OutlierDetector`** (no class by that name; outlier logic is procedural via **`calculate_iqr_bounds`** / **`detect_outliers`**).

- **Ambiguous or model-dependent items requiring human review:** (1) **`1.4826`** MAD scale and **linear map \((z+3)/6\)** for pseudo–z-score — conventional but not unique. (2) **`iqr_multiplier`**, default percentiles **5/95**, and default **clip_range** \([0,1]\) — policy. (3) **Inconsistent empty-/non-finite behaviour:** `robust_normalize` → all-**NaN** vs `GlobalReferenceScaler.transform` / `normalize_log_transform` → **zeros** in some branches. (4) **`detect_outliers`** applies bounds from **finite-only** data but tests membership on the **full** array (NaNs not outliers). (5) **`validate_audio_parameters`** warning heuristics beyond hard thresholds.

- **Recommendation:** A **Pass 15 validation-plan** document is **recommended** so unit tests can pin IQR fences, each `robust_normalize` branch, log1p min–max, clip behaviour, and edge cases (empty, all-NaN, constant vector) without re-deriving `numpy` percentile interpolation.
