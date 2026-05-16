# Formula Validation Plan — Pass 7 — Low-frequency policy

## 1. Scope

Pass 7: **`low_frequency_policy.py`**, as in `FORMULA_EXTRACTION_TABLE_PASS_07_LOW_FREQUENCY_POLICY.md`.

**`calculate_subfundamental_margin_percent` (implementation):** if \(f_0 < 60\) then margin **35%**; else if \(f_0 < 120\) then **25%**; else if \(f_0 < 300\) then **15%**; else **10%**. Invalid or non-positive \(f_0\) returns **10%** (see code).

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 7-01 | Margin piecewise | `f0_hz = 50`, `90`, `150`, `400` | **35**, **25**, **15**, **10** (percent) | `low_frequency_policy.calculate_subfundamental_margin_percent` | exact | Examples sit in \(<60\), \([60,120)\), \([120,300)\), and \(\ge 300\) Hz bands respectively. |
| 7-02 | Adaptive cutoff | `f0_hz=100`, defaults, no leakage | \(m=25\%\) (since \(100<120\)); nominal line \(f_\%=100(1-25/100)=75\) Hz; compare `raw_max`, `adaptive`, `effective_subfundamental_margin_percent` in the returned dict | `low_frequency_policy.calculate_adaptive_subfundamental_cutoff_hz` | dict fields `assert_allclose` | Final `adaptive_subfundamental_cutoff_hz` also applies `min_floor_hz`, optional leakage cutoff, and cap \(0.95\,f_0\). |
| 7-03 | `classify_low_frequency_row` boundaries | `dc_floor_hz=30`, `ad=40`, `phys_hi=200` | **f=20** → dc; **f=35** → subfundamental; **f=150** → physical low; **f=300** → not low | `low_frequency_policy.classify_low_frequency_row` | string equality | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
