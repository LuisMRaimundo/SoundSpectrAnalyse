# Formula Validation Plan — Pass 5 — Harmonic alignment

## 1. Scope

Pass 5: **`harmonic_alignment.py`** (cents, adaptive tolerance, slot counts), as in `FORMULA_EXTRACTION_TABLE_PASS_05_HARMONIC_ALIGNMENT.md`.

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 5-01 | Cents unison | `obs_hz=440`, `exp_hz=440` | **0.0** | `harmonic_alignment._cents(440.0, 440.0)` | `assert_allclose(..., 0.0)` | |
| 5-02 | Cents octave | `obs_hz=880`, `exp_hz=440` | **1200.0** | `harmonic_alignment._cents(880.0, 440.0)` | `assert_allclose` | |
| 5-03 | Adaptive tolerance floor | `sample_rate=None` | **18.0** | `harmonic_alignment._adaptive_tolerance_cents(1000.0, None, None)` | exact | |
| 5-04 | Adaptive tolerance numeric | `expected_hz=100`, `sample_rate=44100`, `n_fft=4096` | Hand-compute \(\Delta f=f_s/N\), \(h^+=100+\Delta f/2\), \(\tau_{\mathrm{bw}}=1200\log_2(h^+/100)\), \(\tau=\max(18,2\tau_{\mathrm{bw}})\) | `harmonic_alignment._adaptive_tolerance_cents` | `assert_allclose`, `atol=1e-6` | |
| 5-05 | Slot count | `f0_hz=100`, `max_frequency_hz=500` | \(N=\lfloor 500/100\rfloor=\) **5** (capped by `max_harmonics` in real call) | `compute_harmonic_alignment_metrics` → `total_expected_harmonic_orders` | match | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
