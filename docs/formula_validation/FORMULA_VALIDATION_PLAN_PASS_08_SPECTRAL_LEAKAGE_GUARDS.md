# Formula Validation Plan — Pass 8 — Spectral leakage guards

## 1. Scope

Pass 8: **`spectral_leakage_guards.py`**, as in `FORMULA_EXTRACTION_TABLE_PASS_08_SPECTRAL_LEAKAGE_GUARDS.md`.

## 2. Validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| 8-01 | `leakage_halfwidth_hz` | `sr=44100`, `n_fft=4096`, default main lobe | **\(0.5\times 4\times 44100/4096\)** Hz ≈ **21.533** | `spectral_leakage_guards.leakage_halfwidth_hz(sr=44100, n_fft=4096)` | `assert_allclose`, `atol=1e-3` | |
| 8-02 | Filter candidates | `inharmonic_candidates=[(100.5, 1.0)]`, `harmonic_rep=[100.0]`, `lh=2.0` | Candidate **dropped** (\(|100.5-100|\le 2\)) | `spectral_leakage_guards.filter_inharmonic_peak_candidates` | `len(out)==0` | |

## 3. Implementation status

No tests are created by this document. This is a validation plan only.
