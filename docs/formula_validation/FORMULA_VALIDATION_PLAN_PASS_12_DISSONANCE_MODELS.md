# Formula Validation Plan — Pass 12 — Dissonance Models

## 1. Scope

This plan defines small, hand-checkable numerical fixtures for the **non-ambiguous** formulas documented in `docs/formula_extraction/FORMULA_EXTRACTION_TABLE_PASS_12_DISSONANCE_MODELS.md` (Pass 12): project-owned **`dissonance_models.py`**, plus **narrow** coupling checks for **`proc_audio.AudioProcessor.calculate_dissonance_metrics`** (bookkeeping only, no full STFT pipeline) and **`compile_metrics.extract_dissonance_metrics`**. It is intended for later `pytest` implementation; **no tests are created here**.

## 2. Included validation cases

| Case | Formula / expression | Input example | Manual expected result | Python target | Suggested assertion | Notes |
|---|---|---|---|---|---|---|
| DT-1 | `total_dissonance` cross sum | `SetharesDissonance()`; `partials1=[(100.0,1.0)]`, `partials2=[(200.0,1.0)]` | \(D = d(100,200,1,1)\) one term | `model.total_dissonance(partials1, partials2)` | `assert_allclose(D, model.pure_tones_dissonance(100.,200.,1.,1.), rtol=1e-12)` | Single cross pair. |
| ST-1 | Shifted spectrum | `DissonanceModel` default: `base=[(440.0,1.0)]`, `interval=2.0` | Shifted list `[(880.0,1.0)]`; `same_timbre` equals `total_dissonance(base, shifted)` | `SetharesDissonance(curve_mode="cross").same_timbre_dissonance([(440.,1.)], 2.)` vs explicit `total_dissonance` | Equality to manual `total_dissonance` on constructed shifted list | Uses **cross** mode so behaviour matches base class path. |
| CV-1 | `np.linspace` keys | `calculate_dissonance_curve(..., 1.0, 2.0, 3)` | Ratio keys `1.0`, `1.5`, `2.0` (order iteration may follow float keys — compare `sorted(curve.keys())`) | `SetharesDissonance().calculate_dissonance_curve([(440.,1.)], 1., 2., 3)` | `assert_allclose(sorted(curve.keys()), [1.,1.5,2.])` | Values are model outputs; only **abscissae** asserted as hand check. |
| LM-1 | Clear interior minimum | `curve = {1.0:0.5, 1.5:0.05, 2.0:0.5}`, `sensitivity=0.01` | Middle satisfies code rule: \(0.05<0.5\), \(0.05<0.5\), and \(0.05<0.5-0.01\) | `DissonanceModel("x","").find_local_minima(curve, 0.01)` | `assert 1.5 in minima` (or `minima==[1.5]`) | Uses abstract class method only. |
| LM-2 | Asymmetric sensitivity | `curve = {1.0:0.06, 1.5:0.055, 2.0:0.06}` | Strict two-sided minimum at `1.5`, but **fails** third clause \(0.055 \nless 0.06-0.01\)) → **excluded** | `find_local_minima(curve, 0.01)` | `assert 1.5 not in minima` | Documents Pass 12 “ambiguous” rule; **do not** treat as psychoacoustic truth. |
| SS-1 | \(s(f)\) | `f=100.0`, defaults | \(s = 0.24/(0.0207\cdot 100+18.96)=0.24/21.03\) | `SetharesDissonance()._s(100.)` | `assert_allclose(out, 0.24/21.03, rtol=1e-12)` | |
| SP-1 | Sethares pairwise elementary | `f1=100,f2=200,a1=1,a2=2`, defaults | Compute \(y=s(100)\cdot100\), then \(d=\min(1,2)\cdot(\mathrm{e}^{-3.5y}-\mathrm{e}^{-5.75y})\), clip at `0` | `SetharesDissonance().pure_tones_dissonance(100.,200.,1.,2.)` | `assert_allclose(out, manual, rtol=1e-9)` | Build `manual` in test with same `math.exp`. |
| SK-1 | `_pairwise_sum` three partials | Frequencies `(300,100,200)` amps all `1` — sorted to `100,200,300` | \(D=d_{12}+d_{13}+d_{23}\) three Sethares calls | `SetharesDissonance()._pairwise_sum([(300.,1.),(100.,1.),(200.,1.)])` | `assert_allclose(out, d12+d13+d23)` | Order-independence after internal sort. |
| SX-1 | `same_timbre` cross vs full | One partial `(100,1)`, `r=2`, default Sethares `curve_mode="full"` vs `curve_mode="cross"` | Two **different** finite scalars; cross equals `total` on base vs shifted only | Two model instances or re-init | `assert cross_val == total_cross_expected`; `assert full_val != cross_val` typically | If equal in a corner case, widen freq separation; goal is structural difference. |
| CM-1 | `calculate_dissonance_metric` modes | `pd.DataFrame` three rows Hz `100,200,300`, `Amplitude` `1,2,1`; `SetharesDissonance(metric_mode=..., metric_scale=10)` | `sum` \(=D_{\mathrm{tot}}\); `mean_pair` \(=D_{\mathrm{tot}}/3\); `mean_pair_scaled` \(=\) mean\(\times 10\); `minamp_norm` \(=D_{\mathrm{tot}}/S_{\min}\) with \(S_{\min}=1+1+1=3\) | `DissonanceModel.calculate_dissonance_metric` via concrete `Sethares` with each `metric_mode` | Relational: `sum==3*mean`, `scaled==10*mean`, `minamp==tot/3` | Compare modes to each other + one absolute `tot` from `sum`. |
| HK-1 | CBW | `f_bar=100` | \(1.72\cdot 100^{0.65}\) | `HutchinsonKnopoffDissonance.cbw(100.)` | `assert_allclose(out, 1.72*(100**0.65), rtol=1e-12)` | |
| HK-2 | `g(y)` table interp | `y=0.25` (knot in `DEFAULT_G_TABLE`) | Table knot \(g(0.25)=0.60\) | `HutchinsonKnopoffDissonance().g(0.25)` | `assert_allclose(out, 0.60, rtol=0, atol=1e-12)` | Or match `numpy.interp` on table arrays. |
| HK-3 | HK pure two tones | e.g. `f1=400,f2=500,a1=a2=1` | Manual: \(\bar f=450\), `cbw`, \(y=\Delta f/\mathrm{cbw}\), `g(y)`, \(d=a_1a_2g/(a_1^2+a_2^2)\)` | `pure_tones_dissonance(400.,500.,1.,1.)` | `assert_allclose(out, manual, rtol=1e-9)` | Pick frequencies so \(y\le1.2\) and \(g>0\). |
| VA-1 | Vassilakis \(R\) | `f1=100,f2=200,a1=1,a2=2` | \(A_1=2,A_2=1\), \(\mathrm{AF}=2/3\), \(x=s(100)\cdot100\), spectral term, \(R=\tfrac12(2)^{0.1}(2/3)^{3.11}(\mathrm{e}^{-b_1x}-\mathrm{e}^{-b_2x})\) | `VassilakisDissonance().pure_tones_dissonance(100.,200.,1.,2.)` | `assert_allclose(out, manual, rtol=1e-9)` | Same \(b_1,b_2,x^\star,s_1,s_2\) as Sethares in code. |
| AM-1 | `calculate_all_dissonance_metrics` | Minimal valid `DataFrame` (≥2 partials, Hz>0, Amplitude>0) | Dict keys ⊆ `{"sethares","hutchinson-knopoff","vassilakis"}`; each value **finite** `float` | `calculate_all_dissonance_metrics(df)` | `assert set(out)==expected_keys`; `assert all(np.isfinite(v) for v in out.values())` | Do not assert universal numeric targets. |
| CP-1 | Min–max normalisation | `vals=[0.2,0.5,0.8]` | Normalised `[0.,0.5,1.]` | Inline same formula as `compare_dissonance_models` | `assert_allclose(norm, [(v-0.2)/(0.6) for v in vals])` | Unit-test helper mirroring `(v-vmin)/(vmax-vmin)` when `vmax>vmin`. |
| PA-1 | Pair count bookkeeping | Integer `n=5` | \(\binom{5}{2}=10\) | Pure formula `n*(n-1)//2` **or** attribute set by a thin stub that only assigns `dissonance_partial_count_after_cap=5` and reads `dissonance_pair_count_after_cap` if exposed | Match `10` | **No** `AudioProcessor` full pipeline required. |
| EX-1 | `extract_dissonance_metrics` | `dfs={"Sheet1": pd.DataFrame({"X Dissonance Y": [np.nan, 1.5, 2.0]})}` | Output `{"X Dissonance Y": 1.5}` | `extract_dissonance_metrics(dfs)` | `assert out["X Dissonance Y"]==1.5` | Substring `"Dissonance"` + first finite row. |

## 3. Deferred / human-review cases

Defer numerical or design “truth” to humans / literature, **not** automated gold values:

- Literature fidelity of Sethares / Vassilakis default constants vs published sources.
- Fidelity of `HutchinsonKnopoffDissonance.DEFAULT_G_TABLE` to Hutchinson & Knopoff (1978) Figure 1.
- Whether `find_local_minima`’s **left-neighbour minus `sensitivity`** clause is intentional UI tuning or a typo vs symmetric strict local minima.
- Physical adequacy of the harmonic partial cap \(K=`DISSONANCE_PAIRWISE_PARTIAL_CAP`\)` relative to real spectra.
- `extract_dissonance_metrics` substring rule (`"Dissonance" in column`) as a long-term export contract.

## 4. Implementation status

No tests are created by this document. This is a validation plan only.
