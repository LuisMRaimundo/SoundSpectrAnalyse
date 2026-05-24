# Post-Merge Public Repository Audit

Date: 2026-05-24  
Scope: post-merge public-documentation hygiene checks only (no metric/formula/GUI/pipeline changes).

| Issue | status before | status now | action taken | remaining risk |
|---|---|---|---|---|
| Documentation test-status contradiction | Present (`VALIDATION_STATUS_812_PASSED_PASSES_1_15.md` and fixed `0 failed` wording conflicted with current documented baseline-failure state) | Fixed | Renamed to `docs/validation/VALIDATION_STATUS.md`; removed fixed full-suite pass/fail claims; updated public references and README wording to point to `docs/FINAL_ACCEPTANCE_REPORT.md` and `docs/KNOWN_BASELINE_TEST_FAILURES.md`. | Low: historical manifests may still include snapshot counts as archival records. |
| instalers typo | Present (`instalers/windows/START-HERE.bat`) | Fixed | Moved file to `installers/windows/START-HERE.bat` via git move; confirmed no remaining `instalers` directory or references. | Low |
| final-density acceptance evidence | Present | Present (verified) | Confirmed existence of `docs/FINAL_ACCEPTANCE_REPORT.md`, `docs/GUI_OPTION_EFFECT_AUDIT.md`, `docs/KNOWN_BASELINE_TEST_FAILURES.md`, `docs/QUICK_GUIDE.md`, `docs/TUTORIAL.md`, `docs/TECHNICAL_MANUAL.md`; verified acceptance claims in final report. | Low |
| GUI audit consistency | Checked | Pass | Verified no `NOT EXPOSED` and no central `FAIL`; central controls are documented as PASS (`density_summation_mode`, weights, salience threshold, ceiling, metadata propagation, workbook propagation). | Low |
| release documentation consistency | Checked | Pass | Reconciled public references for validation status docs and current test-status wording. | Low |

