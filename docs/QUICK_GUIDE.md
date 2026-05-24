# Quick Guide

## 1) What the software computes

SoundSpectrAnalyse computes per-note spectral metrics from pitched note audio:
- harmonic, inharmonic/residual, and subbass/particle descriptors;
- final H/I/S note-density metrics;
- validation and provenance fields;
- dashboard-ready and chart-ready exports.

## 2) How to run a folder

1. Open the GUI (`pipeline_orchestrator_gui.py`) or orchestrator entrypoint.
2. Select an input folder with note audio files.
3. Keep default density settings (below) unless testing sensitivity.
4. Run full pipeline.
5. Read:
   - compiled workbook: `compiled_density_metrics.xlsx`
   - research workbook: `compiled_density_metrics_research*.xlsx`

## 3) Recommended default settings

- `density_summation_mode = his_weighted`
- `harmonic_density_weight = 1.0`
- `inharmonic_density_weight = 0.5`
- `subbass_density_weight = 0.25`
- `density_salience_threshold_db = -45`
- `density_frequency_ceiling_hz = 5000`

## 4) Which metric to use

Use this for final note density:
- `final_note_density_salience_weighted`

Use this for simple weighted count:
- `final_note_density_count_based`

Use this for raw harmonic partial trend:
- `salient_harmonic_order_count_up_to_density_ceiling_hz`
- or `salient_harmonic_order_count_up_to_5000hz` when ceiling is fixed at 5000 Hz

Use this for body/thickness:
- `spectral_body_thickness_index`

## 5) How to read the dashboard

- Start with mean and trend panels of `final_note_density_salience_weighted`.
- Check top/bottom notes and compare H/I/S component contributions.
- Inspect validation counts (especially fallback/unverified f0 rows).

## 6) How to interpret final density

- High `final_note_density_salience_weighted`: more salient weighted components.
- Compare with `final_note_density_count_based` to separate count effects from salience-strength effects.
- Use component fields (`harmonic_density_component`, `inharmonic_density_component`, `subbass_density_component`) to explain source of density.

## 7) How to check validity quickly

- Confirm formula identity for active mode (for example harmonic-only gives count-based == H).
- Check `acoustic_validation_status` and fallback rows.
- Check Metadata contains mode/weights/threshold/ceiling.
- Check `Analysis_Settings_By_Note` has one row per note and populated analysis settings.
- Check no sheet-level Excel formula errors.

## 8) What not to use as final density

Do **not** use as final note density:
- `density_metric_raw`
- `effective_partial_density`
- `spectral_body_thickness_index`
- `Combined Density Metric`
- `Weighted Combined Metric`
- `Total Metric`
- `density_weighted_sum`

## 9) Common problems

- **No effect from GUI changes**: verify `density_*` settings in `Metadata`.
- **Harmonic counts differ by register**: expected under fixed ceiling (`floor(Fc/f0)` behavior).
- **Too many fallback notes**: inspect f0 provenance fields and acoustic status.
- **Cross-run chart mismatch**: `*_norm_for_chart` is run-relative, not absolute.
- **Legacy confusion**: keep legacy metrics in `Legacy_Compatibility` only.

