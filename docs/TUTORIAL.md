# Tutorial

## Tutorial 1 - Basic run

1. Open GUI and select input folder.
2. Keep defaults:
   - `density_summation_mode = his_weighted`
   - `wH = 1.0`, `wI = 0.5`, `wS = 0.25`
   - `density_salience_threshold_db = -45`
   - `density_frequency_ceiling_hz = 5000`
3. Run full pipeline.
4. Open research workbook and read:
   - `Spectral_Density_Metrics`
   - `Dashboard`
5. Locate primary metric:
   - `final_note_density_salience_weighted`

## Tutorial 2 - Harmonic-only mode

1. Set `density_summation_mode = harmonic_only`.
2. Run analysis.
3. Verify per-note identity:
   - `final_note_density_count_based == salient_harmonic_order_count_up_to_density_ceiling_hz`
   - or fixed alias `salient_harmonic_order_count_up_to_5000hz` for 5000 Hz runs.

## Tutorial 3 - Weighted H/I/S mode

1. Set:
   - mode `his_weighted`
   - `wH=1`, `wI=0.5`, `wS=0.25`
2. Run analysis.
3. Interpret components:
   - `harmonic_density_component`
   - `inharmonic_density_component`
   - `subbass_density_component`
4. Verify:
   - `final_note_density_count_based = wH*H + wI*I + wS*S`

## Tutorial 4 - Clarinet register curve

1. Run clarinet corpus.
2. Plot `MIDI` vs `salient_harmonic_order_count_up_to_density_ceiling_hz` (or fixed-5000 alias).
3. Expected pattern:
   - descending raw harmonic-order counts in higher register under fixed ceiling.
4. Interpretation:
   - this reflects fewer available orders under \( \lfloor F_c/f_0 \rfloor \), not a pipeline bug.

## Tutorial 5 - Cello residual/body analysis

1. Run cello corpus.
2. Compare:
   - `final_note_density_salience_weighted`
   - `core_residual_energy_ratio`
   - `spectral_body_thickness_index`
3. Use this to separate:
   - final density behavior (H/I/S),
   - residual energy behavior,
   - body/thickness behavior.

## Tutorial 6 - Validity checks

1. Check `acoustic_validation_status` and fallback rows.
2. Check `Metadata` for mode, weights, threshold, ceiling.
3. Check `Analysis_Settings_By_Note`:
   - one row per note;
   - populated STFT and density settings.
4. Check `Charts_Data` and `Dashboard` consistency.

## Tutorial 7 - Threshold and ceiling sensitivity

1. Keep same corpus and run with thresholds:
   - `-35`, `-45`, `-55` dB.
2. Compare means of:
   - `final_note_density_salience_weighted`
   - `harmonic_density_component`
   - `inharmonic_density_component`
   - `subbass_density_component`
3. Run with ceilings:
   - `3000`, `5000`, `8000` Hz.
4. Compare:
   - `salient_harmonic_order_count_up_to_density_ceiling_hz`
   - `salient_inharmonic_log_bin_count_up_to_density_ceiling_hz`
5. Expected:
   - more permissive threshold usually increases or preserves salience-based metrics;
   - higher ceiling usually increases or preserves ceiling-aware counts.

