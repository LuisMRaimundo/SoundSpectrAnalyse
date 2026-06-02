# Export schema audit repair (v4.0.0)

Addresses the architecture-level incongruences identified in the compiled/research
workbook audit (duplicate Note keys, `density_weighted_sum` semantic overload, phase-2
vs per-note weight conflation, Diagnostic column collisions, merge `_2` duplicates).

## Primary join key

| Column | Use |
|--------|-----|
| **`sample_id`** | **Primary key for joins** — stable per row; survives duplicate `Note` labels (e.g. two G#4 samples) |
| `Note` | Display / pitch label only — **not** safe as sole join key when duplicates exist |

Computed in Stage 2 `Density_Metrics` and propagated to research exports.

## Three density quantities (do not interchange)

| Canonical name | Legacy / companion column | Weights used | Sheet |
|----------------|---------------------------|--------------|-------|
| **`density_raw_phase2_profile_weighted`** | `density_metric_raw` | Phase-2 corpus application profile (H/I/S) | compiled |
| **`density_component_ratio_weighted_sum`** | `density_metric_raw_per_note_balance`, `density_weighted_sum` (compiled) | Per-note `component_*_energy_ratio` | compiled |
| **`richness_weighted_body_density`** | research `density_weighted_sum` | GUI weights × body-ceiling component sums | research |

Compiled `density_weighted_sum_alias_of` now correctly points to
`density_metric_raw_per_note_balance`, not `density_metric_raw`.

## Phase-2 vs per-note weights

| Column | Meaning |
|--------|---------|
| `phase2_harmonic_application_weight` (+ I/S) | Corpus-level adaptive profile applied to `density_metric_raw` |
| `component_harmonic_energy_ratio` (+ I/S) | Per-note **observed** energy ratios |
| `harmonic_density_weight` (Diagnostic sheet) | Renamed to `per_note_harmonic_energy_ratio_diagnostic` on export |

Global `Analysis_Metadata` no longer copies row-0 per-note weights into
`harmonic_density_weight` when phase-2 keys are present.

## Diagnostic vs Density_Metrics collisions

On `Diagnostic_Metrics`, raw-power / wide-frame variants are prefixed, e.g.
`diagnostic_harmonic_energy_sum_raw_power`,
`diagnostic_harmonic_full_spectrum_energy_sum_20khz_raw`.

## Research workbook

- **`Primary_Statistics_Eligible`** replaces `Primary_Statistics_Filtered` (eligibility gate, not QC exclusion).
- **`Stage3_Summary`** holds the former `__STAGE3_SUMMARY__` row; `Stage3_Diagnostics` is note-level only.
- **`sample_row_count`**, **`unique_midi_count`**, **`notes_count_semantics`** clarify row vs pitch counts.
- Identical merge duplicate columns (`*_2`) are dropped before Excel export when values match.

## Re-export required

Existing workbooks on disk retain old semantics until recompiled with v4.0.0+.

See also: `docs/DENSITY_EXPORT_SCHEMA.md`, `docs/EXPORT_COLUMN_DICTIONARY.md`, `CHANGES.md`.
