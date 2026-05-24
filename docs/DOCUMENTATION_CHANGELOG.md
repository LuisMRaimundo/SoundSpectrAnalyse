# Documentation Changelog

## Scope of this pass

Final documentation pass for the accepted H/I/S final-density architecture.

Constraints respected:
- no formula changes in code;
- no metric renames in code;
- no GUI layout changes;
- no density-architecture redesign;
- no algorithm edits for acoustics.

## Files created

- `docs/DOCUMENTATION_CHANGELOG.md`

## Files updated

- `docs/TECHNICAL_MANUAL.md`
- `docs/QUICK_GUIDE.md`
- `docs/TUTORIAL.md`
- `README.md`
- `tests/test_documentation_consistency.py`
- `metrics_dictionary.json` (documentation metadata alignment for final-density entries)

## Formulas documented

- STFT equation and bin-frequency mapping
- salience transform
- final note-density count-based and salience-weighted equations
- density mode identities (`harmonic_only`, `inharmonic_only`, `subbass_only`, `his_weighted`)
- harmonic expected-count and coverage formulations
- body/thickness composite index formulation
- entropy and participation-ratio definitions

## Metrics documented

- final H/I/S density family
- harmonic coverage/occupancy distinction
- body/thickness secondary descriptors
- entropy/participation descriptors
- energy-family separation (`core_*` vs `component_*`)
- legacy/diagnostic exclusions from final-density interpretation

## Tests added/updated

- Updated `tests/test_documentation_consistency.py` to enforce:
  - required docs existence;
  - README links;
  - required formulas/terms in manuals;
  - forbidden claims absent;
  - presence of canonical final-density dictionary entries;
  - legacy metrics marked non-canonical.

## Remaining documentation uncertainties

- Some frequency/magnitude provenance fields can remain unavailable in compiled sources and are explicitly labeled with:
  - `unknown_not_parseable`
  - `frequency_magnitude_fields_recovery_status = partially_unavailable_in_compiled_source`

No unresolved contradiction between acceptance status docs and GUI audit central criteria remains in this documentation pass.

## Formula integrity confirmation

This pass did not modify computational formulas or final-density values. It is documentation and consistency-test focused.
