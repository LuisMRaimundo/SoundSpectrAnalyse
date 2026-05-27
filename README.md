# SoundSpectrAnalyse

SoundSpectrAnalyse is a spectral-analysis pipeline developed in support of doctoral research in musicology at NOVA University of Lisbon, with a focus on musical texture in twentieth-century repertoire. The instrument analyses individual note recordings and produces an auditable per-note and corpus-level decomposition of spectral content into harmonic, inharmonic, and sub-bass components (H/I/S), supplemented by a battery of psychoacoustic and MIR descriptors. The H/I/S model is treated throughout as an operational measurement framework - anchored in primary sources in acoustics and psychoacoustics - rather than as a universal ontology of musical sound. The pipeline is designed for traceable, reproducible analysis at doctoral standard: every exported metric is accompanied by an epistemic contract, every numeric constant by a provenance class, and every methodological change by a versioned and tested phase entry in CHANGES.md.

## Status

- **Version**: 3.7.0.
- **Python**: >=3.10,<3.12.
- **Development status**: Beta.
- **License**: Proprietary — see `LICENSE` at the repository root.

## What this software does

SoundSpectrAnalyse analyses individual note recordings and produces a multi-sheet workbook of spectral, harmonic, inharmonic, sub-bass, and MIR descriptors per note, together with a corpus-level adaptive density profile. The pipeline runs in two stages:

1. **Stage 1 — per-note analysis** (`proc_audio.AudioProcessor`): STFT, peak picking, F0 estimation, harmonic / inharmonic / sub-bass (H/I/S) partitioning, stiff-string inharmonicity fit, sub-bass policy, MIR descriptors (spectral moments, tristimulus, Aures roughness, ERB-weighted density), and optional temporal segmentation. Output: one `spectral_analysis.xlsx` per note.
2. **Stage 2 — compilation** (`compile_metrics.compile_density_metrics_with_pca`): per-note rows aggregated into `compiled_density_metrics.xlsx` with tier-normalized columns, dissonance metrics, PCA scores, and validation summary. A reduced research workbook (`compiled_density_metrics_research.xlsx`) is then produced for publication-oriented analysis.

An online adaptive engine (`adaptive_density_engine.AdaptiveDensityEngine`) learns a corpus-level (H, I, S) density profile across notes, using pure observations decoupled from the prior (Phase 1) and a Jensen-Shannon divergence reliability gate (Phase 7). The engine state is exported to `adaptive_density_engine_state.json` for reproducibility.

The full pipeline architecture and mathematical foundations are documented in [`docs/TECHNICAL_MANUAL_COMPLETE.md`](docs/TECHNICAL_MANUAL_COMPLETE.md).

## Theoretical anchoring

This software is the analytical instrument supporting the doctoral dissertation of Luís Raimundo (NOVA University of Lisbon, in preparation).

## Installation

The software is developed and tested on Python >=3.10,<3.12. Installation from a clean environment:

```bash
git clone <repository-url>
cd "SoundSpectrAnalys_version 55"
pip install -r requirements.txt
```

Principal runtime dependencies (versions pinned in `pyproject.toml`):

- numerical: numpy, scipy, pandas, numba, scikit-learn
- audio / DSP: librosa, soundfile, pydub
- output: openpyxl, xlsxwriter
- visualisation: matplotlib, seaborn, plotly
- GUI: PyQt5 (legacy), tkinter (current Windows GUI is Tk-based and shipped with the standard library)

Optional development dependencies (testing, linting, type checking) are declared under `[project.optional-dependencies] dev` in `pyproject.toml`:

```bash
pip install -e ".[dev]"
```

The full module manifest (48 top-level modules) is declared under `[tool.setuptools] py-modules` in `pyproject.toml`. An installed wheel ships all scientific modules.

## Usage

### Cross-platform CLI

The canonical entry point is `run_orchestrator.py`, which runs the full Stage 1 + Stage 2 pipeline on a folder of audio files:

```bash
python run_orchestrator.py
```

Equivalently, after installation:

```bash
soundspectranalyse
```

### Windows GUI

A Tk-based graphical orchestrator is available via `run.bat` on Windows, which launches `pipeline_orchestrator_gui.py`. The GUI provides per-stage progress reporting and adaptive-engine diagnostics.

## Outputs

For each input folder of audio files, the pipeline produces an `analysis_results/` directory containing:

| Artefact | Description |
|---|---|
| `<note_name>/spectral_analysis.xlsx` | Per-note multi-sheet workbook (spectrum, peaks, partitioning, descriptors). |
| `compiled_density_metrics.xlsx` | Corpus-level compiled workbook (16 sheets including `Density_Metrics`, `Canonical_Metrics`, `Diagnostic_Metrics`, `Validation_Metrics`, `PCA_*`, `Dissonance_Metrics`, `Analysis_Metadata`). |
| `compiled_density_metrics_research.xlsx` | Reduced research workbook for publication-oriented analysis. |
| `phase1_discovered_density_profiles.csv` | Full adaptive trajectory per note (observation triplets, JS divergence, reliability, confidence). |
| `adaptive_density_engine_state.json` | Final engine state (posterior profile, concentration, confidence). |
| `phase2_application_profile.json` | The profile applied during Stage 2 compilation. |

Column-level documentation is provided in [`docs/EXPORT_COLUMN_DICTIONARY.md`](docs/EXPORT_COLUMN_DICTIONARY.md); formula-level documentation is in [`docs/METRIC_FORMULA_INDEX.md`](docs/METRIC_FORMULA_INDEX.md).

## Scientific governance

Methodological changes to the pipeline are tracked in [`CHANGES.md`](CHANGES.md) with explicit phase markers (phases 1, 7, 7.1, 8 at time of writing). Each phase change is accompanied by phase-organised regression tests under `tests/phase_<n>/`. Symbolic-structure tests for the canonical formulae are under [`tests/formula_validation/`](tests/formula_validation/) and documented in [`docs/validation/FORMULA_VALIDATION_STATUS.md`](docs/validation/FORMULA_VALIDATION_STATUS.md).

Principal methodological commitments:

- **FFT-length-aware normalization** (Phase 8). Peak-bin sums are normalized using `peak_amplitude_sum` (`N_ref/N`) or `peak_power_sum` (`(N_ref/N)²`) factors rather than broadband-L2 factors, eliminating the cross-tier discontinuity introduced by FFT-length tier switching. The empirical step discontinuity on a 1 kHz synthetic benchmark is reduced from approximately 29.9 % to approximately 0.87 %. See `CHANGES.md`, Phase 8 entry.
- **Prior / observation decoupling** (Phase 1). The triplet `pure_observation_w_{h,i,s}` carries the unmixed observation; the prior-smoothed values are retained as `smoothed_w_{h,i,s}_legacy` for backward compatibility only. This is a precondition for any defensible Bayesian update of the corpus-level profile.
- **Formula versioning** (Phase 7.1). `obs_w_formula_version = "v56_occupancy_ratio"` and `density_formula_version = "v5_apply_density_metric_adapted_v6_1"` are exported on every row, permitting cross-version comparability of compiled workbooks.
- **Per-metric epistemic contract** (`metric_contract.py`). Every exported density metric carries an explicit record of its formula, input domain, unit/scale, amplitude basis, power basis, normalization scope, physical interpretation, validity boundary, and ontological family.
- **Per-constant provenance registry** ([`docs/CONSTANTS_PROVENANCE.md`](docs/CONSTANTS_PROVENANCE.md)). Every numeric constant exported by `constants.py` is classified as `primary_source`, `derived`, `convention`, or `internal_default`. Internal defaults are tunable engineering choices documented for auditability rather than concealed.

The canonical bibliography for the theoretical anchors of the scientific modules is [`REFERENCES.md`](REFERENCES.md). Inline `References` blocks in each scientific module cite from it in short form.

## License

Proprietary. All rights reserved. The full notice is contained in the `LICENSE` file at the repository root, summarised here for visibility.

Copyright © 2026 Luís Raimundo. All rights reserved.

This repository and its contents — source code, documentation, mathematical formulations, tests, validation reports, data structures, configuration files, and associated materials — are proprietary research material. No open-source licence is granted. No permission is granted to copy, redistribute, modify, publish, sublicense, sell, reuse, incorporate, train on, derive from, or otherwise exploit this work, in whole or in part, without prior written permission from the copyright holder.

Access to this repository, whether private or shared, does not imply any licence to use, reproduce, redistribute, modify, publish, or derive works from the software or documentation. Authorised users may run the software only for the specific purpose for which access has been granted. Any other use requires prior written permission.

For permission requests, contact: lmr.2020@outlook.pt

## Citation

Citation metadata is provided in `CITATION.cff`.

This work was funded by the Fundação para a Ciência e a Tecnologia (FCT) under grant [2020.08817.BD](https://doi.org/10.54499/2020.08817.BD).

## Acknowledgements

This project was developed by Luís Raimundo with the support and funding of the Fundação para a Ciência e a Tecnologia (FCT) and NOVA University of Lisbon, under doctoral grant [2020.08817.BD](https://doi.org/10.54499/2020.08817.BD).

The author extends sincere gratitude to:


#Isabel Pires# for her academic and supervisory support throughout the development of this work.

#oão Lopes# for his extensive IT assistance and technical consultation throughout the project's development.
