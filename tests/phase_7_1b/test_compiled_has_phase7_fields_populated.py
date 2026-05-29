from __future__ import annotations

from pathlib import Path

import pandas as pd

import compile_metrics
from tests.phase_7_1b.helpers import run_stage1_synthetic_notes


PHASE71_FIELDS = (
    "pure_observation_w_h",
    "pure_observation_w_i",
    "pure_observation_w_s",
    "component_strength_h",
    "component_strength_i",
    "component_strength_s",
    "legacy_component_strength_h_v55",
    "legacy_component_strength_i_v55",
    "legacy_component_strength_s_v55",
    "obs_w_formula_version",
)


def test_compiled_has_phase7_fields_populated(tmp_path: Path) -> None:
    _ = run_stage1_synthetic_notes(
        tmp_path,
        notes=[("C4", 261.63), ("E4", 329.63), ("G4", 392.0)],
    )

    run_dir = tmp_path / "run"
    compiled_xlsx = tmp_path / "compiled_density_metrics.xlsx"
    compile_metrics.compile_density_metrics_with_pca(
        folder_path=run_dir,
        output_path=compiled_xlsx,
        file_pattern="spectral_analysis.xlsx",
        include_pca=False,
    )

    density_df = pd.read_excel(compiled_xlsx, sheet_name="Density_Metrics")
    assert len(density_df.index) == 3
    for col in PHASE71_FIELDS:
        assert col in density_df.columns
        if col == "obs_w_formula_version":
            assert density_df[col].astype(str).str.strip().eq("v57_energy_anchored_occupancy").all()
        else:
            assert pd.to_numeric(density_df[col], errors="coerce").notna().all()
