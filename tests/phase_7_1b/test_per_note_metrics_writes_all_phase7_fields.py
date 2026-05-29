from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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


def test_per_note_metrics_writes_all_phase7_fields(tmp_path: Path) -> None:
    workbooks = run_stage1_synthetic_notes(tmp_path, notes=[("C4", 261.63)])
    metrics_df = pd.read_excel(workbooks[0], sheet_name="Metrics")
    row = metrics_df.iloc[0]

    for col in PHASE71_FIELDS:
        assert col in metrics_df.columns

    obs_sum = float(row["pure_observation_w_h"]) + float(row["pure_observation_w_i"]) + float(
        row["pure_observation_w_s"]
    )
    assert abs(obs_sum - 1.0) <= 1e-6

    for col in ("component_strength_h", "component_strength_i", "component_strength_s"):
        val = float(row[col])
        assert np.isfinite(val)

    for col in (
        "legacy_component_strength_h_v55",
        "legacy_component_strength_i_v55",
        "legacy_component_strength_s_v55",
    ):
        val = float(row[col])
        assert np.isfinite(val)

    assert str(row["obs_w_formula_version"]).strip() == "v57_energy_anchored_occupancy"
