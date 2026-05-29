from __future__ import annotations

from pathlib import Path

import pandas as pd

import pipeline_orchestrator_gui as pog
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


def _build_phase1_history_csv_from_workbooks(workbooks: list[Path], out_csv: Path) -> Path:
    rows = []
    for wb in workbooks:
        triplet = pog.RobustOrchestratorApp._extract_note_density_feedback(object(), wb)
        diag = pog.RobustOrchestratorApp._extract_note_density_feedback_diagnostics(object(), wb)
        rows.append(
            {
                "source_workbook": str(wb),
                "obs_wH": float(triplet[0]) if triplet is not None else float("nan"),
                "obs_wI": float(triplet[1]) if triplet is not None else float("nan"),
                "obs_wS": float(triplet[2]) if triplet is not None else float("nan"),
                "pure_observation_w_h": diag.get("pure_observation_w_h"),
                "pure_observation_w_i": diag.get("pure_observation_w_i"),
                "pure_observation_w_s": diag.get("pure_observation_w_s"),
                "component_strength_h": diag.get("component_strength_h"),
                "component_strength_i": diag.get("component_strength_i"),
                "component_strength_s": diag.get("component_strength_s"),
                "legacy_component_strength_h_v55": diag.get("legacy_component_strength_h_v55"),
                "legacy_component_strength_i_v55": diag.get("legacy_component_strength_i_v55"),
                "legacy_component_strength_s_v55": diag.get("legacy_component_strength_s_v55"),
                "obs_w_formula_version": diag.get("obs_w_formula_version"),
            }
        )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
    return out_csv


def test_phase1_csv_has_phase7_fields_populated(tmp_path: Path) -> None:
    workbooks = run_stage1_synthetic_notes(
        tmp_path,
        notes=[("C4", 261.63), ("E4", 329.63), ("G4", 392.0)],
    )
    csv_path = _build_phase1_history_csv_from_workbooks(
        workbooks,
        tmp_path / "run" / "phase1_discovered_density_profiles.csv",
    )
    hist = pd.read_csv(csv_path)

    for col in PHASE71_FIELDS:
        assert col in hist.columns
    for col in PHASE71_FIELDS:
        if col == "obs_w_formula_version":
            assert hist[col].astype(str).str.strip().eq("v57_energy_anchored_occupancy").all()
        else:
            assert pd.to_numeric(hist[col], errors="coerce").notna().all()
