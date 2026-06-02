#!/usr/bin/env python3
"""Stage 3 EWSD contract, diagnostics, and fail-closed assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, List, Optional, Sequence

import numpy as np
import pandas as pd

from tools.ewsd_core import SCRIPT_VERSION

STAGE3_STATUS_OK: Final[str] = "ok"
STAGE3_STATUS_DEGRADED: Final[str] = "degraded"
STAGE3_STATUS_FAILED: Final[str] = "failed"

MERGE_STATUS_FAILED: Final[frozenset[str]] = frozenset(
    {
        "ewsd_computation_failed",
        "no_per_note_workbooks_found",
    }
)


class EwsdStage3Error(RuntimeError):
    """Base error for Stage 3 EWSD contract violations."""


class EwsdComputationFailed(EwsdStage3Error):
    """EWSD core computation raised an exception."""


class EwsdWorkbooksNotFound(EwsdStage3Error):
    """No per-note spectral_analysis workbooks under the analysis folder."""


class EwsdMergeIncomplete(EwsdStage3Error):
    """One or more research rows could not receive EWSD scores."""


@dataclass(frozen=True)
class Stage3MergeResult:
    spectral_density_metrics: pd.DataFrame
    diagnostics: pd.DataFrame
    diagnostics_summary: pd.DataFrame
    status: str
    messages: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return self.status == STAGE3_STATUS_OK


def build_stage3_diagnostics(
    sd: pd.DataFrame,
    *,
    analysis_root: str,
    frequency_ceiling_hz: Optional[float],
    n_workbooks: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise per-note Stage 3 merge outcomes; summary row is a separate metadata frame."""
    empty_note = pd.DataFrame(
        columns=[
            "Note",
            "ewsd_merge_status",
            "ewsd_primary_analysis_eligible",
            "EWSD_score_acoustic_balanced",
            "ewsd_uncertainty_sources",
            "stage3_issue",
        ]
    )
    empty_summary = pd.DataFrame(
        columns=["stage3_status_row", "ewsd_stage3_version", "stage3_issue"]
    )
    if sd is None or sd.empty or "Note" not in sd.columns:
        return empty_note, empty_summary

    rows: list[dict[str, object]] = []
    for _, row in sd.iterrows():
        status = str(row.get("ewsd_merge_status", "")).strip()
        issue = ""
        if status in MERGE_STATUS_FAILED:
            issue = status
        elif status == "note_not_in_ewsd_output":
            issue = "note_not_in_ewsd_output"
        elif not bool(row.get("ewsd_primary_analysis_eligible", False)):
            issue = "not_primary_analysis_eligible"
        rows.append(
            {
                "Note": str(row.get("Note", "")).strip(),
                "ewsd_merge_status": status,
                "ewsd_primary_analysis_eligible": bool(row.get("ewsd_primary_analysis_eligible", False)),
                "EWSD_score_acoustic_balanced": row.get("EWSD_score_acoustic_balanced"),
                "ewsd_uncertainty_sources": row.get("ewsd_uncertainty_sources"),
                "stage3_issue": issue,
            }
        )

    summary = pd.DataFrame(rows)
    meta = pd.DataFrame(
        [
            {
                "stage3_status_row": "__STAGE3_SUMMARY__",
                "ewsd_stage3_version": SCRIPT_VERSION,
                "stage3_issue": (
                    f"analysis_root={analysis_root}; "
                    f"frequency_ceiling_hz={frequency_ceiling_hz}; "
                    f"workbooks={n_workbooks}; rows={len(sd)}"
                ),
            }
        ]
    )
    return summary, meta


def assess_stage3_merge_result(
    sd: pd.DataFrame,
    *,
    include_ewsd: bool,
    global_status: str,
    messages: Sequence[str],
) -> str:
    """Return ok / degraded / failed for the Stage 3 hook."""
    if not include_ewsd:
        return STAGE3_STATUS_OK

    if global_status in MERGE_STATUS_FAILED:
        return STAGE3_STATUS_FAILED

    if sd is None or sd.empty or "ewsd_merge_status" not in sd.columns:
        return STAGE3_STATUS_FAILED

    statuses = sd["ewsd_merge_status"].astype(str)
    if statuses.isin(MERGE_STATUS_FAILED).any():
        return STAGE3_STATUS_FAILED

    missing = statuses.eq("note_not_in_ewsd_output").sum()
    eligible = sd.get("ewsd_primary_analysis_eligible", pd.Series(False)).astype(bool).sum()
    if missing > 0 or eligible == 0:
        return STAGE3_STATUS_DEGRADED

    if any("failed" in str(m).lower() for m in messages):
        return STAGE3_STATUS_DEGRADED

    return STAGE3_STATUS_OK


def enforce_fail_closed(result: Stage3MergeResult) -> None:
    """Raise when Stage 3 status is failed (publication gate)."""
    if result.status != STAGE3_STATUS_FAILED:
        return
    msg = "; ".join(result.messages) if result.messages else "Stage 3 EWSD failed"
    statuses: set[str] = set()
    if "ewsd_merge_status" in result.diagnostics.columns:
        statuses = set(result.diagnostics["ewsd_merge_status"].astype(str))
    if "no_per_note_workbooks_found" in statuses:
        raise EwsdWorkbooksNotFound(msg)
    if "ewsd_computation_failed" in statuses:
        raise EwsdComputationFailed(msg)
    raise EwsdStage3Error(msg)
