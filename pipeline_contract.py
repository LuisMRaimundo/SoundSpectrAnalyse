"""
Canonical publication-facing pipeline contract (Stage 1 + Stage 2 + Stage 3).

This module is the single source of truth for *which* modules and artefacts
define publication-grade acoustic metrics. Other entry points must either
delegate here or declare themselves legacy/diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

PIPELINE_CONTRACT_VERSION: Final[str] = (
    "SSA_CANONICAL_PIPELINE_2026_06_STAGE1_PROC_AUDIO_STAGE2_COMPILE_STAGE3_EWSD"
)

CANONICAL_STAGE1_MODULE: Final[str] = "proc_audio"
CANONICAL_STAGE1_CLASS: Final[str] = "AudioProcessor"
CANONICAL_STAGE2_MODULE: Final[str] = "compile_metrics"
CANONICAL_STAGE2_FUNCTION: Final[str] = "compile_density_metrics_with_pca"
CANONICAL_STAGE3_MODULE: Final[str] = "post_compile_research_export"
CANONICAL_STAGE3_FUNCTION: Final[str] = "run_research_workbook_export"
CANONICAL_STAGE3_EWSD_CORE: Final[str] = "tools.ewsd_core"
CANONICAL_STAGE3_EWSD_INTEGRATION: Final[str] = "tools.ewsd_research_integration"

CANONICAL_PER_NOTE_WORKBOOK: Final[str] = "spectral_analysis.xlsx"
CANONICAL_COMPILED_WORKBOOK: Final[str] = "compiled_density_metrics.xlsx"
CANONICAL_RESEARCH_WORKBOOK: Final[str] = "compiled_density_metrics_research.xlsx"


@dataclass(frozen=True)
class PipelineContract:
    contract_version: str = PIPELINE_CONTRACT_VERSION
    stage1_module: str = CANONICAL_STAGE1_MODULE
    stage1_class: str = CANONICAL_STAGE1_CLASS
    stage2_module: str = CANONICAL_STAGE2_MODULE
    stage2_function: str = CANONICAL_STAGE2_FUNCTION
    stage3_module: str = CANONICAL_STAGE3_MODULE
    stage3_function: str = CANONICAL_STAGE3_FUNCTION
    stage3_ewsd_core: str = CANONICAL_STAGE3_EWSD_CORE
    stage3_ewsd_integration: str = CANONICAL_STAGE3_EWSD_INTEGRATION
    per_note_workbook: str = CANONICAL_PER_NOTE_WORKBOOK
    compiled_workbook: str = CANONICAL_COMPILED_WORKBOOK
    research_workbook: str = CANONICAL_RESEARCH_WORKBOOK
    publication_output_allowed: bool = True


def get_canonical_pipeline_contract() -> PipelineContract:
    """Return the frozen canonical pipeline contract (no runtime mutation)."""
    return PipelineContract()
