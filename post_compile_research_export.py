# -*- coding: utf-8 -*-
"""
Post–Stage-2 hook: build ``compiled_density_metrics_research.xlsx`` beside the compiled workbook.

Safe to call after a successful compile; failures are logged and do not affect analysis status.
The research workbook is written without formal Excel **Table** parts (worksheet AutoFilter
only on data sheets); see ``docs/EXPORT_COLUMN_DICTIONARY.md`` (research workbook section).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

__all__ = ("run_research_workbook_export",)


def run_research_workbook_export(
    compiled_workbook_path: Path | str,
    *,
    log: Optional[logging.Logger] = None,
) -> Optional[Path]:
    """
    If ``compiled_workbook_path`` exists, run the research export (overwrite=True).

    Returns the research workbook path on success, or ``None`` if skipped or on failure.
    """
    _log = log or logging.getLogger(__name__)
    compiled = Path(compiled_workbook_path).expanduser().resolve()

    if not compiled.is_file():
        _log.warning(
            "Research workbook export skipped: compiled_density_metrics.xlsx not found: %s",
            compiled,
        )
        return None

    _log.info("Creating reduced research workbook...")
    _log.info("Creating reduced research workbook from: %s", compiled)

    try:
        from tools.export_research_density_workbook import export_research_workbook

        out = export_research_workbook(compiled, output_path=None, overwrite=True)
    except Exception as exc:  # noqa: BLE001
        _log.error(
            "Research workbook export failed for %s: %s",
            compiled,
            exc,
        )
        return None

    _log.info("[OK] compiled_density_metrics_research.xlsx created: %s", out)
    return out
