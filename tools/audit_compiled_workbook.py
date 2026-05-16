#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audit ``compiled_density_metrics.xlsx`` for Stage-2 structural invariants.

Usage::

    python tools/audit_compiled_workbook.py path/to/compiled_density_metrics.xlsx \\
        [path/to/per_note/spectral_analysis.xlsx]

The optional second path enables Harmonic Spectrum interpolation checks.
Exit code 1 when any hard (``blocker:``) invariant fails; 2 on usage / missing file.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tests.pipeline_workbook_audit import audit_cli_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(audit_cli_main())
