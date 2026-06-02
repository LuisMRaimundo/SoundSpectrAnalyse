# main.py — launcher shim (no PyQt)
#
# Supported workflows after the Stage 1 / Stage 2 / Stage 3 pipeline:
#   - Full CLI pipeline:        python run_orchestrator.py
#   - Windows Tk file picker:   run.bat -> pipeline_orchestrator_gui.py
#   - Tk via integrated entry:  python pipeline_orchestrator_integrated.py --gui
#
# This entry point only forwards to the integrated orchestrator's --gui mode,
# which delegates to the standalone file picker.

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parent
    integrated = root / "pipeline_orchestrator_integrated.py"
    print(
        "main.py: launching the file-picker GUI.\n"
        "Full pipeline (Stage 1 + Stage 2 + Stage 3): python run_orchestrator.py\n",
        file=sys.stderr,
        end="",
    )
    if not integrated.is_file():
        print(f"error: missing {integrated}", file=sys.stderr)
        return 1
    return int(
        subprocess.call([sys.executable, str(integrated), "--gui"])
    )


if __name__ == "__main__":
    sys.exit(main())
