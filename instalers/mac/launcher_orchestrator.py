# -*- coding: utf-8 -*-
"""PyInstaller entry: Tk pipeline orchestrator GUI."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _dev_source_root() -> Path:
    if os.environ.get("SOUNDSPECTRANALYSE_SOURCE"):
        return Path(os.environ["SOUNDSPECTRANALYSE_SOURCE"]).resolve()
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent.parent,
        here.parent,
        here.parent.parent / "SoundSpectrAnalyse-main_6",
        here.parent.parent / "SoundSpectrAnalyse-github-fix",
        here.parent / "SoundSpectrAnalyse-main_6",
    ]
    for base in candidates:
        if (base / "pipeline_orchestrator_gui.py").is_file():
            return base.resolve()
    return (here.parent.parent / "SoundSpectrAnalyse-main_6").resolve()


def _app_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return _dev_source_root()


def main() -> None:
    root = _app_root()
    os.chdir(root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    import tkinter as tk

    import pipeline_orchestrator_gui as pog

    log = logging.getLogger(
        pog.__name__ if hasattr(pog, "__name__") else "pipeline_orchestrator_gui"
    )
    if not log.hasHandlers():
        log.addHandler(logging.StreamHandler(sys.stdout))
    tk_root = tk.Tk()
    pog.RobustOrchestratorApp(tk_root)
    tk_root.mainloop()


if __name__ == "__main__":
    main()
