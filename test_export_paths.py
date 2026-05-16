"""Publication-safe path redaction for repository exports."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import export_paths as ep


def test_redact_path_str_windows_basename():
    s = r"C:\Users\alice\Desktop\project\clip.wav"
    assert ep.redact_path_str(s) == "clip.wav"


def test_sanitize_for_repo_nested_paths():
    prev = os.environ.get("SOUNDSPECTRANALYSE_EXPORT_ABSOLUTE_PATHS")
    try:
        os.environ["SOUNDSPECTRANALYSE_EXPORT_ABSOLUTE_PATHS"] = ""
        raw = {
            "audio_file": r"C:\Secret\a.wav",
            "nested": {"other_path": r"D:\data\file.bin"},
        }
        out = ep.sanitize_for_repo(raw)
        assert out["audio_file"] == "a.wav"
        assert out["nested"]["other_path"] == "file.bin"
    finally:
        if prev is None:
            os.environ.pop("SOUNDSPECTRANALYSE_EXPORT_ABSOLUTE_PATHS", None)
        else:
            os.environ["SOUNDSPECTRANALYSE_EXPORT_ABSOLUTE_PATHS"] = prev


def test_export_absolute_paths_env_toggle():
    prev = os.environ.get("SOUNDSPECTRANALYSE_EXPORT_ABSOLUTE_PATHS")
    try:
        os.environ["SOUNDSPECTRANALYSE_EXPORT_ABSOLUTE_PATHS"] = "1"
        assert ep.export_absolute_paths() is True
        p = r"C:\Users\x\a.wav"
        assert ep.sanitize_for_repo({"p": p})["p"] == p
    finally:
        if prev is None:
            os.environ.pop("SOUNDSPECTRANALYSE_EXPORT_ABSOLUTE_PATHS", None)
        else:
            os.environ["SOUNDSPECTRANALYSE_EXPORT_ABSOLUTE_PATHS"] = prev
