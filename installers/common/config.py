"""Paths and download URLs for SoundSpectrAnalyse autonomous installers."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

APP_NAME = "SoundSpectrAnalyse"
PYTHON_VERSION = "3.11.9"
PBS_TAG = "20240415"

# Pinned versions and hashes for hardcoded external artefacts.
# PBS tarball hashes are NOT pinned here — they are fetched at install time
# from the per-artefact .sha256 files co-located with each upstream release
# asset. See bootstrap._fetch_pbs_sha256.
GET_PIP_PY_URL = "https://bootstrap.pypa.io/pip/get-pip.py"
GET_PIP_PY_SHA256 = "66904bccb878e363db6236ea900e6935e507dcb887e9f178f6212edfe7f46a76"
GET_PIP_PY_PINNED_AT = "2026-05-27"
WIN_EMBED_ZIP_SHA256 = "009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b"

INSTALLERS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = INSTALLERS_DIR.parent
RUNTIME_DIR = INSTALLERS_DIR / "runtime"
STAMP_FILE = RUNTIME_DIR / ".install_ok"
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
GUI_ENTRY = PROJECT_ROOT / "pipeline_orchestrator_gui.py"
CLI_ENTRY = PROJECT_ROOT / "run_orchestrator.py"
REPO_URL = "https://github.com/LuisMRaimundo/SoundSpectrAnalyse"


def platform_key() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    if sys.platform.startswith("linux"):
        return "linux"
    raise RuntimeError(f"Unsupported OS: {sys.platform}")


def machine_key() -> str:
    machine = platform.machine().lower()
    if machine in {"amd64", "x86_64", "x64"}:
        return "x86_64"
    if machine in {"arm64", "aarch64"}:
        return "aarch64"
    raise RuntimeError(f"Unsupported CPU architecture: {platform.machine()}")


def pbs_artifact(platform_name: str, arch: str) -> str:
    triples = {
        ("windows", "x86_64"): "x86_64-pc-windows-msvc",
        ("macos", "x86_64"): "x86_64-apple-darwin",
        ("macos", "aarch64"): "aarch64-apple-darwin",
        ("linux", "x86_64"): "x86_64-unknown-linux-gnu",
        ("linux", "aarch64"): "aarch64-unknown-linux-gnu",
    }
    triple = triples.get((platform_name, arch))
    if not triple:
        raise RuntimeError(f"No portable Python build for {platform_name} / {arch}")
    return f"cpython-{PYTHON_VERSION}+{PBS_TAG}-{triple}-install_only.tar.gz"


def pbs_download_url(platform_name: str, arch: str) -> str:
    name = pbs_artifact(platform_name, arch)
    return (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        f"{PBS_TAG}/{name}"
    )


def windows_embed_zip_url() -> str:
    return (
        f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
        f"python-{PYTHON_VERSION}-embed-amd64.zip"
    )


def runtime_python_dir(platform_name: str) -> Path:
    return RUNTIME_DIR / platform_name / "python"


def runtime_python_exe(platform_name: str) -> Path:
    base = runtime_python_dir(platform_name)
    if platform_name == "windows":
        return base / "python.exe"
    return base / "bin" / "python3"
