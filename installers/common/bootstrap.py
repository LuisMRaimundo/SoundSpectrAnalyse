#!/usr/bin/env python3
"""Bootstrap portable Python + dependencies, then launch SoundSpectrAnalyse."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from config import (
    APP_NAME,
    CLI_ENTRY,
    GET_PIP_PY_PINNED_AT,
    GET_PIP_PY_SHA256,
    GET_PIP_PY_URL,
    GUI_ENTRY,
    PROJECT_ROOT,
    REQUIREMENTS,
    REPO_URL,
    RUNTIME_DIR,
    STAMP_FILE,
    WIN_EMBED_ZIP_SHA256,
    machine_key,
    pbs_download_url,
    platform_key,
    runtime_python_dir,
    runtime_python_exe,
    windows_embed_zip_url,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _log(f"Downloading: {url}")
    urllib.request.urlretrieve(url, dest)


def _sha256_file(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of *path*'s contents."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify_sha256(path: Path, expected_hex: str, label: str) -> None:
    """Raise RuntimeError if *path*'s SHA-256 does not match *expected_hex*.

    The downloaded file is removed on mismatch to prevent accidental reuse.
    """
    expected = expected_hex.strip().lower()
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
        raise RuntimeError(
            f"Internal error: expected hash for {label} is not a valid "
            f"lowercase 64-char hex string"
        )
    actual = _sha256_file(path)
    if actual != expected:
        try:
            path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            f"SHA-256 mismatch for {label}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}\n"
            f"This may indicate a corrupted download or a tampered artefact. "
            f"Please retry; if the failure persists, report it to the project "
            f"maintainer with the URL and the actual hash above."
        )


def _fetch_pbs_sha256(pbs_url: str) -> str:
    """Fetch and parse the per-artefact .sha256 file co-located with a PBS tarball.

    The PBS project (release 20240415 and similar) publishes a file named
    ``<artefact>.sha256`` alongside each release asset. The file's first
    whitespace-delimited token is the hex digest.
    """
    sha_url = pbs_url + ".sha256"
    _log(f"Fetching upstream hash: {sha_url}")
    with urllib.request.urlopen(sha_url) as resp:  # noqa: S310 (trusted upstream)
        raw = resp.read().decode("ascii", errors="replace").strip()
    if not raw:
        raise RuntimeError(f"Empty .sha256 response from {sha_url}")
    token = raw.split()[0].lower()
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise RuntimeError(
            f"Malformed .sha256 content from {sha_url}: {raw!r}"
        )
    return token


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    _log("Running: " + " ".join(cmd))
    subprocess.run(cmd, cwd=cwd or PROJECT_ROOT, env=env, check=True)


def _setup_windows_embed() -> Path:
    py_exe = runtime_python_exe("windows")
    if py_exe.is_file():
        return py_exe

    runtime_dir = runtime_python_dir("windows")
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        zip_path = tmp_path / "python-embed.zip"
        _download(windows_embed_zip_url(), zip_path)
        _verify_sha256(zip_path, WIN_EMBED_ZIP_SHA256, "Windows embeddable Python ZIP")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(runtime_dir)

    get_pip = runtime_dir / "get-pip.py"
    _download(GET_PIP_PY_URL, get_pip)
    _verify_sha256(get_pip, GET_PIP_PY_SHA256, "get-pip.py")

    for pth in runtime_dir.glob("python*._pth"):
        text = pth.read_text(encoding="utf-8")
        if "import site" not in text:
            lines = [ln for ln in text.splitlines() if ln.strip() != "#import site"]
            if not any(ln.strip() == "import site" for ln in lines):
                lines.append("import site")
            pth.write_text("\n".join(lines) + "\n", encoding="utf-8")

    py_exe = runtime_python_exe("windows")
    _run([str(py_exe), str(get_pip)])
    return py_exe


def _setup_pbs(platform_name: str) -> Path:
    py_exe = runtime_python_exe(platform_name)
    if py_exe.is_file():
        return py_exe

    arch = machine_key()
    url = pbs_download_url(platform_name, arch)
    runtime_dir = runtime_python_dir(platform_name)
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    expected_hash = _fetch_pbs_sha256(url)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "python.tar.gz"
        _download(url, archive)
        _verify_sha256(archive, expected_hash, f"PBS Python tarball ({platform_name}/{arch})")
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(tmp_path)
        extracted = next(p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("python"))
        shutil.move(str(extracted), str(runtime_dir))

    if not py_exe.is_file():
        raise RuntimeError(f"Portable Python not found after extract: {py_exe}")
    return py_exe


def ensure_portable_python() -> Path:
    plat = platform_key()
    existing = runtime_python_exe(plat)
    if existing.is_file():
        return existing
    _log(f"Setting up portable Python for {plat} ...")
    if plat == "windows":
        return _setup_windows_embed()
    return _setup_pbs(plat)


def _stamp_payload() -> str:
    req = REQUIREMENTS.stat().st_mtime_ns if REQUIREMENTS.is_file() else 0
    return f"v=1\nroot={PROJECT_ROOT.resolve()}\nrequirements={req}\n"


def ensure_app_installed(py: Path) -> None:
    if STAMP_FILE.is_file():
        try:
            if STAMP_FILE.read_text(encoding="utf-8").strip() == _stamp_payload():
                return
        except OSError:
            pass

    _log(f"Installing {APP_NAME} dependencies (first run may take several minutes) ...")
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"])
    _run([str(py), "-m", "pip", "install", "-r", str(REQUIREMENTS)])
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    STAMP_FILE.write_text(_stamp_payload(), encoding="utf-8")
    _log("Install complete.")


def launch_gui(py: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [str(py), str(GUI_ENTRY)]
    _log(f"Starting {APP_NAME} GUI ...")
    return subprocess.call(cmd, cwd=PROJECT_ROOT, env=env)


def launch_cli(py: Path, passthrough_args: list[str]) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [str(py), str(CLI_ENTRY), *passthrough_args]
    _log(f"Starting {APP_NAME} CLI ...")
    return subprocess.call(cmd, cwd=PROJECT_ROOT, env=env)


def cmd_setup(_: argparse.Namespace) -> int:
    py = ensure_portable_python()
    ensure_app_installed(py)
    _log(f"Ready. Python: {py}")
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    py = ensure_portable_python()
    ensure_app_installed(py)
    if args.cli:
        return launch_cli(py, args.args)
    return launch_gui(py)


def cmd_doctor(_: argparse.Namespace) -> int:
    _log(f"Project root: {PROJECT_ROOT}")
    _log(f"Repository: {REPO_URL}")
    _log(f"get-pip.py pinned at: {GET_PIP_PY_PINNED_AT}")
    _log(f"Platform: {platform_key()} / {machine_key()}")
    py = runtime_python_exe(platform_key())
    _log(f"Portable Python: {py} ({'found' if py.is_file() else 'missing'})")
    _log(f"Install stamp: {STAMP_FILE} ({'ok' if STAMP_FILE.is_file() else 'missing'})")
    _log(f"Requirements file: {REQUIREMENTS} ({'found' if REQUIREMENTS.is_file() else 'missing'})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=f"{APP_NAME} installer bootstrap")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("setup").set_defaults(func=cmd_setup)

    launch = sub.add_parser("launch")
    launch.add_argument("--cli", action="store_true", help="Launch run_orchestrator.py instead of GUI.")
    launch.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to CLI mode.")
    launch.set_defaults(func=cmd_launch)

    sub.add_parser("doctor").set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except subprocess.CalledProcessError as exc:
        _log(f"Command failed with exit code {exc.returncode}.")
        return exc.returncode or 1
    except Exception as exc:
        _log(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if len(sys.argv) == 1:
        sys.argv.append("launch")
    raise SystemExit(main())
