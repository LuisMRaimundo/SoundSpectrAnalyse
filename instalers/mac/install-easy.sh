#!/usr/bin/env bash
# One-click setup for macOS (non-expert). Repo: https://github.com/LuisMRaimundo/SoundSpectrAnalyse
set -euo pipefail
INSTALLER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_ROOT="${HOME}/Applications/SoundSpectrAnalyse"
APP_DIR="${INSTALL_ROOT}/app"
VENV_DIR="${INSTALL_ROOT}/venv"
GITHUB_ZIP="https://github.com/LuisMRaimundo/SoundSpectrAnalyse/archive/refs/heads/main.zip"

echo "=== SoundSpectrAnalyse — Easy install (macOS) ==="

find_python() {
  for c in python3.11 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then
      ver="$("$c" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)"
      if [[ "$ver" -ge 10 && "$ver" -le 11 ]]; then
        echo "$(command -v "$c")"
        return 0
      fi
    fi
  done
  return 1
}

PYTHON="$(find_python || true)"
if [[ -z "${PYTHON}" ]]; then
  echo "Installing Python 3.11 via Homebrew (if available)…"
  if command -v brew >/dev/null 2>&1; then
    brew install python@3.11
    PYTHON="$(brew --prefix python@3.11)/bin/python3.11"
  else
    echo "Install Python 3.11 from https://www.python.org/downloads/ then run this script again."
    exit 1
  fi
fi

mkdir -p "${INSTALL_ROOT}"
if [[ ! -f "${APP_DIR}/pipeline_orchestrator_gui.py" ]]; then
  echo "Downloading from GitHub…"
  tmp="$(mktemp -d)"
  curl -fsSL "${GITHUB_ZIP}" -o "${tmp}/repo.zip"
  unzip -q "${tmp}/repo.zip" -d "${tmp}"
  rm -rf "${APP_DIR}"
  mv "${tmp}/SoundSpectrAnalyse-main" "${APP_DIR}"
  rm -rf "${tmp}"
fi

echo "Installing Python packages (10–20 min first time)…"
"${PYTHON}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip wheel
"${VENV_DIR}/bin/pip" install -r "${APP_DIR}/requirements.txt"
[[ -f "${APP_DIR}/pyproject.toml" ]] && "${VENV_DIR}/bin/pip" install -e "${APP_DIR}" || true

LAUNCHER="${INSTALL_ROOT}/Launch-SoundSpectrAnalyse.command"
cat > "${LAUNCHER}" <<EOF
#!/bin/bash
cd "${APP_DIR}"
exec "${VENV_DIR}/bin/python" pipeline_orchestrator_gui.py
EOF
chmod +x "${LAUNCHER}"
cp -f "${LAUNCHER}" "${HOME}/Desktop/SoundSpectrAnalyse Orchestrator.command" 2>/dev/null || true

echo "Done. Open: ${LAUNCHER}"
echo "Or: Desktop → SoundSpectrAnalyse Orchestrator.command"
