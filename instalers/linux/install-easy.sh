#!/usr/bin/env bash
# One-click setup for Linux. Repo: https://github.com/LuisMRaimundo/SoundSpectrAnalyse
set -euo pipefail
INSTALL_ROOT="${HOME}/.local/share/SoundSpectrAnalyse"
APP_DIR="${INSTALL_ROOT}/app"
VENV_DIR="${INSTALL_ROOT}/venv"
GITHUB_ZIP="https://github.com/LuisMRaimundo/SoundSpectrAnalyse/archive/refs/heads/main.zip"

echo "=== SoundSpectrAnalyse — Easy install (Linux) ==="

if ! command -v python3 >/dev/null 2>&1; then
  echo "Install Python 3.10+ and python3-tk, e.g.:"
  echo "  sudo apt install python3 python3-venv python3-tk"
  exit 1
fi

PYTHON="$(command -v python3)"
minor="$("${PYTHON}" -c 'import sys; print(sys.version_info.minor)')"
if [[ "${minor}" -lt 10 ]]; then
  echo "Python 3.10+ required."
  exit 1
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

BIN="${HOME}/.local/bin/soundspectranalyse-gui"
mkdir -p "${HOME}/.local/bin"
cat > "${BIN}" <<EOF
#!/bin/bash
cd "${APP_DIR}"
exec "${VENV_DIR}/bin/python" pipeline_orchestrator_gui.py
EOF
chmod +x "${BIN}"

DESKTOP="${HOME}/.local/share/applications/soundspectranalyse-orchestrator.desktop"
mkdir -p "$(dirname "${DESKTOP}")"
cat > "${DESKTOP}" <<EOF
[Desktop Entry]
Type=Application
Name=SoundSpectrAnalyse Orchestrator
Exec=${BIN}
Path=${APP_DIR}
Terminal=false
Categories=Audio;Science;
EOF

echo "Done. Run: soundspectranalyse-gui"
echo "Or find it in your application menu."
