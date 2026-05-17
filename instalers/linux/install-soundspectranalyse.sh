#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="${SRC}/SoundSpectrAnalyse-Orchestrator"

if [[ ! -x "${BIN}" ]]; then
  if [[ -f "${SRC}/SoundSpectrAnalyse-Orchestrator" ]]; then
    chmod +x "${SRC}/SoundSpectrAnalyse-Orchestrator"
  else
    echo "ERROR: SoundSpectrAnalyse-Orchestrator not found in ${SRC}" >&2
    echo "Build first: ./build-all.sh" >&2
    exit 1
  fi
fi

INSTALL_DIR="${HOME}/.local/share/SoundSpectrAnalyse"
DESKTOP_FILE="${HOME}/.local/share/applications/soundspectranalyse-orchestrator.desktop"
BIN_LINK="${HOME}/.local/bin/soundspectranalyse-orchestrator"

echo "Installing to ${INSTALL_DIR} ..."
rm -rf "${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}"
cp -a "${SRC}/." "${INSTALL_DIR}/"
chmod +x "${INSTALL_DIR}/SoundSpectrAnalyse-Orchestrator"

mkdir -p "${HOME}/.local/bin" "${HOME}/.local/share/applications"
ln -sf "${INSTALL_DIR}/SoundSpectrAnalyse-Orchestrator" "${BIN_LINK}"

cat > "${DESKTOP_FILE}" <<EOF
[Desktop Entry]
Type=Application
Name=SoundSpectrAnalyse Orchestrator
Comment=Spectral analysis pipeline GUI
Exec=${INSTALL_DIR}/SoundSpectrAnalyse-Orchestrator
Path=${INSTALL_DIR}
Terminal=false
Categories=Audio;Science;Education;
EOF

echo "Installed."
echo "  Run: ${BIN_LINK}"
echo "  Or find 'SoundSpectrAnalyse Orchestrator' in your application menu."
