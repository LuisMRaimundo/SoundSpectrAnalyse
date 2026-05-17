#!/usr/bin/env bash
set -euo pipefail
INSTALLER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${INSTALLER_ROOT}/build-pyinstaller.sh" "$@"

APP_DIR="${INSTALLER_ROOT}/output/app"
TARBALL="${INSTALLER_ROOT}/output/SoundSpectrAnalyse-Linux-x86_64-3.7.0.tar.gz"
if [[ -d "${APP_DIR}" ]]; then
  rm -f "${TARBALL}"
  tar -C "${APP_DIR}" -czf "${TARBALL}" .
  echo "Tarball: ${TARBALL}"
fi

echo "Done."
