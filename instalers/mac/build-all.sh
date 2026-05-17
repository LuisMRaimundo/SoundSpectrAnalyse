#!/usr/bin/env bash
set -euo pipefail
INSTALLER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${INSTALLER_ROOT}/build-pyinstaller.sh" "$@"

APP_DIR="${INSTALLER_ROOT}/output/app"
ZIP="${INSTALLER_ROOT}/output/SoundSpectrAnalyse-macOS-3.7.0.zip"
if [[ -d "${APP_DIR}/SoundSpectrAnalyse.app" ]]; then
  rm -f "${ZIP}"
  (cd "${APP_DIR}" && zip -r -y "${ZIP}" SoundSpectrAnalyse.app install-soundspectranalyse.sh README.txt)
  echo "Zip: ${ZIP}"
fi

if command -v hdiutil >/dev/null 2>&1 && [[ -d "${APP_DIR}/SoundSpectrAnalyse.app" ]]; then
  DMG="${INSTALLER_ROOT}/output/SoundSpectrAnalyse-macOS-3.7.0.dmg"
  STAGING="${INSTALLER_ROOT}/build/dmg-staging"
  rm -rf "${STAGING}" "${DMG}"
  mkdir -p "${STAGING}"
  cp -R "${APP_DIR}/SoundSpectrAnalyse.app" "${STAGING}/"
  ln -s /Applications "${STAGING}/Applications"
  hdiutil create -volname "SoundSpectrAnalyse" -srcfolder "${STAGING}" -ov -format UDZO "${DMG}"
  echo "DMG: ${DMG}"
fi

echo "Done."
