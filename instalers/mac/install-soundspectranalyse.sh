#!/usr/bin/env bash
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP="${SRC}/SoundSpectrAnalyse.app"

if [[ ! -d "${APP}" ]]; then
  echo "ERROR: SoundSpectrAnalyse.app not found in ${SRC}" >&2
  echo "Build first: ./build-all.sh" >&2
  exit 1
fi

DEST="/Applications/SoundSpectrAnalyse.app"
echo "Installing to ${DEST} ..."
rm -rf "${DEST}"
cp -R "${APP}" "${DEST}"
echo "Installed. Open from Applications or: open '${DEST}'"
