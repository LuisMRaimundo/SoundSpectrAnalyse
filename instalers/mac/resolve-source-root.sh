#!/usr/bin/env bash
# shellcheck disable=SC2034
resolve_source_root() {
  local installer_root="$1"
  if [[ -n "${SOUNDSPECTRANALYSE_SOURCE:-}" ]]; then
    echo "$(cd "${SOUNDSPECTRANALYSE_SOURCE}" && pwd)"
    return 0
  fi
  local parent grandparent
  parent="$(cd "$(dirname "${installer_root}")" && pwd)"
  grandparent="$(cd "$(dirname "${parent}")" && pwd)"
  local c
  for c in \
    "${grandparent}" \
    "${parent}/SoundSpectrAnalyse-main_6" \
    "${grandparent}/SoundSpectrAnalyse-main_6" \
    "${grandparent}/SoundSpectrAnalyse-github-fix"; do
    if [[ -f "${c}/pipeline_orchestrator_gui.py" ]]; then
      echo "$(cd "${c}" && pwd)"
      return 0
    fi
  done
  echo "${grandparent}/SoundSpectrAnalyse-main_6"
}
