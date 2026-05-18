#Requires -Version 5.1
$cfg = @{ InstallRoot = Join-Path $env:LOCALAPPDATA 'Programs\SoundSpectrAnalyse' }
$root = $cfg.InstallRoot

Write-Host "Removing $root ..."
if (Test-Path $root) {
    Remove-Item -LiteralPath $root -Recurse -Force
}

$links = @(
    (Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\SoundSpectrAnalyse"),
    (Join-Path ([Environment]::GetFolderPath('Desktop')) 'SoundSpectrAnalyse Orchestrator.lnk')
)
foreach ($l in $links) {
    if (Test-Path $l) { Remove-Item -LiteralPath $l -Recurse -Force -ErrorAction SilentlyContinue }
}
Write-Host 'Uninstall complete. Python was not removed (may be used by other programs).'
