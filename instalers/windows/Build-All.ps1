#Requires -Version 5.1
<#
.SYNOPSIS
  Full pipeline: PyInstaller portable app + optional Inno Setup installer.
#>
[CmdletBinding()]
param(
    [string]$SourceRoot = "",
    [switch]$SkipInno,
    [switch]$SkipPyInstaller
)

$ErrorActionPreference = "Stop"
$root = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
. (Join-Path $root "Resolve-SourceRoot.ps1")

if (-not $SourceRoot) {
    $SourceRoot = Resolve-SoundSpectrSourceRoot -InstallerRoot $root
}

if (-not $SkipPyInstaller) {
    & (Join-Path $root "Build-PyInstaller.ps1") -SourceRoot $SourceRoot
}

if (-not $SkipInno) {
    try {
        & (Join-Path $root "Build-Inno.ps1")
    }
    catch {
        Write-Warning $_.Exception.Message
        Write-Host ""
        Write-Host "Portable app is still usable from: output\app\"
        Write-Host "Install Inno Setup 6 and run .\Build-Inno.ps1 to create the setup .exe"
    }
}

$zip = Join-Path $root "output\SoundSpectrAnalyse-Portable-3.7.0.zip"
$appDir = Join-Path $root "output\app"
if (Test-Path $appDir) {
    if (Test-Path $zip) { Remove-Item -Force $zip }
    Compress-Archive -Path $appDir -DestinationPath $zip -Force
    Write-Host "Zip: $zip"
}

Write-Host "Done."
