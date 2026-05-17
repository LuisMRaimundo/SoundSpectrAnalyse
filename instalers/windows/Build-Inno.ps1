#Requires -Version 5.1
<#
.SYNOPSIS
  Compile SoundSpectrAnalyse-Setup-3.7.0.exe with Inno Setup 6.
#>
[CmdletBinding()]
param(
    [string]$IsccPath = ""
)

$ErrorActionPreference = "Stop"
$root = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$appExe = Join-Path $root "output\app\SoundSpectrAnalyse Orchestrator.exe"
if (-not (Test-Path $appExe)) {
    throw "Run .\Build-PyInstaller.ps1 first. Missing: $appExe"
}

if (-not $IsccPath) {
    $candidates = @(
        "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
        "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
    )
    foreach ($c in $candidates) {
        if (Test-Path $c) { $IsccPath = $c; break }
    }
}
if (-not $IsccPath -or -not (Test-Path $IsccPath)) {
    throw @"
Inno Setup 6 not found. Install from https://jrsoftware.org/isinfo.php
Then re-run, or pass: -IsccPath 'C:\Program Files (x86)\Inno Setup 6\ISCC.exe'
"@
}

$iss = Join-Path $root "inno\SoundSpectrAnalyse.iss"
& $IsccPath $iss
if ($LASTEXITCODE -ne 0) { throw "ISCC failed (exit $LASTEXITCODE)" }

$setup = Join-Path $root "output\SoundSpectrAnalyse-Setup-3.7.0.exe"
Write-Host "SUCCESS: $setup"
