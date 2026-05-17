#Requires -Version 5.1
function Resolve-SoundSpectrSourceRoot {
    param([Parameter(Mandatory)][string]$InstallerRoot)

    if ($env:SOUNDSPECTRANALYSE_SOURCE) {
        return $env:SOUNDSPECTRANALYSE_SOURCE
    }

    $parent = Split-Path $InstallerRoot -Parent
    $grandparent = Split-Path $parent -Parent

    $candidates = @(
        $grandparent,
        (Join-Path $parent "SoundSpectrAnalyse-main_6"),
        (Join-Path $grandparent "SoundSpectrAnalyse-main_6"),
        (Join-Path $grandparent "SoundSpectrAnalyse-github-fix")
    )

    foreach ($c in $candidates) {
        if ($c -and (Test-Path (Join-Path $c "pipeline_orchestrator_gui.py"))) {
            return (Resolve-Path $c).Path
        }
    }

    return (Join-Path $grandparent "SoundSpectrAnalyse-main_6")
}
