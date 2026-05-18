#Requires -Version 5.1

function Write-InstallLog {
    param([string]$Message, [string]$Level = 'INFO')
    $line = "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] [$Level] $Message"
    if ($script:InstallLogPath) {
        Add-Content -LiteralPath $script:InstallLogPath -Value $line -Encoding UTF8
    }
    switch ($Level) {
        'ERROR' { Write-Host $line -ForegroundColor Red }
        'WARN'  { Write-Host $line -ForegroundColor Yellow }
        default { Write-Host $line }
    }
}

function Refresh-SessionPath {
    $machine = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    $user = [Environment]::GetEnvironmentVariable('Path', 'User')
    $env:Path = "$machine;$user"
}

function Test-PythonVersionOk {
    param([string]$PythonExe)
    if (-not (Test-Path -LiteralPath $PythonExe)) { return $false }
    try {
        $out = & $PythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
        if ($out -match '^3\.(\d+)$') {
            $minor = [int]$Matches[1]
            return ($minor -ge $script:SoundSpectrConfig.PythonMinMinor -and $minor -le $script:SoundSpectrConfig.PythonMaxMinor)
        }
    } catch { }
    return $false
}

function Find-ExistingPython {
    $found = @()
    $names = @('python', 'python3', 'py')
    foreach ($name in $names) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { $found += $cmd.Source }
    }
    $roots = @(
        (Join-Path $env:LOCALAPPDATA "Programs\Python"),
        ${env:ProgramFiles},
        ${env:ProgramFiles(x86)}
    ) | Where-Object { $_ -and (Test-Path $_) }
    foreach ($root in $roots) {
        $found += Get-ChildItem -Path $root -Filter 'python.exe' -Recurse -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    }
    foreach ($exe in ($found | Select-Object -Unique)) {
        if (Test-PythonVersionOk -PythonExe $exe) { return (Resolve-Path -LiteralPath $exe).Path }
    }
    return $null
}

function Install-Python311 {
    Write-InstallLog "Python 3.10–3.11 not found. Installing Python $($script:SoundSpectrConfig.PythonVersion)…"

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-InstallLog "Trying winget (Python.Python.3.11, current user)…"
        $p = Start-Process -FilePath 'winget' -ArgumentList @(
            'install', '-e', '--id', 'Python.Python.3.11',
            '--accept-package-agreements', '--accept-source-agreements',
            '--scope', 'user'
        ) -Wait -PassThru -NoNewWindow
        Refresh-SessionPath
        $py = Find-ExistingPython
        if ($py) {
            Write-InstallLog "Python installed via winget: $py"
            return $py
        }
        Write-InstallLog "winget finished (exit $($p.ExitCode)) but python not on PATH yet." 'WARN'
    }

    $installer = Join-Path $env:TEMP 'python-3.11.9-amd64.exe'
    Write-InstallLog "Downloading Python installer from python.org…"
    Invoke-WebRequest -Uri $script:SoundSpectrConfig.PythonInstallerUrl -OutFile $installer -UseBasicParsing
    Write-InstallLog "Running Python installer (quiet, user scope)…"
    $args = @(
        '/quiet', 'InstallAllUsers=0', 'PrependPath=1',
        'Include_test=0', 'Include_pip=1', 'AssociateFiles=0'
    )
    $p = Start-Process -FilePath $installer -ArgumentList $args -Wait -PassThru
    Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    Refresh-SessionPath
    Start-Sleep -Seconds 3
    $py = Find-ExistingPython
    if (-not $py) {
        throw "Python installation did not complete. Install Python 3.11 manually from https://www.python.org/downloads/ and run this installer again."
    }
    Write-InstallLog "Python installed: $py"
    return $py
}

function Get-LocalSourceCopy {
    param([string]$InstallerRoot)
    . (Join-Path $InstallerRoot 'Resolve-SourceRoot.ps1')
    $candidate = Resolve-SoundSpectrSourceRoot -InstallerRoot $InstallerRoot
    if ((Test-Path (Join-Path $candidate $script:SoundSpectrConfig.GuiScript))) {
        Write-InstallLog "Using local source copy: $candidate"
        return (Resolve-Path -LiteralPath $candidate).Path
    }
    return $null
}

function Save-GitHubSource {
    param([string]$DestAppDir)
    $zipUrl = $script:SoundSpectrConfig.GitHubZipUrl
    $zipPath = Join-Path $env:TEMP 'SoundSpectrAnalyse-main.zip'
    $extractRoot = Join-Path $env:TEMP 'SoundSpectrAnalyse-extract'
    Write-InstallLog "Downloading project from GitHub ($zipUrl)…"
    if (Test-Path $extractRoot) { Remove-Item -LiteralPath $extractRoot -Recurse -Force }
    Invoke-WebRequest -Uri $zipUrl -OutFile $zipPath -UseBasicParsing
    Expand-Archive -LiteralPath $zipPath -DestinationPath $extractRoot -Force
    Remove-Item -LiteralPath $zipPath -Force -ErrorAction SilentlyContinue
    $inner = Get-ChildItem -LiteralPath $extractRoot -Directory | Select-Object -First 1
    if (-not $inner) { throw "Downloaded archive was empty." }
    if (Test-Path $DestAppDir) { Remove-Item -LiteralPath $DestAppDir -Recurse -Force }
    New-Item -ItemType Directory -Force -Path (Split-Path $DestAppDir -Parent) | Out-Null
    Move-Item -LiteralPath $inner.FullName -Destination $DestAppDir
    Remove-Item -LiteralPath $extractRoot -Force -Recurse -ErrorAction SilentlyContinue
    Write-InstallLog "Application files saved to: $DestAppDir"
}

function Initialize-AppSource {
    param(
        [string]$InstallerRoot,
        [string]$DestAppDir,
        [switch]$ForceRefresh
    )
    if ((Test-Path $DestAppDir) -and -not $ForceRefresh) {
        if (Test-Path (Join-Path $DestAppDir $script:SoundSpectrConfig.GuiScript)) {
            Write-InstallLog "Using existing install at: $DestAppDir"
            return
        }
    }
    $local = Get-LocalSourceCopy -InstallerRoot $InstallerRoot
    if ($local -and -not $ForceRefresh) {
        Write-InstallLog "Copying local project into install folder…"
        if (Test-Path $DestAppDir) { Remove-Item -LiteralPath $DestAppDir -Recurse -Force }
        Copy-Item -LiteralPath $local -Destination $DestAppDir -Recurse -Force
        return
    }
    Save-GitHubSource -DestAppDir $DestAppDir
}

function Initialize-PythonVenv {
    param(
        [string]$PythonExe,
        [string]$VenvDir,
        [string]$AppDir
    )
    Write-InstallLog "Creating virtual environment…"
    if (Test-Path $VenvDir) { Remove-Item -LiteralPath $VenvDir -Recurse -Force }
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    $pip = Join-Path $VenvDir 'Scripts\pip.exe'
    $req = Join-Path $AppDir 'requirements.txt'
    if (-not (Test-Path $req)) { throw "Missing requirements.txt in $AppDir" }
    Write-InstallLog "Installing Python packages (may take 10–20 minutes on first run)…"
    & $pip install --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $pip install -r $req
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed." }
    $pyproject = Join-Path $AppDir 'pyproject.toml'
    if (Test-Path $pyproject) {
        Write-InstallLog "Installing SoundSpectrAnalyse package (editable)…"
        & $pip install -e $AppDir
        if ($LASTEXITCODE -ne 0) { Write-InstallLog "pip install -e failed; GUI may still run." 'WARN' }
    }
    Write-InstallLog "Dependencies installed."
}

function New-ShortcutFile {
    param(
        [string]$ShortcutPath,
        [string]$TargetPath,
        [string]$Arguments,
        [string]$WorkingDirectory,
        [string]$Description
    )
    $dir = Split-Path $ShortcutPath -Parent
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Force -Path $dir | Out-Null }
    $wsh = New-Object -ComObject WScript.Shell
    $sc = $wsh.CreateShortcut($ShortcutPath)
    $sc.TargetPath = $TargetPath
    $sc.Arguments = $Arguments
    $sc.WorkingDirectory = $WorkingDirectory
    $sc.Description = $Description
    $sc.Save()
}

function Register-Shortcuts {
    param(
        [string]$InstallRoot,
        [string]$AppDir,
        [string]$VenvDir
    )
    $pythonw = Join-Path $VenvDir 'Scripts\pythonw.exe'
    $python = Join-Path $VenvDir 'Scripts\python.exe'
    $launcher = if (Test-Path $pythonw) { $pythonw } else { $python }
    $gui = Join-Path $AppDir $script:SoundSpectrConfig.GuiScript
    $args = "`"$gui`""
    $desc = 'SoundSpectrAnalyse — spectral analysis pipeline'
    $startMenu = Join-Path $env:APPDATA "Microsoft\Windows\Start Menu\Programs\SoundSpectrAnalyse"
    New-ShortcutFile -ShortcutPath (Join-Path $startMenu 'SoundSpectrAnalyse Orchestrator.lnk') `
        -TargetPath $launcher -Arguments $args -WorkingDirectory $AppDir -Description $desc
    $desktop = [Environment]::GetFolderPath('Desktop')
    New-ShortcutFile -ShortcutPath (Join-Path $desktop 'SoundSpectrAnalyse Orchestrator.lnk') `
        -TargetPath $launcher -Arguments $args -WorkingDirectory $AppDir -Description $desc
    $launchBat = Join-Path $InstallRoot 'Launch-SoundSpectrAnalyse.bat'
    @"
@echo off
cd /d "$AppDir"
"$launcher" $args
"@ | Set-Content -LiteralPath $launchBat -Encoding ASCII
    Write-InstallLog "Shortcuts created (Start menu + Desktop)."
}

function Test-TkAvailable {
    param([string]$VenvPython)
    & $VenvPython -c "import tkinter; tkinter.Tk().destroy()" 2>$null
    return ($LASTEXITCODE -eq 0)
}
