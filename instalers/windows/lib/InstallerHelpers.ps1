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

function Test-IsPythonPathCandidate {
    param([string]$PythonExe)
    if (-not $PythonExe) { return $false }
    $normalized = $PythonExe.Replace('/', '\')
    if ($normalized -match '\\WindowsApps\\') { return $false }
    if ($normalized -match '\\Microsoft\\WindowsApps\\') { return $false }
    if (-not (Test-Path -LiteralPath $PythonExe)) { return $false }
    try {
        if ((Get-Item -LiteralPath $PythonExe).Length -lt 1024) { return $false }
    } catch { return $false }
    return $true
}

function Get-KnownPythonCandidatePaths {
    $candidates = @()
    $localPython = Join-Path $env:LOCALAPPDATA 'Programs\Python'
    foreach ($folder in @('Python311', 'Python310')) {
        $candidates += Join-Path $localPython "$folder\python.exe"
    }
    foreach ($root in @(${env:ProgramFiles}, ${env:ProgramFiles(x86)})) {
        if (-not $root) { continue }
        foreach ($folder in @('Python311', 'Python310')) {
            $candidates += Join-Path $root "$folder\python.exe"
        }
    }
    return $candidates | Where-Object { Test-IsPythonPathCandidate -PythonExe $_ }
}

function Resolve-PythonViaPyLauncher {
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if (-not $pyCmd) { return $null }
    foreach ($tag in @('3.11', '3.10')) {
        try {
            $exe = & $pyCmd.Source -$tag -c "import sys; print(sys.executable)" 2>$null
            $exe = ($exe | Select-Object -First 1).Trim()
            if ($exe -and (Test-IsPythonPathCandidate -PythonExe $exe) -and (Test-PythonVersionOk -PythonExe $exe)) {
                return (Resolve-Path -LiteralPath $exe).Path
            }
        } catch { }
    }
    return $null
}

function Find-ExistingPython {
    foreach ($exe in (Get-KnownPythonCandidatePaths)) {
        if (Test-PythonVersionOk -PythonExe $exe) {
            return (Resolve-Path -LiteralPath $exe).Path
        }
    }
    $viaPy = Resolve-PythonViaPyLauncher
    if ($viaPy) { return $viaPy }
    $found = @()
    foreach ($name in @('python', 'python3')) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd -and (Test-IsPythonPathCandidate -PythonExe $cmd.Source)) {
            $found += $cmd.Source
        }
    }
    $roots = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python'),
        ${env:ProgramFiles},
        ${env:ProgramFiles(x86)}
    ) | Where-Object { $_ -and (Test-Path $_) }
    foreach ($root in $roots) {
        $found += Get-ChildItem -Path $root -Filter 'python.exe' -Recurse -Depth 3 -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty FullName
    }
    foreach ($exe in ($found | Select-Object -Unique)) {
        if (-not (Test-IsPythonPathCandidate -PythonExe $exe)) { continue }
        if (Test-PythonVersionOk -PythonExe $exe) { return (Resolve-Path -LiteralPath $exe).Path }
    }
    return $null
}

function Wait-ForPythonAfterInstall {
    param([int]$TimeoutSeconds = 90)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        Refresh-SessionPath
        $py = Find-ExistingPython
        if ($py) { return $py }
        Start-Sleep -Seconds 2
    }
    return $null
}

function Test-PythonInstallerExitOk {
    param([int]$ExitCode)
    return ($ExitCode -eq 0 -or $ExitCode -eq 3010)
}

function Install-Python311 {
    Write-InstallLog "Python 3.10-3.11 not found. Installing Python $($script:SoundSpectrConfig.PythonVersion)..."

    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
    } catch { }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        Write-InstallLog 'Trying winget (Python.Python.3.11, current user)...'
        $p = Start-Process -FilePath 'winget' -ArgumentList @(
            'install', '-e', '--id', 'Python.Python.3.11',
            '--accept-package-agreements', '--accept-source-agreements',
            '--disable-interactivity',
            '--scope', 'user'
        ) -Wait -PassThru -NoNewWindow
        Write-InstallLog "winget exit code: $($p.ExitCode)"
        $py = Wait-ForPythonAfterInstall -TimeoutSeconds 60
        if ($py) {
            Write-InstallLog "Python available after winget: $py"
            return $py
        }
        Write-InstallLog 'winget finished but Python not detected; trying python.org installer.' 'WARN'
    } else {
        Write-InstallLog 'winget not available; using python.org installer.' 'WARN'
    }

    $installer = Join-Path $env:TEMP 'python-3.11.9-amd64.exe'
    Write-InstallLog 'Downloading Python installer from python.org...'
    try {
        Invoke-WebRequest -Uri $script:SoundSpectrConfig.PythonInstallerUrl -OutFile $installer -UseBasicParsing
    } catch {
        throw "Could not download Python installer. Check Internet/firewall. $($_.Exception.Message)"
    }
    if (-not (Test-Path -LiteralPath $installer)) {
        throw 'Python installer download failed (file missing).'
    }

    Write-InstallLog 'Running Python installer (per-user, adds to PATH)...'
    $installArgs = @(
        '/passive', 'InstallAllUsers=0', 'PrependPath=1',
        'Include_test=0', 'Include_pip=1', 'Include_launcher=1',
        'AssociateFiles=0', 'SimpleInstall=1'
    )
    $p = Start-Process -FilePath $installer -ArgumentList $installArgs -Wait -PassThru
    Remove-Item -LiteralPath $installer -Force -ErrorAction SilentlyContinue
    Write-InstallLog "python.org installer exit code: $($p.ExitCode)"

    if (-not (Test-PythonInstallerExitOk -ExitCode $p.ExitCode)) {
        throw "Python installer failed (exit $($p.ExitCode)). Install Python 3.11 from https://www.python.org/downloads/ with Add to PATH, then run INSTALL.bat again."
    }

    $py = Wait-ForPythonAfterInstall -TimeoutSeconds 90
    if (-not $py) {
        throw "Python installed but not found yet. Install Python 3.11 from https://www.python.org/downloads/, enable Add to PATH, then run INSTALL.bat again. Log: $script:InstallLogPath"
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
    Write-InstallLog "Downloading project from GitHub ($zipUrl)..."
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
        Write-InstallLog "Copying local project into install folder..."
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
    Write-InstallLog "Creating virtual environment..."
    if (Test-Path $VenvDir) { Remove-Item -LiteralPath $VenvDir -Recurse -Force }
    & $PythonExe -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "venv creation failed." }
    $pip = Join-Path $VenvDir 'Scripts\pip.exe'
    $req = Join-Path $AppDir 'requirements.txt'
    if (-not (Test-Path $req)) { throw "Missing requirements.txt in $AppDir" }
    Write-InstallLog "Installing Python packages (may take 10-20 minutes on first run)..."
    & $pip install --upgrade pip wheel setuptools
    if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed." }
    & $pip install -r $req
    if ($LASTEXITCODE -ne 0) { throw "pip install -r requirements.txt failed." }
    $pyproject = Join-Path $AppDir 'pyproject.toml'
    if (Test-Path $pyproject) {
        Write-InstallLog "Installing SoundSpectrAnalyse package (editable)..."
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
    $desc = 'SoundSpectrAnalyse - spectral analysis pipeline'
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
