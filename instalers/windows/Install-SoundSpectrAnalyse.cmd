@echo off
setlocal EnableExtensions
title SoundSpectrAnalyse - Install

set "SRC=%~dp0"
if not exist "%SRC%SoundSpectrAnalyse Orchestrator.exe" (
  echo ERROR: Run this from the folder that contains "SoundSpectrAnalyse Orchestrator.exe"
  echo        Build first with Build-All.ps1 on a developer PC.
  pause
  exit /b 1
)

set "DEST=%LOCALAPPDATA%\Programs\SoundSpectrAnalyse"
echo Installing to:
echo   %DEST%
echo.

mkdir "%DEST%" 2>nul
xcopy /E /I /Y "%SRC%*" "%DEST%\" >nul
if errorlevel 1 (
  echo Copy failed.
  pause
  exit /b 1
)

set "START=%APPDATA%\Microsoft\Windows\Start Menu\Programs"
mkdir "%START%\SoundSpectrAnalyse" 2>nul

powershell -NoProfile -Command ^
  "$s = New-Object -ComObject WScript.Shell; " ^
  "$l = $s.CreateShortcut('%START%\SoundSpectrAnalyse\SoundSpectrAnalyse Orchestrator.lnk'); " ^
  "$l.TargetPath = '%DEST%\SoundSpectrAnalyse Orchestrator.exe'; " ^
  "$l.WorkingDirectory = '%DEST%'; $l.Save()"

echo.
echo Installed. Start menu: SoundSpectrAnalyse ^> SoundSpectrAnalyse Orchestrator
echo.
pause
endlocal
