@echo off
setlocal EnableExtensions
title SoundSpectrAnalyse - Install

set "SRC=%~dp0"
if not exist "%SRC%SoundSpectrAnalyse Orchestrator.exe" (
  echo.
  echo  This script is for a pre-built .exe folder ^(developers only^).
  echo  For normal installation ^(Python + app^), use INSTALL.bat instead:
  echo.
  echo    %SRC%INSTALL.bat
  echo.
  choice /C YN /M "Run INSTALL.bat now"
  if errorlevel 2 exit /b 1
  call "%SRC%INSTALL.bat"
  exit /b %ERRORLEVEL%
)

set "DEST=%LOCALAPPDATA%\Programs\SoundSpectrAnalyse"
echo Installing portable .exe to:
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
