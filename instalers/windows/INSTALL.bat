@echo off
title SoundSpectrAnalyse - Easy Installer
cd /d "%~dp0"

echo.
echo  SoundSpectrAnalyse - automatic setup
echo  (Python + libraries + shortcuts)
echo.
echo  GitHub: https://github.com/LuisMRaimundo/SoundSpectrAnalyse
echo.
echo  Do not close this window until finished.
echo.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install-Easy.ps1"
set ERR=%ERRORLEVEL%

echo.
if %ERR% NEQ 0 (
  echo Installation failed. See install.log in:
  echo   %LOCALAPPDATA%\Programs\SoundSpectrAnalyse\
) else (
  echo Done.
)
echo.
pause
exit /b %ERR%
