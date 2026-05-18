# SoundSpectrAnalyse — installers

**GitHub:** https://github.com/LuisMRaimundo/SoundSpectrAnalyse

Scripts for **non-Python users** (easy install) and **developers** (PyInstaller portable builds).

| Folder | Easy install (recommended) | Developer build |
|--------|---------------------------|-----------------|
| [`windows/`](windows/) | **`INSTALL.bat`** — Python + pip + shortcuts | `Build-All.ps1` |
| [`mac/`](mac/) | **`install-easy.sh`** (on macOS) | `build-all.sh` |
| [`linux/`](linux/) | **`install-easy.sh`** (on Linux) | `build-all.sh` |

## Windows (most users)

1. Open folder `instalers\windows`
2. Double-click **`INSTALL.bat`**
3. Wait for completion; use the Desktop shortcut

No command line or Python knowledge required.

## macOS / Linux

```bash
chmod +x install-easy.sh
./install-easy.sh
```

Must be run **on** macOS or Linux respectively.

## What gets installed (easy mode)

- **Python 3.10 or 3.11** (installed automatically on Windows if missing)
- Project from **GitHub** `main` (or a local copy if found next to `instalers/`)
- All **`requirements.txt`** libraries in a private virtual environment
- **Desktop / Start menu** shortcut to the Tk orchestrator GUI

Install location (Windows): `%LocalAppData%\Programs\SoundSpectrAnalyse\`

Built `.exe` / `.app` bundles are optional and live under `output/` (not in git).
