# SoundSpectrAnalyse — Windows

**Repository:** https://github.com/LuisMRaimundo/SoundSpectrAnalyse

## For end users (no Python knowledge) — recommended

1. Download or copy the folder **`instalers/windows`** from GitHub.
2. **Double-click `INSTALL.bat`**
3. Wait until finished (first time: downloads Python if needed, clones the project, installs libraries — **10–25 minutes**).
4. Open **SoundSpectrAnalyse Orchestrator** from the Desktop or Start menu.

Installed to:

`%LocalAppData%\Programs\SoundSpectrAnalyse\`

| Item | Path |
|------|------|
| Application code | `...\app\` |
| Python environment | `...\venv\` |
| Log | `...\install.log` |

**Uninstall:** double-click **`UNINSTALL.bat`** (removes the folder and shortcuts; does not remove Python).

### Optional: FFmpeg

Some formats (e.g. MP3) need **FFmpeg** on PATH: https://ffmpeg.org/download.html

### If install fails

- Run **`INSTALL.bat`** again as a normal user (not required to be Administrator).
- Install **Python 3.11** manually from https://www.python.org/downloads/ — enable **“Add python.exe to PATH”** — then run **`INSTALL.bat`** again.
- Read `%LocalAppData%\Programs\SoundSpectrAnalyse\install.log`

### Advanced: use a local copy instead of GitHub

If you already have the project on disk (e.g. `SoundSpectrAnalyse-main_6` on the Desktop next to `instalers`), the installer **copies that folder** instead of downloading — faster for developers.

---

## For developers: portable `.exe` (PyInstaller)

Requires a working Python environment. Not recommended for non-technical users.

```powershell
.\Build-All.ps1
```

See output under `output\` (gitignored).
