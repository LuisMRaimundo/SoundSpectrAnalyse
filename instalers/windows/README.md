# SoundSpectrAnalyse — Windows installation

**Repository:** https://github.com/LuisMRaimundo/SoundSpectrAnalyse

---

## Standard installation (no prior Python required)

### Requirements

- **Windows 10 or 11** (64-bit)
- **Internet** connection (first install)
- Python **not** required beforehand
- Administrator rights **not** required (per-user install)

### Step-by-step

1. **Obtain the installer files**
   - Clone the repository, or
   - On https://github.com/LuisMRaimundo/SoundSpectrAnalyse choose **Code → Download ZIP**, extract, and open **`instalers\windows`**.

2. **Start installation**
   - Double-click **`INSTALL.bat`** or **`START-HERE.bat`** (same installer).
   - **Do not** use an old ZIP saved on your PC before May 2026 — download fresh from GitHub (step 1).
   - **`SoundSpectrAnalyse Orchestrator.exe` is not required** for normal install (that file is only for developer portable builds).
   - If Windows SmartScreen appears, choose **More info → Run anyway** (local, unsigned script).

3. **Wait for completion**
   - A console window shows progress.
   - On the **first run**, the installer may:
     - Install **Python 3.11** (if absent);
     - Download the project from GitHub;
     - Install scientific libraries (NumPy, SciPy, librosa, …) — **10–25 minutes** is normal.
   - **Do not close** the window until you see “Done” or “SUCCESS”.

4. **Launch the application**
   - Desktop shortcut: **SoundSpectrAnalyse Orchestrator**, or
   - Start menu → **SoundSpectrAnalyse**.

5. **Use the program**
   - The Tk pipeline orchestrator GUI opens.
   - Select audio folders and parameters as described in the project documentation.

### Install location

| Item | Path |
|------|------|
| Application code | `%LocalAppData%\Programs\SoundSpectrAnalyse\app\` |
| Python environment | `%LocalAppData%\Programs\SoundSpectrAnalyse\venv\` |
| Install log | `%LocalAppData%\Programs\SoundSpectrAnalyse\install.log` |

Typical full path: `C:\Users\<YourName>\AppData\Local\Programs\SoundSpectrAnalyse\`

### Uninstall

1. Double-click **`UNINSTALL.bat`** in `instalers\windows`.
2. Removes the install folder and shortcuts.
3. Does **not** remove Python (may be used by other software).

### MP3 and other formats (optional)

Some formats require **FFmpeg** on the system PATH:

1. Download from https://ffmpeg.org/download.html (Windows build).
2. Add the FFmpeg `bin` folder to the **Path** environment variable.
3. Restart SoundSpectrAnalyse.

### Troubleshooting

| Issue | Action |
|-------|--------|
| **Missing .exe** / Build-All.ps1 message | You opened the wrong script or an **old** copy. Use **`INSTALL.bat`** from a **fresh** GitHub download. `Install-SoundSpectrAnalyse.cmd` also forwards to `INSTALL.bat` if no `.exe` is present. |
| **PowerShell: string missing terminator** | Old installer files (Unicode characters). Re-download the repo from GitHub and run **`INSTALL.bat`** from `instalers\windows`. |
| Window closes immediately | Run **`INSTALL.bat`** again and read messages; or open `install.log` (path above). |
| Python error | Install **Python 3.11** from https://www.python.org/downloads/ — enable **Add python.exe to PATH** — run **`INSTALL.bat`** again. |
| Network error | Check Internet and firewall; the installer downloads from GitHub and PyPI. |
| pip / package error | Send **`install.log`** to the software maintainer. |

### Faster install when source is already on disk

If **`SoundSpectrAnalyse-main_6`** (or a clone containing `pipeline_orchestrator_gui.py`) sits on the Desktop next to `instalers`, the installer **copies that tree** instead of downloading from GitHub.

---

## Portable executable (PyInstaller) — developers

For machines that already have Python 3.10/3.11 and project dependencies configured. **Not** recommended for end users.

```powershell
cd path\to\instalers\windows
.\Build-All.ps1
# Optional: .\Build-All.ps1 -SourceRoot "C:\path\SoundSpectrAnalyse-main_6"
```

| Output (local `output\`, not in Git) | Description |
|--------------------------------------|-------------|
| `output\app\` | Folder with `SoundSpectrAnalyse Orchestrator.exe` |
| `output\SoundSpectrAnalyse-Portable-3.7.0.zip` | Zip for distribution |
| `output\SoundSpectrAnalyse-Setup-3.7.0.exe` | Setup wizard (build requires [Inno Setup 6](https://jrsoftware.org/isinfo.php)) |

PyInstaller builds are large (~300 MB) and may trigger SmartScreen if not code-signed.
