# SoundSpectrAnalyse — Windows installer (build scripts)

Scripts to build a **standalone Windows app** (no Python on the end user's PC) from the main project source.

**In Git:** commit only this folder (not `build/`, `dist/`, or `output/`).  
**For users:** publish built `.zip` / `.exe` via [GitHub Releases](https://github.com/LuisMRaimundo/SoundSpectrAnalyse/releases).

Suggested path in the repository: `installer/windows/`.

## Build (developer machine)

Requirements: Windows 10/11, Python 3.10+ with project dependencies, project source (clone or `SoundSpectrAnalyse-main_6`).

```powershell
cd "C:\path\to\installer\windows"
.\Build-All.ps1
```

Optional: `-SourceRoot "C:\path\to\repo"` if auto-detection fails.

| Output (local only, gitignored) | Purpose |
|--------------------------------|---------|
| `output\app\` | Portable folder — `SoundSpectrAnalyse Orchestrator.exe` |
| `output\SoundSpectrAnalyse-Portable-3.7.0.zip` | Zip for sharing |
| `output\SoundSpectrAnalyse-Setup-3.7.0.exe` | Setup wizard (needs [Inno Setup 6](https://jrsoftware.org/isinfo.php)) |

After `Build-PyInstaller.ps1`, `Install-SoundSpectrAnalyse.cmd` is copied into `output\app\` for a simple install without Inno.

## Source detection

`Resolve-SourceRoot.ps1` looks for `pipeline_orchestrator_gui.py` in:

1. Repository root (when this folder lives at `installer/windows/`)
2. `SoundSpectrAnalyse-main_6` next to the parent folder (desktop dev layout)

Or set: `$env:SOUNDSPECTRANALYSE_SOURCE = "C:\path\to\source"`

## Notes

- Expect **~300 MB–1 GB** per portable build.
- Some formats need **FFmpeg** on PATH.
- Unsigned builds may trigger Windows SmartScreen.
