# SoundSpectrAnalyse — cross-platform installers (build scripts)

Scripts to build **standalone apps** (no Python on the end user's machine) from the main project source.

| Folder | Platform | Build on |
|--------|----------|----------|
| **[`windows/`](windows/)** | Windows 10/11 | Windows |
| **[`mac/`](mac/)** | macOS 11+ (Apple Silicon or Intel) | **macOS only** |
| **[`linux/`](linux/)** | Linux x86_64 | **Linux only** |

**Git:** commit only script files in each folder (not `build/`, `dist/`, `output/`).  
**Releases:** publish built `.zip` / `.dmg` / `.tar.gz` via [GitHub Releases](https://github.com/LuisMRaimundo/SoundSpectrAnalyse/releases).

## Source code

Set the project tree explicitly if auto-detection fails:

```bash
export SOUNDSPECTRANALYSE_SOURCE="/path/to/SoundSpectrAnalyse"
```

Default search order: repository root (when `instalers/` lives inside the clone), then `SoundSpectrAnalyse-main_6` on the Desktop.

## Quick start

```powershell
# Windows (PowerShell)
cd windows
.\Build-All.ps1
```

```bash
# macOS
cd mac
chmod +x *.sh
./build-all.sh

# Linux
cd linux
chmod +x *.sh
./build-all.sh
```

## Notes

- Expect **~300 MB–1 GB** per build (NumPy, SciPy, librosa, etc.).
- **FFmpeg** on PATH may be required for some audio formats.
- macOS/Linux builds **cannot** be produced from Windows; use a Mac or Linux machine (or CI).
