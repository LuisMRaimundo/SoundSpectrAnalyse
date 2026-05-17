# SoundSpectrAnalyse — macOS installer

**Build on macOS only** (PyInstaller cannot cross-compile a `.app` from Windows).

See also: [`../README.md`](../README.md).

## Prerequisites

- macOS 11+ (Intel or Apple Silicon)
- Python 3.10+ with project dependencies (`pip install -r requirements.txt` in source tree)
- Xcode Command Line Tools: `xcode-select --install`
- Tk available (usually included with python.org or Homebrew Python)

## Build

```bash
cd ~/Desktop/instalers/mac
chmod +x *.sh
export SOUNDSPECTRANALYSE_SOURCE="$HOME/Desktop/SoundSpectrAnalyse-main_6"   # if needed
./build-all.sh
```

| Output (gitignored) | Purpose |
|---------------------|---------|
| `output/app/SoundSpectrAnalyse.app` | Portable application bundle |
| `output/SoundSpectrAnalyse-macOS-3.7.0.zip` | Zip for sharing |
| `output/SoundSpectrAnalyse-macOS-3.7.0.dmg` | Disk image (drag to Applications) |

## Install (end user)

From `output/app/`:

```bash
./install-soundspectranalyse.sh
```

Or drag **SoundSpectrAnalyse.app** to **Applications**.

## Gatekeeper

Unsigned builds may be blocked. Users: **System Settings → Privacy & Security → Open Anyway**, or sign/notarize the app for distribution.

## Notes

- **FFmpeg:** `brew install ffmpeg` for some audio formats.
- First launch may be slow while macOS verifies the bundle.
