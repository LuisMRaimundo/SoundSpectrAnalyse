# SoundSpectrAnalyse — Linux installer

**Build on Linux only** (x86_64 recommended; same workflow on ARM with native Python stack).

See also: [`../README.md`](../README.md).

## Prerequisites

- Linux with glibc (Ubuntu 22.04+, Fedora, Debian, etc.)
- Python 3.10+ and venv with project dependencies
- Tk: `sudo apt install python3-tk` (Debian/Ubuntu) or distro equivalent

## Build

```bash
cd ~/Desktop/instalers/linux
chmod +x *.sh
export SOUNDSPECTRANALYSE_SOURCE="$HOME/Desktop/SoundSpectrAnalyse-main_6"   # if needed
./build-all.sh
```

| Output (gitignored) | Purpose |
|---------------------|---------|
| `output/app/` | Folder with `SoundSpectrAnalyse-Orchestrator` |
| `output/SoundSpectrAnalyse-Linux-x86_64-3.7.0.tar.gz` | Tarball for sharing |

## Install (end user)

From `output/app/`:

```bash
./install-soundspectranalyse.sh
```

Creates `~/.local/share/SoundSpectrAnalyse`, a `.desktop` launcher, and `~/.local/bin/soundspectranalyse-orchestrator`.

## Notes

- **FFmpeg:** install via package manager for MP3 and some codecs.
- If the GUI fails to start, confirm `python3-tk` / `tk` is installed on the **build** machine (libraries are bundled, but display server must be available at runtime).
