# Autonomous installers (no Python required)

These launchers install a **private copy** of Python and all SoundSpectrAnalyse libraries on
**first run**, then open the desktop GUI. Users do **not** need to install Python, pip, or conda.

**Requirements:** Internet on first run (~250-600 MB download depending on platform and wheel cache),
disk space ~1-2 GB after install, and one of:

- Windows 10/11 (x86_64)
- macOS 11+ (Intel or Apple Silicon)
- Recent Linux (x86_64 or arm64)

---

## Windows 10 / 11

1. Open the project folder (or unpacked ZIP).
2. Double-click:

   **`installers\windows\Install and Run.bat`**

3. Wait for first-time setup.
4. The SoundSpectrAnalyse GUI opens. Keep the console window open while you use the app.

To stop: close the GUI window, then close the console or press **Ctrl+C**.

---

## macOS

1. In Terminal, make launchers executable (once):

   ```bash
   chmod +x "installers/macos/Install and Run.command"
   chmod +x installers/macos/setup-runtime.sh
   ```

2. Double-click **`installers/macos/Install and Run.command`**.
3. If blocked, allow it in **System Settings -> Privacy & Security -> Open Anyway**.

---

## Linux

```bash
chmod +x installers/linux/install-and-run.sh installers/linux/setup-runtime.sh
./installers/linux/install-and-run.sh
```

---

## What gets installed?

| Location | Contents |
|----------|----------|
| `installers/runtime/` | Private Python + pip packages |
| Runtime session | SoundSpectrAnalyse Tk GUI (`pipeline_orchestrator_gui.py`) |

Each completed analysis folder also produces `compiled_density_metrics.xlsx` and, when Stage 2 auto-compile is enabled, `compiled_density_metrics_research.xlsx` with EWSD-R v18 scores in `Spectral_Density_Metrics` (Stage 3).

To reinstall: delete `installers/runtime/` and run the launcher again.

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| Setup failed | Check internet/firewall; retry after deleting `installers/runtime/` |
| GUI does not open | Run `python installers/common/bootstrap.py doctor` and verify paths |
| Dependency install fails | Re-run installer and inspect `installers/runtime/windows/install.log` on Windows |

Diagnostics (any Python 3.10+):

```bash
python installers/common/bootstrap.py doctor
```
