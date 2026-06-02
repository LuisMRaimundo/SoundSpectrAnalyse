# SoundSpectrAnalyse - Windows installer

**Repository:** https://github.com/LuisMRaimundo/SoundSpectrAnalyse

## Standard installation (no Python required)

1. Download the repository ZIP (**Code -> Download ZIP**) or clone it.
2. Open `installers\windows`.
3. Double-click `INSTALL.bat` or `START-HERE.bat` (same as `Install and Run.bat`).
4. Wait for first-time setup (**10-25 minutes** can be normal on slower networks).
5. The SoundSpectrAnalyse GUI opens when setup finishes.

After analysis with auto-compile enabled, each folder also contains `compiled_density_metrics.xlsx` and `compiled_density_metrics_research.xlsx` (Stage 3 EWSD scores in `Spectral_Density_Metrics`).

## Install log

`installers\runtime\windows\install.log`

## Troubleshooting

| Issue | Action |
|-------|--------|
| No window / closes instantly | Run `INSTALL.bat` from a fresh GitHub download. |
| Setup failed | Open `install.log`, check internet/firewall, delete `installers\runtime\` and retry. |
| PowerShell parse error | Re-download from GitHub; old copies may include broken characters. |

## Advanced launch (CLI)

After first installation:

`installers\runtime\windows\python\python.exe installers\common\bootstrap.py launch --cli -- --audio-dir "C:\path\to\wav"`
