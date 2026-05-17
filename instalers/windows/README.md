# SoundSpectrAnalyse — Windows installer

See also: [`../README.md`](../README.md).

## Build

```powershell
cd "C:\Users\lmr20\Desktop\instalers\windows"
.\Build-All.ps1
# optional: .\Build-All.ps1 -SourceRoot "C:\Users\lmr20\Desktop\SoundSpectrAnalyse-main_6"
```

| Output (gitignored) | Purpose |
|---------------------|---------|
| `output\app\` | Portable — `SoundSpectrAnalyse Orchestrator.exe` |
| `output\SoundSpectrAnalyse-Portable-3.7.0.zip` | Zip for sharing |
| `output\SoundSpectrAnalyse-Setup-3.7.0.exe` | Wizard (requires [Inno Setup 6](https://jrsoftware.org/isinfo.php)) |

End users can run `Install-SoundSpectrAnalyse.cmd` from `output\app\` for a Start menu shortcut.
