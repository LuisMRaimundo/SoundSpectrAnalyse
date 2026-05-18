# SoundSpectrAnalyse — Windows easy-install constants
$script:SoundSpectrConfig = @{
    GitHubRepoUrl      = 'https://github.com/LuisMRaimundo/SoundSpectrAnalyse'
    GitHubZipUrl       = 'https://github.com/LuisMRaimundo/SoundSpectrAnalyse/archive/refs/heads/main.zip'
    GitHubBranch       = 'main'
    AppName            = 'SoundSpectrAnalyse'
    PythonVersion      = '3.11'
    PythonMinMinor     = 10
    PythonMaxMinor     = 11
    PythonInstallerUrl = 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe'
    InstallRoot        = Join-Path $env:LOCALAPPDATA 'Programs\SoundSpectrAnalyse'
    GuiScript          = 'pipeline_orchestrator_gui.py'
}
