# SoundSpectrAnalyse — instaladores

**Repositório:** https://github.com/LuisMRaimundo/SoundSpectrAnalyse

Esta pasta contém scripts para instalar o **SoundSpectrAnalyse** em utilizadores **sem conhecimentos de Python**. Escolha a pasta do seu sistema operativo:

| Pasta | Sistema | Instalação recomendada |
|-------|---------|------------------------|
| **[`windows/`](windows/)** | Windows 10/11 (64 bits) | Duplo clique em **`INSTALL.bat`** |
| **[`mac/`](mac/)** | macOS 11 ou superior | Script **`install-easy.sh`** |
| **[`linux/`](linux/)** | Linux (Ubuntu, Debian, Fedora, …) | Script **`install-easy.sh`** |

Cada subpasta tem um **README com instruções detalhadas** para esse sistema.

## O que a instalação “fácil” faz

1. Instala ou deteta **Python 3.10 ou 3.11** (no Windows, instala automaticamente se faltar).
2. Obtém o código a partir de **https://github.com/LuisMRaimundo/SoundSpectrAnalyse** (ramo `main`).
3. Cria um ambiente isolado e instala as bibliotecas (`requirements.txt`).
4. Cria um **atalho** para abrir a interface gráfica (**SoundSpectrAnalyse Orchestrator**).

A primeira instalação pode demorar **10–25 minutos** (download + pacotes científicos). É necessária ligação à Internet.

## O que não está incluído no Git

Pastas `build/`, `dist/`, `output/` e ficheiros `.exe` / `.zip` / `.dmg` compilados **não** vão para o repositório. Para distribuir binários prontos, use [GitHub Releases](https://github.com/LuisMRaimundo/SoundSpectrAnalyse/releases).

## Instalação avançada (desenvolvedores)

Em cada pasta existem também scripts **PyInstaller** (`Build-All.ps1`, `build-all.sh`) para gerar aplicações “portáteis” sem instalar Python no destino. São mais complexos e **não** são recomendados para utilizadores finais.
