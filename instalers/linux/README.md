# SoundSpectrAnalyse — instalação no Linux

**Repositório:** https://github.com/LuisMRaimundo/SoundSpectrAnalyse

**Importante:** execute estes scripts **num computador Linux** (não no Windows ou macOS).

---

## Instalação para utilizadores (sem Python) — recomendado

### Requisitos

- Distribuição com **glibc** (ex.: **Ubuntu 22.04+**, Debian 12, Fedora 38+, Linux Mint)
- Arquitetura **x86_64** (64 bits); noutras arquiteturas pode ser necessário instalar a partir do código-fonte
- Ligação à **Internet**
- Ambiente gráfico (X11 ou Wayland) para a janela Tk

### Passo a passo (Ubuntu / Debian)

1. **Pacotes do sistema** (só uma vez; pede palavra-passe de administrador):
   ```bash
   sudo apt update
   sudo apt install -y python3 python3-venv python3-pip curl unzip
   sudo apt install -y python3-tk
   ```
   O pacote **`python3-tk`** é necessário para a interface gráfica.

2. **Obter os ficheiros do instalador**
   - Clone o repositório ou extraia o ZIP do GitHub e abra a pasta **`instalers/linux`**.

3. **Terminal na pasta do instalador**
   ```bash
   cd ~/Desktop/instalers/linux
   ```
   (Ajuste o caminho conforme a sua pasta.)

4. **Permitir execução e instalar**
   ```bash
   chmod +x install-easy.sh
   ./install-easy.sh
   ```
   - Descarrega o código de https://github.com/LuisMRaimundo/SoundSpectrAnalyse
   - Cria ambiente virtual e instala bibliotecas — **10 a 25 minutos** na primeira vez.

5. **Abrir o programa**
   - No menu de aplicações: procure **SoundSpectrAnalyse Orchestrator**, ou
   - No terminal:
     ```bash
     soundspectranalyse-gui
     ```

### Passo a passo (Fedora)

```bash
sudo dnf install -y python3 python3-pip python3-tkinter curl unzip
cd ~/Desktop/instalers/linux
chmod +x install-easy.sh
./install-easy.sh
```

### Onde fica instalado

| Conteúdo | Localização |
|----------|-------------|
| Aplicação e código | `~/.local/share/SoundSpectrAnalyse/app/` |
| Python e bibliotecas | `~/.local/share/SoundSpectrAnalyse/venv/` |
| Comando no terminal | `~/.local/bin/soundspectranalyse-gui` |
| Entrada no menu | `~/.local/share/applications/soundspectranalyse-orchestrator.desktop` |

Certifique-se de que `~/.local/bin` está no **PATH** (muitas distribuições já incluem por defeito).

### Desinstalar

```bash
rm -rf ~/.local/share/SoundSpectrAnalyse
rm -f ~/.local/bin/soundspectranalyse-gui
rm -f ~/.local/share/applications/soundspectranalyse-orchestrator.desktop
```

### Áudio e FFmpeg (opcional)

```bash
# Ubuntu / Debian
sudo apt install -y ffmpeg

# Fedora
sudo dnf install -y ffmpeg
```

### Se a instalação falhar

| Problema | O que fazer |
|----------|-------------|
| `No module named '_tkinter'` | Instale `python3-tk` (Debian/Ubuntu) ou `python3-tkinter` (Fedora) e volte a correr `./install-easy.sh`. |
| Python &lt; 3.10 | Atualize para Python 3.10 ou 3.11 (`sudo apt install python3.11` ou equivalente). |
| Erro pip / compilação | Instale ferramentas de compilação: `sudo apt install build-essential python3-dev`. |
| Janela não abre em SSH | Use sessão gráfica local ou X11 forwarding; o programa precisa de ecrã. |
| Erro de rede | Verifique proxy e firewall; o script descarrega GitHub e PyPI. |

---

## Instalação avançada — binário portátil (PyInstaller)

Para desenvolvedores com o projeto já configurado em Python.

```bash
cd ~/Desktop/instalers/linux
chmod +x *.sh
./build-all.sh
```

| Saída (pasta `output/`, não vai para o Git) | Descrição |
|--------------------------------------------|-----------|
| `output/app/` | Pasta com executável `SoundSpectrAnalyse-Orchestrator` |
| `output/SoundSpectrAnalyse-Linux-x86_64-3.7.0.tar.gz` | Arquivo para partilhar |

Instalação a partir da build portátil (em `output/app/`):

```bash
./install-soundspectranalyse.sh
```

Cria os mesmos atalhos em `~/.local` que a instalação fácil.
