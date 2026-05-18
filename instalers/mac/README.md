# SoundSpectrAnalyse — instalação no macOS

**Repositório:** https://github.com/LuisMRaimundo/SoundSpectrAnalyse

**Importante:** os scripts desta pasta têm de ser executados **num Mac** (não é possível instalar ou compilar o `.app` a partir do Windows).

---

## Instalação para utilizadores (sem Python) — recomendado

### Requisitos

- **macOS 11 (Big Sur)** ou superior
- **Intel** ou **Apple Silicon (M1/M2/M3)**
- Ligação à **Internet** (primeira instalação)
- Conta de utilizador normal (não exige ser administrador para a instalação em `~/Applications`)

### Passo a passo

1. **Obter os ficheiros**
   - No Mac, clone o repositório ou descarregue o ZIP do GitHub e abra **`instalers/mac`**.

2. **Abrir a Terminal**
   - **Finder → Aplicações → Utilitários → Terminal** (ou pesquise “Terminal”).

3. **Ir à pasta do instalador**
   ```bash
   cd ~/Desktop/instalers/mac
   ```
   (Ajuste o caminho se colocou a pasta noutro sítio.)

4. **Permitir execução do script**
   ```bash
   chmod +x install-easy.sh
   ```

5. **Executar a instalação**
   ```bash
   ./install-easy.sh
   ```
   - Se não tiver Python 3.10/3.11, o script tenta instalar via **Homebrew** (`brew install python@3.11`). Se não tiver Homebrew, instale Python em https://www.python.org/downloads/ e repita o passo 5.
   - O script descarrega o projeto do GitHub e instala as bibliotecas — **10 a 25 minutos** na primeira vez.

6. **Abrir o programa**
   - **Finder → Aplicações → SoundSpectrAnalyse** → duplo clique em **`Launch-SoundSpectrAnalyse.command`**, ou
   - Atalho no **Ambiente de trabalho**: **`SoundSpectrAnalyse Orchestrator.command`**.

7. **Primeira abertura e segurança (Gatekeeper)**
   - Se o macOS disser que a app não pode ser aberta porque é de um programador não identificado:
     - **Definições do Sistema → Privacidade e segurança → Abrir mesmo assim**, ou
     - Clique com o botão direito no `.command` → **Abrir** → **Abrir**.

### Onde fica instalado

| Conteúdo | Localização |
|----------|-------------|
| Aplicação e código | `~/Applications/SoundSpectrAnalyse/app/` |
| Python e bibliotecas | `~/Applications/SoundSpectrAnalyse/venv/` |
| Script de arranque | `~/Applications/SoundSpectrAnalyse/Launch-SoundSpectrAnalyse.command` |

### Desinstalar

```bash
rm -rf ~/Applications/SoundSpectrAnalyse
rm -f ~/Desktop/SoundSpectrAnalyse\ Orchestrator.command
```

(O Python instalado via Homebrew ou python.org **não** é removido.)

### Áudio e FFmpeg (opcional)

Para alguns formatos (ex.: MP3):

```bash
brew install ffmpeg
```

(Requer [Homebrew](https://brew.sh) instalado.)

### Se a instalação falhar

| Problema | O que fazer |
|----------|-------------|
| `command not found: python3` | Instale Python 3.11 de https://www.python.org/downloads/ ou `brew install python@3.11`. |
| Erro `tkinter` | Reinstale Python com o instalador oficial (inclui Tcl/Tk). |
| Erro de rede / pip | Verifique Wi‑Fi e proxy; volte a correr `./install-easy.sh`. |
| Homebrew em falta | Instale em https://brew.sh ou use Python.org em vez de depender do brew. |

---

## Instalação avançada — `.app` portátil (PyInstaller)

Para desenvolvedores que já têm o ambiente Python do projeto configurado.

```bash
cd ~/Desktop/instalers/mac
chmod +x *.sh
./build-all.sh
```

| Saída (pasta `output/`, não vai para o Git) | Descrição |
|--------------------------------------------|-----------|
| `output/app/SoundSpectrAnalyse.app` | Pacote de aplicação |
| `output/SoundSpectrAnalyse-macOS-3.7.0.zip` | Zip para partilhar |
| `output/SoundSpectrAnalyse-macOS-3.7.0.dmg` | Imagem de disco (arrastar para Aplicações) |

Depois de `./build-pyinstaller.sh`, pode correr em `output/app/`:

```bash
./install-soundspectranalyse.sh
```

(ou arrastar `SoundSpectrAnalyse.app` para **Aplicações**).

Aplicações não assinadas podem exigir passos extra em **Privacidade e segurança** (ver acima).
