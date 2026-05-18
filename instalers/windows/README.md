# SoundSpectrAnalyse — instalação no Windows

**Repositório:** https://github.com/LuisMRaimundo/SoundSpectrAnalyse

---

## Instalação para utilizadores (sem Python) — recomendado

### Requisitos

- **Windows 10 ou 11** (64 bits)
- Ligação à **Internet** (primeira instalação)
- **Não** é obrigatório ter Python instalado antes
- **Não** é obrigatório ser administrador (instalação na pasta do utilizador)

### Passo a passo

1. **Obter os ficheiros**
   - Clone o repositório, ou
   - Em https://github.com/LuisMRaimundo/SoundSpectrAnalyse clique **Code → Download ZIP**, extraia, e abra a pasta **`instalers\windows`**.

2. **Iniciar a instalação**
   - Faça **duplo clique** em **`INSTALL.bat`**.
   - Se o Windows SmartScreen avisar, escolha **Mais informações → Executar mesmo assim** (o script é local e não assinado).

3. **Aguardar**
   - A janela preta (PowerShell) mostra o progresso.
   - Na **primeira vez**, o instalador pode:
     - Instalar **Python 3.11** (se não existir no PC);
     - Descarregar o projeto do GitHub;
     - Instalar dezenas de bibliotecas (NumPy, SciPy, librosa, …) — **10 a 25 minutos** é normal.
   - **Não feche** a janela até aparecer “Done” ou “SUCCESS”.

4. **Abrir o programa**
   - Use o atalho no **Ambiente de trabalho**: **SoundSpectrAnalyse Orchestrator**, ou
   - Menu Iniciar → pasta **SoundSpectrAnalyse**.

5. **Utilizar**
   - Abre-se a interface gráfica (Tk) do pipeline de análise espectral.
   - Escolha pastas de áudio e parâmetros conforme o manual do projeto.

### Onde fica instalado

| Conteúdo | Localização |
|----------|-------------|
| Programa e código | `%LocalAppData%\Programs\SoundSpectrAnalyse\app\` |
| Python e bibliotecas | `%LocalAppData%\Programs\SoundSpectrAnalyse\venv\` |
| Registo da instalação | `%LocalAppData%\Programs\SoundSpectrAnalyse\install.log` |

(Caminho típico: `C:\Users\O_SEU_NOME\AppData\Local\Programs\SoundSpectrAnalyse\`)

### Desinstalar

1. Duplo clique em **`UNINSTALL.bat`** (nesta mesma pasta `instalers\windows`).
2. Remove a pasta de instalação e os atalhos.
3. **Não** remove o Python (pode ser usado por outros programas).

### Áudio MP3 e outros formatos (opcional)

Alguns formatos precisam do **FFmpeg** no PATH do Windows:

1. Descarregue em https://ffmpeg.org/download.html (versão Windows).
2. Adicione a pasta `bin` do FFmpeg às variáveis de ambiente **Path**.
3. Reinicie o SoundSpectrAnalyse.

### Se a instalação falhar

| Problema | O que fazer |
|----------|-------------|
| Janela fecha depressa | Abra **`INSTALL.bat`** outra vez e leia as mensagens; ou veja `install.log` (caminho acima). |
| Erro de Python | Instale manualmente **Python 3.11** em https://www.python.org/downloads/ — marque **“Add python.exe to PATH”** — volte a executar **`INSTALL.bat`**. |
| Erro de rede | Verifique Internet e firewall; o instalador descarrega o GitHub e pacotes pip. |
| Erro de pip / pacote | Envie o ficheiro **`install.log`** a quem mantém o software. |

### Instalação mais rápida (já tem o código no PC)

Se na secretária (ou ao lado de `instalers`) existir a pasta **`SoundSpectrAnalyse-main_6`** (ou clone do repositório com `pipeline_orchestrator_gui.py`), o instalador **copia essa pasta** em vez de descarregar do GitHub.

---

## Instalação avançada — executável portátil (PyInstaller)

**Apenas para quem já tem Python 3.10/3.11** e dependências instaladas no código-fonte. **Não** recomendado para utilizadores finais.

```powershell
cd caminho\para\instalers\windows
.\Build-All.ps1
# Opcional: .\Build-All.ps1 -SourceRoot "C:\caminho\SoundSpectrAnalyse-main_6"
```

| Saída (pasta local `output\`, não vai para o Git) | Descrição |
|--------------------------------------------------|-----------|
| `output\app\` | Pasta com `SoundSpectrAnalyse Orchestrator.exe` |
| `output\SoundSpectrAnalyse-Portable-3.7.0.zip` | Zip para partilhar |
| `output\SoundSpectrAnalyse-Setup-3.7.0.exe` | Instalador gráfico (requer [Inno Setup 6](https://jrsoftware.org/isinfo.php) para compilar) |

O `.exe` PyInstaller é grande (~300 MB) e pode ser bloqueado pelo SmartScreen se não estiver assinado digitalmente.
