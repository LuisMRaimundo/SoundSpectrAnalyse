# -*- mode: python ; coding: utf-8 -*-
# macOS PyInstaller spec — SOURCE_ROOT and LAUNCHER_SCRIPT set in patched header.

import sys
from pathlib import Path

block_cipher = None

_source = Path(SOURCE_ROOT)  # noqa: F821
_launcher = Path(LAUNCHER_SCRIPT)  # noqa: F821

a = Analysis(
    [str(_launcher)],
    pathex=[str(_source)],
    binaries=[],
    datas=[
        (str(_source / "metrics_dictionary.json"), "."),
        (str(_source / "audio_analysis" / "batch_config.json"), "audio_analysis"),
    ],
    hiddenimports=[
        "sklearn.utils._typedefs",
        "sklearn.neighbors._partition_nodes",
        "sklearn.utils._weight_vector",
        "sklearn.tree._utils",
        "numba.core.types",
        "librosa",
        "librosa.core",
        "librosa.feature",
        "librosa.util",
        "soundfile",
        "pydub",
        "openpyxl",
        "xlsxwriter",
        "matplotlib.backends.backend_tkagg",
        "pipeline_orchestrator_gui",
        "pipeline_orchestrator_integrated",
        "run_orchestrator",
        "proc_audio",
        "compile_metrics",
        "audio_utils",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython", "jupyter", "notebook", "tkinter.test"],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="SoundSpectrAnalyse-Orchestrator",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=True,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="SoundSpectrAnalyse",
)

app = BUNDLE(
    coll,
    name="SoundSpectrAnalyse.app",
    icon=None,
    bundle_identifier="pt.luisraimundo.soundspectranalyse",
)
