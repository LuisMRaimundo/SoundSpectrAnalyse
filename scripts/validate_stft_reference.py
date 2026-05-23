#!/usr/bin/env python3
"""
Release-style gate: STFT reference tests (Parseval-style energy + bin-aligned partial).

Exit code 0 if all checks pass; non-zero otherwise. Intended for CI or pre-release:

  python scripts/validate_stft_reference.py
"""

from __future__ import annotations

import subprocess
import sys
import platform
from pathlib import Path

import numpy as np
import librosa
import scipy
from scipy.signal import get_window

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _first_mismatch(abs_diff: np.ndarray, tol: float) -> str:
    idx = np.argwhere(abs_diff > tol)
    if idx.size == 0:
        return "(none)"
    i, j = idx[0].tolist()
    return f"(bin={i}, frame={j})"


def _print_diagnostics() -> None:
    tests_file = ROOT / "tests" / "test_stft_reference_goldens.py"
    constants_file = ROOT / "constants.py"
    proc_audio_file = ROOT / "proc_audio.py"
    print("[STFT-GATE] python_version:", sys.version.replace("\n", " "))
    print("[STFT-GATE] platform:", platform.platform())
    print("[STFT-GATE] cwd:", Path.cwd().resolve())
    print("[STFT-GATE] root:", ROOT)
    print("[STFT-GATE] tests_path:", tests_file)
    print("[STFT-GATE] constants_path:", constants_file)
    print("[STFT-GATE] proc_audio_path:", proc_audio_file)
    print("[STFT-GATE] exists tests_path:", tests_file.is_file())
    print("[STFT-GATE] exists constants_path:", constants_file.is_file())
    print("[STFT-GATE] exists proc_audio_path:", proc_audio_file.is_file())
    print("[STFT-GATE] numpy:", np.__version__)
    print("[STFT-GATE] librosa:", librosa.__version__)
    print("[STFT-GATE] scipy:", scipy.__version__)

    from constants import ENERGY_CONSERVATION_TOLERANCE_STRICT
    from proc_audio import AudioProcessor, _normalize_level, _verify_energy_conservation

    sr = 44100
    n_fft = 1024
    hop = 1024
    t = np.linspace(0.0, 1.0, int(sr), endpoint=False)
    y = np.sin(2 * np.pi * 440.0 * t)
    y_norm = _normalize_level(y, target_rms_db=-20.0)
    w = get_window("hann", n_fft, fftbins=True)
    s_parseval = librosa.stft(
        y_norm,
        n_fft=n_fft,
        win_length=len(w),
        hop_length=hop,
        window=w,
        center=True,
    )
    r = _verify_energy_conservation(
        y_norm,
        s_parseval,
        n_fft,
        hop,
        "hann",
        tolerance=ENERGY_CONSERVATION_TOLERANCE_STRICT,
        window_array=w,
    )

    sr2 = 48000
    n = int(sr2 * 0.5)
    t2 = np.arange(n, dtype=float) / sr2
    y2 = 0.08 * np.sin(2 * np.pi * 512.0 * t2)
    ap = AudioProcessor()
    ap.sr = sr2
    ap.y = y2.astype(np.float64)
    ap.n_fft = n_fft
    ap.hop_length = hop
    ap.window = "hann"
    win_arg = ap._get_window_arg()
    s_code = librosa.stft(
        y2,
        n_fft=n_fft,
        win_length=len(win_arg),
        hop_length=hop,
        window=win_arg,
        center=True,
    )
    s_ref = librosa.stft(
        y2,
        n_fft=n_fft,
        win_length=n_fft,
        hop_length=hop,
        window="hann",
        center=True,
    )
    abs_diff = np.abs(s_code - s_ref)
    ref_abs = np.abs(s_ref)
    rel_diff = abs_diff / np.maximum(ref_abs, 1e-20)
    max_abs = float(np.max(abs_diff))
    max_rel = float(np.max(rel_diff))
    idx = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    db_tol = 0.01
    db_diff = 20.0 * np.log10(np.maximum(np.abs(s_code), 1e-20)) - 20.0 * np.log10(
        np.maximum(np.abs(s_ref), 1e-20)
    )
    max_abs_db = float(np.nanmax(np.abs(db_diff)))

    print("[STFT-GATE] parseval S.shape:", s_parseval.shape)
    print("[STFT-GATE] parseval S.dtype:", s_parseval.dtype)
    print("[STFT-GATE] parseval |S| min/max/mean:", float(np.min(np.abs(s_parseval))), float(np.max(np.abs(s_parseval))), float(np.mean(np.abs(s_parseval))))
    print("[STFT-GATE] parseval energy_ratio:", float(r["energy_ratio"]))
    print("[STFT-GATE] parseval deviation:", float(r["deviation"]))
    print("[STFT-GATE] parseval tolerance:", float(r["tolerance"]))
    print("[STFT-GATE] compare S_code.shape:", s_code.shape)
    print("[STFT-GATE] compare S_ref.shape:", s_ref.shape)
    print("[STFT-GATE] compare S_code.dtype:", s_code.dtype)
    print("[STFT-GATE] compare S_ref.dtype:", s_ref.dtype)
    print("[STFT-GATE] compare |S_code| min/max/mean:", float(np.min(np.abs(s_code))), float(np.max(np.abs(s_code))), float(np.mean(np.abs(s_code))))
    print("[STFT-GATE] compare |S_ref|  min/max/mean:", float(np.min(np.abs(s_ref))), float(np.max(np.abs(s_ref))), float(np.mean(np.abs(s_ref))))
    print("[STFT-GATE] max_abs_diff:", max_abs)
    print("[STFT-GATE] max_rel_diff:", max_rel)
    print("[STFT-GATE] max_abs_diff_db:", max_abs_db)
    print("[STFT-GATE] tolerance_db:", db_tol)
    print("[STFT-GATE] first_mismatch_by_abs_tol:", _first_mismatch(abs_diff, 1e-12))
    print(
        "[STFT-GATE] argmax_abs_diff_index/value:",
        idx,
        complex(s_code[idx]),
        complex(s_ref[idx]),
    )


def main() -> int:
    tests = ROOT / "tests" / "test_stft_reference_goldens.py"
    _print_diagnostics()
    if not tests.is_file():
        print("Missing", tests, file=sys.stderr)
        return 2
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(tests),
        "-q",
        "--tb=short",
    ]
    return int(subprocess.call(cmd, cwd=str(ROOT)))


if __name__ == "__main__":
    raise SystemExit(main())
