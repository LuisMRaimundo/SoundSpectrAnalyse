import math
from dataclasses import dataclass
from typing import Optional, List

import tkinter as tk
from tkinter import ttk, messagebox
import argparse


# ---------------------------------------------------------------------------
# Utilitário: próxima potência de 2
# ---------------------------------------------------------------------------

def get_next_power_of_two(n: int) -> int:
    """Return the smallest power of two >= n."""
    if n <= 0:
        return 1
    return 1 << int(math.ceil(math.log2(n)))


# ---------------------------------------------------------------------------
# Dataclass: parâmetros STFT por banda
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BandSTFTParams:
    band_name: str
    f_low_hz: float
    f_high_hz: float
    window_type: str
    window_length: int          # M (samples)
    window_duration_s: float    # M / fs
    hop_size: int               # H (samples)
    overlap_percent: float      # 100 * (1 - H/M)
    n_fft: int
    delta_f_window_hz: float    # fs / M (resolução efetiva da janela)
    delta_f_bins_hz: float      # fs / N_fft (espaçamento entre bins)
    periods_at_f_low: float     # nº de períodos a f_low dentro de M
    zp_factor_recommended: float  # N_target / M
    zp_factor_actual: float       # N_fft / M
    warning: Optional[str] = None


# ---------------------------------------------------------------------------
# Definição de bandas: escala logarítmica entre f_min e f_max
# ---------------------------------------------------------------------------

def define_log_frequency_bands(f_min_hz: float,
                               f_max_hz: float,
                               n_bands: int) -> List[tuple]:
    """
    Divide [f_min_hz, f_max_hz] em n_bands bandas logarítmicas.
    Cada banda é [f_low, f_high] em Hz.
    """
    if f_min_hz <= 0 or f_max_hz <= f_min_hz:
        raise ValueError("f_min_hz must be > 0 and < f_max_hz.")
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1.")

    log_min = math.log10(f_min_hz)
    log_max = math.log10(f_max_hz)
    step = (log_max - log_min) / n_bands

    boundaries = [10 ** (log_min + i * step) for i in range(n_bands + 1)]
    bands = []
    for i in range(n_bands):
        f_low = boundaries[i]
        f_high = boundaries[i + 1]
        bands.append((f_low, f_high))
    return bands


# ---------------------------------------------------------------------------
# Especificações por tipo de janela (W_req e regra base para H)
# ---------------------------------------------------------------------------

def get_window_specs(window_type: str):
    """
    Devolve:
      - W_req: largura de main-lobe / separação requerida em bins;
      - hop_rule(M, W_req): H recomendado para esse tipo de janela.

    Valores canónicos:
      Rectangular: W_req ≈ 2
      Hamming/Hann/Kaiser: W_req ≈ 4
      Blackman-Harris (4-term): W_req ≈ 6
    """
    wt = window_type.lower()

    if wt == "hamming":
        W_req = 4.0
        hop_rule = lambda M, W: max(1, M // 4)   # 75 % overlap
    elif wt in ("hann", "hanning"):
        W_req = 4.0
        hop_rule = lambda M, W: max(1, M // 4)   # 75 % overlap
    elif wt in ("blackman-harris", "blackmanharris", "blackman_harris"):
        W_req = 6.0
        hop_rule = lambda M, W: max(1, M // 6)   # β = 6
    elif wt == "rectangular":
        W_req = 2.0
        hop_rule = lambda M, W: max(1, M // 2)   # 50 % overlap
    elif wt == "kaiser":
        W_req = 4.0
        hop_rule = lambda M, W: max(1, M // 4)
    else:
        raise ValueError(f"Unsupported window_type: {window_type}")

    return W_req, hop_rule


# ---------------------------------------------------------------------------
# Regra para zero padding (factor recomendado)
# ---------------------------------------------------------------------------

def get_recommended_zero_padding_factor(W_req: float) -> float:
    """
    Regra simples baseada na literatura partilhada:

      - zero padding factor R = N / M
      - queremos pelo menos ~8 bins ao longo do main lobe após o padding:
            W_req * R >= 8  ->  R >= 8 / W_req
      - e impomos R >= 2.

      Rectangular (W≈2)        -> R≈4
      Hann/Hamming/Kaiser (4)  -> R≈2
      Blackman-Harris (≈6)     -> R≈2
    """
    if W_req <= 0:
        return 2.0
    r_min_for_8_bins = 8.0 / W_req
    r = max(2.0, math.ceil(r_min_for_8_bins))
    return r


# ---------------------------------------------------------------------------
# Cálculo de parâmetros STFT para uma banda (multi-banda)
# ---------------------------------------------------------------------------

def compute_band_params_for_window(
    band_name: str,
    f_low_hz: float,
    f_high_hz: float,
    fs: float,
    duration_s: float,
    window_type: str,
    w_req_custom: Optional[float] = None,
) -> BandSTFTParams:
    """
    Multi-banda (guidelines por registo):

      fa = f_low (pior caso)
      M_min = ceil(fs * W_req / fa)

    duration_s define apenas o M máximo disponível (plateau comum).
    Se M_min > max_M, corta em max_M e lança aviso.

    Zero padding:
      - factor recomendado R = get_recommended_zero_padding_factor(W_req)
      - N_target = M * R
      - N_fft = próxima potência de 2 >= N_target
      - factor efectivo = N_fft / M
    """
    if f_low_hz <= 0 or f_high_hz <= f_low_hz:
        raise ValueError("Band must satisfy 0 < f_low < f_high.")
    if fs <= 0:
        raise ValueError("fs must be > 0.")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0.")

    W_default, hop_rule = get_window_specs(window_type)
    W_req = w_req_custom if (w_req_custom is not None and w_req_custom > 0) else W_default

    fa = f_low_hz
    M_min = int(math.ceil(fs * W_req / fa))
    max_M = int(duration_s * fs)

    warning_parts = []

    if M_min > max_M:
        M = max_M
        warning_parts.append(
            f"{band_name}: theoretical window length M_min={M_min} exceeds "
            f"available samples {max_M} (duration {duration_s:.3f} s). "
            "Frequency resolution is below the ideal W_req/fa criterion."
        )
    else:
        M = M_min

    if M < 2:
        raise ValueError(f"{band_name}: window length < 2 samples.")
    if M % 2 == 1:
        M += 1  # força par

    # Hop size
    H = hop_rule(M, W_req)
    if H < 1:
        H = 1

    # Critério teórico H <= M / W_req
    H_max = M / max(W_req, 1.0)
    if H > H_max + 1e-9:
        H = max(1, int(math.floor(H_max)))
        warning_parts.append(
            f"{band_name}: hop size adjusted to satisfy H <= M/W_req (H_max≈{H_max:.1f})."
        )

    # Zero padding
    zp_rec = get_recommended_zero_padding_factor(W_req)
    N_target = int(math.ceil(M * zp_rec))
    n_fft = get_next_power_of_two(N_target)
    zp_act = n_fft / float(M)

    window_duration_s = M / fs
    delta_f_window_hz = fs / float(M)
    delta_f_bins_hz = fs / float(n_fft)
    periods_at_f_low = window_duration_s * f_low_hz
    overlap_percent = 100.0 * (1.0 - H / float(M))

    warning = "\n".join(warning_parts) if warning_parts else None

    return BandSTFTParams(
        band_name=band_name,
        f_low_hz=f_low_hz,
        f_high_hz=f_high_hz,
        window_type=window_type,
        window_length=M,
        window_duration_s=window_duration_s,
        hop_size=H,
        overlap_percent=overlap_percent,
        n_fft=n_fft,
        delta_f_window_hz=delta_f_window_hz,
        delta_f_bins_hz=delta_f_bins_hz,
        periods_at_f_low=periods_at_f_low,
        zp_factor_recommended=zp_rec,
        zp_factor_actual=zp_act,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# Cálculo para todas as bandas
# ---------------------------------------------------------------------------

def compute_all_bands(
    fs: float,
    duration_s: float,
    f_min_hz: float,
    f_max_hz: float,
    n_bands: int,
    window_type: str,
    w_req_custom: Optional[float] = None,
) -> List[BandSTFTParams]:
    """
    Multi-banda:
      - bandas logarítmicas entre f_min_hz e f_max_hz;
      - M distinto em cada banda, com M_min = fs * W_req / f_low;
      - duration_s limita M por cima;
      - aconselha factor de zero padding por banda.
    """
    bands = define_log_frequency_bands(f_min_hz, f_max_hz, n_bands)
    params_list: List[BandSTFTParams] = []

    for idx, (f_low, f_high) in enumerate(bands, start=1):
        band_name = f"Band {idx}"
        params_list.append(
            compute_band_params_for_window(
                band_name=band_name,
                f_low_hz=f_low,
                f_high_hz=f_high,
                fs=fs,
                duration_s=duration_s,
                window_type=window_type,
                w_req_custom=w_req_custom,
            )
        )
    return params_list


# ---------------------------------------------------------------------------
# Nota estacionária individual (usa quase toda a duração)
# ---------------------------------------------------------------------------

def compute_single_note_stationary_params(
    f0_hz: float,
    fs: float,
    duration_s: float,
    window_type: str,
    w_req_custom: Optional[float] = None,
) -> BandSTFTParams:
    """
    Nota isolada, estável, com plateau estacionário de duração duration_s.
    Usa M ≈ duration_s * fs, verifica critério teórico mínimo e aplica zero padding.
    """
    if f0_hz <= 0:
        raise ValueError("f0_hz must be > 0.")
    if fs <= 0:
        raise ValueError("fs must be > 0.")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0.")

    W_default, hop_rule = get_window_specs(window_type)
    W_req = w_req_custom if (w_req_custom is not None and w_req_custom > 0) else W_default

    fa = f0_hz
    M_min = int(math.ceil(fs * W_req / fa))
    max_M = int(duration_s * fs)

    warning_parts = []
    if max_M < 2:
        raise ValueError("Stationary plateau too short (< 2 samples).")

    M = max_M
    if max_M < M_min:
        warning_parts.append(
            f"Single note: stationary window M={M} < theoretical M_min={M_min}. "
            "Frequency resolution is below the ideal W_req/fa criterion."
        )

    if M % 2 == 1:
        M += 1

    H = hop_rule(M, W_req)
    if H < 1:
        H = 1

    H_max = M / max(W_req, 1.0)
    if H > H_max + 1e-9:
        H = max(1, int(math.floor(H_max)))
        warning_parts.append(
            f"Single note: hop size adjusted to satisfy H <= M/W_req (H_max≈{H_max:.1f})."
        )

    # Zero padding
    zp_rec = get_recommended_zero_padding_factor(W_req)
    N_target = int(math.ceil(M * zp_rec))
    n_fft = get_next_power_of_two(N_target)
    zp_act = n_fft / float(M)

    window_duration_s = M / fs
    delta_f_window_hz = fs / float(M)
    delta_f_bins_hz = fs / float(n_fft)
    periods_at_f_low = window_duration_s * f0_hz
    overlap_percent = 100.0 * (1.0 - H / float(M))

    warning = "\n".join(warning_parts) if warning_parts else None

    return BandSTFTParams(
        band_name="single_note",
        f_low_hz=f0_hz,
        f_high_hz=f0_hz,
        window_type=window_type,
        window_length=M,
        window_duration_s=window_duration_s,
        hop_size=H,
        overlap_percent=overlap_percent,
        n_fft=n_fft,
        delta_f_window_hz=delta_f_window_hz,
        delta_f_bins_hz=delta_f_bins_hz,
        periods_at_f_low=periods_at_f_low,
        zp_factor_recommended=zp_rec,
        zp_factor_actual=zp_act,
        warning=warning,
    )


# ---------------------------------------------------------------------------
# GUI Tkinter com dois separadores: multi-banda e nota única
# ---------------------------------------------------------------------------

class MultiBandSTFTGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("STFT Parameter Advisor (Multi-band & Single Note)")
        self.resizable(True, True)

        self._init_vars()
        self._build_widgets()

    def _init_vars(self) -> None:
        # Multi-banda (registo)
        self.fs_var = tk.StringVar(value="44100")
        self.duration_var = tk.StringVar(value="2.0")    # plateau mínimo comum
        self.fmin_var = tk.StringVar(value="130.0")
        self.fmax_var = tk.StringVar(value="1000.0")
        self.nbands_var = tk.StringVar(value="5")
        self.window_type_var = tk.StringVar(value="Blackman-Harris")
        self.w_req_var = tk.StringVar(value="6.0")

        # Nota única estacionária
        self.single_f0_var = tk.StringVar(value="440.0")
        self.single_duration_var = tk.StringVar(value="2.0")
        self.single_window_type_var = tk.StringVar(value="Hann")
        self.single_w_req_var = tk.StringVar(value="4.0")

    def _build_widgets(self) -> None:
        padding = {"padx": 8, "pady": 4}
        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        # ----------------- Separador 1: multi-banda (registo) -----------------
        mb_frame = ttk.Frame(notebook, padding=8)
        notebook.add(mb_frame, text="Register (multi-band)")

        row = 0
        ttk.Label(mb_frame, text="Sampling rate fs (Hz):").grid(row=row, column=0, sticky="e", **padding)
        ttk.Entry(mb_frame, textvariable=self.fs_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(mb_frame, text="Max stationary duration (s)\n(usable in all notes):").grid(
            row=row, column=0, sticky="e", **padding
        )
        ttk.Entry(mb_frame, textvariable=self.duration_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(mb_frame, text="Lowest f0 in register (Hz):").grid(row=row, column=0, sticky="e", **padding)
        ttk.Entry(mb_frame, textvariable=self.fmin_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(mb_frame, text="Highest f0 in register (Hz):").grid(row=row, column=0, sticky="e", **padding)
        ttk.Entry(mb_frame, textvariable=self.fmax_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(mb_frame, text="Number of bands:").grid(row=row, column=0, sticky="e", **padding)
        ttk.Entry(mb_frame, textvariable=self.nbands_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(mb_frame, text="Window type:").grid(row=row, column=0, sticky="e", **padding)
        self.window_type_combo = ttk.Combobox(
            mb_frame,
            textvariable=self.window_type_var,
            values=["Hann", "Hamming", "Blackman-Harris", "Rectangular", "Kaiser"],
            state="readonly",
            width=15,
        )
        self.window_type_combo.grid(row=row, column=1, sticky="w")
        self.window_type_combo.bind("<<ComboboxSelected>>", self._on_window_type_change_multiband)

        row += 1
        ttk.Label(mb_frame, text="Required main-lobe / Δs (bins):").grid(
            row=row, column=0, sticky="e", **padding
        )
        ttk.Entry(mb_frame, textvariable=self.w_req_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Button(mb_frame, text="Compute bands", command=self._on_compute_multiband).grid(
            row=row, column=0, columnspan=2, pady=(8, 4)
        )

        row += 1
        ttk.Label(mb_frame, text="Recommended parameters per band:").grid(
            row=row, column=0, columnspan=2, sticky="w", **padding
        )

        row += 1
        cols = (
            "Band",
            "Range (Hz)",
            "Window",
            "M (samples)",
            "M (s)",
            "H (samples)",
            "Overlap (%)",
            "N_fft",
            "Δf_window (Hz)",
            "Δf_bins (Hz)",
            "Periods@f_low",
            "ZP_rec",
            "ZP_act",
        )
        self.tree = ttk.Treeview(mb_frame, columns=cols, show="headings", height=8)
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, anchor="center")
        self.tree.grid(row=row, column=0, columnspan=2, sticky="nsew")

        row += 1
        ttk.Label(mb_frame, text="Warnings and notes:").grid(
            row=row, column=0, columnspan=2, sticky="w", **padding
        )

        row += 1
        self.warning_text = tk.Text(mb_frame, height=6, width=100, state="disabled")
        self.warning_text.grid(row=row, column=0, columnspan=2, sticky="nsew")

        mb_frame.columnconfigure(0, weight=0)
        mb_frame.columnconfigure(1, weight=1)
        mb_frame.rowconfigure(row, weight=1)

        # ----------------- Separador 2: nota única estacionária -----------------
        sn_frame = ttk.Frame(notebook, padding=8)
        notebook.add(sn_frame, text="Single stationary note")

        row = 0
        ttk.Label(sn_frame, text="Sampling rate fs (Hz):").grid(row=row, column=0, sticky="e", **padding)
        ttk.Entry(sn_frame, textvariable=self.fs_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(sn_frame, text="Fundamental f0 (Hz):").grid(row=row, column=0, sticky="e", **padding)
        ttk.Entry(sn_frame, textvariable=self.single_f0_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(sn_frame, text="Stationary duration (s)\n(central plateau of this note):").grid(
            row=row, column=0, sticky="e", **padding
        )
        ttk.Entry(sn_frame, textvariable=self.single_duration_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Label(sn_frame, text="Window type:").grid(row=row, column=0, sticky="e", **padding)
        self.single_window_combo = ttk.Combobox(
            sn_frame,
            textvariable=self.single_window_type_var,
            values=["Hann", "Hamming", "Blackman-Harris", "Rectangular", "Kaiser"],
            state="readonly",
            width=15,
        )
        self.single_window_combo.grid(row=row, column=1, sticky="w")
        self.single_window_combo.bind("<<ComboboxSelected>>", self._on_window_type_change_single)

        row += 1
        ttk.Label(sn_frame, text="Required main-lobe / Δs (bins):").grid(
            row=row, column=0, sticky="e", **padding
        )
        ttk.Entry(sn_frame, textvariable=self.single_w_req_var, width=10).grid(row=row, column=1, sticky="w")

        row += 1
        ttk.Button(sn_frame, text="Compute single-note params", command=self._on_compute_single).grid(
            row=row, column=0, columnspan=2, pady=(8, 4)
        )

        row += 1
        self.single_output = tk.Text(sn_frame, height=12, width=80, state="disabled")
        self.single_output.grid(row=row, column=0, columnspan=2, sticky="nsew")

        sn_frame.columnconfigure(0, weight=0)
        sn_frame.columnconfigure(1, weight=1)
        sn_frame.rowconfigure(row, weight=1)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

    # ----------------- Callbacks -----------------

    def _on_window_type_change_multiband(self, event=None) -> None:
        wt = self.window_type_var.get().lower()
        if wt == "rectangular":
            default_W = 2.0
        elif wt in ("hann", "hanning", "hamming", "kaiser"):
            default_W = 4.0
        elif wt in ("blackman-harris", "blackmanharris", "blackman_harris"):
            default_W = 6.0
        else:
            default_W = 4.0
        self.w_req_var.set(f"{default_W:.1f}")

    def _on_window_type_change_single(self, event=None) -> None:
        wt = self.single_window_type_var.get().lower()
        if wt == "rectangular":
            default_W = 2.0
        elif wt in ("hann", "hanning", "hamming", "kaiser"):
            default_W = 4.0
        elif wt in ("blackman-harris", "blackmanharris", "blackman_harris"):
            default_W = 6.0
        else:
            default_W = 4.0
        self.single_w_req_var.set(f"{default_W:.1f}")

    def _on_compute_multiband(self) -> None:
        try:
            fs = float(self.fs_var.get().replace(",", "."))
            duration = float(self.duration_var.get().replace(",", "."))
            fmin = float(self.fmin_var.get().replace(",", "."))
            fmax = float(self.fmax_var.get().replace(",", "."))
            nbands = int(self.nbands_var.get())
            w_req = float(self.w_req_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric values.")
            return

        window_type = self.window_type_var.get()

        try:
            params_list = compute_all_bands(
                fs=fs,
                duration_s=duration,
                f_min_hz=fmin,
                f_max_hz=fmax,
                n_bands=nbands,
                window_type=window_type,
                w_req_custom=w_req,
            )
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        # Limpa tabela
        for item in self.tree.get_children():
            self.tree.delete(item)

        warnings = []
        for p in params_list:
            self.tree.insert(
                "",
                "end",
                values=(
                    p.band_name,
                    f"{p.f_low_hz:.1f}–{p.f_high_hz:.1f}",
                    p.window_type,
                    p.window_length,
                    f"{p.window_duration_s:.4f}",
                    p.hop_size,
                    f"{p.overlap_percent:.1f}",
                    p.n_fft,
                    f"{p.delta_f_window_hz:.3f}",
                    f"{p.delta_f_bins_hz:.3f}",
                    f"{p.periods_at_f_low:.2f}",
                    f"{p.zp_factor_recommended:.1f}",
                    f"{p.zp_factor_actual:.2f}",
                ),
            )
            if p.warning:
                warnings.append(p.warning)

        self.warning_text.configure(state="normal")
        self.warning_text.delete("1.0", tk.END)
        if warnings:
            self.warning_text.insert("1.0", "\n\n".join(warnings))
        else:
            self.warning_text.insert(
                "1.0",
                "No warnings. Theoretical criteria satisfied (or nearly satisfied) for all bands.",
            )
        self.warning_text.configure(state="disabled")

    def _on_compute_single(self) -> None:
        try:
            fs = float(self.fs_var.get().replace(",", "."))
            f0 = float(self.single_f0_var.get().replace(",", "."))
            duration = float(self.single_duration_var.get().replace(",", "."))
            w_req = float(self.single_w_req_var.get().replace(",", "."))
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric values.")
            return

        window_type = self.single_window_type_var.get()

        try:
            params = compute_single_note_stationary_params(
                f0_hz=f0,
                fs=fs,
                duration_s=duration,
                window_type=window_type,
                w_req_custom=w_req,
            )
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return

        lines = [
            f"Single stationary note, f0 = {params.f_low_hz:.2f} Hz, window = {params.window_type}",
            "",
            f"Window length M       : {params.window_length} samples",
            f"Window duration       : {params.window_duration_s:.4f} s",
            f"Hop size H            : {params.hop_size} samples",
            f"Overlap               : {params.overlap_percent:.1f} %",
            f"FFT size N_fft        : {params.n_fft}",
            f"Δf_window (fs/M)      : {params.delta_f_window_hz:.5f} Hz",
            f"Δf_bins (fs/N_fft)    : {params.delta_f_bins_hz:.5f} Hz",
            f"Periods at f0         : {params.periods_at_f_low:.2f}",
            f"Zero-padding factor R (recommended): {params.zp_factor_recommended:.1f}",
            f"Zero-padding factor R (actual)     : {params.zp_factor_actual:.2f}",
        ]
        if params.warning:
            lines.append("")
            lines.append("Warnings:")
            lines.append(params.warning)

        self.single_output.configure(state="normal")
        self.single_output.delete("1.0", tk.END)
        self.single_output.insert("1.0", "\n".join(lines))
        self.single_output.configure(state="disabled")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def run_cli() -> None:
    print("\n--- STFT Parameter Advisor (CLI) ---")

    def ask_float(prompt: str, default: Optional[float] = None) -> float:
        s = input(f"{prompt} " + (f"[default {default}]: " if default is not None else ": ")).strip()
        if not s:
            if default is None:
                raise ValueError("Value required.")
            return float(default)
        return float(s.replace(",", "."))

    def ask_int(prompt: str, default: Optional[int] = None) -> int:
        s = input(f"{prompt} " + (f"[default {default}]: " if default is not None else ": ")).strip()
        if not s:
            if default is None:
                raise ValueError("Value required.")
            return int(s)

    mode = input("Mode [register/single] [default register]: ").strip().lower()
    if not mode:
        mode = "register"

    if mode == "register":
        fs = ask_float("Sampling rate fs (Hz)", 44100.0)
        duration = ask_float("Max stationary duration usable in all notes (s)", 2.0)
        fmin = ask_float("Lowest f0 in register (Hz)", 130.0)
        fmax = ask_float("Highest f0 in register (Hz)", 1000.0)
        nbands = ask_int("Number of bands", 5)

        wt = input(
            "Window type [Hann/Hamming/Blackman-Harris/Rectangular/Kaiser] "
            "[default Blackman-Harris]: "
        ).strip()
        if not wt:
            wt = "Blackman-Harris"

        w_req_str = input("Required main-lobe / Δs (bins) [Enter = default for window]: ").strip()
        w_req_custom = float(w_req_str.replace(",", ".")) if w_req_str else None

        params_list = compute_all_bands(
            fs=fs,
            duration_s=duration,
            f_min_hz=fmin,
            f_max_hz=fmax,
            n_bands=nbands,
            window_type=wt,
            w_req_custom=w_req_custom,
        )

        print("\n--- Recommended Parameters per Band ---")
        for p in params_list:
            print(f"\n{p.band_name}: {p.f_low_hz:.1f}–{p.f_high_hz:.1f} Hz, window={p.window_type}")
            print(f"  M = {p.window_length} samples  (~{p.window_duration_s:.4f} s)")
            print(f"  H = {p.hop_size} samples  (overlap ≈ {p.overlap_percent:.1f} %)")
            print(f"  N_fft = {p.n_fft}")
            print(f"  Δf_window = {p.delta_f_window_hz:.5f} Hz (fs/M)")
            print(f"  Δf_bins   = {p.delta_f_bins_hz:.5f} Hz (fs/N_fft)")
            print(f"  Periods at f_low ≈ {p.periods_at_f_low:.2f}")
            print(f"  Zero-padding factor R (recommended): {p.zp_factor_recommended:.1f}")
            print(f"  Zero-padding factor R (actual)     : {p.zp_factor_actual:.2f}")
            if p.warning:
                print("  WARNING:")
                for line in p.warning.splitlines():
                    print(f"    {line}")
    else:
        fs = ask_float("Sampling rate fs (Hz)", 44100.0)
        f0 = ask_float("Fundamental f0 (Hz)", 440.0)
        duration = ask_float("Stationary duration of this note (s)", 2.0)

        wt = input(
            "Window type [Hann/Hamming/Blackman-Harris/Rectangular/Kaiser] "
            "[default Hann]: "
        ).strip()
        if not wt:
            wt = "Hann"

        w_req_str = input("Required main-lobe / Δs (bins) [Enter = default for window]: ").strip()
        w_req_custom = float(w_req_str.replace(",", ".")) if w_req_str else None

        params = compute_single_note_stationary_params(
            f0_hz=f0,
            fs=fs,
            duration_s=duration,
            window_type=wt,
            w_req_custom=w_req_custom,
        )

        print("\n--- Recommended Parameters for Single Stationary Note ---")
        print(f"  f0 = {params.f_low_hz:.2f} Hz, window = {params.window_type}")
        print(f"  M = {params.window_length} samples  (~{params.window_duration_s:.4f} s)")
        print(f"  H = {params.hop_size} samples  (overlap ≈ {params.overlap_percent:.1f} %)")
        print(f"  N_fft = {params.n_fft}")
        print(f"  Δf_window = {params.delta_f_window_hz:.5f} Hz (fs/M)")
        print(f"  Δf_bins   = {params.delta_f_bins_hz:.5f} Hz (fs/N_fft)")
        print(f"  Periods at f0 ≈ {params.periods_at_f_low:.2f}")
        print(f"  Zero-padding factor R (recommended): {params.zp_factor_recommended:.1f}")
        print(f"  Zero-padding factor R (actual)     : {params.zp_factor_actual:.2f}")
        if params.warning:
            print("  WARNING:")
            for line in params.warning.splitlines():
                print(f"    {line}")


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Optimal STFT parameters for spectral analysis of monodic orchestral instruments, "
            "with register-based multi-band guidelines, single-note stationary mode, "
            "and advised zero-padding factors."
        )
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in command-line mode instead of launching the Tkinter GUI.",
    )
    args = parser.parse_args()

    if args.cli:
        run_cli()
    else:
        app = MultiBandSTFTGui()
        app.mainloop()


if __name__ == "__main__":
    main()
