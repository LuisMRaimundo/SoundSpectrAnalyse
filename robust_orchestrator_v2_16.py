"""
Robust Orchestrator - Scientific Edition V3.6 (Final)

Features:
1. 12-Tier Granular Clustering (Zero Decalage).
2. Blackman-Harris Alignment (Hop = N/8).
3. Full Adaptive HPF Logic (Restored from Spectranalysis Tool).
4. Smart Zero Padding per Tier.
"""

import matplotlib
matplotlib.use('Agg') # Backend headless (no-GUI) para evitar crashes

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import queue
import logging
import gc
import re
import sys
import datetime
from pathlib import Path
from typing import Dict, List, Any

# --- DEPENDENCIES ---
try:
    import librosa
    import soundfile as sf
    import matplotlib.pyplot as plt
    import proc_audio
    import compile_metrics
except ImportError as e:
    raise ImportError(f"CRITICAL: Falta biblioteca. Detalhes: {e}")

# --- 12-TIER GRANULAR CONFIGURATION ---
# Scientific Alignment:
# ZP diminui antes de N_FFT diminuir. Hop sempre N/8.

FFT_SETTINGS_BY_CLUSTER = {
    # SUB / BASS muito grave (onde Δf em Hz é crítico)
    'Tier_01_Infrasound': {'max_freq': 40,    'n_fft': 16384, 'tolerance': 3.0,  'zp': 2},
    'Tier_02_DeepSub':    {'max_freq': 70,    'n_fft': 8192,  'tolerance': 4.0,  'zp': 2},
    'Tier_03_SubBass_A':  {'max_freq': 95,    'n_fft': 8192,  'tolerance': 5.0,  'zp': 2},
    'Tier_03_SubBass_B':  {'max_freq': 125,   'n_fft': 4096,  'tolerance': 5.5,  'zp': 2},

    # BASS -> LOW MID
    'Tier_04_Bass_A':     {'max_freq': 170,   'n_fft': 4096,  'tolerance': 6.0,  'zp': 2},
    'Tier_04_Bass_B':     {'max_freq': 240,   'n_fft': 4096,  'tolerance': 7.0,  'zp': 2},
    'Tier_05_UpperBass':  {'max_freq': 330,   'n_fft': 2048,  'tolerance': 8.5,  'zp': 2},

    # MIDS
    'Tier_06_LowMid_A':   {'max_freq': 450,   'n_fft': 2048,  'tolerance': 9.5,  'zp': 2},
    'Tier_07_LowMid_B':   {'max_freq': 650,   'n_fft': 2048,  'tolerance': 10.5, 'zp': 2},
    'Tier_08_Mid_A':      {'max_freq': 950,   'n_fft': 1024,  'tolerance': 12.0, 'zp': 2},
    'Tier_08_Mid_B':      {'max_freq': 1400,  'n_fft': 1024,  'tolerance': 13.0, 'zp': 2},

    # HIGH MIDS / TREBLE
    'Tier_09_Mid_C':      {'max_freq': 2200,  'n_fft': 1024,  'tolerance': 15.0, 'zp': 2},
    'Tier_10_HighMid':    {'max_freq': 3500,  'n_fft': 512,   'tolerance': 18.0, 'zp': 2},
    'Tier_11_Presence':   {'max_freq': 6000,  'n_fft': 512,   'tolerance': 20.0, 'zp': 2},
    'Tier_12_Brilliance': {'max_freq': float('inf'),
                           'n_fft': 512,      'tolerance': 22.0, 'zp': 1},
}

VALID_AUDIO_EXTENSIONS = {'.wav', '.mp3', '.aif', '.aiff', '.flac'}

# --- LOGGING ---
log = logging.getLogger("RobustOrchestrator")
log.setLevel(logging.INFO)

class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
    def emit(self, record: logging.LogRecord):
        self.log_queue.put(self.format(record))

class RobustOrchestratorApp:
    def __init__(self, master: tk.Tk):
        self.master = master
        master.title("Robust Orchestrator - Scientific Granular V3.6")
        master.geometry("1000x750")

        self.processing_queue: List[Path] = []
        self.is_running = False
        self.stop_requested = False
        self.log_queue = queue.Queue()
        log.addHandler(QueueLogHandler(self.log_queue))

        self._build_ui()
        self.master.after(100, self._process_log_queue)

    def _build_ui(self):
        # Frame 1: Inputs
        frame_input = ttk.LabelFrame(self.master, text="1. Input Folders")
        frame_input.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(frame_input, text="Add Folder(s)", command=self._add_folders).pack(side=tk.LEFT, padx=5, pady=5)
        ttk.Button(frame_input, text="Clear Queue", command=self._clear_queue).pack(side=tk.LEFT, padx=5, pady=5)
        self.lbl_count = ttk.Label(frame_input, text="Queue: 0 folders")
        self.lbl_count.pack(side=tk.LEFT, padx=15)

        # Frame 2: Settings
        frame_options = ttk.LabelFrame(self.master, text="2. Acoustic Physics & Metrics")
        frame_options.pack(fill=tk.X, padx=10, pady=5)

        # Col 1
        col1 = ttk.Frame(frame_options)
        col1.grid(row=0, column=0, padx=10, pady=5, sticky="n")
        ttk.Label(col1, text="Window Type:").pack(anchor="w")
        self.combo_window = ttk.Combobox(col1, values=["blackmanharris", "blackman", "hann"], state="readonly")
        self.combo_window.set("blackmanharris") 
        self.combo_window.pack(fill=tk.X)
        
        ttk.Label(col1, text="Magnitude Range (dB):").pack(anchor="w", pady=(10,0))
        self.entry_min_db = ttk.Entry(col1, width=10); self.entry_min_db.insert(0, "-90.0"); self.entry_min_db.pack(fill=tk.X)
        self.entry_max_db = ttk.Entry(col1, width=10); self.entry_max_db.insert(0, "0.0"); self.entry_max_db.pack(fill=tk.X)

        # Col 2
        col2 = ttk.Frame(frame_options)
        col2.grid(row=0, column=1, padx=10, pady=5, sticky="n")
        ttk.Label(col2, text="Dissonance Model:").pack(anchor="w")
        self.combo_dissonance = ttk.Combobox(col2, state="readonly", values=["sethares", "hutchinson", "vassilakis", "ALL (Compare)"])
        self.combo_dissonance.set("sethares"); self.combo_dissonance.pack(fill=tk.X)

        ttk.Label(col2, text="Weight Function:").pack(anchor="w", pady=(10,0))
        self.combo_weight = ttk.Combobox(col2, values=["log", "linear", "quadratic"], state="readonly")
        self.combo_weight.set("log"); self.combo_weight.pack(fill=tk.X)
        
        ttk.Label(col2, text="Inharmonic Weight (Noise %):").pack(anchor="w", pady=(5,0))
        self.var_i_weight = tk.IntVar(value=5)
        ttk.Scale(col2, from_=0, to=100, variable=self.var_i_weight, command=self._upd_lbl).pack(fill=tk.X)
        self.lbl_weight = ttk.Label(col2, text="5% Noise / 95% Harm"); self.lbl_weight.pack(anchor="w")

        # Col 3
        col3 = ttk.Frame(frame_options)
        col3.grid(row=0, column=2, padx=10, pady=5, sticky="n")
        self.var_lft = tk.BooleanVar(value=False)
        ttk.Checkbutton(col3, text="Use LFT", variable=self.var_lft).pack(anchor="w")
        
        ttk.Label(col3, text="Time Avg:").pack(anchor="w")
        self.combo_avg = ttk.Combobox(col3, values=["mean", "median", "max"], state="readonly"); 
        self.combo_avg.set("mean"); self.combo_avg.pack(fill=tk.X)
        
        ttk.Separator(col3).pack(fill=tk.X, pady=10)
        self.var_smart = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="12-Tier Granular Clustering", variable=self.var_smart).pack(anchor="w")
        self.var_compile = tk.BooleanVar(value=True)
        ttk.Checkbutton(col3, text="Auto-Compile Report", variable=self.var_compile).pack(anchor="w")

        # Frame 3: Actions
        frame_act = tk.Frame(self.master)
        frame_act.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.btn_run = ttk.Button(frame_act, text="RUN PIPELINE", command=self._run); self.btn_run.pack(fill=tk.X)
        self.btn_stop = ttk.Button(frame_act, text="STOP", state=tk.DISABLED, command=self._stop); self.btn_stop.pack(pady=5)
        self.lbl_status = ttk.Label(frame_act, text="Idle", font=("Arial", 10, "bold")); self.lbl_status.pack()
        self.txt_log = tk.Text(frame_act, height=12, state=tk.DISABLED, bg="#f0f0f0"); self.txt_log.pack(fill=tk.BOTH, expand=True)

    def _upd_lbl(self, v): self.lbl_weight.config(text=f"{int(float(v))}% Noise / {100-int(float(v))}% Harm")
    
    def _process_log_queue(self):
        while not self.log_queue.empty():
            self.txt_log.config(state=tk.NORMAL)
            self.txt_log.insert(tk.END, self.log_queue.get() + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state=tk.DISABLED)
        self.master.after(100, self._process_log_queue)

    def _add_folders(self):
        d = filedialog.askdirectory(mustexist=True)
        if d and Path(d) not in self.processing_queue:
            self.processing_queue.append(Path(d))
            self.lbl_count.config(text=f"Queue: {len(self.processing_queue)}")
            log.info(f"Added: {Path(d).name}")

    def _clear_queue(self):
        if not self.is_running: self.processing_queue.clear(); self.lbl_count.config(text="Queue: 0"); log.info("Queue cleared.")

    def _stop(self): self.stop_requested = True; self.lbl_status.config(text="Stopping...")

    def _run(self):
        if not self.processing_queue: return
        self.is_running = True; self.stop_requested = False
        self.btn_run.config(state=tk.DISABLED); self.btn_stop.config(state=tk.NORMAL)
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            params = {
                'i_weight': self.var_i_weight.get()/100.0, 'lft': self.var_lft.get(),
                'avg': self.combo_avg.get(), 'win': self.combo_window.get(),
                'wf': self.combo_weight.get(), 'diss': self.combo_dissonance.get(),
                'db_min': float(self.entry_min_db.get()), 'db_max': float(self.entry_max_db.get()),
                'compile': self.var_compile.get(), 'smart': self.var_smart.get()
            }
        except Exception as e:
            log.error(f"Input Error: {e}"); self._reset(); return

        total = len(self.processing_queue)
        for i, folder in enumerate(self.processing_queue):
            if self.stop_requested: break
            log.info(f"--- FOLDER {i+1}/{total}: {folder.name} ---")
            self._process_folder(folder, params)
        
        log.info("Done."); messagebox.showinfo("Info", "Finished."); self._reset()

    def _process_folder(self, folder, p):
        if p['smart']:
            clusters = self._cluster_granular(folder)
        else:
            files = [f for f in folder.glob("*") if f.suffix.lower() in VALID_AUDIO_EXTENSIONS]
            clusters = {'Fallback': files} if files else {}

        report_data = []
        for tier, files in clusters.items():
            if self.stop_requested: break
            if not files: continue
            
            settings = FFT_SETTINGS_BY_CLUSTER.get(tier, {'n_fft': 2048, 'zp': 1, 'tolerance': 10.0})
            n_fft = settings['n_fft']
            zp = settings['zp']
            # Blackman-Harris requires high overlap (N/8) to conserve energy
            hop_length = n_fft // 8 

            # --- RESTORED ADAPTIVE HPF LOGIC (FULL) ---
            min_f0 = self._get_min_freq(files)
            
            # Default low fallback
            cutoff = 20.0 
            
            if min_f0 < 99999:
                # Logic from Spectranalysis_tool_5.py
                margin = 10.0 # Default safety margin %
                
                if min_f0 < 60:
                    margin = 35.0 # High margin for unstable sub-bass
                elif min_f0 < 120:
                    margin = 25.0 # Medium margin for bass
                elif min_f0 < 300:
                    margin = 15.0 # Lower margin for low-mids
                
                # Formula: Cutoff = Fundamental * (1 - Margin%)
                # Ensures we cut the rumble BUT keep the Fundamental intact.
                calculated_cutoff = min_f0 * (1.0 - margin / 100.0)
                cutoff = max(20.0, calculated_cutoff)
            
            log.info(f"  > {tier}: {len(files)} files | N={n_fft} | Hop={hop_length} | ZP={zp}x | HPF={cutoff:.1f}Hz")

            proc_args = {
                "n_fft": n_fft, "hop_length": hop_length, "win_length": n_fft,
                "window": p['win'], "freq_min": cutoff, "freq_max": 15000,
                "db_min": p['db_min'], "db_max": p['db_max'],
                "tolerance": settings['tolerance'], "use_adaptive_tolerance": True,
                "harmonic_weight": 1.0-p['i_weight'], "inharmonic_weight": p['i_weight'],
                "weight_function": p['wf'], 
                "dissonance_enabled": True, "compare_models": (p['diss']=="ALL (Compare)"),
                "dissonance_model": "sethares" if p['diss']=="ALL (Compare)" else p['diss'],
                "use_lft": p['lft'], "zero_padding": zp, "time_avg": p['avg'],
                "dissonance_curve": True, "dissonance_scale": True,
                "results_directory": folder, "progress_callback": None
            }

            try:
                pr = proc_audio.AudioProcessor()
                pr.load_audio_files(files)
                pr.apply_filters_and_generate_data(**proc_args)
                del pr; plt.close('all'); gc.collect()
                report_data.append({'tier': tier, 'count': len(files), 'n': n_fft, 'hop': hop_length})
            except Exception as e: log.error(f"Batch failed: {e}")

        if p['compile'] and not self.stop_requested:
            try:
                compile_metrics.compile_density_metrics_with_pca(
                    folder_path=folder, output_path=folder/"compiled_metrics.xlsx",
                    include_pca=True, harmonic_weight=1.0-p['i_weight'], 
                    inharmonic_weight=p['i_weight'], weight_function=p['wf']
                )
                log.info("Excel Compiled.")
            except: log.error("Compile failed.")

    def _cluster_granular(self, folder):
        clusters = {k: [] for k in FFT_SETTINGS_BY_CLUSTER.keys()}; clusters['Fallback'] = []
        pat = re.compile(r"([A-G][#b]?-?\d+)")
        for f in [x for x in folder.glob("*") if x.suffix.lower() in VALID_AUDIO_EXTENSIONS]:
            try:
                with sf.SoundFile(str(f)) as sf_obj: 
                    if sf_obj.frames == 0: continue
            except: continue
            
            hz = 0
            m = pat.findall(f.name)
            if m: 
                try: hz = librosa.note_to_hz(m[-1])
                except: pass
            
            if hz > 0:
                assigned = False
                for tier, cfg in FFT_SETTINGS_BY_CLUSTER.items():
                    if hz < cfg['max_freq']:
                        clusters[tier].append(f); assigned = True; break
                if not assigned: clusters['Tier_12_Brilliance'].append(f)
            else: clusters['Fallback'].append(f)
        return {k:v for k,v in clusters.items() if v}

    def _get_min_freq(self, files):
        m = float('inf'); pat = re.compile(r"([A-G][#b]?-?\d+)")
        for f in files:
            matches = pat.findall(f.name)
            if matches:
                try: 
                    v = librosa.note_to_hz(matches[-1])
                    if v < m: m = v
                except: pass
        return m

    def _reset(self): self.is_running=False; self.btn_run.config(state=tk.NORMAL); self.btn_stop.config(state=tk.DISABLED); self.lbl_status.config(text="Idle")

if __name__ == "__main__":
    if not log.hasHandlers(): log.addHandler(logging.StreamHandler(sys.stdout))
    root = tk.Tk()
    RobustOrchestratorApp(root)
    root.mainloop()