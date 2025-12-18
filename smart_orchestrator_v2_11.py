import matplotlib
# CRITICAL FIX: Force Matplotlib to run in "Headless" mode (No GUI)
# This prevents the "Disappearing Window" crash when plotting from a thread.
matplotlib.use('Agg') 

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from pathlib import Path
import logging
import queue
import os
import re
import gc
import datetime
import sys
from typing import Dict, List

# --- IMPORT DEPENDENCIES ---
try:
    import librosa  # <--- CRUCIAL: Adicionado para ler notas (Hz)
    import Spectranalysis_tool_5 as tool5
    from log_config import configure_root_logger
    import soundfile as sf
    import proc_audio
    import compile_metrics
    import dissonance_models
    import matplotlib.pyplot as plt # Needed for memory cleanup
except ImportError as e:
    print(f"CRITICAL ERROR: Missing dependencies.\n{e}")

# --- SETUP ROBUST LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("pipeline_v3.log", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("SmartOrchestratorV3")

class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue
    def emit(self, record):
        self.log_queue.put(self.format(record))

class SmartOrchestratorApp:
    def __init__(self, master):
        self.master = master
        master.title("Smart Orchestrator V3.1 - Inverted Logic")
        master.geometry("900x650")
        
        self.processing_queue = []
        self.is_running = False
        self.stop_requested = False
        
        self.log_queue = queue.Queue()
        queue_handler = QueueLogHandler(self.log_queue)
        log.addHandler(queue_handler)
        
        self._build_ui()
        self.master.after(100, self.process_log_queue)

    def _build_ui(self):
        # --- TOP FRAME: Inputs ---
        frame_input = ttk.LabelFrame(self.master, text="Input Folders")
        frame_input.pack(fill=tk.X, padx=10, pady=5)
        
        self.btn_add = ttk.Button(frame_input, text="Add Folder(s)", command=self.add_folders)
        self.btn_add.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.btn_clear = ttk.Button(frame_input, text="Clear Queue", command=self.clear_queue)
        self.btn_clear.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.lbl_count = ttk.Label(frame_input, text="Queue: 0 folders")
        self.lbl_count.pack(side=tk.LEFT, padx=15)

        # --- MIDDLE FRAME: Acoustic Settings ---
        frame_options = ttk.LabelFrame(self.master, text="Acoustic Physics & Metrics")
        frame_options.pack(fill=tk.X, padx=10, pady=5)
        
        # Row 1: Weights (INVERTED LOGIC)
        ttk.Label(frame_options, text="Inharmonic Weight (Noise %):").grid(row=0, column=0, padx=5, pady=5, sticky="e")
        
        # Default 5 means 5% Noise (equivalent to old 95% Harmonic)
        self.var_inharmonic_weight = tk.IntVar(value=5) 
        
        self.scale_weight = ttk.Scale(frame_options, from_=0, to=100, orient=tk.HORIZONTAL, 
                                      variable=self.var_inharmonic_weight, command=self.update_slider_label)
        self.scale_weight.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        
        self.lbl_weight_val = ttk.Label(frame_options, text="5% Noise / 95% Harm")
        self.lbl_weight_val.grid(row=0, column=2, padx=5, sticky="w")

        # Row 2: Window & Dissonance
        ttk.Label(frame_options, text="Window Type:").grid(row=1, column=0, padx=5, pady=5, sticky="e")
        self.combo_window = ttk.Combobox(frame_options, values=["hann", "hamming", "blackman", "blackmanharris"], state="readonly", width=15)
        self.combo_window.current(3) # Blackman-Harris default
        self.combo_window.grid(row=1, column=1, padx=5, pady=5)

        ttk.Label(frame_options, text="Dissonance Model:").grid(row=2, column=0, padx=5, pady=5, sticky="e")
        self.combo_dissonance = ttk.Combobox(frame_options, state="readonly", width=18)
        self.combo_dissonance['values'] = [
            "sethares", "hutchinson", "vassilakis", "aures", "stolzenburg", "spectral", "ALL (Compare)"
        ]
        self.combo_dissonance.current(0)
        self.combo_dissonance.grid(row=2, column=1, padx=5, pady=5)

        # Row 3: Advanced Toggles
        self.var_use_lft = tk.BooleanVar(value=True) 
        ttk.Checkbutton(frame_options, text="Use LFT (Linear Frequency Transform)", variable=self.var_use_lft).grid(row=3, column=0, columnspan=2, padx=5, sticky="w")
        
        self.var_smart_mode = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_options, text="Smart Frequency Clustering (6-Tier Granular)", variable=self.var_smart_mode).grid(row=3, column=2, columnspan=2, padx=5, sticky="w")

        self.var_auto_compile = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame_options, text="Auto-Compile Excel Report", variable=self.var_auto_compile).grid(row=4, column=0, columnspan=2, padx=5, sticky="w")
        
        # Weight Function
        ttk.Label(frame_options, text="Weight Func:").grid(row=4, column=2, sticky="e")
        self.combo_weight = ttk.Combobox(frame_options, values=["linear", "log", "quadratic"], state="readonly", width=10)
        self.combo_weight.current(1) # Log default
        self.combo_weight.grid(row=4, column=3, padx=5, sticky="w")

        # --- BOTTOM FRAME: Actions & Log ---
        frame_actions = tk.Frame(self.master)
        frame_actions.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.btn_run = ttk.Button(frame_actions, text="RUN PIPELINE", command=self.run_pipeline)
        self.btn_run.pack(side=tk.TOP, fill=tk.X, pady=5)
        
        self.btn_stop = ttk.Button(frame_actions, text="STOP", state=tk.DISABLED, command=self.request_stop)
        self.btn_stop.pack(side=tk.TOP, pady=2)
        
        self.lbl_status = ttk.Label(frame_actions, text="Idle", font=("Arial", 10, "bold"))
        self.lbl_status.pack(side=tk.TOP, pady=5)

        self.txt_log = tk.Text(frame_actions, height=15, state=tk.DISABLED, bg="#f0f0f0", font=("Consolas", 9))
        self.txt_log.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

    def update_slider_label(self, val):
        i = int(float(val))
        # Now shows Noise first to align with slider
        self.lbl_weight_val.config(text=f"{i}% Noise / {100-i}% Harm")

    def process_log_queue(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.txt_log.config(state=tk.NORMAL)
            self.txt_log.insert(tk.END, msg + "\n")
            self.txt_log.see(tk.END)
            self.txt_log.config(state=tk.DISABLED)
        self.master.after(100, self.process_log_queue)

    def add_folders(self):
        dirs = filedialog.askdirectory(mustexist=True)
        if dirs:
            path = Path(dirs)
            if path not in self.processing_queue:
                self.processing_queue.append(path)
                self.lbl_count.config(text=f"Queue: {len(self.processing_queue)} folders")
                log.info(f"Added to queue: {path.name}")

    def clear_queue(self):
        self.processing_queue = []
        self.lbl_count.config(text="Queue: 0 folders")
        log.info("Queue cleared.")

    def request_stop(self):
        self.stop_requested = True
        self.lbl_status.config(text="Stopping... (Finish current batch)")

    def parse_note_to_hz(self, note_string):
        try:
            return librosa.note_to_hz(note_string)
        except:
            return 440.0

    def check_file_integrity(self, file_path: Path) -> bool:
        try:
            with sf.SoundFile(str(file_path)) as f:
                return f.frames > 0
        except Exception:
            return False

    # --- THE NEW GRANULAR CLUSTERING LOGIC (V3.0) ---
    def cluster_files_by_frequency(self, folder_path: Path) -> Dict[str, List[Path]]:
        clusters = {
            'Tier_0_DeepBass': [], 
            'Tier_1_Bass': [],     
            'Tier_2_LowMid': [],   
            'Tier_3_Mid': [],      
            'Tier_4_HighMid': [],  
            'Tier_5_Treble': [],   
            'Fallback': []
        }
        
        pattern = r"([A-G][#b]?-?\d+)"
        exts = {'.wav', '.mp3', '.aif', '.aiff', '.flac'}
        
        raw_files = [f for f in folder_path.glob("*") if f.suffix.lower() in exts]
        
        for f in raw_files:
            if not self.check_file_integrity(f):
                log.error(f"Skipped CORRUPT file: {f.name}")
                continue

            matches = re.findall(pattern, f.name)
            freq = None
            if matches: 
                try:
                    freq = librosa.note_to_hz(matches[-1]) 
                except: pass
            
            if freq:
                if freq < 50: clusters['Tier_0_DeepBass'].append((f, freq))
                elif freq < 150: clusters['Tier_1_Bass'].append((f, freq))
                elif freq < 400: clusters['Tier_2_LowMid'].append((f, freq))
                elif freq < 1000: clusters['Tier_3_Mid'].append((f, freq))
                elif freq < 3000: clusters['Tier_4_HighMid'].append((f, freq))
                else: clusters['Tier_5_Treble'].append((f, freq))
            else:
                clusters['Fallback'].append((f, 0))
                
        return clusters

    def generate_methodology_report(self, folder_path: Path, report_data: List[Dict]):
        report_path = folder_path / "_Acoustic_Methodology_Report.txt"
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(f"ACOUSTIC ANALYSIS METHODOLOGY REPORT (V3.1)\n")
                f.write(f"Target: {folder_path.name} | Date: {timestamp}\n")
                f.write(f"Config: {self.combo_dissonance.get()} | LFT: {self.var_use_lft.get()}\n")
                f.write("---------------------------------------------------------\n\n")
                for batch in report_data:
                    f.write(f"--- Batch: {batch['tier']} ---\n")
                    f.write(f"  Count: {batch['count']} | Base Freq: {batch['base_freq']:.2f} Hz\n")
                    f.write(f"  Settings: N={batch['n_fft']}, M={batch['win_len']}, Cutoff={batch['cutoff']:.2f}Hz\n\n")
        except Exception: pass

    def run_pipeline(self):
        if not self.processing_queue: return
        self.is_running = True; self.stop_requested = False
        self.btn_run.config(state=tk.DISABLED); self.btn_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text="Processing...")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        total_folders = len(self.processing_queue)
        
        # --- FIXED LOGIC: Slider now represents INHARMONIC weight ---
        i_weight_val = self.var_inharmonic_weight.get() / 100.0
        h_weight_val = 1.0 - i_weight_val
        
        use_lft_val = self.var_use_lft.get()
        weight_func = self.combo_weight.get()
        
        selected_diss_option = self.combo_dissonance.get()
        is_compare_mode = (selected_diss_option == "ALL (Compare)")
        model_to_send = "sethares" if is_compare_mode else selected_diss_option

        for idx, folder in enumerate(self.processing_queue):
            if self.stop_requested: break
            plt.close('all'); gc.collect()
            folder_report_data = []
            try:
                log.info(f"--- FOLDER {idx+1}/{total_folders}: {folder.name} ---")
                
                # CLUSTER
                if self.var_smart_mode.get():
                    log.info(">> Sorting files (V3.0 Granular Logic)...")
                    clusters = self.cluster_files_by_frequency(folder)
                else:
                    valid_files = [f for f in folder.glob("*") if f.suffix in {'.wav','.mp3','.aif','.flac'} and self.check_file_integrity(f)]
                    clusters = {'Fallback': [(f, 0) for f in valid_files]}

                # PROCESS
                for tier_name, file_tuples in clusters.items():
                    if not file_tuples: continue
                    if self.stop_requested: break
                    
                    files_only = [t[0] for t in file_tuples]
                    base_freq = min([t[1] for t in file_tuples if t[1] > 0] or [440.0])
                    
                    log.info(f"   Batch: {tier_name} ({len(files_only)} files) @ ~{base_freq:.1f}Hz")
                    
                    target_n_fft = 2048 
                    if "DeepBass" in tier_name: target_n_fft = 16384
                    elif "Bass" in tier_name: target_n_fft = 4096
                    elif "LowMid" in tier_name: target_n_fft = 2048
                    elif "Mid" in tier_name: target_n_fft = 1024
                    elif "HighMid" in tier_name: target_n_fft = 512
                    elif "Treble" in tier_name: target_n_fft = 256
                    
                    cutoff_hz = max(20.0, base_freq * 0.8)

                    params = {
                        "n_fft": target_n_fft, 
                        "hop_length": target_n_fft // 8, 
                        "win_length": target_n_fft, 
                        "window": self.combo_window.get(), 
                        "freq_min": float(cutoff_hz),
                        "freq_max": 22050.0, 
                        "db_min": -90.0, "db_max": 0.0,
                        "tolerance": 5.0 if base_freq < 100 else 10.0, 
                        "use_adaptive_tolerance": True,
                        "harmonic_weight": h_weight_val, "inharmonic_weight": i_weight_val,
                        "weight_function": weight_func, 
                        "parallel_processing": False, 
                        "export_data_format": "json",
                        "dissonance_enabled": True, 
                        "compare_models": is_compare_mode, 
                        "dissonance_model": model_to_send, 
                        "use_lft": use_lft_val,
                        "dissonance_curve": True, "dissonance_scale": True,
                        "zero_padding": 1, "time_avg": "median"
                    }

                    try:
                        processor = proc_audio.AudioProcessor()
                        processor.load_audio_files(files_only)
                        processor.apply_filters_and_generate_data(
                            **params, results_directory=folder, progress_callback=lambda c, t, l: None
                        )
                        folder_report_data.append({
                            "tier": tier_name, "count": len(files_only), "base_freq": base_freq,
                            "n_fft": target_n_fft, "win_len": target_n_fft, "hop": params['hop_length'],
                            "cutoff": cutoff_hz
                        })
                        del processor; plt.close('all'); gc.collect()
                        log.info(f"   ✓ Batch {tier_name} Complete.")
                    except Exception as e: log.error(f"   !!! Skipped Batch: {e}")

                if not self.stop_requested:
                    self.generate_methodology_report(folder, folder_report_data)
                    if self.var_auto_compile.get():
                        log.info("   >> Compiling Report...")
                        try:
                            compile_metrics.compile_density_metrics_with_pca(
                                folder_path=folder, output_path=folder/"compiled_density_metrics.xlsx",
                                include_pca=True, harmonic_weight=h_weight_val, 
                                inharmonic_weight=i_weight_val, weight_function=weight_func
                            )
                            log.info("   ✓ Saved Excel.")
                        except Exception as e: log.error(f"   !!! Compile Fail: {e}")

            except Exception as e: log.error(f"!!! FATAL ERROR: {e}")

        if not self.stop_requested:
            log.info("\nPIPELINE FINISHED.")
            messagebox.showinfo("Done", "Processing Complete.")
        self.master.after(0, self._reset_ui)

    def _reset_ui(self):
        self.is_running = False
        self.btn_run.config(state=tk.NORMAL); self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="Idle")

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartOrchestratorApp(root)
    root.mainloop()