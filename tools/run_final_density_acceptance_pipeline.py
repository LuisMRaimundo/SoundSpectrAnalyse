from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import compile_metrics
from note_parser import canonical_note_from_filename
from proc_audio import AudioProcessor
from tools.export_research_density_workbook import export_research_workbook


@dataclass
class CorpusRunResult:
    corpus_name: str
    corpus_dir: str
    run_dir: str
    row_count: int
    compiled_path: str
    research_path: str
    log_path: str
    first_per_note_workbook: str


DEFAULT_CFG: dict[str, Any] = {
    "freq_min": 20.0,
    "freq_max": 20000.0,
    "db_min": -90.0,
    "db_max": 0.0,
    "n_fft": 4096,
    "hop_length": 1024,
    "window": "blackmanharris",
    "tolerance": 5.0,
    "use_adaptive_tolerance": True,
    "zero_padding": 2,
    "time_avg": "median",
    "weight_function": "linear",
    "density_summation_mode": "his_weighted",
    "harmonic_density_weight": 1.0,
    "inharmonic_density_weight": 0.5,
    "subbass_density_weight": 0.25,
    "density_salience_threshold_db": -45.0,
    "density_frequency_ceiling_hz": 5000.0,
}


def _pick_window_params(window: str) -> tuple[float | None, float | None]:
    w = window.lower().strip()
    if w == "kaiser":
        return 14.0, None
    if w in ("gaussian", "gauss", "gaussiana"):
        return None, 512.0
    return None, None


def _collect_audio_files(corpus_dir: Path) -> list[Path]:
    files = sorted(
        p
        for p in corpus_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".wav", ".aif", ".aiff"}
    )
    if not files:
        raise RuntimeError(f"No audio files found in {corpus_dir}")
    return files


def run_corpus(corpus_name: str, corpus_dir: Path) -> CorpusRunResult:
    run_dir = corpus_dir / "analysis_results_final_density_acceptance"
    run_dir.mkdir(parents=True, exist_ok=True)

    compiled_path = run_dir / "compiled_density_metrics_final_density_acceptance.xlsx"
    research_path = run_dir / "compiled_density_metrics_research_final_density_acceptance.xlsx"
    log_path = run_dir / "gui_worker_final_density_acceptance.log"

    files = _collect_audio_files(corpus_dir)
    kaiser_beta, gaussian_std = _pick_window_params(str(DEFAULT_CFG["window"]))
    first_per_note_workbook = None

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"run_start corpus={corpus_name} files={len(files)}\n")
        for idx, wav in enumerate(files, start=1):
            note, note_source = canonical_note_from_filename(wav.name, parent_folder=wav.parent.name)
            parent_output_dir = run_dir / wav.stem

            ap = AudioProcessor()
            ap.note_source = note_source
            if note:
                ap.note = note
            ap.load_audio_files([str(wav)])
            ap.apply_filters_and_generate_data(
                freq_min=float(DEFAULT_CFG["freq_min"]),
                freq_max=float(DEFAULT_CFG["freq_max"]),
                db_min=float(DEFAULT_CFG["db_min"]),
                db_max=float(DEFAULT_CFG["db_max"]),
                n_fft=int(DEFAULT_CFG["n_fft"]),
                hop_length=int(DEFAULT_CFG["hop_length"]),
                window=str(DEFAULT_CFG["window"]),
                tolerance=float(DEFAULT_CFG["tolerance"]),
                use_adaptive_tolerance=bool(DEFAULT_CFG["use_adaptive_tolerance"]),
                results_directory=str(parent_output_dir),
                dissonance_enabled=False,
                compare_models=False,
                harmonic_weight=0.5,
                inharmonic_weight=0.5,
                auto_model_weights_from_analysis=True,
                weight_function=str(DEFAULT_CFG["weight_function"]),
                zero_padding=int(DEFAULT_CFG["zero_padding"]),
                time_avg=str(DEFAULT_CFG["time_avg"]),
                spectral_masking_enabled=False,
                density_summation_mode=str(DEFAULT_CFG["density_summation_mode"]),
                harmonic_density_weight=float(DEFAULT_CFG["harmonic_density_weight"]),
                inharmonic_density_weight=float(DEFAULT_CFG["inharmonic_density_weight"]),
                subbass_density_weight=float(DEFAULT_CFG["subbass_density_weight"]),
                density_salience_threshold_db=float(DEFAULT_CFG["density_salience_threshold_db"]),
                density_frequency_ceiling_hz=float(DEFAULT_CFG["density_frequency_ceiling_hz"]),
                tier=None,
                kaiser_beta=kaiser_beta,
                gaussian_std=gaussian_std,
                compile_per_call=False,
                use_tsne=False,
                use_umap=False,
                detect_anomalies=False,
                anomaly_contamination=None,
            )
            if first_per_note_workbook is None:
                note_dir = note if note else ""
                first_per_note_workbook = parent_output_dir / note_dir / "spectral_analysis.xlsx"
            log.write(f"processed {idx}/{len(files)} {wav.name}\n")

        compile_metrics.compile_density_metrics_with_pca(
            folder_path=str(run_dir),
            output_path=str(compiled_path),
            file_pattern="spectral_analysis.xlsx",
            include_pca=True,
            harmonic_weight=0.5,
            inharmonic_weight=0.5,
            weight_function=str(DEFAULT_CFG["weight_function"]),
            use_tsne=False,
            use_umap=False,
            detect_anomalies=False,
            anomaly_contamination=None,
            allow_legacy_super_json=False,
            compilation_extra_metadata={
                "input_schema_validation_status": "not_validated_orchestrator_v2_16",
                "legacy_pipeline_used": False,
                "publication_output_allowed": True,
                "source_corpus_path": str(corpus_dir),
                "output_path": str(run_dir),
            },
        )
        export_research_workbook(compiled_path, research_path, overwrite=True)
        log.write("run_complete\n")

    row_count = int(
        len(pd.read_excel(research_path, sheet_name="Spectral_Density_Metrics", engine="openpyxl"))
    )
    return CorpusRunResult(
        corpus_name=corpus_name,
        corpus_dir=str(corpus_dir),
        run_dir=str(run_dir),
        row_count=row_count,
        compiled_path=str(compiled_path),
        research_path=str(research_path),
        log_path=str(log_path),
        first_per_note_workbook=str(first_per_note_workbook) if first_per_note_workbook else "",
    )


def main() -> None:
    clarinet_dir = Path(os.environ.get("SSA_ACCEPTANCE_CLARINET_DIR", "<clarinet_corpus_path>"))
    cello_dir = Path(os.environ.get("SSA_ACCEPTANCE_CELLO_DIR", "<cello_corpus_path>"))
    if not clarinet_dir.is_dir() or not cello_dir.is_dir():
        raise RuntimeError(
            "Set SSA_ACCEPTANCE_CLARINET_DIR and SSA_ACCEPTANCE_CELLO_DIR to valid corpus folders."
        )
    corpora = [
        (
            "clarinet",
            clarinet_dir,
        ),
        (
            "cello",
            cello_dir,
        ),
    ]
    results = [asdict(run_corpus(name, cdir)) for name, cdir in corpora]
    out_json = REPO_ROOT / "audit_final_density_pipeline_runs.json"
    out_json.write_text(json.dumps({"runs": results}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
