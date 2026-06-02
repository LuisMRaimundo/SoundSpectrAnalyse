#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline orchestrator (integrated)
==================================

Three-stage pipeline:

    Stage 1 - Per-note spectral analysis
        For each audio file, runs proc_audio.AudioProcessor.
        Component energy ratios (harmonic / inharmonic / sub-bass) are
        computed from the current spectral analysis only.
        ``component_profile_source = current_analysis``
        ``model_weights_source     = current_analysis``
        Per-note ``spectral_analysis.xlsx`` is written under
        ``main_analysis_output_dir/<stem>/<note>/``.

    Stage 2 - Compilation
        Reads every per-note ``spectral_analysis.xlsx`` under
        ``main_analysis_output_dir`` and writes
        ``main_analysis_output_dir/compiled_density_metrics.xlsx``.

    Stage 3 - Research export + EWSD
        After Stage 2, ``post_compile_research_export`` builds
        ``compiled_density_metrics_research.xlsx`` and merges EWSD-R v18
        scores (``EWSD_score_total``, ``EWSD_score_acoustic_balanced``,
        provenance columns, ``ewsd_primary_analysis_eligible``) into
        ``Spectral_Density_Metrics`` when per-note workbooks are present.

This module does not run any preprocessing, does not consult any external
energy mapping, does not write any batch summary, and does not create a
batch_results folder. Component ratios come exclusively from the current
proc_audio analysis. If a per-note workbook lacks the canonical
``component_*`` ratios, Stage 2 surfaces ``error/missing_component_weights``
rather than falling back to legacy aliases.

Author: SoundSpectrAnalyse maintainers
"""

from __future__ import annotations

import sys
import logging
import traceback
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent / "audio_analysis"))

import pandas as pd
import numpy as np
import librosa

# Tier settings (FFT sizes per fundamental-frequency cluster).
try:
    from pipeline_orchestrator_gui import FFT_SETTINGS_BY_CLUSTER
except ImportError:
    FFT_SETTINGS_BY_CLUSTER = {
        'Tier_01': {'max_freq': 20, 'n_fft': 16384, 'tolerance': 3.0, 'zp': 2},
        'Tier_12': {'max_freq': 76, 'n_fft': 8192, 'tolerance': 4.6, 'zp': 2},
        'Tier_30': {'max_freq': 260, 'n_fft': 4096, 'tolerance': 8.2, 'zp': 2},
        'Tier_60': {'max_freq': 1450, 'n_fft': 1024, 'tolerance': 14.2, 'zp': 2},
        'Tier_75': {'max_freq': 3600, 'n_fft': 512, 'tolerance': 19.0, 'zp': 2},
        'Tier_90': {'max_freq': float('inf'), 'n_fft': 512, 'tolerance': 27.0, 'zp': 1},
    }

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# Legacy CLI flags removed in the Stage 1 / Stage 2 refactor.
_LEGACY_REJECTED_CLI_TOKENS = (
    "--phase1-mode",
    "--phase1_mode",
    "--phase-1-mode",
    "--excel-summary",
    "--batch-output",
    "--batch-excel",
    "--batch_excel",
    "--legacy-batch",
    "--legacy_batch",
)


def _reject_legacy_cli_flags(argv: List[str]) -> None:
    bad: List[str] = []
    for token in argv:
        lowered = token.lower().split("=", 1)[0]
        if lowered in _LEGACY_REJECTED_CLI_TOKENS:
            bad.append(token)
    if bad:
        sys.stderr.write(
            "error: the following legacy flags were removed in the Stage 1 / "
            "Stage 2 refactor and are no longer accepted: "
            f"{', '.join(bad)}\n"
            "Current pipeline performs per-note spectral analysis only; "
            "there is no preprocessing stage and no external energy mapping.\n"
        )
        sys.exit(2)


def _configure_file_logging(log_path: Path) -> None:
    """Ensure a single FileHandler writing to log_path."""
    root = logging.getLogger()
    log_path = log_path.resolve()
    for handler in list(root.handlers):
        if isinstance(handler, logging.FileHandler):
            try:
                existing_path = Path(handler.baseFilename).resolve()
            except Exception:
                existing_path = None
            if existing_path == log_path:
                return
            root.removeHandler(handler)
            try:
                handler.close()
            except Exception:
                pass
    file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


class RobustOrchestrator:
    """Three-stage orchestrator: per-note analysis, compilation, research export.

    Component energy ratios come from the current spectral analysis only.
    There is no preprocessing stage and no external energy mapping.
    Stage 3 recomputes EWSD-R v18 from per-note component spectra.
    """

    def __init__(
        self,
        audio_files: List[Path],
        main_analysis_output_dir: Path,
        super_analyzer_path: Optional[Path] = None,
        *,
        weight_function: str = "linear",
    ):
        """Initialise the orchestrator.

        Args:
            audio_files: List of audio files to process.
            main_analysis_output_dir: Output directory for per-note workbooks
                and the compiled density metrics workbook.
            super_analyzer_path: Optional path to ``super_audio_analyzer.py``.
                Accepted for backwards-compatible callers; not used at runtime.
            weight_function: Stage 2 weighting algorithm — one of
                ``"linear"`` (default), ``"log"``, or ``"power"``. Passed
                through to ``compile_density_metrics_with_pca`` and
                surfaced to ``proc_audio`` so each per-note
                ``Analysis_Metadata`` records which algorithm Stage 1
                expected Stage 2 to use.
        """
        self.audio_files: List[Path] = [Path(f) for f in audio_files]
        self.main_analysis_output_dir = Path(main_analysis_output_dir)
        self.super_analyzer_path = (
            Path(super_analyzer_path) if super_analyzer_path else None
        )
        # AUDIT FIX (Fgt_pp finding L2) — accept weight_function from the
        # caller so CLI / GUI invocations no longer hard-code "linear".
        wf_norm = str(weight_function or "linear").strip().lower()
        if wf_norm not in ("linear", "log", "power"):
            wf_norm = "linear"
        self.weight_function: str = wf_norm

        self.main_analysis_output_dir.mkdir(parents=True, exist_ok=True)

        if self.audio_files:
            log_dir = self.audio_files[0].parent
        else:
            log_dir = self.main_analysis_output_dir
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "orchestrator.log"
        _configure_file_logging(log_path)
        logger.info(f"Logging to: {log_path}")

        self.main_analysis_results: Dict[str, Any] = {}
        self.compiled_excel_path: Optional[Path] = None

        logger.info(
            "Orchestrator initialised with %d audio files (Stage 1 + Stage 2 + Stage 3 pipeline).",
            len(self.audio_files),
        )

    # ------------------------------------------------------------------
    # Note / tier helpers
    # ------------------------------------------------------------------
    def extract_note_from_filename(self, filename: str) -> Optional[str]:
        """Return the canonical note token in *filename*, or ``None``.

        Delegates to :func:`note_parser.parse_note_token` so the
        orchestrator, the Tk file-picker GUI and ``compile_metrics``
        share the same letter+accidental+mandatory-octave grammar.
        """
        from note_parser import parse_note_token

        return parse_note_token(filename)

    # ------------------------------------------------------------------
    # Back-compat stubs (no-op) for legacy GUI helpers.
    # These intentionally return failure / defaults so any caller that
    # still consults them silently degrades to the current-analysis
    # values from proc_audio. No runtime log strings, file lookups, or
    # external mapping reads are performed.
    # ------------------------------------------------------------------
    @staticmethod
    def _row_has_explicit_batch_ratio_column(row: Any) -> bool:
        return False

    @staticmethod
    def _explicit_batch_ratio_over_one(row: Any) -> bool:
        return False

    @staticmethod
    def _resolve_batch_energy_and_model_weights(
        payload: Dict[str, Any],
    ) -> Tuple[bool, Optional[str], Optional[float], Optional[float], str, Dict[str, Any]]:
        return False, "external_mapping_disabled", None, None, "fallback_default", {}

    def _assign_tier_for_file(
        self, audio_file: Path, note: Optional[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """Assign an FFT tier based on the file's fundamental frequency."""
        hz = 0.0
        if note:
            try:
                hz = librosa.note_to_hz(note)
            except Exception:
                pass

        if hz == 0:
            pat = re.compile(r"([A-G][#b]?-?\d+)")
            matches = pat.findall(audio_file.name)
            if matches:
                try:
                    hz = librosa.note_to_hz(matches[-1])
                except Exception:
                    pass

        if hz > 0:
            sorted_tiers = sorted(
                FFT_SETTINGS_BY_CLUSTER.items(),
                key=lambda x: (
                    x[1]['max_freq']
                    if x[1]['max_freq'] != float('inf')
                    else float('inf')
                ),
            )
            for tier_name, tier_cfg in sorted_tiers:
                if hz < tier_cfg['max_freq']:
                    return tier_name, tier_cfg
            return 'Tier_90', FFT_SETTINGS_BY_CLUSTER.get(
                'Tier_90',
                {'n_fft': 512, 'zp': 1, 'tolerance': 27.0, 'max_freq': float('inf')},
            )
        return 'Fallback', {
            'n_fft': 4096,
            'zp': 1,
            'tolerance': 10.0,
            'max_freq': 20000,
        }

    # ------------------------------------------------------------------
    # Stage 1 - per-note spectral analysis
    # ------------------------------------------------------------------
    def run_stage1_analysis(self) -> bool:
        """Run per-note spectral analysis for every audio file."""
        logger.info("=" * 80)
        logger.info("STAGE 1: Per-note spectral analysis (proc_audio current analysis)")
        logger.info("=" * 80)

        try:
            sys.path.insert(0, str(Path(__file__).parent))
            from proc_audio import AudioProcessor

            successful = 0
            failed = 0

            from note_parser import canonical_note_from_filename

            for audio_file in self.audio_files:
                try:
                    logger.info(f"Processing: {audio_file.name}")

                    note, note_source = canonical_note_from_filename(
                        audio_file.name,
                        parent_folder=audio_file.parent.name,
                    )
                    tier_name, tier_settings = self._assign_tier_for_file(
                        audio_file, note
                    )
                    n_fft = int(tier_settings['n_fft'])
                    zp = int(tier_settings['zp'])
                    hop_length = n_fft // 8
                    base_tolerance = float(tier_settings.get('tolerance', 5.0))

                    tier_max_freq = tier_settings.get('max_freq', 20000)
                    if tier_max_freq == float('inf'):
                        tier_max_freq = 20000
                    tier_list = sorted(FFT_SETTINGS_BY_CLUSTER.keys())
                    tier_idx = (
                        tier_list.index(tier_name) if tier_name in tier_list else 0
                    )
                    prev_max_freq = (
                        FFT_SETTINGS_BY_CLUSTER[tier_list[tier_idx - 1]].get(
                            'max_freq', 20
                        )
                        if tier_idx > 0
                        else 20
                    )
                    center_freq = (prev_max_freq + tier_max_freq) / 2.0
                    tolerance = max(base_tolerance, center_freq * 0.015)

                    logger.info(
                        f"  Tier assignment: {tier_name} (freq range: "
                        f"{prev_max_freq:.1f}-{tier_max_freq:.1f} Hz)"
                    )
                    logger.info(
                        f"  Tier parameters: n_fft={n_fft}, hop_length={hop_length}, "
                        f"zp={zp}, tolerance={tolerance:.2f} Hz"
                    )

                    processor = AudioProcessor()
                    processor.note_source = note_source
                    if note:
                        processor.note = note
                        logger.info(
                            "  Extracted note: %s (note_source=%s)",
                            note,
                            note_source,
                        )
                    else:
                        logger.info(
                            "  No explicit note token in filename "
                            "'%s' (note_source=%s)",
                            audio_file.name,
                            note_source,
                        )

                    adaptive_freq_min = 20.0
                    if note:
                        try:
                            f0 = processor.calculate_fundamental_frequency(note)
                            if 0 < f0 < 60.0:
                                adaptive_freq_min = max(10.0, f0 * 0.5)
                                logger.info(
                                    "  Low note detected (f0=%.2f Hz): adaptive "
                                    "freq_min=%.2f Hz",
                                    f0,
                                    adaptive_freq_min,
                                )
                        except Exception as exc:
                            logger.warning(
                                "  Could not compute f0 for note '%s': %s. "
                                "Using freq_min=20.0 Hz",
                                note,
                                exc,
                            )

                    processor.load_audio_files([str(audio_file)])

                    output_dir = (
                        self.main_analysis_output_dir / audio_file.stem
                    ).resolve()
                    output_dir.mkdir(parents=True, exist_ok=True)

                    # Component energy ratios are derived from the current
                    # analysis (auto_model_weights_from_analysis=True). The
                    # supplied harmonic/inharmonic weights are neutral
                    # placeholders that proc_audio overwrites once the
                    # spectrum has been classified.
                    processor.apply_filters_and_generate_data(
                        freq_min=adaptive_freq_min,
                        freq_max=20000.0,
                        db_min=-90.0,
                        db_max=0.0,
                        window='blackmanharris',
                        n_fft=n_fft,
                        hop_length=hop_length,
                        tolerance=tolerance,
                        use_adaptive_tolerance=True,
                        kaiser_beta=6.5,
                        gaussian_std=None,
                        zero_padding=zp,
                        time_avg='mean',
                        tier=tier_name,
                        harmonic_weight=0.5,
                        inharmonic_weight=0.5,
                        auto_model_weights_from_analysis=True,
                        weight_function=self.weight_function,
                        dissonance_enabled=True,
                        dissonance_model='Sethares',
                        dissonance_curve=True,
                        dissonance_scale=True,
                        compare_models=False,
                        spectral_masking_enabled=False,
                        results_directory=str(output_dir),
                        compile_per_call=False,
                    )

                    result: Dict[str, Any] = {
                        'file': str(audio_file),
                        'note': note,
                        'output_dir': str(output_dir),
                    }

                    try:
                        final_mh = float(
                            getattr(processor, 'model_harmonic_weight', 0.5)
                        )
                        final_mi = float(
                            getattr(processor, 'model_inharmonic_weight', 0.5)
                        )
                    except Exception:
                        final_mh, final_mi = 0.5, 0.5

                    logger.info(
                        "  Final model weights (source: current_analysis): "
                        "harmonic_weight=%.4f, inharmonic_weight=%.4f",
                        final_mh,
                        final_mi,
                    )

                    result['applied_weights'] = {
                        'harmonic': final_mh,
                        'inharmonic': final_mi,
                        'model_harmonic_weight': final_mh,
                        'model_inharmonic_weight': final_mi,
                        'model_weights_source': 'current_analysis',
                        'component_profile_source': 'current_analysis',
                        'model_weight_denominator': 'harmonic_plus_inharmonic',
                        'component_harmonic_energy_ratio': getattr(
                            processor, 'component_harmonic_energy_ratio', None
                        ),
                        'component_inharmonic_energy_ratio': getattr(
                            processor, 'component_inharmonic_energy_ratio', None
                        ),
                        'component_subbass_energy_ratio': getattr(
                            processor, 'component_subbass_energy_ratio', None
                        ),
                    }
                    self.main_analysis_results[audio_file.name] = result

                    successful += 1
                    logger.info(f"  [OK] Completed: {audio_file.name}")

                except Exception as exc:
                    logger.error(f"Error processing {audio_file.name}: {exc}")
                    logger.error(traceback.format_exc())
                    failed += 1

            logger.info(
                "Stage 1 complete: %d successful, %d failed", successful, failed
            )
            return successful > 0

        except Exception as exc:
            logger.error("Error in Stage 1 (per-note analysis): %s", exc)
            logger.error(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Stage 2 - compilation
    # ------------------------------------------------------------------
    def run_stage2_compilation(self) -> bool:
        """Compile per-note spectral_analysis.xlsx files into one workbook."""
        logger.info("=" * 80)
        logger.info("STAGE 2: Compilation (per-note spectral_analysis.xlsx)")
        logger.info("=" * 80)

        try:
            from proc_audio import log_runtime_paths as _log_runtime_paths
            _log_runtime_paths(logger)
        except Exception as exc_log:
            logger.debug("log_runtime_paths failed: %s", exc_log)

        try:
            from compile_metrics import compile_density_metrics_with_pca

            if not self.audio_files:
                logger.error("No audio files to compile")
                return False

            main_out = self.main_analysis_output_dir.resolve()
            main_out.mkdir(parents=True, exist_ok=True)

            def _count_metric_files(root: Path, needle: str) -> int:
                if not root.is_dir():
                    return 0
                nl = needle.lower()
                return sum(
                    1 for p in root.rglob("*")
                    if p.is_file() and nl in p.name.lower()
                )

            n_xlsx = _count_metric_files(main_out, "spectral_analysis.xlsx")
            if n_xlsx == 0:
                logger.error(
                    "Stage 2: no spectral_analysis.xlsx found under %s (recursive).",
                    main_out,
                )
                return False

            search_root = main_out
            file_pattern = "spectral_analysis.xlsx"
            logger.info(
                "Stage 2: using search_root=%s file_pattern=%r — matched %d file(s)",
                search_root,
                file_pattern,
                n_xlsx,
            )

            compiled_output_path = main_out / "compiled_density_metrics.xlsx"
            logger.info(
                "Stage 2: final compiled workbook path: %s", compiled_output_path
            )

            # Schema guard: per-note workbooks must match the current
            # ``single_pass_raw_export_v2`` analysis schema. Bubble up any
            # RuntimeError so the caller surfaces a clear stale-pipeline
            # message rather than silently compiling mismatched data.
            _schema_status = "not_validated_integrated"
            try:
                from compile_metrics import (
                    assert_results_dir_schema_or_raise as _assert_schema,
                    EXPECTED_ANALYSIS_SCHEMA_VERSION as _EXPECTED_VER,
                )

                summary = _assert_schema(search_root)
                logger.info(
                    "Schema guard: %d / %d per-note workbooks at %s pass "
                    "schema=%s.",
                    summary.get("valid", 0),
                    summary.get("total", 0),
                    search_root,
                    _EXPECTED_VER,
                )
                _schema_status = (
                    f"validated_{summary.get('valid', 0)}_of_{summary.get('total', 0)}"
                )
            except RuntimeError:
                raise
            except Exception as exc_guard:
                logger.debug("Schema guard skipped (%s)", exc_guard)
                _schema_status = "schema_guard_skipped"

            # Model weights for compile_metrics: take from the first run.
            # These come from the current per-note analysis.
            harmonic_weight = 1.0
            inharmonic_weight = 0.0
            if self.main_analysis_results:
                first_result = next(iter(self.main_analysis_results.values()))
                aw = first_result.get("applied_weights") or {}
                if "harmonic" in aw and "inharmonic" in aw:
                    harmonic_weight = float(aw["harmonic"])
                    inharmonic_weight = float(aw["inharmonic"])

            compiled_df = compile_density_metrics_with_pca(
                folder_path=search_root,
                output_path=str(compiled_output_path),
                file_pattern=file_pattern,
                include_pca=True,
                harmonic_weight=harmonic_weight,
                inharmonic_weight=inharmonic_weight,
                weight_function=self.weight_function,
                use_tsne=False,
                use_umap=False,
                detect_anomalies=False,
                anomaly_contamination=None,
                allow_legacy_super_json=False,
                compilation_extra_metadata={
                    "input_schema_validation_status": _schema_status,
                },
            )

            if compiled_output_path.is_file():
                legacy_clean = compiled_output_path.parent / (
                    f"{compiled_output_path.stem}_clean"
                    f"{compiled_output_path.suffix}"
                )
                if legacy_clean.is_file():
                    try:
                        legacy_clean.unlink()
                        logger.info(
                            "Removed legacy %s (single compiled workbook export).",
                            legacy_clean.name,
                        )
                    except OSError as lc_err:
                        logger.warning(
                            "Could not remove legacy compiled clean sidecar: %s",
                            lc_err,
                        )

            from post_compile_research_export import run_research_workbook_export

            if compiled_df is not None and not compiled_df.empty:
                self.compiled_excel_path = compiled_output_path
                logger.info(
                    "Stage 2: workbook written OK (rows=%d, cols=%d)",
                    len(compiled_df),
                    len(compiled_df.columns),
                )
                logger.info("Compiled metrics Excel: %s", compiled_output_path)
                logger.info(
                    "[OK] compiled_density_metrics.xlsx created: %s",
                    compiled_output_path,
                )
                run_research_workbook_export(compiled_output_path, log=logger)
                return True

            if compiled_output_path.is_file():
                self.compiled_excel_path = compiled_output_path
                logger.warning(
                    "Stage 2: compiler returned empty DataFrame but output file "
                    "exists (diagnostic export?): %s",
                    compiled_output_path,
                )
                logger.info(
                    "[OK] compiled_density_metrics.xlsx created: %s",
                    compiled_output_path,
                )
                run_research_workbook_export(compiled_output_path, log=logger)
                return True

            logger.error(
                "Stage 2: compilation returned empty DataFrame and no workbook at %s",
                compiled_output_path,
            )
            return False

        except Exception as exc:
            logger.error("Error in Stage 2 (compilation): %s", exc)
            logger.error(traceback.format_exc())
            return False

    # ------------------------------------------------------------------
    # Complete pipeline
    # ------------------------------------------------------------------
    def run_complete_pipeline(self) -> Dict[str, Any]:
        logger.info("=" * 80)
        logger.info("PIPELINE ORCHESTRATOR - COMPLETE PIPELINE (Stage 1 + Stage 2 + Stage 3)")
        logger.info("=" * 80)
        logger.info(f"Start time: {datetime.now()}")

        pipeline_results: Dict[str, Any] = {
            'start_time': datetime.now().isoformat(),
            'audio_files_count': len(self.audio_files),
            'stages': {},
        }

        stage1_success = self.run_stage1_analysis()
        pipeline_results['stages']['stage1_analysis'] = {
            'success': stage1_success,
            'results_count': len(self.main_analysis_results),
            'component_profile_source': 'current_analysis',
            'model_weights_source': 'current_analysis',
        }

        if not stage1_success:
            logger.error("Pipeline failed at Stage 1 (per-note analysis)")
            pipeline_results['status'] = 'failed'
            pipeline_results['end_time'] = datetime.now().isoformat()
            return pipeline_results

        stage2_success = self.run_stage2_compilation()
        pipeline_results['stages']['stage2_compilation'] = {
            'success': stage2_success,
            'compiled_workbook': (
                str(self.compiled_excel_path)
                if self.compiled_excel_path
                else None
            ),
        }

        pipeline_results['status'] = 'success' if stage2_success else 'failed'
        pipeline_results['end_time'] = datetime.now().isoformat()

        # Persist a tiny pipeline summary alongside the compiled workbook.
        results_path = self.main_analysis_output_dir / "orchestrator_results.json"
        try:
            from metadata_sanitizer import (
                publication_redaction_enabled,
                sanitize_metadata_dict,
            )

            if publication_redaction_enabled():
                pipeline_results = sanitize_metadata_dict(
                    json.loads(json.dumps(pipeline_results, default=str))
                )
        except Exception:
            pass
        try:
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(pipeline_results, f, indent=2, default=str)
        except Exception as exc_save:
            logger.warning("Could not write orchestrator results JSON: %s", exc_save)

        logger.info("=" * 80)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 80)
        logger.info(f"Results saved to: {results_path}")
        if self.compiled_excel_path:
            logger.info(f"Final compiled Excel: {self.compiled_excel_path}")

        return pipeline_results


def main() -> int:
    _reject_legacy_cli_flags(sys.argv[1:])

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Pipeline orchestrator — Stage 1 (per-note spectral analysis) followed "
            "by Stage 2 (compilation). Component energy ratios are computed "
            "from the current per-note analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python pipeline_orchestrator_integrated.py\n"
            "  python pipeline_orchestrator_integrated.py --audio-dir "
            "\"C:\\path\\to\\audio\"\n"
            "  python pipeline_orchestrator_integrated.py file1.wav file2.wav\n"
        ),
    )
    parser.add_argument(
        'audio_files',
        nargs='*',
        help='Audio files to process (optional if --audio-dir is provided)',
    )
    parser.add_argument(
        '--audio-dir',
        type=str,
        help='Directory containing audio files to process',
    )
    parser.add_argument(
        '--main-output',
        type=str,
        default='main_analysis_results',
        help='Output directory for per-note analysis and compiled workbook',
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.wav',
        help='File pattern to match (default: *.wav)',
    )
    parser.add_argument(
        '--weight-function',
        type=str,
        default='linear',
        choices=('linear', 'log', 'power'),
        help=(
            'Stage 2 weighting algorithm: linear / log / power. Default: linear.'
        ),
    )
    parser.add_argument(
        '--gui',
        action='store_true',
        help=(
            'Launch the standalone Tk file-picker GUI '
            '(pipeline_orchestrator_gui.py).'
        ),
    )

    args = parser.parse_args()

    if args.gui:
        gui_path = Path(__file__).parent / "pipeline_orchestrator_gui.py"
        if gui_path.exists():
            print("Launching standalone Tk file-picker GUI...")
            subprocess.run([sys.executable, str(gui_path)])
            return 0
        print(f"Error: GUI helper not found at {gui_path}")
        return 1

    audio_files: List[Path] = []
    if args.audio_files:
        for f in args.audio_files:
            p = Path(f)
            if p.exists():
                audio_files.append(p)
            else:
                print(f"Warning: File not found: {f}")

    if args.audio_dir:
        audio_dir = Path(args.audio_dir)
        if audio_dir.exists() and audio_dir.is_dir():
            audio_files.extend(list(audio_dir.glob(args.pattern)))
        else:
            print(f"Error: Directory not found: {audio_dir}")
            return 1

    if not audio_files:
        current_dir = Path.cwd()
        audio_files = list(current_dir.glob(args.pattern))
        if audio_files:
            print(f"Found {len(audio_files)} audio files in current directory")
        else:
            print("No audio files found in current directory.")
            print("Opening file selection dialog...")
            try:
                import tkinter as tk
                from tkinter import filedialog, messagebox

                root = tk.Tk()
                root.withdraw()
                root.attributes('-topmost', True)

                response = messagebox.askyesnocancel(
                    "Select Audio Files",
                    "How would you like to select files?\n\n"
                    "YES = Select a directory (process all audio files in it)\n"
                    "NO = Select individual files\n"
                    "CANCEL = Exit",
                )

                if response is None:
                    print("File selection cancelled.")
                    root.destroy()
                    return 1
                if response:
                    selected_path = filedialog.askdirectory(
                        title="Select Directory with Audio Files",
                        initialdir=str(current_dir),
                    )
                    if selected_path:
                        audio_dir = Path(selected_path)
                        audio_files = list(audio_dir.glob(args.pattern))
                        if audio_files:
                            print(
                                f"Found {len(audio_files)} audio files in: "
                                f"{selected_path}"
                            )
                        else:
                            print(
                                f"No {args.pattern} files found in selected "
                                "directory"
                            )
                            root.destroy()
                            return 1
                    else:
                        print("No directory selected.")
                        root.destroy()
                        return 1
                else:
                    selected_files = filedialog.askopenfilenames(
                        title="Select Audio Files",
                        filetypes=[
                            ("Audio files", "*.wav *.mp3 *.flac *.aif *.aiff"),
                            ("WAV files", "*.wav"),
                            ("All files", "*.*"),
                        ],
                        initialdir=str(current_dir),
                    )
                    if selected_files:
                        audio_files = [Path(f) for f in selected_files]
                        print(f"Selected {len(audio_files)} audio file(s)")
                    else:
                        print("No files selected.")
                        root.destroy()
                        return 1
                root.destroy()
            except ImportError:
                print("Tk not available; specify --audio-dir or files.")
                parser.print_help()
                return 1
            except Exception as exc:
                print(f"Error showing file picker: {exc}")
                parser.print_help()
                return 1

    if not audio_files:
        print("Error: No audio files to process")
        return 1

    print(f"Processing {len(audio_files)} audio file(s)...")
    print(
        f"Files: {[f.name for f in audio_files[:5]]}"
        f"{'...' if len(audio_files) > 5 else ''}"
    )

    try:
        orchestrator = RobustOrchestrator(
            audio_files=audio_files,
            main_analysis_output_dir=Path(args.main_output),
            weight_function=args.weight_function,
        )
        results = orchestrator.run_complete_pipeline()

        print("\n" + "=" * 80)
        print("PIPELINE SUMMARY")
        print("=" * 80)
        print(f"Status: {results['status']}")
        print(f"Audio Files: {results['audio_files_count']}")
        for stage_name, stage_result in results['stages'].items():
            status = "OK" if stage_result.get('success') else "FAIL"
            print(f"{stage_name}: {status}")
            if 'results_count' in stage_result:
                print(f"  -> Results: {stage_result['results_count']}")
            if stage_result.get('compiled_workbook'):
                print(f"  -> Compiled workbook: {stage_result['compiled_workbook']}")
        print("=" * 80)

        return 0 if results['status'] == 'success' else 1

    except Exception as exc:
        print(f"Error: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
