#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoundSpectrAnalyse - Pipeline Entry Point
=========================================

Pipeline:
    Stage 1: Per-note spectral analysis (proc_audio.AudioProcessor)
    Stage 2: Compilation (per-note spectral_analysis.xlsx -> compiled workbook)
    Stage 3: Research export + EWSD-R v18 merge (compiled_density_metrics_research.xlsx)

Usage:
    python run_orchestrator.py                          # all .wav in cwd
    python run_orchestrator.py --audio-dir PATH         # all .wav in PATH
    python run_orchestrator.py file1.wav file2.wav ...  # specific files

There is no Phase 1, no Batch preprocessing, no batch_summary.xlsx, and no
synthetic harmonic/inharmonic percentages. Component energy ratios come only
from the per-note current spectral analysis.
"""

import sys
import argparse
from pathlib import Path

from pipeline_orchestrator_integrated import RobustOrchestrator

# Legacy CLI flags removed in the Stage 1 / Stage 2 refactor. Passing any of
# them must hard-error so old shell invocations fail loudly instead of
# silently reactivating the obsolete batch pipeline.
_LEGACY_REJECTED_TOKENS = (
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


def _reject_legacy_cli_flags(argv: list[str]) -> None:
    """Hard-error if any deprecated batch/phase1 flag is passed."""
    bad: list[str] = []
    for token in argv:
        lowered = token.lower().split("=", 1)[0]
        if lowered in _LEGACY_REJECTED_TOKENS:
            bad.append(token)
    if bad:
        sys.stderr.write(
            "error: the following legacy flags were removed in the Stage 1 / "
            "Stage 2 refactor and are no longer accepted: "
            f"{', '.join(bad)}\n"
            "The current pipeline performs per-note spectral analysis only; "
            "there is no Batch preprocessing.\n"
        )
        sys.exit(2)


def main() -> int:
    _reject_legacy_cli_flags(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description=(
            "SoundSpectrAnalyse pipeline - Stage 1 (per-note spectral "
            "analysis), Stage 2 (compilation), Stage 3 (research export + EWSD)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python run_orchestrator.py\n"
            "  python run_orchestrator.py --audio-dir \"C:\\path\\to\\audio\"\n"
            "  python run_orchestrator.py file1.wav file2.wav file3.wav\n"
        ),
    )
    parser.add_argument(
        "audio_files",
        nargs="*",
        help="Audio files to process (optional if --audio-dir is provided)",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        help="Directory containing audio files to process",
    )
    parser.add_argument(
        "--main-output",
        type=str,
        default="main_analysis_results",
        help="Output directory for per-note analysis and compiled workbook",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.wav",
        help="File pattern to match (default: *.wav)",
    )
    parser.add_argument(
        "--weight-function",
        type=str,
        default="linear",
        choices=("linear", "log", "power"),
        help=(
            "Weighting algorithm used by Stage 2 to fold each component "
            "sheet's per-row amplitudes into a band-level density "
            "(linear: SUM(Amplitude_raw); log: LOG10(1 + SUM(Amplitude_raw)); "
            "power: SUM(Power_raw) or SUM(Amplitude_raw**2)). "
            "Default: linear."
        ),
    )

    args = parser.parse_args()

    audio_files: list[Path] = []
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
            print("Error: No audio files found.")
            print("Please specify files or use --audio-dir option")
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
    print(
        "Pipeline: Stage 1 (per-note spectral analysis) -> "
        "Stage 2 (compilation) -> Stage 3 (research export + EWSD). "
        "Component energy ratios are computed from the current analysis."
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

        for stage_name, stage_result in results["stages"].items():
            status = "OK" if stage_result.get("success") else "FAIL"
            print(f"{stage_name}: {status}")
            if "results_count" in stage_result:
                print(f"  -> Results: {stage_result['results_count']}")
            if "compiled_workbook" in stage_result and stage_result["compiled_workbook"]:
                print(f"  -> Compiled workbook: {stage_result['compiled_workbook']}")

        print("=" * 80)

        return 0 if results["status"] == "success" else 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
