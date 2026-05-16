"""Regression: forbidden legacy Stage 1 / Batch tokens must NOT appear in
runtime code paths, GUI strings, runtime logs, exported metadata, or compiled
workbooks.

Rationale
=========
The pipeline was refactored from the old three-phase architecture
(Phase 1 / Batch -> Phase 2 / Analysis -> Phase 3 / Compilation) into a
two-stage architecture:

    Stage 1: Per-note spectral analysis
    Stage 2: Compilation

The user-visible vocabulary must therefore not contain any of the obsolete
"Phase 1", "Batch", "Synthetic mapping", or `batch_*` metric tokens.

Allowed occurrences
-------------------
* Historical comments / docstrings (we cannot guarantee absolutely zero - the
  rule below targets runtime strings, log emissions, GUI labels, metadata
  keys and exported workbook columns).
* Explicitly deprecated tests (this file and any other file under
  ``tests/`` whose name contains ``legacy`` or ``deprecated``).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Forbidden tokens
# ---------------------------------------------------------------------------
# These tokens must not appear in any runtime emission. We check:
#   * runtime python sources for *string literals* that include these tokens
#     (so we ignore comments, but catch logger.info("Phase 1 ..."), etc.)
#   * GUI labels (interface.py) for literal occurrences
#   * compiled output column metadata produced by publication_metric_columns

FORBIDDEN_RUNTIME_TOKENS: tuple[str, ...] = (
    "Phase 1",
    "Phase 2",
    "Phase 3",
    "Light Phase 1",
    "Skip Phase 1",
    "Phase 1 skipped",
    "Synthetic mapping",
    "global H/I from GUI",
    "Applying percentages",
    "batch_results",
    "batch_summary",
    "batch_summary.xlsx",
    "legacy_batch",
    "batch_excel",
    "batch_harmonic_energy_ratio",
    "batch_inharmonic_energy_ratio",
    "batch_subbass_energy_ratio",
    "full batch pipeline",
    "full batch",
    "GUI \u03b1/\u03b2",  # "GUI α/β"
    "Model weights \u03b1/\u03b2",  # "Model weights α/β"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]

# Files that participate in the active runtime path (logs, GUI labels,
# exported metadata, compiled workbooks). We intentionally exclude legacy
# tests, backup folders, the third-party virtualenv, and the deprecated
# Legacy ``audio_analysis`` batch scripts (documented as optional Phase 1).
_RUNTIME_SOURCES: tuple[Path, ...] = tuple(
    _REPO_ROOT / name
    for name in (
        "proc_audio.py",
        "compile_metrics.py",
        "run_orchestrator.py",
        "pipeline_orchestrator_integrated.py",
        "pipeline_orchestrator_gui.py",
        "interface.py",
        "acoustic_data_analysis_suite.py",
        "gui_model_weight_policy.py",
        "main.py",
        "publication_metric_columns.py",
        "metadata_sanitizer.py",
        "constants.py",
        "density.py",
    )
)


# Single-line string literals only (cannot span newlines). Triple-quoted
# strings are handled separately below.
_DOUBLE_QUOTED_STRING = re.compile(r'"(?:\\.|[^"\\\n])*"')
_SINGLE_QUOTED_STRING = re.compile(r"'(?:\\.|[^'\\\n])*'")
_TRIPLE_DOUBLE_DOCSTRING = re.compile(r'"""[\s\S]*?"""')
_TRIPLE_SINGLE_DOCSTRING = re.compile(r"'''[\s\S]*?'''")


def _extract_string_literals(source: str) -> list[str]:
    """Return every single-line string literal in ``source`` *excluding*
    triple-quoted strings/docstrings and line comments.

    We strip triple-quoted strings and ``# ...`` comments first because the
    rule explicitly allows historical comments / docstrings. We then collect
    the remaining "..."/'...' single-line string literals (the things that
    actually become log messages, GUI labels and metadata keys at runtime).
    """

    without_triples = _TRIPLE_DOUBLE_DOCSTRING.sub("", source)
    without_triples = _TRIPLE_SINGLE_DOCSTRING.sub("", without_triples)
    code_only_lines: list[str] = []
    for line in without_triples.splitlines():
        # Drop everything after the first ``#`` that is not inside a string.
        # We are not building a full Python tokeniser; a conservative split is
        # enough because the forbidden tokens we look for do not contain
        # ``#`` characters.
        stripped = line.split("#", 1)[0]
        code_only_lines.append(stripped)
    literals: list[str] = []
    for line in code_only_lines:
        literals.extend(_DOUBLE_QUOTED_STRING.findall(line))
        literals.extend(_SINGLE_QUOTED_STRING.findall(line))
    return literals


def _literal_body(literal: str) -> str:
    """Return the *body* of a single-line string literal (without quotes)."""

    if len(literal) >= 2 and literal[0] == literal[-1] and literal[0] in ('"', "'"):
        return literal[1:-1]
    return literal


def _is_legacy_cli_flag_literal(literal: str) -> bool:
    """``--phase1-mode`` style strings live in the *rejection* lists used by
    ``_reject_legacy_cli_flags``. They are present in the source precisely so
    that the orchestrator can hard-error if a user passes them on the CLI.
    They must therefore be exempt from the runtime-token regression."""

    body = _literal_body(literal).strip()
    return body.startswith("--")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", _RUNTIME_SOURCES, ids=lambda p: p.name)
def test_runtime_python_sources_have_no_forbidden_string_literals(path: Path) -> None:
    """Every runtime source file must not emit any forbidden token via a
    string literal (log message, GUI label, metadata key, column name)."""

    if not path.is_file():
        pytest.skip(f"runtime source not present: {path.name}")
    source = path.read_text(encoding="utf-8", errors="replace")
    literals = _extract_string_literals(source)
    offenders: list[tuple[str, str]] = []
    for literal in literals:
        if _is_legacy_cli_flag_literal(literal):
            # CLI flags such as "--phase1-mode", "--batch_excel" or
            # "--legacy_batch" are *rejected* by the orchestrator's argument
            # parser; their presence in the source is the feature itself.
            continue
        for token in FORBIDDEN_RUNTIME_TOKENS:
            if token in literal:
                offenders.append((token, literal))
    assert not offenders, (
        f"{path.name} contains forbidden tokens in runtime string literals:\n"
        + "\n".join(f"  [{tok}] -> {lit}" for tok, lit in offenders)
    )


def test_compiled_publication_columns_have_no_batch_metric_columns() -> None:
    """The publication column allow-list defines which canonical metrics may
    surface in compiled workbooks. It must not advertise any ``batch_*``
    energy ratio column."""

    from publication_metric_columns import (
        COMPILED_METRICS_PUBLICATION_COLUMN_ALLOWLIST,
    )

    banned = (
        "batch_harmonic_energy_ratio",
        "batch_inharmonic_energy_ratio",
        "batch_subbass_energy_ratio",
        "batch_total_inharmonic_energy_ratio",
    )
    leaks = [
        c for c in COMPILED_METRICS_PUBLICATION_COLUMN_ALLOWLIST if c in banned
    ]
    assert not leaks, (
        "publication_metric_columns advertises deprecated batch_* columns: "
        f"{leaks}"
    )


def test_gui_model_weight_policy_uses_current_analysis_default() -> None:
    """The GUI policy must default to ``current_analysis`` and must not
    return the legacy 0.95 / 0.05 harmonic/inharmonic placeholder pair."""

    from gui_model_weight_policy import resolve_analysis_model_weights

    h, i, meta = resolve_analysis_model_weights(
        manual_override=False,
        slider_harmonic_fraction=0.5,
    )
    assert meta.get("model_weights_source") == "current_analysis"
    assert meta.get("external_component_profile_used") is False
    assert meta.get("external_h_i_s_mapping_used") is False
    assert (h, i) != (0.95, 0.05)


def test_run_orchestrator_rejects_legacy_cli_flags() -> None:
    """Old CLI flags must hard-error before any analysis happens."""

    from run_orchestrator import _reject_legacy_cli_flags

    legacy_flags = (
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
    for flag in legacy_flags:
        with pytest.raises(SystemExit):
            _reject_legacy_cli_flags([flag])


def test_robust_orchestrator_has_stage_methods_not_phase_methods() -> None:
    """The orchestrator class must expose ``run_stage1_analysis`` /
    ``run_stage2_compilation`` and must NOT expose the old phase methods."""

    from pipeline_orchestrator_integrated import RobustOrchestrator

    assert hasattr(RobustOrchestrator, "run_stage1_analysis")
    assert hasattr(RobustOrchestrator, "run_stage2_compilation")
    assert not hasattr(RobustOrchestrator, "run_preprocessing_phase")
    assert not hasattr(RobustOrchestrator, "load_percentage_mapping")
    assert not hasattr(RobustOrchestrator, "apply_percentages_to_analysis")


# ---------------------------------------------------------------------------
# Tk GUI regression: source-level + runtime widget walk
# ---------------------------------------------------------------------------


# Strings the user explicitly listed as visible-UI offenders that the
# previous Batch / Phase 1 refactor missed in the Tk file-picker GUI.
# A subset (the "Phase N" / "batch_summary*" tokens) are already covered
# by ``FORBIDDEN_RUNTIME_TOKENS``; the GUI-specific test below also
# accepts shorter substrings (``Batch``, ``batch_summary``) that would
# otherwise be too aggressive for the generic runtime-source scan.
_GUI_FORBIDDEN_LITERAL_SUBSTRINGS: tuple[str, ...] = (
    "Light Phase 1",
    "Skip Phase 1",
    "batch_summary.xlsx",
    "batch_summary",
    "GUI \u03b1/\u03b2",
    "full batch pipeline",
    "full batch",
    "Phase 1",
    "Phase 2",
    "Phase 3",
)


def test_pipeline_orchestrator_gui_source_has_no_legacy_ui_strings() -> None:
    """The standalone Tk GUI source must not contain any of the visible
    obsolete UI strings reported during the Stage 1 / Stage 2 cleanup."""

    path = _REPO_ROOT / "pipeline_orchestrator_gui.py"
    if not path.is_file():
        pytest.skip(f"GUI source not present: {path.name}")
    source = path.read_text(encoding="utf-8", errors="replace")
    literals = _extract_string_literals(source)
    offenders: list[tuple[str, str]] = []
    for literal in literals:
        if _is_legacy_cli_flag_literal(literal):
            continue
        for token in _GUI_FORBIDDEN_LITERAL_SUBSTRINGS:
            if token in literal:
                offenders.append((token, literal))
    assert not offenders, (
        f"{path.name} contains legacy GUI strings:\n"
        + "\n".join(f"  [{tok}] -> {lit}" for tok, lit in offenders)
    )


# Strings whose presence on a live widget label/text would prove the
# obsolete checkboxes / explanatory label still ship in the GUI. We do
# NOT include "Phase 2"/"Phase 3" here because the request only calls
# out the live widget-level offenders explicitly: "Light Phase 1",
# "Skip Phase 1", "batch_summary", "Batch", "Phase 1".
_GUI_LIVE_FORBIDDEN_SUBSTRINGS: tuple[str, ...] = (
    "Light Phase 1",
    "Skip Phase 1",
    "batch_summary",
    "Batch",
    "Phase 1",
)


def _collect_tk_widget_text(widget: object) -> list[str]:
    """Return the rendered text of ``widget`` and every descendant.

    Tk widgets expose user-visible strings via the ``text`` configuration
    option (Buttons, Labels, Checkbuttons, LabelFrames, Notebook tabs,
    …). We also collect the window title from the root toplevel.
    """

    texts: list[str] = []
    try:
        cfg = widget.cget("text")  # type: ignore[attr-defined]
    except Exception:
        cfg = None
    if isinstance(cfg, str) and cfg:
        texts.append(cfg)

    # Notebook tab labels
    try:
        import tkinter.ttk as _ttk

        if isinstance(widget, _ttk.Notebook):
            for tab_id in widget.tabs():
                tab_text = widget.tab(tab_id, "text")
                if isinstance(tab_text, str) and tab_text:
                    texts.append(tab_text)
    except Exception:
        pass

    try:
        children = widget.winfo_children()  # type: ignore[attr-defined]
    except Exception:
        children = []
    for child in children:
        texts.extend(_collect_tk_widget_text(child))
    return texts


def test_pipeline_orchestrator_gui_live_widget_text_has_no_legacy_strings() -> None:
    """Launch the Tk app headlessly and confirm that no widget label,
    Notebook tab, or window title contains any forbidden Batch /
    Phase 1 string."""

    tk = pytest.importorskip("tkinter")

    try:
        import pipeline_orchestrator_gui as gui_module
    except Exception as exc:  # pragma: no cover - import-time failure
        pytest.skip(f"GUI module could not be imported: {exc}")

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No DISPLAY available for Tk widget smoke test.")

    try:
        app = gui_module.RobustOrchestratorApp(root)
        title = root.title()
        widget_texts = _collect_tk_widget_text(root)
        all_texts = [title, *widget_texts]
    finally:
        try:
            root.destroy()
        except Exception:
            pass

    offenders: list[tuple[str, str]] = []
    for txt in all_texts:
        for token in _GUI_LIVE_FORBIDDEN_SUBSTRINGS:
            if token in txt:
                offenders.append((token, txt))

    assert not offenders, (
        "Live Tk widget tree contains forbidden Batch / Phase 1 strings:\n"
        + "\n".join(f"  [{tok}] -> {lit!r}" for tok, lit in offenders)
    )

    # Sanity: the canonical Stage 1 / Stage 2 wording must be present
    # somewhere in the visible UI; otherwise the title or labels were
    # mangled.
    joined = "\n".join(all_texts)
    assert "Stage 1" in joined, (
        "Live Tk widget tree should mention 'Stage 1' (per-note "
        f"spectral analysis); got:\n{joined}"
    )
    assert "Stage 2" in joined, (
        "Live Tk widget tree should mention 'Stage 2' (compilation); "
        f"got:\n{joined}"
    )
    # Window title must be the new SoundSpectrAnalyse title.
    assert title.startswith("SoundSpectrAnalyse"), (
        f"Unexpected window title: {title!r}"
    )

    # Keep ``app`` reachable so the variable is not flagged as unused
    # by linters; it is fully owned by the destroyed root above.
    del app
