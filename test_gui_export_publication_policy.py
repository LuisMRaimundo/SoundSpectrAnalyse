"""Audit tests for the GUI/export publication policy (single-pass refactor).

The tests cover the audit points A–F:

A. **GUI / default chart source** — the default plotting sheet must be
   ``Canonical_Metrics``; the default metric must come from the audit-
   approved preference list (``density_normalized_global`` etc.).

B. **Raw sums are not default** — ``Harmonic Partials sum``,
   ``Inharmonic Partials sum``, ``Total sum`` (and the rest of the audit
   ban list) are *never* selected by default, even when they are the only
   numeric columns available.

C. **Canonical coverage** — the compiled workbook's ``Canonical_Metrics``
   sheet must include every metric declared ``status="canonical"`` in
   ``metrics_dictionary.json`` (modulo identifier fields that the
   compiler emits via ``Note`` / ``Folder``).

D. ``component_*`` promotion — when only ``Analysis_Metadata`` carries
   the ``component_*_energy_ratio`` fields, the compile step must
   harvest them into the wide compiled row so they reach
   ``Canonical_Metrics``.

E. **Provenance** — when ``auto_model_weights_from_analysis=True`` and
   the single-pass helper overwrites the placeholder weights,
   ``model_weights_source`` must be ``current_analysis`` (and
   ``component_profile_source`` must be ``integrated_single_pass``).

F. **Diagnostic / legacy warning** — every chart whose metric is not
   canonical receives an explicit warning in its title; the helper
   ``metric_requires_warning`` returns a non-empty text for legacy /
   diagnostic metrics and ``None`` for canonical ones.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import publication_chart_policy as pcp
import compile_metrics as cm


# ---------------------------------------------------------------------------
# A — default chart source: Canonical_Metrics
# ---------------------------------------------------------------------------
def test_default_publication_sheet_is_canonical_metrics() -> None:
    assert pcp.DEFAULT_PUBLICATION_SHEET == "Canonical_Metrics"


def test_default_publication_metric_is_density_normalized_global() -> None:
    # The audit mandates ``density_normalized_global`` as the first choice
    # because it is the only metric with a corpus-wide [0, 1] normalisation.
    assert pcp.DEFAULT_PUBLICATION_METRIC == "density_normalized_global"


def test_default_publication_metric_preference_order_matches_audit() -> None:
    expected = (
        "density_normalized_global",
        "canonical_density_v5_adapted",
        "effective_partial_density",
        "model_harmonic_weight",
        "model_inharmonic_weight",
        "spectral_entropy",
    )
    assert tuple(pcp.DEFAULT_PUBLICATION_METRIC_PREFERENCE) == expected


def test_default_publication_metric_selected_when_canonical_present() -> None:
    cols = [
        "Note",
        "canonical_density_v5_adapted",
        "density_normalized_global",
        "Harmonic Partials sum",
    ]
    chosen = pcp.select_default_publication_metric(cols)
    assert chosen == "density_normalized_global"


def test_default_publication_metric_falls_back_through_preference_order() -> None:
    cols = ["Note", "effective_partial_density", "Harmonic Partials sum"]
    chosen = pcp.select_default_publication_metric(cols)
    assert chosen == "effective_partial_density"


# ---------------------------------------------------------------------------
# B — raw sums are NEVER selected as defaults
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name",
    [
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Sub-bass sum",
        "Total sum",
        "harmonic_density",
        "inharmonic_density",
        "combined_density",
        "Density Metric",
        "Spectral Density Metric",
        "Combined Density Metric",
        "Filtered Density Metric",
    ],
)
def test_forbidden_default_metrics_classified_as_legacy(name: str) -> None:
    assert pcp.classify_metric_for_publication(name) == "legacy"


@pytest.mark.parametrize(
    "name",
    [
        "linear_sum_amplitude_harmonic",
        "linear_sum_amplitude_inharmonic_partial",
        "linear_sum_amplitude_subbass_band",
        "batch_harmonic_energy_ratio",
        "batch_inharmonic_energy_ratio",
        "legacy_bounded_harmonic_weight",
    ],
)
def test_forbidden_default_metric_prefixes_classified_as_legacy(name: str) -> None:
    assert pcp.classify_metric_for_publication(name) == "legacy"


def test_select_default_publication_metric_never_returns_forbidden_column() -> None:
    cols = [
        "Note",
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Total sum",
        "Density Metric",
    ]
    chosen = pcp.select_default_publication_metric(cols)
    assert chosen is None  # no canonical column → refuse, do not fall back


def test_filter_out_forbidden_default_columns_excludes_legacy() -> None:
    cols = [
        "density_normalized_global",
        "Harmonic Partials sum",
        "Total sum",
        "spectral_entropy",
        "batch_harmonic_energy_ratio",
    ]
    kept = pcp.filter_out_forbidden_default_columns(cols)
    assert "Harmonic Partials sum" not in kept
    assert "Total sum" not in kept
    assert "batch_harmonic_energy_ratio" not in kept
    assert "density_normalized_global" in kept
    assert "spectral_entropy" in kept


# ---------------------------------------------------------------------------
# C — canonical coverage contract
# ---------------------------------------------------------------------------
def _load_canonical_names_from_dictionary() -> list[str]:
    path = REPO_ROOT / "metrics_dictionary.json"
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    out = []
    for entry in data.get("metrics", []):
        if entry.get("status") == "canonical":
            out.append(str(entry["name"]))
    return out


def test_canonical_metric_columns_cover_dictionary_canonical_status() -> None:
    """Every ``status="canonical"`` name in the dictionary must appear in
    ``compile_metrics.CANONICAL_METRIC_COLUMNS`` (identifier names included)."""
    dict_canonical = set(_load_canonical_names_from_dictionary())
    declared = set(cm.CANONICAL_METRIC_COLUMNS)
    missing = sorted(dict_canonical - declared)
    assert not missing, (
        f"CANONICAL_METRIC_COLUMNS is missing dictionary-canonical entries: {missing}"
    )


def test_compiled_canonical_sheet_includes_audit_required_metrics(tmp_path: Path) -> None:
    """Synthesize a tiny corpus DataFrame containing every audit-required
    canonical column and confirm that ``_slice_compiled_df_by_status`` keeps
    them all in the ``Canonical_Metrics`` slice.
    """
    required = [
        "component_harmonic_energy_ratio",
        "component_inharmonic_energy_ratio",
        "component_subbass_energy_ratio",
        "component_total_inharmonic_energy_ratio",
        "harmonic_completeness",
        "harmonic_inharmonic_ratio",
        "harmonic_effective_power_density",
        "rolloff_compensated_harmonic_density",
        "density_normalized_global",
    ]
    data = {"Note": ["C4", "E4"]}
    for c in required:
        data[c] = [0.5, 0.25]
    # Throw in some legacy noise; the slice must drop it.
    data["Harmonic Partials sum"] = [12345.6, 7890.1]
    data["batch_harmonic_energy_ratio"] = [0.5, 0.25]
    df = pd.DataFrame(data)
    canon = cm._slice_compiled_df_by_status(df, "canonical")
    for c in required:
        assert c in canon.columns, f"{c} missing from Canonical_Metrics slice"
    assert "Harmonic Partials sum" not in canon.columns
    assert "batch_harmonic_energy_ratio" not in canon.columns


# ---------------------------------------------------------------------------
# D — component_* promotion from Analysis_Metadata
# ---------------------------------------------------------------------------
def _write_synthetic_per_note_workbook(out_path: Path) -> None:
    """Mimic a per-note ``spectral_analysis.xlsx`` where the canonical
    ``component_*`` ratios live in ``Analysis_Metadata`` only (the
    historical layout). The audit fix must harvest them into the wide
    compiled row regardless.
    """
    metrics_sheet = pd.DataFrame([{
        "Note": "C4",
        "weight_function": "linear",
        "canonical_density_v5_adapted": 0.42,
        "effective_partial_density": 5.4,
        "spectral_entropy": 0.91,
        "harmonic_energy_ratio": 0.7,
        "inharmonic_energy_ratio": 0.2,
        "subbass_energy_ratio": 0.1,
    }])
    am_rows = [
        ("component_harmonic_energy_ratio", 0.72),
        ("component_inharmonic_energy_ratio", 0.18),
        ("component_subbass_energy_ratio", 0.10),
        ("component_total_inharmonic_energy_ratio", 0.28),
        ("component_energy_method", "single_pass_proc_audio_energy"),
        ("component_profile_source", "integrated_single_pass"),
        ("model_harmonic_weight", 0.80),
        ("model_inharmonic_weight", 0.20),
        ("model_weights_source", "current_analysis"),
        ("harmonic_completeness", 0.85),
        ("harmonic_inharmonic_ratio", 4.0),
    ]
    am_df = pd.DataFrame(am_rows, columns=["Parameter", "Value"])
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        metrics_sheet.to_excel(w, sheet_name="Metrics", index=False)
        am_df.to_excel(w, sheet_name="Analysis_Metadata", index=False)


def test_component_metrics_promoted_from_analysis_metadata(tmp_path: Path) -> None:
    p = tmp_path / "spectral_analysis.xlsx"
    _write_synthetic_per_note_workbook(p)
    metrics = cm.read_excel_metrics(p)
    assert metrics.get("component_harmonic_energy_ratio") == pytest.approx(0.72)
    assert metrics.get("component_inharmonic_energy_ratio") == pytest.approx(0.18)
    assert metrics.get("component_subbass_energy_ratio") == pytest.approx(0.10)
    assert metrics.get("component_total_inharmonic_energy_ratio") == pytest.approx(0.28)
    assert metrics.get("model_harmonic_weight") == pytest.approx(0.80)
    assert metrics.get("model_inharmonic_weight") == pytest.approx(0.20)
    assert metrics.get("harmonic_completeness") == pytest.approx(0.85)
    assert metrics.get("harmonic_inharmonic_ratio") == pytest.approx(4.0)
    # Provenance / textual fields should also be picked up.
    assert metrics.get("component_energy_method") == "single_pass_proc_audio_energy"
    assert metrics.get("component_profile_source") == "integrated_single_pass"
    assert metrics.get("model_weights_source") == "current_analysis"


# ---------------------------------------------------------------------------
# E — model_weights_source provenance
# ---------------------------------------------------------------------------
class _DummyLogger:
    def __init__(self) -> None:
        self.messages = []

    def info(self, *args, **kwargs) -> None:
        self.messages.append(("info", args, kwargs))

    def warning(self, *args, **kwargs) -> None:
        self.messages.append(("warning", args, kwargs))

    def debug(self, *args, **kwargs) -> None:
        self.messages.append(("debug", args, kwargs))


def test_set_model_weights_marks_provenance_current_analysis() -> None:
    """When ``_set_model_weights_from_current_component_energy`` runs in
    auto mode it must set ``model_weights_source = current_analysis`` and
    ``component_profile_source = current_analysis`` regardless of any
    pre-existing ``gui_weight_resolution_meta`` payload.
    """
    import proc_audio as pa
    proc = pa.AudioProcessor()
    proc.logger = _DummyLogger()
    proc.harmonic_energy_sum = 0.7
    proc.inharmonic_energy_sum = 0.2
    proc.subbass_energy_sum = 0.1
    proc.auto_model_weights_from_analysis = True
    # Pre-populate misleading meta (e.g. ``apply_filters_arguments``) and
    # confirm the current-analysis helper overrides it.
    proc.gui_weight_resolution_meta = {"model_weights_source": "apply_filters_arguments"}
    proc._set_model_weights_from_current_component_energy()

    assert proc.model_weights_source == "current_analysis"
    assert proc.component_profile_source == "current_analysis"
    # The gui meta must have been corrected in-place so the per_note_row
    # export does not re-introduce the wrong label.
    assert proc.gui_weight_resolution_meta["model_weights_source"] == "current_analysis"
    assert proc.gui_weight_resolution_meta["component_profile_source"] == "current_analysis"
    assert proc.model_harmonic_weight == pytest.approx(0.7 / 0.9)
    assert proc.model_inharmonic_weight == pytest.approx(0.2 / 0.9)


def test_set_model_weights_legacy_mode_preserves_external_source() -> None:
    import proc_audio as pa
    proc = pa.AudioProcessor()
    proc.logger = _DummyLogger()
    proc.harmonic_energy_sum = 0.5
    proc.inharmonic_energy_sum = 0.5
    proc.subbass_energy_sum = 0.0
    proc.auto_model_weights_from_analysis = False
    proc.gui_weight_resolution_meta = {"model_weights_source": "batch_handoff"}
    proc._set_model_weights_from_current_component_energy()
    # In legacy mode we must NOT relabel the source as current_analysis.
    assert getattr(proc, "model_weights_source", None) is None
    assert proc.gui_weight_resolution_meta["model_weights_source"] == "batch_handoff"


# ---------------------------------------------------------------------------
# F — diagnostic / legacy chart warning
# ---------------------------------------------------------------------------
def test_metric_requires_warning_canonical_returns_none() -> None:
    assert pcp.metric_requires_warning("density_normalized_global") is None
    assert pcp.metric_requires_warning("canonical_density_v5_adapted") is None
    assert pcp.metric_requires_warning("spectral_entropy") is None


def test_metric_requires_warning_legacy_returns_publication_text() -> None:
    msg = pcp.metric_requires_warning("Harmonic Partials sum")
    assert msg is not None
    assert "publication" in msg.lower()


def test_metric_requires_warning_diagnostic_returns_warning() -> None:
    # ``harmonic_energy_ratio`` is the diagnostic alias of
    # ``component_harmonic_energy_ratio``; the dictionary marks it as
    # ``status="diagnostic"``.
    msg = pcp.metric_requires_warning("harmonic_energy_ratio")
    assert msg is not None


def test_compose_chart_title_appends_warning_for_non_canonical() -> None:
    title = pcp.compose_chart_title(
        "Canonical_Metrics", "Harmonic Partials sum", status="legacy"
    )
    assert "Canonical_Metrics" in title
    assert "Harmonic Partials sum" in title
    assert "legacy" in title
    assert "WARNING" in title


def test_compose_chart_title_canonical_has_no_warning_suffix() -> None:
    title = pcp.compose_chart_title(
        "Canonical_Metrics", "density_normalized_global", status="canonical"
    )
    assert "density_normalized_global" in title
    assert "canonical" in title
    assert "WARNING" not in title


def test_compose_chart_title_auto_status_detects_canonical_vs_legacy() -> None:
    canon_title = pcp.compose_chart_title("Canonical_Metrics", "spectral_entropy")
    assert "canonical" in canon_title
    assert "WARNING" not in canon_title
    legacy_title = pcp.compose_chart_title(
        "Compiled_Metrics_All", "Density Metric"
    )
    assert "legacy" in legacy_title
    assert "WARNING" in legacy_title


# ---------------------------------------------------------------------------
# C-bis — full workbook round-trip: Canonical_Metrics has every dictionary entry
# ---------------------------------------------------------------------------
def test_write_compiled_excel_canonical_sheet_covers_dictionary(tmp_path: Path) -> None:
    """Synthesize a wide DataFrame containing every dictionary-canonical
    column (with realistic numeric values), call the multi-sheet writer,
    re-read the resulting workbook, and confirm that
    ``Canonical_Metrics`` exposes the expected coverage.
    """
    dict_canonical = _load_canonical_names_from_dictionary()
    # Build a per-row payload: numeric defaults for everything except the
    # text identifier columns.
    text_canonical = {"Note", "source_file_name", "tier"}
    rows = []
    for i in range(2):
        row: dict[str, object] = {}
        for n in dict_canonical:
            if n in text_canonical:
                row[n] = ("C4" if n == "Note" else
                          "synthetic" if n == "source_file_name" else
                          "Tier_1")
            else:
                row[n] = 0.5 + 0.01 * i
        rows.append(row)
    df = pd.DataFrame(rows)
    # Add a few diagnostic / legacy columns to make sure they are filtered
    # out of the canonical sheet.
    df["Harmonic Partials sum"] = [12345.6, 7890.1]
    df["batch_harmonic_energy_ratio"] = [0.5, 0.51]

    outp = tmp_path / "compiled_density_metrics.xlsx"
    cm._write_compiled_excel(
        outp,
        df,
        metadata={"weight_function": "linear"},
        apply_publication_column_filter=False,
        enable_pca_export=False,
    )
    canon_df = pd.read_excel(outp, sheet_name="Canonical_Metrics")
    missing = [n for n in dict_canonical if n not in canon_df.columns]
    assert not missing, (
        f"Canonical_Metrics is missing dictionary-canonical entries: {missing}"
    )
    assert "Harmonic Partials sum" not in canon_df.columns
    assert "batch_harmonic_energy_ratio" not in canon_df.columns


# ---------------------------------------------------------------------------
# A-bis — load_canonical_sheet_with_fallback_warning
# ---------------------------------------------------------------------------
def test_load_canonical_sheet_prefers_canonical_metrics(tmp_path: Path) -> None:
    p = tmp_path / "wb.xlsx"
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        # Write the legacy "first sheet" deliberately first so that a
        # naive ``pd.read_excel(p)`` would pick it.
        pd.DataFrame({"Harmonic Partials sum": [99999.0]}).to_excel(
            w, sheet_name="Density_Metrics", index=False
        )
        pd.DataFrame({"density_normalized_global": [0.5]}).to_excel(
            w, sheet_name="Canonical_Metrics", index=False
        )
    df, used, warns = pcp.load_canonical_sheet_with_fallback_warning(p)
    assert used == "Canonical_Metrics"
    assert "density_normalized_global" in df.columns
    assert "Harmonic Partials sum" not in df.columns
    assert warns == []


def test_load_canonical_sheet_warns_when_missing_canonical(tmp_path: Path) -> None:
    p = tmp_path / "wb.xlsx"
    with pd.ExcelWriter(p, engine="openpyxl") as w:
        pd.DataFrame({"Harmonic Partials sum": [99999.0]}).to_excel(
            w, sheet_name="Density_Metrics", index=False
        )
    df, used, warns = pcp.load_canonical_sheet_with_fallback_warning(p)
    assert used == "Density_Metrics"
    assert warns, "missing Canonical_Metrics must emit a warning"
    assert "publication-grade" in warns[0].lower() or "diagnostic" in warns[0].lower()
