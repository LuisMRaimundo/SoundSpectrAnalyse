"""publication_chart_policy.py
============================

Central policy module for user-facing plots, charts, and default metric
selectors generated from compiled SoundSpectrAnalyse workbooks.

Why this module exists
----------------------
After the single-pass refactor the compiled workbook exposes three
intentionally separated sheets:

    Canonical_Metrics       — final, publication-grade quantities
    Diagnostic_Metrics      — intermediate / provenance / aliases
    Legacy_Compatibility    — back-compat columns (``batch_*``, ``legacy_*``,
                              ``Density Metric``, ``Spectral Density Metric``,
                              ``Combined Density Metric``, ``Filtered Density Metric``)

A GUI / analysis script that defaults to plotting an unbounded legacy column
such as ``Harmonic Partials sum`` is **scientifically misleading** because
those values are raw bin-row amplitude sums (typical magnitudes 10⁴–10⁵) and
they are not the result of the canonical single-pass analysis.

This module concentrates every plotting-related contract so we cannot drift:

    - ``DEFAULT_PUBLICATION_SHEET``         the only acceptable default sheet
    - ``DEFAULT_PUBLICATION_METRIC_PREFERENCE``  ordered preference list
    - ``DEFAULT_PUBLICATION_METRIC``        the single canonical default
    - ``FORBIDDEN_DEFAULT_METRIC_NAMES``    raw legacy columns banned as default
    - ``classify_metric_for_publication()`` canonical / diagnostic / legacy
    - ``select_default_publication_metric()`` honour the policy
    - ``compose_chart_title()``             "sheet — metric — status[ — WARNING]"
    - ``metric_requires_warning()``         returns warning text or None
    - ``forbidden_metric_warning()``        canonical wording from the audit
    - ``load_canonical_sheet_with_fallback_warning()`` strict reader
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
DEFAULT_PUBLICATION_SHEET: str = "Canonical_Metrics"

# Ordered preference for the single default metric: every item must live in
# ``Canonical_Metrics`` and carry ``status="canonical"`` in
# ``metrics_dictionary.json``. The first available column in this order is
# selected by ``select_default_publication_metric``.
DEFAULT_PUBLICATION_METRIC_PREFERENCE: Tuple[str, ...] = (
    "density_normalized_global",
    "canonical_density_v5_adapted",
    "effective_partial_density",
    "model_harmonic_weight",
    "model_inharmonic_weight",
    "spectral_entropy",
)

DEFAULT_PUBLICATION_METRIC: str = DEFAULT_PUBLICATION_METRIC_PREFERENCE[0]

# Per-sheet default preference. When the user explicitly selects a non-
# canonical sheet (e.g. ``Density_Metrics``), this map dictates the
# default plot metric for that sheet. Raw partial sums are never the
# default; they remain available only under the explicit diagnostic
# section, with the warning emitted by ``metric_requires_warning``.
DEFAULT_PUBLICATION_METRIC_BY_SHEET: dict[str, Tuple[str, ...]] = {
    "Canonical_Metrics": DEFAULT_PUBLICATION_METRIC_PREFERENCE,
    # AUDIT FIX (Stage 2 weighted note-density) — when Density_Metrics
    # is the active sheet, the publication-friendly default is the
    # final weighted-density metric in log space:
    #
    #     density_log_weighted = log10(1 + density_weighted_sum)
    #
    # where density_weighted_sum = D_H·w_H + D_I·w_I + D_S·w_S and each
    # D_* follows the compile weight_function (same as density_metric_raw).
    # harmonic_log_amplitude_density is the harmonic-only linear-amplitude
    # fallback when the full weighted metric is unavailable.
    # density_metric_normalized is workbook-relative; it is intentionally
    # NOT defaulted-to here. density_metric_raw equals density_weighted_sum
    # and is unbounded — diagnostic only for auto-default plots.
    # Harmonic Partials sum / Total sum are NEVER selected as defaults.
    "Density_Metrics": (
        "density_log_weighted",
        "harmonic_log_amplitude_density",
        # density_metric_raw / density_metric_normalized / Power_raw-based
        # metrics / Total sum / Harmonic Partials sum are intentionally
        # NOT defaults — diagnostic / legacy only.
    ),
    "Diagnostic_Metrics": DEFAULT_PUBLICATION_METRIC_PREFERENCE,
    "Compiled_Metrics_All": DEFAULT_PUBLICATION_METRIC_PREFERENCE,
}

# Explicit ban list. These columns are raw / legacy / back-compat and must
# never be selected automatically as the default metric for a publication-
# facing chart. They are still **allowed** under an explicit diagnostic /
# legacy section guarded by ``forbidden_metric_warning``.
FORBIDDEN_DEFAULT_METRIC_NAMES: frozenset[str] = frozenset(
    {
        # raw bin-row amplitude sums (unbounded; legacy)
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Sub-bass sum",
        "Total sum",
        # legacy density quantities (semantically deprecated)
        "harmonic_density",
        "inharmonic_density",
        "combined_density",
        # legacy capitalised columns
        "Density Metric",
        "Spectral Density Metric",
        "Combined Density Metric",
        "density_weighted_sum_cdm_mean",
        "Filtered Density Metric",
    }
)

# Prefixes / substrings that force LEGACY classification for any column not on
# the canonical allow-list. Kept in sync with compile_metrics.LEGACY_COLUMN_*.
FORBIDDEN_DEFAULT_METRIC_PREFIXES: Tuple[str, ...] = (
    "linear_sum_amplitude_",
    "batch_",
    "legacy_",
)

# Warning text rendered next to any chart whose metric is diagnostic / legacy.
DIAGNOSTIC_LEGACY_WARNING_TEXT: str = (
    "Raw or legacy diagnostic quantity; not a publication-grade metric."
)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------
def _load_metric_dictionary_status_map() -> dict[str, str]:
    """Read ``metrics_dictionary.json`` (next to this file) and return
    ``{name: status}``. Returns ``{}`` if the dictionary is unavailable so
    callers can fall back to the local heuristics below.
    """
    try:
        import json
        path = Path(__file__).resolve().parent / "metrics_dictionary.json"
        if not path.is_file():
            return {}
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        out: dict[str, str] = {}
        for entry in data.get("metrics", []):
            name = entry.get("name")
            status = entry.get("status")
            if name and status:
                out[str(name)] = str(status)
        return out
    except Exception as exc:
        logger.debug("metric dictionary unavailable for policy lookup: %s", exc)
        return {}


_METRIC_STATUS_CACHE: dict[str, str] = {}


def _metric_status_lookup(name: str) -> Optional[str]:
    """Return the dictionary-declared status for ``name`` or None."""
    global _METRIC_STATUS_CACHE
    if not _METRIC_STATUS_CACHE:
        _METRIC_STATUS_CACHE = _load_metric_dictionary_status_map()
    return _METRIC_STATUS_CACHE.get(str(name))


def classify_metric_for_publication(name: str) -> str:
    """Return one of ``"canonical"``, ``"diagnostic"``, ``"legacy"``.

    Decision order (the dictionary wins when it speaks):

    1. ``FORBIDDEN_DEFAULT_METRIC_NAMES`` literal match → "legacy".
    2. Any prefix in ``FORBIDDEN_DEFAULT_METRIC_PREFIXES`` → "legacy".
    3. ``metrics_dictionary.json`` declares the metric → use that status.
    4. Default → "diagnostic" (conservative).
    """
    c = str(name)
    if c in FORBIDDEN_DEFAULT_METRIC_NAMES:
        return "legacy"
    for pref in FORBIDDEN_DEFAULT_METRIC_PREFIXES:
        if c.startswith(pref):
            return "legacy"
    dict_status = _metric_status_lookup(c)
    if dict_status in ("canonical", "diagnostic", "legacy"):
        return dict_status
    return "diagnostic"


def metric_requires_warning(name: str) -> Optional[str]:
    """Return a human-readable warning text (or None) for ``name``.

    Anything classified as ``"diagnostic"`` or ``"legacy"`` requires an
    explicit warning when shown in a chart, per audit policy.
    """
    status = classify_metric_for_publication(name)
    if status == "canonical":
        return None
    return DIAGNOSTIC_LEGACY_WARNING_TEXT


def forbidden_metric_warning(name: str) -> Optional[str]:
    """Return the publication-policy warning if ``name`` is on the explicit
    ban list (raw / legacy literal). For other diagnostic metrics, returns
    the generic ``DIAGNOSTIC_LEGACY_WARNING_TEXT``. Returns None for
    canonical metrics.
    """
    return metric_requires_warning(name)


# ---------------------------------------------------------------------------
# Default-metric selection
# ---------------------------------------------------------------------------
def select_default_publication_metric(
    columns: Iterable[str],
    *,
    preference: Tuple[str, ...] = DEFAULT_PUBLICATION_METRIC_PREFERENCE,
    sheet_name: Optional[str] = None,
) -> Optional[str]:
    """Return the first ``preference`` entry that is present in ``columns``.

    The policy is conservative: if none of the preferred canonical metrics
    appear, returns ``None`` (the caller must decide whether to fall back
    or to fail loudly). Forbidden columns are *never* returned, even if
    they are the only ones available.

    When ``sheet_name`` is provided and present in
    ``DEFAULT_PUBLICATION_METRIC_BY_SHEET``, the per-sheet preference
    overrides the canonical preference list. Forbidden raw legacy
    columns remain banned regardless of the sheet selected.
    """
    cols = {str(c) for c in columns}
    if sheet_name and sheet_name in DEFAULT_PUBLICATION_METRIC_BY_SHEET:
        preference = DEFAULT_PUBLICATION_METRIC_BY_SHEET[sheet_name]
    for name in preference:
        if name in cols and classify_metric_for_publication(name) != "legacy":
            return name
    return None


def filter_out_forbidden_default_columns(columns: Iterable[str]) -> List[str]:
    """Return ``columns`` minus every entry banned by the policy.

    Useful when assembling a "select default metric" combo box: items left
    in the returned list are eligible defaults. Diagnostic / legacy entries
    can still be offered to the user but must be surfaced with the warning.
    """
    out: List[str] = []
    for c in columns:
        if classify_metric_for_publication(c) != "legacy":
            out.append(str(c))
    return out


# ---------------------------------------------------------------------------
# Chart title composition
# ---------------------------------------------------------------------------
def compose_chart_title(
    sheet_name: str,
    metric: str,
    *,
    status: Optional[str] = None,
) -> str:
    """Compose a chart title with the audit-mandated tag set.

    Format::

        "<sheet_name> — <metric> — <status>"

    A trailing ``" — WARNING: …"`` is appended whenever the metric is not
    canonical so the visualisation cannot be silently misleading.
    """
    s = str(sheet_name).strip() or DEFAULT_PUBLICATION_SHEET
    m = str(metric).strip()
    eff_status = (status or classify_metric_for_publication(m)).strip().lower()
    title = f"{s} — {m} — {eff_status}"
    warn = metric_requires_warning(m)
    if warn:
        title += f" — WARNING: {warn}"
    return title


# ---------------------------------------------------------------------------
# Strict reader: prefer Canonical_Metrics, warn on fallback
# ---------------------------------------------------------------------------
def load_canonical_sheet_with_fallback_warning(
    workbook_path: Union[str, Path],
    *,
    log: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, str, List[str]]:
    """Read the publication sheet from a compiled workbook.

    Returns ``(df, sheet_name_used, warnings_emitted)``.

    Priority:
      1. ``Canonical_Metrics``  — preferred
      2. ``Compiled_Metrics_All`` — only if (1) is absent; emits a warning.
      3. First sheet in the workbook — emits a *strong* warning.

    Raises ``FileNotFoundError`` if the workbook itself is missing.
    """
    lg = log or logger
    path = Path(workbook_path)
    if not path.is_file():
        raise FileNotFoundError(f"Compiled workbook not found: {path}")
    warns: List[str] = []
    try:
        with pd.ExcelFile(path) as xf:
            sheets = list(xf.sheet_names)
            if DEFAULT_PUBLICATION_SHEET in sheets:
                df = xf.parse(DEFAULT_PUBLICATION_SHEET)
                return df, DEFAULT_PUBLICATION_SHEET, warns
            if "Compiled_Metrics_All" in sheets:
                msg = (
                    f"Canonical_Metrics sheet missing in {path.name}; "
                    "falling back to Compiled_Metrics_All. Charts may now "
                    "expose diagnostic/legacy columns — confirm metric "
                    "status before publication."
                )
                warns.append(msg)
                lg.warning(msg)
                df = xf.parse("Compiled_Metrics_All")
                return df, "Compiled_Metrics_All", warns
            first = sheets[0] if sheets else ""
            msg = (
                f"Canonical_Metrics sheet missing in {path.name}; falling "
                f"back to first sheet {first!r}. This is NOT a "
                "publication-grade data source."
            )
            warns.append(msg)
            lg.warning(msg)
            df = xf.parse(first) if first else pd.DataFrame()
            return df, first, warns
    except Exception:
        raise


# ---------------------------------------------------------------------------
# Compatibility re-exports for tests
# ---------------------------------------------------------------------------
__all__ = [
    "DEFAULT_PUBLICATION_SHEET",
    "DEFAULT_PUBLICATION_METRIC",
    "DEFAULT_PUBLICATION_METRIC_PREFERENCE",
    "DEFAULT_PUBLICATION_METRIC_BY_SHEET",
    "FORBIDDEN_DEFAULT_METRIC_NAMES",
    "FORBIDDEN_DEFAULT_METRIC_PREFIXES",
    "DIAGNOSTIC_LEGACY_WARNING_TEXT",
    "classify_metric_for_publication",
    "metric_requires_warning",
    "forbidden_metric_warning",
    "select_default_publication_metric",
    "filter_out_forbidden_default_columns",
    "compose_chart_title",
    "load_canonical_sheet_with_fallback_warning",
]
