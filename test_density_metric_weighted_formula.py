"""Audit tests for the Density_Metrics weighted density formula.

Audit points A–E:

A. **Weighted formula correctness** — for a synthetic single row, the
   contribution terms and ``density_metric_raw`` follow the formula
   ``D_H * w_H + D_I * w_I + D_S * w_S`` exactly.

B. **Normalization** — ``density_metric_normalized`` is the run-relative
   max-normalisation of ``density_metric_raw``.

C. **No legacy source** — in ``integrated_single_pass`` the component
   ratios must be sourced from ``component_*_energy_ratio``, NOT from
   ``batch_*_energy_ratio`` (which is now classified as ``legacy``).

D. **Total sum diagnostic** — ``Total sum`` keeps its unweighted
   semantics (D_H + D_I + D_S) and is *not* the final density metric.
   The publication-policy classification places it on the ban list.

E. **Plot default** — when the active sheet is ``Density_Metrics``, the
   default plot metric is ``density_metric_normalized`` (never
   ``Harmonic Partials sum`` or ``Total sum``).
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

import compile_metrics as cm
import publication_chart_policy as pcp


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _make_density_row(
    note: str,
    *,
    d_h: float,
    d_i: float,
    d_s: float,
    w_h: float,
    w_i: float,
    w_s: float,
    legacy_batch: bool = False,
    add_diagnostic_alias: bool = True,
) -> dict[str, object]:
    """Build a single wide-row dict suitable for ``_build_density_metrics_main_sheet``.

    By default emits the canonical ``component_*_energy_ratio`` fields
    (integrated_single_pass source). When ``legacy_batch=True`` is set,
    emits ONLY ``batch_*_energy_ratio`` instead, to verify that the
    builder refuses to source from legacy aliases. ``add_diagnostic_alias``
    controls the older ``harmonic_energy_ratio``/`inharmonic_energy_ratio``
    fields, which the fallback uses when ``component_*`` is missing.
    """
    row: dict[str, object] = {
        "Note": note,
        "weight_function": "linear",
        "Harmonic Partials sum": float(d_h),
        "Inharmonic Partials sum": float(d_i),
        "Sub-bass sum": float(d_s),
        "Total sum": float(d_h + d_i + d_s),
    }
    if legacy_batch:
        row["batch_harmonic_energy_ratio"] = float(w_h)
        row["batch_inharmonic_energy_ratio"] = float(w_i)
        row["batch_subbass_energy_ratio"] = float(w_s)
    else:
        row["component_harmonic_energy_ratio"] = float(w_h)
        row["component_inharmonic_energy_ratio"] = float(w_i)
        row["component_subbass_energy_ratio"] = float(w_s)
    if add_diagnostic_alias:
        row["harmonic_energy_ratio"] = float(w_h)
        row["inharmonic_energy_ratio"] = float(w_i)
        row["subbass_energy_ratio"] = float(w_s)
    return row


# ---------------------------------------------------------------------------
# A — weighted formula correctness
# ---------------------------------------------------------------------------
def test_density_metric_raw_matches_audit_specification() -> None:
    """Given the audit example::

        D_H = 100, D_I = 50, D_S = 10
        w_H = 0.8, w_I = 0.15, w_S = 0.05

    ``density_metric_raw`` must equal exactly ``88.0``.
    """
    df = pd.DataFrame([
        _make_density_row(
            "audit_example",
            d_h=100.0, d_i=50.0, d_s=10.0,
            w_h=0.8, w_i=0.15, w_s=0.05,
        ),
    ])
    out = cm._build_density_metrics_main_sheet(df, weight_function="linear")
    row = out.iloc[0]
    assert row["weighted_harmonic_density_contribution"] == pytest.approx(80.0)
    assert row["weighted_inharmonic_density_contribution"] == pytest.approx(7.5)
    assert row["weighted_subbass_density_contribution"] == pytest.approx(0.5)
    assert row["density_metric_raw"] == pytest.approx(88.0)


def test_weighted_contributions_are_additive_and_finite() -> None:
    df = pd.DataFrame([
        _make_density_row("R1", d_h=10.0, d_i=5.0, d_s=1.0, w_h=0.7, w_i=0.2, w_s=0.1),
        _make_density_row("R2", d_h=20.0, d_i=8.0, d_s=2.0, w_h=0.6, w_i=0.3, w_s=0.1),
    ])
    out = cm._build_density_metrics_main_sheet(df, weight_function="linear")
    contrib_sum = (
        out["weighted_harmonic_density_contribution"]
        + out["weighted_inharmonic_density_contribution"]
        + out["weighted_subbass_density_contribution"]
    )
    pd.testing.assert_series_equal(
        contrib_sum.rename("density_metric_raw"),
        out["density_metric_raw"],
        check_dtype=False,
    )
    assert np.isfinite(out["density_metric_raw"]).all()


# ---------------------------------------------------------------------------
# B — normalization
# ---------------------------------------------------------------------------
def test_density_metric_normalized_max_normalises_within_workbook() -> None:
    """For raw values [88, 44, 22], the normalized series is [1.0, 0.5, 0.25]."""
    df = pd.DataFrame([
        # Row whose raw is 88 (audit example).
        _make_density_row("R88", d_h=100.0, d_i=50.0, d_s=10.0,
                          w_h=0.8, w_i=0.15, w_s=0.05),
        # Row whose raw is 44 = half of 88.
        _make_density_row("R44", d_h=50.0, d_i=25.0, d_s=5.0,
                          w_h=0.8, w_i=0.15, w_s=0.05),
        # Row whose raw is 22 = quarter of 88.
        _make_density_row("R22", d_h=25.0, d_i=12.5, d_s=2.5,
                          w_h=0.8, w_i=0.15, w_s=0.05),
    ])
    out = cm._build_density_metrics_main_sheet(df, weight_function="linear")
    raw = out["density_metric_raw"].tolist()
    norm = out["density_metric_normalized"].tolist()
    assert raw == [pytest.approx(88.0), pytest.approx(44.0), pytest.approx(22.0)]
    assert norm == [pytest.approx(1.0), pytest.approx(0.5), pytest.approx(0.25)]


def test_density_metric_normalized_is_nan_when_max_is_nonpositive(caplog) -> None:
    """When every weighted contribution is zero (e.g. ratios are zero), the
    function returns ``NaN`` for the normalized column and emits a warning."""
    df = pd.DataFrame([
        _make_density_row("R0a", d_h=100.0, d_i=50.0, d_s=10.0,
                          w_h=0.0, w_i=0.0, w_s=0.0),
        _make_density_row("R0b", d_h=200.0, d_i=70.0, d_s=20.0,
                          w_h=0.0, w_i=0.0, w_s=0.0),
    ])
    with caplog.at_level("WARNING", logger="compile_metrics"):
        out = cm._build_density_metrics_main_sheet(df, weight_function="linear")
    assert out["density_metric_normalized"].isna().all()
    assert any("density_metric_normalized" in rec.message for rec in caplog.records)


# ---------------------------------------------------------------------------
# C — no legacy source in single-pass mode
# ---------------------------------------------------------------------------
def test_weighted_density_uses_component_not_batch() -> None:
    """If both ``component_*`` and ``batch_*`` ratios are present, the
    builder MUST pick ``component_*`` (the canonical single-source-of-
    truth) — even when ``batch_*`` would yield a numerically different
    answer.
    """
    row = _make_density_row(
        "Cdom",
        d_h=100.0, d_i=50.0, d_s=10.0,
        w_h=0.8, w_i=0.15, w_s=0.05,
        legacy_batch=False,
        add_diagnostic_alias=False,
    )
    # Inject misleading batch_* values that, if used, would change the answer.
    row["batch_harmonic_energy_ratio"] = 0.1
    row["batch_inharmonic_energy_ratio"] = 0.1
    row["batch_subbass_energy_ratio"] = 0.1
    df = pd.DataFrame([row])
    out = cm._build_density_metrics_main_sheet(df, weight_function="linear")
    assert out.iloc[0]["density_metric_raw"] == pytest.approx(88.0)
    assert out.iloc[0]["component_harmonic_energy_ratio"] == pytest.approx(0.8)
    assert out.iloc[0]["component_inharmonic_energy_ratio"] == pytest.approx(0.15)
    assert out.iloc[0]["component_subbass_energy_ratio"] == pytest.approx(0.05)


def test_legacy_batch_ratios_alone_are_not_used_as_source() -> None:
    """When only ``batch_*`` ratios exist (no ``component_*`` and no
    diagnostic alias), the builder treats the component ratios as
    missing and returns NaN for the weighted columns rather than
    silently using the legacy fields.
    """
    row = _make_density_row(
        "Lonly",
        d_h=100.0, d_i=50.0, d_s=10.0,
        w_h=0.8, w_i=0.15, w_s=0.05,
        legacy_batch=True,
        add_diagnostic_alias=False,
    )
    df = pd.DataFrame([row])
    out = cm._build_density_metrics_main_sheet(df, weight_function="linear")
    assert pd.isna(out.iloc[0]["component_harmonic_energy_ratio"])
    assert pd.isna(out.iloc[0]["component_inharmonic_energy_ratio"])
    assert pd.isna(out.iloc[0]["component_subbass_energy_ratio"])
    assert pd.isna(out.iloc[0]["density_metric_raw"])


# ---------------------------------------------------------------------------
# D — Total sum remains diagnostic, not the final density metric
# ---------------------------------------------------------------------------
def test_total_sum_remains_unweighted_diagnostic() -> None:
    df = pd.DataFrame([
        _make_density_row("T1", d_h=100.0, d_i=50.0, d_s=10.0,
                          w_h=0.8, w_i=0.15, w_s=0.05),
        _make_density_row("T2", d_h=20.0, d_i=8.0, d_s=2.0,
                          w_h=0.6, w_i=0.3, w_s=0.1),
    ])
    out = cm._build_density_metrics_main_sheet(df, weight_function="linear")
    expected_total = (
        out["Harmonic Partials sum"] + out["Inharmonic Partials sum"] + out["Sub-bass sum"]
    )
    pd.testing.assert_series_equal(
        out["Total sum"].astype(float).rename(None),
        expected_total.astype(float).rename(None),
        check_dtype=False,
    )
    # And Total sum must NEVER reach the canonical sheet via the policy.
    assert pcp.classify_metric_for_publication("Total sum") == "legacy"


def test_total_sum_not_selected_as_default_plot_metric_for_density_metrics() -> None:
    # AUDIT FIX (Stage 2 weighted note-density) — the publication-friendly
    # default for Density_Metrics is the absolute weighted-density value
    # in log space (``density_log_weighted``). Workbook-relative
    # ``density_metric_normalized`` and any raw partial-sum column are
    # never chosen as the default.
    cols = [
        "Note",
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Sub-bass sum",
        "Total sum",
        "density_metric_normalized",
        "density_metric_raw",
        "density_log_weighted",
        "harmonic_log_amplitude_density",
    ]
    chosen = pcp.select_default_publication_metric(
        cols, sheet_name="Density_Metrics"
    )
    assert chosen == "density_log_weighted"
    assert chosen != "Total sum"
    assert chosen != "Harmonic Partials sum"
    assert chosen != "density_metric_normalized"


# ---------------------------------------------------------------------------
# E — plot default for Density_Metrics
# ---------------------------------------------------------------------------
def test_density_metrics_default_plot_is_density_log_weighted() -> None:
    cols = [
        "Note",
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Total sum",
        "density_metric_raw",
        "density_metric_normalized",
        "density_log_weighted",
        "harmonic_log_amplitude_density",
    ]
    chosen = pcp.select_default_publication_metric(
        cols, sheet_name="Density_Metrics"
    )
    assert chosen == "density_log_weighted"


def test_density_metrics_default_plot_falls_back_to_harmonic_log_when_weighted_missing() -> None:
    cols = [
        "Note",
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Total sum",
        "density_metric_raw",
        "density_metric_normalized",
        "harmonic_log_amplitude_density",
    ]
    chosen = pcp.select_default_publication_metric(
        cols, sheet_name="Density_Metrics"
    )
    assert chosen == "harmonic_log_amplitude_density"


def test_density_metrics_default_never_falls_back_to_forbidden_when_missing_normalized() -> None:
    """If ``density_metric_normalized`` is unavailable, the per-sheet
    preference must return ``None`` rather than fall back to a forbidden
    raw column."""
    cols = [
        "Note",
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Total sum",
    ]
    chosen = pcp.select_default_publication_metric(
        cols, sheet_name="Density_Metrics"
    )
    assert chosen is None


def test_canonical_sheet_default_unchanged() -> None:
    """The per-sheet override must not regress the canonical default for
    ``Canonical_Metrics`` (still ``density_normalized_global``)."""
    cols = [
        "Note",
        "density_normalized_global",
        "canonical_density_v5_adapted",
        "effective_partial_density",
    ]
    chosen = pcp.select_default_publication_metric(
        cols, sheet_name="Canonical_Metrics"
    )
    assert chosen == "density_normalized_global"


# ---------------------------------------------------------------------------
# F (bonus) — metrics_dictionary.json coverage of the new entries
# ---------------------------------------------------------------------------
def _dict_entries() -> dict[str, dict]:
    p = REPO_ROOT / "metrics_dictionary.json"
    with p.open(encoding="utf-8") as fh:
        data = json.load(fh)
    return {m["name"]: m for m in data["metrics"]}


def test_metrics_dictionary_declares_new_density_entries() -> None:
    entries = _dict_entries()
    required = {
        "weighted_harmonic_density_contribution",
        "weighted_inharmonic_density_contribution",
        "weighted_subbass_density_contribution",
        "density_metric_raw",
        "density_metric_normalized",
    }
    missing = required - set(entries)
    assert not missing, f"missing dictionary entries: {missing}"
    # density_metric_raw and weighted_* are diagnostic — they must NEVER
    # be classified as canonical (the user must not be tricked into
    # plotting them as final results).
    for k in (
        "density_metric_raw",
        "weighted_harmonic_density_contribution",
        "weighted_inharmonic_density_contribution",
        "weighted_subbass_density_contribution",
    ):
        assert entries[k]["status"] == "diagnostic", k
    # density_metric_normalized is run-relative; we document it as
    # diagnostic with an explicit "run-relative" clause.
    norm_entry = entries["density_metric_normalized"]
    assert norm_entry["status"] == "diagnostic"
    assert "run-relative" in norm_entry["do_not_interpret_as"].lower()


# ---------------------------------------------------------------------------
# Full workbook round-trip — confirm Density_Metrics has the new layout
# ---------------------------------------------------------------------------
def test_compiled_density_metrics_sheet_contains_weighted_columns(tmp_path: Path) -> None:
    rows = []
    for i in range(3):
        rows.append(_make_density_row(
            f"N{i}",
            d_h=100.0 + 10.0 * i,
            d_i=50.0 + 5.0 * i,
            d_s=10.0 + 1.0 * i,
            w_h=0.7, w_i=0.2, w_s=0.1,
        ))
    df = pd.DataFrame(rows)
    outp = tmp_path / "compiled.xlsx"
    cm._write_compiled_excel(outp, df, {}, enable_pca_export=False)
    dm = pd.read_excel(outp, sheet_name="Density_Metrics")
    for c in (
        "Harmonic Partials sum",
        "Inharmonic Partials sum",
        "Sub-bass sum",
        "Total sum",
        "component_harmonic_energy_ratio",
        "component_inharmonic_energy_ratio",
        "component_subbass_energy_ratio",
        "weighted_harmonic_density_contribution",
        "weighted_inharmonic_density_contribution",
        "weighted_subbass_density_contribution",
        "density_metric_raw",
        "density_metric_normalized",
    ):
        assert c in dm.columns, c
    # Audit example raw / normalized sanity check.
    expected_raw = 0.7 * dm["Harmonic Partials sum"] \
                 + 0.2 * dm["Inharmonic Partials sum"] \
                 + 0.1 * dm["Sub-bass sum"]
    np.testing.assert_allclose(
        dm["density_metric_raw"].astype(float).to_numpy(),
        expected_raw.astype(float).to_numpy(),
        rtol=0.0, atol=1e-9,
    )
    mx = float(np.nanmax(dm["density_metric_raw"]))
    np.testing.assert_allclose(
        dm["density_metric_normalized"].astype(float).to_numpy(),
        (dm["density_metric_raw"] / mx).astype(float).to_numpy(),
        rtol=0.0, atol=1e-9,
    )
