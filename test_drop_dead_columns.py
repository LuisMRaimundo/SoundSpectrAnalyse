"""Regression test for ``compile_metrics._drop_dead_columns``.

Pinning the contract documented in `compile_metrics.py` (audit fix for
the Clarinete_mf workbook-clutter complaint):

* All-NaN columns are dropped.
* All-blank-string / "None" / "<NA>" columns are dropped.
* All-zero numeric columns are KEPT (0.0 is a legitimate observation).
* The ``Note`` column is protected and never dropped.
* Mixed (partially populated) columns are kept.
* Empty / single-column-empty inputs are returned unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from compile_metrics import _drop_dead_columns


def test_drop_all_nan_columns() -> None:
    df = pd.DataFrame(
        {
            "Note": ["A4", "C5", "E5"],
            "alive_numeric": [1.0, 2.0, 3.0],
            "dead_all_nan": [np.nan, np.nan, np.nan],
            "alive_zero_numeric": [0.0, 0.0, 0.0],
            "dead_all_blank_string": ["", "", ""],
            "dead_all_none_string": ["None", "None", "None"],
            "alive_mixed_string": ["x", "", "y"],
            "alive_partial_nan_numeric": [1.0, np.nan, 3.0],
        }
    )
    out = _drop_dead_columns(df)
    assert list(out.columns) == [
        "Note",
        "alive_numeric",
        "alive_zero_numeric",
        "alive_mixed_string",
        "alive_partial_nan_numeric",
    ]


def test_all_zero_numeric_is_preserved() -> None:
    """``subbass_energy_sum`` may legitimately be 0 for a note with no
    sub-bass content. The column must survive so analysts can tell
    "measured zero" from "not measured".
    """
    df = pd.DataFrame(
        {
            "Note": ["A4"],
            "subbass_energy_sum": [0.0],
            "Total sum": [0.0],
            "weight_function": ["log"],
        }
    )
    out = _drop_dead_columns(df)
    assert set(out.columns) == {
        "Note",
        "subbass_energy_sum",
        "Total sum",
        "weight_function",
    }


def test_note_is_protected_even_if_blank() -> None:
    """Defence in depth: a buggy upstream stage producing a blank
    ``Note`` column must not lose the row key.
    """
    df = pd.DataFrame(
        {
            "Note": ["", "", ""],
            "alive": [1.0, 2.0, 3.0],
        }
    )
    out = _drop_dead_columns(df)
    assert "Note" in out.columns
    assert "alive" in out.columns


def test_empty_dataframe_passthrough() -> None:
    assert _drop_dead_columns(pd.DataFrame()).empty


def test_returns_original_when_nothing_to_drop() -> None:
    df = pd.DataFrame({"Note": ["A4"], "x": [1.0]})
    out = _drop_dead_columns(df)
    assert out is df  # identity (no copy when nothing changed)


def test_drop_only_after_strip() -> None:
    """Strings that are whitespace-only should be treated as blank-like."""
    df = pd.DataFrame(
        {
            "Note": ["A4", "C5"],
            "dead_whitespace": ["   ", "\t"],
            "alive_one_real": ["", "data"],
        }
    )
    out = _drop_dead_columns(df)
    assert "dead_whitespace" not in out.columns
    assert "alive_one_real" in out.columns
