"""Canonical v5-adapted density, global [0,1] normalization, and synthetic profile separation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from compile_metrics import compile_density_metrics, validate_compiled_density_workbook
from density import apply_density_metric, CANONICAL_DENSITY_FORMULA_VERSION


def _compile_row(note: str, *, canonical: float, hoc: int = 8) -> dict:
    # AUDIT FIX — added the *modern* partial-sum column names required by
    # ``compile_metrics._build_density_metrics_display_df`` so the Density_Metrics
    # sheet can be built from the fixture instead of emitting a synthetic
    # "compilation_error" placeholder column (which the validator rejects).
    #
    # Triage of the original failure:
    # Type A (invalid fixture): the fixture supplied power sums
    # (``*_energy_sum``) but not the linear partial-sum columns the compiled
    # workbook actually exports. Independent of the single-pass refactor —
    # the test predates a compile_metrics column rename.
    return {
        "Note": note,
        "canonical_density_v5_adapted": canonical,
        "density_per_component": canonical / float(hoc),
        "effective_partial_density": 2.0,
        "harmonic_energy_sum": 1.0,
        "inharmonic_energy_sum": 0.2,
        "subbass_energy_sum": 0.1,
        "total_component_energy": 1.3,
        "harmonic_energy_ratio": 0.77,
        "inharmonic_energy_ratio": 0.15,
        "subbass_energy_ratio": 0.08,
        "harmonic_order_count": hoc,
        "spectral_entropy": 0.4,
        "rolloff_compensated_harmonic_density": 1.0,
        "rolloff_compensated_harmonic_density_alpha": 1.5,
        "rolloff_compensated_harmonic_density_component_count": hoc,
        "rolloff_compensated_harmonic_density_status": "computed",
        "density_metric_per_harmonic": 0.1,
        # Linear partial-sum columns expected by the Density_Metrics sheet
        # builder. We synthesise plausible amplitude sums consistent with the
        # power sums above (Σ|A| roughly proportional to √(ΣA²)).
        "Harmonic Partials sum": 1.0,
        "Inharmonic Partials sum": 0.45,
        "Sub-bass sum": 0.32,
        "Total sum": 1.77,
        "weight_function": "linear",
    }


def test_flute_like_sparse_vs_violin_like_rich_canonical(tmp_path) -> None:
    f0 = 300.0
    n_fl, n_vn = 8, 12
    f_fl = f0 * np.arange(1, n_fl + 1, dtype=float)
    a_fl = np.array([1.0] + [0.03] * (n_fl - 1), dtype=float)
    f_vn = f0 * np.arange(1, n_vn + 1, dtype=float)
    a_vn = np.array([1.0, 0.5, 0.42, 0.35, 0.3, 0.25, 0.2, 0.16, 0.12, 0.1, 0.08, 0.06], dtype=float)
    d_fl = float(
        apply_density_metric(
            a_fl,
            "linear",
            frequencies=f_fl,
            fundamental_freq=f0,
            account_for_spectral_rolloff=True,
            prevent_domination=True,
        )
    )
    d_vn = float(
        apply_density_metric(
            a_vn,
            "linear",
            frequencies=f_vn,
            fundamental_freq=f0,
            account_for_spectral_rolloff=True,
            prevent_domination=True,
        )
    )
    assert d_vn > d_fl, "richer harmonic ladder should yield higher canonical density than sparse flute-like"


def test_global_norm_bounded_and_density_metric_normalized_distinct(tmp_path) -> None:
    """``density_normalized_global`` is bounded in [0, 1] and equals the
    canonical density max-norm.

    AUDIT FIX (single-pass weighted density) — ``density_metric_normalized``
    is no longer an alias of ``density_normalized_global``. It is now the
    run-relative max-norm of the weighted partial-sum ``density_metric_raw``.
    The two metrics are conceptually distinct and may take different values
    on the same compiled workbook (each is bounded in [0, 1] independently).
    """
    root = tmp_path
    for i, (note, c) in enumerate([("G4", 2.0), ("A4", 8.0), ("B4", 4.0)]):
        d = root / f"Note_{note}_{i}"
        d.mkdir(parents=True)
        pd.DataFrame([_compile_row(note, canonical=c)]).to_excel(
            d / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False
        )
    outp = root / "out.xlsx"
    df = compile_density_metrics(root, output_path=outp, file_pattern="spectral_analysis.xlsx", enable_pca_export=False)
    assert df is not None
    g = pd.to_numeric(df["density_normalized_global"], errors="coerce")
    assert g.max() <= 1.0 + 1e-9
    assert g.min() >= 0.0 - 1e-9
    # density_normalized_global is the canonical density max-norm: [0.25, 1.0, 0.5]
    # for canonical=[2.0, 8.0, 4.0].
    import numpy as np
    np.testing.assert_allclose(
        g.to_numpy(dtype=float),
        np.array([0.25, 1.0, 0.5]),
        rtol=0.0, atol=1e-9,
    )
    a = pd.to_numeric(df["density_metric_normalized"], errors="coerce")
    # density_metric_normalized must also be bounded in [0, 1] but may
    # differ from density_normalized_global because it normalises a
    # different raw quantity (density_metric_raw, not canonical_density_v5_adapted).
    if a.notna().any():
        assert a.dropna().max() <= 1.0 + 1e-9
        assert a.dropna().min() >= 0.0 - 1e-9
    assert validate_compiled_density_workbook(outp) == []


def test_g3_a3_b3_c4_sequence_metadata_and_bounded_norm(tmp_path) -> None:
    root = tmp_path
    seq = [("G3", 5.329499568), ("A3", 5.579282334), ("B3", 1.541345581), ("C4", 1.031128718)]
    for i, (note, c) in enumerate(seq):
        d = root / f"Run_{note}_{i}"
        d.mkdir(parents=True)
        pd.DataFrame([_compile_row(note, canonical=c)]).to_excel(
            d / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False
        )
    outp = root / "seq.xlsx"
    df = compile_density_metrics(root, output_path=outp, file_pattern="spectral_analysis.xlsx", enable_pca_export=False)
    assert df is not None
    g = pd.to_numeric(df["density_normalized_global"], errors="coerce")
    assert (g <= 1.0 + 1e-9).all()
    fv = df["density_formula_version"].astype(str).str.strip()
    assert fv.nunique(dropna=False) == 1
    assert fv.iloc[0] == CANONICAL_DENSITY_FORMULA_VERSION
    fs = df["density_source_formula"].astype(str).str.strip()
    assert fs.nunique(dropna=False) == 1
    assert validate_compiled_density_workbook(outp) == []


def test_density_per_component_not_equal_to_canonical_when_count_gt_one(tmp_path) -> None:
    d = tmp_path / "Note_X1"
    d.mkdir(parents=True)
    cval = 12.0
    hoc = 4
    row = _compile_row("X1", canonical=cval, hoc=hoc)
    pd.DataFrame([row]).to_excel(d / "spectral_analysis.xlsx", sheet_name="Density_Metrics", index=False)
    df = compile_density_metrics(tmp_path, output_path=None, file_pattern="spectral_analysis.xlsx")
    assert df is not None
    r0 = df.iloc[0]
    assert r0["canonical_density_v5_adapted"] == pytest.approx(cval)
    assert r0["density_per_component"] == pytest.approx(cval / hoc)
