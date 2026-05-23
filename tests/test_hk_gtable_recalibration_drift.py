from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from dissonance_models import HutchinsonKnopoffDissonance


def _load_hk_table(path: Path, *, clamp: bool) -> pd.DataFrame:
    df = pd.read_csv(path, comment="#")
    assert list(df.columns) == ["y", "g"], f"Unexpected columns in {path}: {list(df.columns)}"

    y = pd.to_numeric(df["y"], errors="raise").astype(float)
    g = pd.to_numeric(df["g"], errors="raise").astype(float)
    assert (np.diff(y.to_numpy()) >= 0.0).all(), f"y is not monotone non-decreasing in {path}"

    if clamp:
        g_clamped = g.clip(lower=0.0, upper=1.0)
        n_clamped = int((g_clamped != g).sum())
        if n_clamped:
            warnings.warn(f"Clamped {n_clamped} g-values in {path.name} into [0,1].", RuntimeWarning)
        g = g_clamped

    out = pd.DataFrame({"y": y, "g": g})
    return out


def _total_hk_dissonance(table_df: pd.DataFrame, freqs: np.ndarray, amps: np.ndarray) -> float:
    model = HutchinsonKnopoffDissonance(g_table=list(table_df.itertuples(index=False, name=None)))
    partials = list(zip(freqs.tolist(), amps.tolist()))
    return float(model.total_dissonance(partials, []))


def test_hk_gtable_recalibration_drift() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    parent_dir = repo_root.parent
    new_path = repo_root / "data" / "hk1978_g_table.csv"
    legacy_path = repo_root / "data" / "hk1978_g_table_legacy.csv"

    new_df = _load_hk_table(new_path, clamp=True)
    legacy_df = _load_hk_table(legacy_path, clamp=False)

    for name, df in [("new", new_df), ("legacy", legacy_df)]:
        assert len(df) >= 20, f"{name} HK table has fewer than 20 points"
        assert float(df["y"].min()) >= 0.0 and float(df["y"].max()) <= 1.5, f"{name} y out of range"
        assert float(df["g"].min()) >= 0.0 and float(df["g"].max()) <= 1.0, f"{name} g out of range"

    model_new = HutchinsonKnopoffDissonance(g_table=list(new_df.itertuples(index=False, name=None)))
    y_checks = [0.0, 0.25, 0.95, float(new_df["y"].max())]
    g_vals = [model_new.g(y) for y in y_checks]
    assert all(np.isfinite(g) for g in g_vals)
    assert model_new.g(0.25) == model_new.g(0.25)

    np.random.seed(20260101)
    n_partials = 20
    freqs = np.sort(np.random.uniform(200.0, 3000.0, n_partials))
    amps = np.random.uniform(0.05, 1.00, n_partials)

    new_total = _total_hk_dissonance(new_df, freqs, amps)
    legacy_total = _total_hk_dissonance(legacy_df, freqs, amps)

    assert np.isfinite(new_total) and new_total > 0.0
    assert np.isfinite(legacy_total) and legacy_total > 0.0

    drift_abs = abs(new_total - legacy_total)
    drift_rel = drift_abs / max(legacy_total, 1e-12)
    assert drift_rel <= 0.50

    out_json = parent_dir / "_hk_recalibration_drift.json"
    out_json.write_text(
        json.dumps(
            {
                "seed": 20260101,
                "n_partials": n_partials,
                "legacy_total_hk_dissonance": legacy_total,
                "new_total_hk_dissonance": new_total,
                "drift_abs": drift_abs,
                "drift_rel": drift_rel,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
