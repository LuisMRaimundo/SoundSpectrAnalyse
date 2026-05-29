from __future__ import annotations

import pandas as pd

from acoustic_density_core import compute_acoustic_density_descriptors


def _synthetic_peaks(f0_hz: float) -> pd.DataFrame:
    freqs = [f0_hz, 2.0 * f0_hz, 3.0 * f0_hz, 1.37 * f0_hz, 0.42 * f0_hz]
    powers = [1.0, 0.45, 0.2, 0.03, 0.01]
    return pd.DataFrame({"frequency_hz": freqs, "power": powers})


def test_obs_w_formula_version_is_tagged() -> None:
    out = compute_acoustic_density_descriptors(
        _synthetic_peaks(220.0),
        f0_hz=220.0,
        f0_fit_accepted=True,
        density_summation_mode="his_note_adaptive",
    )
    assert out.get("obs_w_formula_version") == "v57_energy_anchored_occupancy"
