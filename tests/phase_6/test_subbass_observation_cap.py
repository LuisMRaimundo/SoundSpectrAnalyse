from __future__ import annotations

import pandas as pd

from acoustic_density_core import compute_acoustic_density_descriptors


def test_obs_w_s_stays_below_005_on_harmonic_like_note() -> None:
    peaks = pd.DataFrame(
        {
            "frequency_hz": [220.0, 440.0, 660.0, 880.0, 1100.0],
            "power": [10.0, 5.0, 3.0, 2.0, 1.5],
        }
    )
    out = compute_acoustic_density_descriptors(
        peaks,
        f0_hz=220.0,
        f0_fit_accepted=True,
    )
    assert float(out["pure_observation_w_s"]) < 0.05


def test_energy_anchoring_suppresses_noise_floor_subbass_band() -> None:
    """v57 regression (cello C2 defect): a spectrally narrow sub-bass band
    populated only by weak noise-floor partials must NOT dominate the adaptive
    observation when it carries negligible energy.

    Reproduces the failure mode: strong harmonic comb (low f0, many orders) plus
    a cloud of very weak sub-bass particles. Pre-fix, the occupancy-only
    strength let the sub-bass band saturate and draw w_s ~ 0.5; post-fix the
    energy gate must keep w_s tiny and w_h the majority component.
    """
    f0 = 65.41  # cello C2
    harmonic_freqs = [f0 * n for n in range(1, 41)]
    harmonic_powers = [100.0 / n for n in range(1, 41)]  # strong, 1/n falloff
    # Weak sub-bass "noise floor" particles below f0/2 (~32 Hz).
    subbass_freqs = [21.0, 24.0, 27.0, 30.0]
    subbass_powers = [1e-4, 1e-4, 1e-4, 1e-4]
    peaks = pd.DataFrame(
        {
            "frequency_hz": harmonic_freqs + subbass_freqs,
            "power": harmonic_powers + subbass_powers,
        }
    )
    out = compute_acoustic_density_descriptors(
        peaks,
        f0_hz=f0,
        f0_fit_accepted=True,
        density_summation_mode="his_note_adaptive",
    )
    w_h = float(out["pure_observation_w_h"])
    w_s = float(out["pure_observation_w_s"])
    # The energy-empty sub-bass band must be suppressed, and harmonic must lead.
    assert w_s < 0.05, f"sub-bass over-weighted despite ~0 energy: w_s={w_s:.3f}"
    assert w_h > w_s
    assert w_h == max(w_h, float(out["pure_observation_w_i"]), w_s)
    # The energy gate for sub-bass must reflect its negligible energy share.
    assert float(out["component_strength_energy_gate_s"]) < 0.05
