from __future__ import annotations

"""
Additional scientifically-motivated coverage for inharmonicity_model.py.

Canonical model under test (Fletcher 1962, stiff string):

    f_n = n * f0 * sqrt(1 + B * n^2)

Focus areas (no production code changes):
- pure harmonic series -> B ~ 0, zero residual, "ok" status;
- known-B recovery with relative tolerance and stretch monotonicity in B;
- joint (f0, B) estimation recovers the true fundamental from a sharp seed;
- register invariance of the dimensionless B coefficient;
- insufficient / invalid / degenerate inputs return the documented
  "insufficient_partials" fallback structure without crashing;
- garbage candidate entries (NaN/inf/non-positive) are ignored exactly;
- input-order invariance and duplicate-partial robustness;
- residual semantics (nonnegative, finite, ordered by perturbation);
- conservative rejection gate on spectrally incoherent series;
- f0 relocation guard: the joint fit refines, never relocates, f0.

Property-style and metamorphic assertions are preferred. The only exact
values asserted are the documented fallback defaults (B = 0.0, NaN f0/residual,
empty prediction grid, "insufficient_partials") and canonical relations
directly implied by the implemented formula.
"""

import numpy as np
import pytest

from inharmonicity_model import fit_inharmonicity_coefficient


def _stiff_string_series(f0_hz: float, b: float, n_orders: int) -> np.ndarray:
    orders = np.arange(1, n_orders + 1, dtype=float)
    return orders * f0_hz * np.sqrt(1.0 + b * orders**2)


def _assert_insufficient_fallback(fit: dict) -> None:
    """The documented no-fit structure: explicit status, neutral B, NaN values."""
    assert fit["fit_status"] == "insufficient_partials"
    assert float(fit["inharmonicity_coefficient_B"]) == 0.0
    assert np.isnan(float(fit["inharmonicity_fit_f0_hz"]))
    assert np.isnan(float(fit["fit_residual_std_cents"]))
    assert np.asarray(fit["stretched_harmonic_predicted_freqs_hz"]).size == 0
    assert str(fit["method"]) != ""


# ---------------------------------------------------------------------------
# 1. Pure harmonic series
# ---------------------------------------------------------------------------

def test_pure_harmonic_series_full_diagnostics() -> None:
    f0 = 110.0
    fit = fit_inharmonicity_coefficient(
        _stiff_string_series(f0, 0.0, 20), f0_hz=f0, order_cap=40
    )
    assert fit["fit_status"] == "ok"
    # The n^4 significance gate must suppress spurious B on exact harmonics.
    assert float(fit["inharmonicity_coefficient_B"]) < 1e-9
    assert float(fit["inharmonicity_fit_f0_hz"]) == pytest.approx(f0, rel=1e-9)
    res = float(fit["fit_residual_std_cents"])
    assert np.isfinite(res) and 0.0 <= res <= 0.1
    # Prediction grid: one entry per order up to order_cap, canonical formula.
    grid = np.asarray(fit["stretched_harmonic_predicted_freqs_hz"], dtype=float)
    assert grid.size == 40
    assert np.all(np.isfinite(grid))
    assert np.all(np.diff(grid) > 0.0)
    n = np.arange(1, 41, dtype=float)
    b_fit = float(fit["inharmonicity_coefficient_B"])
    f0_fit = float(fit["inharmonicity_fit_f0_hz"])
    expected_grid = n * f0_fit * np.sqrt(1.0 + b_fit * n**2)
    assert grid == pytest.approx(expected_grid, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Known-B recovery and stretch monotonicity
# ---------------------------------------------------------------------------

def test_known_B_recovery_and_monotonic_stretch() -> None:
    f0 = 110.0
    b_low, b_high = 5e-5, 5e-4
    series_low = _stiff_string_series(f0, b_low, 16)
    series_high = _stiff_string_series(f0, b_high, 16)
    # Physical premise: larger B stretches every partial upward.
    assert np.all(series_high > series_low)

    fit_low = fit_inharmonicity_coefficient(series_low, f0_hz=f0)
    fit_high = fit_inharmonicity_coefficient(series_high, f0_hz=f0)
    assert fit_low["fit_status"] == "ok"
    assert fit_high["fit_status"] == "ok"
    b_low_fit = float(fit_low["inharmonicity_coefficient_B"])
    b_high_fit = float(fit_high["inharmonicity_coefficient_B"])
    assert b_low_fit == pytest.approx(b_low, rel=0.25)
    assert b_high_fit == pytest.approx(b_high, rel=0.25)
    # Monotonicity of the recovered coefficient.
    assert b_high_fit > b_low_fit
    # The fitted prediction grid is more stretched for the higher coefficient.
    grid_low = np.asarray(fit_low["stretched_harmonic_predicted_freqs_hz"])
    grid_high = np.asarray(fit_high["stretched_harmonic_predicted_freqs_hz"])
    assert grid_high[-1] / grid_high[0] > grid_low[-1] / grid_low[0]


def test_joint_fit_recovers_true_f0_from_sharp_seed() -> None:
    # Raison d'etre of the joint (f0, B) fit: a seed that drifted sharp must
    # not masquerade as inharmonicity, and the fitted f0 must return to truth.
    f0_true = 110.0
    fit = fit_inharmonicity_coefficient(
        _stiff_string_series(f0_true, 0.0, 20), f0_hz=f0_true * 1.01
    )
    assert fit["fit_status"] == "ok"
    assert float(fit["inharmonicity_fit_f0_hz"]) == pytest.approx(f0_true, rel=1e-3)
    assert float(fit["inharmonicity_coefficient_B"]) < 1e-9


def test_B_is_register_invariant_for_identical_stretch() -> None:
    # B is dimensionless: the same B at f0 and at 8*f0 (three octaves up,
    # an exact power-of-two scale) must fit to the same coefficient.
    b_true = 2e-4
    fit_low = fit_inharmonicity_coefficient(
        _stiff_string_series(55.0, b_true, 16), f0_hz=55.0
    )
    fit_high = fit_inharmonicity_coefficient(
        _stiff_string_series(440.0, b_true, 16), f0_hz=440.0
    )
    assert fit_low["fit_status"] == fit_high["fit_status"] == "ok"
    assert float(fit_low["inharmonicity_coefficient_B"]) == pytest.approx(
        float(fit_high["inharmonicity_coefficient_B"]), rel=1e-9
    )


# ---------------------------------------------------------------------------
# 3. Insufficient data
# ---------------------------------------------------------------------------

def test_empty_candidate_array_returns_documented_fallback() -> None:
    fit = fit_inharmonicity_coefficient(np.asarray([], dtype=float), f0_hz=110.0)
    _assert_insufficient_fallback(fit)


def test_two_partials_are_underdetermined_and_fail_safely() -> None:
    fit = fit_inharmonicity_coefficient(np.asarray([110.0, 220.0]), f0_hz=110.0)
    _assert_insufficient_fallback(fit)


# ---------------------------------------------------------------------------
# 4. Invalid or degenerate inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad_f0", ["not-a-number", None])
def test_non_numeric_f0_returns_documented_fallback(bad_f0: object) -> None:
    fit = fit_inharmonicity_coefficient(
        _stiff_string_series(110.0, 0.0, 10), f0_hz=bad_f0  # type: ignore[arg-type]
    )
    _assert_insufficient_fallback(fit)


@pytest.mark.parametrize("bad_f0", [0.0, -10.0, float("nan"), float("inf")])
def test_non_positive_or_non_finite_f0_returns_documented_fallback(bad_f0: float) -> None:
    fit = fit_inharmonicity_coefficient(
        _stiff_string_series(110.0, 0.0, 10), f0_hz=bad_f0
    )
    _assert_insufficient_fallback(fit)


def test_candidates_with_no_valid_frequency_return_documented_fallback() -> None:
    fit = fit_inharmonicity_coefficient(
        np.asarray([float("nan"), float("inf"), -50.0, 0.0]), f0_hz=110.0
    )
    _assert_insufficient_fallback(fit)


def test_garbage_candidate_entries_are_ignored_exactly() -> None:
    # Metamorphic: NaN / inf / non-positive entries must not influence the fit.
    clean = _stiff_string_series(110.0, 1e-4, 16)
    dirty = np.concatenate([[float("nan"), float("inf"), -50.0, 0.0], clean])
    fit_clean = fit_inharmonicity_coefficient(clean, f0_hz=110.0)
    fit_dirty = fit_inharmonicity_coefficient(dirty, f0_hz=110.0)
    assert fit_clean["fit_status"] == fit_dirty["fit_status"] == "ok"
    assert float(fit_clean["inharmonicity_coefficient_B"]) == float(
        fit_dirty["inharmonicity_coefficient_B"]
    )
    assert float(fit_clean["inharmonicity_fit_f0_hz"]) == float(
        fit_dirty["inharmonicity_fit_f0_hz"]
    )
    assert float(fit_clean["fit_residual_std_cents"]) == float(
        fit_dirty["fit_residual_std_cents"]
    )


def test_duplicate_partials_do_not_break_the_fit() -> None:
    # Each harmonic order consumes at most one peak; exact duplicates must be
    # absorbed without corrupting B on a perfectly harmonic series.
    duplicated = np.repeat(_stiff_string_series(110.0, 0.0, 8), 2)
    fit = fit_inharmonicity_coefficient(duplicated, f0_hz=110.0)
    assert fit["fit_status"] == "ok"
    assert float(fit["inharmonicity_coefficient_B"]) < 1e-9


def test_candidate_order_is_irrelevant() -> None:
    # The order matcher scans all candidates per harmonic order, so a shuffled
    # input list is part of the public contract and must fit identically.
    series = _stiff_string_series(110.0, 1e-4, 20)
    shuffled = series.copy()
    np.random.default_rng(7).shuffle(shuffled)
    fit_sorted = fit_inharmonicity_coefficient(series, f0_hz=110.0)
    fit_shuffled = fit_inharmonicity_coefficient(shuffled, f0_hz=110.0)
    assert fit_sorted["fit_status"] == fit_shuffled["fit_status"]
    assert float(fit_sorted["inharmonicity_coefficient_B"]) == float(
        fit_shuffled["inharmonicity_coefficient_B"]
    )
    assert float(fit_sorted["fit_residual_std_cents"]) == float(
        fit_shuffled["fit_residual_std_cents"]
    )


# ---------------------------------------------------------------------------
# 5. Residual semantics
# ---------------------------------------------------------------------------

def test_residuals_are_nonnegative_finite_and_ordered_by_perturbation() -> None:
    f0 = 110.0
    pure = _stiff_string_series(f0, 0.0, 20)
    rng = np.random.default_rng(11)
    jitter_cents = rng.uniform(-10.0, 10.0, size=pure.size)
    perturbed = pure * np.power(2.0, jitter_cents / 1200.0)

    fit_pure = fit_inharmonicity_coefficient(pure, f0_hz=f0)
    fit_pert = fit_inharmonicity_coefficient(perturbed, f0_hz=f0)
    assert fit_pure["fit_status"] == "ok"
    assert fit_pert["fit_status"] == "ok"
    res_pure = float(fit_pure["fit_residual_std_cents"])
    res_pert = float(fit_pert["fit_residual_std_cents"])
    for res in (res_pure, res_pert):
        assert np.isfinite(res) and res >= 0.0
    # A deliberately perturbed series cannot fit tighter than the exact one.
    assert res_pert > res_pure
    # +/-10 cents of frequency jitter cannot legitimately produce a residual
    # spread above the jitter magnitude scale.
    assert res_pert <= 10.0


# ---------------------------------------------------------------------------
# 6. Rejection gate and guards
# ---------------------------------------------------------------------------

def test_incoherent_series_is_rejected_as_poor_fit() -> None:
    # Five partials with large alternating-sign detune (within the matching
    # window but spectrally incoherent): no stiff-string model can absorb
    # alternating stretch, so the conservative gate must reject the fit.
    orders = np.arange(1, 6, dtype=float)
    detune_cents = np.array([70.0, -70.0, 55.0, -55.0, 70.0])
    freqs = orders * 110.0 * np.power(2.0, detune_cents / 1200.0)
    fit = fit_inharmonicity_coefficient(freqs, f0_hz=110.0, cents_window=80.0)
    assert fit["fit_status"] == "rejected_poor_fit"
    res = float(fit["fit_residual_std_cents"])
    # Documented gate: ok requires res <= max(25, cents_window * 0.5) = 40.
    assert np.isfinite(res) and res > 40.0
    assert float(fit["inharmonicity_coefficient_B"]) >= 0.0


def test_f0_relocation_guard_keeps_fit_anchored_to_seed() -> None:
    # Candidates are exact harmonics of 3x the seed. With a permissive window
    # they all match low orders, and an unguarded joint fit would relocate f0
    # to ~3x the seed. The documented guard confines the fitted f0 to
    # [0.5, 2.0] x seed (refine, not relocate).
    seed = 100.0
    freqs = _stiff_string_series(3.0 * seed, 0.0, 5)
    fit = fit_inharmonicity_coefficient(
        freqs, f0_hz=seed, order_cap=10, cents_window=2000.0
    )
    f0_fit = float(fit["inharmonicity_fit_f0_hz"])
    assert 0.5 * seed <= f0_fit <= 2.0 * seed
    assert f0_fit == pytest.approx(seed, rel=1e-9)
    assert float(fit["inharmonicity_coefficient_B"]) >= 0.0
    assert np.isfinite(float(fit["fit_residual_std_cents"]))


def test_sparse_heavily_jittered_series_keeps_finite_diagnostics() -> None:
    # Four partials with +/-70 cent mixed detune: iterative re-matching may
    # collapse below 3 partials after the first (f0, B) update. The fit must
    # then fall back to the last valid matched set with finite diagnostics
    # instead of crashing or emitting non-finite values.
    orders = np.arange(1, 5, dtype=float)
    detune_cents = np.array([70.0, -70.0, -70.0, 70.0])
    freqs = orders * 110.0 * np.power(2.0, detune_cents / 1200.0)
    fit = fit_inharmonicity_coefficient(freqs, f0_hz=110.0, cents_window=80.0)
    assert fit["fit_status"] in ("ok", "rejected_poor_fit")
    assert np.isfinite(float(fit["fit_residual_std_cents"]))
    assert float(fit["inharmonicity_coefficient_B"]) >= 0.0
    f0_fit = float(fit["inharmonicity_fit_f0_hz"])
    assert np.isfinite(f0_fit) and 0.5 * 110.0 <= f0_fit <= 2.0 * 110.0
