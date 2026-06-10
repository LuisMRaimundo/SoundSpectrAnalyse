from __future__ import annotations

"""
Additional scientifically-motivated coverage for spectral_normalization.py.

Public API under test: ``n_fft_normalization_factor`` — the documented
multiplier that brings N-dependent spectral quantities onto the reference-N
scale:

    peak_amplitude_sum     -> N_ref / N
    peak_power_sum         -> (N_ref / N)^2
    broadband_amplitude_l2 -> sqrt(N_ref / N)
    broadband_power_l2     -> N_ref / N

Focus areas (no production code changes):
- exact canonical factors for all four quantity kinds and the identity case;
- amplitude/power consistency (power factor == amplitude factor squared);
- FFT-size invariance of model quantities normalised with the documented
  factor (the function's purpose);
- multiplicative chaining / inverse symmetry of the ratio family;
- invalid n_fft handling (non-positive -> ValueError; current conversion
  errors for NaN/inf/non-numeric asserted as-is);
- quantity-kind parsing (case/whitespace-insensitive, None -> default,
  unknown -> ValueError);
- legacy ``kind=`` keyword: equivalence with the canonical broadband paths,
  DeprecationWarning on every call, invalid legacy kind -> ValueError;
- linear scaling of normalised quantities and determinism.

The exact factor values are explicit formulas in the implementation, so
exact assertions are appropriate.
"""

import math

import pytest

from spectral_normalization import n_fft_normalization_factor


# ---------------------------------------------------------------------------
# 1. Canonical factors
# ---------------------------------------------------------------------------

def test_all_four_quantity_kinds_canonical_factors() -> None:
    # ratio = 8192 / 4096 = 2
    assert n_fft_normalization_factor(4096, 8192, "peak_amplitude_sum") == 2.0
    assert n_fft_normalization_factor(4096, 8192, "peak_power_sum") == 4.0
    assert n_fft_normalization_factor(4096, 8192, "broadband_amplitude_l2") == pytest.approx(
        math.sqrt(2.0), abs=1e-15
    )
    assert n_fft_normalization_factor(4096, 8192, "broadband_power_l2") == 2.0


@pytest.mark.parametrize(
    "qk",
    ["peak_amplitude_sum", "peak_power_sum", "broadband_amplitude_l2", "broadband_power_l2"],
)
def test_identity_at_reference_n_fft(qk: str) -> None:
    assert n_fft_normalization_factor(8192, 8192, qk) == 1.0


@pytest.mark.parametrize("n_fft", [1024, 2048, 4096, 8192, 16384])
def test_power_factor_is_square_of_amplitude_factor(n_fft: int) -> None:
    amp = n_fft_normalization_factor(n_fft, 8192, "peak_amplitude_sum")
    pwr = n_fft_normalization_factor(n_fft, 8192, "peak_power_sum")
    bb_amp = n_fft_normalization_factor(n_fft, 8192, "broadband_amplitude_l2")
    bb_pwr = n_fft_normalization_factor(n_fft, 8192, "broadband_power_l2")
    assert pwr == pytest.approx(amp * amp, rel=1e-12)
    assert bb_pwr == pytest.approx(bb_amp * bb_amp, rel=1e-12)
    # The amplitude and power paths stay distinct away from the reference N.
    if n_fft != 8192:
        assert pwr != amp


# ---------------------------------------------------------------------------
# 2. FFT-size invariance (the documented purpose of the factor)
# ---------------------------------------------------------------------------

def test_peak_quantities_normalised_to_reference_scale_are_nfft_invariant() -> None:
    # Model from the docstring: peak-bin magnitudes scale linearly with N
    # (A_N = c * N), peak powers quadratically (P_N = (c * N)^2). After the
    # documented multipliers both collapse to the reference-N value.
    c = 3.5e-4
    n_ref = 8192
    amp_at_ref = c * n_ref
    for n in (1024, 4096, 16384):
        a_raw = c * n
        p_raw = (c * n) ** 2
        a_norm = a_raw * n_fft_normalization_factor(n, n_ref, "peak_amplitude_sum")
        p_norm = p_raw * n_fft_normalization_factor(n, n_ref, "peak_power_sum")
        assert a_norm == pytest.approx(amp_at_ref, rel=1e-12), n
        assert p_norm == pytest.approx(amp_at_ref**2, rel=1e-12), n


def test_factor_chaining_and_inverse_symmetry() -> None:
    # The ratio family is multiplicative: f(a->b) * f(b->c) == f(a->c),
    # and normalising there-and-back is the identity.
    f_ab = n_fft_normalization_factor(1024, 4096, "peak_amplitude_sum")
    f_bc = n_fft_normalization_factor(4096, 16384, "peak_amplitude_sum")
    f_ac = n_fft_normalization_factor(1024, 16384, "peak_amplitude_sum")
    assert f_ab * f_bc == pytest.approx(f_ac, rel=1e-12)
    inv = n_fft_normalization_factor(4096, 8192, "peak_amplitude_sum")
    rev = n_fft_normalization_factor(8192, 4096, "peak_amplitude_sum")
    assert inv * rev == pytest.approx(1.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 3. Quantity-kind parsing
# ---------------------------------------------------------------------------

def test_quantity_kind_is_case_and_whitespace_insensitive() -> None:
    assert n_fft_normalization_factor(4096, 8192, "  Peak_Amplitude_Sum  ") == 2.0
    assert n_fft_normalization_factor(4096, 8192, "BROADBAND_POWER_L2") == 2.0


def test_quantity_kind_none_or_empty_defaults_to_peak_amplitude_sum() -> None:
    assert n_fft_normalization_factor(4096, 8192, None) == 2.0  # type: ignore[arg-type]
    assert n_fft_normalization_factor(4096, 8192, "") == 2.0
    # Default argument path matches the explicit canonical name.
    assert n_fft_normalization_factor(4096, 8192) == n_fft_normalization_factor(
        4096, 8192, "peak_amplitude_sum"
    )


def test_unknown_quantity_kind_raises_value_error() -> None:
    with pytest.raises(ValueError, match="quantity_kind"):
        n_fft_normalization_factor(4096, 8192, "bogus_kind")


# ---------------------------------------------------------------------------
# 4. Invalid n_fft inputs (current behaviour asserted as-is)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("n", "n_ref"), [(0, 8192), (-1, 8192), (4096, 0), (4096, -8)])
def test_non_positive_n_fft_raises_value_error(n: int, n_ref: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        n_fft_normalization_factor(n, n_ref)


def test_non_numeric_and_non_finite_n_fft_current_conversion_errors() -> None:
    with pytest.raises(ValueError):
        n_fft_normalization_factor("abc", 8192)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        n_fft_normalization_factor(None, 8192)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        n_fft_normalization_factor(float("nan"), 8192)  # type: ignore[arg-type]
    with pytest.raises(OverflowError):
        n_fft_normalization_factor(float("inf"), 8192)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. Legacy ``kind=`` keyword (backward compatibility)
# ---------------------------------------------------------------------------

def test_legacy_kind_aliases_agree_with_canonical_quantity_kinds() -> None:
    with pytest.warns(DeprecationWarning):
        legacy_amp = n_fft_normalization_factor(4096, 8192, kind="amplitude")
    with pytest.warns(DeprecationWarning):
        legacy_pwr = n_fft_normalization_factor(4096, 8192, kind="power")
    assert legacy_amp == n_fft_normalization_factor(4096, 8192, "broadband_amplitude_l2")
    assert legacy_pwr == n_fft_normalization_factor(4096, 8192, "broadband_power_l2")


def test_legacy_kind_is_case_insensitive_and_overrides_quantity_kind() -> None:
    with pytest.warns(DeprecationWarning):
        val = n_fft_normalization_factor(
            4096, 8192, quantity_kind="peak_amplitude_sum", kind="  AMPLITUDE  "
        )
    # The legacy keyword takes precedence (documented mapping).
    assert val == pytest.approx(math.sqrt(2.0), abs=1e-15)


def test_invalid_legacy_kind_raises_value_error() -> None:
    with pytest.raises(ValueError, match="kind must be one of"):
        n_fft_normalization_factor(4096, 8192, kind="magnitude")


# ---------------------------------------------------------------------------
# 6. Scaling and determinism
# ---------------------------------------------------------------------------

def test_normalisation_is_a_linear_multiplier() -> None:
    factor = n_fft_normalization_factor(4096, 8192, "peak_amplitude_sum")
    raw = 0.123
    for s in (2.0, 1e3):
        assert (s * raw) * factor == pytest.approx(s * (raw * factor), rel=1e-12)


def test_repeated_calls_are_deterministic_and_finite() -> None:
    for qk in (
        "peak_amplitude_sum",
        "peak_power_sum",
        "broadband_amplitude_l2",
        "broadband_power_l2",
    ):
        a = n_fft_normalization_factor(2048, 8192, qk)
        b = n_fft_normalization_factor(2048, 8192, qk)
        assert a == b
        assert math.isfinite(a) and a > 0.0
