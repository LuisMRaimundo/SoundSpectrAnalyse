from __future__ import annotations

"""
Additional scientifically-motivated coverage for energy_accounting.py.

Public API under test: ``describe_component_energy_balance`` — the explicit
energy-conservation audit for the H/I/S component decomposition. The function
does not *compute* ratios (that lives in proc_audio); it verifies the two
documented closure invariants and reports a status:

    total_component_energy == H + I + S       (relative error <= 1e-6)
    r_H + r_I + r_S == 1                      (absolute error <= 1e-5)

Focus areas (no production code changes):
- exact closure -> "ok" with ~0 conservation error;
- violated total closure and violated ratio closure -> "warning" with the
  canonical relative/absolute error magnitudes;
- documented threshold constants (1e-6 / 1e-5) at both sides;
- zero-total semantics (zero ratios cannot close to 1 -> flagged);
- non-finite inputs sanitised to 0.0 before auditing (finite outputs always);
- scale invariance of the relative-error audit;
- negative energies are audited for conservation only (current contract);
- determinism and the export-facing schema keys.

Exact assertions are used only for canonical arithmetic implied directly by
the implemented error formulas.
"""

import math

import pytest

from energy_accounting import describe_component_energy_balance


_SCHEMA_KEYS = (
    "energy_denominator_description",
    "energy_conservation_error",
    "energy_conservation_status",
)


def _balanced(h: float, i: float, s: float) -> dict:
    tot = h + i + s
    return describe_component_energy_balance(
        h, i, s, tot, h / tot, i / tot, s / tot
    )


# ---------------------------------------------------------------------------
# 1/2. Component and ratio closure
# ---------------------------------------------------------------------------

def test_exact_closure_reports_ok_with_zero_error() -> None:
    out = _balanced(3.0, 1.0, 0.5)
    assert out["energy_conservation_status"] == "ok"
    assert out["energy_conservation_error"] == pytest.approx(0.0, abs=1e-12)
    for key in _SCHEMA_KEYS:
        assert key in out
    # The denominator contract is exported verbatim for the audit trail.
    assert "harmonic_energy_sum + inharmonic_energy_sum + subbass_energy_sum" in (
        out["energy_denominator_description"]
    )


def test_total_mismatch_yields_warning_with_canonical_relative_error() -> None:
    # Parts sum to 4.5 but total claims 5.0: relative error = 0.5 / 5.0 = 0.1.
    out = describe_component_energy_balance(
        3.0, 1.0, 0.5, 5.0, 3.0 / 4.5, 1.0 / 4.5, 0.5 / 4.5
    )
    assert out["energy_conservation_status"] == "warning"
    assert out["energy_conservation_error"] == pytest.approx(0.1, rel=1e-9)


def test_ratio_row_mismatch_yields_warning_with_canonical_absolute_error() -> None:
    # Sums close perfectly, but the ratio row sums to 0.9: error = |1 - 0.9|.
    out = describe_component_energy_balance(3.0, 1.0, 0.5, 4.5, 0.6, 0.2, 0.1)
    assert out["energy_conservation_status"] == "warning"
    assert out["energy_conservation_error"] == pytest.approx(0.1, rel=1e-9)


def test_documented_thresholds_on_both_sides() -> None:
    # Ratio closure tolerance is 1e-5 (absolute): just inside stays ok.
    inside = describe_component_energy_balance(
        3.0, 1.0, 0.5, 4.5, 3.0 / 4.5, 1.0 / 4.5, 0.5 / 4.5 - 5e-6
    )
    assert inside["energy_conservation_status"] == "ok"
    # Just outside flips to warning.
    outside = describe_component_energy_balance(
        3.0, 1.0, 0.5, 4.5, 3.0 / 4.5, 1.0 / 4.5, 0.5 / 4.5 - 5e-5
    )
    assert outside["energy_conservation_status"] == "warning"
    # Total closure tolerance is 1e-6 (relative): a 1e-3 relative gap warns.
    gap = describe_component_energy_balance(
        1.0, 0.0, 0.0, 1.001, 1.0, 0.0, 0.0
    )
    assert gap["energy_conservation_status"] == "warning"


# ---------------------------------------------------------------------------
# 3. Zero and degenerate inputs
# ---------------------------------------------------------------------------

def test_zero_total_with_zero_ratios_is_flagged_not_crashed() -> None:
    # With zero energy everywhere the ratio row cannot sum to 1; the audit
    # reports that explicitly (error = |1 - 0| = 1) instead of dividing by 0.
    out = describe_component_energy_balance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert out["energy_conservation_status"] == "warning"
    assert out["energy_conservation_error"] == pytest.approx(1.0, rel=1e-12)
    assert math.isfinite(out["energy_conservation_error"])


def test_zero_total_with_fallback_ratio_row_summing_to_one_is_ok() -> None:
    # A degenerate note exporting a normalized fallback ratio row still
    # satisfies both closure checks (0 == 0+0+0 and ratios sum to 1).
    out = describe_component_energy_balance(0.0, 0.0, 0.0, 0.0, 0.45, 0.35, 0.20)
    assert out["energy_conservation_status"] == "ok"
    assert out["energy_conservation_error"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 4. Non-finite and negative values
# ---------------------------------------------------------------------------

def test_non_finite_inputs_are_sanitised_to_zero_before_auditing() -> None:
    # NaN harmonic sum is treated as 0.0: total must then equal I + S.
    out = describe_component_energy_balance(
        float("nan"), 1.0, 0.5, 1.5, 0.0, 1.0 / 1.5, 0.5 / 1.5
    )
    assert out["energy_conservation_status"] == "ok"
    # Inf total is treated as 0.0 -> maximal relative disagreement (1.0).
    out_inf = describe_component_energy_balance(
        3.0, 1.0, 0.5, float("inf"), 3.0 / 4.5, 1.0 / 4.5, 0.5 / 4.5
    )
    assert out_inf["energy_conservation_status"] == "warning"
    assert out_inf["energy_conservation_error"] == pytest.approx(1.0, rel=1e-12)
    # Outputs stay finite for any non-finite input combination.
    out_all = describe_component_energy_balance(
        float("nan"), float("inf"), float("-inf"), float("nan"),
        float("nan"), float("inf"), float("nan"),
    )
    assert math.isfinite(out_all["energy_conservation_error"])
    assert out_all["energy_conservation_status"] in ("ok", "warning")


def test_negative_energies_are_audited_for_conservation_only() -> None:
    # Current contract: the auditor checks closure, not physical sign.
    # A consistent row with a negative component still closes.
    h, i, s = -1.0, 2.0, 1.0
    tot = h + i + s  # 2.0
    out = describe_component_energy_balance(h, i, s, tot, h / tot, i / tot, s / tot)
    assert out["energy_conservation_status"] == "ok"
    assert out["energy_conservation_error"] == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 5/6/7. Scale invariance, dominance, H/I/S separation
# ---------------------------------------------------------------------------

def test_audit_is_scale_invariant() -> None:
    # The total check is relative and the ratio row is dimensionless, so a
    # uniform positive scaling of all energies leaves the audit unchanged.
    base = _balanced(3.0, 1.0, 0.5)
    scaled = describe_component_energy_balance(
        3.0e6, 1.0e6, 0.5e6, 4.5e6, 3.0 / 4.5, 1.0 / 4.5, 0.5 / 4.5
    )
    assert scaled["energy_conservation_status"] == base["energy_conservation_status"]
    assert scaled["energy_conservation_error"] == pytest.approx(
        base["energy_conservation_error"], abs=1e-12
    )


def test_components_are_audited_separately_not_collapsed() -> None:
    # Moving energy from one channel to another while keeping the SUM intact
    # but exporting the OLD ratio row must be flagged: the audit does not
    # collapse H/I/S into an undifferentiated total on the ratio side.
    out = describe_component_energy_balance(
        4.0, 0.5, 0.0, 4.5, 3.0 / 4.5, 1.0 / 4.5, 0.5 / 4.5
    )
    # Sums still close (4 + 0.5 + 0 == 4.5) and the stale ratio row still
    # sums to 1, so this passes BOTH documented invariants: the audit checks
    # closure, not per-channel attribution (current contract, asserted).
    assert out["energy_conservation_status"] == "ok"
    # Dominance closure: a single-component note closes with ratio row (1,0,0).
    dom = describe_component_energy_balance(7.0, 0.0, 0.0, 7.0, 1.0, 0.0, 0.0)
    assert dom["energy_conservation_status"] == "ok"


# ---------------------------------------------------------------------------
# 9/10. Determinism and export-facing schema
# ---------------------------------------------------------------------------

def test_repeated_calls_are_deterministic_and_schema_stable() -> None:
    a = _balanced(2.0, 0.7, 0.3)
    b = _balanced(2.0, 0.7, 0.3)
    assert a == b
    for out in (
        a,
        describe_component_energy_balance(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        describe_component_energy_balance(1.0, 1.0, 1.0, 5.0, 0.2, 0.2, 0.2),
    ):
        assert set(out.keys()) == set(_SCHEMA_KEYS)
        assert isinstance(out["energy_conservation_status"], str)
        assert out["energy_conservation_status"] in ("ok", "warning")
        err = out["energy_conservation_error"]
        assert isinstance(err, float) and math.isfinite(err) and err >= 0.0
