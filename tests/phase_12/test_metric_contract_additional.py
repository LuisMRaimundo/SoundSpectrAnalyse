from __future__ import annotations

"""
Additional contract-level coverage for metric_contract.py.

Public API under test:
- ``MetricDefinition`` (frozen epistemic contract dataclass);
- ``build_metric_contracts`` / ``get_metric_definition`` (canonical registry);
- ``as_export_fields`` (workbook-row flattening);
- ``density_metric_basis_label`` (weight-function -> basis token);
- ``classify_f0_epistemic_status`` (tri-state f0 provenance gate).

Focus areas (no production code changes):
- stable canonical metric identifiers and the legacy-alias relationship
  (``density_weighted_sum`` is a declared alias of ``density_metric_raw``,
  same formula, distinct ontology family — no ambiguous naming);
- the exact canonical density formula token (consistent with the
  formula-validation suite);
- export-field flattening schema (prefix, key set, string values, empty
  fallback for unknown names, copy semantics);
- registry uniqueness, determinism, and immutability of definitions;
- basis-label tokens for all documented weight-function keys;
- the f0 epistemic tri-state classification across all documented branches.

All asserted values are formal tokens declared verbatim in the module, so
exact assertions are appropriate.
"""

import dataclasses

import pytest

from metric_contract import (
    MetricDefinition,
    as_export_fields,
    build_metric_contracts,
    classify_f0_epistemic_status,
    density_metric_basis_label,
    get_metric_definition,
)


_CANONICAL_FORMULA = "D_H*w_H + D_I*w_I + D_S*w_S"
_FIELD_NAMES = (
    "name",
    "formula",
    "input_domain",
    "unit_or_scale",
    "amplitude_basis",
    "power_basis",
    "normalization_scope",
    "physical_interpretation",
    "not_valid_for",
    "ontology_family",
)


# ---------------------------------------------------------------------------
# 1. Canonical identifiers and registry consistency
# ---------------------------------------------------------------------------

def test_registry_contains_exactly_the_canonical_identifiers() -> None:
    contracts = build_metric_contracts()
    assert set(contracts.keys()) == {"density_metric_raw", "density_weighted_sum"}
    # Key <-> definition-name consistency and uniqueness.
    for key, definition in contracts.items():
        assert definition.name == key
    names = [d.name for d in contracts.values()]
    assert len(names) == len(set(names))


def test_every_contract_field_is_a_non_empty_string() -> None:
    for definition in build_metric_contracts().values():
        for field in _FIELD_NAMES:
            value = getattr(definition, field)
            assert isinstance(value, str) and value.strip() != "", (
                definition.name,
                field,
            )


def test_canonical_density_formula_token() -> None:
    raw = get_metric_definition("density_metric_raw")
    assert raw is not None
    assert raw.formula == _CANONICAL_FORMULA
    # Three additive terms, one per H/I/S component channel.
    terms = [t.strip() for t in raw.formula.split("+")]
    assert len(terms) == 3
    assert terms == ["D_H*w_H", "D_I*w_I", "D_S*w_S"]


def test_legacy_alias_shares_formula_but_is_marked_legacy() -> None:
    raw = get_metric_definition("density_metric_raw")
    alias = get_metric_definition("density_weighted_sum")
    assert raw is not None and alias is not None
    # Alias contract: identical formula and bases, explicitly distinct
    # ontology family so it can never be mistaken for an independent metric.
    assert alias.formula == raw.formula
    assert alias.amplitude_basis == raw.amplitude_basis
    assert alias.power_basis == raw.power_basis
    assert raw.ontology_family == "composite_metric"
    assert alias.ontology_family == "legacy_only"
    assert "alias" in alias.physical_interpretation.lower()
    assert "density_metric_raw" in alias.not_valid_for


# ---------------------------------------------------------------------------
# 2. Lookup behaviour
# ---------------------------------------------------------------------------

def test_unknown_metric_lookup_returns_none() -> None:
    assert get_metric_definition("not_a_metric") is None
    # Non-string keys are coerced via str() and miss (current contract).
    assert get_metric_definition(123) is None  # type: ignore[arg-type]


def test_definitions_are_immutable() -> None:
    raw = get_metric_definition("density_metric_raw")
    assert raw is not None
    with pytest.raises(dataclasses.FrozenInstanceError):
        raw.formula = "tampered"  # type: ignore[misc]


def test_registry_builds_are_deterministic_and_independent() -> None:
    a = build_metric_contracts()
    b = build_metric_contracts()
    assert a == b
    # Fresh dict per call: mutating one build must not corrupt lookups.
    a.pop("density_metric_raw")
    assert "density_metric_raw" in build_metric_contracts()
    assert get_metric_definition("density_metric_raw") is not None


# ---------------------------------------------------------------------------
# 3. Export-field flattening
# ---------------------------------------------------------------------------

def test_export_fields_flatten_with_stable_prefix_and_full_schema() -> None:
    out = as_export_fields("density_metric_raw")
    assert set(out.keys()) == {f"metric_contract_{f}" for f in _FIELD_NAMES}
    for key, value in out.items():
        assert key.startswith("metric_contract_")
        assert isinstance(value, str) and value != ""
    assert out["metric_contract_name"] == "density_metric_raw"
    assert out["metric_contract_formula"] == _CANONICAL_FORMULA


def test_export_fields_unknown_metric_returns_empty_dict() -> None:
    assert as_export_fields("not_a_metric") == {}


def test_export_fields_are_fresh_copies() -> None:
    first = as_export_fields("density_metric_raw")
    first["metric_contract_formula"] = "tampered"
    second = as_export_fields("density_metric_raw")
    assert second["metric_contract_formula"] == _CANONICAL_FORMULA
    # Repeated calls are deterministic.
    assert second == as_export_fields("density_metric_raw")


# ---------------------------------------------------------------------------
# 4. Density basis labels
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("weight_function", "expected"),
    [
        ("log", "log-amplitude"),
        ("power", "power"),
        ("linear", "amplitude"),
        ("", "amplitude"),          # empty -> documented linear default
        (None, "amplitude"),        # None -> documented linear default
        ("  LOG  ", "log-amplitude"),  # case/whitespace-insensitive
        ("d17", "amplitude"),       # unknown keys fall back to amplitude
    ],
)
def test_density_metric_basis_labels(weight_function: object, expected: str) -> None:
    assert density_metric_basis_label(weight_function) == expected  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 5. f0 epistemic tri-state classification
# ---------------------------------------------------------------------------

def test_nominal_guided_accepted_fit_is_verified_and_primary_eligible() -> None:
    status, primary = classify_f0_epistemic_status(
        f0_fit_accepted=True,
        acoustic_f0_status="fit_accepted_acoustically_verified",
        f0_validation_mode="nominal_guided_f0_validation",
    )
    assert status == "nominal_guided_acoustically_verified"
    assert primary is True
    # Mode matching is case-insensitive.
    status_uc, primary_uc = classify_f0_epistemic_status(
        f0_fit_accepted=True,
        acoustic_f0_status="verified",
        f0_validation_mode="  NOMINAL_GUIDED_F0_VALIDATION  ",
    )
    assert status_uc == "nominal_guided_acoustically_verified" and primary_uc is True


@pytest.mark.parametrize(
    "acoustic_status",
    ["fit_accepted_acoustically_verified", "robust_fit", "accepted"],
)
def test_free_fit_accepted_statuses_are_verified(acoustic_status: str) -> None:
    status, primary = classify_f0_epistemic_status(
        f0_fit_accepted=True, acoustic_f0_status=acoustic_status
    )
    assert status == "free_fit_acoustically_verified"
    assert primary is True


@pytest.mark.parametrize(
    ("accepted", "acoustic_status"),
    [
        (False, "fit_accepted_acoustically_verified"),  # not accepted -> never verified
        (False, "nominal_fallback_used_not_acoustically_verified"),  # pipeline fallback pairing
        (False, ""),
        (True, "rejected_poor_fit"),                    # rejection keyword wins
        (True, "nominal_fallback_used"),                # fallback keyword wins
        (True, "weird_unrecognised_status"),            # final fallback branch
    ],
)
def test_non_verified_paths_collapse_to_nominal_fallback(
    accepted: bool, acoustic_status: str
) -> None:
    status, primary = classify_f0_epistemic_status(
        f0_fit_accepted=accepted, acoustic_f0_status=acoustic_status
    )
    assert status == "nominal_fallback_not_verified"
    assert primary is False


def test_accepted_fit_verified_keyword_takes_precedence() -> None:
    # Current precedence contract: when the caller asserts an ACCEPTED fit,
    # a status containing "verified" classifies as verified even if the
    # string also carries fallback wording. In the real pipeline this input
    # combination does not occur (fallback statuses are always paired with
    # f0_fit_accepted=False, covered above); this documents the precedence.
    status, primary = classify_f0_epistemic_status(
        f0_fit_accepted=True,
        acoustic_f0_status="nominal_fallback_used_not_acoustically_verified",
    )
    assert status == "free_fit_acoustically_verified"
    assert primary is True


def test_classification_returns_stable_types_and_is_deterministic() -> None:
    args = dict(
        f0_fit_accepted=True,
        acoustic_f0_status="fit_accepted_acoustically_verified",
        f0_validation_mode="nominal_guided_f0_validation",
    )
    a = classify_f0_epistemic_status(**args)  # type: ignore[arg-type]
    b = classify_f0_epistemic_status(**args)  # type: ignore[arg-type]
    assert a == b
    assert isinstance(a, tuple) and len(a) == 2
    assert isinstance(a[0], str) and isinstance(a[1], bool)
