from __future__ import annotations

"""
Additional contract-level coverage for pipeline_contract.py.

The module is purely declarative: module-level canonical tokens, a frozen
``PipelineContract`` dataclass, and ``get_canonical_pipeline_contract``.
It exposes no row/payload validators, so the meaningful contract surface is:

- the exact pipeline contract version token (bumping it must be deliberate);
- stage module / class / function names that must resolve to real importable
  objects (the contract is the single source of truth for which code defines
  publication-grade metrics — a stale name would silently break Stage 1/2/3
  hand-offs);
- EWSD vs density-stage separation (EWSD modules namespaced under
  ``tools.ewsd_*``, distinct from the Stage 2 density compiler);
- canonical artefact (workbook) filenames consumed across the Phase 7/11
  suites and the compile/export code;
- dataclass-default <-> module-constant agreement, frozen immutability,
  schema field set, uniqueness, and factory determinism.

Complements tests/phase_11/test_pipeline_contract_ewsd.py (which loosely pins
the three EWSD v18.1 module names). All values asserted exactly are formal
tokens declared verbatim in the module.
"""

import dataclasses
import importlib

import pytest

import pipeline_contract as pc
from pipeline_contract import PipelineContract, get_canonical_pipeline_contract


_EXPECTED_FIELDS = (
    "contract_version",
    "stage1_module",
    "stage1_class",
    "stage2_module",
    "stage2_function",
    "stage3_module",
    "stage3_function",
    "stage3_ewsd_pure",
    "stage3_ewsd_uncertainty",
    "stage3_ewsd_contract",
    "stage3_ewsd_core",
    "stage3_ewsd_integration",
    "per_note_workbook",
    "compiled_workbook",
    "research_workbook",
    "publication_output_allowed",
)


# ---------------------------------------------------------------------------
# 1. Version token and schema field set
# ---------------------------------------------------------------------------

def test_contract_version_is_the_exact_documented_token() -> None:
    assert pc.PIPELINE_CONTRACT_VERSION == (
        "SSA_CANONICAL_PIPELINE_2026_06_STAGE1_STAGE2_STAGE3_EWSD_v18_1_UQ"
    )
    c = get_canonical_pipeline_contract()
    assert c.contract_version == pc.PIPELINE_CONTRACT_VERSION


def test_contract_schema_field_set_is_stable() -> None:
    c = get_canonical_pipeline_contract()
    payload = dataclasses.asdict(c)
    assert tuple(payload.keys()) == _EXPECTED_FIELDS
    # All identity fields are non-empty strings; the publication gate is bool.
    for name, value in payload.items():
        if name == "publication_output_allowed":
            assert isinstance(value, bool)
        else:
            assert isinstance(value, str) and value.strip() != "", name
    assert c.publication_output_allowed is True


def test_dataclass_defaults_agree_with_module_constants() -> None:
    c = get_canonical_pipeline_contract()
    expected = {
        "contract_version": pc.PIPELINE_CONTRACT_VERSION,
        "stage1_module": pc.CANONICAL_STAGE1_MODULE,
        "stage1_class": pc.CANONICAL_STAGE1_CLASS,
        "stage2_module": pc.CANONICAL_STAGE2_MODULE,
        "stage2_function": pc.CANONICAL_STAGE2_FUNCTION,
        "stage3_module": pc.CANONICAL_STAGE3_MODULE,
        "stage3_function": pc.CANONICAL_STAGE3_FUNCTION,
        "stage3_ewsd_pure": pc.CANONICAL_STAGE3_EWSD_PURE,
        "stage3_ewsd_uncertainty": pc.CANONICAL_STAGE3_EWSD_UNCERTAINTY,
        "stage3_ewsd_contract": pc.CANONICAL_STAGE3_EWSD_CONTRACT,
        "stage3_ewsd_core": pc.CANONICAL_STAGE3_EWSD_CORE,
        "stage3_ewsd_integration": pc.CANONICAL_STAGE3_EWSD_INTEGRATION,
        "per_note_workbook": pc.CANONICAL_PER_NOTE_WORKBOOK,
        "compiled_workbook": pc.CANONICAL_COMPILED_WORKBOOK,
        "research_workbook": pc.CANONICAL_RESEARCH_WORKBOOK,
    }
    for field, value in expected.items():
        assert getattr(c, field) == value, field


# ---------------------------------------------------------------------------
# 2. Stage identity tokens resolve to real code (downstream compatibility)
# ---------------------------------------------------------------------------

def test_stage1_and_stage2_tokens_resolve_to_importable_objects() -> None:
    c = get_canonical_pipeline_contract()
    stage1 = importlib.import_module(c.stage1_module)
    assert hasattr(stage1, c.stage1_class)
    assert isinstance(getattr(stage1, c.stage1_class), type)
    stage2 = importlib.import_module(c.stage2_module)
    assert callable(getattr(stage2, c.stage2_function))


def test_stage3_and_ewsd_tokens_resolve_to_importable_objects() -> None:
    c = get_canonical_pipeline_contract()
    stage3 = importlib.import_module(c.stage3_module)
    assert callable(getattr(stage3, c.stage3_function))
    for field in (
        "stage3_ewsd_pure",
        "stage3_ewsd_uncertainty",
        "stage3_ewsd_contract",
        "stage3_ewsd_core",
        "stage3_ewsd_integration",
    ):
        module_path = getattr(c, field)
        assert importlib.import_module(module_path) is not None, field


# ---------------------------------------------------------------------------
# 3. EWSD vs density-stage separation; uniqueness
# ---------------------------------------------------------------------------

def test_ewsd_modules_are_namespaced_and_not_conflated_with_density_stage() -> None:
    c = get_canonical_pipeline_contract()
    ewsd_fields = (
        c.stage3_ewsd_pure,
        c.stage3_ewsd_uncertainty,
        c.stage3_ewsd_contract,
        c.stage3_ewsd_core,
        c.stage3_ewsd_integration,
    )
    # EWSD lives under the tools.ewsd namespace, distinct from the Stage 1/2
    # density modules — the contract keeps the two metric families separate.
    for module_path in ewsd_fields:
        assert module_path.startswith("tools.ewsd"), module_path
        assert module_path != c.stage2_module
        assert module_path != c.stage1_module
    assert c.stage2_module == "compile_metrics"
    assert c.stage1_module == "proc_audio"
    # No duplicates among the module identity tokens.
    modules = (c.stage1_module, c.stage2_module, c.stage3_module, *ewsd_fields)
    assert len(modules) == len(set(modules))


def test_canonical_workbook_artefact_names() -> None:
    c = get_canonical_pipeline_contract()
    # Exact artefact tokens consumed by the Stage 2/3 hand-off and by the
    # Phase 7/11 suites (file_pattern / compiled / research workbooks).
    assert c.per_note_workbook == "spectral_analysis.xlsx"
    assert c.compiled_workbook == "compiled_density_metrics.xlsx"
    assert c.research_workbook == "compiled_density_metrics_research.xlsx"
    names = (c.per_note_workbook, c.compiled_workbook, c.research_workbook)
    assert len(names) == len(set(names))
    for name in names:
        assert name.endswith(".xlsx")
    # The research workbook is the compiled workbook's declared sibling.
    assert c.research_workbook == c.compiled_workbook.replace(
        ".xlsx", "_research.xlsx"
    )


# ---------------------------------------------------------------------------
# 4. Immutability, determinism, equality semantics
# ---------------------------------------------------------------------------

def test_contract_is_frozen() -> None:
    c = get_canonical_pipeline_contract()
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.publication_output_allowed = False  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.stage2_module = "tampered"  # type: ignore[misc]


def test_factory_is_deterministic_with_value_equality() -> None:
    a = get_canonical_pipeline_contract()
    b = get_canonical_pipeline_contract()
    assert a == b
    assert dataclasses.asdict(a) == dataclasses.asdict(b)
    # Fresh instances each call: frozen value objects, identity not shared.
    assert a is not b
    # Mutating an asdict() snapshot cannot corrupt subsequent contracts.
    snapshot = dataclasses.asdict(a)
    snapshot["stage2_module"] = "tampered"
    assert get_canonical_pipeline_contract().stage2_module == "compile_metrics"


def test_explicit_construction_matches_canonical_defaults() -> None:
    # PipelineContract() with no arguments IS the canonical contract.
    assert PipelineContract() == get_canonical_pipeline_contract()
