"""Architectural tests: single publication-facing compile path (no acoustic drift)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_contract_module() -> None:
    import pipeline_contract as pc

    c = pc.get_canonical_pipeline_contract()
    assert c.contract_version == pc.PIPELINE_CONTRACT_VERSION
    assert c.stage1_module == "proc_audio"
    assert c.stage1_class == "AudioProcessor"
    assert c.stage2_module == "compile_metrics"
    assert c.stage2_function == "compile_density_metrics_with_pca"
    assert c.per_note_workbook == "spectral_analysis.xlsx"
    assert c.compiled_workbook == "compiled_density_metrics.xlsx"


def test_publication_flags_modules() -> None:
    import compile_metrics as cm
    import proc_audio as pa

    assert pa.PUBLICATION_OUTPUT_ALLOWED is True
    assert cm.PUBLICATION_OUTPUT_ALLOWED is True


def test_legacy_analyzer_flags() -> None:
    path = REPO_ROOT / "audio_analysis" / "super_audio_analyzer.py"
    spec = importlib.util.spec_from_file_location("super_audio_analyzer", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert getattr(mod, "PUBLICATION_OUTPUT_ALLOWED", None) is False
    assert getattr(mod, "CANONICAL_PIPELINE_ROLE", "") == "legacy_diagnostic"


def test_legacy_batch_flags() -> None:
    import sys

    audio_dir = str(REPO_ROOT / "audio_analysis")
    if audio_dir not in sys.path:
        sys.path.insert(0, audio_dir)
    path = REPO_ROOT / "audio_analysis" / "batch_audio_analyzer.py"
    spec = importlib.util.spec_from_file_location("batch_audio_analyzer", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    assert getattr(mod, "PUBLICATION_OUTPUT_ALLOWED", None) is False
    assert getattr(mod, "CANONICAL_PIPELINE_ROLE", "") == "legacy_batch_wrapper"
    try:
        sys.path.remove(audio_dir)
    except ValueError:
        pass


def test_orchestrator_v2_stage2_default_no_super_json(tmp_path: Path) -> None:
    from pipeline_orchestrator_gui import resolve_stage2_compile_file_pattern

    legacy = tmp_path / "super_analysis_results.json"
    legacy.write_text("{}", encoding="utf-8")
    assert resolve_stage2_compile_file_pattern(tmp_path, allow_legacy_super_json=False) is None
    assert resolve_stage2_compile_file_pattern(tmp_path, allow_legacy_super_json=True) == (
        "super_analysis_results.json"
    )


def test_compile_density_metrics_delegates_to_with_pca(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import compile_metrics as cm

    called: dict[str, bool] = {"canonical": False}

    def _fake(*args: object, **kwargs: object) -> pd.DataFrame:
        called["canonical"] = True
        return pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(cm, "compile_density_metrics_with_pca", _fake)
    out = cm.compile_density_metrics(folder_path=tmp_path, output_path=None)
    assert called["canonical"]
    assert isinstance(out, pd.DataFrame)


def test_compiled_workbook_analysis_metadata_keys(tmp_path: Path) -> None:
    import compile_metrics as cm

    outp = tmp_path / "compiled_density_metrics.xlsx"
    df = pd.DataFrame({"Note": ["C4"], "Density Metric": [1.0]})
    meta: dict = {"file_pattern": "spectral_analysis.xlsx"}
    cm._write_compiled_excel(
        outp,
        df,
        meta,
        apply_publication_column_filter=False,
        enable_pca_export=False,
        compile_file_pattern="spectral_analysis.xlsx",
        allow_legacy_super_json=False,
        input_schema_validation_status="test_fixture",
    )
    assert outp.is_file()
    wide = pd.read_excel(outp, sheet_name="Analysis_Metadata")
    row = wide.iloc[0].to_dict()
    keys = {str(k) for k in row.keys()}
    for k in (
        "pipeline_contract_version",
        "stage1_module",
        "stage1_class",
        "stage2_function",
        "legacy_pipeline_used",
    ):
        assert k in keys
    assert str(row.get("legacy_pipeline_used", "")).lower() in ("false", "0")


def test_super_audio_analyzer_defines_no_central_policy_tokens() -> None:
    text = (REPO_ROOT / "audio_analysis" / "super_audio_analyzer.py").read_text(encoding="utf-8")
    forbidden = (
        "MISSING_METRIC_POLICY_VERSION",
        "LOW_FREQUENCY_POLICY_VERSION",
        "NONHARMONIC_POLICY_VERSION",
        "HARMONIC_FREQUENCY_POLICY_VERSION",
    )
    for token in forbidden:
        assert token not in text


def test_canonical_compile_from_xlsx_not_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage-2 resolver must not pick JSON when canonical xlsx exists."""
    from pipeline_orchestrator_gui import resolve_stage2_compile_file_pattern

    (tmp_path / "sub" / "n1").mkdir(parents=True)
    xlsx = tmp_path / "sub" / "n1" / "spectral_analysis.xlsx"
    xlsx.write_bytes(b"fake")
    (tmp_path / "sub" / "n1" / "super_analysis_results.json").write_text("{}", encoding="utf-8")
    assert resolve_stage2_compile_file_pattern(tmp_path, allow_legacy_super_json=False) == (
        "spectral_analysis.xlsx"
    )
