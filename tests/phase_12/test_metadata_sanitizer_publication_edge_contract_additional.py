from __future__ import annotations

"""
Publication-edge contract layer for metadata_sanitizer.py.

Complements existing Phase 12 metadata_sanitizer tests with helper-level
regression coverage for empty/partial/nested structures, home-directory
fragment leakage, scientific provenance preservation alongside path
redaction, internal-field omission, and idempotence across sanitizers.

No production code changes. No real audio, GUI, plotting, or E2E pipeline.
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest

import metadata_sanitizer as ms


_WIN = r"C:\Users\eve\Music\corpus\note.wav"
_POSIX = "/Users/frank/Desktop/project/note.wav"
_HOME = "/home/grace/data/note.wav"
_UNC = r"\\server\share\corpus\note.wav"

# Known privacy/export gaps — xfail records them without pinning weak output as contract.
SUSPICIOUS_CURRENT_BEHAVIOUR = pytest.mark.xfail(
    reason="Suspicious current behaviour — not a publication-safe contract",
    strict=False,
)


def _no_user_leaks(blob: str, *names: str) -> None:
    low = blob.lower()
    for name in names:
        assert name.lower() not in low


# ---------------------------------------------------------------------------
# Empty, partial, and missing structures
# ---------------------------------------------------------------------------


def test_sanitize_metadata_dict_empty_input_returns_empty_copy() -> None:
    src: dict[str, object] = {}
    out = ms.sanitize_metadata_dict(src)
    assert out == {}
    assert out is not src


def test_sanitize_export_metadata_dict_empty_and_none_containers() -> None:
    assert ms.sanitize_export_metadata_dict({}) == {}
    assert ms.sanitize_export_metadata_dict(None) is None
    assert ms.sanitize_export_metadata_dict([]) == []


def test_filter_analysis_meta_rows_empty_list_is_deterministic() -> None:
    assert ms.filter_analysis_meta_rows_publication_clean([]) == []


def test_apply_publication_clean_meta_flat_partial_scientific_only() -> None:
    meta = {
        "analysis_schema_version": "v403",
        "density_formula_version": "canonical_v5",
        "doi": "https://doi.org/10.1234/abc.def",
        "canonical_density": 0.55,
    }
    out = ms.apply_publication_clean_meta_flat(meta)
    assert out == meta
    assert "cwd" not in out and "proc_audio_file" not in out


def test_apply_publication_clean_research_metadata_fields_partial_empty_paths() -> None:
    rows = {"note": "C4", "compiled_from": "", "output_path": None}
    out = ms.apply_publication_clean_research_metadata_fields(
        rows, workbook_basename="research.xlsx"
    )
    assert out["note"] == "C4"
    assert out["research_export_source_workbook"] == "research.xlsx"
    assert out["compiled_from"] == ""
    assert out["output_path"] is None


def test_sanitize_dataframe_for_publication_empty_frame_passthrough() -> None:
    empty = pd.DataFrame()
    assert ms.sanitize_dataframe_for_publication(empty) is empty


def test_drop_publication_noise_columns_none_dataframe_passthrough() -> None:
    assert ms.drop_publication_noise_columns_from_dataframe(None) is None


# ---------------------------------------------------------------------------
# Nested None, NaN, and partial trees
# ---------------------------------------------------------------------------


def test_sanitize_metadata_value_preserves_none_and_nan_scalars() -> None:
    assert ms.sanitize_metadata_value(None) is None
    nan = float("nan")
    out = ms.sanitize_metadata_value(nan)
    assert isinstance(out, float) and math.isnan(out)


def test_nested_partial_metadata_redacts_paths_preserves_scientific_fields() -> None:
    src = {
        "provenance": {
            "doi": "https://doi.org/10.5678/xyz",
            "analysis_schema_version": "v403",
            "density_formula_version": "canonical_v5",
            "local_path": _WIN,
        },
        "metrics": {"canonical_density": 0.33, "harmonic_count": 12},
        "optional": None,
    }
    snapshot = json.loads(json.dumps(src, default=str))
    out = ms.sanitize_metadata_dict(src)
    assert src == snapshot
    assert out["provenance"]["doi"] == src["provenance"]["doi"]
    assert out["provenance"]["analysis_schema_version"] == "v403"
    assert out["provenance"]["density_formula_version"] == "canonical_v5"
    assert out["provenance"]["local_path"] == ms.REDACT_TOKEN
    assert out["metrics"]["canonical_density"] == pytest.approx(0.33)
    assert out["metrics"]["harmonic_count"] == 12
    assert out["optional"] is None


def test_export_metadata_nested_partial_tree_strips_internal_paths() -> None:
    obj = {
        "run": {
            "input_path": _HOME,
            "schema_version": "v403",
            "formula_version": "canonical_v5",
            "note": None,
        },
        "dataset_root": "/secret/corpus",
    }
    out = ms.sanitize_export_metadata_dict(obj)
    assert "dataset_root" not in out
    assert out["run"]["schema_version"] == "v403"
    assert out["run"]["formula_version"] == "canonical_v5"
    assert out["run"]["note"] is None
    assert out["run"]["input_path"] == "note.wav"
    _no_user_leaks(json.dumps(out), "grace", "home")


def test_dataframe_with_nan_path_and_numeric_cells_is_stable() -> None:
    df = pd.DataFrame(
        {
            "Note": ["D4"],
            "canonical_density": [0.71],
            "file_path": [float("nan")],
            "doi": ["https://doi.org/10.9999/test"],
        }
    )
    out = ms.sanitize_dataframe_for_publication(df)
    assert out["canonical_density"].iloc[0] == pytest.approx(0.71)
    assert pd.isna(out["file_path"].iloc[0])
    assert out["doi"].iloc[0] == "https://doi.org/10.9999/test"


# ---------------------------------------------------------------------------
# Path-key fragments, home/username leakage (publication-safe contracts)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "audio_path",
        "output_directory",
        "results_directory",
        "parent_directory",
        "compiled_excel_path",
    ],
)
def test_path_sensitive_key_fragments_always_redact_absolute_strings(key: str) -> None:
    out = ms.sanitize_metadata_dict({key: _WIN, "note": "E4"})
    assert out[key] == ms.REDACT_TOKEN
    assert out["note"] == "E4"


def test_detect_private_path_leakage_uses_username_hints(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("USERNAME", "HelenH")
    text = r"D:\Projects\HelenH\corpus\note.wav"
    assert ms.detect_private_path_leakage_fragment(text) is True


@SUSPICIOUS_CURRENT_BEHAVIOUR
def test_unc_paths_should_be_redacted_for_publication_exports() -> None:
    """Heuristic gap: UNC host paths must not be treated as publication-safe passthrough."""
    assert ms.redact_path(_UNC) == ms.REDACT_TOKEN
    assert ms.detect_absolute_local_path(_UNC) is True


@SUSPICIOUS_CURRENT_BEHAVIOUR
def test_redact_public_path_leakage_should_not_leave_windows_drive_fragments() -> None:
    """Colon-partition branch must not emit partial host material such as drive letters."""
    leaked = ms.redact_public_path_leakage(r"stored under C:\Users\eve\Desktop\runs")
    assert leaked in (ms.REDACT_TOKEN, "redacted_local_path")
    assert ms.string_fails_publication_scan(leaked) is False
    _no_user_leaks(leaked, "eve", "Desktop", "Users")


def test_posix_path_handling_in_export_sanitizer_preserves_basename_only() -> None:
    out = ms.sanitize_export_metadata_dict({"source_path": _POSIX})
    assert out["source_path"] == "note.wav"
    _no_user_leaks(json.dumps(out), "frank", "Desktop")


def test_windows_path_handling_in_metadata_value_embedded_in_provenance_text() -> None:
    text = f"run exported from {_WIN}; doi=https://doi.org/10.1/2"
    cleaned = ms.sanitize_metadata_value(text)
    assert ms.REDACT_TOKEN in cleaned
    assert "doi.org" in cleaned
    _no_user_leaks(cleaned, "eve", "Music")


# ---------------------------------------------------------------------------
# Internal/local path fields removed per publication-clean contract
# ---------------------------------------------------------------------------


def test_publication_flat_meta_omits_all_internal_dev_keys() -> None:
    meta = {k: f"/secret/{k}" for k in ms.PUBLICATION_META_FLAT_KEYS_OMIT}
    meta["analysis_schema_version"] = "v403"
    out = ms.apply_publication_clean_meta_flat(meta)
    assert set(out.keys()) == {"analysis_schema_version"}
    assert out["analysis_schema_version"] == "v403"


def test_analysis_meta_rows_omit_internal_parameters_but_keep_schema_version() -> None:
    rows = [
        ("analysis_schema_version", "v403"),
        ("density_formula_version", "canonical_v5"),
        ("sys_executable", r"C:\Python312\python.exe"),
        ("cwd", _WIN),
        ("proc_audio_file", "/secret/proc_audio.py"),
    ]
    out = dict(ms.filter_analysis_meta_rows_publication_clean(rows))
    assert out["analysis_schema_version"] == "v403"
    assert out["density_formula_version"] == "canonical_v5"
    assert "sys_executable" not in out
    assert "cwd" not in out
    assert "proc_audio_file" not in out


# ---------------------------------------------------------------------------
# Idempotence across sanitizers
# ---------------------------------------------------------------------------


def test_sanitize_metadata_value_string_idempotent() -> None:
    once = ms.sanitize_metadata_value(_WIN)
    twice = ms.sanitize_metadata_value(once)
    assert once == twice == ms.REDACT_TOKEN


def test_sanitize_dataframe_for_publication_is_idempotent() -> None:
    df = pd.DataFrame({"file_path": [_WIN], "density": [0.9]})
    once = ms.sanitize_dataframe_for_publication(df)
    twice = ms.sanitize_dataframe_for_publication(once)
    pd.testing.assert_frame_equal(once, twice)


def test_filter_analysis_meta_rows_publication_clean_is_idempotent() -> None:
    rows = [
        ("platform", "Linux x86_64"),
        ("custom_path", _POSIX),
        ("doi", "https://doi.org/10.0/1"),
    ]
    once = ms.filter_analysis_meta_rows_publication_clean(rows)
    twice = ms.filter_analysis_meta_rows_publication_clean(once)
    assert once == twice


@SUSPICIOUS_CURRENT_BEHAVIOUR
def test_apply_publication_clean_meta_flat_should_be_idempotent_on_platform_label() -> None:
    """Second pass must not degrade an already-coarse platform label."""
    meta = {"platform": "Darwin 21", "density": 0.4, "cwd": _HOME}
    once = ms.apply_publication_clean_meta_flat(meta)
    twice = ms.apply_publication_clean_meta_flat(once)
    assert twice == once
    assert twice["platform"] == "macOS"


def test_sanitize_run_parameters_json_is_idempotent() -> None:
    raw = json.dumps({"file_path": _WIN, "schema_version": "v403"})
    once = ms.sanitize_run_parameters_json(raw)
    twice = ms.sanitize_run_parameters_json(once)
    assert once == twice
    assert json.loads(twice)["schema_version"] == "v403"


# ---------------------------------------------------------------------------
# Input non-mutation and provenance sufficiency
# ---------------------------------------------------------------------------


def test_sanitize_export_metadata_dict_does_not_mutate_input_mapping() -> None:
    src = {"audio_file": _WIN, "doi": "https://doi.org/10.42/x"}
    snapshot = dict(src)
    out = ms.sanitize_export_metadata_dict(src)
    assert src == snapshot
    assert out["doi"] == snapshot["doi"]
    assert out["audio_file"] == "note.wav"


def test_publication_research_metadata_retains_provenance_without_host_paths() -> None:
    rows = {
        "doi": "https://doi.org/10.5555/pub",
        "analysis_schema_version": "v403",
        "density_formula_version": "canonical_v5",
        "folder_path": _WIN,
        "canonical_density": 0.62,
    }
    out = ms.apply_publication_clean_research_metadata_fields(
        rows, workbook_basename="compiled.xlsx"
    )
    assert out["doi"] == rows["doi"]
    assert out["analysis_schema_version"] == "v403"
    assert out["density_formula_version"] == "canonical_v5"
    assert out["canonical_density"] == pytest.approx(0.62)
    assert out["folder_path"] == "note.wav"
    _no_user_leaks(json.dumps(out), "eve")


def test_enrich_and_redact_batch_audio_result_does_not_mutate_source_row(tmp_path: Path) -> None:
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"x")
    row = {"density": 0.25, "doi": "https://doi.org/10.1/1"}
    snapshot = dict(row)
    out = ms.enrich_and_redact_batch_audio_result(row, audio, note_name="G3")
    assert row == snapshot
    assert out["density"] == pytest.approx(0.25)
    assert out["doi"] == snapshot["doi"]
    assert "source_file_basename" in out


def test_sanitize_path_for_publication_with_dataset_root_on_posix_string(
    tmp_path: Path,
) -> None:
    root = tmp_path / "corpus"
    audio = root / "notes" / "note.wav"
    audio.parent.mkdir(parents=True)
    audio.write_bytes(b"x")
    rel = ms.sanitize_path_for_publication(str(audio.resolve()), dataset_root=root)
    assert rel in {"notes/note.wav", "note.wav"}
    fields = ms.publication_audio_path_fields(str(audio.resolve()), dataset_root=root)
    assert ms.detect_absolute_local_path(fields["audio_relative_path"]) is False
