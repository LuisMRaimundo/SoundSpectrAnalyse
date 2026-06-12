from __future__ import annotations

"""
Second helper-level contract layer for metadata_sanitizer.py.

Complements ``test_metadata_sanitizer_additional.py`` with export-safety,
idempotence, URL preservation, Windows/POSIX edge strings, nested traversal,
and small in-memory DataFrame / workbook helpers. No production code changes.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

import metadata_sanitizer as ms


_WIN_DOCS = r"C:\Users\alice\Documents\Thèse\sample.wav"
_WIN_DL = r"C:\Users\bob\Downloads\run\clip.wav"
_WIN_MIXED = "C:/Users/carol/Desktop/projet/sample.wav"
_POSIX_TMP = "/tmp/ci-run-12345/output/note.json"
_POSIX_MNT = "/mnt/c/Users/dev/corpus/sample.wav"
_HOME = "/home/bob/project/data/sample.wav"
_MAC = "/Users/dana/Music/sample.wav"


def _assert_no_local_leaks(text: str, *, forbidden: tuple[str, ...] = ("alice", "bob", "carol", "dana", "dev")) -> None:
    low = text.lower()
    for frag in forbidden:
        assert frag not in low


# ---------------------------------------------------------------------------
# 1. Windows path redaction
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "path_text",
    [_WIN_DOCS, _WIN_DL, _WIN_MIXED, r"D:\Users\eve\Desktop\file.wav"],
)
def test_windows_user_and_special_folder_paths_are_redacted(path_text: str) -> None:
    redacted = ms.redact_path(path_text)
    assert redacted == ms.REDACT_TOKEN
    _assert_no_local_leaks(redacted)


def test_windows_path_with_accented_characters_is_redacted() -> None:
    assert ms.detect_absolute_local_path(_WIN_DOCS) is True
    assert ms.redact_path(_WIN_DOCS) == ms.REDACT_TOKEN


def test_mixed_slash_windows_path_is_treated_as_local_absolute() -> None:
    assert ms.is_absolute_path_like(_WIN_MIXED) is True
    assert ms.redact_path(_WIN_MIXED) == ms.REDACT_TOKEN


# ---------------------------------------------------------------------------
# 2. POSIX / temporary paths
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path_text", [_HOME, _MAC, _POSIX_MNT])
def test_posix_home_and_mnt_paths_are_detected_and_redacted(path_text: str) -> None:
    assert ms.detect_absolute_local_path(path_text) is True
    assert ms.redact_path(path_text) == ms.REDACT_TOKEN


def test_tmp_paths_are_detected_but_not_redacted_by_redact_path() -> None:
    """``detect_absolute_local_path`` flags /tmp; ``redact_path`` uses a narrower heuristic."""
    assert ms.detect_absolute_local_path(_POSIX_TMP) is True
    assert ms.redact_path(_POSIX_TMP) == _POSIX_TMP


def test_tilde_home_prefix_is_detected_as_local() -> None:
    assert ms.detect_absolute_local_path("~/Desktop/project") is True


# ---------------------------------------------------------------------------
# 3. URL / URI preservation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/org/SoundSpectrAnalyse",
        "http://example.com/audio.wav",
        "https://doi.org/10.1234/abc.def",
    ],
)
def test_http_and_doi_urls_are_not_treated_as_local_paths(url: str) -> None:
    assert ms.detect_absolute_local_path(url) is False
    assert ms.sanitize_export_metadata_value(url) == url
    assert ms.sanitize_metadata_value(url) == url


def test_file_uri_is_treated_as_local_path_material() -> None:
    uri = "file:///Users/alice/private.wav"
    assert ms.detect_absolute_local_path(uri) is True
    assert ms.sanitize_path_for_publication(uri) == "private.wav"


def test_scientific_provenance_strings_survive_free_text_sanitization() -> None:
    text = (
        "analysis_schema_version=v403; density_formula_version=canonical_v5; "
        "doi=https://doi.org/10.1234/abc"
    )
    cleaned = ms.sanitize_metadata_value(text)
    assert cleaned == text
    assert "doi.org" in cleaned
    assert "canonical_v5" in cleaned


# ---------------------------------------------------------------------------
# 4. Nested structures, export trees, Analysis_Metadata shape
# ---------------------------------------------------------------------------

def test_deeply_nested_metadata_is_sanitized_without_input_mutation() -> None:
    src = {
        "layers": {
            "runtime": {
                "paths": [
                    _WIN_DOCS,
                    {"output_dir": _HOME, "status": "validated_pipeline"},
                ],
                "metric": 0.875,
            }
        }
    }
    snapshot = json.loads(json.dumps(src))
    out = ms.sanitize_metadata_dict(src)
    assert src == snapshot
    assert out["layers"]["runtime"]["paths"][0] == ms.REDACT_TOKEN
    assert out["layers"]["runtime"]["paths"][1]["output_dir"] == ms.REDACT_TOKEN
    assert out["layers"]["runtime"]["paths"][1]["status"] == "validated_pipeline"
    assert out["layers"]["runtime"]["metric"] == 0.875


def test_analysis_metadata_like_rows_keep_public_labels_and_redact_runtime_paths() -> None:
    rows = [
        ("analysis_schema_version", "v403"),
        ("density_formula_version", "canonical_v5"),
        ("fit_status", "ok"),
        ("runtime_path", _WIN_DOCS),
    ]
    out = dict(ms.filter_analysis_meta_rows_publication_clean(rows))
    assert out["analysis_schema_version"] == "v403"
    assert out["density_formula_version"] == "canonical_v5"
    assert out["fit_status"] == "ok"
    assert out["runtime_path"] == "sample.wav"
    _assert_no_local_leaks(json.dumps(out), forbidden=("alice", "Documents", "Thèse"))


def test_export_file_path_branch_expands_public_audio_fields() -> None:
    obj = {"file_path": _MAC, "hop_length": 512}
    out = ms.sanitize_export_metadata_dict(obj)
    assert out["file_path"] == "sample.wav"
    assert out["audio_file_name"] == "sample.wav"
    assert out["hop_length"] == 512
    assert out["audio_path_policy"] == "sanitized_for_publication"


def test_export_working_directory_key_sanitizes_to_basename() -> None:
    out = ms.sanitize_export_metadata_dict({"working_directory": _POSIX_TMP})
    assert out["working_directory"] == "note.json"


def test_export_json_string_value_is_parsed_and_resanitized() -> None:
    payload = json.dumps({"file_path": _WIN_DL})
    cleaned = ms.sanitize_export_metadata_value(payload)
    parsed = json.loads(cleaned)
    assert parsed["file_path"] == "clip.wav"
    assert "Users" not in cleaned


# ---------------------------------------------------------------------------
# 5. DataFrame helper contracts
# ---------------------------------------------------------------------------

def test_dataframe_sanitizer_preserves_numeric_columns_and_nan() -> None:
    df = pd.DataFrame(
        {
            "Note": ["C4"],
            "canonical_density": [0.42],
            "file_path": [_WIN_DOCS],
            "missing": [float("nan")],
        }
    )
    out = ms.sanitize_dataframe_for_publication(df)
    assert out["canonical_density"].iloc[0] == pytest.approx(0.42)
    assert pd.isna(out["missing"].iloc[0])
    assert list(out.columns) == list(df.columns)
    assert out["file_path"].iloc[0] == ms.REDACT_TOKEN


def test_drop_publication_noise_removes_all_nan_numeric_column() -> None:
    df = pd.DataFrame({"Note": ["C4"], "all_nan_metric": [float("nan")]})
    out = ms.drop_publication_noise_columns_from_dataframe(df)
    assert list(out.columns) == ["Note"]


# ---------------------------------------------------------------------------
# 6. Edge cases and path-sensitive key typing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", ["", None, 42, True])
def test_redact_path_passthrough_for_non_path_strings(value: object) -> None:
    assert ms.redact_path(value) == value


def test_path_sensitive_dict_key_with_non_string_value_is_preserved() -> None:
    out = ms.sanitize_metadata_dict({"output_dir": 42, "note": "C4"})
    assert out["output_dir"] == 42
    assert out["note"] == "C4"


def test_long_string_with_multiple_embedded_paths_is_fully_redacted() -> None:
    text = f"before {_WIN_DOCS} middle {_HOME} after"
    cleaned = ms.sanitize_metadata_value(text)
    assert ms.REDACT_TOKEN in cleaned
    _assert_no_local_leaks(cleaned, forbidden=("alice", "bob", "Documents", "home"))


def test_path_like_token_without_separators_is_not_redacted() -> None:
    token = "analysis_schema_version_v403"
    assert ms.sanitize_metadata_value(token) == token
    assert ms.string_fails_publication_scan(token) is False


# ---------------------------------------------------------------------------
# 7. Determinism and idempotence
# ---------------------------------------------------------------------------

def test_metadata_dict_sanitization_is_idempotent() -> None:
    src = {"file_path": _WIN_DOCS, "nested": {"audio_path": _MAC}}
    once = ms.sanitize_metadata_dict(src)
    twice = ms.sanitize_metadata_dict(once)
    assert once == twice


def test_export_metadata_dict_sanitization_is_idempotent() -> None:
    src = {"audio_file": _WIN_DL, "note": "A4"}
    once = ms.sanitize_export_metadata_dict(src)
    twice = ms.sanitize_export_metadata_dict(once)
    assert once == twice


def test_redact_path_on_already_redacted_token_is_stable() -> None:
    assert ms.redact_path(ms.REDACT_TOKEN) == ms.REDACT_TOKEN


def test_repeated_sanitize_metadata_dict_produces_identical_output() -> None:
    src = {"file_path": _HOME, "status": "validated_pipeline"}
    results = [ms.sanitize_metadata_dict(src) for _ in range(3)]
    assert results[0] == results[1] == results[2]


# ---------------------------------------------------------------------------
# 8. Scan helpers and validator edge contracts
# ---------------------------------------------------------------------------

def test_string_fails_publication_scan_detects_mnt_and_users_patterns() -> None:
    assert ms.string_fails_publication_scan(_POSIX_MNT) is True
    assert ms.string_fails_publication_scan("harmonic_density") is False


def test_redact_public_path_leakage_strips_colon_suffixed_absolute_tail() -> None:
    leaked = ms.redact_public_path_leakage(f"source: {_WIN_DOCS}")
    assert leaked == "source"
    _assert_no_local_leaks(leaked)


def test_validate_no_private_paths_skips_local_debug_logs(tmp_path: Path) -> None:
    (tmp_path / "local_debug_logs").mkdir()
    (tmp_path / "local_debug_logs" / "debug.json").write_text(_WIN_DOCS, encoding="utf-8")
    (tmp_path / "clean.json").write_text('{"note":"C4"}', encoding="utf-8")
    ok, errors = ms.validate_no_private_paths(tmp_path)
    assert ok is True
    assert errors == []


def test_list_publication_path_violations_in_excel_and_inplace_sanitize(tmp_path: Path) -> None:
    openpyxl = pytest.importorskip("openpyxl")
    workbook = tmp_path / "meta.xlsx"
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Metadata"
    ws["A1"] = _WIN_DOCS
    wb.save(workbook)

    violations = ms.list_publication_path_violations_in_excel(workbook)
    assert violations
    assert "Metadata" in violations[0]

    ms.sanitize_excel_workbook_for_publication(workbook)
    wb2 = openpyxl.load_workbook(workbook)
    assert wb2["Metadata"]["A1"].value == ms.REDACT_TOKEN


# ---------------------------------------------------------------------------
# 9. Thesis / publication regression guards
# ---------------------------------------------------------------------------

def test_sanitized_export_tree_contains_no_windows_username_fragments() -> None:
    obj = {
        "input_file": _WIN_DOCS,
        "output_dir": _WIN_DL,
        "analysis_schema_version": "v403",
        "fit_status": "ok",
    }
    cleaned = ms.sanitize_export_metadata_dict(obj)
    blob = json.dumps(cleaned)
    _assert_no_local_leaks(blob, forbidden=("alice", "bob", "Users", "Documents", "Downloads"))
    assert cleaned["analysis_schema_version"] == "v403"
    assert cleaned["fit_status"] == "ok"


def test_numeric_metric_values_are_not_altered_by_dict_sanitizer() -> None:
    src = {
        "file_path": _MAC,
        "canonical_density": 0.418,
        "harmonic_count": 17,
    }
    out = ms.sanitize_metadata_dict(src)
    assert out["canonical_density"] == pytest.approx(0.418)
    assert out["harmonic_count"] == 17


def test_publication_audio_fields_handle_extensionless_windows_names() -> None:
    fields = ms.publication_audio_path_fields(r"C:\Users\a\audiofile")
    assert fields["audio_file_name"] == "audiofile"
    assert fields["audio_file_extension"] == ""
    assert ms.detect_absolute_local_path(fields["audio_relative_path"]) is False


def test_apply_publication_clean_meta_flat_neutralizes_orchestrator_status() -> None:
    out = ms.apply_publication_clean_meta_flat(
        {"input_schema_validation_status": "not_validated_orchestrator_v2", "density": 0.5}
    )
    assert out["input_schema_validation_status"] == "gui_orchestrator_pipeline"
    assert out["density"] == pytest.approx(0.5)
