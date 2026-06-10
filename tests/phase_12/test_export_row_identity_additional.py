from __future__ import annotations

"""
Additional identity-contract coverage for export_row_identity.py.

Public API under test:
- ``compute_sample_id`` — deterministic per-row slug ``<slug>__<sha256[:12]>``
  built from (note, source-file stem, row index);
- ``assign_sample_ids`` / ``sample_id_fully_populated`` — column-level
  assignment and population checks;
- ``attach_sample_id_from_density`` — authoritative ID propagation from
  Density_Metrics onto satellite sheets;
- ``primary_merge_keys`` / ``merge_keys_for_frames`` — merge-key selection;
- ``drop_dead_columns`` / ``dedupe_identical_columns`` — export hygiene.

Focus areas (no production code changes):
- determinism and collision resistance of the sample id (note, stem, and
  row index all participate; directory and extension do not);
- slug sanitisation, 80-char truncation (digest still distinguishes), and
  the "sample" fallback for fully non-alphanumeric inputs;
- assignment short-circuits (existing non-blank ids preserved verbatim),
  source-column aliases, empty-frame pass-through;
- blank-like sample_id detection ("", "nan", "none", "<NA>", NaN);
- satellite attachment: creation, blank-only filling, column placement
  after Note, duplicate-Note last-wins policy, no-op guards;
- merge-key selection across all documented preference branches;
- dead-column pruning guards (protected names, all-zero numerics kept);
- suffix-duplicate cleanup (numeric and string paths, missing base kept);
- copy semantics (inputs never mutated).

Exact assertions are used only for formal tokens (slug format, key lists)
and documented deterministic behaviour.
"""

import re

import pandas as pd
import pytest

from export_row_identity import (
    DEAD_COLUMN_PROTECTED_NAMES,
    assign_sample_ids,
    attach_sample_id_from_density,
    compute_sample_id,
    dedupe_identical_columns,
    drop_dead_columns,
    merge_keys_for_frames,
    primary_merge_keys,
    sample_id_fully_populated,
)


_SAMPLE_ID_RE = re.compile(r"^[a-zA-Z0-9._-]+__[0-9a-f]{12}$")


# ---------------------------------------------------------------------------
# 1. Deterministic identity and collision resistance
# ---------------------------------------------------------------------------

def test_sample_id_is_deterministic_and_well_formed() -> None:
    a = compute_sample_id(note="C4", source_file_name="Piano_C4_mf.wav", row_index=3)
    b = compute_sample_id(note="C4", source_file_name="Piano_C4_mf.wav", row_index=3)
    assert a == b
    assert _SAMPLE_ID_RE.match(a), a


def test_each_identity_bearing_field_changes_the_id() -> None:
    base = compute_sample_id(note="C4", source_file_name="a.wav", row_index=0)
    assert compute_sample_id(note="D4", source_file_name="a.wav", row_index=0) != base
    assert compute_sample_id(note="C4", source_file_name="b.wav", row_index=0) != base
    assert compute_sample_id(note="C4", source_file_name="a.wav", row_index=1) != base


def test_directory_and_extension_do_not_participate_in_identity() -> None:
    # Identity uses the source-file stem (basename without extension).
    # Directories and extensions must not affect the hash key.
    base = compute_sample_id(note="C4", source_file_name="a.wav", row_index=0)
    assert compute_sample_id(note="C4", source_file_name="runs/a.wav", row_index=0) == base
    assert compute_sample_id(note="C4", source_file_name="C:\\data\\a.wav", row_index=0) == base
    assert compute_sample_id(note="C4", source_file_name="a.aiff", row_index=0) == base


@pytest.mark.parametrize(
    "source_file_name",
    [
        "sample.wav",
        "runs/sample.wav",
        "/dir/sample.wav",
        "C:\\folder\\sample.wav",
        "D:/archive/sample.aiff",
        "sample.aiff",
    ],
)
def test_cross_platform_path_strings_share_identity_for_same_basename(
    source_file_name: str,
) -> None:
    # POSIX and Windows-style path strings must reduce to the same stem and
    # therefore the same sample_id when note and row index match.
    reference = compute_sample_id(note="G3", source_file_name="sample.wav", row_index=2)
    assert compute_sample_id(note="G3", source_file_name=source_file_name, row_index=2) == reference


def test_slug_sanitisation_truncation_and_fallback() -> None:
    # Unsafe characters collapse to underscores; the slug stays filesystem-safe.
    messy = compute_sample_id(note="C4", source_file_name="weird name (1)!.wav", row_index=0)
    assert _SAMPLE_ID_RE.match(messy), messy
    # Slugs are truncated at 80 chars, but the digest still distinguishes
    # stems that only differ beyond the truncation point.
    long_a = "x" * 90 + "alpha"
    long_b = "x" * 90 + "beta"
    id_a = compute_sample_id(note="C4", source_file_name=f"{long_a}.wav", row_index=0)
    id_b = compute_sample_id(note="C4", source_file_name=f"{long_b}.wav", row_index=0)
    slug_a, digest_a = id_a.rsplit("__", 1)
    slug_b, digest_b = id_b.rsplit("__", 1)
    assert len(slug_a) <= 80 and len(slug_b) <= 80
    assert slug_a == slug_b          # truncation collides on the slug...
    assert digest_a != digest_b      # ...but the hash keeps rows distinct.
    # Fully non-alphanumeric inputs fall back to the documented "sample" slug.
    anonymous = compute_sample_id(note="###", source_file_name="", row_index=0)
    assert anonymous.startswith("sample__")
    assert compute_sample_id(note="", source_file_name="", row_index=0).startswith("sample__")


# ---------------------------------------------------------------------------
# 2. Column-level assignment
# ---------------------------------------------------------------------------

def test_assign_sample_ids_passthrough_for_empty_frame() -> None:
    empty = pd.DataFrame()
    assert assign_sample_ids(empty) is empty


def test_assign_sample_ids_preserves_existing_non_blank_ids() -> None:
    df = pd.DataFrame({"Note": ["C4", "D4"], "sample_id": ["keep__000000000001", ""]})
    out = assign_sample_ids(df)
    # Current contract: ANY non-blank id short-circuits reassignment, so the
    # frame is returned with existing values verbatim (blanks included).
    assert out["sample_id"].tolist() == ["keep__000000000001", ""]


def test_assign_sample_ids_recognises_filename_alias_and_does_not_mutate_input() -> None:
    df = pd.DataFrame({"Note": ["D3", "D3"], "filename": ["a.wav", "b.wav"]})
    out = assign_sample_ids(df)
    assert "sample_id" not in df.columns  # input untouched (copy semantics)
    assert out["sample_id"].nunique() == 2
    expected_first = compute_sample_id(note="D3", source_file_name="a.wav", row_index=0)
    assert out["sample_id"].iloc[0] == expected_first


# ---------------------------------------------------------------------------
# 3. Blank-like detection
# ---------------------------------------------------------------------------

def test_sample_id_fully_populated_detects_blank_like_markers() -> None:
    assert sample_id_fully_populated(pd.DataFrame({"sample_id": ["a__000000000000"]})) is True
    assert sample_id_fully_populated(pd.DataFrame()) is False
    assert sample_id_fully_populated(pd.DataFrame({"Note": ["C4"]})) is False
    for blank in ("", "nan", "None", "<NA>", float("nan")):
        df = pd.DataFrame({"sample_id": ["ok__000000000000", blank]})
        assert sample_id_fully_populated(df) is False, repr(blank)


# ---------------------------------------------------------------------------
# 4. Satellite attachment from Density_Metrics
# ---------------------------------------------------------------------------

def _density_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"Note": ["C4", "D4"], "sample_id": ["c4__000000000001", "d4__000000000002"]}
    )


def test_attach_creates_sample_id_column_on_satellite_without_one() -> None:
    satellite = pd.DataFrame({"Note": ["C4", "D4"], "metric": [1.0, 2.0]})
    out = attach_sample_id_from_density(satellite, _density_frame())
    assert out["sample_id"].tolist() == ["c4__000000000001", "d4__000000000002"]
    # Current contract: on the creation path the merge appends the column at
    # the end (the post-Note reordering only runs on the fill path below,
    # where the suffix collision is detected).
    assert list(out.columns) == ["Note", "metric", "sample_id"]


def test_attach_fills_only_blank_ids_and_places_column_after_note() -> None:
    satellite = pd.DataFrame(
        {
            "metric": [1.0, 2.0],
            "Note": ["C4", "D4"],
            "sample_id": ["already__000000000009", "nan"],
        }
    )
    out = attach_sample_id_from_density(satellite, _density_frame())
    assert out["sample_id"].tolist() == ["already__000000000009", "d4__000000000002"]
    # Documented layout on the fill path: sample_id sits immediately after Note.
    cols = list(out.columns)
    assert cols.index("sample_id") == cols.index("Note") + 1


def test_attach_no_op_guards() -> None:
    empty = pd.DataFrame()
    assert attach_sample_id_from_density(empty, _density_frame()) is empty
    satellite = pd.DataFrame({"Note": ["C4"], "metric": [1.0]})
    assert attach_sample_id_from_density(satellite, pd.DataFrame()) is satellite
    # Fully populated satellites are returned unchanged.
    populated = pd.DataFrame({"Note": ["C4"], "sample_id": ["x__000000000003"]})
    assert attach_sample_id_from_density(populated, _density_frame()) is populated
    # Without a Note key on the satellite, nothing can be joined.
    no_note = pd.DataFrame({"metric": [1.0], "sample_id": [""]})
    out = attach_sample_id_from_density(no_note, _density_frame())
    assert out["sample_id"].tolist() == [""]


def test_attach_duplicate_density_notes_last_wins() -> None:
    density = pd.DataFrame(
        {"Note": ["C4", "C4"], "sample_id": ["first__000000000001", "last__000000000002"]}
    )
    satellite = pd.DataFrame({"Note": ["C4"], "metric": [1.0]})
    out = attach_sample_id_from_density(satellite, density)
    # Documented policy: drop_duplicates(keep="last") on the Note map.
    assert out["sample_id"].tolist() == ["last__000000000002"]


# ---------------------------------------------------------------------------
# 5. Merge-key selection
# ---------------------------------------------------------------------------

def test_primary_merge_keys_preference_ladder() -> None:
    # Unique, fully populated sample_id wins.
    df = pd.DataFrame({"Note": ["C4", "C4"], "sample_id": ["a__1", "b__2"]})
    assert primary_merge_keys(df) == ["sample_id"]
    # Blank sample_id + unique Note -> Note.
    df_blank = pd.DataFrame({"Note": ["C4", "D4"], "sample_id": ["", ""]})
    assert primary_merge_keys(df_blank) == ["Note"]
    # Both non-unique, sample_id present -> sample_id (documented preference).
    df_dup = pd.DataFrame({"Note": ["C4", "C4"], "sample_id": ["x__1", "x__1"]})
    assert primary_merge_keys(df_dup) == ["sample_id"]
    # Neither column -> Note fallback token.
    assert primary_merge_keys(pd.DataFrame({"metric": [1.0]})) == ["Note"]
    assert primary_merge_keys(None) == ["Note"]  # type: ignore[arg-type]


def test_merge_keys_for_frames_fallback_branches() -> None:
    right = pd.DataFrame({"Note": ["C4"], "sample_id": ["a__1"]})
    # Empty/None anchor delegates to primary_merge_keys.
    assert merge_keys_for_frames(pd.DataFrame(), right) == ["Note"]
    assert merge_keys_for_frames(None, right) == ["Note"]  # type: ignore[arg-type]
    # Note key missing on the satellite -> primary keys of the anchor.
    left = pd.DataFrame({"Note": ["C4"], "sample_id": ["a__1"]})
    no_note_right = pd.DataFrame({"metric": [1.0]})
    assert merge_keys_for_frames(left, no_note_right) == ["sample_id"]
    # Non-unique sample_ids disqualify the id join -> Note.
    left_dup = pd.DataFrame({"Note": ["C4", "D4"], "sample_id": ["x__1", "x__1"]})
    right_dup = pd.DataFrame({"Note": ["C4", "D4"], "sample_id": ["x__1", "x__1"]})
    assert merge_keys_for_frames(left_dup, right_dup) == ["Note"]


# ---------------------------------------------------------------------------
# 6. Export hygiene helpers
# ---------------------------------------------------------------------------

def test_drop_dead_columns_guards_and_protected_names() -> None:
    empty = pd.DataFrame()
    assert drop_dead_columns(empty) is empty
    df = pd.DataFrame(
        {
            "Note": ["", ""],            # protected: kept even when blank
            "sample_id": [None, None],   # protected: kept even when all-NaN
            "zeros": [0.0, 0.0],         # all-zero numeric: documented keep
            "dead_text": ["nan", "<NA>"],
        }
    )
    out = drop_dead_columns(df)
    assert list(out.columns) == ["Note", "sample_id", "zeros"]
    # Custom protected set: a dead column survives when explicitly protected.
    out_custom = drop_dead_columns(df, protected=frozenset({"dead_text"}))
    assert "dead_text" in out_custom.columns
    assert DEAD_COLUMN_PROTECTED_NAMES == frozenset({"Note", "sample_id"})


def test_dedupe_identical_columns_numeric_string_and_guard_paths() -> None:
    empty = pd.DataFrame()
    assert dedupe_identical_columns(empty) is empty
    df = pd.DataFrame(
        {
            "metric": [1.0, None],
            "metric_2": [1.0, None],          # numeric + NaN-aligned duplicate -> dropped
            "metric_3": [1.0, 5.0],           # numerically different -> kept
            "flag": [True, False],
            "flag_2": ["True", "False"],      # string repr of base -> string-path drop
            "orphan_2": [9.0, 9.0],           # no base column -> kept
        }
    )
    out = dedupe_identical_columns(df)
    assert "metric_2" not in out.columns
    assert "metric_3" in out.columns
    assert "flag_2" not in out.columns
    assert "orphan_2" in out.columns
    # Copy semantics: the input frame keeps its original columns.
    assert "metric_2" in df.columns and "flag_2" in df.columns


def test_dedupe_string_suffix_columns_drop_only_on_exact_match() -> None:
    # REGRESSION (NaN-coercion bug): pure-string pairs coerce to all-NaN
    # numeric series on both sides, which used to compare equal and wrongly
    # drop a differing suffixed text column. The numeric branch must defer
    # to the exact string comparison for such pairs.
    df = pd.DataFrame(
        {
            "label": ["a", "b"],
            "label_2": ["a", "b"],            # identical strings -> dropped
            "label_3": ["a", "DIFFERENT"],    # differing strings -> preserved
        }
    )
    out = dedupe_identical_columns(df)
    assert "label_2" not in out.columns
    assert "label_3" in out.columns
    assert out["label_3"].tolist() == ["a", "DIFFERENT"]
    # Copy semantics unchanged: the input keeps all its columns.
    assert list(df.columns) == ["label", "label_2", "label_3"]


def test_dedupe_all_nan_numeric_base_does_not_swallow_differing_text() -> None:
    # An all-NaN numeric base column must not cause a differing string
    # suffix column to be treated as its duplicate.
    df = pd.DataFrame(
        {
            "value": [None, None],
            "value_2": ["x", "y"],            # textual content -> preserved
            "value_3": [None, None],          # genuinely identical -> dropped
        }
    )
    out = dedupe_identical_columns(df)
    assert "value_2" in out.columns
    assert "value_3" not in out.columns
