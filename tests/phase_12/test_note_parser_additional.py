from __future__ import annotations

"""
Helper-level contract tests for note_parser.py.

Protects canonical note-token extraction from filenames, parent folders,
and manifest metadata. No production code changes. No audio, GUI, plotting,
or full proc_audio pipeline runs.
"""

import librosa
import pytest

import note_parser as np_mod
from note_parser import (
    NOTE_SOURCE_FALLBACK_NO_OCTAVE,
    NOTE_SOURCE_FILENAME,
    NOTE_SOURCE_MANIFEST,
    NOTE_SOURCE_PARENT_FOLDER,
    NOTE_SOURCE_UNKNOWN,
    VALID_NOTE_SOURCES,
    canonical_note_from_filename,
    parse_note_token,
)


# ---------------------------------------------------------------------------
# 1. Basic note-name parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("C4", "C4"),
        ("D5", "D5"),
        ("E3", "E3"),
        ("F6", "F6"),
        ("G2", "G2"),
        ("A4", "A4"),
        ("B1", "B1"),
        ("c4", "C4"),
        ("a#5", "A#5"),
        ("bb3", "Bb3"),
        ("  A4  ", "A4"),
        ("Viola_C4_mf.wav", "C4"),
        ("A#3_3.72sec_Sustains.wav", "A#3"),
        ("Bb4_3.80sec_Sustains.wav", "Bb4"),
        ("D6_3.88sec_shifted_Sustains.wav", "D6"),
    ],
)
def test_parse_note_token_natural_and_octave_notes(text: str, expected: str) -> None:
    assert parse_note_token(text) == expected


def test_parse_note_token_returns_first_valid_token() -> None:
    assert parse_note_token("G5_C4_test.wav") == "G5"


def test_parse_note_token_requires_mandatory_octave_digits() -> None:
    assert parse_note_token("A#") is None
    assert parse_note_token("Bb") is None
    assert parse_note_token("C") is None


# ---------------------------------------------------------------------------
# 2. Accidentals and enharmonics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("C#4", "C#4"),
        ("F#5", "F#5"),
        ("G#2", "G#2"),
        ("Db4", "Db4"),
        ("Eb3", "Eb3"),
        ("Bb6", "Bb6"),
        ("A\u266f4", "A#4"),
        ("B\u266d3", "Bb3"),
    ],
)
def test_parse_note_token_accidentals_and_unicode_normalisation(
    text: str, expected: str
) -> None:
    assert parse_note_token(text) == expected


def test_enharmonic_spellings_preserved_in_output() -> None:
    assert parse_note_token("C#4") == "C#4"
    assert parse_note_token("Db4") == "Db4"


def test_enharmonic_spellings_map_to_same_librosa_hz() -> None:
    """Parsed tokens must agree on pitch height (no silent semitone drift)."""
    c_sharp = parse_note_token("C#4")
    d_flat = parse_note_token("Db4")
    assert c_sharp is not None and d_flat is not None
    assert librosa.note_to_hz(c_sharp) == pytest.approx(librosa.note_to_hz(d_flat), rel=1e-12)


@pytest.mark.parametrize("text", ["C##4", "Dbb4", "E###4"])
def test_double_and_triple_accidentals_not_supported(text: str) -> None:
    assert parse_note_token(text) is None


# ---------------------------------------------------------------------------
# 3. Octave semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("C0", "C0"),
        ("C10", "C10"),
        ("A12", "A12"),
    ],
)
def test_parse_note_token_positive_octave_digits(text: str, expected: str) -> None:
    assert parse_note_token(text) == expected


@pytest.mark.parametrize("text", ["C-1", "A-2", "G#-3"])
def test_negative_octave_notation_not_supported(text: str) -> None:
    assert parse_note_token(text) is None


def test_decimal_octave_suffix_matches_leading_digit_group_only() -> None:
    assert parse_note_token("C4.5") == "C4"


# ---------------------------------------------------------------------------
# 4. librosa / frequency contract (parsed tokens only; no formula duplication)
# ---------------------------------------------------------------------------

def test_parsed_a4_maps_to_440_hz_via_librosa() -> None:
    token = parse_note_token("A4")
    assert token == "A4"
    assert librosa.note_to_hz(token) == pytest.approx(440.0, rel=1e-9)


def test_octave_step_doubles_frequency_per_octave_for_same_letter() -> None:
    c4 = parse_note_token("C4")
    c5 = parse_note_token("C5")
    assert c4 is not None and c5 is not None
    assert librosa.note_to_hz(c5) == pytest.approx(2.0 * librosa.note_to_hz(c4), rel=1e-9)


def test_semitone_step_increments_hz_by_twelfth_root_of_two() -> None:
    c4 = parse_note_token("C4")
    c_sharp = parse_note_token("C#4")
    assert c4 is not None and c_sharp is not None
    ratio = librosa.note_to_hz(c_sharp) / librosa.note_to_hz(c4)
    assert ratio == pytest.approx(2.0 ** (1.0 / 12.0), rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Filename / token extraction and source priority
# ---------------------------------------------------------------------------

def test_canonical_note_manifest_priority_over_filename() -> None:
    note, source = canonical_note_from_filename(
        "wrong_F9.wav", manifest_note="G2"
    )
    assert note == "G2"
    assert source == NOTE_SOURCE_MANIFEST


def test_canonical_note_filename_token_when_no_manifest() -> None:
    note, source = canonical_note_from_filename("Viola_C4_mf.wav")
    assert note == "C4"
    assert source == NOTE_SOURCE_FILENAME


def test_canonical_note_parent_folder_when_filename_has_no_token() -> None:
    note, source = canonical_note_from_filename(
        "sustains.wav", parent_folder="results_D2_final"
    )
    assert note == "D2"
    assert source == NOTE_SOURCE_PARENT_FOLDER


def test_canonical_note_filename_beats_parent_folder() -> None:
    note, source = canonical_note_from_filename("F3.wav", parent_folder="G4")
    assert note == "F3"
    assert source == NOTE_SOURCE_FILENAME


def test_canonical_note_fallback_no_octave_from_filename() -> None:
    note, source = canonical_note_from_filename("Viola_A#_mf.wav")
    assert note == "A#"
    assert source == NOTE_SOURCE_FALLBACK_NO_OCTAVE


def test_canonical_note_fallback_no_octave_from_parent_folder() -> None:
    note, source = canonical_note_from_filename("x.wav", parent_folder="Bb")
    assert note == "Bb"
    assert source == NOTE_SOURCE_FALLBACK_NO_OCTAVE


def test_canonical_note_unknown_when_nothing_parseable() -> None:
    note, source = canonical_note_from_filename("noname.wav")
    assert note is None
    assert source == NOTE_SOURCE_UNKNOWN


def test_manifest_letter_only_does_not_override_filename_with_octave() -> None:
    note, source = canonical_note_from_filename("C4.wav", manifest_note="A#")
    assert note == "C4"
    assert source == NOTE_SOURCE_FILENAME


def test_manifest_invalid_token_falls_through_to_filename() -> None:
    note, source = canonical_note_from_filename("C4.wav", manifest_note="XYZ")
    assert note == "C4"
    assert source == NOTE_SOURCE_FILENAME


def test_note_at_beginning_middle_and_with_separators() -> None:
    assert parse_note_token("A4_sustain.wav") == "A4"
    assert parse_note_token("run_E5_batch") == "E5"
    assert parse_note_token("Bn-ord-A#1-pp-N-N_Sustains.wav") == "A#1"


def test_leading_position_predicate_skips_embedded_letters_in_tokens() -> None:
    assert parse_note_token("Bn-ord-Sustains.wav") is None
    assert parse_note_token("Bn-ord") is None


def test_clarinet_prefix_does_not_block_valid_token_after_separator() -> None:
    assert parse_note_token("Clarinet_C4.wav") == "C4"


# ---------------------------------------------------------------------------
# 6. Invalid input handling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", None])
def test_parse_note_token_empty_or_none_returns_none(text: object) -> None:
    assert parse_note_token(text) is None  # type: ignore[arg-type]


def test_parse_note_token_non_string_coercion_without_match() -> None:
    assert parse_note_token(123) is None  # type: ignore[arg-type]


@pytest.mark.parametrize("text", ["H4", "R3", "XYZ", "random text"])
def test_parse_note_token_invalid_letters_or_text(text: str) -> None:
    assert parse_note_token(text) is None


@pytest.mark.parametrize("text", ["pp4", "ord3", "N4", "mf4"])
def test_dynamic_and_ambiguous_tokens_not_parsed_as_notes(text: str) -> None:
    assert parse_note_token(text) is None


def test_canonical_note_unknown_for_dynamic_only_filenames() -> None:
    for name in ("pp_sustain.wav", "mf.wav", "ff_ord.wav"):
        note, source = canonical_note_from_filename(name)
        assert note is None
        assert source == NOTE_SOURCE_UNKNOWN


def test_canonical_note_accepts_none_filename_and_parent() -> None:
    note, source = canonical_note_from_filename(None, parent_folder=None)
    assert note is None
    assert source == NOTE_SOURCE_UNKNOWN


# ---------------------------------------------------------------------------
# 7. Non-mutation and determinism
# ---------------------------------------------------------------------------

def test_parse_note_token_is_deterministic() -> None:
    sample = "Bn-ord-A#1-pp-N-N_Sustains.wav"
    first = parse_note_token(sample)
    second = parse_note_token(sample)
    assert first == second == "A#1"


def test_canonical_note_from_filename_is_deterministic() -> None:
    kwargs = {
        "filename": "Viola_C4_mf.wav",
        "manifest_note": None,
        "parent_folder": "Bb_batch",
    }
    assert canonical_note_from_filename(**kwargs) == canonical_note_from_filename(**kwargs)


def test_valid_note_sources_tuple_is_stable() -> None:
    assert VALID_NOTE_SOURCES == (
        NOTE_SOURCE_MANIFEST,
        NOTE_SOURCE_FILENAME,
        NOTE_SOURCE_PARENT_FOLDER,
        NOTE_SOURCE_FALLBACK_NO_OCTAVE,
        NOTE_SOURCE_UNKNOWN,
    )
    assert set(np_mod.VALID_NOTE_SOURCES) == set(VALID_NOTE_SOURCES)


# ---------------------------------------------------------------------------
# 8. Thesis-critical regression guards
# ---------------------------------------------------------------------------

def test_octave_not_shifted_for_canonical_doc_examples() -> None:
    """Docstring examples must keep expected octave (not letter-only fallback)."""
    assert parse_note_token("A#3_3.72sec_Sustains.wav") == "A#3"
    assert parse_note_token("Bn-ord-A#1-pp-N-N_Sustains.wav") == "A#1"
    assert parse_note_token("D6_3.88sec_shifted_Sustains.wav") == "D6"


def test_fallback_no_octave_never_returned_by_parse_note_token() -> None:
    """Letter-only labels are diagnostic-only via canonical_note_from_filename."""
    assert parse_note_token("trumpet_Bb.wav") is None
    note, source = canonical_note_from_filename("trumpet_Bb.wav")
    assert note == "Bb"
    assert source == NOTE_SOURCE_FALLBACK_NO_OCTAVE


def test_invalid_filename_tokens_do_not_silently_become_valid_notes() -> None:
    assert parse_note_token("ord_C4.wav") == "C4"
    assert parse_note_token("ord.wav") is None
    note, source = canonical_note_from_filename("ord.wav")
    assert note is None
    assert source == NOTE_SOURCE_UNKNOWN


def test_first_token_wins_not_stem_split_heuristic() -> None:
    """Ornamentation prefix must not be mistaken for a note when no octave follows."""
    assert parse_note_token("ord_C4.wav") == "C4"
    assert parse_note_token("Bn-ord-A#1") == "A#1"
    assert parse_note_token("Bn-ord") is None
