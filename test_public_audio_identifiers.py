"""Phase 2 — public_audio_identifiers stable sample_id and publication fields."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import public_audio_identifiers as pai


def test_compute_sample_id_stable():
    sid = pai.compute_sample_id("IOWA_Vla_arco_mf.A4", "abcdef0123456789" * 4)
    assert sid.startswith("IOWA_Vla_arco_mf_A4__sha256_abcdef01")
    sid2 = pai.compute_sample_id("IOWA_Vla_arco_mf.A4", "abcdef0123456789" * 4)
    assert sid == sid2


def test_build_block_from_real_file(tmp_path: Path) -> None:
    p = tmp_path / "IOWA_Vla_arco_mf.A4.wav"
    p.write_bytes(b"\x00\x01\x02" * 4000)
    d = pai.build_public_audio_identifier_block(
        p,
        dataset_root=None,
        y=None,
        sr=44100,
        detected_note_label="A4",
        channel_mode="mono",
    )
    assert d["audio_file_name"] == "IOWA_Vla_arco_mf.A4.wav"
    assert d["audio_file_stem"] == "IOWA_Vla_arco_mf.A4"
    assert d["audio_file_extension"] == ".wav"
    assert len(d["audio_sha256"] or "") == 64
    assert d["sample_id"].endswith(d["audio_sha256"][:8])
    assert d["dataset_relative_path"] == "IOWA_Vla_arco_mf.A4.wav"
    assert d["detected_note_label"] == "A4"


def test_dataset_root_relative_path(tmp_path: Path) -> None:
    ds = tmp_path / "ds"
    ds.mkdir()
    sub = ds / "vio"
    sub.mkdir()
    p = sub / "clip.wav"
    p.write_bytes(b"abc" * 500)
    out = pai.build_public_audio_identifier_block(p, dataset_root=ds, sr=48000, channel_mode="mono")
    assert out["dataset_relative_path"] == "vio/clip.wav"
    assert "clip.wav" in (out["audio_file_name"],)


def test_parse_heuristic_labels():
    lab = pai.parse_filename_heuristic_labels("IOWA_Vla_arco_mf")
    assert lab.get("parsed_instrument_label") == "Vla"
    assert lab.get("parsed_technique_label") == "arco"
    assert lab.get("parsed_dynamic_label") == "MF"
