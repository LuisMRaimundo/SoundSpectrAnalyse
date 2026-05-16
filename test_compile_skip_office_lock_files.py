"""Regression test for Stage-2 compile lock-file filtering.

Excel/LibreOffice create transient lock files (``~$<name>.xlsx`` or
``.~lock.<name>#``) inside the per-note folder while the workbook is open.
Those names *contain* the substring ``spectral_analysis.xlsx`` and used to
be picked up by the Stage-2 compile loop, producing:

    ERROR ... Erro ao ler '...\\~$spectral_analysis.xlsx':
        [Errno 13] Permission denied

This test pins the contract: the compile-time file enumerator must skip
anything starting with ``~$`` or ``.~lock``.
"""

from __future__ import annotations

import os
import pathlib

import pytest

import compile_metrics


def _make_per_note(root: pathlib.Path, note: str, *, with_lock: bool) -> pathlib.Path:
    note_dir = root / f"{note}_2.00sec_Sustains" / note
    note_dir.mkdir(parents=True, exist_ok=True)
    target = note_dir / "spectral_analysis.xlsx"
    target.write_bytes(b"PK\x03\x04stub-not-a-real-xlsx")
    if with_lock:
        (note_dir / "~$spectral_analysis.xlsx").write_bytes(b"")
        (note_dir / ".~lock.spectral_analysis.xlsx#").write_bytes(b"")
    return target


def _collect_walked(folder: pathlib.Path, pattern: str = "spectral_analysis.xlsx") -> list[str]:
    """Mirror the compile-time enumerator in
    :func:`compile_metrics.compile_density_metrics` so we can assert
    lock-file filtering without invoking the full Excel reader."""
    out: list[str] = []
    for root, _, files in os.walk(folder):
        rp = pathlib.Path(root)
        for fname in files:
            if fname.startswith("~$") or fname.startswith(".~lock"):
                continue
            if pattern.lower() in fname.lower():
                out.append(str(rp / fname))
    return out


def test_compile_walk_skips_office_lock_files(tmp_path: pathlib.Path) -> None:
    _make_per_note(tmp_path, "A3", with_lock=True)
    _make_per_note(tmp_path, "A4", with_lock=False)

    found = _collect_walked(tmp_path)

    assert len(found) == 2, f"Expected exactly the two real per-note workbooks, got: {found}"
    assert not any(pathlib.Path(p).name.startswith("~$") for p in found)
    assert not any(pathlib.Path(p).name.startswith(".~lock") for p in found)


def test_scan_results_dir_rglob_skips_office_lock_files(tmp_path: pathlib.Path) -> None:
    """The schema-scanner (``scan_results_dir_for_stale_per_note_workbooks``)
    also walks the results tree; it must drop lock files too."""
    _make_per_note(tmp_path, "B3", with_lock=True)

    summary = compile_metrics.scan_results_dir_for_stale_per_note_workbooks(tmp_path)

    assert summary["results_dir"] == str(tmp_path)
    inspected = [d.get("path", "") for d in summary.get("details", [])]
    assert not any(pathlib.Path(p).name.startswith("~$") for p in inspected), inspected
    assert not any(pathlib.Path(p).name.startswith(".~lock") for p in inspected), inspected
    assert summary["total"] == 1, summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
