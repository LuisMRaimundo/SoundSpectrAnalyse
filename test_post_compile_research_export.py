"""Tests for ``post_compile_research_export.run_research_workbook_export``."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from post_compile_research_export import run_research_workbook_export
from tests.test_research_density_export import _write_minimal_compiled_workbook


def test_helper_creates_research_workbook(tmp_path: Path) -> None:
    src = tmp_path / "compiled_density_metrics.xlsx"
    _write_minimal_compiled_workbook(src)
    log = logging.getLogger("test_post_compile")
    out = run_research_workbook_export(src, log=log)
    assert out is not None
    assert out.name == "compiled_density_metrics_research.xlsx"
    assert out.parent == src.parent
    assert out.is_file()


def test_helper_twice_overwrites(tmp_path: Path) -> None:
    src = tmp_path / "compiled_density_metrics.xlsx"
    _write_minimal_compiled_workbook(src)
    log = logging.getLogger("test_post_compile_b")
    p1 = run_research_workbook_export(src, log=log)
    p2 = run_research_workbook_export(src, log=log)
    assert p1 == p2
    assert p2.is_file()


def test_source_compiled_not_modified(tmp_path: Path) -> None:
    src = tmp_path / "compiled_density_metrics.xlsx"
    _write_minimal_compiled_workbook(src)
    h1 = hashlib.sha256(src.read_bytes()).hexdigest()
    run_research_workbook_export(src, log=logging.getLogger("t"))
    h2 = hashlib.sha256(src.read_bytes()).hexdigest()
    assert h1 == h2


def test_helper_returns_none_when_export_raises(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    src = tmp_path / "compiled_density_metrics.xlsx"
    _write_minimal_compiled_workbook(src)
    caplog.set_level(logging.ERROR)
    with patch(
        "tools.export_research_density_workbook.export_research_workbook",
        side_effect=RuntimeError("simulated export failure"),
    ):
        r = run_research_workbook_export(src, log=logging.getLogger("t"))
    assert r is None
    assert "Research workbook export failed" in caplog.text


def test_helper_returns_none_when_missing_file(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING)
    missing = tmp_path / "nope_compiled_density_metrics.xlsx"
    r = run_research_workbook_export(missing, log=logging.getLogger("t"))
    assert r is None
    assert "skipped" in caplog.text.lower()
