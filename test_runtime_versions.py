"""Runtime fingerprint helpers (CI / logs only — no export schema changes)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime_versions import runtime_stack_fingerprint, runtime_versions_dict


def test_runtime_versions_dict_has_core_keys():
    d = runtime_versions_dict()
    for k in ("python", "numpy", "scipy", "librosa"):
        assert k in d
        assert d[k] and d[k] != "unavailable"


def test_fingerprint_matches_numpy_import():
    d = runtime_versions_dict()
    fp = runtime_stack_fingerprint()
    assert "numpy=" + d["numpy"] in fp
    assert fp.count("=") >= 5


def test_fingerprint_stable_key_order():
    fp1 = runtime_stack_fingerprint()
    fp2 = runtime_stack_fingerprint()
    assert fp1 == fp2
    assert fp1.index("python=") < fp1.index("numpy=")
