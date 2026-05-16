"""metrics_summary clarity: safe dissonance_curve serialization and static interpretation notes."""

from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).parent.parent
ANALYZER_DIR = ROOT / "audio_analysis"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ANALYZER_DIR))

from super_audio_analyzer import (  # noqa: E402
    _METRICS_SUMMARY_NOTES,
    _write_metrics_summary_mapping,
)


def _ms_line(s: str) -> str:
    return str(s)


def test_metrics_summary_mapping_avoids_float_key_string_format_error() -> None:
    """Regression: float interval keys must not be formatted with %s (was: format error)."""
    curve = {float(i) / 10.0: 0.05 * i for i in range(5)}
    data = {
        "harmonic_energy_percentage": 71.0,
        "inharmonic_energy_percentage": 29.0,
        "harmonic_energy_percentage_peak_based": 68.0,
        "inharmonic_energy_percentage_peak_based": 32.0,
        "harmonic_density": 0.5,
        "dissonance_curve": curve,
    }
    buf = StringIO()
    buf.write(_METRICS_SUMMARY_NOTES)
    _write_metrics_summary_mapping(buf, data, _ms_line)
    text = buf.getvalue()
    assert "format error" not in text.lower()
    assert "Component counts in this batch report" in text
    assert "effective_partial_density" in text
    assert "Bin-based energy profile" in text
    assert "Peak-based validation profile" in text
    assert "dissonance_curve_points" in text
    assert "super_analysis_results.json" in text
