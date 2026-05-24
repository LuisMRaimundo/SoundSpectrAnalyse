from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from proc_audio import AudioProcessor  # noqa: E402


def test_canonical_f0_path_ignores_lowest_harmonic_row() -> None:
    ap = AudioProcessor()
    ap.f0_final = 220.0
    ap.f0_final_source = "prior_constrained_harmonic_fit"
    ap.f0_fit_accepted = True
    ap.harmonic_list_df = pd.DataFrame(
        {
            "Frequency (Hz)": [440.0, 660.0, 880.0],
            "Amplitude": [1.0, 0.8, 0.4],
        }
    )
    f0_hz, src, status = ap._canonical_f0_triplet_for_analysis()
    assert f0_hz == 220.0
    assert src == "prior_constrained_harmonic_fit"
    assert status == "fit_accepted_acoustically_verified"


def test_proc_audio_source_contains_no_min_frequency_f0_inference() -> None:
    text = (ROOT / "proc_audio.py").read_text(encoding="utf-8")
    assert 'nsmallest(1, "Frequency (Hz)")' not in text
