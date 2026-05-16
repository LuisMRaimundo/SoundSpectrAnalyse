"""Unit tests for the GUI / compile harmonic-inharmonic model-weight
resolution helper (no Qt).

In the Stage 1 + Stage 2 pipeline the resolver no longer consults an
external batch payload: per-note ``proc_audio`` derives the model weights
from the current spectrum and the resolver only handles two surfaces:

* manual GUI override (slider value);
* implicit ``current_analysis`` default (neutral 0.5 / 0.5 placeholders
  that ``proc_audio`` overwrites once the spectrum is classified).
"""

from gui_model_weight_policy import resolve_analysis_model_weights


def test_manual_override_sets_source_and_manual_fields():
    a, b, meta = resolve_analysis_model_weights(
        manual_override=True,
        slider_harmonic_fraction=0.6,
    )
    assert abs(a - 0.6) < 1e-9
    assert abs(b - 0.4) < 1e-9
    assert meta["model_weights_source"] == "manual_override"
    assert meta["manual_model_harmonic_weight"] == 0.6
    assert meta["manual_model_inharmonic_weight"] == 0.4
    assert "manual_model_weight_override" in meta.get("model_weights_warning", "")


def test_default_uses_current_analysis_placeholders():
    a, b, meta = resolve_analysis_model_weights(
        manual_override=False,
        slider_harmonic_fraction=0.5,
    )
    assert meta["model_weights_source"] == "current_analysis"
    assert meta["component_profile_source"] == "current_analysis"
    assert meta["external_component_profile_used"] is False
    assert meta["external_h_i_s_mapping_used"] is False
    # Neutral placeholders, not the legacy 0.95 / 0.05 pair.
    assert (a, b) == (0.5, 0.5)
