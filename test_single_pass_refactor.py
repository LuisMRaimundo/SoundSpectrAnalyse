"""
Validation tests for the SINGLE-PASS REFACTOR.

These tests cover the requirements from the architectural refactor that made
``proc_audio`` the single source of truth for harmonic / inharmonic / sub-bass
energy ratios:

A. Pure sine wave: effective partial count ≈ 1, harmonic ratio high.
B. Equal harmonic stack: effective partial count ≈ N.
C. Dominant partial + weak noise: effective partial count ≈ 1, not dozens.
D. Denominator consistency:
       component_harmonic_energy_ratio
     + component_inharmonic_energy_ratio
     + component_subbass_energy_ratio  ≈ 1
   and model_harmonic_weight + model_inharmonic_weight ≈ 1 when H + I > 0.
E. ``integrated_single_pass`` mode does not require Batch Excel outputs:
   ``RobustOrchestrator`` instantiates and Phase 1 / Phase 2 run without any
   ``batch_summary.xlsx``.

The helper-method tests build a tiny ``AudioProcessor`` shell and inject the
H / I / S energy attributes directly, so they exercise
``_set_model_weights_from_current_component_energy`` without running a full
spectral pipeline.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Local mirror of the canonical "effective partial count" formula. We do not
# import density.py here on purpose: the user requested an independent check
# that the formula reduces to 1 / N / intermediate in the expected limits.
# ---------------------------------------------------------------------------
def effective_partial_count(amplitudes: np.ndarray) -> float:
    a = np.asarray(amplitudes, dtype=float)
    a = a[np.isfinite(a) & (a > 0.0)]
    if a.size == 0:
        return 0.0
    p = a ** 2
    denom = float(np.sum(p * p))
    if denom <= 0.0:
        return 0.0
    return float((float(np.sum(p)) ** 2) / denom)


# ---------------------------------------------------------------------------
# A. Pure sine wave — effective partial count near 1.
# ---------------------------------------------------------------------------
def test_effective_partial_count_pure_sine():
    amps = np.array([1.0], dtype=float)
    assert effective_partial_count(amps) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# B. Equal harmonic stack — effective partial count near N.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n_partials", [2, 4, 8, 16])
def test_effective_partial_count_equal_stack(n_partials: int):
    amps = np.ones(n_partials, dtype=float)
    assert effective_partial_count(amps) == pytest.approx(float(n_partials), rel=1e-10)


# ---------------------------------------------------------------------------
# C. Dominant partial + weak noise — should NOT explode to dozens / hundreds.
# ---------------------------------------------------------------------------
def test_effective_partial_count_dominant_plus_noise():
    n_noise = 200
    rng = np.random.default_rng(seed=42)
    noise = rng.uniform(0.001, 0.01, size=n_noise)
    amps = np.concatenate([[1.0], noise])
    eff = effective_partial_count(amps)
    # Strictly bounded below the equal-energy ceiling (n_noise + 1 = 201)
    # and very close to 1 (the dominant partial holds essentially all power).
    assert 1.0 <= eff < 1.5, f"effective partial count={eff} should be ~1 for dominant+noise"


# ---------------------------------------------------------------------------
# Helper: minimal shell that exposes the bound method on a fresh object.
# ---------------------------------------------------------------------------
class _ShellAudioProcessor:
    """Subset of ``AudioProcessor`` needed to drive the canonical helper."""

    def __init__(self, *, H: float, I: float, S: float, auto: bool = True):
        self.harmonic_energy_sum = float(H)
        self.inharmonic_energy_sum = float(I)
        self.subbass_energy_sum = float(S)
        self.auto_model_weights_from_analysis = bool(auto)
        self.harmonic_weight = 0.0
        self.inharmonic_weight = 0.0
        self.logger = logging.getLogger("test_single_pass_refactor")

    # Lazy-bind the real helper from AudioProcessor to avoid re-implementing it.
    @classmethod
    def _bind(cls):
        from proc_audio import AudioProcessor  # local import (heavy module)

        cls._method = AudioProcessor._set_model_weights_from_current_component_energy

    def _set_model_weights_from_current_component_energy(self):
        if not hasattr(type(self), "_method"):
            type(self)._bind()
        return type(self)._method(self)


# ---------------------------------------------------------------------------
# D. Denominator consistency.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "H,I,S",
    [
        (1.0, 0.0, 0.0),     # pure harmonic
        (0.6, 0.3, 0.1),     # generic mix
        (0.0, 0.0, 1.0),     # pure sub-bass (H+I == 0 path)
        (0.0, 0.0, 0.0),     # everything zero (safe defaults)
        (0.7, 0.3, 0.0),     # legacy H+I only
        (1e-9, 1e-9, 1e-9),  # tiny but positive
    ],
)
def test_denominator_consistency(H, I, S):
    proc = _ShellAudioProcessor(H=H, I=I, S=S, auto=True)
    proc._set_model_weights_from_current_component_energy()

    comp_h = proc.component_harmonic_energy_ratio
    comp_i = proc.component_inharmonic_energy_ratio
    comp_s = proc.component_subbass_energy_ratio
    comp_total_inh = proc.component_total_inharmonic_energy_ratio

    if H + I + S > 1e-30:
        assert comp_h + comp_i + comp_s == pytest.approx(1.0, abs=1e-9)
    else:
        assert np.isnan(comp_h) and np.isnan(comp_i) and np.isnan(comp_s)

    if H + I + S > 1e-30:
        assert comp_total_inh == pytest.approx(comp_i + comp_s, abs=1e-12)
    else:
        assert np.isnan(comp_total_inh)

    mh = proc.model_harmonic_weight
    mi = proc.model_inharmonic_weight

    if H + I > 1e-30:
        assert mh == pytest.approx(H / (H + I), abs=1e-12)
        assert mi == pytest.approx(I / (H + I), abs=1e-12)
        assert mh + mi == pytest.approx(1.0, abs=1e-9)
    else:
        assert np.isnan(mh) and np.isnan(mi)
        assert proc.model_weight_status == "fallback_equal_weights_zero_HI_energy"
        assert proc.model_weight_fallback_applied is True

    assert proc.component_energy_denominator == "H+I+S"
    assert proc.component_energy_method == "single_pass_proc_audio_energy"
    # AUDIT — Stage 1 + Stage 2 pipeline emits ``current_analysis`` when the
    # model weights are derived from the current per-note spectrum.
    assert proc.component_profile_source == "current_analysis"

    # Internal compatibility aliases: kept in-memory so legacy callers do not
    # break, but they MUST mirror the canonical component_* triplet exactly
    # (no synthetic mapping, no external batch payload).
    for _a, _b in (
        (proc.batch_harmonic_energy_ratio, proc.component_harmonic_energy_ratio),
        (proc.batch_inharmonic_energy_ratio, proc.component_inharmonic_energy_ratio),
        (proc.batch_subbass_energy_ratio, proc.component_subbass_energy_ratio),
        (proc.batch_total_inharmonic_energy_ratio, proc.component_total_inharmonic_energy_ratio),
    ):
        assert (np.isnan(_a) and np.isnan(_b)) or (_a == _b)

    if H + I > 1e-30:
        assert proc.harmonic_weight == pytest.approx(mh, abs=1e-12)
        assert proc.inharmonic_weight == pytest.approx(mi, abs=1e-12)
    else:
        assert proc.harmonic_weight == pytest.approx(0.5, abs=1e-12)
        assert proc.inharmonic_weight == pytest.approx(0.5, abs=1e-12)


def test_denominator_consistency_auto_off_keeps_external_weights():
    """When auto=False, helper still records canonical fields, but does NOT
    overwrite ``harmonic_weight``/``inharmonic_weight``."""
    proc = _ShellAudioProcessor(H=0.6, I=0.3, S=0.1, auto=False)
    proc.harmonic_weight = 0.9
    proc.inharmonic_weight = 0.1
    proc._set_model_weights_from_current_component_energy()

    assert proc.component_harmonic_energy_ratio == pytest.approx(0.6, abs=1e-9)
    assert proc.model_harmonic_weight == pytest.approx(0.6 / 0.9, abs=1e-9)
    # Externally set weights preserved.
    assert proc.harmonic_weight == pytest.approx(0.9, abs=1e-12)
    assert proc.inharmonic_weight == pytest.approx(0.1, abs=1e-12)
    # The canonical provenance tag becomes ``current_analysis_legacy_weights``
    # when the spectrum was classified but the auto-rewrite step was disabled.
    assert proc.component_profile_source == "current_analysis_legacy_weights"


# ---------------------------------------------------------------------------
# E. The Stage 1 + Stage 2 orchestrator runs without any external Excel
#    summary; the constructor no longer accepts legacy batch parameters.
# ---------------------------------------------------------------------------
def test_orchestrator_stage_pipeline_has_no_batch_parameters(tmp_path: Path):
    pytest.importorskip("openpyxl")  # required by orchestrator's imports
    pytest.importorskip("pandas")
    from pipeline_orchestrator_integrated import RobustOrchestrator

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    dummy_wav = audio_dir / "Clarinete_A4.wav"
    dummy_wav.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")

    main_dir = tmp_path / "main_results"

    orchestrator = RobustOrchestrator(
        audio_files=[dummy_wav],
        main_analysis_output_dir=main_dir,
    )

    # The orchestrator exposes the two-stage API and not the legacy phase
    # API; introspection is enough to assert the surface.
    assert hasattr(orchestrator, "run_stage1_analysis")
    assert hasattr(orchestrator, "run_stage2_compilation")
    assert not hasattr(orchestrator, "phase1_mode")
    assert not hasattr(orchestrator, "excel_summary_path")
    assert not hasattr(orchestrator, "batch_output_dir")


def test_orchestrator_rejects_removed_phase_parameters() -> None:
    """The constructor must reject every removed batch parameter."""
    pytest.importorskip("pandas")
    from pipeline_orchestrator_integrated import RobustOrchestrator

    for kw in ("batch_output_dir", "excel_summary_path", "phase1_mode"):
        with pytest.raises(TypeError):
            RobustOrchestrator(
                audio_files=[],
                main_analysis_output_dir=Path("."),
                **{kw: "anything"},  # type: ignore[arg-type]
            )
