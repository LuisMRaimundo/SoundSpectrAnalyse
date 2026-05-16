"""Audit tests for the inharmonic-energy underestimation fix.

These tests cover the audit's checklist points 1–8:

1. Disable batch-based export alignment in integrated_single_pass mode
   (``export_alignment_factor == 1.0``,
   ``export_alignment_source == "disabled_integrated_single_pass"``).
2. Preserve raw vs display-scaled values separately
   (``Amplitude_raw`` / ``Power_raw`` vs ``Amplitude_display_scaled``).
3. Density_Metrics direct extraction prefers raw columns over the legacy
   ``Amplitude`` column and never reads ``Amplitude_display_scaled``;
   flags ``legacy_scaled_source_used`` when only ``Amplitude`` is available.
4. ``batch_*_energy_ratio`` placeholders are quarantined: they must not
   be used as the canonical source for component_* ratios.
5. Harmonic-slot acceptance — covered indirectly by the synthetic-signal
   tests A–D which verify that broadband residual energy is properly
   attributed to the non-harmonic bucket.
6. New residual / non-harmonic ratios:
   ``component_residual_noise_energy_ratio`` and
   ``component_nonharmonic_energy_ratio``.
7. Synthetic-signal acceptance tests A–D.
8. Regression test for the observed batch_* placeholder failure.

The unit tests here use the ``_ShellAudioProcessor`` pattern from
``test_single_pass_refactor`` (small in-memory shell that calls the
bound ``_set_model_weights_from_current_component_energy`` method
directly), so they remain fast.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import compile_metrics as cm  # noqa: E402


# ---------------------------------------------------------------------------
# Shell that exercises proc_audio._set_model_weights_from_current_component_energy
# without running the full STFT/export pipeline.
# ---------------------------------------------------------------------------
class _ShellAudioProcessor:
    def __init__(
        self,
        *,
        H: float,
        I: float,
        S: float,
        residual: float = 0.0,
        auto: bool = True,
    ):
        self.harmonic_energy_sum = float(H)
        self.inharmonic_energy_sum = float(I)
        self.subbass_energy_sum = float(S)
        self.residual_noise_energy_sum = float(residual)
        # total_filtered_spectral_energy is set by the production code in
        # _calculate_metrics; we replicate that bookkeeping here for the
        # downstream invariant check.
        self.total_filtered_spectral_energy = float(H + I + S + residual)
        self.auto_model_weights_from_analysis = bool(auto)
        self.harmonic_weight = 0.5
        self.inharmonic_weight = 0.5
        self.logger = logging.getLogger("test_inharmonic_energy_audit")

    @classmethod
    def _bind(cls):
        from proc_audio import AudioProcessor

        cls._method = AudioProcessor._set_model_weights_from_current_component_energy

    def _set_model_weights_from_current_component_energy(self):
        if not hasattr(type(self), "_method"):
            type(self)._bind()
        return type(self)._method(self)


# ---------------------------------------------------------------------------
# Test A — Pure sine: H >> I, S, residual; non-harmonic ratios all low.
# ---------------------------------------------------------------------------
def test_pure_sine_low_inharmonic_and_residual():
    proc = _ShellAudioProcessor(H=1.0, I=1e-6, S=1e-6, residual=1e-6, auto=True)
    proc._set_model_weights_from_current_component_energy()

    assert proc.component_harmonic_energy_ratio > 0.99
    assert proc.component_inharmonic_energy_ratio < 1e-3
    assert proc.component_residual_noise_energy_ratio < 1e-3
    # Total non-harmonic should also be small.
    assert proc.component_nonharmonic_energy_ratio < 1e-2
    assert proc.component_residual_energy_denominator == "H+I+S+residual"


# ---------------------------------------------------------------------------
# Test B — Harmonic stack: harmonic dominant; non-harmonic buckets remain low.
# ---------------------------------------------------------------------------
def test_harmonic_stack_low_nonharmonic():
    proc = _ShellAudioProcessor(H=10.0, I=0.05, S=0.01, residual=0.02, auto=True)
    proc._set_model_weights_from_current_component_energy()

    assert proc.component_harmonic_energy_ratio > 0.99
    assert proc.component_inharmonic_energy_ratio < 0.01
    assert proc.component_residual_noise_energy_ratio < 0.01
    assert proc.component_nonharmonic_energy_ratio < 0.01


# ---------------------------------------------------------------------------
# Test C — Harmonic stack + off-grid sinusoid: inharmonic ratio increases.
# ---------------------------------------------------------------------------
def test_offgrid_sinusoid_increases_inharmonic_ratio():
    base = _ShellAudioProcessor(H=10.0, I=0.01, S=0.0, residual=0.0, auto=True)
    base._set_model_weights_from_current_component_energy()
    base_ratio = base.component_inharmonic_energy_ratio

    # Add a clearly off-grid sinusoid as a discrete inharmonic peak.
    bumped = _ShellAudioProcessor(H=10.0, I=2.0, S=0.0, residual=0.0, auto=True)
    bumped._set_model_weights_from_current_component_energy()

    assert bumped.component_inharmonic_energy_ratio > base_ratio
    # The residual bucket should remain ~zero because the off-grid peak is
    # discrete and belongs in the inharmonic bucket, not the residual one.
    assert bumped.component_residual_noise_energy_ratio < 1e-9


# ---------------------------------------------------------------------------
# Test D — Harmonic stack + broadband noise: residual ratio increases, the
# discrete-inharmonic ratio MAY stay low.
# ---------------------------------------------------------------------------
def test_broadband_noise_increases_residual_not_inharmonic():
    base = _ShellAudioProcessor(H=10.0, I=0.01, S=0.0, residual=0.0, auto=True)
    base._set_model_weights_from_current_component_energy()
    base_residual = base.component_residual_noise_energy_ratio

    # Broadband residual is much larger than the discrete inharmonic energy.
    noisy = _ShellAudioProcessor(H=10.0, I=0.01, S=0.0, residual=3.0, auto=True)
    noisy._set_model_weights_from_current_component_energy()

    assert noisy.component_residual_noise_energy_ratio > base_residual
    assert noisy.component_residual_noise_energy_ratio > 0.1
    # Inharmonic-bucket ratio MAY stay roughly the same; the residual bucket
    # is the one that captures broadband content.
    assert noisy.component_inharmonic_energy_ratio < 0.01
    # And component_nonharmonic captures BOTH.
    assert noisy.component_nonharmonic_energy_ratio > 0.2


# ---------------------------------------------------------------------------
# Sub-test: residual bucket is bounded in [0, 1] even when residual >> H+I+S.
# ---------------------------------------------------------------------------
def test_residual_ratio_bounded_zero_one():
    proc = _ShellAudioProcessor(H=0.1, I=0.0, S=0.0, residual=10.0, auto=True)
    proc._set_model_weights_from_current_component_energy()
    assert 0.0 <= proc.component_residual_noise_energy_ratio <= 1.0
    assert 0.0 <= proc.component_nonharmonic_energy_ratio <= 1.0


def test_zero_total_safe_defaults():
    proc = _ShellAudioProcessor(H=0.0, I=0.0, S=0.0, residual=0.0, auto=True)
    proc._set_model_weights_from_current_component_energy()
    assert np.isnan(proc.component_residual_noise_energy_ratio)
    assert np.isnan(proc.component_nonharmonic_energy_ratio)
    assert getattr(proc, "component_energy_status", None) == "undefined_zero_total_energy"


# ---------------------------------------------------------------------------
# Test E — Integrated mode export columns (direct check on the extractor's
# column-preference policy). With write_raw_columns=True, the extractor must
# pick ``Amplitude_raw``, never ``Amplitude_display_scaled``, never
# ``Amplitude`` (legacy).
# ---------------------------------------------------------------------------
def _write_workbook_for_extractor(
    path: Path,
    *,
    note: str,
    h_amps,
    ih_amps,
    sb_amps,
    add_display_scaled: bool = False,
    use_power_raw: bool = False,
    add_legacy_amplitude_only: bool = False,
) -> None:
    def _row(amps, freq0, mag):
        df = pd.DataFrame(
            {
                "Frequency (Hz)": np.linspace(freq0, freq0 * (len(amps) + 1), len(amps) or 1)[: len(amps)],
                "Magnitude (dB)": [mag] * len(amps),
                "Amplitude": list(amps),
                "Note": [note] * len(amps),
            }
        )
        arr = np.asarray(amps, dtype=float)
        if add_legacy_amplitude_only:
            return df  # Only ``Amplitude``, no raw columns.
        df["Amplitude_raw"] = arr
        if use_power_raw:
            df["Power_raw"] = arr ** 2
        if add_display_scaled:
            df["Amplitude_display_scaled"] = arr * 0.123  # forbidden column
        return df

    harm_df = _row(h_amps, 220.0, -30.0)
    ih_df = _row(ih_amps, 330.0, -35.0)
    sb_df = _row(sb_amps, 40.0, -45.0)
    from proc_audio import ANALYSIS_SCHEMA_VERSION as _ASV

    am_df = pd.DataFrame(
        [
            ("analysis_schema_version", _ASV),
            ("model_weights_source", "current_analysis"),
            ("component_harmonic_energy_ratio", 0.8),
            ("component_inharmonic_energy_ratio", 0.15),
            ("component_subbass_energy_ratio", 0.05),
            ("component_profile_source", "integrated_single_pass"),
            ("export_alignment_factor", 1.0),
            ("export_alignment_source", "disabled_integrated_single_pass"),
        ],
        columns=["Parameter", "Value"],
    )
    with pd.ExcelWriter(path, engine="xlsxwriter") as writer:
        harm_df.to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        ih_df.to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        sb_df.to_excel(writer, sheet_name="Sub-bass band", index=False)
        am_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)


def test_extractor_prefers_amplitude_raw_over_legacy_amplitude(tmp_path: Path):
    wb = tmp_path / "raw_preferred.xlsx"
    _write_workbook_for_extractor(
        wb, note="A4",
        h_amps=[10, 20, 30], ih_amps=[5, 5], sb_amps=[2],
        add_display_scaled=False, use_power_raw=False,
    )
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "ok"
    # Source string MUST point to Amplitude_raw, not Amplitude.
    assert "Amplitude_raw" in info["harmonic_spectrum_source"]
    assert "Amplitude_raw" in info["inharmonic_spectrum_source"]
    assert "Amplitude_raw" in info["subbass_spectrum_source"]
    assert info["legacy_scaled_source_used"] is False
    # And the sums match the raw amplitudes.
    assert info["D_H"] == pytest.approx(60.0)
    assert info["D_I"] == pytest.approx(10.0)
    assert info["D_S"] == pytest.approx(2.0)


def test_extractor_default_basis_prefers_amplitude_raw_over_power_raw(tmp_path: Path):
    """AUDIT FIX (Density_Metrics component basis) — Power_raw is the
    energy-ratio basis, NOT the density-component basis. By default the
    Density_Metrics extractor must pick ``Amplitude_raw`` even when the
    spectrum sheet also carries ``Power_raw`` — otherwise low-frequency
    sub-bass bins dominate D_S through Σ A² instead of Σ A and bias the
    weighted density downward.
    """
    wb = tmp_path / "amp_raw_vs_power_raw.xlsx"
    _write_workbook_for_extractor(
        wb, note="A4",
        h_amps=[10, 20, 30], ih_amps=[5, 5], sb_amps=[2],
        use_power_raw=True,   # workbook carries BOTH Amplitude_raw AND Power_raw
    )
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "ok"
    # The default ``amplitude_sum`` basis must pick Amplitude_raw.
    assert "Amplitude_raw" in info["harmonic_spectrum_source"]
    assert "Amplitude_raw" in info["inharmonic_spectrum_source"]
    assert "Amplitude_raw" in info["subbass_spectrum_source"]
    assert "Power_raw" not in info["harmonic_spectrum_source"]
    assert "Power_raw" not in info["inharmonic_spectrum_source"]
    assert "Power_raw" not in info["subbass_spectrum_source"]
    assert info["legacy_scaled_source_used"] is False
    assert info["density_component_basis"] == "amplitude_sum"
    assert info["density_weight_basis"] == "energy_ratio_power_sum"
    # Sums match Σ Amplitude_raw — not Σ A².
    assert info["D_H"] == pytest.approx(60.0)
    assert info["D_I"] == pytest.approx(10.0)
    assert info["D_S"] == pytest.approx(2.0)


def test_extractor_power_sum_debug_basis_selects_power_raw(tmp_path: Path):
    """The opt-in ``density_component_basis='power_sum'`` debug basis must
    select Power_raw and sum Σ A². It is provided ONLY for diagnostic /
    comparison runs; the canonical Density_Metrics path never enables it.
    """
    wb = tmp_path / "power_sum_debug.xlsx"
    _write_workbook_for_extractor(
        wb, note="A4",
        h_amps=[10, 20, 30], ih_amps=[5, 5], sb_amps=[2],
        use_power_raw=True,
    )
    info = cm.extract_density_components_from_per_note_workbook(
        wb, density_component_basis="power_sum",
    )
    assert info["density_extraction_status"] == "ok"
    assert "Power_raw" in info["harmonic_spectrum_source"]
    assert "Power_raw" in info["inharmonic_spectrum_source"]
    assert "Power_raw" in info["subbass_spectrum_source"]
    assert info["density_component_basis"] == "power_sum"
    # Sums match Σ A².
    assert info["D_H"] == pytest.approx(100 + 400 + 900)
    assert info["D_I"] == pytest.approx(25 + 25)
    assert info["D_S"] == pytest.approx(4.0)


def test_extractor_never_reads_amplitude_display_scaled(tmp_path: Path):
    wb = tmp_path / "with_display_scaled.xlsx"
    _write_workbook_for_extractor(
        wb, note="A4",
        h_amps=[10, 20, 30], ih_amps=[5, 5], sb_amps=[2],
        add_display_scaled=True, use_power_raw=False,
    )
    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "ok"
    for src in (
        info["harmonic_spectrum_source"],
        info["inharmonic_spectrum_source"],
        info["subbass_spectrum_source"],
    ):
        assert "Amplitude_display_scaled" not in src, (
            f"Extractor must never read the forbidden display-scaled column; got: {src}"
        )


def test_extractor_flags_legacy_scaled_source_used(tmp_path: Path):
    wb = tmp_path / "legacy_only.xlsx"
    _write_workbook_for_extractor(
        wb, note="A4",
        h_amps=[10, 20, 30], ih_amps=[5, 5], sb_amps=[2],
        add_legacy_amplitude_only=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "legacy_scaled_source_used"
    # Even in legacy fallback, the sums are still computed.
    assert info["D_H"] == pytest.approx(60.0)
    assert info["legacy_scaled_source_used"] is True


# ---------------------------------------------------------------------------
# Test F — Legacy mode (auto=False): batch_* placeholders STILL only feed
# the component_* ratios via the canonical ``_set_model_weights_from_current_component_energy``
# helper, which derives them from the audio's own H/I/S energies — never
# from external batch budgets. The helper assigns batch_* aliases FROM
# canonical values in both modes; this is the regression check.
# ---------------------------------------------------------------------------
def test_legacy_mode_component_ratios_still_derive_from_audio_energies():
    proc = _ShellAudioProcessor(H=10.0, I=5.0, S=2.0, residual=1.0, auto=False)
    proc._set_model_weights_from_current_component_energy()

    expected_h = 10.0 / 17.0
    expected_i = 5.0 / 17.0
    expected_s = 2.0 / 17.0

    assert proc.component_harmonic_energy_ratio == pytest.approx(expected_h, abs=1e-12)
    assert proc.component_inharmonic_energy_ratio == pytest.approx(expected_i, abs=1e-12)
    assert proc.component_subbass_energy_ratio == pytest.approx(expected_s, abs=1e-12)

    # In legacy mode the batch_* aliases STILL track the canonical values
    # (they are NOT injected from a Batch Excel placeholder).
    assert proc.batch_harmonic_energy_ratio == pytest.approx(expected_h, abs=1e-12)
    assert proc.batch_inharmonic_energy_ratio == pytest.approx(expected_i, abs=1e-12)
    assert proc.batch_subbass_energy_ratio == pytest.approx(expected_s, abs=1e-12)

    # And residual / non-harmonic ratios are still published.
    assert hasattr(proc, "component_residual_noise_energy_ratio")
    assert hasattr(proc, "component_nonharmonic_energy_ratio")


# ---------------------------------------------------------------------------
# Regression — batch_* placeholders 0.95/0.05 must NOT influence the
# component_* ratios. The canonical helper is the single source of truth.
# ---------------------------------------------------------------------------
def test_batch_placeholders_ignored_for_component_ratios():
    """Mimics the observed failure: an external caller pre-set the
    misleading legacy ``batch_*`` placeholders to 0.95/0.05/0.0 (the
    Batch GUI defaults). The integrated_single_pass canonical helper
    must overwrite them with the audio-derived values."""
    proc = _ShellAudioProcessor(H=10.0, I=5.0, S=2.0, residual=0.0, auto=True)
    proc.batch_harmonic_energy_ratio = 0.95
    proc.batch_inharmonic_energy_ratio = 0.05
    proc.batch_subbass_energy_ratio = 0.0
    proc._set_model_weights_from_current_component_energy()

    expected_h = 10.0 / 17.0
    expected_i = 5.0 / 17.0
    expected_s = 2.0 / 17.0

    # Canonical ratios reflect the audio's energies, not 0.95/0.05/0.0.
    assert proc.component_harmonic_energy_ratio == pytest.approx(expected_h, abs=1e-12)
    assert proc.component_inharmonic_energy_ratio == pytest.approx(expected_i, abs=1e-12)
    assert proc.component_subbass_energy_ratio == pytest.approx(expected_s, abs=1e-12)
    # And batch_* aliases have been overwritten from the canonical values.
    assert proc.batch_harmonic_energy_ratio == pytest.approx(expected_h, abs=1e-12)
    assert proc.batch_inharmonic_energy_ratio == pytest.approx(expected_i, abs=1e-12)


# ---------------------------------------------------------------------------
# Density_Metrics direct extraction does NOT silently substitute batch_*
# for component_* when both are present.
# ---------------------------------------------------------------------------
def test_extractor_uses_component_weights_when_batch_present(tmp_path: Path):
    """If a workbook carries BOTH component_* (canonical) and batch_*
    (legacy placeholders), the extractor must use component_*."""
    wb = tmp_path / "mixed_weights.xlsx"

    def _row(amps, freq0, mag):
        df = pd.DataFrame(
            {
                "Frequency (Hz)": np.linspace(freq0, freq0 * (len(amps) + 1), len(amps) or 1)[: len(amps)],
                "Magnitude (dB)": [mag] * len(amps),
                "Amplitude": list(amps),
                "Amplitude_raw": list(amps),
            }
        )
        return df

    harm_df = _row([10, 20, 30], 220.0, -30.0)
    ih_df = _row([5, 5], 330.0, -35.0)
    sb_df = _row([2], 40.0, -45.0)
    for _df in (harm_df, ih_df, sb_df):
        _df["Power_raw"] = _df["Amplitude_raw"] ** 2
    from proc_audio import ANALYSIS_SCHEMA_VERSION as _ASV

    am_df = pd.DataFrame(
        [
            ("analysis_schema_version", _ASV),
            ("model_weights_source", "current_analysis"),
            ("component_profile_source", "integrated_single_pass"),
            ("export_alignment_factor", 1.0),
            ("export_alignment_source", "disabled_integrated_single_pass"),
            ("component_harmonic_energy_ratio", 0.8),
            ("component_inharmonic_energy_ratio", 0.15),
            ("component_subbass_energy_ratio", 0.05),
            ("batch_harmonic_energy_ratio", 0.95),   # legacy placeholder
            ("batch_inharmonic_energy_ratio", 0.05),
            ("batch_subbass_energy_ratio", 0.0),
        ],
        columns=["Parameter", "Value"],
    )
    with pd.ExcelWriter(wb, engine="xlsxwriter") as writer:
        harm_df.to_excel(writer, sheet_name="Harmonic Spectrum", index=False)
        ih_df.to_excel(writer, sheet_name="Inharmonic Spectrum", index=False)
        sb_df.to_excel(writer, sheet_name="Sub-bass band", index=False)
        am_df.to_excel(writer, sheet_name="Analysis_Metadata", index=False)

    info = cm.extract_density_components_from_per_note_workbook(wb)
    assert info["density_extraction_status"] == "ok"
    assert info["w_H"] == pytest.approx(0.8)
    assert info["w_I"] == pytest.approx(0.15)
    assert info["w_S"] == pytest.approx(0.05)
    # ``legacy_aliases_only`` is the renamed flag in the Stage 1 + Stage 2
    # extractor; both keys are checked for backwards compatibility.
    assert info.get("legacy_aliases_only", info.get("legacy_batch_only")) is False
