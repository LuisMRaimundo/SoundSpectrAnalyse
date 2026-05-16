"""Semantics tests for component balance pie charts (visualisation only)."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from proc_audio import (
    AudioProcessor,
    COMPONENT_AMPLITUDE_MASS_PIE_BASIS_FOOTNOTE,
    COMPONENT_AMPLITUDE_MASS_PIE_FILENAME,
    COMPONENT_AMPLITUDE_MASS_PIE_TITLE_PREFIX,
    COMPONENT_ENERGY_PIE_LEGACY_ALIAS_FILENAME,
    COMPONENT_ENERGY_RATIO_PIE_FILENAME,
    _COMPONENT_AMPLITUDE_MASS_PIE_LEGEND_LABELS,
    _energy_ratio_pie_values,
)


def _assert_amplitude_surface_non_misleading(text: str) -> None:
    lowered = text.lower()
    assert "inharmonic components" not in lowered
    assert "ground noise" not in lowered
    assert "sub-bass noise" not in lowered
    assert "inharmonic partials" not in lowered


def test_amplitude_mass_legends_and_title_are_non_misleading() -> None:
    surface = (
        " ".join(_COMPONENT_AMPLITUDE_MASS_PIE_LEGEND_LABELS)
        + " "
        + COMPONENT_AMPLITUDE_MASS_PIE_TITLE_PREFIX
        + " "
        + COMPONENT_AMPLITUDE_MASS_PIE_BASIS_FOOTNOTE
    )
    _assert_amplitude_surface_non_misleading(surface)
    assert "amplitude" in COMPONENT_AMPLITUDE_MASS_PIE_TITLE_PREFIX.lower()
    assert "amplitude" in COMPONENT_AMPLITUDE_MASS_PIE_BASIS_FOOTNOTE.lower()


def test_energy_ratio_pie_values_clarinet_like_row() -> None:
    trip = _energy_ratio_pie_values(0.9945, 0.0055, 0.0)
    assert trip is not None
    h, i, s = trip
    assert abs(h + i + s - 1.0) < 1e-9
    assert abs(h - 0.9945) < 1e-9
    assert abs(i - 0.0055) < 1e-9
    assert s == 0.0


def test_energy_ratio_pie_values_none_without_ratios() -> None:
    assert _energy_ratio_pie_values(None, 0.5, 0.0) is None
    assert _energy_ratio_pie_values(0.5, None, 0.0) is None


def test_energy_ratio_pie_values_subbass_none_defaults_zero() -> None:
    trip = _energy_ratio_pie_values(0.8, 0.2, None)
    assert trip == (0.8, 0.2, 0.0)


def test_save_spectral_data_to_excel_accepts_export_output_dir() -> None:
    sig = inspect.signature(AudioProcessor._save_spectral_data_to_excel)
    assert "export_output_dir" in sig.parameters


def test_component_balance_pies_outputs_and_metadata(tmp_path: Path) -> None:
    proc = AudioProcessor()
    proc.linear_sum_amplitude_harmonic = 0.819
    proc.linear_sum_amplitude_inharmonic_partial = 0.127
    proc.linear_sum_amplitude_subbass_band = 0.055
    proc.harmonic_energy_ratio = 0.9945
    proc.inharmonic_energy_ratio = 0.0055
    proc.subbass_energy_ratio = 0.0

    proc._save_component_balance_pies(tmp_path, "D5")

    amp = tmp_path / COMPONENT_AMPLITUDE_MASS_PIE_FILENAME
    legacy = tmp_path / COMPONENT_ENERGY_PIE_LEGACY_ALIAS_FILENAME
    en = tmp_path / COMPONENT_ENERGY_RATIO_PIE_FILENAME
    assert amp.is_file()
    assert legacy.is_file()
    assert en.is_file()
    assert amp.read_bytes() == legacy.read_bytes()

    assert proc.amplitude_mass_chart_file == COMPONENT_AMPLITUDE_MASS_PIE_FILENAME
    assert proc.energy_ratio_chart_file == COMPONENT_ENERGY_RATIO_PIE_FILENAME
    assert COMPONENT_AMPLITUDE_MASS_PIE_FILENAME != COMPONENT_ENERGY_RATIO_PIE_FILENAME
    assert proc.amplitude_mass_chart_basis == "linear_amplitude_sum"
    assert proc.energy_ratio_chart_basis == "component_power_energy_ratios"
    assert proc.amplitude_mass_chart_interpretation == "diagnostic_candidate_mass_not_energy"
    assert proc.energy_ratio_chart_interpretation == "acoustic_energy_balance"
    assert proc.amplitude_mass_chart_status == "saved"
    assert proc.energy_ratio_chart_status == "saved"
    assert proc.component_energy_pie_file == COMPONENT_ENERGY_PIE_LEGACY_ALIAS_FILENAME
    assert proc.component_energy_pie_alias_basis == "legacy_alias_of_amplitude_mass_chart"


def test_amplitude_and_legacy_logged_when_saved(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    proc = AudioProcessor()
    proc.linear_sum_amplitude_harmonic = 1.0
    proc.linear_sum_amplitude_inharmonic_partial = 0.2
    proc.linear_sum_amplitude_subbass_band = 0.0
    with caplog.at_level("INFO"):
        proc._save_component_balance_pies(tmp_path, "N2")
    joined = " ".join(rec.message for rec in caplog.records)
    assert "Candidate amplitude-mass pie saved:" in joined
    assert "Basis: linear amplitude sums; not power/energy ratios." in joined
    assert "Legacy component_energy_pie.png copied from component_amplitude_mass_pie.png:" in joined


def test_energy_ratio_chart_skipped_when_ratios_unavailable(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    proc = AudioProcessor()
    proc.linear_sum_amplitude_harmonic = 1.0
    proc.linear_sum_amplitude_inharmonic_partial = 0.2
    proc.linear_sum_amplitude_subbass_band = 0.0
    proc.harmonic_energy_ratio = None
    proc.inharmonic_energy_ratio = None
    proc.subbass_energy_ratio = None
    proc.component_harmonic_energy_ratio = None
    proc.component_inharmonic_energy_ratio = None
    proc.component_subbass_energy_ratio = None

    with caplog.at_level("WARNING"):
        proc._save_component_balance_pies(tmp_path, "N1")

    assert (tmp_path / COMPONENT_AMPLITUDE_MASS_PIE_FILENAME).is_file()
    assert (tmp_path / COMPONENT_ENERGY_RATIO_PIE_FILENAME).is_file() is False
    assert proc.energy_ratio_chart_file == ""
    assert proc.energy_ratio_chart_status == "skipped_missing_component_energy_ratios"
    assert "Component energy-ratio pie skipped: missing values" in caplog.text


def test_energy_ratio_uses_component_fallback_when_primary_missing(tmp_path: Path) -> None:
    proc = AudioProcessor()
    proc.linear_sum_amplitude_harmonic = 0.5
    proc.linear_sum_amplitude_inharmonic_partial = 0.3
    proc.linear_sum_amplitude_subbass_band = 0.2
    proc.harmonic_energy_ratio = None
    proc.inharmonic_energy_ratio = None
    proc.component_harmonic_energy_ratio = 0.9
    proc.component_inharmonic_energy_ratio = 0.1
    proc.component_subbass_energy_ratio = 0.0
    proc._save_component_balance_pies(tmp_path, "Fb")
    assert (tmp_path / COMPONENT_ENERGY_RATIO_PIE_FILENAME).is_file()
    assert proc.energy_ratio_chart_status == "saved"


def test_energy_pie_data_matches_stored_ratios_not_amplitude_sums() -> None:
    """Regression guard: energy wedges must follow *_energy_ratio, not ΣA."""
    proc = AudioProcessor()
    proc.linear_sum_amplitude_harmonic = 0.5
    proc.linear_sum_amplitude_inharmonic_partial = 0.4
    proc.linear_sum_amplitude_subbass_band = 0.1
    proc.harmonic_energy_ratio = 0.99
    proc.inharmonic_energy_ratio = 0.01
    proc.subbass_energy_ratio = 0.0
    trip = _energy_ratio_pie_values(
        proc.harmonic_energy_ratio,
        proc.inharmonic_energy_ratio,
        proc.subbass_energy_ratio,
    )
    assert trip == (0.99, 0.01, 0.0)
    lin_tot = (
        float(proc.linear_sum_amplitude_harmonic)
        + float(proc.linear_sum_amplitude_inharmonic_partial)
        + float(proc.linear_sum_amplitude_subbass_band)
    )
    assert abs(lin_tot - 1.0) < 1e-9
    assert trip[0] != 0.5


def test_metadata_like_rows_after_pies(tmp_path: Path) -> None:
    """Fields written to Analysis_Metadata in proc_audio — mirror here."""
    proc = AudioProcessor()
    proc.linear_sum_amplitude_harmonic = 0.6
    proc.linear_sum_amplitude_inharmonic_partial = 0.3
    proc.linear_sum_amplitude_subbass_band = 0.1
    proc.harmonic_energy_ratio = 0.95
    proc.inharmonic_energy_ratio = 0.05
    proc.subbass_energy_ratio = 0.0
    proc._save_component_balance_pies(tmp_path, "M1")
    meta = {
        "amplitude_mass_chart_file": proc.amplitude_mass_chart_file,
        "amplitude_mass_chart_basis": proc.amplitude_mass_chart_basis,
        "amplitude_mass_chart_interpretation": proc.amplitude_mass_chart_interpretation,
        "energy_ratio_chart_file": proc.energy_ratio_chart_file,
        "energy_ratio_chart_basis": proc.energy_ratio_chart_basis,
        "energy_ratio_chart_interpretation": proc.energy_ratio_chart_interpretation,
        "amplitude_mass_chart_status": proc.amplitude_mass_chart_status,
        "energy_ratio_chart_status": proc.energy_ratio_chart_status,
        "component_energy_pie_file": proc.component_energy_pie_file,
        "component_energy_pie_basis": proc.component_energy_pie_alias_basis,
    }
    assert meta["amplitude_mass_chart_file"] == COMPONENT_AMPLITUDE_MASS_PIE_FILENAME
    assert meta["amplitude_mass_chart_basis"] in ("linear_amplitude_sum", "harmonic_amplitude_sum")
    assert meta["amplitude_mass_chart_interpretation"] == "diagnostic_candidate_mass_not_energy"
    assert meta["energy_ratio_chart_file"] == COMPONENT_ENERGY_RATIO_PIE_FILENAME
    assert meta["energy_ratio_chart_basis"] == "component_power_energy_ratios"
    assert meta["energy_ratio_chart_interpretation"] == "acoustic_energy_balance"


def test_preferred_amplitude_triple_used_when_all_present(tmp_path: Path) -> None:
    proc = AudioProcessor()
    proc.harmonic_amplitude_sum = 0.7
    proc.inharmonic_amplitude_sum = 0.2
    proc.subbass_amplitude_sum = 0.1
    proc.linear_sum_amplitude_harmonic = 0.1
    proc.linear_sum_amplitude_inharmonic_partial = 0.1
    proc.linear_sum_amplitude_subbass_band = 0.1
    proc.harmonic_energy_ratio = 0.8
    proc.inharmonic_energy_ratio = 0.2
    proc.subbass_energy_ratio = 0.0
    proc._save_component_balance_pies(tmp_path, "P1")
    assert proc.amplitude_mass_chart_basis == "harmonic_amplitude_sum"
    assert (tmp_path / COMPONENT_AMPLITUDE_MASS_PIE_FILENAME).is_file()
