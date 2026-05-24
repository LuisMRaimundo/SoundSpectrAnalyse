from __future__ import annotations

from pathlib import Path

from validate_canonical_metrics import MetricDictionary


REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def test_required_docs_exist() -> None:
    for p in (
        REPO_ROOT / "docs" / "TECHNICAL_MANUAL.md",
        REPO_ROOT / "docs" / "QUICK_GUIDE.md",
        REPO_ROOT / "docs" / "TUTORIAL.md",
    ):
        assert p.is_file(), str(p)


def test_readme_links_to_required_docs() -> None:
    readme = _read(REPO_ROOT / "README.md")
    assert "docs/TECHNICAL_MANUAL.md" in readme
    assert "docs/QUICK_GUIDE.md" in readme
    assert "docs/TUTORIAL.md" in readme


def test_technical_manual_contains_required_final_density_content() -> None:
    manual = _read(REPO_ROOT / "docs" / "TECHNICAL_MANUAL.md")
    required_tokens = (
        "final_note_density_salience_weighted",
        "final_note_density_count_based",
        "salience_i",
        "final_note_density_count_based = wH*H + wI*I + wS*S",
        "X_m[k]",
        "f0_final(valid) -> f0_initial(valid) -> f0_prior_hz(valid) -> NaN",
        "Analysis_Settings_By_Note",
        "Legacy_Compatibility",
    )
    for token in required_tokens:
        assert token in manual, token


def test_quick_guide_declares_primary_final_density_metric() -> None:
    quick = _read(REPO_ROOT / "docs" / "QUICK_GUIDE.md")
    assert "final_note_density_salience_weighted" in quick


def test_forbidden_claims_are_absent() -> None:
    corpus = "\n".join(
        _read(REPO_ROOT / "docs" / name)
        for name in ("TECHNICAL_MANUAL.md", "QUICK_GUIDE.md", "TUTORIAL.md")
    ).lower()
    forbidden = (
        "density_metric_raw is the final density",
        "effective_partial_density is the final density",
        "combined density metric is the primary metric",
        "f0 fallback is acoustically verified",
    )
    for phrase in forbidden:
        assert phrase not in corpus


def test_metrics_dictionary_has_canonical_final_density_entries() -> None:
    dictionary = MetricDictionary.load(REPO_ROOT / "metrics_dictionary.json")
    for name in (
        "final_note_density_salience_weighted",
        "final_note_density_count_based",
        "salient_harmonic_order_count_up_to_density_ceiling_hz",
        "salient_inharmonic_log_bin_count_up_to_density_ceiling_hz",
        "salient_subbass_particle_count",
        "harmonic_density_component",
        "inharmonic_density_component",
        "subbass_density_component",
    ):
        assert name in dictionary.metrics


def test_legacy_metrics_are_not_marked_canonical() -> None:
    dictionary = MetricDictionary.load(REPO_ROOT / "metrics_dictionary.json")
    for name in (
        "density_metric_raw",
        "density_weighted_sum",
        "energy_weighted_component_density_diagnostic",
        "Combined Density Metric",
        "Weighted Combined Metric",
        "Total sum",
        "density_weighted_sum_cdm_mean",
    ):
        if name in dictionary.metrics:
            assert dictionary.metrics[name]["status"] in {"diagnostic", "legacy"}

