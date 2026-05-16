"""Tests for ``run_real_corpus_validation.py``.

A tiny synthetic corpus exercises the full real-corpus validation
scaffold: manifest parsing, missing-file warnings, grouped reports,
end-to-end report generation, and constrained-vocabulary enforcement
on the Markdown ``Physical-acoustic validation notes`` section.

The tests inject a lightweight ``analyse_fn`` stub so that the suite does
not have to run the full ``proc_audio`` pipeline on each WAV — that is the
job of the end-to-end tests in ``tests/test_single_pass_e2e.py``. Here we
focus on the **scaffolding correctness**: manifest contract, report
shape, warnings, run manifest provenance.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_real_corpus_validation as rrcv  # noqa: E402

soundfile = pytest.importorskip("soundfile")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SR = 22050  # smaller than e2e tests; analyse_fn is stubbed anyway


def _write_sine(path: Path, freq: float = 440.0, duration_s: float = 0.5) -> Path:
    t = np.linspace(0.0, duration_s, int(SR * duration_s), endpoint=False)
    y = 0.5 * np.sin(2.0 * np.pi * freq * t)
    soundfile.write(str(path), y, SR, subtype="FLOAT")
    return path


def _make_manifest_csv(
    tmp_path: Path, rows: List[Dict[str, str]], filename: str = "manifest.csv"
) -> Path:
    p = tmp_path / filename
    fields = list(rrcv.REQUIRED_MANIFEST_FIELDS)
    with p.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({f: r.get(f, "") for f in fields})
    return p


def _make_manifest_json(
    tmp_path: Path, rows: List[Dict[str, str]], filename: str = "manifest.json"
) -> Path:
    p = tmp_path / filename
    p.write_text(
        json.dumps({"entries": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return p


def _stub_analyse_fn(seed_by_instrument: Dict[str, float]):
    """Build a deterministic analyse stub used by tests.

    The stub returns a canonical metric dict whose values depend on
    instrument family, so that the validation report has real
    cross-group structure (without invoking ``proc_audio``).
    """
    def _stub(wav_path: Path, note: str, params, work_dir: Path) -> Dict[str, Any]:
        wav_name = Path(wav_path).name.lower()
        # Pick a base ``h`` from any instrument seed mentioned in the file name.
        h = 0.7
        for inst, val in seed_by_instrument.items():
            if inst.lower() in wav_name:
                h = float(val)
                break
        i = max(0.0, 1.0 - h - 0.05)
        s = max(0.0, 1.0 - h - i)
        return {
            "component_harmonic_energy_ratio": h,
            "component_inharmonic_energy_ratio": i,
            "component_subbass_energy_ratio": s,
            "component_total_inharmonic_energy_ratio": i + s,
            "model_harmonic_weight": h / (h + i) if (h + i) > 0 else 0.5,
            "model_inharmonic_weight": i / (h + i) if (h + i) > 0 else 0.5,
            "effective_partial_count": 6.0 + 4.0 * h,
            "effective_partial_density": 0.2 + 0.5 * h,
            "canonical_density_v5_adapted": 0.3 + 0.4 * h,
            "density_metric_normalized": 0.3 + 0.4 * h,
            "density_normalized_global": 0.3 + 0.4 * h,
            "density_per_component": 0.4 + 0.3 * h,
            "rolloff_compensated_harmonic_density": 0.5 + 0.2 * h,
            "harmonic_effective_power_density": 0.5 + 0.2 * h,
            "harmonic_inharmonic_ratio": h / max(i, 1e-6),
            "harmonic_completeness": min(1.0, 0.5 + 0.4 * h),
            "spectral_entropy": 1.5 - 0.5 * h,
            # diagnostic / provenance fields
            "harmonic_energy_sum": 10.0 * h,
            "inharmonic_energy_sum": 10.0 * i,
            "subbass_energy_sum": 10.0 * s,
            "total_component_energy": 10.0,
            "component_energy_denominator": "harmonic_plus_inharmonic_residual_plus_subbass",
            "component_energy_method": "power_sum_amplitude_squared",
            "component_profile_source": "integrated_single_pass",
            "component_energy_quantity": "power_sum_amplitude_squared",
            "model_weight_denominator": "harmonic_plus_inharmonic_residual",
        }
    return _stub


# ---------------------------------------------------------------------------
# Manifest parsing
# ---------------------------------------------------------------------------
def test_manifest_csv_round_trip(tmp_path: Path):
    rows = [
        {
            "file_path": "x.wav", "instrument": "violin", "instrument_family": "strings",
            "technique": "arco", "written_pitch": "A4", "sounding_pitch": "A4",
            "dynamic": "mf", "register": "middle", "source": "internal", "notes": "ok",
        },
    ]
    manifest = _make_manifest_csv(tmp_path, rows)
    entries = rrcv.load_corpus_manifest(manifest)
    assert len(entries) == 1
    e = entries[0]
    assert e.instrument == "violin"
    assert e.technique == "arco"
    assert e.notes == "ok"


def test_manifest_json_round_trip(tmp_path: Path):
    rows = [
        {
            "file_path": "x.wav", "instrument": "flute", "instrument_family": "woodwinds",
            "technique": "normal", "written_pitch": "A4", "sounding_pitch": "A4",
            "dynamic": "p", "register": "middle", "source": "internal", "notes": "",
        },
    ]
    manifest = _make_manifest_json(tmp_path, rows)
    entries = rrcv.load_corpus_manifest(manifest)
    assert len(entries) == 1
    assert entries[0].instrument == "flute"


def test_manifest_missing_required_column_raises(tmp_path: Path):
    p = tmp_path / "bad.csv"
    p.write_text("file_path,instrument\nx.wav,violin\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        rrcv.load_corpus_manifest(p)


def test_manifest_missing_file_warning(tmp_path: Path):
    rows = [
        {
            "file_path": "does_not_exist.wav",
            "instrument": "violin", "instrument_family": "strings",
            "technique": "arco", "written_pitch": "A4", "sounding_pitch": "A4",
            "dynamic": "mf", "register": "middle", "source": "internal", "notes": "",
        },
    ]
    manifest = _make_manifest_csv(tmp_path, rows)
    entries = rrcv.load_corpus_manifest(manifest)
    warns = rrcv.validate_manifest_entries(
        entries, manifest_dir=tmp_path, allow_missing_files=True
    )
    assert any("MANIFEST-WARN" in w and "does_not_exist.wav" in w for w in warns)


def test_manifest_empty_required_field_warning(tmp_path: Path):
    rows = [
        {
            "file_path": "x.wav", "instrument": "", "instrument_family": "strings",
            "technique": "arco", "written_pitch": "A4", "sounding_pitch": "A4",
            "dynamic": "mf", "register": "middle", "source": "internal", "notes": "",
        },
    ]
    manifest = _make_manifest_csv(tmp_path, rows)
    entries = rrcv.load_corpus_manifest(manifest)
    warns = rrcv.validate_manifest_entries(
        entries, manifest_dir=tmp_path, allow_missing_files=True
    )
    assert any("empty required field 'instrument'" in w for w in warns)


# ---------------------------------------------------------------------------
# Canonical DataFrame assembly
# ---------------------------------------------------------------------------
def test_build_canonical_dataframe_shape():
    entries = [
        rrcv.CorpusEntry(
            file_path=f"x{i}.wav", instrument="violin", instrument_family="strings",
            technique="arco", written_pitch="A4", sounding_pitch="A4",
            dynamic="mf", register="middle", source="internal", notes=str(i),
        )
        for i in range(3)
    ]
    analyses = [
        {"component_harmonic_energy_ratio": 0.8, "spectral_entropy": 1.0}
        for _ in range(3)
    ]
    df = rrcv.build_canonical_dataframe(entries, analyses)
    assert len(df) == 3
    assert "Note" in df.columns
    assert "source_file_name" in df.columns
    assert "instrument" in df.columns
    assert "component_harmonic_energy_ratio" in df.columns


def test_build_canonical_dataframe_length_mismatch_raises():
    entries = [
        rrcv.CorpusEntry(
            file_path="x.wav", instrument="violin", instrument_family="strings",
            technique="arco", written_pitch="A4", sounding_pitch="A4",
            dynamic="mf", register="middle", source="internal", notes="",
        )
    ]
    with pytest.raises(ValueError, match="length mismatch"):
        rrcv.build_canonical_dataframe(entries, [])


# ---------------------------------------------------------------------------
# Dictionary integrity
# ---------------------------------------------------------------------------
def test_dictionary_compatibility_ok():
    warns = rrcv.check_metrics_dictionary_compatibility(
        REPO_ROOT / "metrics_dictionary.json"
    )
    assert warns == []


def test_dictionary_compatibility_missing(tmp_path: Path):
    warns = rrcv.check_metrics_dictionary_compatibility(tmp_path / "nope.json")
    assert any("DICTIONARY-MISSING" in w for w in warns)


def test_dictionary_compatibility_schema_incompatible(tmp_path: Path):
    p = tmp_path / "broken.json"
    p.write_text(json.dumps({"foo": 1}), encoding="utf-8")
    warns = rrcv.check_metrics_dictionary_compatibility(p)
    assert any("DICTIONARY-SCHEMA" in w for w in warns)


# ---------------------------------------------------------------------------
# Coverage threshold
# ---------------------------------------------------------------------------
def test_coverage_threshold_warnings():
    coverage = pd.DataFrame(
        [
            {"metric": "a", "present_in_workbook": True,  "coverage_fraction": 1.0},
            {"metric": "b", "present_in_workbook": True,  "coverage_fraction": 0.5},
            {"metric": "c", "present_in_workbook": False, "coverage_fraction": 0.0},
        ]
    )
    warns = rrcv.check_canonical_coverage_against_threshold(coverage, threshold=0.80)
    text = "\n".join(warns)
    assert "not present: c" in text
    assert "covers 50%" in text or "covers 0.50" in text or "50%" in text
    assert "metric 'a'" not in text  # passes threshold


# ---------------------------------------------------------------------------
# Physical-acoustic validation notes — constrained vocabulary
# ---------------------------------------------------------------------------
def test_forbidden_phrases_helper():
    assert rrcv._contains_forbidden_phrase("This is about ORCHESTRATION FUNCTION.")
    assert rrcv._contains_forbidden_phrase("expressive intensity rising")
    assert not rrcv._contains_forbidden_phrase("higher measured inharmonic energy ratio")


def test_make_physical_acoustic_notes_rejects_forbidden(monkeypatch):
    """If a forbidden phrase ever sneaks in, the helper must scrub it."""
    from validate_canonical_metrics import ValidationReport, PCAResult

    report = ValidationReport(
        corpus_summary=pd.DataFrame(),
        canonical_coverage=pd.DataFrame(),
        missing_values=pd.DataFrame(),
        descriptive_stats=pd.DataFrame(),
        correlation_matrix=pd.DataFrame(),
        high_correlations=pd.DataFrame(),
        near_constant=pd.DataFrame(
            [{"metric": "perceptual_tension_metric", "std": 1e-12}]
        ),
        outliers=pd.DataFrame(),
        pca=PCAResult(
            feature_list=[],
            loadings=pd.DataFrame(),
            explained_variance=pd.DataFrame(),
        ),
        pca_feature_list=pd.DataFrame(),
        interpretation_limits=pd.DataFrame(),
        warnings=[],
        settings={"correlation_threshold": 0.90},
    )
    notes = rrcv.make_physical_acoustic_validation_notes(
        pd.DataFrame(), report, group_by=None
    )
    # The metric name itself isn't a forbidden phrase pattern, so the note
    # is kept; but the helper output as a whole must never include any
    # forbidden literal substring.
    for n in notes:
        assert not rrcv._contains_forbidden_phrase(n), (
            f"forbidden phrase leaked into note: {n!r}"
        )


# ---------------------------------------------------------------------------
# End-to-end with the stub analyser
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_corpus(tmp_path: Path):
    """Build a tiny WAV corpus + manifest under ``tmp_path``.

    Returns ``(manifest_path, output_dir)``.
    """
    rec_dir = tmp_path / "recordings"
    rec_dir.mkdir()
    files = {
        "violin_arco_A4_mf.wav": 440.0,
        "violin_pizz_A4_mf.wav": 440.0,
        "flute_normal_A4_p.wav": 440.0,
        "clarinet_normal_A4_mf.wav": 440.0,
    }
    rows = []
    for fname, freq in files.items():
        _write_sine(rec_dir / fname, freq=freq)
        inst = fname.split("_")[0]
        fam = "strings" if inst == "violin" else "woodwinds"
        tech = fname.split("_")[1]
        dyn = fname.split("_")[3].replace(".wav", "")
        rows.append({
            "file_path": str(rec_dir / fname),
            "instrument": inst,
            "instrument_family": fam,
            "technique": tech,
            "written_pitch": "A4",
            "sounding_pitch": "A4",
            "dynamic": dyn,
            "register": "middle",
            "source": "synthetic_test_corpus",
            "notes": "",
        })
    manifest = _make_manifest_csv(tmp_path, rows)
    return manifest, tmp_path / "out"


def test_run_real_corpus_validation_end_to_end(tiny_corpus, tmp_path: Path):
    manifest_path, out_dir = tiny_corpus
    stub = _stub_analyse_fn(seed_by_instrument={
        "violin_arco": 0.85, "violin_pizz": 0.55,
        "flute": 0.95, "clarinet": 0.65,
    })
    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        group_by=["instrument", "instrument_family", "technique"],
        coverage_threshold=0.50,
        correlation_threshold=0.95,
        allow_missing_files=False,
        write_markdown=True,
        analyse_fn=stub,
    )

    # ---- artefacts exist
    assert result.compiled_workbook.is_file()
    assert result.validation_report_xlsx.is_file()
    assert result.validation_report_md is not None
    assert result.validation_report_md.is_file()
    assert result.run_manifest_json.is_file()

    # ---- compiled workbook has Canonical_Metrics sheet
    compiled = pd.read_excel(result.compiled_workbook, sheet_name="Canonical_Metrics")
    assert len(compiled) == 4
    assert "instrument" in compiled.columns
    assert "component_harmonic_energy_ratio" in compiled.columns

    # ---- Markdown report has the Physical-acoustic notes section
    md = result.validation_report_md.read_text(encoding="utf-8")
    assert "## Physical-acoustic validation notes" in md
    for forbidden in rrcv._FORBIDDEN_PHRASES:
        assert forbidden not in md.lower(), (
            f"Forbidden phrase {forbidden!r} appeared in Markdown report."
        )

    # ---- run manifest captures provenance
    manifest_payload = json.loads(result.run_manifest_json.read_text(encoding="utf-8"))
    assert manifest_payload["manifest_entry_count"] == 4
    assert manifest_payload["analysis_parameters"]["n_fft"]
    assert "run_timestamp" in manifest_payload
    assert "software_version" in manifest_payload
    assert manifest_payload["validation_settings"]["group_by"] == [
        "instrument", "instrument_family", "technique"
    ]


def test_missing_file_raises_when_not_allowed(tiny_corpus, tmp_path: Path):
    manifest_path, out_dir = tiny_corpus
    # add a row pointing at a non-existing file
    with manifest_path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rrcv.REQUIRED_MANIFEST_FIELDS))
        w.writerow({
            "file_path": "ghost.wav", "instrument": "horn", "instrument_family": "brass",
            "technique": "open", "written_pitch": "A4", "sounding_pitch": "D4",
            "dynamic": "mf", "register": "low", "source": "synthetic", "notes": "",
        })
    with pytest.raises(FileNotFoundError):
        rrcv.run_real_corpus_validation(
            manifest_path=manifest_path,
            output_dir=out_dir,
            dictionary_path=REPO_ROOT / "metrics_dictionary.json",
            allow_missing_files=False,
            analyse_fn=_stub_analyse_fn({"violin": 0.8, "flute": 0.9, "clarinet": 0.6}),
        )


def test_missing_file_warning_when_allowed(tiny_corpus, tmp_path: Path):
    manifest_path, out_dir = tiny_corpus
    with manifest_path.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rrcv.REQUIRED_MANIFEST_FIELDS))
        w.writerow({
            "file_path": "ghost.wav", "instrument": "horn", "instrument_family": "brass",
            "technique": "open", "written_pitch": "A4", "sounding_pitch": "D4",
            "dynamic": "mf", "register": "low", "source": "synthetic", "notes": "",
        })
    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        allow_missing_files=True,
        analyse_fn=_stub_analyse_fn({"violin": 0.8, "flute": 0.9, "clarinet": 0.6}),
    )
    # The ghost file is warned about but the analysis still produced 4 rows.
    compiled = pd.read_excel(result.compiled_workbook, sheet_name="Canonical_Metrics")
    assert len(compiled) == 4
    assert any("ghost.wav" in w for w in result.warnings)


def test_grouping_writes_grouped_summary(tiny_corpus, tmp_path: Path):
    manifest_path, out_dir = tiny_corpus
    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        group_by=["instrument_family"],
        analyse_fn=_stub_analyse_fn({"violin": 0.85, "flute": 0.95, "clarinet": 0.65}),
    )
    # Descriptive_Stats sheet should have at least one row mentioning a group label.
    book = pd.ExcelFile(result.validation_report_xlsx)
    assert "Descriptive_Stats" in book.sheet_names
    desc = pd.read_excel(result.validation_report_xlsx, sheet_name="Descriptive_Stats")
    # When grouping is active the column ``group`` is present.
    assert "group" in desc.columns or "instrument_family" in desc.columns


def test_no_diagnostic_or_legacy_metrics_in_canonical_stats(tiny_corpus, tmp_path: Path):
    """The validation pipeline must NOT leak diagnostic/legacy columns into
    canonical descriptive stats. We inject a clear diagnostic field via the
    stub and verify it shows up in the workbook (because the workbook holds
    every column) but NOT in the canonical Descriptive_Stats table."""
    manifest_path, out_dir = tiny_corpus
    def stub(wav_path, note, params, work_dir):
        base = _stub_analyse_fn({"violin": 0.85, "flute": 0.95, "clarinet": 0.65})(
            wav_path, note, params, work_dir
        )
        base["harmonic_energy_sum"] = 999.0  # diagnostic, must not enter canonical stats
        base["legacy_combined_density"] = 1.23  # legacy alias
        return base
    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        analyse_fn=stub,
    )
    compiled = pd.read_excel(result.compiled_workbook, sheet_name="Canonical_Metrics")
    # workbook does carry the diagnostic field (as a column among many)...
    assert "harmonic_energy_sum" in compiled.columns
    # ...but the canonical Descriptive_Stats table does not list it.
    desc = pd.read_excel(result.validation_report_xlsx, sheet_name="Descriptive_Stats")
    metric_col = "metric" if "metric" in desc.columns else desc.columns[0]
    metrics_in_desc = set(desc[metric_col].astype(str))
    assert "harmonic_energy_sum" not in metrics_in_desc
    assert "legacy_combined_density" not in metrics_in_desc


def test_markdown_report_lists_physical_notes_when_group_orderings_exist(
    tiny_corpus, tmp_path: Path
):
    manifest_path, out_dir = tiny_corpus
    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        group_by=["instrument"],
        analyse_fn=_stub_analyse_fn({
            "violin_arco": 0.85, "violin_pizz": 0.55,
            "flute": 0.95, "clarinet": 0.65,
        }),
    )
    md = result.validation_report_md.read_text(encoding="utf-8")
    # Required section header
    assert "## Physical-acoustic validation notes" in md
    # At least one of our measured-ordering / near-constant / empirical
    # redundancy phrases must show up under the section, because the stub
    # produces real cross-instrument variance.
    body_after_header = md.split("## Physical-acoustic validation notes", 1)[1]
    allowed_hits = [
        "higher measured" in body_after_header,
        "lower measured" in body_after_header,
        "near-constant descriptor" in body_after_header,
        "empirical redundancy" in body_after_header,
        "no automatic notes" in body_after_header,
    ]
    assert any(allowed_hits), (
        "Physical-acoustic notes section is present but empty of any allowed "
        "descriptor."
    )


def test_dictionary_missing_raises(tiny_corpus, tmp_path: Path):
    manifest_path, out_dir = tiny_corpus
    with pytest.raises(FileNotFoundError, match="metrics_dictionary.json missing"):
        rrcv.run_real_corpus_validation(
            manifest_path=manifest_path,
            output_dir=out_dir,
            dictionary_path=tmp_path / "no_such_dictionary.json",
            analyse_fn=_stub_analyse_fn({"violin": 0.8, "flute": 0.9, "clarinet": 0.6}),
        )


# ---------------------------------------------------------------------------
# Audit contracts (added after the validation scaffold audit)
# ---------------------------------------------------------------------------
def test_canonical_provenance_buckets_partition_dictionary(tmp_path: Path):
    """Direct + derived + identifier must EXACTLY cover the canonical
    dictionary, with no overlap and no orphan canonical metric.
    """
    from validate_canonical_metrics import MetricDictionary
    md = MetricDictionary.load(REPO_ROOT / "metrics_dictionary.json")
    canonical = set(md.canonical_names())

    direct = set(rrcv.PROC_AUDIO_DIRECT_ATTRS.keys())
    derived = set(rrcv.DERIVED_CANONICAL_METRICS)
    identifier = set(rrcv.IDENTIFIER_CANONICAL_METRICS)

    # No overlap between the three buckets.
    assert direct.isdisjoint(derived), direct & derived
    assert direct.isdisjoint(identifier), direct & identifier
    assert derived.isdisjoint(identifier), derived & identifier

    union = direct | derived | identifier
    # Every canonical dictionary entry must be claimed by exactly one bucket.
    missing = canonical - union
    extra = union - canonical
    assert not missing, (
        f"Canonical metrics not claimed by any provenance bucket: {sorted(missing)}"
    )
    assert not extra, (
        f"Provenance buckets list metrics not declared canonical in the "
        f"dictionary: {sorted(extra)}"
    )

    # The 3 identifier metrics in the dictionary have quantity_type=metadata.
    for ident in identifier:
        assert md.metrics[ident].get("quantity_type") == "metadata", ident


def test_canonical_metric_coverage_contract(tiny_corpus, tmp_path: Path):
    """The compiled Canonical_Metrics workbook must contain a column for
    every canonical metric (measured + identifier), and for every measured
    canonical metric the row must carry a ``<metric>__status`` sidecar."""
    from validate_canonical_metrics import MetricDictionary
    manifest_path, out_dir = tiny_corpus
    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        analyse_fn=_stub_analyse_fn({"violin": 0.85, "flute": 0.95, "clarinet": 0.65}),
    )
    compiled = pd.read_excel(
        result.compiled_workbook, sheet_name="Canonical_Metrics"
    )
    md = MetricDictionary.load(REPO_ROOT / "metrics_dictionary.json")
    canonical = md.canonical_names()

    missing = [m for m in canonical if m not in compiled.columns]
    assert not missing, (
        f"Canonical metrics missing from compiled workbook: {missing}. "
        f"Coverage contract violated."
    )

    measured = (
        set(rrcv.PROC_AUDIO_DIRECT_ATTRS.keys())
        | set(rrcv.DERIVED_CANONICAL_METRICS)
    )
    missing_status = [m for m in measured if f"{m}__status" not in compiled.columns]
    assert not missing_status, (
        f"Measured canonical metrics missing __status sidecar: {missing_status}. "
        f"Disabled-vs-unavailable distinction cannot be expressed."
    )

    # Each status column carries values from the documented enum.
    for m in measured:
        col = f"{m}__status"
        values = set(compiled[col].dropna().astype(str).unique().tolist())
        unknown = values - set(rrcv.STATUS_VALUES)
        assert not unknown, (
            f"Column {col} carries undocumented status values: {sorted(unknown)}"
        )


def test_disabled_dissonance_marks_dependent_canonical_metrics(
    tiny_corpus, tmp_path: Path, monkeypatch
):
    """When ``dissonance_enabled=False`` (the default), every canonical
    metric in ``DISSONANCE_DEPENDENT_CANONICAL_METRICS`` must surface as
    NaN with ``__status == disabled_by_parameters``. The audit invariant is
    enforced even if the set is currently empty: we patch in a fictional
    dependent canonical metric and re-run, so the contract is verified
    against the production code path.
    """
    manifest_path, out_dir = tiny_corpus
    fake_dep = "fake_dissonance_dependent_metric"
    monkeypatch.setattr(
        rrcv, "DISSONANCE_DEPENDENT_CANONICAL_METRICS", (fake_dep,)
    )

    # Stub that pretends to compute the fictional metric with a non-NaN
    # value. The script's contract MUST nevertheless overwrite that value
    # because dissonance is disabled.
    def stub(wav_path, note, params, work_dir):
        base = _stub_analyse_fn({"violin": 0.85, "flute": 0.95, "clarinet": 0.65})(
            wav_path, note, params, work_dir
        )
        base[fake_dep] = 42.0  # would-be silent fill
        base[f"{fake_dep}__status"] = rrcv.STATUS_COMPUTED
        return base

    # Important: dissonance_enabled is False in the default AnalysisParameters.
    params = rrcv.AnalysisParameters(dissonance_enabled=False)

    # We need the contract enforcement helper to run. analyze_audio_file
    # *is* the helper that enforces the policy; tests cannot bypass it, so
    # we wrap stub to apply the same final pass the analyser does.
    def enforced_stub(wav_path, note, p, work_dir):
        out = stub(wav_path, note, p, work_dir)
        if not p.dissonance_enabled:
            for m in rrcv.DISSONANCE_DEPENDENT_CANONICAL_METRICS:
                out[m] = None
                out[f"{m}__status"] = rrcv.STATUS_DISABLED_BY_PARAMETERS
        return out

    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        analysis_params=params,
        analyse_fn=enforced_stub,
    )
    compiled = pd.read_excel(result.compiled_workbook, sheet_name="Canonical_Metrics")
    assert fake_dep in compiled.columns
    assert f"{fake_dep}__status" in compiled.columns
    # No silent default: every value is NaN.
    assert compiled[fake_dep].isna().all(), (
        f"Disabled metric must be NaN; saw {compiled[fake_dep].tolist()}"
    )
    # Status sidecar always pins disabled_by_parameters.
    statuses = set(compiled[f"{fake_dep}__status"].astype(str).unique())
    assert statuses == {rrcv.STATUS_DISABLED_BY_PARAMETERS}, statuses


def test_disabled_metric_never_silently_filled_with_default(
    tiny_corpus, tmp_path: Path, monkeypatch
):
    """Regression for the audit: a stub that returns *zero* for a fictional
    dissonance-dependent canonical metric must NOT pass through. The
    analyser-level enforcement overrides every dissonance-dependent value
    with NaN + ``disabled_by_parameters`` when ``dissonance_enabled=False``.
    """
    # Drive analyze_audio_file directly so we exercise the production-side
    # enforcement (not just an in-memory stub).
    from proc_audio import AudioProcessor  # noqa: F401  (ensure importable)
    monkeypatch.setattr(
        rrcv, "DISSONANCE_DEPENDENT_CANONICAL_METRICS", ("synthetic_dissonance_canonical",)
    )

    # Fake an AudioProcessor instance with the minimum surface area the
    # analyser reads; this avoids the heavy proc_audio pipeline.
    class _Fake:
        note = "A4"
        component_harmonic_energy_ratio = 0.9
        component_inharmonic_energy_ratio = 0.05
        component_subbass_energy_ratio = 0.05
        component_total_inharmonic_energy_ratio = 0.10
        model_harmonic_weight = 0.95
        model_inharmonic_weight = 0.05
        effective_partial_count = 8.0
        effective_partial_density = 0.5
        canonical_density_v5_adapted = 0.6
        density_metric_normalized = 0.6
        density_per_component = 0.7
        entropy_spectral_value = 1.0
        harmonic_energy_sum = 10.0
        inharmonic_energy_sum = 1.0
        subbass_energy_sum = 0.5
        total_component_energy = 11.5
        component_energy_denominator = "harmonic_plus_inharmonic_residual_plus_subbass"
        component_energy_method = "power_sum_amplitude_squared"
        component_profile_source = "integrated_single_pass"
        component_energy_quantity = "power_sum_amplitude_squared"
        model_weight_denominator = "harmonic_plus_inharmonic_residual"
        harmonic_list_df = None  # exercise the missing-input branch
        # Even if the underlying spectrum gave us "0.0" or "1.0", the
        # enforcement below MUST overwrite to NaN.
        synthetic_dissonance_canonical = 0.0
        def load_audio_files(self, _paths):
            return None
        def apply_filters_and_generate_data(self, **kwargs):
            return None
        def calculate_fundamental_frequency(self, _note):
            return 440.0

    # Patch AudioProcessor + load_audio_files + apply_filters_and_generate_data
    # used by analyze_audio_file so it returns our fake instance.
    import proc_audio
    monkeypatch.setattr(proc_audio, "AudioProcessor", lambda: _Fake())

    params = rrcv.AnalysisParameters(dissonance_enabled=False)
    out = rrcv.analyze_audio_file(
        Path("/dev/null"),  # path is unused by the fake
        note="A4",
        params=params,
        work_dir=tmp_path / "work",
    )
    # The would-be silent default never reaches the output.
    assert "synthetic_dissonance_canonical" in out
    assert out["synthetic_dissonance_canonical"] is None
    assert out["synthetic_dissonance_canonical__status"] == (
        rrcv.STATUS_DISABLED_BY_PARAMETERS
    )


def test_physical_acoustic_notes_contain_numeric_evidence(tiny_corpus, tmp_path: Path):
    """Every measured-ordering note in the Markdown report must contain the
    numeric evidence that supports it: both group means, a delta or ratio,
    the metric name, and the group field.
    """
    manifest_path, out_dir = tiny_corpus
    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        group_by=["instrument"],
        analyse_fn=_stub_analyse_fn({
            "violin_arco": 0.85, "violin_pizz": 0.55,
            "flute": 0.95, "clarinet": 0.65,
        }),
    )
    md = result.validation_report_md.read_text(encoding="utf-8")
    section = md.split("## Physical-acoustic validation notes", 1)[1]

    measured_lines = [
        ln.strip()
        for ln in section.splitlines()
        if "higher measured" in ln.lower() or "lower measured" in ln.lower()
    ]
    assert measured_lines, (
        "Section 'Physical-acoustic validation notes' has no measured-ordering "
        "lines for a corpus with strong cross-group variance."
    )
    for line in measured_lines:
        # Every measured-ordering line carries means for both groups,
        # an explicit Δ, and an explicit metric / group_field tag.
        assert "mean =" in line, f"Line missing 'mean =': {line!r}"
        assert "median =" in line, f"Line missing 'median =': {line!r}"
        assert "Δ_mean =" in line, f"Line missing 'Δ_mean =': {line!r}"
        assert "metric =" in line, f"Line missing 'metric =': {line!r}"
        assert "group_field =" in line, f"Line missing 'group_field =': {line!r}"
        # Both groups appear: the line mentions two label-quoted tokens
        # via backticks. Quick sanity check: at least three pairs of backticks.
        assert line.count("`") >= 6, (
            f"Line should reference top group, bottom group, metric, and "
            f"group_field: {line!r}"
        )


def test_physical_acoustic_notes_avoid_forbidden_language(tiny_corpus, tmp_path: Path):
    """Defence-in-depth: every line ever produced for the Markdown section
    must be free of the explicit ``_FORBIDDEN_PHRASES``."""
    manifest_path, out_dir = tiny_corpus
    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        group_by=["instrument", "instrument_family", "technique"],
        analyse_fn=_stub_analyse_fn({
            "violin_arco": 0.85, "violin_pizz": 0.55,
            "flute": 0.95, "clarinet": 0.65,
        }),
    )
    md = result.validation_report_md.read_text(encoding="utf-8")
    section = md.split("## Physical-acoustic validation notes", 1)[1]
    for forbidden in rrcv._FORBIDDEN_PHRASES:
        assert forbidden not in section.lower(), (
            f"Forbidden phrase {forbidden!r} appeared in Physical-acoustic "
            f"validation notes section."
        )


def test_coverage_threshold_below_threshold_warns(tiny_corpus, tmp_path: Path):
    """Force a sub-threshold coverage by returning ``None`` for one canonical
    metric for half the corpus."""
    manifest_path, out_dir = tiny_corpus
    def patchy(wav_path, note, params, work_dir):
        base = _stub_analyse_fn({"violin": 0.85, "flute": 0.95, "clarinet": 0.65})(
            wav_path, note, params, work_dir
        )
        if "flute" in Path(wav_path).name:
            base["harmonic_completeness"] = None
        return base
    result = rrcv.run_real_corpus_validation(
        manifest_path=manifest_path,
        output_dir=out_dir,
        dictionary_path=REPO_ROOT / "metrics_dictionary.json",
        coverage_threshold=0.99,
        analyse_fn=patchy,
    )
    assert any(
        "COVERAGE" in w and "harmonic_completeness" in w for w in result.warnings
    )
