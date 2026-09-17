"""Prevent interrupted clean timing from being relabeled or pooled on collection."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def evidence(tmp_path):
    """A complete original series and a distinct diagnostics-only process."""
    source = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/experiments/issue409-packed-values/collect.py"
    )
    spec = importlib.util.spec_from_file_location("packed_evidence_collector", source)
    collector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(collector)
    clean = {
        "phase": "changed",
        "aos": 768,
        "case": "fixture",
        "native_identity": "native",
        "library_sha256": "library",
        "source_patch_sha256": "patch",
        "reference_sha256": "original-reference",
        "checkpoint_sha256": "checkpoint",
        "changed_reference_sha256": "changed-reference",
        "frozen_density_sha256": {"density_0": "density"},
        "coordinates_bohr": [[0, 0, 0]],
        "changed_coordinates_bohr": [[0.001, 0, 0]],
        "controls": {"VIBEQC_DF_FINAL_PROJECTION": "auto"},
        "slurm_job_id": "original-job",
        "runner_sha256": "original-runner",
        "samples": [
            {"repeat": i, "policy": p, "iterations": 9, "seconds": 1.0}
            for i in range(7)
            for p in (("dense", "packed") if i % 2 == 0 else ("packed", "dense"))
        ],
        "diagnostics": [],
    }
    companion = {
        **copy.deepcopy(clean),
        "samples": [],
        "slurm_job_id": "diagnostic-job",
        "runner_sha256": "diagnostic-runner",
        "diagnostics": [
            {
                "policy": p,
                "iterations": 9,
                "seconds": 2.0,
                "maximum_energy_error": 1e-11,
                "maximum_force_error": 1e-10,
                "convergence": [{"converged": True}],
            }
            for p in ("dense", "packed")
        ],
    }
    campaign = {
        "status": "passed",
        "diagnostics_only": True,
        "exit_code": 0,
        "slurm_job_id": "diagnostic-job",
    }
    clean_path = tmp_path / "original.json"
    directory = tmp_path / "companion"
    directory.mkdir()

    def save():
        clean_path.write_text(json.dumps(clean) + "\n")
        campaign["predecessor_sha256"] = hashlib.sha256(
            clean_path.read_bytes()
        ).hexdigest()
        (directory / "manifest.json").write_text(json.dumps(campaign) + "\n")
        (directory / "768-changed-clean.json").write_text(json.dumps(companion) + "\n")

    save()
    return (
        collector.bind_changed_diagnostics,
        clean_path,
        directory,
        clean,
        companion,
        campaign,
        save,
    )


def test_binding_keeps_original_samples_and_distinct_process_identity(evidence):
    bind, path, directory, clean, companion, _, _ = evidence
    original = path.read_bytes()
    joined = bind(path, directory)
    assert path.read_bytes() == original
    assert joined["samples"] == clean["samples"]
    assert joined["diagnostics"] == companion["diagnostics"]
    provenance = joined["diagnostic_companion"]
    assert provenance["clean_original_sha256"] == hashlib.sha256(original).hexdigest()
    assert provenance["clean_slurm_job_id"] != provenance["diagnostic_slurm_job_id"]


@pytest.mark.parametrize(
    "field",
    [
        "library_sha256",
        "frozen_density_sha256",
        "changed_coordinates_bohr",
        "changed_reference_sha256",
        "controls",
    ],
)
def test_changed_input_or_binary_cannot_supply_diagnostics(evidence, field):
    bind, path, directory, _, companion, _, save = evidence
    companion[field] = "different"
    save()
    with pytest.raises(ValueError, match=field):
        bind(path, directory)


def test_an_old_predecessor_hash_is_rejected(evidence):
    bind, path, directory, _, _, _, _ = evidence
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="clean file hash"):
        bind(path, directory)


def test_new_clean_samples_cannot_be_pooled(evidence):
    bind, path, directory, clean, companion, _, save = evidence
    companion["samples"] = [clean["samples"][0]]
    save()
    with pytest.raises(ValueError, match="contains clean repeats"):
        bind(path, directory)


def test_diagnostics_cannot_complete_an_unfinished_clean_series(evidence):
    bind, path, directory, clean, _, _, save = evidence
    clean["samples"].pop()
    save()
    with pytest.raises(ValueError, match="incomplete original clean series"):
        bind(path, directory)


@pytest.mark.parametrize("error", [1.01e-8, float("nan")])
def test_failed_or_nonfinite_diagnostic_gate_is_rejected(evidence, error):
    bind, path, directory, _, companion, _, save = evidence
    companion["diagnostics"][0]["maximum_force_error"] = error
    save()
    with pytest.raises(ValueError, match="numerical gate"):
        bind(path, directory)


def test_different_scf_branch_is_retained_and_labeled(evidence):
    bind, path, directory, _, companion, _, save = evidence
    companion["diagnostics"][0]["iterations"] = 10
    save()
    joined = bind(path, directory)
    assert joined["diagnostics"][0]["iterations"] == 10
    assert joined["diagnostic_companion"][
        "diagnostic_iterations_observed_in_clean"
    ] == {"dense": False, "packed": True}
