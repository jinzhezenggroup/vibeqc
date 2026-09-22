"""CPU-only contract tests for the Slice C qualification runner."""

from __future__ import annotations

import tarfile
from typing import TYPE_CHECKING

import pytest
from vibeqc_compiler.common.evidence import canonical_hash, file_hash, validate_evidence

from benchmarks.tensor_cub_qualification import (
    _cub_candidate,
    _nvidia_smi_metadata,
    _validation_record,
    _verified_source_snapshot,
    qualification_case,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_qualification_case_fixes_equation_precision_and_layout_domain() -> None:
    program, fixtures = qualification_case()

    assert len(fixtures) == 2
    assert fixtures[0]["x"].shape == (65, 4097)
    assert fixtures[0]["x"].dtype.name == "float64"
    assert fixtures[0]["x"].flags.c_contiguous
    assert not fixtures[1]["x"].flags.c_contiguous
    assert program.outputs["result"].spec.shape == (65,)
    assert program.outputs["result"].spec.dtype == "float64"


@pytest.mark.parametrize(("rows", "inner"), [(0, 4097), (65, 31), (1.0, 4097)])
def test_qualification_case_rejects_unqualified_shapes(
    rows: object, inner: int
) -> None:
    with pytest.raises(ValueError, match="inner >= 32"):
        qualification_case(rows, inner)  # type: ignore[arg-type]


def test_cub_candidate_requires_complete_endpoint_evidence() -> None:
    incomplete = {
        "status": "rejected",
        "plan": {"schedule": {"reduction_provider": "cub"}},
        "reason": "compile failed",
    }
    with pytest.raises(RuntimeError, match="did not complete"):
        _cub_candidate({"candidates": [incomplete]})


def test_source_snapshot_binds_revision_and_execution_tree(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "tracked.txt").write_text("reviewed source")
    path = tmp_path / "source.tar.gz"
    revision = "a" * 40
    with tarfile.open(
        path, "w:gz", format=tarfile.PAX_FORMAT, pax_headers={"comment": revision}
    ) as archive:
        archive.add(root / "tracked.txt", arcname="tracked.txt")
    digest = file_hash(path)

    snapshot = _verified_source_snapshot(path, digest, revision, root)
    assert snapshot["git_revision"] == revision
    assert snapshot["verified_files"] == 1

    with pytest.raises(ValueError, match="Git revision"):
        _verified_source_snapshot(path, digest, "b" * 40, root)
    (root / "untracked.txt").write_text("not in archive")
    with pytest.raises(ValueError, match="outside"):
        _verified_source_snapshot(path, digest, revision, root)


def test_nvidia_smi_uuid_must_match_executed_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")
    completed = type(
        "Completed",
        (),
        {
            "stdout": "NVIDIA H200, GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, 570.1, 143771\n"
        },
    )()
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: completed)
    metadata = _nvidia_smi_metadata(0, {"uuid": "aaaaaaaabbbbccccddddeeeeeeeeeeee"})
    assert metadata["name"] == "NVIDIA H200"
    assert metadata["driver_version"] == "570.1"
    assert metadata["memory_total_mib"] == 143771
    with pytest.raises(ValueError, match="UUID differs"):
        _nvidia_smi_metadata(0, {"uuid": "f" * 32})


def test_validation_wrapper_is_a_publishable_nonpromotion_record() -> None:
    program, fixtures = qualification_case(1, 32)
    inputs_hash = canonical_hash({"fixture": "unit"})
    samples = [
        {
            "selection": selection,
            "seconds": 0.001,
            "inputs_hash": inputs_hash,
            "workload": "unchanged-geometry",
            "synchronized": True,
            "diagnostics": {},
        }
        for selection in (
            "baseline",
            "candidate",
            "candidate",
            "baseline",
            "baseline",
            "candidate",
            "candidate",
            "baseline",
            "baseline",
            "candidate",
        )
    ]
    artifact = {
        "identity": {
            "generated": "a" * 64,
            "toolchain": {"nvcc": "unit"},
            "host_compiler": "unit-cxx",
        },
        "compile_seconds": 0.25,
        "generated_source_bytes": 1024,
        "binary_bytes": 2048,
        "resources": [
            {
                "function": "kernel",
                "registers": 32,
                "stack_bytes": 0,
                "spill_store_bytes": 0,
                "spill_load_bytes": 0,
                "shared_bytes": 256,
                "local_bytes": None,
            }
        ],
    }
    candidate = {
        "artifact": artifact,
        "gates": [{"passed": False}],
        "max_absolute_error": 0.0,
        "plan": {"schedule": {"reduction_provider": "cub"}},
        "plan_identity": "b" * 64,
        "profiles": [
            {
                "candidate": {
                    "owned_device_bytes": 256,
                    "provider_retained_bytes": 0,
                    "predicted_peak_bytes": 512,
                }
            }
        ],
        "samples": [samples],
        "shared_gates": [{"status": "not-run"}],
        "endpoint_profitability": [],
        "status": "rejected",
    }
    tuning = {
        "baseline_artifact": {
            "compile_seconds": 0.2,
            "generated_source_bytes": 900,
            "resources": artifact["resources"],
        },
        "baseline_plan": {"target": {"architecture": "sm_90"}},
        "device": {"name": "unit-gpu", "architecture": "sm_90"},
    }
    record = _validation_record(
        program=program,
        fixtures=fixtures,
        environment={"git": {"commit": "c" * 40}},
        tuning=tuning,
        candidate=candidate,
        allocation_id="unit-allocation",
        source_snapshot={
            "archive_sha256": "d" * 64,
            "git_revision": "c" * 40,
            "verified_files": 1,
            "verified_bytes": 1,
            "execution_root": "/source",
        },
        nvidia_smi={
            "name": "NVIDIA H200",
            "uuid": "GPU-unit",
            "driver_version": "570.1",
            "memory_total_mib": 143771,
        },
    )

    validate_evidence(record)
    assert record["stages"]["production"]["status"] == "not-run"
    assert record["performance"]["status"] == "not-run"
    assert record["settings"]["source_snapshot"]["archive_sha256"] == "d" * 64


@pytest.mark.parametrize("device", [0, 1])
def test_nvidia_smi_selects_executed_uuid_not_physical_ordinal(
    monkeypatch: pytest.MonkeyPatch, device: int
) -> None:
    """CUDA-visible ordinals need not equal nvidia-smi's physical indices."""
    from types import SimpleNamespace

    expected = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    monkeypatch.setattr("shutil.which", lambda name: f"/usr/bin/{name}")

    def run(command: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        assert f"--id={expected}" in command
        return SimpleNamespace(stdout=f"NVIDIA H200, {expected}, 570.1, 143771\n")

    monkeypatch.setattr("subprocess.run", run)
    metadata = _nvidia_smi_metadata(
        device, {"uuid": "aaaaaaaabbbbccccddddeeeeeeeeeeee"}
    )
    assert metadata["uuid"] == expected
