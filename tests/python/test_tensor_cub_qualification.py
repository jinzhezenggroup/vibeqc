"""CPU-only contract tests for the Slice C qualification runner."""

from __future__ import annotations

import pytest
from vibeqc_compiler.common.evidence import canonical_hash, validate_evidence

from benchmarks.tensor_cub_qualification import (
    _cub_candidate,
    _validation_record,
    qualification_case,
)


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
    )

    validate_evidence(record)
    assert record["stages"]["production"]["status"] == "not-run"
    assert record["performance"]["status"] == "not-run"
