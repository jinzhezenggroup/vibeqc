"""Exact compiled-CPU evidence for automatic Libxc point bindings."""

from __future__ import annotations

import shutil
from copy import deepcopy

import pytest
from vibeqc_compiler.common.evidence import canonical_hash
from vibeqc_compiler.xc.bulk_point_program import (
    SemilocalPointBinding,
    bind_runtime_semilocal_point_program,
)
from vibeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_DENSITY_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
)
from vibeqc_compiler.xc.compiled_cpu_evidence import (
    QUALIFICATION_SCHEMA,
    RESULT_SCHEMA,
    build_result,
    stage_evidence,
    validate_qualification,
    validate_result,
)

from tools.qualify_libxc_compiled_cpu import qualify_compiled_cpu

NAME = "GGA_X_PBE_SOL"


def _binding(*, candidate: bool = True) -> SemilocalPointBinding:
    program = build_bulk_runtime_program(
        NAME,
        spin="polarized",
        order=1,
        **({"domain": PRODUCTION_DENSITY_CANDIDATE_DOMAIN} if candidate else {}),
    )
    return bind_runtime_semilocal_point_program(program)


def _outcome() -> dict:
    return {
        "status": "pass",
        "reason": None,
        "compiler": {
            "executable_sha256": "a" * 64,
            "version": "test-cxx 1.0",
        },
        "translation_unit_sha256": "b" * 64,
        "executable_sha256": "c" * 64,
        "smoke": {
            "status": "pass",
            "case_labels": ["interior", "vacuum"],
            "input_identity": "d" * 64,
            "expected": [float(index) for index in range(22)],
            "observed": [float(index) for index in range(22)],
            "absolute_tolerance": 1.0e-12,
            "maximum_absolute_error": 0.0,
        },
    }


def test_compiled_cpu_result_promotes_exact_binding_evidence() -> None:
    binding = _binding()
    result = build_result(
        NAME,
        binding,
        _outcome(),
        evidence="artifact://compiled-cpu/GGA_X_PBE_SOL.json",
    )
    envelope = stage_evidence(NAME, result)

    assert result["schema"] == RESULT_SCHEMA
    assert validate_result(NAME, result) == result
    assert envelope["status"] == "pass"
    assert envelope["stage"] == "compiled-cpu"
    assert envelope["qualification"]["schema"] == QUALIFICATION_SCHEMA
    assert envelope["qualification"]["binding_identity"] == binding.identity
    assert envelope["qualification"]["binding"]["domain"] == (
        PRODUCTION_DENSITY_CANDIDATE_DOMAIN
    )
    assert envelope["qualification"]["binding"]["domain_version"] == 3
    assert envelope["qualification"]["binding"]["density_threshold"] is not None
    assert result["identity"] in envelope["evidence"]


def test_compiled_cpu_qualification_is_exact_and_canonical() -> None:
    result = build_result(
        NAME,
        _binding(),
        _outcome(),
        evidence="test://qualification",
    )
    qualification = stage_evidence(NAME, result)["qualification"]

    assert validate_qualification(NAME, qualification) == qualification

    tampered = {
        **qualification,
        "binding": {
            **qualification["binding"],
            "density_threshold": qualification["binding"]["density_threshold"] * 2.0,
        },
    }
    with pytest.raises(ValueError, match="binding identity mismatch"):
        validate_qualification(NAME, tampered)


def test_compiled_cpu_result_rejects_wrong_domain_and_tampering() -> None:
    with pytest.raises(ValueError, match="production candidate domain"):
        build_result(
            NAME,
            _binding(candidate=False),
            _outcome(),
            evidence="test://wrong-domain",
        )

    result = build_result(
        NAME,
        _binding(),
        _outcome(),
        evidence="test://valid",
    )
    tampered = {
        **result,
        "executable_sha256": "e" * 64,
    }
    with pytest.raises(ValueError, match="result identity mismatch"):
        validate_result(NAME, tampered)


def test_rehashed_binding_cannot_change_pinned_density_threshold() -> None:
    result = build_result(
        NAME, _binding(), _outcome(), evidence="test://pinned-threshold"
    )
    forged = deepcopy(result)
    forged["binding"]["density_threshold"] *= 2.0
    forged["binding_identity"] = canonical_hash(forged["binding"])
    payload = {key: value for key, value in forged.items() if key != "identity"}
    forged["identity"] = canonical_hash(payload)

    with pytest.raises(ValueError, match="pinned density threshold mismatch"):
        validate_result(NAME, forged)


def test_passing_compiled_cpu_smoke_must_be_numerically_valid() -> None:
    outcome = _outcome()
    outcome["smoke"] = {
        **outcome["smoke"],
        "maximum_absolute_error": 2.0e-12,
        "absolute_tolerance": 1.0e-12,
    }
    with pytest.raises(ValueError, match="exceeds"):
        build_result(NAME, _binding(), outcome, evidence="test://bad-smoke")


def test_passing_smoke_recomputes_error_from_observed_values() -> None:
    """A producer's zero error cannot override an incorrect native output."""
    outcome = _outcome()
    outcome["smoke"]["observed"][0] += 1.0
    with pytest.raises(ValueError, match="inconsistent"):
        build_result(NAME, _binding(), outcome, evidence="test://false-pass")


@pytest.mark.parametrize("field", ("absolute_tolerance", "maximum_absolute_error"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), True))
def test_smoke_rejects_invalid_numeric_gates(field: str, value: float) -> None:
    outcome = _outcome()
    outcome["smoke"][field] = value
    with pytest.raises(ValueError, match="finite"):
        build_result(NAME, _binding(), outcome, evidence="test://invalid-gate")


def test_failed_compilation_remains_nonpromoting_evidence() -> None:
    outcome = {
        "status": "fail",
        "reason": "compiler failed",
        "compiler": {
            "executable_sha256": "a" * 64,
            "version": "test-cxx 1.0",
        },
        "translation_unit_sha256": "b" * 64,
        "executable_sha256": None,
        "smoke": None,
    }
    result = build_result(NAME, _binding(), outcome, evidence="test://compile-fail")
    envelope = stage_evidence(NAME, result)

    assert envelope["status"] == "fail"
    assert envelope["reason"] == "compiler failed"
    assert envelope["qualification"]["executable_sha256"] is None


def test_real_noncurated_gga_binding_compiles_and_executes() -> None:
    if not (shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")):
        pytest.skip("C++ compiler unavailable")

    payload = qualify_compiled_cpu(
        NAME,
        evidence="test://compiled-cpu-real",
        timeout=60.0,
    )

    assert payload["stage_evidence"]["status"] == "pass"
    qualification = payload["stage_evidence"]["qualification"]
    assert qualification["binding"]["domain"] == PRODUCTION_DENSITY_CANDIDATE_DOMAIN
    assert qualification["binding"]["domain_version"] == 3
    assert qualification["binding"]["density_threshold"] is not None
    assert qualification["smoke"]["status"] == "pass"
    assert qualification["smoke"]["case_labels"] == ["interior", "vacuum"]
    assert (
        qualification["smoke"]["maximum_absolute_error"]
        <= (qualification["smoke"]["absolute_tolerance"])
    )
