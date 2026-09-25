"""Exact compiled-CPU evidence for automatic Libxc point bindings."""

from __future__ import annotations

import shutil

import pytest
from vibeqc_compiler.xc.bulk_point_program import (
    SemilocalPointBinding,
    bind_runtime_semilocal_point_program,
)
from vibeqc_compiler.xc.bulk_runtime import (
    PRODUCTION_CANDIDATE_DOMAIN,
    build_bulk_runtime_program,
)
from vibeqc_compiler.xc.compiled_cpu_evidence import (
    QUALIFICATION_SCHEMA,
    RESULT_SCHEMA,
    build_result,
    stage_evidence,
    validate_result,
)

from tools.qualify_libxc_compiled_cpu import qualify_compiled_cpu

NAME = "GGA_X_PBE_SOL"


def _binding(*, candidate: bool = True) -> SemilocalPointBinding:
    program = build_bulk_runtime_program(
        NAME,
        spin="polarized",
        order=1,
        **({"domain": PRODUCTION_CANDIDATE_DOMAIN} if candidate else {}),
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
            "input_identity": "d" * 64,
            "expected": [1.0, 2.0, 3.0],
            "observed": [1.0, 2.0, 3.0],
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
        PRODUCTION_CANDIDATE_DOMAIN
    )
    assert envelope["qualification"]["binding"]["domain_version"] == 2
    assert result["identity"] in envelope["evidence"]


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


def test_passing_compiled_cpu_smoke_must_be_numerically_valid() -> None:
    outcome = _outcome()
    outcome["smoke"] = {
        **outcome["smoke"],
        "maximum_absolute_error": 2.0e-12,
        "absolute_tolerance": 1.0e-12,
    }
    with pytest.raises(ValueError, match="exceeds"):
        build_result(NAME, _binding(), outcome, evidence="test://bad-smoke")


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
    assert qualification["binding"]["domain"] == PRODUCTION_CANDIDATE_DOMAIN
    assert qualification["binding"]["domain_version"] == 2
    assert qualification["smoke"]["status"] == "pass"
    assert qualification["smoke"]["maximum_absolute_error"] <= (
        qualification["smoke"]["absolute_tolerance"]
    )
