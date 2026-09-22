"""Scientific admission tests for selective XC arithmetic precision."""

from __future__ import annotations

import pytest
from vibeqc_compiler.xc.precision import (
    STRICT_MATH_MODE,
    admit_selective_precision,
)
from vibeqc_compiler.xc.program import build_program
from vibeqc_compiler.xc.spec import functional

SENSITIVE = {
    "reciprocal",
    "power",
    "exp",
    "expm1",
    "log",
    "log1p",
    "atan",
    "asinh",
    "erf",
    "select_le",
}


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
@pytest.mark.parametrize(("observable", "order"), [("energy", 0), ("potential", 1)])
def test_pbe_selective_precision_is_conservative(
    spin: str, observable: str, order: int
) -> None:
    program = build_program(functional("PBE", spin=spin), order=order)
    admission = admit_selective_precision(program, observable=observable)

    assert admission.enabled
    assert admission.functional == "PBE"
    assert admission.storage_dtype == "float64"
    assert admission.reduction_dtype == "float64"
    assert admission.strict_audit_dtype == "float64"
    assert admission.math_mode == STRICT_MATH_MODE
    assert admission.reasons == ()

    entries = {value.identifier: value for value in admission.values}
    candidates = [entries[identifier] for identifier in admission.candidate_nodes]
    assert candidates
    assert {value.operation for value in candidates} <= {"add", "multiply"}

    graph = program.graph
    for identifier, value in entries.items():
        if graph.nodes[identifier].operation in SENSITIVE:
            assert value.compute_dtype == "float64"
            assert value.reason == "sensitive-operation"

    for root in program.roots:
        if graph.nodes[root.identifier].operation not in {"constant", "variable"}:
            assert entries[root.identifier].compute_dtype == "float64"


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_sensitive_dependency_cones_stay_fp64(spin: str) -> None:
    program = build_program(functional("PBE", spin=spin), order=1)
    admission = admit_selective_precision(program, observable="potential")
    entries = {value.identifier: value for value in admission.values}

    graph = program.graph
    sensitive = [
        identifier
        for identifier in entries
        if graph.nodes[identifier].operation in SENSITIVE
    ]
    assert sensitive

    strict_closure = set(sensitive)
    stack = list(sensitive)
    while stack:
        identifier = stack.pop()
        for argument in graph.nodes[identifier].arguments:
            if argument not in strict_closure:
                strict_closure.add(argument)
                stack.append(argument)

    for identifier in strict_closure:
        if identifier in entries:
            assert entries[identifier].compute_dtype == "float64"


@pytest.mark.parametrize("spin", ["unpolarized", "polarized"])
def test_r2scan_is_fail_closed(spin: str) -> None:
    program = build_program(functional("R2SCAN", spin=spin), order=1)
    admission = admit_selective_precision(program, observable="potential")

    assert not admission.enabled
    assert not admission.candidate_nodes
    assert admission.strict_nodes
    assert admission.reasons == ("R2SCAN has no selective-precision qualification",)
    assert {value.compute_dtype for value in admission.values} == {"float64"}


@pytest.mark.parametrize("observable", ["response", "geometry"])
def test_unqualified_pbe_observables_are_fail_closed(observable: str) -> None:
    program = build_program(functional("PBE", spin="polarized"), order=1)
    admission = admit_selective_precision(program, observable=observable)

    assert not admission.enabled
    assert admission.reasons == (
        "selective precision is qualified only for PBE energy/potential DAGs",
    )


def test_second_derivative_pbe_is_fail_closed() -> None:
    program = build_program(functional("PBE", spin="unpolarized"), order=2)
    admission = admit_selective_precision(program, observable="potential")

    assert not admission.enabled
    assert admission.reasons == (
        "selective precision is qualified only for PBE energy/potential DAGs",
    )


def test_admission_identity_is_deterministic_and_expression_bound() -> None:
    pbe = build_program(functional("PBE", spin="polarized"), order=1)
    first = admit_selective_precision(pbe, observable="potential")
    second = admit_selective_precision(pbe, observable="potential")
    energy = admit_selective_precision(
        build_program(functional("PBE", spin="polarized"), order=0),
        observable="energy",
    )

    assert first == second
    assert first.identity == second.identity
    assert len(first.identity) == 64
    assert first.expression_hash == pbe.expression_hash
    assert first.identity != energy.identity
