"""Method-neutral HF/KS/RCCSD ElectronicMethodIR composition gates."""

from __future__ import annotations

import json

import pytest
from vibeqc_compiler.method import (
    ElectronicMethodIR,
    EnergySpec,
    IterationSpec,
    OperatorSpec,
    StateSpec,
    gfn_electronic_method_ir,
    rccsd_electronic_method_ir,
    resolve_method,
    resolve_xtb_method,
    rhf_electronic_method_ir,
    rks_electronic_method_ir,
)

from tools.vibeqc_cc.doubles import build_ccsd_program


def _before(order: tuple[str, ...], first: str, second: str) -> bool:
    return order.index(first) < order.index(second)


def test_rhf_uses_common_state_operator_and_iteration_schema() -> None:
    graph = rhf_electronic_method_ir()
    assert graph.family == "hf"
    assert graph.reference == "restricted"
    assert graph.energy == EnergySpec("total_energy")
    assert graph.iteration == IterationSpec(
        ("density",),
        (("density", "density_error"),),
        (("density", "density_next"),),
        "scf-fixed-point-v1",
    )
    assert _before(graph.operator_order, "build_fock", "diagonalize_fock")
    assert _before(graph.operator_order, "diagonalize_fock", "build_density")
    assert _before(graph.operator_order, "build_density", "density_residual")
    json.dumps(graph.to_payload(), sort_keys=True, allow_nan=False)


def test_rks_reuses_existing_dft_method_identity_without_copying_xc_graph() -> None:
    pbe = resolve_method("PBE")
    pbe0 = resolve_method("PBE0")
    semilocal = rks_electronic_method_ir(pbe)
    hybrid = rks_electronic_method_ir(pbe0)

    assert semilocal.composition_identity == pbe.identity
    assert hybrid.composition_identity == pbe0.identity
    assert {operator.kind for operator in semilocal.operators} >= {
        "coulomb",
        "semilocal_xc",
        "fock",
    }
    assert "exact_exchange" not in {operator.kind for operator in semilocal.operators}
    exchange = [
        operator for operator in hybrid.operators if operator.kind == "exact_exchange"
    ]
    assert len(exchange) == 1
    assert exchange[0].ir_identity == pbe0.identity


def test_dft_aliases_share_structural_identity_but_keep_manifest_name() -> None:
    canonical = rks_electronic_method_ir("PBE0")
    alias = rks_electronic_method_ir("PBEH")
    assert canonical.identity == alias.identity
    assert canonical.manifest_identity != alias.manifest_identity


@pytest.mark.parametrize(
    ("name", "expected_states"),
    (
        ("GFN1-xTB", {"shell_charge", "charge", "magnetization"}),
        ("GFN2-xTB", {"charge", "dipole", "quadrupole", "magnetization"}),
    ),
)
def test_gfn_reuses_xtb_method_identity_without_copying_scientific_graph(
    name: str, expected_states: set[str]
) -> None:
    method = resolve_xtb_method(name)
    graph = gfn_electronic_method_ir(method)

    assert graph.family == "semiempirical"
    assert graph.reference == method.reference
    assert graph.composition_identity == method.identity
    assert {state.name for state in graph.states} == expected_states
    assert graph.iteration is not None
    assert set(graph.iteration.states) == expected_states
    assert graph.iteration.solver_contract == "gfn-scc-fixed-point-v1"
    assert dict(graph.iteration.residuals) == {
        state: f"{state}_residual" for state in expected_states
    }
    assert dict(graph.iteration.updates) == {
        state: f"{state}_next" for state in expected_states
    }
    scientific = {
        operator.ir_identity
        for operator in graph.operators
        if operator.ir_identity is not None
    }
    assert scientific == {method.identity}
    assert method.capability["runtime_executable"] is False
    json.dumps(graph.to_payload(), sort_keys=True, allow_nan=False)


def test_gfn_orchestration_keeps_eigensolver_and_runtime_policy_outside_xtb_ir() -> None:
    method = resolve_xtb_method("GFN2-xTB")
    graph = gfn_electronic_method_ir(method)

    assert _before(
        graph.operator_order, "assemble_hamiltonian", "diagonalize_hamiltonian"
    )
    assert _before(
        graph.operator_order, "diagonalize_hamiltonian", "evaluate_occupations"
    )
    assert _before(graph.operator_order, "evaluate_occupations", "project_scc_state")
    assert "mixing_policy" in graph.sources
    assert "occupation_policy" in graph.sources
    assert "mixing_policy" not in method.semantic_payload()
    assert "occupation_policy" not in method.semantic_payload()


def test_rccsd_wraps_existing_tensorir_by_identity() -> None:
    program = build_ccsd_program(2, 2, form="optimized", diagnostics=False)
    graph = rccsd_electronic_method_ir(program)
    assert graph.family == "cc"
    assert graph.energy == EnergySpec("correlation_energy", kind="correlation")
    assert {state.name for state in graph.states} == {"t1", "t2"}
    equations = next(
        operator for operator in graph.operators if operator.name == "rccsd_equations"
    )
    assert equations.ir_identity == program.logical_hash
    assert graph.composition_identity == program.logical_hash
    assert graph.iteration is not None
    assert dict(graph.iteration.residuals) == {
        "t1": "singles_residual",
        "t2": "doubles_residual",
    }


def test_constructor_canonicalizes_order_for_stable_structural_hash() -> None:
    graph = rhf_electronic_method_ir()
    reordered = ElectronicMethodIR(
        identifier="RHF-alias",
        family=graph.family,
        reference=graph.reference,
        sources=tuple(reversed(graph.sources)),
        states=tuple(reversed(graph.states)),
        operators=tuple(reversed(graph.operators)),
        energy=graph.energy,
        iteration=graph.iteration,
    )
    assert reordered.identity == graph.identity
    assert reordered.manifest_identity != graph.manifest_identity


def test_validator_rejects_dangling_operator_input() -> None:
    with pytest.raises(ValueError, match="dangling inputs"):
        ElectronicMethodIR(
            identifier="bad",
            family="hf",
            reference="restricted",
            sources=(),
            states=(StateSpec("density", "density", "ao-density-v1"),),
            operators=(
                OperatorSpec("energy", "energy", ("missing",), ("total_energy",)),
            ),
            energy=EnergySpec("total_energy"),
        )


def test_validator_rejects_operator_cycles() -> None:
    with pytest.raises(ValueError, match="dependency cycle"):
        ElectronicMethodIR(
            identifier="cycle",
            family="hf",
            reference="restricted",
            sources=(),
            states=(StateSpec("density", "density", "ao-density-v1"),),
            operators=(
                OperatorSpec("left", "transform", ("right_value",), ("left_value",)),
                OperatorSpec("right", "transform", ("left_value",), ("right_value",)),
            ),
            energy=EnergySpec("left_value"),
        )


def test_validator_rejects_iteration_state_not_in_method_state() -> None:
    with pytest.raises(ValueError, match="iteration references undeclared state"):
        ElectronicMethodIR(
            identifier="bad-iteration",
            family="hf",
            reference="restricted",
            sources=(),
            states=(StateSpec("density", "density", "ao-density-v1"),),
            operators=(
                OperatorSpec("residual", "residual", ("density",), ("error", "next")),
                OperatorSpec("energy", "energy", ("density",), ("total_energy",)),
            ),
            energy=EnergySpec("total_energy"),
            iteration=IterationSpec(
                ("ghost",),
                (("ghost", "error"),),
                (("ghost", "next"),),
                "fixed-point-v1",
            ),
        )
