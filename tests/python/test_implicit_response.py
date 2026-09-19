"""Reproduce #293 MP2 Z-vectors with generated RHF residual transposes.

Dense AO integrals here are existing tiny independent fixtures, not a new
production response backend. The only declared equation is the *primal*
linearized RHF stationarity action. The compiler derives its transpose/source.
"""

import numpy as np
import pytest
from vibeqc_compiler.method import ImplicitSolveSpec
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    TensorSpec,
    add,
    einsum,
    execute,
    input_tensor,
    multiply,
    transpose,
)

from tools.vibeqc_mp2.gradient import (
    canonical_energy_adjoint,
    canonical_orbital_rhs,
    solve_canonical_orbital_response,
)
from tools.vibeqc_posthf.fixtures import fixture_snapshot, load_fixture
from tools.vibeqc_response import (
    DenseAOResponseBackend,
    GMRESOptions,
    RHFResponseOperator,
)
from tools.vibeqc_response.implicit import BoundImplicitState, ResponseGMRES


def _rhf_equation(reference, arrays, problem):
    ao = IndexSpace("ao", "ao", reference.nmo)
    occ = IndexSpace("occupied", "occupied", reference.nocc)
    virt = IndexSpace("virtual", "virtual", reference.nmo - reference.nocc)
    p, q, r, s = (Index(name, ao) for name in "pqrs")
    i, a = Index("i", occ), Index("a", virt)

    def tensor(name, indices, differentiable=False):
        return input_tensor(
            name,
            TensorSpec(tuple(indices), role="parameter", differentiable=differentiable),
        )

    x, perturbation = tensor("x", (i, a), True), tensor("q", (i, a), True)
    co, cv = tensor("co", (p, i)), tensor("cv", (p, a))
    eri, gap = tensor("eri", (p, q, r, s)), tensor("gap", (i, a))
    occupied_virtual = einsum("pi,ia->pa", co, x)
    half_density = einsum("pa,qa->pq", occupied_virtual, cv)
    density = add(half_density, transpose(half_density, (1, 0)), coefficients=(2, 2))
    coulomb = einsum("pqrs,rs->pq", eri, density)
    exchange = einsum("prqs,rs->pq", eri, density)
    fock = add(coulomb, exchange, coefficients=(1, "-1/2"))
    projected = einsum("pi,pq->iq", co, fock)
    action = add(einsum("iq,qa->ia", projected, cv), multiply(gap, x))
    equation = Program({"residual": add(action, perturbation)})
    spec = ImplicitSolveSpec(
        equation,
        "x",
        ("q",),
        problem.operator_identity,
        state_layout=problem.layout.identity,
        residual_layout=problem.layout.identity,
        gauge=problem.gauge,
    )
    no = reference.nocc
    feeds = {
        "x": np.zeros((no, reference.nmo - no)),
        "q": np.zeros((no, reference.nmo - no)),
        "co": reference.coefficients[:, :no],
        "cv": reference.coefficients[:, no:],
        "eri": arrays["ao"],
        "gap": reference.orbital_energies[no:][None, :]
        - reference.orbital_energies[:no, None],
    }
    return spec, feeds


@pytest.mark.parametrize("name", ["h2", "water"])
def test_generated_implicit_response_matches_existing_physical_mp2_zvector(name):
    metadata, arrays = load_fixture(name)
    reference = fixture_snapshot(metadata, arrays)
    backend = DenseAOResponseBackend(arrays["ao"])
    problem = RHFResponseOperator.build_problem(reference, backend)
    operator = RHFResponseOperator(problem, backend)
    spec, feeds = _rhf_equation(reference, arrays, problem)
    plan = spec.compile()
    vector = np.random.default_rng(465).normal(size=problem.dimension)
    for stage, apply in (
        ("jacobian", operator.apply),
        ("transpose", operator.apply_transpose),
    ):
        generated = (
            execute(
                plan.programs[stage],
                {
                    **feeds,
                    "__implicit_vector": vector.reshape(spec.state_spec.shape),
                },
            )
            .outputs["value"]
            .reshape(-1)
        )
        np.testing.assert_allclose(generated, apply(vector), atol=2e-12, rtol=2e-12)
    no = reference.nocc
    eri = arrays["conventional_mo"]
    g = eri[:no, no:, :no, no:].transpose(0, 2, 1, 3)
    adjoint = canonical_energy_adjoint(
        g,
        reference.orbital_energies,
        no,
        reference_identity=reference.identity,
        hamiltonian_id=reference.hamiltonian_id,
    )
    hcore = reference.coefficients.T @ arrays["conventional_h"] @ reference.coefficients
    rhs = canonical_orbital_rhs(hcore, eri, adjoint, no)
    options = GMRESOptions(rtol=1e-11, atol=1e-13, max_iterations=80)
    expected = solve_canonical_orbital_response(
        reference, backend, rhs, options=options
    )
    bound = BoundImplicitState(
        plan,
        feeds,
        reference_identity=reference.identity,
        solver=ResponseGMRES(problem.dimension, options),
    )
    # #293 eliminates occupied/occupied and virtual/virtual canonical response
    # before constructing this RHS. Its raw energy_gradient predates those
    # contributions and is intentionally NOT the reduced state cotangent.
    result = bound.vjp(-rhs.response_rhs, reference_identity=reference.identity)
    np.testing.assert_allclose(
        result.adjoint.reshape(-1), expected.solution, atol=3e-11, rtol=3e-10
    )
    np.testing.assert_allclose(
        result.parameter_cotangents["q"], result.adjoint, atol=1e-13
    )
    assert result.adjoint_residual_norm < 1e-12
    assert result.plan_identity == plan.identity
    if name == "water":
        # Water tests a nontrivial orbital response, not only symmetric H2's
        # near-zero MP2 orbital RHS.
        assert np.linalg.norm(result.adjoint) > 1e-5
