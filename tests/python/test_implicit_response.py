"""Reproduce #293 MP2 Z-vectors with generated RHF residual transposes.

Dense AO integrals here are existing tiny independent fixtures, not a new
production response backend. The only declared equation is the *primal*
linearized RHF stationarity action. The compiler derives its transpose/source.
"""

import typing

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
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import (
    DenseAOResponseBackend,
    GMRESOptions,
    NativeJKBackend,
    ResponseCompatibilityError,
    RHFResponseOperator,
)
from tools.vibeqc_response.implicit import (
    BoundImplicitState,
    ImplicitSolveError,
    ResponseGMRES,
    checked_transpose_solve,
)
from tools.vibeqc_response.krylov import _HostKrylovEngine


def _rhf_equation(
    reference: typing.Any, arrays: typing.Any, problem: typing.Any
) -> typing.Any:
    ao = IndexSpace("ao", "ao", reference.nmo)
    occ = IndexSpace("occupied", "occupied", reference.nocc)
    virt = IndexSpace("virtual", "virtual", reference.nmo - reference.nocc)
    p, q, r, s = (Index(name, ao) for name in "pqrs")
    i, a = Index("i", occ), Index("a", virt)

    def tensor(
        name: typing.Any, indices: typing.Any, differentiable: typing.Any = False
    ) -> typing.Any:
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
def test_generated_implicit_response_matches_existing_physical_mp2_zvector(
    name: typing.Any,
) -> None:
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


@pytest.mark.parametrize("name", ["h2", "water"])
def test_native_response_operator_is_bound_to_generated_implicit_vjp(
    name: typing.Any,
) -> None:
    metadata, arrays = load_fixture(name)
    reference = fixture_snapshot(metadata, arrays)
    try:
        source = NativeSource(**source_arguments(metadata))
    except (OSError, FileNotFoundError, AttributeError) as error:
        pytest.skip(f"native post-HF library unavailable: {error}")
    with source:
        backend = NativeJKBackend(source, axis_tile=2)
        problem = RHFResponseOperator.build_problem(reference, backend)
        operator = RHFResponseOperator(problem, backend)
        spec, feeds = _rhf_equation(reference, arrays, problem)
        plan = spec.compile()

        # Independent generated/native action agreement qualifies the identity
        # declared by this test fixture before the runtime delegates the solve.
        probe = np.random.default_rng(1465).normal(size=problem.dimension)
        generated = (
            execute(
                plan.programs["transpose"],
                {
                    **feeds,
                    "__implicit_vector": probe.reshape(spec.residual_spec.shape),
                },
            )
            .outputs["value"]
            .reshape(-1)
        )
        np.testing.assert_allclose(
            generated, operator.apply_transpose(probe), atol=2e-10, rtol=2e-10
        )

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
        hcore = (
            reference.coefficients.T @ arrays["conventional_h"] @ reference.coefficients
        )
        rhs = canonical_orbital_rhs(hcore, eri, adjoint, no)
        options = GMRESOptions(rtol=1e-11, atol=1e-13, max_iterations=80)
        expected = solve_canonical_orbital_response(
            reference, backend, rhs, options=options
        )
        solver = ResponseGMRES(problem.dimension, options)
        minimum_host = (
            plan.reference_workspace_bytes
            + solver.workspace_bytes
            + operator.host_workspace_bytes
        )
        live = {"reference": reference.identity}
        with pytest.raises(ImplicitSolveError, match="host workspace budget"):
            BoundImplicitState(
                plan,
                feeds,
                reference_identity=reference.identity,
                solver=solver,
                response_operator=operator,
                current_reference=lambda: live["reference"],
                max_bytes=minimum_host - 1,
            )

        bound = BoundImplicitState(
            plan,
            feeds,
            reference_identity=reference.identity,
            solver=solver,
            response_operator=operator,
            current_reference=lambda: live["reference"],
            max_bytes=minimum_host,
        )
        actions_before_vjp = backend.statistics["actions"]
        result = bound.vjp(-rhs.response_rhs, reference_identity=reference.identity)
        np.testing.assert_allclose(
            result.adjoint.reshape(-1),
            expected.solution,
            atol=3e-11,
            rtol=3e-10,
        )
        assert result.logical_reserved_host_bytes == minimum_host
        assert result.logical_reserved_device_bytes == 0
        assert "NativeJKBackend" in result.transpose_backend
        assert backend.statistics["actions"] > actions_before_vjp

        live["reference"] = "stale-reference"
        with pytest.raises(ResponseCompatibilityError, match="reference"):
            bound.vjp(-rhs.response_rhs, reference_identity=reference.identity)


def test_checked_transpose_solve_preserves_resident_engine_and_final_check() -> None:
    matrix = np.diag(np.array([1.5, 2.0, 3.0], dtype=np.float64))
    counters = {"resident": 0, "host": 0, "checks": 0}

    class ResidentEngine:
        resident = True
        vector_slots = 128

        def __init__(self) -> None:
            self.dimension = 3
            self._host = _HostKrylovEngine(self.dimension)

        def __getattr__(self, name: str) -> typing.Any:
            return getattr(self._host, name)

        def apply(self, operator: typing.Any, value: typing.Any) -> np.ndarray:
            del operator
            counters["resident"] += 1
            return matrix @ np.asarray(value, dtype=np.float64)

    class Operator:
        dimension = 3

        def __init__(self) -> None:
            self._krylov_engine = ResidentEngine()

        def apply(self, vector: typing.Any) -> np.ndarray:
            counters["host"] += 1
            return matrix @ np.asarray(vector, dtype=np.float64)

    def current() -> None:
        counters["checks"] += 1

    rhs = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    result = checked_transpose_solve(
        Operator(),
        rhs,
        solver=ResponseGMRES(
            3,
            GMRESOptions(
                rtol=0.0,
                atol=1e-12,
                restart=3,
                max_iterations=8,
            ),
        ),
        assert_current=current,
    )

    np.testing.assert_allclose(
        result.solution, np.linalg.solve(matrix, rhs), atol=1e-12
    )
    assert counters["resident"] > 0
    assert counters["host"] == 1
    assert result.operator_actions == counters["resident"] + counters["host"]
    assert counters["checks"] >= 2 * counters["resident"] + 2


def test_response_operator_binding_rejects_wrong_operator_identity() -> None:
    metadata, arrays = load_fixture("h2")
    reference = fixture_snapshot(metadata, arrays)
    dense = DenseAOResponseBackend(arrays["ao"])
    dense_problem = RHFResponseOperator.build_problem(reference, dense)
    spec, feeds = _rhf_equation(reference, arrays, dense_problem)
    try:
        source = NativeSource(**source_arguments(metadata))
    except (OSError, FileNotFoundError, AttributeError) as error:
        pytest.skip(f"native post-HF library unavailable: {error}")
    with source:
        native = NativeJKBackend(source)
        native_problem = RHFResponseOperator.build_problem(reference, native)
        operator = RHFResponseOperator(native_problem, native)
        with pytest.raises(ResponseCompatibilityError, match="operator identity"):
            BoundImplicitState(
                spec.compile(),
                feeds,
                reference_identity=reference.identity,
                response_operator=operator,
                current_reference=lambda: reference.identity,
            )


def test_response_operator_device_resources_require_joint_budget() -> None:
    metadata, arrays = load_fixture("h2")
    reference = fixture_snapshot(metadata, arrays)
    backend = DenseAOResponseBackend(arrays["ao"])
    backend.device_workspace_bytes = 4096
    problem = RHFResponseOperator.build_problem(reference, backend)
    operator = RHFResponseOperator(problem, backend)
    spec, feeds = _rhf_equation(reference, arrays, problem)
    plan = spec.compile()
    live = lambda: reference.identity

    with pytest.raises(ValueError, match="current_reference"):
        BoundImplicitState(
            plan,
            feeds,
            reference_identity=reference.identity,
            response_operator=operator,
        )
    with pytest.raises(ImplicitSolveError, match="explicit combined device budget"):
        BoundImplicitState(
            plan,
            feeds,
            reference_identity=reference.identity,
            response_operator=operator,
            current_reference=live,
        )
    with pytest.raises(ImplicitSolveError, match="device workspace budget"):
        BoundImplicitState(
            plan,
            feeds,
            reference_identity=reference.identity,
            response_operator=operator,
            current_reference=live,
            max_device_bytes=4095,
        )

    bound = BoundImplicitState(
        plan,
        feeds,
        reference_identity=reference.identity,
        response_operator=operator,
        current_reference=live,
        max_device_bytes=4096,
    )
    assert bound.logical_reserved_device_bytes == 4096
    result = bound.vjp(
        np.zeros(spec.state_spec.shape), reference_identity=reference.identity
    )
    assert result.logical_reserved_device_bytes == 4096

    backend.device_workspace_bytes = 8192
    with pytest.raises(ResponseCompatibilityError, match="resource contract"):
        bound.vjp(
            np.zeros(spec.state_spec.shape), reference_identity=reference.identity
        )


def test_response_binding_rejects_cpks_even_with_declared_resources() -> None:
    from dataclasses import replace
    from types import SimpleNamespace

    from tools.vibeqc_response.implicit import ResponseTransposeBinding
    from tools.vibeqc_response.problem import ResponseProblem

    metadata, arrays = load_fixture("h2")
    reference = replace(
        fixture_snapshot(metadata, arrays),
        algorithm="KS",
        functional_identity="test-xc",
        grid_identity="test-grid",
        hf_backend="test-ks",
    )
    problem = ResponseProblem.from_reference(
        reference, method="cpks", operator_identity="cpks-test-operator"
    )
    spec, _ = _rhf_equation(reference, arrays, problem)
    operator = SimpleNamespace(
        problem=problem,
        dimension=problem.dimension,
        identity=problem.operator_identity,
        backend=SimpleNamespace(identity="declared-backend"),
        host_workspace_bytes=100,
        device_workspace_bytes=100,
        resource_identity="declared",
    )
    with pytest.raises(ResponseCompatibilityError, match="RHF only"):
        ResponseTransposeBinding(spec.compile(), operator)
