"""Fast independent mathematics and failure gates for the #465 primitive."""

import copy
import json
import typing
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.method import ImplicitSolveSpec, ImplicitVJPPlan
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    PackedLayout,
    Program,
    Symmetry,
    TensorSpec,
    add,
    einsum,
    execute,
    input_tensor,
    multiply,
)
from vibeqc_compiler.tensor.cuda_plan import TensorSchedule, plan_cuda

from tools.vibeqc_response import GMRESOptions, ResponseCompatibilityError
from tools.vibeqc_response.implicit import (
    BoundImplicitState,
    ImplicitSolveError,
    ReferenceTensorExecutor,
    ResponseGMRES,
)


def _parameter(
    name: typing.Any, indices: typing.Any = (), **kwargs: typing.Any
) -> typing.Any:
    return input_tensor(
        name,
        TensorSpec(tuple(indices), role="parameter", differentiable=True, **kwargs),
    )


def _scalar() -> typing.Any:
    x, q = _parameter("x"), _parameter("q")
    residual = Program({"residual": add(multiply(x, x), q, coefficients=(1, -1))})
    return ImplicitSolveSpec(residual, "x", ("q",), "positive-square-root"), {
        "x": np.array(2.0),
        "q": np.array(4.0),
    }


def _nonsymmetric(*, weights: typing.Any = False) -> typing.Any:
    space = IndexSpace("coordinates", "batch", 3)
    i, j = Index("i", space), Index("j", space)
    x, q, a = _parameter("x", (i,)), _parameter("q", (i,)), _parameter("a", (i, j))
    residual = add(
        einsum("ij,j->i", a, x), multiply(x, x), q, coefficients=(1, "1/5", -1)
    )
    spec = ImplicitSolveSpec(
        Program({"residual": residual}),
        "x",
        ("q", "a"),
        "nonlinear-nonsymmetric",
        state_metric=(1.0, 2.0, 1.0) if weights else (),
        residual_metric=(5.0, 2.0, 9.0) if weights else (),
    )
    matrix = np.array([[3.0, 0.8, -0.2], [-0.1, 2.0, 0.4], [0.6, 0.0, 4.0]])
    root = np.array([0.2, -0.3, 0.4])
    return spec, {"a": matrix, "x": root, "q": matrix @ root + root**2 / 5}


def _bound(spec: typing.Any, feeds: typing.Any, **kwargs: typing.Any) -> typing.Any:
    return BoundImplicitState(
        spec.compile(), feeds, reference_identity="reference-A", **kwargs
    )


def _vjp(bound: typing.Any, seed: typing.Any, **kwargs: typing.Any) -> typing.Any:
    return bound.vjp(seed, reference_identity="reference-A", **kwargs)


def _options(**kwargs: typing.Any) -> typing.Any:
    return GMRESOptions(rtol=1e-12, atol=0.0, max_iterations=40, **kwargs)


@pytest.mark.parametrize("seed", [1.0, -2.5, 0.0, 1e-100, 1e100])
def test_scalar_analytic_vjp_includes_direct_parameter_dependence(
    seed: typing.Any,
) -> None:
    spec, feeds = _scalar()
    result = _vjp(
        _bound(spec, feeds), np.array(seed), direct={"q": np.array(0.1 * seed)}
    )
    assert result.parameter_cotangents["q"] == pytest.approx(
        0.35 * seed, rel=1e-12, abs=0.0
    )
    assert result.adjoint == pytest.approx(-0.25 * seed, rel=1e-12, abs=0.0)
    assert result.adjoint_residual_norm <= 1e-10 * abs(seed)
    assert result.tensor_backend == "numpy-cpu-interpreter"
    assert result.solver_backend == "python-response-gmres"


def test_scalar_derivative_converges_against_resolved_finite_differences() -> None:
    spec, feeds = _scalar()
    expected = _vjp(_bound(spec, feeds), np.array(1.0)).parameter_cotangents["q"]
    errors = []
    for step in (1e-2, 3e-3, 1e-3):
        finite = (np.sqrt(4 + step) - np.sqrt(4 - step)) / (2 * step)
        errors.append(abs(expected - finite))
    assert errors[-1] < 3e-9
    assert errors[-1] < errors[0] / 50


@pytest.mark.parametrize("weighted", [False, True])
def test_nonsymmetric_nonlinear_vjp_dot_and_resolved_directional_differences(
    weighted: typing.Any,
) -> None:
    spec, feeds = _nonsymmetric(weights=weighted)
    plan = spec.compile()
    bound = _bound(spec, feeds, solver=ResponseGMRES(3, _options()))
    seed = np.array([0.7, -0.4, 0.2])
    wx = np.array(spec.state_metric or (1.0,) * 3)
    jacobian = feeds["a"] + np.diag(0.4 * feeds["x"])
    euclidean_multiplier = np.linalg.solve(jacobian.T, -wx * seed)
    direct = {"q": 0.6 * feeds["q"], "a": 0.4 * feeds["a"]}
    result = _vjp(bound, seed, direct=direct)
    expected_q = direct["q"] - euclidean_multiplier
    expected_a = direct["a"] + np.outer(euclidean_multiplier, feeds["x"])
    np.testing.assert_allclose(
        result.parameter_cotangents["q"], expected_q, atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        result.parameter_cotangents["a"], expected_a, atol=2e-12, rtol=2e-12
    )
    np.testing.assert_allclose(
        result.adjoint,
        euclidean_multiplier / np.array(spec.residual_metric or (1.0,) * 3),
        atol=2e-12,
    )
    # A nonsymmetric toy distinguishes an actual transpose from reusing J.
    assert (
        np.linalg.norm(np.linalg.solve(jacobian, -wx * seed) - euclidean_multiplier)
        > 0.01
    )
    left, right = np.array([0.3, -0.5, 0.7]), np.array([-0.2, 0.6, 0.4])
    jv = execute(
        plan.programs["jacobian"], {**feeds, "__implicit_vector": right}
    ).outputs["value"]
    jtl = execute(
        plan.programs["transpose"], {**feeds, "__implicit_vector": left}
    ).outputs["value"]
    np.testing.assert_allclose(left @ jv, jtl @ right, atol=1e-12, rtol=1e-12)
    da = np.array([[0.1, -0.2, 0.3], [0.0, 0.3, 0.1], [-0.1, 0.1, 0.2]])
    dq = np.array([-0.2, 0.1, 0.3])
    expected = np.sum(expected_a * da) + expected_q @ dq

    def objective(t: typing.Any) -> typing.Any:
        a, q = feeds["a"] + t * da, feeds["q"] + t * dq
        x = feeds["x"].copy()
        # Independent tiny Newton oracle only; never used by the primitive.
        for _ in range(6):
            x -= np.linalg.solve(a + np.diag(0.4 * x), a @ x + x**2 / 5 - q)
        assert np.linalg.norm(a @ x + x**2 / 5 - q) < 1e-13
        return (wx * seed) @ x + 0.3 * (q @ q) + 0.2 * np.sum(a * a)

    errors = []
    for step in (1e-2, 1e-3, 3e-4):
        errors.append(abs((objective(step) - objective(-step)) / (2 * step) - expected))
    assert errors[-1] < 1e-9
    assert errors[-1] < errors[0] / 30


def test_existing_packed_orbit_metric_is_not_an_unweighted_transpose() -> None:
    axis = Index("i", IndexSpace("symmetric", "batch", 2))
    layout = PackedLayout(
        TensorSpec((axis, Index("j", axis.space)), symmetries=(Symmetry((1, 0)),))
    )
    assert layout.weights == (1, 2, 1)
    spec, feeds = _nonsymmetric(weights=True)
    spec = replace(
        spec,
        state_layout=canonical_hash(layout.to_payload()),
        state_metric=tuple(map(float, layout.weights)),
    )
    seed = np.array([0.1, 0.7, -0.2])
    weighted = _vjp(_bound(spec, feeds), seed)
    unweighted = _vjp(_bound(replace(spec, state_metric=()), feeds), seed)
    assert (
        np.linalg.norm(
            weighted.parameter_cotangents["q"] - unweighted.parameter_cotangents["q"]
        )
        > 0.1
    )
    # Residual coordinates may be rescaled without changing parameter derivatives.
    rescaled = _vjp(_bound(replace(spec, residual_metric=(3.0, 1.0, 2.0)), feeds), seed)
    for name in spec.parameter_names:
        np.testing.assert_allclose(
            rescaled.parameter_cotangents[name],
            weighted.parameter_cotangents[name],
            atol=1e-11,
        )


def test_scalar_state_can_have_a_differently_shaped_residual() -> None:
    from vibeqc_compiler.tensor import reshape

    spec, feeds = _scalar()
    axis = Index("a", IndexSpace("residual_coordinate", "batch", 1))
    spec = replace(
        spec,
        program=Program(
            {"residual": reshape(spec.program.outputs["residual"], (axis,))}
        ),
    )
    result = _vjp(_bound(spec, feeds), np.array(1.0))
    assert result.adjoint.shape == (1,)
    assert result.parameter_cotangents["q"] == pytest.approx(0.25)


def test_replay_regenerates_math_and_is_backend_independent() -> None:
    spec, feeds = _nonsymmetric(weights=True)
    plan = spec.compile()
    replay = ImplicitVJPPlan.from_payload(json.loads(json.dumps(plan.to_payload())))
    assert replay.identity == plan.identity
    assert replay.spec.identity == spec.identity
    rebound = BoundImplicitState(replay, feeds, reference_identity="reference-A")
    expected = _vjp(_bound(spec, feeds), np.ones(3))
    actual = _vjp(rebound, np.ones(3))
    for name in spec.parameter_names:
        np.testing.assert_array_equal(
            actual.parameter_cotangents[name], expected.parameter_cotangents[name]
        )
    for stage, program in plan.programs.items():
        for target in ("sm_80", "sm_120"):
            planned = plan_cuda(
                program,
                cuda_target_info(target),
                max_bytes=128 << 20,
                schedule=TensorSchedule(tile_m=8, tile_n=8, tile_k=8),
            )
            assert planned.program.logical_hash == program.logical_hash
        assert all(node.spec.size <= 9 for node in program.live_nodes), stage
    assert plan.identity == replay.identity


@pytest.mark.parametrize(
    "change",
    [
        {"operator_identity": "another-provider"},
        {"state_layout": "another-map"},
        {"residual_layout": "another-residual-map"},
        {"gauge": "another-independent-gauge"},
        {"state_metric": (2.0, 1.0, 1.0)},
        {"residual_metric": (2.0, 1.0, 1.0)},
    ],
)
def test_math_identity_includes_operator_layout_gauge_and_metrics(
    change: typing.Any,
) -> None:
    spec, _ = _nonsymmetric()
    assert replace(spec, **change).compile().identity != spec.compile().identity


def test_documentary_provenance_does_not_change_math_identity() -> None:
    spec, _ = _scalar()
    other = replace(
        spec,
        program=Program(
            spec.program.outputs, provenance={"note": "independent provenance"}
        ),
    )
    assert other.identity == spec.identity
    assert other.compile().identity == spec.compile().identity


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", True),
        ("rule", "unknown"),
        ("derivative_orders", [1, 2]),
        ("parameter_metric", "weighted"),
        ("extra", 1),
        ("state_metric", "not-an-array"),
    ],
)
def test_spec_payload_rejects_unknown_or_unsupported_contracts(
    field: typing.Any, value: typing.Any
) -> None:
    spec, _ = _scalar()
    payload = spec.to_payload()
    payload[field] = value
    with pytest.raises((ValueError, TypeError)):
        ImplicitSolveSpec.from_payload(payload)


@pytest.mark.parametrize(
    "mutation", ["identity", "transpose", "missing", "higher-order"]
)
def test_plan_payload_cannot_substitute_derivative_graphs(
    mutation: typing.Any,
) -> None:
    spec, _ = _scalar()
    payload = spec.compile().to_payload()
    if mutation == "identity":
        payload["identity"] = "wrong"
    elif mutation == "transpose":
        payload["programs"]["transpose"] = copy.deepcopy(payload["programs"]["rhs"])
    elif mutation == "missing":
        del payload["programs"]
    else:
        payload["spec"]["derivative_orders"] = [2]
    with pytest.raises(ValueError):
        ImplicitVJPPlan.from_payload(payload)


@pytest.mark.parametrize(
    "change",
    [
        {"parameter_names": ("q", "q")},
        {"parameter_names": ("x",)},
        {"parameter_names": ("absent",)},
        {"parameter_names": ["q"]},
        {"operator_identity": ""},
        {"state_metric": (-1.0,)},
        {"state_metric": (True,)},
        {"state_metric": (float("nan"),)},
        {"residual_metric": (1.0, 2.0)},
        {"residual_metric": [1.0]},
    ],
)
def test_invalid_declarations_fail_before_compilation(change: typing.Any) -> None:
    spec, _ = _scalar()
    with pytest.raises((ValueError, TypeError)):
        replace(spec, **change)


def test_unsupported_dtype_and_redundant_symmetry_are_not_silently_accepted() -> None:
    for dtype in ("float32",):
        x, q = _parameter("x", dtype=dtype), _parameter("q", dtype=dtype)
        with pytest.raises(NotImplementedError, match="FP64"):
            ImplicitSolveSpec(Program({"residual": add(x, q)}), "x", ("q",), "toy")
    axis = Index("i", IndexSpace("symmetry", "batch", 2))
    x = _parameter("x", (axis, Index("j", axis.space)), symmetries=(Symmetry((1, 0)),))
    q = _parameter("q", (axis, Index("j", axis.space)))
    with pytest.raises(NotImplementedError, match="independent"):
        ImplicitSolveSpec(Program({"residual": add(x, q)}), "x", ("q",), "toy")


@pytest.mark.parametrize(
    "value",
    [
        np.array(2, dtype=np.float32),
        np.array(2 + 1j),
        np.array(float("nan")),
        np.ones(1),
    ],
)
def test_strict_input_and_cotangent_shape_dtype_finiteness(
    value: typing.Any,
) -> None:
    spec, feeds = _scalar()
    with pytest.raises(ValueError):
        _bound(spec, {**feeds, "x": value})
    bound = _bound(spec, feeds)
    with pytest.raises(ValueError):
        _vjp(bound, value)
    with pytest.raises(ValueError):
        _vjp(bound, np.array(1.0), direct={"q": value})


def test_primal_and_unknown_parameter_inputs_are_checked() -> None:
    spec, feeds = _scalar()
    with pytest.raises(ImplicitSolveError, match="primal is not converged"):
        _bound(spec, {**feeds, "x": np.array(3.0)})
    with pytest.raises(ValueError, match="exactly match"):
        _bound(spec, {**feeds, "unused": np.array(1.0)})
    with pytest.raises(ValueError, match="unknown"):
        _vjp(_bound(spec, feeds), np.array(1.0), direct={"unknown": np.array(1.0)})


def test_state_inputs_and_published_results_cannot_be_mutated() -> None:
    spec, feeds = _nonsymmetric()
    # Non-contiguous caller views are accepted and snapshotted.
    feeds = {
        name: np.asfortranarray(value) if value.ndim == 2 else value[::-1][::-1]
        for name, value in feeds.items()
    }
    bound = _bound(spec, feeds)
    result = _vjp(bound, np.ones(3))
    feeds["a"][:] = 999.0
    again = _vjp(bound, np.ones(3))
    np.testing.assert_array_equal(
        result.parameter_cotangents["q"], again.parameter_cotangents["q"]
    )
    for array in (bound.feeds["a"], result.adjoint, result.parameter_cotangents["q"]):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        bound.feeds["a"] = feeds["a"]
    with pytest.raises(AttributeError, match="cannot assign"):
        bound.reference_identity = "changed"
    with pytest.raises(AttributeError, match="cannot delete"):
        del bound.feeds
    with pytest.raises((FrozenInstanceError, AttributeError)):
        bound.plan.identity = "changed"


def test_changed_reference_or_equations_change_state_and_execution_identity() -> None:
    spec, feeds = _scalar()
    first = _bound(spec, feeds)
    other = _bound(spec, {"x": np.array(3.0), "q": np.array(9.0)})
    assert first.state_identity != other.state_identity
    assert first.execution_identity != other.execution_identity
    with pytest.raises(ResponseCompatibilityError, match="reference"):
        first.vjp(np.array(1.0), reference_identity="reference-B")


class _ProbeExecutor:
    def __init__(self, plan: typing.Any) -> None:
        self.reference = ReferenceTensorExecutor(plan)
        self.plan_identity = plan.identity
        self.identity = self.reference.identity
        self.backend = self.reference.backend
        self.workspace_bytes = 0
        self.calls = []
        self.after = lambda stage: None

    def execute(self, stage: typing.Any, feeds: typing.Any) -> typing.Any:
        self.calls.append(stage)
        result = self.reference.execute(stage, feeds)
        self.after(stage)
        return result


@pytest.mark.parametrize("stage", ["transpose", "source", "adjoint"])
def test_live_reference_change_during_execution_fails_before_publication(
    stage: typing.Any,
) -> None:
    spec, feeds = _scalar()
    plan = spec.compile()
    executor = _ProbeExecutor(plan)
    identity = ["reference-A"]
    bound = _bound(
        spec, feeds, executor=executor, current_reference=lambda: identity[0]
    )
    executor.after = lambda current: (
        identity.__setitem__(0, "changed") if current == stage else None
    )
    with pytest.raises(ResponseCompatibilityError, match="stale"):
        _vjp(bound, np.array(1.0))
    identity[0] = "reference-A"
    executor.after = lambda _: None
    assert _vjp(bound, np.array(1.0)).parameter_cotangents["q"] == pytest.approx(0.25)


def test_callback_identity_and_backend_switches_fail_closed() -> None:
    spec, feeds = _scalar()
    executor = _ProbeExecutor(spec.compile())
    bound = _bound(spec, feeds, executor=executor)
    executor.identity = "changed"
    with pytest.raises(ResponseCompatibilityError, match="contract changed"):
        _vjp(bound, np.array(1.0))
    executor = _ProbeExecutor(spec.compile())
    executor.backend = "cuda-claimed"
    with pytest.raises(ResponseCompatibilityError, match="no fallback"):
        _bound(spec, feeds, executor=executor)


@pytest.mark.parametrize("max_bytes", [0, 1, -1, True, 2**64])
def test_combined_budget_preflight_precedes_tensor_execution(
    max_bytes: typing.Any,
) -> None:
    spec, feeds = _scalar()
    executor = _ProbeExecutor(spec.compile())
    with pytest.raises((ImplicitSolveError, ValueError)):
        _bound(spec, feeds, max_bytes=max_bytes, executor=executor)
    assert executor.calls == []


def test_reservation_includes_simultaneous_solver_and_executor_storage() -> None:
    spec, feeds = _scalar()
    plan = spec.compile()
    solver = ResponseGMRES(1, _options())
    executor = _ProbeExecutor(plan)
    executor.workspace_bytes = 8192
    required = (
        plan.reference_workspace_bytes
        + solver.workspace_bytes
        + executor.workspace_bytes
    )
    with pytest.raises(ImplicitSolveError, match="workspace"):
        _bound(spec, feeds, solver=solver, executor=executor, max_bytes=required - 1)
    assert executor.calls == []
    result = _vjp(
        _bound(spec, feeds, solver=solver, executor=executor, max_bytes=required),
        np.array(1.0),
    )
    assert result.logical_reserved_host_bytes == required


def test_nonconvergence_singularity_and_solver_budget_do_not_publish_results() -> None:
    spec, feeds = _nonsymmetric()
    for options in (
        GMRESOptions(rtol=1e-15, max_iterations=1, restart=1),
        GMRESOptions(max_workspace_bytes=1),
    ):
        bound = _bound(spec, feeds, solver=ResponseGMRES(3, options))
        with pytest.raises(ImplicitSolveError, match="adjoint failed"):
            _vjp(bound, np.ones(3))
    spec, feeds = _scalar()
    singular = _bound(spec, {"x": np.array(0.0), "q": np.array(0.0)})
    with pytest.raises(ImplicitSolveError, match="adjoint failed"):
        _vjp(singular, np.array(1.0))


@pytest.mark.parametrize(
    "failure",
    [
        "wrong-solution",
        "nonfinite-residual",
        "extra-workspace",
        "bad-iterations",
        "false-success",
    ],
)
def test_solver_report_is_not_a_substitute_for_true_residual(
    failure: typing.Any,
) -> None:
    spec, feeds = _scalar()
    real = ResponseGMRES(1, _options())

    class LyingSolver:
        identity = "deliberately-invalid-test-callback"
        backend = real.backend
        workspace_bytes = real.workspace_bytes
        rtol, atol = real.rtol, real.atol

        def solve(self, operator: typing.Any, rhs: typing.Any) -> typing.Any:
            answer = real.solve(operator, rhs)
            changes = {
                "wrong-solution": {"solution": np.ones(1), "residual_norm": 0.0},
                "nonfinite-residual": {"residual_norm": float("nan")},
                "extra-workspace": {"workspace_bytes": self.workspace_bytes + 1},
                "bad-iterations": {"iterations": -1},
                "false-success": {"converged": 1},
            }
            return replace(answer, **changes[failure])

    executor = _ProbeExecutor(spec.compile())
    bound = _bound(spec, feeds, solver=LyingSolver(), executor=executor)
    with pytest.raises(ImplicitSolveError):
        _vjp(bound, np.array(1.0))
    assert "source" not in executor.calls


def test_missing_executor_outputs_fail_without_success() -> None:
    spec, feeds = _scalar()
    executor = _ProbeExecutor(spec.compile())
    bound = _bound(spec, feeds, executor=executor)
    executor.execute = lambda stage, feeds: SimpleNamespace(
        outputs={}, backend=executor.backend
    )
    with pytest.raises(ImplicitSolveError, match="incomplete outputs"):
        _vjp(bound, np.array(1.0))


def test_generated_objective_vjp_composes_without_omitting_the_direct_term() -> None:
    from vibeqc_compiler.tensor import transpose_program

    spec, feeds = _scalar()
    x, q = _parameter("x"), _parameter("q")
    observable = Program({"value": add(multiply(x, multiply(x, x)), multiply(q, q))})
    reverse = transpose_program(observable, ["value"], inputs=["x", "q"])
    upstream = execute(reverse.program, {**feeds, "bar_value": np.array(1.0)}).outputs
    result = _vjp(
        _bound(spec, feeds), upstream["bar_x"], direct={"q": upstream["bar_q"]}
    )
    # f(q) = q^(3/2) + q^2 on the positive-root branch.
    assert result.parameter_cotangents["q"] == pytest.approx(11.0)


def test_matrix_free_derivative_storage_is_linear_and_independent_of_iterations() -> (
    None
):
    from implicit_fixtures import rank_one_problem

    reservations = []
    for dimension in (8, 32, 128):
        spec, _ = rank_one_problem(dimension)
        plan = spec.compile()
        for program in plan.programs.values():
            assert all(node.spec.size <= dimension for node in program.live_nodes)
        reservations.append(plan.reference_workspace_bytes)
    assert reservations[-1] < reservations[0] * 17
    # Iteration budgets affect the runtime contract, not the mathematical DAG.
    spec, feeds = _scalar()
    first = _bound(
        spec, feeds, solver=ResponseGMRES(1, GMRESOptions(max_iterations=10))
    )
    second = _bound(
        spec, feeds, solver=ResponseGMRES(1, GMRESOptions(max_iterations=100))
    )
    assert first.plan.identity == second.plan.identity
    assert first.execution_identity != second.execution_identity
    assert first.logical_reserved_host_bytes < second.logical_reserved_host_bytes


def test_cuda_global_admission_fails_before_any_compilation_or_device_probe(
    tmp_path: typing.Any,
) -> None:
    from vibeqc_compiler.common.resources import ResourceBudget

    from tools.vibeqc_response.implicit_cuda import PreparedImplicitCuda

    spec, _ = _scalar()
    compiler = SimpleNamespace(target=cuda_target_info("sm_120"))
    with pytest.raises(MemoryError, match="no supported plan fits"):
        PreparedImplicitCuda(
            spec.compile(),
            ResponseGMRES(1),
            compiler,
            tmp_path,
            budget=ResourceBudget(host_bytes=1 << 20, device_bytes=0),
        )
    assert list(tmp_path.iterdir()) == []


def test_executor_reused_output_buffers_do_not_alias_published_weights() -> None:
    spec, feeds = _scalar()

    class ReusingExecutor(_ProbeExecutor):
        def __init__(self, plan: typing.Any) -> None:
            super().__init__(plan)
            self.buffer = np.zeros(())

        def execute(self, stage: typing.Any, feeds: typing.Any) -> typing.Any:
            output = super().execute(stage, feeds)
            assert len(output.outputs) == 1
            name, value = next(iter(output.outputs.items()))
            self.buffer[...] = value
            return SimpleNamespace(outputs={name: self.buffer}, backend=self.backend)

    executor = ReusingExecutor(spec.compile())
    bound = _bound(spec, feeds, executor=executor)
    result = _vjp(bound, np.array(1.0))
    assert result.parameter_cotangents["q"] == pytest.approx(0.25)
    assert result.adjoint == pytest.approx(-0.25)
    executor.buffer[...] = 999.0
    assert result.parameter_cotangents["q"] == pytest.approx(0.25)
