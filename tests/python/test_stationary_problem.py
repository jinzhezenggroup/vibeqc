"""Independent analytic/finite-difference checks for #181's compiler boundary.

Tiny dense solves occur ONLY in these test oracles. Production generation has
no solver, nuclear Jacobian, reference library or native runtime dependency.
"""

import json
import typing
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from vibeqc_compiler.method.stationary import (
    ParameterSource,
    StationaryDerivativePlan,
    StationaryProblem,
    StationaryState,
)
from vibeqc_compiler.tensor import (
    Index,
    IndexSpace,
    Program,
    Symmetry,
    TensorSpec,
    add,
    broadcast,
    einsum,
    execute,
    input_tensor,
    multiply,
    reduce_sum,
)


def _input(
    name: typing.Any,
    indices: typing.Any = (),
    *,
    diff: typing.Any = True,
    symmetries: typing.Any = (),
    dtype: typing.Any = "float64",
) -> typing.Any:
    return input_tensor(
        name,
        TensorSpec(
            tuple(indices),
            role="parameter",
            differentiable=diff,
            symmetries=symmetries,
            dtype=dtype,
        ),
    )


def _axis(name: typing.Any, size: typing.Any) -> typing.Any:
    return Index(name, IndexSpace(name, "batch", size))


def _state(
    name: typing.Any = "x", residual: typing.Any = "residual", **kwargs: typing.Any
) -> typing.Any:
    return StationaryState(
        name, residual, f"{name}-independent-v1", "fixed-gauge-v1", **kwargs
    )


def _scalar() -> typing.Any:
    x, q = _input("x"), _input("q")
    residual = add(multiply(x, x), q, coefficients=(1, -1))
    energy = add(multiply(multiply(x, x), x), q, coefficients=(1, 2))
    return StationaryProblem(
        Program({"energy": energy, "residual": residual}),
        "energy",
        (_state(),),
        (ParameterSource("q", "external-q-v1"),),
        "nonvariational-scalar-v1",
        "test-only-transpose-solve-v1",
    )


def _partials(
    plan: typing.Any, feeds: typing.Any, multipliers: typing.Any
) -> typing.Any:
    return execute(
        plan.partials,
        {
            **feeds,
            **{
                plan.multiplier_inputs[name]: np.asarray(value)
                for name, value in multipliers.items()
            },
        },
    ).outputs


def test_nonvariational_scalar_sign_and_resolved_finite_differences() -> None:
    problem = _scalar()
    plan = problem.compile()
    q = 2.3
    x = np.sqrt(q)
    feeds = {"q": np.asarray(q), "x": np.asarray(x)}
    rhs = execute(plan.rhs, feeds).outputs["x"]
    np.testing.assert_allclose(rhs, -3 * x * x, rtol=1e-14)
    multiplier = float(rhs / (2 * x))
    result = _partials(plan, feeds, {"x": multiplier})
    np.testing.assert_allclose(result[plan.stationarity_outputs["x"]], 0, atol=2e-14)
    weight = result[plan.weight_outputs["q"]]
    np.testing.assert_allclose(weight, 2 + 1.5 * x, rtol=1e-14)
    for step in (1e-4, 1e-5, 1e-6):
        energy = lambda q: q**1.5 + 2 * q
        finite = (energy(q + step) - energy(q - step)) / (2 * step)
        np.testing.assert_allclose(weight, finite, rtol=2e-9, atol=1e-9)
    # The wrong multiplier sign is not accepted as a solved adjoint equation.
    wrong = _partials(plan, feeds, {"x": -multiplier})
    assert abs(wrong[plan.stationarity_outputs["x"]]) > 1


def _nonsymmetric(*, matrix_diff: typing.Any = True) -> typing.Any:
    space = IndexSpace("state", "batch", 3)
    i, j = Index("i", space), Index("j", space)
    x, q, a, c = (
        _input("x", (i,)),
        _input("q", (i,)),
        _input("A", (i, j), diff=matrix_diff),
        _input("c", (i,), diff=False),
    )
    residual = add(einsum("ij,j->i", a, x), q, coefficients=(1, -1))
    energy = add(einsum("i,i->", x, x, coefficient="1/2"), einsum("i,i->", c, q))
    problem = StationaryProblem(
        Program({"energy": energy, "residual": residual}),
        "energy",
        (_state(),),
        tuple(ParameterSource(name, f"{name}-v1") for name in ("A", "q", "c")),
        "nonsymmetric-linear-v1",
        "test-only-transpose-solve-v1",
    )
    a = np.array([[2.0, 0.3, -0.4], [0.0, 1.7, 0.5], [0.2, 0.0, 2.4]])
    q = np.array([0.8, -1.1, 2.0])
    return problem, {
        "A": a,
        "q": q,
        "x": np.linalg.solve(a, q),
        "c": np.array([0.3, -0.2, 0.5]),
    }


def test_declared_stationary_state_dispatches_generic_implicit_plan() -> None:
    problem, feeds = _nonsymmetric()
    state = replace(
        problem.states[0],
        implicit_operator_identity="nonsymmetric-response-v1",
        residual_layout_identity=problem.states[0].coordinate_identity,
    )
    problem = replace(problem, states=(state,))
    plan = problem.compile()
    assert set(plan.implicit_plans) == {"x"}
    implicit = plan.implicit_plans["x"]
    assert implicit.spec.operator_identity == "nonsymmetric-response-v1"
    assert implicit.spec.state_layout == state.coordinate_identity
    assert implicit.spec.residual_layout == state.coordinate_identity
    assert implicit.spec.gauge == state.gauge_identity
    assert set(implicit.spec.parameter_names) == {"A", "q"}

    vector = np.array([0.3, -0.2, 0.7])
    generated = execute(
        implicit.programs["transpose"],
        {**feeds, "__implicit_vector": vector},
    ).outputs["value"]
    np.testing.assert_allclose(generated, feeds["A"].T @ vector, atol=2e-15)

    replay = StationaryDerivativePlan.loads(plan.dumps())
    assert replay.implicit_plans["x"].identity == implicit.identity
    changed = replace(
        problem,
        states=(replace(state, implicit_operator_identity="different-response-v2"),),
    ).compile()
    assert changed.implicit_plans["x"].identity != implicit.identity


def test_automatic_implicit_dispatch_rejects_unpacked_coupled_states() -> None:
    problem = _constrained()
    states = tuple(
        replace(
            state,
            implicit_operator_identity="kkt-row-v1",
            residual_layout_identity=state.coordinate_identity,
        )
        if state.name == "x"
        else state
        for state in problem.states
    )
    with pytest.raises(NotImplementedError, match="coupled"):
        replace(problem, states=states).compile()


@pytest.mark.parametrize("matrix_diff", [False, True])
def test_nonsymmetric_coupled_state_and_parameter_weights(
    matrix_diff: typing.Any,
) -> None:
    problem, feeds = _nonsymmetric(matrix_diff=matrix_diff)
    plan = problem.compile()
    rhs = execute(plan.rhs, feeds).outputs["x"]
    multiplier = np.linalg.solve(feeds["A"].T, rhs)
    result = _partials(plan, feeds, {"x": multiplier})
    np.testing.assert_allclose(result[plan.stationarity_outputs["x"]], 0, atol=2e-15)
    wq = result[plan.weight_outputs["q"]]
    np.testing.assert_allclose(wq, feeds["c"] - multiplier, atol=2e-15)
    assert "c" not in plan.weight_outputs
    assert ("A" in plan.weight_outputs) is matrix_diff
    dq = np.array([0.3, -0.4, 0.7])
    da = (
        np.array([[0.1, 0.2, 0.0], [-0.1, 0.1, 0.3], [0.2, 0.1, -0.1]])
        if matrix_diff
        else np.zeros((3, 3))
    )
    directional = float(wq @ dq)
    if matrix_diff:
        wa = result[plan.weight_outputs["A"]]
        np.testing.assert_allclose(wa, np.outer(multiplier, feeds["x"]), atol=2e-15)
        directional += float(np.sum(wa * da))

    def energy(t: typing.Any) -> typing.Any:
        q = feeds["q"] + t * dq
        x = np.linalg.solve(feeds["A"] + t * da, q)
        return 0.5 * (x @ x) + feeds["c"] @ q

    for step in (1e-4, 1e-5, 1e-6):
        np.testing.assert_allclose(
            directional,
            (energy(step) - energy(-step)) / (2 * step),
            rtol=3e-9,
            atol=1e-9,
        )


def _constrained() -> typing.Any:
    i = _axis("state", 2)
    x, a, mu, q = _input("x", (i,)), _input("a", (i,)), _input("mu"), _input("q")
    difference = add(x, a, coefficients=(1, -1))
    rx = add(difference, broadcast(mu, (i,), ()))
    constraint = add(reduce_sum(x, (0,)), q, coefficients=(1, -1))
    energy = add(
        einsum("i,i->", difference, difference, coefficient="1/2"),
        q,
        coefficients=(1, "3/10"),
    )
    return StationaryProblem(
        Program({"energy": energy, "rx": rx, "constraint": constraint}),
        "energy",
        (_state("x", "rx"), _state("mu", "constraint", kind="constraint")),
        (
            ParameterSource("a", "target-v1"),
            ParameterSource("q", "constraint-value-v1"),
        ),
        "constrained-quadratic-v1",
        "test-only-kkt-solve-v1",
    )


def test_constraint_is_a_declared_equation_not_inferred_from_energy() -> None:
    problem = _constrained()
    plan = problem.compile()
    a, q = np.array([1.2, -0.4]), 0.3
    mu = (sum(a) - q) / 2
    x = a - mu
    feeds = {"x": x, "a": a, "mu": np.asarray(mu), "q": np.asarray(q)}
    rhs = execute(plan.rhs, feeds).outputs
    # Independent KKT matrix, ordered (x0,x1,mu), NOT compiler differentiation.
    kkt = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 0.0]])
    multiplier = np.linalg.solve(kkt.T, np.r_[rhs["x"], rhs["mu"]])
    result = _partials(plan, feeds, {"x": multiplier[:2], "mu": multiplier[2]})
    for output in plan.stationarity_outputs.values():
        np.testing.assert_allclose(result[output], 0, atol=2e-15)
    np.testing.assert_allclose(result[plan.weight_outputs["q"]], 0.3 - mu, atol=2e-15)
    np.testing.assert_allclose(result[plan.weight_outputs["a"]], [mu, mu], atol=2e-15)
    assert problem.dependency_graph["implicit_region"]["constraints"] == ["constraint"]
    da, dq = np.array([0.2, -0.1]), -0.3
    expected = (
        result[plan.weight_outputs["a"]] @ da + result[plan.weight_outputs["q"]] * dq
    )

    def energy(t: typing.Any) -> typing.Any:
        at, qt = a + t * da, q + t * dq
        mut = (sum(at) - qt) / 2
        return mut**2 + 0.3 * qt

    for step in (1e-4, 1e-5, 1e-6):
        np.testing.assert_allclose(
            expected, (energy(step) - energy(-step)) / (2 * step), rtol=1e-8, atol=1e-10
        )
    with pytest.raises(ValueError, match="declared residuals"):
        replace(problem, states=tuple(s for s in problem.states if s.name != "mu"))


def _provider_problem() -> typing.Any:
    x, s, h = _input("x"), _input("overlap"), _input("h")
    return StationaryProblem(
        Program(
            {
                "energy": multiply(x, h),
                "residual": add(multiply(s, x), h, coefficients=(1, -1)),
            }
        ),
        "energy",
        (_state(),),
        (
            ParameterSource("geometry", "geometry-v1"),
            ParameterSource(
                "overlap", "overlap-field-v1", ("geometry",), "overlap-pullback-v1"
            ),
            ParameterSource("h", "h-field-v1", ("geometry",), "h-pullback-v1"),
        ),
        "provider-toy-v1",
        "test-only-solver-v1",
    )


def test_provider_dag_preserves_overlap_and_stops_at_source_boundaries() -> None:
    problem = _provider_problem()
    graph = problem.dependency_graph
    assert graph["providers"] == {
        "geometry": [],
        "h": ["geometry"],
        "overlap": ["geometry"],
    }
    assert graph["implicit_region"]["residual_dependencies"]["residual"] == [
        "h",
        "overlap",
        "x",
    ]
    assert graph["provider_order"][0] == "geometry"
    assert set(graph["provider_pullback_order"]) == {"overlap", "h"}
    plan = problem.compile()
    assert set(plan.weight_outputs) == {"overlap", "h"}
    assert "geometry" not in plan.weight_outputs  # provider adjoints are NOT executed
    assert {s.pullback_identity for s in plan.provider_pullbacks} == {
        "overlap-pullback-v1",
        "h-pullback-v1",
    }
    with pytest.raises(ValueError, match="missing parameter source"):
        replace(
            problem, sources=tuple(s for s in problem.sources if s.name != "overlap")
        )
    with pytest.raises(ValueError, match="missing provider dependency"):
        replace(
            problem, sources=tuple(s for s in problem.sources if s.name != "geometry")
        )


@pytest.mark.parametrize(
    "case", ["duplicate-name", "duplicate-identity", "cycle", "state-cycle", "unused"]
)
def test_source_graph_rejects_ambiguous_or_incomplete_dependencies(
    case: typing.Any,
) -> None:
    problem = _scalar()
    q = problem.sources[0]
    if case == "duplicate-name":
        sources = (q, replace(q, identity="other-field"))
    elif case == "duplicate-identity":
        sources = (q, ParameterSource("other", q.identity))
    elif case == "cycle":
        sources = (
            replace(q, dependencies=("other",), pullback_identity="q-rule"),
            ParameterSource("other", "other-field", ("q",), "other-rule"),
        )
    elif case == "state-cycle":
        sources = (replace(q, dependencies=("x",), pullback_identity="q-rule"),)
    else:
        sources = (q, ParameterSource("overlap", "overlap-field"))
    with pytest.raises(ValueError, match="duplicate|cycle|unused|dependency"):
        replace(problem, sources=sources)


@pytest.mark.parametrize("factory", [_scalar, _provider_problem, _constrained])
def test_serialization_and_identity_are_canonical_and_data_only(
    factory: typing.Any,
) -> None:
    problem = factory()
    replay = StationaryProblem.loads(problem.dumps())
    assert replay.identity == problem.identity
    assert replay.dumps() == problem.dumps()
    assert replay.dependency_graph == problem.dependency_graph
    reordered = replace(
        problem,
        states=tuple(reversed(problem.states)),
        sources=tuple(reversed(problem.sources)),
    )
    assert reordered.dumps() == problem.dumps()
    first, second = problem.compile(), replay.compile()
    assert first.identity == second.identity
    for name in ("lagrangian", "rhs", "partials"):
        graph = getattr(first, name)
        assert Program.loads(graph.dumps()).logical_hash == graph.logical_hash
    for name in ("model_identity", "solver_contract"):
        assert replace(problem, **{name: "different-v2"}).identity != problem.identity
    state = replace(problem.states[0], gauge_identity="other-gauge")
    assert (
        replace(problem, states=(state, *problem.states[1:])).identity
        != problem.identity
    )
    source = replace(problem.sources[0], identity="other-field")
    assert (
        replace(problem, sources=(source, *problem.sources[1:])).identity
        != problem.identity
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("schema_version", 3),
        ("convention", "opposite-sign"),
        ("identity", "tampered"),
        ("model_identity", "altered"),
        ("unknown", "extra"),
    ],
)
def test_problem_replay_rejects_tampering(field: typing.Any, value: typing.Any) -> None:
    payload = _scalar().to_payload()
    payload[field] = value
    with pytest.raises(ValueError):
        StationaryProblem.from_payload(payload)


def test_duplicate_json_fields_are_rejected() -> None:
    source = (
        _scalar()
        .dumps()
        .replace('"schema_version": 2', '"schema_version": 2, "schema_version": 2')
    )
    with pytest.raises(ValueError, match="duplicate JSON field"):
        StationaryProblem.loads(source)


def test_records_and_plan_maps_are_immutable_snapshots() -> None:
    deps = ["geometry"]
    source = ParameterSource("q", "q-field", deps, "q-rule")
    deps.append("wrong")
    assert source.dependencies == ("geometry",)
    problem = _scalar()
    with pytest.raises(FrozenInstanceError):
        problem.objective = "wrong"
    plan = problem.compile()
    with pytest.raises(TypeError):
        plan.multiplier_inputs["x"] = "wrong"
    changed = problem.dependency_graph
    changed["implicit_region"]["states"].clear()
    assert problem.dependency_graph["implicit_region"]["states"] == ["x"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inner_product": "unweighted-packed"},
        {"coordinate_identity": ""},
        {"gauge_identity": ""},
        {"kind": "guessed"},
        {"residual_layout_identity": "layout-without-operator"},
        {"name": "stationary_bad"},
    ],
)
def test_state_contract_fails_closed(kwargs: typing.Any) -> None:
    with pytest.raises(ValueError):
        replace(_state(), **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dependencies": ("a",)},
        {"dependencies": ("a", "a"), "pullback_identity": "rule"},
        {"dependencies": "a", "pullback_identity": "rule"},
        {"pullback_identity": "unused"},
        {"identity": ""},
    ],
)
def test_provider_contract_fails_closed(kwargs: typing.Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        ParameterSource("q", **{"identity": "q-field", **kwargs})


def test_no_sources_and_objective_independent_state_are_supported() -> None:
    x, c = _input("x"), _input("c", diff=False)
    problem = StationaryProblem(
        Program({"energy": c, "residual": x}),
        "energy",
        (_state(),),
        (ParameterSource("c", "fixed-c-v1"),),
        "constant-objective-v1",
        "test-solver-v1",
    )
    plan = problem.compile()
    assert plan.weight_outputs == {}
    feeds = {"x": np.asarray(0.0), "c": np.asarray(1.0)}
    np.testing.assert_array_equal(execute(plan.rhs, feeds).outputs["x"], 0)
    np.testing.assert_array_equal(
        _partials(plan, feeds, {"x": 0.0})[plan.stationarity_outputs["x"]], 0
    )
    free = replace(
        problem,
        equations=Program({"energy": multiply(x, x), "residual": x}),
        sources=(),
    )
    assert free.compile().provider_pullbacks == ()


def test_redundant_state_coordinates_and_fp32_are_not_silently_admitted() -> None:
    space = IndexSpace("state", "batch", 2)
    i, j = Index("i", space), Index("j", space)
    x = _input("x", (i, j), symmetries=(Symmetry((1, 0), 1),))
    with pytest.raises(ValueError, match="independent coordinates"):
        replace(
            _scalar(),
            equations=Program({"energy": einsum("ij,ij->", x, x), "residual": x}),
            sources=(),
        )
    x = _input("x", dtype="float32")
    with pytest.raises(ValueError, match="float64"):
        replace(
            _scalar(),
            equations=Program({"energy": multiply(x, x), "residual": x}),
            sources=(),
        )


def test_dense_symmetric_parameter_uses_existing_projected_adjoint() -> None:
    space = IndexSpace("source", "batch", 2)
    i, j = Index("i", space), Index("j", space)
    q = _input("q", (i, j), symmetries=(Symmetry((1, 0), 1),))
    w = _input("w", (i, j), diff=False)
    x = _input("x")
    problem = StationaryProblem(
        Program(
            {
                "energy": add(multiply(x, x), einsum("ij,ij->", w, q)),
                "residual": add(x, einsum("ii->", q), coefficients=(1, -1)),
            }
        ),
        "energy",
        (_state(),),
        (ParameterSource("q", "symmetric-field"), ParameterSource("w", "fixed-weight")),
        "symmetric-source-v1",
        "test-solver-v1",
    )
    plan = problem.compile()
    feeds = {
        "x": np.asarray(3.0),
        "q": np.array([[1.0, 0.2], [0.2, 2.0]]),
        "w": np.array([[1.0, 2.0], [5.0, 3.0]]),
    }
    result = _partials(plan, feeds, {"x": -6.0})
    expected = (feeds["w"] + feeds["w"].T) / 2 + 6 * np.eye(2)
    np.testing.assert_allclose(result[plan.weight_outputs["q"]], expected, atol=1e-14)
    with pytest.raises(ValueError, match="element budget"):
        problem.compile(max_elements=0)


def test_large_diagonal_problem_does_not_materialize_state_jacobian() -> None:
    axis = _axis("state", 2000)
    x, q = _input("x", (axis,)), _input("q", (axis,))
    problem = replace(
        _scalar(),
        equations=Program(
            {
                "energy": einsum("i,i->", x, x, coefficient="1/2"),
                "residual": add(multiply(x, x), q, coefficients=(1, -1)),
            }
        ),
    )
    plan = problem.compile()
    assert len(plan.partials.live_nodes) < 60
    assert max(node.spec.size for node in plan.partials.live_nodes) <= 2000


def test_generated_fragments_reuse_cuda_planning_without_device_or_runtime() -> None:
    from vibeqc_compiler.integral.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    problem, _ = _nonsymmetric()
    generated = problem.compile()
    for program in (generated.rhs, generated.partials):
        plan = plan_cuda(program, cuda_target_info("sm_80"))
        assert plan.peak_bytes > 0
        assert plan.program.logical_hash == program.logical_hash
    # This is planning, NOT native/GPU execution qualification.


@pytest.mark.parametrize(
    "group,field",
    [
        ("states", "inner_product"),
        ("states", "kind"),
        ("sources", "dependencies"),
        ("sources", "pullback_identity"),
    ],
)
def test_replay_requires_complete_versioned_record_fields(
    group: typing.Any, field: typing.Any
) -> None:
    payload = _scalar().to_payload()
    del payload[group][0][field]
    with pytest.raises(ValueError, match="fields"):
        StationaryProblem.from_payload(payload)


def test_plan_replay_regenerates_derivatives_and_rejects_tampering() -> None:
    plan = _scalar().compile()
    replay = StationaryDerivativePlan.loads(plan.dumps())
    assert replay.identity == plan.identity
    assert replay.dumps() == plan.dumps()
    payload = json.loads(plan.dumps())
    payload["programs"]["rhs"] = plan.partials.to_payload()
    with pytest.raises(ValueError, match="regenerated equations"):
        StationaryDerivativePlan.from_payload(payload)
    payload = plan.to_payload()
    payload["schema_version"] = True
    with pytest.raises(ValueError, match="regenerated equations"):
        StationaryDerivativePlan.from_payload(payload)
    with pytest.raises(ValueError, match="fields"):
        StationaryDerivativePlan.from_payload({})


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"states": ()}, "state blocks"),
        ({"states": ("x",)}, "state blocks"),
        ({"sources": ("q",)}, "ParameterSource"),
        ({"equations": "fake"}, "TensorIR"),
        ({"objective": "missing"}, "declared residuals"),
        ({"model_identity": ""}, "identity"),
        ({"solver_contract": ""}, "identity"),
    ],
)
def test_problem_requires_complete_typed_declarations(
    kwargs: typing.Any, match: typing.Any
) -> None:
    with pytest.raises((ValueError, TypeError), match=match):
        replace(_scalar(), **kwargs)


def test_unclassified_output_scalar_objective_and_state_domain_checks() -> None:
    problem = _scalar()
    outputs = dict(problem.equations.outputs)
    with pytest.raises(ValueError, match="declared residuals"):
        replace(
            problem, equations=Program({**outputs, "unclassified": outputs["energy"]})
        )
    with pytest.raises(ValueError, match="duplicate"):
        replace(problem, states=(*problem.states, *problem.states))
    with pytest.raises(ValueError, match="disjoint"):
        replace(problem, sources=(*problem.sources, ParameterSource("x", "bad-source")))
    x, q = _input("x"), _input("q")
    with pytest.raises(ValueError, match="depend on a declared state"):
        replace(problem, equations=Program({"energy": x, "residual": q}))
    with pytest.raises(ValueError, match="live differentiable input"):
        replace(problem, states=(_state("absent"),))
    xf = _input("x", diff=False)
    with pytest.raises(ValueError, match="live differentiable input"):
        replace(problem, equations=Program({"energy": multiply(xf, q), "residual": xf}))
    i = _axis("v", 2)
    vector = _input("x", (i,))
    with pytest.raises(ValueError, match="objective must be scalar"):
        replace(
            problem,
            equations=Program({"energy": vector, "residual": vector}),
            sources=(),
        )
    with pytest.raises(ValueError, match="coordinate semantics"):
        replace(
            problem,
            equations=Program(
                {
                    "energy": reduce_sum(vector, (0,)),
                    "residual": reduce_sum(vector, (0,)),
                }
            ),
            sources=(),
        )
    other = _input("q", (_axis("different_domain", 2),))
    with pytest.raises(ValueError, match="coordinate semantics"):
        replace(
            problem,
            equations=Program({"energy": reduce_sum(vector, (0,)), "residual": other}),
        )


def test_derived_sources_cannot_be_declared_frozen_and_missing_equations_reject() -> (
    None
):
    problem = _provider_problem()
    x, s, h = _input("x"), _input("overlap", diff=False), _input("h")
    with pytest.raises(ValueError, match="freeze"):
        replace(
            problem,
            equations=Program(
                {
                    "energy": multiply(x, h),
                    "residual": add(multiply(s, x), h, coefficients=(1, -1)),
                }
            ),
        )
    with pytest.raises(ValueError, match="unused declared"):
        replace(
            problem,
            equations=Program(
                {"energy": multiply(x, h), "residual": add(x, h, coefficients=(1, -1))}
            ),
        )


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_compile_budget_is_explicit_and_strict(value: typing.Any) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        _scalar().compile(max_elements=value)


def test_repeated_named_inputs_are_accumulated_by_shared_tensor_ad() -> None:
    x, q1, q2 = _input("x"), _input("q"), _input("q")
    problem = replace(
        _scalar(),
        equations=Program(
            {
                "energy": add(multiply(x, x), q1, q2),
                "residual": add(x, q1, coefficients=(1, -1)),
            }
        ),
    )
    plan = problem.compile()
    result = _partials(plan, {"x": np.asarray(2.0), "q": np.asarray(2.0)}, {"x": -4.0})
    np.testing.assert_allclose(result[plan.weight_outputs["q"]], 6.0, atol=1e-14)


def test_diagnostic_executor_rejects_nonfinite_inputs_and_enforces_its_budget() -> None:
    plan = _scalar().compile()
    with pytest.raises((ValueError, FloatingPointError), match="finite|non-finite"):
        _partials(plan, {"x": np.asarray(np.nan), "q": np.asarray(2.0)}, {"x": -1.0})
    with pytest.raises((ValueError, MemoryError), match="budget|bytes"):
        execute(
            plan.partials,
            {
                "x": np.asarray(2.0),
                "q": np.asarray(4.0),
                plan.multiplier_inputs["x"]: np.asarray(-3.0),
            },
            max_bytes=0,
        )


@pytest.mark.parametrize("name", ["bad name", "", 42])
def test_identifiers_reject_invalid_types_and_spellings(name: typing.Any) -> None:
    with pytest.raises(ValueError, match="identifier"):
        replace(_state(), name=name)


def test_malformed_replay_records_and_wrong_compiler_object_reject() -> None:
    from vibeqc_compiler.method.stationary import compile_stationary

    payload = _scalar().to_payload()
    payload["states"] = 42
    with pytest.raises(ValueError, match="malformed"):
        StationaryProblem.from_payload(payload)
    with pytest.raises(TypeError, match="StationaryProblem"):
        compile_stationary("fake")
