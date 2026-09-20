"""Generated Lambda RHS/J* actions against independent determinant differences."""

import typing
from functools import lru_cache

import numpy as np
import pytest
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor import Program, execute
from vibeqc_compiler.tensor.cuda_plan import plan_cuda

from tools.vibeqc_cc.equations import amplitude_layouts
from tools.vibeqc_cc.lambda_equations import build_lambda_programs
from tools.vibeqc_cc.oracle import DeterminantOracle, dense_feeds, random_case


@lru_cache(maxsize=8)
def _programs(o: typing.Any, v: typing.Any, form: typing.Any = "shared") -> typing.Any:
    return build_lambda_programs(o, v, form=form)


def _pack(layouts: typing.Any, arrays: typing.Any) -> typing.Any:
    return np.concatenate([layout.pack(a) for layout, a in zip(layouts, arrays)])


def _unpack(layouts: typing.Any, vector: typing.Any) -> typing.Any:
    split = layouts[0].size
    return layouts[0].unpack(vector[:split]), layouts[1].unpack(vector[split:])


def _rhs(programs: typing.Any, feeds: typing.Any) -> typing.Any:
    result = execute(
        programs.energy_vjp.program,
        {**feeds, "bar_correlation_energy": np.asarray(-1.0)},
    ).outputs
    return result["bar_t1"], result["bar_t2"]


def _transpose(
    programs: typing.Any, feeds: typing.Any, arrays: typing.Any
) -> typing.Any:
    result = execute(
        programs.residual_vjp.program,
        {**feeds, "bar_singles_residual": arrays[0], "bar_doubles_residual": arrays[1]},
    ).outputs
    return result["bar_t1"], result["bar_t2"]


@pytest.mark.parametrize("o,v", [(1, 1), (1, 2), (2, 2), (2, 3)])
def test_generated_energy_and_residual_actions_against_determinant_directions(
    o: typing.Any, v: typing.Any
) -> None:
    programs = _programs(o, v)
    f, g, t1, t2 = random_case(o, v, 152)
    feeds = dense_feeds(f, g, t1, t2)
    layouts = amplitude_layouts(o, v)
    rng = np.random.default_rng(153)
    u = tuple(layout.unpack(rng.normal(size=layout.size)) for layout in layouts)
    lam = tuple(layout.unpack(rng.normal(size=layout.size)) for layout in layouts)
    forward = execute(
        programs.residual_jvp.program, {**feeds, "d_t1": u[0], "d_t2": u[1]}
    ).outputs
    j_u = (forward["d_singles_residual"], forward["d_doubles_residual"])
    jt_lam = _transpose(programs, feeds, lam)
    rhs = _rhs(programs, feeds)
    np.testing.assert_allclose(
        sum(np.sum(a * b) for a, b in zip(lam, j_u)),
        sum(np.sum(a * b) for a, b in zip(jt_lam, u)),
        atol=1e-11,
        rtol=1e-11,
    )
    oracle = DeterminantOracle(f, g, o)
    for h in (1e-4, 3e-5, 1e-5):
        plus = oracle.evaluate_full(t1 + h * u[0], t2 + h * u[1])
        minus = oracle.evaluate_full(t1 - h * u[0], t2 - h * u[1])
        np.testing.assert_allclose(
            -sum(np.sum(a * b) for a, b in zip(rhs, u)),
            (plus[0] - minus[0]) / (2 * h),
            atol=1e-8,
            rtol=1e-8,
        )
        for actual, upper, lower in zip(j_u, plus[1:], minus[1:]):
            np.testing.assert_allclose(
                actual, (upper - lower) / (2 * h), atol=2e-7, rtol=1e-7
            )
    np.testing.assert_allclose(
        jt_lam[1], jt_lam[1].transpose(1, 0, 3, 2), atol=1e-13, rtol=0
    )


@pytest.fixture(scope="module")
def numerical_jacobian() -> typing.Any:
    o, v = 2, 2
    f, g, t1, t2 = random_case(o, v, 152)
    layouts = amplitude_layouts(o, v)
    point = _pack(layouts, (t1, t2))
    oracle = DeterminantOracle(f, g, o)
    matrix = np.empty((point.size, point.size))
    energy_derivative = np.empty(point.size)
    step = 1e-5
    for column in range(point.size):
        direction = np.zeros_like(point)
        direction[column] = step
        upper = oracle.evaluate_full(*_unpack(layouts, point + direction))
        lower = oracle.evaluate_full(*_unpack(layouts, point - direction))
        matrix[:, column] = (_pack(layouts, upper[1:]) - _pack(layouts, lower[1:])) / (
            2 * step
        )
        energy_derivative[column] = (upper[0] - lower[0]) / (2 * step)
    weights = np.concatenate([np.asarray(layout.weights) for layout in layouts])
    return dense_feeds(f, g, t1, t2), layouts, matrix, energy_derivative, weights


def test_numerical_jacobian_transpose_and_orbit_multiplicity(
    numerical_jacobian: typing.Any,
) -> None:
    feeds, layouts, matrix, energy_derivative, weights = numerical_jacobian
    programs = _programs(2, 2)
    lam = np.random.default_rng(154).normal(size=len(weights))
    actual = _pack(layouts, _transpose(programs, feeds, _unpack(layouts, lam)))
    expected = (matrix.T @ (weights * lam)) / weights
    np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(
        _pack(layouts, _rhs(programs, feeds)),
        -energy_derivative / weights,
        atol=1e-10,
        rtol=1e-9,
    )
    # Ordinary Euclidean transposition in unweighted packed coordinates is wrong.
    assert np.max(np.abs(actual - matrix.T @ lam)) > 1e-4


def test_generated_actions_plug_into_existing_gmres(
    numerical_jacobian: typing.Any,
) -> None:
    from tools.vibeqc_response.krylov import GMRESOptions, solve

    feeds, layouts, matrix, energy_derivative, weights = numerical_jacobian
    programs = _programs(2, 2)
    sqrt_weights = np.sqrt(weights)

    class Operator:
        dimension = len(weights)

        def apply(self, vector: typing.Any) -> typing.Any:
            dense = _unpack(layouts, vector / sqrt_weights)
            return sqrt_weights * _pack(layouts, _transpose(programs, feeds, dense))

    rhs = sqrt_weights * _pack(layouts, _rhs(programs, feeds))
    result = solve(
        Operator(),
        rhs,
        options=GMRESOptions(rtol=1e-12, atol=1e-12, restart=20, max_iterations=30),
    )
    assert result.converged, result.reason
    assert result.residual_norm <= 1e-11
    euclidean_matrix = sqrt_weights[:, None] * matrix / sqrt_weights[None, :]
    reference = np.linalg.solve(euclidean_matrix.T, -energy_derivative / sqrt_weights)
    np.testing.assert_allclose(result.solution, reference, atol=1e-9, rtol=1e-9)
    assert np.max(np.abs(Operator().apply(result.solution) - rhs)) <= 1e-11
    failed = solve(Operator(), rhs, options=GMRESOptions(max_workspace_bytes=1))
    assert not failed.converged and failed.reason == "workspace_limit"
    # This is an operator/solver interoperability test, not a converged-CC response API.


def test_replay_equation_forms_and_provenance() -> None:
    arrays = random_case(2, 2, 155)
    feeds = dense_feeds(*arrays)
    expected = None
    for form in ("shared", "expanded", "optimized"):
        programs = _programs(2, 2, form)
        provenance = programs.provenance()
        for field in ("energy_vjp", "residual_vjp", "residual_jvp"):
            assert (
                provenance[field]["primal_logical_hash"] == programs.primal.logical_hash
            )
        for derivative, seeds in (
            (programs.energy_vjp, {"bar_correlation_energy": np.asarray(-1.0)}),
            (
                programs.residual_vjp,
                {"bar_singles_residual": arrays[2], "bar_doubles_residual": arrays[3]},
            ),
        ):
            original = execute(derivative.program, {**feeds, **seeds}).outputs
            replay = execute(
                Program.loads(derivative.program.dumps()), {**feeds, **seeds}
            ).outputs
            for key in original:
                np.testing.assert_array_equal(original[key], replay[key])
        action = _transpose(programs, feeds, arrays[2:])
        if expected is None:
            expected = action
        for actual, reference in zip(action, expected):
            np.testing.assert_allclose(actual, reference, atol=1e-12, rtol=1e-12)


def test_lambda_programs_have_no_amplitude_squared_projection_and_plan_for_cuda() -> (
    None
):
    target = cuda_target_info("sm_80")
    sizes = []
    for o, v in ((2, 2), (4, 6)):
        programs = _programs(o, v)
        current = []
        for derivative in (
            programs.energy_vjp,
            programs.residual_vjp,
            programs.residual_jvp,
        ):
            nodes = derivative.program.live_nodes
            assert not any(node.op in ("constant", "gather") for node in nodes)
            assert max(len(node.spec.indices) for node in nodes) <= 4
            current.append(len(nodes))
            if (o, v) == (2, 2):
                plan = plan_cuda(derivative.program, target)
                assert plan.peak_bytes > 0
        sizes.append(current)
    assert sizes[0] == sizes[1]


@pytest.mark.parametrize("kind", ["shape", "dtype", "symmetry", "nonfinite", "budget"])
def test_linearization_fails_closed_for_bad_seeds_and_budgets(
    kind: typing.Any,
) -> None:
    programs = _programs(2, 2)
    feeds = dense_feeds(*random_case())
    tangent1, tangent2 = np.ones((2, 2)), np.ones((2, 2, 2, 2))
    if kind == "shape":
        tangent1 = np.ones(4)
    elif kind == "dtype":
        tangent1 = tangent1.astype(complex)
    elif kind == "symmetry":
        tangent2[0, 1, 0, 1] = 2
    elif kind == "nonfinite":
        tangent1[0, 0] = np.nan
    with pytest.raises(ValueError):
        execute(
            programs.residual_jvp.program,
            {**feeds, "d_t1": tangent1, "d_t2": tangent2},
            max_bytes=1 if kind == "budget" else 256 << 20,
        )
