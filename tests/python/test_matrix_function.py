"""Small independent first-order matrix-function gates; no native library/GPU."""

import json
import typing
from dataclasses import replace

import numpy as np
import pytest
from vibeqc_compiler.method.matrix_function import SymmetricMatrixFunctionSpec
from vibeqc_compiler.method.matrix_function_cuda import (
    emit_pseudoinverse_vjp_cuda,
)
from vibeqc_compiler.tensor import Program, execute


def _symmetric(rng: typing.Any, n: typing.Any) -> typing.Any:
    raw = rng.normal(size=(n, n))
    return (raw + raw.T) / 2


def _fixture(eigenvalues: typing.Any) -> typing.Any:
    rng = np.random.default_rng(466)
    n = len(eigenvalues)
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    matrix = (q * eigenvalues) @ q.T
    tangent = _symmetric(rng, n)
    tangent /= np.linalg.norm(tangent)
    return matrix, tangent, rng.normal(size=(n, n))


def _reference(matrix: typing.Any, threshold: typing.Any) -> typing.Any:
    """Independent forward spectral evaluation, never calls the new primitive."""
    values, vectors = np.linalg.eigh(matrix)
    keep = values > (0 if threshold is None else threshold * values[-1])
    return sum(
        np.outer(vectors[:, i], vectors[:, i]) / np.sqrt(values[i])
        for i in range(len(values))
        if keep[i]
    )


def test_scalar_and_diagonal_closed_form() -> None:
    state = SymmetricMatrixFunctionSpec(1, "scalar").prepare(np.array([[4.0]]))
    np.testing.assert_array_equal(state.value, [[0.5]])
    np.testing.assert_array_equal(state.jvp(np.array([[3.0]])), [[-3 / 16]])
    matrix = np.diag([1.0, 4.0, 9.0])
    bar = np.arange(9, dtype=float).reshape(3, 3)
    state = SymmetricMatrixFunctionSpec(3, "diagonal").prepare(matrix)
    roots = np.array([1.0, 2.0, 3.0])
    expected = (
        -(bar + bar.T)
        / 2
        / (roots[:, None] * roots[None, :] * (roots[:, None] + roots[None, :]))
    )
    np.testing.assert_allclose(state.vjp(bar), expected, atol=1e-15, rtol=1e-14)


@pytest.mark.parametrize(
    "spectrum,threshold",
    [
        ([1.0, 2.0, 5.0], None),
        ([2.0, 2.0, 5.0], None),
        ([1.0, 1.0 + 1e-13, 4.0], None),
        ([0.02, 0.05, 1.0, 3.0], 0.1),
        ([0.03, 0.03, 2.0, 2.0], 0.1),
    ],
)
def test_multistep_finite_differences_and_full_frobenius_dot(
    spectrum: typing.Any, threshold: typing.Any
) -> None:
    matrix, tangent, bar = _fixture(spectrum)
    state = SymmetricMatrixFunctionSpec(len(spectrum), "random", threshold).prepare(
        matrix
    )
    expected = state.jvp(tangent)
    np.testing.assert_allclose(
        state.value, _reference(matrix, threshold), atol=2e-14, rtol=2e-14
    )
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        finite = (
            _reference(matrix + step * tangent, threshold)
            - _reference(matrix - step * tangent, threshold)
        ) / (2 * step)
        errors.append(np.max(np.abs(finite - expected)))
    assert errors[-1] < 2e-8
    assert errors[-1] < errors[0] * 0.05
    np.testing.assert_allclose(
        np.vdot(expected, bar), np.vdot(tangent, state.vjp(bar)), atol=3e-14, rtol=3e-13
    )


def test_independent_sylvester_oracle() -> None:
    """Different algorithm: solve S dX + dX S = -X dM X for X=S^-1."""
    root = np.array([[2.0, 0.3, -0.2], [0.3, 1.7, 0.1], [-0.2, 0.1, 2.5]])
    matrix = root @ root
    tangent = np.array([[0.7, -0.1, 0.3], [-0.1, 0.2, 0.4], [0.3, 0.4, -0.2]])
    inverse = np.linalg.inv(root)
    operator = np.kron(root, np.eye(3)) + np.kron(np.eye(3), root)
    expected = np.linalg.solve(
        operator, (-inverse @ tangent @ inverse).ravel()
    ).reshape(3, 3)
    state = SymmetricMatrixFunctionSpec(3, "sylvester").prepare(matrix)
    np.testing.assert_allclose(state.value, inverse, atol=2e-15, rtol=2e-14)
    np.testing.assert_allclose(state.jvp(tangent), expected, atol=2e-15, rtol=2e-14)


def test_fixed_rank_keeps_cross_subspace_response_including_psd_nullspace() -> None:
    matrix = np.diag([0.0, 4.0])
    tangent = np.array([[0.0, 1.0], [1.0, 0.0]])
    state = SymmetricMatrixFunctionSpec(2, "psd", 0.1).prepare(matrix)
    np.testing.assert_array_equal(state.value, np.diag([0.0, 0.5]))
    np.testing.assert_array_equal(state.jvp(tangent), tangent / 8)
    assert state.rank == 1
    assert state.retained == (False, True)
    # A frozen-projector implementation would return zero: detect that error.
    assert np.linalg.norm(state.jvp(tangent)) > 0.1


def test_repeated_eigenspace_gauge_invariance(monkeypatch: typing.Any) -> None:
    matrix = np.diag([0.02, 0.02, 2.0, 2.0])
    spec = SymmetricMatrixFunctionSpec(4, "gauge", 0.1)
    state = spec.prepare(matrix)
    seed = np.arange(16, dtype=float).reshape(4, 4)
    eigh = np.linalg.eigh
    angle = 0.41
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    gauge = np.zeros((4, 4))
    gauge[:2, :2] = rotation
    gauge[2:, 2:] = rotation.T

    def rotated_eigh(value: typing.Any) -> typing.Any:
        eigenvalues, eigenvectors = eigh(value)
        return eigenvalues, eigenvectors @ gauge

    monkeypatch.setattr(np.linalg, "eigh", rotated_eigh)
    rotated = spec.prepare(matrix)
    np.testing.assert_allclose(rotated.value, state.value, atol=1e-15, rtol=1e-14)
    np.testing.assert_allclose(
        rotated.vjp(seed), state.vjp(seed), atol=3e-15, rtol=3e-14
    )
    assert rotated.identity == state.identity


def test_rotation_covariance_and_scale_covariance() -> None:
    matrix, tangent, bar = _fixture([0.03, 0.04, 1.0, 3.0])
    spec = SymmetricMatrixFunctionSpec(4, "covariant", 0.1)
    state = spec.prepare(matrix)
    q, _ = np.linalg.qr(np.random.default_rng(5).normal(size=(4, 4)))
    rotated = spec.prepare(q @ matrix @ q.T)
    np.testing.assert_allclose(
        rotated.vjp(q @ bar @ q.T), q @ state.vjp(bar) @ q.T, atol=3e-14, rtol=3e-13
    )
    for scale in (1e-8, 1e8):
        scaled = spec.prepare(scale * matrix)
        np.testing.assert_allclose(
            scaled.value * np.sqrt(scale), state.value, atol=5e-14, rtol=1e-12
        )
        np.testing.assert_allclose(
            scaled.jvp(tangent) * scale**1.5, state.jvp(tangent), atol=5e-14, rtol=1e-12
        )
        assert scaled.rank == state.rank


def test_rank_change_and_cutoff_failure_and_state_rebinding() -> None:
    spec = SymmetricMatrixFunctionSpec(2, "branch", 0.1)
    state = spec.prepare(np.diag([0.05, 1.0]))
    with pytest.raises(ValueError, match="unresolved"):
        spec.prepare(np.diag([0.1, 1.0]))
    with pytest.raises(ValueError, match="rank changed"):
        state.rebind(np.diag([0.2, 1.0]))
    moved = state.rebind(np.array([[0.06, 0.01], [0.01, 1.0]]))
    assert moved.rank == state.rank
    assert moved.identity != state.identity
    assert abs(moved.value[0, 1]) > 0
    # A new, explicitly prepared branch is legal; no hidden global rank state.
    assert spec.prepare(np.diag([0.2, 1.0])).rank == 2


def test_manifest_and_generated_program_roundtrip() -> None:
    matrix, tangent, _ = _fixture([0.02, 1.0, 3.0])
    spec = SymmetricMatrixFunctionSpec(3, "roundtrip", 0.1)
    restored = SymmetricMatrixFunctionSpec.from_payload(
        json.loads(json.dumps(spec.to_payload()))
    )
    assert restored == spec
    assert restored.identity == spec.identity
    state = restored.prepare(matrix)
    program = spec.response_program()
    replayed = Program.loads(program.dumps())
    result = execute(
        replayed, {"vectors": state.vectors, "divided": state.divided, "seed": tangent}
    ).outputs["response"]
    np.testing.assert_array_equal(result, state.jvp(tangent))
    assert replayed.logical_hash == program.logical_hash
    assert json.loads(json.dumps(state.manifest))["rank"] == 2
    assert replace(spec, matrix_identity="other").identity != spec.identity
    assert replace(spec, relative_threshold=0.2).identity != spec.identity
    assert replace(spec, branch_guard=2e-12).identity != spec.identity
    assert all(len(node.spec.shape) <= 2 for node in program.live_nodes)


@pytest.mark.parametrize(
    "change",
    [
        {"unexpected": 3},
        {"version": "future"},
        {"function": "log"},
        {"dtype": "complex128"},
        {"inner_product": "packed"},
        {"derivative_orders": [1, 2]},
        {"derivative_orders": [True]},
        {"derivative_rule": "unknown"},
        {"kind": "other"},
    ],
)
def test_payload_rejects_unknown_or_mismatched_contract(
    change: typing.Any,
) -> None:
    payload = SymmetricMatrixFunctionSpec(2, "payload").to_payload()
    payload.update(change)
    with pytest.raises(ValueError):
        SymmetricMatrixFunctionSpec.from_payload(payload)


def test_missing_payload_field_rejected() -> None:
    payload = SymmetricMatrixFunctionSpec(2, "payload").to_payload()
    del payload["branch_guard"]
    with pytest.raises(ValueError, match="missing"):
        SymmetricMatrixFunctionSpec.from_payload(payload)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": True},
        {"size": 0},
        {"size": -1},
        {"size": 2.0},
        {"size": 2**31},
        {"size": 2**30},
        {"matrix_identity": ""},
        {"relative_threshold": 0.0},
        {"relative_threshold": 1.0},
        {"relative_threshold": np.nan},
        {"relative_threshold": True},
        {"relative_threshold": "0.1"},
        {"branch_guard": -1.0},
        {"branch_guard": np.inf},
        {"relative_threshold": 1e-13},
    ],
)
def test_invalid_spec(kwargs: typing.Any) -> None:
    with pytest.raises(ValueError):
        SymmetricMatrixFunctionSpec(
            **({"size": 2, "matrix_identity": "invalid"} | kwargs)
        )


@pytest.mark.parametrize(
    "matrix",
    [
        np.eye(2, dtype=np.float32),
        np.eye(2, dtype=complex),
        np.eye(2, dtype=int),
        np.ones((2, 3)),
        np.ones(2),
        np.array([[1.0, 0.1], [0.0, 2.0]]),
        np.diag([np.nan, 1.0]),
        np.diag([np.inf, 1.0]),
        np.diag([-1.0, 2.0]),
        np.zeros((2, 2)),
    ],
)
def test_invalid_matrix_fails_before_result(matrix: typing.Any) -> None:
    with pytest.raises(ValueError):
        SymmetricMatrixFunctionSpec(2, "bad", 0.1).prepare(matrix)


def test_full_rank_singular_and_unresolved_positive_input_rejected() -> None:
    spec = SymmetricMatrixFunctionSpec(2, "spd")
    with pytest.raises(ValueError, match="SPD"):
        spec.prepare(np.diag([0.0, 1.0]))
    with pytest.raises(ValueError, match="unresolved"):
        spec.prepare(np.diag([1e-14, 1.0]))
    with pytest.raises(ValueError, match="nonfinite"):
        SymmetricMatrixFunctionSpec(1, "overflow").prepare(np.array([[1e-300]]))


def test_workspace_preflight_before_eigensolve(monkeypatch: typing.Any) -> None:
    spec = SymmetricMatrixFunctionSpec(2, "budget")

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        raise AssertionError("must reject before factorization")

    monkeypatch.setattr(np.linalg, "eigh", forbidden)
    with pytest.raises(ValueError, match="budget"):
        spec.prepare(np.eye(2), max_bytes=spec.logical_workspace_bytes - 1)
    with pytest.raises(ValueError, match="expected rank"):
        spec.prepare(np.eye(2), expected_rank=True)
    with pytest.raises(ValueError, match="budget"):
        spec.prepare(np.eye(2), max_bytes=True)


def test_exact_logical_budget_and_detached_immutable_state() -> None:
    spec = SymmetricMatrixFunctionSpec(2, "snapshot")
    original = np.diag([1.0, 4.0])[::-1, ::-1]
    saved = original.copy()
    state = spec.prepare(original, max_bytes=spec.logical_workspace_bytes)
    original[0, 0] = 100.0
    np.testing.assert_allclose(state.value, _reference(saved, None))
    for array in (state.value, state.vectors, state.divided):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    seed = np.eye(2)
    seed.setflags(write=False)
    response = state.vjp(seed)
    response[:] = 99
    assert not np.any(state.vjp(seed) == 99)
    with pytest.raises(ValueError, match="symmetric"):
        state.jvp(np.array([[0.0, 1.0], [0.0, 0.0]]))
    with pytest.raises(ValueError, match="finite"):
        state.vjp(np.full((2, 2), np.nan))


def test_matches_existing_293_metric_response_without_runtime_loading() -> None:
    from tools.vibeqc_mp2.gradient import _inverse_sqrt_metric_response

    for spectrum in ([0.02, 0.05, 1.0, 3.0], [1.0, 2.0, 4.0]):
        matrix, _, bar = _fixture(spectrum)
        result = SymmetricMatrixFunctionSpec(len(spectrum), "ri-oracle", 0.1).prepare(
            matrix
        )
        oracle = _inverse_sqrt_metric_response(matrix, bar, 0.1)
        np.testing.assert_allclose(result.vjp(bar), oracle, atol=2e-14, rtol=3e-13)


def test_composes_with_generated_objective_vjp() -> None:
    """An ordinary #151 reverse graph supplies the custom matrix-function seed."""
    from vibeqc_compiler.tensor import (
        Index,
        IndexSpace,
        TensorSpec,
        einsum,
        input_tensor,
        transpose_program,
    )

    matrix, tangent, weights = _fixture([0.02, 1.0, 3.0])
    state = SymmetricMatrixFunctionSpec(3, "composed", 0.1).prepare(matrix)
    axis = IndexSpace("matrix", "ao", 3)
    indices = (Index("p", axis), Index("q", axis))
    x = input_tensor("x", TensorSpec(indices, role="parameter", differentiable=True))
    w = input_tensor("w", TensorSpec(indices, role="input", differentiable=False))
    objective = Program({"energy": einsum("pq,pq,pq->", w, x, x)})
    reverse = transpose_program(objective, ["energy"], inputs=["x"])
    bar_x = execute(
        reverse.program,
        {
            "x": state.value,
            "w": weights,
            "bar_energy": np.array(1.0),
        },
    ).outputs["bar_x"]
    analytic = np.vdot(state.vjp(bar_x), tangent)
    step = 1e-4
    plus = np.sum(weights * _reference(matrix + step * tangent, 0.1) ** 2)
    minus = np.sum(weights * _reference(matrix - step * tangent, 0.1) ** 2)
    np.testing.assert_allclose(
        analytic, (plus - minus) / (2 * step), atol=2e-8, rtol=2e-8
    )


def test_response_graph_can_be_planned_for_cuda_without_a_device() -> None:
    from vibeqc_compiler.integral.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    program = SymmetricMatrixFunctionSpec(3, "lowering", 0.1).response_program()
    plan = plan_cuda(program, cuda_target_info("sm_80"))
    assert plan.program is program
    assert plan.peak_bytes > 0
    # Planning is not GPU execution or a native eigensolver capability claim.


def _reference_pseudoinverse(matrix: typing.Any, threshold: typing.Any) -> typing.Any:
    """Independent spectral value used only as a finite-difference oracle."""
    values, vectors = np.linalg.eigh(matrix)
    keep = values > (0 if threshold is None else threshold * values[-1])
    return sum(
        np.outer(vectors[:, i], vectors[:, i]) / values[i]
        for i in range(len(values))
        if keep[i]
    )


def test_pseudoinverse_rule_value_vjp_and_fixed_rank_finite_difference() -> None:
    matrix, tangent, bar = _fixture([0.02, 0.05, 1.0, 3.0])
    spec = SymmetricMatrixFunctionSpec(
        4, "df-metric-pseudoinverse", 0.1, function="pseudoinverse"
    )
    state = spec.prepare(matrix)
    np.testing.assert_allclose(
        state.value, _reference_pseudoinverse(matrix, 0.1), atol=2e-14, rtol=3e-13
    )
    analytic = state.jvp(tangent)
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        finite = (
            _reference_pseudoinverse(matrix + step * tangent, 0.1)
            - _reference_pseudoinverse(matrix - step * tangent, 0.1)
        ) / (2 * step)
        errors.append(np.max(np.abs(finite - analytic)))
    assert errors[-1] < 3e-8
    assert errors[-1] < errors[0] * 0.05
    np.testing.assert_allclose(
        np.vdot(analytic, bar),
        np.vdot(tangent, state.vjp(bar)),
        atol=4e-14,
        rtol=4e-13,
    )
    assert state.rank == 2
    assert spec.to_payload()["derivative_rule"] == "pseudoinverse-frechet-v1"


def test_pseudoinverse_full_rank_closed_form() -> None:
    matrix, _, bar = _fixture([1.0, 2.0, 5.0])
    state = SymmetricMatrixFunctionSpec(
        3, "full-rank-pseudoinverse", function="pseudoinverse"
    ).prepare(matrix)
    inverse = np.linalg.inv(matrix)
    symmetric_bar = (bar + bar.T) / 2
    np.testing.assert_allclose(state.value, inverse, atol=2e-15, rtol=3e-14)
    np.testing.assert_allclose(
        state.vjp(bar),
        -inverse @ symmetric_bar @ inverse,
        atol=3e-14,
        rtol=4e-13,
    )


def test_pseudoinverse_cuda_lowering_carries_custom_rule_identity() -> None:
    source = emit_pseudoinverse_vjp_cuda()
    assert "custom-rule: pseudoinverse-frechet-v1" in source
    assert "launch_symmetric_pseudoinverse_vjp" in source
    assert "li > cutoff" in source and "lj > cutoff" in source


def test_nonobject_payload_is_a_type_error() -> None:
    with pytest.raises(TypeError, match="object"):
        SymmetricMatrixFunctionSpec.from_payload([])


def test_nonfinite_eigensystem_rejected(monkeypatch: typing.Any) -> None:
    monkeypatch.setattr(
        np.linalg, "eigh", lambda matrix: (np.array([np.nan, 1.0]), np.eye(2))
    )
    with pytest.raises(ValueError, match="eigensystem"):
        SymmetricMatrixFunctionSpec(2, "bad-eigh").prepare(np.eye(2))
