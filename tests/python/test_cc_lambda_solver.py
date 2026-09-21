"""Converged Lambda, independent small Jacobians and fail-closed state binding."""

import typing
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from test_cc_solver import fixture_problem
from vibeqc_compiler.tensor import execute

from tools.cc_endpoint_fixtures import load, source_arguments
from tools.vibeqc_cc import (
    BoundCCSDLambda,
    LambdaOptions,
    SolverOptions,
    solve,
)
from tools.vibeqc_cc import lambda_solver as consumer
from tools.vibeqc_cc.oracle import DeterminantOracle
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response.implicit import ImplicitSolveError, ResponseGMRES
from tools.vibeqc_response.krylov import GMRESOptions
from tools.vibeqc_response.problem import ResponseCompatibilityError


@pytest.fixture(scope="module", params=("h2", "h2o", "nh3"))
def molecular_state(request: typing.Any) -> typing.Any:
    snapshot, provider, _meta, arrays = fixture_problem(request.param)
    cc = solve(
        snapshot,
        provider,
        options=SolverOptions(
            residual_tolerance=1e-11,
            energy_tolerance=1e-13,
        ),
    )
    assert cc.converged, cc.reason
    bound = BoundCCSDLambda(snapshot, cc)
    result = bound.solve(reference_identity=snapshot.identity)
    return request.param, snapshot, cc, arrays, bound, result


@pytest.fixture(scope="module")
def small_state() -> typing.Any:
    snapshot, provider, _meta, arrays = fixture_problem("h2")
    cc = solve(snapshot, provider)
    assert cc.converged
    return snapshot, cc, arrays


def _numerical_multiplier(
    bound: typing.Any, evaluate: typing.Any, step: typing.Any = 1e-5
) -> typing.Any:
    """Test-only dense Jacobian in independent coordinates, not solver code."""
    point = bound._pack((bound.feeds["t1"], bound.feeds["t2"]))
    jacobian = np.empty((point.size, point.size))
    gradient = np.empty(point.size)
    for column in range(point.size):
        perturbation = np.zeros_like(point)
        perturbation[column] = step
        upper = evaluate(*bound._unpack(point + perturbation))
        lower = evaluate(*bound._unpack(point - perturbation))
        jacobian[:, column] = (bound._pack(upper[1:]) - bound._pack(lower[1:])) / (
            2 * step
        )
        gradient[column] = (upper[0] - lower[0]) / (2 * step)
    return np.linalg.solve(jacobian.T, -gradient) / bound.sqrt_weights**2


def test_native_cpu_lambda_does_not_fall_back_to_interpreter(
    small_state: typing.Any,
    tmp_path: typing.Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot, cc, _arrays = small_state
    monkeypatch.setenv("VIBEQC_TENSOR_CACHE", str(tmp_path))

    def forbidden(*_args: typing.Any, **_kwargs: typing.Any) -> typing.NoReturn:
        raise AssertionError("native Lambda path called the NumPy TensorIR interpreter")

    monkeypatch.setattr(consumer, "execute", forbidden)
    bound = BoundCCSDLambda(snapshot, cc, backend="native-cpu")
    result = bound.solve(reference_identity=snapshot.identity)
    assert result.converged
    assert result.provenance["tensor_backend"] == "native-cpu-tensorir"
    assert bound.tensor_executor is not None
    assert bound.tensor_executor.compiled_program_count >= 3


def test_molecular_lambda_matches_numerical_transpose(
    molecular_state: typing.Any,
) -> None:
    name, snapshot, cc, arrays, bound, result = molecular_state
    assert result.converged and result.status == "converged"
    assert result.reference_identity == snapshot.identity
    assert result.cc_state_identity == bound.cc_state_identity
    assert result.scf_residual == snapshot.scf_residual
    assert max(result.cc_r1_max, result.cc_r2_max) <= 1e-9
    assert result.lambda_residual_norm <= 1e-11
    assert result.independent_lambda_residual_norm <= 1.1e-11
    assert result.independent_lambda_residual_max <= 1e-11
    assert result.operator_actions >= result.iterations + 1
    assert result.provenance["reference_binding"] == "detached-immutable-snapshot"
    assert result.provenance["tensor_backend"] == "numpy-cpu-interpreter"
    assert result.provenance["solver_backend"] == "python-response-gmres"
    assert result.logical_reserved_host_bytes <= bound.options.max_bytes
    if name == "h2":
        fock = snapshot.coefficients.T @ snapshot.fock @ snapshot.coefficients
        oracle = DeterminantOracle(fock, arrays["g"], snapshot.nocc)
        evaluate = oracle.evaluate_full
    else:
        # Separately expanded physical equations at many displaced amplitudes;
        # numerical Jacobians here are deliberately restricted to tiny fixtures.
        def evaluate(t1: typing.Any, t2: typing.Any) -> typing.Any:
            out = execute(
                bound.independent.primal, {**bound.feeds, "t1": t1, "t2": t2}
            ).outputs
            return (
                float(out["correlation_energy"]),
                out["singles_residual"],
                out["doubles_residual"],
            )

    expected = _numerical_multiplier(bound, evaluate)
    actual = bound._pack((result.lambda1, result.lambda2))
    np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)
    if name != "h2":
        assert np.max(np.abs(actual - expected * bound.sqrt_weights**2)) > 1e-4
    np.testing.assert_array_equal(result.lambda2, result.lambda2.transpose(1, 0, 3, 2))
    # The response must not change the converged primal state.
    np.testing.assert_array_equal(cc.t1, bound.feeds["t1"])
    np.testing.assert_array_equal(cc.t2, bound.feeds["t2"])


def test_lagrangian_is_stationary_at_three_steps(
    molecular_state: typing.Any,
) -> None:
    _name, _snapshot, _cc, _arrays, bound, result = molecular_state
    direction = np.random.default_rng(152).normal(size=len(bound.sqrt_weights))
    direction /= np.linalg.norm(direction)
    d1, d2 = bound._unpack(direction)

    def lagrangian(step: typing.Any) -> typing.Any:
        out = execute(
            bound.independent.primal,
            {
                **bound.feeds,
                "t1": bound.feeds["t1"] + step * d1,
                "t2": bound.feeds["t2"] + step * d2,
            },
        ).outputs
        return (
            float(out["correlation_energy"])
            + np.sum(result.lambda1 * out["singles_residual"])
            + np.sum(result.lambda2 * out["doubles_residual"])
        )

    for step in (1e-4, 3e-5, 1e-5):
        assert abs((lagrangian(step) - lagrangian(-step)) / (2 * step)) < 2e-8


def test_bound_state_and_result_are_immutable_and_detached(
    small_state: typing.Any,
) -> None:
    snapshot, cc, _ = small_state
    copied = replace(
        cc, replay_inputs=deepcopy(cc.replay_inputs), provenance=dict(cc.provenance)
    )
    bound = BoundCCSDLambda(snapshot, copied)
    first = bound.solve(reference_identity=snapshot.identity)
    copied.replay_inputs["ovov"][0][0][0][0] += 1
    copied.provenance["reference_id"] = "changed-after-binding"
    second = bound.solve(reference_identity=snapshot.identity)
    np.testing.assert_array_equal(first.lambda1, second.lambda1)
    np.testing.assert_array_equal(first.lambda2, second.lambda2)
    for array in (
        *bound.feeds.values(),
        bound.sqrt_weights,
        first.lambda1,
        first.lambda2,
    ):
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(TypeError):
        bound.feeds["t1"] = np.zeros_like(cc.t1)
    with pytest.raises(TypeError):
        bound._solver_contract["atol"] = 1.0
    with pytest.raises(FrozenInstanceError):
        bound.reference_identity = "changed"
    with pytest.raises(TypeError):
        first.provenance["tensor_backend"] = "cuda"


@pytest.mark.parametrize(
    "field",
    [
        "reference_id",
        "hamiltonian_id",
        "equation_hash",
        "independent_equation_hash",
        "integral_hash",
    ],
)
def test_rejects_changed_provenance(small_state: typing.Any, field: typing.Any) -> None:
    snapshot, cc, _ = small_state
    with pytest.raises(ResponseCompatibilityError):
        BoundCCSDLambda(
            snapshot, replace(cc, provenance={**cc.provenance, field: "stale"})
        )


@pytest.mark.parametrize(
    "kind", ["reference", "fock", "integral", "missing", "nonfinite"]
)
def test_rejects_corrupt_replay_inputs(
    small_state: typing.Any, kind: typing.Any
) -> None:
    snapshot, cc, _ = small_state
    replay = deepcopy(cc.replay_inputs)
    if kind == "reference":
        replay["snapshot"]["generation_id"] = "another-reference"
    elif kind == "fock":
        replay["foo"][0][0] += 1e-3
    elif kind == "integral":
        replay["ovov"][0][0][0][0] += 1e-3
    elif kind == "missing":
        del replay["ovov"]
    else:
        replay["ovov"][0][0][0][0] = float("nan")
    with pytest.raises((ResponseCompatibilityError, ValueError)):
        BoundCCSDLambda(snapshot, replace(cc, replay_inputs=replay))


def test_fock_reference_comparison_even_with_recomputed_input_hash(
    small_state: typing.Any,
) -> None:
    snapshot, cc, _ = small_state
    replay = deepcopy(cc.replay_inputs)
    replay["foo"][0][0] += 0.01
    feeds = {
        name: np.asarray(replay[name])
        for name in (
            "foo",
            "fov",
            "fvv",
            "ovov",
            "ovvo",
            "oovv",
            "ovvv",
            "ovoo",
            "oooo",
            "vvvv",
        )
    }
    modified = replace(
        cc,
        replay_inputs=replay,
        provenance={
            **cc.provenance,
            "integral_hash": consumer._feed_hash(feeds),
        },
    )
    with pytest.raises(ResponseCompatibilityError, match="Fock feeds"):
        BoundCCSDLambda(snapshot, modified)


@pytest.mark.parametrize(
    "kind", ["nonconverged", "nonfinite", "energy", "false_root", "dtype", "shape"]
)
def test_invalid_primal_cannot_publish_response(
    small_state: typing.Any, kind: typing.Any
) -> None:
    snapshot, cc, _ = small_state
    if kind == "nonconverged":
        cc = replace(cc, status="not_converged")
    elif kind == "nonfinite":
        cc = replace(cc, t1=cc.t1 * np.nan)
    elif kind == "energy":
        cc = replace(cc, correlation_energy=cc.correlation_energy + 0.1)
    elif kind == "dtype":
        cc = replace(cc, t1=cc.t1.astype(np.float32))
    elif kind == "shape":
        cc = replace(cc, t1=cc.t1.reshape(-1))
    else:
        probe = BoundCCSDLambda(snapshot, cc)
        t1 = cc.t1 + 0.01
        e = float(
            execute(probe.independent.primal, {**probe.feeds, "t1": t1}).outputs[
                "correlation_energy"
            ]
        )
        # Forge a consistent energy and claimed convergence, not the root.
        cc = replace(
            cc, t1=t1, correlation_energy=e, total_energy=snapshot.reference_energy + e
        )
    with pytest.raises((ImplicitSolveError, ValueError)):
        BoundCCSDLambda(snapshot, cc)


def test_unconverged_or_noncanonical_scf_rejected(
    small_state: typing.Any,
) -> None:
    snapshot, cc, _ = small_state
    for changes in (
        {"converged": False},
        {"scf_residual": 1e-3},
        {"fock": snapshot.fock + 0.01},
    ):
        with pytest.raises(ValueError):
            BoundCCSDLambda(replace(snapshot, **changes), cc)


def test_dense_t2_symmetry_is_not_silently_projected(
    molecular_state: typing.Any,
) -> None:
    name, snapshot, cc, _arrays, _bound, _result = molecular_state
    if name == "h2":
        return  # no distinct simultaneous-pair orbit
    t2 = np.array(cc.t2)
    t2[0, 1, 0, 1] += 1e-3
    with pytest.raises(ValueError, match="symmetry"):
        BoundCCSDLambda(snapshot, replace(cc, t2=t2))


def test_stale_live_generation_and_explicit_identity(
    small_state: typing.Any,
) -> None:
    snapshot, cc, _ = small_state
    current = [snapshot.identity]
    bound = BoundCCSDLambda(snapshot, cc, current_reference=lambda: current[0])
    result = bound.solve(reference_identity=snapshot.identity)
    assert result.provenance["reference_binding"] == "live-reference-callback"
    with pytest.raises(ResponseCompatibilityError):
        bound.solve(reference_identity="different-geometry")
    current[0] = "new-generation"
    with pytest.raises(ResponseCompatibilityError):
        bound.solve(reference_identity=snapshot.identity)
    with pytest.raises(ResponseCompatibilityError):
        BoundCCSDLambda(snapshot, cc, current_reference=lambda: current[0])


def test_combined_budget_rejects_before_interpreter(
    small_state: typing.Any, monkeypatch: typing.Any
) -> None:
    snapshot, cc, _ = small_state
    bound = BoundCCSDLambda(snapshot, cc)
    exact = BoundCCSDLambda(
        snapshot, cc, options=LambdaOptions(max_bytes=bound.logical_reserved_host_bytes)
    )
    exact.solve(reference_identity=snapshot.identity)
    monkeypatch.setattr(
        consumer, "execute", lambda *a, **k: pytest.fail("execution before admission")
    )
    with pytest.raises(ImplicitSolveError, match="workspace budget exceeded"):
        BoundCCSDLambda(
            snapshot,
            cc,
            options=LambdaOptions(max_bytes=bound.logical_reserved_host_bytes - 1),
        )


class AdversarialSolver:
    """Exercise the public opaque solver contract rather than patch CC math."""

    backend = "adversarial-test-only"

    def __init__(
        self, dimension: typing.Any, mode: typing.Any, current: typing.Any = None
    ) -> None:
        self.delegate = ResponseGMRES(dimension, GMRESOptions(rtol=0, atol=1e-11))
        self.mode, self.current = mode, current
        self.rtol = 0.0
        self.atol = 1.0 if mode == "loose" else 1e-11
        self.identity = "adversarial-" + mode
        self.workspace_bytes = self.delegate.workspace_bytes

    def solve(self, operator: typing.Any, rhs: typing.Any) -> typing.Any:
        result = self.delegate.solve(operator, rhs)
        if self.mode == "false_success" or self.mode == "loose":
            return replace(
                result, solution=np.zeros_like(result.solution), residual_norm=0.0
            )
        if self.mode == "nonfinite":
            return replace(result, solution=np.full_like(result.solution, np.nan))
        if self.mode == "workspace":
            return replace(result, workspace_bytes=self.workspace_bytes + 1)
        if self.mode == "status":
            return replace(result, converged=False, reason="injected_stagnation")
        if self.mode == "stale":
            self.current[0] = "changed-during-solve"
        if self.mode == "identity":
            self.identity += "-changed"
        if self.mode == "iterations":
            return replace(result, iterations=-1)
        return result


@pytest.mark.parametrize(
    "mode",
    [
        "false_success",
        "nonfinite",
        "workspace",
        "status",
        "stale",
        "identity",
        "loose",
        "iterations",
    ],
)
def test_adversarial_solver_cannot_publish_lambda(
    small_state: typing.Any, mode: typing.Any
) -> None:
    snapshot, cc, _ = small_state
    current = [snapshot.identity]
    solver = AdversarialSolver(2, mode, current)
    bound = BoundCCSDLambda(
        snapshot, cc, solver=solver, current_reference=lambda: current[0]
    )
    expected = "independent physical residual" if mode == "loose" else None
    with pytest.raises(
        (ImplicitSolveError, ResponseCompatibilityError, ValueError), match=expected
    ):
        bound.solve(reference_identity=snapshot.identity)


def test_real_solver_workspace_failure(small_state: typing.Any) -> None:
    snapshot, cc, _ = small_state
    options = LambdaOptions(gmres=GMRESOptions(max_workspace_bytes=1))
    bound = BoundCCSDLambda(snapshot, cc, options=options)
    with pytest.raises(ImplicitSolveError, match="workspace_limit"):
        bound.solve(reference_identity=snapshot.identity)


def test_real_solver_nonconvergence_is_separate_from_cc(
    molecular_state: typing.Any,
) -> None:
    name, snapshot, cc, _arrays, _bound, _result = molecular_state
    if name == "h2":
        return
    bound = BoundCCSDLambda(
        snapshot,
        cc,
        options=LambdaOptions(
            gmres=GMRESOptions(rtol=0, atol=1e-12, max_iterations=1),
        ),
    )
    assert cc.converged
    with pytest.raises(ImplicitSolveError, match="adjoint failed"):
        bound.solve(reference_identity=snapshot.identity)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cc_tolerance": 1e-8},
        {"lambda_tolerance": 0.0},
        {"lambda_tolerance": float("nan")},
        {"max_bytes": -1},
        {"max_bytes": True},
        {"gmres": object()},
    ],
)
def test_invalid_options(kwargs: typing.Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        LambdaOptions(**kwargs)


def test_explicit_unsupported_backend_and_bad_types(
    small_state: typing.Any,
) -> None:
    snapshot, cc, _ = small_state
    with pytest.raises(NotImplementedError, match="CPU tooling only"):
        BoundCCSDLambda(snapshot, cc, backend="cuda")
    for first, second in ((object(), cc), (snapshot, object())):
        with pytest.raises(TypeError):
            BoundCCSDLambda(first, second)
    with pytest.raises(TypeError):
        BoundCCSDLambda(snapshot, cc, current_reference="not callable")
    with pytest.raises(TypeError):
        BoundCCSDLambda(snapshot, cc, options=object())


def test_generated_graphs_do_not_contain_dense_coordinate_maps(
    molecular_state: typing.Any,
) -> None:
    _name, _snapshot, _cc, _arrays, bound, _result = molecular_state
    for bundle in (bound.programs, bound.independent):
        for program in (bundle.energy_vjp.program, bundle.residual_vjp.program):
            assert not any(n.op in ("constant", "gather") for n in program.live_nodes)
            assert all(len(n.spec.indices) <= 4 for n in program.live_nodes)


@pytest.mark.parametrize(
    "name,shift", [("h2", 0.0), ("h2", 0.15), ("h2o", 0.0), ("h4", 0.0)]
)
def test_native_hf_cc_lambda_complete_small_endpoint(
    name: typing.Any, shift: typing.Any
) -> None:
    meta, _ = load("h2" if name == "h4" else name)
    inputs = deepcopy(meta["inputs"])
    inputs["coordinates"][1][2] += shift
    if name == "h4":
        # Two weakly coupled H2 molecules: four orbitals, nontrivial T2
        # multiplicities, and a small (70-determinant) independent oracle.
        inputs["atomic_numbers"] *= 2
        inputs["coordinates"] += [
            [position[0] + 5.0, position[1] + 0.3, position[2] + 0.2]
            for position in inputs["coordinates"]
        ]
        copies = deepcopy(inputs["shells"])
        for shell in copies:
            shell["atom_index"] += 2
        inputs["shells"] += copies
    try:
        source = NativeSource(**source_arguments(inputs))
    except (OSError, FileNotFoundError) as error:
        pytest.skip(str(error))
    except RuntimeError as error:
        if "native library was not found" not in str(error):
            raise
        pytest.skip(str(error))
    with source:
        snapshot, _hf = export_rhf(source, tolerance=1e-12, max_iterations=150)
        assert snapshot.hf_backend == "native-cpu"
        with ConventionalProvider(snapshot, source) as provider:

            def current() -> typing.Any:
                source._check_open()
                if provider._closed:
                    raise ResponseCompatibilityError("integral provider closed")
                return provider.snapshot.identity

            cc = solve(
                snapshot,
                provider,
                options=SolverOptions(
                    residual_tolerance=1e-11,
                    energy_tolerance=1e-13,
                ),
            )
            assert cc.converged
            bound = BoundCCSDLambda(snapshot, cc, current_reference=current)
            result = bound.solve(reference_identity=snapshot.identity)
            assert max(result.cc_r1_max, result.cc_r2_max) <= 1e-11
            assert result.independent_lambda_residual_norm <= 1.1e-11
            assert result.reference_identity == cc.provenance["reference_id"]
            if name in ("h2", "h4"):
                # Fresh native AO/MO inputs, but an independent determinant
                # representation for the final response equation check.

                from tools.vibeqc_posthf import MOBlock

                # Full tiny MO tensor is a test-only oracle input.
                block = MOBlock(tuple(tuple(range(snapshot.nmo)) for _ in range(4)))
                g = provider.get(block).to_host()
                f = snapshot.coefficients.T @ snapshot.fock @ snapshot.coefficients
                expected = _numerical_multiplier(
                    bound, DeterminantOracle(f, g, snapshot.nocc).evaluate_full
                )
                np.testing.assert_allclose(
                    bound._pack((result.lambda1, result.lambda2)),
                    expected,
                    atol=1e-8,
                    rtol=1e-8,
                )
        with pytest.raises(ResponseCompatibilityError, match="provider closed"):
            bound.solve(reference_identity=snapshot.identity)


def test_tensor_backend_switch_is_not_silent_fallback(
    small_state: typing.Any, monkeypatch: typing.Any
) -> None:
    snapshot, cc, _ = small_state
    original = consumer.execute

    def switched(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        return replace(original(*args, **kwargs), backend="unexpected-backend")

    monkeypatch.setattr(consumer, "execute", switched)
    with pytest.raises(ResponseCompatibilityError, match="backend changed"):
        BoundCCSDLambda(snapshot, cc)


def test_nonfinite_transpose_output_cannot_reach_solver(
    small_state: typing.Any, monkeypatch: typing.Any
) -> None:
    snapshot, cc, _ = small_state
    bound = BoundCCSDLambda(snapshot, cc)
    original = consumer.execute

    def corrupted(
        program: typing.Any, *args: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        result = original(program, *args, **kwargs)
        if program is bound.programs.residual_vjp.program:
            return replace(
                result,
                outputs={
                    **result.outputs,
                    "bar_t1": np.full_like(result.outputs["bar_t1"], np.nan),
                },
            )
        return result

    monkeypatch.setattr(consumer, "execute", corrupted)
    with pytest.raises(ValueError, match="finite"):
        bound.solve(reference_identity=snapshot.identity)


def test_a_changed_reference_cannot_reuse_old_converged_cc(
    small_state: typing.Any,
) -> None:
    snapshot, cc, _ = small_state
    other = replace(snapshot, generation_id="same-shape-different-generation")
    with pytest.raises(ResponseCompatibilityError, match="reference/Hamiltonian"):
        BoundCCSDLambda(other, cc)
