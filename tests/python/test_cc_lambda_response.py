"""Generated correlation input weights, re-solved differences and state gates."""

import typing
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from unittest.mock import patch

import numpy as np
import pytest
from test_cc_solver import fixture_problem
from vibeqc_compiler.tensor import PackedLayout, Program, execute

from tools.cc_endpoint_fixtures import load, source_arguments
from tools.vibeqc_cc import (
    BoundCCSDLambda,
    BoundCCSDResponse,
    PreparedCCSD,
    SolverOptions,
    solve,
)
from tools.vibeqc_cc import lambda_response as response_module
from tools.vibeqc_cc import solver as solver_module
from tools.vibeqc_cc.lambda_equations import (
    PARAMETERS,
    build_lambda_programs,
    build_parameter_vjp,
)
from tools.vibeqc_cc.oracle import DeterminantOracle, dense_feeds, random_case
from tools.vibeqc_posthf import MOBlock
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response.implicit import ImplicitSolveError
from tools.vibeqc_response.problem import ResponseCompatibilityError

_STRICT = SolverOptions(
    max_iterations=120, residual_tolerance=1e-12, energy_tolerance=1e-14
)


@lru_cache(maxsize=4)
def _state(name: typing.Any = "h2") -> typing.Any:
    snapshot, provider, _, arrays = fixture_problem(name)
    cc = solve(snapshot, provider, options=_STRICT)
    assert cc.converged, cc.reason
    bound = BoundCCSDLambda(snapshot, cc)
    lam = bound.solve(reference_identity=snapshot.identity)
    response = BoundCCSDResponse(bound, lam)
    return snapshot, provider, cc, bound, lam, response, arrays


def _direction(spec: typing.Any, seed: typing.Any = 152) -> typing.Any:
    layout = PackedLayout.from_spec(spec)  # tiny TEST-ONLY coordinate map
    direction = layout.unpack(np.random.default_rng(seed).normal(size=layout.size))
    return direction / np.linalg.norm(direction)


def _resolved_correlation(
    snapshot: typing.Any, provider: typing.Any, cc: typing.Any, changes: typing.Any
) -> typing.Any:
    """Test-only fixed-mathematical-input seam in the EXISTING CC solver.

    A displaced noncanonical F is deliberately NOT exported as a converged
    RHF ReferenceSnapshot. Replace only the prepared equation feeds, retain
    the original preconditioner and re-solve the physical CC equations. These
    synthetic result metadata are never used to bind a response/native state.
    This tests the q boundary without weakening the production reference gate.
    """
    prepared = PreparedCCSD(snapshot, provider, _STRICT, t1=cc.t1, t2=cc.t2)
    for name, change in changes.items():
        prepared.feeds[name] = prepared.feeds[name] + change
    with patch.object(solver_module, "PreparedCCSD", return_value=prepared):
        result = solver_module.solve(snapshot, provider, options=_STRICT)
    assert result.converged, (result.reason, result.history[-1])
    assert (
        max(
            result.history[-1]["independent_r1_max"],
            result.history[-1]["independent_r2_max"],
        )
        <= 1e-12
    )
    return result.correlation_energy


@pytest.mark.parametrize("parameter", PARAMETERS)
def test_all_water_blocks_against_reconverged_three_step_differences(
    parameter: typing.Any,
) -> None:
    s, provider, cc, bound, lam, response, _ = _state("h2o")
    weight = response.weight(parameter, reference_identity=s.identity)
    direction = _direction(weight.spec)
    expected = weight.contract(direction)
    # The direct energy partial is not the amplitude-relaxed derivative.
    reverse = build_parameter_vjp(bound.programs.primal, parameter)
    direct = execute(
        reverse.program,
        {
            **bound.feeds,
            "bar_correlation_energy": np.asarray(1.0),
            "bar_singles_residual": np.zeros_like(lam.lambda1),
            "bar_doubles_residual": np.zeros_like(lam.lambda2),
        },
    ).outputs["bar_" + parameter]
    direct_contraction = float(np.sum(direct * direction))
    for h in (1e-4, 3e-5, 1e-5):
        energies = [
            _resolved_correlation(s, provider, cc, {parameter: sign * h * direction})
            for sign in (-1, 1)
        ]
        derivative = (energies[1] - energies[0]) / (2 * h)
        np.testing.assert_allclose(derivative, expected, atol=3e-8, rtol=2e-6)
        fixed = [
            float(
                execute(
                    bound.independent.primal,
                    {
                        **bound.feeds,
                        parameter: bound.feeds[parameter] + sign * h * direction,
                    },
                ).outputs["correlation_energy"]
            )
            for sign in (-1, 1)
        ]
        np.testing.assert_allclose(
            (fixed[1] - fixed[0]) / (2 * h), direct_contraction, atol=3e-10, rtol=1e-8
        )
    if parameter in ("foo", "fov", "ovov"):
        assert abs(expected - direct_contraction) > 1e-4
    assert weight.independent_max_abs_error <= 1e-12
    assert weight.provenance["hf_reference_energy"] == "excluded"
    assert weight.provenance["normal_ordering_pullback"] == "excluded"
    assert weight.provenance["orbital_response"] == "excluded"
    assert weight.provenance["physical_rdm"] is False


@pytest.mark.parametrize("name", ("h2", "h2o", "nh3"))
def test_diagonal_fock_shift_invariance_includes_diagonal_dependencies(
    name: typing.Any,
) -> None:
    s, _, _, _, _, response, _ = _state(name)
    occupied = response.weight("foo", reference_identity=s.identity)
    virtual = response.weight("fvv", reference_identity=s.identity)
    np.testing.assert_allclose(
        occupied.contract(np.eye(s.nocc)) + virtual.contract(np.eye(s.nmo - s.nocc)),
        0,
        atol=1e-12,
        rtol=0,
    )
    assert np.linalg.norm(occupied.values) > 1e-4


@pytest.mark.parametrize("o,v", [(1, 2), (2, 2)])
def test_generated_full_input_directions_against_independent_determinant(
    o: typing.Any, v: typing.Any
) -> None:
    f, g, t1, t2 = random_case(o, v, 152)
    df, dg, l1, l2 = random_case(o, v, 153)
    df /= np.linalg.norm(df)
    dg /= np.linalg.norm(dg)
    feeds = dense_feeds(f, g, t1, t2)
    directions = dense_feeds(df, dg, t1 * 0, t2 * 0)
    programs = build_lambda_programs(o, v)
    analytic = 0.0
    for parameter in PARAMETERS:
        reverse = build_parameter_vjp(programs.primal, parameter)
        assert set(reverse.program.outputs) == {"bar_" + parameter}
        assert reverse.input_names == (parameter,)
        assert not any(n.op == "gather" for n in reverse.program.live_nodes)
        assert all(len(n.spec.indices) <= 4 for n in reverse.program.live_nodes)
        restored = Program.from_payload(reverse.program.to_payload())
        assert restored.logical_hash == reverse.program.logical_hash
        seeds = {
            **feeds,
            "bar_correlation_energy": np.asarray(1.0),
            "bar_singles_residual": l1,
            "bar_doubles_residual": l2,
        }
        original = execute(reverse.program, seeds).outputs["bar_" + parameter]
        replay = execute(restored, seeds).outputs["bar_" + parameter]
        np.testing.assert_array_equal(original, replay)
        analytic += np.sum(original * directions[parameter])
    for h in (1e-4, 3e-5, 1e-5):
        values = []
        for sign in (-1, 1):
            e, r1, r2 = DeterminantOracle(
                f + sign * h * df, g + sign * h * dg, o
            ).evaluate_full(t1, t2)
            values.append(e + np.sum(l1 * r1) + np.sum(l2 * r2))
        np.testing.assert_allclose(
            (values[1] - values[0]) / (2 * h), analytic, atol=3e-9, rtol=2e-8
        )


def test_parameter_generator_selection_and_identity() -> None:
    programs = build_lambda_programs(1, 1)
    first = build_parameter_vjp(programs.primal, "foo")
    second = build_parameter_vjp(programs.primal, "fvv")
    assert first.derivative_hash != second.derivative_hash
    assert (
        first.derivative_hash
        == build_parameter_vjp(programs.primal, "foo").derivative_hash
    )
    for name in ("t1", "t2", "full_rdm", "fvo", None, 1):
        with pytest.raises(ValueError):
            build_parameter_vjp(programs.primal, name)
    with pytest.raises(TypeError):
        build_parameter_vjp(object(), "foo")
    with pytest.raises(ValueError, match="exactly"):
        build_parameter_vjp(
            Program({"energy": programs.primal.outputs["correlation_energy"]}), "foo"
        )


def test_streamed_blocks_are_immutable_and_do_not_reinvoke_solver(
    monkeypatch: typing.Any,
) -> None:
    s, _, _, bound, lam, response, _ = _state()

    def unexpected(*a: typing.Any, **kw: typing.Any) -> None:
        pytest.fail("CC weight generation must not re-solve the adjoint")

    monkeypatch.setattr(type(bound.solver), "solve", unexpected)
    seen = []
    for weight in response.iter_weights(reference_identity=s.identity):
        seen.append(weight.parameter)
        assert weight.response_identity == response.response_identity
        assert weight.lambda_identity == response.lambda_identity
        assert weight.reference_identity == s.identity
        assert weight.cc_state_identity == bound.cc_state_identity
        assert weight.logical_reserved_host_bytes <= response.max_bytes
        assert not weight.values.flags.writeable
        with pytest.raises(ValueError):
            weight.values.setflags(write=True)
        with pytest.raises(TypeError):
            weight.provenance["physical_rdm"] = True
        with pytest.raises(FrozenInstanceError):
            weight.parameter = "different"
    assert tuple(seen) == response.parameters
    np.testing.assert_array_equal(response.lambda1, lam.lambda1)
    np.testing.assert_array_equal(response.lambda2, lam.lambda2)
    for value in (response.lambda1, response.lambda2):
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        response.max_bytes = 1


def test_multiplier_input_is_copied_before_use() -> None:
    s, _, _, bound, lam, _, _ = _state()
    copied = replace(
        lam,
        lambda1=np.array(lam.lambda1),
        lambda2=np.array(lam.lambda2),
        provenance=dict(lam.provenance),
    )
    response = BoundCCSDResponse(bound, copied)
    first = response.weight("foo", reference_identity=s.identity)
    copied.lambda2[:] = 1e50
    copied.provenance["lagrangian"] = "changed"
    np.testing.assert_array_equal(
        response.weight("foo", reference_identity=s.identity).values, first.values
    )


@pytest.mark.parametrize(
    "kind",
    (
        "reference",
        "state",
        "equation",
        "sign",
        "backend",
        "zero_lambda",
        "nonfinite",
        "dtype",
        "shape",
    ),
)
def test_false_lambda_reports_cannot_publish_weights(kind: typing.Any) -> None:
    _, _, _, bound, lam, _, _ = _state()
    if kind == "reference":
        lam = replace(lam, reference_identity="other-reference")
    elif kind == "state":
        lam = replace(lam, cc_state_identity="other-amplitudes")
    elif kind in ("equation", "sign", "backend"):
        field = {
            "equation": "equation_identity",
            "sign": "lagrangian",
            "backend": "tensor_backend",
        }[kind]
        lam = replace(lam, provenance={**lam.provenance, field: "wrong"})
    elif kind == "zero_lambda":
        lam = replace(
            lam,
            lambda2=np.zeros_like(lam.lambda2),
            lambda_residual_norm=0.0,
            independent_lambda_residual_norm=0.0,
        )
    elif kind == "nonfinite":
        lam = replace(lam, lambda2=np.full_like(lam.lambda2, np.nan))
    elif kind == "dtype":
        lam = replace(lam, lambda2=lam.lambda2.astype(np.float32))
    else:
        lam = replace(lam, lambda2=lam.lambda2.reshape(-1))
    with pytest.raises((ResponseCompatibilityError, ImplicitSolveError, ValueError)):
        BoundCCSDResponse(bound, lam)


def test_bad_lambda_pair_symmetry_rejected() -> None:
    _, _, _, bound, lam, _, _ = _state("h2o")
    values = np.array(lam.lambda2)
    values[0, 1, 0, 1] += 1e-3
    with pytest.raises(ValueError, match="symmetry"):
        BoundCCSDResponse(bound, replace(lam, lambda2=values))


def test_stream_rechecks_expected_identity_and_live_lifetime() -> None:
    s, _, cc, _, _, _, _ = _state()
    current = [s.identity]
    bound = BoundCCSDLambda(s, cc, current_reference=lambda: current[0])
    lam = bound.solve(reference_identity=s.identity)
    response = BoundCCSDResponse(bound, lam)
    with pytest.raises(ResponseCompatibilityError):
        response.weight("foo", reference_identity="wrong")
    stream = response.iter_weights(("foo", "fvv"), reference_identity=s.identity)
    first = next(stream)
    assert first.provenance["reference_binding"] == "live-reference-callback"
    current[0] = "another-generation"
    with pytest.raises(ResponseCompatibilityError):
        next(stream)
    with pytest.raises(ResponseCompatibilityError):
        BoundCCSDResponse(bound, lam)


def test_state_budget_rejects_before_numeric_execution(
    monkeypatch: typing.Any,
) -> None:
    _, _, _, bound, lam, response, _ = _state()
    from tools.vibeqc_cc import lambda_solver

    monkeypatch.setattr(
        lambda_solver,
        "execute",
        lambda *a, **kw: pytest.fail("execution before admission"),
    )
    with pytest.raises(ImplicitSolveError, match="state exceeds"):
        BoundCCSDResponse(
            bound, lam, max_bytes=response.logical_reserved_host_bytes - 1
        )


def test_block_budget_accounts_for_bound_state_and_rejects_before_execution(
    monkeypatch: typing.Any,
) -> None:
    s, _, _, bound, lam, response, _ = _state()
    needed = response.required_bytes("ovov", reference_identity=s.identity)
    exact = BoundCCSDResponse(bound, lam, max_bytes=needed)
    assert (
        exact.weight("ovov", reference_identity=s.identity).logical_reserved_host_bytes
        == needed
    )
    small = BoundCCSDResponse(bound, lam, max_bytes=needed - 1)
    monkeypatch.setattr(
        response_module,
        "execute",
        lambda *a, **kw: pytest.fail("weight execution before admission"),
    )
    with pytest.raises(ImplicitSolveError, match="host budget exceeded"):
        small.weight("ovov", reference_identity=s.identity)


@pytest.mark.parametrize(
    "mode", ("wrong_independent", "nonfinite", "dtype", "shape", "backend", "alias")
)
def test_generated_output_cannot_bypass_checks(
    monkeypatch: typing.Any, mode: typing.Any
) -> None:
    s, _, _, _, _, response, _ = _state()
    _, independent, _, _ = response._prepare("ovov")
    original = response_module.execute
    shared_storage = None

    def corrupt(program: typing.Any, *a: typing.Any, **kw: typing.Any) -> typing.Any:
        nonlocal shared_storage
        result = original(program, *a, **kw)
        if mode == "backend":
            return replace(result, backend="unapproved-cuda-fallback")
        value = result.outputs["bar_ovov"]
        if mode == "alias":
            if shared_storage is None:
                shared_storage = np.array(value)
            else:
                shared_storage[:] += 1e-3
            return replace(result, outputs={"bar_ovov": shared_storage})
        if program.logical_hash != independent.program.logical_hash:
            return result
        if mode == "wrong_independent":
            value = value + 1e-3
        elif mode == "nonfinite":
            value = np.full_like(value, np.nan)
        elif mode == "dtype":
            value = value.astype(np.float32)
        elif mode == "shape":
            value = value.reshape(-1)
        return replace(result, outputs={"bar_ovov": value})

    monkeypatch.setattr(response_module, "execute", corrupt)
    with pytest.raises((ResponseCompatibilityError, ImplicitSolveError, ValueError)):
        response.weight("ovov", reference_identity=s.identity)


def test_stale_during_weight_execution_cannot_publish(
    monkeypatch: typing.Any,
) -> None:
    s, _, cc, _, _, _, _ = _state()
    current = [s.identity]
    bound = BoundCCSDLambda(s, cc, current_reference=lambda: current[0])
    response = BoundCCSDResponse(bound, bound.solve(reference_identity=s.identity))
    original = response_module.execute

    def changed(*a: typing.Any, **kw: typing.Any) -> typing.Any:
        result = original(*a, **kw)
        current[0] = "replaced-in-flight"
        return result

    monkeypatch.setattr(response_module, "execute", changed)
    with pytest.raises(ResponseCompatibilityError):
        response.weight("ovov", reference_identity=s.identity)


@pytest.mark.parametrize(
    "parameters", ("foo", (), ("foo", "foo"), ("foo", "t1"), (True,))
)
def test_invalid_stream_requests(parameters: typing.Any) -> None:
    s, _, _, _, _, response, _ = _state()
    with pytest.raises((TypeError, ValueError)):
        list(response.iter_weights(parameters, reference_identity=s.identity))


@pytest.mark.parametrize("budget", (-1, True, 1.5))
def test_invalid_budget_types(budget: typing.Any) -> None:
    _, _, _, bound, lam, _, _ = _state()
    with pytest.raises(ValueError):
        BoundCCSDResponse(bound, lam, max_bytes=budget)


def test_invalid_consumer_types_and_directions() -> None:
    s, _, _, bound, lam, response, _ = _state("h2o")
    for first, second in ((object(), lam), (bound, object())):
        with pytest.raises(TypeError):
            BoundCCSDResponse(first, second)
    w = response.weight("foo", reference_identity=s.identity)
    direction = np.eye(s.nocc)
    for bad in (
        direction.astype(np.float32),
        direction.reshape(-1),
        direction * np.nan,
    ):
        with pytest.raises(ValueError):
            w.contract(bad)
    direction[0, 1] = 0.1
    with pytest.raises(ValueError, match="symmetry"):
        w.contract(direction)


def _determinant_root(
    oracle: typing.Any, bound: typing.Any, point: typing.Any
) -> typing.Any:
    """Tiny independent numerical Newton solve; never used in a production path."""
    point = np.array(point)
    for _ in range(12):
        out = oracle.evaluate_full(*bound._unpack(point))
        residual = bound._pack(out[1:])
        if np.max(np.abs(residual)) < 5e-13:
            return out[0]
        jacobian = np.empty((point.size, point.size))
        for column in range(point.size):
            direction = np.zeros_like(point)
            direction[column] = 1e-5
            plus = oracle.evaluate_full(*bound._unpack(point + direction))
            minus = oracle.evaluate_full(*bound._unpack(point - direction))
            jacobian[:, column] = (
                bound._pack(plus[1:]) - bound._pack(minus[1:])
            ) / 2e-5
        point -= np.linalg.solve(jacobian, residual)
    pytest.fail("independent perturbed determinant CC root did not converge")


@pytest.mark.parametrize("name,shift", [("h2", 0.0), ("h2", 0.15), ("h4", 0.0)])
def test_native_hf_cc_lambda_weights_vs_resolved_determinant(
    name: typing.Any, shift: typing.Any
) -> None:
    meta, _ = load("h2" if name == "h4" else name)
    inputs = deepcopy(meta["inputs"])
    inputs["coordinates"][1][2] += shift
    if name == "h4":
        inputs["atomic_numbers"] *= 2
        inputs["coordinates"] += [
            [p[0] + 5.0, p[1] + 0.3, p[2] + 0.2] for p in inputs["coordinates"]
        ]
        shells = deepcopy(inputs["shells"])
        for shell in shells:
            shell["atom_index"] += 2
        inputs["shells"] += shells
    try:
        source = NativeSource(**source_arguments(inputs))
    except (OSError, FileNotFoundError) as error:
        pytest.skip(str(error))
    except RuntimeError as error:
        if "native library was not found" not in str(error):
            raise
        pytest.skip(str(error))
    with source:
        s, _ = export_rhf(source, tolerance=1e-12, max_iterations=150)
        assert s.hf_backend == "native-cpu"
        with ConventionalProvider(s, source) as provider:

            def current() -> typing.Any:
                source._check_open()
                if provider._closed:
                    raise ResponseCompatibilityError("provider closed")
                return provider.snapshot.identity

            cc = solve(s, provider, options=_STRICT)
            assert cc.converged
            bound = BoundCCSDLambda(s, cc, current_reference=current)
            response = BoundCCSDResponse(
                bound, bound.solve(reference_identity=s.identity)
            )
            f = s.coefficients.T @ s.fock @ s.coefficients
            g = provider.get(MOBlock((tuple(range(s.nmo)),) * 4)).to_host()
            df, dg, _, _ = random_case(s.nocc, s.nmo - s.nocc, 152)
            df /= np.linalg.norm(df)
            dg /= np.linalg.norm(dg)
            directions = dense_feeds(df, dg, cc.t1 * 0, cc.t2 * 0)
            expected = sum(
                weight.contract(directions[weight.parameter])
                for weight in response.iter_weights(reference_identity=s.identity)
            )
            point = bound._pack((cc.t1, cc.t2))
            for h in (1e-4, 3e-5, 1e-5):
                energies = [
                    _determinant_root(
                        DeterminantOracle(f + sign * h * df, g + sign * h * dg, s.nocc),
                        bound,
                        point,
                    )
                    for sign in (-1, 1)
                ]
                np.testing.assert_allclose(
                    (energies[1] - energies[0]) / (2 * h),
                    expected,
                    atol=3e-8,
                    rtol=1e-6,
                )
        with pytest.raises(ResponseCompatibilityError, match="provider closed"):
            response.weight("foo", reference_identity=s.identity)
