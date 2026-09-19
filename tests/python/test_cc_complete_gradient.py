"""Complete small CPU RCCSD gradients, generated chains and failure boundaries."""

import builtins
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.tensor import Index, IndexSpace, Program, TensorSpec, execute

from tools.cc_gradient_fixtures import CASES, inputs, load, source_arguments
from tools.vibeqc_cc import BoundCCSDLambda, BoundCCSDResponse, solve
from tools.vibeqc_cc import complete_gradient as module
from tools.vibeqc_cc.complete_gradient import (
    BoundCCSDGradient,
    CCSDGradientOptions,
    complete_gradient_validation,
)
from tools.vibeqc_cc.gradient_equations import (
    build_ao_weight_program,
    build_hamiltonian_programs,
)
from tools.vibeqc_cc.lambda_equations import PARAMETERS
from tools.vibeqc_cc.oracle import dense_feeds, random_case
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response.implicit import ImplicitSolveError
from tools.vibeqc_response.oracle import _expm_small, explicit_rhf_response_matrix
from tools.vibeqc_response.problem import ResponseCompatibilityError


def _source(value):
    try:
        return NativeSource(**source_arguments(value))
    except (OSError, FileNotFoundError) as error:
        pytest.skip(str(error))
    except RuntimeError as error:
        if "native library was not found" not in str(error):
            raise
        pytest.skip(str(error))


@lru_cache(maxsize=len(CASES))
def _result(name):
    with _source(inputs(name)) as source:
        return complete_gradient_validation(source)


@contextmanager
def _prepared(name="h2"):
    options = CCSDGradientOptions()
    with _source(inputs(name)) as source:
        reference, _ = export_rhf(
            source, tolerance=options.scf_tolerance, max_iterations=150
        )
        with ConventionalProvider(reference, source) as provider:
            cc = solve(reference, provider, options=options.cc_options)
            assert cc.converged

            def current():
                source._check_open()
                if provider._closed:
                    raise ResponseCompatibilityError("provider closed")
                return provider.snapshot.identity

            bound = BoundCCSDLambda(reference, cc, current_reference=current)
            response = BoundCCSDResponse(
                bound, bound.solve(reference_identity=reference.identity)
            )
            yield response, provider


@pytest.fixture(scope="module")
def tiny_state():
    with _prepared() as pair:
        yield BoundCCSDGradient(*pair)


@pytest.fixture(scope="module")
def water_state():
    with _prepared("h2o") as pair:
        yield BoundCCSDGradient(*pair)


def _direct_fields(h, g, rotation, o):
    """Independent NumPy/loop primal, not the generated reverse graph."""
    h = rotation.T @ h @ rotation
    g = np.einsum("up,vq,wr,xs,uvwx->pqrs", *(rotation,) * 4, g, optimize=True)
    f = h.copy()
    for i in range(o):
        f += 2 * g[:, :, i, i] - g[:, i, i, :]
    hf = sum(h[i, i] + f[i, i] for i in range(o))
    result = dense_feeds(
        f, g, np.zeros((o, len(h) - o)), np.zeros((o, o, len(h) - o, len(h) - o))
    )
    return {name: result[name] for name in PARAMETERS} | {
        "reference_electronic_energy": hf,
        "fock": f,
    }


@pytest.mark.parametrize("o,v", [(1, 2), (2, 2)])
def test_generated_raw_hamiltonian_and_full_pullback_directions(o, v):
    n = o + v
    programs = build_hamiltonian_programs(o, v)
    h, g, _, _ = random_case(o, v, 152)
    dh, dg, _, _ = random_case(o, v, 153)
    rng = np.random.default_rng(154)
    u = np.eye(n) + rng.normal(scale=0.01, size=(n, n))
    du = rng.normal(scale=0.02, size=(n, n))
    fields = _direct_fields(h, g, u, o)
    actual = execute(programs.primal, {"h": h, "g": g, "rotation": u}).outputs
    for name in fields:
        np.testing.assert_allclose(actual[name], fields[name], atol=2e-13, rtol=2e-13)
    seeds = {
        "bar_" + name: np.asarray(rng.normal(size=np.shape(value)))
        for name, value in fields.items()
        if name != "fock"
    }
    bars = execute(
        programs.pullback.program, {"h": h, "g": g, "rotation": u, **seeds}
    ).outputs
    analytic = sum(
        np.sum(bars[name] * direction)
        for name, direction in (("bar_h", dh), ("bar_g", dg), ("bar_rotation", du))
    )

    def objective(sign, step):
        values = _direct_fields(
            h + sign * step * dh, g + sign * step * dg, u + sign * step * du, o
        )
        return sum(
            np.sum(seeds["bar_" + name] * values[name])
            for name in values
            if name != "fock"
        )

    for step in (1e-4, 3e-5, 1e-5):
        np.testing.assert_allclose(
            (objective(1, step) - objective(-1, step)) / (2 * step),
            analytic,
            atol=2e-8,
            rtol=2e-8,
        )
    restored = Program.from_payload(programs.weights.to_payload())
    assert restored.logical_hash == programs.weights.logical_hash
    assert all(len(node.spec.indices) <= 4 for node in programs.weights.live_nodes)
    # No packed-T2 incidence matrix or full CC Jacobian exists in this graph.
    assert not any(node.op == "gather" for node in programs.weights.live_nodes)


def test_full_orbital_population_is_not_relabelled_as_ao_or_occupied():
    mo = IndexSpace("all_mo", "orbital", 4)
    assert mo != IndexSpace("all_mo", "ao", 4)
    assert mo != IndexSpace("all_mo", "occupied", 4)
    spec = TensorSpec((Index("p", mo),))
    assert spec.indices[0].space.kind == "orbital"
    programs = build_hamiltonian_programs(2, 2)
    for node in programs.primal.live_nodes:
        assert all(index.space.kind == "orbital" for index in node.spec.indices)
    ao = build_ao_weight_program(4)
    assert all(index.space.kind == "ao" for index in ao.outputs["eri"].spec.indices)
    for bad in ((0, 1), (1, 0), (True, 1), (7, 7)):
        with pytest.raises(ValueError):
            build_hamiltonian_programs(*bad)
    with pytest.raises(ValueError):
        build_ao_weight_program(13)


def test_ao_transform_preserves_ordered_contractions():
    rng = np.random.default_rng(153)
    n = 4
    c = rng.normal(size=(n, n))
    weights = {
        "hcore": rng.normal(size=(n, n)),
        "overlap": rng.normal(size=(n, n)),
        "eri": rng.normal(size=(n,) * 4),
    }
    out = execute(build_ao_weight_program(n), {"coefficients": c, **weights}).outputs
    for name in ("hcore", "overlap"):
        direction = rng.normal(size=(n, n))
        np.testing.assert_allclose(
            np.sum(out[name] * direction),
            np.sum(weights[name] * (c.T @ direction @ c)),
            atol=1e-11,
            rtol=1e-12,
        )
    direction = rng.normal(size=(n,) * 4)
    transformed = np.einsum(
        "up,vq,wr,xs,uvwx->pqrs", *(c,) * 4, direction, optimize=True
    )
    np.testing.assert_allclose(
        np.sum(out["eri"] * direction),
        np.sum(weights["eri"] * transformed),
        atol=1e-10,
        rtol=1e-12,
    )


@pytest.mark.parametrize("name", CASES)
def test_complete_native_endpoint_matches_pinned_pyscf_gradient(name):
    result = _result(name)
    oracle = load(name)
    np.testing.assert_allclose(
        result.total_energy, oracle["total_energy"], atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(result.gradient, oracle["gradient"], atol=1e-6, rtol=0)
    assert max(result.scf_residual, result.cc_residual, result.lambda_residual) <= 1e-9
    assert result.z_residual <= 1e-10
    assert result.orbital_stationarity <= 1e-8
    assert result.minimum_orbital_curvature > 1e-8
    np.testing.assert_allclose(
        sum(result.physical_components.values()),
        result.gradient,
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(
        sum(result.integral_components.values()),
        result.gradient,
        atol=1e-11,
        rtol=1e-11,
    )
    np.testing.assert_allclose(result.forces, -result.gradient, atol=0, rtol=0)
    np.testing.assert_allclose(result.gradient.sum(axis=0), 0, atol=1e-9, rtol=0)
    xyz = np.array(oracle["inputs"]["coordinates"])
    np.testing.assert_allclose(
        np.cross(xyz, result.gradient).sum(axis=0), 0, atol=2e-8, rtol=0
    )
    assert result.diagnostics["dense_cc_jacobian"] is False
    assert result.diagnostics["native_public_force_capability"] is False
    assert result.diagnostics["triples_gradient"] is False
    assert result.diagnostics["tensor_backend"] == "numpy-cpu-interpreter"
    for array in (
        result.gradient,
        result.forces,
        *result.physical_components.values(),
        *result.integral_components.values(),
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.total_energy = 0.0
    with pytest.raises(TypeError):
        result.integral_components["overlap"] = np.zeros_like(result.gradient)


def _fresh_energy(value):
    options = CCSDGradientOptions()
    with _source(value) as source:
        reference, _ = export_rhf(
            source, tolerance=options.scf_tolerance, max_iterations=150
        )
        with ConventionalProvider(reference, source) as provider:
            cc = solve(reference, provider, options=options.cc_options)
            assert cc.converged
            return cc.total_energy, reference.identity


@pytest.mark.parametrize("name", ("h2", "h2o", "nh3", "h2_d_spherical"))
def test_nuclear_finite_differences_resolve_hf_and_cc_at_three_steps(name):
    result = _result(name)
    value = inputs(name)
    direction = np.random.default_rng(153).normal(size=result.gradient.shape)
    direction /= np.linalg.norm(direction)
    analytic = float(np.sum(result.gradient * direction))
    identities = set()
    errors = []
    for step in (1e-3, 3e-4, 1e-4):
        energies = []
        for sign in (-1, 1):
            displaced = deepcopy(value)
            displaced["coordinates"] = (
                np.asarray(value["coordinates"]) + sign * step * direction
            ).tolist()
            energy, identity = _fresh_energy(displaced)
            energies.append(energy)
            identities.add(identity)
        errors.append(abs((energies[1] - energies[0]) / (2 * step) - analytic))
    assert len(identities) == 6
    assert errors[-1] < 2e-7, errors
    assert min(errors[1:]) < 2e-7, errors


def test_overlap_and_orbital_omissions_are_detectable():
    result = _result("h2o")
    oracle = np.array(load("h2o")["gradient"])
    without_overlap = result.gradient - result.integral_components["overlap"]
    without_z = result.gradient - result.physical_components["orbital_response"]
    assert np.max(abs(without_overlap - oracle)) > 1e-3
    assert np.max(abs(without_z - oracle)) > 1e-6


def test_rigid_motion_covariance_and_changed_geometry():
    name = "h2_d_spherical"
    value = inputs(name)
    result = _result(name)
    q, _ = np.linalg.qr(np.random.default_rng(154).normal(size=(3, 3)))
    moved = deepcopy(value)
    moved["coordinates"] = (
        np.asarray(value["coordinates"]) @ q + [0.3, -0.7, 0.2]
    ).tolist()
    with _source(moved) as source:
        transformed = complete_gradient_validation(source)
    np.testing.assert_allclose(
        transformed.total_energy, result.total_energy, atol=1e-9, rtol=0
    )
    np.testing.assert_allclose(
        transformed.gradient, result.gradient @ q, atol=3e-8, rtol=0
    )
    assert transformed.source_identity != result.source_identity
    assert _result("h2_shifted").source_identity != _result("h2").source_identity
    assert abs(_result("h2_shifted").total_energy - _result("h2").total_energy) > 1e-3


def test_shared_z_operator_matches_generated_and_independent_mo_matrix(water_state):
    state = water_state
    expected = explicit_rhf_response_matrix(state.operator.problem, state.provider)
    np.testing.assert_allclose(state.orbital_matrix, expected, atol=1e-10, rtol=1e-10)
    rng = np.random.default_rng(153)
    left, right = (rng.normal(size=state.operator.dimension) for _ in range(2))
    assert state.operator.dot_identity(left, right) < 1e-12
    np.testing.assert_allclose(
        state.operator.apply(right),
        state._generated_orbital_action(right),
        atol=1e-10,
        rtol=1e-10,
    )
    # Full-CI-in-space redundancies are not solved using singular oo/vv gaps.
    assert state.same_space_stationarity < 1e-8
    np.testing.assert_allclose(
        np.trace(state.weights["hcore"]),
        state.reference.electron_count,
        atol=1e-9,
        rtol=0,
    )
    np.testing.assert_allclose(state.weights["stationarity"], 0, atol=1e-8, rtol=0)
    x = right / np.linalg.norm(right)
    k = state.operator.problem.layout.generator_matrix(x)
    for step in (1e-4, 3e-5, 1e-5):
        values = []
        for sign in (-1, 1):
            fields = _direct_fields(
                state.raw_inputs["h"],
                state.raw_inputs["g"],
                _expm_small(sign * step * k),
                state.reference.nocc,
            )
            values.append(fields["fov"].reshape(-1))
        np.testing.assert_allclose(
            -(values[1] - values[0]) / (2 * step),
            state.orbital_matrix @ x,
            atol=1e-7,
            rtol=1e-7,
        )


@pytest.mark.parametrize(
    "changes",
    [
        {"max_bytes": 0},
        {"max_bytes": True},
        {"provider_budget_bytes": -1},
        {"scf_max_iterations": False},
        {"scf_tolerance": 1e-5},
        {"orbital_residual_tolerance": 1e-5},
        {"stationarity_tolerance": 1e-5},
        {"minimum_orbital_curvature": 0},
        {"minimum_orbital_curvature": float("nan")},
        {"cc_options": object()},
        {"lambda_options": object()},
        {"z_options": object()},
    ],
)
def test_invalid_options(changes):
    with pytest.raises((ValueError, TypeError)):
        CCSDGradientOptions(**changes)


def test_budget_rejects_before_hf_and_before_raw_integrals(tiny_state, monkeypatch):
    with _source(inputs("h2")) as source:
        monkeypatch.setattr(
            module,
            "export_rhf",
            lambda *a, **kw: pytest.fail("HF before budget admission"),
        )
        with pytest.raises(ImplicitSolveError, match="before HF"):
            complete_gradient_validation(
                source, options=CCSDGradientOptions(max_bytes=1)
            )
    needed = tiny_state.logical_reserved_host_bytes
    options = CCSDGradientOptions(max_bytes=needed)
    exact = BoundCCSDGradient(tiny_state.response, tiny_state.provider, options=options)
    assert exact.logical_reserved_host_bytes == needed
    exact.gradient()
    monkeypatch.setattr(
        tiny_state.provider,
        "get",
        lambda *a, **kw: pytest.fail("integral read before admission"),
    )
    with pytest.raises(ImplicitSolveError, match="before integral reads"):
        BoundCCSDGradient(
            tiny_state.response,
            tiny_state.provider,
            options=replace(options, max_bytes=needed - 1),
        )


def test_unsupported_source_and_cpu_scope(monkeypatch):
    with pytest.raises(TypeError):
        complete_gradient_validation(object())
    with _source(inputs("h2")) as source:
        with pytest.raises(TypeError):
            complete_gradient_validation(source, options=object())
        monkeypatch.setattr(source, "backend", "cuda")
        with pytest.raises(TypeError, match="native CPU"):
            complete_gradient_validation(source)
    args = source_arguments(inputs("h2"))
    args.update(charge=1, multiplicity=2)
    with (
        NativeSource(**args) as source,
        pytest.raises(ValueError, match="closed-shell"),
    ):
        complete_gradient_validation(source)


def test_false_scf_and_cc_success_cannot_return_gradient(monkeypatch):
    with _source(inputs("h2")) as source:
        with monkeypatch.context() as m:
            m.setattr(
                module,
                "export_rhf",
                lambda *a, **k: (SimpleNamespace(scf_residual=1e-4), {}),
            )
            with pytest.raises(ImplicitSolveError, match="strictly converged"):
                complete_gradient_validation(source)
        with monkeypatch.context() as m:
            m.setattr(
                module,
                "solve",
                lambda *a, **k: SimpleNamespace(
                    converged=False, reason="injected nonconvergence"
                ),
            )
            with pytest.raises(ImplicitSolveError, match="primal failed"):
                complete_gradient_validation(source)


def test_real_cc_and_z_nonconvergence_fail_closed(tiny_state, water_state):
    with _source(inputs("h2")) as source:
        options = CCSDGradientOptions(
            cc_options=replace(CCSDGradientOptions().cc_options, max_iterations=1)
        )
        with pytest.raises(ImplicitSolveError, match="primal failed"):
            complete_gradient_validation(source, options=options)
    opts = CCSDGradientOptions(
        z_options=replace(CCSDGradientOptions().z_options, max_iterations=1, atol=1e-14)
    )
    with pytest.raises(ImplicitSolveError, match="adjoint failed"):
        BoundCCSDGradient(water_state.response, water_state.provider, options=opts)


def test_false_z_report_rechecked_against_independent_equation(
    water_state, monkeypatch
):
    def false(*args, **kwargs):
        return replace(
            water_state.z_result,
            solution=np.zeros_like(water_state.z_result.solution),
            residual_norm=0.0,
        )

    monkeypatch.setattr(module, "checked_transpose_solve", false)
    with pytest.raises(ImplicitSolveError, match="physical Z-vector"):
        BoundCCSDGradient(water_state.response, water_state.provider)


def test_negative_or_near_singular_orbital_curvature_rejected(tiny_state, monkeypatch):
    original = BoundCCSDGradient._generated_orbital_action
    monkeypatch.setattr(
        BoundCCSDGradient,
        "_generated_orbital_action",
        lambda self, x: -original(self, x),
    )
    with pytest.raises(ImplicitSolveError, match="unstable or near-singular"):
        BoundCCSDGradient(tiny_state.response, tiny_state.provider)


@pytest.mark.parametrize("kind", ("missing", "shape", "nonfinite", "dtype"))
def test_native_derivative_outputs_are_validated(tiny_state, monkeypatch, kind):
    original = tiny_state.source.integral_derivatives

    def broken(**kwargs):
        values = original(**kwargs)
        if kind == "missing":
            del values["overlap"]
        elif kind == "shape":
            values["hcore"] = values["hcore"].reshape(-1)
        elif kind == "nonfinite":
            values["eri"] = np.full_like(values["eri"], np.nan)
        else:
            values["nuclear"] = values["nuclear"].astype(np.float32)
        return values

    monkeypatch.setattr(tiny_state.source, "integral_derivatives", broken)
    with pytest.raises((ValueError, ResponseCompatibilityError)):
        tiny_state.gradient()


def test_changed_tensor_backend_and_live_provider_rejected(tiny_state, monkeypatch):
    original = module.execute
    with monkeypatch.context() as m:
        m.setattr(
            module,
            "execute",
            lambda *a, **k: replace(original(*a, **k), backend="unexpected"),
        )
        with pytest.raises(ResponseCompatibilityError, match="backend changed"):
            tiny_state.ao_weights()
    with monkeypatch.context() as m:
        reference = replace(
            tiny_state.provider.snapshot, generation_id="different-live-generation"
        )
        m.setattr(tiny_state.provider, "snapshot", reference)
        with pytest.raises(ResponseCompatibilityError, match="stale"):
            tiny_state.gradient()


def test_raw_integrals_cannot_be_swapped_under_a_cc_result(tiny_state, monkeypatch):
    original = tiny_state.provider.get

    def wrong(block):
        result = original(block)
        return replace(result, values=result.values + 0.01)

    monkeypatch.setattr(tiny_state.provider, "get", wrong)
    with pytest.raises((ResponseCompatibilityError, ValueError)):
        BoundCCSDGradient(tiny_state.response, tiny_state.provider)


def test_no_external_qc_solver_is_a_runtime_dependency(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in ("pyscf", "torch", "cupy"):
            pytest.fail("external QC/AD runtime imported by native complete gradient")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with _source(inputs("h2")) as source:
        result = complete_gradient_validation(source)
    assert result.gradient.shape == (2, 3)


@pytest.mark.parametrize(
    "kind", ("spin", "triples", "frozen", "ecp", "units", "auxiliary", "unknown")
)
def test_cli_input_schema_never_drops_unsupported_physics(kind):
    value = inputs("h2")
    if kind == "spin":
        value["multiplicity"] = 3
    elif kind == "triples":
        value["cc_settings"]["method"] = "CCSD(T)"
    elif kind == "frozen":
        value["cc_settings"]["frozen_core"] = 1
    elif kind == "ecp":
        value["conventions"]["hamiltonian"] = "ECP"
    elif kind == "units":
        value["conventions"]["length_unit"] = "angstrom"
    elif kind == "auxiliary":
        value["auxiliary_centers"] = [{"atom_index": 0}]
    else:
        value["ecp"] = "unsupported"
    with pytest.raises(ValueError):
        source_arguments(value)


def test_cli_json_and_existing_output_are_safe(tmp_path, monkeypatch, tiny_state):
    import json

    from tools import run_ccsd_gradient as driver

    result = tiny_state.gradient()
    path = tmp_path / "gradient.json"
    monkeypatch.setattr(driver, "complete_gradient_validation", lambda *a, **k: result)
    monkeypatch.setattr("sys.argv", ["driver", "--case", "h2", "--output", str(path)])
    driver.main()
    content = path.read_bytes()
    data = json.loads(content)
    np.testing.assert_array_equal(data["gradient"], result.gradient)
    assert data["diagnostics"]["triples_gradient"] is False
    with pytest.raises(SystemExit):
        driver.main()
    assert path.read_bytes() == content


def test_closed_source_cannot_publish_a_gradient():
    with _prepared() as pair:
        state = BoundCCSDGradient(*pair)
        state.source.close()
        with pytest.raises(RuntimeError, match="closed"):
            state.gradient()
