"""Triples orbital-energy, total Z and overlap response tests for #155 A."""

import typing
from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from test_cc_complete_gradient import _direct_fields, _source
from test_cc_native_tensor_cuda import _fake_executor
from vibeqc_compiler.tensor import Program, execute

from tools.cc_gradient_fixtures import inputs
from tools.vibeqc_cc import (
    BoundCCSDLambda,
    BoundCCSDTOrbitalResponse,
    BoundCCSDTResponse,
    CCSDGradientOptions,
    solve,
    solve_corrected_lambda,
)
from tools.vibeqc_cc.gradient_equations import build_fock_weight_program
from tools.vibeqc_cc.oracle import random_case
from tools.vibeqc_cc.triples_orbital_response import (
    _minimum_same_space_gap,
    _same_space_fock_cotangent,
)
from tools.vibeqc_posthf.export import export_rhf
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_response import NativeJKBackend, RHFResponseOperator
from tools.vibeqc_response.krylov import _HostKrylovEngine
from tools.vibeqc_response.problem import ResponseCompatibilityError


@contextmanager
def _prepared_ccsdt(
    name: str = "h2o", *, parameter_executor: typing.Any = None
) -> typing.Any:
    options = CCSDGradientOptions()
    with _source(inputs(name)) as source:
        reference, _ = export_rhf(
            source,
            tolerance=options.scf_tolerance,
            max_iterations=options.scf_max_iterations,
        )
        with ConventionalProvider(reference, source) as provider:
            cc = solve(reference, provider, options=options.cc_options)
            assert cc.converged, cc.reason

            def current() -> str:
                source._check_open()
                if provider._closed:
                    raise ResponseCompatibilityError("provider closed")
                return provider.snapshot.identity

            bound = BoundCCSDLambda(
                reference,
                cc,
                options=options.lambda_options,
                current_reference=current,
            )
            baseline = bound.solve(reference_identity=reference.identity)
            corrected = solve_corrected_lambda(
                bound,
                baseline,
                vir_chunk_size=1,
            )
            response = BoundCCSDTResponse(
                bound,
                baseline,
                corrected,
                vir_chunk_size=1,
                parameter_executor=parameter_executor,
            )
            yield response, provider, options


@pytest.fixture(scope="module")
def water_state() -> typing.Any:
    with _prepared_ccsdt() as (response, provider, options):
        yield BoundCCSDTOrbitalResponse(response, provider, options=options)


@pytest.mark.parametrize("o,v", ((1, 2), (2, 2)))
def test_generated_fock_cotangent_matches_independent_directional_differences(
    o: int, v: int
) -> None:
    n = o + v
    h, g, _, _ = random_case(o, v, 155)
    dh, dg, _, _ = random_case(o, v, 156)
    rng = np.random.default_rng(157)
    rotation = np.eye(n) + rng.normal(scale=0.01, size=(n, n))
    d_rotation = rng.normal(scale=0.02, size=(n, n))
    diagonal = rng.normal(size=n)
    bar_fock = np.diag(diagonal)

    program = build_fock_weight_program(o, v)
    bars = execute(
        program,
        {
            "h": h,
            "g": g,
            "rotation": rotation,
            "bar_fock": bar_fock,
        },
    ).outputs
    analytic = sum(
        np.sum(bars[name] * direction)
        for name, direction in (
            ("hcore", dh),
            ("eri", dg),
            ("rotation_gradient", d_rotation),
        )
    )

    def objective(sign: float, step: float) -> float:
        fock = _direct_fields(
            h + sign * step * dh,
            g + sign * step * dg,
            rotation + sign * step * d_rotation,
            o,
        )["fock"]
        return float(np.sum(bar_fock * fock))

    for step in (1.0e-4, 3.0e-5, 1.0e-5):
        np.testing.assert_allclose(
            (objective(1.0, step) - objective(-1.0, step)) / (2 * step),
            analytic,
            atol=3e-8,
            rtol=3e-8,
        )

    restored = Program.from_payload(program.to_payload())
    assert restored.logical_hash == program.logical_hash
    assert program.provenance["scope"] == "direct canonical-orbital-energy response"


def test_total_ccsdt_orbital_rhs_and_raw_weights_decompose(
    water_state: BoundCCSDTOrbitalResponse,
) -> None:
    state = water_state
    correlation_parts = (
        "ccsd_baseline",
        "direct_triples",
        "delta_lambda",
        "triples_denominator",
        "same_space_canonicalization",
    )
    for field in (
        "hcore",
        "eri",
        "overlap",
        "rotation_gradient",
        "stationarity",
        "orbital_rhs",
    ):
        expected_correlation = sum(
            (state.component_weights[name][field] for name in correlation_parts),
            np.zeros_like(state.correlation_weights[field]),
        )
        np.testing.assert_allclose(
            state.correlation_weights[field],
            expected_correlation,
            atol=2e-11,
            rtol=2e-11,
        )
        expected_total = expected_correlation.copy()
        expected_total += state.component_weights["hf"][field]
        expected_total += state.component_weights["orbital_response"][field]
        np.testing.assert_allclose(
            state.weights[field],
            expected_total,
            atol=2e-11,
            rtol=2e-11,
        )

    np.testing.assert_array_equal(
        state.orbital_rhs,
        state.correlation_weights["orbital_rhs"].reshape(-1),
    )
    assert np.linalg.norm(state.orbital_rhs - state.baseline.orbital_rhs) > 1.0e-8
    assert state.same_space_stationarity <= state.options.stationarity_tolerance
    assert state.orbital_stationarity <= state.options.stationarity_tolerance
    assert state.z_residual <= state.options.orbital_residual_tolerance
    assert state.minimum_orbital_curvature > state.options.minimum_orbital_curvature


def test_cuda_tensor_owner_covers_hamiltonian_response_without_bound_cpu_replay(
    water_state: BoundCCSDTOrbitalResponse,
    monkeypatch: typing.Any,
    tmp_path: typing.Any,
) -> None:
    executor, budgets, compiled = _fake_executor(tmp_path / "cc-response")
    with _prepared_ccsdt(parameter_executor=executor) as (
        response,
        provider,
        options,
    ):
        # Exact identities include the RHF export generation. Compare dispatch
        # paths on this same state; the independently exported water fixture
        # remains the numerical oracle below, not an interchangeable owner.
        same_state_reference = BoundCCSDTOrbitalResponse(
            response, provider, options=options
        )

        def reject_bound_tensor_execution(
            *args: object, **kwargs: object
        ) -> typing.NoReturn:
            del args, kwargs
            raise AssertionError("bound CPU Hamiltonian/parameter TensorIR replayed")

        monkeypatch.setattr(
            BoundCCSDLambda,
            "_tensor_execute",
            reject_bound_tensor_execution,
        )
        actual = BoundCCSDTOrbitalResponse(response, provider, options=options)

    assert actual.baseline.tensor_backend == executor.backend
    assert actual.response_identity == same_state_reference.response_identity
    for field in (
        "hcore",
        "eri",
        "overlap",
        "rotation_gradient",
        "stationarity",
        "orbital_rhs",
    ):
        np.testing.assert_allclose(
            actual.weights[field],
            water_state.weights[field],
            atol=2e-11,
            rtol=2e-11,
        )
    np.testing.assert_allclose(
        actual.z_result.solution,
        water_state.z_result.solution,
        atol=2e-11,
        rtol=2e-11,
    )
    assert budgets and set(budgets) == {64 << 20}
    assert compiled
    assert executor.compiled_program_count == len(set(compiled))


def test_borrowed_response_backend_owns_physical_z_actions(
    water_state: BoundCCSDTOrbitalResponse,
    monkeypatch: typing.Any,
) -> None:
    with _prepared_ccsdt() as (response, provider, options):
        same_state_reference = BoundCCSDTOrbitalResponse(
            response, provider, options=options
        )
        borrowed = NativeJKBackend(
            provider.source,
            axis_tile=max(provider.source.shell_sizes),
            budget_bytes=options.provider_budget_bytes,
        )

        def reject_internal_backend(*args: object, **kwargs: object) -> typing.NoReturn:
            del args, kwargs
            raise AssertionError("orbital response rebuilt its hard-coded CPU backend")

        from tools.vibeqc_cc import complete_gradient as complete_gradient_module

        monkeypatch.setattr(
            complete_gradient_module,
            "NativeJKBackend",
            reject_internal_backend,
        )
        actual = BoundCCSDTOrbitalResponse(
            response,
            provider,
            options=options,
            response_backend=borrowed,
        )

    assert actual.baseline.response_backend is borrowed
    assert actual.baseline.operator.backend is borrowed
    assert borrowed.statistics["actions"] > 0
    assert actual.response_identity == same_state_reference.response_identity
    np.testing.assert_allclose(
        actual.z_result.solution,
        water_state.z_result.solution,
        atol=2e-11,
        rtol=2e-11,
    )
    for field in ("hcore", "eri", "overlap", "orbital_rhs"):
        np.testing.assert_allclose(
            actual.weights[field],
            water_state.weights[field],
            atol=2e-11,
            rtol=2e-11,
        )


def test_resident_z_execution_reuses_checked_krylov_engine(
    water_state: BoundCCSDTOrbitalResponse,
) -> None:
    class FakeResident:
        resident = True

        def __init__(
            self,
            matrix: np.ndarray,
            *,
            vector_slots: int,
            device_budget_bytes: int,
        ) -> None:
            self.matrix = np.asarray(matrix, dtype=np.float64)
            self.dimension = self.matrix.shape[0]
            self.vector_slots = vector_slots
            self.workspace_bytes = min(
                device_budget_bytes,
                max(1, self.dimension * vector_slots * 8),
            )
            self.diagnostics = {
                "owned_device_bytes": self.workspace_bytes,
                "operator_actions": 0,
                "fake_resident": 1,
            }
            self._host = _HostKrylovEngine(self.dimension)
            self.closed = False

        def __getattr__(self, name: str) -> typing.Any:
            return getattr(self._host, name)

        def apply(self, operator: typing.Any, value: typing.Any) -> np.ndarray:
            del operator
            self.diagnostics["operator_actions"] += 1
            return self.matrix @ np.asarray(value, dtype=np.float64)

        def close(self) -> None:
            self.closed = True

    class ResidentBackend:
        def __init__(self, inner: NativeJKBackend) -> None:
            self.inner = inner
            self.identity = inner.identity
            self.hamiltonian_id = inner.hamiltonian_id
            self.nbf = inner.nbf
            self.host_workspace_bytes = inner.host_workspace_bytes
            self.device_workspace_bytes = 0
            self.device_resident_bytes = 4096
            self.statistics = inner.statistics
            self.last_resident: FakeResident | None = None

        def validate_reference(self, reference: typing.Any) -> typing.Any:
            return self.inner.validate_reference(reference)

        def coulomb_exchange(self, density: typing.Any) -> typing.Any:
            return self.inner.coulomb_exchange(density)

        def resident_response(
            self,
            problem: typing.Any,
            *,
            vector_slots: int,
            device_budget_bytes: int,
        ) -> FakeResident:
            matrix = RHFResponseOperator(problem, self).to_dense()
            owner = FakeResident(
                matrix,
                vector_slots=vector_slots,
                device_budget_bytes=device_budget_bytes,
            )
            self.last_resident = owner
            return owner

    with _prepared_ccsdt() as (response, provider, options):
        same_state_host = BoundCCSDTOrbitalResponse(response, provider, options=options)
        inner = NativeJKBackend(
            provider.source,
            axis_tile=max(provider.source.shell_sizes),
            budget_bytes=options.provider_budget_bytes,
        )
        backend = ResidentBackend(inner)
        # Exact response identity also hashes the solved Z values. A dense
        # resident action can differ from a host J/K action in the last bit;
        # require exact identity only for a deterministic same-state replay.
        same_state_reference = BoundCCSDTOrbitalResponse(
            response,
            provider,
            options=options,
            response_backend=backend,
            response_execution="cuda-resident",
            response_device_budget_bytes=1 << 20,
        )
        actual = BoundCCSDTOrbitalResponse(
            response,
            provider,
            options=options,
            response_backend=backend,
            response_execution="cuda-resident",
            response_device_budget_bytes=1 << 20,
        )

    resident = backend.last_resident
    assert resident is not None
    assert resident.closed is True
    assert resident.diagnostics["operator_actions"] > 0
    assert actual.response_execution == "cuda-resident"
    assert actual.resident_response_diagnostics["fake_resident"] == 1
    assert actual.response_identity == same_state_reference.response_identity
    assert actual.reference_identity == same_state_host.reference_identity
    assert (
        actual.fixed_orbital_response_identity
        == same_state_host.fixed_orbital_response_identity
    )
    assert (
        actual.baseline.operator_identity == same_state_host.baseline.operator_identity
    )
    np.testing.assert_allclose(
        actual.z_result.solution,
        water_state.z_result.solution,
        atol=2e-11,
        rtol=2e-11,
    )
    for field in ("hcore", "eri", "overlap", "orbital_rhs"):
        np.testing.assert_allclose(
            actual.weights[field],
            water_state.weights[field],
            atol=2e-11,
            rtol=2e-11,
        )


def test_real_denominator_sources_chain_through_canonical_fock(
    water_state: BoundCCSDTOrbitalResponse,
) -> None:
    state = water_state
    rng = np.random.default_rng(158)
    raw = state.baseline.raw_inputs
    dh = rng.normal(size=raw["h"].shape)
    dh = 0.5 * (dh + dh.T)
    dg = rng.normal(size=raw["g"].shape)
    # Preserve all eight chemists-ERI permutations in the perturbation.
    dg = 0.125 * (
        dg
        + dg.transpose(1, 0, 2, 3)
        + dg.transpose(0, 1, 3, 2)
        + dg.transpose(1, 0, 3, 2)
        + dg.transpose(2, 3, 0, 1)
        + dg.transpose(3, 2, 0, 1)
        + dg.transpose(2, 3, 1, 0)
        + dg.transpose(3, 2, 1, 0)
    )
    du = rng.normal(scale=0.02, size=raw["rotation"].shape)
    scale = np.sqrt(np.sum(dh * dh) + np.sum(dg * dg) + np.sum(du * du))
    dh, dg, du = dh / scale, dg / scale, du / scale

    denominator = state.component_weights["triples_denominator"]
    analytic = (
        np.sum(denominator["hcore"] * dh)
        + np.sum(denominator["eri"] * dg)
        + np.sum(denominator["rotation_gradient"] * du)
    )
    source = np.concatenate(
        (
            state.orbital_energy_weights["eps_o"],
            state.orbital_energy_weights["eps_v"],
        )
    )

    def objective(sign: float, step: float) -> float:
        fock = _direct_fields(
            raw["h"] + sign * step * dh,
            raw["g"] + sign * step * dg,
            raw["rotation"] + sign * step * du,
            state.reference.nocc,
        )["fock"]
        return float(np.sum(source * np.diag(fock)))

    for step in (1.0e-4, 3.0e-5, 1.0e-5):
        np.testing.assert_allclose(
            (objective(1.0, step) - objective(-1.0, step)) / (2 * step),
            analytic,
            atol=3e-8,
            rtol=3e-7,
        )


def test_denominator_source_changes_orbital_and_overlap_response(
    water_state: BoundCCSDTOrbitalResponse,
) -> None:
    state = water_state
    denominator = state.component_weights["triples_denominator"]
    assert np.linalg.norm(denominator["hcore"]) > 1.0e-10
    assert np.linalg.norm(denominator["overlap"]) > 1.0e-10
    assert np.linalg.norm(denominator["orbital_rhs"]) > 1.0e-10

    without_denominator_rhs = state.orbital_rhs - denominator["orbital_rhs"].reshape(-1)
    assert np.linalg.norm(state.orbital_rhs - without_denominator_rhs) > 1.0e-10
    without_denominator_overlap = (
        state.correlation_weights["overlap"] - denominator["overlap"]
    )
    assert (
        np.linalg.norm(
            state.correlation_weights["overlap"] - without_denominator_overlap
        )
        > 1.0e-10
    )

    sources = state.orbital_energy_weights
    np.testing.assert_allclose(
        np.sum(sources["eps_o"]) + np.sum(sources["eps_v"]),
        0.0,
        atol=1.0e-14,
        rtol=0,
    )


def test_response_state_is_immutable_and_does_not_claim_forces(
    water_state: BoundCCSDTOrbitalResponse,
) -> None:
    state = water_state
    assert state.response.corrected.provenance["orbital_response"] == "excluded"
    assert not hasattr(state, "gradient")
    assert not hasattr(state, "forces")
    assert not hasattr(state.baseline, "gradient")
    assert not hasattr(state.baseline, "ao_program")
    assert not hasattr(state.baseline, "z_result")
    assert not hasattr(state.baseline, "logical_reserved_host_bytes")
    for value in (
        state.orbital_rhs,
        state.weights["hcore"],
        state.weights["eri"],
        state.weights["overlap"],
    ):
        assert not value.flags.writeable
        with pytest.raises(ValueError):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        state.response_identity = "forged"


def test_degenerate_same_space_gap_is_explicitly_out_of_scope() -> None:
    assert _minimum_same_space_gap(np.array([-1.0, 0.5]), 1) == float("inf")
    assert _minimum_same_space_gap(np.array([-1.0, -1.0, 0.5]), 2) == 0.0


@pytest.mark.parametrize("o,v", ((2, 2), (2, 3)))
def test_same_space_canonicalization_cotangent_cancels_generated_stationarity(
    o: int, v: int
) -> None:
    n = o + v
    _, g, _, _ = random_case(o, v, 159)
    rotation = np.eye(n)
    fock_without_h = _direct_fields(np.zeros((n, n)), g, rotation, o)["fock"]
    energies = np.linspace(-1.4, 0.8, n)
    h = np.diag(energies) - fock_without_h
    np.testing.assert_allclose(
        _direct_fields(h, g, rotation, o)["fock"],
        np.diag(energies),
        atol=2e-12,
        rtol=0,
    )

    target = np.zeros((n, n))
    for start, stop, value in ((0, o, 0.37), (o, n, -0.23)):
        if stop - start >= 2:
            target[start, start + 1] = value
            target[start + 1, start] = -value

    bar_fock = _same_space_fock_cotangent(target, energies, o)
    np.testing.assert_allclose(bar_fock, bar_fock.T, atol=0, rtol=0)
    np.testing.assert_array_equal(bar_fock[:o, o:], 0.0)
    np.testing.assert_array_equal(bar_fock[o:, :o], 0.0)

    generated = execute(
        build_fock_weight_program(o, v),
        {
            "h": h,
            "g": g,
            "rotation": rotation,
            "bar_fock": bar_fock,
        },
    ).outputs["stationarity"]
    np.testing.assert_allclose(
        generated[:o, :o],
        -target[:o, :o],
        atol=3e-11,
        rtol=3e-11,
    )
    np.testing.assert_allclose(
        generated[o:, o:],
        -target[o:, o:],
        atol=3e-11,
        rtol=3e-11,
    )


@pytest.mark.parametrize("o,v", [(2, 2), (3, 3)])
def test_same_space_multiplier_matches_independent_eigenvector_response(
    o: int, v: int
) -> None:
    """Differentiate a canonical gauge by eigh, not by the generated Fock VJP."""
    rng = np.random.default_rng(709)
    energies = np.linspace(-2.3, 3.1, o + v)
    raw = rng.normal(size=(o + v, o + v))
    stationarity = raw - raw.T
    direction = np.zeros_like(stationarity)
    for start, stop in ((0, o), (o, o + v)):
        block = rng.normal(scale=0.05, size=(stop - start, stop - start))
        direction[start:stop, start:stop] = block + block.T
    multiplier = _same_space_fock_cotangent(stationarity, energies, o)
    expected = float(np.sum(multiplier * direction))

    def gauge_objective(step: float) -> float:
        result = 0.0
        for start, stop in ((0, o), (o, o + v)):
            matrix = (
                np.diag(energies[start:stop]) + step * direction[start:stop, start:stop]
            )
            _, vectors = np.linalg.eigh(matrix)
            vectors *= np.where(np.diag(vectors) < 0.0, -1.0, 1.0)
            for p in range(stop - start):
                for q in range(p + 1, stop - start):
                    result += stationarity[start + p, start + q] * vectors[p, q]
        return result

    for step in (1e-4, 3e-5, 1e-5):
        actual = (gauge_objective(step) - gauge_objective(-step)) / (2 * step)
        np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-8)
    np.testing.assert_array_equal(multiplier[:o, o:], 0.0)
    np.testing.assert_array_equal(np.diag(multiplier), 0.0)
