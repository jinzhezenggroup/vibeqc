"""Real-GPU J/K inside the otherwise host directional nuclear response chain."""

import os
import typing

import numpy as np
import pytest

from tools.vibeqc_hessian import NativeRHFState, directional_rhf_response
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import NativeJKBackend
from tools.vibeqc_validation.hessian_fixtures import fixture_inputs

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated real GPU",
)


@pytest.mark.parametrize("name", ["h2", "water"])
def test_directional_native_response_with_cuda_jk_and_independent_D_W_differences(
    name: typing.Any, monkeypatch: typing.Any
) -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    with NativeSource(**fixture_inputs(name)) as source:
        s = NativeRHFState.from_source(source)
        v = np.random.default_rng(180).normal(size=(s.nat, 3))
        v /= np.linalg.norm(v)
        cpu = directional_rhf_response(s, v)
        values = []
        step = 3e-4
        for sign in (1, -1):
            atoms = [
                (a.atomic_number, xyz)
                for a, xyz in zip(source.atoms, s.coords + sign * step * v, strict=True)
            ]
            with NativeSource(
                atoms, basis=source.shells, charge=source.charge
            ) as other:
                displaced = NativeRHFState.from_source(other, tolerance=1e-13)
                c = displaced.C[:, : displaced.nocc]
                values.append(
                    (displaced.P0, (c * (2 * displaced.eps[: displaced.nocc])) @ c.T)
                )

        def forbidden(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            raise AssertionError(
                "CUDA metric/CPHF action used a CPU J/K or dense-input fallback"
            )

        monkeypatch.setattr(NativeRHFState, "first_order_inputs", property(forbidden))
        monkeypatch.setattr(source, "tile", forbidden)
        monkeypatch.setattr(source, "requests", forbidden)
        monkeypatch.setattr(source, "integral_derivatives", forbidden)
        monkeypatch.setattr(source, "rhf_density", forbidden)
        monkeypatch.setattr(NativeJKBackend, "coulomb_exchange", forbidden)
        actual = directional_rhf_response(s, v, jk_backend="cuda")
        for field, plus, minus in zip(
            ("density_derivative", "energy_weighted_density_derivative"),
            *values,
            strict=True,
        ):
            result = getattr(actual.response, field)
            np.testing.assert_allclose(
                result, getattr(cpu.response, field), atol=3e-9, rtol=3e-9
            )
            np.testing.assert_allclose(
                result, (plus - minus) / (2 * step), atol=2e-5, rtol=0
            )
        np.testing.assert_allclose(
            actual.response.rhs, cpu.response.rhs, atol=2e-10, rtol=2e-10
        )
        assert actual.diagnostics["jk_backend"] == "cuda"
        assert actual.diagnostics["first_integral_derivatives"] == "cpu-generated"
        assert actual.diagnostics["krylov_execution"] == "host"
        assert actual.diagnostics["jk_statistics"]["actions"] >= 2
        assert actual.response.solve_result.residual_norm < 1e-9
        assert not actual.diagnostics["molecular_hvp"]
        assert actual.diagnostics["first_order_matrix_bytes"] == 2 * s.nbf**2 * 8


def test_directional_cuda_budget_failure_and_valid_replay() -> None:
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        v = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
        with pytest.raises(MemoryError):
            directional_rhf_response(state, v, jk_backend="cuda", device_budget_bytes=1)
        result = directional_rhf_response(state, v, jk_backend="cuda")
        assert result.response.solve_result.converged


def test_directional_cuda_closes_plan_after_derivative_failure(
    monkeypatch: typing.Any,
) -> None:
    from tools.vibeqc_hessian import directional
    from tools.vibeqc_response import CudaDirectJKBackend

    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    with NativeSource(**fixture_inputs("h2")) as source:
        state = NativeRHFState.from_source(source)
        v = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]])
        closed = []
        original_close = CudaDirectJKBackend.close

        def close(owner: typing.Any) -> typing.Any:
            if owner._plan is not None:
                closed.append(owner.identity)
            return original_close(owner)

        def failed(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
            raise FloatingPointError("injected directional derivative failure")

        with monkeypatch.context() as patch:
            patch.setattr(CudaDirectJKBackend, "close", close)
            patch.setattr(directional, "generated_directional_first_order", failed)
            with pytest.raises(FloatingPointError, match="directional derivative"):
                directional_rhf_response(state, v, jk_backend="cuda")
        assert len(closed) == 1
        assert directional_rhf_response(
            state, v, jk_backend="cuda"
        ).response.solve_result.converged
