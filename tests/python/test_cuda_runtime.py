import os

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell


def _cuda_tolerances() -> tuple[float, float]:
    # CuMetal emulates FP64 with paired FP32 arithmetic on current Apple Silicon.
    # Keep these tests as functional runtime gates there; the existing NVIDIA
    # CUDA regression tests retain the strict FP64 numerical tolerances.
    if os.environ.get("CUMETAL_ROOT"):
        return 2.0e-6, 2.0e-5
    return 2.0e-10, 2.0e-9


def test_cuda_minimal_rhf_matches_cpu_reference():
    """Exercise one real RHF CUDA calculation without batch/replay overhead."""

    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    basis = (
        Shell(0, 0, (Primitive(1.0, 1.0),)),
        Shell(1, 0, (Primitive(1.0, 1.0),)),
    )
    options = {
        "basis": basis,
        "energy_tolerance": 1.0e-10,
        "density_tolerance": 1.0e-8,
    }
    reference = Calculator(device="cpu", **options).singlepoint(atoms)
    try:
        result = Calculator(device="cuda", **options).singlepoint(atoms)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    energy_atol, force_atol = _cuda_tolerances()
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(reference.energy, abs=energy_atol)
    assert np.allclose(result.forces, reference.forces, atol=force_atol, rtol=0.0)


def test_cuda_minimal_uhf_matches_cpu_reference():
    """Exercise the unrestricted CUDA path with a one-electron one-shell case."""

    atoms = [("H", (0.0, 0.0, 0.0))]
    basis = (Shell(0, 0, (Primitive(1.0, 1.0),)),)
    options = {
        "method": "uhf",
        "basis": basis,
        "energy_tolerance": 1.0e-10,
        "density_tolerance": 1.0e-8,
    }
    reference = Calculator(device="cpu", **options).singlepoint(atoms, multiplicity=2)
    try:
        result = Calculator(device="cuda", **options).singlepoint(atoms, multiplicity=2)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    energy_atol, force_atol = _cuda_tolerances()
    assert result.executed_backend == "cuda"
    assert result.energy == pytest.approx(reference.energy, abs=energy_atol)
    assert np.allclose(result.forces, reference.forces, atol=force_atol, rtol=0.0)


@pytest.mark.parametrize("fixture_name", ("minimal_h2", "water", "water_sdf"))
def test_cuda_resident_rhf_response_matches_host_operator(fixture_name):
    """B2: keep RHF response operator/Krylov vectors resident on the CUDA stream."""
    if (
        os.environ.get("CUMETAL_ROOT")
        or os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1"
    ):
        pytest.skip(
            "resident RHF response requires explicitly allocated native NVIDIA CUDA"
        )
    from tools.vibeqc_posthf.export import export_rhf
    from tools.vibeqc_posthf.sources import NativeSource
    from tools.vibeqc_response import (
        CudaDirectJKBackend,
        GMRESOptions,
        RHFResponseOperator,
        solve,
    )

    atoms = [(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]
    basis = (
        Shell(0, 0, (Primitive(1.0, 1.0),)),
        Shell(1, 0, (Primitive(1.0, 1.0),)),
    )
    inputs = {"atoms": atoms, "basis": basis}
    if fixture_name != "minimal_h2":
        from tools.vibeqc_validation.hessian_fixtures import fixture_inputs

        # Non-square occupied/virtual blocks expose layout and gap-action bugs
        # that the original one-dimensional H2 response cannot exercise.
        inputs = fixture_inputs(fixture_name)
    with NativeSource(**inputs) as source:
        reference, _ = export_rhf(source, backend="cpu", tolerance=1e-12)
        with CudaDirectJKBackend(source, device_budget_bytes=64 << 20) as backend:
            problem = RHFResponseOperator.build_problem(reference, backend)
            operator = RHFResponseOperator(problem, backend)
            vector = np.linspace(0.2, 0.2 * problem.dimension, problem.dimension)
            expected_action = operator.apply(vector)
            host_result = solve(
                operator,
                np.ones(problem.dimension),
                options=GMRESOptions(rtol=1e-11, atol=1e-12),
                collect_basis=False,
            )
            host_result.require_converged()
            host_backend_actions = backend.statistics["actions"]

            with backend.resident_response(
                problem,
                vector_slots=128,
                device_budget_bytes=16 << 20,
            ) as resident:
                device_input = resident.from_host(vector)
                before = resident.diagnostics
                device_output = resident.apply(operator, device_input)
                after_apply = resident.diagnostics
                assert after_apply["h2d_bytes"] == before["h2d_bytes"]
                assert after_apply["d2h_bytes"] == before["d2h_bytes"] + 4
                actual_action = resident.to_host(device_output)
                device_input.release()
                device_output.release()

                action_atol, _ = _cuda_tolerances()
                np.testing.assert_allclose(
                    actual_action, expected_action, atol=10 * action_atol, rtol=0
                )
                assert backend.statistics["actions"] == host_backend_actions

                operator._krylov_engine = resident
                resident_result = solve(
                    operator,
                    np.ones(problem.dimension),
                    options=GMRESOptions(rtol=1e-11, atol=1e-12),
                    collect_basis=False,
                )
                resident_result.require_converged()
                np.testing.assert_allclose(
                    resident_result.solution,
                    host_result.solution,
                    atol=20 * action_atol,
                    rtol=0,
                )
                final = resident.diagnostics
                assert final["operator_actions"] >= resident_result.operator_actions
                assert final["blas_calls"] > 0
                assert final["owned_device_bytes"] <= 16 << 20
                # Agreement between two failed iterates is not solver acceptance.
                # Re-evaluate the final residual using the separate host-transform
                # operator, not the resident engine's own convergence report.
                rhs = np.ones(problem.dimension)
                residual = rhs - operator.apply(resident_result.solution)
                assert np.linalg.norm(residual) <= max(
                    1e-12, 1e-11 * np.linalg.norm(rhs)
                )
