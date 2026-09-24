"""Public native RCCSD acceptance for #149 C."""

from __future__ import annotations

import ctypes as ct
import json
import os
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc import Calculator, _native, method_capabilities

ROOT = Path(__file__).resolve().parents[2]
REFERENCES = {
    "h2": ROOT / "tests/reference_data/cc/gradients/h2_shifted.json",
    "h2o": ROOT / "tests/reference_data/cc/gradients/h2o.json",
}


def _reference_case(
    name: str = "h2",
) -> tuple[list[tuple[int, list[float]]], dict[str, typing.Any]]:
    record = json.loads(REFERENCES[name].read_text())
    inputs = record["inputs"]
    atoms = list(zip(inputs["atomic_numbers"], inputs["coordinates"], strict=True))
    return atoms, record


def _calculator(device: str = "cpu", **kwargs: typing.Any) -> Calculator:
    options = {
        "method": "rccsd",
        "basis": "sto-3g",
        "device": device,
        "max_iterations": 200,
        "energy_tolerance": 1e-13,
        "density_tolerance": 1e-11,
        "ccsd_max_iterations": 150,
        "ccsd_energy_tolerance": 1e-13,
        "ccsd_residual_tolerance": 1e-11,
    }
    options.update(kwargs)
    return Calculator(**options)


@pytest.fixture(params=("cpu", "cuda"))
def device(request: pytest.FixtureRequest) -> str:
    if request.param == "cuda" and os.environ.get("VIBEQC_RCCSD_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated RTX/CUDA native library")
    return request.param


def test_native_rccsd_capability_is_honest_energy_only_batch() -> None:
    caps = method_capabilities("rccsd")
    assert caps.family == "coupled_cluster"
    assert caps.available and caps.supports_batch
    assert caps.supported_properties == frozenset({"energy"})
    triples = method_capabilities("ccsd(t)")
    assert triples.available and triples.supports_batch
    assert triples.supported_properties == frozenset({"energy", "forces"})


@pytest.mark.parametrize("case", ("h2", "h2o"))
def test_public_native_rccsd_matches_pinned_pyscf_endpoint(
    device: str, case: str
) -> None:
    atoms, reference = _reference_case(case)
    result = _calculator(device).singlepoint(atoms, properties=("energy",))
    diag = result.correlation
    assert result.converged and result.forces is None
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert abs(result.energy - reference["total_energy"]) <= 2e-9
    assert abs(diag.ccsd_correlation_energy - reference["correlation_energy"]) <= 2e-9
    assert diag.ccsd_iterations <= 150
    assert diag.ccsd_energy_change <= 1e-13
    assert diag.ccsd_replay_singles_residual_max <= 1e-11
    assert diag.ccsd_replay_doubles_residual_max <= 1e-11
    assert len(diag.equation_hash) == 64
    assert len(diag.ccsd_replay_equation_hash) == 64
    assert diag.equation_hash != diag.ccsd_replay_equation_hash
    assert 0 < diag.minimum_absolute_denominator
    assert diag.numeric_capacity_bytes <= 256 << 20
    if device == "cuda":
        assert diag.correlation_owned_device_bytes > 0
        assert diag.ccsd_setup_h2d_bytes > 0
        assert diag.ccsd_scalar_d2h_bytes > 0
        assert diag.ccsd_amplitude_d2h_bytes > 0
        assert diag.ccsd_synchronizations >= diag.ccsd_iterations
        assert diag.mo_host_staging
    else:
        assert diag.correlation_owned_device_bytes == 0
        assert diag.ccsd_setup_h2d_bytes == 0
        assert diag.ccsd_amplitude_d2h_bytes == 0
        assert not diag.mo_host_staging


def test_public_rccsd_zero_diis_uses_native_jacobi() -> None:
    atoms, reference = _reference_case()
    result = _calculator(ccsd_diis_history=0).singlepoint(atoms, properties=("energy",))
    assert result.converged
    assert result.energy == pytest.approx(reference["total_energy"], abs=2e-9)
    assert result.correlation.ccsd_diis_restarts == 0


def test_public_rccsd_rejects_unsupported_reference_force_precision_df_and_frozen_core() -> (
    None
):
    atoms, _ = _reference_case()
    calc = _calculator()
    with pytest.raises(ValueError, match=r"does not support.*forces"):
        calc.singlepoint(atoms, properties=("energy", "forces"))
    with pytest.raises(NotImplementedError, match=r"closed-shell"):
        calc.singlepoint(atoms, multiplicity=3, properties=("energy",))
    with pytest.raises(ValueError, match=r"precision=.*fp64"):
        Calculator(method="rccsd", precision="auto")
    with pytest.raises(NotImplementedError, match=r"frozen-core"):
        Calculator(method="rccsd", ccsd_frozen_core=1)
    with pytest.raises(NotImplementedError, match=r"density fitting"):
        Calculator(method="rccsd", density_fitting="cpu")


def test_public_rccsd_budget_nonconvergence_and_denominator_failures_are_diagnosable() -> (
    None
):
    atoms, _ = _reference_case()
    with pytest.raises(RuntimeError, match=r"memory budget|error 7"):
        _calculator(correlation_memory_budget_bytes=1024).singlepoint(
            atoms, properties=("energy",)
        )
    with pytest.raises(RuntimeError, match=r"maximum RCCSD iterations|conver"):
        _calculator(ccsd_max_iterations=1).singlepoint(atoms, properties=("energy",))
    with pytest.raises(RuntimeError, match=r"denominator"):
        _calculator(ccsd_denominator_threshold=100.0).singlepoint(
            atoms, properties=("energy",)
        )


def test_public_rccsd_homogeneous_batch_repeats_and_isolates_partial_failure(
    device: str,
) -> None:
    atoms, reference = _reference_case()
    moved = [(z, (xyz[0], xyz[1], xyz[2] + 0.03)) for z, xyz in atoms]
    calc = _calculator(device)
    with calc.prepare_batch([atoms, moved]) as prepared:
        first = prepared.execute(strict=True)
        assert all(item.converged for item in first.items)
        assert abs(first.items[0].energy - reference["total_energy"]) <= 2e-9
        assert all(item.correlation is not None for item in first.items)
        assert all(
            item.correlation.ccsd_replay_doubles_residual_max <= 1e-11
            for item in first.items
        )
        partial = prepared.execute([np.zeros((1, 3)), None])
        assert partial.failure_indices == (0,)
        assert partial.items[0].correlation is None
        assert partial.items[1].correlation is not None
        assert partial.items[1].converged
        assert partial.items[1].energy == pytest.approx(
            first.items[1].energy, abs=2e-10
        )
        repeated = prepared.execute(strict=True)
        assert tuple(item.energy for item in repeated.items) == pytest.approx(
            tuple(item.energy for item in first.items), abs=2e-10
        )


def test_public_rccsd_rejects_ragged_prepared_batch() -> None:
    h2, _ = _reference_case()
    water = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, 1.43, 1.11)),
        ("H", (0.0, -1.43, 1.11)),
    ]
    with pytest.raises(NotImplementedError, match=r"homogeneous.*nocc,nvir|ragged"):
        _calculator().prepare_batch([h2, water])


def _prepare_c_owner(
    calc: Calculator, atoms: typing.Any
) -> tuple[ct.c_void_p, ct.c_void_p]:
    from vibeqc import Atom

    atoms = tuple(Atom(int(z), tuple(xyz)) for z, xyz in atoms)
    lib = calc._library
    context = ct.c_void_p()
    system = ct.c_void_p()
    calculation = ct.c_void_p()
    descriptor = calc._context_descriptor()
    _native.check(
        lib, lib.vibeqc_context_create(ct.byref(descriptor), ct.byref(context))
    )
    try:
        system = calc._create_native_system(context, atoms, 0, 1)
        method = calc._method_descriptor()
        _native.check(
            lib,
            lib.vibeqc_calculation_prepare(
                context, system, ct.byref(method), ct.byref(calculation)
            ),
            context=context,
        )
    finally:
        if system:
            lib.vibeqc_system_destroy(system)
    return context, calculation


def _execute_c_owner(
    calc: Calculator, context: ct.c_void_p, calculation: ct.c_void_p
) -> tuple[_native.ResultDescriptor, _native.CorrelationDiagnostic]:
    out = _native.ResultDescriptor()
    out.struct_size = ct.sizeof(out)
    out.abi_version = _native.ABI_VERSION
    _native.check(
        calc._library,
        calc._library.vibeqc_calculation_execute(calculation, ct.byref(out)),
        context=context,
    )
    diag = _native.CorrelationDiagnostic()
    diag.struct_size = ct.sizeof(diag)
    diag.abi_version = _native.ABI_VERSION
    _native.check(
        calc._library,
        calc._library.vibeqc_calculation_get_correlation_diagnostic(
            calculation, ct.byref(diag)
        ),
        context=context,
    )
    return out, diag


def test_c_api_rccsd_nonconvergence_retains_last_finite_diagnostic() -> None:
    atoms, _ = _reference_case()
    calc = _calculator(ccsd_max_iterations=1)
    context, calculation = _prepare_c_owner(calc, atoms)
    try:
        out = _native.ResultDescriptor()
        out.struct_size = ct.sizeof(out)
        out.abi_version = _native.ABI_VERSION
        status = calc._library.vibeqc_calculation_execute(calculation, ct.byref(out))
        assert status == _native.STATUS_NOT_CONVERGED
        assert not out.converged and np.isfinite(out.energy)

        diag = _native.CorrelationDiagnostic()
        diag.struct_size = ct.sizeof(diag)
        diag.abi_version = _native.ABI_VERSION
        _native.check(
            calc._library,
            calc._library.vibeqc_calculation_get_correlation_diagnostic(
                calculation, ct.byref(diag)
            ),
            context=context,
        )
        assert diag.ccsd_iterations == 2
        assert np.isfinite(diag.ccsd_correlation_energy)
        assert diag.ccsd_singles_residual_max > 0
        assert diag.ccsd_doubles_residual_max > 0
    finally:
        calc._library.vibeqc_calculation_destroy(calculation)
        calc._library.vibeqc_context_destroy(context)


def test_c_api_rccsd_owner_outlives_input_system_repeats_and_isolates_two_contexts() -> (
    None
):
    atoms, reference = _reference_case()
    calc = _calculator()
    owners = [_prepare_c_owner(calc, atoms) for _ in range(2)]
    try:
        energies = []
        for context, calculation in owners:
            first, diag = _execute_c_owner(calc, context, calculation)
            second, second_diag = _execute_c_owner(calc, context, calculation)
            assert first.energy == pytest.approx(second.energy, abs=2e-12)
            assert diag.ccsd_correlation_energy == pytest.approx(
                second_diag.ccsd_correlation_energy, abs=2e-12
            )
            assert diag.ccsd_replay_singles_residual_max <= 1e-11
            assert diag.ccsd_replay_doubles_residual_max <= 1e-11
            energies.append(first.energy)
        assert energies == pytest.approx(
            [reference["total_energy"], reference["total_energy"]], abs=2e-9
        )
    finally:
        for context, calculation in owners:
            if calculation:
                calc._library.vibeqc_calculation_destroy(calculation)
            if context:
                calc._library.vibeqc_context_destroy(context)
