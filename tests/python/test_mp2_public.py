"""Public native HF-to-MP2 energy acceptance, separate from fixture consumers."""

import ctypes as ct
import os
import time
import typing
from pathlib import Path

import numpy as np
import pytest
from vibeqc import (
    Calculator,
    ObservableTarget,
    Primitive,
    ResourceBudget,
    Shell,
    TargetAccuracy,
    _native,
    method_capabilities,
)

from tools.vibeqc_posthf.fixtures import load_fixture, source_arguments


@pytest.fixture(params=["cpu", "cuda"])
def device(request: typing.Any) -> typing.Any:
    if request.param == "cuda" and os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated CUDA device and native library")
    return request.param


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
def test_public_native_hf_to_mp2_components(
    name: typing.Any, device: typing.Any
) -> None:
    meta, arrays = load_fixture(name)
    args = source_arguments(meta)
    started = time.perf_counter()
    calc = Calculator(
        method="mp2",
        basis=args["basis"],
        basis_representation=args["representation"],
        device=device,
    )
    result = calc.singlepoint(
        args["atoms"], charge=args["charge"], properties=("energy",)
    )
    ref = meta["records"]["conventional"]
    no = ref["electron_count"] // 2
    n = len(arrays["conventional_eps"])
    g = arrays["conventional_mo"][
        np.ix_(range(no), range(no, n), range(no), range(no, n))
    ].transpose(0, 2, 1, 3)
    t = arrays["conventional_t2"]
    os_ref, ss_ref = float(np.sum(t * g)), float(np.sum(t * (g - g.swapaxes(2, 3))))
    diag = result.correlation
    assert result.forces is None and result.converged
    assert result.executed_backend == ("cuda" if device == "cuda" else "cpu_reference")
    assert abs(result.energy - ref["hf_energy"] - ref["correlation_energy"]) <= 1e-9
    np.testing.assert_allclose(
        [diag.opposite_spin_energy, diag.same_spin_energy],
        [os_ref, ss_ref],
        atol=1e-11,
        rtol=1e-10,
    )
    assert diag.minimum_absolute_denominator > 1e-10
    assert diag.reference_residual <= 1e-8
    assert diag.numeric_capacity_bytes <= 256 << 20
    assert diag.mo_host_staging == (device == "cuda")
    if directory := os.environ.get("VIBEQC_MP2_EVIDENCE_DIR"):
        from tools.vibeqc_mp2.evidence import record_public_result

        record_public_result(
            calc,
            result,
            meta,
            os_ref,
            ss_ref,
            time.perf_counter() - started,
            Path(directory) / f"{name}-{device}.json",
        )


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
def test_public_native_df_hf_to_ri_mp2_components(
    name: typing.Any, device: typing.Any
) -> None:
    meta, arrays = load_fixture(name)
    args = source_arguments(meta)
    calc = Calculator(
        method="mp2",
        basis=args["basis"],
        auxiliary_basis=args["auxiliary_basis"],
        basis_representation=args["representation"],
        density_fitting="cuda" if device == "cuda" else "cpu",
        device=device,
    )
    result = calc.singlepoint(
        args["atoms"], charge=args["charge"], properties=("energy",)
    )
    ref = meta["records"]["df"]
    no = ref["electron_count"] // 2
    n = len(arrays["df_eps"])
    g = arrays["df_mo"][
        np.ix_(range(no), range(no, n), range(no), range(no, n))
    ].transpose(0, 2, 1, 3)
    t = arrays["df_t2"]
    os_ref = float(np.sum(t * g))
    ss_ref = float(np.sum(t * (g - g.swapaxes(2, 3))))
    assert result.forces is None and result.converged
    assert abs(result.energy - ref["hf_energy"] - ref["correlation_energy"]) <= 1e-9
    np.testing.assert_allclose(
        [result.correlation.opposite_spin_energy, result.correlation.same_spin_energy],
        [os_ref, ss_ref],
        atol=1e-11,
        rtol=1e-10,
    )
    assert result.correlation.minimum_absolute_denominator > 1e-10
    assert result.correlation.numeric_capacity_bytes <= 256 << 20


def test_public_mp2_rejects_unimplemented_controls() -> None:
    target = TargetAccuracy(
        (ObservableTarget("energy", "absolute", "Eh", absolute=1e-6),)
    )
    with pytest.raises(NotImplementedError, match=r"target_accuracy"):
        Calculator(method="mp2", target_accuracy=target)
    with pytest.raises(NotImplementedError, match=r"resource_budget"):
        Calculator(method="mp2", resource_budget=ResourceBudget())
    with pytest.raises(ValueError, match=r"precision=.*fp64"):
        Calculator(method="mp2", precision="auto")
    calculator = Calculator(method="mp2")
    with pytest.raises(NotImplementedError, match=r"resource planning"):
        calculator.estimate_resources([[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]])
    assert (
        calculator.resolved_model([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]).method
        == "mp2"
    )


def test_hf_identity_ignores_mp2_only_controls() -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    for method in ("rhf", "uhf"):
        identities = {
            Calculator(method=method, **options).basis_metadata(atoms)["model_identity"]
            for options in (
                {},
                {"correlation_memory_budget_bytes": 128 << 20},
                {"mp2_denominator_threshold": 1e-8},
            )
        }
        assert len(identities) == 1


@pytest.mark.parametrize("mode", ["cpu", "cpu_reference", "cuda", "auto", True])
def test_mp2_model_resolves_density_fitting(mode: typing.Any) -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    model = Calculator(method="mp2", density_fitting=mode).resolved_model(atoms)
    assert model.method == "mp2" and model.approximation == "density_fitting"
    assert model.auxiliary_basis_hash and model.metric_relative_threshold == 1e-10


def test_mp2_model_can_be_reconstructed_as_density_fitting() -> None:
    from dataclasses import replace

    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    model = Calculator(method="mp2").resolved_model(atoms)
    fitted = replace(
        model,
        approximation="density_fitting",
        auxiliary_basis_hash=model.basis_hash,
        metric_relative_threshold=1e-10,
    )
    assert fitted.method == "mp2" and fitted.approximation == "density_fitting"


def test_public_mp2_identity_includes_correlation_controls() -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    # Metadata must distinguish changed execution controls even before native
    # work begins, just as it does for the SCF tolerance and precision policy.
    identities = {
        Calculator(method="mp2", **options).basis_metadata(atoms)["model_identity"]
        for options in (
            {},
            {"correlation_memory_budget_bytes": 128 << 20},
            {"mp2_denominator_threshold": 1e-8},
        )
    }
    assert len(identities) == 3


def test_ri_mp2_composes_df_reference_capacity_before_allocation() -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    auxiliary = tuple(
        Shell(index % 2, 0, (Primitive(0.05 + 0.01 * index, 1.0),))
        for index in range(200)
    )
    # Each standalone reference/correlation estimate fits in this window, but
    # the CPU DF plan remains live alongside the large DIIS/reference state.
    with pytest.raises(RuntimeError, match="RI-MP2 DF reference state"):
        Calculator(
            method="mp2",
            basis="sto-3g",
            auxiliary_basis=auxiliary,
            density_fitting="cpu",
            diis_history=170_000,
            correlation_memory_budget_bytes=19_000 << 10,
        ).singlepoint(atoms)
    accepted = Calculator(
        method="mp2",
        basis="sto-3g",
        auxiliary_basis=auxiliary,
        density_fitting="cpu",
        diis_history=170_000,
        correlation_memory_budget_bytes=19_240 << 10,
    ).singlepoint(atoms)
    assert 19_000 << 10 < accepted.correlation.numeric_capacity_bytes <= 19_240 << 10


def test_cpu_ri_mp2_accepts_supported_g_auxiliary_capacity() -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    sto3g = (
        Primitive(3.42525091, 0.1543289673),
        Primitive(0.62391373, 0.5353281423),
        Primitive(0.1688554, 0.4446345422),
    )
    auxiliary = (
        Shell(0, 0, sto3g),
        Shell(1, 0, sto3g),
        Shell(0, 4, (Primitive(0.5, 1.0),)),
    )
    result = Calculator(
        method="mp2", density_fitting="cpu", auxiliary_basis=auxiliary
    ).singlepoint(atoms)
    assert np.isfinite(result.energy) and result.correlation is not None


def test_cuda_calculator_uses_selected_cpu_df_basis_capability() -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    auxiliary = (
        Shell(0, 0, (Primitive(1.0, 1.0),)),
        Shell(1, 0, (Primitive(1.0, 1.0),)),
        Shell(0, 4, (Primitive(0.5, 1.0),)),
    )
    model = Calculator(
        method="mp2",
        device="cuda",
        density_fitting="cpu",
        auxiliary_basis=auxiliary,
    ).resolved_model(atoms)
    assert model.method == "mp2" and model.approximation == "density_fitting"
    for mode in ("cuda", "auto"):
        with pytest.raises(NotImplementedError, match=r"cuda/df_metric.*l<=3"):
            Calculator(
                method="mp2",
                device="cuda",
                density_fitting=mode,
                auxiliary_basis=auxiliary,
            ).resolved_model(atoms)


def test_public_unsupported_budget_scf_and_neighbors(device: typing.Any) -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calc = Calculator(method="mp2", device=device)
    assert method_capabilities("mp2").supported_properties == frozenset(
        {"energy", "forces"}
    )
    with pytest.raises(RuntimeError, match="error 7|memory budget"):
        Calculator(
            method="mp2", device=device, correlation_memory_budget_bytes=1024
        ).singlepoint(atoms)
    with pytest.raises(RuntimeError, match="error 7|memory budget"):
        Calculator(
            method="mp2",
            device=device,
            density_fitting="cuda" if device == "cuda" else "cpu",
            correlation_memory_budget_bytes=1024,
        ).singlepoint(atoms)
    with pytest.raises(RuntimeError, match="RI-MP2 reference and correlation"):
        Calculator(
            method="mp2",
            device=device,
            density_fitting="cuda" if device == "cuda" else "cpu",
            correlation_memory_budget_bytes=12 << 20,
        ).singlepoint(atoms)
    with pytest.raises(RuntimeError, match="converge"):
        Calculator(method="mp2", device=device, max_iterations=1).singlepoint(atoms)
    with pytest.raises(RuntimeError, match="near-zero"):
        Calculator(
            method="mp2", device=device, mp2_denominator_threshold=100
        ).singlepoint(atoms)
    with pytest.raises(NotImplementedError, match="closed-shell"):
        calc.singlepoint(atoms, multiplicity=3)
    fitted = Calculator(
        method="mp2",
        device=device,
        density_fitting="cuda" if device == "cuda" else "cpu",
    ).singlepoint(atoms)
    assert (
        fitted.correlation is not None
        and fitted.correlation.minimum_absolute_denominator > 0
    )
    first = calc.singlepoint(atoms)
    changed = calc.singlepoint([("H", (0, 0, -0.8)), ("H", (0, 0, 0.8))])
    assert abs(first.energy - changed.energy) > 1e-6
    assert abs(calc.singlepoint(atoms).energy - first.energy) <= 1e-12


def test_public_force_reports_response_measurement_without_promoting_endpoint() -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calc = Calculator(method="mp2", basis="sto-3g", device="cpu")
    energy = calc.singlepoint(atoms, properties=("energy",)).correlation
    assert energy.measured_response_workspace_peak_bytes == 0
    assert energy.response_workspace_allocation_count == 0
    force = calc.singlepoint(atoms, properties=("energy", "forces")).correlation
    assert (
        0
        < force.measured_response_workspace_peak_bytes
        < force.response_workspace_bytes
    )
    assert force.response_workspace_allocation_count > 0
    # Response-only telemetry cannot satisfy complete-endpoint qualification.
    assert force.measured_endpoint_peak_bytes == 0


def test_public_conventional_mp2_force_cpu_matches_resolved_finite_difference() -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calc = Calculator(method="mp2", device="cpu")
    result = calc.singlepoint(atoms, properties=("energy", "forces"))
    assert result.converged and result.forces.shape == (2, 3)
    step = 1e-4
    plus = [("H", (0, 0, -0.7 + step)), atoms[1]]
    minus = [("H", (0, 0, -0.7 - step)), atoms[1]]
    finite = (
        calc.singlepoint(plus, properties=("energy",)).energy
        - calc.singlepoint(minus, properties=("energy",)).energy
    ) / (2 * step)
    assert abs(result.forces[0, 2] + finite) < 2e-6
    np.testing.assert_allclose(result.forces.sum(axis=0), 0.0, atol=2e-9)
    with pytest.raises(NotImplementedError, match=r"RI-MP2.*force"):
        Calculator(method="mp2", density_fitting="cpu").singlepoint(
            atoms, properties=("energy", "forces")
        )


def test_public_conventional_mp2_force_accepts_zero_derivative_degenerate_subspace() -> (
    None
):
    meta, _ = load_fixture("lih")
    args = source_arguments(meta)
    result = Calculator(
        method="mp2",
        basis=args["basis"],
        basis_representation=args["representation"],
        device="cpu",
    ).singlepoint(args["atoms"], charge=args["charge"], properties=("energy", "forces"))
    assert result.converged and np.isfinite(result.forces).all()
    np.testing.assert_allclose(result.forces.sum(axis=0), 0.0, atol=2e-8)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1",
    reason="requires explicitly allocated CUDA device and native library",
)
def test_public_conventional_mp2_force_cuda_matches_cpu() -> None:
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    cpu = Calculator(method="mp2", device="cpu").singlepoint(
        atoms, properties=("energy", "forces")
    )
    cuda = Calculator(method="mp2", device="cuda").singlepoint(
        atoms, properties=("energy", "forces")
    )
    assert cuda.executed_backend == "cuda"
    np.testing.assert_allclose(cuda.forces, cpu.forces, atol=2e-9, rtol=1e-9)
    assert cuda.correlation.derivative_workspace_bytes > 0
    assert (
        cuda.correlation.planned_endpoint_peak_bytes
        <= cuda.correlation.numeric_capacity_bytes
    )


def test_c_api_conventional_force_is_transactional_across_repeated_execution() -> None:
    calc = Calculator(method="mp2", device="cpu")
    lib = calc._library
    context, system, calculation = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
    from vibeqc import Atom

    atoms = (Atom(1, (0, 0, -0.7)), Atom(1, (0, 0, 0.7)))
    _native.check(
        lib,
        lib.vibeqc_context_create(
            ct.byref(calc._context_descriptor()), ct.byref(context)
        ),
    )
    try:
        system = calc._create_native_system(context, atoms, 0, 1)
        _native.check(
            lib,
            lib.vibeqc_calculation_prepare(
                context,
                system,
                ct.byref(calc._method_descriptor()),
                ct.byref(calculation),
            ),
        )
        forces = (ct.c_double * 6)(*([123.0] * 6))
        out = _native.ResultDescriptor(
            ct.sizeof(_native.ResultDescriptor), 0, 987.0, forces, 6, 0, 0, 0, 0, 0
        )
        _native.check(
            lib,
            lib.vibeqc_calculation_execute(calculation, ct.byref(out)),
            context=context,
        )
        first_energy = out.energy
        first_forces = list(forces)
        assert first_energy < 0 and first_forces != [123.0] * 6
        diag = _native.CorrelationDiagnostic()
        diag.struct_size = ct.sizeof(diag)
        diag.abi_version = 0
        _native.check(
            lib,
            lib.vibeqc_calculation_get_correlation_diagnostic(
                calculation, ct.byref(diag)
            ),
        )
        assert diag.opposite_spin_energy < 0
        assert diag.response_restarts <= diag.response_iterations
        assert np.isfinite(diag.response_absolute_residual)
        assert np.isfinite(diag.response_relative_residual)
        assert diag.response_absolute_residual < 1e-10
        assert 0.0 <= diag.response_relative_residual <= 1.0
        assert diag.response_workspace_bytes > 0
        assert (
            0
            < diag.measured_response_workspace_peak_bytes
            < diag.response_workspace_bytes
        )
        assert diag.response_workspace_allocation_count > 0
        assert diag.derivative_workspace_bytes > 0
        # This bounded slice has a plan but no endpoint allocation telemetry.
        assert diag.measured_endpoint_peak_bytes == 0
        assert diag.planned_endpoint_peak_bytes <= diag.numeric_capacity_bytes
        assert diag.force_provenance_flags == 0x7
        assert diag.response_operator_hash == b"rhf-canonical-response-v1"

        class LegacyCorrelationDiagnostic(ct.Structure):
            _fields_ = _native.CorrelationDiagnostic._fields_[:18]

        legacy_size = ct.sizeof(LegacyCorrelationDiagnostic)
        assert ct.sizeof(_native.CorrelationDiagnostic) > legacy_size
        storage = (ct.c_ubyte * (legacy_size + 32))(*([0xA5] * (legacy_size + 32)))
        legacy = ct.cast(storage, ct.POINTER(LegacyCorrelationDiagnostic))
        legacy.contents.struct_size = legacy_size
        legacy.contents.abi_version = _native.ABI_VERSION
        _native.check(
            lib,
            lib.vibeqc_calculation_get_correlation_diagnostic(
                calculation,
                ct.cast(legacy, ct.POINTER(_native.CorrelationDiagnostic)),
            ),
        )
        assert legacy.contents.opposite_spin_energy < 0
        assert bytes(storage[legacy_size:]) == bytes([0xA5] * 32)
        assert legacy.contents.struct_size == legacy_size
        _native.check(
            lib,
            lib.vibeqc_calculation_get_correlation_diagnostic(
                calculation, ct.cast(legacy, ct.POINTER(_native.CorrelationDiagnostic))
            ),
        )
        assert legacy.contents.struct_size == legacy_size
        assert bytes(storage[legacy_size:]) == bytes([0xA5] * 32)

        # The preceding B2 ABI prefix must also remain bounded by struct_size.
        ccsd_extension = next(
            index
            for index, (name, _ctype) in enumerate(
                _native.CorrelationDiagnostic._fields_
            )
            if name == "ccsd_iterations"
        )

        class PreviousB2Diagnostic(ct.Structure):
            _fields_ = _native.CorrelationDiagnostic._fields_[:ccsd_extension]

        previous_size = ct.sizeof(PreviousB2Diagnostic)
        storage = (ct.c_ubyte * (previous_size + 32))(*([0xA5] * (previous_size + 32)))
        previous = ct.cast(storage, ct.POINTER(PreviousB2Diagnostic))
        previous.contents.struct_size = previous_size
        previous.contents.abi_version = _native.ABI_VERSION
        _native.check(
            lib,
            lib.vibeqc_calculation_get_correlation_diagnostic(
                calculation,
                ct.cast(previous, ct.POINTER(_native.CorrelationDiagnostic)),
            ),
        )
        assert previous.contents.response_operator_hash == b"rhf-canonical-response-v1"
        assert bytes(storage[previous_size:]) == bytes([0xA5] * 32)
        assert previous.contents.struct_size == previous_size
        _native.check(
            lib,
            lib.vibeqc_calculation_get_correlation_diagnostic(
                calculation,
                ct.cast(previous, ct.POINTER(_native.CorrelationDiagnostic)),
            ),
        )
        assert previous.contents.struct_size == previous_size
        assert bytes(storage[previous_size:]) == bytes([0xA5] * 32)

        failed_forces = (ct.c_double * 6)(*([456.0] * 6))
        failed = _native.ResultDescriptor(
            ct.sizeof(_native.ResultDescriptor),
            0,
            654.0,
            failed_forces,
            5,
            11,
            12.0,
            13.0,
            14,
            15,
        )
        assert (
            lib.vibeqc_calculation_execute(calculation, ct.byref(failed))
            == _native.STATUS_INVALID_ARGUMENT
        )
        assert failed.energy == 654.0 and list(failed_forces) == [456.0] * 6
        assert (
            lib.vibeqc_calculation_get_correlation_diagnostic(
                calculation, ct.byref(diag)
            )
            == _native.STATUS_NOT_IMPLEMENTED
        )
        retry_forces = (ct.c_double * 6)(*([789.0] * 6))
        retry = _native.ResultDescriptor(
            ct.sizeof(_native.ResultDescriptor),
            0,
            321.0,
            retry_forces,
            6,
            0,
            0,
            0,
            0,
            0,
        )
        _native.check(
            lib,
            lib.vibeqc_calculation_execute(calculation, ct.byref(retry)),
            context=context,
        )
        assert abs(retry.energy - first_energy) < 1e-12
        np.testing.assert_allclose(list(retry_forces), first_forces, atol=1e-12)
    finally:
        if calculation:
            lib.vibeqc_calculation_destroy(calculation)
        if system:
            lib.vibeqc_system_destroy(system)
        lib.vibeqc_context_destroy(context)


def test_generated_cpu_and_capacity_sources_are_reproducible() -> None:
    from tools.generate_mp2_native import cpu_header
    from tools.vibeqc_posthf.plan_spec import native_header

    root = Path(__file__).resolve().parents[2]
    for name, expected in (
        ("mp2_cpu_generated.hpp", cpu_header()),
        ("block_capacity_generated.hpp", native_header()),
    ):
        actual = (root / "src/posthf" / name).read_text()
        import re

        def tokens(value: typing.Any) -> typing.Any:
            return "".join(re.sub(r"//[^\n]*", "", value).split())

        assert tokens(actual) == tokens(expected)


def test_native_cuda_generation_keeps_each_architecture_and_tile_distinct(
    tmp_path: typing.Any,
) -> None:
    """Exercise the actual generator in CPU CI before the expensive CUDA build."""
    from tools.generate_mp2_native import cuda_sources

    cuda_sources(tmp_path, "75;120-real;120-virtual")
    table = (tmp_path / "mp2_cuda_table.cu").read_text()
    assert len(list(tmp_path.glob("*_runtime.cu"))) == 8
    for arch in (75, 120):
        for tile in (1, 2, 4, 8):
            prefix = f"mp2_sm{arch}_t{tile}_"
            source = (tmp_path / f"{prefix}runtime.cu").read_text()
            assert f"namespace {prefix}generated" in source
            assert f'extern "C" int {prefix}tensor_create' in source
            assert f"{prefix}tensor_create,{prefix}tensor_destroy,{prefix}run" in table
            # The generic Tensor CUDA ABI supports distinct FP32/FP64 inputs.
            # C++ does not convert double** to void** at the adapter boundary.
            assert "const void* inputs[]={" in source
            assert "void* outputs[]={out,out+1};" in source
