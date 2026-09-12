"""Public native HF-to-MP2 energy acceptance, separate from fixture consumers."""

import ctypes as ct
import os
import time
from pathlib import Path

import numpy as np
import pytest
from vibeqc import (
    Calculator,
    ObservableTarget,
    ResourceBudget,
    TargetAccuracy,
    _native,
    method_capabilities,
)

from tools.vibeqc_posthf.fixtures import load_fixture, source_arguments


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1":
        pytest.skip("requires explicitly allocated CUDA device and native library")
    return request.param


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
def test_public_native_hf_to_mp2_components(name, device):
    meta, arrays = load_fixture(name)
    args = source_arguments(meta)
    started = time.perf_counter()
    calc = Calculator(
        method="mp2",
        basis=args["basis"],
        basis_representation=args["representation"],
        device=device,
    )
    result = calc.singlepoint(args["atoms"], charge=args["charge"])
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


def test_public_mp2_rejects_unimplemented_controls():
    target = TargetAccuracy(
        (ObservableTarget("energy", "absolute", "Eh", absolute=1e-6),)
    )
    with pytest.raises(NotImplementedError, match="target_accuracy"):
        Calculator(method="mp2", target_accuracy=target)
    with pytest.raises(NotImplementedError, match="resource_budget"):
        Calculator(method="mp2", resource_budget=ResourceBudget())
    with pytest.raises(ValueError, match="precision=.*fp64"):
        Calculator(method="mp2", precision="auto")
    calculator = Calculator(method="mp2")
    with pytest.raises(NotImplementedError, match="resource planning"):
        calculator.estimate_resources([[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]])
    with pytest.raises(NotImplementedError, match="accuracy model resolution"):
        calculator.resolved_model([("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))])


def test_public_mp2_identity_includes_correlation_controls():
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


def test_public_unsupported_budget_scf_and_neighbors(device):
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calc = Calculator(method="mp2", device=device)
    assert method_capabilities("mp2").supported_properties == frozenset({"energy"})
    with pytest.raises(NotImplementedError, match="forces"):
        calc.singlepoint(atoms, properties=("energy", "forces"))
    with pytest.raises(RuntimeError, match="error 7|memory budget"):
        Calculator(
            method="mp2", device=device, correlation_memory_budget_bytes=1024
        ).singlepoint(atoms)
    with pytest.raises(RuntimeError, match="converge"):
        Calculator(method="mp2", device=device, max_iterations=1).singlepoint(atoms)
    with pytest.raises(RuntimeError, match="near-zero"):
        Calculator(
            method="mp2", device=device, mp2_denominator_threshold=100
        ).singlepoint(atoms)
    with pytest.raises(NotImplementedError, match="closed-shell"):
        calc.singlepoint(atoms, multiplicity=3)
    with pytest.raises(NotImplementedError, match="RI/DF"):
        Calculator(method="mp2", device=device, density_fitting="cpu").singlepoint(
            atoms
        )
    first = calc.singlepoint(atoms)
    changed = calc.singlepoint([("H", (0, 0, -0.8)), ("H", (0, 0, 0.8))])
    assert abs(first.energy - changed.energy) > 1e-6
    assert calc.singlepoint(atoms).energy == first.energy


def test_c_api_force_request_never_returns_success_or_writes_placeholder(device):
    calc = Calculator(method="mp2", device=device)
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
        assert (
            lib.vibeqc_calculation_execute(calculation, ct.byref(out))
            == _native.STATUS_NOT_IMPLEMENTED
        )
        assert out.energy == 987.0 and list(forces) == [123.0] * 6
        energy_only = _native.ResultDescriptor(
            ct.sizeof(_native.ResultDescriptor), 0, 0, None, 0, 0, 0, 0, 0, 0
        )
        _native.check(
            lib,
            lib.vibeqc_calculation_execute(calculation, ct.byref(energy_only)),
            context=context,
        )
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
        assert (
            lib.vibeqc_calculation_execute(calculation, ct.byref(out))
            == _native.STATUS_NOT_IMPLEMENTED
        )
        assert (
            lib.vibeqc_calculation_get_correlation_diagnostic(
                calculation, ct.byref(diag)
            )
            == _native.STATUS_NOT_IMPLEMENTED
        )
        # Ctypes releases the GIL during native calls. Both plans and error
        # state share one Context; its serialization must cover both writes.
        from concurrent.futures import ThreadPoolExecutor

        def concurrent_call(k):
            force = (ct.c_double * 6)(*([123.0] * 6)) if k % 2 else None
            value = _native.ResultDescriptor(
                ct.sizeof(_native.ResultDescriptor),
                0,
                0,
                force,
                6 if force is not None else 0,
                0,
                0,
                0,
                0,
                0,
            )
            status = lib.vibeqc_calculation_execute(calculation, ct.byref(value))
            detail = lib.vibeqc_context_last_error(context)
            assert isinstance(detail, bytes)
            assert status == (
                _native.STATUS_NOT_IMPLEMENTED if force is not None else 0
            )
            if force is not None:
                assert list(force) == [123.0] * 6
            else:
                assert abs(value.energy - energy_only.energy) < 1e-12

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(concurrent_call, range(8)))
    finally:
        if calculation:
            lib.vibeqc_calculation_destroy(calculation)
        if system:
            lib.vibeqc_system_destroy(system)
        lib.vibeqc_context_destroy(context)


def test_generated_cpu_and_capacity_sources_are_reproducible():
    from tools.generate_mp2_native import cpu_header
    from tools.vibeqc_posthf.plan_spec import native_header

    root = Path(__file__).resolve().parents[2]
    for name, expected in (
        ("mp2_cpu_generated.hpp", cpu_header()),
        ("block_capacity_generated.hpp", native_header()),
    ):
        actual = (root / "src/posthf" / name).read_text()
        import re

        def tokens(value):
            return "".join(re.sub(r"//[^\n]*", "", value).split())

        assert tokens(actual) == tokens(expected)


def test_native_cuda_generation_keeps_each_architecture_and_tile_distinct(tmp_path):
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
