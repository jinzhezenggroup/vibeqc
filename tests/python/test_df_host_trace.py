"""Actual solver counts and fault-rejecting #206 host ledger gates."""

import copy
import json
import os
import typing

import pytest

from benchmarks.df_component_ledger import (
    aggregate_host,
    read_host_trace,
    validate_host_record,
)


def actual_calls(components: typing.Any, reason: typing.Any) -> typing.Any:
    """Work-elimination tests count leaves from either qualified provider."""
    return sum(
        components[key].get(reason, {}).get("calls", 0)
        for key in ("eigensolves_by_reason", "device_eigensolves_by_reason")
    )


def host_record() -> typing.Any:
    base = {"reason": "overlap", "item": 0, "nbf": 2, "finished": True, "failed": False}
    return {
        "schema": "vibeqc.df_host_trace",
        "version": 1,
        "id": 0,
        "valid": True,
        "regions": [
            dict(base, name="endpoint", parent=-1, wall_ms=5, cpu_ms=3),
            dict(base, name="reference_eigensolve", parent=0, wall_ms=2, cpu_ms=1),
        ],
    }


def test_host_ledger_counts_actual_leaves_and_keeps_clocks_separate(
    tmp_path: typing.Any,
) -> None:
    record = host_record()
    path = tmp_path / "host.jsonl"
    path.write_text(json.dumps(record) + "\n")
    summary = aggregate_host(read_host_trace(path))
    assert summary["exclusive_phases"]["endpoint"] == {
        "calls": 1,
        "wall_ms": 3,
        "cpu_ms": 2,
    }
    assert summary["eigensolves_by_reason"]["overlap"] == {
        "calls": 1,
        "failed_calls": 0,
        "wall_ms": 2,
        "cpu_ms": 1,
    }
    for row in record["regions"]:
        row["cpu_ms"] = None
    assert aggregate_host([record])["exclusive_phases"]["endpoint"]["cpu_ms"] is None
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="duplicate"):
        read_host_trace(path)


@pytest.mark.parametrize(
    "reason", ("overlap", "iteration", "final_fock", "seed_validation")
)
def test_device_solver_reasons_separate_setup_and_finalization(
    tmp_path: typing.Any, reason: typing.Any
) -> None:
    """Adding cold setup calls must not inflate the final-provider ablation."""
    record = host_record()
    record["regions"][1]["reason"] = reason
    record["regions"][1]["name"] = "device_eigensolve"
    path = tmp_path / "device.host.jsonl"
    path.write_text(json.dumps(record) + "\n")
    summary = aggregate_host(read_host_trace(path))
    assert summary["eigensolves_by_reason"] == {}
    assert summary["device_eigensolves_by_reason"][reason]["calls"] == 1
    assert summary["device_eigensolves"][0]["item"] == 0


@pytest.mark.parametrize("suffix", ("", "\n{"))
def test_host_trace_rejects_missing_record_terminator(
    tmp_path: typing.Any, suffix: typing.Any
) -> None:
    """Even valid JSON must carry the writer's final record terminator."""
    path = tmp_path / "interrupted.host.jsonl"
    path.write_text(json.dumps(host_record()) + suffix)
    with pytest.raises(ValueError, match="incomplete host trace"):
        read_host_trace(path)


@pytest.mark.parametrize(
    "key,value",
    [
        ("finished", False),
        ("wall_ms", 8),
        ("cpu_ms", float("nan")),
        ("parent", 1),
        ("nbf", 0),
        ("reason", "invented"),
    ],
)
def test_partial_or_mistimed_host_records_cannot_pass(
    key: typing.Any, value: typing.Any
) -> None:
    record = copy.deepcopy(host_record())
    record["regions"][1][key] = value
    with pytest.raises(ValueError):
        validate_host_record(record)


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_native_solver_calls_include_warm_preparation_and_finalization(
    tmp_path: typing.Any,
    monkeypatch: typing.Any,
    method: typing.Any,
    device: typing.Any,
) -> None:
    """Actual leaves detect reintroduced warm guesses, retaining other solves.

    Both spin modes still require actual physical-F validation; removing
    trace hooks must never satisfy the zero-reference-call gates.
    """
    if device == "cuda" and os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    from vibeqc import Calculator

    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    charge, spin = (0, 1) if method == "rhf" else (1, 2)
    calculator = Calculator(
        method=method,
        device=device,
        density_fitting=device,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch(
        [atoms, atoms], charges=[charge] * 2, multiplicities=[spin] * 2
    ) as batch:
        cold = batch.execute(strict=True, properties=("energy",))
        batch.set_warm_start_updates(False)
        path = tmp_path / "warm.host.jsonl"
        monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
        warm = batch.execute(strict=True, properties=("energy",))
        monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
        assert [r.energy for r in warm.items] == pytest.approx(
            [r.energy for r in cold.items], abs=1e-10
        )
        records = read_host_trace(path)
        assert any(r["regions"][0]["name"] == "batch_execute" for r in records)
        summary = aggregate_host(records)
        by_reason = summary["eigensolves_by_reason"]
        assert by_reason.get("overlap", {}).get("calls", 0) == (
            0 if device == "cuda" else 2
        )
        assert by_reason.get("core_guess", {}).get("calls", 0) == 0
        assert (
            sum(
                row["name"] == "initial_density"
                for record in records
                for row in record["regions"]
            )
            == 2
        )
        if device == "cuda":
            assert summary["exclusive_phases"]["overlap_cache_hit"]["calls"] == 2
            assert by_reason.get("final_fock", {}).get("calls", 0) == 0
            assert (
                summary["exclusive_phases"].get("device_eigensolve", {}).get("calls", 0)
                == 0
            )
            assert summary["exclusive_phases"]["final_state_fock_build"]["calls"] == 2
            assert summary["exclusive_phases"]["final_state_validation"]["calls"] == 2
            assert summary["exclusive_phases"]["final_state_reuse"]["calls"] == 2
            assert not by_reason.get("fallback", {}).get("calls", 0)
            device_leaves = [
                row
                for record in records
                for row in record["regions"]
                if row["name"] == "final_state_fock_build"
            ]
            assert {row["item"] for row in device_leaves} == {0, 1}
        before = path.read_bytes()
        batch.execute(strict=True, properties=("energy",))
        assert path.read_bytes() == before
        if device == "cuda":
            eager_path = tmp_path / "eager.host.jsonl"
            monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(eager_path))
            monkeypatch.setenv("VIBEQC_DF_EAGER_CORE_GUESS", "1")
            eager = batch.execute(strict=True, properties=("energy",))
            eager_components = aggregate_host(read_host_trace(eager_path))
            assert actual_calls(eager_components, "core_guess") == 2
            assert eager.energies == pytest.approx(warm.energies, abs=1e-10)


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
@pytest.mark.parametrize("budget", (0, 8 << 20))
def test_overlap_cache_survives_output_replans_and_isolates_changed_items(
    tmp_path: typing.Any,
    monkeypatch: typing.Any,
    method: typing.Any,
    representation: typing.Any,
    budget: typing.Any,
) -> None:
    """Same-sized neighbors keep separate X; geometry and failures cannot alias it."""
    if os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    import numpy as np
    from vibeqc import Calculator

    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calculator = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        density_fitting="cuda",
        density_fitting_memory_budget_bytes=budget,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    sequence = 0

    def traced(
        batch: typing.Any,
        coordinates: typing.Any = None,
        properties: typing.Any = ("energy", "forces"),
        strict: typing.Any = True,
    ) -> typing.Any:
        nonlocal sequence
        path = tmp_path / f"step-{sequence}.jsonl"
        sequence += 1
        monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
        try:
            result = batch.execute(coordinates, properties=properties, strict=strict)
        finally:
            monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
        return result, aggregate_host(read_host_trace(path))

    def overlap_calls(summary: typing.Any) -> typing.Any:
        return actual_calls(summary, "overlap")

    with calculator.prepare_batch([atoms, atoms]) as batch:
        cold, cold_trace = traced(batch)
        assert overlap_calls(cold_trace) == 2
        batch.set_warm_start_updates(False)
        for properties in (("energy",), ("energy", "forces")):
            warm, trace = traced(batch, properties=properties)
            assert overlap_calls(trace) == 0
            assert trace["exclusive_phases"]["overlap_cache_hit"]["calls"] == 2
            np.testing.assert_allclose(warm.energies, cold.energies, atol=1e-9, rtol=0)
            if len(properties) == 2:
                for a, b in zip(warm.items, cold.items, strict=True):
                    np.testing.assert_allclose(a.forces, b.forces, atol=1e-8, rtol=0)
        changed = np.array([atom[1] for atom in atoms])
        changed[-1, 2] += 0.03
        actual, trace = traced(batch, [None, changed])
        assert overlap_calls(trace) == 1
        assert trace["exclusive_phases"]["overlap_cache_hit"]["calls"] == 1
        changed_atoms = [
            (atom[0], tuple(xyz)) for atom, xyz in zip(atoms, changed, strict=True)
        ]
        with calculator.prepare_batch([atoms, changed_atoms]) as reference:
            expected = reference.execute(strict=True)
        np.testing.assert_allclose(
            actual.energies, expected.energies, atol=1e-9, rtol=0
        )
        for a, b in zip(actual.items, expected.items, strict=True):
            np.testing.assert_allclose(a.forces, b.forces, atol=1e-8, rtol=0)
        broken = changed.copy()
        broken[-1, 0] = np.nan
        failed, trace = traced(batch, [None, broken], strict=False)
        assert failed.items[0].status == 0
        assert failed.items[1].status != 0
        assert overlap_calls(trace) == 0
        recovered, trace = traced(batch, [None, changed])
        assert overlap_calls(trace) == 0
        np.testing.assert_allclose(
            recovered.energies, expected.energies, atol=1e-9, rtol=0
        )


@pytest.mark.parametrize("method", ("rhf", "uhf"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_prepared_single_overlap_survives_energy_force_dispatch(
    tmp_path: typing.Any,
    monkeypatch: typing.Any,
    method: typing.Any,
    representation: typing.Any,
) -> None:
    """The C prepared single owner retains X across all output selections.

    Its API has no warm-density input: core guesses remain necessary while
    repeated overlap solves must disappear, independently of SCF iterations.
    """
    if os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    import ctypes as ct

    import numpy as np
    from vibeqc import Atom, Calculator, _native

    assert os.environ.get("SLURM_JOB_ID")
    calc = Calculator(
        method=method,
        basis="def2-svp",
        basis_representation=representation,
        device="cuda",
        density_fitting="cuda",
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    lib = calc._library
    context, system, calculation = ct.c_void_p(), ct.c_void_p(), ct.c_void_p()
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
        values, gradients = [], []
        for step, force in enumerate((False, True, False, True)):
            path = tmp_path / f"single-{step}.jsonl"
            monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
            forces = (ct.c_double * 6)() if force else None
            out = _native.ResultDescriptor(
                ct.sizeof(_native.ResultDescriptor),
                _native.ABI_VERSION,
                0,
                forces,
                6 if force else 0,
                0,
                0,
                0,
                0,
                0,
            )
            _native.check(
                lib,
                lib.vibeqc_calculation_execute(calculation, ct.byref(out)),
                context=context,
            )
            monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
            trace = aggregate_host(read_host_trace(path))
            assert actual_calls(trace, "overlap") == (step == 0)
            assert actual_calls(trace, "core_guess") == 1
            values.append(out.energy)
            if force:
                gradients.append(list(forces))
        np.testing.assert_allclose(values, values[0], atol=1e-9, rtol=0)
        np.testing.assert_allclose(gradients[0], gradients[1], atol=1e-8, rtol=0)
    finally:
        lib.vibeqc_calculation_destroy(calculation)
        lib.vibeqc_system_destroy(system)
        lib.vibeqc_context_destroy(context)


@pytest.mark.parametrize("spin", ("restricted", "unrestricted"))
@pytest.mark.parametrize("representation", ("cartesian", "spherical"))
def test_independent_fock_overlap_owners_distinguish_same_size_basis(
    tmp_path: typing.Any,
    monkeypatch: typing.Any,
    spin: typing.Any,
    representation: typing.Any,
) -> None:
    """Equal AO counts cannot let distinct basis owners share an orthogonalizer."""
    if os.environ.get("VIBEQC_RESOURCE_CUDA_TEST") != "1":
        pytest.skip("requires an explicitly Slurm-allocated GPU")
    import numpy as np
    from vibeqc import Primitive, Shell
    from vibeqc.fock import FockBuildSpec, FockPlan
    from vibeqc_compiler.dft import NativeAO

    assert os.environ.get("SLURM_JOB_ID")
    atoms = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    bases = [
        tuple(Shell(i, 0, (Primitive(exponent, 1.0),)) for i in range(2))
        for exponent in (0.5, 1.2)
    ]
    spec = FockBuildSpec.hf(spin, coulomb="density_fitted", exchange="density_fitted")
    state = {"charge": 1, "multiplicity": 2} if spin == "unrestricted" else {}
    with (
        NativeAO(atoms, basis=bases[0], representation=representation, **state) as a,
        NativeAO(atoms, basis=bases[1], representation=representation, **state) as b,
        FockPlan(a, spec, device="cuda") as first,
        FockPlan(b, spec, device="cuda") as second,
    ):
        sequence = 0

        def traced(plan: typing.Any, **kwargs: typing.Any) -> typing.Any:
            nonlocal sequence
            path = tmp_path / f"independent-{sequence}.jsonl"
            sequence += 1
            monkeypatch.setenv("VIBEQC_DF_HOST_TRACE", str(path))
            try:
                result = plan.solve(
                    energy_tolerance=1e-12, density_tolerance=1e-10, **kwargs
                )
            finally:
                monkeypatch.delenv("VIBEQC_DF_HOST_TRACE")
            trace = aggregate_host(read_host_trace(path))
            return result, actual_calls(trace, "overlap")

        cold_a, count_a = traced(first, compute_forces=False)
        cold_b, count_b = traced(second, compute_forces=False)
        assert count_a == count_b == 1
        assert abs(cold_a.energy - cold_b.energy) > 1e-3
        for plan, cold in ((first, cold_a), (second, cold_b)):
            warm, count = traced(plan, initial_density=cold.density)
            assert count == 0 and warm.initial_density_used
            monkeypatch.setenv("VIBEQC_DF_REBUILD_OVERLAP", "1")
            rebuilt, count = traced(plan, initial_density=cold.density)
            monkeypatch.delenv("VIBEQC_DF_REBUILD_OVERLAP")
            assert count == 1
            assert rebuilt.energy == pytest.approx(warm.energy, abs=1e-10)
            np.testing.assert_allclose(
                rebuilt.density, warm.density, atol=1e-10, rtol=0
            )
            np.testing.assert_allclose(rebuilt.forces, warm.forces, atol=1e-9, rtol=0)
            _, count = traced(plan, initial_density=cold.density, compute_forces=False)
            assert count == 0
