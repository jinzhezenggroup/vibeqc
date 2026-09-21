"""Complete bounded CPU ECP public forces, independent references and replay."""

from __future__ import annotations

import hashlib
import json
import os
import typing
from fractions import Fraction
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
from test_ecp import fixture
from test_ecp_stationary_cpu import GRID, reference
from vibeqc import Calculator, KsOptions, ResourceBudget, _native
from vibeqc._dft_gradient import StationaryKsState
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.method import MethodSpec, resolve_method


@pytest.fixture(autouse=True)
def retain_endpoint_evidence(request: typing.Any) -> typing.Iterator[None]:
    """Retain timings/errors/work even when CI does not request JUnit XML."""
    yield
    path = Path(".artifacts/spd-cpu")
    path.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16]
    (path / f"{key}.json").write_text(
        json.dumps(
            {
                "test": request.node.nodeid,
                "measurements": dict(request.node.user_properties),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def calculator(
    record: typing.Any, method: typing.Any = "pbe-rks", **kwargs: typing.Any
) -> typing.Any:
    return Calculator(
        basis=record,
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
        **kwargs,
    )


@pytest.mark.parametrize("method", ["lda-rks", "pbe-rks", "lda-uks", "pbe-uks"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_public_ecp_force_analytic_and_reconverged_fd(
    method: typing.Any,
    representation: typing.Any,
    record_property: typing.Any,
    tmp_path: typing.Any,
    d_shell: bool = False,
) -> None:
    spin = int(method.endswith("uks"))
    atoms, record, mol = fixture(
        spin=spin, representation=representation, d_shell=d_shell
    )
    # Exercise serialized spherical ECP data through the real public endpoint.
    public_basis = record
    if representation == "spherical":
        path = tmp_path / "spherical-ecp.json"
        record.write(path)
        public_basis = path
    calc = calculator(public_basis, method)
    started = perf_counter()
    result = calc.singlepoint(atoms, charge=spin, multiplicity=spin + 1)
    record_property("complete_endpoint_seconds", perf_counter() - started)
    assert result.executed_backend == "cpu_reference"
    assert result.converged and np.isfinite(result.forces).all()
    with (
        calc.prepare_batch([atoms], charges=[spin], multiplicities=[spin + 1]) as batch,
        NativeAO(
            atoms,
            basis=record,
            representation=representation,
            charge=spin,
            multiplicity=spin + 1,
        ) as basis,
    ):
        energy = batch.execute(strict=True, properties=("energy",)).items[0]
        assert energy.forces is None
        state = StationaryKsState.from_native(batch, basis)
        try:
            expected_energy, gradient = reference(mol, state, method)
        finally:
            state._source.close()
    assert abs(result.energy - expected_energy) < 2e-8
    np.testing.assert_allclose(result.forces, -gradient, atol=1e-7, rtol=0)
    np.testing.assert_allclose(result.forces.sum(axis=0), 0, atol=1e-9, rtol=0)
    record_property(
        "analytic_max_error", float(np.max(np.abs(result.forces + gradient)))
    )
    direction = np.array([[0.2, -0.13, 0.07], [-0.11, 0.08, 0.19]])
    projection = float(np.sum(result.forces * direction))
    errors = []
    for step in (3e-4, 1e-4):
        energies = []
        for sign in (1, -1):
            moved = [
                (a, np.asarray(r) + sign * step * d)
                for (a, r), d in zip(atoms, direction)
            ]
            energies.append(
                calc.singlepoint(
                    moved, charge=spin, multiplicity=spin + 1, properties=("energy",)
                ).energy
            )
        errors.append(abs(-(energies[0] - energies[1]) / (2 * step) - projection))
    assert max(errors) < 2e-7
    record_property("fd_errors", errors)
    if representation == "spherical" and not d_shell:
        # s/p real spherical and Cartesian spaces are equivalent, but public
        # normalized AO ordering/representation identities remain distinct.
        from dataclasses import replace

        cartesian = calculator(replace(record, representation="cartesian"), method)
        other = cartesian.singlepoint(atoms, charge=spin, multiplicity=spin + 1)
        assert abs(result.energy - other.energy) < 2e-9
        np.testing.assert_allclose(result.forces, other.forces, atol=1e-9, rtol=0)
        record_property(
            "representation_force_error",
            float(np.max(np.abs(result.forces - other.forces))),
        )


@pytest.mark.parametrize("method", ["pbe-rks", "pbe-uks"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_public_ecp_budgeted_ragged_replay_and_failure_recovery(
    method: typing.Any,
    representation: typing.Any,
    record_property: typing.Any,
    d_shell: bool = False,
) -> None:
    spin = int(method.endswith("uks"))
    atoms, record, mol = fixture(
        spin=spin, representation=representation, d_shell=d_shell
    )
    fragment = [("H", (0, 0, 0))] if spin else [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    systems, charges, multiplicities = [atoms, fragment], [spin, 0], [spin + 1] * 2
    calc = calculator(record, method)
    plan = calc.estimate_resources(
        systems, charges=charges, multiplicities=multiplicities
    ).require_feasible()
    assert "forces" in plan.requests[0].identity.observables
    too_small = calculator(
        record,
        method,
        resource_budget=ResourceBudget(host_bytes=plan.peak_bytes["host"] - 1),
    )
    with pytest.raises(MemoryError):
        too_small.prepare_batch(systems, charges=charges, multiplicities=multiplicities)
    bounded = calculator(
        record,
        method,
        resource_budget=ResourceBudget(
            host_bytes=plan.peak_bytes["host"],
            device_bytes=plan.peak_bytes["device"],
        ),
    )
    with bounded.prepare_batch(
        systems, charges=charges, multiplicities=multiplicities
    ) as batch:
        started = perf_counter()
        cold = batch.execute(strict=True)
        record_property("batch_cold_seconds", perf_counter() - started)
        started = perf_counter()
        warm = batch.execute(strict=True)
        record_property("batch_warm_seconds", perf_counter() - started)
        assert all(item.warm_start_used for item in warm.items)
        for a, b in zip(cold.items, warm.items):
            np.testing.assert_allclose(a.forces, b.forces, atol=1e-9, rtol=0)
        xyz = mol.atom_coords()
        moved = xyz.copy()
        moved[1] += [0.03, -0.02, 0.09]
        started = perf_counter()
        replay = batch.execute(coordinates=[moved, None], strict=True)
        record_property("batch_changed_geometry_seconds", perf_counter() - started)
        fresh_atoms = [(a, r) for (a, _), r in zip(atoms, moved)]
        fresh = calc.singlepoint(fresh_atoms, charge=spin, multiplicity=spin + 1)
        np.testing.assert_allclose(
            replay.items[0].forces, fresh.forces, atol=1e-9, rtol=0
        )
        assert np.max(np.abs(replay.items[0].forces - cold.items[0].forces)) > 1e-5
        failed = batch.execute(coordinates=[[0.0], None])
        assert not failed.items[0].succeeded and failed.items[0].forces is None
        assert failed.items[1].succeeded
        np.testing.assert_allclose(
            failed.items[1].forces, cold.items[1].forces, atol=1e-9, rtol=0
        )
        restored = batch.execute(coordinates=[xyz, None], strict=True)
        np.testing.assert_allclose(
            restored.items[0].forces, cold.items[0].forces, atol=1e-9, rtol=0
        )
        record_property("planned_peaks", dict(plan.peak_bytes))
        work = batch.resource_diagnostics["generated_force"]
        assert [item["index"] for item in work] == [0, 1]
        assert work[0]["work"]["ecp_quadrature_pair_samples"] > 0
        assert work[1]["work"]["ecp_quadrature_pair_samples"] == 0
        if d_shell:
            schedule = work[0]["work"]
            assert schedule["primitive_compiled_kernels"] == 362
            assert schedule["primitive_translation_units"] == 46
            assert schedule["primitive_generated_source_bytes"] <= 64 << 20
            assert schedule["primitive_largest_source_bytes"] <= 4 << 20
        for item in work:
            assert item["work"]["additional_host_numeric_bound"] <= 256 << 20
        assert all(
            item["work"]["ecp_provider"] == "checked-native-cpu-two-grid-v1"
            for item in work
        )
        record_property("generated_force", work)


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
@pytest.mark.parametrize("budget", ["max_ecp_pair_samples", "max_host_bytes"])
def test_public_ecp_force_failure_is_transactional_and_closes_snapshot(
    monkeypatch: typing.Any, representation: typing.Any, budget: typing.Any
) -> None:
    from vibeqc import _stationary_cpu
    from vibeqc._ks_snapshot import NativeKsSnapshot

    atoms, record, _ = fixture(representation=representation)
    fragment = [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]
    calc = calculator(record)
    sources = []
    original = _stationary_cpu.complete_rks_gradient_diagnostic

    def rejected(
        state: typing.Any, basis: typing.Any, **kwargs: typing.Any
    ) -> typing.Any:
        sources.append(state._source)
        if state._source.hamiltonian == "scalar-semilocal-ecp":
            # Exercise the real admission path, before the ECP provider runs.
            kwargs[budget] = 1
        return original(state, basis, **kwargs)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("ECP provider ran after rejected work admission")

    with calc.prepare_batch([atoms, fragment]) as batch:
        with monkeypatch.context() as patch:
            patch.setattr(_stationary_cpu, "complete_rks_gradient_diagnostic", rejected)
            patch.setattr(NativeKsSnapshot, "ecp_derivatives", forbidden)
            result = batch.execute()
            assert result.items[0].status == _native.STATUS_INVALID_ARGUMENT
            assert result.items[0].forces is None
            assert (
                result.items[1].succeeded and np.isfinite(result.items[1].forces).all()
            )
            with pytest.raises(RuntimeError):
                batch.execute(strict=True)
        assert sources and all(not source._handle for source in sources)
        energy = batch.execute(strict=True, properties=("energy",))
        assert all(item.forces is None for item in energy.items)
        recovered = batch.execute(strict=True)
        assert all(
            item.succeeded and np.isfinite(item.forces).all()
            for item in recovered.items
        )


def test_cpu_force_budget_rejects_before_snapshot_export(
    monkeypatch: typing.Any,
) -> None:
    from vibeqc import _cpu_force_resources

    atoms, record, _ = fixture()
    calc = calculator(record)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("snapshot exported after host-cap rejection")

    with calc.prepare_batch([atoms]) as batch:
        batch.execute(strict=True, properties=("energy",))
        monkeypatch.setattr(_cpu_force_resources, "CPU_FORCE_HOST_CAP", 1)
        monkeypatch.setattr(StationaryKsState, "from_native", forbidden)
        with pytest.raises(ValueError, match="additional-host byte"):
            batch._public_dft_cpu_force(0, batch._systems[0])


def check_spd_arbitrary_ordered_weights_against_libcint_energy_differences(
    representation: str,
) -> None:
    from vibeqc._stationary_cpu_components import ComponentPrimitiveExecutor
    from vibeqc._stationary_cpu_streaming import CompiledComponentExecutor
    from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter

    atoms, record, mol = fixture(representation=representation, d_shell=True)
    direction = np.array([[0.17, -0.11, 0.29], [-0.23, 0.31, -0.07]])
    with NativeAO(atoms, basis=record, representation=representation) as basis:
        executor = ComponentPrimitiveExecutor(
            basis,
            Path(os.environ.get("VIBEQC_STATIONARY_CACHE", ".cache/stationary-cpu")),
            2,
            CppCompilerAdapter(Path(os.environ.get("CXX", "c++"))),
        )
        candidates = [
            CompiledComponentExecutor(
                basis,
                Path(
                    os.environ.get("VIBEQC_STATIONARY_CACHE", ".cache/stationary-cpu")
                ),
                tile,
                CppCompilerAdapter(Path(os.environ.get("CXX", "c++"))),
            )
            for tile in (1, 2, 128)
        ]
        n = basis.nao
        norms = np.sqrt(mol.intor("int1e_ovlp").diagonal())
        for op, intor, selections in (
            (
                "overlap",
                "int1e_ovlp",
                [(8, n - 1), (n - 2, n - 1), (n - 1, 8), (8, n - 2)],
            ),
            (
                "kinetic",
                "int1e_kin",
                [(8, n - 1), (n - 2, n - 1), (n - 1, 8), (8, n - 2)],
            ),
            (
                "nuclear_attraction",
                "int1e_nuc",
                [(8, n - 1), (n - 2, n - 1), (n - 1, 8), (8, n - 2)],
            ),
            (
                "four_center_eri",
                "int2e",
                [
                    (8, 1, n - 2, n - 1),
                    (n - 2, n - 1, 8, 1),
                    (n - 2, 8, n - 2, n - 1),
                    (n - 1, n - 1, 8, 8),
                ],
            ),
        ):
            gradient = np.zeros((2, 3))
            weights = (0.37, -0.81, 0.21, -0.49)
            for indices, weight in zip(selections, weights):
                nuclei = range(2) if op == "nuclear_attraction" else (None,)
                for nucleus in nuclei:
                    owners, values = executor.integral(
                        op,
                        indices,
                        weight
                        if nucleus is None
                        else weight * mol.atom_charge(nucleus),
                        nucleus,
                    )
                    for candidate in candidates:
                        mapped, actual = candidate.integral(
                            op,
                            indices,
                            weight
                            if nucleus is None
                            else weight * mol.atom_charge(nucleus),
                            nucleus,
                        )
                        assert mapped == owners
                        np.testing.assert_allclose(
                            actual, values, atol=2e-13, rtol=2e-13
                        )
                        assert candidate.records == executor.records
                    np.add.at(gradient, owners, values)
            for step in (3e-4, 1e-4):
                energies = []
                for sign in (1, -1):
                    displaced = mol.copy()
                    displaced.set_geom_(
                        mol.atom_coords() + sign * step * direction, unit="Bohr"
                    )
                    values = displaced.intor(intor)
                    energies.append(
                        sum(
                            weight * values[indices] / np.prod(norms[list(indices)])
                            for indices, weight in zip(selections, weights)
                        )
                    )
                assert (
                    abs(
                        (energies[0] - energies[1]) / (2 * step)
                        - np.sum(gradient * direction)
                    )
                    < 2e-8
                )
        # A failed native derivative publishes nothing and does not advance
        # semantic work; restoring the dispatch permits a valid subsequent call.
        request = ("nuclear", ())
        original = executor.calls[request]
        before = executor.records
        executor.calls[request] = (lambda *args: 1, 0)
        with pytest.raises(ArithmeticError, match="component derivative failed"):
            executor.nuclear(0, 1, mol.atom_charges())
        assert executor.records == before
        executor.calls[request] = original
        assert np.isfinite(executor.nuclear(0, 1, mol.atom_charges())).all()
        assert executor.records == before + 1
        for candidate in candidates:
            before = candidate.records
            with pytest.raises(ArithmeticError, match="component derivative failed"):
                candidate.integral("overlap", (8, n - 1), float("nan"))
            assert candidate.records == before
            _, zero = candidate.integral("four_center_eri", (8, 8, n - 1, n - 1), 0.0)
            np.testing.assert_array_equal(zero, 0)


def check_spd_paired_endpoint(
    representation: str,
    monkeypatch: typing.Any,
    record_property: typing.Any,
) -> None:
    """Same-runner complete batches, exact budgets, replay and failure recovery.

    The shared mathematical source cache is populated by the numerical gates;
    cold here means a fresh prepared SCF batch, not a fresh C++ toolchain cache.
    """
    from vibeqc import _stationary_cpu

    original = _stationary_cpu.complete_rks_gradient_diagnostic
    results = {}
    for strategy in ("python", "native"):
        measurements = {}
        gradients = []

        def selected(
            state: typing.Any,
            basis: typing.Any,
            strategy: str = strategy,
            gradients: list = gradients,
            **kwargs: typing.Any,
        ) -> typing.Any:
            value = original(state, basis, component_execution=strategy, **kwargs)
            gradients.append(value.gradient.copy())
            return value

        with monkeypatch.context() as patch:
            patch.setattr(_stationary_cpu, "complete_rks_gradient_diagnostic", selected)
            test_public_ecp_budgeted_ragged_replay_and_failure_recovery(
                "pbe-rks",
                representation,
                measurements.__setitem__,
                d_shell=True,
            )
        record_property(strategy, measurements)
        results[strategy] = (measurements, gradients)
    baseline, candidate = results["python"], results["native"]
    assert len(baseline[1]) == len(candidate[1])
    for old, new in zip(baseline[1], candidate[1], strict=True):
        np.testing.assert_allclose(new, old, atol=2e-12, rtol=0)
    for old, new in zip(
        baseline[0]["generated_force"], candidate[0]["generated_force"], strict=True
    ):
        for counter in (
            "primitive_records",
            "primitive_record_bound",
            "ecp_quadrature_pair_samples",
            "ordered_quartets",
        ):
            assert old["work"][counter] == new["work"][counter]


@pytest.mark.parametrize("method", ["pbe-rks", "pbe-uks"])
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_ecp_force_promotion_rejects_named_and_custom_hybrids_before_preparation(
    method: str, representation: str, monkeypatch: typing.Any
) -> None:
    """Only unit semilocal, zero-K compositions may inherit public ECP forces."""
    spin = int(method.endswith("uks"))
    atoms, record, _ = fixture(spin=spin, representation=representation)

    # Positive controls: the already-qualified pure LDA/PBE ECP endpoints stay public.
    suffix = "uks" if spin else "rks"
    for pure_method in (f"lda-{suffix}", f"pbe-{suffix}"):
        pure = calculator(record, pure_method)
        assert pure.ks_options.coefficients == (1.0, 1.0, 0.0)
        assert "forces" in pure._capabilities.supported_properties

    named_hybrids = [
        calculator(record, f"pbe0-{suffix}"),
        calculator(record, f"b3lyp-{suffix}"),
    ]
    for named in named_hybrids:
        assert named.ks_options.coefficients != (1.0, 1.0, 0.0)
        assert "forces" not in named._capabilities.supported_properties

    graph = resolve_method(
        MethodSpec(
            "PBE50-ecp-force-boundary",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        spin="polarized" if spin else "unpolarized",
    )
    custom = Calculator(
        basis=record,
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID, composition=graph),
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    assert custom.ks_options.coefficients != (1.0, 1.0, 0.0)
    assert "forces" not in custom._capabilities.supported_properties

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("unqualified hybrid forces reached batch preparation")

    monkeypatch.setattr(Calculator, "prepare_batch", forbidden)
    for candidate in (*named_hybrids, custom):
        with pytest.raises(ValueError, match="does not support properties: forces"):
            candidate.singlepoint(
                atoms,
                charge=spin,
                multiplicity=spin + 1,
                properties=("energy", "forces"),
            )


def test_cpu_f_ecp_forces_rejected_before_preparation(monkeypatch: typing.Any) -> None:
    atoms, record, _ = fixture(f_shell=True)
    calc = calculator(record)

    def forbidden(*args: typing.Any, **kwargs: typing.Any) -> None:
        pytest.fail("unqualified f-shell forces reached preparation")

    monkeypatch.setattr(calc, "prepare_batch", forbidden)
    with pytest.raises(ValueError, match="forces"):
        calc.singlepoint(atoms, properties=("energy", "forces"))
