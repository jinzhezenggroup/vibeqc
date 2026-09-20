"""Complete bounded CPU ECP public forces, independent references and replay."""

from __future__ import annotations

import typing
from time import perf_counter

import numpy as np
import pytest
from test_ecp import fixture
from test_ecp_stationary_cpu import GRID, reference
from vibeqc import Calculator, KsOptions, ResourceBudget, _native
from vibeqc._dft_gradient import StationaryKsState
from vibeqc_compiler.dft import NativeAO


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
) -> None:
    spin = int(method.endswith("uks"))
    atoms, record, mol = fixture(spin=spin, representation=representation)
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
    if representation == "spherical":
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
    method: typing.Any, representation: typing.Any, record_property: typing.Any
) -> None:
    spin = int(method.endswith("uks"))
    atoms, record, mol = fixture(spin=spin, representation=representation)
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
        cold = batch.execute(strict=True)
        warm = batch.execute(strict=True)
        assert all(item.warm_start_used for item in warm.items)
        for a, b in zip(cold.items, warm.items):
            np.testing.assert_allclose(a.forces, b.forces, atol=1e-9, rtol=0)
        xyz = mol.atom_coords()
        moved = xyz.copy()
        moved[1] += [0.03, -0.02, 0.09]
        replay = batch.execute(coordinates=[moved, None], strict=True)
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
