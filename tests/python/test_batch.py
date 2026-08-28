import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell


def systems():
    return [
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
        [("He", (0.0, 0.0, 0.0))],
        [("H", (0.0, 0.0, -1.0)), ("H", (0.0, 0.0, 1.0))],
        [("H", (-1.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.0)), ("H", (1.0, 0.0, 0.0))],
    ]


def test_ragged_batch_matches_independent_results_and_buckets():
    calculator = Calculator()
    batch_systems = systems()
    charges = [0, 0, 0, 1]
    independent = [
        calculator.singlepoint(system, charge=charge)
        for system, charge in zip(batch_systems, charges, strict=True)
    ]
    result = calculator.batch_singlepoint(batch_systems, charges=charges, strict=True)
    assert np.allclose(result.energies, [item.energy for item in independent], atol=2e-10)
    assert [item.forces.shape for item in result.items] == [(2, 3), (1, 3), (2, 3), (3, 3)]
    assert result.items[0].bucket_id == result.items[2].bucket_id
    assert result.items[0].bucket_id != result.items[1].bucket_id
    assert result.items[0].bucket_id != result.items[3].bucket_id


def test_prepared_batch_warm_start_coordinate_updates_and_failure_isolation():
    calculator = Calculator()
    batch_systems = systems()
    with calculator.prepare_batch(batch_systems, charges=[0, 0, 0, 1]) as prepared:
        cold = prepared.execute(strict=True)
        prepared.set_warm_start_updates(False)
        warm = prepared.execute(strict=True)
        assert all(not item.warm_start_used for item in cold.items)
        assert all(item.warm_start_used for item in warm.items)
        assert warm.items[3].iterations < cold.items[3].iterations

        changed_h2 = np.array([[0.0, 0.0, -0.75], [0.0, 0.0, 0.75]])
        changed = prepared.execute([changed_h2, None, None, None], strict=True)
        reference = calculator.singlepoint(
            [("H", (0.0, 0.0, -0.75)), ("H", (0.0, 0.0, 0.75))]
        )
        assert changed.items[0].energy == pytest.approx(reference.energy, abs=2e-10)

        isolated = prepared.execute([None, None, np.zeros((1, 3)), None])
        assert isolated.failure_indices == (2,)
        assert isolated.items[2].forces is None
        assert all(isolated.items[index].succeeded for index in (0, 1, 3))
        assert np.isnan(isolated.energies[2])
        with pytest.raises(RuntimeError, match="2: invalid argument"):
            isolated.raise_for_failures()

        prepared.clear_warm_starts()
        prepared.set_warm_start_updates(True)
        cold_again = prepared.execute(strict=True)
        assert all(not item.warm_start_used for item in cold_again.items)


def test_closed_prepared_batch_rejects_warm_start_policy_update():
    prepared = Calculator().prepare_batch(systems()[:1])
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.set_warm_start_updates(False)


def test_closed_prepared_batch_rejects_execution():
    prepared = Calculator().prepare_batch(systems()[:1])
    prepared.close()
    with pytest.raises(RuntimeError, match="closed"):
        prepared.execute()


def test_shell_class_profile_requires_explicit_opt_in():
    with Calculator().prepare_batch(systems()[:1]) as prepared:
        with pytest.raises(RuntimeError, match="shell_class_profiling=True"):
            prepared.last_shell_class_profile()
        with pytest.raises(RuntimeError, match="shell_class_profiling=True"):
            prepared.last_ppps_queue_profile()


def test_inactive_eigensolver_profile_requires_explicit_opt_in():
    with Calculator().prepare_batch(systems()[:1]) as prepared, pytest.raises(
        RuntimeError, match="inactive_eigensolver_profiling=True"
    ):
        prepared.last_inactive_eigensolver_profile()


def test_nonconverged_item_does_not_abort_converged_neighbor():
    calculator = Calculator(max_iterations=2)
    batch_systems = [systems()[0], systems()[3]]
    result = calculator.batch_singlepoint(batch_systems, charges=[0, 1])
    assert result.items[0].succeeded
    assert result.items[0].converged
    assert result.items[1].status_message == "SCF did not converge"
    assert not result.items[1].converged
    assert result.failure_indices == (1,)


def test_real_spherical_batch_reuses_fixed_topology_plan():
    """Keep spherical transforms compatible with native batch warm starts."""

    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    systems = [
        [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
        [("He", (0.0, 0.0, -0.8)), ("H", (0.0, 0.0, 0.8))],
    ]
    calculator = Calculator(
        basis=basis,
        basis_representation="spherical",
        device="cpu",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    independent = [
        calculator.singlepoint(system, charge=1) for system in systems
    ]
    with calculator.prepare_batch(systems, charges=[1, 1]) as prepared:
        cold = prepared.execute(strict=True)
        warm = prepared.execute(strict=True)

    assert np.allclose(
        cold.energies,
        [result.energy for result in independent],
        atol=3.0e-12,
    )
    assert all(item.executed_backend == "cpu_reference" for item in cold.items)
    assert all(item.warm_start_used for item in warm.items)
    assert all(
        warm.items[index].iterations <= cold.items[index].iterations
        for index in range(len(systems))
    )


def test_cuda_real_spherical_batch_reuses_fixed_topology_plan():
    """Exercise public CUDA batching with sparse spherical AO expansions."""

    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    systems = [
        [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
        [("He", (0.0, 0.0, -0.8)), ("H", (0.0, 0.0, 0.8))],
    ]
    try:
        calculator = Calculator(
            basis=basis,
            basis_representation="spherical",
            device="cuda",
            energy_tolerance=1.0e-12,
            density_tolerance=1.0e-10,
        )
        with calculator.prepare_batch(
            systems, charges=[1, 1], warm_start=True
        ) as prepared:
            cold = prepared.execute(strict=True)
            warm = prepared.execute(strict=True)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert np.allclose(
        cold.energies,
        [-2.3341870407859284, -2.2509247051464096],
        atol=3.0e-9,
    )
    assert all(item.executed_backend == "cuda" for item in cold.items)
    assert all(item.warm_start_used for item in warm.items)
    assert all(
        warm.items[index].iterations <= cold.items[index].iterations
        for index in range(len(systems))
    )


def test_device_resident_cuda_rhf_matches_reference_when_device_is_available():
    batch_systems = [systems()[0], systems()[2], systems()[0], systems()[2]]
    reference = Calculator(device="cpu").batch_singlepoint(
        batch_systems, strict=True
    )
    try:
        candidate = Calculator(device="cuda")
        with candidate.prepare_batch(batch_systems) as prepared:
            result = prepared.execute(strict=True)
            warm = prepared.execute(strict=True)
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")
    assert all(item.executed_backend == "cuda" for item in result.items)
    assert all(item.warm_start_used for item in warm.items)
    assert np.allclose(result.energies, reference.energies, atol=2e-10)
    for expected, actual in zip(reference.items, result.items, strict=True):
        assert np.allclose(actual.forces, expected.forces, atol=2e-9)

    isolated = Calculator(device="cuda", max_iterations=2).batch_singlepoint(
        [systems()[0], systems()[3]], charges=[0, 1]
    )
    assert isolated.items[0].succeeded
    assert isolated.items[0].executed_backend == "cuda"
    assert isolated.items[1].status_message == "SCF did not converge"


def test_cuda_direct_jk_batch_reuses_stable_pair_tasks():
    """Exercise deterministic direct-J/K pair metadata for multiple peers."""

    basis = (
        Shell(0, 0, (Primitive(1.5, 1.0),)),
        Shell(0, 2, (Primitive(0.8, 1.0),)),
        Shell(0, 3, (Primitive(0.6, 1.0),)),
        Shell(1, 0, (Primitive(1.2, 1.0),)),
    )
    batch_systems = [
        [("He", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
        [("He", (0.0, 0.0, -0.8)), ("H", (0.0, 0.0, 0.8))],
    ]
    calculator = Calculator(
        basis=basis,
        device="cuda",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    try:
        with calculator.prepare_batch(
            batch_systems, charges=[1, 1], warm_start=False
        ) as prepared:
            first = prepared.execute(strict=True)
            repeated = prepared.execute(strict=True)
        independent = [
            calculator.singlepoint(system, charge=1) for system in batch_systems
        ]
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    # Device compaction and Fock assembly use atomics, so independent cold
    # replays are numerically stable rather than bitwise ordered. Keep this
    # tolerance far below the independent integral-reference gates.
    assert np.allclose(first.energies, repeated.energies, rtol=0.0, atol=1.0e-13)
    assert np.allclose(
        first.energies,
        [item.energy for item in independent],
        atol=2.0e-11,
    )
    for batched, replayed, standalone in zip(
        first.items, repeated.items, independent, strict=True
    ):
        assert batched.executed_backend == "cuda"
        assert np.allclose(batched.forces, replayed.forces, atol=2.0e-12)
        assert np.allclose(batched.forces, standalone.forces, atol=5.0e-10)


def test_cuda_uhf_ragged_batch_warm_start_and_failure_isolation():
    """Keep open-shell spin states independent inside a persistent GPU plan."""

    batch_systems = [
        [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
        [("H", (0.0, 0.0, -0.9)), ("H", (0.0, 0.0, 0.9))],
    ]
    reference = Calculator(method="uhf", device="cpu").batch_singlepoint(
        batch_systems, charges=[1, 1], multiplicities=[2, 2], strict=True
    )
    calculator = Calculator(
        method="uhf",
        device="cuda",
        energy_tolerance=1.0e-12,
        density_tolerance=1.0e-10,
    )
    try:
        with calculator.prepare_batch(
            batch_systems,
            charges=[1, 1],
            multiplicities=[2, 2],
            warm_start=True,
        ) as prepared:
            cold = prepared.execute(strict=True)
            warm = prepared.execute(strict=True)
            isolated = prepared.execute([None, np.zeros((1, 3))])
    except RuntimeError as error:
        pytest.skip(f"CUDA device unavailable: {error}")

    assert all(item.executed_backend == "cuda" for item in cold.items)
    assert all(item.warm_start_used for item in warm.items)
    assert np.allclose(cold.energies, reference.energies, atol=2.0e-10)
    for expected, actual in zip(reference.items, cold.items, strict=True):
        assert np.allclose(actual.forces, expected.forces, atol=3.0e-9)
    assert isolated.failure_indices == (1,)
    assert isolated.items[0].succeeded
    assert isolated.items[0].executed_backend == "cuda"
