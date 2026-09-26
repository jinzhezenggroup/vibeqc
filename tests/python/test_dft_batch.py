"""Native ragged KS replay, spin normalization and last-good seed isolation."""

import ctypes
import typing

import numpy as np
import pytest
from vibeqc import Calculator, _native


@pytest.fixture(params=("cpu", "cuda"))
def device(request: typing.Any) -> typing.Any:
    if request.param == "cuda":
        library = _native.load_library()
        descriptor = _native.ContextDescriptor(
            ctypes.sizeof(_native.ContextDescriptor),
            _native.ABI_VERSION,
            0,
            _native.BACKEND_CUDA,
        )
        context = ctypes.c_void_p()
        try:
            _native.check(
                library,
                library.vibeqc_context_create(
                    ctypes.byref(descriptor), ctypes.byref(context)
                ),
            )
        except RuntimeError as error:
            pytest.skip(f"CUDA context unavailable: {error}")
        library.vibeqc_context_destroy(context)
    return request.param


def warm_snapshot(prepared: typing.Any, index: typing.Any) -> typing.Any:
    """Explicit seed export; normal energy-only execution needs no AO download."""
    state = _native.HfWarmState(ctypes.sizeof(_native.HfWarmState), _native.ABI_VERSION)
    getter = prepared._library.vibeqc_batch_get_hf_warm_state
    _native.check(
        prepared._library, getter(prepared._batch, index, ctypes.byref(state))
    )
    if not state.present:
        return None
    density = np.empty(state.density_count)
    coordinates = np.empty(state.coordinate_count)
    state.density = density.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    state.coordinates = coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    _native.check(
        prepared._library, getter(prepared._batch, index, ctypes.byref(state))
    )
    return density, coordinates, state.energy


def restore_snapshots(prepared: typing.Any, snapshots: typing.Any) -> typing.Any:
    """Keep the caller-owned arrays alive through the atomic native import."""
    states = (_native.HfWarmState * len(snapshots))()
    for state, snapshot in zip(states, snapshots):
        state.struct_size, state.abi_version = ctypes.sizeof(state), _native.ABI_VERSION
        if snapshot is None:
            continue
        density, coordinates, energy = snapshot
        state.present = 1
        state.density = density.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        state.density_count = density.size
        state.coordinates = coordinates.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        state.coordinate_count = coordinates.size
        state.energy = energy
    return prepared._library.vibeqc_batch_restore_hf_warm_states(
        prepared._batch, states, len(states)
    )


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks", "lda-uks", "pbe-uks"))
def test_ragged_replay_geometry_failure_and_frozen_seed(
    method: typing.Any, device: typing.Any
) -> None:
    unrestricted = method.endswith("uks")
    if unrestricted:
        systems = [
            [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
            [("Li", (0, 0, 0))],
            [("H", (0, 0, 0))],
        ]
        charges, multiplicities = [1, 0, 0], [2, 2, 2]
    else:
        systems = [
            [("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))],
            [("He", (0, 0, 0))],
            [("H", (0, 0, -0.9)), ("H", (0, 0, 0.9))],
        ]
        charges, multiplicities = [0, 0, 0], [1, 1, 1]
    calculator = Calculator(
        method=method,
        basis="sto-3g",
        device=device,
        max_iterations=150,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch(
        systems, charges=charges, multiplicities=multiplicities
    ) as prepared:
        # Omitted properties follow the method's energy-only capability.
        cold = prepared.execute(strict=True)
        assert all(not item.warm_start_used for item in cold.items)
        for i, item in enumerate(cold.items):
            reference = calculator.singlepoint(
                systems[i], charge=charges[i], multiplicity=multiplicities[i]
            )
            assert item.index == i and item.forces is None
            assert item.energy == pytest.approx(reference.energy, abs=1e-9)
            assert item.physical_residual_rms < 1e-9
            assert item.executed_backend == (
                "cuda" if device == "cuda" else "cpu_reference"
            )
        replay = prepared.execute(strict=True)
        assert all(item.warm_start_used for item in replay.items)
        assert all(not item.warm_start_fallback for item in replay.items)
        assert replay.energies == pytest.approx(cold.energies, abs=1e-9)

        prepared.set_warm_start_updates(False)
        before = [warm_snapshot(prepared, i) for i in range(3)]
        changed = np.array([[0, 0, -0.85], [0, 0, 0.85]])
        target = [(atom[0], position) for atom, position in zip(systems[0], changed)]
        moved = prepared.execute([changed, None, None], strict=True)
        independent = calculator.singlepoint(
            target, charge=charges[0], multiplicity=multiplicities[0]
        )
        assert moved.items[0].energy == pytest.approx(independent.energy, abs=1e-9)
        assert abs(moved.items[0].energy - cold.items[0].energy) > 1e-5
        # Repeating a frozen changed-geometry run starts from the same dm0.
        repeat = prepared.execute([changed, None, None], strict=True)
        assert [item.iterations for item in repeat.items] == [
            item.iterations for item in moved.items
        ]
        for i in range(3):
            after = warm_snapshot(prepared, i)
            assert np.array_equal(after[0], before[i][0])
            assert np.array_equal(after[1], before[i][1])
            assert after[2] == before[i][2]

        for invalid in (np.zeros((1, 3)), np.full((2, 3), np.nan), np.zeros((2, 3))):
            failed = prepared.execute([invalid, None, None])
            assert failed.failure_indices == (0,)
            assert failed.items[0].physical_residual_rms is None
            assert failed.items[1].energy == pytest.approx(cold.items[1].energy)
            assert np.array_equal(warm_snapshot(prepared, 0)[0], before[0][0])
        # None means the ORIGINAL prepared geometry even after a moved solve.
        restored = prepared.execute(strict=True)
        assert restored.energies == pytest.approx(cold.energies, abs=1e-9)
        with pytest.raises((ValueError, RuntimeError), match="forces|gradient"):
            prepared.execute(properties=("energy", "forces"))
        prepared.clear_warm_starts()
        assert all(warm_snapshot(prepared, i) is None for i in range(3))
        no_seed = prepared.execute(strict=True)
        assert all(not item.warm_start_used for item in no_seed.items)
        assert all(warm_snapshot(prepared, i) is None for i in range(3))
        prepared.set_warm_start_updates(True)
        prepared.execute(strict=True)
        assert all(warm_snapshot(prepared, i) is not None for i in range(3))
        snapshots = [warm_snapshot(prepared, i) for i in range(3)]
        invalid = (2 * snapshots[1][0], snapshots[1][1], snapshots[1][2])
        assert restore_snapshots(prepared, [before[0], invalid, None]) == (
            _native.STATUS_INVALID_ARGUMENT
        )
        for i in range(3):
            assert np.array_equal(warm_snapshot(prepared, i)[0], snapshots[i][0])
        prepared.clear_warm_starts()
        assert (
            restore_snapshots(prepared, [before[0], None, None])
            == _native.STATUS_SUCCESS
        )
        imported = prepared.execute(strict=True)
        assert [item.warm_start_used for item in imported.items] == [True, False, False]
        assert not imported.items[0].warm_start_fallback
        assert imported.energies == pytest.approx(cold.energies, abs=1e-9)


def test_nonconverged_batch_does_not_establish_seed(device: typing.Any) -> None:
    calculator = Calculator(method="pbe-rks", device=device, max_iterations=1)
    with calculator.prepare_batch(
        [[("H", (0, 0, -0.7)), ("H", (0, 0, 0.7))]]
    ) as prepared:
        first = prepared.execute()
        assert first.failure_indices == (0,)
        assert first.items[0].status == _native.STATUS_NOT_CONVERGED
        assert np.isfinite(first.items[0].physical_residual_rms)
        assert warm_snapshot(prepared, 0) is None
        assert not prepared.execute().items[0].warm_start_used


def test_batch_scf_query_abi_and_stale_record() -> None:
    calculator = Calculator(method="pbe-uks", device="cpu")
    with calculator.prepare_batch(
        [[("H", (0, 0, 0))], [("Li", (0, 0, 0))]], multiplicities=[2, 2]
    ) as prepared:
        getter = prepared._library.vibeqc_batch_get_scf_diagnostic
        record = _native.ScfDiagnostic(
            ctypes.sizeof(_native.ScfDiagnostic), _native.ABI_VERSION, -7, -11
        )
        original = bytes(record)
        assert getter(prepared._batch, 0, None) == _native.STATUS_NOT_IMPLEMENTED
        assert getter(prepared._batch, 2, None) == _native.STATUS_INVALID_ARGUMENT
        assert (
            getter(prepared._batch, 0, ctypes.byref(record))
            == _native.STATUS_NOT_IMPLEMENTED
        )
        assert bytes(record) == original
        prepared.execute(strict=True)
        assert (
            getter(prepared._batch, 0, ctypes.byref(record)) == _native.STATUS_SUCCESS
        )
        # The exactly stationary one-orbital atom has a real measured zero.
        assert record.physical_residual_rms == 0
        for size, abi in (
            (ctypes.sizeof(record) - 1, _native.ABI_VERSION),
            (ctypes.sizeof(record), _native.ABI_VERSION + 1),
        ):
            record.struct_size, record.abi_version = size, abi
            original = bytes(record)
            assert (
                getter(prepared._batch, 0, ctypes.byref(record))
                == _native.STATUS_ABI_MISMATCH
            )
            assert bytes(record) == original
        prepared.execute([np.zeros((2, 3)), None])
        assert getter(prepared._batch, 0, None) == _native.STATUS_NOT_IMPLEMENTED
        assert getter(prepared._batch, 1, None) == _native.STATUS_SUCCESS
        outputs = (_native.BatchItemResultDescriptor * 1)()
        assert (
            prepared._library.vibeqc_batch_execute(prepared._batch, None, 0, outputs, 1)
            == _native.STATUS_INVALID_ARGUMENT
        )
        assert getter(prepared._batch, 1, None) == _native.STATUS_NOT_IMPLEMENTED


@pytest.mark.parametrize("method,ao_order", (("lda-rks", 0), ("pbe-rks", 1)))
def test_energy_batch_resolves_only_required_operator_derivatives(
    monkeypatch: typing.Any, method: typing.Any, ao_order: typing.Any
) -> None:
    import vibeqc.calculator as calculator_module

    calls = []
    original = calculator_module.require_basis

    def record(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        calls.append((kwargs["operator"], kwargs["derivative_order"]))
        return original(*args, **kwargs)

    calculator = Calculator(method=method, device="cpu")
    monkeypatch.setattr(calculator_module, "require_basis", record)
    with calculator.prepare_batch([[("He", (0, 0, 0))]]):
        assert ("ao", ao_order) in calls
        assert all(order == 0 for operator, order in calls if operator != "ao")


@pytest.mark.parametrize("method", ("lda-uks", "pbe-uks"))
def test_open_shell_large_solver_ragged_replay_and_failure(
    method: typing.Any, device: typing.Any
) -> None:
    """OH uses the >16-AO solver beside an independent one-electron item."""
    systems = [
        [("O", (0, 0, 0)), ("H", (0, 0, 1.8))],
        [("H", (0, 0, 0))],
    ]
    calculator = Calculator(
        method=method,
        basis="def2-svp",
        device=device,
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    )
    with calculator.prepare_batch(systems, multiplicities=[2, 2]) as prepared:
        cold = prepared.execute(strict=True)
        for item in cold.items:
            assert item.physical_residual_rms < 1e-9
            assert item.executed_backend == (
                "cuda" if device == "cuda" else "cpu_reference"
            )
        original_seed = warm_snapshot(prepared, 0)
        assert original_seed[0].size > 2 * 16 * 16
        replay = prepared.execute(strict=True)
        assert all(item.warm_start_used for item in replay.items)
        assert replay.energies == pytest.approx(cold.energies, abs=1e-9)
        prepared.set_warm_start_updates(False)
        moved_coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.9]])
        moved = prepared.execute([moved_coordinates, None], strict=True)
        fresh = calculator.singlepoint(
            [("O", moved_coordinates[0]), ("H", moved_coordinates[1])], multiplicity=2
        )
        assert moved.items[0].energy == pytest.approx(fresh.energy, abs=1e-9)
        assert moved.items[0].physical_residual_rms < 1e-9
        assert abs(moved.items[0].energy - cold.items[0].energy) > 1e-5
        saved_seed = warm_snapshot(prepared, 0)
        failed = prepared.execute([np.zeros((2, 3)), None])
        assert failed.failure_indices == (0,)
        assert failed.items[0].status == _native.STATUS_NUMERICAL_FAILURE
        assert failed.items[0].physical_residual_rms is None
        assert failed.items[1].energy == pytest.approx(cold.items[1].energy, abs=1e-9)
        assert np.array_equal(warm_snapshot(prepared, 0)[0], saved_seed[0])
        restored = prepared.execute(strict=True)
        assert restored.energies == pytest.approx(cold.energies, abs=1e-9)
        assert all(item.physical_residual_rms < 1e-9 for item in restored.items)
