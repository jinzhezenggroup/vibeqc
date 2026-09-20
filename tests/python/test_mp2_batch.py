"""Public conventional MP2 prepared-batch ownership and isolation."""

import os

import numpy as np
import pytest
from vibeqc import Calculator, method_capabilities

H2 = [
    [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))],
    [("H", (0.0, 0.0, -0.8)), ("H", (0.0, 0.0, 0.8))],
]


def test_mp2_advertises_batches_and_rejects_unsupported_flags() -> None:
    assert method_capabilities("mp2").supports_batch
    calculator = Calculator(method="mp2", device="cpu")
    with pytest.raises(RuntimeError, match="MP2 batch does not support warm starts"):
        calculator.prepare_batch(H2)
    with pytest.raises(RuntimeError, match="MP2 batch does not support profiling"):
        calculator.prepare_batch(H2, warm_start=False, shell_class_profiling=True)


def test_mp2_batch_energy_force_replay_geometry_and_order_independence() -> None:
    calculator = Calculator(method="mp2", device="cpu")
    expected = [calculator.singlepoint(system) for system in H2]
    with calculator.prepare_batch(H2, warm_start=False) as batch:
        first = batch.execute(strict=True)
        second = batch.execute(strict=True)
        energy_only = batch.execute(properties=("energy",), strict=True)
        changed_coordinates = np.asarray(
            [[0.0, 0.0, -0.75], [0.0, 0.0, 0.75]], dtype=np.float64
        )
        changed = batch.execute([changed_coordinates, None], strict=True)

    for replay in (first, second):
        for item, reference in zip(replay.items, expected, strict=True):
            assert item.energy == pytest.approx(reference.energy, abs=1.0e-10)
            np.testing.assert_allclose(
                item.forces, reference.forces, atol=2.0e-9, rtol=0
            )
            assert not item.warm_start_used and not item.warm_start_fallback
    np.testing.assert_allclose(
        energy_only.energies, first.energies, atol=1.0e-10, rtol=0
    )
    assert all(item.forces is None for item in energy_only.items)
    changed_reference = calculator.singlepoint(
        [("H", (0.0, 0.0, -0.75)), ("H", (0.0, 0.0, 0.75))]
    )
    assert changed.items[0].energy == pytest.approx(
        changed_reference.energy, abs=1.0e-10
    )
    np.testing.assert_allclose(
        changed.items[0].forces, changed_reference.forces, atol=2.0e-9, rtol=0
    )
    assert changed.items[1].energy == pytest.approx(expected[1].energy, abs=1.0e-10)

    reversed_result = calculator.batch_singlepoint(list(reversed(H2)), strict=True)
    np.testing.assert_allclose(
        reversed_result.energies, first.energies[::-1], atol=1.0e-10, rtol=0
    )


def test_mp2_batch_failure_is_item_local_and_later_replay_is_clean() -> None:
    calculator = Calculator(method="mp2", device="cpu")
    with calculator.prepare_batch(H2, warm_start=False) as batch:
        baseline = batch.execute(strict=True)
        failed = batch.execute([np.zeros((1, 3)), None])
        assert failed.failure_indices == (0,)
        assert failed.items[0].forces is None
        assert failed.items[1].succeeded
        assert failed.items[1].energy == pytest.approx(
            baseline.items[1].energy, abs=1.0e-10
        )
        np.testing.assert_allclose(
            failed.items[1].forces, baseline.items[1].forces, atol=2.0e-9, rtol=0
        )
        recovered = batch.execute(strict=True)
    np.testing.assert_allclose(
        recovered.energies, baseline.energies, atol=1.0e-10, rtol=0
    )
    for item, reference in zip(recovered.items, baseline.items, strict=True):
        np.testing.assert_allclose(item.forces, reference.forces, atol=2.0e-9, rtol=0)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1",
    reason="requires explicitly allocated CUDA device and native library",
)
def test_mp2_cuda_batch_matches_cpu_and_isolates_failed_items() -> None:
    cpu = Calculator(method="mp2", device="cpu").batch_singlepoint(H2, strict=True)
    calculator = Calculator(method="mp2", device="cuda")
    with calculator.prepare_batch(H2, warm_start=False) as batch:
        cuda = batch.execute(strict=True)
        failed = batch.execute([np.zeros((1, 3)), None])
        recovered = batch.execute(strict=True)
    np.testing.assert_allclose(cuda.energies, cpu.energies, atol=1.0e-9, rtol=0)
    for item, reference in zip(cuda.items, cpu.items, strict=True):
        assert item.executed_backend == "cuda"
        np.testing.assert_allclose(item.forces, reference.forces, atol=2.0e-9, rtol=0)
    assert failed.failure_indices == (0,)
    assert failed.items[0].forces is None
    assert failed.items[1].succeeded and failed.items[1].executed_backend == "cuda"
    np.testing.assert_allclose(
        failed.items[1].forces, cuda.items[1].forces, atol=2.0e-9, rtol=0
    )
    np.testing.assert_allclose(recovered.energies, cuda.energies, atol=1.0e-9, rtol=0)
