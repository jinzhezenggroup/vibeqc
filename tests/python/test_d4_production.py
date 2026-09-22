"""Production D4(BJ)-EEQ runtime gates against an independent dftd4 fixture."""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Calculator, D4CorrectionBatch, ResourceBudget, evaluate_d4_correction

_NUMBERS = np.array([6, 8, 7, 1], dtype=np.int32)
_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],
        [2.2487740883046663, 0.24566439620135014, -0.075589044985030815],
        [-0.88817127857411193, 2.4755412232597589, 0.41573974741766945],
        [0.58581509863398873, -1.1905274585142351, 1.6818562509169355],
    ],
    dtype=np.float64,
)
_ENERGY = -0.0014151210214956252
_GRADIENT = np.array(
    [
        [-5.2831716631508658e-06, -8.3862351553618974e-06, -2.9748087148538252e-06],
        [-1.211398469656972e-05, -2.3753812372069782e-06, 1.2806873212180398e-05],
        [4.1185850552751853e-06, 6.4799601530310636e-06, -1.3029967085337256e-06],
        [1.3278571304445404e-05, 4.2816562395378085e-06, -8.5290677887928447e-06],
    ],
    dtype=np.float64,
)
_CHARGES = np.array(
    [
        0.30521400078622163,
        -0.27300129311378285,
        -0.24205034919130103,
        0.20983764151886222,
    ],
    dtype=np.float64,
)


def _system() -> tuple[np.ndarray, np.ndarray, float]:
    return _NUMBERS, _POSITIONS, 0.0


def test_production_cpu_matches_independent_dftd4() -> None:
    result = evaluate_d4_correction(
        "PBE-D4(BJ-EEQ-ATM)", _NUMBERS, _POSITIONS, device="cpu"
    )
    assert result.ok
    assert result.backend == "cpu"
    assert result.energy == pytest.approx(_ENERGY, abs=2.0e-13)
    np.testing.assert_allclose(result.gradient, _GRADIENT, atol=2.0e-12, rtol=0.0)
    np.testing.assert_allclose(result.charges, _CHARGES, atol=1.0e-10, rtol=0.0)


def test_production_replay_identity_resources_and_changed_geometry() -> None:
    with D4CorrectionBatch("PBE-D4(BJ-EEQ-ATM)", [_system()], device="cpu") as batch:
        before = batch.diagnostic()
        assert before.execution_count == 0
        assert before.system_count == 1
        assert before.total_atoms == len(_NUMBERS)
        assert before.workspace_slots == 1
        assert before.workspace_bytes > 0
        assert before.peak_host_bytes >= before.plan_host_bytes
        assert before.provider_identity == "vibeqc-native-d4-bj-eeq-v1"
        assert (
            before.scheduler_identity
            == "bounded-eeq-workers-cooperative-fixed-charge-v1"
        )
        assert len(before.cache_identity) == 64

        initial = batch.execute()[0]
        replay = batch.execute()[0]
        assert initial.energy == pytest.approx(replay.energy, abs=2.0e-15)

        moved = _POSITIONS.copy()
        moved[1, 0] += 0.013
        changed = batch.execute([moved])[0]
        independent = evaluate_d4_correction(
            "PBE-D4(BJ-EEQ-ATM)", _NUMBERS, moved, device="cpu"
        )
        assert changed.energy == pytest.approx(independent.energy, abs=2.0e-15)
        np.testing.assert_allclose(changed.gradient, independent.gradient, atol=2.0e-14)

        after = batch.diagnostic()
        assert after.execution_count == 3
        assert after.unchanged_geometry_replays == 2
        assert after.changed_geometry_replays == 1


def test_production_budget_is_bounded() -> None:
    with pytest.raises(RuntimeError, match="maximum_bytes"):
        D4CorrectionBatch("PBE-D4(BJ-EEQ-ATM)", [_system()], maximum_bytes=1)


def test_public_named_pbe_d4_cpu_adds_native_correction() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    pbe = Calculator(method="pbe-rks", basis="sto-3g", device="cpu").singlepoint(
        atoms, properties=("energy",)
    )
    d4 = evaluate_d4_correction(
        "PBE-D4(BJ-EEQ-ATM)",
        np.array([1, 1], dtype=np.int32),
        np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=np.float64),
        device="cpu",
    )
    combined = Calculator(
        method="pbe-d4-rks", basis="sto-3g", device="cpu"
    ).singlepoint(atoms, properties=("energy",))
    assert combined.executed_backend == "cpu_reference"
    assert combined.energy == pytest.approx(pbe.energy + d4.energy, abs=2.0e-12)
    with pytest.raises(NotImplementedError, match="strict FP64"):
        Calculator(method="pbe-d4-rks", basis="sto-3g", device="cuda", precision="auto")


def test_public_named_pbe_d4_cpu_force_matches_reconverged_total_energy() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    calculator = Calculator(method="pbe-d4-rks", basis="sto-3g", device="cpu")
    result = calculator.singlepoint(atoms, properties=("energy", "forces"))
    assert result.dispersion is not None

    numbers = np.array([1, 1], dtype=np.int32)
    positions = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=np.float64)
    d4 = evaluate_d4_correction("PBE-D4(BJ-EEQ-ATM)", numbers, positions, device="cpu")
    assert result.dispersion.energy == pytest.approx(d4.energy, abs=2.0e-13)
    np.testing.assert_allclose(
        result.dispersion.gradient, d4.gradient, atol=2.0e-12, rtol=0.0
    )

    step = 2.0e-4
    energies = []
    for sign in (-1.0, 1.0):
        displaced = positions.copy()
        displaced[1, 2] += sign * step
        moved = [
            (int(z), tuple(xyz)) for z, xyz in zip(numbers, displaced, strict=True)
        ]
        energies.append(
            Calculator(method="pbe-d4-rks", basis="sto-3g", device="cpu")
            .singlepoint(moved, properties=("energy",))
            .energy
        )
    finite_difference_force = -(energies[1] - energies[0]) / (2.0 * step)
    assert result.forces is not None
    assert result.forces[1, 2] == pytest.approx(
        finite_difference_force, abs=3.0e-5, rel=0.0
    )


def test_public_named_pbe_d4_force_changed_geometry_replay() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    moved = np.array([[0.0, 0.0, -0.72], [0.0, 0.0, 0.72]], dtype=np.float64)
    calculator = Calculator(method="pbe-d4-rks", basis="sto-3g", device="cpu")
    with calculator.prepare_batch([atoms], warm_start=False) as batch:
        first = batch.execute(strict=True, properties=("energy", "forces")).items[0]
        replay = batch.execute(
            [moved], strict=True, properties=("energy", "forces")
        ).items[0]
        diagnostic = batch.dispersion_diagnostic
        assert diagnostic is not None
        assert diagnostic.execution_count == 2
        assert diagnostic.changed_geometry_replays == 1

    fresh_atoms = [(1, tuple(position)) for position in moved]
    fresh = Calculator(method="pbe-d4-rks", basis="sto-3g", device="cpu").singlepoint(
        fresh_atoms, properties=("energy", "forces")
    )
    assert first.dispersion is not None
    assert replay.dispersion is not None
    assert replay.energy == pytest.approx(fresh.energy, abs=2.0e-11)
    np.testing.assert_allclose(replay.forces, fresh.forces, atol=2.0e-9, rtol=0.0)


def test_public_named_pbe_d4_global_resource_plan_fails_closed() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    calculator = Calculator(
        method="pbe-d4-rks",
        basis="sto-3g",
        device="cpu",
        resource_budget=ResourceBudget(host_bytes=1 << 30),
    )
    with pytest.raises(NotImplementedError, match="KS planning supports"):
        calculator.estimate_resources([atoms])


def test_production_cuda_matches_cpu_and_replay_accounting() -> None:
    try:
        batch = D4CorrectionBatch("PBE-D4(BJ-EEQ-ATM)", [_system()], device="cuda")
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(f"CUDA D4 runtime unavailable: {error}")

    with batch:
        initial = batch.execute()[0]
        replay = batch.execute()[0]
        assert initial.ok and replay.ok
        assert initial.backend == "cuda"
        assert initial.energy == pytest.approx(_ENERGY, abs=2.0e-13)
        np.testing.assert_allclose(initial.gradient, _GRADIENT, atol=2.0e-12, rtol=0.0)
        np.testing.assert_allclose(initial.charges, _CHARGES, atol=1.0e-10, rtol=0.0)

        moved = _POSITIONS.copy()
        moved[2, 1] -= 0.017
        changed = batch.execute([moved])[0]
        cpu = evaluate_d4_correction("PBE-D4(BJ-EEQ-ATM)", _NUMBERS, moved)
        assert changed.energy == pytest.approx(cpu.energy, abs=2.0e-13)
        np.testing.assert_allclose(
            changed.gradient, cpu.gradient, atol=2.0e-12, rtol=0.0
        )

        diagnostic = batch.diagnostic()
        assert diagnostic.backend == "cuda"
        assert diagnostic.worker_blocks == 1
        assert diagnostic.workspace_slots == 1
        assert diagnostic.kernel_launches == 27
        assert diagnostic.unchanged_geometry_replays == 2
        assert diagnostic.changed_geometry_replays == 1
        assert diagnostic.coordinate_h2d_bytes == moved.size * moved.itemsize


def test_public_named_pbe_d4_cuda_force_matches_cpu() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    cpu = Calculator(method="pbe-d4-rks", basis="sto-3g", device="cpu").singlepoint(
        atoms, properties=("energy", "forces")
    )
    try:
        cuda = Calculator(
            method="pbe-d4-rks", basis="sto-3g", device="cuda"
        ).singlepoint(atoms, properties=("energy", "forces"))
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(f"CUDA PBE-D4 force runtime unavailable: {error}")

    assert cuda.executed_backend == "cuda"
    assert cuda.dispersion is not None
    assert cuda.dispersion.backend == "cuda"
    assert cuda.energy == pytest.approx(cpu.energy, abs=1.0e-10)
    np.testing.assert_allclose(cuda.forces, cpu.forces, atol=2.0e-7, rtol=0.0)


def test_public_named_pbe_d4_cuda_matches_cpu() -> None:
    atoms = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
    numbers = np.array([1, 1], dtype=np.int32)
    positions = np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]], dtype=np.float64)
    pair_cpu = evaluate_d4_correction(
        "PBE-D4(BJ-EEQ-ATM)", numbers, positions, device="cpu"
    )
    try:
        pair_cuda = evaluate_d4_correction(
            "PBE-D4(BJ-EEQ-ATM)", numbers, positions, device="cuda"
        )
    except (RuntimeError, NotImplementedError) as error:
        pytest.skip(f"CUDA D4 runtime unavailable: {error}")
    assert pair_cuda.energy == pytest.approx(pair_cpu.energy, abs=2.0e-13)
    np.testing.assert_allclose(
        pair_cuda.charges, pair_cpu.charges, atol=1.0e-12, rtol=0.0
    )
    assert pair_cpu.atm_energy == pytest.approx(0.0, abs=1.0e-18)
    assert pair_cuda.atm_energy == pytest.approx(0.0, abs=1.0e-18)

    cpu = Calculator(method="pbe-d4-rks", basis="sto-3g", device="cpu").singlepoint(
        atoms, properties=("energy",)
    )
    try:
        cuda = Calculator(
            method="pbe-d4-rks", basis="sto-3g", device="cuda"
        ).singlepoint(atoms, properties=("energy",))
    except RuntimeError as error:
        pytest.skip(f"CUDA KS/D4 runtime unavailable: {error}")
    assert cuda.executed_backend == "cuda"
    assert cuda.energy == pytest.approx(cpu.energy, abs=1.0e-10)
