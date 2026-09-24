from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

from tools.vibeqc_cc.gpu_state import AmplitudeSnapshot
from tools.vibeqc_cc.history_transport import (
    HistoryRecyclePolicy,
    TargetResidualEvaluator,
    recycle_diis_history,
)
from tools.vibeqc_cc.state_transport import (
    StateIdentity,
    StateTransport,
    StateTransportRequest,
    TransportCompatibility,
)

if TYPE_CHECKING:
    from collections.abc import Callable


def _frame_hash(value: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(value, dtype="<f8").tobytes()
    ).hexdigest()


def _identity(
    *,
    reference_id: str,
    basis_id: str,
    coefficients: np.ndarray,
    nvir: int,
    equation_id: str = "rccsd-equations",
) -> StateIdentity:
    nmo = 1 + nvir
    return StateIdentity(
        reference_id=reference_id,
        geometry_id="geometry",
        basis_id=basis_id,
        ao_representation="spherical",
        reference_kind="RHF",
        precision="float64",
        screening_tolerance=0.0,
        overlap_threshold=1e-10,
        reference_energy=-1.0,
        orbital_energies=(-0.8,) + tuple(0.2 + 0.1 * i for i in range(nvir)),
        functional_identity=None,
        grid_identity=None,
        coefficient_frame_hash=_frame_hash(coefficients),
        spin=0,
        electron_count=2,
        occupations=(2.0,) + (0.0,) * nvir,
        occupied_orbitals=(0,),
        virtual_orbitals=tuple(range(1, nmo)),
        frozen_core_orbitals=(),
        hamiltonian_id="hamiltonian",
        equation_id=equation_id,
        integral_provider_id="integrals",
    )


def _transport(
    *, projected: bool = False, incompatible: bool = False
) -> StateTransport:
    source_coefficients = np.eye(2, dtype=np.float64)
    target_coefficients = np.eye(3 if projected else 2, dtype=np.float64)
    source = _identity(
        reference_id="source-reference",
        basis_id="source-basis",
        coefficients=source_coefficients,
        nvir=1,
    )
    target = _identity(
        reference_id="target-reference",
        basis_id="target-basis",
        coefficients=target_coefficients,
        nvir=2 if projected else 1,
        equation_id="different-equations" if incompatible else "rccsd-equations",
    )
    if projected:
        cross = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=np.float64)
    else:
        cross = np.eye(2, dtype=np.float64)
    return StateTransport.classify(
        StateTransportRequest(
            source,
            target,
            source_coefficients,
            target_coefficients,
            cross,
        )
    )


def _snapshot(
    value: float, *, reference_id: str = "source-reference"
) -> AmplitudeSnapshot:
    return AmplitudeSnapshot(
        reference_id,
        np.array([[value]], dtype=np.float64),
        np.array([[[[value * value]]]], dtype=np.float64),
    )


def _target_residual(amplitudes: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
    return amplitudes.t1 * 2.0, amplitudes.t2 * 3.0


def _evaluator(
    transport: StateTransport,
    evaluate: Callable[
        [AmplitudeSnapshot], tuple[np.ndarray, np.ndarray]
    ] = _target_residual,
) -> TargetResidualEvaluator:
    return TargetResidualEvaluator(transport.target.identity, evaluate)


def test_exact_history_recomputes_every_target_residual() -> None:
    transport = _transport()
    assert transport.compatibility is TransportCompatibility.exact_orbital_rotation
    seen: list[str] = []

    def residual(amplitudes: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        seen.append(amplitudes.reference_id)
        return _target_residual(amplitudes)

    result = recycle_diis_history(
        transport, [_snapshot(0.1), _snapshot(0.2)], _evaluator(transport, residual)
    )

    assert result.kind == "target_recomputed_diis_history"
    assert result.transport_id == transport.identity
    assert result.target_reference_id == transport.target.reference_id
    assert result.target_residual_evaluations == 2
    assert seen == [transport.target.reference_id] * 2
    np.testing.assert_array_equal(result.entries[1].residual_singles, [[0.4]])
    np.testing.assert_array_equal(result.entries[1].residual_doubles, [[[[0.12]]]])
    assert not result.entries[1].residual_singles.flags.writeable
    assert not result.entries[1].residual_doubles.flags.writeable


def test_projected_history_uses_target_space_before_residual() -> None:
    transport = _transport(projected=True)
    assert transport.compatibility is TransportCompatibility.projected_warm_start
    seen: list[np.ndarray] = []

    def residual(amplitudes: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        seen.append(amplitudes.t1.copy())
        return _target_residual(amplitudes)

    result = recycle_diis_history(
        transport, [_snapshot(0.2)], _evaluator(transport, residual)
    )

    np.testing.assert_array_equal(seen[0], [[0.2, 0.0]])
    np.testing.assert_array_equal(result.entries[0].amplitudes.t1, [[0.2, 0.0]])
    np.testing.assert_array_equal(result.entries[0].residual_singles, [[0.4, 0.0]])


def test_history_capacity_retains_newest_vectors_with_source_indices() -> None:
    transport = _transport()
    result = recycle_diis_history(
        transport,
        [_snapshot(0.1), _snapshot(0.2), _snapshot(0.3)],
        _evaluator(transport),
        policy=HistoryRecyclePolicy(maximum_vectors=2),
    )

    assert result.source_count == 3
    assert result.dropped_prefix_count == 1
    assert [entry.source_index for entry in result.entries] == [1, 2]
    np.testing.assert_array_equal(result.entries[0].amplitudes.t1, [[0.2]])
    np.testing.assert_array_equal(result.entries[1].amplitudes.t1, [[0.3]])


def test_history_budget_fails_before_target_operator_call() -> None:
    transport = _transport()
    calls = 0

    def residual(amplitudes: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return _target_residual(amplitudes)

    with pytest.raises(ValueError, match="element budget"):
        recycle_diis_history(
            transport,
            [_snapshot(0.1), _snapshot(0.2)],
            _evaluator(transport, residual),
            policy=HistoryRecyclePolicy(maximum_vectors=2, maximum_total_elements=7),
        )
    assert calls == 0


def test_source_reference_mismatch_fails_before_target_operator_call() -> None:
    transport = _transport()
    calls = 0

    def residual(amplitudes: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return _target_residual(amplitudes)

    with pytest.raises(ValueError, match="transport source"):
        recycle_diis_history(
            transport,
            [_snapshot(0.1, reference_id="stale")],
            _evaluator(transport, residual),
        )
    assert calls == 0


def test_invalid_target_residual_cannot_publish_history() -> None:
    transport = _transport()

    def wrong_shape(amplitudes: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros((2, 2), dtype=np.float64), amplitudes.t2.copy()

    with pytest.raises(ValueError, match="matching target amplitudes"):
        recycle_diis_history(
            transport, [_snapshot(0.1)], _evaluator(transport, wrong_shape)
        )


def test_incompatible_transport_requires_reset_without_target_call() -> None:
    transport = _transport(incompatible=True)
    assert transport.compatibility is TransportCompatibility.incompatible
    calls = 0

    def residual(amplitudes: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return _target_residual(amplitudes)

    with pytest.raises(ValueError, match="history reset"):
        recycle_diis_history(
            transport, [_snapshot(0.1)], _evaluator(transport, residual)
        )
    assert calls == 0


def test_target_residual_identity_mismatch_fails_before_operator_call() -> None:
    transport = _transport()
    calls = 0

    def residual(amplitudes: AmplitudeSnapshot) -> tuple[np.ndarray, np.ndarray]:
        nonlocal calls
        calls += 1
        return _target_residual(amplitudes)

    evaluator = TargetResidualEvaluator("stale-target-identity", residual)
    with pytest.raises(ValueError, match="target identity"):
        recycle_diis_history(transport, [_snapshot(0.1)], evaluator)
    assert calls == 0
