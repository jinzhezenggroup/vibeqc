from __future__ import annotations

import hashlib

import numpy as np
import pytest

from tools.vibeqc_cc.projected_transport import (
    ProjectedAmplitudePolicy,
    project_amplitude_guess,
)
from tools.vibeqc_cc.state_transport import (
    StateIdentity,
    StateTransport,
    StateTransportRequest,
    TransportCompatibility,
)


def _frame_hash(value: np.ndarray) -> str:
    data = np.ascontiguousarray(value, dtype="<f8").tobytes()
    return hashlib.sha256(data).hexdigest()


def _identity(
    *,
    reference_id: str,
    basis_id: str,
    coefficients: np.ndarray,
    nvir: int,
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
        equation_id="rccsd-equations",
        integral_provider_id="integrals",
    )


def _projected_transport(*, virtual_scale: float = 1.0) -> StateTransport:
    source_coefficients = np.eye(2, dtype=np.float64)
    target_coefficients = np.eye(3, dtype=np.float64)
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
        nvir=2,
    )
    cross = np.array([[1.0, 0.0], [0.0, virtual_scale], [0.0, 0.0]], dtype=np.float64)
    transport = StateTransport.classify(
        StateTransportRequest(
            source,
            target,
            source_coefficients,
            target_coefficients,
            cross,
        )
    )
    assert transport.compatibility is TransportCompatibility.projected_warm_start
    return transport


def _exact_transport() -> StateTransport:
    source_coefficients = np.eye(2, dtype=np.float64)
    target_coefficients = np.eye(2, dtype=np.float64)
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
        nvir=1,
    )
    transport = StateTransport.classify(
        StateTransportRequest(
            source,
            target,
            source_coefficients,
            target_coefficients,
            np.eye(2, dtype=np.float64),
        )
    )
    assert transport.compatibility is TransportCompatibility.exact_orbital_rotation
    return transport


def _source_amplitudes() -> tuple[np.ndarray, np.ndarray]:
    t1 = np.array([[0.2]], dtype=np.float64)
    t2 = np.array([[[[0.04]]]], dtype=np.float64)
    return t1, t2


def test_projected_guess_zero_fills_missing_virtual_direction() -> None:
    transport = _projected_transport()
    t1, t2 = _source_amplitudes()

    guess = project_amplitude_guess(transport, t1, t2)

    assert guess.kind == "projected_warm_start"
    assert guess.transport_id == transport.identity
    assert guess.amplitudes.reference_id == transport.target.reference_id
    np.testing.assert_array_equal(guess.amplitudes.t1, [[0.2, 0.0]])
    expected_t2 = np.zeros((1, 1, 2, 2), dtype=np.float64)
    expected_t2[0, 0, 0, 0] = 0.04
    np.testing.assert_array_equal(guess.amplitudes.t2, expected_t2)
    assert not guess.amplitudes.t1.flags.writeable
    assert not guess.amplitudes.t2.flags.writeable
    assert guess.diagnostics.source_elements == 2
    assert guess.diagnostics.target_elements == 6
    assert guess.diagnostics.virtual_rank == 1
    assert guess.diagnostics.target_virtual_nullity == 1
    assert guess.diagnostics.singles_norm_ratio == pytest.approx(1.0)
    assert guess.diagnostics.doubles_norm_ratio == pytest.approx(1.0)


def test_projection_rejects_exact_transport_to_preserve_exact_owner() -> None:
    t1, t2 = _source_amplitudes()
    with pytest.raises(ValueError, match="projected_warm_start"):
        project_amplitude_guess(_exact_transport(), t1, t2)


def test_projection_preflights_target_amplitude_budget() -> None:
    t1, t2 = _source_amplitudes()
    with pytest.raises(ValueError, match="element budget"):
        project_amplitude_guess(
            _projected_transport(),
            t1,
            t2,
            policy=ProjectedAmplitudePolicy(maximum_elements=5),
        )


def test_projection_rejects_overlap_map_that_amplifies_norm() -> None:
    t1, t2 = _source_amplitudes()
    with pytest.raises(ValueError, match="bounded overlap contraction"):
        project_amplitude_guess(_projected_transport(virtual_scale=1.01), t1, t2)


def test_projection_rejects_source_shape_that_only_matches_itself() -> None:
    t1 = np.array([[0.1], [0.2]], dtype=np.float64)
    t2 = np.zeros((2, 2, 1, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="transport identity"):
        project_amplitude_guess(_projected_transport(), t1, t2)


def test_zero_amplitudes_remain_zero_without_nan_diagnostics() -> None:
    transport = _projected_transport()
    t1 = np.zeros((1, 1), dtype=np.float64)
    t2 = np.zeros((1, 1, 1, 1), dtype=np.float64)

    guess = project_amplitude_guess(transport, t1, t2)

    assert not np.any(guess.amplitudes.t1)
    assert not np.any(guess.amplitudes.t2)
    assert guess.diagnostics.singles_norm_ratio == 1.0
    assert guess.diagnostics.doubles_norm_ratio == 1.0
