"""Independent algebra and fail-closed identity tests for #190 slice A."""

from __future__ import annotations

import typing
from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_cc import (
    StateTransport,
    StateTransportRequest,
    TransportCompatibility,
)
from tools.vibeqc_posthf.reference import ReferenceSnapshot


def _snapshot(
    nmo: int,
    nocc: int,
    coefficients: np.ndarray,
    *,
    basis: str = "basis-a",
    generation: str = "generation-a",
    algorithm: str = "RHF",
    screening_tolerance: float = 0.0,
    reference_energy: float = -1.0,
    orbital_energies: np.ndarray | None = None,
) -> ReferenceSnapshot:
    energies = (
        np.concatenate((np.full(nocc, -1.0), np.full(nmo - nocc, 0.5)))
        if orbital_energies is None
        else np.asarray(orbital_energies, dtype=np.float64)
    )
    occupations = np.zeros(nmo)
    occupations[:nocc] = 2.0
    fock = coefficients @ np.diag(energies) @ coefficients.T
    return ReferenceSnapshot(
        overlap=np.eye(nmo),
        hcore=np.zeros((nmo, nmo)),
        fock=fock,
        coefficients=coefficients,
        orbital_energies=energies,
        occupations=occupations,
        electron_count=2 * nocc,
        reference_energy=reference_energy,
        scf_residual=0.0,
        geometry_hash="same-geometry",
        basis_hash=basis,
        generation_id=generation,
        algorithm=algorithm,
        screening_tolerance=screening_tolerance,
        functional_identity="test-functional" if algorithm == "KS" else None,
        grid_identity="test-grid" if algorithm == "KS" else None,
    )


def _request(
    source: ReferenceSnapshot,
    target: ReferenceSnapshot,
    cross_overlap: np.ndarray,
    *,
    source_equation: str = "rccsd-equations-v1",
    target_equation: str = "rccsd-equations-v1",
    source_provider: str = "conventional-integrals-v1",
    target_provider: str = "conventional-integrals-v1",
    source_settings: typing.Mapping[str, str] | tuple[tuple[str, str], ...] = (),
    target_settings: typing.Mapping[str, str] | tuple[tuple[str, str], ...] = (),
) -> StateTransportRequest:
    return StateTransportRequest.from_snapshots(
        source,
        target,
        cross_overlap,
        source_equation_id=source_equation,
        target_equation_id=target_equation,
        source_integral_provider_id=source_provider,
        target_integral_provider_id=target_provider,
        source_approximation_settings=source_settings,
        target_approximation_settings=target_settings,
    )


def _amplitudes(
    rng: np.random.Generator, occupied: int, virtual: int
) -> tuple[np.ndarray, np.ndarray]:
    t1 = rng.normal(size=(occupied, virtual))
    raw = rng.normal(size=(occupied, occupied, virtual, virtual))
    t2 = 0.5 * (raw + raw.transpose(1, 0, 3, 2))
    return t1, t2


def _explicit_rotate(
    occupied_map: np.ndarray,
    virtual_map: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    out1 = np.zeros((occupied_map.shape[0], virtual_map.shape[0]))
    out2 = np.zeros(
        (
            occupied_map.shape[0],
            occupied_map.shape[0],
            virtual_map.shape[0],
            virtual_map.shape[0],
        )
    )
    for k in range(out1.shape[0]):
        for c in range(out1.shape[1]):
            for i in range(t1.shape[0]):
                for a in range(t1.shape[1]):
                    out1[k, c] += occupied_map[k, i] * t1[i, a] * virtual_map[c, a]
    for k in range(out2.shape[0]):
        for l in range(out2.shape[1]):
            for c in range(out2.shape[2]):
                for d in range(out2.shape[3]):
                    for i in range(t2.shape[0]):
                        for j in range(t2.shape[1]):
                            for a in range(t2.shape[2]):
                                for b in range(t2.shape[3]):
                                    out2[k, l, c, d] += (
                                        occupied_map[k, i]
                                        * occupied_map[l, j]
                                        * virtual_map[c, a]
                                        * virtual_map[d, b]
                                        * t2[i, j, a, b]
                                    )
    return out1, out2


def _rccsd_energy_terms(
    fock: np.ndarray, eri: np.ndarray, t1: np.ndarray, t2: np.ndarray
) -> tuple[float, float, float]:
    occupied = t1.shape[0]
    fov = fock[:occupied, occupied:]
    ovov = eri[:occupied, occupied:, :occupied, occupied:]
    coulomb = ovov.transpose(0, 2, 1, 3)
    exchange = ovov.transpose(0, 2, 3, 1)
    weighted = 2.0 * coulomb - exchange
    return (
        float(2.0 * np.einsum("ia,ia", fov, t1)),
        float(np.einsum("ijab,ijab", weighted, t2)),
        float(np.einsum("ijab,ia,jb", weighted, t1, t1)),
    )


def _rotate_rank_four(frame: np.ndarray, tensor: np.ndarray) -> np.ndarray:
    return np.einsum(
        "xp,yq,zr,ws,pqrs->xyzw",
        frame,
        frame,
        frame,
        frame,
        tensor,
        optimize=False,
    )


def test_identity_transport_preserves_exact_arrays_and_complete_identities() -> None:
    snapshot = _snapshot(4, 2, np.eye(4))
    request = _request(snapshot, snapshot, np.eye(4))
    plan = StateTransport.classify(request)
    assert plan.compatibility is TransportCompatibility.identity
    assert plan.source.reference_id == snapshot.identity
    assert plan.source.basis_id == snapshot.basis_hash
    assert plan.source.occupied_orbitals == (0, 1)
    assert plan.source.virtual_orbitals == (2, 3)
    assert plan.source.frozen_core_orbitals == ()
    assert plan.source.equation_id == "rccsd-equations-v1"
    assert plan.source.integral_provider_id == "conventional-integrals-v1"
    t1, t2 = _amplitudes(np.random.default_rng(2), 2, 2)
    output = plan.rotate_amplitudes(t1, t2)
    assert output.reference_id == snapshot.identity
    np.testing.assert_array_equal(output.t1, t1)
    np.testing.assert_array_equal(output.t2, t2)


def test_block_rotation_matches_explicit_loops_and_transformed_energy() -> None:
    occupied_rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    virtual_seed = np.array([[1.0, 1.0, 0.0], [-1.0, 1.0, 1.0], [1.0, -1.0, 2.0]])
    virtual_rotation, _ = np.linalg.qr(virtual_seed)
    frame = np.zeros((5, 5))
    frame[:2, :2] = occupied_rotation
    frame[2:, 2:] = virtual_rotation
    source = _snapshot(5, 2, np.eye(5))
    target = _snapshot(5, 2, frame, generation="generation-b")
    plan = StateTransport.classify(_request(source, target, np.eye(5)))
    assert plan.compatibility is TransportCompatibility.exact_orbital_rotation
    np.testing.assert_allclose(plan.occupied_map, occupied_rotation.T, atol=1e-14)
    np.testing.assert_allclose(plan.virtual_map, virtual_rotation.T, atol=1e-14)
    assert plan.diagnostics.occupied_virtual_mixing == 0.0
    assert plan.diagnostics.occupied_unitarity_error < 1e-14
    assert plan.diagnostics.virtual_unitarity_error < 1e-14

    rng = np.random.default_rng(14)
    t1, t2 = _amplitudes(rng, 2, 3)
    expected1, expected2 = _explicit_rotate(plan.occupied_map, plan.virtual_map, t1, t2)
    output = plan.rotate_amplitudes(t1, t2)
    np.testing.assert_allclose(output.t1, expected1, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(output.t2, expected2, atol=4e-14, rtol=4e-14)
    np.testing.assert_allclose(output.t2, output.t2.transpose(1, 0, 3, 2))

    source_fock = rng.normal(size=(5, 5))
    source_fock = 0.5 * (source_fock + source_fock.T)
    source_eri = rng.normal(size=(5, 5, 5, 5))
    target_from_source = frame.T
    target_fock = target_from_source @ source_fock @ target_from_source.T
    target_eri = _rotate_rank_four(target_from_source, source_eri)
    np.testing.assert_allclose(
        _rccsd_energy_terms(source_fock, source_eri, t1, t2),
        _rccsd_energy_terms(target_fock, target_eri, output.t1, output.t2),
        atol=2e-12,
        rtol=2e-12,
    )


def test_rank_compatible_basis_expansion_is_only_a_projected_candidate() -> None:
    source = _snapshot(3, 1, np.eye(3), basis="basis-small")
    target = _snapshot(4, 1, np.eye(4), basis="basis-large")
    cross = np.zeros((4, 3))
    cross[:3, :3] = np.eye(3)
    plan = StateTransport.classify(_request(source, target, cross))
    assert plan.compatibility is TransportCompatibility.projected_warm_start
    assert plan.diagnostics.occupied_rank == 1
    assert plan.diagnostics.virtual_rank == 2
    assert plan.diagnostics.source_virtual_nullity == 0
    assert plan.diagnostics.target_virtual_nullity == 1
    t1, t2 = _amplitudes(np.random.default_rng(8), 1, 2)
    with pytest.raises(ValueError, match="projected_warm_start"):
        plan.rotate_amplitudes(t1, t2)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_equation": "different-equations"}, "equation identity"),
        ({"target_provider": "different-provider"}, "integral-provider identity"),
        (
            {"target_settings": {"screening": "loose"}},
            "approximation settings",
        ),
    ],
)
def test_shape_match_does_not_override_equation_provider_or_approximation_identity(
    overrides: dict[str, typing.Any], message: str
) -> None:
    source = _snapshot(4, 2, np.eye(4))
    target = _snapshot(4, 2, np.eye(4), generation="generation-b")
    plan = StateTransport.classify(_request(source, target, np.eye(4), **overrides))
    assert plan.compatibility is TransportCompatibility.incompatible
    assert any(message in reason for reason in plan.messages)


def test_core_and_occupation_identity_changes_fail_closed() -> None:
    source = _snapshot(4, 2, np.eye(4))
    target = _snapshot(4, 2, np.eye(4), generation="generation-b")
    request = _request(source, target, np.eye(4))
    changed_core = replace(request.target, frozen_core_orbitals=(0,))
    core_plan = StateTransport.classify(replace(request, target=changed_core))
    assert core_plan.compatibility is TransportCompatibility.incompatible
    assert any("frozen-core" in reason for reason in core_plan.messages)
    changed_occupation = replace(request.target, occupations=(1.0, 1.0, 0.0, 0.0))
    occupation_plan = StateTransport.classify(
        replace(request, target=changed_occupation)
    )
    assert occupation_plan.compatibility is TransportCompatibility.incompatible
    assert any("2/0 occupation" in reason for reason in occupation_plan.messages)


def test_reference_kind_and_spin_changes_fail_closed() -> None:
    source = _snapshot(4, 2, np.eye(4))
    target = _snapshot(4, 2, np.eye(4), generation="generation-b", algorithm="KS")
    request = _request(source, target, np.eye(4))
    reference_plan = StateTransport.classify(request)
    assert reference_plan.compatibility is TransportCompatibility.incompatible
    assert any("reference kind" in reason for reason in reference_plan.messages)
    spin_plan = StateTransport.classify(
        replace(request, target=replace(request.target, reference_kind="RHF", spin=1))
    )
    assert spin_plan.compatibility is TransportCompatibility.incompatible
    assert any("spin identity" in reason for reason in spin_plan.messages)


def test_reference_settings_and_operator_changes_fail_closed() -> None:
    source = _snapshot(4, 2, np.eye(4))
    screened = _snapshot(
        4,
        2,
        np.eye(4),
        generation="generation-b",
        screening_tolerance=1e-8,
    )
    screening_plan = StateTransport.classify(_request(source, screened, np.eye(4)))
    assert screening_plan.compatibility is TransportCompatibility.incompatible
    assert any("screening" in reason for reason in screening_plan.messages)

    shifted_reference = _snapshot(
        4, 2, np.eye(4), generation="generation-b", reference_energy=-0.9
    )
    reference_plan = StateTransport.classify(
        _request(source, shifted_reference, np.eye(4))
    )
    assert reference_plan.compatibility is TransportCompatibility.incompatible
    assert any(
        "reference energy/operator" in reason for reason in reference_plan.messages
    )

    shifted_orbitals = _snapshot(
        4,
        2,
        np.eye(4),
        generation="generation-b",
        orbital_energies=np.array([-1.0, -1.0, 0.5, 0.7]),
    )
    orbital_plan = StateTransport.classify(
        _request(source, shifted_orbitals, np.eye(4))
    )
    assert orbital_plan.compatibility is TransportCompatibility.incompatible
    assert any(
        "reference energy/operator" in reason for reason in orbital_plan.messages
    )


def test_occupied_virtual_mixing_rejects_reference_change() -> None:
    angle = 0.15
    mixed = np.eye(4)
    mixed[0, 0] = mixed[2, 2] = np.cos(angle)
    mixed[0, 2] = -np.sin(angle)
    mixed[2, 0] = np.sin(angle)
    source = _snapshot(4, 2, np.eye(4))
    target = _snapshot(4, 2, mixed, generation="generation-b")
    plan = StateTransport.classify(_request(source, target, np.eye(4)))
    assert plan.compatibility is TransportCompatibility.incompatible
    assert plan.diagnostics.occupied_virtual_mixing > 0.1
    assert any("reference determinant" in reason for reason in plan.messages)


def test_lost_occupied_rank_and_stale_same_reference_map_are_rejected() -> None:
    source = _snapshot(3, 1, np.eye(3), basis="basis-small")
    target = _snapshot(4, 1, np.eye(4), basis="basis-large")
    cross = np.zeros((4, 3))
    cross[1:3, 1:3] = np.eye(2)
    lost = StateTransport.classify(_request(source, target, cross))
    assert lost.compatibility is TransportCompatibility.incompatible
    assert any("loses rank" in reason for reason in lost.messages)

    same = _snapshot(3, 1, np.eye(3))
    stale_overlap = np.eye(3)
    stale_overlap[0, 0] = -1.0
    stale = StateTransport.classify(_request(same, same, stale_overlap))
    assert stale.compatibility is TransportCompatibility.incompatible
    assert any("identical orbital frame" in reason for reason in stale.messages)


def test_endpoint_frame_arrays_are_bound_and_transport_is_factory_only() -> None:
    occupied_rotation = np.array([[0.0, -1.0], [1.0, 0.0]])
    frame = np.eye(4)
    frame[:2, :2] = occupied_rotation
    source = _snapshot(4, 2, np.eye(4))
    target = _snapshot(4, 2, frame, generation="generation-b")
    request = _request(source, target, np.eye(4))
    with pytest.raises(ValueError, match="target coefficient array"):
        replace(request, target_coefficients=source.coefficients)

    plan = StateTransport.classify(request)
    with pytest.raises(TypeError, match=r"created by classify\(\)"):
        StateTransport(
            plan.source,
            plan.target,
            TransportCompatibility.exact_orbital_rotation,
            np.eye(2),
            np.eye(2),
            plan.diagnostics,
            ("forged exact decision",),
            plan.policy,
        )


def test_equal_core_index_masks_cannot_hide_core_active_rotation() -> None:
    occupied_rotation = np.array([[0.8, -0.6], [0.6, 0.8]])
    frame = np.eye(4)
    frame[:2, :2] = occupied_rotation
    source = _snapshot(4, 2, np.eye(4))
    target = _snapshot(4, 2, frame, generation="generation-b")
    request = _request(source, target, np.eye(4))
    request = replace(
        request,
        source=replace(request.source, frozen_core_orbitals=(0,)),
        target=replace(request.target, frozen_core_orbitals=(0,)),
    )
    plan = StateTransport.classify(request)
    assert plan.compatibility is TransportCompatibility.incompatible
    assert any("frozen-core state transport" in reason for reason in plan.messages)


@pytest.mark.parametrize(
    ("occupied", "virtual"),
    (((0, 0), (1, 2)), ((0,), (1, 1, 2))),
)
def test_state_identity_rejects_duplicate_partition_indices(
    occupied: tuple[int, ...], virtual: tuple[int, ...]
) -> None:
    """A partition must neither duplicate amplitude axes nor omit orbitals."""
    snapshot = _snapshot(3, 1, np.eye(3))
    request = _request(snapshot, snapshot, np.eye(3))
    with pytest.raises(ValueError, match="partition"):
        replace(
            request.source,
            occupied_orbitals=occupied,
            virtual_orbitals=virtual,
        )
