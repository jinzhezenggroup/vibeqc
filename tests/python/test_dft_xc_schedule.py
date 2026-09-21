"""DFT09 typed schedule identities, admission, and endpoint promotion gates."""

from __future__ import annotations

import copy

import pytest
from vibeqc.autotune import dft_endpoint_gate
from vibeqc_compiler.dft.grid import GridSpec, MolecularGrid, molecular_grid_identity
from vibeqc_compiler.dft.xc_schedule import (
    DEVICE_FUSED,
    HOST_UNFUSED,
    GridXcCandidateLimits,
    GridXcCandidateShape,
    GridXcScientificIdentity,
    assess_grid_xc_schedule,
    grid_xc_schedule,
)


def test_compact_grid_identity_matches_materialized_molecular_grid() -> None:
    atoms = [("H", (0.1, -0.2, -0.7)), ("H", (0.2, 0.1, 0.8))]
    spec = GridSpec(
        radial_points=8,
        angular_polar=4,
        angular_azimuth=8,
        partition_iterations=2,
    )
    compact = molecular_grid_identity(atoms, spec, charge=0, multiplicity=1)
    materialized = MolecularGrid(atoms, spec, charge=0, multiplicity=1)
    assert compact == materialized.identity


def scientific() -> GridXcScientificIdentity:
    return GridXcScientificIdentity(
        architecture="sm_120",
        functional="PBE",
        functional_identity="f" * 64,
        ingredients=("rho", "gradient", "sigma"),
        jet_outputs=((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)),
        grid_identity="g" * 64,
        grid_model="GridSpec-v2",
        screening_identity="m" * 64,
        precision="fp64",
        spin="polarized",
        observable="potential",
        density_route="density_matrix",
        source_identity="s" * 64,
    )


def test_scientific_identity_is_schedule_independent() -> None:
    identity = scientific()
    fused = DEVICE_FUSED.resolved(256)
    unfused = HOST_UNFUSED.resolved(256)
    assert fused.identity != unfused.identity
    assert identity.identity == scientific().identity
    assert "schedule" not in identity.to_payload()
    assert grid_xc_schedule(unfused.to_payload()) == unfused
    with pytest.raises(ValueError, match="profile payload"):
        grid_xc_schedule({**unfused.to_payload(), "unexpected": True})
    with pytest.raises(ValueError, match="fusion policy disagree"):
        grid_xc_schedule(
            {
                **unfused.to_payload(),
                "fusion": "device_xc_vxc",
            }
        )
    with pytest.raises(ValueError, match="jet residency"):
        grid_xc_schedule(
            {
                **unfused.to_payload(),
                "resident_jets": True,
            }
        )


def test_schedule_shape_requires_measured_resource_inputs() -> None:
    with pytest.raises(ValueError, match="measured workspace"):
        GridXcCandidateShape(
            npoint=1,
            tile_points=1,
            nao=1,
            max_active_ao=1,
            spins=1,
            jet_components=1,
            device_workspace_bytes=0,
            generated_source_bytes=1,
        )


def test_schedule_admission_is_deterministic_and_conservative() -> None:
    shape = GridXcCandidateShape(
        npoint=4096,
        tile_points=256,
        nao=96,
        max_active_ao=48,
        spins=2,
        jet_components=4,
        device_workspace_bytes=8 << 20,
        generated_source_bytes=180_000,
    )
    limits = GridXcCandidateLimits(
        device_bytes=32 << 20, live_values=1_000_000, source_bytes=300_000
    )
    first = assess_grid_xc_schedule(
        DEVICE_FUSED,
        shape,
        limits,
        device_xc_available=True,
        observable="potential",
        functional="PBE",
    )
    second = assess_grid_xc_schedule(
        DEVICE_FUSED,
        shape,
        limits,
        device_xc_available=True,
        observable="potential",
        functional="PBE",
    )
    assert first == second and first.legal
    assert first.live_values > 0

    rejected = assess_grid_xc_schedule(
        DEVICE_FUSED,
        shape,
        GridXcCandidateLimits(device_bytes=1, live_values=1, source_bytes=100),
        device_xc_available=False,
        observable="energy",
        functional="R2SCAN",
    )
    assert not rejected.legal
    assert {
        "native device XC capability is unavailable",
        "device-fused schedule only supports potential output",
        "device-fused schedule only supports canonical LDA/PBE",
        "estimated live-value bound exceeds target limit",
        "planned device workspace exceeds target limit",
        "generated source/compile-size bound exceeds target limit",
    } == set(rejected.reasons)

    fallback = assess_grid_xc_schedule(
        HOST_UNFUSED,
        shape,
        limits,
        device_xc_available=False,
        observable="potential",
        functional="R2SCAN",
    )
    assert fallback.legal
    assert fallback.live_values > first.live_values


def endpoint_sample(
    seconds: float,
    pair_id: int,
    *,
    scientific_identity: str = "workload",
    schedule_identity: str = "baseline-schedule",
    source_hash: str = "baseline-source",
    iterations: int = 7,
) -> dict:
    return {
        "seconds": seconds,
        "iterations": [iterations],
        "converged": True,
        "backend": ["cuda"],
        "energies": [-1.0],
        "forces": [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]],
        "scientific_identity": scientific_identity,
        "schedule_identity": schedule_identity,
        "source_hash": source_hash,
        "endpoint_kind": "energy_force",
        "synchronized": True,
        "interleaved": True,
        "pair_id": pair_id,
    }


def test_dft_endpoint_gate_requires_matched_complete_interleaved_evidence() -> None:
    baseline = [endpoint_sample(1.0, i) for i in range(6)]
    candidate = [
        endpoint_sample(
            0.8,
            i,
            schedule_identity="candidate-schedule",
            source_hash="candidate-source",
        )
        for i in range(6)
    ]
    accepted = dft_endpoint_gate(baseline, candidate)
    assert accepted["passed"]
    assert accepted["complete_energy_force"]
    assert accepted["scientific_identity"] == "workload"
    assert accepted["interleaved"] and accepted["synchronized"]
    assert accepted["baseline_schedule_hash"] == "baseline-schedule"
    assert accepted["candidate_schedule_hash"] == "candidate-schedule"
    assert accepted["candidate_source_hash"] == "candidate-source"

    slower = [
        endpoint_sample(
            1.1,
            i,
            schedule_identity="candidate-schedule",
            source_hash="candidate-source",
        )
        for i in range(6)
    ]
    assert not dft_endpoint_gate(baseline, slower)["passed"]

    changed = copy.deepcopy(candidate)
    changed[0]["scientific_identity"] = "different-grid"
    assert not dft_endpoint_gate(baseline, changed)["passed"]

    unsynchronized = copy.deepcopy(candidate)
    unsynchronized[0]["synchronized"] = False
    assert not dft_endpoint_gate(baseline, unsynchronized)["passed"]

    not_interleaved = copy.deepcopy(candidate)
    not_interleaved[0]["pair_id"] = 99
    assert not dft_endpoint_gate(baseline, not_interleaved)["passed"]

    incomplete = copy.deepcopy(candidate)
    incomplete[0]["endpoint_kind"] = "energy"
    assert not dft_endpoint_gate(baseline, incomplete)["passed"]

    same_schedule = copy.deepcopy(candidate)
    for sample in same_schedule:
        sample["schedule_identity"] = "baseline-schedule"
    assert not dft_endpoint_gate(baseline, same_schedule)["passed"]

    mismatched_source = copy.deepcopy(candidate)
    mismatched_source[0]["source_hash"] = "other-source"
    assert not dft_endpoint_gate(baseline, mismatched_source)["passed"]

    with pytest.raises(ValueError, match="at least five"):
        dft_endpoint_gate(baseline[:4], candidate[:4])


@pytest.mark.parametrize("side", ["baseline", "candidate"])
@pytest.mark.parametrize(
    "energies,forces",
    [
        ([-1.0, -1.0], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
        ([-1.0], [[[0.0, 0.0, 0.0]]]),
        ([-1.0], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        ([-1.0], [[[0.0], [0.0]]]),
        (-1.0, [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]),
    ],
    ids=[
        "system-count",
        "atom-count",
        "missing-system-axis",
        "cartesian-axis",
        "scalar-energy",
    ],
)
def test_dft_endpoint_gate_rejects_broadcastable_layouts(
    side: str, energies: object, forces: object
) -> None:
    baseline = [endpoint_sample(1.0, i) for i in range(5)]
    candidate = [
        endpoint_sample(
            0.8,
            i,
            schedule_identity="candidate-schedule",
            source_hash="candidate-source",
        )
        for i in range(5)
    ]
    samples = baseline if side == "baseline" else candidate
    samples[0].update(energies=energies, forces=forces)
    with pytest.raises(ValueError, match="DFT endpoint.*layout"):
        dft_endpoint_gate(baseline, candidate)
