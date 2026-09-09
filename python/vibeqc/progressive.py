"""Explicit source-solve, basis-projection and target-SCF initialization workflow."""

from __future__ import annotations

import ctypes
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np

from . import _native
from .calculator import Atom
from .checkpoint import _check_batch, _controls, _descriptor, _pointer, _resource_check
from .overlap import cross_overlap
from .projection import (
    ProjectionPolicy,
    ProjectionRejected,
    _immutable,
    project_density,
)


def _dimension(calculator, atoms):
    return sum(
        2 * s.angular_momentum + 1
        if calculator._basis_representation == _native.BASIS_SPHERICAL
        else (s.angular_momentum + 1) * (s.angular_momentum + 2) // 2
        for s in calculator._shells_for_atoms(atoms)
    )


def initialize_from(
    target, source, *, policy=None, strict=True, maximum_host_bytes=256 << 20
):
    """Install per-item metric-projected densities into a fresh prepared target.

    Source and target must have the same ordered nuclei, current geometry,
    electron/core/spin state and RHF/UHF method. Basis/DF approximations may
    differ; each target still rebuilds its own Hamiltonian. With ``strict=False``
    an invalid/failed source or lost occupied rank leaves that item cold. All
    accepted items are validated before the native transactional seed install.
    Existing target seeds must be cleared explicitly before initialization.
    """
    start = time.perf_counter()
    policy = ProjectionPolicy() if policy is None else policy
    if (
        type(strict) is not bool
        or type(maximum_host_bytes) is not int
        or maximum_host_bytes < 1
    ):
        raise ValueError("explicit strict flag and positive host budget required")
    _check_batch(source)
    _check_batch(target)
    if source is target or source.system_count != target.system_count:
        raise ProjectionRejected(
            "distinct source/target fleets with matching item count required"
        )
    if not target._warm_enabled:
        raise ProjectionRejected("target initialization requires warm starts enabled")
    if source._calculator._method != target._calculator._method:
        raise ProjectionRejected("source/target HF spin methods must match")
    if source._calculator._method not in (_native.METHOD_RHF, _native.METHOD_UHF):
        raise ProjectionRejected(
            "only RHF/UHF occupied density projection is supported"
        )
    count = target.system_count
    states = (_native.HfWarmState * count)(*(_descriptor() for _ in range(count)))
    storage, reports = [], []
    metadata = list(target._warm_metadata)
    for index in range(count):
        current = _descriptor()
        _native.check(
            target._library,
            target._library.vibeqc_batch_get_hf_warm_state(
                target._batch, index, ctypes.byref(current)
            ),
        )
        if current.present:
            raise ProjectionRejected(
                "target already has a seed; clear warm starts before projection"
            )
    retained_bytes = 0
    for index, atoms in enumerate(target._systems):
        report = {
            "index": index,
            "accepted": False,
            "reason": None,
            "source_model_identity": source._basis_metadata[index]["model_identity"],
            "target_model_identity": target._basis_metadata[index]["model_identity"],
            "spins": [],
        }
        reports.append(report)
        try:
            left = target._basis_metadata[index]["orbital"]["electrons"]
            right = source._basis_metadata[index]["orbital"]["electrons"]
            if left != right or any(left["ecp_core_electrons"]):
                raise ProjectionRejected(
                    "ordered nuclei, electrons, core treatment and spin must match"
                )
            if (
                source._last_statuses is None
                or source._last_statuses[index] != _native.STATUS_SUCCESS
                or index in source._projection_indices
            ):
                raise ProjectionRejected(
                    "source item has no successful current SCF result"
                )
            state = _descriptor()
            _native.check(
                source._library,
                source._library.vibeqc_batch_get_hf_warm_state(
                    source._batch, index, ctypes.byref(state)
                ),
            )
            if not state.present:
                raise ProjectionRejected("source item has no retained density")
            spins = 2 if source._calculator._method == _native.METHOD_UHF else 1
            ns, nt = (
                math.isqrt(state.density_count // spins),
                _dimension(target._calculator, atoms),
            )
            if ns > policy.maximum_ao or nt > policy.maximum_ao:
                raise ProjectionRejected(
                    "source/target AO count exceeds projection policy"
                )
            # Bound live matrices, eigensolver operands and staged native seed
            # copies; caller-owned source/target HF arenas are separate budgets.
            workspace = (
                256 * (ns * ns + nt * nt + ns * nt) + 4 * state.coordinate_count * 8
            )
            if retained_bytes + workspace > maximum_host_bytes:
                raise MemoryError("basis projection exceeds maximum_host_bytes")
            _resource_check(target, retained_bytes + workspace, 8 * spins * nt * nt)
            density = np.empty((spins, ns, ns), dtype=np.float64)
            coordinates = np.empty(state.coordinate_count, dtype=np.float64)
            state.density, state.coordinates = _pointer(density), _pointer(coordinates)
            _native.check(
                source._library,
                source._library.vibeqc_batch_get_hf_warm_state(
                    source._batch, index, ctypes.byref(state)
                ),
            )
            if not np.array_equal(
                coordinates.reshape(-1, 3), np.array([a.position for a in atoms])
            ):
                raise ProjectionRejected(
                    "source density geometry differs from the target's ordered atoms"
                )
            options = {
                "charge": target.charges[index],
                "multiplicity": target.multiplicities[index],
                "maximum_bytes": maximum_host_bytes,
            }
            ss = cross_overlap(source._calculator, source._calculator, atoms, **options)
            tt = cross_overlap(target._calculator, target._calculator, atoms, **options)
            ts = cross_overlap(target._calculator, source._calculator, atoms, **options)
            occupations = (
                [left["nalpha"], left["nbeta"]] if spins == 2 else [left["nalpha"]]
            )
            projected = [
                project_density(
                    ss,
                    tt,
                    ts,
                    block,
                    occupied_orbitals=occupied,
                    occupation=1 if spins == 2 else 2,
                    policy=policy,
                )
                for block, occupied in zip(density, occupations, strict=True)
            ]
            proposal = np.ascontiguousarray([p.density for p in projected])
            report["spins"] = [asdict(p.diagnostics) for p in projected]
            report["numeric_workspace_bound_bytes"] = workspace
            report["source_energy"] = state.energy
            report["source_iterations"] = state.iterations
            candidate = states[index]
            candidate.density, candidate.coordinates = (
                _pointer(proposal),
                _pointer(coordinates),
            )
            candidate.density_count, candidate.coordinate_count = (
                proposal.size,
                coordinates.size,
            )
            for name in ("energy", "energy_change", "density_rms", "iterations"):
                setattr(candidate, name, getattr(state, name))
            candidate.present = 1
            storage.extend((proposal, coordinates))
            retained_bytes += proposal.nbytes + coordinates.nbytes
            metadata[index] = {
                "controls": _controls(target._calculator),
                "backend": "cpu",
            }
            report["accepted"] = True
        except ProjectionRejected as error:
            report["reason"] = str(error)
            if strict:
                raise ProjectionRejected(f"item {index}: {error}") from error
    accepted = {r["index"] for r in reports if r["accepted"]}
    if accepted:
        status = target._library.vibeqc_batch_restore_hf_warm_states(
            target._batch, states, count
        )
        if status != _native.STATUS_SUCCESS:
            detail = target._library.vibeqc_context_get_last_detail(
                target._context
            ).decode()
            raise ProjectionRejected(
                f"native target seed validation failed: {detail or status}"
            )
    target._warm_metadata = metadata
    target._projection_indices = accepted
    report = {
        "schema": "vibeqc.basis_projection",
        "version": 1,
        "policy": asdict(policy),
        "items": reports,
        "seconds": time.perf_counter() - start,
        "placement": "native CPU cross-overlap and NumPy metric projection; native target seed import",
        "target_verification": "pending_execute",
    }
    target.projection_diagnostics = report
    return deepcopy(report)


@dataclass(frozen=True)
class ProgressiveResult:
    """The actual target result and complete source/projection/target cost record."""

    target: object
    source: object
    diagnostics: dict
    target_density: np.ndarray


def _retained_density(batch, index=0):
    """Detach the actual converged target density for root comparisons."""
    state = _descriptor()
    _native.check(
        batch._library,
        batch._library.vibeqc_batch_get_hf_warm_state(
            batch._batch, index, ctypes.byref(state)
        ),
    )
    if not state.present:
        raise ProjectionRejected("target has no retained converged density")
    spins = 2 if batch._calculator._method == _native.METHOD_UHF else 1
    n = math.isqrt(state.density_count // spins)
    density = np.empty((spins, n, n), dtype=np.float64)
    coordinates = np.empty(state.coordinate_count, dtype=np.float64)
    state.density, state.coordinates = _pointer(density), _pointer(coordinates)
    _native.check(
        batch._library,
        batch._library.vibeqc_batch_get_hf_warm_state(
            batch._batch, index, ctypes.byref(state)
        ),
    )
    return _immutable(density)


def projected_singlepoint(
    target,
    source,
    atoms,
    *,
    charge=0,
    multiplicity=1,
    policy=None,
    maximum_host_bytes=256 << 20,
):
    """Solve a source basis, project its seed if valid, and converge the target HF.

    Rejected/failed source items use the target's ordinary cold guess. Native
    warm-start safeguards also retry cold after target stagnation. Total timing
    includes both plan setups, the source solve, projection and target solve;
    a faster target stage alone is not an overall speedup.
    """
    atoms = tuple(Atom.from_value(a) for a in atoms)
    started = time.perf_counter()
    with source.prepare_batch(
        [atoms], charges=[charge], multiplicities=[multiplicity]
    ) as source_batch:
        source_result = source_batch.execute(strict=False).items[0]
        source_seconds = time.perf_counter() - started
        setup = time.perf_counter()
        with target.prepare_batch(
            [atoms], charges=[charge], multiplicities=[multiplicity]
        ) as target_batch:
            target_setup_seconds = time.perf_counter() - setup
            projection = initialize_from(
                target_batch,
                source_batch,
                policy=policy,
                strict=False,
                maximum_host_bytes=maximum_host_bytes,
            )
            execution = time.perf_counter()
            result = target_batch.execute(strict=True).items[0]
            density = _retained_density(target_batch)
            target_seconds = time.perf_counter() - execution
            diagnostics = deepcopy(target_batch.projection_diagnostics)
    diagnostics.update(
        source_seconds=source_seconds,
        projection_seconds=projection["seconds"],
        target_setup_seconds=target_setup_seconds,
        target_seconds=target_seconds,
        total_seconds=time.perf_counter() - started,
    )
    return ProgressiveResult(result, source_result, diagnostics, density)
