"""Pipek-Mezey occupied localization with explicit Mulliken atomic populations.

The Jacobi objective is sum(A,i) q[A,i]**2 for spatial occupied orbitals.
Symmetrized Mulliken charge operators sum to the identity in the occupied
metric. They are population operators, not positive atomic probabilities.
"""

import time
from dataclasses import dataclass

import numpy as np

from tools.vibeqc_posthf.reference import immutable

from .common import (
    checked_budget,
    checked_reference,
    fingerprint,
    number,
    orthogonality,
)


@dataclass(frozen=True, eq=False)
class OccupiedLocalization:
    """Owned occupied rotation with the noncanonical occupied Fock retained.

    ``rotation[k,i]`` transforms canonical occupied k to localized i. MP2
    consumers must transform audited canonical amplitudes or retain the full
    occupied Fock coupling; diagonal local denominators are not equivalent.
    """

    reference_id: str
    rotation: np.ndarray
    coefficients: np.ndarray
    occupied_fock: np.ndarray
    populations: np.ndarray
    initial_objective: float
    objective: float
    gradient_max: float
    sweeps: int
    seconds: float
    numeric_peak_bytes: int
    ao_atoms: tuple[int, ...]
    method: str = "pipek_mezey_mulliken"

    def __post_init__(self):
        if not self.reference_id or self.method != "pipek_mezey_mulliken":
            raise ValueError("invalid occupied-localization identity")
        u = immutable(self.rotation)
        if u.ndim != 2 or u.shape[0] != u.shape[1] or not len(u):
            raise ValueError("occupied localization requires a square rotation")
        if orthogonality(u) > 1e-10:
            raise ValueError("occupied rotation is not orthogonal")
        c, f, q = map(
            immutable, (self.coefficients, self.occupied_fock, self.populations)
        )
        if c.ndim != 2 or c.shape[1] != len(u) or f.shape != u.shape:
            raise ValueError("inconsistent occupied-localization dimensions")
        if q.ndim != 2 or q.shape[1] != len(u) or not len(q):
            raise ValueError("invalid atomic populations")
        atoms = tuple(self.ao_atoms)
        if len(atoms) != len(c) or any(
            type(a) is not int or not 0 <= a < len(q) for a in atoms
        ):
            raise ValueError("invalid atomic population partition")
        object.__setattr__(self, "ao_atoms", atoms)
        if np.max(np.abs(f - f.T)) > 1e-10 or np.max(np.abs(q.sum(axis=0) - 1)) > 1e-8:
            raise ValueError("invalid occupied Fock/population invariants")
        for name, value in (
            ("rotation", u),
            ("coefficients", c),
            ("occupied_fock", f),
            ("populations", q),
        ):
            object.__setattr__(self, name, value)
        for name in ("initial_objective", "objective", "gradient_max", "seconds"):
            number(getattr(self, name), name)

    @property
    def identity(self):
        """Tie local pair labels to this specific occupied gauge and reference."""
        return fingerprint(
            {
                "reference": self.reference_id,
                "method": self.method,
                "ao_atoms": self.ao_atoms,
            },
            rotation=self.rotation,
        )


def population_operators(snapshot, ao_atoms):
    """Project symmetrized AO Mulliken partitions into the occupied metric."""
    no, _ = checked_reference(snapshot)
    atoms = tuple(ao_atoms)
    if len(atoms) != snapshot.nmo or any(
        type(a) is not int or not 0 <= a < snapshot.nmo for a in atoms
    ):
        raise ValueError("ao_atoms must label every AO with a nonnegative atom index")
    if set(atoms) != set(range(max(atoms) + 1)):
        raise ValueError("atomic population labels must be contiguous")
    c = snapshot.coefficients[:, :no]
    sc = snapshot.overlap @ c
    operators = np.empty((max(atoms) + 1, no, no))
    for atom in range(len(operators)):
        rows = np.asarray(atoms) == atom
        charge = c[rows].T @ sc[rows]
        operators[atom] = (charge + charge.T) / 2
    if np.max(np.abs(operators.sum(axis=0) - np.eye(no))) > 1e-8:
        raise ValueError("Mulliken operators do not resolve the occupied identity")
    return operators


def _gradient(operators):
    # d/dtheta sum q_ii^2 at theta=0 for each occupied Jacobi rotation.
    diagonal = np.diagonal(operators, axis1=1, axis2=2)
    gradient = 4 * np.einsum(
        "Aij,Aij->ij", operators, diagonal[:, :, None] - diagonal[:, None, :]
    )
    return float(np.max(np.abs(gradient)))


def localize_occupied(
    snapshot,
    ao_atoms,
    *,
    tolerance=1e-10,
    max_sweeps=200,
    initial_rotation=None,
    budget_bytes=128 << 20,
):
    """Maximize the PM objective by analytic two-orbital Jacobi rotations.

    Failure to reach the gradient gate raises explicitly. The initial rotation
    is optional and must be orthogonal; it supports reproducible alternative
    starts without altering the reference. A converged local maximum is not
    claimed to be the unique/global optimum.
    """
    started = time.perf_counter()
    no, _ = checked_reference(snapshot)
    tolerance = number(tolerance, "localization tolerance", positive=True)
    if type(max_sweeps) is not int or max_sweeps < 1:
        raise ValueError("max_sweeps must be a positive integer")
    ao_atoms = tuple(ao_atoms)
    # Charge operators, Jacobi copies, occupied matrices and immutable result
    # publication coexist. Interpreter and BLAS implementation overhead is excluded.
    peak = 8 * (6 * snapshot.nmo * no + 8 * no**2 * (snapshot.nmo + 1))
    if peak > checked_budget(budget_bytes):
        raise MemoryError(f"occupied localization needs {peak} numeric bytes")
    u = np.eye(no) if initial_rotation is None else immutable(initial_rotation).copy()
    if u.shape != (no, no) or not np.isfinite(u).all() or orthogonality(u) > 1e-10:
        raise ValueError("invalid initial occupied rotation")
    operators = population_operators(snapshot, ao_atoms)
    operators = u.T @ operators @ u
    initial = float(np.sum(np.diagonal(operators, axis1=1, axis2=2) ** 2))
    sweeps = 0
    for _ in range(max_sweeps):
        if _gradient(operators) <= tolerance:
            break
        for p in range(no):
            for q in range(p):
                d = (operators[:, p, p] - operators[:, q, q]) / 2
                e = operators[:, p, q]
                a, b, c = d @ d, d @ e, e @ e
                if abs(b) <= tolerance / (8 * max(1, no)) and a >= c:
                    continue
                angle = 0.25 * np.arctan2(2 * b, a - c)
                cosine, sine = np.cos(angle), np.sin(angle)
                rotation = np.array([[cosine, -sine], [sine, cosine]])
                u[:, [p, q]] = u[:, [p, q]] @ rotation
                operators[:, :, [p, q]] = operators[:, :, [p, q]] @ rotation
                operators[:, [p, q], :] = rotation.T @ operators[:, [p, q], :]
        sweeps += 1
    gradient = _gradient(operators)
    if gradient > tolerance:
        raise RuntimeError(
            f"Pipek-Mezey localization failed after {max_sweeps} sweeps: gradient={gradient}"
        )
    coefficients = snapshot.coefficients[:, :no] @ u
    # Deterministic phase and ordering affect pair labels, never the occupied projector.
    phase = np.sign(
        coefficients[np.argmax(np.abs(coefficients), axis=0), np.arange(no)]
    )
    u *= phase
    coefficients *= phase
    populations = np.diagonal(operators, axis1=1, axis2=2).copy()
    order = sorted(
        range(no),
        key=lambda i: (
            int(np.argmax(populations[:, i])),
            tuple(-np.round(populations[:, i], 12)),
            tuple(-np.round(np.abs(coefficients[:, i]), 12)),
        ),
    )
    u, coefficients, populations = (
        u[:, order],
        coefficients[:, order],
        populations[:, order],
    )
    objective = float(np.sum(populations**2))
    if (
        objective < initial - 1e-10
        or orthogonality(coefficients, snapshot.overlap) > 1e-8
    ):
        raise RuntimeError("localization lost objective/metric invariants")
    occupied_fock = u.T @ np.diag(snapshot.orbital_energies[:no]) @ u
    return OccupiedLocalization(
        snapshot.identity,
        u,
        coefficients,
        occupied_fock,
        populations,
        initial,
        objective,
        gradient,
        sweeps,
        time.perf_counter() - started,
        peak,
        ao_atoms,
    )
