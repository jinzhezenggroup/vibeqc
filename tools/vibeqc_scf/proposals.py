"""Deterministic traditional proposals and explicit metric transformations.

All proposal generators are independent of target Fock construction and the
native safeguard. Coordinate transformations are timed; invalid electron
counts are never repaired. No second-order convergence claim is made for the
simple diagonal occupied-virtual preconditioner below.
"""

import time
from dataclasses import dataclass

import numpy as np

from .state import DensityProposal, metric_root, spin_counts


def _columns(value):
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.size > 12 * 12:
        raise ValueError("orbital action exceeds the small-system output budget")
    if np.iscomplexobj(raw):
        raise ValueError("complex orbitals are unsupported")
    raw = np.asarray(raw, dtype=np.float64, order="C")
    return np.frombuffer(raw.tobytes(), dtype=np.float64).reshape(raw.shape)


@dataclass(frozen=True, eq=False)
class OccupiedProposal:
    """Occupied AO columns with prescribed spin populations and S-orthonormality."""

    parent_id: str
    columns: tuple

    def __post_init__(self):
        object.__setattr__(self, "columns", tuple(_columns(c) for c in self.columns))

    def materialize(self, state):
        started = time.perf_counter()
        if self.parent_id != state.identity:
            raise ValueError("stale_state")
        counts, weight = spin_counts(state.model)
        if len(self.columns) != len(counts):
            raise ValueError("spin populations")
        density = []
        for columns, count in zip(self.columns, counts, strict=True):
            no = int(count / weight)
            if (
                columns.shape != (len(state.overlap), no)
                or not np.isfinite(columns).all()
            ):
                raise ValueError("occupied column shape/finiteness")
            if not np.allclose(
                columns.T @ state.overlap @ columns, np.eye(no), atol=1e-8, rtol=0
            ):
                raise ValueError("metric_orthonormality")
            density.append(weight * columns @ columns.T)
        return DensityProposal(
            self.parent_id,
            density,
            "determinant_density",
            time.perf_counter() - started,
            "occupied_columns_to_density",
        )


@dataclass(frozen=True, eq=False)
class RotationProposal:
    """Occupied-virtual action in an explicit full S-orthonormal orbital gauge.

    ``rotations[x][i,a]`` parameterizes a skew generator; the Cayley transform
    preserves orthonormality at any finite step size. A preconditioner declares
    the same output coordinates using ``kind='preconditioner_action'``. The
    resulting density still passes the target-operator safeguard.
    """

    parent_id: str
    coefficients: tuple
    rotations: tuple
    kind: str = "occupied_virtual_rotation"

    def __post_init__(self):
        for name in ("coefficients", "rotations"):
            object.__setattr__(
                self, name, tuple(_columns(c) for c in getattr(self, name))
            )
        if self.kind not in ("occupied_virtual_rotation", "preconditioner_action"):
            raise ValueError("unknown rotation action")

    def materialize(self, state):
        started = time.perf_counter()
        if self.parent_id != state.identity:
            raise ValueError("stale_state")
        n = len(state.overlap)
        counts, weight = spin_counts(state.model)
        if len(self.coefficients) != len(counts) or len(self.rotations) != len(counts):
            raise ValueError("spin populations")
        columns = []
        for c, rotation, count in zip(
            self.coefficients, self.rotations, counts, strict=True
        ):
            no = int(count / weight)
            if c.shape != (n, n) or rotation.shape != (no, n - no):
                raise ValueError("rotation shape")
            if not np.isfinite(c).all() or not np.isfinite(rotation).all():
                raise ValueError("nonfinite rotation")
            if not np.allclose(c.T @ state.overlap @ c, np.eye(n), atol=1e-8, rtol=0):
                raise ValueError("metric_orthonormality")
            generator = np.zeros((n, n))
            generator[:no, no:] = -rotation
            generator[no:, :no] = rotation.T
            unitary = np.linalg.solve(
                np.eye(n) - generator / 2, np.eye(n) + generator / 2
            )
            columns.append((c @ unitary)[:, :no])
        proposal = OccupiedProposal(self.parent_id, tuple(columns)).materialize(state)
        return DensityProposal(
            state.identity,
            proposal.density,
            "determinant_density",
            time.perf_counter() - started,
            self.kind + "_cayley",
        )


def diis_density(state):
    """Propose the existing native DIIS/Aufbau update with the extra safeguard."""
    return DensityProposal(state.identity, state.baseline, "determinant_density")


def mixing(fraction):
    """Mix current and traditional densities inside the physical ensemble domain."""
    if isinstance(fraction, bool) or not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("mixing fraction must lie in (0,1]")

    def propose(state):
        return DensityProposal(
            state.identity, (1 - fraction) * state.density + fraction * state.baseline
        )

    return propose


def project_occupied(columns, overlap):
    """Explicit Löwdin projection into a new overlap; rank loss is an error.

    Caller must verify compatible atom/AO topology, record the source/target
    identities, and count this setup time in any complete-solve comparison.
    This is same-AO coefficient transport, not a cross-basis overlap projector.
    """
    started = time.perf_counter()
    columns = _columns(columns)
    if (
        columns.ndim != 2
        or columns.shape[0] != len(overlap)
        or not np.isfinite(columns).all()
    ):
        raise ValueError("invalid occupied columns")
    eigenvalues, vectors = np.linalg.eigh(columns.T @ overlap @ columns)
    if len(eigenvalues) and eigenvalues[0] < 1e-10:
        raise ValueError("occupied_projection_rank_loss")
    projected = columns @ ((vectors / np.sqrt(eigenvalues)) @ vectors.T)
    return _columns(projected), time.perf_counter() - started


def diagonal_preconditioner(state):
    """Deterministic OV gradient step in current natural-orbital coordinates.

    Near-zero gaps fail explicitly; no denominator clipping or nominal Newton
    solver is hidden here. Ensembles are first projected to their highest
    occupied natural orbitals, a recorded transformation subsequently safeguarded.
    """
    root = metric_root(state.overlap)
    x = np.linalg.inv(root)
    counts, weight = spin_counts(state.model)
    coefficients, rotations = [], []
    for density, fock, count in zip(state.density, state.fock, counts, strict=True):
        _, natural = np.linalg.eigh(root @ density @ root)
        c = x @ natural[:, ::-1]
        no = int(count / weight)
        f = c.T @ fock @ c
        denominator = np.diag(f)[:no, None] - np.diag(f)[None, no:]
        if denominator.size and np.min(abs(denominator)) < 1e-8:
            raise ValueError("small_preconditioner_denominator")
        coefficients.append(c)
        rotations.append(f[:no, no:] / denominator)
    return RotationProposal(
        state.identity, tuple(coefficients), tuple(rotations), "preconditioner_action"
    )
