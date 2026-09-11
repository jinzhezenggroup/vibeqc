"""Metric least-squares transport of occupied HF subspaces between AO bases.

Cross overlaps are rectangular ``S_target,source`` matrices. Projection is an
initial-state proposal: the target solver must rebuild and converge its own
Hamiltonian. No virtual orbitals or correlated amplitudes are transported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ProjectionRejected(ValueError):
    """The source state or target space cannot support a valid occupied proposal."""


@dataclass(frozen=True)
class ProjectionPolicy:
    """Reproducible spectral cutoffs and a worst-direction norm-loss gate.

    Metric eigenvalues at or below ``relative_threshold * largest`` are removed.
    The residual is the largest principal sine between source occupied space
    and its target-metric least-squares image, before reorthonormalization.
    """

    relative_threshold: float = 1e-10
    validation_tolerance: float = 1e-7
    maximum_residual: float = 0.5
    maximum_ao: int = 4096

    def __post_init__(self):
        for name in ("relative_threshold", "validation_tolerance", "maximum_residual"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{name} must be finite and in (0,1]")
        if type(self.maximum_ao) is not int or self.maximum_ao < 1:
            raise ValueError("maximum_ao must be a positive integer")


@dataclass(frozen=True)
class ProjectionDiagnostics:
    """Numerical validity of the seed, separate from target SCF convergence."""

    source_metric_rank: int
    target_metric_rank: int
    occupied_rank: int
    minimum_projected_norm_squared: float
    projection_residual: float
    normal_equation_residual: float
    metric_orthogonality_error: float
    electron_trace: float
    metric_idempotency_error: float


@dataclass(frozen=True)
class OccupiedProjection:
    """Detached immutable occupied coefficients and their target AO density."""

    coefficients: np.ndarray
    density: np.ndarray
    diagnostics: ProjectionDiagnostics


def _immutable(array):
    array = np.ascontiguousarray(array, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _array(value, shape, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ProjectionRejected(f"{name} requires real orbitals")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ProjectionRejected(f"{name} has invalid dimensions or nonfinite values")
    return array


def _metric(value, name, policy):
    raw = np.asarray(value)
    if (
        raw.ndim != 2
        or raw.shape[0] != raw.shape[1]
        or not 0 < len(raw) <= policy.maximum_ao
    ):
        raise ProjectionRejected(f"{name} must be a bounded nonempty square AO metric")
    matrix = _array(value, raw.shape, name)
    scale = max(float(np.linalg.norm(matrix, ord=2)), np.finfo(float).tiny)
    if np.max(np.abs(matrix - matrix.T)) > policy.validation_tolerance * scale:
        raise ProjectionRejected(f"{name} is not Hermitian")
    matrix = (matrix + matrix.T) / 2
    values, vectors = np.linalg.eigh(matrix)
    if values[-1] <= 0 or values[0] < -policy.validation_tolerance * values[-1]:
        raise ProjectionRejected(f"{name} is not positive semidefinite")
    keep = values > policy.relative_threshold * values[-1]
    return matrix, values[keep], vectors[:, keep]


def project_occupied(
    source_overlap,
    target_overlap,
    cross_overlap,
    source_coefficients,
    *,
    occupation=2,
    policy=None,
):
    """Project a real occupied subspace, then orthonormalize in the target metric.

    Coefficients have shape ``(source_AOs, occupied_orbitals)`` and must satisfy
    ``C.T @ S_source @ C = I``. ``occupation=2`` produces an RHF density; use
    separate calls with ``occupation=1`` for UHF alpha/beta subspaces. Input
    rotations/phases change coefficient gauge but preserve the output density.
    Rank loss rejects the entire proposal; missing orbitals are never invented.
    """
    policy = ProjectionPolicy() if policy is None else policy
    if type(occupation) is not int or occupation not in (1, 2):
        raise ValueError("occupation must be 1 (one spin) or 2 (restricted)")
    source, source_values, source_vectors = _metric(
        source_overlap, "source overlap", policy
    )
    target, target_values, target_vectors = _metric(
        target_overlap, "target overlap", policy
    )
    cross = _array(cross_overlap, (len(target), len(source)), "cross overlap")
    raw_coefficients = np.asarray(source_coefficients)
    if raw_coefficients.ndim != 2 or raw_coefficients.shape[0] != len(source):
        raise ProjectionRejected("source occupied coefficients have invalid dimensions")
    occupied = raw_coefficients.shape[1]
    if occupied > min(len(source_values), len(target_values)):
        raise ProjectionRejected(
            "source or target metric has insufficient occupied rank"
        )
    coefficients = _array(
        raw_coefficients, (len(source), occupied), "source coefficients"
    )
    identity = np.eye(occupied)
    if not np.allclose(
        coefficients.T @ source @ coefficients,
        identity,
        atol=policy.validation_tolerance,
        rtol=0,
    ):
        raise ProjectionRejected("source occupied orbitals are not metric orthonormal")
    # Discarded source metric directions cannot carry an arbitrary null-space
    # coefficient that contaminates the rectangular overlap multiplication.
    retained = source_vectors @ (source_vectors.T @ coefficients)
    if np.linalg.norm(coefficients - retained) > policy.validation_tolerance * max(
        1.0, np.linalg.norm(coefficients)
    ):
        raise ProjectionRejected(
            "source occupied orbitals contain discarded metric directions"
        )
    right = cross @ coefficients
    projected = target_vectors @ ((target_vectors.T @ right) / target_values[:, None])
    gram = projected.T @ target @ projected
    norms, rotation = np.linalg.eigh((gram + gram.T) / 2)
    minimum = float(norms[0]) if occupied else 1.0
    if occupied and minimum <= policy.relative_threshold:
        raise ProjectionRejected("occupied projection lost rank in the target basis")
    if occupied and norms[-1] > 1 + policy.validation_tolerance:
        raise ProjectionRejected(
            "cross overlap is inconsistent with normalized source and target metrics"
        )
    residual = float(np.sqrt(max(0.0, 1.0 - minimum)))
    if residual > policy.maximum_residual:
        raise ProjectionRejected(
            f"projection residual {residual:.6g} exceeds {policy.maximum_residual:.6g}"
        )
    orthogonal = projected @ ((rotation / np.sqrt(norms)) @ rotation.T)
    density = occupation * orthogonal @ orthogonal.T
    metric_error = float(np.linalg.norm(orthogonal.T @ target @ orthogonal - identity))
    idempotency = float(
        np.linalg.norm(density @ target @ density - occupation * density)
    )
    trace = float(np.trace(density @ target))
    if (
        metric_error > policy.validation_tolerance * max(1, occupied)
        or abs(trace - occupation * occupied)
        > policy.validation_tolerance * max(1, occupation * occupied)
        or idempotency
        > policy.validation_tolerance * occupation * max(1.0, np.linalg.norm(density))
    ):
        raise ProjectionRejected("target seed failed metric/electron validation")
    diagnostics = ProjectionDiagnostics(
        len(source_values),
        len(target_values),
        occupied,
        minimum,
        residual,
        float(
            np.linalg.norm(target @ projected - right) / max(1.0, np.linalg.norm(right))
        ),
        metric_error,
        trace,
        idempotency,
    )
    return OccupiedProjection(_immutable(orthogonal), _immutable(density), diagnostics)


def project_density(
    source_overlap,
    target_overlap,
    cross_overlap,
    source_density,
    *,
    occupied_orbitals,
    occupation=2,
    policy=None,
):
    """Reconstruct a pure HF occupied space from a validated AO density, then project.

    Diagonalize ``S_source**(1/2) D S_source**(1/2)`` in the retained metric
    space. Its eigenvalues must be the declared 0/1 or 0/2 occupations. Mixed,
    fractional, wrong-electron and non-idempotent source states are rejected.
    No canonical/virtual orbital completion or source Fock is needed.
    """
    policy = ProjectionPolicy() if policy is None else policy
    if type(occupied_orbitals) is not int or occupied_orbitals < 0:
        raise ValueError("occupied_orbitals must be a nonnegative integer")
    if type(occupation) is not int or occupation not in (1, 2):
        raise ValueError("occupation must be 1 or 2")
    metric, values, vectors = _metric(source_overlap, "source overlap", policy)
    density = _array(source_density, metric.shape, "source density")
    if not np.allclose(density, density.T, atol=policy.validation_tolerance, rtol=0):
        raise ProjectionRejected("source density is not Hermitian")
    if occupied_orbitals > len(values):
        raise ProjectionRejected("source metric has insufficient occupied rank")
    rooted = vectors * np.sqrt(values)
    orthogonal_density = rooted.T @ density @ rooted
    populations, orbitals = np.linalg.eigh(
        (orthogonal_density + orthogonal_density.T) / 2
    )
    expected = np.zeros(len(values))
    if occupied_orbitals:
        expected[-occupied_orbitals:] = occupation
    if not np.allclose(populations, expected, atol=policy.validation_tolerance, rtol=0):
        raise ProjectionRejected(
            "source density has inconsistent electron count or fractional occupations"
        )
    selected = orbitals[:, len(values) - occupied_orbitals :]
    coefficients = (vectors / np.sqrt(values)) @ selected
    reconstructed = occupation * coefficients @ coefficients.T
    if not np.allclose(
        reconstructed,
        density,
        atol=policy.validation_tolerance,
        rtol=policy.validation_tolerance,
    ):
        raise ProjectionRejected(
            "source density contains unsupported discarded metric components"
        )
    return project_occupied(
        metric,
        target_overlap,
        cross_overlap,
        coefficients,
        occupation=occupation,
        policy=policy,
    )
