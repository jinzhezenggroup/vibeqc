# ruff: noqa: PLC0414
"""Owned, immutable canonical-RHF snapshots for internal correlated methods.

Arrays are backed by immutable bytes, not only NumPy's reversible write flag.
Snapshot identities cover values as well as topology: a geometry replay that
produces different C cannot hit an old integral cache, even with a reused ID.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
from vibeqc.profiles import canonical_hash
from vibeqc_compiler.common.arrays import immutable as immutable


@dataclass(frozen=True, eq=False)
class ReferenceSnapshot:
    """Validated real, all-electron, canonical closed-shell HF/KS state.

    C[mu,p] uses occupied columns first, then virtual columns; occupations are
    spatial 2/0. Buffers always belong to this host snapshot. ``hf_backend``
    records where HF ran; it does not change this ownership contract. Device
    providers explicitly copy/own their state for the lifetime of a block.
    Frozen masks are representable identity metadata but any nonempty mask is
    rejected until frozen-core equations receive independent validation.
    """

    overlap: np.ndarray
    hcore: np.ndarray
    fock: np.ndarray
    coefficients: np.ndarray
    orbital_energies: np.ndarray
    occupations: np.ndarray
    electron_count: int
    reference_energy: float
    scf_residual: float
    geometry_hash: str
    basis_hash: str
    generation_id: str
    hamiltonian_id: str = "conventional-unscreened"
    algorithm: str = "RHF"
    representation: str = "cartesian"
    precision: str = "float64"
    screening_tolerance: float = 0.0
    hf_backend: str = "cpu-reference"
    device_id: int | None = None
    converged: bool = True
    frozen_mask: tuple[int, ...] = ()
    validation_tolerance: float = 1e-8
    overlap_threshold: float = 1e-10
    functional_identity: str | None = None
    grid_identity: str | None = None
    identity: str = field(init=False)
    diagnostics: tuple[tuple[str, float], ...] = field(init=False)

    def __post_init__(self):
        if self.algorithm not in ("RHF", "KS") or self.precision != "float64":
            raise ValueError("only real FP64 RHF/KS references are supported")
        if self.algorithm == "KS":
            if not self.functional_identity or not self.grid_identity:
                raise ValueError(
                    "KS references require nonempty functional and grid identities"
                )
        elif self.functional_identity or self.grid_identity:
            raise ValueError(
                "RHF references cannot carry DFT functional/grid identities"
            )
        if self.representation not in ("cartesian", "real_spherical"):
            raise ValueError("unknown AO representation")
        if not self.converged:
            raise ValueError("unconverged HF cannot enter a correlated calculation")
        if self.frozen_mask:
            raise ValueError("frozen-core calculations are not yet validated")
        object.__setattr__(self, "frozen_mask", tuple(self.frozen_mask))
        for key in (
            "geometry_hash",
            "basis_hash",
            "generation_id",
            "hamiltonian_id",
            "hf_backend",
        ):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a nonempty identity")
        for key in (
            "reference_energy",
            "scf_residual",
            "screening_tolerance",
            "validation_tolerance",
            "overlap_threshold",
        ):
            value = getattr(self, key)
            if not math.isfinite(value) or (key != "reference_energy" and value < 0):
                raise ValueError(f"invalid {key}")
        if (
            not 0 < self.validation_tolerance <= 1e-6
            or not 0 < self.overlap_threshold <= 1e-6
        ):
            raise ValueError("validation tolerances must be positive and at most 1e-6")
        c = immutable(self.coefficients)
        if c.ndim != 2 or c.shape[0] != c.shape[1]:
            raise ValueError(
                "rectangular/linear-dependence-reduced orbital spaces are unsupported"
            )
        n = c.shape[0]
        if (
            type(self.electron_count) is not int
            or self.electron_count <= 0
            or self.electron_count % 2
            or self.electron_count >= 2 * n
        ):
            raise ValueError(
                "RHF requires occupied and virtual orbitals with an even electron count"
            )
        object.__setattr__(self, "coefficients", c)
        for name in ("overlap", "hcore", "fock"):
            a = immutable(getattr(self, name), shape=(n, n))
            if np.max(np.abs(a - a.T)) > self.validation_tolerance:
                raise ValueError(f"{name} is not symmetric")
            object.__setattr__(self, name, a)
        for name in ("orbital_energies", "occupations"):
            object.__setattr__(self, name, immutable(getattr(self, name), shape=(n,)))
        occupied = self.electron_count // 2
        expected = np.zeros(n)
        expected[:occupied] = 2
        if not np.array_equal(self.occupations, expected):
            raise ValueError(
                "occupations must be ordered doubly occupied then virtual, with no frozen orbitals"
            )
        eps = self.orbital_energies
        if np.any(np.diff(eps) < -self.validation_tolerance):
            raise ValueError("canonical MO energies must be ascending")
        smallest = float(np.linalg.eigvalsh(self.overlap)[0])
        if smallest <= self.overlap_threshold:
            raise ValueError("linearly dependent AO overlap is unsupported")
        ortho = float(np.max(np.abs(c.T @ self.overlap @ c - np.eye(n))))
        canonical = float(np.max(np.abs(c.T @ self.fock @ c - np.diag(eps))))
        generalized = float(np.max(np.abs(self.fock @ c - (self.overlap @ c) * eps)))
        if (
            max(ortho, canonical, generalized, self.scf_residual)
            > self.validation_tolerance
        ):
            raise ValueError(
                f"invalid RHF reference: orthogonality={ortho}, canonicality={canonical}, eigen_residual={generalized}, scf_residual={self.scf_residual}"
            )
        diagnostics = (
            ("orthogonality", ortho),
            ("canonicality", canonical),
            ("eigen_residual", generalized),
            ("overlap_min_eigenvalue", smallest),
        )
        object.__setattr__(self, "diagnostics", diagnostics)
        metadata = {
            key: getattr(self, key)
            for key in (
                "electron_count",
                "reference_energy",
                "scf_residual",
                "geometry_hash",
                "basis_hash",
                "generation_id",
                "hamiltonian_id",
                "algorithm",
                "representation",
                "precision",
                "screening_tolerance",
                "hf_backend",
                "device_id",
                "frozen_mask",
                "validation_tolerance",
                "overlap_threshold",
                "functional_identity",
                "grid_identity",
            )
        }
        for name in (
            "overlap",
            "hcore",
            "fock",
            "coefficients",
            "orbital_energies",
            "occupations",
        ):
            metadata[name] = sha256(
                getattr(self, name).astype("<f8", copy=False).tobytes()
            ).hexdigest()
        object.__setattr__(self, "identity", canonical_hash(metadata))

    @property
    def nmo(self):
        return len(self.orbital_energies)

    @property
    def nocc(self):
        return self.electron_count // 2

    @property
    def numeric_bytes(self):
        return sum(
            getattr(self, name).nbytes
            for name in (
                "overlap",
                "hcore",
                "fock",
                "coefficients",
                "orbital_energies",
                "occupations",
            )
        )

    @property
    def ownership(self):
        return "snapshot-owned immutable host arrays"
