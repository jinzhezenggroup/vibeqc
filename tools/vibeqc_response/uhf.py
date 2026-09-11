"""Validated UHF response snapshot, spin layout and matrix-free Jacobian."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from hashlib import sha256

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import immutable

from .backends import CudaDFJKBackend
from .problem import ResponseCompatibilityError


def _symmetric_matrix(value, nbf, name, tolerance):
    """Return one finite symmetric AO matrix in the common UHF basis."""
    array = immutable(value, shape=(nbf, nbf))
    if np.max(np.abs(array - array.T)) > tolerance:
        raise ValueError(f"{name} is not symmetric")
    return array


@dataclass(frozen=True, eq=False)
class UHFReferenceSnapshot:
    """Immutable canonical alpha/beta UHF state for orbital response.

    This is intentionally separate from ``ReferenceSnapshot``: that existing
    public contract models a single closed-shell 2/0 spatial density, while a
    UHF response state must preserve two independently canonical spin blocks.
    """

    overlap: np.ndarray
    hcore: np.ndarray
    fock_alpha: np.ndarray
    fock_beta: np.ndarray
    coefficients_alpha: np.ndarray
    coefficients_beta: np.ndarray
    orbital_energies_alpha: np.ndarray
    orbital_energies_beta: np.ndarray
    occupations_alpha: np.ndarray
    occupations_beta: np.ndarray
    reference_energy: float
    scf_residual: float
    geometry_hash: str
    basis_hash: str
    generation_id: str
    hamiltonian_id: str = "conventional-unscreened"
    representation: str = "cartesian"
    precision: str = "float64"
    hf_backend: str = "cpu-reference"
    converged: bool = True
    validation_tolerance: float = 1e-8
    identity: str = field(init=False)

    def __post_init__(self):
        if self.precision != "float64":
            raise ValueError("UHF response requires real FP64 reference buffers")
        if self.representation not in ("cartesian", "real_spherical"):
            raise ValueError("unknown AO representation")
        if not self.converged:
            raise ValueError("unconverged UHF cannot enter response")
        if not 0 < self.validation_tolerance <= 1e-6:
            raise ValueError("validation_tolerance must be in (0, 1e-6]")
        for name in ("reference_energy", "scf_residual"):
            value = getattr(self, name)
            if not math.isfinite(value) or (name == "scf_residual" and value < 0):
                raise ValueError(f"invalid {name}")
        if self.scf_residual > self.validation_tolerance:
            raise ValueError(
                "UHF scf_residual exceeds validation_tolerance; "
                "an unconverged reference cannot enter response"
            )
        for name in (
            "geometry_hash",
            "basis_hash",
            "generation_id",
            "hamiltonian_id",
            "hf_backend",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a nonempty identity")

        overlap = immutable(self.overlap)
        if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
            raise ValueError("UHF overlap must be a square AO matrix")
        nbf = overlap.shape[0]
        object.__setattr__(
            self,
            "overlap",
            _symmetric_matrix(overlap, nbf, "overlap", self.validation_tolerance),
        )
        for name in ("hcore", "fock_alpha", "fock_beta"):
            object.__setattr__(
                self,
                name,
                _symmetric_matrix(
                    getattr(self, name), nbf, name, self.validation_tolerance
                ),
            )
        identity_spins = {}
        occupied_counts = {}
        for spin in ("alpha", "beta"):
            coefficients = immutable(
                getattr(self, f"coefficients_{spin}"), shape=(nbf, nbf)
            )
            if (
                np.max(
                    np.abs(coefficients.T @ self.overlap @ coefficients - np.eye(nbf))
                )
                > self.validation_tolerance
            ):
                raise ValueError(f"{spin} coefficients are not overlap orthonormal")
            energies = immutable(
                getattr(self, f"orbital_energies_{spin}"), shape=(nbf,)
            )
            occupations = immutable(getattr(self, f"occupations_{spin}"), shape=(nbf,))
            if not np.array_equal(occupations, (occupations > 0.5).astype(np.float64)):
                raise ValueError(f"{spin} occupations must be explicit 1/0 values")
            nocc = int(np.sum(occupations))
            expected = np.zeros(nbf)
            expected[:nocc] = 1.0
            if not np.array_equal(occupations, expected):
                raise ValueError(
                    f"{spin} occupations require ordered occupied and virtual orbitals"
                )
            occupied_counts[spin] = nocc
            if np.any(np.diff(energies) < -self.validation_tolerance):
                raise ValueError(f"{spin} canonical orbital energies must be ascending")
            fock = getattr(self, f"fock_{spin}")
            residual = np.max(
                np.abs(fock @ coefficients - self.overlap @ coefficients * energies)
            )
            if residual > self.validation_tolerance:
                raise ValueError(
                    f"{spin} Fock/coefficient canonical residual is too large"
                )
            object.__setattr__(self, f"coefficients_{spin}", coefficients)
            object.__setattr__(self, f"orbital_energies_{spin}", energies)
            object.__setattr__(self, f"occupations_{spin}", occupations)
            identity_spins[spin] = {
                "coefficients": sha256(
                    coefficients.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "energies": sha256(
                    energies.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
                "occupations": sha256(
                    occupations.astype("<f8", copy=False).tobytes()
                ).hexdigest(),
            }
        if sum(occupied_counts.values()) == 0:
            raise ValueError(
                "UHF reference requires at least one occupied spin orbital"
            )
        object.__setattr__(
            self,
            "identity",
            canonical_hash(
                {
                    "kind": "uhf-response-reference-v1",
                    "overlap": sha256(
                        self.overlap.astype("<f8", copy=False).tobytes()
                    ).hexdigest(),
                    "hcore": sha256(
                        self.hcore.astype("<f8", copy=False).tobytes()
                    ).hexdigest(),
                    "fock_alpha": sha256(
                        self.fock_alpha.astype("<f8", copy=False).tobytes()
                    ).hexdigest(),
                    "fock_beta": sha256(
                        self.fock_beta.astype("<f8", copy=False).tobytes()
                    ).hexdigest(),
                    "spins": identity_spins,
                    "energy": self.reference_energy,
                    "residual": self.scf_residual,
                    "geometry": self.geometry_hash,
                    "basis": self.basis_hash,
                    "generation": self.generation_id,
                    "hamiltonian": self.hamiltonian_id,
                    "representation": self.representation,
                    "backend": self.hf_backend,
                }
            ),
        )

    @property
    def nbf(self):
        """Number of AOs and canonical MOs in each spin channel."""
        return self.overlap.shape[0]

    def nocc(self, spin):
        """Return the occupied alpha or beta orbital count."""
        if spin not in ("alpha", "beta"):
            raise ValueError("spin must be alpha or beta")
        return int(np.sum(getattr(self, f"occupations_{spin}")))


@dataclass(frozen=True)
class UHFSpinRotationLayout:
    """Packed alpha-then-beta nonredundant occupied-virtual rotations."""

    alpha_occupied: tuple[int, ...]
    alpha_virtual: tuple[int, ...]
    beta_occupied: tuple[int, ...]
    beta_virtual: tuple[int, ...]

    def __post_init__(self):
        for spin in ("alpha", "beta"):
            occupied = tuple(getattr(self, f"{spin}_occupied"))
            virtual = tuple(getattr(self, f"{spin}_virtual"))
            object.__setattr__(self, f"{spin}_occupied", occupied)
            object.__setattr__(self, f"{spin}_virtual", virtual)
            # A valid high-spin reference can have an empty occupied block in
            # one spin channel. The packed response contribution is then
            # zero-dimensional, while the other channel remains active.
            values = (*occupied, *virtual)
            if not values or any(type(i) is not int or i < 0 for i in values):
                raise ValueError(
                    f"{spin} layout requires nonnegative occupied/virtual spaces"
                )
            if len(set(values)) != len(values) or sorted(values) != list(
                range(max(values) + 1)
            ):
                raise ValueError(f"{spin} rotation spaces must be contiguous from zero")

    @classmethod
    def from_reference(cls, reference):
        """Build spin layouts from a canonical UHF reference."""
        return cls(
            tuple(range(reference.nocc("alpha"))),
            tuple(range(reference.nocc("alpha"), reference.nbf)),
            tuple(range(reference.nocc("beta"))),
            tuple(range(reference.nocc("beta"), reference.nbf)),
        )

    def spaces(self, spin):
        """Return the occupied and virtual MO indices for one spin block."""
        if spin not in ("alpha", "beta"):
            raise ValueError("spin must be alpha or beta")
        return getattr(self, f"{spin}_occupied"), getattr(self, f"{spin}_virtual")

    def block_dimension(self, spin):
        """Return the number of independent rotations in one spin block."""
        occupied, virtual = self.spaces(spin)
        return len(occupied) * len(virtual)

    @property
    def dimension(self):
        """Total alpha-plus-beta vector size."""
        return self.block_dimension("alpha") + self.block_dimension("beta")

    @property
    def identity(self):
        """Hash the spin-resolved parameterization and ordering convention."""
        return canonical_hash(
            {
                "alpha": self.spaces("alpha"),
                "beta": self.spaces("beta"),
                "ordering": "alpha-then-beta; occupied-major-virtual-minor",
                "parameterization": "spin-density-symmetric-ov",
            }
        )

    def validate_vector(self, values):
        """Validate and freeze one alpha-then-beta response vector."""
        array = np.asarray(values)
        if array.shape != (self.dimension,):
            raise ValueError(
                f"UHF response vector must have shape ({self.dimension},), got {array.shape}"
            )
        if np.iscomplexobj(array) or not np.isfinite(array).all():
            raise ValueError("UHF response vectors must be finite real FP64 values")
        return immutable(array)

    def split(self, values):
        """Unpack alpha/beta occupied-virtual matrices from a response vector."""
        vector = self.validate_vector(values)
        alpha_dimension = self.block_dimension("alpha")
        alpha_shape = (len(self.alpha_occupied), len(self.alpha_virtual))
        beta_shape = (len(self.beta_occupied), len(self.beta_virtual))
        return vector[:alpha_dimension].reshape(alpha_shape), vector[
            alpha_dimension:
        ].reshape(beta_shape)

    def pack(self, alpha, beta):
        """Pack alpha/beta occupied-virtual matrices in canonical vector order."""
        blocks = []
        for spin, value in (("alpha", alpha), ("beta", beta)):
            occupied, virtual = self.spaces(spin)
            array = np.asarray(value)
            if array.shape != (len(occupied), len(virtual)):
                raise ValueError(f"{spin} rotation array has incompatible shape")
            blocks.append(array.reshape(-1))
        return self.validate_vector(np.concatenate(blocks))

    def density_matrix(self, spin, values):
        """Return the spin-density response ``sym_ov(x_spin)`` in MO space."""
        alpha, beta = self.split(values)
        rotations = {"alpha": alpha, "beta": beta}
        occupied, virtual = self.spaces(spin)
        matrix = np.zeros((len(occupied) + len(virtual),) * 2)
        matrix[np.ix_(occupied, virtual)] = rotations[spin]
        matrix[np.ix_(virtual, occupied)] = rotations[spin].T
        return matrix

    def generator_matrix(self, spin, values):
        """Return the skew orbital generator for a finite UHF rotation."""
        alpha, beta = self.split(values)
        rotations = {"alpha": alpha, "beta": beta}
        occupied, virtual = self.spaces(spin)
        matrix = np.zeros((len(occupied) + len(virtual),) * 2)
        matrix[np.ix_(virtual, occupied)] = -rotations[spin].T
        matrix[np.ix_(occupied, virtual)] = rotations[spin]
        return matrix


@dataclass(frozen=True)
class UHFResponseProblem:
    """Reference-bound UHF response problem compatible with shared GMRES."""

    reference: UHFReferenceSnapshot
    layout: UHFSpinRotationLayout
    operator_identity: str
    perturbation_labels: tuple[str, ...] = ()
    identity: str = field(init=False)

    def __post_init__(self):
        if not isinstance(self.reference, UHFReferenceSnapshot):
            raise TypeError("UHF response requires UHFReferenceSnapshot")
        if not isinstance(self.layout, UHFSpinRotationLayout):
            raise TypeError("UHF response requires UHFSpinRotationLayout")
        if not isinstance(self.operator_identity, str) or not self.operator_identity:
            raise ValueError("operator_identity must be a nonempty identity")
        for spin in ("alpha", "beta"):
            occupied, virtual = self.layout.spaces(spin)
            nocc = self.reference.nocc(spin)
            if set(occupied) != set(range(nocc)) or set(virtual) != set(
                range(nocc, self.reference.nbf)
            ):
                raise ValueError(f"{spin} rotation layout does not match occupations")
        labels = tuple(self.perturbation_labels)
        if any(not isinstance(label, str) or not label for label in labels) or len(
            set(labels)
        ) != len(labels):
            raise ValueError("perturbation labels must be unique nonempty strings")
        object.__setattr__(self, "perturbation_labels", labels)
        object.__setattr__(
            self,
            "identity",
            canonical_hash(
                {
                    "reference": self.reference.identity,
                    "layout": self.layout.identity,
                    "operator": self.operator_identity,
                    "method": "uhf",
                }
            ),
        )

    @classmethod
    def from_reference(cls, reference, *, operator_identity, perturbation_labels=()):
        """Bind a converged UHF reference to a concrete response backend."""
        return cls(
            reference,
            UHFSpinRotationLayout.from_reference(reference),
            operator_identity,
            perturbation_labels,
        )

    @property
    def dimension(self):
        """Combined alpha/beta vector dimension."""
        return self.layout.dimension

    @property
    def compatibility_identity(self):
        """Compatibility identity used by Krylov recycling."""
        return self.identity

    def validate_rhs(self, values):
        """Normalize a single or multiple finite UHF RHS vectors."""
        array = np.asarray(values)
        if array.ndim == 1:
            array = array[:, None]
        if array.ndim != 2 or array.shape[0] != self.dimension:
            raise ValueError(
                f"UHF RHS must have shape ({self.dimension}, nrhs), got {array.shape}"
            )
        if self.perturbation_labels and array.shape[1] != len(self.perturbation_labels):
            raise ValueError("UHF RHS column count does not match perturbation labels")
        if np.iscomplexobj(array) or not np.isfinite(array).all():
            raise ValueError("UHF RHS values must be finite real FP64")
        return immutable(array)

    def assert_compatible(self, other):
        """Reject stale alpha/beta recycle spaces before any projection."""
        if not isinstance(other, UHFResponseProblem) or self.identity != other.identity:
            raise ResponseCompatibilityError(
                "UHF response compatibility identity differs"
            )


def uhf_operator_identity(backend):
    """Return the identity for a generic UHF Coulomb/exchange backend."""
    return canonical_hash(
        {
            "method": "uhf",
            "backend": backend.identity,
            "coulomb": "J[delta-P-alpha + delta-P-beta]",
            "exchange": "-K[delta-P-spin]",
            "parameterization": "spin-density-symmetric-ov",
        }
    )


class UHFResponseOperator:
    """Matrix-free UHF Jacobian with Coulomb spin coupling and spin exchange."""

    def __init__(self, problem, backend):
        if not isinstance(problem, UHFResponseProblem):
            raise TypeError("expected UHFResponseProblem")
        if isinstance(backend, CudaDFJKBackend):
            raise NotImplementedError(
                "UHF response requires a validated spin-resolved CUDA J/K plan"
            )
        # Matching dimensions/reference metadata cannot certify the ERI action.
        # Recycling must retain the identity of the backend actually applied.
        if problem.operator_identity != uhf_operator_identity(backend):
            raise ValueError("problem operator_identity does not match its UHF backend")
        self.problem = problem
        self.backend = backend
        validate = getattr(backend, "validate_reference", None)
        if validate is not None:
            validate(problem.reference)
        self.dimension = problem.dimension

        self.statistics = {
            "actions": 0,
            "transpose_actions": 0,
            "peak_workspace_bytes": 0,
        }

    @classmethod
    def build_problem(cls, reference, backend, *, perturbation_labels=()):
        """Create one UHF response problem for a verified generic J/K backend."""
        if not isinstance(reference, UHFReferenceSnapshot):
            raise TypeError("UHFResponseOperator requires UHFReferenceSnapshot")
        return UHFResponseProblem.from_reference(
            reference,
            operator_identity=uhf_operator_identity(backend),
            perturbation_labels=perturbation_labels,
        )

    @property
    def identity(self):
        """The bound UHF operator identity."""
        return self.problem.operator_identity

    def apply(self, vector):
        """Apply the coupled alpha/beta UHF response Jacobian.

        The native UHF Fock equations define Coulomb from the total density and
        exchange from the matching spin density; using three generic J/K calls
        preserves those semantics for both dense-oracle and streamed backends.
        """
        vector = self.problem.layout.validate_vector(vector)
        reference = self.problem.reference
        alpha_mo = self.problem.layout.density_matrix("alpha", vector)
        beta_mo = self.problem.layout.density_matrix("beta", vector)
        alpha_density = (
            reference.coefficients_alpha @ alpha_mo @ reference.coefficients_alpha.T
        )
        beta_density = (
            reference.coefficients_beta @ beta_mo @ reference.coefficients_beta.T
        )
        coulomb, _ = self.backend.coulomb_exchange(alpha_density + beta_density)
        _, alpha_exchange = self.backend.coulomb_exchange(alpha_density)
        _, beta_exchange = self.backend.coulomb_exchange(beta_density)
        alpha_x, beta_x = self.problem.layout.split(vector)
        response = []
        for spin, rotations, fock_response in (
            ("alpha", alpha_x, coulomb - alpha_exchange),
            ("beta", beta_x, coulomb - beta_exchange),
        ):
            occupied, virtual = self.problem.layout.spaces(spin)
            coefficients = getattr(reference, f"coefficients_{spin}")
            energies = getattr(reference, f"orbital_energies_{spin}")
            response_mo = coefficients.T @ fock_response @ coefficients
            response.append(
                (energies[list(virtual)][None, :] - energies[list(occupied)][:, None])
                * rotations
                + response_mo[np.ix_(virtual, occupied)].T
            )
        self.statistics["actions"] += 1
        self.statistics["peak_workspace_bytes"] = max(
            self.statistics["peak_workspace_bytes"],
            10 * reference.nbf * reference.nbf * 8,
        )
        return self.problem.layout.pack(*response)

    def apply_transpose(self, vector):
        """Apply the transpose in the real canonical spin-orbital gauge."""
        self.statistics["transpose_actions"] += 1
        return self.apply(vector)

    def apply_many(self, matrix):
        """Apply the response action to every RHS column."""
        values = self.problem.validate_rhs(matrix)
        return immutable(
            np.column_stack(
                [self.apply(values[:, column]) for column in range(values.shape[1])]
            )
        )

    def dot_identity(self, left, right):
        """Return the Euclidean transpose-identity defect."""
        return abs(
            float(np.dot(self.apply(left), right))
            - float(np.dot(left, self.apply_transpose(right)))
        )

    def to_dense(self):
        """Materialize a tiny diagnostic response matrix through JVP columns."""
        if self.dimension == 0:
            return immutable(np.empty((0, 0), dtype=np.float64))
        eye = np.eye(self.dimension)
        return immutable(
            np.column_stack(
                [self.apply(eye[:, column]) for column in range(self.dimension)]
            )
        )
