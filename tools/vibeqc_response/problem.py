"""Immutable response-problem snapshots and nonredundant rotation layouts."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from vibeqc.profiles import canonical_hash

from tools.vibeqc_posthf.reference import ReferenceSnapshot, immutable


class ResponseUnsupported(ValueError):
    """The requested response model, reference or derivative is unavailable."""


class ResponseCompatibilityError(ValueError):
    """A response object belongs to a different problem snapshot."""


class ResponseSolveError(RuntimeError):
    """A response solve failed without producing a valid converged solution."""

    def __init__(self, result):
        self.result = result
        super().__init__(
            f"response solve failed after {result.iterations} iterations "
            f"with true residual {result.residual_norm:.3e}: {result.reason}"
        )


@dataclass(frozen=True)
class RotationLayout:
    """Explicit nonredundant occupied-virtual spatial-orbital rotations.

    A response vector stores ``x[i,a]`` in occupied-major, virtual-minor
    order.  Its density response is the symmetric MO matrix
    ``2 * sym_ov(x)``; the corresponding orbital generator is
    ``K[a,i] = -x[i,a]`` and ``K[i,a] = x[i,a]``.  This convention keeps the
    RHF Jacobian symmetric under the Euclidean inner product and avoids
    differentiating redundant occupied-occupied/virtual-virtual rotations.
    """

    occupied: tuple[int, ...]
    virtual: tuple[int, ...]
    spin_blocks: tuple[str, ...] = ("restricted",)

    def __post_init__(self):
        object.__setattr__(self, "occupied", tuple(self.occupied))
        object.__setattr__(self, "virtual", tuple(self.virtual))
        object.__setattr__(self, "spin_blocks", tuple(self.spin_blocks))
        if not self.occupied or not self.virtual:
            raise ValueError("rotation layout requires occupied and virtual spaces")
        all_indices = (*self.occupied, *self.virtual)
        if (
            any(type(i) is not int or i < 0 for i in all_indices)
            or len(set(all_indices)) != len(all_indices)
            or sorted(all_indices) != list(range(max(all_indices) + 1))
        ):
            raise ValueError("rotation spaces must be unique and contiguous from zero")
        if self.spin_blocks != ("restricted",):
            raise ResponseUnsupported(
                "only a restricted closed-shell spin block is implemented"
            )

    @classmethod
    def from_reference(cls, reference: ReferenceSnapshot):
        """Build the canonical occupied-then-virtual RHF layout."""
        if reference.algorithm not in ("RHF", "KS"):
            raise ResponseUnsupported(
                "rotation layout requires a closed-shell RHF/KS reference"
            )
        return cls(
            tuple(range(reference.nocc)),
            tuple(range(reference.nocc, reference.nmo)),
        )

    @property
    def nocc(self):
        return len(self.occupied)

    @property
    def nvirt(self):
        return len(self.virtual)

    @property
    def nmo(self):
        return self.nocc + self.nvirt

    @property
    def dimension(self):
        return self.nocc * self.nvirt

    @property
    def identity(self):
        return canonical_hash(
            {
                "occupied": self.occupied,
                "virtual": self.virtual,
                "spin_blocks": self.spin_blocks,
                "ordering": "occupied-major-virtual-minor",
                "parameterization": "density-response-symmetric-ov",
            }
        )

    def validate_vector(self, values):
        """Return a finite immutable response vector of the declared shape."""
        vector = np.asarray(values)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"response vector must have shape ({self.dimension},), got {vector.shape}"
            )
        if np.iscomplexobj(vector) or not np.isfinite(vector).all():
            raise ValueError("response vectors must be finite real FP64 values")
        return immutable(vector)

    def as_ia(self, values):
        """Return ``x[i,a]`` with shape ``(nocc,nvirt)``."""
        return self.validate_vector(values).reshape(self.nocc, self.nvirt)

    def pack(self, values):
        """Pack an ``(nocc,nvirt)`` array into the canonical vector order."""
        array = np.asarray(values)
        if array.shape != (self.nocc, self.nvirt):
            raise ValueError(
                f"rotation array must have shape ({self.nocc},{self.nvirt})"
            )
        return self.validate_vector(array.reshape(-1))

    def density_matrix(self, values):
        """Return the symmetric MO density-response matrix ``2 sym_ov(x)``."""
        x = self.as_ia(values)
        result = np.zeros((self.nmo, self.nmo))
        result[np.ix_(self.occupied, self.virtual)] = x
        result[np.ix_(self.virtual, self.occupied)] = x.T
        return 2.0 * result

    def generator_matrix(self, values):
        """Return the skew orbital generator ``K`` used by finite rotations."""
        x = self.as_ia(values)
        result = np.zeros((self.nmo, self.nmo))
        result[np.ix_(self.virtual, self.occupied)] = -x.T
        result[np.ix_(self.occupied, self.virtual)] = x
        return result


@dataclass(frozen=True)
class ResponseProblem:
    """Complete compatibility snapshot for one orbital-response operator.

    ``identity`` binds the reference values and every numerical/model choice.
    Equal dimensions are deliberately insufficient: changing the reference
    coefficients, Hamiltonian, functional/grid or operator backend invalidates
    the problem and therefore any retained Krylov subspace.
    """

    reference: ReferenceSnapshot
    layout: RotationLayout
    method: str
    operator_identity: str
    model_hash: str
    rhs_layout: str = "ov-response-vector"
    gauge: str = "canonical-nonredundant-ov"
    overlap_metric: str = "mo-orthonormal"
    perturbation_labels: tuple[str, ...] = ()
    identity: str = field(init=False)

    def __post_init__(self):
        if self.method not in ("rhf", "cpks"):
            raise ResponseUnsupported(f"unsupported response method {self.method!r}")
        if self.reference.algorithm == "RHF":
            if self.method != "rhf":
                raise ResponseUnsupported(
                    "CPKS requires a converged KS reference, not an RHF reference"
                )
        elif self.reference.algorithm == "KS":
            if self.method != "cpks":
                raise ResponseUnsupported(
                    "a KS reference requires the CPKS response method"
                )
            if (
                not self.reference.functional_identity
                or not self.reference.grid_identity
            ):
                raise ResponseUnsupported(
                    "KS response requires functional and grid identities"
                )
        else:
            raise ResponseUnsupported(
                f"unsupported reference algorithm {self.reference.algorithm!r}"
            )
        if not self.reference.converged:
            raise ResponseUnsupported("unconverged references cannot enter response")
        if (
            self.layout.nmo != self.reference.nmo
            or set(self.layout.occupied) != set(range(self.reference.nocc))
            or set(self.layout.virtual)
            != set(range(self.reference.nocc, self.reference.nmo))
        ):
            raise ValueError("rotation layout does not match the reference occupations")
        for name in ("operator_identity", "model_hash", "rhs_layout", "gauge"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a nonempty identity")
        if self.overlap_metric != "mo-orthonormal":
            raise ResponseUnsupported(
                "only the canonical MO orthonormal overlap metric is implemented"
            )
        if self.gauge != "canonical-nonredundant-ov":
            raise ResponseUnsupported(
                f"unsupported gauge {self.gauge!r}: only canonical "
                "nonredundant occupied-virtual response is implemented"
            )
        if self.rhs_layout != "ov-response-vector":
            raise ResponseUnsupported(f"unsupported RHS layout {self.rhs_layout!r}")
        labels = tuple(self.perturbation_labels)
        if any(not isinstance(label, str) or not label for label in labels):
            raise ValueError("perturbation labels must be nonempty strings")
        if len(set(labels)) != len(labels):
            raise ValueError("perturbation labels must be unique")
        object.__setattr__(self, "perturbation_labels", labels)
        object.__setattr__(self, "identity", canonical_hash(self._payload()))

    def _payload(self):
        return {
            "reference_identity": self.reference.identity,
            "reference_algorithm": self.reference.algorithm,
            "reference_dimension": self.reference.nmo,
            "reference_electron_count": self.reference.electron_count,
            "layout_identity": self.layout.identity,
            "method": self.method,
            "operator_identity": self.operator_identity,
            "model_hash": self.model_hash,
            "rhs_layout": self.rhs_layout,
            "gauge": self.gauge,
            "overlap_metric": self.overlap_metric,
        }

    @property
    def compatibility_identity(self):
        """Identity for safe Krylov reuse across different perturbation RHSs.

        Human-readable perturbation labels are intentionally excluded.  The
        reference, operator, model, layout, gauge and metric remain binding.
        """
        return self.identity

    @classmethod
    def from_reference(
        cls,
        reference: ReferenceSnapshot,
        *,
        method="rhf",
        operator_identity,
        model_hash=None,
        rhs_layout="ov-response-vector",
        gauge="canonical-nonredundant-ov",
        overlap_metric="mo-orthonormal",
        perturbation_labels=(),
    ):
        """Bind a converged reference to one concrete operator backend."""
        layout = RotationLayout.from_reference(reference)
        if model_hash is None:
            model_hash = canonical_hash(
                {
                    "hamiltonian_id": reference.hamiltonian_id,
                    "functional_identity": reference.functional_identity,
                    "grid_identity": reference.grid_identity,
                    "basis_hash": reference.basis_hash,
                    "geometry_hash": reference.geometry_hash,
                    "screening_tolerance": reference.screening_tolerance,
                    "precision": reference.precision,
                }
            )
        return cls(
            reference,
            layout,
            method,
            operator_identity,
            model_hash,
            rhs_layout,
            gauge,
            overlap_metric,
            tuple(perturbation_labels),
        )

    @property
    def dimension(self):
        return self.layout.dimension

    @property
    def reference_identity(self):
        return self.reference.identity

    def _active_rotation_gaps(self):
        """Return occupied-virtual energy denominators of active rotations.

        The nonredundant response space contains only occupied-virtual
        rotations. Occupied-occupied and virtual-virtual splittings are
        redundant directions that are excluded from the unknowns, so they must
        not enter the stability gate. The result is indexed
        ``[virtual, occupied]``.
        """
        energies = np.asarray(self.reference.orbital_energies, dtype=float)
        occupied = np.asarray(self.layout.occupied, dtype=int)
        virtual = np.asarray(self.layout.virtual, dtype=int)
        return energies[virtual][:, None] - energies[occupied][None, :]

    @property
    def diagnostics(self):
        """Report response-relevant reference conditioning without clipping.

        ``minimum_ov_gap`` is the smallest absolute occupied-virtual orbital
        energy denominator in the active rotation space. Same-occupancy
        degeneracies are excluded because those rotations are redundant and
        never appear as response unknowns.
        """
        gaps = self._active_rotation_gaps()
        minimum_gap = float(np.min(np.abs(gaps))) if gaps.size else float("inf")
        return {
            "minimum_ov_gap": minimum_gap,
            "near_degenerate": bool(minimum_gap <= 1e-8),
            "overlap_min_eigenvalue": float(
                np.linalg.eigvalsh(self.reference.overlap)[0]
            ),
            "scf_residual": float(self.reference.scf_residual),
        }

    def require_stable(self, *, orbital_gap_tolerance=1e-8):
        """Fail closed when a caller requires a non-degenerate reference.

        A near-degenerate reference is not silently altered or regularized.
        Callers that can solve it with a true residual may proceed explicitly;
        consumers requiring derivative stability must call this gate.
        """
        if not np.isfinite(orbital_gap_tolerance) or orbital_gap_tolerance < 0:
            raise ValueError("orbital_gap_tolerance must be finite and nonnegative")
        diagnostics = self.diagnostics
        if diagnostics["minimum_ov_gap"] <= orbital_gap_tolerance:
            raise ResponseUnsupported(
                "near-degenerate occupied-virtual response reference: "
                f"minimum gap {diagnostics['minimum_ov_gap']:.3e} <= "
                f"{orbital_gap_tolerance:.3e}; no denominator was clamped"
            )
        return self

    def validate_rhs(self, values):
        """Return an immutable ``(dimension,k)`` RHS with a matching layout."""
        rhs = np.asarray(values)
        if rhs.ndim == 1:
            rhs = rhs[:, None]
        if rhs.ndim != 2 or rhs.shape[0] != self.dimension:
            raise ValueError(
                f"RHS must have shape ({self.dimension},k), got {rhs.shape}"
            )
        if rhs.shape[1] < 1 or np.iscomplexobj(rhs) or not np.isfinite(rhs).all():
            raise ValueError("RHS must contain finite real FP64 columns")
        if self.perturbation_labels and len(self.perturbation_labels) != rhs.shape[1]:
            raise ValueError("perturbation labels do not match RHS columns")
        return immutable(rhs)

    def assert_compatible(self, other):
        """Reject any retained state that does not describe this exact problem."""
        if not isinstance(other, ResponseProblem):
            raise TypeError("expected ResponseProblem")
        if self.identity != other.identity:
            fields = []
            for name in (
                "reference_identity",
                "method",
                "operator_identity",
                "model_hash",
                "rhs_layout",
                "gauge",
                "overlap_metric",
            ):
                if getattr(self, name) != getattr(other, name):
                    fields.append(name)
            if self.layout.identity != other.layout.identity:
                fields.append("layout")
            raise ResponseCompatibilityError(
                "response problem mismatch in " + ", ".join(sorted(fields))
            )
