"""Backend-neutral periodic cell identity, transforms, and lattice images.

This module is intentionally internal. It defines the scientific cell contract
needed to share periodic topology across GFN and future Gaussian methods; it does
not admit any public periodic calculation capability by itself.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash

CELL_SCHEMA = "vibeqc.periodic-cell.v1"
DEFAULT_MAX_IMAGE_CANDIDATES = 1_000_000


def _vector3(value: typing.Any, *, label: str) -> tuple[float, float, float]:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite length-3 vector")
    return (float(array[0]), float(array[1]), float(array[2]))


def _matrix3(
    value: typing.Any, *, label: str
) -> tuple[tuple[float, float, float], ...]:
    if np.iscomplexobj(value):
        raise ValueError(f"{label} must be real")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    return (
        (float(array[0, 0]), float(array[0, 1]), float(array[0, 2])),
        (float(array[1, 0]), float(array[1, 1]), float(array[1, 2])),
        (float(array[2, 0]), float(array[2, 1]), float(array[2, 2])),
    )


def _nonnegative_scalar(value: typing.Any, *, label: str) -> float:
    if isinstance(value, (str, bytes)) or np.iscomplexobj(value):
        raise TypeError(f"{label} must be a real scalar")
    array = np.asarray(value)
    if array.shape != ():
        raise TypeError(f"{label} must be a real scalar")
    try:
        scalar = float(array)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError(f"{label} must be a real scalar") from error
    if not math.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return scalar


@dataclass(frozen=True)
class PeriodicCell:
    """Immutable 3-D periodic cell with row-vector lattice semantics.

    ``lattice`` contains the direct lattice vectors ``a``, ``b`` and ``c`` as
    rows in Bohr. A fractional row vector ``u`` maps to Cartesian coordinates
    as ``u @ lattice``. Partial periodicity is deliberately rejected in v1.
    """

    lattice: typing.Any
    periodic_axes: tuple[bool, bool, bool] = (True, True, True)
    units: str = "Bohr"

    def __post_init__(self) -> None:
        lattice = _matrix3(self.lattice, label="lattice")
        axes = tuple(self.periodic_axes)
        if len(axes) != 3 or any(type(axis) is not bool for axis in axes):
            raise TypeError("periodic_axes must contain exactly three booleans")
        if axes != (True, True, True):
            raise NotImplementedError("PeriodicCell v1 supports only 3-D periodicity")
        if not isinstance(self.units, str) or self.units != "Bohr":
            raise ValueError("PeriodicCell v1 lattice units must be Bohr")

        determinant = float(np.linalg.det(np.asarray(lattice, dtype=np.float64)))
        if not np.isfinite(determinant) or determinant <= 0.0:
            raise ValueError("lattice must be finite, nonsingular, and right-handed")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "periodic_axes", axes)

    @property
    def volume(self) -> float:
        """Return the positive cell volume in Bohr^3."""
        return float(np.linalg.det(np.asarray(self.lattice, dtype=np.float64)))

    @property
    def reciprocal_lattice(self) -> tuple[tuple[float, float, float], ...]:
        """Return row reciprocal vectors satisfying ``A @ B.T = 2*pi*I``."""
        direct = np.asarray(self.lattice, dtype=np.float64)
        reciprocal = 2.0 * np.pi * np.linalg.inv(direct).T
        return _matrix3(reciprocal, label="reciprocal lattice")

    @property
    def identity(self) -> str:
        """Return the exact scientific identity of this cell contract."""
        return canonical_hash(self.to_payload())

    def to_payload(self) -> dict[str, typing.Any]:
        """Return a detached, versioned machine-readable cell description."""
        return {
            "schema": CELL_SCHEMA,
            "lattice_bohr": [list(row) for row in self.lattice],
            "periodic_axes": list(self.periodic_axes),
            "units": self.units,
            "row_vector_convention": "fractional @ lattice",
        }

    def fractional_to_cartesian(self, value: typing.Any) -> tuple[float, float, float]:
        """Map one fractional row vector to Cartesian Bohr coordinates."""
        fractional = np.asarray(_vector3(value, label="fractional coordinate"))
        cartesian = fractional @ np.asarray(self.lattice, dtype=np.float64)
        return _vector3(cartesian, label="Cartesian coordinate")

    def cartesian_to_fractional(self, value: typing.Any) -> tuple[float, float, float]:
        """Map one Cartesian row vector in Bohr to fractional coordinates."""
        cartesian = np.asarray(_vector3(value, label="Cartesian coordinate"))
        inverse = np.linalg.inv(np.asarray(self.lattice, dtype=np.float64))
        fractional = cartesian @ inverse
        return _vector3(fractional, label="fractional coordinate")

    def wrap_fractional(self, value: typing.Any) -> tuple[float, float, float]:
        """Explicitly wrap one fractional coordinate into the half-open unit cell."""
        fractional = np.asarray(_vector3(value, label="fractional coordinate"))
        wrapped = fractional - np.floor(fractional)
        wrapped[wrapped == 1.0] = 0.0
        return _vector3(wrapped, label="wrapped fractional coordinate")

    def wrap_cartesian(self, value: typing.Any) -> tuple[float, float, float]:
        """Explicitly wrap one Cartesian point without changing atom identity."""
        return self.fractional_to_cartesian(
            self.wrap_fractional(self.cartesian_to_fractional(value))
        )

    def lattice_image_offsets(
        self,
        cutoff_bohr: typing.Any,
        *,
        max_candidates: int = DEFAULT_MAX_IMAGE_CANDIDATES,
    ) -> tuple[tuple[int, int, int], ...]:
        """Enumerate lattice translations whose Cartesian norm is within ``cutoff_bohr``.

        The returned offsets are integer coefficients ``n`` for translations
        ``n @ lattice`` and are ordered lexicographically. The origin is included.
        Reciprocal-lattice norms provide a complete finite integer search box:
        ``|n_i| <= |T| |b_i| / (2*pi)``. The box is conservatively rounded
        outward, then every candidate is filtered by its exact Cartesian norm.

        ``max_candidates`` bounds the search box before allocation or iteration.
        This is a safety bound, not a scientific cutoff, and prevents a
        near-singular-but-valid cell plus a large radius from creating unbounded
        host work. No atom is wrapped or otherwise mutated by this operation.
        """

        cutoff = _nonnegative_scalar(cutoff_bohr, label="image cutoff")
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            raise ValueError("max_candidates must be a positive integer")

        reciprocal = np.asarray(self.reciprocal_lattice, dtype=np.float64)
        scaled_bounds = cutoff * np.linalg.norm(reciprocal, axis=1) / (2.0 * math.pi)
        if not np.all(np.isfinite(scaled_bounds)):
            raise ValueError("periodic image enumeration bounds overflow")

        bounds = tuple(int(math.ceil(float(bound))) for bound in scaled_bounds)
        candidate_count = math.prod(2 * bound + 1 for bound in bounds)
        if candidate_count > max_candidates:
            raise ValueError(
                "periodic image candidate bound "
                f"{candidate_count} exceeds max_candidates={max_candidates}"
            )

        direct = np.asarray(self.lattice, dtype=np.float64)
        inclusive_cutoff = math.nextafter(cutoff, math.inf)
        offsets: list[tuple[int, int, int]] = []
        for first in range(-bounds[0], bounds[0] + 1):
            for second in range(-bounds[1], bounds[1] + 1):
                for third in range(-bounds[2], bounds[2] + 1):
                    offset = (first, second, third)
                    translation = np.asarray(offset, dtype=np.float64) @ direct
                    if (
                        math.hypot(
                            float(translation[0]),
                            float(translation[1]),
                            float(translation[2]),
                        )
                        <= inclusive_cutoff
                    ):
                        offsets.append(offset)
        return tuple(offsets)
