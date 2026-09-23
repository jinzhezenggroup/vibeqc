"""Backend-neutral periodic cell identity and coordinate transforms.

This module is intentionally internal. It defines the scientific cell contract
needed to share periodic topology across GFN and future Gaussian methods; it does
not admit any public periodic calculation capability by itself.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

from vibeqc_compiler.common.provenance import canonical_hash

CELL_SCHEMA = "vibeqc.periodic-cell.v1"


def _vector3(value: typing.Any, *, label: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite length-3 vector")
    return (float(array[0]), float(array[1]), float(array[2]))


def _matrix3(
    value: typing.Any, *, label: str
) -> tuple[tuple[float, float, float], ...]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite 3x3 matrix")
    return (
        (float(array[0, 0]), float(array[0, 1]), float(array[0, 2])),
        (float(array[1, 0]), float(array[1, 1]), float(array[1, 2])),
        (float(array[2, 0]), float(array[2, 1]), float(array[2, 2])),
    )


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
        if self.units != "Bohr":
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
