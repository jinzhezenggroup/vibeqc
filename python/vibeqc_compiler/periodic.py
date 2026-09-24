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
SYSTEM_SCHEMA = "vibeqc.periodic-system.v1"
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


def _integer_scalar(value: typing.Any, *, label: str, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if nonnegative and result < 0:
        raise ValueError(f"{label} must be nonnegative")
    return result


def _atomic_numbers(value: typing.Any) -> tuple[int, ...]:
    if np.iscomplexobj(value):
        raise TypeError("atomic_numbers must contain integers")
    array = np.asarray(value)
    if (
        array.ndim != 1
        or array.size == 0
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(
            "atomic_numbers must be a non-empty one-dimensional integer array"
        )
    numbers = tuple(int(number) for number in array.tolist())
    if any(number < 1 or number > 118 for number in numbers):
        raise ValueError("atomic_numbers must lie in the inclusive range 1..118")
    return numbers


def _positions_bohr(
    value: typing.Any, *, atom_count: int
) -> tuple[tuple[float, float, float], ...]:
    if np.iscomplexobj(value):
        raise ValueError("positions_bohr must be real")
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise TypeError("positions_bohr must be numeric") from error
    if array.shape != (atom_count, 3) or not np.all(np.isfinite(array)):
        raise ValueError(f"positions_bohr must be a finite {atom_count}x3 matrix")
    return tuple(
        tuple(float(component) for component in row)
        for row in array
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

        bounds = tuple(math.ceil(float(bound)) for bound in scaled_bounds)
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


@dataclass(frozen=True)
class PeriodicSystem:
    """Immutable internal periodic system with explicit cache generations.

    Cartesian atom positions are retained verbatim in Bohr. Constructing or
    updating a system never wraps, rescales, or reorders atoms. Scientific
    identities are separated from monotonic preparation generations so callers
    can distinguish equivalent data from a deliberately invalidated cache owner.
    """

    cell: PeriodicCell
    atomic_numbers: typing.Any
    positions_bohr: typing.Any
    charge: int = 0
    spin: int = 0
    geometry_generation: int = 0
    cell_generation: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell")
        numbers = _atomic_numbers(self.atomic_numbers)
        positions = _positions_bohr(self.positions_bohr, atom_count=len(numbers))
        charge = _integer_scalar(self.charge, label="charge")
        spin = _integer_scalar(self.spin, label="spin", nonnegative=True)
        geometry_generation = _integer_scalar(
            self.geometry_generation, label="geometry_generation", nonnegative=True
        )
        cell_generation = _integer_scalar(
            self.cell_generation, label="cell_generation", nonnegative=True
        )
        object.__setattr__(self, "atomic_numbers", numbers)
        object.__setattr__(self, "positions_bohr", positions)
        object.__setattr__(self, "charge", charge)
        object.__setattr__(self, "spin", spin)
        object.__setattr__(self, "geometry_generation", geometry_generation)
        object.__setattr__(self, "cell_generation", cell_generation)

    def geometry_payload(self) -> dict[str, typing.Any]:
        """Return atom identity and unwrapped Cartesian geometry only."""
        return {
            "atomic_numbers": list(self.atomic_numbers),
            "positions_bohr": [list(position) for position in self.positions_bohr],
            "position_units": "Bohr",
        }

    @property
    def geometry_identity(self) -> str:
        """Return the canonical atom/position identity, independent of the cell."""
        return canonical_hash(self.geometry_payload())

    @property
    def cell_identity(self) -> str:
        """Return the canonical direct-cell identity."""
        return self.cell.identity

    @property
    def identity(self) -> str:
        """Return the canonical scientific identity of this periodic system."""
        return canonical_hash(
            {
                "schema": SYSTEM_SCHEMA,
                "cell_identity": self.cell_identity,
                "geometry_identity": self.geometry_identity,
                "charge": self.charge,
                "spin": self.spin,
            }
        )

    @property
    def prepared_identity(self) -> str:
        """Return a cache identity that also includes explicit generations."""
        return canonical_hash(
            {
                "schema": SYSTEM_SCHEMA,
                "system_identity": self.identity,
                "geometry_generation": self.geometry_generation,
                "cell_generation": self.cell_generation,
            }
        )

    def to_payload(self) -> dict[str, typing.Any]:
        """Return a detached machine-readable system description."""
        return {
            "schema": SYSTEM_SCHEMA,
            "cell": self.cell.to_payload(),
            **self.geometry_payload(),
            "charge": self.charge,
            "spin": self.spin,
            "geometry_generation": self.geometry_generation,
            "cell_generation": self.cell_generation,
        }

    def with_positions(self, positions_bohr: typing.Any) -> PeriodicSystem:
        """Return a geometry-invalidating copy without wrapping atom positions."""
        return PeriodicSystem(
            cell=self.cell,
            atomic_numbers=self.atomic_numbers,
            positions_bohr=positions_bohr,
            charge=self.charge,
            spin=self.spin,
            geometry_generation=self.geometry_generation + 1,
            cell_generation=self.cell_generation,
        )

    def with_cell(self, cell: PeriodicCell) -> PeriodicSystem:
        """Return a cell-invalidating copy while preserving Cartesian positions."""
        return PeriodicSystem(
            cell=cell,
            atomic_numbers=self.atomic_numbers,
            positions_bohr=self.positions_bohr,
            charge=self.charge,
            spin=self.spin,
            geometry_generation=self.geometry_generation,
            cell_generation=self.cell_generation + 1,
        )
