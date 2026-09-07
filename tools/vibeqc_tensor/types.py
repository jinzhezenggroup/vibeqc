"""Logical tensor types; strides, devices, and allocation policy are not IR.

Index names are equation notation. Space identity, spin, and the selected
half-open range are semantic and cannot be inferred from equal dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

MAX_ELEMENTS = (1 << 63) - 1
SPACE_KINDS = frozenset({"occupied", "virtual", "ao", "auxiliary", "batch", "spin"})


def checked_size(value: int, name: str) -> int:
    """Use a portable signed-64-bit size contract, rejecting booleans too."""
    if type(value) is not int or not 0 <= value <= MAX_ELEMENTS:
        raise ValueError(f"{name} must be a nonnegative signed-64-bit integer")
    return value


def checked_shape(shape: tuple[int, ...], itemsize: int = 8) -> int:
    """Check dimensions and byte products before any interpreter allocation."""
    for n in shape:
        checked_size(n, "dimension")
    size = prod(shape)
    checked_size(size * itemsize, "tensor byte count")
    return size


@dataclass(frozen=True)
class IndexSpace:
    """An explicitly sized orbital, auxiliary, batch, or spin population.

    ``name`` distinguishes, for example, active occupied and frozen-core
    populations. Optional alpha/beta labels remain part of space identity.
    """

    name: str
    kind: str
    size: int
    spin: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("space name must be an identifier")
        if self.kind not in SPACE_KINDS:
            raise ValueError(f"unsupported index space kind: {self.kind}")
        checked_size(self.size, "space size")
        if self.spin not in (None, "alpha", "beta"):
            raise ValueError("spin must be alpha, beta, or unspecified")


@dataclass(frozen=True)
class Index:
    """One named axis with a range and optional ordered gather coordinates."""

    name: str
    space: IndexSpace
    start: int = 0
    stop: int | None = None
    selection: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("index name must be an identifier")
        if not isinstance(self.space, IndexSpace):
            raise TypeError("index requires an IndexSpace")
        if self.stop is None:
            object.__setattr__(self, "stop", self.space.size)
        checked_size(self.start, "range start")
        checked_size(self.stop, "range stop")
        if not self.start <= self.stop <= self.space.size:
            raise ValueError("index range must lie inside its space")
        if self.selection is not None:
            object.__setattr__(self, "selection", tuple(self.selection))
            if any(
                type(i) is not int or not self.start <= i < self.stop
                for i in self.selection
            ):
                raise ValueError("gather coordinates must lie inside the index range")

    @property
    def extent(self) -> int:
        return (
            len(self.selection)
            if self.selection is not None
            else self.stop - self.start
        )

    @property
    def domain(self) -> tuple:
        """Compare axis meanings independently of dummy-index spelling."""
        return self.space, self.start, self.stop, self.selection

    def coordinate(self, local: int) -> int:
        """Map a validated local coordinate back to its mathematical space."""
        return (
            self.selection[local] if self.selection is not None else self.start + local
        )


@dataclass(frozen=True, order=True)
class Symmetry:
    """Declare ``T = sign * transpose(T, permutation)``.

    Simultaneous pair exchange for restricted spatial t2 is (1,0,3,2), +1.
    Separate occupied/virtual antisymmetries must be declared separately for
    spin-orbital amplitudes; they are never inferred from a rank-four shape.
    """

    permutation: tuple[int, ...]
    sign: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "permutation", tuple(self.permutation))
        if any(type(i) is not int for i in self.permutation) or sorted(
            self.permutation
        ) != list(range(len(self.permutation))):
            raise ValueError("symmetry must be a permutation")
        if type(self.sign) is not int or self.sign not in (-1, 1):
            raise ValueError("symmetry sign must be +1 or -1")


@dataclass(frozen=True)
class TensorSpec:
    """Logical shape/order, real dtype, symmetry, and parameter/AD contracts.

    The IR is in SSA form: inputs are read-only and each primitive defines a
    new value. Differentiability declares a future AD boundary, not a supplied
    JVP/VJP implementation. Layout packing is a separate explicit mapping.
    """

    indices: tuple[Index, ...] = ()
    dtype: str = "float64"
    symmetries: tuple[Symmetry, ...] = ()
    representation: str = "general"
    role: str = "intermediate"
    differentiable: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        if any(not isinstance(i, Index) for i in self.indices):
            raise TypeError("tensor axes must be Index objects")
        if len({i.name for i in self.indices}) != len(self.indices):
            raise ValueError("tensor axis names must be unique; use einsum for traces")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("only real float32/float64 tensors are supported")
        if self.representation not in ("general", "restricted_spatial", "spin_orbital"):
            raise ValueError("unsupported orbital representation")
        if self.role not in ("input", "constant", "parameter", "intermediate"):
            raise ValueError("unsupported tensor parameter role")
        if type(self.differentiable) is not bool:
            raise ValueError("differentiable must be a Boolean")
        if self.role == "constant" and self.differentiable:
            raise ValueError("constants cannot be differentiable")
        symmetries = tuple(sorted(set(self.symmetries)))
        object.__setattr__(self, "symmetries", symmetries)
        for symmetry in symmetries:
            if not isinstance(symmetry, Symmetry):
                raise TypeError("symmetries must be Symmetry objects")
            if len(symmetry.permutation) != len(self.indices) or any(
                self.indices[i].domain != self.indices[j].domain
                for i, j in enumerate(symmetry.permutation)
            ):
                raise ValueError("symmetry cannot exchange different index domains")
        checked_shape(self.shape, self.itemsize)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(i.extent for i in self.indices)

    @property
    def itemsize(self) -> int:
        return 8 if self.dtype == "float64" else 4

    @property
    def size(self) -> int:
        return prod(self.shape)

    @property
    def signature(self) -> tuple:
        """Layout semantics for safe rewrites, excluding names and input roles."""
        return (
            tuple(i.domain for i in self.indices),
            self.dtype,
            self.symmetries,
            self.representation,
        )

    def result(self, *, indices=None, symmetries=(), differentiable=None):
        """Construct a result type without inventing unproved symmetries."""
        return replace(
            self,
            indices=self.indices if indices is None else tuple(indices),
            symmetries=tuple(symmetries),
            role="intermediate",
            differentiable=self.differentiable
            if differentiable is None
            else differentiable,
        )


def spec_to_payload(spec: TensorSpec, *, logical: bool = False) -> dict:
    """Serialize types; logical identities omit purely notational axis names."""
    return {
        "indices": [
            {
                **({} if logical else {"name": index.name}),
                "space": {
                    "name": index.space.name,
                    "kind": index.space.kind,
                    "size": index.space.size,
                    "spin": index.space.spin,
                },
                "start": index.start,
                "stop": index.stop,
                "selection": index.selection,
            }
            for index in spec.indices
        ],
        "dtype": spec.dtype,
        "symmetries": [
            {"permutation": list(s.permutation), "sign": s.sign}
            for s in spec.symmetries
        ],
        "representation": spec.representation,
        "role": spec.role,
        "differentiable": spec.differentiable,
    }


def spec_from_payload(payload: dict) -> TensorSpec:
    """Reconstruct and revalidate a complete versioned tensor type."""
    if not isinstance(payload, dict) or set(payload) != {
        "indices",
        "dtype",
        "symmetries",
        "representation",
        "role",
        "differentiable",
    }:
        raise ValueError("invalid tensor spec fields")
    for item in payload["indices"]:
        if not isinstance(item, dict) or set(item) != {
            "name",
            "space",
            "start",
            "stop",
            "selection",
        }:
            raise ValueError("invalid tensor index fields")
    return TensorSpec(
        indices=tuple(
            Index(
                item["name"],
                IndexSpace(**item["space"]),
                item["start"],
                item["stop"],
                item["selection"],
            )
            for item in payload["indices"]
        ),
        dtype=payload["dtype"],
        symmetries=tuple(Symmetry(**s) for s in payload["symmetries"]),
        representation=payload["representation"],
        role=payload["role"],
        differentiable=payload["differentiable"],
    )
