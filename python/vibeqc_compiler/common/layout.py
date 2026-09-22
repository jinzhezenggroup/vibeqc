"""Backend-neutral physical layouts, separate from logical TensorSpec.

Dense orders name logical axes from slowest to fastest. Exact symmetric-pair
storage is a separate representation whose legality must be supplied by the
scientific provider. Layouts own no storage or allocation lifetime; arbitrary
affine strides and padding remain outside this contract.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from math import prod

from .provenance import canonical_hash
from .resources import byte_product, checked_bytes


@dataclass(frozen=True)
class DenseLayout:
    """A bijective permutation of a dense logical tensor's physical axes.

    ``alignment`` is a required base-address alignment in bytes, not a promise
    about every element. Axis ordinals, rather than scientific index names, keep
    physical identity independent of equation notation.
    """

    shape: tuple[int, ...]
    order: tuple[int, ...] | None = None
    alignment: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        order = (
            tuple(range(len(self.shape))) if self.order is None else tuple(self.order)
        )
        object.__setattr__(self, "order", order)
        for extent in self.shape:
            checked_bytes(extent, "layout dimension")
        byte_product(*self.shape)
        if any(type(axis) is not int for axis in order) or sorted(order) != list(
            range(len(self.shape))
        ):
            raise ValueError("layout order must be a permutation of logical axes")
        checked_bytes(self.alignment, "layout alignment")
        if not self.alignment or self.alignment & (self.alignment - 1):
            raise ValueError("layout alignment must be a positive power of two")
        for stride in self.element_strides:
            checked_bytes(stride, "layout stride")

    @property
    def element_strides(self) -> tuple[int, ...]:
        result, stride = [0] * len(self.shape), 1
        order = self.order
        assert order is not None
        for axis in reversed(order):
            result[axis] = stride
            stride *= self.shape[axis]
        return tuple(result)

    @property
    def is_c_contiguous(self) -> bool:
        order = self.order
        assert order is not None
        return not prod(self.shape) or tuple(
            axis for axis in order if self.shape[axis] > 1
        ) == tuple(axis for axis, extent in enumerate(self.shape) if extent > 1)

    def equivalent(self, other: object) -> bool:
        """Whether logical coordinates name identical dense element offsets.

        Singleton strides do not affect addressing; empty tensors have no
        addresses. Alignment requirements remain separate from view equivalence.
        """
        if not isinstance(other, DenseLayout) or self.shape != other.shape:
            return False
        return not prod(self.shape) or all(
            extent <= 1 or left == right
            for extent, left, right in zip(
                self.shape, self.element_strides, other.element_strides, strict=True
            )
        )

    def physical_index(self, logical_index: int) -> int:
        """Map a C-order logical flat index to its physical element offset."""
        checked_bytes(logical_index, "logical index")
        if logical_index >= prod(self.shape):
            raise ValueError("logical index is outside the layout")
        return sum(
            (logical_index // prod(self.shape[axis + 1 :]) % extent) * stride
            for axis, (extent, stride) in enumerate(
                zip(self.shape, self.element_strides, strict=True)
            )
        )

    def logical_index(self, physical_index: int) -> int:
        """Inverse of physical_index, useful for independent layout tests."""
        checked_bytes(physical_index, "physical index")
        if physical_index >= prod(self.shape):
            raise ValueError("physical index is outside the layout")
        return sum(
            (physical_index // stride % extent) * prod(self.shape[axis + 1 :])
            for axis, (extent, stride) in enumerate(
                zip(self.shape, self.element_strides, strict=True)
            )
        )

    def transpose(self, axes: typing.Any) -> DenseLayout:
        """Describe a transposed view without changing its physical storage."""
        axes = tuple(axes)
        if any(type(axis) is not int for axis in axes) or sorted(axes) != list(
            range(len(self.shape))
        ):
            raise ValueError("transpose axes must be a permutation")
        order = self.order
        assert order is not None
        return DenseLayout(
            tuple(self.shape[axis] for axis in axes),
            tuple(axes.index(axis) for axis in order),
            self.alignment,
        )

    @property
    def storage_elements(self) -> int:
        return byte_product(*self.shape)

    def storage_bytes(self, itemsize: int) -> int:
        checked_bytes(itemsize, "layout item size")
        return byte_product(self.storage_elements, itemsize)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "vibeqc.tensor.dense-layout.v1",
            "shape": self.shape,
            "order": self.order,
            "element_strides": self.element_strides,
            "alignment": self.alignment,
            "c_contiguous": self.is_c_contiguous,
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())


@dataclass(frozen=True)
class SymmetricPairLayout:
    """Exact lower-triangular storage for two interchangeable logical axes.

    The first two logical axes both have extent and are represented by one
    unit-weight lower-triangle coordinate. Any trailing axes remain dense and
    contiguous. This is a physical representation contract only: a scientific
    provider must prove exchange of the first two coordinates is legal.
    """

    extent: int
    trailing_shape: tuple[int, ...] = ()
    alignment: int = 1

    def __post_init__(self) -> None:
        checked_bytes(self.extent, "symmetric-pair extent")
        trailing = tuple(self.trailing_shape)
        object.__setattr__(self, "trailing_shape", trailing)
        for extent in trailing:
            checked_bytes(extent, "symmetric-pair trailing dimension")
        checked_bytes(self.extent + 1, "symmetric-pair extent successor")
        byte_product(self.pair_count, *trailing)
        byte_product(self.extent, self.extent, *trailing)
        checked_bytes(self.alignment, "layout alignment")
        if not self.alignment or self.alignment & (self.alignment - 1):
            raise ValueError("layout alignment must be a positive power of two")

    @property
    def pair_count(self) -> int:
        successor = self.extent + 1
        return (
            byte_product(self.extent, successor // 2)
            if self.extent % 2
            else byte_product(self.extent // 2, successor)
        )

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return (self.extent, self.extent, *self.trailing_shape)

    @property
    def storage_shape(self) -> tuple[int, ...]:
        return (self.pair_count, *self.trailing_shape)

    @property
    def storage_elements(self) -> int:
        return byte_product(*self.storage_shape)

    @property
    def dense_elements(self) -> int:
        return byte_product(*self.logical_shape)

    def pair_index(self, first: int, second: int) -> int:
        """Map either logical pair order to the canonical lower-triangle row."""
        for coordinate in (first, second):
            checked_bytes(coordinate, "symmetric-pair coordinate")
            if coordinate >= self.extent:
                raise ValueError("symmetric-pair coordinate is outside the layout")
        high, low = max(first, second), min(first, second)
        successor = high + 1
        base = (
            byte_product(high, successor // 2)
            if high % 2
            else byte_product(high // 2, successor)
        )
        return checked_bytes(base + low, "symmetric-pair index")

    def storage_bytes(self, itemsize: int) -> int:
        checked_bytes(itemsize, "layout item size")
        return byte_product(self.storage_elements, itemsize)

    def dense_equivalent_bytes(self, itemsize: int) -> int:
        checked_bytes(itemsize, "layout item size")
        return byte_product(self.dense_elements, itemsize)

    def dense_materialization_bytes(
        self,
        itemsize: int,
        *,
        rows: int | None = None,
        trailing_shape: tuple[int, ...] | None = None,
    ) -> int:
        """Bytes written by one bounded packed-to-dense logical expansion."""
        checked_bytes(itemsize, "layout item size")
        rows = self.extent if rows is None else rows
        checked_bytes(rows, "materialized symmetric-pair rows")
        if rows > self.extent:
            raise ValueError("materialized rows exceed the symmetric-pair extent")
        tile = self.trailing_shape if trailing_shape is None else tuple(trailing_shape)
        if len(tile) != len(self.trailing_shape):
            raise ValueError("materialized trailing rank does not match the layout")
        for requested, available in zip(tile, self.trailing_shape, strict=True):
            checked_bytes(requested, "materialized trailing dimension")
            if requested > available:
                raise ValueError("materialized trailing tile exceeds the layout")
        return byte_product(rows, self.extent, *tile, itemsize)

    def unpack_traffic_bytes(
        self,
        itemsize: int,
        *,
        rows: int | None = None,
        trailing_shape: tuple[int, ...] | None = None,
    ) -> int:
        """Semantic packed reads plus dense writes for the current CUDA unpack."""
        return byte_product(
            2,
            self.dense_materialization_bytes(
                itemsize, rows=rows, trailing_shape=trailing_shape
            ),
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "vibeqc.symmetric-pair-layout.v1",
            "logical_shape": self.logical_shape,
            "storage_shape": self.storage_shape,
            "pair_axes": (0, 1),
            "triangle": "lower",
            "pair_index": "max(i,j)*(max(i,j)+1)/2+min(i,j)",
            "trailing_order": "dense-c",
            "alignment": self.alignment,
            "exact": True,
        }

    @property
    def identity(self) -> str:
        return canonical_hash(self.to_payload())
