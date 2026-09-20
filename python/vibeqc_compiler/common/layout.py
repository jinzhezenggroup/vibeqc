"""Backend-neutral dense physical layouts, separate from logical TensorSpec.

Orders name logical axes from slowest to fastest. These layouts own no storage
and imply neither tensor symmetry nor alias permission. A transpose can describe
the same storage through another logical view; allocation/lifetime ownership is
still the execution plan's responsibility. Arbitrary affine strides and padding
are deliberately outside this first contract.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass
from math import prod

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

    def to_payload(self) -> dict[str, object]:
        return {
            "schema": "vibeqc.tensor.dense-layout.v1",
            "shape": self.shape,
            "order": self.order,
            "element_strides": self.element_strides,
            "alignment": self.alignment,
            "c_contiguous": self.is_c_contiguous,
        }
