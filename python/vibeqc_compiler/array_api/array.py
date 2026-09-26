"""Symbolic array values that retain TensorIR scientific semantics."""

from __future__ import annotations

import typing
from dataclasses import dataclass
from fractions import Fraction

from vibeqc_compiler.tensor.ir import Node

ExactScalar: typing.TypeAlias = int | str | Fraction


@dataclass(frozen=True, eq=False)
class VibeArray:
    """One symbolic Array-API value backed by an ordinary TensorIR node."""

    node: Node

    def __post_init__(self) -> None:
        if not isinstance(self.node, Node):
            raise TypeError("VibeArray requires a TensorIR Node")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.node.spec.shape

    @property
    def dtype(self) -> str:
        return self.node.spec.dtype

    @property
    def ndim(self) -> int:
        return len(self.node.spec.indices)

    def __bool__(self) -> bool:
        raise TypeError("symbolic VibeArray values cannot drive Python control flow")

    def __eq__(self, other: object) -> bool:
        raise TypeError("symbolic VibeArray comparisons are not supported")

    def __ne__(self, other: object) -> bool:
        raise TypeError("symbolic VibeArray comparisons are not supported")

    def __add__(self, other: object) -> VibeArray:
        from . import namespace

        return namespace.add(self, other)

    def __radd__(self, other: object) -> VibeArray:
        from . import namespace

        return namespace.add(other, self)

    def __sub__(self, other: object) -> VibeArray:
        from . import namespace

        return namespace.subtract(self, other)

    def __rsub__(self, other: object) -> VibeArray:
        from . import namespace

        return namespace.subtract(other, self)

    def __mul__(self, other: object) -> VibeArray:
        from . import namespace

        return namespace.multiply(self, other)

    def __rmul__(self, other: object) -> VibeArray:
        from . import namespace

        return namespace.multiply(other, self)

    def __truediv__(self, other: object) -> VibeArray:
        from . import namespace

        return namespace.divide(self, other)

    def __rtruediv__(self, other: object) -> VibeArray:
        from . import namespace

        return namespace.divide(other, self)

    def __pow__(self, exponent: object) -> VibeArray:
        from . import namespace

        return namespace.pow(self, exponent)

    def __neg__(self) -> VibeArray:
        from . import namespace

        return namespace.negative(self)

    def __pos__(self) -> VibeArray:
        return self

    def __getitem__(self, key: object) -> VibeArray:
        from . import namespace

        return namespace._getitem(self, key)
