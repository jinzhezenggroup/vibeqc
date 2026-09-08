"""Conservative binary-einsum contracts for FP64 cuBLAS lowering.

This module describes matrix coordinates, not allocation or execution. A
contraction requiring a one-sided reduction, repeated input labels, or more
than two operands keeps the general TensorIR kernel. No shape-based orbital
or spin symmetry is inferred when grouping labels into matrix dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from math import prod

from .ir import Node
from .types import checked_size


def fp64_coefficient(pair) -> float:
    """Round an exact rational once, as in the independent CPU interpreter."""
    try:
        value = float(Fraction(*pair))
    except OverflowError as error:
        raise ValueError("tensor coefficient is outside finite FP64") from error
    if not math.isfinite(value):
        raise ValueError("tensor coefficient is outside finite FP64")
    return value


@dataclass(frozen=True)
class GemmContract:
    """Logical C[batch,M,N] = alpha * A[batch,M,K] B[batch,K,N].

    Each capitalized group may contain multiple original index labels. Empty
    groups have extent one, allowing outer products, dot products, and scalar
    operands without inventing axes in TensorIR. The lowerer must pack/scatter
    when physical strides do not implement these orders directly.
    """

    a_labels: tuple[int, ...]
    b_labels: tuple[int, ...]
    output_labels: tuple[int, ...]
    batch_labels: tuple[int, ...]
    m_labels: tuple[int, ...]
    n_labels: tuple[int, ...]
    k_labels: tuple[int, ...]
    extents: tuple[int, ...]
    coefficient: float

    def extent(self, labels) -> int:
        """Flatten only declared groups, checking integer products eagerly."""
        return checked_size(prod(self.extents[i] for i in labels), "GEMM dimension")

    @property
    def batch(self) -> int:
        return self.extent(self.batch_labels)

    @property
    def m(self) -> int:
        return self.extent(self.m_labels)

    @property
    def n(self) -> int:
        return self.extent(self.n_labels)

    @property
    def k(self) -> int:
        return self.extent(self.k_labels)

    @property
    def a_order(self) -> tuple[int, ...]:
        return self.batch_labels + self.m_labels + self.k_labels

    @property
    def b_order(self) -> tuple[int, ...]:
        return self.batch_labels + self.k_labels + self.n_labels

    @property
    def c_order(self) -> tuple[int, ...]:
        return self.batch_labels + self.m_labels + self.n_labels

    @property
    def flops(self) -> int:
        """Operation-count estimate, not a claim about the selected algorithm."""
        return 2 * self.batch * self.m * self.n * self.k

    def panel_bytes(self, tile_m: int, tile_n: int, tile_k: int) -> int:
        """One batch's A/B/C packing panels, including partial-tail capacity.

        The caller reuses these panels sequentially across batches and output
        tiles. Concurrent batches or double buffering require separate storage.
        Empty output or reduction domains need no GEMM packing workspace.
        """
        for value in (tile_m, tile_n, tile_k):
            checked_size(value, "GEMM tile extent")
            if value == 0:
                raise ValueError("GEMM tile extents must be positive")
        if not self.batch or not self.m or not self.n or not self.k:
            return 0
        m, n, k = min(tile_m, self.m), min(tile_n, self.n), min(tile_k, self.k)
        return checked_size(8 * (m * k + k * n + m * n), "GEMM panel bytes")

    def matrix_coordinates(self, batch: int, row: int, column: int, reduction: int):
        """Reference coordinate map for independently checking pack/scatter code.

        CUDA code emits the corresponding integer maps; this host helper is
        used only by planning tests and never performs runtime contractions.
        """
        coordinates = {}
        for value, labels in (
            (batch, self.batch_labels),
            (row, self.m_labels),
            (column, self.n_labels),
            (reduction, self.k_labels),
        ):
            if type(value) is not int or not 0 <= value < self.extent(labels):
                raise ValueError("matrix coordinate is outside its label group")
            for label in reversed(labels):
                coordinates[label] = value % self.extents[label]
                value //= self.extents[label]
        return tuple(
            tuple(coordinates[i] for i in labels)
            for labels in (self.a_labels, self.b_labels, self.output_labels)
        )


def gemm_contract(node: Node) -> GemmContract | None:
    """Recognize exactly representable binary contractions without emitting code.

    A label appearing in only one operand must survive in the output. This
    deliberately rejects one-sided reductions instead of silently dropping
    their summation. Diagonals/traces and arbitrary n-ary einsums retain the
    general generated path, whose mathematical domain is broader than GEMM.
    """
    if node.op != "einsum" or len(node.inputs) != 2:
        return None
    if node.spec.dtype != "float64":
        return None
    a, b = node.attrs["labels"]
    output = node.attrs["output"]
    if len(set(a)) != len(a) or len(set(b)) != len(b):
        return None
    a_set, b_set, out = set(a), set(b), set(output)
    if (a_set ^ b_set) - out:
        return None
    common = a_set & b_set
    batch = tuple(i for i in output if i in common)
    m = tuple(i for i in output if i in a_set - b_set)
    n = tuple(i for i in output if i in b_set - a_set)
    k = tuple(i for i in a if i in common - out)
    extents = {}
    for operand, labels in zip(node.inputs, (a, b), strict=True):
        extents.update(zip(labels, operand.spec.shape, strict=True))
    result = GemmContract(
        a,
        b,
        output,
        batch,
        m,
        n,
        k,
        tuple(extents[i] for i in range(len(extents))),
        fp64_coefficient(node.attrs["coefficient"]),
    )
    # Check every collapsed dimension even if a later schedule tiles it.
    for labels in (batch, m, n, k):
        result.extent(labels)
    return result
