"""Bounded FP64 shell tiles and external-weight consumer contracts.

These host helpers assemble synthetic/reference data; they do not evaluate
integrals or imply a CUDA implementation. Input derivatives are independent
center blocks in dense ``(xyz, shell_0, ...)`` order. Recovery precedes atom
accumulation. Layout strides are in elements, and budgets cover numeric input,
output, weights, and recovery storage, excluding Python object overhead and
provider-owned caches. No molecular N**4 tensor is constructed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import product
from math import fsum, isfinite, prod
from typing import TYPE_CHECKING, Protocol

from .shell_signature import CenterBinding, checked_index

if TYPE_CHECKING:
    from .ir import IntegralIR


def _sign(value: int, name: str) -> None:
    if type(value) is not int or value not in (-1, 1):
        raise ValueError(f"{name} must be +1 or -1")


@dataclass(frozen=True, slots=True)
class TensorLayout:
    """Named FP64 axes with explicit positive, nonoverlapping element strides.

    Dense permutations and padded layouts are supported. Broadcast, negative,
    and interleaved overlapping strides are rejected at this boundary.
    """

    indices: tuple[str, ...]
    shape: tuple[int, ...]
    strides: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "indices", tuple(self.indices))
        object.__setattr__(self, "shape", tuple(self.shape))
        if not self.indices or len(set(self.indices)) != len(self.indices):
            raise ValueError("tensor indices must be non-empty and unique")
        if any(not isinstance(i, str) or not i for i in self.indices):
            raise ValueError("tensor indices must be non-empty strings")
        if len(self.indices) != len(self.shape):
            raise ValueError("tensor index and shape ranks differ")
        for extent in self.shape:
            checked_index(extent, "tensor extent", minimum=1)
        checked_index(prod(self.shape), "tensor element count", minimum=1)
        strides = self.strides
        if strides is None:
            strides = tuple(prod(self.shape[i + 1 :]) for i in range(len(self.shape)))
        object.__setattr__(self, "strides", tuple(strides))
        if len(strides) != len(self.shape):
            raise ValueError("tensor stride and shape ranks differ")
        for stride in strides:
            checked_index(stride, "tensor stride", minimum=1)
        span = 1
        for stride, extent in sorted(
            (s, n) for s, n in zip(strides, self.shape) if n > 1
        ):
            if stride < span:
                raise ValueError("tensor strides overlap or interleave axes")
            span += stride * (extent - 1)
        checked_index(self.storage_bytes, "tensor storage bytes", minimum=1)

    @property
    def storage_elements(self) -> int:
        """Include padding in the required allocation size."""
        return 1 + sum((n - 1) * s for n, s in zip(self.shape, self.strides))

    @property
    def storage_bytes(self) -> int:
        """Return the checked FP64 allocation size."""
        return 8 * self.storage_elements

    def offsets(self):
        """Iterate logical row-major coordinates through this physical layout."""
        for coordinate in product(*(range(n) for n in self.shape)):
            yield sum(i * s for i, s in zip(coordinate, self.strides))

    def to_payload(self) -> dict[str, object]:
        """Expose ordering, extents, element strides, and fixed scalar type."""
        return {
            "indices": list(self.indices),
            "shape": list(self.shape),
            "strides": list(self.strides),
            "dtype": "float64",
        }


@dataclass(frozen=True, slots=True)
class RawBlock:
    """Request one value or derivative tile in the declared output layout."""

    layout: TensorLayout
    memory_budget_bytes: int
    output_sign: int = 1

    def __post_init__(self) -> None:
        checked_index(self.memory_budget_bytes, "memory budget", minimum=1)
        _sign(self.output_sign, "output sign")

    @property
    def consumer(self) -> str:
        return "raw_block"

    @property
    def kernel_consumer(self) -> None:
        """Raw tensors have no legacy Fock/force registry category."""
        return None


@dataclass(frozen=True, slots=True)
class WeightDescriptor:
    """Identify a tiled external weight provider without requiring HF density.

    The contraction is ``output_sign * sign * prefactor * sum(W * dI)``.
    ``source`` is a caller-resolved key, never serialized executable code.
    """

    source: str
    layout: TensorLayout
    sign: int = 1
    prefactor: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("weight source must be a non-empty provider key")
        _sign(self.sign, "weight sign")
        if isinstance(self.prefactor, bool) or not isfinite(self.prefactor):
            raise ValueError("weight prefactor must be finite")
        object.__setattr__(self, "prefactor", float(self.prefactor))


@dataclass(frozen=True, slots=True)
class WeightedDerivative:
    """Contract arbitrary scalar weights with first nuclear derivatives.

    Nuclear derivative rows follow requested center order. Atomic force rows
    use sorted distinct physical atom indices; force sign is explicit rather
    than inferred from the output name. Rows are local to this shell tile.
    """

    weights: WeightDescriptor
    output_layout: TensorLayout
    memory_budget_bytes: int
    output: str = "nuclear_derivative"
    output_sign: int = 1

    def __post_init__(self) -> None:
        from .ir import ContractionOutput

        object.__setattr__(self, "output", ContractionOutput(self.output))
        if self.output not in (
            ContractionOutput.NUCLEAR_DERIVATIVE,
            ContractionOutput.ATOMIC_FORCE,
        ):
            raise ValueError(
                "weighted derivative requires nuclear_derivative or atomic_force output"
            )
        checked_index(self.memory_budget_bytes, "memory budget", minimum=1)
        _sign(self.output_sign, "output sign")

    @property
    def consumer(self) -> str:
        return "weighted_derivative"

    @property
    def kernel_consumer(self) -> None:
        """External weights do not imply a direct-HF contraction."""
        return None


@dataclass(frozen=True, slots=True)
class ShellTile:
    """AO component offsets and extents within one shell tuple."""

    offsets: tuple[int, ...]
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "offsets", tuple(self.offsets))
        object.__setattr__(self, "shape", tuple(self.shape))
        if len(self.offsets) != len(self.shape) or len(self.shape) not in (2, 3, 4):
            raise ValueError(
                "shell tile requires matching two-, three-, or four-slot extents"
            )
        for offset, extent in zip(self.offsets, self.shape):
            checked_index(offset, "tile offset")
            checked_index(extent, "tile extent", minimum=1)
            checked_index(offset + extent, "tile end", minimum=1)
        checked_index(prod(self.shape), "tile element count", minimum=1)


@dataclass(frozen=True, slots=True)
class BlockRequest:
    """A bounded request with complete center/atom binding and tensor metadata.

    ``center_bindings`` resolves legacy task-time bindings. Already fixed atom
    identities cannot be overridden. No allocation happens during validation.
    ``shell_indices`` identifies runtime shells in each slot's orbital or
    auxiliary space; synthetic data providers may leave these indices unset.
    """

    request_id: str
    integral: IntegralIR
    tile: ShellTile
    consumer_index: int = 0
    center_bindings: tuple[CenterBinding, ...] | None = None
    shell_indices: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request_id, str) or not self.request_id:
            raise ValueError("block request requires a non-empty request id")
        checked_index(self.consumer_index, "consumer index")
        if self.consumer_index >= len(self.integral.contractions):
            raise ValueError("consumer index is outside the integral request")
        if not isinstance(self.consumer, (RawBlock, WeightedDerivative)):
            raise TypeError("bounded blocks require raw_block or weighted_derivative")
        if self.integral.derivative is not None and self.integral.derivative.order != 1:
            raise ValueError(
                "bounded block interfaces expose only order-one derivatives"
            )
        full_shape = self.integral.signature.component_shape
        if self.shell_indices is not None:
            object.__setattr__(self, "shell_indices", tuple(self.shell_indices))
            if len(self.shell_indices) != len(full_shape):
                raise ValueError(
                    "runtime shell indices must match the basis slot count"
                )
            for index in self.shell_indices:
                checked_index(index, "runtime shell index")
        if len(full_shape) != len(self.tile.shape) or any(
            offset + extent > full
            for offset, extent, full in zip(
                self.tile.offsets, self.tile.shape, full_shape
            )
        ):
            raise ValueError("tile exceeds shell bounds")
        layout = (
            self.consumer.layout
            if isinstance(self.consumer, RawBlock)
            else self.consumer.weights.layout
        )
        if layout.shape[-len(full_shape) :] != self.tile.shape:
            raise ValueError("tile shape does not match consumer tensor layout")
        declared = self.integral.signature.center_bindings
        bindings = (
            declared if self.center_bindings is None else tuple(self.center_bindings)
        )
        if tuple(b.center for b in bindings) != self.integral.operator.centers:
            raise ValueError(
                "block center bindings must follow the complete operator center inventory"
            )
        if any(b.atom_index is None for b in bindings):
            raise ValueError("block requests must resolve every physical atom index")
        fixed = {b.center: b.atom_index for b in declared}
        if any(
            fixed[b.center] is not None and fixed[b.center] != b.atom_index
            for b in bindings
        ):
            raise ValueError("block bindings cannot override fixed physical atoms")
        object.__setattr__(self, "center_bindings", bindings)
        if isinstance(self.consumer, WeightedDerivative):
            rows = (
                len(self.atom_indices)
                if self.consumer.output == "atomic_force"
                else len(self.integral.requested_derivative_centers)
            )
            if self.consumer.output_layout.shape != (rows, 3):
                raise ValueError(
                    "weighted output layout does not match center/atom rows"
                )
        if self.required_bytes > self.consumer.memory_budget_bytes:
            raise ValueError(
                f"block requires {self.required_bytes} bytes, exceeding memory budget {self.consumer.memory_budget_bytes}"
            )

    @property
    def consumer(self) -> RawBlock | WeightedDerivative:
        return self.integral.contractions[self.consumer_index]

    @property
    def output_layout(self) -> TensorLayout:
        return (
            self.consumer.layout
            if isinstance(self.consumer, RawBlock)
            else self.consumer.output_layout
        )

    @property
    def atom_indices(self) -> tuple[int, ...]:
        """Local atomic-output rows for the selected derivative parameters."""
        selected = (
            self.integral.requested_derivative_centers or self.integral.operator.centers
        )
        return tuple(
            sorted({b.atom_index for b in self.center_bindings if b.center in selected})
        )

    @property
    def required_bytes(self) -> int:
        """Conservatively account for FP64 buffers, including recovery workspace."""
        elements = prod(self.tile.shape)
        if self.integral.derivative is None:
            input_bytes = elements * 8
        else:
            # Input independent blocks and a complete reconstructed center
            # workspace coexist. Count both even if a provider can alias them.
            centers = len(self.integral.independent_derivative_centers) + len(
                self.integral.requested_derivative_centers
            )
            input_bytes = centers * 3 * elements * 8
        weight_bytes = (
            self.consumer.weights.layout.storage_bytes
            if isinstance(self.consumer, WeightedDerivative)
            else 0
        )
        return checked_index(
            input_bytes + weight_bytes + self.output_layout.storage_bytes,
            "block storage bytes",
            minimum=1,
        )

    def to_payload(self) -> dict[str, object]:
        """Serialize the bounded request; provider callbacks are never included."""
        from .ir_serialization import integral_to_payload

        return {
            "request_id": self.request_id,
            "integral": integral_to_payload(self.integral),
            "consumer_index": self.consumer_index,
            "tile": {
                "offsets": list(self.tile.offsets),
                "shape": list(self.tile.shape),
            },
            "center_bindings": [
                {"center": b.center, "atom_index": b.atom_index}
                for b in self.center_bindings
            ],
            "required_bytes": self.required_bytes,
            "shell_indices": None
            if self.shell_indices is None
            else list(self.shell_indices),
        }


class BlockStatus(str, Enum):
    """Unsupported execution is distinguishable from a successful empty block."""

    OK = "ok"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True, slots=True)
class BlockResponse:
    """Result metadata plus a single FP64 shell tile, including padding."""

    request: BlockRequest
    status: BlockStatus
    values: tuple[float, ...] = ()
    reason: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", BlockStatus(self.status))
        object.__setattr__(self, "values", tuple(self.values))
        if self.status == BlockStatus.OK:
            _validate_buffer(self.values, self.request.output_layout.storage_elements)
        elif self.values or not self.reason:
            raise ValueError("unsupported responses require a reason and no data")

    @property
    def center_atoms(self) -> tuple[tuple[int, int], ...]:
        return tuple((b.center, b.atom_index) for b in self.request.center_bindings)

    @property
    def atom_indices(self) -> tuple[int, ...]:
        return self.request.atom_indices

    def to_payload(self) -> dict[str, object]:
        """Include exact request identity, row maps, strides, and support status."""
        from .cache import integral_cache_key

        return {
            "schema": "vibeqc.integral_block_response",
            "schema_version": 1,
            "request_id": self.request.request_id,
            "integral_key": integral_cache_key(self.request.integral),
            "consumer_index": self.request.consumer_index,
            "shell_indices": None
            if self.request.shell_indices is None
            else list(self.request.shell_indices),
            "status": self.status.value,
            "reason": self.reason,
            "tile": {
                "offsets": list(self.request.tile.offsets),
                "shape": list(self.request.tile.shape),
            },
            "layout": self.request.output_layout.to_payload(),
            "center_atoms": [list(row) for row in self.center_atoms],
            "atom_indices": list(self.atom_indices),
            "derivative_centers": list(
                self.request.integral.requested_derivative_centers
            ),
            "values": list(self.values),
        }


@dataclass(frozen=True, slots=True)
class WeightTile:
    """Caller-owned weight buffer for precisely the requested tile layout."""

    layout: TensorLayout
    values: Sequence[float]

    def __post_init__(self) -> None:
        _validate_buffer(self.values, self.layout.storage_elements)


class WeightTileProvider(Protocol):
    """Callback boundary for CC or other consumers with nonfactorizable weights."""

    def __call__(
        self, descriptor: WeightDescriptor, request: BlockRequest
    ) -> WeightTile: ...


def _validate_buffer(values: Sequence[float], length: int) -> None:
    if len(values) != length:
        raise ValueError(f"buffer has {len(values)} elements; expected {length}")
    if any(not isfinite(value) for value in values):
        raise ValueError("buffer values must be finite")


def _center_data(
    request: BlockRequest, derivatives: Mapping[int, Sequence[float]]
) -> dict[int, Sequence[float]]:
    """Reconstruct position derivatives exactly once before atom accumulation."""
    integral = request.integral
    if set(derivatives) != set(integral.independent_derivative_centers):
        raise ValueError("derivative data must contain exactly the independent centers")
    count = 3 * prod(request.tile.shape)
    for values in derivatives.values():
        _validate_buffer(values, count)
    data = dict(derivatives)
    for recovered in integral.recovered_derivative_centers:
        data[recovered] = tuple(
            -fsum(data[c][i] for c in integral.independent_derivative_centers)
            for i in range(count)
        )
    return data


def _pack(layout: TensorLayout, values) -> tuple[float, ...]:
    result = [0.0] * layout.storage_elements
    for offset, value in zip(layout.offsets(), values, strict=True):
        result[offset] = float(value)
    return tuple(result)


def assemble_raw_block(
    request: BlockRequest, values: Sequence[float] | Mapping[int, Sequence[float]]
) -> BlockResponse:
    """Assemble provided reference values/derivatives into a bounded raw response."""
    if not isinstance(request.consumer, RawBlock):
        raise TypeError("raw assembly requires a raw_block consumer")
    if request.integral.derivative is None:
        if isinstance(values, Mapping):
            raise ValueError("value blocks require a flat buffer")
        _validate_buffer(values, prod(request.tile.shape))
        logical = iter(values)
    else:
        if not isinstance(values, Mapping):
            raise ValueError("derivative blocks require independent center buffers")
        data = _center_data(request, values)
        logical = (
            x for c in request.integral.requested_derivative_centers for x in data[c]
        )
    packed = _pack(
        request.output_layout, (request.consumer.output_sign * x for x in logical)
    )
    return BlockResponse(request, BlockStatus.OK, packed)


def contract_weighted_derivative(
    request: BlockRequest,
    derivatives: Mapping[int, Sequence[float]],
    provider: WeightTileProvider,
) -> BlockResponse:
    """Reference contraction of arbitrary tiled weights, with explicit signs."""
    consumer = request.consumer
    if not isinstance(consumer, WeightedDerivative):
        raise TypeError("external contraction requires a weighted_derivative consumer")
    data = _center_data(request, derivatives)
    tile = provider(consumer.weights, request)
    if tile.layout != consumer.weights.layout:
        raise ValueError("provider weight layout does not match the descriptor")
    _validate_buffer(tile.values, tile.layout.storage_elements)
    count = prod(request.tile.shape)
    scale = consumer.output_sign * consumer.weights.sign * consumer.weights.prefactor
    centers = request.integral.requested_derivative_centers
    center_atoms = {b.center: b.atom_index for b in request.center_bindings}
    rows = request.atom_indices if consumer.output == "atomic_force" else centers

    def results():
        for row in rows:
            selected = (
                tuple(c for c in centers if center_atoms[c] == row)
                if consumer.output == "atomic_force"
                else (row,)
            )
            for xyz in range(3):
                # Accumulate the position derivatives before contraction so
                # coincident shells obey the chain rule for arbitrary weights.
                yield scale * fsum(
                    tile.values[offset]
                    * fsum(data[c][xyz * count + i] for c in selected)
                    for i, offset in enumerate(tile.layout.offsets())
                )

    return BlockResponse(
        request, BlockStatus.OK, _pack(request.output_layout, results())
    )


def unsupported_block_response(
    request: BlockRequest, *, backend: str = "cuda"
) -> BlockResponse:
    """Report unavailable backend execution without synthesizing successful data."""
    from .capabilities import query_integral_capability

    capability = query_integral_capability(request.integral, backend=backend)
    if capability.supported:
        raise ValueError(
            "supported requests must be submitted to their backend executor"
        )
    return BlockResponse(
        request, BlockStatus.UNSUPPORTED, reason="; ".join(capability.reasons)
    )
