"""Bounded shell-weight callbacks lowered to native primitive input streams.

No SCF density is constructed. Public weights are pulled back on the host with
the caller's fixed basis transforms, and primitive products are streamed rather
than materialized. Input primitive coefficients use the native basis layer's
radial normalization; Cartesian component factors are applied here once.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from itertools import product

import numpy as np

from .blocks import BlockRequest, BlockResponse, BlockStatus, WeightedDerivative
from .ir import OperatorFamily
from .shell_signature import BasisConvention, checked_index
from .shell_spec import cartesian_components

PRIMITIVE_RECORD = struct.Struct("<II12I4d12d3d")
assert PRIMITIVE_RECORD.size == 208


@dataclass(frozen=True)
class WeightedEriPrimitiveStream:
    """Frozen topology/geometry/weights with bounded primitive-product traversal.

    host_peak_bytes conservatively bounds numeric provider storage, projection
    matrices, pullback intermediates, normalized primitive lists, the full
    angular/weight inventory, and one yielded record.
    Python object overhead and caller-owned caches outside these inputs are
    excluded. A caller reserves its native upload/result budget separately.
    """

    request: BlockRequest
    primitives: tuple
    centers: tuple
    components: tuple
    fused_weights: tuple | None
    output_tile: int
    host_peak_bytes: int
    public_pullback_on_host: bool

    @property
    def record_count(self) -> int:
        count = math.prod(len(shell) for shell in self.primitives)
        return count * (1 if self.fused_weights is not None else len(self.components))

    def records(self):
        """Yield fixed little-endian native POD records without a product array."""
        if not self.components:
            return
        positions = tuple(x for center in self.centers for x in center)
        for quartet in product(*self.primitives):
            exponents = tuple(a for a, _ in quartet)
            coefficient = math.prod(c for _, c in quartet)
            if self.fused_weights is not None:
                weights = tuple(coefficient * w for w in self.fused_weights)
                if not all(math.isfinite(w) for w in weights):
                    raise ValueError("primitive contraction weight overflow")
                yield PRIMITIVE_RECORD.pack(
                    1,
                    self.output_tile,
                    1,
                    *([0] * 11),
                    *exponents,
                    *positions,
                    *weights,
                )
            else:
                for angular, weight in self.components:
                    weight *= coefficient
                    if not math.isfinite(weight):
                        raise ValueError("primitive contraction weight overflow")
                    yield PRIMITIVE_RECORD.pack(
                        0,
                        self.output_tile,
                        *angular,
                        *exponents,
                        *positions,
                        weight,
                        0.0,
                        0.0,
                    )


def prepare_weighted_eri_stream(
    request: BlockRequest,
    primitives,
    centers,
    provider,
    *,
    projections=None,
    generated=True,
    output_tile=0,
) -> WeightedEriPrimitiveStream:
    """Freeze one full/padded/partial weight tile, then stream normalized records.

    projections[slot] maps normalized Cartesian source functions to the
    public functions in that slot, with shape (ncart, npublic). It is required
    for non-Cartesian signatures. Arbitrary weights can be ordered or already
    orbit-folded; this adapter applies neither symmetry factors nor screening.
    """
    integral, consumer = request.integral, request.consumer
    if integral.operator.family != OperatorFamily.FOUR_CENTER_ERI or not isinstance(
        consumer, WeightedDerivative
    ):
        raise ValueError(
            "native weighted input requires an external ERI derivative request"
        )
    if integral.operator.centers != (0, 1, 2, 3) or tuple(
        s.center for s in integral.signature.shells
    ) != (0, 1, 2, 3):
        raise ValueError(
            "native weighted input requires quartet center slots (0, 1, 2, 3)"
        )
    if integral.derivative is None or integral.derivative.order != 1:
        raise ValueError("native weighted input requires first derivatives")
    angular = integral.signature.angular
    if any(l > 3 for l in angular):
        raise ValueError("native external ERI weights support s/p/d/f shells")
    checked_index(output_tile, "output tile")
    if output_tile >= 2**32:
        raise ValueError("native weighted output tile exceeds uint32")
    shape = integral.signature.component_shape
    if (
        len(primitives) != 4
        or any(not shell for shell in primitives)
        or len(centers) != 4
        or any(len(c) != 3 for c in centers)
    ):
        raise ValueError(
            "four nonempty primitive lists and three-dimensional centers are required"
        )
    noncartesian = any(
        s.convention != BasisConvention.CARTESIAN for s in integral.signature.shells
    )
    if noncartesian and projections is None:
        raise ValueError(
            "public spherical weights require explicit Cartesian pullback matrices"
        )
    if projections is not None and len(projections) != 4:
        raise ValueError("four public projection matrices are required")
    cart_shape = tuple(len(cartesian_components(l)) for l in angular)
    matrix_shapes = tuple((ncart, npublic) for ncart, npublic in zip(cart_shape, shape))
    if projections is not None and any(
        np.shape(p) != expected for p, expected in zip(projections, matrix_shapes)
    ):
        raise ValueError("public projection shape or values are invalid")
    # Preflight every owned numeric allocation before copying primitives,
    # invoking the provider, or allocating a full public tile. The maximum
    # component inventory includes twelve angular integers and one FP64 weight
    # per component (all counted as eight-byte scalars), even for sparse inputs.
    fixed_bytes = (
        consumer.weights.layout.storage_bytes
        + 16 * sum(map(len, primitives))
        + 12 * 8
        + PRIMITIVE_RECORD.size
    )
    if projections is not None:
        fixed_bytes += 8 * sum(math.prod(s) for s in matrix_shapes)
    current_shape = list(shape)
    current_bytes = 8 * math.prod(current_shape)
    peak = fixed_bytes + current_bytes
    if projections is not None:
        for axis, ncart in enumerate(cart_shape):
            current_shape[axis] = ncart
            next_bytes = 8 * math.prod(current_shape)
            peak = max(peak, fixed_bytes + 2 * current_bytes + next_bytes)
            current_bytes = next_bytes
    peak = max(peak, fixed_bytes + current_bytes + 104 * math.prod(cart_shape) + 24)
    if peak > consumer.memory_budget_bytes:
        raise ValueError("external weighted stream exceeds its declared numeric budget")
    primitives = tuple(
        tuple((float(a), float(c)) for a, c in shell) for shell in primitives
    )
    centers = tuple(tuple(float(x) for x in center) for center in centers)
    if any(not math.isfinite(x) for c in centers for x in c) or any(
        a <= 0 or not math.isfinite(a) or not math.isfinite(c)
        for shell in primitives
        for a, c in shell
    ):
        raise ValueError("primitive exponents must be positive and all inputs finite")
    tile = provider(consumer.weights, request)
    if (
        tile.layout != consumer.weights.layout
        or len(tile.values) != tile.layout.storage_elements
    ):
        raise ValueError("external provider must return the declared weight layout")
    weights = np.zeros(shape)
    for coordinate, offset in zip(
        product(*(range(n) for n in request.tile.shape)),
        tile.layout.offsets(),
        strict=True,
    ):
        logical = tuple(
            begin + index for begin, index in zip(request.tile.offsets, coordinate)
        )
        value = float(tile.values[offset])
        if not math.isfinite(value):
            raise ValueError("external ERI weights must be finite")
        weights[logical] = value
    if projections is not None:
        matrices = tuple(np.asarray(p, dtype=float) for p in projections)
        for axis, matrix in enumerate(matrices):
            if not np.isfinite(matrix).all():
                raise ValueError("public projection shape or values are invalid")
            weights = np.moveaxis(
                np.tensordot(matrix, weights, axes=(1, axis)), 0, axis
            )
    scale = consumer.output_sign * consumer.weights.sign * consumer.weights.prefactor
    components = []
    labels = tuple(cartesian_components(l) for l in angular)
    for coordinate in np.ndindex(weights.shape):
        quantums = tuple(
            labels[axis][component].count(x)
            for axis, component in enumerate(coordinate)
            for x in "xyz"
        )
        norm = math.prod(math.prod(range(1, 2 * power, 2)) for power in quantums)
        weight = float(weights[coordinate]) * scale / math.sqrt(norm)
        if not math.isfinite(weight):
            raise ValueError("normalized external weight overflow")
        if weight != 0.0:
            components.append((quantums, weight))
    fused_weights = None
    if generated and angular == (1, 0, 0, 0) and components:
        fused = [0.0] * 3
        for quantums, weight in components:
            fused[quantums[:3].index(1)] = weight
        fused_weights = tuple(fused)
    return WeightedEriPrimitiveStream(
        request,
        primitives,
        centers,
        tuple(components),
        fused_weights,
        output_tile,
        peak,
        projections is not None,
    )


def weighted_eri_response(request: BlockRequest, native_result) -> BlockResponse:
    """Map the native four-center result to requested center or physical-atom rows."""
    if (
        request.integral.operator.centers != (0, 1, 2, 3)
        or tuple(s.center for s in request.integral.signature.shells) != (0, 1, 2, 3)
    ) or not isinstance(request.consumer, WeightedDerivative):
        raise ValueError(
            "native weighted response requires an external quartet request"
        )
    values = np.asarray(native_result, dtype=float)
    if values.shape != (13,) or not np.isfinite(values).all():
        raise ValueError(
            "native weighted response must contain thirteen finite scalars"
        )
    gradient = values[1:].reshape(4, 3)
    centers = request.integral.requested_derivative_centers
    if request.consumer.output == "atomic_force":
        selected = np.zeros((len(request.atom_indices), 3))
        for center in centers:
            atom = request.center_bindings[center].atom_index
            selected[request.atom_indices.index(atom)] += gradient[center]
    else:
        selected = gradient[list(centers)]
    output = np.zeros(request.output_layout.storage_elements)
    for offset, value in zip(
        request.output_layout.offsets(), selected.flat, strict=True
    ):
        output[offset] = value
    return BlockResponse(request, BlockStatus.OK, tuple(output))
