"""Bounded public-basis shell contractions for generated partial Hessian tiles.

Primitive coefficients follow the native radial normalization convention. The
shared first/second weight pullback applies angular normalization exactly once.
External cotangents and public transforms stay fixed during differentiation.
"""

import json
import math
from dataclasses import dataclass
from itertools import product

import numpy as np

from vibeqc_compiler.common.resources import (
    ResourceCandidate,
    ResourceEstimate,
    ResourceIdentity,
    ResourceRequest,
    checked_bytes,
)

from .blocks import WeightTile
from .second_derivatives_execute import SecondPrimitive, pack_second_primitive
from .shell_signature import BasisConvention, ShellSignature
from .shell_spec import cartesian_components
from .weight_pullback import normalized_cartesian_components, pullback_public_weights


@dataclass(frozen=True)
class SecondShellStream:
    """Frozen public pullback with streamed primitive products and numeric bounds."""

    program_identity: str
    primitives: tuple
    centers: tuple
    weights: tuple
    direction: tuple | None
    component_scale: float
    output_tile: int
    host_peak_bytes: int
    public_conventions: tuple

    @property
    def record_count(self):
        return math.prod(map(len, self.primitives))

    @property
    def resource_request(self):
        """Reserve the adapter's conservative peak alongside the native owner."""
        return ResourceRequest(
            "second_public_stream",
            ResourceIdentity(
                "integrals",
                "second_public_pullback",
                "cpu",
                "fp64",
                json.dumps({"program": self.program_identity}),
                ("fixed_public_weight",),
                "streamed_primitive_products",
            ),
            (
                ResourceCandidate(
                    "shared_public_pullback",
                    "streamed",
                    (
                        ResourceEstimate(
                            "public_stream",
                            self.host_peak_bytes,
                            "pageable",
                            1,
                            1,
                            "persistent",
                        ),
                    ),
                ),
            ),
            ("Python object metadata",),
        )

    def __iter__(self):
        """Yield one packed-component primitive at a time, without product storage."""
        for primitives in product(*self.primitives):
            scale = self.component_scale * math.prod(c for _, c in primitives)
            if not math.isfinite(scale):
                raise ValueError(
                    "second derivative primitive contraction coefficient overflow"
                )
            yield SecondPrimitive(
                tuple(a for a, _ in primitives),
                self.centers,
                self.weights,
                self.direction,
                scale,
                self.output_tile,
            )


def prepare_second_shell_stream(
    artifact,
    primitives,
    centers,
    weights=None,
    *,
    public_signature=None,
    projections=None,
    direction=None,
    output_tile=0,
    budget_bytes=4 << 20,
):
    """Freeze a full public weight tile and its normalized primitive contraction.

    Weighted consumers accept a common WeightTile, including padded layouts.
    An explicit public signature and Cartesian-to-public matrices enable real
    spherical weights. Every nonzero pulled-back component must be compiled;
    incomplete coverage fails before execution. Raw consumers select their one
    compiled Cartesian component. Public raw diagnostics can use a weighted
    Hessian consumer with a unit public cotangent.

    Operator and weight-descriptor signs/prefactors belong to the generated
    program. This adapter supplies only the public pullback, angular factors
    and primitive radial coefficients; it never repeats those generated signs.
    """
    artifact.validate()
    checked_bytes(budget_bytes, "second public stream budget")
    checked_bytes(output_tile, "second public output tile")
    if output_tile >= 2**32:
        raise ValueError("second public output tile exceeds uint32")
    integral = artifact.integral
    signature = integral.signature if public_signature is None else public_signature
    if not isinstance(signature, ShellSignature):
        raise TypeError("public signature must be a ShellSignature")
    if signature.angular != integral.signature.angular or tuple(
        s.center for s in signature.shells
    ) != tuple(s.center for s in integral.signature.shells):
        raise ValueError(
            "public signature must preserve compiled shell angular momenta and centers"
        )
    # A public AO transform changes representation, never basis-space roles,
    # the external-center inventory or an already-fixed physical atom binding.
    declared = integral.signature
    if (
        tuple(s.role for s in signature.shells)
        != tuple(s.role for s in declared.shells)
        or tuple(b.center for b in signature.center_bindings)
        != integral.operator.centers
        or any(
            original.atom_index is not None and original.atom_index != public.atom_index
            for original, public in zip(
                declared.center_bindings, signature.center_bindings, strict=True
            )
        )
    ):
        raise ValueError(
            "public signature must preserve compiled roles and center bindings"
        )
    count, centers_count = len(signature.shells), len(integral.operator.centers)
    if (
        len(primitives) != count
        or any(not shell for shell in primitives)
        or len(centers) != centers_count
        or any(len(center) != 3 for center in centers)
    ):
        raise ValueError(
            "second public stream primitive and center dimensions mismatch"
        )
    shape = signature.component_shape
    cart_shape = integral.signature.component_shape
    raw = integral.contractions[0].weights is None
    if raw and (weights is not None or projections is not None or shape != cart_shape):
        raise ValueError(
            "raw public diagnostics require a Cartesian component or a weighted unit cotangent"
        )
    if not raw and (
        not isinstance(weights, WeightTile)
        or weights.layout.shape != shape
        or weights.layout.indices != signature.tensor_indices
    ):
        raise ValueError(
            "public second derivative weights require a full shell WeightTile"
        )
    if (
        any(s.convention != BasisConvention.CARTESIAN for s in signature.shells)
        and projections is None
    ):
        raise ValueError(
            "public spherical weights require explicit Cartesian pullback matrices"
        )
    if projections is not None and (
        len(projections) != count
        or any(
            np.shape(matrix) != (ncart, npublic)
            for matrix, ncart, npublic in zip(
                projections, cart_shape, shape, strict=True
            )
        )
    ):
        raise ValueError("public projection shape or values are invalid")
    # Count all numeric copies and the full angular inventory before allocating
    # a public tensor. The primitive Cartesian product itself remains streamed.
    fixed = (
        (0 if raw else weights.layout.storage_bytes)
        + 16 * sum(map(len, primitives))
        + 8 * (3 * centers_count + 12)
    )
    fixed += artifact.record_format.size + (3 * count + 2) * 8 * len(
        artifact.component_indices
    )
    if projections is not None:
        fixed += 8 * sum(a * b for a, b in zip(cart_shape, shape, strict=True))
    current_shape, current = list(shape), 8 * math.prod(shape)
    peak = fixed + current
    if projections is not None:
        for axis, ncart in enumerate(cart_shape):
            current_shape[axis] = ncart
            next_bytes = 8 * math.prod(current_shape)
            peak = max(peak, fixed + 2 * current + next_bytes)
            current = next_bytes
    peak = checked_bytes(
        max(peak, fixed + current + (3 * count + 1) * 8 * math.prod(cart_shape)),
        "second public stream peak",
    )
    if peak > budget_bytes:
        raise ValueError("second public stream exceeds its declared numeric budget")
    if np.iscomplexobj(centers) or any(np.iscomplexobj(shell) for shell in primitives):
        raise ValueError("public primitive data must be real")
    primitives = tuple(
        tuple((float(a), float(c)) for a, c in shell) for shell in primitives
    )
    centers = tuple(tuple(float(x) for x in center) for center in centers)
    if any(not math.isfinite(x) for center in centers for x in center) or any(
        a <= 0 or not math.isfinite(a) or not math.isfinite(c)
        for shell in primitives
        for a, c in shell
    ):
        raise ValueError(
            "public primitive exponents must be positive and all inputs finite"
        )
    public_weights = np.zeros(shape)
    if raw:
        public_weights.flat[artifact.component_indices[0]] = 1
    else:
        if np.iscomplexobj(weights.values):
            raise ValueError("public component weights must be real")
        for i, offset in enumerate(weights.layout.offsets()):
            public_weights.flat[i] = weights.values[offset]
    public_weights = pullback_public_weights(public_weights, projections)
    components = normalized_cartesian_components(signature.angular, public_weights)
    labels = tuple(cartesian_components(l) for l in signature.angular)
    # Cartesian components are ordered by the established shell inventory.
    selected = {}
    for packed, index in enumerate(artifact.component_indices):
        coordinate = np.unravel_index(index, cart_shape)
        powers = tuple(
            labels[slot][component].count(axis)
            for slot, component in enumerate(coordinate)
            for axis in "xyz"
        )
        selected[powers] = packed
    packed_weights = [0.0] * len(artifact.component_indices)
    for powers, weight in components:
        if powers not in selected:
            raise ValueError(
                "public second derivative pullback exceeds its compiled Cartesian subset"
            )
        packed_weights[selected[powers]] = weight
    scale = packed_weights[0] if raw else 1.0
    if raw:
        packed_weights[0] = 1.0
    direction = (
        None if direction is None else tuple(tuple(x for x in row) for row in direction)
    )
    stream = SecondShellStream(
        artifact.program_identity,
        primitives,
        centers,
        tuple(packed_weights),
        direction,
        scale,
        output_tile,
        peak,
        tuple(s.convention.value for s in signature.shells),
    )
    # Reuse the actual record boundary for direction and raw-unit validation,
    # without visiting or allocating the Cartesian primitive product.
    probe = SecondPrimitive(
        tuple(shell[0][0] for shell in primitives),
        centers,
        stream.weights,
        direction,
        scale,
        output_tile,
    )
    pack_second_primitive(artifact, probe)
    return stream
