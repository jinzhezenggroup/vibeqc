"""Shared fixed public-weight pullbacks and Cartesian component normalization.

Consumers preflight numeric buffers before calling these arithmetic helpers.
The same ordered operations serve first and second derivatives, with neither
HF density factors nor implicit permutation multiplicities.
"""

import math

import numpy as np

from .shell_spec import cartesian_components


def pullback_public_weights(weights, projections):
    """Pull a shell-local public cotangent through fixed per-slot AO transforms."""
    if projections is not None:
        if any(np.iscomplexobj(p) for p in projections):
            raise ValueError("public projection matrices must be real")
        matrices = tuple(np.asarray(p, dtype=float) for p in projections)
        for axis, matrix in enumerate(matrices):
            if not np.isfinite(matrix).all():
                raise ValueError("public projection shape or values are invalid")
            weights = np.moveaxis(
                np.tensordot(matrix, weights, axes=(1, axis)), 0, axis
            )
    return weights


def normalized_cartesian_components(angular, weights, scale=1.0):
    """Apply angular normalization once to weights for radially normalized inputs.

    The returned sparse inventory retains original shell-slot xyz powers.
    Primitive radial coefficients are multiplied later while streaming their
    Cartesian product, so no primitive-product tensor is materialized here.
    """
    components = []
    labels = tuple(cartesian_components(order) for order in angular)
    if weights.shape != tuple(map(len, labels)) or np.iscomplexobj(weights):
        raise ValueError("Cartesian weight shape or scalar type is invalid")
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
    return tuple(components)
