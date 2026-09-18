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


def normalized_radial_primitives(angular, primitives):
    """Match the native contracted-shell radial normalization exactly once.

    Cartesian angular double-factorial factors remain the public-weight
    adapter's responsibility. No coordinates, electronic state or engine is
    needed to normalize fixed basis coefficients.
    """
    primitives = tuple(tuple(row) for row in primitives)
    if type(angular) is not int or not 0 <= angular <= 4 or not primitives:
        raise ValueError("a nonempty s/p/d/f/g primitive shell is required")
    if any(
        len(row) != 2 or not all(math.isfinite(x) for x in row) or row[0] <= 0
        for row in primitives
    ):
        raise ValueError("primitive exponents must be positive and coefficients finite")
    norm2 = sum(
        ca * cb * (2 * math.sqrt(a * b) / (a + b)) ** (angular + 1.5)
        for a, ca in primitives
        for b, cb in primitives
    )
    if not math.isfinite(norm2) or norm2 <= 0:
        raise ValueError("contracted shell has invalid normalization")
    result = tuple(
        (
            a,
            c
            * (2 * a / math.pi) ** 0.75
            * (4 * a) ** (0.5 * angular)
            / math.sqrt(norm2),
        )
        for a, c in primitives
    )
    if any(not math.isfinite(c) for _, c in result):
        raise ValueError("normalized primitive coefficient overflow")
    return result
