"""Generated directional response of the existing unpruned Becke grid.

This is a tiled CPU diagnostic/consumer building block for #163 B and #180,
not a complete KS gradient/HVP or a native derivative capability. Scalar first
and mixed-second derivatives use the common Graph. Pair/product reductions
retain O(point_tile * atom) storage; no coordinate-by-grid Jacobian, partition
Hessian tensor or SCF iteration tape is constructed.
"""

import typing
from dataclasses import dataclass
from hashlib import sha256

import numpy as np

from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft.grid import GridTile, MolecularGrid, checked_int

from .grid_response_ir import (
    GridResponseProgram,  # noqa: F401 -- public compatibility export
    grid_mixed_response_program,
    grid_response_program,
)


def _norm(delta: typing.Any, motion: typing.Any) -> typing.Any:
    # Homogeneity permits a frozen common scale in primal and JVP. This avoids
    # squaring huge/tiny unscaled coordinates without inventing a distance floor.
    scale = np.max(np.abs(delta), axis=-1)
    if np.any(scale == 0):
        raise ValueError("point/center collision is outside the smooth response branch")
    values = {}
    for k, name in enumerate(("x", "y", "z")):
        values[name] = delta[..., k] / scale
        values[f"d{name}"] = motion[..., k] / scale
    value, tangent = grid_response_program("norm").evaluate(**values)
    return value * scale, tangent * scale


def _norm_mixed(
    delta: typing.Any,
    left_motion: typing.Any,
    right_motion: typing.Any,
    mixed_motion: typing.Any,
) -> typing.Any:
    """Evaluate a norm and its two first/mixed directional derivatives."""
    scale = np.max(np.abs(delta), axis=-1)
    if np.any(scale == 0):
        raise ValueError("point/center collision is outside the smooth response branch")
    values = {}
    for k, name in enumerate(("x", "y", "z")):
        values[name] = delta[..., k] / scale
        values[f"l{name}"] = left_motion[..., k] / scale
        values[f"r{name}"] = right_motion[..., k] / scale
        values[f"lr{name}"] = mixed_motion[..., k] / scale
    result = grid_mixed_response_program("norm").evaluate(**values)
    return tuple(value * scale for value in result)


@dataclass(frozen=True, eq=False)
class PartitionResponse:
    """Normalized ownership and one directional derivative, both [point,atom]."""

    weights: np.ndarray
    directional: np.ndarray
    branch_identity: str


def partition_response(
    points: typing.Any,
    centers: typing.Any,
    *,
    point_motion: typing.Any,
    center_motion: typing.Any,
    iterations: typing.Any = 3,
    coincident_tolerance: typing.Any = 1e-12,
) -> typing.Any:
    """Differentiate point AND every partition-center dependence exactly once.

    This first smooth-branch consumer rejects coincident atom centers (including
    the tolerance boundary) and point/center collisions. The value-only grid's
    equal-split coincidence rule remains unchanged. Clipped pair endpoints and
    exact-zero products are handled without division by zero; their branch
    identity must be checked separately in displaced-grid validation.
    """
    checked_int(iterations, "partition iterations", high=5)
    if not np.isfinite(coincident_tolerance) or coincident_tolerance < 0:
        raise ValueError("invalid coincident-center tolerance")
    points, centers = immutable(points), immutable(centers)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or centers.ndim != 2
        or centers.shape[1:] != (3,)
        or not len(centers)
    ):
        raise ValueError("points/centers require (n,3) and at least one center")
    dp = immutable(point_motion, shape=points.shape)
    dc = immutable(center_motion, shape=centers.shape)
    npnt, natom = len(points), len(centers)
    branch = sha256(
        canonical_hash(
            {
                "schema": "vibeqc.becke-response-branch/v1",
                "shape": (npnt, natom),
                "iterations": iterations,
                "coincident_tolerance": coincident_tolerance,
                "pair_program": grid_response_program("becke", iterations).identity,
            }
        ).encode()
    )
    if natom == 1 or npnt == 0:
        return PartitionResponse(
            immutable(np.ones((npnt, natom))),
            immutable(np.zeros((npnt, natom))),
            branch.hexdigest(),
        )
    distances, tangents = np.empty((2, npnt, natom))
    for a in range(natom):
        distances[:, a], tangents[:, a] = _norm(points - centers[a], dp - dc[a])

    # Log products retain the existing value prescription's dynamic range.
    # Keep exact-zero factors separately: d(product) with ONE zero factor is
    # its tangent times the product of the remaining factors, not 0*(dp/p).
    logs = np.zeros((npnt, natom))
    rates = np.zeros_like(logs)
    zeros = np.zeros((npnt, natom), dtype=np.int64)
    zero_tangents = np.zeros_like(logs)
    ratio = grid_response_program("ratio")
    pair_program = grid_response_program("becke", iterations)
    log_program = grid_response_program("log")
    for a in range(natom):
        for b in range(a):
            separation, ds = _norm(centers[a] - centers[b], dc[a] - dc[b])
            if separation <= coincident_tolerance:
                raise ValueError(
                    "coincident centers are outside the smooth response branch"
                )
            mu, dmu = ratio.evaluate(
                a=distances[:, a] - distances[:, b],
                b=separation,
                da=tangents[:, a] - tangents[:, b],
                db=ds,
            )
            clipped = np.abs(mu) >= 1
            branch.update(np.asarray(np.sign(mu) * clipped, dtype=np.int8).tobytes())
            pair, derivative = pair_program.evaluate(
                mu=np.clip(mu, -1, 1), dmu=np.where(clipped, 0, dmu)
            )
            outside = (pair < 0) | (pair > 1)
            branch.update(outside.tobytes())
            derivative = np.where(outside, 0, derivative)
            pair = np.clip(pair, 0, 1)
            branch.update(
                np.asarray((pair == 0) + 2 * (pair == 1), dtype=np.int8).tobytes()
            )
            for atom, value, tangent in (
                (a, pair, derivative),
                (b, 1 - pair, -derivative),
            ):
                active = value > 0
                log_value, log_tangent = log_program.evaluate(
                    p=value[active], dp=tangent[active]
                )
                logs[active, atom] += log_value
                rates[active, atom] += log_tangent
                zeros[~active, atom] += 1
                zero_tangents[~active, atom] += tangent[~active]

    live = zeros == 0
    maximum = np.max(np.where(live, logs, -np.inf), axis=1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ArithmeticError("invalid Becke partition normalization")
    shifted = logs - maximum
    products, dproducts = np.zeros_like(logs), np.zeros_like(logs)
    # Exponential underflow is the same zero-product limit as the value path.
    with np.errstate(under="ignore"):
        products[live] = np.exp(shifted[live])
        dproducts[live] = products[live] * rates[live]
        single = (zeros == 1) & (zero_tangents != 0)
        dproducts[single] = np.exp(shifted[single]) * zero_tangents[single]
    weights, derivative = ratio.evaluate(
        a=products,
        b=products.sum(axis=1, keepdims=True),
        da=dproducts,
        db=dproducts.sum(axis=1, keepdims=True),
    )
    return PartitionResponse(
        immutable(weights), immutable(derivative), branch.hexdigest()
    )


@dataclass(frozen=True, eq=False)
class PartitionMixedResponse:
    """Normalized ownership, two JVPs and their mixed derivative."""

    weights: np.ndarray
    left: np.ndarray
    right: np.ndarray
    mixed: np.ndarray
    branch_identity: str


def partition_mixed_response(
    points: typing.Any,
    centers: typing.Any,
    *,
    left_point_motion: typing.Any,
    left_center_motion: typing.Any,
    right_point_motion: typing.Any,
    right_center_motion: typing.Any,
    iterations: typing.Any = 3,
    coincident_tolerance: typing.Any = 1e-12,
) -> typing.Any:
    """Apply the smooth-branch Becke partition Hessian to two directions.

    This is the mixed derivative d_right(d_left w) at fixed direction vectors.
    Scalar primitive second derivatives are generated from the same Graph roots
    as the first response. Product/normalization algebra retains exact-zero
    factors explicitly, including the one- and two-zero mixed derivative limits.
    """
    checked_int(iterations, "partition iterations", high=5)
    if not np.isfinite(coincident_tolerance) or coincident_tolerance < 0:
        raise ValueError("invalid coincident-center tolerance")
    points, centers = immutable(points), immutable(centers)
    if (
        points.ndim != 2
        or points.shape[1:] != (3,)
        or centers.ndim != 2
        or centers.shape[1:] != (3,)
        or not len(centers)
    ):
        raise ValueError("points/centers require (n,3) and at least one center")
    lp = immutable(left_point_motion, shape=points.shape)
    lc = immutable(left_center_motion, shape=centers.shape)
    rp = immutable(right_point_motion, shape=points.shape)
    rc = immutable(right_center_motion, shape=centers.shape)
    npnt, natom = len(points), len(centers)
    branch = sha256(
        canonical_hash(
            {
                "schema": "vibeqc.becke-response-branch/v1",
                "shape": (npnt, natom),
                "iterations": iterations,
                "coincident_tolerance": coincident_tolerance,
                "pair_program": grid_response_program("becke", iterations).identity,
            }
        ).encode()
    )
    if natom == 1 or npnt == 0:
        shape = (npnt, natom)
        return PartitionMixedResponse(
            immutable(np.ones(shape)),
            immutable(np.zeros(shape)),
            immutable(np.zeros(shape)),
            immutable(np.zeros(shape)),
            branch.hexdigest(),
        )

    distances = np.empty((npnt, natom))
    left_distances = np.empty_like(distances)
    right_distances = np.empty_like(distances)
    mixed_distances = np.empty_like(distances)
    zero_points = np.zeros_like(points)
    for a in range(natom):
        (
            distances[:, a],
            left_distances[:, a],
            right_distances[:, a],
            mixed_distances[:, a],
        ) = _norm_mixed(
            points - centers[a],
            lp - lc[a],
            rp - rc[a],
            zero_points,
        )

    logs = np.zeros((npnt, natom))
    left_rates = np.zeros_like(logs)
    right_rates = np.zeros_like(logs)
    mixed_rates = np.zeros_like(logs)
    zeros = np.zeros((npnt, natom), dtype=np.int64)
    zero_left = np.zeros_like(logs)
    zero_right = np.zeros_like(logs)
    zero_mixed = np.zeros_like(logs)
    zero_left_right_same = np.zeros_like(logs)
    ratio = grid_mixed_response_program("ratio")
    pair_program = grid_mixed_response_program("becke", iterations)
    log_program = grid_mixed_response_program("log")
    zero_center = np.zeros(3)

    for a in range(natom):
        for b in range(a):
            separation, ls, rs, lrs = _norm_mixed(
                centers[a] - centers[b],
                lc[a] - lc[b],
                rc[a] - rc[b],
                zero_center,
            )
            if separation <= coincident_tolerance:
                raise ValueError(
                    "coincident centers are outside the smooth response branch"
                )
            mu, lmu, rmu, lrmu = ratio.evaluate(
                a=distances[:, a] - distances[:, b],
                b=separation,
                la=left_distances[:, a] - left_distances[:, b],
                lb=ls,
                ra=right_distances[:, a] - right_distances[:, b],
                rb=rs,
                lra=mixed_distances[:, a] - mixed_distances[:, b],
                lrb=lrs,
            )
            clipped = np.abs(mu) >= 1
            branch.update(np.asarray(np.sign(mu) * clipped, dtype=np.int8).tobytes())
            pair, lpair, rpair, lrpair = pair_program.evaluate(
                mu=np.clip(mu, -1, 1),
                lmu=np.where(clipped, 0, lmu),
                rmu=np.where(clipped, 0, rmu),
                lrmu=np.where(clipped, 0, lrmu),
            )
            outside = (pair < 0) | (pair > 1)
            branch.update(outside.tobytes())
            lpair = np.where(outside, 0, lpair)
            rpair = np.where(outside, 0, rpair)
            lrpair = np.where(outside, 0, lrpair)
            pair = np.clip(pair, 0, 1)
            branch.update(
                np.asarray((pair == 0) + 2 * (pair == 1), dtype=np.int8).tobytes()
            )
            for atom, value, left, right, mixed in (
                (a, pair, lpair, rpair, lrpair),
                (b, 1 - pair, -lpair, -rpair, -lrpair),
            ):
                active = value > 0
                log_value, log_left, log_right, log_mixed = log_program.evaluate(
                    p=value[active],
                    lp=left[active],
                    rp=right[active],
                    lrp=mixed[active],
                )
                logs[active, atom] += log_value
                left_rates[active, atom] += log_left
                right_rates[active, atom] += log_right
                mixed_rates[active, atom] += log_mixed
                inactive = ~active
                zeros[inactive, atom] += 1
                zero_left[inactive, atom] += left[inactive]
                zero_right[inactive, atom] += right[inactive]
                zero_mixed[inactive, atom] += mixed[inactive]
                zero_left_right_same[inactive, atom] += left[inactive] * right[inactive]

    live = zeros == 0
    maximum = np.max(np.where(live, logs, -np.inf), axis=1, keepdims=True)
    if not np.isfinite(maximum).all():
        raise ArithmeticError("invalid Becke partition normalization")
    shifted = logs - maximum
    products = np.zeros_like(logs)
    left_products = np.zeros_like(logs)
    right_products = np.zeros_like(logs)
    mixed_products = np.zeros_like(logs)
    with np.errstate(under="ignore"):
        products[live] = np.exp(shifted[live])
        left_products[live] = products[live] * left_rates[live]
        right_products[live] = products[live] * right_rates[live]
        mixed_products[live] = products[live] * (
            mixed_rates[live] + left_rates[live] * right_rates[live]
        )

        single = zeros == 1
        q_single = np.exp(shifted[single])
        left_products[single] = q_single * zero_left[single]
        right_products[single] = q_single * zero_right[single]
        mixed_products[single] = q_single * (
            zero_mixed[single]
            + zero_left[single] * right_rates[single]
            + zero_right[single] * left_rates[single]
        )

        double = zeros == 2
        q_double = np.exp(shifted[double])
        mixed_products[double] = q_double * (
            zero_left[double] * zero_right[double] - zero_left_right_same[double]
        )

    weights, left, right, mixed = ratio.evaluate(
        a=products,
        b=products.sum(axis=1, keepdims=True),
        la=left_products,
        lb=left_products.sum(axis=1, keepdims=True),
        ra=right_products,
        rb=right_products.sum(axis=1, keepdims=True),
        lra=mixed_products,
        lrb=mixed_products.sum(axis=1, keepdims=True),
    )
    return PartitionMixedResponse(
        immutable(weights),
        immutable(left),
        immutable(right),
        immutable(mixed),
        branch.hexdigest(),
    )


@dataclass(frozen=True, eq=False)
class GridResponseTile:
    """Moved quadrature and its directional response with explicit source stamps."""

    grid: GridTile
    point_motion: np.ndarray
    weight_motion: np.ndarray
    grid_identity: str
    direction_identity: str
    branch_identity: str


def grid_response_tiles(
    grid: typing.Any, center_motion: typing.Any, *, tile_points: typing.Any = 256
) -> typing.Any:
    """Stream owner-point motion and complete partition-weight motion in Bohr.

    Element radial scales and topology are fixed. The output can contract with
    #163-A's separate point/weight partials; it is not a force or SCF-state proof.
    """
    if not isinstance(grid, MolecularGrid):
        raise TypeError("grid response requires MolecularGrid")
    checked_int(tile_points, "tile points")
    motion = immutable(center_motion, shape=grid.centers.shape)
    direction_identity = canonical_hash(
        {"grid": grid.identity, "center_motion": motion.tolist()}
    )
    for raw in grid._raw_tiles(tile_points):
        owner = np.asarray(raw.owners)
        point_motion = immutable(motion[owner])
        response = partition_response(
            raw.points,
            grid.centers,
            point_motion=point_motion,
            center_motion=motion,
            iterations=grid.spec.partition_iterations,
            coincident_tolerance=grid.spec.coincident_tolerance,
        )
        selected = (np.arange(len(owner)), owner)
        yield GridResponseTile(
            GridTile(
                raw.begin,
                raw.points,
                immutable(raw.weights * response.weights[selected]),
                raw.owners,
            ),
            point_motion,
            immutable(raw.weights * response.directional[selected]),
            grid.identity,
            direction_identity,
            response.branch_identity,
        )


@dataclass(frozen=True, eq=False)
class GridMixedResponseTile:
    """Moved quadrature with two first and one mixed weight response."""

    grid: GridTile
    left_point_motion: np.ndarray
    right_point_motion: np.ndarray
    left_weight_motion: np.ndarray
    right_weight_motion: np.ndarray
    mixed_weight_motion: np.ndarray
    grid_identity: str
    left_direction_identity: str
    right_direction_identity: str
    branch_identity: str


def grid_mixed_response_tiles(
    grid: typing.Any,
    left_center_motion: typing.Any,
    right_center_motion: typing.Any,
    *,
    tile_points: typing.Any = 256,
) -> typing.Any:
    """Stream first/mixed molecular-grid responses for a Hessian bilinear.

    Raw atom-centred point positions are affine in their owner center, so their
    mixed motion is exactly zero. The nontrivial second geometric term is the
    generated Becke partition-weight response.
    """
    if not isinstance(grid, MolecularGrid):
        raise TypeError("grid mixed response requires MolecularGrid")
    checked_int(tile_points, "tile points")
    left = immutable(left_center_motion, shape=grid.centers.shape)
    right = immutable(right_center_motion, shape=grid.centers.shape)
    left_identity = canonical_hash(
        {"grid": grid.identity, "center_motion": left.tolist()}
    )
    right_identity = canonical_hash(
        {"grid": grid.identity, "center_motion": right.tolist()}
    )
    for raw in grid._raw_tiles(tile_points):
        owner = np.asarray(raw.owners)
        left_point = immutable(left[owner])
        right_point = immutable(right[owner])
        response = partition_mixed_response(
            raw.points,
            grid.centers,
            left_point_motion=left_point,
            left_center_motion=left,
            right_point_motion=right_point,
            right_center_motion=right,
            iterations=grid.spec.partition_iterations,
            coincident_tolerance=grid.spec.coincident_tolerance,
        )
        selected = (np.arange(len(owner)), owner)
        yield GridMixedResponseTile(
            GridTile(
                raw.begin,
                raw.points,
                immutable(raw.weights * response.weights[selected]),
                raw.owners,
            ),
            left_point,
            right_point,
            immutable(raw.weights * response.left[selected]),
            immutable(raw.weights * response.right[selected]),
            immutable(raw.weights * response.mixed[selected]),
            grid.identity,
            left_identity,
            right_identity,
            response.branch_identity,
        )
