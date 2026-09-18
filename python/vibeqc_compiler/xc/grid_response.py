"""Generated directional response of the existing unpruned Becke grid.

This is a tiled CPU diagnostic/consumer building block for #163 B, not a
complete KS gradient or a native force capability. Scalar derivatives use the
common Graph. Pair/product reductions retain O(point_tile * atom) storage;
no coordinate-by-grid Jacobian or SCF iteration tape is constructed.
"""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256

import numpy as np

from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.common.arrays import immutable
from vibeqc_compiler.common.provenance import canonical_hash
from vibeqc_compiler.dft.grid import GridTile, MolecularGrid, checked_int
from vibeqc_compiler.integral.expr import Graph


@dataclass(frozen=True)
class GridResponseProgram:
    """Shared scalar primal/JVP roots, independent of runtime or molecule size."""

    graph: object
    roots: tuple
    identity: str

    def evaluate(self, **variables):
        return evaluate_array_graph(self.graph, self.roots, variables)


@lru_cache(maxsize=8, typed=True)
def grid_response_program(kind, iterations=3):
    """Generate local mathematics once, not one graph per nuclear coordinate."""
    checked_int(iterations, "partition iterations", high=5)
    graph = Graph()
    if kind == "norm":
        names = ("x", "y", "z")
        xyz = [graph.variable(name) for name in names]
        primal = graph.power(graph.sum(v * v for v in xyz), 0.5)
    elif kind == "ratio":
        names = ("a", "b")
        primal = graph.variable("a") / graph.variable("b")
    elif kind == "log":
        names = ("p",)
        primal = graph.stable_unary("log", graph.variable("p"))
    elif kind == "becke":
        names = ("mu",)
        mu = graph.variable("mu")
        for _ in range(iterations):
            mu = 0.5 * mu * (3 - mu * mu)
        primal = 0.5 * (1 - mu)
    else:
        raise ValueError("unknown grid response primitive")
    tangent = graph.differentiate(
        primal,
        graph.variable("direction"),
        {name: graph.variable(f"d{name}") for name in names},
    )
    roots = (primal, tangent)
    reachable = graph.topological_order(roots)
    indices = {node: i for i, node in enumerate(reachable)}
    identity = canonical_hash(
        {
            "schema": "vibeqc.grid-response-program/v1",
            "kind": kind,
            "nodes": [
                (
                    graph.nodes[i].operation,
                    [indices[j] for j in graph.nodes[i].arguments],
                    str(graph.nodes[i].payload),
                )
                for i in reachable
            ],
            "roots": [indices[root.identifier] for root in roots],
        }
    )
    return GridResponseProgram(graph, roots, identity)


def _norm(delta, motion):
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


@dataclass(frozen=True, eq=False)
class PartitionResponse:
    """Normalized ownership and one directional derivative, both [point,atom]."""

    weights: np.ndarray
    directional: np.ndarray
    branch_identity: str


def partition_response(
    points,
    centers,
    *,
    point_motion,
    center_motion,
    iterations=3,
    coincident_tolerance=1e-12,
):
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
class GridResponseTile:
    """Moved quadrature and its directional response with explicit source stamps."""

    grid: GridTile
    point_motion: np.ndarray
    weight_motion: np.ndarray
    grid_identity: str
    direction_identity: str
    branch_identity: str


def grid_response_tiles(grid, center_motion, *, tile_points=256):
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
