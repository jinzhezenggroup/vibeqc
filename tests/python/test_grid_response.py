"""Small independent gates for generated #163-B1 grid directional response."""

import subprocess
import sys
from dataclasses import replace
from decimal import Decimal, localcontext

import numpy as np
import pytest
from vibeqc_compiler.dft.grid import GridSpec, MolecularGrid, partition_weights
from vibeqc_compiler.xc.contractions import GeometryPartials
from vibeqc_compiler.xc.grid_response import (
    grid_response_program,
    grid_response_tiles,
    partition_response,
)

CENTERS = np.array([[0.1, -0.2, 0.3], [1.2, 0.3, -0.1], [-0.5, 1.1, 0.8]])
POINTS = np.array([[0.2, 0.4, -0.3], [0.6, -0.4, 0.7], [-0.2, 0.7, 0.4]])
DC = np.array([[0.2, -0.1, 0.3], [-0.3, 0.4, 0.1], [0.1, 0.2, -0.2]])
DP = np.array([[0.1, -0.3, 0.4], [0.2, 0.1, -0.2], [-0.4, 0.2, 0.1]])


@pytest.mark.parametrize("iterations", [1, 3, 5])
@pytest.mark.parametrize("source", ["point", "center", "both"])
def test_partition_sources_match_independent_multistep_differences(iterations, source):
    dp = DP if source != "center" else np.zeros_like(DP)
    dc = DC if source != "point" else np.zeros_like(DC)
    result = partition_response(
        POINTS, CENTERS, point_motion=dp, center_motion=dc, iterations=iterations
    )
    # This retained value implementation does not use Graph or its derivatives.
    np.testing.assert_allclose(
        result.weights,
        partition_weights(POINTS, CENTERS, iterations=iterations),
        atol=2e-15,
        rtol=2e-14,
    )
    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        fd = (
            partition_weights(POINTS + h * dp, CENTERS + h * dc, iterations=iterations)
            - partition_weights(
                POINTS - h * dp, CENTERS - h * dc, iterations=iterations
            )
        ) / (2 * h)
        errors.append(np.max(abs(result.directional - fd)))
    assert errors[-1] < 3e-9
    assert errors[-1] < errors[0] / 100
    np.testing.assert_allclose(result.directional.sum(axis=1), 0, atol=4e-15)


def decimal_partition(points, centers, iterations):
    """Independent scalar high-precision direct products (not generated/log-DAG)."""

    def distance(a, b):
        return sum((x - y) ** 2 for x, y in zip(a, b, strict=True)).sqrt()

    answer = []
    for point in points:
        products = [Decimal(1) for _ in centers]
        for a in range(len(centers)):
            for b in range(a):
                mu = (
                    distance(point, centers[a]) - distance(point, centers[b])
                ) / distance(centers[a], centers[b])
                for _ in range(iterations):
                    mu = mu * (3 - mu * mu) / 2
                pair = (1 - mu) / 2
                products[a] *= pair
                products[b] *= 1 - pair
        total = sum(products)
        answer.append([p / total for p in products])
    return answer


@pytest.mark.parametrize("iterations", [1, 3, 5])
def test_decimal_oracle_is_independent_of_generated_primal(iterations):
    with localcontext() as context:
        context.prec = 60
        h = Decimal("1e-16")
        arrays = [
            [[Decimal(str(v)) for v in row] for row in a]
            for a in (POINTS, CENTERS, DP, DC)
        ]
        points, centers, dp, dc = arrays

        def displaced(values, motion, sign):
            return [
                [x + sign * h * dx for x, dx in zip(row, direction, strict=True)]
                for row, direction in zip(values, motion, strict=True)
            ]

        plus, minus = [
            decimal_partition(
                displaced(points, dp, s), displaced(centers, dc, s), iterations
            )
            for s in (1, -1)
        ]
        derivative = np.array(
            [
                [float((p - m) / (2 * h)) for p, m in zip(a, b, strict=True)]
                for a, b in zip(plus, minus, strict=True)
            ]
        )
    result = partition_response(
        POINTS, CENTERS, point_motion=DP, center_motion=DC, iterations=iterations
    )
    np.testing.assert_allclose(result.directional, derivative, atol=6e-15, rtol=5e-13)


def test_translation_permutation_and_empty_partition():
    translation = np.array([0.3, -0.7, 0.2])
    rigid = partition_response(
        POINTS,
        CENTERS,
        point_motion=np.broadcast_to(translation, POINTS.shape),
        center_motion=np.broadcast_to(translation, CENTERS.shape),
    )
    np.testing.assert_array_equal(rigid.directional, 0)
    ordinary = partition_response(POINTS, CENTERS, point_motion=DP, center_motion=DC)
    order = [2, 0, 1]
    permuted = partition_response(
        POINTS, CENTERS[order], point_motion=DP, center_motion=DC[order]
    )
    np.testing.assert_allclose(permuted.weights, ordinary.weights[:, order], atol=2e-15)
    np.testing.assert_allclose(
        permuted.directional, ordinary.directional[:, order], atol=3e-15
    )
    empty = partition_response(
        np.empty((0, 3)), CENTERS, point_motion=np.empty((0, 3)), center_motion=DC
    )
    assert empty.weights.shape == empty.directional.shape == (0, 3)
    single = partition_response(
        POINTS, CENTERS[:1], point_motion=DP, center_motion=DC[:1]
    )
    np.testing.assert_array_equal(single.weights, 1)
    np.testing.assert_array_equal(single.directional, 0)


def test_exact_zero_pair_products_do_not_divide_by_zero():
    centers = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    points = np.array([[-1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    response = partition_response(
        points, centers, point_motion=np.zeros_like(points), center_motion=DC
    )
    np.testing.assert_array_equal(response.weights, [[1, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(response.directional, 0)
    assert np.isfinite(response.directional).all()


def grid_at(centers=CENTERS, spec=None):
    if spec is None:
        spec = GridSpec(
            radial_points=4,
            angular_polar=3,
            angular_azimuth=5,
            element_radii=((1, 0.8), (8, 1.2)),
        )
    return MolecularGrid(
        [(z, xyz) for z, xyz in zip((8, 1, 1), centers, strict=True)], spec
    )


def gather(grid, motion, tile_points=17):
    tiles = list(grid_response_tiles(grid, motion, tile_points=tile_points))
    return (
        tiles,
        np.concatenate([t.point_motion for t in tiles]),
        np.concatenate([t.weight_motion for t in tiles]),
    )


def test_owner_motion_weight_motion_and_partial_tile_match_rebuilt_grids():
    grid = grid_at()
    tiles, dx, dw = gather(grid, DC)
    explicit = grid.explicit()
    assert len(tiles[-1].grid.points) < 17
    np.testing.assert_array_equal(dx, DC[np.asarray(explicit.owners)])
    np.testing.assert_array_equal(
        np.concatenate([t.grid.points for t in tiles]), explicit.points
    )
    np.testing.assert_allclose(
        np.concatenate([t.grid.weights for t in tiles]),
        explicit.weights,
        atol=2e-12,
        rtol=2e-13,
    )
    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        plus, minus = [grid_at(CENTERS + s * h * DC).explicit() for s in (1, -1)]
        np.testing.assert_allclose(
            (plus.points - minus.points) / (2 * h), dx, atol=2e-9, rtol=2e-9
        )
        fd = (plus.weights - minus.weights) / (2 * h)
        errors.append(np.max(abs(fd - dw)))
    # Raw far-radial measures are large: first verify O(h^2) convergence,
    # then the final error floor, instead of requiring a coarse difference to
    # already meet the fine-step gate. These are weights, not energy gradients.
    assert errors[1] < errors[0] / 80
    assert errors[-1] < errors[0] / 1000
    np.testing.assert_allclose(fd, dw, atol=3e-6, rtol=3e-7)
    _, dx2, dw2 = gather(grid, DC, tile_points=grid.npoint)
    np.testing.assert_array_equal(dx, dx2)
    np.testing.assert_array_equal(dw, dw2)
    assert all(t.grid_identity == grid.identity for t in tiles)
    assert len({t.direction_identity for t in tiles}) == 1
    assert tiles[0].direction_identity != gather(grid, -DC)[0][0].direction_identity
    with pytest.raises(ValueError):
        tiles[0].weight_motion.flags.writeable = True


def test_point_and_weight_sources_contract_once_with_existing_geometry_partials():
    grid = grid_at()
    _, dx, dw = gather(grid, DC)
    explicit = grid.explicit()
    values = np.exp(-0.4 * np.sum(explicit.points**2, axis=1))
    partials = GeometryPartials(
        centers=np.zeros_like(CENTERS),
        points=-0.8 * explicit.points * (values * explicit.weights)[:, None],
        weights=values,
    )
    derivative = partials.directional(centers=DC, points=dx, weights=dw)
    point_term = np.sum(partials.points * dx)
    weight_term = partials.weights @ dw
    assert abs(point_term) > 1e-3 and abs(weight_term) > 1e-3
    errors = []
    for h in (1e-3, 1e-4, 1e-5):
        energies = []
        for sign in (1, -1):
            moved = grid_at(CENTERS + sign * h * DC).explicit()
            energies.append(
                moved.weights @ np.exp(-0.4 * np.sum(moved.points**2, axis=1))
            )
        fd = (energies[0] - energies[1]) / (2 * h)
        errors.append(abs(fd - derivative))
    assert errors[-1] < 2e-8 and errors[-1] < errors[0] / 50
    # Dedicated negative gates: neither omitted source can pass the full derivative.
    assert abs(fd - point_term) > 1e-3
    assert abs(fd - weight_term) > 1e-3
    rigid = np.broadcast_to([0.3, -0.2, 0.1], CENTERS.shape)
    np.testing.assert_array_equal(gather(grid, rigid)[2], 0)


@pytest.mark.parametrize(
    "bad",
    ["coincident", "tolerance", "point_collision", "nan", "complex", "motion_shape"],
)
def test_invalid_or_nonsmooth_inputs_fail_closed(bad):
    centers, points, dc = CENTERS.copy(), POINTS.copy(), DC.copy()
    tolerance = 1e-12
    if bad == "coincident":
        centers[1] = centers[0]
    elif bad == "tolerance":
        tolerance = 10.0
    elif bad == "point_collision":
        points[0] = centers[0]
    elif bad == "nan":
        points[0, 0] = np.nan
    elif bad == "complex":
        dc = dc.astype(complex) + 1j
    else:
        dc = dc[:2]
    with pytest.raises(ValueError):
        partition_response(
            points,
            centers,
            point_motion=DP,
            center_motion=dc,
            coincident_tolerance=tolerance,
        )


def test_branch_program_identity_and_no_runtime_import_during_generation():
    one = grid_response_program("becke", 1)
    three = grid_response_program("becke", 3)
    assert one.identity != three.identity
    grid_response_program.cache_clear()
    assert three.identity == grid_response_program("becke", 3).identity
    normal = partition_response(POINTS, CENTERS, point_motion=DP, center_motion=DC)
    other = partition_response(POINTS, CENTERS, point_motion=-DP, center_motion=-DC)
    assert normal.branch_identity == other.branch_identity
    # A cached integer program must not bypass strict size validation for
    # equal-valued bool/float keys.
    grid_response_program("becke", 1)
    for invalid in (True, 1.0):
        with pytest.raises(ValueError):
            grid_response_program("becke", invalid)
    with pytest.raises(ValueError):
        grid_response_program("not-a-primitive")
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'vibeqc', 'pyscf', 'torch', 'cupy'}:
            raise RuntimeError('generation imported ' + fullname)
sys.meta_path.insert(0, BlockRuntime())
from vibeqc_compiler.xc.grid_response import grid_response_program
for kind in ('norm', 'ratio', 'log', 'becke'):
    assert grid_response_program(kind).identity
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_grid_spec_and_bad_tile_boundaries_are_not_silently_changed():
    grid = grid_at(spec=replace(grid_at().spec, partition_iterations=1))
    _, _, derivative = gather(grid, DC)
    assert derivative.shape == (grid.npoint,)
    with pytest.raises(ValueError):
        list(grid_response_tiles(grid, DC, tile_points=0))
    with pytest.raises(TypeError):
        list(grid_response_tiles(grid.explicit(), DC))
