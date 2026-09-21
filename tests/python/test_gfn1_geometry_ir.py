"""GFN1 geometry compiler qualification against independent pinned oracles."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.geometry import (
    GFN1_CUTOFF_BOHR,
    GFN1_SHORT_RANGE_PARAMETER_IDENTITY,
    Gfn1ShortRangeProgram,
    build_gfn1_pair_topology,
    build_gfn1_short_range_program,
    gfn1_element_parameters,
    gfn1_geometry,
)
from vibeqc_compiler.geometry._gfn1_data import (
    GFN1_GEOMETRY_ELEMENT_ROWS,
    GFN1_MINIMUM_DISTANCE_SQUARED_BOHR2,
    GFN1_PARAMETER_JSON_SHA256,
)
from vibeqc_compiler.tensor import execute

ROOT = Path(__file__).resolve().parents[2]

# mstore MB16-43 structure 01. The CN vector was independently evaluated in
# 80-digit decimal arithmetic from pinned mctc-lib GFN1 radii/equations. The
# repulsion energy is the pinned tblite GFN1 effective-repulsion fixture.
MINDLESS01_ATOMIC_NUMBERS = (
    11,
    1,
    8,
    1,
    9,
    1,
    1,
    8,
    7,
    1,
    1,
    17,
    5,
    5,
    7,
    13,
)
MINDLESS01_COORDINATES = np.array(
    [
        -1.85528263484662,
        3.58670515364616,
        -2.41763729306344,
        4.40178023537845,
        0.02338844412653,
        -4.95457749372945,
        -2.98706033463438,
        4.76252065456814,
        1.27043301573532,
        0.79980886075526,
        1.41103455609189,
        -5.04655321620119,
        -4.20647469409936,
        1.84275767548460,
        4.55038084858449,
        -3.54356121843970,
        -3.18835665176557,
        1.46240021785588,
        2.70032160109941,
        1.06818452504054,
        -1.73234650374438,
        3.73114088824361,
        -2.07001543363453,
        2.23160937604731,
        -1.75306819230397,
        0.35951417150421,
        1.05323406177129,
        5.41755788583825,
        -1.57881830078929,
        1.75394002750038,
        -2.23462868255966,
        -2.13856505054269,
        4.10922285746451,
        1.01565866207568,
        -3.21952154552768,
        -3.36050963020778,
        2.42119255723593,
        0.26626435093114,
        -3.91862474360560,
        -3.02526098819107,
        2.53667889095925,
        2.31664984740423,
        -2.00438948664892,
        -2.29235136977220,
        2.19782807357059,
        1.12226554109716,
        -1.36942007032045,
        0.48455055461782,
    ],
    dtype=np.float64,
).reshape(-1, 3)
MINDLESS01_CN = np.array(
    [
        4.1506636895139719,
        0.9788680263897811,
        2.0108098563385948,
        1.4786569782781824,
        1.0357782244211671,
        1.0120699431478062,
        1.5032977712740092,
        1.9985846827260887,
        3.8918192753932410,
        1.0432337336073976,
        1.0152658445063574,
        1.9931521322735404,
        4.6352656088968285,
        3.8731226063933475,
        3.9931680067788431,
        5.4506822690388841,
    ],
    dtype=np.float64,
)
MINDLESS01_REPULSION = 0.16777923624986593


def _compile(
    elements: tuple[int, ...], coordinates: np.ndarray
) -> Gfn1ShortRangeProgram:
    geometry = gfn1_geometry(elements)
    topology = build_gfn1_pair_topology(geometry, coordinates)
    return build_gfn1_short_range_program(geometry, topology)


def test_gfn1_geometry_data_is_generated_from_canonical_snapshot() -> None:
    assert GFN1_PARAMETER_JSON_SHA256 == (
        "0ecdc3f5f12990c5a7e0f0bd7e6fe931ecf72d7630e6a6a3cc396c51766a40a0"
    )
    assert len(GFN1_GEOMETRY_ELEMENT_ROWS) == 86
    assert tuple(row[0] for row in GFN1_GEOMETRY_ELEMENT_ROWS) == tuple(range(1, 87))
    subprocess.run(
        [sys.executable, "tools/parameters/generate_gfn1_geometry.py", "--check"],
        cwd=ROOT,
        check=True,
    )


def test_gfn1_parameter_subset_covers_h_through_rn() -> None:
    for atomic_number in range(1, 87):
        item = gfn1_element_parameters(atomic_number)
        assert item.atomic_number == atomic_number
        assert item.covalent_radius_bohr > 0
        assert item.arep > 0
        assert item.zeff > 0
        assert item.atomic_radius_bohr > 0
        assert item.xbond >= 0
    with pytest.raises(ValueError, match="1..86"):
        gfn1_element_parameters(87)


def test_gfn1_coordination_matches_independent_mindless01_oracle() -> None:
    compiled = _compile(MINDLESS01_ATOMIC_NUMBERS, MINDLESS01_COORDINATES)
    compiled.validate_coordinates(MINDLESS01_COORDINATES)
    actual = execute(
        compiled.program,
        {"coordinates": MINDLESS01_COORDINATES},
    ).outputs["coordination"]
    np.testing.assert_allclose(actual, MINDLESS01_CN, rtol=0, atol=5e-13)
    assert compiled.parameter_identity == GFN1_SHORT_RANGE_PARAMETER_IDENTITY


def test_gfn1_repulsion_matches_pinned_tblite_mindless01_oracle() -> None:
    compiled = _compile(MINDLESS01_ATOMIC_NUMBERS, MINDLESS01_COORDINATES)
    actual = (
        execute(
            compiled.program,
            {"coordinates": MINDLESS01_COORDINATES},
        )
        .outputs["repulsion_energy"]
        .item()
    )
    assert actual == pytest.approx(MINDLESS01_REPULSION, rel=0, abs=1.1e-13)


@pytest.mark.parametrize(
    ("elements", "distance", "expected_cn", "expected_energy", "expected_force"),
    [
        ((1, 1), 1.4, 0.9190366235286143, 0.022893402746661507, 0.10613642871993496),
        ((6, 8), 2.3, 0.9997222452945929, 0.037151145741047484, 0.15162165837073494),
    ],
)
def test_gfn1_independent_pair_goldens_and_generated_repulsion_vjp(
    elements: tuple[int, int],
    distance: float,
    expected_cn: float,
    expected_energy: float,
    expected_force: float,
) -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
    compiled = _compile(elements, coordinates)
    outputs = execute(compiled.program, {"coordinates": coordinates}).outputs
    np.testing.assert_allclose(
        outputs["coordination"], (expected_cn, expected_cn), rtol=0, atol=4e-16
    )
    assert outputs["repulsion_energy"].item() == pytest.approx(
        expected_energy, rel=0, abs=5e-17
    )

    gradient = execute(
        compiled.coordinate_vjp("repulsion_energy").program,
        {
            "coordinates": coordinates,
            "bar_repulsion_energy": np.array(1.0),
        },
    ).outputs["bar_coordinates"]

    # xTBloom's oracle stores forces = -dE/dR.
    assert gradient[0, 0] == pytest.approx(expected_force, rel=0, abs=3e-15)
    assert gradient[1, 0] == pytest.approx(-expected_force, rel=0, abs=3e-15)
    np.testing.assert_allclose(gradient[:, 1:], 0.0, rtol=0, atol=2e-16)


def test_generated_coordination_vjp_matches_multistep_finite_difference() -> None:
    elements = (8, 1, 1, 6)
    coordinates = np.array(
        [
            [0.1, -0.2, 0.3],
            [1.7, 0.1, -0.1],
            [-0.6, 1.5, 0.2],
            [4.0, -1.0, 0.5],
        ],
        dtype=np.float64,
    )
    compiled = _compile(elements, coordinates)
    weights = np.array([0.3, -0.7, 1.1, -0.2])
    gradient = execute(
        compiled.coordinate_vjp("coordination").program,
        {
            "coordinates": coordinates,
            "bar_coordination": weights,
        },
    ).outputs["bar_coordinates"]
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, rtol=0, atol=3e-14)

    for step in (4e-6, 2e-6, 1e-6):
        for atom, axis in ((0, 0), (1, 1), (3, 2)):
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            cplus = execute(compiled.program, {"coordinates": plus}).outputs[
                "coordination"
            ]
            cminus = execute(compiled.program, {"coordinates": minus}).outputs[
                "coordination"
            ]
            numerical = float(weights @ (cplus - cminus)) / (2 * step)
            assert gradient[atom, axis] == pytest.approx(numerical, rel=2e-7, abs=2e-8)


def test_generated_repulsion_vjp_matches_multistep_finite_difference() -> None:
    elements = (8, 1, 1, 6)
    coordinates = np.array(
        [
            [0.1, -0.2, 0.3],
            [1.7, 0.1, -0.1],
            [-0.6, 1.5, 0.2],
            [4.0, -1.0, 0.5],
        ],
        dtype=np.float64,
    )
    compiled = _compile(elements, coordinates)
    gradient = execute(
        compiled.coordinate_vjp("repulsion_energy").program,
        {
            "coordinates": coordinates,
            "bar_repulsion_energy": np.array(1.0),
        },
    ).outputs["bar_coordinates"]
    np.testing.assert_allclose(gradient.sum(axis=0), 0.0, rtol=0, atol=3e-14)

    for step in (4e-6, 2e-6, 1e-6):
        for atom, axis in ((0, 0), (1, 1), (3, 2)):
            plus = coordinates.copy()
            minus = coordinates.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            eplus = (
                execute(compiled.program, {"coordinates": plus})
                .outputs["repulsion_energy"]
                .item()
            )
            eminus = (
                execute(compiled.program, {"coordinates": minus})
                .outputs["repulsion_energy"]
                .item()
            )
            numerical = (eplus - eminus) / (2 * step)
            assert gradient[atom, axis] == pytest.approx(numerical, rel=2e-7, abs=2e-9)


def test_gfn1_pair_topology_preserves_reference_boundaries() -> None:
    geometry = gfn1_geometry((1, 8))
    threshold = GFN1_MINIMUM_DISTANCE_SQUARED_BOHR2**0.5

    exact_minimum = np.array([[0.0, 0.0, 0.0], [threshold, 0.0, 0.0]])
    assert float(threshold * threshold) == GFN1_MINIMUM_DISTANCE_SQUARED_BOHR2
    assert build_gfn1_pair_topology(geometry, exact_minimum).pairs == ((0, 1),)

    below = exact_minimum.copy()
    below[1, 0] = np.nextafter(threshold, 0.0)
    assert build_gfn1_pair_topology(geometry, below).pairs == ()

    exact_cutoff = np.array([[0.0, 0.0, 0.0], [GFN1_CUTOFF_BOHR, 0.0, 0.0]])
    assert build_gfn1_pair_topology(geometry, exact_cutoff).pairs == ((0, 1),)

    above = exact_cutoff.copy()
    above[1, 0] = np.nextafter(GFN1_CUTOFF_BOHR, np.inf)
    assert build_gfn1_pair_topology(geometry, above).pairs == ()


def test_changed_coordinates_crossing_cutoff_require_rebuilt_topology() -> None:
    geometry = gfn1_geometry((1, 1))
    near = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    far = np.array([[0.0, 0.0, 0.0], [GFN1_CUTOFF_BOHR + 2.0, 0.0, 0.0]])
    compiled = build_gfn1_short_range_program(
        geometry,
        build_gfn1_pair_topology(geometry, near),
    )
    compiled.validate_coordinates(near)
    with pytest.raises(ValueError, match="stale GFN1 pair topology"):
        compiled.validate_coordinates(far)


def test_gfn1_primal_and_generated_vjps_lower_through_shared_cuda_tensorir() -> None:
    from vibeqc_compiler.common.cuda_target import CUDA_TARGETS
    from vibeqc_compiler.tensor.cuda_emit import emit_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    geometry = gfn1_geometry((1, 6, 8))
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.2, 0.0], [-1.0, 2.1, 0.4]],
        dtype=np.float64,
    )
    compiled = build_gfn1_short_range_program(
        geometry,
        build_gfn1_pair_topology(geometry, coordinates),
    )
    target = CUDA_TARGETS["sm_80"]
    programs = (
        compiled.program,
        compiled.coordinate_vjp("coordination").program,
        compiled.coordinate_vjp("repulsion_energy").program,
    )
    for program in programs:
        source = emit_cuda(plan_cuda(program, target))
        assert "tensor_create" in source
        assert "tensor_run" in source


def test_pair_slice_does_not_claim_halogen_lowering() -> None:
    coordinates = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    compiled = _compile((35, 8), coordinates)
    assert compiled.program.provenance["halogen_lowering"] == "requires-triplet-ir"
    assert "halogen_energy" not in compiled.program.outputs
