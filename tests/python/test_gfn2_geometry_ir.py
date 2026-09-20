"""GFN2 CN/repulsion compiler vertical slice qualification (#504)."""

from __future__ import annotations

import os
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
import pytest
from vibeqc_compiler.geometry import (
    GFN2_CUTOFF_BOHR,
    GFN2_SHORT_RANGE_PARAMETER_IDENTITY,
    build_gfn2_batch_pair_topology,
    build_gfn2_pair_topology,
    build_gfn2_short_range_batch_program,
    build_gfn2_short_range_program,
    gfn2_element_parameters,
    gfn2_geometry,
)
from vibeqc_compiler.tensor import execute

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from numpy.typing import ArrayLike
    from vibeqc_compiler.geometry import Gfn2ShortRangeProgram

# Independent values frozen in xTBloom's mctc-lib/tblite qualification tests.
CN_ATOMIC_NUMBERS = (11, 1, 8, 1, 9, 1, 1, 8, 7, 1, 1, 17, 5, 5, 7, 13)
CN_COORDINATES = np.array(
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
        1.8427576754846,
        4.55038084858449,
        -3.5435612184397,
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
        -3.9186247436056,
        -3.02526098819107,
        2.53667889095925,
        2.31664984740423,
        -2.00438948664892,
        -2.2923513697722,
        2.19782807357059,
        1.12226554109716,
        -1.36942007032045,
        0.48455055461782,
    ],
    dtype=np.float64,
).reshape(-1, 3)
CN_EXPECTED = np.array(
    [
        4.11453659059991,
        0.932058998762811,
        2.03554597140311,
        1.42227835389358,
        1.12812426574031,
        1.05491602558828,
        1.52709064704269,
        1.95070367247232,
        3.8375988919654,
        1.09388314007182,
        1.0709077369534,
        2.0028525408283,
        4.36400837813955,
        3.8346986054608,
        3.91542517673963,
        5.5857168241996,
    ],
    dtype=np.float64,
)

# xtb LGPL reference geometry and energy frozen by xTBloom's repulsion oracle.
REP_ATOMIC_NUMBERS = (
    6,
    7,
    6,
    7,
    6,
    6,
    6,
    8,
    7,
    6,
    8,
    7,
    6,
    6,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
)
REP_COORDINATES = np.array(
    [
        2.02799738646442,
        0.09231312124713,
        -0.14310895950963,
        4.75011007621,
        0.02373496014051,
        -0.14324124033844,
        6.33434307654413,
        2.07098865582721,
        -0.1423530690593,
        8.72860718071825,
        1.38002919517619,
        -0.14265542523943,
        8.6531882110361,
        -1.19324866489847,
        -0.14231527453678,
        6.23857175648671,
        -2.08353643730276,
        -0.14218299370797,
        5.63266886875962,
        -4.69950321056008,
        -0.13940509630299,
        3.44931709749015,
        -5.48092386085491,
        -0.14318454855466,
        7.77508917214346,
        -6.24427872938674,
        -0.13107140408805,
        10.30229550927022,
        -5.39739796609292,
        -0.1367216852043,
        12.07410272485492,
        -6.91573621641911,
        -0.13666499342053,
        10.70038521493902,
        -2.79078533715849,
        -0.14148379504141,
        13.24597858727017,
        -1.76969072232377,
        -0.14218299370797,
        7.40891694074004,
        -8.95905928176407,
        -0.11636933482904,
        1.38702118184179,
        2.05575746325296,
        -0.14178615122154,
        1.34622199478497,
        -0.86356704498496,
        1.55590600570783,
        1.34624089204623,
        -0.86133716815647,
        -1.84340893849267,
        5.65596919189118,
        4.0017218385948,
        -0.14131371969009,
        14.67430918222276,
        -3.26230980007732,
        -0.14344911021228,
        13.5089717722029,
        -0.60815166181684,
        1.54898960808727,
        13.50780014200488,
        -0.60614855212345,
        -1.83214617078268,
        5.41408424778406,
        -9.49239668625902,
        -0.11022772492007,
        8.31919801555568,
        -9.74947502841788,
        1.56539243085954,
        8.31511620712388,
        -9.76854236502758,
        -1.79108242206824,
    ],
    dtype=np.float64,
).reshape(-1, 3)
REP_EXPECTED_ENERGY = 0.49222837261241


def _compiled(elements: Iterable[int], coordinates: ArrayLike) -> Gfn2ShortRangeProgram:
    geometry = gfn2_geometry(elements)
    topology = build_gfn2_pair_topology(geometry, coordinates)
    return build_gfn2_short_range_program(
        geometry,
        topology,
    )


def test_parameter_subset_covers_complete_gfn2_element_domain() -> None:
    assert len({gfn2_element_parameters(z).atomic_number for z in range(1, 87)}) == 86
    for z in range(1, 87):
        item = gfn2_element_parameters(z)
        assert item.covalent_radius_bohr > 0
        assert item.arep > 0
        assert item.zeff > 0
    with pytest.raises(ValueError, match="1..86"):
        gfn2_element_parameters(87)


def test_gfn2_coordination_matches_pinned_mctc_xtbloom_oracle() -> None:
    compiled = _compiled(
        CN_ATOMIC_NUMBERS,
        CN_COORDINATES,
    )
    compiled.validate_coordinates(CN_COORDINATES)
    result = execute(
        compiled.program,
        {"coordinates": CN_COORDINATES},
    ).outputs["coordination"]
    np.testing.assert_allclose(
        result,
        CN_EXPECTED,
        rtol=0,
        atol=5e-13,
    )
    assert compiled.parameter_identity == GFN2_SHORT_RANGE_PARAMETER_IDENTITY


def test_gfn2_repulsion_matches_pinned_xtb_oracle() -> None:
    compiled = _compiled(
        REP_ATOMIC_NUMBERS,
        REP_COORDINATES,
    )
    energy = (
        execute(
            compiled.program,
            {"coordinates": REP_COORDINATES},
        )
        .outputs["repulsion_energy"]
        .item()
    )
    assert energy == pytest.approx(
        REP_EXPECTED_ENERGY,
        abs=1.1e-13,
        rel=0,
    )


def test_generated_repulsion_vjp_matches_finite_difference_and_translation() -> None:
    compiled = _compiled(
        REP_ATOMIC_NUMBERS,
        REP_COORDINATES,
    )
    reverse = compiled.coordinate_vjp("repulsion_energy").program
    gradient = execute(
        reverse,
        {
            "coordinates": REP_COORDINATES,
            "bar_repulsion_energy": np.array(1.0),
        },
    ).outputs["bar_coordinates"]
    np.testing.assert_allclose(
        gradient.sum(axis=0),
        0.0,
        atol=2e-13,
        rtol=0,
    )

    for step in (4e-6, 2e-6, 1e-6):
        for atom, axis in (
            (0, 0),
            (7, 1),
            (14, 2),
            (23, 0),
        ):
            plus = REP_COORDINATES.copy()
            minus = REP_COORDINATES.copy()
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
            assert gradient[atom, axis] == pytest.approx(numerical, rel=8e-8, abs=8e-9)


def test_generated_coordination_vjp_matches_weighted_finite_difference() -> None:
    compiled = _compiled(
        CN_ATOMIC_NUMBERS,
        CN_COORDINATES,
    )
    weights = np.linspace(
        -0.4,
        0.7,
        len(CN_ATOMIC_NUMBERS),
    )
    reverse = compiled.coordinate_vjp("coordination").program
    gradient = execute(
        reverse,
        {
            "coordinates": CN_COORDINATES,
            "bar_coordination": weights,
        },
    ).outputs["bar_coordinates"]
    np.testing.assert_allclose(
        gradient.sum(axis=0),
        0.0,
        atol=3e-13,
        rtol=0,
    )

    for step in (4e-6, 2e-6, 1e-6):
        for atom, axis in (
            (0, 1),
            (5, 2),
            (11, 0),
        ):
            plus = CN_COORDINATES.copy()
            minus = CN_COORDINATES.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            cplus = execute(compiled.program, {"coordinates": plus}).outputs[
                "coordination"
            ]
            cminus = execute(compiled.program, {"coordinates": minus}).outputs[
                "coordination"
            ]
            numerical = float(weights @ (cplus - cminus)) / (2 * step)
            assert gradient[atom, axis] == pytest.approx(numerical, rel=1e-7, abs=1e-8)


def _ragged_batch_fixture() -> tuple[tuple[int, ...], tuple[int, ...], np.ndarray]:
    systems = (
        (
            (8, 1, 1),
            np.array(
                [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [-0.4, 1.4, 0.2]],
                dtype=np.float64,
            ),
        ),
        (
            (6, 1, 1, 1, 1),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.7, 0.0, 0.0],
                    [-0.6, 1.6, 0.1],
                    [-0.5, -0.8, 1.4],
                    [-0.5, -0.8, -1.4],
                ],
                dtype=np.float64,
            ),
        ),
    )
    elements = tuple(z for system, _ in systems for z in system)
    coordinates = np.vstack([coordinates for _, coordinates in systems])
    offsets = (0, len(systems[0][0]), len(elements))
    return elements, offsets, coordinates


def test_gfn2_ragged_batch_matches_independent_single_system_graphs() -> None:
    elements, offsets, coordinates = _ragged_batch_fixture()
    geometry = gfn2_geometry(elements)
    topology = build_gfn2_batch_pair_topology(geometry, offsets, coordinates)
    compiled = build_gfn2_short_range_batch_program(geometry, offsets, topology)
    actual = execute(compiled.program, {"coordinates": coordinates}).outputs

    expected_cn = []
    expected_repulsion = []
    for begin, end in pairwise(offsets):
        single = _compiled(elements[begin:end], coordinates[begin:end])
        result = execute(
            single.program, {"coordinates": coordinates[begin:end]}
        ).outputs
        expected_cn.append(result["coordination"])
        expected_repulsion.append(result["repulsion_energy"].item())
    np.testing.assert_allclose(
        actual["coordination"], np.concatenate(expected_cn), rtol=0, atol=3e-14
    )
    np.testing.assert_allclose(
        actual["repulsion_energy"], expected_repulsion, rtol=0, atol=3e-14
    )

    assert topology.pairs == tuple(sorted(topology.pairs))
    for first, second in topology.pairs:
        assert any(begin <= first < second < end for begin, end in pairwise(offsets))

    # Both systems intentionally reuse the origin. Cross-system coincidence is
    # valid because pair ownership is constrained by the ragged partition.
    assert np.array_equal(coordinates[0], coordinates[offsets[1]])


def test_gfn2_ragged_batch_is_system_and_atom_permutation_invariant() -> None:
    elements, offsets, coordinates = _ragged_batch_fixture()
    geometry = gfn2_geometry(elements)
    compiled = build_gfn2_short_range_batch_program(
        geometry,
        offsets,
        build_gfn2_batch_pair_topology(geometry, offsets, coordinates),
    )
    baseline = execute(compiled.program, {"coordinates": coordinates}).outputs

    system_order = (1, 0)
    slices = tuple((offsets[i], offsets[i + 1]) for i in range(2))
    permuted_elements = tuple(
        z for i in system_order for z in elements[slices[i][0] : slices[i][1]]
    )
    permuted_coordinates = np.vstack(
        [coordinates[slices[i][0] : slices[i][1]] for i in system_order]
    )
    permuted_offsets = (0, slices[1][1] - slices[1][0], len(elements))
    permuted_geometry = gfn2_geometry(permuted_elements)
    permuted = build_gfn2_short_range_batch_program(
        permuted_geometry,
        permuted_offsets,
        build_gfn2_batch_pair_topology(
            permuted_geometry, permuted_offsets, permuted_coordinates
        ),
    )
    permuted_result = execute(
        permuted.program, {"coordinates": permuted_coordinates}
    ).outputs
    np.testing.assert_allclose(
        permuted_result["repulsion_energy"],
        baseline["repulsion_energy"][::-1],
        rtol=0,
        atol=3e-14,
    )
    np.testing.assert_allclose(
        permuted_result["coordination"],
        np.concatenate((baseline["coordination"][3:], baseline["coordination"][:3])),
        rtol=0,
        atol=3e-14,
    )

    atom_order = (2, 0, 1)
    first_elements = tuple(elements[i] for i in atom_order)
    first_coordinates = coordinates[list(atom_order)]
    first = _compiled(first_elements, first_coordinates)
    first_result = execute(first.program, {"coordinates": first_coordinates}).outputs
    np.testing.assert_allclose(
        first_result["coordination"],
        baseline["coordination"][:3][list(atom_order)],
        rtol=0,
        atol=3e-14,
    )
    assert first_result["repulsion_energy"].item() == pytest.approx(
        baseline["repulsion_energy"][0], rel=0, abs=3e-14
    )


def test_gfn2_ragged_batch_rejects_cross_system_and_stale_pair_state() -> None:
    elements, offsets, coordinates = _ragged_batch_fixture()
    geometry = gfn2_geometry(elements)
    topology = build_gfn2_batch_pair_topology(geometry, offsets, coordinates)
    compiled = build_gfn2_short_range_batch_program(geometry, offsets, topology)
    changed = coordinates.copy()
    changed[1] = [GFN2_CUTOFF_BOHR + 3.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="stale GFN2 batch pair topology"):
        compiled.validate_coordinates(changed)

    from vibeqc_compiler.geometry import PairCutoff, PairTopology

    cross_system = PairTopology(
        len(elements),
        ((0, offsets[1]),),
        cutoff=PairCutoff(GFN2_CUTOFF_BOHR),
    )
    with pytest.raises(ValueError, match="cross-system pair"):
        build_gfn2_short_range_batch_program(geometry, offsets, cross_system)


def test_changed_geometry_requires_rebuilt_25_bohr_topology() -> None:
    geometry = gfn2_geometry((1, 1))
    near = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    far = np.array(
        [
            [0.0, 0.0, 0.0],
            [GFN2_CUTOFF_BOHR + 2.0, 0.0, 0.0],
        ]
    )
    first = build_gfn2_pair_topology(
        geometry,
        near,
    )
    second = build_gfn2_pair_topology(
        geometry,
        far,
    )
    assert first.pairs == ((0, 1),)
    assert second.pairs == ()
    assert first.identity != second.identity

    compiled = build_gfn2_short_range_program(
        geometry,
        first,
    )
    compiled.validate_coordinates(near)
    with pytest.raises(
        ValueError,
        match="stale GFN2 pair topology",
    ):
        compiled.validate_coordinates(far)


def test_gfn2_primal_and_generated_vjps_lower_through_shared_cuda_tensorir() -> None:
    from vibeqc_compiler.common.cuda_target import (
        CUDA_TARGETS,
    )
    from vibeqc_compiler.tensor.cuda_emit import (
        emit_cuda,
    )
    from vibeqc_compiler.tensor.cuda_plan import (
        plan_cuda,
    )

    geometry = gfn2_geometry((1, 6, 8))
    coordinates = np.array(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.2, 0.0],
            [-1.0, 2.1, 0.4],
        ]
    )
    topology = build_gfn2_pair_topology(
        geometry,
        coordinates,
    )
    compiled = build_gfn2_short_range_program(
        geometry,
        topology,
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


def test_gfn2_integration_preserves_existing_d3_and_scf_history_contracts() -> None:
    from vibeqc_compiler import geometry
    from vibeqc_compiler.tensor import Index, IndexSpace

    assert geometry.D3CompilerSpec is not None
    assert callable(geometry.compile_d3_bj)
    assert callable(geometry.execute_d3_bj)
    assert Index("step", IndexSpace("diis_history", "history", 2)).extent == 2


@pytest.mark.skipif(
    os.environ.get("VIBEQC_GFN2_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
def test_gfn2_ragged_geometry_primal_and_vjps_execute_on_cuda(
    tmp_path: Path,
) -> None:
    from vibeqc.profiles import find_nvcc
    from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.integral.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_GFN2_CUDA_TEST requires a CUDA compiler")
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )
    elements, offsets, coordinates = _ragged_batch_fixture()
    geometry = gfn2_geometry(elements)
    topology = build_gfn2_batch_pair_topology(geometry, offsets, coordinates)
    compiled = build_gfn2_short_range_batch_program(geometry, offsets, topology)
    programs = (
        (compiled.program, {"coordinates": coordinates}),
        (
            compiled.coordinate_vjp("coordination").program,
            {
                "coordinates": coordinates,
                "bar_coordination": np.linspace(-0.35, 0.45, len(elements)),
            },
        ),
        (
            compiled.coordinate_vjp("repulsion_energy").program,
            {
                "coordinates": coordinates,
                "bar_repulsion_energy": np.array([0.7, -0.2]),
            },
        ),
    )
    for index, (program, feeds) in enumerate(programs):
        plan = plan_cuda(program, compiler.target)
        expected = execute(program, feeds).outputs
        cache = tmp_path / f"ragged-geometry-{index}"
        cache.mkdir()
        with PreparedCuda(plan, compile_cuda(plan, compiler, cache)) as prepared:
            actual = prepared.execute(feeds).outputs
        assert actual.keys() == expected.keys()
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=5e-13)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_GFN2_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)
def test_gfn2_geometry_primal_and_vjps_execute_on_cuda(tmp_path: Path) -> None:
    from vibeqc.profiles import find_nvcc
    from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
    from vibeqc_compiler.integral.cuda_target import cuda_target_info
    from vibeqc_compiler.tensor.cuda_execute import PreparedCuda, compile_cuda
    from vibeqc_compiler.tensor.cuda_plan import plan_cuda

    nvcc = find_nvcc()
    if nvcc is None:
        pytest.fail("VIBEQC_GFN2_CUDA_TEST requires a CUDA compiler")
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )

    geometry = gfn2_geometry((1, 6, 8))
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.2, 0.0], [-1.0, 2.1, 0.4]],
        dtype=np.float64,
    )
    topology = build_gfn2_pair_topology(geometry, coordinates)
    compiled = build_gfn2_short_range_program(geometry, topology)
    programs = (
        (compiled.program, {"coordinates": coordinates}),
        (
            compiled.coordinate_vjp("coordination").program,
            {
                "coordinates": coordinates,
                "bar_coordination": np.array([0.3, -0.2, 0.5]),
            },
        ),
        (
            compiled.coordinate_vjp("repulsion_energy").program,
            {
                "coordinates": coordinates,
                "bar_repulsion_energy": np.array(1.0),
            },
        ),
    )
    for index, (program, feeds) in enumerate(programs):
        plan = plan_cuda(program, compiler.target)
        expected = execute(program, feeds).outputs
        cache = tmp_path / f"geometry-{index}"
        cache.mkdir()
        with PreparedCuda(plan, compile_cuda(plan, compiler, cache)) as prepared:
            actual = prepared.execute(feeds).outputs
        assert actual.keys() == expected.keys()
        for name in expected:
            np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=5e-13)
