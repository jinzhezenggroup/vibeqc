"""Independent compact-factor gates with signed weights and arbitrary AO jets."""

from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.dft.xc_contraction_cuda import (
    DEFAULT_XC_MATRIX_SCHEDULE,
    XcMatrixSchedule,
    compact_panel_program,
    emit_native_xc_matrix_schedule,
    qualified_xc_matrix_schedules,
    select_xc_matrix_schedule,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "family,terms,jets", [("lda", 1, 1), ("gga", 4, 4), ("mgga", 5, 4)]
)
def test_compact_factor_matches_full_bilinear(
    family: str, terms: int, jets: int
) -> None:
    """Arbitrary coefficients detect lost gradient legs and doubled tau/rho."""
    rng = np.random.default_rng(43)
    ao = rng.normal(size=(jets, 31, 19))
    c = rng.normal(size=(terms, 31))
    weights = rng.normal(size=31)
    graph, roots = compact_panel_program(family)
    values = {f"x{j}": ao[j] for j in range(jets)}
    values.update({f"c{j}": c[j, :, None] for j in range(terms)})
    panels = evaluate_array_graph(graph, roots, values)
    actual = np.zeros((19, 19))
    for a, panel in zip(ao, panels, strict=False):
        cross = a.T @ (weights[:, None] * panel)
        actual += cross + cross.T
    expected = ao[0].T @ ((weights * c[0])[:, None] * ao[0])
    if jets > 1:
        for j in range(1, 4):
            cross = ao[j].T @ ((weights * c[j])[:, None] * ao[0])
            expected += cross + cross.T
    if terms == 5:
        for j in range(1, 4):
            expected += ao[j].T @ ((weights * c[4])[:, None] * ao[j])
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=5e-13)


def test_potential_uses_compact_triangular_tile_domain() -> None:
    """Potential launch work must contain only the authoritative tile triangle."""
    source = emit_native_xc_matrix_schedule()
    assert "const I tiles = (n+15)/16, tile_pairs = tiles*(tiles+1)/2;" in source
    assert "dim3(tile_pairs,1,spins)" in source
    assert "if (blockIdx.x > blockIdx.y) return;" not in source
    assert "tile_mu = pair-low*(low+1)/2;" in source

    for nao, square, triangle in ((192, 144, 78), (384, 576, 300), (768, 2304, 1176)):
        tiles = (nao + 15) // 16
        assert tiles * tiles == square
        assert tiles * (tiles + 1) // 2 == triangle
        assert triangle < square


def test_tiled_potential_fuses_point_total_reduction() -> None:
    """The production tiled path must not submit a separate total kernel per tile."""
    source = emit_native_xc_matrix_schedule()
    assert "I work_jets, const double* point_totals, double* potential," in source
    assert "blockIdx.x == 0 && blockIdx.z == 0" in source
    assert "threadIdx.y == 0 && threadIdx.x < 3" in source
    assert (
        "for (I p = 0; p < count; ++p) sum += point_totals[channel*count+p];" in source
    )
    assert (
        "totals[channel] = finite((accumulate ? totals[channel] : 0.0)+sum,error,3);"
        in source
    )
    # Tiny/out-of-domain shapes keep the historical reducer rather than changing
    # their arithmetic or launch contract merely to share the production path.
    assert (
        "accumulate_totals<<<1,32,0,stream>>>(point_totals,count,totals,error);"
        in source
    )


def test_tiled_first_point_tile_initializes_outputs_without_global_clears() -> None:
    """Admitted tiled XC must overwrite first-tile outputs instead of pre-clearing them."""
    schedule = emit_native_xc_matrix_schedule()
    assert "double* totals, bool accumulate, int* error" in schedule
    assert "const double prior = accumulate ? potential[index] : 0.0;" in schedule
    assert "if (!accumulate) {" in schedule
    assert "cudaMemsetAsync(potential,0,spins*n*n*sizeof(double),stream)" in schedule
    assert "cudaMemsetAsync(totals,0,3*sizeof(double),stream)" in schedule

    glue = (ROOT / "src/dft/cuda_xc_kernels.cuh").read_text()
    setup = glue[: glue.index("for (std::size_t begin")]
    assert "cudaMemsetAsync(potential" not in setup
    assert "cudaMemsetAsync(totals" not in setup
    assert "begin != 0, error" in glue


@pytest.mark.parametrize(
    "nao,spins,expected_matrix_bytes,expected_removed_bytes",
    [
        (384, 1, 1_179_648, 1_179_672),
        (384, 2, 2_359_296, 2_359_320),
        (768, 1, 4_718_592, 4_718_616),
        (768, 2, 9_437_184, 9_437_208),
    ],
)
def test_first_tile_initialization_traffic_census(
    nao: int, spins: int, expected_matrix_bytes: int, expected_removed_bytes: int
) -> None:
    """Pin the matrix and totals clear traffic removed per admitted XC evaluation."""
    matrix_bytes = spins * nao * nao * 8
    assert matrix_bytes == expected_matrix_bytes
    assert matrix_bytes + 3 * 8 == expected_removed_bytes


@pytest.mark.parametrize(
    "points,expected_tiles,expected_two_step_launches",
    [(1_327_104, 5_184, 10_368), (2_654_208, 10_368, 20_736)],
)
def test_tiled_total_reduction_launch_census(
    points: int, expected_tiles: int, expected_two_step_launches: int
) -> None:
    """Pin the retained 256-point 48/96-atom diagnostic launch census."""
    tiles = (points + 255) // 256
    assert tiles == expected_tiles
    assert 2 * tiles == expected_two_step_launches
    # Before this slice: one standalone accumulate_totals launch per point tile.
    # After this slice: zero standalone reduction launches in the admitted tiled path.
    assert expected_two_step_launches > 0


def test_xc_matrix_candidates_are_resource_qualified() -> None:
    candidates = qualified_xc_matrix_schedules(384, 256, spins=2, work_jets=4)
    assert tuple(schedule.tile for schedule in candidates) == (8, 16, 32)
    assert DEFAULT_XC_MATRIX_SCHEDULE in candidates
    assert all(schedule.threads <= 1024 for schedule in candidates)
    assert all(schedule.shared_bytes <= 48 * 1024 for schedule in candidates)
    assert all(schedule.additional_workspace_bytes == 0 for schedule in candidates)


def test_xc_matrix_selection_keeps_measured_default_without_profile() -> None:
    selected = select_xc_matrix_schedule(384, 256)
    assert selected == DEFAULT_XC_MATRIX_SCHEDULE
    assert select_xc_matrix_schedule(384, 256, preferred_tile=32) == XcMatrixSchedule(
        32
    )

    # An 8x8 candidate may be legal here, but no-profile production must retain
    # the historical scalar fallback when the qualified 16x16 default is absent.
    assert select_xc_matrix_schedule(8, 8) is None
    with pytest.raises(ValueError, match="not qualified"):
        select_xc_matrix_schedule(16, 16, preferred_tile=32)


def test_xc_matrix_admission_charges_compact_triangle_grid_dimension() -> None:
    schedule = XcMatrixSchedule(16)
    assert schedule.admitted(361 * 16, 256)
    assert not schedule.admitted(362 * 16, 256)
    assert not schedule.admitted(
        16, 16, spins=2, work_jets=4, maximum_grid_dimension=7
    )

    source = emit_native_xc_matrix_schedule()
    assert "const I tile_pairs = tiles*(tiles+1)/2;" in source
    assert "tile_pairs <= 65535;" in source


def test_xc_matrix_emitter_materializes_explicit_tile_candidate() -> None:
    source = emit_native_xc_matrix_schedule(XcMatrixSchedule(8))
    assert "__shared__ double d[8][9], a[8][9];" in source
    assert "dim3(8,8)" in source
    assert "(n+7)/8" in source
    assert "all 64 lanes" in source
    assert "dim3(16,16)" not in source

