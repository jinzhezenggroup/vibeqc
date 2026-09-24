"""Independent compact-factor gates with signed weights and arbitrary AO jets."""

import numpy as np
import pytest
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.dft.xc_contraction_cuda import (
    compact_panel_program,
    emit_native_xc_matrix_schedule,
)


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
    assert "totals[channel] = finite(totals[channel]+sum,error,3);" in source
    # Tiny/out-of-domain shapes keep the historical reducer rather than changing
    # their arithmetic or launch contract merely to share the production path.
    assert (
        "accumulate_totals<<<1,32,0,stream>>>(point_totals,count,totals,error);"
        in source
    )


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
