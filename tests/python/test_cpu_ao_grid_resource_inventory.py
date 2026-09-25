"""The optional CPU RKS AO grid must fit the same dry-run resource envelope."""

from types import SimpleNamespace

import pytest
from vibeqc.resources_ks import _item_host_inventory


def inventory(
    n: int,
    points: int,
    *,
    backend: str = "cpu",
    pbe: bool = True,
    spins: int = 1,
) -> dict[str, int]:
    item = {
        "atoms": 2,
        "spins": spins,
        "grid_points": points,
        "orbital": {
            "nbf": n,
            "cartesian_nbf": n,
            "shells": 2,
            "primitives": 6,
            "maximum_angular": 0,
        },
    }
    model = SimpleNamespace(
        grid=SimpleNamespace(radial_points=24, angular_polar=8),
        tile_points=31,
        xc_schedule="device_fused",
        has_nonlocal_correlation=False,
    )
    return _item_host_inventory(
        item,
        diis_history=8,
        max_iterations=100,
        pbe=pbe,
        backend=backend,
        model=model,
    )


@pytest.mark.parametrize("n", [1, 2, 7, 16, 64, 512, 768])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_cache_is_charged_up_to_but_not_above_native_cap(n: int, offset: int) -> None:
    cap = 64 * 1024 * 1024
    points = cap // (32 * n) + offset
    cached = inventory(n, points)
    reference = inventory(n, points, pbe=False)
    expected = points * n * 32 if points * n * 32 <= cap else 0
    assert cached["ao_grid_cache"] == expected
    # GGA also retains the historical four-jet tile allowance independently.
    tile_extra = 3 * 8 * min(points, 31) * n
    assert cached["scf_workspace"] - reference["scf_workspace"] == expected + tile_extra
    assert cached["retained"] == reference["retained"]
    assert cached["setup_workspace"] == reference["setup_workspace"]


@pytest.mark.parametrize(
    "backend,pbe,spins",
    [("cpu", False, 1), ("cpu", True, 2), ("cuda", True, 1), ("cuda", True, 2)],
)
def test_unaffected_routes_do_not_reserve_cpu_rks_cache(
    backend: str, pbe: bool, spins: int
) -> None:
    assert (
        inventory(7, 1024, backend=backend, pbe=pbe, spins=spins)["ao_grid_cache"] == 0
    )
