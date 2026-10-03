"""Native force capacity is authoritative; explicit caps remain hard limits."""

import ctypes as ct
import os
from fractions import Fraction
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from generativeqc import GridSpec, KsOptions, _native
from generativeqc.ks import native_ks_options, resolve_ks_options
from generativeqc.nonlocal_runtime import (
    NonlocalFixedGridPlan,
    _ResidentNonlocalForceOwner,
)
from generativeqc_compiler.method import original_nonlocal_correlation


def test_capacity_query_forwards_shapes_without_creating_an_owner() -> None:
    def query(points: int, tile: int, output: object) -> int:
        assert (points, tile) == (1_179_648, 256)
        ct.cast(output, ct.POINTER(ct.c_uint64))[0] = 283_189_268
        return 0

    library = SimpleNamespace(
        generativeqc_internal_nonlocal_cuda_force_bytes_v1=MagicMock(side_effect=query),
        generativeqc_internal_nonlocal_cuda_force_create_v1=MagicMock(),
    )
    assert (
        _ResidentNonlocalForceOwner.required_device_bytes(library, 1_179_648, 256)
        == 283_189_268
    )
    library.generativeqc_internal_nonlocal_cuda_force_create_v1.assert_not_called()


@pytest.mark.parametrize("points,tile", [(0, 1), (1, 0), (2**32, 1), (1, True)])
def test_capacity_query_rejects_invalid_abi_extents_before_native_call(
    points: int, tile: int
) -> None:
    library = SimpleNamespace(
        generativeqc_internal_nonlocal_cuda_force_bytes_v1=MagicMock()
    )
    with pytest.raises((TypeError, ValueError)):
        _ResidentNonlocalForceOwner.required_device_bytes(library, points, tile)
    library.generativeqc_internal_nonlocal_cuda_force_bytes_v1.assert_not_called()


def test_missing_capacity_query_fails_closed() -> None:
    with pytest.raises(RuntimeError, match="capacity query"):
        _ResidentNonlocalForceOwner.required_device_bytes(SimpleNamespace(), 100, 16)


@pytest.mark.parametrize("cap", [None, 1 << 20, 256 << 20])
def test_nonlocal_budget_default_and_explicit_caps_survive_semantic_abi(
    cap: int | None,
) -> None:
    kwargs = {} if cap is None else {"nonlocal_memory_budget_bytes": cap}
    options = resolve_ks_options("wb97m-v-rks", KsOptions(grid=GridSpec(), **kwargs))
    expected = (1 << 30) if cap is None else cap
    assert options.nonlocal_memory_budget_bytes == expected
    assert native_ks_options(options).nonlocal_maximum_bytes == expected


@pytest.mark.parametrize("points", [1, 127, 128, 129, 589_824, 1_179_648, 2_359_296])
def test_cuda_capacity_query_matches_allocation_and_exact_cap(points: int) -> None:
    """Include 48/96-atom full-grid sizes without performing quadratic pairs."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_WB97MV_CUDA=1 inside Slurm")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    library = _native.load_library(device="cuda", device_id=0)
    required = _ResidentNonlocalForceOwner.required_device_bytes(library, points, 256)
    # These complete 48/96-atom grids used 283,189,268/566,378,516 bytes.
    # Phase reuse must recover 7*N doubles without changing the hard cap or
    # dropping the full-grid pair/seed inventory from allocation accounting.
    expected_large = {1_179_648: 217_128_980, 2_359_296: 434_257_940}
    if points in expected_large:
        assert required == expected_large[points]
    spec = original_nonlocal_correlation("rvv10")
    coordinates = np.zeros((points, 3), dtype=np.float64)
    weights = np.ones(points, dtype=np.float64)
    with NonlocalFixedGridPlan(
        spec, points, device="cuda", maximum_bytes=1 << 30, library=library
    ) as plan:
        options = {
            "coefficient": Fraction(1),
            "tile_points": 256,
            "density_threshold": 1e-8,
            "device_id": 0,
            "context": plan._context,
            "library": library,
        }
        with _ResidentNonlocalForceOwner(
            spec, coordinates, weights, maximum_bytes=required, **options
        ) as owner:
            assert owner.diagnostic().device_bytes == required
        with pytest.raises(RuntimeError, match="GENERATIVEQC error 7"):
            # The Python ABI maps native OUT_OF_MEMORY to RuntimeError.
            _ResidentNonlocalForceOwner(
                spec, coordinates, weights, maximum_bytes=required - 1, **options
            )
