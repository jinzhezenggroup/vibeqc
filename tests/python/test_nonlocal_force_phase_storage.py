"""Qualify private VV10 force-buffer lifetimes on a scheduled CUDA device.

The oracle differentiates the discrete fixed-grid functional independently of
the native owner. All six seed channels, input immutability, changing screening
after reset, tiled collection, and asynchronous failure recovery are checked.
"""

from __future__ import annotations

import ctypes as ct
import os
from fractions import Fraction
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from generativeqc import _native
from generativeqc.nonlocal_runtime import (
    NonlocalFixedGridPlan,
    _ResidentNonlocalForceOwner,
)
from generativeqc_compiler.dft.cuda import GridTaskView
from generativeqc_compiler.dft.nonlocal_reference import (
    nonlocal_explicit_geometry_derivatives_reference,
    nonlocal_feature_derivatives_reference,
)
from generativeqc_compiler.method import original_nonlocal_correlation

if TYPE_CHECKING:
    from generativeqc_compiler.common.nonlocal_correlation import (
        NonlocalCorrelationSpec,
    )

THRESHOLD = 1e-8
COEFFICIENT = Fraction(3, 5)


@pytest.fixture(scope="module")
def cuda_runtime() -> tuple[Any, ct.CDLL]:
    """Import CUDA only after the explicit Slurm qualification opt-in."""
    if os.environ.get("GENERATIVEQC_TEST_WB97MV_CUDA") != "1":
        pytest.skip("set GENERATIVEQC_TEST_WB97MV_CUDA=1 inside Slurm")
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    import cupy as cp

    return cp, _native.load_library(device="cuda", device_id=0)


def _inputs(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Retain signed and active zero weights, plus initially screened rows."""
    rng = np.random.default_rng(9821 + count)
    points = rng.normal(size=(count, 3))
    weights = rng.uniform(0.1, 1.1, size=count)
    weights[1::5] *= -1.0
    weights[::11] = 0.0
    weights[5::17] = -0.0
    density = rng.uniform(0.3, 1.2, size=count)
    gradient = rng.normal(scale=0.04, size=(count, 3))
    density[::7] = THRESHOLD / 2
    if count > 2:
        density[2] = THRESHOLD
        gradient[2] = 0.0
    return points, weights, density, gradient


def _reference(
    points: np.ndarray,
    weights: np.ndarray,
    density: np.ndarray,
    gradient: np.ndarray,
    spec: NonlocalCorrelationSpec,
) -> np.ndarray:
    """Apply MolecularV1 then independently evaluate feature/geometry seeds."""
    inactive = density < THRESHOLD
    effective_weights = np.where(inactive, 0.0, weights)
    effective_density = np.where(inactive, 1.0, density)
    effective_gradient = np.where(inactive[:, None], 0.0, gradient)
    rho, sigma = nonlocal_feature_derivatives_reference(
        points, effective_weights, effective_density, effective_gradient, spec
    )
    point, weight = nonlocal_explicit_geometry_derivatives_reference(
        points, effective_weights, effective_density, effective_gradient, spec
    )
    seeds = float(COEFFICIENT) * np.vstack((rho, sigma, point.T, weight))
    seeds[:, inactive] = 0.0
    return seeds


def _owner(
    plan: NonlocalFixedGridPlan,
    points: np.ndarray,
    weights: np.ndarray,
    library: ct.CDLL,
) -> _ResidentNonlocalForceOwner:
    """Use the exact native inventory; the public primitive stays independent."""
    required = _ResidentNonlocalForceOwner.required_device_bytes(
        library, len(weights), 64
    )
    return _ResidentNonlocalForceOwner(
        plan.spec,
        points,
        weights,
        coefficient=COEFFICIENT,
        tile_points=64,
        maximum_bytes=required,
        density_threshold=THRESHOLD,
        device_id=0,
        context=plan._context,
        library=library,
    )


def _download_seeds(
    cp: Any,
    owner: _ResidentNonlocalForceOwner,
    consumer: Any,
) -> np.ndarray:
    """Drain the borrowed generation before reset or release, as consumers must."""
    seed = owner.execute()
    assert seed.stream == consumer.ptr and seed.stride == owner.point_count
    memory = cp.cuda.UnownedMemory(seed.pointer, 6 * seed.stride * 8, owner)
    array = cp.ndarray(
        (6, seed.stride), dtype=cp.float64, memptr=cp.cuda.MemoryPointer(memory, 0)
    )
    return array.get(stream=consumer)


def _seed_device(
    cp: Any,
    owner: _ResidentNonlocalForceOwner,
    producer: Any,
    consumer: Any,
    density: np.ndarray,
    gradient: np.ndarray,
) -> np.ndarray:
    """Exercise the production D2D bridge, including distinct producer streams."""
    with producer:
        device_density = cp.asarray(density)
        device_gradient = cp.asarray(gradient)
    view = GridTaskView(version=1, npoint=len(density), stream=consumer.ptr)
    seed = owner._library.generativeqc_internal_nonlocal_cuda_force_seed_device_v1
    seed.argtypes = [
        ct.c_void_p,
        ct.c_void_p,
        ct.c_int,
        ct.c_void_p,
        ct.c_void_p,
        ct.c_size_t,
        ct.c_void_p,
        ct.POINTER(GridTaskView),
    ]
    seed.restype = ct.c_int
    _native.check(
        owner._library,
        seed(
            owner._owner,
            owner._context,
            0,
            device_density.data.ptr,
            device_gradient.data.ptr,
            len(density),
            producer.ptr,
            ct.byref(view),
        ),
        context=owner._context,
    )
    result = _download_seeds(cp, owner, consumer)
    # In-place domain preparation is legal only on the owner's private copies.
    np.testing.assert_array_equal(device_density.get(stream=producer), density)
    np.testing.assert_array_equal(device_gradient.get(stream=producer), gradient)
    return result


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
@pytest.mark.parametrize("count", (1, 129, 257))
@pytest.mark.parametrize("cross_stream", (False, True))
def test_private_force_phase_reuse_matches_oracle_across_reset(
    cuda_runtime: tuple[Any, ct.CDLL], variant: str, count: int, cross_stream: bool
) -> None:
    """Unscreening must recover immutable weights and replace all aliased data."""
    cp, library = cuda_runtime
    spec = original_nonlocal_correlation(variant)
    points, weights, density, gradient = _inputs(count)
    consumer = cp.cuda.Stream(non_blocking=True)
    producer = cp.cuda.Stream(non_blocking=True) if cross_stream else consumer
    with (
        NonlocalFixedGridPlan(spec, count, device="cuda", library=library) as plan,
        _owner(plan, points, weights, library) as owner,
    ):
        for generation in range(3):
            if generation:
                owner.reset()
                # Rows that were padded must regain their original signed
                # weights; other rows become newly inactive in each pass.
                density = np.full(count, 0.7 + 0.1 * generation)
                density[generation::9] = THRESHOLD / 2
                gradient = np.roll(gradient, 1, axis=0).copy()
            actual = _seed_device(cp, owner, producer, consumer, density, gradient)
            expected = _reference(points, weights, density, gradient, spec)
            np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=1e-12)
            assert owner.diagnostic().generation == generation + 1


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
@pytest.mark.parametrize("count", (129, 257))
def test_force_phase_reuse_after_tiled_feature_collection(
    cuda_runtime: tuple[Any, ct.CDLL],
    variant: str,
    count: int,
) -> None:
    """The bounded fallback must repopulate private panels, including tail rows."""
    cp, library = cuda_runtime
    spec = original_nonlocal_correlation(variant)
    points, weights, density, gradient = _inputs(count)
    consumer = cp.cuda.Stream(non_blocking=True)
    with (
        NonlocalFixedGridPlan(spec, count, device="cuda", library=library) as plan,
        _owner(plan, points, weights, library) as owner,
    ):
        for generation in range(2):
            if generation:
                owner.reset()
                density = np.full(count, 0.9)
                gradient = -gradient
            # Keep borrowed feature arrays alive until the consumer drain.
            retained = []
            with consumer:
                error = cp.zeros(1, dtype=cp.int32)
                for begin in range(0, count, 63):
                    end = min(begin + 63, count)
                    features = np.zeros((13, end - begin))
                    features[0] = features[5] = density[begin:end] / 2
                    features[1:4] = features[6:9] = gradient[begin:end].T / 2
                    device = cp.asarray(features)
                    retained.append(device)
                    view = GridTaskView(
                        version=1,
                        generation=generation + 1,
                        npoint=end - begin,
                        features=ct.cast(device.data.ptr, ct.POINTER(ct.c_double)),
                        stream=consumer.ptr,
                        error=ct.cast(error.data.ptr, ct.POINTER(ct.c_int)),
                    )
                    owner.collect(
                        SimpleNamespace(view=view, _owner=SimpleNamespace(device_id=0)),
                        begin,
                    )
            actual = _download_seeds(cp, owner, consumer)
            np.testing.assert_allclose(
                actual,
                _reference(points, weights, density, gradient, spec),
                rtol=2e-11,
                atol=1e-12,
            )


@pytest.mark.parametrize("variant", ("vv10", "rvv10"))
@pytest.mark.parametrize("invalid", ("density_nan", "density_negative", "gradient_nan"))
def test_force_phase_aliases_poison_invalid_generation_then_recover(
    cuda_runtime: tuple[Any, ct.CDLL], variant: str, invalid: str
) -> None:
    """Invalid screened rows must poison every channel; reset clears all errors."""
    cp, library = cuda_runtime
    count = 129
    spec = original_nonlocal_correlation(variant)
    points, weights, density, gradient = _inputs(count)
    consumer = cp.cuda.Stream(non_blocking=True)
    producer = cp.cuda.Stream(non_blocking=True)
    bad_density, bad_gradient = density.copy(), gradient.copy()
    if invalid == "density_nan":
        bad_density[0] = np.nan
    elif invalid == "density_negative":
        bad_density[0] = -1.0
    else:
        bad_gradient[0, 1] = np.nan
    with (
        NonlocalFixedGridPlan(spec, count, device="cuda", library=library) as plan,
        _owner(plan, points, weights, library) as owner,
    ):
        poisoned = _seed_device(
            cp, owner, producer, consumer, bad_density, bad_gradient
        )
        assert np.isnan(poisoned).all()
        owner.reset()
        recovered = _seed_device(cp, owner, producer, consumer, density, gradient)
        np.testing.assert_allclose(
            recovered,
            _reference(points, weights, density, gradient, spec),
            rtol=2e-11,
            atol=1e-12,
        )
