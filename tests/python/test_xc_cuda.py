"""Opt-in generated FP64 CUDA gates; run only in a finite Slurm GPU job."""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_validation.schema import block_error
from tools.vibeqc_xc import UnsupportedXC, build_program, functional
from tools.vibeqc_xc.capabilities import query_capability
from tools.vibeqc_xc.cuda import CudaXC, compile_cuda
from tools.vibeqc_xc.cuda_emit import XCSchedule
from tools.vibeqc_xc.fixtures import load_fixture
from tools.vibeqc_xc.spec import CATALOG

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_XC_CUDA_TEST") != "1", reason="opt-in Slurm CUDA gate"
)


@lru_cache(maxsize=64)
def compiled(name, spin, variant, order=2, outputs=None):
    program = build_program(functional(name, spin=spin), order=order, outputs=outputs)
    artifact = compile_cuda(
        program,
        CudaCompilerAdapter(
            Path(os.environ.get("VIBEQC_NVCC", "/group/software/cuda-12.9.1/bin/nvcc")),
            cuda_target_info("sm_120"),
        ),
        os.environ.get("VIBEQC_XC_CACHE", "/tmp/xc161-cuda-cache"),
        schedule=XCSchedule(variant),
    )
    return program, artifact


def check(actual, expected, tolerance):
    result = block_error(actual, expected, **tolerance)
    assert result["passed"], result


@pytest.mark.parametrize("name", CATALOG)
@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
@pytest.mark.parametrize("variant", ["baseline", "fused", "split"])
def test_every_kernel_value_derivative_boundary_and_partial_tile(name, spin, variant):
    program, artifact = compiled(name, spin, variant)
    for capacity in (7, 31):
        with CudaXC(program, artifact, tile_points=capacity) as device:
            for domain in ("typical", "boundary"):
                meta, x, expected, _ = load_fixture(name, spin=spin, domain=domain)
                tolerance = meta[f"{domain}_tolerance"]
                result = np.concatenate(
                    [
                        device.evaluate(x[:, i : i + capacity])
                        for i in range(0, x.shape[1], capacity)
                    ],
                    axis=1,
                )
                check(result, expected, tolerance)
                check(result, program.evaluate(x), tolerance)
            metrics = device.metrics()
            assert metrics["owned_device_bytes"] == device.plan.device_bytes
            assert metrics["provider_retained_bytes"] == 0
            assert metrics["kernel_ms"] > 0
            assert metrics["library_ms"] == 0
            assert metrics["input_ms"] > 0 and metrics["output_ms"] > 0
            assert device.evaluate(np.empty((len(program.spec.features), 0))).shape == (
                len(program.outputs),
                0,
            )
        with pytest.raises(RuntimeError, match="closed"):
            device.evaluate(x[:, :1])


def test_replay_changed_input_capacity_failure_and_independent_threads():
    program, artifact = compiled("PBE", "polarized", "split")
    _, x, expected, _ = load_fixture("PBE")
    tolerance = {"atol": 1e-11, "rtol": 1e-10}
    with pytest.raises(ValueError, match="budget"):
        CudaXC(program, artifact, budget_bytes=1)
    with CudaXC(program, artifact, tile_points=7) as device:
        with pytest.raises(ValueError, match="tile shape"):
            device.evaluate(x)
        with pytest.raises(UnsupportedXC):
            device.evaluate(np.zeros((7, 1)))
        for tile in (x[:, :7], x[:, -5:], x[:, :7]):
            check(device.evaluate(tile), program.evaluate(tile), tolerance)
        cap = query_capability(program, schedule=XCSchedule("split"), artifact=artifact)
        assert cap["compiled"] and not cap["validated"] and not cap["promoted"]

    def worker(begin):
        with CudaXC(program, artifact, tile_points=7) as device:
            return device.evaluate(x[:, begin : begin + 7])

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(worker, (0, 7)))
    check(np.concatenate(result, axis=1), expected[:, :14], tolerance)


def test_vacuum_energy_and_pruned_outputs_and_binary_contract():
    program, artifact = compiled("PBE", "polarized", "fused", order=0)
    with CudaXC(program, artifact, tile_points=7) as device:
        check(
            device.evaluate(np.zeros((7, 2))),
            np.zeros((1, 2)),
            {"atol": 1e-11, "rtol": 1e-10},
        )
    pruned, binary = compiled("PBE", "polarized", "fused", outputs=((0,), (2, 3)))
    _, x, _, _ = load_fixture("PBE")
    with CudaXC(pruned, binary, tile_points=31) as device:
        check(
            device.evaluate(x[:, :31]),
            pruned.evaluate(x[:, :31]),
            {"atol": 1e-11, "rtol": 1e-10},
        )
    with pytest.raises(ValueError, match="program mismatch"):
        CudaXC(pruned, artifact)
