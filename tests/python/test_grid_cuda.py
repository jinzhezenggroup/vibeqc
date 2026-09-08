"""Opt-in real-device gates; callers must use a finite Slurm GPU allocation."""

import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_dft import GridSpec, NativeAO
from tools.vibeqc_dft.cuda import CudaGrid, compile_cuda
from tools.vibeqc_dft.fixtures import NAMES, basis_arguments, load_fixture
from tools.vibeqc_dft.prepared import PreparedGrid, PreparedGridBatch
from tools.vibeqc_validation.schema import block_error

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_GRID_CUDA_TEST") != "1", reason="opt-in Slurm CUDA gate"
)


@pytest.fixture(scope="module")
def artifact():
    return compile_cuda(
        CudaCompilerAdapter(
            Path(os.environ.get("VIBEQC_NVCC", "/group/software/cuda-12.9.1/bin/nvcc")),
            cuda_target_info("sm_120"),
        ),
        Path("/tmp/dft160-cuda-cache"),
    )


def check(actual, expected):
    result = block_error(actual, expected, atol=1e-11, rtol=1e-10)
    assert result["passed"], result


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("tile_points", [7, 31])
def test_all_jets_features_partial_tiles_and_resident_density(
    artifact, name, tile_points
):
    meta, arrays = load_fixture(name)
    with NativeAO(**basis_arguments(meta)) as basis:
        with CudaGrid(basis, artifact, order=3, tile_points=tile_points) as cuda:
            with pytest.raises(ValueError, match="supplied density"):
                cuda.evaluate(arrays["points"][:1])
            cuda.set_density(arrays["density"])
            for begin in range(0, len(arrays["points"]), tile_points):
                end = begin + tile_points
                result = cuda.evaluate(arrays["points"][begin:end], download_jets=True)
                for key, value in result.items():
                    expected = arrays[key][:, begin:end]
                    check(value, expected)
            first = cuda.evaluate(arrays["points"][-7:])
            cuda.set_density(2 * arrays["density"])
            second = cuda.evaluate(arrays["points"][-7:])
            for key in first:
                check(second[key], first[key] * (4 if key == "sigma" else 2))
            metrics = cuda.metrics()
            assert metrics["owned_device_bytes"] == cuda.plan.allocation_bytes
            assert metrics["provider_retained_bytes"] <= cuda.plan.provider_bytes
            assert metrics["kernel_ms"] > 0 and metrics["library_ms"] > 0
            assert metrics["input_ms"] > 0 and metrics["output_ms"] > 0
            assert cuda.evaluate(np.empty((0, 3)), download_jets=True)[
                "ao_jets"
            ].shape == (20, 0, basis.nao)
        with pytest.raises(RuntimeError, match="closed"):
            cuda.evaluate(arrays["points"][:1])


def test_orders_zero_to_three_and_budget_rejection(artifact):
    meta, arrays = load_fixture("f_spherical")
    with NativeAO(**basis_arguments(meta)) as basis:
        with pytest.raises(ValueError, match="budget"):
            CudaGrid(basis, artifact, budget_bytes=1)
        for order, jets in enumerate((1, 4, 10, 20)):
            with CudaGrid(basis, artifact, order=order, tile_points=13) as cuda:
                actual = cuda.evaluate(
                    arrays["points"][-13:], features=False, download_jets=True
                )
                check(actual["ao_jets"], arrays["ao_jets"][:jets, -13:])
                with pytest.raises(ValueError, match="tile shape"):
                    cuda.evaluate(arrays["points"])


def test_prepared_reuse_changed_geometry_and_ragged_failures(artifact):
    items, densities = [], []
    for name in ("h2", "water", "f_spherical"):
        meta, arrays = load_fixture(name)
        items.append(
            {
                **basis_arguments(meta),
                "spec": GridSpec(3, 3, 6),
                "tile_points": 13,
                "order": 3,
                "artifact": artifact,
                "backend": "cuda",
            }
        )
        densities.append(arrays["density"])
    with PreparedGridBatch(items) as gpu:
        first = gpu.execute(densities)
        again = gpu.execute(densities)
        for index, item in enumerate(items):
            with PreparedGrid(**{**item, "backend": "cpu"}) as cpu:
                reference = cpu.integrate(densities[index])
                for key in ("electrons", "integrated_tau"):
                    check(first[index]["result"][key], reference[key])
                    check(again[index]["result"][key], reference[key])
        assert [r["point_begin"] for r in first] == [0, 108, 270]
        failure = gpu.execute(
            [densities[0], np.full_like(densities[1], np.nan), densities[2]]
        )
        assert [r["status"] for r in failure] == ["pass", "fail", "pass"]
        invalid_type = gpu.execute([densities[0], {"invalid": 1}, densities[2]])
        assert [r["status"] for r in invalid_type] == ["pass", "fail", "pass"]
    with PreparedGrid(**items[0]) as plan:
        first = plan.integrate(densities[0])
        xyz = np.array(items[0]["atoms"][1][1]) + [0.07, -0.02, 0.03]
        coordinates = [items[0]["atoms"][0][1], xyz]
        plan.reconfigure(coordinates=coordinates)
        changed = plan.integrate(densities[0])
        assert changed["identity"] != first["identity"]
        with PreparedGrid(
            **{**items[0], "atoms": [(1, r) for r in coordinates], "backend": "cpu"}
        ) as cpu:
            expected = cpu.integrate(densities[0])
            check(changed["electrons"], expected["electrons"])
        with pytest.raises(ValueError):
            plan.reconfigure(spec=replace(plan.grid.spec, radial_points=0))
        check(plan.integrate(densities[0])["electrons"], changed["electrons"])


def test_shared_posthf_runtime_after_cache_extraction():
    """Exercise the existing cuBLAS MO consumer after extracting shared caching."""
    from tools.vibeqc_posthf.conventions import MOBlock
    from tools.vibeqc_posthf.cuda import compile_cuda as compile_posthf
    from tools.vibeqc_posthf.fixtures import fixture_snapshot, source_arguments
    from tools.vibeqc_posthf.fixtures import load_fixture as load_posthf
    from tools.vibeqc_posthf.providers import ConventionalProvider
    from tools.vibeqc_posthf.sources import NativeSource

    artifact = compile_posthf(
        CudaCompilerAdapter(
            Path("/group/software/cuda-12.9.1/bin/nvcc"), cuda_target_info("sm_120")
        ),
        Path("/tmp/dft160-posthf-cache"),
    )
    meta, arrays = load_posthf("h2")
    reference = fixture_snapshot(meta, arrays)
    with (
        NativeSource(**source_arguments(meta)) as source,
        ConventionalProvider(
            reference, source, backend="cuda", cuda_artifact=artifact
        ) as provider,
    ):
        result = provider.get(MOBlock.from_spaces(reference, "ovov"))
        check(result.to_host(), arrays["conventional_mo"][0:1, 1:2, 0:1, 1:2])


def test_gpu_iterator_density_isolation_and_failed_update(artifact):
    meta, data = load_fixture("h2")
    options = {
        **basis_arguments(meta),
        "spec": GridSpec(3, 3, 6),
        "tile_points": 7,
        "backend": "cuda",
        "artifact": artifact,
    }
    with PreparedGrid(**options) as plan:
        first = plan.iter_features(data["density"])
        original = next(first)
        second = plan.iter_features(data["density"] * 2)
        doubled = next(second)
        check(doubled.features["rho"], 2 * original.features["rho"])
        with pytest.raises(RuntimeError, match="stale"):
            next(first)
        identity = plan.identity
        with pytest.raises(ValueError, match="budget"):
            plan.reconfigure(budget_bytes=plan.plan.peak_bytes)
        assert plan.identity == identity
        assert np.isfinite(next(second).features["rho"]).all()
