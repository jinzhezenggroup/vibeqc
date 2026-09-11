"""Real-device tier; opt in only inside an allocated GPU job."""

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info

from tools.vibeqc_posthf.conventions import MOBlock
from tools.vibeqc_posthf.cuda import compile_cuda
from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.mp2 import restricted_mp2
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import CudaDFSource, NativeSource

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_POSTHF_CUDA_TEST") != "1",
    reason="requires explicitly allocated real GPU",
)


@pytest.fixture(scope="module")
def artifact():
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm on this host"
    compiler = CudaCompilerAdapter(
        Path(os.environ.get("VIBEQC_NVCC", "/group/software/cuda-12.9.1/bin/nvcc")),
        cuda_target_info("sm_120"),
    )
    return compile_cuda(compiler, Path("/tmp/posthf147-cuda-cache"))


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
@pytest.mark.parametrize("tile", [1, 2, 4])
def test_cuda_mo_blocks_mp2_and_owned_memory(artifact, name, tile):
    meta, a = load_fixture(name)
    snapshot = fixture_snapshot(meta, a)
    with NativeSource(**source_arguments(meta)) as source:
        with ConventionalProvider(
            snapshot,
            source,
            backend="cuda",
            axis_tile=tile,
            cuda_artifact=artifact,
            budget_bytes=512 << 20,
        ) as provider:
            for spaces in ("ovov", "oovv", "ovvv"):
                block = MOBlock.from_spaces(snapshot, spaces)
                result = provider.get(block)
                np.testing.assert_allclose(
                    result.to_host(),
                    a["conventional_mo"][np.ix_(*block.slots)],
                    atol=1e-11,
                    rtol=1e-10,
                )
                measured = result.values.metrics()
                plan = provider.plan(block)
                assert measured["owned_device_bytes"] == plan.allocation_bytes
                assert measured["provider_retained_bytes"] <= 96 << 20
                assert result.values.device_pointer
                assert provider.get(block).values is result.values
            result = restricted_mp2(snapshot, provider)
            assert (
                abs(
                    result.correlation_energy
                    - meta["records"]["conventional"]["correlation_energy"]
                )
                < 1e-9
            )
            assert provider.statistics["peak_bytes"] <= provider.budget_bytes
        with pytest.raises(RuntimeError, match="closed"):
            provider.get(block)


def test_independent_cuda_items_and_closed_exports(artifact):
    meta, a = load_fixture("h2")
    snapshot = fixture_snapshot(meta, a)
    with NativeSource(**source_arguments(meta)) as source:
        providers = [
            ConventionalProvider(
                snapshot, source, backend="cuda", cuda_artifact=artifact
            )
            for _ in range(2)
        ]
        block = MOBlock.from_spaces(snapshot, "ovov")
        with ThreadPoolExecutor(2) as pool:
            outputs = list(pool.map(lambda p: p.get(block), providers))
        assert outputs[0].values.device_pointer != outputs[1].values.device_pointer
        providers[0].close()
        with pytest.raises(RuntimeError, match="closed"):
            outputs[0].to_host()
        np.testing.assert_allclose(
            outputs[1].to_host(), a["conventional_mo"][np.ix_(*block.slots)], atol=1e-11
        )
        providers[1].close()


@pytest.mark.parametrize("name", ["h2", "water", "lih", "f_heh"])
def test_generated_df_source_staging_and_same_hamiltonian(name):
    assert os.environ.get("SLURM_JOB_ID")
    meta, a = load_fixture(name)
    with CudaDFSource(**source_arguments(meta), tile_capacity=64) as source:
        factor = MetricFactor.from_source(source)
        np.testing.assert_allclose(
            source._read("coulomb_metric", (0, 0), (source.naux, source.naux)),
            a["metric"],
            atol=1e-11,
            rtol=1e-10,
        )
        snapshot = fixture_snapshot(meta, a, label="df", metric=factor)
        # Final partial tiles exercise the generated native source, including f.
        shells = (
            (1, 2, 1)
            if name == "f_heh"
            else (0, len(source.shells) - 1, len(source.auxiliary_shells) - 1)
        )
        request = next(
            r
            for r in source.requests("three_center_eri", axis_tile=3)
            if r.shell_indices == shells
            and (name != "f_heh" or r.tile.offsets == (6, 0, 6))
        )
        begin = source.global_offsets(request)
        tile = source.tile(request)
        slices = tuple(slice(b, b + n) for b, n in zip(begin, tile.shape))
        np.testing.assert_allclose(
            tile, a["raw_three_center"][slices], atol=1e-11, rtol=1e-10
        )
        with DFProvider(snapshot, source, factor, auxiliary_tile=3) as provider:
            result = restricted_mp2(snapshot, provider)
            np.testing.assert_allclose(
                result.amplitudes, a["df_t2"], atol=1e-11, rtol=1e-10
            )
            assert (
                abs(
                    result.correlation_energy
                    - meta["records"]["df"]["correlation_energy"]
                )
                < 1e-9
            )
        assert source.source_device_bytes > 0
