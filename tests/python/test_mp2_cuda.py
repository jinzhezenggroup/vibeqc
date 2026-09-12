"""Opt-in A1 tile numerics on an allocated GPU; not public-method acceptance."""

import os
from pathlib import Path

import numpy as np
import pytest

from tools.vibeqc_codegen.cuda_adapter import CudaCompilerAdapter
from tools.vibeqc_codegen.cuda_target import cuda_target_info
from tools.vibeqc_mp2 import PreparedMP2Energy
from tools.vibeqc_posthf.cuda import compile_cuda
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.sources import NativeSource

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1",
    reason="DEVICE_VALIDATION_PENDING: requires explicitly allocated real CUDA GPU",
)


@pytest.fixture(scope="module")
def cuda_setup(tmp_path_factory):
    # Explicit allocation identity for Slurm or an authorized platform Notebook.
    # This test neither acquires resources nor manufactures a Slurm job identity.
    assert os.environ.get("SLURM_JOB_ID") or os.environ.get(
        "VIBEQC_MP2_GPU_ALLOCATION"
    ), "run inside an explicitly allocated GPU job or Notebook"
    compiler = CudaCompilerAdapter(
        Path(os.environ["VIBEQC_NVCC"]),
        cuda_target_info(os.environ.get("VIBEQC_MP2_ARCH", "sm_120")),
    )
    cache = tmp_path_factory.mktemp("mp2-cuda-cache")
    return compiler, cache, compile_cuda(compiler, cache / "transform")


@pytest.mark.parametrize("name", ["h2", "water", "lih"])
@pytest.mark.parametrize("tiles", [(1, 2), (2, 3)])
def test_native_cuda_tile_components_replay_and_failures(cuda_setup, name, tiles):
    compiler, cache, artifact = cuda_setup
    meta, a = load_fixture(name)
    s = fixture_snapshot(meta, a)
    # Independent pinned PySCF amplitudes and MO elements: this reference
    # never calls either the new TensorIR program or its CPU interpreter.
    ovov = a["conventional_mo"][
        np.ix_(range(s.nocc), range(s.nocc, s.nmo), range(s.nocc), range(s.nocc, s.nmo))
    ]
    g, t = ovov.transpose(0, 2, 1, 3), a["conventional_t2"]
    os_ref = float(np.sum(t * g))
    ss_ref = float(np.sum(t * (g - g.swapaxes(2, 3))))
    with (
        NativeSource(**source_arguments(meta)) as source,
        PreparedMP2Energy(
            s,
            source,
            occupied_tile=tiles[0],
            virtual_tile=tiles[1],
            energy_backend="cuda",
            integral_backend="cuda",
            compiler=compiler,
            cache=cache / "equations",
            transform_artifact=artifact,
        ) as p,
    ):
        for _ in range(2):
            r = p.execute()
            np.testing.assert_allclose(
                [r.opposite_spin, r.same_spin],
                [os_ref, ss_ref],
                atol=1e-11,
                rtol=1e-10,
            )
            assert (
                abs(
                    r.correlation_energy
                    - meta["records"]["conventional"]["correlation_energy"]
                )
                <= 1e-9
            )
            assert r.energy_backend == "native-cuda-tile"
            assert r.integral_backend == "cuda" and r.mo_host_staging
            assert r.numeric_capacity_bytes <= 256 << 20
        with pytest.raises(NotImplementedError, match="energy only"):
            p.execute(properties=("forces",))
        assert p.state == "failed" and p.last_result is None
