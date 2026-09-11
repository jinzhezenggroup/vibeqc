"""Scheduled CUDA prefix reuse, same-approximation consumers and owner checks."""

import os
from pathlib import Path

import numpy as np
import pytest
from test_low_rank import DenseColumns, factors
from test_low_rank_consumers import dense_approximation
from test_low_rank_source import source_for
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.resources import ResourceBudget

from tools.vibeqc_posthf.coulomb_columns import CoulombColumns
from tools.vibeqc_posthf.low_rank import IncrementalCholesky
from tools.vibeqc_posthf.low_rank_consumers import LowRankProvider
from tools.vibeqc_posthf.low_rank_cuda import (
    CudaIncrementalCholesky,
    compile_cholesky_cuda,
)
from tools.vibeqc_posthf.sources import pointer


@pytest.fixture(scope="module")
def artifact(tmp_path_factory):
    if os.environ.get("VIBEQC_TEST_LOW_RANK_CUDA") != "1":
        pytest.skip("set VIBEQC_TEST_LOW_RANK_CUDA=1 in a scheduled GPU job")
    if not os.environ.get("SLURM_JOB_ID"):
        pytest.fail("low-rank real GPU tests require Slurm")
    compiler = CudaCompilerAdapter(
        Path("/group/software/cuda-12.9.1/bin/nvcc"), cuda_target_info("sm_120")
    )
    return compile_cholesky_cuda(
        compiler, tmp_path_factory.mktemp("low-rank-cuda-cache")
    )


def test_cuda_budget_preflight_never_reads_source_or_loads_a_device():
    source = DenseColumns(np.eye(3), 2)
    with pytest.raises(MemoryError):
        CudaIncrementalCholesky(
            source, None, rank_capacity=3, budget=ResourceBudget(device_bytes=0)
        )
    assert source.calls == []


@pytest.mark.parametrize("spins", [1, 2])
def test_resident_cuda_prefix_and_jk_match_independent_dense_tensor(artifact, spins):
    rng = np.random.default_rng(19)
    raw = rng.normal(size=(6, 6))
    matrix = raw @ raw.T + np.eye(6) * 0.1
    with CudaIncrementalCholesky(
        DenseColumns(matrix, 3), artifact, rank_capacity=6, pair_tile=4
    ) as factor:
        factor.refine(0, maximum_rank=2)
        old = factors(factor).copy()
        density = rng.normal(size=(spins, 3, 3))
        density += density.swapaxes(1, 2)
        argument = density[0] if spins == 1 else density
        provider = LowRankProvider(factor)
        result = provider.jk(argument)
        tensor = dense_approximation(factor)
        np.testing.assert_allclose(
            result.coulomb,
            np.einsum("mnrs,rs->mn", tensor, density.sum(axis=0)),
            atol=3e-13,
        )
        expected = np.einsum("mrns,xrs->xmn", tensor, density)
        np.testing.assert_allclose(
            result.exchange, expected[0] if spins == 1 else expected, atol=3e-13
        )
        assert result.diagnostics["consumer_backend"] == "cuda"
        assert result.diagnostics["native_metrics"]["owned_device_bytes"] > 0
        assert factor.refine(1e-13).status == "threshold_met"
        np.testing.assert_array_equal(factors(factor)[:2], old)
        np.testing.assert_allclose(
            factors(factor).T @ factors(factor), matrix, atol=3e-14
        )
        with pytest.raises(ValueError, match="generation changed"):
            provider.jk(argument)
        # Native rank/stride validation leaves the caller's publication alone.
        out = np.full((spins + 1, 3, 3), 19.0)
        with pytest.raises(RuntimeError, match="generation differs"):
            factor._native.call(
                "posthf_cholesky_jk_v1",
                factor._native._handle,
                factor.rank - 1,
                pointer(density),
                spins,
                pointer(out),
                out.size,
            )
        np.testing.assert_array_equal(out, 19.0)
        replay = LowRankProvider(factor).jk(argument)
        tensor = dense_approximation(factor)
        np.testing.assert_allclose(
            replay.coulomb,
            np.einsum("mnrs,rs->mn", tensor, density.sum(axis=0)),
            atol=3e-13,
        )
        repeated = LowRankProvider(factor).jk(argument)
        np.testing.assert_array_equal(repeated.exchange, replay.exchange)
    with pytest.raises(RuntimeError, match="closed"):
        factor.refine(0)


def test_cuda_native_source_extension_and_zero_rank(artifact):
    source, arrays = source_for("water")
    with (
        source,
        CudaIncrementalCholesky(
            CoulombColumns(source), artifact, rank_capacity=28, pair_tile=5
        ) as factor,
    ):
        zero = LowRankProvider(factor).jk(np.eye(source.nbf))
        np.testing.assert_array_equal(zero.coulomb, 0)
        np.testing.assert_array_equal(zero.exchange, 0)
        factor.refine(1e-11, maximum_rank=3)
        prefix = factors(factor).copy()
        factor.refine(1e-11)
        np.testing.assert_array_equal(factors(factor)[:3], prefix)
        actual = LowRankProvider(factor).jk(np.eye(source.nbf))
        np.testing.assert_allclose(
            actual.coulomb, np.einsum("mnrr->mn", arrays["ao"]), atol=3e-11, rtol=1e-10
        )
        np.testing.assert_allclose(
            actual.exchange, np.einsum("mrnr->mn", arrays["ao"]), atol=3e-11, rtol=1e-10
        )
        cpu = IncrementalCholesky(CoulombColumns(source), rank_capacity=28, pair_tile=2)
        cpu.refine(1e-11)
        assert [row.pair_index for row in factor.history] == [
            row.pair_index for row in cpu.history
        ]
        np.testing.assert_allclose(factors(factor), factors(cpu), atol=1e-12)


def test_cuda_rejected_schur_column_preserves_native_prefix(artifact):
    source = DenseColumns([[1, 2, 0], [2, 1, 0], [0, 0, 1]], 2)
    with CudaIncrementalCholesky(source, artifact, rank_capacity=3) as factor:
        with pytest.raises(ValueError, match="PSD Cauchy"):
            factor.refine(0)
        assert factor.rank == 0
        np.testing.assert_array_equal(LowRankProvider(factor).jk(np.eye(2)).exchange, 0)


def test_cuda_staged_initializer_matches_exact_cleanup_under_budget(artifact):
    from test_low_rank_refinement import basis_for
    from vibeqc.fock import FockPlan

    from tools.vibeqc_posthf.low_rank_refinement import (
        RefinementStage,
        solve_refined_rhf,
    )

    source, _ = source_for("water")
    budget = ResourceBudget(host_bytes=9 << 20, device_bytes=101 << 20)
    with (
        source,
        basis_for(source) as basis,
        FockPlan(basis, screening_tolerance=0) as target,
        CudaIncrementalCholesky(
            CoulombColumns(source),
            artifact,
            rank_capacity=28,
            pair_tile=5,
            budget=budget,
        ) as factor,
    ):
        expected = target.solve(compute_forces=False)
        actual = solve_refined_rhf(
            factor,
            target,
            [RefinementStage(0.1, 2, 3), RefinementStage(1e-6, 4)],
            compute_forces=False,
        )
        np.testing.assert_allclose(
            actual.exact.energy, expected.energy, atol=1e-10, rtol=0
        )
        np.testing.assert_allclose(
            actual.exact.density, expected.density, atol=2e-7, rtol=0
        )
        assert (
            actual.stages[1]["transition"]["rank"]
            > actual.stages[0]["transition"]["rank"]
        )
