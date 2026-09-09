"""Real-device response-operator tier; run only inside an allocated Slurm job."""

import os

import numpy as np
import pytest

from tools.vibeqc_posthf.df import DFProvider, MetricFactor
from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.sources import CudaDFSource
from tools.vibeqc_response import (
    CudaDFJKBackend,
    GMRESOptions,
    RHFResponseOperator,
    explicit_rhf_response_matrix,
    solve_many,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_RESPONSE_CUDA_TEST") != "1",
    reason="requires an explicitly allocated real GPU",
)


def test_cuda_df_matrix_free_rhf_response_matches_explicit_and_solves():
    assert os.environ.get("SLURM_JOB_ID"), "real GPU tests require Slurm"
    meta, arrays = load_fixture("h2")
    with CudaDFSource(**source_arguments(meta), tile_capacity=64) as source:
        metric = MetricFactor.from_source(source)
        snapshot = fixture_snapshot(meta, arrays, label="df", metric=metric)
        with CudaDFJKBackend(
            source,
            metric_threshold=metric.relative_threshold,
            hamiltonian_id=metric.hamiltonian_id,
            metric=metric,
        ) as backend:
            problem = RHFResponseOperator.build_problem(snapshot, backend)
            operator = RHFResponseOperator(problem, backend)
            with DFProvider(snapshot, source, metric) as provider:
                explicit = explicit_rhf_response_matrix(problem, provider)
            rng = np.random.default_rng(179)
            vector = rng.normal(size=problem.dimension)
            np.testing.assert_allclose(
                operator.apply(vector),
                explicit @ vector,
                atol=2e-9,
                rtol=2e-9,
            )
            assert (
                operator.dot_identity(vector, rng.normal(size=problem.dimension))
                < 1e-11
            )
            rhs = rng.normal(size=(problem.dimension, 2))
            options = GMRESOptions(rtol=1e-11, atol=1e-12, restart=8, max_iterations=80)
            expected = np.linalg.solve(explicit, rhs)
            for strategy in ("sequential", "blocked", "recycled"):
                result = solve_many(operator, rhs, strategy=strategy, options=options)
                assert result.converged
                assert max(item.residual_norm for item in result.results) < 1e-9
                np.testing.assert_allclose(
                    result.solution, expected, atol=2e-8, rtol=2e-8
                )
            assert backend.statistics["actions"] > 0
            assert backend.peak_device_bytes > 0
