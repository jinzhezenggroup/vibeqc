"""Matrix-free RHF response actions against explicit and finite-rotation oracles."""

from dataclasses import replace

import numpy as np
import pytest

from tools.vibeqc_posthf.fixtures import (
    fixture_snapshot,
    load_fixture,
    source_arguments,
)
from tools.vibeqc_posthf.providers import ConventionalProvider
from tools.vibeqc_posthf.sources import NativeSource
from tools.vibeqc_response import (
    DenseAOResponseBackend,
    GMRESOptions,
    NativeJKBackend,
    RHFResponseOperator,
    explicit_rhf_response_matrix,
    finite_rotation_jvp,
    solve_many,
)


@pytest.mark.parametrize("name", ["h2", "water"])
def test_matrix_free_jvp_matches_explicit_mo_matrix_and_transpose_identity(name):
    meta, arrays = load_fixture(name)
    snapshot = fixture_snapshot(meta, arrays)
    try:
        source = NativeSource(**source_arguments(meta))
    except (OSError, FileNotFoundError, AttributeError) as error:
        pytest.skip(f"native post-HF library unavailable: {error}")
    try:
        with ConventionalProvider(snapshot, source) as provider:
            backend = DenseAOResponseBackend(arrays["ao"])
            problem = RHFResponseOperator.build_problem(snapshot, backend)
            explicit = explicit_rhf_response_matrix(problem, provider)
            operator = RHFResponseOperator(problem, backend)
            dense = operator.to_dense()
            np.testing.assert_allclose(dense, explicit, atol=2e-10, rtol=2e-10)
            rng = np.random.default_rng(179)
            left = rng.normal(size=problem.dimension)
            right = rng.normal(size=problem.dimension)
            np.testing.assert_allclose(
                operator.apply(left), explicit @ left, atol=2e-10, rtol=2e-10
            )
            assert operator.dot_identity(left, right) < 1e-12
            native_backend = NativeJKBackend(source, axis_tile=2)
            native_problem = RHFResponseOperator.build_problem(snapshot, native_backend)
            native_operator = RHFResponseOperator(native_problem, native_backend)
            np.testing.assert_allclose(
                native_operator.apply(left),
                explicit @ left,
                atol=2e-10,
                rtol=2e-10,
            )
            assert native_operator.dot_identity(left, right) < 1e-12
            assert native_backend.statistics["actions"] >= 2
    finally:
        source.close()


@pytest.mark.parametrize("name", ["h2", "water"])
def test_finite_orbital_rotation_matches_jvp(name):
    meta, arrays = load_fixture(name)
    snapshot = fixture_snapshot(meta, arrays)
    backend = DenseAOResponseBackend(arrays["ao"])
    problem = RHFResponseOperator.build_problem(snapshot, backend)
    operator = RHFResponseOperator(problem, backend)
    rng = np.random.default_rng(179)
    vector = rng.normal(size=problem.dimension)
    finite = finite_rotation_jvp(problem, backend, vector, step=1e-6)
    np.testing.assert_allclose(finite, operator.apply(vector), atol=3e-8, rtol=3e-8)


def test_native_backend_rejects_nonconventional_source():
    meta, arrays = load_fixture("h2")
    del arrays
    try:
        source = NativeSource(**source_arguments(meta))
    except (OSError, FileNotFoundError, AttributeError) as error:
        pytest.skip(f"native post-HF library unavailable: {error}")
    try:
        with pytest.raises(NotImplementedError, match="CPU-streamed"):
            NativeJKBackend(source, backend="cuda")
    finally:
        source.close()


def test_native_backend_rejects_same_sized_unrelated_reference():
    meta, arrays = load_fixture("h2")
    snapshot = fixture_snapshot(meta, arrays)
    try:
        source = NativeSource(**source_arguments(meta))
    except (OSError, FileNotFoundError, AttributeError) as error:
        pytest.skip(f"native post-HF library unavailable: {error}")
    try:
        backend = NativeJKBackend(source)
        problem = RHFResponseOperator.build_problem(snapshot, backend)
        bad_reference = replace(snapshot, geometry_hash="different-geometry")
        with pytest.raises(ValueError, match="geometry mismatch"):
            RHFResponseOperator(replace(problem, reference=bad_reference), backend)
    finally:
        source.close()


def test_native_rhf_multirhs_residuals_permutation_and_recycling():
    meta, arrays = load_fixture("water")
    snapshot = fixture_snapshot(meta, arrays)
    try:
        source = NativeSource(**source_arguments(meta))
    except (OSError, FileNotFoundError, AttributeError) as error:
        pytest.skip(f"native post-HF library unavailable: {error}")
    try:
        with ConventionalProvider(snapshot, source) as provider:
            native_backend = NativeJKBackend(source, axis_tile=2)
            problem = RHFResponseOperator.build_problem(snapshot, native_backend)
            operator = RHFResponseOperator(problem, native_backend)
            explicit = explicit_rhf_response_matrix(problem, provider)
            rng = np.random.default_rng(179)
            rhs = rng.normal(size=(problem.dimension, 2))
            options = GMRESOptions(
                rtol=1e-11, atol=1e-12, restart=10, max_iterations=120
            )
            sequential = solve_many(
                operator, rhs, strategy="sequential", options=options
            )
            blocked = solve_many(operator, rhs, strategy="blocked", options=options)
            recycled = solve_many(operator, rhs, strategy="recycled", options=options)
            expected = np.linalg.solve(explicit, rhs)
            for result in (sequential, blocked, recycled):
                assert result.converged
                np.testing.assert_allclose(
                    result.solution, expected, atol=2e-8, rtol=2e-8
                )
                assert max(item.residual_norm for item in result.results) < 1e-9
            assert any(item.recycled_vectors > 0 for item in recycled.results[1:])
            permutation = (1, 0)
            permuted = solve_many(
                operator,
                rhs[:, permutation],
                strategy="recycled",
                options=options,
            )
            np.testing.assert_allclose(
                permuted.solution,
                sequential.solution[:, permutation],
                atol=2e-8,
                rtol=2e-8,
            )
    finally:
        source.close()


@pytest.mark.parametrize("mismatch", ["threshold", "auxiliary", "label"])
def test_cuda_df_metric_preflight_rejects_stale_identity_without_device(mismatch):
    """Metric identity errors must fail even with a CPU-only native library."""
    from tools.vibeqc_posthf.df import MetricFactor
    from tools.vibeqc_response import CudaDFJKBackend

    meta, _ = load_fixture("h2")
    try:
        source = NativeSource(**source_arguments(meta))
    except (OSError, FileNotFoundError, AttributeError) as error:
        pytest.skip(f"native post-HF library unavailable: {error}")
    with source:
        metric = MetricFactor.from_source(source)
        kwargs = {"metric": metric, "hamiltonian_id": metric.hamiltonian_id}
        if mismatch == "threshold":
            kwargs["metric_threshold"] = metric.relative_threshold * 10
        elif mismatch == "auxiliary":
            kwargs["metric"] = replace(metric, auxiliary_hash="another-auxiliary-basis")
        else:
            kwargs["hamiltonian_id"] = "density-fitting:another-metric"
        with pytest.raises(ValueError, match="metric.*mismatch|metric mismatch"):
            CudaDFJKBackend(source, **kwargs)
