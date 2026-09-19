"""Generated implicit VJP executed on an explicitly allocated real GPU.

The Krylov controller is #179's host implementation, never mislabeled as a
native/device-resident solver. No native molecular force capability is granted.
"""

import os
from pathlib import Path

import numpy as np
import pytest
from implicit_fixtures import rank_one_problem
from vibeqc.profiles import find_nvcc
from vibeqc_compiler.common.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.common.cuda_target import cuda_target_info
from vibeqc_compiler.common.resources import ResourceBudget

from tools.vibeqc_response import GMRESOptions, ResponseCompatibilityError
from tools.vibeqc_response.implicit import ImplicitSolveError, ResponseGMRES
from tools.vibeqc_response.implicit_cuda import PreparedImplicitCuda

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_IMPLICIT_CUDA_TEST") != "1",
    reason="requires explicit allocated-GPU opt-in",
)


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    assert os.environ.get("SLURM_JOB_ID"), "implicit CUDA tests require Slurm"
    nvcc = find_nvcc()
    assert nvcc is not None, "set VIBEQC_NVCC to the supported allocated-GPU compiler"
    compiler = CudaCompilerAdapter(
        nvcc, cuda_target_info(os.environ.get("VIBEQC_TENSOR_ARCH", "sm_120"))
    )
    spec, _ = rank_one_problem()
    solver = ResponseGMRES(
        spec.dimension,
        GMRESOptions(rtol=1e-11, atol=1e-13, restart=17, max_iterations=100),
    )
    cache = (
        Path(os.environ["VIBEQC_IMPLICIT_CACHE"])
        if "VIBEQC_IMPLICIT_CACHE" in os.environ
        else tmp_path_factory.mktemp("implicit-cuda")
    )
    with PreparedImplicitCuda(
        spec.compile(),
        solver,
        compiler,
        cache,
        budget=ResourceBudget(host_bytes=64 << 20, device_bytes=1 << 30),
    ) as executor:
        yield executor


@pytest.mark.parametrize("changed", [False, True])
def test_real_cuda_implicit_vjp_same_graph_true_residual_and_no_cpu_fallback(
    prepared, changed, monkeypatch
):
    import tools.vibeqc_response.implicit as runtime

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU scientific fallback executed")

    monkeypatch.setattr(runtime, "execute", forbidden)
    spec, feeds = rank_one_problem()
    if changed:
        feeds["x"] = feeds["x"] + 0.01
        feeds["q"] = (
            feeds["diagonal"] * feeds["x"]
            + feeds["u"] * (feeds["v"] @ feeds["x"])
            + 0.2 * feeds["x"] ** 2
        )
    reference = f"root-{changed}"
    bound = prepared.bind(feeds, reference_identity=reference)
    seed = np.linspace(-0.5, 0.7, spec.dimension)
    direct = np.linspace(0.2, -0.1, spec.dimension)
    result = bound.vjp(seed, reference_identity=reference, direct={"q": direct})
    # This dense Jacobian is an independent *test-only* oracle. Every emitted
    # program contains only O(n) vectors/reductions, not any n-by-n input/node.
    jacobian = np.diag(feeds["diagonal"] + 0.4 * feeds["x"]) + np.outer(
        feeds["u"], feeds["v"]
    )
    expected = (
        np.linalg.solve(jacobian.T, np.asarray(spec.state_metric) * seed) + direct
    )
    np.testing.assert_allclose(
        result.parameter_cotangents["q"], expected, rtol=2e-10, atol=2e-11
    )
    assert result.adjoint_residual_norm < 1e-10
    assert result.plan_identity == spec.compile().identity == prepared.plan_identity
    assert result.tensor_backend == "cuda-fp64-ordinary-stream"
    assert result.solver_backend == "python-response-gmres"
    for program in prepared.plan.programs.values():
        assert all(node.spec.size <= spec.dimension for node in program.live_nodes)
    left = np.linspace(0.1, -0.3, spec.dimension)
    right = np.linspace(-0.2, 0.4, spec.dimension)
    jv = prepared.execute("jacobian", {**feeds, "__implicit_vector": right}).outputs[
        "value"
    ]
    jtl = prepared.execute("transpose", {**feeds, "__implicit_vector": left}).outputs[
        "value"
    ]
    np.testing.assert_allclose(left @ jv, jtl @ right, rtol=1e-11, atol=1e-12)
    assert set(prepared.last_metrics) == set(prepared.plan.programs)
    for space in ("device", "host"):
        measured = sum(
            metrics[f"tracked_{space}_bytes"]
            for metrics in prepared.last_metrics.values()
        )
        assert measured <= prepared.resource_plan.peak_bytes[space]
    with pytest.raises(ResponseCompatibilityError, match="reference"):
        bound.vjp(seed, reference_identity="stale")


def test_cuda_primal_failure_does_not_poison_next_bound_state(prepared):
    spec, feeds = rank_one_problem()
    bad = {**feeds, "q": feeds["q"] + 1.0}
    with pytest.raises(ImplicitSolveError, match="primal is not converged"):
        prepared.bind(bad, reference_identity="bad")
    result = prepared.bind(feeds, reference_identity="good").vjp(
        np.ones(spec.dimension), reference_identity="good"
    )
    assert result.adjoint_residual_norm < 1e-10


def test_cuda_adjoint_nonconvergence_is_not_published(prepared):
    from tools.vibeqc_response.implicit import BoundImplicitState

    spec, feeds = rank_one_problem()
    solver = ResponseGMRES(
        spec.dimension, GMRESOptions(rtol=1e-15, max_iterations=1, restart=1)
    )
    bound = BoundImplicitState(
        prepared.plan,
        feeds,
        reference_identity="short-solve",
        solver=solver,
        executor=prepared,
        max_bytes=prepared.host_capacity,
    )
    with pytest.raises(ImplicitSolveError, match="adjoint failed"):
        bound.vjp(np.ones(spec.dimension), reference_identity="short-solve")
