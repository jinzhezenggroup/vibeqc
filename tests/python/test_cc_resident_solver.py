"""#149-B device-resident RCCSD iteration and transfer contract."""

import os
from pathlib import Path

import numpy as np
import pytest
from test_cc_api import fixture_problem
from vibeqc_compiler.integral.cuda_adapter import CudaCompilerAdapter
from vibeqc_compiler.integral.cuda_target import cuda_target_info
from vibeqc_compiler.tensor.cuda_resident_emit import resident_source

from tools.vibeqc_cc.gpu_state import solver_plans
from tools.vibeqc_cc.resident_solver import (
    PreparedResidentCCSD,
    _resident_extension,
    solve_gpu_resident,
)
from tools.vibeqc_cc.solver import SolverOptions


def test_resident_solver_extension_owns_iteration_state_without_new_equations():
    options = SolverOptions()
    primary, replay, diagnostic = solver_plans(
        2, 3, cuda_target_info("sm_120"), options
    )
    extension = _resident_extension(
        primary, diagnostic["state_segments"], 2 * 3, 2**2 * 3**2, options.diis_size
    )
    source = resident_source(primary, extension=extension)
    assert primary.program.provenance["physical_equation"]
    assert replay.program.logical_hash != primary.program.logical_hash
    for name in (
        "resident_cc_initialize",
        "resident_cc_status",
        "resident_cc_advance_trial",
        "resident_cc_diis",
        "resident_cc_download_amplitudes",
        "vibeqc::cc::residual_partials",
        "vibeqc::cc::diis_gram",
        "vibeqc::cc::diis_combine_slice",
    ):
        assert name in source
    # Iteration-state mutation is post-processing around the exact generated
    # residual/Jacobi DAG, not a second handwritten residual equation.
    assert "singles_residual" not in extension
    assert "doubles_residual" not in extension
    assert "primitive_eri" not in extension


def test_resident_solver_budget_rejection_precedes_integral_reads(
    monkeypatch, tmp_path
):
    snapshot, provider, _, _ = fixture_problem("h2")
    monkeypatch.setattr(
        provider,
        "get",
        lambda *a, **kw: pytest.fail("integral read before budget gate"),
    )
    compiler = CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120"))
    with pytest.raises(ValueError, match="budget exhausted"):
        PreparedResidentCCSD(
            snapshot,
            provider,
            compiler,
            tmp_path,
            provider_peak_bytes=SolverOptions().max_bytes,
        )


def test_resident_solver_requires_explicit_compiler_cache_and_rhf(tmp_path):
    snapshot, provider, _, _ = fixture_problem("h2")
    with pytest.raises(TypeError, match="CudaCompilerAdapter"):
        solve_gpu_resident(snapshot, provider, compiler=None, cache=tmp_path)
    with pytest.raises(TypeError, match="pathlib.Path"):
        solve_gpu_resident(
            snapshot,
            provider,
            compiler=CudaCompilerAdapter(Path("nvcc"), cuda_target_info("sm_120")),
            cache="cache",
        )


_REAL = os.environ.get("VIBEQC_CC_RESIDENT_CUDA_TEST") == "1"


def _compiler_cache(tmp_path):
    from vibeqc.profiles import find_nvcc

    return (
        CudaCompilerAdapter(
            find_nvcc(), cuda_target_info(os.environ["VIBEQC_TENSOR_ARCH"])
        ),
        Path(os.environ.get("VIBEQC_TENSOR_CACHE", tmp_path / "cache")),
    )


@pytest.mark.skipif(
    not _REAL, reason="requires explicitly allocated resident CUDA window"
)
@pytest.mark.parametrize("name", ("h2", "h2o", "ch4"))
def test_real_resident_solver_matches_reference_without_per_iteration_large_transfers(
    name, tmp_path
):
    snapshot, provider, meta, amplitudes = fixture_problem(name)
    compiler, cache = _compiler_cache(tmp_path)
    result = solve_gpu_resident(
        snapshot,
        provider,
        compiler=compiler,
        cache=cache,
        options=SolverOptions(residual_tolerance=1e-10, energy_tolerance=1e-12),
    )
    assert result.converged, (name, result.reason, result.history[-1])
    assert abs(result.total_energy - meta["total_energy"]) <= 1e-8
    assert result.provenance["backend"] == "cuda-fp64-resident"
    np.testing.assert_allclose(result.t1, amplitudes["t1"], atol=1e-8, rtol=1e-8)
    np.testing.assert_allclose(result.t2, amplitudes["t2"], atol=1e-8, rtol=1e-8)
    assert result.provenance["resident_state_identity"]
    transfer = result.provenance["transfer"]
    assert transfer["per_iteration_large_h2d_bytes"] == 0
    assert transfer["per_iteration_large_d2h_bytes"] == 0
    assert transfer["initial_large_h2d_bytes"] > result.t1.nbytes + result.t2.nbytes
    assert transfer["final_amplitude_d2h_bytes"] == result.t1.nbytes + result.t2.nbytes
    assert transfer["runs"] >= len(result.history)
    assert transfer["d2h_bytes"] == 4 * transfer["runs"]
    assert transfer["scalar_control_d2h_bytes"] < 64 * transfer["runs"]
    assert (
        max(
            result.history[-1]["independent_r1_max"],
            result.history[-1]["independent_r2_max"],
        )
        <= 1e-10
    )


@pytest.mark.skipif(
    not _REAL, reason="requires explicitly allocated resident CUDA window"
)
def test_resident_owner_preserves_converged_device_state_for_follow_on_consumers(
    tmp_path,
):
    snapshot, provider, meta, _ = fixture_problem("h2")
    compiler, cache = _compiler_cache(tmp_path)
    with PreparedResidentCCSD(
        snapshot,
        provider,
        compiler,
        cache,
        options=SolverOptions(residual_tolerance=1e-10, energy_tolerance=1e-12),
    ) as prepared:
        provider._closed = True
        provider.source._check_open = lambda: (_ for _ in ()).throw(
            RuntimeError("released fixture provider must not be consulted")
        )
        result = prepared.solve()
        assert (
            result.converged and abs(result.total_energy - meta["total_energy"]) <= 1e-8
        )
        resident = prepared.amplitudes()
        np.testing.assert_allclose(resident[0], result.t1, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(resident[1], result.t2, atol=2e-12, rtol=2e-12)
        assert prepared.state_identity == result.provenance["resident_state_identity"]


@pytest.mark.skipif(
    not _REAL, reason="requires explicitly allocated resident CUDA window"
)
def test_resident_nonconvergence_is_explicit_and_serializable(tmp_path):
    snapshot, provider, _, _ = fixture_problem("ch4")
    compiler, cache = _compiler_cache(tmp_path)
    result = solve_gpu_resident(
        snapshot,
        provider,
        compiler=compiler,
        cache=cache,
        options=SolverOptions(max_iterations=1),
    )
    assert result.status == "not_converged" and not result.converged
    assert np.isfinite(result.t1).all() and np.isfinite(result.t2).all()
    result.write(tmp_path / "resident-nonconverged.json")


@pytest.mark.skipif(
    not _REAL, reason="requires explicitly allocated resident CUDA window"
)
def test_resident_zero_diis_stays_device_resident(tmp_path):
    snapshot, provider, meta, _ = fixture_problem("h2")
    compiler, cache = _compiler_cache(tmp_path)
    result = solve_gpu_resident(
        snapshot,
        provider,
        compiler=compiler,
        cache=cache,
        options=SolverOptions(
            diis_size=0, residual_tolerance=1e-9, energy_tolerance=1e-11
        ),
    )
    assert result.converged and abs(result.total_energy - meta["total_energy"]) <= 1e-8
    assert result.provenance["transfer"]["diis_solve_attempts"] == 0
    assert result.provenance["transfer"]["per_iteration_large_h2d_bytes"] == 0
    assert result.provenance["transfer"]["per_iteration_large_d2h_bytes"] == 0


@pytest.mark.skipif(
    not _REAL, reason="requires explicitly allocated resident CUDA window"
)
def test_internal_energy_facade_selects_resident_backend(tmp_path):
    from tools.vibeqc_cc.api import energy

    snapshot, provider, meta, _ = fixture_problem("h2")
    compiler, cache = _compiler_cache(tmp_path)
    result = energy(
        snapshot,
        provider,
        backend="cuda-resident",
        compiler=compiler,
        cache=cache,
        options=SolverOptions(residual_tolerance=1e-10, energy_tolerance=1e-12),
    )
    assert result.backend == "cuda-resident" and result.converged
    assert result.provenance["backend"] == "cuda-fp64-resident"
    assert abs(result.total_energy - meta["total_energy"]) <= 1e-8
