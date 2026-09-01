"""Source-level contracts for the CUDA density-fitting setup path."""

from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_cuda_df_metric_uses_generic_cusolver_api():
    """Prevent the deprecated typed eigensolver from returning unnoticed."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_density_fitting.cu").read_text(
        encoding="utf-8"
    )
    assert "cusolverDnCreateParams" in source
    assert "cusolverDnXsyevd_bufferSize" in source
    assert "cusolverDnXsyevd(" in source
    assert "cusolverDnDsyevd" not in source
    assert "solver_host_workspace" in source


def test_cuda_df_scf_has_device_resident_iteration_boundary():
    """Keep the DF SCF bridge from regressing to host J/K staging."""

    source = (REPOSITORY_ROOT / "src" / "scf" / "cuda_density_fitting.cu").read_text(
        encoding="utf-8"
    )
    assert "execute_cuda_density_fitting_rhf_jk_device" in source
    assert "execute_cuda_density_fitting_uhf_jk_device" in source
    assert "execute_cuda_density_fitting_rhf_force_response" in source
    assert "execute_cuda_density_fitting_uhf_force_response" in source
    assert "reduce_force_response_kernel" in source
    assert "run_cuda_density_fitting_rhf_device_scf" in source
    assert "run_cuda_density_fitting_uhf_device_scf" in source
    assert "update_device_convergence_kernel" in source
    assert "cudaStreamBeginCapture" in source
    assert "cudaGraphLaunch" in source


def test_cuda_df_metric_diagnostics_are_publicly_wired():
    """Keep metric/allocation evidence available through every public layer."""

    header = (REPOSITORY_ROOT / "include" / "vibeqc" / "vibeqc.h").read_text(
        encoding="utf-8"
    )
    native = (REPOSITORY_ROOT / "python" / "vibeqc" / "_native.py").read_text(
        encoding="utf-8"
    )
    batch = (REPOSITORY_ROOT / "python" / "vibeqc" / "batch.py").read_text(
        encoding="utf-8"
    )
    assert "vibeqc_density_fitting_metric_diagnostic" in header
    assert "vibeqc_batch_get_last_density_fitting_metric_diagnostics" in header
    assert "DensityFittingMetricDiagnostic" in native
    assert "last_density_fitting_metric_diagnostics" in batch
