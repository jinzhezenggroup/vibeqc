"""Source-level contracts for the CUDA density-fitting setup path."""

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def test_cuda_df_metric_uses_generic_cusolver_api():
    """Prevent the deprecated typed eigensolver from returning unnoticed."""

    source = (
        REPOSITORY_ROOT / "src" / "scf" / "cuda_density_fitting.cu"
    ).read_text(encoding="utf-8")
    assert "cusolverDnCreateParams" in source
    assert "cusolverDnXsyevd_bufferSize" in source
    assert "cusolverDnXsyevd(" in source
    assert "cusolverDnDsyevd" not in source
    assert "solver_host_workspace" in source
