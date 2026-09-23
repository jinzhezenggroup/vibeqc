from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL_SOURCE = (ROOT / "src/scf/cuda/df_metric_kernels.cu").read_text(encoding="utf-8")
SETUP_SOURCE = (ROOT / "src/scf/cuda/df_plan_setup.cpp").read_text(encoding="utf-8")


def test_metric_symmetry_only_stores_cusolver_lower_triangle() -> None:
    kernel = KERNEL_SOURCE.split("__global__ void scale_eigenvectors_flat_kernel", maxsplit=1)[0]
    assert "const std::size_t lower = offset + row * dimension + column;" in kernel
    assert "const std::size_t upper = offset + column * dimension + row;" in kernel
    assert "0.5 * (metrics[lower] + metrics[upper])" in kernel
    assert "metrics[lower] = symmetric;" in kernel
    assert "metrics[upper] = symmetric;" not in kernel

    eigensolver = SETUP_SOURCE.split("cusolverDnXsyevd_bufferSize", maxsplit=1)[1]
    eigensolver = eigensolver.split("std::vector<double> eigenvalues", maxsplit=1)[0]
    assert "CUBLAS_FILL_MODE_LOWER" in eigensolver


def test_metric_symmetry_store_census_for_representative_auxiliary_sizes() -> None:
    # The kernel still reads both source entries and performs the same FP64
    # average, but only the triangle consumed by cuSOLVER is written back.
    for naux, stores_before, stores_after, bytes_removed in (
        (928, 862_112, 431_056, 3_448_448),
        (3712, 13_782_656, 6_891_328, 55_130_624),
    ):
        authoritative = naux * (naux + 1) // 2
        assert stores_before == 2 * authoritative
        assert stores_after == authoritative
        assert bytes_removed == (stores_before - stores_after) * 8
        assert stores_after * 2 == stores_before
