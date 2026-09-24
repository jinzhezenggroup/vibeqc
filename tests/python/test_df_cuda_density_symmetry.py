from pathlib import Path

import pytest

ROOT = Path(__file__).parents[2]
SOURCE = ROOT / "src/scf/cuda/df_scf_kernels.cu"
RHF = ROOT / "src/scf/cuda/df_rhf_scf.cpp"
UHF = ROOT / "src/scf/cuda/df_uhf_scf.cpp"


def _density_kernel() -> str:
    source = SOURCE.read_text()
    start = source.index("__global__ void build_device_density_kernel")
    end = source.index("__global__ void compute_device_energy_kernel", start)
    return source[start:end]


def _compact(path: Path) -> str:
    return " ".join(path.read_text().split())


def test_df_cuda_density_contracts_unique_pairs_for_production_weights() -> None:
    kernel = _density_kernel()

    assert "occupation_weight == 1.0 || occupation_weight == 2.0" in kernel
    assert "if (symmetric_density && row > column) return;" in kernel
    assert "density[element] = value;" in kernel
    assert "density[offset + column + row * nbf] = value;" in kernel
    assert kernel.index(
        "if (symmetric_density && row > column) return;"
    ) < kernel.index("for (std::int32_t orbital")


def test_df_cuda_density_preserves_generic_weight_fallback() -> None:
    kernel = _density_kernel()

    assert "if (row > column) return;" not in kernel
    assert "if (symmetric_density && row != column)" in kernel

    rhf = _compact(RHF)
    uhf = _compact(UHF)
    assert "launch_build_device_density_kernel" in rhf
    assert "2.0, d_next_density" in rhf
    assert uhf.count("launch_build_device_density_kernel") == 2
    assert uhf.count("1.0, d_next_") == 2


@pytest.mark.parametrize(
    ("nbf", "full_pairs", "unique_pairs"),
    [
        (384, 147_456, 73_920),
        (768, 589_824, 295_296),
    ],
)
def test_df_cuda_density_pair_work_census(
    nbf: int, full_pairs: int, unique_pairs: int
) -> None:
    assert nbf * nbf == full_pairs
    assert nbf * (nbf + 1) // 2 == unique_pairs
    assert unique_pairs < full_pairs
