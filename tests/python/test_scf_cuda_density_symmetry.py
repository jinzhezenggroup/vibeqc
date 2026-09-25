from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[2] / "src/scf/cuda/scf_density_kernels.cu"


def _function(source: str, name: str, next_name: str) -> str:
    start = source.index(f"__global__ void {name}")
    end = source.index(f"__global__ void {next_name}", start)
    return source[start:end]


def test_cuda_density_builds_contract_only_unique_ao_pairs() -> None:
    source = SOURCE.read_text()
    rhf = _function(source, "build_density_kernel", "build_spin_density_kernel")
    spin = _function(source, "build_spin_density_kernel", "mix_open_shell_guess_kernel")

    for kernel in (rhf, spin):
        assert "if (row > column) return;" in kernel
        assert "density[element] = value;" in kernel
        assert "density[offset + matrix_index(column, row, n)] = value;" in kernel
        assert kernel.index("if (row > column) return;") < kernel.index(
            "for (std::int32_t orbital"
        )


def test_weighted_density_keeps_full_square_rounding_order() -> None:
    source = SOURCE.read_text()
    weighted = _function(
        source, "build_weighted_density_kernel", "build_spin_weighted_density_kernel"
    )
    spin_weighted = _function(
        source, "build_spin_weighted_density_kernel", "sum_uhf_spin_matrices_kernel"
    )

    # The orbital-energy factor is not a power-of-two scale, so swapping AO
    # operands can change FP64 rounding. This optimization intentionally does
    # not mirror the weighted-density kernels.
    for kernel in (weighted, spin_weighted):
        assert "if (row > column) return;" not in kernel
        assert "matrix_index(column, row, n)] = value" not in kernel


@pytest.mark.parametrize(
    ("nbf", "full_pairs", "unique_pairs"),
    [
        (384, 147_456, 73_920),
        (768, 589_824, 295_296),
    ],
)
def test_cuda_density_pair_work_census(
    nbf: int, full_pairs: int, unique_pairs: int
) -> None:
    assert nbf * nbf == full_pairs
    assert nbf * (nbf + 1) // 2 == unique_pairs
    assert unique_pairs < full_pairs
