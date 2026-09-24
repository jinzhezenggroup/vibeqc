"""CUDA SCF density science must be emitted from the canonical TensorIR."""

from pathlib import Path

import pytest
from vibeqc_compiler.array_api.scf import density_program as array_density_program
from vibeqc_compiler.tensor.scf_cuda import density_template_hash, emit_density_cuda

from tools.generate_scf_array_native import template_hash

SOURCE = Path(__file__).parents[2] / "src/scf/cuda/scf_density_kernels.cu"


def test_cuda_density_uses_same_logical_tensor_identity_as_cpu() -> None:
    assert density_template_hash() == template_hash(
        array_density_program(1, 3, spin_count=2, orbital_count=2)
    )


def test_cuda_density_is_compiler_owned_and_symmetric() -> None:
    generated = emit_density_cuda()
    source = SOURCE.read_text()

    assert "template <int OccupationWeight>" in generated
    assert "if (row > column) return;" in generated
    assert "density[element] = value;" in generated
    assert "density[offset + column + row * n] = value;" in generated
    assert "if constexpr (OccupationWeight == 2)" in generated

    assert "__global__ void build_density_kernel" not in source
    assert "__global__ void build_spin_density_kernel" not in source
    assert "generated::occupied_density_kernel<2>" in source
    assert "generated::occupied_density_kernel<1>" in source


def test_weighted_density_keeps_native_full_square_rounding_order() -> None:
    source = SOURCE.read_text()
    start = source.index("__global__ void build_weighted_density_kernel")
    end = source.index("__global__ void build_spin_weighted_density_kernel", start)
    weighted = source[start:end]
    start = end
    end = source.index("__global__ void sum_uhf_spin_matrices_kernel", start)
    spin_weighted = source[start:end]

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
