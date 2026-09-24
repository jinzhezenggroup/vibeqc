"""CUDA SCF density science must be emitted from the canonical TensorIR."""

from pathlib import Path

import pytest
from vibeqc_compiler.array_api.scf import (
    density_program as array_density_program,
    weighted_density_program as array_weighted_density_program,
)
from vibeqc_compiler.tensor.scf_cuda import (
    density_template_hash,
    emit_density_cuda,
    weighted_density_template_hash,
)

from tools.generate_scf_array_native import template_hash

SOURCE = Path(__file__).parents[2] / "src/scf/cuda/scf_density_kernels.cu"


def test_cuda_density_uses_same_logical_tensor_identity_as_cpu() -> None:
    assert density_template_hash() == template_hash(
        array_density_program(1, 3, spin_count=2, orbital_count=2)
    )
    assert weighted_density_template_hash() == template_hash(
        array_weighted_density_program(1, 3, spin_count=2, orbital_count=2)
    )


def test_cuda_plain_density_is_compiler_owned_and_symmetric() -> None:
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


def test_cuda_weighted_density_is_generated_but_keeps_full_square_order() -> None:
    generated = emit_density_cuda()
    source = SOURCE.read_text()
    start = generated.index("__global__ void occupied_weighted_density_kernel")
    weighted = generated[start:]

    assert "if (row > column) return;" not in weighted
    assert "weighted_density[element] = value;" in weighted
    assert (
        "2.0 * orbital_energies[eigen_offset + orbital] *" in weighted
    )

    assert "__global__ void build_weighted_density_kernel" not in source
    assert "__global__ void build_spin_weighted_density_kernel" not in source
    assert "generated::occupied_weighted_density_kernel<2>" in source
    assert "generated::occupied_weighted_density_kernel<1>" in source


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
