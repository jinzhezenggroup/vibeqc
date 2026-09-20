"""Build-time native SCF helpers must remain tied to Array frontend TensorIR."""

from vibeqc_compiler.array_api.scf import density_program, weighted_density_program

from tools.generate_scf_array_native import native_header, template_hash


def test_array_scf_native_template_identity_is_shape_independent() -> None:
    small_density = density_program(1, 3, orbital_count=2)
    large_density = density_program(2, 5, orbital_count=4)
    small_weighted = weighted_density_program(1, 3, orbital_count=2)
    large_weighted = weighted_density_program(2, 5, orbital_count=4)

    assert template_hash(small_density) == template_hash(large_density)
    assert template_hash(small_weighted) == template_hash(large_weighted)
    assert template_hash(small_density) != template_hash(small_weighted)


def test_generated_header_records_frontend_tensorir_templates() -> None:
    header = native_header()
    density_hash = template_hash(density_program(1, 3, orbital_count=2))
    weighted_hash = template_hash(weighted_density_program(1, 3, orbital_count=2))

    assert density_hash in header
    assert weighted_hash in header
    assert "density_from_orbitals" in header
    assert "weighted_density_from_orbitals" in header
    assert "array_api.scf -> TensorIR" in header
