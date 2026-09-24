"""Regression gates for the symmetric generated CPU SCF density schedule."""

from __future__ import annotations

import pytest

from tools.generate_scf_array_native import native_header


def _function_body(header: str, name: str, next_name: str) -> str:
    start = header.index(f"inline void {name}")
    end = header.index(f"inline void {next_name}", start)
    return header[start:end]


def test_density_codegen_contracts_only_unique_ao_pairs() -> None:
    header = native_header()
    density = _function_body(
        header, "density_from_orbitals", "weighted_density_from_orbitals"
    )
    weighted = _function_body(header, "weighted_density_from_orbitals", "diis_gram")

    assert "for (std::size_t nu = mu; nu < nbf; ++nu)" in density
    assert "if (mu != nu) output[nu * nbf + mu] = value;" in density
    assert "for (std::size_t nu = 0; nu < nbf; ++nu)" not in density

    # Weighted density retains its historical full-square evaluation: moving the
    # orbital-energy factor across the two AO coefficients can change FP64
    # rounding, so this optimization is intentionally limited to plain density.
    assert "for (std::size_t nu = 0; nu < nbf; ++nu)" in weighted


@pytest.mark.parametrize(
    ("nbf", "full_pairs", "unique_pairs"),
    (
        (384, 147_456, 73_920),
        (768, 589_824, 295_296),
    ),
)
def test_density_symmetry_work_census(
    nbf: int, full_pairs: int, unique_pairs: int
) -> None:
    assert nbf * nbf == full_pairs
    assert nbf * (nbf + 1) // 2 == unique_pairs
    assert unique_pairs < full_pairs
