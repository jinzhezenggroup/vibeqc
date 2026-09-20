"""Build-time native SCF helpers must remain tied to Array frontend TensorIR."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from vibeqc_compiler.array_api import VibeArray
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


def test_native_generator_participates_in_source_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pathlib import Path

    from vibeqc import autotune

    root = Path(__file__).resolve().parents[2]
    before = autotune.source_identity(root)
    original = autotune.file_hash
    generator = root / "tools/generate_scf_array_native.py"
    monkeypatch.setattr(
        autotune,
        "file_hash",
        lambda path: "f" * 64 if path == generator else original(path),
    )
    assert autotune.source_identity(root) != before


@pytest.mark.parametrize("weighted", (False, True))
def test_fixed_native_specialization_rejects_changed_coefficient_layout(
    monkeypatch: pytest.MonkeyPatch, weighted: bool
) -> None:
    from dataclasses import replace

    from vibeqc_compiler.array_api import namespace as xp
    from vibeqc_compiler.array_api import trace
    from vibeqc_compiler.tensor.scf import (
        density_input_specs,
        weighted_density_input_specs,
    )

    from tools import generate_scf_array_native as generator

    specs = (weighted_density_input_specs if weighted else density_input_specs)(
        1, 3, orbital_count=2
    )
    spec = specs["coefficients"]
    specs["coefficients"] = replace(
        spec, indices=(*spec.indices[:2], spec.indices[3], spec.indices[2])
    )
    output = "weighted_density" if weighted else "density"

    def equation(
        coefficients: VibeArray,
        occupations: VibeArray,
        orbital_energies: VibeArray | None = None,
    ) -> dict[str, VibeArray]:
        weight = (
            occupations if orbital_energies is None else occupations * orbital_energies
        )
        return {
            output: xp.einsum("bsip,bsi,bsiq->bspq", coefficients, weight, coefficients)
        }

    program = trace(equation, specs, provenance={"construction": "array_frontend"})
    monkeypatch.setattr(generator, output + "_program", lambda *args, **kwargs: program)
    with pytest.raises(ValueError, match="layout|topology"):
        generator.native_header()
