"""HF force coordinates use the Cartesian domain admitted by native lowering."""

import numpy as np
import pytest
from vibeqc_compiler.tensor import execute
from vibeqc_compiler.tensor.scf import hf_force_program


@pytest.mark.parametrize("spins", (1, 2))
@pytest.mark.parametrize("coordinates", (3, 6, 12))
def test_hf_force_cartesian_domain_and_reference(
    spins: int, coordinates: int
) -> None:
    program = hf_force_program(2, 3, spin_count=spins, coordinate_count=coordinates)
    output = program.outputs["forces"]
    assert tuple(index.space.kind for index in output.spec.indices) == (
        "batch",
        "cartesian",
    )
    rng = np.random.default_rng(1263 + coordinates + spins)
    density = rng.normal(size=(2, spins, 3, 3))
    weighted = rng.normal(size=(2, spins, 3, 3))
    dh = rng.normal(size=(2, coordinates, 3, 3))
    ds = rng.normal(size=(2, coordinates, 3, 3))
    nuclear = rng.normal(size=(2, coordinates))
    two = rng.normal(size=(2, coordinates))
    actual = execute(
        program,
        {
            "density": density,
            "weighted_density": weighted,
            "hcore_derivative": dh,
            "overlap_derivative": ds,
            "nuclear_repulsion_derivative": nuclear,
            "two_electron": two,
        },
    ).outputs["forces"]
    expected = -nuclear - two
    for b in range(2):
        for c in range(coordinates):
            for s in range(spins):
                for p in range(3):
                    for q in range(3):
                        expected[b, c] -= density[b, s, p, q] * dh[b, c, p, q]
                        expected[b, c] += weighted[b, s, p, q] * ds[b, c, p, q]
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=1e-13)
