"""Independent topology and directional-adjoint gates for fixed-state GFN2."""

from dataclasses import replace

import numpy as np
import pytest
from test_gfn2_electronic_ir import _restricted_feeds, _topology
from vibeqc_compiler.method import build_gfn2_electronic_program
from vibeqc_compiler.tensor import execute


def test_one_shell_cannot_assign_orbitals_to_different_atoms() -> None:
    topology = _topology()
    owners = list(topology.orbital_to_atom)
    owners[2] = 2  # Same system, but shell 1 would then have two atomic owners.
    with pytest.raises(ValueError, match="shell.*atom"):
        replace(topology, orbital_to_atom=tuple(owners))


@pytest.mark.parametrize(
    "name", ["overlap", "dipole_integrals", "quadrupole_integrals"]
)
@pytest.mark.parametrize(
    "output", ["hamiltonian", "hamiltonian_alpha", "hamiltonian_beta"]
)
def test_integral_vjp_matches_recomputed_primal_difference(
    name: str, output: str
) -> None:
    topology = _topology()
    reference = "restricted" if output == "hamiltonian" else "unrestricted"
    compiled = build_gfn2_electronic_program("GFN2-xTB", topology, reference=reference)
    feeds = _restricted_feeds()
    rng = np.random.default_rng(624)
    if reference == "unrestricted":
        for field, shape in (
            ("shell_magnetization_potential", (topology.shell_count,)),
            ("atomic_dipole_magnetization_potential", (topology.atom_count, 3)),
            ("atomic_quadrupole_magnetization_potential", (topology.atom_count, 6)),
        ):
            feeds[field] = rng.normal(size=shape) * 0.07
    cotangent = rng.normal(size=topology.matrix_count)
    direction = rng.normal(size=feeds[name].shape)
    direction /= np.linalg.norm(direction)
    generated = execute(
        compiled.integral_vjp(output).program,
        {**feeds, "bar_" + output: cotangent},
    ).outputs["bar_" + name]
    projection = float(np.sum(generated * direction))
    assert abs(projection) > 1e-5
    for step in (2e-5, 2e-6):
        samples = [
            float(
                np.dot(
                    cotangent,
                    execute(
                        compiled.program,
                        {**feeds, name: feeds[name] + sign * step * direction},
                    ).outputs[output],
                )
            )
            for sign in (-1, 1)
        ]
        np.testing.assert_allclose(
            projection, (samples[1] - samples[0]) / (2 * step), atol=2e-9, rtol=2e-8
        )
