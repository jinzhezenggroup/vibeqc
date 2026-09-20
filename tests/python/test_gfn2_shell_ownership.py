"""Negative ownership admission gates for the GFN2 population compiler."""

import pytest
from vibeqc_compiler.method import (
    Gfn2ElectronicTopology,
    build_gfn2_population_program,
)


@pytest.mark.parametrize("split_owner", (False, True))
def test_population_rejects_incomplete_or_split_shell_ownership(
    split_owner: bool,
) -> None:
    message = "share one atom owner" if split_owner else "at least one orbital"
    with pytest.raises(ValueError, match=message):
        topology = Gfn2ElectronicTopology(
            system_atom_offsets=(0, 2),
            system_shell_offsets=(0, 3),
            system_orbital_offsets=(0, 2),
            orbital_to_shell=(0, 0) if split_owner else (0, 2),
            orbital_to_atom=(0, 1),
        )
        build_gfn2_population_program(
            "GFN2-xTB",
            topology,
            spin_channels=(1,),
            reference_shell_occupations=(1.0, 0.0, 1.0),
        )
