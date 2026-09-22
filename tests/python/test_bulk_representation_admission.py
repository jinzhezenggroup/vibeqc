"""Automatic representation must not create a production-domain evaluator."""

import pytest
from vibeqc_compiler.xc import UnsupportedXC, build_program, functional
from vibeqc_compiler.xc.spec import AUTO_BULK_COMPONENTS


@pytest.mark.parametrize("spin", ("polarized", "unpolarized"))
def test_entire_automatic_inventory_remains_nonexecutable(spin: str) -> None:
    """Check every automatic component without admitting molecular execution."""
    for name in AUTO_BULK_COMPONENTS:
        spec = functional(name, spin=spin)
        assert spec.to_payload()["production_admitted"] is False
        with pytest.raises(UnsupportedXC, match="not production-domain admitted"):
            build_program(spec)
