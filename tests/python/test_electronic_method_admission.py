"""Method graph state names and schema versions are unambiguous contracts."""

from dataclasses import replace

import pytest
from vibeqc_compiler.method import StateSpec, rhf_electronic_method_ir


@pytest.mark.parametrize("version", (True, 1.0, "1"))
def test_schema_version_requires_an_integer(version: object) -> None:
    with pytest.raises(ValueError, match="version"):
        replace(rhf_electronic_method_ir(), version=version)


def test_persistent_state_names_cannot_have_two_owners() -> None:
    graph = rhf_electronic_method_ir()
    conflicting = StateSpec("density", "amplitude", "different-state-coordinates")
    with pytest.raises(ValueError, match="duplicate state"):
        replace(graph, states=(*graph.states, conflicting))


@pytest.mark.parametrize("field", ("states", "operators"))
def test_untyped_nodes_fail_before_name_sorting(field: str) -> None:
    with pytest.raises(ValueError, match="typed"):
        replace(rhf_electronic_method_ir(), **{field: (object(),)})
