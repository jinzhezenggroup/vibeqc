"""Reject lossy GFN2 geometry input before selecting a pair topology."""

import numpy as np
import pytest
from vibeqc_compiler.geometry import (
    build_gfn2_pair_topology,
    build_gfn2_short_range_program,
    gfn2_geometry,
)


@pytest.mark.parametrize("kind", ["complex", "text", "object", "bool"])
@pytest.mark.parametrize("operation", ["build", "replay"])
def test_gfn2_rejects_nonreal_coordinates(kind: str, operation: str) -> None:
    geometry = gfn2_geometry((1, 1))
    valid = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    arrays = {
        "complex": valid.astype(complex) + 0.01j,
        "text": valid.astype(str),
        "object": valid.astype(object),
        "bool": valid.astype(bool),
    }
    topology = build_gfn2_pair_topology(geometry, valid)
    program = build_gfn2_short_range_program(geometry, topology)
    with pytest.raises(ValueError, match="real numeric"):
        if operation == "build":
            build_gfn2_pair_topology(geometry, arrays[kind])
        else:
            program.validate_coordinates(arrays[kind])


@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_gfn2_keeps_valid_numeric_and_strided_inputs(dtype: str) -> None:
    geometry = gfn2_geometry((1, 1))
    array = np.array([[0, 0, 0], [2, 0, 0]], dtype=dtype)
    original = array.copy()
    expected = build_gfn2_pair_topology(geometry, array)
    assert build_gfn2_pair_topology(geometry, array[:, ::-1]).pairs == expected.pairs
    array.flags.writeable = False
    assert build_gfn2_pair_topology(geometry, array).identity == expected.identity
    np.testing.assert_array_equal(array, original)
