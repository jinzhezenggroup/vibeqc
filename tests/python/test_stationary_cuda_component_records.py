"""CUDA record packing must preserve canonical coordinates and actual memory."""

import ast
from pathlib import Path

import numpy as np
import pytest
from vibeqc._stationary_cuda import _CudaSources
from vibeqc_compiler.integral.first_derivative_schedule import derivative_binding


@pytest.mark.parametrize(
    "operator,components",
    [
        ("overlap", ("y", "")),
        ("kinetic", ("zz", "y")),
        ("nuclear_attraction", ("xz", "y")),
        ("four_center_eri", ("yz", "x", "zz", "")),
        ("overlap", ("", "")),
    ],
)
def test_cuda_records_permute_input_coordinates_with_output_axes(
    operator: str, components: tuple[str, ...]
) -> None:
    """Distinct centers/axes make a missed input rotation observable without CUDA."""
    count = len(components)
    binding = derivative_binding(operator, components)
    owner = object.__new__(_CudaSources)
    owner.centers = np.array(
        [[0.2, -0.4, 0.7], [1.2, 0.8, -0.5], [-0.6, 1.7, 2.1], [2.3, -1.1, 0.9]]
    )
    owner.primitives = np.array([[0.3, 1.1], [0.7, -0.9], [1.2, 0.6], [1.8, 1.3]])
    owner.aos = np.zeros((count, 16))
    owner.aos[:, 0] = range(count)
    owner.aos[:, 1] = range(count)
    owner.aos[:, 2] = 1
    owner.expansions = tuple(((c, (-0.5) ** i),) for i, c in enumerate(components))
    owner.kinds = {binding.request: 0}
    owner.pending = (0, 0)
    owner.buffer = np.ones((4, 26))
    owner.maps = np.full((4, 12), -1, dtype=np.int64)
    owner.used = 0
    nucleus = 2 if operator == "nuclear_attraction" else None
    owner.integral(0, operator, tuple(range(count)), 0.25, nucleus=nucleus, charge=2.0)
    assert owner.used == 1
    centers = list(range(count)) + ([2] if nucleus is not None else [])
    expected = owner.centers[centers][np.ix_(binding.centers, binding.axes)]
    np.testing.assert_array_equal(
        owner.buffer[0, 4 : 4 + expected.size], expected.ravel()
    )
    np.testing.assert_array_equal(
        owner.maps[0, : expected.size],
        [3 * centers[c] + axis for c in binding.centers for axis in binding.axes],
    )
    np.testing.assert_array_equal(
        owner.buffer[0, :count], owner.primitives[list(binding.centers[:count]), 0]
    )
    assert owner.buffer[0, 24] == 0.25 * np.prod([(-0.5) ** i for i in range(count)])


def test_cuda_host_bound_covers_record_and_axis_map_growth() -> None:
    """Check the actual admission coefficient against ABI allocation sizes."""
    source = Path(__file__).resolve().parents[2] / "python/vibeqc/_stationary_cuda.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "complete_rks_cuda_gradient_diagnostic"
    )
    assignment = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "host_bound" for t in node.targets)
    )
    terms = [
        node
        for node in ast.walk(assignment.value)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Mult)
        and isinstance(node.left, ast.Constant)
        and isinstance(node.right, ast.Name)
        and node.right.id == "primitive_tile"
    ]
    assert len(terms) == 1
    coefficient = terms[0].left
    assert isinstance(coefficient, ast.Constant) and type(coefficient.value) is int
    actual_per_record = np.empty((1, 26), dtype=np.float64).nbytes
    actual_per_record += np.empty((1, 12), dtype=np.int64).nbytes
    assert 8 * coefficient.value >= actual_per_record
