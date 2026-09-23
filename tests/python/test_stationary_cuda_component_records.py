"""CUDA record packing must preserve canonical coordinates and actual memory."""

import ast
from pathlib import Path

import numpy as np
import pytest
from vibeqc._stationary_cuda import _CudaSources
from vibeqc_compiler.integral.first_derivative_schedule import (
    derivative_binding,
    derivative_dispatch_id,
)
from vibeqc_compiler.method.stationary_cuda import _emit_stationary_dispatch


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
def test_cuda_tasks_preserve_component_dispatch_and_primitive_work(
    operator: str, components: tuple[str, ...]
) -> None:
    """The task ABI carries ordered components; the wrapper owns permutations."""
    count = len(components)
    binding = derivative_binding(operator, components)
    owner = object.__new__(_CudaSources)
    owner.primitives = np.array([[0.3, 1.1], [0.7, -0.9], [1.2, 0.6], [1.8, 1.3]])
    owner.aos = np.zeros((count, 16))
    owner.aos[:, 0] = range(count)
    owner.aos[:, 1] = range(count)
    owner.aos[:, 2] = 1
    owner.expansions = tuple(((c, (-0.5) ** i),) for i, c in enumerate(components))
    owner.extended = True
    owner.kinds = {binding.request: 0}
    owner.tasks = np.full((4, 9), -1, dtype=np.int64)
    owner.charges = np.ones(4)
    owner.used = 0
    nucleus = 2 if operator == "nuclear_attraction" else None
    owner.integral(
        0, operator, tuple(range(count)), nucleus=nucleus, charge=0.25
    )
    assert owner.used == 1
    task = owner.tasks[0]
    assert tuple(task[:4]) == (
        derivative_dispatch_id(operator, components),
        0,
        count,
        -1 if nucleus is None else nucleus,
    )
    assert tuple(task[4 : 4 + count]) == tuple(range(count))
    assert task[8] == np.prod([int(row[2]) for row in owner.aos])
    coefficient = np.prod([(-0.5) ** i for i in range(count)])
    assert owner.charges[0] == 0.25 * coefficient


@pytest.mark.parametrize(
    "operator,components",
    [
        ("overlap", ("y", "")),
        ("kinetic", ("zz", "y")),
        ("nuclear_attraction", ("xz", "y")),
        ("four_center_eri", ("yz", "x", "zz", "")),
    ],
)
def test_cuda_dispatch_adapter_preserves_input_and_output_permutations(
    operator: str, components: tuple[str, ...]
) -> None:
    """The emitted task adapter applies binding axes in both directions."""
    binding = derivative_binding(operator, components)
    source = _emit_stationary_dispatch(
        [
            (
                derivative_dispatch_id(operator, components),
                7,
                binding.centers,
                binding.axes,
            )
        ]
    )
    assert "cc[3 * i + a] = c[3 * d.centers[i] + d.axes[a]]" in source
    assert "out[3 * d.centers[i] + d.axes[a]] = vc[3 * i + a]" in source


def test_cuda_host_bound_covers_record_and_axis_map_growth() -> None:
    """Check the actual admission coefficient against ABI allocation sizes."""
    source = Path(__file__).resolve().parents[2] / "python/vibeqc/_stationary_cuda.py"
    tree = ast.parse(source.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_complete_rks_cuda_gradient_diagnostic"
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
