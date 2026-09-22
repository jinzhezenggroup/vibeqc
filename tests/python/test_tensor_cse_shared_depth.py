"""Execution equivalence keys must stay compact for shared TensorIR DAGs."""

import os
import subprocess
import sys
from pathlib import Path


def test_exact_cse_handles_deep_shared_expression_without_tree_expansion() -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import numpy as np
from vibeqc_compiler.tensor import Index, IndexSpace, TensorSpec, input_tensor, add, Program, execute
from vibeqc_compiler.tensor.optimize import rewrite
index = Index('i', IndexSpace('shared_depth', 'batch', 1))
x = input_tensor('x', TensorSpec((index,), role='input'))
left = right = x
for _ in range(40):
    left = add(left, left)
    right = add(right, right)
program = Program({'left': left, 'right': right})
result = rewrite(program, 'exact_cse')
assert len(result.live_nodes) == 41
assert result.outputs['left'] is result.outputs['right']
assert result.logical_hash == program.logical_hash
values = execute(result, {'x': np.array([1.0])}).outputs
np.testing.assert_array_equal(values['left'], [float(2**40)])
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "python")},
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
