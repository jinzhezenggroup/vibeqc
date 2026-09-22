"""The public TensorIR callable namespace must not depend on import order."""

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "first_import",
    (
        "from vibeqc_compiler.tensor import PASSES",
        "from vibeqc_compiler.tensor import rewrite",
        "from vibeqc_compiler.tensor.optimize import PASSES",
        "from vibeqc_compiler.tensor import *",
        "import runpy; runpy.run_path('tools/generate_df_hf_response.py')",
    ),
)
def test_public_optimize_remains_callable(first_import: str) -> None:
    root = Path(__file__).resolve().parents[2]
    code = f"""{first_import}
from vibeqc_compiler.tensor import optimize, rewrite
from vibeqc_compiler.tensor.optimize import optimize as canonical
assert callable(optimize)
assert optimize is canonical
assert callable(rewrite)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root / "python")},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
