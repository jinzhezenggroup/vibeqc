"""RCCSD generation must not replace canonical evidence implementation objects."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("preload", [False, True])
def test_rccsd_generator_preserves_canonical_evidence(preload: bool) -> None:
    script = f"""
import importlib, runpy, sys
first = importlib.import_module('vibeqc_compiler.common.evidence') if {preload!r} else None
runpy.run_path('tools/generate_rccsd_native.py', run_name='review_generator')
after = importlib.import_module('vibeqc_compiler.common.evidence')
from vibeqc_compiler.common import evidence
assert evidence is after
assert first is None or after is first
assert after.__file__.endswith('common/evidence.py')
assert callable(after.block_error) and callable(after.validate_evidence)
assert after.block_error([1.0], [1.0], atol=1e-12, rtol=0)['passed']
assert after.outcome('not-run', 'no device')['status'] == 'not-run'
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "python") + os.pathsep + str(ROOT),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_rccsd_evidence_bootstrap_requires_no_numpy() -> None:
    script = """
import runpy, sys
from vibeqc_compiler.common import evidence
assert 'numpy' not in sys.modules
runpy.run_path('tools/generate_rccsd_native.py', run_name='review_generator')
from vibeqc_compiler.common import evidence as after
assert after is evidence and callable(after.block_error)
assert after.canonical_hash({'b': 2, 'a': 1}) == evidence.canonical_hash({'a': 1, 'b': 2})
assert 'numpy' not in sys.modules
"""
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "python") + os.pathsep + str(ROOT),
    }
    completed = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
