"""D3 compiler imports must not require annotation-only optional packages."""

import subprocess
import sys
from pathlib import Path

_IMPORT_PROBE = """
import importlib.abc
import sys

# Source paths are explicit: isolated mode must not rely on PYTHONPATH or an
# already installed VibeQC package. NumPy is a documented compiler dependency.
sys.path[:0] = [sys.argv[1], sys.argv[2]]
import numpy
sys.modules.pop("typing_extensions", None)

class BlockOptionalTyping(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "typing_extensions" or fullname.startswith("typing_extensions."):
            raise ModuleNotFoundError("typing_extensions is absent in this compiler probe")
        return None

sys.meta_path.insert(0, BlockOptionalTyping())
import vibeqc_compiler.geometry as geometry
assert callable(geometry.compile_d3_bj_batch)
assert callable(geometry.PreparedD3CudaBatch)
"""


def test_d3_geometry_import_without_typing_extensions() -> None:
    root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [sys.executable, "-I", "-c", _IMPORT_PROBE, str(root / "python"), str(root)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
