"""The nonlocal runtime imports on the declared Python 3.10 feature surface."""

import subprocess
import sys
from pathlib import Path


def test_nonlocal_runtime_does_not_require_typing_self() -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import sys
import builtins
original_import = builtins.__import__
def checked_import(name, globals=None, locals=None, fromlist=(), level=0):
    if (name in ("typing", "typing_extensions") and "Self" in (fromlist or ())
            and (globals or {}).get("__name__") == "vibeqc.nonlocal_runtime"):
        raise ImportError("Self is annotation-only and unavailable at runtime")
    return original_import(name, globals, locals, fromlist, level)
builtins.__import__ = checked_import
sys.path.insert(0, sys.argv[1])
from vibeqc.nonlocal_runtime import NonlocalFixedGridPlan
assert callable(NonlocalFixedGridPlan.__enter__)
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script, str(root / "python")],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
