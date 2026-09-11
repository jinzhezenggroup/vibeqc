"""CLI wrapper for the shell codegen capability report."""

# Source-tree CLI bootstrap; importing the compiler needs no native runtime.
import sys as _compiler_sys
from pathlib import Path as _CompilerPath

_compiler_sys.path.insert(
    0, str(_CompilerPath(__file__).resolve().parents[1] / "python")
)

from vibeqc_compiler.integral.capabilities import main

if __name__ == "__main__":
    main()
