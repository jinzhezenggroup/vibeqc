"""CPU compilation shares native cache integrity and finite process semantics."""

import ctypes
import shutil
import sys
from pathlib import Path

import pytest
from vibeqc_compiler.common.compiler_process import run_compiler
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.common.native_runtime import compile_runtime


def test_cpu_cache_executes_exact_headers_and_flags_and_rejects_corruption(tmp_path):
    if shutil.which("c++") is None:
        pytest.skip("native C++ compiler unavailable")
    compiler = CppCompilerAdapter(Path("c++"))
    header = tmp_path / "value.hpp"
    header.write_text("constexpr int value = 21;\n")
    source = tmp_path / "value.cpp"
    source.write_text(
        '#include "value.hpp"\n#ifndef SCALE\n#define SCALE 2\n#endif\nextern "C" int answer() { return value * SCALE; }\n'
    )
    cache = tmp_path / "cache"

    def build(**kwargs):
        artifact = compile_runtime(compiler, cache, source, headers=(header,), **kwargs)
        library = ctypes.CDLL(str(artifact.library))
        library.answer.restype = ctypes.c_int
        return artifact, library.answer()

    first, answer = build()
    assert answer == 42
    replay, answer = build()
    assert answer == 42 and replay == first
    header.write_text("constexpr int value = 7;\n")
    changed, answer = build()
    assert answer == 14 and changed.metadata["key"] != first.metadata["key"]
    flags, answer = build(options=("-ffp-contract=off", "-DSCALE=3"))
    assert answer == 21 and flags.metadata["key"] != changed.metadata["key"]
    # Metadata corruption must fail before loading a possibly wrong binary.
    metadata = flags.library.parent / "artifact.json"
    text = metadata.read_text().replace(flags.metadata["binary_sha256"], "0" * 64)
    metadata.write_text(text)
    with pytest.raises(ValueError, match="integrity"):
        compile_runtime(
            compiler,
            cache,
            source,
            headers=(header,),
            options=("-ffp-contract=off", "-DSCALE=3"),
        )


def test_finite_compiler_process_reports_timeout_and_failure():
    failed = run_compiler(
        [sys.executable, "-c", "raise SystemExit(19)"], 10, label="probe"
    )
    assert failed.returncode == 19 and not failed.timed_out
    timed = run_compiler(
        [sys.executable, "-c", "import time; time.sleep(30)"], 0.05, label="probe"
    )
    assert timed.returncode == 124 and timed.timed_out
    assert "probe compilation timed out" in timed.stderr


def test_missing_requested_cpu_compiler_does_not_select_another_backend():
    with pytest.raises(ValueError, match=r"requested C\+\+ compiler"):
        CppCompilerAdapter(Path("vibeqc-no-such-cxx"))
