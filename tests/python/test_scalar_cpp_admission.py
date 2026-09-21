"""Compile scalar lowering with adversarial names and failed output publication."""

import ctypes as ct
import shutil
import subprocess
import typing
from pathlib import Path

import pytest
from vibeqc_compiler.tensor import Program, TensorSpec, input_tensor, multiply
from vibeqc_compiler.tensor.scalar_cpp import emit_scalar_cpp


def _compiled(tmp_path: Path, name: str) -> typing.Any:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    x = input_tensor(name, TensorSpec((), role="input"))
    source = emit_scalar_cpp(
        Program({"a": x, "b": multiply(x, x)}),
        function_name="scalar_probe",
        check_intermediates=False,
    )
    path = tmp_path / "probe.cpp"
    path.write_text(
        "#include <cmath>\n#include <algorithm>\n"
        + source
        + '\nextern "C" int call(double x, double* a, double* b) { return scalar_probe(x,*a,*b); }\n'
    )
    library = tmp_path / "probe.so"
    subprocess.run(
        [compiler, "-std=c++17", "-shared", "-fPIC", str(path), "-o", str(library)],
        check=True,
        capture_output=True,
        timeout=60,
    )
    call = ct.CDLL(str(library)).call
    call.argtypes = [ct.c_double, ct.POINTER(ct.c_double), ct.POINTER(ct.c_double)]
    call.restype = ct.c_int
    return call


@pytest.mark.parametrize("name", ("v0", "out_a", "x"))
def test_scalar_input_names_cannot_collide_with_emitter(
    tmp_path: Path, name: str
) -> None:
    call = _compiled(tmp_path, name)
    a, b = ct.c_double(-7), ct.c_double(-9)
    assert call(2.0, ct.byref(a), ct.byref(b)) == 1
    assert (a.value, b.value) == (2.0, 4.0)


def test_scalar_failure_preserves_all_output_references(tmp_path: Path) -> None:
    call = _compiled(tmp_path, "x")
    a, b = ct.c_double(-7), ct.c_double(-9)
    assert call(1e200, ct.byref(a), ct.byref(b)) == 0
    assert (a.value, b.value) == (-7.0, -9.0)
