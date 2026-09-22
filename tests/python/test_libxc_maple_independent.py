"""Imported PBE-X and its emitted C share an independent Libxc oracle."""

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_libxc_maple_import import FEATURES, _imported_pbe_x
from vibeqc_compiler.common.array_graph import evaluate_array_graph
from vibeqc_compiler.integral.scalar_c import ScalarCEmitter


def test_imported_and_compiled_pbe_x_match_independent_libxc(tmp_path: Path) -> None:
    libxc = pytest.importorskip("pyscf.dft.libxc")
    if libxc.__version__ != "7.0.0":
        pytest.skip("independent oracle is pinned to Libxc 7.0.0")
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    _, graph, roots, _ = _imported_pbe_x()
    emitter = ScalarCEmitter(graph, {name: name for name in FEATURES})
    emitter.emit(roots)
    source = "\n".join(
        [
            "#include <cmath>",
            'extern "C" void evaluate(const double* input, double* output) {',
            *(f"const double {name} = input[{i}];" for i, name in enumerate(FEATURES)),
            *emitter.lines,
            *(
                f"output[{i}] = {emitter.reference(root)};"
                for i, root in enumerate(roots)
            ),
            "}",
        ]
    )
    path = tmp_path / "pbe.cpp"
    path.write_text(source)
    output = tmp_path / "pbe.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O2",
            "-fPIC",
            "-shared",
            str(path),
            "-o",
            str(output),
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )
    library = ctypes.CDLL(str(output))
    pointer = ctypes.POINTER(ctypes.c_double)
    library.evaluate.argtypes = (pointer, pointer)
    library.evaluate.restype = None

    count = 128
    rng = np.random.default_rng(7382026)
    density = np.exp(rng.uniform(np.log(1e-3), np.log(4.0), (2, count)))
    gradient = 0.25 * rng.normal(size=(2, 3, count)) * density[:, None, :] ** (4 / 3)
    rho = np.concatenate((density[:, None, :], gradient), axis=1)
    features = np.vstack(
        (
            density,
            np.sum(gradient[0] ** 2, axis=0),
            np.sum(gradient[0] * gradient[1], axis=0),
            np.sum(gradient[1] ** 2, axis=0),
            np.zeros((2, count)),
        )
    )
    interpreted = np.stack(
        [
            np.broadcast_to(value, (count,))
            for value in evaluate_array_graph(
                graph, roots, dict(zip(FEATURES, features, strict=True))
            )
        ]
    )
    compiled = np.empty_like(interpreted)
    for point, values in enumerate(features.T):
        packed = np.ascontiguousarray(values)
        result = np.empty(len(roots))
        library.evaluate(packed.ctypes.data_as(pointer), result.ctypes.data_as(pointer))
        compiled[:, point] = result
    exc, vxc, _, _ = libxc.eval_xc("GGA_X_PBE", rho, spin=1, deriv=1)
    expected = np.vstack(
        (exc * density.sum(axis=0), np.concatenate((vxc[0], vxc[1]), axis=1).T)
    )
    np.testing.assert_allclose(interpreted, expected, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(compiled, expected, atol=2e-13, rtol=2e-12)
    np.testing.assert_allclose(compiled, interpreted, atol=2e-14, rtol=2e-13)
