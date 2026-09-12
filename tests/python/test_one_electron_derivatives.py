"""Independent analytic and finite-difference gates for generated S/T/V gradients."""

import ctypes
import math
import shutil
import subprocess
from functools import cache
from itertools import product

import numpy as np
import pytest
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
    build_one_electron_derivative_kernel,
    evaluate_one_electron_derivative_primitive,
)
from vibeqc_compiler.integral.one_electron_values import evaluate_one_electron_primitive
from vibeqc_compiler.integral.shell_spec import cartesian_components


@cache
def kernel(family, angular, components):
    return build_one_electron_derivative_kernel(
        build_one_electron_derivative_ir(family, angular, charge=2.3), components
    )


def gaussian_norm(exponent, component):
    """Closed Cartesian even moments remove libcint's shell radial convention."""
    return (math.pi / (2 * exponent)) ** 1.5 * math.prod(
        math.prod(range(1, 2 * component.count(axis), 2))
        / (4 * exponent) ** component.count(axis)
        for axis in "xyz"
    )


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
@pytest.mark.parametrize("angular", list(product(range(4), repeat=2)))
def test_all_cartesian_derivatives_match_independent_libcint(family, angular):
    gto = pytest.importorskip("pyscf.gto")
    positions = [(0.2, -0.3, 0.1), (-0.4, 0.15, 0.5), (0.17, -0.11, -0.4)]
    exponents = (0.8, 0.35)
    components = tuple(
        cartesian_components(angular_momentum) for angular_momentum in angular
    )
    mol = gto.M(
        atom=[("ghost-H", positions[0]), ("ghost-He", positions[1])],
        basis={
            "H": [[angular[0], [exponents[0], 1]]],
            "He": [[angular[1], [exponents[1], 1]]],
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    overlap = mol.intor("int1e_ovlp_cart")
    norms = [gaussian_norm(e, c) for e, cs in zip(exponents, components) for c in cs]
    scales = np.sqrt(overlap.diagonal() / norms)
    n = len(components[0])
    operator, factor = {
        "overlap": ("ovlp", 1),
        "kinetic": ("kin", 1),
        "nuclear_attraction": ("rinv", -2.3),
    }[family]
    with mol.with_rinv_origin(positions[2]):
        # libcint's ip derivative acts on the electronic coordinate of the
        # bra. Moving its Gaussian center has the opposite sign.
        first = -factor * mol.intor_by_shell(f"int1e_ip{operator}_cart", (0, 1), comp=3)
        second = -factor * mol.intor_by_shell(
            f"int1e_ip{operator}_cart", (1, 0), comp=3
        ).transpose(0, 2, 1)
    reference = np.array(
        [first, second] + ([-first - second] if family == "nuclear_attraction" else [])
    )
    reference /= scales[:n][None, None, :, None] * scales[n:][None, None, None, :]
    for i, a in enumerate(components[0]):
        for j, b in enumerate(components[1]):
            actual = evaluate_one_electron_derivative_primitive(
                kernel(family, angular, (a, b)),
                exponents,
                positions if family == "nuclear_attraction" else positions[:2],
            )
            np.testing.assert_allclose(
                actual, reference[:, :, i, j], atol=4e-11, rtol=5e-12
            )


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
@pytest.mark.parametrize("coincident", [False, True])
def test_center_sign_translation_and_arbitrary_fixed_weights(family, coincident):
    positions = np.array([[0.2, -0.3, 0.1], [-0.4, 0.15, 0.5], [0.17, -0.11, -0.4]])
    if coincident:
        positions[:] = positions[0]
    if family != "nuclear_attraction":
        positions = positions[:2]
    exponents = (0.8, 0.35)
    angular = (3, 2)
    pairs = list(
        product(
            *(cartesian_components(angular_momentum) for angular_momentum in angular)
        )
    )
    weights = np.random.default_rng(141).normal(size=len(pairs))
    programs = [kernel(family, angular, c) for c in pairs]
    actual = sum(
        w
        * np.array(evaluate_one_electron_derivative_primitive(k, exponents, positions))
        for w, k in zip(weights, programs)
    )
    np.testing.assert_allclose(actual.sum(axis=0), 0, atol=1e-13)
    for h in (2e-4, 5e-5):
        for center, axis in product(range(len(positions)), range(3)):
            plus, minus = positions.copy(), positions.copy()
            plus[center, axis] += h
            minus[center, axis] -= h
            difference = sum(
                w
                * (
                    evaluate_one_electron_primitive(k, exponents, plus)
                    - evaluate_one_electron_primitive(k, exponents, minus)
                )
                for w, k in zip(weights, programs)
            ) / (2 * h)
            assert actual[center, axis] == pytest.approx(difference, abs=2e-7, rel=2e-7)


def test_derivative_contract_keeps_external_center_and_rejects_unsupported_shells():
    ir = build_one_electron_derivative_ir("nuclear_attraction", (3, 3), weighted=True)
    assert ir.derivative.independent_centers(ir.operator) == (0, 1)
    assert ir.derivative.recovered_centers(ir.operator) == (2,)
    assert ir.contractions[0].weights.source == "external_weight"
    assert kernel("nuclear_attraction", (3, 3), ("xxx", "zzz")).boys_count == 8
    with pytest.raises(ValueError, match="s/p/d/f"):
        build_one_electron_derivative_ir("kinetic", (5, 0))


def test_emitted_derivatives_normalized_raw_and_spherical_blocks(tmp_path):
    """Exercise emitted CSE/geometry/Boys boundaries against independent blocks."""
    pytest.importorskip("pyscf")
    from vibeqc_compiler.integral.one_electron_derivatives_cuda import (
        emit_one_electron_derivatives_cuda,
    )

    from tools.vibeqc_validation.one_electron_derivatives import (
        one_electron_derivative_matrix,
    )
    from tools.vibeqc_validation.one_electron_derivatives_cuda import (
        derivative_evaluation_body,
    )

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    header = emit_one_electron_derivatives_cuda().replace(
        "#include <cuda_runtime.h>", ""
    )
    source = (
        "#define __device__\n#define __forceinline__ inline\n#define __noinline__\n"
        + header
    )
    source += '\nnamespace one = vibeqc::scf::generated_one_electron_derivatives;\nextern "C" void evaluate(const double* inputs, double* outputs, unsigned count) {\nfor (unsigned i=0;i<count;++i) { const double* p=inputs+14*i; double* out=outputs+27*i;\n'
    source += derivative_evaluation_body() + "\n}}\n"
    path = tmp_path / "fixture.cpp"
    path.write_text(source)
    libpath = tmp_path / "fixture.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-shared",
            "-fPIC",
            str(path),
            "-o",
            str(libpath),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=180,
    )
    library = ctypes.CDLL(str(libpath))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.evaluate.argtypes = [pointer, pointer, ctypes.c_uint]
    library.evaluate.restype = None
    for fixture in one_electron_derivative_matrix():
        values = np.zeros((len(fixture.records), 27))
        library.evaluate(fixture.records, values, len(values))
        actual = fixture.contract(values)
        np.testing.assert_allclose(
            actual,
            fixture.reference,
            atol=3e-11,
            rtol=6e-12,
            err_msg=fixture.inputs["name"],
        )
        np.testing.assert_allclose(
            fixture.spherical(actual),
            fixture.spherical_reference,
            atol=3e-11,
            rtol=6e-12,
            err_msg=fixture.inputs["name"],
        )
