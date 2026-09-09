"""Independent scalar value gates before selecting any CUDA schedule."""

import ctypes
import math
import shutil
import subprocess
from functools import cache
from itertools import product

import numpy as np
import pytest

from tools.vibeqc_codegen.one_electron_values import (
    build_one_electron_component_kernel,
    build_one_electron_value_ir,
    evaluate_one_electron_primitive,
)
from tools.vibeqc_codegen.shell_spec import cartesian_components


@cache
def kernel(family, angular, components, charge=2.3):
    return build_one_electron_component_kernel(
        build_one_electron_value_ir(family, angular, charge=charge), components
    )


def gaussian_self_norm(exponent, component):
    """Closed one-dimensional even moments of an unnormalized primitive."""
    result = (math.pi / (2 * exponent)) ** 1.5
    for axis in "xyz":
        n = component.count(axis)
        result *= math.prod(range(1, 2 * n, 2)) / (4 * exponent) ** n
    return result


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
@pytest.mark.parametrize("angular", list(product(range(4), repeat=2)))
def test_every_public_cartesian_component_against_pyscf(family, angular):
    gto = pytest.importorskip("pyscf.gto")
    components = tuple(cartesian_components(l) for l in angular)
    alpha, beta = 0.8, 0.35
    charge = 2.3
    for a, b, c in (
        ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        ((0.2, -0.3, 0.1), (-0.4, 0.15, 0.5), (0.17, -0.11, -0.4)),
        ((0.2, -0.3, 0.1), (0.2 + 1e-10, -0.3, 0.1), (4.0, -3.0, 2.0)),
        ((0.2, -0.3, 0.1), (-0.4, 0.15, 0.5), (50.0, 30.0, -40.0)),
    ):
        mol = gto.M(
            atom=[("ghost-H", a), ("ghost-He", b)],
            basis={
                "H": [[angular[0], [alpha, 1.0]]],
                "He": [[angular[1], [beta, 1.0]]],
            },
            unit="Bohr",
            cart=True,
            verbose=0,
        )
        overlap = mol.intor("int1e_ovlp_cart")
        if family == "overlap":
            reference = overlap
        elif family == "kinetic":
            reference = mol.intor("int1e_kin_cart")
        else:
            with mol.with_rinv_origin(c):
                reference = -charge * mol.intor("int1e_rinv_cart")
        # PySCF's Cartesian d/f functions share shell radial normalization.
        # Recover its component scale from independent diagonal overlaps and
        # closed Gaussian moments; no VibeQC contraction convention enters.
        exponents = [alpha] * len(components[0]) + [beta] * len(components[1])
        raw_norms = [
            gaussian_self_norm(e, name)
            for e, name in zip(exponents, components[0] + components[1])
        ]
        scales = np.sqrt(overlap.diagonal() / raw_norms)
        raw_reference = reference / np.outer(scales, scales)
        positions = (a, b, c) if family == "nuclear_attraction" else (a, b)
        for i, first in enumerate(components[0]):
            for j, second in enumerate(components[1]):
                actual = evaluate_one_electron_primitive(
                    kernel(family, angular, (first, second)), (alpha, beta), positions
                )
                assert actual == pytest.approx(
                    raw_reference[i, len(components[0]) + j], abs=2e-11, rel=3e-12
                )


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
def test_sign_translation_and_shell_exchange(family):
    positions = np.array([[0.2, -0.3, 0.4], [-0.2, 0.5, 0.1], [0.1, -0.2, -0.4]])
    if family != "nuclear_attraction":
        positions = positions[:2]
    first = kernel(family, (3, 2), ("xyz", "xx"))
    second = kernel(family, (2, 3), ("xx", "xyz"))
    value = evaluate_one_electron_primitive(first, (0.7, 1.3), positions)
    translated = evaluate_one_electron_primitive(
        first, (0.7, 1.3), positions + [1.5, -2.2, 0.1]
    )
    swapped = positions.copy()
    swapped[[0, 1]] = swapped[[1, 0]]
    reverse = evaluate_one_electron_primitive(second, (1.3, 0.7), swapped)
    assert translated == pytest.approx(value, abs=2e-13)
    assert reverse == pytest.approx(value, abs=2e-13)
    if family == "nuclear_attraction":
        ss = kernel(family, (0, 0), ("", ""), 1.0)
        assert evaluate_one_electron_primitive(ss, (0.7, 1.3), positions) < 0
        assert first.integral.operator.centers == (0, 1, 2)


def test_kinetic_raised_states_do_not_widen_public_shells():
    program = kernel("kinetic", (3, 3), ("xxx", "xxx"))
    assert program.integral.signature.component_shape == (10, 10)
    assert max(state[2] for state in program.hermite_states) == 5
    assert program.boys_argument is None and program.boys_count == 0
    for invalid in ((4, 0), (-1, 0), (True, 0), (0, 0, 0)):
        with pytest.raises(ValueError):
            build_one_electron_value_ir("overlap", invalid)
    with pytest.raises(ValueError):
        build_one_electron_value_ir("four_center_eri", (0, 0))


def test_interpreter_rejects_invalid_scientific_inputs():
    program = kernel("overlap", (0, 0), ("", ""))
    for exponents, centers in [
        ((0.0, 1.0), [(0, 0, 0)] * 2),
        ((1.0, 1.0), [(0, 0, 0)] * 3),
        ((1.0, 1.0), [(0, 0, math.nan)] * 2),
    ]:
        with pytest.raises(ValueError):
            evaluate_one_electron_primitive(program, exponents, centers)


@pytest.fixture(scope="module")
def emitted_host(tmp_path_factory):
    """Execute emitted arithmetic on CPU to catch lowering/CSE boundary errors.

    This is explicitly a source-lowering test. CUDA compilation, device
    resources and numerical evidence remain separate manual GPU tiers.
    """
    from tools.vibeqc_codegen.one_electron_cuda import emit_one_electron_values_cuda

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("host C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("one-electron-emitted-host")
    source = emit_one_electron_values_cuda().replace("#include <cuda_runtime.h>", "")
    source = (
        source.replace("__device__", "")
        .replace("__forceinline__", "inline")
        .replace("__noinline__", "")
    )
    source += r"""
extern "C" void evaluate(const double* inputs, double* outputs, unsigned count) {
  namespace one = vibeqc::scf::generated_one_electron;
  for (unsigned i = 0; i < count; ++i) {
    const double* p = inputs + 14 * i;
    const auto pair = one::make_pair(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]);
    const auto st = one::overlap_kinetic(pair, p[12], p[13]);
    outputs[3 * i] = st.overlap;
    outputs[3 * i + 1] = st.kinetic;
    outputs[3 * i + 2] = p[11] * one::attraction(pair, p[12], p[13], p[8], p[9], p[10]);
  }
}
"""
    path = directory / "fixture.cpp"
    path.write_text(source)
    library_path = directory / "fixture.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O0",
            "-shared",
            "-fPIC",
            str(path),
            "-o",
            str(library_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    library = ctypes.CDLL(str(library_path))
    pointer = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    library.evaluate.argtypes = [pointer, pointer, ctypes.c_uint]
    library.evaluate.restype = None
    return library.evaluate


def test_emitted_arithmetic_all_pairs_and_normalized_contractions(emitted_host):
    pytest.importorskip("pyscf")
    from tools.vibeqc_validation.one_electron_values import one_electron_value_matrix

    for fixture in one_electron_value_matrix():
        values = np.zeros((len(fixture.records), 3))
        emitted_host(fixture.records, values, len(values))
        actual = fixture.contract(values)
        np.testing.assert_allclose(
            actual,
            fixture.reference,
            atol=1e-11,
            rtol=3e-12,
            err_msg=fixture.inputs["name"],
        )
        np.testing.assert_allclose(
            fixture.spherical(actual),
            fixture.spherical_reference,
            atol=1e-11,
            rtol=3e-12,
            err_msg=fixture.inputs["name"],
        )


def test_one_electron_inventory_retains_operator_and_output_contracts():
    from tools.vibeqc_codegen.one_electron_cuda import one_electron_program_inventory

    inventory = one_electron_program_inventory()
    assert len(inventory["programs"]) == 48
    assert inventory["precision"] == "fp64"
    assert inventory["schedules"] == ["thread", "shell_warp"]
    from tools.vibeqc_codegen.capabilities import query_integral_capability

    request = build_one_electron_value_ir("kinetic", (3, 3))
    assert query_integral_capability(
        request, backend="cuda_one_electron_values"
    ).supported
    # The legacy quartet task ABI must still reject this two-center request.
    assert not query_integral_capability(request, backend="cuda").supported
