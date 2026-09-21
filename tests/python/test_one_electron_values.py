"""Independent scalar value gates before selecting any CUDA schedule."""

import ctypes
import math
import shutil
import subprocess
import typing
from functools import cache
from itertools import product

import numpy as np
import pytest
from vibeqc_compiler.integral.one_electron_cuda import emit_one_electron_values_cuda
from vibeqc_compiler.integral.one_electron_values import (
    build_one_electron_component_kernel,
    build_one_electron_value_ir,
    evaluate_one_electron_primitive,
)
from vibeqc_compiler.integral.shell_spec import cartesian_components


def test_generated_value_header_helpers_have_internal_linkage() -> None:
    """Header-defined noinline device helpers must be reusable by multiple CUDA TUs."""

    source = emit_one_electron_values_cuda()
    assert source.count("static __device__ __noinline__ ST overlap_kinetic_") == 16
    assert source.count("static __device__ __noinline__ double attraction_") == 16
    assert "\n__device__ __noinline__ ST overlap_kinetic_" not in source
    assert "\n__device__ __noinline__ double attraction_" not in source


@cache
def kernel(
    family: typing.Any,
    angular: typing.Any,
    components: typing.Any,
    charge: typing.Any = 2.3,
) -> typing.Any:
    return build_one_electron_component_kernel(
        build_one_electron_value_ir(family, angular, charge=charge), components
    )


def gaussian_self_norm(exponent: typing.Any, component: typing.Any) -> typing.Any:
    """Closed one-dimensional even moments of an unnormalized primitive."""
    result = (math.pi / (2 * exponent)) ** 1.5
    for axis in "xyz":
        n = component.count(axis)
        result *= math.prod(range(1, 2 * n, 2)) / (4 * exponent) ** n
    return result


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
@pytest.mark.parametrize("angular", list(product(range(4), repeat=2)))
def test_every_public_cartesian_component_against_pyscf(
    family: typing.Any, angular: typing.Any
) -> None:
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
@pytest.mark.parametrize("angular", [(4, 0), (0, 4), (4, 4)])
def test_selected_g_cartesian_components_against_pyscf(
    family: typing.Any, angular: typing.Any
) -> None:
    """Qualify representative g-shell values without expanding the broad CI matrix."""

    gto = pytest.importorskip("pyscf.gto")
    a, b, c = (0.2, -0.3, 0.1), (-0.4, 0.15, 0.5), (0.17, -0.11, -0.4)
    alpha, beta = 0.8, 0.35
    charge = 2.3
    components = tuple(cartesian_components(l) for l in angular)
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
    exponents = [alpha] * len(components[0]) + [beta] * len(components[1])
    raw_norms = [
        gaussian_self_norm(exponent, component)
        for exponent, component in zip(
            exponents, components[0] + components[1], strict=True
        )
    ]
    scales = np.sqrt(overlap.diagonal() / raw_norms)
    raw_reference = reference / np.outer(scales, scales)
    first_indices = sorted({0, len(components[0]) // 2, len(components[0]) - 1})
    second_indices = sorted({0, len(components[1]) // 2, len(components[1]) - 1})
    positions = (a, b, c) if family == "nuclear_attraction" else (a, b)
    for i in first_indices:
        for j in second_indices:
            actual = evaluate_one_electron_primitive(
                kernel(family, angular, (components[0][i], components[1][j])),
                (alpha, beta),
                positions,
            )
            assert actual == pytest.approx(
                raw_reference[i, len(components[0]) + j],
                abs=2e-9,
                rel=2e-11,
            )


def test_g_codegen_is_explicit_and_production_capability_stays_fail_closed() -> None:
    from vibeqc_compiler.integral.capabilities import query_integral_capability
    from vibeqc_compiler.integral.one_electron_cuda import (
        _component_layout,
        _emit_component_index,
        _emit_support_cuda,
        _shell_index_expression,
        one_electron_program_inventory,
    )

    counts, offsets, limits, total = _component_layout(4)
    assert counts == (1, 3, 6, 10, 15)
    assert offsets == (0, 1, 4, 10, 20)
    assert limits == (1, 4, 10, 20, 35)
    assert total == 35
    assert len(one_electron_program_inventory()["programs"]) == 48
    assert len(one_electron_program_inventory(4)["programs"]) == 75

    index_source = _emit_component_index(4)
    assert "switch (x * 25U + y * 5U + z)" in index_source
    assert "return 35U;" in index_source
    assert (
        _shell_index_expression("first", limits)
        == "first < 1 ? 0 : first < 4 ? 1 : first < 10 ? 2 : first < 20 ? 3 : 4"
    )
    assert "static_assert(Order <= 8);" in _emit_support_cuda(8)

    request = build_one_electron_value_ir("kinetic", (4, 0))
    production = query_integral_capability(request, backend="cuda_one_electron_values")
    assert not production.supported
    assert "production one-electron CUDA tables support l<=3" in production.reasons[0]
    bounded = query_integral_capability(
        request,
        backend="cuda_bounded_component",
        component_indices=(0,),
    )
    assert not bounded.supported
    assert "first-derivative raw IR" in bounded.reasons[0]
    from vibeqc_compiler.integral.one_electron_derivatives import (
        build_one_electron_derivative_ir,
    )

    derivative = build_one_electron_derivative_ir("kinetic", (4, 0))
    bounded_derivative = query_integral_capability(
        derivative,
        backend="cuda_bounded_component",
        component_indices=(0,),
    )
    assert bounded_derivative.supported


@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
def test_sign_translation_and_shell_exchange(family: typing.Any) -> None:
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


def test_kinetic_raised_states_do_not_widen_public_shells() -> None:
    program = kernel("kinetic", (3, 3), ("xxx", "xxx"))
    assert program.integral.signature.component_shape == (10, 10)
    assert max(state[2] for state in program.hermite_states) == 5
    assert program.boys_argument is None and program.boys_count == 0
    for invalid in ((5, 0), (-1, 0), (True, 0), (0, 0, 0)):
        with pytest.raises(ValueError):
            build_one_electron_value_ir("overlap", invalid)
    with pytest.raises(ValueError):
        build_one_electron_value_ir("four_center_eri", (0, 0))


def test_interpreter_rejects_invalid_scientific_inputs() -> None:
    program = kernel("overlap", (0, 0), ("", ""))
    for exponents, centers in [
        ((0.0, 1.0), [(0, 0, 0)] * 2),
        ((1.0, 1.0), [(0, 0, 0)] * 3),
        ((1.0, 1.0), [(0, 0, math.nan)] * 2),
    ]:
        with pytest.raises(ValueError):
            evaluate_one_electron_primitive(program, exponents, centers)


@pytest.fixture(scope="module")
def emitted_host(tmp_path_factory: typing.Any) -> typing.Any:
    """Execute emitted arithmetic on CPU to catch lowering/CSE boundary errors.

    This is explicitly a source-lowering test. CUDA compilation, device
    resources and numerical evidence remain separate manual GPU tiers.
    """
    from vibeqc_compiler.integral.one_electron_cuda import emit_one_electron_values_cuda

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


def test_emitted_arithmetic_all_pairs_and_normalized_contractions(
    emitted_host: typing.Any,
) -> None:
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


def test_one_electron_inventory_retains_operator_and_output_contracts() -> None:
    from vibeqc_compiler.integral.one_electron_cuda import (
        one_electron_program_inventory,
    )

    inventory = one_electron_program_inventory()
    assert len(inventory["programs"]) == 48
    assert inventory["precision"] == "fp64"
    assert inventory["schedules"] == ["thread", "shell_warp"]
    from vibeqc_compiler.integral.capabilities import query_integral_capability

    request = build_one_electron_value_ir("kinetic", (3, 3))
    assert query_integral_capability(
        request, backend="cuda_one_electron_values"
    ).supported
    # The legacy quartet task ABI must still reject this two-center request.
    assert not query_integral_capability(request, backend="cuda").supported
