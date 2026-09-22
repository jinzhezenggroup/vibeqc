"""BASIS02: g capabilities, independent raw blocks and complete CPU HF paths.

PySCF is an optional test oracle, never a runtime dependency. Heavy molecular
checks are opt-in: VIBEQC_HIGH_L_MOLECULAR_TEST=1. CUDA scalar numerical checks
are separately opt-in: VIBEQC_HIGH_L_CUDA_TEST=1.
"""

import ctypes
import json
import math
import os
import shutil
import subprocess
import time
import typing

import numpy as np
import pytest
from vibeqc import Atom, Calculator, Primitive, Shell, basis_capability, import_bse
from vibeqc_compiler.integral.bounded_component import emit_bounded_component
from vibeqc_compiler.integral.capabilities import query_integral_capability
from vibeqc_compiler.integral.one_electron_derivatives import (
    build_one_electron_derivative_ir,
    build_one_electron_derivative_kernel,
    evaluate_one_electron_derivative_primitive,
)
from vibeqc_compiler.integral.one_electron_values import evaluate_one_electron_primitive

from tools.vibeqc_posthf.sources import NativeSource


def test_capabilities_are_backend_operator_and_derivative_specific() -> None:
    shells = [Shell(0, 4, (Primitive(0.7, 1),))]
    atoms = [("He", (0, 0, 0))]
    for role in ("orbital", "auxiliary"):
        for operator in (
            "overlap",
            "kinetic",
            "nuclear_attraction",
            "eri",
            "df_metric",
            "df_three_center",
        ):
            for order in (0, 1):
                assert basis_capability(
                    shells, atoms, role=role, operator=operator, derivative_order=order
                )["eligible"]
                assert not basis_capability(
                    shells,
                    atoms,
                    backend="cuda",
                    role=role,
                    operator=operator,
                    derivative_order=order,
                )["eligible"]
    assert not basis_capability(shells, atoms, operator="ao")["eligible"]
    assert not basis_capability(shells, atoms, derivative_order=2)["eligible"]
    ir = build_one_electron_derivative_ir("kinetic", (4, 0))
    assert not query_integral_capability(
        ir, backend="cuda_one_electron_derivatives"
    ).supported
    assert query_integral_capability(
        ir, backend="cuda_bounded_component", component_indices=(0,)
    ).supported
    assert not query_integral_capability(ir, backend="cuda_bounded_component").supported
    assert not query_integral_capability(
        ir, backend="cpu_bounded_component", component_indices=(15,)
    ).supported
    with pytest.raises(ValueError):
        build_one_electron_derivative_ir("kinetic", (5, 0))


@pytest.mark.parametrize("angular", [(4, 0), (0, 4), (4, 2), (4, 4)])
@pytest.mark.parametrize("family", ["overlap", "kinetic", "nuclear_attraction"])
def test_g_one_electron_all_components_libcint(
    family: typing.Any, angular: typing.Any
) -> None:
    from test_one_electron_derivatives import (
        test_all_cartesian_derivatives_match_independent_libcint as derivatives,
    )
    from test_one_electron_values import (
        test_every_public_cartesian_component_against_pyscf as values,
    )

    values(family, angular)
    derivatives(family, angular)


@pytest.mark.parametrize("angular", [(4, 0), (4, 4), (4, 0, 0), (0, 0, 4), (2, 1, 4)])
def test_g_df_independent_derivatives(angular: typing.Any) -> None:
    from test_df_derivatives import (
        test_physical_value_dag_derivatives_match_libcint as check,
    )

    check(angular)


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_g_contracted_raw_blocks_and_spherical_order(
    representation: typing.Any,
) -> None:
    gto = pytest.importorskip("pyscf.gto")
    atoms = [("He", (0.2, -0.3, 0.1)), ("H", (-0.4, 0.15, 1.2))]
    basis = [
        Shell(0, 4, (Primitive(0.8, 0.7), Primitive(0.3, -0.2))),
        Shell(1, 0, (Primitive(0.6, 1),)),
    ]
    mol = gto.M(
        atom=atoms,
        basis={"He": [[4, [0.8, 0.7], [0.3, -0.2]]], "H": [[0, [0.6, 1]]]},
        unit="Bohr",
        charge=1,
        cart=representation == "cartesian",
        verbose=0,
    )
    n = mol.nao_nr()
    overlap = mol.intor("int1e_ovlp")
    scale = 1 / np.sqrt(overlap.diagonal())
    with NativeSource(
        atoms, basis, auxiliary_basis=basis, charge=1, representation=representation
    ) as source:
        actual = source._read("overlap", (0, 0), (n, n))
        np.testing.assert_allclose(actual, overlap * np.outer(scale, scale), atol=2e-12)
        if representation == "spherical":
            np.testing.assert_allclose(actual[:9, :9], np.eye(9), atol=2e-12)
        hcore = mol.intor("int1e_kin") + mol.intor("int1e_nuc")
        np.testing.assert_allclose(
            source._read("hcore", (0, 0), (n, n)),
            hcore * np.outer(scale, scale),
            atol=2e-11,
        )
        eri = mol.intor("int2e", aosym="s1") * np.einsum(
            "i,j,k,l->ijkl", scale, scale, scale, scale
        )
        # gsss, gsgs and gggg component tiles, including pair/pair exchange.
        for begin, shape in [
            ((0, n - 1, n - 1, n - 1), (n - 1, 1, 1, 1)),
            ((0, n - 1, 0, n - 1), (2, 1, 2, 1)),
            ((0, 0, 0, 0), (2, 2, 2, 2)),
        ]:
            expected = eri[tuple(slice(b, b + s) for b, s in zip(begin, shape))]
            np.testing.assert_allclose(
                source._read("four_center_eri", begin, shape), expected, atol=3e-11
            )
        metric = mol.intor("int2c2e") * np.outer(scale, scale)
        np.testing.assert_allclose(
            source._read("coulomb_metric", (0, 0), (n, n)), metric, atol=4e-11
        )
        from pyscf import df

        three = df.incore.aux_e2(mol, mol, intor="int3c2e", aosym="s1")
        three *= np.einsum("i,j,k->ijk", scale, scale, scale)
        np.testing.assert_allclose(
            source._read("three_center_eri", (0, n - 1, 0), (2, 1, 2)),
            three[:2, n - 1 : n, :2],
            atol=4e-11,
        )


def molecular_basis(tmp_path: typing.Any, representation: typing.Any) -> typing.Any:
    data = {
        "elements": {},
        "name": "heh-g-regression",
        "version": "1",
        "function_types": ["gto"],
        "molssi_bse_schema": {"schema_type": "complete", "schema_version": "0.1"},
    }
    for z, shells in (("2", [(0, 1.5), (4, 0.6)]), ("1", [(0, 1.2)])):
        data["elements"][z] = {
            "electron_shells": [
                {
                    "angular_momentum": [l],
                    "exponents": [str(e)],
                    "coefficients": [["1"]],
                    "function_type": "gto",
                }
                for l, e in shells
            ]
        }
    path = tmp_path / "heh-g.json"
    path.write_text(json.dumps(data))
    return import_bse(
        path,
        source="BASIS02 synthetic HeH+ g regression",
        source_version="1",
        license="CC0-1.0",
        representation=representation,
    )


@pytest.mark.parametrize("distance", [0.0, 1e-8, 3.0, 5.0, 12.0, 50.0])
def test_native_gggg_boys_regimes(distance: typing.Any) -> None:
    gto = pytest.importorskip("pyscf.gto")
    atoms = [("He", (0.0, 0.0, 0.0)), ("H", (distance, 0.0, 0.0))]
    basis = [Shell(0, 4, (Primitive(0.8, 1),)), Shell(1, 4, (Primitive(0.35, 1),))]
    mol = gto.M(
        atom=atoms,
        basis={"He": [[4, [0.8, 1]]], "H": [[4, [0.35, 1]]]},
        unit="Bohr",
        charge=1,
        cart=True,
        verbose=0,
    )
    scale = 1 / np.sqrt(mol.intor("int1e_ovlp").diagonal())
    reference = mol.intor_by_shell("int2e_cart", (0, 0, 1, 1))[0, 0, 0, 0]
    reference *= scale[0] ** 2 * scale[15] ** 2
    with NativeSource(atoms, basis, charge=1) as source:
        actual = source._read("four_center_eri", (0, 0, 15, 15), (1, 1, 1, 1)).item()
    assert actual == pytest.approx(reference, abs=3e-12, rel=3e-11)


@pytest.mark.skipif(
    os.environ.get("VIBEQC_HIGH_L_MOLECULAR_TEST") != "1",
    reason="opt-in CPU g molecular endpoint",
)
@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_loaded_g_rhf_energy_and_forces(
    tmp_path: typing.Any, representation: typing.Any
) -> None:
    from pyscf import gto, scf

    atoms = [("He", (0.1, -0.2, -0.7)), ("H", (-0.2, 0.1, 0.7))]
    basis = molecular_basis(tmp_path, representation)
    assert any(
        s.angular_momentum == 4
        for s in basis.shells_for([Atom.from_value(a) for a in atoms])
    )
    mol = gto.M(
        atom=atoms,
        basis={"He": [[0, [1.5, 1]], [4, [0.6, 1]]], "H": [[0, [1.2, 1]]]},
        unit="Bohr",
        charge=1,
        cart=representation == "cartesian",
        verbose=0,
    )
    oracle = scf.RHF(mol)
    oracle.conv_tol = 1e-13
    oracle.kernel()
    reference_forces = -oracle.nuc_grad_method().kernel()
    start = time.perf_counter()
    result = Calculator(
        basis=basis, energy_tolerance=1e-12, density_tolerance=1e-10
    ).singlepoint(atoms, charge=1)
    assert result.converged and oracle.converged
    np.testing.assert_allclose(result.energy, oracle.e_tot, atol=3e-10, rtol=0)
    np.testing.assert_allclose(result.forces, reference_forces, atol=3e-9, rtol=0)
    np.testing.assert_allclose(np.sum(result.forces, axis=0), 0, atol=3e-12)
    print(
        json.dumps(
            {
                "representation": representation,
                "nao": mol.nao_nr(),
                "energy": result.energy,
                "energy_error": result.energy - oracle.e_tot,
                "force_error": float(np.max(np.abs(result.forces - reference_forces))),
                "seconds": time.perf_counter() - start,
            }
        )
    )


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_g_auxiliary_df_rhf_force_path(representation: typing.Any) -> None:
    pytest.importorskip("pyscf")
    from pyscf import gto, scf

    atoms = [("He", (0.1, -0.2, -0.7)), ("H", (-0.2, 0.1, 0.7))]
    orbital = [Shell(0, 0, (Primitive(1.5, 1),)), Shell(1, 0, (Primitive(1.2, 1),))]
    auxiliary = [
        Shell(0, 0, (Primitive(2.0, 1),)),
        Shell(0, 4, (Primitive(0.6, 1),)),
        Shell(1, 0, (Primitive(2.4, 1),)),
    ]
    mol = gto.M(
        atom=atoms,
        basis={"He": [[0, [1.5, 1]]], "H": [[0, [1.2, 1]]]},
        unit="Bohr",
        charge=1,
        cart=representation == "cartesian",
        verbose=0,
    )
    oracle = scf.RHF(mol).density_fit(
        auxbasis={"He": [[0, [2.0, 1]], [4, [0.6, 1]]], "H": [[0, [2.4, 1]]]}
    )
    oracle.conv_tol = 1e-13
    oracle.kernel()
    expected = -oracle.nuc_grad_method().kernel()
    result = Calculator(
        basis=orbital,
        auxiliary_basis=auxiliary,
        density_fitting=True,
        basis_representation=representation,
        energy_tolerance=1e-12,
        density_tolerance=1e-10,
    ).singlepoint(atoms, charge=1)
    assert result.converged and oracle.converged
    np.testing.assert_allclose(result.energy, oracle.e_tot, atol=3e-10, rtol=0)
    np.testing.assert_allclose(result.forces, expected, atol=3e-9, rtol=0)
    print(
        json.dumps(
            {
                "method": "df-rhf",
                "representation": representation,
                "energy_error": result.energy - oracle.e_tot,
                "force_error": float(np.max(np.abs(result.forces - expected))),
            }
        )
    )


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
@pytest.mark.parametrize(
    "family",
    ["overlap", "kinetic", "nuclear_attraction", "coulomb_metric", "three_center_eri"],
)
def test_bounded_g_component_compiled_arithmetic(
    tmp_path: typing.Any, backend: typing.Any, family: typing.Any
) -> None:
    if backend == "cuda" and os.environ.get("VIBEQC_HIGH_L_CUDA_TEST") != "1":
        pytest.skip("opt-in allocated CUDA device")
    compiler = shutil.which("nvcc" if backend == "cuda" else "c++")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    is_df = family in ("coulomb_metric", "three_center_eri")
    if is_df:
        from vibeqc_compiler.integral.df_derivatives import (
            build_df_derivative_ir,
            build_df_derivative_kernel,
            evaluate_df_derivative,
        )
        from vibeqc_compiler.integral.df_values import (
            build_df_component_kernel,
            build_df_value_ir,
            evaluate_df_primitive,
        )

        angular = (0, 0, 4) if family == "three_center_eri" else (4, 4)
        components = ("", "", "xyyz") if len(angular) == 3 else ("xxxy", "xyzz")
        ir = build_df_derivative_ir(family, angular)
        program = build_df_derivative_kernel(ir, components)
        value_program = build_df_component_kernel(
            build_df_value_ir(family, angular), components
        )
    else:
        ir = build_one_electron_derivative_ir(family, (4, 4))
        components = ("xxxy", "xyzz")
        program = build_one_electron_derivative_kernel(ir, components)
    exponent_count = len(ir.signature.shells)
    input_count = exponent_count + 3 * len(ir.operator.centers)
    source = emit_bounded_component(ir, components, backend=backend)
    count = 1 + 3 * len(ir.operator.centers)
    if backend == "cuda":
        source = (
            "#include <cuda_runtime.h>\n"
            + source
            + f"""
__global__ void worker(const double* x, double* y) {{ evaluate(x,y); }}
extern "C" int launch(const double* x, double* y) {{
  double *dx=nullptr, *dy=nullptr;
  cudaError_t status = cudaMalloc(&dx, {input_count}*sizeof(double));
  if (status != cudaSuccess) return status;
  status = cudaMalloc(&dy, {count}*sizeof(double));
  if (status == cudaSuccess) status = cudaMemcpy(dx,x,{input_count}*sizeof(double),cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {{ worker<<<1,1>>>(dx,dy); status=cudaGetLastError(); }}
  if (status == cudaSuccess) status = cudaMemcpy(y,dy,{count}*sizeof(double),cudaMemcpyDeviceToHost);
  cudaFree(dx); cudaFree(dy); return status;
}}
"""
        )
    path = tmp_path / ("component.cu" if backend == "cuda" else "component.cpp")
    path.write_text(source)
    library = tmp_path / "component.so"
    command = [compiler, "-std=c++17", "-O1", "-shared"]
    cuda_arch = os.environ.get("VIBEQC_HIGH_L_CUDA_ARCH", "sm_89")
    command += (
        ["-Xcompiler", "-fPIC", f"-arch={cuda_arch}", "-Xptxas=-v"]
        if backend == "cuda"
        else ["-fPIC"]
    )
    start = time.perf_counter()
    build = subprocess.run(
        command + [str(path), "-o", str(library)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    lib = ctypes.CDLL(str(library))
    evaluate = lib.launch if backend == "cuda" else lib.evaluate
    evaluate.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ] * 2
    evaluate.restype = ctypes.c_int if backend == "cuda" else None
    print(
        json.dumps(
            {
                "backend": backend,
                "family": family,
                "cuda_arch": cuda_arch if backend == "cuda" else None,
                "compile_seconds": time.perf_counter() - start,
                "source_bytes": path.stat().st_size,
                "binary_bytes": library.stat().st_size,
                "compiler_resources": build.stderr,
            }
        )
    )
    weights = np.random.default_rng(170).normal(size=3)
    for distance in (0.0, 0.7, 4.0, 30.0):
        centers = np.array(
            [
                [0.0, 0.0, 0.0],
                [distance, -0.1 * distance, 0.2 * distance],
                [0.3, 0.2, -0.4],
            ]
        )
        centers = centers[: len(ir.operator.centers)]
        inputs = np.zeros(input_count)
        inputs[:exponent_count] = [0.8, 0.35, 1.1][:exponent_count]
        inputs[exponent_count:] = centers.ravel()
        out = np.empty(count)
        status = evaluate(inputs, out)
        assert status in (None, 0)
        if is_df:
            value = evaluate_df_primitive(
                value_program, inputs[:exponent_count], centers
            )
            gradient = np.array(
                evaluate_df_derivative(program, inputs[:exponent_count], centers)
            )
        else:
            value = evaluate_one_electron_primitive(program, inputs[:2], centers)
            gradient = np.array(
                evaluate_one_electron_derivative_primitive(program, inputs[:2], centers)
            )
        np.testing.assert_allclose(
            out, np.r_[value, gradient.ravel()], atol=3e-11, rtol=3e-11
        )
        np.testing.assert_allclose(out[1:].reshape(-1, 3).sum(axis=0), 0, atol=3e-12)
        # Arbitrary scalar cotangent, with no HF density/sign assumptions.
        for weight in weights:
            np.testing.assert_allclose(
                weight * out[1:], weight * gradient.ravel(), atol=5e-11
            )


def test_bounded_four_center_cpu_codegen_compiles_and_executes(
    tmp_path: typing.Any,
) -> None:
    """Lower one ERI component through the shared compiler DAG to native C++."""

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native compiler unavailable")

    from vibeqc_compiler.integral.blocks import RawBlock, TensorLayout
    from vibeqc_compiler.integral.ir import FOUR_CENTER_ERI_OPERATOR, build_integral_ir
    from vibeqc_compiler.integral.shell_signature import ShellSignature
    from vibeqc_compiler.integral.shell_spec import PSSS_SPEC

    signature = ShellSignature.from_shell_class(PSSS_SPEC)
    layout = TensorLayout(
        ("center", "xyz", *signature.tensor_indices),
        (4, 3, *signature.component_shape),
    )
    ir = build_integral_ir(
        PSSS_SPEC,
        operator=FOUR_CENTER_ERI_OPERATOR,
        derivative=FOUR_CENTER_ERI_OPERATOR.nuclear_derivative(),
        contractions=(RawBlock(layout, layout.storage_bytes),),
    )
    assert query_integral_capability(
        ir, backend="cpu_bounded_component", component_indices=(0,)
    ).supported

    source = emit_bounded_component(ir, ("x", "", "", ""), backend="cpu")
    assert source == emit_bounded_component(ir, ("x", "", "", ""), backend="cpu")
    assert 'extern "C" void evaluate' in source
    assert "__device__" not in source

    path = tmp_path / "eri_component.cpp"
    library = tmp_path / "eri_component.so"
    path.write_text(source)
    build = subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O1",
            "-shared",
            "-fPIC",
            str(path),
            "-o",
            str(library),
        ],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    evaluate = ctypes.CDLL(str(library)).evaluate
    evaluate.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ] * 2
    evaluate.restype = None
    inputs = np.array(
        [
            1.3,
            0.7,
            0.9,
            0.5,
            0.1,
            -0.3,
            0.2,
            -0.4,
            0.2,
            0.5,
            0.6,
            -0.1,
            -0.2,
            -0.2,
            0.4,
            -0.6,
        ],
        dtype=np.float64,
    )
    outputs = np.empty(13)
    evaluate(inputs, outputs)
    assert np.isfinite(outputs).all()

    step = 2.0e-6
    numerical = np.empty(12)
    for coordinate in range(12):
        plus, minus = inputs.copy(), inputs.copy()
        plus[4 + coordinate] += step
        minus[4 + coordinate] -= step
        plus_output, minus_output = np.empty(13), np.empty(13)
        evaluate(plus, plus_output)
        evaluate(minus, minus_output)
        numerical[coordinate] = (plus_output[0] - minus_output[0]) / (2.0 * step)
    np.testing.assert_allclose(outputs[1:], numerical, atol=2e-9, rtol=2e-8)
    np.testing.assert_allclose(outputs[1:].reshape(4, 3).sum(axis=0), 0.0, atol=3e-12)


def _raw_four_center_ir(
    angular: tuple[int, int, int, int], *, convention: str = "cartesian"
) -> typing.Any:
    from vibeqc_compiler.integral.blocks import RawBlock, TensorLayout
    from vibeqc_compiler.integral.ir import FOUR_CENTER_ERI_OPERATOR, build_integral_ir
    from vibeqc_compiler.integral.shell_signature import (
        BasisShell,
        CenterBinding,
        ShellSignature,
    )

    signature = ShellSignature(
        tuple(
            BasisShell(slot, slot, value, convention=convention)
            for slot, value in enumerate(angular)
        ),
        tuple(CenterBinding(center) for center in range(4)),
    )
    layout = TensorLayout(
        ("center", "xyz", *signature.tensor_indices),
        (4, 3, *signature.component_shape),
    )
    return build_integral_ir(
        signature,
        operator=FOUR_CENTER_ERI_OPERATOR,
        derivative=FOUR_CENTER_ERI_OPERATOR.nuclear_derivative(),
        contractions=(RawBlock(layout, layout.storage_bytes),),
    )


def test_g_four_center_capability_is_generic_but_not_native_cuda() -> None:
    ir = _raw_four_center_ir((4, 0, 0, 0))
    for backend in ("cpu_bounded_component", "cuda_bounded_component"):
        report = query_integral_capability(ir, backend=backend, component_indices=(0,))
        assert report.supported, report.reasons
        assert report.schedules == ("explicit_scalar_component_v1",)

    production = query_integral_capability(ir, backend="cuda")
    assert not production.supported
    assert any("legacy ShellClassSpec" in reason for reason in production.reasons)

    spherical = _raw_four_center_ir((4, 0, 0, 0), convention="real_spherical")
    report = query_integral_capability(
        spherical, backend="cpu_bounded_component", component_indices=(0,)
    )
    assert not report.supported
    assert any("Cartesian primitive signatures" in reason for reason in report.reasons)


def _libcint_raw_gsss(
    exponents: np.ndarray, centers: np.ndarray
) -> tuple[float, np.ndarray]:
    gto = pytest.importorskip("pyscf.gto")
    angular = (4, 0, 0, 0)
    labels = [f"H{i}" for i in range(4)]
    mol = gto.M(
        atom=list(zip(labels, centers)),
        basis={
            label: [[value, [exponent, 1.0]]]
            for label, value, exponent in zip(labels, angular, exponents)
        },
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    from vibeqc_compiler.integral.shell_spec import cartesian_components

    normalization = []
    for slot, (value, exponent) in enumerate(zip(angular, exponents)):
        raw_overlap = []
        for component in cartesian_components(value):
            powers = tuple(component.count(axis) for axis in "xyz")
            raw_overlap.append(
                (math.pi / (2 * exponent)) ** 1.5
                * math.prod(math.prod(range(1, 2 * power, 2)) for power in powers)
                / (4 * exponent) ** value
            )
        normalization.append(
            np.sqrt(
                np.diag(mol.intor_by_shell("int1e_ovlp", (slot, slot))) / raw_overlap
            )
        )
    factors = np.einsum("i,j,k,l->ijkl", *normalization)
    raw_value = mol.intor_by_shell("int2e", (0, 1, 2, 3)) / factors
    raw_gradients = []
    for permutation in ((0, 1, 2, 3), (1, 0, 2, 3), (2, 3, 0, 1), (3, 2, 0, 1)):
        block = mol.intor_by_shell("int2e_ip1", permutation)
        block = np.transpose(block, (0, *(permutation.index(i) + 1 for i in range(4))))
        raw_gradients.append(-block / factors)
    return float(raw_value[0, 0, 0, 0]), np.asarray(raw_gradients)[:, :, 0, 0, 0, 0]


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_bounded_gsss_four_center_matches_libcint(
    tmp_path: typing.Any, backend: str
) -> None:
    if backend == "cuda" and os.environ.get("VIBEQC_HIGH_L_CUDA_TEST") != "1":
        pytest.skip("opt-in allocated CUDA device")
    compiler = shutil.which("nvcc" if backend == "cuda" else "c++")
    if compiler is None:
        pytest.skip("native compiler unavailable")

    ir = _raw_four_center_ir((4, 0, 0, 0))
    report = query_integral_capability(
        ir, backend=f"{backend}_bounded_component", component_indices=(0,)
    )
    assert report.supported, report.reasons
    source = emit_bounded_component(ir, ("xxxx", "", "", ""), backend=backend)
    input_count, output_count = 16, 13
    if backend == "cuda":
        source = (
            "#include <cuda_runtime.h>\n"
            + source
            + f"""
__global__ void worker(const double* x, double* y) {{ evaluate(x,y); }}
extern "C" int launch(const double* x, double* y) {{
  double *dx=nullptr, *dy=nullptr;
  cudaError_t status = cudaMalloc(&dx, {input_count}*sizeof(double));
  if (status != cudaSuccess) return status;
  status = cudaMalloc(&dy, {output_count}*sizeof(double));
  if (status == cudaSuccess) status = cudaMemcpy(dx,x,{input_count}*sizeof(double),cudaMemcpyHostToDevice);
  if (status == cudaSuccess) {{ worker<<<1,1>>>(dx,dy); status=cudaGetLastError(); }}
  if (status == cudaSuccess) status = cudaMemcpy(y,dy,{output_count}*sizeof(double),cudaMemcpyDeviceToHost);
  cudaFree(dx); cudaFree(dy); return status;
}}
"""
        )
    path = tmp_path / ("gsss.cu" if backend == "cuda" else "gsss.cpp")
    library = tmp_path / "gsss.so"
    path.write_text(source)
    cuda_arch = os.environ.get("VIBEQC_HIGH_L_CUDA_ARCH", "sm_89")
    command = [compiler, "-std=c++17", "-O1", "-shared"]
    command += (
        ["-Xcompiler", "-fPIC", f"-arch={cuda_arch}", "-Xptxas=-v"]
        if backend == "cuda"
        else ["-fPIC"]
    )
    started = time.perf_counter()
    build = subprocess.run(
        command + [str(path), "-o", str(library)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    compile_seconds = time.perf_counter() - started
    native = (
        ctypes.CDLL(str(library)).launch
        if backend == "cuda"
        else ctypes.CDLL(str(library)).evaluate
    )
    native.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    ] * 2
    native.restype = ctypes.c_int if backend == "cuda" else None
    print(
        json.dumps(
            {
                "backend": backend,
                "family": "four_center_eri",
                "angular": [4, 0, 0, 0],
                "cuda_arch": cuda_arch if backend == "cuda" else None,
                "compile_seconds": compile_seconds,
                "source_bytes": path.stat().st_size,
                "binary_bytes": library.stat().st_size,
                "compiler_resources": build.stderr,
            }
        )
    )

    exponents = np.array([0.8, 0.6, 1.1, 0.45])
    geometries = (
        np.array(
            [[0.0, 0.0, 0.0], [0.7, -0.2, 0.1], [-0.4, 0.8, 0.3], [0.3, -0.5, 1.2]]
        ),
        np.array(
            [[0.0, 0.0, 0.0], [1.1, 0.1, -0.2], [-0.2, 0.5, 0.7], [0.6, -0.3, 1.5]]
        ),
    )
    weights = np.array([-1.7, 0.25, 2.3])
    for centers in geometries:
        inputs = np.r_[exponents, centers.ravel()].astype(np.float64)
        output = np.empty(output_count)
        status = native(inputs, output)
        assert status in (None, 0)
        reference, gradients = _libcint_raw_gsss(exponents, centers)
        np.testing.assert_allclose(output[0], reference, atol=2e-11, rtol=2e-10)
        np.testing.assert_allclose(
            output[1:].reshape(4, 3), gradients, atol=3e-10, rtol=3e-9
        )
        np.testing.assert_allclose(output[1:].reshape(4, 3).sum(axis=0), 0, atol=2e-11)
        for weight in weights:
            np.testing.assert_allclose(
                weight * output[1:], weight * gradients.ravel(), atol=6e-10, rtol=3e-9
            )
