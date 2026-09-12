"""BASIS02: g capabilities, independent raw blocks and complete CPU HF paths.

PySCF is an optional test oracle, never a runtime dependency. Heavy molecular
checks are opt-in: VIBEQC_HIGH_L_MOLECULAR_TEST=1. CUDA scalar numerical checks
are separately opt-in: VIBEQC_HIGH_L_CUDA_TEST=1.
"""

import ctypes
import json
import os
import shutil
import subprocess
import time

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


def test_capabilities_are_backend_operator_and_derivative_specific():
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
def test_g_one_electron_all_components_libcint(family, angular):
    from test_one_electron_derivatives import (
        test_all_cartesian_derivatives_match_independent_libcint as derivatives,
    )
    from test_one_electron_values import (
        test_every_public_cartesian_component_against_pyscf as values,
    )

    values(family, angular)
    derivatives(family, angular)


@pytest.mark.parametrize("angular", [(4, 0), (4, 4), (4, 0, 0), (0, 0, 4), (2, 1, 4)])
def test_g_df_independent_derivatives(angular):
    from test_df_derivatives import (
        test_physical_value_dag_derivatives_match_libcint as check,
    )

    check(angular)


@pytest.mark.parametrize("representation", ["cartesian", "spherical"])
def test_g_contracted_raw_blocks_and_spherical_order(representation):
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


def molecular_basis(tmp_path, representation):
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
def test_native_gggg_boys_regimes(distance):
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
def test_loaded_g_rhf_energy_and_forces(tmp_path, representation):
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
def test_g_auxiliary_df_rhf_force_path(representation):
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
def test_bounded_g_component_compiled_arithmetic(tmp_path, backend, family):
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
    command += (
        ["-Xcompiler", "-fPIC", "-arch=sm_89", "-Xptxas=-v"]
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
