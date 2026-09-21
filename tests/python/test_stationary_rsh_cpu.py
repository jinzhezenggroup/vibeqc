"""CPU RSH stationary exchange binding against an independent integral oracle."""

import shutil
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc._stationary_cpu import _PrimitiveExecutor
from vibeqc._stationary_rsh_cpu import RangeExchangeExecutor
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.method import resolve_method
from vibeqc_compiler.method.spec import RangeSeparatedExchangePrimitive
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.tensor import execute

ATOMS = [
    ("H", (0.13, -0.17, -0.71)),
    ("H", (-0.09, 0.11, 0.79)),
]


def _fake_s_basis() -> SimpleNamespace:
    centers = np.array(
        [
            [0.13, -0.31, 0.24],
            [-0.43, 0.27, 0.51],
            [0.68, -0.14, -0.22],
            [-0.21, 0.48, -0.63],
        ]
    )
    primitives = np.array([[0.57, 0.83], [0.71, -0.19], [0.89, 0.67], [1.13, 0.42]])
    aos = np.zeros((4, 16))
    for index in range(4):
        aos[index, :8] = (index, index, 1, 1, 0, 0, 0, 1)
    return SimpleNamespace(
        natom=4,
        nao=4,
        nprimitive=4,
        shells=tuple(SimpleNamespace(angular_momentum=0) for _ in range(4)),
        packed=np.concatenate((centers.ravel(), primitives.ravel(), aos.ravel())),
    )


def test_range_exchange_radial_partition_matches_full_coulomb(
    tmp_path: Path,
) -> None:
    compiler_path = shutil.which("c++")
    if compiler_path is None:
        pytest.skip("native C++ compiler unavailable")
    compiler = CppCompilerAdapter(Path(compiler_path))
    basis = _fake_s_basis()
    method = resolve_method("CAM-B3LYP", spin="unpolarized")
    ranges = {
        primitive.operator: primitive
        for primitive in method.primitives
        if type(primitive) is RangeSeparatedExchangePrimitive
    }
    full = _PrimitiveExecutor(basis, tmp_path / "full", 8, compiler)
    expected_owners, expected = full.integral("four_center_eri", (0, 1, 2, 3), 1.0)
    ranged = RangeExchangeExecutor(basis, tmp_path / "range", 8, compiler)
    try:
        actual_owners, short = ranged.integral(ranges["short-range"], (0, 1, 2, 3), 1.0)
        _, long = ranged.integral(ranges["long-range"], (0, 1, 2, 3), 1.0)
    finally:
        ranged.close()
    assert actual_owners == expected_owners == [0, 1, 2, 3]
    np.testing.assert_allclose(short + long, expected, atol=3e-12, rtol=3e-11)
    np.testing.assert_allclose((short + long).sum(axis=0), 0, atol=2e-12, rtol=0)


def _pyscf_energy(
    basis: NativeAO,
    density: np.ndarray,
    primitive: RangeSeparatedExchangePrimitive,
    coordinates: np.ndarray,
) -> float:
    from pyscf import gto
    from pyscf.data.elements import ELEMENTS

    labels = [
        f"{ELEMENTS[atom.atomic_number]}{index}"
        for index, atom in enumerate(basis.atoms)
    ]
    shells = {label: [] for label in labels}
    for shell in basis.shells:
        shells[labels[shell.atom_index]].append(
            [
                shell.angular_momentum,
                *[(p.exponent, p.coefficient) for p in shell.primitives],
            ]
        )
    mol = gto.M(
        atom=list(zip(labels, coordinates, strict=True)),
        basis=shells,
        unit="Bohr",
        cart=True,
        charge=basis.charge,
        spin=basis.multiplicity - 1,
        verbose=0,
    )
    omega = float(primitive.omega)
    signed_omega = -omega if primitive.operator == "short-range" else omega
    with mol.with_range_coulomb(signed_omega):
        eri = mol.intor("int2e")
    return (
        -float(primitive.coefficient)
        * np.einsum("ij,kl,ikjl->", density, density, eri)
        / 4
    )


@pytest.mark.parametrize("operator", ("short-range", "long-range"))
def test_range_exchange_executor_matches_fixed_density_finite_difference(
    tmp_path: Path, operator: str
) -> None:
    pytest.importorskip(
        "pyscf", reason="independent RSH integral oracle requires PySCF"
    )
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("native C++ compiler unavailable")

    with NativeAO(ATOMS, basis="sto-3g") as basis:
        method = resolve_method("CAM-B3LYP", spin="unpolarized")
        primitive = next(
            item
            for item in method.primitives
            if type(item) is RangeSeparatedExchangePrimitive
            and item.operator == operator
        )
        plan = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
        source = {
            "short-range": "exchange_short_range",
            "long-range": "exchange_long_range",
        }[operator]
        n = basis.nao
        density = np.array([[1.17, -0.23], [-0.23, 0.81]])
        tuples = tuple(product(range(n), repeat=4))
        ids = np.asarray(tuples)
        block = plan.integral_block(source, terms=len(tuples))
        weights = execute(
            block.weights,
            {
                "density_left": density[None, ids[:, 0], ids[:, 2]],
                "density_right": density[None, ids[:, 1], ids[:, 3]],
            },
        ).outputs["weights"]

        actual = np.zeros((basis.natom, 3))
        executor = RangeExchangeExecutor(
            basis, tmp_path, 32, CppCompilerAdapter(Path(compiler))
        )
        try:
            for indices, weight in zip(tuples, weights, strict=True):
                owners, values = executor.integral(primitive, indices, weight)
                np.add.at(actual, owners, values)
            assert executor.records > 0
        finally:
            executor.close()

        coordinates = np.asarray([atom.position for atom in basis.atoms])
        expected = np.zeros_like(actual)
        step = 1.0e-4
        for atom, axis in product(range(basis.natom), range(3)):
            displacement = np.zeros_like(coordinates)
            displacement[atom, axis] = step
            plus = _pyscf_energy(basis, density, primitive, coordinates + displacement)
            minus = _pyscf_energy(basis, density, primitive, coordinates - displacement)
            expected[atom, axis] = (plus - minus) / (2 * step)

        np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-7)
        np.testing.assert_allclose(actual.sum(axis=0), 0, atol=2e-11, rtol=0)
