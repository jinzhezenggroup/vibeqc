"""Tau must survive prepared dense and spatial XC composition boundaries."""

import ctypes as ct
import shutil
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc_compiler.common.cpp_adapter import CppCompilerAdapter
from vibeqc_compiler.dft import ExplicitGrid, NativeAO
from vibeqc_compiler.dft.features import density_features
from vibeqc_compiler.xc import functional
from vibeqc_compiler.xc.contractions import ContractionProgram
from vibeqc_compiler.xc.native import NativeContractionProgram
from vibeqc_compiler.xc.prepared import PreparedXCContractions


class AnalyticGaussianAO(NativeAO):
    """Supply independent analytic jets without loading the native AO library.

    PreparedXCContractions and generated native XC remain the real consumers.
    This is a composition test, not a qualification of the native AO evaluator.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._handle = ct.c_void_p()
        self.nao, self.natom = 3, 1
        self.numeric_bytes = 1024
        self.identity = "analytic-gaussian-collocation-537"

    def evaluate(self, points, order, **kwargs):
        assert order in (0, 1)
        exponents = np.array([0.4, 0.9, 1.7])
        value = np.exp(-np.sum(points**2, axis=1)[:, None] * exponents)
        jets = np.stack(
            [value] + [-2 * points[:, k, None] * exponents * value for k in range(3)]
        )
        return jets[: 1 if order == 0 else 4]

    def close(self):
        pass


def _check_prepared(tmp_path, spin, basis):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    rng = np.random.default_rng(537)
    points = rng.uniform(-0.6, 0.6, (13, 3))
    weights = rng.uniform(0.1, 0.5, 13)
    grid = ExplicitGrid(points, weights, (0,) * 13, {"units": "Bohr"})
    spec = functional("R2SCAN", spin=spin)
    program = NativeContractionProgram(
        spec, compiler=CppCompilerAdapter(Path(compiler)), cache=tmp_path
    )
    density = np.stack(
        (
            np.eye(basis.nao),
            (0.7 if spin == "polarized" else 1.0) * np.eye(basis.nao),
        )
    )
    reference = ContractionProgram(spec)
    expected = reference.evaluate(basis.evaluate(points, 1), density, weights)
    with PreparedXCContractions(program, basis, grid, tile_points=5) as prepared:
        actual = prepared.execute(density)
        for name in ("energy", "potential", "electrons"):
            np.testing.assert_allclose(
                actual[name], expected[name], atol=2e-12, rtol=2e-12
            )
        assert prepared.statistics["tiles"] == 3
        direction = 0.03 * density
        step = 1e-5
        plus = prepared.execute(density + step * direction)["energy"]
        minus = prepared.execute(density - step * direction)["energy"]
        physical_direction = (
            direction if spin == "polarized" else direction.sum(axis=0)[None, ...]
        )
        np.testing.assert_allclose(
            (plus - minus) / (2 * step),
            np.sum(actual["potential"] * physical_direction),
            atol=2e-8,
            rtol=1e-7,
        )


@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_prepared_meta_gga_keeps_tau_and_density_derivative(tmp_path, spin):
    with AnalyticGaussianAO() as basis:
        _check_prepared(tmp_path, spin, basis)


@pytest.mark.parametrize("spin", ["polarized", "unpolarized"])
def test_real_native_ao_prepared_meta_gga(tmp_path, spin):
    with NativeAO([(1, (0.0, 0.0, -0.7)), (1, (0.0, 0.0, 0.7))]) as basis:
        _check_prepared(tmp_path, spin, basis)


@pytest.mark.parametrize("name", ["LDA_XC_PW", "PBE", "R2SCAN"])
def test_spatial_meta_gga_requests_every_active_ingredient(name):
    """Exercise the actual spatial request boundary, not a source-text check."""
    program = ContractionProgram(functional(name))
    basis = AnalyticGaussianAO()
    points = np.array([[0.2, 0.3, 0.4], [-0.1, 0.3, -0.2]])
    jets = basis.evaluate(points, program.contract.ao_order)
    density = np.stack((np.eye(3), 0.7 * np.eye(3)))
    seen = []

    def tiles(d, *, include_jets, ingredients, order):
        assert include_jets and order == program.contract.ao_order
        seen.extend(ingredients)
        yield SimpleNamespace(
            point_ids=np.arange(2),
            ao_ids=np.arange(3),
            weights=np.ones(2),
            ao_jets=jets,
            features=density_features(jets, d, ingredients=ingredients),
        )

    prepared = object.__new__(PreparedXCContractions)
    prepared.program = program
    prepared.density_grid = None
    prepared.spatial = SimpleNamespace(iter_features=tiles)
    result = list(prepared._collocation(density))
    assert set(program.spec.ingredients) <= set(seen)
    expected = program.features(jets, density)
    assert set(result[0][4]) == set(expected)
    for key, value in expected.items():
        np.testing.assert_array_equal(result[0][4][key], value)


@pytest.mark.parametrize("missing", ["rho", "sigma", "tau"])
def test_active_meta_gga_ingredients_cannot_be_zero_filled(missing):
    program = ContractionProgram(functional("R2SCAN"))
    basis = AnalyticGaussianAO()
    jets = basis.evaluate(np.array([[0.2, 0.3, 0.4]]), 1)
    density = np.stack((np.eye(3), 0.7 * np.eye(3)))
    features = program.features(jets, density)
    del features[missing]
    with pytest.raises(ValueError, match="missing active XC ingredients"):
        program.scalar_values(features)
