"""Complete bounded LDA/PBE RKS molecular HVP acceptance."""

import typing

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._stationary_cpu import complete_rks_gradient_diagnostic
from vibeqc_compiler.dft import NativeAO

from tools.vibeqc_hessian import rks_hvp
from tools.vibeqc_response import NativeRKSResponse

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
GRID = GridSpec(radial_points=12, angular_polar=4, angular_azimuth=8)
SOURCE_ORDER = (
    "one_electron",
    "coulomb",
    "xc_ao",
    "xc_grid",
    "xc_weight",
    "overlap_pulay",
    "nuclear",
)


def _calculator(method: str) -> Calculator:
    return Calculator(
        method=method,
        device="cpu",
        ks_options=KsOptions(grid=GRID),
        max_iterations=200,
        energy_tolerance=1e-13,
        density_tolerance=1e-11,
    )


def _gradient(method: str, atoms: typing.Any, cache: typing.Any) -> np.ndarray:
    with _calculator(method).prepare_batch([atoms]) as batch, NativeAO(atoms) as basis:
        batch.execute(strict=True)
        state = StationaryKsState.from_native(batch, basis)
        try:
            result = complete_rks_gradient_diagnostic(
                state,
                basis,
                cache=cache,
                tile_points=257,
                integral_terms=16,
                primitive_tile=64,
                execution="reference",
            )
            return np.array(result.gradient, copy=True)
        finally:
            state._source.close()


@pytest.mark.parametrize("method", ("lda-rks", "pbe-rks"))
def test_complete_rks_hvp_matches_reconverged_gradient_difference(
    method: str, tmp_path: typing.Any
) -> None:
    direction = np.array([[0.13, -0.21, 0.31], [-0.17, 0.09, -0.05]], dtype=np.float64)
    direction /= np.linalg.norm(direction)
    with _calculator(method).prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        batch.execute(strict=True)
        with NativeRKSResponse.from_native(batch, basis, tile_points=257) as response:
            actual = rks_hvp(
                response,
                direction,
                cache=tmp_path / "hvp",
                tile_points=257,
            )

    assert tuple(actual.components) == SOURCE_ORDER
    assert actual.diagnostics["complete_source_coverage"]
    assert actual.diagnostics["molecular_hvp"]
    assert not actual.diagnostics["full_molecular_hessian_allocated"]
    assert not actual.diagnostics["public_calculator_capability"]
    np.testing.assert_allclose(
        actual.value,
        sum(actual.components.values(), np.zeros_like(actual.value)),
        atol=2e-12,
        rtol=0,
    )
    assert all(not value.flags.writeable for value in actual.components.values())

    coords = np.asarray([xyz for _, xyz in H2], dtype=np.float64)
    errors = []
    for step in (1.0e-3, 3.0e-4):
        gradients = []
        for sign in (1.0, -1.0):
            displaced = coords + sign * step * direction
            atoms = [
                (symbol, position)
                for (symbol, _), position in zip(H2, displaced, strict=True)
            ]
            gradients.append(_gradient(method, atoms, tmp_path / f"gradient-{method}"))
        numeric = (gradients[0] - gradients[1]) / (2.0 * step)
        errors.append(float(np.max(np.abs(numeric - actual.value))))
    assert errors[-1] < 2e-4, errors
    assert errors[-1] < max(0.35 * errors[0], 2e-6), errors


def test_pbe_rks_hvp_raw_bilinear_symmetry(tmp_path: typing.Any) -> None:
    directions = (
        np.array([[0.11, -0.07, 0.19], [-0.23, 0.05, 0.17]]),
        np.array([[-0.09, 0.21, 0.08], [0.14, -0.16, 0.12]]),
    )
    u, v = (value / np.linalg.norm(value) for value in directions)
    with _calculator("pbe-rks").prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        batch.execute(strict=True)
        with NativeRKSResponse.from_native(batch, basis, tile_points=257) as response:
            hu = rks_hvp(response, u, cache=tmp_path / "hvp", tile_points=257).value
            hv = rks_hvp(response, v, cache=tmp_path / "hvp", tile_points=257).value

    left = float(np.einsum("ax,ax->", u, hv))
    right = float(np.einsum("ax,ax->", v, hu))
    assert abs(left - right) < 2e-7
