"""Protect merged PBE0/r2SCAN registration, snapshot and point-wire contracts."""

import ctypes as ct
from fractions import Fraction

import numpy as np
import pytest
from vibeqc import Calculator, GridSpec, KsOptions, _native
from vibeqc._dft_gradient import StationaryKsState
from vibeqc._ks_snapshot import _scf_xc_points
from vibeqc.ks import native_ks_options, resolve_ks_options
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.method import MethodSpec, resolve_method

from tools.vibeqc_response import (
    NativeRKSResponse,
    NativeUKSResponse,
    ResponseUnsupported,
)

H2 = [("H", (0.0, 0.0, -0.7)), ("H", (0.0, 0.0, 0.7))]
GRID = GridSpec(radial_points=24, angular_polar=8, angular_azimuth=16)


@pytest.mark.parametrize("method", ("r2scan-rks", "r2scan-uks"))
def test_r2scan_keeps_unscaled_composition_descriptor(method: str) -> None:
    options = resolve_ks_options(method)
    assert options.coefficients == (1.0, 1.0, 0.0)
    assert not options.requires_composition_v2
    native = native_ks_options(options)
    assert native.semilocal_exchange_scale == native.semilocal_correlation_scale == 1.0
    assert native.fock_exchange_coefficient == 0.0


@pytest.mark.parametrize("functional", (0, 1))
def test_v1_and_v2_point_layouts_remain_compatible(functional: int) -> None:
    library = _native.load_library(device="cpu")
    rho = np.array([[0.8, 0.4], [0.3, 0.2]])
    gradient = np.arange(12, dtype=np.float64).reshape(2, 2, 3) * 0.002
    pointer = ct.POINTER(ct.c_double)
    old = library.vibeqc_xc_point_batch_v1
    old.argtypes = [ct.c_uint32, pointer, pointer, ct.c_size_t, pointer, ct.c_size_t]
    old.restype = ct.c_int
    output = np.empty((2, 9))
    _native.check(
        library,
        old(
            functional,
            rho.ctypes.data_as(pointer),
            gradient.ctypes.data_as(pointer),
            2,
            output.ctypes.data_as(pointer),
            output.size,
        ),
    )
    current = _scf_xc_points(library, functional, rho, gradient)
    packed = np.column_stack(
        [
            current["energy"],
            current["rho"].T,
            current["gradient"].transpose(1, 0, 2).reshape(-1, 6),
        ]
    )
    np.testing.assert_array_equal(output, packed)
    np.testing.assert_array_equal(current["kinetic"], np.zeros((2, 2)))


def test_scaled_pbe_v3_preserves_component_linearity() -> None:
    library = _native.load_library(device="cpu")
    rho = np.array([[0.8, 0.4], [0.3, 0.2]])
    gradient = np.arange(12, dtype=np.float64).reshape(2, 2, 3) * 0.002
    exchange = _scf_xc_points(library, 1, rho, gradient, scales=(1.0, 0.0))
    correlation = _scf_xc_points(library, 1, rho, gradient, scales=(0.0, 1.0))
    hybrid = _scf_xc_points(library, 1, rho, gradient, scales=(0.75, 1.0))
    for key in ("energy", "rho", "gradient", "kinetic"):
        np.testing.assert_allclose(
            hybrid[key], 0.75 * exchange[key] + correlation[key], atol=2e-15, rtol=2e-13
        )


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks", "r2scan-rks", "r2scan-uks"))
def test_both_method_families_keep_native_snapshot_identity(method: str) -> None:
    calculator = Calculator(
        method=method, device="cpu", ks_options=KsOptions(grid=GRID), max_iterations=200
    )
    with calculator.prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        assert batch.execute(strict=True).items[0].converged
        state = StationaryKsState.from_native(batch, basis)
        try:
            assert state.identity.method == method
            assert state._source.metadata[2] == (1 if method.endswith("-rks") else 2)
            assert state._source.coefficients == calculator.ks_options.coefficients
            assert state._source.metadata[6] == (
                2 if method.startswith("r2scan") else 1
            )
        finally:
            state._source.close()


@pytest.mark.parametrize("method", ("pbe0-rks", "pbe0-uks", "r2scan-rks", "r2scan-uks"))
def test_cpu_cpks_does_not_mislabel_new_families_as_pbe(method: str) -> None:
    calculator = Calculator(
        method=method, device="cpu", ks_options=KsOptions(grid=GRID), max_iterations=200
    )
    with calculator.prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        batch.execute(strict=True)
        adapter = NativeRKSResponse if method.endswith("-rks") else NativeUKSResponse
        with pytest.raises(ResponseUnsupported, match="LDA/PBE"):
            adapter.from_native(batch, basis)


@pytest.mark.parametrize("restricted", (True, False))
def test_custom_hybrid_does_not_reuse_pure_ks_response(restricted: bool) -> None:
    """A PBE family selector does not authorize ignoring its nonzero K source."""
    method = "pbe-rks" if restricted else "pbe-uks"
    spin = "unpolarized" if restricted else "polarized"
    graph = resolve_method(
        MethodSpec(
            "PBE50-response-boundary",
            (("GGA_X_PBE", Fraction(1, 2)), ("GGA_C_PBE", Fraction(1))),
            exact_exchange=Fraction(1, 2),
        ),
        spin=spin,
    )
    adapter = NativeRKSResponse if restricted else NativeUKSResponse
    calculator = Calculator(
        method=method, ks_options=KsOptions(composition=graph, grid=GRID)
    )
    with calculator.prepare_batch([H2]) as batch, NativeAO(H2) as basis:
        batch.execute(strict=True)
        with pytest.raises(ResponseUnsupported, match="unscaled LDA/PBE"):
            adapter.from_native(batch, basis)
