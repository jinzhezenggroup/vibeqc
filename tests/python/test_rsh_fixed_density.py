"""#167A range-separated composition and fixed-density variational gates."""

import json
import typing
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
from vibeqc import assemble_fixed_density_exchange
from vibeqc.mean_field import compile_fixed_density_method
from vibeqc.profiles import file_hash
from vibeqc_compiler.integral.range_separation import (
    CoulombKernel,
    reference_moments,
)
from vibeqc_compiler.method import (
    MethodSpec,
    RangeSeparatedExchangePrimitive,
    SemilocalXCPrimitive,
    UnsupportedMethod,
    resolve_method,
)
from vibeqc_compiler.xc.program import build_program


def test_cam_manifests_resolve_explicit_sr_lr_primitives_without_new_science() -> None:
    cam = resolve_method("CAM-B3LYP")
    camh = resolve_method("CAMH-B3LYP")
    for method, long_range in ((cam, Fraction(65, 100)), (camh, Fraction(1, 2))):
        assert method.requirements["operators"] == (
            "semilocal-xc",
            "short-range-exchange",
            "long-range-exchange",
        )
        local, short, long = method.primitives
        assert isinstance(local, SemilocalXCPrimitive)
        assert local.functional.range_omega == Fraction(33, 100)
        assert isinstance(short, RangeSeparatedExchangePrimitive)
        assert isinstance(long, RangeSeparatedExchangePrimitive)
        assert (short.operator, short.coefficient, short.omega) == (
            "short-range",
            Fraction(19, 100),
            Fraction(33, 100),
        )
        assert (long.operator, long.coefficient, long.omega) == (
            "long-range",
            long_range,
            Fraction(33, 100),
        )
    assert {name for name, _ in cam.primitives[0].functional.components} == {
        "GGA_X_B88",
        "GGA_X_ITYH",
        "LDA_C_VWN",
        "GGA_C_LYP",
    }
    with pytest.raises(UnsupportedMethod, match="does not support primitive"):
        compile_fixed_density_method(cam)


def test_cam_b3lyp_scalar_energy_and_potential_coefficients_match_pinned_oracle() -> (
    None
):
    method = resolve_method("CAM-B3LYP", spin="polarized")
    spec = method.primitives[0].functional
    point = np.array([[0.3, 0.2, 0.015, 0.003, 0.01, 0.0, 0.0]]).T
    actual = build_program(spec, order=1).evaluate(point).ravel()
    # Pinned independently with PySCF 2.11.0 / Libxc CAMB3LYP eval_xc.
    expected = np.array(
        [
            -0.22534883092171914,
            -0.6376091098611569,
            -0.5721238021867927,
            -0.012862968002481867,
            0.001656668925640544,
            -0.01898573870842516,
            0.0,
            0.0,
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)

    restricted = resolve_method("CAM-B3LYP").primitives[0].functional
    u = np.array([[0.5, 0.031, 0.0]]).T
    actual_u = build_program(restricted, order=1).evaluate(u).ravel()
    expected_u = np.array(
        [
            -0.22356234835336405,
            -0.6071484191376727,
            -0.007332384853535235,
            0.0,
        ]
    )
    np.testing.assert_allclose(actual_u, expected_u, rtol=2e-12, atol=2e-13)


def test_b3lyp_vwn_rpa_scalar_energy_and_potential_match_pinned_oracle() -> None:
    method = resolve_method("B3LYP", spin="polarized")
    spec = method.primitives[0].functional
    point = np.array([[0.3, 0.2, 0.015, 0.003, 0.01, 0.0, 0.0]]).T
    actual = build_program(spec, order=1).evaluate(point).ravel()
    # Pinned independently with PySCF 2.14.0 / Libxc 7.0.0 B3LYP.
    expected = np.array(
        [
            -0.26232290280116083,
            -0.7122471845474007,
            -0.647404446702975,
            -0.014654598299102754,
            0.0016566689256405436,
            -0.022748740576278376,
            0.0,
            0.0,
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-13)

    restricted = resolve_method("B3LYP").primitives[0].functional
    u = np.array([[0.5, 0.031, 0.0]]).T
    actual_u = build_program(restricted, order=1).evaluate(u).ravel()
    expected_u = np.array(
        [
            -0.26053117903257633,
            -0.6822162975610385,
            -0.00859813384794804,
            0.0,
        ]
    )
    np.testing.assert_allclose(actual_u, expected_u, rtol=2e-12, atol=2e-13)

    # VWN5 is intentionally a different named scientific method.
    vwn5 = resolve_method("B3LYP5", spin="polarized").primitives[0].functional
    vwn5_value = build_program(vwn5, order=0).evaluate(point)[0, 0]
    assert abs(vwn5_value - actual[0]) > 1e-8


def test_rsh_libxc_source_manifest_is_pinned() -> None:
    root = Path(__file__).resolve().parents[2]
    source = root / "upstream/libxc/7.0.0"
    manifest_root = root / "manifests/libxc/7.0.0"
    manifest = json.loads((manifest_root / "rsh-manifest.json").read_text())
    for name, item in manifest["files"].items():
        assert file_hash(source / name) == item["sha256"]


def _cam_spec(identifier: typing.Any, omega: typing.Any) -> typing.Any:
    return MethodSpec(
        identifier,
        (
            ("GGA_X_B88", Fraction(35, 100)),
            ("GGA_X_ITYH", Fraction(46, 100)),
            ("LDA_C_VWN", Fraction(19, 100)),
            ("GGA_C_LYP", Fraction(81, 100)),
        ),
        short_range_exchange=Fraction(19, 100),
        long_range_exchange=Fraction(65, 100),
        range_omega=omega,
    )


def test_cam_range_parameter_changes_semilocal_and_operator_identity_and_rejects_stale_k() -> (
    None
):
    low = resolve_method(_cam_spec("CAM-test-low", Fraction(1, 5)))
    high = resolve_method(_cam_spec("CAM-test-high", Fraction(2, 5)))
    assert low.identity != high.identity

    point = np.array([[0.5, 0.031, 0.0]]).T
    low_energy = build_program(low.primitives[0].functional, order=0).evaluate(point)[
        0, 0
    ]
    high_energy = build_program(high.primitives[0].functional, order=0).evaluate(point)[
        0, 0
    ]
    assert low_energy != pytest.approx(high_energy, rel=0.0, abs=1e-12)

    density = np.array([[0.7]])
    stale = {
        ("short-range", Fraction(1, 5)): np.array([[0.2]]),
        ("long-range", Fraction(1, 5)): np.array([[0.3]]),
    }
    with pytest.raises(ValueError, match="operator set"):
        assemble_fixed_density_exchange(high, density, stale)


def test_cam_polarized_zero_spin_limit_recovers_unpolarized_scalar_energy() -> None:
    rho, sigma = 0.6, 0.024
    restricted = resolve_method("CAM-B3LYP").primitives[0].functional
    polarized = resolve_method("CAM-B3LYP", spin="polarized").primitives[0].functional
    e_r = build_program(restricted, order=0).evaluate(np.array([[rho, sigma, 0.0]]).T)[
        0, 0
    ]
    e_u = build_program(polarized, order=0).evaluate(
        np.array([[rho / 2, rho / 2, sigma / 4, sigma / 4, sigma / 4, 0.0, 0.0]]).T
    )[0, 0]
    np.testing.assert_allclose(e_u, e_r, rtol=2e-13, atol=2e-14)


def _range_moments(omega: typing.Any) -> typing.Any:
    argument, rho = 0.41, 0.73
    full = reference_moments(0, argument, rho, CoulombKernel())[0]
    short = reference_moments(0, argument, rho, CoulombKernel("short_range", omega))[0]
    long = reference_moments(0, argument, rho, CoulombKernel("long_range", omega))[0]
    np.testing.assert_allclose(short + long, full, rtol=4e-13, atol=1e-15)
    return short, long


def test_restricted_fixed_density_exchange_uses_166_sr_lr_operator_values() -> None:
    method = resolve_method("CAM-B3LYP")
    omega = Fraction(33, 100)
    short, long = _range_moments(float(omega))

    def evaluate(d: typing.Any) -> typing.Any:
        density = np.array([[d]])
        raw = {
            ("short-range", omega): density * short,
            ("long-range", omega): density * long,
        }
        return assemble_fixed_density_exchange(method, density, raw)

    d = 1.7
    result = evaluate(d)
    expected_v = -0.5 * d * (0.19 * short + 0.65 * long)
    np.testing.assert_allclose(result.potential, [[expected_v]], rtol=2e-15)
    np.testing.assert_allclose(result.energy, 0.5 * d * expected_v, rtol=2e-15)
    step = 1e-5
    derivative = (evaluate(d + step).energy - evaluate(d - step).energy) / (2 * step)
    np.testing.assert_allclose(derivative, expected_v, rtol=3e-11)


def test_unrestricted_fixed_density_exchange_has_same_variational_coefficient() -> None:
    method = resolve_method("CAM-B3LYP", spin="polarized")
    omega = Fraction(33, 100)
    short, long = _range_moments(float(omega))
    density = np.array([[[0.8]], [[0.3]]])
    raw = {
        ("short-range", omega): density * short,
        ("long-range", omega): density * long,
    }
    result = assemble_fixed_density_exchange(method, density, raw)
    expected = -density * (0.19 * short + 0.65 * long)
    np.testing.assert_allclose(result.potential, expected, rtol=2e-15)
    np.testing.assert_allclose(
        result.energy, 0.5 * np.sum(density * expected), rtol=2e-15
    )
