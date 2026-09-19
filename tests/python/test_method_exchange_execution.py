"""Executable MethodIR exact-exchange boundary and PBE0 fixed-density tests."""

import os
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from vibeqc.fock import FockBuildSpec, FockPlan, FockTerm
from vibeqc.mean_field import FixedDensityMeanField, compile_fixed_density_method
from vibeqc_compiler.dft import NativeAO
from vibeqc_compiler.dft.fixtures import basis_arguments
from vibeqc_compiler.method import MethodSpec, resolve_method
from vibeqc_compiler.xc import FixedDensityXC
from vibeqc_compiler.xc.integration_fixtures import load_integration_fixture
from vibeqc_compiler.xc.spec import FunctionalSpec, functional

DEVICE = os.environ.get("VIBEQC_TEST_FOCK_DEVICE", "cpu")


@pytest.mark.parametrize(
    "method,spin,expected",
    (
        ("PBE0", "unpolarized", -0.125),
        ("PBE0", "polarized", -0.25),
    ),
)
def test_exact_exchange_primitive_lowers_to_common_jk_provider(method, spin, expected):
    graph = resolve_method(method, spin=spin)
    plan = compile_fixed_density_method(graph)

    assert plan.fock_spec.spin == graph.reference
    assert plan.fock_spec.coulomb.present
    assert plan.fock_spec.coulomb.coefficient == 1.0
    assert plan.fock_spec.exchange.present
    assert plan.fock_spec.exchange.operator == "full_range"
    assert plan.fock_spec.exchange.coefficient == expected
    assert plan.functional == graph.primitives[0].functional
    assert plan.capabilities == ("energy", "fock")
    assert plan.to_payload()["method_identity"] == graph.identity


def test_exact_exchange_execution_is_composition_driven_not_pbe0_named():
    custom = MethodSpec(
        "custom-global-hybrid",
        (("GGA_X_PBE", Fraction(2, 3)), ("GGA_C_PBE", Fraction(1))),
        exact_exchange=Fraction(1, 3),
    )
    restricted = compile_fixed_density_method(resolve_method(custom))
    unrestricted = compile_fixed_density_method(
        resolve_method(custom, spin="polarized"),
        coulomb_approximation="density_fitted",
        exchange_approximation="density_fitted",
    )
    assert restricted.fock_spec.exchange.coefficient == pytest.approx(-1 / 6)
    assert unrestricted.fock_spec.exchange.coefficient == pytest.approx(-1 / 3)
    assert unrestricted.fock_spec.coulomb.approximation == "density_fitted"
    assert unrestricted.fock_spec.exchange.approximation == "density_fitted"


def test_pure_pbe_compiles_to_same_boundary_with_absent_exchange():
    plan = compile_fixed_density_method(resolve_method("PBE"))
    assert plan.fock_spec.coulomb.present
    assert not plan.fock_spec.exchange.present
    assert plan.fock_spec.exchange.coefficient == 0.0


def test_method_binding_compares_resolved_absent_exchange_semantics():
    meta, data, grid = load_integration_fixture("h2")
    graph = resolve_method("PBE")
    requested = FockBuildSpec(
        spin="restricted",
        derivative_order=0,
        coulomb=FockTerm(),
        exchange=FockTerm(
            False,
            7.0,
            operator="long_range",
            omega=0.8,
            approximation="density_fitted",
        ),
    )
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        FockPlan(basis, requested, device=DEVICE) as provider,
    ):
        assert provider.diagnostics["resolved"]["exchange"] == {
            "present": False,
            "coefficient": 0.0,
            "operator": "full_range",
            "omega": 0.0,
            "approximation": "exact",
        }
        consumer = FixedDensityMeanField.from_method(provider, graph)
        result = consumer.integrate(grid, data["density_total"], tile_points=7)
        assert result.method_identity == graph.identity
        assert not consumer.method_plan.fock_spec.exchange.present


@pytest.mark.parametrize(
    "method_spin,density_key",
    (("unpolarized", "density_total"), ("polarized", "density_spin")),
)
def test_pbe0_energy_and_fock_use_same_exact_exchange_weight(method_spin, density_key):
    meta, data, grid = load_integration_fixture("h2")
    density = data[density_key]
    graph = resolve_method("PBE0", spin=method_spin)
    executable = compile_fixed_density_method(graph)
    fock_spin = graph.reference
    ck = executable.fock_spec.exchange.coefficient

    pbe = FixedDensityXC(functional("PBE", spin=method_spin))
    pbe_x = FixedDensityXC(
        FunctionalSpec(
            "independent-pbe-x",
            (("GGA_X_PBE", Fraction(1)),),
            spin=method_spin,
        )
    )

    coulomb_spec = FockBuildSpec(
        spin=fock_spin,
        derivative_order=0,
        coulomb=FockTerm(),
        exchange=FockTerm(False, 0.0),
    )
    raw_k_spec = FockBuildSpec(
        spin=fock_spin,
        derivative_order=0,
        coulomb=FockTerm(False, 0.0),
        exchange=FockTerm(coefficient=0.0),
    )

    with (
        NativeAO(**basis_arguments(meta)) as basis,
        FockPlan(basis, executable.fock_spec, device=DEVICE) as provider,
        FockPlan(basis, coulomb_spec, device=DEVICE) as coulomb,
        FockPlan(basis, raw_k_spec, device=DEVICE) as raw_k,
    ):
        consumer = FixedDensityMeanField.from_method(provider, graph)
        result = consumer.integrate(grid, density, tile_points=7)
        j_reference = coulomb.evaluate(density)
        k_reference = raw_k.evaluate(density)
        pbe_reference = pbe.integrate(basis, grid, density, tile_points=7)
        x_reference = pbe_x.integrate(basis, grid, density, tile_points=7)

        local_energy = pbe_reference.energy - 0.25 * x_reference.energy
        local_fock = pbe_reference.potential - 0.25 * x_reference.potential
        exchange_energy = 0.5 * ck * np.sum(density * k_reference.exchange)
        expected_energy = j_reference.energy + exchange_energy + local_energy
        expected_fock = j_reference.fock + ck * k_reference.exchange + local_fock

        np.testing.assert_allclose(result.energy, expected_energy, atol=3e-11)
        np.testing.assert_allclose(result.fock, expected_fock, atol=3e-11)
        assert result.method_identity == graph.identity
        assert result.method_plan_identity == consumer.method_plan.identity
        assert provider.diagnostics["resolved"]["exchange"]["coefficient"] == ck

        direction = 0.01 * density
        step = 1e-5
        plus = consumer.integrate(grid, density + step * direction, tile_points=7)
        minus = consumer.integrate(grid, density - step * direction, tile_points=7)
        np.testing.assert_allclose(
            (plus.energy - minus.energy) / (2 * step),
            np.sum(result.fock * direction),
            atol=3e-8,
        )


def test_method_binding_rejects_wrong_exchange_factor_before_execution():
    meta, _, _ = load_integration_fixture("h2")
    graph = resolve_method("PBE0")
    executable = compile_fixed_density_method(graph)
    wrong = replace(
        executable.fock_spec,
        exchange=replace(executable.fock_spec.exchange, coefficient=-0.25),
    )
    with (
        NativeAO(**basis_arguments(meta)) as basis,
        FockPlan(basis, wrong, device=DEVICE) as provider,
        pytest.raises(ValueError, match="executable MethodIR plan"),
    ):
        FixedDensityMeanField.from_method(provider, graph)


@pytest.mark.parametrize("field", ["functional", "exchange", "method", "spin"])
def test_executable_plan_rejects_inconsistent_declared_physics(field):
    plan = compile_fixed_density_method(resolve_method("PBE0"))
    changes = {
        "functional": {"functional": functional("PBE")},
        "exchange": {
            "fock_spec": replace(
                plan.fock_spec,
                exchange=replace(plan.fock_spec.exchange, coefficient=-0.25),
            )
        },
        "method": {"method": resolve_method("PBE")},
        "spin": {"fock_spec": replace(plan.fock_spec, spin="unrestricted")},
    }
    with pytest.raises(ValueError, match="executable MethodIR plan"):
        replace(plan, **changes[field])
