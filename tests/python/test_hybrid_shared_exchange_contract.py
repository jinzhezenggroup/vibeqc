"""Fixed-density hybrid exchange contract across MethodIR, J/K and force plans.

These algebra checks do not admit a CUDA KS or molecular-force endpoint.
"""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest
from vibeqc import assemble_fixed_density_exchange
from vibeqc.ks import KsOptions, native_ks_options, resolve_ks_options
from vibeqc.mean_field import compile_fixed_density_method
from vibeqc_compiler.dft.grid import GridSpec
from vibeqc_compiler.method import (
    MethodIR,
    MethodSpec,
    compile_ks_execution_plan,
    resolve_method,
)
from vibeqc_compiler.method.stationary_gradient import (
    SCF_POINT_MODEL,
    StationaryGradientPlan,
    StationaryMeanField,
)
from vibeqc_compiler.tensor import execute


def _assert_full_range_exchange_contract(
    method: MethodIR, fraction: Fraction, selector: str
) -> None:
    ks = compile_ks_execution_plan(method)
    provider = compile_fixed_density_method(method)
    gradient = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
    spin = method.spin
    expected_fock = -fraction / (2 if spin == "unpolarized" else 1)

    assert method.full_range_exact_exchange == fraction
    assert len(ks.exchange) == 1
    assert ks.exchange[0].coefficient == fraction
    assert ks.exchange[0].fock_coefficient == expected_fock
    assert provider.fock_spec.exchange.coefficient == float(expected_fock)
    assert gradient.exchange.coefficient == fraction
    assert gradient.source_names.count("exact_exchange") == 1
    assert provider.method.identity == ks.method.identity == gradient.method.identity

    options = resolve_ks_options(
        selector, KsOptions(composition=method, grid=GridSpec())
    )
    native = native_ks_options(options)
    assert native.exchange_term_count == 1
    assert native.exchange_terms[0].coefficient == float(fraction)
    assert native.exchange_terms[0].fock_coefficient == float(expected_fock)

    # Form raw K from an independent four-index ERI contraction, rather than
    # borrowing the provider's already weighted exchange matrix.
    rng = np.random.default_rng(1187)
    pairs = rng.normal(size=(3, 2, 2))
    pairs = 0.5 * (pairs + pairs.transpose(0, 2, 1))
    eri = np.einsum("rij,rkl->ijkl", pairs, pairs)
    spins = 1 if spin == "unpolarized" else 2
    density = rng.normal(size=(spins, 2, 2))
    density = 0.15 * (density + density.transpose(0, 2, 1))
    raw_k = np.einsum("skl,ikjl->sij", density, eri)
    actual = assemble_fixed_density_exchange(
        method,
        density[0] if spins == 1 else density,
        {("full-range", Fraction(0)): raw_k[0] if spins == 1 else raw_k},
    )
    expected_v = float(expected_fock) * raw_k
    energy_factor = -float(fraction) / (4 if spins == 1 else 2)

    def independent_energy(d: np.ndarray) -> float:
        return energy_factor * float(np.einsum("sij,skl,ikjl->", d, d, eri))

    expected_e = independent_energy(density)
    np.testing.assert_allclose(
        actual.potential, expected_v[0] if spins == 1 else expected_v
    )
    np.testing.assert_allclose(actual.energy, expected_e)
    assert actual.method_identity == method.identity
    direction = rng.normal(size=density.shape)
    direction = 0.5 * (direction + direction.transpose(0, 2, 1))
    step = 1e-5
    directional_fd = (
        independent_energy(density + step * direction)
        - independent_energy(density - step * direction)
    ) / (2 * step)
    np.testing.assert_allclose(
        directional_fd, np.einsum("sij,sij->", expected_v, direction), atol=1e-11
    )

    # The stationary ERI derivative must use the same energy coefficient once,
    # with no cross-spin density products.
    quartets = tuple(product(range(2), repeat=4))
    left = np.array([[d[i, j] for i, k, j, l in quartets] for d in density])
    right = np.array([[d[k, l] for i, k, j, l in quartets] for d in density])
    block = gradient.integral_block("exact_exchange", terms=len(quartets))
    weight = execute(
        block.weights, {"density_left": left, "density_right": right}
    ).outputs["weights"]
    independent = energy_factor * np.sum(left * right, axis=0)
    np.testing.assert_allclose(weight, independent)


@pytest.mark.parametrize(
    "name,fraction", (("PBE0", Fraction(1, 4)), ("B3LYP", Fraction(1, 5)))
)
@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_public_hybrid_exchange_uses_one_coefficient_across_consumers(
    name: str, fraction: Fraction, spin: str
) -> None:
    method = resolve_method(name, spin=spin)
    selector = f"{name.lower()}-{'rks' if spin == 'unpolarized' else 'uks'}"
    _assert_full_range_exchange_contract(method, fraction, selector)


@pytest.mark.parametrize("spin", ("unpolarized", "polarized"))
def test_zero_and_changed_exchange_keep_provider_and_gradient_in_sync(
    spin: str,
) -> None:
    for fraction in (Fraction(0), Fraction(1, 2)):
        method = resolve_method(
            MethodSpec(
                f"pbe-exchange-{fraction}",
                (("GGA_X_PBE", 1 - fraction), ("GGA_C_PBE", Fraction(1))),
                exact_exchange=fraction,
            ),
            spin=spin,
        )
        ks = compile_ks_execution_plan(method)
        provider = compile_fixed_density_method(method)
        gradient = StationaryGradientPlan(method, StationaryMeanField(SCF_POINT_MODEL))
        assert len(ks.exchange) == (1 if fraction else 0)
        assert provider.fock_spec.exchange.present == bool(fraction)
        assert ("exact_exchange" in gradient.source_names) == bool(fraction)
        assert provider.fock_spec.exchange.coefficient == float(
            -fraction / (2 if spin == "unpolarized" else 1)
        )
        if fraction:
            selector = "pbe0-rks" if spin == "unpolarized" else "pbe0-uks"
            _assert_full_range_exchange_contract(method, fraction, selector)

    hybrid = resolve_method("PBE0", spin=spin)
    density = np.eye(2) if spin == "unpolarized" else np.stack((np.eye(2), np.eye(2)))
    raw = np.ones_like(density)
    with pytest.raises(ValueError, match="operator set"):
        assemble_fixed_density_exchange(hybrid, density, {})
    with pytest.raises(ValueError, match="operator set"):
        assemble_fixed_density_exchange(
            hybrid,
            density,
            {("full-range", Fraction(0)): raw, ("extra", Fraction(0)): raw},
        )
    gradient = StationaryGradientPlan(hybrid, StationaryMeanField(SCF_POINT_MODEL))
    with pytest.raises(ValueError, match="duplicate gradient source"):
        gradient.reduction_program(
            atoms=1, sources=(*gradient.source_names, "exact_exchange")
        )
    with pytest.raises(ValueError, match="incomplete"):
        gradient.reduction_program(
            atoms=1,
            sources=tuple(s for s in gradient.source_names if s != "exact_exchange"),
        )
