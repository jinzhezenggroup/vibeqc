"""Weight-folding checks for issue #180 (the two-electron term of the graph).

``docs/developer/hessian.md`` states the frozen-density two-electron skeleton as
``Σ W_μνλσ ∂²(μν|λσ)`` with ``W = ½ P_μν P_λσ − ¼ P_μλ P_νσ``. Whether that
needs another factor of the energy's own ``½`` is exactly the kind of question a
prose dependency graph gets wrong and a consumer then propagates, so the factor
is checked here against the energy it is meant to reproduce rather than asserted
in the document alone.
"""

import typing

import numpy as np
import pytest

from tools.vibeqc_hessian import (
    two_electron_energy,
    two_electron_weight,
    weight_energy,
)


def _symmetric_density(n: typing.Any = 4, seed: typing.Any = 180) -> typing.Any:
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n))
    return raw + raw.T


def _symmetric_eri(n: typing.Any = 4, seed: typing.Any = 181) -> typing.Any:
    """A tensor with the permutational symmetry of chemist-notation ``(μν|λσ)``.

    ``(μν|λσ) = (νμ|λσ) = (μν|σλ) = (λσ|μν)``. The folding under test is an
    index relabelling, so it is the symmetry of the integral that makes the
    comparison meaningful.
    """
    rng = np.random.default_rng(seed)
    raw = rng.normal(size=(n, n, n, n))
    eri = raw + raw.transpose(1, 0, 2, 3)
    eri = eri + eri.transpose(0, 1, 3, 2)
    eri = eri + eri.transpose(2, 3, 0, 1)
    return eri / 4.0


def test_folded_weight_reproduces_the_two_electron_energy() -> None:
    """``Σ W (μν|λσ)`` must equal ``½ Tr[P G(P)]``, with no further factor.

    This is the check that fails when the outer ``½`` is applied twice: the
    folded contraction then comes out at exactly half the energy.
    """

    density = _symmetric_density()
    eri = _symmetric_eri()
    folded = weight_energy(two_electron_weight(density), eri)
    direct = two_electron_energy(density, eri)

    assert folded == pytest.approx(direct, rel=1.0e-12, abs=1.0e-12)
    # Guard the guard: an energy of zero would make the comparison vacuous.
    assert abs(direct) > 1.0e-3


def test_applying_the_outer_half_twice_is_detectably_wrong() -> None:
    """``½ Σ W (μν|λσ)`` is pinned as the losing alternative.

    Keeping it explicit means an edit that re-adds the outer ``½`` fails here
    rather than producing a plausibly small numerical disagreement downstream.
    """

    density = _symmetric_density()
    eri = _symmetric_eri()
    halved = 0.5 * weight_energy(two_electron_weight(density), eri)
    direct = two_electron_energy(density, eri)

    assert halved != pytest.approx(direct, rel=1.0e-6)


def test_folded_weight_reproduces_the_energy_for_real_integrals() -> None:
    """The same identity must hold for integrals a real code produced.

    The synthetic tensor pins the algebra; this pins the notation, because a
    chemist-versus-physicist index-order mismatch would survive the synthetic
    check if the synthetic tensor were built in the same wrong order.
    """

    gto = pytest.importorskip("pyscf.gto")
    mol = gto.M(
        atom=[["H0", [0.0, 0.0, -0.7]], ["H1", [0.0, 0.0, 0.7]]],
        basis="sto-3g",
        unit="Bohr",
        verbose=0,
    )
    eri = mol.intor("int2e")
    density = _symmetric_density(n=mol.nao, seed=182)

    folded = weight_energy(two_electron_weight(density), eri)
    direct = two_electron_energy(density, eri)
    assert folded == pytest.approx(direct, rel=1.0e-12, abs=1.0e-12)


def test_two_electron_weight_rejects_a_malformed_density() -> None:
    with pytest.raises(ValueError, match="square matrix"):
        two_electron_weight(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="finite"):
        two_electron_weight(np.array([[np.nan, 0.0], [0.0, 1.0]]))


def test_weight_energy_rejects_a_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        weight_energy(np.zeros((2, 2, 2, 2)), np.zeros((3, 3, 3, 3)))
