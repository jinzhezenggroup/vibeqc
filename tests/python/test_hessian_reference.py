"""Independent semi-numerical Hessian checks with explicit oracle boundaries.

Fixtures cover H2/STO-3G and a custom 12-AO water stress basis (5 occupied,
7 virtual); additional comparisons use genuine 7-AO water/STO-3G.
"""

import typing

import numpy as np
import pytest

pytest.importorskip("pyscf")

from tools.vibeqc_hessian.reference import (
    System,
    build_mol,
    fd_hessian,
    hessian_components,
    hessian_total,
)

# --- fixtures (STO-3G, Bohr) -----------------------------------------------
_S = [
    [
        0,
        (3.425250914, 0.1543289673),
        (0.6239137298, 0.5353281423),
        (0.168855404, 0.4446345422),
    ]
]
# Custom O s/p/d stress basis: 1 + 3 + 6 Cartesian AOs, not STO-3G.
_O = [
    [
        0,
        (18.5950316763, 0.0499684015),
        (5.6203025291, 0.1505027689),
        (1.3720603452, 0.4117903918),
        (0.3434943771, 0.4106611356),
    ],
    [
        1,
        (7.1153375795, 0.0102428309),
        (2.0218099978, 0.0314286582),
        (0.5271803969, 0.0814495192),
        (0.1703101032, 0.1646776769),
        (0.0628267133, 0.2917579986),
    ],
    [2, (0.0628267133, 0.1506938663), (0.0351485943, 0.2198166377)],
]


def _h2_mol() -> typing.Any:
    return build_mol(
        [(1, [0.0, 0.0, 0.0]), (1, [0.1, 0.2, 1.4])],
        {"H0": list(_S), "H1": list(_S)},
        charge=0,
        spin=0,
    )


def _water_mol() -> typing.Any:
    return build_mol(
        [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.958, 0.587]), (1, [0.0, -0.958, 0.587])],
        {"O0": list(_O), "H1": list(_S), "H2": list(_S)},
        charge=0,
        spin=0,
    )


# --- tests ----------------------------------------------------------------


@pytest.mark.parametrize(
    "mol_fn, label, fd_tol",
    [
        (_h2_mol, "h2", 5e-6),
        (_water_mol, "water", 5e-4),
    ],
)
def test_reference_matches_energy_fd(
    mol_fn: typing.Any, label: typing.Any, fd_tol: typing.Any
) -> None:
    mol = mol_fn()
    s = System(mol)
    s.derive()
    H = hessian_total(s)
    H_fd = fd_hessian(mol, h=1e-4)
    diff = np.abs(H - H_fd).max()
    print(f"[{label}] max|reference - FD| = {diff:.3e}")
    assert diff < fd_tol, f"{label}: reference differs from FD by {diff:.3e}"


@pytest.mark.parametrize(
    "mol_fn, label, sym_tol",
    [
        (_h2_mol, "h2", 1e-8),
        (_water_mol, "water", 1e-8),
    ],
)
def test_symmetry(mol_fn: typing.Any, label: typing.Any, sym_tol: typing.Any) -> None:
    mol = mol_fn()
    s = System(mol)
    s.derive()
    H = hessian_total(s)
    diff = np.abs(H - H.transpose(1, 0, 3, 2)).max()
    print(f"[{label}] sym viol = {diff:.3e}")
    assert diff < sym_tol, f"{label}: asymmetry {diff:.3e}"


@pytest.mark.parametrize(
    "mol_fn, label, tr_tol",
    [
        (_h2_mol, "h2", 1e-6),
        (_water_mol, "water", 1e-4),
    ],
)
def test_translation_invariance(
    mol_fn: typing.Any, label: typing.Any, tr_tol: typing.Any
) -> None:
    mol = mol_fn()
    s = System(mol)
    s.derive()
    H = hessian_total(s).transpose(0, 2, 1, 3).reshape(3 * mol.natm, 3 * mol.natm)
    row = np.abs(H.sum(axis=1)).max()
    col = np.abs(H.sum(axis=0)).max()
    print(f"[{label}] translation: row={row:.3e} col={col:.3e}")
    assert row < tr_tol and col < tr_tol, f"{label}: translation row={row:.3e}"


@pytest.mark.parametrize(
    "mol_fn, label",
    [
        (_h2_mol, "h2"),
        (_water_mol, "water"),
    ],
)
def test_component_sum_equals_total(mol_fn: typing.Any, label: typing.Any) -> None:
    mol = mol_fn()
    s = System(mol)
    s.derive()
    comps = hessian_components(s)
    total_from_parts = (
        comps["nuclear"]
        + comps["core"]
        + comps["pulay"]
        + comps["two_electron"]
        + comps["relaxation"]
    )
    diff = np.abs(total_from_parts - comps["total"]).max()
    print(f"[{label}] component sum vs total: {diff:.3e}")
    assert diff < 1e-10, f"{label}: component sum off by {diff:.3e}"


@pytest.mark.parametrize(
    "mol_fn, label, key",
    [
        (_h2_mol, "h2", "relaxation"),
        (_h2_mol, "h2", "pulay"),
        (_h2_mol, "h2", "two_electron"),
        (_h2_mol, "h2", "nuclear"),
        (_water_mol, "water", "relaxation"),
        (_water_mol, "water", "pulay"),
        (_water_mol, "water", "two_electron"),
        (_water_mol, "water", "nuclear"),
    ],
)
def test_component_nonzero(
    mol_fn: typing.Any, label: typing.Any, key: typing.Any
) -> None:
    """Each component must be non-trivial (negative case: removing it breaks)."""
    mol = mol_fn()
    s = System(mol)
    s.derive()
    comps = hessian_components(s)
    mag = np.abs(comps[key]).max()
    print(f"[{label}] |{key}| max = {mag:.5f}")
    assert mag > 1e-4, f"{label}: {key} is trivially small ({mag:.3e})"


@pytest.mark.parametrize(
    "mol_fn, label, tol",
    [
        (_h2_mol, "h2", 5e-5),
        (_water_mol, "water", 5e-3),
    ],
)
def test_frozen_density_wrong(
    mol_fn: typing.Any, label: typing.Any, tol: typing.Any
) -> None:
    """Negative case: removing orbital relaxation must shift the Hessian
    by more than the reference-vs-FD error."""
    mol = mol_fn()
    s = System(mol)
    s.derive()
    H_full = hessian_total(s)
    H_frozen = hessian_total(s, with_relax=False)
    H_fd = fd_hessian(mol, h=1e-4)
    shift = np.abs(H_full - H_frozen).max()
    fd_err = np.abs(H_full - H_fd).max()
    print(f"[{label}] frozen-dens shift={shift:.4e}  fd_err={fd_err:.4e}")
    assert shift > 3 * fd_err, (
        f"{label}: frozen density barely changes the Hessian "
        f"(shift={shift:.3e} vs fd_err={fd_err:.3e})"
    )


def _sto3g_water() -> typing.Any:
    from pyscf import gto

    # String atom input also checks fresh-geometry reconstruction without
    # depending on the caller's original Python input representation.
    return gto.M(
        atom="O 0.1 0.2 0.1; H 0.3 1.43 .99; H .1 -1.4 1.07",
        basis="sto-3g",
        unit="Bohr",
        cart=True,
        verbose=0,
    )


@pytest.mark.parametrize("mol_fn", [_h2_mol, _water_mol, _sto3g_water])
def test_reference_matches_analytic_hessian_and_gradient_differences(
    mol_fn: typing.Any,
) -> None:
    from tools.vibeqc_hessian.reference import _converged_rhf

    mol = mol_fn()
    state = System(mol)
    state.derive()
    actual = hessian_total(state)
    analytic = _converged_rhf(mol).Hessian().kernel()
    np.testing.assert_allclose(actual, analytic, atol=5e-6, rtol=0)
    # Differentiate analytic gradients, with all steps gated independently.
    # No symmetrization is applied to either side of the comparison.
    for step in (2e-3, 7e-4, 2e-4):
        numerical = np.empty_like(actual)
        coords = mol.atom_coords()
        for atom in range(mol.natm):
            for axis in range(3):
                delta = np.zeros_like(coords)
                delta[atom, axis] = step
                plus = mol.copy().set_geom_(coords + delta, unit="Bohr")
                minus = mol.copy().set_geom_(coords - delta, unit="Bohr")
                numerical[:, atom, :, axis] = (
                    _converged_rhf(plus).nuc_grad_method().kernel()
                    - _converged_rhf(minus).nuc_grad_method().kernel()
                ) / (2 * step)
        np.testing.assert_allclose(actual, numerical, atol=2e-4, rtol=0)


@pytest.mark.parametrize("step", [0.0, -1e-4, np.inf, np.nan])
def test_invalid_steps_rejected(step: typing.Any) -> None:
    mol = _h2_mol()
    with pytest.raises(ValueError, match="finite and positive"):
        System(mol, h1=step)
    with pytest.raises(ValueError, match="finite and positive"):
        System(mol, h2=step)
    with pytest.raises(ValueError, match="finite and positive"):
        fd_hessian(mol, h=step)


def test_unconverged_reference_is_rejected(monkeypatch: typing.Any) -> None:
    from tools.vibeqc_hessian import reference

    class FailedRHF:
        converged = False
        e_tot = -1.0

        def kernel(self) -> typing.Any:
            return self.e_tot

    monkeypatch.setattr(reference.scf, "RHF", lambda mol: FailedRHF())
    with pytest.raises(RuntimeError, match="did not converge"):
        System(_h2_mol())
    with pytest.raises(RuntimeError, match="did not converge"):
        fd_hessian(_h2_mol())


def test_open_shell_and_spherical_reference_are_rejected() -> None:
    from pyscf import gto

    hydrogen = gto.M(atom="H 0 0 0", basis="sto-3g", spin=1, cart=True, verbose=0)
    with pytest.raises(ValueError, match="closed-shell"):
        System(hydrogen)
    spherical = _h2_mol()
    spherical.cart = False
    with pytest.raises(ValueError, match="Cartesian"):
        System(spherical)
