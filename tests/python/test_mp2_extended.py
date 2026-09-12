"""Independent >12 AO public molecular gate; explicit opt-in reference tier."""

import os

import numpy as np
import pytest
from vibeqc import Calculator, Primitive, Shell

pytestmark = pytest.mark.skipif(
    os.environ.get("VIBEQC_MP2_EXTENDED_TEST") != "1",
    reason="opt-in independent molecular gate with pinned PySCF 2.14.0",
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fourteen_ao_all_electron_public_reference_and_tail(device):
    if device == "cuda" and os.environ.get("VIBEQC_MP2_CUDA_TEST") != "1":
        pytest.skip("requires allocated CUDA device")
    pyscf = pytest.importorskip("pyscf")
    assert pyscf.__version__ == "2.14.0"
    from pyscf import gto, mp, scf

    # Seven uncontracted s shells per He: 14 AOs and 12 virtuals, so the native
    # eight-column tile has a four-column final block. This exceeds CG10's
    # old 12-AO convenience exporter without changing its cap or calling it.
    exponents = (0.08, 0.2, 0.5, 1.2, 3.0, 7.0, 16.0)
    atoms = [("He", (0.0, 0.0, 0.0)), ("He", (0.2, 0.1, 3.1))]
    shells = tuple(
        Shell(atom, 0, (Primitive(e, 1.0),)) for atom in range(2) for e in exponents
    )
    mol = gto.M(
        atom=atoms,
        basis={"He": [[0, [e, 1.0]] for e in exponents]},
        unit="Bohr",
        cart=True,
        verbose=0,
    )
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-13
    hf.conv_tol_grad = 1e-11
    hf.max_cycle = 200
    hf.kernel()
    assert hf.converged and mol.nao_nr() == 14
    independent = mp.MP2(hf)
    independent.kernel()
    result = Calculator(
        method="mp2",
        basis=shells,
        device=device,
        max_iterations=200,
        energy_tolerance=1e-12,
        density_tolerance=1e-12,
    ).singlepoint(atoms)
    assert result.forces is None
    assert abs(result.energy - independent.e_tot) <= 1e-9
    np.testing.assert_allclose(
        [result.correlation.opposite_spin_energy, result.correlation.same_spin_energy],
        [independent.e_corr_os, independent.e_corr_ss],
        atol=1e-11,
        rtol=1e-10,
    )
    assert result.correlation.energy_tile_count == 16
    from pyscf import ao2mo
    from test_mp2_fixed_native import check_native

    from tools.vibeqc_posthf.sources import NativeSource

    # The exact same PySCF orbitals/integrals feed the newly added native
    # provider/energy adapters, independently of the new VibeQC SCF result.
    d = hf.make_rdm1()
    arrays = {
        "S": hf.get_ovlp(),
        "h": hf.get_hcore(),
        "F": hf.get_fock(dm=d),
        "C": hf.mo_coeff,
        "D": d,
        "eps": hf.mo_energy,
    }
    mo = ao2mo.restore(1, ao2mo.kernel(mol, hf.mo_coeff), 14)
    with NativeSource(atoms=atoms, basis=shells) as source:
        check_native(
            source,
            arrays,
            hf.e_tot,
            mo,
            independent.e_corr_os,
            independent.e_corr_ss,
            int(device == "cuda"),
            tiles=(8,),
        )
