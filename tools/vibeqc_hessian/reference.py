"""Independent semi-numerical RHF Hessian reference for tiny Cartesian systems.

Integral coordinate derivatives come from fresh-molecule finite differences;
a dense CPHF solve supplies orbital relaxation. This optional PySCF oracle is
separate from the generated #178 providers and shared #179 response solver.
It neither implements nor qualifies the production analytic Hessian path.

Hessians use PySCF's (atom, atom, xyz, xyz) layout and Eh/Bohr**2 units.
The dense reference is deliberately bounded to 18 AOs and four atoms.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf

_H1 = 1e-5  # 1st-derivative integral step
_H2 = 3e-4  # 2nd-derivative integral step (tunable; see module docstring)


def build_mol(atoms, basis, charge=0, spin=0):
    """Build a PySCF mol in Bohr / cartesian from VibeQC-style atoms+basis.

    ``atoms`` is a list of (Z, [x, y, z]); ``basis`` maps an element tag to a
    list of shells, each shell = [L, (a1, c1), (a2, c2), ...].
    """
    atom, bas = [], {}
    for ai, (z, xyz) in enumerate(atoms):
        tag = f"{gto.mole._symbol(z)}{ai}"
        atom.append([tag, [float(v) for v in xyz]])
        bas[tag] = basis[tag]
    return gto.M(
        atom=atom,
        basis=bas,
        charge=charge,
        spin=spin,
        unit="Bohr",
        cart=True,
        verbose=0,
    )


def _mol_at(mol0, coords):
    """Rebuild ``mol0``'s geometry with new cartesian coordinates (Bohr)."""
    atom = [
        [mol0.atom_symbol(i), list(map(float, row))] for i, row in enumerate(coords)
    ]
    return gto.M(
        atom=atom,
        basis=mol0.basis,
        charge=mol0.charge,
        spin=mol0.spin,
        unit="Bohr",
        cart=True,
        verbose=0,
    )


def _validate_step(step):
    """Reject invalid difference steps before rebuilding any molecules."""
    if not np.isfinite(step) or step <= 0:
        raise ValueError("finite-difference step must be finite and positive")


def _validate_molecule(mol):
    """Keep this dense all-electron RHF oracle within its qualified domain."""
    if (
        not mol.cart
        or mol.spin != 0
        or mol.nelectron % 2
        or mol.has_ecp()
        or mol.pseudo
    ):
        raise ValueError("reference requires Cartesian all-electron closed-shell RHF")
    if mol.nao_nr() > 18 or not 1 <= mol.natm <= 4:
        raise ValueError("reference is bounded to 18 AOs and four atoms")


def _converged_rhf(mol):
    """Use the same tight SCF policy at every displaced reference geometry."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-13
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 400
    mf.kernel()
    if not mf.converged or not np.isfinite(mf.e_tot):
        raise RuntimeError("reference RHF did not converge")
    return mf


class System:
    """Converged RHF state + coordinate FD derivatives of every integral."""

    def __init__(self, mol, h1=_H1, h2=_H2):
        _validate_molecule(mol)
        _validate_step(h1)
        _validate_step(h2)
        self.mol = mol
        self.nat = mol.natm
        self.nd = 3 * mol.natm
        self.h = h2
        self.h1_ = h1
        self.coords = np.array(mol.atom_coords(), float)
        mf = _converged_rhf(mol)
        self.C = mf.mo_coeff
        self.eps = mf.mo_energy
        self.nocc = int(np.sum(mf.mo_occ > 0.5))
        self.nmo = self.C.shape[1]
        self.nbf = self.C.shape[0]
        if self.nocc == 0 or self.nocc == self.nmo:
            raise ValueError("reference requires occupied and virtual orbitals")
        if np.min(self.eps[self.nocc :]) - np.max(self.eps[: self.nocc]) <= 1e-8:
            raise ValueError("reference requires a nonzero occupied/virtual gap")
        self.P0 = 2 * self.C[:, : self.nocc] @ self.C[:, : self.nocc].T
        self.S0 = mol.intor("int1e_ovlp_cart")
        self.ERI = mol.intor("int2e_cart", aosym=1)
        self.Z = np.array(mol.atom_charges())
        self.occ = np.arange(self.nocc)
        self.virt = np.arange(self.nocc, self.nmo)

    def _h_ao(self, c):
        m = _mol_at(self.mol, c)
        return m.intor("int1e_kin_cart") + m.intor("int1e_nuc")

    def _s_ao(self, c):
        return _mol_at(self.mol, c).intor("int1e_ovlp_cart")

    def _eri(self, c):
        return _mol_at(self.mol, c).intor("int2e_cart", aosym=1)

    def _enuc(self, c):
        c = np.asarray(c)
        e = 0.0
        for a in range(self.nat):
            for b in range(a + 1, self.nat):
                e += self.Z[a] * self.Z[b] / np.linalg.norm(c[a] - c[b])
        return e

    def _d1(self, F, i):
        h = self.h1_
        b = self.coords
        cp = b.copy()
        cp.flat[i] += h
        cm = b.copy()
        cm.flat[i] -= h
        return (F(cp) - F(cm)) / (2 * h)

    def _d2(self, F, i, j):
        h = self.h
        b = self.coords
        if i == j:
            cp = b.copy()
            cp.flat[i] += h
            cm = b.copy()
            cm.flat[i] -= h
            return (F(cp) - 2 * F(b.copy()) + F(cm)) / h**2
        c1 = b.copy()
        c1.flat[i] += h
        c1.flat[j] += h
        c2 = b.copy()
        c2.flat[i] += h
        c2.flat[j] -= h
        c3 = b.copy()
        c3.flat[i] -= h
        c3.flat[j] += h
        c4 = b.copy()
        c4.flat[i] -= h
        c4.flat[j] -= h
        return (F(c1) - F(c2) - F(c3) + F(c4)) / (4 * h**2)

    def _sym(self, d):
        for i in range(self.nd):
            for j in range(i):
                d[(i, j)] = d[(j, i)]
        return d

    def derive(self):
        """Materialize finite-difference integral derivatives at the fixed state."""
        nd = self.nd
        self.h1 = {i: self._d1(self._h_ao, i) for i in range(nd)}
        self.h2 = self._sym(
            {
                (i, j): self._d2(self._h_ao, i, j)
                for i in range(nd)
                for j in range(i, nd)
            }
        )
        self.S1 = {i: self._d1(self._s_ao, i) for i in range(nd)}
        self.S2 = self._sym(
            {
                (i, j): self._d2(self._s_ao, i, j)
                for i in range(nd)
                for j in range(i, nd)
            }
        )
        self.ERI1 = {i: self._d1(self._eri, i) for i in range(nd)}
        self.ERI2 = self._sym(
            {(i, j): self._d2(self._eri, i, j) for i in range(nd) for j in range(i, nd)}
        )
        self.En2 = self._sym(
            {
                (i, j): self._d2(self._enuc, i, j)
                for i in range(nd)
                for j in range(i, nd)
            }
        )

    def Wof(self, eri, P):
        return _wof(eri, P)


def _wof(eri, P):
    return np.einsum("mnls,ls->mn", eri, P) - 0.5 * np.einsum("mlns,ls->mn", eri, P)


def _s2_block(s, ia, ja):
    return np.array(
        [[s.S2[(ia * 3 + x, ja * 3 + y)] for y in range(3)] for x in range(3)]
    )


def _h2_block(s, ia, ja):
    return np.array(
        [[s.h2[(ia * 3 + x, ja * 3 + y)] for y in range(3)] for x in range(3)]
    )


def _eri2_block(s, ia, ja, W2):
    return np.array(
        [
            [
                np.einsum("uvls,uvls->", W2, s.ERI2[(ia * 3 + x, ja * 3 + y)])
                for y in range(3)
            ]
            for x in range(3)
        ]
    )


def _en2_block(s, ia, ja):
    return np.array(
        [[s.En2[(ia * 3 + x, ja * 3 + y)] for y in range(3)] for x in range(3)]
    )


def h1ao(s):
    C = s.C
    P0 = s.P0
    nat = s.nat
    nbf = C.shape[0]
    H = np.zeros((nat, 3, nbf, nbf))
    for ia in range(nat):
        for x in range(3):
            R = ia * 3 + x
            H[ia, x] = s.h1[R] + _wof(s.ERI1[R], P0)
    return H


def _first_order_mo1_e1(s, h1ao):
    """AO-space mo1[ia][x] (nbf, nocc) and mo_e1[ia][x] (nocc, nocc).

    Independent dense solve in the full (nmo, nocc) space. The occupied
    block is fixed by the metric gauge (-S1_mo/2), and its induced density
    participates in the virtual response. A correctly constructed reduced
    occupied/virtual solve is equivalent; this oracle keeps the full space
    to provide a separate implementation for validating the shared solver.
    """
    C = s.C
    eps = s.eps
    nocc = s.nocc
    occ = s.occ
    virt = s.virt
    mocc = C[:, occ]
    nat = s.nat
    nbf = C.shape[0]
    nmo = s.nmo
    e_i = eps[occ]
    e_a = eps[virt]
    e_ai = 1 / (e_a[:, None] - e_i[None, :])  # (nvir, nocc)
    mo1s = np.zeros((nat, 3, nbf, nocc))
    e1s = np.zeros((nat, 3, nocc, nocc))
    for ia in range(nat):
        for x in range(3):
            R = ia * 3 + x
            h1_mo = C.T @ h1ao[ia, x] @ mocc  # (nmo, nocc)
            s1_mo = C.T @ s.S1[R] @ mocc
            hs0 = h1_mo - s1_mo * e_i[None, :]
            mo1base = hs0.copy()
            mo1base[virt, :] = -hs0[virt, :] * e_ai
            mo1base[occ, :] = -s1_mo[occ, :] * 0.5

            def F(mo1):
                dm = C @ (2 * mo1) @ mocc.T
                dm = dm + dm.T
                v = C.T @ _wof(s.ERI, dm) @ mocc
                out = v.copy()
                out[virt, :] *= e_ai
                out[occ, :] = 0
                return out

            dim = nmo * nocc
            Mat = np.zeros((dim, dim))
            for col in range(dim):
                mo1 = np.zeros((nmo, nocc))
                mo1.ravel()[col] = 1.0
                Mat[:, col] = F(mo1).ravel()
            matrix = np.eye(dim) + Mat
            solution = np.linalg.solve(matrix, mo1base.ravel())
            residual = np.linalg.norm(matrix @ solution - mo1base.ravel(), ord=np.inf)
            if not np.isfinite(solution).all() or residual > 1e-10 * (
                1 + np.linalg.norm(mo1base, ord=np.inf)
            ):
                raise RuntimeError("reference CPHF true residual exceeds tolerance")
            mo1 = solution.reshape(nmo, nocc)
            mo1[occ, :] = mo1base[occ, :]
            dm = C @ (2 * mo1) @ mocc.T
            dm = dm + dm.T
            fvind_full = C.T @ _wof(s.ERI, dm) @ mocc
            hs = hs0 + fvind_full
            mo1[virt, :] = hs[virt, :] / (e_i[None, :] - e_a[:, None])
            mo1[occ, :] = mo1base[occ, :]
            mo1s[ia, x] = C @ mo1  # AO-space (nbf, nocc)
            e1s[ia, x] = hs[occ, :] + mo1[occ, :] * (e_i[:, None] - e_i)
    return mo1s, e1s


def _first_order_mo1_e1_vir_only(s, h1ao):
    """Equivalent dense solve on the nonredundant virtual/occupied block.

    The occupied response is known from the metric gauge. Eliminating it from
    ``(I + F) x = b`` gives ``(I + F_vv) x_v = b_v - F_vo b_o``;
    fixing its value never licenses dropping its induced Fock contribution.
    This reference path verifies equivalence without using the full dense solve.
    """
    C = s.C
    eps = s.eps
    nocc = s.nocc
    occ = s.occ
    virt = s.virt
    mocc = C[:, occ]
    nat = s.nat
    nbf = C.shape[0]
    nmo = s.nmo
    nvirt = len(virt)
    e_i = eps[occ]
    e_a = eps[virt]
    e_ai = 1 / (e_a[:, None] - e_i[None, :])  # (nvir, nocc)
    mo1s = np.zeros((nat, 3, nbf, nocc))
    e1s = np.zeros((nat, 3, nocc, nocc))
    for ia in range(nat):
        for x in range(3):
            R = ia * 3 + x
            h1_mo = C.T @ h1ao[ia, x] @ mocc
            s1_mo = C.T @ s.S1[R] @ mocc
            hs0 = h1_mo - s1_mo * e_i[None, :]
            mo1base = hs0.copy()
            mo1base[virt, :] = -hs0[virt, :] * e_ai
            mo1base[occ, :] = -s1_mo[occ, :] * 0.5

            def F(mo1):
                dm = C @ (2 * mo1) @ mocc.T
                dm = dm + dm.T
                v = C.T @ _wof(s.ERI, dm) @ mocc
                out = v.copy()
                out[virt, :] *= e_ai
                out[occ, :] = 0
                return out

            # Reduced (nvirt*nocc) operator: virtual columns only.
            dim = nvirt * nocc
            Fvv = np.zeros((dim, dim))
            for col in range(dim):
                mo1 = np.zeros((nmo, nocc))
                # Advanced indexing followed by ravel() returns a copy.
                # Write through actual row/column indices to seed this basis vector.
                mo1[virt[col // nocc], col % nocc] = 1.0
                Fvv[:, col] = F(mo1)[virt, :].ravel()
            matrix = np.eye(dim) + Fvv
            occupied_response = np.zeros((nmo, nocc))
            occupied_response[occ, :] = mo1base[occ, :]
            rhs = (mo1base - F(occupied_response))[virt, :].ravel()
            Xv = np.linalg.solve(matrix, rhs)
            residual = np.linalg.norm(matrix @ Xv - rhs, ord=np.inf)
            if not np.isfinite(Xv).all() or residual > 1e-10 * (
                1 + np.linalg.norm(rhs, ord=np.inf)
            ):
                raise RuntimeError("reduced CPHF true residual exceeds tolerance")
            mo1 = np.zeros((nmo, nocc))
            mo1[virt, :] = Xv.reshape(nvirt, nocc)
            mo1[occ, :] = mo1base[occ, :]  # frozen at base (not iterated)
            dm = C @ (2 * mo1) @ mocc.T
            dm = dm + dm.T
            fvind_full = C.T @ _wof(s.ERI, dm) @ mocc
            hs = hs0 + fvind_full
            mo1[virt, :] = hs[virt, :] / (e_i[None, :] - e_a[:, None])
            mo1[occ, :] = mo1base[occ, :]
            mo1s[ia, x] = C @ mo1
            e1s[ia, x] = hs[occ, :] + mo1[occ, :] * (e_i[:, None] - e_i)
    return mo1s, e1s


def hessian_total(
    s,
    with_relax=True,
    with_pulay=True,
    with_2e=True,
    with_nuc=True,
    with_core=True,
    mo1e1_fn=_first_order_mo1_e1,
):
    """Semi-numerical reference Hessian, shaped (nat, nat, 3, 3).

    Returns the full Hessian; component isolation is available by toggling the
    ``with_*`` flags (see the negative-case tests).
    """
    C = s.C
    P0 = s.P0
    eps = s.eps
    nocc = s.nocc
    occ = s.occ
    nat = s.nat
    nbf = C.shape[0]
    mocc = C[:, occ]
    W_e = sum(2 * eps[k] * np.outer(C[:, k], C[:, k]) for k in range(nocc))
    W2 = 0.5 * np.einsum("uv,ls->uvls", P0, P0) - 0.25 * np.einsum(
        "ul,vs->uvls", P0, P0
    )

    h1ao = np.zeros((nat, 3, nbf, nbf))
    for ia in range(nat):
        for x in range(3):
            R = ia * 3 + x
            h1ao[ia, x] = s.h1[R] + _wof(s.ERI1[R], P0)

    if with_relax:
        mo1s, e1s = mo1e1_fn(s, h1ao)

    H = np.zeros((nat, nat, 3, 3))
    # Evaluate both atom orders independently. Copying one triangle would
    # hide an inconsistent response or derivative convention in symmetry tests.
    for i0, ia in enumerate(range(nat)):
        for j0, ja in enumerate(range(nat)):
            if with_pulay:
                # S2 already contains both moving AO slots, so the full
                # energy-weighted trace has no extra factor of two.
                H[i0, j0] -= np.einsum("xypq,pq->xy", _s2_block(s, ia, ja), W_e)
            if with_core:
                H[i0, j0] += np.einsum("xypq,pq->xy", _h2_block(s, ia, ja), P0)
            if with_2e:
                H[i0, j0] += _eri2_block(s, ia, ja, W2)
            if with_relax:
                s1ao = np.stack([s.S1[ia * 3 + x] for x in range(3)])
                s1oo = np.einsum("xpq,pi,qj->xij", s1ao, mocc, mocc)
                for x in range(3):
                    for y in range(3):
                        dm1 = mo1s[ja, y] @ mocc.T
                        dm1e = (mo1s[ja, y] * eps[occ][None, :]) @ mocc.T
                        H[i0, j0][x, y] += np.einsum("pq,pq->", h1ao[ia, x], dm1) * 4
                        H[i0, j0][x, y] -= np.einsum("pq,pq->", s1ao[x], dm1e) * 4
                        H[i0, j0][x, y] -= np.einsum("pq,pq->", s1oo[x], e1s[ja, y]) * 2
            if with_nuc:
                H[i0, j0] += _en2_block(s, ia, ja)
    return H


def hessian_components(s):
    """Return each component separately for negative-case isolation.

    Keys: nuclear, core, pulay, two_electron, relaxation. Their sum equals
    hessian_total(s) with every flag on.
    """
    full = hessian_total(s)
    nuc = hessian_total(
        s,
        with_relax=False,
        with_pulay=False,
        with_2e=False,
        with_core=False,
        with_nuc=True,
    )
    core = hessian_total(
        s,
        with_relax=False,
        with_pulay=False,
        with_2e=False,
        with_nuc=False,
        with_core=True,
    )
    pulay = hessian_total(
        s,
        with_relax=False,
        with_2e=False,
        with_core=False,
        with_nuc=False,
        with_pulay=True,
    )
    two_e = hessian_total(
        s,
        with_relax=False,
        with_pulay=False,
        with_core=False,
        with_nuc=False,
        with_2e=True,
    )
    relax = full - (nuc + core + pulay + two_e)
    return {
        "nuclear": nuc,
        "core": core,
        "pulay": pulay,
        "two_electron": two_e,
        "relaxation": relax,
        "total": full,
    }


def fd_hessian(mol, h=1e-4):
    """Total-energy FD Hessian (central), shaped (nat, nat, 3, 3)."""
    _validate_molecule(mol)
    _validate_step(h)
    nat = mol.natm
    nd = 3 * nat

    def mol_at(c):
        return _mol_at(mol, c)

    def E(m):
        return _converged_rhf(m).e_tot

    c0 = np.array(mol.atom_coords())
    H = np.zeros((nd, nd))
    E0 = E(mol_at(c0))
    for i in range(nd):
        cp = c0.copy()
        cp.flat[i] += h
        cm = c0.copy()
        cm.flat[i] -= h
        H[i, i] = (E(mol_at(cp)) - 2 * E0 + E(mol_at(cm))) / h**2
    for i in range(nd):
        for j in range(i + 1, nd):
            pp = c0.copy()
            pp.flat[i] += h
            pp.flat[j] += h
            pm = c0.copy()
            pm.flat[i] += h
            pm.flat[j] -= h
            mp = c0.copy()
            mp.flat[i] -= h
            mp.flat[j] += h
            mm = c0.copy()
            mm.flat[i] -= h
            mm.flat[j] -= h
            H[i, j] = (
                E(mol_at(pp)) - E(mol_at(pm)) - E(mol_at(mp)) + E(mol_at(mm))
            ) / (4 * h**2)
            H[j, i] = H[i, j]
    return H.reshape(nat, 3, nat, 3).transpose(0, 2, 1, 3)


__all__ = [
    "System",
    "build_mol",
    "fd_hessian",
    "h1ao",
    "hessian_components",
    "hessian_total",
]
