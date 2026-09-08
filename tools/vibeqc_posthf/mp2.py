"""Small canonical RHF MP2 bridge; neither a registered method nor a solver."""

from dataclasses import dataclass

import numpy as np

from .conventions import MOBlock, ovov_to_ijab
from .reference import immutable


@dataclass(frozen=True)
class MP2Result:
    """Restricted t2[i,j,a,b]=(ia|jb)/(eps_i+eps_j-eps_a-eps_b)."""

    correlation_energy: float
    amplitudes: np.ndarray
    minimum_absolute_denominator: float
    reference_id: str
    hamiltonian_id: str


def restricted_mp2(snapshot, provider, *, denominator_threshold=1e-10):
    """E=sum_ijab t2_ijab [2(ia|jb)-(ib|ja)]; do not clamp denominators.

    Small denominators are a diagnosed failure. All occupied and virtual
    spatial orbitals participate, with restricted pair symmetry ijab<->jiba.
    Device integrals are explicitly downloaded for this CPU bridge.
    """
    if provider.snapshot.identity != snapshot.identity:
        raise ValueError("MP2 provider/reference mismatch")
    if not np.isfinite(denominator_threshold) or denominator_threshold <= 0:
        raise ValueError("denominator_threshold must be finite and positive")
    g = ovov_to_ijab(provider.get(MOBlock.from_spaces(snapshot, "ovov")).to_host())
    o = snapshot.orbital_energies[: snapshot.nocc]
    v = snapshot.orbital_energies[snapshot.nocc :]
    denominator = (
        o[:, None, None, None]
        + o[None, :, None, None]
        - v[None, None, :, None]
        - v[None, None, None, :]
    )
    minimum = float(np.min(np.abs(denominator)))
    if minimum <= denominator_threshold:
        where = tuple(
            int(i)
            for i in np.unravel_index(np.argmin(np.abs(denominator)), denominator.shape)
        )
        raise ValueError(
            f"near-zero MP2 denominator at ijab={where}: {denominator[where]}; no regularization applied"
        )
    if np.any(denominator >= 0):
        raise ValueError("MP2 requires occupied energies below virtual energies")
    t = g / denominator
    energy = float(
        np.einsum("ijab,ijab->", t, 2 * g - g.swapaxes(2, 3), optimize=False)
    )
    if not np.isfinite(energy) or not np.isfinite(t).all():
        raise ValueError("nonfinite MP2 result")
    return MP2Result(
        energy, immutable(t), minimum, snapshot.identity, snapshot.hamiltonian_id
    )


def spin_orbital_mp2(snapshot, ovov, *, maximum_spin_orbitals=24):
    """Independent explicit spin summation, E=1/4 sum |<IJ||AB>|**2/D.

    <iσ jτ||aυ bω>=(ia|jb)δσυδτω-(ib|ja)δσωδτυ. This small oracle
    independently verifies restricted exchange/factors without reusing t2.
    """
    if 2 * snapshot.nmo > min(maximum_spin_orbitals, 24):
        raise ValueError("spin-orbital oracle is small-only")
    no = snapshot.nocc
    nv = snapshot.nmo - no
    eps = snapshot.orbital_energies
    energy = 0.0
    for I in range(2 * no):
        i, si = divmod(I, 2)
        for J in range(2 * no):
            j, sj = divmod(J, 2)
            for A in range(2 * nv):
                a, sa = divmod(A, 2)
                for B in range(2 * nv):
                    b, sb = divmod(B, 2)
                    element = (ovov[i, a, j, b] if si == sa and sj == sb else 0.0) - (
                        ovov[i, b, j, a] if si == sb and sj == sa else 0.0
                    )
                    denominator = eps[i] + eps[j] - eps[no + a] - eps[no + b]
                    if abs(denominator) <= 1e-10:
                        raise ValueError("near-zero spin-orbital MP2 denominator")
                    energy += 0.25 * element * element / denominator
    return energy
