# Basis sets

Gaussian-basis methods represent molecular orbitals using a finite set of atom-centered functions. The basis set therefore limits the space in which the electronic problem is solved.

A larger or more flexible basis usually costs more computation and can reduce basis-set error. Energies from different basis sets should not be compared as though only geometry changed.

Angular-momentum support also matters: a backend must support every shell required by the chosen basis.

Effective core potentials (ECPs) replace selected core-electron physics with an effective operator, so an ECP calculation is not automatically equivalent to an all-electron calculation.

See [external basis data](../user/external_basis.md), [higher angular momentum](../user/high_angular_momentum.md), and [ECPs](../user/ecp.md).

Next: [Hartree-Fock and SCF](hf-and-scf.md).
