# Hartree-Fock and SCF

Hartree-Fock (HF) approximates the many-electron wavefunction with a single Slater determinant. Its effective one-electron equations depend on the orbitals being solved for, so HF is normally solved iteratively.

The **self-consistent field (SCF)** loop starts from a density/orbital guess, builds a Fock operator, solves for updated orbitals, forms a new density, and repeats until the convergence contract is satisfied.

“SCF converged” means the iterative equations converged. It does not prove that the chosen method and basis are chemically accurate.

RHF is a restricted form; UHF permits different alpha and beta spin orbitals. Kohn-Sham DFT uses a related self-consistent structure.

Next: [DFT and functionals](dft-and-xc.md).
