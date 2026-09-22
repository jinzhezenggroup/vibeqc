# Post-HF methods

Hartree-Fock misses electron correlation beyond its mean-field treatment. Post-HF methods add correlation using an orbital reference.

**MP2** is a perturbative correlation method. **CCSD** and **CCSD(T)** use coupled-cluster excitation amplitudes and are generally much more expensive.

Higher cost does not mean universal correctness: multireference character, basis limitations, relativistic effects, and other modeling choices can dominate.

For current availability, use the generated [public method table](../public_methods.md). Implementation details belong in the [Developer Guide](../developer/index.md).
