# Quantum chemistry in ten minutes

A molecular electronic-structure calculation starts from nuclei, electron count, a representation of the electronic problem, and a chosen approximation.

For ordinary molecular calculations the nuclei are treated as fixed while the electronic problem is solved. The resulting electronic energy is a function of nuclear geometry.

A practical calculation therefore requires explicit choices:

- **System:** elements, coordinates, total charge, and spin state.
- **Representation:** for Gaussian-orbital methods, a basis set.
- **Method:** for example HF, a DFT functional, MP2, or CCSD(T).
- **Property:** energy, forces/gradient, Hessian, or another response.
- **Numerics/backend:** tolerances and execution strategy should change numerical cost/error, not silently change the scientific method.

The most important beginner rule is that an energy is meaningful mainly through **consistent comparisons**. Changing method, basis, charge/spin, Hamiltonian, or numerical accuracy can change the scientific problem.

Next: [molecules, charge, and spin](molecules-charge-spin.md).
