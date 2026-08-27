# Density-fitting implementation boundary

VibeQC currently provides the correctness and planning foundation for density
fitting. Production RHF and UHF calculations still use the existing direct
four-center J/K path; selecting a density-fitting SCF mode is not yet supported.

## Implemented foundation

- Independent host evaluation of the auxiliary Coulomb metric `(P|Q)` and
  three-center integrals `(mu nu|P)`.
- First nuclear derivatives with coordinates shared by the orbital and
  auxiliary bases, including translation-invariance and finite-difference
  checks.
- Independent Cartesian and real-spherical transforms for orbital and
  auxiliary shells through the public `s`-through-`f` basis limit.
- Symmetric metric inverse square root with a configurable relative
  linear-dependence threshold, effective-rank reporting, and a condition-number
  diagnostic.
- Host-reference metric-orthonormalized three-center tensors and RHF/UHF RI-J/K
  contractions. The density-based exchange schedule uses the same pair of
  matrix multiplications intended for blocked accelerator execution without
  materializing four-center ERIs.
- A persistent homogeneous CUDA J/K plan performs device-side metric
  eigendecomposition and inverse-square-root construction, cuBLAS three-center
  transforms and RI-J, and auxiliary-tiled two-GEMM RI-K for RHF and UHF. The
  transformed tensor remains resident across repeated density contractions.
- A deterministic planner for batch, AO-pair, auxiliary, and occupied-orbital
  tiles. Its workspace estimate includes the permanent metric factor and does
  not require the full three-center tensor when that tensor exceeds the budget.

The integral routines are a CPU numerical oracle for subsequent accelerator
kernels. They intentionally do not provide a silent CPU fallback from a future
GPU density-fitting mode.

## Remaining in issue #5

- CUDA batched two-/three-center integral evaluation kernels.
- Prepared, reusable auxiliary-basis topology and workspaces for fixed-topology
  batches.
- Device-resident SCF integration, CUDA Graph replay, and planner-driven
  streaming when the complete transformed three-center tensor exceeds budget.
- Complete RI-J/K analytic-force response, including all auxiliary and Pulay
  terms.
- Warm CUDA Graph replay, per-system failure isolation, memory measurements,
  and the 96-/192-AO performance gates against GPU4PySCF density fitting.

Issue #5 remains open until those energy, force, integration, and performance
acceptance criteria are met.
