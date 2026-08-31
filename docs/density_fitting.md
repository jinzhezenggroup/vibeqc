# Density-fitting implementation boundary

VibeQC provides CPU-reference and CUDA density-fitting execution for RHF and
UHF.  Single systems and homogeneous prepared-batch buckets share the same
metric-factorization and RI-J/K implementation; the CUDA bucket path uses one
batched plan for all compatible systems while retaining per-item convergence
and failure status.

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
- Host-reference RHF/UHF two-electron analytic force response. The contraction
  differentiates raw three-center values and the Coulomb metric through its
  pseudoinverse, retaining auxiliary-basis and metric (Pulay) terms without
  differentiating an eigenvector gauge.
- `build_density_fitting_rhf_forces` and `build_density_fitting_uhf_forces`
  combine that response with orbital one-electron derivatives, overlap Pulay
  terms, and nuclear repulsion for callers assembling a complete force pass.
- Hartree-Fock method descriptors can opt into `CPU_REFERENCE`, `CUDA`, or
  `AUTO` density-fitting execution. The prepared single-system and ragged-batch
  paths retain the optional auxiliary shell topology, follow replay geometries,
  and preserve warm-start/result-order semantics. CUDA DF SCF uses a persistent
  J/K plan for a single system and one batched plan per compatible fleet bucket;
  densities, Fock assembly, batched eigensolves, and convergence reductions stay
  on the device, with a host-orchestrated fallback for provider limitations.
  CUDA finalization now dispatches the raw three-center, metric-response, and
  exchange quadratic contractions through the persistent plan stream. The
  one-electron/overlap Pulay assembly remains on the host, with the validated
  host two-electron oracle retained as an automatic fallback for unsupported
  devices or scratch-allocation failures.
- A persistent homogeneous CUDA J/K plan performs device-side metric
  eigendecomposition and inverse-square-root construction, cuBLAS three-center
  transforms and RI-J, and auxiliary-tiled two-GEMM RI-K for RHF and UHF. The
  transformed tensor remains resident across repeated density contractions when
  it fits the selected tile policy. For memory-bounded plans, raw
  three-center values and the metric inverse are retained on the host and one
  transformed auxiliary tile is uploaded per contraction, avoiding a full
  device-resident tensor.
- A deterministic planner for batch, AO-pair, auxiliary, and occupied-orbital
  tiles. Its workspace estimate includes the permanent metric factor and does
  not require the full three-center tensor when that tensor exceeds the budget.
- CUDA DF batch plans retain setup diagnostics for every compatible slot:
  effective rank, metric condition number, solver workspace, selected auxiliary
  tile, and conservative host/device resident and peak byte counts. Native
  C++ callers use `FleetPlan::last_density_fitting_metric_diagnostics()`, C
  callers use `vibeqc_batch_get_last_density_fitting_metric_diagnostics`, and
  the Python equivalent is
  `PreparedBatch.last_density_fitting_metric_diagnostics()`.

The CPU integral routines remain an independent numerical oracle. CUDA DF
preparation now generates raw Cartesian metric/three-center values and first
derivatives on device, then applies the shared public-basis transform; it does
not silently fall back to CPU integral evaluation when CUDA generation fails.

## Remaining in issue #5

- CUDA two-/three-center integral evaluation kernels are now packed across
  homogeneous fleet buckets, including coordinate-major derivative output;
  one-electron generation is exposed through the same batch boundary but still
  dispatches validated per-system launches internally.
- Prepared CUDA DF fleet buckets now retain their SCF state allocations and
  Graph executable for fixed-topology, non-streamed replays.  Inputs and
  convergence masks are refreshed in place, while geometry changes invalidate
  only the affected bucket's geometry-dependent plan.  Streamed plans retain
  bounded AO-pair/auxiliary tiles and use the same persistent workspaces; their
  host tile transfers intentionally remain outside Graph capture.
- Device-resident raw RI-J/K analytic-force response is implemented for RHF
  and UHF, including metric pseudoinverse and auxiliary response terms.
  One-electron and overlap-Pulay assembly remains host-side and uses the same
  variational weighted-density convention as the CPU oracle.
- `benchmarks/real_molecule_gate.py --density-fitting cuda` runs a separate
  DF acceptance matrix for 96-, 192-, and 384-AO workloads.  It records the
  selected DF settings, metric conditioning/effective rank, resident and peak
  allocation diagnostics, and cold setup versus warm contraction timing.  The
  historical direct-SCF matrix remains unchanged when the flag is omitted.

The streamed host-transfer boundary and external GPU4PySCF availability remain
explicitly visible in benchmark artifacts; no direct-SCF gate is weakened when
the DF matrix is unavailable on a given machine.
