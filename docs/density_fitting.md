# Density-fitting implementation boundary

The [generated DF response](df_derivatives.md) contracts independent
three-center and metric weights on CUDA without complete nuclear-derivative
tensors. That page documents the RHF/UHF reverse chain, bounded host staging,
metric rank-crossing diagnostics, and measured endpoint evidence.

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
  CUDA finalization constructs bounded raw-value response weights and contracts
  the shared generated center derivatives on the persistent plan stream.
  One-electron/overlap Pulay assembly remains on the host. CUDA response failures
  propagate; independent CPU response remains the implementation for CPU callers.
- A persistent homogeneous CUDA J/K plan performs device-side metric
  eigendecomposition and inverse-square-root construction, cuBLAS three-center
  transforms and RI-J, and auxiliary-tiled two-GEMM RI-K for RHF and UHF. The
  transformed tensor remains resident across repeated density contractions when
  it fits the selected tile policy. For memory-bounded plans, raw
  three-center values and the metric inverse are retained on the host and one
  transformed auxiliary tile is uploaded per contraction, avoiding a full
  device-resident tensor.
- A deterministic planner for batch, AO-pair, auxiliary, and occupied-orbital
  tiles. Its positive budget bounds the persistent CUDA plan and bounded
  generation/contraction tiles. A positive DF request is split equally between
  this value/J/K plan and generated force staging. Metric diagnostics describe
  the value/J/K plan; the whole-HF resource ledger also charges response storage.
  CUDA HF calls do not retain full derivative tensors between executions.
- CUDA DF batch plans retain setup diagnostics for every compatible slot:
  effective rank, metric condition number, solver workspace, selected auxiliary
  tile, and conservative host/device resident and peak byte counts. Native
  C++ callers use `FleetPlan::last_density_fitting_metric_diagnostics()`, C
  callers use `vibeqc_batch_get_last_density_fitting_metric_diagnostics`, and
  the Python equivalent is
  `PreparedBatch.last_density_fitting_metric_diagnostics()`.

The CPU integral routines remain an independent numerical oracle. CUDA DF
preparation generates raw Cartesian metric/three-center values on device and
applies the shared public-basis transform. Raw derivative APIs use the same
generated center definitions as weighted HF response. CUDA generation failures
propagate instead of silently selecting CPU integral evaluation.

## Execution notes and acceptance boundary

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
- Generated weighted RI-J/K analytic-force response supports RHF and UHF,
  including metric pseudoinverse and auxiliary response terms. Bounded host
  weight staging and raw-value transfers remain part of the execution cost.
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

## Generated raw integral values

`tools/generate_df_kernels.py` emits the native DF value header from the shared
mathematical IR and expression/CUDA emitters. Its inventory contains every
ordered auxiliary pair (16 metric signatures) and orbital/orbital/auxiliary
triple (64 three-center signatures) through `f`. The primitive expressions use
the exact one- through five-root Rys order, with pruned Gaussian moment DAGs.
The independent Hermite/Wick IR interpreter and libcint fixtures provide
separate arithmetic references. Primitive contraction lengths remain runtime
extents; auxiliary exponents are positive physical exponents, with no
normalized zero-exponent shell in the generated expressions.

The raw outputs are `M[P,Q]=(P|Q)` and `A[mu,nu,P]=(mu nu|P)`, before applying
the metric inverse square root. Layouts are row-major with auxiliary functions
contiguous. AO pairs use `pair=mu*nbf+nu`, including both symmetric entries with
unit weight. There are no hidden `sqrt(2)` or off-diagonal compression factors.
Metric exchange and orbital `mu/nu` exchange are the only declared operator
permutations. The public Cartesian component normalization and libcint-ordered
real spherical conventions use the existing basis transforms.

The existing bounded source supplies ragged AO-pair/auxiliary tiles for each
batch item; final partial tiles preserve the same layout. Empty extents at
valid offsets are no-ops, and dimensions are checked before launches. Source
metadata and per-item public-basis transforms are fixed for a prepared topology.
Geometry changes follow the existing Fleet invalidation and rebuild rules.
Auxiliary shells beyond `l=3` are rejected explicitly, including when a named
auxiliary basis contains them. No shell is silently dropped.

Data placement depends on the existing planner route:

| Route | Integral values | Public-basis transform | Setup staging |
| --- | --- | --- | --- |
| Positive-budget source | CUDA, requested tiles | CUDA at tile writes | Transform metadata H2D; metric D2H then H2D for cuSOLVER |
| Bulk Cartesian source | CUDA | Host, existing independent transform | Raw M/A D2H; normalized J/K input H2D |

`cuda_density_fitting_integral_source_diagnostic` reports the frozen value
backend, mapping, device public transform, and host metric staging. Existing
metric diagnostics retain threshold, effective rank, condition number, and
host/device resident and peak byte counts. The eigendecomposition, regular
cuBLAS contractions and direct-SCF acceptance gates remain native runtime and
independent validation concerns. Removing obsolete derivative scratch from plan
estimates does not by itself establish a measured whole-process peak reduction.

Generated values are the default. For transformed bounded-source outputs,
the compiler assigns four lanes to each primitive product partition and eight
auxiliary source terms to the warp; every lane reaches the final output sum,
including ragged source tails. Raw source tiles keep a full primitive-reduction
warp. The bulk compatibility builder retains one thread per output. The
promotion evidence is in
[`benchmarks/results/generated-df-values-142`](../benchmarks/results/generated-df-values-142/README.md).
`VIBEQC_DF_VALUES` is retired. Reproducing the previous Hermite evaluator requires
the exact historical checkout recorded in that archive.
`VIBEQC_DF_VALUE_MAPPING=auxiliary|component|primitive` compares contiguous
auxiliary writes, contiguous AO-pair work, and one primitive-reduction warp per
output. Auxiliary/component mappings remain diagnostic overrides; the component
mapping was rejected for automatic selection after its endpoint regression.
These source choices are frozen at source creation. Coordinate derivatives
instantiate the shared generated center policy; weighted HF derivatives use
their independently generated four-lane schedule. The mappings share scientific
definitions and rank-generic traversal; no parallel handwritten DF recurrence
or separate four-center tuning pipeline is duplicated. The final source and
energy-only gates are retained in the
[DF retirement bundle](../benchmarks/results/cuda-ownership/df/README.md).

Manual validation tools must run through a finite Slurm allocation:

- `tools/validate_df_values.py` checks every complete Cartesian shell block and
  independent host spherical projection against libcint, including contracted
  and coincident-center fixtures, and records exact object resource reports.
- `tools/validate_df_source.py --probe build/cuda/vibeqc_df_value_probe` checks
  reconstructed full tensors across native tile boundaries, different batch
  primitive offsets and geometries, and RHF/UHF RI-J/K with identical metric
  thresholds. It compares all three source mappings against independent libcint
  tensors and NumPy RI-J/K. Add `--derivatives` to check bulk/source responses.
- `tools/validate_df_endpoints.py` compares public RHF/UHF energy-force
  endpoints at batch one/multiple and two positive budgets plus the bulk route.
  Cold setup/SCF, changed-geometry rebuilding, and warm reuse are reported
  separately with metric ranks and allocation diagnostics.
