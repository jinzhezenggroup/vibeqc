# Density-fitting implementation boundary

The [generated DF response](df_derivatives.md) contracts independent
three-center and metric weights on CUDA without complete nuclear-derivative
tensors. That page documents the RHF/UHF reverse chain, metric rank-crossing
diagnostics, and measured endpoint evidence. The positive-budget path uses
[bounded device response](df_device_replay.md) for both J/K and final forces.

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
  One-electron/overlap Pulay response uses the separate #141 generated consumer.
  CUDA response failures propagate; independent CPU response serves CPU callers.
- A persistent homogeneous CUDA J/K plan performs device-side metric
  eigendecomposition and inverse-square-root construction, cuBLAS three-center
  transforms and RI-J, and auxiliary-tiled two-GEMM RI-K for RHF and UHF. The
  transformed tensor remains resident across repeated density contractions when
  it fits the selected storage policy. Generated setup borrows bounded K scratch
  for raw panels and writes all transformed Q into retained B; retaining B no
  longer requires three full-tensor contraction buffers. Positive-budget plans
  use at most 128 contraction auxiliaries when B is retained, shrinking Q further
  when needed. Plans that cannot retain B regenerate public-basis tiles on device.
  Both routes retain the metric factors and eigensystem. The native
  compatibility API also supports explicitly host-backed input tensors.
- A deterministic planner for batch, AO-pair, auxiliary, and occupied-orbital
  tiles. Its positive budget bounds the persistent CUDA plan and bounded
  generation/contraction tiles. A positive force request is split equally between
  this value/J/K plan and generated force staging; energy-only J/K uses the whole
  allowance. Property changes replan cached value storage. Metric diagnostics describe
  the value/J/K plan; the whole-HF resource ledger also charges response storage.
  CUDA HF calls do not retain full derivative tensors between executions.
  When retained B and its bounded scratch do not fit, positive budgets fund larger panels
  from the remaining allowance. Dense and occupied K share a row/raw-P/output-Q
  policy that minimizes raw source evaluations across repeated column visits and
  Q blocks. Every raw panel feeds all active output auxiliaries through GEMM.
  Tight capacities still require recomputation; zero keeps the established
  allocation defaults. The progress journal reports the executed widths and
  predicted raw tensor passes so this cost remains visible.
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
  bounded AO-pair/auxiliary tiles and use persistent workspaces. Source-backed
  J/K regeneration launches on the owning stream without host tensor transfers.
- Generated weighted RI-J/K analytic-force response supports RHF and UHF,
  including metric pseudoinverse and auxiliary response terms. Positive-budget
  source plans construct those weights on device and report metadata/density
  uploads separately from tensor transfers. The #141 one-electron and Pulay
  consumer retains the CPU oracle's variational weighted-density convention.
- `benchmarks/real_molecule_gate.py --density-fitting cuda` runs a separate
  DF acceptance matrix for 96-, 192-, and 384-AO workloads.  It records the
  selected DF settings, metric conditioning/effective rank, resident and peak
  allocation diagnostics, and cold setup versus warm contraction timing.  The
  historical direct-SCF matrix remains unchanged when the flag is omitted.

Host-backed compatibility execution and external GPU4PySCF availability are
explicit in benchmark artifacts. The direct-SCF gates are independent of DF
execution placement.

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

### Rejected SCF graph capture

A prepared CUDA DF SCF owner may use the same device eigensolver through
ordinary stream execution when graph capture is unsupported for its actual
signature. After ending a failed capture, the runtime clears the expected
capture error before submitting ordinary work; assigning a local success
status alone does not clear CUDA's error slot. The prepared owner remembers
that rejected capture, including on warm replay. A new owner checks again.
This policy applies to both RHF and UHF and adds no fixed AO dimension limit.
Unrelated runtime errors and allocation failures remain failures. SCF
convergence and strict final-state checks are unchanged.

### Compact DIIS and numerical recovery

Production CUDA DF SCF honors the requested DIIS history inside the compact
device iteration. It builds the physical `FDS-SDF` residual with cuBLAS and
reuses the existing normalized device DIIS kernel; UHF joins both spin
residuals under one coefficient vector per item. Energy is evaluated before
extrapolation, and strict final selection still validates the physical Fock at
the returned density. DIIS proposals are not accepted as physical final frames
without those checks.

Overlap, history matrices, residual/packing scratch, the small Gram solve and
ring controls belong to the prepared SCF owner. History resets on every solve;
inactive items retain their state within a solve. History changes invalidate
the plan's memory choice, and the native and global planners reserve the same
capacity before choosing retained/streamed K panels. Requested iteration limits
and failure statuses remain authoritative. The old internal overload/ABI keeps
its fixed-point behavior. `VIBEQC_DF_DISABLE_DEVICE_DIIS=1` restores that compact
behavior for diagnostics while preserving the same declared memory reservation.

Metric setup and compact batched eigensolves also have separate checked
workspace allowances, including their fixed provider floors. Their actual
queries must fit those allowances before allocation. Small DF budgets that
cannot hold these owners and DIIS fail explicitly; a value/SCF plan cannot
borrow the response half of a force budget. Diagnostic peak estimates include
both setup and lazy SCF reservations conservatively, even when their lifetimes
do not overlap. Whole-process acceptance remains owned by the global ledger.

When the compact device iteration needs the existing host DIIS retry, CUDA
DF keeps using the prepared ordinary device eigensolver. This applies to
single and fleet RHF/UHF, and records the actual device leaves as `fallback`.
The DIIS sequence, iteration limits and convergence thresholds are unchanged;
provider failures propagate. `VIBEQC_DF_REFERENCE_ITERATION_EIGEN=1` selects the
independent reference operation explicitly for a diagnostic comparison. It
controls only retry iterations, independently of setup and finalization.
