# Batched HF contract and verification

This document defines what “batched HF” means for the current roadmap phase.
The CUDA backend is device resident for the currently accepted RHF/UHF
contracted Cartesian and real spherical s-p-d-f scope; component-unrolled/Rys
shell-task and DF algorithms remain roadmap work.

| Requirement | Implementation | Verification |
| --- | --- | --- |
| Native ragged systems | `vibeqc_batch` stores independent `System` objects and force buffers; no padded molecule tensor exists | Mixed He/H2/H3+/H4 C and Python tests assert exact per-item force shapes |
| Batch scheduler | `FleetPlan` sorts by `(nbf, nocc, primitive_count)` and restores input order | Bucket IDs and independent-energy ordering are tested |
| Compatible-work parallelism | CPU buckets use bounded native worker groups; CUDA buckets use batched integral/matrix kernels and device eigensolves with an active mask | 64-system repeated stress test and CUDA same-bucket comparison |
| Large eigensolves | Above 32 AOs, generic FP64 `XsyevBatched` is used only after an exact setup-time device-launch Graph probe; rejected signatures keep Fock/matrix work in Graphs and use ordinary-stream `Xsyevd` one matrix at a time | API/product boundary tests plus 512/513/768-AO probe and fallback checks on an allocated RTX 5090 |
| Per-system convergence | Every item owns status, iteration count, residuals, and convergence flag | H2 succeeds while H3+ intentionally fails with `max_iterations=2` |
| Failure isolation | Invalid coordinates and nonconvergence do not abort structurally valid neighbors | Native and Python isolation tests |
| Fixed-topology plans | Prepared systems own reusable CUDA arenas, solver state, Graph executables, and exact-coordinate geometry caches; changed coordinates rebuild all geometry-derived state before replay | Same-coordinate warm replay and coordinate-update energy are compared with independent calculations |
| Warm starts | One converged AO density is retained per topology, symmetrized, and electron-trace normalized for new geometry | Cold/warm iteration counts and explicit clear operation are tested |
| Differentiable batches | Torch accepts a sequence of ragged coordinate tensors and calls native analytic forces in backward | Ragged H2/H3+ backward checks total gradient invariance |
| Throughput evidence | Benchmarks report independent, cold-batch, and warm-batch timing for configurable batch size/backend | `benchmarks/batch_throughput.py` exercised at batch 1/16/64 |

## Backend labels

- `cpu_reference`: integrals, SCF, and gradients use the reference backend;
  compatible systems execute in bounded native CPU worker groups.
- `hybrid_cuda`: retained as an ABI value for older prototypes; current code
  does not report it.
- `cuda`: overlap, core Hamiltonian, ERI, nuclear repulsion, SCF matrices,
  device eigensolves, convergence state, final energy, and analytic forces
  execute on the GPU for the supported RHF/UHF contracted Cartesian or real
  spherical s-p-d-f basis scope.

For the current small validation workloads, the CUDA path is a correctness and
execution-architecture milestone rather than a speed claim. Published
production performance claims require generated shell-quartet kernels,
DF J/K, finer AO-level compaction, broader spherical workloads, and larger
realistic benchmark oracles. A device-only shell-quartet compaction pass is
implemented and validated on allocated RHF/UHF workloads. The present policy
retains
ERIs for AO count <=16 and uses
O(N^2)-memory screened direct J/K above that threshold. Direct buckets retain a
canonical packed-pair table in their topology arena. For s/p/d/f topologies,
the 55 exact shell-class evaluators consume normalized Cartesian source AOs
under the same symmetry-reduced scheduler. Public real-spherical densities are
mapped with `C^T D C` before direct J/K, and the Cartesian Fock is restored with
`C F C^T`; all transforms remain inside the device Graph.
Fixed shell-pair-pair offsets define eightfold-symmetry-unique quartets; one
logical AO-quartet tile evaluates each unique integral once and scatters all
RHF/UHF J/K contributions. A fixed topology-derived tile multiplicity exposes
large d/f groups across one-warp blocks without per-quartet descriptors. Each
256-quartet logical descriptor expands virtually into eight subtiles, keeping
many high-register recurrence blocks independently schedulable without
growing the descriptor arena or wasting sub-warp lanes. Before every direct
Fock build, device reductions form separate Coulomb and exchange density
maxima for each shell pair. Graph-resident compaction combines those values
with the Schwarz bounds, so SCF iterations avoid quartets that cannot
materially update J or K at the requested threshold. The analytic-force pass
contracts the final-density task list and visits only its shell-center
coordinates. ERI translational invariance evaluates `N-1` unique
centers and reconstructs the last derivative, reducing force recurrence work
without changing the stationary gradient. The one-electron force assigns one
warp to each public AO pair, forms exact raised/lowered overlap and kinetic
derivatives on lane zero, and distributes point-charge nuclei across the warp.
All active nuclear lanes reuse one shared primitive/component Hermite table;
the nuclear-center attraction derivative follows from translation, so this
path needs neither a host launch per nucleus nor a full dual recurrence per
atom coordinate. Dedicated analytic formulas handle
total angular orders zero through five; their sparse high-order formulas share
each raised Coulomb state across all centers and reconstruct the fourth basis
center from translational invariance. Orders six and above use a force-only
three-component forward scalar that propagates x/y/z together so the exact
shell-class value recurrence executes once rather than three times.
Every converged system retains the density paired with its last raw Fock
matrix. A final step at or below `1e-12` RMS reuses that snapshot; only looser
systems restore their next densities and rebuild Fock. Because systems can
converge on different device-tail iterations, a multi-system bucket recreates
the final transformed densities, shell-pair bounds, and compact task list after
the selective rebuild. This metadata-only pass makes the force domain
consistent without reevaluating J/K for tight peers.
After force kernels consume the selected snapshots, reused systems restore the
accepted next densities as their returned warm states. An unchanged-geometry
replay whose warm density exactly matches the previous successful return also
reuses the resident final energy as its comparison baseline. It can therefore
converge in one SCF Graph iteration when the unchanged energy and density tests
already pass; changed coordinates or external warm inputs retain the original
two-energy-evaluation protocol.
Because the force pass follows the final SCF Graph replay, orders zero through
five consume their compacted partitions through persistent device task queues
rather than fixed topology-capacity grids. One-warp blocks steal subtiles until
the device count is exhausted, with eight worker blocks per multiprocessor.
The generic orders six through twelve keep fixed grids to avoid increasing the
resource pressure of their 254-register kernels. This scheduler changes only
which worker consumes an already compacted quartet; screening, density
contraction, integral formulas, and force accumulation are unchanged.
For unchanged coordinates, prepared plans retain overlap, core Hamiltonian,
Schwarz/shell-pair bounds, nuclear repulsion, and the orthogonalizer. Exact
coordinate comparison invalidates this cache before a coordinate update. If
all systems carry valid warm densities, their immediately overwritten
core-Hamiltonian guesses are skipped; mixed warm/cold batches retain the cold
guess path. SCF and force state remain dynamic on every execution.
Finer AO-level active compaction remains a later scheduler milestone.

## Eigensolver dispatch

The solver hierarchy is explicit per fixed-topology bucket. Matrices through
16 AOs use the low-overhead native kernel, 17--32 AOs use cuSOLVER batched
Jacobi, and larger FP64 matrices first undergo checked `XsyevBatched` API
eligibility. CUDA 12.9 documents `n <= 32768` and
`n * lda * solver_batch <= INT32_MAX`; `solver_batch` is the physical RHF
batch and the doubled alpha/beta batch for UHF.

API eligibility does not imply device-launch Graph eligibility. Setup probes
the exact device, CUDA/cuSOLVER stack, matrix dimension, solver batch, and
workspace signature on a private stream. It validates ordinary execution,
capture, device-launch instantiation, host replay, one device-tail replay,
eigenvalue residuals, and orthogonality. A provider that rejects Graph capture
keeps the standard ordinary-stream `Xsyevd` implementation without failing
the calculation, and warm replays reuse the recorded decision without probing
or allocating again. On the
measured RTX 5090 with CUDA 12.9, 512 AOs passes this stronger Graph contract
while 513 and 768 AOs execute normally in cuSOLVER but reject capture. Their
Fock/matrix work stays in reusable Graphs, while setup, finalization, and the
ordinary-stream iteration gap use the standard single-matrix `Xsyevd` provider
(one call per physical or spin matrix), matching GPU4PySCF's robust large-
matrix strategy. This observed 512/513 transition is not a cuSOLVER matrix
size limit.

For diagnosing or gating a known provider regression, setting
`VIBEQC_GRAPH_EIGENSOLVER_OVERRIDE=graph_native` selects the Graph-native
solver for both captured iterations and ordinary/finalization solves. This
override bypasses the cuSOLVER probe entirely, so a rejected or unstable
provider cannot re-enter through a split ordinary-stream path. It is an
explicit benchmark/diagnostic switch; normal runs continue to use the
setup-qualified dispatch above.

The cache is plan-local rather than a process-global manifest: a different
device or matrix/batch signature creates a different bucket plan, and a new
process/toolkit stack probes again. Device ordinal, name, UUID, compute
capability, runtime/driver versions, and cuSOLVER version remain in the record
so any future persistent profile can reject a stale key explicitly.

`PreparedBatch.last_eigensolver_diagnostics()` and the corresponding C ABI
getter publish one record per executed bucket, including ordinary and Graph
provider families, selection source, dimensions, CUDA/cuSOLVER versions,
workspace sizes, API decision, probe stage/status codes, and validation errors.
Batch comparison JSON stores the same records under
`vibeqc.eigensolver_diagnostics`.

CUDA density-fitting batches expose an analogous metric/allocation record via
`PreparedBatch.last_density_fitting_metric_diagnostics()` and
`vibeqc_batch_get_last_density_fitting_metric_diagnostics`. Each record reports
the effective metric rank, condition number, selected auxiliary tile, solver
workspace, and conservative host/device resident and peak byte counts, along
with the bucket id and original prepared-batch item index.
