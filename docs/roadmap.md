# Implementation roadmap

The repository deliberately separates implemented capability from reserved
interfaces. A milestone is complete only after independent numerical oracles
and performance measurements pass.

## M0: vertical RHF reference — implemented

- Versioned C ABI and C++20 core.
- CUDA 12.9 / `sm_120` context initialization.
- Contracted all-electron Cartesian s-p-d-f Gaussian overlap, kinetic,
  nuclear-attraction, and four-center ERI formulas with first derivatives.
- Closed-shell RHF, DIIS, generalized diagonalization, analytic forces, Python
  `Calculator`, and Torch custom backward.
- H2 and He STO-3G reference tests plus force finite differences.

## M1: production integral foundation

- Specialize/generate the validated McMurchie-Davidson s-p-d-f recurrence by
  shell quartet; evaluate a Rys path where its root formulation improves GPU
  throughput. CUDA direct J/K now maps AO quartets into 55 symmetry-reduced
  exact shell classes, canonicalizes their ERI arguments, and allocates each
  pair's Hermite workspace from its compile-time angular bounds. A 13-node
  total-order scheduler preserves low Graph overhead. Component-unrolled shell
  recurrence and Rys generation remain.
- GPU Schwarz bounds and an O(N^2)-memory fused direct J/K fallback are
  implemented. Every topology retains packed AO-pair metadata used by
  one-electron values/forces; direct buckets also use it for Schwarz bounds.
  Unique shell primitives plus ragged shell-to-AO/primitive metadata are also
  implemented. Canonical ragged shell-pair topology, shell-level Schwarz
  maxima, and fixed shell-quartet offsets are implemented. Direct s/p/d/f RHF
  and UHF now evaluate eightfold-symmetry-unique AO quartets once and scatter
  all J/K contributions. Fixed topology-derived AO-quartet tiling exposes
  large d/f groups across multiple blocks without an O(N^4) descriptor list;
  analytic two-electron forces use the same unique quartets and only the
  participating shell centers. A geometry-dependent device compaction pass now
  selects active shell quartets from those bounds without host readback and
  expands them into only their populated AO tiles. Direct Fock and force reuse
  those exact descriptors, avoiding global-maximum tile padding and repeated
  system/pair decoding. Fixed topology now also records exact tile capacities
  and prefix offsets for every total angular order from 0 through 12. Device
  compaction writes screened tiles directly into those partitions, and direct
  RHF/UHF Fock and force launch separately compiled total-order kernels so
  lower-order tasks no longer inherit the ffff stack footprint. Exact
  shell-class Hermite workspaces reduce Fock stack at orders 0/4/12 to
  0/2440/21240 bytes and force stack to 16/4544/42400 bytes. Clean RTX 5090
  artifacts validate energy, forces, and homogeneous-batch throughput for the
  exact `sdf18-direct`, Cartesian water/def2-SVP, and open-shell
  OH/def2-SVP UHF cases. Broader allocated-GPU regression/performance gates
  remain.
- Compare random values and derivatives against libcint/PySCF before enabling
  each angular-momentum quartet.
- Bundled, reproducible STO-3G/def2-SVP/def2-TZVP Cartesian basis packs for
  H-Ar are generated from the pinned Basis Set Exchange revision. CPU
  real-spherical s-p-d-f transforms and analytic gradients are implemented and
  validated against PySCF/libcint. Sparse CUDA spherical AO consumers are
  public and validated on an RTX 5090 for s/d RHF, s/f RHF, s/d UHF, warm
  batches, standard pure water/def2-SVP RHF, and standard pure OH/def2-SVP UHF
  energy and forces. Their published batch-8 artifacts record 3.17x and 5.33x
  scoped warm speedups over GPU4PySCF's sequential single-system interface.
- Exact shell-class recurrence workspaces and the device-compacted active-task
  scheduler are implemented. Replace the remaining component loops and generic
  symmetry scatter with generated shell-class/Rys kernels where profiling
  justifies them. Retain the present CPU code as an auditable oracle rather
  than a hidden fallback.

## M0.5: native fleet semantics — implemented

- Persistent ragged batch plans and input-ordered per-system results.
- Workload bucketing without global molecule/AO padding.
- Native bounded parallel execution inside compatible CPU buckets.
- Device-resident CUDA buckets with GPU integrals, active-set SCF state,
  cuSOLVER batched eigensolves, and GPU analytic forces.
- Per-system convergence and failure isolation.
- Topology-compatible density warm starts, coordinate updates, cold fallback,
  explicit state clearing, and ragged Torch analytic backward.
- The full CUDA path currently covers the executable contracted Cartesian and
  real spherical s-p-d-f scope. Device DIIS, device-tail Graph control,
  persistent plan arenas,
  Schwarz screening, and a memory-bounded direct J/K fallback are implemented;
  symmetry-reduced direct s/p/d/f quartets contract both Cartesian targets and
  sparse multi-term real-spherical targets through exact shell-class
  workspaces. Component-unrolled/Rys shell kernels, DF J/K, and broader active
  compaction gates remain M1/M2 work and are not implied by the backend label.

## M2: production RHF and UHF

- Packed device matrices, device DIIS, a device-tail Graph, fixed-topology
  replay, small-matrix/cuSOLVER dispatch, and strided-batched cuBLAS matrix
  transforms for direct-J/K AO sizes are implemented for s-p-d-f shells.
- UHF alpha/beta occupations, persistent and screened-direct J/K, combined
  spin DIIS, warm starts, ragged batches, and analytic gradients are
  implemented on CPU and CUDA for the Cartesian and real spherical s-p-d-f
  scope.
- Stream-ordered CUDA memory-pool allocation, exact device shell/AO-tile
  compaction, and reproducible benchmark publication ledgers are implemented.
  Small matrices retain the low-overhead native product kernel, and provider or
  Graph-capture failures retry once through that path. Clean RTX 5090 evidence
  now shows 36.91x warm throughput for `sdf18-direct` batch 16 and 6.07x for
  Cartesian water/def2-SVP batch 8 against sequential GPU4PySCF single-system
  execution. The named-basis OH/def2-SVP UHF batch reaches 12.68x under the
  same interface boundary. The allocated comparison harness now accepts
  minimum-speedup and maximum energy/force error gates, records failures in
  JSON, and returns a failing status after preserving the artifact. Expand the
  gated workload matrix and tune the cuBLAS crossover from measured profiles.
- Deterministic frontier mixing now keeps exact-symmetry open-shell core
  guesses out of known higher-energy occupation fixed points. Add SAD/atomic
  guesses, stability analysis, robust DIIS recovery, level shift, convergence
  diagnostics, ROHF, broader controls, and production UHF performance tuning.
- The cold/warm GPU4PySCF harness includes Cartesian and standard real
  spherical water/def2-SVP cases plus matching Cartesian/spherical OH UHF
  definitions, and publishes raw homogeneous-batch samples. The spherical RHF
  and UHF batch-8 cases pass their allocated accuracy and minimum-speedup gates.
  Water/def2-TZVP Cartesian and spherical definitions now provide the next
  larger-basis profiling and allocated-gate targets.

## M3: density fitting and fleet mode

- Two- and three-center integral kernels, Coulomb metric factorization, DF J/K,
  and complete auxiliary-basis gradient response.
- Extend the implemented CUDA J/K active set with persistent device ERIs,
  streams, CUDA graphs where profitable, and batched small-matrix operations;
  retain identical failure and result ordering.

## Reserved method boundaries

- `WB97M_V` remains discoverable but unavailable until grid, meta-GGA,
  range-separated exchange, VV10, and their force responses have independent
  designs and tests.
- AO-to-MO and MP2 must precede any CCSD implementation. The contraction IR and
  memory planner will be implemented from those real contractions, not from a
  speculative generic tensor API.
