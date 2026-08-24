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
  total-order scheduler preserves low Graph overhead. The canonical
  `(p s | s s)` class now uses a closed two-term first-order contraction for
  values and three-axis derivatives. The total-order-2 `(d s | s s)`,
  `(p p | s s)`, and `(p s | p s)` classes use compact four-term pair
  expansions and closed Coulomb derivatives for the same value/three-axis
  scalar paths. Total-order-3 `(f s | s s)`, `(d p | s s)`,
  `(d s | p s)`, and `(p p | p s)` classes use exact eight-term quantum
  expansions with same-axis Gaussian contractions and closed third-order
  Coulomb derivatives. Total-order-4 `(f p | s s)`, `(d d | s s)`,
  `(f s | p s)`, `(d p | p s)`, `(d s | d s)`, `(d s | p p)`, and
  `(p p | p p)` classes extend that expansion through all same-axis single
  and disjoint double Wick contractions and closed fourth-order Coulomb
  derivatives. Component-unrolled recurrence for order five and above and Rys
  generation remain.
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
  participating shell centers. Translational invariance now reduces their
  derivative evaluations from N unique centers to N-1 and reconstructs the
  final center exactly. The remaining x/y/z derivatives for each center now
  propagate through one force-only three-component scalar, avoiding three
  repetitions of the shell-class value recurrence. A geometry-dependent
  device compaction pass now selects active shell quartets from those bounds
  without host readback and expands them into only their populated AO tiles.
  Direct Fock and force reuse those exact descriptors, avoiding global-maximum
  tile padding and repeated system/pair decoding. Fixed topology now also
  records exact tile capacities and prefix offsets for every total angular
  order from 0 through 12. Device
  compaction writes screened tiles directly into those partitions, and direct
  RHF/UHF Fock and force launch separately compiled total-order kernels so
  lower-order tasks no longer inherit the ffff stack footprint. Each compact
  256-quartet descriptor expands virtually into eight one-warp subtiles,
  exposing independent blocks for these high-register consumers without
  growing fixed-topology metadata or wasting sub-warp lanes. Cartesian-source
  shell-class workspaces use Fock stack at orders 0/4/12 of
  304/2488/21288 bytes. Three-component force propagation uses
  576/8608/84384 bytes at those orders; its larger per-thread workspace is
  offset by evaluating each center once instead of once per Cartesian axis.
  Closed all-center derivatives further reduce order-0/order-1 force stacks
  from 576/680 bytes to 112/208 bytes. On the 96-AO WATER27 batch-1 profile,
  their kernels fall from 97.52/356.34 ms to 55.30/115.40 ms.
  The closed order-2 path reduces its Fock/force stacks from 1072/3184 bytes
  to 464/888 bytes; order three falls from 1568/5200 bytes to 512/1256 bytes;
  order four falls from 2488/8608 bytes to 752/2024 bytes.
  Clean RTX 5090
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
  A larger 43-AO pure water/def2-TZVP batch-4 artifact records 13.27x after the
  closed order-1/order-2/order-3/order-4 recurrences, virtual one-warp tiling,
  Cartesian-source contraction, the Graph-native eigensolver, ERI force-center
  reduction, and three-component force propagation; the matching 48-AO
  Cartesian artifact records 5.84x.
- Exact shell-class recurrence workspaces, Cartesian-source contraction for
  public spherical direct J/K, and the device-compacted active-task scheduler
  are implemented. Replace the remaining recurrence component loops and
  generic symmetry scatter with generated shell-class/Rys kernels where
  profiling justifies them. Retain the present CPU code as an auditable oracle
  rather than a hidden fallback.

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
  symmetry-reduced direct s/p/d/f quartets use exact Cartesian-source
  shell-class workspaces. Public real-spherical densities and Fock matrices
  transform on device around those consumers. Component-unrolled/Rys shell
  kernels, DF J/K, and broader active compaction gates remain M1/M2 work and
  are not implied by the backend label.

## M2: production RHF and UHF

- Packed device matrices, device DIIS, a device-tail Graph, fixed-topology
  replay, small-matrix/cuSOLVER/Graph-native Jacobi dispatch, and
  strided-batched cuBLAS matrix transforms for direct-J/K AO sizes are
  implemented for s-p-d-f shells. The cooperative Jacobi path removes the
  provider capture failure above 32 AOs and is validated at 48 AOs with
  water/def2-TZVP energy and forces. Its 33--256-AO range now uses parallel
  cyclic sweeps over disjoint round-robin rotations; the prior unbounded
  maximum-pivot kernel remains as the larger-matrix fallback. WATER27 checks
  exercise the new path at 96 and 192 AOs: energies remain within
  `5.2e-13 Eh` and `7.4e-12 Eh`, respectively, of the prior path, and the
  192-AO batch-4 force arrays remain within `9.8e-11 Eh/bohr`.
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
  Water/def2-TZVP Cartesian and spherical batch-4 cases also pass; extend the
  matrix to more molecular topologies.

### Explicit realistic scale gates

- The fixed 96-AO workload is the optimized GMTKN55/WATER27 hydrogen-bonded
  water tetramer with the standard real-spherical def2-SVP basis. Batch 1 and
  batch 4 must each stay at
  or below `3e-11 Eh` maximum energy error and `3e-11 Eh/bohr` maximum force
  error under `1e-12 Eh`/`1e-10` SCF energy/density tolerances and a `1e-14`
  QCE Schwarz-screening threshold matching GPU4PySCF's direct-SCF threshold.
  GPU4PySCF uses its separate `1e-9` orbital-gradient threshold at this size;
  `1e-10` can cycle at its numerical-noise floor for the scaled batch
  endpoints. A full batch-4 replay at `1e-9` converged all reference systems
  with `1.02e-12 Eh` maximum energy error and `1.81e-11 Eh/bohr` maximum force
  error.
  Both engines receive up to 100 SCF iterations. Each point must be at least
  `1.0x` as fast as GPU4PySCF under the documented
  sequential single-system interface boundary.
- The fixed 192-AO workload is the optimized GMTKN55/WATER27 S4 water octamer
  with the same basis and representation. Batch 1 and batch 4 must stay below
  `1e-10 Eh` maximum energy error and `5e-10 Eh/bohr` maximum force error now.
  GPU4PySCF uses a `1e-8` orbital-gradient threshold here; tighter settings
  reach its numerical-noise floor for some scaled batch members while the
  `1e-8` and `1e-9` converged energy plateaus agree within `1e-11 Eh` on the
  base geometry. Their performance threshold is deliberately unset
  for the direct-J/K phase; after complete DF J/K lands, both points acquire a
  `1.0x` minimum-speedup gate.
- `benchmarks/real_molecule_gate.py` runs all four points and preserves one
  child artifact per point plus an aggregate summary. Replicated or separated
  water grids remain useful scaling diagnostics but do not satisfy this
  acceptance gate.

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
