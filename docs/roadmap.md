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
  throughput.
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
  participating shell centers. Consume this next
  in geometry-dependent active compaction, generated angular-specialized
  quartet kernels, and spherical transforms.
- Compare random values and derivatives against libcint/PySCF before enabling
  each angular-momentum quartet.
- Bundled, reproducible STO-3G/def2-SVP/def2-TZVP Cartesian basis packs for
  H-Ar are generated from the pinned Basis Set Exchange revision. Extend the
  pack only after pure transforms define standard named-basis semantics.
- Replace the generic quartet recurrence/scatter with generated shell-class
  kernels and a production active-task scheduler. Retain the present CPU code
  as an auditable oracle rather than a hidden fallback.

## M0.5: native fleet semantics — implemented

- Persistent ragged batch plans and input-ordered per-system results.
- Workload bucketing without global molecule/AO padding.
- Native bounded parallel execution inside compatible CPU buckets.
- Device-resident CUDA buckets with GPU integrals, active-set SCF state,
  cuSOLVER batched eigensolves, and GPU analytic forces.
- Per-system convergence and failure isolation.
- Topology-compatible density warm starts, coordinate updates, cold fallback,
  explicit state clearing, and ragged Torch analytic backward.
- The full CUDA path currently covers the executable contracted Cartesian
  s-p-d-f scope. Device DIIS, device-tail Graph control, persistent plan arenas,
  Schwarz screening, and a memory-bounded direct J/K fallback are implemented;
  symmetry-reduced direct s/p/d quartets are also implemented. Spherical
  transforms, generated shell kernels, DF J/K, and active compaction remain
  M1/M2 work and are not implied by the backend label.

## M2: production RHF and UHF

- Packed device matrices, device DIIS, a device-tail Graph, fixed-topology
  replay, and small-matrix/cuSOLVER dispatch are implemented for s-p-d-f shells.
- UHF alpha/beta occupations, persistent and screened-direct J/K, combined
  spin DIIS, warm starts, ragged batches, and analytic gradients are
  implemented on CPU and CUDA for the Cartesian s-p-d-f scope.
- Add a CUDA memory pool, active compaction, scalable BLAS-backed matrix
  transforms, and production error/publication ledgers.
- Core/SAD guesses, robust DIIS recovery, level shift, convergence diagnostics,
  ROHF, broader convergence controls, and production UHF performance tuning.
- Extend the existing cold/warm GPU4PySCF s/p microbenchmark to realistic
  common basis sets after spherical transforms and shell-task direct J/K are
  validated.

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
