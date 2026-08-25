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
  repetitions of the shell-class value recurrence. Each Graph Fock build now
  reduces the current RHF/UHF density to separate shell-pair Coulomb/exchange
  maxima. A device compaction pass combines them with the geometry-dependent
  Schwarz bounds without host readback and expands only surviving shell
  quartets into their populated AO tiles.
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
  to 464/888 bytes. Its explicit all-center force derivative further lowers
  the force kernel from 254 registers and 888 stack bytes to 188 registers
  and 864 stack bytes. Interleaved candidate/baseline/candidate WATER27
  measurements reduce batch-1 from 2.743 s to 2.499--2.516 s and batch-4 from
  10.698 s to 9.227--9.798 s. The 96-AO batch-1/batch-4 and 192-AO batch-1
  accuracy checks remain within their explicit limits. Order three falls from
  1568/5200 bytes to 512/1256 bytes; order four falls from 2488/8608 bytes to
  752/2024 bytes. Density-conditioned task generation then reduces an
  interleaved 96-AO baseline of 2.509/9.873 s at batch 1/4 to
  2.365--2.382/9.240--9.605 s. The 192-AO batch-1 warm time falls from
  36.110 s to 19.980 s. Formal candidate checks retain 96-AO batch-1/4 errors
  below `4.5e-12 Eh` and `2.6e-12 Eh/bohr`; 192-AO batch-1/4 remain below
  `1.9e-11 Eh` and `2.5e-10 Eh/bohr`.
  Explicit total-order-3 force derivatives subsequently reduce the RHF/UHF
  kernel from 255 registers and 1,256 stack bytes to 180 registers and 672
  stack bytes. Clean WATER27 medians improve to 2.206/8.407 s at 96 AO and
  18.489/92.097 s at 192 AO for batch 1/4, with all four accuracy points
  inside their explicit limits. A profile of the exact 96-AO batch-1 gate
  then identifies total-order-5 force as the largest remaining quartet
  kernel at 197.6 ms per force pass. Its dedicated subset/Wick derivative
  path lowers that kernel to 71.8 ms and its resource use from 254 registers
  and 13,296 stack bytes to 230 registers and 1,952 stack bytes. Interleaved
  96-AO batch-4 candidate/baseline/candidate medians are 7.916/8.412/7.912 s,
  while the 192-AO batch-1/4 correctness checks remain below `5.6e-11` and
  `2.3e-10 Eh/bohr` maximum force error.
  Total-order-4 then reuses the same exact subset/Wick derivative machinery
  through Coulomb order five. Its RHF/UHF kernel falls from 255 registers and
  2,024 stack bytes to 212 registers and 1,376 stack bytes, and its exact
  96-AO force-pass profile falls from 146.7 ms to 86.7 ms. Interleaved
  candidate/baseline/candidate batch-1 medians are 2.021/2.082/2.021 s;
  batch-4 records 7.675/7.894/8.020 s, with the final candidate requiring one
  additional SCF iteration for one system. The 192-AO batch-1/4 checks remain
  below `1.2e-11 Eh` maximum energy error and `2.4e-10 Eh/bohr` maximum force
  error. Dedicated analytic force formulas therefore cover every total
  angular order from zero through five; order six and above remain on the
  general three-component forward scalar.
  The one-electron force now assigns one public AO pair to each worker and
  uses exact Gaussian raising/lowering identities for overlap, kinetic, and
  nuclear-attraction derivatives. Each nucleus shares one Hermite, Boys, and
  Coulomb recurrence across all six basis-center attraction derivatives, with
  its own center derivative recovered by translation. This reduces the
  kernel from 254 registers and 23,208 stack bytes to 167 registers and 9,176
  stack bytes, and the exact 96-AO batch-1 force-pass profile from 109.0 ms to
  22.1 ms. Interleaved candidate/baseline/candidate endpoint medians are
  1.936/2.023/1.936 s at batch 1 and 7.532/7.658/7.007 s at batch 4. The
  batch-4 samples are SCF-branch-sensitive: the first candidate used
  4/2/2/2 iterations while the baseline and final candidate used 2/2/2/2.
  The 192-AO batch-1/4 correctness checks remain below `8.3e-12 Eh` maximum
  energy error and `2.3e-10 Eh/bohr` maximum force error.
  The final analytic force now replaces topology-capacity launches for total
  orders zero through five with device-resident persistent task queues. Eight
  one-warp worker blocks per multiprocessor dynamically consume only the
  compacted final-density subtiles; generic orders six through twelve retain
  fixed grids because queue state regressed their 254-register kernels. On the
  exact 96-AO batch-1 profile, the complete two-electron force pass falls from
  729.8 ms to 466.6 ms. Iteration-matched endpoint A/B measurements improve
  batch-1 from 1.930--1.940 s to 1.671--1.684 s and batch-4 from
  7.326--7.334 s to 5.916--5.929 s. Candidate 96-AO comparisons remain inside
  the energy/force limits but reach only `0.812x` and `0.923x` for batch 1/4,
  so the explicit parity milestone is still open. The 192-AO batch-1/4
  candidate errors remain below `9.7e-12 Eh` and `2.3e-10 Eh/bohr`.
  Prepared CUDA plans now reuse geometry-derived overlap, core Hamiltonian,
  Schwarz/shell-pair bounds, nuclear repulsion, and orthogonalizer state while
  the exact coordinate vector is unchanged. Coordinate updates invalidate and
  rebuild that state; density and convergence state remain dynamic. Fully warm
  homogeneous buckets also skip the core-Hamiltonian guess that warm-density
  normalization immediately replaces. The exact 96-AO batch-1 warm profile
  consequently removes the one-electron/Schwarz preparation kernels and drops
  cyclic eigensolver instances from three to one. Candidate medians improve
  from 1.673/6.242 s to 1.596/6.170 s at batch 1/4, with 96-AO errors below
  `4.4e-12 Eh` and `2.6e-12 Eh/bohr`. The 192-AO batch-1/4 candidate remains
  below `1.1e-11 Eh` and `5.8e-11 Eh/bohr`.
  Dedicated total-order-6 force derivatives now extend the sparse pair
  expansion through three disjoint Wick contractions and canonical 6/0, 5/1,
  4/2, and 3/3 pair splits. Order-specific Coulomb workspaces avoid charging
  lower orders for the larger recurrence. On the clean 96-AO batch-1 profile,
  order six falls from 87.429 to 44.045 ms and the complete two-electron force
  from 465.236 to 423.957 ms. The clean acceptance observations reach
  `0.893x` at batch 1 and an iteration-matched `0.969x` at batch 4, with errors
  below `4.5e-12 Eh` and `2.7e-12 Eh/bohr`. Both required speed gates remain
  open. Order five is now the largest isolated force gap against GPU4PySCF,
  followed by order two and the combined order-6--8 fallback.
  High-order pair-coefficient gradients now move outside the first/second-term
  Cartesian product: the smaller canonical second-pair gradients are cached
  once, while each first-pair gradient is evaluated once per outer term. On
  the next clean profile, order 4/5/6 fall to 58.539/50.458/30.582 ms and the
  complete force pass reaches 385.819 ms. Clean QCE medians improve to
  1.320/5.060 s at batch 1/4 with both accuracy gates passing. Independently
  faster GPU4PySCF samples leave the scoped speed gates at `0.812x`/`0.847x`,
  so the milestone remains open. Order two is now the largest isolated force
  deficit and becomes the next generated/cooperative contraction target.
  The order-two force path now retains only the first-center gradient of each
  pair coefficient, forms the first three full center derivatives with one
  shared raised Coulomb value per coordinate, and reconstructs center four by
  translational invariance. The RHF/UHF persistent kernel falls from 188
  registers and 864 stack bytes to 169 registers and 624 stack bytes. Its
  clean 96-AO force time falls from 88.488 to 65.231 ms; three additional
  isolated captures span 64.708--65.402 ms. The complete two-electron force
  reaches 364.973 ms. Clean endpoint observations pass both accuracy limits
  but remain below parity at `0.920x`/`0.956x`; their raw timings also expose
  repeat-to-repeat SCF-state sensitivity that the harness does not yet record
  per repeat. Order five is now the largest isolated force deficit, while the
  eigensolver and one-electron force are comparable system-level targets.
  High-order pair coefficients and first-center gradients now share one
  subset/Wick matching traversal, and only the compact pair geometry plus the
  smaller canonical second-pair term array remain live. Order-4/5/6 resources
  fall from `215/226/255` registers and `1472/2064/3376` stack bytes to
  `207/218/224` registers and `880/992/1216` bytes. Five 96-AO captures span
  54.125--55.164, 47.453--47.572, and 28.988--29.164 ms respectively; the
  clean complete two-electron force reaches 354.977 ms. A new formal
  three-repeat endpoint run remains accuracy-valid but records only
  `0.922x`/`0.949x` at batch 1/4, so neither required speed gate closes.
  Eigensolver is now the largest system-level gap, followed by order five,
  one-electron force, and order one. The next experimental branch will test
  conservative FP32/log-domain screening metadata while retaining FP64 ERI,
  J/K, force accumulation, SCF matrices, and eigensolver arithmetic.
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
