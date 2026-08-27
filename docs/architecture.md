# Architecture decisions

## HF-first boundary

RHF and UHF are executable in the HF vertical prototype. `WB97M_V` and
`RCCSD_T` have stable method identifiers for capability discovery but return
`VIBEQC_STATUS_NOT_IMPLEMENTED`. No DFT grid or coupled-cluster tensor framework
is created before a real method requires it.

## Derivative policy

Nuclear forces are native analytic derivatives. The current reference engine
uses forward derivative values inside the primitive integral formulas, then
assembles the stationary RHF derivative from derivative integrals, the
energy-weighted density matrix, and nuclear repulsion. This is deliberately
different from differentiating the eigensolver, DIIS, or SCF iteration trace.

The CUDA s-p-d-f path contracts stationary RHF/UHF gradients on the GPU. Its
one-electron force differentiates Cartesian Gaussians with exact raised/lowered
angular-momentum identities. One warp owns one public AO pair: lane zero forms
the compact overlap/kinetic derivatives, while the remaining lanes treat the
nuclei as a batched point-charge auxiliary dimension. The warp builds one
primitive/component Hermite table in shared memory, reuses it across all active
nuclear-center Coulomb recurrences, reduces the two basis-center derivatives,
and recovers each nuclear-center derivative by translation. Its
Cartesian McMurchie-Davidson recurrence is shared mathematically with the CPU
oracle but implemented independently in CUDA. The canonical first-order
`(p s | s s)`
class generates its two reachable Hermite terms in closed form, with the same
scalar expression serving values and three-axis forward derivatives. The
canonical total-order-2 `(d s | s s)`, `(p p | s s)`, and `(p s | p s)`
classes similarly retain only their at most four nonzero pair terms and form
the reachable Coulomb derivatives directly from the first three Boys values.
For forces, the kernel differentiates the pair coefficients and Gaussian
decay explicitly for participating centers, while coordinate derivatives of
the Coulomb term raise the sparse Cartesian derivative state by one. This shares
the primitive quartet's Boys values, pair centers, decay, and prefactor across
all centers without carrying a general forward-derivative scalar through the
recurrence. Total-order-3 force quartets use the same subset/Wick pair
representation with a three-bit derivative state, evaluate each raised
Coulomb derivative once per Cartesian axis, and reconstruct the fourth basis
center from translational invariance. Total-order-4 and total-order-5 force
quartets enumerate all single and disjoint double Wick contractions of their
surviving pair quanta and evaluate the required Cartesian Coulomb derivatives
through orders five and six, respectively. Orders six and above retain the
general three-component forward scalar until their dedicated derivative
recurrences land.
Total-order-3 shell pairs generate their exact 1/2/4/8 terms as subsets of
their angular quanta; quanta sharing one Cartesian axis add the required
Gaussian pair contractions without allocating recurrence arrays. The quartet
then forms its reachable third-order Coulomb derivatives directly from the
first four Boys values. Total-order-4 shell pairs use the same subset
representation with a widened derivative-state encoding. All same-axis
single Wick contractions and the three possible disjoint double contractions
are accumulated into the fixed 16-term bound, and the quartet evaluates the
reachable fourth-order Coulomb derivatives directly from the first five Boys
values. One runtime Cartesian-component path serves both
double values and the three-axis forward scalar, avoiding component-dependent
template divergence within a warp. Real spherical target AOs carry sparse,
geometry-independent Cartesian expansion terms through the same device value
and derivative consumers. Torch/JAX
bindings call the native gradient as a custom backward; they do not define the
scientific implementation.

The CPU oracle first evaluates normalized Cartesian integrals and their forward
derivatives, then applies sparse, geometry-independent real-solid-harmonic
transforms to every AO index. The d/f matrices use the CCA Cartesian order and
PySCF/libcint real-spherical order. Transforming derivative tensors with the
same matrices preserves the analytic-gradient variational relationship. CUDA
spherical execution uses the identical sparse expansions in its device kernels
and never triggers a host fallback.

## ABI stability

All public descriptors begin with `struct_size` and `abi_version`. ABI version
0 is experimental: callers can detect compatibility, but semantic stability is
not promised until a 1.0 release.

## Runtime ownership

Contexts own device selection and runtime resources. Systems and calculations
are opaque handles. Caller-provided output storage remains caller-owned, and
no hidden process-global calculation state is used.

CUDA context initialization retains compute capability, warp size, SM/thread/
block limits, register-file limits, shared-memory limits, and SM count. It also
selects one generated `KernelSet` for that device. Exact measured profiles are
preferred, compatible profiles must opt in explicitly in the manifest, and an
empty portable profile selects the generic CUDA implementation. A schedule
measured for `sm_120` is never selected on another compute capability.

## Code-generation boundaries

The scientific compiler boundary is:

```text
IntegralIR
  -> backend lowering
  -> CudaTargetInfo + CudaScheduleIR
  -> CUDA source emitter
  -> CUDA compiler/resource/device/benchmark adapters
  -> production registry
```

`IntegralIR` owns only shell structure, requested observables, independent
derivative centers, recurrence choice, precision/screening semantics, and
scientific invariants. It contains no warp, block, register, shared-memory,
compiler, or vendor fields. Backend-neutral `TargetInfo`/`TargetScheduleShape`
tests can therefore validate a synthetic subgroup width without importing a
CUDA emitter. CUDA legality, occupancy candidates, and tuning resource gates
receive an explicit `CudaTargetInfo` instead of module constants.

## Fixed-topology basis layout

CUDA plans retain contracted primitives once per physical Gaussian shell.
Ragged `system_shell_offsets`, `shell_ao_offsets`, and
`shell_primitive_offsets` describe the unique shell storage. Canonical ragged
`system_shell_pair_offsets` plus shell-pair system/first/second arrays retain
consumer topology once per fixed plan. Each target AO stores its shell index
and up to three normalized CCA Cartesian expansion terms. This follows
gpuxtb's separation of immutable topology from expanded numerical consumers
and avoids duplicating a primitive contraction for every d/f component.

The internal `inspect_rhf_cuda_basis_layout` diagnostic makes this invariant
testable without allocating a GPU plan. For the validated 18-AO s/d/f case,
four unique primitive records replace 18 component-expanded references, and
the complete device basis-topology payload, including ten shell pairs and 55
unique shell-pair-pair quartets, is 1,016 bytes. Shell angular momenta,
AO/primitive ranges, shell-pair bounds, and ragged per-system quartet offsets
are resident in the fixed plan.

## Fleet execution

`vibeqc_batch` is a persistent, non-reentrant fleet plan. It copies system
topologies during preparation, so caller system handles can be released. Each
execution optionally supplies new ragged coordinates without rebuilding basis
metadata. Compatible systems are bucketed by AO count, occupied count, and
primitive count; item results are restored to input order.

Every item has an independent status, convergence diagnostics, force buffer,
and retained AO density. A failed item leaves its previous warm state intact
and does not stop other systems. Warm densities are symmetrized and rescaled to
the correct electron trace under the new overlap before use; if the warm solve
fails, the item retries from the cold core guess.

Open-shell cold starts apply a deterministic 45-degree rotation between the
minority-spin frontier occupied and virtual core orbitals before constructing
the density. The rotation preserves electron count and orbital orthogonality
but prevents exact molecular symmetry from trapping UHF in a higher-energy
occupation fixed point, as occurs for the sigma-hole state of linear OH. Warm
densities bypass this seed and retain their converged state.

The CPU reference backend uses a bounded native worker group within each
bucket. A CUDA context builds overlap, core-Hamiltonian, ERI, and nuclear terms
on the device, uses cuSOLVER batched Jacobi eigensolves, retains all SCF matrix
and convergence state on the device, and assembles the analytic force without
a scientific host fallback. The host submits a fixed maximum iteration chain;
an active mask stops converged or failed systems while peers continue.
Python never orchestrates per-system SCF loops.

The current CUDA implementation is complete for the public executable scope
(RHF/UHF with contracted Cartesian or real spherical s-p-d-f shells), but is
not yet a component-unrolled/Rys or DF HF engine. Its SCF loop uses device DIIS
and a device-tail-launched CUDA Graph.
AO matrices up to 16 use the low-overhead serial device Jacobi kernel, sizes
17--32 use cuSOLVER's batched Jacobi provider, and larger matrices use a
Graph-native cooperative Jacobi kernel with one block per physical or spin
state. For 33--256 AOs, one 256-thread block applies disjoint
round-robin rotations as parallel cyclic sweeps, reducing diagonalization from
the maximum-pivot path's O(n^4) work to O(n^3). Larger matrices retain the
unbounded 64-thread maximum-pivot fallback. Both paths reuse the plan's
temporary matrix for eigenvectors, impose no public AO-count limit, and avoid
provider routines
that synchronize the host and invalidate stream capture above their small
batched range.
Each fixed-topology bucket owns and replays one packed arena and Graph, so warm
executions do not recreate streams, provider handles, workspaces, or graph
executables. Analytic forces are decomposed over coordinates and integral
quartets rather than serializing one complete gradient behind each coordinate
thread. The dedicated force paths through total angular order five compute all
center derivatives from one shared set of Gaussian product and Boys values,
then recover omitted centers from translational invariance; orders six and
above retain the general three-component Dual path.
The one-electron force likewise assigns one public AO pair to each warp and
accumulates overlap, kinetic, basis-center attraction, and per-nucleus
attraction derivatives directly into the stationary force contraction. One
kernel launch covers every AO pair and nucleus; there is no per-nucleus host
loop. The previous scalar AO-pair worker remains an explicit diagnostic
fallback through `VIBEQC_ONE_ELECTRON_FORCE_SCALAR`.
Coulomb auxiliary states are stored in
a four-dimensional simplex (1,820 states through f) rather than a dense 13^4
thread-local array.

The prepared plan separately caches the exact coordinate vector associated
with its geometry-derived arena state. Equal coordinates reuse overlap, core
Hamiltonian, retained small-system ERIs or direct Schwarz bounds, shell-pair
bounds, nuclear repulsion, and the orthogonalizer. Any coordinate difference
uploads positions and rebuilds all of them on the owning stream before SCF.
Density, DIIS, convergence, and force state are never part of this geometry
cache. A homogeneous valid warm-density replay also omits the
core-Hamiltonian eigenguess because the following warm-density normalization
would replace it exactly; mixed warm/cold buckets retain the common cold-guess
path so every cold member remains initialized.

J/K uses two runtime policies selected by fixed AO topology. Buckets through
16 AOs compute ERIs once per geometry and retain them in the bucket arena,
matching the persistent-cache strategy used successfully in GPUxtb. Larger
buckets allocate no N^4 tensor: a device kernel builds O(N^2) Schwarz bounds,
and a fused direct kernel screens and evaluates Coulomb/exchange integrals
inside every SCF Graph iteration. The analytic two-electron force applies the
same pair-symmetric screening decision, preserving its finite-difference
relationship to the screened energy. For s/p/d/f topologies, direct Fock decodes
only canonical shell-pair-pair and AO-pair-pair lower triangles. Each unique
eightfold-symmetric ERI is evaluated once, then its distinct permutations
scatter Coulomb and matching-spin exchange contributions into RHF or UHF Fock
matrices. The force kernel contracts the identical unique integral set and
differentiates only the at most four participating shell centers. A fixed
topology-derived tile multiplicity stripes large d/f AO-quartet groups across
one-warp blocks while retaining O(N^2) numerical storage. Each compact logical
descriptor still covers 256 quartets and expands into eight virtual subtiles,
so the finer execution grain does not multiply fixed-topology metadata. The
one-warp block is important because exact Fock and force recurrences sit near
the per-thread register ceiling: more independent blocks improve latency
hiding without the inactive lanes of a sub-warp block. Empty subtiles exit
after decoding instead of requiring a per-AO-quartet descriptor array. Double
atomics make this fast but not bitwise deterministic: a 50-replay 21-AO test
showed about 1.5e-14 Eh energy span and 1.4e-12 Eh/bohr maximum force span.
The final analytic force is not part of the iterative SCF Graph, so it need
not preserve the Graph's fixed topology-capacity launch dimensions. For total
angular orders zero through five, one device-resident atomic task head feeds a
persistent pool of one-warp blocks. The pool is capped at eight workers per
multiprocessor: this fits the specialized kernels' measured register demand
while dynamically balancing the irregular compacted AO-quartet derivatives.
The host never reads the final active count. Orders six through twelve retain
fixed-capacity grids because their general three-component derivative kernels
already use 254 registers and the additional queue state reduced, rather than
improved, throughput. Task heads are reset on the execution stream immediately
before force assembly, so prepared-plan replay retains no queue state between
calculations.
The ERI force kernel additionally uses translational invariance: derivatives
over all unique basis centers sum to zero, so it evaluates only `N-1` centers
and restores the final center from the negative sum. This halves two-center
and removes one third of three-center derivative work. Total angular orders
zero through six use dedicated all-center derivative formulas; the sparse
order-two through order-six paths evaluate each raised Coulomb state once per
axis and recover their fourth basis center by translation. Orders seven and
above use a
force-only three-component forward scalar, seeding its Cartesian axes together
and returning all three derivatives from one exact shell-class recurrence.
Both paths avoid repeating the
geometry-independent ERI value work for x, y, and z while preserving the same
screening and symmetry domain.
Canonical AO-pair arrays remain resident for one-electron triangles and
Schwarz bounds, following gpuxtb's immutable pair-metadata pattern. Their
one-electron force consumer evaluates every pair once, uses translation for
overlap/kinetic center derivatives, and distributes point-charge nuclei across
one worker warp after sharing its attraction Hermite table across both basis
centers. Each Fock
build reduces the current density to shell-pair absolute maxima before task
generation. RHF keeps the full Coulomb magnitude and its one-half exchange
magnitude separately; UHF keeps total-density Coulomb and maximum same-spin
exchange magnitudes. A device pass then compacts only shell quartets whose
shell-level Schwarz product survives both the configured threshold and at
least one of the two Coulomb or four crossed exchange density blocks. This
reduction and compaction are part of the captured SCF Graph. A single-system
bucket retains `P_n`, `F(P_n)`, its transformed direct density, and its compact
quartet list when the accepted final density step is at most `1e-12` RMS. Final
canonicalization and analytic forces reuse that consistent snapshot instead of
rebuilding Fock. After force evaluation, the already accepted `P_{n+1}` becomes
the returned warm state so subsequent replays retain the legacy convergence
branch. A looser accepted step automatically restores `P_{n+1}` before
finalization and uses the rebuild, preserving force accuracy for relaxed SCF
requests.
Multi-system buckets still rebuild from their converged
densities because peers can stop on different device-tail Graph launches and
therefore need their final compact lists regenerated together. The diagnostic
`VIBEQC_FINAL_FOCK_REBUILD=1` restores the rebuild for single-system A/B checks.
Component-unrolled/Rys quartet kernels and finer AO-level task compaction
remain subsequent scheduler work. Direct quartets
always operate on normalized Cartesian source AOs. For a public real-spherical
basis, capture-safe kernels form `D_cart = C^T D_spherical C`, run the same
exact shell-class Fock and force consumers, and restore
`F_spherical = C F_cart C^T`. This keeps DIIS, eigensolves, energies, warm
states, and returned matrices in the public AO representation while removing
repeated sparse spherical term products from the dominant ERI recurrences.

The packed arena and optional cuSOLVER workspace use CUDA's stream-ordered
memory pool. Allocation occurs before Graph capture and release is enqueued on
the owning bucket stream, so plan construction and destruction do not require
legacy device-wide `cudaMalloc`/`cudaFree` synchronization.

UHF reuses the same fixed topology and physical-system active mask while its
matrix arena stores adjacent alpha/beta states. Coulomb kernels consume
`D_alpha + D_beta`; exchange and its derivative consume only the matching spin.
The two spin commutators are concatenated into one DIIS metric per physical
system, so convergence and failure isolation remain molecular rather than
being reported as unrelated spin jobs.
