# Architecture decisions

## HF-first boundary

RHF and UHF are executable in the HF vertical prototype. `WB97M_V` and
`RCCSD_T` have stable method identifiers for capability discovery but return
`QCE_STATUS_NOT_IMPLEMENTED`. No DFT grid or coupled-cluster tensor framework
is created before a real method requires it.

## Derivative policy

Nuclear forces are native analytic derivatives. The current reference engine
uses forward derivative values inside the primitive integral formulas, then
assembles the stationary RHF derivative from derivative integrals, the
energy-weighted density matrix, and nuclear repulsion. This is deliberately
different from differentiating the eigensolver, DIIS, or SCF iteration trace.

The CUDA s-p-d-f path evaluates one-coordinate dual forms directly in device
kernels and contracts stationary RHF/UHF gradients on the GPU. Its Cartesian
McMurchie-Davidson recurrence is shared mathematically with the CPU oracle but
implemented independently in CUDA. Real spherical target AOs carry sparse,
geometry-independent Cartesian expansion terms through the same device value
and derivative consumers. Torch/JAX bindings call the native gradient as a
custom backward; they do not define the scientific implementation.

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

`qce_batch` is a persistent, non-reentrant fleet plan. It copies system
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
Graph-native cooperative Jacobi kernel with one 64-thread block per physical
or spin state. The large path reuses the plan's temporary matrix for
eigenvectors, has no compile-time AO-count limit, and avoids provider routines
that synchronize the host and invalidate stream capture above their small
batched range.
Each fixed-topology bucket owns and replays one packed arena and Graph, so warm
executions do not recreate streams, provider handles, workspaces, or graph
executables. Analytic forces are decomposed
over coordinates and integral quartets rather than serializing one complete
gradient behind each coordinate thread. Coulomb auxiliary states are stored in
a four-dimensional simplex (1,820 states through f) rather than a dense 13^4
thread-local array.

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
blocks while retaining O(N^2) numerical storage; empty tiles exit after
decoding instead of requiring an O(N^4) descriptor array. Double
atomics make this fast but not bitwise deterministic: a 50-replay 21-AO test
showed about 1.5e-14 Eh energy span and 1.4e-12 Eh/bohr maximum force span.
The ERI force kernel additionally uses translational invariance: derivatives
over all unique basis centers sum to zero, so it evaluates only `N-1` centers
and restores the final center from the negative sum. This halves two-center
and removes one third of three-center Dual-integral work.
Canonical AO-pair arrays remain
resident for one-electron triangles and Schwarz bounds,
following gpuxtb's immutable pair-metadata pattern. A geometry-dependent
device pass compacts shell quartets whose shell-level Schwarz product survives
the current threshold; the SCF Graph and analytic-force pass consume the same
list without host readback. Component-unrolled/Rys quartet kernels and finer
AO-level task compaction remain subsequent scheduler work. The exact
shell-class evaluator contracts every sparse real-spherical Cartesian term
within the same symmetry-reduced task, so pure d/f targets do not fall back to
the generic matrix-element direct kernel.

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
