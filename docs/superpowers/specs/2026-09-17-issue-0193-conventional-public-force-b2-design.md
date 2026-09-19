# Issue 193 conventional public force B2 design

## Scope

Deliver Issue #193 B2 by promoting the complete conventional canonical MP2
gradient chain validated by #293 into a bounded native owner and publishing
analytic forces through the existing method, C, Python, and homogeneous batch
boundaries.

The supported force domain is real FP64, closed-shell, all-electron canonical
RHF-MP2 with an exact conventional four-centre correlation provider on CPU or
CUDA. The reference and correlation calculations use the same geometry, basis,
Hamiltonian, orbital frame, denominator policy, and unscreened integral model.
`force = -gradient`; coordinates are Bohr and the public force unit is
Hartree/Bohr.

This slice does not publish RI-MP2 forces, UMP2, ROHF-MP2, frozen-core MP2,
ECP MP2, complex orbitals, approximate screening, Hessians, or mixed precision.
RI energy remains supported, but an RI force request fails explicitly as C2
work rather than silently selecting conventional correlation or a different HF
reference.

## Chosen approach

Extract reusable native C++ owners from the already validated #293 workflow:

- canonical MP2 energy adjoints and orbital-energy dependence;
- streamed construction of the RHF orbital-response right-hand side;
- bounded restarted true-residual GMRES for the RHF Z-vector;
- relaxed one-electron, overlap, and two-electron MO weights;
- shell-local MO-to-AO cotangent transformation;
- CPU and CUDA one- and two-electron derivative contractions;
- nuclear-repulsion assembly and force-sign conversion.

The MP2 method adapter composes these owners with the existing
`scf::PhysicalReference`, `posthf::RawSource`, `posthf::NativeBlockProvider`,
conventional MP2 energy owner, and generated derivative consumers. Production
code does not call the Python validation facade or the exported development C
bridges.

The alternatives rejected are:

- calling `tools/vibeqc_mp2` from the public native method, which would make the
  C/C++ method depend on Python and would not establish native lifetime or
  memory accounting;
- duplicating the complete workflow inside `mp2_method.cpp`, which would turn a
  method adapter into an untestable scientific implementation and duplicate
  response/provider semantics;
- exposing the #293 small-system facade as production, which retains its
  twelve-AO ceiling and lacks a complete simultaneous resource plan;
- enabling only CUDA forces while advertising a general MP2 force capability,
  which would violate the required CPU/CUDA capability and failure evidence.

## Native ownership and interfaces

Add a post-HF conventional-gradient owner with an interface conceptually
equivalent to:

```text
prepare_conventional_force(reference, source, policy, budget, backend)
execute_conventional_force(plan) -> ConventionalForceResult
```

Preparation validates the complete model identity and constructs an immutable
resource plan. Execution owns every transient and returns one unpublished
candidate result. The result contains the correlation energy components,
gradient components, response status, resource measurements, and provider/
equation identities needed by the method boundary.

The scientific implementation is separated into testable native components:

1. a canonical MP2 adjoint owner that consumes the same occupied/virtual
   blocks and denominator policy as the energy owner;
2. an RHF response problem exposing matrix-vector products without recording
   the SCF or DIIS trajectory;
3. a reusable bounded real restarted GMRES implementation with a recomputed
   true residual, deterministic stopping status, explicit iteration/restart
   limits, and exact workspace accounting;
4. relaxed-weight assembly tied to the reference, Hamiltonian, orbital, and
   provider identities;
5. derivative consumers that stream shell quartets and one-electron shell
   pairs and accumulate directly into `3 * natom` gradient storage.

The CPU path uses the existing generated/native integral derivative machinery
through a native shell-stream consumer. The CUDA path reuses the existing
generated one-electron gradient and weighted-ERI consumers after moving their
production-safe orchestration out of development-only bridge code. Both paths
implement the same mathematical weights and component convention.

No global cache is introduced. A prepared owner is serialized by its method
owner and cannot outlive its `RawSource`, physical reference, CUDA device
selection, or prepared calculation.

## State identity and invalidation

Every force execution binds:

- geometry, nuclear charges, electron count, multiplicity, and all-electron
  model;
- orbital basis and AO-to-atom ownership;
- RHF energy, converged physical Fock, overlap, coefficients, orbital energies,
  occupied/virtual layout, and solve epoch;
- exact conventional provider and unscreened Coulomb identity;
- MP2 equation hash, denominator threshold, backend, precision, and resource
  policy.

Matching dimensions or copied hashes are insufficient. Any changed geometry,
basis, method descriptor, reference epoch, provider, backend, threshold, or
budget creates a new plan and invalidates prior result eligibility. Warm SCF
seeds never authorize reuse of a prior MP2 response or force.

Before every execution attempt, the method invalidates its correlation
diagnostic and prior result. Failed, nonconverged, stale, pending, or nonfinite
state is never eligible for publication.

## Bounded resource plan

One numeric budget covers the maximum simultaneous lifetime of the complete
energy-plus-force endpoint, not separate optimistic phase estimates. The plan
accounts for:

- the converged detached RHF physical reference retained during correlation;
- conventional provider and transformation scratch;
- occupied/virtual MP2 amplitudes and adjoint arrays retained by the selected
  blocking policy;
- orbital RHS, Z-vector, preconditioner, Arnoldi basis, Hessenberg data, true
  residual, and GMRES temporary vectors;
- relaxed one-electron, overlap, and MO two-electron weights;
- one shell-local MO-to-AO quartet cotangent tile;
- CPU or CUDA derivative-consumer staging and device allocations;
- the unpublished `3 * natom` component and total-gradient candidates;
- diagnostics and transfer staging whose lifetimes overlap execution.

Blocking and restart dimensions are selected only from checked arithmetic and
the available budget. Preparation fails with `OUT_OF_MEMORY` before scientific
execution when the minimum valid plan cannot fit. Runtime allocation is checked
against the prepared bound and a reported peak is a measured high-water mark,
not merely the estimate.

The owner never materializes a full AO rank-four cotangent, a global nuclear
derivative integral tensor, solver-history tape, or all-displacements finite-
difference state. MO-space tensors required by canonical MP2 may be retained
only when explicitly charged to the budget; AO transformation and derivative
contraction remain shell-local.

## Response and failure semantics

The native response solve is accepted only when its recomputed true residual
satisfies both the configured absolute and relative gates. It reports
converged, maximum-iteration, breakdown, nonfinite, invalid-problem, and budget
failure distinctly. A small denominator, stale identity, unsupported shell,
invalid AO ownership, failed derivative consumer, CUDA error, or nonfinite
component also fails the complete force execution.

The method constructs energy, force, convergence, and correlation/response
diagnostics in local candidates. It publishes all candidates together only
after RHF, MP2 adjoint, response, derivative contraction, component assembly,
and finiteness checks succeed. Consequently:

- force failure publishes neither forces nor a new total energy;
- no partial force component is visible;
- no diagnostic from the previous successful execution survives a later
  attempt;
- caller-owned C output buffers remain byte-for-byte unchanged on failure;
- a successful later execution does not inherit failure status or stale
  response data.

Append-only correlation diagnostic fields may record response iterations,
true residual, response workspace, derivative workspace, measured endpoint
peak, and component/provenance hashes. Existing callers using an older
`struct_size` remain ABI-compatible.

## Public method integration

`Mp2Prepared::execute(compute_forces=true)` selects the conventional force
owner only when density fitting is disabled. Energy-only execution continues
to use the existing bounded energy path without allocating gradient state.
Conventional force success returns total RHF plus MP2 energy and forces from
the same execution identity.

The method registry and Python `Calculator` expose MP2 force support after the
native gates pass. Documentation and errors state that this is conventional
canonical RHF-MP2 force support; RI force requests remain explicit
`NOT_IMPLEMENTED` until C2. Existing unsupported spin/core/ECP/precision and
screening behavior is unchanged.

## Homogeneous prepared batch

Add an MP2 prepared-batch owner following the existing HF/DFT item-isolation
contract. B2 batch support is limited to homogeneous conventional MP2 force or
energy requests in the supported domain.

Each item owns its system, RHF reference, post-HF plan, diagnostic eligibility,
and candidate output. Items share immutable method options but do not share
mutable response vectors, provider cursors, CUDA staging, diagnostics, or
publication tokens. Batch execution may schedule independent items, while
resource admission and device serialization obey the existing batch/runtime
policy.

One item failure sets only that item's status and leaves its output buffers
unchanged; successful neighbours publish normally. A changed-geometry replay
prepares a new item identity and cannot reuse a prior response. Unsupported
warm-start, profiling, RI-force, or mixed-domain flags fail explicitly instead
of being ignored. An invalid item does not poison later batch calls.

## Tests and validation

Native CPU structural and component tests cover:

- adjoint, orbital RHS, true-residual GMRES, relaxed weights, and component
  assembly against independent dense small-system oracles;
- identity mismatch, stale epoch, changed geometry, denominator, nonfinite,
  response breakdown/max-iteration, unsupported shell, and every budget edge;
- checked resource-plan arithmetic and measured peak not exceeding the plan;
- gradient/force sign and component omission/sign-reversal negative controls;
- C API output-buffer and diagnostic transactionality;
- repeated execution after success and failure;
- batch neighbour isolation, changed geometry, item-order independence, and
  unsupported-option rejection.

Scientific endpoint qualification covers H2, LiH, H2O, and at least one
f-shell case, including one system beyond the #293 twelve-AO ceiling. For each
supported CPU/CUDA route it records:

- agreement with an independent PySCF conventional MP2 analytic gradient when
  method definitions match;
- fully re-solved central finite differences at no fewer than three step sizes,
  with a stable convergence region rather than one selected step;
- translation and rotation covariance/sum-rule checks;
- changed-geometry execution and force/energy identity;
- CPU/CUDA agreement within an FP64 tolerance justified by the references;
- response residual, iteration count, memory plan, measured peak, transfers,
  provider identity, equation hash, hardware, and exact code commit;
- CUDA memcheck for the public single and batch force endpoints.

Skipped, unsupported, failed, and not-run cases are recorded separately. A
successful build, energy-only run, #293 facade result, or historical artifact
does not qualify the public force endpoint.

## Commit and PR structure

Use one reviewable B2 PR with meaningful commits:

1. this approved design and exact conventional B2 completion boundary;
2. reusable native adjoint, response, relaxed-weight, and resource-plan
   components with CPU contract tests;
3. conventional native force owner plus single-system public C/Python API and
   transactional tests;
4. homogeneous prepared batch ownership and per-item failure tests;
5. CPU/CUDA scientific evidence, resource evidence, documentation, and final
   capability promotion.

Each implementation commit must pass its relevant focused tests before push.
The PR closes only B2, links Issue #193, and states that RI production forces
remain C2. It is ready to merge only after required CI, focused review,
real-device evidence, and all public/native force gates pass.
