# Decision: retire xTBloom identity through native compiler consumers

Status: implemented
Date: 2026-09-22

## Problem

The embedded GFN2 snapshot still advertised an unimplemented external C API and
used xTBloom runtime identity. PR #1045 addressed identifier ownership, but that
alone leaves scientific duplication. At master `84ab4f5f`, existing compiler
H0 factor/adjoint graphs were consumed by CUDA forces while CPU H0 values and
analytic derivatives, CUDA H0 values and CPU/CUDA spin arithmetic remained
handwritten.

The imported snapshot also carried a complete standalone runtime: native
periodic Ewald and multipole sums, image topology and integrals, ALPB,
asynchronous request/plan APIs and large diagnostic snapshots. VibeQC's sole
production bridge constructs one molecule, fresh SCC, and energy/force outputs;
none of those standalone features has a public VibeQC caller.

## Decision

Move the native implementation to `src/xtb/native`, remove the imported public
C entry-point declarations/version header, and use private VibeQC descriptors.
Preserve upstream licensing, parameter identities and reference provenance.
Remove declaration-only external Context/Request owners and their unused
public workspace, context, request and DLPack descriptor records; retain native
the completion events needed by synchronous execution and the live descriptor
layout checks. Delete the unused native periodic feature tree and its CPU
SCC/force branches, ALPB, independent CUDA asynchronous request/plan entry points,
outer request graph, diagnostic snapshots, DLPack/result-arena API, and CPU batch worker/checkpoint
infrastructure. A shared private admission gate rejects retired descriptors
before any pointer staging; these cannot silently lose an energy contribution.
Compiler component kernels retain their independently tested ragged layouts.

Promote the existing H0 TensorIR emission to a host/device artifact shared by
CPU values/VJP and CUDA values/forces. Generate spin's ordered potential and
energy updates with the existing scalar emitter's explicit FMA lowering; bind
the symmetric atom-local energy to a reverse-AD witness. Runtime packing,
admission, status publication and SCC policy remain native.

## Rejected alternatives

- Rename-only retirement: changes ownership labels without removing duplicate
  equations or the stale public API declarations.
- Delete the complete runtime: removes production SCC, eigensolver, allocation
  and failure handling that are explicitly outside compiler ownership.
- Change upstream parameter manifests to match native namespaces: destroys the
  independent pinned-byte gate. Instead, reconstruct only the upstream
  namespace before comparing the original GFN1 header digest.
- Claim all native science retired: D4 SCC, representation transforms,
  and optional interaction primitives still need qualified cutovers. Periodic
  extensions and ALPB are removed because they never served the public method.

## Invariants

- Public GFN2 energy/force and restricted open-shell admission are preserved.
- No installed xTBloom library or CPU/reference fallback is introduced on CUDA.
- H0 values now use the same pair-scale multiplication order as the existing
  force IR. This can change last-bit rounding from the former value expression;
  qualification uses independent energy/force and finite-difference gates.
- Generated scalar checks reject nonfinite local H0 intermediates instead of
  publishing them. Coincident-atom derivatives keep their explicit rejection.
- Spin keeps the original atom/row/column FMA sequence; unfused products would
  change rounding and can overflow before a representable FMA cancellation.
- Native scientific ownership is counted from actual implementations, not from
  directory names. Historical Agent Notes retain the old paths.

## Evidence

The compiler and shared-native structure checks, generated spin matrix/AD/FMA
and ragged-publication tests, independent xTB/tblite endpoint goldens, force
finite differences and CUDA source ownership inventory are the acceptance
gates. See `docs/xtb_native_ownership.md` for maintained reproduction entry
points. Local test artifacts live under the worktree's ignored `.artifacts/`.

Qualification after pruning and rebasing onto master `fbaa162a`:

- 53 native CPU tests passed.
- 165 related Python tests passed in the CPU build; 18 GPU checks were skipped.
- A private-OpenBLAS wheel configuration passed 13 CPU endpoint/oracle tests
  without `LD_LIBRARY_PATH`, exercising native shim discovery.
- Seven independent tblite 0.6.0 molecular fixtures include Cl/Si d shells.
  CPU/CUDA acceptance gates are `5e-7 Eh` for energies and `5e-7 Eh/bohr`
  for forces, in addition to finite-difference and invariance checks.
- Retired descriptor tests supply unreadable pointers and require rejection
  before staging, protecting against silently dropping optional contributions.
- Compiler structure, source registry, shared-native boundaries and the current
  CUDA ownership inventory are required integrity gates.

Fixtures and their reproduction settings are in
`tests/data/gfn2_native_tblite.json`; the oracle package is not a production or
test-run dependency.

## Consequences

The native C/C++/CUDA source inventory shrinks from 120,662 to 106,780
physical lines (13,882 fewer, 11.5%). This counts actual source contents, so
renames do not contribute to the reduction. The CUDA ledger records 3,299
fewer handwritten-scientific code lines; spin's remaining 440 code lines are
reclassified as runtime after their arithmetic moves to generated TensorIR.

The code reduction removes unreachable implementations instead of moving them
into generated templates. CUDA keeps its production SCC graph, synchronous
publication/settlement and bounded fallback. There is no endpoint performance
claim: acceptance is scientific equivalence and removal of maintained code.

Runtime identity and public/private boundaries are native. Existing generated
science is reused by more consumers, and CPU H0 derivative equations and both
spin arithmetic copies are removed. This does not close #560's separate public
ragged-batch, memory/performance or CUDA wheel admission gates.

## Revisit when

The remaining native scientific terms have generated representations with
independent oracle, precision/range, derivative and complete endpoint gates.

## References

- #560: production GFN2 runtime and compiler replacement boundary.
- #1045: initial identifier-only de-bootstrap proposal.
- #953: existing compiler H0 factor and adjoint graphs reused here.
- `docs/xtb_native_ownership.md`.
