# Scientific compiler ownership

`python/vibeqc_compiler` is the importable scientific compilation subsystem.
It is installed beside `vibeqc`, whose public molecular API and native-library
selection remain separate. `tools/generate_*.py` and benchmark/reproduction
commands are clients of these packages. Source generation does not import the
public runtime, load `libvibeqc`, probe a GPU, or import PySCF, Torch or CuPy.
NumPy remains the existing dependency for recurrence/reference arithmetic.

| Owner | Responsibility | Allowed compiler dependencies |
| --- | --- | --- |
| `integral` | IntegralIR, scalar algebra, recurrence lowering, integral schedules and promotion | `common` |
| `tensor` | TensorIR, AD, optimization, planning, tensor CUDA emission/execution | `common` |
| `array_api` | Bounded symbolic array frontend that lowers ordinary array expressions directly to TensorIR; no method/runtime policy | `tensor` |
| `dft` | Discrete grids, AO jets, density ingredients, prepared tile execution | `common`; `ao_cuda` alone also uses the existing scalar `integral.expr` and `integral.cuda` |
| `xc` | Audited functional expressions, derivatives, point coefficients and XC execution | `common`, `integral`, `dft` |
| `method` | Canonical MethodIR, stationary-gradient source plans and implicit-solve derivative rules; no solver/runtime policy | `common`, `xc`, `tensor` |
| `common` | Backend/target contracts, finite compiler processes, artifacts, hashes, resources and evidence | none of the scientific or user-runtime packages |

The compiler owns mathematical IR and lowering. The compiler-internal
[array frontend](array_api_frontend.md) sits strictly above TensorIR: it may
construct ordinary TensorIR nodes/programs, but TensorIR does not import the
frontend and no frontend-only node survives lowering. It retains TensorIR's
index-space, representation, symmetry, exact-coefficient and AD semantics
rather than replacing them with shape-only array semantics. The architectural
rationale is recorded in the
[Array API frontend note](../../.agents/notes/implemented/architecture/2026-09-20-array-api-tensorir-frontend.md).

`method` is the composition front end above XC/TensorIR; representability
there does not imply runtime support.
The [implicit-response primitive](implicit_response.md) emits ordinary TensorIR
JVP/VJP/RHS/source programs without importing the runtime solver.
`src/integrals`, `src/tensor` and `src/dft` own the corresponding native interfaces, runtime allocation and
execution templates; method and SCF code consume these interfaces. A compiler
package move does not promote a new scientific capability or retire a native
fallback. Native scientific ownership and retirement are tracked by #231.

`dft.ao.NativeAO` is an explicit adapter to the existing normalized native
basis ABI. Its runtime imports occur only during preparation. Likewise,
`dft.grid.owned_atoms` uses the public Atom conversion only when accepting
molecular input, and the fixture adapter constructs public shell records only
when requested. These three narrow exceptions are enumerated by the dependency
check. No SCF policy belongs in generic compiler code.

AO CUDA lowering borrows the same scalar graph and emitter as XC. Those two
neutral modules retain their historical IntegralIR paths; the dependency check
allows only these exact imports from `dft.ao_cuda`, without permitting DFT to
depend on integral recurrence or method scheduling. This avoids a second
scientific algebra implementation.

## Lowering and tuning modules

`integral.cuda_lowering` and `integral.cuda_emitter` retain callable facades.
The implementation resides in `integral.lowering`:

- `selection`, `algebra` and `common` handle validation, structured expression
  emission and shared CUDA support.
- `fock`, `fock_component` and `fock_tiled` own Fock consumer families.
- `force_packed`, `force_rys_thread`, `force_rys_component`,
  `force_rys_uniform`, `force_resident` and `force_subgroup` own force schedules.
- `dispatch` assembles the common kernel envelope and selects consumers;
  `legacy` retains explicit compatibility entry points.

Consumer modules depend on shared emission/algebra helpers. Shared helpers do
not call back into dispatch. Recurrence mathematics is still represented by
the existing IR and scalar graph; splitting files introduces no new equations.

`integral.autotune` delegates to `integral.tuning`. `analysis` estimates
candidate structure, `policy` enumerates/promotes schedules, `emission` writes
candidate source, `manifest` serializes promotion records, `resources` applies
resource gates, `process` handles external processes, `inputs` normalizes input,
`driver` coordinates a run, and `cli` parses arguments. After the caller's
candidate bound, policy groups packed schedules that differ only in algebra
ordering; emission hashes unsuffixed generated CUDA only for those groups and
removes byte-identical no-op variants while retaining protected
production/resource baselines. This is compile-work deduplication, never a
profitability or promotion decision. The type-only reference from analysis to
policy is guarded by `TYPE_CHECKING`.

Generic CUDA targets, compilation, parsed resource records, artifact handles,
metrics and preparation synchronization are owned by `common`. Static
`CudaTargetInfo` catalog entries contain architecture-level limits only; concrete
device topology such as SM count is unknown until a runtime probe enriches the
target. Generic scheduling APIs require an explicit target or architecture
instead of silently selecting `sm_120`. Measured production and local profiles
may still record device topology as qualification provenance. The rationale is
retained in
`.agents/notes/implemented/compatibility/2026-09-20-explicit-cuda-target-topology.md`.

Integral and TensorIR execution no longer depend on one another's runtime
classes. DFT and XC use that same artifact cache and allocation lock. The public
global resource planner and local-profile hashing/atomic-JSON helpers are re-exported
from their original `vibeqc` APIs; their implementations are not duplicated. Shared evidence
and timing helpers do not import benchmark command modules.

## ProgramIR storage overlays

`common.program_storage` binds physical alias, effect and ownership-transfer
facts to an immutable `ProgramIR` without moving allocation policy into the
scientific graph. Donation is explicit and fail-closed: the donor must be a
compiler-owned physical owner at its final use, the recipient must be produced
by the same call with equal capacity in the same memory space, and opaque
provider effects cannot participate.

The packed polarized PBE CPU XC tile is the first native provider consumer of
this ownership-transfer contract. Its `vxc` call donates the complete scalar-XC
row owner to the equal-sized compact coefficient output. The generated native
coefficient loop preloads every feature-gradient scalar before writing any
coefficient row, so the declared overlap is alias-safe. Unsupported layouts,
including current LDA/meta-GGA row-count mismatches, keep separate owners rather
than silently forcing an in-place path.

## Workload specialization contract

`common.specialization` is the pure, backend-neutral selection contract. It is
not wired into production DF or benchmark dispatch yet; those adapters remain
separate work. The module contains no measured thresholds or method policy.

- `WorkloadSignature` contains a consumer kind and named scalar workload facts.
  `TargetCapabilities` wraps the existing `TargetInfo` plus explicit capability
  and resource facts. Unknown facts are omitted, not guessed; product names,
  UUIDs and benchmark identities stay in the existing provenance records.
- `CompilationIdentity` references the existing scientific and compiler hashes.
  The compiler identity owner must include relevant source, IR/generator/ABI
  versions, toolchain and compile options. `ImplementationProfile` references
  the original artifact key, schedule hash and versioned tuning-profile hash.
- A `SpecializationGuard` is a conjunction of declarative equality or inclusive
  lower/upper bounds. Equality distinguishes booleans, integers and floats;
  bounds accept finite numbers, not truthy strings or booleans. Missing facts
  fail the predicate. Input pairs are copied, sorted and kept immutable.
- Correctness and performance guards are independent. A missing performance
  guard means **not promoted**. A present guard asserts a qualification supplied
  by the existing evidence owner; the selection module does not create evidence
  or relax numerical/resource gates. Even a matching performance guard cannot
  bypass correctness or scientific/compiler identity checks.

`select_specialization(workload=..., target=..., identity=..., profiles=...,
fallback=...)` chooses the first eligible promoted implementation in caller
priority order. Callers that require a promoted implementation pass
`fallback=None`; a miss is then `unsupported` instead of a silent downgrade. If
an explicit fallback is supplied, it must pass its own identity and correctness
checks. `unsupported` means no supplied implementation is eligible, not proof
that the mathematical method is impossible. There is no implicit CPU execution
or compilation on a miss.

The result exposes the original selected artifact and a detached JSON diagnostic
record with profile/schedule/scientific/compiler identities and separate rejection
reasons. `selection_key` uses the existing canonical hash over the versioned
request, ordered profiles, guards and fallback. It changes after relevant
workload, capability, identity, schedule, profile or priority changes, while
nearby workloads can still select the same artifact. Persist the input records
with the decision when retaining evidence. **The selection key is not an
executable cache key**: existing loaders retain compatibility, ownership and
binary-integrity checks and may reject an artifact selected from stale metadata.
No cache layout or existing hash algorithm is replaced.

CPU-only contract tests are in `tests/python/test_specialization.py`. Synthetic
guard domains are not CUDA qualification or performance evidence. Rationale and
consumer migration boundaries are recorded in the
[specialization contract note](../../.agents/notes/implemented/architecture/2026-09-19-specialization-contract.md).

## Checkout and installed usage

With the existing NumPy dependency available, CMake can run the generator
scripts directly from an uninstalled checkout. Each script bootstraps the
explicit `python/` package root; compiler libraries never manipulate `sys.path`.
CMake and `vibeqc.autotune.source_identity` expand the same
`cmake/VibeQCSourceIdentity.json` inventory. CMake retains `CONFIGURE_DEPENDS`
for recursive groups so adding or removing a covered source reconfigures the
build before the compatibility hash is reused.

Generated-source invalidation is intentionally narrower than that complete
compatibility identity. The shared codegen runner records repository-local
Python modules actually loaded by each successful generator invocation and emits
a Make/Ninja depfile. Explicit manifests, parameter files and other non-imported
inputs remain ordinary CMake dependencies. An unrelated compiler edit can
therefore refresh the complete source identity without forcing every generated
family to run again.

The wheel includes the integral manifests, required native templates and their
transitive local headers, plus the audited Libxc source and license provenance from
`upstream/libxc/7.0.0`. The generated compatibility manifests retained
under `manifests/libxc/7.0.0` remain package assets; the source snapshots are
included in the sdist for offline regeneration.
`pyproject.toml` configures scikit-build-core to package the CMake-installed native
library and copy canonical JIT inputs through `wheel.force-include`, without
importing either Python package; there is no second editable native source tree.
`common.paths` resolves each input by its stable repository-relative name in a
checkout or wheel. Independent numerical fixtures remain checkout inputs.

Ordinary user calculations do not import tuning/reference dependencies. User
autotuning still requires a matching source checkout, CUDA toolkit and the
existing optional PySCF validation dependency. A wheel may drive a byte-identical
checkout: the complete loaded compiler inventory must match before tuning.
An already imported, different compiler is rejected instead of loading a second
set of IR classes via path manipulation.

## Identities and compatibility

IntegralIR/TensorIR mathematical serialization and equation hashes, schedules,
and generated integral source are unchanged by the move. Source compatibility inventories necessarily
change because they now hash the first-class source paths and all nested leaves.
Tensor artifact and XC emission contracts explicitly use layout version **2**.
XC's expression hash already includes the exact expression-module bytes, so
its import changes intentionally alter provenance. The emitted source includes
the versioned contract identity; changing it does not change the scalar graph
or arithmetic. Across all seven functionals, both spin modes, orders 0–2 and
three schedules, 126 emitted XC variants differ only on the identity line.
Twelve TensorIR example/schedule sources remain exactly identical. Both installed and checkout compiler
inventories use the same logical paths, without absolute installation paths,
timestamps, bytecode or compatibility shim bytes. Old artifacts must be rebuilt;
binary content verification and numerical promotion gates are unchanged.


The compiler now has a single import surface under python/vibeqc_compiler.
The former tools/vibeqc_codegen, tools/vibeqc_tensor, tools/vibeqc_xc and
tools/vibeqc_dft forwarding packages were removed rather than retained as
compatibility aliases. Repository generators and tests import canonical owners
directly; generated-artifact identities therefore contain no shim bytes.

## Structural verification

Run `python tools/check_compiler_structure.py` to check import directions, or
add `--json` for a module-size/dependency inventory. The same check is a
pre-commit hook. Tests also import every compiler module in a fresh process
that rejects runtime/reference imports, check legacy module identity, and run
an uninstalled generator from an unrelated working directory.

The decomposition/package-migration rationale, historical module sizes,
byte-identical generation evidence, timing observations, and regression counts
are preserved in the
[compiler package ownership note](../../.agents/notes/implemented/architecture/2026-09-15-compiler-package-ownership.md).
Those measurements are migration evidence rather than a current runtime
performance claim. Raw run logs and generated build products belong in ignored
`.artifacts/`, according to the [evidence retention policy](../maintainer/evidence_retention.md).


## Stationary semilocal gradient plans

`vibeqc_compiler.method.StationaryGradientPlan` combines a resolved semilocal
MethodIR with an explicit `StationaryMeanField` envelope. The envelope declares
all-electron/direct full-range Coulomb, fixed integer occupations, real FP64,
the XC point model and a stable differentiable grid branch. An XC graph alone
cannot supply the Hamiltonian or overlap constraint.

The implemented compiler/diagnostic slice provides:

- bounded ordered-element one-electron, Coulomb and overlap/Pulay objectives;
  TensorIR VJP generates their source weights with an exact unit seed;
- generated contractions with provider-supplied derivative tiles, without a
  global four-AO-index cotangent; restricted densities already include occupation
  two, while unrestricted Coulomb uses the total alpha/beta density;
- a strict component reduction requiring AO-center XC, physical grid motion,
  partition-weight response, one-electron, J, Pulay and nuclear sources once each.
  XC functional coefficients are applied upstream, not again in the reduction.

`integral_block(source, terms=..., coordinates=...)` takes **full ordered**
AO-pair/quartet tuple data. It does not infer packed symmetry multiplicities or
atom mappings. Its `objective`, `weights` and `contraction` are ordinary TensorIR
programs; the existing CPU interpreter and CUDA planner/emitter consume the same
equations. CUDA source generation is not hardware execution qualification.
`reduce_diagnostic` checks complete input coverage, shapes, FP64, finite values
and the interpreter's logical byte budget; it is not a public force endpoint.

Plan identity includes method/envelope semantics, not descriptive aliases or
live solve epochs. Block identity additionally includes generated equation hashes
and shapes. The existing CUDA planner owns schedule/target identities. Native
state leases remain runtime-owned and must be checked before and after execution;
this compile-time plan neither creates nor renews a lease.

**Still unavailable in this slice:** live native gradient/provider binding,
complete CPU/CUDA molecular gradients, native XC/grid derivative integration and
public DFT forces. `require_native_endpoint` rejects both backends explicitly.
The existing `StationaryDerivativeContract` gates remain unchanged; detached or
stale arrays cannot gain force capability by constructing a plan. Hybrid, ECP,
DF, meta-GGA and unsupported occupation requests do not inherit this capability.

Tests: `tests/python/test_stationary_gradient_plan.py`. Rationale:
[generated stationary source composition](../../.agents/notes/implemented/architecture/2026-09-19-stationary-gradient-plan.md).
