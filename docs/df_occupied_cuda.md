# Occupied-factor CUDA exchange and force response

`VIBEQC_DF_EXCHANGE=auto` (also the unset default) selects occupied RI-K for
the qualified resident RHF domain: one system, 768 orbital and auxiliary AOs,
occupied rank 160, full host-raw J/K scratch, and NVIDIA GeForce RTX 5090
(`sm_120`). Other domains retain dense exchange. `dense` and `occupied` remain
explicit comparison overrides. This changes execution of the existing DF
exchange and complete analytic force model without changing its approximation.

For one spin, `D = w C C^T` with canonical occupation w=2 (RHF) or w=1 (UHF).
On a full resident plan, project the existing pair-major tensor directly:
`U[mu,i,Q] = sum_nu B[mu,nu,Q] C[nu,i]`. One strided-batched GEMM builds U;
a Gram contraction over `(i,Q)` produces `K = w U U^T`. The triangular route
uses SYRK and mirrors its result; providers without SYRK use full GEMM.
Generated and bounded panels retain `U = C^T L_row^T` followed by
`K_row,column += w U_row^T U_column`. Host `OccupiedDensityFactor` snapshots
already contain `sqrt(w)*C` in row-major AO/occupied layout and use w=1 in the
CUDA product. Device SCF retains column-major C and applies w once at the
second product. Existing Fock consumers retain their RHF -1/2 and UHF -1
exchange prefactors. Empty spin rank produces zero K without a GEMM.

Dense resident K keeps square per-Q products but divides Q into half-capacity
panels. Gathered B and per-Q contributions occupy disjoint halves of one
buffer; the other holds the density projection. Raw A stays untouched. The
running auxiliary sum continues across panel boundaries in its original order;
odd auxiliary counts have a bounded tail. The `flat` ablation instead projects
`B_mu D` and contracts the combined `(nu,Q)` dimension, using one scratch
tensor. Both identities preserve nonsymmetric diagnostic B and D.
`VIBEQC_DF_RESIDENT_EXCHANGE=legacy|full|flat|auto` selects original J/K,
resident full-Gram K, resident flattened dense K, or the default panel-dense
and triangular occupied route. The plan freezes this policy; changing it
rebuilds captured SCF work.

The [split Gram decision](../.agents/notes/rejected/2026-09-17-split-occupied-gram.md)
records the endpoint qualification behind retaining this single-Gram policy.

The same plan supports resident tensors, generated panels and compatibility
host-backed tiles. Full AO panels follow #282's capacity rebalance and reuse
the diagonal T before any column replacement. Tight row traversal retains
bounded column regeneration; its generated tile trace exposes that work.
The T, column and output matrices fit the original three/four tile buffers
because occupied rank never exceeds nbf. No additional three-center tensor is
allocated. Compatibility host uploads drain before overwriting pageable
staging; generated plans stay on their existing stream and permit capture.

The native fixed-density API checks the immutable factor's exact density
witness, spin, orbital/density generations and a process-unique physical plan
identity. Rebuilding geometry, basis or metric policy creates a new plan
identity. Matching dimensions or caller-provided generation labels alone
cannot authorize use. A missing or incompatible factor runs dense K for that
item while compatible neighbors retain factorized K. Factor upload borrows
the existing density-transpose staging; no allocation is needed for host B.

Every device SCF invocation starts with one seed iteration. Imported and warm
densities have no trustworthy orbital factor, but a checked algebraic factor
can replace its dense K. `VIBEQC_DF_SEED_EXCHANGE=dense|factor|auto` controls this
choice; `factor` enables guarded factorization and `auto` selects it only in the
qualified RTX 5090 RHF domain: 768 AOs, 768 auxiliaries and 160 occupied orbitals.
Factorization requires an occupied-SCF singleton resident RHF plan
with full AO/auxiliary tiles and existing factor capacity. UHF, batch, streamed
and generated-source plans keep dense seeds.

The seed reuses the compact GPU eigensolver and transient Fock/eigenvalue
scratch to form `L = V sqrt(lambda)`, then builds occupied K with weight one.
Eigenvalues below `-1e-13` reject the seed. Values at most `1e-13` may be
discarded only if their combined Frobenius norm is at most `1e-12`; retained
rank cannot exceed the reserved occupied rank. A full `L L^T` reconstruction
must match both triangles of the input with maximum error at most `1e-12` and
RMS error at most `1e-13`. No canonical identity is assigned to this factor.
Numerical rejection returns to dense K; CUDA failures propagate. The ordinary
path adds two explicit stream synchronizations and downloads the spectrum,
solver status and two reconstruction scalars. It allocates no new persistent
device buffer. `VIBEQC_DF_SEED_VERIFY=1` additionally compares candidate and
dense K for the identical density under max/RMS gates `1e-10`/`1e-11`, restores
candidate K, and records K/Fock errors in the progress journal. This intrusive
validation must be disabled for clean endpoint timing.

`VIBEQC_DF_FINAL_EXCHANGE=dense|occupied|auto` independently controls final
physical Fock evaluation; `occupied` enables retained-factor qualification,
while `auto` uses the same qualified 768/768/160 RTX 5090 domain as the seed.
The singleton resident RHF route requires
an exact current final-state token, matching device generation/solver status,
and entry-for-entry equality of the supplied and retained densities. It uses
the full retained coefficients with RHF weight two. The strict physical Fock
and final-state gates still run; changed densities or correction generations
use dense K. No previous physical Fock is reused.

The seed iteration stores
the exact C that constructs the next D before the convergence update. Each
spin owns `batch*nbf*max_occupied` values plus generation controls. Inactive
systems retain both density and factor; eigensolver scratch is never borrowed
as persistent C. Occupation/policy changes rebuild captured GEMM shapes.
The first seed iteration counts against the original iteration limit, even
when that limit is one. Generation checks run before occupied K and at final
readback; a stale generation rejects the device result and preserves the
caller's established numerical recovery. Force evaluation receives only the
validated converged density and keeps its full metric/center/Pulay response.

Native and common resource ledgers reserve two full AO matrices for spin
factors plus generation flags for explicit occupied selection or a potential
768/768 batch-one automatic domain; actual
allocation uses the bucket's occupied ranks. Dense mode retains its previous
minimum-budget and residency boundaries. A native plan freezes this reservation
at creation and rejects occupied SCF before allocation if it reserved only dense
storage. Ordinary prepared batches rebuild the value/SCF plan on policy changes,
retaining their geometry response cache. Batches with a global `ResourceBudget`
freeze `VIBEQC_DF_EXCHANGE` in the resource identity: changing it after preparation
requires preparing a new batch and is rejected before native execution. The
fixed-density factor API needs no additional allocation and still borrows the
existing tiles independently of the SCF reservation.

`ri_k_occupied` traces report factor bytes, rank, projection/exchange products,
GEMM dimensions, intermediate bytes and panel hits. Captured records describe graph
construction; `occupied_scf_provenance` separately reports executed iteration
counts, dense seeding and final generation validation. Uninstrumented complete
endpoints remain the performance selection gate. The
[density exchange seed note](../.agents/notes/implemented/performance/2026-09-16-density-exchange-seed.md)
records the factorization, final-state audit and qualification rationale.

## Resident raw ownership and response storage

A full, single-system host-raw plan retains the original FP64 values in its
former exchange-contribution buffer as `A[Q,mu,nu]`. Setup fills this buffer
while the original raw device input is still live. The new resident K path
fits its temporaries in the two other tensors, so no extra full tensor is
allocated and both the setup and persistent reservations remain unchanged. Transformed B
alone cannot recover discarded metric directions needed by exact forces.

The prepared HF owner binds immutable raw, atom and shell allocations plus
both basis representations. These allocations survive moves into the prepared
cache. A force call must match these bindings before receiving a
`CudaDfRawTensorView`: pointer, dimensions, strides, process-unique owner
identity, and the original metric eigensystem/cutoff. The view always denotes
untruncated raw values, never a streamed panel or transformed B. Rebuilding
geometry, either basis, or the metric owner invalidates the previous view.
Standalone tensor-plan callers have no immutable source binding and upload
through the compatibility adapter.

J/K and response share one stream. Two full buffers become mutable response
scratch; the retained raw buffer remains read-only until the bridge drains.
Matching warm resident calls perform zero raw-tensor H2D copies or transposes.
`VIBEQC_DF_RAW_REUSE=off` retains the upload ablation. An upload revokes raw
validity before submission and restores it only after successful response
from the matching immutable source, including failure/retry handling.

`VIBEQC_DF_RESPONSE_STORAGE=auto` borrows this capacity for the qualified
384/384 and 768/768 batch-one RHF shell/BLAS endpoints on RTX 5090. The
384 case keeps dense response algebra while evaluating its expensive all-Q
projections only once. `panel` preserves bounded execution; `jk-scratch`
requests validated borrowing explicitly. Batch and source-backed plans retain
their existing bounded/upload contracts. Borrowed capacity is reported once
alongside owned scratch and transfers; reuse is never inferred from dimensions
or a small component-local budget alone.

## Exact occupied force response

`VIBEQC_DF_RESPONSE_SPACE=auto` (also unset) selects occupied response in the
same measured 768/768 RHF rank-160 device domain, with automatic resident
storage and the default generated shell schedule. Explicit panel storage and
diagnostic schedules preserve their original route. `dense` retains the full-AO
comparison; `occupied` requests factor validation on compatible resident plans.
The 384-AO borrowed path and smaller systems keep dense automatic response.

The method passes its verified final-state token. The response owner checks
source identity, solve epoch, system, model, occupations, exact canonical device
density, and each device factor generation before borrowing C. Missing/stale
tokens, corrected determinants, external densities, unreserved plans and
unsupported factors keep dense response. UHF additionally verifies the exact
sum of its spin densities and admits both rank-squared projections together.

The response computes `T_Q=C^T A_Q C` and `U_P=sum_Q V_PQ T_Q` from raw
three-center values, preserving finite discarded metric directions. In the
qualified domain it feeds at most 64 auxiliary slices of packed symmetric
AO-pair weights to the existing generated derivative consumers. It does not
retain a full response-weight tensor. Projections, transformed projections and raw
values occupy the three already charged resident J/K tensors; the consumed
projection buffer becomes panel storage. `VIBEQC_DF_RESPONSE_BATCHING=auto`
concatenates Q slices for a large `C^T [A_0 ... A_(a-1)]` GEMM and batches the
second multiplication by C. Previously retained spin factors remain outside
that staging range. Pseudo-density expansion batches `C U_P` and lower
rectangular products across each bounded auxiliary panel. Capacity checks
include weights, projected factors and rectangular outputs simultaneously;
`off` and insufficient capacity preserve serial projections. The algebraic
FLOP count stays fixed while BLAS submissions and repeated factor reads fall.
Additional response workspace contains
four auxiliary matrices, three AO matrices, densities and auxiliary charges.
`DfGradientResources` reports the executed route and borrowed capacity.

`VIBEQC_DF_DERIVATIVE_PAIRS=auto` selects packed weights for the qualified
768/768-AO, rank-160 RHF occupied response on RTX 5090. The previously promoted
resident 192--384-AO shell route instead folds the existing dense weights,
preserving its response producer. Other automatic routes retain their original
execution. Explicit `full` executes the ordered dense shell product;
`symmetric` adds the two dense off-diagonal shell weights and executes each
unordered shell pair once. `packed` requests packed production when a trusted
occupied response and complete generated shell consumer are available, otherwise
retaining the dense producer and symmetric shell consumer. Generic and host
consumers retain their dense contracts. These controls are captured in resource
plan identity and cannot be changed inside a frozen global resource plan.

Packed weights store `W_ii` once and `W_ij+W_ji` at `i*(i+1)/2+j` for `i>j`.
Diagonal-shell AO pairs and normalized spherical/Cartesian expansions preserve
their physical atom derivatives. Auxiliary panels end at whole shells whenever
the cap permits; a smaller explicit cap can still split a shell. Generated
derivatives reduce directly into the atomic gradient, without a derivative
tensor or a second derivative formula implementation.

The producer expands only lower rectangular AO blocks, with 256 rows by
default. `VIBEQC_DF_PACKED_AO_BLOCK_ROWS=64|128|256|384` retains diagnostic
alternatives. Upper off-diagonal blocks are skipped; unused upper entries
inside diagonal blocks are counted as executed work. Counters distinguish
shell pairs/triples, public weight loads including zeros, primitive products,
nonzero Cartesian contractions, rectangular GEMM entries and panel bytes.
`VIBEQC_DF_SHELL_COUNTERS=1` enables diagnostic device atomics and must be
disabled for clean timings.

Packing halves the weight handoff, not the complete resident plan allocation.
Raw uploads, validated raw reuse and borrowed J/K capacity are reported
separately; packed weights do not imply a smaller persistent value plan.
Tiny explicit domains can fit in one panel or one AO block; counters report
their actual dense and packed materialization rather than implying a saving.

The [derivation and lifetime note](../.agents/notes/implemented/performance/2026-09-15-occupied-df-response.md)
records RHF/UHF coefficients, metric response and rejected schedules.
The [qualification evidence](../benchmarks/results/issue377-379-df/README.md)
retains frozen-density policy comparisons, independent strict force gates,
component/work counters, reservation and priming costs. The automatic selector
is deliberately limited to that measured domain; it establishes no TZ/QZ,
other-device or COSX crossover. Memory diagnostics report charged capacity,
not a measured global GPU peak.

The [packed derivative note](../.agents/notes/implemented/performance/2026-09-15-packed-df-derivative-pairs.md)
documents symmetry, diagonal-shell treatment, the block-size tradeoff and
[current qualification](../benchmarks/results/issue382-packed-df/README.md).

## Compact SCF DIIS and solver timing

`VIBEQC_DF_DIIS_DOTS=auto` forms deterministic partial residual dots in blocks
of 4096 elements, then reduces the partials in the existing small DIIS solve.
The completed residual-product temporary supplies the partial storage. No
allocation, atomic dot accumulation, or host synchronization is added.
Physical slots are validated against the chronological circular history,
including a short window after dependent-history retirement. Inactive systems
and unpopulated entries remain untouched. Small reservations fall back to a
warp reduction; `serial` retains the original dot order for comparison.

Normalization, pivot threshold, chronological retirement, Fock construction
and physical convergence gates are unchanged. Floating-point reduction order
can change the iteration branch, so endpoint evidence must retain both the
starting checkpoint identity and actual update counts. The diagnostic policy
is frozen with captured work and all response/exchange controls participate
in global resource-plan identity.

Compact eigensolves keep the existing cuSOLVER provider and retained host and
device workspaces. `compact_diis`, `diis_residual_products`,
`diis_history_update` and `compact_eigensolve_provider` expose their separate
GPU/host scopes. A provider host call can wait for preceding queued DIIS work;
subtracting its GPU events from host duration does not measure CPU eigensolver
work. Component tracing introduces fences and must be qualified with separate
host-only or Nsight timelines and clean complete endpoints.

The [resident DF dataflow note](../.agents/notes/implemented/performance/2026-09-16-resident-df-dataflow.md)
records the algebra, rejected variants, numerical gates and retained evidence
for these paths.
