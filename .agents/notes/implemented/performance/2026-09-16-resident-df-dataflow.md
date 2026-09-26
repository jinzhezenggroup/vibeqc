# Decision: retain raw DF values and reuse derivative and response work

Status: implemented
Date: 2026-09-16

## Problem

Issues #388–#391 concern four costs remaining after #386: the complete raw
three-center upload before resident forces, duplicated inner derivative
polynomial work, serial occupied-response products, and resident occupied K
plus a misleading compact-eigensolver host interval. These costs need separate
ablations because response storage, contraction order and convergence interact.

## Ownership and lifetime

The prepared HF owner already holds immutable host raw values and model data.
The CUDA value plan transforms a device raw input into `B[mu,nu,Q]`, then
previously discards the original device raw input. Dense K's gathered tensor,
density projection and per-Q output consumed all three reserved tensor
temporaries. Thus the old response API was both an ownership boundary and an
actual lifetime/layout mismatch, rather than merely a redundant copy call.

Dense K now divides Q into half-capacity panels. Gathered B and per-Q outputs
occupy disjoint halves of one tensor; a second contains the density projection.
The running reduction continues across panel boundaries in the original Q
order. The third allocation retains untruncated `A[Q,mu,nu]` while the original
setup input is still live. Neither setup peak nor persistent tensor capacity
increases. Occupied K needs only its projected factor temporary.

`CudaDfRawTensorView` records the immutable pointer, dimensions and strides,
process-unique owner identity, and forward metric eigensystem/cutoff. The
prepared owner also binds the original raw, atom and shell allocations and
both basis representations. These allocations survive moves into the prepared
cache. Equal dimensions or recycled addresses alone do not authorize a view
from another owner. Geometry, basis and metric changes construct a new owner;
standalone tensor-plan callers do not bind this immutable source contract.

The same stream orders setup, SCF, response and the next SCF replay. Response
borrows two mutable buffers and the immutable raw buffer, and drains before
releasing them. An explicit upload ablation revokes validity before writing;
only successful response from the matching source restores it. Raw includes
discarded metric directions: reconstructing it from transformed B would change
the exact spectral derivative. Batch, generated/source and constrained-memory
plans keep their explicit upload/panel contracts.

## Matrix work ledger

Let n be orbital AOs, a auxiliary AOs, r occupied rank, `D=w C C^T`, and
`V=M+` the same retained metric inverse. The qualified 768 case has
`n=a=768`, `r=160`, and one RHF spin term.

| Consumer | Previous organization | Resident organization |
| --- | --- | --- |
| Occupied K projection | Gather Q-major B, then `(r,n,n; batch=a)` GEMM | Direct B projection `(a,r,n; batch=n)` into `U[mu,i,Q]` |
| Occupied K output | `(n,n,r; batch=a)` GEMM, full per-Q outputs, reduction | Gram product over `(i,Q)`, `(n,n,a*r)`, optionally triangular SYRK plus mirror |
| Response projection `T_Q=C^T A_Q C` | Two products per Q: `2a=1536` | Concatenate first products, then batch second: `1+a=769`, two BLAS calls |
| Metric adjoint | Contract all `T_Q:T_P` | Same contraction and spectral reverse map |
| Weight transform `U_P=sum_Q V_PQ T_Q` | All-auxiliary contraction | Same contraction; its output remains occupied-space |
| Packed pseudo-density | `C U_P` then three lower rectangular AO blocks per P: `4a=3072` | One concatenated CU per shell-aligned panel plus batched blocks: `13+3a=2317`, 52 BLAS calls |

The legacy headline K already uses one full AO/Q panel: one `panel()` and one
`transform()` call, with a diagonal projected-panel cache hit. Repeated column
projection is a constrained-row issue, not the cause of the resident headline.
The new K route removes the full gather/output/reduction traversal and moves
auxiliary reduction into BLAS. Projection work is unchanged; triangular Gram
work is `a*r*n*(n+1)` FLOPs versus `2*a*r*n*n` for full output. SYRK eligibility
follows the Gram identity even for nonsymmetric diagnostic B.

The complete fresh ledger also includes two mandatory dense K builds, about
1.71 s at 768 AOs, in addition to the two occupied builds. Dense seeding and
final physical-state reconstruction dominate this SCF ledger; the initial
issue snapshot's occupied-K emphasis must not hide them. RI-J remains about
18 ms. An imported density has no trusted orbital factor, so skipping its
dense seed would violate the existing state contract. Full Gram and SYRK have
similar measured times here; the measured occupied-K saving primarily comes
from the new layout and removed gather/per-Q output traversal, not a claimed
twofold SYRK speedup.

Response projection remains 175,154,135,040 FLOPs; packed pseudo-density
expansion remains 126,835,752,960 FLOPs. The improvement is batching and data
reuse, not a low-rank approximation or fewer required contractions. Projection
staging uses 754,974,720 bytes within a previously reserved tensor. Previously
retained spin U factors are excluded from that staging range. The consumed T
buffer later holds disjoint bounded W, CU and rectangular output panels; their
combined live capacity is checked before batching. Full output preserves its
orientation; packed output symmetrizes the exact occupied adjoint before
folding. Inadequate capacity retains the serial path.

The 384 automatic policy also borrows resident response capacity, retaining
dense response algebra. Its all-Q products execute once rather than once per
consumer panel (`6144 -> 768` products). Whole-domain shell consumption avoids
some auxiliary-shell panel revisits, so primitive counters decrease slightly;
this is removed duplicate work, not additional screening.

## Generated derivative algebra

The compiler owns both changes in `integral/df_shell_derivatives.py`:

1. For each differentiated axis, form
   `H_i=sum_jk v_j*w_k*F_(i+j+k)` once for both orbital centers. Geometry, Boys
   evaluation and the scalar moment DAG were already shared per primitive.
2. Derive the raised-B moment from
   `M_(x,y+1,z)=M_(x+1,y,z)+(A-B)*M_(x,y,z)`. Base coefficients above their
   polynomial degree are zero. Remove the raised-B cache boundary.

The per-axis cache changes from
`(A+2)(B+2)(C+1)(A+B+C+4)/2` doubles to
`(A+2)(B+1)(C+1)(A+B+C+3)/2`. The old doubly raised boundary was allocated but
never evaluated; the work ledger distinguishes capacity from evaluated
polynomials. Across the measured 768 primitive histogram, repeated other-axis
loop-body products decrease from 46,587,727,872 to 23,293,863,936. These are exact
generated loop counts before compiler optimization, not measured instructions
or global-memory transactions.

GPU4PySCF's `sum_ejk_int3c2e_ip1_kernel` at `db6bb7f` computes Rys roots once per
primitive, shares xy/xz/yz products between centers, and uses the same horizontal
center identity. VibeQC retains its independent Hermite/Boys polynomial
formulation. Its general coefficient convolution is structurally more work
than root-wise products, particularly as angular degree rises. Sharing H and
shrinking the moment cache address that difference without copying scientific
CUDA or introducing a second recurrence implementation.

Public spherical weights already contract into compact Cartesian shell
components before primitive derivatives. Moving that expansion is not the
identified duplication. Legal AO-pair symmetry is applied before primitive
work; full rectangular mode remains ordered. Both independent centers and all
three axes share the same primitive data, with the auxiliary derivative
recovered by translation. Gradient accumulation and signature packet scheduling
are unchanged. The 768 work domain remains 28,385,280 shell triples,
175,132,672 primitive products and 226,787,328 public derivative weights.
The weight handoff proxy is 1,814,298,624 bytes; it is not a measured DRAM count.

The three derivative libraries were built with identical Release/sm_120 flags,
without line information: legacy 203,146,168 bytes, shared-H 198,378,424 bytes,
and shared-H/center-identity 198,255,544 bytes. Corresponding derivative objects
are 75,736,928, 70,971,104 and 70,850,064 bytes. The historical #386 baseline was
built with `-lineinfo`; its much larger library is not a valid binary-size
comparison. No additional specialization family or launch explosion is added.
The smaller cache does not improve every class's register limit. For example,
the measured SSS class increases its theoretical resident thread limit from
512 to 640, while PPP falls from 512 to 384 as register use rises from 128 to
140. Endpoints, rather than a uniform occupancy claim, select the combined
change.

## DIIS and eigensolver ownership

The old warm DIIS assigns one lane to each Gram entry; a small history leaves
most lanes idle while each occupied lane traverses an entire residual vector.
Auto now forms deterministic block partials over 4096 elements, then reduces
those partials in the existing small solve. Completed residual-product scratch
supplies storage, with explicit capacity/grid checks and a warp fallback. No
allocation or host wait is introduced. Normalization, the `1e-14` pivot gate,
chronological history retirement and the Fock proposal remain unchanged.

Retired histories may occupy a short window that wraps around physical slots.
Testing `slot < count` was incorrect; the partial kernel validates the circular
live window and substitutes the current residual at the pending head. An
independent long-double regression poisons unused slots and checks RHF/UHF,
wrapped, retired, inactive and ragged histories. The earlier faulty candidate
is excluded from promotion evidence.

The semantic ownership ledger counts both DIIS numerical kernels as scientific:
history selection and Gram assembly are method-specific even when their inner
sum is a reduction. The update region also includes the existing normalized
solve, dependent-history retirement and Fock extrapolation; wrappers remain
runtime. Of the resulting +300 scientific / -113 runtime line deltas against
`b29649f`, 151 unchanged lines move categories. The retained ownership record
separates this correction from physical edits (+172/-23 scientific and +61/-23
runtime). No numerical code is retired by reclassification. A generated DIIS
replacement needs the same independent history, capacity, convergence and full
endpoint gates before the native regions can be removed.

cuSOLVER selection and retained workspaces do not change. Host-only provider
regions and separate Nsight captures distinguish queued DIIS work waited on
inside the provider from actual eigensolver GPU work. Component tracing adds
fences: it can move the wait into the preceding DIIS region and must not by
itself be interpreted as an improvement. External-checkpoint overlap validation
is a separate explicit seed-import path and is not a normal CUDA eigensolve.

With legacy K held fixed and three updates in both host-only captures, the
provider's inclusive host intervals total 1503.300 ms (serial DIIS) and
1424.318 ms (auto). Three 16-byte D2H `cudaMemcpyAsync` calls account for
1499.636 and 1420.586 ms respectively; the actual copies take only about
0.0013 ms total. They wait on queued GPU work despite the API's `Async` name.
All 125.583 ms of serial DIIS kernel activity overlaps these waits, versus
43.760 ms for auto's history update and partial kernels. These unfenced waits
also include earlier K work, so their roughly 1.5 s total is not comparable
to the old component-fenced 203 ms provider interval. The DIIS contribution
nevertheless explains most of that earlier approximately 150 ms excess over
GPU eigensolver work. Separate component runs keep eigensolver GPU time near
52 ms for three solves, with 19,918,488 retained workspace bytes and no host
workspace. Final-state validation is separately visible at about 1.17 s and
has not been removed or mislabeled as eigensolver setup.

## Rejected alternatives and boundaries

- Flattening dense K into a long-K GEMM preserves raw storage but is slower on
  the measured device. The `flat` control retains that isolated experiment.
- Initially summing each dense panel separately reassociated Q reduction and
  changed the common-seed 768 branch from three to five iterations. Continuing
  the original sum across panels restores the three-iteration branch.
- One-warp DIIS reduction alone did not remove enough long serial work; bounded
  block partials supply parallelism without a new allocation.
- Transformed B cannot replace raw A under a truncated metric. A disconnected
  duplicate raw tensor would unnecessarily increase simultaneous residency.
- Two 768-AO prepared owners exceed the available 32 GiB. Derivative ablations
  reconstruct sequential owners in declared ABCCBA blocks, two then three
  clean samples per variant, and verify the same exported density hash.
- The explicit host-tensor streamed compatibility route still performs bounded
  CPU metric-panel construction, pageable H2D and a drain before reuse. It is
  absent from the resident traces. Generated source-backed execution is the
  existing production alternative. Replacing this compatibility provider needs
  its own pinned-storage accounting and measured constrained-memory endpoint;
  resident results do not justify relabeling its costs or forcing residency.

## Invariants and validation

No energy, density, force, screening, precision or metric gates are relaxed.
Independent CPU Libcint tests cover all 64 angular classes at asymmetric and
coincident geometries; CUDA tests cover packed/symmetric/full, RHF/UHF,
Cartesian/spherical, source fallbacks, ragged panels, corrected states, and
model/geometry/property replay. Resource plans freeze the six added execution
controls. Older schema-1 checkpoints may omit those extensions but still need
explicit warm-compatible restart, retaining their original provenance.

The retained [evidence](../../../../benchmarks/results/issues388-391-df/README.md)
contains final clean samples, separate component and host profiles, branch
identities, memory/resource ledgers, validation outcomes, measured source
patches and reproduction commands. Normal-convergence measurements and
matched-update energy-only diagnostics are separate. Hardware counter
availability limits occupancy/stall claims; CUDA resource/occupancy APIs and
Nsight activities do not measure achieved occupancy or branch efficiency.

The final normal warm medians are 0.920401 s (384) and 5.308092 s (768),
versus baseline 2.641318 and 6.436194 s, with three updates in every warm
sample. Each normal run uses its own post-cold density. The 768 cold solve
changes from 21 to 24 updates, so cold timing is not matched-work evidence.
Isolated controls use identical density hashes. Resident K alone is slightly
slower at 384 (0.940694 to 0.949764 s with raw reuse disabled); enabling the
raw reuse it makes possible saves more than this cost. The 768 DIIS comparison
holds K at legacy to retain three updates in both variants, giving 5.703966
to 5.627464 s and energy-only 3.585720 to 3.499338 s. The five-versus-three
default-K DIIS branch is retained only as an unmatched diagnostic.

## Revisit when

Reopen policy selection for other GPUs, larger bases, different occupied ranks,
or tighter memory bounds using complete endpoints. Consider further recurrence
or reduction work only with evidence separating arithmetic, cache/resource
pressure and atom-gradient accumulation. Preserve exact raw/metric identity,
all supported symmetry orientations and failure-safe borrowing in every variant.
