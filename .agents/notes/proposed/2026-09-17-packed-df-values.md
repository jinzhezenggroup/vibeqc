# Proposal: exact packed DF values with bounded expansion

Status: proposed; native and high-level implementation validated, endpoint qualification pending
Date: 2026-09-17

This is the historical proposal and experimental log. The completed
[retention decision](../implemented/performance/2026-09-17-packed-df-retention.md)
supersedes its earlier pending states; raw observations below remain unchanged.

## Problem and experiment boundary

Issue #409 changes persistent raw/transformed three-center storage. Exact
triangular AO-pair storage roughly halves each tensor at 768/768, but expanded
K/response scratch and repeated unpacking can erase that benefit. Pin the first
independent comparison to `25e8efebb069e6067ba3649f220c37e844ecc8f6`, retaining
#415's derivative mapping and the existing single SYRK Gram. #412's failed
split experiment is separate; do not add its local kernel saving to this work.

Phase A stores every unique symmetric pair in FP64 with no new screening. It
must end with a measured latency/capacity policy or a retained negative result.
Smaller factor storage alone does not qualify a complete endpoint.

## Intended contract

Use typed pair representation and deterministic triangular ordering. Distinguish
raw A from whitened B, binding each view to geometry, both bases, provider,
metric/cutoff and physical owner generations. Existing dense public/nonsymmetric
tensor APIs keep an explicit dense representation. Equal dimensions cannot
authorize a packed view or let packed/full cache entries alias.

Raw A retains discarded metric directions. Never reconstruct it from B through
a truncated inverse transformation. Dense exports may materialize explicitly
requested outputs; ordinary packed execution must not retain a hidden full copy.

For J, packed density weights are D(mu,mu) on the diagonal and
D(mu,nu)+D(nu,mu) off diagonal, followed by direct B contraction and mirrored J.
There is no implicit sqrt(2) convention. This also preserves nonsymmetric D when
the physical factor itself is symmetric.

For occupied K, first compare bounded auxiliary-block expansion into the existing
projection consumer. Write each projected Q block directly into the original
full-U layout using its full leading dimension, then use one final Gram and the
existing final-U lease. New BLAS shapes can still change rounding/SCF work.
A packed-aware projection requires separate evidence, not duplicate equations.

## Confirmed extension points

- `src/scf/cuda/df_plan_setup.cpp::materialize_generated_tensor` already generates
  bounded raw pair/P panels and whitens directly into retained B without a full
  raw setup tensor. Triangular row slices can reuse its dense-contiguous-pair
  producer, but extra generation/whitening launches must be measured.
- `src/scf/cuda/df_generated_tiles.hpp::generate_metric_panel` preserves disjoint
  staging and ascending raw-auxiliary sums. Do not repeat raw recurrences per Q
  when reusable larger raw panels fit.
- `src/scf/cuda_df_gradient.hpp::CudaDfRawTensorView` currently describes only
  dense A through three strides. Packed raw needs an explicit typed boundary.
- Packed response weights in `df_response_weights.cu`/`df_gradient_bridge.cu`
  are separate from persistent value packing; reuse their consumer contracts.
- Generic compiler `tensor/packing.py::PackedLayout` is explicitly a bounded
  reference enumerator, not a large-system planner. Do not enumerate the full
  Naux*NAO^2 tensor through it for production planning.
- `src/scf/rhf.cpp::make_cuda_density_fitting_plan` and its batch variant already
  receive an occupied-rank bound for planning, but the native create calls do not
  forward it as scratch capacity. They currently select generated-source creation
  only for a positive value budget; the zero-budget route consumes full raw host
  vectors. Packed admission must happen before full raw construction, independently
  of that old budget-triggered route choice.
- `src/scf/density_fitting.cpp::workspace_bytes` explicitly charges full resident
  tensors and three/four tile buffers. A storage tag must reach this common
  planner, the native allocator/diagnostics and Python resource identities.
- Occupied/seed/final selectors currently exclude integral-source-backed plans.
  A new packed resident source needs explicit validated admission and raw-owner
  binding; removing all source-backed exclusions would admit unrelated streamed
  plans. Otherwise a nominal packed/full comparison could silently change dense
  versus occupied algebra as well as storage.

## Questions to resolve before selecting an implementation

1. Propagate known occupied-rank capacity from physical SCF preparation. Unknown
   rank in fixed-density APIs still requires an exact bounded route. Keeping all
   old full-size scratch would hide most expected savings.
2. Charge packed A/B, pair metadata, expanded blocks, U, dense seed fallback,
   response projections, library workspace and overlapping replacement state
   through #203 before allocation. Borrowing cannot destroy raw A or a live U.
3. Preserve dense K for rejected seed factors and other legitimate consumers;
   symmetric physical factors do not imply that every supplied D is PSD.
4. Packed raw response must support truncated metrics without a full raw copy.
5. Choose the diagnostic selector after inspecting source/plan ownership. Freeze
   representation identity explicitly; automatic selection stays full initially.

## Validation and decision gates to declare before measurement

Independent raw-integral/J/K/energy/full-force checks retain existing gates.
Cover diagonal/off-diagonal, Cartesian/spherical, RHF/UHF/empty spin, odd/tail
auxiliary blocks, unequal NAO/Naux, batch, changed geometry, failed neighbors,
stale maps, overflow, constrained storage and nonsymmetric fallback.

Measure generation/whitening, J, K unpack/projection/Gram and raw response with
true traffic/work counts, then complete cold, changed, warm energy and forces.
Declare numerical, latency/no-regression and useful capacity margins beforehand.
Include 192/384/768 dataflow, 96/192 regressions, a constrained case and a supported
larger case where residency/recomputation can change. Finish all builds before
clean timings and run every GPU command through finite Slurm allocations.

Keep frozen starting states and normal convergence; label changed-work outcomes
separately. Optional Phase-B screening cannot rescue a negative exact-packing
experiment. #206 retains fresh matched stock GPU4PySCF qualification.

## References

- #409, #203, #206; independent Gram rejection in #412.
- [Performance engineering](../../../docs/performance_engineering.md).
- [Occupied CUDA ownership](../../../docs/df_occupied_cuda.md).

## Initial bounded experiments (2026-09-17)

The experiments in `benchmarks/experiments/issue409-packed-values/` keep the
native baseline unchanged. A bounded preload capture from post-#415 native
identity `d747b6d8a857889ca8afef1b7d734b0c0adecf5681824a16496bab0785d862e9`
retained four eager occupied builds each at 384/rank80 and 768/rank160. Original
B symmetry errors were at most 1.805e-15. Synthetic prefix/tail fixtures cover
192, unequal 97/193/rank37, small 13/7/rank9, and rank zero; these are not new
molecular endpoint domains.

Seven interleaved repeats on the RTX 5090 show the direct 64-by-16 FP64 tiled
iterator reducing **projection plus unchanged Gram/mirror** time by 8.92–8.94%
at 384 and 4.97–5.24% at 768. Bounded unpack32 was 2.18–2.20% slower at 384
and 2.86–2.88% slower at 768. These are fixed-input component results, not
complete SCF, energy or force qualification. The direct iterator is the next
integration candidate; keep the bounded expansion fallback. No split Gram
is introduced. No hardware DRAM or whole-process memory saving is inferred
from the logical traffic and common-arena allocation counts.

The first compatibility run converted C to row-major; its data is retained
separately. The selection run restores native SCF column-major C before timing,
including corresponding cuBLAS operands and coalesced shared coefficient loads.
Both runs passed complete U/K comparisons and independent long-double sampled
projection checks. Memcheck with full leak checking, initcheck and synccheck
passed the odd unequal case across every candidate.

A separate direct-row producer uses the existing raw tile generator to fill
packed A without building dense A, whitens all Q, and contracts packed J.
Independent libcint/NumPy checks passed ten batch items spanning Cartesian,
spherical, long contractions, unequal public representations and truncated
metrics, using nonsymmetric D. The first truncated fixture's 0.05 cutoff
retained every direction; its coverage failure was kept. The corrected 0.2
fixture retained 4/16 and 8/16 directions and passed unchanged error gates.
This test injects an oracle X; production must reuse the native metric owner.
It does **not** validate packed force response or its discarded-space terms.

Local evidence is under `.artifacts/issue409/projection/`: `trials-column/`
is the production-layout comparison, `trials-v1/` the row-major compatibility
run, `producer-validation/` and `producer-followup/` retain the coverage failure
and correction, and `sanitizers/` retains tool output. The initial preload link
failure is also retained. Evidence is not yet packaged as a completed Phase A.

## Native integration in progress (2026-09-17)

The explicit source constructor now owns distinct packed A/B allocations and
uses native cuSOLVER metric setup. Common/native capacities reserve complete U
only for the requested rank, keeping exact bounded J/K fallbacks. Packed J uses
the sum of both density triangles; occupied K keeps the original U layout and
one Gram. Automatic high-level selection remains unchanged.

Raw response has a separate typed packed view. It reuses the existing occupied
response equations and derivative consumers with checked unequal scratch
capacities. Final-U reuse still requires a full-rank metric and a valid exclusive
lease. Otherwise direct raw projection or one-Q unpack includes discarded metric
directions. Failed provenance/capacity validation uses bounded dense response
from immutable packed raw values, without recurrence regeneration or a full raw
copy. Explicit seed/final overrides narrowly admit packed resident sources.

Native tests now cover packed J/K against independent raw-integral references,
nonsymmetric density, batch/changed geometry and rank-capacity fallback. Force
tests add RHF/UHF/empty spin, unequal dimensions, auxiliary tails, truncated
metrics, multiple occupied orbitals, final-U lease consumption, raw projection,
stale/absent tokens and insufficient rank-squared capacity. Their execution and
sanitizer results must be recorded before treating this integration as validated.
High-level early preparation, representation/cache/resource identity and complete
endpoint/capacity qualification remain pending.

## Native qualification and high-level integration (2026-09-17)

The first native integration passed density-fitting and occupied-response suites
in Slurm job 9833. Response memcheck/initcheck/synccheck in 9834 and both suites'
full leak checks in 9835 reported no errors or leaks. The resource query's 14
CPU-safe tests passed. A frozen library, reconstruction patch, build identity and
logs are retained in `.artifacts/issue409/native/validated-stage1/`. The response
trace contains 354 physical packed records, covering direct projection, sliced
raw projection, final-U reuse and bounded unpack. These small s-shell force
fixtures do not establish general molecular or endpoint qualification.

High-level preparation now accepts `VIBEQC_DF_VALUE_STORAGE=packed` before any
complete host raw tensor is constructed, even for zero budget. Its default stays
dense. Prepared/cache/Fock/resource identities include storage, and fixed-density
Fock reserves rank zero with bounded exact K. Existing qualified 768/768/rank160
seed/final and derivative policies narrowly admit explicit packed experiments,
so representation comparisons preserve the established algorithmic policy.
`VIBEQC_DF_RAW_REUSE=off` restores bounded raw regeneration for an ablation.
The complete Python inventory charges both immutable owners and unequal scratch
capacities, retaining its existing small-orbital domain restriction.

Molecular RHF/UHF, Cartesian/spherical, unequal auxiliary, empty-spin, batch,
zero/positive budget, geometry-change and representation-replacement tests have
been added against independent PySCF energy/full forces. Runtime validation of
this high-level stage and complete Phase-A measurements are still pending.

## High-level validation and first complete endpoints (2026-09-17)

Frozen native identity
`2d59ce1e7d001a6e9d0a28b90f44e6fcbcf8e752a6682b1d80a98738c72b3133`
passed both native suites, 15 CPU-safe resource cases and 15 molecular/Fock GPU
cases in job 9836. The latter include independent PySCF energies/full forces,
representation replacement and warm reuse, changed geometry, Cartesian/spherical,
RHF/UHF/empty spin, zero/positive budgets, an enforced global ledger, raw-reuse
off and arbitrary-density composed Fock. A private additive Fock query reports
storage from the actual prepared owner without changing the public diagnostic
struct's ABI. Molecular memcheck in job 9838 reports zero errors and zero leaks;
two neighbor tests were initially skipped because their separate tier flag was
missing. The corrected neighbor-only job 9839 passes both budget cases.

The intrusive preflight in job 9837 retains an important negative result:
768-AO packing changes the frozen solve from 3 to 6 updates. Complete forces
remain within the unchanged numerical gates, but diagnostic latency increases
from 2.536159282 to 3.203406938 seconds. Final-U reuse occurs in both arms. The
native plan's reported setup peak decreases from 18,693,577,387 to 6,165,909,877
bytes; this is a plan capacity, not a sampled complete-process peak or a useful
capacity-crossover demonstration by itself. A single post-call process reading
also decreases, but is not sufficient peak evidence.

At 384 AO, both arms retain 3 updates, but packed force response cannot borrow
the baseline's three full dense buffers. With the established dense-exchange
policy there is no authorized occupied factor, so the exact bounded raw loader
is used. Its repeated dense response products dominate. Do not hide this cost
by silently changing the exchange policy or treating the direct projection
microbenchmark as an endpoint prediction.

Seven interleaved clean pairs completed so far retain the following medians
(dense -> packed, seconds): 96 forces 0.108850834 -> 0.105526933; 192 forces
0.165658090 -> 0.130553789; 384 forces 0.693052435 -> 1.820831211; 384 energy
0.284203221 -> 0.269924065. Update counts match within these four cells, and
all numerical gates pass. Their separate intrusive work records still require
full operator-count reconciliation. The 768 clean matrix, cold/changed and
unequal/larger/constrained endpoints remain pending. No automatic selection or
complete Phase-A acceptance follows these partial results.

## Completed warm matrix and work reconciliation (2026-09-17)

Job 9840 completed all six declared seven-pair clean cells. The final 768-AO
medians are 2.548116406 -> 3.216367704 seconds for complete forces and
1.013974837 -> 1.676097540 seconds for energy only. Every dense sample has three
updates and every packed sample six; these are normal-convergence regressions,
not matched-work projection comparisons. All unchanged numerical gates pass.

Separate intrusive records reconcile actual J/K/eigensystem work, combining
eager events with observed graph replays and consecutive device readbacks.
Graph construction is never counted as execution. At 96 AO both arms execute
3 J, 3 K and 2 eigensystems; at 192/384 both execute 4 J, 4 K and 3 eigensystems.
The 768 eager traces directly show 4 -> 7 J/K/eigensystems, including the
density-seed eigensystem. Every cell performs one final physical Fock and zero
final eigensolves, density corrections or candidate rejections. These are
logical operator counts, not kernel counts or inferred GPU times.

At 192 AO, the packed response eliminates 56,623,104 bytes of raw host upload
and the associated host gathering/pinned-panel synchronization. Both arms
retain 384 AO matrix products, 384 density products, 36,864 metric dots and
identical generated derivative shell work. The separate force diagnostics are
164.149 -> 125.997 ms; they help explain the clean complete-force gain but are
not themselves clean timing samples.

The first constrained diagnostic in job 9841 passes a 384-AO packed calculation
at a 3-GiB total DF budget and closes its native ledger to zero. At 768 AO and
12 GiB, the dense arm also passes, but its diagnostic force endpoint takes
162.144786 seconds and five updates. It generates 1,626 raw blocks and performs
3,072 AO matrix products; bounded one-electron derivative export is another
large cost. Its native observed peak is 12,133,970,329 bytes and sampled warm
process peak 18,647,875,584 bytes. This numerical ledger is an observation
ceiling, not large-domain Python inventory admission or a whole-process budget
guarantee. The packed arm and clean constrained repeats remain required.

The remaining reproduction scripts separate cold/changed clean timing from
intrusive traces and retain the declared seven pairs. Practical unequal
auxiliary coverage uses the unmodified cc-pVDZ/cc-pVDZ-JKFIT definitions,
including f shells; no unsupported shell is removed. The common def2 universal
fitting basis includes g shells for O/N/C, outside the current CUDA capability.
The proposed unequal cases must pass runtime validation before they count as
coverage. No default representation is changed by these pending experiments.

The completed 12-GiB packed diagnostic in job 9841 takes 65.127802 seconds and
six updates, with energy/force errors within the same gates. Its native observed
peak is 5,892,268,465 bytes and sampled warm process peak 12,431,917,056 bytes,
reductions of 51.4% and 33.3% respectively. Sampled host peak increases 17.2%,
from 6,448,332,800 to 7,558,451,200 bytes. Both ledgers close to zero. Both value
plans remain resident; this is evidence of different constrained response work,
not a demonstrated streamed-to-resident transition. The packed response borrows
its retained raw/final U and takes about 1.447 seconds, while both arms still
spend about 58.56 seconds exporting one-electron derivatives. Seven clean pairs
under the identical budget are queued separately; the diagnostic ratio alone
does not satisfy the performance gate.

Job 9845 completes seven 384-AO cold pairs: dense 10.135832240 seconds versus
packed 11.948631649 seconds. Packing regresses 17.9% despite reducing updates
from 20 to 18; all numerical gates pass. The changed-geometry smoke test also
passes its independent full-force reference and immutable starting-density
check, with twelve updates in each arm. The full changed and 768-AO cells remain
in progress. Failed first harness runs are retained: their errors concerned
GPU4PySCF metadata interfaces and CDERI lifetime after gradient evaluation,
not failed numerical gates. The corrected stock reader observes the active
device-ordered CDERI list before the normal gradient releases it.

## Draft review boundary

The completed 384-AO changed-geometry seven-pair median is 9.606932915 ->
11.256079176 seconds (17.2% slower), with twelve updates in both arms. Its
independent full-force reference and frozen-density checks pass. The 768-AO
cold/changed cells, practical unequal-auxiliary matrix and clean constrained
matrix are still running or queued when this draft is prepared.

The [review bundle](../../../benchmarks/results/issue409-packed-values/README.md)
retains compact raw samples and source reconstruction for the measured frozen
library. The configured clang-format hook subsequently formatted the changed
native sources; those formatted sources require a fresh build and validation
after clean timing finishes. The draft must not be treated as merge-qualified.
The portable warm runner is lint/help checked but has not yet been exercised
on the GPU; exact historical harnesses are retained separately.

The direct packed projection is an explicit handwritten performance exception:
ordinary dense contraction strides cannot express its triangular gather. It
adds no new physical equation or generated capability. Retire it if no complete
endpoint/capacity domain qualifies, or replace it once compiler-owned triangular
gathering passes the same numerical, resource and endpoint gates. The dense
default and bounded exact fallback remain necessary under the current results.

The completed 768-AO cold cell retains seven clean pairs and two separate
diagnostics: dense 85.685922099 -> packed 82.768054998 seconds (3.4% lower
ordinary latency), with 23 -> 24 updates. All numerical gates pass and the
logical work counts reconcile. This result is separate from matched-work
comparisons; changed-geometry qualification is still in progress.

The storage/work audit distinguishes graph execution from counter availability.
At 96/192/384 AO, untimed priming caches the graph before tracing is enabled;
only the final eager J has a packed FLOP counter. The observed replay count and
two-GEMV source contract determine logical total J FLOPs, recorded separately
from eager counters. Multiplying nonexistent construction records would silently
undercount the solve. The audit also charges both immutable factors and all three
simultaneous shared scratch buffers; at 768 AO these total 5,591,531,520 bytes
before source/metric/SCF/library allocations. Semantic transfer counters remain
per operation, and no complete hardware traffic is inferred by adding them.

## Additional completed cells and retained failures (2026-09-17)

All seven 768-AO changed-geometry clean pairs complete before job 9845 reaches
its 90-minute limit: 86.357965270 -> 191.511216193 seconds, nine updates in both
arms, and unchanged numerical gates pass. The timeout interrupts the subsequent
diagnostic portion. Keep the exact original series and scheduler terminal record;
job 9851 supplies only a separate diagnostic companion. Do not pool interrupted
and retried series, or infer the regression's cause from residuals alone.

The complete-shell cc-pVDZ/cc-pVDZ-JKFIT campaign passes seven warm pairs and
separate diagnostics at 24/116 and 96/464 AO/auxiliary sizes. Their force medians
are 0.019747868 -> 0.018764724 and 0.568663857 -> 0.552863172 seconds, with two
updates in every arm. These do not qualify the larger 384/1856 workload: its
dense cold preflight fails the 1e-8 force gate with 1.300395833e-8 Eh/Bohr;
energy error is 3.98e-10 Eh and both metric ranks are 1856. The failure precedes
packed execution. Preserve the complete failed record and leave this domain
unqualified while diagnosing the independent-reference difference.

## Binding separately completed diagnostics (2026-09-17)

The evidence collector can attach a completed diagnostics-only companion to
the original seven 768-AO changed-geometry pairs. It verifies the original file
hash, companion process identity, library/source identity, both geometries,
reference/checkpoint hashes, frozen density and controls. It rejects new clean
samples in the companion and rejects incomplete original interleaving or failed
diagnostic numerical gates. Both input files remain retained; the derived record
names their hashes and separate scheduler identities. A changed diagnostic SCF
branch is retained and labeled, rather than presented as the original clean
work distribution. Twelve CPU tests exercise accepted bindings and invalid
provenance, pooling and numerical cases. This collector support does not itself
complete the still-pending GPU diagnostic campaign.

## Completed constrained endpoint and warm CUPTI records (2026-09-17)

Job 9848 completes all seven 768-AO pairs under the same 12-GiB DF allowance.
Dense/packed complete-force medians are 162.023276958/65.028559662 seconds;
the paired packed/dense ratio interval is [0.401023237, 0.401756930]. Every
sample satisfies the unchanged gates, with maximum energy/force errors of
9.05e-11 Eh/1.99e-10 Eh/Bohr. Updates remain five versus six, so this is a
59.86% ordinary-latency reduction with different work. The separate unprofiled
capacity observations retain the 51.4% native allocation peak reduction,
33.3% sampled process-device reduction and 17.2% sampled host increase. Both B
plans remain resident. This supports a useful constrained endpoint; it does not
erase the unconstrained warm/changed regressions or complete final qualification.

Job 9850 completes six separate warm Nsight processes, explicitly tracing CUDA
Graph nodes. SQLite/CUPTI records confirm the 192-AO raw-upload reduction of
56,623,104 bytes. At 384 AO, total H2D bytes are unchanged while kernel launches
grow from 1,189 to 20,106. The API account also separates the native tracer's
own fences: progress mode adds one per eager region, and destruction adds one
per eager operation. All 9,796 event synchronizations in the profiled 384-AO
packed run are accounted for by those tracing fences; treating them as clean
production synchronization work would invent a different explanation for the
real endpoint regression. Kernel busy intervals are unioned, and nested API
times remain separate. Nsight process-memory samples include profiler overhead;
the unprofiled capacity measurements retain their independent acceptance role.

## Completed changed-geometry attribution

Job 9851 supplies the missing diagnostics for job 9845's already completed seven
clean pairs; no clean sample is rerun or pooled. The collector binds native and
library identities, geometry, model/reference/checkpoint hashes and frozen D,
and preserves both original process records. Both arms accept the occupied seed
and retain occupied exchange: nine updates, sixteen J/K calls (ten occupied K,
six dense K), and six final density corrections.

The response fallback explains the observed work expansion. Dense borrows full
all-Q scratch and takes 3.495 intrusive seconds. Packed cannot lend its bounded
scratch without eligible occupied factors, executes 77 auxiliary blocks and
118,272 AO products versus 1,536, and takes 112.362 seconds. Its immutable raw A
is reused, but 35,326,918,656 logical matrix elements are unpacked. That fallback
also falls outside automatic 768-AO borrowed shell/BLAS selection. The retained
response-attribution record pins source and observation hashes and keeps parent
response durations separate from exclusive child scopes. It identifies the
executed path, without attributing a counterfactual saving to any one selector.

Do not explain this result as rejection of the occupied seed or permanent loss
of occupied SCF exchange. Repeated bounded algebra is the observed regression;
smaller storage alone does not bound that work. Default dense remains necessary.

Job 9854 independently checks the larger unequal failure using CPU
PySCF/libcint at tighter convergence. Native dense differs by 1.297054950e-8
Eh/Bohr, while the original stock reference differs by 1.219315759e-10. The
384/1856 domain remains excluded; the reference is not the source of the
failed threshold, and no packed result or speed claim exists for that cell.
The passing 24/116 and 96/464 physical bases remain the unequal qualification.
