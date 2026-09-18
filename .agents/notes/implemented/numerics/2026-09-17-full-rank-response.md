# Full-rank response factors before metric quadratic products

Status: implemented (qualified domains and outstanding boundaries recorded below)
Date: 2026-09-17

## Evidence and candidate

The independent diagnosis retained by PR431 identifies raw quadratic-product
rounding followed by the metric spectral reverse map as a concrete source of
practical-auxiliary force error. Slurm9909 independently evaluates an all-FP64
CUDA inverse-first prototype at retained native densities. With the unchanged
derivative backend its OH and water force errors versus the tight CPU oracle
are 1.89e-12 and 1.65e-12. These are untimed diagnostics, not native endpoint
or performance acceptance. A one-sided inverse-after-product alternative
fails water at 9.13e-10 (Slurm9910); do not repeat it for a favorable result.

The candidate forms B=A M^-1 before density products. Full-rank A adjoints and
metric adjoints then use the same B and c=D:B. Dense full tiles reuse their
two already charged panels; resident dense plans preserve immutable raw and
borrow two mutable tensors. Occupied response forms the Gram from its existing
inverse-applied projected factors. No new resident tensor is allocated.

The metric owner explicitly publishes full rank and its immutable plan-generation
token in the borrowed view. Identity checks include both, even when a smaller
panel borrows no raw tensor that could witness the owner's identity. Unspecified
views and rank-deficient metrics retain
the original spectral map, including retained/discarded subspace motion.
The rank-crossing guard remains at the owning plan.

## First native candidate: retained failure

Slurm9911 tests the frozen library with SHA-256
`b3227d794283bf0dd8ddd1e09c9e587de9677a677d87da9c52da2822b9dfd7f1`.
All 18 cold/priming/control results are retained. OH stays within 8.34e-12,
but water reaches 2.38e-10 and still fails the original 3e-11 gate. All 40
cold/warm/changed-geometry regression endpoints are retained as well: the five
OH cases pass and all five water cases fail. Requesting occupied response alone
did not admit occupied factors in these tests; traces show the dense full-tile
fallback. The next regression explicitly reserves occupied exchange factors.

Four native eigensystem/final-state/recovery tests pass. The occupied-response
test stops on an outdated accounting assertion: the small full-rank fallback
now reports its immutable forward-tensor borrow, whereas the old assertion
expected zero. Update that literal accounting expectation without relaxing
any scientific gate; the remainder of this native test still needs to run.

The failed full-tile implementation formed X X^T and then A M^-1. The next
candidate instead evaluates `((A Q) / lambda) Q^T`, preserving eigendirections
until after division, as in the independently successful Slurm9909 prototype.
It reuses the same two charged panels and counts both GEMMs and their work.
The unchanged-library CPU-only OH ordering probe gives 1.28e-13 combined force
error for A(X X), 2.36e-15 for (A X)X, and 6.55e-16 for the eigenfactor order;
those small OH errors did not predict water's native failure and are not
GPU or endpoint acceptance. All failed candidate-v1 records remain intact.

Slurm9912 tests eigenfactor-order candidate-v2, SHA-256
`08111379f437b0c658ac5fd976589ce329e981f23be075bb672af0b28b08093a`.
All 18 original cold/priming/control rows now pass the force gate (water cold
2.55e-11, warm/control at most 2.37e-12); the complete occupied native test also
passes, including truncated-rank and stale-view cases. The independent practical
regression still has five passing OH cases and five failed water cases. Three
dense-response water cases fail changed-geometry energy, about 5.9e-11, with
the first changed force about 3.91e-11. Actually admitted dense/packed occupied
response also fails force, reaching 3.39e-10 before its inverse-order correction.

Slurm9913 captures each actual candidate-v2 output density across cold/warm and
changed geometry. The independent physical energy at those densities agrees
with the tight CPU oracle within 6.83e-13; native changed-geometry energy remains
about 5.90e-11 away. Thus tightening density alone cannot repair that energy
discrepancy. The first changed density also has an 8.81e-11 commutator and a
3.05e-11 consistent CPU-force error, so stationarity is a separate force limit.
Preserve this evidence rather than treating a warm repeat as a cold/changed pass.

Slurm9914 evaluates four occupied-factor orderings at the original retained
densities with independent A/M and the unchanged derivative backend. Applying
the inverse through eigenfactors after the linear occupied projection agrees
with the independent complete-force oracle within 1.89e-12 for OH and 1.68e-12
for water. Candidate-v3 applies this ordering to both occupied charges and
projected exchange factors, with a counted device copy retaining each spin's
factors in its disjoint borrowed interval. Its native results are recorded below.

Slurm9915 qualifies candidate-v3's response change: all 18 controls and the
occupied native test pass; dense and packed occupied water cold/warm errors
now agree with the stable dense response. The practical suite still fails the
five water cases on the retained changed-geometry energy/first-force limits.
Slurm9916 then passes all 122 existing DF Python regressions and CUDA memcheck
on the occupied native test reports zero errors. These results qualify neither
the unfinished streaming boundary nor complete #206 performance.

## Forward whitening ordering

Slurm9917 reuses saved native densities without SCF and exports the unchanged
CUDA integral producers, native forward eigensystem, white tensor and J/K.
CPU Cholesky on the **same native A/M/H** agrees with the independent fixed-D
energy within 4.55e-13, while native white/J/K reproduce the changed-geometry
error of about 5.82e-11. Independent A-only, M-only and H substitutions are each
below 7e-13. This isolates forward whitening arithmetic, not an inaccurate
density or a large primitive-integral contribution.

The CPU-only ordering audit keeps those exact native Q/lambda values. Applying
`(A Q) / sqrt(lambda)` before rotating back reduces changed-geometry energy
error versus Cholesky on native integrals to 1.26e-12. Slurm9918 confirms the
ordering on CUDA: projecting first then multiplying by scaled Q gives
1.99e-12 versus the independent fixed-D reference, whereas forming the explicit
inverse root first reproduces 5.89e-11. All four CUDA alternatives and both
geometries are retained; no new eigensolver or precision is needed for this
diagnosed difference.

Candidate-v4 applies `Q^T A` then `(Q / sqrt(lambda)) projected` while reusing
existing exchange scratch. Resident generated panels still evaluate every raw
pair once; packed immutable raw remains intact. Both extra matrix products
and their FLOPs are counted. Rank-deficient preparation retains its established
path. Truly streamed preparation and unusually small panels that cannot hold
one complete auxiliary column still need a qualified fallback.

Slurm9919 tests frozen candidate-v4, SHA-256
`05ba16ea2a59453fac06fbe4f2e66ea012cb61c155d25424d0f6c40023d72cb2`.
All 18 controls and the native occupied-response test pass. Changed-geometry
water energy is now within 2.4e-12, but all five water regression cases still
fail the first-changed force at about 3.956e-11 with the original 1e-10 density
tolerance. Cold force is about 2.58e-11 and changed-warm force about 8.1e-12;
neither can erase that first-changed failure. Slurm9920 passes five native
tests, 122 existing Python DF regressions and CUDA memcheck with zero errors.

Slurm9921 predeclares density tolerances 1e-11 and 1e-12, dense/packed values,
OH/water and all four geometry phases. All 32 records pass the unchanged
3e-11 energy and force gates. At 1e-12, water force errors are about 2--4e-13.
This separates stationarity from arithmetic; it does not reclassify the
original 1e-10 failure or establish performance.

Slurm9922 keeps density tolerance 1e-12 and adds 128 MiB budget cases. Twelve
tests pass; dense water at 128 MiB fails energy (5.514e-11) and force
(2.380e-9), and packed water rejects the cold endpoint as out of memory.
All 52 completed endpoint arrays, both failures and traces are retained in
`practical-tight-v1/`. The dense budget trace uses the old response map,
two response blocks and 986 raw-slice productions (72,695,808 bytes), with
67,090,752 bytes of response scratch. Thus this is an actual streamed
boundary, not a tested resident-forward borrow. OH's budget cases still fit
complete response tensors and do not cover that boundary. This plain pytest
job selected frozen v4 through VIBEQC_LIBRARY but did not query the actual
loaded build; earlier controls did. Subsequent tests record loaded metadata
and persist resource failures before raising.

## Candidate-v5 constrained-memory extension (unqualified)

Source-generated full-rank forward panels split one existing tile into raw
and eigen-projection buffers, then rotate the scaled projections into the
requested output panel. Every raw pair/direction is still generated once per
requested panel. Smaller pair batches add launches but no raw-source pass.
The streamed Coulomb route applies Q/lambda/Q^T to its completed raw charge
vector while retaining its two raw-tensor passes.

When one complete A/B tensor and a weight panel fit the response allowance,
the bridge can retain that tensor instead of requiring two complete tensors.
An auxiliary transform never mixes AO pairs, so it projects disjoint AO-pair
batches into two already charged, currently dead metric scratch matrices
before overwriting raw A in place. Weight panels then read B directly and
contract their metric block with one GEMM. This uses one raw-source pass,
independent of response panel count. It does not claim a solution for budgets
that cannot retain even one fitted tensor. Native and budget results are
recorded below; no performance or default promotion is authorized by this evidence.

Slurm9923 tests frozen v5, SHA-256
`c723f9b102698351b88804f7aeb158b80b25d55efa8561d82343887b354809d2`.
All 18 controls and five native tests pass. Thirteen practical cases pass;
the packed water 128 MiB cold rejection remains. Dense water at 128 MiB now
passes all four phases (energy <=6.822e-13, force <=4.538e-13), with the same
67,090,752-byte response scratch. The one-tensor response generates 464 raw
slices / 34,209,792 bytes once, performs 928 AO matrix products instead of
1856, and uses two metric GEMMs. In-place factor application costs 20 GEMMs
over disjoint AO-pair batches. All 52 completed arrays and the resource
failure remain recorded.

The initial four streamed-probe tests exited 127 before CUDA execution:
the frozen library had no SONAME symlink. Slurm9924 fixes only that loader
environment through a separate `libvibeqc.so.0` symlink to frozen v5. All
126 existing DF Python regressions (including those four streamed tests),
five native tests and native occupied CUDA memcheck then pass. The original
loader failures remain retained and are not numerical samples.

Slurm9925 separately predeclares private response and total-budget boundaries.
Dense and packed resident water both pass all four phases under a 16 MiB
private response cap (force <=8.951e-13 / 7.013e-13). Their eight response
blocks make 64 fitted projections / metric GEMMs with no raw regeneration;
packed projection uses 3712 GEMVs, an important work cost. Total dense 80 MiB
rejects cold execution. Dense total 96 MiB passes all four phases (energy
<=6.253e-13, force <=5.359e-13); packed total 160 MiB rejects cold execution.
These are separate resource rejections, not failed numerical comparisons.
The CPU-only planner probe establishes lower bounds of 43,241,659 bytes
(dense) and 85,079,227 bytes (packed), even before source/DIIS reservations.
The force endpoint assigns half the public allowance to this value plan,
so the 80/160 MiB rejections do not establish an allocator regression.
The packed v5 trace advertises a raw view's bytes even when this bounded route
reads only the whitened owner; do not interpret that counter as raw traffic.
Candidate-v6 fixes this counter to report raw reuse only on an executed raw
route.

## Candidate-v6 streamed factor convention (unqualified)

The v5 96 MiB boundary exposes a projection-work cliff: forming symmetric
whitened output panels requires a complete eigen-projection for every small
output panel. One captured occupied-K operation reports 138,891,755,520
whitening FLOPs and 4556 GEMMs, although raw source evaluations are unchanged.
Memory bounding and raw-pass counting alone did not qualify this arithmetic.

The private streamed-K consumer sums over the entire whitened auxiliary axis.
Consequently `E=A Q diag(lambda^-1/2)` and symmetric `C=E Q^T` yield the same
exchange matrix, including its occupied-factor contraction. Candidate-v6
keeps E for full-rank **source-generated streamed K only**. Each output panel
accumulates its Q columns over the original bounded raw blocks, then divides
by sqrt(lambda) once. This restores the original raw-block/GEMM work order
without an explicit inverse root or a second auxiliary rotation, and permits
small panels without adding a full eigendirection scratch tensor.

Resident/packed forward owners still publish symmetric C. A streamed owner
publishes no resident whitened view and cannot lend a final projection to the
response path; its response reads raw A and applies Q/lambda/Q^T independently.
This ownership boundary is essential: a future consumer must not interpret E
as symmetric C. Rank-deficient and host-streamed compatibility routes keep
their prior convention. Independent complete-force, streamed-J/K and work
checks are required before qualifying this candidate.

Slurm9927 tests frozen v6, SHA-256
`3f039fc3af10da3ef8756436deae4d60e62133951d286d9303daae4f8f7d87c4`.
All 18 controls, five native tests and four independent streamed-J/K tests
pass. Thirteen practical cases pass, retaining the packed 128 MiB resource
rejection. Every one of 52 completed endpoints passes the original 3e-11
gates at the diagnostic 1e-12 density tolerance; maximum force error is
4.924e-13. An augmented native test, recorded separately after the library
freeze, checks the one-tensor route against the raw-integral derivative
oracle with a one-slice allowance, a partial AO-pair tail and one source pass.
The native library hash is unchanged by this test-only build.

For water at 128 MiB, a streamed K operation changes from 196 whitening
GEMMs / 31,746,686,976 FLOPs in v5 to 49 / 3,968,335,872 in v6. Logical raw
evaluations remain 29,933,568, while raw tile launches change from 98 to 49.
The trace distinguishes ordinary execution from graph construction; these
counts are per recorded operation, not an inferred whole-SCF replay total.
No clean latency conclusion follows from these instrumented checks.

Slurm9926's full v5 water/128-MiB memcheck exceeds its predeclared 20-minute
limit before completing a force endpoint. Preserve its partial trace, timeout
and lack of a sanitizer summary; this is **not** a memory-safety pass. No
unchanged retry is admitted. The separately frozen small native boundary
test and rectangular streamed-J/K probes are scheduled under memcheck for
v6, and larger-size/batch checks have independent CPU references prepared.

## Candidate-v7 packed bounded projections (unqualified)

The 16 MiB packed v6 response makes 3712 separate GEMVs, each rereading the
resident forward tensor. The v7 candidate projects a complete packed panel
with one GEMM into the tail of the already charged dense destination. It
then stages each packed Q slice in the existing spare AO matrix before
expanding it, in ascending Q order. The end of expanded slice Q is no later
than the start of packed slice Q+1; staging the current slice avoids its
own in-kernel read/write alias. No allocation or raw regeneration is added.
Device staging copies and bytes are reported explicitly, and the native
raw-integral derivative regression also exercises a three-Q panel and tail.
Independent numerical and sanitizer results are still required.

## Completed v6/v7 qualification checkpoint

Slurm9928 completes the v6 budget diagnostics with 12 passing force endpoints
and the same two cold resource rejections (dense 80 MiB / packed 160 MiB).
Maximum energy/force errors are 2.388e-12 / 9.004e-13. Dense 96 MiB retains
the one-tensor route, 464 raw slices once and 50,280,768 owned response bytes.
The dense/packed 16 MiB private caps use resident whitened borrowing with
16,660,800 owned response bytes and no raw regeneration.

Slurm9929 reports zero memcheck errors for the separately frozen augmented
native boundary test and for all four rectangular/partial-tail streamed-J/K
tests under child-process instrumentation. This targeted coverage does not
reclassify Slurm9926's complete-force timeout as a sanitizer pass.

Slurm9930 completes four dense/packed configurations: water octamer B=1 and
water tetramer B=2, each with cold, warm, first-changed and changed-warm phases.
All 24 item endpoints pass, with maximum energy/force errors 5.685e-12 /
1.515e-12 at diagnostic density tolerance 1e-12. Independent CPU references
and both geometries were frozen before execution. All metric ranks are full
(928 octamer / 464 tetramer) in every phase. Both octamer configurations
actually execute bounded resident-whitened response; requesting occupied
response in the packed configuration did **not** admit occupied factors.
The tetramer B=2 packed route does admit rank-20 occupied factors, whereas
B=2 dense uses complete dense response. The v6 packed octamer still reports
5568 fitted-panel GEMVs; these results do not qualify v7's larger packed path.

Slurm9931 qualifies frozen v7, library SHA-256
`f5f94b67d153e164f7fdf24fa713703bd5adb2eb1c6122ad0610528c98171d77`.
The augmented native regression and its memcheck pass (zero errors), as do
41 packed/resident Python regressions. All eight dense/packed 16 MiB force
endpoints pass, with maximum energy/force errors 2.388e-12 / 8.871e-13.
For every packed phase, 3712 fitted GEMVs become 64 fitted GEMMs at unchanged
16,038,690,816 projection FLOPs and 16,660,800 owned response bytes. No raw
generation/upload or packed-raw reuse occurs on this route. Its 3712 staging
copies / 138,264,576 bytes are explicitly reported; reducing BLAS calls alone
does not establish a complete-force speedup.

The all-sample audit verifies each loaded library/source identity, every
endpoint, executed route, metric rank and resource/work counter. The proposed
Python regression now separates admitted numerical cases from explicit
resource-contract cases below the native planner's lower bound. In particular,
packed water 128 MiB is an out-of-memory status test; OH packed 128 MiB remains
a force-accuracy test. Frozen prior test sources and qualification failures
remain unchanged. No numerical gate, production convergence policy or clean
benchmark admission has changed.

## Unfinished boundary

The candidate handles complete dense tiles and admitted resident/occupied
buffers. Smaller unborrowed tiles now borrow the owner's resident forward
whitened tensor when available and project two bounded panels. They form the
metric adjoint directly from W_P:B_Q; no raw integral is regenerated. Repeated
projections are counted, including dense GEMMs and packed projection/unpack
calls, and must be checked for work amplification at larger shapes.

A truly streamed owner that cannot retain even one fitted response tensor
still uses the old general response and is **not yet a qualified
strict-accuracy fallback**. Source-generated forward K now has the separate
v6 eigenbasis candidate; host-streamed compatibility whitening is unchanged.
Completing that boundary is required before declaring this numerical repair
finished or promoting it. Naively reconstructing each fitted slice from every
raw slice inside every outer panel creates a severe source-work cliff; it is
not an admitted solution.

A candidate direction for the remaining canonical-density boundary is to
separate validated occupied-factor borrowing from full raw/J/K tensor
borrowing. Stream raw A once into raw charges and `C^T A_P C`, then apply
Q/lambda/Q^T to those linear occupied projections before their Gram. This
needs O(Naux*Nocc^2) factor storage rather than O(Naux*Nao^2). If even that
does not fit, blocks of occupied rows can cover all occupied columns;
each block contributes its metric Gram and A-adjoint before eviction. That
repeats one raw pass per occupied-row block, rather than nesting a complete
raw reconstruction inside both auxiliary-panel loops. It must preserve
canonical-density/generation validation, account all repeated source and
derivative work, and cannot silently reuse invalid or corrected factors.
This is a design direction, not implemented or qualified evidence. Arbitrary
density and truncated-rank fallbacks still require separate treatment.

## Candidate-v8 independent canonical-factor borrow (unqualified)

The current candidate implements the all-occupied-projection portion of the
direction above; occupied-row blocking remains unimplemented. Factor selection
now validates canonical density/token/device generation independently of a
mutable J/K tensor lease. Resident consumers still check their actual unequal
capacities after selection. A source-generated, streamed, full-rank owner may
lend only its immutable canonical coefficients when response space is not
explicitly dense. It never lends the streamed K eigenbasis projection.

When a full fitted tensor cannot fit, the bridge may allocate all spin
projections plus one reusable projected/weight buffer inside its existing
private allowance. Each generated raw slice feeds every charge and spin's
`C^T A_Q C` before eviction. The third existing AO temporary holds raw A while
the first holds A*C. The small projected factors then follow the same
Q/lambda/Q^T ordering and Gram/derivative expansion as resident occupied
response. Both spins retain disjoint intervals throughout inverse application.
One raw pass, projection products, factor copies, owned bytes and borrowed
coefficient bytes are counted separately from full raw/J/K borrows.

The option requires enough space for all occupied projections; it does not
solve arbitrary/corrected densities or lower allowances. Explicit scalar,
serial-dot and scatter ablations retain the old general route. Native tests
cover RHF/UHF including empty beta and unequal ranks, physical raw-integral
derivatives, invalid owners/shapes/factors, and allowances smaller than a full
fitted tensor. Molecular checks predeclare water at 128 MiB total with 16/12
MiB private response allowances and require actual route admission plus a
single raw pass in every cold/warm/changed/changed-warm phase. Results are
pending; no new numerical or performance acceptance follows from this design.

Slurm9934 tests frozen v8, library SHA-256
`58518f5cbb1e727838c703148215e991e6389481c2636bda39aeb13466dee103`.
The augmented native physical-oracle/provenance tests and memcheck pass with
zero errors. All 61 Python regressions pass: 41 packed/resident, four streamed
J/K, 13 numerical-success cases and three explicit resource-rejection cases.
The 52 numerical endpoints reach maximum energy/force errors 2.388e-12 /
4.942e-13. These existing molecular cases do not exercise the new owned
occupied route. Both requested water boundaries reject before force because
the existing API forbids combining a positive public budget with a private
response override; no production contract was changed to admit that request.

The replacement experiment predeclares octamer total budgets 384/512 MiB,
with the existing half-budget response shares (192/256 MiB). The CPU-only
planner lower bound is 162,842,671 bytes for dense values before source/DIIS
reservations; a full fitted tensor is 273,678,336 bytes. Slurm9935 fails its
reference-provenance preflight before force because it compares the historical
generator hash to the edited live test. The corrected script verifies that
generator in frozen v6 and the unchanged basis snapshots; neither the old
report nor the reference arrays are rewritten.

Slurm9936 starts the corrected frozen-v8 octamer experiment. The 384 MiB
source-generated streamed SCF takes approximately 110 seconds per observed
iteration with intrusive diagnostics and has not reached force response.
The job is deliberately stopped before its finite 20-minute limit to release
the GPU for targeted compatibility checks. This is an incomplete, cancelled
diagnostic, **not** a timeout, force-accuracy result or clean latency sample.
No force endpoint completes; the 512 MiB setting never starts. Partial traces,
the explicit stop decision, iteration timestamps, original running reports and
Slurm cancellation terminal are retained. The new molecular streamed-response
boundary therefore remains unqualified. Repeating the same costly SCF setup
without addressing its repeated raw-source work is not an admission strategy.

Candidate-v9 changes only the trace counter for dense weight-panel elements
from `Naux*Nao^2` to the actual `tile*Nao^2`, and documents that the internal
layout descriptor can represent bridge-owned scratch. Its numerical algorithm
is identical to v8. This corrects an old misleading counter; literal allocation,
peak-panel and full-tensor counters remain available independently.

Slurm9937 selects frozen v9, library SHA-256
`4f3061c36b2c0b786d23ab6be1b667a1bc8b14c8825991f8d63722a7e4075b40`.
The native regression and all 46 occupied-response, final-state and
final-projection Python tests pass. The evidence audit verifies that only
the counter and descriptor documentation differ between frozen v8/v9 native
sources, and checks actual dense-panel accounting in the retained traces.
All candidate files and the test runners remain frozen. No numerical commit,
PR, clean benchmark or performance admission has been made for this candidate.

### Source-first streamed occupied exchange (v10 candidate)

The cancelled octamer run used 96 AO rows and 45 output auxiliaries per panel,
regenerating the logical raw tensor 42 times per occupied K. The new candidate
contracts raw AO input with occupied coefficients before metric whitening.
All auxiliary directions of a smaller projected row block then fit, so its
eigenbasis projection is divided by square roots before forming complete K
Gram blocks. The symmetric rotation cancels in K; these private factors are
never exposed as final symmetric-C leases or response projections.

The existing four disjoint plan buffers hold left/right factors, raw input and
metric-projection scratch. There is no allocation, increased planner allowance
or host staging. Both coefficient layouts, full and triangular exchange, zero
rank, partial AO/auxiliary tails and capture/replay keep their existing contracts.
Selection requires a source-generated streamed full-rank metric, one fitting
raw/projected row, BLAS-compatible dimensions and strictly less predicted
logical raw-source work than the previous schedule. Other cases, including
truncated metrics, retain their bounded fallback.

At the observed octamer capacity of 829,440 doubles, projected blocks contain
22 AO rows with a final 16-row tail. Triangular traversal generates 984 rows,
equivalent to 5.125 full logical raw passes rather than 42. This predicts work,
not endpoint speed. Trace counters distinguish row work, projected capacities,
the existing four-buffer capacity and device-copy bytes.

New native physical-oracle checks cover ordinary execution, CUDA capture and
replay after coefficient changes, both coefficient layouts, zero/one/two ranks,
truncated/full metrics and AO tails at the unchanged 3e-11 gate. The independent
fixed-density PySCF checks now cover full and triangular policies and assert
exact generated row work at their existing 1e-9 gate. Molecular regressions
retain density tolerance 1e-12 and energy/force gates 3e-11; they do not replace
the failed original #206 tolerance or qualify clean timing.

The first build failed on mixed enum/int lambda return inference; its log is
retained as `incremental-build-v10.txt`. An explicit `vibeqc_status` return type
fixes compilation (`incremental-build-v10-fix1.txt`). Numerical and sanitizer
results for this changed candidate are pending.

Slurm9938 qualifies frozen v10, library SHA-256
`5dc5c23038ac30bb19d9846218a8b9f0729b3bce488964ac3f08e54f2f211170`.
The augmented native physical-oracle/capture tests and native memcheck pass,
with zero sanitizer errors. All eight full/triangular streamed J/K tests and
38 occupied-response/practical/resource-contract tests pass. The 52 completed
molecular endpoints have maximum energy/force errors 2.388e-12 / 4.312e-13;
their loaded library and source identities match the frozen candidate. The
fixed-density traces verify eight source-first records, including one-pass,
2.5-pass triangular and four-pass full schedules at unchanged buffer capacities.
The initial incorrect audit command duplicated its root path and wrote no
output; the corrected invocation and all source/results are preserved.

Slurm9939 is a separate predeclared diagnostic of the changed v10 candidate
at octamer total384MiB only. It retains the independent cached CPU references
and the same four endpoint phases, and must establish both source-first K
and the owned occupied-response route. The old cancelled Slurm9936 remains
incomplete. Cold setup still needs the existing dense-density seed before
canonical occupied factors become available. Capture trace counters describe
one graph construction's work, not an observed sum over replays. The separate
diagnostic must complete before extending molecular coverage; no clean timing
or stock-GPU4PySCF performance conclusion follows from these work counters.

The v11 build fixes only clang-format's wrapping of the explicit lambda return
type. A source comparison verifies that only whitespace differs from v10.
All ELF sections of the native test are identical; the library differs only in
its build ID and source-identity string in `.rodata`. All CPU code, CUDA fatbins
and other sections match byte for byte. The equivalence records preserve both
identities. Numerical evidence remains attached to the binaries actually run
in Slurm9938/9939; formatting introduces no separate numerical result.

Slurm9939 reaches a completed **failed** cold octamer endpoint: energy error
0.0 and maximum force error `4.8908516056545e-9`, above the unchanged `3e-11`
gate. The new occupied K schedule is selected (984 generated rows, nine row
blocks); its captured counters describe one graph construction. SCF converges
after 20 iterations. Strict finalization rejects the original frame once,
performs one eigen solve/density update and two physical Fock evaluations.
Those Fock evaluations use the existing dense fallback. The selected corrected
state cannot match the original SCF density generation: factor selection
correctly refuses to lend the old canonical factors. No
`occupied_response_provenance` or owned streamed occupied-response counter is
present. Force uses the old general bounded route, with 4,348 raw tile
productions / 1,282,277,376 raw bytes and 201,044,576 owned scratch bytes.
The intended new molecular occupied-response boundary remains unqualified.

The requested scheduling extension from 20 to 45 minutes was denied by Slurm;
the original finite limit remained in force and the denial is recorded.
After the completed cold error was observed, the job was deliberately cancelled
while warm SCF was running. The actual exit is 143; changed and changed-warm
phases never start. This is a failed cold endpoint plus an incomplete cancelled
warm sample, **not** a timeout, all-phase pass or clean timing result. Original
running reports, traces, arrays, source identities, terminal output and explicit
stop decision are retained. The 52 passing smaller endpoints do not override
this larger failure.

The next repair must either give the verified post-correction frame its own
exact density/generation-bound factors or stabilize the bounded general
response for corrected/arbitrary densities. Do not weaken exact token/density
validation, relabel SCF factors as the corrected frame, or rerun unchanged
endpoints hoping for admission. This diagnosis also explains why finalization
retains expensive dense source work despite faster occupied SCF iterations.

### General bounded response after strict density correction (v12 candidate)

The next candidate stabilizes the general full-rank source-generated response
instead of weakening factor provenance. Its fitted-panel provider regenerates
all auxiliary values for one bounded AO-pair block, projects with Q^T, divides
by eigenvalues, and rotates only the requested public auxiliary columns into
B=A*M^-1. The existing bounded response then forms both three-center and metric
adjoints from fitted B before quadratic products. It needs no SCF factor lease,
idempotency assumption or newly published final-state owner.

Two otherwise dead metric matrices at workspace[2*a*a:4*a*a] hold raw input
and eigenprojection; the live metric adjoint occupies a separate matrix. Each
pair block contains at most a pairs, including a partial last block. Existing
weight/fitted panel capacities and the private allowance are unchanged, with
no extra allocation. Full-tensor, resident and qualified occupied paths retain
their selection. Truncated metrics retain the spectral Frechet response.

This correctness fallback deliberately records its additional work: each
fitted-panel request regenerates one logical raw tensor, and the current
bounded metric contraction requests ceil(a/tile)^2 such panels. Counters report
every raw byte, pair-block production, projection product and reserved scratch.
Memory bounds do not imply an endpoint speedup. Independent native physical
derivative tests include non-idempotent RHF/UHF inputs, empty beta, truncated
metrics and both AO-pair/auxiliary tails at the 3e-11 gate. Build completed;
numerical and memory-safety qualification is pending.

Slurm9940 tests frozen v12, library SHA-256
`f1883cb1e960b2c08609f700568211ada3b493fc0b5e9da72839c7ccdb2555b0`.
The augmented native physical tests and memcheck pass with zero errors. All
97 Python occupied/practical/response-weight/derivative regressions pass.
The 52 complete small molecular endpoints reach maximum energy/force errors
2.388e-12 / 4.271e-13 at diagnostic density tolerance1e-12.

An independent CPU PySCF2.14.0 fixture supplies a converged octamer density,
overlap and the complete fixed-D two-electron derivative, including auxiliary
basis response. Its physical commutator is7.267e-13 and the928-direction metric
is full rank at the unchanged1e-10 relative cutoff. A Slurm-only native probe
checks the independent overlap, builds no SCF state and lends no occupied
factors. The initial controller launch names a misplaced script and exits2
before numerical work; the source and failed terminal are preserved, then
the corrected controller is launched under a new version.

Slurm9942 compares the same fixture across frozen v11/v12. The old192MiB
general response fails at4.775117190547462e-9. The new192MiB general response
passes at8.810729923425242e-13, while v12's384MiB single-fitted-tensor control
passes at6.252776074688882e-13. The two192MiB arms both own201,044,576 bytes,
borrow nothing and use a292-auxiliary weight tile. New fitting uses640 raw
pair-block productions and16 logical raw passes /4,378,853,376 bytes; old
response uses4,348 slices /1,282,277,376 bytes. The improvement is numerical,
not a raw-work or endpoint-latency claim. Actual loaded identities, arrays,
all failures and counters are retained in `general-response-v2/`.

Slurm9943 starts the separate frozen-v12 complete endpoint check at384MiB
total, with a finite45-minute limit declared at launch. All cold/warm/changed/
changed-warm phases must pass the same3e-11 gates and actually select stable
general fitting or exactly validated canonical occupied response. This new
test does not reclassify the failed v10 cold endpoint or the original1e-10
stationarity failure. Its results remain pending.

An intermediate Slurm9943 observation records completed cold and warm endpoints,
with maximum force errors 9.98438311194949e-13 and 1.2298972604241065e-12,
respectively. Cold uses general fitting: 16 logical raw passes, 640 productions,
4,378,853,376 raw bytes and 201,044,576 owned response bytes. Warm selects the
exact canonical occupied response: one raw pass, 928 productions, 273,678,336
raw bytes, 59,568,736 owned bytes and 61,440 borrowed occupied-factor bytes.
The source and library identities match frozen v12. Changed geometry and its
warm evaluation remain pending; these two results do not qualify the whole job.

The intermediate observer also exposes a bookkeeping defect in the original
controller: requiring a new `ri_k_occupied` capture record in every phase rejects
warm graph reuse. Cold progress scope175 constructs the graph and records19
replays; warm scope13306 records one replay, no construction, and convergence
at iteration2. `df_rhf_scf.cpp` preserves the graph across compatible solves.
The original observer failure and original controller are retained unchanged.
The version2 observation records numerical/response success separately from
the failed per-phase K-capture gate; it does not mark the whole case passed or
rerun endpoints. Full execution/work accounting must use replay and provenance
evidence rather than interpret capture counters as aggregate execution work.

The separate replay audit verifies unchanged committed graph-owner code, the
loaded v12 identity, the completed readback sequence, per-solve occupied
provenance, and the capture body associated with each geometry. Cold has19
executed occupied iterations; warm has1. Stream operator counts plus verified
replays imply77,553,598,464 and26,307,330,048 generated raw bytes respectively.
These are generated values, not measured DRAM traffic, and exclude setup
outside the selected J/K/response operators. Warm's two dense K builds alone
generate22,988,980,224 bytes. Seed factorization and retained final exchange
currently reject streamed plans explicitly; enabling those paths safely needs
its own identity, reconstruction and complete-endpoint qualification.

Slurm9943 subsequently completes all four endpoints. The original controller
exits1 at `assert occupied, phase` for warm; this failure and the original
reports remain intact. The independent completed replay audit passes every
unchanged numerical, response, memory and source-work gate against the existing
arrays and trace/progress records, with no new GPU run. Maximum energy error is
6.139089236967266e-12 and maximum force error is1.2298972604241065e-12.
Cold/changed use general response, while warm/changed-warm use the canonical
occupied response. Their SCF iteration counts are20/2/14/2 and their occupied
replay counts are19/1/13/1. Final physical Fock counts are2/1/5/1; changed
geometry needs four actual density corrections. Generated raw J/K/response
bytes are77,553,598,464 /26,307,330,048 /101,979,389,952 /26,307,330,048.
This closes this specific192-AO/928-auxiliary,384MiB, density1e-12 numerical
boundary. It does not qualify the original1e-10 request, other supported
boundaries, direct-path acceptance or stock GPU4PySCF performance.

### Independent saved-density stationarity correction study

`stationarity-corrections-v1/` is CPU-only PySCF evidence on the preserved v2
tetramer densities. It predeclares zero through four additional physical-Fock
corrections and retains every array, including the initial failure. For changed
geometry, the initial consistent-weight force error is3.040634410922394e-11.
One **additional** correction after the saved native endpoint reduces that to
1.5272227926743653e-11; the lagged-frame weight gives2.170574830984151e-11.
After four corrections the corresponding errors are2.0090595853616833e-12 and
2.6776358907909525e-12. This diagnoses stationarity and weight consistency; it
does not qualify current native production behavior or justify a universal
one-correction rule. The original failed v4 changed traces already contain two
physical final Fock evaluations. Simply forcing the existing rebuild is not
the additional correction measured here. A prepared, not yet launched,
`diagnose-changed-density-v2.py` will capture current frozen-v12 densities at
the original1e-10 request before selecting a production repair.

Slurm9944 runs that diagnostic on the actual frozen-v12 library, with a finite
10-minute limit, and completes all four phases. Current first-changed native
force error remains3.955578833231277e-11; energy error is2.2737367544323206e-12.
The saved density has an independently reconstructed commutator of
8.755773883706297e-11 and physical fixed-point density drift of
1.0993760762856297e-10. Native finalization reports two physical Fock builds,
one density correction, and final maximum commutator8.7444440577399973e-11.
It reports zero density RMS because the returned density reconstructs from
its own retained frame; that value is not an independent fixed-point defect
against the current physical Fock.

At the exact saved density, CPU consistent `D F D /2` weights still produce
3.0257574223924166e-11 force error. Replacing only the weights therefore does
not meet the unchanged3e-11 gate. Reconstructed canonical weights with the
unmodified saved density give7.150413594558813e-11, so mixing those states is
also not a repair. Preserve `changed-density-v2/` as current-candidate evidence;
the next production change must address physical stationarity and consistent
force state, without bypassing provenance or merely relabeling tighter requests.
No production convergence-policy change or performance admission is made here.

### Physical fixed-point force selection (v13 candidate)

The next candidate changes the production force-state selection, keeping the
external request and numerical gates fixed. After the existing frame, physical
commutator and energy checks pass, force selection solves the current physical
Fock and compares its occupied projector with the candidate density. Both the
maximum elementwise defect and RMS defect must satisfy the requested density
tolerance and the existing absolute cap. This differs from reconstructing the
same retained frame, which can report zero drift at a nonstationary density.

A rejected probe reuses its already validated eigenframe and density for the
bounded correction. Generations still advance; stale occupied-factor leases
remain invalid. The existing maximum correction count does not change.
Energy-only selection does not acquire this extra force-specific solve.
All physical-Fock solves, fixed-point probes, promotions and density updates
are counted separately; a promoted probe is not counted as a second solve.
The work-ledger checks retain the old schema while accounting for these new
provider leaves explicitly.

The force weight is now D F[D] D /spin_weight at the same accepted state on CPU
and CUDA. CUDA uses two existing validation scratch matrices for the products,
with two counted GEMMs per spin and no new device allocation. A physical Fock
download is required for the existing eigenprovider interface; its transfers
and the new solves remain endpoint work. This is a correctness candidate with
an unqualified latency cost, not a claim of additional speedup.

Analytic native fixtures isolate a small-gap state that passes the former
commutator/frame gates yet exceeds the physical projector tolerance; RHF, UHF
and empty beta cases require one promoted correction without duplicate solves.
Another fixture holds the density fixed while changing the physical diagonal
Fock within the eigenframe tolerance, requiring the physical rather than lagged
Pulay weight. The practical molecular suite separately restores the original
1e-10 request for all five dense/packed response configurations on RHF water
and UHF OH, retaining all four phases and the original3e-11 gates. Frozen v12
failures remain unchanged. v13 numerical qualification is pending.

Slurm9945 passes the new native CPU and CUDA analytic suites, then stops at an
obsolete snapshot-test callback that forbids every eigensolve on a valid force
candidate. The new contract intentionally performs a physical fixed-point
probe without necessarily updating that candidate. This original test failure
is retained; sanitizer and molecular groups were not started. v14 changes the
snapshot test to provide the fixture's independent analytic physical frame and
assert one probe per spin, zero density updates and unchanged identity. Its
corrupt-generation/provider-failure cases still forbid any probe/correction.
Production code is unchanged from v13; neither frozen report is overwritten.

Slurm9946 completes the v14 qualification: native CPU/CUDA analytic fixtures,
retained snapshot lifecycle, occupied response, zero-error CUDA memcheck, and
138 Python tests all pass. The 23 molecular configurations retain 92 complete
endpoints. The 40 original-density1e-10 endpoints have maximum energy/force
errors2.274e-12 /2.5614921642103106e-11; the 52 tight-density1e-12 endpoints
have maxima2.387e-12 /4.330e-13. Both groups use the unchanged3e-11 gates.
Historical v12 original-request failures and the v13 snapshot-test failure
remain preserved. These results qualify the changed small-molecule candidate,
not the remaining original #206 matrix or the changed large/streamed boundary.

The CPU-only saved-sample audit verifies every loaded library/source identity,
all four phases and all arrays against the saved qualification records. After
the first physical final-Fock record, traces contain80 final provider solves
across the40 original-request endpoints (range1..4 per endpoint), and146
across the52 tight-request endpoints (range1..6). Physical Pulay construction
uses120 /160 GEMMs respectively, exactly two per spin per endpoint. These
component records count real work but do not separate fixed-point probes from
promoted corrections because progress tracing was disabled in this run. That
split must be recorded explicitly in subsequent larger diagnostics; do not
infer a complete graph-replay ledger or clean timing from these records.

Required next evidence includes independent OH/water native endpoints, rank
deficiency and crossing, stale views, cold/warm/changed geometry, constrained
budgets, UHF and representative larger shapes. Record actual source reads,
matrix products, borrowed/owned memory and complete latency before any
performance conclusion. Original failed #206 clean cells remain preserved.

### v14 larger/batch qualification and independent residual boundary

Slurm9947 completes the predeclared dense/packed192-AO B1,96-AO B4 and192-AO B4
matrix at both density1e-10 and1e-12, with a finite15-minute allocation. All
144 native endpoints pass the unchanged3e-11 energy/force gates and the host
final-state work audit. Each request group has72 endpoints. Original-request
maximum energy/force errors are6.139e-12 /2.5594859565103434e-11; tight-request
maxima are6.139e-12 /1.6833825999817975e-12. Cold, warm, changed and changed-warm
arrays and exact retained densities are saved for every batch item.

Original-request finalization executes158 physical Focks/provider solves,
86 density corrections,80 fixed-point checks and8 promoted probes. The tight
request executes164 Focks/solves,92 corrections,82 checks and10 promotions.
Each group performs144 physical-weight GEMMs. Provider work satisfies
corrections +checks -promotions, while checks equal endpoints +promotions;
promoted probes never add a duplicate solve. Resource diagnostics repeat
bucket ownership for each item; the audit deduplicates those reports. The
largest reported bucket peak is3,237,007,445 device bytes. This is a planner
resource report, not a measured complete-process GPU high-water mark. Separate
response scratch/borrow records and all intrusive traces remain retained.

The CPU-only audit independently reconstructs F[D] and its occupied projector
using PySCF2.14.0/libcint at every saved native density, with no SCF retry or
correction. It retains144 endpoint rows and48 distinct density/geometry states.
All72 original1e-10 endpoints pass the same requested physical residual gate:
maximum commutator8.114027917442701e-11 and maximum projector defect
6.257594442615755e-11. However,24 tight1e-12 octamer endpoints fail independent
commutator and/or maximum-projector gates, despite passing native energy/force
and native stationarity checks. The independent maximum commutator is
1.0794698468430397e-12 and maximum projector defect1.4492851363456793e-12.
These are failed qualification gates; do not describe the entire matrix as
qualified or relax them because the energy/force arrays pass.

A further CPU-only diagnostic preserves those failures and compares Cholesky
versus eigen metric fitting and generalized versus symmetric-overlap
eigenvectors at four unchanged saved octamer densities. The auxiliary metric
retains all928 directions and has condition8.719e6; overlap condition is196.8.
Changing metric decomposition shifts physical F by up to5.33e-13. Both metric
decompositions and both overlap eigenproviders still give tight changed-state
projector defects above1e-12 (approximately1.23e-12 through1.46e-12). This does
not identify the native/independent discrepancy fully and is not an alternate
admission path. Next diagnosis must compare current native and independent
Fock/overlap/projectors at the exact saved densities, isolating integral/metric
arithmetic from final-state selection. Unchanged endpoint reruns cannot erase
the recorded failure, and earlier v12 constrained-memory results cannot
qualify the changed v14 force-state algorithm.

### Mainline priority after user steering

The user explicitly asked that the stricter residual discrepancy not become a
long-term blocker for the main effort. Keep the1e-12 independent failures as a
separate numerical follow-up while advancing #206 at its original requests and
unchanged case-specific paired energy/force gates. This changes work priority,
not the recorded verdict or the requirements for claiming broader accuracy.

Before that steering, Slurm9948 attempted an exact-density native matrix capture.
It failed during plan construction because the diagnostic supplied a nonzero
occupied rank capacity to the dense compatibility constructor. The loaded v14
identity is verified, but no native Fock comparison was completed. Preserve this
failed diagnostic and defer repairing its harness; do not spend another GPU run
on it before the mainline comparison.

The branch fast-forwards to upstream e3ea91d (#431), which adds historical
evidence/documentation only; every modified native source still matches frozen
v14. Slurm9949 starts19 predeclared stock comparison cells with seven interleaved
clean repetitions per engine, under a finite30-minute limit. This covers the
original96/192 B1/B4 energy/force points,384/768 energy/force, OH/ammonia and five
explicit cc-pVDZ/JKFIT force points. No diagnostic trace hooks or build/profiler
process may overlap clean timing. Raw failures and each domain's original gates
remain visible; this warm campaign alone cannot close #206 or qualify direct
paths, low-memory execution, or repeated cold/changed performance.

### Review scope and completed mainline comparison

Slurm9949 completed all 19 cells and exited 1 because three cells failed their
original paired numerical gates. The CPU-only audit verifies the frozen library,
native source, full retained ranks, normal cuTENSOR provider, seven sample pairs,
measurement order and unchanged gates. Predeclared relative MAD <= 3%, median
reduction >= 2%, and a paired bootstrap 95% lower bound > 2% give 11 warm wins,
3 warm losses, 1 variability-limited result, 1 tie/inconclusive result and
3 failed numerical cells. All original equal-basis 96/192 B1/B4 energy/force
cells pass and win; 384 loses, 768 energy is variable and 768 force is a tie.
Practical 96 B1 passes but loses; practical 96 B4 and 192 B1/B4 remain failed.
OH and ammonia win in the tested domains. This is ordinary complete warm
latency, with engine-local frozen seeds and normal convergence work, not a
matched-work claim or a pooled statement that VibeQC exceeds GPU4PySCF.

The cached independent CPU reference has exactly the practical 192 B1 geometry
and basis records. Across all seven saved warm outputs, native force error is
at most 7.524e-13 while stock error reaches 1.4375e-10. This diagnoses that
particular paired failure without changing its failed verdict. B4 geometries
are scaled, so the cached identical-geometry reference cannot qualify them.

The review bundle under `benchmarks/results/issue206-stable-response/` retains
the complete warm samples, qualifiers, controller, audit, native qualification
receipts, independent residual reports and historical failures. Full local
arrays and traces remain retained by the immutable artifact manifest. No
library or build products enter Git. A final comment corrects the bounded
reader's source-regeneration description; production code otherwise matches
frozen v14 byte for byte, and the exact comment-only delta is retained.

## Consequences and revisit conditions

This numerical repair adds conservative handwritten response/BLAS adapters.
The ownership comparison against e3ea91d records scientific +578/-40,
oracle +5/-0, and runtime +9/-0 noncomment CUDA lines, with no unchanged-line
reclassification. Thus all handwritten scientific roles total +583/-40.
Generated integral and derivative capabilities are unchanged. Retain the
rank-deficient spectral map, bounded source routes and scalar/serial oracles
until a generated contraction plan covers these operations, borrows, lifetimes,
metric subspace motion and work counts at the same independent gates.

Keep the strict 1e-12 independent octamer residual discrepancy as a separate
follow-up; it is not an original-request mainline prerequisite. Remaining #206
acceptance includes repeated cold/changed performance, direct-path failures,
exact-geometry independent residual coverage, complete work/memory accounting,
and v14 qualification of the larger constrained-memory boundary. Small v14
128-MiB cases and older v12 192-AO results do not establish that last boundary.

PR433 integrates upstream d756162 (#425 ECP host-grid generation) after its
submission. The only conflict is the generated current CUDA ownership snapshot;
regeneration preserves both subsystems. Executable DF sources and all frozen
benchmark arrays remain unchanged. Ownership delta against this new base is
identical, and 96 CPU benchmark/ownership/ECP IR tests pass after integration.
Issue434 tracks the deferred strict residual discrepancy independently.
