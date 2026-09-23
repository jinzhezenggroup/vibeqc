# Decision: share the first streamed DF J/K raw-source traversal

Status: proposed (96-atom endpoint passed; promotion gates pending)
Date: 2026-09-23

## Problem

The accepted #1111 schedule generates one full raw tensor for occupied K in
the two-row-block 768-AO / 3712-auxiliary domain. Streamed J independently
generates another full raw tensor for its charge and a second one for the AO
output. Thus an admitted occupied J+K build still requests three raw tensors
before retries, final physical Fock and force response. The 96-atom energy and
analytic-force endpoint remains uncompleted under its original value budget.

## Candidate decision

Extend the compiler's two-slot triangular projection visitor to identify only
the first projection of each outer AO-row block. While its raw tile is live,
native code accumulates the density-weighted auxiliary charge, then performs
the existing occupied K projection. Each AO row contributes exactly once,
including ragged blocks. After K finishes, J transforms this complete raw
charge with the original full-rank metric eigensystem and generates only its
second, output-side raw pass. No source mathematics, density, metric cutoff,
SCF tolerance, final-state factor or response path changes.

The candidate is admitted only for singleton RHF, a validated occupied or
exact seed factor, a full-rank streamed generated source, triangular K, the
existing factor-first J capacity and at most two K row blocks. It uses the
already-charged raw, metric, and two retained-projection buffers without a
new allocation or response-budget loan. The candidate remains off by default
until independent GPU acceptance: `VIBEQC_DF_JK_SHARED_SOURCE=1` selects the
admitted shared route, while `0` selects the original independent J/K route
for matched ablation on a separate prepared owner. Changing this environment
variable between captured replays is not a valid comparison.

## Rejected alternatives

- Retaining a complete raw tensor exceeds the original value allowance.
- Whitening each raw output tile before J charge would add repeated metric
  products and change the established two-vector J contraction strategy.
- Sharing private K eigendirection projections with final physical K or
  force-response consumers would violate their distinct projection leases.
- Broadening the policy to more than two row blocks without endpoint and
  work/memory profitability evidence risks trading source savings for launch,
  copy, whitening or GEMM overhead.

## Invariants

The existing dense, resident, packed, UHF, batch and unqualified-streamed
routes remain unchanged. K must finish using both occupied projection slots
before J borrows either slot for its output pass. The source-to-charge path
must preserve AO-pair order and the original J metric transform. A new SCF
capture or changed geometry must not borrow previous geometry or factors.
Keep seed/iterative/retry/final-physical/force-response counts separate.

## Evidence and remaining acceptance

The compiler's independent host visitor test verifies each AO row is charged
once across divisible and ragged schedules, and that callback errors stop
without further source work. At the observed n=768, a=3712, rank=160 and
four 768*768*579-double buffers after automatic factor reservation (or 580
before reservation), the emitted two-block visitor predicts one
K tensor plus one J output tensor: 4,378,853,376 generated values versus
6,568,280,064 for the previously independent three-pass path. These are
source-work predictions, **not complete endpoint or latency measurements**.

The exact-head Release sm_120 library (`1e604c41`, SHA-256
`6bd481f295cb1874ee6b2877886aa9c65f52ca437ab604bd052675acac72ee0d`)
passes the three host policy tests and, in Slurm job 11474, all four independent
PySCF cold/warm/changed-geometry energy and force cases on both switch settings.
The earlier fast-compile library also passes all four cases (job 11473); only
the Release binary is used for endpoint timing.

Slurm job 11475 ran clean, separate-process energy-only endpoints for 24 atoms,
192 AOs, 928 auxiliaries, a 256 MiB public DF allowance and two warm samples
per arm. The original (`0`) and candidate (`1`) controls respectively took
15.2955/15.3061 s cold and 5.0852/5.0863 s warm median, both with 17 cold and
three warm VibeQC iterations and at most 2.28e-13 Eh error against matched
GPU4PySCF. The job 11476 diagnostic trace shows **three** K row blocks and
256 generated K rows per uncached build (45,613,056 raw K values); J separately
generates 68,419,584 raw values. No `ri_jk_shared` operation occurs on either
arm. Therefore the nearly identical timing verifies the explicit bounded
fallback, not shared-source profitability or a 24-atom speedup.

At a 320 MiB allowance, job 11478 exercises a streamed two-block candidate
on the same 192/928 case. Separate clean controls (`0` then `1`) take
12.1292/12.0191 s cold and 4.1992/3.0590 s warm median. Both pass the
independent energy gate within 2.28e-13 Eh, but cold SCF iterations differ
(17/16) and warm iterations differ (4/3): **these timings do not isolate a
per-work speedup**. Job 11480, traced separately, confirms 192 K source rows
and 192 charge rows in the candidate. Each uncached J/K requests 68,419,584
raw values on the shared route versus 102,629,376 on the independent route.

The original 96-atom, 768-AO / 3712-auxiliary energy-plus-analytic-force
request **completes** in job 11477 with the original 21,421,977,600-byte
public DF allowance and shared source enabled. Its clean native execution
takes 690.670 s (23 SCF iterations), returns -2431.2337038241376 Eh and
finite 96-by-3 forces, and passes the original 900-second deadline. Separately,
job 11479 obtains an independent GPU4PySCF result at the same orbital and
JKFIT auxiliary snapshots, full-Fock policy and 1e-12/1e-10 convergence
tolerances. Absolute errors are 1.37e-12 Eh and 1.51e-10 Eh/Bohr, passing the
unchanged 1e-9/1e-8 energy/force gates. The reference's shorter 58.48-second
full solve is a distinct engine/SCF branch, not an iteration-matched speed
comparison with native execution.

The separately traced, deliberately 180-second-limited Slurm 11481 run
completed eight `ri_jk_shared` operations before stopping inside SCF. Each
reports two K row blocks, 768 generated K rows and 768 charged J rows: one
2,189,426,688-value K raw tensor plus one equally sized J output tensor,
with 15 tile productions. The exact-head public resource policy resolves the
21,421,977,600-byte total into 13,685,173,124 value and 7,736,804,476
response bytes; the reported **value-plan** peak is 12,244,625,525 bytes.
This is planner accounting and an intrusive partial-SCF work trace, **not**
sampled whole-endpoint GPU memory or a clean timing run.

The identical-budget exact-head independent route (`VIBEQC_DF_JK_SHARED_SOURCE=0`)
in Slurm 11482 still had not returned an energy or force result when its
extended **1,500-second** bound expired. It also misses the original 900-second
deadline, but this incomplete off arm supplies **no** measured endpoint
latency or finite on/off speedup ratio. The matching library, geometry and
budget are recorded in its clean input receipt.

Separately, job 11483 completes the full traced 96-atom endpoint in 691.269 s
with the same energy, forces and 23 SCF iterations. Its valid operation records
count 24 shared K source passes plus 24 J output passes, the final-physical J
with two raw passes, seven final-physical K passes and one force-response pass:
**58 complete raw-tensor equivalents / 126,986,747,904 generated values**
across the whole endpoint. Force response independently owns 1,520,435,200
bytes of occupied projections, reports 1,980,523,536 bytes of response scratch
and reconstructs rather than reuses the final projection. One-second process
sampling within the allocated job gives 13,558 MiB maximum GPU memory across
673 samples, a **sampled lower bound**, not the allocator's high-water mark.
Tracing and GPU sampling make this run diagnostic; its similar duration must
not be substituted for the clean 690.670-second measurement.

The stricter 1e-13/1e-11 numerical ablation in job 11484 also passes energy
gates, but does not remove the branch mismatch: at 192/928 with 320 MiB, the
off/on cold solves take 25/35 iterations and the warm solves take 7/9. Neither
this pair nor the original-tolerance pair establishes an iteration-matched
endpoint speedup.

At 384 MiB, Slurm 11485 and interleaved repeat job 11486 provide **three
separate clean cold energy-only endpoints per arm** at 192/928, all with 16
native SCF iterations and at most 4.55e-13 Eh error against independent
GPU4PySCF. The independent/shared cold medians are 13.654839/10.739714 s,
a 21.35% reduction for this complete prepared-batch energy execution, not a
cross-engine speedup. Separately traced job 11487 confirms two K row blocks,
192 generated K rows and 192 shared charge rows: each uncached J/K requests
102,629,376 values on the independent route versus 68,419,584 with the
candidate. The warm arms still take two versus three SCF iterations, so the
384-MiB warm timings do not support a matched-work speed claim.

Pending: matched-work warm endpoint profitability and an allocator high-water
measurement if required for resource promotion. Keep the bounded fallback and
default-off control until the broader acceptance gates pass.

## Revisit when

Real >2-block traces and matched endpoint ablations establish a complete
profitability model including whitening, GEMM, copies, launches and storage.

## References

- #1117, #1078, #1111, #682
- `.agents/notes/implemented/performance/2026-09-23-df-projection-slot-reuse.md`
