# Decision: share the first streamed DF J/K raw-source traversal

Status: proposed (CUDA endpoint qualification pending)
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
four 768*768*580-double buffers, the emitted two-block visitor predicts one
K tensor plus one J output tensor: 4,378,853,376 generated values versus
6,568,280,064 for the previously independent three-pass path. These are
source-work predictions, **not complete endpoint or latency measurements**.

Pending: exact-head Release sm_120 build; independent cold/warm/changed-
geometry energy and force gates on and off the candidate; actual raw-pass,
peak-memory and SCF/final-state work counts; clean matched endpoint timings
separate from tracing; and the original 96-atom bounded energy-plus-force
completion. If an admitted case regresses, keep the bounded fallback and do
not promote it as a speedup.

## Revisit when

Real >2-block traces and matched endpoint ablations establish a complete
profitability model including whitening, GEMM, copies, launches and storage.

## References

- #1117, #1078, #1111, #682
- `.agents/notes/implemented/performance/2026-09-23-df-projection-slot-reuse.md`
