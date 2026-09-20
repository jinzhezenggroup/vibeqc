# Workload/device-aware DF resource policy (#598)

## Problem

The public DF memory value uses zero to request the implementation default, but the force-response path interpreted that as a fixed 128 MiB allowance and positive force budgets were mechanically split 50/50. That made zero insensitive to the workload and device, could not describe the actual ownership split, and left replay/evidence without the resolved policy identity.

## Decision

DF preparation resolves one `DfResolvedBudget` from workload dimensions, the selected CUDA device memory envelope when available, and the public request. A positive request is the exact hard upper bound and is never enlarged. Zero uses a bounded workload target; a live CUDA probe reserves both absolute and fractional headroom, while probe failure/CPU use a deterministic conservative fallback.

Value and response ownership are proportional to estimated staged work with bounded fractions rather than an unconditional 50/50 split. The resolved contract is stored on `DensityFittingScfData`, so warm replay uses the same value/response identity and never re-probes merely because free memory changed. Existing source-backed DF, planner, DIIS, force bridge, progress journal and metric diagnostics consume that owner; no second cache or autotuning framework is introduced.

## Rejected alternatives

- Keep 128 MiB for zero and only change documentation: does not satisfy resource-aware default semantics.
- Probe CUDA free memory on every warm replay: nondeterministic cache identity and plan thrash.
- Add a global resource-policy service: duplicates existing CUDA ownership and the narrower DF preparation contract.
- Continue 50/50 ownership: hides workload differences and prevents policy provenance from explaining actual value/response allocations.

## Invariants

- DF mathematics, metric thresholds, pair storage, exchange policy and numerical tolerances are unchanged.
- Positive public budgets are hard upper bounds.
- Warm replay reuses the prepared resolved contract.
- Probe failure is deterministic and never guesses live free memory.
- OOM remains a bounded transactional failure; no CPU/reference fallback is added.
- CUDA live probing stays inside the existing DF CUDA owner.

## Evidence

The synthetic policy test covers explicit hard caps, energy/force ownership, small/large workloads, deterministic no-probe fallback, live constrained/roomy envelopes and infeasible tiny force budgets. Native replay tests bind the generated response budget to the prepared owner. `benchmarks/df_policy_endpoint.py` retains resolved policy fields together with DF-plan and process peak-memory evidence; final CPU/CUDA qualification is recorded in the PR and repository checks.

## Subbudget provenance review

Partition the already resolved envelope rather than re-entering automatic
selection with a remainder. The earlier sequence reset requested_bytes and
reserved_headroom_bytes: a live automatic envelope reserving 2 GiB became a
subplan reporting no reservation. The subbudget helper preserves parent request,
probe and headroom identity, and rejects exhaustion before subtraction. It
never re-probes CUDA. A compiled host regression covers energy/force, explicit
and automatic requests, and exhausted or overdrawn remainders.
These host checks do not supply the still-pending #598 real-device default-policy
qualification. No new endpoint timing or whole-device memory claim is made.

Agent: ChatGPT
Model: GPT-6 Astra Pro
