# Decision: qualify DF consumers by work and query only required device attributes

Status: implemented
Date: 2026-09-19

## Problem

After #480 makes the generated Rys manifest available independently of AO size,
the gradient bridge still chooses a generic consumer below 192 AO and binds
signature packets to a water histogram and equal orbital/auxiliary dimensions.
Existing exact shell and packet implementations can therefore remain unused.
Blindly enabling them everywhere is also wrong: independent complete endpoints
show shell execution loses on single water, ammonia and OH.

## Decision

Separate the existing source/metric/state/resource gates from a small shared
work profile. The first sm_120 profile admits shell work at `N_AO²*N_aux >= 2^18`
and packets at `P_orbital²*P_auxiliary >= 2^22` with heterogeneous contraction
lengths. The comparison is overflow-safe. No molecule histogram, exact AO/rank
pair or equal-auxiliary rule enters these two decisions. Small and unknown
profiles retain fallbacks. The separate legacy packed-response boundary remains;
this is a bounded #444/#445/#435 slice, not completion of all three issues.

Read compute capability with two `cudaDeviceGetAttribute` calls through one
shared helper. The first policy experiment queried the complete device property
record and added roughly 1 ms to small endpoints. A 200-repeat same-device probe
measured 1.047210155 ms for the full query versus 0.000112265 ms for the two
attributes. Avoid that query in both the bridge and each derivative panel.
This micro-measurement explains a host overhead; complete endpoints, not the
micro ratio, authorize performance claims. The discarded version remains in
the evidence alongside the final candidate.

## Invariants and alternatives

Generated mathematics, FP64 arithmetic, metric cutoff/rank/response, physical
force-state checks, force sign, and screening are unchanged. Full CUDA errors
propagate; no global device-ordinal cache is introduced. Existing explicit
controls remain diagnostic alternatives. Every changed geometry rebuilds its
own metadata. Allocations remain arena-owned and charged.

The conservative threshold intentionally leaves the 58-AO ammonia dimer generic
even though its explicit shell trial wins modestly. The 87/116-AO ammonia
holdouts and unequal cc-pVDZ/JKFIT workloads test generalization. A more aggressive
small-work threshold needs its own evidence; do not append a new molecule name
to the selector to recover that unpromoted win.

## Evidence and scope

`benchmarks/results/issue445-df-work-admission/` retains the qualification record,
source/library identities, unchanged-state replays, numerical gates, rejected
experiments and reproduction commands. CPU tests cover integer boundaries,
unknown profiles, environment restoration, evidence failures and runtime query
error propagation. Real-device tests exercise genuine auto selection, cold/warm
and changed geometries, batch four, practical auxiliary f shells and UHF fallback.

This does not implement screening (#437), response-factor fusion (#419), native
post-HF device providers, or a shared compiler-wide specialization layer (#459).
No GPU4PySCF superiority claim or automatic selection on an unmeasured target
architecture follows from this local profile.

Agent: ChatGPT
Model: GPT-6 Astra Pro
