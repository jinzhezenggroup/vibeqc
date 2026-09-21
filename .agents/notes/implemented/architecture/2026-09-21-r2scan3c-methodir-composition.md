# Decision: compose canonical r2SCAN-3c above the native KS method selector

Status: implemented
Date: 2026-09-21

## Problem

The canonical `R2SCAN-3c` MethodIR already bound r2SCAN, the exact H-Ar
def2-mTZVPP basis, D4(BJ)-EEQ-ATM, and gCP, while each scientific component had
an independently qualified implementation. Public execution still stopped at
representation: the native KS registry had no reason to gain another named
scientific branch, and gCP had no production ABI consumed by Calculator.

## Decision

Keep r2SCAN-3c as a MethodIR composite rather than adding a new native method
ID. Calculator validates the exact canonical manifest, automatically binds the
pinned basis when omitted, removes post-SCF correction primitives, and sends
the remaining r2SCAN graph through the already-qualified `r2scan-rks` or
`r2scan-uks` owner.

Post-SCF execution is explicit. D4 uses the existing bounded
`D4CorrectionBatch` on the requested CPU/CUDA backend. The audited native gCP
kernel is exposed through a small C ABI and remains CPU-owned. A
`R2SCAN3CCorrectionBatch` evaluates both components exactly once and returns
their individual results plus their energy/gradient sum. CUDA electronic/D4
plus CPU gCP therefore reports a mixed `cuda+cpu` correction backend rather
than pretending the entire correction is device resident.

## Rejected alternatives

A new `VIBEQC_METHOD_R2SCAN3C_*` selector and a corresponding
`dft_method.cpp` branch were rejected because they would duplicate composition
already represented by MethodIR and make future `-3c` methods require named
scientific drivers.

Calling the Python gCP reference from production was rejected. The reference
remains an independent oracle; production calls the existing native audited
kernel through the stable C ABI.

## Invariants

- Canonical `R2SCAN-3c` identity is preserved; altered basis, D4, or gCP inputs
  require another explicit method identity and fail closed here.
- The first public domain remains H-Ar with the exact spherical def2-mTZVPP
  snapshot.
- Electronic r2SCAN, D4, and gCP are each evaluated once.
- Correction gradients are `dE/dR`; public forces subtract them from the
  electronic force.
- Mixed backend execution is inspectable in the returned correction result and
  diagnostic.
- The global resource budget does not silently claim ownership of the separate
  correction plan.

## Evidence

- `tests/python/test_r2scan3c_execution.py` compares the native gCP ABI against
  the independent simple-dftd3-derived reference.
- The same test checks D4 + gCP replay/component sums and Calculator total
  energy against an independently executed plain r2SCAN endpoint.
- An opt-in Slurm CUDA gate checks total public forces against plain r2SCAN
  forces minus the complete D4 + gCP gradient.
- Existing D3 Calculator composition tests remain part of the focused
  regression set.

## Consequences

No public native ABI method ID is consumed by r2SCAN-3c. The correction owner
is reusable at the Python/public MethodIR layer, while native KS continues to
select physics from the compiler-resolved electronic plan. gCP remains CPU
owned until a separately qualified device lowering exists.

## Revisit when

Replace the r2SCAN-3c-specific composite owner with a generic post-SCF MethodIR
correction scheduler when that scheduler can preserve component capability,
resource, failure-isolation, and backend diagnostics without reintroducing
method-name dispatch.

## References

- #172
- #396
- #577
- #747
- #767

Agent: ChatGPT
Model: GPT-5.6 Sol
