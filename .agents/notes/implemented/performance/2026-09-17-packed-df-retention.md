# Decision: retain exact packed DF values as an explicit option

Status: implemented
Date: 2026-09-17

## Decision and evidence

Retain `VIBEQC_DF_VALUE_STORAGE=packed`; unset and `auto` select dense. This
completes #409 Phase A with measured endpoint domains, without automatic
promotion or numerical screening. The [original proposal and experimental
history](../../proposed/2026-09-17-packed-df-values.md) preserve earlier pending
states and discarded projection variants; this decision supersedes them.

Seven clean interleaved pairs show a useful 192-AO warm-force improvement and a
768-AO/12-GiB complete-force reduction from 162.023277 to 65.028560 seconds.
The latter has five versus six SCF updates and is ordinary converged latency.
Separate unprofiled diagnostics reduce native/device peaks by 51.4%/33.3%, while
host peak rises 17.2%. Both B plans are resident. The result does not demonstrate
a streamed-to-resident crossover.

Large warm and changed-geometry regressions rule out broad automatic selection.
Corrected final states can lose occupied response borrowing: the measured
768-AO changed packed fallback expands response work to 77 auxiliary blocks.
An independent CPU oracle confirms the failed 384/1856 unequal force gate is a
native discrepancy. That domain stays excluded; 24/116 and 96/464 retain their
passing unmodified physical bases.

Sixteen warm/cold CUPTI profiles retain actual transfer bytes and graph-node
work, separating native tracing fences from other observed synchronization.
Hardware DRAM traffic and raw-integral recurrence FLOPs remain unmeasured.
The final formatted source passes both native suites, 17 molecular/neighbor
cases, 15 resource cases, sanitizer checks and a fresh seven-pair 192-AO endpoint.
The final binary identity remains distinct from the frozen campaign identity.

## Invariants and revisiting the decision

Keep raw A and whitened B as separate immutable owners, unit-weight lower pairs,
direct generation/whitening, and checked simultaneous scratch. Corrected/stale
factors must retain exact bounded fallback instead of reusing an invalid lease.
The Python resource inventory's existing size limit is unchanged. Public dense
and nonsymmetric tensor semantics remain explicit.

Revisit automatic selection only with complete endpoint evidence in the new
domain, including response fallback and geometry rebuild. Do not use smaller
factor size, a microkernel win, or relaxed force gates as admission evidence.
Retire the handwritten gather when a compiler-owned equivalent passes the same
correctness/resource/endpoint requirements. #412 split Gram is retired; no
combined saving or #206 external superiority follows from this decision.

## Reproduction

The [evidence bundle](../../../../benchmarks/results/issue409-packed-values/README.md)
contains raw clean samples, work/transfer accounts, failed admissions, source
reconstruction and final validation. Use its linked portable runner for a fresh
measurement; archived workstation paths describe provenance.
