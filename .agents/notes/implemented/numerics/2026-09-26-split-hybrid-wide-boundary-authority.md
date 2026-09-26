# Decision: use independent wide Libxc at unstable M06-2X boundaries

Status: implemented; full endpoint qualification still pending
Date: 2026-09-26

## Problem

The earlier split-hybrid follow-up required the generated empty-spin M06-2X
derivatives to agree with both compiled binary64 Libxc and independent
113-bit Libxc source algebra at the *same* inputs and tolerances. At the
empty-beta point, binary64 Libxc reports `6.589350853277056e16` for
`vsigma_bb`, versus `6.589321868264429e16` from the 113-bit source. Their
distance is 14.663 times the original tolerance, more than seven times the
sum of the two allowable error intervals. No generated result can pass both
gates: this is a mathematical contradiction, not a defect to optimize away.

## Decision

Keep both full Libxc binary64 and wide evaluations in the report. Require
the *unchanged* binary64 tolerances for all original nonempty-spin rows. At
both original exact-empty points, and immediately adjacent majority-density
floats in both directions, require the same tolerances against independent
113-bit source algebra instead. The binary64 discrepancies at the original
empty points remain an explicit failed diagnostic, never a hidden skip or an
enlarged tolerance. No method receives a qualification flag until the wide
gate and the independent complete-endpoint reference both pass.

## Rejected alternatives

Requiring the generated output to match both disjoint intervals is
impossible. Increasing the tolerance to fit rounded Libxc would also admit
the source cancellation error; accepting whichever neighboring binary64
value is closest would admit one-ULP sensitivity without scientific proof.
The targeted direct-fraction rewrite did not pass the wide gate and was
reverted. The established r2SCAN wide-oracle boundary precedent is recorded
in `2026-09-23-scan-stable-spin-fractions.md`.

## Evidence and revisit when

The original 50 points, six wide boundary points, source hashes and both
backend outputs are retained under local ignored artifacts. At the
empty-beta point, the pre-repair generated host error was 3.994 times the
wide gate; on the adjacent majority double below it, 33.319 times. Slurm
RTX 5090 reproduced the latter. The subsequent Stoll cancellation repair is
described in `2026-09-26-split-hybrid-stoll-conditioning.md`. Revisit only with
independently validated conditioning or a superseding domain/reference policy. This decision
supersedes the incompatible simultaneous-reference requirement in
`2026-09-26-split-hybrid-work-boundary-followup.md`.
