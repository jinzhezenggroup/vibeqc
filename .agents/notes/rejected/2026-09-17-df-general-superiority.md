# Decision: do not claim general DF superiority or #206 completion

Status: rejected (general superiority and completion claim)
Date: 2026-09-17

Fresh matched stock GPU4PySCF measurements pass the 192-AO batch-1/4 ordinary
warm-force timing requirements. They do not establish general superiority:
384/768-AO energy and force endpoints regress, 192-AO cold is slower, and the
changed endpoint fails its existing numerical gate. Direct acceptance also
retains both strict 96-AO force failures and 192-AO CUDA cold-start errors.

Keep every original threshold and failed result. The tighter CPU diagnosis
shows that both 96/b4 DF results can individually be within 3e-11 of the oracle
while their pairwise difference exceeds 3e-11. Tightening reference convergence
is a separate new experiment; it cannot retroactively qualify original timing.

Engine-local frozen densities and equivalent priming preserve legitimate
convergence routes, which differ substantially. A displayed ordinary latency
ratio is not evidence of equal J/K work. Stock gradients normally release CDERI;
retaining those arrays solely for a benchmark would change the reference route.

Native and stock profiling both add fences. Keep raw CUDA transfer activity,
component scopes and unprofiled timing distinct, and do not certify process
memory from a run under Nsight. #409's storage benefit is independently scoped;
it does not close the external gates. #412's rejected Gram candidate remains out.

Revisit completion only after the named numerical/runtime failures and missing
matrix/resource coverage pass. The [current evidence](../../../benchmarks/results/issue206-current-df/README.md)
maps every original #5 acceptance criterion to retained evidence or open work.
