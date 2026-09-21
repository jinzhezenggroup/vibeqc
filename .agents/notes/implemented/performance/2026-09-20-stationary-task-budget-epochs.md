# Decision: separate task admission epochs from cumulative metrics

Status: implemented
Date: 2026-09-20

The stationary task owner reports cumulative primitive counts, but its finite
work cap bounds one force execution. Reusing the cumulative count for admission
made a legal second execution fail after reset; partial failed work likewise
consumed the next execution's allowance.

Reset now records the current cumulative count as an epoch baseline. Task and
nuclear admission use only work since that baseline, and cumulative uint64
counter overflow is rejected before launch. Metrics remain monotonic; neither
scientific work nor failed work is erased from the diagnostic history. No kernel,
precision, source order, workspace capacity or work cap is changed.

The host regression compiles the real Owner, guarded checks and reset/task/
nuclear host functions extracted from the runtime header. CUDA operations are
explicit stubs, so this is not GPU or numerical qualification. The original
header fails on the second legal reset/replay; the repair passes that case,
same-epoch excess, partial failure/recovery, nuclear counts and counter overflow.
Full integrated NVCC/GPU/endpoint-scaling acceptance remains separate.

Agent: ChatGPT
Model: GPT-6 Astra Pro

## Prepared-replay metric deltas

Task-descriptor and task-batch counters are cumulative owner metrics for the same
reason as primitive work. A prepared owner can serve multiple force executions,
so endpoint evidence must subtract the pre-execution values for these counters as
well. Reporting cumulative task counts next to per-execution primitive/transfer
counts would make repeated-force evidence internally inconsistent without
changing any scientific work. The prepared metric delta therefore includes both
`task_descriptors` and `task_batches`, with a regression covering a nonzero
pre-execution baseline.

Agent: ChatGPT
Model: GPT-5.6 Sol
