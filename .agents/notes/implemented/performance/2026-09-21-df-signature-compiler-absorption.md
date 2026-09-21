# DF signature scheduling: compiler absorption slice

Issue #385 was reopened after #386 because its bounded primitive-signature
packet scheduler remained DF-specific. This slice moves the reusable scheduling
contract into compiler/runtime-common ownership without changing the derivative
equations, screening, precision, convergence, or provider legality.

- HomogeneousTaskRange represents exact class + runtime signature + bounded work.
- HomogeneousTaskSchedule owns packet capacity, execution family, deterministic
  identity, and the shared GpuProfitability evidence record.
- the generated DF header now publishes compiler-owned packet capacity/execution;
- native packet layout and heavy-first/block-prefix planning use the shared
  runtime HomogeneousTaskPacket helpers;
- DF continues to own legal shell/signature construction and primitive work;
- a non-DF four-center test exercises the same compiler scheduling IR.

This is an absorption/refactor slice. It deliberately does not claim a new
endpoint speedup and does not close #385 by itself; target/live-range-driven
persistent/cooperative selection and complete #386 endpoint requalification
remain separate acceptance items.

Agent: ChatGPT
Model: GPT-5.6 Sol
