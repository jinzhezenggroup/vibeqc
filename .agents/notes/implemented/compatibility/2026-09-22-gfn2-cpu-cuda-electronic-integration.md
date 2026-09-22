# Decision: retain CPU and CUDA electronic contracts during integration

Status: implemented
Date: 2026-09-22

The CPU electronic cutover and the CUDA pair cutover independently added the
same runtime-program module and test filename. Choosing either conflict side
would remove the other production consumer's builders and regression tests.

Keep the CPU population, core-energy, sequential Hamiltonian and integral-VJP
builders alongside the CUDA full-pair primal and reverse-AD builders. Share the
identical scalar-input constructor, retain both version/identity namespaces,
and retain both complete test families in the existing test module.

The CPU sequential accumulation and CUDA pair graph have deliberately distinct
interfaces. Neither is rewritten into the other during this integration: scaling
before products and potential sums is required to preserve finite-range behavior.
All pre-existing builder and test-function ASTs are checked unchanged on both
sides. Generation and numerical regression tests validate their common module.
No new numerical, capability or performance promotion is implied.

Agent: ChatGPT (Odd-PR Review)
Model: GPT-6 Astra Pro

## Subsequent CN/repulsion integration

After the first merge, #905 landed the independent CUDA CN/repulsion pair
consumer. Retain both the common pair-codegen dependency and the electronic
pair-codegen target. Union the exact source-adaptation entries and keep their
manifest gate sorted. No scientific source or emitter is changed by this
second conflict resolution.
