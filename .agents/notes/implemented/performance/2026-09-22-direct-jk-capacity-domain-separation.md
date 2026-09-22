# Direct J/K generated-task capacity domain separation

Agent: ChatGPT
Model: GPT-5.6 Sol
Date: 2026-09-22
Issue: #597
Regression source: #898

## Problem

#898 correctly moved Direct J/K resource decisions behind `CudaTargetInfo`, but
one resolved `generated_task_arena_maximum_bytes` was reused for two different
allocation domains:

- fixed-topology descriptors are long-lived prepared-state storage;
- bounded-streaming descriptors are reusable page scratch.

The fixed-topology policy reserves at most 1/32 of reported device memory. On
the RTX 5090 used for qualification, CUDA reports 33,666,498,560 bytes, so that
budget is 1,052,078,080 bytes. Applying it to the 192-byte
`GeneratedShellTask` page reduced the bounded capacity from 8,388,608 to
5,479,573 tasks.

For the 96-atom / 768-AO README topology this changes generated class paging
from approximately 339 to 508 page windows. The measured endpoint moved from
3.0089 s before #898 to 3.1881 s after it.

## Decision

Represent fixed-topology and bounded-streaming resources as distinct policy
types. They have independent profile ceilings, target-memory budgets, and
regression tests.

- fixed topology: profile ceiling 1 GiB, target budget 1/32 of global memory;
- bounded streaming: profile ceiling 8M tasks / 1.5 GiB, target budget 1/16;
- unknown targets retain conservative 256 MiB fallbacks for each domain;
- occupancy and stack policies remain unchanged.

The 1.5-GiB bounded ceiling is the memory required by the previously qualified
8M-task page at the current 192-byte task ABI. If the task ABI grows, the byte
budget automatically lowers task capacity rather than silently exceeding the
resource claim.

## Regression guards

Native policy tests use the actual RTX 5090 `totalGlobalMem` value and require
that the qualified target still resolves the full 8M bounded-streaming page.
A 2-GiB synthetic target must shrink both resource domains independently.

Two cross-domain tests enforce the architectural invariant:

1. tightening fixed-topology storage must not shrink bounded pages;
2. tightening bounded scratch must not alter fixed-topology admission.

The Direct J/K source-level guard also names both the fixed-topology admission
and bounded-streaming capacity resolver so future refactors cannot collapse the
two concepts without updating explicit tests.

## Performance validation

An isolated #898 A/B on the RTX 5090 changed only the bounded-streaming task
capacity while keeping the rest of #898 intact. The same 96-atom / 768-AO
README protocol recovered the warm energy-plus-force median from 3.1881 s to
3.0157 s, a 5.41% reduction. The pre-#898 observation was 3.0089 s, so the
recovered result is within 0.23% of that baseline.

All three VibeQC warm repeats used one SCF iteration, matching GPU4PySCF.
The A/B passed the numerical gate with maximum paired errors of 2.046e-11 Eh
and 2.444e-10 Eh/bohr. This isolates the regression to the accidental
bounded-page capacity shrink rather than to a scientific or convergence change.
