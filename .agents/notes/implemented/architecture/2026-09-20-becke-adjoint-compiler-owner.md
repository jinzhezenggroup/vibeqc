# Decision: emit the shared Becke reverse traversal from the compiler

Status: implemented
Date: 2026-09-20

## Boundary

The compiler emits the existing two-pass normalized-product reverse traversal
for CPU JIT, stationary CUDA JIT and build-time XC geometry. Scalar norm,
ratio, log and Becke partials remain Graph-generated; the traversal is explicit
compiler-emitted C++, not a newly invented scalar algebra or schedule.
Native owners retain scratch allocation, task mapping, streams and transactional
publication. The retired grid_response_adjoint.hpp is not a second live owner.

## Preserved semantics and evidence

Preserve scaled norms, frozen log normalization, exact-zero product handling,
clipping branches, atom/point motion, accumulation order and failure signaling.
The old complete header and emitted source are C++-token-identical after removing
only pragma once and ignoring comments/whitespace. CPU/grid-response and source
composition tests verify executable behavior and that generation requires no
public runtime, GPU, compiler invocation or reference package.

## Alternatives and consequences

Keeping a forwarding native scientific header would retain two ownership paths.
Generating a different recurrence during this move would mix numerical change
with ownership cleanup. Both were rejected. Artifact identities intentionally
follow the new source/asset closure; no performance or capability is promoted.
Revisit for a separately qualified traversal/schedule, retaining the independent
Python directional response oracle. Refs #349, #163 and tests/python/test_grid_response.py.

Agent: ChatGPT
Model: GPT-6 Astra Pro
