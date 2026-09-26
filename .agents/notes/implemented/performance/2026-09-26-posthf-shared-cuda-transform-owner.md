# Decision: share one CUDA transform owner across post-HF MO batches

Status: implemented
Date: 2026-09-26

## Problem

The source-reuse boundary removed repeated AO integral scans, but native CUDA
`NativeBlockProvider::get_many` still constructed one stream, cuBLAS handle,
workspace and transform owner per requested MO block. Every AO source tile was
therefore copied H2D once per block, and each block introduced its own profiled
input/library synchronization boundary.

For RCCSD this duplicated the same raw AO-tile upload across up to seven
simultaneously requested blocks even after the blocks shared one source scan.

## Decision

Keep the single-block CUDA ABI unchanged for Python and compatibility consumers.
Add a native multi-request owner used by `NativeBlockProvider::get_many`.

One batch owns one CUDA context, stream, cuBLAS handle, explicit workspace and
provider allowance. Coefficient panels for all requests are uploaded once at
construction. Each AO source tile is uploaded once to shared device staging;
all first-axis transformations consume that resident tile before the scratch
buffer is reused. Remaining axis transforms and DAXPY accumulation execute on
the same stream and handle.

Validation scans every requested output before any download. All requested
outputs are then copied in one output-timed section. Failure before publication
therefore leaves every host output unpublished.

## Resource boundary

This slice deliberately keeps higher-level admission conservative. Existing
per-request CUDA capacities are still summed before execution. The shared owner
computes its smaller actual combined arena and rejects it if it exceeds that
already-admitted allocation ceiling. A later change may teach the compiler
resource planner the shared provider allowance, but this change does not spend
that memory reduction.

## Invariants

- scientific AO-to-MO equations and request order are unchanged;
- one source scan still supplies the same exact AO values;
- no full AO tensor is materialized;
- the single-block ABI and its publication semantics remain unchanged;
- batch validation is all-or-nothing before host publication;
- no wall-time speedup is claimed without an allocated-GPU endpoint run.

## Validation

Static regression coverage requires one raw H2D copy and one profiled
input/library/output section per native CUDA batch. Fault-injected host CUDA
stubs execute the batch publication path and verify that two valid outputs
publish together while a nonfinite batch publishes neither output. Existing
MP2/RCCSD CUDA suites remain the numerical and lifecycle gates.

References: #1401; #1411; #1415; #1419.

Agent: ChatGPT
Model: GPT-5.6 Sol
