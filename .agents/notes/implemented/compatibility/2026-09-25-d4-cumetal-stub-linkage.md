# Decision: give D4 CUDA host stubs linkable implementation symbols

Status: implemented; Apple-GPU execution qualification remains required
Date: 2026-09-25

## Problem

PR #1245 compiles the existing production D4 CUDA implementation into a standalone CuMetal benchmark. In Actions run 36050734790, job 107807969400, the production build/runtime prerequisite passed but the benchmark failed to link all seven D4 launch stubs. Their definitions had anonymous-namespace internal linkage.

The pinned CuMetal revision `8a1434ffc1a4f7a17d2f8a41e157afbac22af171`, `compiler/cumetalc/main.cpp`, parses host launch stubs and emits their registration in a separate translation unit. `native_registration_source` references their addresses using external assembly-symbol declarations. Those declarations cannot resolve internally linked definitions in the host object.

## Decision and invariants

Use one named `vibeqc::dft::dispersion::d4_cuda_detail` implementation namespace and import it only inside the existing public launch wrapper. This changes linkage, not D4 equations or launch order. Every kernel body, input/output buffer, resource charge, precision, error path and pre-timing energy/gradient/charge-response acceptance gate remains unchanged. The detail symbols are not a new supported public API.

Do not copy D4 mathematics into a benchmark-only implementation, substitute a proxy or CPU backend, remove the numerical check, or relax the tolerance to obtain performance telemetry. A linked executable still must pass its existing independent host comparison before emitting READY.

## Evidence

A Linux Clang 17 CUDA-host-only probe reproduces local anonymous-namespace launch stubs versus global named-namespace stubs. `test_d4_cuda_stub_linkage.py` extracts the seven actual production declarations and namespace, compiles host stubs without a CUDA toolkit, and links their addresses from a second object. The original implementation scope fails that positive test; the named scope passes. A forced anonymous-namespace negative control continues to fail linking as expected.

This isolates symbol linkage using empty kernel bodies and dummy argument types. It does not test CUDA/Metal arithmetic, argument-layout compatibility, synchronization, the complete D4 numerical suite or timing. The source-matched CuMetal CI benchmark must provide those execution results; a host-linkage pass is not an Apple-GPU pass.

## Revisit when

CuMetal can register internally linked launch stubs without external cross-object references, or the implementation ownership changes. Preserve one production scientific body and actual numerical validation before benchmark publication.

Agent: ChatGPT (Odd-PR Review)
Model: GPT-6 Astra Pro
