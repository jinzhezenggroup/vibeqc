# Decision: reuse the generated DF source for CUDA post-HF consumers

Status: implemented
Date: 2026-09-20

## Problem

`CudaDFSource` generated three-center values on the GPU but its public tile
adapter always copied them to host. CPU `DFProvider` consumers legitimately
needed that behavior, but using the same adapter as the basis of a CUDA
performance path creates a GPU→CPU→GPU staging risk. The CUDA response backend
also prepared a second generated integral source from the raw molecular source,
so source setup/identity was duplicated rather than handed directly to the GPU
consumer.

## Decision

Keep the host-staged route as an explicit compatibility/oracle contract and add
a one-way device-resident provider handoff. When `CudaDFJKBackend` receives a
`CudaDFSource`, the already-prepared `CudaDensityFittingIntegralSource` is
transferred into the existing CUDA DF J/K plan. The plan therefore reuses the
same generated-integral implementation, metadata and bounded DF resource
contracts; no new integral evaluator is introduced.

The handoff consumes the generated source because the existing native plan API
already has a transfer-of-ownership contract. Raw tile reads from that
`CudaDFSource` fail after handoff. Independent host oracles use a separate
source. Multi-stage consumers that must read the same source after response
explicitly select the re-prepared device-resident route instead; it uses the
same generated integral implementation without transferring caller ownership.
`NativeSource` naturally follows that re-prepared route as well.

## Rejected alternatives

- Do not copy generated tiles D2H and then upload them again to a GPU
  transformation. That preserves an avoidable transfer proportional to
  generated value volume.
- Do not add a second post-HF integral implementation. The SCF generated DF
  source already defines shell mapping, generated Rys math, tiling and resource
  ownership.
- Do not silently turn `DFProvider` into a CUDA path. Its NumPy transformations
  and host cache are valuable as an independent compatibility/oracle surface.

## Invariants

- Host-staged and device-resident execution paths are named in provenance.
- Reused and re-prepared CUDA paths report zero raw DF D2H and zero raw/derived DF H2D.
- Density input and final J/K output transfers remain explicit and counted.
- Complete RI-MP2 gradient explicitly re-prepares response state so its caller
  source remains available for the later relaxed-gradient contractions.
- Generated-value bytes and tile counts are counted at the native DF source,
  so plan setup/materialization and streamed replay are both visible.
- Metric identity, threshold and numerical response gates are unchanged.
- The plan/source lifetime is single-owner after handoff; no raw source access
  is permitted afterward.

## Evidence

Real-device tests exercise the original host-staged DF/MP2 oracle and the
device-resident response path. The response test compares the matrix-free CUDA
operator and linear solves against an explicit DF matrix constructed from a
separate source, while asserting zero raw DF transfer counters and nonzero
generated work. Complete source and J/K endpoint timings are reported alongside
transfer and generated-value counts. The real-device H2/LiH/H2O records and
qualification are retained in `benchmarks/results/posthf-369/`.

Reproduce with:

```bash
PYTHONPATH=.:python VIBEQC_LIBRARY=$PWD/build/cuda/libvibeqc.so \
  VIBEQC_POSTHF_CUDA_TEST=1 VIBEQC_RESPONSE_CUDA_TEST=1 \
  python -m pytest tests/python/test_posthf_cuda.py \
  tests/python/test_response_cuda.py -q
```

## Consequences

A handed-off `CudaDFSource` cannot simultaneously serve a CPU raw-tile oracle
and that CUDA J/K consumer. Disposable response sources use handoff/reuse.
Multi-stage consumers that need the source later set `reuse_generated_source=False`;
that pays a second generated-source setup but still avoids raw-value host staging
inside the CUDA response endpoint.

## Revisit when

Revisit if a future native source supports safe multi-consumer borrowing with
stream/lifetime fencing, or if RI-MP2/CC gains a native CUDA DF contraction that
can share the same provider-handoff protocol.

## References

- GitHub issue #369
- `src/posthf/df_bridge.cu`
- `tools/vibeqc_posthf/sources.py`
- `tools/vibeqc_response/backends.py`
- `docs/posthf.md`
