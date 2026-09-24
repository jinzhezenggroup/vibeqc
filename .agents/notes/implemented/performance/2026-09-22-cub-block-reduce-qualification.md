# Decision: retain generated CUDA as the default reduction provider

Status: implemented
Date: 2026-09-22

## Problem

The opt-in TensorIR CUB BlockReduce provider from #980 had real-device numerical
and compile-resource coverage, but no matched complete-endpoint comparison with
the generated cooperative reduction. PTXAS improvements alone cannot justify a
production-provider promotion.

## Decision

Keep `vibeqc.generated_cuda` as the production reduction default. Retain CUB as
an explicit opt-in provider. Qualification uses the existing TensorIR tuning
path so both implementations run identical FP64 mathematical work and include
caller validation/staging, H2D, reduction, D2H, and detached output allocation
in every timed endpoint.

Lowering diagnostics are canonicalized by candidate identity before their
shared provenance identity is computed. Provider enumeration order therefore
cannot change `ScheduleContract` identity for the same candidate set.

## Rejected alternatives

- Do not promote CUB from its lower register count or isolated kernel timing.
  Those are resource/component facts, not complete-endpoint evidence.
- Do not treat source-size reduction as compile-cost evidence. The measured CUB
  source was 107 bytes smaller but compiled more slowly.
- Do not add a CUB-specific profile cache or selector. Any future promotion must
  continue through the shared schedule/profile contracts.

## Invariants

- Generated CUDA remains the legal default and fallback.
- CUB admission requires explicit typed-true CCCL header capability evidence.
- Generated and CUB comparisons use the same precision, equation, input values,
  semantic work, and endpoint boundary.
- Numerical correctness and performance promotion remain separate decisions.
- Raw compilation resources must come from PTXAS; spills are never inferred from
  static planning estimates.

## Evidence

Source revision `104723008455ea6169d89d9fdfbc6625b0e46138` was transferred as a
verified Git archive with SHA-256
`4ecf5bd75eefafc003067f74545b9e507859e528e30903daaf3b6ce3e3d970d2`.
Before compilation, the runner matched the archive's PAX Git revision and
verified 6,250 extracted files totaling 130,781,880 bytes against the current
execution tree. It also cross-checked the `nvidia-smi` UUID against the CUDA
runtime probe.
The finite qz Job ran on one NVIDIA H200 (`sm_90`, driver 570.124.06, CUDA
runtime/toolkit 12.8) using two deterministic `65 x 4097` FP64 fixtures, one
contiguous and one strided, with eight ABBA samples per implementation and
fixture.

- The independent TensorIR CPU interpreter gate passed with maximum absolute
  error `9.094947017729282e-13` at `atol=1e-11`, `rtol=1e-10`.
- Generated CUDA: 9,452 source bytes, 2.232284 s compile time, 1,063,888 binary
  bytes; the reduction kernel used 24 registers, 32 shared bytes, zero stack and
  zero spills.
- CUB: 9,345 source bytes, 2.732225 s compile time, 1,068,592 binary bytes; the
  reduction kernel used 18 registers, 48 shared bytes, zero stack and zero
  spills.
- Both arms reported 2,131,712 owned device bytes, 4,397,840 predicted peak
  bytes, and identical 6,392,360 semantic traffic bytes.
- Complete-endpoint median speedups were `1.002238x` and `0.999735x`; bootstrap
  lower bounds were `0.991782` and `0.990805`. Neither met the `1.02x` gate or
  the shared noise gate.

The reviewed record is retained under
`benchmarks/results/issue971-cub-block-reduce-h200/`. The candidate is rejected
for performance promotion, not for numerical or execution correctness.

## Consequences

Slice C now has a reproducible real-device qualification path and negative
promotion evidence. CUB remains available for explicit experiments without
changing existing AOT behavior or numerical policy.

## Revisit when

Reconsider the default only when representative larger/legal reduction domains
on supported production targets pass the same independent numerical gate and
show a repeatable complete-endpoint gain above the shared noise threshold. A
compiler/toolkit change that materially alters CUB code generation also warrants
requalification.

## References

- #971
- #980
- `benchmarks/tensor_cub_qualification.py`
- `benchmarks/results/issue971-cub-block-reduce-h200/publication.json`
