# Decision: propagate native XC scalar rows into the Vxc coefficient ABI

Status: implemented
Date: 2026-09-21

## Problem

#534 removed the DFT-feature to native-scalar-XC gather/stack/scatter boundary, but the next fixed-density potential stage still reconstructed the scalar derivatives into a dense seven-row feature-gradient matrix. The generic native coefficient adapter then copied that matrix and the density gradient, stacked the reachable variables, and copied the stack again through the immutable validation boundary. Those materializations were hidden behind provider-local APIs, so ProgramIR's physical `xc_rows` layout did not describe what the Vxc consumer actually wanted.

## Decision

For qualified polarized native CPU potential tiles, emit a second packed ABI from the existing scalar and coefficient graphs. The packed scalar producer writes row 0 as energy and rows 1: as the complete functional feature gradient, with inactive derivative rows set to exact zero. The packed coefficient consumer borrows those derivative rows and the existing DFT-owned contiguous density-gradient block directly.

The generic native point-function ABI remains intact for unqualified, response, geometry, unpolarized, and diagnostic routes. ProgramIR records `xc_rows` using the consumer-ready physical shape, and generated metadata records the physical root-row mapping; both therefore participate in deterministic identity.

## Rejected alternatives

- Merely annotate the copy as removable without changing runtime execution: this would repeat the descriptive-only gap that #831 is intended to close.
- Replace the generic point-function ABI globally: unrelated response/geometry paths do not need this layout and would take compatibility risk for no measured benefit.
- Overwrite the feature owner in place: Vxc still needs the feature rows after scalar evaluation, so donation is not legal at that boundary.

## Invariants

- Scalar and coefficient mathematics remain the existing generated graphs; there is no second XC algebra.
- Packed execution is admitted only for the already-qualified polarized fixed-density potential path.
- Generic execution remains a bounded fallback.
- Producer-owned arrays must be FP64, finite and C-contiguous before the packed native call.

## Evidence

Using the same Release CPU library, PBE polarized inputs, seven-point tiling, warmed construction and 101 interleaved measurements per route, the only A/B switch was whether `scalar_values_packed` retained its physical derivative owner or was converted to a plain dict to force the old downstream materialization path. Energy, electron counts and potential were array-identical before timing.

- H2 (32 points, 5 tiles): 5.983938 ms -> 5.583212 ms median, 1.07177x.
- f-spherical (32 points, 5 tiles): 6.224128 ms -> 5.845052 ms median, 1.06485x.
- `tests/python/test_program_ir_xc.py` plus storage tests: 33 passed with the local Release CPU library.
- Combined ProgramIR/storage/native XC/meta-GGA focused regression: 130 passed.
- Compiler dependency audit: 289 modules checked, 0 errors; Release CPU native build completed successfully.
- The focused ownership test proves active scalar derivative rows share memory with the propagated seven-row owner and forces failure if the generic stacked coefficient ABI is used.

For polarized PBE the retired downstream input path materialized 42 FP64 rows per point in total: seven rows for `_gradient`, seven for the coefficient bind's immutable v copy, six for the immutable density-gradient copy, eleven for `np.stack`, and eleven for the immutable stack copy. The new scalar owner is two rows wider than the previous compact PBE scalar output (eight versus six), so the optimization intentionally trades a small producer-layout expansion for removing substantially more transient input data movement.

## Consequences

The native source/artifact identity changes because the packed ABI and root-row layout are compiler inputs. The ProgramIR `xc_rows` identity also changes with its physical shape. This is deliberate: replaying an artifact with the old compact row layout is not layout-compatible with the new direct consumer.

## Revisit when

Extend the same split-buffer ABI only when another producer/consumer pair has measured complete-endpoint evidence. Do not generalize it into a new allocator or scientific IR.

## References

#831, #460, #534.
