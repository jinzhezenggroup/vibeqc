# Decision: compact public-AO traversal for CUDA DF values

Status: implemented
Date: 2026-09-15

## Problem and causal boundary

The CUDA source scanned mostly zero dense public-to-Cartesian rows for each
requested integral. At 768 public AOs, its complete 452,984,832-value generation
performed 39,319,083,417,600 logical coefficient reads in primitive-warp mode.
The compact traversal performs 55,292,461,056, with the same 644,972,544 nonzero
expansion triples and 2,258,403,328 primitive products. These are launch-derived
logical lane counts, not physical DRAM transactions or SASS instructions.

This source kernel was not the original unbudgeted resident setup exporter.
That route generates a complete Cartesian tensor and applies a host reference
transform. Changing source metadata alone does not change that setup.
Replacing it with the compact source made the complete cold endpoint slower:
about 104.6 seconds with the existing exporter versus 120.6 seconds with the
new exporter, with exchange held occupied and dense response. This integration
was rejected. The source probe's approximately 103.6-second baseline must not
be described as the unbudgeted resident exporter's runtime.

## Decision

Pack each public AO's normalized molecule expansion as a count, up to three
Cartesian indices and coefficients (40 bytes). Sort by Cartesian index to
preserve the previous dense traversal's summation order. Each batch item owns
its own maps; equal dimensions do not imply equal shell layouts. Cartesian
public AOs naturally have a single term. `contracted_df()` and all generated
integral/derivative equations remain unchanged.

Preserve the separate resident Cartesian exporter. Positive-budget
source-backed planning and bounded regeneration consume the faster compact
source without changing their existing storage contracts. This change adds
no complete tensor or alternate integral API.

The source owns 194,250 device metadata bytes at 768 AOs instead of 9,963,210.
The v1 resource query remains conservative by charging the greater of its old
dense-transform estimate and an aligned sparse upper bound; it does not silently
reduce caller admission estimates before knowing the actual public shape.

## Evidence and invariants

The complete source-only 768 experiment reduced generation from about 103.8
to 24.25 seconds, with all 452,984,832 outputs bit-identical. Current-master
192/384/768 full-generation repeats, raw array hashes/parity, metadata sizes,
resource reports and the rejected resident integration are retained in the
[qualification evidence](../../../../benchmarks/results/issue377-379-df/README.md).
The raw-value kernel changes from 254 to 255 registers and a 208 to 240-byte
stack frame; shared/local usage is unchanged. The measured complete source
speedup, rather than source-level sparsity alone, justifies this tradeoff.

The source validator independently forms Cartesian libcint values/derivatives
and transforms each public basis. It covers every spherical/Cartesian pairing,
s/p/d/f shells, long contractions, unequal shell layouts, reordered equal-size
batch items, geometry changes and deficient metrics across mappings and tiles.
Resident/source-backed J/K and complete force gates retain their strict limits.

An early counter report mislabeled the primitive mapping as one participating
lane instead of 32. Retain it only as provisional negative diagnostic history;
the final corrected probe counts actual launched mapping lanes. No timing was
collected with intrusive counter atomics.

## Revisit when

The final cold profile identifies one-electron/nuclear derivative generation
as the largest measured device component of the separate unbudgeted resident
endpoint (57.84 seconds). Its scope must remain distinct from the source-backed
raw-generation bottleneck and the rejected exporter substitution.

Generated primitive evaluation is now the next source optimization boundary.
Further recurrence/scheduling changes require their own independent numerical
and complete-generation evidence; this traversal change establishes none.
Extend expansion capacity only through the normalized basis owner, and keep
batch identity, summation order and resource overflow checks explicit.

## References

#379; #206; #308; #331; #351 generated recurrence ownership; #373 work amplification.
