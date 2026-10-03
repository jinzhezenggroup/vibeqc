# Decision: complete the retained DF factor cotangent before source response

Status: implemented
Date: 2026-10-04

## Problem

The native DF Lambda owner in #1789 publishes five retained integral VJPs and
virtual-residual-only Bov/Bvv VJPs. Feeding only the latter into a molecular
source derivative omits the energy and retained residual terms. The native
source also discarded Boo, which is required to differentiate oovv, ovoo and
oooo. The user prioritized running complete forces on hundreds-AO systems.

## Decision

Keep the original solver/Lambda mathematical boundary. Native molecular
sources now publish Boo along with Bov/Bvv and the five retained blocks;
`Problem::df_boo` is optional for existing supplied energy/Lambda inputs.
Source publication, solver host storage and capacity queries include it.

Build a physical-sector program from the existing `BLOCK_FACTORS` inventory:
Boo/Bvv are pair projected, Bvo is Bov transposed, and Bov/Bvv identity outputs
accept the separately generated virtual-only cotangents. Shared TensorIR AD
generates the complete compressed factor pullback. No native handwritten
scientific contraction is introduced. Reference/Fock and explicit triples
factor derivatives remain separate terms to be composed by their owners.

The internal native action owns one stream and one input/scratch allocation.
Inputs are uploaded once. Every generated operation shares the arithmetic
error flag and completes before any detached result is published. All host
download destinations outlive stream-dependent buffers on exceptional exits.
Admission counts the borrowed input spans, device arena/error and detached
outputs; the caller additionally charges other owners and excess capacities.

## Invariants and rejected alternatives

- The compressed Bov cotangent includes ov and transposed vo contributions.
  A full symmetric BMO embedding must use half in each cross sector. Copying
  it into both cross blocks would double the pair-projected source derivative.
- Boo/Bvv cotangents use the symmetric dense Frobenius metric. Arbitrary
  nonsymmetric retained-block seeds must still produce the correct derivative.
- No complete N^4, ovvv or vvvv tensor, numerical Jacobian, CC iteration tape,
  CPU numerical fallback or reference-library production dependency is added.
- Generated intermediates have rank at most three. Exact contraction work is
  `2 Q (3 o^2 v^2 + o^3 v + o^4)` summands. The current generated scalar kernels
  retain this work bound; this is not a claim about BLAS throughput or complete
  force timing. A later compiler-owned packing/GEMM lowering can preserve the
  mathematical identity and same derivative gates.
- Do not recover Boo by fitting back from retained blocks: that would change
  the auxiliary-frame identity and introduce an additional ill-conditioned
  inverse problem. Retaining `Q o^2` values preserves the source frame cheaply.
- Do not interpose a focused DSO against old Problem/DFSourceResult layouts.
  Qualify a complete rebuilt library for every native consumer.

## Evidence and remaining work

The regression suite compares the generated action with an independently
embedded complete ERI cotangent and central differences for unequal occupied,
virtual and auxiliary extents. Native tests additionally compose a cold CCSD
solve, audited native Lambda and this factor response; reconverged factor
energy differences and an independent two-electron determinant Hamiltonian
provide separate numerical gates. Tests cover exact-budget admission,
one-byte-short rejection, caller storage, nonsymmetric input rejection,
nonfinite seeds and arithmetic overflow without partial publication.

The complete rebuilt CUDA library passed 47 focused cases under a finite
Slurm RTX 5090 allocation: 40 factor/Lambda/source cases and seven existing
public conventional CCSD/CCSD(T) force cases. The latter include H2/water,
live-PySCF water, batch forces, exact/near-degenerate methane and the 14-AO
water cluster with directional energy differences. The qualified library SHA256
is `bb8d7fd8e0d11e76b33c5c0f74674bd451d7937ebb99a7dc9bc54974e82cd0ca`.
The same final library passed all 30 factor/source cases under CUDA memcheck
with zero errors. All compiler/ownership hooks and focused type checks passed.
Local detailed logs are ignored under `.artifacts/factor-response/`;
no new tracked benchmark evidence payload is added.

The large ethane230 cold geometry-to-CCSD(T) energy result belongs to #1792,
not this derivative action. Neither it nor these factor derivatives establish
complete nuclear forces. Source orbital/metric pullback, DF triples response,
conventional-reference Z/Pulay response and nuclear integral contractions
remain necessary before public DF forces can be enabled.

## References

- [Native DF Lambda](2026-10-03-native-df-lambda.md), PR #1789.
- PRs #1763–#1765: compiler mathematics, independent FP64 gates, bounded owners.
- `tests/python/test_df_cc_factor_response.py` and
  `tests/python/test_df_cc_molecular_source.py`.
