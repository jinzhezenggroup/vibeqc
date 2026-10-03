# Decision: differentiate the bounded occupied-tile DF triples schedule

Status: implemented
Date: 2026-10-04

## Problem and scope

The physical-source response in #1797 completes the fixed-Fock CCSD factor
chain. CCSD(T) also needs triples amplitude sources and explicit factor,
retained-integral and Fock derivatives. The existing conventional triples
response requires resident ovvv, defeating the large DF memory contract.
Following #1763–#1765, use shared compiler mathematics, independent FP64 gates
and complete work/ownership accounting.

This change provides the internal native CUDA fixed-canonical-input triples
pullback. It returns the original energy and all nine input cotangents. It
does not enable public DF molecular forces: full same-space Fock response,
conventional-RHF orbital/Pulay response and native nuclear contractions remain.

## Decision

Apply shared AD to the existing occupied-domain panel, W/V moment and exact
scalar energy programs. Panel and moment VJPs lower to two/four BLAS products
on local views. Signed seeds, physical strides and contraction-specific matrix
cuts are carried into the existing GEMM lowering. Mathematical expressions
remain compiler-owned; native code owns views, buffers and traversal.

Retain all six occupied permutations, including repeats, and the original
6/2/1 multiplicity. Demand-driven scalar AD emits 43 local derivative programs.
Inverse virtual-permutation gathers deterministically assemble six W cotangent
cubes without floating atomics. One reusable cube handles each V cotangent and
then the gap derivative. A panel cotangent sums all permutations sharing an
occupied index before two factor pullbacks. No ovvv or rank-six array is formed.

The generated gap reverse reduces one occupied scalar and three virtual
marginals. Native scatter adds all three independent occupied slots, including
repeats. It composes gap times multiplicity exactly once and never clears an
earlier sticky arithmetic error. Every BLAS output is audited in its logical
strided view, and scalar AD audits intermediate arithmetic.

The complete admission includes borrowed host inputs, detached host outputs,
CUDA arena/provider storage and caller-specified other live numeric state.
Three panel buffers have an explicit one-panel fallback. Overflow/work checks
precede input access. Outputs outlive exception-path stream drains; publication
follows the final finite audit and synchronization. There is no CPU mathematical
fallback. Bvv's returned cotangent is dense Frobenius and unprojected; the
physical factor embedding owns its symmetry projection.

## Layout failures retained as rationale

V's global ovov[i,:,j,:] slice is noncontiguous after flattening a,b. A bounded
v-by-v buffer gathers it for the t1 derivative, is overwritten by bar_ovov in
the last generated V product, and is scattered into the global strided view.
Using that original slice as a dense vector is incorrect.

Likewise one fixed leading dimension per logical cube is incorrect: W1 views
the same cube as [ab,c], whereas W2 views it as [a,bc]. Compiled BLAS derivative
tests exposed this failure. Contiguous leading dimensions now follow the
particular contraction's matrix cut, physical views keep explicit strides, and
vectors use a unit leading dimension. The pre-existing forward header prefix
was verified byte-identical when adding the local reverse BLAS lowering.

## Work accounting

Let T=o(o+1)(o+2)/6 be occupied tiles and G=o²(o+1)/2 the total number of
distinct occupied indices summed across these tiles. Let P be actual panel
GEMMs, including regeneration under the selected cache capacity. Then:

- Forward W GEMMs: 12T; reverse GEMMs: 48T+2G.
- BLAS contraction summands: (P+2G)Qv³ + 18T(v⁴+ov³) + 24Tv³.
- Scalar response evaluations: 43Tv³ (an evaluation has multiple scalar ops).
- Gap reverse additionally performs four cubic reductions per tile; those
  reductions and scalar work are not included in BLAS contraction summands.
- Energy scalar/reduction and finite-audit kernels also execute and are counted
  separately. Memory boundedness alone is not a performance qualification.

## Independent evidence

The focused host and native tests compare every input cotangent with original
virtual-triangle triples energy differences, including physical symmetric
factor/T2 directions, repeated occupied indices, exact/near same-space
degeneracy and both panel capacities. Compiled C++ tests compare local BLAS
products with shared AD on aliased and strided physical views. Rejection gates
exercise nonfinite/overflowing arithmetic, budget boundaries and preflight
before null input access.

Two corrected-Lambda chain cases, (o,v)=(2,3),(3,2), pass native triples t1/t2
sources into the existing Lambda owner, add explicit triples retained/factor
terms and compare the resulting native factor response against reconverged
CCSD plus independent original (T) energies at two finite-difference steps.
The amplitude-relaxation contribution is explicitly nonzero.

The initial 24-case response module passed CUDA memcheck with zero errors.
Two later null-input preflight cases pass without reaching any CUDA API; running
only those under memcheck yields its expected no-instrumented-API termination,
not a sanitizer qualification. All 47 final combined occupied-energy/response
tests pass in 57.65 s. Focused type, formatting, compiler-boundary and native-owner
hooks pass. Ignored logs live under `.artifacts/triples-response/`.

### 230-AO ethane component

Frozen complete library SHA-256:
`5a33b9ef2f2895fdb12c72f688283cef3e460dfe7b490f483393fff8950071ea`.
Local Slurm job 12202 used one RTX 5090. Inputs are the frozen native C/T from
#1792's cold ethane solve; the validation executable regenerates native DF
source and uses its explicit host-reference Fock transformation fixture.
This is not another cold CC solve or a nuclear force endpoint.

For (N,o,v,Q)=(230,9,221,488), complete cold fixed-input response, including
uploads, outputs and teardown, took 113.949970156 s. Source regeneration took
1.155719927 s; the validation process, also including energy-difference checks,
took 126.25 s externally. The (T) energy is -0.014391214097531203 Eh, matching
the prior complete native solve (-0.014391214097531201 Eh).

Numeric capacity is 2,777,638,232 bytes, including workspace 2,186,033,920 bytes.
There are 165 occupied tiles, 152 panel GEMMs, 1,980 forward moment GEMMs,
8,730 reverse GEMMs, 12,483,272,948,276 BLAS contraction summands and
76,582,443,795 scalar response evaluations. Transfers are 263,060,792 H2D and
263,060,804 D2H bytes. All nine outputs are finite. The amplitude directional
derivative is 0.00028506892478883462; native energy finite difference gives
0.00028506892479386592, an absolute error of 5.031294391527608e-15.

## Remaining force boundary

Canonical epsilon sources do not replace full same-space Fock derivatives.
The conventional force owner uses a same-space orbital-gap construction only
above an explicit threshold and uses the shared resolvent at degeneracy.
Benzene requires the latter. `triples_fock_response.py` defines X=PW/D and
Y=R3(P(W+V/2))/D and derives complete oo/vv moments without dividing same-space
gaps; its existing virtual-page source still requires conventional ovvv.

A bounded DF occupied-domain lowering is the next step. Virtual moments can
contract cubic X/Y by BLAS; occupied moments couple (i,j,k) with (l,j,k) and need
bounded cross-occupied storage/recomputation and complete work accounting.
Retain degeneracy/covariance and independent recanonicalized-energy gates.
Do not enable public forces using epsilon-only response or infer complete-force
practicality from this component timing. Large-source strict factor gates from
#1782 also remain unqualified; their tolerances have not changed.
