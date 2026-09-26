# Decision: preserve packed AO pairs through generated DF derivatives

Status: implemented
Date: 2026-09-15

## Problem

The occupied response from #381 retains small occupied projections but expands
its final weights into bounded dense AO matrices. The generated shell consumer
then executes both ordered orbital shell pairs. The fresh 768-AO baseline spends
5.254 seconds in three-center derivatives, out of 6.187 seconds in the complete
force stage and 9.855 seconds in the instrumented warm endpoint.

The issue's 56,623,104 shell triples describe the dense single-panel response.
The occupied response actually visits 57,212,928 triples: four auxiliary shells
are split by the 64-AO panel boundaries. These are distinct execution domains,
even though they contract the same 452,984,832 public response weights.

## Decision and invariants

Keep the generated derivative equations from #143. Runtime traversal visits
each unordered orbital shell pair once: all different angular-class products
in one orientation and a compact triangle within equal angular classes. No
quadratic shell-pair list is allocated. The dense diagnostic still executes the
full ordered product.

For different shells, their atomic gradients satisfy `g(A,B)=g(B,A)` after
scattering derivatives to physical atoms. Thus dense folded execution reads
`W_AB + W_BA` before applying normalized public-AO expansions. It is exact even
for nonsymmetric external weights. A diagonal shell retains all its ordered AO
pairs in this route; doubling an entire diagonal shell would be incorrect.

Packed storage contains `W_ii` and `W_ij+W_ji` at `i*(i+1)/2+j`, for `i>j`,
with the auxiliary index outermost. Within a diagonal shell, only lower public
AO pairs are consumed. Both orbital derivative slots still scatter to their
shared atom. Spherical/Cartesian expansion, normalization and auxiliary-only
centers keep their existing semantics. The generated subgroup reduces directly
to the atomic gradient; no coordinate-by-integral derivative tensor is created.

The trusted occupied producer computes the same `T_Q=C^T A_Q C` and
`U_P=sum_Q V_PQ T_Q` as #381. For packed expansion it uses
`sym(U_P)=(U_P+U_P^T)/2`, because the antisymmetric part cancels against symmetric
integral derivatives. It then computes `C sym(U_P)` and only lower rectangular
AO blocks of `C sym(U_P) C^T`, folding off-diagonal multiplicity during scatter.
The Coulomb contribution reads and sums both density entries directly.

Each rectangular block has at most 256 AO rows by default. Upper off-diagonal blocks are
never evaluated. Small unused upper triangles inside diagonal blocks are
counted explicitly: for `n` divisible by the block size `b`, the second products
generate `n*(n+b)/2` entries per auxiliary, versus `n*n` in dense expansion. This is
not a dense-panel allocation followed by a compression pass. Tiny systems can
fit a whole AO matrix in one rectangular block; counters report this honestly.

Packed auxiliary panels end at shell boundaries when possible, never exceeding
the existing 64-AO cap or a smaller caller cap. A shell wider than an explicit
cap still splits and contracts exactly. The 768-AO packed route therefore
reaches the exact `(384*385/2)*384 = 28,385,280` shell-triple domain.

Metric thresholds, finite discarded metric directions, the spectral Frechet
map, occupied-factor provenance and every force/energy gate remain unchanged.
Missing, stale or corrected occupied states keep the dense response. Explicit
full and symmetric controls remain independently usable; generic/host consumers
retain their established dense contracts.

## Storage and ownership

The occupied projection buffer is reused only after every projection has been
consumed. The same stream orders generation, packed consumption and subsequent
J/K reuse; the bridge drains on failure as well as success.

At 768 AOs, a 64-auxiliary packed panel contains 18,898,944 doubles
(151,191,552 bytes), versus 37,748,736 doubles (301,989,888 bytes) when dense.
The complete derivative handoff consumes 226,787,328 doubles rather than
452,984,832. The rectangular GEMM scratch is at most `min(256,n)*n` doubles and reuses
an existing AO workspace matrix. These reductions concern produced/consumed
weights, not total device allocation: the three already charged resident J/K
buffers and the 3,623,878,656-byte raw upload are unchanged.

The compiler continues to own derivative algebra. Pair enumeration, packing,
BLAS panel scheduling and reductions remain native execution adapters; no new
response IR or second derivative formula family is introduced.

## Evidence and revisit conditions

The independent native shell oracle covers sparse nonsymmetric weights, s/p/d/f,
all public representation combinations, partial auxiliary shells and all three
subgroup schedules. Complete RHF/UHF gates include zero occupied beta rank,
changed geometries, batch reuse, corrected-state fallback, finite discarded
metric modes and multiple AO blocks/panels. Keep those gates when changing
packed ordering or block sizes.

The five-pair Slice A warm medians are 0.270 to 0.207 seconds (192 AOs),
3.262 to 2.832 seconds (384), and 9.810 to 7.294 seconds (768). All samples use
three SCF iterations and the same frozen post-cold density. The largest force
error is below 1.3e-10 Ha/Bohr, against an unchanged 1e-8 gate. See the retained
evidence for packed qualification, separate profiling and GPU4PySCF comparison.

The first packed producer used 64-row AO blocks. It correctly halved weights
and reached the exact triangular shell domain, but increased warm latency from
7.319 to 7.558 seconds. Separate traces identified 374 ms of weight production
versus 135 ms for dense expansion; derivative time stayed about 2.72 seconds.
The block sweep measured 372/202/144/190 ms for 64/128/256/384 rows. Five clean
comparisons selected 256 rows (7.305 seconds) over 384 (7.356 seconds).
The selected route computes more diagonal-block entries than the 64-row route,
but submits 3,072 pseudo-density products instead of 9,984. It preserves the
151.2-MB packed panel and never builds a dense panel before packing. Retain this
negative experiment: fewer FLOPs alone did not imply a faster complete endpoint.

Automatic packing is limited to the qualified 768/768 rank-160 RHF occupied
state on RTX 5090. Smaller promoted shell defaults use dense folded weights;
explicit controls retain all diagnostic routes and block sizes.

Revisit additional occupied/derivative fusion only if measured contraction or
storage remains dominant. Shrinking panels without counting repeated shell
work previously regressed this endpoint. A remaining gap to GPU4PySCF does not
authorize changing the DF model or numerical gates.

## References

#382, #206, #143 and #381. This supersedes only the dense derivative-weight
handoff in the [occupied response note](2026-09-15-occupied-df-response.md).
The [retained evidence](../../../../benchmarks/results/issue382-packed-df/README.md)
contains numerical results, controls, source identities and reproduction.
