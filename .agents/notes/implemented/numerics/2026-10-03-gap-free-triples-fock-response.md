# Decision: triples Fock cotangents from a separable resolvent

Status: implemented
Date: 2026-10-03

## Problem and boundary

The native 56-AO water-cluster force frontier contains 14 same-occupancy
near-degenerate pairs. The retained diagnostic worktree records physical
rotation stationarity below 7.5e-13, but division by an approximately 1e-15
orbital gap creates a spurious multiplier as large as 190.94. Matching diagonal
energy cotangents do not prove that the whole degenerate Fock block is scalar.
The complete independently validated resolvent now admits the 56-AO frontier.
Do not perturb fixture symmetry,
clip gaps, set an unproved off-diagonal response to zero, or loosen residual gates.

## Resolvent identity

All labels below reuse the audited W/V/R3 numerator inventory in
`python/generativeqc_compiler/cc/triples.py`. They introduce no alternative
triples numerator equations. Let P sum the six simultaneous occupied/virtual
pair permutations. Define A=PW, C=P(W+V/2). R3 is central in the occupied
permutation algebra and therefore commutes with P. On the complete ordered
virtual domain, the canonical energy is

```text
E_T = (1/3) <A, D^-1 R3(C)>
D   = Foo[1] + Foo[2] + Foo[3] - Fvv[1] - Fvv[2] - Fvv[3]
```

Here each Fock action operates on one tensor index. At a diagonal Fock matrix,
D is the usual triples denominator. Occupied/virtual separation keeps this
operator nonsingular even when Foo or Fvv has internal degeneracies.

Use `d(D^-1) = -D^-1 (dD) D^-1`. With X=A/D and Y=R3(C)/D at the canonical
point, the three equivalent pair marginals cancel the factor 1/3:

```text
Boo[p,q] = -sym(sum_abcjk X[abc,pjk] Y[abc,qjk])
Bvv[p,q] = +sym(sum_bcijk X[pbc,ijk] Y[qbc,ijk])
sym(M)   = (M + M^T)/2
```

These are complete same-space Fock cotangents, including diagonal entries.
Their diagonals match the existing generated eps VJP. In a nondegenerate
block their off-diagonals match `-S_pq / (2*(eps_p-eps_q))` with the existing
rotation convention. They remain finite at degeneracy and transform covariantly
under arbitrary rotations within a degenerate subspace.

The native implementation replaces the diagonal denominator seed AND
its associated same-space canonicalization term with this complete Fock seed.
Do not add both routes and double-count. Complete molecular forces verify this
composition through corrected Lambda, raw Hamiltonian, orbital response and
nuclear derivatives. The old nondegenerate schedule remains selected when all
same-space gaps exceed 1e-10. The physical stationarity and residual gates are
unchanged.

## Bounded schedule proof

Sweep b>=c and retain X[a,i,j,k], Y[a,i,j,k] for all a at one virtual pair.
Both moment contractions and energy use multiplicity `2-delta_bc`. The
simultaneous (b,j)<->(c,k) interchange proves this weight for the cross-a moment
as well as the energy. No full six-index owner is required.

For the 56-AO frontier (o=40,v=16), this is 136 panels and two principal
8,192,000-byte resolvent arrays, versus 4,194,304,000 bytes for their full dense
counterparts. These figures exclude inputs, numerator/generated arenas and
outputs; they are not a complete admitted-memory claim. General larger v uses two page pairs and recomputes right pages for each left
page. For m=ceil(v/q), vector and virtual moments each execute
[v(v+1)/2]*m(m+1)/2 times; occupied moments execute [v(v+1)/2]*m times.
The owner shrinks q to its complete numeric allowance and rejects below q=1.
The complete-force planner reserves the default page capacity conservatively.
Do not silently make v the memory budget.

## Private evidence and reproduction

Experiments are in the isolated worktree
`/home/jzzeng/codes/qc-cc-degenerate-resolvent-20261003`, under ignored
`.artifacts/cc-degenerate-resolvent/`:

- `dense_probe.py`: independent dense Kronecker-sum inverse versus canonical
  eigendecomposition followed by the existing triangular energy; full ordered
  energy; generated diagonal eps VJP; nondegenerate multiplier sign/factor;
  degenerate rotation covariance; and arbitrary symmetric Fock directional
  finite differences at three steps. Six cases cover unequal and equal occupied/
  virtual sizes, with/without repeated eigenvalues. Dense inverse errors are
  <=1.81e-16 and final finite-difference error <=1.23e-10.
- The same probe constructs an equal-diagonal degenerate cotangent block with
  nonzero off-diagonal elements, explicitly disproving the shortcut based only
  on equal eps cotangents.
- `panel_probe.py`: the virtual-pair schedule versus the dense resolvent
  reference for (o,v)=(2,3),(3,2),(3,3),(4,5), with exact degeneracies. Energy and
  full-matrix discrepancies are <=4.45e-16.
- `tensor_probe.py`: a 149-node TensorIR graph reuses the current runtime
  W/V/R3 builders, then expresses the two moments. It matches the independent
  dense reference to <=4.45e-16 in the same four shapes. These preliminary results motivated the generated native CPU/CUDA owner;
  current tests additionally cover capacities 1/2/4, tails and page cross terms.

- `molecular_probe.py`: CPU H2O validation tooling builds the complete corrected
  Lambda and raw-Fock orbital-response state. The new panel moment matches the
  sum of existing diagonal denominator and canonicalization Fock weights to
  4.07e-13. This checks complete-chain sign, factor and non-double-counting for
  one nondegenerate molecule; it does not qualify degenerate nuclear forces.

Run each with CPU library thread counts set to one and `PYTHONPATH=python:.`.
The molecular probe additionally uses `tests/python` in `PYTHONPATH` and the
frozen #1748 library through `GENERATIVEQC_LIBRARY` (CPU execution only).
The scripts write exact JSON results adjacent to themselves. They are temporary
research evidence, not retained production acceptance or benchmark publications.

## Implementation and acceptance

- The compiler owns the new mathematical programs and reuses the audited W/V/R3
  helpers. CPU/CUDA mathematical hashes and arena layouts agree. All old
  generated CPU/CC/triples-response source bytes remain unchanged.
- Native CPU/CUDA rational fixtures have maximum error 1.04e-17, including
  capacities 1/2/3, exact budgets, automatic page shrinking, undersized refusal,
  invalid denominators and nonfinite inputs. Unfiltered CUDA memcheck reports
  zero errors for the native fixture, complete methane force/finite-difference
  tests, and the complete 56-AO force (Slurm n2 job2137).
- Four complete 56-AO GPU calls (cold, warm, repeat, changed geometry) satisfy
  independent pinned PySCF energy (3e-9), triples (2e-9), force (1e-6) and
  response-residual (1e-9) gates. Maximum observed force error is 1.12e-7.
- Three complete-energy directional differences at steps 4e-4, 2e-4 and 1e-4
  have maximum error 1.34e-9. A physical 1e-9-bohr displacement also passes.
- Tetrahedral methane provides a small independent PySCF 2.14 corrected-Lambda
  oracle, exact/near-degenerate force regression and two-step directional
  differences. Real CPU allocation interception verifies this selected path's
  exact complete budget, narrow provider fallback and one-byte-short refusal.
- The 56-AO full Fock owner reports 136 pair panels, 136 vector pages and 136
  moments of each type, page capacity 16, 298,927,104 complete owned numeric
  bytes and 298,911,744 device bytes. These include generated scratch rather
  than claiming only the two principal resolvent arrays as the total.

Complete qualification records, exact binary/source identities, work counts and
sanitizer results are retained under
`benchmarks/results/cc-triples-fock-response-20261003/`. Frozen source archives,
raw traces and build logs remain in the ignored worktree artifacts. These are
complete endpoint measurements; the old code rejected the symmetric 56-AO
force, so its newly available execution is not a before/after speedup claim.

## Consequences and revisit conditions

The resolvent adds a separate bounded phase only when internal gaps make the
old schedule unsafe. It does not remove the occupied/virtual gap requirement,
RHF stability checks, or other scientific acceptance gates. The 56-AO endpoint
remains dominated by work elsewhere in the response chain; bounded storage is
not proof of bounded algorithmic work. Revisit page reuse and generated moments
only with full endpoint timing, equivalent physical work and independent gates.

This decision supersedes the internal-degeneracy rejection in the earlier
[canonical-gap decision](2026-09-20-ccsdt-orbital-response.md), preserving its
occupied/virtual separation and physical stationarity requirements.
