# Decision: restore physical DF pair symmetry before sector selection

Status: implemented
Date: 2026-10-03

## Problem

The native source introduced in #1777/#1781 can produce asymmetric MO factors
after its FP64 orbital contractions and metric whitening. In the 230-AO ethane
frame, the largest B_vv pair difference was 2.876198152042264e-9, although the
native raw AO pair difference was only 3.774758283725532e-15. The existing CCSD
admission correctly rejected this input at its unchanged 1e-10 symmetry gate.

This frame is ill-conditioned: max |C| is about 61, cond(S) about 5.4e5 and
cond(metric) about 3.5e8. Symmetry of exact integrals does not guarantee
bitwise symmetry of separately accumulated transformed pairs.

## Decision

The physical molecular DF-CC source explicitly selects the TensorIR projection
B[p,q,Q] := (B[p,q,Q] + B[q,p,Q])/2 before selecting any sector. All four
factor sectors and all five retained integral blocks therefore use the same
fitted Hamiltonian. The generic supplied-tensor factor program still defaults
to preserving asymmetric inputs. The solver's input and convergence gates are
unchanged. No CPU/oracle correction enters production.

The projection is part of the generated equation identity and generated arena
plan. The initial implementation retains the ordinary materialized TensorIR
schedule: two additional N*N*Q arrays, two additional packing kernels, and two
weighted summands per factor value. This adds 413,043,200 bytes of packing
workspace at (N,Q)=(230,488), and 742,680,576 bytes at (264,666). Phase admission
includes these bytes before allocation; exact-budget and one-byte-short tests
exercise the resulting plan. The three source GEMMs' aggregate work remains
2*N^3*Q + N^2*Q^2, with one raw source scan. This is a correctness repair, not a
complete-endpoint performance claim.

## Rejected alternatives

- Relaxing the solver symmetry tolerance would leave inconsistent factors in
  the physical equations.
- Averaging only B_vv after publication would leave its retained oovv block
  derived from a different factor.
- Symmetric AO inputs alone do not prevent MO contraction roundoff.
- Promoting experimental high-accuracy Rys2/3/4 or polynomial-only DF values
  is not justified by the source-only comparisons below. Those experiments
  remain ignored local artifacts, not this implementation.

## Independent-reference accuracy investigation

The failed original per-factor 3e-10 absolute/relative large-source gates must
not be silently relabeled as passing. They compare against an insufficiently
stable direct B_MO reference in these frames:

- With identical PySCF raw/metric/C inputs, projecting both orbital indices
  before whitening versus whitening before projection changes B_MO by
  9.02021e-9. Reconstructing the lower-triangle AO Hamiltonian actually consumed
  by the independent PySCF solver changes the stored B_MO by 5.95586e-10.
- A strongly amplified s/s/s term provides an independent analytic check. For
  normalized primitives alpha=beta=0.1285 at (0,0,-z), gamma=0.10153627861 at
  (0,0,z), z=1.4550891159150976 Bohr, evaluating the analytic Boys F0 formula
  at 90 decimal digits from the exact binary64 inputs gives
  5.55819383594290434659468908658126174154. Live PySCF 2.14.0 returns
  5.558193835942818; the experimental native polynomial result is
  5.5581938359429. PySCF radial normalization rounding contributes only
  8.7941e-16, leaving an approximately -8.6987e-14 integral discrepancy.
  Tightening PySCF integral screening through 1e-60 does not change it.
- The largest transformed polynomial-raw versus PySCF-raw difference is
  5.96528e-8; a long-double sum of those contributions reproduces it. Raw
  reference accuracy, transform order and physical consumer sensitivity must
  therefore be audited separately.

These findings justify investigating a conditioning-aware reference audit;
they do not establish a replacement tolerance or qualify a large endpoint.
Strict energy, amplitude and expanded residual gates remain necessary.

## Evidence and limits

The committed small independent source/energy tests check all sectors and
blocks, Cartesian/spherical auxiliary-g cases, rank truncation, native H2
endpoints and publication safety. B_vv now has an exact pair-symmetry check.
The real-GPU suite also passes full memcheck with zero errors.

An additional 230-AO diagnostic supplies the independent converged amplitudes
and orbital frame, but generates every DF factor and retained block natively.
The correlation energy differs by 2.00839e-13 Eh and freshly expanded R1/R2
maxima are 2.80326e-11/3.18210e-11. The adapter uses the allowed damping nearest
one for two evaluations, preserving the reference amplitudes to about 1e-28.
This is a source/residual diagnostic, not a cold-start molecular solve.

The analogous 264-AO benzene diagnostic also passes: energy difference
4.07230e-13 Eh, expanded R1/R2 maxima 1.91752e-11/1.16870e-11, and amplitude
changes below 3e-29. Its two primary evaluations plus expanded replay take
449.028 seconds on the RTX 5090; the corresponding ethane diagnostic takes
37.366 seconds. These timings include the solver's diagnostic evaluation and
publication, exclude RHF, and are not complete molecular endpoint timings.

Compact qualification evidence is in
`benchmarks/results/df-cc-pair-symmetry-20261003/qualification.json`. Full local
logs, diagnostic adapters, reference-order audits and rejected precision
experiments remain under `.artifacts/conditioned/` in the isolated
`qc-df-conditioned-source-20261003` worktree. No raw arrays are committed.

## Revisit when

Use compiler-owned view/elementwise fusion to eliminate materialized projection
and packing intermediates only with independent output checks, explicit work
and capacity accounting, and complete-consumer measurements (#1763). Precision
and overlap work (#1764/#1765) still requires its separate numerical and
ownership qualifications. This projection does not add native DF triples,
Lambda or forces, and does not change their current admission boundaries.
