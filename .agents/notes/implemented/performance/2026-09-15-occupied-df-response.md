# Decision: generation-linked occupied DF force response

Status: implemented
Date: 2026-09-15

The endpoint-specific occupied admission described here is superseded by the
[general work policy](2026-09-18-general-occupied-df-policy.md). The numerical
formulas, owner/generation checks and historical evidence remain applicable.

## Problem

The resident response avoids repeated auxiliary projections after #373, but
still forms two cubic AO products per auxiliary and a complete AO response
weight tensor. A canonical occupied factor permits an exact smaller contraction.

## Equations and conventions

Write the symmetric raw three-center slice as `A_P[mu,nu]`, the auxiliary
metric as `M`, and the thresholded spectral inverse as `V = f(M)`.
For each density descriptor `(D, cJ, cK)`, the current method energy terms are

```text
q_P = A_P : D
E_J = (cJ/2) q^T V q
E_K = -cK sum_PQ V_PQ Tr(D^T A_Q D A_P^T).
```

RHF has one descriptor `(D,1,1/4)` with `D=2 C C^T`. UHF has
`(D_alpha+D_beta,1,0)`, `(D_alpha,0,1/2)`, and `(D_beta,0,1/2)`,
where each spin density is `D_s=C_s C_s^T`.
For an exchange descriptor with `D=w C C^T`, define

```text
T_Q = C^T A_Q C
U_P = sum_Q V_PQ T_Q
bar_A_P = cJ (Vq)_P D - 2 cK w^2 C U_P C^T
bar_V_PQ = (cJ/2) q_P q_Q - cK w^2 (T_P : T_Q).
```

Sum descriptors before the metric reverse map. The existing spectral Frechet
map transforms `bar_V` into `bar_M`. In particular, projecting `T_Q` uses
**raw** `A_Q` in every auxiliary direction. Reconstructing it from whitened
retained values would lose the finite discarded directions required by the
off-diagonal retained/discarded divided differences.

The generated #143 derivative consumers contract `bar_A` and `bar_M` exactly
as before. One-electron, Pulay, nuclear, metric threshold, and basis conventions
are unchanged. No coordinate-by-integral derivative tensor is needed.

## Storage and lifetime

Reuse the resident value plan's charged, stream-ordered scratch. Project each
`Q` once into `T[Q,i,j]`, retain rank-squared projections, and produce bounded
auxiliary panels of pseudo-density weights for the derivative consumer.
Account for retained projection and bounded-panel sizes explicitly; the
existing full AO raw-value staging remains visible in the memory ledger.

Only a verified final-state token from the method handoff may authorize reuse.
It must match the current source owner, solve epoch, system, model, occupations,
and canonical density, with independently checked device factor generations.
The occupied-factor iteration counter starts at zero while final-state density
generation counts imported density as one; compare these using their documented
one-step offset. Corrected final determinants, external densities, missing or
stale factors, and plans without admitted storage use the dense response.

## Qualification

The automatic domain is one resident host-raw RHF system, 768/768 AOs and
occupied rank 160 on NVIDIA GeForce RTX 5090, with full admitted J/K scratch.
It requires the default response schedule and automatic resident storage.
Small-domain defaults remain dense. Explicit dense response is the comparison
reference; explicit occupied response still requires every identity check.

The current-master five-pair 768 RI-K experiment reduced the complete warm
endpoint from approximately 12.05 to 10.81 seconds with occupied K. With K
held occupied, the final five-pair response experiment reduced the median from
10.8092 to 9.8138 seconds. All samples use the same frozen post-cold density,
three SCF iterations, and unchanged independent 1e-9 Ha / 1e-8 Ha/Bohr gates.
Policy-transition priming is retained separately. Detailed small-domain and
GPU4PySCF comparisons are in the linked evidence, alongside actual work counts,
factor identity, residual, transfer and scratch diagnostics.

Independent RHF/UHF force tests include both occupied spin channels and a
zero-rank beta channel, finite discarded metric directions, geometry/property
replays and corrected-state fallback. Native adversarial tests independently
check null/stale tokens, model/rank/epoch changes, exact-density changes and
corrupted device generations.

## Rejected alternatives

The first bounded schedule used small panels from private response scratch.
It was scientifically correct but fragmented generated derivative work: the
768 endpoint increased from about 10.81 to 11.57 seconds. The retained design
reuses dead projection storage for up to 64 auxiliary AO matrices, recovering
the derivative schedule without a complete 768-slice response-weight tensor.
Keep this negative experiment: shrinking scratch alone does not reduce work.

Reconstructing raw projections from the retained whitened tensor is invalid
for finite discarded metric directions. Rebuilding C by diagonalizing D would
also bypass the existing trusted-state protocol and add unnecessary work.

## Revisit when

Extend automatic selection only with complete endpoint evidence for a new
rank, shape, spin, residency or backend domain. Revisit borrowed full resident
capacity when the value planner offers an independently qualified smaller
allocation; this change reports that existing capacity rather than claiming
to reduce the whole-plan peak.

## References

#377, #378, #379; #284 factor ownership; #373 resident response;
#358 method arithmetic ownership; #143 generated derivatives.

[Retained qualification](../../../../benchmarks/results/issue377-379-df/README.md).

The subsequent [packed derivative handoff](2026-09-15-packed-df-derivative-pairs.md)
supersedes this note's dense AO weight panels while preserving its occupied
projections, metric equations, factor provenance and scratch ownership.
