# DF-CCSD(T) analytic-gradient composition

Status: issue #158 slice A1. This slice implements and validates the reusable
reverse edge from whitened MO three-index factors to raw DF sources. It does not
claim a complete DF-CCSD(T) force.

## Dependency graph

The first supported method remains the #157 correlation-only DF Hamiltonian:
conventional all-electron RHF supplies the reference Fock/orbitals, while the
correlation two-electron interaction is density fitted.

```text
E_DF-CCSD(T)
  |
  +-- converged T1/T2 -------------------------- #157
  +-- CCSD Lambda / residual response --------- #158 future A2
  +-- standard-(T) + corrected Lambda --------- #158 future A3
  +-- orbital / overlap response -------------- reuse #155 machinery
  |
  v
cotangents for the exact DF B blocks used by the energy/residual equations
  |
  v
B[Q,p,q] = sum_P A_MO[P,p,q] W[P,Q]
A_MO[P,p,q] = sum_mn C[m,p] C[n,q] A[m,n,P]
W = M^(-1/2)
  |
  +-- bar_A ------------------------------------ #143 derivative consumer
  +-- bar_M -- fixed-rank spectral VJP -------- #466/#656/#657
```

Every source is counted once. The conventional-RHF reference/Fock branch remains
separate from the correlation-DF branch and must later be composed before a
public force is published.
## Slice A1 boundary

`tools.vibeqc_cc.pullback_df_three_index` accepts one or more cotangents for
specific B blocks and returns physical raw three-center and metric weights.

The implementation deliberately reuses
`SymmetricMatrixFunctionSpec(..., function="inverse_sqrt")` for the metric
pullback. It therefore inherits the shared fixed-effective-rank rule, including
retained/discarded subspace mixing, branch-gap diagnostics, and rejection at an
unresolved cutoff. There is no CC-specific metric inverse formula.

For one block, with `bar_B` supplied by the upstream CC/(T)/Lambda graph,

```text
bar_A_MO[P,p,q] = sum_Q W[P,Q] bar_B[Q,p,q]
bar_W[P,Q]      = sum_pq A_MO[P,p,q] bar_B[Q,p,q]
bar_A[m,n,P]    = sum_pq C[m,p] C[n,q] bar_A_MO[P,p,q]
bar_M           = D(M^(-1/2))^*[bar_W]
```

Multiple B blocks accumulate into the same `bar_A` and `bar_M`. Because raw
three-center and metric sources are symmetric physical objects, their
cotangents are projected to the corresponding symmetric source spaces exactly
once before publication.
## Validation

The focused tests cover three distinct properties:

1. A small same-Hamiltonian dense four-index functional
   `g_DF[p,q,r,s] = sum_Q B[Q,p,q] B[Q,r,s]` is differentiated through B and
   then through A/M. The returned `bar_A` and `bar_M` match a complete central
   finite difference that perturbs both raw A and M on a fixed rank branch.
2. Independent `B_ov` and `B_vv` cotangents accumulated in one call reproduce
   the sum of separate pullbacks.
3. An eigenvalue exactly on the metric threshold is rejected, and the logical
   memory guard fails before the pullback executes when the budget is too small.

The dense four-index object exists only inside the tiny validation objective; the
production pullback itself never reconstructs `g_DF`.

## Remaining #158 work

Slice A2 must generate cotangents of the factorized #157 RCCSD residual/energy
equations with TensorIR AD, including all retained smaller four-index blocks and
the factorized `ovvv/vvvv` path. Slice A3 must add the standard-(T) numerator,
denominator, direct-triples, and corrected-Lambda response on the same DF
Hamiltonian.

After those cotangents exist, slice B composes the conventional-reference
orbital/Pulay response and fixed-rank diagnostics. Slice C connects `bar_A` and
`bar_M` to the existing generated fused derivative consumer, qualifies
resident/streamed execution, and performs complete nuclear finite differences.
