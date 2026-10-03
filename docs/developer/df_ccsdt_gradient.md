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
  +-- CCSD Lambda / residual response --------- native fixed-orbital A2
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

`tools.generativeqc_cc.pullback_df_three_index` accepts one or more cotangents for
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

Native CUDA Lambda now accepts the explicit factorized `cc::Problem`
representation. `cc/df_lambda.py` derives retained-core energy RHS, amplitude
transpose actions and all eight retained Fock/integral parameter VJPs from the
shared RCCSD residual. Separate expanded-core actions provide the final audit.
The owner composes every Q slice of the existing `cc/df_equations.py` amplitude
VJP with each core transpose, including the independent replay. It never
reconstructs `ovvv/vvvv` or a dense amplitude Jacobian.

`solve_lambda_parameter_response_cuda` also returns Q-major `df_bov/df_bvv`
cotangents for the **virtual residual contribution**. The eight retained-block
cotangents are returned separately; their Gram-product pullbacks must be added
before these factor weights describe the complete correlation Hamiltonian.
`ovvv/vvvv` response vectors stay empty. Optional explicit T1/T2 energy sources
use the same corrected-Lambda interface as conventional calculations; this API
does not generate the DF triples sources itself.

The same host GMRES, packed pair-symmetry coordinates, diagonal preconditioner,
fresh CC replay and independent Lambda acceptance gates are retained. Native
scientific actions execute on CUDA. One staged input state and one reusable
scratch arena cover all actions; stream order consumes each borrowed output
before reuse. The sticky arithmetic flag spans the complete core-plus-Q action.
The DF numeric bound includes both host Krylov/publication storage and the
device arena, together with borrowed reference/CC state. Diagnostics report
complete auxiliary visits, generated kernels, contraction summands and transfers.
Composed operator hashes include both retained-core and virtual derivative
identities. This is still a fixed-orbital response phase, not a nuclear force.

`tests/python/test_df_cc_lambda.py` verifies the retained-plus-Q action against
the complete integral equations, native Lambda/parameters against conventional
CUDA on the same Gram Hamiltonian, physical factor energy finite differences
after fresh CC convergence, an independent two-electron determinant energy
derivative, and exact-budget/one-byte-short publication behavior.
Set `GENERATIVEQC_DF_LAMBDA_CUDA_TEST=1` and `GENERATIVEQC_LIBRARY` in a finite
Slurm allocation for native tests. See the
[decision note](../../.agents/notes/implemented/numerics/2026-10-03-native-df-lambda.md).

Slice A2 still needs the retained Gram/source pullback and molecular composition.
Slice A3 must add the standard-(T) numerator,
denominator, direct-triples, and corrected-Lambda response on the same DF
Hamiltonian.

After those cotangents exist, slice B composes the conventional-reference
orbital/Pulay response and fixed-rank diagnostics. Slice C connects `bar_A` and
`bar_M` to the existing generated fused derivative consumer, qualifies
resident/streamed execution, and performs complete nuclear finite differences.
