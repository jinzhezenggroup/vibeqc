# Decision: stream CCSD correlation input weights at the verified Lambda boundary

Status: implemented
Date: 2026-09-19

## Problem

#513 qualifies a CPU solved-state Lambda but stops short of q derivatives.
Fixed-amplitude energy partials omit amplitude response; a correct Lambda flag
alone does not prove that a later input-weight request uses that exact state.
The next consumer must also distinguish independent F/g block partials from
raw-Hamiltonian/RDM/nuclear derivatives, avoiding double counting upstream.

## Decision

Generate one parameter VJP of the existing correlation energy and unpreconditioned
singles/doubles residuals. Seed E with +1 and residuals with the verified dense
Frobenius multipliers, producing E_q + R_q*lambda. Reuse #151 symmetry projection
and both shared/expanded equation DAGs; add no handwritten scientific equation,
new CC/adjoint solver, native CUDA arithmetic or production dense Jacobian.

Bind and copy the existing Lambda result, compare all state/equation/convention
identities, and freshly recompute shared/expanded physical stationarity before
acceptance. A forged status or residual diagnostic is not proof. Per-block
execution checks both generated forms, finite FP64 values and input symmetry,
then the live owner callback again before publication. Freeze the first result
before the second graph runs, protecting independence from recycled output
buffers. Immutable output blocks carry exact CC/Lambda/response/program ids.

## Boundary and rejected alternatives

The current StationaryProblem rejects redundant dense T2, and forcing it through
packed gather expansion can introduce an amplitude-quadratic incidence matrix.
Do not relax that generic invariant or add a second derivative rule. Retain the
#513 block-boundary adapter until independent tensor coordinates can be lowered
with qualified linear-memory maps. The generic #151 AD remains the sole owner
of derivative mathematics.

F and the seven g blocks are independent mathematical inputs. The HF energy,
g-to-F normal-ordering chain, orbital response, overlap and nuclear terms are
excluded. Raw full-g perturbations affect multiple overlapping/permuted blocks;
contract every affected block direction once. No factor guessed from another
code's Lambda or RDM conventions belongs in this boundary. Diagonal F response
comes from physical residuals, not differentiation of solver denominators.

Do not call these correlation block weights relaxed physical RDMs or forces.
Do not compare them to a raw-h-fixed g derivative without an upstream pullback.
Do not weaken the production RHF validator merely to perturb a noncanonical
mathematical F during tests. The tests instead use a narrow prepared-feed seam
in the existing physical CC solver, plus a genuinely independent determinant
root for tiny native endpoints.

## Resource invariants

Generate, qualify and contract/release one parameter block at a time; no complete
NMO^4 weight tensor is assembled. The selected block is still dense, including
vvvv. This is block streaming, not intra-block tiling or a resident GPU solver.
Budget admission includes the existing bound-state logical reservation, copied
multipliers, current generated execution arrays and independent/output copies.
Python objects, opaque BLAS workspace and caller-retained previous outputs remain
excluded. Streaming is explicitly not an atomic full-gradient transaction.

## Evidence

`tests/python/test_cc_lambda_response.py` covers all ten water input blocks with
three-step re-converged CC differences, separate fixed-T partials, nonzero Lambda
response, diagonal-F shift invariance, input symmetry/orbit conventions and graph
replay. Native H2, displaced H2 and H4 endpoints compare against re-solved tiny
determinant CC problems. Failure tests cover identity/sign/backend corruption,
false Lambda residual reports, nonfinite/shape/dtype/symmetry errors, stale live
owners, per-state/per-block budgets, false independent weights and reused output
buffers. See the PR validation record for actually executed source/library ids
and test totals; there is no GPU qualification or performance promotion here.

## Revisit when

Generic method composition supports symmetry-qualified independent CC coordinates,
or native providers require intra-block tiling with composed live allocation
ownership. Retain the fixed-q and independent-root tests as migration acceptance.

## References

#152, #513, #483, #495, #151, #181; `docs/rccsd_lambda.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
