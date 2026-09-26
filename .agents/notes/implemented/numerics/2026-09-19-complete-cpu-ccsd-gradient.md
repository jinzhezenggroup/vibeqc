# Decision: validate the complete CPU CCSD gradient chain before native promotion

Status: implemented
Date: 2026-09-19

## Problem

#525 supplies correct amplitude-relaxed derivatives at independent Fock/ERI
input blocks, but those are not complete h/g or nuclear derivatives. Advancing
only another disconnected weight interface would leave normal ordering,
orbital relaxation, metric response and nuclear conventions untested together.
#153 requires a complete staged Lagrangian and independent molecular evidence.

## Decision

Specify the primal raw h/g -> U-transformed h/g -> closed-shell F and reference
energy map in existing TensorIR, then generate its VJP/JVP. Shared raw-g inputs
accumulate all overlapping/permuted q fields once. Retain exact dense-Frobenius
Lambda conventions from #513/#525; generate raw h/g weights and orbital G from
the same graph. Add a complete-MO `orbital` index-space kind rather than label
all MOs as occupied or AO. Existing space identities and serialized old graphs
remain unchanged.

Use the existing native-streamed RHF operator and checked GMRES transpose solve
for Z. The orbital matrix from the independent generated Fock JVP is tiny and
explicitly used for curvature, native-action parity and final-residual checks.
No CC Jacobian is materialized and no new scientific adjoint solver is added.
Under K_ia=x, K_ai=-x the Fock residual derivative is -A, so augment the
Lagrangian with -z.F_ov after solving A.T z=-dL_corr/dx. Check the full orbital
antisymmetric derivative, including redundant oo/vv directions, instead of
introducing same-space gap divisions.

Generate the symmetric-metric contribution W_S=-(G+G.T)/4 and staged AO
cotangent transforms. Add HF reference energy and nuclear repulsion once, not
once per CC input block. The complete runtime contracts the existing native
analytic derivatives; nuclear differences occur only in tests. Return unprojected
physical and integral component decompositions so missing terms stay visible.

## Rejected alternatives

- Handwritten normal-ordering, Z/overlap or CC derivative formulae: duplicate
  scientific ownership and weaken the existing generated-math invariant.
- Reusing fixed-q bars as physical RDMs or raw-h-fixed derivatives: omits the
  Fock normal-ordering chain and reference terms.
- Differentiating SCF/CC/DIIS history or adding a new Lambda solver: unnecessary
  and inconsistent with the existing stationary-response boundary.
- Regularizing a negative/near-singular orbital response silently, or accepting
  a zero RHS without checking curvature: could hide an unstable HF reference.
- Calling this a production GPU/native method from a dense CPU validation path:
  mismatches the actual provider and memory ownership.

## Scope and ownership

The original native HF exporter and dense analytic derivative bridge support
at most 12 AOs. The complete endpoint deliberately shares that limit and owns
ordinary NumPy dense MO/AO arrays. Native J/K remains shell-streamed, using
whole-shell local tiles to avoid repeated partial-shell evaluations. The logical
budget includes retained response/gradient numeric arrays and solver/checks;
HF/provider internals, native derivative evaluator scratch, Python objects and
opaque BLAS allocation remain explicitly separate. This is no process-RSS or
end-to-end speedup claim.

The CLI accepts an explicit supported molecular schema and Bohr units; it must
not discard spin, frozen-core, triples or ECP requests. It refuses existing
output files. Public Calculator/native force flags stay unchanged. DF, frozen
core, open shells, ECP, perturbative-triples forces, resident GPU weights and
intra-block tiling remain unqualified.

## Evidence

`tests/python/test_cc_complete_gradient.py` covers generated h/g/U directional
pullbacks, AO transform duality, semantic orbital/AO domains, graph replay,
shared and independent orbital-response actions, and fresh native complete
molecular gradients. Independent PySCF 2.14.0 fixture generation is separate
from runtime and pinned with settings/source/content identities. Tests include
H2, changed H2, H2O, NH3, CH4 and Cartesian/spherical d-shell cases; corresponding
representations are compared separately because their spaces differ.

Three-step nuclear differences re-run HF+CC, while orbital finite rotations
isolate the response convention. Omitting overlap or Z must visibly fail.
Numerical failure, false success reports, nonfinite output, stale state/source,
invalid schema and insufficient budgets cannot publish successful gradients.
See the PR for final executed source/library hashes and test totals. Preliminary
small-system agreement is not evidence of production GPU coverage.

## Revisit when

Native weighted derivative consumers, tiled raw-h/g pullbacks and composed
allocation owners replace the dense <=12-AO bridge. Preserve all complete-chain,
component-omission and independent-reference tests when changing the execution
path. Native/public capability promotion requires that larger-system/backend
qualification rather than lifting the validation limit alone.

## References

#153, #152, #525, #513, #179, #151, #141, #144; `docs/ccsd_gradient.md`.

Agent: ChatGPT
Model: GPT-6 Astra Pro
