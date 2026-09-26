# Independent read-only review and disposition

Both reviewers were newly spawned after implementation and CPU validation;
neither participated in implementation. They inspected staged Git tree
`7a265dd0def716edbb659a23e7c7ffb1d9df9b74`, checking `git write-tree` before and
after review. No reviewer modified files or repeated the full test suite.

## Mathematics, independent oracle and counterexamples

Reviewer `review_xc_math`: no actionable findings.

The reviewer checked total/separate density conventions, the total-density
chain rule `(Va+Vb)/2`, the `2 V_ij` symmetric off-diagonal variation, energy
per volume, weights applied once, both gradient AO legs, and the tau half
factor. PySCF's complete independent numerical-integration route and
`D_pyscf=SDS`, `V=SV_pyscf S` transformations were found consistent, including
actual f shells in both representations. Multiple finite-difference steps,
spin directions, factor fault injection and explicit domain failures were
reviewed. All five source hashes in `numerical.json` matched the review files;
24 independent energy/matrix rows and 48 saved FD samples passed.

## Architecture, interfaces, resources and regression risk

Reviewer `review_xc_architecture`: no actionable findings.

The reviewer confirmed reuse of `NativeAO`, grid tiles, density features, XC
program and `potential_coefficients`; no SCF/J/K scheduler or alternate global
budget was introduced. Numerical results are recomputed on each integration;
grid/basis/density/functional identities and immutable output arrays are
explicit. Molecular-grid compatibility and fixed laboratory-frame explicit
grid semantics were checked. Point tiling and matrix panels avoid a
point-by-AO-by-AO tensor, while documentation identifies the unbudgeted
interpreter/matrix/copy/BLAS overhead. The logged focused/full/native/format
and reference-regeneration checks were consistent with the reported scope.

## Main-agent disposition

The main agent independently checked the reports against the implementation,
the fixed tree, saved source hashes, full test logs and numerical samples.
Neither report proposed a change; no review-driven code changes or extra
numerical reruns were necessary. Evidence and review documentation were added
after the code tree was frozen; scientific code, tests and reference inputs
remain exactly those reviewed and verified on qz.

Both reviews explicitly limit acceptance to this fixed-density CPU slice.
`interior-v1` tail/limit support, native CPU RKS integration, convergence,
public UKS, nuclear gradients, GPU prepared execution and #202/#203's later
method-level interfaces remain outside this acceptance. Two reviews are not
a substitute for those future numerical gates.
