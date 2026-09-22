# Decision: share SCF and CPKS vacuum reference admission

Status: implemented
Date: 2026-09-20

## Problem

The production-grid integration admits a subnormal reference gradient when AO
contractions round its spin density to zero. The SCF evaluator already accepts
that reference, but the RKS/UKS directional point evaluators still required an
exactly zero reference gradient. Consequently the same valid SCF reference
could reject an exactly zero response direction. This was reproduced in the
native point contracts for both LDA and PBE after #690 merged.

## Decision

Use one reference-gradient admission predicate in `xc_point.hpp` for SCF and
both shared CPU/device response functions. It validates finite components and
preserves the existing SCF `abs(gradient) < DBL_MIN` allowance only at exactly
zero spin density. RKS takes total gradients, so it tests their rounded
half-spin components. Comparing the total against `2*DBL_MIN` is insufficient:
the immediately lower total rounds up to a normal spin component. The density
predicate only selects exact vacuum; positive-density representability gates
remain as before. No input or point formula is modified.

Empty-spin tangent admission is separate and unchanged: density and gradient
directions must be exactly zero. Positive-density algebra, independently scaled
roundoff gates and explicit nonrepresentable-output rejection remain unchanged.
No full empty-spin Hessian or extra functional domain is promoted.

## Rejected alternatives

- Bypassing reference validation for a zero direction would also accept invalid
  normal vacuum gradients or nonfinite inputs.
- Broadening tangent admission would change the response's derivative domain;
  this defect concerns an already accepted reference.
- Adding a positive-density floor would modify qualified low-density physics.
- Duplicating the SCF point evaluation for validation would add work where a
  common admission predicate suffices.

## Evidence

New private-ABI RKS and UKS regressions reject the pre-fix library at the
SCF-admitted reference with a zero direction. They cover both signs, every
Cartesian component/spin, minimum and maximum subnormal gradients, RKS total
packing and half-spin rounding ties, normal boundary rejection and unchanged
nonzero tangent rejection. The device test
checks the same zero-response invariant directly and preserves all 30 RKS plus
48 UKS independent 450-digit reference directions and their numerical gates.

A separate baseline probe on the actual default 55,296-point native grids found
ordinary H2 RKS and LiH+ UKS LDA/PBE zero/nonzero actions already passed; those
specific molecular states contained no affected PBE vacuum residues. The
point-contract failure is not being relabeled as an observed molecular failure.

Final qualification at source `1329d0bcbc2e79e97c7988b0bb5174b9c8a3b0ce`
used matching Release builds on n5. Finite Slurm jobs 1143, 1144 and 1145
completed with exit status zero; GPU execution used the `main` partition and
one `gpu:5090`, preserving scheduler-assigned device visibility.

- CPU: all 45 native CTests passed; all 63 native RKS/UKS, CPKS, Krylov and
  response-problem Python tests passed without skips in 268.48 seconds.
- The default-grid probe passed all eight zero/nonzero actions across H2 RKS
  and LiH+ UKS with LDA/PBE on 55,296-point grids.
- CUDA: all 78 independent point directions and 192 vacuum reference/tangent
  checks passed. All 55 native CPKS, spin response, resident block response and
  resident HVP Python tests passed without skips in 137.29 seconds.
- CPU library SHA-256:
  `d287b7f0cd852e711113acf766100f051109aa864c66ef28f985419bce3dda5f`.
- CUDA library SHA-256:
  `27fbf23feb48a9d7727b9707a1937bbcbace30980babf2680e5de921b9878e6d`.

The intermediate implementation checked total RKS gradients and failed the
added packed-reference regression against its matching library. The final
qualification above includes the rounded half-spin correction; it is not
inherited from the intermediate run. Neither campaign changes the existing
independent numerical acceptance gates or asserts a performance improvement.

After integrating master `b259dfd1` (including the upstream #721 transport lint
fix), matching builds at `3d0b4c4b98529a9865bc65a8223a8c19e3796a94` were
qualified again. The Krylov conflict was resolved by retaining the upstream
file verbatim; the final change relative to master remains the seven vacuum
reference implementation, test and documentation files. The integration also
includes master's KS provenance and generated force changes, so earlier
branch results were not substituted for this combined validation.

- All 48 Krylov/resident-lifecycle tests passed in 0.49 seconds after adopting
  upstream's transport implementation; all-file pre-commit hooks passed.
- Slurm jobs 1146/1147/1148 completed with exit status zero using the same
  finite CPU/build/GPU protocol and unchanged independent acceptance gates.
- CPU: 45/45 native CTests, 63/63 selected Python tests in 267.61 seconds,
  and all eight default-grid native KS actions passed without skips.
- CUDA: 78/78 independent point directions, 192/192 reference/tangent checks,
  and 55/55 selected integration tests in 311.91 seconds passed without skips.
- Integrated CPU library SHA-256:
  `00ea4d2546717a9c199e137520d0ee968b615094d64a8f8764034bc153244606`.
- Integrated CUDA library SHA-256:
  `76016a993cac8eb5ccf1d897b7c84b22c190e9d92c1f572fdd18696a70c587a4`.

## Consequences and revisit conditions

SCF and response cannot drift independently on reference-gradient admission.
Any future change to empty-spin tangent semantics requires separate derivative
qualification; it must not be inferred from a reference-domain change.

## References

- Issues #179 and #690.
- `src/dft/xc_point.hpp`, `src/dft/xc_point_response.hpp`.
- `tests/native/test_rks_response.cpp`, `tests/native/test_uks_response.cpp`.
- `tests/native/test_xc_response_cuda.cu`, `docs/xc_scf_domain.md`.
