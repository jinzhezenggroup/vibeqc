# Independent review of the final C candidate

Review base: `3f4c6c4942efc0e6712e1b36b0c8938fcaae86e7`.
Fixed candidate tree: `55762b33a028dc3dfba2e6d2aaa9d918d0709fd4`.
Both reviewers were new, uninvolved in implementation and read-only. The
previous execution process had exited with a model-capacity error after the
remote verification finished; the coordinator took over without restarting
tests or changing the candidate.

## Mathematics and scientific acceptance

Reviewer `review_148c_math`: no P0/P1/P2 findings.

Checked separation of physical residuals from shifts, damping and DIIS; the
unshifted initial guess, paired DIIS vectors/errors and coordinate weights;
energy plus freshly expanded residual acceptance; actual native HF export,
independent AO-to-MO comparisons, aligned amplitudes and external residuals.

The reviewer verified all ten state input/record hashes and full histories.
Observed maxima were total energy `7.03e-13 Eh`, amplitudes `1.11e-11`, MO
integrals `4.09e-14`, and external physical residual below `1.12e-11`.
Additional independent Fock reconstruction from saved AO h/ERI and actual C
agreed within `8.23e-14`. Determinant-space FCI/projections for all four H2/He
paths agreed within `2.27e-14 Eh` in energy and `1.79e-14` in residual.

Additional water zero-initial-guess experiments with shift 0.4 / damping 0.15 /
DIIS 6, and shift 0 / damping 0.2 / no DIIS converged in 20 and 44 steps to the
reference root within `1.97e-12 Eh`. These supplement, not replace, the saved
mandatory endpoint evidence. The reviewer did not rerun PySCF or the qz suite.

## Engineering, state and evidence

Reviewer `review_148c_engineering`: no P0/P1/P2 findings.

Checked all 19 candidate hashes and 370 source-hash entries across ten
endpoint reports against the staged Git blobs; all matched. Independently
ran three targeted preflight, false-convergence and nonfinite/replay tests;
all passed. Replayed the actual saved H2 same-C and water native-HF states;
their 7/15-state energy and R1/R2 histories and final amplitudes agreed at
`1e-12`. No production external CC solver or duplicated shared interface was
found. Source tree and staged index were unchanged before/after review.

The reviewer did not have local PySCF and did not rerun remote native-HF or
the full suite. The coordinator separately read back the native library hash
and checked the two saved reference generations; see versions.log and
reference-stability.json.

## Disposition and limits

No scientific correction was required. Only the evidence and review records
were added after the frozen candidate. Acceptance covers the internal
conventional, real, closed-shell, all-electron CPU RCCSD scope of #148.
Public GPU execution, (T), Lambda, gradients and general convergence guarantees
are not inferred. The repository's stricter existing numerical tests remain.

## Subsequent collective-budget review

The later `c_engineering_review` identified a P2 in collective provider-budget
preflight, beyond the interpreter-budget rejection checked above. It was fixed
in a follow-up without rewriting this C snapshot. The same uninvolved reviewer
confirmed the correction; qz's 16 solver tests passed. See
[the correction and evidence](provider-preflight.md). The concurrent independent
`c_math_review` reconfirmed the frozen C equations and endpoint evidence without
mathematical blockers. These later checks preserve the original review record.
