# Diagnose strict force accuracy with practical auxiliary bases

Status: proposed (numerical correction not yet identified)
Date: 2026-09-17

## Evidence and boundary

The first explicit cc-pVDZ/cc-pVDZ-JKFIT campaign uses water tetramer RHF
96/464 and OH UHF 19/93. Every input shell and contraction is retained in both
engines, and stock/native effective ranks match. Both seven-pair force cells
fail the predeclared 3e-11 gate, despite energy errors below 2e-12.

A tighter independent CPU DF oracle attributes the dominant discrepancy to
native forces: about 1.75e-9 for water and 1.24e-10 for OH. Stock errors against
the same oracle are about 1.74e-11 and 6.67e-12. This differs from the earlier
equal-basis/direct convergence-sensitive failures; do not reuse that diagnosis
without checking the new model.

Untimed Slurm 9905 comparisons freeze one native post-cold density. Host final
validation, shell contraction, BLAS response, shell+BLAS and occupied response
all retain the force discrepancy. Returning to auto reproduces it. None is a
qualified numerical fix or performance promotion. The response arithmetic
changes error magnitude, but conditioning, derivative error and state error
have not yet been independently separated.

## Next diagnostic

Compare native and independent final densities/physical residuals, then separate
one-electron, three-center and metric force terms using identical explicit
weights. Retain full auxiliary/metric response and every full-rank direction.
Use existing validation interfaces and independent CPU derivatives before
changing precision, factorization or production response algorithms.

Do not truncate g shells from the common O/N def2 JKFIT records to manufacture a
supported practical basis. The standard cc-pVDZ/cc-pVDZ-JKFIT pairing above is
supported through f. Do not rerun failed clean cells merely for admission, relax
their declared gates, or present untimed diagnostics as clean performance.

## References

- #206 remains open.
- `benchmarks/results/issue206-practical-auxiliary/README.md` retains first
  failures, exact basis/library provenance, all raw pairs and diagnostic arrays.
- #428's default promotion remains limited to equal-auxiliary 384/768 AO.
