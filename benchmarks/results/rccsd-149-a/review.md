# Independent review: fixed-amplitude A scope

Both reviewers were uninvolved in implementation and worked read-only. They
checked the nine A-owned files against `source-snapshot.json` and the retained
`frozen-source/` originals. None of the resident B files was included or accepted.

## Engineering and evidence (`review_149a`)

No P0/P1/P2 findings. Confirmed direct reuse of the #148 DAG, trace output
lifetimes charged to the existing planner, four nonzero-amplitude shapes at
two budgets and rejection at one byte below the tested minimum. All eight
records and 5664 checks passed. Ordinary and minimum-budget memcheck logs
report zero errors and corresponding return codes are zero. Repeat calls,
failure recovery and separate contexts are tested. State primitives are not
misrepresented as an implemented resident solver or public method.

## Mathematics and independent references (`review_148c_math`)

No P0/P1/P2 findings. The accepted #148 reference Git blob, cases/input hashes,
PySCF 2.14.0 and upstream manifests match. The reference generator verifies
the two pinned RCCSD source files and reconstructs physical residuals using
the actual shifted denominators and deliberately noncanonical diagonal inputs.

Each shape retains 152 trace nodes plus 11 outputs with independent reference
values (energy, R1/R2 and eight intermediates). The reviewer reconstructed the
expected check-key sets; all 5664 entries were present with unchanged gates.
Maximum recorded error is 3.5527e-15. A fresh local CPU comparison passed, and
an additional determinant-space random-input oracle agreed for E/R1/R2 within
2.78e-17 / 2.78e-16 / 1.67e-16 respectively.

DIIS Gram weighting matches the accepted packed metric; coefficient
normalization, singular/nonfinite cases and max-norm semantics are consistent.
Jacobi denominators do not modify physical residual equations and warm state
uses the exact reference identity.

## Disposition and limits

No implementation correction was required for A. The coordinator verified
283 source hashes and all eight numerical records independently, then added
only the archive README and this disposition. The tested source was not
changed by review. Runtime helpers/routine logs not staged for publication
remain local; the scientific JSON, source snapshots, device records and
sanitizer logs are retained.

Neither reviewer reran a GPU experiment or allocated resources. Recorded
execution plus frozen source supports fixed-amplitude A acceptance. Per-node
CPU parity is not an independent mathematical oracle for every node: only the
11 named outputs also have independent external formula references. Profile
samples/shared-device monitoring do not establish performance leadership or
exclusive peak-memory evidence. Resident iteration, full recovery, molecular
convergence, registration and batch semantics remain B/C work.
