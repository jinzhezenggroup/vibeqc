# Deferred SSS Rys prototype

#394 remains open. Only `000` is implemented, with automatic promotion disabled.
There is no completed full-library force qualification or 384/768-AO endpoint
measurement. Work stopped when the user requested consolidation; the running
full-library build was interrupted. This is an unqualified prototype, not a
measured performance failure or a production optimization.

`prototype.patch` preserves the source and tests against
`1c3f2ab8d6df6f06bee526b89a807511ef726301`. It contains identity response folding,
one-shell ownership, independently generated one/two-root quadrature, SSS moment
IR lowering, explicit math policy and immutable prepared-source provenance.
Applying this archival patch is a separate experiment. No files under `src/`,
`python/`, `tests/` or the build configuration change in this publication.

`status.json` retains the exact patch identity and validation limits:

- Independent CPU numerical/generation tests passed. The combined run had a
  work-ledger field-name failure after removing C grouping; that reader was
  corrected and all 22 ledger tests then passed.
- Slurm job 9756 passed both generated CUDA root-grid tests in 4.11 s. They
  compare nodes, weights and defining moments with a 75-digit independent
  reference over dense/random/boundary/extreme arguments, at relative error
  below `5e-14`.
- `earlier-warp-smoke.json` records an earlier, narrowed O0 native experiment:
  all fixtures were converted to SSS, with only `000` and polynomial `200` in
  the dispatcher. Native checks, memcheck, synccheck and racecheck passed. It
  used the now-rejected warp owner, so these results do **not** qualify the
  archived scalar-owner prototype or a complete force library.

The intended incremental sequence was `000`, `100/001`, `110/101`, `200/002`.
The last six classes are absent. A future restart would first complete the
SSS build, native/force/prepared-replay/sanitizer gates and five-sample endpoint
comparison before extending the class set. Existing FP64, screening, rank,
convergence and strict numerical gates must be preserved.

All real GPU validation must use finite Slurm `main` allocations with
`--gres=gpu:5090:1` and preserve the assigned device visibility. No new benchmark
or GPU validation was run to prepare this consolidation.
