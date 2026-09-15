# Integral subsystem instructions

These rules apply under `src/integrals/`.

## Scientific invariants

- Preserve basis normalization, shell/component ordering, symmetry conventions,
  derivative sign/coordinate conventions, and accumulation semantics across
  generated and compatibility paths.
- Do not duplicate recurrence mathematics merely to introduce a new schedule or
  consumer. Generated and native compatibility paths must share an auditable
  scientific contract.
- Precision changes are scientific changes. State the accumulation/storage policy,
  validate against an independent oracle, and test the sizes and angular-momentum
  regimes where cancellation or overflow risk can change.
- For derivatives and response, contract generated derivatives with final
  weights as early as practical when this avoids materializing large derivative
  tensors without changing the scientific result.

## Performance and scheduling

- Measure complete consumers/endpoints and semantic work counts, not only kernel
  time. Audit nested tiling for repeated integral/source work and data movement.
- Prefer source-driven reuse when an expensive integral/intermediate is invariant
  to an outer consumer tile. Preserve an explicit bounded fallback when reuse
  depends on resident capacity or lifetime/stream ownership.
- Validate schedule changes at a larger size capable of exposing work-amplification
  cliffs, not only small AO counts.

Non-trivial recurrence, precision, derivative-contraction, scheduling, or fallback
choices should preserve their rationale in `.agents/notes/`. Cross-cutting
performance rules are in `docs/performance_engineering.md`.
