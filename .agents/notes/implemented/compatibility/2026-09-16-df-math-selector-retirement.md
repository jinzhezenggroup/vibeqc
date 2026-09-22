# Decision: retire the SSS-only DF mathematical selector

Status: implemented
Date: 2026-09-16

## Problem and decision

`VIBEQC_DF_SHELL_MATH_000` was useful when only SSS had a Rys derivative
candidate. The complete seven-class candidate matrix and generated manifest
now provide the same comparison mechanism for every supported class. Remove
the SSS override from native dispatch and current resource/checkpoint identity;
the manifest owns mathematical selection. The production manifest and its
qualified automatic domain remain unchanged.

## Compatibility invariants

Schema-1 checkpoints may contain the retired key with either an explicit value
or `null`. Accept it only as historical source metadata, retaining its exact
value on inspection and re-export. It is not an active environment control.
The old controls differ from current controls, so density import requires
`allow_warm=True` and the usual target verification. Do not delete the key to
claim exact numerical-policy compatibility. Unknown keys and missing original
required controls remain errors.

## Evidence and consequences

Checkpoint tests cover `null`, `auto`, `rys`, and `polynomial` source values,
explicit warm import, numerical replay and unchanged re-export provenance.
Current checkpoints omit the key even when that environment variable is set.
Native shell-pair fixtures assert polynomial work outside the automatic
manifest domain. The standalone candidate matrix retains direct qualification
of all polynomial/Rys choices without a class-specific runtime override.

Historical evidence and reproduction commands remain tied to their measured
source revisions; removal does not relabel earlier benchmark artifacts.

References: #394, #404; [historical SSS qualification](../performance/2026-09-16-000-rys-qualification.md),
[current tuning](../../../../docs/developer/df_tuning.md).
