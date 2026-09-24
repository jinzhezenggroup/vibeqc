# Decision: native cuda perturbative triples

Status: implemented
Date: 2026-09-22

## Decision

Generate the native CUDA standard-(T) evaluator from the retained triples equations and route the explicit CUDA method owner to it. Keep shape, denominator, workspace and backend failures explicit.

## Invariants and rejected alternatives

Do not claim CPU fallback as CUDA execution or expand this energy evaluator into analytic-force support. Coordinate with #1047 to preserve the current force and resource owner.

## Evidence and remaining qualification

Source generation and repository checks are local gates. Current CUDA compilation, CPU/CUDA oracle agreement, denominator failures and complete RCCSD(T) endpoint/resource qualification remain required before merge.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.
