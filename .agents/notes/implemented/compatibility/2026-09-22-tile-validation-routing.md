# Decision: tile validation routing

Status: implemented
Date: 2026-09-22

## Decision

Make the diagnostic tile-validation request explicit when selecting bounded Direct-HF execution, preserving ordinary production admission as the default.

## Invariants and rejected alternatives

Diagnostic settings must not silently redefine the scientific method. Coordinate with the bounded high-l fallback change so diagnostics do not bypass required class coverage.

## Evidence and remaining qualification

Native precision-policy tests cover the routing contract. CUDA diagnostic and ordinary RHF/UHF endpoint comparisons remain merge gates.

## Revisit when

Revisit the candidate when the stated endpoint gates pass or the shared owner changes.
