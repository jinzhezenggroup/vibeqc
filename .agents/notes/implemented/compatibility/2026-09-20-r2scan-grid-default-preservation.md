# Decision: preserve the qualified r2SCAN grid default

Status: implemented
Date: 2026-09-20

## Problem

The new production grid policy is qualified for LDA/PBE only, but the shared
KS option resolver called it for every native semilocal method. Consequently,
previously supported default r2SCAN RKS and UKS requests raised before native
execution. Moving precision validation earlier did not repair this default.

## Decision

Keep the existing version-one GridSpec for default r2SCAN requests. Explicit
GridSpec values remain authoritative. A nondefault r2SCAN grid accuracy profile
is rejected with an explicit-grid diagnostic, not silently ignored. LDA/PBE
continue to resolve the new production policy unchanged.

## Rejected alternatives

Do not assign a PBE grid to meta-GGA without numerical qualification. Do not
make existing r2SCAN users spell out the historical default to preserve behavior.
Do not interpret a caller's tight-profile request as a standard legacy grid.

## Evidence and boundary

Both default spin-mode regressions fail before this change and pass afterward.
Ten new policy-contract cases cover defaults, explicit grids and refusal of an
unqualified accuracy profile. No quadrature, XC formula or tolerance changes.
