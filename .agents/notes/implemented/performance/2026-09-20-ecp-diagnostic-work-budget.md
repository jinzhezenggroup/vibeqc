# Decision: bound CUDA ECP diagnostic quadrature work separately

Status: implemented
Date: 2026-09-20

## Problem

The complete CUDA stationary diagnostic bounds primitive integrals and Becke
partition work. Its ECP provider had only AO/atom/term and workspace caps, so
those existing work budgets could not reject a costly ECP quadrature traversal.

## Decision

Admit center / unordered-AO-pair / radial-angular sample counts across both
fixed ECP grids before compiling or invoking the ECP provider. Read grid sizes
from the canonical compiler policy. The default cap is 100,000,000 samples;
an explicit positive integer may override it. Include diagonal AO pairs and
only centers with positive core counts, matching the provider's traversal.
Report this count and cap independently from primitive and Becke counters.
The count is a bounded quadrature inventory, not FLOPs or measured throughput;
primitive and term multiplicities retain the existing separate domain caps.

## Invariants and evidence

No quadrature, equation, derivative, numerical tolerance or public capability
changes. Identical admitted inputs execute the same providers. GPU regression
checks rejection at one less than the declared count, admission at equality,
invalid limit types, explicit host/device ECP budget rejection and recovery.
All-electron gradients report zero ECP work and accept a one-sample ECP limit.
The existing four independent analytic/finite-difference ECP gates remain the
acceptance checks for admitted numerical execution.

Refs #171 and merged #586. A workspace-only limit is not a work limit. Revisit
this count when the provider's grid or pair schedule changes; new schedules must
also report complete endpoint work and resource evidence.
